# Engine internals

MegaGemm contains two local inference surfaces with different execution
contracts:

- `InferenceEngine` runs autoregressive text generation with paged KV cache,
  architecture-specific prefill/decode, sampling, and continuous batching.
- `EmbeddingEngine` runs encoder checkpoints with throughput-aware batching,
  pooling, projection heads, and optional normalization.

MicroGemm is a separate native CPU runtime, and MegaMesh is the experimental
distributed layer. Neither is an alternate code path hidden inside
`InferenceEngine`.

## Generation runtime

### Construction and loading

`InferenceEngine` accepts a Hugging Face model ID, a local checkpoint snapshot,
or an MGX artifact. The initialization flow is:

```text
model reference
    |
    +-- Hugging Face/local snapshot -> config parser -> architecture validation
    |                                  -> meta-device model -> streamed weights
    |
    +-- MGX artifact -----------------> manifest/payload validation -> runtime weights
                                       |
                                       v
tokenizer/chat template -> layer-aware cache geometry -> BlockManager -> ready engine
```

The loader creates the model on the `meta` device before materializing weights.
This avoids constructing a complete temporary CPU model. Supported streaming
paths load and transform weights layer by layer; MGX can additionally reuse an
extracted payload cache. Tokenizer bootstrap data and chat templates are loaded
from the checkpoint or MGX manifest.

Architecture parsing is part of execution correctness. It materializes such
properties as QK normalization, partial RoPE, attention output gates, MoE
topology, Qwen 3.5 full/linear layer layout, and Gemma 4 per-layer attention/KV
geometry. These values determine cache allocation and kernel dispatch, not only
module names.

### Public construction parameters

| Parameter | Runtime meaning |
|---|---|
| `model_name` | Hugging Face ID, local snapshot, or `.mgx` artifact |
| `dtype` | Base FP16/BF16/FP32 tensor dtype; optimized GPU paths primarily target FP16/BF16 |
| `device` | Execution device, normally `cuda`; CPU execution here is the PyTorch path, not MicroGemm |
| `quantize` | `int8` selects streaming INT8 W8A16; AWQ comes from compatible pre-quantized checkpoints |
| `n_gpu_layers` | `-1` keeps every layer on GPU; non-negative values enable model-layer offload |
| `block_size` | Tokens represented by one paged-KV block |
| `num_blocks` | Explicit GPU KV capacity; `0` enables automatic sizing |
| `max_batch_size` | Scheduler request capacity and an input to automatic KV sizing |
| `max_seq_len` | Prompt-plus-generation capacity target used by RoPE/cache setup |
| `kv_alloc` | `auto` sizes to the declared workload; `greedy` uses the configured free-VRAM fraction |
| `kv_offload` | Enables `TieredBlockManager` for compatible homogeneous KV layouts |
| `deterministic` | Enables seeded deterministic PyTorch/CUDA execution on the same hardware/software stack |
| `monitor`, `dashboard` | Opt-in request telemetry and local HTML monitoring |

The legacy `fp8` argument currently aliases the INT8 W8A16 loader. It does not
identify a separate FP8 weight format.

### Cache geometry and memory policy

`BlockManager` stores K and V in fixed-size pages and maps logical sequences to
physical block IDs. When `num_blocks=0`, the engine measures free device memory
after model loading and releases temporary allocator state before sizing the
cache. `kv_alloc=auto` targets
`ceil(max_seq_len / block_size) * max_batch_size` blocks, capped by the available
VRAM policy. `kv_alloc=greedy` allocates toward that VRAM cap without the workload
target.

Allocation is architecture-aware:

- ordinary dense full-attention models allocate KV for every decoder layer;
- Qwen 3.5 allocates paged KV only for full-attention layers, while linear
  layers keep convolution and recurrent delta-rule state;
- Gemma 4 allocates independent storage only for KV source layers and accounts
  for per-layer KV-head/head-dimension differences;
- KV-shared Gemma 4 layers reference a compatible earlier source layer instead
  of duplicating pages.

`TieredBlockManager` can evict cold blocks to pinned CPU memory and restore them
before use. Gemma 4 heterogeneous KV is rejected by this path because one
homogeneous offload layout cannot encode its per-layer geometry and sharing.

### Prefill

The sequential API prefills one request. The scheduler-driven batch API groups
waiting requests and selects a supported packed or padded path according to the
architecture, request lengths, memory headroom, and graph eligibility.

For ordinary packed full-attention prefill, backend dispatch is:

1. FlashAttention varlen when the optional `flash-attn` integration is present;
2. a MegaGemm Triton packed kernel when enabled and eligible;
3. one uniform-batch PyTorch SDPA call for equal-length batches with sufficient
   memory headroom;
4. per-sequence PyTorch SDPA as the portable fallback.

Qwen 3.5 linear layers use recurrent prefill for short sequences and a chunked
delta-rule path for longer sequences. Gemma 4 uses dedicated sliding/full
attention preparation and long-prefill kernels because its layer-local geometry,
PLE inputs, K=V mode, and KV sharing cannot be represented by the generic
homogeneous path.

### Decode

Decode updates request state one token step at a time while the scheduler batches
active sequences. The selected route can contain:

- paged-attention decode for ordinary full-attention layers;
- Qwen 3.5 fused convolution/recurrent updates for linear layers;
- Gemma 4 layer-local attention, PLE, shared-KV, dense or MoE execution;
- reusable workspaces and flat loops that avoid Python module dispatch in the
  hot path;
- fused greedy-token selection on eligible shapes;
- eager single-step, multi-step, or CUDA Graph replay according to explicit
  eligibility and runtime policy.

“Fast path” is therefore not synonymous with CUDA Graphs. A run can use fused
projections, flat single-step decode, persistent buffers, specialized attention,
and scheduler reuse while intentionally remaining eager. Graph capture is one
optimization family among several and is disabled for shapes whose replay cost,
memory, mutability, or correctness contract is unfavorable.

### Fast-path selection and observability

Dispatch decisions are constrained by architecture, shape, dtype, GPU capability,
batch size, context regime, quantization layout, and correctness guards. Some
projection paths are microprofiled against their baseline on the active GPU and
cached only when they clear a minimum measured gain.

Runtime counters are route-specific. A zero counter means only that the named
route did not run; it does not prove that the model fell back globally. For
example, Gemma 4 KV-sharing reduces the number of independent QKV source layers,
and a generic/GQA2 paged-attention counter does not describe a GQA4 segmented
kernel. A useful audit records all of the following together:

- execution mode (`eager`, flat step, multi-step, or graph replay);
- architecture and exact tensor geometry;
- fused-kernel hit counters and relevant fallback reasons;
- scheduler reuse and decode-step counts;
- graph captures, replays, and failures;
- prefill/decode latency and peak memory.

Experimental environment variables expose profiling and shape-specific tuning.
They are policy controls rather than stable API guarantees; publication runners
record the selected variables in their manifests.

### Generation APIs

`generate()` processes one prompt at a time and exposes sampling, XAI, and Logit
Lens options. `generate_batch()` submits multiple text or pretokenized requests
to the continuous-batching scheduler. `generate_batch_stream()` yields completed
batch items incrementally.

```python
import torch
from megagemm.engine import InferenceEngine

engine = InferenceEngine(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.float16,
    max_batch_size=8,
    max_seq_len=4096,
    kv_alloc="auto",
)

single = engine.generate("Explain paged KV cache.", max_new_tokens=64)
batch = engine.generate_batch(
    ["Explain GQA.", "Explain chunked prefill."],
    max_new_tokens=64,
    temperature=0.0,
)
```

### Declarative inference runner

`megagemm run` maps a versioned JSON or YAML document to the same public
`InferenceEngine`, `generate()`, and `generate_batch()` APIs. It is an interface
over the owned MegaGemm runtime, not a subprocess wrapper or a translation to a
comparison backend.

```yaml
version: 1
task: generate
model: Qwen/Qwen2.5-3B-Instruct

engine:
  device: cuda
  dtype: bf16
  max_seq_len: 4096
  max_batch_size: 8
  kv_alloc: auto
  deterministic: true
  seed: 42

generation:
  max_new_tokens: 128
  temperature: 0.0
  top_k: 50
  top_p: 0.9

prompts:
  - Explain paged attention.
  - Explain chunked prefill.

output:
  format: jsonl
  path: outputs/results.jsonl
  include_prompt: true
```

```bash
megagemm run inference.yaml
megagemm run inference.yaml --dry-run
megagemm run inference.yaml --format json --output -
```

Version 1 accepts exactly one input source:

| Field | Execution path |
|---|---|
| `prompt` | Single-request `generate()` |
| `prompts` | Inline continuous batch through `generate_batch()` |
| `prompts_file` | Non-empty lines loaded as a continuous batch; paths are relative to the config file |

The `engine` mapping covers the stable constructor surface: device/dtype, KV
allocation, sequence and batch limits, CPU/layer offload, quantization,
monitor/dashboard, deterministic execution, and MGX payload controls. The
`generation` mapping covers token count, temperature, top-k/top-p, stop tokens,
verbosity, and single-request XAI/Logit Lens. Batch generation supports
`ignore_eos`; its current API does not support repetition penalty or XAI, and the
validator rejects configurations that would imply otherwise.

Unknown keys, wrong scalar types, unsupported enum values, multiple input
sources, and invalid cross-field combinations fail before the model is loaded.
YAML is parsed only through `yaml.safe_load`. JSON uses the standard library;
YAML requires the optional `.[config]` extra. `--dry-run` validates and prints
normalized settings and prompt count, but does not claim to resolve the future
model-specific kernel/memory `RuntimePlan`.

Monitoring records request latency, token throughput, memory, and percentile
summaries. XAI is opt-in and adds token probability, entropy/confidence, and
optional layer-probe collection; its uncertainty field is not a factuality
detector.

### Architecture-specific boundaries

| Path | Current boundary |
|---|---|
| Qwen 3.5 linear attention | FP16/BF16 text-backbone execution; linear-layer INT8/AWQ loading is rejected |
| Gemma 4 | Text backbone only; vision/audio towers are not executed |
| Gemma 4 quantization | FP16/BF16 and streaming INT8 W8A16; AWQ is rejected |
| Gemma 4 KV offload | Rejected for heterogeneous/shared KV geometry |
| MegaMesh layer shards | Gemma 4 text shards are rejected; shard mode has its own documented model limits |
| Generic CPU execution | PyTorch fallback path; published llama.cpp comparisons belong to MicroGemm |

## Embedding runtime

`EmbeddingEngine` targets encoder-style vector generation rather than token
generation. It supports plain Hugging Face encoder checkpoints and common
Sentence Transformers directory layouts.

```text
texts
  -> optional query/document prompt
  -> length/token-budget batch planner
  -> tokenizer
  -> native BERT or Hugging Face encoder
  -> CLS/mean/max/mean-sqrt/weighted/last-token pooling
  -> optional Dense projection modules
  -> optional L2 normalization
  -> vectors restored to input order
```

The native backend covers compatible BERT-family absolute-position encoders.
It supports padding-free execution by packing valid tokens, preparing sequence
metadata once, and reusing that metadata across encoder layers. The automatic
backend falls back to the Hugging Face model when the native configuration is
not supported.

`max_batch_tokens` limits padded token work, while `batch_size` limits the number
of texts. Length sorting reduces padding and results are restored to their
original order. Sentence Transformers prompts, Dense modules, Normalize modules,
and pooling configuration are read directly from the model layout.

```python
from megagemm.embeddings import EmbeddingEngine

encoder = EmbeddingEngine(
    "sentence-transformers/all-MiniLM-L6-v2",
    backend="auto",
    max_batch_tokens=8192,
)

queries = encoder.encode_query(["paged attention", "linear attention"])
documents = encoder.encode_document(["A technical document."])
stats = encoder.benchmark(["example"] * 128, batch_size=32)
```

The equivalent CLI commands are:

```bash
megagemm embed --model sentence-transformers/all-MiniLM-L6-v2 "first text" "second text"
megagemm embed-bench --model sentence-transformers/all-MiniLM-L6-v2 --copies 128 --batch-size 32
```

The `embeddings` optional dependency profile supplies the Hugging Face,
Safetensors, Transformers, and SentencePiece integrations. The base package
continues to require only PyTorch.
