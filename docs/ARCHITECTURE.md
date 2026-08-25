# Architecture

MegaGemm is an experimental full-stack inference runtime rather than a single
kernel library. The project is solo-designed and architected by Gabriel Yogi.

## Runtime flow

```text
Hugging Face checkpoint or MGX artifact
                  |
                  v
       model loader / architecture config
                  |
                  v
       MegaGemmLlama-compatible model
                  |
        +---------+----------+
        |                    |
        v                    v
  prefill kernels       decode kernels
        |                    |
        +---------+----------+
                  |
                  v
       paged/layer-aware KV cache
                  |
                  v
       scheduler and generation API
                  |
        +---------+----------+
        |                    |
        v                    v
 monitoring/XAI       serving/MegaMesh
```

## Repository map

| Path | Responsibility | Status |
|---|---|---|
| `megagemm/engine/` | Generation, scheduler, KV cache, offload, monitoring, deterministic mode, XAI, Prophet | Core with experimental extensions |
| `megagemm/models/` | Architecture configuration, checkpoint loading, model execution, MGX artifacts | Core |
| `megagemm/kernels/` | Triton, PyTorch, and CPU kernel implementations | Core/experimental by kernel |
| `megagemm/quantization/` | INT8, AWQ, native W4A16, and related formats | Core/experimental by format |
| `megagemm/mesh/` | MegaMesh routing, planning, transport, and sharded execution | Experimental |
| `megagemm/embeddings/` | Encoder/embedding runtime | Experimental |
| `src/` | C, C++, and CUDA native sources | Core/experimental by extension |
| `pytorch_binding/` | PyTorch bindings for native CUDA extensions | Core |
| `microgemm/` | Standalone native CPU inference runtime | Experimental, independently benchmarked |
| `tests/` | Unit, integration, harness, policy, and GPU tests | Active; see `tests/README.md` |
| `benchmarks/` | Performance matrices, microbenchmarks, sweeps, and comparison runners | Research infrastructure |
| `docs/` | Architecture notes and dated benchmark reports | Publication surface |

## Major subsystems

### Inference engine

`InferenceEngine` owns model loading, tokenizer setup, sequence allocation,
prefill, decode, sampling, generation fast paths, and optional monitoring. The
engine supports sequential generation and scheduler-driven batching.

The public execution contract, memory policy, prefill/decode modes, fast-path
selection, model-specific limits, and separate encoder runtime are documented in
[`ENGINE.md`](ENGINE.md).

### Model execution

The main model implementation contains architecture-specific paths for dense,
MoE, full-attention, and hybrid linear-attention families. Specialized fast
paths coexist with fallbacks because the runtime targets multiple GPU
generations and model shapes.

### Qwen 3.5 hybrid full/linear attention

Qwen 3.5 is represented as a layer-local hybrid rather than a conventional
full-attention transformer. When a checkpoint does not provide an explicit
layout, the runtime synthesizes one full-attention layer followed by three
linear-attention layers. Full-attention blocks retain paged KV, QK normalization,
partial RoPE, and the output gate. Linear blocks instantiate `GatedDeltaNet` and
store depthwise-convolution plus recurrent delta-rule state in the block manager.
Layer-aware cache allocation therefore reserves ordinary KV pages only for the
full-attention subset.

`GatedDeltaNet` combines fused Q/K/V and beta/A/z projections, a depthwise causal
convolution, learned decay and update gates, a recurrent key/value state,
RMSNorm+SiLU output gating, and a final projection. Decode has dedicated Triton
routes for causal-convolution update, recurrent gated-delta update, fused A/B
gate formation, RMSNorm+input projection, and RMSNormGated+output projection.
Reusable decode buffers and the flat hybrid loop remove intermediate allocation
and Python/module dispatch from the hot path.

Prefill is regime-dependent. Short prompts use recurrent Triton prefill; longer
prompts use the chunked delta rule, a local triangular solve, and an interchunk
state scan. The scan treats each chunk as an affine transition
`S' = A @ S + B`, so chunk prefixes can be composed associatively.

Two parallel prefix families are implemented:

- **Hillis–Steele:** an inclusive logarithmic-stage affine scan, with a Triton
  route and a tensor fallback;
- **Blelloch:** power-of-two padding, upsweep, downsweep, and final inclusive
  composition, implemented both as a Torch reference and Triton affine kernels.

The choice between them is based on measured wall time rather than asymptotic
work alone. For `C` chunk transitions, Hillis–Steele performs an inclusive scan
in `ceil(log2(C))` regular stages. It does `O(C log C)` affine compositions, but
every stage has the same direct prefix layout and maps cleanly to dense batched
GPU work. Blelloch is work-efficient at `O(C)`: it pads to a power of two, builds
an upsweep reduction tree, performs a downsweep to obtain exclusive prefixes,
and converts them back to the required inclusive result. Its parallel depth is
still logarithmic, but it has two tree phases plus padding, intermediate state,
and additional synchronization/launch boundaries.

The current Triton implementations make that wall-time difference concrete.
Hillis–Steele launches one composition kernel for `A` and one for `B` at each
offset stage. Blelloch launches the corresponding `A`/`B` pair throughout both
the upsweep and downsweep, then launches another pair to combine the exclusive
tree result with each original transition. The work-efficient tree can therefore
execute fewer affine compositions while still paying more kernel-launch and
phase-boundary overhead at the chunk counts used by the model.

Those constants matter because each scan element is not a scalar addition. A
chunk represents the affine map `(A, B)` and ordered composition is
`(A2 @ A1, A2 @ B1 + B2)`. The scan therefore moves and multiplies dense state
blocks. On the measured T4 long-prefill shapes, Blelloch's lower theoretical
work did not become lower end-to-end latency: power-of-two padding, tree
materialization, extra phases, and less favorable dense-affine GPU occupancy
outweighed the saved compositions. Hillis–Steele performed more arithmetic but
finished sooner because its stages were simpler and more regular.

Blelloch is consequently retained as implemented and correctness-tested research
work, while the current selector maps Blelloch aliases to Hillis–Steele. This is
an empirical policy decision, not a claim that Hillis–Steele is universally the
better scan. The default policy also keeps the parallel scan opt-in and rejects
wide-head cases unless explicitly forced.

Implementing this path required more than selecting a named prefix algorithm:
the runtime derives associative affine chunk transitions from the delta rule,
keeps Torch references for both scan families, implements Hillis–Steele and
Blelloch Triton stages, manages padded/intermediate workspaces, validates ordered
composition against recurrent execution, and feeds measured regressions back
into the hardware/shape policy.

Runtime policy is hardware- and shape-aware. GPU capability selects the short
prefill threshold, scan window, and warp count; the window chooser avoids small
tail launches, and stable chunk buckets reduce Triton recompilation. Separate
on-device microprofiles compare fused and baseline input/output projection paths,
cache the result by tensor signature, and require the configured minimum gain
before enabling a fusion.

The dedicated regression surface contains 56 tests. The linear-attention and
gated-normalization kernel files contain 20 Triton JIT routines. Coverage spans
partial RoPE, hybrid scheduling, scan policy, Hillis–Steele and Blelloch affine
correctness, recurrent/chunked parity, convolution state, GQA, fused gates,
continuous batching, packed prefill, and layer-aware KV behavior.

### Gemma 4 mixed-layer text backbone

Gemma 4 has a dedicated architecture contract in the runtime; it is not routed
through the ordinary dense-transformer path with only renamed weights. The outer
multimodal checkpoint layout is recognized by the loader, while current
execution covers its text backbone rather than the vision/audio towers.

The configuration parser materializes the properties that vary by layer:

- attention type, with a default repeating schedule of five sliding-window
  layers and one full-attention layer;
- head dimension, KV-head count, rotary dimension, and RoPE theta;
- sliding-window versus global attention state;
- K=V behavior for full attention;
- cross-layer KV sharing and the source-layer map used by the cache;
- optional double-width MLPs on shared-KV layers;
- optional per-layer embedding dimensions and MoE topology.

This changes allocation as well as math. KV-shared layers reference the most
recent earlier unshared layer with a compatible attention type, so only the
source layers receive independent KV-cache storage. The attention module keeps
separate layer-local RoPE caches and uses Gemma 4-specific prefill/decode paths
because a single homogeneous QKV layout cannot represent these semantics.

#### Per-layer embeddings (PLE)

The Gemma 4 per-layer embedding path combines two streams before decoder
execution:

1. `embed_tokens_per_layer(input_ids)` is scaled by
   `sqrt(hidden_size_per_layer_input)` and reshaped into one slice per layer.
2. The base token embedding is projected by `per_layer_model_projection`, scaled
   by `hidden_size ** -0.5`, reshaped by layer, and RMS-normalized.
3. Both streams are added and scaled by `1/sqrt(2)`.

At each decoder layer, the corresponding slice passes through a hidden-to-PLE
gate, GELU-tanh activation, elementwise conditioning, PLE-to-hidden projection,
post-PLE RMSNorm, and residual addition. A learned layer scalar closes the
layer. Dedicated prefill and decode buffers preserve this path in both eager and
flat execution.

#### Gemma 4 MoE

When `enable_moe_block` is active, `Gemma4MoeMLP` follows the architecture's two
parallel feed-forward branches: a shared dense MLP and a routed expert MLP. The
router applies no-weight RMSNorm, a router scale, expert logits, top-k selection,
optional top-k renormalization, and per-expert scales. Pre/post feed-forward
norms preserve the branch order before the outputs are combined with the
residual.

The GPU implementation includes dedicated attention preparation, long
sliding/full prefill, a Gemma 4 top-k router, grouped and segmented expert
prefill, deterministic route packing/reduction, parallel shared/expert streams,
fused norm/residual bridges, and CUDA-graph workspaces. Selected A100 paths are
shape-gated for the implemented A4B topology (hidden size 2816, 30 layers, 128
experts, top-8 routing, 704 expert intermediate size, and 2112 shared
intermediate size); other shapes retain general/fallback paths.

FP16/BF16 and streaming INT8 W8A16 loading cover Gemma 4. AWQ loading for this
architecture is rejected explicitly. The test tree contains 16 Gemma 4-specific
modules with 236 test functions spanning configuration, PLE, attention/KV
semantics, MoE routing and execution, long prefill, INT8, graph policies, and
flat/eager parity.

### Encoder embedding runtime

`EmbeddingEngine` is separate from the autoregressive generation engine. It
loads plain Hugging Face encoder checkpoints and common Sentence Transformers
module layouts, then executes the pipeline `Transformer -> Pooling -> Dense* ->
Normalize` without requiring the `sentence-transformers` package itself.

Compatible BERT-family absolute-position encoders can use the native backend.
That path fuses Q/K/V projection and supports padding-free packed attention, with
packed sequence metadata prepared once and reused across layers. Unsupported
encoder configurations retain the Hugging Face model backend while using the
same MegaGemm batching, pooling, prompt, projection, and normalization surface.

The batching policy can sort by length and bound padded work with
`max_batch_tokens`, rather than limiting only the number of input strings.
Supported pooling modes are CLS, mean, max, mean-sqrt-length, weighted mean, and
last token. Query/document prompts, optional Dense heads, L2 normalization,
pinned host transfers, CPU fallback, and CLI benchmark entry points are part of
the same runtime. Two dedicated regression modules currently contain 15 tests
covering layout parsing, prompts, token-budget batching, native/Hugging Face
parity, padding-free correctness, state-dict normalization, and metadata reuse.

### KV cache and scheduling

The block manager provides paged KV allocation. Layer-aware allocation avoids
reserving full-attention KV for layers that use recurrent/linear-attention
state. The tiered manager adds CPU offload. The scheduler manages request state,
prefill admission, decode iteration, and cleanup.

### Kernel layers

The repository contains custom Triton routines, native CUDA kernels, C++ hot
path helpers, and PyTorch fallbacks. As of the August 2026 source inventory,
there are 157 `@triton.jit` routines and 14 native CUDA `__global__` bodies.
These are source-definition counts, not a count of runtime launches: wrappers,
autotuned specializations, graph capture, and fallback dispatch make “launch
point” ambiguous unless a specific workload and counting method are named.

### MGX and Prophet

MGX packages transformed runtime weights and optional session state into a
project-specific artifact. Prophet stores and retrieves persistent semantic
state/KV snapshots. These original experimental subsystems are separate from
the stable checkpoint-loading path.

### MegaMesh

MegaMesh provides replica routing and layer-sharded inference over a custom
tensor transport. It targets heterogeneous and potentially WAN-connected
workers, so its design constraints differ from homogeneous NVLink tensor
parallelism.

### MicroGemm

MicroGemm is the standalone native CPU runtime under `microgemm/`. CPU claims
against llama.cpp belong specifically to this backend unless a report states
otherwise; the generic PyTorch CPU path in `InferenceEngine` has different
performance characteristics.

## Architectural objective: capability-directed runtime

MegaGemm owns its model execution, scheduling, cache management, sampling, and
kernel dispatch. It is not a compatibility layer over another inference engine.
The corresponding architectural objective is to make the runtime
**capability-directed**: the selected model, device, dtype, quantization mode, and
workload determine which pieces are installed, imported, materialized, compiled,
and executed.

Modularity has several distinct surfaces. A small dependency list alone does not
prove that a runtime is fully modular:

1. **Installation surface:** which external distributions must exist before a
   feature can run.
2. **Import surface:** which Python modules and kernel definitions enter the
   process for a selected engine path.
3. **Materialization surface:** which weights, KV/recurrent state, buffers, and
   services consume host or device memory.
4. **Compilation surface:** which native extensions, Triton specializations, and
   CUDA graphs are built or captured.
5. **Artifact surface:** which checkpoint shards, tokenizer assets, and metadata
   must be downloaded or shipped for deployment.

### Current state and target boundary

| Surface | Current implementation | Architectural target |
|---|---|---|
| Base installation | One mandatory direct dependency: `torch`; integrations live behind extras | Preserve the small base and keep unrelated serving, distributed, monitoring, and benchmark stacks optional |
| Top-level import | `megagemm`, `megagemm.engine`, and `megagemm.embeddings` expose public objects through lazy `__getattr__` dispatch | Preserve lazy public imports |
| Engine import | The unified model implementation feature-probes a broad set of architecture and kernel modules | Resolve an architecture manifest first, then import only its model path and required kernel families |
| Kernel execution | Triton kernels compile on first selected use; importing their definitions does not compile every specialization | Produce an explicit kernel plan and compile/cache only plan entries |
| Weight materialization | Config-specific models are constructed on `meta`; loading streams selected tensors to their destination | Preserve streaming materialization and make selection visible in the runtime plan |
| Multimodal checkpoints | Text-only Gemma 4 loading filters non-text towers before materialization | Generalize component filtering to every multimodal architecture |
| Runtime state | Layer-aware KV allocation avoids allocating paged KV for layers represented by recurrent or shared state | Derive every cache/state allocation from the architecture plan |
| Checkpoint download | Hugging Face loading can still download all matching Safetensors files before unused components are filtered | Read the index first and fetch only shards containing selected components |
| Native build | Available native helpers/extensions can be attempted as a group during package build | Select prebuilt or local extensions from device and capability requirements |
| MGX deployment | MGX stores transformed runtime weights and optional session state; tokenizer sidecars may still come from the original checkpoint | Embed a runtime manifest and the complete tokenizer asset set for a self-contained model-specific deployment |

The current split is therefore precise: **dependency installation, public imports,
and runtime weight/state materialization already provide meaningful modularity**;
the engine's internal import graph, checkpoint download selection, native build,
and some artifact assets remain broader than the intended end state.

### Runtime-plan model

The intended selection flow is:

```text
checkpoint config + device capabilities + execution options
                         │
                         ▼
                    RuntimePlan
          ┌──────────────┼────────────────┐
          ▼              ▼                ▼
 architecture path   state/cache plan   kernel plan
          │              │                │
          └──────────────┼────────────────┘
                         ▼
        validate only required optional integrations
                         ▼
       fetch/materialize selected components and compile on demand
```

For example, a text-only Gemma 4 E2B plan should select the Gemma 4 text
backbone, its heterogeneous attention/PLE state rules, the scheduler, sampler,
layer-aware cache, and the kernel families chosen for the detected GPU. It should
not require or initialize Qwen 3.5 linear-attention scans, Qwen MoE paths,
embedding-engine code, MegaMesh, monitoring, XAI, or an unused multimodal tower.
An E4B plan may select a different execution and memory policy even within the
same architecture family; model scale is an input to planning, not merely a
larger allocation passed through an identical fast path.

### Packaging rule

Capability-directed loading does not require fragmenting the project into many
independently versioned Python distributions. The preferred boundary is one
source package with a small mandatory core, feature extras for external
integrations, lazy internal registries, and model-specific runtime plans. This
keeps installation understandable without recreating cross-package dependency
coordination inside MegaGemm.

Comparison backends are deliberately outside this boundary. vLLM and llama.cpp
belong to isolated benchmark environments and are not dependencies of the
MegaGemm engine. Their installation state must not alter which MegaGemm code path
or package profile is required for inference.

## Maturity labels

Issues and documentation use these labels:

- **Core:** part of the primary model loading/generation path.
- **Supported:** has a documented runnable path and regression coverage.
- **Experimental:** working research path whose API or tuning may change.
- **Harness:** validation/benchmark tooling, not runtime API.
- **Archived result:** dated evidence retained for historical comparison.


## Current structural boundaries

- Architecture-specific execution paths currently coexist in a large unified
  model implementation.
- CPU tests, hardware-gated GPU tests, standalone harnesses, and source-policy
  checks use several collection mechanisms.
- Dated benchmark reports have uneven raw-artifact coverage; each report records
  its own evidence boundary.
- Stable loading/generation APIs and experimental environment-controlled tuning
  knobs currently share the same package surface.
