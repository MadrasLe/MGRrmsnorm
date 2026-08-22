# 🔥 MegaGemm — High Performance LLM Inference Engine

<div align="center">

[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-6C4DC4?style=for-the-badge&logo=openai&logoColor=white)](https://triton-lang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

Full-stack LLM inference engine built from scratch with custom CUDA, Triton, and C++ kernels.
Loads HuggingFace models, quantizes on-the-fly, and generates text with paged KV cache — no `vllm`,
no `llama.cpp`, just PyTorch + custom kernels.

In the current same-session NVIDIA L4 matrix, MegaGemm FP16 reaches **93.1% of vLLM 0.27.1's
geometric-mean throughput**, stays within 10% in five of six rows, and is within 3% for the measured
batch-1 short/medium prompts. vLLM still wins all uncached FP16 rows and leads by 15.7% at the
measured batch-8 / 2048-token point. Earlier T4, Hugging Face, and hybrid-model measurements remain
historical and are not promoted until their raw evidence is restored or the workloads are rerun.
In a separate warm exact-prefix reuse matrix, **MGX+Prophet reached 102.8% of vLLM prefix caching's
geometric-mean throughput and won three of four measured rows**.
See [Performance](#-performance) for the complete breakdown and limitations.

**Project guides:** [Architecture](docs/ARCHITECTURE.md) ·
[Dependencies](docs/DEPENDENCIES.md) · [Benchmark evidence](docs/BENCHMARKS.md) ·
[Test suite](tests/README.md) · [Contributing](CONTRIBUTING.md)

---

## 📊 Performance

> **Benchmark status (August 2026):** Qwen 2.5 3B FP16 now has a current compact same-session L4
> matrix against vLLM 0.27.1. The T4, Hugging Face, Qwen 3.5, quantized, and CPU measurements from
> May-June remain historical until they receive equivalent reruns.
>
> **How to read this section:** every table cites the GPU, the date, the dtype, and the run
> conditions. Throughput is median generated tokens/sec, wall-clock, `--ignore-eos` (fixed decode
> length). The current raw L4 archive lives in [`docs/`](docs/); retained CPU summaries live in
> [`bench_results/notes/`](bench_results/notes/). These are project-run measurements on accessible
> hardware, not independent datacenter validation or a universal leaderboard.

### Historical GPU baselines

Earlier development runs compared MegaGemm with Hugging Face Transformers and older vLLM releases
on T4/L4 allocations. Their raw reports are not part of the current curated publication bundle, so
they are not promoted as headline metrics here. The benchmark runners remain available for clean
reruns.

### vs vLLM — honest, regime-dependent

vLLM is a strong, mature baseline. The truth is **it depends on the regime**:

**Current: Qwen 2.5 3B Instruct · 1× NVIDIA L4 · FP16 · 2026-08-22**
(vLLM 0.27.1+cu129, prefix caching OFF, 128 generated tokens;
[report](docs/qwen25_l4_results_20260822.md) ·
[raw archive](docs/publication_20260822_145527.zip))

| Batch | Prompt | MegaGemm FP16 | vLLM FP16 | MegaGemm / vLLM |
|---:|---:|---:|---:|---:|
| 1 | 128 | 37.82 | 38.81 | **97.4%** |
| 1 | 512 | 37.39 | 38.50 | **97.1%** |
| 1 | 2048 | 33.68 | 36.25 | 92.9% |
| 8 | 128 | 281.20 | 294.65 | **95.4%** |
| 8 | 512 | 244.32 | 266.24 | 91.8% |
| 8 | 2048 | 155.01 | 183.78 | 84.3% |

> **Current takeaway:** vLLM won all six uncached FP16 rows. MegaGemm nevertheless delivered 93.1%
> of its geometric-mean throughput, remained within 10% in five rows, and came within 3% on the two
> measured batch-1 prompts up to 512 tokens. This was a compact one-warmup/three-repeat matrix; see
> the report for environment metadata and limitations.

### MGX+Prophet cache reuse versus vLLM prefix cache

**Current: Qwen 2.5 3B Instruct · 1× NVIDIA L4 · FP16 · 2026-08-22**
(warm exact-prefix reuse on both sides, 32 generated tokens, five repetitions;
[report](docs/qwen25_l4_prophet_results_20260822.md))

| Batch | Prompt | MGX+Prophet | vLLM prefix cache | Prophet / vLLM |
|---:|---:|---:|---:|---:|
| 1 | 512 | 38.44 | 38.60 | 99.6% |
| 8 | 512 | 296.67 | 286.83 | **103.4%** |
| 1 | 2048 | 38.56 | 37.96 | **101.6%** |
| 8 | 2048 | 268.82 | 251.76 | **106.8%** |

> **Cache-reuse takeaway:** Prophet won three of four rows and reached **102.8%** of vLLM's
> geometric-mean cached throughput, a 2.8% lead in this matrix. This measures repeated exact-prefix
> hits, not semantically similar prompts. Prophet validation was disabled, and the raw JSON/CSV
> bundle is not present in this repository. The linked report records the exact evidence boundary.

### Where vLLM leads

- **High-batch, long-context, full-attention throughput** remains the clearest gap: in the current
  L4 Qwen 2.5 FP16 row at batch 8 / prompt 2048, vLLM leads by 15.7%. Older T4 configurations
  recorded substantially larger gaps.
- **Breadth and battle-testing**: far more models, quantization formats, and production hardening.
- MegaGemm has **no FlashAttention path on its packed attention**; that path falls back to
  per-sequence SDPA on the PyTorch backend.

### CPU: MicroGemm versus llama.cpp

The CPU comparisons belong to the standalone native [`microgemm/`](microgemm/) backend, not the
generic PyTorch CPU path in `InferenceEngine`.

**Current: Qwen 2.5 0.5B · Intel Xeon 2.20 GHz · 8 logical CPUs · 2026-08-22**
(MicroGemm INT8 versus llama.cpp Q8_0, same session, 64 requested prompt tokens, 128 generated
tokens, two warmups and five repetitions; [report](docs/qwen25_cpu_microgemm_vs_llamacpp_20260822.md))

| Batch | Micro engine | llama.cpp engine | Engine ratio | Decode ratio | Prefill ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.11 | 43.49 | 1.04x | 1.03x | 1.25x |
| 2 | 56.20 | 57.94 | 0.97x | 0.96x | 1.29x |
| 4 | 87.82 | 51.93 | **1.69x** | **1.58x** | **1.42x** |
| 8 | 110.62 | 68.82 | **1.61x** | **1.75x** | **1.51x** |

> **CPU takeaway:** MicroGemm and llama.cpp are at engine/decode parity across batches 1-2, while
> MicroGemm's continuous batching scales to a 1.65x engine and 1.66x decode geometric-mean lead
> across batches 4-8. Across all four batches, MicroGemm reaches **1.286x geometric-mean engine
> throughput**. The formats are not quality-identical, the MicroGemm canary was slow, and the raw
> JSON/CSV bundle is not present in this repository. The linked report records these limitations.

---

## ✨ Features

### Inference Engine
- 🧠 **Multi-Model** — LLaMA 3.x, Mistral 7B, Qwen 2.5, Qwen 3, **Qwen 3.5 (hybrid linear-attention)**, Gemma 2, and **Gemma 4 mixed-layer/MoE text backbones**
- 🔢 **INT8 W8A16** — per-channel, 2x compression, ~0.999 cosine similarity, **faster than FP16**
- 🧱 **AWQ 4-bit** — load pre-quantized AWQ models (4x compression)
- 📄 **Paged KV Cache** — BlockManager with configurable block size + **layer-aware allocation**
- 🔄 **KV Cache CPU Offload** — `TieredBlockManager` moves cold KV to pinned CPU RAM (10x capacity, 0% quality loss)
- ⚡ **Paged Attention** — Triton decode kernel with online softmax
- 🧬 **GQA / MQA** — grouped-query and multi-query attention
- 🧬 **Gemma 4 heterogeneous attention** — per-layer head/KV geometry and RoPE, sliding/full attention schedules, K=V and cross-layer KV sharing
- 🧬 **Gemma 4 PLE** — learned per-layer embeddings combined with projected token embeddings and injected through gated, normalized layer-local branches
- 🧠 **Gemma 4 MoE** — shared dense branch plus routed experts, with custom top-k routing and grouped/segmented prefill paths
- 🔄 **Continuous Batching** — iteration-level scheduler
- 💾 **Layer Offloading** — run models larger than VRAM (GPU+CPU)

### 🕸️ MegaMesh — Distributed Inference (experimental)
- **Replica mode** — full model per worker, weighted prompt router with failover
- **Layer-shard (pipeline) mode** — each worker owns a layer range; hidden states stream over **TTP**
- **TTP** — persistent TCP + length-prefixed binary tensor frames, optional pinned-memory pool and native `recv` C extension
- **AGron planner** — cost-model layer assignment from measured per-node speed and directed per-link latency/bandwidth — built for **heterogeneous Kaggle/Colab/RunPod/local nodes over WAN**, not a homogeneous NVLink cluster
- **Intra-layer primitives** — vocab-sharded `lm_head`, MLP intermediate sharding, replicated shard pipelines, decode microbatching + continuous batching over shards

### 📦 MGX — Compiled Artifacts
- Binary container (`MGX1`) of already-fused, already-quantized runtime weights — cold start skips HF snapshot parsing, QKV fusion, and on-load quantization
- Optional embedded **session state** (`MGXS`)

### 🔮 MGX Prophet — Persistent Semantic State Library
- On-disk KV/session snapshots retrievable across sessions/restarts
- Three-way lookup: **exact text hash → token-prefix hash → semantic embedding cosine similarity**
- Compatibility fingerprinting (model/tokenizer/quant/dtype), resident GPU cache, context fork + speculative restore
- Unlike vLLM prefix caching (exact-prefix, in-memory, per-process), Prophet warm-starts a **new, semantically similar** prompt from a prior session

### 🔍 XAI — Interpretability and uncertainty diagnostics
- Top-K token probabilities, confidence score (geometric mean of chosen-token probs)
- Entropy-based **uncertainty heuristic** (LOW/MEDIUM/HIGH), **Logit Lens** layer-by-layer
- Inference monitoring (P50/P95/P99, TPS), GPU VRAM, JSON/TXT/JSONL export
- **Live HTML dashboard** with auto-refresh, zero external dependencies

### 🖥️ CPU Backend (microgemm)
- Hand-written C++ inference kernels (`microgemm/`) for CPU-only decode
- Same-session comparison harnesses for llama.cpp with wall, decode, and prefill metrics
- Recorded parity-to-leading throughput in selected Qwen 2.5 and Mistral AVX2/FMA workloads

### Custom Kernels
- 🚀 **RMSNorm (CUDA)** — FP32/FP16/BF16, `float4`/`half2` vectorized loads, warp-shuffle reductions, FP32 accumulators
- ⚡ **SwiGLU (Triton)** — fused gate+up activation
- 🔄 **RoPE (CUDA)** — rotary position embeddings with half-rotate
- 📄 **PagedAttention (Triton)** — paged KV decode with online softmax
- 🧩 **Qwen 3.5 fused decode** — fused RMSNorm+in-proj for `GatedDeltaNet` linear attention, fused RoPE+attn, deepfusion MLP, fused `lm_head` argmax
- 🌊 **Qwen 3.5 scan kernels** — chunked delta-rule prefill, interchunk affine composition, Hillis–Steele and Blelloch implementations, and GPU-aware launch policy

---

## 🚀 Quick Start

```bash
git clone https://github.com/MadrasLe/MGRrmsnorm.git
cd MGRrmsnorm
python scripts/install_smart.py --editable
```

Direct extras-based installation also works:

```bash
pip install -e ".[inference]" --no-build-isolation
```

The base package has **one direct dependency (`torch`)**. A normal GPU inference install has six
direct dependencies total; CPU, monitoring, benchmarks, AWQ, and hardware-specific accelerators are
optional extras. See the exact counts and install profiles in
[`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

### Inference (3 lines)

```python
from megagemm.engine import InferenceEngine

engine = InferenceEngine("meta-llama/Llama-3.2-3B-Instruct")
print(engine.generate("Explain quantum computing in 3 sentences", max_new_tokens=200))
```

### INT8 (faster + 2x smaller)

```python
engine = InferenceEngine("Qwen/Qwen2.5-7B-Instruct", quantize='int8')
print(engine.generate("What is machine learning?", max_new_tokens=200))
```

### AWQ 4-bit (4x compression)

```python
engine = InferenceEngine("Qwen/Qwen2.5-7B-Instruct-AWQ")
```

### Layer offloading (models > VRAM)

```python
engine = InferenceEngine("Qwen/Qwen2.5-32B-Instruct", quantize='int8', n_gpu_layers=40)
```

### KV cache CPU offloading (more concurrent users)

```python
engine = InferenceEngine(
    "Qwen/Qwen2.5-7B-Instruct",
    kv_offload=True,        # GPU↔CPU KV tiering
    num_blocks=4096,        # hot blocks on GPU
    num_cpu_blocks=8192,    # warm blocks on pinned CPU RAM
    gpu_window=32,          # recent blocks kept on GPU per sequence
)
```

### 🔍 XAI — Interpretability diagnostics

```python
engine = InferenceEngine("Qwen/Qwen2.5-7B-Instruct")

text, report = engine.generate("What is the capital of France?", xai=True, xai_top_k=5)
print(report.confidence_score)     # e.g. 0.896
print(report.hallucination_risk)   # legacy field name: uncertainty heuristic
print(report.mean_entropy)         # Shannon entropy (lower = more confident)
print(report.summary())            # human-readable report with 🟢🟡🔴

# Logit Lens (layer-by-layer)
text, report = engine.generate("What is AI?", xai=True, logit_lens=4)  # every 4th layer

report.to_json("report.json")
report.to_txt("report.txt")
```

> XAI is **opt-in** via `xai=True`. Disabled by default → behavior and performance are unchanged.
> The `hallucination_risk` field is entropy/confidence-based uncertainty telemetry, not a factuality
> detector; a confident model can still be wrong.

### 📊 Monitoring + Live Dashboard

```python
engine = InferenceEngine("Qwen/Qwen2.5-7B-Instruct", monitor=True)
engine.generate("What is AI?", xai=True)
print(engine.monitor_summary())                 # terminal dashboard
engine.export_monitor_log("monitor.jsonl")

# Live web dashboard (auto-refresh, zero deps)
engine = InferenceEngine("model", dashboard=True, dashboard_port=8080)
# → http://localhost:8080
```

---

## 🕸️ MegaMesh — Distributed Inference

MegaMesh is the experimental distributed layer. It is intentionally isolated from the core decode
path: replica mode may call the engine inside a worker, but mesh modules do not modify the model,
scheduler, attention kernels, or decode loop.

**Replica mode** (full model per worker, prompts/text cross the wire):

```bash
# Start a worker per pod
python -m megagemm mesh-worker --model Qwen/Qwen3.5-4B --device cuda \
  --host 0.0.0.0 --port 8088 --name l4-0 --weight 2 --num-blocks 1024

# Generate through the weighted router
python -m megagemm mesh-generate \
  --workers http://10.0.0.10:8088@1#t4,http://10.0.0.11:8088@2#l4 \
  --max-tokens 128 "Explain Qwen 3.5 linear attention."
```

**Layer-shard mode** (a model that doesn't fit one GPU, split across workers over TTP):

```bash
# Plan layer ranges, then probe + plan from measured TTP links (AGron)
python -m megagemm mesh-plan --num-layers 48 \
  --workers ttp://host0:9090@1#s0,ttp://host1:9091@1#s1
python -m megagemm mesh-agron-probe --stages ttp://host0:9090#s0,ttp://host1:9091#s1 > prof.json
python -m megagemm mesh-agron-plan --num-layers 48 --profile-json prof.json --objective balanced
```

See [docs/megamesh.md](docs/megamesh.md) for replica, shard, vocab-sharded `lm_head`, MLP sharding,
microbatching, and continuous batching over shards.

**Current limitations:** greedy generation in shard mode, per-prompt prefill (decode microbatches),
FP16/BF16/FP32 only (no quantized shards yet), CPU-mediated TTP transport (no NCCL/GPU-direct yet).

---

## 🔮 MGX & Prophet

```python
from megagemm import export_to_mgx, load_from_mgx, MGXProphetLibrary

# Compile fused+quantized weights to an MGX artifact (faster cold start)
export_to_mgx(
    "Qwen/Qwen2.5-7B-Instruct",
    "qwen25-7b-int8.mgx",
    dtype="fp16",
    quantize="int8",
)
engine = load_from_mgx("qwen25-7b-int8.mgx")

# Persistent semantic state library
lib = MGXProphetLibrary("./prophet_store")
lib.capture(engine, seq_id, text=prompt)                  # snapshot KV/session to disk
result = lib.restore_best(engine, new_prompt)             # warm-start by exact/prefix/semantic match
```

---

### Structured sparsity 2:4 (experimental)

MGX can magnitude-prune eligible FP16/BF16 projection weights so every consecutive
group of four values along the input-feature dimension keeps two values. The portable
artifact stores the two retained values plus packed position metadata (56.25% of the
original bytes for each converted FP16/BF16 tensor). On SM80-SM89, FP16 artifacts can use
MegaGemm's standalone CUDA kernel, which issues
`mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32`
directly on CUDA 12.5+ (legacy `mma.sp` on older toolkits). The v3 kernel uses
coalesced register loads through L1/L2 for up to eight rows, and its packed projections
participate directly in MegaGemm's flat decode loop. For more than eight rows, v3
stages 512 logical K columns at once in a bank-swizzled tile,
amortizing two barriers across 16 sparse MMA instructions while every M-warp reuses the
same compressed values and PTX metadata.
The compact Triton and PyTorch semi-structured paths remain safe fallbacks. In `auto`
mode each unique `(batch, K, N)` shape is correctness-checked, timed once, and cached.
Unsupported systems load the same pruned weights through an explicit dense fallback.

```bash
python -m megagemm export-mgx \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out artifacts/qwen25-7b-fp16-sparse24.mgx \
  --dtype fp16 --quantize none --sparsity 2:4

# Sparse runtime enabled (the default on a compatible GPU)
python benchmarks/benchmark_mgx.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --mgx artifacts/qwen25-7b-fp16-sparse24.mgx \
  --sparsity 2:4 --mgx-sparse24-runtime on --mgx-sparse24-kernel auto

# Dense-pruned control using exactly the same artifact (A/B comparison)
python benchmarks/benchmark_mgx.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --mgx artifacts/qwen25-7b-fp16-sparse24.mgx \
  --sparsity 2:4 --mgx-sparse24-runtime off

# Fair FP16 dense vs FP16 2:4 suite (no INT4 in either worker)
python benchmarks/benchmark_fp16_sparse24.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --batch-sizes 1,16,64 --kernel native --warm-tokens 256 --warm-runs 5
```

Important limitations:

- The portable FP16/BF16 2:4 route is not combinable with INT8 or AWQ. The direct
  `mma.sp` kernel is FP16-only and targets SM80-SM89 (including L4). BF16 and
  other GPU generations retain the existing fallbacks. MGX also
  has a separate standalone native W4A16 + 2:4 route described below.
- Only supported `nn.Linear` attention/MLP projections whose two dimensions are
  multiples of 64 are converted. Embeddings, `lm_head`, routers, and 3-D MoE expert
  tensors remain dense.
- Magnitude pruning changes the model. Validate perplexity and task quality before
  deployment; a smaller artifact does not imply preserved accuracy.
- A fully native FP16 artifact can use MegaGemm's flat decode path. Partial, BF16,
  Triton, and PyTorch sparse preparations retain the safe module path. The legacy
  compact GEMV targets up to 64 rows by default; larger inputs use the native or
  regular sparse GEMM. Change the Triton limit with
  `MEGAGEMM_MGX_SPARSE24_TRITON_MAX_ROWS` after benchmarking the target GPU.
- `--mgx-sparse24-kernel auto` selects between native `mma.sp`, compact Triton and
  PyTorch per shape. Use `native` to force and verify the standalone Sparse Tensor Core
  route, `triton` for the compact legacy kernel, or `torch` for the library fallback.
- Set `MEGAGEMM_MGX_SPARSE24_RUNTIME=0` to force the dense-pruned fallback. Runtime
  activation, selected routes, hit counts, and failures appear under
  `decode_runtime_stats()["mgx_sparsity_runtime"]`.

The native instruction and metadata contract follow NVIDIA's
[PTX sparse MMA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-sparse-mma).

### Standalone native W4A16 + 2:4 (experimental)

MGX can quantize a regular FP16/BF16 checkpoint into its own symmetric W4A16
format. This path does **not** require AutoAWQ, Marlin, TorchAO, or cuSPARSELt.
Dense and sparse artifacts use the same groupwise quantizer, so their benchmark
is an apples-to-apples INT4 comparison:

```bash
python -X utf8 benchmarks/benchmark_native_w4a16_sparse24.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16 --batch-sizes 1,16,64 --warm-tokens 256 --warm-runs 5
```

The command exports both artifacts when absent and measures them in isolated
workers. `--kernel triton` is the default and fails loudly instead of silently
reporting a PyTorch fallback as a kernel result. Use `--force-export` after a
format change.

Native layouts:

- Dense: two signed INT4 weights per byte (`[N, K/2]`).
- 2:4: two retained signed INT4 weights per quartet (`[N, K/4]`) plus two packed
  position codes per metadata byte (`[N, K/8]`). This is 25% less packed
  weight/metadata traffic than native dense INT4, before the common scales.
- Batch 1-4 uses a direct sparse/dense Triton GEMV. Larger row counts use a
  Tensor-Core GEMM that expands packed values in registers; it never constructs
  a persistent dense FP16 weight.

This is not a claim of native INT4 sparse-Tensor-Core execution on T4/L4. It is
a standalone load-packed/compute-specialized backend. Throughput must be measured
on the target GPU, and 2:4 magnitude pruning still requires a quality evaluation.

---

## 🔢 Quantization

### INT8 W8A16 (on-the-fly)

```python
from megagemm import Int8Linear
int8_layer = Int8Linear.from_linear(original_linear)   # or engine(..., quantize='int8')
```

| Metric | Value |
|--------|-------|
| Compression | 2.0x |
| Cosine similarity | ~0.999 per layer |
| Speed | **Faster than FP16** (1.31x avg on T4 core suite) |

### AWQ 4-bit (pre-quantized)

```python
engine = InferenceEngine("Qwen/Qwen2.5-7B-Instruct-AWQ")
```

Compression 4.0x, group size 128, minimal quality impact.

### MGX native W4A16 (export-time)

```bash
python -m megagemm export-mgx --model Qwen/Qwen2.5-1.5B-Instruct --out artifacts/qwen-native-int4.mgx --dtype fp16 --quantize native-int4 --sparsity none
```

Add `--sparsity 2:4` for the standalone packed sparse variant. Existing FP16,
BF16, INT8, and AWQ loading/export paths remain unchanged.

> ⚠️ INT8 results above are **performance** results. The repository does not contain
> accuracy/perplexity evidence for every quoted quantized configuration, so these figures do not
> establish model-quality parity.

---

## 📊 Kernel Benchmarks

### RMSNorm (CUDA)

| GPU | PyTorch | MegaGemm | Speedup |
|-----|---------|----------|---------|
| **NVIDIA L4** | 0.818 ms | 0.270 ms | **3.03x** 🔥 |
| **Tesla T4** | 21,752 TPS | 36,447 TPS | **1.67x** |

> Microbenchmark: batch=32, seq=128, hidden=4096, FP16. RMSNorm is memory-bound — the kernel uses
> `half2`/`float4` vectorized loads, warp-shuffle reductions, and FP32 accumulators.

### SwiGLU (Triton)

| GPU | PyTorch | MegaGemm | Notes |
|-----|---------|----------|-------|
| **NVIDIA L4** | 58.78 ms | 56.64 ms | Fused gate+up; main benefit is memory efficiency |

---

## ⚙️ Architecture

```
HuggingFace Model
       │
       ▼
┌─────────────────────┐
│  Loader (loader.py)  │  Auto-detect model type, fuse QKV, quantize
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Engine (engine.py)  │  Chat template, tokenize, generate loop
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
 Prefill        Decode
 (SDPA)     (PagedAttn / C++ loop / fused hybrid)
    │             │
    ▼             ▼
 ┌──────────────────────────────┐
 │  Custom Kernels               │
 │  • RMSNorm (CUDA)             │
 │  • SwiGLU (Triton)            │
 │  • RoPE (CUDA)                │
 │  • PagedAttn (Triton)         │
 │  • Qwen3.5 fused decode       │
 │  • Gemma4 attention prepare   │
 │  • Gemma4 router/grouped MoE  │
 └──────────────────────────────┘
```

### Why Qwen 3.5 has a dedicated hybrid runtime

Qwen 3.5 alternates ordinary full-attention blocks with `GatedDeltaNet` linear
attention. The default synthesized layout places one full-attention layer every
four layers, so execution, state storage, prefill, and decode cannot use one
homogeneous transformer path.

| Mechanism | MegaGemm implementation |
|---|---|
| **Hybrid layer topology** | Full-attention layers use paged KV, QK normalization, partial RoPE, and an attention-output gate. Linear layers carry convolution and recurrent delta-rule state instead of allocating ordinary KV pages. |
| **GatedDeltaNet core** | Fused Q/K/V and beta/A/z input projections feed a depthwise causal convolution, learned decay/update gates, recurrent state update, RMSNorm+SiLU output gating, and the output projection. |
| **Prefill regimes** | Short sequences use a recurrent Triton prefill path. Longer sequences use the chunked delta rule, a chunk-local triangular solve, and an interchunk scan; runtime thresholds vary by GPU capability. |
| **Affine prefix scans** | Each chunk is represented as `S' = A @ S + B`. The repository implements an inclusive Hillis–Steele scan plus Torch and Triton Blelloch upsweep/downsweep paths. Blelloch remains preserved and tested, but the current selector maps it to Hillis–Steele because the dense-affine Blelloch route regressed Qwen 3.5 long-prefill on T4. |
| **Profile-driven policy** | Scan window size, warp count, and short-prefill threshold are selected from GPU generation and problem geometry. Fused RMSNorm+input-projection and RMSNormGated+output-projection routes are measured against their baselines on-device and cached only when they clear the configured gain threshold. |
| **Decode fast path** | Reusable buffers, fused causal-convolution update, fused gate/recurrent update, FP16 core output, fast linear backends, flat hybrid execution, deepfusion MLP, and fused `lm_head` argmax reduce launch and allocation overhead. |

The Qwen 3.5 kernel surface contains **20 Triton JIT routines** across linear
attention and its fused gated-normalization projections. Its regression module
contains **56 test functions** covering hybrid config parsing, partial RoPE,
runtime policy, Hillis–Steele/Blelloch correctness, recurrent-versus-chunked
parity, causal convolution, GQA, fused projections, sparse KV allocation,
continuous batching, and packed prefill.

The repository also preserves T4/L4 suite runners and vLLM matrix presets for
Qwen 3.5 0.8B, 2B, 4B, and 9B. Older project runs recorded strong leads over
vLLM in selected Qwen 3.5 regimes, but their raw reports are not present in the
current publication bundle; they therefore remain historical development
evidence rather than a numeric headline claim.

### Why Gemma 4 has a dedicated execution path

Gemma 4 is not handled as a renamed dense decoder. Its text backbone mixes layer
types and carries architecture state that changes the tensor contract from one
layer to the next:

| Mechanism | MegaGemm implementation |
|---|---|
| **Mixed attention layout** | The config materializes a per-layer schedule of sliding-window and full-attention blocks; the default layout is five sliding layers followed by one full layer. |
| **Layer-local geometry** | Head dimension, KV-head count, rotary dimension, and RoPE theta are stored per layer. Full and sliding layers can therefore use different Q/K/V shapes and positional encodings. |
| **KV reuse semantics** | Late KV-shared layers resolve to an earlier compatible source layer instead of allocating duplicate cache storage. Full-attention layers can also use the architecture's K=V mode. |
| **Per-layer embeddings (PLE)** | Token-indexed per-layer embeddings are scaled by `sqrt(hidden_size_per_layer_input)`. A projection of the base token embedding is independently scaled and RMS-normalized; both streams are combined with `1/sqrt(2)`. Each decoder layer then applies a GELU gate, elementwise conditioning, projection, post-PLE RMSNorm, residual addition, and a learned layer scalar. |
| **Gemma 4 MoE** | `Gemma4MoeMLP` implements the architecture's shared dense branch and routed expert branch. Routing includes input RMSNorm, router scaling, top-k selection, optional probability renormalization, and per-expert scales. |
| **Specialized GPU paths** | Dedicated Triton paths fuse attention preparation (Q/K/V normalization, RoPE, layout conversion, and KV write), long sliding/full prefill, MoE routing, grouped expert prefill, deterministic route packing/reduction, and selected A100 A4B-shaped decode/prefill graphs. |

The loader accepts Gemma 4 multimodal checkpoint layouts but executes the **text
backbone**; the vision/audio towers are outside the current runtime. FP16/BF16
and streaming INT8 W8A16 loading are wired for this path, while Gemma 4 AWQ is
explicitly unsupported. The regression surface contains **16 Gemma 4-specific
test modules and 236 test functions**, covering config layout, PLE, heterogeneous
KV/cache semantics, attention, MoE routing, prefill, decode, INT8, graph gates,
and eager/flat-path parity.

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Model** | [`megagemm/models/llama.py`](megagemm/models/llama.py) | Unified dense, hybrid, and MoE execution, including Gemma 4 layer-local contracts |
| **Loader** | [`megagemm/models/loader.py`](megagemm/models/loader.py) | HF weight loading, QKV fusion, quantization |
| **MGX** | [`megagemm/models/mgx.py`](megagemm/models/mgx.py) | Compiled artifact format + session state |
| **Engine** | [`megagemm/engine/engine.py`](megagemm/engine/engine.py) | Generation loop, sampling, chat templates |
| **Scheduler** | [`megagemm/engine/scheduler.py`](megagemm/engine/scheduler.py) | Continuous batching |
| **KV Cache** | [`megagemm/engine/kv_cache.py`](megagemm/engine/kv_cache.py) | Paged BlockManager + TieredBlockManager (GPU+CPU) |
| **Prophet** | [`megagemm/engine/prophet.py`](megagemm/engine/prophet.py) | Semantic state library |
| **Mesh** | [`megagemm/mesh/`](megagemm/mesh/) | Distributed replica + layer-shard inference over TTP |
| **Attention** | [`megagemm/kernels/paged_attention.py`](megagemm/kernels/paged_attention.py) | Triton paged attention decode |
| **Qwen 3.5 linear attn** | [`megagemm/kernels/linear_attention.py`](megagemm/kernels/linear_attention.py) | Recurrent/chunked GatedDeltaNet, affine scans, Hillis–Steele, and experimental Blelloch paths |
| **Qwen 3.5 fused gates** | [`megagemm/kernels/rmsnorm_gated.py`](megagemm/kernels/rmsnorm_gated.py), [`rmsnorm_gated_linear.py`](megagemm/kernels/rmsnorm_gated_linear.py) | Fused RMSNorm+SiLU gate and gated output projection |
| **Qwen3 MoE** | [`megagemm/kernels/qwen3_moe.py`](megagemm/kernels/qwen3_moe.py) | Mixture-of-experts kernels |
| **Gemma 4 attention** | [`megagemm/kernels/gemma4_attention_prepare.py`](megagemm/kernels/gemma4_attention_prepare.py) | Fused Gemma 4 Q/K/V normalization, RoPE, layout preparation, and KV write |
| **Gemma 4 MoE** | [`megagemm/kernels/gemma4_moe_router.py`](megagemm/kernels/gemma4_moe_router.py), [`gemma4_grouped_prefill.py`](megagemm/kernels/gemma4_grouped_prefill.py) | Top-k routing and grouped expert-prefill kernels |
| **RMSNorm** | [`src/rmsnorm_kernel.cu`](src/rmsnorm_kernel.cu) | CUDA kernel |
| **INT8 / W4A16 / AWQ** | [`megagemm/quantization/`](megagemm/quantization/) | W8A16 / native W4A16(+2:4) / AWQ |
| **XAI** | [`megagemm/engine/xai.py`](megagemm/engine/xai.py) | Token uncertainty telemetry + Logit Lens |
| **CPU backend** | [`microgemm/`](microgemm/) | Hand-written C++ CPU inference |

---

## 🧪 Supported Models

| Family | Example | FP16 | INT8 | AWQ | Notes |
|--------|---------|------|------|-----|-------|
| **LLaMA 3.x** | `meta-llama/Llama-3.2-3B-Instruct` | ✅ | ✅ | ✅ | Validated |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ | ✅ | — | Validated |
| **Qwen 2.5** | `Qwen/Qwen2.5-7B-Instruct` | ✅ | ✅ | ✅ | Validated |
| **Qwen 3** | `Qwen/Qwen3-8B` | ✅ | ✅ | — | Validated |
| **Qwen 3.5** | `Qwen/Qwen3.5-4B` | ✅ | — | — | Native hybrid full/GatedDeltaNet runtime with recurrent state and chunked prefill (text backbone) |
| **Gemma 2** | `google/gemma-2-*` | ✅ | ✅ | — | Dense Gemma path |
| **Gemma 4 text backbone** | `gemma4_text` architecture | ✅ | ✅ | — | Sliding/full attention, per-layer RoPE/head geometry, KV sharing, and PLE |
| **Gemma 4 MoE** | `gemma4_text` + `enable_moe_block` | ✅ | ✅ | — | Shared dense + routed experts; selected A100 A4B shapes have tuned paths |

> Models auto-detect architecture, RoPE convention, and chat templates from HuggingFace config.
> Gemma 4 support currently refers to text-backbone inference, not execution of multimodal towers.

---

## 🔬 API Reference

```python
from megagemm.engine import InferenceEngine

engine = InferenceEngine(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.float16,
    quantize='int8',          # 'int8', 'fp8', or None
    n_gpu_layers=None,        # layers on GPU (None = all)
    kv_offload=False,         # GPU↔CPU KV tiering
    num_cpu_blocks=0,         # CPU KV blocks (pinned)
    gpu_window=64,            # KV blocks kept on GPU per seq
    monitor=False,            # collect latency/quality metrics
    dashboard=False,          # live HTML dashboard
)

output = engine.generate(
    prompt="Explain paged attention in one paragraph.",
    max_new_tokens=200,
    temperature=0.7,          # 0 = greedy
    top_p=0.9,
    repetition_penalty=1.1,
    xai=False,                # opt-in interpretability report
)
```

```python
from megagemm import RMSNorm, MegaGemmTriton, RoPE

norm = RMSNorm(hidden_size=4096).cuda().half()         # drop-in nn.LayerNorm replacement
swiglu = MegaGemmTriton(d_model=4096).cuda().half()    # fused gate+up
rope = RoPE(head_dim=128, max_seq_len=2048)
```

---

## ⚠️ Honest Status & Limitations

- **Single-node throughput vs vLLM** is regime-dependent (see [Performance](#-performance)). In the
  current uncached L4 FP16 matrix, vLLM wins all six rows while MegaGemm reaches 93.1% of its
  geometric mean. Selected older quantized, hybrid-model, and capacity results are historical.
- **No FlashAttention path** on packed attention yet (per-sequence SDPA fallback on PyTorch backend).
- **MegaMesh** layer-shard mode is experimental: greedy-only, per-prompt prefill, no quantized
  shards, CPU-mediated TTP transport (no NCCL/GPU-direct).
- **INT8 numbers are speed/memory**, not validated quality; they do not establish
  accuracy or perplexity parity.
- Benchmarks are project-run **Colab/Kaggle T4 & L4** measurements (May-August 2026), generally
  fixed-decode with `--ignore-eos`. Absolute numbers remain hardware- and stack-dependent.

---

## ✅ Testing

The repository contains 46 Python test files spanning CPU logic, engine behavior, serialization,
kernel policy, CUDA/Triton correctness, model integration, and remote GPU harnesses. The suite is
currently split between unittest/pytest modules and executable assertion scripts.

Fast CPU-safe checks:

```bash
python -X utf8 tests/test_xai.py
python -X utf8 tests/test_monitor.py
python -X utf8 tests/test_deterministic.py
```

See [`tests/README.md`](tests/README.md) for test classes, GPU requirements, discovery limitations,
and current suite boundaries.

---

## 📝 Citation

```bibtex
@software{megagemm2025,
  author = {Gabriel Yogi},
  title  = {MegaGemm: High Performance LLM Inference Engine with Custom CUDA/Triton Kernels},
  year   = {2025},
  url    = {https://github.com/MadrasLe/MGRrmsnorm}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

**Author:** Gabriel Yogi
