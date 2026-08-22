# Architecture

MegaGemm is an experimental full-stack inference runtime rather than a single
kernel library. The project is solo-designed and architected by Gabriel Yogi;
AI-assisted development is used to accelerate implementation and bounded
validation work such as kernel tests, latency measurement, and comparisons.

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

### Model execution

The main model implementation contains architecture-specific paths for dense,
MoE, full-attention, and hybrid linear-attention families. Specialized fast
paths coexist with fallbacks because the runtime targets multiple GPU
generations and model shapes.

### KV cache and scheduling

The block manager provides paged KV allocation. Layer-aware allocation avoids
reserving full-attention KV for layers that use recurrent/linear-attention
state. The tiered manager adds CPU offload. The scheduler manages request state,
prefill admission, decode iteration, and cleanup.

### Kernel layers

The repository contains custom Triton routines, native CUDA kernels, C++ hot
path helpers, and PyTorch fallbacks. As of the August 2026 source inventory,
there are 154 `@triton.jit` routines and 14 native CUDA `__global__` bodies;
approximately 163 distinct custom GPU launch points are directly identifiable.
These counts describe source entry points, not an ABI guarantee.

### MGX and Prophet

MGX packages transformed runtime weights and optional session state into a
project-specific artifact. Prophet stores and retrieves persistent semantic
state/KV snapshots. These are original experimental subsystems and should be
presented separately from the stable checkpoint-loading path.

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

## Maturity labels

Use these labels in issues and documentation:

- **Core:** part of the primary model loading/generation path.
- **Supported:** has a documented runnable path and regression coverage.
- **Experimental:** working research path whose API or tuning may change.
- **Harness:** validation/benchmark tooling, not runtime API.
- **Archived result:** dated evidence retained for historical comparison.

The package metadata intentionally uses `Development Status :: 3 - Alpha`.
This reflects breadth and rapid development, not a lack of technical depth.

## Known consolidation targets

- Split the large model implementation by architecture and execution path.
- Create one curated CPU test command and hardware-specific GPU test matrices.
- Promote selected benchmark summaries while leaving generated sweep output
  outside version control.
- Record exact software versions and raw JSON/CSV artifacts for every result
  quoted on the project front page.
- Separate stable public APIs from research knobs and environment variables.
