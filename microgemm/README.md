# MicroGemm

MicroGemm is the standalone, minimal runtime track for MegaGemm.

The real target is not just "smaller MegaGemm".
The target is closer to the spirit of `llama.cpp`, but with a cleaner and more controlled architecture:

- no Python;
- no PyTorch;
- no Hugging Face;
- CPU-first for weak machines, old desktops, mini PCs, integrated graphics, and IoT-class deployments;
- tiny native runtime plus native conversion tooling;
- predictable memory behavior;
- low dependency surface;
- one clean codebase focused on inference, not research features.

This folder is intentionally separate from the main `megagemm/` package.
MicroGemm is not allowed to depend on that package, or on any Python-side model stack, to function.

## Design priorities

MicroGemm is designed around:

- purity: the runtime and converter stand on their own;
- portability: easy to cross-compile and ship on constrained systems;
- simplicity: fewer moving parts than the main MegaGemm codebase;
- mechanical sympathy: tight CPU kernels, low allocator pressure, explicit buffers, predictable layouts;
- practical performance: strong real-world decode speed on modest hardware.

These constraints position MicroGemm as a focused systems project rather than a
Python-first ML stack.

## Current Scope

This first cut sets up:

- a standalone native project layout;
- a small public C API;
- a binary container format for exported models;
- model manifest loading and inspection;
- runtime scratch allocation sized for decoder-only models;
- a MicroGemm-owned INT8 weight loader for `.mgm` payloads;
- a CPU-native greedy decode smoke path driven by token ids;
- a native converter utility that can inspect `config.json` and convert `config.json + model.safetensors -> .mgm` without Python;
- a first native text CLI that reads `tokenizer.json`, tokenizes prompt text, runs generation, and decodes output without Python;
- an Android ARM64 CPU build path through CMake/NDK, with NEON kernels and no OpenMP dependency on Android;
- a backend capabilities API/CLI entrypoint for future MegaMesh workers;
- a native kernel self-test command for scalar/vector CPU primitive validation;
- initial CPU-native primitives extracted from the MegaGemm direction:
  - FP32 -> INT8 activation quantization
  - RMSNorm
  - INT8 GEMV
  - decode attention dot/value accumulation

The project is now past the "manifest only" stage, but it is not yet a finished standalone product.
What exists today is enough to compile, inspect a model container, load INT8 weights, run a CPU-only greedy decode with explicit token ids, inspect native compatibility from a real `config.json`, produce a first `.mgm` directly from a single-file `model.safetensors`, and run a first text-first path from `tokenizer.json`.
What still does not exist yet is the harder production layer: better tokenizer coverage, chat-template handling, multi-file checkpoint ingestion, sampling, and broader model-family coverage.

## Layout

```text
microgemm/
  include/microgemm/
  src/
```

## Build

Unix-like environments:

```bash
cd microgemm
make
```

Windows PowerShell:

```powershell
Set-Location microgemm
./build.ps1
```

Both flows expect a system C compiler to already exist.

CMake host build:

```bash
cmake -S microgemm -B microgemm/build-host
cmake --build microgemm/build-host
```

Android ARM64 build with the Android NDK:

```bash
cmake -S microgemm -B microgemm/build-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-23 \
  -DMICROGEMM_BUILD_CLI=ON \
  -DMICROGEMM_BUILD_TEXT=ON \
  -DMICROGEMM_BUILD_CONVERT=OFF
cmake --build microgemm/build-android-arm64
```

The Android target is CPU-only in this phase: ARM64 NEON is used for the hot
primitive kernels, OpenMP is intentionally not linked, and `microgemm-convert`
stays a host-side tool.

## CLI

Planned command style:

```bash
microgemm version
microgemm capabilities
microgemm kernel-selftest
microgemm inspect model.mgm
microgemm runtime-dryrun --hidden-size 2048 --intermediate-size 5632 --layers 24 --q-heads 16 --kv-heads 4 --head-dim 128 --vocab-size 32000 --max-seq-len 2048
microgemm decode-smoke model.mgm --tokens 1,2,3 --generate 4
microgemm generate-ids model.mgm --tokens 1,2,3 --max-new-tokens 16
microgemm-convert inspect-config path/to/config.json
microgemm-convert from-files path/to/config.json path/to/model.safetensors out/model.mgm
microgemm-convert from-dir path/to/model-dir out/model.mgm
microgemm-text generate out/model.mgm path/to/tokenizer.json --prompt "Explain the sky in one sentence." --temperature 0.8 --top-k 40 --top-p 0.95
microgemm-text batch-generate out/model.mgm path/to/tokenizer.json --prompt-file a.txt --prompt-file b.txt --max-new-tokens 128 --ignore-eos
```

## External Benchmark

To compare MicroGemm fairly against the Hugging Face CPU path without turning
the runtime itself into a Python dependency, use the standalone harness in
`tools/benchmark_vs_hf.py`.

Example from inside the standalone `microgemm/` folder:

```bash
python tools/benchmark_vs_hf.py \
  --model-dir /path/to/model-dir \
  --mgm-path out/model.mgm \
  --microgemm-text-bin ./microgemm-text \
  --prompt "Explain why the sky looks blue in one short sentence." \
  --max-new-tokens 32 \
  --threads 2 \
  --temperature 0.0
```

That benchmark reports `prefill_ms`, `decode_ms`, `total_ms`, `prefill_tps`,
`decode_tps`, and `total_tps` for both MicroGemm and Hugging Face side by side.

The project-wide CPU inference matrix uses the MicroGemm backend:

```bash
python ../benchmarks/benchmark_inference_matrix.py \
  --backend microgemm \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hardware-label colab-xeon \
  --device cpu \
  --batch-sizes 1,2,4,8 \
  --prompt-tokens 64,256,512,1024 \
  --max-new-tokens 128 \
  --repeats 5 \
  --warmup 1 \
  --max-seq-len 2048 \
  --ignore-eos \
  --microgemm-batch-mode adaptive \
  --out-dir bench_results/qwen25_cpu_microgemm_matrix \
  --run-id qwen25_05b_microgemm_cpu_xeon
```

This matrix backend defaults to `--microgemm-batch-mode adaptive`: batch 1 uses
the single-request fast path, small CPU batches use concurrent workers, and
larger batches use `microgemm-text batch-generate`: one model load, one KV state
per request, and a MicroGemm-owned active-request decode scheduler inside the
native process. Pure `continuous` mode remains available to measure continuous
batching. Prompt prefill uses a persistent native C++ worker pool and assigns
OpenMP only to the inner GEMV work, avoiding nested OpenMP as the control plane
and avoiding per-token thread creation. Decode now advances the active request
set through one batched C call, batching the int8 linear projections for `qkv`,
`o_proj`, `gate_up`, `down`, and `lm_head`; attention/KV remains per request
because each sequence has its own cache layout and context length. Matrix
`output_tps` is computed from
MicroGemm's reported runtime `total_ms`. `batch-generate` reports `setup_ms`
separately for per-request KV/workspace/logit allocation, reports
`scheduler_outer_threads` / `scheduler_inner_threads` /
`scheduler_lm_head_threads`, reports `batched_decode_calls` /
`batched_decode_tokens` plus legacy `batched_lm_head_*` counters, and CLI
wall-clock is preserved separately in the raw row under `microgemm.wall_*`.

The native converter supports `--quant int8` and `--quant int4`. INT4 stores
linear, embedding, and LM-head weights as packed signed 4-bit rowwise tensors
(`weight_i4` plus row scales and `row_sum`). The INT4 runtime keeps those
tensors packed in memory and uses packed GEMV kernels for decode/prefill/logits
paths. Persisted `row_sum` tensors avoid scanning and unpacking all INT4 weights
at model load time. INT4 batch decode uses AVX2 batched-row tiles for generic
GEMV paths and a fused packed `gate_up` path before SwigLU. INT8 still uses its
separate fused fast kernels. The per-op profile is the evidence surface for
remaining INT4 bottlenecks.

To compare against `llama.cpp`, use `tools/benchmark_vs_llamacpp.py` with a
GGUF of the same model:

```bash
python tools/benchmark_vs_llamacpp.py \
  --model-dir /path/to/model-dir \
  --gguf-path /path/to/model-q8_0.gguf \
  --mgm-path out/model.mgm \
  --microgemm-text-bin ./microgemm-text \
  --prompt "Explain why the sky looks blue in one short sentence." \
  --max-new-tokens 32 \
  --threads 2 \
  --temperature 0.0
```

`tools/colab_benchmark_all.sh` is the single Colab entrypoint for the full flow
without benchmarking from Google Drive.
It copies the standalone `microgemm/` tree into `/content`, recompiles there,
downloads the HF snapshot and matching GGUF, builds `llama.cpp`, converts the
model to `.mgm`, and runs the final CPU benchmark:

```bash
%%bash
cd /content/drive/MyDrive/microgemm
bash tools/colab_benchmark_all.sh
```

The prompt and selected benchmark knobs are configurable through environment
variables:

```bash
%%bash
cd /content/drive/MyDrive/microgemm
MICROGEMM_PROMPT="Explique por que o ceu parece azul em uma frase curta." \
MICROGEMM_THREADS=2 \
MICROGEMM_MAX_NEW_TOKENS=32 \
MICROGEMM_TEMPERATURE=0.0 \
bash tools/colab_benchmark_all.sh
```

## Export

There is currently no accepted Python-based export path.
If a workflow needs Python, PyTorch, or Hugging Face to produce a `.mgm`, that workflow does not satisfy the MicroGemm architecture constraint.

The official direction is the native converter.
The current implemented path is:

```bash
microgemm-convert inspect-config path/to/config.json
microgemm-convert from-files path/to/config.json path/to/model.safetensors out/model.mgm
```

That path reads a Hugging Face-style `config.json` natively, parses a single-file `model.safetensors` natively, builds the packed INT8 layout the runtime expects, and writes a `.mgm` without Python.

## Container Direction

The `.mgm` container is meant to hold:

- a compact header;
- model config;
- tensor directory;
- packed tensor payloads laid out for the standalone runtime.

For now, the runtime already knows how to open and inspect a container manifest.
It can also materialize an INT8 model from the container and run a minimal greedy decode smoke path on CPU.

## Design Notes

- `src/microgemm_runtime.c` owns runtime state and scratch sizing.
- `include/microgemm/microgemm_platform.h` centralizes CPU feature macros for scalar, AVX2, and ARM64 NEON builds.
- `src/microgemm_format.c` owns binary container validation helpers.
- `src/microgemm_ops_cpu.c` is the seed for reusable CPU-native kernels, including scalar/x86/ARM64 primitive paths.
- `src/microgemm_decode_cpu.c` now holds a MicroGemm-owned INT8 decode foundation with its own weight structs, KV layout, workspace, and `decode_step`.
- `src/microgemm_model_i8.c` now materializes a MicroGemm-owned INT8 model directly from `.mgm` tensor payloads.
- `src/microgemm_runtime.c` can already inspect tensor metadata, find tensors by name, and read raw tensor payload bytes from a `.mgm` file.
- `src/microgemm_convert.cpp` now owns the first native `config.json + model.safetensors -> .mgm` conversion path.
- `src/microgemm_text.cpp` now owns the first native `tokenizer.json + prompt text -> ids -> generated text` path.
- `microgemm-text` now also reports `setup_ms` for native batch setup plus `prefill_ms`, `decode_ms`, `total_ms`, `prefill_tps`, `decode_tps`, and `total_tps`.

## Current Export Scope

The native exporter currently targets the simplest family that matches the decode backend:

- LLaMA-like models
- Mistral-like models
- Qwen2-like models

The current native converter is intentionally narrow but can ingest the dense
text paths used by Llama/Mistral-style models, Qwen2/Qwen3 dense checkpoints,
and Gemma/Gemma2/Gemma3 text checkpoints:

- single-file and sharded `safetensors` checkpoints;
- common decoder-only `model.layers.*` / text-model prefixed naming layouts;
- row-wise INT8 and packed INT4 storage for linear weights;
- QK norm, Gemma-style extra RMSNorms, Gemma embedding scaling, and Gemma
  attention/logit softcapping;
- rope cache generated natively into the container.

The first native text path is also intentionally narrow:

- `tokenizer.json` with `model.type = BPE`;
- byte-level BPE style tokenizers only;
- greedy by default, with first native `temperature` / `top-k` / `top-p` sampling controls;
- plain prompt text only, no native chat-template renderer yet.

It still rejects more advanced variants that the runtime does not yet model correctly, such as:

- partial rotary models
- mixed full/linear attention layouts
- MoE checkpoints such as Qwen3-MoE
- long-context hybrid local/global attention semantics that require per-layer
  attention masks or per-layer RoPE variants

## Current non-goals

The current scope excludes:

- a second research sandbox;
- a large abstraction framework;
- a runtime that depends on the main MegaGemm package to execute inference;
- a runtime or converter that depends on Python, PyTorch, or Hugging Face.

Portability, simplicity, and runtime purity are acceptance constraints for new
runtime surface.

## Current implementation boundary

- The first decoder-only target layout remains versioned rather than frozen.
- Validation is concentrated on Linux/Colab CPU and the hardware represented in
  dated benchmark notes.
- Native tokenizer coverage is currently limited to the documented BPE path.
- The converter does not ingest multi-file safetensors checkpoints directly.
- Sampling exposes the documented initial temperature/top-k/top-p controls.
- Model loading is not memory-mapped, and deployment packaging remains minimal.
