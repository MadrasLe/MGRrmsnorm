# Dependency model

MegaGemm keeps the base package small and places feature-specific integrations
behind optional extras. Dependency counts below refer to **direct declarations**
in `pyproject.toml`; packages such as PyTorch and Transformers have their own
transitive dependency trees.

## Direct dependency count

| Installation profile | Direct external dependencies | Purpose |
|---|---:|---|
| Base package | 1 | `torch` |
| Typical GPU inference (`.[inference]`) | 6 total | Base + Triton, Hugging Face Hub, Safetensors, Transformers, SentencePiece |
| CPU helpers (`.[cpu]`) | 2 total | Base + NumPy |
| Stable full development profile (`.[all]`) | 12 total | Inference, CPU, monitoring, benchmark helpers, AWQ, and test tools |
| Every declared path including hardware-specific accelerators | 14 total | Stable full profile + `causal-conv1d` and `flash-attn` |

The project currently declares these 14 unique direct package names across all
profiles:

```text
torch
triton
huggingface_hub
safetensors
transformers
sentencepiece
numpy
psutil
tqdm
autoawq
pytest
pytest-benchmark
causal-conv1d
flash-attn
```

Only `torch` is mandatory for the base installation. The final number of
installed distributions is larger because dependencies are transitive; the
direct count is not the complete environment size.

## Extras

| Extra | Use |
|---|---|
| `inference` | GPU inference from Hugging Face checkpoints |
| `embeddings` | Encoder and Sentence Transformers-compatible embeddings |
| `cpu` | NumPy-backed CPU and tensor-codec helpers |
| `mesh` | MegaMesh model loading and binary tensor transport |
| `monitoring` | Host-memory telemetry through `psutil` |
| `benchmark` | Benchmark telemetry and progress output |
| `quantization` | AutoAWQ integration |
| `performance` | Optional hardware-specific acceleration; not included in `all` |
| `dev` | Pytest and pytest-benchmark |
| `all` | Stable extras, excluding fragile hardware-specific accelerators |

## Install examples

Minimal package:

```bash
MEGAGEMM_SKIP_NATIVE=1 pip install -e . --no-build-isolation
```

Normal GPU inference development:

```bash
pip install -e ".[inference,dev]" --no-build-isolation
```

CPU-oriented work:

```bash
MEGAGEMM_SKIP_NATIVE=1 pip install -e ".[cpu,dev]" --no-build-isolation
```

On PowerShell, set the environment switch before invoking pip:

```powershell
$env:MEGAGEMM_SKIP_NATIVE = "1"
python -m pip install -e ".[cpu,dev]" --no-build-isolation
```

`MEGAGEMM_SKIP_NATIVE=1` skips the optional TTP C helper, C++ decode-loop
helper, and CUDA extensions. The PyTorch/Triton fallbacks remain available.
`MEGAGEMM_SKIP_CUDA=1` skips only CUDA extensions while retaining native CPU/C
helpers.

## Intentionally external comparison backends

`vllm` is not a MegaGemm runtime dependency. It is installed only in isolated
benchmark environments. `llama.cpp` is an external executable/build used by
the MicroGemm comparison harnesses. Keeping both out of the package extras
prevents benchmark baselines from inflating the runtime dependency surface.

## Imports supplied transitively or by the environment

- `tokenizers` and `jinja2` are used by tokenizer/chat-template paths and are
  normally installed by Transformers.
- `IPython` and `google.colab` are optional notebook display integrations and
  are guarded at runtime.
- `awq_ext` is supplied by the AutoAWQ installation when available.
- `rmsnorm_cuda_ops`, `sparse24_cuda_ops`, `megagemm_decode_ops`, and
  `megagemm_ttp_native` are built from sources in this repository.

## System dependencies

Python package counts do not include system-level build/runtime requirements:

- a C/C++ compiler for native helpers;
- CUDA Toolkit and `nvcc` for CUDA extensions;
- CMake or Make for the standalone MicroGemm CPU runtime;
- a compatible NVIDIA driver for CUDA execution.
