# Colab GPU rebenchmark protocol

This protocol refreshes the dated Qwen 2.5 3B GPU headline using MegaGemm and
vLLM in the same Colab runtime. Start with the compact matrix; expand only after
the pipeline and artifact format are confirmed.

## 1. Start a fresh GPU runtime

Before importing PyTorch, record the allocation:

```bash
!nvidia-smi
!lscpu
```

Use a fresh runtime so cached packages and previously loaded CUDA libraries do
not contaminate the environment.

## 2. Make the project available

After the repository is pushed:

```bash
!git clone https://github.com/MadrasLe/MGRrmsnorm.git /content/MGRrmsnorm
%cd /content/MGRrmsnorm
```

Until then, upload/copy the current project into `/content/MGRrmsnorm` and
change to that directory. If the project is in Google Drive, use its exact
path rather than searching for the first directory with a matching name. For
example, the 2026-08-22 L4 run used:

```python
%cd /content/drive/MyDrive/mg/MGRrmsnorm
```

Before installing, verify that both the package and publication runner belong
to the intended copy:

```bash
!test -f pyproject.toml
!test -f benchmarks/run_publication_gpu_suite.py
```

## 3. Install one shared software stack

Install vLLM first so its PyTorch/CUDA constraints define the shared benchmark
environment. Then install MegaGemm editable without replacing that stack. Pin
the complete wheel variant: an unqualified `pip install -U vllm` selected the
CUDA 13 wheel in the August 2026 Colab image, replacing the image's CUDA 12
stack and making the result harder to reproduce.

```bash
!python -m pip install -q -U uv
!uv pip install --system --upgrade \
  "https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --torch-backend=cu129
!python -m pip install -e ".[benchmark]" --no-build-isolation
```

This pin reproduces the published L4 stack (`vllm 0.27.1+cu129`, `torch
2.13.0+cu129`). A newer vLLM release defines a new dated benchmark series rather
than silently replacing this baseline.

Run installation and benchmarks in shell child processes and do not reset or
replace the Colab runtime between them. A new backend allocation loses the
installed wheel and invalidates the same-environment premise.

If the editable build reports a CUDA toolkit mismatch, save the complete log
instead of silently skipping it. Do not install a different PyTorch between the
MegaGemm and vLLM runs.

Record the final environment:

```bash
!python -m pip freeze > megagemm_colab_freeze.txt
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print(torch.cuda.get_device_name(0))"
!python -c "import vllm, transformers, triton; print('vllm', vllm.__version__); print('transformers', transformers.__version__); print('triton', triton.__version__)"
```

## 4. Dry-run the compact matrix

```bash
!python benchmarks/run_publication_gpu_suite.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --variants megagemm-fp16,vllm-fp16 \
  --batch-sizes 1,8 \
  --prompt-tokens 128,512,2048 \
  --max-new-tokens 128 \
  --repeats 3 \
  --warmup 1 \
  --dry-run
```

Check that the automatically detected hardware label matches the allocation.

## 5. Run FP16 versus FP16

```bash
!python benchmarks/run_publication_gpu_suite.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --variants megagemm-fp16,vllm-fp16 \
  --batch-sizes 1,8 \
  --prompt-tokens 128,512,2048 \
  --max-new-tokens 128 \
  --repeats 3 \
  --warmup 1
```

The wrapper:

- runs each backend in a fresh child process;
- uses greedy decoding and fixed output length;
- disables vLLM prefix caching;
- preserves CUDA Graphs unless explicitly disabled;
- writes JSONL, summary JSON, CSV, comparison CSV, and a manifest;
- records package, driver, GPU, Git revision, and selected environment data;
- creates one ZIP artifact at the end.

## 6. Optional MegaGemm INT8 pass

After FP16 completes successfully:

```bash
!python benchmarks/run_publication_gpu_suite.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --variants megagemm-fp16,vllm-fp16,megagemm-int8 \
  --batch-sizes 1,8 \
  --prompt-tokens 128,512,2048 \
  --max-new-tokens 128 \
  --repeats 3 \
  --warmup 1
```

INT8 must be reported as a separate performance/memory configuration, not as a
like-for-like FP16 backend comparison.

## 7. What to send back

Attach:

1. the generated `publication_*.zip`;
2. `megagemm_colab_freeze.txt`;
3. the first `nvidia-smi` output;
4. any installation, OOM, or fallback warnings from the console.

Do not send only the final tok/s line. The summary JSON contains the metadata
needed to decide whether a result is publishable.

## 8. Full matrix after validation

Once the compact run is clean, use:

```text
batch sizes:   1,2,4,8
prompt tokens: 128,512,1024,2048
output tokens: 128
warmup:        2
repeats:       5
```

Run the full matrix only if Colab time and model-download limits permit it.
