#!/usr/bin/env bash
set -euo pipefail

# Exact one-layer B8xC2048 shared-MLP gate. No model or package setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_long_shared_mlp_prefill_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON=1

echo "== GEMMA4 LONG SHARED-MLP PREFILL GATE =="
echo "harness_rev: gemma4-long-shared-mlp-prefill-v1-fused-gelu-mul"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "package_install: disabled"
echo "shape: rows=16384 (one B8 x C2048 chunk), 30 layers x 2 chunks"
echo "candidates: current cuBLAS path vs fused GELU-mul blocks and DeepFusion control"
echo "timeout_min: benchmark=${BENCH_TIMEOUT_MIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
if "a100" not in name.lower().replace("-", " ").split() or vram_gb < 70.0:
    raise SystemExit(f"A100 80GB required, found {name} ({vram_gb:.2f}GB)")
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("gpu:", name)
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_long_shared_mlp_prefill_microbench.py \
    --warmup 3 \
    --iterations 3 \
    --repeats 7 \
    --out-json "${OUT_JSON}"
