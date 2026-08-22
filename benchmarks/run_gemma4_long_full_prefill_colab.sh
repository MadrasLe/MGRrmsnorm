#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-free A100 gate for Gemma4 H512/GQA8 long full attention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_long_full_prefill_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GEMMA4 LONG FULL-ATTENTION PREFILL GATE =="
echo "harness_rev: gemma4-long-full-prefill-v1"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "package_install: disabled"
echo "shape: B8 x C2048 H512/GQA8 (two chunks, five global layers)"
echo "timeout_min: benchmark=${BENCH_TIMEOUT_MIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
gpu = torch.cuda.get_device_name(0)
if "a100" not in gpu.lower().replace("-", " ").split():
    raise SystemExit(f"This gate requires an A100, found: {gpu}")
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("gpu:", gpu)
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_long_full_prefill_microbench.py \
    --warmup 3 \
    --iterations 5 \
    --repeats 7 \
    --minimum-speedup 1.10 \
    --out-json "${OUT_JSON}"
