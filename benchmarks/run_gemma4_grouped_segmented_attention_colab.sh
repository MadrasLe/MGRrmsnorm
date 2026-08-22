#!/usr/bin/env bash
set -euo pipefail

# One paid A100 gate. No checkpoint, Hugging Face request, vLLM install, or
# package mutation is allowed in this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_grouped_segmented_attention_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GEMMA4 GROUPED SEGMENTED ATTENTION GATE =="
echo "harness_rev: gemma4-grouped-segmented-attention-v1"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "runtime_install: disabled"
echo "benchmark_timeout_min: ${BENCH_TIMEOUT_MIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if "a100" not in torch.cuda.get_device_name(0).lower():
    raise SystemExit("This paid gate is intentionally restricted to A100")
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_grouped_segmented_attention_microbench.py \
    --context 64 \
    --table-blocks 6 \
    --warmup 5 \
    --iterations 100 \
    --repeats 5 \
    --out-json "${OUT_JSON}"
