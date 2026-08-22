#!/usr/bin/env bash
set -euo pipefail

# Fresh-VM, one-layer Gemma4 expert-prefill tuning. This script intentionally
# does not download the model or install vLLM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-5}"
BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_expert_prefill_tune_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGAGEMM_DISABLE_CUDA_RMSNORM=1

echo "== GEMMA4 EXPERT PREFILL TUNE =="
echo "harness_rev: gemma4-expert-prefill-tune-v4-fixed-route-pack"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "timeouts_min: install=${INSTALL_TIMEOUT_MIN} benchmark=${BENCH_TIMEOUT_MIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "${INSTALL_RUNTIME}" = "1" ]; then
  timeout --foreground --signal=INT --kill-after=30s "${INSTALL_TIMEOUT_MIN}m" \
    python -m pip install -q -U \
      "transformers>=4.57" safetensors accelerate sentencepiece
fi

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_hot_layer_microbench.py \
    --dtype bf16 \
    --only-expert-prefill \
    --prefill-fixed-route-pack-only \
    --warmup 12 \
    --prefill-iterations 10 \
    --prefill-target-savings-ms 7.50 \
    --repeats 7 \
    --out-json "${OUT_JSON}"
