#!/usr/bin/env bash
set -euo pipefail

# Exact B16x25 full-model topology gate. No checkpoint download or vLLM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-5}"
BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_b16_full_model_graph_preflight_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGAGEMM_DISABLE_CUDA_RMSNORM=1
export MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL=1
export MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK=1
export MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS=4096
export MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS=4096

echo "== GEMMA4 B16 CHECKPOINT-FREE FULL-MODEL GRAPH GATE =="
echo "harness_rev: gemma4-full-model-graph-v105-direct-deferred-kv-output"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "shape: batch=16 context=25 rows=400 layers=30 dtype=bf16"
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
  python benchmarks/run_gemma4_b16_full_model_graph_preflight.py \
    --replays 3 \
    --out-json "${OUT_JSON}"
