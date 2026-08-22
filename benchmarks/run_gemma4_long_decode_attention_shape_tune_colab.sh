#!/usr/bin/env bash
set -euo pipefail

# Fresh-VM, model-free gate after the paid long-segment promotion.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-5}"
BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_long_decode_attention_shape_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GEMMA4 B16 LONG DECODE ATTENTION SHAPE TUNE =="
echo "harness_rev: gemma4-long-decode-attention-shape-v1"
echo "fresh_vm: supported"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "shape: batch=16 context=2111"
echo "baseline: promoted sliding=32 segments, full=8 segments"
echo "candidates: tile 16/32/64 plus focused warp/stage/reduce variants"
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
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
if "a100" not in name.lower() or vram_gb < 70.0:
    raise SystemExit(f"A100 80GB required, found {name} ({vram_gb:.2f}GB)")
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("gpu:", name)
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_long_decode_attention_shape_tune.py \
    --context 2111 \
    --table-blocks 132 \
    --warmup 5 \
    --iterations 100 \
    --repeats 7 \
    --out-json "${OUT_JSON}"
