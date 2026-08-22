#!/usr/bin/env bash
set -euo pipefail

# One model-free gate for long Q/K/V attention preparation on the paid A100.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_long_attention_prepare_a100.json}"
ONLY="${ONLY:-sliding}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== GEMMA4 LONG ATTENTION-PREPARE GATE =="
echo "harness_rev: gemma4-long-attention-prepare-v2-preconditioned"
echo "model_download: disabled"
echo "vllm_install: disabled"
echo "package_install: disabled"
echo "shape: B8 x C2048, sliding H256/GQA2 and full H512/GQA8"
echo "candidate: fused RMSNorm + RoPE + Q/K/V layout preparation"
echo "measured_attention_types: ${ONLY}"
echo "precondition_pairs: 5"
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
  python benchmarks/run_gemma4_long_attention_prepare_microbench.py \
    --only "${ONLY}" \
    --precondition-pairs 5 \
    --warmup 2 \
    --iterations 2 \
    --repeats 7 \
    --out-json "${OUT_JSON}"
