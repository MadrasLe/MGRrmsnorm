#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

python - <<'PY'
import torch
import triton

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
assert torch.cuda.is_available(), "CUDA indisponível"
PY

RUN_ID="${RUN_ID:-gemma4_e2b_lm_head_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_lm_head/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/run_gemma4_e2b_lm_head_sweep.py \
  --warmups "${WARMUPS:-10}" \
  --iterations "${ITERATIONS:-40}" \
  --repeats "${REPEATS:-7}" \
  --output "$OUT/decision.json"

echo "RESULTADO: $OUT/decision.json"
