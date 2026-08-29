#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RUN_ID="${RUN_ID:-gemma4_e2b_ple_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$REPO/bench_results/gemma4_e2b_ple_conditioned_gelu/$RUN_ID}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige uma NVIDIA L4."
  exit 2
}

python - <<'PY'
import megagemm
from megagemm.kernels.swiglu import conditioned_gelu_tanh_forward

print("MegaGemm:", megagemm.__file__)
print("PLE conditioned GELU kernel: OK", callable(conditioned_gelu_tanh_forward))
PY

python benchmarks/run_gemma4_e2b_ple_conditioned_gelu.py \
  --model "$MODEL" \
  --out-root "$OUT_ROOT" \
  --max-new-tokens 64 \
  --repeats 5 \
  --warmup 2

echo "Artifacts: $OUT_ROOT"
