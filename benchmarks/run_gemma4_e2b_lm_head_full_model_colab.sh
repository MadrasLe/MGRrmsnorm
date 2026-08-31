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

# Self-contained MegaGemm model-loading dependencies. No vLLM, torchvision,
# torchcodec, CUDA runtime replacement, or editable package build.
python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

RUN_ID="${RUN_ID:-gemma4_e2b_lm_head_full_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_lm_head_full/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/run_gemma4_e2b_lm_head_full_model_gate.py \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-7}" \
  --output "$OUT/decision.json"

echo "RESULTADO: $OUT/decision.json"
