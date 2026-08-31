#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este perfil exige NVIDIA L4."
  exit 2
}

# MegaGemm-only dependencies; no competing engine or native rebuild.
python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

RUN_ID="${RUN_ID:-gemma4_e2b_mm_shapes_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_mm_shapes/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/profile_gemma4_decode.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --dtype bf16 \
  --batch-size 8 \
  --prompt-tokens 2048 \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --warmup-tokens "${WARMUP_TOKENS:-4}" \
  --max-seq-len 2304 \
  --max-batch-size 8 \
  --ignore-eos \
  --out "$OUT/profile.json"

echo "RESULTADO: $OUT/profile.json"
