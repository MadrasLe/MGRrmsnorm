#!/usr/bin/env bash
set -euo pipefail

# Self-contained loaded-model gate for a fresh Colab session.  This script
# intentionally performs no repository sync, competing backend install, or
# native-extension operation.  It installs only missing Python inference
# dependencies and otherwise leaves the environment untouched.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate foi delimitado para NVIDIA L4."
  exit 2
}

if ! python - <<'PY'
import huggingface_hub
import psutil
import safetensors
import sentencepiece
import tqdm
import transformers
import triton
PY
then
  python -m pip install -q \
    huggingface_hub safetensors transformers sentencepiece psutil tqdm
fi

python - <<'PY'
import megagemm
import torch
import triton

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
print("Triton:", triton.__version__)
assert torch.cuda.is_available(), "CUDA indisponível"
PY

RUN_ID="${RUN_ID:-gemma4_e2b_h512_bridge_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$REPO/bench_results/gemma4_e2b_h512_bridge_full_model/$RUN_ID}"

python benchmarks/run_gemma4_e2b_h512_bridge_full_model_gate.py \
  --model "$MODEL" \
  --out-root "$OUT_ROOT" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --repeats "${REPEATS:-5}" \
  --warmup "${WARMUPS:-2}"

echo "Artefato final: $OUT_ROOT/decision.json"
