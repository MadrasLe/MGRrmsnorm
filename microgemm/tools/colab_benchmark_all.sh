#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORK_DIR="${MICROGEMM_WORK_DIR:-/content/microgemm_bench}"
MODEL_REPO="${MICROGEMM_MODEL_REPO:-HuggingFaceTB/SmolLM2-135M-Instruct}"
GGUF_REPO="${MICROGEMM_GGUF_REPO:-lmstudio-community/SmolLM2-135M-Instruct-GGUF}"
GGUF_FILE="${MICROGEMM_GGUF_FILE:-SmolLM2-135M-Instruct-Q8_0.gguf}"
PROMPT="${MICROGEMM_PROMPT:-Explique por que o ceu parece azul em uma frase curta.}"
THREADS="${MICROGEMM_THREADS:-2}"
MAX_NEW_TOKENS="${MICROGEMM_MAX_NEW_TOKENS:-32}"
TEMPERATURE="${MICROGEMM_TEMPERATURE:-0.0}"
TOP_K="${MICROGEMM_TOP_K:-0}"
TOP_P="${MICROGEMM_TOP_P:-1.0}"
SEED="${MICROGEMM_SEED:-42}"
BUILD_JOBS="${MICROGEMM_BUILD_JOBS:-2}"

need_pkg() {
  command -v "$1" >/dev/null 2>&1 || return 0
  return 1
}

echo "[1/7] Instalando dependencias do sistema"
PACKAGES=()
need_pkg rsync && PACKAGES+=("rsync")
need_pkg git && PACKAGES+=("git")
need_pkg cmake && PACKAGES+=("cmake")
if need_pkg make || need_pkg cc || need_pkg c++; then
  PACKAGES+=("build-essential")
fi
if ((${#PACKAGES[@]} > 0)); then
  apt-get update -y >/dev/null
  apt-get install -y "${PACKAGES[@]}" >/dev/null
fi

echo "[2/7] Garantindo huggingface_hub"
python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
PY

echo "[3/7] Sincronizando projeto para ${WORK_DIR}"
mkdir -p "$WORK_DIR"
rsync -a --delete \
  --exclude ".cache/" \
  --exclude "out/" \
  --exclude "*.o" \
  --exclude "libmicrogemm.a" \
  --exclude "microgemm" \
  --exclude "microgemm-convert" \
  --exclude "microgemm-text" \
  "$SRC_DIR/" "$WORK_DIR/"

echo "[4/7] Baixando snapshot do modelo base"
MODEL_DIR="$(
MODEL_REPO="$MODEL_REPO" python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id=os.environ["MODEL_REPO"])
print(path)
PY
)"
echo "[microgemm] modelo HF em ${MODEL_DIR}"

echo "[5/7] Compilando MicroGemm localmente"
cd "$WORK_DIR"
make clean
make -j"$BUILD_JOBS"
chmod +x ./microgemm ./microgemm-convert ./microgemm-text
mkdir -p out
./microgemm-convert from-dir "$MODEL_DIR" out/model.mgm

echo "[6/7] Baixando GGUF equivalente"
GGUF_PATH="$(
GGUF_REPO="$GGUF_REPO" GGUF_FILE="$GGUF_FILE" WORK_DIR="$WORK_DIR" python - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id=os.environ["GGUF_REPO"],
    filename=os.environ["GGUF_FILE"],
    local_dir=os.path.join(os.environ["WORK_DIR"], ".cache", "gguf"),
)
print(path)
PY
)"
echo "[llama.cpp] GGUF em ${GGUF_PATH}"

echo "[7/7] Buildando llama.cpp e rodando benchmark"
python -u tools/benchmark_vs_llamacpp.py \
  --model-dir "$MODEL_DIR" \
  --gguf-path "$GGUF_PATH" \
  --llama-dir ".cache/llama.cpp" \
  --llama-ctx-size 512 \
  --mgm-path "out/model.mgm" \
  --microgemm-text-bin "./microgemm-text" \
  --microgemm-convert-bin "./microgemm-convert" \
  --skip-warmup \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --threads "$THREADS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --seed "$SEED"
