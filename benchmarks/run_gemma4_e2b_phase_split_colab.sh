#!/usr/bin/env bash
set -euo pipefail

# Fresh-session, same-environment Gemma 4 E2B phase decomposition.  The Drive
# checkout is the source of truth; this harness performs no git operation.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

# Pin the only vLLM stack that already completed this exact Gemma 4 E2B/L4
# workload.  Validation is performed on every fresh session; no persistence is
# assumed.  The CPU torchcodec wheel avoids importing a mismatched CUDA image
# codec while torchvision remains on the same CUDA line as PyTorch.
if ! python - <<'PY'
import torch
import torchvision
import transformers
import vllm

assert torch.__version__.startswith("2.11.0+cu129"), torch.__version__
assert torchvision.__version__.startswith("0.26.0+cu129"), torchvision.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
assert vllm.__version__.startswith("0.26.0"), vllm.__version__
assert torch.cuda.is_available()
PY
then
  python -m pip install -q uv
  uv pip install --system --upgrade \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  uv pip install --system --upgrade \
    "transformers==5.14.1" \
    "numpy==2.5.2"
  uv pip install --system --reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    "torchcodec==0.16.0+cpu"
fi

python - <<'PY'
import megagemm
import torch
import torchvision
import transformers
import vllm

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("MegaGemm:", megagemm.__file__)
assert "/content/drive/MyDrive/mg/MGRrmsnorm/" in megagemm.__file__
PY

RUN_ID="${RUN_ID:-gemma4_e2b_phase_split_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_phase_split/$RUN_ID}"
mkdir -p "$OUT"

COMMON=(
  --model "$MODEL"
  --batch-size 8
  --prompt-tokens 2048
  --short-tokens 1
  --long-tokens 128
  --warmups "${WARMUPS:-3}"
  --repeats "${REPEATS:-5}"
  --max-seq-len 2304
)

echo "=== vLLM: paired 1/128-token split ==="
python benchmarks/run_gemma4_e2b_phase_split.py measure \
  --backend vllm \
  "${COMMON[@]}" \
  --output "$OUT/vllm.json"

echo "=== MegaGemm: paired 1/128-token split ==="
python benchmarks/run_gemma4_e2b_phase_split.py measure \
  --backend megagemm \
  "${COMMON[@]}" \
  --output "$OUT/megagemm.json"

echo "=== Comparison ==="
python benchmarks/run_gemma4_e2b_phase_split.py compare \
  --megagemm-json "$OUT/megagemm.json" \
  --vllm-json "$OUT/vllm.json" \
  --output "$OUT/comparison.json"

python -m zipfile -c "$OUT.zip" \
  "$OUT/megagemm.json" \
  "$OUT/vllm.json" \
  "$OUT/comparison.json"

echo "Resultados: $OUT"
echo "ZIP: $OUT.zip"
