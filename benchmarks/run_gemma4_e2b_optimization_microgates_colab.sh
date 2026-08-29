#!/usr/bin/env bash
set -euo pipefail

# Self-contained Colab runner.  It uses the Drive copy directly, performs no
# git operation, downloads no model, installs no vLLM, and rebuilds the native
# extension in every fresh runtime before the cuBLASLt gate.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
RUN_ID="${RUN_ID:-gemma4_e2b_microgates_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$REPO/bench_results/gemma4_e2b_optimization_microgates/$RUN_ID}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_ROOT"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: estes gates exigem uma NVIDIA L4."
  exit 2
}

python - <<'PY'
import megagemm
import torch

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
PY

echo "=== 1/3: full attention H512/GQA8 ==="
python benchmarks/run_gemma4_e2b_h512_attention_gate.py \
  --context 2175 \
  --table-blocks 144 \
  --iterations 100 \
  --repeats 7 \
  --out-json "$OUT_ROOT/h512_attention.json"

echo "=== 2/3: dense attention -> MLP bridge ==="
python benchmarks/run_gemma4_e2b_dense_attn_mlp_bridge_gate.py \
  --iterations 500 \
  --repeats 9 \
  --out-json "$OUT_ROOT/dense_attn_mlp_bridge.json"

echo "=== native CUDA rebuild for the cuBLASLt gate ==="
TORCH_LIB_DIR="$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MEGAGEMM_BUILD_ONLY_RMSNORM_CUDA=1 \
MAX_JOBS="${MAX_JOBS:-2}" \
python setup.py build_ext --inplace

python - <<'PY'
import rmsnorm_cuda_ops
from megagemm.kernels.mlp_prefill_native import HAS_CUBLASLT_BF16_LINEAR

print("Native extension:", rmsnorm_cuda_ops.__file__)
assert HAS_CUBLASLT_BF16_LINEAR, "BF16 cuBLASLt binding is unavailable"
PY

echo "=== 3/3: cuBLASLt BF16 gate-up algorithm sweep ==="
python benchmarks/run_gemma4_e2b_cublaslt_gateup_sweep.py \
  --maximum-algorithms 32 \
  --iterations 200 \
  --repeats 7 \
  --out-json "$OUT_ROOT/cublaslt_gateup.json"

echo "Artifacts: $OUT_ROOT"
