#!/usr/bin/env bash
set -euo pipefail

# Fresh-VM, checkpoint-free comparison against the exact vLLM long-decode core.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-10}"
BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"
TOTAL_TIMEOUT_MIN="${TOTAL_TIMEOUT_MIN:-15}"
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-0.24.0}"
PINNED_TRANSFORMERS_VERSION="${PINNED_TRANSFORMERS_VERSION:-5.13.1}"
OUT_JSON="${OUT_JSON:-bench_results/gemma4_long_decode_vllm_attention_parity_a100.json}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

if [ "${MEGAGEMM_GEMMA4_LONG_ATTN_PARITY_TIMEOUT_GUARD:-0}" != "1" ]; then
  export MEGAGEMM_GEMMA4_LONG_ATTN_PARITY_TIMEOUT_GUARD=1
  set +e
  timeout --foreground --signal=INT --kill-after=30s \
    "${TOTAL_TIMEOUT_MIN}m" bash "$0" "$@"
  rc=$?
  set -e
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "TOTAL TIMEOUT: long attention parity gate exceeded ${TOTAL_TIMEOUT_MIN} minutes" >&2
  fi
  exit "${rc}"
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PINNED_VLLM_VERSION PINNED_TRANSFORMERS_VERSION

echo "== GEMMA4 B16 LONG ATTENTION vLLM PARITY GATE =="
echo "harness_rev: gemma4-vllm-attention-parity-v2-long-current"
echo "fresh_vm: supported"
echo "model_download: disabled"
echo "huggingface_download: disabled"
echo "shape: batch=16 context=2111 sliding_window=1024"
echo "megagemm: paid grouped-segmented baseline"
echo "vllm: ${PINNED_VLLM_VERSION} Triton attention core only"
echo "timeouts_min: total=${TOTAL_TIMEOUT_MIN} install=${INSTALL_TIMEOUT_MIN} benchmark=${BENCH_TIMEOUT_MIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "${INSTALL_RUNTIME}" = "1" ]; then
  timeout --foreground --signal=INT --kill-after=30s "${INSTALL_TIMEOUT_MIN}m" \
    python -m pip install -q -U uv
  timeout --foreground --signal=INT --kill-after=30s "${INSTALL_TIMEOUT_MIN}m" \
    python -m uv pip install --system --reinstall \
      "vllm==${PINNED_VLLM_VERSION}" \
      "transformers==${PINNED_TRANSFORMERS_VERSION}" \
      --torch-backend=cu129
fi

CUDA_PY_LIB_DIRS="$(python - <<'PY'
import site
import sysconfig
from pathlib import Path

roots = []
for raw in list(site.getsitepackages()) + [site.getusersitepackages()]:
    path = Path(raw)
    if path.exists() and path not in roots:
        roots.append(path)
for raw in sysconfig.get_paths().values():
    if raw:
        path = Path(raw)
        if path.exists() and path not in roots:
            roots.append(path)

dirs = []
for root in roots:
    for pattern in ("nvidia/**/lib", "nvidia/**/lib64"):
        for path in root.glob(pattern):
            text = str(path)
            if text not in dirs:
                dirs.append(text)
print(":".join(dirs))
PY
)"
if [ -n "${CUDA_PY_LIB_DIRS}" ]; then
  export LD_LIBRARY_PATH="${CUDA_PY_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

python - <<'PY'
import os
import torch
import transformers
import triton
import vllm

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
if "a100" not in name.lower() or vram_gb < 70.0:
    raise SystemExit(f"A100 80GB required, found {name} ({vram_gb:.2f}GB)")
assert vllm.__version__ == os.environ["PINNED_VLLM_VERSION"]
assert transformers.__version__ == os.environ["PINNED_TRANSFORMERS_VERSION"]
print("python runtime: torch", torch.__version__, "triton", triton.__version__)
print("vllm:", vllm.__version__, "transformers:", transformers.__version__)
print("gpu:", name)
PY

mkdir -p "$(dirname "${OUT_JSON}")"
timeout --foreground --signal=INT --kill-after=30s "${BENCH_TIMEOUT_MIN}m" \
  python benchmarks/run_gemma4_vllm_attention_parity_microbench.py \
    --context 2111 \
    --table-blocks 132 \
    --warmup 5 \
    --iterations 100 \
    --repeats 7 \
    --minimum-speedup 1.02 \
    --out-json "${OUT_JSON}"
