#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGAGEMM_DISABLE_CUDA_RMSNORM=1

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python benchmarks/run_gemma4_hot_layer_microbench.py \
  --dtype bf16 \
  --warmup 8 \
  --graph-iterations 100 \
  --prefill-iterations 5 \
  --repeats 5 \
  --out-json bench_results/gemma4_hot_layer_microbench_a100.json
