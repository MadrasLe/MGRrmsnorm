#!/usr/bin/env bash
set -euo pipefail

cd /content/drive/MyDrive/MGRrmsnorm

python -m pip install -q huggingface_hub hf_xet safetensors transformers sentencepiece

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/content/hf_cache_gemma4}"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MEGAGEMM_FLAT_DECODE=1
export MEGAGEMM_DISABLE_CUDA_RMSNORM=1
export MEGAGEMM_QWEN3_MOE_GROUPED_DECODE=1
export MEGAGEMM_QWEN3_MOE_GROUPED_DEBUG=1
export MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE=1
export MEGAGEMM_DECODE_TIMING=0
export MEGAGEMM_DECODE_CUDA_GRAPHS=0

python benchmarks/run_gemma4_moe_flat_benchmark.py \
  --model google/gemma-4-26B-A4B-it \
  --dtype bf16 \
  --max-seq-len 2048 \
  --warmup-tokens 16 \
  --max-new-tokens 64 \
  --graph-ab
