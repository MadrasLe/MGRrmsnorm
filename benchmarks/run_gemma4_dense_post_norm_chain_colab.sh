#!/usr/bin/env bash
set -euo pipefail

# Same-session MegaGemm-only A/B. This harness deliberately has no vLLM phase
# and does not install or mutate any comparison backend.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
HARDWARE_LABEL="${HARDWARE_LABEL:-1xl4}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-bench_results/gemma4_dense_post_norm_ab}"
BASE_ID="gemma4_dense_tail_baseline_${STAMP}"
CANDIDATE_ID="gemma4_dense_tail_candidate_${STAMP}"

cd "$REPO"

# Python uses benchmarks/ as sys.path[0] when these files are executed by
# pathname.  Make the checkout importable in a fresh Colab runtime without
# requiring an editable install or compiling optional native extensions.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate foi desenhado para uma NVIDIA L4."
  exit 2
}

COMMON_ARGS=(
  --model "$MODEL"
  --variants megagemm-bf16
  --batch-sizes 1,8
  --prompt-tokens 128,512,2048
  --max-new-tokens 128
  --repeats 5
  --warmup 3
  --max-seq-len 2304
  --max-batch-size 8
  --hardware-label "$HARDWARE_LABEL"
  --out-dir "$OUT_ROOT"
)

case "${MODEL,,}" in
  *e2b*) HIDDEN_SIZE=1536 ;;
  *e4b*) HIDDEN_SIZE=2560 ;;
  *)
    echo "ERRO: MODEL deve ser um checkpoint Gemma 4 E2B ou E4B."
    exit 2
    ;;
esac

echo "=== kernel preflight: H=$HIDDEN_SIZE, BF16, rows 1 e 8 ==="
python benchmarks/run_gemma4_dense_post_norm_chain_preflight.py \
  --hidden-size "$HIDDEN_SIZE" \
  --min-speedup 1.0

echo "=== baseline: dense post-norm chain OFF ==="
MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE=0 \
python benchmarks/run_publication_gpu_suite.py \
  "${COMMON_ARGS[@]}" \
  --run-id "$BASE_ID"

echo "=== candidate: dense post-norm chain ON ==="
MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE=1 \
python benchmarks/run_publication_gpu_suite.py \
  "${COMMON_ARGS[@]}" \
  --run-id "$CANDIDATE_ID"

python benchmarks/compare_inference_summaries.py \
  --left "$OUT_ROOT/$CANDIDATE_ID/*megagemm_summary.json" \
  --right "$OUT_ROOT/$BASE_ID/*megagemm_summary.json" \
  --left-name candidate \
  --right-name baseline \
  --csv "$OUT_ROOT/gemma4_dense_tail_ab_${STAMP}.csv"

echo "BASELINE_RUN=$OUT_ROOT/$BASE_ID"
echo "CANDIDATE_RUN=$OUT_ROOT/$CANDIDATE_ID"
