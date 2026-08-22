#!/usr/bin/env bash
set -euo pipefail

# One fresh paid A100 run: download once, measure natural generation, then
# compare an identical continuation before installing and measuring vLLM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

HARNESS_REV="gemma4-long-context-ab-v37-restore-all-prefill-candidates"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/content/models/gemma-4-26b-a4b}"
DTYPE="${DTYPE:-bf16}"
CONTEXTS="${CONTEXTS:-2048}"
BATCH_SIZES="${BATCH_SIZES:-16}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2112}"
WARMUPS="${WARMUPS:-1}"
MEGAGEMM_MIN_WARMUPS="${MEGAGEMM_MIN_WARMUPS:-3}"
MEGAGEMM_MAX_WARMUPS="${MEGAGEMM_MAX_WARMUPS:-8}"
MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS="${MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS:-2}"
MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO="${MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO:-1.10}"
REPEATS="${REPEATS:-3}"
ROUTE_NORMALIZED_REPEATS="${ROUTE_NORMALIZED_REPEATS:-3}"
MAX_TOTAL_PREFILL_TOKENS="${MAX_TOTAL_PREFILL_TOKENS:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-0}"
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-0.24.0}"
PINNED_TRANSFORMERS_VERSION="${PINNED_TRANSFORMERS_VERSION:-5.13.1}"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-16}"
INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
RESUME_MEGAGEMM="${RESUME_MEGAGEMM:-0}"
RUN_LONG_DECODE_BURST_GATE_ONLY="${RUN_LONG_DECODE_BURST_GATE_ONLY:-0}"
LONG_DECODE_BURST_GATE_REPEATS="${LONG_DECODE_BURST_GATE_REPEATS:-2}"
MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS="${MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS:-32768}"
RUN_LONG_SORTED_PARTIAL_GATE="${RUN_LONG_SORTED_PARTIAL_GATE:-0}"
STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED="${STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED:-1}"
LONG_SORTED_PARTIAL_GATE_TIMEOUT_MIN="${LONG_SORTED_PARTIAL_GATE_TIMEOUT_MIN:-5}"
LONG_ASYNC_TILE_MAX_ASSIGNMENTS="${LONG_ASYNC_TILE_MAX_ASSIGNMENTS:-262144}"
RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE="${RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE:-0}"
STOP_IF_DECODE_ACTIVE_LIST_FRONTIER_REJECTED="${STOP_IF_DECODE_ACTIVE_LIST_FRONTIER_REJECTED:-1}"
DECODE_ACTIVE_LIST_FRONTIER_GATE_TIMEOUT_MIN="${DECODE_ACTIVE_LIST_FRONTIER_GATE_TIMEOUT_MIN:-5}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-10}"
DOWNLOAD_TIMEOUT_MIN="${DOWNLOAD_TIMEOUT_MIN:-12}"
BACKEND_TIMEOUT_MIN="${BACKEND_TIMEOUT_MIN:-15}"
TOTAL_TIMEOUT_MIN="${TOTAL_TIMEOUT_MIN:-45}"
RUN_ID="${RUN_ID:-gemma4_long_context_ab_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-bench_results/${RUN_ID}}"
MIN_BF16_AB_VRAM_MIB=71680

if [ "${RESUME_MEGAGEMM}" != "0" ]; then
  echo "RESUME_MEGAGEMM is unsupported: every paid Colab run must be fresh and self-contained" >&2
  echo "No package installation, checkpoint download, or GPU benchmark was started" >&2
  exit 2
fi
if [ "${RUN_LONG_DECODE_BURST_GATE_ONLY}" != "0" ] \
  && [ "${RUN_LONG_DECODE_BURST_GATE_ONLY}" != "1" ]; then
  echo "RUN_LONG_DECODE_BURST_GATE_ONLY must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${ROUTE_NORMALIZED_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROUTE_NORMALIZED_REPEATS must be a positive integer" >&2
  exit 2
fi
for flag_name in \
  RUN_LONG_SORTED_PARTIAL_GATE \
  STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED \
  RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE \
  STOP_IF_DECODE_ACTIVE_LIST_FRONTIER_REJECTED; do
  flag_value="${!flag_name}"
  if [ "${flag_value}" != "0" ] && [ "${flag_value}" != "1" ]; then
    echo "${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done
if ! [[ "${LONG_ASYNC_TILE_MAX_ASSIGNMENTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LONG_ASYNC_TILE_MAX_ASSIGNMENTS must be a positive integer" >&2
  exit 2
fi
if [ "${RUN_LONG_DECODE_BURST_GATE_ONLY}" != "1" ] \
  && [ "${PINNED_VLLM_VERSION}" = "0.25.1" ]; then
  echo "vLLM 0.25.1 is blocked for this checkpoint: its Gemma4 loader builds a 256-wide parameter for a 512-wide global-attention weight" >&2
  echo "Use the proven default vLLM 0.24.0; no paid work was started" >&2
  exit 2
fi
if [ "${RUN_LONG_DECODE_BURST_GATE_ONLY}" != "1" ] \
  && [ "${PINNED_VLLM_VERSION}" = "0.24.0" ] \
  && [ "${PINNED_TRANSFORMERS_VERSION}" != "5.13.1" ]; then
  echo "vLLM 0.24.0 requires the proven Transformers 5.13.1 stack for Gemma4 in this harness" >&2
  echo "No paid work was started" >&2
  exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

if [ "${MEGAGEMM_GEMMA4_LONG_AB_TIMEOUT_GUARD:-0}" != "1" ]; then
  export MEGAGEMM_GEMMA4_LONG_AB_TIMEOUT_GUARD=1
  export RUN_ID OUT_DIR RUN_LONG_DECODE_BURST_GATE_ONLY ROUTE_NORMALIZED_REPEATS
  export LONG_DECODE_BURST_GATE_REPEATS
  export RUN_LONG_SORTED_PARTIAL_GATE STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED
  export LONG_SORTED_PARTIAL_GATE_TIMEOUT_MIN LONG_ASYNC_TILE_MAX_ASSIGNMENTS
  export RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE
  export STOP_IF_DECODE_ACTIVE_LIST_FRONTIER_REJECTED
  export DECODE_ACTIVE_LIST_FRONTIER_GATE_TIMEOUT_MIN
  export MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS
  set +e
  timeout --foreground --signal=INT --kill-after=30s \
    "${TOTAL_TIMEOUT_MIN}m" bash "$0" "$@"
  rc=$?
  set -e
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "TOTAL TIMEOUT: long-context A/B exceeded ${TOTAL_TIMEOUT_MIN} minutes" >&2
  fi
  exit "${rc}"
fi

run_with_timeout() {
  local label="$1"
  local minutes="$2"
  shift 2
  echo "== ${label} (hard timeout ${minutes}m) =="
  set +e
  timeout --foreground --signal=INT --kill-after=30s "${minutes}m" "$@"
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
      echo "TIMEOUT: ${label} exceeded ${minutes} minute(s)" >&2
    else
      echo "FAILED: ${label} exited with status ${rc}" >&2
    fi
    return "${rc}"
  fi
}

max_csv_value() {
  local raw="$1"
  local maximum=0
  local value
  IFS=',' read -ra values <<< "${raw}"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -le 0 ]; then
      echo "Invalid positive integer in list: ${raw}" >&2
      return 2
    fi
    if [ "${value}" -gt "${maximum}" ]; then
      maximum="${value}"
    fi
  done
  echo "${maximum}"
}

MAX_CONTEXT="$(max_csv_value "${CONTEXTS}")"
MAX_BATCH="$(max_csv_value "${BATCH_SIZES}")"
if [ $((MAX_CONTEXT + MAX_TOKENS)) -gt "${MAX_SEQ_LEN}" ]; then
  echo "Invalid shape: context ${MAX_CONTEXT} + output ${MAX_TOKENS} exceeds max_seq_len ${MAX_SEQ_LEN}" >&2
  exit 2
fi
if [ $((MAX_CONTEXT * MAX_BATCH)) -gt "${MAX_TOTAL_PREFILL_TOKENS}" ]; then
  echo "Safety cap exceeded: B${MAX_BATCH} x C${MAX_CONTEXT} > ${MAX_TOTAL_PREFILL_TOKENS}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
PROMPT_TOKEN_IDS_JSON="${OUT_DIR}/long_prompt_token_ids.json"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/content/hf_cache_gemma4}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PINNED_VLLM_VERSION PINNED_TRANSFORMERS_VERSION
export MEGAGEMM_GEMMA4_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS="${MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS}"

# Long prefill remains eager because loaded-checkpoint graph replay was rejected
# on token correctness. Global padded-BMM and BM128 are retired; the stable
# skew-aware segmented tile and promoted async metadata stay fixed while v30
# gates contiguous FP32 partial writes with an inverse map for fixed-order reduce.
export MEGAGEMM_PREFILL_CUDA_GRAPHS=0
export MEGAGEMM_DISABLE_CUDA_RMSNORM=1
export MEGAGEMM_FLAT_DECODE=1
export MEGAGEMM_DECODE_CUDA_GRAPHS=1
export MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP=1
export MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE=1
export MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE=0
export MEGAGEMM_REUSE_REQUEST_SCHEDULER=1
export MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS=1
export MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL=1
export MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL=1
export MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND=0
export MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL=1
export MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL=1
export MEGAGEMM_GEMMA4_LONG_FULL_PREFILL=1
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_M=64
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_N=256
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_K=64
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_FUSED_GATE_BLOCK_N=128
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_WARPS=4
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_STAGES=3
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_COMPACT_ROUTE_PACK=0
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_ASYNC_TILES_MAX_ASSIGNMENTS="${LONG_ASYNC_TILE_MAX_ASSIGNMENTS}"
export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL=1
export MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL=0
export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL=1
export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MIN_SKEW=7.5
export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO=1.25
export MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV=1
export MEGAGEMM_TRITON_PREFILL_KV_SCATTER=1
export MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL=1
export MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK=1
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL=1
export MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL=1
export MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL=0
export MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE=1
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE=1
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=1
export MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE=1
export MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE=1
export MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE=1
export MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD=1
export MEGAGEMM_DECODE_GRAPH_TOKEN_BURST=0
export MEGAGEMM_MULTI_STEP_BURST_BATCH=8
export MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN=0
export MEGAGEMM_GEMMA4_B16_LONG_GRAPH_TOKEN_BURST_PROVEN=1
export MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX=0
export MEGAGEMM_GEMMA4_B16_FUSED_SOFTCAP_ARGMAX_PROVEN=1
export MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK=0
export MEGAGEMM_GEMMA4_B16_PERSISTENT_TOKEN_FEEDBACK_PROVEN=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS=4
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES=3
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES=3
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES=3
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE=8
export MEGAGEMM_PAGED_DECODE_GQA2=0
export MEGAGEMM_PAGED_DECODE_WARPS=0
export MEGAGEMM_PAGED_DECODE_WARPS_H256=8
export MEGAGEMM_PAGED_DECODE_WARPS_H512=4

echo "== GEMMA4 LONG-CONTEXT SAME-VM A/B =="
echo "harness_rev: ${HARNESS_REV}"
echo "run_id: ${RUN_ID}"
echo "out_dir: ${OUT_DIR}"
echo "model: ${MODEL}"
echo "contexts: ${CONTEXTS}"
echo "batch_sizes: ${BATCH_SIZES}"
echo "max_tokens: ${MAX_TOKENS}"
echo "max_seq_len: ${MAX_SEQ_LEN}"
echo "warmups: ${WARMUPS}"
echo "megagemm_adaptive_warmups: min=${MEGAGEMM_MIN_WARMUPS} max=${MEGAGEMM_MAX_WARMUPS} stable_pairs=${MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS} max_pair_ratio=${MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO}"
echo "repeats: ${REPEATS}"
echo "resume_megagemm: ${RESUME_MEGAGEMM}"
echo "long_decode_burst_gate_only: ${RUN_LONG_DECODE_BURST_GATE_ONLY}"
echo "token_reference: stable segmented warmup, then first measured repeat"
echo "warmup_gate: full token matrix + runtime topology + timing, bounded before vLLM"
echo "execution_policy: promoted one-chunk 32k prefill + async expert metadata + contiguous FP32 partial + burst8 decode"
echo "route_normalized_diagnostic: full LM head, one identical continuation token, repeats=${ROUTE_NORMALIZED_REPEATS}"
echo "decode_graph_owner: one compatible idle Scheduler persists across requests; shared cross-Scheduler cache remains off"
echo "long_decode_attention: sliding H256/GQA2=32 segments/tile64; full H512/GQA8=8 segments/tile16"
echo "decode_graph_burst: promoted only for A100 BF16 B16/C2048 with 64 outputs; exact runtime counters required"
echo "softcap_graph_evidence: capture counters survive same-mode graph replay; real mode changes reset them"
echo "persistent_graph_feedback: hard disabled; v21 promoted explicit GPU-to-GPU feedback between burst replays"
echo "deterministic_moe_contract: dominant expert hybrid or guarded segmented fallback must remain exact and stable"
echo "deterministic_prefill_max_batched_tokens: ${MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS}"
echo "prefill_chunk_policy: 32768 promoted by v28; the paid loaded checkpoint gate is retired"
echo "long_async_tiles: promoted max_assignments=${LONG_ASYNC_TILE_MAX_ASSIGNMENTS}"
echo "long_sorted_partial: promoted by v30 exact 1.093x gate; optional_recheck=${RUN_LONG_SORTED_PARTIAL_GATE}"
echo "decode_active_list_frontier: v32 promoted exact geomean=1.473x low-active=2.143x max-regression=0.51%; optional_recheck=${RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE}"
echo "prefill_stage_profile: one excluded post-measurement request, token-checked, no extra model load"
echo "long_sliding_prefill: promoted for A100 BF16 B8/B16 C2048; runtime hits required"
echo "long_full_prefill: promoted BN32/W8/S2 for A100 BF16 B8/B16 C2048 H512/GQA8; runtime hits required"
echo "long_attention_prepare: fused RMSNorm+RoPE+layouts promoted for sliding/full B8/B16 C2048; runtime hits required"
echo "long_routed_expert_prefill: B16/C2048 dominant expert hybrid, skew>=7.5x, light padding<=1.25x; exact segmented fallback"
echo "dominant_prefill_promotion: adaptive excluded segmented reference, exact token/topology gate, minimum real prefill speedup=1.02x"
echo "largest_prefill_tokens: $((MAX_CONTEXT * MAX_BATCH))"
if [ "${VLLM_MAX_NUM_BATCHED_TOKENS}" -gt 0 ]; then
  echo "vllm_max_num_batched_tokens: ${VLLM_MAX_NUM_BATCHED_TOKENS}"
else
echo "vllm_max_num_batched_tokens: $((MAX_CONTEXT * MAX_BATCH)) (full batch)"
fi
echo "vllm_stack: vllm=${PINNED_VLLM_VERSION} transformers=${PINNED_TRANSFORMERS_VERSION}"
echo "model_transport: Qwen3 snapshot_download path, workers=${HF_DOWNLOAD_WORKERS}"
echo "timeouts_min: total=${TOTAL_TIMEOUT_MIN} install=${INSTALL_TIMEOUT_MIN} download=${DOWNLOAD_TIMEOUT_MIN} backend=${BACKEND_TIMEOUT_MIN}"

echo
echo "== BASE GPU AND EARLY VRAM PREFLIGHT =="
nvidia-smi
GPU_MEMORY_MIB="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | sed -n '1p' \
    | tr -d '[:space:]'
)"
GPU_NAME="$(
  nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed -n '1p' \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)"
if ! [[ "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] \
  || [ "${GPU_MEMORY_MIB}" -lt "${MIN_BF16_AB_VRAM_MIB}" ]; then
  echo "GPU PREFLIGHT FAILED: ${GPU_NAME:-unknown} has ${GPU_MEMORY_MIB:-unknown} MiB" >&2
  echo "No package installation or checkpoint download was started" >&2
  exit 2
fi
echo "GPU PREFLIGHT OK: ${GPU_NAME}, ${GPU_MEMORY_MIB} MiB"

echo
echo "== INSTALL BASE RUNTIME =="
if [ "${INSTALL_RUNTIME}" = "1" ]; then
  run_with_timeout "INSTALL BASE RUNTIME" "${INSTALL_TIMEOUT_MIN}" \
    python -m pip install -q -U uv "transformers>=4.57" huggingface_hub hf_xet safetensors accelerate sentencepiece
else
  echo "Skipping base runtime installation"
fi

refresh_cuda_python_libs() {
  CUDA_PY_LIB_DIRS="$(python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
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
}

refresh_cuda_python_libs
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after base runtime installation")
print("gpu", torch.cuda.get_device_name(0), "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

if [ "${RUN_LONG_SORTED_PARTIAL_GATE}" = "1" ]; then
  echo
  echo "== CHECKPOINT-FREE LONG SORTED-PARTIAL GATE =="
  echo "No model download or vLLM installation has started"
  SORTED_PARTIAL_GATE_JSON="${OUT_DIR}/gemma4_long_sorted_partial_a100.json"
  run_with_timeout "LONG SORTED-PARTIAL GATE" "${LONG_SORTED_PARTIAL_GATE_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_long_skew_segmented_prefill_microbench.py \
      --minimum-speedup 1.01 \
      --minimum-profile-speedup 1.0 \
      --out-json "${SORTED_PARTIAL_GATE_JSON}"
  SORTED_PARTIAL_GATE_DECISION="$(
    env GATE_JSON="${SORTED_PARTIAL_GATE_JSON}" python -c \
      'import json, os; print(json.load(open(os.environ["GATE_JSON"], encoding="utf-8"))["decision"])'
  )"
  if [ "${SORTED_PARTIAL_GATE_DECISION}" = "APPLY" ]; then
    export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL=1
    echo "LONG SORTED-PARTIAL GATE: APPLY (FP32 remains unchanged)"
  else
    export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL=0
    echo "LONG SORTED-PARTIAL GATE: ${SORTED_PARTIAL_GATE_DECISION}"
    if [ "${STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED}" = "1" ]; then
      echo "Stopping before checkpoint download and vLLM installation to protect paid GPU time"
      exit 0
    fi
  fi
else
  echo "Skipping retired sorted-partial gate; using the exact v30 promotion"
fi

if [ "${RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE}" = "1" ]; then
  echo
  echo "== CHECKPOINT-FREE B16 DECODE ACTIVE-LIST FRONTIER GATE =="
  echo "No model download or vLLM installation has started"
  ACTIVE_LIST_GATE_JSON="${OUT_DIR}/gemma4_active_list_frontier_a100.json"
  run_with_timeout \
    "B16 DECODE ACTIVE-LIST FRONTIER GATE" \
    "${DECODE_ACTIVE_LIST_FRONTIER_GATE_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_active_list_early_exit_microbench.py \
      --warmup 4 \
      --iterations 100 \
      --repeats 7 \
      --active-expert-profiles 8,16,32,64,90,128 \
      --minimum-speedup 1.02 \
      --minimum-low-active-speedup 1.10 \
      --maximum-regression-ratio 1.02 \
      --out-json "${ACTIVE_LIST_GATE_JSON}"
  ACTIVE_LIST_GATE_DECISION="$(
    env GATE_JSON="${ACTIVE_LIST_GATE_JSON}" python -c \
      'import json, os; print(json.load(open(os.environ["GATE_JSON"], encoding="utf-8"))["decision"])'
  )"
  if [ "${ACTIVE_LIST_GATE_DECISION}" = "APPLY" ]; then
    export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST=1
    export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=1
    echo "B16 DECODE ACTIVE-LIST FRONTIER GATE: APPLY"
    echo "The full A/B will require all 30 decode layers to report active-list early-exit"
  else
    export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST=0
    export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=0
    echo "B16 DECODE ACTIVE-LIST FRONTIER GATE: ${ACTIVE_LIST_GATE_DECISION}"
    if [ "${STOP_IF_DECODE_ACTIVE_LIST_FRONTIER_REJECTED}" = "1" ]; then
      echo "Stopping before checkpoint download and vLLM installation to protect paid GPU time"
      exit 0
    fi
  fi
else
  echo "Skipping retired active-list frontier gate; using the exact v32 promotion"
fi

echo
echo "== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="
if [ -d "${MODEL}" ]; then
  MODEL_PATH="${MODEL}"
else
  run_with_timeout "DOWNLOAD MODEL" "${DOWNLOAD_TIMEOUT_MIN}" \
    env MODEL_REPO="${MODEL}" MODEL_DIR="${LOCAL_MODEL_DIR}" WORKERS="${HF_DOWNLOAD_WORKERS}" \
    python -c $'import json, os\nfrom pathlib import Path\nfrom huggingface_hub import snapshot_download\nrepo = os.environ["MODEL_REPO"]\nroot = Path(os.environ["MODEL_DIR"])\nworkers = int(os.environ["WORKERS"])\nroot.mkdir(parents=True, exist_ok=True)\nindex_path = root / "model.safetensors.index.json"\ndef complete():\n    if not index_path.is_file():\n        return False\n    index = json.loads(index_path.read_text(encoding="utf-8"))\n    shards = sorted(set(index.get("weight_map", {}).values()))\n    return bool(shards) and all((root / name).is_file() for name in shards)\nif complete():\n    print(f"Using complete local snapshot {root}")\nelse:\n    print(f"Downloading {repo} to {root} with {workers} workers")\n    snapshot_download(repo_id=repo, local_dir=str(root), max_workers=workers)\nif not complete():\n    raise SystemExit(f"Checkpoint verification failed under {root}")\nindex = json.loads(index_path.read_text(encoding="utf-8"))\nshards = sorted(set(index["weight_map"].values()))\nprint(f"Verified {len(shards)} safetensors shard(s) under {root}")'
  MODEL_PATH="${LOCAL_MODEL_DIR}"
fi

if [ "${RUN_LONG_DECODE_BURST_GATE_ONLY}" = "1" ]; then
  echo
  echo "== RUN LOADED-CHECKPOINT LONG DECODE GPU-FEEDBACK BURST GATE =="
  echo "vLLM installation and full backend sweeps are disabled for this gate"
  run_with_timeout "LONG DECODE GPU-FEEDBACK BURST GATE" "${BACKEND_TIMEOUT_MIN}" \
    env MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL=0 \
    python benchmarks/run_gemma4_long_decode_burst_gate.py \
      --model "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --batch-size 16 \
      --context 2048 \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --burst-size 8 \
      --repeats "${LONG_DECODE_BURST_GATE_REPEATS}" \
      --prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}" \
      --out-json "${OUT_DIR}/long_decode_burst_gate.json"
  exit 0
fi

COMMON_ARGS=(
  --model "${MODEL_PATH}"
  --dtype "${DTYPE}"
  --contexts "${CONTEXTS}"
  --batch-sizes "${BATCH_SIZES}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --max-tokens "${MAX_TOKENS}"
  --warmups "${WARMUPS}"
  --megagemm-min-warmups "${MEGAGEMM_MIN_WARMUPS}"
  --megagemm-max-warmups "${MEGAGEMM_MAX_WARMUPS}"
  --megagemm-required-stable-warmup-pairs "${MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS}"
  --megagemm-warmup-max-last-pair-ratio "${MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO}"
  --repeats "${REPEATS}"
  --route-normalized-diagnostic
  --route-normalized-repeats "${ROUTE_NORMALIZED_REPEATS}"
  --require-request-scheduler-reuse
  --max-total-prefill-tokens "${MAX_TOTAL_PREFILL_TOKENS}"
  --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}"
)
echo
echo "== RUN MEGAGEMM LONG-CONTEXT SWEEP =="
run_with_timeout "MEGAGEMM LONG-CONTEXT SWEEP" "${BACKEND_TIMEOUT_MIN}" \
  python benchmarks/run_gemma4_long_context_vs_vllm.py \
    --backend megagemm \
    "${COMMON_ARGS[@]}" \
    --megagemm-determinism-auto-fallback \
    --profile-prefill-stages \
    --out-json "${OUT_DIR}/megagemm.json"

echo
echo "== INSTALL vLLM AFTER MEGAGEMM COMPLETES =="
if [ "${INSTALL_RUNTIME}" = "1" ]; then
  run_with_timeout "INSTALL vLLM" "${INSTALL_TIMEOUT_MIN}" \
    python -m uv pip install --system --reinstall \
      "vllm==${PINNED_VLLM_VERSION}" \
      "transformers==${PINNED_TRANSFORMERS_VERSION}" \
      --torch-backend=cu129
else
  echo "Skipping vLLM installation"
fi
refresh_cuda_python_libs
python - <<'PY'
import torch
import os
import transformers
import vllm
assert vllm.__version__ == os.environ["PINNED_VLLM_VERSION"], (vllm.__version__, os.environ["PINNED_VLLM_VERSION"])
assert transformers.__version__ == os.environ["PINNED_TRANSFORMERS_VERSION"], (transformers.__version__, os.environ["PINNED_TRANSFORMERS_VERSION"])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__, "import ok")
print("transformers", transformers.__version__, "compatibility pin ok")
PY

echo
echo "== RUN vLLM LONG-CONTEXT SWEEP =="
run_with_timeout "vLLM LONG-CONTEXT SWEEP" "${BACKEND_TIMEOUT_MIN}" \
  python benchmarks/run_gemma4_long_context_vs_vllm.py \
    --backend vllm \
    "${COMMON_ARGS[@]}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --out-json "${OUT_DIR}/vllm.json"

echo
python benchmarks/run_gemma4_long_context_vs_vllm.py \
  --backend compare \
  --megagemm-json "${OUT_DIR}/megagemm.json" \
  --vllm-json "${OUT_DIR}/vllm.json" \
  --out-json "${OUT_DIR}/comparison.json"
