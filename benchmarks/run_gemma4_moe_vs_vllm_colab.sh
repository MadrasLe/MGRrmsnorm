#!/usr/bin/env bash
set -euo pipefail

# One-command fresh-VM A/B. Both engines use the same local BF16 checkpoint,
# exact pretokenized prompt IDs, GPU, output length, and warm-before-measure protocol.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
DTYPE="${DTYPE:-bf16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REPEATS="${REPEATS:-3}"
BATCH_SIZES="${BATCH_SIZES:-1}"
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-${VLLM_VERSION:-0.24.0}}"
PINNED_TRANSFORMERS_VERSION="${PINNED_TRANSFORMERS_VERSION:-5.13.1}"
export PINNED_VLLM_VERSION PINNED_TRANSFORMERS_VERSION
unset VLLM_VERSION
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-16}"
RUN_BACKENDS="${RUN_BACKENDS:-both}"
REUSE_MEGAGEMM_RESULT="${REUSE_MEGAGEMM_RESULT:-0}"
INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"
PROFILE_BREAKDOWN="${PROFILE_BREAKDOWN:-0}"
PREFILL_TIMING="${PREFILL_TIMING:-0}"
RUN_KERNEL_PREFLIGHT="${RUN_KERNEL_PREFLIGHT:-0}"
PREFILL_FINITE_TRACE_ONLY="${PREFILL_FINITE_TRACE_ONLY:-0}"
RUN_ROUTER_COMPACT_PACK_GATE="${RUN_ROUTER_COMPACT_PACK_GATE:-0}"
STOP_IF_ROUTER_COMPACT_PACK_REJECTED="${STOP_IF_ROUTER_COMPACT_PACK_REJECTED:-0}"
RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE="${RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE:-0}"
# Do not spend a second runtime install unless the new exact decode candidate wins.
STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION="${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION:-0}"
# v85 measured MegaGemm's routed expert core faster than vLLM, and v88 proved
# the grouped segmented attention port. Both parity gates are now opt-in.
RUN_VLLM_MOE_PARITY_GATE="${RUN_VLLM_MOE_PARITY_GATE:-0}"
RUN_VLLM_ATTENTION_PARITY_GATE="${RUN_VLLM_ATTENTION_PARITY_GATE:-0}"
VLLM_MOE_PARITY_ROWS="${VLLM_MOE_PARITY_ROWS:-16}"
case "${VLLM_MOE_PARITY_ROWS}" in
  16|400)
    ;;
  *)
    echo "Invalid VLLM_MOE_PARITY_ROWS=${VLLM_MOE_PARITY_ROWS}; use 16 or 400" >&2
    exit 2
    ;;
esac
PREFLIGHT_TIMEOUT_MIN="${PREFLIGHT_TIMEOUT_MIN:-3}"
GRAPH_PREFLIGHT_TIMEOUT_MIN="${GRAPH_PREFLIGHT_TIMEOUT_MIN:-5}"
INSTALL_TIMEOUT_MIN="${INSTALL_TIMEOUT_MIN:-10}"
BACKEND_TIMEOUT_MIN="${BACKEND_TIMEOUT_MIN:-10}"
TOTAL_TIMEOUT_MIN="${TOTAL_TIMEOUT_MIN:-30}"
RUN_ID="${RUN_ID:-gemma4_moe_ab_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-bench_results/${RUN_ID}}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/content/models/gemma-4-26b-a4b}"
MIN_BF16_AB_VRAM_MIB=71680

PARITY_GATE_REQUESTED=0
if [ "${RUN_VLLM_MOE_PARITY_GATE}" = "1" ] \
  || [ "${RUN_VLLM_ATTENTION_PARITY_GATE}" = "1" ]; then
  PARITY_GATE_REQUESTED=1
fi

PARITY_ONLY_MODE=0
if [[ "${RUN_BACKENDS}" == "both" || "${RUN_BACKENDS}" == "all" ]] \
  && [[ ",${BATCH_SIZES}," == *",16,"* ]] \
  && [ "${PARITY_GATE_REQUESTED}" = "1" ] \
  && [ "${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION}" = "1" ]; then
  PARITY_ONLY_MODE=1
fi

# Colab's Drive FUSE can disconnect while pip replaces large CUDA packages.
# For the no-checkpoint parity gate, stage only the source needed by the run
# under /content before any installation, then execute entirely off Drive.
if [ "${MEGAGEMM_GEMMA4_LOCAL_STAGE_GUARD:-0}" != "1" ] \
  && [[ "${PWD}" == /content/drive/* ]] \
  && [ "${PARITY_ONLY_MODE}" = "1" ]; then
  LOCAL_STAGE_DIR="$(mktemp -d /content/megagemm-gemma4-parity.XXXXXX)"
  echo "Staging parity source off Google Drive: ${LOCAL_STAGE_DIR}"
  mkdir -p "${LOCAL_STAGE_DIR}"
  cp -a "${PWD}/megagemm" "${LOCAL_STAGE_DIR}/megagemm"
  cp -a "${PWD}/benchmarks" "${LOCAL_STAGE_DIR}/benchmarks"
  export MEGAGEMM_GEMMA4_LOCAL_STAGE_GUARD=1
  export RUN_ID
  export OUT_DIR="/content/bench_results/${RUN_ID}"
  cd "${LOCAL_STAGE_DIR}"
  exec bash benchmarks/run_gemma4_moe_vs_vllm_colab.sh "$@"
fi

if [ "${BATCH_SIZES}" = "1" ]; then
  BATCH_MODE=0
  HARNESS_REV="gemma4-ab-qwen-snapshot-v130-pinned-vllm-transformers-stack"
else
  BATCH_MODE=1
  HARNESS_REV="gemma4-ab-qwen-snapshot-v130-pinned-vllm-transformers-stack"
fi
MEGAGEMM_BATCH_DETERMINISTIC="${MEGAGEMM_BATCH_DETERMINISTIC:-1}"
if [ "${BATCH_MODE}" = "1" ] && [ "${MEGAGEMM_BATCH_DETERMINISTIC}" = "1" ]; then
  # This must be present before Python imports torch or creates a cuBLAS handle.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  THROUGHPUT_MODE="proven stable deterministic baseline"
elif [ "${BATCH_MODE}" = "1" ]; then
  # Fast mode remains available as an explicit audit, not the paid-run default.
  unset CUBLAS_WORKSPACE_CONFIG
  THROUGHPUT_MODE="explicit fast-mode audit"
else
  unset CUBLAS_WORKSPACE_CONFIG
  THROUGHPUT_MODE="single-request standard mode"
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required; refusing an unbounded paid GPU run" >&2
  exit 2
fi

if [ "${MEGAGEMM_GEMMA4_AB_TIMEOUT_GUARD:-0}" != "1" ]; then
  export MEGAGEMM_GEMMA4_AB_TIMEOUT_GUARD=1
  export RUN_ID OUT_DIR
  set +e
  timeout --foreground --signal=INT --kill-after=30s \
    "${TOTAL_TIMEOUT_MIN}m" bash "$0" "$@"
  rc=$?
  set -e
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "TOTAL TIMEOUT: Gemma4 A/B exceeded ${TOTAL_TIMEOUT_MIN} minutes; terminated" >&2
  fi
  exit "${rc}"
fi

mkdir -p "${OUT_DIR}"
PROMPT_TOKEN_IDS_JSON="${OUT_DIR}/prompt_token_ids.json"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export BATCH_SIZES
export HF_HOME="${HF_HOME:-/content/hf_cache_gemma4}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Loaded-checkpoint runs v99, v105, and v106 reproduced the same B16 divergence
# after a prefill-graph replay even though the isolated synthetic gates passed.
# Keep the proven eager prefill as the paid-run default; graph prefill remains an
# explicit diagnostic opt-in guarded by the loaded-checkpoint token probe.
PREFILL_CUDA_GRAPH_DEFAULT=0
export MEGAGEMM_PREFILL_CUDA_GRAPHS="${MEGAGEMM_PREFILL_CUDA_GRAPHS:-${PREFILL_CUDA_GRAPH_DEFAULT}}"
if [ "${MEGAGEMM_PREFILL_CUDA_GRAPHS}" = "1" ]; then
  # v102 captured the fused frontend but faulted on its first full-model replay.
  # Keep eager and graph on the last proven-exact v98 attention frontend.
  export MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL=0
  export MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL=0
  export MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND=0
else
  export MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL="${MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL:-1}"
  export MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL="${MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL:-1}"
  export MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND=0
fi
# A100 B16x25: implicit causality won 10.16x on the 25 sliding layers but
# regressed 2.65% on the 5 full layers. Runtime selection is sliding-only.
export MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL="${MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL:-1}"
export MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV="${MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV:-1}"
# v121 attributed 64.67 ms of the 127.03 ms profiled prefill body to the two
# generic advanced-indexing K/V writes. Use one dedicated Triton scatter.
export MEGAGEMM_TRITON_PREFILL_KV_SCATTER="${MEGAGEMM_TRITON_PREFILL_KV_SCATTER:-1}"
# A100 BF16 B16x25: shared MLP on a side stream overlapped the routed experts
# by 2.1%, recovering 0.793 ms across the 30 MoE layers.
export MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL="${MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL:-1}"
export MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS="${MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS:-4096}"
export MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS="${MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS:-4096}"
export MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS="${MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS:-512}"
export MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK="${MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK:-1}"
# v124 keeps the exact two-kernel router bridge and makes vLLM warmup timing
# diagnostic once the bounded warmup budget has produced identical tokens.
# both it and the 400-row matrix router against the proven baseline in-process.
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL="${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL:-1}"
export MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL="${MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL:-1}"
# v72 measured the post-MoE tail at only 1.58%, below the 2% promotion gate.
export MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS="${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS:-4}"
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES="${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES:-3}"
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES="${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES:-3}"
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES="${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES:-3}"
# Paid A100 results rejected the prior alternatives. Keep these locks non-overridable so
# an inherited shell variable cannot silently change the baseline again.
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS=0
export MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE="${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE:-1}"
# v110 proved this bridge byte-exact, 1.276x faster in its direct gate, and
# exercised it in all 30 B16 layers. It is baseline now, not another paid gate.
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE=1
# v127 proved the one-kernel replacement byte-exact, alias-safe, stable, and
# 1.329x faster (10.629 -> 7.997 us/layer), then exercised it for all 60
# loaded-model decode hits. It is baseline now; the gate remains opt-in only.
export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=1
# v111 measured the persistent router+pack candidate 12.67x slower. Retire it
# from paid runs and preserve the exact 9.85 us two-stage baseline.
export MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE=0
# Both isolated kernels won their microbench, but their first combined full
# capture/replay lifecycle crashed before measurement. Keep the paid A/B on the
# last graph-stable implementation; the experimental code remains opt-in.
export MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE=0
export MEGAGEMM_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE=1
# v81 proved this exact for all 16x64 greedy tokens and 1.221% faster in full
# graph replay. Keep it as the non-negotiable baseline; do not pay to remeasure it.
export MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE=1
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID=0
export MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE=8
# v79 proved H512/W4 exact and 1.4 percent faster. v80 proved that explicitly
# regrouping QK still did not make H256/W4 exact, so that branch is retired.
export MEGAGEMM_PAGED_DECODE_GQA2=0
export MEGAGEMM_PAGED_DECODE_WARPS=0
export MEGAGEMM_PAGED_DECODE_WARPS_H256=8
export MEGAGEMM_PAGED_DECODE_WARPS_H512=4
export MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE=1
export MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD=1
export MEGAGEMM_DECODE_GRAPH_TOKEN_BURST=1
export MEGAGEMM_MULTI_STEP_BURST_BATCH=8
export MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN=1
# v119 proved the fused softcap/argmax plus persistent graph feedback path
# exact over both measured repeats, with 63/63 device-feedback steps and a
# stable 3.217 ms decode saving on A100 BF16 B16. Keep both features off until
# the scheduler gate consumes the PROVEN flags, so earlier correctness oracles
# remain on their established baseline.
export MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX=0
export MEGAGEMM_GEMMA4_B16_FUSED_SOFTCAP_ARGMAX_PROVEN=1
export MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK=0
export MEGAGEMM_GEMMA4_B16_PERSISTENT_TOKEN_FEEDBACK_PROVEN=1

echo "== RUN =="
echo "harness_rev: ${HARNESS_REV}"
echo "run_id: ${RUN_ID}"
echo "out_dir: ${OUT_DIR}"
echo "model: ${MODEL}"
echo "local_model_dir: ${LOCAL_MODEL_DIR}"
echo "dtype: ${DTYPE}"
echo "batch_sizes: ${BATCH_SIZES}"
echo "max_seq_len: ${MAX_SEQ_LEN}"
echo "max_tokens: ${MAX_TOKENS}"
echo "repeats: ${REPEATS}"
echo "run_backends: ${RUN_BACKENDS}"
echo "reuse_megagemm_result: ${REUSE_MEGAGEMM_RESULT}"
echo "vllm_version: ${PINNED_VLLM_VERSION}"
echo "transformers_version_for_vllm: ${PINNED_TRANSFORMERS_VERSION}"
echo "install_runtime: ${INSTALL_RUNTIME}"
echo "profile_breakdown: ${PROFILE_BREAKDOWN}"
echo "prefill_timing: ${PREFILL_TIMING}"
echo "run_kernel_preflight: ${RUN_KERNEL_PREFLIGHT}"
echo "prefill_finite_trace_only: ${PREFILL_FINITE_TRACE_ONLY}"
echo "router_bridge_gate: retired after exact v110 promotion"
echo "router_single_kernel_gate: retired after exact v127 promotion (opt-in audit=${RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE})"
echo "router_compact_pack_gate: retired after v111 (12.67x slower)"
echo "run_router_compact_pack_gate: ${RUN_ROUTER_COMPACT_PACK_GATE}"
echo "stop_if_router_compact_pack_rejected: ${STOP_IF_ROUTER_COMPACT_PACK_REJECTED}"
echo "stop_before_vllm_if_no_decode_promotion: ${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION}"
echo "run_vllm_moe_parity_gate: ${RUN_VLLM_MOE_PARITY_GATE}"
echo "vllm_moe_parity_rows: ${VLLM_MOE_PARITY_ROWS}"
echo "run_vllm_attention_parity_gate: ${RUN_VLLM_ATTENTION_PARITY_GATE}"
echo "b16_graph_preflight: conditional on prefill_cuda_graphs=1"
echo "prefill_cuda_graphs: ${MEGAGEMM_PREFILL_CUDA_GRAPHS}"
echo "prefill_graph_default: retired after v106 loaded-checkpoint token divergence"
echo "fused_qkv_prefill: ${MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL}"
echo "fused_attn_prepare_prefill: ${MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL}"
echo "implicit_causal_prefill: ${MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL}"
echo "vectorized_prefill_kv: ${MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV}"
echo "triton_prefill_kv_scatter: ${MEGAGEMM_TRITON_PREFILL_KV_SCATTER}"
echo "fused_attn_moe_bridge_prefill: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL}"
echo "fused_matrix_router_prefill: ${MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL}"
echo "single_accumulator_prefill_gate: retired after v76 (19 percent slower)"
echo "compact_decode_gate: v84 active-list branch retired after measuring 1.26 percent slower"
echo "attention_decode: v88 grouped segmented core promoted for B16 H256/GQA2 and H512/GQA8"
echo "decode_baseline: v81 fused post-MoE layer-scalar plus next-layer RMSNorm"
echo "decode_candidate_gate: retired after v88; grouped segmented attention is now the default"
echo "grouped_segmented_attention_default: ${MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE}"
echo "attn_moe_decode_bridge_default: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE}"
echo "attn_moe_router_decode_bridge_default: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE}"
echo "attn_moe_router_single_kernel_default: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE}"
echo "router_compact_pack_decode_candidate: retired (${MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE})"
echo "prompt_token_ids_json: ${PROMPT_TOKEN_IDS_JSON}"
echo "decode_stage_tuning: retired after v82 (0.156 percent microkernel gain)"
echo "fused_post_moe_norm_residual_prefill: ${MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL}"
echo "batch_deterministic: ${MEGAGEMM_BATCH_DETERMINISTIC}"
echo "throughput_mode: ${THROUGHPUT_MODE}"
echo "cublas_workspace_config: ${CUBLAS_WORKSPACE_CONFIG:-unset}"
echo "compact_active_list: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST}"
echo "compact_active_list_early_exit: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT}"
echo "compact_expert_grid_pack: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK}"
echo "compact_coalesced_weights: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS}"
echo "compact_num_warps: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS}"
echo "compact_num_stages: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES}"
echo "compact_gate_num_stages: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES}"
echo "compact_down_num_stages: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES}"
echo "compact_experts_per_program: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM}"
echo "compact_paired_gate_up_dot: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT}"
echo "compact_split_gate_up: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP}"
echo "compact_empty_expert_early_exit: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT}"
echo "compact_l2_grouped_grid: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID}"
echo "compact_l2_group_size: ${MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE}"
echo "paged_decode_gqa2_baseline: ${MEGAGEMM_PAGED_DECODE_GQA2}"
echo "paged_decode_warps_baseline: ${MEGAGEMM_PAGED_DECODE_WARPS}"
echo "paged_decode_warps_h256_baseline: ${MEGAGEMM_PAGED_DECODE_WARPS_H256}"
echo "paged_decode_warps_h512_baseline: ${MEGAGEMM_PAGED_DECODE_WARPS_H512}"
echo "batch_cublas_lm_head_default: ${MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD}"
echo "decode_graph_token_burst_default: ${MEGAGEMM_DECODE_GRAPH_TOKEN_BURST}"
echo "decode_graph_token_burst_size: ${MEGAGEMM_MULTI_STEP_BURST_BATCH}"
echo "decode_graph_token_burst_proven: ${MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN}"
echo "fused_softcap_argmax_candidate: ${MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX}"
echo "persistent_token_feedback_candidate: ${MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK}"
echo "decode_graph_kv_binding: canonical idle recycle + physical rebind rejection"
echo "fused_decode_router: ${MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE}"
echo "fused_attn_moe_bridge_decode: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE}"
echo "fused_attn_moe_router_bridge_decode: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE}"
echo "fused_attn_moe_router_single_kernel_decode: ${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE}"
echo "fused_post_moe_norm_residual: ${MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE}"
echo "fused_router_expert_input_norm: ${MEGAGEMM_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE}"
echo "fused_expert_reduce_post_moe: ${MEGAGEMM_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE}"
echo "fused_next_attn_norm_baseline: ${MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE}"
echo "timeouts_min: total=${TOTAL_TIMEOUT_MIN} preflight=${PREFLIGHT_TIMEOUT_MIN} graph_preflight=${GRAPH_PREFLIGHT_TIMEOUT_MIN} install=${INSTALL_TIMEOUT_MIN} backend=${BACKEND_TIMEOUT_MIN}"
echo "model_transport: Qwen3 snapshot_download path, workers=${HF_DOWNLOAD_WORKERS}"

RUN_MEGAGEMM=0
RUN_VLLM=0
case "${RUN_BACKENDS}" in
  both|all)
    RUN_MEGAGEMM=1
    RUN_VLLM=1
    ;;
  megagemm|mgx)
    RUN_MEGAGEMM=1
    ;;
  vllm)
    RUN_VLLM=1
    ;;
  *)
    echo "Invalid RUN_BACKENDS=${RUN_BACKENDS}; use both, megagemm, or vllm" >&2
    exit 2
    ;;
esac
COMPARISON_SCOPE="same_vm"
if [ "${REUSE_MEGAGEMM_RESULT}" = "1" ]; then
  if [ "${RUN_MEGAGEMM}" != "0" ] || [ "${RUN_VLLM}" != "1" ]; then
    echo "REUSE_MEGAGEMM_RESULT=1 requires RUN_BACKENDS=vllm" >&2
    exit 2
  fi
  OUT_DIR="${OUT_DIR}" BATCH_SIZES="${BATCH_SIZES}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUT_DIR"]) / "megagemm.json"
if not path.is_file():
    raise SystemExit(
        f"Persisted MegaGemm result is missing: {path}. "
        "No install, download, or GPU benchmark was started."
    )
result = json.loads(path.read_text(encoding="utf-8"))
if result.get("backend") != "megagemm" or "cases" not in result:
    raise SystemExit(f"Invalid persisted MegaGemm batch result: {path}")
requested = [item.strip() for item in os.environ["BATCH_SIZES"].split(",")]
missing = [batch for batch in requested if batch not in result["cases"]]
if missing:
    raise SystemExit(
        f"Persisted MegaGemm result lacks requested batches {missing}: {path}"
    )
if not result.get("prompt_contract"):
    raise SystemExit(f"Persisted MegaGemm result lacks prompt contract: {path}")
manifest_path = Path(os.environ["OUT_DIR"]) / "prompt_token_ids.json"
if not manifest_path.is_file():
    raise SystemExit(
        f"Persisted prompt token manifest is missing: {manifest_path}. "
        "No install, download, or GPU benchmark was started."
    )
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("contract") != result["prompt_contract"]:
    raise SystemExit(
        "Persisted prompt token manifest does not match MegaGemm result: "
        f"{manifest_path}"
    )
print(
    "PERSISTED_MEGAGEMM_RESULT_OK "
    + json.dumps(
        {
            "path": str(path),
            "prompt_manifest": str(manifest_path),
            "batches": requested,
            "prompt_sha256": result["prompt_contract"]["sha256"],
        },
        sort_keys=True,
    )
)
PY
  COMPARISON_SCOPE="persisted_megagemm_current_vllm"
elif [ "${RUN_MEGAGEMM}" = "0" ] && [ "${RUN_VLLM}" = "1" ]; then
  COMPARISON_SCOPE="vllm_only"
elif [ "${RUN_MEGAGEMM}" = "1" ] && [ "${RUN_VLLM}" = "0" ]; then
  COMPARISON_SCOPE="megagemm_only"
fi
export COMPARISON_SCOPE
if [ "${PREFILL_FINITE_TRACE_ONLY}" = "1" ]; then
  if [ "${BATCH_MODE}" != "1" ]; then
    echo "PREFILL_FINITE_TRACE_ONLY=1 requires batched mode" >&2
    exit 2
  fi
  RUN_MEGAGEMM=1
  RUN_VLLM=0
  echo "Trace-only mode: forcing MegaGemm-only and skipping all benchmark sweeps."
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
      echo "TIMEOUT: ${label} exceeded ${minutes} minute(s); process terminated" >&2
    else
      echo "FAILED: ${label} exited with status ${rc}" >&2
    fi
    return "${rc}"
  fi
}

echo
echo "== BASE GPU =="
nvidia-smi

if [ "${PARITY_ONLY_MODE}" != "1" ]; then
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
  if ! [[ "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]]; then
    echo "GPU PREFLIGHT FAILED: could not read total VRAM from nvidia-smi." >&2
    echo "No HTTPS preflight, package installation, or checkpoint download was started." >&2
    exit 2
  fi
  if [ "${GPU_MEMORY_MIB}" -lt "${MIN_BF16_AB_VRAM_MIB}" ]; then
    echo "GPU PREFLIGHT FAILED: ${GPU_NAME} has ${GPU_MEMORY_MIB} MiB." >&2
    echo "Gemma4-26B-A4B BF16 B16 requires an A100-SXM4-80GB-class GPU (>= ${MIN_BF16_AB_VRAM_MIB} MiB)." >&2
    echo "The 48.1 GiB checkpoint cannot fit safely on an A100 40GB with KV cache and workspaces." >&2
    echo "No HTTPS preflight, package installation, checkpoint download, or GPU benchmark was started." >&2
    exit 2
  fi
  echo "GPU PREFLIGHT OK: ${GPU_NAME}, ${GPU_MEMORY_MIB} MiB"
fi

echo
if [ "${PARITY_ONLY_MODE}" = "1" ]; then
  echo "== NO-CHECKPOINT PARITY MODE =="
  echo "Skipping model HTTPS preflight and checkpoint download."
elif [ -d "${MODEL}" ]; then
  echo "== PUBLIC MODEL HTTPS PREFLIGHT =="
  echo "Local model path supplied; skipping Hub HTTPS preflight"
else
  echo "== PUBLIC MODEL HTTPS PREFLIGHT =="
  env MODEL_REPO="${MODEL}" python - <<'PY'
import json
import os
import urllib.error
import urllib.request

repo_id = os.environ["MODEL_REPO"]
base = f"https://huggingface.co/{repo_id}/resolve/main"
headers = {"User-Agent": "megagemm-colab-preflight/2"}

for filename in ("config.json", "model.safetensors.index.json"):
    request = urllib.request.Request(f"{base}/{filename}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Public HTTPS preflight failed for {filename}: HTTP {exc.code}. "
            "No GPU benchmark or vLLM installation was started."
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"Public HTTPS preflight failed for {filename}: {type(exc).__name__}: {exc}. "
            "No GPU benchmark or vLLM installation was started."
        ) from exc
    json.loads(payload)
    print(f"HTTPS preflight OK: {filename} status={status} bytes={len(payload)}")
PY
fi

echo
echo "== INSTALL SAME-VM RUNTIME =="
if [ "${INSTALL_RUNTIME}" = "1" ]; then
  run_with_timeout "INSTALL BASE RUNTIME" "${INSTALL_TIMEOUT_MIN}" \
    python -m pip install -q -U uv "transformers>=4.57" huggingface_hub hf_xet safetensors accelerate sentencepiece
else
  echo "Skipping installation because INSTALL_RUNTIME=${INSTALL_RUNTIME}"
fi

refresh_cuda_python_libs() {
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
}

refresh_cuda_python_libs

python - <<'PY'
import huggingface_hub
import sys
import torch

print("python", sys.version)
print("huggingface_hub", huggingface_hub.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after dependency installation")
print("gpu", torch.cuda.get_device_name(0), "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
    raise SystemExit("This BF16 same-GPU A/B requires an 80GB-class GPU")

PY

python - <<'PY'
import os

import torch

from megagemm.engine.kv_cache import BlockManager
from megagemm.kernels.qwen3_moe import (
    qwen3_moe_grouped_runtime_config,
    qwen3_moe_prepare_segmented_prefill_graph_workspace,
)
from megagemm.models.llama import (
    _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS,
    _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN,
    _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS,
    _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS,
    _GEMMA4_BATCH_CUBLAS_LM_HEAD,
    _GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE,
    _GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE,
    _GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL,
    _GEMMA4_FUSED_MOE_ROUTER_PREFILL,
    _GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE,
)

cfg = qwen3_moe_grouped_runtime_config()
batch_sizes = [int(value) for value in os.environ["BATCH_SIZES"].split(",")]
required_assignments = max(batch_sizes) * 25 * 8
graph_workspace_bytes = required_assignments * 2816 * 4 * 30

# The paid B16 run reuses decode graphs across fresh Scheduler instances.  Prove
# before any checkpoint download that identical idle-to-idle requests recycle the
# exact same physical KV layout required by that shared graph.
block_manager = BlockManager(
    num_layers=1,
    num_blocks=128,
    block_size=16,
    num_kv_heads=1,
    head_dim=1,
    dtype=torch.float32,
    device="cpu",
)

def allocate_recycle_batch(blocks_per_sequence):
    tables = []
    for seq_id in range(16):
        block_manager.allocate_sequence(
            seq_id,
            num_tokens=int(blocks_per_sequence) * block_manager.block_size,
        )
        tables.append(tuple(block_manager.block_tables[seq_id]))
    for seq_id in range(16):
        block_manager.free_sequence(seq_id)
    return tuple(tables)

allocate_recycle_batch(6)
capture_tables = allocate_recycle_batch(2)
replay_tables = allocate_recycle_batch(2)
if capture_tables != replay_tables:
    raise SystemExit(
        "Shared decode graph KV block recycle is unstable; refusing the paid run: "
        f"capture={capture_tables[:2]} replay={replay_tables[:2]}"
    )
if block_manager.free_blocks != list(range(block_manager.num_blocks)):
    raise SystemExit("Idle KV free-list order was not restored canonically")
print(
    "decode graph KV recycle: OK",
    {"capture_head": capture_tables[:2], "idle_resets": block_manager._idle_free_order_resets},
)

if not callable(qwen3_moe_prepare_segmented_prefill_graph_workspace):
    raise SystemExit(
        "Shape-persistent segmented prefill graph workspace is missing; "
        "refusing the paid run"
    )
if not cfg.get("segmented_prefill_async_tiles"):
    raise SystemExit("Async compact MoE tile patch is not active; refusing the paid run")
limit = max(
    int(cfg.get("segmented_prefill_async_tiles_max_assignments") or 0),
    int(_GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS.get("async_tiles_max_assignments") or 0),
)
if limit < required_assignments:
    raise SystemExit(
        f"Async compact MoE tile limit {limit} is below required "
        f"{required_assignments} assignments"
    )
if not cfg.get("segmented_prefill_partial_reduce"):
    raise SystemExit("Atomic-free segmented partial reduction patch is not active")
partial_limit = int(cfg.get("segmented_prefill_partial_reduce_max_assignments") or 0)
if partial_limit < required_assignments:
    raise SystemExit(
        f"Partial reduction limit {partial_limit} is below required "
        f"{required_assignments} assignments"
    )
partial_cache_limit = int(
    cfg.get("segmented_prefill_partial_cache_max_assignments") or 0
)
if partial_cache_limit > 512:
    raise SystemExit(
        f"Large segmented partial buffers would persist per layer: cache limit "
        f"{partial_cache_limit} exceeds 512"
    )
fixed_route_pack = bool(
    _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS.get("fixed_route_pack")
)
if not fixed_route_pack:
    raise SystemExit("Gemma4 fixed route pack is not active; refusing the paid run")
compact_route_pack = bool(
    _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS.get("compact_route_pack")
)
if not compact_route_pack:
    raise SystemExit("Gemma4 compact route pack is not active; refusing the paid run")
if (
    int(_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN) != 400
    or int(_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS.get("block_m") or 0) != 32
):
    raise SystemExit("Gemma4 B16 M32 prefill baseline is not active; refusing the paid run")
if 16 in batch_sizes and not _GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL:
    raise SystemExit(
        "Gemma4 exact prefill attention-to-MoE/router bridge gate is unavailable"
    )
if 16 in batch_sizes and not _GEMMA4_FUSED_MOE_ROUTER_PREFILL:
    raise SystemExit(
        "Gemma4 400-row fused router gate is unavailable"
    )
if 16 in batch_sizes and not _GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE:
    raise SystemExit(
        "Gemma4 B16 attention-to-MoE/router bridge is not active; refusing the paid run"
    )
if 16 in batch_sizes and not _GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE:
    raise SystemExit(
        "Gemma4 B16 promoted single-kernel router bridge is not active; "
        "refusing the paid run"
    )
if cfg.get("segmented_prefill_single_accumulator"):
    raise SystemExit(
        "Rejected v76 single-accumulator prefill leaked into the baseline"
    )
if not cfg.get("expert_grouped_compact_expert_grid_pack"):
    raise SystemExit("Gemma4 B16 expert-grid decode pack is not active; refusing the paid run")
if cfg.get("expert_grouped_compact_active_list"):
    raise SystemExit("Active-list decode candidate leaked into the baseline")
if cfg.get("expert_grouped_compact_active_list_early_exit"):
    raise SystemExit(
        "Active-list early-exit candidate leaked into the baseline"
    )
if cfg.get("expert_grouped_compact_coalesced_weights"):
    raise SystemExit("Rejected coalesced-weight candidate leaked into the baseline")
compact_gate_stages = int(
    cfg.get("expert_grouped_compact_gate_num_stages") or 0
)
compact_down_stages = int(
    cfg.get("expert_grouped_compact_down_num_stages") or 0
)
if compact_gate_stages != 3 or compact_down_stages != 3:
    raise SystemExit(
        "Gemma4 split-pipeline gate must start from the proven 3/3 baseline, "
        f"got gate={compact_gate_stages} down={compact_down_stages}"
    )
experts_per_program = int(
    cfg.get("expert_grouped_compact_experts_per_program") or 0
)
if experts_per_program != 1:
    raise SystemExit(
        "Gemma4 persistent expert dispatch must start from the EPP=1 baseline"
    )
if cfg.get("expert_grouped_compact_paired_gate_up_dot"):
    raise SystemExit(
        "Gemma4 paired gate/up dot tuner must start from the disabled baseline"
    )
if cfg.get("expert_grouped_compact_split_gate_up"):
    raise SystemExit(
        "Gemma4 split gate/up tuner must start from the disabled baseline"
    )
if cfg.get("expert_grouped_compact_empty_expert_early_exit"):
    raise SystemExit(
        "Gemma4 empty-expert early-exit tuner must start from the disabled baseline"
    )
if cfg.get("expert_grouped_compact_l2_grouped_grid"):
    raise SystemExit(
        "Gemma4 L2-grouped grid tuner must start from the disabled baseline"
    )
l2_group_size = int(cfg.get("expert_grouped_compact_l2_group_size") or 0)
if l2_group_size != 8:
    raise SystemExit(
        f"Gemma4 L2-grouped grid must start with group size 8, got {l2_group_size}"
    )
if not _GEMMA4_BATCH_CUBLAS_LM_HEAD:
    raise SystemExit(
        "Gemma4 proven batch greedy-token LM-head default is not active"
    )
if not _GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE:
    raise SystemExit(
        "Gemma4 v81 exact next-attention RMSNorm baseline is not active"
    )
print(
    "segmented MoE patches: OK",
    {
        "async_max_assignments": limit,
        "partial_reduce_max_assignments": partial_limit,
        "partial_cache_max_assignments": partial_cache_limit,
        "required_assignments": required_assignments,
        "graph_workspace_bytes": graph_workspace_bytes,
        "fixed_route_pack": fixed_route_pack,
        "compact_route_pack": compact_route_pack,
        "single_accumulator_retired_after": "v76",
        "single_accumulator_baseline": bool(
            cfg.get("segmented_prefill_single_accumulator")
        ),
        "single_accumulator_group_size": int(
            cfg.get("segmented_prefill_group_size_m") or 0
        ),
        "expert_grid_decode_pack": bool(
            cfg.get("expert_grouped_compact_expert_grid_pack")
        ),
        "active_list_baseline": bool(
            cfg.get("expert_grouped_compact_active_list")
        ),
        "active_list_early_exit_baseline": bool(
            cfg.get("expert_grouped_compact_active_list_early_exit")
        ),
        "compact_gate_num_stages_baseline": compact_gate_stages,
        "compact_down_num_stages_baseline": compact_down_stages,
        "experts_per_program_baseline": experts_per_program,
        "paired_gate_up_dot_baseline": bool(
            cfg.get("expert_grouped_compact_paired_gate_up_dot")
        ),
        "split_gate_up_baseline": bool(
            cfg.get("expert_grouped_compact_split_gate_up")
        ),
        "empty_expert_early_exit_baseline": bool(
            cfg.get("expert_grouped_compact_empty_expert_early_exit")
        ),
        "l2_grouped_grid_baseline": bool(
            cfg.get("expert_grouped_compact_l2_grouped_grid")
        ),
        "l2_group_size": l2_group_size,
        "batch_cublas_lm_head_default": bool(_GEMMA4_BATCH_CUBLAS_LM_HEAD),
        "v81_next_attn_norm_baseline": bool(
            _GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE
        ),
    },
)
PY

if [ "${PREFILL_FINITE_TRACE_ONLY}" = "1" ]; then
  echo
  echo "== GEMMA4 B16 FINAL-NORM STRIDE PREFLIGHT (NO MODEL DOWNLOAD) =="
  run_with_timeout "GEMMA4 FINAL-NORM STRIDE PREFLIGHT" "${PREFLIGHT_TIMEOUT_MIN}" python - <<'PY'
import json

import torch

from megagemm.kernels.rmsnorm_triton import rmsnorm_triton

torch.manual_seed(117)
device = torch.device("cuda")
hidden = torch.randn((16, 25, 2816), device=device, dtype=torch.bfloat16)
last_token = hidden[:, -1:, :]
weight = torch.randn((2816,), device=device, dtype=torch.bfloat16)
eps = 1e-6

input_row_stride = int(last_token.reshape(16, 2816).stride(0))
actual = rmsnorm_triton(last_token, weight, eps, offset=True)
variance = last_token.float().pow(2).mean(dim=-1, keepdim=True)
expected = (
    last_token * torch.rsqrt(variance + eps) * (weight + 1.0)
).to(last_token.dtype)
torch.cuda.synchronize()

finite_rows = torch.isfinite(actual).reshape(16, -1).all(dim=1)
max_abs_error = float((actual.float() - expected.float()).abs().max().item())
torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
if not bool(finite_rows.all().item()):
    raise SystemExit(
        f"Final RMSNorm produced non-finite rows: {finite_rows.tolist()}"
    )

print(
    "GEMMA4_FINAL_NORM_STRIDE_PREFLIGHT",
    json.dumps(
        {
            "status": "PASS",
            "shape": list(last_token.shape),
            "input_row_stride": input_row_stride,
            "output_row_stride": int(actual.reshape(16, 2816).stride(0)),
            "finite_rows": int(finite_rows.sum().item()),
            "max_abs_error": max_abs_error,
        },
        sort_keys=True,
    ),
)
PY
fi

if [ "${RUN_MEGAGEMM}" = "1" ] && [ "${RUN_KERNEL_PREFLIGHT}" = "1" ]; then
  echo
  echo "== ASYNC COMPACT MOE TILE GPU PREFLIGHT =="
  run_with_timeout "GEMMA4 KERNEL PREFLIGHT" "${PREFLIGHT_TIMEOUT_MIN}" python - <<'PY'
import statistics
import time

import torch

import megagemm.kernels.qwen3_moe as moe_kernel
from megagemm.kernels.qwen3_moe import qwen3_moe_segmented_prefill

device = torch.device("cuda")
rows, hidden, intermediate, experts, top_k = 25, 2816, 704, 128, 8
hidden_states = torch.zeros((rows, hidden), device=device, dtype=torch.bfloat16)
hidden_states[:, 0] = 1.0
gate_up = torch.zeros(
    (experts, 2 * intermediate, hidden),
    device=device,
    dtype=torch.bfloat16,
)
down = torch.zeros(
    (experts, hidden, intermediate),
    device=device,
    dtype=torch.bfloat16,
)
expert_signal = torch.linspace(0.01, 0.25, experts, device=device, dtype=torch.bfloat16)
intermediate_signal = torch.linspace(
    0.25, 1.0, intermediate, device=device, dtype=torch.bfloat16
)
gate_up[:, :intermediate, 0] = expert_signal[:, None] * intermediate_signal[None, :]
gate_up[:, intermediate:, 0] = 1.0
output_ids = torch.arange(hidden, device=device, dtype=torch.int64)
output_signal = torch.linspace(0.5, 1.0, hidden, device=device, dtype=torch.bfloat16)
down[:, output_ids, output_ids.remainder(intermediate)] = output_signal[None, :]
selected = torch.arange(rows * top_k, device=device, dtype=torch.int64).reshape(rows, top_k)
selected.remainder_(54)
routing = torch.full((rows, top_k), 1.0 / top_k, device=device, dtype=torch.bfloat16)
default_moe_config = (16, 64, 64, 4, 3)

def run(async_enabled, partial_reduce_enabled, workspace, config=default_moe_config):
    moe_kernel._CFG_SEGMENTED_PREFILL_ASYNC_TILES = bool(async_enabled)
    moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = bool(partial_reduce_enabled)
    block_m, block_n, block_k, num_warps, num_stages = config
    with torch.no_grad():
        return qwen3_moe_segmented_prefill(
            hidden_states,
            gate_up,
            down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            workspace=workspace,
            force=True,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            fused_gate=True,
            dense_grid=False,
            route_scatter=True,
        )

baseline_workspace = {}
async_workspace = {}
for _ in range(3):
    baseline = run(False, False, baseline_workspace)
    candidate = run(True, False, async_workspace)
torch.cuda.synchronize()
torch.testing.assert_close(candidate, baseline, atol=2e-3, rtol=2e-3)
if int(async_workspace.get("segmented_prefill_async_tiles", 0)) != 1:
    raise SystemExit(f"Async compact tiles did not activate: {async_workspace}")

def measure(async_enabled, partial_reduce_enabled, workspace, config=default_moe_config):
    samples = []
    for _ in range(7):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            run(async_enabled, partial_reduce_enabled, workspace, config)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / 5.0)
    return statistics.median(samples)

baseline_ms = measure(False, False, baseline_workspace)
async_ms = measure(True, False, async_workspace)
speedup = baseline_ms / async_ms
print(
    "async compact MoE tile GPU preflight:",
    {
        "baseline_ms": round(baseline_ms, 4),
        "async_ms": round(async_ms, 4),
        "speedup": round(speedup, 4),
        "max_tiles": async_workspace.get("segmented_prefill_max_tiles"),
    },
)
if speedup <= 1.0:
    raise SystemExit("Async compact MoE tiles did not beat baseline; refusing the paid full run")

partial_workspace = {}
for _ in range(3):
    atomic_output = run(True, False, async_workspace)
    partial_output = run(True, True, partial_workspace)
torch.cuda.synchronize()
torch.testing.assert_close(partial_output, atomic_output, atol=2e-3, rtol=2e-3)
if int(partial_workspace.get("segmented_prefill_partial_reduce", 0)) != 1:
    raise SystemExit(f"Partial reduction did not activate: {partial_workspace}")

atomic_ms = measure(True, False, async_workspace)
partial_ms = measure(True, True, partial_workspace)
partial_speedup = atomic_ms / partial_ms
print(
    "atomic-free partial reduction GPU preflight:",
    {
        "atomic_ms": round(atomic_ms, 4),
        "partial_ms": round(partial_ms, 4),
        "speedup": round(partial_speedup, 4),
    },
)
if partial_speedup <= 1.0:
    raise SystemExit("Partial reduction did not beat atomic accumulation; refusing the paid full run")

from megagemm.kernels.rmsnorm_triton import (
    rmsnorm_triton,
    rmsnorm_triton_add,
    rmsnorm_triton_dual,
)

norm_x = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)
norm_lhs = torch.randn_like(norm_x)
norm_rhs = torch.randn_like(norm_x)
norm_weight_a = torch.randn((hidden,), device=device, dtype=torch.bfloat16)
norm_weight_b = torch.randn((hidden,), device=device, dtype=torch.bfloat16)
norm_eps = 1e-6

baseline_a = rmsnorm_triton(norm_x, norm_weight_a, norm_eps, False)
baseline_b = rmsnorm_triton(norm_x, norm_weight_b, norm_eps, False)
dual_a, dual_b = rmsnorm_triton_dual(
    norm_x,
    norm_weight_a,
    norm_weight_b,
    norm_eps,
)
baseline_add = rmsnorm_triton(
    (norm_lhs + norm_rhs).to(norm_lhs.dtype),
    norm_weight_a,
    norm_eps,
    False,
)
fused_add = rmsnorm_triton_add(
    norm_lhs,
    norm_rhs,
    norm_weight_a,
    norm_eps,
)
torch.cuda.synchronize()
torch.testing.assert_close(dual_a, baseline_a, atol=2e-3, rtol=2e-3)
torch.testing.assert_close(dual_b, baseline_b, atol=2e-3, rtol=2e-3)
torch.testing.assert_close(fused_add, baseline_add, atol=2e-3, rtol=2e-3)

def measure_callable(fn, iterations=100):
    samples = []
    for _ in range(7):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / iterations)
    return statistics.median(samples)

dual_baseline_ms = measure_callable(
    lambda: (
        rmsnorm_triton(norm_x, norm_weight_a, norm_eps, False),
        rmsnorm_triton(norm_x, norm_weight_b, norm_eps, False),
    )
)
dual_fused_ms = measure_callable(
    lambda: rmsnorm_triton_dual(norm_x, norm_weight_a, norm_weight_b, norm_eps)
)
add_baseline_ms = measure_callable(
    lambda: rmsnorm_triton(
        (norm_lhs + norm_rhs).to(norm_lhs.dtype),
        norm_weight_a,
        norm_eps,
        False,
    )
)
add_fused_ms = measure_callable(
    lambda: rmsnorm_triton_add(norm_lhs, norm_rhs, norm_weight_a, norm_eps)
)
dual_speedup = dual_baseline_ms / dual_fused_ms
add_speedup = add_baseline_ms / add_fused_ms
print(
    "Gemma4 fused FFN norms GPU preflight:",
    {
        "dual_speedup": round(dual_speedup, 4),
        "add_norm_speedup": round(add_speedup, 4),
        "dual_baseline_ms": round(dual_baseline_ms, 5),
        "dual_fused_ms": round(dual_fused_ms, 5),
        "add_baseline_ms": round(add_baseline_ms, 5),
        "add_fused_ms": round(add_fused_ms, 5),
    },
)
if dual_speedup <= 1.0 or add_speedup <= 1.0:
    raise SystemExit("Fused FFN norms did not beat baseline; refusing the paid full run")

winner = default_moe_config
print("Gemma4 segmented MoE config: locked to measured default", winner)

del (
    fused_add,
    baseline_add,
    dual_b,
    dual_a,
    baseline_b,
    baseline_a,
    norm_weight_b,
    norm_weight_a,
    norm_rhs,
    norm_lhs,
    norm_x,
    partial_output,
    atomic_output,
    candidate,
    baseline,
    routing,
    output_signal,
    output_ids,
    intermediate_signal,
    selected,
    down,
    gate_up,
    hidden_states,
)
torch.cuda.empty_cache()
PY
  echo "Selected Gemma4 prefill policy: segmented kernels with shape-specific tile tuning"
else
  echo "Skipping repeated GPU kernel preflight (RUN_KERNEL_PREFLIGHT=${RUN_KERNEL_PREFLIGHT})"
fi

if [ "${RUN_MEGAGEMM}" = "1" ] \
  && [ "${RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE}" = "1" ] \
  && [[ ",${BATCH_SIZES}," == *",16,"* ]]; then
  echo
  echo "== B16 SINGLE-KERNEL ATTN-to-MoE/ROUTER BRIDGE GATE (NO MODEL DOWNLOAD) =="
  run_with_timeout \
    "GEMMA4 B16 SINGLE-KERNEL ROUTER BRIDGE GATE" \
    "${PREFLIGHT_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_attn_moe_decode_bridge_microbench.py \
      --rows 16 \
      --warmup 20 \
      --iterations 200 \
      --repeats 7 \
      --minimum-speedup 1.02 \
      --target-gap-ms 0.706 \
      --out-json "${OUT_DIR}/attn_moe_router_single_kernel_gate.json"
  ATTN_MOE_ROUTER_SINGLE_KERNEL_DECISION="$(
    OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["OUT_DIR"]) / "attn_moe_router_single_kernel_gate.json")
    .read_text(encoding="utf-8")
)
print(result.get("decision", "MISSING"))
PY
  )"
  case "${ATTN_MOE_ROUTER_SINGLE_KERNEL_DECISION}" in
    APPLY_SINGLE_KERNEL)
      export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=1
      echo "Single-kernel router bridge accepted; continuing to the full same-VM A/B."
      ;;
    KEEP_TWO_KERNEL)
      export MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=0
      echo "Single-kernel router bridge rejected; keeping the exact two-kernel baseline."
      echo "Stopping before checkpoint download and vLLM installation."
      exit 0
      ;;
    *)
      echo "INVALID SINGLE-KERNEL ROUTER BRIDGE DECISION: ${ATTN_MOE_ROUTER_SINGLE_KERNEL_DECISION}" >&2
      exit 2
      ;;
  esac
fi

if [ "${RUN_MEGAGEMM}" = "1" ] \
  && [ "${RUN_ROUTER_COMPACT_PACK_GATE}" = "1" ] \
  && [[ ",${BATCH_SIZES}," == *",16,"* ]]; then
  echo
  echo "== B16 ROUTER TOP-K/COMPACT-PACK GATE (NO MODEL DOWNLOAD) =="
  run_with_timeout \
    "GEMMA4 B16 ROUTER COMPACT-PACK GATE" \
    "${PREFLIGHT_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_router_compact_pack_microbench.py \
      --rows 16 \
      --warmup 20 \
      --iterations 200 \
      --repeats 7 \
      --minimum-speedup 1.02 \
      --target-gap-ms 0.85 \
      --out-json "${OUT_DIR}/router_compact_pack_gate.json"
  ROUTER_COMPACT_PACK_DECISION="$(
    OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["OUT_DIR"]) / "router_compact_pack_gate.json")
    .read_text(encoding="utf-8")
)
print(result.get("decision", "MISSING"))
PY
  )"
  case "${ROUTER_COMPACT_PACK_DECISION}" in
    APPLY)
      echo "Router compact pack accepted; continuing to the full same-VM A/B."
      ;;
    KEEP_BASELINE)
      echo "Router compact pack rejected; the proven B16 baseline remains selected."
      export MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE=0
      if [ "${STOP_IF_ROUTER_COMPACT_PACK_REJECTED}" = "1" ]; then
        echo "Stopping before checkpoint download and vLLM installation."
        exit 0
      fi
      ;;
    *)
      echo "INVALID ROUTER COMPACT-PACK DECISION: ${ROUTER_COMPACT_PACK_DECISION}" >&2
      exit 2
      ;;
  esac
fi

if [ "${RUN_MEGAGEMM}" = "1" ] \
  && [ "${MEGAGEMM_PREFILL_CUDA_GRAPHS}" = "1" ] \
  && [[ ",${BATCH_SIZES}," == *",16,"* ]]; then
  echo
  echo "== B16 PREFILL GRAPH SAFETY GATE (NO MODEL DOWNLOAD) =="
  run_with_timeout \
    "GEMMA4 B16 PREFILL GRAPH PREFLIGHT" \
    "${GRAPH_PREFLIGHT_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_b16_prefill_graph_preflight.py \
      --replays 5 \
      --out-json "${OUT_DIR}/b16_prefill_graph_preflight.json"

  echo
  echo "== B16 FULL-MODEL GRAPH-SAFE REPLAY GATE (NO MODEL DOWNLOAD) =="
  run_with_timeout \
    "GEMMA4 B16 FULL-MODEL GRAPH PREFLIGHT" \
    "${GRAPH_PREFLIGHT_TIMEOUT_MIN}" \
    python benchmarks/run_gemma4_b16_full_model_graph_preflight.py \
      --replays 3 \
      --out-json "${OUT_DIR}/b16_full_model_graph_preflight.json"
fi

if [ "${BATCH_MODE}" = "1" ] \
  && [ "${RUN_MEGAGEMM}" = "1" ] \
  && [ "${RUN_VLLM}" = "1" ] \
  && [ "${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION}" = "1" ] \
  && [ "${PARITY_GATE_REQUESTED}" = "1" ] \
  && [[ ",${BATCH_SIZES}," == *",16,"* ]]; then
  echo
  echo "== INSTALL vLLM FOR NO-CHECKPOINT PARITY GATE =="
  if [ "${INSTALL_RUNTIME}" = "1" ]; then
    run_with_timeout "INSTALL vLLM FOR PARITY GATE" "${INSTALL_TIMEOUT_MIN}" \
      python -m uv pip install --system --reinstall \
        "vllm==${PINNED_VLLM_VERSION}" \
        "transformers==${PINNED_TRANSFORMERS_VERSION}" \
        --torch-backend=cu129
  else
    echo "Skipping vLLM installation because INSTALL_RUNTIME=${INSTALL_RUNTIME}"
  fi
  refresh_cuda_python_libs
  python - <<'PY'
import torch
import os
import transformers
import vllm

assert vllm.__version__ == os.environ["PINNED_VLLM_VERSION"]
assert transformers.__version__ == os.environ["PINNED_TRANSFORMERS_VERSION"]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__, "import ok for no-checkpoint parity gate")
print("transformers", transformers.__version__, "compatibility pin ok")
PY

  if [ "${RUN_VLLM_MOE_PARITY_GATE}" = "1" ]; then
    echo
    echo "== MEGAGEMM-vLLM FUSED-MoE PRE-DOWNLOAD GATE =="
    run_with_timeout \
      "GEMMA4 MEGAGEMM-vLLM FUSED-MoE PARITY GATE" \
      "${GRAPH_PREFLIGHT_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_vllm_moe_parity_microbench.py \
        --rows "${VLLM_MOE_PARITY_ROWS}" \
        --warmup 3 \
        --iterations 25 \
        --repeats 5 \
        --minimum-speedup 1.02 \
        --out-json "${OUT_DIR}/vllm_moe_parity_gate.json"
    VLLM_MOE_PARITY_DECISION="$(
      OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["OUT_DIR"]) / "vllm_moe_parity_gate.json")
    .read_text(encoding="utf-8")
)
print(result.get("decision", "MISSING"))
PY
    )"
    echo
    case "${VLLM_MOE_PARITY_DECISION}" in
      PORT_VLLM_FUSED_MOE)
        echo "PARITY RESULT: vLLM fused-MoE is at least 2 percent faster."
        echo "Next action is to port that concrete backend."
        ;;
      MOVE_OFF_MOE)
        echo "PARITY RESULT: vLLM fused-MoE does not beat the MegaGemm kernel by 2 percent."
        echo "The B16 gap is outside this expert kernel."
        ;;
      *)
        echo "INVALID MoE PARITY DECISION: ${VLLM_MOE_PARITY_DECISION}" >&2
        exit 2
        ;;
    esac
  fi

  if [ "${RUN_VLLM_ATTENTION_PARITY_GATE}" = "1" ]; then
    echo
    echo "== MEGAGEMM-vLLM ATTENTION-CORE PRE-DOWNLOAD GATE =="
    run_with_timeout \
      "GEMMA4 MEGAGEMM-vLLM ATTENTION-CORE PARITY GATE" \
      "${GRAPH_PREFLIGHT_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_vllm_attention_parity_microbench.py \
        --context 64 \
        --table-blocks 6 \
        --warmup 5 \
        --iterations 100 \
        --repeats 5 \
        --minimum-speedup 1.02 \
        --out-json "${OUT_DIR}/vllm_attention_parity_gate.json"
    VLLM_ATTENTION_PARITY_DECISION="$(
      OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["OUT_DIR"]) / "vllm_attention_parity_gate.json")
    .read_text(encoding="utf-8")
)
print(result.get("decision", "MISSING"))
PY
    )"
    echo
    case "${VLLM_ATTENTION_PARITY_DECISION}" in
      PORT_VLLM_ATTENTION_CORE)
        echo "PARITY RESULT: vLLM attention wins by at least 2 percent on a Gemma4 topology."
        echo "Next action is to port only the measured winning topology."
        ;;
      MOVE_OFF_ATTENTION_CORE)
        echo "PARITY RESULT: vLLM attention does not beat MegaGemm by 2 percent."
        echo "The remaining B16 gap is outside the attention core."
        ;;
      *)
        echo "INVALID ATTENTION PARITY DECISION: ${VLLM_ATTENTION_PARITY_DECISION}" >&2
        exit 2
        ;;
    esac
  fi

  echo "No checkpoint was downloaded and no full vLLM engine was started."
  exit 0
fi

echo
echo "== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="
if [ -d "${MODEL}" ]; then
  MODEL_PATH="${MODEL}"
else
  MODEL_REPO="${MODEL}" MODEL_DIR="${LOCAL_MODEL_DIR}" WORKERS="${HF_DOWNLOAD_WORKERS}" python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = Path(os.environ["MODEL_DIR"])
workers = int(os.environ.get("WORKERS", "16"))

local_dir.mkdir(parents=True, exist_ok=True)
print(f"Downloading {repo_id} to {local_dir} with {workers} workers")
snapshot_download(repo_id=repo_id, local_dir=str(local_dir), max_workers=workers)

index_path = local_dir / "model.safetensors.index.json"
if index_path.exists():
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
else:
    shards = sorted(path.name for path in local_dir.glob("*.safetensors"))

missing = [name for name in shards if not (local_dir / name).exists()]
if missing:
    raise SystemExit(f"Model download is incomplete; missing shards: {missing[:8]}")
if not shards:
    raise SystemExit(f"No safetensors shards found under {local_dir}")

print(f"Verified {len(shards)} safetensors shard(s) under {local_dir}")
PY
  MODEL_PATH="${LOCAL_MODEL_DIR}"
fi

echo
if [ "${BATCH_MODE}" = "1" ] && [ "${MEGAGEMM_PREFILL_CUDA_GRAPHS}" = "1" ]; then
  echo "== RUN MEGAGEMM (BF16, B16 PREFILL GRAPH, DECODE GRAPH REPLAY) =="
else
  echo "== RUN MEGAGEMM (BF16, EAGER PREFILL, DECODE CUDA GRAPH REPLAY) =="
fi
if [ "${RUN_MEGAGEMM}" = "1" ]; then
  PROFILE_ARGS=()
  if [ "${PROFILE_BREAKDOWN}" = "1" ]; then
    PROFILE_ARGS+=(--profile-breakdown)
  fi
  if [ "${PREFILL_TIMING}" = "1" ]; then
    PROFILE_ARGS+=(--prefill-timing)
  fi
  if [ "${BATCH_MODE}" = "1" ]; then
    if [ "${PREFILL_TIMING}" = "1" ]; then
      echo "Batch matrix does not mix prefill timing instrumentation into latency samples" >&2
      exit 2
    fi
    BATCH_PROMOTION_ARGS=()
    if [ "${RUN_VLLM}" = "1" ] \
      && [ "${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION}" = "1" ]; then
      BATCH_PROMOTION_ARGS=(--stop-after-no-decode-promotion)
    fi
    BATCH_DETERMINISTIC_ARGS=(--no-deterministic)
    if [ "${MEGAGEMM_BATCH_DETERMINISTIC}" = "1" ]; then
      BATCH_DETERMINISTIC_ARGS=(--deterministic)
    fi
    BATCH_TRACE_ARGS=()
    if [ "${PREFILL_FINITE_TRACE_ONLY}" = "1" ]; then
      BATCH_TRACE_ARGS=(--prefill-finite-trace-only)
    fi
    run_with_timeout "MEGAGEMM BATCH MATRIX" "${BACKEND_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_moe_batch_vs_vllm.py \
      --backend megagemm \
      --model "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --repeats "${REPEATS}" \
      --batch-sizes "${BATCH_SIZES}" \
      --prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}" \
      "${PROFILE_ARGS[@]}" \
      "${BATCH_PROMOTION_ARGS[@]}" \
      "${BATCH_DETERMINISTIC_ARGS[@]}" \
      "${BATCH_TRACE_ARGS[@]}" \
      --out-json "${OUT_DIR}/megagemm.json"
  else
    run_with_timeout "MEGAGEMM BENCHMARK" "${BACKEND_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_moe_vs_vllm.py \
      --backend megagemm \
      --model "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --repeats "${REPEATS}" \
      "${PROFILE_ARGS[@]}" \
        --out-json "${OUT_DIR}/megagemm.json"
  fi
else
  echo "Skipping MegaGemm because RUN_BACKENDS=${RUN_BACKENDS}"
fi

if [ "${BATCH_MODE}" = "1" ] \
  && [ "${RUN_MEGAGEMM}" = "1" ] \
  && [ "${RUN_VLLM}" = "1" ] \
  && [ "${STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION}" = "1" ]; then
  DECODE_PROMOTION_DECISION="$(
    OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(
    (Path(os.environ["OUT_DIR"]) / "megagemm.json").read_text(encoding="utf-8")
)
compact = (
    result.get("compact_kernel_gate", {})
    .get("decode_kernel_tune", {})
    .get("decision", "MISSING")
)
attention = result.get("attention_kernel_gate", {}).get("decision", "MISSING")
scheduler = result.get("scheduler_burst_gate", {}).get("decision", "MISSING")
promotions = []
if compact == "APPLY":
    promotions.append("compact_decode")
if attention == "APPLY":
    promotions.append("attention_decode")
if str(scheduler).startswith("APPLY_"):
    promotions.append("softcap_decode")
print(
    "APPLY:" + ",".join(promotions)
    if promotions
    else (
        f"KEEP:compact={compact},attention={attention},scheduler={scheduler}"
    )
)
PY
  )"
  if [[ "${DECODE_PROMOTION_DECISION}" != APPLY:* ]]; then
    echo
    echo "NO DECODE PROMOTION (${DECODE_PROMOTION_DECISION}): stopping before vLLM install."
    echo "The v81 exact baseline remains enabled; no new exact candidate earned a paid vLLM rerun."
    RUN_VLLM=0
  fi
fi

if [ "${INSTALL_RUNTIME}" = "1" ] && [ "${RUN_VLLM}" = "1" ]; then
  echo
  echo "== INSTALL vLLM AFTER MEGAGEMM PASSES =="
  run_with_timeout "INSTALL vLLM" "${INSTALL_TIMEOUT_MIN}" \
    python -m uv pip install --system --reinstall \
      "vllm==${PINNED_VLLM_VERSION}" \
      "transformers==${PINNED_TRANSFORMERS_VERSION}" \
      --torch-backend=cu129
  refresh_cuda_python_libs
  python - <<'PY'
import torch
import os
import transformers
import vllm

assert vllm.__version__ == os.environ["PINNED_VLLM_VERSION"]
assert transformers.__version__ == os.environ["PINNED_TRANSFORMERS_VERSION"]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__, "import ok")
print("transformers", transformers.__version__, "compatibility pin ok")
PY
fi

echo
echo "== RUN vLLM (BF16, CUDA GRAPHS, PREFIX CACHE OFF) =="
if [ "${RUN_VLLM}" = "1" ]; then
  if [ "${BATCH_MODE}" = "1" ]; then
    run_with_timeout "vLLM BATCH MATRIX" "${BACKEND_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_moe_batch_vs_vllm.py \
      --backend vllm \
      --model "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --repeats "${REPEATS}" \
      --batch-sizes "${BATCH_SIZES}" \
      --prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}" \
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --out-json "${OUT_DIR}/vllm.json"
  else
    run_with_timeout "vLLM BENCHMARK" "${BACKEND_TIMEOUT_MIN}" \
      python benchmarks/run_gemma4_moe_vs_vllm.py \
      --backend vllm \
      --model "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --max-tokens "${MAX_TOKENS}" \
      --repeats "${REPEATS}" \
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --out-json "${OUT_DIR}/vllm.json"
  fi
else
  echo "Skipping vLLM because RUN_BACKENDS=${RUN_BACKENDS}"
fi

echo
if [ "${COMPARISON_SCOPE}" = "same_vm" ]; then
  echo "== FAIR SAME-VM RESULT =="
elif [ "${COMPARISON_SCOPE}" = "vllm_only" ]; then
  echo "== STANDALONE vLLM RESULT =="
elif [ "${COMPARISON_SCOPE}" = "megagemm_only" ]; then
  echo "== STANDALONE MEGAGEMM RESULT =="
else
  echo "== PERSISTED MEGAGEMM + CURRENT vLLM RESULT =="
fi
OUT_DIR="${OUT_DIR}" COMPARISON_SCOPE="${COMPARISON_SCOPE}" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
comparison_scope = os.environ.get("COMPARISON_SCOPE", "same_vm")
mg_path = out_dir / "megagemm.json"
vl_path = out_dir / "vllm.json"
if not mg_path.exists() or not vl_path.exists():
    if comparison_scope == "vllm_only" and vl_path.exists():
        vl = json.loads(vl_path.read_text(encoding="utf-8"))
        print(f"out_dir: {out_dir}")
        print(f"comparison_scope: {comparison_scope}")
        print(f"vLLM version: {vl.get('version', 'unknown')}")
        for batch_size, case in sorted(
            vl.get("cases", {}).items(), key=lambda item: int(item[0])
        ):
            print(
                f"B={batch_size} "
                + json.dumps(case.get("summary", {}), sort_keys=True)
            )
        raise SystemExit(0)
    if comparison_scope == "megagemm_only" and mg_path.exists():
        mg = json.loads(mg_path.read_text(encoding="utf-8"))
        print(f"out_dir: {out_dir}")
        print(f"comparison_scope: {comparison_scope}")
        for batch_size, case in sorted(
            mg.get("cases", {}).items(), key=lambda item: int(item[0])
        ):
            print(
                f"B={batch_size} "
                + json.dumps(case.get("summary", {}), sort_keys=True)
            )
        raise SystemExit(0)
    print(f"Comparison pending under {out_dir}: megagemm={mg_path.exists()} vllm={vl_path.exists()}")
    raise SystemExit(0)
mg = json.loads(mg_path.read_text(encoding="utf-8"))
vl = json.loads(vl_path.read_text(encoding="utf-8"))

def value(data, key):
    raw = data["summary"].get(key)
    return None if raw is None else float(raw)

if "cases" in mg or "cases" in vl:
    if "cases" not in mg or "cases" not in vl:
        raise SystemExit("Batch comparison requires matrix output from both backends")
    print(f"out_dir: {out_dir}")
    print(f"comparison_scope: {comparison_scope}")
    print(f"GPU: {mg['gpu']['name']}")
    print(f"model: {mg['model']}")
    print(f"dtype: {mg['dtype']} (both, unquantized)")
    print(f"shape: prompt={mg['prompt_tokens']} output={mg['max_tokens']}")
    print(f"workload: {mg.get('workload', 'unknown')}")
    mg_prompt_contract = mg.get("prompt_contract")
    vl_prompt_contract = vl.get("prompt_contract")
    if not mg_prompt_contract or mg_prompt_contract != vl_prompt_contract:
        raise SystemExit(
            "Cross-backend prompt token contract mismatch: "
            f"megagemm={mg_prompt_contract} vllm={vl_prompt_contract}"
        )
    print(
        "input IDs: exact shared manifest "
        f"sha256={mg_prompt_contract['sha256']}"
    )
    print(f"vLLM version: {vl['version']}")
    print("vLLM prefix cache: OFF")
    prefill_gate = mg.get("prefill_kernel_gate") or {}
    for label, key in (
        ("attention-to-MoE/router bridge", "attn_moe_bridge"),
        ("400-row matrix router", "router"),
    ):
        selection = prefill_gate.get(key) or {}
        if selection:
            print(
                f"MegaGemm {label}: {selection.get('decision')} "
                f"selected={selection.get('selected')} "
                f"speedup={float(selection.get('speedup', 0.0) or 0.0):.3f}x "
                f"estimated_savings_ms={float(selection.get('estimated_savings_ms_per_prefill', 0.0) or 0.0):.3f}"
            )
    print()
    print(
        f"{'B':>3} {'MG total':>10} {'vLLM total':>11} {'ratio':>8} "
        f"{'MG decode':>10} {'vLLM decode':>11} {'ratio':>8} "
        f"{'MG prefill':>11} {'vLLM prefill':>12}"
    )

    def number(raw, width):
        return f"{raw:>{width}.2f}" if raw is not None else f"{'n/a':>{width}}"

    def ratio_text(left, right):
        return f"{left / right:>7.3f}x" if left is not None and right else f"{'n/a':>8}"

    all_tokens_exact = True
    for batch_size in mg["batch_sizes"]:
        key = str(batch_size)
        mg_case = mg["cases"][key]
        vl_case = vl["cases"][key]
        mg_total = value(mg_case, "output_tok_s_total_median")
        vl_total = value(vl_case, "output_tok_s_total_median")
        mg_decode = value(mg_case, "decode_tok_s_median")
        vl_decode = value(vl_case, "decode_tok_s_median")
        mg_prefill = value(mg_case, "prefill_ms_median")
        vl_prefill = value(vl_case, "prefill_ms_median")
        print(
            f"{batch_size:>3} {number(mg_total, 10)} {number(vl_total, 11)} "
            f"{ratio_text(mg_total, vl_total)} {number(mg_decode, 10)} "
            f"{number(vl_decode, 11)} {ratio_text(mg_decode, vl_decode)} "
            f"{number(mg_prefill, 11)} {number(vl_prefill, 12)}"
        )
        mg_ids = mg_case["samples"][0].get("token_ids", [])
        vl_ids = vl_case["samples"][0].get("token_ids", [])
        alignment = vl_case["samples"][0].get("output_alignment", {})
        exact = mg_ids == vl_ids
        all_tokens_exact = all_tokens_exact and exact
        total_verdict = f"{(mg_total / vl_total - 1.0) * 100.0:+.1f}%"
        decode_verdict = (
            f"{(mg_decode / vl_decode - 1.0) * 100.0:+.1f}%"
            if mg_decode is not None and vl_decode
            else "n/a (vLLM request phase metrics unavailable)"
        )
        print(
            f"    B={batch_size} verdict: total={total_verdict} "
            f"decode={decode_verdict} tokens_exact={exact}"
        )
        runtime_gate = mg_case.get("runtime_gate") or {}
        print(
            f"    MegaGemm B={batch_size} decode router bridge: "
            f"single_kernel_enabled="
            f"{runtime_gate.get('fused_attn_moe_router_single_kernel_decode_enabled', False)} "
            f"hits={runtime_gate.get('fused_attn_moe_router_single_kernel_decode_hits', 0)}"
        )
        print(
            "    vLLM output alignment: "
            f"method={alignment.get('method', 'unknown')} "
            f"reordered={alignment.get('reordered', False)}"
        )
        warmup_stability = vl_case.get("warmup_stability", {})
        token_stability = vl_case.get("token_stability", {})
        print(
            "    vLLM warm-state contract: "
            f"stable={warmup_stability.get('stable', False)} "
            f"accepted={warmup_stability.get('accepted', False)} "
            f"warmups={warmup_stability.get('completed_warmups', 0)} "
            f"last_pair={warmup_stability.get('last_pair_total_ratio')} "
            f"reason={warmup_stability.get('acceptance_reason', 'unknown')} "
            f"measured_tokens_exact={token_stability.get('exact', False)}"
        )
        scatter = (mg_case.get("runtime_gate") or {}).get(
            "prefill_kv_scatter", {}
        )
        if scatter:
            print(
                "    MegaGemm Triton prefill K/V scatter: "
                f"hits={scatter.get('hits', 0)} "
                f"disabled={scatter.get('disabled', False)} "
                f"failures={scatter.get('failures', 0)}"
            )
        prefill_profile = mg_case.get("prefill_profile_excluded") or {}
        ranked_stages = prefill_profile.get("ranked_stages") or []
        if ranked_stages:
            top_stages = ", ".join(
                f"{item['stage']}={float(item['ms']):.2f}ms"
                for item in ranked_stages[:5]
            )
            print(f"    MegaGemm excluded prefill profile: {top_stages}")
        if not exact:
            first_mismatch = None
            for row_index, (mg_row, vl_row) in enumerate(zip(mg_ids, vl_ids)):
                for token_index, (mg_token, vl_token) in enumerate(zip(mg_row, vl_row)):
                    if mg_token != vl_token:
                        first_mismatch = {
                            "row": row_index,
                            "token": token_index,
                            "megagemm": mg_token,
                            "vllm": vl_token,
                        }
                        break
                if first_mismatch is not None:
                    break
            print(
                "    first_cross_backend_mismatch="
                + json.dumps(first_mismatch, sort_keys=True)
            )
    print(f"Cross-backend greedy token matrix exact: {all_tokens_exact}")
    if all_tokens_exact:
        print("Cross-backend result class: PERFORMANCE_AND_TOKEN_PARITY_PASS")
    else:
        print(
            "Cross-backend result class: SHAPE_MATCHED_PERFORMANCE_ONLY "
            "(token parity failed)"
        )
    raise SystemExit(0)

print(f"out_dir: {out_dir}")
print(f"GPU: {mg['gpu']['name']}")
print(f"model: {mg['model']}")
print(f"dtype: {mg['dtype']} (both, unquantized)")
print(f"shape: batch=1 prompt={mg['prompt_tokens']} output={mg['max_tokens']}")
print(f"vLLM version: {vl['version']}")
print("vLLM prefix cache: OFF")

metrics = (
    ("Total output tok/s", "output_tok_s_total_median", True),
    ("Decode tok/s", "decode_tok_s_median", True),
    ("Total ms", "total_ms_median", False),
    ("Prefill ms", "prefill_ms_median", False),
    ("Decode ms", "decode_ms_median", False),
)
print()
print(f"{'Metric':<22} {'MegaGemm':>12} {'vLLM':>12} {'MG/vLLM':>12}")
for label, key, higher_better in metrics:
    left = value(mg, key)
    right = value(vl, key)
    if left is None or right is None:
        print(f"{label:<22} {str(left):>12} {str(right):>12} {'n/a':>12}")
        continue
    ratio = left / right if higher_better else right / left
    print(f"{label:<22} {left:>12.2f} {right:>12.2f} {ratio:>11.3f}x")

total_ratio = value(mg, "output_tok_s_total_median") / value(vl, "output_tok_s_total_median")
mg_decode = value(mg, "decode_tok_s_median")
vl_decode = value(vl, "decode_tok_s_median")
decode_ratio = mg_decode / vl_decode if mg_decode is not None and vl_decode else None
print()
print(f"VERDICT total:  MegaGemm is {(total_ratio - 1.0) * 100.0:+.1f}% vs vLLM")
if decode_ratio is None:
    print("VERDICT decode: n/a (vLLM request phase metrics unavailable)")
else:
    print(f"VERDICT decode: MegaGemm is {(decode_ratio - 1.0) * 100.0:+.1f}% vs vLLM")
print("MegaGemm decode method:", ", ".join(mg["summary"]["decode_measurement_methods"]))
print("vLLM decode method:", ", ".join(vl["summary"]["decode_measurement_methods"]))

mg_ids = [int(item) for item in mg["samples"][0].get("token_ids", [])]
vl_ids = [int(item) for item in vl["samples"][0].get("token_ids", [])]
common = 0
for left, right in zip(mg_ids, vl_ids):
    if left != right:
        break
    common += 1
print(
    f"Cross-backend greedy tokens: exact={mg_ids == vl_ids} "
    f"common_prefix={common}/{min(len(mg_ids), len(vl_ids))}"
)
PY
