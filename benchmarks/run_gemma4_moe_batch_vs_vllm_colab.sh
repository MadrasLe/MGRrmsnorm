#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_SIZES="${BATCH_SIZES:-2,4,8,16}"
export PREFILL_FINITE_TRACE_ONLY="${PREFILL_FINITE_TRACE_ONLY:-0}"
exec bash "${SCRIPT_DIR}/run_gemma4_moe_vs_vllm_colab.sh" "$@"
