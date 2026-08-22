#!/usr/bin/env python3
"""Compare Qwen3 MoE decode speed between MegaGemm and vLLM.

The script intentionally runs one backend per process. Qwen3-Coder-30B-A3B
is large enough that keeping MegaGemm and vLLM resident at the same time can
change the result or simply OOM.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import glob
import io
import json
import os
import re
import site
import statistics
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TIMING_RE = re.compile(
    r"Prefill:\s*([0-9.]+)ms\s*\((\d+)\s+tokens\)\s*\|\s*"
    r"Decode:\s*([0-9.]+)ms\s*\((\d+)\s+tokens\)\s*\|\s*"
    r"Speed:\s*([0-9.]+)\s+tok/s"
)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gb": round(props.total_memory / 1024**3, 2),
    }


def looks_like_qwen3_coder_30b_a3b(model: str) -> bool:
    text = str(model).replace("\\", "/").lower()
    if "qwen3-coder-30b-a3b" in text:
        return True
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return False
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(cfg.get("model_type", "")).lower() == "qwen3_moe"
        and int(cfg.get("num_hidden_layers", 0) or 0) >= 48
        and int(cfg.get("hidden_size", 0) or 0) >= 2048
        and int(cfg.get("num_experts", cfg.get("moe_intermediate_size", 0)) or 0) > 0
    )


def preflight_qwen3_moe_vram(args: argparse.Namespace) -> None:
    if args.dtype not in ("bf16", "fp16"):
        return
    if getattr(args, "quantize", "none") != "none":
        return
    if args.allow_low_vram or os.environ.get("ALLOW_LOW_VRAM", "0") == "1":
        return
    if not torch.cuda.is_available() or not looks_like_qwen3_coder_30b_a3b(args.model):
        return
    min_vram_gb = float(os.environ.get("QWEN3_MOE_MIN_VRAM_GB", "64"))
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    if vram_gb >= min_vram_gb:
        return
    raise SystemExit(
        "Refusing to load Qwen3-Coder-30B-A3B in BF16/FP16 on this GPU: "
        f"{torch.cuda.get_device_name(0)} has {vram_gb:.2f} GiB, but this benchmark "
        f"needs at least {min_vram_gb:.0f} GiB. The weights alone are about 58 GiB "
        "before KV/workspace. Use A100-80GB, RTX PRO 6000 Blackwell 96GB, "
        "or set ALLOW_LOW_VRAM=1 only for an explicitly quantized/offload experiment."
    )


def dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def vllm_dtype_from_arg(name: str) -> str:
    if name == "bf16":
        return "bfloat16"
    if name == "fp16":
        return "float16"
    return name


def python_lib_roots() -> list[Path]:
    roots: list[Path] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        path = Path(raw)
        if path.exists() and path not in roots:
            roots.append(path)

    try:
        for raw in site.getsitepackages():
            add(raw)
    except Exception:
        pass
    try:
        add(site.getusersitepackages())
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        try:
            add(sysconfig.get_paths().get(key))
        except Exception:
            pass
    return roots


def preload_vllm_cuda_runtime() -> dict[str, str]:
    """Preload pip-packaged CUDA runtime when vLLM wheel needs it."""
    if sys.platform == "win32":
        return {"status": "skipped", "reason": "windows"}

    explicit = os.environ.get("VLLM_CUDA_RUNTIME_LIB")
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    patterns = (
        "nvidia/cuda_runtime/lib/libcudart.so.13*",
        "nvidia/cuda_runtime/lib64/libcudart.so.13*",
        "nvidia/**/libcudart.so.13*",
        "nvidia/**/lib/libcudart.so.13*",
        "nvidia/**/lib64/libcudart.so.13*",
        "nvidia/cuda_runtime/lib/libcudart.so*",
        "nvidia/cuda_runtime/lib64/libcudart.so*",
        "nvidia/**/libcudart.so*",
        "nvidia/**/lib/libcudart.so*",
        "nvidia/**/lib64/libcudart.so*",
    )
    for root in python_lib_roots():
        for pattern in patterns:
            candidates.extend(glob.glob(str(root / pattern)))

    unique = sorted({item for item in candidates if Path(item).exists()})
    unique.sort(key=lambda item: (not item.endswith(".so.13"), item))
    errors: list[str] = []
    for candidate in unique:
        try:
            ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return {"status": "preloaded", "path": candidate}
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    return {
        "status": "not_found",
        "reason": "; ".join(errors[-3:]) if errors else "no candidate under site-packages",
    }


def load_tokenizer(model: str, cache_dir: str | None = None):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return AutoTokenizer.from_pretrained(model, **kwargs)


def build_prompt(tokenizer, prompt: str, use_chat_template: bool) -> tuple[str, int]:
    formatted = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return formatted, len(token_ids)


def build_synthetic_prompt_for_target(
    tokenizer,
    target_tokens: int,
    marker: str,
) -> tuple[str, int]:
    """Build a deterministic code-like prompt near an exact token target."""

    target_tokens = int(target_tokens)
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")

    header = (
        f"Repository analysis request {marker}.\n"
        "Read the following files and reason about performance-critical code paths.\n\n"
    )
    chunk = (
        "```python\n"
        "class KernelPlanner:\n"
        "    def __init__(self, hidden_size, experts, top_k):\n"
        "        self.hidden_size = hidden_size\n"
        "        self.experts = experts\n"
        "        self.top_k = top_k\n\n"
        "    def plan(self, tokens, routing):\n"
        "        grouped = {}\n"
        "        for token_id, expert_id in routing:\n"
        "            grouped.setdefault(expert_id, []).append(token_id)\n"
        "        return grouped\n"
        "```\n\n"
        "Notes: measure prefill, decode, router, expert projection, attention, "
        "and scheduler overhead. Keep the comparison deterministic.\n\n"
    )

    text = header
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += chunk

    for _ in range(12):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        count = len(token_ids)
        if count == target_tokens:
            return text, count
        if count > target_tokens:
            text = tokenizer.decode(
                token_ids[:target_tokens],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            continue
        # Under target after decode/encode roundtrip. Append simple ASCII filler;
        # Qwen tokenizers encode this stably enough to converge in a few passes.
        text += " analysis" * max(1, target_tokens - count)

    final_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(final_ids) > target_tokens:
        text = tokenizer.decode(
            final_ids[:target_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        final_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, len(final_ids)


def build_prompt_batch(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
    batch_size: int,
    prompt_token_target: int = 0,
) -> tuple[list[str], list[int]]:
    formatted_prompts: list[str] = []
    token_counts: list[int] = []
    width = max(2, len(str(max(1, int(batch_size)))))
    for idx in range(max(1, int(batch_size))):
        if prompt_token_target > 0:
            formatted, count = build_synthetic_prompt_for_target(
                tokenizer,
                prompt_token_target,
                marker=f"batch-{idx:0{width}d}",
            )
            formatted_prompts.append(formatted)
            token_counts.append(count)
            continue

        raw_prompt = prompt
        if batch_size > 1:
            raw_prompt = f"{prompt}\n\nRequest marker: {idx:0{width}d}."
        formatted, count = build_prompt(tokenizer, raw_prompt, use_chat_template)
        formatted_prompts.append(formatted)
        token_counts.append(count)
    return formatted_prompts, token_counts


def exact_prompt_patch(engine, exact_prompt: str) -> None:
    """Force MegaGemm to consume the same already-formatted prompt as vLLM."""

    def _prepare_prompt_inputs(prompt: str):
        del prompt
        input_ids = engine.tokenizer.encode(
            exact_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(engine.device)
        return exact_prompt, input_ids

    engine._prepare_prompt_inputs = _prepare_prompt_inputs


def _normalize_stop_ids(tokenizer, ignore_eos: bool) -> set[int]:
    if ignore_eos:
        return set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(item) for item in eos if item is not None}
    return {int(eos)}


def parse_megagemm_timing(log: str) -> dict[str, Any]:
    match = TIMING_RE.search(log)
    if not match:
        return {
            "prefill_ms": None,
            "prompt_tokens": None,
            "decode_ms": None,
            "decode_tokens": None,
            "decode_tok_s": None,
        }
    return {
        "prefill_ms": float(match.group(1)),
        "prompt_tokens": int(match.group(2)),
        "decode_ms": float(match.group(3)),
        "decode_tokens": int(match.group(4)),
        "decode_tok_s": float(match.group(5)),
    }


def megagemm_diagnostics(engine) -> dict[str, Any]:
    model = getattr(engine, "model", None)
    diagnostics: dict[str, Any] = {"runtime_probe": megagemm_runtime_probe()}
    if model is not None:
        diagnostics["model_fastpath"] = {
            "all_full_attention": bool(
                getattr(model, "_all_full_attention", False)
            ),
            "flat_decode_ready": bool(
                getattr(model, "_flat_decode_ready", False)
            ),
            "flat_decode_failed": bool(
                getattr(model, "_flat_decode_failed", False)
            ),
            "flat_decode_failed_reason": str(
                getattr(model, "_flat_decode_failed_reason", "")
            ),
            "flat_kind": (
                "gemma4"
                if bool(getattr(model, "_flat_is_gemma4", False))
                else (
                    "hybrid"
                    if bool(getattr(model, "_flat_is_hybrid", False))
                    else "dense"
                )
            ),
        }
    runtime_stats = getattr(model, "decode_runtime_stats", None)
    if callable(runtime_stats):
        try:
            stats = runtime_stats()
            if stats:
                diagnostics["decode_runtime_stats"] = stats
        except Exception as exc:
            diagnostics["decode_runtime_stats_error"] = f"{type(exc).__name__}: {exc}"
    decode_timing = getattr(model, "get_last_decode_timing", None)
    if callable(decode_timing):
        try:
            last_timing = decode_timing()
            if last_timing:
                diagnostics["last_decode_timing"] = last_timing
        except Exception as exc:
            diagnostics["last_decode_timing_error"] = f"{type(exc).__name__}: {exc}"
    prefill_timing = getattr(model, "get_last_prefill_timing", None)
    if callable(prefill_timing):
        try:
            last_prefill = prefill_timing()
            if last_prefill:
                diagnostics["last_prefill_timing"] = last_prefill
        except Exception as exc:
            diagnostics["last_prefill_timing_error"] = f"{type(exc).__name__}: {exc}"
    return diagnostics


def format_megagemm_diagnostics_log(diagnostics: dict[str, Any]) -> str:
    stats = diagnostics.get("decode_runtime_stats")
    if not isinstance(stats, dict):
        runtime = diagnostics.get("runtime_probe")
        if isinstance(runtime, dict):
            fastpath = diagnostics.get("model_fastpath")
            fields = [
                f"grouped_decode={runtime.get('qwen3_moe_grouped', '')}",
                f"fused_router={runtime.get('qwen3_moe_fused_router', '')}",
                f"fused_router_max_rows={runtime.get('qwen3_moe_fused_router_max_rows', '')}",
                f"router_k_splits={runtime.get('qwen3_moe_router_k_splits', '')}",
                f"grouped_fused_gate={runtime.get('qwen3_moe_grouped_fused_gate', '')}",
                f"grouped_dot={runtime.get('qwen3_moe_grouped_dot', '')}",
                f"grouped_dot_allow_graphs={runtime.get('qwen3_moe_grouped_dot_allow_cuda_graphs', '')}",
                f"shared_route={runtime.get('qwen3_moe_shared_route_decode', '')}",
                f"shared_route_batch_max_rows={runtime.get('qwen3_moe_shared_route_batch_max_rows', '')}",
                f"shared_route_assume_identical={runtime.get('qwen3_moe_shared_route_assume_identical', '')}",
                f"single_row_gemv={runtime.get('qwen3_moe_single_row_gemv', '')}",
                f"route_matrix={runtime.get('qwen3_moe_route_matrix_decode', '')}",
                f"route_matrix_max_rows={runtime.get('qwen3_moe_route_matrix_max_rows', '')}",
                f"shared_route_partial_reduce={runtime.get('qwen3_moe_shared_route_partial_reduce', '')}",
                f"shared_route_coalesced_weights={runtime.get('qwen3_moe_shared_route_coalesced_weights', '')}",
                f"shared_route_token_accum={runtime.get('qwen3_moe_shared_route_token_accum', '')}",
                f"shared_route_block_m={runtime.get('qwen3_moe_shared_route_block_m', '')}",
                f"shared_route_gate_block_n={runtime.get('qwen3_moe_shared_route_gate_block_n', '')}",
                f"shared_route_gate_k_splits={runtime.get('qwen3_moe_shared_route_gate_k_splits', '')}",
                f"shared_route_down_block_n={runtime.get('qwen3_moe_shared_route_down_block_n', '')}",
                f"shared_route_split_gate={runtime.get('qwen3_moe_shared_route_split_gate', '')}",
                f"expert_grouped_general={runtime.get('qwen3_moe_expert_grouped_general_decode', '')}",
                f"expert_grouped={runtime.get('qwen3_moe_expert_grouped_decode', '')}",
                f"expert_grouped_dense={runtime.get('qwen3_moe_expert_grouped_dense_decode', '')}",
                f"expert_grouped_compact={runtime.get('qwen3_moe_expert_grouped_compact_decode', '')}",
                f"expert_grouped_compact_fused_pack={runtime.get('qwen3_moe_expert_grouped_compact_fused_pack', '')}",
                f"expert_grouped_compact_partial_reduce={runtime.get('qwen3_moe_expert_grouped_compact_partial_reduce', '')}",
                f"expert_grouped_compact_active_list={runtime.get('qwen3_moe_expert_grouped_compact_active_list', '')}",
                f"expert_grouped_compact_token_accum={runtime.get('qwen3_moe_expert_grouped_compact_token_accum', '')}",
                f"expert_grouped_compact_gate_block_n={runtime.get('qwen3_moe_expert_grouped_compact_gate_block_n', '')}",
                f"expert_grouped_compact_down_block_n={runtime.get('qwen3_moe_expert_grouped_compact_down_block_n', '')}",
                f"expert_grouped_compact_direct_out={runtime.get('qwen3_moe_expert_grouped_compact_direct_out', '')}",
                f"token_accum={runtime.get('qwen3_moe_token_accum', '')}",
            ]
            if isinstance(fastpath, dict):
                fields.extend(
                    [
                        "all_full_attention="
                        f"{int(bool(fastpath.get('all_full_attention')))}",
                        f"flat_kind={fastpath.get('flat_kind', 'unknown')}",
                        "flat_ready="
                        f"{int(bool(fastpath.get('flat_decode_ready')))}",
                        "flat_failed="
                        f"{int(bool(fastpath.get('flat_decode_failed')))}",
                    ]
                )
            stats_error = diagnostics.get("decode_runtime_stats_error")
            if stats_error:
                fields.append(f"stats_error={stats_error!r}")
            return "[MegaGemm] Qwen3 MoE runtime flags: " + " ".join(fields)
        return ""

    segmented_hits = stats.get("qwen3_moe_segmented_prefill_total_hits")
    bucketed_hits = stats.get("qwen3_moe_bucketed_prefill_total_hits")
    batched_hits = stats.get("qwen3_moe_batched_prefill_total_hits")
    sorted_hits = stats.get("qwen3_moe_sorted_prefill_total_hits")
    if (
        segmented_hits is None
        and bucketed_hits is None
        and batched_hits is None
        and sorted_hits is None
        and "qwen3_moe_grouped_decode_enabled" not in stats
    ):
        return ""

    fields = [
        f"all_full_attention={int(bool(diagnostics.get('model_fastpath', {}).get('all_full_attention', False)))}",
        f"flat_kind={diagnostics.get('model_fastpath', {}).get('flat_kind', 'unknown')}",
        f"grouped_decode={int(bool(stats.get('qwen3_moe_grouped_decode_enabled', False)))}",
        f"grouped_hits={int(stats.get('qwen3_moe_grouped_decode_total_hits') or 0)}",
        f"fused_router={int(bool(stats.get('qwen3_moe_fused_router', False)))}",
        f"fused_router_max_rows={int(stats.get('qwen3_moe_fused_router_max_rows') or 0)}",
        f"router_k_splits={int(stats.get('qwen3_moe_router_k_splits') or 1)}",
        f"token_accum={int(bool(stats.get('qwen3_moe_grouped_decode_token_accum', False)))}",
        f"grouped_fused_gate={int(bool(stats.get('qwen3_moe_grouped_decode_fused_gate', False)))}",
        f"grouped_dot={int(bool(stats.get('qwen3_moe_grouped_decode_dot', False)))}",
        f"grouped_dot_requested={int(bool(stats.get('qwen3_moe_grouped_decode_dot_requested', False)))}",
        f"grouped_dot_graph_disabled={int(bool(stats.get('qwen3_moe_grouped_decode_dot_graph_disabled', False)))}",
        f"shared_route={int(bool(stats.get('qwen3_moe_shared_route_decode', False)))}",
        f"shared_route_batch_max_rows={int(stats.get('qwen3_moe_shared_route_batch_max_rows') or 1)}",
        f"shared_route_assume_identical={int(bool(stats.get('qwen3_moe_shared_route_assume_identical', False)))}",
        f"shared_route_gate_k_splits={int(stats.get('qwen3_moe_shared_route_gate_k_splits') or 1)}",
        f"shared_route_last_token_accum_layers={int(stats.get('qwen3_moe_shared_route_decode_last_token_accum_layers') or 0)}",
        f"shared_route_last_split_gate_layers={int(stats.get('qwen3_moe_shared_route_decode_last_split_gate_layers') or 0)}",
        f"shared_route_last_partial_reduce_layers={int(stats.get('qwen3_moe_shared_route_decode_last_partial_reduce_layers') or 0)}",
        f"shared_route_residual_fused_layers={int(stats.get('qwen3_moe_shared_route_decode_residual_fused_layers') or 0)}",
        f"single_row_gemv={int(bool(stats.get('qwen3_moe_single_row_gemv', False)))}",
        f"shared_route_disabled={int(stats.get('qwen3_moe_shared_route_decode_disabled_layers') or 0)}",
        f"route_matrix={int(bool(stats.get('qwen3_moe_route_matrix_decode', False)))}",
        f"route_matrix_max_rows={int(stats.get('qwen3_moe_route_matrix_decode_max_rows') or 1)}",
        f"route_matrix_disabled={int(stats.get('qwen3_moe_route_matrix_decode_disabled_layers') or 0)}",
        f"expert_grouped_general={int(bool(stats.get('qwen3_moe_expert_grouped_general_decode', False)))}",
        f"expert_grouped_general_disabled={int(stats.get('qwen3_moe_expert_grouped_general_decode_disabled_layers') or 0)}",
        f"expert_grouped={int(bool(stats.get('qwen3_moe_expert_grouped_decode', False)))}",
        f"expert_grouped_disabled={int(stats.get('qwen3_moe_expert_grouped_decode_disabled_layers') or 0)}",
        f"expert_grouped_compact={int(bool(stats.get('qwen3_moe_expert_grouped_compact_decode', False)))}",
        f"expert_grouped_compact_fused_pack={int(bool(stats.get('qwen3_moe_expert_grouped_compact_fused_pack', False)))}",
        f"expert_grouped_compact_partial_reduce={int(bool(stats.get('qwen3_moe_expert_grouped_compact_partial_reduce', False)))}",
        f"expert_grouped_compact_active_list={int(bool(stats.get('qwen3_moe_expert_grouped_compact_active_list', False)))}",
        f"expert_grouped_compact_token_accum={int(bool(stats.get('qwen3_moe_expert_grouped_compact_token_accum', False)))}",
        f"expert_grouped_compact_gate_block_n={int(stats.get('qwen3_moe_expert_grouped_compact_gate_block_n') or 0)}",
        f"expert_grouped_compact_down_block_n={int(stats.get('qwen3_moe_expert_grouped_compact_down_block_n') or 0)}",
        f"expert_grouped_compact_direct_out={int(bool(stats.get('qwen3_moe_expert_grouped_compact_direct_out', False)))}",
        f"expert_grouped_compact_disabled={int(stats.get('qwen3_moe_expert_grouped_compact_decode_disabled_layers') or 0)}",
        f"grouped_int8={int(bool(stats.get('qwen3_moe_grouped_decode_int8', False)))}",
        f"expert_int8_layers={int(stats.get('qwen3_moe_expert_int8_layers') or 0)}",
        f"int8_dequant_prefill_hits={int(stats.get('qwen3_moe_int8_dequant_prefill_total_hits') or 0)}",
        f"int8_dequant_prefill_disabled={int(stats.get('qwen3_moe_int8_dequant_prefill_disabled_layers') or 0)}",
        f"grouped_dot_disabled={int(stats.get('qwen3_moe_grouped_decode_dot_disabled_layers') or 0)}",
        f"segmented_hits={int(segmented_hits or 0)}",
        f"bucketed_hits={int(bucketed_hits or 0)}",
        f"batched_hits={int(batched_hits or 0)}",
        f"sorted_hits={int(sorted_hits or 0)}",
    ]
    if "fused_rmsnorm_lm_head_argmax_use" in stats:
        fields.append(
            "fused_rmsnorm_lm_head_use="
            f"{int(bool(stats.get('fused_rmsnorm_lm_head_argmax_use')))}"
        )
        fields.append(
            "fused_rmsnorm_lm_head_enabled="
            f"{int(bool(stats.get('fused_rmsnorm_lm_head_argmax_enabled')))}"
        )
        fields.append(
            "fused_rmsnorm_lm_head_checked="
            f"{int(bool(stats.get('fused_rmsnorm_lm_head_argmax_checked')))}"
        )
        fields.append(
            "fused_rmsnorm_lm_head_disabled="
            f"{int(bool(stats.get('fused_rmsnorm_lm_head_argmax_disabled')))}"
        )
        rms_skip = str(stats.get("fused_rmsnorm_lm_head_argmax_skip_reason") or "")
        rms_error = str(stats.get("fused_rmsnorm_lm_head_argmax_error") or "")
        if rms_skip:
            fields.append(f"fused_rmsnorm_lm_head_skip={rms_skip!r}")
        if rms_error:
            fields.append(f"fused_rmsnorm_lm_head_error={rms_error!r}")
    if "fused_lm_head_argmax_use" in stats:
        fields.append(
            "fused_lm_head_use="
            f"{int(bool(stats.get('fused_lm_head_argmax_use')))}"
        )
        fields.append(
            "fused_lm_head_enabled="
            f"{int(bool(stats.get('fused_lm_head_argmax_decode_enabled')))}"
        )
        fields.append(
            "fused_lm_head_checked="
            f"{int(bool(stats.get('fused_lm_head_argmax_checked')))}"
        )
        fields.append(
            "fused_lm_head_disabled="
            f"{int(bool(stats.get('fused_lm_head_argmax_disabled')))}"
        )
        lm_skip = str(stats.get("fused_lm_head_argmax_skip_reason") or "")
        lm_error = str(stats.get("fused_lm_head_argmax_error") or "")
        if lm_skip:
            fields.append(f"fused_lm_head_skip={lm_skip!r}")
        if lm_error:
            fields.append(f"fused_lm_head_error={lm_error!r}")
    if "fused_rmsnorm_qkv_graph_guard_enabled" in stats:
        fields.append(
            "fused_qkv_graph_guard="
            f"{int(bool(stats.get('fused_rmsnorm_qkv_graph_guard_enabled')))}"
        )
    if "qwen3_moe_segmented_prefill_dense_grid" in stats:
        fields.append(
            "segmented_dense_grid="
            f"{int(bool(stats.get('qwen3_moe_segmented_prefill_dense_grid')))}"
        )
    if "qwen3_moe_segmented_prefill_fused_gate" in stats:
        fields.append(
            "segmented_fused_gate="
            f"{int(bool(stats.get('qwen3_moe_segmented_prefill_fused_gate')))}"
        )
    if "qwen3_moe_segmented_prefill_residual_fused_layers" in stats:
        fields.append(
            "segmented_residual_fused_layers="
            f"{int(stats.get('qwen3_moe_segmented_prefill_residual_fused_layers') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_residual_fused_hits" in stats:
        fields.append(
            "segmented_residual_fused_hits="
            f"{int(stats.get('qwen3_moe_segmented_prefill_residual_fused_hits') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_route_scatter" in stats:
        fields.append(
            "segmented_route_scatter="
            f"{int(bool(stats.get('qwen3_moe_segmented_prefill_route_scatter')))}"
        )
    if "qwen3_moe_segmented_prefill_route_block" in stats:
        fields.append(
            "segmented_route_block="
            f"{int(stats.get('qwen3_moe_segmented_prefill_route_block') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_route_scatter_hits" in stats:
        fields.append(
            "segmented_route_scatter_hits="
            f"{int(stats.get('qwen3_moe_segmented_prefill_route_scatter_hits') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_route_argsort_hits" in stats:
        fields.append(
            "segmented_route_argsort_hits="
            f"{int(stats.get('qwen3_moe_segmented_prefill_route_argsort_hits') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_tiles" in stats:
        fields.append(
            f"segmented_tiles={int(stats.get('qwen3_moe_segmented_prefill_tiles') or 0)}"
        )
    if "qwen3_moe_segmented_prefill_assignments" in stats:
        fields.append(
            "segmented_assignments="
            f"{int(stats.get('qwen3_moe_segmented_prefill_assignments') or 0)}"
        )
    if "qwen3_moe_bucketed_prefill_pad_waste" in stats:
        fields.append(
            "bucketed_pad_waste="
            f"{float(stats.get('qwen3_moe_bucketed_prefill_pad_waste') or 0.0) * 100.0:.1f}%"
        )
    if "qwen3_moe_bucketed_prefill_bucket_launches" in stats:
        fields.append(
            f"bucket_launches={int(stats.get('qwen3_moe_bucketed_prefill_bucket_launches') or 0)}"
        )
    if "qwen3_moe_bucketed_prefill_valid_assignments" in stats:
        fields.append(
            f"valid_assignments={int(stats.get('qwen3_moe_bucketed_prefill_valid_assignments') or 0)}"
        )
    if "qwen3_moe_bucketed_prefill_padded_assignments" in stats:
        fields.append(
            f"padded_assignments={int(stats.get('qwen3_moe_bucketed_prefill_padded_assignments') or 0)}"
        )
    failure = str(stats.get("qwen3_moe_bucketed_prefill_first_failure") or "")
    if failure:
        fields.append(f"bucketed_failure={failure}")
    segmented_failure = str(stats.get("qwen3_moe_segmented_prefill_first_failure") or "")
    if segmented_failure:
        fields.append(f"segmented_failure={segmented_failure}")
    route_failure = str(
        stats.get("qwen3_moe_segmented_prefill_route_scatter_first_failure") or ""
    )
    if route_failure:
        fields.append(f"segmented_route_failure={route_failure}")
    return "[MegaGemm] Qwen3 MoE prefill diagnostics: " + " ".join(fields)


def run_megagemm_once(engine, prompt: str, max_tokens: int, ignore_eos: bool) -> dict[str, Any]:
    eos_token_id = getattr(engine.tokenizer, "eos_token_id", None)
    if ignore_eos:
        engine.tokenizer.eos_token_id = None
    buffer = io.StringIO()
    sync_cuda()
    total_start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buffer):
            text = engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0,
                verbose=True,
            )
    finally:
        if ignore_eos:
            engine.tokenizer.eos_token_id = eos_token_id
    sync_cuda()
    total_ms = (time.perf_counter() - total_start) * 1000.0
    log = buffer.getvalue()
    row = parse_megagemm_timing(log)
    if ignore_eos and int(row.get("decode_tokens") or 0) != int(max_tokens):
        raise RuntimeError(
            "MegaGemm fixed-length benchmark produced "
            f"{row.get('decode_tokens')} tokens; expected exactly {max_tokens}"
        )
    prefill_ms = float(row.get("prefill_ms") or 0.0)
    decode_ms = float(row.get("decode_ms") or 0.0)
    row.update(
        {
            "total_ms": total_ms,
            "non_decode_ms": max(0.0, total_ms - decode_ms),
            "scheduler_overhead_ms": max(0.0, total_ms - prefill_ms - decode_ms),
            "output_tok_s_total": (
                float(row["decode_tokens"]) / (total_ms / 1000.0)
                if row.get("decode_tokens") and total_ms > 0.0
                else None
            ),
            "text_prefix": str(text)[:240],
            "raw_log": log.strip(),
        }
    )
    diagnostics = megagemm_diagnostics(engine)
    if diagnostics:
        row["diagnostics"] = diagnostics
        diag_log = format_megagemm_diagnostics_log(diagnostics)
        if diag_log:
            row["raw_log"] = (row["raw_log"] + "\n" + diag_log).strip()
    return row


def run_megagemm_batch_once(
    engine,
    exact_prompts: list[str],
    max_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    from megagemm.engine.scheduler import Scheduler

    stop_ids = _normalize_stop_ids(engine.tokenizer, ignore_eos)
    scheduler = Scheduler(
        model=engine.model,
        block_manager=engine.block_manager,
        max_batch_size=engine.max_batch_size,
        device=engine.device,
        materialize_generated_tokens=False,
    )
    for prompt in exact_prompts:
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        scheduler.add_request(
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop_token_ids=stop_ids,
        )

    sync_cuda()
    start = time.perf_counter()
    while scheduler.has_pending():
        scheduler.step()
    sync_cuda()
    total_ms = (time.perf_counter() - start) * 1000.0

    stats = scheduler.get_stats()
    completed = int(stats.get("completed", len(exact_prompts)) or 0)
    output_tokens = int(stats.get("total_tokens", 0) or 0)
    decode_ms = float(stats.get("decode_time_ms", 0.0) or 0.0)
    decode_tokens = max(0, output_tokens - completed)
    expected_output_tokens = int(max_tokens) * len(exact_prompts)
    if ignore_eos and output_tokens != expected_output_tokens:
        raise RuntimeError(
            "MegaGemm fixed-length batch benchmark produced "
            f"{output_tokens} tokens; expected exactly {expected_output_tokens}"
        )
    prefill_ms = float(stats.get("prefill_time_ms", 0.0) or 0.0)
    non_decode_ms = max(0.0, total_ms - decode_ms)
    scheduler_overhead_ms = max(0.0, total_ms - prefill_ms - decode_ms)
    row = {
        "prefill_ms": prefill_ms,
        "prompt_tokens": None,
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "decode_tok_s": decode_tokens / (decode_ms / 1000.0) if decode_ms > 0.0 else None,
        "total_ms": total_ms,
        "non_decode_ms": non_decode_ms,
        "scheduler_overhead_ms": scheduler_overhead_ms,
        "output_tokens": output_tokens,
        "output_tok_s_total": output_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
        "batch_size": len(exact_prompts),
        "scheduler_stats": stats,
        "text_prefix": "",
        "raw_log": (
            f"[MegaGemm] Batch complete: {len(exact_prompts)} prompts, "
            f"{output_tokens} tokens in {total_ms:.1f}ms | "
            f"Throughput: {output_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0:.1f} tok/s"
        ),
    }
    engine._last_scheduler = scheduler
    diagnostics = megagemm_diagnostics(engine)
    if diagnostics:
        row["diagnostics"] = diagnostics
        diag_log = format_megagemm_diagnostics_log(diagnostics)
        if diag_log:
            row["raw_log"] = (row["raw_log"] + "\n" + diag_log).strip()
    return row


def megagemm_runtime_probe() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "decode_timing": os.environ.get("MEGAGEMM_DECODE_TIMING", ""),
        "decode_timing_detail": os.environ.get("MEGAGEMM_DECODE_TIMING_DETAIL", ""),
        "decode_timing_print": os.environ.get("MEGAGEMM_DECODE_TIMING_PRINT", ""),
        "prefill_timing": os.environ.get("MEGAGEMM_PREFILL_TIMING", ""),
        "prefill_timing_print": os.environ.get("MEGAGEMM_PREFILL_TIMING_PRINT", ""),
        "decode_cuda_graphs": os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS", ""),
        "decode_cuda_graphs_prefer_step": os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP", ""),
        "decode_cuda_graphs_min_batch": os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS_MIN_BATCH", ""),
        "decode_cuda_graphs_shared_shape_cache": os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", ""),
        "decode_cuda_graphs_allow_qwen3_moe": os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS_ALLOW_QWEN3_MOE", ""),
        "fused_rmsnorm_qkv_allow_cuda_graphs": os.environ.get(
            "MEGAGEMM_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS",
            "",
        ),
        "generate_cuda_graphs": os.environ.get("MEGAGEMM_GENERATE_CUDA_GRAPHS", ""),
        "generate_multi_step_cuda_graphs": os.environ.get("MEGAGEMM_GENERATE_MULTI_STEP_CUDA_GRAPHS", ""),
        "generate_step_cuda_graphs": os.environ.get("MEGAGEMM_GENERATE_STEP_CUDA_GRAPHS", ""),
        "generate_gpu_token_chain": os.environ.get("MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN", ""),
        "generate_gpu_token_chain_allow_qwen3_moe": os.environ.get(
            "MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN_ALLOW_QWEN3_MOE",
            "",
        ),
        "generate_skip_token_materialization": os.environ.get(
            "MEGAGEMM_GENERATE_SKIP_TOKEN_MATERIALIZATION",
            "",
        ),
        "generate_fused_argmax_step": os.environ.get(
            "MEGAGEMM_GENERATE_FUSED_ARGMAX_STEP",
            "",
        ),
        "generate_direct_graph_inputs": os.environ.get(
            "MEGAGEMM_GENERATE_DIRECT_GRAPH_INPUTS",
            "",
        ),
        "generate_persistent_step_graph_inputs": os.environ.get(
            "MEGAGEMM_GENERATE_PERSISTENT_STEP_GRAPH_INPUTS",
            "",
        ),
        "generate_cuda_graphs_stable_max_blocks": os.environ.get("MEGAGEMM_GENERATE_CUDA_GRAPHS_STABLE_MAX_BLOCKS", ""),
        "int8_skip_ops": os.environ.get("MEGAGEMM_INT8_SKIP_OPS", ""),
        "disable_cuda_rmsnorm": os.environ.get("MEGAGEMM_DISABLE_CUDA_RMSNORM", ""),
        "flat_decode": os.environ.get("MEGAGEMM_FLAT_DECODE", ""),
        "require_qwen3_moe_dense_flat": os.environ.get(
            "MEGAGEMM_REQUIRE_QWEN3_MOE_DENSE_FLAT",
            "",
        ),
        "packed_attn_uniform_batch": os.environ.get("MEGAGEMM_PACKED_ATTN_UNIFORM_BATCH", ""),
        "packed_attn_uniform_max_score_mb": os.environ.get("MEGAGEMM_PACKED_ATTN_UNIFORM_MAX_SCORE_MB", ""),
        "packed_attn_uniform_reserve_mb": os.environ.get("MEGAGEMM_PACKED_ATTN_UNIFORM_RESERVE_MB", ""),
        "prefill_gqa_mode": os.environ.get("MEGAGEMM_PREFILL_GQA_MODE", ""),
        "qwen3_moe_grouped": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", ""),
        "qwen3_moe_fused_router": os.environ.get("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER", ""),
        "qwen3_moe_fused_router_max_rows": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_FUSED_ROUTER_MAX_ROWS",
            "",
        ),
        "qwen3_moe_router_k_splits": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_ROUTER_K_SPLITS",
            "",
        ),
        "qwen3_moe_grouped_fused_gate": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_FUSED_GATE", ""),
        "qwen3_moe_grouped_dot": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_DOT", ""),
        "qwen3_moe_grouped_dot_allow_cuda_graphs": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_GROUPED_DOT_ALLOW_CUDA_GRAPHS",
            "",
        ),
        "qwen3_moe_expert_grouped_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DECODE", ""),
        "qwen3_moe_shared_route_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DECODE", ""),
        "qwen3_moe_expert_grouped_general_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_GENERAL_DECODE", ""),
        "qwen3_moe_expert_grouped_dense_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DENSE_DECODE", ""),
        "qwen3_moe_expert_grouped_compact_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DECODE", ""),
        "qwen3_moe_expert_grouped_compact_fused_pack": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_FUSED_PACK", ""),
        "qwen3_moe_expert_grouped_compact_partial_reduce": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE", ""),
        "qwen3_moe_expert_grouped_compact_active_list": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST", ""),
        "qwen3_moe_expert_grouped_compact_token_accum": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM", ""),
        "qwen3_moe_expert_grouped_compact_gate_block_n": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N", ""),
        "qwen3_moe_expert_grouped_compact_down_block_n": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N", ""),
        "qwen3_moe_expert_grouped_compact_direct_out": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DIRECT_OUT", ""),
        "qwen3_moe_expert_grouped_min_rows": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_MIN_ROWS", ""),
        "qwen3_moe_expert_grouped_max_rows": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_MAX_ROWS", ""),
        "qwen3_moe_expert_grouped_block_m": os.environ.get("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_BLOCK_M", ""),
        "qwen3_moe_shared_route_batch_max_rows": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_BATCH_MAX_ROWS", ""),
        "qwen3_moe_shared_route_assume_identical": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_ASSUME_IDENTICAL", ""),
        "qwen3_moe_single_row_gemv": os.environ.get("MEGAGEMM_QWEN3_MOE_SINGLE_ROW_GEMV", ""),
        "qwen3_moe_route_matrix_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_ROUTE_MATRIX_DECODE", ""),
        "qwen3_moe_route_matrix_max_rows": os.environ.get("MEGAGEMM_QWEN3_MOE_ROUTE_MATRIX_MAX_ROWS", ""),
        "qwen3_moe_shared_route_partial_reduce": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_PARTIAL_REDUCE", ""),
        "qwen3_moe_shared_route_coalesced_weights": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_COALESCED_WEIGHTS",
            "",
        ),
        "qwen3_moe_shared_route_token_accum": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM", ""),
        "qwen3_moe_shared_route_token_accum_num_warps": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS", ""
        ),
        "qwen3_moe_shared_route_block_m": os.environ.get("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_BLOCK_M", ""),
        "qwen3_moe_shared_route_gate_block_n": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_BLOCK_N",
            "",
        ),
        "qwen3_moe_shared_route_gate_k_splits": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_K_SPLITS",
            "",
        ),
        "qwen3_moe_shared_route_down_block_n": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DOWN_BLOCK_N",
            "",
        ),
        "qwen3_moe_shared_route_split_gate": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE",
            "",
        ),
        "qwen3_moe_int8_decode": os.environ.get("MEGAGEMM_QWEN3_MOE_INT8_DECODE", ""),
        "qwen3_moe_int8_dequant_prefill": os.environ.get("MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL", ""),
        "qwen3_moe_int8_dequant_prefill_min_assignments": os.environ.get(
            "MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS", ""
        ),
        "qwen3_moe_segmented_prefill": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL", ""),
        "qwen3_moe_segmented_prefill_dense_grid": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_DENSE_GRID", ""),
        "qwen3_moe_segmented_prefill_fused_gate": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE", ""),
        "qwen3_moe_segmented_prefill_route_scatter": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_SCATTER", ""),
        "qwen3_moe_segmented_prefill_route_block": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_BLOCK", ""),
        "qwen3_moe_segmented_prefill_min_assignments": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS", ""),
        "qwen3_moe_segmented_prefill_block_m": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_M", ""),
        "qwen3_moe_segmented_prefill_block_n": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_N", ""),
        "qwen3_moe_segmented_prefill_block_k": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_K", ""),
        "qwen3_moe_segmented_prefill_fused_gate_block_n": os.environ.get("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N", ""),
        "qwen3_moe_batched_prefill": os.environ.get("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL", ""),
        "qwen3_moe_batched_prefill_min_assignments": os.environ.get("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS", ""),
        "qwen3_moe_bucketed_prefill": os.environ.get("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL", ""),
        "qwen3_moe_bucketed_prefill_min_assignments": os.environ.get("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS", ""),
        "qwen3_moe_bucketed_prefill_bucket_size": os.environ.get("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE", ""),
        "qwen3_moe_sorted_prefill": os.environ.get("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL", ""),
        "qwen3_moe_sorted_prefill_min_assignments": os.environ.get("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL_MIN_ASSIGNMENTS", ""),
        "qwen3_moe_token_accum": os.environ.get("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM", ""),
        "qwen3_moe_token_accum_min_rows": os.environ.get("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM_MIN_ROWS", ""),
        "qwen3_moe_grouped_block_n": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_N", ""),
        "qwen3_moe_grouped_block_k": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_K", ""),
        "qwen3_moe_grouped_num_warps": os.environ.get("MEGAGEMM_QWEN3_MOE_GROUPED_NUM_WARPS", ""),
        "fused_lm_head_argmax_decode": os.environ.get("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_DECODE", ""),
        "fused_rmsnorm_lm_head_argmax_decode": os.environ.get(
            "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE",
            "",
        ),
        "fused_rmsnorm_lm_head_argmax_min_gain": os.environ.get(
            "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_MIN_GAIN",
            "",
        ),
        "fused_rmsnorm_lm_head_argmax_force_use": os.environ.get(
            "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_FORCE_USE",
            "",
        ),
        "paged_decode_splits": os.environ.get("MEGAGEMM_PAGED_DECODE_SPLITS", ""),
        "paged_decode_max_splits": os.environ.get("MEGAGEMM_PAGED_DECODE_MAX_SPLITS", ""),
        "paged_decode_split_min_blocks": os.environ.get("MEGAGEMM_PAGED_DECODE_SPLIT_MIN_BLOCKS", ""),
        "paged_decode_target_warps_per_sm": os.environ.get("MEGAGEMM_PAGED_DECODE_TARGET_WARPS_PER_SM", ""),
        "paged_decode_reduce_warps": os.environ.get("MEGAGEMM_PAGED_DECODE_REDUCE_WARPS", ""),
        "paged_decode_gqa4_split": os.environ.get("MEGAGEMM_PAGED_DECODE_GQA4_SPLIT", ""),
        "paged_decode_gqa8_split": os.environ.get("MEGAGEMM_PAGED_DECODE_GQA8_SPLIT", ""),
        "paged_decode_gqa2_split": os.environ.get("MEGAGEMM_PAGED_DECODE_GQA2_SPLIT", ""),
        "paged_decode_log": os.environ.get("MEGAGEMM_PAGED_DECODE_LOG", ""),
    }
    try:
        import rmsnorm_cuda_ops  # noqa: F401

        probe["rmsnorm_cuda_ops_import"] = True
    except Exception as exc:
        probe["rmsnorm_cuda_ops_import"] = False
        probe["rmsnorm_cuda_ops_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from megagemm.kernels import rmsnorm as rmsnorm_mod

        probe["rmsnorm_kernel_cuda_available"] = bool(
            getattr(rmsnorm_mod, "_CUDA_AVAILABLE", False)
        )
        probe["rmsnorm_allow_untested_arch"] = bool(
            getattr(rmsnorm_mod, "_ALLOW_UNTESTED_CUDA_RMSNORM_ARCH", False)
        )
    except Exception as exc:
        probe["rmsnorm_kernel_probe_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def run_megagemm(args: argparse.Namespace, exact_prompts: list[str]) -> dict[str, Any]:
    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_PACKED_ATTN_UNIFORM_BATCH", "1")
    os.environ.setdefault("MEGAGEMM_PACKED_ATTN_UNIFORM_MAX_SCORE_MB", "4096")
    os.environ.setdefault("MEGAGEMM_PACKED_ATTN_UNIFORM_RESERVE_MB", "512")
    os.environ.setdefault("MEGAGEMM_PREFILL_GQA_MODE", "native")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_ROUTER_K_SPLITS", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_FUSED_GATE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DOT", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_FUSED_PACK", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N", "128")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DIRECT_OUT", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SINGLE_ROW_GEMV", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_PARTIAL_REDUCE", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_COALESCED_WEIGHTS", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS", "4")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_BLOCK_M", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_K_SPLITS", "4")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DOWN_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE_BLOCK_M", "16")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES", "4")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_INT8_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS", "257")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_DENSE_GRID", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_SCATTER", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_BLOCK", "256")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS", "4096")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_M", "32")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_N", "128")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_K", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS", "4096")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE", "512")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS", "65")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL_MIN_ASSIGNMENTS", "65")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_K", "128")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_NUM_WARPS", "4")
    os.environ.setdefault("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_MIN_GAIN", "0.0")
    os.environ.setdefault("MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_FORCE_USE", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_SPLITS", "")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_MAX_SPLITS", "")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_TARGET_WARPS_PER_SM", "32")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_WARPS", "4")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_REDUCE_WARPS", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA4_SPLIT", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA8_SPLIT", "0")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA2_SPLIT", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_LOG", "1")
    graph_enabled = os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS", "0") == "1"
    if graph_enabled:
        os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", "1")
        os.environ.setdefault("MEGAGEMM_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS", "0")
    decode_timing_enabled = (
        os.environ.get("MEGAGEMM_DECODE_TIMING", "0") == "1"
        or os.environ.get("MEGAGEMM_DECODE_TIMING_DETAIL", "0") == "1"
    )
    if decode_timing_enabled and graph_enabled:
        print(
            "  NOTE: MEGAGEMM_DECODE_TIMING is enabled, but CUDA graph replay "
            "does not expose per-op decode_timing internals. For a per-op "
            "eager breakdown, rerun with MEGAGEMM_DECODE_CUDA_GRAPHS=0."
        )
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM_MIN_ROWS", "1")

    from megagemm.engine import InferenceEngine

    runtime_probe = megagemm_runtime_probe()
    print(f"  megagemm_runtime: {runtime_probe}")

    engine = InferenceEngine(
        args.model,
        dtype=dtype_from_arg(args.dtype),
        device="cuda",
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        quantize=None if args.quantize == "none" else args.quantize,
    )
    require_dense_flat = (
        os.environ.get("MEGAGEMM_REQUIRE_QWEN3_MOE_DENSE_FLAT", "0") == "1"
        and getattr(getattr(engine, "config", None), "model_type", "") == "qwen3_moe"
    )
    if require_dense_flat and not bool(
        getattr(getattr(engine, "model", None), "_all_full_attention", False)
    ):
        raise RuntimeError(
            "Qwen3 MoE dense-flat preflight failed before warmup: "
            "model._all_full_attention is false. Refusing to spend the rest of "
            "the run on the legacy hybrid path."
        )
    if len(exact_prompts) == 1:
        exact_prompt_patch(engine, exact_prompts[0])

    megagemm_warmup_tokens = int(args.warmup_tokens)
    if graph_enabled:
        megagemm_warmup_tokens = max(megagemm_warmup_tokens, int(args.max_tokens))
    if megagemm_warmup_tokens != int(args.warmup_tokens):
        print(
            "== MegaGemm graph warmup adjusted: "
            f"{args.warmup_tokens} -> {megagemm_warmup_tokens} tokens "
            "to capture the measured decode shape before repeats =="
        )
    print(f"== MegaGemm warmup {megagemm_warmup_tokens} tokens ==")
    if len(exact_prompts) == 1:
        warmup = run_megagemm_once(engine, exact_prompts[0], megagemm_warmup_tokens, args.ignore_eos)
    else:
        warmup = run_megagemm_batch_once(engine, exact_prompts, megagemm_warmup_tokens, args.ignore_eos)
    print(warmup["raw_log"])
    if require_dense_flat:
        model = getattr(engine, "model", None)
        flat_ready = bool(getattr(model, "_flat_decode_ready", False))
        flat_hybrid = bool(getattr(model, "_flat_is_hybrid", False))
        flat_failed = bool(getattr(model, "_flat_decode_failed", False))
        print(
            "[MegaGemm] Qwen3 MoE dense-flat preflight: "
            f"all_full_attention={int(bool(getattr(model, '_all_full_attention', False)))} "
            f"flat_ready={int(flat_ready)} "
            f"flat_kind={'hybrid' if flat_hybrid else 'dense'} "
            f"flat_failed={int(flat_failed)}"
        )
        if not flat_ready or flat_hybrid or flat_failed:
            reason = str(getattr(model, "_flat_decode_failed_reason", ""))
            raise RuntimeError(
                "Qwen3 MoE dense-flat preflight failed after warmup: "
                f"flat_ready={flat_ready}, flat_hybrid={flat_hybrid}, "
                f"flat_failed={flat_failed}, reason={reason!r}. Refusing to run "
                "the measured repeats or vLLM comparison."
            )

    samples = []
    for repeat_idx in range(args.repeats):
        print(f"== MegaGemm repeat {repeat_idx + 1}/{args.repeats} ==")
        if len(exact_prompts) == 1:
            row = run_megagemm_once(engine, exact_prompts[0], args.max_tokens, args.ignore_eos)
        else:
            row = run_megagemm_batch_once(engine, exact_prompts, args.max_tokens, args.ignore_eos)
        row["repeat"] = repeat_idx + 1
        samples.append(row)
        print(row["raw_log"])
    profile_summary = None
    if bool(getattr(args, "profile_decode_breakdown", False)):
        print("== MegaGemm profile_decode_breakdown ==")
        profile_prompt: str | list[str] = exact_prompts[0] if len(exact_prompts) == 1 else exact_prompts
        profile_summary = engine.profile_decode_breakdown(
            profile_prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            ignore_eos=bool(args.ignore_eos),
        )
        print(json.dumps(profile_summary, indent=2, sort_keys=True))
    result = {
        "backend": "megagemm",
        "batch_size": len(exact_prompts),
        "warmup_tokens_requested": int(args.warmup_tokens),
        "warmup_tokens_effective": int(megagemm_warmup_tokens),
        "graph_shape_warmup": bool(graph_enabled and megagemm_warmup_tokens != int(args.warmup_tokens)),
        "warmup": warmup,
        "samples": samples,
        "megagemm_runtime": runtime_probe,
    }
    if profile_summary is not None:
        result["profile_decode_breakdown"] = profile_summary
    return result


def make_vllm_llm(args: argparse.Namespace):
    runtime = preload_vllm_cuda_runtime()
    try:
        from vllm import LLM
    except Exception as exc:
        hint = ""
        if "libcudart.so.13" in str(exc):
            hint = (
                " The installed vLLM wheel expects CUDA 13, but this Python "
                "environment only exposes CUDA 12 runtime libraries. Reinstall "
                "vLLM for CUDA 12.9, for example: "
                "python -m pip install -q -U uv && "
                "uv pip install --system --reinstall vllm --torch-backend=cu129."
            )
        raise RuntimeError(
            f"vLLM import failed ({type(exc).__name__}: {exc}). "
            f"CUDA runtime preload: {runtime}.{hint}"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": vllm_dtype_from_arg(args.dtype),
        "trust_remote_code": True,
        "max_model_len": args.max_seq_len,
        "max_num_seqs": max(1, int(args.batch_size)),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "enable_prefix_caching": bool(args.vllm_prefix_caching),
        "disable_log_stats": True,
    }
    if args.vllm_enforce_eager:
        kwargs["enforce_eager"] = True
    if args.cache_dir:
        kwargs["download_dir"] = args.cache_dir

    try:
        return LLM(**kwargs), runtime, kwargs
    except TypeError:
        for key in ("trust_remote_code", "enable_prefix_caching", "disable_log_stats", "max_num_seqs"):
            kwargs.pop(key, None)
        return LLM(**kwargs), runtime, kwargs


def run_vllm_once(llm, exact_prompts: list[str], max_tokens: int, ignore_eos: bool) -> dict[str, Any]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=bool(ignore_eos),
    )
    sync_cuda()
    start = time.perf_counter()
    try:
        outputs = llm.generate(exact_prompts, sampling, use_tqdm=False)
    except TypeError:
        outputs = llm.generate(exact_prompts, sampling)
    sync_cuda()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    generated_tokens = 0
    text_prefixes = []
    metrics_dicts: list[dict[str, float]] = []
    first_token_times = []
    finished_times = []
    for request_output in outputs:
        candidate = request_output.outputs[0]
        token_ids = getattr(candidate, "token_ids", None)
        request_tokens = len(token_ids) if token_ids is not None else 0
        if request_tokens <= 0 and ignore_eos:
            request_tokens = max_tokens
        generated_tokens += request_tokens
        text_prefixes.append(str(getattr(candidate, "text", ""))[:120])

        metrics = getattr(request_output, "metrics", None)
        one_metrics: dict[str, float] = {}
        if metrics is not None:
            for name in (
                "arrival_time",
                "first_scheduled_time",
                "first_token_time",
                "finished_time",
                "time_in_queue",
            ):
                value = getattr(metrics, name, None)
                if value is not None:
                    one_metrics[name] = float(value)
            if "first_token_time" in one_metrics:
                first_token_times.append(one_metrics["first_token_time"])
            if "finished_time" in one_metrics:
                finished_times.append(one_metrics["finished_time"])
        metrics_dicts.append(one_metrics)

    expected_tokens = int(max_tokens) * len(exact_prompts)
    if ignore_eos and generated_tokens != expected_tokens:
        raise RuntimeError(
            "vLLM fixed-length benchmark produced "
            f"{generated_tokens} tokens; expected exactly {expected_tokens}"
        )

    metrics_dict: dict[str, Any] = {}
    decode_ms = None
    decode_tokens = None
    decode_tok_s = None
    if first_token_times and finished_times and generated_tokens > len(outputs):
        metrics_dict["first_token_time_min"] = min(first_token_times)
        metrics_dict["finished_time_max"] = max(finished_times)
        decode_ms = max(0.0, (max(finished_times) - min(first_token_times)) * 1000.0)
        decode_tokens = generated_tokens - len(outputs)
        if decode_ms > 0.0:
            decode_tok_s = decode_tokens / (decode_ms / 1000.0)

    return {
        "total_ms": elapsed_ms,
        "output_tokens": generated_tokens,
        "output_tok_s_total": generated_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0.0 else 0.0,
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "decode_tok_s": decode_tok_s,
        "vllm_request_metrics": metrics_dict,
        "vllm_request_metrics_per_request": metrics_dicts,
        "batch_size": len(exact_prompts),
        "text_prefix": text_prefixes[0] if text_prefixes else "",
        "text_prefixes": text_prefixes[:3],
    }


def run_vllm(args: argparse.Namespace, exact_prompts: list[str]) -> dict[str, Any]:
    llm, runtime, llm_kwargs = make_vllm_llm(args)

    print(f"== vLLM warmup {args.warmup_tokens} tokens ==")
    warmup = run_vllm_once(llm, exact_prompts, args.warmup_tokens, args.ignore_eos)
    print(json.dumps(warmup, sort_keys=True))
    delta_probe_warmup = None
    if args.vllm_decode_delta_probe and args.max_tokens > 1:
        print("== vLLM decode delta probe warmup ==")
        delta_probe_warmup = run_vllm_once(llm, exact_prompts, 1, args.ignore_eos)
        delta_probe_warmup["probe_tokens"] = 1
        print(json.dumps(delta_probe_warmup, sort_keys=True))

    samples = []
    delta_probes = []
    for repeat_idx in range(args.repeats):
        delta_probe = None
        if args.vllm_decode_delta_probe and args.max_tokens > 1:
            print(f"== vLLM decode delta probe {repeat_idx + 1}/{args.repeats} ==")
            delta_probe = run_vllm_once(llm, exact_prompts, 1, args.ignore_eos)
            delta_probe["repeat"] = repeat_idx + 1
            delta_probe["probe_tokens"] = 1
            delta_probes.append(delta_probe)
            print(json.dumps(delta_probe, sort_keys=True))

        print(f"== vLLM repeat {repeat_idx + 1}/{args.repeats} ==")
        row = run_vllm_once(llm, exact_prompts, args.max_tokens, args.ignore_eos)
        row["repeat"] = repeat_idx + 1
        if delta_probe is not None:
            delta_ms = float(row["total_ms"]) - float(delta_probe["total_ms"])
            delta_tokens = int(row["output_tokens"]) - int(delta_probe["output_tokens"])
            if delta_ms > 0.0 and delta_tokens > 0:
                row["decode_ms"] = delta_ms
                row["decode_tokens"] = delta_tokens
                row["decode_tok_s"] = delta_tokens / (delta_ms / 1000.0)
                row["decode_estimate_method"] = "total_delta_vs_1_token_probe"
                row["decode_delta_probe"] = {
                    "probe_tokens": 1,
                    "total_ms": float(delta_probe["total_ms"]),
                    "output_tokens": int(delta_probe["output_tokens"]),
                    "output_tok_s_total": float(delta_probe["output_tok_s_total"]),
                }
        samples.append(row)
        print(json.dumps(row, sort_keys=True))

    return {
        "backend": "vllm",
        "batch_size": len(exact_prompts),
        "warmup": warmup,
        "decode_delta_probe_warmup": delta_probe_warmup,
        "decode_delta_probes": delta_probes,
        "samples": samples,
        "vllm_cuda_runtime": runtime,
        "vllm_kwargs": llm_kwargs,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [float(row["decode_tok_s"]) for row in samples if row.get("decode_tok_s") is not None]
    decode_ms = [float(row["decode_ms"]) for row in samples if row.get("decode_ms") is not None]
    prefill_ms = [float(row["prefill_ms"]) for row in samples if row.get("prefill_ms") is not None]
    total_ms = [float(row["total_ms"]) for row in samples if row.get("total_ms") is not None]
    non_decode_ms = [
        float(row["non_decode_ms"])
        for row in samples
        if row.get("non_decode_ms") is not None
    ]
    non_decode_ms_estimated = [
        max(0.0, float(row["total_ms"]) - float(row["decode_ms"]))
        for row in samples
        if row.get("total_ms") is not None and row.get("decode_ms") is not None
    ]
    scheduler_overhead_ms = [
        float(row["scheduler_overhead_ms"])
        for row in samples
        if row.get("scheduler_overhead_ms") is not None
    ]
    output_total = [
        float(row["output_tok_s_total"])
        for row in samples
        if row.get("output_tok_s_total") is not None
    ]
    summary: dict[str, Any] = {
        "output_tok_s_total_median": statistics.median(output_total) if output_total else None,
        "output_tok_s_total_mean": statistics.mean(output_total) if output_total else None,
        "decode_ms_median": statistics.median(decode_ms) if decode_ms else None,
        "prefill_ms_median": statistics.median(prefill_ms) if prefill_ms else None,
        "total_ms_median": statistics.median(total_ms) if total_ms else None,
        "non_decode_ms_median": statistics.median(non_decode_ms) if non_decode_ms else None,
        "non_decode_ms_estimated_median": (
            statistics.median(non_decode_ms_estimated) if non_decode_ms_estimated else None
        ),
        "scheduler_overhead_ms_median": (
            statistics.median(scheduler_overhead_ms) if scheduler_overhead_ms else None
        ),
    }
    decode_methods = sorted(
        {
            str(row.get("decode_estimate_method"))
            for row in samples
            if row.get("decode_estimate_method")
        }
    )
    if decode_methods:
        summary["decode_estimate_method"] = ",".join(decode_methods)
    if speeds:
        summary.update(
            {
                "decode_tok_s_median": statistics.median(speeds),
                "decode_tok_s_mean": statistics.mean(speeds),
            }
        )
    else:
        summary.update({"decode_tok_s_median": None, "decode_tok_s_mean": None})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["megagemm", "vllm"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--quantize", choices=["none", "int8", "int4", "awq"], default="none")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt", default="Write a complete Python function that computes Fibonacci numbers iteratively, with a short explanation.")
    parser.add_argument(
        "--prompt-token-target",
        type=int,
        default=0,
        help=(
            "Use a deterministic synthetic prompt near this exact token count. "
            "Useful for long-prefill sweeps; bypasses chat-template sizing."
        ),
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--vllm-decode-delta-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate vLLM decode-only throughput by subtracting a 1-token same-batch "
            "probe from each measured max-token run when vLLM request metrics are empty."
        ),
    )
    parser.add_argument("--decode-timing", action="store_true", help="Enable MegaGemm decode timing events.")
    parser.add_argument("--decode-timing-detail", action="store_true", help="Enable detailed MegaGemm per-op decode timing.")
    parser.add_argument(
        "--decode-timing-print",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print MegaGemm decode timing lines in captured logs.",
    )
    parser.add_argument("--prefill-timing", action="store_true", help="Enable MegaGemm prefill timing events.")
    parser.add_argument(
        "--prefill-timing-print",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print MegaGemm prefill timing lines in captured logs.",
    )
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Bypass the Qwen3-Coder-30B-A3B BF16/FP16 VRAM guard.",
    )
    parser.add_argument(
        "--profile-decode-breakdown",
        action="store_true",
        help="Run InferenceEngine.profile_decode_breakdown after MegaGemm graph warmup.",
    )
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    decode_timing_enabled = bool(args.decode_timing or args.decode_timing_detail)
    if decode_timing_enabled:
        os.environ["MEGAGEMM_DECODE_TIMING"] = "1"
        if args.decode_timing_detail:
            os.environ["MEGAGEMM_DECODE_TIMING_DETAIL"] = "1"
        else:
            os.environ.setdefault("MEGAGEMM_DECODE_TIMING_DETAIL", "0")
        if args.decode_timing_print is None:
            os.environ.setdefault("MEGAGEMM_DECODE_TIMING_PRINT", "1")
        else:
            os.environ["MEGAGEMM_DECODE_TIMING_PRINT"] = "1" if args.decode_timing_print else "0"
    elif args.decode_timing_print is not None:
        os.environ["MEGAGEMM_DECODE_TIMING_PRINT"] = "1" if args.decode_timing_print else "0"
    if args.prefill_timing:
        os.environ["MEGAGEMM_PREFILL_TIMING"] = "1"
        if args.prefill_timing_print is None:
            os.environ.setdefault("MEGAGEMM_PREFILL_TIMING_PRINT", "1")
        else:
            os.environ["MEGAGEMM_PREFILL_TIMING_PRINT"] = "1" if args.prefill_timing_print else "0"
    elif args.prefill_timing_print is not None:
        os.environ["MEGAGEMM_PREFILL_TIMING_PRINT"] = "1" if args.prefill_timing_print else "0"

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_batch_size < args.batch_size:
        args.max_batch_size = args.batch_size

    preflight_qwen3_moe_vram(args)

    tokenizer = load_tokenizer(args.model, args.cache_dir or None)
    exact_prompts, prompt_token_counts = build_prompt_batch(
        tokenizer,
        args.prompt,
        use_chat_template=not args.no_chat_template,
        batch_size=args.batch_size,
        prompt_token_target=max(0, int(args.prompt_token_target)),
    )
    prompt_tokens = prompt_token_counts[0] if prompt_token_counts else 0
    if args.prompt_token_target > 0 and prompt_tokens + args.max_tokens > args.max_seq_len:
        new_max_seq_len = int(prompt_tokens + args.max_tokens)
        print(
            "  AUTO: prompt_token_target + max_tokens exceeds max_seq_len "
            f"({prompt_tokens} + {args.max_tokens} > {args.max_seq_len}). "
            f"Bumping max_seq_len to {new_max_seq_len} for this run."
        )
        args.max_seq_len = new_max_seq_len

    print("Qwen3 MoE backend compare")
    print(f"  backend:       {args.backend}")
    print(f"  model:         {args.model}")
    print(f"  dtype:         {args.dtype}")
    print(f"  quantize:      {args.quantize}")
    print(f"  max_seq_len:   {args.max_seq_len}")
    print(f"  batch_size:    {args.batch_size}")
    print(
        "  prompt_tokens: "
        f"min={min(prompt_token_counts) if prompt_token_counts else 0} "
        f"max={max(prompt_token_counts) if prompt_token_counts else 0}"
    )
    print(f"  max_tokens:    {args.max_tokens}")
    print(f"  ignore_eos:    {args.ignore_eos}")
    print(f"  gpu:           {gpu_snapshot()}")

    if args.backend == "megagemm":
        backend_result = run_megagemm(args, exact_prompts)
    else:
        backend_result = run_vllm(args, exact_prompts)

    result = {
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype,
        "quantize": args.quantize,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "prompt_tokens": prompt_tokens,
        "prompt_token_target": int(args.prompt_token_target),
        "prompt_token_counts": prompt_token_counts,
        "prompt_tokens_min": min(prompt_token_counts) if prompt_token_counts else None,
        "prompt_tokens_max": max(prompt_token_counts) if prompt_token_counts else None,
        "warmup_tokens": args.warmup_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "vllm_decode_delta_probe": bool(args.vllm_decode_delta_probe),
        "ignore_eos": args.ignore_eos,
        "gpu": gpu_snapshot(),
        **backend_result,
    }
    result["summary"] = summarize(backend_result["samples"])

    print("== SUMMARY ==")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
