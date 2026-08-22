#!/usr/bin/env python3
"""
Sweep Qwen3 MoE grouped decode kernel configs with one model load.

Example:
  python benchmarks/run_qwen3_moe_grouped_sweep.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --configs 32x64x4,64x64x4,64x128x4,128x64x4 \
    --include-baseline
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_TIMING_RE = re.compile(
    r"Prefill:\s*([0-9.]+)ms\s*\((\d+)\s+tokens\)\s*\|\s*"
    r"Decode:\s*([0-9.]+)ms\s*\((\d+)\s+tokens\)\s*\|\s*"
    r"Speed:\s*([0-9.]+)\s+tok/s"
)
_BATCH_TIMING_RE = re.compile(
    r"Batch complete:\s*(\d+)\s+prompts,\s*(\d+)\s+tokens\s+in\s+([0-9.]+)ms\s*\|\s*"
    r"Throughput:\s*([0-9.]+)\s+tok/s"
)


def _parse_config(raw: str) -> tuple[int, int, int, int]:
    parts = raw.lower().replace(":", "x").split("x")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"Invalid config '{raw}'. Use BLOCK_NxBLOCK_KxWARPS or BLOCK_NxBLOCK_KxWARPSxSTAGES."
        )
    block_n, block_k, warps = (int(parts[0]), int(parts[1]), int(parts[2]))
    stages = int(parts[3]) if len(parts) == 4 else 2
    if min(block_n, block_k, warps, stages) <= 0:
        raise argparse.ArgumentTypeError(f"Invalid non-positive config value in '{raw}'")
    return block_n, block_k, warps, stages


def _parse_configs(raw: str) -> list[tuple[int, int, int, int]]:
    configs = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            configs.append(_parse_config(part))
    if not configs:
        raise argparse.ArgumentTypeError("At least one config is required")
    return configs


def _parse_compact_config(raw: str) -> tuple[int, int, int, int, int]:
    parts = raw.lower().replace(":", "x").split("x")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            f"Invalid compact config '{raw}'. Use GATExDOWNxBLOCK_KxWARPS or "
            "GATExDOWNxBLOCK_KxWARPSxSTAGES."
        )
    gate_block_n, down_block_n, block_k, warps = (int(parts[idx]) for idx in range(4))
    stages = int(parts[4]) if len(parts) == 5 else 2
    if min(gate_block_n, down_block_n, block_k, warps, stages) <= 0:
        raise argparse.ArgumentTypeError(f"Invalid non-positive compact config value in '{raw}'")
    return gate_block_n, down_block_n, block_k, warps, stages


def _parse_compact_configs(raw: str) -> list[tuple[int, int, int, int, int]]:
    configs = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            configs.append(_parse_compact_config(part))
    if not configs:
        raise argparse.ArgumentTypeError("At least one compact config is required")
    return configs


def _run_generate(
    engine,
    prompts: list[str],
    max_tokens: int,
    *,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
) -> tuple[str, dict[str, Any]]:
    buffer = io.StringIO()
    eos_token_id = getattr(engine.tokenizer, "eos_token_id", None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        if len(prompts) == 1:
            engine.tokenizer.eos_token_id = None
        with contextlib.redirect_stdout(buffer):
            if len(prompts) == 1:
                text = engine.generate(
                    prompts[0],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty,
                    verbose=True,
                )
            else:
                text = engine.generate_batch(
                    prompts,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=0.9,
                    ignore_eos=True,
                    verbose=True,
                    decode_outputs=False,
                )
    finally:
        if len(prompts) == 1:
            engine.tokenizer.eos_token_id = eos_token_id
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log = buffer.getvalue()
    match = _TIMING_RE.search(log)
    batch_match = _BATCH_TIMING_RE.search(log)
    row: dict[str, Any] = {
        "elapsed_ms": elapsed_ms,
        "batch_size": len(prompts),
        "text_prefix": str(text)[:120],
        "raw_log": log.strip(),
    }
    if match:
        decode_ms = float(match.group(3))
        decode_tokens = int(match.group(4))
        row.update(
            {
                "prefill_ms": float(match.group(1)),
                "prompt_tokens": int(match.group(2)),
                "decode_ms": decode_ms,
                "decode_tokens": decode_tokens,
                "tok_s": (
                    decode_tokens / (decode_ms / 1000.0)
                    if decode_ms > 0.0
                    else float(match.group(5))
                ),
            }
        )
    elif batch_match:
        scheduler = getattr(engine, "_last_scheduler", None)
        stats = scheduler.get_stats() if scheduler is not None else {}
        output_tokens = int(batch_match.group(2))
        decode_ms = float(stats.get("decode_time_ms", 0.0) or 0.0)
        decode_tokens = max(0, output_tokens - int(batch_match.group(1)))
        row.update(
            {
                "prefill_ms": float(stats.get("prefill_time_ms", 0.0) or 0.0),
                "prompt_tokens": None,
                "decode_ms": decode_ms if decode_ms > 0.0 else None,
                "decode_tokens": decode_tokens,
                "tok_s": (
                    decode_tokens / (decode_ms / 1000.0)
                    if decode_ms > 0.0
                    else float(batch_match.group(4))
                ),
                "output_tokens_total": output_tokens,
                "output_tok_s_total": float(batch_match.group(4)),
                "scheduler_stats": stats,
            }
        )
    else:
        row.update(
            {
                "prefill_ms": None,
                "prompt_tokens": None,
                "decode_ms": None,
                "decode_tokens": None,
                "tok_s": None,
            }
        )
    diagnostics = _megagemm_diagnostics(engine)
    if diagnostics:
        row["diagnostics"] = diagnostics
    return log, row


def _require_fixed_decode_length(row: dict[str, Any], expected: int, label: str) -> None:
    actual = row.get("decode_tokens")
    if actual is None:
        raise RuntimeError(f"{label}: benchmark log did not expose decode token count")
    if int(actual) != int(expected):
        raise RuntimeError(
            f"{label}: expected exactly {expected} decode tokens with EOS disabled, got {actual}"
        )


def _megagemm_diagnostics(engine) -> dict[str, Any]:
    model = getattr(engine, "model", None)
    diagnostics: dict[str, Any] = {}
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
    return diagnostics


def _clear_decode_graph_cache(engine) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generate_states = getattr(engine, "_generate_multi_step_graph_states", None)
    if isinstance(generate_states, dict):
        generate_states.clear()
    if hasattr(engine, "_generate_graph_log_count"):
        try:
            engine._generate_graph_log_count = 0
        except Exception:
            pass
    block_manager = getattr(engine, "block_manager", None)
    if block_manager is not None:
        try:
            setattr(
                block_manager,
                "_decode_graph_shape_cache",
                {"states": {}, "warm_keys": set(), "failed_keys": set()},
            )
        except Exception:
            pass
    scheduler = getattr(engine, "_last_scheduler", None)
    if scheduler is not None:
        invalidate = getattr(scheduler, "_invalidate_decode_graph", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:
                pass
        for attr, value in (
            ("_decode_graph_shape_states", {}),
            ("_decode_graph_shape_warm_keys", set()),
            ("_decode_graph_shape_failed_keys", set()),
        ):
            if hasattr(scheduler, attr):
                try:
                    setattr(scheduler, attr, value)
                except Exception:
                    pass


def _timing_brief(row: dict[str, Any]) -> str:
    timing = row.get("diagnostics", {}).get("last_decode_timing")
    if not timing:
        return ""
    return "TIMING " + " | ".join(
        [
            f"path={timing.get('decode_path', 'unknown')}",
            f"ms/token={float(timing.get('ms_per_token', 0.0)):.3f}",
            f"attn={float(timing.get('attn_ms', 0.0)):.1f}ms",
            f"moe={float(timing.get('flat_moe_ms', timing.get('mlp_ms', 0.0))):.1f}ms",
            f"router={float(timing.get('moe_router_ms', 0.0)):.1f}ms",
            f"experts={float(timing.get('moe_experts_ms', 0.0)):.1f}ms",
            f"lm_head={float(timing.get('lm_head_ms', 0.0)):.1f}ms",
        ]
    )


def _looks_like_qwen3_coder_30b_a3b(model: str) -> bool:
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


def _preflight_qwen3_moe_vram(args: argparse.Namespace) -> None:
    if args.device != "cuda" or args.dtype not in ("bf16", "fp16"):
        return
    if args.allow_low_vram or os.environ.get("ALLOW_LOW_VRAM", "0") == "1":
        return
    if not torch.cuda.is_available() or not _looks_like_qwen3_coder_30b_a3b(args.model):
        return
    min_vram_gb = float(os.environ.get("QWEN3_MOE_MIN_VRAM_GB", "64"))
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    if vram_gb >= min_vram_gb:
        return
    name = torch.cuda.get_device_name(0)
    raise SystemExit(
        "Refusing to load Qwen3-Coder-30B-A3B in BF16/FP16 on this GPU: "
        f"{name} has {vram_gb:.2f} GiB, but this path needs at least "
        f"{min_vram_gb:.0f} GiB. The model weights alone are about 58 GiB "
        "before KV/workspace. Use A100-80GB, RTX PRO 6000 Blackwell 96GB, "
        "or set ALLOW_LOW_VRAM=1 only for an explicitly quantized/offload experiment."
    )


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [float(s["tok_s"]) for s in samples if s.get("tok_s") is not None]
    decodes = [float(s["decode_ms"]) for s in samples if s.get("decode_ms") is not None]
    if not speeds:
        return {"tok_s_median": None, "decode_ms_median": None}
    return {
        "tok_s_median": statistics.median(speeds),
        "tok_s_mean": statistics.mean(speeds),
        "decode_ms_median": statistics.median(decodes) if decodes else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--configs",
        type=_parse_configs,
        default=_parse_configs("32x64x4,64x64x4,64x128x4,128x64x4,32x128x4,64x64x8"),
        help="Comma-separated BLOCK_NxBLOCK_KxWARPS[xSTAGES] configs.",
    )
    parser.add_argument("--max-assignments", type=int, default=64)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument(
        "--include-router-ab",
        action="store_true",
        help="Also compare the default grouped kernel with fused router disabled/enabled.",
    )
    parser.add_argument(
        "--include-router-k-split-ab",
        action="store_true",
        help="At batch 1, compare fused router K splits 1, 2, 4, and 8.",
    )
    parser.add_argument(
        "--include-token-accum-ab",
        action="store_true",
        help="Also compare the grouped down kernel with token-owned accumulation disabled/enabled.",
    )
    parser.add_argument(
        "--include-expert-grouped-ab",
        action="store_true",
        help="Also compare the current grouped decode path against the dense expert-grouped decode path.",
    )
    parser.add_argument(
        "--include-expert-grouped-compact-ab",
        action="store_true",
        help="Also compare the current grouped decode path against the compact active-expert grouped decode path.",
    )
    parser.add_argument(
        "--include-grouped-dot-ab",
        action="store_true",
        help="Also compare scalar grouped GEMV against tl.dot grouped decode with CUDA graph capture allowed.",
    )
    parser.add_argument(
        "--include-single-row-gemv-ab",
        action="store_true",
        help="At batch 1, compare the shared-route tl.dot path against the scalar grouped GEMV path.",
    )
    parser.add_argument(
        "--include-shared-route-block-m-ab",
        action="store_true",
        help="At batch 1, compare shared-route tl.dot with physical BLOCK_M 1 versus 16.",
    )
    parser.add_argument(
        "--include-shared-route-partial-ab",
        action="store_true",
        help=(
            "Compare shared-route expert atomic accumulation against per-expert partials "
            "followed by a final reduction."
        ),
    )
    parser.add_argument(
        "--include-shared-route-layout-ab",
        action="store_true",
        help="Compare the legacy strided shared-route weight load against coalesced [N,K] loads.",
    )
    parser.add_argument(
        "--include-shared-route-tiles-ab",
        action="store_true",
        help="Compare shared-route gate/down tiles 64/64 against the asymmetric 64/128 layout.",
    )
    parser.add_argument(
        "--include-shared-route-split-gate-ab",
        action="store_true",
        help="Compare fused gate+SwiGLU against the low-register split gate/up assignment path.",
    )
    parser.add_argument(
        "--include-shared-route-gate-k-split-ab",
        action="store_true",
        help="At batch 1, compare shared-route gate/up K splits 1, 2, and 4.",
    )
    parser.add_argument(
        "--no-config-cases",
        action="store_true",
        help="Skip the plain --configs cases; useful for focused A/B runs that avoid extra model time.",
    )
    parser.add_argument(
        "--compact-configs",
        type=_parse_compact_configs,
        default=[],
        help=(
            "Comma-separated compact decode configs as GATExDOWNxBLOCK_KxWARPS[xSTAGES]. "
            "These force the safe compact expert-grouped decode path with active-list/direct-out off."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Write a complete Python function that computes Fibonacci numbers iteratively, with a short explanation.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--decode-timing", action="store_true", help="Enable MegaGemm decode timing events.")
    parser.add_argument("--decode-timing-detail", action="store_true", help="Enable detailed per-op decode timing events.")
    parser.add_argument(
        "--profile-decode-breakdown",
        action="store_true",
        help="After warmup, run torch.profiler on the stable fused-gate decode path.",
    )
    parser.add_argument(
        "--profile-tokens",
        type=int,
        default=16,
        help="Generated tokens used by --profile-decode-breakdown.",
    )
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Bypass the Qwen3-Coder-30B-A3B BF16/FP16 VRAM guard.",
    )
    parser.add_argument(
        "--decode-timing-print",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print MegaGemm decode timing lines in captured logs.",
    )
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

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_batch_size < args.batch_size:
        args.max_batch_size = args.batch_size

    _preflight_qwen3_moe_vram(args)

    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DEBUG", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_MAX_ASSIGNMENTS", str(args.max_assignments))

    from megagemm.engine import InferenceEngine
    import megagemm.kernels.qwen3_moe as moe_kernel
    import megagemm.models.llama as llama_mod

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print("== GPU ==")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    else:
        print("cuda unavailable")

    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )
    prompt_width = max(2, len(str(args.batch_size)))
    prompts = [
        args.prompt
        if args.batch_size == 1
        else f"{args.prompt}\n\nRequest marker: {idx:0{prompt_width}d}."
        for idx in range(args.batch_size)
    ]

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    cases: list[
        tuple[
            str,
            tuple[int, int, int, int] | None,
            bool,
            bool | None,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool | None,
            bool | None,
            bool | None,
            tuple[int, int, int, int, int] | None,
        ]
    ] = []
    if args.include_router_ab:
        default_cfg = (64, 64, 8, 2)
        cases.append(("router_off_bn64_bk64_w8_s2", default_cfg, False, None, False, False, False, False, False, None, None, None, None))
        cases.append(("router_on_bn64_bk64_w8_s2", default_cfg, True, None, False, False, False, False, False, None, None, None, None))
    if args.include_router_k_split_ab:
        if args.batch_size != 1:
            raise ValueError("--include-router-k-split-ab requires --batch-size 1")
        default_cfg = (64, 128, 4, 2)
        cases.append(("router_k_split1", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("router_k_split2", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("router_k_split4", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("router_k_split8", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_token_accum_ab:
        default_cfg = (64, 64, 8, 2)
        cases.append(("token_accum_off_bn64_bk64_w8_s2", default_cfg, True, False, False, False, False, False, False, None, None, None, None))
        cases.append(("token_accum_on_bn64_bk64_w8_s2", default_cfg, True, True, False, False, False, False, False, None, None, None, None))
    if args.include_expert_grouped_ab:
        default_cfg = (64, 64, 8, 2)
        cases.append(("expert_grouped_off_bn64_bk64_w8_s2", default_cfg, True, True, False, False, False, False, False, None, None, None, None))
        cases.append(("expert_grouped_dense_bn64_bk64_w8_s2", default_cfg, True, True, True, True, False, False, False, None, None, None, None))
    if args.include_expert_grouped_compact_ab:
        default_cfg = (64, 64, 8, 2)
        cases.append(("expert_grouped_compact_off_bn64_bk64_w8_s2", default_cfg, True, True, False, False, False, False, False, None, None, None, None))
        cases.append(("expert_grouped_compact_on_bn64_bk64_w8_s2", default_cfg, True, True, True, False, False, True, False, None, None, None, None))
    if args.include_grouped_dot_ab:
        default_cfg = (64, 64, 8, 2)
        cases.append(("grouped_dot_off_bn64_bk64_w8_s2", default_cfg, True, True, False, False, False, False, False, False, False, None, None))
        cases.append(("grouped_dot_on_bn64_bk64_w8_s2", default_cfg, True, True, False, False, False, False, False, True, True, None, None))
    if args.include_single_row_gemv_ab:
        if args.batch_size != 1:
            raise ValueError("--include-single-row-gemv-ab requires --batch-size 1")
        default_cfg = (64, 128, 4, 2)
        cases.append(("single_row_shared_route", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("single_row_gemv", default_cfg, True, True, True, False, False, True, True, None, None, True, None))
    if args.include_shared_route_block_m_ab:
        if args.batch_size != 1:
            raise ValueError("--include-shared-route-block-m-ab requires --batch-size 1")
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_block_m1", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_block_m16", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_shared_route_partial_ab:
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_atomic", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_partial", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_shared_route_layout_ab:
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_strided_weights", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_coalesced_weights", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_shared_route_tiles_ab:
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_tiles_64x64", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_tiles_64x128", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_shared_route_split_gate_ab:
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_fused_gate", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_split_gate", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_shared_route_gate_k_split_ab:
        if args.batch_size != 1:
            raise ValueError("--include-shared-route-gate-k-split-ab requires --batch-size 1")
        default_cfg = (64, 128, 4, 2)
        cases.append(("shared_route_gate_k_split1", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_gate_k_split2", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
        cases.append(("shared_route_gate_k_split4", default_cfg, True, True, True, False, False, True, True, None, None, False, None))
    if args.include_baseline:
        cases.append(("baseline", None, True, None, False, False, False, False, False, None, None, None, None))
    if not args.no_config_cases:
        for cfg in args.configs:
            cases.append((f"bn{cfg[0]}_bk{cfg[1]}_w{cfg[2]}_s{cfg[3]}", cfg, True, None, False, False, False, False, False, None, None, None, None))
    for compact_cfg in args.compact_configs:
        gate_block_n, down_block_n, block_k, warps, stages = compact_cfg
        label = f"compact_g{gate_block_n}_d{down_block_n}_bk{block_k}_w{warps}_s{stages}"
        cfg = (gate_block_n, block_k, warps, stages)
        cases.append((label, cfg, True, True, True, False, False, True, False, None, None, None, compact_cfg))

    original_token_accum = getattr(moe_kernel, "_CFG_TOKEN_ACCUM", False)
    original_router_k_splits = getattr(moe_kernel, "_CFG_ROUTER_K_SPLITS", 1)
    original_expert_grouped = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_DECODE", False)
    original_expert_grouped_dense = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_DENSE_DECODE", False)
    original_expert_grouped_general = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_GENERAL_DECODE", False)
    original_expert_grouped_compact = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_DECODE", False)
    original_compact_fused_pack = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK", True)
    original_compact_partial_reduce = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE", False)
    original_compact_active_list = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST", False)
    original_compact_token_accum = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM", False)
    original_compact_gate_block_n = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N", 64)
    original_compact_down_block_n = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N", 128)
    original_compact_direct_out = getattr(moe_kernel, "_CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT", False)
    original_shared_route = getattr(moe_kernel, "_CFG_SHARED_ROUTE_DECODE", False)
    original_shared_route_block_m = getattr(moe_kernel, "_CFG_SHARED_ROUTE_BLOCK_M", 1)
    original_shared_route_gate_block_n = getattr(
        moe_kernel,
        "_CFG_SHARED_ROUTE_GATE_BLOCK_N",
        64,
    )
    original_shared_route_gate_k_splits = getattr(
        moe_kernel,
        "_CFG_SHARED_ROUTE_GATE_K_SPLITS",
        1,
    )
    original_shared_route_down_block_n = getattr(
        moe_kernel,
        "_CFG_SHARED_ROUTE_DOWN_BLOCK_N",
        64,
    )
    original_shared_route_split_gate = getattr(moe_kernel, "_CFG_SHARED_ROUTE_SPLIT_GATE", False)
    original_shared_route_partial_reduce = getattr(moe_kernel, "_CFG_SHARED_ROUTE_PARTIAL_REDUCE", False)
    original_shared_route_coalesced_weights = getattr(
        moe_kernel,
        "_CFG_SHARED_ROUTE_COALESCED_WEIGHTS",
        True,
    )
    original_shared_route_token_accum = getattr(moe_kernel, "_CFG_SHARED_ROUTE_TOKEN_ACCUM", False)
    original_single_row_gemv = getattr(moe_kernel, "_CFG_SINGLE_ROW_GEMV", False)
    original_grouped_dot = getattr(moe_kernel, "_CFG_GROUPED_DOT", False)
    original_grouped_dot_allow_graphs = getattr(moe_kernel, "_CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS", False)
    for (
        label,
        cfg,
        fused_router,
        token_accum,
        expert_grouped,
        expert_grouped_dense,
        expert_grouped_general,
        expert_grouped_compact,
        shared_route,
        grouped_dot,
        grouped_dot_allow_graphs,
        single_row_gemv,
        compact_cfg,
    ) in cases:
        moe_kernel._CFG_FUSED_ROUTER = bool(fused_router)
        if label == "router_k_split1":
            moe_kernel._CFG_ROUTER_K_SPLITS = 1
        elif label == "router_k_split2":
            moe_kernel._CFG_ROUTER_K_SPLITS = 2
        elif label == "router_k_split4":
            moe_kernel._CFG_ROUTER_K_SPLITS = 4
        elif label == "router_k_split8":
            moe_kernel._CFG_ROUTER_K_SPLITS = 8
        else:
            moe_kernel._CFG_ROUTER_K_SPLITS = int(original_router_k_splits)
        if token_accum is None:
            moe_kernel._CFG_TOKEN_ACCUM = bool(original_token_accum)
        else:
            moe_kernel._CFG_TOKEN_ACCUM = bool(token_accum)
        if grouped_dot is None:
            moe_kernel._CFG_GROUPED_DOT = bool(original_grouped_dot)
        else:
            moe_kernel._CFG_GROUPED_DOT = bool(grouped_dot)
        if grouped_dot_allow_graphs is None:
            moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS = bool(original_grouped_dot_allow_graphs)
        else:
            moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS = bool(grouped_dot_allow_graphs)
        if single_row_gemv is None:
            moe_kernel._CFG_SINGLE_ROW_GEMV = bool(original_single_row_gemv)
        else:
            moe_kernel._CFG_SINGLE_ROW_GEMV = bool(single_row_gemv)
        if label == "shared_route_block_m1":
            moe_kernel._CFG_SHARED_ROUTE_BLOCK_M = 1
        elif label == "shared_route_block_m16":
            moe_kernel._CFG_SHARED_ROUTE_BLOCK_M = 16
        else:
            moe_kernel._CFG_SHARED_ROUTE_BLOCK_M = int(original_shared_route_block_m)
        if label == "shared_route_tiles_64x64":
            moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N = 64
            moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N = 64
        elif label == "shared_route_tiles_64x128":
            moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N = 64
            moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N = 128
        else:
            moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N = int(original_shared_route_gate_block_n)
            moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N = int(original_shared_route_down_block_n)
        if label == "shared_route_gate_k_split1":
            moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
        elif label == "shared_route_gate_k_split2":
            moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 2
        elif label == "shared_route_gate_k_split4":
            moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 4
        else:
            moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = int(
                original_shared_route_gate_k_splits
            )
        if label == "shared_route_fused_gate":
            moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = False
        elif label == "shared_route_split_gate":
            moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = True
        else:
            moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = bool(original_shared_route_split_gate)
        if label == "shared_route_atomic":
            moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
            moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
        elif label == "shared_route_partial":
            moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = True
            moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
        else:
            moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = bool(original_shared_route_partial_reduce)
            moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = bool(original_shared_route_token_accum)
        if label == "shared_route_strided_weights":
            moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = False
        elif label == "shared_route_coalesced_weights":
            moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = True
        else:
            moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = bool(
                original_shared_route_coalesced_weights
            )
        if compact_cfg is None:
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = bool(original_compact_fused_pack)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = bool(original_compact_partial_reduce)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = bool(original_compact_active_list)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = bool(original_compact_token_accum)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = int(original_compact_gate_block_n)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = int(original_compact_down_block_n)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = bool(original_compact_direct_out)
        else:
            gate_block_n, down_block_n, _compact_block_k, _compact_warps, _compact_stages = compact_cfg
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = True
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = False
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = False
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = False
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = int(gate_block_n)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = int(down_block_n)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = False
        if (
            args.include_expert_grouped_ab
            or args.include_expert_grouped_compact_ab
            or args.include_shared_route_partial_ab
            or args.include_shared_route_layout_ab
            or args.include_shared_route_tiles_ab
            or args.include_shared_route_split_gate_ab
            or compact_cfg is not None
        ):
            moe_kernel._CFG_EXPERT_GROUPED_DECODE = bool(expert_grouped)
            moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE = bool(expert_grouped_dense)
            moe_kernel._CFG_EXPERT_GROUPED_GENERAL_DECODE = bool(expert_grouped_general)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = bool(expert_grouped_compact)
            moe_kernel._CFG_SHARED_ROUTE_DECODE = bool(shared_route)
        else:
            moe_kernel._CFG_EXPERT_GROUPED_DECODE = bool(original_expert_grouped)
            moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE = bool(original_expert_grouped_dense)
            moe_kernel._CFG_EXPERT_GROUPED_GENERAL_DECODE = bool(original_expert_grouped_general)
            moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = bool(original_expert_grouped_compact)
            moe_kernel._CFG_SHARED_ROUTE_DECODE = bool(original_shared_route)
        if cfg is None:
            llama_mod._USE_QWEN3_MOE_GROUPED_DECODE = False
            print(
                f"\n== CASE {label}: grouped off FUSED_ROUTER={int(fused_router)} "
                f"TOKEN_ACCUM={int(bool(moe_kernel._CFG_TOKEN_ACCUM))} "
                f"GROUPED_DOT={int(bool(moe_kernel._CFG_GROUPED_DOT))} "
                f"DOT_ALLOW_GRAPHS={int(bool(moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS))} "
                f"EXPERT_GROUPED={int(bool(moe_kernel._CFG_EXPERT_GROUPED_DECODE))} "
                f"DENSE={int(bool(moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE))} "
                f"COMPACT={int(bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE))} =="
            )
        else:
            block_n, block_k, warps, stages = cfg
            llama_mod._USE_QWEN3_MOE_GROUPED_DECODE = True
            llama_mod._QWEN3_MOE_GROUPED_LOGGED = False
            moe_kernel._CFG_BLOCK_N = int(block_n)
            moe_kernel._CFG_BLOCK_K = int(block_k)
            moe_kernel._CFG_NUM_WARPS = int(warps)
            moe_kernel._CFG_NUM_STAGES = int(stages)
            moe_kernel._CFG_MAX_ASSIGNMENTS = int(args.max_assignments)
            print(
                f"\n== CASE {label}: BLOCK_N={block_n} BLOCK_K={block_k} "
                f"WARPS={warps} STAGES={stages} FUSED_ROUTER={int(fused_router)} "
                f"ROUTER_K_SPLITS={int(moe_kernel._CFG_ROUTER_K_SPLITS)} "
                f"TOKEN_ACCUM={int(bool(moe_kernel._CFG_TOKEN_ACCUM))} "
                f"GROUPED_DOT={int(bool(moe_kernel._CFG_GROUPED_DOT))} "
                f"DOT_ALLOW_GRAPHS={int(bool(moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS))} "
                f"EXPERT_GROUPED={int(bool(moe_kernel._CFG_EXPERT_GROUPED_DECODE))} "
                f"DENSE={int(bool(moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE))} "
                f"GENERAL={int(bool(moe_kernel._CFG_EXPERT_GROUPED_GENERAL_DECODE))} "
                f"COMPACT={int(bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE))} "
                f"COMPACT_GATE_N={int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N)} "
                f"COMPACT_DOWN_N={int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N)} "
                f"COMPACT_ACTIVE={int(bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST))} "
                f"COMPACT_DIRECT={int(bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT))} "
                f"SINGLE_ROW_GEMV={int(bool(moe_kernel._CFG_SINGLE_ROW_GEMV))} "
                f"SHARED_ROUTE_BLOCK_M={int(moe_kernel._CFG_SHARED_ROUTE_BLOCK_M)} "
                f"SHARED_ROUTE_GATE_N={int(moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N)} "
                f"SHARED_ROUTE_GATE_K_SPLITS={int(moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS)} "
                f"SHARED_ROUTE_DOWN_N={int(moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N)} "
                f"SHARED_ROUTE_SPLIT_GATE={int(bool(moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE))} "
                f"SHARED_ROUTE_PARTIAL={int(bool(moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE))} "
                f"SHARED_ROUTE_COALESCED={int(bool(moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS))} "
                f"SHARED_ROUTE_TOKEN_ACCUM={int(bool(moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM))} "
                f"SHARED_ROUTE={int(bool(moe_kernel._CFG_SHARED_ROUTE_DECODE))} =="
            )

        _clear_decode_graph_cache(engine)
        print("-- warmup --")
        warmup_log, warmup_row = _run_generate(
            engine,
            prompts,
            args.warmup_tokens,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.0,
        )
        _require_fixed_decode_length(warmup_row, args.warmup_tokens, f"{label} warmup")
        print(warmup_log.strip())
        warmup_brief = _timing_brief(warmup_row)
        if warmup_brief:
            print(warmup_brief)

        samples = []
        for repeat_idx in range(max(1, args.repeats)):
            print(f"-- measure repeat {repeat_idx + 1}/{max(1, args.repeats)} --")
            log, row = _run_generate(
                engine,
                prompts,
                args.max_tokens,
                temperature=0.0,
                top_k=1,
                repetition_penalty=1.0,
            )
            _require_fixed_decode_length(
                row,
                args.max_tokens,
                f"{label} repeat {repeat_idx + 1}",
            )
            print(log.strip())
            brief = _timing_brief(row)
            if brief:
                print(brief)
            row.update(
                {
                    "case": label,
                    "repeat": repeat_idx,
                    "block_n": cfg[0] if cfg else None,
                    "block_k": cfg[1] if cfg else None,
                    "warps": cfg[2] if cfg else None,
                    "stages": cfg[3] if cfg else None,
                    "grouped": cfg is not None,
                    "fused_router": bool(fused_router),
                    "router_k_splits": int(moe_kernel._CFG_ROUTER_K_SPLITS),
                    "token_accum": bool(moe_kernel._CFG_TOKEN_ACCUM),
                    "grouped_dot": bool(moe_kernel._CFG_GROUPED_DOT),
                    "grouped_dot_allow_cuda_graphs": bool(moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS),
                    "expert_grouped": bool(moe_kernel._CFG_EXPERT_GROUPED_DECODE),
                    "expert_grouped_dense": bool(moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE),
                    "expert_grouped_general": bool(moe_kernel._CFG_EXPERT_GROUPED_GENERAL_DECODE),
                    "expert_grouped_compact": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE),
                    "expert_grouped_compact_gate_block_n": int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N),
                    "expert_grouped_compact_down_block_n": int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N),
                    "expert_grouped_compact_active_list": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST),
                    "expert_grouped_compact_direct_out": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT),
                    "single_row_gemv": bool(moe_kernel._CFG_SINGLE_ROW_GEMV),
                    "shared_route_block_m": int(moe_kernel._CFG_SHARED_ROUTE_BLOCK_M),
                    "shared_route_gate_block_n": int(moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N),
                    "shared_route_gate_k_splits": int(moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS),
                    "shared_route_down_block_n": int(moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N),
                    "shared_route_split_gate": bool(moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE),
                    "shared_route_partial_reduce": bool(moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE),
                    "shared_route_coalesced_weights": bool(
                        moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS
                    ),
                    "shared_route_token_accum": bool(moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM),
                    "compact_config": compact_cfg,
                    "shared_route": bool(moe_kernel._CFG_SHARED_ROUTE_DECODE),
                }
            )
            samples.append(row)
            rows.append(row)

        summary = _summarize(samples)
        case_summaries.append(
            {
                "case": label,
                "config": cfg,
                "fused_router": bool(fused_router),
                "router_k_splits": int(moe_kernel._CFG_ROUTER_K_SPLITS),
                "token_accum": bool(moe_kernel._CFG_TOKEN_ACCUM),
                "grouped_dot": bool(moe_kernel._CFG_GROUPED_DOT),
                "grouped_dot_allow_cuda_graphs": bool(moe_kernel._CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS),
                "expert_grouped": bool(moe_kernel._CFG_EXPERT_GROUPED_DECODE),
                "expert_grouped_dense": bool(moe_kernel._CFG_EXPERT_GROUPED_DENSE_DECODE),
                "expert_grouped_general": bool(moe_kernel._CFG_EXPERT_GROUPED_GENERAL_DECODE),
                "expert_grouped_compact": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE),
                "expert_grouped_compact_gate_block_n": int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N),
                "expert_grouped_compact_down_block_n": int(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N),
                "expert_grouped_compact_active_list": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST),
                "expert_grouped_compact_direct_out": bool(moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT),
                "single_row_gemv": bool(moe_kernel._CFG_SINGLE_ROW_GEMV),
                "shared_route_block_m": int(moe_kernel._CFG_SHARED_ROUTE_BLOCK_M),
                "shared_route_gate_block_n": int(moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N),
                "shared_route_gate_k_splits": int(moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS),
                "shared_route_down_block_n": int(moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N),
                "shared_route_split_gate": bool(moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE),
                "shared_route_partial_reduce": bool(moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE),
                "shared_route_coalesced_weights": bool(
                    moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS
                ),
                "shared_route_token_accum": bool(moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM),
                "compact_config": compact_cfg,
                "shared_route": bool(moe_kernel._CFG_SHARED_ROUTE_DECODE),
                **summary,
            }
        )
        print(f"SUMMARY {label}: {json.dumps(summary, sort_keys=True)}")

    valid = [row for row in rows if row.get("tok_s") is not None]
    valid.sort(key=lambda row: float(row["tok_s"]), reverse=True)
    print("\n== RANKING ==")
    for idx, row in enumerate(valid[:10], start=1):
        print(
            f"{idx:02d}. {row['case']}: {row['tok_s']:.2f} tok/s "
            f"decode={row['decode_ms']:.1f}ms"
        )

    profile_summary = None
    if args.profile_decode_breakdown:
        print("\n== CUDA DECODE PROFILE: STABLE FUSED-GATE PATH ==")
        moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = False
        moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
        moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = False
        moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
        moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N = 64
        moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
        moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N = 64
        _clear_decode_graph_cache(engine)
        warmup_log, _ = _run_generate(
            engine,
            prompts,
            max(8, min(32, int(args.profile_tokens))),
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.0,
        )
        print(warmup_log.strip())
        profile_summary = engine.profile_decode_breakdown(
            prompts if args.batch_size > 1 else prompts[0],
            max_new_tokens=max(2, int(args.profile_tokens)),
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.0,
        )

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "decode_timing": {
            "enabled": os.environ.get("MEGAGEMM_DECODE_TIMING", ""),
            "detail": os.environ.get("MEGAGEMM_DECODE_TIMING_DETAIL", ""),
            "print": os.environ.get("MEGAGEMM_DECODE_TIMING_PRINT", ""),
        },
        "case_summaries": case_summaries,
        "rows": rows,
        "profile_decode_breakdown": profile_summary,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
