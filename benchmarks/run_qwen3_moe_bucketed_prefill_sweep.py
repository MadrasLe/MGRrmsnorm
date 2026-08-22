#!/usr/bin/env python3
"""Sweep Qwen3 MoE bucketed prefill bucket sizes with one model load."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_qwen3_moe_vs_vllm import (  # noqa: E402
    build_prompt_batch,
    gpu_snapshot,
    load_tokenizer,
    preflight_qwen3_moe_vram,
    run_megagemm_batch_once,
)


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("bucket sizes must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one bucket size is required")
    return values


def _dtype_from_arg(dtype: str) -> torch.dtype:
    return torch.bfloat16 if str(dtype).lower() == "bf16" else torch.float16


def _reset_prefill_counters(model) -> None:
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue
        for name in (
            "_bucketed_prefill_hits",
            "_bucketed_prefill_valid_assignments",
            "_bucketed_prefill_padded_assignments",
            "_bucketed_prefill_bucket_launches",
            "_batched_prefill_hits",
            "_sorted_prefill_hits",
        ):
            if hasattr(experts, name):
                setattr(experts, name, 0)
        if hasattr(experts, "_bucketed_prefill_disabled"):
            experts._bucketed_prefill_disabled = False
            experts._bucketed_prefill_fail_reason = ""


def _extract_metrics(row: dict[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("diagnostics") or {}
    timing = diagnostics.get("last_prefill_timing") or {}
    stats = diagnostics.get("decode_runtime_stats") or {}
    return {
        "prefill_ms": row.get("prefill_ms"),
        "decode_ms": row.get("decode_ms"),
        "output_tok_s_total": row.get("output_tok_s_total"),
        "moe_router_ms": timing.get("moe_router_ms"),
        "moe_experts_ms": timing.get("moe_experts_ms"),
        "prefill_timing_total_ms": timing.get("total_ms"),
        "bucketed_hits": stats.get("qwen3_moe_bucketed_prefill_total_hits"),
        "batched_hits": stats.get("qwen3_moe_batched_prefill_total_hits"),
        "sorted_hits": stats.get("qwen3_moe_sorted_prefill_total_hits"),
        "bucketed_pad_waste": stats.get("qwen3_moe_bucketed_prefill_pad_waste"),
        "bucket_launches": stats.get("qwen3_moe_bucketed_prefill_bucket_launches"),
        "valid_assignments": stats.get("qwen3_moe_bucketed_prefill_valid_assignments"),
        "padded_assignments": stats.get("qwen3_moe_bucketed_prefill_padded_assignments"),
    }


def _median(rows: list[dict[str, Any]], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prefill_ms_median": _median(rows, "prefill_ms"),
        "decode_ms_median": _median(rows, "decode_ms"),
        "output_tok_s_total_median": _median(rows, "output_tok_s_total"),
        "moe_router_ms_median": _median(rows, "moe_router_ms"),
        "moe_experts_ms_median": _median(rows, "moe_experts_ms"),
        "prefill_timing_total_ms_median": _median(rows, "prefill_timing_total_ms"),
        "bucketed_pad_waste_median": _median(rows, "bucketed_pad_waste"),
        "bucket_launches_median": _median(rows, "bucket_launches"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-token-target", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--bucket-sizes",
        type=_parse_int_list,
        default=_parse_int_list("256,512,768,1024,1536"),
        help="Comma-separated MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE values.",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--allow-low-vram", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_batch_size < args.batch_size:
        args.max_batch_size = args.batch_size

    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS", "4096")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM", "1")
    os.environ.setdefault("MEGAGEMM_PREFILL_TIMING", "1")
    os.environ.setdefault("MEGAGEMM_PREFILL_TIMING_PRINT", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS", "0")

    preflight_qwen3_moe_vram(args)

    from megagemm.engine import InferenceEngine
    import megagemm.models.llama as llama_mod

    print("Qwen3 MoE bucketed prefill sweep")
    print(f"  model:         {args.model}")
    print(f"  dtype:         {args.dtype}")
    print(f"  max_seq_len:   {args.max_seq_len}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  prompt_target: {args.prompt_token_target}")
    print(f"  max_tokens:    {args.max_tokens}")
    print(f"  bucket_sizes:  {','.join(str(v) for v in args.bucket_sizes)}")
    print(f"  gpu:           {gpu_snapshot()}")

    tokenizer = load_tokenizer(args.model, args.cache_dir or None)
    prompts, prompt_counts = build_prompt_batch(
        tokenizer,
        prompt="",
        use_chat_template=False,
        batch_size=args.batch_size,
        prompt_token_target=max(0, int(args.prompt_token_target)),
    )
    print(
        "  prompt_tokens: "
        f"min={min(prompt_counts) if prompt_counts else 0} "
        f"max={max(prompt_counts) if prompt_counts else 0}"
    )

    engine = InferenceEngine(
        args.model,
        dtype=_dtype_from_arg(args.dtype),
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )

    all_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for bucket_size in args.bucket_sizes:
        print(f"\n== BUCKET_SIZE {bucket_size} ==")
        llama_mod._USE_QWEN3_MOE_BUCKETED_PREFILL = True
        llama_mod._QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE = int(bucket_size)
        os.environ["MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE"] = str(bucket_size)

        print("-- warmup --")
        _reset_prefill_counters(engine.model)
        warmup = run_megagemm_batch_once(
            engine,
            prompts,
            max_tokens=args.warmup_tokens,
            ignore_eos=True,
        )
        print(warmup.get("raw_log", ""))

        rows: list[dict[str, Any]] = []
        for repeat_idx in range(max(1, int(args.repeats))):
            print(f"-- repeat {repeat_idx + 1}/{max(1, int(args.repeats))} --")
            _reset_prefill_counters(engine.model)
            row = run_megagemm_batch_once(
                engine,
                prompts,
                max_tokens=args.max_tokens,
                ignore_eos=True,
            )
            metrics = _extract_metrics(row)
            row.update(metrics)
            row["bucket_size"] = int(bucket_size)
            row["repeat"] = repeat_idx + 1
            print(row.get("raw_log", ""))
            rows.append(row)
            all_rows.append(row)

        summary = {"bucket_size": int(bucket_size), **_summarize(rows)}
        case_summaries.append(summary)
        print("SUMMARY " + json.dumps(summary, sort_keys=True))

    valid = [
        item
        for item in case_summaries
        if item.get("output_tok_s_total_median") is not None
    ]
    valid.sort(key=lambda item: float(item["output_tok_s_total_median"]), reverse=True)
    print("\n== RANKING ==")
    for idx, item in enumerate(valid, start=1):
        waste = item.get("bucketed_pad_waste_median")
        waste_text = f"{float(waste) * 100.0:.1f}%" if waste is not None else "n/a"
        print(
            f"{idx:02d}. bucket={item['bucket_size']}: "
            f"total={float(item['output_tok_s_total_median']):.2f} tok/s "
            f"prefill={float(item['prefill_ms_median']):.1f}ms "
            f"experts={float(item['moe_experts_ms_median']):.1f}ms "
            f"pad_waste={waste_text} "
            f"bucket_launches={item.get('bucket_launches_median')}"
        )

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "prompt_token_target": args.prompt_token_target,
        "prompt_token_counts": prompt_counts,
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "case_summaries": case_summaries,
        "rows": all_rows,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
