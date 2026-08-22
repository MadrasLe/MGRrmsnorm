#!/usr/bin/env python3
"""Sweep Qwen3 MoE paged decode split counts with one MegaGemm model load."""

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

from run_qwen3_moe_grouped_sweep import _clear_decode_graph_cache, _run_generate
from run_qwen3_moe_vs_vllm import build_prompt_batch, dtype_from_arg, gpu_snapshot, load_tokenizer


def _parse_splits(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise argparse.ArgumentTypeError("split counts must be >= 0")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one split count is required")
    return values


def _parse_gqa_groups(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values or any(value not in (4, 8) for value in values):
        raise argparse.ArgumentTypeError("GQA groups must be a comma-separated subset of 4,8")
    return values


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = rows[1:] if len(rows) > 1 else rows
    tok_s = [float(row["tok_s"]) for row in measured if row.get("tok_s") is not None]
    decode_ms = [
        float(row["decode_ms"])
        for row in measured
        if row.get("decode_ms") is not None
    ]
    total_tps = [
        float(row["output_tok_s_total"])
        for row in measured
        if row.get("output_tok_s_total") is not None
    ]
    prefill_ms = [
        float(row["prefill_ms"])
        for row in measured
        if row.get("prefill_ms") is not None
    ]
    return {
        "decode_tok_s_median": statistics.median(tok_s) if tok_s else None,
        "decode_tok_s_mean": statistics.mean(tok_s) if tok_s else None,
        "decode_ms_median": statistics.median(decode_ms) if decode_ms else None,
        "output_tok_s_total_median": statistics.median(total_tps) if total_tps else None,
        "prefill_ms_median": statistics.median(prefill_ms) if prefill_ms else None,
        "measured_repeats": len(measured),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-token-target", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--splits", type=_parse_splits, default=_parse_splits("1,2,4,8"))
    parser.add_argument("--gqa-groups", type=_parse_gqa_groups, default=_parse_gqa_groups("4,8"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_MIN_BATCH", str(args.batch_size))
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_ALLOW_QWEN3_MOE", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_LOG_LIMIT", "32")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL", "0")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE", "512")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_REDUCE_WARPS", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA2_SPLIT", "0")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA4_SPLIT", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_GQA8_SPLIT", "1")
    os.environ.setdefault("MEGAGEMM_PAGED_DECODE_LOG", "1")

    from megagemm.engine import InferenceEngine
    import megagemm.kernels.paged_attention as paged_attention

    tokenizer = load_tokenizer(args.model)
    exact_prompts, prompt_counts = build_prompt_batch(
        tokenizer,
        "Write a concise Python Fibonacci function.",
        True,
        args.batch_size,
        args.prompt_token_target,
    )

    print("Qwen3 MoE paged decode split sweep")
    print(f"  model:      {args.model}")
    print(f"  dtype:      {args.dtype}")
    print(f"  batch:      {args.batch_size}")
    print(f"  prompt:     min={min(prompt_counts)} max={max(prompt_counts)}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  splits:     {','.join(str(item or 'auto') for item in args.splits)}")
    print(f"  gqa_groups: {','.join(str(item) for item in args.gqa_groups)}")
    print(f"  graphs:     {os.environ.get('MEGAGEMM_DECODE_CUDA_GRAPHS', '0')}")
    print(f"  gqa2_split: {os.environ.get('MEGAGEMM_PAGED_DECODE_GQA2_SPLIT', '')}")
    print(f"  reduce_warps: {os.environ.get('MEGAGEMM_PAGED_DECODE_REDUCE_WARPS', '')}")
    print(f"  gpu:        {gpu_snapshot()}")

    engine = InferenceEngine(
        args.model,
        dtype=dtype_from_arg(args.dtype),
        device=args.device,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    results: list[dict[str, Any]] = []
    for gqa_group in args.gqa_groups:
        for split_count in args.splits:
            if split_count <= 0:
                os.environ.pop("MEGAGEMM_PAGED_DECODE_SPLITS", None)
                split_label = "auto"
            else:
                os.environ["MEGAGEMM_PAGED_DECODE_SPLITS"] = str(split_count)
                split_label = str(split_count)
            os.environ["MEGAGEMM_PAGED_DECODE_GQA4_SPLIT"] = "1"
            os.environ["MEGAGEMM_PAGED_DECODE_GQA8_SPLIT"] = "1" if gqa_group == 8 else "0"
            paged_attention._GQA4_DECODE_DISABLED = False
            paged_attention._GQA4_DECODE_LOGGED = False
            paged_attention._GQA8_DECODE_DISABLED = False
            paged_attention._GQA8_DECODE_LOGGED = False
            _clear_decode_graph_cache(engine)
            label = f"gqa{gqa_group}_splits{split_label}"

            print(f"\n== GQA {gqa_group} SPLITS {split_label} ==")
            rows: list[dict[str, Any]] = []
            warmup_log, warmup_row = _run_generate(
                engine,
                exact_prompts,
                args.warmup_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            print("-- warmup --")
            print(warmup_log.strip())
            rows.append(warmup_row)

            for repeat in range(1, max(1, args.repeats) + 1):
                log, row = _run_generate(
                    engine,
                    exact_prompts,
                    args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
                row["repeat"] = repeat
                print(f"-- repeat {repeat}/{args.repeats} --")
                print(log.strip())
                rows.append(row)

            summary = _summarize(rows)
            result = {
                "splits": split_count,
                "gqa_group": gqa_group,
                "label": label,
                "summary": summary,
                "rows": rows,
            }
            results.append(result)
            print(f"SUMMARY {label}: {json.dumps(summary, sort_keys=True)}")

    ranked = sorted(
        results,
        key=lambda row: float(row["summary"].get("decode_tok_s_median") or 0.0),
        reverse=True,
    )
    print("\n== RANKING ==")
    for idx, row in enumerate(ranked, 1):
        summary = row["summary"]
        print(
            f"{idx:02d}. {row['label']}: "
            f"{summary.get('decode_tok_s_median')} tok/s decode_ms={summary.get('decode_ms_median')}"
        )

    payload = {
        "model": args.model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_token_target": args.prompt_token_target,
        "prompt_counts": prompt_counts,
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "gqa_groups": args.gqa_groups,
        "gpu": gpu_snapshot(),
        "results": results,
        "ranking": [
            {
                "splits": row["splits"],
                "gqa_group": row["gqa_group"],
                "label": row["label"],
                "decode_tok_s_median": row["summary"].get("decode_tok_s_median"),
            }
            for row in ranked
        ],
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
