"""Synthetic Gemma4-26B-A4B prefill QKV benchmark (no model download)."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def measure(
    fn: Callable[[], None], *, warmup: int, iterations: int, repeats: int
) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "measurement": "eager_cuda_events",
    }


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    return {
        "max_abs_error": float((ref - cand).abs().max().item()),
        "cosine": float(F.cosine_similarity(ref, cand, dim=0).item()),
    }


def benchmark_shape(
    *,
    name: str,
    rows: int,
    hidden: int,
    q_size: int,
    k_size: int,
    v_size: int,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[dict]:
    x = torch.empty((rows, hidden), device="cuda", dtype=dtype).normal_(0.0, 0.02)
    q_weight = torch.empty((q_size, hidden), device="cuda", dtype=dtype).normal_(0.0, 0.02)
    k_weight = torch.empty((k_size, hidden), device="cuda", dtype=dtype).normal_(0.0, 0.02)
    v_weight = torch.empty((v_size, hidden), device="cuda", dtype=dtype).normal_(0.0, 0.02)
    fused_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()

    q_out = torch.empty((rows, q_size), device="cuda", dtype=dtype)
    k_out = torch.empty((rows, k_size), device="cuda", dtype=dtype)
    v_out = torch.empty((rows, v_size), device="cuda", dtype=dtype)
    fused_out = torch.empty(
        (rows, q_size + k_size + v_size), device="cuda", dtype=dtype
    )

    def separate() -> None:
        torch.mm(x, q_weight.t(), out=q_out)
        torch.mm(x, k_weight.t(), out=k_out)
        torch.mm(x, v_weight.t(), out=v_out)

    def fused() -> None:
        torch.mm(x, fused_weight.t(), out=fused_out)

    separate()
    fused()
    torch.cuda.synchronize()
    reference = torch.cat((q_out, k_out, v_out), dim=-1)
    rows_out = []
    for case, fn, output in (
        ("separate_q_k_v", separate, lambda: torch.cat((q_out, k_out, v_out), dim=-1)),
        ("fused_qkv", fused, lambda: fused_out),
    ):
        timing = measure(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        fn()
        torch.cuda.synchronize()
        error = tensor_error(reference, output())
        row = {
            "attention_type": name,
            "case": case,
            "rows": rows,
            "hidden": hidden,
            "q_size": q_size,
            "k_size": k_size,
            "v_size": v_size,
            **timing,
            **error,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out-json", default="bench_results/gemma4_prefill_qkv_a100.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print("Gemma4 prefill QKV microbenchmark")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  rows: {args.rows}")
    print(f"  dtype: {args.dtype}")

    torch.manual_seed(20260711)
    all_rows = []
    with torch.inference_mode():
        # 25 sliding layers: 16 Q heads and 8 KV heads, head_dim=256.
        all_rows.extend(
            benchmark_shape(
                name="sliding",
                rows=args.rows,
                hidden=2816,
                q_size=4096,
                k_size=2048,
                v_size=2048,
                dtype=dtype,
                args=args,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
        # 5 full layers: 16 Q heads and 2 KV heads, global_head_dim=512.
        all_rows.extend(
            benchmark_shape(
                name="full",
                rows=args.rows,
                hidden=2816,
                q_size=8192,
                k_size=1024,
                v_size=1024,
                dtype=dtype,
                args=args,
            )
        )

    by_key = {(row["attention_type"], row["case"]): row for row in all_rows}
    separate_us = (
        25.0 * by_key[("sliding", "separate_q_k_v")]["median_us"]
        + 5.0 * by_key[("full", "separate_q_k_v")]["median_us"]
    )
    fused_us = (
        25.0 * by_key[("sliding", "fused_qkv")]["median_us"]
        + 5.0 * by_key[("full", "fused_qkv")]["median_us"]
    )
    speedup = separate_us / fused_us
    correct = all(
        row["max_abs_error"] <= 0.03125 and row["cosine"] >= 0.999
        for row in all_rows
        if row["case"] == "fused_qkv"
    )
    decision = {
        "decision": "APPLY_FUSED_QKV" if correct and speedup >= 1.02 else "KEEP_SEPARATE",
        "separate_ms_per_prefill_30_layers": separate_us / 1000.0,
        "fused_ms_per_prefill_30_layers": fused_us / 1000.0,
        "estimated_savings_ms": max(0.0, separate_us - fused_us) / 1000.0,
        "speedup": speedup,
        "correct": correct,
    }
    print("\nDECISION " + json.dumps(decision, sort_keys=True))
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "rows": args.rows,
        "results": all_rows,
        "decision": decision,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
