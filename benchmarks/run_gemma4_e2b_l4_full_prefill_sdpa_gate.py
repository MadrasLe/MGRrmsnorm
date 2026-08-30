#!/usr/bin/env python3
"""Gate implicit-causal SDPA for Gemma 4 E2B/L4 full H512 prefill."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


TensorFn = Callable[[], torch.Tensor]


def _measure_us(
    fn: TensorFn,
    *,
    warmups: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
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
        "median_us": float(statistics.median(samples)),
        "samples_us": samples,
        "spread_ratio": max(samples) / min(samples),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


def decide(
    rows: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    by_name = {str(row["case"]): row for row in rows}
    baseline = by_name.get("explicit_native_recheck") or {}
    baseline_us = (
        float(baseline["median_us"])
        if baseline.get("correct") and baseline.get("median_us") is not None
        else None
    )
    candidates = [
        row
        for row in rows
        if str(row["case"]).startswith("implicit_")
        and row.get("correct")
        and row.get("median_us") is not None
    ]
    candidates.sort(key=lambda row: float(row["median_us"]))
    winner = candidates[0] if candidates else None
    winner_us = float(winner["median_us"]) if winner is not None else None
    speedup = (
        baseline_us / winner_us
        if baseline_us is not None and winner_us is not None and winner_us > 0.0
        else None
    )
    stable = bool(
        winner is not None
        and float(winner.get("spread_ratio") or float("inf")) <= maximum_spread
    )
    apply_change = bool(
        speedup is not None
        and speedup >= minimum_speedup
        and stable
    )
    if apply_change and winner is not None:
        decision = (
            "PROMOTE_IMPLICIT_CAUSAL_EXPANDED_SDPA"
            if winner["case"] == "implicit_expanded"
            else "PROMOTE_IMPLICIT_CAUSAL_NATIVE_SDPA"
        )
    else:
        decision = "KEEP_EXPLICIT_AND_BUILD_TRITON_FULL_H512"
    return {
        "decision": decision,
        "apply_change": apply_change,
        "baseline": "explicit_native_recheck",
        "baseline_us": baseline_us,
        "winner": None if winner is None else winner["case"],
        "winner_us": winner_us,
        "speedup": speedup,
        "stable": stable,
        "minimum_speedup": minimum_speedup,
        "maximum_spread": maximum_spread,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "l4" not in gpu.lower().replace("-", " ").split():
        raise RuntimeError(f"this gate requires NVIDIA L4, found {gpu}")

    torch.manual_seed(20260830)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    q_heads = 8
    kv_heads = 1
    seq_len = int(args.seq_len)
    head_dim = 512
    scale = 1.0 / math.sqrt(head_dim)
    if not 2048 <= seq_len <= 2304:
        raise ValueError("--seq-len must stay inside the E2B production gate")

    q = torch.randn(
        batch_size, q_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril_()
    explicit_mask = torch.zeros(
        batch_size, 1, seq_len, seq_len, device=device, dtype=dtype
    )
    explicit_mask.masked_fill_(~causal.view(1, 1, seq_len, seq_len), float("-inf"))

    cases: list[tuple[str, TensorFn]] = [
        (
            "explicit_native",
            lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=explicit_mask,
                is_causal=False,
                enable_gqa=True,
                scale=scale,
            ),
        ),
        (
            "implicit_native",
            lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=True,
                scale=scale,
            ),
        ),
        (
            "implicit_expanded",
            lambda: F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(q_heads // kv_heads, dim=1),
                v.repeat_interleave(q_heads // kv_heads, dim=1),
                attn_mask=None,
                is_causal=True,
                scale=scale,
            ),
        ),
        (
            "explicit_native_recheck",
            lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=explicit_mask,
                is_causal=False,
                enable_gqa=True,
                scale=scale,
            ),
        ),
    ]

    print("Gemma 4 E2B/L4 full-prefill SDPA causal gate")
    print(f"  gpu: {gpu}")
    print(
        f"  shape: B{batch_size} Q{q_heads}/KV{kv_heads} "
        f"S{seq_len} H{head_dim} BF16"
    )
    print("  model_download: disabled")
    print("  competing_engine_install: disabled")

    reference = cases[0][1]().detach().clone()
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for name, fn in cases:
        try:
            first = fn().detach().clone()
            second = fn().detach().clone()
            torch.cuda.synchronize()
            delta = (first.float() - reference.float()).abs()
            repeat_delta = (first.float() - second.float()).abs()
            finite = bool(torch.isfinite(first).all().item())
            repeat_exact = bool(torch.equal(first, second))
            cosine = _cosine(first, reference)
            max_abs_error = float(delta.max().item())
            mean_abs_error = float(delta.mean().item())
            correct = bool(
                finite
                and repeat_exact
                and cosine >= 0.9999
                and max_abs_error <= 0.125
            )
            row = {
                "case": name,
                "error": None,
                "correct": correct,
                "finite": finite,
                "repeat_exact": repeat_exact,
                "repeat_max_abs_error": float(repeat_delta.max().item()),
                "cosine": cosine,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                **_measure_us(
                    fn,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    repeats=args.repeats,
                ),
            }
            del first, second, delta, repeat_delta
        except Exception as exc:
            row = {
                "case": name,
                "error": f"{type(exc).__name__}: {exc}",
                "correct": False,
                "median_us": None,
                "samples_us": [],
                "spread_ratio": None,
            }
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        rows.append(row)
        print("CASE " + json.dumps(row, sort_keys=True))
        gc.collect()

    decision = decide(
        rows,
        minimum_speedup=args.minimum_speedup,
        maximum_spread=args.maximum_spread,
    )
    payload = {
        "benchmark": "gemma4_e2b_l4_full_prefill_sdpa_gate",
        "gpu": gpu,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "shape": {
            "batch_size": batch_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "dtype": "bf16",
        },
        "rows": rows,
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DECISION " + json.dumps(decision, sort_keys=True))
    print(f"Wrote: {output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2057)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--maximum-spread", type=float, default=1.05)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("warmups/iterations must be positive and repeats >= 3")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
