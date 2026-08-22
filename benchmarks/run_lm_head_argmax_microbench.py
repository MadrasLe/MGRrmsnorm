#!/usr/bin/env python3
"""Microbenchmark greedy lm_head argmax paths without loading a model."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megagemm.kernels.lm_head_argmax import (  # noqa: E402
    HAS_FUSED_LM_HEAD_ARGMAX,
    lm_head_argmax,
    lm_head_argmax_prefers_triton_shape,
    lm_head_rmsnorm_argmax,
)


def _measure_graph_us(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(max(1, repeats)):
        for _ in range(max(1, warmup)):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, iterations)):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / float(iterations))
    return samples


def _measure_eager_us(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(max(1, repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, iterations)):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / float(iterations))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_grad_enabled(False)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype) * 0.02
    norm_weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype) * 0.02 + 1.0
    weight = torch.randn(args.vocab_size, args.hidden_size, device="cuda", dtype=dtype) * 0.02
    out_tokens = torch.empty(args.rows, device="cuda", dtype=torch.long)
    # Cover the smallest supported BLOCK_N so every tested kernel can reuse
    # these buffers instead of allocating inside CUDA graph capture.
    n_blocks = (args.vocab_size + 63) // 64
    partial_vals = torch.empty(args.rows, n_blocks, device="cuda", dtype=torch.float32)
    partial_idxs = torch.empty(args.rows, n_blocks, device="cuda", dtype=torch.int32)

    def base_call():
        rms = torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + float(args.eps))
        normed = (hidden * rms * norm_weight.float()).to(dtype=dtype)
        return torch.nn.functional.linear(normed, weight).argmax(dim=-1)

    def fused_call():
        rms = torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + float(args.eps))
        normed = (hidden * rms * norm_weight.float()).to(dtype=dtype)
        return lm_head_argmax(
            normed,
            weight,
            out_tokens=out_tokens,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        )

    def fused_rmsnorm_call():
        return lm_head_rmsnorm_argmax(
            hidden,
            norm_weight,
            args.eps,
            False,
            weight,
            out_tokens=out_tokens,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        )

    print("LM head argmax microbenchmark")
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  capability:", torch.cuda.get_device_capability(0))
    print(
        f"  shape: rows={args.rows} hidden={args.hidden_size} "
        f"vocab={args.vocab_size} dtype={args.dtype}"
    )
    print("  has_fused:", bool(HAS_FUSED_LM_HEAD_ARGMAX))
    print(
        "  prefers_shape:",
        bool(lm_head_argmax_prefers_triton_shape(args.hidden_size, args.vocab_size, args.rows)),
    )
    print("  measurement:", "eager" if args.eager else "cuda_graph")

    reference = base_call().clone()
    torch.cuda.synchronize()
    cases = [
        ("base_norm_linear_argmax", base_call),
        ("fused_lm_head_argmax", fused_call),
        ("fused_rmsnorm_lm_head_argmax", fused_rmsnorm_call),
    ]
    measure = _measure_eager_us if args.eager else _measure_graph_us
    rows = []
    for name, fn in cases:
        error = None
        samples = []
        equal = False
        try:
            actual = fn()
            torch.cuda.synchronize()
            equal = bool(torch.equal(actual, reference))
            samples = measure(
                fn,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        median_us = statistics.median(samples) if samples else None
        row = {
            "case": name,
            "median_us": median_us,
            "lm_head_ms_per_128_tokens": median_us * 128.0 / 1000.0 if median_us is not None else None,
            "tokens_equal": equal,
            "samples_us": samples,
            "error": error,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    valid = [row for row in rows if row["median_us"] is not None and row["error"] is None]
    valid.sort(key=lambda row: float(row["median_us"]))
    print("\n== RANKING ==")
    for index, row in enumerate(valid, 1):
        print(
            f"{index:02d}. {row['case']}: {row['median_us']:.3f} us/call "
            f"({row['lm_head_ms_per_128_tokens']:.3f} ms/128 tok)"
        )

    current = next(
        (row for row in valid if row["case"] == "fused_lm_head_argmax"),
        None,
    )
    fused_norm = next(
        (row for row in valid if row["case"] == "fused_rmsnorm_lm_head_argmax"),
        None,
    )
    winner = valid[0] if valid else None
    speedup = (
        float(current["median_us"]) / float(fused_norm["median_us"])
        if current is not None and fused_norm is not None
        else None
    )
    decision = "KEEP_CURRENT_PATH"
    if (
        fused_norm is not None
        and fused_norm["tokens_equal"]
        and speedup is not None
        and speedup >= 1.02
    ):
        decision = "APPLY_RMSNORM_FUSION"
    decision_row = {
        "decision": decision,
        "winner": winner["case"] if winner is not None else None,
        "rmsnorm_fusion_speedup_vs_current": speedup,
    }
    print("\nDECISION", json.dumps(decision_row, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"results": rows, **decision_row}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
