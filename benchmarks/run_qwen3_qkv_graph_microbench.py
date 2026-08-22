#!/usr/bin/env python3
"""Compare Qwen3 decode QKV backends in the production CUDA-graph regime."""

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

from megagemm.kernels.fast_gemv import fast_linear
from megagemm.kernels.fused_rmsnorm_linear import fused_rmsnorm_linear
from megagemm.kernels.rmsnorm_triton import rmsnorm_triton


def _measure_graph_us(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(max(3, warmup)):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    torch.cuda.synchronize()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / float(iterations))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--qkv-size", type=int, default=5120)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    x = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype)
    norm_weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype) * 0.02 + 1.0
    weight = torch.randn(
        args.qkv_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    fused_out = torch.empty(args.rows, args.qkv_size, device="cuda", dtype=dtype)
    cublas_out = torch.empty_like(fused_out)
    fast_out = torch.empty_like(fused_out)

    def fused_call():
        return fused_rmsnorm_linear(
            x,
            norm_weight,
            args.eps,
            weight,
            out=fused_out,
        )

    def cublas_call():
        normed = rmsnorm_triton(x, norm_weight, args.eps, False)
        torch.mm(normed, weight.t(), out=cublas_out)
        return cublas_out

    def fast_call():
        normed = rmsnorm_triton(x, norm_weight, args.eps, False)
        return fast_linear(normed, weight, out=fast_out)

    reference = cublas_call().clone()
    cases = (
        ("fused_rmsnorm_linear", fused_call, fused_out),
        ("rmsnorm_plus_cublas", cublas_call, cublas_out),
        ("rmsnorm_plus_fast_linear", fast_call, fast_out),
    )

    print("Qwen3 decode QKV CUDA-graph microbenchmark")
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  capability:", torch.cuda.get_device_capability(0))
    print(
        f"  shape: rows={args.rows} hidden={args.hidden_size} "
        f"qkv={args.qkv_size} dtype={args.dtype}"
    )

    rows = []
    for name, fn, output in cases:
        error = None
        samples = []
        max_abs_error = None
        cosine = None
        try:
            fn()
            torch.cuda.synchronize()
            max_abs_error = float((output.float() - reference.float()).abs().max().item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    output.flatten().float(),
                    reference.flatten().float(),
                    dim=0,
                ).item()
            )
            samples = _measure_graph_us(
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
            "qkv_ms_per_token_48_layers": (
                median_us * 48.0 / 1000.0 if median_us is not None else None
            ),
            "max_abs_error": max_abs_error,
            "cosine": cosine,
            "samples_us": samples,
            "error": error,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    valid = [row for row in rows if row["median_us"] is not None and row["error"] is None]
    valid.sort(key=lambda row: float(row["median_us"]))
    print("\n== RANKING ==")
    for index, row in enumerate(valid, start=1):
        print(
            f"{index:02d}. {row['case']}: {row['median_us']:.3f} us/call "
            f"({row['qkv_ms_per_token_48_layers']:.3f} ms/token for 48 layers)"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0 if len(valid) == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
