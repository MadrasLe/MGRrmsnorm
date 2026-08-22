#!/usr/bin/env python3
"""Microbenchmark Qwen3 MoE fused-router K splitting without loading a model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


def _load_kernel_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "megagemm" / "kernels" / "qwen3_moe.py"
    spec = importlib.util.spec_from_file_location("megagemm_qwen3_moe_microbench", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_splits(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        value = int(item.strip())
        if value < 1 or value & (value - 1):
            raise argparse.ArgumentTypeError("splits must be positive powers of two")
        values.append(value)
    return values


def _measure_us(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
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
        samples.append(float(start.elapsed_time(end)) * 1000.0 / float(iterations))
    return samples


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
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--splits", type=_parse_splits, default=_parse_splits("1,2,4,8"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Measure eager Python launches instead of the production CUDA-graph regime.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    kernel = _load_kernel_module()
    if not getattr(kernel, "_HAS_TRITON", False):
        raise SystemExit("Triton is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    hidden = torch.randn(
        args.rows,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    )
    router_weight = torch.randn(
        args.num_experts,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02

    logits = torch.nn.functional.linear(hidden, router_weight)
    ref_values, ref_experts = torch.topk(logits.float(), args.top_k, dim=-1)
    ref_weights = torch.softmax(ref_values, dim=-1).to(dtype)

    print("Qwen3 MoE fused-router split-K microbenchmark")
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  capability:", torch.cuda.get_device_capability(0))
    print(
        f"  shape: rows={args.rows} hidden={args.hidden_size} "
        f"experts={args.num_experts} top_k={args.top_k} dtype={args.dtype}"
    )
    print("  splits:", ",".join(str(value) for value in args.splits))
    print("  measurement:", "eager" if args.eager else "cuda_graph")

    original_splits = int(kernel._CFG_ROUTER_K_SPLITS)
    rows = []
    try:
        for split_count in args.splits:
            kernel._CFG_ROUTER_K_SPLITS = int(split_count)
            workspace = {}

            def call():
                return kernel.qwen3_moe_router_topk_softmax(
                    hidden,
                    router_weight,
                    args.top_k,
                    workspace=workspace,
                )

            error = None
            samples = []
            experts_equal = False
            max_weight_error = None
            try:
                actual_weights, actual_experts = call()
                torch.cuda.synchronize()
                experts_equal = bool(torch.equal(actual_experts, ref_experts))
                max_weight_error = float(
                    (actual_weights.float() - ref_weights.float()).abs().max().item()
                )
                measure = _measure_us if args.eager else _measure_graph_us
                samples = measure(
                    call,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            median_us = statistics.median(samples) if samples else None
            result = {
                "splits": int(split_count),
                "median_us": median_us,
                "router_ms_per_token_48_layers": (
                    median_us * 48.0 / 1000.0 if median_us is not None else None
                ),
                "experts_equal": experts_equal,
                "max_weight_error": max_weight_error,
                "samples_us": samples,
                "measurement": "eager" if args.eager else "cuda_graph",
                "error": error,
            }
            rows.append(result)
            print(json.dumps(result, sort_keys=True))
    finally:
        kernel._CFG_ROUTER_K_SPLITS = original_splits

    valid = [
        row
        for row in rows
        if row["median_us"] is not None and row["error"] is None and row["experts_equal"]
    ]
    valid.sort(key=lambda row: float(row["median_us"]))
    print("\n== RANKING ==")
    for index, row in enumerate(valid, start=1):
        print(
            f"{index:02d}. splits={row['splits']}: {row['median_us']:.3f} us/call "
            f"({row['router_ms_per_token_48_layers']:.3f} ms/token for 48 layers)"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    if len(valid) != len(args.splits):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
