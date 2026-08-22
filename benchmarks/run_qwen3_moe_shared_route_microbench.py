#!/usr/bin/env python3
"""Microbenchmark Qwen3 MoE shared-route decode without loading a model."""

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
    spec = importlib.util.spec_from_file_location("megagemm_qwen3_moe_shared_route_microbench", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        for _ in range(iterations):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--no-eager-fallback",
        action="store_true",
        help="Do not fall back to eager measurement if synthetic CUDA graph capture fails.",
    )
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_grad_enabled(False)

    kernel = _load_kernel_module()
    if not getattr(kernel, "_HAS_TRITON", False):
        raise SystemExit("Triton is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype) * 0.02
    gate_up = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    down = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    selected = torch.empty(args.rows, args.top_k, device="cuda", dtype=torch.int64)
    for row in range(args.rows):
        selected[row] = (torch.arange(args.top_k, device="cuda") + row * args.top_k) % args.num_experts
    route = torch.softmax(
        torch.randn(args.rows, args.top_k, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(dtype)

    cases = [
        {
            "name": "atomic_strided",
            "coalesced": False,
            "token_accum": False,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "atomic_coalesced",
            "coalesced": True,
            "token_accum": False,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "partial_strided",
            "coalesced": False,
            "token_accum": False,
            "partial_reduce": True,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "partial_coalesced",
            "coalesced": True,
            "token_accum": False,
            "partial_reduce": True,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "token_accum_strided",
            "coalesced": False,
            "token_accum": True,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "token_accum_coalesced",
            "coalesced": True,
            "token_accum": True,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 1,
        },
        {
            "name": "gate_k2_atomic_strided",
            "coalesced": False,
            "token_accum": False,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 2,
        },
        {
            "name": "gate_k4_atomic_strided",
            "coalesced": False,
            "token_accum": False,
            "partial_reduce": False,
            "split_gate": False,
            "gate_k_splits": 4,
        },
        {
            "name": "split_gate_atomic_strided",
            "coalesced": False,
            "token_accum": False,
            "partial_reduce": False,
            "split_gate": True,
            "gate_k_splits": 1,
        },
    ]

    old_values = {
        "_CFG_SHARED_ROUTE_DECODE": kernel._CFG_SHARED_ROUTE_DECODE,
        "_CFG_SHARED_ROUTE_COALESCED_WEIGHTS": kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS,
        "_CFG_SHARED_ROUTE_TOKEN_ACCUM": kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM,
        "_CFG_SHARED_ROUTE_PARTIAL_REDUCE": kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE,
        "_CFG_SHARED_ROUTE_SPLIT_GATE": kernel._CFG_SHARED_ROUTE_SPLIT_GATE,
        "_CFG_SHARED_ROUTE_GATE_K_SPLITS": kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS,
        "_CFG_SINGLE_ROW_GEMV": kernel._CFG_SINGLE_ROW_GEMV,
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE": kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE,
    }

    print("Qwen3 MoE shared-route decode microbenchmark")
    print("  gpu:", torch.cuda.get_device_name(0))
    print("  capability:", torch.cuda.get_device_capability(0))
    print(
        f"  shape: rows={args.rows} hidden={args.hidden_size} "
        f"intermediate={args.intermediate_size} experts={args.num_experts} "
        f"top_k={args.top_k} dtype={args.dtype}"
    )
    print("  measurement:", "eager" if args.eager else "cuda_graph")

    rows_out = []
    try:
        kernel._CFG_SHARED_ROUTE_DECODE = True
        kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
        kernel._CFG_SHARED_ROUTE_SPLIT_GATE = False
        kernel._CFG_SINGLE_ROW_GEMV = False
        kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = False

        kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = False
        kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
        reference = kernel.qwen3_moe_grouped_decode(
            hidden,
            gate_up,
            down,
            selected,
            route,
            activation="silu",
            out=torch.empty_like(hidden),
            workspace={},
        ).clone()
        torch.cuda.synchronize()

        for case in cases:
            name = str(case["name"])
            coalesced = bool(case["coalesced"])
            token_accum = bool(case["token_accum"])
            partial_reduce = bool(case["partial_reduce"])
            split_gate = bool(case["split_gate"])
            gate_k_splits = int(case["gate_k_splits"])
            kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = bool(coalesced)
            kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = bool(token_accum)
            kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = bool(partial_reduce)
            kernel._CFG_SHARED_ROUTE_SPLIT_GATE = bool(split_gate)
            kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = int(gate_k_splits)
            workspace = {}
            out = torch.empty_like(hidden)

            def call():
                return kernel.qwen3_moe_grouped_decode(
                    hidden,
                    gate_up,
                    down,
                    selected,
                    route,
                    activation="silu",
                    out=out,
                    workspace=workspace,
                )

            error = None
            samples = []
            max_abs_error = None
            cosine = None
            last_token_accum = None
            measurement = "eager" if args.eager else "cuda_graph"
            graph_error = None
            try:
                actual = call()
                torch.cuda.synchronize()
                max_abs_error = float((actual.float() - reference.float()).abs().max().item())
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        actual.flatten().float(),
                        reference.flatten().float(),
                        dim=0,
                    ).item()
                )
                last_token_accum = int(workspace.get("shared_route_decode_last_token_accum", -1))
                measure = _measure_eager_us if args.eager else _measure_graph_us
                samples = measure(
                    call,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if not args.eager and not args.no_eager_fallback:
                    graph_error = error
                    try:
                        torch.cuda.synchronize()
                        samples = _measure_eager_us(
                            call,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            repeats=args.repeats,
                        )
                        measurement = "eager_fallback_after_graph_error"
                        error = None
                    except Exception as eager_exc:
                        error += f" | eager_fallback={type(eager_exc).__name__}: {eager_exc}"

            median_us = statistics.median(samples) if samples else None
            row = {
                "case": name,
                "coalesced": bool(coalesced),
                "token_accum": bool(token_accum),
                "partial_reduce": bool(partial_reduce),
                "split_gate": bool(split_gate),
                "gate_k_splits": int(gate_k_splits),
                "measurement": measurement,
                "last_token_accum": last_token_accum,
                "last_partial_reduce": int(
                    workspace.get("shared_route_decode_last_partial_reduce", -1)
                ),
                "last_split_gate": int(workspace.get("shared_route_decode_last_split_gate", -1)),
                "last_gate_k_splits": int(
                    workspace.get("shared_route_decode_last_gate_k_splits", -1)
                ),
                "shared_route_disabled": int(workspace.get("shared_route_decode_disabled", 0) or 0),
                "shared_route_fail_reason": workspace.get("shared_route_decode_fail_reason"),
                "compact_disabled": int(workspace.get("expert_grouped_compact_decode_disabled", 0) or 0),
                "compact_fail_reason": workspace.get("expert_grouped_compact_decode_fail_reason"),
                "median_us": median_us,
                "moe_decode_ms_per_token_48_layers": (
                    median_us * 48.0 / 1000.0 if median_us is not None else None
                ),
                "max_abs_error": max_abs_error,
                "cosine": cosine,
                "samples_us": samples,
                "graph_error": graph_error,
                "error": error,
            }
            rows_out.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        for name, value in old_values.items():
            setattr(kernel, name, value)

    valid = [row for row in rows_out if row["median_us"] is not None and row["error"] is None]
    valid.sort(key=lambda row: float(row["median_us"]))
    print("\n== RANKING ==")
    for index, row in enumerate(valid, start=1):
        print(
            f"{index:02d}. {row['case']}: {row['median_us']:.3f} us/call "
            f"({row['moe_decode_ms_per_token_48_layers']:.3f} ms/token for 48 layers)"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": rows_out}, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
