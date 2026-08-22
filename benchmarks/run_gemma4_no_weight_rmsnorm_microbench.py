"""Gemma4 no-weight RMSNorm benchmark for router and V-Norm (no model)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.kernels.rmsnorm_triton import rmsnorm_triton_no_weight


def reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)


def measure(
    fn: Callable[[], torch.Tensor], *, warmup: int, iterations: int, repeats: int
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


def error(reference_out: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference_out.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    return {
        "max_abs_error": float((ref - cand).abs().max().item()),
        "cosine": float(F.cosine_similarity(ref, cand, dim=0).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--out-json", default="bench_results/gemma4_no_weight_rmsnorm_a100.json"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    eps = 1e-6
    cases = (
        ("router_input_norm", (25, 2816), 30),
        ("sliding_v_norm", (1, 8, 25, 256), 25),
        ("full_v_norm", (1, 2, 25, 512), 5),
    )
    print("Gemma4 no-weight RMSNorm microbenchmark")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  dtype: {args.dtype}")

    torch.manual_seed(20260712)
    rows = []
    with torch.inference_mode():
        for name, shape, calls in cases:
            x = torch.randn(shape, device="cuda", dtype=dtype)
            ref_out = reference(x, eps)
            for backend, fn in (
                ("pytorch", lambda x=x: reference(x, eps)),
                ("triton", lambda x=x: rmsnorm_triton_no_weight(x, eps)),
            ):
                timing = measure(
                    fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                candidate = fn()
                torch.cuda.synchronize()
                row = {
                    "case": name,
                    "backend": backend,
                    "shape": list(shape),
                    "calls_per_prefill": calls,
                    **timing,
                    **error(ref_out, candidate),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    lookup = {(row["case"], row["backend"]): row for row in rows}
    pytorch_us = sum(
        int(calls) * float(lookup[(name, "pytorch")]["median_us"])
        for name, _, calls in cases
    )
    triton_us = sum(
        int(calls) * float(lookup[(name, "triton")]["median_us"])
        for name, _, calls in cases
    )
    correct = all(
        row["max_abs_error"] <= 0.03125 and row["cosine"] >= 0.999
        for row in rows
        if row["backend"] == "triton"
    )
    speedup = pytorch_us / triton_us
    decision = {
        "decision": "KEEP_TRITON_NO_WEIGHT" if correct and speedup >= 1.02 else "REVERT",
        "correct": correct,
        "pytorch_ms_per_prefill": pytorch_us / 1000.0,
        "triton_ms_per_prefill": triton_us / 1000.0,
        "estimated_savings_ms": (pytorch_us - triton_us) / 1000.0,
        "speedup": speedup,
    }
    print("\nDECISION " + json.dumps(decision, sort_keys=True))
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "results": rows,
        "decision": decision,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
