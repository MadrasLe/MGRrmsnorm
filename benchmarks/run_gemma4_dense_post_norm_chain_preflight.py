"""Correctness and latency gate for the experimental Gemma 4 dense tail.

This deliberately avoids loading a checkpoint.  It exercises the exact E2B
and E4B hidden widths on CUDA, compares the fused kernel with the staged eager
sequence it replaces, and reports same-process CUDA-event timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch

from megagemm.kernels import (
    rmsnorm_triton,
    rmsnorm_triton_residual_scale_next,
)
from megagemm.models.llama import _decode_rmsnorm


def timed_us(fn: Callable[[], None], *, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
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
    return float(statistics.median(samples))


def run_case(rows: int, hidden_size: int, *, iterations: int, repeats: int) -> dict:
    torch.manual_seed(20260825 + rows + hidden_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    branch = torch.randn((rows, hidden_size), device=device, dtype=dtype)
    residual_seed = torch.randn_like(branch)
    norm_weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    next_norm_weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    layer_scalar = torch.tensor([0.91], device=device, dtype=dtype)

    base_residual = torch.empty_like(branch)
    base_post_norm = torch.empty_like(branch)
    base_next_norm = torch.empty_like(branch)
    fused_residual = torch.empty_like(branch)
    fused_next_norm = torch.empty_like(branch)
    prefer_triton = hidden_size == 1536

    def baseline_exact() -> None:
        base_residual.copy_(residual_seed)
        rmsnorm_triton(branch, norm_weight, eps, False, out=base_post_norm)
        torch.add(base_residual, base_post_norm, out=base_residual)
        base_residual.mul_(layer_scalar)
        rmsnorm_triton(
            base_residual,
            next_norm_weight,
            eps,
            False,
            out=base_next_norm,
        )

    def baseline_runtime() -> None:
        base_residual.copy_(residual_seed)
        _decode_rmsnorm(
            branch,
            norm_weight,
            eps,
            False,
            out=base_post_norm,
            prefer_triton=prefer_triton,
        )
        torch.add(base_residual, base_post_norm, out=base_residual)
        base_residual.mul_(layer_scalar)
        _decode_rmsnorm(
            base_residual,
            next_norm_weight,
            eps,
            False,
            out=base_next_norm,
            prefer_triton=prefer_triton,
        )

    def candidate() -> None:
        fused_residual.copy_(residual_seed)
        rmsnorm_triton_residual_scale_next(
            branch,
            fused_residual,
            norm_weight,
            layer_scalar,
            next_norm_weight,
            eps,
            norm_offset=False,
            next_norm_offset=False,
            out_hidden=fused_residual,
            next_norm_out=fused_next_norm,
        )

    baseline_exact()
    exact_hidden = base_residual.clone()
    exact_next_norm = base_next_norm.clone()
    baseline_runtime()
    runtime_hidden = base_residual.clone()
    runtime_next_norm = base_next_norm.clone()
    candidate()
    torch.cuda.synchronize()
    hidden_equal = bool(torch.equal(exact_hidden, fused_residual))
    next_equal = bool(torch.equal(exact_next_norm, fused_next_norm))
    runtime_hidden_close = bool(
        torch.allclose(runtime_hidden, fused_residual, rtol=0.02, atol=0.03125)
    )
    runtime_next_close = bool(
        torch.allclose(runtime_next_norm, fused_next_norm, rtol=0.02, atol=0.03125)
    )
    hidden_diff = float((exact_hidden.float() - fused_residual.float()).abs().max())
    next_diff = float((exact_next_norm.float() - fused_next_norm.float()).abs().max())
    runtime_hidden_diff = float(
        (runtime_hidden.float() - fused_residual.float()).abs().max()
    )
    runtime_next_diff = float(
        (runtime_next_norm.float() - fused_next_norm.float()).abs().max()
    )

    baseline_us = timed_us(
        baseline_runtime,
        warmup=20,
        iterations=iterations,
        repeats=repeats,
    )
    candidate_us = timed_us(candidate, warmup=20, iterations=iterations, repeats=repeats)
    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "runtime_rmsnorm": "triton" if prefer_triton else "native-if-available",
        "hidden_exact": hidden_equal,
        "next_norm_exact": next_equal,
        "runtime_hidden_close": runtime_hidden_close,
        "runtime_next_norm_close": runtime_next_close,
        "max_abs_hidden": hidden_diff,
        "max_abs_next_norm": next_diff,
        "max_abs_runtime_hidden": runtime_hidden_diff,
        "max_abs_runtime_next_norm": runtime_next_diff,
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "speedup": baseline_us / candidate_us if candidate_us > 0.0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, choices=(1536, 2560), required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERRO: o preflight requer CUDA")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu:
        raise SystemExit(f"ERRO: esperado NVIDIA L4, encontrado {gpu}")

    cases = [
        run_case(rows, args.hidden_size, iterations=args.iterations, repeats=args.repeats)
        for rows in (1, 8)
    ]
    correctness_ok = all(
        case["hidden_exact"]
        and case["next_norm_exact"]
        and case["runtime_hidden_close"]
        and case["runtime_next_norm_close"]
        for case in cases
    )
    latency_ok = all(case["speedup"] >= args.min_speedup for case in cases)
    report = {
        "status": "passed" if correctness_ok and latency_ok else "failed",
        "gpu": gpu,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": "bf16",
        "correctness_ok": correctness_ok,
        "latency_ok": latency_ok,
        "min_speedup": args.min_speedup,
        "cases": cases,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
