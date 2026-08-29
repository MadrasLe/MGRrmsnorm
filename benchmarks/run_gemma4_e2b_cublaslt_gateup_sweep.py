#!/usr/bin/env python3
"""Search explicit cuBLASLt BF16 algorithms for E2B's two B8 gate-up shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
import torch.nn.functional as F


M = 8
K = 1536
OUTPUT_DIMS = (12288, 24576)
MAX_ABS_ERROR = 0.25
MIN_COSINE = 0.9999
MAX_BASELINE_STABILITY_RATIO = 1.04
MIN_SPEEDUP = 1.02


def _measure_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return samples


def _capture(
    name: str,
    call: Callable[[], torch.Tensor],
    out: torch.Tensor,
    reference: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        first = out.clone()
        graph.replay()
        torch.cuda.synchronize()
        second = out.clone()
        samples = _measure_graph(
            graph,
            iterations=iterations,
            repeats=repeats,
        )
        delta = first.float() - reference.float()
        return {
            "case": name,
            "median_us": float(statistics.median(samples)),
            "samples_us": samples,
            "finite": bool(torch.isfinite(first).all().item()),
            "max_abs_error": float(delta.abs().max().item()),
            "mean_abs_error": float(delta.abs().mean().item()),
            "cosine": float(
                F.cosine_similarity(
                    first.float().flatten(),
                    reference.float().flatten(),
                    dim=0,
                ).item()
            ),
            "repeat_exact": bool(torch.equal(first, second)),
            "error": None,
        }
    except Exception as exc:
        return {
            "case": name,
            "median_us": None,
            "samples_us": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("error") is None
        and row.get("finite") is True
        and row.get("repeat_exact") is True
        and float(row["max_abs_error"]) <= MAX_ABS_ERROR
        and float(row["cosine"]) >= MIN_COSINE
    )


@torch.inference_mode()
def run_shape(
    n: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from megagemm.kernels.mlp_prefill_native import (
        cublaslt_bf16_algorithm_count_cuda,
        cublaslt_bf16_linear_cuda,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260829 + n)
    x = torch.empty((M, K), device="cuda", dtype=torch.bfloat16).normal_(
        mean=0.0,
        std=0.2,
        generator=generator,
    )
    weight = torch.empty(
        (n, K),
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.02, generator=generator)
    weight_t = weight.t()
    reference = torch.mm(x, weight_t)
    torch.cuda.synchronize()

    baseline_out = torch.empty((M, n), device="cuda", dtype=torch.bfloat16)

    def baseline_call() -> torch.Tensor:
        return torch.mm(x, weight_t, out=baseline_out)

    baseline_first = _capture(
        "torch_mm_first",
        baseline_call,
        baseline_out,
        reference,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    algorithm_count = cublaslt_bf16_algorithm_count_cuda(
        x,
        weight,
        args.maximum_algorithms,
    )
    algorithms: list[dict[str, Any]] = []
    for algorithm_index in range(algorithm_count):
        candidate_out = torch.empty_like(baseline_out)

        def candidate_call(index: int = algorithm_index) -> torch.Tensor:
            return cublaslt_bf16_linear_cuda(
                x,
                weight,
                out=candidate_out,
                algorithm_index=index,
            )

        row = _capture(
            f"cublaslt_algo_{algorithm_index}",
            candidate_call,
            candidate_out,
            reference,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        row["algorithm_index"] = algorithm_index
        algorithms.append(row)
    baseline_recheck = _capture(
        "torch_mm_recheck",
        baseline_call,
        baseline_out,
        reference,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )

    baseline_valid = _valid(baseline_first) and _valid(baseline_recheck)
    if baseline_valid:
        baseline_samples = (
            float(baseline_first["median_us"]),
            float(baseline_recheck["median_us"]),
        )
        baseline_us = min(baseline_samples)
        stability = max(baseline_samples) / baseline_us
    else:
        baseline_us = None
        stability = None
    valid_algorithms = [row for row in algorithms if _valid(row)]
    valid = bool(
        baseline_us is not None
        and stability is not None
        and stability <= MAX_BASELINE_STABILITY_RATIO
        and valid_algorithms
    )
    winner = (
        min(valid_algorithms, key=lambda row: float(row["median_us"]))
        if valid
        else None
    )
    winner_us = float(winner["median_us"]) if winner is not None else None
    speedup = baseline_us / winner_us if winner_us is not None else None
    return {
        "shape": {"m": M, "n": n, "k": K, "dtype": "bf16"},
        "valid": valid,
        "algorithm_count": algorithm_count,
        "baseline_us": baseline_us,
        "baseline_stability_ratio": stability,
        "winner_algorithm_index": (
            int(winner["algorithm_index"]) if winner is not None else None
        ),
        "winner_us": winner_us,
        "speedup": speedup,
        "baseline": [baseline_first, baseline_recheck],
        "algorithms": algorithms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--maximum-algorithms", type=int, default=32)
    parser.add_argument("--minimum-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_e2b_cublaslt_gateup_l4.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if "l4" not in gpu.lower():
        raise SystemExit(f"NVIDIA L4 required, found {gpu}")
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    try:
        from megagemm.kernels.mlp_prefill_native import (
            HAS_CUBLASLT_BF16_LINEAR,
        )
    except Exception as exc:
        raise SystemExit(f"native extension import failed: {exc}") from exc
    if not HAS_CUBLASLT_BF16_LINEAR:
        raise SystemExit(
            "rmsnorm_cuda_ops was not rebuilt with the BF16 cuBLASLt sweep"
        )

    print("Gemma4 E2B/L4 B8 BF16 cuBLASLt gate-up algorithm sweep")
    print(f"  gpu: {gpu}")
    print("  shapes: M8 K1536 N12288 and N24576")
    print("  baseline: torch.mm with preallocated output")
    print("  candidate: explicit cuBLASLt heuristic with preallocated output")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    shapes = [run_shape(n, args) for n in OUTPUT_DIMS]
    valid = all(bool(row["valid"]) for row in shapes)
    functional = all(
        int(row.get("algorithm_count") or 0) > 0
        and all(case.get("error") is None for case in row.get("baseline", ()))
        and any(_valid(case) for case in row.get("algorithms", ()))
        for row in shapes
    )
    micro_wins = bool(
        valid
        and all(
            row["speedup"] is not None
            and float(row["speedup"]) >= args.minimum_speedup
            for row in shapes
        )
    )
    selected = {
        str(row["shape"]["n"]): row["winner_algorithm_index"]
        for row in shapes
    }
    result = {
        "decision": (
            "TEST_FULL_MODEL"
            if micro_wins
            else (
                "KEEP_TORCH_MM"
                if valid
                else "KEEP_TORCH_MM_UNSTABLE_MEASUREMENT"
            )
        ),
        "apply_change": False,
        "micro_wins": micro_wins,
        "valid": valid,
        "functional": functional,
        "minimum_speedup": args.minimum_speedup,
        "selected_algorithms": selected,
        "required_full_model_env": (
            {
                "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_DECODE": "1",
                "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_N12288_ALGO": str(
                    selected["12288"]
                ),
                "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_N24576_ALGO": str(
                    selected["24576"]
                ),
            }
            if micro_wins
            else None
        ),
        "promotion_rule": (
            "never promote from this micro-sweep; require a same-session "
            "loaded-model wall-time win over torch.mm"
        ),
        "shapes": shapes,
        "runtime": {
            "gpu": gpu,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    print("DECISION " + json.dumps(result, sort_keys=True))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    # An unstable timing gate is a conservative KEEP decision, not a broken
    # CUDA run.  Reserve a failing exit status for functional/build failures.
    return 0 if functional else 2


if __name__ == "__main__":
    raise SystemExit(main())
