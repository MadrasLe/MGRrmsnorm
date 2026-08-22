#!/usr/bin/env python3
"""Gate Gemma 4 H512/GQA8 long full attention without a checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Optional

import torch

import megagemm.kernels.paged_attention as paged_attention


TensorFn = Callable[[], torch.Tensor]


def _measure_us(
    fn: TensorFn,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
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
    return samples


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_full_prefill_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("warmup/iterations must be positive and repeats >= 3")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "a100" not in gpu.lower().replace("-", " ").split():
        raise SystemExit(f"This shape gate requires an A100, found: {gpu}")

    torch.manual_seed(20260809)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    seq_len = 2048
    q_heads = 16
    kv_heads = 2
    head_dim = 512
    full_layers = 5
    b16_chunks = 2
    target_prefill_gap_ms = 1345.75
    configs = (
        {"case": "triton_bn32_w8", "block_n": 32, "num_warps": 8},
        {"case": "triton_bn64_w8", "block_n": 64, "num_warps": 8},
        {"case": "triton_bn128_w8", "block_n": 128, "num_warps": 8},
        {"case": "triton_bn64_w4", "block_n": 64, "num_warps": 4},
    )

    print("Gemma4 long full-attention prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"batch={batch_size} context={seq_len} q_heads={q_heads} "
        f"kv_heads={kv_heads} head_dim={head_dim} dtype=bf16"
    )
    print(
        "  B16 estimate: "
        f"{b16_chunks} x B{batch_size} chunks, {full_layers} full layers"
    )

    q = torch.randn(
        batch_size,
        q_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        kv_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    causal = torch.ones(
        seq_len,
        seq_len,
        device=device,
        dtype=torch.bool,
    ).tril_()
    explicit_mask = torch.zeros(
        batch_size,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    explicit_mask.masked_fill_(
        ~causal.unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )

    def baseline() -> torch.Tensor:
        return paged_attention.prefill_attention(
            q,
            k,
            v,
            is_causal=True,
            attn_mask=explicit_mask,
            scale=1.0,
        )

    def candidate_optional(config: dict) -> Optional[torch.Tensor]:
        return paged_attention.gemma4_long_full_prefill_attention(
            q,
            k,
            v,
            scale=1.0,
            block_n=int(config["block_n"]),
            num_warps=int(config["num_warps"]),
            num_stages=2,
            force=True,
        )

    def candidate_fn(config: dict) -> TensorFn:
        def run() -> torch.Tensor:
            output = candidate_optional(config)
            if output is None:
                failure = str(
                    getattr(
                        paged_attention,
                        "_GEMMA4_LONG_FULL_PREFILL_FAILURE",
                        "",
                    )
                    or "candidate returned None"
                )
                raise RuntimeError(failure)
            return output

        return run

    reference = baseline()
    torch.cuda.synchronize()
    candidate_rows: list[dict] = []
    for config in configs:
        row = dict(config)
        row["num_stages"] = 2
        try:
            output = candidate_optional(config)
            if output is None:
                raise RuntimeError(
                    str(
                        getattr(
                            paged_attention,
                            "_GEMMA4_LONG_FULL_PREFILL_FAILURE",
                            "",
                        )
                        or "candidate returned None"
                    )
                )
            repeated = candidate_fn(config)()
            torch.cuda.synchronize()
            difference = (reference.float() - output.float()).abs()
            row.update(
                {
                    "error": None,
                    "finite": bool(torch.isfinite(output).all().item()),
                    "cosine": _cosine(reference, output),
                    "max_abs_error": float(difference.max().item()),
                    "mean_abs_error": float(difference.mean().item()),
                    "repeat_exact": bool(torch.equal(output, repeated)),
                    "repeat_max_abs_error": float(
                        (output.float() - repeated.float()).abs().max().item()
                    ),
                }
            )
            row["correct"] = bool(
                row["finite"]
                and row["repeat_exact"]
                and row["cosine"] >= 0.9999
                and row["max_abs_error"] <= 0.125
            )
            del output, repeated, difference
        except Exception as exc:
            row.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                    "finite": False,
                    "cosine": None,
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "repeat_exact": False,
                    "repeat_max_abs_error": None,
                }
            )
        candidate_rows.append(row)
        torch.cuda.empty_cache()

    baseline_samples = _measure_us(
        baseline,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    for row, config in zip(candidate_rows, configs):
        if not row["correct"]:
            row["median_us"] = None
            row["samples_us"] = []
            continue
        samples = _measure_us(
            candidate_fn(config),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        row["samples_us"] = samples
        row["median_us"] = statistics.median(samples)
    baseline_recheck_samples = _measure_us(
        baseline,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )

    baseline_first_us = statistics.median(baseline_samples)
    baseline_recheck_us = statistics.median(baseline_recheck_samples)
    baseline_us = min(baseline_first_us, baseline_recheck_us)
    baseline_stability_ratio = (
        max(baseline_first_us, baseline_recheck_us) / baseline_us
    )
    baseline_rows = [
        {
            "case": "current_explicit_mask_sdpa",
            "median_us": baseline_first_us,
            "samples_us": baseline_samples,
        },
        {
            "case": "current_recheck",
            "median_us": baseline_recheck_us,
            "samples_us": baseline_recheck_samples,
        },
    ]
    for row in [*baseline_rows, *candidate_rows]:
        print(json.dumps(row, sort_keys=True))

    valid_candidates = [
        row
        for row in candidate_rows
        if row["correct"] and row["median_us"] is not None
    ]
    winner = (
        min(valid_candidates, key=lambda row: row["median_us"])
        if valid_candidates
        else None
    )
    stable = baseline_stability_ratio <= 1.05
    speedup = baseline_us / winner["median_us"] if winner is not None else None
    estimated_baseline_ms = baseline_us * full_layers * b16_chunks / 1000.0
    estimated_candidate_ms = (
        winner["median_us"] * full_layers * b16_chunks / 1000.0
        if winner is not None
        else None
    )
    estimated_savings_ms = (
        estimated_baseline_ms - estimated_candidate_ms
        if estimated_candidate_ms is not None
        else None
    )
    apply_change = bool(
        winner is not None
        and stable
        and speedup is not None
        and speedup >= float(args.minimum_speedup)
    )
    ranking = sorted(
        [*baseline_rows, *valid_candidates],
        key=lambda row: row["median_us"],
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_BASELINE",
        "apply_change": apply_change,
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "dtype": "bf16",
        },
        "b16_chunk_count": b16_chunks,
        "full_layers": full_layers,
        "baseline_us": baseline_us,
        "baseline_stability_ratio": baseline_stability_ratio,
        "stable": stable,
        "minimum_speedup": float(args.minimum_speedup),
        "winner": None if winner is None else winner["case"],
        "winner_config": (
            None
            if winner is None
            else {
                "block_n": winner["block_n"],
                "num_warps": winner["num_warps"],
                "num_stages": winner["num_stages"],
            }
        ),
        "candidate_us": None if winner is None else winner["median_us"],
        "speedup": speedup,
        "estimated_baseline_ms_b16_prefill": estimated_baseline_ms,
        "estimated_candidate_ms_b16_prefill": estimated_candidate_ms,
        "estimated_savings_ms_b16_prefill": estimated_savings_ms,
        "target_prefill_gap_ms": target_prefill_gap_ms,
        "estimated_gap_coverage": (
            estimated_savings_ms / target_prefill_gap_ms
            if estimated_savings_ms is not None
            else None
        ),
        "ranking": ranking,
        "candidates": candidate_rows,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    # A fully rejected candidate set is a valid benchmark result. Keep the
    # notebook successful after persisting KEEP_BASELINE.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
