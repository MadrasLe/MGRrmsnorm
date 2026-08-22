#!/usr/bin/env python3
"""Gate the A100 Gemma 4 long sliding-prefill kernel without a checkpoint."""

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


def _candidate_error() -> str:
    return str(
        getattr(
            paged_attention,
            "_GEMMA4_LONG_SLIDING_PREFILL_FAILURE",
            "",
        )
        or "candidate returned None for the required A100 topology"
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_sliding_prefill_a100.json",
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
    kv_heads = 8
    head_dim = 256
    sliding_window = 1024
    sliding_layers = 25
    b16_chunks = 2

    print("Gemma4 long sliding-prefill kernel gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"batch={batch_size} context={seq_len} q_heads={q_heads} "
        f"kv_heads={kv_heads} head_dim={head_dim} window={sliding_window} "
        "dtype=bf16"
    )
    print(
        "  B16 estimate: "
        f"{b16_chunks} x B{batch_size} chunks, {sliding_layers} sliding layers"
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

    positions = torch.arange(seq_len, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    allowed = (
        (key_positions <= query_positions)
        & (key_positions >= query_positions - sliding_window + 1)
    )
    explicit_mask = torch.zeros(
        batch_size,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    explicit_mask.masked_fill_(
        ~allowed.unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )

    def baseline() -> torch.Tensor:
        return paged_attention.prefill_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=explicit_mask,
            scale=1.0,
        )

    def candidate_optional() -> Optional[torch.Tensor]:
        return paged_attention.gemma4_long_sliding_prefill_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            scale=1.0,
            force=True,
        )

    def candidate() -> torch.Tensor:
        output = candidate_optional()
        if output is None:
            raise RuntimeError(_candidate_error())
        return output

    reference = baseline()
    candidate_output = candidate_optional()
    if candidate_output is None:
        summary = {
            "decision": "REJECT_CANDIDATE",
            "apply_change": False,
            "correct": False,
            "error": _candidate_error(),
            "gpu": gpu,
            "capability": list(capability),
        }
        print("DECISION " + json.dumps(summary, sort_keys=True))
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path}")
        return 2

    candidate_repeat = candidate()
    torch.cuda.synchronize()
    reference_f32 = reference.float()
    candidate_f32 = candidate_output.float()
    difference = (reference_f32 - candidate_f32).abs()
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    cosine = _cosine(reference, candidate_output)
    finite = bool(torch.isfinite(candidate_output).all().item())
    repeat_max_abs_error = float(
        (candidate_output.float() - candidate_repeat.float()).abs().max().item()
    )
    repeat_exact = bool(torch.equal(candidate_output, candidate_repeat))
    correct = bool(
        finite
        and repeat_exact
        and cosine >= 0.9999
        and max_abs_error <= 0.125
    )

    del reference_f32, candidate_f32, difference, candidate_repeat
    torch.cuda.empty_cache()

    baseline_samples = _measure_us(
        baseline,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    candidate_samples = _measure_us(
        candidate,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
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
    candidate_us = statistics.median(candidate_samples)
    speedup = baseline_us / candidate_us
    estimated_baseline_ms = (
        baseline_us * sliding_layers * b16_chunks / 1000.0
    )
    estimated_candidate_ms = (
        candidate_us * sliding_layers * b16_chunks / 1000.0
    )
    estimated_savings_ms = estimated_baseline_ms - estimated_candidate_ms
    stable = baseline_stability_ratio <= 1.05
    apply_change = bool(
        correct
        and stable
        and speedup >= float(args.minimum_speedup)
    )

    result_rows = [
        {
            "case": "current_explicit_mask_sdpa",
            "median_us": baseline_first_us,
            "samples_us": baseline_samples,
        },
        {
            "case": "triton_long_sliding",
            "median_us": candidate_us,
            "samples_us": candidate_samples,
        },
        {
            "case": "current_recheck",
            "median_us": baseline_recheck_us,
            "samples_us": baseline_recheck_samples,
        },
    ]
    for row in result_rows:
        print(json.dumps(row, sort_keys=True))

    summary = {
        "decision": "APPLY" if apply_change else "KEEP_BASELINE",
        "apply_change": apply_change,
        "correct": correct,
        "stable": stable,
        "finite": finite,
        "repeat_exact": repeat_exact,
        "repeat_max_abs_error": repeat_max_abs_error,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "sliding_window": sliding_window,
            "dtype": "bf16",
        },
        "b16_chunk_count": b16_chunks,
        "sliding_layers": sliding_layers,
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "baseline_stability_ratio": baseline_stability_ratio,
        "minimum_speedup": float(args.minimum_speedup),
        "speedup": speedup,
        "estimated_baseline_ms_b16_prefill": estimated_baseline_ms,
        "estimated_candidate_ms_b16_prefill": estimated_candidate_ms,
        "estimated_savings_ms_b16_prefill": estimated_savings_ms,
        "ranking": sorted(result_rows, key=lambda row: row["median_us"]),
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0 if correct else 2


if __name__ == "__main__":
    raise SystemExit(main())
