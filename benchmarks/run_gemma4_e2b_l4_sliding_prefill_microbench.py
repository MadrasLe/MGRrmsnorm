#!/usr/bin/env python3
"""Gate the exact Gemma 4 E2B/L4 long-sliding prefill kernel."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from megagemm.kernels import paged_attention


TensorFn = Callable[[], torch.Tensor]


def _measure_us(
    fn: TensorFn,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
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
    return {
        "median_us": statistics.median(samples),
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


def _required(output: Optional[torch.Tensor]) -> torch.Tensor:
    if output is None:
        failure = getattr(
            paged_attention,
            "_GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE",
            "",
        )
        raise RuntimeError(
            "E2B/L4 candidate rejected the exact shape"
            + (f": {failure}" if failure else "")
        )
    return output


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2057,
        help="Runtime length; 2057 matches prompt=2048 plus chat template.",
    )
    parser.add_argument(
        "--winner-only",
        action="store_true",
        help="Retest only the selected g4/BM8/BN64/4-warp candidate.",
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--maximum-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="/tmp/gemma4_e2b_l4_sliding_prefill_microbench.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("warmup/iterations must be positive and repeats >= 3")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "l4" not in gpu.lower().replace("-", " ").split():
        raise SystemExit(f"This exact gate requires an NVIDIA L4, found: {gpu}")

    torch.manual_seed(20260827)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    q_heads = 8
    kv_heads = 1
    seq_len = int(args.seq_len)
    head_dim = 256
    sliding_window = 512
    scale = 1.0 / math.sqrt(head_dim)
    if not 2048 <= seq_len <= 2304:
        raise SystemExit("--seq-len must be in the production gate [2048, 2304]")

    print("Gemma 4 E2B/L4 sliding-prefill Triton gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(
        "  shape: "
        f"B{batch_size} Q{q_heads}/KV{kv_heads} S{seq_len} "
        f"H{head_dim} W{sliding_window} BF16"
    )
    print("  model_download: disabled")
    print("  vllm_install: disabled")

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
    allowed = (
        (positions[None, :] <= positions[:, None])
        & (
            positions[None, :]
            >= positions[:, None] - sliding_window + 1
        )
    )
    local_mask = torch.zeros(
        1,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    local_mask.masked_fill_(
        ~allowed.view(1, 1, seq_len, seq_len),
        float("-inf"),
    )

    def baseline() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=local_mask,
            is_causal=False,
            enable_gqa=True,
            scale=scale,
        )

    configs = (
        {
            "case": "g1_bm32_bn64_w8_s2",
            "group_heads": 1,
            "block_m": 32,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g2_bm16_bn64_w8_s2",
            "group_heads": 2,
            "block_m": 16,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g4_bm8_bn64_w8_s2",
            "group_heads": 4,
            "block_m": 8,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g8_bm4_bn64_w8_s2",
            "group_heads": 8,
            "block_m": 4,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g2_bm16_bn128_w8_s2",
            "group_heads": 2,
            "block_m": 16,
            "block_n": 128,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g4_bm8_bn128_w8_s2",
            "group_heads": 4,
            "block_m": 8,
            "block_n": 128,
            "num_warps": 8,
            "num_stages": 2,
        },
        {
            "case": "g2_bm16_bn64_w4_s2",
            "group_heads": 2,
            "block_m": 16,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 2,
        },
        {
            "case": "g4_bm8_bn64_w4_s2",
            "group_heads": 4,
            "block_m": 8,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 2,
        },
    )
    if args.winner_only:
        configs = tuple(
            config
            for config in configs
            if config["case"] == "g4_bm8_bn64_w4_s2"
        )

    def candidate(config: dict[str, Any]) -> TensorFn:
        def invoke() -> torch.Tensor:
            return _required(
                paged_attention.gemma4_e2b_l4_sliding_prefill_attention(
                    q,
                    k,
                    v,
                    sliding_window=sliding_window,
                    scale=scale,
                    group_heads=int(config["group_heads"]),
                    block_m=int(config["block_m"]),
                    block_n=int(config["block_n"]),
                    num_warps=int(config["num_warps"]),
                    num_stages=int(config["num_stages"]),
                    force=True,
                )
            )

        return invoke

    reference = baseline().detach().clone()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(reference).all().item()):
        raise SystemExit("SDPA baseline produced non-finite output")

    rows: list[dict[str, Any]] = []

    def profile_case(
        name: str,
        fn: TensorFn,
        config: Optional[dict[str, Any]],
    ) -> None:
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
            timing = _measure_us(
                fn,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            row = {
                "case": name,
                "config": config,
                "error": None,
                "correct": correct,
                "finite": finite,
                "repeat_exact": repeat_exact,
                "repeat_max_abs_error": float(repeat_delta.max().item()),
                "cosine": cosine,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                **timing,
            }
            del first, second, delta, repeat_delta
        except Exception as exc:
            row = {
                "case": name,
                "config": config,
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

    profile_case("sdpa_current", baseline, None)
    for config in configs:
        profile_case(str(config["case"]), candidate(config), config)
    profile_case("sdpa_recheck", baseline, None)

    by_name = {row["case"]: row for row in rows}
    baseline_row = by_name["sdpa_recheck"]
    baseline_us = (
        float(baseline_row["median_us"])
        if baseline_row.get("correct")
        and baseline_row.get("median_us") is not None
        else None
    )
    candidates = [
        row
        for row in rows
        if row["case"] not in {"sdpa_current", "sdpa_recheck"}
        and row.get("correct")
        and row.get("median_us") is not None
    ]
    winner = (
        min(candidates, key=lambda row: float(row["median_us"]))
        if candidates
        else None
    )
    speedup = (
        baseline_us / float(winner["median_us"])
        if baseline_us is not None and winner is not None
        else None
    )
    baseline_samples = [
        float(sample)
        for name in ("sdpa_current", "sdpa_recheck")
        for sample in by_name[name].get("samples_us", [])
    ]
    winner_samples = (
        [float(sample) for sample in winner.get("samples_us", [])]
        if winner is not None
        else []
    )
    conservative_speedup = (
        min(baseline_samples) / max(winner_samples)
        if baseline_samples and winner_samples
        else None
    )
    stable = bool(
        winner is not None
        and float(winner["spread_ratio"]) <= float(args.maximum_spread)
        and float(baseline_row["spread_ratio"]) <= float(args.maximum_spread)
    )
    sample_ranges_dominate = bool(
        conservative_speedup is not None
        and conservative_speedup >= float(args.minimum_speedup)
    )
    apply_change = bool(
        speedup is not None
        and speedup >= float(args.minimum_speedup)
        and (stable or sample_ranges_dominate)
    )
    savings_ms = (
        (baseline_us - float(winner["median_us"])) * 28.0 / 1000.0
        if baseline_us is not None and winner is not None
        else None
    )
    summary = {
        "decision": "TEST_FULL_MODEL" if apply_change else "KEEP_SDPA",
        "apply_change": apply_change,
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "sliding_window": sliding_window,
            "dtype": "bf16",
        },
        "winner": winner["case"] if winner is not None else None,
        "winner_config": winner.get("config") if winner is not None else None,
        "baseline_us": baseline_us,
        "winner_us": (
            float(winner["median_us"]) if winner is not None else None
        ),
        "speedup": speedup,
        "conservative_speedup": conservative_speedup,
        "stable": stable,
        "sample_ranges_dominate": sample_ranges_dominate,
        "decision_basis": (
            "stable_medians"
            if stable
            else "non_overlapping_sample_ranges"
            if sample_ranges_dominate
            else "insufficient_gain"
        ),
        "estimated_savings_ms_28_layers": savings_ms,
        "minimum_speedup": float(args.minimum_speedup),
        "maximum_spread": float(args.maximum_spread),
        "cases": rows,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
