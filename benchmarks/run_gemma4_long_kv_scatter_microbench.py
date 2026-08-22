"""Model-free A100 gate for token-tiled Gemma 4 long-prefill K/V scatter."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from megagemm.kernels.paged_attention import (
    paged_kv_cache_scatter,
    paged_kv_cache_scatter_token_tiled,
)


CaseFn = Callable[[], None]


def _measure_us(
    fn: CaseFn,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
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
        samples.append(float(start.elapsed_time(end) * 1000.0 / iterations))
    return {
        "median_us": float(statistics.median(samples)),
        "samples_us": samples,
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--near-tie-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_kv_scatter_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("warmup/iterations must be positive and repeats >= 3")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "a100" not in gpu.lower().replace("-", " ").split():
        raise SystemExit(f"This shape gate requires an A100, found: {gpu}")
    if vram_gb < 70.0:
        raise SystemExit(f"This gate requires an A100 80GB, found {vram_gb:.2f}GB")

    torch.manual_seed(20260812)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    context = 2_048
    rows = batch_size * context
    block_size = 16
    target_gap_ms = 266.688685

    print("Gemma4 long K/V-scatter token-tiling gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        f"  shared shape: batch={batch_size} context={context} rows={rows} "
        f"block_size={block_size} dtype=bf16"
    )
    print("  candidates: current BT1 vs token-tiled BT2/BT4/BT8")

    def run_shape(
        *,
        attention_type: str,
        kv_heads: int,
        head_dim: int,
        layer_invocations: int,
    ) -> dict[str, Any]:
        blocks_per_seq = context // block_size
        num_blocks = batch_size * blocks_per_seq
        token_idx = torch.arange(rows, device=device, dtype=torch.long)
        seq_idx = token_idx // context
        seq_position = token_idx - seq_idx * context
        physical_blocks = seq_idx * blocks_per_seq + seq_position // block_size
        block_offsets = seq_position % block_size

        k = torch.empty(
            (rows, kv_heads, head_dim), device=device, dtype=dtype
        ).normal_(0.0, 0.02)
        v = torch.empty_like(k).normal_(0.0, 0.02)
        cache = torch.empty(
            (num_blocks, 2, kv_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )

        def current() -> None:
            if not paged_kv_cache_scatter(
                k,
                v,
                cache,
                physical_blocks,
                block_offsets,
            ):
                raise RuntimeError("current Triton K/V scatter is unavailable")

        def tiled(tokens_per_program: int) -> CaseFn:
            def run() -> None:
                if not paged_kv_cache_scatter_token_tiled(
                    k,
                    v,
                    cache,
                    physical_blocks,
                    block_offsets,
                    tokens_per_program=tokens_per_program,
                ):
                    raise RuntimeError("token-tiled Triton K/V scatter is unavailable")

            return run

        cache.zero_()
        current()
        torch.cuda.synchronize()
        reference = cache.detach().clone()

        measured: list[dict[str, Any]] = []

        def profile_case(
            name: str,
            fn: CaseFn,
            tokens_per_program: int,
        ) -> None:
            try:
                cache.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                first = cache.detach().clone()
                fn()
                torch.cuda.synchronize()
                repeat_exact = bool(torch.equal(first, cache))
                exact = bool(torch.equal(reference, cache))
                max_abs_error = float(
                    (reference.float() - cache.float()).abs().max().item()
                )
                correct = bool(exact and repeat_exact and max_abs_error == 0.0)
                timing = _measure_us(
                    fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                row = {
                    "attention_type": attention_type,
                    "case": name,
                    "tokens_per_program": tokens_per_program,
                    "error": None,
                    "correct": correct,
                    "exact": exact,
                    "max_abs_error": max_abs_error,
                    "repeat_exact": repeat_exact,
                    **timing,
                }
                row["sample_spread_ratio"] = (
                    max(timing["samples_us"]) / min(timing["samples_us"])
                )
                del first
            except Exception as exc:
                row = {
                    "attention_type": attention_type,
                    "case": name,
                    "tokens_per_program": tokens_per_program,
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                    "median_us": None,
                    "samples_us": [],
                    "sample_spread_ratio": None,
                }
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            measured.append(row)
            print(json.dumps(row, sort_keys=True))
            gc.collect()

        profile_case("current_bt1", current, 1)
        profile_case("tiled_bt2", tiled(2), 2)
        profile_case("tiled_bt4", tiled(4), 4)
        profile_case("tiled_bt8", tiled(8), 8)
        profile_case("current_recheck", current, 1)

        by_name = {row["case"]: row for row in measured}
        cold_baseline = by_name["current_bt1"]
        baseline = by_name["current_recheck"]
        baseline_us = (
            float(baseline["median_us"])
            if baseline.get("correct") and baseline.get("median_us") is not None
            else None
        )
        candidates = [
            row
            for row in measured
            if row["case"].startswith("tiled_")
            and row.get("correct")
            and row.get("median_us") is not None
        ]
        raw_winner = (
            min(candidates, key=lambda row: float(row["median_us"]))
            if candidates
            else None
        )
        near_tied = (
            [
                row
                for row in candidates
                if float(row["median_us"])
                <= float(raw_winner["median_us"]) * float(args.near_tie_ratio)
            ]
            if raw_winner is not None
            else []
        )
        measured_winner = (
            min(
                near_tied,
                key=lambda row: (
                    float(row["sample_spread_ratio"]),
                    int(row["tokens_per_program"]),
                    float(row["median_us"]),
                ),
            )
            if near_tied
            else None
        )
        baseline_spread = baseline.get("sample_spread_ratio")
        candidate_spread = (
            measured_winner.get("sample_spread_ratio")
            if measured_winner is not None
            else None
        )
        stable = bool(
            baseline_spread is not None
            and float(baseline_spread) <= float(args.maximum_baseline_spread)
            and candidate_spread is not None
            and float(candidate_spread) <= float(args.maximum_candidate_spread)
        )
        speedup = (
            baseline_us / float(measured_winner["median_us"])
            if baseline_us is not None and measured_winner is not None
            else None
        )
        apply_change = bool(
            stable
            and speedup is not None
            and speedup >= float(args.minimum_speedup)
        )
        selected_us = (
            float(measured_winner["median_us"])
            if apply_change and measured_winner is not None
            else baseline_us
        )
        baseline_ms = (
            baseline_us * layer_invocations / 1000.0
            if baseline_us is not None
            else None
        )
        selected_ms = (
            selected_us * layer_invocations / 1000.0
            if selected_us is not None
            else None
        )
        summary = {
            "attention_type": attention_type,
            "decision": "APPLY" if apply_change else "KEEP_CURRENT",
            "apply_change": apply_change,
            "winner": (
                measured_winner["case"]
                if apply_change and measured_winner is not None
                else "current_bt1"
            ),
            "measured_winner": (
                measured_winner["case"] if measured_winner is not None else None
            ),
            "raw_measured_winner": (
                raw_winner["case"] if raw_winner is not None else None
            ),
            "selected_tokens_per_program": (
                int(measured_winner["tokens_per_program"])
                if apply_change and measured_winner is not None
                else 1
            ),
            "baseline_us": baseline_us,
            "cold_baseline_us": (
                cold_baseline.get("median_us")
                if cold_baseline.get("correct")
                else None
            ),
            "selected_us": selected_us,
            "baseline_sample_spread_ratio": baseline_spread,
            "candidate_sample_spread_ratio": candidate_spread,
            "stable": stable,
            "speedup": speedup,
            "layer_invocations": layer_invocations,
            "estimated_baseline_ms_b16_prefill": baseline_ms,
            "estimated_selected_ms_b16_prefill": selected_ms,
            "estimated_savings_ms_b16_prefill": (
                baseline_ms - selected_ms
                if baseline_ms is not None and selected_ms is not None
                else None
            ),
            "shape": {
                "rows": rows,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "block_size": block_size,
                "num_blocks": num_blocks,
            },
            "cases": measured,
        }
        print("SCATTER_DECISION " + json.dumps(summary, sort_keys=True))

        del (
            token_idx,
            seq_idx,
            seq_position,
            physical_blocks,
            block_offsets,
            k,
            v,
            cache,
            reference,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return summary

    sliding = run_shape(
        attention_type="sliding",
        kv_heads=8,
        head_dim=256,
        layer_invocations=25 * 2,
    )
    full = run_shape(
        attention_type="full",
        kv_heads=2,
        head_dim=512,
        layer_invocations=5 * 2,
    )

    baseline_parts = (
        sliding.get("estimated_baseline_ms_b16_prefill"),
        full.get("estimated_baseline_ms_b16_prefill"),
    )
    selected_parts = (
        sliding.get("estimated_selected_ms_b16_prefill"),
        full.get("estimated_selected_ms_b16_prefill"),
    )
    baseline_ms = (
        sum(float(value) for value in baseline_parts)
        if all(value is not None for value in baseline_parts)
        else None
    )
    selected_ms = (
        sum(float(value) for value in selected_parts)
        if all(value is not None for value in selected_parts)
        else None
    )
    savings_ms = (
        baseline_ms - selected_ms
        if baseline_ms is not None and selected_ms is not None
        else None
    )
    summary = {
        "decision": (
            "APPLY_SELECTED" if sliding["apply_change"] or full["apply_change"]
            else "KEEP_CURRENT"
        ),
        "apply_change": bool(sliding["apply_change"] or full["apply_change"]),
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "context": context,
            "rows": rows,
            "block_size": block_size,
            "dtype": "bf16",
        },
        "minimum_speedup": float(args.minimum_speedup),
        "near_tie_ratio": float(args.near_tie_ratio),
        "estimated_baseline_ms_b16_prefill": baseline_ms,
        "estimated_selected_ms_b16_prefill": selected_ms,
        "estimated_savings_ms_b16_prefill": savings_ms,
        "target_prefill_gap_ms": target_gap_ms,
        "estimated_gap_coverage": (
            savings_ms / target_gap_ms if savings_ms is not None else None
        ),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "sliding": sliding,
        "full": full,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
