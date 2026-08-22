#!/usr/bin/env python3
"""Gate contiguous deterministic partials under checkpoint-like route skew."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from megagemm.kernels.qwen3_moe import qwen3_moe_segmented_prefill


TensorFn = Callable[[], torch.Tensor]

ROWS = 32_768
HIDDEN_DIM = 2_816
INTERMEDIATE_DIM = 704
NUM_EXPERTS = 128
TOP_K = 8
LAYER_INVOCATIONS = 30

# v23 observed per-layer global-padding expansion from 7.641x to 15.281x.
# These counts reproduce the low, middle, and high ends without a checkpoint.
SKEW_PROFILES = (
    ("skew_7p64x", 15_648),
    ("skew_11x", 22_528),
    ("skew_15p28x", 31_296),
)

BASE_CONFIG: dict[str, Any] = {
    "force": True,
    "block_m": 64,
    "block_n": 256,
    "block_k": 64,
    "fused_gate_block_n": 128,
    "num_warps": 4,
    "num_stages": 3,
    "fused_gate": True,
    "dense_grid": False,
    "route_scatter": True,
    "compact_route_pack": False,
    "async_tiles_max_assignments": ROWS * TOP_K,
    "sorted_partial": False,
    "deterministic_reduce": True,
}

CANDIDATE_CHANGES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("sorted_contiguous_partial", {"sorted_partial": True}),
)


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
        "median_us": float(statistics.median(samples)),
        "samples_us": samples,
        "sample_spread_ratio": float(max(samples) / min(samples)),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


def _skewed_route(
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    heavy_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Build unique top-k routes with one expert receiving heavy_count rows."""
    if not 0 <= int(heavy_count) <= int(rows):
        raise ValueError("heavy_count must be between zero and rows")
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).reshape(-1, 1)
    top_ids = torch.arange(top_k, device=device, dtype=torch.int64).reshape(1, -1)
    selected = 1 + (row_ids * 17 + top_ids * 13).remainder(num_experts - 1)
    if heavy_count:
        selected[:heavy_count, 0] = 0
    return selected.contiguous()


def _config(changes: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(BASE_CONFIG)
    if changes:
        config.update(changes)
    return config


def _contract_matches(workspace: dict[str, Any], config: dict[str, Any]) -> bool:
    sorted_partial_expected = bool(config["sorted_partial"])
    return bool(
        int(workspace.get("segmented_prefill_deterministic_reduce", 0)) == 1
        and int(workspace.get("segmented_prefill_partial_reduce", 0)) == 1
        and workspace.get("segmented_prefill_partial_dtype") == "torch.float32"
        and int(workspace.get("segmented_prefill_route_scatter", 0)) == 1
        and int(workspace.get("segmented_prefill_selected_block_m", 0))
        == int(config["block_m"])
        and int(workspace.get("segmented_prefill_selected_block_n", 0))
        == int(config["block_n"])
        and int(workspace.get("segmented_prefill_selected_block_k", 0))
        == int(config["block_k"])
        and int(workspace.get("segmented_prefill_selected_fused_gate_block_n", 0))
        == int(config["fused_gate_block_n"])
        and int(workspace.get("segmented_prefill_selected_num_warps", 0))
        == int(config["num_warps"])
        and int(workspace.get("segmented_prefill_selected_num_stages", 0))
        == int(config["num_stages"])
        and bool(workspace.get("segmented_prefill_async_tiles", 0))
        and int(workspace.get("segmented_prefill_max_tiles", 0)) > 0
        and bool(workspace.get("segmented_prefill_sorted_partial", 0))
        == sorted_partial_expected
        and (
            not sorted_partial_expected
            or int(workspace.get("segmented_prefill_slot_inverse_bytes", 0))
            == ROWS * TOP_K * 8
        )
    )


def _make_invoke(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    selected: torch.Tensor,
    routing: torch.Tensor,
    config: dict[str, Any],
) -> tuple[TensorFn, dict[str, Any]]:
    output = torch.empty_like(hidden)
    workspace: dict[str, Any] = {}

    def invoke() -> torch.Tensor:
        return qwen3_moe_segmented_prefill(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=output,
            workspace=workspace,
            **config,
        )

    return invoke, workspace


def _profile_case(
    name: str,
    config: dict[str, Any],
    *,
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    selected: torch.Tensor,
    routing: torch.Tensor,
    reference: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    invoke: TensorFn | None = None
    workspace: dict[str, Any] = {}
    try:
        invoke, workspace = _make_invoke(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            config,
        )
        first = invoke().detach().clone()
        second = invoke().detach().clone()
        torch.cuda.synchronize()
        delta = (first.float() - reference.float()).abs()
        repeat_delta = (first.float() - second.float()).abs()
        finite = bool(torch.isfinite(first).all().item())
        cosine = _cosine(first, reference)
        max_abs_error = float(delta.max().item())
        exact = bool(torch.equal(first, reference))
        repeat_exact = bool(torch.equal(first, second))
        contract = _contract_matches(workspace, config)
        correct = bool(
            finite
            and exact
            and repeat_exact
            and cosine >= 0.9999
            and max_abs_error <= 0.125
            and contract
        )
        timing = _measure_us(
            invoke,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        return {
            "case": name,
            "config": config,
            "error": None,
            "correct": correct,
            "contract": contract,
            "finite": finite,
            "exact": exact,
            "cosine": cosine,
            "max_abs_error": max_abs_error,
            "mean_abs_error": float(delta.mean().item()),
            "repeat_exact": repeat_exact,
            "repeat_max_abs_error": float(repeat_delta.max().item()),
            **timing,
        }
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return {
            "case": name,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "correct": False,
            "contract": False,
            "median_us": None,
            "samples_us": [],
            "sample_spread_ratio": None,
        }
    finally:
        del invoke, workspace


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--minimum-profile-speedup", type=float, default=1.0)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--maximum-candidate-drift", type=float, default=1.05)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_sorted_partial_a100.json",
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
        raise SystemExit(f"This gate requires an A100, found: {gpu}")
    if vram_gb < 70.0:
        raise SystemExit(f"This gate requires an A100 80GB, found {vram_gb:.2f}GB")

    torch.manual_seed(20260814)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Gemma4 long sorted-partial segmented-prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print(
        "  shape: "
        f"rows={ROWS} H={HIDDEN_DIM} I={INTERMEDIATE_DIM} "
        f"E={NUM_EXPERTS} top_k={TOP_K} dtype=bf16"
    )

    def random_weight(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 0.02)

    hidden = random_weight((ROWS, HIDDEN_DIM))
    gate_up = random_weight((NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM))
    down = random_weight((NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM))

    profiles: list[dict[str, Any]] = []
    candidate_names = [name for name, _ in CANDIDATE_CHANGES]
    for profile_index, (profile_name, heavy_count) in enumerate(SKEW_PROFILES):
        selected = _skewed_route(
            rows=ROWS,
            top_k=TOP_K,
            num_experts=NUM_EXPERTS,
            heavy_count=heavy_count,
            device=device,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(20260814 + profile_index)
        routing = torch.rand(
            (ROWS, TOP_K),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        routing.div_(routing.sum(dim=-1, keepdim=True))
        counts = torch.bincount(selected.reshape(-1), minlength=NUM_EXPERTS)
        max_count = int(counts.max().item())
        global_padding_ratio = (
            float(max_count * NUM_EXPERTS) / float(ROWS * TOP_K)
        )
        route_summary = {
            "profile": profile_name,
            "heavy_count": int(heavy_count),
            "max_count": max_count,
            "min_count": int(counts.min().item()),
            "active_experts": int(counts.gt(0).sum().item()),
            "global_padding_ratio": global_padding_ratio,
        }
        print("ROUTE " + json.dumps(route_summary, sort_keys=True))

        baseline_invoke, _ = _make_invoke(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            _config(),
        )
        reference = baseline_invoke().detach().clone()
        reference_repeat = baseline_invoke().detach().clone()
        torch.cuda.synchronize()
        if not torch.equal(reference, reference_repeat):
            raise SystemExit(f"baseline is not repeat-exact for {profile_name}")
        del baseline_invoke, reference_repeat

        cases: list[dict[str, Any]] = []
        case_specs: list[tuple[str, dict[str, Any]]] = [
            ("current", _config()),
            *[
                (candidate_name, _config(changes))
                for candidate_name, changes in CANDIDATE_CHANGES
            ],
            ("current_recheck", _config()),
            *[
                (f"{candidate_name}_recheck", _config(changes))
                for candidate_name, changes in CANDIDATE_CHANGES
            ],
        ]
        for case_name, case_config in case_specs:
            row = _profile_case(
                case_name,
                case_config,
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                reference=reference,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            row["profile"] = profile_name
            row["global_padding_ratio"] = global_padding_ratio
            cases.append(row)
            print(json.dumps(row, sort_keys=True))
            gc.collect()
            torch.cuda.empty_cache()

        profiles.append({**route_summary, "cases": cases})
        del selected, routing, counts, reference
        gc.collect()
        torch.cuda.empty_cache()

    aggregate_candidates: list[dict[str, Any]] = []
    for candidate_name in candidate_names:
        baseline_total_us = 0.0
        candidate_total_us = 0.0
        profile_speedups: list[float] = []
        baseline_drifts: list[float] = []
        candidate_drifts: list[float] = []
        correct = True
        stable = True
        selected_config: dict[str, Any] | None = None
        for profile in profiles:
            by_name = {row["case"]: row for row in profile["cases"]}
            cold = by_name["current"]
            settled = by_name["current_recheck"]
            candidate = by_name[candidate_name]
            candidate_recheck = by_name[f"{candidate_name}_recheck"]
            if (
                not cold.get("correct")
                or not settled.get("correct")
                or not candidate.get("correct")
                or not candidate_recheck.get("correct")
                or cold.get("median_us") is None
                or settled.get("median_us") is None
                or candidate.get("median_us") is None
                or candidate_recheck.get("median_us") is None
            ):
                correct = False
                stable = False
                continue
            cold_us = float(cold["median_us"])
            settled_us = float(settled["median_us"])
            candidate_first_us = float(candidate["median_us"])
            candidate_recheck_us = float(candidate_recheck["median_us"])
            # The first baseline is intentionally diagnostic. On a fresh A100 it
            # can straddle the clock-ramp transition even after JIT warmup. Use
            # the faster baseline median and slower candidate median so promotion
            # remains conservative without treating cold-start drift as a kernel
            # regression.
            baseline_us = min(cold_us, settled_us)
            candidate_us = max(candidate_first_us, candidate_recheck_us)
            baseline_drift = max(cold_us, settled_us) / min(cold_us, settled_us)
            candidate_drift = (
                max(candidate_first_us, candidate_recheck_us)
                / min(candidate_first_us, candidate_recheck_us)
            )
            profile_speedup = baseline_us / candidate_us
            baseline_drifts.append(baseline_drift)
            candidate_drifts.append(candidate_drift)
            profile_speedups.append(profile_speedup)
            baseline_total_us += baseline_us
            candidate_total_us += candidate_us
            selected_config = dict(candidate_recheck["config"])
            stable = bool(
                stable
                and candidate_drift <= float(args.maximum_candidate_drift)
                and float(settled["sample_spread_ratio"])
                <= float(args.maximum_spread)
                and float(candidate["sample_spread_ratio"])
                <= float(args.maximum_spread)
                and float(candidate_recheck["sample_spread_ratio"])
                <= float(args.maximum_spread)
            )
        aggregate_speedup = (
            baseline_total_us / candidate_total_us
            if correct and candidate_total_us > 0.0
            else None
        )
        aggregate_candidates.append(
            {
                "case": candidate_name,
                "config": selected_config,
                "correct": correct,
                "stable": stable,
                "baseline_total_us": baseline_total_us or None,
                "candidate_total_us": candidate_total_us or None,
                "aggregate_speedup": aggregate_speedup,
                "profile_speedups": profile_speedups,
                "minimum_profile_speedup": (
                    min(profile_speedups) if profile_speedups else None
                ),
                "maximum_baseline_drift": (
                    max(baseline_drifts) if baseline_drifts else None
                ),
                "maximum_candidate_drift": (
                    max(candidate_drifts) if candidate_drifts else None
                ),
            }
        )

    valid_candidates = [
        row
        for row in aggregate_candidates
        if row["correct"]
        and row["stable"]
        and row["aggregate_speedup"] is not None
    ]
    measured_winner = (
        max(valid_candidates, key=lambda row: float(row["aggregate_speedup"]))
        if valid_candidates
        else None
    )
    apply_change = bool(
        measured_winner is not None
        and float(measured_winner["aggregate_speedup"])
        >= float(args.minimum_speedup)
        and float(measured_winner["minimum_profile_speedup"])
        >= float(args.minimum_profile_speedup)
    )
    selected_config = (
        dict(measured_winner["config"])
        if apply_change and measured_winner is not None
        else dict(BASE_CONFIG)
    )
    average_savings_us = (
        (
            float(measured_winner["baseline_total_us"])
            - float(measured_winner["candidate_total_us"])
        )
        / len(SKEW_PROFILES)
        if apply_change and measured_winner is not None
        else 0.0
    )
    estimated_savings_ms = average_savings_us * LAYER_INVOCATIONS / 1000.0
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_CURRENT",
        "apply_change": apply_change,
        "winner": (
            str(measured_winner["case"])
            if apply_change and measured_winner is not None
            else "current"
        ),
        "measured_winner": (
            str(measured_winner["case"])
            if measured_winner is not None
            else None
        ),
        "selected_config": selected_config,
        "minimum_speedup": float(args.minimum_speedup),
        "minimum_profile_speedup_required": float(args.minimum_profile_speedup),
        "maximum_spread": float(args.maximum_spread),
        "maximum_candidate_drift_allowed": float(args.maximum_candidate_drift),
        "estimated_savings_ms_b16_prefill": estimated_savings_ms,
        "target_prefill_gap_ms": 113.693,
        "estimated_gap_coverage": estimated_savings_ms / 113.693,
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "rows": ROWS,
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "dtype": "bf16",
        },
        "profiles": profiles,
        "aggregate_candidates": aggregate_candidates,
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
