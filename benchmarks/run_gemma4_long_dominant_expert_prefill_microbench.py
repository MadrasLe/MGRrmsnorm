#!/usr/bin/env python3
"""Gate a dominant-expert split against Gemma4's long-prefill baseline."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from megagemm.kernels.qwen3_moe import (
    qwen3_moe_dominant_expert_padded_bmm_prefill,
    qwen3_moe_segmented_prefill,
)


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
        "sample_spread_ratio": max(samples) / min(samples),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


def _build_skewed_route(
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    heavy_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Build unique-per-row top-k traffic with an exact expert-0 count."""
    assignments = int(rows) * int(top_k)
    if not 0 < int(heavy_count) <= int(rows):
        raise ValueError("heavy_count must be positive and at most rows")
    if int(top_k) >= int(num_experts):
        raise ValueError("top_k must be smaller than num_experts")

    flat = torch.empty(assignments, device=device, dtype=torch.int64)
    heavy_slots = (
        torch.arange(heavy_count, device=device, dtype=torch.int64) * int(top_k)
    )
    light_mask = torch.ones(assignments, device=device, dtype=torch.bool)
    light_mask[heavy_slots] = False
    flat[heavy_slots] = 0
    light_assignments = assignments - int(heavy_count)
    flat[light_mask] = (
        torch.arange(light_assignments, device=device, dtype=torch.int64)
        .remainder(int(num_experts) - 1)
        .add_(1)
    )
    return flat.reshape(int(rows), int(top_k)).contiguous()


@torch.inference_mode()
def _run_candidate_preflight(device: torch.device) -> dict[str, Any]:
    """Compile and validate the whole candidate before its large allocations."""
    dtype = torch.bfloat16
    rows = 32
    hidden_dim = 64
    intermediate_dim = 32
    num_experts = 16
    top_k = 4
    heavy_count = 24

    hidden = torch.empty(
        (rows, hidden_dim), device=device, dtype=dtype
    ).normal_(0.0, 0.2)
    gate_up = torch.empty(
        (num_experts, 2 * intermediate_dim, hidden_dim),
        device=device,
        dtype=dtype,
    ).normal_(0.0, 0.05)
    down = torch.empty(
        (num_experts, hidden_dim, intermediate_dim),
        device=device,
        dtype=dtype,
    ).normal_(0.0, 0.05)
    selected = _build_skewed_route(
        rows=rows,
        top_k=top_k,
        num_experts=num_experts,
        heavy_count=heavy_count,
        device=device,
    )
    routing = torch.rand((rows, top_k), device=device, dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))
    output = torch.empty_like(hidden)
    workspace: dict[str, Any] = {}

    def invoke() -> torch.Tensor:
        return qwen3_moe_dominant_expert_padded_bmm_prefill(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=output,
            workspace=workspace,
            align_m=8,
            route_pack_block=128,
            activation_block=128,
            reduce_block_n=64,
            reduce_num_warps=4,
            minimum_dominant_skew=2.0,
            max_light_padding_ratio=1.25,
        )

    first = invoke().detach().clone()
    second = invoke().detach().clone()
    torch.cuda.synchronize()

    reference = torch.zeros(
        (rows, hidden_dim), device=device, dtype=torch.float32
    )
    for expert in range(num_experts):
        positions = torch.nonzero(selected == expert, as_tuple=False)
        if int(positions.shape[0]) == 0:
            continue
        token_ids = positions[:, 0]
        top_ids = positions[:, 1]
        expert_hidden = hidden.index_select(0, token_ids)
        expert_gate_up = torch.mm(
            expert_hidden,
            gate_up[expert].transpose(0, 1),
        )
        gate, up = expert_gate_up.chunk(2, dim=-1)
        activated = (
            F.gelu(gate.float(), approximate="tanh") * up.float()
        ).to(dtype)
        projected = torch.bmm(
            activated.unsqueeze(0),
            down[expert : expert + 1].transpose(1, 2),
            out_dtype=torch.float32,
        ).squeeze(0)
        weighted = projected * routing[token_ids, top_ids].float().unsqueeze(1)
        reference.index_copy_(
            0,
            token_ids,
            reference.index_select(0, token_ids) + weighted,
        )
    reference = reference.to(dtype)

    finite = bool(torch.isfinite(first).all().item())
    repeat_exact = bool(torch.equal(first, second))
    cosine = _cosine(first, reference)
    max_abs_error = float((first.float() - reference.float()).abs().max().item())
    contract = bool(
        int(workspace.get("dominant_padded_bmm_prefill", 0))
        and int(workspace.get("dominant_padded_bmm_deterministic_reduce", 0))
        and int(workspace.get("dominant_padded_bmm_heavy_expert", -1)) == 0
        and int(workspace.get("dominant_padded_bmm_heavy_count", -1))
        == heavy_count
        and workspace.get("dominant_padded_bmm_route_pack") == "atomic_split"
    )
    result = {
        "status": "PASS",
        "shape": {
            "rows": rows,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "finite": finite,
        "repeat_exact": repeat_exact,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "contract": contract,
    }
    if not (
        finite
        and repeat_exact
        and cosine >= 0.9999
        and max_abs_error <= 0.125
        and contract
    ):
        result["status"] = "FAIL"
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-profile-speedup", type=float, default=1.07)
    parser.add_argument("--minimum-aggregate-speedup", type=float, default=1.07)
    parser.add_argument("--maximum-candidate-drift", type=float, default=1.05)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_dominant_expert_prefill_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 5:
        raise SystemExit("warmup/iterations must be positive and repeats >= 5")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "a100" not in gpu.lower().replace("-", " ").split() or vram_gb < 70.0:
        raise SystemExit(
            f"This gate requires an A100 80GB, found {gpu} ({vram_gb:.2f}GB)"
        )

    torch.manual_seed(20260816)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = 32_768
    hidden_dim = 2_816
    intermediate_dim = 704
    num_experts = 128
    top_k = 8
    assignments = rows * top_k
    layers = 30
    target_prefill_gap_ms = 46.04
    profiles = (
        ("skew_7p64x", 15_648),
        ("skew_11x", 22_528),
        ("skew_15p28x", 31_296),
    )

    print("Gemma4 long dominant-expert prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"rows={rows} H={hidden_dim} I={intermediate_dim} "
        f"E={num_experts} top_k={top_k} assignments={assignments} dtype=bf16"
    )
    print("  candidate: dense dominant expert + padded BMM for light experts")

    rng_state = torch.cuda.get_rng_state(device)
    try:
        preflight = _run_candidate_preflight(device)
    except Exception as exc:
        print(
            "DOMINANT_HYBRID_PREFLIGHT "
            + json.dumps(
                {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 2
    finally:
        torch.cuda.set_rng_state(rng_state, device)
    print("DOMINANT_HYBRID_PREFLIGHT " + json.dumps(preflight, sort_keys=True))
    del preflight
    gc.collect()
    torch.cuda.empty_cache()

    def random_weight(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 0.02)

    hidden = random_weight((rows, hidden_dim))
    gate_up = random_weight((num_experts, 2 * intermediate_dim, hidden_dim))
    down = random_weight((num_experts, hidden_dim, intermediate_dim))
    baseline_config: dict[str, Any] = {
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
        "async_tiles_max_assignments": 4_096,
        "deterministic_reduce": True,
    }
    candidate_config: dict[str, Any] = {
        "align_m": 16,
        "route_pack_block": 256,
        "activation_block": 512,
        "reduce_block_n": 256,
        "reduce_num_warps": 4,
        "minimum_dominant_skew": 4.0,
        "max_light_padding_ratio": 1.25,
    }

    all_rows: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []

    for profile_name, heavy_count in profiles:
        selected = _build_skewed_route(
            rows=rows,
            top_k=top_k,
            num_experts=num_experts,
            heavy_count=heavy_count,
            device=device,
        )
        routing = torch.rand((rows, top_k), device=device, dtype=dtype)
        routing.div_(routing.sum(dim=-1, keepdim=True))
        counts = torch.bincount(selected.reshape(-1), minlength=num_experts)
        max_count = int(counts.max().item())
        min_count = int(counts.min().item())
        light_counts = counts.clone()
        light_counts[0] = 0
        max_light_count = int(light_counts.max().item())
        light_padded_count = ((max_light_count + 15) // 16) * 16
        light_assignments = assignments - heavy_count
        light_padding_ratio = (
            num_experts * light_padded_count / light_assignments
        )
        route_row = {
            "profile": profile_name,
            "active_experts": int((counts > 0).sum().item()),
            "heavy_count": heavy_count,
            "max_count": max_count,
            "min_count": min_count,
            "global_padding_ratio": max_count * num_experts / assignments,
            "light_max_count": max_light_count,
            "light_padding_ratio": light_padding_ratio,
            "hybrid_capacity_ratio": (
                num_experts * light_padded_count + heavy_count
            )
            / assignments,
        }
        print("ROUTE " + json.dumps(route_row, sort_keys=True))

        def make_baseline() -> tuple[TensorFn, dict[str, Any]]:
            workspace: dict[str, Any] = {}
            output = torch.empty_like(hidden)

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
                    **baseline_config,
                )

            return invoke, workspace

        def make_candidate() -> tuple[TensorFn, dict[str, Any]]:
            workspace: dict[str, Any] = {}
            output = torch.empty_like(hidden)

            def invoke() -> torch.Tensor:
                return qwen3_moe_dominant_expert_padded_bmm_prefill(
                    hidden,
                    gate_up,
                    down,
                    selected,
                    routing,
                    activation="gelu_pytorch_tanh",
                    out=output,
                    workspace=workspace,
                    **candidate_config,
                )

            return invoke, workspace

        reference_fn, _ = make_baseline()
        reference = reference_fn().detach().clone()
        reference_repeat = reference_fn().detach().clone()
        torch.cuda.synchronize()
        if not torch.equal(reference, reference_repeat):
            raise SystemExit(f"{profile_name}: production baseline is not repeat-exact")
        del reference_fn, reference_repeat

        profile_rows: list[dict[str, Any]] = []

        def profile_case(name: str, path: str) -> None:
            invoke: TensorFn | None = None
            workspace: dict[str, Any] = {}
            try:
                if path == "segmented":
                    invoke, workspace = make_baseline()
                else:
                    invoke, workspace = make_candidate()
                first = invoke().detach().clone()
                second = invoke().detach().clone()
                torch.cuda.synchronize()
                delta = (first.float() - reference.float()).abs()
                repeat_delta = (first.float() - second.float()).abs()
                finite = bool(torch.isfinite(first).all().item())
                cosine = _cosine(first, reference)
                max_abs_error = float(delta.max().item())
                mean_abs_error = float(delta.mean().item())
                repeat_exact = bool(torch.equal(first, second))
                repeat_max_abs_error = float(repeat_delta.max().item())
                if path == "segmented":
                    contract = bool(
                        int(workspace.get("segmented_prefill_deterministic_reduce", 0))
                        and int(workspace.get("segmented_prefill_partial_reduce", 0))
                        and workspace.get("segmented_prefill_partial_dtype")
                        == "torch.float32"
                        and int(workspace.get("segmented_prefill_route_scatter", 0))
                    )
                    correct = bool(
                        finite
                        and repeat_exact
                        and torch.equal(first, reference)
                        and contract
                    )
                else:
                    contract = bool(
                        int(workspace.get("dominant_padded_bmm_prefill", 0))
                        and int(
                            workspace.get(
                                "dominant_padded_bmm_deterministic_reduce", 0
                            )
                        )
                        and int(
                            workspace.get("dominant_padded_bmm_heavy_expert", -1)
                        )
                        == 0
                        and int(
                            workspace.get("dominant_padded_bmm_heavy_count", -1)
                        )
                        == heavy_count
                        and workspace.get(
                            "dominant_padded_bmm_down_output_dtype"
                        )
                        == "torch.float32"
                        and workspace.get("dominant_padded_bmm_route_pack")
                        == "atomic_split"
                        and float(
                            workspace.get(
                                "dominant_padded_bmm_light_padding_ratio", 99.0
                            )
                        )
                        <= 1.25
                    )
                    correct = bool(
                        finite
                        and repeat_exact
                        and cosine >= 0.9999
                        and max_abs_error <= 0.125
                        and contract
                    )
                timing = _measure_us(
                    invoke,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                row = {
                    "profile": profile_name,
                    "case": name,
                    "path": path,
                    "error": None,
                    "correct": correct,
                    "contract": contract,
                    "finite": finite,
                    "cosine": cosine,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "repeat_exact": repeat_exact,
                    "repeat_max_abs_error": repeat_max_abs_error,
                    **timing,
                }
                if path == "dominant_hybrid":
                    row["heavy_expert"] = int(
                        workspace["dominant_padded_bmm_heavy_expert"]
                    )
                    row["heavy_count"] = int(
                        workspace["dominant_padded_bmm_heavy_count"]
                    )
                    row["light_padding_ratio"] = float(
                        workspace["dominant_padded_bmm_light_padding_ratio"]
                    )
                    row["hybrid_capacity_ratio"] = float(
                        workspace["dominant_padded_bmm_capacity_ratio"]
                    )
                del first, second, delta, repeat_delta
            except Exception as exc:
                row = {
                    "profile": profile_name,
                    "case": name,
                    "path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                    "contract": False,
                    "median_us": None,
                    "samples_us": [],
                    "sample_spread_ratio": None,
                }
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            profile_rows.append(row)
            all_rows.append(row)
            print(json.dumps(row, sort_keys=True))
            del invoke, workspace
            gc.collect()

        profile_case("current", "segmented")
        profile_case("dominant_hybrid", "dominant_hybrid")
        profile_case("dominant_hybrid_recheck", "dominant_hybrid")
        profile_case("current_recheck", "segmented")

        by_name = {row["case"]: row for row in profile_rows}
        baselines = [by_name["current"], by_name["current_recheck"]]
        candidates = [
            by_name["dominant_hybrid"],
            by_name["dominant_hybrid_recheck"],
        ]
        valid_baseline = all(
            row.get("correct") and row.get("median_us") is not None
            for row in baselines
        )
        valid_candidate = all(
            row.get("correct") and row.get("median_us") is not None
            for row in candidates
        )
        baseline_us = (
            min(float(row["median_us"]) for row in baselines)
            if valid_baseline
            else None
        )
        candidate_us = (
            max(float(row["median_us"]) for row in candidates)
            if valid_candidate
            else None
        )
        baseline_drift = (
            max(float(row["median_us"]) for row in baselines)
            / min(float(row["median_us"]) for row in baselines)
            if valid_baseline
            else None
        )
        candidate_drift = (
            max(float(row["median_us"]) for row in candidates)
            / min(float(row["median_us"]) for row in candidates)
            if valid_candidate
            else None
        )
        speedup = (
            baseline_us / candidate_us
            if baseline_us is not None and candidate_us is not None
            else None
        )
        savings_ms = (
            (baseline_us - candidate_us) * layers / 1000.0
            if baseline_us is not None and candidate_us is not None
            else None
        )
        profile_summary = {
            **route_row,
            "baseline_us_conservative": baseline_us,
            "candidate_us_conservative": candidate_us,
            "baseline_drift": baseline_drift,
            "candidate_drift": candidate_drift,
            "correct": bool(valid_baseline and valid_candidate),
            "candidate_stable": bool(
                candidate_drift is not None
                and candidate_drift <= float(args.maximum_candidate_drift)
            ),
            "speedup": speedup,
            "estimated_savings_ms_per_prefill_30_layers": savings_ms,
        }
        profile_summaries.append(profile_summary)
        print("PROFILE_DECISION " + json.dumps(profile_summary, sort_keys=True))
        del reference, selected, routing, counts, light_counts
        gc.collect()
        torch.cuda.empty_cache()

    valid_profiles = all(
        row["correct"]
        and row["candidate_stable"]
        and row["speedup"] is not None
        and row["estimated_savings_ms_per_prefill_30_layers"] is not None
        for row in profile_summaries
    )
    aggregate_baseline_us = sum(
        float(row["baseline_us_conservative"] or 0.0)
        for row in profile_summaries
    )
    aggregate_candidate_us = sum(
        float(row["candidate_us_conservative"] or 0.0)
        for row in profile_summaries
    )
    aggregate_speedup = (
        aggregate_baseline_us / aggregate_candidate_us
        if valid_profiles and aggregate_candidate_us > 0.0
        else None
    )
    minimum_profile_speedup = (
        min(float(row["speedup"]) for row in profile_summaries)
        if valid_profiles
        else None
    )
    minimum_savings_ms = (
        min(
            float(row["estimated_savings_ms_per_prefill_30_layers"])
            for row in profile_summaries
        )
        if valid_profiles
        else None
    )
    closes_gap = bool(
        minimum_savings_ms is not None
        and minimum_savings_ms >= target_prefill_gap_ms
    )
    advance_candidate = bool(
        valid_profiles
        and aggregate_speedup is not None
        and aggregate_speedup >= float(args.minimum_aggregate_speedup)
        and minimum_profile_speedup is not None
        and minimum_profile_speedup >= float(args.minimum_profile_speedup)
        and closes_gap
    )
    summary = {
        "decision": "APPLY_HYBRID" if advance_candidate else "KEEP_SEGMENTED",
        "advance_candidate": advance_candidate,
        "candidate": "dense_dominant_plus_light_padded_bmm",
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "rows": rows,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "assignments": assignments,
            "dtype": "bf16",
        },
        "profiles": profile_summaries,
        "cases": all_rows,
        "correct": valid_profiles,
        "aggregate_speedup": aggregate_speedup,
        "minimum_profile_speedup": minimum_profile_speedup,
        "minimum_required_profile_speedup": float(args.minimum_profile_speedup),
        "minimum_required_aggregate_speedup": float(
            args.minimum_aggregate_speedup
        ),
        "maximum_candidate_drift": float(args.maximum_candidate_drift),
        "minimum_estimated_savings_ms_per_prefill_30_layers": minimum_savings_ms,
        "target_prefill_gap_ms": target_prefill_gap_ms,
        "closes_gap": closes_gap,
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
