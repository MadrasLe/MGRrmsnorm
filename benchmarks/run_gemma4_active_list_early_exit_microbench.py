#!/usr/bin/env python3
"""Gate Gemma4 B16 active-expert early-exit without downloading the model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

import megagemm.kernels.qwen3_moe as moe_kernel
from megagemm.kernels.qwen3_moe import qwen3_moe_grouped_decode


ROWS = 16
HIDDEN = 2816
INTERMEDIATE = 704
NUM_EXPERTS = 128
TOP_K = 8
ASSIGNMENTS = ROWS * TOP_K


def build_v82_route() -> torch.Tensor:
    """Build 90 active experts with the 60/22/8 singleton/double/triple mix."""
    counts = [1] * 60 + [2] * 22 + [3] * 8 + [0] * 38
    rows: list[list[int]] = [[] for _ in range(ROWS)]
    tie_cursor = 0
    for expert, count in sorted(
        enumerate(counts),
        key=lambda item: (-item[1], item[0]),
    ):
        for _ in range(count):
            candidates = [
                row
                for row in range(ROWS)
                if len(rows[row]) < TOP_K and expert not in rows[row]
            ]
            if not candidates:
                raise RuntimeError("could not construct the fixed B16 route")
            minimum = min(len(rows[row]) for row in candidates)
            shortest = [row for row in candidates if len(rows[row]) == minimum]
            row = shortest[tie_cursor % len(shortest)]
            tie_cursor += 1
            rows[row].append(expert)

    if any(len(row) != TOP_K or len(set(row)) != TOP_K for row in rows):
        raise RuntimeError(f"invalid fixed route: {rows}")
    flat = [expert for row in rows for expert in row]
    actual = [flat.count(expert) for expert in range(NUM_EXPERTS)]
    if actual != counts:
        raise RuntimeError("fixed route count contract changed")
    return torch.tensor(rows, dtype=torch.int64)


def build_active_expert_route(active_experts: int) -> torch.Tensor:
    """Build a rectangular B16 route with an exact active-expert frontier."""
    active_experts = int(active_experts)
    if active_experts == 90:
        return build_v82_route()
    if (
        active_experts < TOP_K
        or active_experts > ASSIGNMENTS
        or ASSIGNMENTS % active_experts != 0
    ):
        raise ValueError(
            "active expert profiles must divide 128 and be in [8, 128], "
            "except for the preserved 90-expert v82 route"
        )

    flat = [index % active_experts for index in range(ASSIGNMENTS)]
    rows = [flat[start : start + TOP_K] for start in range(0, ASSIGNMENTS, TOP_K)]
    if any(len(set(row)) != TOP_K for row in rows):
        raise RuntimeError(f"route profile E={active_experts} repeats an expert in a row")
    counts = [flat.count(expert) for expert in range(NUM_EXPERTS)]
    if sum(count > 0 for count in counts) != active_experts:
        raise RuntimeError(f"route profile E={active_experts} has the wrong frontier")
    return torch.tensor(rows, dtype=torch.int64)


def parse_active_expert_profiles(raw: str) -> list[int]:
    profiles: list[int] = []
    for item in str(raw).split(","):
        value = int(item.strip())
        build_active_expert_route(value)
        if value not in profiles:
            profiles.append(value)
    if not profiles:
        raise ValueError("at least one active-expert profile is required")
    return profiles


def configure_compact(*, active_list: bool, early_exit: bool) -> None:
    moe_kernel._CFG_SHARED_ROUTE_DECODE = False
    moe_kernel._CFG_ROUTE_MATRIX_DECODE = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = bool(active_list)
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = bool(
        early_exit
    )
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = 64
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = 128
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS = 4
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES = 3
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES = 3
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES = 3
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM = 1
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID = False
    moe_kernel._CFG_BLOCK_K = 128
    moe_kernel._CFG_EXPERT_GROUPED_BLOCK_M = 16
    moe_kernel._CFG_EXPERT_GROUPED_MIN_ROWS = 1
    moe_kernel._CFG_EXPERT_GROUPED_MAX_ROWS = 16


@torch.inference_mode()
def run_case(
    *,
    name: str,
    active_list: bool,
    early_exit: bool,
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    selected: torch.Tensor,
    routing: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    configure_compact(active_list=active_list, early_exit=early_exit)
    workspace: dict[str, Any] = {}
    out = torch.empty_like(hidden)
    if torch.is_grad_enabled():
        raise RuntimeError("active-list gate must run with gradients disabled")
    if not moe_kernel.qwen3_moe_grouped_prefers_triton_shape(
        ROWS,
        TOP_K,
        HIDDEN,
        INTERMEDIATE,
        max_assignments=ASSIGNMENTS,
    ):
        raise RuntimeError(
            "active-list gate shape is not eligible for grouped Triton decode"
        )

    def call() -> torch.Tensor:
        return qwen3_moe_grouped_decode(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=out,
            workspace=workspace,
            max_assignments=128,
            expert_grouped_compact=True,
            expert_grouped_min_rows=9,
            expert_grouped_max_rows=16,
            expert_grouped_compact_partial_reduce=True,
        )

    for _ in range(max(int(warmup), 1)):
        call()
    torch.cuda.synchronize()
    eager_path = str(workspace.get("grouped_decode_last_path") or "")
    if eager_path != "expert_grouped_compact":
        raise RuntimeError(
            "active-list gate dispatch failed before CUDA graph capture: "
            f"expected 'expert_grouped_compact', got {eager_path!r}"
        )
    expected_active_list = int(bool(active_list))
    runtime_active_list = int(
        workspace.get("expert_grouped_compact_decode_last_active_list", 0) or 0
    )
    runtime_early_exit = int(
        workspace.get(
            "expert_grouped_compact_decode_last_active_list_early_exit",
            0,
        )
        or 0
    )
    if runtime_active_list != expected_active_list:
        raise RuntimeError(
            "active-list gate telemetry mismatch before capture: "
            f"expected active_list={expected_active_list}, "
            f"got {runtime_active_list}"
        )
    if runtime_early_exit != int(bool(early_exit)):
        raise RuntimeError(
            "active-list gate telemetry mismatch before capture: "
            f"expected early_exit={int(bool(early_exit))}, "
            f"got {runtime_early_exit}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph_warmup_replays = max(3, int(iterations) * 4)
    for _ in range(graph_warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    second = out.clone()
    repeat_max_abs_error = float(
        (second.float() - first.float()).abs().max().item()
    )

    samples_us: list[float] = []
    for _ in range(int(repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(iterations)):
            graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(
            float(start.elapsed_time(end)) * 1000.0 / int(iterations)
        )

    row = {
        "case": name,
        "active_list": bool(active_list),
        "active_list_early_exit": bool(early_exit),
        "measurement": "cuda_graph",
        "median_us": float(statistics.median(samples_us)),
        "samples_us": samples_us,
        "repeat_max_abs_error": repeat_max_abs_error,
        "graph_warmup_replays": graph_warmup_replays,
        "path": str(workspace.get("grouped_decode_last_path") or ""),
        "runtime_active_list": int(
            workspace.get(
                "expert_grouped_compact_decode_last_active_list",
                0,
            )
            or 0
        ),
        "runtime_active_list_early_exit": int(
            workspace.get(
                "expert_grouped_compact_decode_last_active_list_early_exit",
                0,
            )
            or 0
        ),
    }
    del graph
    return row, first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        default=1.02,
    )
    parser.add_argument(
        "--minimum-low-active-speedup",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--maximum-regression-ratio",
        type=float,
        default=1.02,
    )
    parser.add_argument(
        "--active-expert-profiles",
        default="8,16,32,64,90,128",
    )
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_active_list_early_exit_gate.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if int(args.iterations) <= 0 or int(args.repeats) < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")
    active_expert_profiles = parse_active_expert_profiles(
        args.active_expert_profiles
    )
    low_active_profiles = [value for value in active_expert_profiles if value <= 32]
    if not low_active_profiles:
        raise SystemExit("the frontier must contain at least one profile <= 32 experts")

    device_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16
    print("Gemma4 B16 active-list route-frontier pre-download gate")
    print("  harness_rev: gemma4-active-list-frontier-v3-route-sensitive")
    print(f"  gpu: {device_name}")
    print(
        "  shape: "
        f"rows={ROWS} H={HIDDEN} I={INTERMEDIATE} "
        f"E={NUM_EXPERTS} top_k={TOP_K} assignments={ASSIGNMENTS}"
    )
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print(f"  active_expert_profiles: {active_expert_profiles}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260727)
    hidden = torch.empty(
        (ROWS, HIDDEN),
        device="cuda",
        dtype=dtype,
    ).uniform_(-0.02, 0.02, generator=generator)
    gate_up = torch.empty(
        (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        device="cuda",
        dtype=dtype,
    ).uniform_(-0.02, 0.02, generator=generator)
    down = torch.empty(
        (NUM_EXPERTS, HIDDEN, INTERMEDIATE),
        device="cuda",
        dtype=dtype,
    ).uniform_(-0.02, 0.02, generator=generator)
    routing_fp32 = torch.rand(
        (ROWS, TOP_K),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    routing = (
        routing_fp32 / routing_fp32.sum(dim=-1, keepdim=True)
    ).to(dtype)

    mutable_names = (
        "_CFG_SHARED_ROUTE_DECODE",
        "_CFG_ROUTE_MATRIX_DECODE",
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE",
        "_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK",
        "_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE",
        "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST",
        "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT",
        "_CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK",
        "_CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS",
        "_CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM",
        "_CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT",
        "_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N",
        "_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N",
        "_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS",
        "_CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES",
        "_CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES",
        "_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES",
        "_CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM",
        "_CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT",
        "_CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP",
        "_CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT",
        "_CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID",
        "_CFG_BLOCK_K",
        "_CFG_EXPERT_GROUPED_BLOCK_M",
        "_CFG_EXPERT_GROUPED_MIN_ROWS",
        "_CFG_EXPERT_GROUPED_MAX_ROWS",
    )
    original = {name: getattr(moe_kernel, name) for name in mutable_names}
    rows: list[dict[str, Any]] = []
    profile_results: list[dict[str, Any]] = []
    try:
        for active_experts in active_expert_profiles:
            selected = build_active_expert_route(active_experts).to(device="cuda")
            route_counts = torch.bincount(
                selected.reshape(-1),
                minlength=NUM_EXPERTS,
            )
            singleton_experts = int((route_counts == 1).sum().item())
            empty_experts = int((route_counts == 0).sum().item())

            current, reference = run_case(
                name="current",
                active_list=False,
                early_exit=False,
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            candidate, candidate_out = run_case(
                name="active_list_early_exit",
                active_list=True,
                early_exit=True,
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            recheck, recheck_out = run_case(
                name="current_recheck",
                active_list=False,
                early_exit=False,
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )

            current["max_abs_error"] = 0.0
            current["exact"] = True
            candidate["max_abs_error"] = float(
                (candidate_out.float() - reference.float()).abs().max().item()
            )
            candidate["exact"] = bool(torch.equal(candidate_out, reference))
            recheck["max_abs_error"] = float(
                (recheck_out.float() - reference.float()).abs().max().item()
            )
            recheck["exact"] = bool(torch.equal(recheck_out, reference))
            for row in (current, candidate, recheck):
                row["active_experts"] = int(active_experts)
                row["singleton_experts"] = singleton_experts
                row["empty_experts"] = empty_experts
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

            baseline_us = min(
                float(current["median_us"]),
                float(recheck["median_us"]),
            )
            baseline_stability = (
                max(float(current["median_us"]), float(recheck["median_us"]))
                / baseline_us
            )
            candidate_us = float(candidate["median_us"])
            correct = bool(
                candidate["exact"]
                and recheck["exact"]
                and candidate["repeat_max_abs_error"] == 0.0
                and recheck["repeat_max_abs_error"] == 0.0
                and candidate["path"] == "expert_grouped_compact"
                and candidate["runtime_active_list"] == 1
                and candidate["runtime_active_list_early_exit"] == 1
            )
            profile_results.append(
                {
                    "active_experts": int(active_experts),
                    "singleton_experts": singleton_experts,
                    "empty_experts": empty_experts,
                    "baseline_us": baseline_us,
                    "candidate_us": candidate_us,
                    "baseline_stability_ratio": baseline_stability,
                    "speedup": baseline_us / candidate_us,
                    "regression_ratio": candidate_us / baseline_us,
                    "correct": correct,
                }
            )
    finally:
        for name, value in original.items():
            setattr(moe_kernel, name, value)

    speedups = [float(profile["speedup"]) for profile in profile_results]
    geomean_speedup = math.exp(
        sum(math.log(value) for value in speedups) / len(speedups)
    )
    low_speedups = [
        float(profile["speedup"])
        for profile in profile_results
        if int(profile["active_experts"]) <= 32
    ]
    low_active_geomean_speedup = math.exp(
        sum(math.log(value) for value in low_speedups) / len(low_speedups)
    )
    maximum_regression_ratio = max(
        float(profile["regression_ratio"]) for profile in profile_results
    )
    maximum_baseline_stability = max(
        float(profile["baseline_stability_ratio"])
        for profile in profile_results
    )
    correct = all(bool(profile["correct"]) for profile in profile_results)
    apply = bool(
        correct
        and maximum_baseline_stability <= 1.03
        and geomean_speedup >= float(args.minimum_speedup)
        and low_active_geomean_speedup
        >= float(args.minimum_low_active_speedup)
        and maximum_regression_ratio <= float(args.maximum_regression_ratio)
    )
    decision = {
        "decision": "APPLY" if apply else "KEEP_CURRENT",
        "apply_change": apply,
        "minimum_speedup": float(args.minimum_speedup),
        "minimum_low_active_speedup": float(args.minimum_low_active_speedup),
        "maximum_regression_ratio_allowed": float(
            args.maximum_regression_ratio
        ),
        "maximum_baseline_stability_ratio": 1.03,
        "geomean_speedup": geomean_speedup,
        "low_active_geomean_speedup": low_active_geomean_speedup,
        "maximum_regression_ratio": maximum_regression_ratio,
        "maximum_observed_baseline_stability_ratio": maximum_baseline_stability,
        "correct": correct,
        "active_expert_profiles": active_expert_profiles,
        "profile_results": profile_results,
        "cases": rows,
    }
    print("DECISION " + json.dumps(decision, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
