#!/usr/bin/env python3
"""Tune the promoted routed-expert kernel at one Gemma4 B8xC2048 chunk."""

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
    qwen3_moe_padded_bmm_prefill,
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
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--near-tie-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_routed_expert_prefill_a100.json",
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

    torch.manual_seed(20260810)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = 16_384
    hidden_dim = 2_816
    intermediate_dim = 704
    num_experts = 128
    top_k = 8
    layers = 30
    chunks = 2
    layer_invocations = layers * chunks
    target_gap_ms = 433.605810

    print("Gemma4 long routed-expert prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"rows={rows} H={hidden_dim} I={intermediate_dim} "
        f"E={num_experts} top_k={top_k} dtype=bf16"
    )
    print(f"  B16 estimate: {chunks} chunks x {layers} layers")

    def random_weight(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 0.02)

    hidden = random_weight((rows, hidden_dim))
    gate_up = random_weight((num_experts, 2 * intermediate_dim, hidden_dim))
    down = random_weight((num_experts, hidden_dim, intermediate_dim))

    # Balanced deterministic traffic: 1,024 assignments per expert. This is the
    # exact route used by the preceding long parallel-MoE component accounting.
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).reshape(-1, 1)
    top_ids = torch.arange(top_k, device=device, dtype=torch.int64).reshape(1, -1)
    selected = (row_ids * 17 + top_ids * 13).remainder(num_experts).contiguous()
    routing = torch.rand((rows, top_k), device=device, dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))
    assignments = rows * top_k

    base_config: dict[str, Any] = {
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

    def make_segmented(
        changes: dict[str, int],
    ) -> tuple[TensorFn, dict[str, Any], dict[str, Any]]:
        config = dict(base_config)
        config.update(changes)
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
                **config,
            )

        return invoke, workspace, config

    def make_padded_bmm(
        route_pack: str,
        route_pack_block: int,
        *,
        fused_activation: bool,
        activation_block: int,
        reduce_block_n: int,
        reduce_num_warps: int,
    ) -> tuple[TensorFn, dict[str, Any], dict[str, Any]]:
        workspace: dict[str, Any] = {}
        output = torch.empty_like(hidden)
        config = {
            "down_output_dtype": "fp32",
            "align_m": 16,
            "route_pack": str(route_pack),
            "route_pack_block": int(route_pack_block),
            "fused_activation": bool(fused_activation),
            "activation_block": int(activation_block),
            "reduce_block_n": int(reduce_block_n),
            "reduce_num_warps": int(reduce_num_warps),
        }

        def invoke() -> torch.Tensor:
            return qwen3_moe_padded_bmm_prefill(
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

        return invoke, workspace, config

    reference_fn, _, _ = make_segmented({})
    reference = reference_fn().detach().clone()
    reference_repeat = reference_fn().detach().clone()
    torch.cuda.synchronize()
    if not torch.equal(reference, reference_repeat):
        raise SystemExit("Production baseline is not repeat-exact")
    del reference_fn, reference_repeat

    measured: list[dict[str, Any]] = []

    def profile_case(
        name: str,
        factory: Callable[[], tuple[TensorFn, dict[str, Any], dict[str, Any]]],
        *,
        path: str,
    ) -> None:
        invoke: TensorFn | None = None
        workspace: dict[str, Any] = {}
        try:
            invoke, workspace, config = factory()
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
            elif path == "padded_bmm":
                contract = bool(
                    int(workspace.get("padded_bmm_prefill", 0))
                    and int(workspace.get("padded_bmm_deterministic_reduce", 0))
                    and int(workspace.get("padded_bmm_padded_assignments", 0))
                    >= assignments
                    and workspace.get("padded_bmm_down_output_dtype")
                    == "torch.float32"
                    and workspace.get("padded_bmm_route_pack")
                    == config.get("route_pack")
                    and int(workspace.get("padded_bmm_route_pack_block", 0))
                    == int(config.get("route_pack_block", 0))
                    and bool(workspace.get("padded_bmm_fused_activation", 0))
                    == bool(config.get("fused_activation", False))
                    and int(workspace.get("padded_bmm_activation_block", 0))
                    == int(config.get("activation_block", 0))
                    and int(workspace.get("padded_bmm_reduce_block_n", 0))
                    == int(config.get("reduce_block_n", 0))
                    and int(workspace.get("padded_bmm_reduce_num_warps", 0))
                    == int(config.get("reduce_num_warps", 0))
                )
            else:
                contract = False

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
                "case": name,
                "path": path,
                "config": config,
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
            row["sample_spread_ratio"] = (
                max(timing["samples_us"]) / min(timing["samples_us"])
            )
            del first, second, delta, repeat_delta
        except Exception as exc:
            row = {
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
        measured.append(row)
        print(json.dumps(row, sort_keys=True))
        del invoke, workspace
        gc.collect()

    profile_case(
        "current",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=256,
            reduce_num_warps=4,
        ),
        path="padded_bmm",
    )
    profile_case(
        "reduce_bn64_w4",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=64,
            reduce_num_warps=4,
        ),
        path="padded_bmm",
    )
    profile_case(
        "reduce_bn64_w8",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=64,
            reduce_num_warps=8,
        ),
        path="padded_bmm",
    )
    profile_case(
        "reduce_bn128_w4",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=128,
            reduce_num_warps=4,
        ),
        path="padded_bmm",
    )
    profile_case(
        "reduce_bn128_w8",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=128,
            reduce_num_warps=8,
        ),
        path="padded_bmm",
    )
    profile_case(
        "reduce_bn256_w8",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=256,
            reduce_num_warps=8,
        ),
        path="padded_bmm",
    )
    profile_case(
        "current_recheck",
        lambda: make_padded_bmm(
            "atomic",
            256,
            fused_activation=True,
            activation_block=512,
            reduce_block_n=256,
            reduce_num_warps=4,
        ),
        path="padded_bmm",
    )

    by_name = {row["case"]: row for row in measured}
    cold_baseline = by_name["current"]
    stable_baseline = by_name["current_recheck"]
    baseline_us = (
        float(stable_baseline["median_us"])
        if stable_baseline.get("correct")
        and stable_baseline.get("median_us") is not None
        else None
    )
    cold_baseline_us = (
        float(cold_baseline["median_us"])
        if cold_baseline.get("correct") and cold_baseline.get("median_us") is not None
        else None
    )
    cold_baseline_ratio = (
        max(cold_baseline_us, baseline_us) / min(cold_baseline_us, baseline_us)
        if cold_baseline_us is not None
        and baseline_us is not None
        and min(cold_baseline_us, baseline_us) > 0.0
        else None
    )
    baseline_sample_spread_ratio = stable_baseline.get("sample_spread_ratio")

    candidates = [
        row
        for row in measured
        if row["case"] not in ("current", "current_recheck")
        and row.get("correct")
        and row.get("median_us") is not None
    ]
    raw_measured_winner = (
        min(candidates, key=lambda row: float(row["median_us"]))
        if candidates
        else None
    )
    near_tied_candidates = (
        [
            row
            for row in candidates
            if float(row["median_us"])
            <= float(raw_measured_winner["median_us"])
            * float(args.near_tie_ratio)
        ]
        if raw_measured_winner is not None
        else []
    )
    # Kernel medians inside the near-tie window are operationally tied. Prefer
    # the tighter spread, then the lower median.
    measured_winner = (
        min(
            near_tied_candidates,
            key=lambda row: (
                float(row["sample_spread_ratio"]),
                float(row["median_us"]),
            ),
        )
        if near_tied_candidates
        else None
    )
    candidate_sample_spread_ratio = (
        float(measured_winner["sample_spread_ratio"])
        if measured_winner is not None
        and measured_winner.get("sample_spread_ratio") is not None
        else None
    )
    stable = bool(
        baseline_sample_spread_ratio is not None
        and float(baseline_sample_spread_ratio)
        <= float(args.maximum_baseline_spread)
        and candidate_sample_spread_ratio is not None
        and candidate_sample_spread_ratio <= float(args.maximum_candidate_spread)
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
    estimated_baseline_ms = (
        baseline_us * layer_invocations / 1000.0
        if baseline_us is not None
        else None
    )
    estimated_selected_ms = (
        selected_us * layer_invocations / 1000.0
        if selected_us is not None
        else None
    )
    estimated_savings_ms = (
        estimated_baseline_ms - estimated_selected_ms
        if estimated_baseline_ms is not None
        and estimated_selected_ms is not None
        else None
    )
    winner_name = (
        str(measured_winner["case"])
        if apply_change and measured_winner is not None
        else "current"
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_CURRENT",
        "apply_change": apply_change,
        "winner": winner_name,
        "measured_winner": (
            measured_winner["case"] if measured_winner is not None else None
        ),
        "raw_measured_winner": (
            raw_measured_winner["case"]
            if raw_measured_winner is not None
            else None
        ),
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
        "layers": layers,
        "chunks": chunks,
        "layer_invocations": layer_invocations,
        "baseline_us": baseline_us,
        "selected_us": selected_us,
        "cold_baseline_us": cold_baseline_us,
        "cold_baseline_ratio": cold_baseline_ratio,
        "baseline_sample_spread_ratio": baseline_sample_spread_ratio,
        "candidate_sample_spread_ratio": candidate_sample_spread_ratio,
        "stable": stable,
        "maximum_baseline_spread": float(args.maximum_baseline_spread),
        "maximum_candidate_spread": float(args.maximum_candidate_spread),
        "minimum_speedup": float(args.minimum_speedup),
        "near_tie_ratio": float(args.near_tie_ratio),
        "speedup": speedup,
        "estimated_baseline_ms_b16_prefill": estimated_baseline_ms,
        "estimated_selected_ms_b16_prefill": estimated_selected_ms,
        "estimated_savings_ms_b16_prefill": estimated_savings_ms,
        "target_prefill_gap_ms": target_gap_ms,
        "estimated_gap_coverage": (
            estimated_savings_ms / target_gap_ms
            if estimated_savings_ms is not None
            else None
        ),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "cases": measured,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
