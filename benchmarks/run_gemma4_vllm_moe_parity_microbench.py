#!/usr/bin/env python3
"""Compare MegaGemm and vLLM Gemma4 MoE kernels without a checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import megagemm.kernels.qwen3_moe as moe_kernel
from megagemm.kernels.qwen3_moe import (
    qwen3_moe_grouped_decode,
    qwen3_moe_prepare_segmented_prefill_graph_workspace,
    qwen3_moe_segmented_prefill,
)


HIDDEN = 2816
INTERMEDIATE = 704
NUM_EXPERTS = 128
TOP_K = 8
MAX_BASELINE_STABILITY_RATIO = 1.03
MAX_ABS_ERROR = 0.05
MAX_RELATIVE_L2_ERROR = 0.01
MIN_COSINE = 0.999


def build_fixed_b16_route() -> torch.Tensor:
    """Build the measured 90-expert B16 routing distribution."""
    rows_count = 16
    counts = [1] * 60 + [2] * 22 + [3] * 8 + [0] * 38
    rows: list[list[int]] = [[] for _ in range(rows_count)]
    tie_cursor = 0
    for expert, count in sorted(
        enumerate(counts),
        key=lambda item: (-item[1], item[0]),
    ):
        for _ in range(count):
            candidates = [
                row
                for row in range(rows_count)
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


def build_prefill_route(
    rows: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic, model-like top-k route for prefill parity."""
    logits = torch.randn(
        (rows, NUM_EXPERTS),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    selected_logits, selected = torch.topk(logits, TOP_K, dim=-1)
    routing = torch.softmax(selected_logits, dim=-1).to(torch.bfloat16)
    return selected.contiguous(), routing.contiguous()


def configure_megagemm_baseline() -> None:
    moe_kernel._CFG_SHARED_ROUTE_DECODE = False
    moe_kernel._CFG_ROUTE_MATRIX_DECODE = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = True
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = False
    moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = False
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


def measure_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples_us: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(
            float(start.elapsed_time(end)) * 1000.0 / iterations
        )
    return samples_us


def output_error(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    candidate_fp32 = candidate.float()
    reference_fp32 = reference.float()
    delta = candidate_fp32 - reference_fp32
    reference_norm = float(torch.linalg.vector_norm(reference_fp32).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    return {
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": delta_norm / max(reference_norm, 1.0e-12),
        "cosine": float(
            F.cosine_similarity(
                candidate_fp32.flatten(),
                reference_fp32.flatten(),
                dim=0,
            ).item()
        ),
    }


@torch.inference_mode()
def run_megagemm_case(
    *,
    name: str,
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    selected: torch.Tensor,
    routing: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    rows = int(hidden.shape[0])
    assignments = rows * TOP_K
    is_decode = rows == 16
    if is_decode:
        configure_megagemm_baseline()
    else:
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = True
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = max(
            4096,
            assignments,
        )
    workspace: dict[str, Any] = {}
    out = torch.empty_like(hidden)

    def call() -> torch.Tensor:
        if is_decode:
            return qwen3_moe_grouped_decode(
                hidden,
                gate_up,
                down,
                selected,
                routing,
                activation="gelu_pytorch_tanh",
                out=out,
                workspace=workspace,
                max_assignments=assignments,
                expert_grouped_compact=True,
                expert_grouped_min_rows=9,
                expert_grouped_max_rows=16,
                expert_grouped_compact_partial_reduce=True,
            )
        return qwen3_moe_segmented_prefill(
            hidden,
            gate_up,
            down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=out,
            workspace=workspace,
            force=True,
            block_m=32,
            block_n=128,
            block_k=64,
            fused_gate_block_n=64,
            num_warps=4,
            num_stages=3,
            fused_gate=True,
            dense_grid=False,
            route_scatter=True,
            fixed_route_pack=False,
            compact_route_pack=True,
            async_tiles_max_assignments=4096,
            single_accumulator=False,
            group_size_m=8,
        )

    for _ in range(max(warmup, 1)):
        call()
    torch.cuda.synchronize()
    if is_decode:
        path = str(workspace.get("grouped_decode_last_path") or "")
        expected_path = "expert_grouped_compact"
        failure = str(
            workspace.get("expert_grouped_compact_decode_fail_reason") or ""
        )
    else:
        compact_pack = int(
            workspace.get("segmented_prefill_compact_route_pack", 0) or 0
        )
        partial_reduce = int(
            workspace.get("segmented_prefill_partial_reduce", 0) or 0
        )
        path = (
            "segmented_prefill_compact"
            if compact_pack and partial_reduce
            else "segmented_prefill_fallback"
        )
        expected_path = "segmented_prefill_compact"
        failure = str(
            workspace.get("segmented_prefill_route_scatter_fail_reason") or ""
        )
        qwen3_moe_prepare_segmented_prefill_graph_workspace(
            workspace,
            assignments=assignments,
            hidden_dim=HIDDEN,
            device=hidden.device,
            num_experts=NUM_EXPERTS,
            block_m=32,
            route_dtype=routing.dtype,
        )
    if path != expected_path:
        raise RuntimeError(
            "MegaGemm baseline dispatch failed before capture: "
            f"expected={expected_path!r} actual={path!r} "
            f"failure={failure!r}"
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    if not is_decode:
        graph_route_pack = int(
            workspace.get("segmented_prefill_graph_route_pack", 0) or 0
        )
        graph_partial_cached = int(
            workspace.get("segmented_prefill_graph_partial_cached", 0) or 0
        )
        route_pack_passes = int(
            workspace.get("segmented_prefill_compact_route_pack_passes", 0)
            or 0
        )
        if (
            graph_route_pack != 1
            or graph_partial_cached != 1
            or route_pack_passes != 2
        ):
            raise RuntimeError(
                "MegaGemm prefill graph contract failed: "
                f"graph_route_pack={graph_route_pack} "
                f"graph_partial_cached={graph_partial_cached} "
                f"route_pack_passes={route_pack_passes}"
            )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    second = out.clone()
    samples_us = measure_graph(
        graph,
        iterations=iterations,
        repeats=repeats,
    )
    row = {
        "case": name,
        "backend": "megagemm",
        "callable": (
            "qwen3_moe_grouped_decode"
            if is_decode
            else "qwen3_moe_segmented_prefill"
        ),
        "measurement": "cuda_graph",
        "path": path,
        "median_us": float(statistics.median(samples_us)),
        "samples_us": samples_us,
        "repeat_max_abs_error": float(
            (second.float() - first.float()).abs().max().item()
        ),
        "error": None,
    }
    if not is_decode:
        row.update(
            {
                "graph_route_pack": graph_route_pack,
                "graph_partial_cached": graph_partial_cached,
                "route_pack_passes": route_pack_passes,
                "partial_bytes": int(
                    workspace.get("segmented_prefill_partial_bytes", 0) or 0
                ),
                "max_tiles": int(
                    workspace.get("segmented_prefill_max_tiles", 0) or 0
                ),
            }
        )
    del graph
    return row, first


@torch.inference_mode()
def run_vllm_case(
    *,
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    selected: torch.Tensor,
    routing: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    def call() -> torch.Tensor:
        return fused_experts(
            hidden,
            gate_up,
            down,
            routing,
            selected,
            activation=MoEActivation.GELU_TANH,
            apply_router_weight_on_input=False,
            global_num_experts=NUM_EXPERTS,
        )

    warmup_out: torch.Tensor | None = None
    for _ in range(max(warmup, 1)):
        warmup_out = call()
    torch.cuda.synchronize()
    del warmup_out

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    first = captured_out.clone()
    graph.replay()
    torch.cuda.synchronize()
    second = captured_out.clone()
    samples_us = measure_graph(
        graph,
        iterations=iterations,
        repeats=repeats,
    )
    row = {
        "case": "vllm_fused_experts",
        "backend": "vllm",
        "callable": "vllm.model_executor.layers.fused_moe.fused_moe.fused_experts",
        "activation": "MoEActivation.GELU_TANH",
        "measurement": "cuda_graph",
        "median_us": float(statistics.median(samples_us)),
        "samples_us": samples_us,
        "repeat_max_abs_error": float(
            (second.float() - first.float()).abs().max().item()
        ),
        "error": None,
    }
    del graph
    return row, first


def failed_case(name: str, backend: str, exc: Exception) -> dict[str, Any]:
    return {
        "case": name,
        "backend": backend,
        "measurement": "cuda_graph",
        "median_us": None,
        "samples_us": [],
        "repeat_max_abs_error": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        choices=(16, 400),
        default=16,
        help="16 measures B16 decode; 400 measures B16 x 25-token prefill",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_vllm_moe_parity_gate.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    try:
        import vllm
    except Exception as exc:
        raise SystemExit(
            "vLLM must be installed before this no-checkpoint parity gate: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    rows = int(args.rows)
    assignments = rows * TOP_K
    mode = "decode" if rows == 16 else "prefill"
    print(f"Gemma4 B16 {mode} MegaGemm-vLLM fused-MoE parity gate")
    print("  harness_rev: gemma4-vllm-moe-parity-v2-prefill")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(f"  vllm: {vllm.__version__}")
    print(
        "  shape: "
        f"rows={rows} H={HIDDEN} I={INTERMEDIATE} "
        f"E={NUM_EXPERTS} top_k={TOP_K} assignments={assignments}"
    )
    print("  model_download: disabled")
    print("  comparison: same tensors, same route, CUDA graph vs CUDA graph")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260727)
    dtype = torch.bfloat16
    hidden = torch.empty(
        (rows, HIDDEN),
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
    if rows == 16:
        selected = build_fixed_b16_route().to(device="cuda")
        routing_fp32 = torch.rand(
            (rows, TOP_K),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        routing = (
            routing_fp32 / routing_fp32.sum(dim=-1, keepdim=True)
        ).to(dtype)
    else:
        selected, routing = build_prefill_route(
            rows,
            generator=generator,
        )

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
        "_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE",
        "_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS",
    )
    original = {name: getattr(moe_kernel, name) for name in mutable_names}
    cases: list[dict[str, Any]] = []
    reference: torch.Tensor | None = None
    recheck_out: torch.Tensor | None = None
    try:
        try:
            current, reference = run_megagemm_case(
                name="megagemm_current",
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            cases.append(current)
        except Exception as exc:
            cases.append(failed_case("megagemm_current", "megagemm", exc))

        try:
            recheck, recheck_out = run_megagemm_case(
                name="megagemm_current_recheck",
                hidden=hidden,
                gate_up=gate_up,
                down=down,
                selected=selected,
                routing=routing,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            if reference is not None:
                recheck.update(output_error(recheck_out, reference))
            cases.append(recheck)
        except Exception as exc:
            cases.append(
                failed_case("megagemm_current_recheck", "megagemm", exc)
            )
    finally:
        for name, value in original.items():
            setattr(moe_kernel, name, value)

    vllm_out: torch.Tensor | None = None
    try:
        vllm_case, vllm_out = run_vllm_case(
            hidden=hidden,
            gate_up=gate_up,
            down=down,
            selected=selected,
            routing=routing,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        if reference is not None:
            vllm_case.update(output_error(vllm_out, reference))
        cases.append(vllm_case)
    except Exception as exc:
        cases.append(failed_case("vllm_fused_experts", "vllm", exc))

    for row in cases:
        print(json.dumps(row, sort_keys=True))

    current_rows = [
        row
        for row in cases
        if row["case"] in {
            "megagemm_current",
            "megagemm_current_recheck",
        }
        and row.get("median_us") is not None
        and row.get("error") is None
    ]
    vllm_rows = [
        row
        for row in cases
        if row["case"] == "vllm_fused_experts"
        and row.get("median_us") is not None
        and row.get("error") is None
    ]
    baseline_us = (
        min(float(row["median_us"]) for row in current_rows)
        if len(current_rows) == 2
        else None
    )
    baseline_stability = (
        max(float(row["median_us"]) for row in current_rows) / baseline_us
        if baseline_us is not None
        else None
    )
    vllm_row = vllm_rows[0] if len(vllm_rows) == 1 else None
    vllm_us = (
        float(vllm_row["median_us"]) if vllm_row is not None else None
    )
    speedup = (
        baseline_us / vllm_us
        if baseline_us is not None and vllm_us is not None
        else None
    )
    vllm_correct = bool(
        vllm_row is not None
        and vllm_row.get("repeat_max_abs_error") == 0.0
        and float(vllm_row.get("max_abs_error", float("inf")))
        <= MAX_ABS_ERROR
        and float(vllm_row.get("relative_l2_error", float("inf")))
        <= MAX_RELATIVE_L2_ERROR
        and float(vllm_row.get("cosine", 0.0)) >= MIN_COSINE
    )
    baseline_correct = bool(
        len(current_rows) == 2
        and reference is not None
        and recheck_out is not None
        and current_rows[0].get("repeat_max_abs_error") == 0.0
        and current_rows[1].get("repeat_max_abs_error") == 0.0
        and torch.equal(reference, recheck_out)
    )
    valid = bool(
        baseline_correct
        and vllm_correct
        and baseline_stability is not None
        and baseline_stability <= MAX_BASELINE_STABILITY_RATIO
    )
    if not valid:
        decision_name = "INVALID_PARITY_GATE"
    elif speedup is not None and speedup >= args.minimum_speedup:
        decision_name = "PORT_VLLM_FUSED_MOE"
    else:
        decision_name = "MOVE_OFF_MOE"

    decision = {
        "decision": decision_name,
        "valid": valid,
        "model_download": False,
        "minimum_port_speedup": float(args.minimum_speedup),
        "maximum_baseline_stability_ratio": MAX_BASELINE_STABILITY_RATIO,
        "maximum_abs_error": MAX_ABS_ERROR,
        "maximum_relative_l2_error": MAX_RELATIVE_L2_ERROR,
        "minimum_cosine": MIN_COSINE,
        "baseline_correct": baseline_correct,
        "vllm_correct": vllm_correct,
        "baseline_us": baseline_us,
        "vllm_us": vllm_us,
        "baseline_stability_ratio": baseline_stability,
        "vllm_speedup_vs_megagemm": speedup,
        "mode": mode,
        "active_experts": int(torch.unique(selected).numel()),
        "singleton_experts": int(
            (torch.bincount(selected.reshape(-1), minlength=NUM_EXPERTS) == 1)
            .sum()
            .item()
        ),
        "empty_experts": int(
            (torch.bincount(selected.reshape(-1), minlength=NUM_EXPERTS) == 0)
            .sum()
            .item()
        ),
        "shape": {
            "rows": rows,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "dtype": "bf16",
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
        "cases": cases,
    }
    print("DECISION " + json.dumps(decision, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
