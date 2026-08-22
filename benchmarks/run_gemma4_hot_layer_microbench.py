"""Cheap synthetic microbench for the Gemma4 MoE hot layer.

This allocates one layer worth of BF16 weights and never downloads a model.
It measures the exact 26B-A4B text shapes on CUDA:
  hidden=2816, shared_intermediate=2112, expert_intermediate=704,
  experts=128, top_k=8.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import megagemm.kernels.qwen3_moe as moe_kernel
import megagemm.models.llama as llama_mod
from megagemm.kernels.qwen3_moe import (
    qwen3_moe_grouped_decode,
    qwen3_moe_router_topk_softmax,
    qwen3_moe_segmented_prefill,
    qwen3_moe_topk_softmax,
)
from megagemm.models.llama import LlamaConfig, LlamaMLP, Qwen3MoeExperts


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def sync() -> None:
    torch.cuda.synchronize()


def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    max_abs = float((ref - cand).abs().max().item()) if ref.numel() else 0.0
    cosine = float(F.cosine_similarity(ref, cand, dim=0).item()) if ref.numel() else 1.0
    return {"max_abs_error": max_abs, "cosine": cosine}


def measure_graph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            fn()
        sync()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(3):
            graph.replay()
        sync()

        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            sync()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
        return {
            "measurement": "cuda_graph",
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "error": None,
        }
    except Exception as exc:
        return {
            "measurement": "cuda_graph",
            "median_us": None,
            "samples_us": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def measure_multistream_graph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    """Capture a fork/join graph from a non-default origin stream."""
    origin = torch.cuda.Stream()
    caller = torch.cuda.current_stream()
    try:
        origin.wait_stream(caller)
        with torch.cuda.stream(origin):
            for _ in range(warmup):
                fn()
        caller.wait_stream(origin)
        sync()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=origin):
            fn()
        caller.wait_stream(origin)
        sync()

        with torch.cuda.stream(origin):
            for _ in range(3):
                graph.replay()
        caller.wait_stream(origin)
        sync()

        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(origin):
                start.record()
                for _ in range(iterations):
                    graph.replay()
                end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
        return {
            "measurement": "cuda_graph_multistream",
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "error": None,
        }
    except Exception as exc:
        return {
            "measurement": "cuda_graph_multistream",
            "median_us": None,
            "samples_us": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def measure_eager(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            fn()
        sync()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            sync()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
        return {
            "measurement": "eager_cuda_events",
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "error": None,
        }
    except Exception as exc:
        return {
            "measurement": "eager_cuda_events",
            "median_us": None,
            "samples_us": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def random_weight(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, device="cuda", dtype=dtype).normal_(mean=0.0, std=0.02)


def accepted(row: dict[str, Any], *, max_abs: float = 0.08, min_cosine: float = 0.999) -> bool:
    return (
        row.get("error") is None
        and row.get("median_us") is not None
        and float(row.get("max_abs_error", 0.0)) <= max_abs
        and float(row.get("cosine", 1.0)) >= min_cosine
    )


def add_speedups(rows: list[dict[str, Any]], baseline_case: str) -> None:
    baseline = next((row for row in rows if row["case"] == baseline_case), None)
    base_us = float(baseline["median_us"]) if baseline and baseline.get("median_us") else 0.0
    for row in rows:
        value = float(row["median_us"]) if row.get("median_us") else 0.0
        row["speedup_vs_baseline"] = base_us / value if base_us > 0.0 and value > 0.0 else None


def best_case(rows: list[dict[str, Any]], baseline_case: str) -> dict[str, Any]:
    valid = [row for row in rows if accepted(row)]
    valid.sort(key=lambda row: float(row["median_us"]))
    winner = valid[0] if valid else next(row for row in rows if row["case"] == baseline_case)
    baseline = next(row for row in rows if row["case"] == baseline_case)
    improved = (
        winner.get("median_us") is not None
        and baseline.get("median_us") is not None
        and float(winner["median_us"]) < float(baseline["median_us"]) * 0.98
    )
    return {
        "baseline": baseline_case,
        "winner": winner["case"] if improved else baseline_case,
        "measured_winner": winner["case"],
        "apply_change": bool(improved and winner["case"] != baseline_case),
        "speedup": winner.get("speedup_vs_baseline"),
    }


def benchmark_shared_mlp(
    dtype: torch.dtype,
    hidden_dim: int,
    intermediate_dim: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("\n== SHARED MLP DECODE ==")
    hidden = random_weight((1, hidden_dim), dtype)
    norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    gate_up_weight = random_weight((2 * intermediate_dim, hidden_dim), dtype)
    down_weight = random_weight((hidden_dim, intermediate_dim), dtype)
    gate_up_t = gate_up_weight.t()
    down_t = down_weight.t()
    gate_up_out = torch.empty((1, 2 * intermediate_dim), device="cuda", dtype=dtype)
    down_out = torch.empty((1, hidden_dim), device="cuda", dtype=dtype)

    def cublas_tail() -> torch.Tensor:
        normed = llama_mod._decode_rmsnorm(hidden, norm_weight, 1e-6, False)
        torch.mm(normed, gate_up_t, out=gate_up_out)
        activated = F.gelu(gate_up_out[:, :intermediate_dim], approximate="tanh")
        activated.mul_(gate_up_out[:, intermediate_dim:])
        torch.mm(activated, down_t, out=down_out)
        return down_out

    cases: list[tuple[str, Callable[[], torch.Tensor]]] = [("cublas_two_stage", cublas_tail)]
    if llama_mod.deepfusion_swiglu_down is not None:
        def cublas_deepfusion() -> torch.Tensor:
            normed = llama_mod._decode_rmsnorm(hidden, norm_weight, 1e-6, False)
            torch.mm(normed, gate_up_t, out=gate_up_out)
            return llama_mod.deepfusion_swiglu_down(
                gate_up_out,
                down_weight,
                out=down_out,
                activation="gelu_tanh",
            )

        cases.append(("cublas_plus_deepfusion", cublas_deepfusion))
    if llama_mod.fused_rmsnorm_linear is not None:
        def fused_gate_cublas_tail() -> torch.Tensor:
            llama_mod.fused_rmsnorm_linear(
                hidden,
                norm_weight,
                1e-6,
                gate_up_weight,
                norm_offset=False,
                out=gate_up_out,
            )
            activated = F.gelu(gate_up_out[:, :intermediate_dim], approximate="tanh")
            activated.mul_(gate_up_out[:, intermediate_dim:])
            torch.mm(activated, down_t, out=down_out)
            return down_out

        cases.append(("fused_norm_gate_plus_cublas", fused_gate_cublas_tail))
        if llama_mod.deepfusion_swiglu_down is not None:
            def fully_fused_ends() -> torch.Tensor:
                llama_mod.fused_rmsnorm_linear(
                    hidden,
                    norm_weight,
                    1e-6,
                    gate_up_weight,
                    norm_offset=False,
                    out=gate_up_out,
                )
                return llama_mod.deepfusion_swiglu_down(
                    gate_up_out,
                    down_weight,
                    out=down_out,
                    activation="gelu_tanh",
                )

            cases.append(("fused_norm_gate_plus_deepfusion", fully_fused_ends))

    reference = cublas_tail().detach().clone()
    sync()
    rows = []
    for name, fn in cases:
        timing = measure_graph(
            fn,
            warmup=args.warmup,
            iterations=args.graph_iterations,
            repeats=args.repeats,
        )
        try:
            output = fn().detach().clone()
            sync()
            error = tensor_error(reference, output)
        except Exception as exc:
            error = {"max_abs_error": float("inf"), "cosine": 0.0}
            if timing["error"] is None:
                timing["error"] = f"{type(exc).__name__}: {exc}"
        row = {"case": name, **timing, **error}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    add_speedups(rows, "cublas_two_stage")
    recommendation = best_case(rows, "cublas_two_stage")
    del hidden, norm_weight, gate_up_weight, down_weight, gate_up_out, down_out, reference
    empty_cache()
    return rows, recommendation


def benchmark_router(
    dtype: torch.dtype,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("\n== GEMMA4 ROUTER DECODE ==")
    hidden = random_weight((1, hidden_dim), dtype)
    norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    hidden_scale = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    expert_scale = torch.ones(num_experts, device="cuda", dtype=dtype)
    router_weight = random_weight((num_experts, hidden_dim), dtype)
    logits = torch.empty((1, num_experts), device="cuda", dtype=dtype)
    root = hidden_dim ** -0.5

    legacy_workspace: dict[str, torch.Tensor] = {}
    legacy_result: dict[str, torch.Tensor] = {}

    def legacy() -> torch.Tensor:
        normed = llama_mod._decode_rmsnorm(hidden, norm_weight, 1e-6, False)
        normed.mul_(hidden_scale).mul_(root)
        torch.mm(normed, router_weight.t(), out=logits)
        weights, experts = qwen3_moe_topk_softmax(logits, top_k, workspace=legacy_workspace)
        weights.mul_(expert_scale[experts])
        legacy_result["weights"] = weights
        legacy_result["experts"] = experts
        return weights

    original_fused = bool(moe_kernel._CFG_FUSED_ROUTER)
    original_splits = int(moe_kernel._CFG_ROUTER_K_SPLITS)
    original_max_rows = int(moe_kernel._CFG_FUSED_ROUTER_MAX_ROWS)
    rows = []
    try:
        moe_kernel._CFG_FUSED_ROUTER_MAX_ROWS = 1
        moe_kernel._CFG_FUSED_ROUTER = False
        reference_weights = legacy().detach().clone()
        reference_experts = legacy_result["experts"].detach().clone()
        sync()
        timing = measure_graph(
            legacy,
            warmup=args.warmup,
            iterations=args.graph_iterations,
            repeats=args.repeats,
        )
        row = {
            "case": "legacy_mm_plus_topk",
            **timing,
            "max_abs_error": 0.0,
            "cosine": 1.0,
            "experts_equal": True,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

        for splits in (1, 2, 4, 8):
            workspace: dict[str, torch.Tensor] = {}
            result: dict[str, torch.Tensor] = {}
            moe_kernel._CFG_FUSED_ROUTER = True
            moe_kernel._CFG_ROUTER_K_SPLITS = splits

            def fused() -> torch.Tensor:
                normed = llama_mod._decode_rmsnorm(hidden, norm_weight, 1e-6, False)
                normed.mul_(hidden_scale).mul_(root)
                weights, experts = qwen3_moe_router_topk_softmax(
                    normed,
                    router_weight,
                    top_k,
                    workspace=workspace,
                )
                weights.mul_(expert_scale[experts])
                result["weights"] = weights
                result["experts"] = experts
                return weights

            timing = measure_graph(
                fused,
                warmup=args.warmup,
                iterations=args.graph_iterations,
                repeats=args.repeats,
            )
            try:
                output = fused().detach().clone()
                experts = result["experts"].detach().clone()
                sync()
                error = tensor_error(reference_weights, output)
                experts_equal = bool(torch.equal(reference_experts, experts))
            except Exception as exc:
                error = {"max_abs_error": float("inf"), "cosine": 0.0}
                experts_equal = False
                if timing["error"] is None:
                    timing["error"] = f"{type(exc).__name__}: {exc}"
            row = {
                "case": f"fused_projection_topk_splits{splits}",
                "splits": splits,
                **timing,
                **error,
                "experts_equal": experts_equal,
            }
            if not experts_equal and row["error"] is None:
                row["error"] = "selected experts differ from legacy"
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        moe_kernel._CFG_FUSED_ROUTER = original_fused
        moe_kernel._CFG_ROUTER_K_SPLITS = original_splits
        moe_kernel._CFG_FUSED_ROUTER_MAX_ROWS = original_max_rows

    add_speedups(rows, "legacy_mm_plus_topk")
    recommendation = best_case(rows, "legacy_mm_plus_topk")
    del hidden, norm_weight, hidden_scale, expert_scale, router_weight, logits
    empty_cache()
    return rows, recommendation


def benchmark_expert_decode(
    dtype: torch.dtype,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor, torch.Tensor]:
    print("\n== ROUTED EXPERT DECODE ==")
    hidden = random_weight((1, hidden_dim), dtype)
    gate_up_weight = random_weight((num_experts, 2 * intermediate_dim, hidden_dim), dtype)
    down_weight = random_weight((num_experts, hidden_dim, intermediate_dim), dtype)
    selected = torch.arange(top_k, device="cuda", dtype=torch.int64).reshape(1, top_k)
    routing = torch.rand((1, top_k), device="cuda", dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))

    original = {
        "block_n": int(moe_kernel._CFG_BLOCK_N),
        "block_k": int(moe_kernel._CFG_BLOCK_K),
        "warps": int(moe_kernel._CFG_NUM_WARPS),
        "stages": int(moe_kernel._CFG_NUM_STAGES),
        "max_assignments": int(moe_kernel._CFG_MAX_ASSIGNMENTS),
        "token_accum": bool(moe_kernel._CFG_TOKEN_ACCUM),
        "grouped_fused_gate": bool(moe_kernel._CFG_GROUPED_FUSED_GATE),
        "shared_route": bool(moe_kernel._CFG_SHARED_ROUTE_DECODE),
        "shared_route_coalesced": bool(moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS),
        "shared_route_split_gate": bool(moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE),
        "single_row_gemv": bool(moe_kernel._CFG_SINGLE_ROW_GEMV),
    }
    configs = [
        ("current", 64, 128, 4, 2, False, False, False, False, False),
        ("bk64", 64, 64, 4, 2, False, False, False, False, False),
        ("bn128_bk64", 128, 64, 4, 2, False, False, False, False, False),
        ("warps8", 64, 128, 8, 2, False, False, False, False, False),
        ("token_accum", 64, 128, 4, 2, True, False, False, False, False),
        ("fused_gate_token_accum", 64, 128, 4, 2, True, True, False, False, False),
        ("shared_route", 64, 128, 4, 2, False, False, True, False, False),
        ("shared_route_coalesced", 64, 128, 4, 2, False, False, True, True, False),
        ("shared_route_split_gate", 64, 128, 4, 2, False, False, True, False, True),
    ]
    rows = []
    reference = None
    try:
        moe_kernel._CFG_MAX_ASSIGNMENTS = 64
        for (
            label,
            block_n,
            block_k,
            warps,
            stages,
            token_accum,
            fused_gate,
            shared_route,
            shared_route_coalesced,
            shared_route_split_gate,
        ) in configs:
            moe_kernel._CFG_BLOCK_N = block_n
            moe_kernel._CFG_BLOCK_K = block_k
            moe_kernel._CFG_NUM_WARPS = warps
            moe_kernel._CFG_NUM_STAGES = stages
            moe_kernel._CFG_TOKEN_ACCUM = token_accum
            moe_kernel._CFG_GROUPED_FUSED_GATE = fused_gate
            moe_kernel._CFG_SHARED_ROUTE_DECODE = shared_route
            moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = shared_route_coalesced
            moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = shared_route_split_gate
            moe_kernel._CFG_SINGLE_ROW_GEMV = False
            workspace: dict[str, torch.Tensor] = {}
            out = torch.empty_like(hidden)

            def run() -> torch.Tensor:
                return qwen3_moe_grouped_decode(
                    hidden,
                    gate_up_weight,
                    down_weight,
                    selected,
                    routing,
                    activation="gelu_pytorch_tanh",
                    out=out,
                    workspace=workspace,
                )

            timing = measure_graph(
                run,
                warmup=args.warmup,
                iterations=args.graph_iterations,
                repeats=args.repeats,
            )
            try:
                output = run().detach().clone()
                sync()
                if reference is None:
                    reference = output
                error = tensor_error(reference, output)
            except Exception as exc:
                error = {"max_abs_error": float("inf"), "cosine": 0.0}
                if timing["error"] is None:
                    timing["error"] = f"{type(exc).__name__}: {exc}"
            case = label
            row = {
                "case": case,
                "block_n": block_n,
                "block_k": block_k,
                "warps": warps,
                "stages": stages,
                "token_accum": token_accum,
                "grouped_fused_gate": fused_gate,
                "shared_route": shared_route,
                "shared_route_coalesced": shared_route_coalesced,
                "shared_route_split_gate": shared_route_split_gate,
                **timing,
                **error,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        moe_kernel._CFG_BLOCK_N = original["block_n"]
        moe_kernel._CFG_BLOCK_K = original["block_k"]
        moe_kernel._CFG_NUM_WARPS = original["warps"]
        moe_kernel._CFG_NUM_STAGES = original["stages"]
        moe_kernel._CFG_MAX_ASSIGNMENTS = original["max_assignments"]
        moe_kernel._CFG_TOKEN_ACCUM = original["token_accum"]
        moe_kernel._CFG_GROUPED_FUSED_GATE = original["grouped_fused_gate"]
        moe_kernel._CFG_SHARED_ROUTE_DECODE = original["shared_route"]
        moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = original["shared_route_coalesced"]
        moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = original["shared_route_split_gate"]
        moe_kernel._CFG_SINGLE_ROW_GEMV = original["single_row_gemv"]

    add_speedups(rows, "current")
    recommendation = best_case(rows, "current")
    del hidden, selected, routing
    empty_cache()
    return rows, recommendation, gate_up_weight, down_weight


def benchmark_parallel_moe_decode(
    dtype: torch.dtype,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    num_experts: int,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the current serial Gemma4 MoE branches with a two-stream fork/join."""
    print("\n== PARALLEL SHARED + ROUTED EXPERT DECODE ==")
    hidden = random_weight((1, hidden_dim), dtype)
    shared_norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    expert_norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    shared_post_norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    expert_post_norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)
    combined_post_norm_weight = torch.ones(hidden_dim, device="cuda", dtype=dtype)

    shared_gate_up_weight = random_weight((2 * shared_intermediate, hidden_dim), dtype)
    shared_down_weight = random_weight((hidden_dim, shared_intermediate), dtype)
    shared_gate_up_t = shared_gate_up_weight.t()
    shared_down_t = shared_down_weight.t()
    shared_gate_up_out = torch.empty(
        (1, 2 * shared_intermediate), device="cuda", dtype=dtype
    )
    shared_out = torch.empty_like(hidden)

    expert_gate_up_weight = random_weight(
        (num_experts, 2 * expert_intermediate, hidden_dim), dtype
    )
    expert_down_weight = random_weight(
        (num_experts, hidden_dim, expert_intermediate), dtype
    )
    selected = torch.arange(top_k, device="cuda", dtype=torch.int64).reshape(1, top_k)
    routing = torch.rand((1, top_k), device="cuda", dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))
    expert_out = torch.empty_like(hidden)
    combined_out = torch.empty_like(hidden)
    expert_workspace: dict[str, torch.Tensor] = {}

    original = {
        "block_n": int(moe_kernel._CFG_BLOCK_N),
        "block_k": int(moe_kernel._CFG_BLOCK_K),
        "warps": int(moe_kernel._CFG_NUM_WARPS),
        "stages": int(moe_kernel._CFG_NUM_STAGES),
        "max_assignments": int(moe_kernel._CFG_MAX_ASSIGNMENTS),
        "token_accum": bool(moe_kernel._CFG_TOKEN_ACCUM),
        "grouped_fused_gate": bool(moe_kernel._CFG_GROUPED_FUSED_GATE),
        "shared_route": bool(moe_kernel._CFG_SHARED_ROUTE_DECODE),
        "single_row_gemv": bool(moe_kernel._CFG_SINGLE_ROW_GEMV),
    }
    moe_kernel._CFG_BLOCK_N = 64
    moe_kernel._CFG_BLOCK_K = 128
    moe_kernel._CFG_NUM_WARPS = 4
    moe_kernel._CFG_NUM_STAGES = 2
    moe_kernel._CFG_MAX_ASSIGNMENTS = 64
    moe_kernel._CFG_TOKEN_ACCUM = False
    moe_kernel._CFG_GROUPED_FUSED_GATE = False
    moe_kernel._CFG_SHARED_ROUTE_DECODE = False
    moe_kernel._CFG_SINGLE_ROW_GEMV = False

    def shared_branch() -> torch.Tensor:
        normed = llama_mod._decode_rmsnorm(
            hidden, shared_norm_weight, 1e-6, False
        )
        torch.mm(normed, shared_gate_up_t, out=shared_gate_up_out)
        activated = F.gelu(
            shared_gate_up_out[:, :shared_intermediate], approximate="tanh"
        )
        activated.mul_(shared_gate_up_out[:, shared_intermediate:])
        torch.mm(activated, shared_down_t, out=shared_out)
        return llama_mod._decode_rmsnorm(
            shared_out, shared_post_norm_weight, 1e-6, False
        )

    def expert_branch() -> torch.Tensor:
        normed = llama_mod._decode_rmsnorm(hidden, expert_norm_weight, 1e-6, False)
        qwen3_moe_grouped_decode(
            normed,
            expert_gate_up_weight,
            expert_down_weight,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=expert_out,
            workspace=expert_workspace,
        )
        return llama_mod._decode_rmsnorm(
            expert_out, expert_post_norm_weight, 1e-6, False
        )

    def sequential() -> torch.Tensor:
        shared_normed = shared_branch()
        expert_normed = expert_branch()
        torch.add(shared_normed, expert_normed, out=combined_out)
        return llama_mod._decode_rmsnorm(
            combined_out, combined_post_norm_weight, 1e-6, False
        )

    side_stream = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    join_event = torch.cuda.Event()

    def parallel() -> torch.Tensor:
        main_stream = torch.cuda.current_stream()
        fork_event.record(main_stream)
        with torch.cuda.stream(side_stream):
            side_stream.wait_event(fork_event)
            shared_normed = shared_branch()
            join_event.record(side_stream)
        expert_normed = expert_branch()
        main_stream.wait_event(join_event)
        torch.add(shared_normed, expert_normed, out=combined_out)
        return llama_mod._decode_rmsnorm(
            combined_out, combined_post_norm_weight, 1e-6, False
        )

    rows: list[dict[str, Any]] = []
    try:
        reference = sequential().detach().clone()
        sync()
        for name, fn in (("sequential", sequential), ("parallel_two_streams", parallel)):
            measure = measure_multistream_graph if name == "parallel_two_streams" else measure_graph
            timing = measure(
                fn,
                warmup=args.warmup,
                iterations=args.graph_iterations,
                repeats=args.repeats,
            )
            try:
                output = fn().detach().clone()
                sync()
                error = tensor_error(reference, output)
            except Exception as exc:
                error = {"max_abs_error": float("inf"), "cosine": 0.0}
                if timing["error"] is None:
                    timing["error"] = f"{type(exc).__name__}: {exc}"
            row = {
                "case": name,
                **timing,
                **error,
            }
            if row.get("median_us") is not None:
                row["ms_per_token_30_layers"] = float(row["median_us"]) * 30.0 / 1000.0
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    finally:
        moe_kernel._CFG_BLOCK_N = original["block_n"]
        moe_kernel._CFG_BLOCK_K = original["block_k"]
        moe_kernel._CFG_NUM_WARPS = original["warps"]
        moe_kernel._CFG_NUM_STAGES = original["stages"]
        moe_kernel._CFG_MAX_ASSIGNMENTS = original["max_assignments"]
        moe_kernel._CFG_TOKEN_ACCUM = original["token_accum"]
        moe_kernel._CFG_GROUPED_FUSED_GATE = original["grouped_fused_gate"]
        moe_kernel._CFG_SHARED_ROUTE_DECODE = original["shared_route"]
        moe_kernel._CFG_SINGLE_ROW_GEMV = original["single_row_gemv"]

    add_speedups(rows, "sequential")
    recommendation = best_case(rows, "sequential")
    empty_cache()
    return rows, recommendation


def make_shared_mlp_module(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
) -> LlamaMLP:
    cfg = LlamaConfig.from_dict(
        {
            "model_type": "llama",
            "hidden_size": hidden_dim,
            "intermediate_size": intermediate_dim,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 176,
            "vocab_size": 256,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
        }
    )
    with torch.device("meta"):
        mlp = LlamaMLP(cfg, 0)
    mlp.gate_up_proj.weight = torch.nn.Parameter(
        gate_up_weight,
        requires_grad=False,
    )
    mlp.down_proj.weight = torch.nn.Parameter(
        down_weight,
        requires_grad=False,
    )
    return mlp.eval()


def benchmark_parallel_moe_prefill(
    dtype: torch.dtype,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    num_experts: int,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure the production B16 prefill branches serially and on two streams."""
    rows_count = int(args.prefill_rows)
    print(f"\n== PARALLEL SHARED + ROUTED EXPERT PREFILL ({rows_count} ROWS) ==")

    shared_in = random_weight((rows_count, hidden_dim), dtype)
    expert_in = random_weight((rows_count, hidden_dim), dtype)
    shared_gate_up = random_weight(
        (2 * shared_intermediate, hidden_dim),
        dtype,
    )
    shared_down = random_weight(
        (hidden_dim, shared_intermediate),
        dtype,
    )
    expert_gate_up = random_weight(
        (num_experts, 2 * expert_intermediate, hidden_dim),
        dtype,
    )
    expert_down = random_weight(
        (num_experts, hidden_dim, expert_intermediate),
        dtype,
    )

    shared_mlp = make_shared_mlp_module(
        shared_gate_up,
        shared_down,
        hidden_dim,
        shared_intermediate,
    )
    experts = make_experts_module(
        expert_gate_up,
        expert_down,
        hidden_dim,
        expert_intermediate,
        num_experts,
        top_k,
        model_type="gemma4_text",
    )
    row_ids = torch.arange(
        rows_count,
        device="cuda",
        dtype=torch.int64,
    ).reshape(-1, 1)
    top_ids = torch.arange(
        top_k,
        device="cuda",
        dtype=torch.int64,
    ).reshape(1, -1)
    selected = (row_ids * 7 + top_ids * 11).remainder(num_experts).contiguous()
    routing = torch.rand(
        (rows_count, top_k),
        device="cuda",
        dtype=dtype,
    )
    routing.div_(routing.sum(dim=-1, keepdim=True))
    combined_out = torch.empty_like(shared_in)

    def shared_branch() -> torch.Tensor:
        return shared_mlp(shared_in, is_prefill=True)

    def expert_branch() -> torch.Tensor:
        return experts(
            expert_in,
            selected,
            routing,
            use_grouped_decode=False,
        )

    def sequential() -> torch.Tensor:
        shared_out = shared_branch()
        expert_out = expert_branch()
        torch.add(shared_out, expert_out, out=combined_out)
        return combined_out

    side_stream = torch.cuda.Stream()
    fork_event = torch.cuda.Event()
    join_event = torch.cuda.Event()

    def parallel() -> torch.Tensor:
        main_stream = torch.cuda.current_stream()
        fork_event.record(main_stream)
        with torch.cuda.stream(side_stream):
            side_stream.wait_event(fork_event)
            shared_out = shared_branch()
            join_event.record(side_stream)
        expert_out = expert_branch()
        main_stream.wait_event(join_event)
        torch.add(shared_out, expert_out, out=combined_out)
        return combined_out

    reference = sequential().detach().clone()
    sync()
    cases = (
        ("sequential", sequential),
        ("parallel_two_streams", parallel),
        ("sequential_recheck", sequential),
    )
    rows: list[dict[str, Any]] = []
    for name, fn in cases:
        timing = measure_eager(
            fn,
            warmup=max(2, args.warmup // 2),
            iterations=args.prefill_iterations,
            repeats=args.repeats,
        )
        try:
            output = fn().detach().clone()
            sync()
            error = tensor_error(reference, output)
        except Exception as exc:
            error = {"max_abs_error": float("inf"), "cosine": 0.0}
            if timing["error"] is None:
                timing["error"] = f"{type(exc).__name__}: {exc}"
        row = {
            "case": name,
            "rows": rows_count,
            "assignments": rows_count * top_k,
            **timing,
            **error,
        }
        if row.get("median_us") is not None:
            row["ms_per_prefill_30_layers"] = (
                float(row["median_us"]) * 30.0 / 1000.0
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    baseline_row = next(row for row in rows if row["case"] == "sequential")
    recheck_row = next(
        row for row in rows if row["case"] == "sequential_recheck"
    )
    candidate_row = next(
        row for row in rows if row["case"] == "parallel_two_streams"
    )
    baseline_values = [
        float(row["median_us"])
        for row in (baseline_row, recheck_row)
        if row.get("median_us") is not None
    ]
    baseline_us = min(baseline_values) if baseline_values else 0.0
    baseline_stability_ratio = (
        max(baseline_values) / baseline_us
        if baseline_us > 0.0 and len(baseline_values) == 2
        else float("inf")
    )
    candidate_us = float(candidate_row.get("median_us") or 0.0)
    speedup = (
        baseline_us / candidate_us
        if baseline_us > 0.0 and candidate_us > 0.0
        else None
    )
    apply_change = bool(
        accepted(candidate_row)
        and baseline_stability_ratio <= 1.03
        and speedup is not None
        and speedup >= 1.02
    )
    recommendation = {
        "baseline": "sequential",
        "winner": "parallel_two_streams" if apply_change else "sequential",
        "measured_winner": (
            "parallel_two_streams"
            if candidate_us > 0.0 and candidate_us < baseline_us
            else "sequential"
        ),
        "apply_change": apply_change,
        "speedup": speedup,
        "baseline_us": baseline_us,
        "candidate_us": candidate_us or None,
        "baseline_stability_ratio": baseline_stability_ratio,
        "stable": baseline_stability_ratio <= 1.03,
        "full_ab_total_gap_ms": float(args.prefill_target_savings_ms),
        "estimated_savings_ms_per_prefill_30_layers": (
            max(0.0, baseline_us - candidate_us) * 30.0 / 1000.0
            if candidate_us > 0.0
            else 0.0
        ),
    }
    recommendation["closes_full_ab_total_gap"] = bool(
        recommendation["estimated_savings_ms_per_prefill_30_layers"]
        >= recommendation["full_ab_total_gap_ms"]
    )

    del (
        shared_in,
        expert_in,
        shared_mlp,
        experts,
        selected,
        routing,
        reference,
    )
    empty_cache()
    return rows, recommendation


def make_experts_module(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    *,
    model_type: str = "qwen3_moe",
) -> Qwen3MoeExperts:
    cfg = LlamaConfig.from_dict(
        {
            "model_type": model_type,
            "hidden_size": hidden_dim,
            "intermediate_size": 8192,
            "moe_intermediate_size": intermediate_dim,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 176,
            "vocab_size": 256,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
            "num_experts": num_experts,
            "num_experts_per_tok": top_k,
            "enable_moe_block": True,
            "decoder_sparse_step": 1,
            "norm_topk_prob": True,
        }
    )
    with torch.device("meta"):
        experts = Qwen3MoeExperts(cfg)
    experts.gate_up_proj = torch.nn.Parameter(gate_up_weight, requires_grad=False)
    experts.down_proj = torch.nn.Parameter(down_weight, requires_grad=False)
    return experts.eval()


def benchmark_expert_prefill(
    dtype: torch.dtype,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print("\n== ROUTED EXPERT PREFILL TUNING (25 TOKENS) ==")
    rows_count = 25
    active_experts = 54
    hidden = random_weight((rows_count, hidden_dim), dtype)
    row_ids = torch.arange(rows_count, device="cuda", dtype=torch.int64).reshape(-1, 1)
    top_ids = torch.arange(top_k, device="cuda", dtype=torch.int64).reshape(1, -1)
    selected = (row_ids * 7 + top_ids * 11).remainder(active_experts).contiguous()
    routing = torch.rand((rows_count, top_k), device="cuda", dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))
    experts = make_experts_module(
        gate_up_weight,
        down_weight,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k,
    )

    def padded_bmm() -> torch.Tensor:
        return experts._forward_batched_prefill(hidden, selected, routing)

    reference = padded_bmm().detach().clone()
    sync()
    rows: list[dict[str, Any]] = []

    original = {
        "enabled": bool(moe_kernel._CFG_SEGMENTED_PREFILL),
        "min_assignments": int(moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS),
        "block_m": int(moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M),
        "block_n": int(moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N),
        "block_k": int(moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K),
        "warps": int(moe_kernel._CFG_SEGMENTED_PREFILL_NUM_WARPS),
        "stages": int(moe_kernel._CFG_SEGMENTED_PREFILL_NUM_STAGES),
        "route_scatter": bool(moe_kernel._CFG_SEGMENTED_PREFILL_ROUTE_SCATTER),
        "fused_gate": bool(moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE),
        "fused_gate_block_n": int(
            moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N
        ),
        "async_tiles": bool(moe_kernel._CFG_SEGMENTED_PREFILL_ASYNC_TILES),
        "partial_reduce": bool(moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE),
    }
    baseline = {
        "block_m": 16,
        "block_n": 64,
        "block_k": 128,
        "fused_gate_block_n": 64,
        "warps": 8,
        "stages": 4,
        "fixed_route_pack": False,
    }
    # Async compact tiles and partial reduction already won their A100 A/Bs.
    # This sweep changes one previously unmeasured launch parameter at a time.
    axes: list[tuple[str, list[dict[str, int]]]] = [
        ("down_block_n", [{"block_n": 32}, {"block_n": 128}]),
        (
            "fused_gate_block_n",
            [{"fused_gate_block_n": 32}, {"fused_gate_block_n": 128}],
        ),
        ("block_k", [{"block_k": 32}, {"block_k": 128}]),
        ("warps", [{"warps": 2}, {"warps": 8}]),
        ("stages", [{"stages": 2}, {"stages": 4}]),
    ]
    if args.prefill_fixed_route_pack_only:
        axes = [("fixed_route_pack", [{"fixed_route_pack": True}])]

    def case_name(config: dict[str, int], suffix: str = "") -> str:
        name = (
            f"segmented_bm{config['block_m']}_dn{config['block_n']}_"
            f"gn{config['fused_gate_block_n']}_bk{config['block_k']}_"
            f"w{config['warps']}_s{config['stages']}_"
            f"frp{int(config['fixed_route_pack'])}"
        )
        return f"{name}_{suffix}" if suffix else name

    def run_config(config: dict[str, int], *, axis: str, suffix: str = "") -> dict[str, Any]:
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M = config["block_m"]
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N = config["block_n"]
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K = config["block_k"]
        moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N = config[
            "fused_gate_block_n"
        ]
        moe_kernel._CFG_SEGMENTED_PREFILL_NUM_WARPS = config["warps"]
        moe_kernel._CFG_SEGMENTED_PREFILL_NUM_STAGES = config["stages"]
        workspace: dict[str, torch.Tensor] = {}
        out = torch.empty_like(hidden)

        def segmented() -> torch.Tensor:
            return qwen3_moe_segmented_prefill(
                hidden,
                gate_up_weight,
                down_weight,
                selected,
                routing,
                activation="gelu_pytorch_tanh",
                out=out,
                workspace=workspace,
                fixed_route_pack=bool(config["fixed_route_pack"]),
            )

        timing = measure_eager(
            segmented,
            warmup=max(2, args.warmup // 2),
            iterations=args.prefill_iterations,
            repeats=args.repeats,
        )
        try:
            output = segmented().detach().clone()
            sync()
            error = tensor_error(reference, output)
        except Exception as exc:
            error = {"max_abs_error": float("inf"), "cosine": 0.0}
            if timing["error"] is None:
                timing["error"] = f"{type(exc).__name__}: {exc}"
        row = {
            "case": case_name(config, suffix),
            "axis": axis,
            **config,
            "fused_gate": True,
            "async_tiles": True,
            "partial_reduce": True,
            "active_experts": active_experts,
            **timing,
            **error,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
        return row

    baseline_case = case_name(baseline)
    try:
        moe_kernel._CFG_SEGMENTED_PREFILL = True
        moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = 1
        moe_kernel._CFG_SEGMENTED_PREFILL_ROUTE_SCATTER = True
        moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE = True
        moe_kernel._CFG_SEGMENTED_PREFILL_ASYNC_TILES = True
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = True

        baseline_row = run_config(baseline.copy(), axis="baseline")
        axis_rows: dict[str, list[tuple[dict[str, int], dict[str, Any]]]] = {}
        for axis, changes_list in axes:
            axis_rows[axis] = []
            for changes in changes_list:
                config = {**baseline, **changes}
                row = run_config(config, axis=axis)
                axis_rows[axis].append((changes, row))

        baseline_us = float(baseline_row["median_us"] or 0.0)
        combined = baseline.copy()
        combined_axes: list[str] = []
        if baseline_us > 0.0:
            for axis, candidates in axis_rows.items():
                valid = [item for item in candidates if accepted(item[1])]
                if not valid:
                    continue
                changes, winner = min(valid, key=lambda item: float(item[1]["median_us"]))
                if float(winner["median_us"]) < baseline_us * 0.99:
                    combined.update(changes)
                    combined_axes.append(axis)
        if combined != baseline and len(combined_axes) > 1:
            run_config(
                combined,
                axis="combined",
                suffix="combined_" + "_".join(combined_axes),
            )

        run_config(baseline.copy(), axis="stability", suffix="recheck")
    finally:
        moe_kernel._CFG_SEGMENTED_PREFILL = original["enabled"]
        moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = original["min_assignments"]
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M = original["block_m"]
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N = original["block_n"]
        moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K = original["block_k"]
        moe_kernel._CFG_SEGMENTED_PREFILL_NUM_WARPS = original["warps"]
        moe_kernel._CFG_SEGMENTED_PREFILL_NUM_STAGES = original["stages"]
        moe_kernel._CFG_SEGMENTED_PREFILL_ROUTE_SCATTER = original["route_scatter"]
        moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE = original["fused_gate"]
        moe_kernel._CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N = original[
            "fused_gate_block_n"
        ]
        moe_kernel._CFG_SEGMENTED_PREFILL_ASYNC_TILES = original["async_tiles"]
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = original["partial_reduce"]

    add_speedups(rows, baseline_case)
    ranked_rows = [row for row in rows if row["axis"] != "stability"]
    recommendation = best_case(ranked_rows, baseline_case)
    selected_row = next(
        row for row in ranked_rows if row["case"] == recommendation["winner"]
    )
    baseline_row = next(row for row in rows if row["case"] == baseline_case)
    recheck_row = next(row for row in rows if row["axis"] == "stability")
    baseline_us = float(baseline_row["median_us"] or 0.0)
    selected_us = float(selected_row["median_us"] or baseline_us)
    recheck_us = float(recheck_row["median_us"] or baseline_us)
    drift = abs(recheck_us - baseline_us) / baseline_us if baseline_us > 0.0 else 1.0
    estimated_savings_ms = max(0.0, baseline_us - selected_us) * 30.0 / 1000.0
    recommendation.update(
        {
            "baseline_recheck_drift_pct": drift * 100.0,
            "estimated_savings_ms_per_prefill_30_layers": estimated_savings_ms,
            "full_ab_total_gap_ms": float(args.prefill_target_savings_ms),
            "closes_full_ab_total_gap": (
                estimated_savings_ms >= float(args.prefill_target_savings_ms)
            ),
            "stable": drift <= 0.03,
        }
    )
    if drift > 0.03:
        recommendation["apply_change"] = False
        recommendation["winner"] = baseline_case
        recommendation["reason"] = "baseline drift exceeded 3%"
    del hidden, selected, routing, experts, reference
    empty_cache()
    return rows, recommendation


def print_recommendations(recommendations: dict[str, dict[str, Any]]) -> None:
    print("\n== DECISION ==")
    for section, row in recommendations.items():
        action = "APPLY" if row["apply_change"] else "KEEP_BASELINE"
        speedup = row.get("speedup")
        speedup_text = f"{float(speedup):.3f}x" if speedup is not None else "n/a"
        print(
            f"{section}: {action} winner={row['winner']} "
            f"measured_winner={row['measured_winner']} speedup={speedup_text}"
        )
        if "estimated_savings_ms_per_prefill_30_layers" in row:
            print(
                "  objective: "
                f"estimated_savings={row['estimated_savings_ms_per_prefill_30_layers']:.3f}ms "
                f"target={row['full_ab_total_gap_ms']:.3f}ms "
                f"closes_gap={int(row['closes_full_ab_total_gap'])} "
                f"stable={int(row['stable'])}"
            )


def estimate_model_impact(
    sections: dict[str, list[dict[str, Any]]],
    recommendations: dict[str, dict[str, Any]],
    *,
    layers: int,
) -> dict[str, float]:
    baseline_us = 0.0
    selected_us = 0.0
    for name, rows in sections.items():
        recommendation = recommendations[name]
        baseline = next(row for row in rows if row["case"] == recommendation["baseline"])
        selected = next(row for row in rows if row["case"] == recommendation["winner"])
        if baseline.get("median_us") is None or selected.get("median_us") is None:
            continue
        baseline_us += float(baseline["median_us"])
        selected_us += float(selected["median_us"])
    return {
        "baseline_us_per_layer": baseline_us,
        "selected_us_per_layer": selected_us,
        "estimated_savings_ms_per_token": max(0.0, baseline_us - selected_us) * layers / 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--graph-iterations", type=int, default=100)
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument(
        "--prefill-target-savings-ms",
        type=float,
        default=8.47,
        help="Measured end-to-end gap that the expert-prefill candidate must recover",
    )
    parser.add_argument(
        "--prefill-fixed-route-pack-only",
        action="store_true",
        help="Compare current routing preparation with the fused fixed-shape packer",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-json", default="bench_results/gemma4_hot_layer_microbench.json")
    parser.add_argument("--only-expert-decode", action="store_true")
    parser.add_argument("--only-parallel-moe-decode", action="store_true")
    parser.add_argument("--only-parallel-moe-prefill", action="store_true")
    parser.add_argument("--only-shared-mlp", action="store_true")
    parser.add_argument("--only-expert-prefill", action="store_true")
    parser.add_argument(
        "--prefill-rows",
        type=int,
        default=400,
        help="Flattened token rows; B16 x context 25 is 400",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    dtype = dtype_from_name(args.dtype)
    hidden_dim = 2816
    shared_intermediate = 2112
    expert_intermediate = 704
    num_experts = 128
    top_k = 8

    props = torch.cuda.get_device_properties(0)
    print("Gemma4 MoE hot-layer microbenchmark")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  vram_gb: {props.total_memory / 1024**3:.2f}")
    print(f"  dtype: {args.dtype}")
    print(
        f"  shape: H={hidden_dim} shared_I={shared_intermediate} "
        f"expert_I={expert_intermediate} E={num_experts} top_k={top_k}"
    )

    torch.manual_seed(20260711)
    with torch.inference_mode():
        if args.only_expert_prefill:
            expert_gate_up = random_weight(
                (num_experts, 2 * expert_intermediate, hidden_dim), dtype
            )
            expert_down = random_weight(
                (num_experts, hidden_dim, expert_intermediate), dtype
            )
            prefill_rows, prefill_rec = benchmark_expert_prefill(
                dtype,
                hidden_dim,
                expert_intermediate,
                num_experts,
                top_k,
                expert_gate_up,
                expert_down,
                args,
            )
            recommendations = {"expert_prefill": prefill_rec}
            print_recommendations(recommendations)
            payload = {
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": props.total_memory / 1024**3,
                },
                "dtype": args.dtype,
                "shape": {
                    "rows": 25,
                    "hidden_dim": hidden_dim,
                    "expert_intermediate": expert_intermediate,
                    "num_experts": num_experts,
                    "top_k": top_k,
                },
                "expert_prefill": prefill_rows,
                "recommendations": recommendations,
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"\nwrote {out_path}")
            return 0
        if args.only_shared_mlp:
            shared_rows, shared_rec = benchmark_shared_mlp(
                dtype, hidden_dim, shared_intermediate, args
            )
            recommendations = {"shared_mlp_decode": shared_rec}
            print_recommendations(recommendations)
            payload = {
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": props.total_memory / 1024**3,
                },
                "dtype": args.dtype,
                "shape": {
                    "hidden_dim": hidden_dim,
                    "shared_intermediate": shared_intermediate,
                },
                "shared_mlp_decode": shared_rows,
                "recommendations": recommendations,
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"\nwrote {out_path}")
            return 0
        if args.only_parallel_moe_decode:
            parallel_rows, parallel_rec = benchmark_parallel_moe_decode(
                dtype,
                hidden_dim,
                shared_intermediate,
                expert_intermediate,
                num_experts,
                top_k,
                args,
            )
            recommendations = {"parallel_moe_decode": parallel_rec}
            print_recommendations(recommendations)
            payload = {
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": props.total_memory / 1024**3,
                },
                "dtype": args.dtype,
                "shape": {
                    "hidden_dim": hidden_dim,
                    "shared_intermediate": shared_intermediate,
                    "expert_intermediate": expert_intermediate,
                    "num_experts": num_experts,
                    "top_k": top_k,
                },
                "parallel_moe_decode": parallel_rows,
                "recommendations": recommendations,
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"\nwrote {out_path}")
            return 0
        if args.only_parallel_moe_prefill:
            parallel_rows, parallel_rec = benchmark_parallel_moe_prefill(
                dtype,
                hidden_dim,
                shared_intermediate,
                expert_intermediate,
                num_experts,
                top_k,
                args,
            )
            recommendations = {"parallel_moe_prefill": parallel_rec}
            print_recommendations(recommendations)
            payload = {
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": props.total_memory / 1024**3,
                },
                "dtype": args.dtype,
                "shape": {
                    "rows": args.prefill_rows,
                    "hidden_dim": hidden_dim,
                    "shared_intermediate": shared_intermediate,
                    "expert_intermediate": expert_intermediate,
                    "num_experts": num_experts,
                    "top_k": top_k,
                },
                "parallel_moe_prefill": parallel_rows,
                "recommendations": recommendations,
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"\nwrote {out_path}")
            return 0
        if args.only_expert_decode:
            decode_rows, decode_rec, _, _ = benchmark_expert_decode(
                dtype,
                hidden_dim,
                expert_intermediate,
                num_experts,
                top_k,
                args,
            )
            recommendations = {"expert_decode": decode_rec}
            print_recommendations(recommendations)
            decode_impact = estimate_model_impact(
                {"expert_decode": decode_rows},
                recommendations,
                layers=30,
            )
            print("DECODE_IMPACT", json.dumps(decode_impact, sort_keys=True))
            payload = {
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": props.total_memory / 1024**3,
                },
                "dtype": args.dtype,
                "shape": {
                    "hidden_dim": hidden_dim,
                    "expert_intermediate": expert_intermediate,
                    "num_experts": num_experts,
                    "top_k": top_k,
                },
                "expert_decode": decode_rows,
                "recommendations": recommendations,
                "decode_impact": decode_impact,
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nwrote {out_path}")
            return 0
        shared_rows, shared_rec = benchmark_shared_mlp(
            dtype, hidden_dim, shared_intermediate, args
        )
        router_rows, router_rec = benchmark_router(
            dtype, hidden_dim, num_experts, top_k, args
        )
        decode_rows, decode_rec, expert_gate_up, expert_down = benchmark_expert_decode(
            dtype,
            hidden_dim,
            expert_intermediate,
            num_experts,
            top_k,
            args,
        )
        prefill_rows, prefill_rec = benchmark_expert_prefill(
            dtype,
            hidden_dim,
            expert_intermediate,
            num_experts,
            top_k,
            expert_gate_up,
            expert_down,
            args,
        )

    recommendations = {
        "shared_mlp_decode": shared_rec,
        "router_decode": router_rec,
        "expert_decode": decode_rec,
        "expert_prefill": prefill_rec,
    }
    print_recommendations(recommendations)
    decode_impact = estimate_model_impact(
        {
            "shared_mlp_decode": shared_rows,
            "router_decode": router_rows,
            "expert_decode": decode_rows,
        },
        recommendations,
        layers=30,
    )
    prefill_impact = estimate_model_impact(
        {"expert_prefill": prefill_rows},
        recommendations,
        layers=30,
    )
    print("DECODE_IMPACT", json.dumps(decode_impact, sort_keys=True))
    print("PREFILL_IMPACT", json.dumps(prefill_impact, sort_keys=True))

    payload = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_gb": props.total_memory / 1024**3,
        },
        "dtype": args.dtype,
        "shape": {
            "hidden_dim": hidden_dim,
            "shared_intermediate": shared_intermediate,
            "expert_intermediate": expert_intermediate,
            "num_experts": num_experts,
            "top_k": top_k,
            "prefill_rows": 25,
        },
        "shared_mlp_decode": shared_rows,
        "router_decode": router_rows,
        "expert_decode": decode_rows,
        "expert_prefill": prefill_rows,
        "recommendations": recommendations,
        "decode_impact": decode_impact,
        "prefill_impact": prefill_impact,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
