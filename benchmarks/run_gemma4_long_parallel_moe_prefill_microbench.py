#!/usr/bin/env python3
"""Gate parallel Gemma4 shared+routed MoE at the real long-context chunk."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from megagemm.kernels.qwen3_moe import qwen3_moe_segmented_prefill


TensorFn = Callable[[], torch.Tensor]


def _measure_us(
    fn: TensorFn,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
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
            "error": None,
            "median_us": statistics.median(samples),
            "samples_us": samples,
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "median_us": None,
            "samples_us": [],
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
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_parallel_moe_prefill_a100.json",
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

    torch.manual_seed(20260809)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = 16_384
    hidden_dim = 2_816
    shared_intermediate = 2_112
    expert_intermediate = 704
    num_experts = 128
    top_k = 8
    layers = 30
    chunks = 2
    target_gap_ms = 814.46

    print("Gemma4 long parallel-MoE prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"rows={rows} H={hidden_dim} shared_I={shared_intermediate} "
        f"expert_I={expert_intermediate} E={num_experts} top_k={top_k} "
        "dtype=bf16"
    )
    print(f"  B16 estimate: {chunks} chunks x {layers} layers")

    def random_weight(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 0.02)

    shared_in = random_weight((rows, hidden_dim))
    expert_in = random_weight((rows, hidden_dim))
    shared_gate_up = random_weight((2 * shared_intermediate, hidden_dim))
    shared_down = random_weight((hidden_dim, shared_intermediate))
    expert_gate_up = random_weight(
        (num_experts, 2 * expert_intermediate, hidden_dim)
    )
    expert_down = random_weight(
        (num_experts, hidden_dim, expert_intermediate)
    )

    row_ids = torch.arange(rows, device=device, dtype=torch.int64).reshape(-1, 1)
    top_ids = torch.arange(top_k, device=device, dtype=torch.int64).reshape(1, -1)
    selected = (row_ids * 17 + top_ids * 13).remainder(num_experts).contiguous()
    routing = torch.rand((rows, top_k), device=device, dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))

    shared_gate_up_out = torch.empty(
        (rows, 2 * shared_intermediate), device=device, dtype=dtype
    )
    shared_out = torch.empty((rows, hidden_dim), device=device, dtype=dtype)
    expert_out = torch.empty_like(shared_out)
    combined_out = torch.empty_like(shared_out)
    expert_workspace: dict[str, torch.Tensor] = {}

    def shared_branch() -> torch.Tensor:
        torch.mm(shared_in, shared_gate_up.t(), out=shared_gate_up_out)
        gate = shared_gate_up_out[:, :shared_intermediate]
        value = shared_gate_up_out[:, shared_intermediate:]
        activated = F.gelu(gate, approximate="tanh").mul_(value)
        torch.mm(activated, shared_down.t(), out=shared_out)
        return shared_out

    def expert_branch() -> torch.Tensor:
        return qwen3_moe_segmented_prefill(
            expert_in,
            expert_gate_up,
            expert_down,
            selected,
            routing,
            activation="gelu_pytorch_tanh",
            out=expert_out,
            workspace=expert_workspace,
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
            compact_route_pack=False,
            async_tiles_max_assignments=4_096,
            deterministic_reduce=True,
        )

    def sequential() -> torch.Tensor:
        shared = shared_branch()
        expert = expert_branch()
        torch.add(shared, expert, out=combined_out)
        return combined_out

    side_stream = torch.cuda.Stream(device=device)
    fork_event = torch.cuda.Event()
    join_event = torch.cuda.Event()

    def parallel_two_streams() -> torch.Tensor:
        main_stream = torch.cuda.current_stream(device)
        fork_event.record(main_stream)
        with torch.cuda.stream(side_stream):
            side_stream.wait_event(fork_event)
            shared = shared_branch()
            join_event.record(side_stream)
        expert = expert_branch()
        main_stream.wait_event(join_event)
        torch.add(shared, expert, out=combined_out)
        return combined_out

    reference = sequential().detach().clone()
    candidate = parallel_two_streams().detach().clone()
    repeated = parallel_two_streams().detach().clone()
    torch.cuda.synchronize()
    difference = (reference.float() - candidate.float()).abs()
    correctness = {
        "finite": bool(torch.isfinite(candidate).all().item()),
        "cosine": _cosine(reference, candidate),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "repeat_exact": bool(torch.equal(candidate, repeated)),
        "repeat_max_abs_error": float(
            (candidate.float() - repeated.float()).abs().max().item()
        ),
    }
    correctness["correct"] = bool(
        correctness["finite"]
        and correctness["repeat_exact"]
        and correctness["cosine"] >= 0.9999
        and correctness["max_abs_error"] <= 0.125
    )
    del candidate, repeated, difference

    cases = (
        ("sequential", sequential),
        ("parallel_two_streams", parallel_two_streams),
        ("shared_only", shared_branch),
        ("routed_experts_only", expert_branch),
        ("sequential_recheck", sequential),
    )
    measured: list[dict[str, Any]] = []
    for name, fn in cases:
        row = {
            "case": name,
            **_measure_us(
                fn,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            ),
        }
        if name == "parallel_two_streams":
            row.update(correctness)
        measured.append(row)
        print(json.dumps(row, sort_keys=True))

    by_name = {row["case"]: row for row in measured}
    baseline_values = [
        float(by_name[name]["median_us"])
        for name in ("sequential", "sequential_recheck")
        if by_name[name]["median_us"] is not None
    ]
    baseline_us = min(baseline_values) if len(baseline_values) == 2 else None
    baseline_stability_ratio = (
        max(baseline_values) / baseline_us
        if baseline_us is not None and baseline_us > 0.0
        else None
    )
    candidate_us = by_name["parallel_two_streams"]["median_us"]
    speedup = (
        baseline_us / float(candidate_us)
        if baseline_us is not None and candidate_us is not None
        else None
    )
    stable = bool(
        baseline_stability_ratio is not None
        and baseline_stability_ratio <= 1.03
    )
    apply_change = bool(
        correctness["correct"]
        and stable
        and speedup is not None
        and speedup >= float(args.minimum_speedup)
    )
    layer_invocations = layers * chunks
    estimated_baseline_ms = (
        baseline_us * layer_invocations / 1000.0
        if baseline_us is not None
        else None
    )
    estimated_candidate_ms = (
        float(candidate_us) * layer_invocations / 1000.0
        if candidate_us is not None
        else None
    )
    estimated_savings_ms = (
        estimated_baseline_ms - estimated_candidate_ms
        if estimated_baseline_ms is not None
        and estimated_candidate_ms is not None
        else None
    )
    shared_us = by_name["shared_only"]["median_us"]
    expert_us = by_name["routed_experts_only"]["median_us"]
    component_sum_us = (
        float(shared_us) + float(expert_us)
        if shared_us is not None and expert_us is not None
        else None
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_SEQUENTIAL",
        "apply_change": apply_change,
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "rows": rows,
            "hidden_dim": hidden_dim,
            "shared_intermediate": shared_intermediate,
            "expert_intermediate": expert_intermediate,
            "num_experts": num_experts,
            "top_k": top_k,
            "dtype": "bf16",
        },
        "deterministic_reduce": True,
        "layers": layers,
        "chunks": chunks,
        "layer_invocations": layer_invocations,
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "baseline_stability_ratio": baseline_stability_ratio,
        "stable": stable,
        "minimum_speedup": float(args.minimum_speedup),
        "speedup": speedup,
        "correctness": correctness,
        "shared_us": shared_us,
        "routed_experts_us": expert_us,
        "component_sum_us": component_sum_us,
        "component_accounting_ratio": (
            component_sum_us / baseline_us
            if component_sum_us is not None
            and baseline_us is not None
            and baseline_us > 0.0
            else None
        ),
        "estimated_baseline_ms_b16_prefill": estimated_baseline_ms,
        "estimated_candidate_ms_b16_prefill": estimated_candidate_ms,
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
