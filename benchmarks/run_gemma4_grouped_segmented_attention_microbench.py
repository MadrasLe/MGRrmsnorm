#!/usr/bin/env python3
"""Gate the Gemma4 B16 grouped segmented attention port without a model."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROWS = 16
BLOCK_SIZE = 16
MIN_SPEEDUP = 1.02
MAX_BASELINE_STABILITY_RATIO = 1.03
MAX_ABS_ERROR = 0.05
MAX_RELATIVE_L2_ERROR = 0.02
MIN_COSINE = 0.999
REMAINING_DECODE_GAP_MS = 2.1


@dataclass(frozen=True)
class Topology:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    sliding_window: int | None
    prior_vllm_us: float


TOPOLOGIES = (
    Topology(
        name="sliding_h256_gqa2",
        q_heads=16,
        kv_heads=8,
        head_dim=256,
        layers=25,
        sliding_window=1024,
        prior_vllm_us=29.83936071395874,
    ),
    Topology(
        name="full_h512_gqa8",
        q_heads=16,
        kv_heads=2,
        head_dim=512,
        layers=5,
        sliding_window=None,
        prior_vllm_us=19.066879749298096,
    ),
)


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


def capture_case(
    *,
    name: str,
    call: Callable[[], torch.Tensor],
    out: torch.Tensor,
    reference: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        for _ in range(max(1, warmup)):
            call()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
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
            "measurement": "cuda_graph",
            "median_us": float(statistics.median(samples_us)),
            "samples_us": samples_us,
            "repeat_max_abs_error": float(
                (second.float() - first.float()).abs().max().item()
            ),
            "error": None,
        }
        row.update(output_error(first, reference))
        del graph
        return row
    except Exception as exc:
        return {
            "case": name,
            "measurement": "cuda_graph",
            "median_us": None,
            "samples_us": [],
            "repeat_max_abs_error": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def preprocess_query(
    raw_query: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = raw_query.float()
    variance = query.square().mean(dim=-1, keepdim=True)
    query = query * torch.rsqrt(variance + norm_eps)
    query = query * norm_weight.float().view(1, 1, -1)

    half = query.shape[-1] // 2
    cos_pos = cos[positions.long()].float().unsqueeze(1)
    sin_pos = sin[positions.long()].float().unsqueeze(1)
    first = query[..., :half]
    second = query[..., half:]
    rotated = torch.cat(
        (
            first * cos_pos - second * sin_pos,
            second * cos_pos + first * sin_pos,
        ),
        dim=-1,
    )
    return rotated, rotated.to(raw_query.dtype)


def torch_reference(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    scale: float,
    sliding_window: int | None,
) -> torch.Tensor:
    rows, q_heads, _ = query.shape
    kv_heads = kv_cache.shape[2]
    blocks_used = math.ceil(context / BLOCK_SIZE)
    gqa = q_heads // kv_heads
    visible_start = (
        max(0, context - sliding_window)
        if sliding_window is not None
        else 0
    )
    outputs: list[torch.Tensor] = []
    for row in range(rows):
        physical = block_table[row, :blocks_used].long()
        keys = kv_cache[physical, 0].permute(1, 0, 2, 3)
        values = kv_cache[physical, 1].permute(1, 0, 2, 3)
        keys = keys.reshape(kv_heads, -1, query.shape[-1])
        values = values.reshape(kv_heads, -1, query.shape[-1])
        keys = keys[:, visible_start:context]
        values = values[:, visible_start:context]
        keys = keys.repeat_interleave(gqa, dim=0)
        values = values.repeat_interleave(gqa, dim=0)
        scores = torch.einsum(
            "hd,htd->ht",
            query[row].float(),
            keys.float(),
        )
        probs = torch.softmax(scores * scale, dim=-1)
        outputs.append(
            torch.einsum("ht,htd->hd", probs, values.float())
        )
    return torch.stack(outputs).to(torch.bfloat16)


def make_inputs(
    topology: Topology,
    *,
    context: int,
    table_blocks: int,
    norm_eps: float,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(
        20260729 + topology.head_dim + topology.kv_heads
    )
    dtype = torch.bfloat16
    num_blocks = ROWS * table_blocks
    raw_query = torch.empty(
        (ROWS, topology.q_heads, topology.head_dim),
        device="cuda",
        dtype=dtype,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    kv_cache = torch.empty(
        (
            num_blocks,
            2,
            topology.kv_heads,
            BLOCK_SIZE,
            topology.head_dim,
        ),
        device="cuda",
        dtype=dtype,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    norm_weight = torch.empty(
        (topology.head_dim,),
        device="cuda",
        dtype=dtype,
    ).normal_(mean=1.0, std=0.05, generator=generator)
    positions = torch.full(
        (ROWS,),
        context - 1,
        device="cuda",
        dtype=torch.int32,
    )
    half = topology.head_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, half, device="cuda", dtype=torch.float32)
            / half
        )
    )
    all_positions = torch.arange(
        context + 1,
        device="cuda",
        dtype=torch.float32,
    )
    angles = all_positions[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    query_fp32, query_bf16 = preprocess_query(
        raw_query,
        norm_weight,
        cos,
        sin,
        positions,
        norm_eps,
    )
    block_table = torch.arange(
        num_blocks,
        device="cuda",
        dtype=torch.int32,
    ).reshape(ROWS, table_blocks)
    seq_lens = torch.full(
        (ROWS,),
        context,
        device="cuda",
        dtype=torch.int32,
    )
    return {
        "raw_query": raw_query,
        "query_fp32": query_fp32,
        "query_bf16": query_bf16,
        "kv_cache": kv_cache,
        "norm_weight": norm_weight,
        "cos": cos,
        "sin": sin,
        "positions": positions,
        "block_table": block_table,
        "seq_lens": seq_lens,
    }


@torch.inference_mode()
def run_topology(
    topology: Topology,
    *,
    context: int,
    table_blocks: int,
    norm_eps: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode,
        _triton_paged_decode_fused,
        _triton_paged_decode_grouped_segmented,
        _triton_paged_decode_grouped_segmented_fused,
    )

    tensors = make_inputs(
        topology,
        context=context,
        table_blocks=table_blocks,
        norm_eps=norm_eps,
    )
    raw_query = tensors["raw_query"]
    query_fp32 = tensors["query_fp32"]
    query_bf16 = tensors["query_bf16"]
    kv_cache = tensors["kv_cache"]
    norm_weight = tensors["norm_weight"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    positions = tensors["positions"]
    block_table = tensors["block_table"]
    seq_lens = tensors["seq_lens"]
    scale = 1.0

    core_reference = torch_reference(
        query=query_bf16,
        kv_cache=kv_cache,
        block_table=block_table,
        context=context,
        scale=scale,
        sliding_window=topology.sliding_window,
    )
    fused_reference = torch_reference(
        query=query_fp32,
        kv_cache=kv_cache,
        block_table=block_table,
        context=context,
        scale=scale,
        sliding_window=topology.sliding_window,
    )

    current_core_out = torch.empty_like(query_bf16)

    def current_core_call() -> torch.Tensor:
        return _triton_paged_decode(
            query_bf16,
            kv_cache,
            block_table,
            seq_lens,
            scale,
            out=current_core_out,
            sliding_window=topology.sliding_window,
        )

    candidate_core_out = torch.empty_like(query_bf16)

    def candidate_core_call() -> torch.Tensor:
        return _triton_paged_decode_grouped_segmented(
            query_bf16,
            kv_cache,
            block_table,
            seq_lens,
            scale,
            out=candidate_core_out,
            sliding_window=topology.sliding_window,
            force=True,
        )

    current_fused_out = torch.empty_like(raw_query)

    def current_fused_call() -> torch.Tensor:
        return _triton_paged_decode_fused(
            raw_query,
            kv_cache,
            block_table,
            seq_lens,
            scale,
            cos,
            sin,
            positions,
            half_rotate=True,
            rotary_dim=topology.head_dim,
            q_norm_weight=norm_weight,
            norm_eps=norm_eps,
            out=current_fused_out,
            sliding_window=topology.sliding_window,
        )

    candidate_fused_out = torch.empty_like(raw_query)

    def candidate_fused_call() -> torch.Tensor:
        return _triton_paged_decode_grouped_segmented_fused(
            raw_query,
            kv_cache,
            block_table,
            seq_lens,
            scale,
            cos,
            sin,
            positions,
            half_rotate=True,
            rotary_dim=topology.head_dim,
            q_norm_weight=norm_weight,
            norm_eps=norm_eps,
            out=candidate_fused_out,
            sliding_window=topology.sliding_window,
            force=True,
        )

    cases = [
        capture_case(
            name="current_core",
            call=current_core_call,
            out=current_core_out,
            reference=core_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        capture_case(
            name="grouped_segmented_core",
            call=candidate_core_call,
            out=candidate_core_out,
            reference=core_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        capture_case(
            name="current_fused",
            call=current_fused_call,
            out=current_fused_out,
            reference=fused_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        capture_case(
            name="current_fused_recheck",
            call=current_fused_call,
            out=current_fused_out,
            reference=fused_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        capture_case(
            name="grouped_segmented_fused",
            call=candidate_fused_call,
            out=candidate_fused_out,
            reference=fused_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    ]
    by_name = {str(row["case"]): row for row in cases}
    required = (
        "current_core",
        "grouped_segmented_core",
        "current_fused",
        "current_fused_recheck",
        "grouped_segmented_fused",
    )
    complete = all(by_name[name].get("error") is None for name in required)
    if complete:
        current_fused_us = float(by_name["current_fused"]["median_us"])
        recheck_us = float(
            by_name["current_fused_recheck"]["median_us"]
        )
        baseline_us = min(current_fused_us, recheck_us)
        stability = max(current_fused_us, recheck_us) / baseline_us
        candidate_us = float(
            by_name["grouped_segmented_fused"]["median_us"]
        )
        core_speedup = (
            float(by_name["current_core"]["median_us"])
            / float(by_name["grouped_segmented_core"]["median_us"])
        )
        fused_speedup = baseline_us / candidate_us
        numeric_valid = all(
            float(by_name[name]["repeat_max_abs_error"]) == 0.0
            and float(by_name[name]["max_abs_error"]) <= MAX_ABS_ERROR
            and float(by_name[name]["relative_l2_error"])
            <= MAX_RELATIVE_L2_ERROR
            and float(by_name[name]["cosine"]) >= MIN_COSINE
            for name in required
        )
        valid = bool(
            numeric_valid
            and stability <= MAX_BASELINE_STABILITY_RATIO
        )
    else:
        baseline_us = None
        stability = None
        candidate_us = None
        core_speedup = None
        fused_speedup = None
        valid = False

    return {
        "topology": topology.name,
        "valid": valid,
        "shape": {
            "rows": ROWS,
            "q_heads": topology.q_heads,
            "kv_heads": topology.kv_heads,
            "head_dim": topology.head_dim,
            "layers": topology.layers,
            "context": context,
            "table_blocks": table_blocks,
            "block_size": BLOCK_SIZE,
            "sliding_window": topology.sliding_window,
        },
        "baseline_fused_us": baseline_us,
        "candidate_fused_us": candidate_us,
        "baseline_stability_ratio": stability,
        "core_speedup": core_speedup,
        "fused_speedup": fused_speedup,
        "prior_vllm_us": topology.prior_vllm_us,
        "candidate_vs_prior_vllm": (
            candidate_us / topology.prior_vllm_us
            if candidate_us is not None
            else None
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--table-blocks", type=int, default=6)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--out-json",
        default=(
            "bench_results/"
            "gemma4_grouped_segmented_attention_a100.json"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if "a100" not in torch.cuda.get_device_name(0).lower():
        raise SystemExit("This paid gate is intentionally restricted to A100")
    if args.context <= 0:
        raise SystemExit("context must be positive")
    if math.ceil(args.context / BLOCK_SIZE) > args.table_blocks:
        raise SystemExit(
            "table-blocks is too small for the requested context"
        )
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    # Preserve the exact paid v81 scalar baseline during this gate.
    os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"] = "0"
    os.environ["MEGAGEMM_PAGED_DECODE_GQA2"] = "0"
    os.environ["MEGAGEMM_PAGED_DECODE_WARPS"] = "0"
    os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H256"] = "8"
    os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H512"] = "4"

    print("Gemma4 B16 grouped segmented attention gate")
    print("  harness_rev: gemma4-grouped-segmented-attention-v1")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(
        "  workload: "
        f"rows={ROWS} context={args.context} "
        f"table_blocks={args.table_blocks} block_size={BLOCK_SIZE}"
    )
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  measurement: CUDA graph, core and fused runtime path")

    results: list[dict[str, Any]] = []
    for topology in TOPOLOGIES:
        result = run_topology(
            topology,
            context=args.context,
            table_blocks=args.table_blocks,
            norm_eps=args.norm_eps,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    valid = all(bool(result["valid"]) for result in results)
    if valid:
        baseline_us_per_step = sum(
            float(result["baseline_fused_us"])
            * int(result["shape"]["layers"])
            for result in results
        )
        candidate_us_per_step = sum(
            float(result["candidate_fused_us"])
            * int(result["shape"]["layers"])
            for result in results
        )
        weighted_speedup = baseline_us_per_step / candidate_us_per_step
        savings_ms = (
            baseline_us_per_step - candidate_us_per_step
        ) / 1000.0
    else:
        baseline_us_per_step = None
        candidate_us_per_step = None
        weighted_speedup = None
        savings_ms = None

    apply_change = bool(
        valid
        and weighted_speedup is not None
        and weighted_speedup >= args.minimum_speedup
    )
    decision = {
        "decision": (
            "APPLY_GROUPED_SEGMENTED_DEFAULT"
            if apply_change
            else "KEEP_SCALAR_BASELINE"
        ),
        "apply_change": apply_change,
        "valid": valid,
        "model_download": False,
        "vllm_install": False,
        "minimum_speedup": float(args.minimum_speedup),
        "maximum_baseline_stability_ratio": (
            MAX_BASELINE_STABILITY_RATIO
        ),
        "maximum_abs_error": MAX_ABS_ERROR,
        "maximum_relative_l2_error": MAX_RELATIVE_L2_ERROR,
        "minimum_cosine": MIN_COSINE,
        "baseline_us_per_decode_step_30_layers": baseline_us_per_step,
        "candidate_us_per_decode_step_30_layers": candidate_us_per_step,
        "weighted_speedup": weighted_speedup,
        "estimated_savings_ms_per_decode_step": savings_ms,
        "estimated_remaining_gap_coverage": (
            savings_ms / REMAINING_DECODE_GAP_MS
            if savings_ms is not None
            else None
        ),
        "remaining_decode_gap_ms": REMAINING_DECODE_GAP_MS,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "topologies": results,
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
