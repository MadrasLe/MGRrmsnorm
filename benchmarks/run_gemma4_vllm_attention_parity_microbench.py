#!/usr/bin/env python3
"""Compare MegaGemm and vLLM Gemma4 B16 attention cores without a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROWS = 16
BLOCK_SIZE = 16
NUM_VLLM_SEGMENTS = 16
MAX_BASELINE_STABILITY_RATIO = 1.03
MAX_ABS_ERROR = 0.05
MAX_RELATIVE_L2_ERROR = 0.02
MIN_COSINE = 0.999
SHORT_REMAINING_DECODE_GAP_MS = 2.1
LONG_REMAINING_DECODE_GAP_MS = 120.58
DECODE_TOKENS = 64


@dataclass(frozen=True)
class Topology:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    sliding_window: int | None
    vllm_seq_threshold_3d: int
    megagemm_segments: int
    megagemm_tile_size: int


TOPOLOGIES = (
    Topology(
        name="sliding_h256_gqa2",
        q_heads=16,
        kv_heads=8,
        head_dim=256,
        layers=25,
        sliding_window=1024,
        vllm_seq_threshold_3d=16,
        megagemm_segments=32,
        megagemm_tile_size=64,
    ),
    Topology(
        name="full_h512_gqa8",
        q_heads=16,
        kv_heads=2,
        head_dim=512,
        layers=5,
        sliding_window=None,
        vllm_seq_threshold_3d=32,
        megagemm_segments=8,
        megagemm_tile_size=16,
    ),
)


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


def capture_case(
    *,
    name: str,
    backend: str,
    call: Callable[[], torch.Tensor],
    out: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
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
        "backend": backend,
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


def torch_reference(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    scale: float,
    sliding_window: int | None,
) -> torch.Tensor:
    rows, q_heads, _ = query.shape
    kv_heads = key_cache.shape[2]
    blocks_used = math.ceil(context / BLOCK_SIZE)
    gqa = q_heads // kv_heads
    outputs: list[torch.Tensor] = []
    for row in range(rows):
        physical = block_table[row, :blocks_used].long()
        keys = key_cache[physical].reshape(
            -1,
            kv_heads,
            query.shape[-1],
        )[:context]
        values = value_cache[physical].reshape(
            -1,
            kv_heads,
            query.shape[-1],
        )[:context]
        if sliding_window is not None:
            keys = keys[-sliding_window:]
            values = values[-sliding_window:]
        keys = keys.permute(1, 0, 2).repeat_interleave(gqa, dim=0)
        values = values.permute(1, 0, 2).repeat_interleave(gqa, dim=0)
        scores = torch.einsum(
            "hd,htd->ht",
            query[row].float(),
            keys.float(),
        )
        probs = torch.softmax(scores * scale, dim=-1)
        outputs.append(
            torch.einsum("ht,htd->hd", probs, values.float())
        )
    return torch.stack(outputs).to(query.dtype)


def make_inputs(
    topology: Topology,
    *,
    context: int,
    table_blocks: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(
        20260728 + topology.head_dim + topology.kv_heads
    )
    dtype = torch.bfloat16
    num_blocks = ROWS * table_blocks
    query = torch.empty(
        (ROWS, topology.q_heads, topology.head_dim),
        device="cuda",
        dtype=dtype,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    vllm_cache = torch.empty(
        (
            num_blocks,
            2,
            BLOCK_SIZE,
            topology.kv_heads,
            topology.head_dim,
        ),
        device="cuda",
        dtype=dtype,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    megagemm_cache = vllm_cache.permute(0, 1, 3, 2, 4).contiguous()
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
    query_start = torch.arange(
        ROWS + 1,
        device="cuda",
        dtype=torch.int32,
    )
    return {
        "query": query,
        "vllm_cache": vllm_cache,
        "megagemm_cache": megagemm_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "query_start": query_start,
    }


@torch.inference_mode()
def run_topology(
    topology: Topology,
    *,
    context: int,
    table_blocks: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode_grouped_segmented,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )

    tensors = make_inputs(
        topology,
        context=context,
        table_blocks=table_blocks,
    )
    query = tensors["query"]
    vllm_cache = tensors["vllm_cache"]
    megagemm_cache = tensors["megagemm_cache"]
    block_table = tensors["block_table"]
    seq_lens = tensors["seq_lens"]
    query_start = tensors["query_start"]
    # Gemma4 uses Q/K RMSNorm and an attention scale of exactly 1.0.
    scale = 1.0
    reference = torch_reference(
        query=query,
        key_cache=vllm_cache[:, 0],
        value_cache=vllm_cache[:, 1],
        block_table=block_table,
        context=context,
        scale=scale,
        sliding_window=topology.sliding_window,
    )

    mega_out = torch.empty_like(query)

    def mega_call() -> torch.Tensor:
        return _triton_paged_decode_grouped_segmented(
            query,
            megagemm_cache,
            block_table,
            seq_lens,
            scale,
            out=mega_out,
            sliding_window=topology.sliding_window,
            force=True,
            num_segments_override=topology.megagemm_segments,
            tile_size_override=topology.megagemm_tile_size,
        )

    current, current_out = capture_case(
        name="megagemm_current",
        backend="megagemm",
        call=mega_call,
        out=mega_out,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    current.update(output_error(current_out, reference))

    recheck, recheck_out = capture_case(
        name="megagemm_current_recheck",
        backend="megagemm",
        call=mega_call,
        out=mega_out,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    recheck.update(output_error(recheck_out, reference))

    vllm_out = torch.empty_like(query)
    segments = NUM_VLLM_SEGMENTS
    threshold = topology.vllm_seq_threshold_3d
    segment_out = torch.empty(
        (
            threshold,
            topology.q_heads,
            segments,
            topology.head_dim,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    segment_max = torch.empty(
        (threshold, topology.q_heads, segments),
        device="cuda",
        dtype=torch.float32,
    )
    segment_sum = torch.empty_like(segment_max)
    window = (
        (topology.sliding_window - 1, 0)
        if topology.sliding_window is not None
        else (-1, -1)
    )

    def vllm_call() -> torch.Tensor:
        unified_attention(
            q=query,
            k=vllm_cache[:, 0],
            v=vllm_cache[:, 1],
            out=vllm_out,
            cu_seqlens_q=query_start,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=context,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=threshold,
            num_par_softmax_segments=segments,
            softmax_segm_output=segment_out,
            softmax_segm_max=segment_max,
            softmax_segm_expsum=segment_sum,
        )
        return vllm_out

    vllm_row, vllm_result = capture_case(
        name="vllm_unified_attention_3d",
        backend="vllm",
        call=vllm_call,
        out=vllm_out,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    vllm_row.update(output_error(vllm_result, reference))

    cases = [current, recheck, vllm_row]
    baseline_us = min(
        float(current["median_us"]),
        float(recheck["median_us"]),
    )
    stability = max(
        float(current["median_us"]),
        float(recheck["median_us"]),
    ) / baseline_us
    vllm_us = float(vllm_row["median_us"])
    valid = bool(
        current["repeat_max_abs_error"] == 0.0
        and recheck["repeat_max_abs_error"] == 0.0
        and vllm_row["repeat_max_abs_error"] == 0.0
        and stability <= MAX_BASELINE_STABILITY_RATIO
        and float(current["max_abs_error"]) <= MAX_ABS_ERROR
        and float(recheck["max_abs_error"]) <= MAX_ABS_ERROR
        and float(vllm_row["max_abs_error"]) <= MAX_ABS_ERROR
        and float(current["relative_l2_error"]) <= MAX_RELATIVE_L2_ERROR
        and float(recheck["relative_l2_error"]) <= MAX_RELATIVE_L2_ERROR
        and float(vllm_row["relative_l2_error"]) <= MAX_RELATIVE_L2_ERROR
        and float(current["cosine"]) >= MIN_COSINE
        and float(recheck["cosine"]) >= MIN_COSINE
        and float(vllm_row["cosine"]) >= MIN_COSINE
    )
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
            "megagemm_segments": topology.megagemm_segments,
            "megagemm_tile_size": topology.megagemm_tile_size,
            "vllm_segments": NUM_VLLM_SEGMENTS,
        },
        "baseline_us": baseline_us,
        "vllm_us": vllm_us,
        "baseline_stability_ratio": stability,
        "vllm_speedup_vs_megagemm": baseline_us / vllm_us,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=2111)
    parser.add_argument("--table-blocks", type=int, default=132)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_vllm_attention_parity_gate.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if args.context <= 0:
        raise SystemExit("context must be positive")
    if math.ceil(args.context / BLOCK_SIZE) > args.table_blocks:
        raise SystemExit(
            "table-blocks is too small for the requested context"
        )
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    try:
        import vllm
    except Exception as exc:
        raise SystemExit(
            "vLLM must be installed before this no-checkpoint parity gate: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("Gemma4 B16 MegaGemm-vLLM long attention-core parity gate")
    print("  harness_rev: gemma4-vllm-attention-parity-v2-long-current")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(f"  vllm: {vllm.__version__}")
    print(
        "  workload: "
        f"rows={ROWS} context={args.context} "
        f"table_blocks={args.table_blocks} block_size={BLOCK_SIZE}"
    )
    print("  model_download: disabled")
    print("  comparison: same logical Q/K/V, CUDA graph vs CUDA graph")
    print("  MegaGemm: paid grouped-segmented production baseline")
    print("  vLLM: runtime 3D Triton core, 16 softmax segments")

    results: list[dict[str, Any]] = []
    for topology in TOPOLOGIES:
        try:
            result = run_topology(
                topology,
                context=args.context,
                table_blocks=args.table_blocks,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
        except Exception as exc:
            result = {
                "topology": topology.name,
                "valid": False,
                "shape": {
                    "rows": ROWS,
                    "q_heads": topology.q_heads,
                    "kv_heads": topology.kv_heads,
                    "head_dim": topology.head_dim,
                    "layers": topology.layers,
                    "context": args.context,
                    "table_blocks": args.table_blocks,
                    "block_size": BLOCK_SIZE,
                    "sliding_window": topology.sliding_window,
                    "megagemm_segments": topology.megagemm_segments,
                    "megagemm_tile_size": topology.megagemm_tile_size,
                    "vllm_segments": NUM_VLLM_SEGMENTS,
                },
                "baseline_us": None,
                "vllm_us": None,
                "baseline_stability_ratio": None,
                "vllm_speedup_vs_megagemm": None,
                "cases": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    valid = all(bool(result.get("valid")) for result in results)
    megagemm_us_per_step = (
        sum(
            float(result["baseline_us"]) * int(result["shape"]["layers"])
            for result in results
        )
        if valid
        else None
    )
    vllm_us_per_step = (
        sum(
            float(result["vllm_us"]) * int(result["shape"]["layers"])
            for result in results
        )
        if valid
        else None
    )
    shape_selective_us_per_step = (
        sum(
            min(float(result["baseline_us"]), float(result["vllm_us"]))
            * int(result["shape"]["layers"])
            for result in results
        )
        if valid
        else None
    )
    full_vllm_speedup = (
        megagemm_us_per_step / vllm_us_per_step
        if megagemm_us_per_step is not None
        and vllm_us_per_step is not None
        else None
    )
    selective_speedup = (
        megagemm_us_per_step / shape_selective_us_per_step
        if megagemm_us_per_step is not None
        and shape_selective_us_per_step is not None
        else None
    )
    port_shapes = [
        str(result["topology"])
        for result in results
        if result.get("valid")
        and float(result["vllm_speedup_vs_megagemm"])
        >= args.minimum_speedup
    ]
    savings_ms_per_step = (
        (megagemm_us_per_step - shape_selective_us_per_step) / 1000.0
        if megagemm_us_per_step is not None
        and shape_selective_us_per_step is not None
        else None
    )
    savings_ms_per_64_tokens = (
        savings_ms_per_step * DECODE_TOKENS
        if savings_ms_per_step is not None
        else None
    )
    remaining_decode_gap_ms = (
        LONG_REMAINING_DECODE_GAP_MS
        if args.context >= 1024
        else SHORT_REMAINING_DECODE_GAP_MS
    )
    if not valid:
        decision_name = "INVALID_ATTENTION_PARITY_GATE"
    elif (
        selective_speedup is not None
        and selective_speedup >= args.minimum_speedup
        and port_shapes
    ):
        decision_name = (
            "PORT_VLLM_LONG_ATTENTION_CORE"
            if args.context >= 1024
            else "PORT_VLLM_ATTENTION_CORE"
        )
    else:
        decision_name = (
            "MOVE_OFF_LONG_ATTENTION_CORE"
            if args.context >= 1024
            else "MOVE_OFF_ATTENTION_CORE"
        )

    decision = {
        "decision": decision_name,
        "valid": valid,
        "model_download": False,
        "minimum_port_speedup": float(args.minimum_speedup),
        "maximum_baseline_stability_ratio": MAX_BASELINE_STABILITY_RATIO,
        "maximum_abs_error": MAX_ABS_ERROR,
        "maximum_relative_l2_error": MAX_RELATIVE_L2_ERROR,
        "minimum_cosine": MIN_COSINE,
        "megagemm_us_per_decode_step_30_layers": megagemm_us_per_step,
        "vllm_us_per_decode_step_30_layers": vllm_us_per_step,
        "shape_selective_us_per_decode_step_30_layers": (
            shape_selective_us_per_step
        ),
        "vllm_full_speedup_vs_megagemm": full_vllm_speedup,
        "shape_selective_speedup_vs_megagemm": selective_speedup,
        "port_shapes": port_shapes,
        "decode_tokens": DECODE_TOKENS,
        "estimated_savings_ms_per_decode_step": savings_ms_per_step,
        "estimated_savings_ms_per_64_tokens": savings_ms_per_64_tokens,
        "estimated_remaining_gap_coverage": (
            savings_ms_per_64_tokens / remaining_decode_gap_ms
            if savings_ms_per_64_tokens is not None
            else None
        ),
        "remaining_decode_gap_ms": remaining_decode_gap_ms,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
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
