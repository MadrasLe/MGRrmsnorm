#!/usr/bin/env python3
"""CUDA-graph microbenchmark for long-context paged decode attention."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megagemm.kernels.paged_attention import _triton_paged_decode_fused


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(max(1, int(item)))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def capture_case(
    *,
    splits: int,
    gqa_group: int,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    scale: float,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[torch.Tensor, list[float]]:
    os.environ["MEGAGEMM_PAGED_DECODE_SPLITS"] = str(int(splits))
    os.environ["MEGAGEMM_PAGED_DECODE_GQA4_SPLIT"] = "1"
    os.environ["MEGAGEMM_PAGED_DECODE_GQA8_SPLIT"] = "1" if int(gqa_group) == 8 else "0"
    os.environ["MEGAGEMM_PAGED_DECODE_LOG"] = "0"

    out = torch.empty_like(query)

    def call() -> None:
        _triton_paged_decode_fused(
            query,
            kv_cache,
            block_table,
            seq_lens,
            scale,
            cos,
            sin,
            positions,
            half_rotate=True,
            rotary_dim=int(query.shape[-1]),
            q_norm_weight=q_norm_weight,
            norm_eps=1e-6,
            out=out,
            max_blocks_override=int(block_table.shape[1]),
        )

    for _ in range(max(1, warmup)):
        call()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    result = out.clone()

    samples_us = []
    for _ in range(max(1, repeats)):
        for _ in range(max(1, warmup)):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, iterations)):
            graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(
            float(start.elapsed_time(end)) * 1000.0 / max(1, iterations)
        )
    return result, samples_us


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--context-tokens", type=int, default=2176)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--splits", type=parse_int_list, default=parse_int_list("16,24,32,40,48,64"))
    parser.add_argument("--gqa-groups", type=parse_int_list, default=parse_int_list("4,8"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out-json", default="bench_results/paged_attention_graph_microbench.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.q_heads // args.kv_heads != 8:
        raise SystemExit("this benchmark currently targets GQA=8")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    table_blocks = (args.context_tokens + args.block_size - 1) // args.block_size
    query = torch.randn(
        1,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache = torch.randn(
        table_blocks,
        2,
        args.kv_heads,
        args.block_size,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    block_table = torch.arange(
        table_blocks,
        device=device,
        dtype=torch.int32,
    ).reshape(1, table_blocks)
    seq_lens = torch.tensor(
        [args.context_tokens],
        device=device,
        dtype=torch.int32,
    )
    positions = torch.tensor(
        [args.context_tokens - 1],
        device=device,
        dtype=torch.long,
    )
    half_dim = args.head_dim // 2
    inv_freq = 1.0 / (
        1_000_000.0
        ** (
            torch.arange(0, half_dim, device=device, dtype=torch.float32)
            / half_dim
        )
    )
    freqs = torch.outer(
        torch.arange(args.context_tokens + 1, device=device, dtype=torch.float32),
        inv_freq,
    )
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    q_norm_weight = torch.ones(args.head_dim, device=device, dtype=dtype)
    scale = args.head_dim ** -0.5

    print("Paged attention CUDA-graph microbenchmark")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(
        "  shape: "
        f"seqs=1 q_heads={args.q_heads} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} context={args.context_tokens} "
        f"blocks={table_blocks} block_size={args.block_size} dtype={args.dtype}"
    )
    print(f"  splits: {','.join(str(value) for value in args.splits)}")
    print(f"  gqa_groups: {','.join(str(value) for value in args.gqa_groups)}")
    print("  measurement: cuda_graph")

    reference = None
    rows = []
    for gqa_group in args.gqa_groups:
        if gqa_group not in (4, 8):
            raise SystemExit("--gqa-groups must contain only 4 and/or 8")
        for splits in args.splits:
            import megagemm.kernels.paged_attention as paged_attention
            paged_attention._GQA4_DECODE_DISABLED = False
            paged_attention._GQA4_DECODE_LOGGED = False
            paged_attention._GQA8_DECODE_DISABLED = False
            paged_attention._GQA8_DECODE_LOGGED = False
            result, samples_us = capture_case(
                splits=splits,
                gqa_group=gqa_group,
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                cos=cos,
                sin=sin,
                positions=positions,
                q_norm_weight=q_norm_weight,
                scale=scale,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            if reference is None:
                reference = result
            actual = result.float()
            expected = reference.float()
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    actual.reshape(1, -1),
                    expected.reshape(1, -1),
                ).item()
            )
            max_abs_error = float((actual - expected).abs().max().item())
            median_us = float(statistics.median(samples_us))
            row = {
                "gqa_group": int(gqa_group),
                "splits": int(splits),
                "median_us": median_us,
                "attention_ms_per_token_48_layers": median_us * 48.0 / 1000.0,
                "cosine_vs_first": cosine,
                "max_abs_error_vs_first": max_abs_error,
                "samples_us": samples_us,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    ranking = sorted(rows, key=lambda row: row["median_us"])
    print("\n== RANKING ==")
    for index, row in enumerate(ranking, 1):
        print(
            f"{index:02d}. gqa={row['gqa_group']} splits={row['splits']}: "
            f"{row['median_us']:.3f} us/layer "
            f"({row['attention_ms_per_token_48_layers']:.3f} ms/token)"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "config": vars(args),
                "rows": rows,
                "ranking": ranking,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
