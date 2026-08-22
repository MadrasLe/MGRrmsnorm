"""Cheap CUDA-graph safety gate for the Gemma 4 A4B B16 prefill.

The gate never downloads the model. It validates both real attention layouts,
external paged-KV writes, one synthetic expert layer, deterministic route
packing, and repeated numerical output before the paid harness loads the
48 GiB checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import megagemm.kernels.qwen3_moe as moe_kernel
from megagemm.kernels.qwen3_moe import (
    qwen3_moe_prepare_segmented_prefill_graph_workspace,
    qwen3_moe_segmented_prefill,
)
from megagemm.engine.kv_cache import BlockManager
from megagemm.kernels.gemma4_attention_prepare import (
    gemma4_prefill_attention_prepare,
)
from megagemm.kernels.paged_attention import prefill_attention


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float().reshape(-1)
    candidate_f = candidate.float().reshape(-1)
    return {
        "max_abs_error": float((reference_f - candidate_f).abs().max().item()),
        "cosine": float(F.cosine_similarity(reference_f, candidate_f, dim=0).item()),
    }


def run_attention_graph_case(
    *,
    name: str,
    batch_size: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    k_eq_v: bool,
    replays: int,
) -> dict:
    dtype = torch.bfloat16
    q_raw = torch.randn(
        (batch_size, seq_len, q_heads * head_dim),
        device="cuda",
        dtype=dtype,
    ).mul_(0.02)
    k_raw = torch.randn(
        (batch_size, seq_len, kv_heads * head_dim),
        device="cuda",
        dtype=dtype,
    ).mul_(0.02)
    v_raw = k_raw if k_eq_v else torch.randn_like(k_raw).mul_(0.02)
    q_weight = torch.randn((head_dim,), device="cuda", dtype=dtype).mul_(0.02)
    k_weight = torch.randn((head_dim,), device="cuda", dtype=dtype).mul_(0.02)
    positions = (
        torch.arange(seq_len, device="cuda", dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    frequencies = torch.randn(
        (seq_len, head_dim // 2), device="cuda", dtype=torch.float32
    ).mul_(0.01)
    cos = torch.cos(frequencies)
    sin = torch.sin(frequencies)
    q_out = torch.empty(
        (batch_size, q_heads, seq_len, head_dim), device="cuda", dtype=dtype
    )
    k_out = torch.empty(
        (batch_size, kv_heads, seq_len, head_dim), device="cuda", dtype=dtype
    )
    v_out = torch.empty_like(k_out)
    k_cache = torch.empty(
        (batch_size, seq_len, kv_heads, head_dim), device="cuda", dtype=dtype
    )
    v_cache = torch.empty_like(k_cache)
    causal = torch.tril(
        torch.ones((seq_len, seq_len), device="cuda", dtype=torch.bool)
    )
    attn_mask = torch.zeros(
        (batch_size, 1, seq_len, seq_len), device="cuda", dtype=dtype
    )
    attn_mask.masked_fill_(~causal.view(1, 1, seq_len, seq_len), float("-inf"))

    def run() -> torch.Tensor:
        q, k, v, _, _ = gemma4_prefill_attention_prepare(
            q_raw,
            k_raw,
            v_raw,
            q_weight,
            k_weight,
            cos,
            sin,
            positions,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            eps=1e-6,
            q_out=q_out,
            k_out=k_out,
            v_out=v_out,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        return prefill_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            scale=head_dim ** -0.5,
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    eager_reference = run().detach().clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    total_tokens = batch_size * seq_len
    block_size = 16
    num_blocks = (total_tokens + block_size - 1) // block_size
    manager = BlockManager(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
    )
    kv_phys = torch.arange(total_tokens, device="cuda", dtype=torch.long) // block_size
    kv_offs = torch.arange(total_tokens, device="cuda", dtype=torch.long) % block_size
    cu_seqlens = (
        torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * seq_len
    )

    first = None
    repeat_max_abs_error = 0.0
    kv_max_abs_error = 0.0
    for _ in range(max(1, int(replays))):
        graph.replay()
        manager.write_kv_prefill_packed(
            [],
            0,
            k_cache.reshape(total_tokens, kv_heads, head_dim),
            v_cache.reshape(total_tokens, kv_heads, head_dim),
            cu_seqlens,
            kv_mapping=(kv_phys, kv_offs),
        )
        torch.cuda.synchronize()
        current = graph_out.detach().clone()
        if first is None:
            first = current
        else:
            repeat_max_abs_error = max(
                repeat_max_abs_error,
                float((first.float() - current.float()).abs().max().item()),
            )
        cache = manager.kv_caches[0]
        cached_k = cache[kv_phys, 0, :, kv_offs, :]
        cached_v = cache[kv_phys, 1, :, kv_offs, :]
        kv_max_abs_error = max(
            kv_max_abs_error,
            float(
                (
                    cached_k.float()
                    - k_cache.reshape(total_tokens, kv_heads, head_dim).float()
                )
                .abs()
                .max()
                .item()
            ),
            float(
                (
                    cached_v.float()
                    - v_cache.reshape(total_tokens, kv_heads, head_dim).float()
                )
                .abs()
                .max()
                .item()
            ),
        )

    graph_error = tensor_error(eager_reference, first)
    correct = bool(
        graph_error["cosine"] >= 0.9999
        and graph_error["max_abs_error"] <= 0.03125
        and repeat_max_abs_error <= 0.03125
        and kv_max_abs_error == 0.0
        and torch.isfinite(first).all().item()
    )
    result = {
        "attention_type": name,
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "graph_error": graph_error,
        "repeat_max_abs_error": repeat_max_abs_error,
        "external_kv_max_abs_error": kv_max_abs_error,
        "external_kv_writes": max(1, int(replays)),
        "correct": correct,
    }
    del graph, graph_out, manager
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_b16_prefill_graph_preflight.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    dtype = torch.bfloat16
    rows = 400
    hidden_dim = 2816
    intermediate_dim = 704
    num_experts = 128
    top_k = 8
    assignments = rows * top_k
    block_m = 16

    print("Gemma4 B16 full-prefill CUDA-graph component preflight")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(
        f"  shape: rows={rows} H={hidden_dim} I={intermediate_dim} "
        f"E={num_experts} top_k={top_k} assignments={assignments}"
    )
    print(f"  grad_enabled: {torch.is_grad_enabled()}")
    if torch.is_grad_enabled():
        raise RuntimeError("Preflight must run with autograd disabled")

    torch.manual_seed(20260722)
    attention_graphs = []
    attention_shapes = (
        ("sliding", 8, 256, True),
        ("full", 2, 512, True),
    )
    for name, kv_heads, head_dim, k_eq_v in attention_shapes:
        attention_result = run_attention_graph_case(
            name=name,
            batch_size=16,
            seq_len=25,
            q_heads=16,
            kv_heads=kv_heads,
            head_dim=head_dim,
            k_eq_v=k_eq_v,
            replays=args.replays,
        )
        attention_graphs.append(attention_result)
        print("ATTENTION_GRAPH_PREFLIGHT " + json.dumps(attention_result, sort_keys=True))

    hidden = torch.randn((rows, hidden_dim), device="cuda", dtype=dtype).mul_(0.02)
    gate_up = torch.randn(
        (num_experts, 2 * intermediate_dim, hidden_dim),
        device="cuda",
        dtype=dtype,
    ).mul_(0.02)
    down = torch.randn(
        (num_experts, hidden_dim, intermediate_dim),
        device="cuda",
        dtype=dtype,
    ).mul_(0.02)
    # Exercise the exact compact-buffer upper bound. The first 64 experts receive
    # 33 routes each and the other 64 receive 17, yielding 64*3 + 64*2 = 320
    # tiles while keeping every row's top-8 experts distinct.
    selected_rows: list[list[int]] = []
    first_cursor = 0
    second_cursor = 0
    for row in range(rows):
        first_count = 6 if row < 112 else 5
        second_count = top_k - first_count
        first = [(first_cursor + offset) % 64 for offset in range(first_count)]
        second = [
            64 + (second_cursor + offset) % 64 for offset in range(second_count)
        ]
        selected_rows.append(first + second)
        first_cursor += first_count
        second_cursor += second_count
    selected = torch.tensor(selected_rows, device="cuda", dtype=torch.int64)
    routing = torch.rand((rows, top_k), device="cuda", dtype=dtype)
    routing.div_(routing.sum(dim=-1, keepdim=True))
    out = torch.empty_like(hidden)
    workspace: dict[str, torch.Tensor] = {}

    qwen3_moe_prepare_segmented_prefill_graph_workspace(
        workspace,
        assignments=assignments,
        hidden_dim=hidden_dim,
        device=hidden.device,
        num_experts=num_experts,
        block_m=block_m,
        route_dtype=dtype,
    )

    original_partial = bool(moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE)
    original_partial_max = int(
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS
    )
    moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = True
    moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = max(
        4096,
        assignments,
    )

    def run() -> torch.Tensor:
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
            block_m=block_m,
            block_n=128,
            block_k=64,
            num_warps=4,
            num_stages=3,
            fused_gate=True,
            dense_grid=False,
            route_scatter=True,
            fixed_route_pack=False,
            compact_route_pack=True,
            async_tiles_max_assignments=4096,
        )

    try:
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        reference = run().detach().clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run()
        graph.replay()
        torch.cuda.synchronize()
        first = graph_out.detach().clone()
        first_error = tensor_error(reference, first)

        repeat_max_abs_error = 0.0
        previous = first
        for _ in range(max(1, int(args.replays))):
            graph.replay()
            torch.cuda.synchronize()
            current = graph_out.detach().clone()
            repeat_max_abs_error = max(
                repeat_max_abs_error,
                float((previous.float() - current.float()).abs().max().item()),
            )
            previous = current

        num_tiles = int(workspace["segmented_compact_num_tiles"].item())
        max_tiles = int(workspace.get("segmented_prefill_max_tiles", 0) or 0)
        payload = {
            "status": "PASS",
            "gpu": torch.cuda.get_device_name(0),
            "shape": {
                "rows": rows,
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
                "num_experts": num_experts,
                "top_k": top_k,
                "assignments": assignments,
            },
            "graph_route_pack": int(
                workspace.get("segmented_prefill_graph_route_pack", 0) or 0
            ),
            "route_pack_passes": int(
                workspace.get("segmented_prefill_compact_route_pack_passes", 0) or 0
            ),
            "graph_partial_cached": int(
                workspace.get("segmented_prefill_graph_partial_cached", 0) or 0
            ),
            "num_tiles": num_tiles,
            "max_tiles": max_tiles,
            "reference_error": first_error,
            "repeat_max_abs_error": repeat_max_abs_error,
            "replays": max(1, int(args.replays)),
            "attention_graphs": attention_graphs,
        }
        correct = bool(
            all(result["correct"] for result in attention_graphs)
            and
            payload["graph_route_pack"] == 1
            and payload["route_pack_passes"] == 2
            and payload["graph_partial_cached"] == 1
            and num_tiles == max_tiles == 320
            and first_error["cosine"] >= 0.9999
            and first_error["max_abs_error"] <= 0.03125
            and repeat_max_abs_error == 0.0
        )
        if not correct:
            payload["status"] = "FAIL"
        print("PREFILL_GRAPH_PREFLIGHT " + json.dumps(payload, sort_keys=True))
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out_path}")
        return 0 if correct else 2
    finally:
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = original_partial
        moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = (
            original_partial_max
        )


if __name__ == "__main__":
    raise SystemExit(main())
