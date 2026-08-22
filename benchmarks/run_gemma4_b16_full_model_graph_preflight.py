"""Checkpoint-free full-model CUDA-graph gate for Gemma 4 A4B B16 prefill.

The test builds the production 30-layer topology on meta, shares identical
parameter storage across layers, and materializes only about 3 GiB on CUDA.
It still exercises every real layer, every per-layer MoE graph workspace,
both attention layouts, the 262k LM head, deferred K/V outputs, and repeated
full-model graph replay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
os.environ.setdefault("MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL", "1")
os.environ.setdefault("MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK", "1")
# v102 proved that the fused frontend can capture but faults on the first
# full-model replay. Use the graph-stable v98 frontend for both eager and graph
# references; the fused kernels remain available outside this B16 graph regime.
os.environ["MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL"] = "0"
os.environ["MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL"] = "0"
os.environ["MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND"] = "0"
os.environ.setdefault(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS",
    "4096",
)
os.environ.setdefault(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS",
    "4096",
)

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.models.llama import LlamaConfig, MegaGemmLlama


BATCH_SIZE = 16
SEQ_LEN = 25
ROWS = BATCH_SIZE * SEQ_LEN
HIDDEN_SIZE = 2816
SHARED_INTERMEDIATE = 2112
EXPERT_INTERMEDIATE = 704
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 30
VOCAB_SIZE = 262144


def exact_config(layer_limit: int = NUM_LAYERS) -> LlamaConfig:
    layer_limit = max(1, min(NUM_LAYERS, int(layer_limit)))
    layer_types = ["sliding_attention"] * layer_limit
    for layer_idx in (5, 11, 17, 23, 29):
        if layer_idx < layer_limit:
            layer_types[layer_idx] = "full_attention"
    return LlamaConfig.from_dict(
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": SHARED_INTERMEDIATE,
                "num_hidden_layers": layer_limit,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_global_key_value_heads": 2,
                "vocab_size": VOCAB_SIZE,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": True,
                "attention_k_eq_v": True,
                "hidden_activation": "gelu_pytorch_tanh",
                "layer_types": layer_types,
                "sliding_window": 1024,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": False,
                "enable_moe_block": True,
                "num_experts": NUM_EXPERTS,
                "top_k_experts": TOP_K,
                "moe_intermediate_size": EXPERT_INTERMEDIATE,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": VOCAB_SIZE,
                "final_logit_softcapping": 30.0,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                    },
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                    },
                },
            },
        }
    )


def _set_parameter(root: nn.Module, name: str, value: nn.Parameter) -> None:
    parent_name, _, leaf_name = name.rpartition(".")
    owner = root.get_submodule(parent_name) if parent_name else root
    owner._parameters[leaf_name] = value


def share_identical_layer_parameters(model: MegaGemmLlama) -> None:
    """Share storage while retaining 30 independent layer/workspace objects."""
    shared: dict[tuple[str, tuple[int, ...], torch.dtype], nn.Parameter] = {}
    for layer in model.layers:
        for name, parameter in tuple(
            layer.named_parameters(remove_duplicate=False)
        ):
            signature = (name, tuple(parameter.shape), parameter.dtype)
            canonical = shared.setdefault(signature, parameter)
            _set_parameter(layer, name, canonical)
    model.lm_head.weight = model.embed_tokens.weight


def materialize_meta_module(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Materialize meta tensors without breaking shared Parameter aliases."""
    parameters: dict[int, nn.Parameter] = {}
    buffers: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is None:
                continue
            key = id(parameter)
            materialized = parameters.get(key)
            if materialized is None:
                target_dtype = dtype if parameter.is_floating_point() else parameter.dtype
                materialized = nn.Parameter(
                    torch.empty(
                        tuple(parameter.shape),
                        device=device,
                        dtype=target_dtype,
                    ),
                    requires_grad=False,
                )
                parameters[key] = materialized
            module._parameters[name] = materialized
        for name, buffer in tuple(module._buffers.items()):
            if buffer is None:
                continue
            key = id(buffer)
            materialized_buffer = buffers.get(key)
            if materialized_buffer is None:
                target_dtype = dtype if buffer.is_floating_point() else buffer.dtype
                materialized_buffer = torch.empty(
                    tuple(buffer.shape),
                    device=device,
                    dtype=target_dtype,
                )
                buffers[key] = materialized_buffer
            module._buffers[name] = materialized_buffer


def initialize_synthetic_weights(model: MegaGemmLlama) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for name, parameter in model.named_parameters():
            if (
                ("norm" in name and name.endswith(".weight"))
                or name.endswith(".gate.scale")
                or name.endswith(".gate.per_expert_scale")
            ):
                parameter.fill_(1)
            elif name.endswith(".gate.proj.weight"):
                parameter.normal_(mean=0.0, std=0.02)
            elif name.endswith(
                (
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.o_proj.weight",
                )
            ):
                parameter.normal_(
                    mean=0.0,
                    std=0.02 * (int(parameter.shape[-1]) ** -0.5),
                )
        for name, buffer in model.named_buffers():
            if name.endswith("layer_scalar"):
                buffer.fill_(1)
        model.embed_tokens.weight[:ROWS].normal_(mean=0.0, std=0.02)


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float().reshape(-1)
    candidate_f = candidate.float().reshape(-1)
    if not bool(
        torch.isfinite(reference_f).all().item()
        and torch.isfinite(candidate_f).all().item()
    ):
        return {
            "max_abs_error": 1.0e30,
            "cosine": -1.0,
        }
    max_abs_error = float((reference_f - candidate_f).abs().max().item())
    reference_norm = float(torch.linalg.vector_norm(reference_f).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate_f).item())
    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 1.0 if max_abs_error == 0.0 else 0.0
    else:
        cosine = float(
            F.cosine_similarity(reference_f, candidate_f, dim=0).item()
        )
    return {
        "max_abs_error": max_abs_error,
        "cosine": cosine,
    }


def deferred_shapes_are_exact(deferred_kv: tuple, layer_limit: int) -> bool:
    if len(deferred_kv) != int(layer_limit):
        return False
    full_layers = {5, 11, 17, 23, 29}
    for layer_idx, k_cache, v_cache in deferred_kv:
        expected = (
            (ROWS, 2, 512)
            if int(layer_idx) in full_layers
            else (ROWS, 8, 256)
        )
        if tuple(k_cache.shape) != expected or tuple(v_cache.shape) != expected:
            return False
    return True


def clone_deferred_kv(deferred_kv: tuple) -> tuple:
    return tuple(
        (int(layer_idx), k_cache.detach().clone(), v_cache.detach().clone())
        for layer_idx, k_cache, v_cache in deferred_kv
    )


def deferred_kv_max_abs_error(reference: tuple, candidate: tuple) -> float:
    if len(reference) != len(candidate):
        return 1.0e30
    maximum = 0.0
    for expected, actual in zip(reference, candidate):
        expected_idx, expected_k, expected_v = expected
        actual_idx, actual_k, actual_v = actual
        if (
            int(expected_idx) != int(actual_idx)
            or tuple(expected_k.shape) != tuple(actual_k.shape)
            or tuple(expected_v.shape) != tuple(actual_v.shape)
        ):
            return 1.0e30
        maximum = max(
            maximum,
            float((expected_k.float() - actual_k.float()).abs().max().item()),
            float((expected_v.float() - actual_v.float()).abs().max().item()),
        )
    return maximum


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=NUM_LAYERS,
        help="Capture only this exact prefix of the 30-layer topology.",
    )
    parser.add_argument(
        "--skip-final-projection",
        action="store_true",
        help="Replace final norm/LM head with identity for fault isolation.",
    )
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_b16_full_model_graph_preflight.json",
    )
    args = parser.parse_args()
    layer_limit = max(1, min(NUM_LAYERS, int(args.layer_limit)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260729)
    torch.cuda.reset_peak_memory_stats()

    print("Gemma4 B16 checkpoint-free full-model prefill graph gate")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(
        "  topology: "
        f"batch={BATCH_SIZE} context={SEQ_LEN} rows={ROWS} "
        f"layers={layer_limit}/{NUM_LAYERS} H={HIDDEN_SIZE} "
        f"E={NUM_EXPERTS} top_k={TOP_K}"
    )
    print("  synthetic_storage: identical layer parameters shared")
    print(
        "  final_projection: "
        f"{'disabled' if args.skip_final_projection else 'enabled'}"
    )

    build_start = time.perf_counter()
    config = exact_config(layer_limit)
    with torch.device("meta"):
        model = MegaGemmLlama(config).eval()
    share_identical_layer_parameters(model)
    materialize_meta_module(model, device=device, dtype=dtype)
    model.lm_head.weight = model.embed_tokens.weight
    model.set_rope_cache_max_seq_len(SEQ_LEN, device=device)
    model._refresh_gemma4_runtime_buffers(device=device, dtype=dtype)
    initialize_synthetic_weights(model)
    if args.skip_final_projection:
        model.norm = nn.Identity()
        model.lm_head = nn.Identity()
        model.final_logit_softcapping = 0.0
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    unique_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    workspace_refs = tuple(
        model.prepare_prefill_cuda_graph_workspace(
            total_tokens=ROWS,
            device=device,
        )
    )
    workspace_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in workspace_refs
    )
    expected_workspace_bytes = (
        layer_limit * ROWS * TOP_K * HIDDEN_SIZE * 4
    )
    input_ids = torch.arange(ROWS, device=device, dtype=torch.long).reshape(
        BATCH_SIZE,
        SEQ_LEN,
    )
    cu_seqlens = (
        torch.arange(BATCH_SIZE + 1, device=device, dtype=torch.int32) * SEQ_LEN
    )
    kv_phys = torch.arange(ROWS, device=device, dtype=torch.long)
    kv_offs = torch.zeros(ROWS, device=device, dtype=torch.long)
    unused_block_manager = object()

    def run(*, graph_safe_prefill: bool = True):
        return model.prefill_batch_graph(
            input_ids,
            cu_seqlens,
            unused_block_manager,
            kv_phys,
            kv_offs,
            defer_kv_writes=True,
            graph_safe_prefill=graph_safe_prefill,
        )

    # The first pass settles JIT/runtime policy. The second pass is the eager
    # graph-safe reference and must use exactly the capture dispatches. Do not
    # call the ordinary eager route pack here: this checkpoint-free model aliases
    # expert parameter storage across 30 layers, which is intentionally outside
    # the production eager path's ownership contract. The paid real-checkpoint
    # gate compares three generated tokens against ordinary eager before vLLM.
    print("FULL_MODEL_GRAPH_STAGE warmup_start", flush=True)
    warm_start = time.perf_counter()
    warm_result = run()
    torch.cuda.synchronize()
    del warm_result
    warm_result = run()
    torch.cuda.synchronize()
    print("FULL_MODEL_GRAPH_STAGE warmup_complete", flush=True)
    warm_ms = (time.perf_counter() - warm_start) * 1000.0
    reference_logits = warm_result[0].detach().clone()
    reference_kv = clone_deferred_kv(tuple(warm_result[1]))
    warm_shapes_exact = deferred_shapes_are_exact(
        tuple(warm_result[1]),
        layer_limit,
    )
    del warm_result

    print("FULL_MODEL_GRAPH_STAGE capture_start", flush=True)
    capture_start = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_result = run()
    print("FULL_MODEL_GRAPH_STAGE capture_closed", flush=True)
    # Attribute asynchronous faults to capture execution rather than the first
    # replay. The parent diagnostic runs every prefix in a fresh CUDA process.
    torch.cuda.synchronize()
    print("FULL_MODEL_GRAPH_STAGE capture_synchronized", flush=True)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    first = None
    repeat_max_abs_error = 0.0
    replay_samples_ms: list[float] = []
    for _ in range(max(1, int(args.replays))):
        print("FULL_MODEL_GRAPH_STAGE replay_start", flush=True)
        replay_start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        print("FULL_MODEL_GRAPH_STAGE replay_synchronized", flush=True)
        replay_samples_ms.append((time.perf_counter() - replay_start) * 1000.0)
        current = graph_result[0].detach().clone()
        if first is None:
            first = current
        else:
            repeat_max_abs_error = max(
                repeat_max_abs_error,
                float((first.float() - current.float()).abs().max().item()),
            )

    graph_error = tensor_error(reference_logits, first)
    graph_deferred_kv_max_abs_error = deferred_kv_max_abs_error(
        reference_kv,
        tuple(graph_result[1]),
    )
    graph_shapes_exact = deferred_shapes_are_exact(
        tuple(graph_result[1]),
        layer_limit,
    )
    runtime = model.decode_runtime_stats()
    graph_route_pack_layers = int(
        runtime.get("qwen3_moe_segmented_prefill_graph_route_pack_layers", 0)
        or 0
    )
    graph_safe_attention_frontend_layers = sum(
        bool(
            getattr(layer.self_attn, "_gemma4_fused_qkv_prefill_skip_reason", "")
            == "prefill CUDA graph safety guard"
            and getattr(
                layer.self_attn,
                "_gemma4_fused_attn_prepare_skip_reason",
                "",
            )
            == "prefill CUDA graph safety guard"
        )
        for layer in model.layers
    )
    graph_fused_attention_frontend_layers = sum(
        bool(
            int(getattr(layer.self_attn, "_gemma4_fused_qkv_prefill_hits", 0)) > 0
            and int(getattr(layer.self_attn, "_gemma4_fused_attn_prepare_hits", 0)) > 0
        )
        for layer in model.layers
    )
    graph_safe_moe_layers = sum(
        bool(
            getattr(
                getattr(getattr(layer, "mlp", None), "experts", None),
                "_segmented_prefill_workspace",
                {},
            ).get("segmented_prefill_graph_mode_requested", 0)
        )
        for layer in model.layers
    )
    persistent_deferred_kv_buffers = int(
        runtime.get("gemma4_prefill_graph_deferred_kv_buffers", 0) or 0
    )
    persistent_deferred_kv_bytes = int(
        runtime.get("gemma4_prefill_graph_deferred_kv_bytes", 0) or 0
    )
    persistent_deferred_kv_copy_dispatches = int(
        runtime.get(
            "gemma4_prefill_graph_deferred_kv_copy_dispatches",
            0,
        )
        or 0
    )
    payload = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "rows": ROWS,
            "layers": layer_limit,
            "total_model_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "shared_intermediate": SHARED_INTERMEDIATE,
            "expert_intermediate": EXPERT_INTERMEDIATE,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "vocab_size": VOCAB_SIZE,
            "dtype": "bf16",
        },
        "model_download": False,
        "vllm_install": False,
        "reference_mode": "graph_safe_eager",
        "attention_frontend_mode": "unfused_graph_stable_direct_kv_v105",
        "shared_parameter_storage": True,
        "nonzero_attention_weights": True,
        "skip_final_projection": bool(args.skip_final_projection),
        "unique_parameter_bytes": unique_parameter_bytes,
        "workspace_tensors": len(workspace_refs),
        "workspace_bytes": workspace_bytes,
        "expected_workspace_bytes": expected_workspace_bytes,
        "capture_body_warmups": 2,
        "warm_deferred_shapes_exact": warm_shapes_exact,
        "graph_deferred_shapes_exact": graph_shapes_exact,
        "graph_route_pack_layers": graph_route_pack_layers,
        "graph_safe_attention_frontend_layers": (
            graph_safe_attention_frontend_layers
        ),
        "graph_fused_attention_frontend_layers": (
            graph_fused_attention_frontend_layers
        ),
        "graph_safe_moe_layers": graph_safe_moe_layers,
        "persistent_deferred_kv_buffers": persistent_deferred_kv_buffers,
        "persistent_deferred_kv_bytes": persistent_deferred_kv_bytes,
        "persistent_deferred_kv_copy_dispatches": (
            persistent_deferred_kv_copy_dispatches
        ),
        "reference_error": graph_error,
        "graph_deferred_kv_max_abs_error": graph_deferred_kv_max_abs_error,
        "repeat_max_abs_error": repeat_max_abs_error,
        "finite": bool(
            torch.isfinite(reference_logits).all().item()
            and torch.isfinite(first).all().item()
        ),
        "build_ms": build_ms,
        "warm_ms": warm_ms,
        "capture_ms": capture_ms,
        "replay_samples_ms": replay_samples_ms,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    correct = bool(
        len(workspace_refs) == layer_limit
        and workspace_bytes == expected_workspace_bytes
        and warm_shapes_exact
        and graph_shapes_exact
        and graph_route_pack_layers == layer_limit
        and graph_safe_attention_frontend_layers == layer_limit
        and graph_fused_attention_frontend_layers == 0
        and graph_safe_moe_layers == layer_limit
        and persistent_deferred_kv_buffers == layer_limit
        and persistent_deferred_kv_bytes > 0
        and persistent_deferred_kv_copy_dispatches >= layer_limit * 3
        and payload["finite"]
        and graph_error["cosine"] >= 0.9999
        and graph_error["max_abs_error"] <= 0.03125
        and graph_deferred_kv_max_abs_error == 0.0
        and repeat_max_abs_error == 0.0
    )
    if not correct:
        payload["status"] = "FAIL"

    print("FULL_MODEL_GRAPH_PREFLIGHT " + json.dumps(payload, sort_keys=True))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0 if correct else 2


if __name__ == "__main__":
    raise SystemExit(main())
