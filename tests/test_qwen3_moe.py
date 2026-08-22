import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megagemm.models.llama import LlamaConfig, MegaGemmLlama, Qwen3MoeMLP
import megagemm.models.llama as llama_mod
from megagemm.models.loader import _load_fp16_streaming, _map_weights
import megagemm.kernels.qwen3_moe as qwen3_moe_kernel
from megagemm.kernels.qwen3_moe import (
    HAS_QWEN3_MOE_GROUPED,
    qwen3_moe_grouped_decode,
    qwen3_moe_prepare_segmented_prefill_graph_workspace,
    qwen3_moe_router_topk_softmax,
    qwen3_moe_segmented_prefill,
    qwen3_moe_topk_softmax,
)


def test_segmented_prefill_graph_partial_workspace_is_shape_persistent():
    workspace = {}
    first = qwen3_moe_prepare_segmented_prefill_graph_workspace(
        workspace,
        assignments=3200,
        hidden_dim=4,
        device=torch.device("cpu"),
        num_experts=128,
        block_m=16,
        route_dtype=torch.bfloat16,
    )
    same = qwen3_moe_prepare_segmented_prefill_graph_workspace(
        workspace,
        assignments=3200,
        hidden_dim=4,
        device=torch.device("cpu"),
    )
    second_shape = qwen3_moe_prepare_segmented_prefill_graph_workspace(
        workspace,
        assignments=800,
        hidden_dim=4,
        device=torch.device("cpu"),
    )

    assert first is same
    assert first.shape == (3200, 4)
    assert first.dtype == torch.float32
    assert second_shape.shape == (800, 4)
    assert workspace["segmented_graph_partial_out_3200_4"] is first
    assert workspace["segmented_graph_partial_out_800_4"] is second_shape
    assert workspace["segmented_prefill_graph_partial_bytes"] == 800 * 4 * 4
    assert workspace["segmented_prefill_graph_partial_dtype"] == "torch.float32"
    assert workspace["segmented_compact_sorted_tokens"].shape == (3200,)
    assert workspace["segmented_compact_sorted_route"].dtype == torch.bfloat16
    assert workspace["segmented_compact_tile_experts"].shape == (320,)
    assert workspace["segmented_graph_route_counts"].shape == (128,)
    assert workspace["segmented_graph_route_tile_offsets"].dtype == torch.int32
    assert workspace["segmented_prefill_graph_route_workspace_bytes"] > 0


def _qwen3_moe_config(**overrides):
    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "attention_bias": False,
        "hidden_act": "silu",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
    }
    config.update(overrides)
    return config


def test_qwen3_moe_config_parsing():
    cfg = LlamaConfig.from_dict(_qwen3_moe_config())

    assert cfg.model_type == "qwen3_moe"
    assert cfg.qk_norm is True
    assert cfg.rope_half_rotate is True
    assert cfg.num_experts == 4
    assert cfg.num_experts_per_tok == 2
    assert cfg.moe_intermediate_size == 6
    assert cfg.is_moe_layer(0) is True
    assert cfg.is_moe_layer(1) is True


def test_qwen3_moe_full_attention_uses_all_full_fastpath():
    cfg = LlamaConfig.from_dict(_qwen3_moe_config())
    model = MegaGemmLlama(cfg)

    assert all(layer.layer_type == "full_attention" for layer in model.layers)
    assert model._all_full_attention is True


def test_qwen3_coder_batch1_graph_blocks_slow_fused_qkv():
    class FakeCudaTensor:
        is_cuda = True
        shape = (1, 2048)

        @staticmethod
        def dim():
            return 2

    names = (
        "_USE_FUSED_RMSNORM_QKV_DECODE",
        "_DECODE_CUDA_GRAPHS_ENABLED",
        "_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS",
        "fused_rmsnorm_linear",
        "fused_rmsnorm_linear_prefers_triton_shape",
    )
    old_values = {name: getattr(llama_mod, name) for name in names}
    try:
        llama_mod._USE_FUSED_RMSNORM_QKV_DECODE = True
        llama_mod._DECODE_CUDA_GRAPHS_ENABLED = True
        llama_mod._FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS = False
        llama_mod.fused_rmsnorm_linear = object()
        llama_mod.fused_rmsnorm_linear_prefers_triton_shape = (
            lambda _in_dim, _out_dim, _rows: True
        )
        with torch.no_grad():
            assert not llama_mod._can_use_fused_rmsnorm_qkv_for(
                FakeCudaTensor(),
                5120,
            )
            assert llama_mod._can_use_fused_rmsnorm_qkv_for(
                FakeCudaTensor(),
                4096,
            )
            llama_mod._FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS = True
            assert llama_mod._can_use_fused_rmsnorm_qkv_for(
                FakeCudaTensor(),
                5120,
            )
    finally:
        for name, value in old_values.items():
            setattr(llama_mod, name, value)


def test_qwen3_moe_respects_dense_mlp_only_layers():
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(num_hidden_layers=3, decoder_sparse_step=1, mlp_only_layers=[1])
    )

    assert cfg.is_moe_layer(0) is True
    assert cfg.is_moe_layer(1) is False
    assert cfg.is_moe_layer(2) is True


def test_qwen3_moe_forward_matches_reference():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen3_moe_config(hidden_size=4, moe_intermediate_size=5, num_experts=3))
    moe = Qwen3MoeMLP(cfg)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight))
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj))
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj))

    x = torch.randn(2, 3, cfg.hidden_size)
    actual = moe(x)

    flat = x.reshape(-1, cfg.hidden_size)
    logits = torch.nn.functional.linear(flat, moe.gate.weight)
    probs = torch.nn.functional.softmax(logits, dtype=torch.float32, dim=-1)
    weights, experts = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(logits.dtype)

    expected = torch.zeros_like(flat)
    for token_idx in range(flat.shape[0]):
        for top_idx in range(cfg.num_experts_per_tok):
            expert_idx = int(experts[token_idx, top_idx])
            gate_up = torch.nn.functional.linear(
                flat[token_idx : token_idx + 1],
                moe.experts.gate_up_proj[expert_idx],
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = torch.nn.functional.silu(gate) * up
            expert_out = torch.nn.functional.linear(
                expert_out,
                moe.experts.down_proj[expert_idx],
            )
            expected[token_idx] += expert_out.squeeze(0) * weights[token_idx, top_idx]

    expected = expected.reshape_as(actual)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen3_moe_topk_softmax_matches_full_softmax_renorm():
    torch.manual_seed(3)
    logits = torch.randn(5, 17)
    top_k = 4

    actual_weights, actual_experts = qwen3_moe_topk_softmax(logits, top_k)

    probs = torch.nn.functional.softmax(logits, dtype=torch.float32, dim=-1)
    expected_weights, expected_experts = torch.topk(probs, top_k, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights.to(logits.dtype)

    assert torch.equal(actual_experts, expected_experts)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_segmented_tile_upper_bound_covers_all_expert_partitions():
    block_m = 4
    assignments = 9
    num_experts = 5
    bound = qwen3_moe_kernel._segmented_tile_upper_bound(
        assignments,
        num_experts,
        block_m,
    )

    def visit(remaining, expert, counts):
        if expert == num_experts - 1:
            candidate = counts + [remaining]
            actual = sum((count + block_m - 1) // block_m for count in candidate)
            assert actual <= bound
            return
        for count in range(remaining + 1):
            visit(remaining - count, expert + 1, counts + [count])

    visit(assignments, 0, [])
    assert bound == 6


def test_segmented_argsort_preserves_original_assignment_slots():
    rows, top_k = 3, 2
    flat_experts = torch.tensor([2, 0, 1, 2, 0, 1], dtype=torch.int64)
    flat_route = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.3, 0.7])

    sorted_tokens, sorted_route, sorted_slots = (
        qwen3_moe_kernel._route_assignments_by_expert_argsort(
            flat_experts,
            flat_route,
            rows=rows,
            top_k=top_k,
            workspace={},
        )
    )

    assert torch.equal(sorted_tokens, sorted_slots // top_k)
    assert torch.equal(sorted_route, flat_route.index_select(0, sorted_slots))
    assert torch.equal(flat_experts.index_select(0, sorted_slots), flat_experts.sort().values)


def test_qwen3_moe_router_topk_softmax_matches_reference():
    torch.manual_seed(4)
    hidden = torch.randn(3, 11)
    weight = torch.randn(17, 11)
    top_k = 4

    actual_weights, actual_experts = qwen3_moe_router_topk_softmax(hidden, weight, top_k)

    logits = torch.nn.functional.linear(hidden, weight)
    probs = torch.nn.functional.softmax(logits, dtype=torch.float32, dim=-1)
    expected_weights, expected_experts = torch.topk(probs, top_k, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights.to(logits.dtype)

    assert torch.equal(actual_experts, expected_experts)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_qwen3_moe_router_topk_softmax_cuda_matches_reference_if_available():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(5)
    hidden = torch.randn(1, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    top_k = 4

    logits = torch.nn.functional.linear(hidden, weight)
    probs = torch.nn.functional.softmax(logits, dtype=torch.float32, dim=-1)
    expected_weights, expected_experts = torch.topk(probs, top_k, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights.to(logits.dtype)

    old_router_k_splits = qwen3_moe_kernel._CFG_ROUTER_K_SPLITS
    try:
        for router_k_splits in (1, 2, 4):
            qwen3_moe_kernel._CFG_ROUTER_K_SPLITS = router_k_splits
            workspace = {}
            actual_weights, actual_experts = qwen3_moe_router_topk_softmax(
                hidden,
                weight,
                top_k,
                workspace=workspace,
            )
            torch.cuda.synchronize()
            assert set(["router_weights", "router_experts"]).issubset(workspace)
            assert workspace.get("router_last_k_splits") == router_k_splits
            assert torch.equal(actual_experts.cpu(), expected_experts.cpu())
            assert torch.allclose(
                actual_weights.cpu(),
                expected_weights.cpu(),
                atol=1e-5,
                rtol=1e-5,
            )
    finally:
        qwen3_moe_kernel._CFG_ROUTER_K_SPLITS = old_router_k_splits


def test_qwen3_moe_grouped_decode_fallback_matches_reference():
    torch.manual_seed(1)
    cfg = LlamaConfig.from_dict(_qwen3_moe_config(hidden_size=4, moe_intermediate_size=5, num_experts=3))
    moe = Qwen3MoeMLP(cfg)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight))
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj))
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj))

    hidden = torch.randn(2, cfg.hidden_size)
    _, routing_weights, selected_experts = moe.gate(hidden)

    actual = qwen3_moe_grouped_decode(
        hidden,
        moe.experts.gate_up_proj,
        moe.experts.down_proj,
        selected_experts,
        routing_weights,
        activation=cfg.hidden_act,
    )

    expected = moe.experts(
        hidden,
        selected_experts,
        routing_weights,
        use_grouped_decode=False,
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen3_moe_sorted_prefill_matches_reference():
    torch.manual_seed(7)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=4,
            moe_intermediate_size=5,
            num_experts=5,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.1)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.1)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.1)

    hidden = torch.randn(9, cfg.hidden_size) * 0.1
    with torch.inference_mode():
        _, routing_weights, selected_experts = moe.gate(hidden)
        actual = moe.experts._forward_sorted_prefill(
            hidden,
            selected_experts,
            routing_weights,
        )
        expected = moe.experts(
            hidden,
            selected_experts,
            routing_weights,
            use_grouped_decode=False,
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen3_moe_batched_prefill_matches_reference():
    torch.manual_seed(8)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=4,
            moe_intermediate_size=5,
            num_experts=5,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.1)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.1)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.1)

    hidden = torch.randn(11, cfg.hidden_size) * 0.1
    with torch.inference_mode():
        _, routing_weights, selected_experts = moe.gate(hidden)
        actual = moe.experts._forward_batched_prefill(
            hidden,
            selected_experts,
            routing_weights,
        )
        expected = moe.experts(
            hidden,
            selected_experts,
            routing_weights,
            use_grouped_decode=False,
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen3_moe_bucketed_prefill_matches_reference():
    torch.manual_seed(9)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=4,
            moe_intermediate_size=5,
            num_experts=7,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.1)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.1)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.1)

    hidden = torch.randn(17, cfg.hidden_size) * 0.1
    with torch.inference_mode():
        _, routing_weights, selected_experts = moe.gate(hidden)
        actual = moe.experts._forward_bucketed_prefill(
            hidden,
            selected_experts,
            routing_weights,
        )
        expected = moe.experts(
            hidden,
            selected_experts,
            routing_weights,
            use_grouped_decode=False,
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_qwen3_moe_segmented_prefill_cuda_matches_batched_if_available():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(10)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=16,
            moe_intermediate_size=16,
            num_experts=5,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.float32).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.05)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.05)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.05)

    hidden = torch.randn(23, cfg.hidden_size, device="cuda", dtype=torch.float32) * 0.05
    old_min = qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS
    old_block_m = qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M
    old_block_n = qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N
    old_block_k = qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K
    try:
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = 1
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M = 8
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N = 16
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K = 16
        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)
            actual = qwen3_moe_segmented_prefill(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                workspace={},
            )
            expected = moe.experts._forward_batched_prefill(
                hidden,
                selected_experts,
                routing_weights,
            )
    finally:
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = old_min
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_M = old_block_m
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_N = old_block_n
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_BLOCK_K = old_block_k

    torch.cuda.synchronize()
    max_abs = (actual - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        expected.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 2e-3
    assert cosine > 0.99999


def test_segmented_prefill_deterministic_defaults_cover_gemma4_batch16():
    cfg = qwen3_moe_kernel.qwen3_moe_grouped_runtime_config()
    assert cfg["segmented_prefill_async_tiles_max_assignments"] >= 3200
    assert cfg["segmented_prefill_partial_reduce_max_assignments"] >= 3200
    assert cfg["segmented_prefill_partial_cache_max_assignments"] <= 512
    assert "segmented_prefill_sorted_partial" in cfg


def test_segmented_prefill_sorted_partial_inverse_preserves_topk_reduction():
    rows, top_k, hidden_dim = 3, 4, 5
    sorted_slots = torch.tensor([7, 0, 5, 9, 2, 11, 1, 8, 4, 6, 10, 3])
    slot_to_sorted = torch.empty_like(sorted_slots)
    slot_to_sorted[sorted_slots] = torch.arange(sorted_slots.numel())

    original_partial = torch.arange(
        rows * top_k * hidden_dim,
        dtype=torch.float32,
    ).reshape(rows * top_k, hidden_dim)
    sorted_partial = original_partial[sorted_slots]

    baseline = original_partial.reshape(rows, top_k, hidden_dim).sum(dim=1)
    restored = sorted_partial[slot_to_sorted].reshape(rows, top_k, hidden_dim).sum(dim=1)

    assert torch.equal(restored, baseline)


def test_segmented_prefill_sorted_partial_kernel_contract_is_wired():
    import inspect

    module_source = inspect.getsource(qwen3_moe_kernel)
    assert "def _qwen3_moe_invert_sorted_slots_kernel(" in module_source
    assert "partial_rows = row_ids" in module_source
    assert "partial_rows = tl.load(slot_to_sorted_ptr + slots)" in module_source
    assert "SORTED_PARTIAL=sorted_partial_active" in module_source


def test_async_tile_builder_has_compact_grid_and_no_gpu_scalar_sync():
    import inspect

    builder_source = inspect.getsource(
        qwen3_moe_kernel._build_segmented_tile_tensors_gpu_async
    )
    module_source = inspect.getsource(qwen3_moe_kernel)

    assert ".item()" not in builder_source
    assert "[(max_tiles,)]" in builder_source
    assert "SEARCH_STEPS" in builder_source
    assert "def _qwen3_moe_build_compact_tiles_kernel(" in module_source
    assert "tl.static_range(0, SEARCH_STEPS)" in module_source
    assert "middle_offset + middle_tiles <= tile" in module_source


def test_compact_active_list_flag_does_not_require_global_master_switch():
    old_has_triton = qwen3_moe_kernel._HAS_TRITON
    old_master = qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE
    old_active = qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST
    old_early_exit = (
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT
    )
    try:
        qwen3_moe_kernel._HAS_TRITON = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = True
        cfg = qwen3_moe_kernel.qwen3_moe_grouped_runtime_config()
        assert cfg["expert_grouped_compact_active_list"] is True
        assert cfg["expert_grouped_compact_active_list_early_exit"] is True
    finally:
        qwen3_moe_kernel._HAS_TRITON = old_has_triton
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = old_master
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = old_active
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = (
            old_early_exit
        )


def test_segmented_prefill_partial_reduce_is_repeatable_above_old_512_limit_cuda():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(20260716)
    rows, hidden_dim, intermediate_dim = 300, 16, 16
    num_experts, top_k = 5, 2
    hidden = torch.randn(
        (rows, hidden_dim), device="cuda", dtype=torch.bfloat16
    ).mul_(0.05)
    gate_up = torch.randn(
        (num_experts, 2 * intermediate_dim, hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.05)
    down = torch.randn(
        (num_experts, hidden_dim, intermediate_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.05)
    selected = torch.arange(
        rows * top_k, device="cuda", dtype=torch.int64
    ).reshape(rows, top_k).remainder_(num_experts)
    routing = torch.full(
        (rows, top_k),
        1.0 / top_k,
        device="cuda",
        dtype=torch.bfloat16,
    )
    workspace = {}
    old_partial = qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE
    old_partial_limit = (
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS
    )
    old_cache_limit = (
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS
    )
    try:
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = True
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = 4096
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS = 512
        with torch.inference_mode():
            first = qwen3_moe_segmented_prefill(
                hidden,
                gate_up,
                down,
                selected,
                routing,
                workspace=workspace,
                force=True,
                block_m=16,
                block_n=16,
                block_k=16,
                num_warps=4,
                num_stages=2,
                fused_gate=True,
                dense_grid=False,
                route_scatter=True,
                async_tiles_max_assignments=4096,
            ).clone()
            second = qwen3_moe_segmented_prefill(
                hidden,
                gate_up,
                down,
                selected,
                routing,
                workspace=workspace,
                force=True,
                block_m=16,
                block_n=16,
                block_k=16,
                num_warps=4,
                num_stages=2,
                fused_gate=True,
                dense_grid=False,
                route_scatter=True,
                async_tiles_max_assignments=4096,
            ).clone()
    finally:
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = old_partial
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = (
            old_partial_limit
        )
        qwen3_moe_kernel._CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS = (
            old_cache_limit
        )

    torch.cuda.synchronize()
    assert torch.equal(first, second)
    assert workspace.get("segmented_prefill_partial_reduce") == 1
    assert workspace.get("segmented_prefill_partial_cached") == 0
    assert "segmented_partial_out" not in workspace
    assert workspace.get("segmented_prefill_partial_dtype") == "torch.float32"


def test_qwen3_moe_grouped_decode_cuda_matches_eager_and_reuses_workspace():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(2)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=8,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.float32).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.1)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.1)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.1)

    hidden = torch.randn(3, cfg.hidden_size, device="cuda", dtype=torch.float32) * 0.1
    workspace = {}
    out = torch.empty_like(hidden)

    with torch.inference_mode():
        _, routing_weights, selected_experts = moe.gate(hidden)
        actual = qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            out=out,
            workspace=workspace,
        )
        expected = moe.experts(
            hidden,
            selected_experts,
            routing_weights,
            use_grouped_decode=False,
        )

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert set(["token_ids", "gate_up", "accum"]).issubset(workspace)
    max_abs = (actual - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        expected.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 1e-4
    assert cosine > 0.99999

    ptrs = {name: workspace[name].data_ptr() for name in ("token_ids", "gate_up", "accum")}
    with torch.inference_mode():
        qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            out=out,
            workspace=workspace,
        )
    torch.cuda.synchronize()
    assert {name: workspace[name].data_ptr() for name in ptrs} == ptrs


def test_qwen3_moe_token_accum_cuda_matches_eager_if_available():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(6)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=8,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.float32).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.1)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.1)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.1)

    hidden = torch.randn(4, cfg.hidden_size, device="cuda", dtype=torch.float32) * 0.1
    workspace = {}
    out = torch.empty_like(hidden)
    old_token_accum = qwen3_moe_kernel._CFG_TOKEN_ACCUM
    old_min_rows = qwen3_moe_kernel._CFG_TOKEN_ACCUM_MIN_ROWS
    try:
        qwen3_moe_kernel._CFG_TOKEN_ACCUM = True
        qwen3_moe_kernel._CFG_TOKEN_ACCUM_MIN_ROWS = 1
        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
            expected = moe.experts(
                hidden,
                selected_experts,
                routing_weights,
                use_grouped_decode=False,
            )
    finally:
        qwen3_moe_kernel._CFG_TOKEN_ACCUM = old_token_accum
        qwen3_moe_kernel._CFG_TOKEN_ACCUM_MIN_ROWS = old_min_rows

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert "gate_up" in workspace
    max_abs = (actual - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        expected.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 1e-4
    assert cosine > 0.99999


def test_qwen3_moe_compact_direct_out_bf16_cuda_matches_compact_baseline():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return
    if not torch.cuda.is_bf16_supported():
        return

    torch.manual_seed(8)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.03)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.03)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.03)

    hidden = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.03
    old_values = {
        "_CFG_SHARED_ROUTE_DECODE": qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE,
        "_CFG_ROUTE_MATRIX_DECODE": qwen3_moe_kernel._CFG_ROUTE_MATRIX_DECODE,
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE,
        "_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK,
        "_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE,
        "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST,
        "_CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM,
        "_CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT,
        "_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N,
        "_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N,
        "_CFG_EXPERT_GROUPED_MIN_ROWS": qwen3_moe_kernel._CFG_EXPERT_GROUPED_MIN_ROWS,
        "_CFG_EXPERT_GROUPED_MAX_ROWS": qwen3_moe_kernel._CFG_EXPERT_GROUPED_MAX_ROWS,
    }
    try:
        qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE = False
        qwen3_moe_kernel._CFG_ROUTE_MATRIX_DECODE = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = 64
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = 128
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_MIN_ROWS = 1
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_MAX_ROWS = 16

        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)

            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = False
            baseline = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=torch.empty_like(hidden),
                workspace={},
            ).clone()

            workspace = {}
            out = torch.empty_like(hidden)
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = True
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
    finally:
        for name, value in old_values.items():
            setattr(qwen3_moe_kernel, name, value)

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert workspace.get("expert_grouped_compact_decode_last_direct_out") == 1
    assert workspace.get("expert_grouped_compact_decode_last_gate_block_n") == 64
    assert workspace.get("expert_grouped_compact_decode_last_down_block_n") == 128
    max_abs = (actual.float() - baseline.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        baseline.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 2e-2
    assert cosine > 0.999


def test_qwen3_moe_single_row_gemv_bf16_cuda_matches_reference():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return
    if not torch.cuda.is_bf16_supported():
        return

    torch.manual_seed(10)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.03)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.03)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.03)

    hidden = torch.randn(1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.03
    names = (
        "_CFG_SHARED_ROUTE_DECODE",
        "_CFG_SHARED_ROUTE_BLOCK_M",
        "_CFG_SINGLE_ROW_GEMV",
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE",
        "_CFG_GROUPED_FUSED_GATE",
        "_CFG_GROUPED_DOT",
        "_CFG_TOKEN_ACCUM",
        "_CFG_TOKEN_ACCUM_MIN_ROWS",
    )
    old_values = {name: getattr(qwen3_moe_kernel, name) for name in names}
    try:
        qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE = True
        qwen3_moe_kernel._CFG_SHARED_ROUTE_BLOCK_M = 16
        qwen3_moe_kernel._CFG_SINGLE_ROW_GEMV = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
        qwen3_moe_kernel._CFG_GROUPED_FUSED_GATE = True
        qwen3_moe_kernel._CFG_GROUPED_DOT = False
        qwen3_moe_kernel._CFG_TOKEN_ACCUM = True
        qwen3_moe_kernel._CFG_TOKEN_ACCUM_MIN_ROWS = 1
        workspace = {}
        out = torch.empty_like(hidden)
        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
            expected = moe.experts(
                hidden,
                selected_experts,
                routing_weights,
                use_grouped_decode=False,
            )
    finally:
        for name, value in old_values.items():
            setattr(qwen3_moe_kernel, name, value)

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert "activated" in workspace
    assert "shared_route_activated" not in workspace
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        expected.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 2e-2
    assert cosine > 0.999


def test_qwen3_moe_shared_route_coalesced_bf16_cuda_matches_reference():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return
    if not torch.cuda.is_bf16_supported():
        return

    torch.manual_seed(11)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.03)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.03)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.03)

    hidden = torch.randn(1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.03
    names = (
        "_CFG_SHARED_ROUTE_DECODE",
        "_CFG_SHARED_ROUTE_BLOCK_M",
        "_CFG_SHARED_ROUTE_GATE_BLOCK_N",
        "_CFG_SHARED_ROUTE_GATE_K_SPLITS",
        "_CFG_SHARED_ROUTE_DOWN_BLOCK_N",
        "_CFG_SHARED_ROUTE_SPLIT_GATE",
        "_CFG_SHARED_ROUTE_SPLIT_GATE_BLOCK_M",
        "_CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES",
        "_CFG_SHARED_ROUTE_PARTIAL_REDUCE",
        "_CFG_SHARED_ROUTE_COALESCED_WEIGHTS",
        "_CFG_SHARED_ROUTE_TOKEN_ACCUM",
        "_CFG_SINGLE_ROW_GEMV",
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE",
        "_CFG_GROUPED_FUSED_GATE",
        "_CFG_GROUPED_DOT",
        "_CFG_TOKEN_ACCUM",
        "_CFG_TOKEN_ACCUM_MIN_ROWS",
        "_CFG_BLOCK_N",
        "_CFG_BLOCK_K",
        "_CFG_NUM_WARPS",
    )
    old_values = {name: getattr(qwen3_moe_kernel, name) for name in names}
    try:
        qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE = True
        qwen3_moe_kernel._CFG_SHARED_ROUTE_BLOCK_M = 1
        qwen3_moe_kernel._CFG_SHARED_ROUTE_GATE_BLOCK_N = 32
        qwen3_moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
        qwen3_moe_kernel._CFG_SHARED_ROUTE_DOWN_BLOCK_N = 64
        qwen3_moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = False
        qwen3_moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE_BLOCK_M = 16
        qwen3_moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES = 4
        qwen3_moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = True
        qwen3_moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
        qwen3_moe_kernel._CFG_SINGLE_ROW_GEMV = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
        qwen3_moe_kernel._CFG_GROUPED_FUSED_GATE = True
        qwen3_moe_kernel._CFG_GROUPED_DOT = False
        qwen3_moe_kernel._CFG_TOKEN_ACCUM = True
        qwen3_moe_kernel._CFG_TOKEN_ACCUM_MIN_ROWS = 1
        qwen3_moe_kernel._CFG_BLOCK_N = 32
        qwen3_moe_kernel._CFG_BLOCK_K = 32
        qwen3_moe_kernel._CFG_NUM_WARPS = 4

        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)
            expected = moe.experts(
                hidden,
                selected_experts,
                routing_weights,
                use_grouped_decode=False,
            )
            for coalesced_weights in (False, True):
                qwen3_moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = coalesced_weights
                for partial_reduce in (False, True):
                    qwen3_moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = partial_reduce
                    workspace = {}
                    out = torch.empty_like(hidden)
                    actual = qwen3_moe_grouped_decode(
                        hidden,
                        moe.experts.gate_up_proj,
                        moe.experts.down_proj,
                        selected_experts,
                        routing_weights,
                        activation=cfg.hidden_act,
                        out=out,
                        workspace=workspace,
                    )
                    torch.cuda.synchronize()
                    assert actual.data_ptr() == out.data_ptr()
                    assert workspace.get("shared_route_decode_last_partial_reduce") == int(
                        partial_reduce
                    )
                    max_abs = (actual.float() - expected.float()).abs().max().item()
                    cosine = torch.nn.functional.cosine_similarity(
                        actual.flatten().float(),
                        expected.flatten().float(),
                        dim=0,
                    ).item()
                    assert max_abs < 2e-2
                    assert cosine > 0.999

            qwen3_moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 2
            qwen3_moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
            qwen3_moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = False
            qwen3_moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
            workspace = {}
            out = torch.empty_like(hidden)
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
            torch.cuda.synchronize()
            assert actual.data_ptr() == out.data_ptr()
            assert workspace.get("shared_route_decode_last_gate_k_splits") == 2
            max_abs = (actual.float() - expected.float()).abs().max().item()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten().float(),
                expected.flatten().float(),
                dim=0,
            ).item()
            assert max_abs < 2e-2
            assert cosine > 0.999

            qwen3_moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
            qwen3_moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
            qwen3_moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = True
            qwen3_moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = True
            workspace = {}
            out = torch.empty_like(hidden)
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
            torch.cuda.synchronize()
            assert actual.data_ptr() == out.data_ptr()
            assert workspace.get("shared_route_decode_last_token_accum") == 1
            max_abs = (actual.float() - expected.float()).abs().max().item()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten().float(),
                expected.flatten().float(),
                dim=0,
            ).item()
            assert max_abs < 2e-2
            assert cosine > 0.999

            qwen3_moe_kernel._CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
            qwen3_moe_kernel._CFG_SHARED_ROUTE_SPLIT_GATE = True
            qwen3_moe_kernel._CFG_SHARED_ROUTE_PARTIAL_REDUCE = False
            qwen3_moe_kernel._CFG_SHARED_ROUTE_COALESCED_WEIGHTS = False
            qwen3_moe_kernel._CFG_SHARED_ROUTE_TOKEN_ACCUM = False
            workspace = {}
            out = torch.empty_like(hidden)
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
            torch.cuda.synchronize()
            assert actual.data_ptr() == out.data_ptr()
            assert workspace.get("shared_route_decode_last_split_gate") == 1
            max_abs = (actual.float() - expected.float()).abs().max().item()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten().float(),
                expected.flatten().float(),
                dim=0,
            ).item()
            assert max_abs < 2e-2
            assert cosine > 0.999
    finally:
        for name, value in old_values.items():
            setattr(qwen3_moe_kernel, name, value)


def test_qwen3_moe_compact_active_list_cuda_matches_compact_baseline():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(9)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=32,
            num_experts_per_tok=4,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.float32).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn_like(moe.gate.weight) * 0.03)
        moe.experts.gate_up_proj.copy_(torch.randn_like(moe.experts.gate_up_proj) * 0.03)
        moe.experts.down_proj.copy_(torch.randn_like(moe.experts.down_proj) * 0.03)

    hidden = torch.randn(8, cfg.hidden_size, device="cuda", dtype=torch.float32) * 0.03
    old_values = {
        "_CFG_SHARED_ROUTE_DECODE": qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE,
        "_CFG_ROUTE_MATRIX_DECODE": qwen3_moe_kernel._CFG_ROUTE_MATRIX_DECODE,
        "_CFG_EXPERT_GROUPED_COMPACT_DECODE": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE,
        "_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK,
        "_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE,
        "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST,
        "_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT,
        "_CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK,
        "_CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID,
        "_CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM,
        "_CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT,
        "_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N,
        "_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N": qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N,
        "_CFG_EXPERT_GROUPED_MIN_ROWS": qwen3_moe_kernel._CFG_EXPERT_GROUPED_MIN_ROWS,
        "_CFG_EXPERT_GROUPED_MAX_ROWS": qwen3_moe_kernel._CFG_EXPERT_GROUPED_MAX_ROWS,
    }
    try:
        qwen3_moe_kernel._CFG_SHARED_ROUTE_DECODE = False
        qwen3_moe_kernel._CFG_ROUTE_MATRIX_DECODE = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DECODE = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK = True
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = False
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = 64
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = 128
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_MIN_ROWS = 1
        qwen3_moe_kernel._CFG_EXPERT_GROUPED_MAX_ROWS = 16

        with torch.inference_mode():
            _, routing_weights, selected_experts = moe.gate(hidden)

            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = False
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = False
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID = False
            baseline = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=torch.empty_like(hidden),
                workspace={},
            ).clone()

            workspace = {}
            out = torch.empty_like(hidden)
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = True
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = True
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID = True
            actual = qwen3_moe_grouped_decode(
                hidden,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                selected_experts,
                routing_weights,
                activation=cfg.hidden_act,
                out=out,
                workspace=workspace,
            )
    finally:
        for name, value in old_values.items():
            setattr(qwen3_moe_kernel, name, value)

    torch.cuda.synchronize()
    assert actual.data_ptr() == out.data_ptr()
    assert workspace.get("expert_grouped_compact_decode_last_active_list") == 1
    assert (
        workspace.get(
            "expert_grouped_compact_decode_last_active_list_early_exit"
        )
        == 1
    )
    assert workspace.get("expert_grouped_compact_decode_last_expert_grid_pack") == 1
    assert workspace.get("expert_grouped_compact_decode_last_l2_grouped_grid") == 1
    assert workspace.get("expert_grouped_compact_decode_last_gate_block_n") == 64
    assert workspace.get("expert_grouped_compact_decode_last_down_block_n") == 128
    max_abs = (actual - baseline).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(),
        baseline.flatten().float(),
        dim=0,
    ).item()
    assert max_abs < 1e-4
    assert cosine > 0.99999


def test_qwen3_moe_partial_reduce_cuda_is_repeatable_for_batch_paths():
    if not torch.cuda.is_available() or not HAS_QWEN3_MOE_GROUPED:
        return

    torch.manual_seed(17)
    cfg = LlamaConfig.from_dict(
        _qwen3_moe_config(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=4,
        )
    )
    moe = Qwen3MoeMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    hidden = torch.randn(
        8,
        cfg.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.03)

    with torch.inference_mode():
        _, routing_weights, selected_experts = moe.gate(hidden)

        assignment_workspace = {}
        assignment_first = qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            workspace=assignment_workspace,
            expert_grouped_compact=False,
            assignment_partial_reduce=True,
        ).clone()
        assignment_second = qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            workspace={},
            expert_grouped_compact=False,
            assignment_partial_reduce=True,
        ).clone()

        compact_workspace = {}
        compact_first = qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            workspace=compact_workspace,
            expert_grouped_compact=True,
            expert_grouped_min_rows=1,
            expert_grouped_max_rows=16,
            expert_grouped_compact_partial_reduce=True,
        ).clone()
        compact_second = qwen3_moe_grouped_decode(
            hidden,
            moe.experts.gate_up_proj,
            moe.experts.down_proj,
            selected_experts,
            routing_weights,
            activation=cfg.hidden_act,
            workspace={},
            expert_grouped_compact=True,
            expert_grouped_min_rows=1,
            expert_grouped_max_rows=16,
            expert_grouped_compact_partial_reduce=True,
        ).clone()

    torch.cuda.synchronize()
    assert assignment_workspace.get("grouped_decode_last_partial_reduce") == 1
    assert compact_workspace.get("grouped_decode_last_path") == "expert_grouped_compact"
    assert compact_workspace.get(
        "expert_grouped_compact_decode_last_partial_reduce"
    ) == 1
    assert torch.equal(assignment_first, assignment_second)
    assert torch.equal(compact_first, compact_second)
    cosine = torch.nn.functional.cosine_similarity(
        compact_first.flatten().float(),
        assignment_first.flatten().float(),
        dim=0,
    ).item()
    assert cosine > 0.999


def test_qwen3_moe_loader_mapping_keeps_expert_tensors_3d():
    cfg = LlamaConfig.from_dict(_qwen3_moe_config(num_hidden_layers=1))
    hf = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.randn(cfg.hidden_size),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(cfg.head_dim),
        "model.layers.0.self_attn.k_norm.weight": torch.randn(cfg.head_dim),
        "model.layers.0.mlp.gate.weight": torch.randn(cfg.num_experts, cfg.hidden_size),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            cfg.num_experts,
            2 * cfg.moe_intermediate_size,
            cfg.hidden_size,
        ),
        "model.layers.0.mlp.experts.down_proj": torch.randn(
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        ),
        "model.layers.0.input_layernorm.weight": torch.randn(cfg.hidden_size),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(cfg.hidden_size),
    }

    mapped = _map_weights(hf, cfg)

    assert "layers.0.mlp.gate.weight" in mapped
    assert "layers.0.mlp.experts.gate_up_proj" in mapped
    assert "layers.0.mlp.experts.down_proj" in mapped
    assert "layers.0.mlp.gate_up_proj.weight" not in mapped
    assert mapped["layers.0.mlp.experts.gate_up_proj"].shape == (
        cfg.num_experts,
        2 * cfg.moe_intermediate_size,
        cfg.hidden_size,
    )


def test_qwen3_moe_loader_mapping_stacks_split_hf_experts():
    cfg = LlamaConfig.from_dict(_qwen3_moe_config(num_hidden_layers=1, num_experts=3))
    hf = _tiny_qwen3_moe_hf_weights(cfg, split_experts=True)

    mapped = _map_weights(hf, cfg)

    gate_up = mapped["layers.0.mlp.experts.gate_up_proj"]
    down = mapped["layers.0.mlp.experts.down_proj"]

    assert gate_up.shape == (cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    assert down.shape == (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    assert torch.equal(
        gate_up[1, : cfg.moe_intermediate_size],
        torch.full((cfg.moe_intermediate_size, cfg.hidden_size), 11.0),
    )
    assert torch.equal(
        gate_up[1, cfg.moe_intermediate_size :],
        torch.full((cfg.moe_intermediate_size, cfg.hidden_size), 21.0),
    )
    assert torch.equal(
        down[1],
        torch.full((cfg.hidden_size, cfg.moe_intermediate_size), 31.0),
    )


def test_qwen3_moe_streaming_load_stacks_split_hf_experts():
    cfg = LlamaConfig.from_dict(_qwen3_moe_config(num_hidden_layers=1, num_experts=3))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(_tiny_qwen3_moe_hf_weights(cfg, split_experts=True), os.path.join(tmpdir, "model.safetensors"))

        with torch.device("meta"):
            model = MegaGemmLlama(cfg)

        _load_fp16_streaming(model, cfg, tmpdir, torch.float32, "cpu")

    assert not any(param.device.type == "meta" for param in model.parameters())
    gate_up = model.layers[0].mlp.experts.gate_up_proj
    down = model.layers[0].mlp.experts.down_proj

    assert torch.equal(
        gate_up[2, : cfg.moe_intermediate_size],
        torch.full((cfg.moe_intermediate_size, cfg.hidden_size), 12.0),
    )
    assert torch.equal(
        gate_up[2, cfg.moe_intermediate_size :],
        torch.full((cfg.moe_intermediate_size, cfg.hidden_size), 22.0),
    )
    assert torch.equal(
        down[2],
        torch.full((cfg.hidden_size, cfg.moe_intermediate_size), 32.0),
    )


def _tiny_qwen3_moe_hf_weights(cfg, *, split_experts):
    weights = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.randn(cfg.hidden_size),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
        ),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
        ),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim
        ),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(cfg.head_dim),
        "model.layers.0.self_attn.k_norm.weight": torch.randn(cfg.head_dim),
        "model.layers.0.mlp.gate.weight": torch.randn(cfg.num_experts, cfg.hidden_size),
        "model.layers.0.input_layernorm.weight": torch.randn(cfg.hidden_size),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(cfg.hidden_size),
    }

    if split_experts:
        for expert_idx in range(cfg.num_experts):
            expert_pre = f"model.layers.0.mlp.experts.{expert_idx}"
            weights[f"{expert_pre}.gate_proj.weight"] = torch.full(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                10.0 + expert_idx,
            )
            weights[f"{expert_pre}.up_proj.weight"] = torch.full(
                (cfg.moe_intermediate_size, cfg.hidden_size),
                20.0 + expert_idx,
            )
            weights[f"{expert_pre}.down_proj.weight"] = torch.full(
                (cfg.hidden_size, cfg.moe_intermediate_size),
                30.0 + expert_idx,
            )
    else:
        weights["model.layers.0.mlp.experts.gate_up_proj"] = torch.randn(
            cfg.num_experts,
            2 * cfg.moe_intermediate_size,
            cfg.hidden_size,
        )
        weights["model.layers.0.mlp.experts.down_proj"] = torch.randn(
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        )

    return weights
