import os
import sys

import torch
import megagemm.models.llama as llama_mod
import megagemm.kernels.linear_attention as linear_attn_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megagemm.engine.scheduler import Scheduler
from megagemm.engine.kv_cache import BlockManager
from megagemm.kernels.rope import apply_rotary_emb, precompute_freqs_cis
from megagemm.models.llama import (
    GatedDeltaNet,
    LlamaAttention,
    LlamaConfig,
    MGRMSNorm,
    MegaGemmLlama,
    RMSNormGated,
    _fused_linear_gates,
    recurrent_gated_delta_decode_step,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    recurrent_gated_delta_prefill_short_sequence,
)
from megagemm.models.loader import (
    _is_text_backbone_weight,
    _normalize_hf_weight_key,
    _validate_supported_architecture,
)
from megagemm.kernels.linear_attention import (
    HAS_TRITON_LINEAR_ATTN,
    _validate_qkv_group_shapes,
    chunk_interchunk,
    chunk_interchunk_scan,
    chunk_state_projection,
    chunk_state_update,
    recurrent_gated_delta_decode,
    recurrent_gated_delta_prefill,
    solve_chunk_local_attention,
)
from megagemm.kernels.rmsnorm_gated import HAS_RMSNORM_GATED, rmsnorm_gated
from megagemm.kernels.rmsnorm_triton import HAS_TRITON_RMSNORM, rmsnorm_triton


def _qwen35_text_config(**overrides):
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 32000,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": False,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.5,
            "full_attention_interval": 2,
            "hidden_act": "silu",
        },
    }
    config["text_config"].update(overrides)
    return config


def test_qwen35_config_parsing():
    cfg = LlamaConfig.from_dict(_qwen35_text_config(layer_types=["full_attention"] * 4))

    assert cfg.model_type == "qwen3_5_text"
    assert cfg.qk_norm is True
    assert cfg.qk_norm_offset is True
    assert cfg.norm_offset is True
    assert cfg.attention_output_gate is True
    assert cfg.rope_half_rotate is True
    assert cfg.rotary_dim == 8


def test_qwen35_partial_rope_keeps_tail():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    cos, sin = precompute_freqs_cis(4, 3)

    q_rot, k_rot = apply_rotary_emb(
        q, k, cos, sin,
        half_rotate=True,
        rotary_dim=4,
    )

    assert torch.allclose(q_rot[..., 4:], q[..., 4:])
    assert torch.allclose(k_rot[..., 4:], k[..., 4:])
    assert not torch.allclose(q_rot[..., :4], q[..., :4])
    assert not torch.allclose(k_rot[..., :4], k[..., :4])


def test_qwen35_runtime_policy_cpu_defaults():
    threshold, chunk_scan = llama_mod._resolve_qwen35_runtime_policy(torch.device("cpu"))
    assert threshold == 160
    assert chunk_scan is False


def test_qwen35_runtime_policy_env_overrides_take_priority(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_QWEN35_SHORT_PREFILL_THRESHOLD", "123")
    monkeypatch.setenv("MEGAGEMM_QWEN35_ENABLE_CHUNK_SCAN", "0")
    cfg = LlamaConfig.from_dict(_qwen35_text_config(layer_types=["linear_attention"] * 4))
    block = GatedDeltaNet(cfg, layer_idx=0)
    threshold, chunk_scan = block._runtime_policy(torch.device("cpu"))
    assert threshold == 123
    assert chunk_scan is False


def test_qwen35_chunk_scan_launch_policy_env_override_and_auto_bounds(monkeypatch):
    query = torch.randn(1, 1, 4, 8, 16)  # [batch, heads, num_chunks, chunk_len, key_dim]
    max_chunks, warps = linear_attn_mod._resolve_chunk_scan_launch_policy(
        query, num_chunks=4, key_dim=16, value_dim=16,
    )
    assert max_chunks == 4
    assert 1 <= warps <= 8

    monkeypatch.setenv("MEGAGEMM_QWEN35_SCAN_MAX_CHUNKS", "20")
    monkeypatch.setenv("MEGAGEMM_QWEN35_SCAN_NUM_WARPS", "6")
    max_chunks, warps = linear_attn_mod._resolve_chunk_scan_launch_policy(
        query, num_chunks=4, key_dim=16, value_dim=16,
    )
    assert max_chunks == 20
    assert warps == 6


def test_qwen35_chunk_scan_window_balances_windows_and_tail():
    assert linear_attn_mod._choose_scan_window_chunks(24, 24) == 24
    assert linear_attn_mod._choose_scan_window_chunks(32, 24) == 16
    assert linear_attn_mod._choose_scan_window_chunks(40, 24) == 20
    assert linear_attn_mod._choose_scan_window_chunks(33, 24) == 20
    assert linear_attn_mod._choose_scan_window_chunks(41, 24) == 24


def test_qwen35_chunk_scan_block_c_bucket_stable():
    assert linear_attn_mod._bucket_scan_block_c(1) == 8
    assert linear_attn_mod._bucket_scan_block_c(8) == 8
    assert linear_attn_mod._bucket_scan_block_c(9) == 16
    assert linear_attn_mod._bucket_scan_block_c(17) == 24
    assert linear_attn_mod._bucket_scan_block_c(33) == 48
    assert linear_attn_mod._bucket_scan_block_c(80) == 64


def test_qwen35_parallel_scan_guard_cpu_disabled_without_force(monkeypatch):
    q = torch.randn(1, 1, 32, 8, 128)
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN", "1")
    monkeypatch.delenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", raising=False)
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is False
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", "1")
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is True


def test_qwen35_parallel_scan_guard_cuda_thresholds(monkeypatch):
    class _CudaLike:
        is_cuda = True

    q = _CudaLike()
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN", "1")
    monkeypatch.delenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", raising=False)

    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=15, key_dim=32) is False
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is False
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=32) is True

    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_ALGO", "blelloch")
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=31, key_dim=128) is False
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is False


def test_qwen35_parallel_scan_guard_cuda_triton_opt_in_requires_force_for_wide_heads(monkeypatch):
    class _CudaLike:
        is_cuda = True

    q = _CudaLike()
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN", "1")
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_TRITON", "1")
    monkeypatch.delenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", raising=False)
    monkeypatch.setattr(linear_attn_mod, "_HAS_TRITON", True)
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is False
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", "1")
    assert linear_attn_mod._parallel_scan_allowed(q, num_chunks=32, key_dim=128) is True


def test_qwen35_attention_full_path_runs():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(layer_types=["full_attention"] * 4))
    attn = LlamaAttention(cfg, layer_idx=0)

    hidden_states = torch.randn(1, 3, cfg.hidden_size)
    positions = torch.arange(3).unsqueeze(0)
    cos, sin = precompute_freqs_cis(cfg.rotary_dim, cfg.max_position_embeddings)

    output, k_cache, v_cache = attn(
        hidden_states,
        cos,
        sin,
        positions,
        is_prefill=True,
    )

    assert output.shape == (1, 3, cfg.hidden_size)
    assert k_cache.shape == (1, 3, cfg.num_key_value_heads, cfg.head_dim)
    assert v_cache.shape == (1, 3, cfg.num_key_value_heads, cfg.head_dim)


def test_qwen35_partial_rope_fused_decode_runs_if_available():
    if not torch.cuda.is_available() or not getattr(llama_mod, "_HAS_FUSED_ROPE_ATTN", False):
        return

    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(layer_types=["full_attention"] * 4))
    attn = LlamaAttention(cfg, layer_idx=0).cuda().half()

    hidden_states = torch.randn(1, 1, cfg.hidden_size, device="cuda", dtype=torch.float16)
    positions = torch.zeros(1, 1, device="cuda", dtype=torch.long)
    cos, sin = precompute_freqs_cis(
        cfg.rotary_dim, cfg.max_position_embeddings, cfg.rope_theta, device=torch.device("cuda")
    )
    kv_cache = torch.zeros(
        4, 2, cfg.num_key_value_heads, 16, cfg.head_dim, device="cuda", dtype=torch.float16
    )
    block_table = torch.zeros(1, 4, device="cuda", dtype=torch.long)
    seq_lens = torch.zeros(1, device="cuda", dtype=torch.long)

    original_decode = llama_mod.paged_attention_decode

    def _fallback_decode_should_not_run(*args, **kwargs):
        raise AssertionError("partial-rope fused decode unexpectedly fell back")

    llama_mod.paged_attention_decode = _fallback_decode_should_not_run
    try:
        output, k_cache, v_cache = attn(
            hidden_states,
            cos,
            sin,
            positions,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            is_prefill=False,
        )
    finally:
        llama_mod.paged_attention_decode = original_decode

    assert output.shape == (1, 1, cfg.hidden_size)
    assert k_cache.shape == (1, 1, cfg.num_key_value_heads, cfg.head_dim)
    assert v_cache.shape == (1, 1, cfg.num_key_value_heads, cfg.head_dim)


def test_qwen35_hybrid_attention_is_accepted():
    hf_config = _qwen35_text_config()
    cfg = LlamaConfig.from_dict(hf_config)
    _validate_supported_architecture(cfg, hf_config)


def test_qwen35_multimodal_text_only_is_accepted():
    hf_config = _qwen35_text_config()
    hf_config["vision_config"] = {"hidden_size": 128}
    cfg = LlamaConfig.from_dict(hf_config)
    _validate_supported_architecture(cfg, hf_config)


def test_qwen35_text_only_weight_filter_ignores_vision_tensors():
    assert _is_text_backbone_weight("model.embed_tokens.weight") is True
    assert _is_text_backbone_weight("model.language_model.embed_tokens.weight") is True
    assert _is_text_backbone_weight("model.layers.0.self_attn.q_proj.weight") is True
    assert _is_text_backbone_weight("model.language_model.layers.0.self_attn.q_proj.weight") is True
    assert _is_text_backbone_weight("model.norm.weight") is True
    assert _is_text_backbone_weight("model.language_model.norm.weight") is True
    assert _is_text_backbone_weight("lm_head.weight") is True
    assert _is_text_backbone_weight("visual.patch_embed.proj.weight") is False
    assert _is_text_backbone_weight("model.visual.blocks.0.attn.qkv.weight") is False


def test_qwen35_language_model_prefix_is_normalized():
    assert _normalize_hf_weight_key("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert _normalize_hf_weight_key(
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ) == "model.layers.3.self_attn.q_proj.weight"
    assert _normalize_hf_weight_key("lm_head.weight") == "lm_head.weight"


def test_qwen35_delta_rule_prefill_matches_recurrent_decode():
    torch.manual_seed(0)
    batch, seq_len, heads, key_dim, value_dim = 2, 5, 4, 8, 6
    query = torch.randn(batch, seq_len, heads, key_dim)
    key = torch.randn(batch, seq_len, heads, key_dim)
    value = torch.randn(batch, seq_len, heads, value_dim)
    gate = torch.randn(batch, seq_len, heads)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))

    chunk_out, chunk_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    recurrent_outs = []
    recurrent_state = None
    for idx in range(seq_len):
        step_out, recurrent_state = torch_recurrent_gated_delta_rule(
            query[:, idx : idx + 1],
            key[:, idx : idx + 1],
            value[:, idx : idx + 1],
            gate[:, idx : idx + 1],
            beta[:, idx : idx + 1],
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_outs.append(step_out)

    recurrent_out = torch.cat(recurrent_outs, dim=1)
    assert torch.allclose(chunk_out, recurrent_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(chunk_state, recurrent_state, atol=1e-3, rtol=1e-3)


def test_qwen35_recurrent_delta_rule_single_step_matches_manual():
    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 4, 8, 6
    query = torch.randn(batch, 1, heads, key_dim)
    key = torch.randn(batch, 1, heads, key_dim)
    value = torch.randn(batch, 1, heads, value_dim)
    gate = torch.randn(batch, 1, heads)
    beta = torch.sigmoid(torch.randn(batch, 1, heads))
    initial_state = torch.randn(batch, heads, key_dim, value_dim)

    out, next_state = torch_recurrent_gated_delta_rule(
        query, key, value, gate, beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    q = torch.nn.functional.normalize(query[:, 0].float(), dim=-1)
    k = torch.nn.functional.normalize(key[:, 0].float(), dim=-1)
    q = q * (1.0 / (key_dim ** 0.5))
    v = value[:, 0].float()
    g = gate[:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)
    b = beta[:, 0].float().unsqueeze(-1)
    state = initial_state.float() * g
    kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * b
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    manual_out = (state * q.unsqueeze(-1)).sum(dim=-2).unsqueeze(1).to(out.dtype)

    assert torch.allclose(out, manual_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(next_state, state, atol=1e-5, rtol=1e-5)


def test_qwen35_short_prefill_recurrent_matches_chunk():
    torch.manual_seed(0)
    batch, seq_len, heads, key_dim, value_dim = 2, 17, 4, 8, 6
    query = torch.randn(batch, seq_len, heads, key_dim)
    key = torch.randn(batch, seq_len, heads, key_dim)
    value = torch.randn(batch, seq_len, heads, value_dim)
    gate = torch.randn(batch, seq_len, heads)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))

    short_out, short_state = recurrent_gated_delta_prefill_short_sequence(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    chunk_out, chunk_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.allclose(short_out, chunk_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(short_state, chunk_state, atol=1e-4, rtol=1e-4)


def test_qwen35_chunk_rule_non_multiple_chunk_matches_recurrent():
    torch.manual_seed(0)
    batch, seq_len, heads, key_dim, value_dim = 2, 73, 4, 8, 6
    query = torch.randn(batch, seq_len, heads, key_dim)
    key = torch.randn(batch, seq_len, heads, key_dim)
    value = torch.randn(batch, seq_len, heads, value_dim)
    gate = torch.randn(batch, seq_len, heads)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))

    chunk_out, chunk_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        chunk_size=64,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    recurrent_outs = []
    recurrent_state = None
    for idx in range(seq_len):
        step_out, recurrent_state = torch_recurrent_gated_delta_rule(
            query[:, idx : idx + 1],
            key[:, idx : idx + 1],
            value[:, idx : idx + 1],
            gate[:, idx : idx + 1],
            beta[:, idx : idx + 1],
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_outs.append(step_out)

    recurrent_out = torch.cat(recurrent_outs, dim=1)
    assert torch.allclose(chunk_out, recurrent_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(chunk_state, recurrent_state, atol=1e-3, rtol=1e-3)


def test_qwen35_long_prefill_recurrent_matches_chunk_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, seq_len, heads, key_dim, value_dim = 2, 73, 4, 8, 6
    query = torch.randn(batch, seq_len, heads, key_dim, device='cuda', dtype=torch.float16)
    key = torch.randn(batch, seq_len, heads, key_dim, device='cuda', dtype=torch.float16)
    value = torch.randn(batch, seq_len, heads, value_dim, device='cuda', dtype=torch.float16)
    gate = torch.randn(batch, seq_len, heads, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads, device='cuda', dtype=torch.float32))

    recurrent_out, recurrent_state = recurrent_gated_delta_prefill_short_sequence(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    chunk_out, chunk_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    assert torch.allclose(recurrent_out, chunk_out, atol=3e-3, rtol=3e-3)
    assert torch.allclose(recurrent_state, chunk_state, atol=3e-3, rtol=3e-3)


def test_qwen35_chunk_interchunk_scan_matches_stepwise():
    torch.manual_seed(0)
    batch, heads, num_chunks, chunk_len, key_dim, value_dim = 2, 3, 4, 16, 8, 6
    query = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key_cumdecay = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    value = torch.randn(batch, heads, num_chunks, chunk_len, value_dim)
    gate = torch.randn(batch, heads, num_chunks, chunk_len)
    state = torch.randn(batch, heads, key_dim, value_dim)

    scan_value_new, scan_attn_inter, scan_state = chunk_interchunk_scan(
        query,
        key,
        key_cumdecay,
        value,
        gate,
        state.clone(),
    )

    ref_value_new = torch.empty_like(value)
    ref_attn_inter = torch.empty_like(value)
    ref_state = state.clone()
    for idx in range(num_chunks):
        ref_value_new[:, :, idx], ref_attn_inter[:, :, idx], ref_state = chunk_interchunk(
            query[:, :, idx],
            key[:, :, idx],
            key_cumdecay[:, :, idx],
            value[:, :, idx],
            gate[:, :, idx],
            ref_state,
        )

    assert torch.allclose(scan_value_new, ref_value_new, atol=1e-5, rtol=1e-5)
    assert torch.allclose(scan_attn_inter, ref_attn_inter, atol=1e-5, rtol=1e-5)
    assert torch.allclose(scan_state, ref_state, atol=1e-5, rtol=1e-5)


def test_qwen35_chunk_interchunk_scan_large_chunk_count_matches_stepwise_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, num_chunks, chunk_len, key_dim, value_dim = 1, 2, 20, 16, 8, 6
    query = torch.randn(batch, heads, num_chunks, chunk_len, key_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, heads, num_chunks, chunk_len, key_dim, device="cuda", dtype=torch.float16)
    key_cumdecay = torch.randn(batch, heads, num_chunks, chunk_len, key_dim, device="cuda", dtype=torch.float16)
    value = torch.randn(batch, heads, num_chunks, chunk_len, value_dim, device="cuda", dtype=torch.float16)
    gate = torch.randn(batch, heads, num_chunks, chunk_len, device="cuda", dtype=torch.float16)
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=torch.float16)

    scan_value_new, scan_attn_inter, scan_state = chunk_interchunk_scan(
        query,
        key,
        key_cumdecay,
        value,
        gate,
        state.clone(),
    )

    ref_value_new = torch.empty_like(scan_value_new)
    ref_attn_inter = torch.empty_like(scan_attn_inter)
    ref_state = state.clone()
    for idx in range(num_chunks):
        ref_value_new[:, :, idx], ref_attn_inter[:, :, idx], ref_state = chunk_interchunk(
            query[:, :, idx],
            key[:, :, idx],
            key_cumdecay[:, :, idx],
            value[:, :, idx],
            gate[:, :, idx],
            ref_state,
        )
    torch.cuda.synchronize()

    assert torch.allclose(scan_value_new, ref_value_new, atol=3e-3, rtol=3e-3)
    assert torch.allclose(scan_attn_inter, ref_attn_inter, atol=3e-3, rtol=3e-3)
    assert torch.allclose(scan_state, ref_state, atol=3e-3, rtol=3e-3)


def test_qwen35_chunk_interchunk_parallel_scan_matches_stepwise():
    torch.manual_seed(0)
    batch, heads, num_chunks, chunk_len, key_dim, value_dim = 1, 2, 6, 8, 8, 6
    query = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key_cumdecay = torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    value = torch.randn(batch, heads, num_chunks, chunk_len, value_dim)
    gate = torch.randn(batch, heads, num_chunks, chunk_len)
    state = torch.randn(batch, heads, key_dim, value_dim)

    par_value_new, par_attn_inter, par_state = linear_attn_mod._chunk_interchunk_scan_parallel_torch(
        query,
        key,
        key_cumdecay,
        value,
        gate,
        state.clone(),
    )

    ref_value_new = torch.empty_like(par_value_new)
    ref_attn_inter = torch.empty_like(par_attn_inter)
    ref_state = state.clone()
    for idx in range(num_chunks):
        ref_value_new[:, :, idx], ref_attn_inter[:, :, idx], ref_state = chunk_interchunk(
            query[:, :, idx],
            key[:, :, idx],
            key_cumdecay[:, :, idx],
            value[:, :, idx],
            gate[:, :, idx],
            ref_state,
        )

    assert torch.allclose(par_value_new, ref_value_new, atol=2e-4, rtol=2e-4)
    assert torch.allclose(par_attn_inter, ref_attn_inter, atol=2e-4, rtol=2e-4)
    assert torch.allclose(par_state, ref_state, atol=2e-4, rtol=2e-4)


def test_qwen35_affine_blelloch_scan_matches_reference():
    torch.manual_seed(0)
    bh, num_chunks, key_dim, value_dim = 2, 5, 4, 3
    eye = torch.eye(key_dim).view(1, 1, key_dim, key_dim)
    A = eye + 0.02 * torch.randn(bh, num_chunks, key_dim, key_dim)
    B = 0.02 * torch.randn(bh, num_chunks, key_dim, value_dim)

    got_A, got_B = linear_attn_mod._prefix_scan_affine_blelloch_torch(A, B)

    ref_A = torch.empty_like(got_A)
    ref_B = torch.empty_like(got_B)
    cur_A = A[:, 0]
    cur_B = B[:, 0]
    ref_A[:, 0] = cur_A
    ref_B[:, 0] = cur_B
    for idx in range(1, num_chunks):
        cur_A, cur_B = linear_attn_mod._compose_affine(
            A[:, idx],
            B[:, idx],
            cur_A,
            cur_B,
        )
        ref_A[:, idx] = cur_A
        ref_B[:, idx] = cur_B

    assert torch.allclose(got_A, ref_A, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got_B, ref_B, atol=1e-5, rtol=1e-5)


def test_qwen35_chunk_interchunk_blelloch_scan_matches_stepwise(monkeypatch):
    torch.manual_seed(1)
    monkeypatch.setenv("MEGAGEMM_QWEN35_PARALLEL_SCAN_ALGO", "blelloch")
    batch, heads, num_chunks, chunk_len, key_dim, value_dim = 1, 2, 7, 8, 8, 6
    scale = 0.1
    query = scale * torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key = scale * torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    key_cumdecay = scale * torch.randn(batch, heads, num_chunks, chunk_len, key_dim)
    value = scale * torch.randn(batch, heads, num_chunks, chunk_len, value_dim)
    gate = scale * torch.randn(batch, heads, num_chunks, chunk_len)
    state = scale * torch.randn(batch, heads, key_dim, value_dim)

    par_value_new, par_attn_inter, par_state = linear_attn_mod._chunk_interchunk_scan_parallel_torch(
        query,
        key,
        key_cumdecay,
        value,
        gate,
        state.clone(),
    )

    ref_value_new = torch.empty_like(par_value_new)
    ref_attn_inter = torch.empty_like(par_attn_inter)
    ref_state = state.clone()
    for idx in range(num_chunks):
        ref_value_new[:, :, idx], ref_attn_inter[:, :, idx], ref_state = chunk_interchunk(
            query[:, :, idx],
            key[:, :, idx],
            key_cumdecay[:, :, idx],
            value[:, :, idx],
            gate[:, :, idx],
            ref_state,
        )

    assert torch.allclose(par_value_new, ref_value_new, atol=2e-4, rtol=2e-4)
    assert torch.allclose(par_attn_inter, ref_attn_inter, atol=2e-4, rtol=2e-4)
    assert torch.allclose(par_state, ref_state, atol=2e-4, rtol=2e-4)


def test_qwen35_chunk_state_projection_matches_reference():
    torch.manual_seed(0)
    query = torch.randn(2, 4, 16, 8)
    key_cumdecay = torch.randn(2, 4, 16, 8)
    value = torch.randn(2, 4, 16, 6)
    gate = torch.randn(2, 4, 16)
    state = torch.randn(2, 4, 8, 6)

    value_new, attn_inter = chunk_state_projection(
        query, key_cumdecay, value, gate, state.clone(),
    )
    ref_value_new = value - (key_cumdecay @ state)
    ref_attn_inter = (query * gate.exp().unsqueeze(-1)) @ state

    assert torch.allclose(value_new, ref_value_new, atol=1e-5, rtol=1e-5)
    assert torch.allclose(attn_inter, ref_attn_inter, atol=1e-5, rtol=1e-5)


def test_qwen35_chunk_state_update_matches_reference():
    torch.manual_seed(0)
    key = torch.randn(2, 4, 16, 8)
    gate = torch.randn(2, 4, 16)
    value_new = torch.randn(2, 4, 16, 6)
    state = torch.randn(2, 4, 8, 6)

    out = chunk_state_update(key, gate, value_new, state.clone())
    ref = state * gate[..., -1, None, None].exp()
    ref = ref + (key * (gate[..., -1, None] - gate).exp().unsqueeze(-1)).transpose(-1, -2) @ value_new

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_qwen35_qkv_group_validation_message():
    try:
        _validate_qkv_group_shapes(q_heads=4, kv_heads=15, num_kv_groups=4, op_name="unit_test")
        assert False, "Expected ValueError for incompatible q/kv grouping"
    except ValueError as exc:
        msg = str(exc)
        assert "unit_test" in msg
        assert "kv_heads=15" in msg
        assert "q_heads=4" in msg


def test_qwen35_triton_recurrent_decode_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 4, 8, 6
    query = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float16)
    key = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float16)
    value = torch.randn(batch, heads, value_dim, device='cuda', dtype=torch.float16)
    gate = torch.randn(batch, heads, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, heads, device='cuda', dtype=torch.float32))
    state = torch.randn(batch, heads, key_dim, value_dim, device='cuda', dtype=torch.float32)

    ref_state = state.clone()
    gate_t = gate.exp().unsqueeze(-1).unsqueeze(-1)
    kv_mem = ((ref_state * gate_t) * key.float().unsqueeze(-1)).sum(dim=-2)
    delta = (value.float() - kv_mem) * beta.unsqueeze(-1)
    ref_state = ref_state * gate_t + key.float().unsqueeze(-1) * delta.unsqueeze(-2)
    ref_out = (ref_state * query.float().unsqueeze(-1)).sum(dim=-2)

    out = recurrent_gated_delta_decode(
        query.float(), key.float(), value.float(), gate, beta, state,
    )

    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state, ref_state, atol=1e-4, rtol=1e-4)


def test_qwen35_triton_recurrent_decode_with_qk_l2norm_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 4, 8, 6
    query = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float32)
    key = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float32)
    value = torch.randn(batch, heads, value_dim, device='cuda', dtype=torch.float32)
    gate = torch.randn(batch, heads, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, heads, device='cuda', dtype=torch.float32))
    state = torch.randn(batch, heads, key_dim, value_dim, device='cuda', dtype=torch.float32)

    ref_state = state.clone()
    q = torch.nn.functional.normalize(query, dim=-1) * (1.0 / (key_dim ** 0.5))
    k = torch.nn.functional.normalize(key, dim=-1)
    gate_t = gate.exp().unsqueeze(-1).unsqueeze(-1)
    kv_mem = ((ref_state * gate_t) * k.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta.unsqueeze(-1)
    ref_state = ref_state * gate_t + k.unsqueeze(-1) * delta.unsqueeze(-2)
    ref_out = (ref_state * q.unsqueeze(-1)).sum(dim=-2)

    out = recurrent_gated_delta_decode(
        query.float(),
        key.float(),
        value.float(),
        gate,
        beta,
        state,
        query_scale=(1.0 / (key_dim ** 0.5)),
        normalize_qk=True,
    )
    torch.cuda.synchronize()

    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state, ref_state, atol=1e-4, rtol=1e-4)


def test_qwen35_triton_recurrent_decode_with_half_inputs_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 4, 8, 6
    query = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float16)
    key = torch.randn(batch, heads, key_dim, device='cuda', dtype=torch.float16)
    value = torch.randn(batch, heads, value_dim, device='cuda', dtype=torch.float16)
    gate = torch.randn(batch, heads, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, heads, device='cuda', dtype=torch.float32))
    state = torch.randn(batch, heads, key_dim, value_dim, device='cuda', dtype=torch.float32)

    ref_state = state.clone()
    q = torch.nn.functional.normalize(query.float(), dim=-1) * (1.0 / (key_dim ** 0.5))
    k = torch.nn.functional.normalize(key.float(), dim=-1)
    gate_t = gate.exp().unsqueeze(-1).unsqueeze(-1)
    kv_mem = ((ref_state * gate_t) * k.unsqueeze(-1)).sum(dim=-2)
    delta = (value.float() - kv_mem) * beta.unsqueeze(-1)
    ref_state = ref_state * gate_t + k.unsqueeze(-1) * delta.unsqueeze(-2)
    ref_out = (ref_state * q.unsqueeze(-1)).sum(dim=-2)

    out = recurrent_gated_delta_decode(
        query,
        key,
        value,
        gate,
        beta,
        state,
        query_scale=(1.0 / (key_dim ** 0.5)),
        normalize_qk=True,
    )
    torch.cuda.synchronize()

    assert torch.allclose(out, ref_out, atol=2e-3, rtol=2e-3)
    assert torch.allclose(state, ref_state, atol=2e-3, rtol=2e-3)


def test_qwen35_triton_recurrent_prefill_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, seq_len, key_dim, value_dim = 2, 4, 7, 8, 6
    query = torch.randn(batch, heads, seq_len, key_dim, device='cuda', dtype=torch.float32)
    key = torch.randn(batch, heads, seq_len, key_dim, device='cuda', dtype=torch.float32)
    value = torch.randn(batch, heads, seq_len, value_dim, device='cuda', dtype=torch.float32)
    gate = torch.randn(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, heads, seq_len, device='cuda', dtype=torch.float32))
    state = torch.randn(batch, heads, key_dim, value_dim, device='cuda', dtype=torch.float32)

    ref_state = state.clone()
    ref_outs = []
    for idx in range(seq_len):
        gate_t = gate[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, idx].unsqueeze(-1)
        ref_state = ref_state * gate_t
        kv_mem = (ref_state * key[:, :, idx].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, idx] - kv_mem) * beta_t
        ref_state = ref_state + key[:, :, idx].unsqueeze(-1) * delta.unsqueeze(-2)
        ref_outs.append((ref_state * query[:, :, idx].unsqueeze(-1)).sum(dim=-2))
    ref_out = torch.stack(ref_outs, dim=2)

    out = recurrent_gated_delta_prefill(
        query, key, value, gate, beta, state,
    )

    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state, ref_state, atol=1e-4, rtol=1e-4)


def test_qwen35_triton_recurrent_prefill_with_qk_l2norm_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    batch, heads, seq_len, key_dim, value_dim = 2, 4, 7, 8, 6
    query = torch.randn(batch, heads, seq_len, key_dim, device='cuda', dtype=torch.float32)
    key = torch.randn(batch, heads, seq_len, key_dim, device='cuda', dtype=torch.float32)
    value = torch.randn(batch, heads, seq_len, value_dim, device='cuda', dtype=torch.float32)
    gate = torch.randn(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, heads, seq_len, device='cuda', dtype=torch.float32))
    state = torch.randn(batch, heads, key_dim, value_dim, device='cuda', dtype=torch.float32)

    ref_state = state.clone()
    ref_outs = []
    q = torch.nn.functional.normalize(query, dim=-1) * (1.0 / (key_dim ** 0.5))
    k = torch.nn.functional.normalize(key, dim=-1)
    for idx in range(seq_len):
        gate_t = gate[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, idx].unsqueeze(-1)
        ref_state = ref_state * gate_t
        kv_mem = (ref_state * k[:, :, idx].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, idx] - kv_mem) * beta_t
        ref_state = ref_state + k[:, :, idx].unsqueeze(-1) * delta.unsqueeze(-2)
        ref_outs.append((ref_state * q[:, :, idx].unsqueeze(-1)).sum(dim=-2))
    ref_out = torch.stack(ref_outs, dim=2)

    out = recurrent_gated_delta_prefill(
        query,
        key,
        value,
        gate,
        beta,
        state,
        query_scale=(1.0 / (key_dim ** 0.5)),
        normalize_qk=True,
    )
    torch.cuda.synchronize()

    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state, ref_state, atol=1e-4, rtol=1e-4)


def test_qwen35_triton_chunk_local_solver_matches_pytorch_if_available():
    if not (HAS_TRITON_LINEAR_ATTN and torch.cuda.is_available()):
        return

    torch.manual_seed(0)
    chunk = 16
    attn = torch.randn(3, chunk, chunk, device='cuda', dtype=torch.float32).tril(-1)
    ref = attn.clone()

    for i in range(1, chunk):
        row = ref[:, i, :i].clone()
        sub = ref[:, :i, :i].clone()
        ref[:, i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    diag = torch.arange(chunk, device='cuda')
    ref[:, diag, diag] += 1.0

    out = solve_chunk_local_attention(attn.clone())
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_qwen35_causal_conv1d_single_step_matches_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 6, 1)
    conv_state = torch.randn(2, 6, 4)
    conv_weight = torch.randn(6, 4)

    ref_input = torch.cat([conv_state, hidden_states], dim=-1)
    ref_state = ref_input[:, :, -conv_state.shape[-1]:]
    ref_out = torch.nn.functional.conv1d(
        ref_input, conv_weight.unsqueeze(1), padding=0, groups=hidden_states.shape[1],
    )
    ref_out = torch.nn.functional.silu(ref_out[:, :, -1:])

    out, next_state = torch_causal_conv1d_update(
        hidden_states, conv_state.clone(), conv_weight,
    )

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(next_state, ref_state, atol=1e-6, rtol=1e-6)


def test_qwen35_causal_conv1d_single_step_matches_reference_noncontiguous_state():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 6, 1)
    base_state = torch.randn(2, 6, 8)
    conv_state = base_state[:, :, ::2]
    conv_weight = torch.randn(6, 4)

    assert not conv_state.is_contiguous()

    ref_input = torch.cat([conv_state.contiguous(), hidden_states], dim=-1)
    ref_state = ref_input[:, :, -conv_state.shape[-1]:]
    ref_out = torch.nn.functional.conv1d(
        ref_input, conv_weight.unsqueeze(1), padding=0, groups=hidden_states.shape[1],
    )
    ref_out = torch.nn.functional.silu(ref_out[:, :, -1:])

    out, next_state = torch_causal_conv1d_update(
        hidden_states, conv_state, conv_weight,
    )

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(next_state, ref_state, atol=1e-6, rtol=1e-6)


def test_qwen35_causal_conv1d_multi_step_matches_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 6, 3)
    conv_state = torch.randn(2, 6, 4)
    conv_weight = torch.randn(6, 4)

    ref_input = torch.cat([conv_state, hidden_states], dim=-1)
    ref_state = ref_input[:, :, -conv_state.shape[-1]:]
    ref_out = torch.nn.functional.conv1d(
        ref_input, conv_weight.unsqueeze(1), padding=0, groups=hidden_states.shape[1],
    )
    ref_out = torch.nn.functional.silu(ref_out[:, :, -hidden_states.shape[-1]:])

    out, next_state = torch_causal_conv1d_update(
        hidden_states, conv_state.clone(), conv_weight,
    )

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(next_state, ref_state, atol=1e-6, rtol=1e-6)


def test_qwen35_causal_conv1d_single_step_cuda_matches_reference_if_available():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    hidden_states = torch.randn(2, 64, 1, device='cuda', dtype=torch.float16)
    conv_state = torch.randn(2, 64, 4, device='cuda', dtype=torch.float16)
    conv_weight = torch.randn(64, 4, device='cuda', dtype=torch.float16)

    ref_input = torch.cat([conv_state, hidden_states], dim=-1)
    ref_state = ref_input[:, :, -conv_state.shape[-1]:]
    ref_out = torch.nn.functional.conv1d(
        ref_input, conv_weight.unsqueeze(1), padding=0, groups=hidden_states.shape[1],
    )
    ref_out = torch.nn.functional.silu(ref_out[:, :, -1:])

    out, next_state = torch_causal_conv1d_update(
        hidden_states, conv_state.clone(), conv_weight,
    )
    torch.cuda.synchronize()

    assert torch.allclose(out, ref_out, atol=2e-3, rtol=2e-3)
    assert torch.allclose(next_state, ref_state, atol=2e-3, rtol=2e-3)


def test_qwen35_fused_linear_gates_matches_reference():
    torch.manual_seed(0)
    a = torch.randn(2, 5, 4)
    b = torch.randn(2, 5, 4)
    a_log = torch.randn(4)
    dt_bias = torch.randn(4)

    beta, gk = _fused_linear_gates(a, b, a_log, dt_bias)
    ref_beta = b.float().sigmoid()
    ref_gk = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())

    assert torch.allclose(beta, ref_beta, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gk, ref_gk, atol=1e-6, rtol=1e-6)


def test_qwen35_gqa_decode_step_matches_repeated_reference():
    torch.manual_seed(0)
    batch, q_heads, kv_groups, key_dim, value_dim = 2, 2, 3, 8, 6
    kv_heads = q_heads * kv_groups
    query = torch.randn(batch, q_heads, key_dim)
    key = torch.randn(batch, q_heads, key_dim)
    value = torch.randn(batch, kv_heads, value_dim)
    gate = torch.randn(batch, kv_heads)
    beta = torch.sigmoid(torch.randn(batch, kv_heads))
    initial_state = torch.randn(batch, kv_heads, key_dim, value_dim)

    out, next_state = recurrent_gated_delta_decode_step(
        query, key, value, gate, beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        num_kv_groups=kv_groups,
        use_qk_l2norm_in_kernel=True,
    )
    ref_out, ref_next_state = recurrent_gated_delta_decode_step(
        query.repeat_interleave(kv_groups, dim=1),
        key.repeat_interleave(kv_groups, dim=1),
        value,
        gate,
        beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        num_kv_groups=1,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(next_state, ref_next_state, atol=1e-6, rtol=1e-6)


def test_qwen35_gqa_short_prefill_matches_repeated_reference():
    torch.manual_seed(0)
    batch, seq_len, q_heads, kv_groups, key_dim, value_dim = 2, 17, 2, 3, 8, 6
    kv_heads = q_heads * kv_groups
    query = torch.randn(batch, seq_len, q_heads, key_dim)
    key = torch.randn(batch, seq_len, q_heads, key_dim)
    value = torch.randn(batch, seq_len, kv_heads, value_dim)
    gate = torch.randn(batch, seq_len, kv_heads)
    beta = torch.sigmoid(torch.randn(batch, seq_len, kv_heads))

    out, next_state = recurrent_gated_delta_prefill_short_sequence(
        query,
        key,
        value,
        gate,
        beta,
        output_final_state=True,
        num_kv_groups=kv_groups,
        use_qk_l2norm_in_kernel=True,
    )
    ref_out, ref_next_state = recurrent_gated_delta_prefill_short_sequence(
        query.repeat_interleave(kv_groups, dim=2),
        key.repeat_interleave(kv_groups, dim=2),
        value,
        gate,
        beta,
        output_final_state=True,
        num_kv_groups=1,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(next_state, ref_next_state, atol=1e-5, rtol=1e-5)


def test_qwen35_rmsnorm_gated_module_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(7, 16)
    gate = torch.randn(7, 16)
    norm = RMSNormGated(16, eps=1e-6)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(variance + norm.eps) * norm.weight * torch.nn.functional.silu(gate)).to(x.dtype)
    out = norm(x, gate)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_qwen35_rmsnorm_gated_kernel_matches_reference_if_available():
    if not torch.cuda.is_available() or not HAS_RMSNORM_GATED:
        return

    torch.manual_seed(0)
    x = torch.randn(11, 32, device="cuda", dtype=torch.float16)
    gate = torch.randn(11, 32, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, device="cuda", dtype=torch.float16)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(variance + 1e-6) * weight * torch.nn.functional.silu(gate)).to(x.dtype)
    out = rmsnorm_gated(x, gate, weight, 1e-6)
    torch.cuda.synchronize()

    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)


def test_qwen35_mgrmsnorm_offset_module_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(5, 24)
    norm = MGRMSNorm(24, eps=1e-6, offset=True)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(variance + norm.eps) * (norm.weight + 1.0)).to(x.dtype)
    out = norm(x)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_qwen35_mgrmsnorm_offset_kernel_matches_reference_if_available():
    if not torch.cuda.is_available() or not HAS_TRITON_RMSNORM:
        return

    torch.manual_seed(0)
    x = torch.randn(13, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, device="cuda", dtype=torch.float16)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(variance + 1e-6) * (weight + 1.0)).to(x.dtype)
    out = rmsnorm_triton(x, weight, 1e-6, offset=True)
    torch.cuda.synchronize()

    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)


def test_qwen35_gated_delta_fused_baz_matches_separate_projections():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config())
    layer = GatedDeltaNet(cfg, layer_idx=0)
    hidden_states = torch.randn(2, 3, cfg.hidden_size)

    fused = layer.in_proj_baz(hidden_states)
    z_fused, b_fused, a_fused = fused.split(
        [layer.value_dim, layer.num_v_heads, layer.num_v_heads], dim=-1,
    )

    z_weight, b_weight, a_weight = layer.in_proj_baz.weight.split(
        [layer.value_dim, layer.num_v_heads, layer.num_v_heads], dim=0,
    )
    z_ref = torch.nn.functional.linear(hidden_states, z_weight)
    b_ref = torch.nn.functional.linear(hidden_states, b_weight)
    a_ref = torch.nn.functional.linear(hidden_states, a_weight)

    assert torch.allclose(z_fused, z_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(b_fused, b_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(a_fused, a_ref, atol=1e-6, rtol=1e-6)


def test_qwen35_gated_delta_fused_in_proj_matches_separate_projections():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config())
    layer = GatedDeltaNet(cfg, layer_idx=0)
    hidden_states = torch.randn(2, 3, cfg.hidden_size)

    fused_weight = layer._get_fused_in_proj_weight()
    fused_out = torch.nn.functional.linear(hidden_states, fused_weight)
    qkv_fused, baz_fused = fused_out.split([layer.conv_dim, layer.in_proj_baz_dim], dim=-1)

    qkv_ref = layer.in_proj_qkv(hidden_states)
    baz_ref = layer.in_proj_baz(hidden_states)

    assert torch.allclose(qkv_fused, qkv_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(baz_fused, baz_ref, atol=1e-6, rtol=1e-6)


def test_qwen35_tiny_hybrid_prefill_decode_runs():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    seq_id = 7
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    positions = torch.arange(6).unsqueeze(0)

    block_manager.allocate_sequence(seq_id, num_tokens=10)
    logits = model.prefill(input_ids, positions, block_manager, seq_id)
    assert logits.shape == (1, 6, cfg.vocab_size)
    assert block_manager.seq_lens[seq_id] == 6

    linear_layers = [i for i, layer_type in enumerate(cfg.layer_types) if layer_type == "linear_attention"]
    for layer_idx in linear_layers:
        conv_state, recurrent_state = block_manager.get_linear_state(seq_id, layer_idx)
        assert conv_state is not None
        assert recurrent_state is not None

    decode_ids = torch.randint(0, cfg.vocab_size, (1, 1))
    decode_pos = torch.tensor([[6]])
    next_logits = model.decode_step(decode_ids, decode_pos, block_manager, [seq_id])
    assert next_logits.shape == (1, 1, cfg.vocab_size)
    assert block_manager.seq_lens[seq_id] == 7


def test_qwen35_tiny_hybrid_decode_multi_matches_step():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)

    prefill_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    seq_id = 9
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    positions = torch.arange(6).unsqueeze(0)
    prefill_manager.allocate_sequence(seq_id, num_tokens=16)

    prefill_logits = model.prefill(input_ids, positions, prefill_manager, seq_id)
    first_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    first_pos = torch.tensor([[6]])
    snapshot = prefill_manager.serialize_sequence(seq_id)

    step_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    step_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)

    step_input = first_token.clone()
    step_pos = first_pos.clone()
    step_tokens = []
    step_final_logits = None
    for _ in range(4):
        step_final_logits = model.decode_step(step_input, step_pos, step_manager, [seq_id])
        next_token = step_final_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        step_tokens.append(int(next_token.item()))
        step_input = next_token
        step_pos += 1

    multi_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    multi_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)

    multi_tokens, multi_final_logits = model.decode_multi_step(
        first_token, first_pos, multi_manager, [seq_id], num_steps=4,
    )

    assert step_tokens == multi_tokens[0].tolist()
    assert torch.allclose(step_final_logits, multi_final_logits, atol=1e-5, rtol=1e-5)
    assert step_manager.seq_lens[seq_id] == multi_manager.seq_lens[seq_id]


def test_qwen35_tiny_hybrid_decode_multi_teacher_forced_matches_step():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)

    prefill_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    seq_id = 10
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    positions = torch.arange(6).unsqueeze(0)
    prefill_manager.allocate_sequence(seq_id, num_tokens=16)
    model.prefill(input_ids, positions, prefill_manager, seq_id)
    snapshot = prefill_manager.serialize_sequence(seq_id)
    first_pos = torch.tensor([[6]])
    forced_tokens = [7, 11, 13, 17]

    step_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    step_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)

    step_pos = first_pos.clone()
    step_final_logits = None
    for token_id in forced_tokens:
        step_input = torch.tensor([[token_id]], dtype=torch.long)
        step_final_logits = model.decode_step(step_input, step_pos, step_manager, [seq_id])
        step_pos += 1

    multi_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    multi_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)

    forced_next = torch.tensor([forced_tokens[1:]], dtype=torch.long)
    multi_tokens, multi_final_logits = model.decode_multi_step(
        torch.tensor([[forced_tokens[0]]], dtype=torch.long),
        first_pos,
        multi_manager,
        [seq_id],
        num_steps=len(forced_tokens),
        forced_next_token_ids=forced_next,
    )

    expected_next_tokens = forced_tokens[1:] + [int(step_final_logits[:, -1, :].argmax(dim=-1).item())]
    assert multi_tokens[0].tolist() == expected_next_tokens
    assert torch.allclose(step_final_logits, multi_final_logits, atol=1e-5, rtol=1e-5)
    assert step_manager.seq_lens[seq_id] == multi_manager.seq_lens[seq_id]

    step_snapshot = step_manager.serialize_sequence(seq_id)
    multi_snapshot = multi_manager.serialize_sequence(seq_id)
    assert step_snapshot["seq_len"] == multi_snapshot["seq_len"]
    for layer_idx, ref_layer in step_snapshot["kv_data_by_layer"].items():
        assert torch.allclose(ref_layer, multi_snapshot["kv_data_by_layer"][layer_idx], atol=1e-5, rtol=1e-5)


def test_qwen35_tiny_full_attention_suffix_prefill_matches_step():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(layer_types=["full_attention"] * 4))
    model = MegaGemmLlama(cfg)

    prefill_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    seq_id = 11
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    positions = torch.arange(6).unsqueeze(0)
    prefill_manager.allocate_sequence(seq_id, num_tokens=16)
    model.prefill(input_ids, positions, prefill_manager, seq_id)
    snapshot = prefill_manager.serialize_sequence(seq_id)

    prefix_len = 3
    tail_token_ids = input_ids[0, prefix_len:].tolist()

    step_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    step_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)
    step_manager.truncate_sequence(seq_id, prefix_len)

    step_pos = torch.tensor([[prefix_len]], dtype=torch.long)
    step_final_logits = None
    for token_id in tail_token_ids:
        step_input = torch.tensor([[token_id]], dtype=torch.long)
        step_final_logits = model.decode_step(step_input, step_pos, step_manager, [seq_id])
        step_pos += 1

    suffix_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    suffix_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=8)
    suffix_manager.truncate_sequence(seq_id, prefix_len)

    suffix_input = torch.tensor([tail_token_ids], dtype=torch.long)
    suffix_positions = torch.arange(prefix_len, prefix_len + len(tail_token_ids)).unsqueeze(0)
    suffix_final_logits = model.prefill_suffix(
        suffix_input,
        suffix_positions,
        suffix_manager,
        seq_id,
    )

    assert torch.allclose(step_final_logits, suffix_final_logits, atol=1e-5, rtol=1e-5)
    assert step_manager.seq_lens[seq_id] == suffix_manager.seq_lens[seq_id]

    step_snapshot = step_manager.serialize_sequence(seq_id)
    suffix_snapshot = suffix_manager.serialize_sequence(seq_id)
    assert step_snapshot["seq_len"] == suffix_snapshot["seq_len"]
    for layer_idx, ref_layer in step_snapshot["kv_data_by_layer"].items():
        assert torch.allclose(suffix_snapshot["kv_data_by_layer"][layer_idx], ref_layer, atol=1e-5, rtol=1e-5)


def test_qwen35_tiny_hybrid_continuous_batching_runs():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=64,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    scheduler = Scheduler(model, block_manager, max_batch_size=4, device='cpu')

    req1 = scheduler.add_request([1, 2, 3, 4], max_new_tokens=4, temperature=0.0)
    req2 = scheduler.add_request([5, 6, 7], max_new_tokens=4, temperature=0.0)

    completed = []
    for _ in range(8):
        completed.extend(scheduler.step())
        if not scheduler.has_pending():
            break

    assert not scheduler.has_pending()
    assert sorted(req.request_id for req in completed) == [req1, req2]
    for req in completed:
        assert len(req.generated_ids) == 4


def test_qwen35_tiny_hybrid_continuous_batching_runs_with_sparse_kv_layers():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)
    kv_layer_indices = [i for i, layer_type in enumerate(cfg.layer_types) if layer_type != "linear_attention"]
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=64,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
        kv_layer_indices=kv_layer_indices,
    )
    scheduler = Scheduler(model, block_manager, max_batch_size=4, device='cpu')

    req1 = scheduler.add_request([1, 2, 3, 4], max_new_tokens=4, temperature=0.0)
    req2 = scheduler.add_request([5, 6, 7], max_new_tokens=4, temperature=0.0)

    completed = []
    for _ in range(8):
        completed.extend(scheduler.step())
        if not scheduler.has_pending():
            break

    assert not scheduler.has_pending()
    assert sorted(req.request_id for req in completed) == [req1, req2]
    for req in completed:
        assert len(req.generated_ids) == 4


def test_qwen35_sparse_kv_can_start_with_linear_layers():
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    ))
    kv_layer_indices = [i for i, layer_type in enumerate(cfg.layer_types) if layer_type != "linear_attention"]
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=32,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
        kv_layer_indices=kv_layer_indices,
    )

    assert block_manager.get_kv_cache(0) is None
    assert block_manager.get_kv_cache(1) is None
    assert block_manager.get_kv_cache(2) is None
    assert block_manager.get_kv_cache(3) is not None


def test_qwen35_tiny_hybrid_prefill_packed_does_not_fallback_to_padded():
    torch.manual_seed(0)
    cfg = LlamaConfig.from_dict(_qwen35_text_config(
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
    ))
    model = MegaGemmLlama(cfg)
    block_manager = BlockManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=64,
        block_size=8,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    prompts = [[1, 2, 3, 4], [5, 6, 7]]
    seq_ids = [101, 102]
    lengths = torch.tensor([len(p) for p in prompts], dtype=torch.long)
    cu_seqlens = torch.tensor([0, len(prompts[0]), len(prompts[0]) + len(prompts[1])], dtype=torch.int32)
    packed_tokens = torch.tensor([prompts[0] + prompts[1]], dtype=torch.long)

    for sid, prompt in zip(seq_ids, prompts):
        block_manager.allocate_sequence(sid, num_tokens=len(prompt) + 8)

    original_prefill_batch = model.prefill_batch
    prev_env = os.environ.get("MEGAGEMM_QWEN35_PACKED_LINEAR_NATIVE")
    os.environ["MEGAGEMM_QWEN35_PACKED_LINEAR_NATIVE"] = "1"

    def _fail_prefill_batch(*args, **kwargs):
        raise AssertionError("prefill_packed unexpectedly fell back to prefill_batch")

    model.prefill_batch = _fail_prefill_batch
    try:
        logits = model.prefill_packed(packed_tokens, cu_seqlens, lengths, block_manager, seq_ids)
    finally:
        model.prefill_batch = original_prefill_batch
        if prev_env is None:
            os.environ.pop("MEGAGEMM_QWEN35_PACKED_LINEAR_NATIVE", None)
        else:
            os.environ["MEGAGEMM_QWEN35_PACKED_LINEAR_NATIVE"] = prev_env

    assert logits.shape == (2, 1, cfg.vocab_size)
    assert block_manager.seq_lens[seq_ids[0]] == len(prompts[0])
    assert block_manager.seq_lens[seq_ids[1]] == len(prompts[1])

    linear_layers = [i for i, layer_type in enumerate(cfg.layer_types) if layer_type == "linear_attention"]
    for sid in seq_ids:
        for layer_idx in linear_layers:
            conv_state, recurrent_state = block_manager.get_linear_state(sid, layer_idx)
            assert conv_state is not None
            assert recurrent_state is not None
