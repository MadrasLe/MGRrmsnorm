import torch

from megagemm.models.llama import LlamaAttention, LlamaConfig, LlamaMLP, MegaGemmLlama
from megagemm.quantization.w8a16 import Int8Linear


def _tiny_cfg():
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=512,
        max_position_embeddings=128,
        layer_types=["full_attention"],
    )


def test_int8_attention_decode_qkv_fallback_without_weight_attr():
    cfg = _tiny_cfg()
    attn = LlamaAttention(cfg, layer_idx=0)
    attn.qkv_proj = Int8Linear.from_linear(attn.qkv_proj)

    x = torch.randn(2, 1, cfg.hidden_size)
    qkv = attn._decode_qkv_linear(x)
    assert qkv.shape == (2, 1, attn._q_proj_size + attn._k_size + attn._v_size)

    qkv_normed = attn._decode_qkv_from_raw_hidden(
        x,
        torch.ones(cfg.hidden_size),
        input_norm_eps=cfg.rms_norm_eps,
        input_norm_offset=cfg.norm_offset,
    )
    assert qkv_normed.shape == qkv.shape


def test_int8_mlp_decode_fallback_without_weight_attr():
    cfg = _tiny_cfg()
    mlp = LlamaMLP(cfg)
    mlp.gate_up_proj = Int8Linear.from_linear(mlp.gate_up_proj)
    mlp.down_proj = Int8Linear.from_linear(mlp.down_proj)

    x = torch.randn(2, 1, cfg.hidden_size)
    gate_up = mlp._decode_gate_up_linear(x)
    assert gate_up.shape == (2, 1, 2 * cfg.intermediate_size)

    out = mlp.forward_decode(x)
    assert out.shape == x.shape

    residual = torch.zeros_like(out)
    out_res = mlp.forward_decode_add_residual(x, residual)
    assert out_res.shape == x.shape


def test_flat_decode_is_disabled_for_int8_projections():
    cfg = _tiny_cfg()
    model = MegaGemmLlama(cfg)
    layer = model.layers[0]
    layer.self_attn.qkv_proj = Int8Linear.from_linear(layer.self_attn.qkv_proj)

    model._prepare_flat_decode()
    assert model._flat_decode_failed is True
    assert model._flat_decode_ready is False
