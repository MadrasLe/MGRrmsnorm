import os
import hashlib
import json
import struct
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from megagemm.engine import MGXProphetLibrary
from megagemm.engine.engine import InferenceEngine
from megagemm.engine.kv_cache import BlockManager
from megagemm.models import (
    attach_session_state_to_mgx,
    export_to_mgx,
    extract_session_state_from_mgx,
    inspect_mgx,
    load_from_hf,
    load_from_mgx,
)
from megagemm.models.llama import LlamaConfig, MegaGemmLlama
from megagemm.models.mgx import (
    MGXFormatError,
    MGX_HEADER_SIZE,
    _align_up,
    _encode_header,
    _validate_manifest,
    prime_mgx_payload_cache,
)


def _tiny_hf_config(*, tie_word_embeddings: bool = False) -> dict:
    return {
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tie_word_embeddings,
    }


def _tiny_gemma4_hf_config() -> dict:
    return {
        "model_type": "gemma4",
        "vision_config": {"model_type": "gemma4_vision"},
        "audio_config": {"model_type": "gemma4_audio"},
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "num_global_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "sliding_window": 3,
            "num_kv_shared_layers": 1,
            "use_double_wide_mlp": True,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 128,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            },
        },
    }


def _tiny_qwen3_moe_hf_config(**overrides) -> dict:
    cfg = {
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "hidden_act": "silu",
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
    }
    cfg.update(overrides)
    return cfg


def _tiny_config(*, tie_word_embeddings: bool = False) -> LlamaConfig:
    return LlamaConfig.from_dict(_tiny_hf_config(tie_word_embeddings=tie_word_embeddings))


def _megagemm_to_hf_state(model: MegaGemmLlama) -> dict[str, torch.Tensor]:
    cfg = model.config
    state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {}

    hf_state["model.embed_tokens.weight"] = state["embed_tokens.weight"].clone()
    hf_state["model.norm.weight"] = state["norm.weight"].clone()
    hf_state["lm_head.weight"] = state["lm_head.weight"].clone()

    q_size = cfg.num_attention_heads * cfg.head_dim
    k_size = cfg.num_key_value_heads * cfg.head_dim
    v_size = cfg.num_key_value_heads * cfg.head_dim

    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        qkv = state[f"{prefix}.self_attn.qkv_proj.weight"]
        hf_state[f"{hf_prefix}.self_attn.q_proj.weight"] = qkv[:q_size].clone()
        hf_state[f"{hf_prefix}.self_attn.k_proj.weight"] = qkv[q_size : q_size + k_size].clone()
        hf_state[f"{hf_prefix}.self_attn.v_proj.weight"] = qkv[q_size + k_size : q_size + k_size + v_size].clone()
        hf_state[f"{hf_prefix}.self_attn.o_proj.weight"] = state[f"{prefix}.self_attn.o_proj.weight"].clone()

        gate_up = state[f"{prefix}.mlp.gate_up_proj.weight"]
        half = cfg.intermediate_size
        hf_state[f"{hf_prefix}.mlp.gate_proj.weight"] = gate_up[:half].clone()
        hf_state[f"{hf_prefix}.mlp.up_proj.weight"] = gate_up[half:].clone()
        hf_state[f"{hf_prefix}.mlp.down_proj.weight"] = state[f"{prefix}.mlp.down_proj.weight"].clone()

        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[
            f"{prefix}.input_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[
            f"{prefix}.post_attention_layernorm.weight"
        ].clone()

    return hf_state


def _megagemm_gemma4_to_hf_state(model: MegaGemmLlama) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.embed_tokens_per_layer.weight": state["embed_tokens_per_layer.weight"].clone(),
        "model.per_layer_model_projection.weight": state["per_layer_model_projection.weight"].clone(),
        "model.per_layer_projection_norm.weight": state["per_layer_projection_norm.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
        "lm_head.weight": state["lm_head.weight"].clone(),
    }

    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        hf_state[f"{hf_prefix}.layer_scalar"] = state[f"{prefix}.layer_scalar"].clone()
        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[f"{prefix}.input_layernorm.weight"].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[
            f"{prefix}.post_attention_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = state[
            f"{prefix}.pre_feedforward_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.post_feedforward_layernorm.weight"] = state[
            f"{prefix}.post_feedforward_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.per_layer_input_gate.weight"] = state[
            f"{prefix}.per_layer_input_gate.weight"
        ].clone()
        hf_state[f"{hf_prefix}.per_layer_projection.weight"] = state[
            f"{prefix}.per_layer_projection.weight"
        ].clone()
        hf_state[f"{hf_prefix}.post_per_layer_input_norm.weight"] = state[
            f"{prefix}.post_per_layer_input_norm.weight"
        ].clone()

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"{prefix}.self_attn.{proj_name}.weight"
            if key in state:
                hf_state[f"{hf_prefix}.self_attn.{proj_name}.weight"] = state[key].clone()
        for norm_name in ("q_norm", "k_norm"):
            key = f"{prefix}.self_attn.{norm_name}.weight"
            if key in state:
                hf_state[f"{hf_prefix}.self_attn.{norm_name}.weight"] = state[key].clone()

        gate_up = state[f"{prefix}.mlp.gate_up_proj.weight"]
        half = gate_up.shape[0] // 2
        hf_state[f"{hf_prefix}.mlp.gate_proj.weight"] = gate_up[:half].clone()
        hf_state[f"{hf_prefix}.mlp.up_proj.weight"] = gate_up[half:].clone()
        hf_state[f"{hf_prefix}.mlp.down_proj.weight"] = state[f"{prefix}.mlp.down_proj.weight"].clone()

    return hf_state


def _megagemm_qwen3_moe_to_hf_state(model: MegaGemmLlama) -> dict[str, torch.Tensor]:
    cfg = model.config
    state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
        "lm_head.weight": state["lm_head.weight"].clone(),
    }

    q_size = cfg.num_attention_heads * cfg.head_dim
    k_size = cfg.num_key_value_heads * cfg.head_dim
    v_size = cfg.num_key_value_heads * cfg.head_dim

    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        qkv = state[f"{prefix}.self_attn.qkv_proj.weight"]
        hf_state[f"{hf_prefix}.self_attn.q_proj.weight"] = qkv[:q_size].clone()
        hf_state[f"{hf_prefix}.self_attn.k_proj.weight"] = qkv[
            q_size : q_size + k_size
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.v_proj.weight"] = qkv[
            q_size + k_size : q_size + k_size + v_size
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.o_proj.weight"] = state[
            f"{prefix}.self_attn.o_proj.weight"
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.q_norm.weight"] = state[
            f"{prefix}.self_attn.q_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.k_norm.weight"] = state[
            f"{prefix}.self_attn.k_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[
            f"{prefix}.input_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[
            f"{prefix}.post_attention_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.mlp.gate.weight"] = state[
            f"{prefix}.mlp.gate.weight"
        ].clone()

        gate_up = state[f"{prefix}.mlp.experts.gate_up_proj"]
        down = state[f"{prefix}.mlp.experts.down_proj"]
        for expert_idx in range(cfg.num_experts):
            expert_prefix = f"{hf_prefix}.mlp.experts.{expert_idx}"
            hf_state[f"{expert_prefix}.gate_proj.weight"] = gate_up[
                expert_idx, : cfg.moe_intermediate_size
            ].clone()
            hf_state[f"{expert_prefix}.up_proj.weight"] = gate_up[
                expert_idx, cfg.moe_intermediate_size :
            ].clone()
            hf_state[f"{expert_prefix}.down_proj.weight"] = down[expert_idx].clone()

    return hf_state


def _build_local_snapshot(root: Path, *, tie_word_embeddings: bool = False) -> Path:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)

    hf_config = _tiny_hf_config(tie_word_embeddings=tie_word_embeddings)
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    if tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    hf_state = _megagemm_to_hf_state(model)
    if tie_word_embeddings:
        hf_state.pop("lm_head.weight", None)

    _write_snapshot_metadata(root, hf_config)

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))
    return root


def _build_local_qwen3_moe_snapshot(root: Path) -> Path:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)

    hf_config = _tiny_qwen3_moe_hf_config()
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    hf_state = _megagemm_qwen3_moe_to_hf_state(model)

    _write_snapshot_metadata(root, hf_config)

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))
    return root


def _build_local_qwen3_moe_awq_snapshot(root: Path, *, group_size: int = 4) -> Path:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)

    hf_config = _tiny_qwen3_moe_hf_config(
        moe_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        num_key_value_heads=2,
    )
    hf_config["quantization_config"] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
    }
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    state = model.state_dict()

    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
        "lm_head.weight": state["lm_head.weight"].clone(),
    }

    q_size = config.num_attention_heads * config.head_dim
    k_size = config.num_key_value_heads * config.head_dim
    v_size = config.num_key_value_heads * config.head_dim

    def _add_awq(prefix: str, in_features: int, out_features: int) -> None:
        qweight, scales, qzeros = _fake_awq_tensors(in_features, out_features, group_size)
        hf_state[f"{prefix}.qweight"] = qweight
        hf_state[f"{prefix}.scales"] = scales
        hf_state[f"{prefix}.qzeros"] = qzeros

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"
        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[
            f"{prefix}.input_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[
            f"{prefix}.post_attention_layernorm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.q_norm.weight"] = state[
            f"{prefix}.self_attn.q_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.self_attn.k_norm.weight"] = state[
            f"{prefix}.self_attn.k_norm.weight"
        ].clone()
        hf_state[f"{hf_prefix}.mlp.gate.weight"] = state[
            f"{prefix}.mlp.gate.weight"
        ].clone()

        _add_awq(f"{hf_prefix}.self_attn.q_proj", config.hidden_size, q_size)
        _add_awq(f"{hf_prefix}.self_attn.k_proj", config.hidden_size, k_size)
        _add_awq(f"{hf_prefix}.self_attn.v_proj", config.hidden_size, v_size)
        _add_awq(f"{hf_prefix}.self_attn.o_proj", config.hidden_size, config.hidden_size)
        for expert_idx in range(config.num_experts):
            expert_prefix = f"{hf_prefix}.mlp.experts.{expert_idx}"
            _add_awq(f"{expert_prefix}.gate_proj", config.hidden_size, config.moe_intermediate_size)
            _add_awq(f"{expert_prefix}.up_proj", config.hidden_size, config.moe_intermediate_size)
            _add_awq(f"{expert_prefix}.down_proj", config.moe_intermediate_size, config.hidden_size)

    _write_snapshot_metadata(root, hf_config)

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))
    return root


def _build_local_gemma4_snapshot(root: Path) -> Path:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)

    hf_config = _tiny_gemma4_hf_config()
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    hf_state = _megagemm_gemma4_to_hf_state(model)

    _write_snapshot_metadata(root, hf_config)

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))
    return root


def _write_snapshot_metadata(root: Path, hf_config: dict) -> None:
    with (root / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(hf_config, fh)

    tokenizer_cfg = {
        "chat_template": "<bos>{{ messages[0]['content'] }}<eos>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    with (root / "tokenizer_config.json").open("w", encoding="utf-8") as fh:
        json.dump(tokenizer_cfg, fh)
    with (root / "tokenizer.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "unk_token": "<unk>",
                    "vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4},
                },
            },
            fh,
        )
    with (root / "special_tokens_map.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
            fh,
        )


def _fake_awq_tensors(in_features: int, out_features: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert out_features % 8 == 0
    assert in_features % group_size == 0
    num_groups = in_features // group_size
    qweight = torch.zeros(in_features, out_features // 8, dtype=torch.int32)
    qzeros = torch.zeros(num_groups, out_features // 8, dtype=torch.int32)
    scales = torch.ones(num_groups, out_features, dtype=torch.float16)
    return qweight, scales, qzeros


def _build_local_awq_snapshot(root: Path, *, group_size: int = 8) -> Path:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)

    hf_config = _tiny_hf_config()
    hf_config["quantization_config"] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
    }
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
        "lm_head.weight": state["lm_head.weight"].clone(),
    }

    q_size = config.num_attention_heads * config.head_dim
    k_size = config.num_key_value_heads * config.head_dim
    v_size = config.num_key_value_heads * config.head_dim

    def _add_awq(prefix: str, in_features: int, out_features: int) -> None:
        qweight, scales, qzeros = _fake_awq_tensors(in_features, out_features, group_size)
        hf_state[f"{prefix}.qweight"] = qweight
        hf_state[f"{prefix}.scales"] = scales
        hf_state[f"{prefix}.qzeros"] = qzeros

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"
        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[f"{prefix}.input_layernorm.weight"].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[f"{prefix}.post_attention_layernorm.weight"].clone()

        _add_awq(f"{hf_prefix}.self_attn.q_proj", config.hidden_size, q_size)
        _add_awq(f"{hf_prefix}.self_attn.k_proj", config.hidden_size, k_size)
        _add_awq(f"{hf_prefix}.self_attn.v_proj", config.hidden_size, v_size)
        _add_awq(f"{hf_prefix}.self_attn.o_proj", config.hidden_size, config.hidden_size)
        _add_awq(f"{hf_prefix}.mlp.gate_proj", config.hidden_size, config.intermediate_size)
        _add_awq(f"{hf_prefix}.mlp.up_proj", config.hidden_size, config.intermediate_size)
        _add_awq(f"{hf_prefix}.mlp.down_proj", config.intermediate_size, config.hidden_size)

    _write_snapshot_metadata(root, hf_config)

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))
    return root


def _hybrid_kv_sources(config: LlamaConfig) -> dict[int, int]:
    return {
        layer_idx: int(source_idx)
        for layer_idx, source_idx in enumerate(getattr(config, "kv_share_sources", []) or [])
        if source_idx is not None
    }


def _make_block_manager(config: LlamaConfig, dtype: torch.dtype, device: str = "cpu") -> BlockManager:
    kwargs = {}
    if getattr(config, "kv_cache_layer_indices", None):
        kwargs["kv_layer_indices"] = config.kv_cache_layer_indices
    if getattr(config, "per_layer_num_kv_heads", None):
        kwargs["per_layer_num_kv_heads"] = config.per_layer_num_kv_heads
    if getattr(config, "per_layer_head_dims", None):
        kwargs["per_layer_head_dims"] = config.per_layer_head_dims
    kv_layer_sources = _hybrid_kv_sources(config)
    if kv_layer_sources:
        kwargs["kv_layer_sources"] = kv_layer_sources
    return BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=64,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
        **kwargs,
    )


def _greedy_tokens(model: MegaGemmLlama, prompt_ids: list[int], steps: int) -> list[int]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cfg = model.config
    block_manager = _make_block_manager(cfg, dtype=dtype, device=device.type)
    seq_id = 0
    block_manager.allocate_sequence(seq_id, len(prompt_ids) + steps)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_ids), dtype=torch.long, device=device).unsqueeze(0)
    logits = model.prefill(input_ids, positions, block_manager, seq_id)

    next_token = int(logits[:, -1, :].argmax(dim=-1).item())
    out = [next_token]
    for step in range(1, steps):
        cur_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        cur_pos = torch.tensor([[len(prompt_ids) + step - 1]], dtype=torch.long, device=device)
        logits = model.decode_step(cur_ids, cur_pos, block_manager, [seq_id])
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        out.append(next_token)

    return out


def _read_payload_bytes(path: Path) -> bytes:
    info = inspect_mgx(path, validate_payload_hash=False)
    header = info["header"]
    with path.open("rb") as fh:
        fh.seek(header["tensor_offset"])
        return fh.read(header["tensor_size"])


def _rewrite_mgx(path: Path, out_path: Path, manifest: dict, payload: bytes) -> None:
    manifest_bytes = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    manifest_offset = MGX_HEADER_SIZE
    tensor_offset = _align_up(manifest_offset + len(manifest_bytes))
    padding = tensor_offset - (manifest_offset + len(manifest_bytes))
    header = _encode_header(
        manifest_offset=manifest_offset,
        manifest_size=len(manifest_bytes),
        tensor_offset=tensor_offset,
        tensor_size=len(payload),
    )

    with out_path.open("wb") as fh:
        fh.write(header)
        fh.write(manifest_bytes)
        if padding:
            fh.write(b"\0" * padding)
        fh.write(payload)


class _FakeTokenizer:
    def __init__(self, chat_template: str):
        self.chat_template = chat_template
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = None
        self.eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"{self.bos_token}{messages[0]['content']}{self.eos_token}"

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        ids.extend(3 + (ord(ch) % 16) for ch in text[:16])
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return tensor
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        chars = []
        for token_id in ids:
            if skip_special_tokens and token_id in {0, 1, 2}:
                continue
            chars.append(chr(((int(token_id) - 3) % 16) + 97))
        return "".join(chars)


def test_export_import_fp16_roundtrip(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny-fp16.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    manifest = info["manifest"]

    assert artifact.exists()
    assert manifest["magic"] == "MGX1"
    assert manifest["quantization"] == "none"
    assert manifest["architecture"] == "llama"
    assert manifest["source_snapshot_path"] == str(snapshot.resolve())

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert torch.equal(ref.embed_tokens.weight, loaded.embed_tokens.weight)
    assert torch.equal(
        ref.layers[0].self_attn.qkv_proj.weight,
        loaded.layers[0].self_attn.qkv_proj.weight,
    )
    assert torch.equal(ref.cos_cache, loaded.cos_cache)
    assert torch.equal(ref.sin_cache, loaded.sin_cache)


@pytest.mark.parametrize("export_mode", ["normal", "streaming"])
def test_export_import_fp16_roundtrip_supports_both_export_modes(tmp_path, export_mode):
    snapshot = _build_local_snapshot(tmp_path / f"hf-{export_mode}")
    artifact = tmp_path / f"tiny-fp16-{export_mode}.mgx"

    info = export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="none",
        export_mode=export_mode,
    )

    assert info["export_mode"] == export_mode

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert torch.equal(ref.embed_tokens.weight, loaded.embed_tokens.weight)
    assert torch.equal(ref.layers[0].self_attn.qkv_proj.weight, loaded.layers[0].self_attn.qkv_proj.weight)


def test_export_import_gemma4_fp16_roundtrip_and_greedy_equivalence(tmp_path):
    snapshot = _build_local_gemma4_snapshot(tmp_path / "gemma4-hf")
    artifact = tmp_path / "tiny-gemma4-fp16.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    manifest = info["manifest"]

    assert artifact.exists()
    assert manifest["quantization"] == "none"
    assert manifest["architecture"] == "gemma4_text"
    assert "hybrid-sliding-attention" in manifest["compat_flags"]
    assert "full-attention-only" not in manifest["compat_flags"]

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert torch.equal(ref.embed_tokens_per_layer.weight, loaded.embed_tokens_per_layer.weight)
    assert torch.equal(ref.per_layer_model_projection.weight, loaded.per_layer_model_projection.weight)
    assert torch.equal(
        ref.layers[0].per_layer_input_gate.weight,
        loaded.layers[0].per_layer_input_gate.weight,
    )
    assert torch.equal(ref.layers[1].self_attn.k_proj.weight, loaded.layers[1].self_attn.k_proj.weight)
    assert ref.layers[3].self_attn.k_proj is None
    assert loaded.layers[3].self_attn.k_proj is None
    assert torch.equal(ref.layers[3].layer_scalar, loaded.layers[3].layer_scalar)
    assert torch.equal(ref.cos_cache, loaded.cos_cache)
    assert torch.equal(ref.sin_cache, loaded.sin_cache)

    prompt_ids = [5, 7, 9, 11]
    assert _greedy_tokens(ref, prompt_ids, steps=4) == _greedy_tokens(loaded, prompt_ids, steps=4)


def test_export_import_gemma4_int8_roundtrip_and_greedy_equivalence(tmp_path):
    snapshot = _build_local_gemma4_snapshot(tmp_path / "gemma4-hf")
    artifact = tmp_path / "tiny-gemma4-int8.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int8", export_mode="streaming")
    manifest = info["manifest"]

    assert artifact.exists()
    assert manifest["quantization"] == "int8"
    assert manifest["architecture"] == "gemma4_text"

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int8")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert ref.layers[0].self_attn.q_proj.__class__.__name__ == "Int8Linear"
    assert loaded.layers[0].self_attn.q_proj.__class__.__name__ == "Int8Linear"
    assert torch.equal(ref.layers[0].self_attn.q_proj.weight_int8, loaded.layers[0].self_attn.q_proj.weight_int8)
    assert torch.equal(ref.layers[0].self_attn.o_proj.scale, loaded.layers[0].self_attn.o_proj.scale)
    assert torch.equal(ref.layers[0].mlp.gate_up_proj.scale, loaded.layers[0].mlp.gate_up_proj.scale)
    assert loaded.layers[3].self_attn.k_proj is None
    assert loaded.layers[3].self_attn.v_proj is None

    prompt_ids = [5, 7, 9, 11]
    assert _greedy_tokens(ref, prompt_ids, steps=4) == _greedy_tokens(loaded, prompt_ids, steps=4)


def test_validate_manifest_still_rejects_linear_attention_layers():
    manifest = {
        "magic": "MGX1",
        "version": {"major": 1, "minor": 0},
        "target_backend": "megagemm",
        "quantization": "none",
        "config": {"layer_types": ["full_attention", "linear_attention"]},
    }

    with pytest.raises(MGXFormatError, match="unsupported layer types"):
        _validate_manifest(manifest)


def test_export_import_int8_roundtrip_and_greedy_equivalence(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny-int8.mgx"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int8")

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int8")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert ref.layers[0].self_attn.qkv_proj.__class__.__name__ == "Int8Linear"
    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "Int8Linear"
    assert torch.equal(
        ref.layers[0].self_attn.qkv_proj.weight_int8,
        loaded.layers[0].self_attn.qkv_proj.weight_int8,
    )
    assert torch.equal(
        ref.layers[0].mlp.gate_up_proj.scale,
        loaded.layers[0].mlp.gate_up_proj.scale,
    )

    prompt_ids = [5, 7, 9, 11]
    assert _greedy_tokens(ref, prompt_ids, steps=4) == _greedy_tokens(loaded, prompt_ids, steps=4)


def test_export_import_qwen3_moe_int8_roundtrip_preserves_expert_buffers(tmp_path):
    snapshot = _build_local_qwen3_moe_snapshot(tmp_path / "qwen3-moe-hf")
    artifact = tmp_path / "tiny-qwen3-moe-int8.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int8")
    manifest = info["manifest"]

    assert manifest["architecture"] == "qwen3_moe"
    assert manifest["quantization"] == "int8"

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int8")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    ref_experts = ref.layers[0].mlp.experts
    loaded_experts = loaded.layers[0].mlp.experts
    assert ref_experts._has_int8_experts()
    assert loaded_experts._has_int8_experts()
    assert "gate_up_proj" not in loaded_experts._parameters
    assert "down_proj" not in loaded_experts._parameters
    assert torch.equal(ref_experts.gate_up_int8, loaded_experts.gate_up_int8)
    assert torch.equal(ref_experts.down_int8, loaded_experts.down_int8)
    assert torch.equal(ref_experts.gate_up_scale, loaded_experts.gate_up_scale)
    assert torch.equal(ref_experts.down_scale, loaded_experts.down_scale)
    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "Int8Linear"


def test_export_import_qwen3_moe_int4_awq_roundtrip_preserves_expert_buffers(tmp_path):
    snapshot = _build_local_qwen3_moe_awq_snapshot(tmp_path / "qwen3-moe-awq")
    artifact = tmp_path / "tiny-qwen3-moe-int4.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int4")
    manifest = info["manifest"]

    assert manifest["architecture"] == "qwen3_moe"
    assert manifest["quantization"] == "int4"
    assert manifest["quantization_config"]["group_size"] == 4

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int4")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    ref_experts = ref.layers[0].mlp.experts
    loaded_experts = loaded.layers[0].mlp.experts
    assert ref_experts._has_awq_experts()
    assert loaded_experts._has_awq_experts()
    assert loaded_experts.awq_group_size == 4
    assert "gate_up_proj" not in loaded_experts._parameters
    assert "down_proj" not in loaded_experts._parameters
    assert torch.equal(ref_experts.gate_up_qweight, loaded_experts.gate_up_qweight)
    assert torch.equal(ref_experts.gate_up_scales, loaded_experts.gate_up_scales)
    assert torch.equal(ref_experts.gate_up_qzeros, loaded_experts.gate_up_qzeros)
    assert torch.equal(ref_experts.down_qweight, loaded_experts.down_qweight)
    assert torch.equal(ref_experts.down_scales, loaded_experts.down_scales)
    assert torch.equal(ref_experts.down_qzeros, loaded_experts.down_qzeros)
    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "QuantizedLinear"

    x = torch.randn(2, 1, loaded.config.hidden_size, dtype=torch.float16)
    y = loaded.layers[0].mlp(x)
    assert y.shape == x.shape
    assert bool(torch.isfinite(y).all())


def test_export_import_int4_awq_roundtrip(tmp_path):
    snapshot = _build_local_awq_snapshot(tmp_path / "hf-awq", group_size=8)
    artifact = tmp_path / "tiny-int4.mgx"

    info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int4")
    manifest = info["manifest"]

    assert manifest["quantization"] == "int4"
    assert manifest["quantization_config"]["group_size"] == 8
    assert manifest["quantization_config"]["weight_layout"] == "awq-decode-transposed-v1"

    ref = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int4")
    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)

    assert ref.layers[0].self_attn.qkv_proj.__class__.__name__ == "QuantizedLinear"
    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "QuantizedLinear"
    assert loaded.layers[0].self_attn.qkv_proj._transposed is True
    assert loaded.layers[0].mlp.gate_up_proj._transposed is True
    assert torch.equal(
        ref.layers[0].self_attn.qkv_proj.qweight.t().contiguous(),
        loaded.layers[0].self_attn.qkv_proj.qweight,
    )
    assert torch.equal(
        ref.layers[0].self_attn.qkv_proj.scales,
        loaded.layers[0].self_attn.qkv_proj.scales,
    )
    assert torch.equal(
        ref.layers[0].mlp.gate_up_proj.qzeros,
        loaded.layers[0].mlp.gate_up_proj.qzeros,
    )

    prompt_ids = [5, 7, 9, 11]
    assert _greedy_tokens(ref, prompt_ids, steps=4) == _greedy_tokens(loaded, prompt_ids, steps=4)


def test_inspect_rejects_invalid_magic(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    broken = tmp_path / "bad-magic.mgx"
    data = bytearray(artifact.read_bytes())
    data[0:4] = b"BAD!"
    broken.write_bytes(data)

    with pytest.raises(MGXFormatError, match="Invalid MGX magic"):
        inspect_mgx(broken)


def test_inspect_rejects_incompatible_version(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    broken = tmp_path / "bad-version.mgx"
    data = bytearray(artifact.read_bytes())
    struct.pack_into("<I", data, 4, 99)
    broken.write_bytes(data)

    with pytest.raises(MGXFormatError, match="Incompatible MGX major version"):
        inspect_mgx(broken)


def test_inspect_rejects_invalid_offsets(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    broken = tmp_path / "bad-offset.mgx"
    data = bytearray(artifact.read_bytes())
    struct.pack_into("<Q", data, 20, 10**9)
    broken.write_bytes(data)

    with pytest.raises(MGXFormatError, match="exceeds file size"):
        inspect_mgx(broken, validate_payload_hash=False)


def test_load_rejects_missing_required_tensor(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    info = inspect_mgx(artifact, validate_payload_hash=False)
    manifest = info["manifest"]
    payload = _read_payload_bytes(artifact)

    from safetensors.torch import load as load_safetensors
    from safetensors.torch import save as save_safetensors

    tensors = load_safetensors(payload)
    tensors.pop("embed_tokens.weight")
    new_payload = save_safetensors(tensors)
    manifest["tensor_payload_sha256"] = hashlib.sha256(new_payload).hexdigest()

    broken = tmp_path / "missing-tensor.mgx"
    _rewrite_mgx(artifact, broken, manifest, new_payload)

    with pytest.raises(MGXFormatError, match="missing required tensors"):
        load_from_mgx(broken, device="cpu", dtype_override=torch.float16)


def test_engine_initializes_from_mgx_without_hf_loader(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=AssertionError("AutoTokenizer should not be needed for MGX tokenizer fast path"),
    ), patch(
        "megagemm.engine.engine.load_from_hf",
        side_effect=AssertionError("load_from_hf should not be called for MGX"),
    ):
        engine = InferenceEngine(
            str(artifact),
            dtype=torch.float16,
            device="cpu",
            max_batch_size=2,
            max_seq_len=32,
        )

    assert engine.model.__class__.__name__ == "MegaGemmLlama"
    assert str(engine.tokenizer.name_or_path) == str(snapshot.resolve())
    assert engine.get_init_timing()["tokenizer_loader_kind"] in {"mgx-ultrafast", "mgx-fast"}


def test_engine_restores_missing_mgx_tokenizer_snapshot(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    info = inspect_mgx(artifact, validate_payload_hash=False)
    manifest = info["manifest"]
    payload = _read_payload_bytes(artifact)
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    missing_snapshot = tmp_path / "cache" / "snapshots" / revision
    manifest["source_model_id"] = "Qwen/Qwen2.5-7B-Instruct"
    manifest["source_snapshot_path"] = str(missing_snapshot)
    manifest["tokenizer_source_path"] = str(missing_snapshot)
    restored_artifact = tmp_path / "restored-tokenizer.mgx"
    _rewrite_mgx(artifact, restored_artifact, manifest, payload)

    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
        engine = InferenceEngine(
            str(restored_artifact),
            dtype=torch.float16,
            device="cpu",
            max_batch_size=2,
            max_seq_len=32,
        )

    assert str(engine.tokenizer.name_or_path) == str(snapshot)
    kwargs = download.call_args.kwargs
    assert kwargs["repo_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert kwargs["revision"] == revision
    assert "tokenizer*" in kwargs["allow_patterns"]


def test_engine_rejects_tokenizer_hash_mismatch(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    info = inspect_mgx(artifact, validate_payload_hash=False)
    manifest = info["manifest"]
    payload = _read_payload_bytes(artifact)
    manifest["tokenizer_hash"] = "0" * 64

    broken = tmp_path / "bad-tokenizer-hash.mgx"
    _rewrite_mgx(artifact, broken, manifest, payload)

    tokenizer = _FakeTokenizer("<bos>{{ messages[0]['content'] }}<eos>")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        with pytest.raises(MGXFormatError, match="Tokenizer hash mismatch"):
            InferenceEngine(
                str(broken),
                dtype=torch.float16,
                device="cpu",
                max_batch_size=2,
                max_seq_len=32,
            )


def test_prime_payload_cache_writes_reusable_binary(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny-int8.mgx"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="int8")
    cache_info = prime_mgx_payload_cache(artifact, validate_payload_hash=True)

    payload_cache_path = Path(cache_info["payload_cache_path"])
    runtime_cache_path = Path(cache_info["runtime_cache_path"])
    runtime_index_path = Path(cache_info["runtime_cache_index_path"])
    runtime_index = json.loads(runtime_index_path.read_text(encoding="utf-8"))
    assert payload_cache_path.exists()
    assert runtime_cache_path.exists()
    assert runtime_index_path.exists()
    assert cache_info["payload_cache_written"] is True
    assert cache_info["runtime_cache_written"] is True
    assert cache_info["verified"] is True
    assert runtime_index["version"] >= 2
    for entry in runtime_index["entries"]:
        assert int(entry["offset_bytes"]) % 64 == 0


def test_load_from_mgx_prefers_runtime_payload_cache_when_available(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf-runtime")
    artifact = tmp_path / "tiny-runtime.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="none",
        emit_payload_cache=True,
    )

    with patch.dict(os.environ, {"MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "1"}), patch(
        "megagemm.models.mgx._read_tensor_payload",
        side_effect=AssertionError("embedded payload read should not happen"),
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded._load_timing["payload_source"] == "payload_runtime_cache_packed"
    assert loaded._load_timing["runtime_cache_hit"] is True
    assert loaded._load_timing["payload_hydration_mode"] == "runtime_packed"
    assert loaded._load_timing["payload_runtime_cache_blob_count"] > 0
    assert loaded._load_timing["payload_bulk_load_seconds"] >= 0.0
    assert loaded._load_timing["payload_packed_view_seconds"] >= 0.0


def test_load_from_mgx_uses_payload_cache_when_available(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny-int8.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="int8",
        emit_payload_cache=True,
    )

    with patch.dict(os.environ, {"MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "0"}), patch(
        "megagemm.models.mgx._read_tensor_payload",
        side_effect=AssertionError("embedded payload read should not happen"),
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "Int8Linear"
    assert loaded._load_timing["payload_source"] == "payload_cache_streaming"
    assert loaded._load_timing["payload_cache_hit"] is True
    assert loaded._load_timing["payload_open_seconds"] >= 0.0
    assert loaded._load_timing["payload_sorted_keys_seconds"] >= 0.0
    assert loaded._load_timing["payload_get_tensor_seconds"] >= 0.0
    assert loaded._load_timing["payload_device_transfer_seconds"] >= 0.0
    assert loaded._load_timing["payload_pin_memory_seconds"] >= 0.0
    assert loaded._load_timing["payload_dtype_cast_seconds"] >= 0.0
    assert loaded._load_timing["payload_assign_lookup_seconds"] >= 0.0
    assert loaded._load_timing["payload_assign_write_seconds"] >= 0.0
    assert loaded._load_timing["payload_special_case_seconds"] >= 0.0
    assert loaded._load_timing["payload_total_tensor_count"] > 0
    assert loaded._load_timing["payload_pinned_tensor_count"] >= 0
    assert isinstance(loaded._load_timing["payload_top_get_tensor_tensors"], list)
    assert isinstance(loaded._load_timing["payload_top_transfer_tensors"], list)
    assert loaded._load_timing["payload_stream_assign_seconds"] >= (
        loaded._load_timing["payload_get_tensor_seconds"]
        + loaded._load_timing["payload_dtype_cast_seconds"]
        + loaded._load_timing["payload_assign_lookup_seconds"]
        + loaded._load_timing["payload_assign_write_seconds"]
    )


def test_load_from_mgx_uses_payload_cache_when_available_for_int4(tmp_path):
    snapshot = _build_local_awq_snapshot(tmp_path / "hf-awq")
    artifact = tmp_path / "tiny-int4.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="int4",
        emit_payload_cache=True,
    )

    with patch.dict(os.environ, {"MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "0"}), patch(
        "megagemm.models.mgx._read_tensor_payload",
        side_effect=AssertionError("embedded payload read should not happen"),
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded.layers[0].self_attn.qkv_proj.__class__.__name__ == "QuantizedLinear"
    assert loaded._load_timing["payload_source"] == "payload_cache_streaming"
    assert loaded._load_timing["payload_cache_hit"] is True
    assert loaded._load_timing["awq_qweight_layout"] == "awq-decode-transposed-v1"
    assert loaded.layers[0].self_attn.qkv_proj._transposed is True
    assert loaded._load_timing["payload_open_seconds"] >= 0.0
    assert loaded._load_timing["payload_sorted_keys_seconds"] >= 0.0
    assert loaded._load_timing["payload_get_tensor_seconds"] >= 0.0
    assert loaded._load_timing["payload_device_transfer_seconds"] >= 0.0
    assert loaded._load_timing["payload_pin_memory_seconds"] >= 0.0
    assert loaded._load_timing["payload_assign_lookup_seconds"] >= 0.0
    assert loaded._load_timing["payload_assign_write_seconds"] >= 0.0
    assert loaded._load_timing["payload_total_tensor_count"] > 0
    assert loaded._load_timing["payload_pinned_tensor_count"] >= 0
    assert isinstance(loaded._load_timing["payload_top_get_tensor_tensors"], list)
    assert isinstance(loaded._load_timing["payload_top_transfer_tensors"], list)


def test_load_from_mgx_payload_cache_supports_cpu_stage_pinned_override(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf-cpu-stage")
    artifact = tmp_path / "tiny-fp16.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="none",
        emit_payload_cache=True,
    )

    with patch.dict(
        os.environ,
        {
            "MEGAGEMM_MGX_PAYLOAD_HYDRATION": "cpu_stage_pinned",
            "MEGAGEMM_MGX_PAYLOAD_TOPK": "3",
            "MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "0",
        },
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded._load_timing["payload_source"] == "payload_cache_streaming"
    assert loaded._load_timing["payload_cache_hit"] is True
    assert loaded._load_timing["payload_hydration_mode"] == "cpu_stage_pinned"
    assert loaded._load_timing["payload_profile_topk_limit"] == 3
    assert loaded._load_timing["payload_handle_device"] == "cpu"
    assert loaded._load_timing["payload_device_transfer_seconds"] == 0.0
    assert loaded._load_timing["payload_pin_memory_seconds"] == 0.0
    assert loaded._load_timing["payload_pinned_tensor_count"] == 0
    assert len(loaded._load_timing["payload_top_get_tensor_tensors"]) <= 3
    assert len(loaded._load_timing["payload_top_transfer_tensors"]) <= 3


def test_load_from_mgx_payload_cache_supports_cpu_bulk_override(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf-cpu-bulk")
    artifact = tmp_path / "tiny-fp16-bulk.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="none",
        emit_payload_cache=True,
    )

    with patch.dict(
        os.environ,
        {
            "MEGAGEMM_MGX_PAYLOAD_HYDRATION": "cpu_bulk",
            "MEGAGEMM_MGX_PAYLOAD_TOPK": "5",
            "MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "0",
        },
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded._load_timing["payload_source"] == "payload_cache_bulk"
    assert loaded._load_timing["payload_cache_hit"] is True
    assert loaded._load_timing["payload_hydration_mode"] == "cpu_bulk"
    assert loaded._load_timing["payload_handle_device"] == "cpu"
    assert loaded._load_timing["payload_profile_topk_limit"] == 0
    assert loaded._load_timing["payload_bulk_load_seconds"] >= 0.0
    assert loaded._load_timing["payload_stream_assign_seconds"] == 0.0
    assert loaded._load_timing["payload_pin_memory_seconds"] == 0.0
    assert loaded._load_timing["payload_device_transfer_seconds"] == 0.0
    assert loaded._load_timing["payload_top_get_tensor_tensors"] == []
    assert loaded._load_timing["payload_top_transfer_tensors"] == []


def test_load_from_mgx_payload_cache_gpu_bulk_falls_back_to_cpu_without_cuda(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf-gpu-bulk")
    artifact = tmp_path / "tiny-fp16-gpu-bulk.mgx"

    export_to_mgx(
        str(snapshot),
        artifact,
        dtype="fp16",
        quantize="none",
        emit_payload_cache=True,
    )

    with patch.dict(
        os.environ,
        {
            "MEGAGEMM_MGX_PAYLOAD_HYDRATION": "gpu_bulk",
            "MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "0",
        },
    ):
        loaded = load_from_mgx(
            artifact,
            device="cpu",
            dtype_override=torch.float16,
            verify_payload_hash=False,
            prefer_payload_cache=True,
        )

    assert loaded._load_timing["payload_source"] == "payload_cache_bulk"
    assert loaded._load_timing["payload_hydration_mode"] == "cpu_bulk"
    assert loaded._load_timing["payload_handle_device"] == "cpu"


def test_tied_embeddings_stay_shared_in_mgx(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf-tied", tie_word_embeddings=True)
    artifact = tmp_path / "tiny-tied.mgx"

    export_info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    tensor_names = {entry["name"] for entry in export_info["manifest"]["tensor_table"]}

    assert "embed_tokens.weight" in tensor_names
    assert "lm_head.weight" not in tensor_names

    loaded = load_from_mgx(artifact, device="cpu", dtype_override=torch.float16)
    assert loaded.config.tie_word_embeddings is True
    assert loaded.lm_head.weight is loaded.embed_tokens.weight


def test_attach_and_extract_session_state_from_mgx_roundtrip(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    stateful_artifact = tmp_path / "tiny-stateful.mgx"

    export_info = export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    session_snapshot = {
        "seq_len": 5,
        "num_layers": 2,
        "kv_layer_indices": [0, 1],
        "block_size": 16,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "torch.float16",
        "model_name": "synthetic-session",
        "text": "hello prophet",
        "embedding": torch.arange(32, dtype=torch.float32),
        "kv_data_by_layer": {
            0: torch.arange(2 * 2 * 2 * 16 * 8, dtype=torch.float16).reshape(2, 2, 2, 16, 8),
            1: torch.ones((2, 2, 2, 16, 8), dtype=torch.float16),
        },
        "linear_conv_states": {
            1: torch.arange(16, dtype=torch.float32).reshape(1, 16),
        },
        "linear_recurrent_states": {
            1: torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
        },
    }

    info = attach_session_state_to_mgx(
        artifact,
        session_snapshot,
        out_path=stateful_artifact,
    )
    assert info["session_state_present"] is True
    assert info["session_state"]["manifest"]["snapshot"]["seq_len"] == 5

    extracted = extract_session_state_from_mgx(stateful_artifact)
    assert extracted["seq_len"] == session_snapshot["seq_len"]
    assert extracted["text"] == session_snapshot["text"]
    assert extracted["source_model_hash"] == export_info["manifest"]["source_model_hash"]
    assert extracted["tokenizer_hash"] == export_info["manifest"]["tokenizer_hash"]
    assert torch.equal(extracted["embedding"], session_snapshot["embedding"])
    assert torch.equal(
        extracted["kv_data_by_layer"][0],
        session_snapshot["kv_data_by_layer"][0],
    )
    assert torch.equal(
        extracted["linear_conv_states"][1],
        session_snapshot["linear_conv_states"][1],
    )
    assert len(extracted["kv_data"]) == session_snapshot["num_layers"]
    assert torch.equal(extracted["kv_data"][1], session_snapshot["kv_data_by_layer"][1])


def test_engine_save_and_restore_context_via_mgx_session_state(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    stateful_artifact = tmp_path / "tiny-stateful.mgx"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")

    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    seq_id = 3
    input_ids = engine.tokenizer.encode(prompt, return_tensors='pt').to(engine.device)
    positions = torch.arange(input_ids.shape[1], device=engine.device).unsqueeze(0)
    engine.block_manager.allocate_sequence(seq_id, int(input_ids.shape[1]) + 8)
    with torch.inference_mode():
        engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)

    reference_snapshot = engine.save_context(seq_id, text=prompt)
    info = engine.save_context_to_mgx(
        seq_id,
        out_path=str(stateful_artifact),
        text=prompt,
    )
    assert info["session_state_present"] is True

    engine.block_manager.free_sequence(seq_id)

    restored_engine = InferenceEngine(
        str(stateful_artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )
    restored_seq_id = restored_engine.restore_context_from_mgx(seq_id=7, max_new_tokens=8)
    restored_snapshot = restored_engine.save_context(restored_seq_id, text=prompt)

    assert restored_snapshot["seq_len"] == reference_snapshot["seq_len"]
    assert restored_snapshot["source_model_hash"] == reference_snapshot["source_model_hash"]
    assert restored_snapshot["tokenizer_hash"] == reference_snapshot["tokenizer_hash"]
    assert torch.equal(
        restored_snapshot["kv_data_by_layer"][0],
        reference_snapshot["kv_data_by_layer"][0],
    )


def test_prophet_capture_lookup_and_restore_exact_text(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    prophet_dir = tmp_path / "prophet"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    seq_id = 11
    input_ids = engine.tokenizer.encode(prompt, return_tensors='pt').to(engine.device)
    positions = torch.arange(input_ids.shape[1], device=engine.device).unsqueeze(0)
    engine.block_manager.allocate_sequence(seq_id, int(input_ids.shape[1]) + 8)
    with torch.inference_mode():
        engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)
    reference_snapshot = engine.save_context(seq_id, text=prompt)

    entry = engine.prophet_capture(
        str(prophet_dir),
        seq_id,
        text=prompt,
        label="exact-prompt",
    )
    matches = engine.prophet_lookup(str(prophet_dir), prompt, top_k=1, min_similarity=0.0)
    assert len(matches) == 1
    assert matches[0]["entry_id"] == entry["entry_id"]
    assert matches[0]["exact_text_match"] is True
    assert matches[0]["score"] == pytest.approx(1.0)

    engine.block_manager.free_sequence(seq_id)
    result = engine.prophet_restore_best(
        str(prophet_dir),
        prompt,
        seq_id=12,
        max_new_tokens=8,
        min_similarity=0.0,
    )
    assert result["restored"] is True
    restored_snapshot = engine.save_context(result["seq_id"], text=prompt)
    assert restored_snapshot["seq_len"] == reference_snapshot["seq_len"]
    assert restored_snapshot["source_model_hash"] == reference_snapshot["source_model_hash"]
    assert torch.equal(
        restored_snapshot["kv_data_by_layer"][0],
        reference_snapshot["kv_data_by_layer"][0],
    )


def test_prophet_semantic_lookup_prefers_nearest_embedding(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    prophet_dir = tmp_path / "prophet"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )
    library = MGXProphetLibrary(prophet_dir)
    manifest = getattr(engine.model, "_mgx_manifest", None) or {}

    def _snapshot_with_embedding(text: str, emb: torch.Tensor) -> dict:
        return {
            "seq_len": 1,
            "num_layers": engine.config.num_hidden_layers,
            "kv_layer_indices": list(range(engine.config.num_hidden_layers)),
            "block_size": 16,
            "num_kv_heads": engine.config.num_key_value_heads,
            "head_dim": engine.config.head_dim,
            "dtype": "torch.float16",
            "model_name": str(artifact),
            "text": text,
            "embedding": emb.float(),
            "kv_data_by_layer": {},
            "linear_conv_states": {},
            "linear_recurrent_states": {},
            "source_model_hash": manifest.get("source_model_hash"),
            "tokenizer_hash": manifest.get("tokenizer_hash"),
            "chat_template_hash": manifest.get("chat_template_hash"),
            "source_model_id": manifest.get("source_model_id"),
            "quantization": manifest.get("quantization"),
            "target_backend": manifest.get("target_backend"),
        }

    library.record_snapshot(
        _snapshot_with_embedding("math", torch.tensor([1.0, 0.0, 0.0])),
        tokenizer=engine.tokenizer,
        model_name=str(artifact),
    )
    best = library.record_snapshot(
        _snapshot_with_embedding("math-variant", torch.tensor([0.95, 0.05, 0.0])),
        tokenizer=engine.tokenizer,
        model_name=str(artifact),
    )
    library.record_snapshot(
        _snapshot_with_embedding("cooking", torch.tensor([0.0, 1.0, 0.0])),
        tokenizer=engine.tokenizer,
        model_name=str(artifact),
    )

    with patch.object(engine, "extract_embedding", return_value=torch.tensor([0.9, 0.1, 0.0])):
        matches = library.lookup(engine, "question", top_k=3, min_similarity=0.0)

    assert matches
    assert matches[0]["entry_id"] == best["entry_id"]
    assert matches[0]["semantic_similarity"] > matches[-1]["semantic_similarity"]


def test_prefill_context_restore_and_continue_generation(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    prefill = engine.prefill_context(prompt, seq_id=21, max_new_tokens=8)
    saved = engine.save_context(prefill["seq_id"], text=prompt)

    assert saved["token_ids"]
    assert torch.is_tensor(saved["pending_next_logits"])
    assert saved["pending_next_logits"].ndim == 1

    baseline = engine.generate_from_context(
        prefill["seq_id"],
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    engine.free_sequence(prefill["seq_id"])

    restored_engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )
    restored_seq_id = restored_engine.restore_context(saved, seq_id=22, max_new_tokens=8)
    restored = restored_engine.generate_from_context(
        restored_seq_id,
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert restored["token_ids"] == baseline["token_ids"]
    assert restored["pending_next_logits_available"] is True
    restored_engine.free_sequence(restored_seq_id)


def test_truncate_context_and_replay_tokens_restore_exact_state(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "patched replay prompt"
    replay_token_ids = [3, 7, 11, 13]

    def _patched_prepare_prompt_inputs(_prompt: str):
        return "<patched>", torch.tensor([replay_token_ids], dtype=torch.long, device=engine.device)

    with patch.object(engine, "_prepare_prompt_inputs", side_effect=_patched_prepare_prompt_inputs):
        prefill = engine.prefill_context(prompt, seq_id=41, max_new_tokens=8)
    reference = engine.save_context(prefill["seq_id"], text=prompt)
    prefix_len = 2
    tail_token_ids = reference["token_ids"][prefix_len:]

    baseline_seq_id = engine.restore_context(reference, seq_id=42, max_new_tokens=8)
    engine.truncate_context(baseline_seq_id, prefix_len)
    baseline_tokens = list(engine._seq_token_ids.get(baseline_seq_id, []))
    baseline_logits = None
    for token_id in tail_token_ids:
        decode_input = torch.tensor([[token_id]], dtype=torch.long, device=engine.device)
        decode_pos = torch.tensor(
            [[int(engine.block_manager.seq_lens[baseline_seq_id])]],
            dtype=torch.long,
            device=engine.device,
        )
        decode_result = engine.model.decode_step(
            decode_input,
            decode_pos,
            engine.block_manager,
            [baseline_seq_id],
        )
        baseline_logits = decode_result[0] if isinstance(decode_result, tuple) else decode_result
        baseline_tokens.append(int(token_id))
    engine._seq_token_ids[baseline_seq_id] = baseline_tokens
    if baseline_logits is not None:
        engine._seq_pending_logits[baseline_seq_id] = engine._clone_pending_logits(
            baseline_logits[:, -1, :]
        )
    baseline_snapshot = engine.save_context(baseline_seq_id, text=prompt)
    engine.free_sequence(baseline_seq_id)

    truncated = engine.truncate_context(prefill["seq_id"], prefix_len)
    assert truncated["seq_len"] == prefix_len
    assert truncated["pending_next_logits_available"] is False

    replay = engine.replay_tokens_into_context(
        prefill["seq_id"],
        tail_token_ids,
    )
    assert replay["pending_next_logits_available"] is True
    if len(tail_token_ids) > 1:
        assert replay["replay_mode"] == "suffix_prefill"

    rebuilt = engine.save_context(prefill["seq_id"], text=prompt)
    assert rebuilt["seq_len"] == baseline_snapshot["seq_len"]
    assert rebuilt["token_ids"] == baseline_snapshot["token_ids"]
    assert int(rebuilt["pending_next_logits"].argmax().item()) == int(
        baseline_snapshot["pending_next_logits"].argmax().item()
    )
    assert torch.allclose(
        rebuilt["pending_next_logits"],
        baseline_snapshot["pending_next_logits"],
        atol=1e-3,
        rtol=1e-3,
    )
    for layer_idx, ref_layer in baseline_snapshot["kv_data_by_layer"].items():
        assert torch.allclose(
            rebuilt["kv_data_by_layer"][layer_idx],
            ref_layer,
            atol=1e-3,
            rtol=1e-3,
        )

    engine.free_sequence(prefill["seq_id"])


def test_prophet_restore_speculative_exact_and_fallback(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    prophet_dir = tmp_path / "prophet"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    prefill = engine.prefill_context(prompt, seq_id=31, max_new_tokens=8)
    entry = engine.prophet_capture(
        str(prophet_dir),
        prefill["seq_id"],
        text=prompt,
        label="speculative-reference",
    )
    engine.free_sequence(prefill["seq_id"])

    exact_result = engine.prophet_restore_speculative(
        str(prophet_dir),
        prompt,
        seq_id=32,
        max_new_tokens=8,
        min_similarity=0.0,
        validation_mode="full_prefill",
        validation_tokens=2,
        agreement_threshold=1.0,
        fallback_to_prefill=True,
    )
    assert exact_result["restored"] is True
    assert exact_result["speculative_accepted"] is True
    assert exact_result["committed_source"] == "prophet_exact"
    assert exact_result["match"]["entry_id"] == entry["entry_id"]
    engine.free_sequence(exact_result["seq_id"])

    with patch.object(
        engine,
        "generate_from_context",
        side_effect=[
            {"seq_id": 33, "text": "alpha", "token_ids": [1, 2], "stopped": False},
            {"seq_id": 34, "text": "beta", "token_ids": [3, 4], "stopped": False},
        ],
    ):
        fallback_result = engine.prophet_restore_speculative(
            str(prophet_dir),
            "completely different question",
            seq_id=33,
            max_new_tokens=8,
            min_similarity=0.0,
            validation_mode="full_prefill",
            validation_tokens=2,
            agreement_threshold=1.0,
            fallback_to_prefill=True,
        )

    assert fallback_result["restored"] is True
    assert fallback_result["speculative_accepted"] is False
    assert fallback_result["committed_source"] == "prefill_fallback"
    assert fallback_result["reason"] == "validation_failed"
    assert fallback_result["validation"]["accepted"] is False
    assert fallback_result["validation"]["first_token_match"] is False
    engine.free_sequence(fallback_result["seq_id"])


def test_prophet_restore_speculative_prefix_reuse(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    prophet_dir = tmp_path / "prophet"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    prefill = engine.prefill_context(prompt, seq_id=51, max_new_tokens=8)
    reference = engine.save_context(prefill["seq_id"], text=prompt)
    entry = engine.prophet_capture(
        str(prophet_dir),
        prefill["seq_id"],
        text=prompt,
        label="prefix-reference",
    )

    appended_token = int(reference["token_ids"][-1])
    query_token_ids = list(reference["token_ids"]) + [appended_token]

    baseline_seq_id = engine.restore_context(reference, seq_id=52, max_new_tokens=8)
    engine.replay_tokens_into_context(baseline_seq_id, [appended_token])
    baseline_snapshot = engine.save_context(baseline_seq_id)
    engine.free_sequence(baseline_seq_id)
    engine.free_sequence(prefill["seq_id"])

    def _patched_prepare_prompt_inputs(_prompt: str):
        return "<patched>", torch.tensor([query_token_ids], dtype=torch.long, device=engine.device)

    with patch.object(engine, "_prepare_prompt_inputs", side_effect=_patched_prepare_prompt_inputs):
        prefix_result = engine.prophet_restore_speculative(
            str(prophet_dir),
            "patched prefix query",
            seq_id=53,
            max_new_tokens=8,
            top_k=1,
            min_similarity=1.0,
            prefix_tokens=1,
            validation_mode="full_prefill",
            validation_tokens=2,
            agreement_threshold=1.0,
            fallback_to_prefill=True,
        )

    assert prefix_result["restored"] is True
    assert prefix_result["committed_source"] == "prophet_prefix_reuse"
    assert prefix_result["speculative_accepted"] is True
    assert prefix_result["reason"] == "token_prefix_reuse"
    assert prefix_result["match"]["entry_id"] == entry["entry_id"]
    assert prefix_result["validation"]["policy"]["accepted"] is True
    assert prefix_result["validation"]["policy"]["rejected_reasons"] == []
    assert prefix_result["validation"]["policy"]["common_prefix_tokens"] == len(reference["token_ids"])

    restored_snapshot = engine.save_context(prefix_result["seq_id"])
    assert restored_snapshot["seq_len"] == baseline_snapshot["seq_len"]
    assert restored_snapshot["token_ids"] == baseline_snapshot["token_ids"]
    assert torch.equal(restored_snapshot["pending_next_logits"], baseline_snapshot["pending_next_logits"])
    for layer_idx, ref_layer in baseline_snapshot["kv_data_by_layer"].items():
        assert torch.equal(restored_snapshot["kv_data_by_layer"][layer_idx], ref_layer)

    engine.free_sequence(prefix_result["seq_id"])


def test_prophet_restore_speculative_prefix_policy_rejects_weak_reuse(tmp_path):
    snapshot = _build_local_snapshot(tmp_path / "hf")
    artifact = tmp_path / "tiny.mgx"
    prophet_dir = tmp_path / "prophet"

    export_to_mgx(str(snapshot), artifact, dtype="fp16", quantize="none")
    engine = InferenceEngine(
        str(artifact),
        dtype=torch.float16,
        device="cpu",
        max_batch_size=2,
        max_seq_len=32,
    )

    prompt = "abc"
    prefill = engine.prefill_context(prompt, seq_id=61, max_new_tokens=8)
    reference = engine.save_context(prefill["seq_id"], text=prompt)
    engine.prophet_capture(
        str(prophet_dir),
        prefill["seq_id"],
        text=prompt,
        label="prefix-reference",
    )
    engine.free_sequence(prefill["seq_id"])

    weak_query_token_ids = [int(reference["token_ids"][0])] + [int(reference["token_ids"][-1])] * 15

    def _patched_prepare_prompt_inputs(_prompt: str):
        return "<patched>", torch.tensor([weak_query_token_ids], dtype=torch.long, device=engine.device)

    with patch.object(engine, "_prepare_prompt_inputs", side_effect=_patched_prepare_prompt_inputs):
        weak_result = engine.prophet_restore_speculative(
            str(prophet_dir),
            "patched weak prefix query",
            seq_id=62,
            max_new_tokens=8,
            top_k=1,
            min_similarity=1.0,
            prefix_tokens=1,
            validation_mode="full_prefill",
            validation_tokens=2,
            agreement_threshold=1.0,
            fallback_to_prefill=False,
            min_prefix_reuse_score=0.55,
            min_prefix_coverage=0.50,
            max_prefix_rollback_ratio=0.35,
            max_prefix_tail_ratio=0.50,
        )

    assert weak_result["restored"] is False
    assert weak_result["reason"] == "no_match"
    assert weak_result["validation"]["policy"]["accepted"] is False
    assert "prefix_coverage_below_threshold" in weak_result["validation"]["policy"]["rejected_reasons"]
    assert "tail_ratio_above_threshold" in weak_result["validation"]["policy"]["rejected_reasons"]
