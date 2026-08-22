"""
Native BERT-family encoder optimized for embedding inference.

First scope:
- BertModel-compatible checkpoints with absolute position embeddings
- Fused QKV projection per layer
- Lean attention/MLP forward path for inference
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.paged_attention import (
    PackedAttentionMetadata,
    packed_attention,
    prepare_packed_attention_metadata,
)

__all__ = [
    "NativeBertEncoder",
    "is_native_bert_supported",
    "load_native_bert_encoder",
]


def _resolve_hidden_activation(name: str):
    key = str(name).strip().lower()
    if key == "gelu":
        return lambda x: F.gelu(x)
    if key in {"gelu_new", "gelu_fast"}:
        return lambda x: F.gelu(x, approximate="tanh")
    if key == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if key == "relu":
        return F.relu
    if key in {"silu", "swish"}:
        return F.silu
    if key == "tanh":
        return torch.tanh
    return F.gelu


def _load_state_dict_from_dir(model_dir: str) -> dict:
    root = Path(model_dir)
    safetensors_files = sorted(root.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to load native BERT encoder checkpoints: pip install safetensors"
            ) from exc
        state = {}
        for file_path in safetensors_files:
            state.update(load_file(str(file_path)))
        return state

    bin_files = sorted(root.glob("pytorch_model*.bin"))
    if bin_files:
        state = {}
        for file_path in bin_files:
            loaded = torch.load(str(file_path), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                loaded = loaded["state_dict"]
            state.update(loaded)
        return state

    raise FileNotFoundError(f"No model weights found under {model_dir}")


_NATIVE_BERT_CORE_PREFIXES = (
    "embeddings.",
    "encoder.",
    "pooler.",
)
_NATIVE_BERT_WRAPPER_PREFIXES = (
    "module.",
    "_orig_mod.",
    "base_model.",
    "model.",
    "bert.",
)


def _normalize_native_bert_state_dict(state_dict: dict) -> dict:
    normalized = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            if key.startswith(_NATIVE_BERT_CORE_PREFIXES):
                break
            for prefix in _NATIVE_BERT_WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key.endswith(".gamma"):
            key = key[:-6] + ".weight"
        elif key.endswith(".beta"):
            key = key[:-5] + ".bias"
        normalized[key] = value
    return normalized


def _require_state_key(state_dict: dict, key: str) -> torch.Tensor:
    if key not in state_dict:
        sample = ", ".join(sorted(list(state_dict.keys()))[:8])
        raise KeyError(
            f"Missing required BERT backbone weight: {key}. "
            f"Available keys start with: {sample}"
        )
    return state_dict[key]


def is_native_bert_supported(config) -> bool:
    model_type = str(getattr(config, "model_type", "")).lower()
    position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
    if model_type != "bert":
        return False
    if position_embedding_type != "absolute":
        return False
    return True


def _attention_mask_to_cu_seqlens(attention_mask: torch.Tensor):
    seq_lens = attention_mask.to(dtype=torch.int32).sum(dim=1)
    cu_seqlens = torch.zeros(
        seq_lens.shape[0] + 1,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    cu_seqlens[1:] = seq_lens.cumsum(dim=0)
    return seq_lens, cu_seqlens


class _NativeBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        type_vocab = int(getattr(config, "type_vocab_size", 0) or 0)
        self.token_type_embeddings = (
            nn.Embedding(type_vocab, config.hidden_size) if type_vocab > 0 else None
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.word_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(position_ids)

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            hidden_states = hidden_states + self.token_type_embeddings(token_type_ids)

        return self.layer_norm(hidden_states)


class _NativeBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(config.intermediate_size)
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.attn_out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.mlp_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.output_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.activation = _resolve_hidden_activation(getattr(config, "hidden_act", "gelu"))

    def forward(self, hidden_states: torch.Tensor, attention_bias: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn = self.attn_out_proj(attn)
        hidden_states = self.attn_layer_norm(residual + attn)

        residual = hidden_states
        hidden_states = self.mlp_up_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.mlp_down_proj(hidden_states)
        hidden_states = self.output_layer_norm(residual + hidden_states)
        return hidden_states

    def forward_packed(
        self,
        packed_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        packed_meta: Optional[PackedAttentionMetadata] = None,
    ) -> torch.Tensor:
        total_tokens = packed_hidden_states.shape[0]
        residual = packed_hidden_states

        qkv = self.qkv_proj(packed_hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_heads, self.head_dim)
        v = v.view(total_tokens, self.num_heads, self.head_dim)

        attn = packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            causal=False,
            packed_meta=packed_meta,
        )
        attn = attn.reshape(total_tokens, self.hidden_size)
        attn = self.attn_out_proj(attn)
        hidden_states = self.attn_layer_norm(residual + attn)

        residual = hidden_states
        hidden_states = self.mlp_up_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.mlp_down_proj(hidden_states)
        hidden_states = self.output_layer_norm(residual + hidden_states)
        return hidden_states


class NativeBertEncoder(nn.Module):
    def __init__(
        self,
        config,
        padding_free: bool = False,
        padding_free_force: bool = False,
    ):
        super().__init__()
        self.config = config
        self.padding_free = bool(padding_free)
        self.padding_free_force = bool(padding_free_force)
        self.embeddings = _NativeBertEmbeddings(config)
        self.layers = nn.ModuleList([
            _NativeBertLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def _build_attention_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        neg = torch.finfo(dtype).min
        return (1.0 - mask) * neg

    def _should_use_padding_free(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> bool:
        if not self.padding_free or attention_mask is None or attention_mask.ndim != 2:
            return False
        if hidden_states.device.type != "cuda" and not self.padding_free_force:
            return False
        if attention_mask.shape[0] <= 1:
            return False
        seq_lens = attention_mask.to(dtype=torch.int32).sum(dim=1)
        if seq_lens.numel() == 0:
            return False
        return bool((seq_lens.min() != seq_lens.max()).item())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.embeddings(input_ids, token_type_ids=token_type_ids)
        if self._should_use_padding_free(hidden_states, attention_mask):
            pack_mask = attention_mask.to(dtype=torch.bool)
            _, cu_seqlens = _attention_mask_to_cu_seqlens(attention_mask)
            packed_meta = None
            packed_cu_seqlens = cu_seqlens
            if self.layers:
                packed_meta = prepare_packed_attention_metadata(
                    cu_seqlens,
                    head_dim=self.layers[0].head_dim,
                )
                packed_cu_seqlens = packed_meta.cu_seqlens
            packed_hidden_states = hidden_states[pack_mask]
            for layer in self.layers:
                packed_hidden_states = layer.forward_packed(
                    packed_hidden_states,
                    packed_cu_seqlens,
                    packed_meta=packed_meta,
                )
            output = hidden_states.new_zeros(hidden_states.shape)
            output[pack_mask] = packed_hidden_states
            return output
        attention_bias = self._build_attention_bias(attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        return hidden_states

    @classmethod
    def from_hf_state_dict(
        cls,
        config,
        state_dict: dict,
        padding_free: bool = False,
        padding_free_force: bool = False,
    ) -> "NativeBertEncoder":
        state_dict = _normalize_native_bert_state_dict(state_dict)
        model = cls(
            config,
            padding_free=padding_free,
            padding_free_force=padding_free_force,
        )
        model.embeddings.word_embeddings.weight.data.copy_(_require_state_key(state_dict, "embeddings.word_embeddings.weight"))
        model.embeddings.position_embeddings.weight.data.copy_(_require_state_key(state_dict, "embeddings.position_embeddings.weight"))
        if model.embeddings.token_type_embeddings is not None and "embeddings.token_type_embeddings.weight" in state_dict:
            model.embeddings.token_type_embeddings.weight.data.copy_(state_dict["embeddings.token_type_embeddings.weight"])
        model.embeddings.layer_norm.weight.data.copy_(_require_state_key(state_dict, "embeddings.LayerNorm.weight"))
        model.embeddings.layer_norm.bias.data.copy_(_require_state_key(state_dict, "embeddings.LayerNorm.bias"))

        for idx, layer in enumerate(model.layers):
            prefix = f"encoder.layer.{idx}"

            q_w = _require_state_key(state_dict, f"{prefix}.attention.self.query.weight")
            k_w = _require_state_key(state_dict, f"{prefix}.attention.self.key.weight")
            v_w = _require_state_key(state_dict, f"{prefix}.attention.self.value.weight")
            q_b = _require_state_key(state_dict, f"{prefix}.attention.self.query.bias")
            k_b = _require_state_key(state_dict, f"{prefix}.attention.self.key.bias")
            v_b = _require_state_key(state_dict, f"{prefix}.attention.self.value.bias")
            layer.qkv_proj.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            layer.qkv_proj.bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))

            layer.attn_out_proj.weight.data.copy_(_require_state_key(state_dict, f"{prefix}.attention.output.dense.weight"))
            layer.attn_out_proj.bias.data.copy_(_require_state_key(state_dict, f"{prefix}.attention.output.dense.bias"))
            layer.attn_layer_norm.weight.data.copy_(_require_state_key(state_dict, f"{prefix}.attention.output.LayerNorm.weight"))
            layer.attn_layer_norm.bias.data.copy_(_require_state_key(state_dict, f"{prefix}.attention.output.LayerNorm.bias"))
            layer.mlp_up_proj.weight.data.copy_(_require_state_key(state_dict, f"{prefix}.intermediate.dense.weight"))
            layer.mlp_up_proj.bias.data.copy_(_require_state_key(state_dict, f"{prefix}.intermediate.dense.bias"))
            layer.mlp_down_proj.weight.data.copy_(_require_state_key(state_dict, f"{prefix}.output.dense.weight"))
            layer.mlp_down_proj.bias.data.copy_(_require_state_key(state_dict, f"{prefix}.output.dense.bias"))
            layer.output_layer_norm.weight.data.copy_(_require_state_key(state_dict, f"{prefix}.output.LayerNorm.weight"))
            layer.output_layer_norm.bias.data.copy_(_require_state_key(state_dict, f"{prefix}.output.LayerNorm.bias"))

        return model


def load_native_bert_encoder(
    model_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    padding_free: bool = False,
    padding_free_force: bool = False,
):
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "Install transformers to use the native BERT embedding backend: pip install transformers"
        ) from exc

    config = AutoConfig.from_pretrained(model_dir, local_files_only=os.path.isdir(model_dir))
    if not is_native_bert_supported(config):
        raise ValueError(
            f"Native BERT backend does not support model_type={getattr(config, 'model_type', None)!r}"
        )

    state_dict = _load_state_dict_from_dir(model_dir)
    model = NativeBertEncoder.from_hf_state_dict(
        config,
        state_dict,
        padding_free=padding_free,
        padding_free_force=padding_free_force,
    )
    model.eval()
    model.to(device=device)
    if dtype != torch.float32 or device == "cuda":
        model.to(dtype=dtype)
    return model
