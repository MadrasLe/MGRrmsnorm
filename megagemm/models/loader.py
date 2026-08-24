"""
📦 HuggingFace Model Loader for MegaGemm
-----------------------------------------
Downloads and loads model weights from HuggingFace.
Supports safetensors and handles weight name mapping.

Supported: LLaMA 2/3, TinyLlama, Mistral, CodeLlama,
           Qwen 2.5, Qwen 3, Gemma 2
           + AWQ INT4 quantized versions

Author: Gabriel Yogi
"""

import torch
import json
import os
import time
from typing import Callable, Optional, Dict, Tuple
from pathlib import Path

from .llama import (
    LlamaConfig,
    MegaGemmLlama,
    SUPPORTED_MODELS,
    _precompute_layer_rope_cache,
)

__all__ = ['load_from_hf', 'resolve_model_source']


def _parse_int8_skip_ops() -> set[str]:
    """
    Parse MEGAGEMM_INT8_SKIP_OPS (comma-separated) for selective INT8 quantization.

    Supported canonical ops:
      - qkv
      - o_proj
      - gate_up
      - down
    """
    raw = os.environ.get("MEGAGEMM_INT8_SKIP_OPS", "")
    if not raw:
        return set()

    alias = {
        "qkv": "qkv",
        "qkv_proj": "qkv",
        "o": "o_proj",
        "o_proj": "o_proj",
        "gate_up": "gate_up",
        "gate_up_proj": "gate_up",
        "down": "down",
        "down_proj": "down",
    }
    skip = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        mapped = alias.get(key)
        if mapped is not None:
            skip.add(mapped)
    return skip


def _download_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Download model files from HuggingFace Hub. Returns local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface_hub"
        )

    path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    )
    return path


def resolve_model_source(model_name: str, cache_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve a model identifier into a local snapshot path.

    Returns:
        (local_path, source_kind) where source_kind is 'local' or 'huggingface'
    """
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve()), 'local'
    return _download_model(model_name, cache_dir), 'huggingface'


def _load_config(model_path: str) -> dict:
    """Load config.json from model directory."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def _normalize_hf_weight_key(key: str) -> str:
    """Flatten multimodal text-backbone prefixes to decoder-only names."""
    prefix_map = (
        ('model.language_model.model.', 'model.'),
        ('language_model.model.', 'model.'),
        ('model.text_model.model.', 'model.'),
        ('text_model.model.', 'model.'),
        ('model.language_model.lm_head.', 'lm_head.'),
        ('language_model.lm_head.', 'lm_head.'),
        ('model.text_model.lm_head.', 'lm_head.'),
        ('text_model.lm_head.', 'lm_head.'),
        ('model.language_model.', 'model.'),
        ('language_model.', 'model.'),
        ('model.text_model.', 'model.'),
        ('text_model.', 'model.'),
    )
    for prefix, replacement in prefix_map:
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    return key


def _iter_safetensor_files(model_path: str) -> list[str]:
    """Find safetensors shards in the snapshot, including nested layouts."""
    files = []
    for root, _, filenames in os.walk(model_path):
        for fname in filenames:
            if fname.endswith('.safetensors'):
                files.append(os.path.join(root, fname))
    files.sort()
    return files


def _is_text_backbone_weight(key: str) -> bool:
    """Keep only decoder/text-backbone tensors from multimodal checkpoints."""
    key = _normalize_hf_weight_key(key)
    return (
        key == 'model.embed_tokens.weight'
        or key == 'model.embed_tokens_per_layer.weight'
        or key == 'model.per_layer_model_projection.weight'
        or key == 'model.per_layer_projection_norm.weight'
        or key == 'model.norm.weight'
        or key == 'lm_head.weight'
        or key.startswith('model.layers.')
    )


def _load_safetensors(
    model_path: str,
    device: str = 'cpu',
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, torch.Tensor]:
    """Load all safetensors files from model directory.

    Args:
        device: Target device ('cpu' or 'cuda'). Loading directly to GPU
                avoids ~2x peak CPU RAM usage for large models.
        key_filter: Optional predicate to skip unrelated tensors (for example,
                    vision weights in multimodal checkpoints).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")

    weights = {}
    for filepath in _iter_safetensor_files(model_path):
        with safe_open(filepath, framework='pt', device=device) as f:
            for raw_key in f.keys():
                if key_filter is not None and not key_filter(raw_key):
                    continue
                weights[_normalize_hf_weight_key(raw_key)] = f.get_tensor(raw_key)

    if not weights:
        raise FileNotFoundError(f"No .safetensors files found under {model_path}")

    return weights


def _build_safetensor_index(
    model_path: str,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Tuple[str, str]]:
    """Build normalized key -> (file path, raw key) without loading tensors."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")

    index: Dict[str, Tuple[str, str]] = {}
    for filepath in _iter_safetensor_files(model_path):
        with safe_open(filepath, framework='pt') as f:
            for raw_key in f.keys():
                if key_filter is not None and not key_filter(raw_key):
                    continue
                index[_normalize_hf_weight_key(raw_key)] = (filepath, raw_key)
    if not index:
        raise FileNotFoundError(f"No .safetensors weights found under {model_path}")
    return index


def _load_gemma4_fp16_streaming(
    model: MegaGemmLlama,
    config: LlamaConfig,
    model_path: str,
    dtype: torch.dtype,
    device: str,
    *,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> None:
    """Stream Gemma 4 text weights directly into the meta-initialized model."""
    from safetensors import safe_open
    import gc

    key_to_file = _build_safetensor_index(model_path, key_filter=key_filter)
    handles = {}

    def _get(key: str) -> torch.Tensor:
        fpath, raw_key = key_to_file[key]
        handle = handles.get(fpath)
        if handle is None:
            handle = safe_open(fpath, framework='pt', device=device)
            handles[fpath] = handle
        tensor = handle.get_tensor(raw_key)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.detach().clone(memory_format=torch.contiguous_format)

    def _set_weight(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        module.weight = torch.nn.Parameter(_get(key))

    def _set_bias(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        module.bias = torch.nn.Parameter(_get(key))

    def _set_norm(norm_module, key: str) -> None:
        if norm_module is None or key not in key_to_file:
            return
        if not getattr(norm_module, "with_scale", True):
            return
        norm_module.weight = torch.nn.Parameter(_get(key))

    def _set_norm_any(norm_module, *keys: str) -> None:
        if norm_module is None or not getattr(norm_module, "with_scale", True):
            return
        for key in keys:
            if key in key_to_file:
                norm_module.weight = torch.nn.Parameter(_get(key))
                return

    def _set_parameter(module, attr: str, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        module._parameters[attr] = torch.nn.Parameter(_get(key))

    def _set_gemma4_moe(mlp, hf_pre: str) -> None:
        _set_weight(mlp.gate.proj, f'{hf_pre}.router.proj.weight')
        _set_parameter(mlp.gate, 'scale', f'{hf_pre}.router.scale')
        _set_parameter(mlp.gate, 'per_expert_scale', f'{hf_pre}.router.per_expert_scale')
        _set_parameter(mlp.experts, 'gate_up_proj', f'{hf_pre}.experts.gate_up_proj')
        _set_parameter(mlp.experts, 'down_proj', f'{hf_pre}.experts.down_proj')

    _set_weight(model.embed_tokens, 'model.embed_tokens.weight')
    _set_weight(model.embed_tokens_per_layer, 'model.embed_tokens_per_layer.weight')
    _set_weight(model.per_layer_model_projection, 'model.per_layer_model_projection.weight')
    _set_norm(model.per_layer_projection_norm, 'model.per_layer_projection_norm.weight')
    _set_norm(model.norm, 'model.norm.weight')
    if 'lm_head.weight' in key_to_file:
        _set_weight(model.lm_head, 'lm_head.weight')
    elif config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    for i, layer in enumerate(model.layers):
        hf_pre = f'model.layers.{i}'
        attn = layer.self_attn
        mlp = layer.mlp

        _set_weight(attn.q_proj, f'{hf_pre}.self_attn.q_proj.weight')
        _set_bias(attn.q_proj, f'{hf_pre}.self_attn.q_proj.bias')
        _set_weight(attn.k_proj, f'{hf_pre}.self_attn.k_proj.weight')
        _set_bias(attn.k_proj, f'{hf_pre}.self_attn.k_proj.bias')
        _set_weight(attn.v_proj, f'{hf_pre}.self_attn.v_proj.weight')
        _set_bias(attn.v_proj, f'{hf_pre}.self_attn.v_proj.bias')
        _set_weight(attn.o_proj, f'{hf_pre}.self_attn.o_proj.weight')
        _set_bias(attn.o_proj, f'{hf_pre}.self_attn.o_proj.bias')
        _set_norm(attn.q_norm, f'{hf_pre}.self_attn.q_norm.weight')
        _set_norm(attn.k_norm, f'{hf_pre}.self_attn.k_norm.weight')

        dense_mlp = mlp.shared_mlp if config.is_moe_layer(i) else mlp
        gate_key = f'{hf_pre}.mlp.gate_proj.weight'
        up_key = f'{hf_pre}.mlp.up_proj.weight'
        if gate_key in key_to_file and up_key in key_to_file:
            gate_w = _get(gate_key)
            up_w = _get(up_key)
            dense_mlp.gate_up_proj.weight = torch.nn.Parameter(
                torch.cat([gate_w, up_w], dim=0).contiguous()
            )
            del gate_w, up_w
        _set_weight(dense_mlp.down_proj, f'{hf_pre}.mlp.down_proj.weight')
        if config.is_moe_layer(i):
            _set_gemma4_moe(mlp, hf_pre)

        _set_norm(layer.input_layernorm, f'{hf_pre}.input_layernorm.weight')
        _set_norm(layer.post_attention_layernorm, f'{hf_pre}.post_attention_layernorm.weight')
        _set_norm(layer.pre_feedforward_layernorm, f'{hf_pre}.pre_feedforward_layernorm.weight')
        _set_norm(layer.post_feedforward_layernorm, f'{hf_pre}.post_feedforward_layernorm.weight')
        _set_norm_any(
            layer.pre_feedforward_layernorm_2,
            f'{hf_pre}.pre_feedforward_layernorm_2.weight',
            f'{hf_pre}.pre_feedforward_layernorm.weight',
        )
        _set_norm_any(
            layer.post_feedforward_layernorm_1,
            f'{hf_pre}.post_feedforward_layernorm_1.weight',
            f'{hf_pre}.post_feedforward_layernorm.weight',
        )
        _set_norm_any(
            layer.post_feedforward_layernorm_2,
            f'{hf_pre}.post_feedforward_layernorm_2.weight',
            f'{hf_pre}.post_feedforward_layernorm.weight',
        )
        _set_weight(layer.per_layer_input_gate, f'{hf_pre}.per_layer_input_gate.weight')
        _set_weight(layer.per_layer_projection, f'{hf_pre}.per_layer_projection.weight')
        _set_norm(layer.post_per_layer_input_norm, f'{hf_pre}.post_per_layer_input_norm.weight')
        scalar_key = f'{hf_pre}.layer_scalar'
        if scalar_key in key_to_file:
            layer.layer_scalar = _get(scalar_key)

        if (i + 1) % 8 == 0 or i == config.num_hidden_layers - 1:
            gc.collect()
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    handles.clear()
    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    meta_params = [name for name, param in model.named_parameters() if param.device.type == 'meta']
    if meta_params:
        sample = ", ".join(meta_params[:8])
        raise RuntimeError(f"Gemma4 streaming load left meta parameters: {sample}")


def _load_fp16_streaming(
    model: MegaGemmLlama,
    config: LlamaConfig,
    model_path: str,
    dtype: torch.dtype,
    device: str,
    *,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> None:
    """Stream dense FP16/BF16 weights layer-by-layer to avoid GPU load spikes."""
    from safetensors import safe_open
    from contextlib import ExitStack
    import gc

    key_to_file = _build_safetensor_index(model_path, key_filter=key_filter)
    exit_stack = ExitStack()
    handles = {}

    def _get_cpu(key: str) -> torch.Tensor:
        fpath, raw_key = key_to_file[key]
        handle = handles.get(fpath)
        if handle is None:
            handle = exit_stack.enter_context(safe_open(fpath, framework='pt', device='cpu'))
            handles[fpath] = handle
        tensor = handle.get_tensor(raw_key)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.detach().contiguous()

    def _as_param(tensor: torch.Tensor) -> torch.nn.Parameter:
        param = tensor.to(device=device, dtype=dtype).contiguous()
        if param.device.type == 'cpu':
            param = param.clone(memory_format=torch.contiguous_format)
        return torch.nn.Parameter(param)

    def _set_weight(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        weight = _get_cpu(key)
        module.weight = _as_param(weight)
        del weight

    def _set_bias(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        bias = _get_cpu(key)
        module.bias = _as_param(bias)
        del bias

    def _set_norm(norm_module, key: str) -> None:
        if norm_module is None or key not in key_to_file:
            return
        if not getattr(norm_module, "with_scale", True):
            return
        weight = _get_cpu(key)
        norm_module.weight = _as_param(weight)
        del weight

    def _set_parameter(module, attr: str, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        tensor = _get_cpu(key)
        module._parameters[attr] = _as_param(tensor)
        del tensor

    def _set_qwen3_moe_experts(mlp, hf_pre: str) -> None:
        """Load Qwen3 MoE expert tensors from stacked or per-expert HF layouts."""
        stacked_gate_up = f'{hf_pre}.mlp.experts.gate_up_proj'
        stacked_down = f'{hf_pre}.mlp.experts.down_proj'
        if stacked_gate_up in key_to_file and stacked_down in key_to_file:
            _set_parameter(mlp.experts, 'gate_up_proj', stacked_gate_up)
            _set_parameter(mlp.experts, 'down_proj', stacked_down)
            return

        num_experts = int(config.num_experts)
        hidden_size = int(config.hidden_size)
        moe_intermediate = int(config.moe_intermediate_size)
        if num_experts <= 0 or moe_intermediate <= 0:
            raise RuntimeError(f"Invalid Qwen3 MoE expert config at {hf_pre}")

        required = []
        for expert_idx in range(num_experts):
            expert_pre = f'{hf_pre}.mlp.experts.{expert_idx}'
            required.extend(
                [
                    f'{expert_pre}.gate_proj.weight',
                    f'{expert_pre}.up_proj.weight',
                    f'{expert_pre}.down_proj.weight',
                ]
            )
        missing = [key for key in required if key not in key_to_file]
        if missing:
            sample = ", ".join(missing[:6])
            raise RuntimeError(
                f"Qwen3 MoE expert weights for {hf_pre} use an unsupported layout "
                f"or are incomplete. Missing: {sample}"
            )

        gate_up = torch.empty(
            (num_experts, 2 * moe_intermediate, hidden_size),
            dtype=dtype,
            device='cpu',
        )
        down = torch.empty(
            (num_experts, hidden_size, moe_intermediate),
            dtype=dtype,
            device='cpu',
        )

        for expert_idx in range(num_experts):
            expert_pre = f'{hf_pre}.mlp.experts.{expert_idx}'
            gate_w = _get_cpu(f'{expert_pre}.gate_proj.weight')
            up_w = _get_cpu(f'{expert_pre}.up_proj.weight')
            down_w = _get_cpu(f'{expert_pre}.down_proj.weight')

            gate_up[expert_idx, :moe_intermediate].copy_(gate_w)
            gate_up[expert_idx, moe_intermediate:].copy_(up_w)
            down[expert_idx].copy_(down_w)
            del gate_w, up_w, down_w

        mlp.experts._parameters['gate_up_proj'] = _as_param(gate_up)
        mlp.experts._parameters['down_proj'] = _as_param(down)
        del gate_up, down

    _set_weight(model.embed_tokens, 'model.embed_tokens.weight')
    if 'lm_head.weight' in key_to_file:
        _set_weight(model.lm_head, 'lm_head.weight')
    elif config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    _set_weight(model.embed_tokens_per_layer, 'model.embed_tokens_per_layer.weight')
    _set_weight(model.per_layer_model_projection, 'model.per_layer_model_projection.weight')
    _set_norm(model.per_layer_projection_norm, 'model.per_layer_projection_norm.weight')
    _set_norm(model.norm, 'model.norm.weight')

    for i, layer in enumerate(model.layers):
        layer_type = config.layer_types[i] if config.layer_types else 'full_attention'
        if layer_type == 'linear_attention':
            raise NotImplementedError("FP16 streaming loader does not support linear-attention layers yet")

        hf_pre = f'model.layers.{i}'
        attn = layer.self_attn
        mlp = layer.mlp

        q_key = f'{hf_pre}.self_attn.q_proj.weight'
        k_key = f'{hf_pre}.self_attn.k_proj.weight'
        v_key = f'{hf_pre}.self_attn.v_proj.weight'
        if q_key in key_to_file and k_key in key_to_file and v_key in key_to_file:
            q_w = _get_cpu(q_key)
            k_w = _get_cpu(k_key)
            v_w = _get_cpu(v_key)
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            attn.qkv_proj.weight = _as_param(qkv_w)
            del q_w, k_w, v_w, qkv_w

        if config.attention_bias:
            qb_key = f'{hf_pre}.self_attn.q_proj.bias'
            kb_key = f'{hf_pre}.self_attn.k_proj.bias'
            vb_key = f'{hf_pre}.self_attn.v_proj.bias'
            if qb_key in key_to_file and kb_key in key_to_file and vb_key in key_to_file:
                q_b = _get_cpu(qb_key)
                k_b = _get_cpu(kb_key)
                v_b = _get_cpu(vb_key)
                qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
                attn.qkv_proj.bias = _as_param(qkv_b)
                del q_b, k_b, v_b, qkv_b

        _set_weight(attn.o_proj, f'{hf_pre}.self_attn.o_proj.weight')
        _set_bias(attn.o_proj, f'{hf_pre}.self_attn.o_proj.bias')

        if config.qk_norm:
            _set_norm(attn.q_norm, f'{hf_pre}.self_attn.q_norm.weight')
            _set_norm(attn.k_norm, f'{hf_pre}.self_attn.k_norm.weight')

        if config.is_moe_layer(i):
            _set_weight(mlp.gate, f'{hf_pre}.mlp.gate.weight')
            _set_qwen3_moe_experts(mlp, hf_pre)
        else:
            gate_key = f'{hf_pre}.mlp.gate_proj.weight'
            up_key = f'{hf_pre}.mlp.up_proj.weight'
            if gate_key in key_to_file and up_key in key_to_file:
                gate_w = _get_cpu(gate_key)
                up_w = _get_cpu(up_key)
                gate_up_w = torch.cat([gate_w, up_w], dim=0)
                mlp.gate_up_proj.weight = _as_param(gate_up_w)
                del gate_w, up_w, gate_up_w

            _set_weight(mlp.down_proj, f'{hf_pre}.mlp.down_proj.weight')
        _set_norm(layer.input_layernorm, f'{hf_pre}.input_layernorm.weight')
        _set_norm(layer.post_attention_layernorm, f'{hf_pre}.post_attention_layernorm.weight')

        gc.collect()
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 8 == 0 or i == config.num_hidden_layers - 1:
            print(f"   Layer {i + 1}/{config.num_hidden_layers} loaded")

    handles.clear()
    exit_stack.close()
    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    meta_params = [name for name, param in model.named_parameters() if param.device.type == 'meta']
    if meta_params:
        sample = ", ".join(meta_params[:8])
        raise RuntimeError(f"FP16 streaming load left meta parameters: {sample}")


def _is_awq_model(hf_config: dict) -> bool:
    """Check if this is an AWQ quantized model."""
    qc = hf_config.get('quantization_config', {})
    return qc.get('quant_method') == 'awq'


def _get_awq_config(hf_config: dict) -> dict:
    """Extract AWQ quantization config."""
    qc = hf_config.get('quantization_config', {})
    return {
        'bits': qc.get('bits', 4),
        'group_size': qc.get('group_size', 128),
        'zero_point': qc.get('zero_point', True),
    }


def _validate_supported_architecture(config: LlamaConfig, hf_config: dict) -> None:
    """
    Reject model families that this engine would mis-load today.

    Qwen 3.5 uses a hybrid of full and linear attention in its official text
    configs. Multimodal checkpoints are accepted only in text-only mode, where
    the language backbone is loaded and vision weights are ignored.
    """
    has_multimodal_config = (
        hf_config.get('vision_config') is not None
        or hf_config.get('audio_config') is not None
    )
    if has_multimodal_config and config.model_type not in {'qwen3_5_text', 'gemma4_text'}:
        raise NotImplementedError(
            "Multimodal checkpoints are supported only in text-only mode for known text backbones."
        )

    if config.model_type == 'gemma4_text':
        allowed_layers = {'full_attention', 'sliding_attention'}
    else:
        allowed_layers = {'full_attention', 'linear_attention'}

    if config.model_type == 'qwen3_moe' and (
        bool(hf_config.get('use_sliding_window', False))
        or hf_config.get('sliding_window') not in (None, 0)
    ):
        raise NotImplementedError(
            "Qwen3 MoE sliding-window attention is not supported yet; "
            "Qwen3-Coder-30B-A3B uses full attention and is supported."
        )

    if config.model_type == 'gemma4_text' and config.enable_moe_block:
        invalid_moe_fields = {
            'num_experts': config.num_experts,
            'top_k_experts': config.num_experts_per_tok,
            'moe_intermediate_size': config.moe_intermediate_size,
        }
        if any(int(value) <= 0 for value in invalid_moe_fields.values()):
            raise ValueError(
                "Gemma4 checkpoint enables MoE but its expert configuration is invalid: "
                f"{invalid_moe_fields}"
            )

    unsupported = sorted({layer for layer in config.layer_types if layer not in allowed_layers})
    if unsupported:
        raise NotImplementedError(
            f"Unsupported layer types {unsupported}. Supported types are {sorted(allowed_layers)}."
        )


def _refresh_rope_caches(model: MegaGemmLlama, config: LlamaConfig, device: str) -> None:
    """Recompute RoPE buffers after meta initialization."""
    from ..kernels.rope import precompute_freqs_cis

    cache_len = int(getattr(model, "_rope_cache_max_seq_len", config.max_position_embeddings))
    model.cos_cache, model.sin_cache = precompute_freqs_cis(
        config.rotary_dim, cache_len, config.rope_theta
    )
    if config.model_type == 'gemma4_text':
        model.layer_rope_caches = [
            _precompute_layer_rope_cache(config, layer_idx, max_seq_len=cache_len)
            for layer_idx in range(config.num_hidden_layers)
        ]
    model._move_rope_to_device(device)


def _cast_state_dict_preserve_aliases(
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> None:
    """Cast tensors while preserving tied-weight aliases within the state dict."""
    converted: Dict[int, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        alias_key = id(tensor)
        cast_tensor = converted.get(alias_key)
        if cast_tensor is None:
            cast_tensor = tensor.to(dtype)
            converted[alias_key] = cast_tensor
        state_dict[key] = cast_tensor


def _unique_tensor_mb(tensors) -> float:
    """Count unique backing storage once so tied weights are not double-counted."""
    seen = set()
    total = 0
    for tensor in tensors:
        if tensor.device.type == 'meta':
            continue
        try:
            storage = tensor.untyped_storage()
            storage_key = (tensor.device.type, tensor.device.index, storage.data_ptr())
            nbytes = storage.nbytes()
        except Exception:
            storage_key = (tensor.device.type, tensor.device.index, tensor.data_ptr())
            nbytes = tensor.nelement() * tensor.element_size()
        if storage_key in seen:
            continue
        seen.add(storage_key)
        total += nbytes
    return total / (1024**2)


def _map_weights(
    hf_weights: Dict[str, torch.Tensor],
    config: LlamaConfig,
) -> Dict[str, torch.Tensor]:
    """
    Map HuggingFace weight names to MegaGemmLlama weight names.
    Fuses gate_proj + up_proj into gate_up_proj.
    Handles bias mapping for Qwen 2.5 and QK-Norm for Qwen 3.

    Memory-efficient: deletes source tensors after mapping to avoid
    holding 2x copies in memory (critical for 8B+ models).
    """
    for key in list(hf_weights.keys()):
        normalized = _normalize_hf_weight_key(key)
        if normalized != key:
            hf_weights[normalized] = hf_weights.pop(key)

    mapped = {}

    def _pop(key):
        """Move tensor from hf_weights to avoid holding 2 copies."""
        return hf_weights.pop(key)

    # Embedding
    if 'model.embed_tokens.weight' in hf_weights:
        mapped['embed_tokens.weight'] = hf_weights['model.embed_tokens.weight']
    if 'model.embed_tokens_per_layer.weight' in hf_weights:
        mapped['embed_tokens_per_layer.weight'] = _pop('model.embed_tokens_per_layer.weight')
    if 'model.per_layer_model_projection.weight' in hf_weights:
        mapped['per_layer_model_projection.weight'] = _pop('model.per_layer_model_projection.weight')
    if 'model.per_layer_projection_norm.weight' in hf_weights:
        mapped['per_layer_projection_norm.weight'] = _pop('model.per_layer_projection_norm.weight')

    # Final norm
    if 'model.norm.weight' in hf_weights:
        mapped['norm.weight'] = _pop('model.norm.weight')

    # LM head (may be tied to embeddings — shares reference, don't pop embed)
    if 'lm_head.weight' in hf_weights:
        mapped['lm_head.weight'] = _pop('lm_head.weight')
    elif config.tie_word_embeddings:
        mapped['lm_head.weight'] = hf_weights['model.embed_tokens.weight']  # shared ref

    # Now pop embed_tokens (after lm_head tie check)
    if 'model.embed_tokens.weight' in hf_weights:
        _pop('model.embed_tokens.weight')

    # Layers
    for i in range(config.num_hidden_layers):
        hf_pre = f'model.layers.{i}'
        mg_pre = f'layers.{i}'

        layer_type = config.layer_types[i] if config.layer_types else 'full_attention'
        if layer_type == 'linear_attention':
            lin_pre = f'{hf_pre}.linear_attn'
            for suffix in [
                'in_proj_qkv.weight',
                'out_proj.weight',
                'dt_bias',
                'A_log',
                'conv1d.weight',
                'norm.weight',
            ]:
                src = f'{lin_pre}.{suffix}'
                if src in hf_weights:
                    mapped[f'{mg_pre}.linear_attn.{suffix}'] = _pop(src)
            z_key = f'{lin_pre}.in_proj_z.weight'
            b_key = f'{lin_pre}.in_proj_b.weight'
            a_key = f'{lin_pre}.in_proj_a.weight'
            if z_key in hf_weights and b_key in hf_weights and a_key in hf_weights:
                mapped[f'{mg_pre}.linear_attn.in_proj_baz.weight'] = torch.cat(
                    [_pop(z_key), _pop(b_key), _pop(a_key)], dim=0,
                )
        elif config.model_type == 'gemma4_text':
            # Gemma 4 keeps Q/K/V separate: KV-shared layers intentionally omit
            # K/V projections, and full/sliding layers can use different head dims.
            is_kv_shared = (
                bool(config.kv_share_sources)
                and i < len(config.kv_share_sources)
                and config.kv_share_sources[i] is not None
            )
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if is_kv_shared and proj_name in ('k_proj', 'v_proj'):
                    continue
                w_key = f'{hf_pre}.self_attn.{proj_name}.weight'
                b_key = f'{hf_pre}.self_attn.{proj_name}.bias'
                if w_key in hf_weights:
                    mapped[f'{mg_pre}.self_attn.{proj_name}.weight'] = _pop(w_key)
                if b_key in hf_weights:
                    mapped[f'{mg_pre}.self_attn.{proj_name}.bias'] = _pop(b_key)

            for norm_name in ['q_norm', 'k_norm']:
                if is_kv_shared and norm_name == 'k_norm':
                    continue
                w_key = f'{hf_pre}.self_attn.{norm_name}.weight'
                if w_key in hf_weights:
                    mapped[f'{mg_pre}.self_attn.{norm_name}.weight'] = _pop(w_key)
        else:
            # Attention: fuse q_proj + k_proj + v_proj → qkv_proj (FP16, like gate+up fusion)
            q_key = f'{hf_pre}.self_attn.q_proj.weight'
            k_key = f'{hf_pre}.self_attn.k_proj.weight'
            v_key = f'{hf_pre}.self_attn.v_proj.weight'
            o_key = f'{hf_pre}.self_attn.o_proj.weight'

            if q_key in hf_weights and k_key in hf_weights and v_key in hf_weights:
                q_w = _pop(q_key)
                k_w = _pop(k_key)
                v_w = _pop(v_key)
                mapped[f'{mg_pre}.self_attn.qkv_proj.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
                del q_w, k_w, v_w  # Free originals after cat

            if o_key in hf_weights:
                mapped[f'{mg_pre}.self_attn.o_proj.weight'] = _pop(o_key)

            # Attention projection biases (Qwen 2.5) — also fused
            if config.attention_bias:
                qb_key = f'{hf_pre}.self_attn.q_proj.bias'
                kb_key = f'{hf_pre}.self_attn.k_proj.bias'
                vb_key = f'{hf_pre}.self_attn.v_proj.bias'
                if qb_key in hf_weights and kb_key in hf_weights and vb_key in hf_weights:
                    q_b = _pop(qb_key)
                    k_b = _pop(kb_key)
                    v_b = _pop(vb_key)
                    mapped[f'{mg_pre}.self_attn.qkv_proj.bias'] = torch.cat([q_b, k_b, v_b], dim=0)
                    del q_b, k_b, v_b

            # QK-Norm weights (Qwen 3 / Qwen 3.5 full-attention layers)
            if config.qk_norm:
                for norm_name in ['q_norm', 'k_norm']:
                    w_key = f'{hf_pre}.self_attn.{norm_name}.weight'
                    if w_key in hf_weights:
                        mapped[f'{mg_pre}.self_attn.{norm_name}.weight'] = _pop(w_key)

        if config.is_moe_layer(i):
            if config.model_type == 'gemma4_text':
                gate_key = f'{hf_pre}.mlp.gate_proj.weight'
                up_key = f'{hf_pre}.mlp.up_proj.weight'
                if gate_key in hf_weights and up_key in hf_weights:
                    gate_w = _pop(gate_key)
                    up_w = _pop(up_key)
                    mapped[f'{mg_pre}.mlp.shared_mlp.gate_up_proj.weight'] = torch.cat(
                        [gate_w, up_w],
                        dim=0,
                    )
                    del gate_w, up_w
                down_key = f'{hf_pre}.mlp.down_proj.weight'
                if down_key in hf_weights:
                    mapped[f'{mg_pre}.mlp.shared_mlp.down_proj.weight'] = _pop(down_key)

                for src_name, dst_name in (
                    ('router.proj.weight', 'mlp.gate.proj.weight'),
                    ('router.scale', 'mlp.gate.scale'),
                    ('router.per_expert_scale', 'mlp.gate.per_expert_scale'),
                    ('experts.gate_up_proj', 'mlp.experts.gate_up_proj'),
                    ('experts.down_proj', 'mlp.experts.down_proj'),
                ):
                    src = f'{hf_pre}.{src_name}'
                    if src in hf_weights:
                        mapped[f'{mg_pre}.{dst_name}'] = _pop(src)
            else:
                gate_key = f'{hf_pre}.mlp.gate.weight'
                if gate_key in hf_weights:
                    mapped[f'{mg_pre}.mlp.gate.weight'] = _pop(gate_key)

                stacked_gate_up = f'{hf_pre}.mlp.experts.gate_up_proj'
                stacked_down = f'{hf_pre}.mlp.experts.down_proj'
                if stacked_gate_up in hf_weights and stacked_down in hf_weights:
                    mapped[f'{mg_pre}.mlp.experts.gate_up_proj'] = _pop(stacked_gate_up)
                    mapped[f'{mg_pre}.mlp.experts.down_proj'] = _pop(stacked_down)
                else:
                    # Released Qwen3 MoE checkpoints store one dense MLP per expert:
                    #   experts.N.gate_proj.weight / up_proj.weight / down_proj.weight
                    # The runtime stores the same data in HF's kernel-friendly 3D form.
                    num_experts = int(config.num_experts)
                    hidden_size = int(config.hidden_size)
                    moe_intermediate = int(config.moe_intermediate_size)
                    required = []
                    for expert_idx in range(num_experts):
                        expert_pre = f'{hf_pre}.mlp.experts.{expert_idx}'
                        required.extend(
                            [
                                f'{expert_pre}.gate_proj.weight',
                                f'{expert_pre}.up_proj.weight',
                                f'{expert_pre}.down_proj.weight',
                            ]
                        )

                    if all(key in hf_weights for key in required):
                        gate_up = torch.empty(
                            (num_experts, 2 * moe_intermediate, hidden_size),
                            dtype=hf_weights[required[0]].dtype,
                            device=hf_weights[required[0]].device,
                        )
                        down = torch.empty(
                            (num_experts, hidden_size, moe_intermediate),
                            dtype=hf_weights[required[2]].dtype,
                            device=hf_weights[required[2]].device,
                        )
                        for expert_idx in range(num_experts):
                            expert_pre = f'{hf_pre}.mlp.experts.{expert_idx}'
                            gate_w = _pop(f'{expert_pre}.gate_proj.weight')
                            up_w = _pop(f'{expert_pre}.up_proj.weight')
                            down_w = _pop(f'{expert_pre}.down_proj.weight')
                            gate_up[expert_idx, :moe_intermediate].copy_(gate_w)
                            gate_up[expert_idx, moe_intermediate:].copy_(up_w)
                            down[expert_idx].copy_(down_w)
                            del gate_w, up_w, down_w
                        mapped[f'{mg_pre}.mlp.experts.gate_up_proj'] = gate_up.contiguous()
                        mapped[f'{mg_pre}.mlp.experts.down_proj'] = down.contiguous()
        else:
            # MLP: fuse gate_proj + up_proj -> gate_up_proj
            gate_key = f'{hf_pre}.mlp.gate_proj.weight'
            up_key = f'{hf_pre}.mlp.up_proj.weight'
            if gate_key in hf_weights and up_key in hf_weights:
                gate_w = _pop(gate_key)
                up_w = _pop(up_key)
                mapped[f'{mg_pre}.mlp.gate_up_proj.weight'] = torch.cat([gate_w, up_w], dim=0)
                del gate_w, up_w  # Free originals after cat

            # down_proj
            down_key = f'{hf_pre}.mlp.down_proj.weight'
            if down_key in hf_weights:
                mapped[f'{mg_pre}.mlp.down_proj.weight'] = _pop(down_key)

        # Layer norms
        mapped[f'{mg_pre}.input_layernorm.weight'] = _pop(f'{hf_pre}.input_layernorm.weight')
        mapped[f'{mg_pre}.post_attention_layernorm.weight'] = _pop(f'{hf_pre}.post_attention_layernorm.weight')
        if config.model_type == 'gemma4_text':
            if config.is_moe_layer(i):
                norm_fallbacks = [
                    ('pre_feedforward_layernorm_2', 'pre_feedforward_layernorm'),
                    ('post_feedforward_layernorm_1', 'post_feedforward_layernorm'),
                    ('post_feedforward_layernorm_2', 'post_feedforward_layernorm'),
                ]
                for local_name, fallback_name in norm_fallbacks:
                    primary_key = f'{hf_pre}.{local_name}.weight'
                    fallback_key = f'{hf_pre}.{fallback_name}.weight'
                    if primary_key in hf_weights:
                        mapped[f'{mg_pre}.{local_name}.weight'] = _pop(primary_key)
                    elif fallback_key in hf_weights:
                        mapped[f'{mg_pre}.{local_name}.weight'] = hf_weights[fallback_key].clone()
            for norm_name in [
                'pre_feedforward_layernorm',
                'post_feedforward_layernorm',
            ]:
                w_key = f'{hf_pre}.{norm_name}.weight'
                if w_key in hf_weights:
                    mapped[f'{mg_pre}.{norm_name}.weight'] = _pop(w_key)
            for ple_name in ['per_layer_input_gate', 'per_layer_projection']:
                w_key = f'{hf_pre}.{ple_name}.weight'
                if w_key in hf_weights:
                    mapped[f'{mg_pre}.{ple_name}.weight'] = _pop(w_key)
            w_key = f'{hf_pre}.post_per_layer_input_norm.weight'
            if w_key in hf_weights:
                mapped[f'{mg_pre}.post_per_layer_input_norm.weight'] = _pop(w_key)
            scalar_key = f'{hf_pre}.layer_scalar'
            if scalar_key in hf_weights:
                mapped[f'{mg_pre}.layer_scalar'] = _pop(scalar_key)

    return mapped


def _replace_with_quantized(model: MegaGemmLlama, hf_weights: Dict[str, torch.Tensor],
                              config: LlamaConfig, group_size: int = 128):
    """
    Replace nn.Linear modules with QuantizedLinear where AWQ weights exist.
    Loads qweight/qzeros/scales into the quantized modules.
    """
    from ..quantization.w4a16 import QuantizedLinear

    replaced = 0
    quant_mb = 0.0
    fp16_mb = 0.0

    def _tensor_mb(tensor: torch.Tensor) -> float:
        return float(tensor.numel() * tensor.element_size()) / (1024.0 * 1024.0)

    def _set_qwen3_moe_awq_experts(mlp, hf_pre: str) -> tuple[int, float, float]:
        num_experts = int(config.num_experts)
        hidden_size = int(config.hidden_size)
        moe_intermediate = int(config.moe_intermediate_size)
        if num_experts <= 0 or hidden_size <= 0 or moe_intermediate <= 0:
            raise RuntimeError(f"Invalid Qwen3 MoE AWQ expert config at {hf_pre}")

        required = []
        for expert_idx in range(num_experts):
            expert_pre = f"{hf_pre}.mlp.experts.{expert_idx}"
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                required.extend(
                    [
                        f"{expert_pre}.{proj_name}.qweight",
                        f"{expert_pre}.{proj_name}.scales",
                        f"{expert_pre}.{proj_name}.qzeros",
                    ]
                )
        missing = [key for key in required if key not in hf_weights]
        if missing:
            sample = ", ".join(missing[:6])
            raise RuntimeError(
                f"Qwen3 MoE AWQ expert weights for {hf_pre} are incomplete. Missing: {sample}"
            )

        gate0 = hf_weights[f"{hf_pre}.mlp.experts.0.gate_proj.qweight"]
        gate_sc0 = hf_weights[f"{hf_pre}.mlp.experts.0.gate_proj.scales"]
        gate_qz0 = hf_weights[f"{hf_pre}.mlp.experts.0.gate_proj.qzeros"]
        down0 = hf_weights[f"{hf_pre}.mlp.experts.0.down_proj.qweight"]
        down_sc0 = hf_weights[f"{hf_pre}.mlp.experts.0.down_proj.scales"]
        down_qz0 = hf_weights[f"{hf_pre}.mlp.experts.0.down_proj.qzeros"]
        gate_up_qweight = torch.empty(
            (num_experts, int(gate0.shape[0]), int(gate0.shape[1]) * 2),
            dtype=gate0.dtype,
            device="cpu",
        )
        gate_up_scales = torch.empty(
            (num_experts, int(gate_sc0.shape[0]), int(gate_sc0.shape[1]) * 2),
            dtype=torch.float16,
            device="cpu",
        )
        gate_up_qzeros = torch.empty(
            (num_experts, int(gate_qz0.shape[0]), int(gate_qz0.shape[1]) * 2),
            dtype=gate_qz0.dtype,
            device="cpu",
        )
        down_qweight = torch.empty(
            (num_experts, int(down0.shape[0]), int(down0.shape[1])),
            dtype=down0.dtype,
            device="cpu",
        )
        down_scales = torch.empty(
            (num_experts, int(down_sc0.shape[0]), int(down_sc0.shape[1])),
            dtype=torch.float16,
            device="cpu",
        )
        down_qzeros = torch.empty(
            (num_experts, int(down_qz0.shape[0]), int(down_qz0.shape[1])),
            dtype=down_qz0.dtype,
            device="cpu",
        )

        for expert_idx in range(num_experts):
            expert_pre = f"{hf_pre}.mlp.experts.{expert_idx}"
            gate_qw = hf_weights[f"{expert_pre}.gate_proj.qweight"]
            up_qw = hf_weights[f"{expert_pre}.up_proj.qweight"]
            gate_sc = hf_weights[f"{expert_pre}.gate_proj.scales"]
            up_sc = hf_weights[f"{expert_pre}.up_proj.scales"]
            gate_qz = hf_weights[f"{expert_pre}.gate_proj.qzeros"]
            up_qz = hf_weights[f"{expert_pre}.up_proj.qzeros"]
            down_qw = hf_weights[f"{expert_pre}.down_proj.qweight"]
            down_sc = hf_weights[f"{expert_pre}.down_proj.scales"]
            down_qz = hf_weights[f"{expert_pre}.down_proj.qzeros"]

            gate_up_qweight[expert_idx].copy_(torch.cat([gate_qw, up_qw], dim=1))
            gate_up_scales[expert_idx].copy_(torch.cat([gate_sc, up_sc], dim=1).to(torch.float16))
            gate_up_qzeros[expert_idx].copy_(torch.cat([gate_qz, up_qz], dim=1))
            down_qweight[expert_idx].copy_(down_qw)
            down_scales[expert_idx].copy_(down_sc.to(torch.float16))
            down_qzeros[expert_idx].copy_(down_qz)

        mlp.experts._parameters.pop("gate_up_proj", None)
        mlp.experts._parameters.pop("down_proj", None)
        mlp.experts._buffers["gate_up_qweight"] = gate_up_qweight.contiguous()
        mlp.experts._buffers["gate_up_scales"] = gate_up_scales.contiguous()
        mlp.experts._buffers["gate_up_qzeros"] = gate_up_qzeros.contiguous()
        mlp.experts._buffers["down_qweight"] = down_qweight.contiguous()
        mlp.experts._buffers["down_scales"] = down_scales.contiguous()
        mlp.experts._buffers["down_qzeros"] = down_qzeros.contiguous()
        mlp.experts.awq_group_size = int(group_size)

        q_mb = (
            _tensor_mb(gate_up_qweight)
            + _tensor_mb(gate_up_scales)
            + _tensor_mb(gate_up_qzeros)
            + _tensor_mb(down_qweight)
            + _tensor_mb(down_scales)
            + _tensor_mb(down_qzeros)
        )
        fp16_mb = float(num_experts * (3 * hidden_size * moe_intermediate) * 2) / (1024.0 * 1024.0)
        return 3, q_mb, fp16_mb

    for i in range(config.num_hidden_layers):
        hf_pre = f'model.layers.{i}'
        layer = model.layers[i]

        # Attention projections
        attn_has_awq = False
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            qw_key = f'{hf_pre}.self_attn.{proj_name}.qweight'
            if qw_key not in hf_weights:
                continue

            # For Q/K/V: create new QuantizedLinear (qkv_proj doesn't have these)
            # For O: replace existing o_proj
            if proj_name == 'o_proj':
                old_linear = layer.self_attn.o_proj
            else:
                # Compute dims from qweight shape
                qw = hf_weights[qw_key]
                in_f = qw.shape[0]
                out_f = qw.shape[1] * 8
                old_linear = type('_dummy', (), {'in_features': in_f, 'out_features': out_f, 'bias': None})()
                # Check for bias
                b_key = f'{hf_pre}.self_attn.{proj_name}.bias'
                if b_key in hf_weights:
                    old_linear.bias = True

            has_bias = old_linear.bias is not None

            ql = QuantizedLinear(
                old_linear.in_features, old_linear.out_features,
                group_size=group_size, bias=has_bias,
            )

            # Load quantized weights
            ql.qweight.copy_(hf_weights[qw_key])
            ql.scales.copy_(hf_weights[f'{hf_pre}.self_attn.{proj_name}.scales'])
            ql.qzeros.copy_(hf_weights[f'{hf_pre}.self_attn.{proj_name}.qzeros'])

            if has_bias:
                b_key = f'{hf_pre}.self_attn.{proj_name}.bias'
                if b_key in hf_weights:
                    ql.bias.data.copy_(hf_weights[b_key])

            setattr(layer.self_attn, proj_name, ql)
            if proj_name in ('q_proj', 'k_proj', 'v_proj'):
                attn_has_awq = True
            quant_mb += ql.weight_memory_mb
            fp16_mb += ql.fp16_equivalent_mb
            replaced += 1

        # Fuse AWQ Q+K+V into qkv_proj (same idea as gate+up fusion below)
        # so flat decode can use a single fused matmul instead of 3 separate ones.
        q_qw_key = f'{hf_pre}.self_attn.q_proj.qweight'
        k_qw_key = f'{hf_pre}.self_attn.k_proj.qweight'
        v_qw_key = f'{hf_pre}.self_attn.v_proj.qweight'

        if attn_has_awq and q_qw_key in hf_weights and k_qw_key in hf_weights and v_qw_key in hf_weights:
            q_qw = hf_weights[q_qw_key]
            k_qw = hf_weights[k_qw_key]
            v_qw = hf_weights[v_qw_key]
            q_sc = hf_weights[f'{hf_pre}.self_attn.q_proj.scales']
            k_sc = hf_weights[f'{hf_pre}.self_attn.k_proj.scales']
            v_sc = hf_weights[f'{hf_pre}.self_attn.v_proj.scales']
            q_qz = hf_weights[f'{hf_pre}.self_attn.q_proj.qzeros']
            k_qz = hf_weights[f'{hf_pre}.self_attn.k_proj.qzeros']
            v_qz = hf_weights[f'{hf_pre}.self_attn.v_proj.qzeros']

            fused_qw = torch.cat([q_qw, k_qw, v_qw], dim=1)
            fused_sc = torch.cat([q_sc, k_sc, v_sc], dim=1)
            fused_qz = torch.cat([q_qz, k_qz, v_qz], dim=1)

            in_f = q_qw.shape[0]
            q_out = q_qw.shape[1] * 8
            k_out = k_qw.shape[1] * 8
            v_out = v_qw.shape[1] * 8
            total_out = q_out + k_out + v_out

            # Check for bias (Qwen models have QKV bias)
            q_bias_key = f'{hf_pre}.self_attn.q_proj.bias'
            has_qkv_bias = q_bias_key in hf_weights

            fused_ql = QuantizedLinear(in_f, total_out, group_size=group_size, bias=has_qkv_bias)
            fused_ql.qweight.copy_(fused_qw)
            fused_ql.scales.copy_(fused_sc)
            fused_ql.qzeros.copy_(fused_qz)

            if has_qkv_bias:
                q_b = hf_weights[q_bias_key]
                k_b = hf_weights[f'{hf_pre}.self_attn.k_proj.bias']
                v_b = hf_weights[f'{hf_pre}.self_attn.v_proj.bias']
                fused_ql.bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))

            layer.self_attn.qkv_proj = fused_ql
            layer.self_attn._awq_separate = False  # Use fused path!

            # Clean up separate projections to free VRAM
            if hasattr(layer.self_attn, 'q_proj'):
                del layer.self_attn.q_proj
            if hasattr(layer.self_attn, 'k_proj'):
                del layer.self_attn.k_proj
            if hasattr(layer.self_attn, 'v_proj'):
                del layer.self_attn.v_proj

            del fused_qw, fused_sc, fused_qz
        elif attn_has_awq:
            # Could not fuse — keep separate
            layer.self_attn._awq_separate = True
            if hasattr(layer.self_attn, 'qkv_proj'):
                del layer.self_attn.qkv_proj

        # MLP projections — Fuse AWQ gate_proj + up_proj into gate_up_proj
        # by concatenating quantized weights along output dimension.
        # This lets us reuse the existing fused MLP code path (_awq_separate=False).
        if config.is_moe_layer(i):
            moe_replaced, moe_quant_mb, moe_fp16_mb = _set_qwen3_moe_awq_experts(
                layer.mlp,
                hf_pre,
            )
            quant_mb += moe_quant_mb
            fp16_mb += moe_fp16_mb
            replaced += moe_replaced
            continue

        gate_qw_key = f'{hf_pre}.mlp.gate_proj.qweight'
        up_qw_key = f'{hf_pre}.mlp.up_proj.qweight'
        down_qw_key = f'{hf_pre}.mlp.down_proj.qweight'

        if gate_qw_key in hf_weights and up_qw_key in hf_weights:
            # Fuse gate + up: concatenate along output dim (dim=1 for qweight/qzeros, dim=1 for scales)
            gate_qw = hf_weights[gate_qw_key]
            up_qw = hf_weights[up_qw_key]
            gate_sc = hf_weights[f'{hf_pre}.mlp.gate_proj.scales']
            up_sc = hf_weights[f'{hf_pre}.mlp.up_proj.scales']
            gate_qz = hf_weights[f'{hf_pre}.mlp.gate_proj.qzeros']
            up_qz = hf_weights[f'{hf_pre}.mlp.up_proj.qzeros']

            # qweight: [in_f, out_f//8] → cat along dim=1 → [in_f, (2*out_f)//8]
            fused_qw = torch.cat([gate_qw, up_qw], dim=1)
            fused_sc = torch.cat([gate_sc, up_sc], dim=1)
            fused_qz = torch.cat([gate_qz, up_qz], dim=1)

            in_f = gate_qw.shape[0]
            out_f = gate_qw.shape[1] * 8  # per-proj output features

            # Create fused QuantizedLinear with 2x output
            fused_ql = QuantizedLinear(in_f, 2 * out_f, group_size=group_size, bias=False)
            fused_ql.qweight.copy_(fused_qw)
            fused_ql.scales.copy_(fused_sc)
            fused_ql.qzeros.copy_(fused_qz)

            # Replace the FP16 gate_up_proj with quantized fused version
            layer.mlp.gate_up_proj = fused_ql
            layer.mlp._awq_separate = False  # Use fused path in forward()

            quant_mb += fused_ql.weight_memory_mb
            fp16_mb += fused_ql.fp16_equivalent_mb
            replaced += 2  # Counts as 2 replaced layers

            del fused_qw, fused_sc, fused_qz

        # down_proj — always separate
        if down_qw_key in hf_weights:
            qw = hf_weights[down_qw_key]
            sc = hf_weights[f'{hf_pre}.mlp.down_proj.scales']
            qz = hf_weights[f'{hf_pre}.mlp.down_proj.qzeros']
            in_f = qw.shape[0]
            out_f = qw.shape[1] * 8

            ql = QuantizedLinear(in_f, out_f, group_size=group_size, bias=False)
            ql.qweight.copy_(qw)
            ql.scales.copy_(sc)
            ql.qzeros.copy_(qz)

            layer.mlp.down_proj = ql
            quant_mb += ql.weight_memory_mb
            fp16_mb += ql.fp16_equivalent_mb
            replaced += 1

    return replaced, quant_mb, fp16_mb


def _load_int8_streaming(
    model: MegaGemmLlama,
    config: LlamaConfig,
    model_path: str,
    dtype: torch.dtype,
    device: str,
) -> tuple:
    """
    Stream-load weights layer-by-layer with on-the-fly INT8 quantization.

    Uses safetensors lazy loading (safe_open) to read one tensor at a time,
    fuse QKV/gate_up projections, quantize to INT8, and move to GPU.

    Peak CPU RAM: ~2.4GB  (1 fused layer FP16 during quantization)
    Peak GPU VRAM: ~9.5GB (embed FP16 + all layers INT8)

    vs old approach:
    Peak CPU RAM: ~24GB   (all weights FP16 + INT8 copies + model)

    This makes 8B+ models loadable on machines with 12-16GB CPU RAM
    (e.g., Google Colab free tier with T4).
    """
    from safetensors import safe_open
    from ..quantization.w8a16 import Int8Linear, quantize_to_int8
    from ..kernels.rope import precompute_freqs_cis
    import gc

    skip_ops = _parse_int8_skip_ops()
    use_int8_qkv = "qkv" not in skip_ops
    use_int8_o = "o_proj" not in skip_ops
    use_int8_gate_up = "gate_up" not in skip_ops
    use_int8_down = "down" not in skip_ops

    # ── Build key→file mapping (lazy, zero memory) ──
    st_files = sorted(
        os.path.join(model_path, f) for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    )

    key_to_file = {}
    for fpath in st_files:
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                key_to_file[key] = fpath

    # Cache open file handles to avoid re-opening for every tensor
    _handle_cache = {}

    def _get(key, target_device='cpu'):
        """Load single tensor lazily from safetensors."""
        fpath = key_to_file[key]
        cache_key = (fpath, target_device)
        if cache_key not in _handle_cache:
            _handle_cache[cache_key] = safe_open(fpath, framework="pt", device=target_device)
        return _handle_cache[cache_key].get_tensor(key)

    def _tensor_mb(tensor: torch.Tensor) -> float:
        return float(tensor.numel() * tensor.element_size()) / (1024.0 * 1024.0)

    def _set_qwen3_moe_experts_int8(mlp, hf: str) -> tuple[int, float, float]:
        """Quantize Qwen3 MoE experts into stacked W8A16 buffers.

        The dense INT8 loader handles ordinary MLPs with Int8Linear modules. Qwen3
        MoE stores 128 routed experts per layer, so decode needs expert-indexed
        [E, out, in] INT8 tensors plus per-output scales.
        """
        if not (use_int8_gate_up and use_int8_down):
            raise NotImplementedError(
                "Qwen3 MoE INT8 streaming requires quantizing both expert gate/up "
                "and down weights. Remove gate_up/down from MEGAGEMM_INT8_SKIP_OPS."
            )

        num_experts = int(config.num_experts)
        hidden_size = int(config.hidden_size)
        moe_intermediate = int(config.moe_intermediate_size)
        if num_experts <= 0 or hidden_size <= 0 or moe_intermediate <= 0:
            raise RuntimeError(f"Invalid Qwen3 MoE expert config at {hf}")

        required = []
        for expert_idx in range(num_experts):
            expert_pre = f"{hf}.mlp.experts.{expert_idx}"
            required.extend(
                [
                    f"{expert_pre}.gate_proj.weight",
                    f"{expert_pre}.up_proj.weight",
                    f"{expert_pre}.down_proj.weight",
                ]
            )
        missing = [key for key in required if key not in key_to_file]
        if missing:
            sample = ", ".join(missing[:6])
            raise RuntimeError(
                f"Qwen3 MoE INT8 expert weights for {hf} are incomplete. Missing: {sample}"
            )

        gate_up_int8 = torch.empty(
            (num_experts, 2 * moe_intermediate, hidden_size),
            dtype=torch.int8,
            device="cpu",
        )
        gate_up_scale = torch.empty(
            (num_experts, 2 * moe_intermediate),
            dtype=torch.float16,
            device="cpu",
        )
        down_int8 = torch.empty(
            (num_experts, hidden_size, moe_intermediate),
            dtype=torch.int8,
            device="cpu",
        )
        down_scale = torch.empty(
            (num_experts, hidden_size),
            dtype=torch.float16,
            device="cpu",
        )

        fp16_mb = 0.0
        for expert_idx in range(num_experts):
            expert_pre = f"{hf}.mlp.experts.{expert_idx}"
            gate_w = _get(f"{expert_pre}.gate_proj.weight").to(dtype).contiguous()
            up_w = _get(f"{expert_pre}.up_proj.weight").to(dtype).contiguous()
            down_w = _get(f"{expert_pre}.down_proj.weight").to(dtype).contiguous()
            fp16_mb += _tensor_mb(gate_w) + _tensor_mb(up_w) + _tensor_mb(down_w)

            gate_up_w = torch.cat([gate_w, up_w], dim=0).contiguous()
            gu_i8, gu_scale = quantize_to_int8(gate_up_w)
            d_i8, d_scale = quantize_to_int8(down_w)
            gate_up_int8[expert_idx].copy_(gu_i8)
            gate_up_scale[expert_idx].copy_(gu_scale)
            down_int8[expert_idx].copy_(d_i8)
            down_scale[expert_idx].copy_(d_scale)
            del gate_w, up_w, down_w, gate_up_w, gu_i8, gu_scale, d_i8, d_scale

        # Remove the meta FP16 expert parameters; the Qwen3MoeExperts forward path
        # switches to these buffers when all four are present.
        mlp.experts._parameters.pop("gate_up_proj", None)
        mlp.experts._parameters.pop("down_proj", None)
        mlp.experts._buffers["gate_up_int8"] = gate_up_int8.to(device).contiguous()
        mlp.experts._buffers["gate_up_scale"] = gate_up_scale.to(device).contiguous()
        mlp.experts._buffers["down_int8"] = down_int8.to(device).contiguous()
        mlp.experts._buffers["down_scale"] = down_scale.to(device).contiguous()

        q_mb = (
            _tensor_mb(gate_up_int8)
            + _tensor_mb(gate_up_scale)
            + _tensor_mb(down_int8)
            + _tensor_mb(down_scale)
        )
        del gate_up_int8, gate_up_scale, down_int8, down_scale
        return 2, q_mb, fp16_mb

    replaced = 0
    q_mb = 0.0
    fp16_mb = 0.0

    # ── 1. Non-layer weights (embed, norm, lm_head) → directly to GPU ──
    print(f"   Loading embeddings + norms...")

    embed_w = _get('model.embed_tokens.weight', device).to(dtype)
    model.embed_tokens.weight = torch.nn.Parameter(embed_w)

    norm_w = _get('model.norm.weight', device).to(dtype)
    model.norm.weight = torch.nn.Parameter(norm_w)

    if 'lm_head.weight' in key_to_file:
        lm_w = _get('lm_head.weight', device).to(dtype)
        model.lm_head.weight = torch.nn.Parameter(lm_w)
    elif config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight  # Shared

    # ── 2. RoPE caches (recompute — meta device ones are invalid) ──
    model.cos_cache, model.sin_cache = precompute_freqs_cis(
        config.rotary_dim, config.max_position_embeddings, config.rope_theta
    )
    model._move_rope_to_device(device)

    # ── 3. Layer-by-layer: load FP16 → fuse → INT8 → GPU ──
    for i in range(config.num_hidden_layers):
        hf = f'model.layers.{i}'
        layer = model.layers[i]
        if i == 0 and skip_ops:
            print(
                "   INT8 selective mode: keeping FP16 for "
                + ", ".join(sorted(skip_ops))
            )

        # -- Norms (not quantized, stay FP16, small) --
        in_norm_w = _get(f'{hf}.input_layernorm.weight', device).to(dtype)
        post_norm_w = _get(f'{hf}.post_attention_layernorm.weight', device).to(dtype)
        layer.input_layernorm.weight = torch.nn.Parameter(in_norm_w)
        layer.post_attention_layernorm.weight = torch.nn.Parameter(post_norm_w)

        # -- Attention: fuse Q+K+V -> QKV (INT8 or FP16 selective mode) --
        q_w = _get(f'{hf}.self_attn.q_proj.weight').to(dtype)
        k_w = _get(f'{hf}.self_attn.k_proj.weight').to(dtype)
        v_w = _get(f'{hf}.self_attn.v_proj.weight').to(dtype)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        del q_w, k_w, v_w

        has_bias = config.attention_bias
        if use_int8_qkv:
            qkv_int8, qkv_scale = quantize_to_int8(qkv_w)
            del qkv_w

            qkv_linear = Int8Linear(qkv_int8.shape[1], qkv_int8.shape[0], bias=has_bias)
            qkv_linear.register_buffer('weight_int8', qkv_int8)
            qkv_linear.register_buffer('scale', qkv_scale)
            del qkv_int8, qkv_scale

            # QKV bias (Qwen 2.5)
            if has_bias:
                qb_key = f'{hf}.self_attn.q_proj.bias'
                kb_key = f'{hf}.self_attn.k_proj.bias'
                vb_key = f'{hf}.self_attn.v_proj.bias'
                if qb_key in key_to_file:
                    q_b = _get(qb_key).to(dtype)
                    k_b = _get(kb_key).to(dtype)
                    v_b = _get(vb_key).to(dtype)
                    qkv_linear.bias = torch.nn.Parameter(
                        torch.cat([q_b, k_b, v_b], dim=0).to(device)
                    )
                    del q_b, k_b, v_b

            layer.self_attn.qkv_proj = qkv_linear.to(device)
            replaced += 1
            q_mb += qkv_linear.weight_memory_mb
            fp16_mb += qkv_linear.fp16_equivalent_mb
        else:
            layer.self_attn.qkv_proj.weight = torch.nn.Parameter(qkv_w.to(device))
            del qkv_w
            if has_bias:
                qb_key = f'{hf}.self_attn.q_proj.bias'
                kb_key = f'{hf}.self_attn.k_proj.bias'
                vb_key = f'{hf}.self_attn.v_proj.bias'
                if qb_key in key_to_file:
                    q_b = _get(qb_key).to(dtype)
                    k_b = _get(kb_key).to(dtype)
                    v_b = _get(vb_key).to(dtype)
                    layer.self_attn.qkv_proj.bias = torch.nn.Parameter(
                        torch.cat([q_b, k_b, v_b], dim=0).to(device)
                    )
                    del q_b, k_b, v_b

        # -- O projection (INT8 or FP16 selective mode) --
        o_w = _get(f'{hf}.self_attn.o_proj.weight').to(dtype)
        if use_int8_o:
            o_int8, o_scale = quantize_to_int8(o_w)
            del o_w

            o_linear = Int8Linear(o_int8.shape[1], o_int8.shape[0])
            o_linear.register_buffer('weight_int8', o_int8)
            o_linear.register_buffer('scale', o_scale)
            del o_int8, o_scale

            layer.self_attn.o_proj = o_linear.to(device)
            replaced += 1
            q_mb += o_linear.weight_memory_mb
            fp16_mb += o_linear.fp16_equivalent_mb
        else:
            layer.self_attn.o_proj.weight = torch.nn.Parameter(o_w.to(device))
            del o_w

        # -- QK-Norm weights (Qwen 3) --
        if config.qk_norm:
            for norm_name in ['q_norm', 'k_norm']:
                w_key = f'{hf}.self_attn.{norm_name}.weight'
                if w_key in key_to_file:
                    nw = _get(w_key, device).to(dtype)
                    getattr(layer.self_attn, norm_name).weight = torch.nn.Parameter(nw)

        if config.is_moe_layer(i):
            gate_key = f"{hf}.mlp.gate.weight"
            if gate_key in key_to_file:
                layer.mlp.gate.weight = torch.nn.Parameter(
                    _get(gate_key, device).to(dtype).contiguous()
                )
            moe_replaced, moe_q_mb, moe_fp16_mb = _set_qwen3_moe_experts_int8(layer.mlp, hf)
            replaced += moe_replaced
            q_mb += moe_q_mb
            fp16_mb += moe_fp16_mb

            gc.collect()
            if (i + 1) % 8 == 0 or i == config.num_hidden_layers - 1:
                print(f"   Layer {i+1}/{config.num_hidden_layers} quantized")
            continue

        # -- MLP: fuse gate+up -> gate_up_proj (INT8 or FP16 selective mode) --
        gate_w = _get(f'{hf}.mlp.gate_proj.weight').to(dtype)
        up_w = _get(f'{hf}.mlp.up_proj.weight').to(dtype)
        gate_up_w = torch.cat([gate_w, up_w], dim=0)
        del gate_w, up_w

        if use_int8_gate_up:
            gu_int8, gu_scale = quantize_to_int8(gate_up_w)
            del gate_up_w

            gu_linear = Int8Linear(gu_int8.shape[1], gu_int8.shape[0])
            gu_linear.register_buffer('weight_int8', gu_int8)
            gu_linear.register_buffer('scale', gu_scale)
            del gu_int8, gu_scale

            layer.mlp.gate_up_proj = gu_linear.to(device)
            replaced += 1
            q_mb += gu_linear.weight_memory_mb
            fp16_mb += gu_linear.fp16_equivalent_mb
        else:
            layer.mlp.gate_up_proj.weight = torch.nn.Parameter(gate_up_w.to(device))
            del gate_up_w

        # -- down_proj (INT8 or FP16 selective mode) --
        down_w = _get(f'{hf}.mlp.down_proj.weight').to(dtype)
        if use_int8_down:
            d_int8, d_scale = quantize_to_int8(down_w)
            del down_w

            d_linear = Int8Linear(d_int8.shape[1], d_int8.shape[0])
            d_linear.register_buffer('weight_int8', d_int8)
            d_linear.register_buffer('scale', d_scale)
            del d_int8, d_scale

            layer.mlp.down_proj = d_linear.to(device)
            replaced += 1
            q_mb += d_linear.weight_memory_mb
            fp16_mb += d_linear.fp16_equivalent_mb
        else:
            layer.mlp.down_proj.weight = torch.nn.Parameter(down_w.to(device))
            del down_w

        # GC per layer to keep memory low
        gc.collect()

        if (i + 1) % 8 == 0 or i == config.num_hidden_layers - 1:
            print(f"   Layer {i+1}/{config.num_hidden_layers} quantized")

    # Cleanup handles
    _handle_cache.clear()
    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return replaced, q_mb, fp16_mb


def _load_gemma4_int8_streaming(
    model: MegaGemmLlama,
    config: LlamaConfig,
    model_path: str,
    dtype: torch.dtype,
    device: str,
    *,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> tuple[int, float, float]:
    """Stream-load Gemma 4 weights and quantize supported dense projections to INT8."""
    from safetensors import safe_open
    from ..quantization.w8a16 import Int8Linear, quantize_to_int8
    import gc

    skip_ops = _parse_int8_skip_ops()
    use_int8_qkv = "qkv" not in skip_ops
    use_int8_o = "o_proj" not in skip_ops
    use_int8_gate_up = "gate_up" not in skip_ops
    use_int8_down = "down" not in skip_ops

    key_to_file = _build_safetensor_index(model_path, key_filter=key_filter)
    handles: Dict[Tuple[str, str], object] = {}

    def _get(key: str, target_device: str = "cpu") -> torch.Tensor:
        fpath, raw_key = key_to_file[key]
        cache_key = (fpath, target_device)
        handle = handles.get(cache_key)
        if handle is None:
            handle = safe_open(fpath, framework="pt", device=target_device)
            handles[cache_key] = handle
        tensor = handle.get_tensor(raw_key)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

    def _set_weight(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        module.weight = torch.nn.Parameter(_get(key, device).contiguous())

    def _set_bias(module, key: str) -> None:
        if module is None or key not in key_to_file:
            return
        module.bias = torch.nn.Parameter(_get(key, device).contiguous())

    def _set_norm(norm_module, key: str) -> None:
        if norm_module is None or key not in key_to_file:
            return
        if not getattr(norm_module, "with_scale", True):
            return
        norm_module.weight = torch.nn.Parameter(_get(key, device).contiguous())

    def _set_norm_any(norm_module, *keys: str) -> None:
        if norm_module is None or not getattr(norm_module, "with_scale", True):
            return
        for key in keys:
            if key in key_to_file:
                norm_module.weight = torch.nn.Parameter(_get(key, device).contiguous())
                return

    def _quantize_linear_from_weight(
        weight: torch.Tensor,
        *,
        bias_tensor: Optional[torch.Tensor] = None,
    ) -> Int8Linear:
        w_int8, scale = quantize_to_int8(weight)
        linear = Int8Linear(int(w_int8.shape[1]), int(w_int8.shape[0]), bias=bias_tensor is not None)
        linear.register_buffer("weight_int8", w_int8.contiguous())
        linear.register_buffer("scale", scale.contiguous())
        if bias_tensor is not None:
            linear.bias = torch.nn.Parameter(bias_tensor.to(device=device, dtype=torch.float16).contiguous())
        return linear.to(device)

    replaced = 0
    q_mb = 0.0
    fp16_mb = 0.0

    print("   Loading embeddings + norms...")
    _set_weight(model.embed_tokens, "model.embed_tokens.weight")
    _set_weight(model.embed_tokens_per_layer, "model.embed_tokens_per_layer.weight")
    _set_weight(model.per_layer_model_projection, "model.per_layer_model_projection.weight")
    _set_norm(model.per_layer_projection_norm, "model.per_layer_projection_norm.weight")
    _set_norm(model.norm, "model.norm.weight")
    if "lm_head.weight" in key_to_file:
        _set_weight(model.lm_head, "lm_head.weight")
    elif config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    for i, layer in enumerate(model.layers):
        hf_pre = f"model.layers.{i}"
        attn = layer.self_attn
        mlp = layer.mlp
        if i == 0 and skip_ops:
            print(
                "   INT8 selective mode: keeping FP16 for "
                + ", ".join(sorted(skip_ops))
            )

        _set_norm(layer.input_layernorm, f"{hf_pre}.input_layernorm.weight")
        _set_norm(layer.post_attention_layernorm, f"{hf_pre}.post_attention_layernorm.weight")
        _set_norm(layer.pre_feedforward_layernorm, f"{hf_pre}.pre_feedforward_layernorm.weight")
        _set_norm(layer.post_feedforward_layernorm, f"{hf_pre}.post_feedforward_layernorm.weight")
        _set_norm_any(
            layer.pre_feedforward_layernorm_2,
            f"{hf_pre}.pre_feedforward_layernorm_2.weight",
            f"{hf_pre}.pre_feedforward_layernorm.weight",
        )
        _set_norm_any(
            layer.post_feedforward_layernorm_1,
            f"{hf_pre}.post_feedforward_layernorm_1.weight",
            f"{hf_pre}.post_feedforward_layernorm.weight",
        )
        _set_norm_any(
            layer.post_feedforward_layernorm_2,
            f"{hf_pre}.post_feedforward_layernorm_2.weight",
            f"{hf_pre}.post_feedforward_layernorm.weight",
        )
        _set_norm(attn.q_norm, f"{hf_pre}.self_attn.q_norm.weight")
        _set_norm(attn.k_norm, f"{hf_pre}.self_attn.k_norm.weight")

        _set_weight(layer.per_layer_input_gate, f"{hf_pre}.per_layer_input_gate.weight")
        _set_weight(layer.per_layer_projection, f"{hf_pre}.per_layer_projection.weight")
        _set_norm(layer.post_per_layer_input_norm, f"{hf_pre}.post_per_layer_input_norm.weight")
        scalar_key = f"{hf_pre}.layer_scalar"
        if scalar_key in key_to_file:
            layer.layer_scalar = _get(scalar_key, device).contiguous()

        q_key = f"{hf_pre}.self_attn.q_proj.weight"
        q_bias_key = f"{hf_pre}.self_attn.q_proj.bias"
        q_weight = _get(q_key).to(dtype).contiguous()
        q_bias = _get(q_bias_key).to(dtype).contiguous() if q_bias_key in key_to_file else None
        if use_int8_qkv:
            q_linear = _quantize_linear_from_weight(q_weight, bias_tensor=q_bias)
            attn.q_proj = q_linear
            replaced += 1
            q_mb += q_linear.weight_memory_mb
            fp16_mb += q_linear.fp16_equivalent_mb
        else:
            attn.q_proj.weight = torch.nn.Parameter(q_weight.to(device).contiguous())
            if q_bias is not None:
                attn.q_proj.bias = torch.nn.Parameter(q_bias.to(device).contiguous())
        del q_weight, q_bias

        if attn.k_proj is not None:
            k_key = f"{hf_pre}.self_attn.k_proj.weight"
            k_bias_key = f"{hf_pre}.self_attn.k_proj.bias"
            k_weight = _get(k_key).to(dtype).contiguous()
            k_bias = _get(k_bias_key).to(dtype).contiguous() if k_bias_key in key_to_file else None
            if use_int8_qkv:
                k_linear = _quantize_linear_from_weight(k_weight, bias_tensor=k_bias)
                attn.k_proj = k_linear
                replaced += 1
                q_mb += k_linear.weight_memory_mb
                fp16_mb += k_linear.fp16_equivalent_mb
            else:
                attn.k_proj.weight = torch.nn.Parameter(k_weight.to(device).contiguous())
                if k_bias is not None:
                    attn.k_proj.bias = torch.nn.Parameter(k_bias.to(device).contiguous())
            del k_weight, k_bias

        if attn.v_proj is not None and f"{hf_pre}.self_attn.v_proj.weight" in key_to_file:
            v_key = f"{hf_pre}.self_attn.v_proj.weight"
            v_bias_key = f"{hf_pre}.self_attn.v_proj.bias"
            v_weight = _get(v_key).to(dtype).contiguous()
            v_bias = _get(v_bias_key).to(dtype).contiguous() if v_bias_key in key_to_file else None
            if use_int8_qkv:
                v_linear = _quantize_linear_from_weight(v_weight, bias_tensor=v_bias)
                attn.v_proj = v_linear
                replaced += 1
                q_mb += v_linear.weight_memory_mb
                fp16_mb += v_linear.fp16_equivalent_mb
            else:
                attn.v_proj.weight = torch.nn.Parameter(v_weight.to(device).contiguous())
                if v_bias is not None:
                    attn.v_proj.bias = torch.nn.Parameter(v_bias.to(device).contiguous())
            del v_weight, v_bias

        o_key = f"{hf_pre}.self_attn.o_proj.weight"
        o_bias_key = f"{hf_pre}.self_attn.o_proj.bias"
        o_weight = _get(o_key).to(dtype).contiguous()
        o_bias = _get(o_bias_key).to(dtype).contiguous() if o_bias_key in key_to_file else None
        if use_int8_o:
            o_linear = _quantize_linear_from_weight(o_weight, bias_tensor=o_bias)
            attn.o_proj = o_linear
            replaced += 1
            q_mb += o_linear.weight_memory_mb
            fp16_mb += o_linear.fp16_equivalent_mb
        else:
            attn.o_proj.weight = torch.nn.Parameter(o_weight.to(device).contiguous())
            if o_bias is not None:
                attn.o_proj.bias = torch.nn.Parameter(o_bias.to(device).contiguous())
        del o_weight, o_bias

        gate_key = f"{hf_pre}.mlp.gate_proj.weight"
        up_key = f"{hf_pre}.mlp.up_proj.weight"
        gate_weight = _get(gate_key).to(dtype).contiguous()
        up_weight = _get(up_key).to(dtype).contiguous()
        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0).contiguous()
        del gate_weight, up_weight
        if use_int8_gate_up:
            gate_up_linear = _quantize_linear_from_weight(gate_up_weight)
            mlp.gate_up_proj = gate_up_linear
            replaced += 1
            q_mb += gate_up_linear.weight_memory_mb
            fp16_mb += gate_up_linear.fp16_equivalent_mb
        else:
            mlp.gate_up_proj.weight = torch.nn.Parameter(gate_up_weight.to(device).contiguous())
        del gate_up_weight

        down_key = f"{hf_pre}.mlp.down_proj.weight"
        down_bias_key = f"{hf_pre}.mlp.down_proj.bias"
        down_weight = _get(down_key).to(dtype).contiguous()
        down_bias = _get(down_bias_key).to(dtype).contiguous() if down_bias_key in key_to_file else None
        if use_int8_down:
            down_linear = _quantize_linear_from_weight(down_weight, bias_tensor=down_bias)
            mlp.down_proj = down_linear
            replaced += 1
            q_mb += down_linear.weight_memory_mb
            fp16_mb += down_linear.fp16_equivalent_mb
        else:
            mlp.down_proj.weight = torch.nn.Parameter(down_weight.to(device).contiguous())
            if down_bias is not None:
                mlp.down_proj.bias = torch.nn.Parameter(down_bias.to(device).contiguous())
        del down_weight, down_bias

        gc.collect()
        if (i + 1) % 8 == 0 or i == config.num_hidden_layers - 1:
            print(f"   Layer {i+1}/{config.num_hidden_layers} quantized")
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    handles.clear()
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta_params = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if meta_params:
        sample = ", ".join(meta_params[:8])
        raise RuntimeError(f"Gemma4 INT8 streaming load left meta parameters: {sample}")

    return replaced, q_mb, fp16_mb


def load_from_hf(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    n_gpu_layers: int = -1,
    offload_dir: Optional[str] = None,
    quantize: Optional[str] = None,
) -> MegaGemmLlama:
    """
    Load a model from HuggingFace.

    Supports: LLaMA 2/3, TinyLlama, Mistral, CodeLlama,
              Qwen 2.5, Qwen 3, Gemma 2
              + AWQ INT4 quantized versions
              + Streaming INT8 W8A16 quantization

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct-AWQ")
        dtype: Model precision for non-quantized layers (default: float16)
        device: Target device ('cuda' or 'cpu')
        cache_dir: HuggingFace cache directory
        n_gpu_layers: Number of layers on GPU (-1=all, 0=none). Rest offloaded.
        offload_dir: Directory for disk offload (None=CPU only)
        quantize: Quantization mode. 'int8' selects streaming INT8 W8A16;
            'fp8' is a legacy alias for the same INT8 implementation.

    Returns:
        MegaGemmLlama model ready for inference
    """
    timing: dict[str, object] = {
        "loader_kind": "hf",
        "requested_model": model_name,
        "quantize": quantize or "none",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    model_path, source_kind = resolve_model_source(model_name, cache_dir)
    timing["resolve_source_seconds"] = time.perf_counter() - phase_start
    timing["source_kind"] = source_kind
    timing["resolved_model_path"] = model_path
    if source_kind == 'local':
        print(f"[MegaGemm] Using local snapshot {model_path}...")
    else:
        print(f"[MegaGemm] Downloading {model_name}...")

    print("[MegaGemm] Loading config...")
    phase_start = time.perf_counter()
    hf_config = _load_config(model_path)
    config = LlamaConfig.from_dict(hf_config)
    _validate_supported_architecture(config, hf_config)
    timing["config_seconds"] = time.perf_counter() - phase_start
    text_only_multimodal = (
        (
            hf_config.get('vision_config') is not None
            or hf_config.get('audio_config') is not None
        )
        and config.model_type in ('qwen3_5_text', 'gemma4_text')
    )

    # Detect quantization
    is_awq = _is_awq_model(hf_config)
    awq_cfg = _get_awq_config(hf_config) if is_awq else None
    if quantize in ('int4', 'awq') and not is_awq:
        raise NotImplementedError(
            "INT4/AWQ loading requires a pre-quantized AWQ checkpoint. "
            "MegaGemm does not quantize FP16/BF16 checkpoints to INT4 on load yet."
        )
    has_linear_attention = any(layer == 'linear_attention' for layer in config.layer_types)
    if has_linear_attention and (is_awq or quantize in ('int8', 'fp8')):
        raise NotImplementedError(
            "Qwen 3.5 linear attention is currently supported only in FP16/BF16. "
            "Quantized loading for linear_attn layers is not implemented yet."
        )
    if config.model_type == 'gemma4_text' and is_awq:
        raise NotImplementedError(
            "Gemma 4 text AWQ is not wired into the Gemma4-specific "
            "loader/flat-decode path yet. FP16/BF16 and INT8 W8A16 are supported, "
            "but AWQ still needs Gemma4-native weight collection and decode kernels."
        )
    if config.model_type == 'qwen3_moe' and quantize == 'fp8':
        raise NotImplementedError(
            "Qwen3 MoE FP8 loading is not wired yet. "
            "Use quantize='int8' for W8A16 experts or an AWQ checkpoint for INT4 fallback."
        )

    if config.model_type == 'qwen3_5_text':
        print("[MegaGemm] Using native backend for Qwen 3.5 text (work in progress).")

    # Validate model type
    model_type = config.model_type
    if model_type not in SUPPORTED_MODELS:
        print(f"[MegaGemm][warn] Unknown model_type '{model_type}', treating as LLaMA-compatible")

    # Model info
    arch_info = f"{config.num_hidden_layers}L, {config.hidden_size}H, "
    arch_info += f"{config.num_attention_heads}heads"
    if config.num_key_value_heads != config.num_attention_heads:
        arch_info += f", GQA={config.num_attention_heads // config.num_key_value_heads}x"

    features = []
    if is_awq:
        features.append(f"AWQ-{awq_cfg['bits']}bit")
    elif quantize in ('int8', 'fp8'):
        features.append("INT8-W8A16")
    if config.attention_bias:
        features.append("QKV-bias")
    if config.qk_norm:
        features.append("QK-Norm")
    if config.model_type in {'qwen3_moe', 'gemma4_text'} and config.num_experts > 0:
        moe_layers = sum(1 for idx in range(config.num_hidden_layers) if config.is_moe_layer(idx))
        features.append(
            f"MoE={config.num_experts}x top-{config.num_experts_per_tok} "
            f"({moe_layers}/{config.num_hidden_layers} layers)"
        )
    if config.attention_output_gate:
        features.append("Q-gate")
    if config.rotary_dim != config.head_dim:
        features.append(f"partial-rope={config.rotary_dim}/{config.head_dim}")
    if has_linear_attention:
        num_linear = sum(layer == 'linear_attention' for layer in config.layer_types)
        features.append(f"linear-attn={num_linear}")
    if config.model_type == 'gemma4_text':
        num_sliding = sum(layer == 'sliding_attention' for layer in config.layer_types)
        num_full = sum(layer == 'full_attention' for layer in config.layer_types)
        features.append(f"gemma4 sliding/full={num_sliding}/{num_full}")
        if config.num_kv_shared_layers:
            features.append(f"kv-shared={config.num_kv_shared_layers}")
        if config.hidden_size_per_layer_input:
            features.append("PLE")
    if config.norm_offset:
        features.append("RMSNorm+1")
    if config.hidden_act != 'silu':
        features.append(f"act={config.hidden_act}")
    if config.final_logit_softcapping > 0:
        features.append(f"logit-cap={config.final_logit_softcapping}")
    if text_only_multimodal:
        features.append("text-only")

    feat_str = f" [{', '.join(features)}]" if features else ""
    print(f"[MegaGemm] Creating model ({arch_info}){feat_str}...")
    if text_only_multimodal:
        print("[MegaGemm] Multimodal checkpoint detected; loading text backbone only.")

    # Memory-efficient model creation:
    # All paths create on 'meta' device to avoid CPU RAM spike.
    # FP16 used to create real tensors (~8GB CPU RAM) but this caused OOM on Colab.
    use_int8_streaming = (quantize in ('int8', 'fp8') and not is_awq)

    phase_start = time.perf_counter()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device('meta'):
        model = MegaGemmLlama(config)
    torch.set_default_dtype(prev_dtype)
    timing["model_meta_init_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    if use_int8_streaming:
        # ── INT8 Streaming path ──
        # Loads weights one layer at a time using safetensors lazy loading.
        # Peak CPU RAM: ~2.4GB instead of ~24GB (critical for Colab/low-RAM).
        print("[MegaGemm] Loading weights (streaming INT8)...")
        if config.model_type == 'gemma4_text':
            replaced, q_mb, fp16_mb = _load_gemma4_int8_streaming(
                model,
                config,
                model_path,
                dtype,
                device,
                key_filter=_is_text_backbone_weight if text_only_multimodal else None,
            )
            timing["weight_load_mode"] = "gemma4_streaming_int8"
        else:
            replaced, q_mb, fp16_mb = _load_int8_streaming(
                model, config, model_path, dtype, device
            )
            timing["weight_load_mode"] = "streaming_int8"
        compression = fp16_mb / q_mb if q_mb > 0 else 1.0
        print(f"[MegaGemm] INT8 quantized {replaced} layers: "
              f"{q_mb:.0f}MB (vs {fp16_mb:.0f}MB FP16, {compression:.1f}x compression)")
    elif is_awq:
        print("[MegaGemm] Loading weights...")
        hf_weights = _load_safetensors(
            model_path,
            device='cpu',
            key_filter=_is_text_backbone_weight if text_only_multimodal else None,
        )

        # AWQ path: load non-quantized weights normally, then replace linears
        from ..quantization.w4a16 import W4A16_AVAILABLE
        if not W4A16_AVAILABLE:
            raise RuntimeError(
                "Triton is required for AWQ quantized models. "
                "Install with: pip install triton"
            )

        group_size = awq_cfg['group_size']

        # Load non-quantized weights (embeddings, norms, lm_head)
        # assign=True replaces meta tensors instead of copying (which fails on meta)
        non_quant = {}
        for key, val in hf_weights.items():
                if (
                    'embed_tokens' in key
                    or 'norm' in key
                    or 'lm_head' in key
                    or key.endswith('.mlp.gate.weight')
                ):
                    mapped_key = key.replace('model.', '', 1) if key.startswith('model.') else key
                    non_quant[mapped_key] = val.to(dtype)

        missing, _ = model.load_state_dict(non_quant, strict=False, assign=True)
        del non_quant

        # Replace linear layers with quantized versions
        # This replaces meta Linear layers with real QuantizedLinear + data
        replaced, quant_mb, fp16_mb = _replace_with_quantized(
            model, hf_weights, config, group_size
        )
        compression = fp16_mb / quant_mb if quant_mb > 0 else 1.0
        print(f"[MegaGemm] Quantized {replaced} layers: "
              f"{quant_mb:.0f}MB (vs {fp16_mb:.0f}MB FP16, {compression:.1f}x compression)")
        timing["weight_load_mode"] = "awq"
    else:
        # ── FP16 path ──
        # Load weights directly to GPU to avoid CPU RAM spike.
        # Model is on meta device, so assign=True replaces meta tensors.
        load_device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        fp16_streaming = (
            config.model_type == 'qwen3_moe'
            or os.environ.get("MEGAGEMM_FP16_STREAMING", "").strip().lower() in {
            "1", "true", "yes", "on",
            }
        )
        load_label = "streaming" if fp16_streaming and config.model_type != 'gemma4_text' else f"direct to {load_device}"
        print(f"[MegaGemm] Loading weights ({load_label})...")
        if config.model_type == 'gemma4_text':
            _load_gemma4_fp16_streaming(
                model,
                config,
                model_path,
                dtype,
                load_device,
                key_filter=_is_text_backbone_weight if text_only_multimodal else None,
            )
            timing["weight_load_mode"] = "gemma4_fp16_streaming"
        elif fp16_streaming:
            _load_fp16_streaming(
                model,
                config,
                model_path,
                dtype,
                load_device,
                key_filter=_is_text_backbone_weight if text_only_multimodal else None,
            )
            timing["weight_load_mode"] = "fp16_streaming"
        else:
            hf_weights = _load_safetensors(
                model_path,
                device=load_device,
                key_filter=_is_text_backbone_weight if text_only_multimodal else None,
            )

            mapped = _map_weights(hf_weights, config)
            del hf_weights

            _cast_state_dict_preserve_aliases(mapped, dtype)

            missing, unexpected = model.load_state_dict(mapped, strict=False, assign=True)
            del mapped

            if missing:
                real_missing = [k for k in missing if 'lm_head' not in k or not config.tie_word_embeddings]
                if real_missing:
                    print(f"[MegaGemm][warn] Missing keys: {real_missing}")
            if unexpected:
                print(f"[MegaGemm][warn] Unexpected keys: {unexpected}")
            timing["weight_load_mode"] = "fp16_direct"

        # Recompute RoPE caches (meta device ones are invalid)
        _refresh_rope_caches(model, config, load_device)
    timing["weight_load_seconds"] = time.perf_counter() - phase_start

    import gc

    phase_start = time.perf_counter()
    if use_int8_streaming:
        # INT8 streaming path: model already on GPU with correct weights.
        gc.collect()
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif is_awq:
        # AWQ path: careful device/dtype handling
        # Materialize any remaining meta tensors (e.g., RoPE caches from __init__)
        def _module_field(root, dotted_name: str):
            parts = dotted_name.split(".")
            module = root
            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            return module, parts[-1]

        for name, param in list(model.named_parameters()):
            if param.device.type == 'meta':
                module, field = _module_field(model, name)
                module._parameters[field] = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=dtype, device='cpu'),
                    requires_grad=param.requires_grad,
                )
            elif param.dtype == torch.float32:
                param.data = param.data.to(dtype)
        for name, buf in list(model.named_buffers()):
            if buf.device.type == 'meta':
                module, field = _module_field(model, name)
                module._buffers[field] = torch.zeros(buf.shape, dtype=buf.dtype, device='cpu')
            elif buf.dtype == torch.float32:
                buf.data = buf.data.to(dtype)

        # Recompute RoPE caches (were created on meta device)
        _refresh_rope_caches(model, config, device)

        del hf_weights
        gc.collect()

        model = model.to(device=device)
    else:
        # FP16 path: model already has weights on correct device from assign=True
        gc.collect()
        if load_device != device:
            model = model.to(device=device, dtype=dtype)
    if hasattr(model, "_refresh_gemma4_runtime_buffers"):
        model._refresh_gemma4_runtime_buffers(device=device, dtype=dtype)
    timing["materialize_seconds"] = time.perf_counter() - phase_start

    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    model.eval()

    # ── Setup layer offloading (optional) ──
    needs_offload = (n_gpu_layers >= 0 and n_gpu_layers < config.num_hidden_layers)
    phase_start = time.perf_counter()
    if needs_offload:
        from ..engine.offload import LayerOffloadManager
        offloader = LayerOffloadManager(
            num_layers=config.num_hidden_layers,
            n_gpu_layers=n_gpu_layers,
            device=device,
            dtype=dtype,
            offload_dir=offload_dir,
        )
        offloader.setup_layers(model.layers)
        model._offloader = offloader
    timing["offload_setup_seconds"] = time.perf_counter() - phase_start

    # Calculate actual memory usage
    param_mb = _unique_tensor_mb(model.parameters())
    buf_mb = _unique_tensor_mb(model.buffers())
    total_mb = param_mb + buf_mb

    model_label = model_type.upper() if model_type != 'llama' else 'LLaMA'
    quant_label = " (AWQ INT4)" if is_awq else " (INT8 W8A16)" if quantize in ('int8', 'fp8') else ""
    offload_label = ""
    if needs_offload:
        offload_label = f" [offload: {offloader}]"
    print(f"[MegaGemm] {model_label} model loaded{quant_label}! "
          f"Size: {total_mb:.0f}MB ({dtype}){offload_label}")
    timing["reported_model_size_mb"] = total_mb
    timing["total_seconds"] = time.perf_counter() - total_start
    model._load_timing = timing

    return model
