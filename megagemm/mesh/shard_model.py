"""Experimental layer-shard model runtime for MegaMesh.

This file intentionally does not modify ``InferenceEngine`` or the normal
decode path. It builds a partial ``MegaGemmLlama`` stage and executes only a
contiguous layer range. The goal is to prove remote/pod pipeline parallelism
before optimizing the tensor data plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any, Optional

import torch
from torch import nn

from ..engine.kv_cache import BlockManager
from ..models.llama import (
    HAS_RMSNORM_GATED_LINEAR,
    LlamaConfig,
    MegaGemmLlama,
    _FAST_GEMV_ENABLED_OPS,
    _FAST_GEMV_MAX_ROWS,
    _USE_CPP_DECODE_LOOP,
    _decode_loop_ops,
    fused_rmsnorm_linear_runtime_config,
    rmsnorm_gated_linear_runtime_config,
)
from ..models.loader import (
    _build_safetensor_index,
    _load_config,
    _normalize_hf_weight_key,
    _validate_supported_architecture,
    resolve_model_source,
)


@dataclass(frozen=True)
class ShardInfo:
    model_name: str
    model_path: str
    layer_start: int
    layer_end: int
    is_first: bool
    is_last: bool
    device: str
    dtype: str
    num_layers: int
    skip_lm_head: bool = False
    remote_mlp: bool = False


class _RemoteMLPProxy(nn.Module):
    """Transport-agnostic proxy for an MLP owned outside this layer shard."""

    def __init__(self, layer_idx: int, runner) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self._runner = runner

    def forward(
        self,
        x: torch.Tensor,
        timing_events: Optional[dict] = None,
        is_prefill: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        del timing_events, is_prefill
        return self._runner(self.layer_idx, x)

    def forward_decode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if not bool(kwargs.get("input_is_normed", True)):
            raise RuntimeError("remote MLP proxy requires normalized decode input")
        return self._runner(self.layer_idx, x)

    def forward_decode_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not bool(kwargs.get("input_is_normed", True)):
            raise RuntimeError("remote MLP proxy requires normalized decode input")
        out = self._runner(self.layer_idx, x)
        if torch.is_grad_enabled():
            return residual + out
        residual.add_(out)
        return residual


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _maybe_set_weight(mapped: dict[str, torch.Tensor], target: str, source: str, get) -> None:
    try:
        mapped[target] = get(source)
    except KeyError:
        return


def _build_partial_state(
    *,
    config: LlamaConfig,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
    get,
) -> dict[str, torch.Tensor]:
    """Map only the tensors needed by this stage into MegaGemm names."""

    mapped: dict[str, torch.Tensor] = {}

    if is_first:
        _maybe_set_weight(mapped, "embed_tokens.weight", "model.embed_tokens.weight", get)
        _maybe_set_weight(
            mapped,
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
            get,
        )
        _maybe_set_weight(
            mapped,
            "per_layer_model_projection.weight",
            "model.per_layer_model_projection.weight",
            get,
        )
        _maybe_set_weight(
            mapped,
            "per_layer_projection_norm.weight",
            "model.per_layer_projection_norm.weight",
            get,
        )

    if is_last:
        _maybe_set_weight(mapped, "norm.weight", "model.norm.weight", get)
        try:
            mapped["lm_head.weight"] = get("lm_head.weight")
        except KeyError:
            if config.tie_word_embeddings:
                if "embed_tokens.weight" in mapped:
                    mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
                else:
                    mapped["lm_head.weight"] = get("model.embed_tokens.weight")
            else:
                raise

    for layer_idx in range(layer_start, layer_end):
        hf_pre = f"model.layers.{layer_idx}"
        mg_pre = f"layers.{layer_idx}"
        layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"

        if layer_type == "linear_attention":
            lin_pre = f"{hf_pre}.linear_attn"
            for suffix in (
                "in_proj_qkv.weight",
                "out_proj.weight",
                "dt_bias",
                "A_log",
                "conv1d.weight",
                "norm.weight",
            ):
                _maybe_set_weight(
                    mapped,
                    f"{mg_pre}.linear_attn.{suffix}",
                    f"{lin_pre}.{suffix}",
                    get,
                )
            try:
                z_w = get(f"{lin_pre}.in_proj_z.weight")
                b_w = get(f"{lin_pre}.in_proj_b.weight")
                a_w = get(f"{lin_pre}.in_proj_a.weight")
                mapped[f"{mg_pre}.linear_attn.in_proj_baz.weight"] = torch.cat(
                    [z_w, b_w, a_w],
                    dim=0,
                )
            except KeyError:
                pass
        elif config.model_type == "gemma4_text":
            raise NotImplementedError(
                "MegaMesh shard mode does not support Gemma 4 text yet. "
                "Start with LLaMA/Mistral/Qwen full-attention or Qwen 3.5 text."
            )
        else:
            q_w = get(f"{hf_pre}.self_attn.q_proj.weight")
            k_w = get(f"{hf_pre}.self_attn.k_proj.weight")
            v_w = get(f"{hf_pre}.self_attn.v_proj.weight")
            mapped[f"{mg_pre}.self_attn.qkv_proj.weight"] = torch.cat(
                [q_w, k_w, v_w],
                dim=0,
            )
            _maybe_set_weight(
                mapped,
                f"{mg_pre}.self_attn.o_proj.weight",
                f"{hf_pre}.self_attn.o_proj.weight",
                get,
            )
            if config.attention_bias:
                try:
                    q_b = get(f"{hf_pre}.self_attn.q_proj.bias")
                    k_b = get(f"{hf_pre}.self_attn.k_proj.bias")
                    v_b = get(f"{hf_pre}.self_attn.v_proj.bias")
                    mapped[f"{mg_pre}.self_attn.qkv_proj.bias"] = torch.cat(
                        [q_b, k_b, v_b],
                        dim=0,
                    )
                except KeyError:
                    pass
            if config.qk_norm:
                _maybe_set_weight(
                    mapped,
                    f"{mg_pre}.self_attn.q_norm.weight",
                    f"{hf_pre}.self_attn.q_norm.weight",
                    get,
                )
                _maybe_set_weight(
                    mapped,
                    f"{mg_pre}.self_attn.k_norm.weight",
                    f"{hf_pre}.self_attn.k_norm.weight",
                    get,
                )

        gate_w = get(f"{hf_pre}.mlp.gate_proj.weight")
        up_w = get(f"{hf_pre}.mlp.up_proj.weight")
        mapped[f"{mg_pre}.mlp.gate_up_proj.weight"] = torch.cat([gate_w, up_w], dim=0)
        _maybe_set_weight(
            mapped,
            f"{mg_pre}.mlp.down_proj.weight",
            f"{hf_pre}.mlp.down_proj.weight",
            get,
        )
        _maybe_set_weight(
            mapped,
            f"{mg_pre}.input_layernorm.weight",
            f"{hf_pre}.input_layernorm.weight",
            get,
        )
        _maybe_set_weight(
            mapped,
            f"{mg_pre}.post_attention_layernorm.weight",
            f"{hf_pre}.post_attention_layernorm.weight",
            get,
        )

    return mapped


def _resolve_state_parent(module: torch.nn.Module, target: str) -> tuple[torch.nn.Module, str]:
    parts = target.split(".")
    parent = module
    for part in parts[:-1]:
        if part in parent._modules:
            parent = parent._modules[part]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _assign_state_tensor(
    module: torch.nn.Module,
    target: str,
    tensor: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.nn.Parameter | torch.Tensor:
    """Assign one state tensor into a meta-initialized module without a state dict."""

    if tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.to(device=device)

    parent, leaf = _resolve_state_parent(module, target)
    if leaf in parent._parameters:
        old = parent._parameters[leaf]
        requires_grad = bool(getattr(old, "requires_grad", False))
        param = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        parent._parameters[leaf] = param
        return param
    if leaf in parent._buffers:
        parent._buffers[leaf] = tensor
        return tensor
    setattr(parent, leaf, tensor)
    return tensor


def _tie_state_parameter(module: torch.nn.Module, target: str, source: str) -> bool:
    source_parent, source_leaf = _resolve_state_parent(module, source)
    param = source_parent._parameters.get(source_leaf)
    if param is None or param.device.type == "meta":
        return False
    target_parent, target_leaf = _resolve_state_parent(module, target)
    if target_leaf not in target_parent._parameters:
        return False
    target_parent._parameters[target_leaf] = param
    return True


def _load_partial_state_streaming(
    *,
    model: MegaGemmLlama,
    config: LlamaConfig,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
    skip_lm_head: bool,
    skip_mlp: bool,
    get,
    device: str,
    dtype: torch.dtype,
) -> int:
    """Load only this shard directly into the meta model, one tensor at a time.

    The previous shard loader built a full CPU state dict for the whole shard
    before moving it to the GPU.  That is fine for small models, but Qwen3-14B
    can exhaust Kaggle CPU RAM.  This path keeps peak CPU memory close to the
    largest fused tensor for one layer.
    """

    loaded = 0

    def put(target: str, tensor: torch.Tensor) -> None:
        nonlocal loaded
        _assign_state_tensor(model, target, tensor, device=device, dtype=dtype)
        loaded += 1

    def maybe_put(target: str, source: str) -> bool:
        try:
            put(target, get(source))
            return True
        except KeyError:
            return False

    def maybe_put_cat(target: str, sources: list[str]) -> bool:
        try:
            parts = [get(source) for source in sources]
        except KeyError:
            return False
        fused = torch.cat(parts, dim=0)
        del parts
        put(target, fused)
        del fused
        return True

    if is_first:
        maybe_put("embed_tokens.weight", "model.embed_tokens.weight")
        maybe_put("embed_tokens_per_layer.weight", "model.embed_tokens_per_layer.weight")
        maybe_put("per_layer_model_projection.weight", "model.per_layer_model_projection.weight")
        maybe_put("per_layer_projection_norm.weight", "model.per_layer_projection_norm.weight")

    if is_last:
        maybe_put("norm.weight", "model.norm.weight")
        if not skip_lm_head:
            if not maybe_put("lm_head.weight", "lm_head.weight"):
                if config.tie_word_embeddings:
                    if not _tie_state_parameter(model, "lm_head.weight", "embed_tokens.weight"):
                        maybe_put("lm_head.weight", "model.embed_tokens.weight")
                else:
                    raise KeyError("lm_head.weight")

    for layer_idx in range(layer_start, layer_end):
        hf_pre = f"model.layers.{layer_idx}"
        mg_pre = f"layers.{layer_idx}"
        layer_type = config.layer_types[layer_idx] if config.layer_types else "full_attention"

        if layer_type == "linear_attention":
            lin_pre = f"{hf_pre}.linear_attn"
            for suffix in (
                "in_proj_qkv.weight",
                "out_proj.weight",
                "dt_bias",
                "A_log",
                "conv1d.weight",
                "norm.weight",
            ):
                maybe_put(f"{mg_pre}.linear_attn.{suffix}", f"{lin_pre}.{suffix}")
            maybe_put_cat(
                f"{mg_pre}.linear_attn.in_proj_baz.weight",
                [
                    f"{lin_pre}.in_proj_z.weight",
                    f"{lin_pre}.in_proj_b.weight",
                    f"{lin_pre}.in_proj_a.weight",
                ],
            )
        elif config.model_type == "gemma4_text":
            raise NotImplementedError(
                "MegaMesh shard mode does not support Gemma 4 text yet. "
                "Start with LLaMA/Mistral/Qwen full-attention or Qwen 3.5 text."
            )
        else:
            maybe_put_cat(
                f"{mg_pre}.self_attn.qkv_proj.weight",
                [
                    f"{hf_pre}.self_attn.q_proj.weight",
                    f"{hf_pre}.self_attn.k_proj.weight",
                    f"{hf_pre}.self_attn.v_proj.weight",
                ],
            )
            maybe_put(
                f"{mg_pre}.self_attn.o_proj.weight",
                f"{hf_pre}.self_attn.o_proj.weight",
            )
            if config.attention_bias:
                maybe_put_cat(
                    f"{mg_pre}.self_attn.qkv_proj.bias",
                    [
                        f"{hf_pre}.self_attn.q_proj.bias",
                        f"{hf_pre}.self_attn.k_proj.bias",
                        f"{hf_pre}.self_attn.v_proj.bias",
                    ],
                )
            if config.qk_norm:
                maybe_put(f"{mg_pre}.self_attn.q_norm.weight", f"{hf_pre}.self_attn.q_norm.weight")
                maybe_put(f"{mg_pre}.self_attn.k_norm.weight", f"{hf_pre}.self_attn.k_norm.weight")

        if not skip_mlp:
            maybe_put_cat(
                f"{mg_pre}.mlp.gate_up_proj.weight",
                [
                    f"{hf_pre}.mlp.gate_proj.weight",
                    f"{hf_pre}.mlp.up_proj.weight",
                ],
            )
            maybe_put(f"{mg_pre}.mlp.down_proj.weight", f"{hf_pre}.mlp.down_proj.weight")
        maybe_put(
            f"{mg_pre}.input_layernorm.weight",
            f"{hf_pre}.input_layernorm.weight",
        )
        maybe_put(
            f"{mg_pre}.post_attention_layernorm.weight",
            f"{hf_pre}.post_attention_layernorm.weight",
        )
        gc.collect()

    return loaded


class MegaMeshShardModel:
    """One contiguous layer stage with local KV/state ownership."""

    def __init__(
        self,
        model_name: str,
        *,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        num_blocks: int = 512,
        block_size: int = 16,
        max_seq_len: int = 512,
        skip_lm_head: bool = False,
        skip_mlp: bool = False,
    ) -> None:
        self.model_name = model_name
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.is_first = bool(is_first)
        self.is_last = bool(is_last)
        self.skip_lm_head = bool(skip_lm_head)
        self.remote_mlp_enabled = bool(skip_mlp)
        if self.skip_lm_head and not self.is_last:
            raise ValueError("skip_lm_head only makes sense on the last layer shard")
        self.dtype = dtype
        self.device = device
        self.max_seq_len = int(max_seq_len)

        model_path, _ = resolve_model_source(model_name, cache_dir)
        self.model_path = model_path
        hf_config = _load_config(model_path)
        config = LlamaConfig.from_dict(hf_config)
        _validate_supported_architecture(config, hf_config)
        if not (0 <= self.layer_start < self.layer_end <= config.num_hidden_layers):
            raise ValueError(
                f"Invalid shard range {self.layer_start}:{self.layer_end} "
                f"for {config.num_hidden_layers} layers"
            )
        if config.model_type == "gemma4_text":
            raise NotImplementedError("MegaMesh shard mode does not support Gemma 4 text yet")
        self.config = config

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            self.model = MegaGemmLlama(config)
        torch.set_default_dtype(prev_dtype)

        key_to_file = _build_safetensor_index(model_path)
        handles: dict[tuple[str, str], Any] = {}

        def get(raw_key: str, target_device: str = "cpu") -> torch.Tensor:
            key = _normalize_hf_weight_key(raw_key)
            fpath, tensor_key = key_to_file[key]
            handle_key = (fpath, target_device)
            handle = handles.get(handle_key)
            if handle is None:
                from safetensors import safe_open

                handle = safe_open(fpath, framework="pt", device=target_device)
                handles[handle_key] = handle
            tensor = handle.get_tensor(tensor_key)
            if tensor.dtype != dtype and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            return tensor

        print(
            f"[MegaMesh:shard] Loading {model_name} layers "
            f"{self.layer_start}:{self.layer_end} on {device} (streaming)..."
        )
        loaded_tensors = _load_partial_state_streaming(
            model=self.model,
            config=config,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            is_first=self.is_first,
            is_last=self.is_last,
            skip_lm_head=self.skip_lm_head,
            skip_mlp=self.remote_mlp_enabled,
            get=get,
            device=device,
            dtype=dtype,
        )
        handles.clear()
        print(f"[MegaMesh:shard] Loaded {loaded_tensors} tensors into shard {self.layer_start}:{self.layer_end}.")
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.set_rope_cache_max_seq_len(max_seq_len, device=device)
        self.model.eval()
        self._local_layer_indices = list(range(self.layer_start, self.layer_end))
        self._local_layers = [self.model.layers[layer_idx] for layer_idx in self._local_layer_indices]
        self._local_layer_ropes = [
            self.model._get_layer_rope(layer_idx) for layer_idx in self._local_layer_indices
        ]
        self._local_all_full_attention = all(
            layer.layer_type == "full_attention" for layer in self._local_layers
        )
        self._local_same_rope = self.model.layer_rope_caches is None
        self._cpp_decode_loop_enabled = bool(
            _USE_CPP_DECODE_LOOP
            and _decode_loop_ops is not None
            and self._local_all_full_attention
            and self._local_same_rope
        )
        self._local_decode_fns = [
            layer.decode_forward_full_attn_infer for layer in self._local_layers
        ]

        kv_layer_indices = [
            layer_idx
            for layer_idx in range(self.layer_start, self.layer_end)
            if config.layer_types[layer_idx] != "linear_attention"
        ]
        if not kv_layer_indices:
            kv_layer_indices = []
        self.block_manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=max(1, int(num_blocks)),
            block_size=int(block_size),
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
            kv_layer_indices=kv_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
        )
        self._local_kv_caches = [
            self.block_manager.get_kv_cache(layer_idx)
            for layer_idx, layer in zip(self._local_layer_indices, self._local_layers)
            if layer.layer_type != "linear_attention"
        ]
        self._active: set[int] = set()
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_remote_mlp_runner(self, runner) -> None:
        if self.config.model_type == "gemma4_text":
            raise NotImplementedError("remote MLP shards do not support Gemma 4 text yet")
        for layer_idx, layer in zip(self._local_layer_indices, self._local_layers):
            if layer.layer_type != "full_attention":
                raise NotImplementedError(
                    "remote MLP shards currently support only full-attention layers"
                )
            layer._use_fused_norm_gateup_decode = False
            layer.mlp = _RemoteMLPProxy(layer_idx, runner)
        self.remote_mlp_enabled = True
        self._cpp_decode_loop_enabled = False
        self._local_decode_fns = [
            layer.decode_forward_full_attn_infer for layer in self._local_layers
        ]

    @property
    def info(self) -> ShardInfo:
        return ShardInfo(
            model_name=self.model_name,
            model_path=self.model_path,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            is_first=self.is_first,
            is_last=self.is_last,
            device=self.device,
            dtype=_dtype_name(self.dtype),
            num_layers=self.config.num_hidden_layers,
            skip_lm_head=self.skip_lm_head,
            remote_mlp=self.remote_mlp_enabled,
        )

    @property
    def fastpath_info(self) -> dict[str, Any]:
        if self.remote_mlp_enabled:
            name = "python-full-attention-remote-mlp"
        elif self._cpp_decode_loop_enabled:
            name = "cpp-full-attention-loop"
        elif self._local_all_full_attention:
            name = "python-full-attention-infer"
        else:
            name = "python-hybrid-linear-fused"
        return {
            "decode": name,
            "cpp_decode_loop": self._cpp_decode_loop_enabled,
            "local_all_full_attention": self._local_all_full_attention,
            "local_same_rope": self._local_same_rope,
            "linear_raw_decode": not self._local_all_full_attention,
            "remote_lm_head": self.skip_lm_head,
            "remote_mlp": self.remote_mlp_enabled,
        }

    @property
    def kernel_stats(self) -> dict[str, Any]:
        full_attn = [
            layer.self_attn
            for layer in self._local_layers
            if layer.layer_type == "full_attention" and layer.self_attn is not None
        ]
        linear_attn = [
            layer.linear_attn
            for layer in self._local_layers
            if layer.layer_type == "linear_attention" and layer.linear_attn is not None
        ]
        return {
            "local_full_attention_layers": len(full_attn),
            "local_linear_attention_layers": len(linear_attn),
            "remote_mlp_layers": len(self._local_layers) if self.remote_mlp_enabled else 0,
            "fast_gemv_max_rows": int(_FAST_GEMV_MAX_ROWS),
            "fast_gemv_ops": sorted(_FAST_GEMV_ENABLED_OPS),
            "fused_rmsnorm_linear": (
                fused_rmsnorm_linear_runtime_config()
                if callable(fused_rmsnorm_linear_runtime_config)
                else {"has_triton": False}
            ),
            "rmsnorm_gated_linear": (
                rmsnorm_gated_linear_runtime_config()
                if callable(rmsnorm_gated_linear_runtime_config)
                else {"has_triton": False}
            ),
            "rmsnorm_gated_linear_available": bool(HAS_RMSNORM_GATED_LINEAR),
            "full_attn_fused_rope_hits": sum(
                int(getattr(attn, "_fused_decode_hits", 0)) for attn in full_attn
            ),
            "linear_attn_fused_ab_hits": sum(
                int(getattr(layer, "_decode_ab_fused_hits", 0)) for layer in linear_attn
            ),
            "linear_attn_fused_rmsnorm_in_proj_hits": sum(
                int(getattr(layer, "_decode_fused_in_proj_hits", 0)) for layer in linear_attn
            ),
            "linear_attn_fast_in_proj_hits": sum(
                int(getattr(layer, "_decode_fast_in_proj_hits", 0)) for layer in linear_attn
            ),
            "linear_attn_fast_out_proj_hits": sum(
                int(getattr(layer, "_decode_fast_out_proj_hits", 0)) for layer in linear_attn
            ),
            "linear_attn_fused_norm_out_hits": sum(
                int(getattr(layer, "_fused_norm_out_hits", 0)) for layer in linear_attn
            ),
        }

    def _ensure_sequence(self, seq_id: int, num_tokens: int) -> None:
        if seq_id not in self.block_manager.block_tables:
            self.block_manager.allocate_sequence(seq_id, max(1, int(num_tokens)))
        self._active.add(seq_id)

    def free_sequence(self, seq_id: int) -> None:
        self.block_manager.free_sequence(seq_id)
        self._active.discard(seq_id)

    def _ensure_decode_capacity(self, seq_ids: list[int]) -> None:
        """Allocate the next decode KV block before kernels index block_table."""

        block_size = int(self.block_manager.block_size)
        for seq_id in seq_ids:
            seq_id = int(seq_id)
            cur_len = int(self.block_manager.seq_lens[seq_id])
            blocks = self.block_manager.block_tables[seq_id]
            blocks_needed = (cur_len + 1 + block_size - 1) // block_size
            while blocks_needed > len(blocks):
                self.block_manager.allocate_block(seq_id)
                blocks = self.block_manager.block_tables[seq_id]

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        if self.model.embed_scale != 1.0:
            hidden = hidden * self.model.embed_scale
        return hidden

    def _remote_head_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden state for an external/sharded lm_head."""

        return self.model.norm(hidden)

    @torch.inference_mode()
    def prefill(
        self,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if self.is_first:
            if input_ids is None:
                raise ValueError("first shard prefill requires input_ids")
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            hidden = self._embed(input_ids)
        elif hidden is None:
            raise ValueError("non-first shard prefill requires hidden")

        hidden = hidden.to(device=self.device, dtype=self.dtype)
        positions = positions.to(device=self.device, dtype=torch.long)
        seq_len = int(hidden.shape[1])
        self._ensure_sequence(seq_id, seq_len)

        for layer_idx, layer, rope in zip(
            self._local_layer_indices,
            self._local_layers,
            self._local_layer_ropes,
        ):
            linear_conv_state = None
            linear_recurrent_state = None
            if layer.layer_type == "linear_attention":
                linear_conv_state, linear_recurrent_state = self.block_manager.get_linear_state(
                    seq_id,
                    layer_idx,
                    device=self.device,
                )
            hidden, k_cache, v_cache, next_conv, next_recurrent = layer(
                hidden,
                *rope,
                positions,
                is_prefill=True,
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                use_linear_cache=True,
            )
            if k_cache is not None and v_cache is not None:
                self.block_manager.write_kv(seq_id, layer_idx, k_cache[0], v_cache[0])
            if next_conv is not None or next_recurrent is not None:
                self.block_manager.set_linear_state(
                    seq_id,
                    layer_idx,
                    next_conv[0] if next_conv is not None else None,
                    next_recurrent[0] if next_recurrent is not None else None,
                )

        self.block_manager.advance_seq_len(seq_id, seq_len)
        if self.is_last:
            if self.skip_lm_head:
                return {"head_hidden": self._remote_head_hidden(hidden[:, -1:, :])}
            logits = self.model._decode_head_forward(hidden[:, -1:, :])
            return {"next_token": int(torch.argmax(logits[:, -1, :], dim=-1)[0].item())}
        return {"hidden": hidden}

    def _decode_layers_python(
        self,
        *,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: list[int],
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        decode_phys_blocks: torch.Tensor,
        decode_blk_offsets: torch.Tensor,
    ) -> torch.Tensor:
        for layer_idx, layer, rope in zip(
            self._local_layer_indices,
            self._local_layers,
            self._local_layer_ropes,
        ):
            if layer.layer_type == "linear_attention":
                linear_conv_state, linear_recurrent_state = self.block_manager.get_linear_state_batch(
                    seq_ids,
                    layer_idx,
                    device=self.device,
                )
                hidden, _, _, next_conv, next_recurrent = layer(
                    hidden,
                    *rope,
                    positions,
                    kv_cache=None,
                    block_table=None,
                    seq_lens=None,
                    seq_lens_kv=seq_lens_kv,
                    decode_phys_blocks=decode_phys_blocks,
                    decode_blk_offsets=decode_blk_offsets,
                    is_prefill=False,
                    linear_conv_state=linear_conv_state,
                    linear_recurrent_state=linear_recurrent_state,
                    use_linear_cache=True,
                    input_is_normed=False,
                    input_norm_weight=layer.input_layernorm.weight,
                    input_norm_eps=layer._norm_eps,
                    input_norm_offset=layer._norm_offset,
                )
                if next_conv is not None or next_recurrent is not None:
                    self.block_manager.set_linear_state_batch(
                        seq_ids,
                        layer_idx,
                        next_conv,
                        next_recurrent,
                    )
            else:
                hidden = layer.decode_forward_full_attn_infer(
                    hidden,
                    *rope,
                    positions,
                    self.block_manager.get_kv_cache(layer_idx),
                    block_table,
                    seq_lens,
                    seq_lens_kv,
                    decode_phys_blocks,
                    decode_blk_offsets,
                )
        return hidden

    def _decode_layers_fastpath(
        self,
        *,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: list[int],
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        decode_phys_blocks: torch.Tensor,
        decode_blk_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self._cpp_decode_loop_enabled:
            return _decode_loop_ops.run_decode_fns_full_attention(
                self._local_decode_fns,
                self._local_kv_caches,
                hidden,
                self.model.cos_cache,
                self.model.sin_cache,
                positions,
                block_table,
                seq_lens,
                seq_lens_kv,
                decode_phys_blocks,
                decode_blk_offsets,
            )
        return self._decode_layers_python(
            hidden=hidden,
            positions=positions,
            seq_ids=seq_ids,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
        )

    @torch.inference_mode()
    def decode(
        self,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if seq_id not in self._active:
            raise ValueError(f"sequence {seq_id} was not prefilling on this shard")
        if self.is_first:
            if input_ids is None:
                raise ValueError("first shard decode requires input_ids")
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            hidden = self._embed(input_ids)
        elif hidden is None:
            raise ValueError("non-first shard decode requires hidden")

        hidden = hidden.to(device=self.device, dtype=self.dtype)
        positions = positions.to(device=self.device, dtype=torch.long)
        seq_ids = [int(seq_id)]
        self._ensure_decode_capacity(seq_ids)
        block_table = self.block_manager.get_block_table_tensor(seq_ids)
        seq_lens = self.block_manager.get_seq_lens_tensor(seq_ids)
        seq_lens_kv = seq_lens + 1
        block_size = self.block_manager.block_size
        blk_ids = torch.div(seq_lens, block_size, rounding_mode="floor").long()
        decode_blk_offsets = torch.remainder(seq_lens, block_size).long()
        decode_phys_blocks = block_table[
            torch.arange(len(seq_ids), device=block_table.device),
            blk_ids,
        ]

        hidden = self._decode_layers_fastpath(
            hidden=hidden,
            positions=positions,
            seq_ids=seq_ids,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
        )

        self.block_manager.advance_seq_len(seq_id, 1)
        if self.is_last:
            if self.skip_lm_head:
                return {"head_hidden": self._remote_head_hidden(hidden)}
            logits = self.model._decode_head_forward(hidden)
            return {"next_token": int(torch.argmax(logits[:, -1, :], dim=-1)[0].item())}
        return {"hidden": hidden}

    @torch.inference_mode()
    def decode_batch(
        self,
        *,
        seq_ids: list[int],
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Decode one token for multiple active sequences in one layer pass."""

        seq_ids = [int(seq_id) for seq_id in seq_ids]
        if not seq_ids:
            raise ValueError("decode_batch requires at least one sequence")
        missing = [seq_id for seq_id in seq_ids if seq_id not in self._active]
        if missing:
            raise ValueError(f"sequences were not prefilling on this shard: {missing}")

        if self.is_first:
            if input_ids is None:
                raise ValueError("first shard decode_batch requires input_ids")
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            hidden = self._embed(input_ids)
        elif hidden is None:
            raise ValueError("non-first shard decode_batch requires hidden")

        hidden = hidden.to(device=self.device, dtype=self.dtype)
        positions = positions.to(device=self.device, dtype=torch.long)
        if hidden.shape[0] != len(seq_ids):
            raise ValueError(
                f"decode_batch hidden batch {hidden.shape[0]} != {len(seq_ids)} seq_ids"
            )
        if positions.shape[0] != len(seq_ids):
            raise ValueError(
                f"decode_batch positions batch {positions.shape[0]} != {len(seq_ids)} seq_ids"
            )

        self._ensure_decode_capacity(seq_ids)
        block_table = self.block_manager.get_block_table_tensor(seq_ids)
        seq_lens = self.block_manager.get_seq_lens_tensor(seq_ids)
        seq_lens_kv = seq_lens + 1
        block_size = self.block_manager.block_size
        blk_ids = torch.div(seq_lens, block_size, rounding_mode="floor").long()
        decode_blk_offsets = torch.remainder(seq_lens, block_size).long()
        decode_phys_blocks = block_table[
            torch.arange(len(seq_ids), device=block_table.device),
            blk_ids,
        ]

        hidden = self._decode_layers_fastpath(
            hidden=hidden,
            positions=positions,
            seq_ids=seq_ids,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
        )

        self.block_manager.advance_seq_len_batch(seq_ids, 1)
        if self.is_last:
            if self.skip_lm_head:
                return {"head_hidden": self._remote_head_hidden(hidden)}
            logits = self.model._decode_head_forward(hidden)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1).detach().to("cpu")
            return {"next_tokens": [int(token) for token in next_tokens.tolist()]}
        return {"hidden": hidden}


__all__ = ["MegaMeshShardModel", "ShardInfo"]
