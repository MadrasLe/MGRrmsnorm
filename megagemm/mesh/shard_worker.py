"""HTTP worker for experimental MegaMesh layer-shard mode."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from typing import Any, Dict, Optional

import torch

from .binary_codec import (
    CONTENT_TYPE,
    PinnedTensorPool,
    TensorFrameParts,
    decode_tensor_frame,
    encode_tensor_frame,
    encode_tensor_frame_parts,
)
from .ttp import TTPClient, ttp_runtime_info
from .shard_model import MegaMeshShardModel
from .tensor_codec import tensor_from_payload, tensor_to_payload
from ..models.llama import LlamaConfig
from ..models.loader import (
    _build_safetensor_index,
    _load_config,
    _normalize_hf_weight_key,
    _validate_supported_architecture,
    resolve_model_source,
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _split_endpoints(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = [str(part) for part in raw]
    else:
        parts = str(raw).split(",")
    return [part.strip() for part in parts if part.strip()]


def _find_lm_head_key(
    key_to_file: Dict[str, tuple[str, str]],
    config: LlamaConfig,
) -> str:
    if "lm_head.weight" in key_to_file:
        return "lm_head.weight"
    if config.tie_word_embeddings and "model.embed_tokens.weight" in key_to_file:
        return "model.embed_tokens.weight"
    raise KeyError("lm_head.weight")


def _load_safetensor_rows(
    *,
    key_to_file: Dict[str, tuple[str, str]],
    key: str,
    row_start: int,
    row_end: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    from safetensors import safe_open

    normalized = _normalize_hf_weight_key(key)
    fpath, tensor_key = key_to_file[normalized]
    with safe_open(fpath, framework="pt", device="cpu") as handle:
        try:
            tensor_slice = handle.get_slice(tensor_key)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if len(shape) != 2:
                raise ValueError(f"{normalized} must be 2D, got shape={shape}")
            rows = tensor_slice[int(row_start) : int(row_end)]
        except AttributeError:
            full = handle.get_tensor(tensor_key)
            rows = full[int(row_start) : int(row_end)]
        except TypeError:
            full = handle.get_tensor(tensor_key)
            rows = full[int(row_start) : int(row_end)]
    if rows.dtype != dtype and rows.is_floating_point():
        rows = rows.to(dtype)
    rows = rows.contiguous().clone()
    return rows.to(device=device)


def _load_safetensor_slice(
    *,
    key_to_file: Dict[str, tuple[str, str]],
    key: str,
    row_start: int | None = None,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    from safetensors import safe_open

    normalized = _normalize_hf_weight_key(key)
    fpath, tensor_key = key_to_file[normalized]
    with safe_open(fpath, framework="pt", device="cpu") as handle:
        try:
            tensor_slice = handle.get_slice(tensor_key)
            row_slice = slice(row_start, row_end) if row_start is not None or row_end is not None else slice(None)
            col_slice = slice(col_start, col_end) if col_start is not None or col_end is not None else None
            if col_slice is None:
                tensor = tensor_slice[row_slice]
            else:
                tensor = tensor_slice[row_slice, col_slice]
        except (AttributeError, TypeError):
            full = handle.get_tensor(tensor_key)
            row_slice = slice(row_start, row_end) if row_start is not None or row_end is not None else slice(None)
            col_slice = slice(col_start, col_end) if col_start is not None or col_end is not None else slice(None)
            tensor = full[row_slice, col_slice]
    if tensor.dtype != dtype and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    tensor = tensor.contiguous().clone()
    return tensor.to(device=device)


def _layer_intermediate_size(config: LlamaConfig, layer_idx: int) -> int:
    if config.mlp_intermediate_sizes and int(layer_idx) < len(config.mlp_intermediate_sizes):
        return int(config.mlp_intermediate_sizes[int(layer_idx)])
    return int(config.intermediate_size)


class MegaMeshLMHeadWorker:
    """Vocabulary-row shard for the final lm_head argmax.

    This is intentionally separate from the layer shard: it owns only
    ``lm_head.weight[vocab_start:vocab_end]`` (or tied embeddings) and returns
    local greedy winners.  The last layer shard reduces those local winners into
    one global token, so normal MegaGemm inference remains untouched.
    """

    def __init__(
        self,
        model: str,
        *,
        vocab_start: int,
        vocab_end: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        name: str = "",
    ) -> None:
        self.name = name or os.environ.get("MEGAMESH_LM_HEAD_NAME") or "lm-head"
        self.started_at = time.time()
        self.model_name = model
        self.dtype = dtype
        self.device = device
        self.vocab_start = int(vocab_start)
        self.vocab_end = int(vocab_end)

        model_path, _ = resolve_model_source(model, cache_dir)
        self.model_path = model_path
        hf_config = _load_config(model_path)
        config = LlamaConfig.from_dict(hf_config)
        _validate_supported_architecture(config, hf_config)
        self.config = config
        if not (0 <= self.vocab_start < self.vocab_end <= int(config.vocab_size)):
            raise ValueError(
                f"Invalid lm_head vocab range {self.vocab_start}:{self.vocab_end} "
                f"for vocab_size={config.vocab_size}"
            )

        key_to_file = _build_safetensor_index(model_path)
        source_key = _find_lm_head_key(key_to_file, config)
        print(
            f"[MegaMesh:lm_head] Loading {model} vocab "
            f"{self.vocab_start}:{self.vocab_end} from {source_key} on {device}..."
        )
        self.weight = _load_safetensor_rows(
            key_to_file=key_to_file,
            key=source_key,
            row_start=self.vocab_start,
            row_end=self.vocab_end,
            dtype=dtype,
            device=device,
        )
        print(
            f"[MegaMesh:lm_head] Loaded rows={self.weight.shape[0]} "
            f"hidden={self.weight.shape[1]} dtype={_dtype_name(self.weight.dtype)}."
        )

    def health(self) -> Dict[str, Any]:
        gpu = None
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(self.device).index
            if idx is None:
                idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "index": int(idx),
                "free_mb": round(free / 1024**2, 2),
                "total_mb": round(total / 1024**2, 2),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 2),
            }
        return {
            "ok": True,
            "name": self.name,
            "mode": "lm-head-shard-experimental",
            "model": self.model_name,
            "model_path": self.model_path,
            "vocab_start": self.vocab_start,
            "vocab_end": self.vocab_end,
            "rows": int(self.weight.shape[0]),
            "hidden_size": int(self.weight.shape[1]),
            "device": self.device,
            "dtype": _dtype_name(self.dtype),
            "transports": ["ttp"],
            "ttp_runtime": ttp_runtime_info(),
            "uptime_s": round(time.time() - self.started_at, 2),
            "gpu": gpu,
        }

    @torch.inference_mode()
    def lm_head_argmax_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        del meta
        if "hidden" not in tensors:
            raise ValueError("lm_head_argmax requires hidden tensor")
        hidden = tensors["hidden"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        if hidden.ndim == 3:
            hidden_2d = hidden[:, -1, :]
        elif hidden.ndim == 2:
            hidden_2d = hidden
        else:
            raise ValueError(f"hidden must be [B,T,H] or [B,H], got shape={tuple(hidden.shape)}")
        if int(hidden_2d.shape[-1]) != int(self.weight.shape[1]):
            raise ValueError(
                f"hidden size {hidden_2d.shape[-1]} != lm_head shard hidden {self.weight.shape[1]}"
            )

        logits = torch.nn.functional.linear(hidden_2d, self.weight)
        values, local_ids = torch.max(logits, dim=-1)
        token_ids = (local_ids.to(torch.long) + self.vocab_start).detach().to("cpu")
        return encode_tensor_frame_parts(
            {
                "ok": True,
                "vocab_start": self.vocab_start,
                "vocab_end": self.vocab_end,
                "token_ids": [int(token) for token in token_ids.tolist()],
                "logits": [float(value) for value in values.detach().to("cpu").tolist()],
            }
        )


class MegaMeshMLPShardWorker:
    """Intermediate-dimension shard for transformer MLP blocks.

    Each worker owns gate/up rows and matching down-proj columns for a layer
    range. The layer stage sends normalized MLP input, receives partial hidden
    contributions from all MLP shards, sums them, then applies the residual
    locally.
    """

    def __init__(
        self,
        model: str,
        *,
        layer_start: int,
        layer_end: int,
        intermediate_start: int,
        intermediate_end: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        name: str = "",
    ) -> None:
        self.name = name or os.environ.get("MEGAMESH_MLP_NAME") or "mlp-shard"
        self.started_at = time.time()
        self.model_name = model
        self.dtype = dtype
        self.device = device
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.intermediate_start = int(intermediate_start)
        self.intermediate_end = int(intermediate_end)
        self.forward_count = 0

        model_path, _ = resolve_model_source(model, cache_dir)
        self.model_path = model_path
        hf_config = _load_config(model_path)
        config = LlamaConfig.from_dict(hf_config)
        _validate_supported_architecture(config, hf_config)
        self.config = config
        if config.model_type == "gemma4_text":
            raise NotImplementedError("remote MLP shards do not support Gemma 4 text yet")
        if not (0 <= self.layer_start < self.layer_end <= int(config.num_hidden_layers)):
            raise ValueError(
                f"Invalid MLP layer range {self.layer_start}:{self.layer_end} "
                f"for {config.num_hidden_layers} layers"
            )

        key_to_file = _build_safetensor_index(model_path)
        self.layers: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in range(self.layer_start, self.layer_end):
            intermediate = _layer_intermediate_size(config, layer_idx)
            if not (0 <= self.intermediate_start < self.intermediate_end <= intermediate):
                raise ValueError(
                    f"Invalid intermediate range {self.intermediate_start}:{self.intermediate_end} "
                    f"for layer {layer_idx} intermediate_size={intermediate}"
                )
            hf_pre = f"model.layers.{layer_idx}.mlp"
            print(
                f"[MegaMesh:mlp] Loading {model} layer={layer_idx} intermediate "
                f"{self.intermediate_start}:{self.intermediate_end} on {device}..."
            )
            self.layers[layer_idx] = {
                "gate": _load_safetensor_slice(
                    key_to_file=key_to_file,
                    key=f"{hf_pre}.gate_proj.weight",
                    row_start=self.intermediate_start,
                    row_end=self.intermediate_end,
                    dtype=dtype,
                    device=device,
                ),
                "up": _load_safetensor_slice(
                    key_to_file=key_to_file,
                    key=f"{hf_pre}.up_proj.weight",
                    row_start=self.intermediate_start,
                    row_end=self.intermediate_end,
                    dtype=dtype,
                    device=device,
                ),
                "down": _load_safetensor_slice(
                    key_to_file=key_to_file,
                    key=f"{hf_pre}.down_proj.weight",
                    col_start=self.intermediate_start,
                    col_end=self.intermediate_end,
                    dtype=dtype,
                    device=device,
                ),
            }

    def health(self) -> Dict[str, Any]:
        gpu = None
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(self.device).index
            if idx is None:
                idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "index": int(idx),
                "free_mb": round(free / 1024**2, 2),
                "total_mb": round(total / 1024**2, 2),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 2),
            }
        return {
            "ok": True,
            "name": self.name,
            "mode": "mlp-shard-experimental",
            "model": self.model_name,
            "model_path": self.model_path,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "intermediate_start": self.intermediate_start,
            "intermediate_end": self.intermediate_end,
            "layers": sorted(self.layers),
            "device": self.device,
            "dtype": _dtype_name(self.dtype),
            "transports": ["ttp"],
            "ttp_runtime": ttp_runtime_info(),
            "forward_count": self.forward_count,
            "uptime_s": round(time.time() - self.started_at, 2),
            "gpu": gpu,
        }

    @torch.inference_mode()
    def mlp_forward_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        layer_idx = int(meta.get("layer_idx", -1))
        weights = self.layers.get(layer_idx)
        if weights is None:
            raise ValueError(
                f"{self.name} does not own MLP layer {layer_idx}; "
                f"range={self.layer_start}:{self.layer_end}"
            )
        if "hidden" not in tensors:
            raise ValueError("mlp_forward requires hidden tensor")
        x = tensors["hidden"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        gate = torch.nn.functional.linear(x, weights["gate"])
        up = torch.nn.functional.linear(x, weights["up"])
        if self.config.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            activated = torch.nn.functional.gelu(gate, approximate="tanh")
        else:
            activated = torch.nn.functional.silu(gate)
        activated.mul_(up)
        partial = torch.nn.functional.linear(activated, weights["down"])
        self.forward_count += 1
        return encode_tensor_frame_parts(
            {"ok": True, "layer_idx": layer_idx},
            {"partial": partial},
        )


class MegaMeshShardWorker:
    """One stateful layer-stage worker."""

    def __init__(
        self,
        model: str,
        *,
        layer_start: int,
        layer_end: int,
        is_first: bool = False,
        is_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_blocks: int = 512,
        block_size: int = 16,
        max_seq_len: int = 512,
        cache_dir: Optional[str] = None,
        name: str = "",
        ttp_pinned: bool = True,
        lm_head_shards: str | list[str] | tuple[str, ...] | None = None,
        mlp_shards: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.name = name or os.environ.get("MEGAMESH_SHARD_NAME") or "shard"
        self.started_at = time.time()
        self.lock = threading.Lock()
        self.prefill_count = 0
        self.decode_count = 0
        self.ttp_chain_forward_count = 0
        self._ttp_forward_clients: dict[str, TTPClient] = {}
        self.lm_head_shards = _split_endpoints(lm_head_shards)
        if self.lm_head_shards and not is_last:
            raise ValueError("lm_head_shards can only be attached to the last layer shard")
        self._lm_head_clients = [TTPClient(endpoint) for endpoint in self.lm_head_shards]
        self.mlp_shards = _split_endpoints(mlp_shards)
        self._mlp_clients = [TTPClient(endpoint) for endpoint in self.mlp_shards]
        self._mlp_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=len(self._mlp_clients))
            if self._mlp_clients
            else None
        )
        self.ttp_out_pool = PinnedTensorPool(enabled=ttp_pinned)
        self.stage = MegaMeshShardModel(
            model,
            layer_start=layer_start,
            layer_end=layer_end,
            is_first=is_first,
            is_last=is_last,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
            num_blocks=num_blocks,
            block_size=block_size,
            max_seq_len=max_seq_len,
            skip_lm_head=bool(self.lm_head_shards),
            skip_mlp=bool(self.mlp_shards),
        )
        if self._mlp_clients:
            self.stage.set_remote_mlp_runner(self._remote_mlp_forward)

    def health(self) -> Dict[str, Any]:
        gpu = None
        device = self.stage.device
        if str(device).startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(device).index
            if idx is None:
                idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "index": int(idx),
                "free_mb": round(free / 1024**2, 2),
                "total_mb": round(total / 1024**2, 2),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 2),
            }
        info = self.stage.info
        return {
            "ok": True,
            "name": self.name,
            "mode": "layer-shard-experimental",
            "model": info.model_name,
            "layer_start": info.layer_start,
            "layer_end": info.layer_end,
            "is_first": info.is_first,
            "is_last": info.is_last,
            "device": info.device,
            "dtype": info.dtype,
            "transports": ["ttp", "binary", "json"],
            "fastpath": self.stage.fastpath_info,
            "kernel_stats": self.stage.kernel_stats,
            "kernel_policy": {
                "flat_decode": os.environ.get("MEGAGEMM_FLAT_DECODE", ""),
                "disable_cuda_rmsnorm": os.environ.get("MEGAGEMM_DISABLE_CUDA_RMSNORM", ""),
            },
            "lm_head": {
                "mode": "remote-sharded" if self.lm_head_shards else "local",
                "shards": list(self.lm_head_shards),
                "skip_local_lm_head": info.skip_lm_head,
            },
            "mlp": {
                "mode": "remote-sharded" if self.mlp_shards else "local",
                "shards": list(self.mlp_shards),
                "skip_local_mlp": info.remote_mlp,
            },
            "ttp_runtime": ttp_runtime_info(),
            "prefill_count": self.prefill_count,
            "decode_count": self.decode_count,
            "ttp_chain_forward_count": self.ttp_chain_forward_count,
            "ttp_out_pool": self.ttp_out_pool.stats(),
            "uptime_s": round(time.time() - self.started_at, 2),
            "gpu": gpu,
        }

    def _parse_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tensors: Dict[str, torch.Tensor] = {
            "positions": tensor_from_payload(payload["positions"], device="cpu"),
        }
        if payload.get("input_ids") is not None:
            tensors["input_ids"] = tensor_from_payload(payload["input_ids"], device="cpu")
        if payload.get("hidden") is not None:
            tensors["hidden"] = tensor_from_payload(payload["hidden"], device="cpu")
        return self._parse_binary_inputs({"seq_id": int(payload.get("seq_id", 1))}, tensors)

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "hidden" in result:
            return {"ok": True, "hidden": tensor_to_payload(result["hidden"])}
        if "head_hidden" in result:
            raise RuntimeError("remote lm_head result was not reduced before formatting")
        return {"ok": True, "next_token": int(result["next_token"])}

    def _parse_binary_inputs(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "seq_id": int(meta.get("seq_id", 1)),
            "positions": tensors["positions"].to(device=self.stage.device, non_blocking=True),
        }
        if "input_ids" in tensors:
            input_ids = tensors["input_ids"]
            input_ids_cpu = input_ids.detach().to("cpu", dtype=torch.long)
            if input_ids_cpu.numel():
                min_id = int(input_ids_cpu.min().item())
                max_id = int(input_ids_cpu.max().item())
                vocab_size = int(self.stage.config.vocab_size)
                if min_id < 0 or max_id >= vocab_size:
                    raise ValueError(
                        f"input_ids out of range for vocab_size={vocab_size}: "
                        f"min={min_id} max={max_id}"
                    )
            parsed["input_ids"] = input_ids_cpu.to(device=self.stage.device, non_blocking=True)
        if "hidden" in tensors:
            parsed["hidden"] = tensors["hidden"].to(device=self.stage.device, non_blocking=True)
        return parsed

    def _parse_binary_batch_inputs(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        seq_ids = meta.get("seq_ids")
        if not isinstance(seq_ids, list):
            raise ValueError("decode_batch requires meta.seq_ids list")
        parsed: Dict[str, Any] = {
            "seq_ids": [int(seq_id) for seq_id in seq_ids],
            "positions": tensors["positions"].to(device=self.stage.device, non_blocking=True),
        }
        if "input_ids" in tensors:
            input_ids = tensors["input_ids"]
            input_ids_cpu = input_ids.detach().to("cpu", dtype=torch.long)
            if input_ids_cpu.numel():
                min_id = int(input_ids_cpu.min().item())
                max_id = int(input_ids_cpu.max().item())
                vocab_size = int(self.stage.config.vocab_size)
                if min_id < 0 or max_id >= vocab_size:
                    raise ValueError(
                        f"input_ids out of range for vocab_size={vocab_size}: "
                        f"min={min_id} max={max_id}"
                    )
            parsed["input_ids"] = input_ids_cpu.to(device=self.stage.device, non_blocking=True)
        if "hidden" in tensors:
            parsed["hidden"] = tensors["hidden"].to(device=self.stage.device, non_blocking=True)
        return parsed

    def _format_binary_result(self, result: Dict[str, Any]) -> bytes:
        if "hidden" in result:
            return encode_tensor_frame({"ok": True}, {"hidden": result["hidden"]})
        if "head_hidden" in result:
            raise RuntimeError("remote lm_head result was not reduced before formatting")
        if "next_tokens" in result:
            return encode_tensor_frame({"ok": True, "next_tokens": result["next_tokens"]})
        return encode_tensor_frame({"ok": True, "next_token": int(result["next_token"])})

    def _format_ttp_result(self, result: Dict[str, Any]) -> TensorFrameParts:
        if "hidden" in result:
            return encode_tensor_frame_parts(
                {"ok": True},
                {"hidden": result["hidden"]},
                pool=self.ttp_out_pool,
            )
        if "head_hidden" in result:
            raise RuntimeError("remote lm_head result was not reduced before formatting")
        if "next_tokens" in result:
            return encode_tensor_frame_parts({"ok": True, "next_tokens": result["next_tokens"]})
        return encode_tensor_frame_parts({"ok": True, "next_token": int(result["next_token"])})

    def _finish_lm_head_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "head_hidden" not in result:
            return result
        if not self._lm_head_clients:
            raise RuntimeError("last shard produced head_hidden but no lm_head_shards are configured")

        hidden = result["head_hidden"]
        best_tokens: list[int] | None = None
        best_logits: list[float] | None = None
        for client in self._lm_head_clients:
            meta, _ = client.request(
                "lm_head_argmax",
                {},
                {"hidden": hidden},
                device="cpu",
            )
            token_ids = [int(token) for token in meta.get("token_ids", [])]
            logits = [float(value) for value in meta.get("logits", [])]
            if len(token_ids) != len(logits):
                raise ValueError("lm_head shard returned token_ids/logits length mismatch")
            if best_tokens is None:
                best_tokens = token_ids
                best_logits = logits
                continue
            if len(token_ids) != len(best_tokens):
                raise ValueError("lm_head shard returned a different batch length")
            assert best_logits is not None
            for idx, (token, value) in enumerate(zip(token_ids, logits)):
                if value > best_logits[idx]:
                    best_logits[idx] = value
                    best_tokens[idx] = token

        if best_tokens is None or not best_tokens:
            raise ValueError("lm_head shards returned no tokens")
        if len(best_tokens) == 1:
            return {"next_token": int(best_tokens[0])}
        return {"next_tokens": [int(token) for token in best_tokens]}

    def _remote_mlp_forward(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        if not self._mlp_clients:
            raise RuntimeError("remote MLP proxy called without mlp_shards")

        def request(client: TTPClient) -> torch.Tensor:
            _, tensors = client.request(
                "mlp_forward",
                {"layer_idx": int(layer_idx)},
                {"hidden": hidden},
                device=self.stage.device,
            )
            if "partial" not in tensors:
                raise RuntimeError("MLP shard response did not include partial tensor")
            return tensors["partial"].to(device=self.stage.device, non_blocking=True)

        if self._mlp_executor is None or len(self._mlp_clients) == 1:
            partials = [request(client) for client in self._mlp_clients]
        else:
            futures = [self._mlp_executor.submit(request, client) for client in self._mlp_clients]
            partials = [future.result() for future in futures]
        out = partials[0]
        for partial in partials[1:]:
            out = out + partial
        return out

    def _forward_client(self, next_stage: str) -> TTPClient:
        endpoint = str(next_stage).strip()
        if not endpoint:
            raise ValueError("TTP chain requires non-empty next_stage")
        client = self._ttp_forward_clients.get(endpoint)
        if client is None:
            client = TTPClient(endpoint)
            self._ttp_forward_clients[endpoint] = client
        return client

    def probe_peer_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Measure this worker's TTP path to another worker.

        The coordinator asks a source worker to probe a target worker, so the
        measurement follows the same distributed direction used by shard-chain
        hidden-state forwarding. The payload is CPU uint8 on purpose: this
        measures portable TTP/network cost, not local GPU paths.
        """

        del tensors
        target = str(meta.get("target", "")).strip()
        if not target:
            raise ValueError("probe_peer requires meta.target")
        runs = max(1, int(meta.get("runs", 5)))
        warmup = max(0, int(meta.get("warmup", 1)))
        payload_bytes = max(0, int(meta.get("payload_bytes", 0)))
        payload = (
            torch.empty((payload_bytes,), dtype=torch.uint8, device="cpu")
            if payload_bytes > 0
            else None
        )
        client = self._forward_client(target)

        def once() -> float:
            t0 = time.perf_counter()
            client.request(
                "ping",
                {"probe_bytes": payload_bytes},
                {"payload": payload} if payload is not None else None,
                device="cpu",
            )
            return (time.perf_counter() - t0) * 1000.0

        for _ in range(warmup):
            once()
        samples = [once() for _ in range(runs)]
        avg_ms = sum(samples) / len(samples)
        min_ms = min(samples)
        mbps = 0.0
        if payload_bytes > 0 and avg_ms > 0:
            mbps = (payload_bytes * 8.0) / (avg_ms / 1000.0) / 1_000_000.0
        return {
            "ok": True,
            "src": self.name,
            "dst": target,
            "payload_bytes": payload_bytes,
            "runs": runs,
            "warmup": warmup,
            "rtt_ms_avg": round(avg_ms, 4),
            "rtt_ms_min": round(min_ms, 4),
            "one_way_payload_mbps": round(mbps, 3),
            "samples_ms": [round(value, 4) for value in samples],
        }

    @staticmethod
    def _route_from_meta(meta: Dict[str, Any]) -> list[str]:
        raw_route = meta.get("next_stages")
        if isinstance(raw_route, (list, tuple)):
            return [str(stage).strip() for stage in raw_route if str(stage).strip()]
        next_stage = str(meta.get("next_stage", "")).strip()
        return [next_stage] if next_stage else []

    def _forward_hidden_ttp(
        self,
        *,
        op: str,
        meta: Dict[str, Any],
        positions: torch.Tensor,
        result: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        route = self._route_from_meta(meta)
        if "hidden" not in result or not route:
            raise ValueError("TTP chain forward requires hidden result and next_stage(s)")

        forward_meta = {
            key: value
            for key, value in meta.items()
            if key not in {"next_stage", "next_stages", "op"}
        }
        forward_meta.pop("seq_id", None)
        forward_meta.pop("seq_ids", None)
        if "seq_ids" in meta:
            forward_meta["seq_ids"] = [int(seq_id) for seq_id in meta["seq_ids"]]
        else:
            forward_meta["seq_id"] = int(meta.get("seq_id", 1))

        next_stage = route[0]
        remaining_route = route[1:]
        forward_op = op
        if remaining_route:
            forward_op = f"{op}_chain"
            forward_meta["next_stages"] = remaining_route

        client = self._forward_client(str(next_stage))
        out_meta, out_tensors = client.request(
            forward_op,
            forward_meta,
            {
                "positions": positions,
                "hidden": result["hidden"],
            },
            device="cpu",
        )
        self.ttp_chain_forward_count += 1
        out_meta = dict(out_meta)
        out_meta["_chain_hops"] = int(out_meta.get("_chain_hops", 0)) + 1
        return out_meta, out_tensors

    def _forward_ttp_result(
        self,
        *,
        op: str,
        meta: Dict[str, Any],
        positions: torch.Tensor,
        result: Dict[str, Any],
    ) -> TensorFrameParts:
        route = self._route_from_meta(meta)
        if "hidden" not in result or not route:
            return self._format_ttp_result(result)
        out_meta, out_tensors = self._forward_hidden_ttp(
            op=op,
            meta=meta,
            positions=positions,
            result=result,
        )
        return encode_tensor_frame_parts(out_meta, out_tensors)

    def generate_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        """Run a full single-sequence decode loop from the first TTP stage."""

        seq_id = int(meta.get("seq_id", 1))
        route = self._route_from_meta(meta)
        if not route:
            raise ValueError("generate_chain requires meta.next_stage(s)")
        max_new_tokens = max(0, int(meta.get("max_new_tokens", 64)))
        eos_ids = {int(token) for token in meta.get("eos_ids", [])}
        prompt_len = int(tensors["input_ids"].shape[-1])
        generated: list[int] = []
        decode_steps = 0
        total_chain_hops = 0

        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.prefill(**parsed)
            self.prefill_count += 1
        out_meta, _ = self._forward_hidden_ttp(
            op="prefill",
            meta={"seq_id": seq_id, "next_stages": route},
            positions=tensors["positions"],
            result=result,
        )
        total_chain_hops += int(out_meta.get("_chain_hops", 0))
        if "next_token" not in out_meta:
            raise RuntimeError("last shard did not return next_token during remote prefill")

        cur_token = int(out_meta["next_token"])
        for step in range(max_new_tokens):
            generated.append(cur_token)
            if cur_token in eos_ids or len(generated) >= max_new_tokens:
                break

            positions = torch.tensor([[prompt_len + step]], dtype=torch.long)
            input_ids = torch.tensor([[cur_token]], dtype=torch.long)
            with self.lock:
                parsed = self._parse_binary_inputs(
                    {"seq_id": seq_id},
                    {"positions": positions, "input_ids": input_ids},
                )
                result = self.stage.decode(**parsed)
                self.decode_count += 1
            decode_steps += 1
            out_meta, _ = self._forward_hidden_ttp(
                op="decode",
                meta={"seq_id": seq_id, "next_stages": route},
                positions=positions,
                result=result,
            )
            total_chain_hops += int(out_meta.get("_chain_hops", 0))
            if "next_token" not in out_meta:
                raise RuntimeError("last shard did not return next_token during remote decode")
            cur_token = int(out_meta["next_token"])

        return encode_tensor_frame_parts(
            {
                "ok": True,
                "generated": generated,
                "generated_tokens": len(generated),
                "decode_steps": decode_steps,
                "remote_chain_loop": True,
                "chain_forwards": total_chain_hops,
            }
        )

    def generate_batch_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        """Run a full batched decode loop from the first TTP stage."""

        seq_ids = [int(seq_id) for seq_id in meta.get("seq_ids", [])]
        if not seq_ids:
            raise ValueError("generate_batch_chain requires meta.seq_ids")
        route = self._route_from_meta(meta)
        if not route:
            raise ValueError("generate_batch_chain requires meta.next_stage(s)")

        max_new_tokens = max(0, int(meta.get("max_new_tokens", 64)))
        microbatch_size = max(1, int(meta.get("microbatch_size", len(seq_ids))))
        eos_ids = {int(token) for token in meta.get("eos_ids", [])}
        input_rows = tensors["input_ids"].detach().to("cpu", dtype=torch.long)
        prompt_lengths = tensors["prompt_lengths"].detach().to("cpu", dtype=torch.long)
        if input_rows.shape[0] != len(seq_ids) or prompt_lengths.numel() != len(seq_ids):
            raise ValueError("generate_batch_chain input_ids/prompt_lengths size mismatch")

        total_chain_hops = 0
        rows: list[Dict[str, Any]] = []
        decode_steps = 0
        total_decode_chunks = 0
        max_decode_chunks_per_step = 0

        for idx, seq_id in enumerate(seq_ids):
            prompt_len = int(prompt_lengths[idx].item())
            if prompt_len <= 0:
                raise ValueError("generate_batch_chain prompt length must be positive")
            input_ids = input_rows[idx : idx + 1, :prompt_len].contiguous()
            positions = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
            with self.lock:
                parsed = self._parse_binary_inputs(
                    {"seq_id": seq_id},
                    {"positions": positions, "input_ids": input_ids},
                )
                result = self.stage.prefill(**parsed)
                self.prefill_count += 1
            out_meta, _ = self._forward_hidden_ttp(
                op="prefill",
                meta={"seq_id": seq_id, "next_stages": route},
                positions=positions,
                result=result,
            )
            total_chain_hops += int(out_meta.get("_chain_hops", 0))
            if "next_token" not in out_meta:
                raise RuntimeError("last shard did not return next_token during remote prefill")
            rows.append(
                {
                    "seq_id": seq_id,
                    "prompt_len": prompt_len,
                    "generated": [],
                    "cur_token": int(out_meta["next_token"]),
                    "done": False,
                }
            )

        while True:
            ready = [
                row
                for row in rows
                if not row["done"] and len(row["generated"]) < max_new_tokens
            ]
            if not ready:
                break
            decode_steps += 1

            decode_ready = []
            for row in ready:
                cur_token = int(row["cur_token"])
                row["generated"].append(cur_token)
                if cur_token in eos_ids or len(row["generated"]) >= max_new_tokens:
                    row["done"] = True
                else:
                    decode_ready.append(row)

            decode_chunks = []
            for start in range(0, len(decode_ready), microbatch_size):
                chunk = decode_ready[start : start + microbatch_size]
                chunk_seq_ids = [int(row["seq_id"]) for row in chunk]
                chunk_tokens = [[int(row["cur_token"])] for row in chunk]
                chunk_positions = [
                    [int(row["prompt_len"]) + len(row["generated"]) - 1]
                    for row in chunk
                ]
                decode_chunks.append(
                    {
                        "rows": chunk,
                        "seq_ids": chunk_seq_ids,
                        "input_ids": torch.tensor(chunk_tokens, dtype=torch.long),
                        "positions": torch.tensor(chunk_positions, dtype=torch.long),
                    }
                )

            total_decode_chunks += len(decode_chunks)
            max_decode_chunks_per_step = max(max_decode_chunks_per_step, len(decode_chunks))
            for chunk_data in decode_chunks:
                with self.lock:
                    parsed = self._parse_binary_batch_inputs(
                        {"seq_ids": chunk_data["seq_ids"]},
                        {
                            "positions": chunk_data["positions"],
                            "input_ids": chunk_data["input_ids"],
                        },
                    )
                    result = self.stage.decode_batch(**parsed)
                    self.decode_count += len(parsed["seq_ids"])
                out_meta, _ = self._forward_hidden_ttp(
                    op="decode_batch",
                    meta={
                        "seq_ids": chunk_data["seq_ids"],
                        "next_stages": route,
                    },
                    positions=chunk_data["positions"],
                    result=result,
                )
                total_chain_hops += int(out_meta.get("_chain_hops", 0))
                next_tokens = out_meta.get("next_tokens")
                if not isinstance(next_tokens, list):
                    raise RuntimeError("last shard did not return next_tokens during remote decode")
                for row, token in zip(chunk_data["rows"], next_tokens):
                    row["cur_token"] = int(token)

        generated = [[int(token) for token in row["generated"]] for row in rows]
        return encode_tensor_frame_parts(
            {
                "ok": True,
                "generated": generated,
                "generated_tokens": sum(len(tokens) for tokens in generated),
                "decode_steps": decode_steps,
                "total_decode_chunks": total_decode_chunks,
                "max_decode_chunks_per_step": max_decode_chunks_per_step,
                "remote_chain_loop": True,
                "chain_forwards": total_chain_hops,
            }
        )

    def generate_continuous_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        """Run a queue-style continuous batching loop from the first TTP stage.

        This is the MegaMesh shard analogue of the engine scheduler: requests
        move waiting -> running -> completed, decode operates on the live
        running set, and newly admitted requests reuse freed sequence slots.
        """

        seq_ids = [int(seq_id) for seq_id in meta.get("seq_ids", [])]
        if not seq_ids:
            raise ValueError("generate_continuous_chain requires meta.seq_ids")
        route = self._route_from_meta(meta)
        if not route:
            raise ValueError("generate_continuous_chain requires meta.next_stage(s)")

        max_new_tokens = max(0, int(meta.get("max_new_tokens", 64)))
        microbatch_size = max(1, int(meta.get("microbatch_size", len(seq_ids))))
        max_batch_size = max(1, int(meta.get("max_batch_size", microbatch_size)))
        eos_ids = {int(token) for token in meta.get("eos_ids", [])}
        input_rows = tensors["input_ids"].detach().to("cpu", dtype=torch.long)
        prompt_lengths = tensors["prompt_lengths"].detach().to("cpu", dtype=torch.long)
        if input_rows.shape[0] != len(seq_ids) or prompt_lengths.numel() != len(seq_ids):
            raise ValueError("generate_continuous_chain input_ids/prompt_lengths size mismatch")

        rows: list[Dict[str, Any]] = []
        for idx, seq_id in enumerate(seq_ids):
            prompt_len = int(prompt_lengths[idx].item())
            if prompt_len <= 0:
                raise ValueError("generate_continuous_chain prompt length must be positive")
            rows.append(
                {
                    "seq_id": seq_id,
                    "request_index": idx,
                    "prompt_len": prompt_len,
                    "input_ids": input_rows[idx : idx + 1, :prompt_len].contiguous(),
                    "generated": [],
                    "cur_token": None,
                    "done": False,
                    "admitted": False,
                }
            )

        waiting = list(rows)
        running: list[Dict[str, Any]] = []
        completed: list[Dict[str, Any]] = []
        total_chain_hops = 0
        decode_steps = 0
        scheduler_steps = 0
        total_decode_chunks = 0
        max_decode_chunks_per_step = 0
        total_prefills = 0
        max_running = 0
        admission_events = 0

        def admit_available() -> None:
            nonlocal admission_events, total_chain_hops, total_prefills, max_running
            admitted = 0
            while waiting and len(running) < max_batch_size:
                row = waiting.pop(0)
                seq_id = int(row["seq_id"])
                prompt_len = int(row["prompt_len"])
                positions = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
                with self.lock:
                    parsed = self._parse_binary_inputs(
                        {"seq_id": seq_id},
                        {
                            "positions": positions,
                            "input_ids": row["input_ids"],
                        },
                    )
                    result = self.stage.prefill(**parsed)
                    self.prefill_count += 1
                out_meta, _ = self._forward_hidden_ttp(
                    op="prefill",
                    meta={"seq_id": seq_id, "next_stages": route},
                    positions=positions,
                    result=result,
                )
                total_chain_hops += int(out_meta.get("_chain_hops", 0))
                if "next_token" not in out_meta:
                    raise RuntimeError("last shard did not return next_token during continuous prefill")
                row["cur_token"] = int(out_meta["next_token"])
                row["admitted"] = True
                if max_new_tokens <= 0:
                    row["done"] = True
                    completed.append(row)
                else:
                    running.append(row)
                admitted += 1
                total_prefills += 1
            if admitted:
                admission_events += 1
                max_running = max(max_running, len(running))

        while waiting or running:
            scheduler_steps += 1
            admit_available()

            ready = [
                row
                for row in running
                if not row["done"] and len(row["generated"]) < max_new_tokens
            ]
            if not ready:
                continue

            decode_steps += 1
            decode_ready = []
            for row in ready:
                cur_token = int(row["cur_token"])
                row["generated"].append(cur_token)
                if cur_token in eos_ids or len(row["generated"]) >= max_new_tokens:
                    row["done"] = True
                else:
                    decode_ready.append(row)

            still_running = []
            for row in running:
                if row["done"]:
                    completed.append(row)
                else:
                    still_running.append(row)
            running = still_running

            decode_chunks = []
            for start in range(0, len(decode_ready), microbatch_size):
                chunk = decode_ready[start : start + microbatch_size]
                chunk_seq_ids = [int(row["seq_id"]) for row in chunk]
                chunk_tokens = [[int(row["cur_token"])] for row in chunk]
                chunk_positions = [
                    [int(row["prompt_len"]) + len(row["generated"]) - 1]
                    for row in chunk
                ]
                decode_chunks.append(
                    {
                        "rows": chunk,
                        "seq_ids": chunk_seq_ids,
                        "input_ids": torch.tensor(chunk_tokens, dtype=torch.long),
                        "positions": torch.tensor(chunk_positions, dtype=torch.long),
                    }
                )

            total_decode_chunks += len(decode_chunks)
            max_decode_chunks_per_step = max(max_decode_chunks_per_step, len(decode_chunks))
            for chunk_data in decode_chunks:
                with self.lock:
                    parsed = self._parse_binary_batch_inputs(
                        {"seq_ids": chunk_data["seq_ids"]},
                        {
                            "positions": chunk_data["positions"],
                            "input_ids": chunk_data["input_ids"],
                        },
                    )
                    result = self.stage.decode_batch(**parsed)
                    self.decode_count += len(parsed["seq_ids"])
                out_meta, _ = self._forward_hidden_ttp(
                    op="decode_batch",
                    meta={
                        "seq_ids": chunk_data["seq_ids"],
                        "next_stages": route,
                    },
                    positions=chunk_data["positions"],
                    result=result,
                )
                total_chain_hops += int(out_meta.get("_chain_hops", 0))
                next_tokens = out_meta.get("next_tokens")
                if not isinstance(next_tokens, list):
                    raise RuntimeError("last shard did not return next_tokens during continuous decode")
                for row, token in zip(chunk_data["rows"], next_tokens):
                    row["cur_token"] = int(token)

        completed.extend(row for row in running if row not in completed)
        generated_by_index = [None] * len(rows)
        for row in rows:
            generated_by_index[int(row["request_index"])] = [
                int(token) for token in row["generated"]
            ]
        generated = [tokens if tokens is not None else [] for tokens in generated_by_index]

        return encode_tensor_frame_parts(
            {
                "ok": True,
                "generated": generated,
                "generated_tokens": sum(len(tokens) for tokens in generated),
                "continuous_batching": True,
                "scheduler_steps": scheduler_steps,
                "admission_events": admission_events,
                "total_prefills": total_prefills,
                "max_running": max_running,
                "decode_steps": decode_steps,
                "total_decode_chunks": total_decode_chunks,
                "max_decode_chunks_per_step": max_decode_chunks_per_step,
                "remote_chain_loop": True,
                "chain_forwards": total_chain_hops,
            }
        )

    def prefill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            parsed = self._parse_inputs(payload)
            result = self.stage.prefill(**parsed)
            self.prefill_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_result(result)

    def decode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            parsed = self._parse_inputs(payload)
            result = self.stage.decode(**parsed)
            self.decode_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_result(result)

    def prefill_binary(self, meta: Dict[str, Any], tensors: Dict[str, torch.Tensor]) -> bytes:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.prefill(**parsed)
            self.prefill_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_binary_result(result)

    def decode_binary(self, meta: Dict[str, Any], tensors: Dict[str, torch.Tensor]) -> bytes:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.decode(**parsed)
            self.decode_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_binary_result(result)

    def decode_batch_binary(self, meta: Dict[str, Any], tensors: Dict[str, torch.Tensor]) -> bytes:
        with self.lock:
            parsed = self._parse_binary_batch_inputs(meta, tensors)
            result = self.stage.decode_batch(**parsed)
            self.decode_count += len(parsed["seq_ids"])
        result = self._finish_lm_head_result(result)
        return self._format_binary_result(result)

    def prefill_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.prefill(**parsed)
            self.prefill_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_ttp_result(result)

    def prefill_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.prefill(**parsed)
            self.prefill_count += 1
        result = self._finish_lm_head_result(result)
        return self._forward_ttp_result(
            op="prefill",
            meta=meta,
            positions=tensors["positions"],
            result=result,
        )

    def decode_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.decode(**parsed)
            self.decode_count += 1
        result = self._finish_lm_head_result(result)
        return self._format_ttp_result(result)

    def decode_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_inputs(meta, tensors)
            result = self.stage.decode(**parsed)
            self.decode_count += 1
        result = self._finish_lm_head_result(result)
        return self._forward_ttp_result(
            op="decode",
            meta=meta,
            positions=tensors["positions"],
            result=result,
        )

    def decode_batch_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_batch_inputs(meta, tensors)
            result = self.stage.decode_batch(**parsed)
            self.decode_count += len(parsed["seq_ids"])
        result = self._finish_lm_head_result(result)
        return self._format_ttp_result(result)

    def decode_batch_chain_ttp(
        self,
        meta: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
    ) -> TensorFrameParts:
        with self.lock:
            parsed = self._parse_binary_batch_inputs(meta, tensors)
            result = self.stage.decode_batch(**parsed)
            self.decode_count += len(parsed["seq_ids"])
        result = self._finish_lm_head_result(result)
        return self._forward_ttp_result(
            op="decode_batch",
            meta=meta,
            positions=tensors["positions"],
            result=result,
        )

    def free(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        seq_id = int(payload.get("seq_id", 1))
        with self.lock:
            self.stage.free_sequence(seq_id)
        return {"ok": True, "seq_id": seq_id}


def _handler_for(worker: MegaMeshShardWorker):
    class MegaMeshShardHandler(BaseHTTPRequestHandler):
        server_version = "MegaMeshShardHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[MegaMeshShard:{worker.name}] {self.address_string()} - {fmt % args}")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_binary(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length) if length else b""

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(parsed, dict):
                raise ValueError("request body must be a JSON object")
            return parsed

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/health", "/metrics"):
                self._send_json(200, worker.health())
            else:
                self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            try:
                path = self.path.rstrip("/")
                if path == "/prefill.bin":
                    meta, tensors = decode_tensor_frame(
                        self._read_body(),
                        device="cpu",
                    )
                    self._send_binary(200, worker.prefill_binary(meta, tensors))
                elif path == "/decode.bin":
                    meta, tensors = decode_tensor_frame(
                        self._read_body(),
                        device="cpu",
                    )
                    self._send_binary(200, worker.decode_binary(meta, tensors))
                elif path == "/decode_batch.bin":
                    meta, tensors = decode_tensor_frame(
                        self._read_body(),
                        device="cpu",
                    )
                    self._send_binary(200, worker.decode_batch_binary(meta, tensors))
                elif path == "/prefill":
                    payload = self._read_json()
                    self._send_json(200, worker.prefill(payload))
                elif path == "/decode":
                    payload = self._read_json()
                    self._send_json(200, worker.decode(payload))
                elif path == "/free":
                    payload = self._read_json()
                    self._send_json(200, worker.free(payload))
                else:
                    self._send_json(404, {"ok": False, "error": "not_found"})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

    return MegaMeshShardHandler


def _lm_head_handler_for(worker: MegaMeshLMHeadWorker):
    class MegaMeshLMHeadHandler(BaseHTTPRequestHandler):
        server_version = "MegaMeshLMHeadHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[MegaMeshLMHead:{worker.name}] {self.address_string()} - {fmt % args}")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/health", "/metrics"):
                self._send_json(200, worker.health())
            else:
                self._send_json(404, {"ok": False, "error": "not_found"})

    return MegaMeshLMHeadHandler


def _mlp_handler_for(worker: MegaMeshMLPShardWorker):
    class MegaMeshMLPHandler(BaseHTTPRequestHandler):
        server_version = "MegaMeshMLPHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[MegaMeshMLP:{worker.name}] {self.address_string()} - {fmt % args}")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/health", "/metrics"):
                self._send_json(200, worker.health())
            else:
                self._send_json(404, {"ok": False, "error": "not_found"})

    return MegaMeshMLPHandler


def run_lm_head_worker(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8099,
    ttp_port: int = 9099,
    vocab_start: int,
    vocab_end: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    name: str = "",
) -> None:
    worker = MegaMeshLMHeadWorker(
        model,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        dtype=dtype,
        device=device,
        cache_dir=cache_dir,
        name=name,
    )
    server = ThreadingHTTPServer((host, int(port)), _lm_head_handler_for(worker))
    ttp_server = None
    if int(ttp_port) > 0:
        from .ttp import TTPShardServer

        ttp_server = TTPShardServer((host, int(ttp_port)), worker)
        ttp_thread = threading.Thread(target=ttp_server.serve_forever, daemon=True)
        ttp_thread.start()
        print(f"[MegaMeshLMHeadTTP] {worker.name} ready on ttp://{host}:{ttp_port}")
    print(
        f"[MegaMeshLMHead] {worker.name} ready on http://{host}:{port} "
        f"vocab={vocab_start}:{vocab_end} device={device} dtype={_dtype_name(dtype)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MegaMeshLMHead] shutting down")
    finally:
        if ttp_server is not None:
            ttp_server.shutdown()
            ttp_server.server_close()
        server.server_close()


def run_mlp_worker(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8098,
    ttp_port: int = 9098,
    layer_start: int,
    layer_end: int,
    intermediate_start: int,
    intermediate_end: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    name: str = "",
) -> None:
    worker = MegaMeshMLPShardWorker(
        model,
        layer_start=layer_start,
        layer_end=layer_end,
        intermediate_start=intermediate_start,
        intermediate_end=intermediate_end,
        dtype=dtype,
        device=device,
        cache_dir=cache_dir,
        name=name,
    )
    server = ThreadingHTTPServer((host, int(port)), _mlp_handler_for(worker))
    ttp_server = None
    if int(ttp_port) > 0:
        from .ttp import TTPShardServer

        ttp_server = TTPShardServer((host, int(ttp_port)), worker)
        ttp_thread = threading.Thread(target=ttp_server.serve_forever, daemon=True)
        ttp_thread.start()
        print(f"[MegaMeshMLPTTP] {worker.name} ready on ttp://{host}:{ttp_port}")
    print(
        f"[MegaMeshMLP] {worker.name} ready on http://{host}:{port} "
        f"layers={layer_start}:{layer_end} intermediate={intermediate_start}:{intermediate_end} "
        f"device={device} dtype={_dtype_name(dtype)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MegaMeshMLP] shutting down")
    finally:
        if ttp_server is not None:
            ttp_server.shutdown()
            ttp_server.server_close()
        server.server_close()


def run_shard_worker(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8090,
    layer_start: int,
    layer_end: int,
    is_first: bool = False,
    is_last: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    num_blocks: int = 512,
    block_size: int = 16,
    max_seq_len: int = 512,
    cache_dir: Optional[str] = None,
    name: str = "",
    ttp_port: int = 0,
    ttp_pinned: bool = True,
    lm_head_shards: str | list[str] | tuple[str, ...] | None = None,
    mlp_shards: str | list[str] | tuple[str, ...] | None = None,
) -> None:
    worker = MegaMeshShardWorker(
        model,
        layer_start=layer_start,
        layer_end=layer_end,
        is_first=is_first,
        is_last=is_last,
        dtype=dtype,
        device=device,
        num_blocks=num_blocks,
        block_size=block_size,
        max_seq_len=max_seq_len,
        cache_dir=cache_dir,
        name=name,
        ttp_pinned=ttp_pinned,
        lm_head_shards=lm_head_shards,
        mlp_shards=mlp_shards,
    )
    server = ThreadingHTTPServer((host, int(port)), _handler_for(worker))
    ttp_server = None
    if int(ttp_port) > 0:
        from .ttp import TTPShardServer

        ttp_server = TTPShardServer((host, int(ttp_port)), worker)
        ttp_thread = threading.Thread(target=ttp_server.serve_forever, daemon=True)
        ttp_thread.start()
        print(f"[MegaMeshTTP] {worker.name} ready on ttp://{host}:{ttp_port}")
    print(
        f"[MegaMeshShard] {worker.name} ready on http://{host}:{port} "
        f"layers={layer_start}:{layer_end} first={is_first} last={is_last} "
        f"device={device} dtype={_dtype_name(dtype)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MegaMeshShard] shutting down")
    finally:
        if ttp_server is not None:
            ttp_server.shutdown()
            ttp_server.server_close()
        server.server_close()


__all__ = [
    "MegaMeshLMHeadWorker",
    "MegaMeshMLPShardWorker",
    "MegaMeshShardWorker",
    "run_lm_head_worker",
    "run_mlp_worker",
    "run_shard_worker",
]
