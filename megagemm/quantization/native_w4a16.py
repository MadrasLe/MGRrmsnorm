"""Standalone MGX W4A16 quantization, including a native 2:4 layout.

This format is deliberately independent from AWQ/Marlin. We use symmetric
signed INT4 weights with one FP16 scale per output channel and input group.

* dense: qweight [N, K / 2], two signed INT4 values per byte
* 2:4:   qweight [N, K / 4], two retained INT4 values per quartet
         metadata [N, K / 8], two position codes per byte
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NATIVE_W4A16_FORMAT",
    "NATIVE_W4A16_SPARSE24_FORMAT",
    "NativeW4A16Linear",
    "dequantize_native_w4a16",
    "is_native_w4a16_config",
    "native_w4a16_manifest_config",
    "native_w4a16_runtime_stats",
    "quantize_model_native_w4a16",
    "quantize_native_w4a16",
]

NATIVE_W4A16_FORMAT = "mgx-native-w4a16-v1"
NATIVE_W4A16_SPARSE24_FORMAT = "mgx-native-w4a16-sparse24-v1"
NATIVE_W4A16_LAYOUT_DENSE = "output-major-signed-int4-pairs-v1"
NATIVE_W4A16_LAYOUT_SPARSE24 = "output-major-signed-int4-2of4-v1"
DEFAULT_GROUP_SIZE = 128

_VALID_KERNEL_MODES = {"auto", "triton", "torch"}
_FALLBACK_WARNED = False


def _kernel_mode() -> str:
    mode = os.environ.get("MEGAGEMM_NATIVE_W4A16_KERNEL", "auto").strip().lower()
    if mode not in _VALID_KERNEL_MODES:
        warnings.warn(
            f"Ignoring invalid MEGAGEMM_NATIVE_W4A16_KERNEL={mode!r}; "
            "expected auto, triton, or torch.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "auto"
    return mode


def _signed_nibbles(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int16) & 0xF
    return torch.where(values < 8, values, values - 16)


def _validate_shape(weight: torch.Tensor, group_size: int) -> tuple[int, int]:
    if weight.ndim != 2:
        raise ValueError(f"Native W4A16 expects a 2D weight, got {tuple(weight.shape)}")
    out_features, in_features = (int(weight.shape[0]), int(weight.shape[1]))
    if group_size <= 0 or group_size % 8:
        raise ValueError("Native W4A16 group_size must be a positive multiple of 8")
    if in_features % group_size:
        raise ValueError(
            f"Native W4A16 in_features={in_features} is not divisible by group_size={group_size}"
        )
    return out_features, in_features


@torch.no_grad()
def quantize_native_w4a16(
    weight: torch.Tensor,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    sparse24: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight into the standalone MGX W4A16 layout."""
    out_features, in_features = _validate_shape(weight, group_size)
    work = weight.detach().to(torch.float32)

    positions: Optional[torch.Tensor] = None
    if sparse24:
        quartets = work.reshape(out_features, in_features // 4, 4)
        positions = torch.topk(quartets.abs(), k=2, dim=-1, largest=True, sorted=False).indices
        positions = positions.sort(dim=-1).values
        retained = torch.gather(quartets, -1, positions)
        pruned_quartets = torch.zeros_like(quartets)
        pruned_quartets.scatter_(-1, positions, retained)
        work = pruned_quartets.reshape(out_features, in_features)

    groups = work.reshape(out_features, in_features // group_size, group_size)
    max_abs = groups.abs().amax(dim=-1)
    scales_fp32 = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    quantized = torch.round(groups / scales_fp32.unsqueeze(-1)).clamp_(-7, 7)
    quantized = quantized.to(torch.int16).reshape(out_features, in_features)
    scales = scales_fp32.to(torch.float16).contiguous()

    if not sparse24:
        low = quantized[:, 0::2] & 0xF
        high = quantized[:, 1::2] & 0xF
        qweight = (low | (high << 4)).to(torch.uint8).contiguous()
        metadata = torch.empty((out_features, 0), dtype=torch.uint8, device=weight.device)
        return qweight, scales, metadata

    assert positions is not None
    quantized_quartets = quantized.reshape(out_features, in_features // 4, 4)
    retained_q = torch.gather(quantized_quartets, -1, positions)
    qweight = (
        (retained_q[..., 0] & 0xF) | ((retained_q[..., 1] & 0xF) << 4)
    ).to(torch.uint8).contiguous()

    pos0 = positions[..., 0]
    pos1 = positions[..., 1]
    # Pair codes: 01->0, 02->1, 03->2, 12->3, 13->4, 23->5.
    codes = torch.where(
        pos0 == 0,
        pos1 - 1,
        torch.where(pos0 == 1, pos1 + 1, torch.full_like(pos0, 5)),
    ).to(torch.uint8)
    metadata = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return qweight, scales, metadata


def dequantize_native_w4a16(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    metadata: Optional[torch.Tensor],
    *,
    in_features: int,
    group_size: int,
    sparse24: bool,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Portable PyTorch unpacker used for validation and the CPU fallback."""
    out_features = int(qweight.shape[0])
    if in_features % group_size or in_features % 8:
        raise ValueError("Native W4A16 in_features must be divisible by group_size and 8")
    if tuple(scales.shape) != (out_features, in_features // group_size):
        raise ValueError(
            "Native W4A16 scale shape mismatch: "
            f"expected {(out_features, in_features // group_size)}, got {tuple(scales.shape)}"
        )

    if not sparse24:
        if tuple(qweight.shape) != (out_features, in_features // 2):
            raise ValueError("Invalid dense native W4A16 qweight shape")
        low = _signed_nibbles(qweight & 0xF)
        high = _signed_nibbles(qweight >> 4)
        unpacked = torch.stack((low, high), dim=-1).reshape(out_features, in_features)
    else:
        if metadata is None:
            raise ValueError("Sparse native W4A16 requires metadata")
        if tuple(qweight.shape) != (out_features, in_features // 4):
            raise ValueError("Invalid sparse native W4A16 qweight shape")
        if tuple(metadata.shape) != (out_features, in_features // 8):
            raise ValueError("Invalid sparse native W4A16 metadata shape")

        group_count = in_features // 4
        codes = torch.stack((metadata & 0xF, metadata >> 4), dim=-1).reshape(
            out_features, group_count
        )
        if bool((codes > 5).any()):
            raise ValueError("Sparse native W4A16 metadata contains a reserved position code")
        pair_table = torch.tensor(
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
            dtype=torch.long,
            device=qweight.device,
        )
        positions = pair_table[codes.long()]
        values = torch.stack(
            (_signed_nibbles(qweight & 0xF), _signed_nibbles(qweight >> 4)), dim=-1
        )
        quartets = torch.zeros(
            (out_features, group_count, 4), dtype=torch.int16, device=qweight.device
        )
        quartets.scatter_(-1, positions, values)
        unpacked = quartets.reshape(out_features, in_features)

    expanded_scales = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    return (unpacked.to(torch.float32) * expanded_scales).to(dtype)


class NativeW4A16Linear(nn.Module):
    """Standalone symmetric W4A16 linear with optional packed 2:4 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        bias: bool = False,
        sparse24: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if in_features % group_size or in_features % 8:
            raise ValueError(
                f"Native W4A16 requires in_features divisible by {group_size} and 8; "
                f"got {in_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.sparse24 = bool(sparse24)
        packed_k = self.in_features // (4 if self.sparse24 else 2)
        self.register_buffer(
            "qweight",
            torch.empty((self.out_features, packed_k), dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (self.out_features, self.in_features // self.group_size),
                dtype=torch.float16,
                device=device,
            ),
        )
        metadata_k = self.in_features // 8 if self.sparse24 else 0
        self.register_buffer(
            "metadata",
            torch.empty((self.out_features, metadata_k), dtype=torch.uint8, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        sparse24: bool = False,
    ) -> "NativeW4A16Linear":
        result = cls(
            linear.in_features,
            linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
            sparse24=sparse24,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        qweight, scales, metadata = quantize_native_w4a16(
            linear.weight, group_size=group_size, sparse24=sparse24
        )
        result.qweight = qweight
        result.scales = scales
        result.metadata = metadata
        if linear.bias is not None:
            result.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        result.train(linear.training)
        return result

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return dequantize_native_w4a16(
            self.qweight,
            self.scales,
            self.metadata,
            in_features=self.in_features,
            group_size=self.group_size,
            sparse24=self.sparse24,
            dtype=dtype or self.scales.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _FALLBACK_WARNED
        mode = _kernel_mode()
        if mode != "torch" and x.is_cuda:
            try:
                from ..kernels.native_w4a16 import native_w4a16_linear

                result = native_w4a16_linear(
                    x,
                    self.qweight,
                    self.scales,
                    self.metadata,
                    self.bias,
                    group_size=self.group_size,
                    sparse24=self.sparse24,
                )
                if result is not None:
                    return result
                if mode == "triton":
                    from ..kernels.native_w4a16 import get_native_w4a16_kernel_stats

                    raise RuntimeError(
                        "MGX native W4A16 Triton kernel was unavailable or rejected this shape; "
                        f"kernel_stats={get_native_w4a16_kernel_stats()}"
                    )
            except Exception as exc:
                if mode == "triton":
                    raise
                if not _FALLBACK_WARNED:
                    warnings.warn(
                        f"MGX native W4A16 Triton kernel failed; using PyTorch fallback: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _FALLBACK_WARNED = True
        elif mode == "triton" and not x.is_cuda:
            raise RuntimeError("MEGAGEMM_NATIVE_W4A16_KERNEL=triton requires CUDA tensors")

        weight = self.dequantize(dtype=x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, sparse24={self.sparse24}, "
            f"bias={self.bias is not None}"
        )


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, field = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, field, module)


@torch.no_grad()
def quantize_model_native_w4a16(
    model: nn.Module,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    sparse24: bool = False,
    exclude: tuple[str, ...] = ("lm_head",),
) -> dict[str, Any]:
    """Replace eligible nn.Linear modules with native MGX W4A16 modules."""
    candidates = list(model.named_modules())
    replaced: list[str] = []
    skipped: list[str] = []
    original_bytes = 0
    packed_bytes = 0
    for name, module in candidates:
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(name == item or name.startswith(item + ".") for item in exclude):
            skipped.append(name)
            continue
        if module.in_features % group_size or module.in_features % 8:
            skipped.append(name)
            continue
        quantized = NativeW4A16Linear.from_linear(
            module, group_size=group_size, sparse24=sparse24
        )
        original_bytes += int(module.weight.numel() * module.weight.element_size())
        packed_bytes += int(
            quantized.qweight.numel() * quantized.qweight.element_size()
            + quantized.metadata.numel() * quantized.metadata.element_size()
            + quantized.scales.numel() * quantized.scales.element_size()
        )
        _set_submodule(model, name, quantized)
        replaced.append(name)

    if not replaced:
        raise ValueError("No eligible nn.Linear modules were found for native W4A16 quantization")
    return {
        "format": NATIVE_W4A16_FORMAT,
        "sparse24": bool(sparse24),
        "group_size": int(group_size),
        "module_count": len(replaced),
        "skipped_module_count": len(skipped),
        "module_names": replaced,
        "original_weight_bytes": original_bytes,
        "packed_weight_bytes": packed_bytes,
        "storage_ratio": packed_bytes / max(1, original_bytes),
    }


def is_native_w4a16_config(config: Optional[dict[str, Any]]) -> bool:
    return bool(config and config.get("format") == NATIVE_W4A16_FORMAT)


def native_w4a16_manifest_config(export_meta: dict[str, Any]) -> dict[str, Any]:
    sparse24 = bool(export_meta.get("native_w4a16_sparse24", False))
    return {
        "quant_method": "mgx-native",
        "format": NATIVE_W4A16_FORMAT,
        "bits": 4,
        "symmetric": True,
        "group_size": int(export_meta.get("native_w4a16_group_size", DEFAULT_GROUP_SIZE)),
        "activation_dtype": "float16",
        "weight_layout": (
            NATIVE_W4A16_LAYOUT_SPARSE24 if sparse24 else NATIVE_W4A16_LAYOUT_DENSE
        ),
        "sparse24": sparse24,
        "sparsity_format": NATIVE_W4A16_SPARSE24_FORMAT if sparse24 else None,
        "module_count": int(export_meta.get("native_w4a16_module_count", 0)),
        "original_weight_bytes": int(export_meta.get("native_w4a16_original_weight_bytes", 0)),
        "packed_weight_bytes": int(export_meta.get("native_w4a16_packed_weight_bytes", 0)),
    }


def native_w4a16_runtime_stats(model: Optional[nn.Module] = None) -> dict[str, Any]:
    modules = [] if model is None else [
        module for module in model.modules() if isinstance(module, NativeW4A16Linear)
    ]
    result: dict[str, Any] = {
        "active": bool(modules),
        "module_count": len(modules),
        "sparse24_module_count": sum(int(module.sparse24) for module in modules),
        "format": NATIVE_W4A16_FORMAT if modules else "none",
        "kernel_mode": _kernel_mode(),
    }
    try:
        from ..kernels.native_w4a16 import get_native_w4a16_kernel_stats

        result.update(get_native_w4a16_kernel_stats())
    except Exception:
        pass
    return result
