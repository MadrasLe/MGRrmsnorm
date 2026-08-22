"""Standalone FP16 2:4 Sparse Tensor Core dispatch.

This module is intentionally independent of Marlin, TorchAO and cuSPARSELt.
The optional ``sparse24_cuda_ops`` extension is built with MegaGemm and issues
``mma.sp::ordered_metadata.sync.aligned.m16n8k32`` directly on SM80-SM89 GPUs
with CUDA 12.5+, retaining the legacy spelling for older toolkits.  The public
helpers remain importable when the extension was not built, allowing MGX to
retain its existing PyTorch/Triton fallbacks.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import sparse24_cuda_ops as _sparse24_cuda_ops

    _HAS_NATIVE_MMA = True
    _NATIVE_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the local CUDA build
    _sparse24_cuda_ops = None
    _HAS_NATIVE_MMA = False
    _NATIVE_IMPORT_ERROR = str(exc)


_PORTABLE_TO_PTX = torch.tensor([0x4, 0x8, 0xC, 0x9, 0xD, 0xE], dtype=torch.uint8)
_RUNTIME_STATS = {
    "available": bool(_HAS_NATIVE_MMA),
    "import_error": _NATIVE_IMPORT_ERROR,
    "hits": 0,
    "failures": 0,
}


def sparse24_portable_metadata_to_ptx(metadata: torch.Tensor) -> torch.Tensor:
    """Convert MGX's compact combination codes to ordered ``mma.sp`` nibbles.

    MGX stores the six combinations as codes 0..5.  PTX stores the two retained
    positions directly as ``position0 | position1 << 2``, whose meaningful
    ordered values are 4, 8, 12, 9, 13 and 14.
    """
    if metadata.dtype != torch.uint8 or metadata.ndim != 2:
        raise ValueError("portable 2:4 metadata must be a 2-D uint8 tensor")
    low = metadata & 0x0F
    high = (metadata >> 4) & 0x0F
    if bool(((low > 5) | (high > 5)).any().item()):
        raise ValueError("portable 2:4 metadata contains an invalid position code")
    lookup = _PORTABLE_TO_PTX.to(device=metadata.device)
    return (lookup[low.long()] | (lookup[high.long()] << 4)).contiguous()


def sparse24_mma_available() -> bool:
    return bool(_HAS_NATIVE_MMA)


def sparse24_mma_import_error() -> str:
    return str(_NATIVE_IMPORT_ERROR)


def sparse24_mma_eligible(
    x: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
) -> bool:
    if not _HAS_NATIVE_MMA or (torch.is_grad_enabled() and x.requires_grad):
        return False
    if not (x.is_cuda and values.is_cuda and metadata.is_cuda):
        return False
    if x.dtype != torch.float16 or values.dtype != torch.float16:
        return False
    if metadata.dtype != torch.uint8 or values.ndim != 2 or metadata.ndim != 2:
        return False
    if x.ndim < 2 or not values.is_contiguous() or not metadata.is_contiguous():
        return False
    if not x.is_contiguous():
        return False
    capability = torch.cuda.get_device_capability(x.device)
    if int(capability[0]) != 8:
        return False
    m_rows = int(x.numel() // max(1, int(x.shape[-1])))
    n_rows, packed_k = int(values.shape[0]), int(values.shape[1])
    k_cols = int(x.shape[-1])
    return bool(
        m_rows > 0
        and n_rows > 0
        and n_rows % 64 == 0
        and k_cols > 0
        and k_cols % 64 == 0
        and packed_k == k_cols // 2
        and tuple(metadata.shape) == (n_rows, k_cols // 8)
    )


def sparse24_mma_linear(
    x: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ W.T`` through MegaGemm's direct FP16 ``mma.sp`` kernel."""
    if not sparse24_mma_eligible(x, values, metadata):
        raise ValueError("invocation is not eligible for the standalone FP16 mma.sp kernel")
    original_shape = tuple(x.shape)
    x_2d = x if x.ndim == 2 else x.flatten(0, -2)
    n_rows = int(values.shape[0])
    expected_shape = (*original_shape[:-1], n_rows)
    out_2d = None
    if out is not None:
        if tuple(out.shape) != expected_shape or out.device != x.device or out.dtype != x.dtype:
            raise ValueError(
                f"out must have shape/device/dtype {expected_shape}/{x.device}/{x.dtype}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        out_2d = out if out.ndim == 2 else out.flatten(0, -2)

    result = sparse24_mma_linear_unchecked(
        x_2d,
        values,
        metadata,
        bias,
        out=out_2d,
    )
    if out is not None:
        return out
    return result.view(expected_shape)


def sparse24_mma_linear_unchecked(
    x_2d: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-overhead 2-D dispatch for MGX's already-validated decode hot path."""
    try:
        result = _sparse24_cuda_ops.linear(x_2d, values, metadata, bias, out)
        _RUNTIME_STATS["hits"] += 1
        return result
    except Exception:
        _RUNTIME_STATS["failures"] += 1
        raise


def sparse24_mma_runtime_stats() -> dict[str, object]:
    return dict(_RUNTIME_STATS)


__all__ = [
    "sparse24_mma_available",
    "sparse24_mma_eligible",
    "sparse24_mma_import_error",
    "sparse24_mma_linear",
    "sparse24_mma_linear_unchecked",
    "sparse24_mma_runtime_stats",
    "sparse24_portable_metadata_to_ptx",
]
