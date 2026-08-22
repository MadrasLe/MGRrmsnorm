"""Decode-oriented Triton GEMV for CUTLASS 2:4 semi-structured weights.

PyTorch's semi-structured tensor backend is a good fit for compute-bound GEMM,
but a decoder normally presents only a handful of rows to each projection.  In
that regime the library setup/dispatch cost can dominate the useful work.  This
module consumes the already-compressed CUTLASS ``values`` and ``meta`` tensors
directly and performs a compact, memory-bound GEMV without materialising the
zeros.

The metadata addressing below is the inverse of CUTLASS' host-side 2:4 metadata
reordering (``ColumnMajorInterleaved<2>``).  It intentionally supports only the
FP16/BF16 CUTLASS layout; callers must retain the regular PyTorch sparse linear
as a fallback for every unsupported shape or execution mode.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised on CPU/Windows installations
    triton = None
    tl = None
    _HAS_TRITON = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


_TRITON_MAX_ROWS = max(1, _env_int("MEGAGEMM_MGX_SPARSE24_TRITON_MAX_ROWS", 64))


if _HAS_TRITON:
    _SPARSE24_CONFIGS = [
        triton.Config({"BLOCK_N": 1, "BLOCK_META": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 2, "BLOCK_META": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 4, "BLOCK_META": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 8, "BLOCK_META": 16}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_SPARSE24_CONFIGS, key=["M", "K", "N"])
    @triton.jit
    def _sparse24_cutlass_gemv_kernel(
        x_ptr,             # [M, K]
        values_ptr,        # [N, K / 2]
        meta_ptr,          # reordered CUTLASS int16 metadata [N, K / 16]
        bias_ptr,          # [N] or dummy
        out_ptr,           # [M, N]
        stride_xm,
        stride_xk,
        stride_vn,
        stride_vk,
        stride_om,
        stride_on,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        META_COLS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_META: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_meta = tl.arange(0, BLOCK_META)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Each int16 metadata element describes four groups of four input
        # features (16 K values) and therefore eight stored weight values.
        for meta_start in range(0, META_COLS, BLOCK_META):
            cols = meta_start + offs_meta
            col_mask = cols < META_COLS

            logical_rows = offs_n[:, None]
            logical_cols = cols[None, :]

            # Invert PyTorch/CUTLASS _calculate_meta_reordering_scatter_offsets.
            reordered_rows = (
                (logical_rows // 32) * 32
                + (logical_rows % 8) * 4
                + (logical_rows % 32) // 8
            )
            top_right = (reordered_rows % 2 == 0) & (logical_cols % 2 == 1)
            bottom_left = (reordered_rows % 2 == 1) & (logical_cols % 2 == 0)
            delta = top_right.to(tl.int32) - bottom_left.to(tl.int32)
            reordered_rows += delta
            reordered_cols = logical_cols - delta
            meta_offsets = (
                (reordered_cols // 2) * N * 2
                + reordered_rows * 2
                + reordered_cols % 2
            )
            matrix_mask = n_mask[:, None] & col_mask[None, :]
            packed_meta = tl.load(meta_ptr + meta_offsets, mask=matrix_mask, other=0)
            packed_meta = packed_meta.to(tl.int32) & 0xFFFF

            value_base = (
                logical_rows * stride_vn
                + logical_cols * 8 * stride_vk
            )
            x_group_base = (
                pid_m * stride_xm
                + logical_cols * 16 * stride_xk
            )

            # Four 2:4 groups live in every int16 metadata element.  The low
            # two bits of each nibble are the first position; the high two bits
            # are the second position.  Values are stored in that same order.
            for group in tl.static_range(0, 4):
                code = (packed_meta >> (group * 4)) & 0xF
                pos0 = code & 0x3
                pos1 = (code >> 2) & 0x3
                v0 = tl.load(
                    values_ptr + value_base + (group * 2) * stride_vk,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
                v1 = tl.load(
                    values_ptr + value_base + (group * 2 + 1) * stride_vk,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
                # Load the four activation candidates once per CTA and select
                # in registers for every output row.  This preserves x reuse
                # across BLOCK_N even though each weight row has different 2:4
                # positions.
                x_group = x_group_base + group * 4 * stride_xk
                xa = tl.load(
                    x_ptr + x_group,
                    mask=col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                xb = tl.load(
                    x_ptr + x_group + stride_xk,
                    mask=col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                xc = tl.load(
                    x_ptr + x_group + 2 * stride_xk,
                    mask=col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                xd = tl.load(
                    x_ptr + x_group + 3 * stride_xk,
                    mask=col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                selected0 = tl.where(
                    pos0 == 0,
                    xa,
                    tl.where(pos0 == 1, xb, tl.where(pos0 == 2, xc, xd)),
                )
                selected1 = tl.where(
                    pos1 == 0,
                    xa,
                    tl.where(pos1 == 1, xb, tl.where(pos1 == 2, xc, xd)),
                )
                acc += tl.sum(v0 * selected0 + v1 * selected1, axis=1)

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        tl.store(
            out_ptr + pid_m * stride_om + offs_n * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )


def sparse24_triton_available() -> bool:
    return bool(_HAS_TRITON)


def sparse24_triton_max_rows() -> int:
    return int(_TRITON_MAX_ROWS)


def sparse24_cutlass_gemv_eligible(
    x: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
    *,
    max_rows: Optional[int] = None,
) -> bool:
    """Return whether the specialized compact GEMV supports this invocation."""
    if not _HAS_TRITON or (torch.is_grad_enabled() and x.requires_grad):
        return False
    if not (x.is_cuda and values.is_cuda and metadata.is_cuda):
        return False
    if x.dtype not in {torch.float16, torch.bfloat16} or values.dtype != x.dtype:
        return False
    if metadata.dtype != torch.int16 or values.ndim != 2 or metadata.ndim != 2:
        return False
    if not values.is_contiguous() or not metadata.is_contiguous():
        return False
    if x.ndim < 2 or int(x.shape[-1]) <= 0:
        return False
    rows = int(x.numel() // int(x.shape[-1]))
    limit = int(_TRITON_MAX_ROWS if max_rows is None else max_rows)
    n_dim, packed_k = int(values.shape[0]), int(values.shape[1])
    k_dim = int(x.shape[-1])
    return bool(
        0 < rows <= max(1, limit)
        and n_dim > 0
        and n_dim % 32 == 0
        and k_dim % 64 == 0
        and packed_k == k_dim // 2
        and tuple(metadata.shape) == (n_dim, k_dim // 16)
    )


def sparse24_cutlass_gemv(
    x: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ sparse_weight.T`` from CUTLASS' compressed 2:4 storage."""
    if not sparse24_cutlass_gemv_eligible(x, values, metadata):
        raise ValueError("Invocation is not eligible for the MGX Triton 2:4 GEMV")

    original_shape = tuple(x.shape)
    k_dim = int(original_shape[-1])
    x_2d = x.flatten(0, -2)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n_dim = int(values.shape[0])
    expected_shape = (*original_shape[:-1], n_dim)

    if out is None:
        out_2d = torch.empty(
            (int(x_2d.shape[0]), n_dim),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        if tuple(out.shape) != expected_shape or out.device != x.device or out.dtype != x.dtype:
            raise ValueError(
                f"out must have shape/device/dtype {expected_shape}/{x.device}/{x.dtype}"
            )
        out_2d = out if out.ndim == 2 else out.flatten(0, -2)

    bias_ptr = bias if bias is not None else x_2d
    m_rows = int(x_2d.shape[0])
    grid = lambda meta: (m_rows, triton.cdiv(n_dim, meta["BLOCK_N"]))
    _sparse24_cutlass_gemv_kernel[grid](
        x_2d,
        values,
        metadata,
        bias_ptr,
        out_2d,
        x_2d.stride(0),
        x_2d.stride(1),
        values.stride(0),
        values.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        M=m_rows,
        K=k_dim,
        N=n_dim,
        META_COLS=k_dim // 16,
        HAS_BIAS=1 if bias is not None else 0,
    )
    if out is not None:
        return out
    return out_2d.view(expected_shape)


__all__ = [
    "sparse24_cutlass_gemv",
    "sparse24_cutlass_gemv_eligible",
    "sparse24_triton_available",
    "sparse24_triton_max_rows",
]
