"""Triton kernels for standalone MGX dense and 2:4 W4A16 weights.

Small decode batches use a direct reduction GEMV. Larger batches use a
load-packed/dequantize/tensor-core GEMM. The sparse kernel reads only the two
retained INT4 values per quartet and needs no external sparse library.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = ["HAS_NATIVE_W4A16", "get_native_w4a16_kernel_stats", "native_w4a16_linear"]

_STATS: Counter[str] = Counter()
_FAILED_KEYS: dict[tuple, str] = {}


if _HAS_TRITON:

    @triton.jit
    def _decode_position0(code):
        return tl.where(code < 3, 0, tl.where(code < 5, 1, 2))


    @triton.jit
    def _decode_position1(code):
        return tl.where(code == 0, 1, tl.where((code == 1) | (code == 3), 2, 3))


    @triton.jit
    def _signed_int4(value):
        value = value.to(tl.int32) & 0xF
        return tl.where(value < 8, value, value - 16)


    @triton.jit
    def _native_w4_dense_gemv_kernel(
        x_ptr, qweight_ptr, scales_ptr, bias_ptr, out_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_xm, stride_qn, stride_qk, stride_sn, stride_sg, stride_om,
        GROUP_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            packed = tl.load(
                qweight_ptr + offs_n[:, None] * stride_qn
                + (offs_k[None, :] // 2) * stride_qk,
                mask=mask_n[:, None] & mask_k[None, :], other=0,
            )
            shift = (offs_k % 2) * 4
            q = _signed_int4(packed >> shift[None, :]).to(tl.float32)
            scale = tl.load(
                scales_ptr + offs_n * stride_sn + (k_off // GROUP_SIZE) * stride_sg,
                mask=mask_n, other=0.0,
            ).to(tl.float32)
            acc += tl.sum(q * x[None, :], axis=1) * scale
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid_m * stride_om + offs_n, acc, mask=mask_n)


    @triton.jit
    def _native_w4_sparse24_gemv_kernel(
        x_ptr, qweight_ptr, scales_ptr, metadata_ptr, bias_ptr, out_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_xm, stride_qn, stride_qg, stride_sn, stride_sg,
        stride_mn, stride_mg, stride_om,
        GROUP_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for g_off in range(0, K // 4, BLOCK_G):
            offs_g = g_off + tl.arange(0, BLOCK_G)
            mask_g = offs_g < K // 4
            packed = tl.load(
                qweight_ptr + offs_n[:, None] * stride_qn + offs_g[None, :] * stride_qg,
                mask=mask_n[:, None] & mask_g[None, :], other=0,
            )
            meta = tl.load(
                metadata_ptr + offs_n[:, None] * stride_mn
                + (offs_g[None, :] // 2) * stride_mg,
                mask=mask_n[:, None] & mask_g[None, :], other=0,
            ).to(tl.int32)
            code = (meta >> ((offs_g % 2) * 4)[None, :]) & 0xF
            pos0 = _decode_position0(code)
            pos1 = _decode_position1(code)
            q0 = _signed_int4(packed).to(tl.float32)
            q1 = _signed_int4(packed >> 4).to(tl.float32)
            # Load each activation quartet once, then select in registers using
            # the per-output metadata. This avoids BLOCK_N duplicated x loads.
            xq0 = tl.load(
                x_ptr + pid_m * stride_xm + offs_g * 4,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
            xq1 = tl.load(
                x_ptr + pid_m * stride_xm + offs_g * 4 + 1,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
            xq2 = tl.load(
                x_ptr + pid_m * stride_xm + offs_g * 4 + 2,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
            xq3 = tl.load(
                x_ptr + pid_m * stride_xm + offs_g * 4 + 3,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
            x0 = tl.where(
                pos0 == 0, xq0[None, :],
                tl.where(pos0 == 1, xq1[None, :], tl.where(pos0 == 2, xq2[None, :], xq3[None, :])),
            )
            x1 = tl.where(
                pos1 == 0, xq0[None, :],
                tl.where(pos1 == 1, xq1[None, :], tl.where(pos1 == 2, xq2[None, :], xq3[None, :])),
            )
            scale = tl.load(
                scales_ptr + offs_n * stride_sn + ((g_off * 4) // GROUP_SIZE) * stride_sg,
                mask=mask_n, other=0.0,
            ).to(tl.float32)
            acc += tl.sum(q0 * x0 + q1 * x1, axis=1) * scale
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid_m * stride_om + offs_n, acc, mask=mask_n)


    @triton.jit
    def _native_w4_dense_gemm_kernel(
        x_ptr, qweight_ptr, scales_ptr, bias_ptr, out_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_xm, stride_qn, stride_qk, stride_sn, stride_sg,
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        # qweight is output-major, so compute W @ X.T as [N, M]. This keeps
        # packed K reads contiguous and transposes only the final store.
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x_t = tl.load(
                x_ptr + offs_k[:, None] + offs_m[None, :] * stride_xm,
                mask=mask_k[:, None] & mask_m[None, :], other=0.0,
            ).to(tl.float16)
            packed = tl.load(
                qweight_ptr + offs_n[:, None] * stride_qn
                + (offs_k[None, :] // 2) * stride_qk,
                mask=mask_n[:, None] & mask_k[None, :], other=0,
            )
            shift = (offs_k % 2) * 4
            q = _signed_int4(packed >> shift[None, :])
            scale = tl.load(
                scales_ptr + offs_n * stride_sn + (k_off // GROUP_SIZE) * stride_sg,
                mask=mask_n, other=0.0,
            )
            weight = q.to(tl.float16) * scale[:, None].to(tl.float16)
            acc += tl.dot(weight, x_t)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[:, None]
        tl.store(
            out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
            acc, mask=mask_n[:, None] & mask_m[None, :],
        )


    @triton.jit
    def _native_w4_sparse24_gemm_kernel(
        x_ptr, qweight_ptr, scales_ptr, metadata_ptr, bias_ptr, out_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_xm, stride_qn, stride_qg, stride_sn, stride_sg,
        stride_mn, stride_mg, stride_om, stride_on,
        GROUP_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x_t = tl.load(
                x_ptr + offs_k[:, None] + offs_m[None, :] * stride_xm,
                mask=mask_k[:, None] & mask_m[None, :], other=0.0,
            ).to(tl.float16)
            group = offs_k // 4
            position = offs_k % 4
            packed = tl.load(
                qweight_ptr + offs_n[:, None] * stride_qn + group[None, :] * stride_qg,
                mask=mask_n[:, None] & mask_k[None, :], other=0,
            )
            meta = tl.load(
                metadata_ptr + offs_n[:, None] * stride_mn
                + (group[None, :] // 2) * stride_mg,
                mask=mask_n[:, None] & mask_k[None, :], other=0,
            ).to(tl.int32)
            code = (meta >> ((group % 2) * 4)[None, :]) & 0xF
            pos0 = _decode_position0(code)
            pos1 = _decode_position1(code)
            q0 = _signed_int4(packed)
            q1 = _signed_int4(packed >> 4)
            q = tl.where(
                position[None, :] == pos0,
                q0,
                tl.where(position[None, :] == pos1, q1, 0),
            )
            scale = tl.load(
                scales_ptr + offs_n * stride_sn + (k_off // GROUP_SIZE) * stride_sg,
                mask=mask_n, other=0.0,
            )
            weight = q.to(tl.float16) * scale[:, None].to(tl.float16)
            acc += tl.dot(weight, x_t)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[:, None]
        tl.store(
            out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
            acc, mask=mask_n[:, None] & mask_m[None, :],
        )


def _gemm_config(m: int) -> tuple[int, int, int, int, int]:
    if m <= 16:
        return 16, 64, 32, 4, 3
    if m <= 64:
        return 32, 64, 32, 4, 3
    return 64, 64, 32, 8, 3


def native_w4a16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    metadata: torch.Tensor,
    bias: Optional[torch.Tensor],
    *, group_size: int, sparse24: bool,
) -> Optional[torch.Tensor]:
    if not _HAS_TRITON or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        _STATS["unavailable_fallbacks"] += 1
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    original_shape = tuple(x.shape)
    k = int(original_shape[-1])
    x2d = x.reshape(-1, k)
    m = int(x2d.shape[0])
    n = int(qweight.shape[0])
    if k % group_size or k % 8 or group_size < 32:
        return None

    route = "gemv" if m <= 4 else "gemm"
    capability = torch.cuda.get_device_capability(x.device)
    key = (route, bool(sparse24), m, n, k, int(group_size), capability)
    if key in _FAILED_KEYS:
        _STATS["cached_failure_fallbacks"] += 1
        return None

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    bias_ptr = bias if bias is not None else scales
    try:
        if route == "gemv":
            block_n = 32
            grid = (triton.cdiv(n, block_n), m)
            if sparse24:
                _native_w4_sparse24_gemv_kernel[grid](
                    x2d, qweight, scales, metadata, bias_ptr, out,
                    M=m, N=n, K=k, stride_xm=x2d.stride(0),
                    stride_qn=qweight.stride(0), stride_qg=qweight.stride(1),
                    stride_sn=scales.stride(0), stride_sg=scales.stride(1),
                    stride_mn=metadata.stride(0), stride_mg=metadata.stride(1),
                    stride_om=out.stride(0), GROUP_SIZE=group_size,
                    HAS_BIAS=bias is not None, BLOCK_N=block_n, BLOCK_G=group_size // 4,
                    num_warps=4, num_stages=2,
                )
            else:
                _native_w4_dense_gemv_kernel[grid](
                    x2d, qweight, scales, bias_ptr, out,
                    M=m, N=n, K=k, stride_xm=x2d.stride(0),
                    stride_qn=qweight.stride(0), stride_qk=qweight.stride(1),
                    stride_sn=scales.stride(0), stride_sg=scales.stride(1),
                    stride_om=out.stride(0), GROUP_SIZE=group_size,
                    HAS_BIAS=bias is not None, BLOCK_N=block_n, BLOCK_K=group_size,
                    num_warps=4, num_stages=2,
                )
        else:
            block_m, block_n, block_k, num_warps, num_stages = _gemm_config(m)
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
            if sparse24:
                _native_w4_sparse24_gemm_kernel[grid](
                    x2d, qweight, scales, metadata, bias_ptr, out,
                    M=m, N=n, K=k, stride_xm=x2d.stride(0),
                    stride_qn=qweight.stride(0), stride_qg=qweight.stride(1),
                    stride_sn=scales.stride(0), stride_sg=scales.stride(1),
                    stride_mn=metadata.stride(0), stride_mg=metadata.stride(1),
                    stride_om=out.stride(0), stride_on=out.stride(1),
                    GROUP_SIZE=group_size, HAS_BIAS=bias is not None,
                    BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                    num_warps=num_warps, num_stages=num_stages,
                )
            else:
                _native_w4_dense_gemm_kernel[grid](
                    x2d, qweight, scales, bias_ptr, out,
                    M=m, N=n, K=k, stride_xm=x2d.stride(0),
                    stride_qn=qweight.stride(0), stride_qk=qweight.stride(1),
                    stride_sn=scales.stride(0), stride_sg=scales.stride(1),
                    stride_om=out.stride(0), stride_on=out.stride(1),
                    GROUP_SIZE=group_size, HAS_BIAS=bias is not None,
                    BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                    num_warps=num_warps, num_stages=num_stages,
                )
    except Exception as exc:
        _FAILED_KEYS[key] = f"{type(exc).__name__}: {exc}"
        _STATS["compile_or_launch_failures"] += 1
        return None

    _STATS[f"{route}_{'sparse24' if sparse24 else 'dense'}_hits"] += 1
    return out.reshape(*original_shape[:-1], n)


def get_native_w4a16_kernel_stats() -> dict[str, object]:
    result: dict[str, object] = {
        "triton_available": bool(_HAS_TRITON),
        "failed_shape_count": len(_FAILED_KEYS),
        "failed_shapes": [
            {"key": list(key), "error": error} for key, error in list(_FAILED_KEYS.items())[:8]
        ],
    }
    result.update({key: int(value) for key, value in _STATS.items()})
    return result


HAS_NATIVE_W4A16 = _HAS_TRITON
