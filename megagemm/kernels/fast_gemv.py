"""
Fast GEMV kernels for decode-time linear layers.

Optimized for small row-count (typically batch=1, seq=1) where generic GEMM/GEMV
dispatch overhead is a large fraction of latency.
"""

import os
import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# Read runtime knobs once to avoid per-call Python/env overhead in hot path.
_CFG_FORCED_BN = _env_int("MEGAGEMM_FAST_GEMV_BLOCK_N", 0)
_CFG_FORCED_BK = _env_int("MEGAGEMM_FAST_GEMV_BLOCK_K", 0)
_CFG_FORCED_WARPS = _env_int("MEGAGEMM_FAST_GEMV_NUM_WARPS", 0)
_CFG_FORCED_STAGES = _env_int("MEGAGEMM_FAST_GEMV_NUM_STAGES", 0)

_CFG_ROW_FORCED_BK = _env_int("MEGAGEMM_FAST_GEMV_ROW_BLOCK_K", 0)
_CFG_ROW_FORCED_WARPS = _env_int("MEGAGEMM_FAST_GEMV_ROW_NUM_WARPS", 0)
_CFG_ROW_FORCED_STAGES = _env_int("MEGAGEMM_FAST_GEMV_ROW_NUM_STAGES", 0)

_CFG_MODE = os.environ.get("MEGAGEMM_FAST_GEMV_MODE", "").strip().lower()
_CFG_AUTO_ROW = _env_bool("MEGAGEMM_FAST_GEMV_AUTO_ROW", False)
_CFG_AUTOTUNE = _env_bool("MEGAGEMM_FAST_GEMV_AUTOTUNE", False)
_CFG_SHAPE_GUARD = _env_bool("MEGAGEMM_FAST_GEMV_SHAPE_GUARD", True)
_CFG_FORCE_TRITON = _env_bool("MEGAGEMM_FAST_GEMV_FORCE_TRITON", False)
_CFG_MIN_N = max(1, _env_int("MEGAGEMM_FAST_GEMV_MIN_N", 8192))
_CFG_MAX_N = max(_CFG_MIN_N, _env_int("MEGAGEMM_FAST_GEMV_MAX_N", 32768))
_CFG_MIN_K = max(1, _env_int("MEGAGEMM_FAST_GEMV_MIN_K", 1024))
_CFG_SPLITK_MIN_N = max(1, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_MIN_N", 1024))
_CFG_SPLITK_MAX_N = max(_CFG_SPLITK_MIN_N, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_MAX_N", 8192))
_CFG_SPLITK_MIN_K = max(1, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_MIN_K", 4096))
_CFG_SPLITK_SIZE = max(256, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_SIZE", 1024))
_CFG_SPLITK_BLOCK_N = max(8, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_BLOCK_N", 32))
_CFG_SPLITK_BLOCK_K = max(64, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_BLOCK_K", 128))
_CFG_SPLITK_WARPS = max(1, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_NUM_WARPS", 4))
_CFG_SPLITK_STAGES = max(1, _env_int("MEGAGEMM_FAST_GEMV_SPLITK_NUM_STAGES", 2))


if _HAS_TRITON:
    _TILE_AUTOTUNE_CONFIGS = [
        triton.Config({'BLOCK_N': 8, 'BLOCK_K': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 8, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ]

    @triton.jit
    def _fast_gemv_row_kernel(
        x_ptr,          # [M, K]
        w_ptr,          # [N, K]
        b_ptr,          # [N] (or dummy)
        y_ptr,          # [M, N]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        M,
        K,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Row-wise GEMV: one program computes one output feature.
        This maximizes contiguous accesses along K for W and X.
        """
        # 1D launch avoids CUDA grid-y limit (65,535), important for large vocab lm_head.
        pid = tl.program_id(0)
        pid_m = pid // N
        pid_n = pid - pid_m * N
        valid = pid_m < M
        n_mask = pid_n < N
        full_mask = valid & n_mask

        acc = tl.zeros((), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=k_mask, other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + pid_n * stride_wn + offs_k * stride_wk,
                mask=full_mask & k_mask, other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x * w, axis=0)

        if HAS_BIAS:
            acc += tl.load(b_ptr + pid_n, mask=full_mask, other=0.0).to(tl.float32)

        tl.store(
            y_ptr + pid_m * stride_ym + pid_n * stride_yn,
            acc.to(y_ptr.dtype.element_ty),
            mask=full_mask,
        )

    @triton.jit
    def _fast_gemv_kernel(
        x_ptr,          # [M, K]
        w_ptr,          # [N, K]
        b_ptr,          # [N] (or dummy)
        y_ptr,          # [M, N]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        K,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=k_mask, other=0.0,
            ).to(tl.float32)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            acc += tl.sum(w * x[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)

    @triton.autotune(
        configs=_TILE_AUTOTUNE_CONFIGS,
        key=['K', 'N'],
    )
    @triton.jit
    def _fast_gemv_kernel_autotuned(
        x_ptr,          # [M, K]
        w_ptr,          # [N, K]
        b_ptr,          # [N] (or dummy)
        y_ptr,          # [M, N]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        K,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=k_mask, other=0.0,
            ).to(tl.float32)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            acc += tl.sum(w * x[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)

    @triton.jit
    def _fast_gemv_splitk_stage1_kernel(
        x_ptr,          # [M, K]
        w_ptr,          # [N, K]
        partial_ptr,    # [M, SPLITS, N], fp32
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        M,
        K,
        N,
        SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k_base = pid_s * SPLIT_K
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_rel in range(0, SPLIT_K, BLOCK_K):
            offs_k = k_base + k_rel + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=(pid_m < M) & k_mask,
                other=0.0,
            ).to(tl.float32)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        splits = tl.cdiv(K, SPLIT_K)
        partial_base = (pid_m * splits + pid_s) * N
        tl.store(partial_ptr + partial_base + offs_n, acc, mask=(pid_m < M) & n_mask)

    @triton.jit
    def _fast_gemv_splitk_reduce_kernel(
        partial_ptr,    # [M, SPLITS, N], fp32
        b_ptr,          # [N] (or dummy)
        y_ptr,          # [M, N]
        stride_ym, stride_yn,
        M,
        N,
        SPLITS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for s_start in range(0, SPLITS, BLOCK_S):
            offs_s = s_start + tl.arange(0, BLOCK_S)
            s_mask = offs_s < SPLITS
            vals = tl.load(
                partial_ptr + (pid_m * SPLITS + offs_s[:, None]) * N + offs_n[None, :],
                mask=(pid_m < M) & s_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(vals, axis=0)

        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(pid_m < M) & n_mask)

def _pick_kernel_cfg(k_dim: int, n_dim: int, m_rows: int):
    """
    Decode-oriented heuristic:
    - few output rows/program to reduce register pressure
    - larger K tiles to improve work-per-launch and contiguous K reads
    """
    if _CFG_FORCED_BN > 0 and _CFG_FORCED_BK > 0 and _CFG_FORCED_WARPS > 0:
        return _CFG_FORCED_BN, _CFG_FORCED_BK, _CFG_FORCED_WARPS, max(1, _CFG_FORCED_STAGES or 2)

    if n_dim >= 16384:
        # gate_up-like decode shapes (very wide N, moderate K) benefit from
        # more K work per CTA without forcing tiny N tiles globally.
        block_n = 16 if m_rows == 1 else 32
        block_k = 256 if k_dim >= 1024 else 128
        num_warps = 4
        num_stages = 3
    elif n_dim >= 8192:
        block_n = 16
        block_k = 256 if k_dim >= 1024 else 128
        num_warps = 4
        num_stages = 3
    elif n_dim >= 2048:
        block_n = 32
        block_k = 128 if k_dim >= 1024 else 64
        num_warps = 4
        num_stages = 2
    else:
        block_n = 64
        block_k = 64 if k_dim >= 512 else 32
        num_warps = 4
        num_stages = 2

    # Slightly wider N tile when rows>1 to amortize x loads.
    if m_rows > 1:
        block_n = min(64, block_n * 2)

    return block_n, block_k, num_warps, num_stages


def _pick_row_cfg(k_dim: int):
    if _CFG_ROW_FORCED_BK > 0 and _CFG_ROW_FORCED_WARPS > 0:
        return _CFG_ROW_FORCED_BK, _CFG_ROW_FORCED_WARPS, max(1, _CFG_ROW_FORCED_STAGES or 2)

    if k_dim >= 2048:
        return 256, 4, 3
    if k_dim >= 1024:
        return 256, 4, 2
    if k_dim >= 512:
        return 128, 4, 2
    return 64, 2, 2


def _splitk_splits(k_dim: int) -> int:
    return triton.cdiv(k_dim, _CFG_SPLITK_SIZE) if _HAS_TRITON else 1


def _should_use_splitk_path(k_dim: int, n_dim: int, m_rows: int) -> bool:
    if not _HAS_TRITON:
        return False
    if m_rows > 4:
        return False
    if k_dim < _CFG_SPLITK_MIN_K:
        return False
    if not (_CFG_SPLITK_MIN_N <= n_dim <= _CFG_SPLITK_MAX_N):
        return False
    return _splitk_splits(k_dim) > 1


def fast_gemv_splitk_scratch_shape(x: torch.Tensor, weight: torch.Tensor):
    if not (_HAS_TRITON and x.is_cuda and weight.is_cuda):
        return None
    x_2d = _flatten_rows(x)
    k_dim = int(weight.shape[1])
    n_dim = int(weight.shape[0])
    m_rows = int(x_2d.shape[0])
    if not _should_use_splitk_path(k_dim, n_dim, m_rows):
        return None
    return (m_rows, _splitk_splits(k_dim), n_dim)


def _pick_mode(k_dim: int, n_dim: int, m_rows: int) -> str:
    if _CFG_MODE in {"tile", "row", "splitk"}:
        return _CFG_MODE

    # Stable default: tile wins on end-to-end TPS in current decode workloads.
    # Keep row/split-K opt-in or caller-autotuned for targeted experiments.
    if _CFG_AUTO_ROW:
        if m_rows == 1 and 8192 <= n_dim <= 32768 and k_dim >= 1024:
            return "row"
    return "tile"


def _use_tile_autotune(k_dim: int, n_dim: int, m_rows: int) -> bool:
    # Off by default: autotune improved microkernel latency but regressed end-to-end TPS on L4.
    if not _CFG_AUTOTUNE:
        return False
    if m_rows > 4:
        return False
    if n_dim > 32768:
        # Skip lm_head-sized outputs by default to avoid expensive tuning overhead.
        return False
    if k_dim < 256:
        return False
    return True


def _should_use_triton_path(k_dim: int, n_dim: int, m_rows: int) -> bool:
    """
    Conservative default:
    - keep Triton for wide decode GEMV shapes (e.g. gate_up),
    - route smaller/larger projections to cuBLAS-backed addmm.
    """
    if _CFG_FORCE_TRITON:
        return True
    if not _CFG_SHAPE_GUARD:
        return True
    if m_rows > 4:
        return False
    if k_dim < _CFG_MIN_K:
        return False
    return _CFG_MIN_N <= n_dim <= _CFG_MAX_N


def fast_gemv_prefers_triton_shape(in_dim: int, out_dim: int, rows: int) -> bool:
    """Public shape policy helper for caller-side dispatch gating."""
    return (
        _should_use_triton_path(int(in_dim), int(out_dim), int(rows))
        or _should_use_splitk_path(int(in_dim), int(out_dim), int(rows))
    )


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x if x.is_contiguous() else x.contiguous()
    x_2d = x.flatten(0, -2)
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


def _linear_addmm_out(
    x_2d: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    out_2d: torch.Tensor,
) -> None:
    wt = w.transpose(0, 1)
    if bias is None:
        torch.mm(x_2d, wt, out=out_2d)
    else:
        bias_2d = bias.unsqueeze(0).expand(x_2d.shape[0], -1)
        torch.addmm(bias_2d, x_2d, wt, out=out_2d)


def fast_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    out: torch.Tensor = None,
    mode_override: str = "",
    scratch: torch.Tensor = None,
) -> torch.Tensor:
    """
    Decode-oriented linear: y = x @ weight.T + bias.
    Falls back to addmm/mm (cuBLAS-backed) when Triton path is unavailable.
    """
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"in_features mismatch: x={x.shape[-1]} weight={weight.shape[-1]}"
        )

    orig_shape = x.shape
    in_dim = orig_shape[-1]
    x_2d = _flatten_rows(x)
    w = weight if weight.is_contiguous() else weight.contiguous()
    m_rows = x_2d.shape[0]
    out_dim = w.shape[0]

    if out is None:
        out_2d = torch.empty((m_rows, out_dim), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (*orig_shape[:-1], out_dim)
        if tuple(out.shape) != tuple(expected_shape):
            raise ValueError(f"out shape mismatch: got {tuple(out.shape)} expected {tuple(expected_shape)}")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out must match x device and dtype")
        out_2d = out if out.ndim == 2 else out.flatten(0, -2)

    mode = (mode_override or "").strip().lower()
    if mode not in {"tile", "row", "splitk"}:
        mode = _pick_mode(in_dim, out_dim, m_rows)

    use_triton = (
        _HAS_TRITON
        and x.is_cuda
        and weight.is_cuda
        and (
            _should_use_triton_path(in_dim, out_dim, m_rows)
            or (mode == "splitk" and _should_use_splitk_path(in_dim, out_dim, m_rows))
        )
    )
    if not use_triton:
        _linear_addmm_out(x_2d, w, bias, out_2d)
        if out is not None:
            return out
        return out_2d.view(*orig_shape[:-1], out_dim)

    bias_ptr = bias if bias is not None else x_2d

    if mode == "splitk" and _should_use_splitk_path(in_dim, out_dim, m_rows):
        splits = _splitk_splits(in_dim)
        expected = (m_rows, splits, out_dim)
        if (
            scratch is None
            or tuple(scratch.shape) != expected
            or scratch.device != x.device
            or scratch.dtype != torch.float32
        ):
            scratch = torch.empty(expected, device=x.device, dtype=torch.float32)
        block_n = _CFG_SPLITK_BLOCK_N
        block_k = min(_CFG_SPLITK_BLOCK_K, _CFG_SPLITK_SIZE)
        grid_stage = (m_rows, triton.cdiv(out_dim, block_n), splits)
        _fast_gemv_splitk_stage1_kernel[grid_stage](
            x_2d, w, scratch,
            x_2d.stride(0), x_2d.stride(1),
            w.stride(0), w.stride(1),
            m_rows, in_dim, out_dim,
            SPLIT_K=_CFG_SPLITK_SIZE,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=_CFG_SPLITK_WARPS,
            num_stages=_CFG_SPLITK_STAGES,
        )
        grid_reduce = (m_rows, triton.cdiv(out_dim, block_n))
        _fast_gemv_splitk_reduce_kernel[grid_reduce](
            scratch, bias_ptr, out_2d,
            out_2d.stride(0), out_2d.stride(1),
            m_rows, out_dim,
            SPLITS=splits,
            HAS_BIAS=1 if bias is not None else 0,
            BLOCK_N=block_n,
            BLOCK_S=16,
            num_warps=1,
            num_stages=2,
        )
    elif mode == "row":
        block_k, num_warps, num_stages = _pick_row_cfg(in_dim)
        grid = (m_rows * out_dim,)
        _fast_gemv_row_kernel[grid](
            x_2d, w, bias_ptr, out_2d,
            x_2d.stride(0), x_2d.stride(1),
            w.stride(0), w.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            m_rows, in_dim, out_dim,
            HAS_BIAS=1 if bias is not None else 0,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        if _use_tile_autotune(in_dim, out_dim, m_rows):
            grid = lambda META: (m_rows, triton.cdiv(out_dim, META['BLOCK_N']))
            _fast_gemv_kernel_autotuned[grid](
                x_2d, w, bias_ptr, out_2d,
                x_2d.stride(0), x_2d.stride(1),
                w.stride(0), w.stride(1),
                out_2d.stride(0), out_2d.stride(1),
                in_dim, out_dim,
                HAS_BIAS=1 if bias is not None else 0,
            )
        else:
            block_n, block_k, num_warps, num_stages = _pick_kernel_cfg(in_dim, out_dim, m_rows)
            grid = (m_rows, triton.cdiv(out_dim, block_n))
            _fast_gemv_kernel[grid](
                x_2d, w, bias_ptr, out_2d,
                x_2d.stride(0), x_2d.stride(1),
                w.stride(0), w.stride(1),
                out_2d.stride(0), out_2d.stride(1),
                in_dim, out_dim,
                HAS_BIAS=1 if bias is not None else 0,
                BLOCK_N=block_n, BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    if out is not None:
        return out
    return out_2d.view(*orig_shape[:-1], out_dim)


HAS_TRITON_FAST_GEMV = _HAS_TRITON


def fast_gemv_runtime_config() -> dict:
    """Return effective runtime knobs (resolved at module import)."""
    return {
        "has_triton": bool(_HAS_TRITON),
        "mode": _CFG_MODE or "auto",
        "auto_row": bool(_CFG_AUTO_ROW),
        "autotune": bool(_CFG_AUTOTUNE),
        "shape_guard": bool(_CFG_SHAPE_GUARD),
        "force_triton": bool(_CFG_FORCE_TRITON),
        "min_n": int(_CFG_MIN_N),
        "max_n": int(_CFG_MAX_N),
        "min_k": int(_CFG_MIN_K),
        "splitk_min_n": int(_CFG_SPLITK_MIN_N),
        "splitk_max_n": int(_CFG_SPLITK_MAX_N),
        "splitk_min_k": int(_CFG_SPLITK_MIN_K),
        "splitk_size": int(_CFG_SPLITK_SIZE),
        "splitk_block_n": int(_CFG_SPLITK_BLOCK_N),
        "splitk_block_k": int(_CFG_SPLITK_BLOCK_K),
        "forced_block_n": int(_CFG_FORCED_BN),
        "forced_block_k": int(_CFG_FORCED_BK),
        "forced_num_warps": int(_CFG_FORCED_WARPS),
        "forced_num_stages": int(_CFG_FORCED_STAGES),
        "row_forced_block_k": int(_CFG_ROW_FORCED_BK),
        "row_forced_num_warps": int(_CFG_ROW_FORCED_WARPS),
        "row_forced_num_stages": int(_CFG_ROW_FORCED_STAGES),
    }
