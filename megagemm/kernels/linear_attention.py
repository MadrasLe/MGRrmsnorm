"""
Triton kernels for Qwen 3.5 linear attention.

Current scope:
  - Recurrent gated delta rule, decode hot path (sequence_length == 1)
  - Chunk-local triangular solve for chunked prefill
"""

import math
import os
import warnings

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_HAS_TRITON_CHUNK_SOLVE = _HAS_TRITON
_HAS_TRITON_CHUNK_INTERCHUNK = _HAS_TRITON
_HAS_TRITON_CHUNK_INTERCHUNK_FUSED = _HAS_TRITON
_HAS_TRITON_CHUNK_INTERCHUNK_SCAN = _HAS_TRITON
_HAS_TRITON_CHUNK_PARALLEL_AFFINE = _HAS_TRITON
_HAS_TRITON_AFFINE_PREFIX_SCAN = _HAS_TRITON
_HAS_TRITON_AFFINE_BLELLOCH_SCAN = _HAS_TRITON
_MAX_TRITON_CHUNK_LEN = 64
_MAX_TRITON_SCAN_CHUNKS = 16
_SCAN_BLOCK_C_BUCKETS = (8, 16, 24, 32, 48, 64)
_SCAN_LAUNCH_POLICY_CACHE = {}
_BLELLOCH_SCAN_ALIASES = {"blelloch", "hierarchical", "tree"}

__all__ = [
    'HAS_TRITON_LINEAR_ATTN',
    'chunk_interchunk',
    'chunk_interchunk_scan',
    'recurrent_gated_delta_decode',
    'recurrent_gated_delta_decode_from_ab',
    'recurrent_gated_delta_prefill',
    'chunk_state_projection',
    'chunk_state_update',
    'solve_chunk_local_attention',
]


def _debug_linear_attn_enabled() -> bool:
    return os.environ.get("MEGAGEMM_DEBUG_LINEAR_ATTN", "0") == "1"


def _get_env_int(name: str, default: int, min_value: int = 1, max_value: int = 1024) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _get_env_int_optional(name: str, min_value: int = 1, max_value: int = 1024) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return max(min_value, min(max_value, value))


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw == "":
        return default
    return raw not in {"0", "false", "no", "off"}


def _parallel_scan_algo() -> str:
    algo = os.environ.get("MEGAGEMM_QWEN35_PARALLEL_SCAN_ALGO", "").strip().lower()
    if not algo:
        raw = os.environ.get("MEGAGEMM_QWEN35_PARALLEL_SCAN", "").strip().lower()
        if raw in _BLELLOCH_SCAN_ALIASES or raw in {"hillis", "hillis-steele", "hs"}:
            algo = raw
    if algo in _BLELLOCH_SCAN_ALIASES:
        # Dense-affine Blelloch regressed Qwen3.5 long-prefill on T4 and can
        # fail through stale notebook env vars. Keep old requests harmless.
        return "hillis"
    return "hillis"


def _resolve_chunk_scan_launch_policy(
    query: torch.Tensor,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
) -> tuple[int, int]:
    """
    Resolve Triton scan launch policy.

    Env overrides still take precedence:
      - MEGAGEMM_QWEN35_SCAN_MAX_CHUNKS
      - MEGAGEMM_QWEN35_SCAN_NUM_WARPS
    """
    env_max_chunks = _get_env_int_optional(
        "MEGAGEMM_QWEN35_SCAN_MAX_CHUNKS",
        min_value=1,
        max_value=64,
    )
    env_num_warps = _get_env_int_optional(
        "MEGAGEMM_QWEN35_SCAN_NUM_WARPS",
        min_value=1,
        max_value=8,
    )
    if env_max_chunks is not None or env_num_warps is not None:
        max_scan_chunks = env_max_chunks if env_max_chunks is not None else _MAX_TRITON_SCAN_CHUNKS
        scan_num_warps = env_num_warps if env_num_warps is not None else 4
        return max(1, min(64, max_scan_chunks)), max(1, min(8, scan_num_warps))

    # Auto policy by GPU generation + problem size.
    max_scan_chunks = _MAX_TRITON_SCAN_CHUNKS
    scan_num_warps = 4
    if query.is_cuda and torch.cuda.is_available():
        device = query.device
        dev_idx = -1 if device.index is None else int(device.index)
        cache_key = (dev_idx, key_dim, value_dim)
        cached = _SCAN_LAUNCH_POLICY_CACHE.get(cache_key)
        if cached is not None:
            max_scan_chunks, scan_num_warps = cached
        else:
            major, minor = torch.cuda.get_device_capability(device)
            sm = major * 10 + minor
            if sm >= 90:
                max_scan_chunks = 48
                scan_num_warps = 8 if value_dim >= 64 else 4
            elif sm >= 80:
                max_scan_chunks = 32
                scan_num_warps = 8 if value_dim >= 128 else 4
            elif sm >= 75:
                max_scan_chunks = 24
                scan_num_warps = 4
            else:
                max_scan_chunks = 16
                scan_num_warps = 4

            # Very wide heads can increase register pressure.
            # Keep old GPUs conservative; allow newer GPUs to keep wider scan windows.
            if key_dim >= 256 or value_dim >= 256:
                max_scan_chunks = min(max_scan_chunks, 24)
            _SCAN_LAUNCH_POLICY_CACHE[cache_key] = (max_scan_chunks, scan_num_warps)

    # Keep a sensible lower bound for auto mode, but never exceed current workload.
    max_scan_chunks = max(8, min(64, max_scan_chunks))
    max_scan_chunks = min(max_scan_chunks, max(1, num_chunks))
    return max_scan_chunks, scan_num_warps


def _choose_scan_window_chunks(num_chunks: int, max_scan_chunks: int) -> int:
    """
    Pick a scan window size that keeps launch count low while avoiding tiny
    tail windows that add disproportionate kernel overhead.
    """
    num_chunks = max(1, int(num_chunks))
    max_scan_chunks = max(1, int(max_scan_chunks))
    if num_chunks <= max_scan_chunks:
        return num_chunks

    base_candidates = {max_scan_chunks, 32, 24, 20, 16, 12, 10, 8, 6, 4, 2, 1}
    candidates = sorted(
        [cand for cand in base_candidates if cand <= max_scan_chunks and cand <= num_chunks],
        reverse=True,
    )
    if not candidates:
        return min(num_chunks, max_scan_chunks)

    best = candidates[0]
    best_score = None
    for cand in candidates:
        windows = math.ceil(num_chunks / cand)
        rem = num_chunks % cand
        tail = cand if rem == 0 else rem
        tiny_tail = 1 if rem != 0 and tail < max(8, cand // 3) else 0
        imbalance = 0 if rem == 0 else (cand - tail)
        score = (windows, tiny_tail, imbalance, -cand)  # lower is better
        if best_score is None or score < best_score:
            best = cand
            best_score = score
    return best


def _bucket_scan_block_c(chunk_count: int) -> int:
    """
    Keep BLOCK_C in a small set of stable buckets to reduce Triton recompiles
    across varying prompt lengths.
    """
    chunk_count = max(1, int(chunk_count))
    for bucket in _SCAN_BLOCK_C_BUCKETS:
        if chunk_count <= bucket:
            return bucket
    return _SCAN_BLOCK_C_BUCKETS[-1]


def _build_chunk_affine_params_triton(
    key_4d: torch.Tensor,
    key_cumdecay_4d: torch.Tensor,
    value_4d: torch.Tensor,
    gate_3d: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _HAS_TRITON_CHUNK_PARALLEL_AFFINE

    if not _HAS_TRITON_CHUNK_PARALLEL_AFFINE:
        return None
    if not key_4d.is_cuda:
        return None
    if not _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN_TRITON", False):
        return None

    bh, num_chunks, chunk_len, key_dim = key_4d.shape
    value_dim = value_4d.shape[-1]
    block_t = min(triton.next_power_of_2(chunk_len), _MAX_TRITON_CHUNK_LEN)
    block_k = 16 if key_dim > 64 else 32
    block_v = 32 if value_dim <= 64 else 64
    a_out = torch.empty((bh, num_chunks, key_dim, key_dim), device=key_4d.device, dtype=out_dtype)
    b_out = torch.empty((bh, num_chunks, key_dim, value_dim), device=key_4d.device, dtype=out_dtype)

    try:
        grid_a = (
            bh * num_chunks,
            triton.cdiv(key_dim, block_k),
            triton.cdiv(key_dim, block_k),
        )
        _chunk_affine_a_kernel[grid_a](
            key_4d,
            key_cumdecay_4d,
            gate_3d,
            a_out,
            key_4d.stride(0), key_4d.stride(1), key_4d.stride(2), key_4d.stride(3),
            key_cumdecay_4d.stride(0), key_cumdecay_4d.stride(1), key_cumdecay_4d.stride(2), key_cumdecay_4d.stride(3),
            gate_3d.stride(0), gate_3d.stride(1), gate_3d.stride(2),
            a_out.stride(0), a_out.stride(1), a_out.stride(2), a_out.stride(3),
            num_chunks, chunk_len, key_dim,
            BLOCK_KR=block_k,
            BLOCK_KC=block_k,
            BLOCK_T=block_t,
            num_warps=4,
        )

        grid_b = (
            bh * num_chunks,
            triton.cdiv(key_dim, block_k),
            triton.cdiv(value_dim, block_v),
        )
        _chunk_affine_b_kernel[grid_b](
            key_4d,
            value_4d,
            gate_3d,
            b_out,
            key_4d.stride(0), key_4d.stride(1), key_4d.stride(2), key_4d.stride(3),
            value_4d.stride(0), value_4d.stride(1), value_4d.stride(2), value_4d.stride(3),
            gate_3d.stride(0), gate_3d.stride(1), gate_3d.stride(2),
            b_out.stride(0), b_out.stride(1), b_out.stride(2), b_out.stride(3),
            num_chunks, chunk_len, key_dim, value_dim,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return a_out, b_out
    except Exception as exc:
        _HAS_TRITON_CHUNK_PARALLEL_AFFINE = False
        if _debug_linear_attn_enabled():
            warnings.warn(
                f"parallel affine Triton fallback activated: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
        return None


def _prefix_scan_affine_triton(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Inclusive Hillis-Steele scan of affine transitions on GPU:
      compose((A2,B2),(A1,B1)) = (A2@A1, A2@B1 + B2)
    """
    global _HAS_TRITON_AFFINE_PREFIX_SCAN

    if not _HAS_TRITON_AFFINE_PREFIX_SCAN:
        return None
    if not A.is_cuda:
        return None
    if not _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN_TRITON", False):
        return None

    bh, num_chunks, key_dim, _ = A.shape
    value_dim = B.shape[-1]
    if num_chunks <= 1:
        return A, B

    block_k = 16 if key_dim > 64 else 32
    block_v = 32 if value_dim <= 64 else 64

    a_src = A
    b_src = B
    a_dst = torch.empty_like(A)
    b_dst = torch.empty_like(B)

    try:
        offset = 1
        while offset < num_chunks:
            grid_a = (
                bh * num_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(key_dim, block_k),
            )
            _affine_compose_a_kernel[grid_a](
                a_src,
                a_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                num_chunks, key_dim, offset,
                BLOCK_R=block_k,
                BLOCK_C=block_k,
                BLOCK_K=block_k,
                num_warps=4,
            )

            grid_b = (
                bh * num_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(value_dim, block_v),
            )
            _affine_compose_b_kernel[grid_b](
                a_src,
                b_src,
                b_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                b_src.stride(0), b_src.stride(1), b_src.stride(2), b_src.stride(3),
                num_chunks, key_dim, value_dim, offset,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_INNER=block_k,
                num_warps=4,
            )

            a_src, a_dst = a_dst, a_src
            b_src, b_dst = b_dst, b_src
            offset <<= 1
        return a_src, b_src
    except Exception as exc:
        _HAS_TRITON_AFFINE_PREFIX_SCAN = False
        if _debug_linear_attn_enabled():
            warnings.warn(
                f"parallel affine prefix Triton fallback activated: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
        return None


def _compose_affine(
    after_A: torch.Tensor,
    after_B: torch.Tensor,
    before_A: torch.Tensor,
    before_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two affine transitions, applying ``before`` then ``after``."""
    return (
        torch.matmul(after_A, before_A),
        torch.matmul(after_A, before_B) + after_B,
    )


def _prefix_scan_affine_blelloch_torch(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inclusive Blelloch scan for affine chunk transitions.

    The scan is over the chunk axis.  Each element is an affine transition
    ``S' = A @ S + B``.  The returned prefix at chunk ``i`` represents applying
    transitions ``0..i`` in order.
    """
    bh, num_chunks, key_dim, _ = A.shape
    value_dim = B.shape[-1]
    if num_chunks <= 1:
        return A, B

    padded_chunks = 1 << (num_chunks - 1).bit_length()
    eye = torch.eye(key_dim, device=A.device, dtype=A.dtype)
    if padded_chunks == num_chunks:
        work_A = A.clone()
        work_B = B.clone()
    else:
        work_A = eye.view(1, 1, key_dim, key_dim).expand(
            bh, padded_chunks, key_dim, key_dim
        ).clone()
        work_B = torch.zeros(
            (bh, padded_chunks, key_dim, value_dim),
            device=B.device,
            dtype=B.dtype,
        )
        work_A[:, :num_chunks].copy_(A)
        work_B[:, :num_chunks].copy_(B)

    step = 1
    while step < padded_chunks:
        right_idx = torch.arange(
            step * 2 - 1,
            padded_chunks,
            step * 2,
            device=A.device,
        )
        left_idx = right_idx - step
        combined_A, combined_B = _compose_affine(
            work_A[:, right_idx],
            work_B[:, right_idx],
            work_A[:, left_idx],
            work_B[:, left_idx],
        )
        work_A[:, right_idx] = combined_A
        work_B[:, right_idx] = combined_B
        step <<= 1

    work_A[:, padded_chunks - 1] = eye
    work_B[:, padded_chunks - 1].zero_()

    step = padded_chunks >> 1
    while step:
        right_idx = torch.arange(
            step * 2 - 1,
            padded_chunks,
            step * 2,
            device=A.device,
        )
        left_idx = right_idx - step
        left_A = work_A[:, left_idx].clone()
        left_B = work_B[:, left_idx].clone()
        prefix_A = work_A[:, right_idx].clone()
        prefix_B = work_B[:, right_idx].clone()

        work_A[:, left_idx] = prefix_A
        work_B[:, left_idx] = prefix_B
        combined_A, combined_B = _compose_affine(
            left_A,
            left_B,
            prefix_A,
            prefix_B,
        )
        work_A[:, right_idx] = combined_A
        work_B[:, right_idx] = combined_B
        step >>= 1

    exclusive_A = work_A[:, :num_chunks]
    exclusive_B = work_B[:, :num_chunks]
    return _compose_affine(A, B, exclusive_A, exclusive_B)


def _prefix_scan_affine_blelloch_triton(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Inclusive Blelloch scan using Triton kernels for the expensive affine
    composition phases.

    This is still an experimental dense-affine path, but it avoids the Python
    advanced-indexing/matmul loop from ``_prefix_scan_affine_blelloch_torch``.
    It is only enabled when the parallel-scan Triton experiment is explicitly
    requested.
    """
    global _HAS_TRITON_AFFINE_BLELLOCH_SCAN

    if not _HAS_TRITON_AFFINE_BLELLOCH_SCAN:
        return None
    if not A.is_cuda:
        return None
    if not _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN_TRITON", False):
        return None

    bh, num_chunks, key_dim, _ = A.shape
    value_dim = B.shape[-1]
    if num_chunks <= 1:
        return A, B

    tail_max_chunks = _get_env_int(
        "MEGAGEMM_QWEN35_BLELLOCH_TAIL_MAX_CHUNKS",
        8,
        min_value=0,
        max_value=1024,
    )
    main_chunks = 1 << (num_chunks.bit_length() - 1)
    tail_chunks = num_chunks - main_chunks
    if 0 < tail_chunks <= tail_max_chunks and main_chunks >= 2:
        main_prefix = _prefix_scan_affine_blelloch_triton(
            A[:, :main_chunks],
            B[:, :main_chunks],
        )
        if main_prefix is None:
            return None
        main_A, main_B = main_prefix
        tail_A = A[:, main_chunks:]
        tail_B = B[:, main_chunks:]
        if tail_chunks == 1:
            local_tail_A, local_tail_B = tail_A, tail_B
        else:
            local_tail = _prefix_scan_affine_triton(tail_A, tail_B)
            if local_tail is None:
                local_tail_A, local_tail_B = _prefix_scan_affine_blelloch_torch(tail_A, tail_B)
            else:
                local_tail_A, local_tail_B = local_tail

        base_A = main_A[:, -1:].to(local_tail_A.dtype)
        base_B = main_B[:, -1:].to(local_tail_B.dtype)
        global_tail_A, global_tail_B = _compose_affine(
            local_tail_A,
            local_tail_B,
            base_A,
            base_B,
        )
        prefix_A = torch.empty_like(A)
        prefix_B = torch.empty_like(B)
        prefix_A[:, :main_chunks].copy_(main_A)
        prefix_B[:, :main_chunks].copy_(main_B)
        prefix_A[:, main_chunks:].copy_(global_tail_A)
        prefix_B[:, main_chunks:].copy_(global_tail_B)
        return prefix_A, prefix_B

    padded_chunks = 1 << (num_chunks - 1).bit_length()
    block_k = 16 if key_dim > 64 else 32
    block_v = 32 if value_dim <= 64 else 64

    try:
        if padded_chunks == num_chunks:
            work_A = A.clone()
            work_B = B.clone()
        else:
            eye = torch.eye(key_dim, device=A.device, dtype=A.dtype)
            work_A = eye.view(1, 1, key_dim, key_dim).expand(
                bh,
                padded_chunks,
                key_dim,
                key_dim,
            ).clone()
            work_B = torch.zeros(
                (bh, padded_chunks, key_dim, value_dim),
                device=B.device,
                dtype=B.dtype,
            )
            work_A[:, :num_chunks].copy_(A)
            work_B[:, :num_chunks].copy_(B)

        tmp_A = torch.empty_like(work_A)
        tmp_B = torch.empty_like(work_B)
        a_src, a_dst = work_A, tmp_A
        b_src, b_dst = work_B, tmp_B

        step = 1
        while step < padded_chunks:
            grid_a = (
                bh * padded_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(key_dim, block_k),
            )
            grid_b = (
                bh * padded_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(value_dim, block_v),
            )
            _affine_blelloch_upsweep_a_kernel[grid_a](
                a_src,
                a_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                a_dst.stride(0), a_dst.stride(1), a_dst.stride(2), a_dst.stride(3),
                padded_chunks, key_dim,
                STEP=step,
                BLOCK_R=block_k,
                BLOCK_C=block_k,
                BLOCK_K=block_k,
                num_warps=4,
            )
            _affine_blelloch_upsweep_b_kernel[grid_b](
                a_src,
                b_src,
                b_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                b_src.stride(0), b_src.stride(1), b_src.stride(2), b_src.stride(3),
                b_dst.stride(0), b_dst.stride(1), b_dst.stride(2), b_dst.stride(3),
                padded_chunks, key_dim, value_dim,
                STEP=step,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_INNER=block_k,
                num_warps=4,
            )
            a_src, a_dst = a_dst, a_src
            b_src, b_dst = b_dst, b_src
            step <<= 1

        eye = torch.eye(key_dim, device=A.device, dtype=A.dtype)
        a_src[:, padded_chunks - 1].copy_(eye)
        b_src[:, padded_chunks - 1].zero_()

        step = padded_chunks >> 1
        while step:
            grid_a = (
                bh * padded_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(key_dim, block_k),
            )
            grid_b = (
                bh * padded_chunks,
                triton.cdiv(key_dim, block_k),
                triton.cdiv(value_dim, block_v),
            )
            _affine_blelloch_downsweep_a_kernel[grid_a](
                a_src,
                a_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                a_dst.stride(0), a_dst.stride(1), a_dst.stride(2), a_dst.stride(3),
                padded_chunks, key_dim,
                STEP=step,
                BLOCK_R=block_k,
                BLOCK_C=block_k,
                BLOCK_K=block_k,
                num_warps=4,
            )
            _affine_blelloch_downsweep_b_kernel[grid_b](
                a_src,
                b_src,
                b_dst,
                a_src.stride(0), a_src.stride(1), a_src.stride(2), a_src.stride(3),
                b_src.stride(0), b_src.stride(1), b_src.stride(2), b_src.stride(3),
                b_dst.stride(0), b_dst.stride(1), b_dst.stride(2), b_dst.stride(3),
                padded_chunks, key_dim, value_dim,
                STEP=step,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_INNER=block_k,
                num_warps=4,
            )
            a_src, a_dst = a_dst, a_src
            b_src, b_dst = b_dst, b_src
            step >>= 1

        exclusive_A = a_src[:, :num_chunks]
        exclusive_B = b_src[:, :num_chunks]
        prefix_A = torch.empty_like(A)
        prefix_B = torch.empty_like(B)
        grid_a = (
            bh * num_chunks,
            triton.cdiv(key_dim, block_k),
            triton.cdiv(key_dim, block_k),
        )
        grid_b = (
            bh * num_chunks,
            triton.cdiv(key_dim, block_k),
            triton.cdiv(value_dim, block_v),
        )
        _affine_compose_pair_a_kernel[grid_a](
            A,
            exclusive_A,
            prefix_A,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            exclusive_A.stride(0), exclusive_A.stride(1), exclusive_A.stride(2), exclusive_A.stride(3),
            prefix_A.stride(0), prefix_A.stride(1), prefix_A.stride(2), prefix_A.stride(3),
            num_chunks, key_dim,
            BLOCK_R=block_k,
            BLOCK_C=block_k,
            BLOCK_K=block_k,
            num_warps=4,
        )
        _affine_compose_pair_b_kernel[grid_b](
            A,
            B,
            exclusive_B,
            prefix_B,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            exclusive_B.stride(0), exclusive_B.stride(1), exclusive_B.stride(2), exclusive_B.stride(3),
            prefix_B.stride(0), prefix_B.stride(1), prefix_B.stride(2), prefix_B.stride(3),
            num_chunks, key_dim, value_dim,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_INNER=block_k,
            num_warps=4,
        )
        return prefix_A, prefix_B
    except Exception as exc:
        _HAS_TRITON_AFFINE_BLELLOCH_SCAN = False
        if _debug_linear_attn_enabled():
            warnings.warn(
                f"Blelloch affine prefix Triton fallback activated: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
        return None


def _chunk_interchunk_scan_parallel_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    key_cumdecay: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Experimental parallel chunk scan based on affine transition composition.

    Enabled via MEGAGEMM_QWEN35_PARALLEL_SCAN=1. Intended for long prefill
    where chunk count is high and the sequential c-loop in Triton becomes costly.
    """
    batch, heads, num_chunks, chunk_len, key_dim = query.shape
    value_dim = value.shape[-1]
    bh = batch * heads

    use_fp32 = _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN_FP32", False)
    work_dtype = torch.float32 if (use_fp32 or not query.is_cuda) else torch.float16

    query_4d = _flatten_bh_view_5d(query.to(work_dtype))
    key_4d = _flatten_bh_view_5d(key.to(work_dtype))
    key_cumdecay_4d = _flatten_bh_view_5d(key_cumdecay.to(work_dtype))
    value_4d = _flatten_bh_view_5d(value.to(work_dtype))
    gate_3d = _flatten_bh_view_4d(gate.to(torch.float32))
    state_3d = _flatten_bh_view_4d(state.to(work_dtype))

    n = bh * num_chunks
    kc_flat = key_cumdecay_4d.reshape(n, chunk_len, key_dim)
    v_flat = value_4d.reshape(n, chunk_len, value_dim)
    affine_params = _build_chunk_affine_params_triton(
        key_4d,
        key_cumdecay_4d,
        value_4d,
        gate_3d,
        work_dtype,
    )
    if affine_params is not None:
        A, B = affine_params
    else:
        # u_t = exp(g_last - g_t) * k_t
        gate_last = gate_3d[:, :, chunk_len - 1]  # [bh, c]
        decay = torch.exp(gate_last.unsqueeze(-1) - gate_3d).to(work_dtype)  # [bh, c, t]
        u = key_4d * decay.unsqueeze(-1)  # [bh, c, t, k]

        # Per-chunk affine transition:
        #   S' = A * S + B
        # with
        #   A = exp(g_last) * I - sum_t u_t * kc_t^T
        #   B = sum_t u_t * v_t^T
        u_flat = u.reshape(n, chunk_len, key_dim)
        a_left = torch.bmm(u_flat.transpose(1, 2), kc_flat).view(bh, num_chunks, key_dim, key_dim)
        B = torch.bmm(u_flat.transpose(1, 2), v_flat).view(bh, num_chunks, key_dim, value_dim)
        eye = torch.eye(key_dim, device=query.device, dtype=work_dtype).view(1, 1, key_dim, key_dim)
        A = torch.exp(gate_last).to(work_dtype).unsqueeze(-1).unsqueeze(-1) * eye - a_left

    if _parallel_scan_algo() == "blelloch":
        prefix_affine = _prefix_scan_affine_blelloch_triton(A, B)
        if prefix_affine is not None:
            prefix_A, prefix_B = prefix_affine
        else:
            prefix_A, prefix_B = _prefix_scan_affine_blelloch_torch(A, B)
    else:
        prefix_affine = _prefix_scan_affine_triton(A, B)
        if prefix_affine is not None:
            prefix_A, prefix_B = prefix_affine
        else:
            # Inclusive parallel prefix-scan of affine transforms (Hillis-Steele):
            # compose((A2,B2),(A1,B1)) = (A2@A1, A2@B1 + B2)
            prefix_A = A
            prefix_B = B
            tmp_A = torch.empty_like(prefix_A)
            tmp_B = torch.empty_like(prefix_B)
            offset = 1
            while offset < num_chunks:
                tmp_A[:, :offset].copy_(prefix_A[:, :offset])
                tmp_B[:, :offset].copy_(prefix_B[:, :offset])
                left_A = prefix_A[:, offset:]
                right_A = prefix_A[:, :-offset]
                left_B = prefix_B[:, offset:]
                right_B = prefix_B[:, :-offset]
                tmp_A[:, offset:] = torch.matmul(left_A, right_A)
                tmp_B[:, offset:] = torch.matmul(left_A, right_B) + left_B
                prefix_A, tmp_A = tmp_A, prefix_A
                prefix_B, tmp_B = tmp_B, prefix_B
                offset <<= 1

    # Exclusive prefix for chunk-start states.
    state_start = torch.empty((bh, num_chunks, key_dim, value_dim), device=query.device, dtype=work_dtype)
    state_start[:, 0] = state_3d
    if num_chunks > 1:
        state_start[:, 1:] = torch.matmul(prefix_A[:, :-1], state_3d.unsqueeze(1)) + prefix_B[:, :-1]

    query_scaled = query_4d * torch.exp(gate_3d).to(work_dtype).unsqueeze(-1)
    state_start_flat = state_start.reshape(n, key_dim, value_dim)
    value_prime = torch.bmm(kc_flat, state_start_flat).view(bh, num_chunks, chunk_len, value_dim)
    attn_inter = torch.bmm(
        query_scaled.reshape(n, chunk_len, key_dim), state_start_flat,
    ).view(bh, num_chunks, chunk_len, value_dim)
    value_new = value_4d - value_prime

    state_final = torch.matmul(prefix_A[:, -1], state_3d) + prefix_B[:, -1]
    return (
        value_new.view(batch, heads, num_chunks, chunk_len, value_dim).to(torch.float32),
        attn_inter.view(batch, heads, num_chunks, chunk_len, value_dim).to(torch.float32),
        state_final.view(batch, heads, key_dim, value_dim).to(torch.float32),
    )


def _parallel_scan_allowed(
    query: torch.Tensor,
    num_chunks: int,
    key_dim: int,
) -> bool:
    """
    Guard rails for experimental parallel-scan path.

    Parallel-scan is experimental and opt-in. It is intended for long-prefill
    experimentation and may require tuning by GPU.
    """
    if not _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN", False):
        return False
    if _get_env_bool("MEGAGEMM_QWEN35_PARALLEL_SCAN_FORCE", False):
        return True

    if not query.is_cuda:
        return False
    if num_chunks < 16:
        return False
    if key_dim >= 64:
        # Qwen3.5 (head_dim=128) regressed with current parallel-scan implementation.
        # Keep this path disabled by default for wide heads unless FORCE is set.
        return False
    return True


def _flatten_bh_view_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {x.ndim}D")
    if x.stride(0) != x.shape[1] * x.stride(1):
        return x.contiguous().view(x.shape[0] * x.shape[1], x.shape[2])
    return x.as_strided(
        (x.shape[0] * x.shape[1], x.shape[2]),
        (x.stride(1), x.stride(2)),
    )


def _flatten_bh_view_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
    if x.stride(0) != x.shape[1] * x.stride(1):
        return x.contiguous().view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    return x.as_strided(
        (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
        (x.stride(1), x.stride(2), x.stride(3)),
    )


def _flatten_bh_view_5d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got {x.ndim}D")
    if x.stride(0) != x.shape[1] * x.stride(1):
        return x.contiguous().view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    return x.as_strided(
        (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]),
        (x.stride(1), x.stride(2), x.stride(3), x.stride(4)),
    )


def _validate_qkv_group_shapes(q_heads: int, kv_heads: int, num_kv_groups: int, op_name: str) -> None:
    if kv_heads != q_heads * num_kv_groups:
        raise ValueError(
            f"{op_name}: expected kv_heads == q_heads * num_kv_groups, "
            f"got kv_heads={kv_heads}, q_heads={q_heads}, num_kv_groups={num_kv_groups}",
        )


if _HAS_TRITON:
    @triton.jit
    def _recurrent_delta_prefill_kernel(
        query_ptr, key_ptr, value_ptr,
        beta_ptr, gate_ptr,
        state_ptr, out_ptr,
        stride_q_b, stride_q_h, stride_q_s, stride_q_k,
        stride_k_b, stride_k_h, stride_k_s, stride_k_k,
        stride_v_b, stride_v_h, stride_v_s, stride_v_v,
        stride_b_b, stride_b_h, stride_b_s,
        stride_g_b, stride_g_h, stride_g_s,
        stride_s_b, stride_s_h, stride_s_k, stride_s_v,
        stride_o_b, stride_o_h, stride_o_s, stride_o_v,
        seq_len, num_v_heads, num_kv_groups, query_scale, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
        NORMALIZE_QK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)
        batch_idx = bh // num_v_heads
        v_head_idx = bh % num_v_heads
        k_head_idx = v_head_idx // num_kv_groups

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        for t in range(BLOCK_T):
            valid_t = t < seq_len
            q_inv_norm = 1.0
            k_inv_norm = 1.0
            gate = tl.load(
                gate_ptr + batch_idx * stride_g_b + v_head_idx * stride_g_h + t * stride_g_s,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_ptr + batch_idx * stride_b_b + v_head_idx * stride_b_h + t * stride_b_s,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            gate_exp = tl.exp(gate)

            kv_mem_raw = tl.zeros([BLOCK_V], dtype=tl.float32)
            if NORMALIZE_QK:
                q_sumsq = 0.0
                k_sumsq = 0.0
                for k0 in range(0, key_dim, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < key_dim
                    state_ptrs = (
                        state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                        + offs_k[:, None] * stride_s_k
                        + offs_v[None, :] * stride_s_v
                    )
                    state = tl.load(
                        state_ptrs,
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    key_raw = tl.load(
                        key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + t * stride_k_s + offs_k * stride_k_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    query_raw = tl.load(
                        query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + t * stride_q_s + offs_k * stride_q_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    k_sumsq += tl.sum(key_raw * key_raw, axis=0)
                    q_sumsq += tl.sum(query_raw * query_raw, axis=0)
                    kv_mem_raw += tl.sum((state * gate_exp) * key_raw[:, None], axis=0)
                k_inv_norm = tl.rsqrt(k_sumsq + 1e-6)
                q_inv_norm = tl.rsqrt(q_sumsq + 1e-6)
                kv_mem = kv_mem_raw * k_inv_norm
            else:
                for k0 in range(0, key_dim, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < key_dim
                    state_ptrs = (
                        state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                        + offs_k[:, None] * stride_s_k
                        + offs_v[None, :] * stride_s_v
                    )
                    state = tl.load(
                        state_ptrs,
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    key = tl.load(
                        key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + t * stride_k_s + offs_k * stride_k_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    kv_mem_raw += tl.sum((state * gate_exp) * key[:, None], axis=0)
                kv_mem = kv_mem_raw

            value = tl.load(
                value_ptr + batch_idx * stride_v_b + v_head_idx * stride_v_h + t * stride_v_s + offs_v * stride_v_v,
                mask=mask_v & valid_t,
                other=0.0,
            ).to(tl.float32)
            delta = (value - kv_mem) * beta

            out = tl.zeros([BLOCK_V], dtype=tl.float32)
            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim

                state_ptrs = (
                    state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                    + offs_k[:, None] * stride_s_k
                    + offs_v[None, :] * stride_s_v
                )
                state = tl.load(
                    state_ptrs,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                ).to(tl.float32)
                key = tl.load(
                    key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + t * stride_k_s + offs_k * stride_k_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                query = tl.load(
                    query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + t * stride_q_s + offs_k * stride_q_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                if NORMALIZE_QK:
                    key = key * k_inv_norm
                    query = query * q_inv_norm
                query = query * query_scale

                new_state = state * gate_exp + key[:, None] * delta[None, :]
                tl.store(
                    state_ptrs,
                    new_state,
                    mask=mask_k[:, None] & mask_v[None, :],
                )
                out += tl.sum(new_state * query[:, None], axis=0)

            tl.store(
                out_ptr + batch_idx * stride_o_b + v_head_idx * stride_o_h + t * stride_o_s + offs_v * stride_o_v,
                out,
                mask=mask_v & valid_t,
            )


    @triton.jit
    def _chunk_local_attention_solve_kernel(
        attn_ptr,
        stride_m, stride_r, stride_c,
        CHUNK_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        mid = tl.program_id(0)
        cols = tl.arange(0, BLOCK)

        for i in range(1, CHUNK_SIZE):
            row_base = attn_ptr + mid * stride_m + i * stride_r
            row_ptrs = row_base + cols * stride_c
            row = tl.load(row_ptrs, mask=cols < i, other=0.0).to(tl.float32)
            acc = row

            for k in range(i):
                prev_ptrs = attn_ptr + mid * stride_m + k * stride_r + cols * stride_c
                prev = tl.load(prev_ptrs, mask=cols < i, other=0.0).to(tl.float32)
                coeff = tl.load(row_base + k * stride_c).to(tl.float32)
                acc += coeff * prev

            tl.store(row_ptrs, acc, mask=cols < i)

        diag = tl.arange(0, BLOCK)
        diag_ptrs = attn_ptr + mid * stride_m + diag * stride_r + diag * stride_c
        diag_vals = tl.load(diag_ptrs, mask=diag < CHUNK_SIZE, other=0.0).to(tl.float32)
        tl.store(diag_ptrs, diag_vals + 1.0, mask=diag < CHUNK_SIZE)


    @triton.jit
    def _chunk_interchunk_scan_kernel(
        query_ptr, key_ptr, key_cumdecay_ptr, value_ptr, gate_ptr, state_ptr,
        value_new_ptr, attn_inter_ptr,
        stride_q_bh, stride_q_c, stride_q_t, stride_q_k,
        stride_k_bh, stride_k_c, stride_k_t, stride_k_k,
        stride_kc_bh, stride_kc_c, stride_kc_t, stride_kc_k,
        stride_v_bh, stride_v_c, stride_v_t, stride_v_v,
        stride_g_bh, stride_g_c, stride_g_t,
        stride_s_bh, stride_s_k, stride_s_v,
        stride_vn_bh, stride_vn_c, stride_vn_t, stride_vn_v,
        stride_ai_bh, stride_ai_c, stride_ai_t, stride_ai_v,
        num_chunks, chunk_len, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        for c in range(BLOCK_C):
            valid_c = c < num_chunks

            for t in range(BLOCK_T):
                valid_t = valid_c & (t < chunk_len)
                gate = tl.load(
                    gate_ptr + bh * stride_g_bh + c * stride_g_c + t * stride_g_t,
                    mask=valid_t,
                    other=0.0,
                ).to(tl.float32)
                gate_exp = tl.exp(gate)

                value_prime = tl.zeros([BLOCK_V], dtype=tl.float32)
                attn_inter = tl.zeros([BLOCK_V], dtype=tl.float32)

                for k0 in range(0, key_dim, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < key_dim
                    state_ptrs = (
                        state_ptr + bh * stride_s_bh
                        + offs_k[:, None] * stride_s_k
                        + offs_v[None, :] * stride_s_v
                    )
                    state = tl.load(
                        state_ptrs,
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    key_cum = tl.load(
                        key_cumdecay_ptr + bh * stride_kc_bh + c * stride_kc_c + t * stride_kc_t + offs_k * stride_kc_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    query = tl.load(
                        query_ptr + bh * stride_q_bh + c * stride_q_c + t * stride_q_t + offs_k * stride_q_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    value_prime += tl.sum(state * key_cum[:, None], axis=0)
                    attn_inter += tl.sum(state * (query * gate_exp)[:, None], axis=0)

                value = tl.load(
                    value_ptr + bh * stride_v_bh + c * stride_v_c + t * stride_v_t + offs_v * stride_v_v,
                    mask=mask_v & valid_t,
                    other=0.0,
                ).to(tl.float32)
                value_new = value - value_prime

                tl.store(
                    value_new_ptr + bh * stride_vn_bh + c * stride_vn_c + t * stride_vn_t + offs_v * stride_vn_v,
                    value_new,
                    mask=mask_v & valid_t,
                )
                tl.store(
                    attn_inter_ptr + bh * stride_ai_bh + c * stride_ai_c + t * stride_ai_t + offs_v * stride_ai_v,
                    attn_inter,
                    mask=mask_v & valid_t,
                )

            gate_last = tl.load(
                gate_ptr + bh * stride_g_bh + c * stride_g_c + (chunk_len - 1) * stride_g_t,
                mask=valid_c,
                other=0.0,
            ).to(tl.float32)
            gate_last_exp = tl.exp(gate_last)

            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim
                state_ptrs = (
                    state_ptr + bh * stride_s_bh
                    + offs_k[:, None] * stride_s_k
                    + offs_v[None, :] * stride_s_v
                )
                new_state = tl.load(
                    state_ptrs,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                ).to(tl.float32) * gate_last_exp

                for t in range(BLOCK_T):
                    valid_t = valid_c & (t < chunk_len)
                    gate_t = tl.load(
                        gate_ptr + bh * stride_g_bh + c * stride_g_c + t * stride_g_t,
                        mask=valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    decay = tl.exp(gate_last - gate_t)
                    key = tl.load(
                        key_ptr + bh * stride_k_bh + c * stride_k_c + t * stride_k_t + offs_k * stride_k_k,
                        mask=mask_k & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    value_new = tl.load(
                        value_new_ptr + bh * stride_vn_bh + c * stride_vn_c + t * stride_vn_t + offs_v * stride_vn_v,
                        mask=mask_v & valid_t,
                        other=0.0,
                    ).to(tl.float32)
                    new_state += (key * decay)[:, None] * value_new[None, :]

                tl.store(
                    state_ptrs,
                    new_state,
                    mask=mask_k[:, None] & mask_v[None, :],
                )


    @triton.jit
    def _chunk_interchunk_kernel(
        query_ptr, key_ptr, key_cumdecay_ptr, value_ptr, gate_ptr, state_ptr,
        value_new_ptr, attn_inter_ptr,
        stride_q_bh, stride_q_t, stride_q_k,
        stride_k_bh, stride_k_t, stride_k_k,
        stride_kc_bh, stride_kc_t, stride_kc_k,
        stride_v_bh, stride_v_t, stride_v_v,
        stride_g_bh, stride_g_t,
        stride_s_bh, stride_s_k, stride_s_v,
        stride_vn_bh, stride_vn_t, stride_vn_v,
        stride_ai_bh, stride_ai_t, stride_ai_v,
        chunk_len, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        for t in range(BLOCK_T):
            valid_t = t < chunk_len
            gate = tl.load(
                gate_ptr + bh * stride_g_bh + t * stride_g_t,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            gate_exp = tl.exp(gate)

            value_prime = tl.zeros([BLOCK_V], dtype=tl.float32)
            attn_inter = tl.zeros([BLOCK_V], dtype=tl.float32)

            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim
                state_ptrs = (
                    state_ptr + bh * stride_s_bh
                    + offs_k[:, None] * stride_s_k
                    + offs_v[None, :] * stride_s_v
                )
                state = tl.load(
                    state_ptrs,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                ).to(tl.float32)
                key_cum = tl.load(
                    key_cumdecay_ptr + bh * stride_kc_bh + t * stride_kc_t + offs_k * stride_kc_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                query = tl.load(
                    query_ptr + bh * stride_q_bh + t * stride_q_t + offs_k * stride_q_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                value_prime += tl.sum(state * key_cum[:, None], axis=0)
                attn_inter += tl.sum(state * (query * gate_exp)[:, None], axis=0)

            value = tl.load(
                value_ptr + bh * stride_v_bh + t * stride_v_t + offs_v * stride_v_v,
                mask=mask_v & valid_t,
                other=0.0,
            ).to(tl.float32)
            value_new = value - value_prime

            tl.store(
                value_new_ptr + bh * stride_vn_bh + t * stride_vn_t + offs_v * stride_vn_v,
                value_new,
                mask=mask_v & valid_t,
            )
            tl.store(
                attn_inter_ptr + bh * stride_ai_bh + t * stride_ai_t + offs_v * stride_ai_v,
                attn_inter,
                mask=mask_v & valid_t,
            )

        gate_last = tl.load(
            gate_ptr + bh * stride_g_bh + (chunk_len - 1) * stride_g_t,
        ).to(tl.float32)
        gate_last_exp = tl.exp(gate_last)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim
            state_ptrs = (
                state_ptr + bh * stride_s_bh
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            new_state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32) * gate_last_exp

            for t in range(BLOCK_T):
                valid_t = t < chunk_len
                gate_t = tl.load(
                    gate_ptr + bh * stride_g_bh + t * stride_g_t,
                    mask=valid_t,
                    other=0.0,
                ).to(tl.float32)
                decay = tl.exp(gate_last - gate_t)
                key = tl.load(
                    key_ptr + bh * stride_k_bh + t * stride_k_t + offs_k * stride_k_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                value_new = tl.load(
                    value_new_ptr + bh * stride_vn_bh + t * stride_vn_t + offs_v * stride_vn_v,
                    mask=mask_v & valid_t,
                    other=0.0,
                ).to(tl.float32)
                new_state += (key * decay)[:, None] * value_new[None, :]

                tl.store(
                    state_ptrs,
                    new_state,
                    mask=mask_k[:, None] & mask_v[None, :],
                )


    @triton.jit
    def _chunk_affine_a_kernel(
        key_ptr, key_cumdecay_ptr, gate_ptr, a_ptr,
        stride_k_bh, stride_k_c, stride_k_t, stride_k_k,
        stride_kc_bh, stride_kc_c, stride_kc_t, stride_kc_k,
        stride_g_bh, stride_g_c, stride_g_t,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        num_chunks, chunk_len, key_dim,
        BLOCK_KR: tl.constexpr,
        BLOCK_KC: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_kr = tl.program_id(1)
        pid_kc = tl.program_id(2)

        bh = pid_bhc // num_chunks
        c = pid_bhc % num_chunks

        offs_kr = pid_kr * BLOCK_KR + tl.arange(0, BLOCK_KR)
        offs_kc = pid_kc * BLOCK_KC + tl.arange(0, BLOCK_KC)
        mask_kr = offs_kr < key_dim
        mask_kc = offs_kc < key_dim

        gate_last = tl.load(
            gate_ptr + bh * stride_g_bh + c * stride_g_c + (chunk_len - 1) * stride_g_t,
        ).to(tl.float32)
        gate_last_exp = tl.exp(gate_last)

        acc = tl.zeros([BLOCK_KR, BLOCK_KC], dtype=tl.float32)

        for t in range(BLOCK_T):
            valid_t = t < chunk_len
            gate_t = tl.load(
                gate_ptr + bh * stride_g_bh + c * stride_g_c + t * stride_g_t,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            decay = tl.exp(gate_last - gate_t)

            key_row = tl.load(
                key_ptr + bh * stride_k_bh + c * stride_k_c + t * stride_k_t + offs_kr * stride_k_k,
                mask=mask_kr & valid_t,
                other=0.0,
            ).to(tl.float32)
            key_cum_col = tl.load(
                key_cumdecay_ptr + bh * stride_kc_bh + c * stride_kc_c + t * stride_kc_t + offs_kc * stride_kc_k,
                mask=mask_kc & valid_t,
                other=0.0,
            ).to(tl.float32)
            acc += (key_row * decay)[:, None] * key_cum_col[None, :]

        out = -acc
        out += tl.where(offs_kr[:, None] == offs_kc[None, :], gate_last_exp, 0.0)

        tl.store(
            a_ptr + bh * stride_a_bh + c * stride_a_c + offs_kr[:, None] * stride_a_r + offs_kc[None, :] * stride_a_col,
            out,
            mask=mask_kr[:, None] & mask_kc[None, :],
        )


    @triton.jit
    def _chunk_affine_b_kernel(
        key_ptr, value_ptr, gate_ptr, b_ptr,
        stride_k_bh, stride_k_c, stride_k_t, stride_k_k,
        stride_v_bh, stride_v_c, stride_v_t, stride_v_v,
        stride_g_bh, stride_g_c, stride_g_t,
        stride_b_bh, stride_b_c, stride_b_k, stride_b_v,
        num_chunks, chunk_len, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_v = tl.program_id(2)

        bh = pid_bhc // num_chunks
        c = pid_bhc % num_chunks

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < key_dim
        mask_v = offs_v < value_dim

        gate_last = tl.load(
            gate_ptr + bh * stride_g_bh + c * stride_g_c + (chunk_len - 1) * stride_g_t,
        ).to(tl.float32)

        acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

        for t in range(BLOCK_T):
            valid_t = t < chunk_len
            gate_t = tl.load(
                gate_ptr + bh * stride_g_bh + c * stride_g_c + t * stride_g_t,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            decay = tl.exp(gate_last - gate_t)

            key_row = tl.load(
                key_ptr + bh * stride_k_bh + c * stride_k_c + t * stride_k_t + offs_k * stride_k_k,
                mask=mask_k & valid_t,
                other=0.0,
            ).to(tl.float32)
            value_col = tl.load(
                value_ptr + bh * stride_v_bh + c * stride_v_c + t * stride_v_t + offs_v * stride_v_v,
                mask=mask_v & valid_t,
                other=0.0,
            ).to(tl.float32)
            acc += (key_row * decay)[:, None] * value_col[None, :]

        tl.store(
            b_ptr + bh * stride_b_bh + c * stride_b_c + offs_k[:, None] * stride_b_k + offs_v[None, :] * stride_b_v,
            acc,
            mask=mask_k[:, None] & mask_v[None, :],
        )


    @triton.jit
    def _affine_compose_a_kernel(
        a_src_ptr, a_dst_ptr,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        num_chunks, key_dim, offset,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_c = tl.program_id(2)

        bh = pid_bhc // num_chunks
        chunk_idx = pid_bhc % num_chunks

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_r = offs_r < key_dim
        mask_c = offs_c < key_dim
        mask_rc = mask_r[:, None] & mask_c[None, :]

        dst_ptrs = (
            a_dst_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
            + offs_r[:, None] * stride_a_r + offs_c[None, :] * stride_a_col
        )
        src_ptrs = (
            a_src_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
            + offs_r[:, None] * stride_a_r + offs_c[None, :] * stride_a_col
        )

        if chunk_idx < offset:
            vals = tl.load(src_ptrs, mask=mask_rc, other=0.0).to(tl.float32)
            tl.store(dst_ptrs, vals, mask=mask_rc)
            return

        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            left_ptrs = (
                a_src_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
                + offs_r[:, None] * stride_a_r + offs_k[None, :] * stride_a_col
            )
            right_ptrs = (
                a_src_ptr + bh * stride_a_bh + (chunk_idx - offset) * stride_a_c
                + offs_k[:, None] * stride_a_r + offs_c[None, :] * stride_a_col
            )
            left = tl.load(left_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            right = tl.load(right_ptrs, mask=mask_k[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(left, right)

        tl.store(dst_ptrs, acc, mask=mask_rc)


    @triton.jit
    def _affine_compose_b_kernel(
        a_src_ptr, b_src_ptr, b_dst_ptr,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        stride_b_bh, stride_b_c, stride_b_k, stride_b_v,
        num_chunks, key_dim, value_dim, offset,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_v = tl.program_id(2)

        bh = pid_bhc // num_chunks
        chunk_idx = pid_bhc % num_chunks

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < key_dim
        mask_v = offs_v < value_dim
        mask_kv = mask_k[:, None] & mask_v[None, :]

        dst_ptrs = (
            b_dst_ptr + bh * stride_b_bh + chunk_idx * stride_b_c
            + offs_k[:, None] * stride_b_k + offs_v[None, :] * stride_b_v
        )
        src_b_ptrs = (
            b_src_ptr + bh * stride_b_bh + chunk_idx * stride_b_c
            + offs_k[:, None] * stride_b_k + offs_v[None, :] * stride_b_v
        )

        if chunk_idx < offset:
            vals = tl.load(src_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            tl.store(dst_ptrs, vals, mask=mask_kv)
            return

        acc = tl.load(src_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
        for k0 in range(0, key_dim, BLOCK_INNER):
            offs_inner = k0 + tl.arange(0, BLOCK_INNER)
            mask_inner = offs_inner < key_dim

            left_a_ptrs = (
                a_src_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
                + offs_k[:, None] * stride_a_r + offs_inner[None, :] * stride_a_col
            )
            right_b_ptrs = (
                b_src_ptr + bh * stride_b_bh + (chunk_idx - offset) * stride_b_c
                + offs_inner[:, None] * stride_b_k + offs_v[None, :] * stride_b_v
            )
            left = tl.load(left_a_ptrs, mask=mask_k[:, None] & mask_inner[None, :], other=0.0).to(tl.float32)
            right = tl.load(right_b_ptrs, mask=mask_inner[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(left, right)

        tl.store(dst_ptrs, acc, mask=mask_kv)


    @triton.jit
    def _affine_blelloch_upsweep_a_kernel(
        a_src_ptr, a_dst_ptr,
        stride_src_bh, stride_src_c, stride_src_r, stride_src_col,
        stride_dst_bh, stride_dst_c, stride_dst_r, stride_dst_col,
        padded_chunks, key_dim,
        STEP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_c = tl.program_id(2)

        bh = pid_bhc // padded_chunks
        chunk_idx = pid_bhc % padded_chunks

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_r = offs_r < key_dim
        mask_c = offs_c < key_dim
        mask_rc = mask_r[:, None] & mask_c[None, :]

        dst_ptrs = (
            a_dst_ptr + bh * stride_dst_bh + chunk_idx * stride_dst_c
            + offs_r[:, None] * stride_dst_r + offs_c[None, :] * stride_dst_col
        )
        src_ptrs = (
            a_src_ptr + bh * stride_src_bh + chunk_idx * stride_src_c
            + offs_r[:, None] * stride_src_r + offs_c[None, :] * stride_src_col
        )

        if ((chunk_idx + 1) % (STEP * 2)) != 0:
            vals = tl.load(src_ptrs, mask=mask_rc, other=0.0).to(tl.float32)
            tl.store(dst_ptrs, vals, mask=mask_rc)
            return

        left_idx = chunk_idx - STEP
        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim
            after_ptrs = (
                a_src_ptr + bh * stride_src_bh + chunk_idx * stride_src_c
                + offs_r[:, None] * stride_src_r + offs_k[None, :] * stride_src_col
            )
            before_ptrs = (
                a_src_ptr + bh * stride_src_bh + left_idx * stride_src_c
                + offs_k[:, None] * stride_src_r + offs_c[None, :] * stride_src_col
            )
            after = tl.load(after_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            before = tl.load(before_ptrs, mask=mask_k[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(after, before)

        tl.store(dst_ptrs, acc, mask=mask_rc)


    @triton.jit
    def _affine_blelloch_upsweep_b_kernel(
        a_src_ptr, b_src_ptr, b_dst_ptr,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        stride_b_src_bh, stride_b_src_c, stride_b_src_k, stride_b_src_v,
        stride_b_dst_bh, stride_b_dst_c, stride_b_dst_k, stride_b_dst_v,
        padded_chunks, key_dim, value_dim,
        STEP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_v = tl.program_id(2)

        bh = pid_bhc // padded_chunks
        chunk_idx = pid_bhc % padded_chunks

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < key_dim
        mask_v = offs_v < value_dim
        mask_kv = mask_k[:, None] & mask_v[None, :]

        src_b_ptrs = (
            b_src_ptr + bh * stride_b_src_bh + chunk_idx * stride_b_src_c
            + offs_k[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
        )
        dst_b_ptrs = (
            b_dst_ptr + bh * stride_b_dst_bh + chunk_idx * stride_b_dst_c
            + offs_k[:, None] * stride_b_dst_k + offs_v[None, :] * stride_b_dst_v
        )

        if ((chunk_idx + 1) % (STEP * 2)) != 0:
            vals = tl.load(src_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            tl.store(dst_b_ptrs, vals, mask=mask_kv)
            return

        left_idx = chunk_idx - STEP
        acc = tl.load(src_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
        for k0 in range(0, key_dim, BLOCK_INNER):
            offs_inner = k0 + tl.arange(0, BLOCK_INNER)
            mask_inner = offs_inner < key_dim
            after_a_ptrs = (
                a_src_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
                + offs_k[:, None] * stride_a_r + offs_inner[None, :] * stride_a_col
            )
            before_b_ptrs = (
                b_src_ptr + bh * stride_b_src_bh + left_idx * stride_b_src_c
                + offs_inner[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
            )
            after = tl.load(after_a_ptrs, mask=mask_k[:, None] & mask_inner[None, :], other=0.0).to(tl.float32)
            before = tl.load(before_b_ptrs, mask=mask_inner[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(after, before)

        tl.store(dst_b_ptrs, acc, mask=mask_kv)


    @triton.jit
    def _affine_blelloch_downsweep_a_kernel(
        a_src_ptr, a_dst_ptr,
        stride_src_bh, stride_src_c, stride_src_r, stride_src_col,
        stride_dst_bh, stride_dst_c, stride_dst_r, stride_dst_col,
        padded_chunks, key_dim,
        STEP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_c = tl.program_id(2)

        bh = pid_bhc // padded_chunks
        chunk_idx = pid_bhc % padded_chunks

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_r = offs_r < key_dim
        mask_c = offs_c < key_dim
        mask_rc = mask_r[:, None] & mask_c[None, :]

        dst_ptrs = (
            a_dst_ptr + bh * stride_dst_bh + chunk_idx * stride_dst_c
            + offs_r[:, None] * stride_dst_r + offs_c[None, :] * stride_dst_col
        )
        src_ptrs = (
            a_src_ptr + bh * stride_src_bh + chunk_idx * stride_src_c
            + offs_r[:, None] * stride_src_r + offs_c[None, :] * stride_src_col
        )
        position = (chunk_idx + 1) % (STEP * 2)

        if position == STEP:
            right_idx = chunk_idx + STEP
            right_ptrs = (
                a_src_ptr + bh * stride_src_bh + right_idx * stride_src_c
                + offs_r[:, None] * stride_src_r + offs_c[None, :] * stride_src_col
            )
            vals = tl.load(right_ptrs, mask=mask_rc, other=0.0).to(tl.float32)
            tl.store(dst_ptrs, vals, mask=mask_rc)
            return

        if position != 0:
            vals = tl.load(src_ptrs, mask=mask_rc, other=0.0).to(tl.float32)
            tl.store(dst_ptrs, vals, mask=mask_rc)
            return

        left_idx = chunk_idx - STEP
        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim
            left_ptrs = (
                a_src_ptr + bh * stride_src_bh + left_idx * stride_src_c
                + offs_r[:, None] * stride_src_r + offs_k[None, :] * stride_src_col
            )
            right_ptrs = (
                a_src_ptr + bh * stride_src_bh + chunk_idx * stride_src_c
                + offs_k[:, None] * stride_src_r + offs_c[None, :] * stride_src_col
            )
            left = tl.load(left_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            right = tl.load(right_ptrs, mask=mask_k[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(left, right)

        tl.store(dst_ptrs, acc, mask=mask_rc)


    @triton.jit
    def _affine_blelloch_downsweep_b_kernel(
        a_src_ptr, b_src_ptr, b_dst_ptr,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        stride_b_src_bh, stride_b_src_c, stride_b_src_k, stride_b_src_v,
        stride_b_dst_bh, stride_b_dst_c, stride_b_dst_k, stride_b_dst_v,
        padded_chunks, key_dim, value_dim,
        STEP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_v = tl.program_id(2)

        bh = pid_bhc // padded_chunks
        chunk_idx = pid_bhc % padded_chunks

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < key_dim
        mask_v = offs_v < value_dim
        mask_kv = mask_k[:, None] & mask_v[None, :]

        dst_b_ptrs = (
            b_dst_ptr + bh * stride_b_dst_bh + chunk_idx * stride_b_dst_c
            + offs_k[:, None] * stride_b_dst_k + offs_v[None, :] * stride_b_dst_v
        )
        src_b_ptrs = (
            b_src_ptr + bh * stride_b_src_bh + chunk_idx * stride_b_src_c
            + offs_k[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
        )
        position = (chunk_idx + 1) % (STEP * 2)

        if position == STEP:
            right_idx = chunk_idx + STEP
            right_b_ptrs = (
                b_src_ptr + bh * stride_b_src_bh + right_idx * stride_b_src_c
                + offs_k[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
            )
            vals = tl.load(right_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            tl.store(dst_b_ptrs, vals, mask=mask_kv)
            return

        if position != 0:
            vals = tl.load(src_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
            tl.store(dst_b_ptrs, vals, mask=mask_kv)
            return

        left_idx = chunk_idx - STEP
        left_b_ptrs = (
            b_src_ptr + bh * stride_b_src_bh + left_idx * stride_b_src_c
            + offs_k[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
        )
        acc = tl.load(left_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
        for k0 in range(0, key_dim, BLOCK_INNER):
            offs_inner = k0 + tl.arange(0, BLOCK_INNER)
            mask_inner = offs_inner < key_dim
            left_a_ptrs = (
                a_src_ptr + bh * stride_a_bh + left_idx * stride_a_c
                + offs_k[:, None] * stride_a_r + offs_inner[None, :] * stride_a_col
            )
            right_b_ptrs = (
                b_src_ptr + bh * stride_b_src_bh + chunk_idx * stride_b_src_c
                + offs_inner[:, None] * stride_b_src_k + offs_v[None, :] * stride_b_src_v
            )
            left_a = tl.load(left_a_ptrs, mask=mask_k[:, None] & mask_inner[None, :], other=0.0).to(tl.float32)
            right_b = tl.load(right_b_ptrs, mask=mask_inner[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(left_a, right_b)

        tl.store(dst_b_ptrs, acc, mask=mask_kv)


    @triton.jit
    def _affine_compose_pair_a_kernel(
        after_a_ptr, before_a_ptr, out_a_ptr,
        stride_after_bh, stride_after_c, stride_after_r, stride_after_col,
        stride_before_bh, stride_before_c, stride_before_r, stride_before_col,
        stride_out_bh, stride_out_c, stride_out_r, stride_out_col,
        num_chunks, key_dim,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_c = tl.program_id(2)

        bh = pid_bhc // num_chunks
        chunk_idx = pid_bhc % num_chunks

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_r = offs_r < key_dim
        mask_c = offs_c < key_dim
        mask_rc = mask_r[:, None] & mask_c[None, :]

        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim
            after_ptrs = (
                after_a_ptr + bh * stride_after_bh + chunk_idx * stride_after_c
                + offs_r[:, None] * stride_after_r + offs_k[None, :] * stride_after_col
            )
            before_ptrs = (
                before_a_ptr + bh * stride_before_bh + chunk_idx * stride_before_c
                + offs_k[:, None] * stride_before_r + offs_c[None, :] * stride_before_col
            )
            after = tl.load(after_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            before = tl.load(before_ptrs, mask=mask_k[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(after, before)

        out_ptrs = (
            out_a_ptr + bh * stride_out_bh + chunk_idx * stride_out_c
            + offs_r[:, None] * stride_out_r + offs_c[None, :] * stride_out_col
        )
        tl.store(out_ptrs, acc, mask=mask_rc)


    @triton.jit
    def _affine_compose_pair_b_kernel(
        after_a_ptr, after_b_ptr, before_b_ptr, out_b_ptr,
        stride_a_bh, stride_a_c, stride_a_r, stride_a_col,
        stride_after_b_bh, stride_after_b_c, stride_after_b_k, stride_after_b_v,
        stride_before_b_bh, stride_before_b_c, stride_before_b_k, stride_before_b_v,
        stride_out_bh, stride_out_c, stride_out_k, stride_out_v,
        num_chunks, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        pid_bhc = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_v = tl.program_id(2)

        bh = pid_bhc // num_chunks
        chunk_idx = pid_bhc % num_chunks

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < key_dim
        mask_v = offs_v < value_dim
        mask_kv = mask_k[:, None] & mask_v[None, :]

        after_b_ptrs = (
            after_b_ptr + bh * stride_after_b_bh + chunk_idx * stride_after_b_c
            + offs_k[:, None] * stride_after_b_k + offs_v[None, :] * stride_after_b_v
        )
        acc = tl.load(after_b_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
        for k0 in range(0, key_dim, BLOCK_INNER):
            offs_inner = k0 + tl.arange(0, BLOCK_INNER)
            mask_inner = offs_inner < key_dim
            after_a_ptrs = (
                after_a_ptr + bh * stride_a_bh + chunk_idx * stride_a_c
                + offs_k[:, None] * stride_a_r + offs_inner[None, :] * stride_a_col
            )
            before_b_ptrs = (
                before_b_ptr + bh * stride_before_b_bh + chunk_idx * stride_before_b_c
                + offs_inner[:, None] * stride_before_b_k + offs_v[None, :] * stride_before_b_v
            )
            after_a = tl.load(after_a_ptrs, mask=mask_k[:, None] & mask_inner[None, :], other=0.0).to(tl.float32)
            before_b = tl.load(before_b_ptrs, mask=mask_inner[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(after_a, before_b)

        out_ptrs = (
            out_b_ptr + bh * stride_out_bh + chunk_idx * stride_out_c
            + offs_k[:, None] * stride_out_k + offs_v[None, :] * stride_out_v
        )
        tl.store(out_ptrs, acc, mask=mask_kv)


    @triton.jit
    def _chunk_state_projection_kernel(
        query_ptr, key_cumdecay_ptr, value_ptr, gate_ptr, state_ptr,
        value_new_ptr, attn_inter_ptr,
        stride_q_bh, stride_q_t, stride_q_k,
        stride_kc_bh, stride_kc_t, stride_kc_k,
        stride_v_bh, stride_v_t, stride_v_v,
        stride_g_bh, stride_g_t,
        stride_s_bh, stride_s_k, stride_s_v,
        stride_vn_bh, stride_vn_t, stride_vn_v,
        stride_ai_bh, stride_ai_t, stride_ai_v,
        chunk_len, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        for t in range(BLOCK_T):
            valid_t = t < chunk_len
            gate = tl.load(
                gate_ptr + bh * stride_g_bh + t * stride_g_t,
                mask=valid_t,
                other=0.0,
            ).to(tl.float32)
            gate_exp = tl.exp(gate)

            value_prime = tl.zeros([BLOCK_V], dtype=tl.float32)
            attn_inter = tl.zeros([BLOCK_V], dtype=tl.float32)

            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim
                state_ptrs = (
                    state_ptr + bh * stride_s_bh
                    + offs_k[:, None] * stride_s_k
                    + offs_v[None, :] * stride_s_v
                )
                state = tl.load(
                    state_ptrs,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0.0,
                ).to(tl.float32)
                key_cum = tl.load(
                    key_cumdecay_ptr + bh * stride_kc_bh + t * stride_kc_t + offs_k * stride_kc_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                query = tl.load(
                    query_ptr + bh * stride_q_bh + t * stride_q_t + offs_k * stride_q_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)

                value_prime += tl.sum(state * key_cum[:, None], axis=0)
                attn_inter += tl.sum(state * (query * gate_exp)[:, None], axis=0)

            value = tl.load(
                value_ptr + bh * stride_v_bh + t * stride_v_t + offs_v * stride_v_v,
                mask=mask_v & valid_t,
                other=0.0,
            ).to(tl.float32)
            value_new = value - value_prime

            tl.store(
                value_new_ptr + bh * stride_vn_bh + t * stride_vn_t + offs_v * stride_vn_v,
                value_new,
                mask=mask_v & valid_t,
            )
            tl.store(
                attn_inter_ptr + bh * stride_ai_bh + t * stride_ai_t + offs_v * stride_ai_v,
                attn_inter,
                mask=mask_v & valid_t,
            )


    @triton.jit
    def _chunk_state_update_kernel(
        key_ptr, gate_ptr, value_new_ptr, state_ptr,
        stride_k_bh, stride_k_t, stride_k_k,
        stride_g_bh, stride_g_t,
        stride_vn_bh, stride_vn_t, stride_vn_v,
        stride_s_bh, stride_s_k, stride_s_v,
        chunk_len, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        gate_last = tl.load(
            gate_ptr + bh * stride_g_bh + (chunk_len - 1) * stride_g_t,
        ).to(tl.float32)
        gate_last_exp = tl.exp(gate_last)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            state_ptrs = (
                state_ptr + bh * stride_s_bh
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            new_state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32) * gate_last_exp

            for t in range(BLOCK_T):
                valid_t = t < chunk_len
                gate_t = tl.load(
                    gate_ptr + bh * stride_g_bh + t * stride_g_t,
                    mask=valid_t,
                    other=0.0,
                ).to(tl.float32)
                decay = tl.exp(gate_last - gate_t)
                key = tl.load(
                    key_ptr + bh * stride_k_bh + t * stride_k_t + offs_k * stride_k_k,
                    mask=mask_k & valid_t,
                    other=0.0,
                ).to(tl.float32)
                value_new = tl.load(
                    value_new_ptr + bh * stride_vn_bh + t * stride_vn_t + offs_v * stride_vn_v,
                    mask=mask_v & valid_t,
                    other=0.0,
                ).to(tl.float32)
                new_state += (key * decay)[:, None] * value_new[None, :]

            tl.store(
                state_ptrs,
                new_state,
                mask=mask_k[:, None] & mask_v[None, :],
            )


    @triton.jit
    def _recurrent_delta_decode_kernel(
        query_ptr, key_ptr, value_ptr,
        beta_ptr, gate_ptr,
        state_ptr, out_ptr,
        stride_q_b, stride_q_h, stride_q_k,
        stride_k_b, stride_k_h, stride_k_k,
        stride_v_b, stride_v_h, stride_v_v,
        stride_beta_b, stride_beta_h,
        stride_gate_b, stride_gate_h,
        stride_s_b, stride_s_h, stride_s_k, stride_s_v,
        stride_o_b, stride_o_h, stride_o_v,
        num_v_heads, num_kv_groups,
        query_scale, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NORMALIZE_QK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)
        batch_idx = bh // num_v_heads
        v_head_idx = bh % num_v_heads
        k_head_idx = v_head_idx // num_kv_groups

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        gate = tl.load(
            gate_ptr + batch_idx * stride_gate_b + v_head_idx * stride_gate_h
        ).to(tl.float32)
        beta = tl.load(
            beta_ptr + batch_idx * stride_beta_b + v_head_idx * stride_beta_h
        ).to(tl.float32)
        gate_exp = tl.exp(gate)
        q_inv_norm = 1.0
        k_inv_norm = 1.0
        if NORMALIZE_QK:
            q_sumsq = 0.0
            k_sumsq = 0.0
            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim
                key_norm = tl.load(
                    key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                query_norm = tl.load(
                    query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + offs_k * stride_q_k,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                k_sumsq += tl.sum(key_norm * key_norm, axis=0)
                q_sumsq += tl.sum(query_norm * query_norm, axis=0)
            k_inv_norm = tl.rsqrt(k_sumsq + 1e-6)
            q_inv_norm = tl.rsqrt(q_sumsq + 1e-6)

        kv_mem = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            state_ptrs = (
                state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            key = tl.load(
                key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE_QK:
                key = key * k_inv_norm
            kv_mem += tl.sum((state * gate_exp) * key[:, None], axis=0)

        value = tl.load(
            value_ptr + batch_idx * stride_v_b + v_head_idx * stride_v_h + offs_v * stride_v_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        delta = (value - kv_mem) * beta

        out = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            state_ptrs = (
                state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            key = tl.load(
                key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            query = tl.load(
                query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + offs_k * stride_q_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE_QK:
                key = key * k_inv_norm
                query = query * q_inv_norm
            query = query * query_scale

            new_state = state * gate_exp + key[:, None] * delta[None, :]
            tl.store(
                state_ptrs,
                new_state,
                mask=mask_k[:, None] & mask_v[None, :],
            )
            out += tl.sum(new_state * query[:, None], axis=0)

        tl.store(
            out_ptr + batch_idx * stride_o_b + v_head_idx * stride_o_h + offs_v * stride_o_v,
            out,
            mask=mask_v,
        )


    @triton.jit
    def _recurrent_delta_decode_ab_kernel(
        query_ptr, key_ptr, value_ptr,
        a_ptr, b_ptr,
        a_log_ptr, dt_bias_ptr,
        state_ptr, out_ptr,
        stride_q_b, stride_q_h, stride_q_k,
        stride_k_b, stride_k_h, stride_k_k,
        stride_v_b, stride_v_h, stride_v_v,
        stride_a_b, stride_a_h,
        stride_b_b, stride_b_h,
        stride_s_b, stride_s_h, stride_s_k, stride_s_v,
        stride_o_b, stride_o_h, stride_o_v,
        num_v_heads, num_kv_groups,
        query_scale, key_dim, value_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NORMALIZE_QK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pid_v = tl.program_id(1)
        batch_idx = bh // num_v_heads
        v_head_idx = bh % num_v_heads
        k_head_idx = v_head_idx // num_kv_groups

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < value_dim

        a = tl.load(a_ptr + batch_idx * stride_a_b + v_head_idx * stride_a_h).to(tl.float32)
        b = tl.load(b_ptr + batch_idx * stride_b_b + v_head_idx * stride_b_h).to(tl.float32)
        a_log = tl.load(a_log_ptr + v_head_idx).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + v_head_idx).to(tl.float32)

        beta = 1.0 / (1.0 + tl.exp(-b))
        x = a + dt_bias
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        gate = -tl.exp(a_log) * softplus_x
        gate_exp = tl.exp(gate)

        q_inv_norm = 1.0
        k_inv_norm = 1.0
        if NORMALIZE_QK:
            q_sumsq = 0.0
            k_sumsq = 0.0
            for k0 in range(0, key_dim, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < key_dim
                key_norm = tl.load(
                    key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                query_norm = tl.load(
                    query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + offs_k * stride_q_k,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                k_sumsq += tl.sum(key_norm * key_norm, axis=0)
                q_sumsq += tl.sum(query_norm * query_norm, axis=0)
            k_inv_norm = tl.rsqrt(k_sumsq + 1e-6)
            q_inv_norm = tl.rsqrt(q_sumsq + 1e-6)

        kv_mem = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            state_ptrs = (
                state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            key = tl.load(
                key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE_QK:
                key = key * k_inv_norm
            kv_mem += tl.sum((state * gate_exp) * key[:, None], axis=0)

        value = tl.load(
            value_ptr + batch_idx * stride_v_b + v_head_idx * stride_v_h + offs_v * stride_v_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        delta = (value - kv_mem) * beta

        out = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k0 in range(0, key_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < key_dim

            state_ptrs = (
                state_ptr + batch_idx * stride_s_b + v_head_idx * stride_s_h
                + offs_k[:, None] * stride_s_k
                + offs_v[None, :] * stride_s_v
            )
            state = tl.load(
                state_ptrs,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            key = tl.load(
                key_ptr + batch_idx * stride_k_b + k_head_idx * stride_k_h + offs_k * stride_k_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            query = tl.load(
                query_ptr + batch_idx * stride_q_b + k_head_idx * stride_q_h + offs_k * stride_q_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            if NORMALIZE_QK:
                key = key * k_inv_norm
                query = query * q_inv_norm
            query = query * query_scale

            new_state = state * gate_exp + key[:, None] * delta[None, :]
            tl.store(
                state_ptrs,
                new_state,
                mask=mask_k[:, None] & mask_v[None, :],
            )
            out += tl.sum(new_state * query[:, None], axis=0)

        tl.store(
            out_ptr + batch_idx * stride_o_b + v_head_idx * stride_o_h + offs_v * stride_o_v,
            out,
            mask=mask_v,
        )


def recurrent_gated_delta_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    num_kv_groups: int = 1,
    query_scale: float = 1.0,
    normalize_qk: bool = False,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Triton recurrent delta-rule decode for one token.

    Args:
        query/key: [batch, q_heads, key_dim]
        value: [batch, kv_heads, value_dim]
        gate/beta: [batch, kv_heads]
        state: [batch, kv_heads, key_dim, value_dim] float32, updated in-place

    Returns:
        out: [batch, kv_heads, value_dim], float32 by default. Pass
             output_dtype for decode callers that would otherwise cast
             immediately after the kernel.
    """
    if not (_HAS_TRITON and query.is_cuda and key.is_cuda and value.is_cuda and state.is_cuda):
        raise RuntimeError("recurrent_gated_delta_decode requires Triton + CUDA tensors")

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or state.ndim != 4:
        raise ValueError("Invalid tensor ranks for recurrent_gated_delta_decode")

    batch, q_heads, key_dim = query.shape
    kv_heads = value.shape[1]
    value_dim = value.shape[-1]
    _validate_qkv_group_shapes(q_heads, kv_heads, num_kv_groups, "recurrent_gated_delta_decode")
    out_dtype = torch.float32 if output_dtype is None else output_dtype
    out_shape = (batch, kv_heads, value_dim)
    if out is None:
        out = torch.empty(out_shape, device=query.device, dtype=out_dtype)
    elif tuple(out.shape) != out_shape or out.device != query.device or out.dtype != out_dtype:
        raise ValueError(
            "out must match recurrent_gated_delta_decode output shape/device/dtype"
        )

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    grid = (batch * kv_heads, triton.cdiv(value_dim, block_v))

    _recurrent_delta_decode_kernel[grid](
        query, key, value,
        beta, gate,
        state, out,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        beta.stride(0), beta.stride(1),
        gate.stride(0), gate.stride(1),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        kv_heads, num_kv_groups,
        query_scale, key_dim, value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        NORMALIZE_QK=normalize_qk,
        num_warps=4,
    )

    return out


def recurrent_gated_delta_decode_from_ab(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    num_kv_groups: int = 1,
    query_scale: float = 1.0,
    normalize_qk: bool = False,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Triton recurrent delta-rule decode with gate/beta computed in-kernel.

    Args:
        query/key: [batch, q_heads, key_dim]
        value: [batch, kv_heads, value_dim]
        a/b: [batch, kv_heads] raw gate vectors
        a_log/dt_bias: [kv_heads]
        state: [batch, kv_heads, key_dim, value_dim] float32, updated in-place

    Returns:
        out: [batch, kv_heads, value_dim], float32 by default. Pass
             output_dtype for decode callers that would otherwise cast
             immediately after the kernel.
    """
    if not (_HAS_TRITON and query.is_cuda and key.is_cuda and value.is_cuda and state.is_cuda):
        raise RuntimeError("recurrent_gated_delta_decode_from_ab requires Triton + CUDA tensors")

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or state.ndim != 4:
        raise ValueError("Invalid tensor ranks for recurrent_gated_delta_decode_from_ab")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a/b must be [batch, kv_heads] for recurrent_gated_delta_decode_from_ab")

    batch, q_heads, key_dim = query.shape
    kv_heads = value.shape[1]
    value_dim = value.shape[-1]
    _validate_qkv_group_shapes(q_heads, kv_heads, num_kv_groups, "recurrent_gated_delta_decode_from_ab")
    if a.shape[0] != batch or a.shape[1] != kv_heads or b.shape != a.shape:
        raise ValueError("a/b shape must match [batch, kv_heads]")

    out_dtype = torch.float32 if output_dtype is None else output_dtype
    out_shape = (batch, kv_heads, value_dim)
    if out is None:
        out = torch.empty(out_shape, device=query.device, dtype=out_dtype)
    elif tuple(out.shape) != out_shape or out.device != query.device or out.dtype != out_dtype:
        raise ValueError(
            "out must match recurrent_gated_delta_decode_from_ab output shape/device/dtype"
        )
    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    grid = (batch * kv_heads, triton.cdiv(value_dim, block_v))

    _recurrent_delta_decode_ab_kernel[grid](
        query, key, value,
        a, b, a_log, dt_bias,
        state, out,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        kv_heads, num_kv_groups,
        query_scale, key_dim, value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        NORMALIZE_QK=normalize_qk,
        num_warps=4,
    )

    return out


def recurrent_gated_delta_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    num_kv_groups: int = 1,
    query_scale: float = 1.0,
    normalize_qk: bool = False,
) -> torch.Tensor:
    """
    Triton recurrent delta-rule prefill for short sequences.

    Args:
        query/key: [batch, q_heads, seq_len, key_dim]
        value: [batch, kv_heads, seq_len, value_dim]
        gate/beta: [batch, kv_heads, seq_len]
        state: [batch, kv_heads, key_dim, value_dim] float32, updated in-place

    Returns:
        out: [batch, kv_heads, seq_len, value_dim] float32
    """
    if not (_HAS_TRITON and query.is_cuda and key.is_cuda and value.is_cuda and state.is_cuda):
        raise RuntimeError("recurrent_gated_delta_prefill requires Triton + CUDA tensors")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or state.ndim != 4:
        raise ValueError("Invalid tensor ranks for recurrent_gated_delta_prefill")

    batch, q_heads, seq_len, key_dim = query.shape
    kv_heads = value.shape[1]
    value_dim = value.shape[-1]
    _validate_qkv_group_shapes(q_heads, kv_heads, num_kv_groups, "recurrent_gated_delta_prefill")
    if seq_len > 64:
        raise ValueError(f"Unsupported seq_len for recurrent_gated_delta_prefill: {seq_len}")

    out = torch.empty((batch, kv_heads, seq_len, value_dim), device=query.device, dtype=torch.float32)

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    block_t = 64
    grid = (batch * kv_heads, triton.cdiv(value_dim, block_v))

    _recurrent_delta_prefill_kernel[grid](
        query, key, value,
        beta, gate,
        state, out,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        gate.stride(0), gate.stride(1), gate.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len, kv_heads, num_kv_groups, query_scale, key_dim, value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        BLOCK_T=block_t,
        NORMALIZE_QK=normalize_qk,
        num_warps=4,
    )
    return out


def chunk_interchunk(
    query: torch.Tensor,
    key: torch.Tensor,
    key_cumdecay: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _HAS_TRITON_CHUNK_INTERCHUNK_FUSED

    chunk_len = query.shape[-2]
    if not (_HAS_TRITON_CHUNK_INTERCHUNK_FUSED and query.is_cuda and chunk_len <= _MAX_TRITON_CHUNK_LEN):
        value_new, attn_inter = chunk_state_projection(query, key_cumdecay, value, gate, state)
        state = chunk_state_update(key, gate, value_new, state)
        return value_new, attn_inter, state

    batch, heads, _, key_dim = query.shape
    value_dim = value.shape[-1]
    bh = batch * heads
    query_3d = _flatten_bh_view_4d(query)
    key_3d = _flatten_bh_view_4d(key)
    key_cumdecay_3d = _flatten_bh_view_4d(key_cumdecay)
    value_3d = _flatten_bh_view_4d(value)
    gate_2d = _flatten_bh_view_3d(gate)
    state_3d = _flatten_bh_view_4d(state)
    value_new = torch.empty((bh, chunk_len, value_dim), device=query.device, dtype=torch.float32)
    attn_inter = torch.empty_like(value_new)

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    block_t = min(triton.next_power_of_2(chunk_len), _MAX_TRITON_CHUNK_LEN)
    grid = (bh, triton.cdiv(value_dim, block_v))

    try:
        _chunk_interchunk_kernel[grid](
            query_3d, key_3d, key_cumdecay_3d, value_3d, gate_2d, state_3d,
            value_new, attn_inter,
            query_3d.stride(0), query_3d.stride(1), query_3d.stride(2),
            key_3d.stride(0), key_3d.stride(1), key_3d.stride(2),
            key_cumdecay_3d.stride(0), key_cumdecay_3d.stride(1), key_cumdecay_3d.stride(2),
            value_3d.stride(0), value_3d.stride(1), value_3d.stride(2),
            gate_2d.stride(0), gate_2d.stride(1),
            state_3d.stride(0), state_3d.stride(1), state_3d.stride(2),
            value_new.stride(0), value_new.stride(1), value_new.stride(2),
            attn_inter.stride(0), attn_inter.stride(1), attn_inter.stride(2),
            chunk_len, key_dim, value_dim,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return (
            value_new.view(batch, heads, chunk_len, value_dim),
            attn_inter.view(batch, heads, chunk_len, value_dim),
            state,
        )
    except Exception:
        if chunk_len <= 64:
            _HAS_TRITON_CHUNK_INTERCHUNK_FUSED = False
        value_new, attn_inter = chunk_state_projection(query, key_cumdecay, value, gate, state)
        state = chunk_state_update(key, gate, value_new, state)
        return value_new, attn_inter, state


def chunk_interchunk_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    key_cumdecay: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _HAS_TRITON_CHUNK_INTERCHUNK_SCAN

    num_chunks = query.shape[-3]
    chunk_len = query.shape[-2]
    key_dim = query.shape[-1]
    parallel_scan = _parallel_scan_allowed(query, num_chunks, key_dim)
    if parallel_scan and num_chunks > 1 and chunk_len <= _MAX_TRITON_CHUNK_LEN:
        try:
            return _chunk_interchunk_scan_parallel_torch(
                query,
                key,
                key_cumdecay,
                value,
                gate,
                state,
            )
        except Exception as exc:
            if _debug_linear_attn_enabled():
                warnings.warn(
                    f"parallel scan fallback to legacy path: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                )

    value_dim = value.shape[-1]
    max_scan_chunks, scan_num_warps = _resolve_chunk_scan_launch_policy(
        query,
        num_chunks,
        key_dim,
        value_dim,
    )
    scan_window_chunks = _choose_scan_window_chunks(num_chunks, max_scan_chunks)
    tail_fallback_chunks = _get_env_int(
        "MEGAGEMM_QWEN35_SCAN_TAIL_FALLBACK_CHUNKS",
        4,
        min_value=0,
        max_value=32,
    )
    use_triton_scan = (
        _HAS_TRITON_CHUNK_INTERCHUNK_SCAN
        and query.is_cuda
        and chunk_len <= _MAX_TRITON_CHUNK_LEN
    )
    if not use_triton_scan:
        value_news = torch.empty_like(value)
        attn_inters = torch.empty_like(value)
        for idx in range(num_chunks):
            value_news[:, :, idx], attn_inters[:, :, idx], state = chunk_interchunk(
                query[:, :, idx],
                key[:, :, idx],
                key_cumdecay[:, :, idx],
                value[:, :, idx],
                gate[:, :, idx],
                state,
            )
        return value_news, attn_inters, state

    batch, heads, _, _, key_dim = query.shape
    value_dim = value.shape[-1]
    bh = batch * heads
    query_4d = _flatten_bh_view_5d(query)
    key_4d = _flatten_bh_view_5d(key)
    key_cumdecay_4d = _flatten_bh_view_5d(key_cumdecay)
    value_4d = _flatten_bh_view_5d(value)
    gate_3d = _flatten_bh_view_4d(gate)
    state_3d = _flatten_bh_view_4d(state)
    value_new = torch.empty((bh, num_chunks, chunk_len, value_dim), device=query.device, dtype=torch.float32)
    attn_inter = torch.empty_like(value_new)
    value_new_5d = value_new.view(batch, heads, num_chunks, chunk_len, value_dim)
    attn_inter_5d = attn_inter.view(batch, heads, num_chunks, chunk_len, value_dim)

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    block_t = min(triton.next_power_of_2(chunk_len), _MAX_TRITON_CHUNK_LEN)
    grid = (bh, triton.cdiv(value_dim, block_v))

    try:
        if num_chunks <= scan_window_chunks:
            block_c = _bucket_scan_block_c(num_chunks)
            _chunk_interchunk_scan_kernel[grid](
                query_4d, key_4d, key_cumdecay_4d, value_4d, gate_3d, state_3d,
                value_new, attn_inter,
                query_4d.stride(0), query_4d.stride(1), query_4d.stride(2), query_4d.stride(3),
                key_4d.stride(0), key_4d.stride(1), key_4d.stride(2), key_4d.stride(3),
                key_cumdecay_4d.stride(0), key_cumdecay_4d.stride(1), key_cumdecay_4d.stride(2), key_cumdecay_4d.stride(3),
                value_4d.stride(0), value_4d.stride(1), value_4d.stride(2), value_4d.stride(3),
                gate_3d.stride(0), gate_3d.stride(1), gate_3d.stride(2),
                state_3d.stride(0), state_3d.stride(1), state_3d.stride(2),
                value_new.stride(0), value_new.stride(1), value_new.stride(2), value_new.stride(3),
                attn_inter.stride(0), attn_inter.stride(1), attn_inter.stride(2), attn_inter.stride(3),
                num_chunks, chunk_len, key_dim, value_dim,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                num_warps=scan_num_warps,
            )
            return (
                value_new_5d,
                attn_inter_5d,
                state,
            )

        # Run scan kernel in windows, carrying recurrent state across windows.
        for chunk_start in range(0, num_chunks, scan_window_chunks):
            chunk_end = min(chunk_start + scan_window_chunks, num_chunks)
            chunk_count = chunk_end - chunk_start
            is_tail_window = chunk_end == num_chunks

            if is_tail_window and tail_fallback_chunks > 0 and chunk_count <= tail_fallback_chunks:
                for idx in range(chunk_start, chunk_end):
                    value_new_5d[:, :, idx], attn_inter_5d[:, :, idx], state = chunk_interchunk(
                        query[:, :, idx],
                        key[:, :, idx],
                        key_cumdecay[:, :, idx],
                        value[:, :, idx],
                        gate[:, :, idx],
                        state,
                    )
                continue

            block_c = _bucket_scan_block_c(chunk_count)
            state_3d = _flatten_bh_view_4d(state)

            query_win = _flatten_bh_view_5d(query[:, :, chunk_start:chunk_end])
            key_win = _flatten_bh_view_5d(key[:, :, chunk_start:chunk_end])
            key_cumdecay_win = _flatten_bh_view_5d(key_cumdecay[:, :, chunk_start:chunk_end])
            value_win = _flatten_bh_view_5d(value[:, :, chunk_start:chunk_end])
            gate_win = _flatten_bh_view_4d(gate[:, :, chunk_start:chunk_end])
            value_new_win = _flatten_bh_view_5d(value_new_5d[:, :, chunk_start:chunk_end])
            attn_inter_win = _flatten_bh_view_5d(attn_inter_5d[:, :, chunk_start:chunk_end])

            _chunk_interchunk_scan_kernel[grid](
                query_win, key_win, key_cumdecay_win, value_win, gate_win, state_3d,
                value_new_win, attn_inter_win,
                query_win.stride(0), query_win.stride(1), query_win.stride(2), query_win.stride(3),
                key_win.stride(0), key_win.stride(1), key_win.stride(2), key_win.stride(3),
                key_cumdecay_win.stride(0), key_cumdecay_win.stride(1), key_cumdecay_win.stride(2), key_cumdecay_win.stride(3),
                value_win.stride(0), value_win.stride(1), value_win.stride(2), value_win.stride(3),
                gate_win.stride(0), gate_win.stride(1), gate_win.stride(2),
                state_3d.stride(0), state_3d.stride(1), state_3d.stride(2),
                value_new_win.stride(0), value_new_win.stride(1), value_new_win.stride(2), value_new_win.stride(3),
                attn_inter_win.stride(0), attn_inter_win.stride(1), attn_inter_win.stride(2), attn_inter_win.stride(3),
                chunk_count, chunk_len, key_dim, value_dim,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                num_warps=scan_num_warps,
            )

        return (
            value_new_5d,
            attn_inter_5d,
            state,
        )
    except Exception as exc:
        if chunk_len <= 64:
            _HAS_TRITON_CHUNK_INTERCHUNK_SCAN = False
        if _debug_linear_attn_enabled():
            warnings.warn(
                f"chunk_interchunk_scan fallback activated: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
        for idx in range(num_chunks):
            value_new_5d[:, :, idx], attn_inter_5d[:, :, idx], state = chunk_interchunk(
                query[:, :, idx],
                key[:, :, idx],
                key_cumdecay[:, :, idx],
                value[:, :, idx],
                gate[:, :, idx],
                state,
            )
        return (
            value_new_5d,
            attn_inter_5d,
            state,
        )


def chunk_state_projection(
    query: torch.Tensor,
    key_cumdecay: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _HAS_TRITON_CHUNK_INTERCHUNK

    chunk_len = query.shape[-2]
    if not (_HAS_TRITON_CHUNK_INTERCHUNK and query.is_cuda and chunk_len <= _MAX_TRITON_CHUNK_LEN):
        return _chunk_state_projection_fallback(query, key_cumdecay, value, gate, state)

    batch, heads, _, key_dim = query.shape
    value_dim = value.shape[-1]
    bh = batch * heads
    query_3d = _flatten_bh_view_4d(query)
    key_cumdecay_3d = _flatten_bh_view_4d(key_cumdecay)
    value_3d = _flatten_bh_view_4d(value)
    gate_2d = _flatten_bh_view_3d(gate)
    state_3d = _flatten_bh_view_4d(state)
    value_new = torch.empty((bh, chunk_len, value_dim), device=query.device, dtype=torch.float32)
    attn_inter = torch.empty_like(value_new)

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    block_t = min(triton.next_power_of_2(chunk_len), _MAX_TRITON_CHUNK_LEN)
    grid = (bh, triton.cdiv(value_dim, block_v))

    try:
        _chunk_state_projection_kernel[grid](
            query_3d, key_cumdecay_3d, value_3d, gate_2d, state_3d,
            value_new, attn_inter,
            query_3d.stride(0), query_3d.stride(1), query_3d.stride(2),
            key_cumdecay_3d.stride(0), key_cumdecay_3d.stride(1), key_cumdecay_3d.stride(2),
            value_3d.stride(0), value_3d.stride(1), value_3d.stride(2),
            gate_2d.stride(0), gate_2d.stride(1),
            state_3d.stride(0), state_3d.stride(1), state_3d.stride(2),
            value_new.stride(0), value_new.stride(1), value_new.stride(2),
            attn_inter.stride(0), attn_inter.stride(1), attn_inter.stride(2),
            chunk_len, key_dim, value_dim,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return value_new.view(batch, heads, chunk_len, value_dim), attn_inter.view(batch, heads, chunk_len, value_dim)
    except Exception:
        if chunk_len <= 64:
            _HAS_TRITON_CHUNK_INTERCHUNK = False
        return _chunk_state_projection_fallback(query, key_cumdecay, value, gate, state)


def chunk_state_update(
    key: torch.Tensor,
    gate: torch.Tensor,
    value_new: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    global _HAS_TRITON_CHUNK_INTERCHUNK

    chunk_len = key.shape[-2]
    if not (_HAS_TRITON_CHUNK_INTERCHUNK and key.is_cuda and chunk_len <= _MAX_TRITON_CHUNK_LEN):
        return _chunk_state_update_fallback(key, gate, value_new, state)

    batch, heads, _, key_dim = key.shape
    value_dim = value_new.shape[-1]
    bh = batch * heads
    key_3d = _flatten_bh_view_4d(key)
    gate_2d = _flatten_bh_view_3d(gate)
    value_new_3d = _flatten_bh_view_4d(value_new)
    state_3d = _flatten_bh_view_4d(state)

    block_k = min(triton.next_power_of_2(key_dim), 128)
    block_v = 32 if value_dim <= 32 else 64
    block_t = min(triton.next_power_of_2(chunk_len), _MAX_TRITON_CHUNK_LEN)
    grid = (bh, triton.cdiv(value_dim, block_v))

    try:
        _chunk_state_update_kernel[grid](
            key_3d, gate_2d, value_new_3d, state_3d,
            key_3d.stride(0), key_3d.stride(1), key_3d.stride(2),
            gate_2d.stride(0), gate_2d.stride(1),
            value_new_3d.stride(0), value_new_3d.stride(1), value_new_3d.stride(2),
            state_3d.stride(0), state_3d.stride(1), state_3d.stride(2),
            chunk_len, key_dim, value_dim,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return state
    except Exception:
        if chunk_len <= 64:
            _HAS_TRITON_CHUNK_INTERCHUNK = False
        return _chunk_state_update_fallback(key, gate, value_new, state)


def _chunk_state_projection_fallback(
    query: torch.Tensor,
    key_cumdecay: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_prime = key_cumdecay @ state
    value_new = value - value_prime
    attn_inter = (query * gate.exp().unsqueeze(-1)) @ state
    return value_new, attn_inter


def _chunk_state_update_fallback(
    key: torch.Tensor,
    gate: torch.Tensor,
    value_new: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    gate_last = gate[..., -1, None, None].exp()
    state.mul_(gate_last)
    state.add_(
        (key * (gate[..., -1, None] - gate).exp().unsqueeze(-1)).transpose(-1, -2) @ value_new
    )
    return state


def solve_chunk_local_attention(attn: torch.Tensor) -> torch.Tensor:
    """
    In-place Triton solve for the chunk-local lower-triangular recurrence.

    Args:
        attn: [..., chunk_size, chunk_size] float32 tensor where the strictly
            lower-triangular entries contain the base local attention terms and
            the upper triangle / diagonal are zero.

    Returns:
        The same tensor with the recurrence solved and identity added.
    """
    global _HAS_TRITON_CHUNK_SOLVE

    if attn.ndim < 2 or attn.shape[-1] != attn.shape[-2]:
        raise ValueError("attn must have shape [..., chunk_size, chunk_size]")

    chunk_size = attn.shape[-1]
    if chunk_size > _MAX_TRITON_CHUNK_LEN:
        raise ValueError(f"Unsupported chunk_size for Triton solve: {chunk_size}")

    attn_3d = attn.contiguous().view(-1, chunk_size, chunk_size)
    if not (_HAS_TRITON_CHUNK_SOLVE and attn_3d.is_cuda):
        return _solve_chunk_local_attention_fallback(attn_3d).view_as(attn)

    block = min(triton.next_power_of_2(chunk_size), _MAX_TRITON_CHUNK_LEN)
    grid = (attn_3d.shape[0],)

    try:
        _chunk_local_attention_solve_kernel[grid](
            attn_3d,
            attn_3d.stride(0), attn_3d.stride(1), attn_3d.stride(2),
            CHUNK_SIZE=chunk_size,
            BLOCK=block,
            num_warps=8 if block > 64 else 4,
        )
        return attn_3d.view_as(attn)
    except Exception:
        if chunk_size <= 64:
            _HAS_TRITON_CHUNK_SOLVE = False
        return _solve_chunk_local_attention_fallback(attn_3d).view_as(attn)


def _solve_chunk_local_attention_fallback(attn: torch.Tensor) -> torch.Tensor:
    chunk_size = attn.shape[-1]
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    diag = torch.arange(chunk_size, device=attn.device)
    attn[..., diag, diag] += 1.0
    return attn


HAS_TRITON_LINEAR_ATTN = _HAS_TRITON
