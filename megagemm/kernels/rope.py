"""
🔄 RoPE (Rotary Position Embeddings) for MegaGemm
Autor: Gabriel Yogi
Descrição: High-performance RoPE implementation with full backward support

Reference: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
Used in: LLaMA, Mistral, GPT-NeoX, etc.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Triton RoPE disabled — broken on all GPUs with Triton 3.6.0.
# PyTorch RoPE below works fine (same as v0.5).
_HAS_TRITON_ROPE = False

__all__ = [
    'RoPE',
    'precompute_freqs_cis',
    'precompute_proportional_freqs_cis',
    'apply_rotary_emb',
]


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin frequencies for RoPE.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for frequency computation (default 10000)
        device: Device to place tensors
        dtype: Data type for computation

    Returns:
        cos, sin: Tensors of shape [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
    # Shape: [dim//2]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))

    # Compute position indices: [0, 1, 2, ..., max_seq_len-1]
    # Shape: [max_seq_len]
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)

    # Outer product: positions × inv_freq -> angles
    # Shape: [max_seq_len, dim//2]
    angles = torch.outer(positions, inv_freq)

    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def precompute_proportional_freqs_cis(
    dim: int,
    max_seq_len: int,
    partial_rotary_factor: float,
    base: float = 10000.0,
    factor: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute proportional RoPE tables.

    HuggingFace's Gemma 4 ``proportional`` RoPE returns a table for the full
    head dimension. Only the first ``partial_rotary_factor`` fraction receives
    non-zero frequencies; the rest is zero-frequency, which makes RoPE an
    identity transform for those channels while preserving full-head shapes.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    if factor == 0:
        raise ValueError("factor must be non-zero for proportional RoPE")

    rope_angles = int(partial_rotary_factor * dim // 2)
    rope_angles = max(0, min(dim // 2, rope_angles))
    if rope_angles > 0:
        inv_freq_rotated = 1.0 / (
            base ** (
                torch.arange(0, 2 * rope_angles, 2, device=device, dtype=dtype)
                / dim
            )
        )
    else:
        inv_freq_rotated = torch.empty(0, device=device, dtype=dtype)

    nope_angles = dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            [
                inv_freq_rotated,
                torch.zeros(nope_angles, device=device, dtype=dtype),
            ],
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated
    inv_freq = inv_freq / factor

    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    half_rotate: bool = False,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K tensors.

    Uses Triton fused kernel when available (1 launch vs 8+ PyTorch ops),
    falls back to PyTorch otherwise.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Precomputed cos [max_seq_len, head_dim//2]
        sin: Precomputed sin [max_seq_len, head_dim//2]
        position_ids: Optional position indices [batch, seq_len]
        half_rotate: True for Hugging Face-converted LLaMA/Mistral/Qwen/Gemma
            weights (first/second-half ``rotate_half`` convention); False for
            interleaved even/odd rotary layouts.
        rotary_dim: Number of head dims that use RoPE. Remaining dims pass through.

    Returns:
        q_rotated, k_rotated: Rotated tensors with same shape as input
    """
    # Fast path: Triton fused kernel (1 launch vs 8+ PyTorch ops)
    head_dim = q.shape[3]
    rotary_dim = head_dim if rotary_dim is None else min(rotary_dim, head_dim)
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be a positive even number, got {rotary_dim}")

    if _HAS_TRITON_ROPE and q.is_cuda and rotary_dim == head_dim:
        return triton_apply_rotary_emb(q, k, cos, sin, position_ids, half_rotate)

    # Fallback: PyTorch implementation
    seq_len = q.shape[2]

    # Get cos/sin for current sequence positions
    if position_ids is not None:
        # Gather cos/sin based on position_ids
        cos = cos[position_ids]  # [batch, seq_len, head_dim//2]
        sin = sin[position_ids]
        # Add head dimension
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
        sin = sin.unsqueeze(1)
    else:
        # Use first seq_len positions
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Only the rotary subspace uses cos/sin tables.
    half_rotary = rotary_dim // 2
    cos = cos[..., :half_rotary].to(q.dtype)
    sin = sin[..., :half_rotary].to(q.dtype)

    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]
    q_tail = q[..., rotary_dim:]
    k_tail = k[..., rotary_dim:]

    if half_rotate:
        # HuggingFace style: pair first half with second half of head_dim
        # Used by: Qwen2, Qwen3, Gemma2, and other HF-native models
        half = rotary_dim // 2
        q_first, q_second = q_rot[..., :half], q_rot[..., half:]
        k_first, k_second = k_rot[..., :half], k_rot[..., half:]

        q_first_rot = q_first * cos - q_second * sin
        q_second_rot = q_second * cos + q_first * sin
        k_first_rot = k_first * cos - k_second * sin
        k_second_rot = k_second * cos + k_first * sin

        q_rotated = torch.cat([q_first_rot, q_second_rot], dim=-1)
        k_rotated = torch.cat([k_first_rot, k_second_rot], dim=-1)
    else:
        # Interleaved style (Meta LLaMA original): pair even/odd dims
        # Used by: LLaMA 2/3, TinyLlama, Mistral
        q_even = q_rot[..., 0::2]
        q_odd = q_rot[..., 1::2]
        k_even = k_rot[..., 0::2]
        k_odd = k_rot[..., 1::2]

        q_even_rot = q_even * cos - q_odd * sin
        q_odd_rot = q_even * sin + q_odd * cos
        k_even_rot = k_even * cos - k_odd * sin
        k_odd_rot = k_even * sin + k_odd * cos

        q_rotated = torch.stack([q_even_rot, q_odd_rot], dim=-1).flatten(-2)
        k_rotated = torch.stack([k_even_rot, k_odd_rot], dim=-1).flatten(-2)

    if rotary_dim == head_dim:
        return q_rotated, k_rotated

    return torch.cat([q_rotated, q_tail], dim=-1), torch.cat([k_rotated, k_tail], dim=-1)


class RoPE(nn.Module):
    """
    Rotary Position Embedding module.

    Drop-in replacement for position embeddings in transformer models.
    Caches cos/sin tables for efficiency.

    Example:
        rope = RoPE(head_dim=64, max_seq_len=2048)
        q, k = rope(q, k)  # Apply rotation
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        rotary_dim: Optional[int] = None,
        half_rotate: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        self.half_rotate = bool(half_rotate)

        # Precompute and register as buffers (not parameters)
        cos, sin = precompute_freqs_cis(self.rotary_dim, max_seq_len, base)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query [batch, heads, seq_len, head_dim]
            k: Key [batch, heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            q_rotated, k_rotated: Rotated tensors
        """
        seq_len = q.shape[2]

        # Extend cache if needed
        if seq_len > self.max_seq_len:
            cos, sin = precompute_freqs_cis(
                self.rotary_dim, seq_len, self.base,
                device=q.device, dtype=torch.float32
            )
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)
            self.max_seq_len = seq_len

        return apply_rotary_emb(
            q,
            k,
            self.cos,
            self.sin,
            position_ids,
            half_rotate=self.half_rotate,
            rotary_dim=self.rotary_dim,
        )

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"base={self.base}, half_rotate={self.half_rotate}"
        )
