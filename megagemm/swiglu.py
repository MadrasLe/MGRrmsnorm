"""
SwiGLU Triton Module
--------------------
High-performance fused SwiGLU activation with Triton.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _mg_swiglu_fwd_kernel(
    input_ptr,       # [Batch*Seq, 2*H] (Gate + Value contíguos)
    output_ptr,      # [Batch*Seq, H]
    n_cols_half,     # H (Hidden Dim)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    row_input_ptr = input_ptr + pid * (2 * n_cols_half)
    row_output_ptr = output_ptr + pid * n_cols_half
    
    for off in range(0, n_cols_half, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols_half
        
        # Load in FP32 for stability
        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input_ptr + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Fused SiLU(gate) * val
        gate_sig = tl.sigmoid(gate)
        gate_silu = gate * gate_sig
        out = gate_silu * val
        
        tl.store(row_output_ptr + offsets, out, mask=mask)


@triton.jit
def _mg_swiglu_bwd_kernel(
    grad_out_ptr,    # [M, H]
    input_ptr,       # [M, 2H]
    grad_input_ptr,  # [M, 2H]
    n_cols_half,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    row_grad_out = grad_out_ptr + pid * n_cols_half
    row_input = input_ptr + pid * (2 * n_cols_half)
    row_grad_input = grad_input_ptr + pid * (2 * n_cols_half)
    
    for off in range(0, n_cols_half, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols_half
        
        g_out = tl.load(row_grad_out + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Recompute activation
        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate
        
        # Gradients
        d_val = g_out * silu_gate
        term = 1.0 + gate * (1.0 - sig_gate)
        d_silu = sig_gate * term
        d_gate = g_out * val * d_silu
        
        tl.store(row_grad_input + offsets, d_gate, mask=mask)
        tl.store(row_grad_input + n_cols_half + offsets, d_val, mask=mask)


# =============================================================================
# Autograd Function
# =============================================================================

class MegaGemmFunction(Function):
    """Autograd function for fused SwiGLU Triton kernel."""
    
    @staticmethod
    def forward(ctx, w12_out: torch.Tensor, hidden_dim: int):
        w12_out = w12_out.contiguous()
        x_flat = w12_out.view(-1, 2 * hidden_dim)
        M = x_flat.shape[0]
        
        out_flat = torch.empty((M, hidden_dim), device=w12_out.device, dtype=w12_out.dtype)
        
        ctx.save_for_backward(w12_out)
        ctx.hidden_dim = hidden_dim
        
        grid = (M,)
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)
        
        _mg_swiglu_fwd_kernel[grid](
            x_flat, out_flat,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out_flat.view(w12_out.shape[:-1] + (hidden_dim,))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w12_out, = ctx.saved_tensors
        hidden_dim = ctx.hidden_dim
        
        grad_out_flat = grad_output.contiguous().view(-1, hidden_dim)
        x_flat = w12_out.contiguous().view(-1, 2 * hidden_dim)
        M = x_flat.shape[0]
        
        grad_input_flat = torch.empty_like(x_flat)
        
        grid = (M,)
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)
        
        _mg_swiglu_bwd_kernel[grid](
            grad_out_flat, x_flat, grad_input_flat,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_input_flat.view(w12_out.shape), None


# =============================================================================
# NN Module
# =============================================================================

class MegaGemmTriton(nn.Module):
    """
    Mega Gemm Triton: High-performance fused SwiGLU.
    
    Drop-in replacement for standard SwiGLU (gate + value activation) with
    fused Triton kernel that avoids memory copies.
    
    Args:
        d_model (int): Input/output dimension.
        multiple_of (int): Ensure hidden dim is multiple of this (default 256).
        hidden_multiple (float): Expansion factor (default 5/3 like LLaMA).
    
    Example:
        >>> swiglu = MegaGemmTriton(4096).cuda()
        >>> x = torch.randn(32, 128, 4096, device='cuda')
        >>> y = swiglu(x)  # [32, 128, 4096]
    """
    
    def __init__(self, d_model: int, multiple_of: int = 256, hidden_multiple: float = 5/3):
        super().__init__()
        hidden = int(d_model * hidden_multiple)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        
        # Fused W1+W2 for better Tensor Core utilization
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w12_out = self.w12(x)
        hidden = MegaGemmFunction.apply(w12_out, self.hidden)
        return self.w3(hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.w12.in_features}, hidden={self.hidden} (Triton Accelerated)"
