"""
🔥 MEGA GEMM (MG Triton) - High Performance SwiGLU Library
Autor: Gabriel & Ada
Descrição: Implementação Fused SwiGLU com Triton, leitura de offset e suporte a BF16.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function

__all__ = ["MegaGemmTriton"]

# ============================================================================
# KERNELS (Backend)
# ============================================================================

@triton.jit
def _mg_swiglu_fwd_kernel(
    input_ptr,       # [Batch*Seq, 2*H] (Gate + Value contíguos)
    output_ptr,      # [Batch*Seq, H]
    n_cols_half,     # H (Hidden Dim)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Ponteiros com aritmética de offset para evitar views do PyTorch
    row_input_ptr = input_ptr + pid * (2 * n_cols_half)
    row_output_ptr = output_ptr + pid * n_cols_half
    
    for off in range(0, n_cols_half, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols_half
        
        # Leitura Direta: Gate em [0..H], Value em [H..2H]
        # .to(tl.float32) é mandatório para estabilidade numérica
        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input_ptr + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Fused Activation: SiLU(gate) * val
        gate_sig = tl.sigmoid(gate)
        gate_silu = gate * gate_sig
        out = gate_silu * val
        
        tl.store(row_output_ptr + offsets, out, mask=mask)

@triton.jit
def _mg_swiglu_bwd_kernel(
    grad_out_ptr,    # [M, H]
    input_ptr,       # [M, 2H] (Input original w12)
    grad_input_ptr,  # [M, 2H] (Output grad para w12)
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
        
        # Carregar em FP32
        g_out = tl.load(row_grad_out + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Recalcular ativação
        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate
        
        # Derivadas (Chain Rule)
        # dL/dVal = dL/dOut * SiLU(Gate)
        d_val = g_out * silu_gate
        
        # dL/dGate = dL/dOut * Val * (Sig * (1 + Gate * (1 - Sig)))
        term = 1.0 + gate * (1.0 - sig_gate)
        d_silu = sig_gate * term
        d_gate = g_out * val * d_silu
        
        # Gravar no tensor combinado de gradiente
        tl.store(row_grad_input + offsets, d_gate, mask=mask)
        tl.store(row_grad_input + n_cols_half + offsets, d_val, mask=mask)

# ============================================================================
# AUTOGRAD FUNCTION (A Ponte)
# ============================================================================

class MegaGemmFunction(Function):
    @staticmethod
    def forward(ctx, w12_out, hidden_dim):
        # Garante contiguidade e achata para o kernel
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
    def backward(ctx, grad_output):
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

# ============================================================================
# NN.MODULE (A Camada Plugável)
# ============================================================================

class MegaGemmTriton(nn.Module):
    """
    Mega Gemm (MG Triton): Drop-in replacement para SwiGLU.
    
    Args:
        d_model (int): Dimensão de entrada/saída.
        multiple_of (int): Garante que a dimensão oculta seja múltipla de X (padrão 256).
        hidden_multiple (float): Fator de expansão (padrão LLaMA 5/3).
    """
    def __init__(self, d_model: int, multiple_of: int = 256, hidden_multiple: float = 5/3):
        super().__init__()
        hidden = int(d_model * hidden_multiple)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        
        # Camada 1: Fusão W1 (Gate) e W2 (Value) numa única MatMul gigante.
        # Melhora throughput dos Tensor Cores.
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        
        # Camada 2: Projeção de saída (W3)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Mega MatMul (Gate + Value)
        # O resultado é [Batch, Seq, 2*Hidden]
        w12_out = self.w12(x)
        
        # 2. Triton Activation Kernel (Offset Read)
        # Separa e ativa sem criar cópias na memória
        hidden = MegaGemmFunction.apply(w12_out, self.hidden)
        
        # 3. Output Projection
        return self.w3(hidden)

    def extra_repr(self):
        return f"d_model={self.w12.in_features}, hidden={self.hidden} (Triton Accelerated)"