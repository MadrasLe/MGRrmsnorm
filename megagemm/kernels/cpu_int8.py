"""
⚡ CPU INT8 GEMV/GEMM — Python wrapper for AVX2 C kernel
=========================================================
Auto-compiles cpu_gemv.c on first import. Falls back to
PyTorch if compilation fails.

Usage:
    from megagemm.kernels.cpu_int8 import CPUInt8Linear

    # Replace nn.Linear with INT8 version
    linear = CPUInt8Linear.from_float(nn_linear_module)
    output = linear(input_tensor)

Author: Gabriel Yogi
"""

import os
import sys
import ctypes
import subprocess
import torch
import torch.nn as nn
from pathlib import Path

__all__ = ['cpu_gemv', 'cpu_gemm', 'quantize_to_int8', 'CPUInt8Linear']

# ─────────────────────────────────────────────
# Auto-compile and load the C library
# ─────────────────────────────────────────────

_lib = None
_LIB_LOADED = False

def _get_lib():
    """Compile and load libcpu_gemv on first call."""
    global _lib, _LIB_LOADED
    if _LIB_LOADED:
        return _lib

    src = Path(__file__).parent / "cpu_gemv.c"
    if not src.exists():
        print(f"⚠️ cpu_gemv.c not found at {src}")
        _LIB_LOADED = True
        return None

    # Output path
    if sys.platform == 'win32':
        lib_name = "cpu_gemv.dll"
    elif sys.platform == 'darwin':
        lib_name = "libcpu_gemv.dylib"
    else:
        lib_name = "libcpu_gemv.so"

    lib_path = src.parent / lib_name

    # Compile if needed (or if source is newer)
    need_compile = not lib_path.exists()
    if lib_path.exists() and src.stat().st_mtime > lib_path.stat().st_mtime:
        need_compile = True

    if need_compile:
        print(f"🔨 Compiling {src.name} → {lib_name}...")
        compiled = False

        if sys.platform == 'win32':
            # Windows: MSVC with /openmp, then GCC fallback
            for cmd in [
                ["cl", "/O2", "/arch:AVX2", "/openmp", "/LD",
                 str(src), f"/Fe:{lib_path}"],
                ["gcc", "-O3", "-mavx2", "-mfma", "-mavxvnni",
                 "-fopenmp", "-shared", "-o", str(lib_path), str(src)],
                ["gcc", "-O3", "-mavx2", "-mfma",
                 "-fopenmp", "-shared", "-o", str(lib_path), str(src)],
            ]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    compiled = True
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        elif sys.platform == 'darwin':
            # macOS: clang (no OpenMP by default, try with -Xpreprocessor)
            for cmd in [
                ["clang", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
                 "-Xpreprocessor", "-fopenmp", "-lomp",
                 "-o", str(lib_path), str(src)],
                ["clang", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
                 "-o", str(lib_path), str(src)],
            ]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    compiled = True
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        else:
            # Linux: try AVX2+OpenMP first (safe), VNNI only if forced
            for cmd in [
                ["gcc", "-O3", "-mavx2", "-mfma",
                 "-fopenmp", "-shared", "-fPIC",
                 "-o", str(lib_path), str(src)],
            ]:
                try:
                    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  ✅ Compiled AVX2 + OpenMP")
                    compiled = True
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue

        if not compiled:
            print(f"  ⚠️ Compilation failed — falling back to PyTorch")
            _LIB_LOADED = True
            return None

    try:
        _lib = ctypes.CDLL(str(lib_path))

        # Set up function signatures
        _lib.megagemm_gemv_w8a32.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # weights
            ctypes.c_void_p,  # scales
            ctypes.c_void_p,  # input
            ctypes.c_int,     # M
            ctypes.c_int,     # K
        ]
        _lib.megagemm_gemv_w8a32.restype = None

        _lib.megagemm_gemm_w8a32.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # weights
            ctypes.c_void_p,  # scales
            ctypes.c_void_p,  # input
            ctypes.c_int,     # N
            ctypes.c_int,     # M
            ctypes.c_int,     # K
        ]
        _lib.megagemm_gemm_w8a32.restype = None

        _lib.megagemm_quantize_w8.argtypes = [
            ctypes.c_void_p,  # out_weights
            ctypes.c_void_p,  # out_scales
            ctypes.c_void_p,  # fp32_weights
            ctypes.c_int,     # M
            ctypes.c_int,     # K
        ]
        _lib.megagemm_quantize_w8.restype = None

        print(f"  ✅ Loaded {lib_name}")
        _LIB_LOADED = True
        return _lib

    except OSError as e:
        print(f"  ⚠️ Failed to load {lib_name}: {e}")
        _LIB_LOADED = True
        return None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def quantize_to_int8(weight_fp32: torch.Tensor):
    """
    Quantize FP32 weight matrix to INT8 + per-row scales.

    Args:
        weight_fp32: [M, K] float32 weight matrix

    Returns:
        (weight_int8, scales) — [M, K] int8, [M] float32
    """
    lib = _get_lib()
    M, K = weight_fp32.shape
    w = weight_fp32.contiguous().float()

    if lib is not None:
        out_w = torch.empty(M, K, dtype=torch.int8)
        out_s = torch.empty(M, dtype=torch.float32)
        lib.megagemm_quantize_w8(
            out_w.data_ptr(), out_s.data_ptr(), w.data_ptr(), M, K
        )
        return out_w, out_s
    else:
        # PyTorch fallback
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = (amax / 127.0).squeeze(1)
        w_int8 = (w / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        return w_int8, scales


def cpu_gemv(weight_int8: torch.Tensor, scales: torch.Tensor,
             input: torch.Tensor) -> torch.Tensor:
    """
    INT8 GEMV: output = (W_int8 * scales) @ input

    Args:
        weight_int8: [M, K] int8
        scales: [M] float32
        input: [K] float32

    Returns:
        output: [M] float32
    """
    lib = _get_lib()
    M, K = weight_int8.shape

    inp = input.contiguous().float()
    w = weight_int8.contiguous()
    s = scales.contiguous().float()
    output = torch.empty(M, dtype=torch.float32)

    if lib is not None:
        lib.megagemm_gemv_w8a32(
            output.data_ptr(), w.data_ptr(), s.data_ptr(),
            inp.data_ptr(), M, K
        )
    else:
        # PyTorch fallback
        output = (w.float() * s.unsqueeze(1)) @ inp

    return output


def cpu_gemm(weight_int8: torch.Tensor, scales: torch.Tensor,
             input: torch.Tensor) -> torch.Tensor:
    """
    INT8 GEMM: output[N,M] = input[N,K] @ (W_int8 * scales).T

    Args:
        weight_int8: [M, K] int8
        scales: [M] float32
        input: [N, K] float32

    Returns:
        output: [N, M] float32
    """
    lib = _get_lib()
    M, K = weight_int8.shape

    inp_flat = input.contiguous().float()
    if inp_flat.dim() == 1:
        inp_flat = inp_flat.unsqueeze(0)
    N = inp_flat.shape[0]

    w = weight_int8.contiguous()
    s = scales.contiguous().float()
    output = torch.empty(N, M, dtype=torch.float32)

    if lib is not None:
        lib.megagemm_gemm_w8a32(
            output.data_ptr(), w.data_ptr(), s.data_ptr(),
            inp_flat.data_ptr(), N, M, K
        )
    else:
        output = inp_flat @ (w.float() * s.unsqueeze(1)).T

    return output


# ─────────────────────────────────────────────
# Drop-in nn.Module replacement
# ─────────────────────────────────────────────

class CPUInt8Linear(nn.Module):
    """
    INT8 Linear layer for CPU inference.
    Drop-in replacement for nn.Linear with ~2-4x speedup.

    Usage:
        linear = CPUInt8Linear.from_float(existing_linear)
        output = linear(input)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scales', torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'CPUInt8Linear':
        """Convert a float nn.Linear to INT8."""
        has_bias = linear.bias is not None
        layer = cls(linear.in_features, linear.out_features, bias=has_bias)

        w_int8, scales = quantize_to_int8(linear.weight.data.float())
        layer.weight_int8 = w_int8
        layer.scales = scales
        if has_bias:
            layer.bias = linear.bias.data.float()

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, self.in_features)

        out = cpu_gemm(self.weight_int8, self.scales, x_flat)

        if self.bias is not None:
            out = out + self.bias

        return out.view(*shape[:-1], self.out_features)

    def __repr__(self):
        return (f"CPUInt8Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None})")
