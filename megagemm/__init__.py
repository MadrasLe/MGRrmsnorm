"""
🔥 MEGA GEMM - High Performance CUDA Kernels for LLMs
=====================================================

Drop-in replacements for PyTorch's RMSNorm and SwiGLU with:
- ~67% speedup over native PyTorch
- FP32, FP16, and BF16 support
- Full backward pass support
- Triton-accelerated SwiGLU

Usage:
------
>>> from megagemm import RMSNorm, MegaGemmTriton
>>> 
>>> # RMSNorm (CUDA accelerated)
>>> norm = RMSNorm(hidden_size=4096).cuda()
>>> y = norm(x)
>>>
>>> # SwiGLU (Triton accelerated)  
>>> swiglu = MegaGemmTriton(d_model=4096).cuda()
>>> y = swiglu(x)

Author: Gabriel Yogi
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Gabriel Yogi"

# Lazy imports to handle missing CUDA gracefully
def __getattr__(name):
    if name == "RMSNorm":
        from .rmsnorm import RMSNorm
        return RMSNorm
    elif name == "RMSNormFunction":
        from .rmsnorm import RMSNormFunction
        return RMSNormFunction
    elif name == "MegaGemmTriton":
        from .swiglu import MegaGemmTriton
        return MegaGemmTriton
    elif name == "MegaGemmFunction":
        from .swiglu import MegaGemmFunction
        return MegaGemmFunction
    raise AttributeError(f"module 'megagemm' has no attribute '{name}'")

__all__ = [
    "RMSNorm",
    "RMSNormFunction",
    "MegaGemmTriton", 
    "MegaGemmFunction",
    "__version__",
]
