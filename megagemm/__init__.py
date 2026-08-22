"""
🔥 MegaGemm — High Performance LLM Inference Engine
=====================================================

Custom CUDA/Triton kernels + full inference pipeline:
- RMSNorm (CUDA) — FP32/FP16/BF16
- SwiGLU (Triton) — Fused activation
- RoPE (CUDA/PyTorch) — Rotary embeddings
- PagedAttention (Triton) — Paged KV cache decode
- Inference Engine — HuggingFace model loading + generation

Usage:
------
>>> # Kernels
>>> from megagemm import RMSNorm, MegaGemmTriton, RoPE

>>> # Inference
>>> from megagemm.engine import InferenceEngine
>>> engine = InferenceEngine("meta-llama/Llama-3.2-3B-Instruct")
>>> print(engine.generate("Hello world"))

Author: Gabriel Yogi
License: MIT
"""

__version__ = "0.3.0"
__author__ = "Gabriel Yogi"

# Lazy imports to handle missing CUDA/Triton gracefully
def __getattr__(name):
    # --- Kernels ---
    if name == "RMSNorm":
        from .kernels.rmsnorm import RMSNorm
        return RMSNorm
    elif name == "RMSNormFunction":
        from .kernels.rmsnorm import RMSNormFunction
        return RMSNormFunction
    elif name == "MegaGemmTriton":
        from .kernels.swiglu import MegaGemmTriton
        return MegaGemmTriton
    elif name == "MegaGemmFunction":
        from .kernels.swiglu import MegaGemmFunction
        return MegaGemmFunction
    elif name == "RoPE":
        from .kernels.rope import RoPE
        return RoPE
    elif name == "apply_rotary_emb":
        from .kernels.rope import apply_rotary_emb
        return apply_rotary_emb
    elif name == "precompute_freqs_cis":
        from .kernels.rope import precompute_freqs_cis
        return precompute_freqs_cis
    elif name == "paged_attention_decode":
        from .kernels.paged_attention import paged_attention_decode
        return paged_attention_decode
    # --- Engine infrastructure ---
    elif name == "BlockManager":
        from .engine.kv_cache import BlockManager
        return BlockManager
    elif name == "sample_logits":
        from .engine.sampling import sample_logits
        return sample_logits
    elif name == "LayerOffloadManager":
        from .engine.offload import LayerOffloadManager
        return LayerOffloadManager
    elif name == "EmbeddingEngine":
        from .embeddings.engine import EmbeddingEngine
        return EmbeddingEngine
    elif name == "MGXProphetLibrary":
        from .engine.prophet import MGXProphetLibrary
        return MGXProphetLibrary
    elif name == "load_from_mgx":
        from .models.mgx import load_from_mgx
        return load_from_mgx
    elif name == "attach_session_state_to_mgx":
        from .models.mgx import attach_session_state_to_mgx
        return attach_session_state_to_mgx
    elif name == "extract_session_state_from_mgx":
        from .models.mgx import extract_session_state_from_mgx
        return extract_session_state_from_mgx
    elif name == "export_to_mgx":
        from .models.mgx import export_to_mgx
        return export_to_mgx
    elif name == "inspect_mgx":
        from .models.mgx import inspect_mgx
        return inspect_mgx
    # --- Quantization ---
    elif name == "QuantizedLinear":
        from .quantization.w4a16 import QuantizedLinear
        return QuantizedLinear
    elif name == "NativeW4A16Linear":
        from .quantization.native_w4a16 import NativeW4A16Linear
        return NativeW4A16Linear
    elif name == "Int8Linear":
        from .quantization.w8a16 import Int8Linear
        return Int8Linear
    elif name == "XAIReport":
        from .engine.xai import XAIReport
        return XAIReport
    raise AttributeError(f"module 'megagemm' has no attribute '{name}'")

__all__ = [
    # Kernels
    "RMSNorm",
    "RMSNormFunction",
    "MegaGemmTriton",
    "MegaGemmFunction",
    "RoPE",
    "apply_rotary_emb",
    "precompute_freqs_cis",
    # Inference
    "paged_attention_decode",
    "BlockManager",
    "sample_logits",
    "EmbeddingEngine",
    "MGXProphetLibrary",
    "load_from_mgx",
    "attach_session_state_to_mgx",
    "extract_session_state_from_mgx",
    "export_to_mgx",
    "inspect_mgx",
    # Quantization
    "QuantizedLinear",
    "NativeW4A16Linear",
    "Int8Linear",
    # Offload
    "LayerOffloadManager",
    # XAI
    "XAIReport",
    # Meta
    "__version__",
]
