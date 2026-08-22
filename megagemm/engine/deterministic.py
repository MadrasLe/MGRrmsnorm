"""
🔒 Deterministic Inference Mode for MegaGemm
----------------------------------------------
Guarantees bit-exact reproducible output across runs.

When enabled, forces all PyTorch/CUDA operations to use deterministic
algorithms. This includes cuBLAS GEMMs, cuDNN attention backends, and
RNG-based sampling. Custom MegaGemm kernels (RMSNorm CUDA, SwiGLU Triton,
RoPE) are already deterministic by design.

Performance note: ~10-15% overhead from deterministic cuBLAS workspace
configuration. Disable when maximum throughput is needed.

Author: Gabriel Yogi
"""

import os
import torch
from typing import Optional

__all__ = ['enable_deterministic_mode', 'disable_deterministic_mode', 'is_deterministic']

# Module-level state
_deterministic_enabled = False
_original_env: Optional[str] = None


def enable_deterministic_mode(seed: int = 42) -> None:
    """
    Enable fully deterministic inference.

    Sets all necessary PyTorch flags, CUDA environment variables, and RNG
    seeds to guarantee bit-exact reproducibility across runs on the same
    hardware.

    Args:
        seed: Random seed for reproducibility (default: 42)

    What this does:
        1. CUBLAS_WORKSPACE_CONFIG=":4096:8" — Forces cuBLAS to use a
           deterministic workspace for GEMMs (nn.Linear).
        2. torch.use_deterministic_algorithms(True) — Errors on any
           non-deterministic op (SDPA, scatter, etc.).
        3. torch.manual_seed(seed) — Seeds CPU and GPU RNG for
           torch.multinomial (sampling with temperature > 0).
        4. cudnn.deterministic=True — Forces deterministic cuDNN backend
           selection for convolutions and attention.
        5. cudnn.benchmark=False — Disables auto-tuning that can cause
           non-deterministic kernel selection.
    """
    global _deterministic_enabled, _original_env

    # Save original env for restore
    _original_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    # 1. cuBLAS deterministic workspace
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. PyTorch deterministic algorithms (errors on non-deterministic ops)
    torch.use_deterministic_algorithms(True, warn_only=False)

    # 3. Seed all RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _deterministic_enabled = True


def disable_deterministic_mode() -> None:
    """
    Restore default (non-deterministic, maximum performance) mode.
    """
    global _deterministic_enabled, _original_env

    # Restore env
    if _original_env is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _original_env
    elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]

    # Restore PyTorch defaults
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    _deterministic_enabled = False
    _original_env = None


def is_deterministic() -> bool:
    """Check if deterministic mode is currently enabled."""
    return _deterministic_enabled
