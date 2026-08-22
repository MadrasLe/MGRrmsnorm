#!/usr/bin/env python3
"""
🔧 Smart installer for MegaGemm — environment-aware.

Auto-detects the environment (Colab, Kaggle, Docker, bare-metal) and
installs MegaGemm using whatever is already present:

  - Reuses the existing torch if it matches the local CUDA toolkit.
  - Installs a compatible torch wheel only when there's a mismatch.
  - Skips CUDA extension build when nvcc is missing (AMD, CPU-only).
  - Installs inference dependencies (triton, transformers, etc.).
  - Never installs torchvision/torchaudio (MegaGemm doesn't need them).

Usage:
    python scripts/install_smart.py              # standard install
    python scripts/install_smart.py --editable   # editable (dev) install
    python scripts/install_smart.py --cpu-only   # CPU-only, skip CUDA entirely

Author: Gabriel Yogi
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True) -> int:
    print(f"  + {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result.returncode


def normalize_cuda(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    parts = v.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else v


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment() -> str:
    """Detect runtime environment."""
    if os.path.exists("/content") and os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        return "docker"
    return "local"


def detect_nvcc_cuda() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, text=True,
        )
    except Exception:
        return None
    m = re.search(r"release\s+(\d+\.\d+)", out)
    return normalize_cuda(m.group(1) if m else None)


def detect_torch_cuda() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    return normalize_cuda(torch.version.cuda)


def detect_gpu_vendor() -> str:
    """Best-effort GPU detection: 'nvidia', 'amd', or 'none'."""
    # Try nvidia-smi
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return "nvidia"
    except Exception:
        pass
    # Try rocm-smi (AMD)
    try:
        subprocess.check_output(["rocm-smi"], stderr=subprocess.STDOUT)
        return "amd"
    except Exception:
        pass
    return "none"


def pick_pytorch_index(cuda_version: Optional[str]) -> Optional[str]:
    """Return the PyTorch wheel index URL for a given CUDA version."""
    if not cuda_version:
        return None
    mapping = {
        "12.8": "https://download.pytorch.org/whl/cu128",
        "12.6": "https://download.pytorch.org/whl/cu126",
        "12.4": "https://download.pytorch.org/whl/cu124",
        "11.8": "https://download.pytorch.org/whl/cu118",
    }
    url = mapping.get(cuda_version)
    if url:
        return url
    # Fallback: try latest 12.x
    if cuda_version.startswith("12."):
        return "https://download.pytorch.org/whl/cu128"
    return None


# ---------------------------------------------------------------------------
# Torch alignment
# ---------------------------------------------------------------------------

def ensure_torch_matches(cpu_only: bool = False) -> None:
    """
    Ensure torch is installed and matches the CUDA toolkit.

    On CPU-only mode, just ensure torch is installed (any variant).
    """
    if cpu_only:
        try:
            import torch
            print(f"  ✅ torch {torch.__version__} already installed (CPU mode)")
        except ImportError:
            print("  📦 Installing torch (CPU)...")
            run([sys.executable, "-m", "pip", "install", "torch"])
        return

    nvcc_cuda = detect_nvcc_cuda()
    torch_cuda = detect_torch_cuda()

    if nvcc_cuda is None:
        print("  ℹ️  nvcc not found — CUDA extensions will be skipped")
        # Don't re-install torch; whatever is there is fine
        return

    if torch_cuda == nvcc_cuda:
        print(f"  ✅ torch CUDA={torch_cuda} matches nvcc CUDA={nvcc_cuda}")
        return

    # Mismatch — try to align
    index = pick_pytorch_index(nvcc_cuda)
    if not index:
        print(
            f"  ⚠️  Cannot auto-align torch: unsupported nvcc CUDA={nvcc_cuda}.\n"
            f"     CUDA extension build may fail. Install a compatible torch manually."
        )
        return

    print(
        f"  🔄 Aligning torch with nvcc CUDA={nvcc_cuda} "
        f"(current torch CUDA={torch_cuda or 'N/A'})..."
    )
    run([
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--index-url", index, "torch",
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="🔧 Smart installer for MegaGemm",
    )
    parser.add_argument(
        "--editable", action="store_true",
        help="Install in editable mode (pip install -e .)",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="CPU-only install: skip all CUDA/GPU setup",
    )
    parser.add_argument(
        "--project-dir", default=".",
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--extras", default="inference",
        help="pip extras to install (default: inference)",
    )
    args = parser.parse_args()

    project_dir = str(Path(args.project_dir).resolve())

    # ── 1. Environment detection ──
    env = detect_environment()
    gpu = detect_gpu_vendor()
    nvcc = detect_nvcc_cuda()
    torch_cuda = detect_torch_cuda()

    print("=" * 50)
    print("🔧 MegaGemm Smart Installer")
    print("=" * 50)
    print(f"  Environment : {env}")
    print(f"  GPU vendor  : {gpu}")
    print(f"  nvcc CUDA   : {nvcc or 'not found'}")
    print(f"  torch CUDA  : {torch_cuda or 'not found'}")
    print(f"  CPU-only    : {args.cpu_only}")
    print()

    # ── 2. Torch alignment ──
    print("── Step 1: Checking torch ──")
    cpu_only = args.cpu_only or gpu == "none"
    if gpu == "amd" and not args.cpu_only:
        print("  ℹ️  AMD GPU detected. Using torch CPU/ROCm as available.")
        print("     CUDA extensions will be skipped; Triton kernels may work with ROCm.")
    ensure_torch_matches(cpu_only=cpu_only)
    print()

    # ── 3. Install MegaGemm ──
    print("── Step 2: Installing MegaGemm ──")

    # Set env var to skip CUDA build if we know it'll fail
    env_vars = os.environ.copy()
    if nvcc is None or cpu_only or gpu == "amd":
        env_vars["MEGAGEMM_SKIP_CUDA"] = "1"

    install_cmd = [
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
    ]
    extras = f"[{args.extras}]" if args.extras else ""
    if args.editable:
        install_cmd.extend(["-e", f"{project_dir}{extras}"])
    else:
        install_cmd.append(f"{project_dir}{extras}")

    print(f"  + {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd, env=env_vars)
    if result.returncode != 0:
        print("\n❌ Installation failed!")
        return 1
    print()

    # ── 4. Summary ──
    print("=" * 50)
    print("✅ MegaGemm installation complete!")
    print("=" * 50)

    # Quick check what's actually available
    checks = {
        "CUDA kernels (RMSNorm/RoPE)": "import rmsnorm_cuda_ops",
        "Triton (SwiGLU/PagedAttn)": "import triton",
        "HuggingFace Hub": "import huggingface_hub",
        "Transformers": "import transformers",
        "Safetensors": "import safetensors",
    }
    for name, import_stmt in checks.items():
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except Exception:
            print(f"  ⬚  {name} (not available)")

    print()
    print("  Quick test:")
    print("    python -c \"from megagemm.engine import InferenceEngine; print('OK')\"")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
