"""
🔄 Layer Offload Manager for MegaGemm
--------------------------------------
Optional feature: offload transformer layers to CPU RAM or disk
to run models larger than GPU VRAM.

3-tier memory hierarchy:
  - GPU VRAM (fast)   → layers 0..n_gpu_layers-1
  - CPU RAM (medium)  → layers that fit in RAM
  - Disk (slow)       → overflow to NVMe/SSD via safetensors

Double Buffering:
  Pre-allocates 2 GPU layer copies. While GPU computes on buffer A,
  buffer B receives the next layer via async DMA. True overlap.

Usage:
  # No offload (default, zero overhead)
  engine = InferenceEngine("model/8B")

  # Partial offload to CPU
  engine = InferenceEngine("model/70B", n_gpu_layers=20)

  # Offload to CPU + disk
  engine = InferenceEngine("model/70B", n_gpu_layers=10,
                            offload_dir="/tmp/megagemm_offload")

Author: Gabriel Yogi
"""

import torch
import torch.nn as nn
import os
import gc
import copy
import time
from contextlib import nullcontext
from typing import Optional, Set, Dict
from pathlib import Path

__all__ = ['LayerOffloadManager']


class LayerOffloadManager:
    """
    Manages layer placement across GPU, CPU RAM, and optionally disk.

    Uses double-buffered async transfer for maximum GPU utilization:
    - Two pre-allocated GPU buffers alternate roles
    - While buffer A computes, buffer B receives next layer via DMA
    - copy_() with non_blocking=True + pinned memory = true overlap

    When no offload is needed (n_gpu_layers == -1 or >= num_layers),
    this class is never instantiated → zero overhead on default path.
    """

    def __init__(
        self,
        num_layers: int,
        n_gpu_layers: int = -1,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        offload_dir: Optional[str] = None,
        pin_memory: bool = True,
    ):
        """
        Args:
            num_layers: Total number of transformer layers
            n_gpu_layers: Layers to keep on GPU (-1 = all, 0 = none)
            device: GPU device string
            dtype: Model dtype for GPU computation
            offload_dir: Directory for disk offload (None = CPU only)
            pin_memory: Use pinned memory for faster CPU→GPU transfer
        """
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.offload_dir = offload_dir
        self.pin_memory = pin_memory

        # Resolve n_gpu_layers
        if n_gpu_layers < 0:
            n_gpu_layers = num_layers  # All on GPU
        n_gpu_layers = min(n_gpu_layers, num_layers)

        self.n_gpu_layers = n_gpu_layers

        # Layer placement sets
        self.gpu_layers: Set[int] = set(range(n_gpu_layers))
        self.cpu_layers: Set[int] = set(range(n_gpu_layers, num_layers))
        self.disk_layers: Set[int] = set()

        # Double buffer state
        self._buffer_layers = [None, None]  # Two GPU copies
        self._active_buffer = 0  # Which buffer has the ready layer
        self._pending_buffer = -1  # Which buffer is receiving data
        self._pending_layer_idx = -1  # Which layer is being prefetched

        # Async copy stream (separate from compute stream)
        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_event = torch.cuda.Event()  # For synchronization

        # Disk offload paths
        self._disk_paths: Dict[int, str] = {}

        # Stats
        self.transfer_time = 0.0
        self.transfer_count = 0
        self._buffer_mb = 0.0

    @property
    def is_active(self) -> bool:
        """Returns True if any layers are offloaded."""
        return len(self.cpu_layers) > 0 or len(self.disk_layers) > 0

    def _init_double_buffers(self, template_layer: nn.Module):
        """
        Create 2 GPU copies of a layer for double buffering.
        Uses deepcopy to preserve module structure, then moves to GPU.

        Cost: ~2x one layer's VRAM. For 8B (480MB/layer) = ~960MB.
        """
        print("  🔄 Initializing double buffers...")

        # Save original device
        original_devices = {}
        for name, p in template_layer.named_parameters():
            original_devices[name] = p.device
        for name, b in template_layer.named_buffers():
            original_devices[f"_buf_{name}"] = b.device

        # Create two GPU copies
        for i in range(2):
            buf = copy.deepcopy(template_layer)
            buf.to(self.device)
            buf.eval()
            self._buffer_layers[i] = buf

        # Calculate buffer memory
        self._buffer_mb = sum(
            p.nelement() * p.element_size()
            for p in self._buffer_layers[0].parameters()
        ) / 1024**2
        buf_mb = sum(
            b.nelement() * b.element_size()
            for b in self._buffer_layers[0].buffers()
        ) / 1024**2
        self._buffer_mb += buf_mb

        print(f"  📦 Double buffers: 2 × {self._buffer_mb:.0f}MB = "
              f"{self._buffer_mb * 2:.0f}MB VRAM")

    def _copy_layer_to_buffer(self, src_layer: nn.Module, buffer_idx: int,
                               stream=None):
        """
        Copy all parameters + buffers from src layer into GPU buffer.
        Uses copy_() with non_blocking=True for true async DMA.

        This is MUCH faster than .to(device) because:
        1. No GPU memory allocation (buffer already exists)
        2. copy_() with pinned memory = pure DMA transfer
        3. non_blocking + separate stream = overlaps with compute
        """
        buf_layer = self._buffer_layers[buffer_idx]
        ctx = torch.cuda.stream(stream) if stream else nullcontext()

        with ctx:
            # Copy parameters
            for (_, dst_p), (_, src_p) in zip(
                buf_layer.named_parameters(),
                src_layer.named_parameters()
            ):
                dst_p.data.copy_(src_p.data, non_blocking=True)

            # Copy buffers (important for AWQ: qweight, qzeros, scales)
            for (_, dst_b), (_, src_b) in zip(
                buf_layer.named_buffers(),
                src_layer.named_buffers()
            ):
                dst_b.data.copy_(src_b.data, non_blocking=True)

    def setup_layers(self, layers: nn.ModuleList):
        """
        Move layers to their assigned tiers after model loading.
        Initializes double buffers using the first offloaded layer as template.
        """
        if not self.is_active:
            return

        print(f"🔄 Setting up layer offload: "
              f"{len(self.gpu_layers)} GPU, "
              f"{len(self.cpu_layers)} CPU"
              f"{f', disk={self.offload_dir}' if self.offload_dir else ''}")

        # Prepare disk offload directory
        if self.offload_dir:
            os.makedirs(self.offload_dir, exist_ok=True)

        # Initialize double buffers from first offloaded layer (while still on GPU)
        first_offloaded = min(self.cpu_layers)
        self._init_double_buffers(layers[first_offloaded])

        cpu_count = 0
        disk_count = 0

        for idx in range(self.num_layers):
            if idx in self.gpu_layers:
                continue

            layer = layers[idx]

            if self._can_fit_on_cpu(layer):
                self._move_to_cpu(layer, idx)
                cpu_count += 1
            elif self.offload_dir:
                self._save_to_disk(layer, idx)
                self.cpu_layers.discard(idx)
                self.disk_layers.add(idx)
                disk_count += 1
            else:
                self._move_to_cpu(layer, idx)
                cpu_count += 1

        gc.collect()
        torch.cuda.empty_cache()

        gpu_mb = sum(
            p.nelement() * p.element_size()
            for i in self.gpu_layers
            for p in layers[i].parameters()
        ) / 1024**2 if self.gpu_layers else 0

        status = (f"  📍 GPU: {len(self.gpu_layers)} layers ({gpu_mb:.0f}MB) "
                  f"+ 2 buffers ({self._buffer_mb * 2:.0f}MB)")
        if cpu_count:
            status += f" | CPU: {cpu_count} layers"
        if disk_count:
            status += f" | Disk: {disk_count} layers"
        print(status)

    def _can_fit_on_cpu(self, layer: nn.Module) -> bool:
        """Check if layer fits in available CPU RAM."""
        try:
            import psutil
            layer_bytes = sum(p.nelement() * p.element_size() for p in layer.parameters())
            available = psutil.virtual_memory().available
            return layer_bytes < (available - 2 * 1024**3)
        except ImportError:
            return True

    def _move_to_cpu(self, layer: nn.Module, idx: int):
        """Move layer to CPU, optionally with pinned memory."""
        layer.to('cpu')
        if self.pin_memory:
            for param in layer.parameters():
                param.data = param.data.pin_memory()
            for buf in layer.buffers():
                try:
                    buf.data = buf.data.pin_memory()
                except Exception:
                    pass

    def _save_to_disk(self, layer: nn.Module, idx: int):
        """Save layer to disk as safetensors and free memory."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors required for disk offload: pip install safetensors")

        path = os.path.join(self.offload_dir, f"layer_{idx:04d}.safetensors")

        state = {}
        for name, param in layer.named_parameters():
            state[name] = param.data.cpu()
        for name, buf in layer.named_buffers():
            state[f"_buffer_{name}"] = buf.cpu()

        save_file(state, path)
        self._disk_paths[idx] = path
        layer.to('meta')

    def _load_from_disk(self, layer: nn.Module, idx: int):
        """Load layer from disk safetensors into CPU."""
        from safetensors.torch import load_file

        path = self._disk_paths[idx]
        state = load_file(path, device='cpu')

        for name, param in layer.named_parameters():
            if name in state:
                param.data = state[name]
                if self.pin_memory:
                    param.data = param.data.pin_memory()
        for name, buf in layer.named_buffers():
            key = f"_buffer_{name}"
            if key in state:
                buf.data = state[key]
                if self.pin_memory:
                    try:
                        buf.data = buf.data.pin_memory()
                    except Exception:
                        pass

    def get_layer_on_gpu(self, layer_idx: int, layer: nn.Module) -> nn.Module:
        """
        Ensure layer is on GPU and ready for computation.

        For GPU-resident layers: returns the original layer (zero overhead).
        For offloaded layers: returns a GPU buffer with the layer's weights.

        With double buffering:
        - If prefetch hit: wait for async copy, return buffer (fast)
        - If prefetch miss: synchronous copy into buffer (slower)
        """
        if layer_idx in self.gpu_layers:
            return layer  # Already on GPU, no-op

        # Check if this layer was prefetched into a buffer
        if self._pending_layer_idx == layer_idx:
            # Wait for the async copy to finish
            torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)
            result = self._buffer_layers[self._pending_buffer]
            self._active_buffer = self._pending_buffer
            self._pending_layer_idx = -1
            return result

        # No prefetch hit — synchronous copy into active buffer
        t0 = time.perf_counter()

        # If layer is on disk, load to CPU first
        if layer_idx in self.disk_layers:
            self._load_from_disk(layer, layer_idx)

        # Copy into the currently free buffer
        buf_idx = 1 - self._active_buffer  # Use the other buffer
        self._copy_layer_to_buffer(layer, buf_idx)
        torch.cuda.synchronize(self.device)

        dt = time.perf_counter() - t0
        self.transfer_time += dt
        self.transfer_count += 1
        self._active_buffer = buf_idx

        return self._buffer_layers[buf_idx]

    def prefetch_next(self, next_idx: int, layers: nn.ModuleList):
        """
        Start async copy of next layer into the inactive buffer.

        Double buffer pipeline:
          Buffer A: [computing layer N]  ←── GPU main stream
          Buffer B: [receiving layer N+1] ←── copy stream (async DMA)

        When layer N finishes, buffer B is ready with N+1. Swap roles.
        """
        if next_idx >= self.num_layers:
            return

        if next_idx in self.gpu_layers:
            return

        # If layer is on disk, load to CPU first (unavoidable sync)
        if next_idx in self.disk_layers:
            self._load_from_disk(layers[next_idx], next_idx)

        # Copy into the INACTIVE buffer (the one not currently computing)
        target_buf = 1 - self._active_buffer
        self._copy_layer_to_buffer(layers[next_idx], target_buf,
                                    stream=self._copy_stream)

        self._pending_buffer = target_buf
        self._pending_layer_idx = next_idx

    def release_layer(self, layer_idx: int, layer: nn.Module):
        """
        Release layer after computation.

        With double buffering, CPU layers don't need to be moved back —
        they stay on CPU. The GPU buffer is reused for the next layer.
        GPU-resident layers are never touched.
        """
        if layer_idx in self.gpu_layers:
            return  # Keep on GPU permanently
        # CPU/disk layers: nothing to do — the original stays on CPU,
        # and the GPU buffer will be reused by prefetch_next

    def print_stats(self):
        """Print transfer statistics."""
        if self.transfer_count > 0:
            avg = self.transfer_time / self.transfer_count * 1000
            print(f"🔄 Offload stats: {self.transfer_count} sync transfers, "
                  f"{self.transfer_time:.2f}s total, {avg:.1f}ms avg")
            print(f"   (lower is better — sync transfers = prefetch misses)")

    def __repr__(self):
        buf_str = f", buf=2×{self._buffer_mb:.0f}MB" if self._buffer_mb else ""
        return (f"LayerOffloadManager("
                f"gpu={len(self.gpu_layers)}, "
                f"cpu={len(self.cpu_layers)}, "
                f"disk={len(self.disk_layers)}, "
                f"pin={self.pin_memory}{buf_str})")
