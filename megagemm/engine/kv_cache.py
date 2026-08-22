"""
🧱 Paged KV Cache Manager for MegaGemm
---------------------------------------
Manages GPU memory for KV cache using fixed-size blocks.
Supports per-layer KV caches for multi-layer transformer models.

Includes TieredBlockManager for GPU+CPU offloading:
  - GPU pool: hot blocks used by Triton kernels
  - CPU pool: cold blocks in pinned memory
  - Async DMA transfer with CUDA streams
  - Window-based eviction policy

Optimized for minimal Python overhead in the decode hot path.

Author: Gabriel Yogi
"""

import os
import torch
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    from ..kernels.paged_attention import (
        paged_kv_cache_scatter as _paged_kv_cache_scatter,
        paged_kv_cache_scatter_token_tiled as _paged_kv_cache_scatter_token_tiled,
    )
except Exception:
    _paged_kv_cache_scatter = None
    _paged_kv_cache_scatter_token_tiled = None

__all__ = ['BlockManager', 'TieredBlockManager']


class BlockManager:
    """
    Manages paged KV cache blocks on GPU.

    Each transformer layer has its own KV cache. Blocks are shared
    across layers (same block table), but each layer stores K/V independently.

    Per-layer layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        dim 1: 0=K, 1=V
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda',
        kv_layer_indices: Optional[List[int]] = None,
        per_layer_num_kv_heads: Optional[List[int]] = None,
        per_layer_head_dims: Optional[List[int]] = None,
        kv_layer_sources: Optional[Dict[int, int]] = None,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.per_layer_num_kv_heads = list(
            per_layer_num_kv_heads
            if per_layer_num_kv_heads is not None
            else [num_kv_heads] * num_layers
        )
        self.per_layer_head_dims = list(
            per_layer_head_dims
            if per_layer_head_dims is not None
            else [head_dim] * num_layers
        )
        if len(self.per_layer_num_kv_heads) != num_layers:
            raise ValueError("per_layer_num_kv_heads must have one entry per layer")
        if len(self.per_layer_head_dims) != num_layers:
            raise ValueError("per_layer_head_dims must have one entry per layer")
        self.kv_layer_sources: Dict[int, int] = {
            int(layer_idx): int(source_idx)
            for layer_idx, source_idx in (kv_layer_sources or {}).items()
        }

        if kv_layer_indices is None:
            kv_layer_indices = [
                layer_idx
                for layer_idx in range(num_layers)
                if layer_idx not in self.kv_layer_sources
            ]
        else:
            kv_layer_indices = sorted(
                {
                    int(layer_idx)
                    for layer_idx in kv_layer_indices
                    if 0 <= int(layer_idx) < num_layers
                }
            )
        kv_layer_set = set(kv_layer_indices)
        for source_idx in self.kv_layer_sources.values():
            if 0 <= source_idx < num_layers:
                kv_layer_set.add(source_idx)
        kv_layer_indices = sorted(kv_layer_set)
        self.kv_layer_indices: List[int] = kv_layer_indices
        self.num_kv_layers = len(self.kv_layer_indices)
        self._kv_slot_by_layer: Dict[int, int] = {
            layer_idx: slot_idx
            for slot_idx, layer_idx in enumerate(self.kv_layer_indices)
        }
        self._kv_writable_layers: Set[int] = set(self.kv_layer_indices)
        self._kv_source_by_layer: Dict[int, Optional[int]] = {}
        for layer_idx in range(num_layers):
            if layer_idx not in kv_layer_set and layer_idx not in self.kv_layer_sources:
                # Hybrid architectures such as Qwen 3.5 can have layers that do
                # not use paged KV at all (e.g. linear-attention blocks). Those
                # layers should simply report no KV cache instead of being forced
                # to alias a paged source layer.
                self._kv_source_by_layer[layer_idx] = None
                continue
            source_idx = self.kv_layer_sources.get(layer_idx, layer_idx)
            seen = {layer_idx}
            while source_idx in self.kv_layer_sources and source_idx not in seen:
                seen.add(source_idx)
                source_idx = self.kv_layer_sources[source_idx]
            self._kv_source_by_layer[layer_idx] = source_idx
            if source_idx not in self._kv_slot_by_layer:
                raise ValueError(
                    f"Layer {layer_idx} maps to KV source {source_idx}, "
                    "but that source has no allocated KV cache."
                )

        # Per-layer KV cache pools on GPU
        self.kv_caches: List[torch.Tensor] = [
            torch.zeros(
                num_blocks,
                2,
                int(self.per_layer_num_kv_heads[layer_idx]),
                block_size,
                int(self.per_layer_head_dims[layer_idx]),
                dtype=dtype, device=device
            )
            for layer_idx in self.kv_layer_indices
        ]
        scatter_flag = os.environ.get(
            "MEGAGEMM_TRITON_PREFILL_KV_SCATTER", "1"
        ).strip().lower()
        self._prefill_kv_scatter_requested = scatter_flag in {
            "1", "true", "yes", "on"
        }
        self._prefill_kv_scatter_disabled = False
        self._prefill_kv_scatter_hits = 0
        self._prefill_kv_scatter_failures = 0
        self._prefill_kv_scatter_error = ""
        self._prefill_kv_scatter_token_tiled_disabled = False
        self._prefill_kv_scatter_token_tiled_hits = 0
        self._prefill_kv_scatter_token_tiled_failures = 0
        self._prefill_kv_scatter_token_tiled_error = ""

        # Block allocation
        self.free_blocks: List[int] = list(range(num_blocks))
        self._block_refcounts: List[int] = [0] * num_blocks
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lens: Dict[int, int] = {}
        self.linear_conv_states: Dict[int, Dict[int, torch.Tensor]] = {}
        self.linear_recurrent_states: Dict[int, Dict[int, torch.Tensor]] = {}
        # Exact batch-state views produced by set_linear_state_batch().
        # Continuous batching repeatedly asks for the same active seq_ids; keeping
        # the packed tensor avoids re-stacking one large recurrent state per
        # linear-attention layer every decode burst.
        self._linear_batch_state_cache: Dict[
            Tuple[str, int, Tuple[int, ...]], torch.Tensor
        ] = {}

        # Pre-allocated tensors for decode hot path (reused every step)
        self._block_table_tensor: Optional[torch.Tensor] = None
        self._block_table_seq_key: Optional[Tuple[int, ...]] = None
        self._seq_lens_tensor: Optional[torch.Tensor] = None
        self._seq_lens_seq_key: Optional[Tuple[int, ...]] = None
        self._decode_metadata_override: Optional[Dict[str, object]] = None
        self._kv_write_index_cache: Dict[Tuple[Tuple[int, ...], str], torch.Tensor] = {}
        self._idle_free_order_resets = 0

    def _invalidate_block_table_cache(self) -> None:
        self._block_table_tensor = None
        self._block_table_seq_key = None
        self._kv_write_index_cache.clear()

    def _invalidate_seq_lens_cache(self) -> None:
        self._seq_lens_tensor = None
        self._seq_lens_seq_key = None

    def _claim_free_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")
        block = self.free_blocks.pop()
        self._block_refcounts[block] = 1
        return block

    def _retain_block(self, block: int) -> None:
        block = int(block)
        if block < 0:
            raise ValueError("Cannot retain a non-GPU KV block")
        self._block_refcounts[block] += 1

    def _release_block(self, block: int) -> None:
        block = int(block)
        if block < 0:
            return
        count = int(self._block_refcounts[block])
        if count <= 0:
            # Older or externally-mutated tables should not poison the free list.
            return
        count -= 1
        self._block_refcounts[block] = count
        if count == 0:
            self.free_blocks.append(block)

    def _ensure_writable_block(self, seq_id: int, logical_idx: int) -> int:
        blocks = self.block_tables[seq_id]
        phys = int(blocks[logical_idx])
        if phys < 0:
            return phys
        if int(self._block_refcounts[phys]) <= 1:
            return phys
        new_phys = self._claim_free_block()
        for cache in self.kv_caches:
            cache[new_phys].copy_(cache[phys])
        self._release_block(phys)
        blocks[logical_idx] = new_phys
        self._invalidate_block_table_cache()
        return new_phys

    def _invalidate_linear_batch_state_cache(self, layer_idx: Optional[int] = None) -> None:
        if not self._linear_batch_state_cache:
            return
        if layer_idx is None:
            self._linear_batch_state_cache.clear()
            return
        layer_idx = int(layer_idx)
        for key in list(self._linear_batch_state_cache.keys()):
            if key[1] == layer_idx:
                self._linear_batch_state_cache.pop(key, None)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def bytes_per_block(self) -> int:
        """Bytes consumed by one logical KV block across all cached layers."""
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        return sum(
            2
            * int(self.per_layer_num_kv_heads[layer_idx])
            * self.block_size
            * int(self.per_layer_head_dims[layer_idx])
            * dtype_size
            for layer_idx in self.kv_layer_indices
        )

    def can_allocate(self, num_tokens: int) -> bool:
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        return num_needed <= self.num_free_blocks

    def allocate_sequence(self, seq_id: int, num_tokens: int = 0) -> List[int]:
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists")
        num_needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        if num_needed > self.num_free_blocks:
            raise RuntimeError(
                f"OOM: need {num_needed} blocks, have {self.num_free_blocks}"
            )
        blocks = [self._claim_free_block() for _ in range(num_needed)]
        self.block_tables[seq_id] = blocks
        self.seq_lens[seq_id] = 0
        self.linear_conv_states[seq_id] = {}
        self.linear_recurrent_states[seq_id] = {}
        self._invalidate_block_table_cache()
        self._invalidate_seq_lens_cache()
        return blocks

    def allocate_block(self, seq_id: int) -> int:
        block = self._claim_free_block()
        self.block_tables[seq_id].append(block)
        self._invalidate_block_table_cache()
        return block

    def free_sequence(self, seq_id: int):
        if seq_id in self.block_tables:
            for block in self.block_tables[seq_id]:
                self._release_block(block)
            del self.block_tables[seq_id]
            del self.seq_lens[seq_id]
            self.linear_conv_states.pop(seq_id, None)
            self.linear_recurrent_states.pop(seq_id, None)
            self._invalidate_linear_batch_state_cache()
            self._invalidate_block_table_cache()
            self._invalidate_seq_lens_cache()
            if not self.block_tables:
                # Shared decode graphs are bound to persistent tensor storage, but
                # some CUDA kernels also retain the capture-time physical layout.
                # Restore the initial LIFO order at idle boundaries so an identical
                # request shape receives the same physical block table on reuse.
                self.free_blocks.sort()
                self._idle_free_order_resets += 1

    def fork_sequence_prefix(self, source_seq_id: int, new_seq_id: int, extra_tokens: int = 0) -> List[int]:
        """
        Create a decode-ready child sequence that shares a cached prefix.

        Full prefix blocks are retained by refcount. If generation writes into a
        shared partial final block, the write path performs copy-on-write once and
        then continues on private tail blocks.
        """
        if type(self) is not BlockManager:
            raise NotImplementedError("fork_sequence_prefix is implemented only for GPU BlockManager")
        source_seq_id = int(source_seq_id)
        new_seq_id = int(new_seq_id)
        if source_seq_id not in self.block_tables:
            raise ValueError(f"Source sequence {source_seq_id} not found")
        if new_seq_id in self.block_tables:
            raise ValueError(f"Sequence {new_seq_id} already exists")

        seq_len = int(self.seq_lens[source_seq_id])
        data_blocks = max(1, (max(seq_len, 1) + self.block_size - 1) // self.block_size)
        total_blocks = max(
            data_blocks,
            max(1, (max(seq_len + int(extra_tokens), 1) + self.block_size - 1) // self.block_size),
        )
        source_blocks = self.block_tables[source_seq_id]
        if len(source_blocks) < data_blocks:
            raise RuntimeError("Source sequence block table is shorter than its seq_len")

        extra_needed = total_blocks - data_blocks
        if extra_needed > self.num_free_blocks:
            raise RuntimeError(
                f"OOM: need {extra_needed} private tail blocks, have {self.num_free_blocks}"
            )

        shared_blocks = list(source_blocks[:data_blocks])
        for block in shared_blocks:
            self._retain_block(block)
        private_tail = [self._claim_free_block() for _ in range(extra_needed)]

        self.block_tables[new_seq_id] = shared_blocks + private_tail
        self.seq_lens[new_seq_id] = seq_len
        self.linear_conv_states[new_seq_id] = {
            int(layer_idx): state.detach().clone()
            for layer_idx, state in self.linear_conv_states.get(source_seq_id, {}).items()
        }
        self.linear_recurrent_states[new_seq_id] = {
            int(layer_idx): state.detach().clone()
            for layer_idx, state in self.linear_recurrent_states.get(source_seq_id, {}).items()
        }
        self._invalidate_linear_batch_state_cache()
        self._invalidate_block_table_cache()
        self._invalidate_seq_lens_cache()
        return self.block_tables[new_seq_id]

    def truncate_sequence(self, seq_id: int, new_seq_len: int) -> None:
        """
        Truncate a live sequence in-place to a shorter prefix.

        This is used by MGX Prophet to roll a restored snapshot back to the
        longest common token prefix before replaying the remaining tail tokens.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        cur_len = int(self.seq_lens[seq_id])
        new_seq_len = int(new_seq_len)
        if new_seq_len < 0 or new_seq_len > cur_len:
            raise ValueError(
                f"Invalid truncate length {new_seq_len} for sequence {seq_id} "
                f"with current length {cur_len}"
            )
        if new_seq_len == cur_len:
            return

        if self.linear_conv_states.get(seq_id) or self.linear_recurrent_states.get(seq_id):
            raise ValueError(
                "truncate_sequence does not currently support linear-attention stateful layers"
            )

        keep_blocks = max(1, (max(new_seq_len, 1) + self.block_size - 1) // self.block_size)
        blocks = self.block_tables[seq_id]
        released_blocks = blocks[keep_blocks:]
        if released_blocks:
            for block in released_blocks:
                self._release_block(block)
            self.block_tables[seq_id] = blocks[:keep_blocks]

        self.seq_lens[seq_id] = new_seq_len
        self._invalidate_block_table_cache()
        self._invalidate_seq_lens_cache()

    def get_linear_state(self, seq_id: int, layer_idx: int, device=None):
        conv_state = self.linear_conv_states.get(seq_id, {}).get(layer_idx)
        recurrent_state = self.linear_recurrent_states.get(seq_id, {}).get(layer_idx)
        if device is not None:
            if conv_state is not None:
                conv_state = conv_state.to(device)
            if recurrent_state is not None:
                recurrent_state = recurrent_state.to(device)
        return conv_state, recurrent_state

    def set_linear_state(self, seq_id: int, layer_idx: int, conv_state=None, recurrent_state=None):
        if seq_id not in self.linear_conv_states:
            self.linear_conv_states[seq_id] = {}
        if seq_id not in self.linear_recurrent_states:
            self.linear_recurrent_states[seq_id] = {}
        if conv_state is None:
            self.linear_conv_states[seq_id].pop(layer_idx, None)
        else:
            self.linear_conv_states[seq_id][layer_idx] = conv_state.detach().contiguous()
        if recurrent_state is None:
            self.linear_recurrent_states[seq_id].pop(layer_idx, None)
        else:
            self.linear_recurrent_states[seq_id][layer_idx] = recurrent_state.detach().contiguous()
        self._invalidate_linear_batch_state_cache(layer_idx)

    def get_linear_state_batch(self, seq_ids: list, layer_idx: int, device=None):
        if len(seq_ids) == 1:
            conv_state, recurrent_state = self.get_linear_state(seq_ids[0], layer_idx, device=device)
            if conv_state is not None:
                conv_state = conv_state.unsqueeze(0).clone()
            if recurrent_state is not None:
                recurrent_state = recurrent_state.unsqueeze(0).clone()
            return conv_state, recurrent_state

        seq_key = tuple(int(sid) for sid in seq_ids)
        target_device = torch.device(device) if device is not None else None
        conv_cached = self._linear_batch_state_cache.get(("conv", int(layer_idx), seq_key))
        recurrent_cached = self._linear_batch_state_cache.get(("recurrent", int(layer_idx), seq_key))
        if conv_cached is not None and recurrent_cached is not None:
            if target_device is None:
                return conv_cached, recurrent_cached
            if conv_cached.device == target_device and recurrent_cached.device == target_device:
                return conv_cached, recurrent_cached

        conv_states = []
        recurrent_states = []
        has_conv = False
        has_recurrent = False

        for sid in seq_ids:
            conv_state, recurrent_state = self.get_linear_state(sid, layer_idx, device=device)
            conv_states.append(conv_state)
            recurrent_states.append(recurrent_state)
            has_conv = has_conv or conv_state is not None
            has_recurrent = has_recurrent or recurrent_state is not None

        conv_batch = None
        recurrent_batch = None
        if has_conv:
            template = next(state for state in conv_states if state is not None)
            conv_batch = torch.stack([
                state if state is not None else torch.zeros_like(template)
                for state in conv_states
            ], dim=0)
            self._linear_batch_state_cache[("conv", int(layer_idx), seq_key)] = conv_batch
        if has_recurrent:
            template = next(state for state in recurrent_states if state is not None)
            recurrent_batch = torch.stack([
                state if state is not None else torch.zeros_like(template)
                for state in recurrent_states
            ], dim=0)
            self._linear_batch_state_cache[("recurrent", int(layer_idx), seq_key)] = recurrent_batch
        return conv_batch, recurrent_batch

    def set_linear_state_batch(self, seq_ids: list, layer_idx: int, conv_states=None, recurrent_states=None):
        if len(seq_ids) == 1:
            conv_state = None if conv_states is None else conv_states[0]
            recurrent_state = None if recurrent_states is None else recurrent_states[0]
            self.set_linear_state(seq_ids[0], layer_idx, conv_state, recurrent_state)
            return

        self._invalidate_linear_batch_state_cache(layer_idx)
        seq_key = tuple(int(sid) for sid in seq_ids)
        conv_batch = None
        recurrent_batch = None
        if conv_states is None:
            conv_list = [None] * len(seq_ids)
        else:
            conv_batch = conv_states.detach()
            if not conv_batch.is_contiguous():
                conv_batch = conv_batch.contiguous()
            conv_list = list(conv_batch.unbind(0))
        if recurrent_states is None:
            recurrent_list = [None] * len(seq_ids)
        else:
            recurrent_batch = recurrent_states.detach()
            if not recurrent_batch.is_contiguous():
                recurrent_batch = recurrent_batch.contiguous()
            recurrent_list = list(recurrent_batch.unbind(0))
        for sid, conv_state, recurrent_state in zip(seq_ids, conv_list, recurrent_list):
            sid = int(sid)
            if sid not in self.linear_conv_states:
                self.linear_conv_states[sid] = {}
            if sid not in self.linear_recurrent_states:
                self.linear_recurrent_states[sid] = {}
            if conv_state is None:
                self.linear_conv_states[sid].pop(layer_idx, None)
            else:
                self.linear_conv_states[sid][layer_idx] = conv_state.detach().contiguous()
            if recurrent_state is None:
                self.linear_recurrent_states[sid].pop(layer_idx, None)
            else:
                self.linear_recurrent_states[sid][layer_idx] = recurrent_state.detach().contiguous()
        if conv_batch is not None:
            self._linear_batch_state_cache[("conv", int(layer_idx), seq_key)] = conv_batch
        if recurrent_batch is not None:
            self._linear_batch_state_cache[("recurrent", int(layer_idx), seq_key)] = recurrent_batch

    def _kv_slot(self, layer_idx: int) -> Optional[int]:
        source_idx = self._kv_source_by_layer.get(int(layer_idx), int(layer_idx))
        if source_idx is None:
            return None
        return self._kv_slot_by_layer.get(source_idx)

    def _get_kv_write_index_tensor(
        self,
        phys_blocks: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        key_blocks = tuple(int(block) for block in phys_blocks)
        key = (key_blocks, str(device))
        cached = self._kv_write_index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.tensor(key_blocks, dtype=torch.long, device=device)
            self._kv_write_index_cache[key] = cached
        return cached

    def write_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    ):
        """Write K,V for new tokens into a specific layer's cache (vectorized)."""
        if int(layer_idx) not in self._kv_writable_layers:
            return
        slot = self._kv_slot(layer_idx)
        if slot is None:
            return
        num_new = k.shape[0]
        cur_len = self.seq_lens[seq_id]
        blocks = self.block_tables[seq_id]
        cache = self.kv_caches[slot]

        # Ensure enough blocks allocated
        total_needed = cur_len + num_new
        blocks_needed = (total_needed + self.block_size - 1) // self.block_size
        while blocks_needed > len(blocks):
            self.allocate_block(seq_id)
            blocks = self.block_tables[seq_id]

        # For single token (decode), fast path — no loop
        if num_new == 1:
            pos = cur_len
            blk_i = pos // self.block_size
            phys = self._ensure_writable_block(seq_id, blk_i)
            off = pos % self.block_size
            cache[phys, 0, :, off, :] = k[0]
            cache[phys, 1, :, off, :] = v[0]
            return

        # For multiple tokens (prefill), vectorize full physical blocks and keep
        # the scalar chunk path only for unaligned head/tail fragments.
        tok_start = 0

        first_off = cur_len % self.block_size
        if first_off:
            take = min(self.block_size - first_off, num_new)
            tok_end = tok_start + take
            blk_i = cur_len // self.block_size
            phys = self._ensure_writable_block(seq_id, blk_i)
            cache[phys, 0, :, first_off:first_off + take, :] = (
                k[tok_start:tok_end].transpose(0, 1)
            )
            cache[phys, 1, :, first_off:first_off + take, :] = (
                v[tok_start:tok_end].transpose(0, 1)
            )
            tok_start = tok_end

        remaining = num_new - tok_start
        full_blocks = remaining // self.block_size
        if full_blocks:
            blk_start = (cur_len + tok_start) // self.block_size
            blk_end = blk_start + full_blocks
            phys_blocks = [
                self._ensure_writable_block(seq_id, blk_i)
                for blk_i in range(blk_start, blk_end)
            ]
            phys_idx = self._get_kv_write_index_tensor(phys_blocks, cache.device)
            tok_end = tok_start + full_blocks * self.block_size
            k_blocks = (
                k[tok_start:tok_end]
                .reshape(full_blocks, self.block_size, k.shape[1], k.shape[2])
                .permute(0, 2, 1, 3)
            )
            v_blocks = (
                v[tok_start:tok_end]
                .reshape(full_blocks, self.block_size, v.shape[1], v.shape[2])
                .permute(0, 2, 1, 3)
            )
            cache[phys_idx, 0, :, :, :] = k_blocks
            cache[phys_idx, 1, :, :, :] = v_blocks
            tok_start = tok_end

        while tok_start < num_new:
            pos = cur_len + tok_start
            blk_i = pos // self.block_size
            off = pos % self.block_size
            take = min(self.block_size - off, num_new - tok_start)
            tok_end = tok_start + take

            phys = self._ensure_writable_block(seq_id, blk_i)
            cache[phys, 0, :, off:off + take, :] = k[tok_start:tok_end].transpose(0, 1)
            cache[phys, 1, :, off:off + take, :] = v[tok_start:tok_end].transpose(0, 1)
            tok_start = tok_end

    def write_kv_prefill_packed(
        self,
        seq_ids: list,
        layer_idx: int,
        k_all: torch.Tensor,    # [total_tokens, num_kv_heads, head_dim]
        v_all: torch.Tensor,    # [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: torch.Tensor,  # [num_seqs + 1]
        kv_mapping: tuple = None,  # pre-computed (all_phys, all_offs)
        tokens_per_program: int = 1,
    ):
        """Write KV for ALL packed sequences in ONE vectorized scatter.

        If kv_mapping is provided, skips index computation (28x speedup
        when called per-layer with the same mapping).
        """
        if int(layer_idx) not in self._kv_writable_layers:
            return
        slot = self._kv_slot(layer_idx)
        if slot is None:
            return
        cache = self.kv_caches[slot]

        if kv_mapping is not None:
            all_phys, all_offs = kv_mapping
        else:
            all_phys, all_offs = self.compute_kv_mapping(
                seq_ids, cu_seqlens, cache.device,
            )

        token_tile = int(tokens_per_program)
        if (
            token_tile > 1
            and self._prefill_kv_scatter_requested
            and not self._prefill_kv_scatter_token_tiled_disabled
            and _paged_kv_cache_scatter_token_tiled is not None
        ):
            try:
                if _paged_kv_cache_scatter_token_tiled(
                    k_all,
                    v_all,
                    cache,
                    all_phys,
                    all_offs,
                    tokens_per_program=token_tile,
                ):
                    self._prefill_kv_scatter_token_tiled_hits += 1
                    return
            except Exception as exc:
                self._prefill_kv_scatter_token_tiled_disabled = True
                self._prefill_kv_scatter_token_tiled_failures += 1
                self._prefill_kv_scatter_token_tiled_error = (
                    f"{type(exc).__name__}: {exc}"
                )

        # Keep the existing one-token kernel and indexed assignment as fallbacks.
        if (
            self._prefill_kv_scatter_requested
            and not self._prefill_kv_scatter_disabled
            and _paged_kv_cache_scatter is not None
        ):
            try:
                if _paged_kv_cache_scatter(
                    k_all,
                    v_all,
                    cache,
                    all_phys,
                    all_offs,
                ):
                    self._prefill_kv_scatter_hits += 1
                    return
            except Exception as exc:
                self._prefill_kv_scatter_disabled = True
                self._prefill_kv_scatter_failures += 1
                self._prefill_kv_scatter_error = f"{type(exc).__name__}: {exc}"

        cache[all_phys, 0, :, all_offs, :] = k_all
        cache[all_phys, 1, :, all_offs, :] = v_all

    def prefill_kv_scatter_stats(self) -> dict:
        return {
            "requested": bool(self._prefill_kv_scatter_requested),
            "available": _paged_kv_cache_scatter is not None,
            "disabled": bool(self._prefill_kv_scatter_disabled),
            "hits": int(self._prefill_kv_scatter_hits),
            "failures": int(self._prefill_kv_scatter_failures),
            "error": str(self._prefill_kv_scatter_error),
            "token_tiled_available": _paged_kv_cache_scatter_token_tiled is not None,
            "token_tiled_disabled": bool(
                self._prefill_kv_scatter_token_tiled_disabled
            ),
            "token_tiled_hits": int(self._prefill_kv_scatter_token_tiled_hits),
            "token_tiled_failures": int(
                self._prefill_kv_scatter_token_tiled_failures
            ),
            "token_tiled_error": str(self._prefill_kv_scatter_token_tiled_error),
        }

    def compute_kv_mapping(
        self,
        seq_ids: list,
        cu_seqlens: torch.Tensor,
        device: torch.device,
        seq_lengths: Optional[List[int]] = None,
    ) -> tuple:
        """Pre-compute token→(physical_block, slot_offset) mapping.

        Call ONCE per prefill chunk, reuse for all 28 layers.
        ``seq_lengths`` avoids synchronizing each cumulative GPU offset back
        to the host when the scheduler already owns the lengths as integers.
        Returns (all_phys, all_offs) tensors.
        """
        phys_list = []
        offs_list = []

        for i, sid in enumerate(seq_ids):
            if seq_lengths is None:
                start = int(cu_seqlens[i].item())
                end = int(cu_seqlens[i + 1].item())
                seq_len = end - start
            else:
                seq_len = int(seq_lengths[i])
            cur_len = self.seq_lens[sid]
            blocks = self.block_tables[sid]

            positions = torch.arange(cur_len, cur_len + seq_len, device=device)
            blk_idx = positions // self.block_size
            blk_off = positions % self.block_size

            block_tensor = torch.tensor(blocks, dtype=torch.long, device=device)
            phys = block_tensor[blk_idx]

            phys_list.append(phys)
            offs_list.append(blk_off)

        return torch.cat(phys_list), torch.cat(offs_list)

    def write_kv_decode_batch(
        self,
        layer_idx: int,
        seq_ids: list,
        k: torch.Tensor,  # [num_seqs, 1, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_seqs, 1, num_kv_heads, head_dim]
    ):
        """
        Write 1 new K,V token per sequence for a specific layer — fully vectorized.
        Replaces the per-sequence Python for-loop in decode_step.
        """
        if int(layer_idx) not in self._kv_writable_layers:
            return
        slot = self._kv_slot(layer_idx)
        if slot is None:
            return
        cache = self.kv_caches[slot]
        num_seqs = len(seq_ids)

        # Compute physical block + offset for each sequence
        phys_blocks = torch.empty(num_seqs, dtype=torch.long, device=k.device)
        offsets = torch.empty(num_seqs, dtype=torch.long, device=k.device)

        for i, sid in enumerate(seq_ids):
            pos = self.seq_lens[sid]
            blocks = self.block_tables[sid]
            # Ensure block is allocated
            blk_idx = pos // self.block_size
            if blk_idx >= len(blocks):
                self.allocate_block(sid)
                blocks = self.block_tables[sid]
            phys_blocks[i] = self._ensure_writable_block(sid, blk_idx)
            offsets[i] = pos % self.block_size

        # Vectorized scatter: write all seqs at once
        # k shape: [num_seqs, 1, num_kv_heads, head_dim] → squeeze → [num_seqs, num_kv_heads, head_dim]
        k_sq = k[:, 0]  # [num_seqs, num_kv_heads, head_dim]
        v_sq = v[:, 0]

        # Write via advanced indexing — one operation per KV
        # cache[phys, 0, :, offs, :] → result shape [N, H, D] (non-contiguous adv idx)
        cache[phys_blocks, 0, :, offsets, :] = k_sq  # [N, H, D]
        cache[phys_blocks, 1, :, offsets, :] = v_sq

    def advance_seq_len_batch(self, seq_ids: list, num_tokens: int = 1):
        """Advance sequence length for all sequences — batch version.

        Optimized: increments cached _seq_lens_tensor in-place instead
        of invalidating it (avoids tensor reconstruction every decode step).
        """
        for sid in seq_ids:
            self.seq_lens[sid] += num_tokens
        override = self._decode_metadata_override
        if (
            override is not None
            and int(override.get("num_seqs", -1)) == len(seq_ids)
            and override.get("seq_lens") is not None
        ):
            override["seq_lens"].add_(num_tokens)
            return
        # In-place increment if cached tensor exists and matches shape
        if (self._seq_lens_tensor is not None and
            self._seq_lens_tensor.shape[0] == len(seq_ids) and
            self._seq_lens_seq_key == tuple(int(sid) for sid in seq_ids)):
            is_inference = getattr(torch, "is_inference", None)
            if is_inference is not None and is_inference(self._seq_lens_tensor):
                # Snapshots/prefill paths can leave an inference tensor cached.
                # CUDA Graph capture mutates this tensor, so normalize it first.
                with torch.inference_mode(False):
                    self._seq_lens_tensor = self._seq_lens_tensor.clone()
            self._seq_lens_tensor += num_tokens
        else:
            self._seq_lens_tensor = None  # shape mismatch, force rebuild
            self._seq_lens_seq_key = None

    def advance_seq_len(self, seq_id: int, num_tokens: int):
        """Advance sequence length after all layers have written KV."""
        self.seq_lens[seq_id] += num_tokens
        # Invalidate cached tensors
        self._seq_lens_tensor = None
        self._seq_lens_seq_key = None

    def get_kv_cache(self, layer_idx: int) -> Optional[torch.Tensor]:
        slot = self._kv_slot(layer_idx)
        if slot is None:
            return None
        return self.kv_caches[slot]

    def get_block_table_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        """Get block table tensor (cached for decode reuse)."""
        seq_key = tuple(int(sid) for sid in seq_ids)
        override = self._decode_metadata_override
        if (
            override is not None
            and int(override.get("num_seqs", -1)) == len(seq_ids)
            and override.get("block_table") is not None
        ):
            return override["block_table"]
        if self._block_table_tensor is not None and self._block_table_seq_key == seq_key:
            return self._block_table_tensor
        max_blocks = max(len(self.block_tables[s]) for s in seq_ids)
        # decode_step runs under torch.inference_mode(), but CUDA Graph replay
        # needs these cached metadata tensors to remain normal mutable tensors.
        # Inference tensors reject later in-place updates during capture/replay.
        with torch.inference_mode(False):
            table = torch.zeros(
                len(seq_ids), max_blocks, dtype=torch.int32, device=self.device
            )
            for i, sid in enumerate(seq_ids):
                blks = self.block_tables[sid]
                table[i, :len(blks)] = torch.tensor(
                    blks, dtype=torch.int32, device=self.device
                )
        self._block_table_tensor = table
        self._block_table_seq_key = seq_key
        return table

    def get_seq_lens_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        """Get seq lens tensor (cached, invalidated on advance)."""
        seq_key = tuple(int(sid) for sid in seq_ids)
        override = self._decode_metadata_override
        if (
            override is not None
            and int(override.get("num_seqs", -1)) == len(seq_ids)
            and override.get("seq_lens") is not None
        ):
            return override["seq_lens"]
        if self._seq_lens_tensor is not None and self._seq_lens_seq_key == seq_key:
            return self._seq_lens_tensor
        # See get_block_table_tensor(): this tensor is incremented in-place in
        # advance_seq_len_batch(), including from CUDA Graph capture paths.
        with torch.inference_mode(False):
            self._seq_lens_tensor = torch.tensor(
                [self.seq_lens[s] for s in seq_ids],
                dtype=torch.int32, device=self.device
            )
        self._seq_lens_seq_key = seq_key
        return self._seq_lens_tensor

    def set_decode_metadata_override(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_decode_blocks: Optional[int] = None,
    ) -> None:
        self._decode_metadata_override = {
            "block_table": block_table,
            "seq_lens": seq_lens,
            "num_seqs": int(seq_lens.shape[0]),
            "max_decode_blocks": (
                None if max_decode_blocks is None else int(max_decode_blocks)
            ),
        }

    def clear_decode_metadata_override(self) -> None:
        self._decode_metadata_override = None

    def get_decode_max_blocks_override(self, seq_ids: List[int]) -> Optional[int]:
        override = self._decode_metadata_override
        if override is None or int(override.get("num_seqs", -1)) != len(seq_ids):
            return None
        value = override.get("max_decode_blocks")
        return None if value is None else int(value)

    def memory_usage_mb(self) -> float:
        total = sum(c.nelement() * c.element_size() for c in self.kv_caches)
        return total / (1024 * 1024)

    def serialize_sequence(self, seq_id: int) -> dict:
        """
        Serialize KV cache for a sequence into a saveable dict.

        Extracts per-layer block data and moves to CPU for storage.
        The returned dict can be saved with torch.save() or compressed.

        Args:
            seq_id: Sequence to serialize

        Returns:
            Dict with KV data, seq_len, and model config for validation
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        blocks = self.block_tables[seq_id]
        seq_len = self.seq_lens[seq_id]
        used_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Extract per-layer KV data using fancy indexing
        block_indices = blocks[:used_blocks]
        idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)

        kv_data = [None] * self.num_layers
        kv_data_by_layer = {}
        for layer_idx in self.kv_layer_indices:
            slot = self._kv_slot_by_layer[layer_idx]
            # [used_blocks, 2, num_kv_heads, block_size, head_dim]
            layer_data = self.kv_caches[slot][idx_tensor].cpu()
            kv_data[layer_idx] = layer_data
            kv_data_by_layer[layer_idx] = layer_data

        return {
            'seq_len': seq_len,
            'num_layers': self.num_layers,
            'kv_layer_indices': list(self.kv_layer_indices),
            'block_size': self.block_size,
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'per_layer_num_kv_heads': list(self.per_layer_num_kv_heads),
            'per_layer_head_dims': list(self.per_layer_head_dims),
            'kv_layer_sources': dict(self.kv_layer_sources),
            'dtype': str(self.dtype),
            'kv_data': kv_data,
            'kv_data_by_layer': kv_data_by_layer,
            'linear_conv_states': {
                layer_idx: state.cpu()
                for layer_idx, state in self.linear_conv_states.get(seq_id, {}).items()
            },
            'linear_recurrent_states': {
                layer_idx: state.cpu()
                for layer_idx, state in self.linear_recurrent_states.get(seq_id, {}).items()
            },
        }

    def _snapshot_kv_data_by_layer(self, snapshot: dict) -> dict[int, torch.Tensor]:
        if 'kv_data_by_layer' in snapshot:
            return {
                int(layer_idx): layer_data
                for layer_idx, layer_data in snapshot['kv_data_by_layer'].items()
                if layer_data is not None
            }
        kv_data = snapshot.get('kv_data', [])
        return {
            layer_idx: layer_data
            for layer_idx, layer_data in enumerate(kv_data)
            if layer_data is not None
        }

    def _validate_snapshot_layout(self, snapshot: dict) -> None:
        if snapshot['num_layers'] != self.num_layers:
            raise ValueError(
                f"Layer mismatch: snapshot has {snapshot['num_layers']}, "
                f"model has {self.num_layers}"
            )
        if snapshot['block_size'] != self.block_size:
            raise ValueError(
                f"Block size mismatch: snapshot has {snapshot['block_size']}, "
                f"manager has {self.block_size}"
            )
        snapshot_heads = snapshot.get('per_layer_num_kv_heads')
        snapshot_dims = snapshot.get('per_layer_head_dims')
        if snapshot_heads is not None or snapshot_dims is not None:
            if list(snapshot_heads or []) != self.per_layer_num_kv_heads:
                raise ValueError("Per-layer KV head layout mismatch")
            if list(snapshot_dims or []) != self.per_layer_head_dims:
                raise ValueError("Per-layer KV head-dim layout mismatch")
        elif snapshot['num_kv_heads'] != self.num_kv_heads:
            raise ValueError(
                f"KV heads mismatch: snapshot has {snapshot['num_kv_heads']}, "
                f"model has {self.num_kv_heads}"
            )

        snapshot_kv_layers = snapshot.get('kv_layer_indices')
        if snapshot_kv_layers is not None:
            snapshot_kv_layers = [int(layer_idx) for layer_idx in snapshot_kv_layers]
            if snapshot_kv_layers != self.kv_layer_indices:
                raise ValueError(
                    f"KV layer mismatch: snapshot has {snapshot_kv_layers}, "
                    f"manager has {self.kv_layer_indices}"
                )

    def deserialize_sequence(self, seq_id: int, snapshot: dict, extra_tokens: int = 0) -> None:
        """
        Restore KV cache for a sequence from a serialized snapshot.

        Allocates fresh blocks and writes the saved data back.
        Pre-allocates extra blocks for future decode if extra_tokens > 0.

        Args:
            seq_id: Sequence ID to restore into (must not exist)
            snapshot: Dict from serialize_sequence()
            extra_tokens: Extra tokens to pre-allocate blocks for (for decode headroom)
        """
        self._validate_snapshot_layout(snapshot)

        seq_len = snapshot['seq_len']
        kv_data_by_layer = self._snapshot_kv_data_by_layer(snapshot)
        num_data_blocks = 0
        if kv_data_by_layer:
            first_layer_data = next(iter(kv_data_by_layer.values()))
            num_data_blocks = int(first_layer_data.shape[0])

        # Allocate blocks (including headroom for future decode)
        alloc_tokens = seq_len + extra_tokens
        self.allocate_sequence(seq_id, num_tokens=alloc_tokens)
        blocks = self.block_tables[seq_id]

        # Write data per layer
        if num_data_blocks > 0:
            idx_tensor = torch.tensor(
                blocks[:num_data_blocks], dtype=torch.long, device=self.device
            )
            for layer_idx, layer_data in kv_data_by_layer.items():
                slot = self._kv_slot(layer_idx)
                if slot is None:
                    raise ValueError(
                        f"Snapshot contains KV data for layer {layer_idx}, "
                        "but this manager has no KV cache for that layer."
                    )
                self.kv_caches[slot][idx_tensor] = layer_data.to(self.device)

        # Set sequence length and invalidate cached tensors
        self.seq_lens[seq_id] = seq_len
        self.linear_conv_states[seq_id] = {
            int(layer_idx): state.to(self.device)
            for layer_idx, state in snapshot.get('linear_conv_states', {}).items()
        }
        self.linear_recurrent_states[seq_id] = {
            int(layer_idx): state.to(self.device)
            for layer_idx, state in snapshot.get('linear_recurrent_states', {}).items()
        }
        self._invalidate_linear_batch_state_cache()
        self._block_table_tensor = None
        self._block_table_seq_key = None
        self._seq_lens_tensor = None
        self._seq_lens_seq_key = None

    def deserialize_sequences(self, restore_items: list[dict]) -> None:
        """
        Restore multiple snapshots in a layer-batched pass.

        Prophet repeated-prompt hits restore the same shape for every request.
        Writing all restored blocks for one layer at a time avoids the expensive
        seq-by-seq x layer-by-layer Python loop.
        """
        if not restore_items:
            return

        prepared = []
        allocated: list[int] = []
        try:
            for item in restore_items:
                seq_id = int(item['seq_id'])
                snapshot = item['snapshot']
                extra_tokens = int(item.get('extra_tokens', 0) or 0)
                self._validate_snapshot_layout(snapshot)

                seq_len = int(snapshot['seq_len'])
                kv_data_by_layer = self._snapshot_kv_data_by_layer(snapshot)
                num_data_blocks = 0
                if kv_data_by_layer:
                    first_layer_data = next(iter(kv_data_by_layer.values()))
                    num_data_blocks = int(first_layer_data.shape[0])

                self.allocate_sequence(seq_id, num_tokens=seq_len + extra_tokens)
                allocated.append(seq_id)
                blocks = self.block_tables[seq_id]
                prepared.append(
                    {
                        'seq_id': seq_id,
                        'snapshot': snapshot,
                        'seq_len': seq_len,
                        'blocks': blocks,
                        'num_data_blocks': num_data_blocks,
                        'kv_data_by_layer': kv_data_by_layer,
                    }
                )

            for layer_idx in self.kv_layer_indices:
                slot = self._kv_slot(layer_idx)
                if slot is None:
                    continue
                phys_blocks = []
                layer_chunks = []
                for item in prepared:
                    layer_data = item['kv_data_by_layer'].get(layer_idx)
                    if layer_data is None:
                        continue
                    num_data_blocks = int(item['num_data_blocks'])
                    if num_data_blocks <= 0:
                        continue
                    phys_blocks.extend(item['blocks'][:num_data_blocks])
                    layer_chunks.append(layer_data.to(self.device, non_blocking=True))
                if not layer_chunks:
                    continue
                idx_tensor = torch.tensor(phys_blocks, dtype=torch.long, device=self.device)
                if len(layer_chunks) == 1:
                    layer_batch = layer_chunks[0]
                else:
                    layer_batch = torch.cat(layer_chunks, dim=0)
                self.kv_caches[slot][idx_tensor] = layer_batch

            for item in prepared:
                seq_id = int(item['seq_id'])
                snapshot = item['snapshot']
                self.seq_lens[seq_id] = int(item['seq_len'])
                self.linear_conv_states[seq_id] = {
                    int(layer_idx): state.to(self.device)
                    for layer_idx, state in snapshot.get('linear_conv_states', {}).items()
                }
                self.linear_recurrent_states[seq_id] = {
                    int(layer_idx): state.to(self.device)
                    for layer_idx, state in snapshot.get('linear_recurrent_states', {}).items()
                }
        except Exception:
            for seq_id in allocated:
                try:
                    self.free_sequence(seq_id)
                except Exception:
                    pass
            raise

        self._invalidate_linear_batch_state_cache()
        self._block_table_tensor = None
        self._block_table_seq_key = None
        self._seq_lens_tensor = None
        self._seq_lens_seq_key = None

    def __repr__(self) -> str:
        used = self.num_blocks - self.num_free_blocks
        return (
            f"BlockManager(layers={self.num_layers}, kv_layers={self.num_kv_layers}, "
            f"blocks={self.num_blocks}, "
            f"used={used}, free={self.num_free_blocks}, "
            f"block_size={self.block_size}, mem={self.memory_usage_mb():.0f}MB)"
        )


class TieredBlockManager(BlockManager):
    """
    🔄 GPU+CPU Tiered KV Cache Manager.

    Extends BlockManager with CPU offloading for KV cache blocks.
    Cold blocks (old tokens) are evicted to pinned CPU memory,
    freeing GPU VRAM for hot blocks (recent tokens).

    Before the Triton paged attention kernel runs, all needed blocks
    are prefetched back to GPU via async DMA on a separate CUDA stream.

    Memory hierarchy:
        GPU VRAM (fast) ←→ CPU RAM pinned (warm)

    The Triton kernel sees NO changes — it always reads from GPU kv_caches.
    All swapping is transparent and happens before/after kernel calls.

    Usage:
        manager = TieredBlockManager(
            num_layers=32, num_gpu_blocks=256, num_cpu_blocks=2048,
            block_size=16, num_kv_heads=8, head_dim=128,
        )

    Author: Gabriel Yogi
    """

    def __init__(
        self,
        num_layers: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda',
        gpu_window: int = 64,
        kv_layer_indices: Optional[List[int]] = None,
        per_layer_num_kv_heads: Optional[List[int]] = None,
        per_layer_head_dims: Optional[List[int]] = None,
        kv_layer_sources: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            num_gpu_blocks: KV cache blocks on GPU (hot)
            num_cpu_blocks: KV cache blocks on CPU (warm)
            block_size: Tokens per block
            num_kv_heads: Number of KV attention heads
            head_dim: Head dimension
            dtype: Data type (FP16/BF16)
            device: GPU device
            gpu_window: Min blocks to keep on GPU per sequence (recent tokens)
        """
        # Initialize GPU pool via parent
        super().__init__(
            num_layers=num_layers,
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            kv_layer_indices=kv_layer_indices,
            per_layer_num_kv_heads=per_layer_num_kv_heads,
            per_layer_head_dims=per_layer_head_dims,
            kv_layer_sources=kv_layer_sources,
        )

        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        # Clamp gpu_window: must leave at least 25% of GPU blocks as reserve
        # for eviction headroom. Otherwise prefill of long prompts deadlocks.
        max_window = max(1, int(num_gpu_blocks * 0.75))
        if gpu_window > max_window:
            print(f"  ⚠️  gpu_window={gpu_window} clamped to {max_window} "
                  f"(75% of {num_gpu_blocks} GPU blocks)")
            gpu_window = max_window
        self.gpu_window = gpu_window

        # CPU block pool — pinned memory for fast DMA
        self.cpu_kv_caches: List[torch.Tensor] = [
            torch.zeros(
                num_cpu_blocks,
                2,
                int(self.per_layer_num_kv_heads[layer_idx]),
                block_size,
                int(self.per_layer_head_dims[layer_idx]),
                dtype=dtype, device='cpu',
            ).pin_memory()
            for layer_idx in self.kv_layer_indices
        ]

        # CPU block allocation
        self.cpu_free_blocks: List[int] = list(range(num_cpu_blocks))

        # Block location tracking:
        # Maps (seq_id, logical_block_index) → ('gpu', phys_id) or ('cpu', phys_id)
        # The parent's block_tables stores: seq_id → [gpu_phys_block_ids...]
        # We add a separate mapping for CPU blocks.
        #
        # Design: block_tables[seq_id] always holds the CURRENT gpu physical IDs.
        # cpu_block_map[seq_id][logical_idx] = cpu_phys_id (only for offloaded blocks)
        self.cpu_block_map: Dict[int, Dict[int, int]] = {}  # seq_id → {logical_idx: cpu_phys}

        # Async copy stream
        self._copy_stream = torch.cuda.Stream(device=device)

        # Stats
        self._evict_count = 0
        self._fetch_count = 0
        self._evict_time = 0.0
        self._fetch_time = 0.0

    @property
    def num_cpu_free_blocks(self) -> int:
        return len(self.cpu_free_blocks)

    @property
    def total_blocks(self) -> int:
        return self.num_gpu_blocks + self.num_cpu_blocks

    def allocate_sequence(self, seq_id: int, num_tokens: int = 0) -> List[int]:
        """Allocate sequence. Uses GPU blocks first, overflows to CPU."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists")

        num_needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        total_available = self.num_free_blocks + self.num_cpu_free_blocks

        if num_needed > total_available:
            raise RuntimeError(
                f"OOM: need {num_needed} blocks, have {self.num_free_blocks} GPU "
                f"+ {self.num_cpu_free_blocks} CPU = {total_available} total"
            )

        # Allocate from GPU first
        gpu_alloc = min(num_needed, self.num_free_blocks)
        blocks = [self.free_blocks.pop() for _ in range(gpu_alloc)]

        self.block_tables[seq_id] = blocks
        self.seq_lens[seq_id] = 0
        self.cpu_block_map[seq_id] = {}

        # If we need more than GPU can provide, allocate CPU blocks
        cpu_needed = num_needed - gpu_alloc
        if cpu_needed > 0:
            for logical_idx in range(gpu_alloc, num_needed):
                cpu_phys = self.cpu_free_blocks.pop()
                self.cpu_block_map[seq_id][logical_idx] = cpu_phys
                # Placeholder -1 in block_tables (not on GPU yet)
                blocks.append(-1)

        return blocks

    def allocate_block(self, seq_id: int) -> int:
        """
        Allocate one more block for a sequence. GPU first, tries eviction,
        then CPU fallback.

        IMPORTANT: Blocks allocated here may be used immediately by write_kv,
        which does cache[phys, ...]. So we strongly prefer GPU blocks.
        If GPU is full, try evicting a cold block to free a GPU slot first.
        """
        logical_idx = len(self.block_tables[seq_id])

        if self.free_blocks:
            # GPU block available — fast path
            block = self.free_blocks.pop()
            self.block_tables[seq_id].append(block)
            return block

        # GPU full — try evicting a cold block to free a GPU slot
        evicted = self._evict_one_cold_block(exclude_seqs={seq_id})
        if not evicted:
            evicted = self._evict_one_cold_block_from_active([seq_id])

        if self.free_blocks:
            # Eviction freed a GPU slot
            block = self.free_blocks.pop()
            self.block_tables[seq_id].append(block)
            return block
        elif self.cpu_free_blocks:
            # Last resort: allocate on CPU (will need ensure_blocks_on_gpu later)
            cpu_phys = self.cpu_free_blocks.pop()
            self.cpu_block_map[seq_id][logical_idx] = cpu_phys
            self.block_tables[seq_id].append(-1)  # placeholder
            return -1
        else:
            raise RuntimeError(
                f"OOM: no free blocks on GPU ({self.num_gpu_blocks}) "
                f"or CPU ({self.num_cpu_blocks})"
            )

    def free_sequence(self, seq_id: int):
        """Free all blocks (GPU + CPU) for a sequence."""
        if seq_id in self.block_tables:
            # Free GPU blocks (skip placeholders -1)
            for blk in self.block_tables[seq_id]:
                if blk >= 0:
                    self.free_blocks.append(blk)
            del self.block_tables[seq_id]
            del self.seq_lens[seq_id]

            # Free CPU blocks
            if seq_id in self.cpu_block_map:
                for cpu_phys in self.cpu_block_map[seq_id].values():
                    self.cpu_free_blocks.append(cpu_phys)
                del self.cpu_block_map[seq_id]

            self.linear_conv_states.pop(seq_id, None)
            self.linear_recurrent_states.pop(seq_id, None)
            self._invalidate_linear_batch_state_cache()

            self._block_table_tensor = None
            self._seq_lens_tensor = None

    def truncate_sequence(self, seq_id: int, new_seq_len: int) -> None:
        """
        Truncate a live sequence in-place to a shorter prefix.

        For tiered KV, this releases both GPU-resident and CPU-offloaded blocks
        that belong entirely to the truncated suffix.
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        cur_len = int(self.seq_lens[seq_id])
        new_seq_len = int(new_seq_len)
        if new_seq_len < 0 or new_seq_len > cur_len:
            raise ValueError(
                f"Invalid truncate length {new_seq_len} for sequence {seq_id} "
                f"with current length {cur_len}"
            )
        if new_seq_len == cur_len:
            return

        if self.linear_conv_states.get(seq_id) or self.linear_recurrent_states.get(seq_id):
            raise ValueError(
                "truncate_sequence does not currently support linear-attention stateful layers"
            )

        keep_blocks = max(1, (max(new_seq_len, 1) + self.block_size - 1) // self.block_size)
        blocks = self.block_tables[seq_id]

        for logical_idx in range(keep_blocks, len(blocks)):
            gpu_phys = blocks[logical_idx]
            if gpu_phys >= 0:
                self.free_blocks.append(gpu_phys)
            cpu_phys = self.cpu_block_map.get(seq_id, {}).pop(logical_idx, None)
            if cpu_phys is not None:
                self.cpu_free_blocks.append(cpu_phys)

        self.block_tables[seq_id] = blocks[:keep_blocks]
        self.seq_lens[seq_id] = new_seq_len
        self._block_table_tensor = None
        self._seq_lens_tensor = None

    def ensure_blocks_on_gpu(self, seq_ids: List[int]):
        """
        Ensure ALL blocks for given sequences are on GPU before kernel runs.

        Paged attention reads the FULL KV history, so every block with
        data must be on GPU. If total blocks exceed GPU capacity, this
        will raise an error.
        """
        fetch_needed: List[tuple] = []  # (seq_id, logical_idx, cpu_phys)

        for sid in seq_ids:
            blocks = self.block_tables[sid]
            for logical_idx, gpu_phys in enumerate(blocks):
                if gpu_phys < 0 and logical_idx in self.cpu_block_map.get(sid, {}):
                    cpu_phys = self.cpu_block_map[sid][logical_idx]
                    fetch_needed.append((sid, logical_idx, cpu_phys))

        if not fetch_needed:
            return  # All blocks already on GPU — fast path

        t0 = time.perf_counter()

        # Ensure we have enough GPU slots — evict cold blocks if needed
        while len(self.free_blocks) < len(fetch_needed):
            evicted = self._evict_one_cold_block(exclude_seqs=set(seq_ids))
            if not evicted:
                # Try evicting cold blocks from the active sequences themselves
                evicted = self._evict_one_cold_block_from_active(seq_ids)
                if not evicted:
                    raise RuntimeError(
                        f"Cannot free GPU blocks: need {len(fetch_needed)}, "
                        f"have {len(self.free_blocks)}. "
                        f"Increase num_gpu_blocks or gpu_window."
                    )

        # Fetch CPU → GPU using async DMA
        with torch.cuda.stream(self._copy_stream):
            for sid, logical_idx, cpu_phys in fetch_needed:
                gpu_phys = self.free_blocks.pop()

                # Copy data: CPU pool → GPU pool for all layers
                for layer_idx in range(self.num_kv_layers):
                    self.kv_caches[layer_idx][gpu_phys].copy_(
                        self.cpu_kv_caches[layer_idx][cpu_phys],
                        non_blocking=True,
                    )

                # Update mappings
                self.block_tables[sid][logical_idx] = gpu_phys
                del self.cpu_block_map[sid][logical_idx]
                self.cpu_free_blocks.append(cpu_phys)

        # Wait for all copies to finish before kernel runs
        torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)

        # Invalidate cached tensors
        self._block_table_tensor = None

        dt = time.perf_counter() - t0
        self._fetch_count += len(fetch_needed)
        self._fetch_time += dt

    def evict_cold_blocks(self, seq_ids: Optional[List[int]] = None):
        """
        Evict cold (old) blocks from GPU to CPU for given sequences.

        Uses window-based policy: keep the last `gpu_window` blocks on GPU
        per sequence, move older blocks to CPU pinned memory.

        Called AFTER write_kv / advance_seq_len to free GPU memory.
        """
        if seq_ids is None:
            seq_ids = list(self.block_tables.keys())

        t0 = time.perf_counter()
        evicted = 0

        for sid in seq_ids:
            blocks = self.block_tables[sid]
            cur_len = self.seq_lens.get(sid, 0)
            used_blocks = (cur_len + self.block_size - 1) // self.block_size

            # How many blocks should stay on GPU (recent tokens)
            keep_on_gpu = min(self.gpu_window, used_blocks)
            evict_up_to = used_blocks - keep_on_gpu  # logical index boundary

            for logical_idx in range(evict_up_to):
                gpu_phys = blocks[logical_idx]

                # Skip if already on CPU
                if gpu_phys < 0:
                    continue

                # Need a CPU slot
                if not self.cpu_free_blocks:
                    break  # CPU full, stop evicting

                cpu_phys = self.cpu_free_blocks.pop()

                # Copy GPU → CPU for all layers (async)
                with torch.cuda.stream(self._copy_stream):
                    for layer_idx in range(self.num_kv_layers):
                        self.cpu_kv_caches[layer_idx][cpu_phys].copy_(
                            self.kv_caches[layer_idx][gpu_phys],
                            non_blocking=True,
                        )

                # Update mappings
                self.cpu_block_map[sid][logical_idx] = cpu_phys
                blocks[logical_idx] = -1  # placeholder
                self.free_blocks.append(gpu_phys)
                evicted += 1

        if evicted > 0:
            # Wait for GPU→CPU copies to complete
            self._copy_stream.synchronize()
            self._block_table_tensor = None

            dt = time.perf_counter() - t0
            self._evict_count += evicted
            self._evict_time += dt

    def _evict_one_cold_block(self, exclude_seqs: Set[int] = None) -> bool:
        """
        Evict one cold GPU block from any sequence NOT in exclude_seqs.
        Returns True if a block was evicted, False if nothing to evict.
        """
        if exclude_seqs is None:
            exclude_seqs = set()

        for sid, blocks in self.block_tables.items():
            if sid in exclude_seqs:
                continue
            for logical_idx, gpu_phys in enumerate(blocks):
                if gpu_phys >= 0 and not self.cpu_free_blocks:
                    continue
                if gpu_phys >= 0:
                    # Evict this block
                    cpu_phys = self.cpu_free_blocks.pop()
                    for layer_idx in range(self.num_kv_layers):
                        self.cpu_kv_caches[layer_idx][cpu_phys].copy_(
                            self.kv_caches[layer_idx][gpu_phys]
                        )
                    if sid not in self.cpu_block_map:
                        self.cpu_block_map[sid] = {}
                    self.cpu_block_map[sid][logical_idx] = cpu_phys
                    blocks[logical_idx] = -1
                    self.free_blocks.append(gpu_phys)
                    return True
        return False

    def _evict_one_cold_block_from_active(self, seq_ids: List[int]) -> bool:
        """
        Evict the OLDEST GPU block from active sequences.

        First tries respecting gpu_window. If that fails (window covers all
        blocks), falls back to forced eviction of the absolute oldest block.
        This is critical for prefill of long prompts that exceed the window.
        """
        # Pass 1: respect gpu_window
        for sid in seq_ids:
            blocks = self.block_tables[sid]
            cur_len = self.seq_lens.get(sid, 0)
            used_blocks = (cur_len + self.block_size - 1) // self.block_size
            keep_on_gpu = min(self.gpu_window, used_blocks)
            evict_up_to = used_blocks - keep_on_gpu

            for logical_idx in range(evict_up_to):
                gpu_phys = blocks[logical_idx]
                if gpu_phys >= 0 and self.cpu_free_blocks:
                    cpu_phys = self.cpu_free_blocks.pop()
                    for layer_idx in range(self.num_kv_layers):
                        self.cpu_kv_caches[layer_idx][cpu_phys].copy_(
                            self.kv_caches[layer_idx][gpu_phys]
                        )
                    if sid not in self.cpu_block_map:
                        self.cpu_block_map[sid] = {}
                    self.cpu_block_map[sid][logical_idx] = cpu_phys
                    blocks[logical_idx] = -1
                    self.free_blocks.append(gpu_phys)
                    return True

        # Pass 2: forced eviction — ignore gpu_window entirely
        # This is needed when prefill prompt is larger than gpu_window
        for sid in seq_ids:
            blocks = self.block_tables[sid]
            for logical_idx, gpu_phys in enumerate(blocks):
                if gpu_phys >= 0 and self.cpu_free_blocks:
                    cpu_phys = self.cpu_free_blocks.pop()
                    for layer_idx in range(self.num_kv_layers):
                        self.cpu_kv_caches[layer_idx][cpu_phys].copy_(
                            self.kv_caches[layer_idx][gpu_phys]
                        )
                    if sid not in self.cpu_block_map:
                        self.cpu_block_map[sid] = {}
                    self.cpu_block_map[sid][logical_idx] = cpu_phys
                    blocks[logical_idx] = -1
                    self.free_blocks.append(gpu_phys)
                    return True
        return False

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if enough blocks available (GPU + CPU combined)."""
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        return num_needed <= (self.num_free_blocks + self.num_cpu_free_blocks)

    def memory_usage_mb(self) -> float:
        """Total memory usage (GPU + CPU)."""
        gpu = sum(c.nelement() * c.element_size() for c in self.kv_caches)
        cpu = sum(c.nelement() * c.element_size() for c in self.cpu_kv_caches)
        return (gpu + cpu) / (1024 * 1024)

    def gpu_memory_usage_mb(self) -> float:
        """GPU-only memory usage."""
        total = sum(c.nelement() * c.element_size() for c in self.kv_caches)
        return total / (1024 * 1024)

    def cpu_memory_usage_mb(self) -> float:
        """CPU-only memory usage."""
        total = sum(c.nelement() * c.element_size() for c in self.cpu_kv_caches)
        return total / (1024 * 1024)

    def print_stats(self):
        """Print offload transfer statistics."""
        if self._fetch_count > 0 or self._evict_count > 0:
            print(f"🔄 KV Cache Offload Stats:")
            if self._evict_count > 0:
                avg_evict = self._evict_time / self._evict_count * 1000
                print(f"   Evicted: {self._evict_count} blocks, "
                      f"{self._evict_time:.2f}s total, {avg_evict:.1f}ms avg")
            if self._fetch_count > 0:
                avg_fetch = self._fetch_time / self._fetch_count * 1000
                print(f"   Fetched: {self._fetch_count} blocks, "
                      f"{self._fetch_time:.2f}s total, {avg_fetch:.1f}ms avg")

            gpu_used = self.num_gpu_blocks - self.num_free_blocks
            cpu_used = self.num_cpu_blocks - self.num_cpu_free_blocks
            print(f"   GPU: {gpu_used}/{self.num_gpu_blocks} blocks | "
                  f"CPU: {cpu_used}/{self.num_cpu_blocks} blocks")

    def __repr__(self) -> str:
        gpu_used = self.num_gpu_blocks - self.num_free_blocks
        cpu_used = self.num_cpu_blocks - self.num_cpu_free_blocks
        return (
            f"TieredBlockManager("
            f"layers={self.num_layers}, kv_layers={self.num_kv_layers}, "
            f"gpu={gpu_used}/{self.num_gpu_blocks}, "
            f"cpu={cpu_used}/{self.num_cpu_blocks}, "
            f"block_size={self.block_size}, "
            f"window={self.gpu_window}, "
            f"gpu_mem={self.gpu_memory_usage_mb():.0f}MB, "
            f"cpu_mem={self.cpu_memory_usage_mb():.0f}MB)"
        )
