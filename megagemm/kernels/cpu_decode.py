"""
⚡ CPU Decode Loop — Python wrapper for cpu_decode.c
=====================================================
Packs model weights into C-compatible format and calls
the full decode step in C with zero Python overhead.

Usage:
    from megagemm.kernels.cpu_decode import CPUDecoder

    decoder = CPUDecoder(model, config)
    tokens = decoder.generate(first_token, position, kv_cache,
                              block_table, max_tokens=32)

Author: Gabriel Yogi
"""

import ctypes
import subprocess
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from megagemm.kernels.cpu_int8 import quantize_to_int8


# ─────────────────────────────────────────────
# C structs mirrored in Python
# ─────────────────────────────────────────────

class MegaGemmConfig(ctypes.Structure):
    _fields_ = [
        ("hidden_size", ctypes.c_int),
        ("intermediate_size", ctypes.c_int),
        ("num_layers", ctypes.c_int),
        ("num_q_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("rms_norm_eps", ctypes.c_float),
        ("qkv_bias", ctypes.c_int),
        ("norm_offset", ctypes.c_int),
        ("act_gelu", ctypes.c_int),
        ("rope_theta", ctypes.c_float),
        ("kv_block_size", ctypes.c_int),
        ("quant_mode", ctypes.c_int),  # 0=INT8, 1=INT4
    ]

class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("input_norm_w", ctypes.c_void_p),
        ("post_attn_norm_w", ctypes.c_void_p),
        # INT8
        ("qkv_w", ctypes.c_void_p),
        ("qkv_s", ctypes.c_void_p),
        ("qkv_bias", ctypes.c_void_p),
        ("o_w", ctypes.c_void_p),
        ("o_s", ctypes.c_void_p),
        ("gate_up_w", ctypes.c_void_p),
        ("gate_up_s", ctypes.c_void_p),
        ("down_w", ctypes.c_void_p),
        ("down_s", ctypes.c_void_p),
        # INT4
        ("qkv_w4", ctypes.c_void_p),
        ("qkv_s4", ctypes.c_void_p),
        ("o_w4", ctypes.c_void_p),
        ("o_s4", ctypes.c_void_p),
        ("gate_up_w4", ctypes.c_void_p),
        ("gate_up_s4", ctypes.c_void_p),
        ("down_w4", ctypes.c_void_p),
        ("down_s4", ctypes.c_void_p),
    ]

class ModelWeights(ctypes.Structure):
    _fields_ = [
        ("embed_tokens", ctypes.c_void_p),
        ("layers", ctypes.POINTER(LayerWeights)),
        ("final_norm_w", ctypes.c_void_p),
        ("lm_head_w", ctypes.c_void_p),
        ("lm_head_s", ctypes.c_void_p),
        ("lm_head_w4", ctypes.c_void_p),
        ("lm_head_s4", ctypes.c_void_p),
        ("cos_cache", ctypes.c_void_p),
        ("sin_cache", ctypes.c_void_p),
    ]


# ─────────────────────────────────────────────
# Compile and load
# ─────────────────────────────────────────────

_lib = None
_LIB_LOADED = False

def _get_lib():
    global _lib, _LIB_LOADED
    if _LIB_LOADED:
        return _lib

    src = Path(__file__).parent / "cpu_decode.c"
    if not src.exists():
        _LIB_LOADED = True
        return None

    if sys.platform == 'win32':
        lib_name = "cpu_decode.dll"
    elif sys.platform == 'darwin':
        lib_name = "libcpu_decode.dylib"
    else:
        lib_name = "libcpu_decode.so"

    lib_path = src.parent / lib_name

    need_compile = not lib_path.exists()
    if lib_path.exists() and src.stat().st_mtime > lib_path.stat().st_mtime:
        need_compile = True

    if need_compile:
        print(f"🔨 Compiling {src.name} → {lib_name}...")
        compiled = False

        # AVX-512 only if explicitly requested (causes clock throttle on many Xeons)
        use_avx512 = os.environ.get("MEGAGEMM_AVX512", "0") == "1"

        configs = []
        if use_avx512:
            configs.append((["-mavx512f", "-mavx2", "-mfma"], "AVX-512F + OpenMP"))
        configs.append((["-mavx2", "-mfma"], "AVX2 + OpenMP"))
        configs.append(([], "scalar + OpenMP"))  # ultimate fallback

        for flags, label in configs:
            try:
                cmd = ["gcc", "-O3"] + flags + ["-fopenmp",
                       "-shared", "-fPIC", "-lm",
                       "-o", str(lib_path), str(src)]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✅ Compiled {label}")
                compiled = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        if not compiled:
            print(f"  ❌ Compilation failed")
            _LIB_LOADED = True
            return None

    try:
        _lib = ctypes.CDLL(str(lib_path))

        _lib.megagemm_decode_step.argtypes = [
            ctypes.POINTER(MegaGemmConfig),
            ctypes.POINTER(ModelWeights),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
        ]
        _lib.megagemm_decode_step.restype = ctypes.c_int

        _lib.megagemm_decode_multi.argtypes = [
            ctypes.POINTER(MegaGemmConfig),
            ctypes.POINTER(ModelWeights),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]
        _lib.megagemm_decode_multi.restype = ctypes.c_int

        _lib.megagemm_alloc_scratch.argtypes = [
            ctypes.POINTER(MegaGemmConfig), ctypes.c_int,
        ]
        _lib.megagemm_alloc_scratch.restype = None

        _lib.megagemm_free_scratch.argtypes = []
        _lib.megagemm_free_scratch.restype = None

        _lib.megagemm_decode_batch.argtypes = [
            ctypes.POINTER(MegaGemmConfig),
            ctypes.POINTER(ModelWeights),
            ctypes.c_int,                    # batch_size
            ctypes.c_void_p,                 # token_ids
            ctypes.c_void_p,                 # positions
            ctypes.c_void_p,                 # kv_caches
            ctypes.c_void_p,                 # block_tables (flattened)
            ctypes.c_int,                    # max_blocks_per_seq
            ctypes.c_void_p,                 # seq_lens
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # kv strides
            ctypes.c_void_p,                 # output_tokens
        ]
        _lib.megagemm_decode_batch.restype = ctypes.c_int

        print(f"  ✅ Loaded {lib_name}")
        _LIB_LOADED = True
        return _lib
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        _LIB_LOADED = True
        return None


# ─────────────────────────────────────────────
# CPUDecoder: packs weights + calls C decode
# ─────────────────────────────────────────────

class CPUDecoder:
    QUANT_INT8 = 0
    QUANT_INT4 = 1
    W4_GROUP_SIZE = 128

    def __init__(self, model, config, quant='int8'):
        """
        Args:
            model: MegaGemmLlama instance
            config: LlamaConfig instance
            quant: 'int8' or 'int4'
        """
        self.lib = _get_lib()
        if self.lib is None:
            raise RuntimeError("Failed to load cpu_decode library")

        self.config = config
        self.quant = quant
        self._kept = []  # prevent GC

        qm = self.QUANT_INT4 if quant == 'int4' else self.QUANT_INT8

        # Build C config
        self.c_cfg = MegaGemmConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=1 if config.attention_bias else 0,
            norm_offset=1 if getattr(config, 'norm_offset', False) else 0,
            act_gelu=1 if getattr(config, 'hidden_act', 'silu') == 'gelu' else 0,
            rope_theta=config.rope_theta,
            kv_block_size=16,
            quant_mode=qm,
        )

        # Pack weights
        label = 'INT4' if quant == 'int4' else 'INT8'
        print(f"  📦 Packing {config.num_hidden_layers} layers to {label}...")
        self.c_model = self._pack_weights(model)

        # Pre-allocate scratch buffers
        max_seq = 4096
        self.lib.megagemm_alloc_scratch(ctypes.byref(self.c_cfg), max_seq)
        print(f"  ✅ Ready ({label}, scratch buffers pre-allocated)")

    def _keep(self, tensor):
        """Keep tensor alive to prevent GC."""
        self._kept.append(tensor)
        return tensor.data_ptr()

    def _quant(self, weight):
        """Quantize weight to INT8 + scales, return (ptr_w, ptr_s)."""
        w_int8, scales = quantize_to_int8(weight.data.float())
        w_int8 = w_int8.contiguous()
        scales = scales.contiguous()
        self._kept.extend([w_int8, scales])
        return w_int8.data_ptr(), scales.data_ptr()

    def _quant4(self, weight):
        """Quantize weight to INT4 packed + group scales.
        Returns (ptr_packed, ptr_group_scales)."""
        w = weight.data.float()
        M, K = w.shape
        gs = self.W4_GROUP_SIZE
        num_groups = (K + gs - 1) // gs

        # Per-group quantization
        group_scales = torch.zeros(M, num_groups, dtype=torch.float32)
        w_int4 = torch.zeros(M, K, dtype=torch.int8)

        for g in range(num_groups):
            start = g * gs
            end = min(start + gs, K)
            group = w[:, start:end]
            amax = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = amax / 7.0
            group_scales[:, g] = scale.squeeze(1)
            quantized = (group / scale).round().clamp(-8, 7).to(torch.int8)
            w_int4[:, start:end] = quantized

        # Pack 2 INT4 values per byte: low=even, high=odd
        # Store as unsigned: val + 8 → [0, 15]
        w_uint4 = (w_int4 + 8).to(torch.uint8)
        assert K % 2 == 0, f"K={K} must be even for INT4 packing"
        packed = torch.zeros(M, K // 2, dtype=torch.uint8)
        packed = (w_uint4[:, 0::2] & 0x0F) | ((w_uint4[:, 1::2] & 0x0F) << 4)

        packed = packed.contiguous()
        group_scales = group_scales.contiguous()
        self._kept.extend([packed, group_scales])
        return packed.data_ptr(), group_scales.data_ptr()

    def _pack_weights(self, model):
        """Pack all model weights into C structs."""
        cfg = self.config
        n_layers = cfg.num_hidden_layers
        use_w4 = self.quant == 'int4'

        # Embedding (keep as FP32)
        embed = model.embed_tokens.weight.data.float().contiguous()

        # RoPE
        cos = model.cos_cache.float().contiguous()
        sin = model.sin_cache.float().contiguous()

        # Final norm
        final_norm = model.norm.weight.data.float().contiguous()

        # LM head (always INT8 for accuracy on large vocab)
        lm_w, lm_s = self._quant(model.lm_head.weight)
        lm_w4, lm_s4 = (0, 0)
        if use_w4:
            lm_w4, lm_s4 = self._quant4(model.lm_head.weight)

        # Pack per-layer weights
        c_layers = (LayerWeights * n_layers)()

        for i in range(n_layers):
            layer = model.layers[i]
            attn = layer.self_attn
            mlp = layer.mlp

            # Norms
            in_norm = layer.input_layernorm.weight.data.float().contiguous()
            post_norm = layer.post_attention_layernorm.weight.data.float().contiguous()
            self._kept.extend([in_norm, post_norm])

            # INT8 weights (always pack as fallback)
            qkv_w, qkv_s = self._quant(attn.qkv_proj.weight)
            qkv_bias_ptr = 0
            if attn.qkv_proj.bias is not None:
                qkv_b = attn.qkv_proj.bias.data.float().contiguous()
                self._kept.append(qkv_b)
                qkv_bias_ptr = qkv_b.data_ptr()
            o_w, o_s = self._quant(attn.o_proj.weight)
            gu_w, gu_s = self._quant(mlp.gate_up_proj.weight)
            d_w, d_s = self._quant(mlp.down_proj.weight)

            # INT4 weights (if requested)
            qkv_w4, qkv_s4 = (0, 0)
            o_w4, o_s4 = (0, 0)
            gu_w4, gu_s4 = (0, 0)
            d_w4, d_s4 = (0, 0)
            if use_w4:
                qkv_w4, qkv_s4 = self._quant4(attn.qkv_proj.weight)
                o_w4, o_s4 = self._quant4(attn.o_proj.weight)
                gu_w4, gu_s4 = self._quant4(mlp.gate_up_proj.weight)
                d_w4, d_s4 = self._quant4(mlp.down_proj.weight)

            c_layers[i] = LayerWeights(
                input_norm_w=in_norm.data_ptr(),
                post_attn_norm_w=post_norm.data_ptr(),
                qkv_w=qkv_w, qkv_s=qkv_s, qkv_bias=qkv_bias_ptr,
                o_w=o_w, o_s=o_s,
                gate_up_w=gu_w, gate_up_s=gu_s,
                down_w=d_w, down_s=d_s,
                qkv_w4=qkv_w4, qkv_s4=qkv_s4,
                o_w4=o_w4, o_s4=o_s4,
                gate_up_w4=gu_w4, gate_up_s4=gu_s4,
                down_w4=d_w4, down_s4=d_s4,
            )

        self._kept.extend([embed, cos, sin, final_norm, c_layers])

        c_model = ModelWeights(
            embed_tokens=embed.data_ptr(),
            layers=c_layers,
            final_norm_w=final_norm.data_ptr(),
            lm_head_w=lm_w, lm_head_s=lm_s,
            lm_head_w4=lm_w4, lm_head_s4=lm_s4,
            cos_cache=cos.data_ptr(),
            sin_cache=sin.data_ptr(),
        )

        return c_model

    def _make_kv_ptrs(self, block_manager):
        """Build ctypes array of per-layer KV cache data pointers."""
        n = self.config.num_hidden_layers
        arr = (ctypes.c_void_p * n)()
        for i in range(n):
            kv = block_manager.get_kv_cache(i)
            arr[i] = kv.data_ptr()
        self._kv_strides = block_manager.get_kv_cache(0).stride()
        return arr

    def decode_step(self, token_id, position, block_manager, block_table, seq_len):
        """
        Single decode step in C.

        Args:
            token_id: input token ID
            position: position in sequence
            block_manager: BlockManager instance
            block_table: [max_blocks] int tensor of physical block IDs
            seq_len: current KV length

        Returns:
            (next_token_id, logits)
        """
        logits = torch.empty(self.config.vocab_size, dtype=torch.float32)

        kv_ptrs = self._make_kv_ptrs(block_manager)
        strides = self._kv_strides
        bt = block_table.contiguous().int()

        next_id = self.lib.megagemm_decode_step(
            ctypes.byref(self.c_cfg),
            ctypes.byref(self.c_model),
            token_id, position,
            ctypes.cast(kv_ptrs, ctypes.c_void_p), bt.data_ptr(), seq_len,
            strides[0], strides[1], strides[2], strides[3],
            logits.data_ptr(), self.config.vocab_size,
        )

        return next_id, logits

    def generate(self, first_token, start_pos, block_manager, block_table,
                 max_tokens=32, eos_token=151643):
        """
        Generate N tokens entirely in C.

        Returns:
            list of generated token IDs
        """
        output_tokens = torch.zeros(max_tokens, dtype=torch.int32)
        logits = torch.empty(self.config.vocab_size, dtype=torch.float32)

        kv_ptrs = self._make_kv_ptrs(block_manager)
        strides = self._kv_strides
        bt = block_table.contiguous().int()

        n = self.lib.megagemm_decode_multi(
            ctypes.byref(self.c_cfg),
            ctypes.byref(self.c_model),
            first_token, start_pos, max_tokens, eos_token,
            ctypes.cast(kv_ptrs, ctypes.c_void_p), bt.data_ptr(),
            strides[0], strides[1], strides[2], strides[3],
            output_tokens.data_ptr(), logits.data_ptr(), self.config.vocab_size,
        )

        return output_tokens[:n].tolist()

    def batch_decode_step(self, token_ids, positions, block_manager,
                          block_tables, seq_lens):
        """
        Batch decode step: process N sequences simultaneously in C.

        Args:
            token_ids: list of N token IDs
            positions: list of N positions
            block_manager: BlockManager instance
            block_tables: list of N block_table tensors
            seq_lens: list of N sequence lengths

        Returns:
            list of N next token IDs
        """
        import numpy as np
        N = len(token_ids)

        # Convert inputs to contiguous int32 arrays
        tok_arr = torch.tensor(token_ids, dtype=torch.int32).contiguous()
        pos_arr = torch.tensor(positions, dtype=torch.int32).contiguous()
        slen_arr = torch.tensor(seq_lens, dtype=torch.int32).contiguous()
        out_arr = torch.zeros(N, dtype=torch.int32)

        # Flatten block tables: [N, max_blocks] → contiguous
        max_blocks = max(len(bt) for bt in block_tables)
        bt_flat = torch.zeros(N * max_blocks, dtype=torch.int32)
        for i, bt in enumerate(block_tables):
            bt_flat[i * max_blocks: i * max_blocks + len(bt)] = bt.int()

        # KV cache pointers
        kv_ptrs = self._make_kv_ptrs(block_manager)
        strides = self._kv_strides

        self.lib.megagemm_decode_batch(
            ctypes.byref(self.c_cfg),
            ctypes.byref(self.c_model),
            N,
            tok_arr.data_ptr(),
            pos_arr.data_ptr(),
            ctypes.cast(kv_ptrs, ctypes.c_void_p),
            bt_flat.data_ptr(),
            max_blocks,
            slen_arr.data_ptr(),
            strides[0], strides[1], strides[2], strides[3],
            out_arr.data_ptr(),
        )

        return out_arr.tolist()
