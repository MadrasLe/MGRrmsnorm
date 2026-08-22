"""
🧪 MegaGemm Inference Engine Test
==================================
Tests the complete inference pipeline:
1. Import validation
2. KV Cache manager (CPU - no GPU needed)
3. Paged Attention kernel (GPU required)
4. Full model inference (GPU + HF model download)

Usage:
    python test_inference.py              # Run CPU tests only
    python test_inference.py --gpu        # Run GPU tests too
    python test_inference.py --full       # Full test with model download

Author: Gabriel Yogi
"""

import sys
import os
import argparse
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_imports():
    """Test 1: Validate all imports work."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)

    errors = []

    # Core modules
    try:
        from megagemm.engine.kv_cache import BlockManager
        print("  ✅ kv_cache.BlockManager")
    except Exception as e:
        errors.append(f"kv_cache: {e}")
        print(f"  ❌ kv_cache: {e}")

    try:
        from megagemm.kernels.paged_attention import paged_attention_decode, prefill_attention
        print("  ✅ paged_attention")
    except Exception as e:
        errors.append(f"paged_attention: {e}")
        print(f"  ❌ paged_attention: {e}")

    try:
        from megagemm.engine.sampling import sample_logits
        print("  ✅ sampling")
    except Exception as e:
        errors.append(f"sampling: {e}")
        print(f"  ❌ sampling: {e}")

    try:
        from megagemm.models.llama import LlamaConfig, MegaGemmLlama
        print("  ✅ models.llama")
    except Exception as e:
        errors.append(f"models.llama: {e}")
        print(f"  ❌ models.llama: {e}")

    try:
        from megagemm.models.loader import load_from_hf
        print("  ✅ models.loader")
    except Exception as e:
        errors.append(f"models.loader: {e}")
        print(f"  ❌ models.loader: {e}")

    try:
        from megagemm.engine import InferenceEngine
        print("  ✅ engine.InferenceEngine")
    except Exception as e:
        errors.append(f"engine: {e}")
        print(f"  ❌ engine: {e}")

    if errors:
        print(f"\n  ⚠️  {len(errors)} import(s) failed")
    else:
        print("\n  🎉 All imports OK!")

    return len(errors) == 0


def _run_kv_cache():
    """Test 2: KV Cache manager (CPU only)."""
    import torch
    from megagemm.engine.kv_cache import BlockManager

    print("\n" + "=" * 60)
    print("TEST 2: KV Cache Manager (CPU)")
    print("=" * 60)

    bm = BlockManager(
        num_layers=4,
        num_blocks=64,
        block_size=16,
        num_kv_heads=4,
        head_dim=32,
        dtype=torch.float32,
        device='cpu',
    )
    print(f"  Created: {bm}")

    # Allocate
    bm.allocate_sequence(seq_id=0, num_tokens=48)
    print(f"  Allocated seq 0 (48 tokens): {len(bm.block_tables[0])} blocks")
    assert len(bm.block_tables[0]) == 3  # 48/16 = 3 blocks

    # Write KV for layer 0
    k = torch.randn(48, 4, 32)  # [tokens, kv_heads, head_dim]
    v = torch.randn(48, 4, 32)
    bm.write_kv(seq_id=0, layer_idx=0, k=k, v=v)
    print(f"  Wrote KV for layer 0: {k.shape}")

    # Write KV for layer 1
    k2 = torch.randn(48, 4, 32)
    v2 = torch.randn(48, 4, 32)
    bm.write_kv(seq_id=0, layer_idx=1, k=k2, v=v2)
    print(f"  Wrote KV for layer 1: {k2.shape}")

    # Advance
    bm.advance_seq_len(seq_id=0, num_tokens=48)
    assert bm.seq_lens[0] == 48
    print(f"  Seq len after advance: {bm.seq_lens[0]}")

    # Verify data integrity (layer 0 != layer 1)
    cache0 = bm.get_kv_cache(0)
    cache1 = bm.get_kv_cache(1)
    phys_block = bm.block_tables[0][0]
    assert not torch.allclose(cache0[phys_block], cache1[phys_block])
    print("  ✅ Per-layer caches are independent")

    # Block table tensor
    table = bm.get_block_table_tensor([0])
    print(f"  Block table: {table}")

    # Free
    bm.free_sequence(seq_id=0)
    assert bm.num_free_blocks == 64
    print(f"  Freed: {bm}")

    print("\n  🎉 KV Cache tests passed!")
    return True


def _run_sampling():
    """Test 3: Sampling utilities."""
    import torch
    from megagemm.engine.sampling import sample_logits

    print("\n" + "=" * 60)
    print("TEST 3: Sampling (CPU)")
    print("=" * 60)

    logits = torch.randn(2, 1000)  # [batch=2, vocab=1000]

    # Greedy
    tokens = sample_logits(logits, temperature=0.0)
    assert tokens.shape == (2,)
    assert torch.all(tokens == logits.argmax(dim=-1))
    print(f"  ✅ Greedy: {tokens.tolist()}")

    # Temperature
    tokens = sample_logits(logits, temperature=0.5, top_k=10)
    assert tokens.shape == (2,)
    print(f"  ✅ Top-k=10, temp=0.5: {tokens.tolist()}")

    # Top-p
    tokens = sample_logits(logits, temperature=0.8, top_p=0.9)
    print(f"  ✅ Top-p=0.9, temp=0.8: {tokens.tolist()}")

    print("\n  🎉 Sampling tests passed!")
    return True


def _run_model_creation():
    """Test 4: Create LLaMA model (CPU, no weights)."""
    import torch
    from megagemm.models.llama import LlamaConfig, MegaGemmLlama

    print("\n" + "=" * 60)
    print("TEST 4: Model Creation (CPU)")
    print("=" * 60)

    # TinyLlama-like config
    config = LlamaConfig(
        hidden_size=64,       # tiny for testing
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA 2x
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
    )

    model = MegaGemmLlama(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Created model: {num_params:,} params")
    print(f"  Config: {config.num_hidden_layers}L, {config.hidden_size}H, "
          f"GQA={config.num_attention_heads // config.num_key_value_heads}x")

    # Test prefill on CPU (uses PyTorch fallbacks)
    from megagemm.engine.kv_cache import BlockManager
    bm = BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=32,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device='cpu',
    )

    input_ids = torch.randint(0, 256, (1, 8))
    positions = torch.arange(8).unsqueeze(0)

    bm.allocate_sequence(seq_id=0, num_tokens=8)

    with torch.no_grad():
        logits = model.prefill(input_ids, positions, bm, seq_id=0)

    print(f"  Prefill output: logits shape = {logits.shape}")
    assert logits.shape == (1, 8, 256)
    assert bm.seq_lens[0] == 8
    print(f"  ✅ Prefill OK! seq_len={bm.seq_lens[0]}")

    print("\n  🎉 Model creation tests passed!")
    return True


def _run_paged_attention_gpu():
    """Test 5: Paged attention Triton kernel (GPU required)."""
    import torch
    if not torch.cuda.is_available():
        print("\n⏭️  Skipping GPU paged attention test (no CUDA)")
        return True

    from megagemm.kernels.paged_attention import paged_attention_decode, prefill_attention

    print("\n" + "=" * 60)
    print("TEST 5: Paged Attention Triton Kernel (GPU)")
    print("=" * 60)

    # Setup
    num_seqs = 2
    num_q_heads = 8
    num_kv_heads = 4  # GQA 2x
    head_dim = 64
    block_size = 16
    num_blocks = 32
    seq_lens_list = [32, 48]  # different lengths

    # Create KV cache pool
    kv_cache = torch.randn(
        num_blocks, 2, num_kv_heads, block_size, head_dim,
        device='cuda', dtype=torch.float16
    )

    # Block tables (simulated)
    max_blocks = max(
        (sl + block_size - 1) // block_size for sl in seq_lens_list
    )
    block_tables = torch.zeros(
        num_seqs, max_blocks, dtype=torch.int32, device='cuda'
    )
    block_tables[0, :2] = torch.tensor([0, 1])   # seq0: blocks 0,1
    block_tables[1, :3] = torch.tensor([2, 3, 4]) # seq1: blocks 2,3,4

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cuda')

    # Query: 1 token per sequence
    query = torch.randn(
        num_seqs, num_q_heads, head_dim,
        device='cuda', dtype=torch.float16
    )

    # Run kernel
    output = paged_attention_decode(query, kv_cache, block_tables, seq_lens)

    print(f"  Input: query={query.shape}, kv_cache={kv_cache.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (num_seqs, num_q_heads, head_dim)
    assert not torch.isnan(output).any(), "NaN in output!"
    assert not torch.isinf(output).any(), "Inf in output!"
    print(f"  ✅ No NaN/Inf, shape correct")

    # Test prefill attention
    batch = 1
    seq_len = 32
    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    out = prefill_attention(q, k, v, is_causal=True)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    print(f"  ✅ Prefill attention OK: {out.shape}")

    print("\n  🎉 Paged Attention GPU tests passed!")
    return True


def _run_full_inference(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Test 6: Full inference with real model (GPU + download)."""
    import torch
    if not torch.cuda.is_available():
        print("\n⏭️  Skipping full inference test (no CUDA)")
        return True

    from megagemm.engine import InferenceEngine

    print("\n" + "=" * 60)
    print(f"TEST 6: Full Inference ({model_name})")
    print("=" * 60)

    engine = InferenceEngine(model_name, dtype=torch.float16)

    prompt = "The capital of France is"
    print(f"\n  Prompt: '{prompt}'")

    output = engine.generate(
        prompt,
        max_new_tokens=30,
        temperature=0.0,  # greedy for deterministic test
        verbose=True,
    )

    print(f"  Output: '{output}'")
    assert len(output) > 0, "Empty output!"
    print(f"\n  🎉 Full inference test passed!")
    return True


def test_imports():
    assert _run_imports()


def test_kv_cache():
    assert _run_kv_cache()


def test_sampling():
    assert _run_sampling()


def test_model_creation():
    assert _run_model_creation()


def test_paged_attention_gpu():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for paged-attention validation")
    assert _run_paged_attention_gpu()


def test_full_inference():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for full-model inference validation")
    assert _run_full_inference()


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Inference Tests")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--full", action="store_true", help="Full test with model download")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model for full test")
    args = parser.parse_args()

    print("🔥 MegaGemm Inference Engine Tests")
    print("=" * 60)

    results = {}

    # CPU tests (always run)
    results['imports'] = _run_imports()

    if results['imports']:
        results['kv_cache'] = _run_kv_cache()
        results['sampling'] = _run_sampling()
        results['model_creation'] = _run_model_creation()

    # GPU tests
    if args.gpu or args.full:
        results['paged_attn'] = _run_paged_attention_gpu()

    if args.full:
        results['full_inference'] = _run_full_inference(args.model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n{'🎉 All tests passed!' if all_passed else '⚠️ Some tests failed'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
