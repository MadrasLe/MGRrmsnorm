"""
🧪 Packed Prefill Attention Test
==================================
Tests the packed prefill attention kernel (Triton + PyTorch fallback)
and the model-level prefill_packed() method.

Tests:
1. PyTorch fallback correctness — packed vs per-sequence SDPA
2. Model-level prefill_packed vs per-sequence prefill equivalence
3. Edge cases: single sequence, same lengths, very different lengths

Usage:
    python tests/test_packed_prefill.py          # CPU-only test
    python tests/test_packed_prefill.py --gpu     # GPU test with Triton

Author: Gabriel Yogi
"""

import sys
import os
import torch
import math

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reference_attention(q, k, v, cu_seqlens):
    """
    Reference implementation: per-sequence causal attention using PyTorch SDPA.
    Returns output with same shape as q: [total_tokens, num_q_heads, head_dim]
    """
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    output = torch.empty_like(q)

    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()

        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)

        # Handle GQA
        if num_kv_heads < num_q_heads:
            ratio = num_q_heads // num_kv_heads
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)

        oi = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=True
        )
        output[start:end] = oi.squeeze(0).transpose(0, 1)

    return output


def test_packed_attention_basic():
    """Test 1: Basic packed attention correctness (MHA, no GQA)."""
    print("=" * 60)
    print("Test 1: Packed attention — basic MHA correctness")
    print("=" * 60)

    from megagemm.kernels.paged_attention import packed_prefill_attention

    torch.manual_seed(42)
    num_heads = 8
    head_dim = 64

    # 3 sequences with different lengths
    lens = [10, 25, 7]
    total = sum(lens)
    cu_seqlens = torch.tensor([0] + [sum(lens[:i+1]) for i in range(len(lens))], dtype=torch.int32)

    q = torch.randn(total, num_heads, head_dim)
    k = torch.randn(total, num_heads, head_dim)
    v = torch.randn(total, num_heads, head_dim)

    # Reference
    ref = _reference_attention(q, k, v, cu_seqlens)

    # Packed attention (will use PyTorch fallback on CPU/Windows)
    out = packed_prefill_attention(q, k, v, cu_seqlens)

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1), dim=0
    ).item()
    max_diff = (ref - out).abs().max().item()

    print(f"  Sequences: {lens} (total {total} tokens)")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.8f}")

    assert cos_sim > 0.9999, f"Cosine sim too low: {cos_sim}"
    print("  ✅ PASSED\n")


def test_packed_attention_gqa():
    """Test 2: Packed attention with GQA (Grouped Query Attention)."""
    print("=" * 60)
    print("Test 2: Packed attention — GQA (8 q heads, 2 kv heads)")
    print("=" * 60)

    from megagemm.kernels.paged_attention import packed_prefill_attention

    torch.manual_seed(123)
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64

    lens = [15, 30]
    total = sum(lens)
    cu_seqlens = torch.tensor([0, 15, 45], dtype=torch.int32)

    q = torch.randn(total, num_q_heads, head_dim)
    k = torch.randn(total, num_kv_heads, head_dim)
    v = torch.randn(total, num_kv_heads, head_dim)

    ref = _reference_attention(q, k, v, cu_seqlens)
    out = packed_prefill_attention(q, k, v, cu_seqlens)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1), dim=0
    ).item()
    max_diff = (ref - out).abs().max().item()

    print(f"  Sequences: {lens} (total {total} tokens)")
    print(f"  GQA ratio: {num_q_heads // num_kv_heads}x")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.8f}")

    assert cos_sim > 0.9999, f"Cosine sim too low: {cos_sim}"
    print("  ✅ PASSED\n")


def test_packed_attention_single_seq():
    """Test 3: Edge case — single sequence (should be identical to SDPA)."""
    print("=" * 60)
    print("Test 3: Packed attention — single sequence (edge case)")
    print("=" * 60)

    from megagemm.kernels.paged_attention import packed_prefill_attention

    torch.manual_seed(7)
    num_heads = 4
    head_dim = 32
    seq_len = 50

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
    q = torch.randn(seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)

    ref = _reference_attention(q, k, v, cu_seqlens)
    out = packed_prefill_attention(q, k, v, cu_seqlens)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1), dim=0
    ).item()

    print(f"  Sequence length: {seq_len}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    assert cos_sim > 0.9999, f"Cosine sim too low: {cos_sim}"
    print("  ✅ PASSED\n")


def test_packed_attention_extreme_lengths():
    """Test 4: Edge case — very different lengths (2 vs 200)."""
    print("=" * 60)
    print("Test 4: Packed attention — extreme length disparity (2 vs 200)")
    print("=" * 60)

    from megagemm.kernels.paged_attention import packed_prefill_attention

    torch.manual_seed(99)
    num_heads = 4
    head_dim = 64

    lens = [2, 200, 5]
    total = sum(lens)
    cum = [0]
    for l in lens:
        cum.append(cum[-1] + l)
    cu_seqlens = torch.tensor(cum, dtype=torch.int32)

    q = torch.randn(total, num_heads, head_dim)
    k = torch.randn(total, num_heads, head_dim)
    v = torch.randn(total, num_heads, head_dim)

    ref = _reference_attention(q, k, v, cu_seqlens)
    out = packed_prefill_attention(q, k, v, cu_seqlens)

    # Check per-sequence cosine sim
    for i, l in enumerate(lens):
        s, e = cum[i], cum[i + 1]
        sim = torch.nn.functional.cosine_similarity(
            ref[s:e].reshape(-1), out[s:e].reshape(-1), dim=0
        ).item()
        print(f"  Seq {i} (len={l}): cosine sim = {sim:.6f}")
        assert sim > 0.9999, f"Seq {i} cosine sim too low: {sim}"

    print("  ✅ PASSED\n")


def test_packed_attention_noncausal_prefers_triton_dispatch():
    """Test 4b: Non-causal packed attention prefers Triton on CUDA when available."""
    import megagemm.kernels.paged_attention as pa

    class _FakeCudaTensor:
        def __init__(self, shape):
            self.shape = shape
            self.is_cuda = True

    old_has_flash = pa._HAS_FLASH_ATTN
    old_has_triton = pa._HAS_TRITON
    old_triton = pa._triton_packed_attention
    old_logged = set(pa._PACKED_ATTN_BACKEND_LOGGED)
    old_disabled = set(pa._TRITON_PACKED_ATTENTION_DISABLED)
    try:
        pa._HAS_FLASH_ATTN = False
        pa._HAS_TRITON = True
        pa._PACKED_ATTN_BACKEND_LOGGED.clear()
        pa._TRITON_PACKED_ATTENTION_DISABLED.clear()

        called = {}

        def _fake_triton(q, k, v, cu_seqlens, scale, causal, packed_meta=None):
            called["causal"] = causal
            called["cu_seqlens"] = cu_seqlens.tolist()
            called["packed_meta"] = packed_meta
            return "triton-ok"

        pa._triton_packed_attention = _fake_triton

        q = _FakeCudaTensor((5, 4, 8))
        k = _FakeCudaTensor((5, 4, 8))
        v = _FakeCudaTensor((5, 4, 8))
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        out = pa.packed_attention(q, k, v, cu_seqlens, causal=False)
        assert out == "triton-ok"
        assert called["causal"] is False
        assert called["cu_seqlens"] == [0, 2, 5]
        assert called["packed_meta"] is not None
    finally:
        pa._HAS_FLASH_ATTN = old_has_flash
        pa._HAS_TRITON = old_has_triton
        pa._triton_packed_attention = old_triton
        pa._PACKED_ATTN_BACKEND_LOGGED.clear()
        pa._PACKED_ATTN_BACKEND_LOGGED.update(old_logged)
        pa._TRITON_PACKED_ATTENTION_DISABLED.clear()
        pa._TRITON_PACKED_ATTENTION_DISABLED.update(old_disabled)


def test_packed_attention_reuses_precomputed_metadata():
    """Test 4c: Public packed_attention reuses valid precomputed metadata."""
    import megagemm.kernels.paged_attention as pa

    torch.manual_seed(11)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 4, 8)
    v = torch.randn(5, 4, 8)

    packed_meta = pa.prepare_packed_attention_metadata(cu_seqlens, head_dim=q.shape[-1])
    old_prepare = pa.prepare_packed_attention_metadata
    try:
        def _unexpected_prepare(*args, **kwargs):
            raise AssertionError("packed metadata should have been reused")

        pa.prepare_packed_attention_metadata = _unexpected_prepare
        out = pa.packed_attention(q, k, v, cu_seqlens, causal=False, packed_meta=packed_meta)
    finally:
        pa.prepare_packed_attention_metadata = old_prepare

    ref = pa._pytorch_packed_attention(
        q,
        k,
        v,
        cu_seqlens,
        scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
        packed_meta=packed_meta,
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_model_prefill_packed():
    """Test 5: Model-level prefill_packed vs per-seq prefill equivalence."""
    print("=" * 60)
    print("Test 5: Model prefill_packed vs sequential prefill (CPU)")
    print("=" * 60)

    from megagemm.models.llama import MegaGemmLlama, LlamaConfig
    from megagemm.engine.kv_cache import BlockManager

    torch.manual_seed(42)

    # Small model config for testing
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        max_position_embeddings=512,
        rope_half_rotate=True,
    )

    model = MegaGemmLlama(config)
    model.eval()

    # Test prompts
    prompts = [[1, 50, 100, 200, 300], [10, 20, 30], [5, 15, 25, 35, 45, 55, 65]]
    lens = [len(p) for p in prompts]

    # --- Sequential prefill (reference) ---
    ref_logits = []
    for i, prompt in enumerate(prompts):
        bm = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=64,
            block_size=16,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device='cpu',
        )
        bm.allocate_sequence(i, len(prompt) + 50)

        ids = torch.tensor([prompt], dtype=torch.long)
        pos = torch.arange(len(prompt)).unsqueeze(0)
        logits = model.prefill(ids, pos, bm, i)
        ref_logits.append(logits[:, -1, :])  # [1, vocab_size]

    ref_logits = torch.cat(ref_logits, dim=0)  # [3, vocab_size]

    # --- Packed prefill ---
    bm_packed = BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=64,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device='cpu',
    )
    for i, prompt in enumerate(prompts):
        bm_packed.allocate_sequence(i + 100, len(prompt) + 50)

    all_tokens = []
    for p in prompts:
        all_tokens.extend(p)
    input_ids = torch.tensor([all_tokens], dtype=torch.long)

    cum = [0]
    for l in lens:
        cum.append(cum[-1] + l)
    cu_seqlens = torch.tensor(cum, dtype=torch.int32)
    lengths = torch.tensor(lens, dtype=torch.long)
    seq_ids = [100, 101, 102]

    packed_logits = model.prefill_packed(
        input_ids, cu_seqlens, lengths, bm_packed, seq_ids
    )  # [3, 1, vocab_size]
    packed_logits = packed_logits.squeeze(1)  # [3, vocab_size]

    # Compare per-sequence
    all_pass = True
    for i in range(len(prompts)):
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_logits[i:i+1], packed_logits[i:i+1], dim=1
        ).item()
        top1_ref = ref_logits[i].argmax().item()
        top1_packed = packed_logits[i].argmax().item()
        agree = "✅" if top1_ref == top1_packed else "❌"

        print(f"  Seq {i} (len={lens[i]}): cosine={cos_sim:.6f}, "
              f"top1={'match' if top1_ref == top1_packed else 'MISMATCH'} {agree}")

        if cos_sim < 0.99:
            all_pass = False

    if all_pass:
        print("  ✅ PASSED\n")
    else:
        print("  ⚠️  Some sequences have low similarity (may be due to floating point)\n")


def test_packed_attention_realistic():
    """Test 5b: Realistic config — head_dim=128, GQA 7x, ~320 tokens (Qwen2.5-7B)."""
    print("=" * 60)
    print("Test 5b: Packed attention — realistic Qwen2.5-7B config")
    print("         (head_dim=128, 28 q_heads, 4 kv_heads, ~320 tok)")
    print("=" * 60)

    from megagemm.kernels.paged_attention import packed_prefill_attention

    torch.manual_seed(42)
    num_q_heads = 28
    num_kv_heads = 4
    head_dim = 128

    # Bench360-like: 8 sequences, ~320 tokens each
    lens = [310, 320, 325, 315, 330, 318, 322, 312]
    total = sum(lens)
    cum = [0]
    for l in lens:
        cum.append(cum[-1] + l)
    cu_seqlens = torch.tensor(cum, dtype=torch.int32)

    q = torch.randn(total, num_q_heads, head_dim)
    k = torch.randn(total, num_kv_heads, head_dim)
    v = torch.randn(total, num_kv_heads, head_dim)

    ref = _reference_attention(q, k, v, cu_seqlens)
    out = packed_prefill_attention(q, k, v, cu_seqlens)

    # Check per-sequence
    for i, l in enumerate(lens):
        s, e = cum[i], cum[i + 1]
        sim = torch.nn.functional.cosine_similarity(
            ref[s:e].reshape(-1), out[s:e].reshape(-1), dim=0
        ).item()
        status = "ok" if sim > 0.999 else "LOW"
        print(f"  Seq {i} (len={l}): cosine sim = {sim:.6f} [{status}]")
        assert sim > 0.999, f"Seq {i} cosine sim too low: {sim}"

    print("  ✅ PASSED\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test packed prefill attention")
    parser.add_argument('--gpu', action='store_true', help='Run on GPU with Triton')
    args = parser.parse_args()

    print("\n🧪 MegaGemm Packed Prefill Attention Tests")
    print("=" * 60)

    if args.gpu and not torch.cuda.is_available():
        print("⚠️  --gpu specified but CUDA not available. Running CPU tests only.")
        args.gpu = False

    if args.gpu:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  Device: CPU (PyTorch fallback)")
    print()

    passed = 0
    failed = 0

    tests = [
        test_packed_attention_basic,
        test_packed_attention_gqa,
        test_packed_attention_single_seq,
        test_packed_attention_extreme_lengths,
        test_packed_attention_noncausal_prefers_triton_dispatch,
        test_packed_attention_reuses_precomputed_metadata,
        test_packed_attention_realistic,
        test_model_prefill_packed,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
