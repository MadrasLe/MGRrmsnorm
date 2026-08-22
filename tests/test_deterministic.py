"""
🔒 MegaGemm Deterministic Inference Test
==========================================
Verifies that deterministic mode guarantees bit-exact reproducible output.

Tests:
1. deterministic.py module API (enable/disable/is_deterministic)
2. Model forward pass reproducibility (bit-exact logits)
3. Greedy sampling reproducibility (same tokens every time)
4. Sampling with temperature + fixed seed reproducibility

Usage:
    python tests/test_deterministic.py

Author: Gabriel Yogi
"""

import sys
import os
import torch

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _run_deterministic_api():
    """Test 1: deterministic module API."""
    print("=" * 60)
    print("TEST 1: Deterministic Module API")
    print("=" * 60)

    from megagemm.engine.deterministic import (
        enable_deterministic_mode,
        disable_deterministic_mode,
        is_deterministic,
    )

    # Initially off
    assert not is_deterministic(), "Should start disabled"
    print("  ✅ is_deterministic() = False (initial)")

    # Enable
    enable_deterministic_mode(seed=123)
    assert is_deterministic(), "Should be enabled after enable_deterministic_mode()"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    print("  ✅ Enabled: CUBLAS_WORKSPACE_CONFIG set, is_deterministic() = True")

    # Disable
    disable_deterministic_mode()
    assert not is_deterministic(), "Should be disabled after disable_deterministic_mode()"
    print("  ✅ Disabled: is_deterministic() = False")

    print("\n  🎉 Deterministic API tests passed!")
    return True


def _run_model_reproducibility():
    """Test 2: Model forward pass produces bit-exact logits."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Forward Reproducibility (CPU)")
    print("=" * 60)

    from megagemm.models.llama import LlamaConfig, MegaGemmLlama
    from megagemm.engine.kv_cache import BlockManager
    from megagemm.engine.deterministic import (
        enable_deterministic_mode,
        disable_deterministic_mode,
    )

    # Tiny model for testing
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
    )

    # Run 1
    enable_deterministic_mode(seed=42)
    model = MegaGemmLlama(config)

    bm1 = BlockManager(
        num_layers=2, num_blocks=32, block_size=16,
        num_kv_heads=2, head_dim=16, dtype=torch.float32, device='cpu',
    )

    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    positions = torch.arange(5).unsqueeze(0)
    bm1.allocate_sequence(seq_id=0, num_tokens=5)

    with torch.no_grad():
        logits1 = model.prefill(input_ids, positions, bm1, seq_id=0)

    bm1.free_sequence(0)
    disable_deterministic_mode()

    # Run 2 (re-seed, re-init with same seed)
    enable_deterministic_mode(seed=42)
    model2 = MegaGemmLlama(config)

    bm2 = BlockManager(
        num_layers=2, num_blocks=32, block_size=16,
        num_kv_heads=2, head_dim=16, dtype=torch.float32, device='cpu',
    )
    bm2.allocate_sequence(seq_id=0, num_tokens=5)

    with torch.no_grad():
        logits2 = model2.prefill(input_ids, positions, bm2, seq_id=0)

    bm2.free_sequence(0)
    disable_deterministic_mode()

    # Compare bit-exact
    match = torch.equal(logits1, logits2)
    max_diff = (logits1 - logits2).abs().max().item()
    print(f"  Logits shape: {logits1.shape}")
    print(f"  Max difference: {max_diff}")
    print(f"  Bit-exact match: {match}")

    assert match, f"Logits not bit-exact! Max diff: {max_diff}"
    print("  ✅ Forward pass is bit-exact reproducible!")

    print("\n  🎉 Model reproducibility test passed!")
    return True


def _run_greedy_sampling_reproducibility():
    """Test 3: Greedy sampling (argmax) is always deterministic."""
    print("\n" + "=" * 60)
    print("TEST 3: Greedy Sampling Reproducibility")
    print("=" * 60)

    from megagemm.engine.sampling import sample_logits

    logits = torch.randn(1, 1000)

    tokens1 = sample_logits(logits.clone(), temperature=0.0)
    tokens2 = sample_logits(logits.clone(), temperature=0.0)
    tokens3 = sample_logits(logits.clone(), temperature=0.0)

    assert torch.equal(tokens1, tokens2)
    assert torch.equal(tokens2, tokens3)
    print(f"  Token: {tokens1.item()} (same across 3 runs)")
    print("  ✅ Greedy sampling is deterministic!")

    print("\n  🎉 Greedy sampling test passed!")
    return True


def _run_seeded_sampling_reproducibility():
    """Test 4: Sampling with temperature + fixed seed is reproducible."""
    print("\n" + "=" * 60)
    print("TEST 4: Seeded Sampling Reproducibility")
    print("=" * 60)

    from megagemm.engine.sampling import sample_logits

    logits = torch.randn(1, 1000)

    # Run 1 with seed
    torch.manual_seed(42)
    tokens1 = sample_logits(logits.clone(), temperature=0.8, top_k=50)

    # Run 2 with same seed
    torch.manual_seed(42)
    tokens2 = sample_logits(logits.clone(), temperature=0.8, top_k=50)

    # Run 3 with same seed
    torch.manual_seed(42)
    tokens3 = sample_logits(logits.clone(), temperature=0.8, top_k=50)

    assert torch.equal(tokens1, tokens2), f"Run 1 ({tokens1.item()}) != Run 2 ({tokens2.item()})"
    assert torch.equal(tokens2, tokens3), f"Run 2 ({tokens2.item()}) != Run 3 ({tokens3.item()})"
    print(f"  Token: {tokens1.item()} (same across 3 seeded runs)")
    print("  ✅ Seeded temperature sampling is reproducible!")

    print("\n  🎉 Seeded sampling test passed!")
    return True


def test_deterministic_api():
    assert _run_deterministic_api()


def test_model_reproducibility():
    assert _run_model_reproducibility()


def test_greedy_sampling_reproducibility():
    assert _run_greedy_sampling_reproducibility()


def test_seeded_sampling_reproducibility():
    assert _run_seeded_sampling_reproducibility()


def main():
    print("🔒 MegaGemm Deterministic Inference Tests")
    print("=" * 60)

    results = {}

    results['api'] = _run_deterministic_api()
    results['model_repro'] = _run_model_reproducibility()
    results['greedy_sample'] = _run_greedy_sampling_reproducibility()
    results['seeded_sample'] = _run_seeded_sampling_reproducibility()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n{'🎉 All deterministic tests passed!' if all_passed else '⚠️ Some tests failed'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
