"""
🧪 RoPE Correctness Tests
Autor: Gabriel Yogi
Descrição: Validate RoPE implementation against HuggingFace and gradcheck

Run with: python -m pytest tests/test_rope.py -v
"""

import torch
import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megagemm import RoPE, apply_rotary_emb, precompute_freqs_cis


def _has_transformers():
    """Check if transformers library is available."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TestRoPEBasic:
    """Basic functionality tests"""

    def test_precompute_freqs_cis_shape(self):
        """Test that precomputed freqs have correct shape"""
        dim = 64
        max_seq_len = 128

        cos, sin = precompute_freqs_cis(dim, max_seq_len)

        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_precompute_freqs_cis_values(self):
        """Test that cos²+sin²=1"""
        dim = 64
        max_seq_len = 128

        cos, sin = precompute_freqs_cis(dim, max_seq_len)

        identity = cos ** 2 + sin ** 2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-6)

    def test_apply_rotary_emb_shape(self):
        """Test that output shapes match input"""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64

        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos, sin = precompute_freqs_cis(head_dim, seq_len)

        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_module(self):
        """Test RoPE nn.Module"""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64

        rope = RoPE(head_dim=head_dim, max_seq_len=128)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestRoPEHuggingFace:
    """Compare against HuggingFace LLaMA implementation"""

    @pytest.mark.skipif(
        not _has_transformers(),
        reason="transformers not installed"
    )
    def test_vs_huggingface_llama(self):
        """Compare output with HuggingFace LLaMA RoPE"""
        from transformers.models.llama.modeling_llama import (
            LlamaRotaryEmbedding,
            apply_rotary_pos_emb
        )
        from transformers.models.llama.configuration_llama import LlamaConfig

        batch, heads, seq_len, head_dim = 2, 8, 64, 64

        # Our implementation
        our_rope = RoPE(head_dim=head_dim, max_seq_len=128, half_rotate=True)

        # HuggingFace implementation
        try:
            hf_rope = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=128)
        except TypeError:
            # Transformers 4.51+ constructs RoPE from the model config.
            config = LlamaConfig(
                hidden_size=heads * head_dim,
                num_attention_heads=heads,
                num_key_value_heads=heads,
                max_position_embeddings=128,
            )
            hf_rope = LlamaRotaryEmbedding(config=config)

        # Input tensors
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        # Our output
        q_our, k_our = our_rope(q, k)

        # HuggingFace output
        cos, sin = hf_rope(k, position_ids)
        q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)

        # Compare (may need tolerance due to implementation differences)
        assert torch.allclose(q_our, q_hf, atol=1e-4), \
            f"Q mismatch: max diff = {(q_our - q_hf).abs().max()}"
        assert torch.allclose(k_our, k_hf, atol=1e-4), \
            f"K mismatch: max diff = {(k_our - k_hf).abs().max()}"


class TestRoPEGradients:
    """Gradient correctness tests"""

    def test_backward_runs(self):
        """Test that backward pass runs without error"""
        batch, heads, seq_len, head_dim = 2, 4, 32, 32

        rope = RoPE(head_dim=head_dim, max_seq_len=64)
        q = torch.randn(batch, heads, seq_len, head_dim, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, requires_grad=True)

        q_rot, k_rot = rope(q, k)
        loss = (q_rot.sum() + k_rot.sum())
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape

    def test_gradcheck(self):
        """Use torch.autograd.gradcheck for numerical gradient verification"""
        head_dim = 16
        batch, heads, seq_len = 1, 2, 8

        cos, sin = precompute_freqs_cis(head_dim, seq_len)
        cos = cos.double()
        sin = sin.double()

        def rope_fn(q, k):
            return apply_rotary_emb(q, k, cos, sin)

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.double, requires_grad=True)

        # This will raise an error if gradients are incorrect
        assert torch.autograd.gradcheck(rope_fn, (q, k), eps=1e-6, atol=1e-4)


class TestRoPEPrecision:
    """Numerical precision tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_stability(self):
        """Test FP16 doesn't produce NaN/Inf"""
        batch, heads, seq_len, head_dim = 4, 8, 512, 64

        rope = RoPE(head_dim=head_dim, max_seq_len=1024).cuda()
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

        q_rot, k_rot = rope(q, k)

        assert not torch.isnan(q_rot).any(), "NaN in Q output"
        assert not torch.isnan(k_rot).any(), "NaN in K output"
        assert not torch.isinf(q_rot).any(), "Inf in Q output"
        assert not torch.isinf(k_rot).any(), "Inf in K output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bf16_stability(self):
        """Test BF16 doesn't produce NaN/Inf"""
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("BF16 requires Ampere+")

        batch, heads, seq_len, head_dim = 4, 8, 512, 64

        rope = RoPE(head_dim=head_dim, max_seq_len=1024).cuda()
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

        q_rot, k_rot = rope(q, k)

        assert not torch.isnan(q_rot).any(), "NaN in Q output"
        assert not torch.isnan(k_rot).any(), "NaN in K output"


if __name__ == "__main__":
    # Quick manual test
    print("Running basic RoPE tests...")

    # Basic test
    batch, heads, seq_len, head_dim = 2, 8, 64, 64
    rope = RoPE(head_dim=head_dim, max_seq_len=128)
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k)
    print(f"✅ Basic forward: Q {q.shape} -> {q_rot.shape}")

    # Backward test
    q = torch.randn(batch, heads, seq_len, head_dim, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, requires_grad=True)
    q_rot, k_rot = rope(q, k)
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()
    print(f"✅ Backward: grad shapes Q={q.grad.shape}, K={k.grad.shape}")

    # cos²+sin²=1 test
    cos, sin = precompute_freqs_cis(64, 128)
    identity = cos ** 2 + sin ** 2
    max_err = (identity - 1.0).abs().max()
    print(f"✅ cos²+sin²=1: max error = {max_err:.2e}")

    print("\n🎉 All basic tests passed!")
