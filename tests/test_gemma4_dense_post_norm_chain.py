from pathlib import Path

import torch

from megagemm.kernels.rmsnorm_triton import (
    rmsnorm_triton_residual_scale_next,
)


ROOT = Path(__file__).resolve().parents[1]


def _rmsnorm(x, weight, eps, offset=False):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    scale = weight + 1.0 if offset else weight
    return (x * torch.rsqrt(variance + eps) * scale).to(x.dtype)


def test_dense_post_norm_chain_cpu_fallback_matches_staged_reference():
    torch.manual_seed(7)
    branch = torch.randn(3, 17)
    residual = torch.randn_like(branch)
    weight = torch.randn(17)
    next_weight = torch.randn(17)
    scalar = torch.tensor([0.875])
    eps = 1e-6

    expected_branch = _rmsnorm(branch, weight, eps, offset=True)
    expected_hidden = (residual + expected_branch).to(branch.dtype)
    expected_hidden = (expected_hidden * scalar).to(branch.dtype)
    expected_next = _rmsnorm(
        expected_hidden,
        next_weight,
        eps,
        offset=True,
    )

    hidden, next_norm = rmsnorm_triton_residual_scale_next(
        branch,
        residual,
        weight,
        scalar,
        next_weight,
        eps,
        norm_offset=True,
        next_norm_offset=True,
    )

    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(next_norm, expected_next)


def test_dense_post_norm_chain_supports_residual_alias_and_final_layer():
    torch.manual_seed(11)
    branch = torch.randn(2, 9)
    residual = torch.randn_like(branch)
    original_residual = residual.clone()
    weight = torch.randn(9)
    scalar = torch.tensor([1.125])

    expected = (
        original_residual + _rmsnorm(branch, weight, 1e-5)
    ) * scalar
    hidden, next_norm = rmsnorm_triton_residual_scale_next(
        branch,
        residual,
        weight,
        scalar,
        None,
        out_hidden=residual,
    )

    assert hidden.data_ptr() == residual.data_ptr()
    assert next_norm is None
    torch.testing.assert_close(hidden, expected)


def test_candidate_is_not_promoted_without_l4_ab_evidence():
    llama = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    publication = (
        ROOT / "benchmarks" / "run_publication_gpu_suite.py"
    ).read_text(encoding="utf-8")
    harness = (
        ROOT / "benchmarks" / "run_gemma4_dense_post_norm_chain_colab.sh"
    ).read_text(encoding="utf-8")

    assert (
        '"MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE", default=False'
        in llama
    )
    profile_definitions = publication.split("MEGAGEMM_PROFILES =", 1)[0]
    assert (
        "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE"
        not in profile_definitions
    )
    assert "dense_tail_requested" in publication
    assert "--variants megagemm-bf16" in harness
    assert "vllm-bf16" not in harness
    assert 'export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"' in harness
