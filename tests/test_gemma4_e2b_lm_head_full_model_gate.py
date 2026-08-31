from benchmarks.run_gemma4_e2b_lm_head_full_model_gate import (
    summarize_full_model_gate,
)


def _sample(case, pair, decode_ms, digest="same"):
    return {
        "case": case,
        "pair": pair,
        "decode_ms": decode_ms,
        "wall_ms": decode_ms + 100.0,
        "prefill_ms": 100.0,
        "token_digest": digest,
        "fused_rmsnorm_lm_head_use": True,
        "fused_rmsnorm_lm_head_disabled": False,
        "fused_rmsnorm_lm_head_error": "",
    }


def test_full_model_gate_promotes_consistent_exact_candidate():
    samples = []
    for pair in range(1, 6):
        samples.extend(
            [
                _sample("production", pair, 100.0 + pair),
                _sample("candidate_bn64", pair, 98.0 + pair),
            ]
        )
    summary = summarize_full_model_gate(
        samples,
        batch_size=8,
        max_new_tokens=128,
        minimum_speedup=1.005,
        maximum_ratio_spread=1.06,
        minimum_faster_fraction=0.80,
    )
    assert summary["apply_change"] is True
    assert summary["candidate_faster_pairs"] == 5
    assert summary["token_digest_exact"] is True


def test_full_model_gate_rejects_token_mismatch():
    samples = []
    for pair in range(1, 6):
        samples.extend(
            [
                _sample("production", pair, 100.0, digest="current"),
                _sample("candidate_bn64", pair, 90.0, digest="candidate"),
            ]
        )
    summary = summarize_full_model_gate(
        samples,
        batch_size=8,
        max_new_tokens=128,
        minimum_speedup=1.005,
        maximum_ratio_spread=1.06,
        minimum_faster_fraction=0.80,
    )
    assert summary["apply_change"] is False
    assert summary["token_digest_exact"] is False
