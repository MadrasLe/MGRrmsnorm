from benchmarks.run_gemma4_e2b_lm_head_sweep import decide_paired_lm_head_sweep


def _row(
    name,
    speedups,
    *,
    tokens_equal=True,
    error=None,
    current_us=100.0,
):
    ordered = sorted(speedups)
    median_speedup = ordered[len(ordered) // 2]
    return {
        "name": name,
        "tokens_equal": tokens_equal,
        "error": error,
        "pair_speedups": speedups,
        "median_speedup": median_speedup,
        "speedup_spread_ratio": max(speedups) / min(speedups),
        "candidate_faster_pairs": sum(value > 1.0 for value in speedups),
        "current_median_us": current_us,
        "candidate_median_us": current_us / median_speedup,
    }


def test_paired_lm_head_sweep_promotes_stable_consistent_winner():
    decision = decide_paired_lm_head_sweep(
        [
            _row("candidate", [1.04, 1.05, 1.04, 1.06, 1.05, 1.04, 1.05]),
            _row("slower", [0.98, 0.99, 1.00, 0.98, 0.99, 1.00, 0.98]),
        ],
        minimum_speedup=1.03,
        maximum_ratio_spread=1.08,
        minimum_faster_fraction=0.80,
    )
    assert decision["apply_change"] is True
    assert decision["winner"] == "candidate"
    assert decision["speedup"] == 1.05


def test_paired_lm_head_sweep_rejects_noisy_or_incorrect_candidates():
    decision = decide_paired_lm_head_sweep(
        [
            _row("noisy", [0.98, 1.10, 0.99, 1.09, 1.00, 1.08, 1.01]),
            _row(
                "fast_but_wrong",
                [1.10, 1.11, 1.10, 1.12, 1.11, 1.10, 1.11],
                tokens_equal=False,
            ),
        ],
        minimum_speedup=1.03,
        maximum_ratio_spread=1.08,
        minimum_faster_fraction=0.80,
    )
    assert decision["apply_change"] is False
    assert decision["winner"] is None
