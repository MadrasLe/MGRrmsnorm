from benchmarks.run_gemma4_e2b_lm_head_sweep import decide_lm_head_sweep


def _row(name, values, *, baseline=False, tokens_equal=True, error=None):
    return {
        "name": name,
        "baseline": baseline,
        "median_us": sorted(values)[len(values) // 2],
        "samples_us": values,
        "spread_ratio": max(values) / min(values),
        "tokens_equal": tokens_equal,
        "error": error,
    }


def test_lm_head_sweep_promotes_separated_stable_winner():
    decision = decide_lm_head_sweep(
        [
            _row("current", [100.0, 101.0, 102.0], baseline=True),
            _row("candidate", [80.0, 81.0, 82.0]),
            _row("current_recheck", [99.0, 100.0, 101.0], baseline=True),
        ],
        minimum_speedup=1.03,
        maximum_spread=1.05,
    )
    assert decision["apply_change"] is True
    assert decision["winner"] == "candidate"
    assert decision["sample_ranges_separate"] is True


def test_lm_head_sweep_rejects_overlap_or_incorrect_candidate():
    decision = decide_lm_head_sweep(
        [
            _row("current", [100.0, 101.0, 102.0], baseline=True),
            _row("overlap", [97.0, 99.0, 101.0]),
            _row("fast_but_wrong", [70.0, 71.0, 72.0], tokens_equal=False),
            _row("current_recheck", [99.0, 100.0, 101.0], baseline=True),
        ],
        minimum_speedup=1.01,
        maximum_spread=1.05,
    )
    assert decision["apply_change"] is False
    assert decision["winner"] == "overlap"
    assert decision["sample_ranges_separate"] is False
