from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_l4_full_prefill_sdpa_gate.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_l4_full_prefill_sdpa_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("full_prefill_sdpa_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(case: str, us: float, *, correct: bool = True, spread: float = 1.01):
    return {
        "case": case,
        "correct": correct,
        "median_us": us,
        "spread_ratio": spread,
    }


def test_decision_promotes_fast_stable_implicit_candidate():
    module = _load_module()
    result = module.decide(
        [
            _row("explicit_native", 100_000.0),
            _row("implicit_native", 20_000.0),
            _row("implicit_expanded", 30_000.0),
            _row("explicit_native_recheck", 101_000.0),
        ],
        minimum_speedup=1.03,
        maximum_spread=1.05,
    )
    assert result["decision"] == "PROMOTE_IMPLICIT_CAUSAL_NATIVE_SDPA"
    assert result["winner"] == "implicit_native"
    assert result["speedup"] == pytest.approx(5.05)


def test_decision_names_expanded_winner_explicitly():
    module = _load_module()
    result = module.decide(
        [
            _row("implicit_native", 105_000.0),
            _row("implicit_expanded", 13_400.0),
            _row("explicit_native_recheck", 104_000.0),
        ],
        minimum_speedup=1.03,
        maximum_spread=1.05,
    )
    assert result["decision"] == "PROMOTE_IMPLICIT_CAUSAL_EXPANDED_SDPA"
    assert result["winner"] == "implicit_expanded"


def test_decision_rejects_unstable_candidate():
    module = _load_module()
    result = module.decide(
        [
            _row("explicit_native_recheck", 100_000.0),
            _row("implicit_native", 20_000.0, spread=1.20),
        ],
        minimum_speedup=1.03,
        maximum_spread=1.05,
    )
    assert result["decision"] == "KEEP_EXPLICIT_AND_BUILD_TRITON_FULL_H512"
    assert result["apply_change"] is False


def test_colab_harness_is_drive_scoped_and_dependency_free():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "--seq-len 2057" in source
    assert "decision.json" in source
