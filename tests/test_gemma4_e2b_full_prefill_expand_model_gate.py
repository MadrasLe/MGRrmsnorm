from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_full_prefill_expand_model_gate.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_full_prefill_expand_model_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("full_prefill_expand_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(candidate_ms: float = 1900.0):
    rows = []
    for pair, offset in ((1, -10.0), (2, 0.0), (3, 10.0)):
        digest = "same"
        rows.extend(
            [
                {
                    "case": "baseline_explicit",
                    "prefill_ms": 2550.0 + offset,
                    "wall_ms": 2580.0 + offset,
                    "runtime_hits_delta": 0,
                    "runtime_error": "",
                    "token_digest": digest,
                },
                {
                    "case": "candidate_expanded_implicit",
                    "prefill_ms": candidate_ms + offset,
                    "wall_ms": candidate_ms + 30.0 + offset,
                    "runtime_hits_delta": 7,
                    "runtime_error": "",
                    "token_digest": digest,
                },
            ]
        )
    return rows


def test_summary_promotes_correct_stable_full_model_speedup():
    module = _load_module()
    result = module.summarize(
        _samples(), minimum_speedup=1.05, maximum_spread=1.08
    )
    assert result["decision"] == "PROMOTE_E2B_L4_FULL_PREFILL_EXPAND"
    assert result["prefill_speedup"] == pytest.approx(2550.0 / 1900.0)
    assert result["prefill_saved_ms"] == pytest.approx(650.0)
    assert result["token_digest_exact"] is True
    assert result["runtime_hits_exact"] is True


def test_summary_rejects_missing_candidate_hits():
    module = _load_module()
    rows = _samples()
    rows[1]["runtime_hits_delta"] = 0
    result = module.summarize(
        rows, minimum_speedup=1.05, maximum_spread=1.08
    )
    assert result["apply_change"] is False
    assert result["runtime_hits_exact"] is False


def test_source_path_is_opt_in_until_loaded_model_gate_passes():
    policy = (ROOT / "megagemm" / "models" / "runtime_policy.py").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "gemma4_e2b_l4_full_prefill_expand: bool = False" in policy
    assert "MEGAGEMM_GEMMA4_E2B_L4_FULL_PREFILL_EXPAND" in model
    assert "expanded_k = k.repeat_interleave(8, dim=1)" in model
    assert "attn_mask=None" in model


def test_colab_harness_is_drive_scoped_without_competing_engine():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "import vllm" not in source
    assert "full_prefill_expand_model_gate.py" in source
    assert "decision.json" in source
