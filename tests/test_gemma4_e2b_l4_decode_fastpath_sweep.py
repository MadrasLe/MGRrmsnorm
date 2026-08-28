from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_l4_decode_fastpath_sweep.py"


def _module():
    spec = importlib.util.spec_from_file_location("e2b_decode_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_context_decode_sweep_covers_isolated_and_current_paths():
    module = _module()
    cases = {name: overrides for name, overrides in module.CASES}

    assert set(cases) == {
        "cublas_baseline",
        "fused_gateup",
        "deepfusion",
        "current_policy",
    }
    assert cases["cublas_baseline"]["MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE"] == "0"
    assert cases["fused_gateup"]["MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE"] == "1"
    assert cases["deepfusion"]["MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE"] == "1"
    assert cases["current_policy"]["MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE"] is None


def test_case_environment_keeps_the_e2b_production_path_enabled(monkeypatch):
    module = _module()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = module._case_environment(dict(module.CASES[0][1]))

    assert env["PYTHONPATH"] == str(ROOT)
    assert env["MEGAGEMM_FLAT_DECODE"] == "1"
    assert env["MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE"] == "1"
    assert env["MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL"] == "1"
    assert env["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "0"
