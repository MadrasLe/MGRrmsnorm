from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_unrolled_decode_graph.py"


def _module():
    spec = importlib.util.spec_from_file_location("e2b_unrolled_graph", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unrolled_graph_gate_is_process_isolated_and_shape_exact(monkeypatch):
    module = _module()
    cases = {name: values for name, values in module.CASES}

    assert set(cases) == {
        "eager_multi_step",
        "one_step_graph",
        "unrolled_graph",
    }
    assert cases["eager_multi_step"]["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "0"
    assert cases["one_step_graph"]["MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST"] == "0"
    assert cases["unrolled_graph"]["MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST"] == "1"

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = module._environment(cases["unrolled_graph"])
    assert env["PYTHONPATH"] == str(ROOT)
    assert env["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "1"
    assert env["MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK"] == "1"
    assert env["MEGAGEMM_MULTI_STEP_BURST_BATCH"] == "8"
    assert env["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] == "1"
    assert env["MEGAGEMM_NATIVE_DECODE_GRAPH_BURST"] == "0"
    assert env["MEGAGEMM_REUSE_REQUEST_SCHEDULER"] == "0"

    reused_env = module._environment(
        cases["unrolled_graph"], reuse_scheduler=True
    )
    assert reused_env["MEGAGEMM_REUSE_REQUEST_SCHEDULER"] == "1"


def test_unrolled_graph_result_requires_deterministic_tokens_and_real_capture():
    module = _module()
    digest = "a" * 64
    row = {
        "generated_tokens": 512,
        "generated_token_digest": digest,
        "output_tps": 120.0,
        "scheduler_stats": {
            "decode_time_ms": 2000.0,
            "decode_cuda_graphs": {
                "failures": 0,
                "unrolled_token_burst_captures": 2,
                "unrolled_token_burst_replays": 5,
                "unrolled_token_bursts": 7,
                "unrolled_token_burst_steps": 55,
                "unrolled_token_burst_failures": 0,
            },
        },
    }

    result = module._case_result("unrolled_graph", [row, dict(row)])

    assert result["generated_token_digest"] == digest
    assert result["median_decode_tps"] == 256.0
    assert result["unrolled_captures"] == 4
    assert result["unrolled_replays"] == 10
    assert result["unrolled_bursts"] == 14
    assert result["unrolled_steps"] == 110


def test_unrolled_graph_result_accepts_warmup_capture_reused_by_measured_row():
    module = _module()
    row = {
        "generated_tokens": 512,
        "generated_token_digest": "b" * 64,
        "output_tps": 150.0,
        "scheduler_stats": {
            "decode_time_ms": 1600.0,
            "decode_cuda_graphs": {
                "failures": 0,
                "unrolled_token_burst_captures": 0,
                "unrolled_token_burst_replays": 7,
                "unrolled_token_bursts": 7,
                "unrolled_token_burst_steps": 55,
                "unrolled_token_burst_failures": 0,
            },
        },
    }

    result = module._case_result("unrolled_graph", [row])

    assert result["unrolled_captures"] == 0
    assert result["unrolled_replays"] == 7
    assert result["median_decode_tps"] == 320.0
