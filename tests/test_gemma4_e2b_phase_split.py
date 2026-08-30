from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_phase_split.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_phase_split_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_phase_split", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples():
    return [
        {
            "pair_index": 1,
            "max_new_tokens_per_request": 1,
            "elapsed_s": 2.0,
            "internal": {"prefill_ms": 1900.0},
        },
        {
            "pair_index": 1,
            "max_new_tokens_per_request": 128,
            "elapsed_s": 5.0,
            "internal": {"prefill_ms": 1910.0, "decode_ms": 3050.0},
        },
        {
            "pair_index": 2,
            "max_new_tokens_per_request": 128,
            "elapsed_s": 5.3,
            "internal": {"prefill_ms": 1930.0, "decode_ms": 3100.0},
        },
        {
            "pair_index": 2,
            "max_new_tokens_per_request": 1,
            "elapsed_s": 2.2,
            "internal": {"prefill_ms": 1920.0},
        },
    ]


def test_paired_phase_summary_uses_delta_for_127_tokens_per_request():
    module = _load_module()
    result = module.summarize_samples(
        _samples(),
        batch_size=8,
        short_tokens=1,
        long_tokens=128,
    )
    assert result["complete_pairs"] == 2
    assert result["first_token_phase_ms"] == pytest.approx(2100.0)
    assert result["incremental_decode_ms"] == pytest.approx(3050.0)
    assert result["incremental_decode_tokens"] == 1016
    assert result["incremental_decode_tps"] == pytest.approx(1016 / 3.05)
    assert result["long_total_ms"] == pytest.approx(5150.0)
    assert result["internal_long_prefill_ms"] == pytest.approx(1920.0)
    assert result["internal_long_decode_ms"] == pytest.approx(3075.0)


def test_measurement_arguments_are_json_serializable_paths():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "measure",
            "--backend",
            "megagemm",
            "--output",
            "results/phase.json",
        ]
    )
    encoded = module.json.dumps(module._jsonable_args(args))
    assert '"output": "results' in encoded


def test_measure_writes_complete_json_with_one_loaded_runner(monkeypatch, tmp_path):
    module = _load_module()
    output = tmp_path / "vllm.json"
    args = module.build_parser().parse_args(
        [
            "measure",
            "--backend",
            "vllm",
            "--warmups",
            "0",
            "--repeats",
            "2",
            "--output",
            str(output),
        ]
    )
    loads = []

    monkeypatch.setattr(module.matrix, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module.matrix,
        "build_prompts",
        lambda *_args, **_kwargs: (["x"] * 8, 16456),
    )
    monkeypatch.setattr(module.matrix, "cleanup_cuda", lambda: None)
    monkeypatch.setattr(module.matrix, "git_snapshot", lambda: {"commit": "test"})
    monkeypatch.setattr(module.matrix, "gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(module.matrix, "nvidia_smi_snapshot", lambda: {"available": False})
    monkeypatch.setattr(module.matrix, "installed_package_versions", lambda: {})

    def make_runner(*_args):
        loads.append(1)

        def run(_prompts, token_count):
            elapsed = 2.0 if token_count == 1 else 5.0
            return {
                "elapsed_s": elapsed,
                "generated_tokens": 8 * token_count,
                "extra": {},
            }

        return run

    monkeypatch.setattr(module.matrix, "make_runner", make_runner)
    payload = module.measure(args)
    persisted = module.json.loads(output.read_text(encoding="utf-8"))
    assert loads == [1]
    assert payload["summary"]["complete_pairs"] == 2
    assert persisted["args"]["output"] == str(output)
    assert persisted["summary"]["incremental_decode_tokens"] == 1016


def test_phase_comparison_attributes_measured_wall_gap():
    module = _load_module()
    common = {
        "model": module.DEFAULT_MODEL,
        "dtype": "bf16",
        "hardware_label": "1xl4",
        "prompt_tokens_actual_total": 16456,
        "args": {
            "batch_size": 8,
            "prompt_tokens": 2048,
            "short_tokens": 1,
            "long_tokens": 128,
        },
    }
    mg = {
        **common,
        "backend": "megagemm",
        "summary": {
            "first_token_phase_ms": 2000.0,
            "incremental_decode_ms": 3500.0,
            "incremental_decode_tps": 290.0,
            "long_total_ms": 5500.0,
            "long_output_tps": 186.0,
        },
    }
    vl = {
        **common,
        "backend": "vllm",
        "summary": {
            "first_token_phase_ms": 1500.0,
            "incremental_decode_ms": 2500.0,
            "incremental_decode_tps": 406.4,
            "long_total_ms": 4000.0,
            "long_output_tps": 256.0,
        },
    }
    result = module.compare_payloads(mg, vl)
    assert result["gaps"]["first_token_phase_ms"] == pytest.approx(500.0)
    assert result["gaps"]["incremental_decode_ms"] == pytest.approx(1000.0)
    assert result["gaps"]["accounted_gap_ms"] == pytest.approx(1500.0)
    assert result["positive_gap_attribution"]["first_token_phase_fraction"] == pytest.approx(
        1 / 3
    )
    assert result["positive_gap_attribution"]["incremental_decode_fraction"] == pytest.approx(
        2 / 3
    )
    assert result["positive_gap_attribution"]["dominant"] == "incremental_decode"


def test_colab_harness_is_drive_scoped_fresh_session_and_same_stack():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "vllm-0.26.0%2Bcu129" in source
    assert "torchcodec==0.16.0+cpu" in source
    assert source.index("--backend vllm") < source.index("--backend megagemm")
    assert "--short-tokens 1" in source
    assert "--long-tokens 128" in source
    assert "comparison.json" in source
