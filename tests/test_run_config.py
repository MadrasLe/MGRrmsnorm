import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megagemm.__main__ import _create_cli_inference_engine, main
from megagemm.run_config import (
    RunConfigError,
    execute_run_config,
    load_prompts,
    load_run_config,
    normalized_config_dict,
    validate_run_config,
)


class _FakeEngine:
    last_instance = None

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = dict(kwargs)
        self.generate_calls = []
        self.batch_calls = []
        _FakeEngine.last_instance = self

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, dict(kwargs)))
        return f"generated: {prompt}"

    def generate_batch(self, prompts, **kwargs):
        self.batch_calls.append((list(prompts), dict(kwargs)))
        return [f"generated: {prompt}" for prompt in prompts]


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_json_config_is_strict_and_resolves_paths(tmp_path):
    config_path = tmp_path / "inference.json"
    _write_json(
        config_path,
        {
            "version": 1,
            "task": "generate",
            "model": "example/model",
            "engine": {
                "device": "cuda",
                "dtype": "bf16",
                "max_batch_size": 8,
                "max_seq_len": 2304,
                "deterministic": True,
            },
            "generation": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
            },
            "prompt": "Explain paged attention.",
            "output": {"format": "json", "path": "results/out.json"},
        },
    )

    config = load_run_config(config_path)

    assert config.model == "example/model"
    assert config.engine.dtype == "bf16"
    assert config.engine.max_seq_len == 2304
    assert config.generation.temperature == 0.0
    assert config.output.path == (tmp_path / "results" / "out.json").resolve()
    assert validate_run_config(config) == ("Explain paged attention.",)

    normalized = normalized_config_dict(config, prompt_count=1)
    assert normalized["input"]["mode"] == "single"
    assert normalized["input"]["api"] == "generate"
    assert normalized["input"]["prompt_count"] == 1


def test_legacy_cli_uses_shared_typed_engine_config():
    args = SimpleNamespace(
        model="example/model",
        device="cuda",
        bf16=True,
        max_batch_size=4,
        quantize="int8",
        max_seq_len=2048,
        monitor=True,
        dashboard=False,
        deterministic=True,
        seed=9,
        mgx_skip_hash_check=True,
        mgx_prefer_payload_cache=True,
        mgx_payload_cache_dir="cache",
    )
    sentinel = object()

    with patch(
        "megagemm.run_config.create_inference_engine",
        return_value=sentinel,
    ) as create:
        result = _create_cli_inference_engine(args)

    assert result is sentinel
    model, config = create.call_args.args
    assert model == "example/model"
    assert config.dtype == "bf16"
    assert config.max_batch_size == 4
    assert config.quantize == "int8"
    assert config.deterministic is True
    assert config.mgx_verify_payload is False


def test_unknown_field_reports_its_config_path(tmp_path):
    config_path = tmp_path / "bad.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "prompt": "hello",
            "engine": {"max_seq_lenght": 4096},
        },
    )

    with pytest.raises(RunConfigError, match=r"engine.*max_seq_lenght"):
        load_run_config(config_path)


def test_exactly_one_prompt_source_is_required(tmp_path):
    config_path = tmp_path / "bad-input.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "prompt": "one",
            "prompts": ["two"],
        },
    )

    with pytest.raises(RunConfigError, match="exactly one"):
        load_run_config(config_path)


def test_execute_single_prompt_uses_generate_and_stdout(tmp_path):
    config_path = tmp_path / "single.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "engine": {
                "device": "cpu",
                "dtype": "fp32",
                "deterministic": True,
                "seed": 7,
            },
            "generation": {
                "max_new_tokens": 12,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
            },
            "prompt": "hello",
            "output": {"format": "text", "include_prompt": False},
        },
    )
    stdout = StringIO()

    result = execute_run_config(
        config_path,
        engine_factory=_FakeEngine,
        stdout=stdout,
    )

    engine = _FakeEngine.last_instance
    assert engine.model == "example/model"
    assert engine.kwargs["dtype"] is torch.float32
    assert engine.kwargs["device"] == "cpu"
    assert engine.kwargs["deterministic"] is True
    assert engine.generate_calls[0][0] == "hello"
    assert engine.generate_calls[0][1]["max_new_tokens"] == 12
    assert not engine.batch_calls
    assert result.outputs == ("generated: hello",)
    assert stdout.getvalue() == "generated: hello\n"


def test_execute_prompt_batch_uses_continuous_batch_api_and_jsonl(tmp_path):
    config_path = tmp_path / "batch.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "engine": {"max_batch_size": 8},
            "generation": {
                "max_new_tokens": 24,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "prompts": ["first", "second"],
            "output": {"format": "jsonl", "path": "outputs/results.jsonl"},
        },
    )

    result = execute_run_config(config_path, engine_factory=_FakeEngine)

    engine = _FakeEngine.last_instance
    assert not engine.generate_calls
    assert engine.batch_calls[0][0] == ["first", "second"]
    assert engine.batch_calls[0][1]["ignore_eos"] is True
    assert result.output_path == (tmp_path / "outputs" / "results.jsonl").resolve()
    rows = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["index"] for row in rows] == [0, 1]
    assert [row["prompt"] for row in rows] == ["first", "second"]
    assert [row["output"] for row in rows] == [
        "generated: first",
        "generated: second",
    ]


def test_prompts_file_is_relative_to_configuration(tmp_path):
    prompts_path = tmp_path / "inputs" / "prompts.txt"
    prompts_path.parent.mkdir()
    prompts_path.write_text("one\n\n two \n", encoding="utf-8")
    config_path = tmp_path / "file-input.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "prompts_file": "inputs/prompts.txt",
        },
    )

    config = load_run_config(config_path)

    assert config.prompts_file == prompts_path.resolve()
    assert load_prompts(config) == ("one", "two")


def test_batch_rejects_single_only_repetition_penalty(tmp_path):
    config_path = tmp_path / "bad-batch.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "prompts": ["one", "two"],
            "generation": {"repetition_penalty": 1.2},
        },
    )
    config = load_run_config(config_path)

    with pytest.raises(RunConfigError, match="generate_batch"):
        validate_run_config(config)


def test_yaml_config_when_optional_parser_is_available(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        """\
version: 1
task: generate
model: example/model
engine:
  dtype: bf16
generation:
  temperature: 0.0
prompt: hello from yaml
""",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.engine.dtype == "bf16"
    assert config.generation.temperature == 0.0
    assert config.prompt == "hello from yaml"


def test_cli_dry_run_validates_without_loading_model(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "dry-run.json"
    _write_json(
        config_path,
        {
            "model": "example/model",
            "prompt": "hello",
            "output": {"format": "json"},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["megagemm", "run", str(config_path), "--dry-run"],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["input"]["prompt_count"] == 1
    assert payload["note"].startswith("configuration validated")
