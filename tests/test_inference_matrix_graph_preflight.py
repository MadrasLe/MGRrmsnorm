import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"


def load_matrix():
    name = "inference_matrix_graph_preflight"
    spec = importlib.util.spec_from_file_location(name, MATRIX)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inference matrix")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(str(text).split())))

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"t{token_id}" for token_id in token_ids)


class FakeScheduler:
    def __init__(self, completed, graph_stats):
        self._completed = completed
        self._graph_stats = graph_stats

    def get_stats(self):
        if self._graph_stats is None:
            return {}
        return {"decode_cuda_graphs": self._graph_stats}


class FakeEngine:
    def __init__(self, *, mismatch=False):
        self._last_scheduler = None
        self._mismatch = mismatch

    def generate_batch(self, prompts, *, max_new_tokens, **kwargs):
        del kwargs
        graph_enabled = os.environ.get("MEGAGEMM_DECODE_CUDA_GRAPHS") == "1"
        completed = []
        for index, _prompt in enumerate(prompts, start=1):
            tokens = list(range(max_new_tokens))
            if graph_enabled and self._mismatch:
                tokens[-1] += 1
            completed.append(
                SimpleNamespace(
                    request_id=index,
                    generated_ids=tokens,
                )
            )
        graph_stats = None
        if graph_enabled:
            graph_stats = {
                "enabled": True,
                "captures": 1,
                "replays": 5,
                "failures": 0,
            }
        self._last_scheduler = FakeScheduler(completed, graph_stats)
        return [""] * len(prompts)


def test_decode_graph_preflight_requires_exact_tokens_and_real_replays():
    matrix = load_matrix()
    engine = FakeEngine()

    report = matrix.verify_megagemm_decode_graph(
        engine,
        FakeTokenizer(),
        batch_sizes=[1, 8],
    )

    assert report["status"] == "passed"
    assert [case["batch_size"] for case in report["cases"]] == [1, 8]
    assert all(case["token_exact"] for case in report["cases"])
    assert all(case["replays"] == 5 for case in report["cases"])


def test_decode_graph_preflight_rejects_token_mismatch():
    matrix = load_matrix()
    engine = FakeEngine(mismatch=True)

    with pytest.raises(RuntimeError, match="token mismatch"):
        matrix.verify_megagemm_decode_graph(
            engine,
            FakeTokenizer(),
            batch_sizes=[1],
        )

    assert engine._megagemm_decode_graph_preflight["status"] == "failed"
