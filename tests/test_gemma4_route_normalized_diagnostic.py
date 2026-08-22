import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
LONG_RUNNER = ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm.py"
BATCH_RUNNER = ROOT / "benchmarks" / "run_gemma4_moe_batch_vs_vllm.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RouteNormalizedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.long_runner = load_module("gemma4_long_route_test", LONG_RUNNER)
        cls.batch_runner = load_module("gemma4_batch_route_test", BATCH_RUNNER)

    def test_forced_token_is_applied_after_real_token_tensor_exists(self):
        import megagemm.models.llama as llama_model

        previous = llama_model._BENCHMARK_FORCED_TOKEN_ID
        try:
            llama_model._BENCHMARK_FORCED_TOKEN_ID = 7
            natural = torch.tensor([2, 3], dtype=torch.long)
            result = llama_model._apply_benchmark_forced_token(natural, 10)
            self.assertIs(result, natural)
            self.assertEqual(result.tolist(), [7, 7])
            with self.assertRaises(ValueError):
                llama_model._apply_benchmark_forced_token(natural, 7)
        finally:
            llama_model._BENCHMARK_FORCED_TOKEN_ID = previous

        source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
            encoding="utf-8"
        )
        decode_step = source[source.index("def decode_step("):source.index("def decode_multi_step(")]
        self.assertLess(
            decode_step.index("self._decode_next_token_greedy(hidden)"),
            decode_step.index("_apply_benchmark_forced_token("),
        )

    def test_forced_token_context_restores_process_state(self):
        import megagemm.models.llama as llama_model

        env_name = self.long_runner.BENCHMARK_FORCED_TOKEN_ENV
        previous_env = os.environ.get(env_name)
        previous_module = llama_model._BENCHMARK_FORCED_TOKEN_ID
        os.environ.pop(env_name, None)
        llama_model._BENCHMARK_FORCED_TOKEN_ID = -1
        try:
            with self.long_runner.megagemm_benchmark_forced_token(19):
                self.assertEqual(os.environ[env_name], "19")
                self.assertEqual(llama_model._BENCHMARK_FORCED_TOKEN_ID, 19)
            self.assertNotIn(env_name, os.environ)
            self.assertEqual(llama_model._BENCHMARK_FORCED_TOKEN_ID, -1)
        finally:
            llama_model._BENCHMARK_FORCED_TOKEN_ID = previous_module
            if previous_env is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_env

    def test_route_token_and_matrix_contract_are_deterministic(self):
        prompts = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
        self.assertEqual(self.long_runner.route_normalized_token_id(prompts), 12)
        contract = self.long_runner.forced_token_matrix_contract(
            [[12, 12, 12], [12, 12, 12]],
            token_id=12,
            expected_rows=2,
            expected_tokens=3,
        )
        self.assertTrue(contract["exact"])
        bad = self.long_runner.forced_token_matrix_contract(
            [[12, 12, 9], [12, 12, 12]],
            token_id=12,
            expected_rows=2,
            expected_tokens=3,
        )
        self.assertFalse(bad["exact"])

    def test_vllm_request_uses_native_allowed_token_ids(self):
        captured = {}

        class SamplingParams:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class Completion:
            def __init__(self):
                self.token_ids = [17, 17, 17]

        class RequestOutput:
            def __init__(self, prompt):
                self.prompt_token_ids = list(prompt)
                self.outputs = [Completion()]

        class FakeLLM:
            def generate(self, prompts, params, use_tqdm=False):
                del params, use_tqdm
                return [
                    RequestOutput(item["prompt_token_ids"])
                    for item in prompts
                ]

        fake_vllm = types.SimpleNamespace(SamplingParams=SamplingParams)
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            row = self.batch_runner.run_vllm_request(
                FakeLLM(),
                [[1, 2], [3, 4]],
                3,
                allowed_token_id=17,
            )
        self.assertEqual(captured["allowed_token_ids"], [17])
        self.assertEqual(row["allowed_token_id"], 17)
        self.assertEqual(row["token_ids"], [[17, 17, 17], [17, 17, 17]])

    def test_decode_stage_breakdown_avoids_flat_aggregate_double_counting(self):
        breakdown = self.long_runner.decode_stage_breakdown(
            {
                "decode_path": "flat",
                "steps": 8,
                "batch_size": 16,
                "total_ms": 80.0,
                "decode_body_ms": 64.0,
                "flat_moe_ms": 40.0,
                "flat_attn_core_ms": 16.0,
                "lm_head_ms": 8.0,
                "sample_ms": 1.0,
            }
        )
        self.assertEqual(breakdown["leaf_stage_ranking"][0]["stage"], "flat_moe")
        self.assertNotIn(
            "decode_body",
            [row["stage"] for row in breakdown["leaf_stage_ranking"]],
        )
        self.assertAlmostEqual(breakdown["ms_per_step"], 10.0)

    def test_compare_classifies_identical_route_normalized_stream(self):
        runner = self.long_runner
        common = {
            "schema_version": 1,
            "status": "complete",
            "model": "test",
            "dtype": "bf16",
            "contexts": [2048],
            "batch_sizes": [16],
            "max_seq_len": 2112,
            "max_tokens": 4,
            "vllm_max_num_batched_tokens": 32768,
            "prompt_contracts": {"2048": {"sha256": "same"}},
            "gpu": {"name": "test GPU"},
            "route_normalized_diagnostic": True,
            "route_normalized_repeats": 3,
            "route_normalized_policy": runner.ROUTE_NORMALIZED_POLICY,
        }

        def summary(prefill_ms, decode_ms, decode_tok_s, total_tok_s):
            return {
                "prefill_ms_median": prefill_ms,
                "decode_ms_median": decode_ms,
                "decode_tok_s_median": decode_tok_s,
                "output_tok_s_total_median": total_tok_s,
            }

        def route_case(prefill_ms, decode_ms, decode_tok_s, total_tok_s):
            return {
                "enabled": True,
                "forced_token_id": 42,
                "summary": summary(
                    prefill_ms, decode_ms, decode_tok_s, total_tok_s
                ),
                "samples": [{"token_ids": [[42, 42, 42, 42]] * 16}],
            }

        def case(natural_tokens, natural_summary, route):
            return {
                "batch_size": 16,
                "context": 2048,
                "prompt_contract": {"sha256": "same-case"},
                "summary": natural_summary,
                "samples": [{"token_ids": [natural_tokens] * 16}],
                "route_normalized_diagnostic": route,
            }

        megagemm = {
            **common,
            "backend": "megagemm",
            "cases": {
                "b16_c2048": case(
                    [1, 2, 3, 4],
                    summary(1800.0, 120.0, 800.0, 300.0),
                    route_case(1800.0, 30.0, 1600.0, 400.0),
                )
            },
        }
        vllm = {
            **common,
            "backend": "vllm",
            "version": "test",
            "cases": {
                "b16_c2048": case(
                    [1, 9, 9, 9],
                    summary(1700.0, 100.0, 900.0, 320.0),
                    route_case(1700.0, 24.0, 2000.0, 450.0),
                )
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mg_path = root / "megagemm.json"
            vl_path = root / "vllm.json"
            out_path = root / "comparison.json"
            mg_path.write_text(json.dumps(megagemm), encoding="utf-8")
            vl_path.write_text(json.dumps(vllm), encoding="utf-8")
            result = runner.compare_results(mg_path, vl_path, out_path)

        route = result["cases"]["b16_c2048"]["route_normalized"]
        self.assertFalse(result["all_tokens_exact"])
        self.assertTrue(result["route_normalized_all_tokens_exact"])
        self.assertEqual(result["result_class"], "ROUTE_NORMALIZED_PERFORMANCE_VALID")
        self.assertAlmostEqual(route["decode_throughput_ratio"], 0.8)
        self.assertAlmostEqual(
            route["megagemm_minus_vllm_decode_us_per_step"], 2000.0
        )


if __name__ == "__main__":
    unittest.main()
