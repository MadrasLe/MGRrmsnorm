import re
import unittest
from pathlib import Path

import torch

from megagemm.kernels.paged_attention import (
    gemma4_long_full_prefill_attention,
)


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "paged_attention.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_full_prefill_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_full_prefill_colab.sh"


class Gemma4LongFullPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel_source = KERNEL.read_text(encoding="utf-8")
        cls.model_source = MODEL.read_text(encoding="utf-8")
        cls.benchmark_source = BENCHMARK.read_text(encoding="utf-8")
        cls.harness_source = HARNESS.read_text(encoding="utf-8")

    def test_cpu_or_ineligible_shape_returns_none(self):
        q = torch.empty(1, 16, 2, 512, dtype=torch.bfloat16)
        k = torch.empty(1, 2, 2, 512, dtype=torch.bfloat16)
        v = torch.empty_like(k)
        result = gemma4_long_full_prefill_attention(
            q,
            k,
            v,
            scale=1.0,
            force=True,
        )
        self.assertIsNone(result)

    def test_runtime_candidate_is_exact_shape_and_promoted_by_default(self):
        self.assertIn(
            '"MEGAGEMM_GEMMA4_LONG_FULL_PREFILL", default=True',
            self.model_source,
        )
        self.assertIn(
            "if implicit_causal and _GEMMA4_LONG_FULL_PREFILL:",
            self.model_source,
        )
        for expected in (
            "block_n: int = 32",
            "num_q_heads != 16",
            "batch_size not in (8, 16)",
            "seq_len != 2048",
            "head_dim != 512",
            "(batch_size, 2, seq_len, head_dim)",
        ):
            self.assertIn(expected, self.kernel_source)

    def test_full_causal_launcher_reuses_validated_online_softmax_kernel(self):
        function_start = self.kernel_source.index(
            "def gemma4_long_full_prefill_attention("
        )
        function_end = self.kernel_source.index(
            "\ndef prefill_attention(",
            function_start,
        )
        function = self.kernel_source[function_start:function_end]
        self.assertIn("_gemma4_long_sliding_prefill_kernel[grid]", function)
        self.assertIn("SLIDING_WINDOW=seq_len", function)
        self.assertIn("MAX_WINDOW_TILES=max_key_tiles", function)
        self.assertIn("scale=self.scale", self.model_source)

    def test_gate_autotunes_four_configs_in_one_run(self):
        for expected in (
            '"triton_bn32_w8"',
            '"triton_bn64_w8"',
            '"triton_bn128_w8"',
            '"triton_bn64_w4"',
            "batch_size = 8",
            "seq_len = 2048",
            "kv_heads = 2",
            "head_dim = 512",
            "full_layers = 5",
            "b16_chunks = 2",
            "repeat_exact",
            "baseline_stability_ratio <= 1.05",
        ):
            self.assertIn(expected, self.benchmark_source)
        self.assertNotIn(
            "return 0 if valid_candidates else 2",
            self.benchmark_source,
        )
        self.assertRegex(self.benchmark_source, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-full-prefill-v1",
            self.harness_source,
        )
        self.assertIn("model_download: disabled", self.harness_source)
        self.assertIn("vllm_install: disabled", self.harness_source)
        self.assertIn("package_install: disabled", self.harness_source)
        self.assertIn(
            'BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"',
            self.harness_source,
        )
        self.assertNotIn("pip install", self.harness_source)
        self.assertNotIn("snapshot_download", self.harness_source)
        self.assertNotIn("run_gemma4_long_context_vs_vllm", self.harness_source)

    def test_embedded_python_compiles(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.harness_source,
            re.DOTALL,
        )
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], "long_full_prefill_harness_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
