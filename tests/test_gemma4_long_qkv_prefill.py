import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_qkv_prefill_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_qkv_prefill_colab.sh"


class Gemma4LongQkvPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")

    def test_gate_promotes_only_the_measured_long_sliding_shape(self):
        policy = self.model.split(
            "def _gemma4_a100_a4b_fused_qkv_prefill_shape(", 1
        )[1].split("def _gemma4_a100_a4b_fused_attn_prepare_shape(", 1)[0]
        self.assertIn("row_count in (200, 400)", policy)
        self.assertIn("row_count in (16_384, 32_768)", policy)
        self.assertIn("qkv_shape == (4096, 2048, 2048)", policy)
        long_policy = policy.split("row_count in (16_384, 32_768)", maxsplit=1)[1]
        self.assertNotIn("qkv_shape == (8192, 1024, 1024)", long_policy)

    def test_gate_covers_both_real_long_qkv_shapes(self):
        for expected in (
            "rows = 16_384",
            "hidden = 2_816",
            'attention_type="sliding"',
            "q_size=4_096",
            "k_size=2_048",
            "v_size=2_048",
            "layer_invocations=25 * 2",
            'attention_type="full"',
            "q_size=8_192",
            "k_size=1_024",
            "v_size=1_024",
            "layer_invocations=5 * 2",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_compares_all_useful_packing_counts(self):
        for expected in (
            '"current_three_gemms"',
            '"q_plus_fused_kv"',
            '"fused_qkv"',
            '"current_recheck"',
            "torch.mm(x, q_weight.t(), out=q_out)",
            "torch.mm(x, kv_weight.t(), out=kv_out)",
            "torch.mm(x, qkv_weight.t(), out=qkv_out)",
        ):
            self.assertIn(expected, self.benchmark)

    def test_promotion_requires_correct_stable_speedup(self):
        for expected in (
            "repeat_exact",
            'error["cosine"] >= 0.9999',
            'error["max_abs_error"] <= 0.03125',
            'baseline = by_name["current_recheck"]',
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "speedup >= float(args.minimum_speedup)",
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn("harness_rev: gemma4-long-qkv-prefill-v1", self.harness)
        self.assertIn('BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"', self.harness)
        self.assertIn("model_download: disabled", self.harness)
        self.assertIn("vllm_install: disabled", self.harness)
        self.assertIn("package_install: disabled", self.harness)
        self.assertNotIn("pip install", self.harness)
        self.assertNotIn("snapshot_download", self.harness)
        self.assertNotIn("run_gemma4_long_context_vs_vllm", self.harness)

    def test_embedded_python_compiles(self):
        blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", self.harness, re.DOTALL)
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], "long_qkv_prefill_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
