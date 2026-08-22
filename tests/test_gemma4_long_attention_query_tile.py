import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "paged_attention.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_attention_query_tile_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_attention_query_tile_colab.sh"


class Gemma4LongAttentionQueryTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text(encoding="utf-8")
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")

    def test_runtime_defaults_are_the_a100_promoted_configs(self):
        sliding = self.kernel.split(
            "def gemma4_long_sliding_prefill_attention(", 1
        )[1].split("def gemma4_long_full_prefill_attention(", 1)[0]
        full = self.kernel.split(
            "def gemma4_long_full_prefill_attention(", 1
        )[1].split("def prefill_attention(", 1)[0]
        self.assertIn("block_m: int = 64", sliding)
        self.assertIn("num_warps: int = 8", sliding)
        self.assertIn("block_m: int = 32", full)
        self.assertIn("num_warps: int = 4", full)
        self.assertIn("BLOCK_M=block_m", sliding)
        self.assertIn("BLOCK_M=block_m", full)

    def test_gate_covers_both_real_long_attention_shapes(self):
        for expected in (
            'attention_type="sliding"',
            "kv_heads=8",
            "head_dim=256",
            "layer_invocations=25 * b16_chunks",
            'attention_type="full"',
            "kv_heads=2",
            "head_dim=512",
            "layer_invocations=5 * b16_chunks",
            "batch_size = 8",
            "seq_len = 2_048",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_has_bounded_query_tile_candidates(self):
        for expected in (
            '"sliding_bm64_w4"',
            '"sliding_bm128_w8"',
            '"sliding_bm128_w4"',
            '"full_bm64_w4"',
            '"full_bm64_w8"',
            'profile_case("current_recheck", current_config)',
        ):
            self.assertIn(expected, self.benchmark)

    def test_promotion_requires_correct_stable_speedup(self):
        for expected in (
            "repeat_exact",
            "cosine >= 0.9999",
            "max_abs_error <= 0.125",
            'stable_baseline = by_name["current_recheck"]',
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "speedup >= float(args.minimum_speedup)",
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-attention-query-tile-v3-frontier",
            self.harness,
        )
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
        compile(blocks[0], "long_attention_query_tile_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
