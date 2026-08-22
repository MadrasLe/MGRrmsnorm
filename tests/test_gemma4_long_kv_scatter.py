import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "paged_attention.py"
KV_CACHE = ROOT / "megagemm" / "engine" / "kv_cache.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_kv_scatter_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_kv_scatter_colab.sh"


class Gemma4LongKvScatterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text(encoding="utf-8")
        cls.kv_cache = KV_CACHE.read_text(encoding="utf-8")
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")

    def test_candidate_is_promoted_only_through_the_gemma_long_policy(self):
        self.assertIn("def paged_kv_cache_scatter_token_tiled(", self.kernel)
        runtime = self.kv_cache.split("def write_kv_prefill_packed(", 1)[1].split(
            "def prefill_kv_scatter_stats(", 1
        )[0]
        self.assertIn("_paged_kv_cache_scatter(", runtime)
        self.assertIn("_paged_kv_cache_scatter_token_tiled(", runtime)
        model = (ROOT / "megagemm" / "models" / "llama.py").read_text(
            encoding="utf-8"
        )
        policy = model.split(
            "def _gemma4_a100_a4b_long_kv_scatter_tokens_per_program(", 1
        )[1].split("def _gemma4_a100_a4b_fused_router_prefill_shape(", 1)[0]
        for expected in (
            "int(batch_size) in (8, 16)",
            "int(seq_len) == 2048",
            "kv_shape == (8, 256)",
            "return 4",
            "kv_shape == (2, 512)",
            "return 2",
        ):
            self.assertIn(expected, policy)
        self.assertIn("tokens_per_program=tokens_per_program", model)

    def test_gate_uses_the_real_long_paged_layout(self):
        for expected in (
            "batch_size = 8",
            "context = 2_048",
            "rows = batch_size * context",
            "block_size = 16",
            "physical_blocks = seq_idx * blocks_per_seq",
            'attention_type="sliding"',
            "kv_heads=8",
            "head_dim=256",
            "layer_invocations=25 * 2",
            'attention_type="full"',
            "kv_heads=2",
            "head_dim=512",
            "layer_invocations=5 * 2",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_compares_current_and_bounded_token_tiles(self):
        for expected in (
            'profile_case("current_bt1"',
            'profile_case("tiled_bt2"',
            'profile_case("tiled_bt4"',
            'profile_case("tiled_bt8"',
            'profile_case("current_recheck"',
            "tokens_per_program=tokens_per_program",
        ):
            self.assertIn(expected, self.benchmark)

    def test_promotion_requires_exact_stable_speedup(self):
        for expected in (
            "torch.equal(reference, cache)",
            "torch.equal(first, cache)",
            "max_abs_error == 0.0",
            'baseline = by_name["current_recheck"]',
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "speedup >= float(args.minimum_speedup)",
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn("harness_rev: gemma4-long-kv-scatter-v1", self.harness)
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
        compile(blocks[0], "long_kv_scatter_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
