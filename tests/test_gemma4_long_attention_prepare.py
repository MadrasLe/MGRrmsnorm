import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_attention_prepare_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_attention_prepare_colab.sh"


class Gemma4LongAttentionPrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")

    def test_runtime_promotes_only_the_two_measured_long_shapes(self):
        policy = self.model.split(
            "def _gemma4_a100_a4b_fused_attn_prepare_shape(", 1
        )[1].split(
            "def _gemma4_a100_a4b_long_kv_scatter_tokens_per_program(", 1
        )[0]
        for expected in (
            "batch in (8, 16)",
            "seq_len == 2048",
            "head_shape in ((16, 8, 256), (16, 2, 512))",
        ):
            self.assertIn(expected, policy)
        self.assertIn("gemma4_prefill_attention_prepare(", self.benchmark)

    def test_gate_covers_both_real_long_attention_shapes(self):
        for expected in (
            "batch_size = 8",
            "seq_len = 2_048",
            "q_heads = 16",
            'attention_type="sliding"',
            "kv_heads=8",
            "head_dim=256",
            "k_eq_v=False",
            "layer_invocations=25 * b16_chunks",
            'attention_type="full"',
            "kv_heads=2",
            "head_dim=512",
            "k_eq_v=True",
            "layer_invocations=5 * b16_chunks",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_compares_runtime_semantics_and_rechecks_baseline(self):
        for expected in (
            'profile_case("current_prepare", current)',
            'profile_case("fused_prepare", fused)',
            'profile_case("current_recheck", current)',
            'profile_case("fused_recheck", fused)',
            "rmsnorm_triton(q, q_weight, eps, False)",
            "rmsnorm_triton_no_weight(v, eps)",
            "apply_rotary_emb(",
            "k.transpose(1, 2).contiguous()",
            "v.transpose(1, 2).contiguous()",
        ):
            self.assertIn(expected, self.benchmark)

    def test_sliding_recheck_preconditions_both_paths_and_skips_full(self):
        for expected in (
            "for _ in range(args.precondition_pairs):",
            "current()",
            "fused()",
            "candidate_run_ratio",
            "conservative_baseline_us",
            "conservative_candidate_us",
            "conservative_speedup",
            'ONLY="${ONLY:-sliding}"',
            '--only "${ONLY}"',
            "--precondition-pairs 5",
        ):
            self.assertIn(expected, self.benchmark + self.harness)

    def test_promotion_requires_correct_repeat_exact_stable_speedup(self):
        for expected in (
            "repeat_exact",
            'error["cosine"] >= 0.9999',
            'error["max_abs_error"] <= 0.03125',
            'baseline = by_name["current_recheck"]',
            "baseline_run_ratio",
            "candidate_run_ratio",
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "conservative_speedup >= float(args.minimum_speedup)",
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-attention-prepare-v2-preconditioned",
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
        compile(blocks[0], "long_attention_prepare_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
