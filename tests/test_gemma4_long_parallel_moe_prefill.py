import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "benchmarks" / "run_gemma4_long_parallel_moe_prefill_microbench.py"
)
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_parallel_moe_prefill_colab.sh"


class Gemma4LongParallelMoePrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")

    def test_gate_uses_the_real_long_chunk_and_model_shape(self):
        for expected in (
            "rows = 16_384",
            "hidden_dim = 2_816",
            "shared_intermediate = 2_112",
            "expert_intermediate = 704",
            "num_experts = 128",
            "top_k = 8",
            "layers = 30",
            "chunks = 2",
        ):
            self.assertIn(expected, self.benchmark)

    def test_expert_branch_matches_the_production_deterministic_path(self):
        for expected in (
            "qwen3_moe_segmented_prefill(",
            "block_m=32",
            "block_n=128",
            "block_k=64",
            "fused_gate_block_n=64",
            "num_warps=4",
            "num_stages=3",
            "route_scatter=True",
            "async_tiles_max_assignments=4_096",
            "deterministic_reduce=True",
        ):
            self.assertIn(expected, self.benchmark)

    def test_parallel_path_has_an_explicit_fork_and_join(self):
        fork = self.benchmark.index("fork_event.record(main_stream)")
        side = self.benchmark.index("with torch.cuda.stream(side_stream):")
        expert = self.benchmark.index("expert = expert_branch()", side)
        join = self.benchmark.index("main_stream.wait_event(join_event)")
        add = self.benchmark.index("torch.add(shared, expert, out=combined_out)", join)
        self.assertLess(fork, side)
        self.assertLess(side, expert)
        self.assertLess(expert, join)
        self.assertLess(join, add)

    def test_gate_profiles_components_and_requires_stable_correct_speedup(self):
        for expected in (
            '"shared_only"',
            '"routed_experts_only"',
            '"sequential_recheck"',
            'correctness["repeat_exact"]',
            "baseline_stability_ratio <= 1.03",
            "speedup >= float(args.minimum_speedup)",
            "layer_invocations = layers * chunks",
            '"estimated_gap_coverage"',
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-parallel-moe-prefill-v1",
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
        compile(blocks[0], "long_parallel_moe_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
