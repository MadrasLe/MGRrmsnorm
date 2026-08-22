import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "benchmarks" / "run_gemma4_long_routed_expert_prefill_microbench.py"
)
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_routed_expert_prefill_colab.sh"
KERNEL = ROOT / "megagemm" / "kernels" / "qwen3_moe.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"


class Gemma4LongRoutedExpertPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")
        cls.kernel = KERNEL.read_text(encoding="utf-8")
        cls.model = MODEL.read_text(encoding="utf-8")

    def test_gate_uses_exact_long_chunk_and_model_shape(self):
        for expected in (
            "rows = 16_384",
            "hidden_dim = 2_816",
            "intermediate_dim = 704",
            "num_experts = 128",
            "top_k = 8",
            "layers = 30",
            "chunks = 2",
        ):
            self.assertIn(expected, self.benchmark)

    def test_current_case_matches_measured_production_path(self):
        self.assertIn("fused_activation=True", self.benchmark)
        self.assertIn("activation_block=512", self.benchmark)
        self.assertIn("reduce_block_n=256", self.benchmark)
        self.assertIn("reduce_num_warps=4", self.benchmark)
        self.assertIn('down_output_dtype="fp32"', self.model)
        self.assertIn(
            'route_pack="argsort" if deterministic_route_pack else "atomic"',
            self.model,
        )
        self.assertIn("route_pack_block=256", self.model)
        self.assertIn(
            "max_padding_ratio=_GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO",
            self.model,
        )
        self.assertIn("fused_activation=True", self.model)
        self.assertIn("activation_block=512", self.model)

    def test_gate_has_bounded_high_impact_candidates(self):
        for expected in (
            '"reduce_bn64_w4"',
            '"reduce_bn64_w8"',
            '"reduce_bn128_w4"',
            '"reduce_bn128_w8"',
            '"reduce_bn256_w8"',
            "qwen3_moe_padded_bmm_prefill",
            "fused_activation=True",
            '"current_recheck"',
        ):
            self.assertIn(expected, self.benchmark)
        self.assertNotIn("async_tiles_max_assignments=131_072", self.benchmark)

    def test_padded_bmm_candidate_uses_fixed_order_reduce(self):
        for expected in (
            "def qwen3_moe_padded_bmm_prefill(",
            "torch.bmm(padded_hidden",
            "slot_to_padded.scatter_(0, order, padded_offsets)",
            "def _qwen3_moe_padded_bmm_atomic_route_pack_kernel(",
            "tl.atomic_add(counters_ptr + experts, 1, mask=mask)",
            "def _qwen3_moe_padded_bmm_activation_kernel(",
            'workspace["padded_bmm_fused_activation"]',
            'workspace["padded_bmm_reduce_block_n"]',
            'workspace["padded_bmm_reduce_num_warps"]',
            "def _qwen3_moe_padded_bmm_reduce_kernel(",
            "values * route[:, None]",
            'workspace["padded_bmm_deterministic_reduce"] = 1',
            'raise RuntimeError(',
            '"Padded-BMM capacity guard: expert skew would expand "',
        ):
            self.assertIn(expected, self.kernel)
        helper = self.kernel.split("def qwen3_moe_padded_bmm_prefill(", 1)[1]
        helper = helper.split("def qwen3_moe_grouped_runtime_config", 1)[0]
        self.assertNotIn(".index_add_(", helper)

    def test_guarded_fp32_winner_is_promoted_only_for_measured_runtime_shape(self):
        for expected in (
            '"MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL"',
            "_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS, self.hidden_dim",
            "(_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS, 8)",
            "hidden_states.dtype == torch.bfloat16",
            'return "A100" in torch.cuda.get_device_name',
            'down_output_dtype="fp32"',
            "align_m=16",
            'route_pack="argsort" if deterministic_route_pack else "atomic"',
            "route_pack_block=256",
            "_GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO",
            "fused_activation=True",
            "activation_block=512",
        ):
            self.assertIn(expected, self.model)

    def test_promoted_path_falls_back_and_reports_runtime_status(self):
        for expected in (
            "_gemma4_long_padded_bmm_prefill_disabled = True",
            "_gemma4_long_padded_bmm_prefill_fail_reason = str(exc)",
            '"gemma4_long_padded_bmm_prefill_hits"',
            '"gemma4_long_padded_bmm_prefill_assignments"',
            '"gemma4_long_padded_bmm_prefill_disabled_layers"',
            '"gemma4_long_padded_bmm_prefill_first_failure"',
            '"gemma4_long_padded_bmm_prefill_failures"',
        ):
            self.assertIn(expected, self.model)

    def test_promotion_requires_correct_stable_speedup(self):
        for expected in (
            "repeat_exact",
            "cosine >= 0.9999",
            "max_abs_error <= 0.125",
            'stable_baseline = by_name["current_recheck"]',
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "speedup >= float(args.minimum_speedup)",
            "near_tied_candidates",
            'float(row["sample_spread_ratio"])',
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-routed-expert-prefill-v8-fixed-reduce-tune",
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
        compile(blocks[0], "long_routed_expert_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
