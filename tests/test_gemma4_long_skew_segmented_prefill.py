import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "benchmarks" / "run_gemma4_long_skew_segmented_prefill_microbench.py"
)
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm_colab.sh"
MODEL = ROOT / "megagemm" / "models" / "llama.py"


class Gemma4LongSkewSegmentedPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")
        cls.model = MODEL.read_text(encoding="utf-8")

    def test_benchmark_compiles_and_has_no_paid_model_work(self):
        compile(self.benchmark, str(BENCHMARK), "exec")
        self.assertNotIn("snapshot_download", self.benchmark)
        self.assertNotIn("InferenceEngine", self.benchmark)
        self.assertNotIn("import vllm", self.benchmark.lower())
        self.assertNotIn("pip install", self.benchmark.lower())
        self.assertIn('print("  model_download: disabled")', self.benchmark)
        self.assertIn('print("  vllm_install: disabled")', self.benchmark)

    def test_gate_matches_the_real_b16_c2048_chunk_and_moe_shape(self):
        for expected in (
            "ROWS = 32_768",
            "HIDDEN_DIM = 2_816",
            "INTERMEDIATE_DIM = 704",
            "NUM_EXPERTS = 128",
            "TOP_K = 8",
            "LAYER_INVOCATIONS = 30",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_covers_the_observed_checkpoint_skew_frontier(self):
        for expected in (
            '("skew_7p64x", 15_648)',
            '("skew_11x", 22_528)',
            '("skew_15p28x", 31_296)',
            "global_padding_ratio",
            "selected[:heavy_count, 0] = 0",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_changes_only_the_partial_layout(self):
        for expected in (
            '"block_m": 64',
            '"block_n": 256',
            '"block_k": 64',
            '"fused_gate_block_n": 128',
            '"async_tiles_max_assignments": ROWS * TOP_K',
            '"sorted_partial": False',
            '("sorted_contiguous_partial", {"sorted_partial": True})',
            '"current_recheck"',
            'f"{candidate_name}_recheck"',
        ):
            self.assertIn(expected, self.benchmark)
        self.assertNotIn('"block_m": 128', self.benchmark)

    def test_promotion_requires_exact_deterministic_production_contract(self):
        for expected in (
            'workspace.get("segmented_prefill_deterministic_reduce", 0)',
            'workspace.get("segmented_prefill_partial_reduce", 0)',
            'workspace.get("segmented_prefill_partial_dtype") == "torch.float32"',
            'workspace.get("segmented_prefill_route_scatter", 0)',
            'workspace.get("segmented_prefill_async_tiles", 0)',
            'workspace.get("segmented_prefill_max_tiles", 0)',
            'workspace.get("segmented_prefill_sorted_partial", 0)',
            'workspace.get("segmented_prefill_slot_inverse_bytes", 0)',
            "sorted_partial_expected",
            "and exact",
            "repeat_exact",
            "cosine >= 0.9999",
            "max_abs_error <= 0.125",
            "float(args.minimum_speedup)",
            "float(args.minimum_profile_speedup)",
            "float(args.maximum_candidate_drift)",
            "candidate_drift",
        ):
            self.assertIn(expected, self.benchmark)

        self.assertIn(
            "candidate_drift <= float(args.maximum_candidate_drift)",
            self.benchmark,
        )
        self.assertNotIn(
            "baseline_drift <= float(args.maximum_candidate_drift)",
            self.benchmark,
        )
        self.assertIn("baseline_us = min(cold_us, settled_us)", self.benchmark)
        self.assertIn(
            "candidate_us = max(candidate_first_us, candidate_recheck_us)",
            self.benchmark,
        )

    def test_harness_runs_gate_before_paid_work_and_keeps_tile_shape_fixed(self):
        gate = self.harness.index("== CHECKPOINT-FREE LONG SORTED-PARTIAL GATE ==")
        download = self.harness.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        self.assertLess(gate, download)
        self.assertIn(
            "python benchmarks/run_gemma4_long_skew_segmented_prefill_microbench.py",
            self.harness,
        )
        self.assertIn("SORTED_PARTIAL_GATE_DECISION", self.harness)
        self.assertIn("STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED", self.harness)
        self.assertIn(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_ASYNC_TILES_MAX_ASSIGNMENTS",
            self.harness,
        )
        self.assertIn(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL",
            self.harness,
        )
        self.assertIn('RUN_LONG_SORTED_PARTIAL_GATE="${RUN_LONG_SORTED_PARTIAL_GATE:-0}"', self.harness)
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL=1",
            self.harness,
        )
        self.assertIn("using the exact v30 promotion", self.harness)
        for name, value in (
            ("BLOCK_M", "64"),
            ("BLOCK_N", "256"),
            ("BLOCK_K", "64"),
            ("FUSED_GATE_BLOCK_N", "128"),
            ("NUM_WARPS", "4"),
            ("NUM_STAGES", "3"),
        ):
            self.assertIn(
                f"export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_{name}={value}",
                self.harness,
            )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL=0",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL=1",
            self.harness,
        )
        self.assertIn(
            "dominant expert hybrid, skew>=7.5x, light padding<=1.25x",
            self.harness,
        )
        self.assertIn(
            '"MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL"',
            self.model,
        )
        sorted_declaration = self.model.split(
            '"sorted_partial": _env_enabled(',
            1,
        )[1].split("),", 1)[0]
        self.assertIn("default=True", sorted_declaration)

    def test_global_padded_bmm_is_retired_by_default(self):
        declaration = self.model.split(
            "_GEMMA4_A4B_LONG_PADDED_BMM_PREFILL = _env_enabled(",
            1,
        )[1].split(")", 1)[0]
        self.assertIn('"MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL"', declaration)
        self.assertIn("default=False", declaration)
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL=0",
            self.harness,
        )


if __name__ == "__main__":
    unittest.main()
