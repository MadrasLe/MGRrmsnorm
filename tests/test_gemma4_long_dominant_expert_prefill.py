import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "qwen3_moe.py"
BENCHMARK = (
    ROOT
    / "benchmarks"
    / "run_gemma4_long_dominant_expert_prefill_microbench.py"
)
WRAPPER = (
    ROOT / "benchmarks" / "run_gemma4_long_dominant_expert_prefill_colab.sh"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "gemma4_long_dominant_expert_prefill_bench",
        BENCHMARK,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load dominant-expert benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gemma4LongDominantExpertPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = KERNEL.read_text(encoding="utf-8")
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")

    def test_rejected_bf16_partial_branch_is_gone(self):
        self.assertNotIn("_resolve_segmented_prefill_partial_dtype", self.kernel)
        self.assertNotIn("partial_dtype:", self.kernel)
        self.assertIn('dtype=torch.float32,', self.kernel)
        self.assertFalse(
            (
                ROOT
                / "benchmarks"
                / "run_gemma4_long_bf16_partial_prefill_microbench.py"
            ).exists()
        )

    def test_kernel_isolated_dominant_expert_and_fixed_order_reduce(self):
        self.assertIn(
            "def qwen3_moe_dominant_expert_padded_bmm_prefill(",
            self.kernel,
        )
        self.assertIn(
            "def _qwen3_moe_dominant_padded_bmm_route_pack_kernel(",
            self.kernel,
        )
        self.assertIn(
            "def _qwen3_moe_dominant_padded_bmm_reduce_kernel(",
            self.kernel,
        )
        self.assertIn("heavy_gate_up = torch.mm(", self.kernel)
        self.assertIn("heavy_projected = torch.bmm(", self.kernel)
        self.assertIn("light_projected = torch.bmm(", self.kernel)
        self.assertIn(
            'workspace["dominant_padded_bmm_deterministic_reduce"] = 1',
            self.kernel,
        )

    def test_heavy_atomic_uses_block_pointer_and_block_increment(self):
        self.assertIn(
            "counter_offsets = tl.zeros((BLOCK,), dtype=tl.int32)",
            self.kernel,
        )
        self.assertIn(
            "increments = tl.full((BLOCK,), 1, dtype=tl.int32)",
            self.kernel,
        )
        self.assertIn("heavy_counter_ptr + counter_offsets", self.kernel)
        self.assertIn('sem="relaxed"', self.kernel)

    def test_checkpoint_like_routes_have_exact_counts_and_unique_topk(self):
        module = _load_benchmark_module()
        expected = (
            (15_648, 1_940, 1_941),
            (22_528, 1_886, 1_887),
            (31_296, 1_817, 1_818),
        )
        for heavy_count, expected_min, expected_light_max in expected:
            selected = module._build_skewed_route(
                rows=32_768,
                top_k=8,
                num_experts=128,
                heavy_count=heavy_count,
                device=torch.device("cpu"),
            )
            counts = torch.bincount(selected.reshape(-1), minlength=128)
            sorted_rows = torch.sort(selected, dim=1).values
            self.assertEqual(int(counts[0]), heavy_count)
            self.assertEqual(int(counts.min()), expected_min)
            self.assertEqual(int(counts[1:].max()), expected_light_max)
            self.assertTrue(
                bool(torch.all(sorted_rows[:, 1:] != sorted_rows[:, :-1]))
            )

    def test_gate_has_one_candidate_and_requires_closing_real_gap(self):
        self.assertIn("target_prefill_gap_ms = 46.04", self.benchmark)
        self.assertIn("--minimum-profile-speedup", self.benchmark)
        self.assertIn("--minimum-aggregate-speedup", self.benchmark)
        self.assertIn('"closes_gap": closes_gap', self.benchmark)
        self.assertIn('"APPLY_HYBRID"', self.benchmark)
        self.assertNotIn("qwen3_moe_padded_bmm_prefill,", self.benchmark)

    def test_tiny_full_candidate_preflight_precedes_large_allocations(self):
        self.assertIn("def _run_candidate_preflight(", self.benchmark)
        self.assertIn("DOMINANT_HYBRID_PREFLIGHT", self.benchmark)
        self.assertLess(
            self.benchmark.index("_run_candidate_preflight(device)"),
            self.benchmark.index("hidden = random_weight((rows, hidden_dim))"),
        )
        self.assertIn(
            "gemma4-long-dominant-expert-prefill-v2-vector-atomic-preflight",
            self.wrapper,
        )

    def test_wrapper_is_fresh_vm_model_free_and_bounded(self):
        self.assertIn("fresh_vm: supported", self.wrapper)
        self.assertIn("model_download: disabled", self.wrapper)
        self.assertIn("vllm_install: disabled", self.wrapper)
        self.assertIn("package_install: disabled", self.wrapper)
        self.assertIn("timeout --foreground", self.wrapper)
        self.assertNotIn("snapshot_download", self.wrapper)
        self.assertNotIn("RESUME", self.wrapper)
        self.assertNotIn("pip install", self.wrapper)


if __name__ == "__main__":
    unittest.main()
