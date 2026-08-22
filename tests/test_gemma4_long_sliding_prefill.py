import re
import unittest
from pathlib import Path

import torch

from megagemm.kernels.paged_attention import (
    gemma4_long_sliding_prefill_attention,
)


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "paged_attention.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = (
    ROOT / "benchmarks" / "run_gemma4_long_sliding_prefill_microbench.py"
)
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_sliding_prefill_colab.sh"


class Gemma4LongSlidingPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel_source = KERNEL.read_text(encoding="utf-8")
        cls.model_source = MODEL.read_text(encoding="utf-8")
        cls.benchmark_source = BENCHMARK.read_text(encoding="utf-8")
        cls.harness_source = HARNESS.read_text(encoding="utf-8")

    def test_cpu_or_ineligible_shape_falls_back_without_allocating_output(self):
        q = torch.empty(1, 16, 2, 256, dtype=torch.bfloat16)
        k = torch.empty(1, 8, 2, 256, dtype=torch.bfloat16)
        v = torch.empty_like(k)
        result = gemma4_long_sliding_prefill_attention(
            q,
            k,
            v,
            sliding_window=1024,
            scale=1.0,
            force=True,
        )
        self.assertIsNone(result)

    def test_runtime_candidate_is_narrow_and_promoted_by_default(self):
        self.assertIn(
            '"MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL", default=True',
            self.model_source,
        )
        self.assertIn(
            "if implicit_causal and _GEMMA4_LONG_SLIDING_PREFILL:",
            self.model_source,
        )
        self.assertIn("num_q_heads != 16", self.kernel_source)
        self.assertIn("batch_size not in (8, 16)", self.kernel_source)
        self.assertIn("seq_len != 2048", self.kernel_source)
        self.assertIn("head_dim != 256", self.kernel_source)
        self.assertIn("int(sliding_window) != 1024", self.kernel_source)
        self.assertIn('if "a100" not in _device_name_tokens(device_name):', self.kernel_source)

    def test_kernel_masks_both_causal_and_sliding_window_bounds(self):
        self.assertIn(
            "n_offsets[None, :] <= m_offsets[:, None]",
            self.kernel_source,
        )
        self.assertIn(
            "m_offsets[:, None] - SLIDING_WINDOW + 1",
            self.kernel_source,
        )
        self.assertIn("MAX_WINDOW_TILES=max_window_tiles", self.kernel_source)

    def test_benchmark_matches_the_real_b16_chunk_plan(self):
        for expected in (
            "batch_size = 8",
            "seq_len = 2048",
            "q_heads = 16",
            "kv_heads = 8",
            "head_dim = 256",
            "sliding_window = 1024",
            "sliding_layers = 25",
            "b16_chunks = 2",
        ):
            self.assertIn(expected, self.benchmark_source)
        self.assertIn('scale=1.0', self.benchmark_source)
        self.assertIn('"--minimum-speedup", type=float, default=1.10', self.benchmark_source)
        self.assertIn("baseline_stability_ratio <= 1.05", self.benchmark_source)
        self.assertIn("repeat_exact", self.benchmark_source)

    def test_harness_has_no_download_install_or_vllm_execution(self):
        self.assertIn(
            'harness_rev: gemma4-long-sliding-prefill-v1',
            self.harness_source,
        )
        self.assertIn('model_download: disabled', self.harness_source)
        self.assertIn('vllm_install: disabled', self.harness_source)
        self.assertIn('package_install: disabled', self.harness_source)
        self.assertNotIn("pip install", self.harness_source)
        self.assertNotIn("snapshot_download", self.harness_source)
        self.assertNotIn("run_gemma4_long_context_vs_vllm", self.harness_source)
        self.assertIn('BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"', self.harness_source)

    def test_embedded_python_compiles(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.harness_source,
            re.DOTALL,
        )
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], "long_sliding_harness_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
