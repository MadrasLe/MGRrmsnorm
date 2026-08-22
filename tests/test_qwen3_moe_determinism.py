import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import megagemm.kernels.qwen3_moe as qwen3_moe_kernel
import megagemm.models.llama as llama_mod
from megagemm.engine.scheduler import Scheduler
from megagemm.models.llama import LlamaConfig, Qwen3MoeExperts


class Qwen3MoeDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous_deterministic)

    def test_long_prefill_determinism_overrides_normal_partial_limit(self):
        with (
            patch.object(
                qwen3_moe_kernel,
                "_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE",
                True,
            ),
            patch.object(
                qwen3_moe_kernel,
                "_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS",
                4096,
            ),
        ):
            self.assertFalse(
                qwen3_moe_kernel._segmented_prefill_uses_partial_reduce(
                    use_fused_gate=True,
                    assignments=8192,
                    deterministic_reduce=False,
                )
            )
            self.assertTrue(
                qwen3_moe_kernel._segmented_prefill_uses_partial_reduce(
                    use_fused_gate=True,
                    assignments=8192,
                    deterministic_reduce=True,
                )
            )

    def test_model_passes_global_determinism_to_segmented_prefill(self):
        config = LlamaConfig.from_dict(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 6,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        )
        experts = Qwen3MoeExperts(config)
        captured = {}

        def fake_segmented(hidden_states, *args, **kwargs):
            captured.update(kwargs)
            return torch.zeros_like(hidden_states)

        torch.use_deterministic_algorithms(True)
        with patch.object(
            llama_mod,
            "qwen3_moe_segmented_prefill",
            fake_segmented,
        ):
            experts._forward_segmented_prefill(
                torch.zeros((2, 8)),
                torch.zeros((2, 2), dtype=torch.int64),
                torch.full((2, 2), 0.5),
            )

        self.assertTrue(captured["deterministic_reduce"])

    def test_decode_deterministic_reduce_also_covers_batch_one(self):
        torch.use_deterministic_algorithms(True)
        self.assertTrue(llama_mod._deterministic_moe_reduce_requested(True))
        self.assertFalse(llama_mod._deterministic_moe_reduce_requested(False))

    @staticmethod
    def _gemma4_a4b_scheduler():
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._gemma4_deterministic_prefill_max_batched_tokens = 16384
        scheduler.model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="gemma4_text",
                enable_moe_block=True,
                hidden_size=2816,
                num_hidden_layers=30,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=704,
            ),
            embed_tokens=SimpleNamespace(
                weight=SimpleNamespace(dtype=torch.bfloat16),
            ),
        )
        return scheduler

    def test_gemma4_a4b_deterministic_prefill_cap_is_shape_scoped(self):
        scheduler = self._gemma4_a4b_scheduler()
        with patch.object(
            torch,
            "are_deterministic_algorithms_enabled",
            return_value=True,
        ):
            self.assertEqual(
                scheduler._gemma4_deterministic_prefill_token_cap(),
                16384,
            )
            scheduler.model.config.num_experts = 64
            self.assertEqual(
                scheduler._gemma4_deterministic_prefill_token_cap(),
                0,
            )

    def test_gemma4_b16_c2048_prefill_splits_into_two_b8_chunks(self):
        scheduler = self._gemma4_a4b_scheduler()
        requests = [
            SimpleNamespace(prompt_len=2048, max_new_tokens=64)
            for _ in range(16)
        ]
        budget = {
            "strategy": "batched_tokens",
            "max_requests": 16,
            "max_batched_tokens": None,
            "cost_budget_tokens": None,
            "bytes_per_token": 0,
        }
        with (
            patch.object(
                torch,
                "are_deterministic_algorithms_enabled",
                return_value=True,
            ),
            patch.object(
                Scheduler,
                "_estimate_prefill_chunk_budget",
                return_value=budget,
            ),
        ):
            plan = scheduler._plan_prefill_chunks(requests)

        self.assertEqual([len(chunk) for chunk in plan["chunks"]], [8, 8])
        self.assertEqual(
            [row["prompt_tokens"] for row in plan["chunk_meta"]],
            [16384, 16384],
        )
        self.assertEqual(plan["deterministic_moe_token_cap"], 16384)

    def test_gemma4_b16_c2048_prefill_can_use_one_b16_chunk(self):
        scheduler = self._gemma4_a4b_scheduler()
        scheduler._gemma4_deterministic_prefill_max_batched_tokens = 32768
        requests = [
            SimpleNamespace(prompt_len=2048, max_new_tokens=64)
            for _ in range(16)
        ]
        budget = {
            "strategy": "batched_tokens",
            "max_requests": 16,
            "max_batched_tokens": None,
            "cost_budget_tokens": None,
            "bytes_per_token": 0,
        }
        with (
            patch.object(
                torch,
                "are_deterministic_algorithms_enabled",
                return_value=True,
            ),
            patch.object(
                Scheduler,
                "_estimate_prefill_chunk_budget",
                return_value=budget,
            ),
        ):
            plan = scheduler._plan_prefill_chunks(requests)

        self.assertEqual([len(chunk) for chunk in plan["chunks"]], [16])
        self.assertEqual(
            [row["prompt_tokens"] for row in plan["chunk_meta"]],
            [32768],
        )
        self.assertEqual(plan["deterministic_moe_token_cap"], 32768)


if __name__ == "__main__":
    unittest.main()
