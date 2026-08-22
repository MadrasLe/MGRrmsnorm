import unittest

import torch

from megagemm.kernels.gemma4_grouped_prefill import (
    gemma4_grouped_mm_prefill,
    gemma4_grouped_mm_prefill_prefers_shape,
)


def _tiny_cpu_inputs():
    hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
    gate_up = torch.zeros((4, 6, 8), dtype=torch.bfloat16)
    down = torch.zeros((4, 8, 3), dtype=torch.bfloat16)
    experts = torch.zeros((2, 2), dtype=torch.int64)
    routing = torch.full((2, 2), 0.5, dtype=torch.bfloat16)
    return hidden, gate_up, down, experts, routing


class Gemma4GroupedPrefillTests(unittest.TestCase):
    def test_grouped_prefill_rejects_non_target_cpu_shape(self):
        inputs = _tiny_cpu_inputs()
        self.assertFalse(gemma4_grouped_mm_prefill_prefers_shape(*inputs))
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            gemma4_grouped_mm_prefill(*inputs)

    def test_grouped_prefill_rejects_mismatched_route_shapes(self):
        hidden, gate_up, down, experts, routing = _tiny_cpu_inputs()
        self.assertFalse(
            gemma4_grouped_mm_prefill_prefers_shape(
                hidden,
                gate_up,
                down,
                experts,
                routing[:, :1],
            )
        )


if __name__ == "__main__":
    unittest.main()
