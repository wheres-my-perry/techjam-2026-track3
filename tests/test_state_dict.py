from __future__ import annotations

import unittest

import torch

import torch_transformer_benchmark as bench
from v16_1_clean import UserOptimizedTransformer


class StateDictCompatibilityTests(unittest.TestCase):
    def test_active_model_has_reference_state_dict(self) -> None:
        config = bench.TransformerConfig(2, 8, 16, 4, 16, 2, True)
        baseline = bench.BaselineTransformer(config)
        optimized = UserOptimizedTransformer(config)

        bench.copy_model_weights(baseline, optimized, strict=True)

        reference_state = baseline.state_dict()
        optimized_state = optimized.state_dict()
        self.assertEqual(tuple(reference_state), tuple(optimized_state))
        for name, reference in reference_state.items():
            self.assertTrue(torch.equal(reference, optimized_state[name]), name)


if __name__ == "__main__":
    unittest.main()
