from __future__ import annotations

import unittest

import torch

import torch_transformer_benchmark as bench
from v16_1_clean import UserOptimizedTransformer


class MaskPathTests(unittest.TestCase):
    def test_causal_and_noncausal_mask_paths_pass_strict_gate(self) -> None:
        for causal in (False, True):
            for use_mask in (False, True):
                with self.subTest(causal=causal, use_mask=use_mask):
                    torch.manual_seed(1234)
                    config = bench.TransformerConfig(2, 8, 16, 4, 16, 2, causal)
                    baseline = bench.BaselineTransformer(config).eval()
                    optimized = UserOptimizedTransformer(config).eval()
                    bench.copy_model_weights(baseline, optimized, strict=True)

                    x = torch.randn(2, 8, 16)
                    valid_mask = None
                    if use_mask:
                        valid_mask = torch.tensor(
                            [
                                [True, True, True, True, True, False, False, False],
                                [True, True, True, False, False, False, False, False],
                            ]
                        )
                        x = x.masked_fill(~valid_mask[..., None], 0)

                    with torch.inference_mode():
                        reference = baseline(x, valid_mask)
                        candidate = optimized(x, valid_mask)
                    accuracy = bench.compare_outputs(
                        reference,
                        candidate,
                        rtol=0.02,
                        atol=0.002,
                    )
                    self.assertTrue(accuracy.passed, accuracy)


if __name__ == "__main__":
    unittest.main()
