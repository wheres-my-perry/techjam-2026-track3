#!/usr/bin/env python3
"""V6 candidate: V4.3 with tanh-approximated FP16 GELU.

This candidate changes one operation only: the optimized FFN path uses
``F.gelu(..., approximate="tanh")`` instead of the exact/erf formulation.
Attention, mixed-precision caches, LayerNorm, residual accumulation, masking,
fallback behavior and the public FP32 contract are inherited from V4.3.
"""

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v4_3_Flash import UserOptimizedTransformer as V43FlashTransformer


class UserOptimizedTransformer(V43FlashTransformer):
    """V4.3 graph with tanh-approximated GELU on the FP16 FFN path."""

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(
            layer.norm2(x).to(dtype=self.internal_dtype),
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        hidden = F.gelu(hidden, approximate="tanh")
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
