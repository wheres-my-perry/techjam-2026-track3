#!/usr/bin/env python3
"""V12.2 ablation: store both residual-branch projections as FP32.

The attention out-projection comes from V12.1 and the FFN-out projection comes
from V12.  All GEMM inputs, weights, and cached biases remain FP16; only the two
projection outputs that feed FP32 residual additions avoid a final FP16 round.
"""

from __future__ import annotations

import torch

import torch_transformer_benchmark as bench
import v11_FP32PreGELU as v11
import v12_1_FP32OutProj as v12_1
from v12_FP32FFNOut import linear_fp32_output


class UserOptimizedTransformer(v12_1.UserOptimizedTransformer):
    """V11 with FP32 outputs for both attention-out and FFN-out projections."""

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        normalized = layer.norm2(x).to(dtype=self.internal_dtype)
        hidden = v11.fused_ffn_gelu_no_preround(
            normalized,
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        return linear_fp32_output(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        )


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
