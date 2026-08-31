#!/usr/bin/env python3
"""V12 ablation: store the FFN-out projection directly as FP32.

V11 keeps the FFN hidden activation and FFN-out weight/bias in FP16, lets the
GEMM accumulate in FP32, then rounds the projection result to FP16 before
casting it back to FP32 for the residual path.  V12 removes only that final
rounding boundary.  CUDA uses ``torch.mm(..., out_dtype=torch.float32)``;
local CPU diagnostics promote the already-quantized FP16 operands to FP32.

Attention, V11's FP32 pre-GELU epilogue, cache/state-dict behavior, residuals,
LayerNorm, training, and non-FP32 public-input fallbacks are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
import v11_FP32PreGELU as v11


def linear_fp32_output(
    activation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run an FP16-input Linear while preserving its FP32 accumulator output."""

    if activation.dtype != torch.float16:
        raise TypeError("linear_fp32_output requires an FP16 activation")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("linear_fp32_output requires FP16 weight and bias")
    if weight.shape[1] != activation.shape[-1]:
        raise ValueError("incompatible activation and weight shapes")
    if bias.numel() != weight.shape[0]:
        raise ValueError("incompatible weight and bias shapes")

    if not activation.is_cuda:
        return F.linear(
            activation.float(),
            weight.float(),
            bias.float(),
        )

    activation_2d = activation.reshape(-1, activation.shape[-1])
    output_2d = torch.mm(
        activation_2d,
        weight.t(),
        out_dtype=torch.float32,
    )
    output_2d = output_2d + bias.float()
    return output_2d.reshape(*activation.shape[:-1], weight.shape[0])


class UserOptimizedTransformer(v11.UserOptimizedTransformer):
    """V11 with only the FFN-out FP16 output round removed."""

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
