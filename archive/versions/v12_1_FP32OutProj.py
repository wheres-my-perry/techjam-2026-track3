#!/usr/bin/env python3
"""V12.1 ablation: store the attention out-projection directly as FP32.

This keeps V11's FFN path unchanged and removes only the FP16 output rounding
at the attention residual boundary.  Q/K/V, SDPA, out-projection activation,
weight, and cached bias remain FP16; CUDA stores the GEMM accumulator as FP32.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

import torch_transformer_benchmark as bench
import v4_3_Flash as v4_3
import v11_FP32PreGELU as v11
from v12_FP32FFNOut import linear_fp32_output


class UserOptimizedTransformer(v11.UserOptimizedTransformer):
    """V11 with only the attention out-projection FP16 output round removed."""

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = (
            F.linear(
                x.to(dtype=self.internal_dtype),
                attention._qkv_weight_mixed,
                attention._qkv_bias_mixed,
            )
            .reshape(
                batch,
                seq_len,
                3,
                attention.num_heads,
                attention.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        backend_context = (
            sdpa_kernel(
                backends=v4_3.FLASH_FIRST_BACKENDS,
                set_priority=True,
            )
            if self.config.causal
            else nullcontext()
        )
        with backend_context:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if self.config.causal else mask,
                dropout_p=0.0,
                is_causal=self.config.causal,
                scale=attention.scale,
            )

        context = context.transpose(1, 2).reshape(
            batch,
            seq_len,
            attention.d_model,
        )
        return linear_fp32_output(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        )


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
