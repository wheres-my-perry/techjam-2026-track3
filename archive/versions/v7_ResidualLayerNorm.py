#!/usr/bin/env python3
"""V7a candidate: pipeline residual boundaries with the following LayerNorm.

This pure-PyTorch ablation keeps attention and FFN projection outputs in FP16
until the FP32 residual boundary.  Each non-final boundary returns both the new
FP32 residual and the FP16-normalized activation consumed by the next GEMM.
The graph is deliberately expressed this way so TorchInductor can fuse the
cast, residual add, optional padding mask, LayerNorm and output cast.

All model math and public behavior otherwise match V4.3: LayerNorm and
residuals stay FP32, GELU is exact on FP16 hidden values, causal attention is
Flash-first, and training/non-FP32 inputs use the reference fallback.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

import torch_transformer_benchmark as bench
from v4_3_Flash import (
    FLASH_FIRST_BACKENDS,
    UserOptimizedTransformer as V43FlashTransformer,
)


class UserOptimizedTransformer(V43FlashTransformer):
    """V4.3 with LayerNorm activations pipelined across residual boundaries."""

    def _mixed_attention_half(
        self,
        attention,
        normalized: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run attention from an already-normalized FP16 activation."""

        batch, seq_len, _ = normalized.shape
        q, k, v = (
            F.linear(
                normalized,
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

        if self.config.causal:
            with sdpa_kernel(backends=FLASH_FIRST_BACKENDS, set_priority=True):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=attention.scale,
                )
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attention.scale,
            )

        context = context.transpose(1, 2).reshape(
            batch,
            seq_len,
            attention.d_model,
        )
        return F.linear(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        )

    def _mixed_ffn_from_normalized(
        self,
        layer,
        normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Run the exact-GELU FP16 FFN from a pipelined LayerNorm output."""

        hidden = F.linear(
            normalized,
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        hidden = F.gelu(hidden, approximate="none")
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        )

    def _residual_norm_half(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        norm,
        invalid_queries: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the FP32 residual and FP16 normalized activation together."""

        residual = residual + branch.float()
        if invalid_queries is not None:
            residual = residual.masked_fill(invalid_queries, 0)
        normalized = norm(residual).to(dtype=self.internal_dtype)
        return residual, normalized

    def _final_residual_norm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        invalid_queries: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Finish the final residual, mask and public FP32 LayerNorm output."""

        residual = residual + branch.float()
        if invalid_queries is not None:
            residual = residual.masked_fill(invalid_queries, 0)
        output = self.final_norm(residual)
        if invalid_queries is not None:
            output = output.masked_fill(invalid_queries, 0)
        return output

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training or x.dtype != torch.float32:
            return super().forward(x, valid_token_mask)

        mask = None
        invalid_queries = None
        if valid_token_mask is not None:
            mask = valid_token_mask[:, None, None, :]
            invalid_queries = ~valid_token_mask[..., None]

        if not self.layers:
            output = self.final_norm(x)
            if invalid_queries is not None:
                output = output.masked_fill(invalid_queries, 0)
            return output

        normalized1 = self.layers[0].norm1(x).to(dtype=self.internal_dtype)

        for index, layer in enumerate(self.layers):
            attention_output = self._mixed_attention_half(
                layer.attention,
                normalized1,
                mask,
            )
            x, normalized2 = self._residual_norm_half(
                x,
                attention_output,
                layer.norm2,
            )

            ffn_output = self._mixed_ffn_from_normalized(layer, normalized2)
            if index + 1 < len(self.layers):
                x, normalized1 = self._residual_norm_half(
                    x,
                    ffn_output,
                    self.layers[index + 1].norm1,
                    invalid_queries,
                )
            else:
                return self._final_residual_norm(
                    x,
                    ffn_output,
                    invalid_queries,
                )

        raise AssertionError("unreachable")


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
