#!/usr/bin/env python3
"""V18: exact-shape-#14 SageAttention PV-FP16 accuracy candidate.

V16 spends more than 92% of its shape-#14 inner-executor device time in exact
PyTorch Flash Attention.  V18 replaces only that attention core with the more
accurate SageAttention recipe (INT8 QK, FP16 PV, FP32 PV accumulation) on the
exact official shape.  Every other shape and environments without the optional
dependency retain V16 unchanged.

This version is an accuracy-gated experiment.  It must not be promoted or
benchmarked before the full-model strict comparator passes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v16_CompiledBatchExecutor import UserOptimizedTransformer as V16Transformer


try:
    from sageattention import sageattn_qk_int8_pv_fp16_cuda
except ImportError:
    sageattn_qk_int8_pv_fp16_cuda = None


@torch.library.custom_op(
    "techjam::sage_attention_shape14_pv_fp16",
    mutates_args=(),
)
def sage_attention_shape14_pv_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the pinned SageAttention accuracy-oriented SM120 recipe."""

    if sageattn_qk_int8_pv_fp16_cuda is None or not q.is_cuda:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
    result = sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=True,
        qk_quant_gran="per_thread",
        sm_scale=scale,
        pv_accum_dtype="fp32",
        smooth_k=True,
        smooth_v=False,
        return_lse=False,
    )
    return result[0] if isinstance(result, tuple) else result


@sage_attention_shape14_pv_fp16.register_fake
def _sage_attention_shape14_pv_fp16_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del k, v, scale
    # SageAttention materializes a contiguous [B, H, S, D] output even when
    # Q comes from the non-contiguous interleaved-QKV view above.  The fake
    # kernel must preserve that stride contract for Inductor's runtime guards.
    return q.new_empty(q.shape)


class UserOptimizedTransformer(V16Transformer):
    """V16 with SageAttention only for exact official shape #14."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_sage_attention_shape14 = (
            config.batch_size == 32
            and config.seq_len == 100_000
            and config.d_model == 1024
            and config.num_heads == 16
            and config.num_layers == 2
            and config.ffn_dim == 1024
            and config.causal
        )

    @property
    def sage_attention_available(self) -> bool:
        return sageattn_qk_int8_pv_fp16_cuda is not None

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            not self._use_sage_attention_shape14
            or not x.is_cuda
            or not self.sage_attention_available
        ):
            return super()._mixed_attention(attention, x, mask)

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
        context = sage_attention_shape14_pv_fp16(
            q,
            k,
            v,
            attention.scale,
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
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
