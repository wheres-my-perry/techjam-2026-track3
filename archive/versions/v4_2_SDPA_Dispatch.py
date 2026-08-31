#!/usr/bin/env python3
"""V4.2 candidate: select cuDNN SDPA only on validated winning shapes."""

from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_transformer_benchmark as bench
from v4_mixed_precision_common import MixedPrecisionTransformer


# Exact official shapes where the forced-cuDNN ablation beat V4.1 automatic
# dispatch by a clear margin on the target RTX 5090. Unknown shapes retain the
# automatic PyTorch SDPA fallback rather than assuming the same backend wins.
CUDNN_SHAPES = frozenset(
    {
        (64, 128, 128, 4, 4, 128, True),
        (1, 128, 128, 4, 4, 128, True),
        (4, 128, 128, 4, 4, 128, True),
        (16, 128, 128, 4, 4, 128, True),
        (64, 128, 32, 4, 4, 32, True),
        (64, 128, 128, 1, 4, 128, True),
        (64, 1024, 128, 4, 4, 128, True),
    }
)


class UserOptimizedTransformer(MixedPrecisionTransformer):
    internal_dtype = torch.float16
    gelu_internal_dtype = True

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        shape_key = (
            config.batch_size,
            config.seq_len,
            config.d_model,
            config.num_heads,
            config.num_layers,
            config.ffn_dim,
            config.causal,
        )
        self._use_cudnn_sdpa = shape_key in CUDNN_SHAPES

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self._use_cudnn_sdpa:
            return super()._mixed_attention(attention, x, mask)
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            return super()._mixed_attention(attention, x, mask)


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
