#!/usr/bin/env python3
"""V4.3 candidate: Flash-first causal SDPA with backend fallbacks.

For causal self-attention with right padding, valid tokens form a prefix.  A
valid query cannot attend to padded keys in the suffix because the causal rule
already excludes them; invalid query outputs are zeroed after each block.  The
causal optimized path can therefore omit the redundant key-padding attention
mask and make PyTorch Flash SDPA eligible.

PyTorch tries Flash first, then cuDNN, memory-efficient attention, and Math when
a backend does not support the current device or shape.  Non-causal inputs keep
their key mask and use PyTorch's automatic dispatcher.
"""

from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_transformer_benchmark as bench
from v4_mixed_precision_common import MixedPrecisionTransformer


FLASH_FIRST_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class UserOptimizedTransformer(MixedPrecisionTransformer):
    internal_dtype = torch.float16
    gelu_internal_dtype = True

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.causal:
            return super()._mixed_attention(attention, x, mask)

        with sdpa_kernel(backends=FLASH_FIRST_BACKENDS, set_priority=True):
            return super()._mixed_attention(attention, x, None)


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
