#!/usr/bin/env python3
"""V14: memory-bounded batch chunking for official shape #14.

The public shape #14 input and output each occupy about 12.21 GiB in FP32 on
the target RTX 5090.  V11 additionally creates an 18.31 GiB packed-QKV FP16
tensor for the whole batch, which cannot coexist with the input on a 32 GiB
device.  Batch elements are independent in this Transformer, so inference can
run complete batch slices through every layer and write them into one output
tensor without changing the mathematical result.

Only the exact official #14 inference configuration uses this path.  Other
shapes, training, and non-FP32 public inputs retain V11 unchanged.  A chunk
size of one is deliberately conservative: a B=1/S=100000 probe peaked at
2.96 GiB on the target GPU, leaving room for the full public input and output.
"""

from __future__ import annotations

from typing import Optional

import torch

import torch_transformer_benchmark as bench
from v11_FP32PreGELU import UserOptimizedTransformer as V11Transformer


class UserOptimizedTransformer(V11Transformer):
    """V11 with an exact batch-independent, memory-bounded shape-#14 path."""

    _SHAPE14_BATCH_CHUNK = 1

    def _uses_shape14_chunking(self, x: torch.Tensor) -> bool:
        config = self.config
        return (
            not self.training
            and x.dtype == torch.float32
            and x.ndim == 3
            and tuple(x.shape) == (32, 100_000, 1024)
            and config.batch_size == 32
            and config.seq_len == 100_000
            and config.d_model == 1024
            and config.num_heads == 16
            and config.ffn_dim == 1024
            and config.num_layers == 2
            and config.causal
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._uses_shape14_chunking(x):
            return super().forward(x, valid_token_mask)

        output = torch.empty_like(x)
        chunk_size = self._SHAPE14_BATCH_CHUNK
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            chunk_mask = (
                None
                if valid_token_mask is None
                else valid_token_mask[start:end]
            )
            output[start:end].copy_(
                super().forward(x[start:end], chunk_mask)
            )
        return output


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
