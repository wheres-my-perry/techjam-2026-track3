#!/usr/bin/env python3
"""V14.1: V11 plus a memory-bounded large-sequence dispatcher.

Official shapes #1-#13 have sequence length at most 1,024, while shape #14 has
sequence length 100,000.  FP32 inference with ``S >= 8192`` and ``B > 1`` is
therefore evaluated as independent batch slices.  Each slice still uses V11
unchanged; only the batch execution schedule and output storage differ.

The chunked helper is excluded from ``torch.compile`` graph capture.  Capturing
the Python loop for shape #14 could unroll all 32 samples into one very large
graph and defeat the memory bound.  The ordinary V11 fallback remains visible
to Dynamo/Inductor and can still be compiled for the smaller official shapes.
"""

from __future__ import annotations

from typing import Optional

import torch

import torch_transformer_benchmark as bench
from v11_FP32PreGELU import UserOptimizedTransformer as V11Transformer


class UserOptimizedTransformer(V11Transformer):
    """V11 with batch-chunked inference above a sequence-length cutoff."""

    # All public shapes except #14 are <= 1,024.  Keeping a large gap avoids
    # perturbing their hot path while routing shape #14 before its full-batch
    # packed-QKV allocation can exhaust a 32 GiB GPU.
    _LARGE_SEQUENCE_CUTOFF = 8_192
    _LARGE_SEQUENCE_BATCH_CHUNK = 1

    def _uses_large_sequence_chunking(self, x: torch.Tensor) -> bool:
        return (
            not self.training
            and x.dtype == torch.float32
            and x.ndim == 3
            and x.shape[0] > 1
            and x.shape[1] >= self._LARGE_SEQUENCE_CUTOFF
        )

    @torch.compiler.disable
    def _forward_large_sequence(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run independent batch slices without retaining full-batch QKV."""

        output = torch.empty_like(x)
        chunk_size = self._LARGE_SEQUENCE_BATCH_CHUNK
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

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._uses_large_sequence_chunking(x):
            return self._forward_large_sequence(x, valid_token_mask)
        return super().forward(x, valid_token_mask)


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
