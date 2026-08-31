#!/usr/bin/env python3
"""V17: compile two large-sequence batch samples per executor call.

V16 keeps shape #14 memory-bounded by compiling and reusing a B=1 Transformer
executor inside an eager outer loop.  V17 is a one-variable ablation that
raises only that executor batch chunk from one to two.  The outer loop remains
compiler-disabled, and the compile cache/lifecycle remain inherited from V16.

The V14.1 dispatcher must be bypassed inside the executor because a B=2 input
would otherwise recursively re-enter large-sequence chunking.  Calling V11's
forward implementation directly preserves the shape-#14 arithmetic inherited
through V15 while skipping only that scheduling wrapper.
"""

from __future__ import annotations

from typing import Optional

import torch

import torch_transformer_benchmark as bench
from v11_FP32PreGELU import UserOptimizedTransformer as V11Transformer
from v16_CompiledBatchExecutor import UserOptimizedTransformer as V16Transformer


class UserOptimizedTransformer(V16Transformer):
    """V16 with a compiled large-sequence executor batch chunk of two."""

    _LARGE_SEQUENCE_BATCH_CHUNK = 2

    @property
    def large_sequence_executor_batch_size(self) -> int:
        return self._LARGE_SEQUENCE_BATCH_CHUNK

    def _forward_large_sequence_chunk(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # B=2 would recurse through V14.1's forward dispatcher if this used
        # super().forward().  V11.forward is the arithmetic body that V14.1
        # schedules, and V15's exact-#13 attention override remains inactive
        # for the shape-#14 config.
        return V11Transformer.forward(self, x, valid_token_mask)

    def forward_large_sequence_chunk(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one B=1/B=2 large-sequence chunk through the cached executor."""

        if self.training or x.dtype != torch.float32:
            raise ValueError("compiled chunk path requires FP32 eval mode")
        if (
            x.ndim != 3
            or x.shape[0] < 1
            or x.shape[0] > self._LARGE_SEQUENCE_BATCH_CHUNK
        ):
            raise ValueError("compiled chunk path requires shape [B,S,D], B in [1,2]")
        if x.shape[1] < self._LARGE_SEQUENCE_CUTOFF:
            raise ValueError("sequence length is below the large-sequence cutoff")
        if valid_token_mask is not None and valid_token_mask.shape != x.shape[:2]:
            raise ValueError("valid_token_mask must have shape [B,S]")
        return self.prepare_large_sequence_executor()(x, valid_token_mask)

    def forward_large_sequence_sample(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility hook used by V16's eager outer loop and harnesses."""

        return self.forward_large_sequence_chunk(x, valid_token_mask)


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
