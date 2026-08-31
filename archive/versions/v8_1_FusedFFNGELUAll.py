#!/usr/bin/env python3
"""V8.1 ablation: force V8 fused FFN-in GEMM + exact GELU on every shape.

This file intentionally removes only V8's performance dispatcher. It exists to
measure the custom Triton epilogue across all official shapes without changing
the stable V8 candidate or any precision, cache, state-dict, and fallback
semantics.
"""

from __future__ import annotations

import torch_transformer_benchmark as bench
from v8_FusedFFNGELU import UserOptimizedTransformer as V8FusedFFNGELU


class UserOptimizedTransformer(V8FusedFFNGELU):
    """V8 with the fused FFN/GELU path enabled for every mixed-precision shape."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_fused_ffn_gelu = True


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
