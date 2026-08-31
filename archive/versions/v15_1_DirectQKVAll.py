#!/usr/bin/env python3
"""V15.1: force direct-layout QKV on every short causal configuration.

V15 enables its Triton QKV projection only for exact official shape #13.
This versioned ablation removes that exact-shape restriction below the existing
large-sequence cutoff so the same layout can be accuracy- and performance-tested
on official shapes #1-#12.  Shape #14, non-causal inputs, training, and public
non-FP32 fallbacks retain the inherited behavior.
"""

from __future__ import annotations

import torch_transformer_benchmark as bench
from v15_DirectQKVLayout import UserOptimizedTransformer as V15Transformer


class UserOptimizedTransformer(V15Transformer):
    """V15 with direct-layout QKV enabled across short causal shapes."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_direct_qkv_layout = (
            config.causal
            and config.seq_len < self._LARGE_SEQUENCE_CUTOFF
        )


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
