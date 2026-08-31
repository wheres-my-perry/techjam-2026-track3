#!/usr/bin/env python3
"""V19.1.0: V16.1 with parallel large-batch CUDA-stream partitions."""

from __future__ import annotations

import torch_transformer_benchmark as bench
from v16_1_clean import (
    TransformerConfig,
    UserOptimizedTransformer as V161Transformer,
)
from v19_parallel_batch_common import ParallelBatchPartitionsMixin


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


class UserOptimizedTransformer(
    ParallelBatchPartitionsMixin,
    V161Transformer,
):
    """V16.1 arithmetic with a no-CUDA-Graph multi-stream outer scheduler."""


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
