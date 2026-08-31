#!/usr/bin/env python3
"""V19.1.1: V19 CUDA arithmetic plus parallel batch partitions."""

from __future__ import annotations

import torch_transformer_benchmark as bench
from v19_CUDAFP16Checkpoint import (
    TransformerConfig,
    UserOptimizedTransformer as V19Transformer,
)
from v19_parallel_batch_common import ParallelBatchPartitionsMixin


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


class UserOptimizedTransformer(
    ParallelBatchPartitionsMixin,
    V19Transformer,
):
    """V19 checkpointed-FP16 arithmetic with multi-stream batch scheduling."""


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
