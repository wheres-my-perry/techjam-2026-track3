#!/usr/bin/env python3
"""V19.1.1: V19 CUDA arithmetic plus parallel batch partitions."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch_transformer_benchmark as bench
from candidates.v19.cuda_fp16_checkpoint import (
    TransformerConfig,
    UserOptimizedTransformer as V19Transformer,
)
from candidates.v19.parallel_batch_common import ParallelBatchPartitionsMixin


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


class UserOptimizedTransformer(
    ParallelBatchPartitionsMixin,
    V19Transformer,
):
    """V19 checkpointed-FP16 arithmetic with multi-stream batch scheduling."""


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
