#!/usr/bin/env python3
"""V4.1 candidate: V4 FP16 internal compute with GELU kept in FP16."""

import torch

import torch_transformer_benchmark as bench
from v4_mixed_precision_common import MixedPrecisionTransformer


class UserOptimizedTransformer(MixedPrecisionTransformer):
    internal_dtype = torch.float16
    gelu_internal_dtype = True


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
