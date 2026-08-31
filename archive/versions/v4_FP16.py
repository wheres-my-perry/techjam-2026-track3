#!/usr/bin/env python3
"""V4 candidate: FP16 QKV/SDPA/projection/FFN with FP32 norm/residuals."""

import torch

import torch_transformer_benchmark as bench
from v4_mixed_precision_common import MixedPrecisionTransformer


class UserOptimizedTransformer(MixedPrecisionTransformer):
    internal_dtype = torch.float16


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
