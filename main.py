#!/usr/bin/env python3
"""Benchmark entrypoint for the standalone source-clean V16.1 model."""

from __future__ import annotations

import torch_transformer_benchmark as bench
from v16_1_clean import UserOptimizedTransformer


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
