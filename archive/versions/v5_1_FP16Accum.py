#!/usr/bin/env python3
"""V5.1 candidate: V4.3 with full FP16 accumulation for CUDA GEMMs.

PyTorch normally accumulates FP16 GEMMs in FP32.  On recent CUDA GPUs,
``allow_fp16_accumulation`` lets cuBLAS use FP16 accumulation instead, trading
some numerical accuracy for higher GEMM throughput.  Matrix/profile runners
execute each implementation in its own process, so this process-global backend
setting cannot leak into another candidate.

The public model contract remains FP32.  LayerNorm, residual accumulation,
masking and final output remain unchanged from V4.3; only eligible internal
FP16 GEMMs may use FP16 accumulation.
"""

import torch

import torch_transformer_benchmark as bench
from v4_3_Flash import UserOptimizedTransformer as V43FlashTransformer


def _enable_full_fp16_accumulation() -> None:
    matmul_backend = torch.backends.cuda.matmul
    if not hasattr(matmul_backend, "allow_fp16_accumulation"):
        raise RuntimeError(
            "V5.1 requires torch.backends.cuda.matmul."
            "allow_fp16_accumulation"
        )
    matmul_backend.allow_fp16_accumulation = True


_enable_full_fp16_accumulation()


class UserOptimizedTransformer(V43FlashTransformer):
    """V4.3 graph with full FP16 accumulation enabled for CUDA GEMMs."""


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
