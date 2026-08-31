#!/usr/bin/env python3
"""Optimized-only CUDA Event diagnostic for official shape #14.

The original reference cannot be timed at this shape because it materializes
an approximately 18.6 TiB attention-score tensor.  Consequently this script
reports no baseline latency or speedup and must not be presented as a paired
official performance result.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

import torch_transformer_benchmark as bench
from v16_1_clean import UserOptimizedTransformer as V161Transformer
from candidates.v19.cuda_fp16_checkpoint import UserOptimizedTransformer as V19Transformer
from candidates.v19.parallel_batch_v161 import (
    UserOptimizedTransformer as V1910Transformer,
)
from candidates.v19.parallel_batch_v19 import (
    UserOptimizedTransformer as V1911Transformer,
)


SHAPE14 = bench.TransformerConfig(32, 100_000, 1024, 16, 1024, 2, True)
IMPLEMENTATIONS = {
    "main": V161Transformer,
    "v16.1": V161Transformer,
    "v16_1": V161Transformer,
    "v16.1.clean": V161Transformer,
    "v16_1_clean": V161Transformer,
    "v19": V19Transformer,
    "v19.cuda": V19Transformer,
    "v19_cuda": V19Transformer,
    "v19.1.0": V1910Transformer,
    "v19_1_0": V1910Transformer,
    "v19.1.1": V1911Transformer,
    "v19_1_1": V1911Transformer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized-only shape-#14 CUDA Event diagnostic"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--impl", choices=tuple(IMPLEMENTATIONS), default="main")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--disable-inner-compile", action="store_true")
    return parser.parse_args()


def collect_cuda_cache() -> None:
    """Collect Python cycles and release cached blocks between timed calls."""

    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    eager_model = IMPLEMENTATIONS[args.impl](SHAPE14).to(device).eval()
    if hasattr(eager_model, "configure_large_sequence_executor"):
        eager_model.configure_large_sequence_executor(
            enabled=not args.disable_inner_compile,
            mode=args.compile_mode,
        )
    x, valid_mask = bench.generate_random_case(
        SHAPE14,
        device,
        torch.float32,
        args.seed + 100_000,
        0.0,
        1.0,
    )
    if not eager_model._uses_large_sequence_chunking(x):
        raise AssertionError("shape #14 did not enter the chunked path")
    model = (
        torch.compile(eager_model, mode=args.compile_mode)
        if args.compile_user
        else eager_model
    )
    print(
        f"implementation={args.impl} "
        f"cutoff={eager_model._LARGE_SEQUENCE_CUTOFF} "
        f"batch_chunk={eager_model._LARGE_SEQUENCE_BATCH_CHUNK} "
        f"parallel_parts={getattr(eager_model, 'parallel_batch_parts', 1)} "
        f"outer_compile={args.compile_user} "
        f"inner_compile={not args.disable_inner_compile if hasattr(eager_model, 'configure_large_sequence_executor') else False} "
        f"mode={getattr(eager_model, '_large_sequence_compile_mode', args.compile_mode)}"
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid optimized output contract")
            torch.cuda.synchronize(device)
            del output
            collect_cuda_cache()

        samples_ms = []
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid optimized output contract")
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
            samples_ms.append(elapsed_ms)
            print(f"repeat {index + 1:02d}/{args.repeats}: {elapsed_ms:.4f} ms")
            del output
            collect_cuda_cache()

    result = bench.TimingResult(samples_ms)
    tokens_per_second = (
        SHAPE14.batch_size * SHAPE14.seq_len * 1000.0 / result.median_ms
    )
    print("baseline : N/A (explicit score tensor is approximately 18.6 TiB)")
    print(
        f"optimized: median={result.median_ms:.4f} ms | "
        f"mean={result.mean_ms:.4f} ms | p90={result.p90_ms:.4f} ms | "
        f"min={result.min_ms:.4f} ms | throughput={tokens_per_second:.2f} token/s"
    )
    print("speedup  : N/A (no executable paired baseline)")
    print(
        f"peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
