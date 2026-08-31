#!/usr/bin/env python3
"""Memory-bounded strict accuracy validation for official shape #14.

The official baseline materializes [B, H, S, S] attention scores and therefore
cannot execute shape #14 on a 32 GiB GPU.  This diagnostic keeps the exact
baseline formula, weights, dtype, causal/key-mask semantics, and strict
elementwise comparator, but evaluates reference attention in query blocks and
compares one independent batch sample at a time.

This script is an accuracy tool only.  It must not be used to claim baseline
latency or candidate speedup because query blocking changes the execution
schedule of the reference implementation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v16_1_clean import UserOptimizedTransformer as V161Transformer


SHAPE14 = bench.TransformerConfig(
    batch_size=32,
    seq_len=100_000,
    d_model=1024,
    num_heads=16,
    ffn_dim=1024,
    num_layers=2,
    causal=True,
)
IMPLEMENTATIONS = {
    "main": V161Transformer,
    "v16.1": V161Transformer,
    "v16_1": V161Transformer,
    "v16.1.clean": V161Transformer,
    "v16_1_clean": V161Transformer,
}


def memory_bounded_attention(
    attention: bench.BaselineSelfAttention,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    query_chunk: int,
) -> torch.Tensor:
    """Evaluate the baseline attention formula without a full S-by-S score."""

    batch, seq_len, _ = x.shape
    if batch != 1:
        raise ValueError("memory-bounded reference expects one batch sample")

    q = attention._split_heads(attention.q_proj(x))
    k = attention._split_heads(attention.k_proj(x))
    v = attention._split_heads(attention.v_proj(x))
    context = torch.empty_like(q)
    key_positions = torch.arange(seq_len, device=x.device)[None, :]
    invalid_keys = (
        None
        if valid_token_mask is None
        else ~valid_token_mask[:, None, None, :]
    )
    k_transposed = k.transpose(-2, -1)

    for start in range(0, seq_len, query_chunk):
        end = min(start + query_chunk, seq_len)
        scores = torch.matmul(q[:, :, start:end], k_transposed)
        scores.mul_(attention.scale)

        if causal:
            query_positions = torch.arange(
                start, end, device=x.device
            )[:, None]
            scores.masked_fill_(
                key_positions > query_positions,
                float("-inf"),
            )
        if invalid_keys is not None:
            scores.masked_fill_(invalid_keys, float("-inf"))

        # The official baseline computes softmax in FP32, then casts back to
        # the public dtype before P@V.  Shape #14's public dtype is FP32.
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context[:, :, start:end].copy_(torch.matmul(probabilities, v))

    context_merged = (
        context.transpose(1, 2)
        .contiguous()
        .view(batch, seq_len, attention.d_model)
    )
    output = attention.out_proj(context_merged)
    if valid_token_mask is not None:
        output.masked_fill_(~valid_token_mask[..., None], 0)
    return output


def memory_bounded_baseline(
    model: bench.BaselineTransformer,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    query_chunk: int,
) -> torch.Tensor:
    """Run the unchanged Transformer formula using query-blocked attention."""

    invalid_queries = (
        None
        if valid_token_mask is None
        else ~valid_token_mask[..., None]
    )
    for layer in model.layers:
        attention_output = memory_bounded_attention(
            layer.attention,
            layer.norm1(x),
            valid_token_mask,
            model.config.causal,
            query_chunk,
        )
        x = x + attention_output
        x = x + layer.ffn_out(
            F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
        )
        if invalid_queries is not None:
            x.masked_fill_(invalid_queries, 0)

    x = model.final_norm(x)
    if invalid_queries is not None:
        x.masked_fill_(invalid_queries, 0)
    return x


@dataclass
class StreamingAccuracy:
    total: int = 0
    failed: int = 0
    abs_sum: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0

    @property
    def passed(self) -> bool:
        return self.failed == 0


def update_accuracy(
    summary: StreamingAccuracy,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    token_chunk: int,
    rtol: float = 0.02,
    atol: float = 0.002,
) -> None:
    """Apply the official strict OR comparator in memory-bounded token slices."""

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    for start in range(0, reference.shape[1], token_chunk):
        end = min(start + token_chunk, reference.shape[1])
        ref = reference[:, start:end].float()
        opt = candidate[:, start:end].float()
        finite = torch.isfinite(ref) & torch.isfinite(opt)
        abs_error = (opt - ref).abs()
        passed = finite & (
            (abs_error < atol) | (abs_error < rtol * ref.abs())
        )
        relative = abs_error / ref.abs().clamp_min(1e-12)

        summary.total += ref.numel()
        summary.failed += int((~passed).sum().item())
        summary.abs_sum += float(abs_error.double().sum().item())
        summary.max_abs = max(summary.max_abs, float(abs_error.max().item()))
        summary.max_rel = max(summary.max_rel, float(relative.max().item()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict, memory-bounded accuracy validation for shape #14"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-limit", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--compare-token-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--impl", choices=tuple(IMPLEMENTATIONS), default="main")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--disable-inner-compile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_limit <= SHAPE14.batch_size:
        raise ValueError("batch-limit must be in [1, 32]")
    if args.query_chunk <= 0 or args.compare_token_chunk <= 0:
        raise ValueError("chunk sizes must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    baseline = bench.BaselineTransformer(SHAPE14)
    candidate = IMPLEMENTATIONS[args.impl](SHAPE14)
    bench.copy_model_weights(baseline, candidate, strict=True)
    baseline = baseline.to(device=device, dtype=torch.float32).eval()
    candidate = candidate.to(device=device, dtype=torch.float32).eval()
    if hasattr(candidate, "configure_large_sequence_executor"):
        candidate.configure_large_sequence_executor(
            enabled=not args.disable_inner_compile,
            mode=args.compile_mode,
        )
    executor_batch = (
        int(getattr(candidate, "_LARGE_SEQUENCE_BATCH_CHUNK", 1))
        if hasattr(candidate, "forward_large_sequence_sample")
        else 1
    )

    x, valid_mask = bench.generate_random_case(
        config=SHAPE14,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    summary = StreamingAccuracy()
    started = time.perf_counter()
    print(
        "shape=#14 B=32 S=100000 D=1024 H=16 L=2 FFN=1024 causal=True",
        flush=True,
    )
    print(
        "criterion: abs_error < 0.002 OR relative_error < 2%; "
        f"query_chunk={args.query_chunk}; batches={args.batch_limit}; "
        f"implementation={args.impl}; "
        f"executor_batch={executor_batch}; "
        f"inner_compile={not args.disable_inner_compile if hasattr(candidate, 'configure_large_sequence_executor') else False}",
        flush=True,
    )

    with torch.inference_mode():
        for group_start in range(0, args.batch_limit, executor_batch):
            group_end = min(group_start + executor_batch, args.batch_limit)
            if hasattr(candidate, "forward_large_sequence_sample"):
                optimized_group = candidate.forward_large_sequence_sample(
                    x[group_start:group_end],
                    valid_mask[group_start:group_end],
                )
            else:
                optimized_group = candidate(
                    x[group_start:group_end],
                    valid_mask[group_start:group_end],
                )

            for index in range(group_start, group_end):
                sample_started = time.perf_counter()
                sample_x = x[index : index + 1]
                sample_mask = valid_mask[index : index + 1]
                reference = memory_bounded_baseline(
                    baseline,
                    sample_x,
                    sample_mask,
                    args.query_chunk,
                )
                optimized = optimized_group[
                    index - group_start : index - group_start + 1
                ]
                before_failed = summary.failed
                update_accuracy(
                    summary,
                    reference,
                    optimized,
                    args.compare_token_chunk,
                )
                torch.cuda.synchronize(device)
                status = "PASS" if summary.failed == before_failed else "FAIL"
                print(
                    f"batch {index + 1:02d}/{args.batch_limit}: {status} | "
                    f"cumulative_max_abs={summary.max_abs:.6g} | "
                    f"cumulative_max_rel={summary.max_rel:.6g} | "
                    f"cumulative_failed={summary.failed}/{summary.total} | "
                    f"elapsed={time.perf_counter() - sample_started:.3f}s",
                    flush=True,
                )
                del reference, optimized
            del optimized_group

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    mean_abs = summary.abs_sum / summary.total
    print(
        f"summary: {'PASS' if summary.passed else 'FAIL'} | "
        f"max_abs={summary.max_abs:.6g} | max_rel={summary.max_rel:.6g} | "
        f"mean_abs={mean_abs:.6g} | failed={summary.failed}/{summary.total} | "
        f"elapsed_s={elapsed:.3f} | "
        f"peak_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.3f}",
        flush=True,
    )
    return 0 if summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
