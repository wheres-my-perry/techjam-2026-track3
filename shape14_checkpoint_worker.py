#!/usr/bin/env python3
"""Run one isolated shape-#14 stage for one historical checkpoint.

This worker never reports a baseline latency or speedup.  Accuracy uses the
unchanged reference arithmetic with query-blocked attention; native stages call
the selected checkpoint on the complete official [32, 100000, 1024] input.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from dataclasses import replace
from typing import Any, Sequence

import torch

from shape14_accuracy import (
    SHAPE14,
    StreamingAccuracy,
    memory_bounded_baseline,
    update_accuracy,
)
from timeline_adapter import CHECKPOINTS, load_checkpoint, resolve_checkpoint


RESULT_MARKER = "SHAPE14_STAGE_JSON="
STAGES = ("b1-accuracy", "streamed-b32-accuracy", "native-b32-probe", "native-timing")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated checkpoint shape-#14 worker")
    parser.add_argument("--checkpoint", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-limit", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--compare-token-chunk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-mode", default="max-autotune")
    return parser.parse_args(argv)


def configure_runtime(device: torch.device, seed: int) -> None:
    if device.type != "cuda":
        raise ValueError("shape #14 requires a CUDA device")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def prepare_models(args: argparse.Namespace, *, need_reference: bool):
    spec = resolve_checkpoint(args.checkpoint)
    if spec.shape14_strategy == "static_infeasible":
        raise ValueError(f"{args.checkpoint} is statically infeasible for shape #14")

    bench, implementation_class, module_name = load_checkpoint(spec)
    baseline = bench.BaselineTransformer(SHAPE14)
    candidate = implementation_class(SHAPE14)
    bench.copy_model_weights(baseline, candidate, strict=True)

    device = torch.device(args.device)
    candidate = candidate.to(device=device, dtype=torch.float32).eval()
    if need_reference:
        baseline = baseline.to(device=device, dtype=torch.float32).eval()
    else:
        del baseline
        baseline = None
        gc.collect()

    eager_candidate = candidate
    if spec.shape14_strategy == "inner_executor":
        if not hasattr(eager_candidate, "configure_large_sequence_executor"):
            raise RuntimeError("inner-executor strategy has no executor configuration API")
        eager_candidate.configure_large_sequence_executor(
            enabled=True,
            mode=args.compile_mode,
        )
    elif spec.shape14_strategy == "outer_compile":
        candidate = torch.compile(
            eager_candidate,
            mode=args.compile_mode,
            dynamic=False,
        )

    return bench, baseline, candidate, eager_candidate, module_name


def candidate_sample(
    candidate,
    eager_candidate,
    strategy: str,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if strategy == "inner_executor":
        return eager_candidate.forward_large_sequence_sample(x, valid_mask)
    return candidate(x, valid_mask)


def result_base(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_checkpoint(args.checkpoint)
    return {
        "checkpoint_id": args.checkpoint,
        "checkpoint_label": spec.label,
        "stage": args.stage,
        "shape_id": 14,
        "shape": {
            "batch_size": SHAPE14.batch_size,
            "seq_len": SHAPE14.seq_len,
            "d_model": SHAPE14.d_model,
            "heads": SHAPE14.num_heads,
            "layers": SHAPE14.num_layers,
            "ffn_dim": SHAPE14.ffn_dim,
            "causal": SHAPE14.causal,
        },
        "strategy": spec.shape14_strategy,
        "outer_compile": spec.shape14_strategy == "outer_compile",
        "inner_executor": spec.shape14_strategy == "inner_executor",
        "compile_mode": args.compile_mode,
        "seed": args.seed,
        "allow_tf32": True,
        "matmul_precision": "high",
        "baseline_latency_ms": None,
        "speedup": None,
    }


def make_input(bench, args: argparse.Namespace, batch_size: int):
    input_config = replace(SHAPE14, batch_size=batch_size)
    return bench.generate_random_case(
        config=input_config,
        device=torch.device(args.device),
        dtype=torch.float32,
        seed=args.seed,
        padding_ratio=0.0,
        input_scale=1.0,
    )


def accuracy_stage(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    spec = resolve_checkpoint(args.checkpoint)
    bench, baseline, candidate, eager_candidate, module_name = prepare_models(
        args, need_reference=True
    )
    x, valid_mask = make_input(bench, args, batch_size)
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    summary = StreamingAccuracy()
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(batch_size):
            sample_started = time.perf_counter()
            sample_x = x[index : index + 1]
            sample_mask = valid_mask[index : index + 1]
            optimized = candidate_sample(
                candidate,
                eager_candidate,
                spec.shape14_strategy,
                sample_x,
                sample_mask,
            )
            if optimized.shape != sample_x.shape or optimized.dtype != sample_x.dtype:
                raise AssertionError(
                    "invalid candidate output contract: "
                    f"shape={tuple(optimized.shape)}, dtype={optimized.dtype}"
                )
            reference = memory_bounded_baseline(
                baseline,
                sample_x,
                sample_mask,
                args.query_chunk,
            )
            failed_before = summary.failed
            update_accuracy(
                summary,
                reference,
                optimized,
                args.compare_token_chunk,
            )
            torch.cuda.synchronize(device)
            print(
                f"sample {index + 1:02d}/{batch_size}: "
                f"{'PASS' if summary.failed == failed_before else 'FAIL'} | "
                f"failed={summary.failed}/{summary.total} | "
                f"max_abs={summary.max_abs:.6g} | "
                f"elapsed_s={time.perf_counter() - sample_started:.3f}",
                flush=True,
            )
            del reference, optimized

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = result_base(args)
    result.update(
        {
            "status": "PASS" if summary.passed else "ACCURACY_FAIL",
            "resolved_module": module_name,
            "evaluated_batch_samples": batch_size,
            "query_chunk": args.query_chunk,
            "compare_token_chunk": args.compare_token_chunk,
            "failed": summary.failed,
            "total": summary.total,
            "max_abs": summary.max_abs,
            "max_rel": summary.max_rel,
            "mean_abs": summary.abs_sum / summary.total,
            "elapsed_s": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "output_shape": [1, SHAPE14.seq_len, SHAPE14.d_model],
            "output_dtype": "torch.float32",
        }
    )
    return result


def native_probe(args: argparse.Namespace) -> dict[str, Any]:
    bench, _, candidate, _, module_name = prepare_models(args, need_reference=False)
    x, valid_mask = make_input(bench, args, SHAPE14.batch_size)
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = candidate(x, valid_mask)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    expected_shape = (SHAPE14.batch_size, SHAPE14.seq_len, SHAPE14.d_model)
    contract_pass = output.shape == expected_shape and output.dtype == torch.float32
    result = result_base(args)
    result.update(
        {
            "status": "PASS" if contract_pass else "ERROR",
            "resolved_module": module_name,
            "elapsed_s": elapsed,
            "output_contract": "PASS" if contract_pass else "FAIL",
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
    )
    return result


def collect_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def native_timing(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    bench, _, candidate, _, module_name = prepare_models(args, need_reference=False)
    x, valid_mask = make_input(bench, args, SHAPE14.batch_size)
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = candidate(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid optimized output contract")
            torch.cuda.synchronize(device)
            del output
            collect_cuda_cache()

        samples_ms: list[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = candidate(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid optimized output contract")
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
            samples_ms.append(elapsed_ms)
            print(
                f"repeat {index + 1:02d}/{args.repeats}: {elapsed_ms:.4f} ms",
                flush=True,
            )
            del output
            collect_cuda_cache()

    timing = bench.TimingResult(samples_ms)
    throughput = SHAPE14.batch_size * SHAPE14.seq_len * 1000.0 / timing.median_ms
    result = result_base(args)
    result.update(
        {
            "status": "PASS",
            "resolved_module": module_name,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "samples_ms": samples_ms,
            "optimized_median_ms": timing.median_ms,
            "optimized_mean_ms": timing.mean_ms,
            "optimized_p90_ms": timing.p90_ms,
            "optimized_min_ms": timing.min_ms,
            "throughput_tokens_per_s": throughput,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "output_shape": [SHAPE14.batch_size, SHAPE14.seq_len, SHAPE14.d_model],
            "output_dtype": "torch.float32",
        }
    )
    return result


def emit(result: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runtime(torch.device(args.device), args.seed)
    try:
        if args.stage == "b1-accuracy":
            result = accuracy_stage(args, 1)
        elif args.stage == "streamed-b32-accuracy":
            if args.batch_limit != SHAPE14.batch_size:
                raise ValueError("official streamed accuracy requires --batch-limit 32")
            result = accuracy_stage(args, args.batch_limit)
        elif args.stage == "native-b32-probe":
            result = native_probe(args)
        else:
            result = native_timing(args)
    except torch.cuda.OutOfMemoryError as exc:
        result = result_base(args)
        result.update({"status": "OOM", "error": str(exc)})
        emit(result)
        return 3
    except Exception as exc:  # isolated process records the full diagnostic
        result = result_base(args)
        result.update(
            {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        emit(result)
        return 4

    emit(result)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
