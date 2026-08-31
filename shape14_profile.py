#!/usr/bin/env python3
"""Profile the compiled inner executor used by official shape #14.

This is deliberately not a full-forward benchmark. It allocates only the
standalone V16.1 large-sequence executor batch (B=1), so the result is a
memory-safe kernel-attribution diagnostic. Full shape-#14 latency remains the
responsibility of ``shape14_optimized_benchmark.py``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.attention import SDPBackend
from torch.profiler import ProfilerActivity, profile

import torch_transformer_benchmark as bench
from profile_models import device_event_profile, runtime_event_profile
from v16_1_clean import (
    FLASH_FIRST_BACKENDS,
    UserOptimizedTransformer as V161Transformer,
)
from v19_CUDAFP16Checkpoint import UserOptimizedTransformer as V19Transformer
from v19_1_0_ParallelBatchV161 import (
    UserOptimizedTransformer as V1910Transformer,
)
from v19_1_1_ParallelBatchV19 import (
    UserOptimizedTransformer as V1911Transformer,
)


ROOT = Path(__file__).resolve().parent
SHAPE14 = bench.TransformerConfig(32, 100_000, 1024, 16, 1024, 2, True)
IMPLEMENTATIONS = {
    "main": (V161Transformer, ROOT / "v16_1_clean.py"),
    "v16.1": (V161Transformer, ROOT / "v16_1_clean.py"),
    "v16_1": (V161Transformer, ROOT / "v16_1_clean.py"),
    "v16.1.clean": (V161Transformer, ROOT / "v16_1_clean.py"),
    "v16_1_clean": (V161Transformer, ROOT / "v16_1_clean.py"),
    "v19": (V19Transformer, ROOT / "v19_CUDAFP16Checkpoint.py"),
    "v19.cuda": (V19Transformer, ROOT / "v19_CUDAFP16Checkpoint.py"),
    "v19_cuda": (V19Transformer, ROOT / "v19_CUDAFP16Checkpoint.py"),
    "v19.1.0": (V1910Transformer, ROOT / "v19_1_0_ParallelBatchV161.py"),
    "v19_1_0": (V1910Transformer, ROOT / "v19_1_0_ParallelBatchV161.py"),
    "v19.1.1": (V1911Transformer, ROOT / "v19_1_1_ParallelBatchV19.py"),
    "v19_1_1": (V1911Transformer, ROOT / "v19_1_1_ParallelBatchV19.py"),
}
SDPA_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
DEFAULT_SDPA_PRIORITY = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)
CATEGORY_ORDER = (
    "attention",
    "qkv_projection",
    "ffn_gelu",
    "linear_projection",
    "fused_projection_norm",
    "norm_residual_mask",
    "memory",
    "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the memory-safe inner executor for official shape #14"
    )
    parser.add_argument("--impl", choices=tuple(IMPLEMENTATIONS), default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--top-events", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("flash-first", *SDPA_BACKENDS),
        default="flash-first",
        help="backend priority used inside the existing shape-#14 attention path",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--no-allow-tf32", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "repeats": args.repeats,
        "profile-iterations": args.profile_iterations,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these options must be positive: {', '.join(invalid)}")
    if min(args.warmup, args.profile_warmup, args.top_events) < 0:
        raise SystemExit("warmup, profile-warmup and top-events must be non-negative")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    process = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def classify_device_event(name: str, kind: str) -> str:
    """Assign one auditable heuristic category to a raw CUDA event."""

    if kind == "memory":
        return "memory"
    lowered = name.lower()
    if any(token in lowered for token in ("flash", "fmha", "attention")):
        return "attention"

    has_ffn = any(token in lowered for token in ("ffn", "gelu"))
    has_qkv = "qkv" in lowered
    has_projection = any(
        token in lowered
        for token in ("addmm", "gemm", "matmul", "cutlass", "cublas", "_mm")
    )
    has_norm = any(
        token in lowered
        for token in (
            "layer_norm",
            "layernorm",
            "native_layer_norm",
            "masked_fill",
            "bitwise_not",
            "residual",
        )
    )

    if has_norm and (has_qkv or has_projection or has_ffn):
        return "fused_projection_norm"
    if has_qkv:
        return "qkv_projection"
    if has_ffn:
        return "ffn_gelu"
    if has_projection:
        return "linear_projection"
    if has_norm:
        return "norm_residual_mask"
    return "other"


def categorize_device_events(device_events: dict[str, Any]) -> dict[str, Any]:
    categories = {
        name: {"time_ms_per_call": 0.0, "events_per_call": 0.0}
        for name in CATEGORY_ORDER
    }
    categorized_events = []
    for event in device_events["top_events"]:
        category = classify_device_event(event["name"], event["kind"])
        categories[category]["time_ms_per_call"] += event["self_time_ms_per_iter"]
        categories[category]["events_per_call"] += event["calls_per_iter"]
        categorized_events.append({**event, "category": category})

    total_ms = device_events["total_time_ms_per_iter"]
    for item in categories.values():
        item["share"] = item["time_ms_per_call"] / total_ms if total_ms else 0.0
    return {
        "method": "heuristic classification of non-overlapping raw CUDA events",
        "categories": categories,
        "events": categorized_events,
    }


def run_warmup(
    executor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            output = executor(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid inner-executor output contract")
            torch.cuda.synchronize(device)
            del output


def cuda_event_benchmark(
    executor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples_ms = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = executor(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid inner-executor output contract")
            end.record()
            torch.cuda.synchronize(device)
            samples_ms.append(start.elapsed_time(end))
            del output

    result = bench.TimingResult(samples_ms)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "samples_ms": samples_ms,
        "median_ms_per_call": result.median_ms,
        "mean_ms_per_call": result.mean_ms,
        "p90_ms_per_call": result.p90_ms,
        "min_ms_per_call": result.min_ms,
        "allocated_before_mb": allocated_before / 2**20,
        "peak_allocated_mb": peak / 2**20,
        "peak_extra_mb": max(0, peak - allocated_before) / 2**20,
    }


def kineto_profile(
    executor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    trace_path: Path | None,
) -> dict[str, Any]:
    run_warmup(executor, x, valid_mask, args.profile_warmup, device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode(), profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(args.profile_iterations):
            output = executor(x, valid_mask)
            if output.shape != x.shape or output.dtype != x.dtype:
                raise AssertionError("invalid inner-executor output contract")
            del output
    torch.cuda.synchronize(device)

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    # Use a high limit here so category totals cover every unique raw event;
    # terminal display is trimmed separately by --top-events.
    device_events = device_event_profile(profiler, args.profile_iterations, 10_000)
    categorized = categorize_device_events(device_events)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "iterations": args.profile_iterations,
        "warmup": args.profile_warmup,
        "allocated_before_mb": allocated_before / 2**20,
        "reserved_before_mb": reserved_before / 2**20,
        "peak_allocated_mb": peak / 2**20,
        "peak_extra_mb": max(0, peak - allocated_before) / 2**20,
        "device_events": device_events,
        "runtime_events": runtime_event_profile(profiler, args.profile_iterations),
        "attribution": categorized,
        "trace": str(trace_path) if trace_path is not None else None,
    }


def default_output_path(implementation: str, sdpa_backend: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        ROOT
        / "profile-results"
        / f"shape14_inner_{implementation}_{sdpa_backend}_{timestamp}.json"
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("shape14_profile.py requires an available CUDA device")

    transformer_class, implementation_path = IMPLEMENTATIONS[args.impl]
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    allow_tf32 = not args.no_allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if args.sdpa_backend == "flash-first":
        FLASH_FIRST_BACKENDS[:] = DEFAULT_SDPA_PRIORITY
    else:
        FLASH_FIRST_BACKENDS[:] = [SDPA_BACKENDS[args.sdpa_backend]]

    model = transformer_class(SHAPE14).to(device=device, dtype=torch.float32).eval()
    model.configure_large_sequence_executor(enabled=True, mode=args.compile_mode)
    executor_batch = int(model._LARGE_SEQUENCE_BATCH_CHUNK)

    torch.manual_seed(args.seed + 100_000)
    x = torch.randn(
        (executor_batch, SHAPE14.seq_len, SHAPE14.d_model),
        device=device,
        dtype=torch.float32,
    )
    valid_mask = torch.ones(
        (executor_batch, SHAPE14.seq_len), device=device, dtype=torch.bool
    )
    executor = model.forward_large_sequence_sample

    print(
        f"diagnostic=shape14-inner-executor impl={args.impl} "
        f"executor_batch={executor_batch} S={SHAPE14.seq_len} D={SHAPE14.d_model} "
        f"mode={args.compile_mode} sdpa_backend={args.sdpa_backend} tf32={allow_tf32}"
    )
    print("scope=inner executor only; not official full-forward latency or speedup")

    run_warmup(executor, x, valid_mask, args.warmup, device)
    if not model.large_sequence_executor_ready:
        raise AssertionError("compiled large-sequence executor was not prepared")
    gc.collect()
    torch.cuda.synchronize(device)

    timing = cuda_event_benchmark(
        executor, x, valid_mask, args.repeats, device
    )
    timing["executor_batch"] = executor_batch
    timing["median_ms_per_sample"] = timing["median_ms_per_call"] / executor_batch
    timing["p90_ms_per_sample"] = timing["p90_ms_per_call"] / executor_batch
    timing["throughput_tokens_s"] = (
        executor_batch * SHAPE14.seq_len * 1000.0 / timing["median_ms_per_call"]
    )

    output_path = args.output or default_output_path(args.impl, args.sdpa_backend)
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    trace_path = output_path.with_suffix(".trace.json") if args.export_trace else None
    profile_result = kineto_profile(
        executor, x, valid_mask, args, device, trace_path
    )

    properties = torch.cuda.get_device_properties(device)
    artifact = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "invocation": sys.argv,
            "diagnostic_scope": "shape14 compiled inner executor only",
            "official_full_forward_performance": False,
            "git_revision": git_revision(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "unique_kernel_names": os.environ.get(
                "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "matmul_precision": args.matmul_precision,
            "allow_tf32": allow_tf32,
        },
        "implementation": {
            "alias": args.impl,
            "path": str(implementation_path),
            "sha256": file_sha256(implementation_path),
            "compile_mode": args.compile_mode,
            "sdpa_backend": args.sdpa_backend,
            "sdpa_priority": [str(backend) for backend in FLASH_FIRST_BACKENDS],
            "executor_batch": executor_batch,
        },
        "shape": asdict(SHAPE14),
        "input": {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "valid_mask": "all true tensor",
            "seed": args.seed + 100_000,
        },
        "timing": timing,
        "profile": profile_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")

    print(
        f"timing median={timing['median_ms_per_call']:.4f} ms/call "
        f"({timing['median_ms_per_sample']:.4f} ms/sample), "
        f"p90={timing['p90_ms_per_call']:.4f} ms/call"
    )
    print(
        f"raw device={profile_result['device_events']['total_time_ms_per_iter']:.4f} "
        f"ms/call, kernels={profile_result['device_events']['kernel_events_per_iter']:.1f}, "
        f"memory_events={profile_result['device_events']['memory_events_per_iter']:.1f}"
    )
    for category in CATEGORY_ORDER:
        item = profile_result["attribution"]["categories"][category]
        if item["time_ms_per_call"] <= 0:
            continue
        print(
            f"  {category:24s} {item['time_ms_per_call']:10.4f} ms "
            f"{item['share'] * 100:7.3f}%"
        )
    print("top raw CUDA events:")
    for event in profile_result["attribution"]["events"][: args.top_events]:
        print(
            f"  {event['self_time_ms_per_iter']:10.4f} ms "
            f"x{event['calls_per_iter']:.2f} [{event['category']}] {event['name']}"
        )
    print(f"artifact={output_path}")
    if trace_path is not None:
        print(f"trace={trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
