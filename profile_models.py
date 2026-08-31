#!/usr/bin/env python3
"""Profile Track 3 implementations with correctness and timing gates.

The parent process runs every implementation in an isolated subprocess. Each
child validates correctness, measures end-to-end latency, then collects an
ATen/operator-stage attribution for eager execution or raw device/runtime
evidence for torch.compile execution with PyTorch Profiler/Kineto.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrix_runner import SHAPES, print_shapes, resolve_implementation


ROOT = Path(__file__).resolve().parent
CATEGORIES = (
    "gemm",
    "attention",
    "layer_norm",
    "layout_copy",
    "masking",
    "elementwise",
    "gelu",
    "other",
)
STAGES = (
    "pre_attention_layer_norm",
    "qkv_projection",
    "query_projection",
    "key_projection",
    "value_projection",
    "view_and_reshape",
    "attention_scale",
    "attention_core",
    "attention_output_projection",
    "residual_add",
    "pre_ffn_layer_norm",
    "ffn_input_projection",
    "ffn_activation",
    "ffn_output_projection",
    "masking_and_padding",
    "layout_copy",
    "final_layer_norm",
    "linear_gemm_unattributed",
    "layer_norm_unattributed",
    "elementwise_other",
    "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy-check, benchmark and profile one or more Transformer "
            "implementations on an official Track 3 shape"
        )
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        default=("main",),
        help="implementation aliases or Python paths; default: main",
    )
    parser.add_argument("--shape-id", type=int, default=1)
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    parser.add_argument("--profile-warmup", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument("--no-allow-tf32", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--top-ops", type=int, default=8)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument(
        "--export-traces",
        action="store_true",
        help="write one Chrome/Kineto trace JSON per implementation",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "profile-results"
    )

    child = parser.add_argument_group(argparse.SUPPRESS)
    child.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    child.add_argument("--_impl-path", type=Path, help=argparse.SUPPRESS)
    child.add_argument("--_result-json", type=Path, help=argparse.SUPPRESS)
    child.add_argument("--_trace-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.shape_id not in {shape.id for shape in SHAPES}:
        raise SystemExit(f"invalid official shape ID: {args.shape_id}")
    positive = {
        "accuracy-trials": args.accuracy_trials,
        "repeats": args.repeats,
        "benchmark-rounds": args.benchmark_rounds,
        "profile-iterations": args.profile_iterations,
        "timeout": args.timeout,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these options must be positive: {', '.join(invalid)}")
    if min(args.warmup, args.profile_warmup, args.top_ops) < 0:
        raise SystemExit("warmup, profile-warmup and top-ops must be non-negative")
    if not 0.0 <= args.padding_ratio < 1.0:
        raise SystemExit("padding-ratio must be in [0, 1)")
    if args.input_scale <= 0 or args.rtol < 0 or args.atol < 0:
        raise SystemExit("input-scale must be positive; tolerances must be non-negative")


def selected_shape(shape_id: int):
    return next(shape for shape in SHAPES if shape.id == shape_id)


def load_implementation(path: Path):
    sys.path.insert(0, str(ROOT))
    module_name = f"profile_target_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def implementation_api(module):
    if hasattr(module, "TransformerConfig"):
        return module
    if hasattr(module, "bench"):
        return module.bench
    raise RuntimeError("implementation exposes neither TransformerConfig nor bench")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "min_ms": min(samples_ms),
    }


def benchmark_once(model, x, valid_mask, iterations: int, device) -> list[float]:
    import torch

    samples = []
    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            torch.cuda.synchronize(device)
            for start, end in zip(starts, ends):
                start.record()
                model(x, valid_mask)
                end.record()
            torch.cuda.synchronize(device)
            return [start.elapsed_time(end) for start, end in zip(starts, ends)]

        for _ in range(iterations):
            started = time.perf_counter_ns()
            model(x, valid_mask)
            samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def warmup(model, x, valid_mask, iterations: int, device) -> None:
    import torch

    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maybe_compile(model, enabled: bool, mode: str):
    if not enabled:
        return model
    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def json_safe_mapping(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    normalized = {}
    for key, value in values.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = repr(value)
    return normalized


def compile_mode_options(mode: str) -> dict[str, Any] | None:
    try:
        import torch._inductor as inductor

        return json_safe_mapping(inductor.list_mode_options().get(mode))
    except (AttributeError, ImportError, RuntimeError):
        return None


def accuracy_gate(
    api,
    baseline,
    optimized,
    config,
    device,
    dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    passed = True
    max_abs = 0.0
    max_rel = 0.0
    failed = 0
    total = 0
    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            x, valid_mask = api.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=args.seed + trial,
                padding_ratio=args.padding_ratio,
                input_scale=args.input_scale,
            )
            result = api.compare_outputs(
                baseline(x, valid_mask),
                optimized(x, valid_mask),
                rtol=args.rtol,
                atol=args.atol,
            )
            passed &= result.passed
            max_abs = max(max_abs, result.max_abs_error)
            max_rel = max(max_rel, result.max_relative_error)
            failed += result.failed_elements
            total += result.total_elements
    return {
        "passed": passed,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "failed": failed,
        "total": total,
        "rtol": args.rtol,
        "atol": args.atol,
        "trials": args.accuracy_trials,
    }


def classify_operator(name: str) -> str:
    lowered = name.lower()
    if any(
        token in lowered
        for token in (
            "scaled_dot_product",
            "flash_attention",
            "efficient_attention",
            "bmm",
            "softmax",
        )
    ):
        return "attention"
    if "layer_norm" in lowered:
        return "layer_norm"
    if "gelu" in lowered:
        return "gelu"
    if any(token in lowered for token in ("contiguous", "clone", "copy_", "_to_copy")):
        return "layout_copy"
    if any(token in lowered for token in ("addmm", "aten::mm", "gemm", "aten::linear")):
        return "gemm"
    if any(
        token in lowered
        for token in ("masked_fill", "aten::where", "aten::tril", "aten::triu")
    ):
        return "masking"
    if any(token in lowered for token in ("aten::add", "aten::mul", "aten::div")):
        return "elementwise"
    return "other"


def event_time_us(event, device_type: str) -> float:
    if device_type == "cuda":
        return float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
    return float(event.self_cpu_time_total)


def event_device_type_name(event) -> str:
    device_type = getattr(event, "device_type", None)
    name = getattr(device_type, "name", str(device_type))
    return name.lower().removeprefix("devicetype.")


def raw_event_time_us(event) -> float:
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, 0.0)
        if value:
            return float(value)
    time_range = getattr(event, "time_range", None)
    elapsed_us = getattr(time_range, "elapsed_us", None)
    return float(elapsed_us()) if elapsed_us is not None else 0.0


def device_event_kind(name: str) -> str:
    lowered = name.lower()
    if "memcpy" in lowered or "memset" in lowered:
        return "memory"
    return "kernel"


def device_event_profile(profiler, iterations: int, top_n: int) -> dict[str, Any]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    total_time_us = 0.0
    total_events = 0
    kernel_events = 0
    memory_events = 0
    triton_events = 0

    for event in profiler.events():
        if event_device_type_name(event) != "cuda":
            continue
        name = getattr(event, "name", "<unnamed CUDA event>")
        time_us = raw_event_time_us(event)
        if time_us <= 0:
            continue
        kind = device_event_kind(name)
        key = (name, kind)
        item = aggregated.setdefault(
            key,
            {"name": name, "kind": kind, "time_us": 0.0, "count": 0},
        )
        item["time_us"] += time_us
        item["count"] += 1
        total_time_us += time_us
        total_events += 1
        if kind == "memory":
            memory_events += 1
        else:
            kernel_events += 1
        if "triton" in name.lower():
            triton_events += 1

    events = [
        {
            "name": item["name"],
            "kind": item["kind"],
            "self_time_ms_per_iter": item["time_us"] / iterations / 1000.0,
            "calls_per_iter": item["count"] / iterations,
        }
        for item in aggregated.values()
    ]
    events.sort(key=lambda item: item["self_time_ms_per_iter"], reverse=True)
    return {
        "available": bool(events),
        "total_time_ms_per_iter": total_time_us / iterations / 1000.0,
        "events_per_iter": total_events / iterations,
        "kernel_events_per_iter": kernel_events / iterations,
        "memory_events_per_iter": memory_events / iterations,
        "triton_events_per_iter": triton_events / iterations,
        "unique_event_names": len(events),
        "top_events": events[:top_n],
    }


def runtime_event_profile(profiler, iterations: int) -> dict[str, float]:
    counts = {
        "compiled_regions_per_iter": 0.0,
        "triton_launches_per_iter": 0.0,
        "cuda_graph_launches_per_iter": 0.0,
        "kernel_launch_api_calls_per_iter": 0.0,
    }
    for event in profiler.events():
        if event_device_type_name(event) == "cuda":
            continue
        name = getattr(event, "name", "")
        lowered = name.lower()
        if name.startswith("Torch-Compiled Region"):
            counts["compiled_regions_per_iter"] += 1.0 / iterations
        if lowered.startswith("triton"):
            counts["triton_launches_per_iter"] += 1.0 / iterations
        if "cudagraphlaunch" in lowered or "cugraphlaunch" in lowered:
            counts["cuda_graph_launches_per_iter"] += 1.0 / iterations
        if "launchkernel" in lowered:
            counts["kernel_launch_api_calls_per_iter"] += 1.0 / iterations
    return counts


def tensor_identity(tensor) -> tuple[str, int | None, int] | None:
    if tensor is None or tensor.numel() == 0:
        return None
    return (tensor.device.type, tensor.device.index, tensor.data_ptr())


@contextmanager
def stage_instrumentation(model):
    """Add profiler-only scopes without changing the benchmark path."""
    import torch.nn.functional as functional
    from torch.profiler import record_function

    weight_stages = {}

    def register(weight, stage: str) -> None:
        identity = tensor_identity(weight)
        if identity is not None:
            weight_stages[identity] = stage

    for layer in getattr(model, "layers", ()):
        attention = layer.attention
        register(getattr(attention, "_qkv_weight", None), "qkv_projection")
        register(getattr(attention.q_proj, "weight", None), "query_projection")
        register(getattr(attention.k_proj, "weight", None), "key_projection")
        register(getattr(attention.v_proj, "weight", None), "value_projection")
        register(
            getattr(attention.out_proj, "weight", None),
            "attention_output_projection",
        )
        register(getattr(layer.norm1, "weight", None), "pre_attention_layer_norm")
        register(getattr(layer.norm2, "weight", None), "pre_ffn_layer_norm")
        register(getattr(layer.ffn_in, "weight", None), "ffn_input_projection")
        register(getattr(layer.ffn_out, "weight", None), "ffn_output_projection")
    register(getattr(getattr(model, "final_norm", None), "weight", None), "final_layer_norm")

    original_linear = functional.linear
    original_layer_norm = functional.layer_norm
    original_gelu = functional.gelu
    original_sdpa = functional.scaled_dot_product_attention

    def profiled_linear(input_tensor, weight, bias=None):
        stage = weight_stages.get(tensor_identity(weight))
        if stage is None:
            return original_linear(input_tensor, weight, bias)
        with record_function(f"stage::{stage}"):
            return original_linear(input_tensor, weight, bias)

    def profiled_layer_norm(
        input_tensor,
        normalized_shape,
        weight=None,
        bias=None,
        eps=1e-5,
    ):
        stage = weight_stages.get(tensor_identity(weight))
        if stage is None:
            return original_layer_norm(
                input_tensor, normalized_shape, weight, bias, eps
            )
        with record_function(f"stage::{stage}"):
            return original_layer_norm(
                input_tensor, normalized_shape, weight, bias, eps
            )

    def profiled_gelu(input_tensor, approximate="none"):
        with record_function("stage::ffn_activation"):
            return original_gelu(input_tensor, approximate=approximate)

    def profiled_sdpa(*positional, **keywords):
        with record_function("stage::attention_core"):
            return original_sdpa(*positional, **keywords)

    functional.linear = profiled_linear
    functional.layer_norm = profiled_layer_norm
    functional.gelu = profiled_gelu
    functional.scaled_dot_product_attention = profiled_sdpa
    try:
        yield
    finally:
        functional.linear = original_linear
        functional.layer_norm = original_layer_norm
        functional.gelu = original_gelu
        functional.scaled_dot_product_attention = original_sdpa


def enclosing_stage(event) -> str | None:
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        name = getattr(parent, "name", "")
        if name.startswith("stage::"):
            return name.removeprefix("stage::")
        parent = getattr(parent, "cpu_parent", None)
    return None


def fallback_stage(operator_name: str) -> str:
    lowered = operator_name.lower()
    if any(
        token in lowered
        for token in ("view", "reshape", "permute", "transpose", "unbind", "chunk")
    ):
        return "view_and_reshape"
    if operator_name == "aten::add":
        return "residual_add"
    if operator_name in ("aten::mul", "aten::div"):
        return "attention_scale"
    if any(
        token in lowered
        for token in (
            "masked_fill",
            "where",
            "tril",
            "triu",
            "bitwise",
            "logical_not",
            "fill_",
        )
    ):
        return "masking_and_padding"

    category = classify_operator(operator_name)
    return {
        "attention": "attention_core",
        "layout_copy": "layout_copy",
        "masking": "masking_and_padding",
        "gemm": "linear_gemm_unattributed",
        "layer_norm": "layer_norm_unattributed",
        "gelu": "ffn_activation",
        "elementwise": "elementwise_other",
        "other": "other",
    }[category]


def stage_breakdown(profiler, device_type: str, iterations: int) -> dict[str, Any]:
    totals = {
        name: {"self_time_ms_per_iter": 0.0, "aten_calls_per_iter": 0.0}
        for name in STAGES
    }
    for event in profiler.events():
        name = getattr(event, "name", "")
        if not name.startswith("aten::"):
            continue
        stage = enclosing_stage(event) or fallback_stage(name)
        if stage not in totals:
            stage = "other"
        totals[stage]["aten_calls_per_iter"] += 1.0 / iterations
        totals[stage]["self_time_ms_per_iter"] += (
            event_time_us(event, device_type) / iterations / 1000.0
        )

    total_ms = sum(item["self_time_ms_per_iter"] for item in totals.values())
    for item in totals.values():
        item["share"] = item["self_time_ms_per_iter"] / total_ms if total_ms else 0.0
    return {"total_self_time_ms_per_iter": total_ms, "stages": totals}


def operator_profile(
    model,
    x,
    valid_mask,
    device,
    args: argparse.Namespace,
    compiled: bool,
) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    warmup(model, x, valid_mask, args.profile_warmup, device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        allocated_before = None
        reserved_before = None

    instrumentation = nullcontext() if compiled else stage_instrumentation(model)
    with instrumentation:
        with torch.inference_mode(), profile(
            activities=activities,
            record_shapes=args.record_shapes,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            for _ in range(args.profile_iterations):
                model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if args._trace_path is not None:
        args._trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args._trace_path))

    operators = []
    for event in profiler.key_averages():
        # Kineto may expose both an ATen row and its raw CUDA kernel. Keeping
        # only ATen rows avoids counting the same device work twice.
        if not event.key.startswith("aten::"):
            continue
        time_us = event_time_us(event, device.type)
        if time_us <= 0:
            continue
        operators.append(
            {
                "name": event.key,
                "category": classify_operator(event.key),
                "self_time_ms_per_iter": (
                    time_us / args.profile_iterations / 1000.0
                ),
                "calls_per_iter": event.count / args.profile_iterations,
            }
        )

    operators.sort(key=lambda item: item["self_time_ms_per_iter"], reverse=True)
    total_ms = sum(item["self_time_ms_per_iter"] for item in operators)
    category_totals = {
        name: {"self_time_ms_per_iter": 0.0, "calls_per_iter": 0.0}
        for name in CATEGORIES
    }
    for operator in operators:
        totals = category_totals[operator["category"]]
        totals["self_time_ms_per_iter"] += operator["self_time_ms_per_iter"]
        totals["calls_per_iter"] += operator["calls_per_iter"]
    for totals in category_totals.values():
        totals["share"] = (
            totals["self_time_ms_per_iter"] / total_ms if total_ms else 0.0
        )

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        peak_extra_mb = max(0, peak - allocated_before) / (1024 * 1024)
        allocated_before_mb = allocated_before / (1024 * 1024)
        reserved_before_mb = reserved_before / (1024 * 1024)
        peak_allocated_mb = peak / (1024 * 1024)
    else:
        peak_extra_mb = None
        allocated_before_mb = None
        reserved_before_mb = None
        peak_allocated_mb = None

    stages = (
        None
        if compiled
        else stage_breakdown(profiler, device.type, args.profile_iterations)
    )
    device_events = device_event_profile(
        profiler, args.profile_iterations, args.top_ops
    )
    runtime_events = runtime_event_profile(profiler, args.profile_iterations)

    return {
        "kind": "compiled_device" if compiled else "eager_aten",
        "metric": "self_cuda_time" if device.type == "cuda" else "self_cpu_time",
        "iterations": args.profile_iterations,
        "warmup": args.profile_warmup,
        "total_self_time_ms_per_iter": total_ms,
        "timed_aten_calls_per_iter": sum(
            item["calls_per_iter"] for item in operators
        ),
        "peak_extra_memory_mb": peak_extra_mb,
        "allocated_before_profile_mb": allocated_before_mb,
        "reserved_before_profile_mb": reserved_before_mb,
        "peak_allocated_mb": peak_allocated_mb,
        "categories": category_totals,
        "stage_total_self_time_ms_per_iter": (
            stages["total_self_time_ms_per_iter"] if stages is not None else None
        ),
        "stages": stages["stages"] if stages is not None else None,
        "top_operators": operators[: args.top_ops],
        "device_events": device_events,
        "runtime_events": runtime_events,
        "trace": str(args._trace_path) if args._trace_path is not None else None,
    }


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


def child_main(args: argparse.Namespace) -> int:
    import torch

    if args._impl_path is None or args._result_json is None:
        raise SystemExit("child mode requires implementation and result paths")

    impl_path = args._impl_path.resolve()
    shape = selected_shape(args.shape_id)
    module = load_implementation(impl_path)
    api = implementation_api(module)
    dtype = api.resolve_dtype(args.dtype)
    device = api.resolve_device(args.device)
    config = api.TransformerConfig(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.layers,
        causal=shape.causal,
    )
    config.validate()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_allow_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_allow_tf32

    baseline = api.BaselineTransformer(config)
    optimized = module.UserOptimizedTransformer(config)
    api.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    compile_config = {
        "backend": "inductor" if args.compile_baseline or args.compile_user else None,
        "baseline": args.compile_baseline,
        "optimized": args.compile_user,
        "mode": args.compile_mode if args.compile_baseline or args.compile_user else None,
        "mode_options": (
            compile_mode_options(args.compile_mode)
            if args.compile_baseline or args.compile_user
            else None
        ),
        "unique_kernel_names": os.environ.get(
            "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"
        ),
    }

    result: dict[str, Any] = {
        "implementation": impl_path.name,
        "implementation_path": str(impl_path),
        "implementation_sha256": file_sha256(impl_path),
        "shape": asdict(shape),
        "status": "ERROR",
        "compile": compile_config,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "dtype": str(dtype),
            "matmul_precision": args.matmul_precision,
            "allow_tf32": not args.no_allow_tf32,
            "torch_logs": os.environ.get("TORCH_LOGS"),
        },
    }

    accuracy = accuracy_gate(
        api, baseline, optimized, config, device, dtype, args
    )
    result["accuracy"] = accuracy
    if not accuracy["passed"]:
        result["status"] = "ACCURACY_FAIL"
        args._result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 2

    fixed_x, fixed_mask = api.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed + 100000,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    warmup(baseline, fixed_x, fixed_mask, args.warmup, device)
    warmup(optimized, fixed_x, fixed_mask, args.warmup, device)

    baseline_samples: list[float] = []
    optimized_samples: list[float] = []
    for round_index in range(args.benchmark_rounds):
        order = (
            ((baseline, baseline_samples), (optimized, optimized_samples))
            if round_index % 2 == 0
            else ((optimized, optimized_samples), (baseline, baseline_samples))
        )
        for model, samples in order:
            samples.extend(
                benchmark_once(model, fixed_x, fixed_mask, args.repeats, device)
            )

    baseline_timing = timing_summary(baseline_samples)
    optimized_timing = timing_summary(optimized_samples)
    tokens = shape.batch_size * shape.seq_len
    baseline_timing["throughput_tokens_s"] = (
        tokens * 1000.0 / baseline_timing["median_ms"]
    )
    optimized_timing["throughput_tokens_s"] = (
        tokens * 1000.0 / optimized_timing["median_ms"]
    )
    result["benchmark"] = {
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rounds": args.benchmark_rounds,
        "baseline": baseline_timing,
        "optimized": optimized_timing,
        "speedup": baseline_timing["median_ms"]
        / optimized_timing["median_ms"],
    }
    result["profile"] = operator_profile(
        optimized,
        fixed_x,
        fixed_mask,
        device,
        args,
        compiled=args.compile_user,
    )
    result["status"] = "PASS"
    args._result_json.parent.mkdir(parents=True, exist_ok=True)
    args._result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def child_command(
    args: argparse.Namespace,
    impl_path: Path,
    result_path: Path,
    trace_path: Path | None,
) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--_child",
        "--_impl-path",
        str(impl_path),
        "--_result-json",
        str(result_path),
        "--shape-id",
        str(args.shape_id),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--accuracy-trials",
        str(args.accuracy_trials),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--profile-warmup",
        str(args.profile_warmup),
        "--profile-iterations",
        str(args.profile_iterations),
        "--seed",
        str(args.seed),
        "--padding-ratio",
        str(args.padding_ratio),
        "--input-scale",
        str(args.input_scale),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--matmul-precision",
        args.matmul_precision,
        "--compile-mode",
        args.compile_mode,
        "--top-ops",
        str(args.top_ops),
        "--timeout",
        str(args.timeout),
    ]
    if args.no_allow_tf32:
        command.append("--no-allow-tf32")
    if args.compile_baseline:
        command.append("--compile-baseline")
    if args.compile_user:
        command.append("--compile-user")
    if args.record_shapes:
        command.append("--record-shapes")
    if trace_path is not None:
        command.extend(("--_trace-path", str(trace_path)))
    return command


def display_name(result: dict[str, Any]) -> str:
    return Path(result["implementation"]).stem


def execution_mode(result: dict[str, Any]) -> str:
    compile_config = result.get("compile", {})
    if compile_config.get("optimized"):
        return f"compile:{compile_config.get('mode', 'default')}"
    return "eager"


def value(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def shorten(value: str, width: int = 96) -> str:
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def print_results(results: list[dict[str, Any]]) -> None:
    print("\n=== End-to-end latency ===")
    summary_rows = []
    for result in results:
        benchmark = result.get("benchmark", {})
        baseline = benchmark.get("baseline", {})
        optimized = benchmark.get("optimized", {})
        profile_data = result.get("profile", {})
        device_events = profile_data.get("device_events", {})
        accuracy = result.get("accuracy", {})
        eager_attribution = profile_data.get("kind") == "eager_aten"
        summary_rows.append(
            [
                display_name(result),
                execution_mode(result),
                result["status"],
                "PASS" if accuracy.get("passed") else "FAIL",
                value(baseline.get("median_ms")),
                value(optimized.get("median_ms")),
                value(optimized.get("p90_ms")),
                f"{benchmark['speedup']:.3f}x" if "speedup" in benchmark else "-",
                value(
                    profile_data.get("total_self_time_ms_per_iter")
                    if eager_attribution
                    else None
                ),
                value(device_events.get("total_time_ms_per_iter")),
                value(
                    profile_data.get("timed_aten_calls_per_iter")
                    if eager_attribution
                    else None,
                    1,
                ),
                value(device_events.get("events_per_iter"), 1),
                value(profile_data.get("allocated_before_profile_mb"), 1),
                value(profile_data.get("peak_extra_memory_mb"), 1),
            ]
        )
    print_table(
        [
            "impl",
            "mode",
            "status",
            "accuracy",
            "base_ms",
            "opt_ms",
            "opt_p90",
            "speedup",
            "ATen_ms",
            "GPU_ms",
            "ATen_calls",
            "GPU_events",
            "steady_MB",
            "peak+MB",
        ],
        summary_rows,
    )

    passing = [result for result in results if result["status"] == "PASS"]
    if not passing:
        return
    print("\n=== GPU/runtime evidence per forward ===")
    print_table(
        [
            "impl",
            "mode",
            "GPU kernels",
            "GPU memory",
            "Triton GPU",
            "compiled regions",
            "Triton launches",
            "CUDA graph launches",
            "kernel launch APIs",
        ],
        [
            [
                display_name(result),
                execution_mode(result),
                value(
                    result["profile"]["device_events"].get(
                        "kernel_events_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["device_events"].get(
                        "memory_events_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["device_events"].get(
                        "triton_events_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["runtime_events"].get(
                        "compiled_regions_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["runtime_events"].get(
                        "triton_launches_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["runtime_events"].get(
                        "cuda_graph_launches_per_iter"
                    ),
                    1,
                ),
                value(
                    result["profile"]["runtime_events"].get(
                        "kernel_launch_api_calls_per_iter"
                    ),
                    1,
                ),
            ]
            for result in passing
        ],
    )

    eager_passing = [
        result for result in passing if result["profile"]["kind"] == "eager_aten"
    ]
    if eager_passing:
        print("\n=== Eager operator categories (self time per forward and share) ===")
        headers = ["category"] + [display_name(result) for result in eager_passing]
        rows = []
        for category in CATEGORIES:
            row = [category]
            for result in eager_passing:
                totals = result["profile"]["categories"][category]
                row.append(
                    f"{totals['self_time_ms_per_iter']:.4f} ms "
                    f"({totals['share']:.1%})"
                )
            rows.append(row)
        print_table(headers, rows)

    for result in eager_passing:
        stages = result["profile"]["stages"]
        print(f"\n=== Model stages: {display_name(result)} ===")
        print_table(
            ["stage", "self_ms", "share", "ATen calls/fwd"],
            [
                [
                    stage,
                    f"{stages[stage]['self_time_ms_per_iter']:.4f}",
                    f"{stages[stage]['share']:.1%}",
                    f"{stages[stage]['aten_calls_per_iter']:.1f}",
                ]
                for stage in STAGES
            ],
        )

        operators = result["profile"]["top_operators"]
        if not operators:
            continue
        print(f"\n=== Top operators: {display_name(result)} ===")
        print_table(
            ["operator", "category", "self_ms", "calls/fwd"],
            [
                [
                    operator["name"],
                    operator["category"],
                    f"{operator['self_time_ms_per_iter']:.4f}",
                    f"{operator['calls_per_iter']:.1f}",
                ]
                for operator in operators
            ],
        )

    for result in passing:
        device_events = result["profile"]["device_events"]
        if not device_events.get("top_events"):
            continue
        print(
            f"\n=== Top GPU device events: {display_name(result)} "
            f"({execution_mode(result)}) ==="
        )
        print_table(
            ["event", "kind", "self_ms", "calls/fwd"],
            [
                [
                    shorten(event["name"]),
                    event["kind"],
                    f"{event['self_time_ms_per_iter']:.4f}",
                    f"{event['calls_per_iter']:.1f}",
                ]
                for event in device_events["top_events"]
            ],
        )

    if eager_passing:
        print(
            "\nNote: eager attention_core keeps fused SDPA intact. QK^T, "
            "causal/key mask, softmax and probabilities@V cannot be timed "
            "separately without replacing the fused kernel."
        )
    if len(eager_passing) != len(passing):
        print(
            "Note: compiled graphs are attributed with raw GPU/runtime events. "
            "Eager ATen stage scopes are intentionally disabled because they "
            "would alter or recompile the optimized graph."
        )


def parent_main(args: argparse.Namespace) -> int:
    shape = selected_shape(args.shape_id)
    try:
        implementations = [resolve_implementation(value) for value in args.impl]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"profile_shape{shape.id:02d}_{stamp}.json"
    results = []
    print(
        f"official shape #{shape.id}: B={shape.batch_size} S={shape.seq_len} "
        f"D={shape.d_model} H={shape.heads} L={shape.layers} "
        f"FFN={shape.ffn_dim} causal={shape.causal}"
    )
    print(
        f"device={args.device}, dtype={args.dtype}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(
        "execution: "
        f"baseline={'compile:' + args.compile_mode if args.compile_baseline else 'eager'}, "
        f"optimized={'compile:' + args.compile_mode if args.compile_user else 'eager'}"
    )

    with tempfile.TemporaryDirectory(prefix="track3-profile-") as temp_dir:
        temporary = Path(temp_dir)
        for index, impl_path in enumerate(implementations, 1):
            result_path = temporary / f"{impl_path.stem}.json"
            trace_path = (
                args.output_dir
                / (
                    f"profile_shape{shape.id:02d}_{impl_path.stem}_"
                    f"{'compile-' + args.compile_mode if args.compile_user else 'eager'}_"
                    f"{stamp}.trace.json"
                )
                if args.export_traces
                else None
            )
            command = child_command(args, impl_path, result_path, trace_path)
            print(
                f"[{index}/{len(implementations)}] {impl_path.name}: "
                "accuracy -> benchmark -> "
                f"{'compiled-device' if args.compile_user else 'eager-ATen'} profiler",
                flush=True,
            )
            try:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                output = process.stdout
                returncode: int | str = process.returncode
            except subprocess.TimeoutExpired as exc:
                raw = exc.stdout or ""
                output = raw.decode() if isinstance(raw, bytes) else raw
                returncode = "timeout"
            except OSError as exc:
                output = str(exc)
                returncode = "spawn_error"

            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {
                    "implementation": impl_path.name,
                    "implementation_path": str(impl_path),
                    "status": "TIMEOUT" if returncode == "timeout" else "ERROR",
                    "error": "child process did not produce a result",
                }
            result["returncode"] = returncode if isinstance(returncode, int) else None
            result["child_command"] = command
            keep_child_output = result["status"] != "PASS" or bool(
                os.environ.get("TORCH_LOGS")
            )
            if keep_child_output and output.strip():
                result["child_output"] = output
            results.append(result)

            payload = {
                "metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "invocation": [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        *sys.argv[1:],
                    ],
                    "git_revision": git_revision(),
                    "shape": asdict(shape),
                    "device": args.device,
                    "dtype": args.dtype,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "accuracy_trials": args.accuracy_trials,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "benchmark_rounds": args.benchmark_rounds,
                    "profile_warmup": args.profile_warmup,
                    "profile_iterations": args.profile_iterations,
                    "seed": args.seed,
                    "padding_ratio": args.padding_ratio,
                    "input_scale": args.input_scale,
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "compile_baseline": args.compile_baseline,
                    "compile_user": args.compile_user,
                    "compile_mode": args.compile_mode,
                    "torch_logs": os.environ.get("TORCH_LOGS"),
                    "unique_kernel_names": os.environ.get(
                        "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"
                    ),
                },
                "results": results,
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if result["status"] != "PASS":
                print(f"    {result['status']}")
                tail = "\n".join(output.splitlines()[-12:])
                if tail:
                    print(tail)
            elif os.environ.get("TORCH_LOGS") and output.strip():
                print(output.rstrip())

    print_results(results)
    print(f"\nJSON: {output_path}")
    if args.export_traces:
        print(f"Traces: {args.output_dir}")
    return 0 if all(result["status"] == "PASS" for result in results) else 1


def main() -> int:
    args = parse_args()
    if args.list_shapes:
        print_shapes()
        return 0
    validate_args(args)
    if args.compile_baseline or args.compile_user:
        os.environ.setdefault("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES", "1")
    return child_main(args) if args._child else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
