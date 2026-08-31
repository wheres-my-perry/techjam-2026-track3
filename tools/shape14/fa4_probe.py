#!/usr/bin/env python3
"""Isolated PyTorch Flash versus FlashAttention-4 probe for shape #14.

The probe uses the exact attention dimensions of official shape #14 and the
interleaved QKV stride produced by the current packed projection.  It is a
kernel diagnostic, not a full Transformer benchmark or a replacement for the
full-model strict accuracy gate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PyTorch Flash versus FA4 on shape-#14 attention"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100_000)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101_234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "seq-len": args.seq_len,
        "heads": args.heads,
        "head-dim": args.head_dim,
        "repeats": args.repeats,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these options must be positive: {', '.join(invalid)}")
    if args.warmup < 0 or args.rtol < 0 or args.atol < 0:
        raise SystemExit("warmup and tolerances must be non-negative")


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


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": percentile(samples_ms, 0.9),
        "min_ms": min(samples_ms),
    }


def timed_call(
    function: Callable[[], torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    torch.cuda.synchronize(device)
    return output, start.elapsed_time(end)


def strict_compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float,
    atol: float,
) -> dict[str, float | int | bool]:
    if reference.shape != candidate.shape:
        raise AssertionError("attention outputs have different shapes")
    absolute = (candidate.float() - reference.float()).abs()
    relative = absolute / reference.float().abs().clamp_min(1e-30)
    passed = (absolute < atol) | (relative < rtol)
    return {
        "passed": bool(passed.all().item()),
        "failed": int((~passed).sum().item()),
        "total": passed.numel(),
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "max_rel": float(relative.max().item()),
        "rtol": rtol,
        "atol": atol,
    }


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / "profiles" / f"shape14_fa4_probe_{timestamp}.json"


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("tools.shape14.fa4_probe requires an available CUDA device")

    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as error:
        raise SystemExit(
            "FlashAttention-4 is unavailable; install it in an isolated environment"
        ) from error

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # [B,S,3,H,Dh] mirrors the packed projection before its current permute.
    # Unbinding Q/K/V preserves the interleaved sequence stride 3*H*Dh while
    # keeping the head dimension contiguous, exactly like the model path.
    qkv = torch.randn(
        (
            args.batch_size,
            args.seq_len,
            3,
            args.heads,
            args.head_dim,
        ),
        device=device,
        dtype=torch.float16,
    )
    q_nhd, k_nhd, v_nhd = qkv.unbind(2)
    q_hnd = q_nhd.transpose(1, 2)
    k_hnd = k_nhd.transpose(1, 2)
    v_hnd = v_nhd.transpose(1, 2)
    scale = 1.0 / math.sqrt(args.head_dim)

    def pytorch_flash() -> torch.Tensor:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_hnd = F.scaled_dot_product_attention(
                q_hnd,
                k_hnd,
                v_hnd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
        return output_hnd.transpose(1, 2)

    def fa4() -> torch.Tensor:
        result = flash_attn_func(
            q_nhd,
            k_nhd,
            v_nhd,
            softmax_scale=scale,
            causal=True,
        )
        # FA4 b28 returns ``(output, lse_or_none)`` even when return_lse=False.
        return result[0] if isinstance(result, tuple) else result

    print(
        f"diagnostic=shape14-attention B={args.batch_size} S={args.seq_len} "
        f"H={args.heads} Dh={args.head_dim} dtype=fp16 causal=True"
    )
    print(
        f"q_nhd_shape={tuple(q_nhd.shape)} stride={q_nhd.stride()} "
        f"q_hnd_stride={q_hnd.stride()}"
    )
    print("scope=isolated attention kernel; not full-model latency or accuracy")

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = pytorch_flash()
            torch.cuda.synchronize(device)
            del output
            output = fa4()
            torch.cuda.synchronize(device)
            del output

        samples = {"pytorch_flash": [], "fa4": []}
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.repeats):
            order = (
                ("pytorch_flash", pytorch_flash), ("fa4", fa4)
            ) if index % 2 == 0 else (
                ("fa4", fa4), ("pytorch_flash", pytorch_flash)
            )
            for name, function in order:
                output, elapsed_ms = timed_call(function, device)
                samples[name].append(elapsed_ms)
                del output

        reference = pytorch_flash()
        candidate = fa4()
        torch.cuda.synchronize(device)
        accuracy = strict_compare(reference, candidate, args.rtol, args.atol)
        del reference, candidate

    pytorch_timing = summarize(samples["pytorch_flash"])
    fa4_timing = summarize(samples["fa4"])
    speedup = pytorch_timing["median_ms"] / fa4_timing["median_ms"]
    properties = torch.cuda.get_device_properties(device)
    artifact = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "invocation": sys.argv,
            "diagnostic_scope": "isolated shape14 causal attention",
            "official_full_model_result": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "flash_attn_4": importlib.metadata.version("flash-attn-4"),
            "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        },
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "causal": True,
            "dtype": "torch.float16",
            "q_nhd_stride": list(q_nhd.stride()),
            "q_hnd_stride": list(q_hnd.stride()),
        },
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "order": "alternating each repeat",
            "pytorch_flash": pytorch_timing,
            "fa4": fa4_timing,
            "fa4_speedup": speedup,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        },
        "attention_output_accuracy": accuracy,
    }
    output_path = args.output or default_output_path()
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")

    print(
        f"PyTorch Flash median={pytorch_timing['median_ms']:.4f} ms "
        f"p90={pytorch_timing['p90_ms']:.4f} ms"
    )
    print(
        f"FA4           median={fa4_timing['median_ms']:.4f} ms "
        f"p90={fa4_timing['p90_ms']:.4f} ms speedup={speedup:.4f}x"
    )
    print(
        f"strict attention comparator={'PASS' if accuracy['passed'] else 'FAIL'} "
        f"failed={accuracy['failed']}/{accuracy['total']} "
        f"max_abs={accuracy['max_abs']:.9g} mean_abs={accuracy['mean_abs']:.9g}"
    )
    print(f"artifact={output_path}")
    return 0 if accuracy["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
