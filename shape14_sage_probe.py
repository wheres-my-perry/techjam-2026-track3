#!/usr/bin/env python3
"""Isolated PyTorch Flash versus SageAttention2++ probe for shape #14.

The Sage dispatcher uses INT8 QK and FP8 PV on SM120.  This script therefore
requires the strict elementwise comparator before any result can motivate a
full-model candidate.  It is not a full Transformer benchmark.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_transformer_benchmark as bench
from shape14_fa4_probe import strict_compare, summarize, timed_call
from v16_1_clean import UserOptimizedTransformer as V161Transformer


ROOT = Path(__file__).resolve().parent
SAGE_COMMIT = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SHAPE14 = bench.TransformerConfig(32, 100_000, 1024, 16, 1024, 2, True)
PREFIX_CUTOFFS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    100_000,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PyTorch Flash versus SageAttention on shape #14"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100_000)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--distribution",
        choices=("model", "standard"),
        default="model",
        help="use first-layer V16.1-clean QKV or direct standard-normal QKV",
    )
    parser.add_argument(
        "--sage-mode",
        choices=("auto", "pv-fp16", "pv-fp8-fp32"),
        default="auto",
        help="automatic SM120 recipe or more accurate PV controls",
    )
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=101_234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--failure-query-chunk", type=int, default=2_048)
    parser.add_argument("--max-failure-examples", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "seq-len": args.seq_len,
        "heads": args.heads,
        "head-dim": args.head_dim,
        "repeats": args.repeats,
        "failure-query-chunk": args.failure_query_chunk,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these options must be positive: {', '.join(invalid)}")
    if (
        args.warmup < 0
        or args.rtol < 0
        or args.atol < 0
        or args.max_failure_examples < 0
    ):
        raise SystemExit("warmup and tolerances must be non-negative")


def strict_failure_locality(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    query_chunk: int,
    max_examples: int,
) -> dict[str, Any]:
    """Locate strict failures without materializing another full-size mask."""

    if reference.shape != candidate.shape or reference.ndim != 4:
        raise AssertionError("expected matching [B,H,S,Dh] attention outputs")

    seq_len = reference.shape[2]
    cutoffs = tuple(cutoff for cutoff in PREFIX_CUTOFFS if cutoff <= seq_len)
    if not cutoffs or cutoffs[-1] != seq_len:
        cutoffs = (*cutoffs, seq_len)

    failed = 0
    failed_by_head = [0 for _ in range(reference.shape[1])]
    failed_outside_prefix = {str(cutoff): 0 for cutoff in cutoffs}
    failed_queries: set[int] = set()
    examples: list[dict[str, float | int]] = []
    min_query: int | None = None
    max_query: int | None = None

    for start in range(0, seq_len, query_chunk):
        end = min(start + query_chunk, seq_len)
        ref = reference[:, :, start:end].float()
        opt = candidate[:, :, start:end].float()
        absolute = (opt - ref).abs()
        finite = torch.isfinite(ref) & torch.isfinite(opt)
        passed = finite & (
            (absolute < atol) | (absolute < rtol * ref.abs())
        )
        local_indices = torch.nonzero(~passed, as_tuple=False)
        local_failed = int(local_indices.shape[0])
        if local_failed == 0:
            continue

        failed += local_failed
        global_queries = local_indices[:, 2] + start
        query_min = int(global_queries.min().item())
        query_max = int(global_queries.max().item())
        min_query = query_min if min_query is None else min(min_query, query_min)
        max_query = query_max if max_query is None else max(max_query, query_max)
        failed_queries.update(int(value) for value in global_queries.cpu().tolist())

        head_counts = torch.bincount(
            local_indices[:, 1], minlength=reference.shape[1]
        ).cpu()
        for head, count in enumerate(head_counts.tolist()):
            failed_by_head[head] += int(count)
        for cutoff in cutoffs:
            failed_outside_prefix[str(cutoff)] += int(
                (global_queries >= cutoff).sum().item()
            )

        remaining = max_examples - len(examples)
        if remaining <= 0:
            continue
        selected = local_indices[:remaining]
        selected_abs = absolute[
            selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]
        ]
        selected_ref = ref[
            selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]
        ]
        selected_opt = opt[
            selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]
        ]
        selected_rel = selected_abs / selected_ref.abs().clamp_min(1e-30)
        for row, abs_error, ref_value, opt_value, rel_error in zip(
            selected.cpu().tolist(),
            selected_abs.cpu().tolist(),
            selected_ref.cpu().tolist(),
            selected_opt.cpu().tolist(),
            selected_rel.cpu().tolist(),
        ):
            batch, head, query, channel = (int(value) for value in row)
            examples.append(
                {
                    "batch": batch,
                    "head": head,
                    "query": query + start,
                    "channel": channel,
                    "reference": float(ref_value),
                    "candidate": float(opt_value),
                    "abs_error": float(abs_error),
                    "rel_error": float(rel_error),
                }
            )

    return {
        "failed": failed,
        "min_query": min_query,
        "max_query": max_query,
        "minimal_exact_prefix": None if max_query is None else max_query + 1,
        "unique_failed_queries": len(failed_queries),
        "failed_by_head": failed_by_head,
        "failed_outside_prefix": failed_outside_prefix,
        "examples_truncated": failed > len(examples),
        "examples": examples,
    }


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "profile-results" / f"shape14_sage_probe_{timestamp}.json"


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("shape14_sage_probe.py requires an available CUDA device")

    try:
        from sageattention import (
            sageattn,
            sageattn_qk_int8_pv_fp16_cuda,
            sageattn_qk_int8_pv_fp8_cuda,
        )
    except ImportError as error:
        raise SystemExit(
            "SageAttention is unavailable; build it in an isolated environment"
        ) from error

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.distribution == "model":
        if args.heads != 16 or args.head_dim != 64:
            raise SystemExit("model distribution requires shape-#14 H=16, Dh=64")
        torch.manual_seed(args.model_seed)
        model = V161Transformer(SHAPE14).to(device=device, dtype=torch.float32).eval()
        layer = model.layers[0]
        torch.manual_seed(args.seed)
        x = torch.randn(
            (args.batch_size, args.seq_len, SHAPE14.d_model),
            device=device,
            dtype=torch.float32,
        )
        normalized = layer.norm1(x).to(dtype=torch.float16)
        qkv = F.linear(
            normalized,
            layer.attention._qkv_weight_mixed,
            layer.attention._qkv_bias_mixed,
        ).reshape(
            args.batch_size,
            args.seq_len,
            3,
            args.heads,
            args.head_dim,
        )
        del normalized, x, model
    else:
        torch.manual_seed(args.seed)
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
            return F.scaled_dot_product_attention(
                q_hnd,
                k_hnd,
                v_hnd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )

    def sage() -> torch.Tensor:
        common = {
            "tensor_layout": "HND",
            "is_causal": True,
            "sm_scale": scale,
        }
        if args.sage_mode == "auto":
            result = sageattn(q_hnd, k_hnd, v_hnd, **common)
        elif args.sage_mode == "pv-fp16":
            result = sageattn_qk_int8_pv_fp16_cuda(
                q_hnd,
                k_hnd,
                v_hnd,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32",
                smooth_k=True,
                **common,
            )
        else:
            result = sageattn_qk_int8_pv_fp8_cuda(
                q_hnd,
                k_hnd,
                v_hnd,
                qk_quant_gran="per_thread",
                pv_accum_dtype="fp32+fp32",
                smooth_k=True,
                smooth_v=True,
                **common,
            )
        return result[0] if isinstance(result, tuple) else result

    print(
        f"diagnostic=shape14-attention B={args.batch_size} S={args.seq_len} "
        f"H={args.heads} Dh={args.head_dim} dtype=fp16 causal=True "
        f"distribution={args.distribution} sage_mode={args.sage_mode}"
    )
    print(
        f"q_hnd_shape={tuple(q_hnd.shape)} stride={q_hnd.stride()} "
        "sage_sm120=INT8-QK/FP8-PV/fp32+fp16"
    )
    print("scope=isolated attention kernel; not full-model latency or accuracy")

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = pytorch_flash()
            torch.cuda.synchronize(device)
            del output
            output = sage()
            torch.cuda.synchronize(device)
            del output

        samples = {"pytorch_flash": [], "sageattention": []}
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.repeats):
            order = (
                ("pytorch_flash", pytorch_flash), ("sageattention", sage)
            ) if index % 2 == 0 else (
                ("sageattention", sage), ("pytorch_flash", pytorch_flash)
            )
            for name, function in order:
                output, elapsed_ms = timed_call(function, device)
                samples[name].append(elapsed_ms)
                del output

        reference = pytorch_flash()
        candidate = sage()
        torch.cuda.synchronize(device)
        accuracy = strict_compare(reference, candidate, args.rtol, args.atol)
        failure_locality = strict_failure_locality(
            reference,
            candidate,
            rtol=args.rtol,
            atol=args.atol,
            query_chunk=args.failure_query_chunk,
            max_examples=args.max_failure_examples,
        )
        del reference, candidate

    pytorch_timing = summarize(samples["pytorch_flash"])
    sage_timing = summarize(samples["sageattention"])
    speedup = pytorch_timing["median_ms"] / sage_timing["median_ms"]
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
            "sageattention": importlib.metadata.version("sageattention"),
            "sageattention_source_commit": SAGE_COMMIT,
        },
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "causal": True,
            "dtype": "torch.float16",
            "distribution": args.distribution,
            "sage_mode": args.sage_mode,
            "model_seed": args.model_seed if args.distribution == "model" else None,
            "input_seed": args.seed,
            "q_hnd_stride": list(q_hnd.stride()),
        },
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "order": "alternating each repeat",
            "pytorch_flash": pytorch_timing,
            "sageattention": sage_timing,
            "sage_speedup": speedup,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        },
        "attention_output_accuracy": accuracy,
        "failure_locality": failure_locality,
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
        f"SageAttention median={sage_timing['median_ms']:.4f} ms "
        f"p90={sage_timing['p90_ms']:.4f} ms speedup={speedup:.4f}x"
    )
    print(
        f"strict attention comparator={'PASS' if accuracy['passed'] else 'FAIL'} "
        f"failed={accuracy['failed']}/{accuracy['total']} "
        f"max_abs={accuracy['max_abs']:.9g} mean_abs={accuracy['mean_abs']:.9g}"
    )
    print(
        "failure locality: "
        f"query_range={failure_locality['min_query']}..{failure_locality['max_query']} "
        f"unique_queries={failure_locality['unique_failed_queries']} "
        f"minimal_exact_prefix={failure_locality['minimal_exact_prefix']} "
        f"failed_outside_P4096="
        f"{failure_locality['failed_outside_prefix'].get('4096', 'N/A')}"
    )
    print(f"artifact={output_path}")
    return 0 if accuracy["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
