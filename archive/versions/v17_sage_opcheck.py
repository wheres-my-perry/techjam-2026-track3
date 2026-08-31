#!/usr/bin/env python3
"""Accuracy-only custom-op and compile integration gate for V17-Sage."""

from __future__ import annotations

import argparse
import math

import torch

from v17_sage import (
    SAGE_ATTENTION_INSTALLED_VERSION,
    SAGE_ATTENTION_REQUIRED_VERSION,
    sage_attention_exact_prefix,
    sageattn_qk_int8_pv_fp16_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sage custom-op opcheck and eager/compiled equivalence"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--exact-prefix", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "heads": args.heads,
        "seq-len": args.seq_len,
        "head-dim": args.head_dim,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these options must be positive: {', '.join(invalid)}")
    if args.head_dim > 128:
        raise SystemExit("SageAttention 2.2 supports original head_dim <= 128")
    if not 0 <= args.exact_prefix <= args.seq_len:
        raise SystemExit("exact-prefix must be in [0, seq-len]")
    if args.rtol < 0 or args.atol < 0:
        raise SystemExit("tolerances must be non-negative")


def strict_compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[int, int, float, bool]:
    ref = reference.float()
    opt = candidate.float()
    absolute = (opt - ref).abs()
    finite = torch.isfinite(ref) & torch.isfinite(opt)
    passed = finite & (
        (absolute < atol) | (absolute < rtol * ref.abs())
    )
    failed = int((~passed).sum().item())
    return failed, ref.numel(), float(absolute.max().item()), bool(
        torch.equal(reference, candidate)
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    if sageattn_qk_int8_pv_fp16_cuda is None:
        raise SystemExit("sageattention could not be imported")
    if SAGE_ATTENTION_INSTALLED_VERSION != SAGE_ATTENTION_REQUIRED_VERSION:
        raise SystemExit(
            f"requires sageattention {SAGE_ATTENTION_REQUIRED_VERSION}, "
            f"found {SAGE_ATTENTION_INSTALLED_VERSION!r}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("V17-Sage opcheck requires an available CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(args.head_dim)
    example_args = (q, k, v, scale, args.exact_prefix)

    print(
        f"sageattention={SAGE_ATTENTION_INSTALLED_VERSION} shape={shape} "
        f"prefix={args.exact_prefix} mode={args.compile_mode}"
    )
    opcheck_result = torch.library.opcheck(
        sage_attention_exact_prefix,
        example_args,
        raise_exception=True,
    )
    print(f"opcheck=PASS details={opcheck_result}")

    def call(
        q_value: torch.Tensor,
        k_value: torch.Tensor,
        v_value: torch.Tensor,
    ) -> torch.Tensor:
        return sage_attention_exact_prefix(
            q_value,
            k_value,
            v_value,
            scale,
            args.exact_prefix,
        )

    compiled_call = torch.compile(
        call,
        dynamic=False,
        fullgraph=True,
        mode=args.compile_mode,
    )
    with torch.inference_mode():
        eager = call(q, k, v)
        compiled = compiled_call(q, k, v)
        torch.cuda.synchronize(device)

    failed, total, max_abs, bitwise = strict_compare(
        eager,
        compiled,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        f"compiled_eager={'PASS' if failed == 0 else 'FAIL'} "
        f"failed={failed}/{total} max_abs={max_abs:.9g} bitwise={bitwise}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
