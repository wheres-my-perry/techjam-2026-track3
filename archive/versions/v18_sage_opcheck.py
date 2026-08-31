#!/usr/bin/env python3
"""Non-benchmark integration preflight for V18-Sage's direct custom op."""

from __future__ import annotations

import argparse
import math

import torch

from v18_sage import (
    SAGE_ATTENTION_INSTALLED_VERSION,
    SAGE_ATTENTION_REQUIRED_VERSION,
    sage_attention_auto_v18,
    sageattn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check direct Sage eager/custom-op/compile integration"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.batch_size, args.heads, args.seq_len, args.head_dim) <= 0:
        raise SystemExit("shape dimensions must be positive")
    if args.head_dim > 128:
        raise SystemExit("SageAttention 2.2 supports original head_dim <= 128")
    if args.rtol < 0 or args.atol < 0:
        raise SystemExit("tolerances must be non-negative")
    if sageattn is None:
        raise SystemExit("sageattention could not be imported")
    if SAGE_ATTENTION_INSTALLED_VERSION != SAGE_ATTENTION_REQUIRED_VERSION:
        raise SystemExit(
            f"requires sageattention {SAGE_ATTENTION_REQUIRED_VERSION}, "
            f"found {SAGE_ATTENTION_INSTALLED_VERSION!r}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("V18-Sage opcheck requires an available CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(args.head_dim)
    example_args = (q, k, v, scale)

    print(
        f"sageattention={SAGE_ATTENTION_INSTALLED_VERSION} shape={shape} "
        f"mode={args.compile_mode}"
    )
    details = torch.library.opcheck(
        sage_attention_auto_v18,
        example_args,
        raise_exception=True,
    )
    print(f"opcheck=PASS details={details}")

    def call(
        q_value: torch.Tensor,
        k_value: torch.Tensor,
        v_value: torch.Tensor,
    ) -> torch.Tensor:
        return sage_attention_auto_v18(q_value, k_value, v_value, scale)

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

    absolute = (compiled.float() - eager.float()).abs()
    finite = bool(torch.isfinite(eager).all() and torch.isfinite(compiled).all())
    strict = torch.isfinite(eager) & torch.isfinite(compiled) & (
        (absolute < args.atol) | (absolute < args.rtol * eager.float().abs())
    )
    failed = int((~strict).sum().item())
    max_abs = float(absolute.max().item())
    bitwise = bool(torch.equal(eager, compiled))
    passed = finite and eager.shape == compiled.shape and failed == 0
    print(
        f"compiled_eager={'PASS' if passed else 'FAIL'} "
        f"failed={failed}/{eager.numel()} finite={finite} "
        f"max_abs={max_abs:.9g} bitwise={bitwise}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
