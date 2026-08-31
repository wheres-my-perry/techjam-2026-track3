#!/usr/bin/env python3
"""V10a: persistent-CTA FFN-in GEMM + exact-GELU for tall-skinny workloads.

V8 already showed that replacing FFN-in Linear + exact GELU is beneficial on
official shape #6. V10 changes only that kernel's scheduling: a bounded set of
CTAs keeps one FFN-in weight tile live while looping over many token-row tiles.
FFN-out remains the existing PyTorch/CUTLASS GEMM because V9 showed that a
custom second GEMM loses the whole-model benefit.

The optimized path preserves V8/V4.3 precision boundaries: FP16 inputs and
weights, FP32 dot accumulation, FP16 rounding after Linear+bias, exact erf-GELU
from the rounded value, and an FP16 hidden output for FFN-out. Other shapes,
training, and non-FP32 public inputs retain the inherited safe fallbacks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v8_FusedFFNGELU import UserOptimizedTransformer as V8Transformer

try:
    import triton
    import triton.language as tl
except ImportError:  # Local CPU development environment.
    triton = None
    tl = None


if triton is not None:
    _PERSISTENT_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=1, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=8
        ),
    ]

    @triton.autotune(
        configs=_PERSISTENT_CONFIGS,
        key=["M", "N", "K", "NUM_WORKERS"],
    )
    @triton.jit
    def _persistent_ffn_gelu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        NUM_WORKERS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        program_id = tl.program_id(axis=0)
        programs_n = tl.cdiv(N, BLOCK_N)
        worker_id = program_id // programs_n
        program_n = program_id % programs_n

        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, K)

        # Shape #6 has K=128. Load one weight tile before the persistent M loop
        # so the CTA can reuse it across many token tiles.
        weight_ptrs = (
            weight_ptr
            + offsets_k[:, None] * stride_wk
            + offsets_n[None, :] * stride_wn
        )
        weight = tl.load(
            weight_ptrs,
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < N, other=0.0)

        programs_m = tl.cdiv(M, BLOCK_M)
        for program_m in tl.range(worker_id, programs_m, NUM_WORKERS):
            offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
            x_ptrs = (
                x_ptr
                + offsets_m[:, None] * stride_xm
                + offsets_k[None, :] * stride_xk
            )
            x = tl.load(
                x_ptrs,
                mask=offsets_m[:, None] < M,
                other=0.0,
            )
            accumulator = tl.dot(x, weight, out_dtype=tl.float32)

            # Match F.linear's FP16 output rounding before exact GELU.
            linear_fp16 = (accumulator + bias[None, :]).to(tl.float16)
            linear_fp32 = linear_fp16.to(tl.float32)
            gelu = 0.5 * linear_fp32 * (
                1.0 + tl.erf(linear_fp32 * 0.7071067811865476)
            )

            output_ptrs = (
                output_ptr
                + offsets_m[:, None] * stride_om
                + offsets_n[None, :] * stride_on
            )
            tl.store(
                output_ptrs,
                gelu.to(tl.float16),
                mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
            )


@torch.library.custom_op("techjam::persistent_ffn_gelu", mutates_args=())
def persistent_ffn_gelu(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply persistent FP16 FFN-in Linear + exact GELU on eligible CUDA input."""

    if triton is None or not normalized.is_cuda:
        return F.gelu(F.linear(normalized, weight, bias), approximate="none")

    if normalized.dtype != torch.float16:
        raise TypeError("persistent_ffn_gelu requires FP16 CUDA activation")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("persistent_ffn_gelu requires FP16 weight and bias")
    if not normalized.is_contiguous():
        raise ValueError("persistent_ffn_gelu requires contiguous activation")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("persistent_ffn_gelu requires contiguous weight and bias")

    k = normalized.shape[-1]
    n = weight.shape[0]
    if k != 128 or n != 128 or weight.shape[1] != 128 or bias.numel() != 128:
        raise ValueError("persistent_ffn_gelu supports only K=N=128")

    output = torch.empty(
        (*normalized.shape[:-1], n),
        device=normalized.device,
        dtype=torch.float16,
    )
    normalized_2d = normalized.reshape(-1, k)
    output_2d = output.reshape(-1, n)
    m = normalized_2d.shape[0]
    num_workers = min(
        torch.cuda.get_device_properties(normalized.device).multi_processor_count,
        m,
    )

    grid = lambda meta: (
        num_workers * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _persistent_ffn_gelu_kernel[grid](
        normalized_2d,
        weight,
        bias,
        output_2d,
        m,
        n,
        k,
        normalized_2d.stride(0),
        normalized_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        num_workers,
    )
    return output


@persistent_ffn_gelu.register_fake
def _persistent_ffn_gelu_fake(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return normalized.new_empty((*normalized.shape[:-1], weight.shape[0]))


class UserOptimizedTransformer(V8Transformer):
    """V8 with persistent scheduling for its large-token FFN-in/GELU path."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_persistent_ffn_gelu = self._use_fused_ffn_gelu

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        if not self._use_persistent_ffn_gelu:
            return super()._mixed_ffn(layer, x)

        normalized = layer.norm2(x).to(dtype=self.internal_dtype)
        hidden = persistent_ffn_gelu(
            normalized,
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
