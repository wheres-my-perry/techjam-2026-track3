#!/usr/bin/env python3
"""V11 main: exact GELU directly from the FP32 FFN-in accumulator.

V8.1 fuses FFN-in Linear and exact GELU but deliberately rounds the biased
Linear result to FP16 before evaluating GELU.  This implementation removes that
pre-GELU rounding boundary: activation, weight and bias remain FP16, the dot
product accumulates in FP32, exact GELU consumes the FP32 accumulator plus
bias, and only the GELU output is stored as FP16 for FFN-out.

The custom path is forced for every mixed-precision inference shape.  V11 was
promoted as the main candidate after its V8.1 accuracy/latency ablation;
training and non-FP32 public inputs retain the inherited reference fallback.
Triton is optional for local CPU diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
import v8_FusedFFNGELU as v8


triton = v8.triton
tl = v8.tl


if triton is not None:

    @triton.autotune(configs=v8._GEMM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_ffn_gelu_no_preround_kernel(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        program_id = tl.program_id(axis=0)
        programs_m = tl.cdiv(M, BLOCK_M)
        programs_n = tl.cdiv(N, BLOCK_N)
        programs_per_group = GROUP_M * programs_n
        group_id = program_id // programs_per_group
        first_program_m = group_id * GROUP_M
        group_m = min(programs_m - first_program_m, GROUP_M)
        program_m = first_program_m + (program_id % group_m)
        program_n = (program_id % programs_per_group) // group_m

        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)

        x_ptrs = (
            x_ptr
            + offsets_m[:, None] * stride_xm
            + offsets_k[None, :] * stride_xk
        )
        # PyTorch Linear stores weight as [N, K]; tl.dot consumes [K, N].
        weight_ptrs = (
            weight_ptr
            + offsets_k[:, None] * stride_wk
            + offsets_n[None, :] * stride_wn
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            remaining_k = K - k_block * BLOCK_K
            x = tl.load(
                x_ptrs,
                mask=(offsets_m[:, None] < M)
                & (offsets_k[None, :] < remaining_k),
                other=0.0,
            )
            weight = tl.load(
                weight_ptrs,
                mask=(offsets_k[:, None] < remaining_k)
                & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(x, weight, accumulator)
            x_ptrs += BLOCK_K * stride_xk
            weight_ptrs += BLOCK_K * stride_wk

        bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < N, other=0.0)

        # V11's only arithmetic change: exact GELU consumes the FP32 Linear
        # accumulator directly.  The output is still rounded once to FP16 for
        # the existing FFN-out Tensor Core GEMM.
        linear_fp32 = accumulator + bias[None, :]
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


@torch.library.custom_op(
    "techjam::fused_ffn_gelu_no_preround",
    mutates_args=(),
)
def fused_ffn_gelu_no_preround(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply FP16 FFN-in and exact GELU without a pre-GELU FP16 round."""

    if triton is None or not normalized.is_cuda:
        linear_fp32 = F.linear(
            normalized.float(),
            weight.float(),
            bias.float(),
        )
        return F.gelu(linear_fp32, approximate="none").to(normalized.dtype)

    if normalized.dtype != torch.float16:
        raise TypeError(
            "fused_ffn_gelu_no_preround requires FP16 CUDA activation"
        )
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError(
            "fused_ffn_gelu_no_preround requires FP16 weight and bias"
        )
    if not normalized.is_contiguous():
        raise ValueError(
            "fused_ffn_gelu_no_preround requires contiguous activation"
        )
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError(
            "fused_ffn_gelu_no_preround requires contiguous weight and bias"
        )

    k = normalized.shape[-1]
    n = weight.shape[0]
    if weight.shape[1] != k or bias.numel() != n:
        raise ValueError("incompatible activation, weight or bias shape")

    output = torch.empty(
        (*normalized.shape[:-1], n),
        device=normalized.device,
        dtype=torch.float16,
    )
    normalized_2d = normalized.reshape(-1, k)
    output_2d = output.reshape(-1, n)
    m = normalized_2d.shape[0]

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"])
        * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _fused_ffn_gelu_no_preround_kernel[grid](
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
    )
    return output


@fused_ffn_gelu_no_preround.register_fake
def _fused_ffn_gelu_no_preround_fake(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return normalized.new_empty((*normalized.shape[:-1], weight.shape[0]))


class UserOptimizedTransformer(v8.UserOptimizedTransformer):
    """Promoted V8.1-style force-all path with FP32 input to exact GELU."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_fused_ffn_gelu = True

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        normalized = layer.norm2(x).to(dtype=self.internal_dtype)
        hidden = fused_ffn_gelu_no_preround(
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
