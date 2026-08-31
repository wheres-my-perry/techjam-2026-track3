#!/usr/bin/env python3
"""V8a: shape-dispatched Triton FFN-in GEMM + bias + exact-GELU epilogue.

The custom kernel preserves the V4.3 precision boundary: GEMM accumulates in
FP32, the biased Linear result is rounded to FP16, and exact/erf GELU is then
evaluated from that rounded value before the FP16 result is stored for FFN-out.
Everything outside FFN-in + GELU is inherited unchanged from V4.3.
The custom path is enabled only for large-token D=FFN=128 workloads where a
paired whole-model benchmark showed a repeatable gain; other shapes use V4.3.

Triton is optional so the module remains importable on the local CPU-only
development environment.  CPU and non-Triton calls use the exact PyTorch graph;
the optimized model already falls back to the reference path for training and
non-FP32 public inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v4_3_Flash import UserOptimizedTransformer as V43FlashTransformer

try:
    import triton
    import triton.language as tl
except ImportError:  # Local CPU development environment.
    triton = None
    tl = None


if triton is not None:
    _GEMM_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=5,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]

    @triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_ffn_gelu_kernel(
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
        # Weight has PyTorch Linear layout [N, K], while tl.dot needs [K, N].
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

        # Preserve F.linear's FP16 output rounding before exact GELU.
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


@torch.library.custom_op("techjam::fused_ffn_gelu", mutates_args=())
def fused_ffn_gelu(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply FP16 Linear + exact GELU, using Triton on CUDA."""

    if triton is None or not normalized.is_cuda:
        return F.gelu(
            F.linear(normalized, weight, bias),
            approximate="none",
        )

    if normalized.dtype != torch.float16:
        raise TypeError("fused_ffn_gelu requires FP16 CUDA activation")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("fused_ffn_gelu requires FP16 weight and bias")
    if not normalized.is_contiguous():
        raise ValueError("fused_ffn_gelu requires contiguous activation")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("fused_ffn_gelu requires contiguous weight and bias")

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
    _fused_ffn_gelu_kernel[grid](
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


@fused_ffn_gelu.register_fake
def _fused_ffn_gelu_fake(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return normalized.new_empty((*normalized.shape[:-1], weight.shape[0]))


class UserOptimizedTransformer(V43FlashTransformer):
    """V4.3 with shape-dispatched FFN-in Linear + exact GELU fusion."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        # Profiling shows the custom epilogue wins only when the D=FFN=128
        # activation has at least one million token rows (official shape #6).
        # Keep every other configuration on the already-tuned V4.3 graph.
        self._use_fused_ffn_gelu = (
            config.batch_size * config.seq_len >= 1_000_000
            and config.d_model == 128
            and config.ffn_dim == 128
        )

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        if not self._use_fused_ffn_gelu:
            return super()._mixed_ffn(layer, x)

        normalized = layer.norm2(x).to(dtype=self.internal_dtype)
        hidden = fused_ffn_gelu(
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
