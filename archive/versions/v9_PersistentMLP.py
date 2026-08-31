#!/usr/bin/env python3
"""V9a: fully fused FP16 FFN-in + exact GELU + FFN-out Triton kernel.

The kernel keeps FFN hidden tiles on-chip and never materializes the hidden
activation. Both GEMMs accumulate in FP32. It explicitly preserves V4.3's
three FP16 rounding boundaries: after FFN-in+bias, after exact GELU, and after
FFN-out+bias. Unsupported dimensions use the unchanged V4.3 implementation.
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
    _PERSISTENT_MLP_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 32, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 64, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 128, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 64, "BLOCK_F": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_D": 128, "BLOCK_F": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_D": 32, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_D": 64, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_D": 128, "BLOCK_F": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=8,
        ),
    ]

    @triton.autotune(configs=_PERSISTENT_MLP_CONFIGS, key=["M", "D", "FFN"])
    @triton.jit
    def _persistent_mlp_kernel(
        x_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        output_ptr,
        M,
        D: tl.constexpr,
        FFN: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xd: tl.constexpr,
        stride_w1f: tl.constexpr,
        stride_w1d: tl.constexpr,
        stride_w2d: tl.constexpr,
        stride_w2f: tl.constexpr,
        stride_om: tl.constexpr,
        stride_od: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(axis=0)
        program_d = tl.program_id(axis=1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_d = program_d * BLOCK_D + tl.arange(0, BLOCK_D)

        output_accumulator = tl.zeros(
            (BLOCK_M, BLOCK_D),
            dtype=tl.float32,
        )

        for f_start in range(0, FFN, BLOCK_F):
            offsets_f = f_start + tl.arange(0, BLOCK_F)
            hidden_accumulator = tl.zeros(
                (BLOCK_M, BLOCK_F),
                dtype=tl.float32,
            )

            for k_start in range(0, D, BLOCK_K):
                offsets_k = k_start + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr
                    + offsets_m[:, None] * stride_xm
                    + offsets_k[None, :] * stride_xd,
                    mask=(offsets_m[:, None] < M)
                    & (offsets_k[None, :] < D),
                    other=0.0,
                )
                # w1 has PyTorch Linear layout [FFN, D].
                w1 = tl.load(
                    w1_ptr
                    + offsets_k[:, None] * stride_w1d
                    + offsets_f[None, :] * stride_w1f,
                    mask=(offsets_k[:, None] < D)
                    & (offsets_f[None, :] < FFN),
                    other=0.0,
                )
                hidden_accumulator = tl.dot(x, w1, hidden_accumulator)

            b1 = tl.load(
                b1_ptr + offsets_f,
                mask=offsets_f < FFN,
                other=0.0,
            )
            # Preserve FFN-in FP16 output rounding before exact GELU.
            hidden_fp16 = (hidden_accumulator + b1[None, :]).to(tl.float16)
            hidden_fp32 = hidden_fp16.to(tl.float32)
            gelu_fp32 = 0.5 * hidden_fp32 * (
                1.0 + tl.erf(hidden_fp32 * 0.7071067811865476)
            )
            # Preserve GELU's FP16 output rounding before FFN-out.
            gelu_fp16 = gelu_fp32.to(tl.float16)

            # w2 has PyTorch Linear layout [D, FFN].
            w2 = tl.load(
                w2_ptr
                + offsets_f[:, None] * stride_w2f
                + offsets_d[None, :] * stride_w2d,
                mask=(offsets_f[:, None] < FFN)
                & (offsets_d[None, :] < D),
                other=0.0,
            )
            output_accumulator = tl.dot(
                gelu_fp16,
                w2,
                output_accumulator,
            )

        b2 = tl.load(
            b2_ptr + offsets_d,
            mask=offsets_d < D,
            other=0.0,
        )
        # Preserve FFN-out FP16 output rounding before the FP32 residual path.
        output_fp16 = (output_accumulator + b2[None, :]).to(tl.float16)
        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_om
            + offsets_d[None, :] * stride_od,
            output_fp16,
            mask=(offsets_m[:, None] < M) & (offsets_d[None, :] < D),
        )


@torch.library.custom_op("techjam::persistent_mlp", mutates_args=())
def persistent_mlp(
    normalized: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Apply two FP16 Linear layers with exact GELU and no hidden materialization."""

    if triton is None or not normalized.is_cuda:
        hidden = F.gelu(F.linear(normalized, w1, b1), approximate="none")
        return F.linear(hidden, w2, b2)

    tensors = (normalized, w1, b1, w2, b2)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError("persistent_mlp requires FP16 activation and parameters")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("persistent_mlp requires contiguous inputs")

    d = normalized.shape[-1]
    ffn = w1.shape[0]
    if (
        w1.shape[1] != d
        or b1.numel() != ffn
        or w2.shape != (d, ffn)
        or b2.numel() != d
    ):
        raise ValueError("incompatible persistent MLP shapes")
    if d > 128 or ffn > 128 or d % 16 or ffn % 16:
        raise ValueError("persistent_mlp supports D/FFN multiples of 16 up to 128")

    output = torch.empty_like(normalized)
    x_2d = normalized.reshape(-1, d)
    output_2d = output.reshape(-1, d)
    m = x_2d.shape[0]
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(d, meta["BLOCK_D"]),
    )
    _persistent_mlp_kernel[grid](
        x_2d,
        w1,
        b1,
        w2,
        b2,
        output_2d,
        m,
        d,
        ffn,
        x_2d.stride(0),
        x_2d.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
    )
    return output


@persistent_mlp.register_fake
def _persistent_mlp_fake(
    normalized: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    del w1, b1, w2, b2
    return torch.empty_like(normalized)


class UserOptimizedTransformer(V43FlashTransformer):
    """V4.3 with a general persistent full-MLP ablation for small dimensions."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_persistent_mlp = (
            config.d_model <= 128
            and config.ffn_dim <= 128
            and config.d_model % 16 == 0
            and config.ffn_dim % 16 == 0
        )

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        if not self._use_persistent_mlp:
            return super()._mixed_ffn(layer, x)

        normalized = layer.norm2(x).to(dtype=self.internal_dtype)
        return persistent_mlp(
            normalized,
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
