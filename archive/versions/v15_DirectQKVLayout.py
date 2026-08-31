#!/usr/bin/env python3
"""V15: direct-layout QKV projection for official attention-heavy shape #13.

The packed V14.1/V11 projection stores ``[B, S, 3D]`` and exposes Q, K and V
as no-copy views.  Although this avoids a materialized transpose, consecutive
sequence rows of each Q/K/V tensor remain separated by ``3D`` elements.  This
ablation replaces only that projection on exact official shape #13 with a
Triton FP16 GEMM whose FP32 accumulator epilogue writes directly to contiguous
``[3, B, H, S, Dh]`` storage consumed by Flash Attention.

All other arithmetic, cache lifecycle and fallbacks are inherited from V14.1.
Triton is optional so CPU correctness diagnostics can materialize the same
layout with regular PyTorch operations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

import torch_transformer_benchmark as bench
from v14_1_BatchChunked import UserOptimizedTransformer as V141Transformer
from v4_3_Flash import FLASH_FIRST_BACKENDS

try:
    import triton
    import triton.language as tl
except ImportError:  # Local CPU development environment.
    triton = None
    tl = None


if triton is not None:
    # Keep the search focused on the exact Mx128 @ 128x384 projection used by
    # shape #13.  A shorter list also bounds first-run autotune cost.
    _DIRECT_QKV_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
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
    ]

    @triton.autotune(configs=_DIRECT_QKV_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _direct_qkv_projection_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        B: tl.constexpr,
        S: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
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
        # PyTorch Linear weight is [N, K]; tl.dot consumes [K, N].
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

        bias = tl.load(
            bias_ptr + offsets_n,
            mask=offsets_n < N,
            other=0.0,
        )
        projected = (accumulator + bias[None, :]).to(tl.float16)

        # Logical GEMM column n maps to qkv=n//D and channel=n%D.  The store
        # permutation makes each returned [B,H,S,Dh] operand fully contiguous.
        token = offsets_m[:, None]
        batch = token // S
        sequence = token - batch * S
        qkv = offsets_n[None, :] // K
        channel = offsets_n[None, :] - qkv * K
        head = channel // HEAD_DIM
        head_channel = channel - head * HEAD_DIM
        output_offsets = (
            ((((qkv * B + batch) * NUM_HEADS + head) * S + sequence)
            * HEAD_DIM)
            + head_channel
        )
        tl.store(
            output_ptr + output_offsets,
            projected,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


@torch.library.custom_op("techjam::direct_qkv_projection", mutates_args=())
def direct_qkv_projection(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Return Q/K/V in contiguous ``[3,B,H,S,Dh]`` storage."""

    if normalized.ndim != 3:
        raise ValueError("direct_qkv_projection requires [B,S,D] activation")
    batch, seq_len, d_model = normalized.shape
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if weight.shape != (3 * d_model, d_model):
        raise ValueError("packed QKV weight must have shape [3D,D]")
    if bias.shape != (3 * d_model,):
        raise ValueError("packed QKV bias must have shape [3D]")

    head_dim = d_model // num_heads
    if triton is None or not normalized.is_cuda:
        packed = F.linear(normalized, weight, bias)
        return (
            packed.reshape(batch, seq_len, 3, num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

    if normalized.dtype != torch.float16:
        raise TypeError("direct_qkv_projection requires FP16 CUDA activation")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("direct_qkv_projection requires FP16 weight and bias")
    if not normalized.is_contiguous():
        raise ValueError("direct_qkv_projection requires contiguous activation")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("direct_qkv_projection requires contiguous weight/bias")

    output = torch.empty(
        (3, batch, num_heads, seq_len, head_dim),
        device=normalized.device,
        dtype=torch.float16,
    )
    normalized_2d = normalized.reshape(-1, d_model)
    m = normalized_2d.shape[0]
    n = 3 * d_model
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"])
        * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _direct_qkv_projection_kernel[grid](
        normalized_2d,
        weight,
        bias,
        output,
        m,
        n,
        d_model,
        batch,
        seq_len,
        num_heads,
        head_dim,
        normalized_2d.stride(0),
        normalized_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
    )
    return output


@direct_qkv_projection.register_fake
def _direct_qkv_projection_fake(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    del weight, bias
    head_dim = normalized.shape[-1] // num_heads
    return normalized.new_empty(
        (3, normalized.shape[0], num_heads, normalized.shape[1], head_dim)
    )


class UserOptimizedTransformer(V141Transformer):
    """V14.1 plus a direct QKV layout ablation for exact official #13."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._use_direct_qkv_layout = (
            config.batch_size == 64
            and config.seq_len == 1024
            and config.d_model == 128
            and config.num_heads == 4
            and config.num_layers == 4
            and config.ffn_dim == 128
            and config.causal
        )

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self._use_direct_qkv_layout:
            return super()._mixed_attention(attention, x, mask)

        normalized = x.to(dtype=self.internal_dtype)
        q, k, v = direct_qkv_projection(
            normalized,
            attention._qkv_weight_mixed,
            attention._qkv_bias_mixed,
            attention.num_heads,
        ).unbind(0)

        with sdpa_kernel(backends=FLASH_FIRST_BACKENDS, set_priority=True):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=attention.scale,
            )
        context = context.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], attention.d_model
        )
        return F.linear(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
