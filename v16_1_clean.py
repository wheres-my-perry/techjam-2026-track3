#!/usr/bin/env python3
"""Standalone source-clean V16.1 Transformer implementation.

This module contains the complete promoted model implementation and depends
only on PyTorch plus optional Triton.  It does not import the benchmark harness
or any earlier ``v*.py`` implementation.

The optimized FP32-inference path keeps LayerNorm, residuals and public output
in FP32; runs QKV, attention, projections and FFN GEMMs with FP16 operands;
evaluates exact GELU from the FP32 FFN-in accumulator; and uses Flash-first
SDPA for causal right-padded inputs.  FP32 eval with ``B > 1`` and
``S >= 8192`` uses a memory-bounded eager outer loop around one lazily compiled
sample executor.  Training and non-FP32 public inputs use reference arithmetic.

Callers may pass either this module's ``TransformerConfig`` or another config
object exposing the same validated fields.  Parameter names intentionally
match the official reference model for strict ``state_dict`` compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only development and portability fallback.
    triton = None
    tl = None


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


FLASH_FIRST_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.register_buffer(
            "_qkv_weight_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_qkv_bias_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_out_weight_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_out_bias_mixed", torch.empty(0), persistent=False
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

        self.register_buffer(
            "_ffn_in_weight_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_ffn_in_bias_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_ffn_out_weight_mixed", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_ffn_out_bias_mixed", torch.empty(0), persistent=False
        )


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
    "techjam::v16_1_clean_fused_ffn_gelu_no_preround",
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
        raise TypeError("fused FFN/GELU requires FP16 CUDA activation")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("fused FFN/GELU requires FP16 weight and bias")
    if not normalized.is_contiguous():
        raise ValueError("fused FFN/GELU requires contiguous activation")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("fused FFN/GELU requires contiguous weight and bias")

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


class UserOptimizedTransformer(nn.Module):
    """Standalone promoted V16.1 model with no project-internal imports."""

    internal_dtype = torch.float16
    _LARGE_SEQUENCE_CUTOFF = 8_192
    _LARGE_SEQUENCE_BATCH_CHUNK = 1

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self._large_sequence_compile_enabled = True
        self._large_sequence_compile_backend: Optional[str] = None
        self._large_sequence_compile_mode: Optional[str] = "max-autotune"
        self._large_sequence_compile_fullgraph = False
        self.__dict__["_compiled_large_sequence_executor"] = None
        self._refresh_mixed_weights()

    @staticmethod
    def _cast_cache(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(dtype=torch.float16)

    def _refresh_mixed_weights(self) -> None:
        for layer in self.layers:
            attention = layer.attention
            attention._qkv_weight_mixed = self._cast_cache(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    )
                )
            )
            attention._qkv_bias_mixed = self._cast_cache(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    )
                )
            )
            attention._out_weight_mixed = self._cast_cache(
                attention.out_proj.weight
            )
            attention._out_bias_mixed = self._cast_cache(
                attention.out_proj.bias
            )
            layer._ffn_in_weight_mixed = self._cast_cache(layer.ffn_in.weight)
            layer._ffn_in_bias_mixed = self._cast_cache(layer.ffn_in.bias)
            layer._ffn_out_weight_mixed = self._cast_cache(layer.ffn_out.weight)
            layer._ffn_out_bias_mixed = self._cast_cache(layer.ffn_out.bias)

    def _invalidate_large_sequence_executor(self) -> None:
        self.__dict__["_compiled_large_sequence_executor"] = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_mixed_weights()
        self._invalidate_large_sequence_executor()
        return result

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_mixed_weights()
        self._invalidate_large_sequence_executor()
        return result

    def train(self, mode: bool = True):
        changed = self.training != mode
        result = super().train(mode)
        if changed:
            self._invalidate_large_sequence_executor()
        return result

    @staticmethod
    def _mixed_attention(
        attention: _SelfAttention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = (
            F.linear(
                x.to(dtype=torch.float16),
                attention._qkv_weight_mixed,
                attention._qkv_bias_mixed,
            )
            .reshape(
                batch,
                seq_len,
                3,
                attention.num_heads,
                attention.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        if causal:
            with sdpa_kernel(
                backends=FLASH_FIRST_BACKENDS,
                set_priority=True,
            ):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=attention.scale,
                )
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attention.scale,
            )

        context = context.transpose(1, 2).reshape(
            batch, seq_len, attention.d_model
        )
        return F.linear(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        ).float()

    @staticmethod
    def _mixed_ffn(layer: _TransformerBlock, x: torch.Tensor) -> torch.Tensor:
        normalized = layer.norm2(x).to(dtype=torch.float16)
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

    @staticmethod
    def _fallback_attention(
        attention: _SelfAttention,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = attention.split_heads(attention.q_proj(x))
        k = attention.split_heads(attention.k_proj(x))
        v = attention.split_heads(attention.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * attention.scale
        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len),
                device=x.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if valid_token_mask is not None:
            scores = scores.masked_fill(
                ~valid_token_mask[:, None, None, :], float("-inf")
            )

        probabilities = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        context = torch.matmul(probabilities, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, attention.d_model)
        )
        output = attention.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output

    def _fallback_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = x + self._fallback_attention(
                layer.attention,
                layer.norm1(x),
                valid_token_mask,
                self.config.causal,
            )
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)

        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _forward_impl(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.training or x.dtype != torch.float32:
            return self._fallback_forward(x, valid_token_mask)

        mask = None
        invalid_queries = None
        if valid_token_mask is not None:
            mask = valid_token_mask[:, None, None, :]
            invalid_queries = ~valid_token_mask[..., None]

        for layer in self.layers:
            attention_output = self._mixed_attention(
                layer.attention,
                layer.norm1(x),
                mask,
                self.config.causal,
            )
            x = x + attention_output
            x = x + self._mixed_ffn(layer, x)
            if invalid_queries is not None:
                x = x.masked_fill(invalid_queries, 0)

        x = self.final_norm(x)
        if invalid_queries is not None:
            x = x.masked_fill(invalid_queries, 0)
        return x

    def _uses_large_sequence_chunking(self, x: torch.Tensor) -> bool:
        return (
            not self.training
            and x.dtype == torch.float32
            and x.ndim == 3
            and x.shape[0] > 1
            and x.shape[1] >= self._LARGE_SEQUENCE_CUTOFF
        )

    def configure_large_sequence_executor(
        self,
        *,
        enabled: bool = True,
        backend: Optional[str] = None,
        mode: Optional[str] = "max-autotune",
        fullgraph: bool = False,
    ) -> None:
        if backend is not None and mode is not None and backend != "inductor":
            raise ValueError("mode must be None for non-Inductor backends")
        self._large_sequence_compile_enabled = enabled
        self._large_sequence_compile_backend = backend
        self._large_sequence_compile_mode = mode
        self._large_sequence_compile_fullgraph = fullgraph
        self._invalidate_large_sequence_executor()

    @property
    def large_sequence_executor_ready(self) -> bool:
        return self.__dict__.get("_compiled_large_sequence_executor") is not None

    def _forward_large_sequence_chunk(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._forward_impl(x, valid_token_mask)

    def prepare_large_sequence_executor(self) -> Any:
        if self.training:
            raise RuntimeError("large-sequence executor requires eval mode")
        if not self._large_sequence_compile_enabled:
            return self._forward_large_sequence_chunk
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")

        executor = self.__dict__.get("_compiled_large_sequence_executor")
        if executor is not None:
            return executor

        compile_kwargs: dict[str, Any] = {
            "dynamic": False,
            "fullgraph": self._large_sequence_compile_fullgraph,
        }
        if self._large_sequence_compile_backend is not None:
            compile_kwargs["backend"] = self._large_sequence_compile_backend
        if self._large_sequence_compile_mode is not None:
            compile_kwargs["mode"] = self._large_sequence_compile_mode
        executor = torch.compile(
            self._forward_large_sequence_chunk,
            **compile_kwargs,
        )
        self.__dict__["_compiled_large_sequence_executor"] = executor
        return executor

    def forward_large_sequence_sample(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training or x.dtype != torch.float32:
            raise ValueError("compiled sample path requires FP32 eval mode")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("compiled sample path requires shape [1,S,D]")
        if x.shape[1] < self._LARGE_SEQUENCE_CUTOFF:
            raise ValueError("sequence length is below the large-sequence cutoff")
        if valid_token_mask is not None and valid_token_mask.shape != x.shape[:2]:
            raise ValueError("valid_token_mask must have shape [1,S]")
        return self.prepare_large_sequence_executor()(x, valid_token_mask)

    @torch.compiler.disable
    def _forward_large_sequence(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(x)
        chunk_size = self._LARGE_SEQUENCE_BATCH_CHUNK
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            chunk_mask = (
                None
                if valid_token_mask is None
                else valid_token_mask[start:end]
            )
            output[start:end].copy_(
                self.forward_large_sequence_sample(x[start:end], chunk_mask)
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._uses_large_sequence_chunking(x):
            return self._forward_large_sequence(x, valid_token_mask)
        return self._forward_impl(x, valid_token_mask)
