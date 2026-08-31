"""Standalone V4.1 mixed-precision Transformer implementation.

This module contains only configuration and model code. It deliberately has no
benchmark CLI or dependency on ``torch_transformer_benchmark``. Load weights,
move the model to the target device, call ``eval()``, then optionally wrap it in
``torch.compile`` from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


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

        # Parameter names intentionally match the official reference model.
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Detached FP16 inference caches move with the module but stay out of
        # state_dict, so strict official weight loading remains compatible.
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


class UserOptimizedTransformer(nn.Module):
    """FP32 norm/residual with FP16 QKV, SDPA, GEMMs, and GELU."""

    internal_dtype = torch.float16

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
            layer._ffn_in_weight_mixed = self._cast_cache(
                layer.ffn_in.weight
            )
            layer._ffn_in_bias_mixed = self._cast_cache(layer.ffn_in.bias)
            layer._ffn_out_weight_mixed = self._cast_cache(
                layer.ffn_out.weight
            )
            layer._ffn_out_bias_mixed = self._cast_cache(layer.ffn_out.bias)

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_mixed_weights()
        return result

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        # A caller commonly uses model.to(device, dtype=torch.float32), which
        # would otherwise cast every floating buffer back to FP32.
        self._refresh_mixed_weights()
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

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal,
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
        hidden = F.linear(
            layer.norm2(x).to(dtype=torch.float16),
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        hidden = F.gelu(hidden, approximate="none")
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

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Detached FP16 caches are inference-only. Training and models moved to
        # another public dtype use the exact reference-style path.
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
