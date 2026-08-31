#!/usr/bin/env python3
"""V3.1: no-copy packed-QKV SDPA without materialized causal masks."""

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench


class UserOptimizedTransformer(bench.BaselineTransformer):
    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            attention = layer.attention
            attention.register_buffer("_qkv_weight", torch.empty(0), persistent=False)
            attention.register_buffer("_qkv_bias", torch.empty(0), persistent=False)
        self._refresh_qkv()

    def _refresh_qkv(self) -> None:
        for layer in self.layers:
            attention = layer.attention
            attention._qkv_weight = torch.cat(
                (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)
            ).detach()
            attention._qkv_bias = torch.cat(
                (attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias)
            ).detach()

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_qkv()
        return result

    @staticmethod
    def _attention(attention, x, mask, causal):
        batch, seq_len, _ = x.shape
        q, k, v = F.linear(
            x, attention._qkv_weight, attention._qkv_bias
        ).reshape(
            batch, seq_len, 3, attention.num_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=attention.scale,
        )
        return attention.out_proj(
            context.transpose(1, 2).reshape(batch, seq_len, attention.d_model)
        )

    def forward(self, x, valid_token_mask=None):
        if self.training or x.dtype != torch.float32:
            return super().forward(x, valid_token_mask)

        mask = None
        invalid_queries = None
        if valid_token_mask is not None:
            mask = valid_token_mask[:, None, None, :]
            invalid_queries = ~valid_token_mask[..., None]

        for layer in self.layers:
            x = x + self._attention(
                layer.attention,
                layer.norm1(x),
                mask,
                self.config.causal,
            )
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            if invalid_queries is not None:
                x = x.masked_fill(invalid_queries, 0)

        x = self.final_norm(x)
        return x if invalid_queries is None else x.masked_fill(invalid_queries, 0)


bench.UserOptimizedTransformer = UserOptimizedTransformer

if __name__ == "__main__":
    raise SystemExit(bench.main())
