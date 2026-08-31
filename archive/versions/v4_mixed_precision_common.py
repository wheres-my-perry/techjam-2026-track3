"""Shared implementation for V4 FP16/BF16 internal-compute candidates."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench


class MixedPrecisionTransformer(bench.BaselineTransformer):
    """Keep normalization/residuals in FP32 and run heavy kernels lower precision."""

    internal_dtype: torch.dtype
    gelu_internal_dtype = False

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        if self.internal_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("internal_dtype must be torch.float16 or torch.bfloat16")

        for layer in self.layers:
            attention = layer.attention
            attention.register_buffer(
                "_qkv_weight_mixed", torch.empty(0), persistent=False
            )
            attention.register_buffer(
                "_qkv_bias_mixed", torch.empty(0), persistent=False
            )
            attention.register_buffer(
                "_out_weight_mixed", torch.empty(0), persistent=False
            )
            attention.register_buffer(
                "_out_bias_mixed", torch.empty(0), persistent=False
            )
            layer.register_buffer(
                "_ffn_in_weight_mixed", torch.empty(0), persistent=False
            )
            layer.register_buffer(
                "_ffn_in_bias_mixed", torch.empty(0), persistent=False
            )
            layer.register_buffer(
                "_ffn_out_weight_mixed", torch.empty(0), persistent=False
            )
            layer.register_buffer(
                "_ffn_out_bias_mixed", torch.empty(0), persistent=False
            )
        self._refresh_mixed_weights()

    @staticmethod
    def _cast_cache(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.detach().to(dtype=dtype)

    def _refresh_mixed_weights(self) -> None:
        dtype = self.internal_dtype
        for layer in self.layers:
            attention = layer.attention
            attention._qkv_weight_mixed = self._cast_cache(
                torch.cat(
                    (
                        attention.q_proj.weight,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    )
                ),
                dtype,
            )
            attention._qkv_bias_mixed = self._cast_cache(
                torch.cat(
                    (
                        attention.q_proj.bias,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    )
                ),
                dtype,
            )
            attention._out_weight_mixed = self._cast_cache(
                attention.out_proj.weight, dtype
            )
            attention._out_bias_mixed = self._cast_cache(
                attention.out_proj.bias, dtype
            )
            layer._ffn_in_weight_mixed = self._cast_cache(
                layer.ffn_in.weight, dtype
            )
            layer._ffn_in_bias_mixed = self._cast_cache(
                layer.ffn_in.bias, dtype
            )
            layer._ffn_out_weight_mixed = self._cast_cache(
                layer.ffn_out.weight, dtype
            )
            layer._ffn_out_bias_mixed = self._cast_cache(
                layer.ffn_out.bias, dtype
            )

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_mixed_weights()
        return result

    def _apply(self, fn, recurse: bool = True):
        # Module.to() casts every floating buffer to the requested model dtype.
        # Rebuild the inference cache afterwards so it remains FP16/BF16 while
        # the public parameters, LayerNorms and residual path stay FP32.
        result = super()._apply(fn, recurse=recurse)
        self._refresh_mixed_weights()
        return result

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = (
            F.linear(
                x.to(dtype=self.internal_dtype),
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
            is_causal=self.config.causal,
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

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(
            layer.norm2(x).to(dtype=self.internal_dtype),
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
        )
        if self.gelu_internal_dtype:
            hidden = F.gelu(hidden, approximate="none")
        else:
            # V4 reference ablation keeps GELU in FP32, including the casts on
            # both sides. V4.1 overrides the class flag to remove that round-trip.
            hidden = F.gelu(hidden.float(), approximate="none")
            hidden = hidden.to(dtype=self.internal_dtype)
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training or x.dtype != torch.float32:
            return super().forward(x, valid_token_mask)

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
            )
            x = x + attention_output
            x = x + self._mixed_ffn(layer, x)
            if invalid_queries is not None:
                x = x.masked_fill(invalid_queries, 0)

        x = self.final_norm(x)
        if invalid_queries is not None:
            x = x.masked_fill(invalid_queries, 0)
        return x
