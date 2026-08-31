#!/usr/bin/env python3
"""V1 fused QKV with only the attention core replaced by PyTorch SDPA."""

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench


class SDPASelfAttention(bench.BaselineSelfAttention):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.register_buffer("_qkv_weight", torch.empty(0), persistent=False)
        self.register_buffer("_qkv_bias", torch.empty(0), persistent=False)
        self.refresh_qkv()

    def refresh_qkv(self) -> None:
        self._qkv_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        ).detach()
        self._qkv_bias = torch.cat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        ).detach()

    def _project_qkv(self, x):
        if self.training:
            return super()._project_qkv(x)
        return F.linear(x, self._qkv_weight, self._qkv_bias).chunk(3, dim=-1)

    def forward(self, x, valid_token_mask=None, causal=False):
        if x.dtype != torch.float32:
            return super().forward(x, valid_token_mask, causal)

        batch, seq_len, _ = x.shape
        q, k, v = map(self._split_heads, self._project_qkv(x))

        mask = None
        if valid_token_mask is not None:
            mask = valid_token_mask[:, None, None, :]
            if causal:
                mask = mask & torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=x.device
                ).tril()

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal and mask is None,
            scale=self.scale,
        )
        output = self.out_proj(
            context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        )
        return output if valid_token_mask is None else output.masked_fill(
            ~valid_token_mask[..., None], 0
        )


class UserOptimizedTransformer(bench.BaselineTransformer):
    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.attention = SDPASelfAttention(config.d_model, config.num_heads)

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        for layer in self.layers:
            layer.attention.refresh_qkv()
        return result


bench.UserOptimizedTransformer = UserOptimizedTransformer

if __name__ == "__main__":
    raise SystemExit(bench.main())
