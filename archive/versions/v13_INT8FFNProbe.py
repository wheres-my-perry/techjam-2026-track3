#!/usr/bin/env python3
"""V13 accuracy-only INT8 FFN-in quantization probe.

This ablation inherits the promoted V11 graph and changes only FFN-in.  It
uses symmetric per-output-channel weight quantization and dynamic per-token
activation quantization, accumulates the W8A8 dot product in INT32, then
dequantizes to FP32 before the V11 exact-GELU epilogue.  GELU output remains
FP16 for the existing FFN-out path.

The implementation deliberately is not a performance candidate: quantization
and dequantization are expressed with ordinary PyTorch operations so numerical
viability can be tested before writing a Triton/CUTLASS kernel.  Do not report
its latency as INT8 performance.

Set TECHJAM_INT8_PROBE_MODE to one of:

* w8a8 (default): quantize both activation and FFN-in weight.
* w8: quantize only the FFN-in weight (accuracy ceiling/control).
* a8: quantize only the activation (error attribution control).

TECHJAM_INT8_PROBE_LAYERS defaults to ``all`` and may be a comma-separated
list of zero-based layer indices (for example ``3`` or ``0,2``) for scope
isolation.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v11_FP32PreGELU import UserOptimizedTransformer as V11Transformer


_VALID_PROBE_MODES = frozenset(("w8a8", "w8", "a8"))


def _nonzero_symmetric_scale(
    tensor: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Return an absmax/127 scale while preserving all-zero slices exactly."""

    absmax = tensor.abs().amax(dim=dim, keepdim=True)
    scale = absmax / 127.0
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _quantize_symmetric_int8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)


def _dynamic_per_token_quantize(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the final feature dimension independently for every token."""

    tensor_fp32 = tensor.float()
    scale = _nonzero_symmetric_scale(tensor_fp32, dim=-1)
    quantized = _quantize_symmetric_int8(tensor_fp32, scale)
    return quantized, scale


def _int8_linear_accumulator(
    activation_int8: torch.Tensor,
    weight_int8_t: torch.Tensor,
) -> torch.Tensor:
    """Compute exact signed INT8 products with an INT32 accumulator."""

    k = activation_int8.shape[-1]
    activation_2d = activation_int8.reshape(-1, k).contiguous()
    if weight_int8_t.shape[0] != k:
        raise ValueError("incompatible INT8 activation and weight shapes")
    return torch._int_mm(activation_2d, weight_int8_t)


class UserOptimizedTransformer(V11Transformer):
    """V11 with an accuracy-only fake-quantized FFN-in projection."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self.int8_probe_mode = os.environ.get(
            "TECHJAM_INT8_PROBE_MODE",
            "w8a8",
        ).lower()
        if self.int8_probe_mode not in _VALID_PROBE_MODES:
            choices = ", ".join(sorted(_VALID_PROBE_MODES))
            raise ValueError(
                "TECHJAM_INT8_PROBE_MODE must be one of "
                f"{choices}; got {self.int8_probe_mode!r}"
            )

        requested_layers = os.environ.get(
            "TECHJAM_INT8_PROBE_LAYERS",
            "all",
        ).lower()
        if requested_layers == "all":
            layer_indices = set(range(len(self.layers)))
        else:
            try:
                layer_indices = {
                    int(value.strip())
                    for value in requested_layers.split(",")
                    if value.strip()
                }
            except ValueError as error:
                raise ValueError(
                    "TECHJAM_INT8_PROBE_LAYERS must be 'all' or a "
                    "comma-separated list of zero-based integers"
                ) from error
            invalid_layers = sorted(
                index
                for index in layer_indices
                if index < 0 or index >= len(self.layers)
            )
            if not layer_indices or invalid_layers:
                raise ValueError(
                    "TECHJAM_INT8_PROBE_LAYERS selects no valid layers; "
                    f"invalid indices: {invalid_layers}"
                )
        self._int8_probe_layer_ids = frozenset(
            id(self.layers[index]) for index in layer_indices
        )

        for layer in self.layers:
            layer.register_buffer(
                "_ffn_in_weight_int8_t",
                torch.empty(0, dtype=torch.int8),
                persistent=False,
            )
            layer.register_buffer(
                "_ffn_in_weight_int8_scale",
                torch.empty(0, dtype=torch.float32),
                persistent=False,
            )
        self._refresh_int8_weights()

    @torch.no_grad()
    def _refresh_int8_weights(self) -> None:
        for layer in self.layers:
            weight_fp32 = layer.ffn_in.weight.detach().float()
            scale = _nonzero_symmetric_scale(weight_fp32, dim=1)
            weight_int8 = _quantize_symmetric_int8(weight_fp32, scale)
            # Cache [K, N] because torch._int_mm consumes A[M, K] @ B[K, N].
            layer._ffn_in_weight_int8_t = weight_int8.t().contiguous()
            layer._ffn_in_weight_int8_scale = scale.squeeze(1).contiguous()

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_int8_weights()
        return result

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_int8_weights()
        return result

    def _w8_linear_fp32(
        self,
        layer,
        normalized_fp16: torch.Tensor,
    ) -> torch.Tensor:
        weight_scale = layer._ffn_in_weight_int8_scale[:, None]
        weight_fp32 = (
            layer._ffn_in_weight_int8_t.t().float() * weight_scale
        )
        return F.linear(
            normalized_fp16.float(),
            weight_fp32,
            layer._ffn_in_bias_mixed.float(),
        )

    def _a8_linear_fp32(
        self,
        layer,
        normalized_fp16: torch.Tensor,
    ) -> torch.Tensor:
        activation_int8, activation_scale = _dynamic_per_token_quantize(
            normalized_fp16
        )
        activation_fp32 = activation_int8.float() * activation_scale
        return F.linear(
            activation_fp32,
            layer._ffn_in_weight_mixed.float(),
            layer._ffn_in_bias_mixed.float(),
        )

    def _w8a8_linear_fp32(
        self,
        layer,
        normalized_fp16: torch.Tensor,
    ) -> torch.Tensor:
        activation_int8, activation_scale = _dynamic_per_token_quantize(
            normalized_fp16
        )
        accumulator_int32 = _int8_linear_accumulator(
            activation_int8,
            layer._ffn_in_weight_int8_t,
        )

        output_features = layer._ffn_in_weight_int8_scale.numel()
        output_shape = (*normalized_fp16.shape[:-1], output_features)
        accumulator_fp32 = accumulator_int32.reshape(output_shape).float()
        dequant_scale = (
            activation_scale
            * layer._ffn_in_weight_int8_scale.view(
                *((1,) * (normalized_fp16.ndim - 1)),
                output_features,
            )
        )
        return (
            accumulator_fp32 * dequant_scale
            + layer._ffn_in_bias_mixed.float()
        )

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        if id(layer) not in self._int8_probe_layer_ids:
            return super()._mixed_ffn(layer, x)

        # Preserve V11's FP16 activation boundary before the FFN-in dot.
        normalized_fp16 = layer.norm2(x).to(dtype=self.internal_dtype)

        if self.int8_probe_mode == "w8":
            linear_fp32 = self._w8_linear_fp32(layer, normalized_fp16)
        elif self.int8_probe_mode == "a8":
            linear_fp32 = self._a8_linear_fp32(layer, normalized_fp16)
        else:
            linear_fp32 = self._w8a8_linear_fp32(layer, normalized_fp16)

        hidden = F.gelu(linear_fp32, approximate="none").to(
            dtype=self.internal_dtype
        )
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
