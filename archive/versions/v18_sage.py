#!/usr/bin/env python3
"""V18-Sage: direct automatic SageAttention performance probe on V16.1.

This owner-requested candidate is distinct from the historical
``v18_SageAttentionShape14.py`` accuracy experiment.  It inherits the
source-clean V16.1 main and changes only eligible attention calls:

* ``sageattention.sageattn`` selects the native automatic recipe.  On SM120
  SageAttention 2.2.0 dispatches INT8-QK per-warp with FP8-PV accumulation.
* There is no exact-prefix repair and no FP32 attention-output projection.
  This deliberately measures the direct Sage path rather than accuracy fixes.
* Original head dimensions above 128 are unsupported by SageAttention 2.2.0,
  so official shape #8 and other unsupported configurations fall back to
  V16.1 unchanged.

This is a performance-only ablation.  Timing an accuracy failure is diagnostic
and cannot be used as a valid competition result.  Set
``TECHJAM_SAGE_REQUIRE=1`` on GPU runs to reject a missing or wrong-version
dependency instead of silently measuring the V16.1 fallback.
"""

from __future__ import annotations

import os
from importlib import metadata
from typing import Optional

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as bench
from v16_1_NoDirectQKV13 import UserOptimizedTransformer as V161Transformer


SAGE_ATTENTION_SOURCE_COMMIT = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SAGE_ATTENTION_REQUIRED_VERSION = "2.2.0"
MAX_SAGE_HEAD_DIM = 128

try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

try:
    SAGE_ATTENTION_INSTALLED_VERSION: Optional[str] = metadata.version(
        "sageattention"
    )
except metadata.PackageNotFoundError:
    SAGE_ATTENTION_INSTALLED_VERSION = None


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 0/1, false/true, no/yes, off/on")


_CUSTOM_OP_TAGS = (
    (torch.Tag.cudagraph_unsafe,)
    if hasattr(torch, "Tag") and hasattr(torch.Tag, "cudagraph_unsafe")
    else ()
)


@torch.library.custom_op(
    "techjam::sage_attention_auto_v18",
    mutates_args=(),
    device_types="cuda",
    tags=_CUSTOM_OP_TAGS,
)
def sage_attention_auto_v18(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Call SageAttention's automatic GPU dispatcher without correction."""

    if sageattn is None:
        raise RuntimeError(
            "SageAttention is unavailable; install the pinned optional dependency"
        )
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("sage_attention_auto_v18 requires CUDA tensors")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k and v must have matching [B,H,S,Dh] shapes")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SageAttention requires FP16 or BF16 q/k/v")
    if q.shape[-1] > MAX_SAGE_HEAD_DIM:
        raise ValueError(f"unsupported SageAttention head_dim: {q.shape[-1]}")

    result = sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=True,
        sm_scale=scale,
        return_lse=False,
    )
    context = result[0] if isinstance(result, tuple) else result
    return context.contiguous()


@sage_attention_auto_v18.register_fake
def _sage_attention_auto_v18_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del k, v, scale
    return torch.empty_like(q, memory_format=torch.contiguous_format)


class UserOptimizedTransformer(V161Transformer):
    """V16.1 with direct automatic SageAttention on supported CUDA configs."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._sage_dependency_required = _read_bool_env(
            "TECHJAM_SAGE_REQUIRE",
            False,
        )
        if self._sage_dependency_required and not self.sage_attention_available:
            raise RuntimeError(
                "TECHJAM_SAGE_REQUIRE=1 but sageattention could not be imported"
            )
        if (
            self._sage_dependency_required
            and SAGE_ATTENTION_INSTALLED_VERSION
            != SAGE_ATTENTION_REQUIRED_VERSION
        ):
            raise RuntimeError(
                "TECHJAM_SAGE_REQUIRE=1 requires sageattention "
                f"{SAGE_ATTENTION_REQUIRED_VERSION}, found "
                f"{SAGE_ATTENTION_INSTALLED_VERSION!r}"
            )

        head_dim = config.d_model // config.num_heads
        self._sage_supported_config = (
            config.causal and head_dim <= MAX_SAGE_HEAD_DIM
        )

        # SageAttention 2.2 automatic kernels are unsafe under CUDA Graph
        # replay.  Keep V16.1's independently compiled shape-#14 body in the
        # supported no-CUDA-Graph mode as well.
        self._large_sequence_compile_mode = "max-autotune-no-cudagraphs"

    @property
    def sage_attention_available(self) -> bool:
        return sageattn is not None

    @property
    def sage_attention_supported_config(self) -> bool:
        return self._sage_supported_config

    @property
    def sage_attention_enabled(self) -> bool:
        return self._sage_supported_config and self.sage_attention_available

    def _mixed_attention(
        self,
        attention,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            not self.sage_attention_enabled
            or self.training
            or not x.is_cuda
            or x.dtype != torch.float32
        ):
            return super()._mixed_attention(attention, x, mask)

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
        context = sage_attention_auto_v18(q, k, v, attention.scale)
        context = context.transpose(1, 2).reshape(
            batch,
            seq_len,
            attention.d_model,
        )
        return F.linear(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
