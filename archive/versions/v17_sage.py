#!/usr/bin/env python3
"""V17-Sage: cross-shape SageAttention with exact causal-prefix correction.

This user-named candidate is distinct from the historical
``v17_CompiledBatch2.py`` experiment.  It inherits the source-clean V16.1 main
and changes only eligible causal attention calls:

* SageAttention INT8-QK per-thread with FP16 PV and FP32 accumulation computes
  the full attention output.
* PyTorch Flash recomputes the first 32 causal queries exactly and overwrites
  that prefix.  The measured shape-#14 Sage failures were all in queries 1..31.
* The attention output projection stores its FP32 accumulator directly to buy
  additional margin before the residual connection.

The optional Sage dependency is never required for importing this module.
Unsupported configurations fall back to V16.1 unchanged.  Set
``TECHJAM_SAGE_REQUIRE=1`` in GPU validation runs to reject a missing dependency
instead of silently measuring the fallback.  ``TECHJAM_SAGE_EXACT_PREFIX`` can
override the default correction length for accuracy ablations.
"""

from __future__ import annotations

import os
from importlib import metadata
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

import torch_transformer_benchmark as bench
from v4_3_Flash import FLASH_FIRST_BACKENDS
from v12_FP32FFNOut import linear_fp32_output
from v16_1_NoDirectQKV13 import UserOptimizedTransformer as V161Transformer


SAGE_ATTENTION_SOURCE_COMMIT = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SAGE_ATTENTION_REQUIRED_VERSION = "2.2.0"
DEFAULT_EXACT_PREFIX = 32
MAX_SAGE_HEAD_DIM = 128

try:
    from sageattention import sageattn_qk_int8_pv_fp16_cuda
except ImportError:
    sageattn_qk_int8_pv_fp16_cuda = None

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


def _read_nonnegative_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


_CUSTOM_OP_TAGS = (
    (torch.Tag.cudagraph_unsafe,)
    if hasattr(torch, "Tag") and hasattr(torch.Tag, "cudagraph_unsafe")
    else ()
)


@torch.library.custom_op(
    "techjam::sage_attention_exact_prefix",
    mutates_args=(),
    device_types="cuda",
    tags=_CUSTOM_OP_TAGS,
)
def sage_attention_exact_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    exact_prefix: int,
) -> torch.Tensor:
    """Run accuracy-oriented SageAttention and restore a causal query prefix."""

    if sageattn_qk_int8_pv_fp16_cuda is None:
        raise RuntimeError(
            "SageAttention is unavailable; install the pinned optional dependency"
        )
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("sage_attention_exact_prefix requires CUDA tensors")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k and v must have matching [B,H,S,Dh] shapes")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SageAttention requires FP16 or BF16 q/k/v")
    if q.shape[-1] > MAX_SAGE_HEAD_DIM:
        raise ValueError(f"unsupported SageAttention head_dim: {q.shape[-1]}")
    if exact_prefix < 0:
        raise ValueError("exact_prefix must be non-negative")

    result = sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=True,
        qk_quant_gran="per_thread",
        sm_scale=scale,
        pv_accum_dtype="fp32",
        smooth_k=True,
        smooth_v=False,
        return_lse=False,
    )
    context = result[0] if isinstance(result, tuple) else result
    # The fake implementation promises contiguous HND output.  Keep the real
    # contract explicit even if a future Sage build returns a view.
    context = context.contiguous()

    prefix = min(exact_prefix, q.shape[-2])
    if prefix:
        # For causal queries [0, P), all visible keys also lie in [0, P), so a
        # square P-by-P Flash call is mathematically identical to selecting the
        # first P rows from full causal attention while costing only O(P^2).
        with sdpa_kernel(backends=FLASH_FIRST_BACKENDS, set_priority=True):
            exact = F.scaled_dot_product_attention(
                q[..., :prefix, :],
                k[..., :prefix, :],
                v[..., :prefix, :],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
        context[..., :prefix, :].copy_(exact)
    return context


@sage_attention_exact_prefix.register_fake
def _sage_attention_exact_prefix_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    exact_prefix: int,
) -> torch.Tensor:
    del k, v, scale, exact_prefix
    return torch.empty_like(q, memory_format=torch.contiguous_format)


class UserOptimizedTransformer(V161Transformer):
    """V16.1 with corrected SageAttention on every supported causal config."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._sage_exact_prefix = _read_nonnegative_int_env(
            "TECHJAM_SAGE_EXACT_PREFIX",
            DEFAULT_EXACT_PREFIX,
        )
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
            config.causal
            and config.seq_len > self._sage_exact_prefix
            and head_dim <= MAX_SAGE_HEAD_DIM
        )

        # Sage 2.2 is not CUDA-Graph safe.  The custom-op tag protects compiled
        # outer graphs; this default also protects V16.1's independently
        # compiled large-sequence executor when a harness does not override it.
        self._large_sequence_compile_mode = "max-autotune-no-cudagraphs"

    @property
    def sage_attention_available(self) -> bool:
        return sageattn_qk_int8_pv_fp16_cuda is not None

    @property
    def sage_attention_supported_config(self) -> bool:
        return self._sage_supported_config

    @property
    def sage_attention_enabled(self) -> bool:
        return self._sage_supported_config and self.sage_attention_available

    @property
    def sage_exact_prefix(self) -> int:
        return self._sage_exact_prefix

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
        context = sage_attention_exact_prefix(
            q,
            k,
            v,
            attention.scale,
            self._sage_exact_prefix,
        )
        context = context.transpose(1, 2).reshape(
            batch,
            seq_len,
            attention.d_model,
        )
        return linear_fp32_output(
            context,
            attention._out_weight_mixed,
            attention._out_bias_mixed,
        )


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
