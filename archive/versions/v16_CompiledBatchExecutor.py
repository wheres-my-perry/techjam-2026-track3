#!/usr/bin/env python3
"""V16: reuse one compiled B=1 executor inside shape-#14 batch chunking.

V14.1 keeps official shape #14 within 32 GiB by evaluating its 32 independent
batch samples one at a time.  Its compiler-disabled Python loop is intentional,
but it also leaves every B=1 Transformer invocation eager.  V16 keeps that
outer loop eager and compiles only the single-sample callable.  The resulting
executor is reused for all samples without unrolling the batch loop.

The executor is a derived inference cache, not model state.  It is invalidated
whenever weights/device/dtype or train/eval mode changes and is rebuilt lazily
after the model is ready.  V15's exact-shape-#13 direct-QKV path and every
other fallback remain inherited unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

import torch_transformer_benchmark as bench
from v15_DirectQKVLayout import UserOptimizedTransformer as V15Transformer


class UserOptimizedTransformer(V15Transformer):
    """V15 plus a reusable compiled sample executor for large sequences."""

    def __init__(self, config: bench.TransformerConfig) -> None:
        super().__init__(config)
        self._large_sequence_compile_enabled = True
        self._large_sequence_compile_backend: Optional[str] = None
        self._large_sequence_compile_mode: Optional[str] = "max-autotune"
        self._large_sequence_compile_fullgraph = False
        # Bypass nn.Module.__setattr__: a compiled bound callable is an
        # ephemeral runtime cache and must never become a child module/state.
        self.__dict__["_compiled_large_sequence_executor"] = None

    def _invalidate_large_sequence_executor(self) -> None:
        self.__dict__["_compiled_large_sequence_executor"] = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._invalidate_large_sequence_executor()
        return result

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._invalidate_large_sequence_executor()
        return result

    def train(self, mode: bool = True):
        changed = self.training != mode
        result = super().train(mode)
        if changed:
            self._invalidate_large_sequence_executor()
        return result

    def configure_large_sequence_executor(
        self,
        *,
        enabled: bool = True,
        backend: Optional[str] = None,
        mode: Optional[str] = "max-autotune",
        fullgraph: bool = False,
    ) -> None:
        """Configure and invalidate the derived compiled-executor cache.

        ``backend="eager", mode=None`` is useful for CPU/Dynamo semantic
        diagnostics.  The CUDA candidate uses the default Inductor backend.
        """

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
        # The inherited V14.1 dispatcher sees B=1 and therefore executes one
        # ordinary V15/V11 forward without recursively entering batch chunking.
        return super().forward(x, valid_token_mask)

    def prepare_large_sequence_executor(self) -> Any:
        """Create the lazy torch.compile wrapper after model setup."""

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
        """Run one large-sequence sample through the reusable executor."""

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
        """Keep the memory-bounded outer loop eager; compile only its body."""

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


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
