#!/usr/bin/env python3
"""Shared multi-stream large-batch scheduler for V19.1 candidates."""

from __future__ import annotations

import os
from typing import Optional

import torch


SUPPORTED_PARALLEL_PARTS = (1, 2, 4, 8, 16, 32)
PARALLEL_SAFE_COMPILE_MODE = "max-autotune-no-cudagraphs"


def parse_parallel_parts() -> int:
    raw = os.environ.get("TECHJAM_V19_PARALLEL_PARTS", "2").strip()
    try:
        parts = int(raw)
    except ValueError as exc:
        raise ValueError(
            "TECHJAM_V19_PARALLEL_PARTS must be one of 1,2,4,8,16,32"
        ) from exc
    if parts not in SUPPORTED_PARALLEL_PARTS:
        raise ValueError(
            "TECHJAM_V19_PARALLEL_PARTS must be one of 1,2,4,8,16,32"
        )
    return parts


def balanced_batch_ranges(
    batch_size: int,
    requested_parts: int,
) -> tuple[tuple[int, int], ...]:
    """Return contiguous, balanced, non-empty ranges covering ``[0, B)``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if requested_parts <= 0:
        raise ValueError("requested_parts must be positive")

    active_parts = min(batch_size, requested_parts)
    base, remainder = divmod(batch_size, active_parts)
    ranges = []
    start = 0
    for partition in range(active_parts):
        width = base + (1 if partition < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return tuple(ranges)


class ParallelBatchPartitionsMixin:
    """Run V16-style B=1 large-sequence executors on independent streams."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parallel_batch_parts = parse_parallel_parts()
        self.__dict__["_parallel_batch_stream_cache"] = None
        if self._parallel_batch_parts > 1:
            self._large_sequence_compile_mode = PARALLEL_SAFE_COMPILE_MODE

    @property
    def parallel_batch_parts(self) -> int:
        return self._parallel_batch_parts

    @staticmethod
    def parallel_batch_ranges(
        batch_size: int, requested_parts: int
    ) -> tuple[tuple[int, int], ...]:
        return balanced_batch_ranges(batch_size, requested_parts)

    def _invalidate_large_sequence_executor(self) -> None:
        super()._invalidate_large_sequence_executor()
        self.__dict__["_parallel_batch_stream_cache"] = None

    def configure_large_sequence_executor(
        self,
        *,
        enabled: bool = True,
        backend: Optional[str] = None,
        mode: Optional[str] = "max-autotune",
        fullgraph: bool = False,
    ) -> None:
        if (
            enabled
            and self._parallel_batch_parts > 1
            and backend in (None, "inductor")
        ):
            mode = PARALLEL_SAFE_COMPILE_MODE
        super().configure_large_sequence_executor(
            enabled=enabled,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
        )

    def _parallel_streams(
        self, device: torch.device, count: int
    ) -> tuple[torch.cuda.Stream, ...]:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cache = self.__dict__.get("_parallel_batch_stream_cache")
        if cache is not None:
            cached_device, cached_count, streams = cache
            if cached_device == device_index and cached_count == count:
                return streams

        streams = tuple(torch.cuda.Stream(device=device) for _ in range(count))
        self.__dict__["_parallel_batch_stream_cache"] = (
            device_index,
            count,
            streams,
        )
        return streams

    @torch.compiler.disable
    def _forward_large_sequence(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not x.is_cuda or self._parallel_batch_parts == 1:
            return super()._forward_large_sequence(x, valid_token_mask)

        ranges = balanced_batch_ranges(x.shape[0], self._parallel_batch_parts)
        if len(ranges) == 1:
            return super()._forward_large_sequence(x, valid_token_mask)

        # torch.compile is lazy, but constructing the callable and selecting the
        # no-CUDA-Graph policy happen once before switching worker streams.
        executor = self.prepare_large_sequence_executor()
        output = torch.empty_like(x)
        current_stream = torch.cuda.current_stream(device=x.device)
        worker_streams = self._parallel_streams(x.device, len(ranges))

        for worker in worker_streams:
            worker.wait_stream(current_stream)
            x.record_stream(worker)
            output.record_stream(worker)
            if valid_token_mask is not None:
                valid_token_mask.record_stream(worker)

        for worker, (partition_start, partition_end) in zip(
            worker_streams, ranges
        ):
            with torch.cuda.stream(worker):
                for sample in range(partition_start, partition_end):
                    sample_mask = (
                        None
                        if valid_token_mask is None
                        else valid_token_mask[sample : sample + 1]
                    )
                    output[sample : sample + 1].copy_(
                        executor(x[sample : sample + 1], sample_mask)
                    )

        for worker in worker_streams:
            current_stream.wait_stream(worker)
        return output


__all__ = [
    "PARALLEL_SAFE_COMPILE_MODE",
    "SUPPORTED_PARALLEL_PARTS",
    "ParallelBatchPartitionsMixin",
    "balanced_batch_ranges",
]
