#!/usr/bin/env python3
"""V19: V16.1 with a checkpointed-FP16 CUDA FFN-in/GELU kernel.

The CUDA WMMA kernel accumulates in FP16 for a short K interval, promotes the
partial 16x16 output tile into explicit FP32 registers, then resets the FP16
accumulator.  The default checkpoint is K=32.  Bias addition and exact erf
GELU use the final FP32 sum, while the GELU output keeps V16.1's FP16 boundary
for the existing FFN-out Tensor Core GEMM.

This is an experimental candidate, not the promoted main.  It inherits all
attention, cache, fallback and large-sequence scheduling behavior from the
standalone V16.1 artifact.  Select an ablation with
``TECHJAM_V19_CHECKPOINT_K=16|32|64|128|fp32``; the default is ``32``.
"""

from __future__ import annotations

import os
import threading
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

import torch_transformer_benchmark as bench
from v16_1_clean import (
    TransformerConfig,
    UserOptimizedTransformer as V161Transformer,
    fused_ffn_gelu_no_preround as v161_fused_ffn_gelu,
)


__all__ = ["TransformerConfig", "UserOptimizedTransformer"]


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor v19_fused_ffn_gelu_cuda(
    torch::Tensor normalized,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t checkpoint_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_ffn_gelu",
      &v19_fused_ffn_gelu_cuda,
      "V19 checkpointed-FP16 FFN-in plus exact GELU (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <vector>

namespace wmma = nvcuda::wmma;

constexpr int kTile = 16;
constexpr int kWarpSize = 32;
constexpr int kElementsPerLane = (kTile * kTile) / kWarpSize;
constexpr int kWarpsM = 4;
constexpr int kWarpsN = 2;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;
constexpr int kBlockThreads = kWarpsPerBlock * kWarpSize;
constexpr int kBlockM = kWarpsM * kTile;
constexpr int kBlockN = kWarpsN * kTile;

__device__ __forceinline__ half exact_gelu_to_half(float value) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  const float gelu = 0.5f * value * (1.0f + erff(value * kInvSqrt2));
  return __float2half_rn(gelu);
}

template <int CheckpointSteps>
__global__ __launch_bounds__(kBlockThreads) void checkpointed_fp16_kernel(
    const half* __restrict__ normalized,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ __align__(32) half shared_a[kBlockM * kTile];
  __shared__ __align__(32) half shared_b[kTile * kBlockN];
  __shared__ __align__(32) half partial_tiles[
      kWarpsPerBlock * kTile * kTile];

  const int64_t tiles_n = (n + kBlockN - 1) / kBlockN;
  const int64_t block_m = static_cast<int64_t>(blockIdx.x) / tiles_n;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) % tiles_n;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;

  float fp32_accumulator[kElementsPerLane];
#pragma unroll
  for (int item = 0; item < kElementsPerLane; ++item) {
    fp32_accumulator[item] = 0.0f;
  }

  for (int64_t group_k = 0; group_k < k;
       group_k += CheckpointSteps * kTile) {
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, half> half_acc;
    wmma::fill_fragment(half_acc, __float2half(0.0f));

#pragma unroll
    for (int step = 0; step < CheckpointSteps; ++step) {
      const int64_t tile_k = group_k + step * kTile;
      if (tile_k < k) {
        for (int linear = threadIdx.x; linear < kBlockM * kTile;
             linear += kBlockThreads) {
          const int row = linear / kTile;
          const int column = linear % kTile;
          const int64_t global_m = block_m * kBlockM + row;
          shared_a[linear] =
              global_m < m ? normalized[global_m * k + tile_k + column]
                           : __float2half(0.0f);
        }
        // B is staged column-major as [K=16, N=32].  This matches WMMA's
        // matrix_b layout while global weight remains row-major [N,K].
        for (int linear = threadIdx.x; linear < kTile * kBlockN;
             linear += kBlockThreads) {
          const int column = linear / kTile;
          const int row = linear % kTile;
          const int64_t global_n = block_n * kBlockN + column;
          shared_b[linear] =
              global_n < n ? weight[global_n * k + tile_k + row]
                           : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                       wmma::row_major>
            matrix_a;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                       wmma::col_major>
            matrix_b;
        const half* a_ptr = shared_a + warp_m * kTile * kTile;
        const half* b_ptr = shared_b + warp_n * kTile * kTile;
        wmma::load_matrix_sync(matrix_a, a_ptr, kTile);
        wmma::load_matrix_sync(matrix_b, b_ptr, kTile);
        wmma::mma_sync(half_acc, matrix_a, matrix_b, half_acc);
        __syncthreads();
      }
    }

    half* partial_tile = partial_tiles + warp * kTile * kTile;
    wmma::store_matrix_sync(
        partial_tile, half_acc, kTile, wmma::mem_row_major);
    __syncwarp();

    // Each lane owns fixed row-major elements.  This explicit conversion and
    // add is the FP32 checkpoint; the next K group starts from a fresh FP16
    // WMMA accumulator.
#pragma unroll
    for (int item = 0; item < kElementsPerLane; ++item) {
      const int linear = lane + item * kWarpSize;
      fp32_accumulator[item] += __half2float(partial_tile[linear]);
    }
    __syncwarp();
  }

#pragma unroll
  for (int item = 0; item < kElementsPerLane; ++item) {
    const int linear = lane + item * kWarpSize;
    const int row = linear / kTile;
    const int column = linear % kTile;
    const int64_t global_m = block_m * kBlockM + warp_m * kTile + row;
    const int64_t global_n = block_n * kBlockN + warp_n * kTile + column;
    if (global_m < m && global_n < n) {
      const float linear_fp32 =
          fp32_accumulator[item] + __half2float(bias[global_n]);
      output[global_m * n + global_n] = exact_gelu_to_half(linear_fp32);
    }
  }
}

__global__ __launch_bounds__(kBlockThreads) void fp32_control_kernel(
    const half* __restrict__ normalized,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ __align__(32) half shared_a[kBlockM * kTile];
  __shared__ __align__(32) half shared_b[kTile * kBlockN];
  __shared__ __align__(32) float output_tiles[
      kWarpsPerBlock * kTile * kTile];

  const int64_t tiles_n = (n + kBlockN - 1) / kBlockN;
  const int64_t block_m = static_cast<int64_t>(blockIdx.x) / tiles_n;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) % tiles_n;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int64_t tile_k = 0; tile_k < k; tile_k += kTile) {
    for (int linear = threadIdx.x; linear < kBlockM * kTile;
         linear += kBlockThreads) {
      const int row = linear / kTile;
      const int column = linear % kTile;
      const int64_t global_m = block_m * kBlockM + row;
      shared_a[linear] =
          global_m < m ? normalized[global_m * k + tile_k + column]
                       : __float2half(0.0f);
    }
    for (int linear = threadIdx.x; linear < kTile * kBlockN;
         linear += kBlockThreads) {
      const int column = linear / kTile;
      const int row = linear % kTile;
      const int64_t global_n = block_n * kBlockN + column;
      shared_b[linear] =
          global_n < n ? weight[global_n * k + tile_k + row]
                       : __float2half(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half,
                   wmma::row_major>
        matrix_a;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half,
                   wmma::col_major>
        matrix_b;
    const half* a_ptr = shared_a + warp_m * kTile * kTile;
    const half* b_ptr = shared_b + warp_n * kTile * kTile;
    wmma::load_matrix_sync(matrix_a, a_ptr, kTile);
    wmma::load_matrix_sync(matrix_b, b_ptr, kTile);
    wmma::mma_sync(accumulator, matrix_a, matrix_b, accumulator);
    __syncthreads();
  }

  float* output_tile = output_tiles + warp * kTile * kTile;
  wmma::store_matrix_sync(
      output_tile, accumulator, kTile, wmma::mem_row_major);
  __syncwarp();

#pragma unroll
  for (int item = 0; item < kElementsPerLane; ++item) {
    const int linear = lane + item * kWarpSize;
    const int row = linear / kTile;
    const int column = linear % kTile;
    const int64_t global_m = block_m * kBlockM + warp_m * kTile + row;
    const int64_t global_n = block_n * kBlockN + warp_n * kTile + column;
    if (global_m < m && global_n < n) {
      const float linear_fp32 =
          output_tile[linear] + __half2float(bias[global_n]);
      output[global_m * n + global_n] = exact_gelu_to_half(linear_fp32);
    }
  }
}

torch::Tensor v19_fused_ffn_gelu_cuda(
    torch::Tensor normalized,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t checkpoint_k) {
  TORCH_CHECK(normalized.is_cuda(), "normalized must be CUDA");
  TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight and bias must be CUDA");
  TORCH_CHECK(
      normalized.get_device() == weight.get_device() &&
          normalized.get_device() == bias.get_device(),
      "all tensors must be on one CUDA device");
  TORCH_CHECK(
      normalized.scalar_type() == at::kHalf &&
          weight.scalar_type() == at::kHalf && bias.scalar_type() == at::kHalf,
      "V19 CUDA kernel requires FP16 tensors");
  TORCH_CHECK(
      normalized.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(),
      "V19 CUDA kernel requires contiguous tensors");
  TORCH_CHECK(normalized.dim() >= 2, "normalized must have at least two dimensions");
  TORCH_CHECK(weight.dim() == 2 && bias.dim() == 1, "invalid weight or bias rank");

  const int64_t k = normalized.size(-1);
  const int64_t n = weight.size(0);
  const int64_t m = normalized.numel() / k;
  TORCH_CHECK(weight.size(1) == k && bias.numel() == n, "incompatible shapes");
  TORCH_CHECK(
      m % kTile == 0 && n % kTile == 0 && k % kTile == 0,
      "V19 CUDA kernel requires M, N and K divisible by 16");
  TORCH_CHECK(
      checkpoint_k == 0 || checkpoint_k == 16 || checkpoint_k == 32 ||
          checkpoint_k == 64 || checkpoint_k == 128,
      "checkpoint_k must be 0, 16, 32, 64 or 128");

  std::vector<int64_t> output_sizes(normalized.sizes().begin(), normalized.sizes().end());
  output_sizes.back() = n;
  auto output = torch::empty(output_sizes, normalized.options());

  const int64_t grid_x_64 =
      ((m + kBlockM - 1) / kBlockM) * ((n + kBlockN - 1) / kBlockN);
  TORCH_CHECK(grid_x_64 <= 2147483647LL, "V19 CUDA launch grid is too large");
  const dim3 grid(static_cast<unsigned int>(grid_x_64));
  const dim3 block(kBlockThreads);
  c10::cuda::CUDAGuard device_guard(normalized.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const half* normalized_ptr =
      reinterpret_cast<const half*>(normalized.data_ptr<at::Half>());
  const half* weight_ptr =
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
  const half* bias_ptr = reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());

  switch (checkpoint_k) {
    case 0:
      fp32_control_kernel<<<grid, block, 0, stream>>>(
          normalized_ptr, weight_ptr, bias_ptr, output_ptr, m, n, k);
      break;
    case 16:
      checkpointed_fp16_kernel<1><<<grid, block, 0, stream>>>(
          normalized_ptr, weight_ptr, bias_ptr, output_ptr, m, n, k);
      break;
    case 32:
      checkpointed_fp16_kernel<2><<<grid, block, 0, stream>>>(
          normalized_ptr, weight_ptr, bias_ptr, output_ptr, m, n, k);
      break;
    case 64:
      checkpointed_fp16_kernel<4><<<grid, block, 0, stream>>>(
          normalized_ptr, weight_ptr, bias_ptr, output_ptr, m, n, k);
      break;
    case 128:
      checkpointed_fp16_kernel<8><<<grid, block, 0, stream>>>(
          normalized_ptr, weight_ptr, bias_ptr, output_ptr, m, n, k);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[BaseException] = None
_CUDA_EXTENSION_LOCK = threading.Lock()
_WARNED_CUDA_FALLBACK = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_checkpoint_k() -> int:
    raw = os.environ.get("TECHJAM_V19_CHECKPOINT_K", "32").strip().lower()
    if raw == "fp32":
        return 0
    try:
        checkpoint_k = int(raw)
    except ValueError as exc:
        raise ValueError(
            "TECHJAM_V19_CHECKPOINT_K must be 16, 32, 64, 128 or fp32"
        ) from exc
    if checkpoint_k not in {16, 32, 64, 128}:
        raise ValueError(
            "TECHJAM_V19_CHECKPOINT_K must be 16, 32, 64, 128 or fp32"
        )
    return checkpoint_k


def _ensure_cuda_extension():
    global _CUDA_EXTENSION, _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ERROR is not None:
        raise RuntimeError("V19 CUDA extension previously failed to build") from _CUDA_EXTENSION_ERROR

    with _CUDA_EXTENSION_LOCK:
        if _CUDA_EXTENSION is not None:
            return _CUDA_EXTENSION
        if _CUDA_EXTENSION_ERROR is not None:
            raise RuntimeError(
                "V19 CUDA extension previously failed to build"
            ) from _CUDA_EXTENSION_ERROR
        try:
            _CUDA_EXTENSION = load_inline(
                name="techjam_v19_checkpoint_cuda_ext_v2",
                cpp_sources=[_CPP_SOURCE],
                cuda_sources=[_CUDA_SOURCE],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=_env_flag("TECHJAM_V19_CUDA_VERBOSE"),
            )
        except BaseException as exc:
            _CUDA_EXTENSION_ERROR = exc
            raise
    return _CUDA_EXTENSION


def _portable_checkpointed_ffn_gelu(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    checkpoint_k: int,
) -> torch.Tensor:
    """CPU/local structural approximation; never a CUDA performance path."""

    if checkpoint_k == 0:
        linear_fp32 = F.linear(normalized.float(), weight.float(), bias.float())
        return F.gelu(linear_fp32, approximate="none").to(normalized.dtype)

    k = normalized.shape[-1]
    normalized_2d = normalized.reshape(-1, k)
    accumulator = torch.zeros(
        (normalized_2d.shape[0], weight.shape[0]),
        device=normalized.device,
        dtype=torch.float32,
    )
    for start in range(0, k, checkpoint_k):
        end = min(start + checkpoint_k, k)
        partial = torch.matmul(
            normalized_2d[:, start:end],
            weight[:, start:end].transpose(0, 1),
        )
        accumulator.add_(partial.to(dtype=torch.float32))
    accumulator.add_(bias.float())
    output = F.gelu(accumulator, approximate="none").to(normalized.dtype)
    return output.reshape((*normalized.shape[:-1], weight.shape[0]))


@torch.library.custom_op(
    "techjam::v19_cuda_checkpointed_ffn_gelu",
    mutates_args=(),
)
def cuda_checkpointed_ffn_gelu(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    checkpoint_k: int,
) -> torch.Tensor:
    if not normalized.is_cuda:
        return _portable_checkpointed_ffn_gelu(
            normalized, weight, bias, checkpoint_k
        )

    try:
        extension = _ensure_cuda_extension()
    except BaseException:
        if not _env_flag("TECHJAM_V19_ALLOW_CUDA_FALLBACK"):
            raise
        global _WARNED_CUDA_FALLBACK
        if not _WARNED_CUDA_FALLBACK:
            warnings.warn(
                "V19 CUDA extension unavailable; falling back to V16.1 Triton. "
                "This run is not a V19 CUDA performance result.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_CUDA_FALLBACK = True
        return v161_fused_ffn_gelu(normalized, weight, bias)

    return extension.fused_ffn_gelu(normalized, weight, bias, checkpoint_k)


@cuda_checkpointed_ffn_gelu.register_fake
def _cuda_checkpointed_ffn_gelu_fake(
    normalized: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    checkpoint_k: int,
) -> torch.Tensor:
    del bias, checkpoint_k
    return normalized.new_empty((*normalized.shape[:-1], weight.shape[0]))


class UserOptimizedTransformer(V161Transformer):
    """V16.1 with a versioned CUDA FFN-in/GELU accumulation experiment."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._v19_checkpoint_k = _parse_checkpoint_k()

    @property
    def v19_checkpoint_k(self) -> int:
        return self._v19_checkpoint_k

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        parameter = next(self.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            try:
                _ensure_cuda_extension()
            except BaseException:
                if not _env_flag("TECHJAM_V19_ALLOW_CUDA_FALLBACK"):
                    raise
        return result

    def _mixed_ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        k = x.shape[-1]
        n = layer._ffn_in_weight_mixed.shape[0]
        m = x.numel() // k
        if m % 16 != 0 or n % 16 != 0 or k % 16 != 0:
            return V161Transformer._mixed_ffn(layer, x)

        normalized = layer.norm2(x).to(dtype=torch.float16)
        hidden = cuda_checkpointed_ffn_gelu(
            normalized,
            layer._ffn_in_weight_mixed,
            layer._ffn_in_bias_mixed,
            self._v19_checkpoint_k,
        )
        return F.linear(
            hidden,
            layer._ffn_out_weight_mixed,
            layer._ffn_out_bias_mixed,
        ).float()


bench.UserOptimizedTransformer = UserOptimizedTransformer


if __name__ == "__main__":
    raise SystemExit(bench.main())
