#include "cuda_ops/rmsnorm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace cuda_ops {
namespace {

template <typename T>
struct Convert;

template <>
struct Convert<float> {
  __device__ static float toFloat(float value) { return value; }
  __device__ static float fromFloat(float value) { return value; }
};

template <>
struct Convert<__half> {
  __device__ static float toFloat(__half value) { return __half2float(value); }
  __device__ static __half fromFloat(float value) { return __float2half_rn(value); }
};

template <int Threads>
__device__ float blockReduceSum(float value) {
  constexpr int kWarpSize = 32;
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kWarps];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  value = threadIdx.x < kWarps ? warp_sums[lane] : 0.0F;
  if (warp == 0) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
  }
  return value;
}

template <typename T, int Threads, bool HasResidual, bool StoreInvRms>
__global__ __launch_bounds__(Threads) void rmsNormKernel(const T* __restrict__ input,
                                                          const T* __restrict__ residual,
                                                          const T* __restrict__ weight,
                                                          T* __restrict__ output,
                                                          float* __restrict__ inv_rms,
                                                          int64_t cols, int64_t input_stride,
                                                          int64_t residual_stride, int64_t output_stride,
                                                          float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const T* input_row = input + row * input_stride;
  const T* residual_row = HasResidual ? residual + row * residual_stride : nullptr;
  T* output_row = output + row * output_stride;

  float sum_squares = 0.0F;
  for (int64_t col = threadIdx.x; col < cols; col += Threads) {
    float value = Convert<T>::toFloat(input_row[col]);
    if constexpr (HasResidual) value += Convert<T>::toFloat(residual_row[col]);
    sum_squares = fmaf(value, value, sum_squares);
  }
  sum_squares = blockReduceSum<Threads>(sum_squares);
  __shared__ float scale;
  if (threadIdx.x == 0) {
    scale = rsqrtf(sum_squares / static_cast<float>(cols) + epsilon);
    if constexpr (StoreInvRms) inv_rms[row] = scale;
  }
  __syncthreads();

  for (int64_t col = threadIdx.x; col < cols; col += Threads) {
    float value = Convert<T>::toFloat(input_row[col]);
    if constexpr (HasResidual) value += Convert<T>::toFloat(residual_row[col]);
    output_row[col] = Convert<T>::fromFloat(value * scale * Convert<T>::toFloat(weight[col]));
  }
}

bool validTensor(const TensorView& tensor) {
  return tensor.data != nullptr && tensor.rows > 0 && tensor.cols > 0 &&
         tensor.row_stride >= tensor.cols;
}

template <typename T, int Threads>
Status dispatch(const TensorView& input, const TensorView* residual, const TensorView& weight,
                TensorView output, const RmsNormConfig& config, float* inv_rms, cudaStream_t stream) {
  const auto* input_data = static_cast<const T*>(input.data);
  const auto* residual_data = residual == nullptr ? nullptr : static_cast<const T*>(residual->data);
  const auto* weight_data = static_cast<const T*>(weight.data);
  auto* output_data = static_cast<T*>(output.data);
  const dim3 grid(static_cast<unsigned int>(input.rows));
  const dim3 block(Threads);
  if (residual != nullptr) {
    if (config.store_inv_rms) {
      rmsNormKernel<T, Threads, true, true><<<grid, block, 0, stream>>>(input_data, residual_data, weight_data,
          output_data, inv_rms, input.cols, input.row_stride, residual->row_stride, output.row_stride, config.epsilon);
    } else {
      rmsNormKernel<T, Threads, true, false><<<grid, block, 0, stream>>>(input_data, residual_data, weight_data,
          output_data, nullptr, input.cols, input.row_stride, residual->row_stride, output.row_stride, config.epsilon);
    }
  } else if (config.store_inv_rms) {
    rmsNormKernel<T, Threads, false, true><<<grid, block, 0, stream>>>(input_data, nullptr, weight_data,
        output_data, inv_rms, input.cols, input.row_stride, 0, output.row_stride, config.epsilon);
  } else {
    rmsNormKernel<T, Threads, false, false><<<grid, block, 0, stream>>>(input_data, nullptr, weight_data,
        output_data, nullptr, input.cols, input.row_stride, 0, output.row_stride, config.epsilon);
  }
  return cudaPeekAtLastError() == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

}  // namespace

std::string_view statusMessage(Status status) {
  switch (status) {
    case Status::kSuccess: return "success";
    case Status::kInvalidArgument: return "invalid argument";
    case Status::kUnsupported: return "unsupported configuration";
    case Status::kCudaError: return "CUDA launch error";
  }
  return "unknown error";
}

Status rmsNormForward(const TensorView& input, const TensorView* residual, const TensorView& weight,
                      TensorView output, const RmsNormConfig& config, float* inv_rms, cudaStream_t stream) {
  if (!validTensor(input) || !validTensor(weight) || !validTensor(output) || !std::isfinite(config.epsilon) ||
      config.epsilon <= 0.0F || input.cols > std::numeric_limits<int>::max() ||
      input.rows > std::numeric_limits<unsigned int>::max()) return Status::kInvalidArgument;
  if (input.dtype != weight.dtype || input.dtype != output.dtype || output.rows != input.rows ||
      output.cols != input.cols || weight.rows != 1 || weight.cols != input.cols) return Status::kInvalidArgument;
  if (residual != nullptr && (!validTensor(*residual) || residual->dtype != input.dtype ||
      residual->rows != input.rows || residual->cols != input.cols)) return Status::kInvalidArgument;
  if (config.store_inv_rms && inv_rms == nullptr) return Status::kInvalidArgument;

  constexpr int kThreads = CUDA_OPS_RMSNORM_THREADS;
  if (input.dtype == DataType::kFloat16) {
    return dispatch<__half, kThreads>(input, residual, weight, output, config, inv_rms, stream);
  }
  if (input.dtype == DataType::kFloat32) {
    return dispatch<float, kThreads>(input, residual, weight, output, config, inv_rms, stream);
  }
  return Status::kUnsupported;
}

}  // namespace cuda_ops
