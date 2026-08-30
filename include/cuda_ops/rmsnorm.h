#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string_view>

namespace cuda_ops {

enum class DataType : uint8_t { kFloat16, kFloat32 };

// A non-owning, row-major 2D device tensor. Strides are in elements.
struct TensorView {
  void* data{nullptr};
  int64_t rows{0};
  int64_t cols{0};
  int64_t row_stride{0};
  DataType dtype{DataType::kFloat16};
};

struct RmsNormConfig {
  float epsilon{1.0e-6F};
  bool store_inv_rms{false};  // inv_rms must point to `rows` float values when set.
};

enum class Status : uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kUnsupported,
  kCudaError,
};

[[nodiscard]] std::string_view statusMessage(Status status);

// Computes output[row, col] = (input + optional residual) * rsqrt(mean(x^2) + epsilon) * weight[col].
// All tensors must live on the active CUDA device. `weight` has shape [1, cols].
// The launch is asynchronous with respect to the host; CUDA errors from validation are returned directly.
[[nodiscard]] Status rmsNormForward(const TensorView& input, const TensorView* residual,
                                    const TensorView& weight, TensorView output,
                                    const RmsNormConfig& config, float* inv_rms,
                                    cudaStream_t stream = nullptr);

}  // namespace cuda_ops

