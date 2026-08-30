#include "cuda_ops/cuda_check.cuh"
#include "cuda_ops/rmsnorm.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
float toFloat(T value);
template <>
float toFloat(float value) { return value; }
template <>
float toFloat(__half value) { return __half2float(value); }

template <typename T>
T fromFloat(float value);
template <>
float fromFloat(float value) { return value; }
template <>
__half fromFloat(float value) { return __float2half_rn(value); }

template <typename T>
void runCase(int64_t rows, int64_t cols, bool residual, int64_t row_stride = 0) {
  if (row_stride == 0) row_stride = cols;
  std::mt19937 generator(7);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<T> input(rows * row_stride), residual_host(rows * row_stride), weight(cols), output(rows * row_stride);
  std::vector<float> expected(rows * cols), expected_inv_rms(rows), inv_rms(rows);
  for (auto& value : input) value = fromFloat<T>(distribution(generator));
  for (auto& value : residual_host) value = fromFloat<T>(distribution(generator));
  for (auto& value : weight) value = fromFloat<T>(distribution(generator));

  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (int64_t col = 0; col < cols; ++col) {
      float value = toFloat(input[row * row_stride + col]) +
                    (residual ? toFloat(residual_host[row * row_stride + col]) : 0.0F);
      sum += value * value;
    }
    const float scale = 1.0F / std::sqrt(sum / static_cast<float>(cols) + 1.0e-6F);
    expected_inv_rms[row] = scale;
    for (int64_t col = 0; col < cols; ++col) {
      float value = toFloat(input[row * row_stride + col]) +
                    (residual ? toFloat(residual_host[row * row_stride + col]) : 0.0F);
      expected[row * cols + col] = value * scale * toFloat(weight[col]);
    }
  }

  T *d_input = nullptr, *d_residual = nullptr, *d_weight = nullptr, *d_output = nullptr;
  float* d_inv_rms = nullptr;
  const size_t data_bytes = input.size() * sizeof(T);
  CUDA_OPS_CHECK(cudaMalloc(&d_input, data_bytes));
  CUDA_OPS_CHECK(cudaMalloc(&d_residual, data_bytes));
  CUDA_OPS_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(T)));
  CUDA_OPS_CHECK(cudaMalloc(&d_output, data_bytes));
  CUDA_OPS_CHECK(cudaMalloc(&d_inv_rms, inv_rms.size() * sizeof(float)));
  CUDA_OPS_CHECK(cudaMemcpy(d_input, input.data(), data_bytes, cudaMemcpyHostToDevice));
  CUDA_OPS_CHECK(cudaMemcpy(d_residual, residual_host.data(), data_bytes, cudaMemcpyHostToDevice));
  CUDA_OPS_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(T), cudaMemcpyHostToDevice));

  const auto dtype = std::is_same_v<T, float> ? cuda_ops::DataType::kFloat32 : cuda_ops::DataType::kFloat16;
  const cuda_ops::TensorView input_view{d_input, rows, cols, row_stride, dtype};
  const cuda_ops::TensorView residual_view{d_residual, rows, cols, row_stride, dtype};
  const cuda_ops::TensorView weight_view{d_weight, 1, cols, cols, dtype};
  const cuda_ops::TensorView output_view{d_output, rows, cols, row_stride, dtype};
  const cuda_ops::RmsNormConfig config{1.0e-6F, true};
  const auto status = cuda_ops::rmsNormForward(input_view, residual ? &residual_view : nullptr, weight_view,
                                                output_view, config, d_inv_rms);
  if (status != cuda_ops::Status::kSuccess) throw std::runtime_error("kernel launch failed");
  CUDA_OPS_CHECK(cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));
  CUDA_OPS_CHECK(cudaMemcpy(inv_rms.data(), d_inv_rms, inv_rms.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OPS_CHECK(cudaDeviceSynchronize());

  const float tolerance = std::is_same_v<T, float> ? 2.0e-5F : 3.0e-3F;
  float max_error = 0.0F;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      max_error = std::max(max_error, std::abs(toFloat(output[row * row_stride + col]) - expected[row * cols + col]));
    }
  }
  if (max_error > tolerance) {
    throw std::runtime_error("maximum error " + std::to_string(max_error) + " exceeds tolerance");
  }
  for (int64_t row = 0; row < rows; ++row) {
    if (std::abs(inv_rms[row] - expected_inv_rms[row]) > 2.0e-5F) {
      throw std::runtime_error("inv_rms output does not match reference");
    }
  }
  CUDA_OPS_CHECK(cudaFree(d_inv_rms));
  CUDA_OPS_CHECK(cudaFree(d_output));
  CUDA_OPS_CHECK(cudaFree(d_weight));
  CUDA_OPS_CHECK(cudaFree(d_residual));
  CUDA_OPS_CHECK(cudaFree(d_input));
  std::cout << "PASS: " << (std::is_same_v<T, float> ? "fp32" : "fp16") << " rows=" << rows
            << " cols=" << cols << " stride=" << row_stride << " residual=" << residual
            << " max_error=" << max_error << '\n';
}

void expectInvalidArgument() {
  cuda_ops::TensorView invalid{};
  const auto status = cuda_ops::rmsNormForward(invalid, nullptr, invalid, invalid, {}, nullptr);
  if (status != cuda_ops::Status::kInvalidArgument) throw std::runtime_error("invalid-input contract failed");
}

}  // namespace

int main() {
  try {
    expectInvalidArgument();
    runCase<float>(17, 768, false);
    runCase<float>(8, 4097, true);
    runCase<__half>(31, 1024, false);
    runCase<__half>(9, 8192, true);
    runCase<__half>(11, 777, true, 1024);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
