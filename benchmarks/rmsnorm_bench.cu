#include "cuda_ops/cuda_check.cuh"
#include "cuda_ops/rmsnorm.h"

#include <cuda_fp16.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
  int64_t rows{4096};
  int64_t cols{4096};
  int warmup{50};
  int iterations{200};
  bool residual{true};
  cuda_ops::DataType dtype{cuda_ops::DataType::kFloat16};
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string_view token(argv[i]);
    if (token == "--rows" && i + 1 < argc) args.rows = std::atoll(argv[++i]);
    else if (token == "--cols" && i + 1 < argc) args.cols = std::atoll(argv[++i]);
    else if (token == "--warmup" && i + 1 < argc) args.warmup = std::atoi(argv[++i]);
    else if (token == "--iterations" && i + 1 < argc) args.iterations = std::atoi(argv[++i]);
    else if (token == "--dtype" && i + 1 < argc) {
      const std::string_view dtype(argv[++i]);
      args.dtype = dtype == "fp32" ? cuda_ops::DataType::kFloat32 : cuda_ops::DataType::kFloat16;
    } else if (token == "--no-residual") args.residual = false;
    else throw std::invalid_argument("unknown or incomplete argument: " + std::string(token));
  }
  if (args.rows <= 0 || args.cols <= 0 || args.iterations <= 0 || args.warmup < 0) {
    throw std::invalid_argument("rows, cols, and iterations must be positive");
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    const size_t element_size = args.dtype == cuda_ops::DataType::kFloat32 ? sizeof(float) : sizeof(__half);
    const size_t matrix_bytes = static_cast<size_t>(args.rows) * args.cols * element_size;
    void *input = nullptr, *residual = nullptr, *weight = nullptr, *output = nullptr;
    CUDA_OPS_CHECK(cudaMalloc(&input, matrix_bytes));
    CUDA_OPS_CHECK(cudaMalloc(&residual, matrix_bytes));
    CUDA_OPS_CHECK(cudaMalloc(&weight, static_cast<size_t>(args.cols) * element_size));
    CUDA_OPS_CHECK(cudaMalloc(&output, matrix_bytes));
    CUDA_OPS_CHECK(cudaMemset(input, 1, matrix_bytes));
    CUDA_OPS_CHECK(cudaMemset(residual, 1, matrix_bytes));
    CUDA_OPS_CHECK(cudaMemset(weight, 1, static_cast<size_t>(args.cols) * element_size));

    const cuda_ops::TensorView input_view{input, args.rows, args.cols, args.cols, args.dtype};
    const cuda_ops::TensorView residual_view{residual, args.rows, args.cols, args.cols, args.dtype};
    const cuda_ops::TensorView weight_view{weight, 1, args.cols, args.cols, args.dtype};
    const cuda_ops::TensorView output_view{output, args.rows, args.cols, args.cols, args.dtype};
    const cuda_ops::RmsNormConfig config{};
    for (int i = 0; i < args.warmup; ++i) {
      if (cuda_ops::rmsNormForward(input_view, args.residual ? &residual_view : nullptr, weight_view, output_view,
                                   config, nullptr) != cuda_ops::Status::kSuccess) throw std::runtime_error("warmup failed");
    }
    cudaEvent_t start{}, stop{};
    CUDA_OPS_CHECK(cudaEventCreate(&start));
    CUDA_OPS_CHECK(cudaEventCreate(&stop));
    CUDA_OPS_CHECK(cudaEventRecord(start));
    for (int i = 0; i < args.iterations; ++i) {
      if (cuda_ops::rmsNormForward(input_view, args.residual ? &residual_view : nullptr, weight_view, output_view,
                                   config, nullptr) != cuda_ops::Status::kSuccess) throw std::runtime_error("benchmark launch failed");
    }
    CUDA_OPS_CHECK(cudaEventRecord(stop));
    CUDA_OPS_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0F;
    CUDA_OPS_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double latency_us = static_cast<double>(total_ms) * 1000.0 / args.iterations;
    // Read input, optional residual, weight, and write output once per invocation.
    const double traffic_bytes = static_cast<double>(args.rows) * args.cols * element_size * (args.residual ? 4.0 : 3.0);
    const double gb_per_second = traffic_bytes / (latency_us * 1.0e3);
    std::cout << std::fixed << std::setprecision(3)
              << "{\"operator\":\"fused_rmsnorm\",\"rows\":" << args.rows
              << ",\"cols\":" << args.cols << ",\"dtype\":\""
              << (args.dtype == cuda_ops::DataType::kFloat32 ? "fp32" : "fp16")
              << "\",\"residual\":" << (args.residual ? "true" : "false")
              << ",\"latency_us\":" << latency_us << ",\"effective_gbps\":" << gb_per_second << "}" << '\n';
    CUDA_OPS_CHECK(cudaEventDestroy(stop));
    CUDA_OPS_CHECK(cudaEventDestroy(start));
    CUDA_OPS_CHECK(cudaFree(output));
    CUDA_OPS_CHECK(cudaFree(weight));
    CUDA_OPS_CHECK(cudaFree(residual));
    CUDA_OPS_CHECK(cudaFree(input));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
