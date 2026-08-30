#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace cuda_ops {

inline void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA failure at ") + file + ":" + std::to_string(line) +
                             " for " + expression + ": " + cudaGetErrorString(status));
  }
}

}  // namespace cuda_ops

#define CUDA_OPS_CHECK(expression) ::cuda_ops::checkCuda((expression), #expression, __FILE__, __LINE__)

