#include <cuda_ops/rmsnorm.h>

#include <iostream>

int main() {
  std::cout << "cuda_ops package available: "
            << cuda_ops::statusMessage(cuda_ops::Status::kSuccess) << '\n';
  return 0;
}
