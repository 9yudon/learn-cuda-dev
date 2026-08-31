FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_OPS_CUDA_ARCHITECTURES=80 && \
    cmake --build build --parallel
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
