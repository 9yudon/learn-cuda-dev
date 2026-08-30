cmake_minimum_required(VERSION 3.26)
project(cuda_ops_consumer LANGUAGES CXX CUDA)

find_package(cuda_ops CONFIG REQUIRED)

add_executable(cuda_ops_consumer main.cpp)
target_link_libraries(cuda_ops_consumer PRIVATE cuda_ops::cuda_ops)
# A static library containing CUDA device code needs its final consumer to run device linking.
set_target_properties(cuda_ops_consumer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
