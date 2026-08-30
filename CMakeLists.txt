cmake_minimum_required(VERSION 3.26)
project(cuda_ops VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(CUDAToolkit REQUIRED)

option(CUDA_OPS_BUILD_TESTS "Build correctness tests" ON)
option(CUDA_OPS_BUILD_BENCHMARKS "Build microbenchmarks" ON)
set(CUDA_OPS_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures to compile")
set(CUDA_OPS_RMSNORM_THREADS "512" CACHE STRING "Threads per fused RMSNorm block (128, 256, or 512)")

if(NOT CUDA_OPS_RMSNORM_THREADS MATCHES "^(128|256|512)$")
  message(FATAL_ERROR "CUDA_OPS_RMSNORM_THREADS must be 128, 256, or 512")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_OPS_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(cuda_ops STATIC src/rmsnorm.cu)
add_library(cuda_ops::cuda_ops ALIAS cuda_ops)
target_include_directories(cuda_ops PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cuda_ops PUBLIC CUDA::cudart)
target_compile_features(cuda_ops PUBLIC cxx_std_20)
target_compile_definitions(cuda_ops PRIVATE CUDA_OPS_RMSNORM_THREADS=${CUDA_OPS_RMSNORM_THREADS})
target_compile_options(cuda_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
if(MSVC)
  target_compile_options(cuda_ops PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
else()
  target_compile_options(cuda_ops PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>)
endif()
set_target_properties(cuda_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

install(TARGETS cuda_ops EXPORT cuda_opsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cuda_opsTargets NAMESPACE cuda_ops:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_ops)
configure_package_config_file(cmake/cuda_opsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuda_opsConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_ops)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/cuda_opsConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuda_opsConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cuda_opsConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_ops)

if(CUDA_OPS_BUILD_TESTS)
  enable_testing()
  add_executable(rmsnorm_test tests/rmsnorm_test.cu)
  target_link_libraries(rmsnorm_test PRIVATE cuda_ops)
  add_test(NAME rmsnorm_correctness COMMAND rmsnorm_test)
endif()

if(CUDA_OPS_BUILD_BENCHMARKS)
  add_executable(rmsnorm_bench benchmarks/rmsnorm_bench.cu)
  target_link_libraries(rmsnorm_bench PRIVATE cuda_ops)
endif()
