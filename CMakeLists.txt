cmake_minimum_required(VERSION 3.24)
project(lunara_ml LANGUAGES CXX)

option(LUNARA_ENABLE_CUDA      "Build CUDA backend (NVRTC + runtime wrappers)" ON)
option(LUNARA_ENABLE_TESTS     "Build tests" ON)
option(LUNARA_ENABLE_GPUTESTS  "Build GPU execution tests (built only if enabled)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Warnings
function(lunara_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-unused-parameter>
  )
endfunction()

# CUDA optional
if(LUNARA_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70;80 CACHE STRING "CUDA archs" FORCE)
  endif()
endif()

add_library(lunara
  src/util/log.cpp
  src/util/timer.cpp
  src/runtime/tensor.cpp
  src/runtime/cpu_ref.cpp
  src/runtime/interpreter.cpp
  src/passes/shape_infer.cpp
  src/passes/const_fold.cpp 
)

target_include_directories(lunara PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(lunara PUBLIC
  $<$<BOOL:${LUNARA_ENABLE_CUDA}>:lunara_ENABLE_CUDA=1>
)

lunara_warnings(lunara)

target_sources(lunara PRIVATE
  src/lunara/ir/graph.cpp
  src/lunara/ir/verifier.cpp
  src/lunara/ir/printer.cpp
  src/lunara/frontend/json/graph_json_io.cpp
  src/passes/pass_manager.cpp
  src/codegen/cuda/cuda_emitter.cpp
  src/passes/fusion.cpp
  src/runtime/cache/ptx_cache.cpp
  src/runtime/jit/fusion_jit.cpp
)

if(LUNARA_ENABLE_CUDA)
  target_sources(lunara PRIVATE
    src/runtime/cuda/cuda_context.cu
    src/runtime/cuda/cuda_util.cpp
    src/runtime/cuda/nvrtc_compiler.cpp
  )
  target_link_libraries(lunara PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
  set_target_properties(lunara PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Tests (no gtest needed; keep it minimal)
if(LUNARA_ENABLE_TESTS)
  enable_testing()

  add_executable(test_tensor tests/cpu/test_tensor.cpp)
  target_link_libraries(test_tensor PRIVATE lunara)
  lunara_warnings(test_tensor)
  add_test(NAME test_tensor COMMAND test_tensor)

  add_executable(test_cpu_ref_ops tests/cpu/test_cpu_ref_ops.cpp)
  target_link_libraries(test_cpu_ref_ops PRIVATE lunara)
  lunara_warnings(test_cpu_ref_ops)
  add_test(NAME test_cpu_ref_ops COMMAND test_cpu_ref_ops)

  add_executable(test_ir_build tests/cpu/test_ir_build.cpp)
  target_link_libraries(test_ir_build PRIVATE lunara)
  lunara_warnings(test_ir_build)
  add_test(NAME test_ir_build COMMAND test_ir_build)

  add_executable(test_json_import tests/cpu/test_json_import.cpp)
  target_link_libraries(test_json_import PRIVATE lunara)
  lunara_warnings(test_json_import)
  add_test(NAME test_json_import COMMAND test_json_import)

  add_executable(test_shape_infer tests/cpu/test_shape_infer.cpp)
  target_link_libraries(test_shape_infer PRIVATE lunara)
  lunara_warnings(test_shape_infer)
  add_test(NAME test_shape_infer COMMAND test_shape_infer)

  add_executable(test_interpreter_golden tests/cpu/test_interpreter_golden.cpp)
  target_link_libraries(test_interpreter_golden PRIVATE lunara)
  lunara_warnings(test_interpreter_golden)
  add_test(NAME test_interpreter_golden COMMAND test_interpreter_golden)

  add_executable(test_const_fold tests/cpu/test_const_fold.cpp)
  target_link_libraries(test_const_fold PRIVATE lunara)
  lunara_warnings(test_const_fold)
  add_test(NAME test_const_fold COMMAND test_const_fold)

  add_executable(test_fusion_nvrtc tests/cpu/test_fusion_nvrtc.cpp)
  target_link_libraries(test_fusion_nvrtc PRIVATE lunara)
  add_test(NAME test_fusion_nvrtc COMMAND test_fusion_nvrtc)


  if(LUNARA_ENABLE_CUDA)
    # NVRTC compile-only test (no GPU required)
    add_executable(test_codegen_nvrtc tests/cuda/test_codegen_nvrtc.cpp)
    target_link_libraries(test_codegen_nvrtc PRIVATE lunara)
    lunara_warnings(test_codegen_nvrtc)
    add_test(NAME test_codegen_nvrtc COMMAND test_codegen_nvrtc)

    # GPU execution tests (only if enabled)
    if(LUNARA_ENABLE_GPUTESTS)
      add_executable(test_cuda_device_smoke tests/cuda/test_cuda_device_smoke.cu)
      target_link_libraries(test_cuda_device_smoke PRIVATE lunara)
      add_test(NAME test_cuda_device_smoke COMMAND test_cuda_device_smoke)
      
      add_executable(bench_matmul tests/cuda/bench_matmul.cu)
      target_link_libraries(bench_matmul PRIVATE lunara)
      add_test(NAME bench_matmul COMMAND bench_matmul)
    endif()
  endif()
endif()

