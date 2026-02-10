#include "lunara/config.h"
#include "lunara/runtime/cuda/nvrtc_compiler.h"
#include "lunara/util/assert.h"
#include <cstdio>

int main() {
#if !LUNARA_HAS_CUDA
  std::puts("SKIP (built without CUDA/NVRTC)");
  return 0;
#else
  lunara::rt::NvrtcCompiler c;

  const char* src = R"CUDA(
  extern "C" __global__
  void add1(const float* a, const float* b, float* c, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) c[i] = a[i] + b[i];
  }
  )CUDA";

  auto ptx = c.compile_to_ptx(src, "add1",
    {
      "--std=c++14",
      "--use_fast_math"
      // You can add "--gpu-architecture=compute_80" etc on GPU boxes if desired.
    });

  LUNARA_CHECK(ptx.ok());
  LUNARA_CHECK(!ptx.value().empty());

  std::puts("test_codegen_nvrtc OK (PTX generated)");
  return 0;
#endif
}

