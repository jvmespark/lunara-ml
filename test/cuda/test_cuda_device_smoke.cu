#include "lunara/config.h"
#include "lunara/runtime/cuda/cuda_util.h"
#include "lunara/util/assert.h"
#include <cstdio>

#if LUNARA_HAS_CUDA
__global__ void k() {}
#endif

int main() {
#if !LUNARA_HAS_CUDA
  std::puts("SKIP (built without CUDA)");
  return 0;
#else
  if (!lunara::rt::cuda::has_cuda_device()) {
    std::puts("SKIP (no CUDA device present)");
    return 0;
  }
  k<<<1,1>>>();
  cudaDeviceSynchronize();
  std::puts("test_cuda_device_smoke OK");
  return 0;
#endif
}

