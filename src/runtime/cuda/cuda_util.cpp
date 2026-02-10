#include "lunara/runtime/cuda/cuda_util.h"

namespace lunara {
namespace rt {
namespace cuda {

bool has_cuda_device() {
#if !LUNARA_HAS_CUDA
  return false;
#else
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    return false;
  }
  return count > 0;
#endif
}

}
}
}
