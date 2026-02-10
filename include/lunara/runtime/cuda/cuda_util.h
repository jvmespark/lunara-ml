#pragma once
#include "lunara/config.h"
#include "lunara/util/status.h"
#include <string>

#if LUNARA_HAS_CUDA
  #include <cuda_runtime.h>
#endif

namespace lunara {
namespace rt {
namespace cuda {

#if LUNARA_HAS_CUDA
inline lunara::Status cuda_ok(cudaError_t e, const char* what) {
  if (e == cudaSuccess) {
    return lunara::Status::Ok();
  }
  return lunara::Status::Error(std::string(what) + ": " + cudaGetErrorString(e));
}
#endif

bool has_cuda_device(); // uses cudaGetDeviceCount

}
}
}
