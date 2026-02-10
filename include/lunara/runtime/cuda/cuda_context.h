#pragma once
#include "lunara/config.h"

#if LUNARA_HAS_CUDA
  #include <cuda_runtime.h>
#endif

namespace lunara {
namespace rt {

struct CudaContext {
#if LUNARA_HAS_CUDA
  cudaStream_t stream{nullptr};
  CudaContext();
  ~CudaContext();
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;
#else
  CudaContext() = default;
  ~CudaContext() = default;
#endif
};

struct CudaEventTimer {
#if LUNARA_HAS_CUDA
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
  CudaEventTimer();
  ~CudaEventTimer();
  void start(cudaStream_t s);
  void stop(cudaStream_t s);
  float elapsed_ms(); // sync stop event
#else
  CudaEventTimer() = default;
  ~CudaEventTimer() = default;
#endif
};

}
}
