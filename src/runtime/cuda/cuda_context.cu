#include "lunara/runtime/cuda/cuda_context.h"
#include "lunara/runtime/cuda/cuda_util.h"
#include "lunara/util/assert.h"

namespace lunara {
namespace rt {

#if LUNARA_HAS_CUDA
CudaContext::CudaContext() {
  auto st = lunara::rt::cuda::cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate");
  LUNARA_CHECK(st.ok());
}
CudaContext::~CudaContext() {
  if (stream) {
    cudaStreamDestroy(stream);
  }
}

CudaEventTimer::CudaEventTimer() {
  LUNARA_CHECK(cudaEventCreate(&start_) == cudaSuccess);
  LUNARA_CHECK(cudaEventCreate(&stop_) == cudaSuccess);
}
CudaEventTimer::~CudaEventTimer() {
  if (start_) {
    cudaEventDestroy(start_);
  }
  if (stop_) {
    cudaEventDestroy(stop_);
  }
}
void CudaEventTimer::start(cudaStream_t s) {
  cudaEventRecord(start_, s);
}
void CudaEventTimer::stop(cudaStream_t s) {
  cudaEventRecord(stop_, s);
}
float CudaEventTimer::elapsed_ms() {
  cudaEventSynchronize(stop_);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start_, stop_);
  return ms;
}
#endif

}
}
