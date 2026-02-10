#include "lunara/runtime/tensor.h"
#include "lunara/util/assert.h"
#include <new>
#include <cstring>

#if LUNARA_HAS_CUDA
  #include <cuda_runtime.h>
#endif

namespace lunara::rt {

std::int64_t numel(const std::vector<std::int64_t>& shape) {
  std::int64_t n = 1;
  for (auto d : shape) {
    n *= d;
  }
  return n;
}

static std::size_t compute_bytes(const std::vector<std::int64_t>& shape, DType dt) {
  return (std::size_t)numel(shape) * dtype_size(dt);
}

Tensor Tensor::empty_host(std::vector<std::int64_t> shape_, DType dt) {
  Tensor t;
  t.device = Device::Host;
  t.dtype = dt;
  t.shape = std::move(shape_);
  t.bytes = compute_bytes(t.shape, t.dtype);
  t.data = ::operator new(t.bytes);
  std::memset(t.data, 0, t.bytes);
  return t;
}

lunara::Result<Tensor> Tensor::empty_cuda(std::vector<std::int64_t> shape_, DType dt) {
#if !LUNARA_HAS_CUDA
  return lunara::Result<Tensor>::Error("Built without CUDA support");
#else
  Tensor t;
  t.device = Device::Cuda;
  t.dtype = dt;
  t.shape = std::move(shape_);
  t.bytes = compute_bytes(t.shape, t.dtype);
  cudaError_t e = cudaMalloc(&t.data, t.bytes);
  if (e != cudaSuccess) {
    return lunara::Result<Tensor>::Error(std::string("cudaMalloc failed: ") + cudaGetErrorString(e));
  }
  cudaMemset(t.data, 0, t.bytes);
  return lunara::Result<Tensor>::Ok(std::move(t));
#endif
}

Tensor::~Tensor() {
  if (!data) {
    return;
  }
  if (device == Device::Host) {
    ::operator delete(data);
  } else {
#if LUNARA_HAS_CUDA
    cudaFree(data);
#endif
  }
  data = nullptr;
  bytes = 0;
}

Tensor::Tensor(Tensor&& o) noexcept {
  *this = std::move(o);
}
Tensor& Tensor::operator=(Tensor&& o) noexcept {
  if (this == &o) {
    return *this;
  }
  this->~Tensor();
  device = o.device;
  dtype = o.dtype;
  shape = std::move(o.shape);
  bytes = o.bytes;
  data = o.data;
  o.data = nullptr;
  o.bytes = 0;
  return *this;
}

lunara::Status Tensor::to_cuda() {
#if !LUNARA_HAS_CUDA
  return lunara::Status::Error("Built without CUDA support");
#else
  if (device == Device::Cuda) {
    return lunara::Status::ok();
  }
  void* dev = nullptr;
  cudaError_t e = cudaMalloc(&dev, bytes);
  if (e != cudaSuccess) {
    return lunara::Status::Error(std::string("cudaMalloc: ") + cudaGetErrorString(e));
  }
  e = cudaMemcpy(dev, data, bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    return lunara::Status::Error(std::string("cudaMemcpy H2D: ") + cudaGetErrorString(e));
  }
  ::operator delete(data);
  data = dev;
  device = Device::Cuda;
  return lunara::Status::ok();
#endif
}

lunara::Status Tensor::to_host() {
#if !LUNARA_HAS_CUDA
  return lunara::Status::Error("Built without CUDA support");
#else
  if (device == Device::Host) {
    return lunara::Status::ok();
  }
  void* host = ::operator new(bytes);
  cudaError_t e = cudaMemcpy(host, data, bytes, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    return lunara::Status::Error(std::string("cudaMemcpy D2H: ") + cudaGetErrorString(e));
  }
  cudaFree(data);
  data = host;
  device = Device::Host;
  return lunara::Status::ok();
#endif
}

} // namespace lunara::rt

