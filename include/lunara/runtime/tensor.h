#pragma once
#include "lunara/runtime/device.h"
#include "lunara/util/status.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lunara::rt {

enum class DType : std::uint8_t {
  f32
};

inline std::size_t dtype_size(DType) {
  return sizeof(float);
}

struct Tensor {
  Device device{Device::Host};
  DType dtype{DType::f32};
  std::vector<std::int64_t> shape;
  std::size_t bytes{0};
  void* data{nullptr};

  static Tensor empty_host(std::vector<std::int64_t> shape, DType dt);
  static lunara::Result<Tensor> empty_cuda(std::vector<std::int64_t> shape, DType dt);

  lunara::Status to_cuda();
  lunara::Status to_host();

  ~Tensor();
  Tensor() = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) noexcept;
  Tensor& operator=(Tensor&&) noexcept;
};

std::int64_t numel(const std::vector<std::int64_t>& shape);

} // namespace lunara::rt

