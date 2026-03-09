#pragma once
#include "lunara/runtime/tensor.h"
#include "lunara/util/status.h"

namespace lunara::rt::cpu {

// All ops are deterministic, contiguous, f32 only for now
lunara::Status add(const Tensor& a, const Tensor& b, Tensor& out);
lunara::Status mul(const Tensor& a, const Tensor& b, Tensor& out);
lunara::Status relu(const Tensor& a, Tensor& out);
lunara::Status matmul(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace lunara::rt::cpu

