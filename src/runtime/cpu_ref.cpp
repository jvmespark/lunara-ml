#include "lunara/runtime/cpu_ref.h"
#include "lunara/util/assert.h"
#include <cstring>

namespace lunara::rt::cpu {

static lunara::Status check_f32_host_same(const Tensor& a, const Tensor& b, const Tensor& o) {
  if (a.device != Device::Host || b.device != Device::Host || o.device != Device::Host) {
    return lunara::Status::Error("cpu_ref requires Host tensors");
  }
  if (a.dtype != DType::f32 || b.dtype != DType::f32 || o.dtype != DType::f32) {
    return lunara::Status::Error("cpu_ref requires f32");
  }
  if (a.shape != b.shape || a.shape != o.shape) {
    return lunara::Status::Error("cpu_ref requires same shapes");
  }
  return lunara::Status::Ok();
}

static lunara::Status check_f32_host_unary(const Tensor& a, const Tensor& o) {
  if (a.device != Device::Host || o.device != Device::Host) {
    return lunara::Status::Error("cpu_ref requires Host tensors");
  }
  if (a.dtype != DType::f32 || o.dtype != DType::f32) {
    return lunara::Status::Error("cpu_ref requires f32");
  }
  if (a.shape != o.shape) {
    return lunara::Status::Error("cpu_ref requires same shapes");
  }
  return lunara::Status::Ok();
}

lunara::Status add(const Tensor& a, const Tensor& b, Tensor& out) {
  LUNARA_RETURN_IF_ERROR(check_f32_host_same(a,b,out));
  auto n = (std::size_t)(a.bytes / sizeof(float));
  const float* ap = (const float*)a.data;
  const float* bp = (const float*)b.data;
  float* op = (float*)out.data;
  for (std::size_t i = 0; i < n; i++) {
    op[i] = ap[i] + bp[i];
  }
  return lunara::Status::Ok();
}

lunara::Status mul(const Tensor& a, const Tensor& b, Tensor& out) {
  LUNARA_RETURN_IF_ERROR(check_f32_host_same(a,b,out));
  auto n = (std::size_t)(a.bytes / sizeof(float));
  const float* ap = (const float*)a.data;
  const float* bp = (const float*)b.data;
  float* op = (float*)out.data;
  for (std::size_t i = 0; i < n; i++) {
    op[i] = ap[i] * bp[i];
  }
  return lunara::Status::Ok();
}

lunara::Status relu(const Tensor& a, Tensor& out) {
  LUNARA_RETURN_IF_ERROR(check_f32_host_unary(a,out));
  auto n = (std::size_t)(a.bytes / sizeof(float));
  const float* ap = (const float*)a.data;
  float* op = (float*)out.data;
  for (std::size_t i = 0; i < n; i++) {
    float v = ap[i];
    op[i] = (v > 0.0f) ? v : 0.0f;
  }
  return lunara::Status::Ok();
}

lunara::Status matmul(const Tensor& A, const Tensor& B, Tensor& C) {
  if (A.device != Device::Host || B.device != Device::Host || C.device != Device::Host) {
    return lunara::Status::Error("cpu_ref matmul requires Host tensors");
  }
  if (A.dtype != DType::f32 || B.dtype != DType::f32 || C.dtype != DType::f32) {
    return lunara::Status::Error("cpu_ref matmul requires f32");
  }
  if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2) {
    return lunara::Status::Error("cpu_ref matmul requires rank-2 tensors");
  }

  const auto M = A.shape[0];
  const auto K = A.shape[1];
  const auto K2 = B.shape[0];
  const auto N = B.shape[1];

  if (K != K2) {
    return lunara::Status::Error("cpu_ref matmul shape mismatch K");
  }
  if (C.shape[0] != M || C.shape[1] != N) {
    return lunara::Status::Error("cpu_ref matmul output shape mismatch");
  }

  const float* Ap = (const float*)A.data;
  const float* Bp = (const float*)B.data;
  float* Cp = (float*)C.data;

  // TODO: more efficient matmul 
  for (std::int64_t i = 0; i < M; i++) {
    for (std::int64_t j = 0; j < N; j++) {
      float acc = 0.0f;
      for (std::int64_t k = 0; k < K; k++) {
        acc += Ap[i*K + k] * Bp[k*N + j];
      }
      Cp[i*N + j] = acc;
    }
  }

  return lunara::Status::Ok();
}

} // namespace lunara::rt::cpu

