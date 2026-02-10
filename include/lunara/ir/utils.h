#pragma once
#include "lunara/ir/type.h"
#include <cstdint>

namespace lunara::ir {

inline std::int64_t rank(const TensorType& t) {
  return (std::int64_t)t.shape.dims.size();
}

inline bool same_shape(const TensorType& a, const TensorType& b) {
  return a.shape.dims == b.shape.dims;
}

}
