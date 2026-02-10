#pragma once
#include "lunara/ir/dtype.h"
#include "lunara/ir/shape.h"

namespace lunara::ir {

struct TensorType {
  DType dtype{DType::unknown};
  Shape shape{};
};

} // namespace lunara::ir

