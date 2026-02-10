#pragma once
#include "lunara/ir/ids.h"
#include "lunara/ir/type.h"
#include <string>
#include <vector>

namespace lunara::ir {

struct Value {
  ValueId id{};
  TensorType type{};
  OpId producer{ kInvalidId };            // op that defines this value, invalid for graph inputs
  std::vector<OpId> users{};              // ops that consume this value
  std::string name{};                     // optional, for debugging / tool feeds
};

} // namespace lunara::ir

