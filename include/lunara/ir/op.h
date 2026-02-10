#pragma once
#include "lunara/ir/ids.h"
#include <string>
#include <string_view>
#include <vector>

namespace lunara::ir {

enum class OpKind : std::uint16_t {
  Input,
  Constant,
  Add,
  Mul,
  Relu,
  MatMul,
};

inline std::string_view to_string(OpKind k) {
  switch (k) {
    case OpKind::Input:
      return "Input";
    case OpKind::Constant:
      return "Constant";
    case OpKind::Add:
      return "Add";
    case OpKind::Mul:
      return "Mul";
    case OpKind::Relu:
      return "Relu";
    case OpKind::MatMul:
      return "MatMul";
    default:
      return "UnknownOp";
  }
}

struct Attribute {
  std::string key;
  std::string value; // keep v1 simple (stringly typed); upgrade later to variant
};

struct Op {
  OpId id{};
  OpKind kind{OpKind::Input};
  std::vector<ValueId> inputs{};
  std::vector<ValueId> outputs{};
  std::vector<Attribute> attrs{};
  std::string name{};
};

} // namespace lunara::ir

