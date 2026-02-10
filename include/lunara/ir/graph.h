#pragma once
#include "lunara/ir/op.h"
#include "lunara/ir/value.h"
#include <vector>
#include <string>

namespace lunara::ir {

struct Graph {
  std::vector<Value> values;
  std::vector<Op> ops;

  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  ValueId add_value(const TensorType& t, std::string name = "");
  OpId add_op(OpKind kind, const std::vector<ValueId>& in, int out_count, std::string name = "");

  Value& value(ValueId id);
  const Value& value(ValueId id) const;

  Op& op(OpId id);
  const Op& op(OpId id) const;

  void set_graph_outputs(const std::vector<ValueId>& outs);
};

} // namespace lunara::ir

