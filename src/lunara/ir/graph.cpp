#include "lunara/ir/graph.h"
#include <stdexcept>

namespace lunara::ir {

ValueId Graph::add_value(const TensorType& t, std::string name) {
  ValueId id{ static_cast<std::uint32_t>(values.size()) };
  Value v;
  v.id = id;
  v.type = t;
  v.name = std::move(name);
  v.producer = OpId{kInvalidId};
  values.push_back(std::move(v));
  return id;
}

OpId Graph::add_op(OpKind kind, const std::vector<ValueId>& in, int out_count, std::string name) {
  OpId oid{ static_cast<std::uint32_t>(ops.size()) };

  Op o;
  o.id = oid;
  o.kind = kind;
  o.inputs = in;
  o.name = std::move(name);

  // register uses
  for (auto vid : o.inputs) {
    if (vid.v >= values.size()) {
      throw std::runtime_error("add_op: input ValueId out of range");
    }
    values[vid.v].users.push_back(oid);
  }

  // create outputs (types filled later by shape inference)
  o.outputs.reserve((std::size_t)out_count);
  for (int i = 0; i < out_count; i++) {
    TensorType unknown{};
    unknown.dtype = DType::unknown;
    ValueId out = add_value(unknown, "");
    values[out.v].producer = oid;
    o.outputs.push_back(out);
  }

  ops.push_back(std::move(o));
  return oid;
}

Value& Graph::value(ValueId id) {
  return values.at(id.v);
}
const Value& Graph::value(ValueId id) const {
  return values.at(id.v);
}

Op& Graph::op(OpId id) {
  return ops.at(id.v);
}
const Op& Graph::op(OpId id) const {
  return ops.at(id.v);
}

void Graph::set_graph_outputs(const std::vector<ValueId>& outs) {
  outputs = outs;
}

} // namespace lunara::ir

