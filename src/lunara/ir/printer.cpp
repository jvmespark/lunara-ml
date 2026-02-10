#include "lunara/ir/printer.h"
#include "lunara/ir/op.h"
#include "lunara/ir/dtype.h"
#include <sstream>

namespace lunara::ir {

static void dump_shape(std::ostringstream& os, const Shape& s) {
  os << "[";
  for (std::size_t i = 0; i < s.dims.size(); i++) {
    os << s.dims[i];
    if (i + 1 < s.dims.size()) {
      os << ",";
    }
  }
  os << "]";
}

std::string dump_module(const Module& m) {
  const auto& g = m.g;
  std::ostringstream os;

  os << "=== Lunara IR ===\n";
  os << "Inputs:\n";
  for (auto vid : g.inputs) {
    const auto& v = g.values[vid.v];
    os << "  %" << vid.v << " : tensor";
    dump_shape(os, v.type.shape);
    os << "<" << to_string(v.type.dtype) << ">";
    if (!v.name.empty()) {
      os << "  ; name=" << v.name;
    }
    os << "\n";
  }

  os << "Ops:\n";
  for (const auto& op : g.ops) {
    os << "  @" << op.id.v << " " << to_string(op.kind);
    if (!op.name.empty()) {
      os << "  ; name=" << op.name;
    }
    os << "\n    in: ";
    for (std::size_t i = 0; i < op.inputs.size(); i++) {
      os << "%" << op.inputs[i].v;
      if (i + 1 < op.inputs.size()) {
        os << ", ";
      }
    }
    os << "\n    out: ";
    for (std::size_t i = 0; i < op.outputs.size(); i++) {
      const auto& v = g.values[op.outputs[i].v];
      os << "%" << op.outputs[i].v << ":tensor";
      dump_shape(os, v.type.shape);
      os << "<" << to_string(v.type.dtype) << ">";
      if (i + 1 < op.outputs.size()) {
        os << ", ";
      }
    }
    os << "\n";
  }

  os << "Outputs:\n";
  for (auto vid : g.outputs) {
    os << "  %" << vid.v << "\n";
  }

  return os.str();
}

} // namespace lunara::ir

