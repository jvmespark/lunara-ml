#include "lunara/ir/verifier.h"
#include "lunara/ir/ids.h"
#include <unordered_set>

namespace lunara::ir {

static lunara::Status err(const char* m) {
  return lunara::Status::Error(m);
}

lunara::Status verify_module(const Module& m) {
  const auto& g = m.g;

  // Value ids must match index.
  for (std::uint32_t i = 0; i < g.values.size(); i++) {
    if (g.values[i].id.v != i) {
      return err("verify: Value.id mismatch");
    }
  }
  for (std::uint32_t i = 0; i < g.ops.size(); i++) {
    if (g.ops[i].id.v != i) {
      return err("verify: Op.id mismatch");
    }
  }

  // Inputs must be valid and have invalid producer.
  for (auto vid : g.inputs) {
    if (vid.v >= g.values.size()) {
      return err("verify: graph input out of range");
    }
    if (is_valid(g.values[vid.v].producer)) {
      return err("verify: graph input has a producer");
    }
  }

  // Ops: inputs/outputs in range, outputs have correct producer, inputs have this op in users list.
  for (const auto& op : g.ops) {
    for (auto in : op.inputs) {
      if (in.v >= g.values.size()) {
        return err("verify: op input out of range");
      }
      bool found = false;
      for (auto u : g.values[in.v].users) {
        if (u.v == op.id.v) {
          found = true; break;
        }
      }
      if (!found) {
        return err("verify: missing use-list entry");
      }
    }
    for (auto out : op.outputs) {
      if (out.v >= g.values.size()) {
        return err("verify: op output out of range");
      }
      if (g.values[out.v].producer.v != op.id.v) {
        return err("verify: output producer mismatch");
      }
    }
  }

  // Outputs must be valid and either an input or produced by an op.
  for (auto vid : g.outputs) {
    if (vid.v >= g.values.size()) {
      return err("verify: graph output out of range");
    }
    const auto& v = g.values[vid.v];
    bool is_input = false;
    for (auto in : g.inputs) {
      if (in.v == vid.v) {
        is_input = true; break;
      }
    }
    if (!is_input && !is_valid(v.producer)) {
      return err("verify: graph output has no producer and is not an input");
    }
  }

  return lunara::Status::Ok();
}

} // namespace lunara::ir

