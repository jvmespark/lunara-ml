#include "lunara/runtime/interpreter.h"
#include "lunara/runtime/cpu_ref.h"
#include "lunara/ir/op.h"
#include "lunara/ir/dtype.h"

#include <cstring>

namespace lunara::rt {

static std::string value_key(const lunara::ir::Graph& g, lunara::ir::ValueId vid) {
  const auto& v = g.values[vid.v];
  if (!v.name.empty()) return v.name;
  return std::string("%") + std::to_string(vid.v);
}

static lunara::Status ensure_f32_host(const Tensor& t) {
  if (t.device != Device::Host) return lunara::Status::Error("interpreter: expected Host tensor");
  if (t.dtype != DType::f32) return lunara::Status::Error("interpreter: expected f32 tensor");
  return lunara::Status::Ok();
}

static lunara::Status alloc_for_value(const lunara::ir::Value& v, Tensor& out) {
  // v1: only f32
  if (v.type.dtype != lunara::ir::DType::f32) return lunara::Status::Error("interpreter: only f32 supported");
  // require static shapes for now
  for (auto d : v.type.shape.dims) if (d < 0) return lunara::Status::Error("interpreter: requires static shapes");
  std::vector<std::int64_t> shape = v.type.shape.dims;
  out = Tensor::empty_host(shape, DType::f32);
  return lunara::Status::Ok();
}

RunResult CpuInterpreter::run(const lunara::ir::Module& m,
                              const std::unordered_map<std::string, Tensor>& feeds) {
  RunResult rr;
  rr.status = lunara::Status::Ok();

  const auto& g = m.g;

  // Storage for computed values by ValueId
  std::vector<Tensor> slots;
  slots.resize(g.values.size());

  // Bind graph inputs
  for (auto vid : g.inputs) {
    const auto& v = g.values[vid.v];
    auto it = feeds.find(v.name);
    if (it == feeds.end()) {
      rr.status = lunara::Status::Error("interpreter: missing input feed");
      return rr;
    }
    auto st = ensure_f32_host(it->second);
    if (!st.ok()) { rr.status = st; return rr; }
    // shallow move/copy into slot: we need our own storage => copy bytes
    Tensor copy = Tensor::empty_host(it->second.shape, it->second.dtype);
    std::memcpy(copy.data, it->second.data, it->second.bytes);
    slots[vid.v] = std::move(copy);
  }

  // Execute ops in order
  for (const auto& op : g.ops) {
    using lunara::ir::OpKind;

    if (op.kind == OpKind::Add || op.kind == OpKind::Mul) {
      auto a_id = op.inputs[0];
      auto b_id = op.inputs[1];
      auto o_id = op.outputs[0];

      Tensor out;
      auto st = alloc_for_value(g.values[o_id.v], out);
      if (!st.ok()) { rr.status = st; return rr; }

      if (op.kind == OpKind::Add) st = cpu::add(slots[a_id.v], slots[b_id.v], out);
      else st = cpu::mul(slots[a_id.v], slots[b_id.v], out);

      if (!st.ok()) { rr.status = st; return rr; }
      slots[o_id.v] = std::move(out);
    }

    else if (op.kind == OpKind::Relu) {
      auto a_id = op.inputs[0];
      auto o_id = op.outputs[0];

      Tensor out;
      auto st = alloc_for_value(g.values[o_id.v], out);
      if (!st.ok()) { rr.status = st; return rr; }

      st = cpu::relu(slots[a_id.v], out);
      if (!st.ok()) { rr.status = st; return rr; }
      slots[o_id.v] = std::move(out);
    }

    else if (op.kind == OpKind::MatMul) {
      auto A_id = op.inputs[0];
      auto B_id = op.inputs[1];
      auto C_id = op.outputs[0];

      Tensor out;
      auto st = alloc_for_value(g.values[C_id.v], out);
      if (!st.ok()) { rr.status = st; return rr; }

      st = cpu::matmul(slots[A_id.v], slots[B_id.v], out);
      if (!st.ok()) { rr.status = st; return rr; }
      slots[C_id.v] = std::move(out);
    }

    else {
      rr.status = lunara::Status::Error("interpreter: unsupported op kind");
      return rr;
    }
  }

  // Collect outputs
  for (auto vid : g.outputs) {
    rr.outputs[value_key(g, vid)] = std::move(slots[vid.v]);
  }

  return rr;
}

}
