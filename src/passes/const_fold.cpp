#include "lunara/passes/const_fold.h"
#include "lunara/runtime/cpu_ref.h"
#include "lunara/runtime/tensor.h"
#include "lunara/ir/op.h"
#include <sstream>
#include <vector>
#include <cstring>

namespace lunara::passes {

static bool is_const_value(const lunara::ir::Graph& g, lunara::ir::ValueId vid) {
  const auto& v = g.values[vid.v];
  if (v.producer.v == lunara::ir::kInvalidId) return false;
  const auto& p = g.ops[v.producer.v];
  return p.kind == lunara::ir::OpKind::Constant;
}

static bool get_attr(const lunara::ir::Op& op, const char* key, std::string& out) {
  for (const auto& a : op.attrs) {
    if (a.key == key) { out = a.value; return true; }
  }
  return false;
}

static std::vector<float> parse_csv_f32(const std::string& s) {
  std::vector<float> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    v.push_back(std::stof(item));
  }
  return v;
}

static lunara::Status materialize_const_tensor(const lunara::ir::Module& m,
                                               lunara::ir::ValueId vid,
                                               lunara::rt::Tensor& out) {
  const auto& g = m.g;
  const auto& v = g.values[vid.v];
  const auto& op = g.ops[v.producer.v];

  std::string data;
  if (!get_attr(op, "data_f32", data)) return lunara::Status::Error("const_fold: missing data_f32");

  // require static shape and f32
  if (v.type.dtype != lunara::ir::DType::f32) return lunara::Status::Error("const_fold: only f32");
  for (auto d : v.type.shape.dims) if (d < 0) return lunara::Status::Error("const_fold: requires static shape");

  out = lunara::rt::Tensor::empty_host(v.type.shape.dims, lunara::rt::DType::f32);
  auto vec = parse_csv_f32(data);
  if (vec.size() * sizeof(float) != out.bytes) return lunara::Status::Error("const_fold: data size mismatch");

  std::memcpy(out.data, vec.data(), out.bytes);
  return lunara::Status::Ok();
}

static void write_const_attr(lunara::ir::Op& op, const lunara::rt::Tensor& t) {
  // overwrite/add data_f32
  std::ostringstream os;
  const float* p = (const float*)t.data;
  const std::size_t n = t.bytes / sizeof(float);
  for (std::size_t i = 0; i < n; i++) {
    os << p[i];
    if (i + 1 < n) os << ",";
  }
  bool found = false;
  for (auto& a : op.attrs) {
    if (a.key == "data_f32") { a.value = os.str(); found = true; break; }
  }
  if (!found) op.attrs.push_back({"data_f32", os.str()});
}

lunara::Status ConstFoldPass::run(lunara::ir::Module& m) {
  auto& g = m.g;

  // fold only small tensors
  constexpr std::int64_t kMaxFoldNumel = 256;

  for (auto& op : g.ops) {
    using lunara::ir::OpKind;
    if (!(op.kind == OpKind::Add || op.kind == OpKind::Mul || op.kind == OpKind::Relu || op.kind == OpKind::MatMul))
      continue;

    bool all_const = true;
    for (auto in : op.inputs) {
      if (!is_const_value(g, in)) { all_const = false; break; }
    }
    if (!all_const) continue;

    auto out_id = op.outputs[0];
    const auto& outv = g.values[out_id.v];
    if (outv.type.dtype != lunara::ir::DType::f32) continue;
    if (!outv.type.shape.is_static()) continue;
    if (outv.type.shape.numel_static() > kMaxFoldNumel) continue;

    // materialize inputs
    std::vector<lunara::rt::Tensor> ins;
    ins.reserve(op.inputs.size());
    for (auto in : op.inputs) {
      lunara::rt::Tensor t;
      LUNARA_RETURN_IF_ERROR(materialize_const_tensor(m, in, t));
      ins.push_back(std::move(t));
    }

    // compute output
    lunara::rt::Tensor out = lunara::rt::Tensor::empty_host(outv.type.shape.dims, lunara::rt::DType::f32);
    lunara::Status st = lunara::Status::Ok();

    if (op.kind == OpKind::Add) st = lunara::rt::cpu::add(ins[0], ins[1], out);
    else if (op.kind == OpKind::Mul) st = lunara::rt::cpu::mul(ins[0], ins[1], out);
    else if (op.kind == OpKind::Relu) st = lunara::rt::cpu::relu(ins[0], out);
    else if (op.kind == OpKind::MatMul) st = lunara::rt::cpu::matmul(ins[0], ins[1], out);

    if (!st.ok()) return st;

    // Replace op with Constant producing same outputs
    op.kind = OpKind::Constant;
    op.inputs.clear();
    op.attrs.clear();
    write_const_attr(op, out);
  }

  return lunara::Status::Ok();
}

}
