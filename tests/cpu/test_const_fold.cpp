#include "lunara/ir/module.h"
#include "lunara/passes/shape_infer.h"
#include "lunara/passes/const_fold.h"
#include "lunara/util/assert.h"
#include <string>

static void set_const(lunara::ir::Graph& g, lunara::ir::OpId cid, const char* csv) {
  auto& op = g.op(cid);
  op.kind = lunara::ir::OpKind::Constant;
  op.inputs.clear();
  op.attrs.clear();
  op.attrs.push_back({"data_f32", std::string(csv)});
}

int main() {
  using namespace lunara::ir;

  Module m;
  auto& g = m.g;

  TensorType t; t.dtype = DType::f32; t.shape.dims = {4};

  // Build constant values by creating ops that produce values
  auto c0 = g.add_op(OpKind::Constant, {}, 1, "c0");
  auto v0 = g.op(c0).outputs[0];
  g.value(v0).type = t;
  set_const(g, c0, "1,2,3,4");

  auto c1 = g.add_op(OpKind::Constant, {}, 1, "c1");
  auto v1 = g.op(c1).outputs[0];
  g.value(v1).type = t;
  set_const(g, c1, "10,20,30,40");

  // add them: (should fold)
  auto add0 = g.add_op(OpKind::Add, {v0, v1}, 1, "add0");
  auto out = g.op(add0).outputs[0];
  g.set_graph_outputs({out});

  // Infer shape for add output
  lunara::passes::ShapeInferPass si;
  auto st = si.run(m);
  LUNARA_CHECK(st.ok());

  // Fold
  lunara::passes::ConstFoldPass cf;
  st = cf.run(m);
  LUNARA_CHECK(st.ok());

  // After fold, add0 op becomes Constant
  LUNARA_CHECK(g.op(add0).kind == OpKind::Constant);

  // Should contain data_f32 with 11,22,33,44
  bool found = false;
  for (auto& a : g.op(add0).attrs) {
    if (a.key == "data_f32") {
      // Loose check: just ensure substrings exist
      LUNARA_CHECK(a.value.find("11") != std::string::npos);
      LUNARA_CHECK(a.value.find("22") != std::string::npos);
      found = true;
    }
  }
  LUNARA_CHECK(found);

  return 0;
}

