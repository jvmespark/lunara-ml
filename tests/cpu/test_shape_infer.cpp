#include "lunara/ir/module.h"
#include "lunara/passes/shape_infer.h"
#include "lunara/util/assert.h"

int main() {
  using namespace lunara::ir;

  Module m;
  auto& g = m.g;

  TensorType A; A.dtype = DType::f32; A.shape.dims = {2, 3};
  TensorType B; B.dtype = DType::f32; B.shape.dims = {3, 4};

  auto a = g.add_value(A, "A");
  auto b = g.add_value(B, "B");
  g.inputs = {a, b};

  auto mm = g.add_op(OpKind::MatMul, {a, b}, 1, "mm");
  auto c = g.op(mm).outputs[0];
  g.set_graph_outputs({c});

  lunara::passes::ShapeInferPass p;
  auto st = p.run(m);
  LUNARA_CHECK(st.ok());

  const auto& ct = g.value(c).type;
  LUNARA_CHECK(ct.dtype == DType::f32);
  LUNARA_CHECK(ct.shape.dims.size() == 2);
  LUNARA_CHECK(ct.shape.dims[0] == 2);
  LUNARA_CHECK(ct.shape.dims[1] == 4);

  return 0;
}

