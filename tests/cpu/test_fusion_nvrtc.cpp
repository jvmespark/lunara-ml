#include "lunara/ir/module.h"
#include "lunara/passes/shape_infer.h"
#include "lunara/passes/fusion.h"
#include "lunara/runtime/jit/fusion_jit.h"
#include "lunara/util/assert.h"

int main() {
#if !LUNARA_HAS_CUDA
  // If you compiled with CUDA OFF, skip.
  return 0;
#else
  using namespace lunara::ir;

  // Graph: z = relu((x + b) * b)
  Module m;
  auto& g = m.g;

  TensorType t; t.dtype = DType::f32; t.shape.dims = {1024};

  auto x = g.add_value(t, "x");
  auto b = g.add_value(t, "b");
  g.inputs = {x, b};

  auto add0 = g.add_op(OpKind::Add, {x, b}, 1, "add0");
  auto y = g.op(add0).outputs[0];

  auto mul0 = g.add_op(OpKind::Mul, {y, b}, 1, "mul0");
  auto w = g.op(mul0).outputs[0];

  auto relu0 = g.add_op(OpKind::Relu, {w}, 1, "relu0");
  auto z = g.op(relu0).outputs[0];
  g.set_graph_outputs({z});

  // Infer shapes
  lunara::passes::ShapeInferPass si;
  auto st = si.run(m);
  LUNARA_CHECK(st.ok());

  // Fusion plans
  auto plans = lunara::passes::build_fusion_plans(m);
  LUNARA_CHECK(plans.ok());
  LUNARA_CHECK(!plans.value().empty());

  // Compile to PTX (should cache)
  auto ptx1 = lunara::rt::jit::compile_fusion_to_ptx(plans.value()[0]);
  LUNARA_CHECK(ptx1.ok());
  LUNARA_CHECK(!ptx1.value().empty());

  auto ptx2 = lunara::rt::jit::compile_fusion_to_ptx(plans.value()[0]);
  LUNARA_CHECK(ptx2.ok());
  LUNARA_CHECK(!ptx2.value().empty());

  return 0;
#endif
}