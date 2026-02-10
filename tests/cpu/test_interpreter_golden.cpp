#include "lunara/ir/module.h"
#include "lunara/passes/shape_infer.h"
#include "lunara/runtime/interpreter.h"
#include "lunara/util/assert.h"
#include <unordered_map>
#include <cmath>

static void fill_seq(lunara::rt::Tensor& t, float start) {
  float* p = (float*)t.data;
  const int n = (int)(t.bytes / sizeof(float));
  for (int i = 0; i < n; i++) p[i] = start + (float)i;
}

static bool approx_eq(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

int main() {
  using namespace lunara::ir;

  // Graph: z = relu((x + b) * b)
  Module m;
  auto& g = m.g;

  TensorType t; t.dtype = DType::f32; t.shape.dims = {8};

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

  // Infer shapes so interpreter can allocate outputs
  lunara::passes::ShapeInferPass si;
  auto st = si.run(m);
  LUNARA_CHECK(st.ok());

  // Feeds
  std::unordered_map<std::string, lunara::rt::Tensor> feeds;
  feeds["x"] = lunara::rt::Tensor::empty_host({8}, lunara::rt::DType::f32);
  feeds["b"] = lunara::rt::Tensor::empty_host({8}, lunara::rt::DType::f32);
  fill_seq(feeds["x"], -4.0f); // [-4..3]
  fill_seq(feeds["b"],  1.0f); // [1..8]

  lunara::rt::CpuInterpreter interp;
  auto rr = interp.run(m, feeds);
  LUNARA_CHECK(rr.status.ok());

  // Golden check a few entries by hand:
  // y = x + b
  // w = y * b
  // z = relu(w)
  auto& out = rr.outputs["%"+std::to_string(z.v)]; // unnamed output uses "%id"
  float* p = (float*)out.data;

  // i=0: x=-4, b=1 => y=-3 => w=-3 => relu=0
  LUNARA_CHECK(approx_eq(p[0], 0.0f));
  // i=4: x=0, b=5 => y=5 => w=25 => relu=25
  LUNARA_CHECK(approx_eq(p[4], 25.0f));
  // i=7: x=3, b=8 => y=11 => w=88 => relu=88
  LUNARA_CHECK(approx_eq(p[7], 88.0f));

  return 0;
}

