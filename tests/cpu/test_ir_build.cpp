#include "lunara/ir/module.h"
#include "lunara/ir/verifier.h"
#include "lunara/ir/printer.h"
#include "lunara/util/assert.h"
#include <cstdio>

int main() {
  using namespace lunara::ir;

  Module m;
  auto& g = m.g;

  TensorType t;
  t.dtype = DType::f32;
  t.shape.dims = {4, 4};

  // inputs
  auto x = g.add_value(t, "x");
  auto b = g.add_value(t, "b");
  g.inputs = {x, b};

  // ops: add -> relu
  auto add0 = g.add_op(OpKind::Add, {x, b}, 1, "add0");
  auto y = g.op(add0).outputs[0];
  auto relu0 = g.add_op(OpKind::Relu, {y}, 1, "relu0");
  auto z = g.op(relu0).outputs[0];
  g.set_graph_outputs({z});

  auto st = verify_module(m);
  LUNARA_CHECK(st.ok());

  auto dump = dump_module(m);
  LUNARA_CHECK(!dump.empty());
  std::puts(dump.c_str());

  std::puts("test_ir_build OK");
  return 0;
}

