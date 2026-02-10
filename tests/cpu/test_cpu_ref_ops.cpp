#include "lunara/runtime/tensor.h"
#include "lunara/runtime/cpu_ref.h"
#include "lunara/util/assert.h"
#include <cstdio>

int main() {
  using namespace lunara::rt;

  Tensor a = Tensor::empty_host({16}, DType::f32);
  Tensor b = Tensor::empty_host({16}, DType::f32);
  Tensor o = Tensor::empty_host({16}, DType::f32);

  float* ap = (float*)a.data;
  float* bp = (float*)b.data;
  for (int i = 0; i < 16; i++) {
    ap[i] = (float)i - 8.0f;
    bp[i] = 2.0f;
  }

  auto st = cpu::add(a, b, o);
  LUNARA_CHECK(st.ok());
  float* op = (float*)o.data;
  LUNARA_CHECK(op[0] == (-8.0f + 2.0f));
  LUNARA_CHECK(op[15] == (7.0f + 2.0f));

  st = cpu::relu(o, o);
  LUNARA_CHECK(st.ok());
  LUNARA_CHECK(op[0] == 0.0f);
  LUNARA_CHECK(op[15] == 9.0f);

  std::puts("test_cpu_ref_ops OK");
  return 0;
}

