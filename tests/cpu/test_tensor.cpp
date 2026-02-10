#include "lunara/runtime/tensor.h"
#include "lunara/util/assert.h"
#include <cstdio>

int main() {
  using namespace lunara::rt;
  Tensor t = Tensor::empty_host({4, 8}, DType::f32);
  LUNARA_CHECK(t.device == Device::Host);
  LUNARA_CHECK(t.bytes == (size_t)(4*8*sizeof(float)));
  LUNARA_CHECK(t.data != nullptr);

  // deterministic fill
  float* p = (float*)t.data;
  for (int i = 0; i < 32; i++) {
    p[i] = (float)i;
  }

  // verify
  for (int i = 0; i < 32; i++) {
    LUNARA_CHECK(p[i] == (float)i);
  }

  std::puts("test_tensor OK");
  return 0;
}

