#include "lunara/runtime/jit/fusion_jit.h"
#include "lunara/codegen/cuda/cuda_emitter.h"
#include "lunara/runtime/cuda/nvrtc_compiler.h"
#include "lunara/runtime/cache/ptx_cache.h"

namespace lunara::rt::jit {

lunara::Result<std::string> compile_fusion_to_ptx(const lunara::passes::FusionPlan& plan) {
  // Cache key is hash(signature + kernel source format version)
  const std::string key = lunara::rt::cache::fnv1a_64_hex("v1|" + plan.signature);

  auto hit = lunara::rt::cache::load_ptx(key);
  if (hit.ok()) {
    return lunara::Result<std::string>::Ok(hit.value());
  }

  // Emit CUDA source
  lunara::codegen::cuda::CudaEmitOptions opt;
  opt.kernel_name = "lunara_fused";
  std::string src = lunara::codegen::cuda::emit_cuda_source(plan.kir, opt);

  // NVRTC compile
  lunara::rt::NvrtcCompiler nv;
  auto ptx = nv.compile_to_ptx(src, "lunara_fused", { "--std=c++14", "--use_fast_math" });
  if (!ptx.ok()) {
    return lunara::Result<std::string>::Error(ptx.status().message());
  }

  // Store
  auto st = lunara::rt::cache::store_ptx(key, ptx.value());
  if (!st.ok()) {
    return lunara::Result<std::string>::Error(st.message());
  }

  return lunara::Result<std::string>::Ok(ptx.value());
}

} // namespace lunara::rt::jit