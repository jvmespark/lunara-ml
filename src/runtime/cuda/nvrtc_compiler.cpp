#include "lunara/runtime/cuda/nvrtc_compiler.h"
#include "lunara/util/log.h"

#if LUNARA_HAS_CUDA
  #include <nvrtc.h>
#endif

namespace lunara::rt {

lunara::Result<std::string> NvrtcCompiler::compile_to_ptx(
  const std::string& cuda_src,
  const std::string& kernel_name,
  const std::vector<std::string>& options)
{
#if !LUNARA_HAS_CUDA
  (void)cuda_src;
  (void)kernel_name;
  (void)options;
  return lunara::Result<std::string>::Error("Built without CUDA/NVRTC support");
#else
  nvrtcProgram prog{};
  nvrtcResult r = nvrtcCreateProgram(
    &prog,
    cuda_src.c_str(),
    (kernel_name + ".cu").c_str(),
    0, nullptr, nullptr
  );
  if (r != NVRTC_SUCCESS) {
    return lunara::Result<std::string>::error(std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(r));
  }

  std::vector<const char*> opts;
  opts.reserve(options.size());
  for (auto& s : options) {
    opts.push_back(s.c_str());
  }

  r = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  // Always retrieve log (super helpful)
  size_t logSize = 0;
  nvrtcGetProgramLogSize(prog, &logSize);
  std::string log(logSize, '\0');
  if (logSize > 1) {
    nvrtcGetProgramLog(prog, log.data());
  }
  if (!log.empty()) {
    LUNARA_LOG_INFO("NVRTC log:\n%s", log.c_str());
  }

  if (r != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return lunara::Result<std::string>::error(std::string("nvrtcCompileProgram: ") + nvrtcGetErrorString(r));
  }

  size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, ptx.data());

  nvrtcDestroyProgram(&prog);
  return lunara::Result<std::string>::ok(std::move(ptx));
#endif
}

} // namespace lunara::rt

