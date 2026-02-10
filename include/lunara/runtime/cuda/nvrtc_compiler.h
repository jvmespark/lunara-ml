#pragma once
#include "lunara/config.h"
#include "lunara/util/status.h"
#include <string>
#include <vector>

namespace lunara::rt {

struct NvrtcCompiler {
  lunara::Result<std::string> compile_to_ptx(
    const std::string& cuda_src,
    const std::string& kernel_name,
    const std::vector<std::string>& options = {}
  );
};

} // namespace lunara::rt

