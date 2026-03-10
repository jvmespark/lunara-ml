    #pragma once
#include "lunara/util/status.h"
#include "lunara/passes/fusion.h"
#include <string>

namespace lunara::rt::jit {

lunara::Result<std::string> compile_fusion_to_ptx(const lunara::passes::FusionPlan& plan);

} // namespace lunara::rt::jit