#pragma once
#include "lunara/ir/module.h"
#include "lunara/util/status.h"
#include "lunara/codegen/cuda/kernel_ir.h"
#include <vector>

namespace lunara::passes {

struct FusionRegion {
  std::vector<lunara::ir::OpId> ops;
  std::vector<lunara::ir::ValueId> external_inputs;
  lunara::ir::ValueId output;
};

struct FusionPlan {
  FusionRegion region;
  lunara::codegen::cuda::KernelIR kir;
  std::string signature;
};

lunara::Result<std::vector<FusionPlan>> build_fusion_plans(const lunara::ir::Module& m);

} // namespace lunara::passes