#pragma once
#include "lunara/ir/module.h"
#include "lunara/runtime/tensor.h"
#include "lunara/util/status.h"
#include <unordered_map>
#include <string>

namespace lunara::rt {

struct RunResult {
  lunara::Status status;
  std::unordered_map<std::string, Tensor> outputs;
};

struct CpuInterpreter {
  // feeds: map input name -> Tensor (host f32)
  RunResult run(const lunara::ir::Module& m,
                const std::unordered_map<std::string, Tensor>& feeds);
};

}
