#pragma once
#include "lunara/passes/pass.h"
#include <memory>
#include <vector>

namespace lunara::passes {

struct PassManager {
  std::vector<std::unique_ptr<Pass>> passes;

  void add(std::unique_ptr<Pass> p) { passes.push_back(std::move(p)); }
  lunara::Status run(lunara::ir::Module& m);
};

}

