#pragma once
#include "lunara/passes/pass.h"

namespace lunara::passes {

struct ConstFoldPass final : Pass {
  const char* name() const override { return "ConstFold"; }
  lunara::Status run(lunara::ir::Module& m) override;
};

}
