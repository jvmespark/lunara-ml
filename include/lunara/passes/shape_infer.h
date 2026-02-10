#pragma once
#include "lunara/passes/pass.h"

namespace lunara::passes {

struct ShapeInferPass final : Pass {
  const char* name() const override { return "ShapeInfer"; }
  lunara::Status run(lunara::ir::Module& m) override;
};

}

