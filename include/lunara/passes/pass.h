#pragma once
#include "lunara/ir/module.h"
#include "lunara/util/status.h"
#include <string>

namespace lunara::passes {

struct Pass {
  virtual ~Pass() = default;
  virtual const char* name() const = 0;
  virtual lunara::Status run(lunara::ir::Module& m) = 0;
};

}
