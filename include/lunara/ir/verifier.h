#pragma once
#include "lunara/ir/module.h"
#include "lunara/util/status.h"

namespace lunara::ir {

lunara::Status verify_module(const Module& m);

} // namespace lunara::ir

