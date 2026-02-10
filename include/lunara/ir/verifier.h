#pragma once
#include "lunara/ir/module.h"
#include "lunara/util/status.h" // reuse your Stage1 Status, or rename later

namespace lunara::ir {

lunara::Status verify_module(const Module& m);

} // namespace lunara::ir

