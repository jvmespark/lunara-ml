#pragma once
#include "lunara/ir/module.h"
#include "lunara/util/status.h"
#include <string>

namespace lunara::frontend::json {

lunara::Status import_graph_json(const std::string& path, lunara::ir::Module& out);

} // namespace lunara::frontend::json

