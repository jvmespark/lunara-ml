#include "lunara/passes/pass_manager.h"

namespace lunara::passes {

lunara::Status PassManager::run(lunara::ir::Module& m) {
  for (auto& p : passes) {
    auto st = p->run(m);
    if (!st.ok()) {
      return lunara::Status::Error(std::string("Pass failed: ") + p->name() + " :: " + st.message());
    }
  }
  return lunara::Status::Ok();
}

}

