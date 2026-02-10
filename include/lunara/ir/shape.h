#pragma once
#include <cstdint>
#include <vector>

namespace lunara::ir {

struct Shape {
  // dims may contain -1 for unknown
  std::vector<std::int64_t> dims;

  bool is_static() const {
    for (auto d : dims) {
      if (d < 0) {
        return false;
      }
    }
    return true;
  }

  std::int64_t numel_static() const {
    if (!is_static()) {
      return -1;
    }
    std::int64_t n = 1;
    for (auto d : dims) {
      n *= d;
    }
    return n;
  }
};

} // namespace lunara::ir

