#pragma once
#include <cstdint>

namespace lunara::util {

struct HostTimer {
  void start();
  void stop();
  double ms() const;

private:
  std::int64_t t0_{0};
  std::int64_t t1_{0};
};

} // namespace lunara::util

