#include "lunara/util/timer.h"
#include <chrono>

namespace lunara::util {

static std::int64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

void HostTimer::start() {
  t0_ = now_ns();
}
void HostTimer::stop()  {
  t1_ = now_ns();
}
double HostTimer::ms() const {
  return double(t1_ - t0_) / 1e6;
}

} // namespace lunara::util

