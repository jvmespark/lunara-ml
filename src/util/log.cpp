#include "lunara/util/log.h"
#include <cstdio>
#include <cstdarg>

namespace lunara::log {

static void vlog(const char* tag, const char* fmt, va_list ap) {
  std::fprintf(stderr, "[%s] ", tag);
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
}

void info(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); vlog("INFO", fmt, ap); va_end(ap);
}
void warn(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); vlog("WARN", fmt, ap); va_end(ap);
}
void error(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); vlog("ERROR", fmt, ap); va_end(ap);
}

} // namespace lunara::log

