#pragma once
#include <cstdarg>

namespace lunara::log {
  void info(const char* fmt, ...);
  void warn(const char* fmt, ...);
  void error(const char* fmt, ...);
}

#define LUNARA_LOG_INFO(...)  ::lunara::log::info(__VA_ARGS__)
#define LUNARA_LOG_WARN(...)  ::lunara::log::warn(__VA_ARGS__)
#define LUNARA_LOG_ERROR(...) ::lunara::log::error(__VA_ARGS__)

