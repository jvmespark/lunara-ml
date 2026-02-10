#pragma once
#include <cstdint>

namespace lunara::rt {
enum class Device : std::uint8_t {
  Host, Cuda
};
} // namespace lunara::rt

