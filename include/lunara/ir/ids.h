#pragma once
#include <cstdint>

namespace lunara::ir {

struct ValueId {
  std::uint32_t v{0};
};
struct OpId {
  std::uint32_t v{0};
};

inline constexpr std::uint32_t kInvalidId = 0xFFFFFFFFu;

inline bool is_valid(ValueId id) {
  return id.v != kInvalidId;
}
inline bool is_valid(OpId id) {
  return id.v != kInvalidId;
}

} // namespace lunara::ir


