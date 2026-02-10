#pragma once
#include <cstdint>
#include <string_view>

namespace lunara::ir {

enum class DType : std::uint8_t {
  f16, f32, i32, i64, unknown
};

inline std::string_view to_string(DType dt) {
  switch (dt) {
    case DType::f16:
      return "f16";
    case DType::f32:
      return "f32";
    case DType::i32:
      return "i32";
    case DType::i64:
      return "i64";
    default:
      return "unknown";
  }
}

} // namespace lunara::ir

