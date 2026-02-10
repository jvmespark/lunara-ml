#pragma once
#include <string>

namespace lunara {

struct Status {
  bool ok_{true};
  std::string msg;

  static Status Ok() {
    return Status{};
  }
  static Status Error(std::string m) {
    Status s; s.ok_ = false; s.msg = std::move(m); return s;
  }

  bool ok() const {
    return ok_;
  }
  const std::string& message() const {
    return msg;
  }
};

template <class T>
struct Result {
  Status st;
  T val;

  static Result<T> Ok(T v) {
    return Result<T>{Status::Ok(), std::move(v)};
  }
  static Result<T> Error(std::string m) {
    return Result<T>{Status::Error(std::move(m)), T{}};
  }

  bool ok() const {
    return st.ok();
  }
  const Status& status() const {
    return st;
  }
  T& value() {
    return val;
  }
  const T& value() const {
    return val;
  }
};

#define LUNARA_RETURN_IF_ERROR(expr)            \
  do {                                          \
    ::lunara::Status _st = (expr);              \
    if (!_st.ok()) {                            \
      return _st;                               \
    }                                           \
  } while (0)

#define LUNARA_RETURN_IF_ERROR_R(expr, T)               \
  do {                                                  \
    ::lunara::Status _st = (expr);                      \
    if (!_st.ok()) {                                    \
      return ::lunara::Result<T>::Error(_st.message()); \
    }                                                   \
  } while (0)

} // namespace lunara

