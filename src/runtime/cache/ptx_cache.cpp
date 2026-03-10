#include "lunara/runtime/cache/ptx_cache.h"
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace lunara::rt::cache {

static std::string getenv_str(const char* k) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string();
}

std::string cache_dir() {
  auto d = getenv_str("LUNARA_CACHE_DIR");
  if (!d.empty()) {
    return d;
  }
  auto home = getenv_str("HOME");
  if (home.empty()) {
    return ".lunara_cache";
  }
  return home + "/.cache/lunara";
}

std::string fnv1a_64_hex(const std::string& s) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= (std::uint64_t)c;
    h *= 1099511628211ull;
  }
  static const char* hex = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; i--) {
    out[i] = hex[h & 0xF]; h >>= 4;
}
  return out;
}

static std::string ptx_path(const std::string& key_hex) {
  return cache_dir() + "/ptx_" + key_hex + ".ptx";
}

lunara::Result<std::string> load_ptx(const std::string& key_hex) {
  std::ifstream f(ptx_path(key_hex));
  if (!f.good()) {
    return lunara::Result<std::string>::Error("cache: miss");
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return lunara::Result<std::string>::Ok(ss.str());
}

lunara::Status store_ptx(const std::string& key_hex, const std::string& ptx) {
  // mkdir -p behavior without deps: rely on system("mkdir -p")
  std::string cmd = "mkdir -p " + cache_dir();
  std::system(cmd.c_str());

  std::ofstream f(ptx_path(key_hex));
  if (!f.good()) {
    return lunara::Status::Error("cache: cannot write ptx");
  }
  f << ptx;
  return lunara::Status::Ok();
}

} // namespace lunara::rt::cache