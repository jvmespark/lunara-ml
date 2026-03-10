#pragma once
#include "lunara/util/status.h"
#include <string>

namespace lunara::rt::cache {

std::string cache_dir();
std::string fnv1a_64_hex(const std::string& s);

lunara::Result<std::string> load_ptx(const std::string& key_hex);
lunara::Status store_ptx(const std::string& key_hex, const std::string& ptx);

} // namespace lunara::rt::cache