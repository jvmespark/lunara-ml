#pragma once
#include <cstdio>
#include <cstdlib>

#define LUNARA_CHECK(cond) do { \
  if (!(cond)) { \
    std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
    std::abort(); \
  } \
} while (0)

