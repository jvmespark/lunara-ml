#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lunara::frontend::json {

struct JsonInput {
  std::string name;
  std::string dtype;               // "f32" etc
  std::vector<std::int64_t> shape; // dims
};

struct JsonOp {
  std::string kind;                // "Add","Mul","Relu","MatMul"
  std::vector<std::string> inputs; // refs
  std::string name;                // op name
};

struct JsonGraph {
  std::vector<JsonInput> inputs;
  std::vector<JsonOp> ops;
  std::vector<std::string> outputs;
};

} // namespace lunara::frontend::json

