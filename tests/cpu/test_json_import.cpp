#include "lunara/frontend/json/graph_json_io.h"
#include "lunara/ir/printer.h"
#include "lunara/util/assert.h"
#include <cstdio>
#include <fstream>

static void write_file(const char* path, const char* txt) {
  std::ofstream f(path);
  f << txt;
}

int main() {
  const char* path = "tmp_lunara_graph.json";
  const char* txt = R"JSON(
{
  "inputs": [
    {"name":"x","dtype":"f32","shape":[4,4]},
    {"name":"b","dtype":"f32","shape":[4,4]}
  ],
  "ops": [
    {"kind":"Add","inputs":["x","b"],"name":"add0"},
    {"kind":"Relu","inputs":["add0:0"],"name":"relu0"}
  ],
  "outputs": ["relu0:0"]
}
)JSON";
  write_file(path, txt);

  lunara::ir::Module m;
  auto st = lunara::frontend::json::import_graph_json(path, m);
  LUNARA_CHECK(st.ok());

  auto dump = lunara::ir::dump_module(m);
  LUNARA_CHECK(dump.find("Add") != std::string::npos);
  LUNARA_CHECK(dump.find("Relu") != std::string::npos);

  std::puts(dump.c_str());
  std::puts("test_json_import OK");
  return 0;
}

