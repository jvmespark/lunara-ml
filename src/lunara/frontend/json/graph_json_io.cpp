#include "lunara/frontend/json/graph_json_io.h"
#include "lunara/frontend/json/graph_json.h"
#include "lunara/ir/dtype.h"
#include "lunara/ir/verifier.h"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cctype>

namespace lunara::frontend::json {

static lunara::Status err(const char* m) {
  return lunara::Status::Error(m);
}

static std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Extremely small “parser” helpers for our restricted test JSON.
// This is intentionally minimal. If you later add a real JSON lib, replace this file only.

static void skip_ws(const std::string& s, std::size_t& i) {
  while (i < s.size() && std::isspace((unsigned char)s[i])) {
    i++;
  }
}

static bool consume(const std::string& s, std::size_t& i, char c) {
  skip_ws(s, i);
  if (i < s.size() && s[i] == c) {
    i++; return true;
  }
  return false;
}

static lunara::Status expect(const std::string& s, std::size_t& i, char c) {
  if (!consume(s, i, c)) {
    return err("json: expected character");
  }
  return lunara::Status::Ok();
}

static lunara::Result<std::string> parse_string(const std::string& s, std::size_t& i) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '"') {
    return lunara::Result<std::string>::Error("json: expected string");
  }
  i++;
  std::string out;
  while (i < s.size() && s[i] != '"') {
    out.push_back(s[i++]);
  }
  if (i >= s.size() || s[i] != '"') {
    return lunara::Result<std::string>::Error("json: unterminated string");
  }
  i++;
  return lunara::Result<std::string>::Ok(std::move(out));
}

static lunara::Result<std::int64_t> parse_int(const std::string& s, std::size_t& i) {
  skip_ws(s, i);
  bool neg = false;
  if (i < s.size() && s[i] == '-') {
    neg = true; i++;
  }
  if (i >= s.size() || !std::isdigit((unsigned char)s[i])) {
    return lunara::Result<std::int64_t>::Error("json: expected int");
  }
  std::int64_t v = 0;
  while (i < s.size() && std::isdigit((unsigned char)s[i])) {
    v = v * 10 + (s[i] - '0');
    i++;
  }
  return lunara::Result<std::int64_t>::Ok(neg ? -v : v);
}

static lunara::Result<std::vector<std::int64_t>> parse_int_array(const std::string& s, std::size_t& i) {
  if (!consume(s, i, '[')) {
    return lunara::Result<std::vector<std::int64_t>>::Error("json: expected [");
  }
  std::vector<std::int64_t> out;
  skip_ws(s, i);
  if (consume(s, i, ']')) {
    return lunara::Result<std::vector<std::int64_t>>::Ok(out);
  }
  while (true) {
    auto iv = parse_int(s, i);
    if (!iv.ok()) {
      return lunara::Result<std::vector<std::int64_t>>::Error(iv.status().message());
    }
    out.push_back(iv.value());
    skip_ws(s, i);
    if (consume(s, i, ']')) {
      break;
    }
    if (!consume(s, i, ',')) {
      return lunara::Result<std::vector<std::int64_t>>::Error("json: expected ,");
    }
  }
  return lunara::Result<std::vector<std::int64_t>>::Ok(std::move(out));
}

static lunara::Result<std::vector<std::string>> parse_string_array(const std::string& s, std::size_t& i) {
  if (!consume(s, i, '[')) {
    return lunara::Result<std::vector<std::string>>::Error("json: expected [");
  }
  std::vector<std::string> out;
  skip_ws(s, i);
  if (consume(s, i, ']')) {
    return lunara::Result<std::vector<std::string>>::Ok(out);
  }
  while (true) {
    auto sv = parse_string(s, i);
    if (!sv.ok()) {
      return lunara::Result<std::vector<std::string>>::Error(sv.status().message());
    }
    out.push_back(sv.value());
    skip_ws(s, i);
    if (consume(s, i, ']')) {
      break;
    }
    if (!consume(s, i, ',')) {
      return lunara::Result<std::vector<std::string>>::Error("json: expected ,");
    }
  }
  return lunara::Result<std::vector<std::string>>::Ok(std::move(out));
}

// Parse a { "k":"v", ... } object into a string map (values are either strings or arrays; we handle both paths manually later)
static lunara::Status parse_key(const std::string& s, std::size_t& i, std::string& out_key) {
  auto k = parse_string(s, i);
  if (!k.ok()) {
    return lunara::Status::Error(k.status().message());
  }
  out_key = k.value();
  LUNARA_RETURN_IF_ERROR(expect(s, i, ':'));
  return lunara::Status::Ok();
}

static lunara::ir::DType parse_dtype(const std::string& s) {
  if (s == "f16") {
    return lunara::ir::DType::f16;
  }
  if (s == "f32") {
    return lunara::ir::DType::f32;
  }
  if (s == "i32") {
    return lunara::ir::DType::i32;
  }
  if (s == "i64") {
    return lunara::ir::DType::i64;
  }
  return lunara::ir::DType::unknown;
}

static lunara::ir::OpKind parse_opkind(const std::string& s) {
  using lunara::ir::OpKind;
  if (s == "Add") {
    return OpKind::Add;
  }
  if (s == "Mul") {
    return OpKind::Mul;
  }
  if (s == "Relu") {
    return OpKind::Relu;
  }
  if (s == "MatMul") {
    return OpKind::MatMul;
  }
  return OpKind::Input;
}

static lunara::Result<lunara::frontend::json::JsonGraph> parse_graph(const std::string& txt) {
  std::size_t i = 0;
  skip_ws(txt, i);
  if (!consume(txt, i, '{')) {
    return lunara::Result<JsonGraph>::Error("json: expected {");
  }

  JsonGraph g;

  while (true) {
    skip_ws(txt, i);
    if (consume(txt, i, '}')) {
      break;
    }

    std::string key;
    auto st = parse_key(txt, i, key);
    if (!st.ok()) {
      return lunara::Result<JsonGraph>::Error(st.message());
    }

    if (key == "inputs") {
      LUNARA_RETURN_IF_ERROR_R(expect(txt, i, '['), lunara::frontend::json::JsonGraph);
      skip_ws(txt, i);
      if (!consume(txt, i, ']')) {
        while (true) {
          LUNARA_RETURN_IF_ERROR_R(expect(txt, i, '{'), lunara::frontend::json::JsonGraph);
          JsonInput in;
          while (true) {
            skip_ws(txt, i);
            if (consume(txt, i, '}')) {
              break;
            }
            std::string k2;
            LUNARA_RETURN_IF_ERROR_R(parse_key(txt, i, k2), lunara::frontend::json::JsonGraph);
            if (k2 == "name") {
              auto sv = parse_string(txt, i);
              if (!sv.ok()) {
                return lunara::Result<JsonGraph>::Error(sv.status().message());
              }
              in.name = sv.value();
            } else if (k2 == "dtype") {
              auto sv = parse_string(txt, i);
              if (!sv.ok()) {
                return lunara::Result<JsonGraph>::Error(sv.status().message());
              }
              in.dtype = sv.value();
            } else if (k2 == "shape") {
              auto av = parse_int_array(txt, i);
              if (!av.ok()) {
                return lunara::Result<JsonGraph>::Error(av.status().message());
              }
              in.shape = av.value();
            } else {
              return lunara::Result<JsonGraph>::Error("json: unknown input field");
            }
            skip_ws(txt, i);
            consume(txt, i, ',');
          }
          g.inputs.push_back(std::move(in));
          skip_ws(txt, i);
          if (consume(txt, i, ']')) {
            break;
          }
          LUNARA_RETURN_IF_ERROR_R(expect(txt, i, ','), lunara::frontend::json::JsonGraph);
        }
      }
    } else if (key == "ops") {
      LUNARA_RETURN_IF_ERROR_R(expect(txt, i, '['), lunara::frontend::json::JsonGraph);
      skip_ws(txt, i);
      if (!consume(txt, i, ']')) {
        while (true) {
          LUNARA_RETURN_IF_ERROR_R(expect(txt, i, '{'), lunara::frontend::json::JsonGraph);
          JsonOp op;
          while (true) {
            skip_ws(txt, i);
            if (consume(txt, i, '}')) {
              break;
            }
            std::string k2;
            LUNARA_RETURN_IF_ERROR_R(parse_key(txt, i, k2), lunara::frontend::json::JsonGraph);
            if (k2 == "kind") {
              auto sv = parse_string(txt, i); 
              if (!sv.ok()) {
                return lunara::Result<JsonGraph>::Error(sv.status().message());
              }
              op.kind = sv.value();
            } else if (k2 == "inputs") {
              auto arr = parse_string_array(txt, i);
              if (!arr.ok()) {
                return lunara::Result<JsonGraph>::Error(arr.status().message());
              }
              op.inputs = arr.value();
            } else if (k2 == "name") {
              auto sv = parse_string(txt, i);
              if (!sv.ok()) {
                return lunara::Result<JsonGraph>::Error(sv.status().message());
              }
              op.name = sv.value();
            } else {
              return lunara::Result<JsonGraph>::Error("json: unknown op field");
            }
            skip_ws(txt, i);
            consume(txt, i, ',');
          }
          g.ops.push_back(std::move(op));
          skip_ws(txt, i);
          if (consume(txt, i, ']')) {
            break;
          }
          LUNARA_RETURN_IF_ERROR_R(expect(txt, i, ','), lunara::frontend::json::JsonGraph);
        }
      }
    } else if (key == "outputs") {
      auto arr = parse_string_array(txt, i);
      if (!arr.ok()) {
        return lunara::Result<JsonGraph>::Error(arr.status().message());
      }
      g.outputs = arr.value();
    } else {
      return lunara::Result<JsonGraph>::Error("json: unknown top-level key");
    }

    skip_ws(txt, i);
    consume(txt, i, ',');
  }

  return lunara::Result<JsonGraph>::Ok(std::move(g));
}

// Resolve ref strings:
//  - input name: "x"
//  - op output:  "opName:0"
static lunara::Result<lunara::ir::ValueId> resolve_ref(
  const std::string& ref,
  const std::unordered_map<std::string, lunara::ir::ValueId>& sym)
{
  auto it = sym.find(ref);
  if (it != sym.end()) {
    return lunara::Result<lunara::ir::ValueId>::Ok(it->second);
  }
  return lunara::Result<lunara::ir::ValueId>::Error("json: unresolved value ref");
}

lunara::Status import_graph_json(const std::string& path, lunara::ir::Module& out) {
  std::string txt = slurp(path);
  auto pg = parse_graph(txt);
  if (!pg.ok()) {
    return lunara::Status::Error(pg.status().message());
  }

  out = lunara::ir::Module{};
  auto& g = out.g;

  std::unordered_map<std::string, lunara::ir::ValueId> sym;

  // Create input values
  for (const auto& in : pg.value().inputs) {
    lunara::ir::TensorType tt;
    tt.dtype = parse_dtype(in.dtype);
    tt.shape.dims = in.shape;

    auto vid = g.add_value(tt, in.name);
    g.inputs.push_back(vid);
    sym[in.name] = vid;
  }

  // Create ops
  for (const auto& opj : pg.value().ops) {
    std::vector<lunara::ir::ValueId> ins;
    ins.reserve(opj.inputs.size());
    for (const auto& r : opj.inputs) {
      auto rv = resolve_ref(r, sym);
      if (!rv.ok()) {
        return lunara::Status::Error(rv.status().message());
      }
      ins.push_back(rv.value());
    }

    auto kind = parse_opkind(opj.kind);
    if (kind == lunara::ir::OpKind::Input) {
      return lunara::Status::Error("json: unknown op kind");
    }

    auto oid = g.add_op(kind, ins, /*out_count=*/1, opj.name);
    // Register output symbol as "opName:0"
    sym[opj.name + ":0"] = g.op(oid).outputs[0];
  }

  // Set outputs
  std::vector<lunara::ir::ValueId> outs;
  outs.reserve(pg.value().outputs.size());
  for (const auto& r : pg.value().outputs) {
    auto rv = resolve_ref(r, sym);
    if (!rv.ok()) {
      return lunara::Status::Error(rv.status().message());
    }
    outs.push_back(rv.value());
  }
  g.set_graph_outputs(outs);

  // Verify
  return lunara::ir::verify_module(out);
}

} // namespace lunara::frontend::json

