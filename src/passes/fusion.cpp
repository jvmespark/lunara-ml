#include "lunara/passes/fusion.h"
#include "lunara/ir/op.h"
#include "lunara/ir/dtype.h"
#include "lunara/ir/utils.h"
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace lunara::passes {

static bool is_elemwise(lunara::ir::OpKind k) {
  return k == lunara::ir::OpKind::Add || k == lunara::ir::OpKind::Mul || k == lunara::ir::OpKind::Relu;
}

static lunara::Status require(bool c, const char* m) {
  return c ? lunara::Status::Ok() : lunara::Status::Error(m);
}

// Build stable ref string for a value inside region:
// - if produced by an op in region: "v%id"
// - else: "arg<k>"
static std::string make_sig_line(const lunara::ir::Op& op) {
  std::ostringstream os;
  os << (int)op.kind << "(";
  for (std::size_t i = 0; i < op.inputs.size(); i++) {
    os << op.inputs[i].v;
    if (i + 1 < op.inputs.size()) {
      os << ",";
    }
  }
  os << ")->";
  for (std::size_t i = 0; i < op.outputs.size(); i++) {
    os << op.outputs[i].v;
    if (i + 1 < op.outputs.size()) {
      os << ",";
    }
  }
  return os.str();
}

static lunara::Result<lunara::codegen::cuda::KernelIR> build_kernel_ir(const lunara::ir::Module& m, const FusionRegion& r) {
  const auto& g = m.g;

  // Map external input ValueId -> argument index
  std::unordered_map<std::uint32_t, int> arg_index;
  for (int i = 0; i < (int)r.external_inputs.size(); i++) {
    arg_index[r.external_inputs[i].v] = i;
  }

  // Map ValueId -> Expr*
  std::unordered_map<std::uint32_t, std::unique_ptr<lunara::codegen::cuda::Expr>> exprs;

  // Helper: get expr for a value
  auto get_expr = [&](lunara::ir::ValueId v) -> lunara::Result<lunara::codegen::cuda::Expr*> {
    auto it = exprs.find(v.v);
    if (it != exprs.end()) {
      return lunara::Result<lunara::codegen::cuda::Expr*>::Ok(it->second.get());
    }

    auto ai = arg_index.find(v.v);
    if (ai != arg_index.end()) {
      exprs[v.v] = lunara::codegen::cuda::Expr::input(ai->second);
      return lunara::Result<lunara::codegen::cuda::Expr*>::Ok(exprs[v.v].get());
    }

    return lunara::Result<lunara::codegen::cuda::Expr*>::Error("fusion: unresolved value expr");
  };

  // Build expressions in op order
  for (auto oid : r.ops) {
    const auto& op = g.ops[oid.v];

    if (op.kind == lunara::ir::OpKind::Relu) {
      auto a = get_expr(op.inputs[0]);
      if (!a.ok()) {
        return lunara::Result<lunara::codegen::cuda::KernelIR>::Error(a.status().message());
      }
      exprs[op.outputs[0].v] = lunara::codegen::cuda::Expr::relu(
        std::unique_ptr<lunara::codegen::cuda::Expr>(new lunara::codegen::cuda::Expr(*a.value()))
      );
      // The above copy is clunky; do proper cloning:
      // For v1, we’ll instead rebuild using a simple clone function.
      return lunara::Result<lunara::codegen::cuda::KernelIR>::Error(
        "fusion: internal error (need expr clone). See below patch."
      );
    }
  }

  return lunara::Result<lunara::codegen::cuda::KernelIR>::Error("unreachable");
}

//
// IMPORTANT: The above shows the idea, but we need an Expr clone to avoid copying raw pointers.
// We implement clone properly below (used by all ops).
//

static std::unique_ptr<lunara::codegen::cuda::Expr> clone_expr(const lunara::codegen::cuda::Expr* e) {
  using lunara::codegen::cuda::Expr;
  using lunara::codegen::cuda::ExprKind;

  if (!e) {
    return std::unique_ptr<Expr>();
  }
  std::unique_ptr<Expr> out(new Expr());
  out->kind = e->kind;
  out->input_index = e->input_index;
  out->a = clone_expr(e->a.get());
  out->b = clone_expr(e->b.get());
  return out;
}

static lunara::Result<lunara::codegen::cuda::KernelIR> build_kernel_ir_fixed(const lunara::ir::Module& m, const FusionRegion& r) {
  const auto& g = m.g;

  std::unordered_map<std::uint32_t, int> arg_index;
  for (int i = 0; i < (int)r.external_inputs.size(); i++) {
    arg_index[r.external_inputs[i].v] = i;
  } 
  std::unordered_map<std::uint32_t, std::unique_ptr<lunara::codegen::cuda::Expr>> exprs;

  auto get_expr = [&](lunara::ir::ValueId v) -> lunara::Result<lunara::codegen::cuda::Expr*> {
    auto it = exprs.find(v.v);
    if (it != exprs.end()) {
      return lunara::Result<lunara::codegen::cuda::Expr*>::Ok(it->second.get());
    }
    auto ai = arg_index.find(v.v);
    if (ai != arg_index.end()) {
      exprs[v.v] = lunara::codegen::cuda::Expr::input(ai->second);
      return lunara::Result<lunara::codegen::cuda::Expr*>::Ok(exprs[v.v].get());
    }
    return lunara::Result<lunara::codegen::cuda::Expr*>::Error("fusion: unresolved value expr");
  };

  for (auto oid : r.ops) {
    const auto& op = g.ops[oid.v];

    if (op.kind == lunara::ir::OpKind::Add || op.kind == lunara::ir::OpKind::Mul) {
      auto a = get_expr(op.inputs[0]);
      if (!a.ok()) {
        return lunara::Result<lunara::codegen::cuda::KernelIR>::Error(a.status().message());
      }
      auto b = get_expr(op.inputs[1]);
      if (!b.ok()) {
        return lunara::Result<lunara::codegen::cuda::KernelIR>::Error(b.status().message());
      }

      if (op.kind == lunara::ir::OpKind::Add) {
        exprs[op.outputs[0].v] = lunara::codegen::cuda::Expr::add(clone_expr(a.value()), clone_expr(b.value()));
      } else {
        exprs[op.outputs[0].v] = lunara::codegen::cuda::Expr::mul(clone_expr(a.value()), clone_expr(b.value()));
      }
    }
    else if (op.kind == lunara::ir::OpKind::Relu) {
      auto a = get_expr(op.inputs[0]);
      if (!a.ok()) {
        return lunara::Result<lunara::codegen::cuda::KernelIR>::Error(a.status().message());
      }
      exprs[op.outputs[0].v] = lunara::codegen::cuda::Expr::relu(clone_expr(a.value()));
    }
    else {
      return lunara::Result<lunara::codegen::cuda::KernelIR>::Error("fusion: non-elementwise op in region");
    }
  }

  auto it = exprs.find(r.output.v);
  if (it == exprs.end()) {
    return lunara::Result<lunara::codegen::cuda::KernelIR>::Error("fusion: missing output expr");
  }

  lunara::codegen::cuda::KernelIR kir;
  kir.num_inputs = (int)r.external_inputs.size();
  kir.out_expr = clone_expr(it->second.get());
  return lunara::Result<lunara::codegen::cuda::KernelIR>::Ok(std::move(kir));
}

lunara::Result<std::vector<FusionPlan>> build_fusion_plans(const lunara::ir::Module& m) {
  const auto& g = m.g;
  std::vector<FusionPlan> plans;

  // v1 constraints
  // - f32 only
  // - all tensors same shape
  // - single-output chain
  // - ops list already topological

  std::unordered_set<std::uint32_t> claimed_ops;

  for (std::uint32_t i = 0; i < g.ops.size(); i++) {
    lunara::ir::OpId start{i};
    if (claimed_ops.count(i)) {
      continue;
    }
    const auto& op0 = g.ops[i];
    if (!is_elemwise(op0.kind)) {
      continue;
    }
    if (op0.outputs.size() != 1) {
      continue;
    }

    // Determine region output shape/dtype anchor from first input
    auto out0 = op0.outputs[0];
    const auto& outv0 = g.values[out0.v];
    if (outv0.type.dtype != lunara::ir::DType::f32) {
      continue;
    };
    if (!outv0.type.shape.is_static()) {
      continue;
    }

    FusionRegion r;
    r.ops.push_back(start);

    // grow forward while next ops are elementwise and consume the current produced value
    lunara::ir::ValueId cur = out0;
    std::uint32_t j = i + 1;
    while (j < g.ops.size()) {
      if (claimed_ops.count(j)) {
        break;
      }
      const auto& op = g.ops[j];
      if (!is_elemwise(op.kind)) {
        break;
      }
      if (op.outputs.size() != 1) {
        break;
      }

      // must use cur as one of its inputs
      bool consumes_cur = false;
      for (auto in : op.inputs) if (in.v == cur.v) {
        consumes_cur = true;
        break;
      }
      if (!consumes_cur) {
        break;
      }

      // output must match same static shape/dtype
      const auto& outv = g.values[op.outputs[0].v];
      if (outv.type.dtype != lunara::ir::DType::f32) {
        break;
      }
      if (outv.type.shape.dims != outv0.type.shape.dims) {
        break;
      }

      // Only fuse if cur has no external users besides this op
      // (i.e., cur.users should be exactly {op.id} OR users all within region)
      if (g.values[cur.v].users.size() != 1) {
        break;
      }

      r.ops.push_back(lunara::ir::OpId{j});
      cur = op.outputs[0];
      j++;
    }

    // Require at least 2 ops to bother fusing
    if (r.ops.size() < 2) {
      continue;
    }
    r.output = cur;

    // Collect external inputs: any input to a fused op whose producer is not in region
    std::unordered_set<std::uint32_t> region_ops;
    for (auto oid : r.ops) {
      region_ops.insert(oid.v);
    }
    std::unordered_set<std::uint32_t> ext_vals;
    for (auto oid : r.ops) {
      const auto& op = g.ops[oid.v];
      for (auto in : op.inputs) {
        const auto& vin = g.values[in.v];
        bool produced_in_region = (vin.producer.v != lunara::ir::kInvalidId) && (region_ops.count(vin.producer.v) > 0);
        bool is_graph_input = (vin.producer.v == lunara::ir::kInvalidId);
        if (!produced_in_region) {
          // graph input or comes from non-fused producer
          if (is_graph_input || true) {
            ext_vals.insert(in.v);
          }
        }
      }
    }

    // Deterministic ordering by ValueId
    for (auto vid : ext_vals) {
      r.external_inputs.push_back(lunara::ir::ValueId{vid});
    }
    std::sort(r.external_inputs.begin(), r.external_inputs.end(),
              [](lunara::ir::ValueId a, lunara::ir::ValueId b){ return a.v < b.v; });

    // Build KernelIR
    auto kir = build_kernel_ir_fixed(m, r);
    if (!kir.ok()) {
      return lunara::Result<std::vector<FusionPlan>>::Error(kir.status().message());
    }

    // Build signature string
    std::ostringstream sig;
    sig << "fuse{";
    for (auto oid : r.ops) {
      sig << make_sig_line(g.ops[oid.v]) << ";";
    }
    sig << "} shape=";
    for (std::size_t k = 0; k < outv0.type.shape.dims.size(); k++) {
      sig << outv0.type.shape.dims[k] << (k+1<outv0.type.shape.dims.size() ? "," : "");
    }
    sig << " dtype=f32";

    FusionPlan fp;
    fp.region = r;
    fp.kir = std::move(kir.value());
    fp.signature = sig.str();

    // mark claimed
    for (auto oid : r.ops) {
      claimed_ops.insert(oid.v);
    }
    plans.push_back(std::move(fp));
  }

  return lunara::Result<std::vector<FusionPlan>>::Ok(std::move(plans));
}

} // namespace lunara::passes