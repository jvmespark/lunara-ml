#include "lunara/passes/shape_infer.h"
#include "lunara/ir/utils.h"
#include "lunara/ir/op.h"

namespace lunara::passes {

static lunara::Status require(bool cond, const char* msg) {
  return cond ? lunara::Status::Ok() : lunara::Status::Error(msg);
}

lunara::Status ShapeInferPass::run(lunara::ir::Module& m) {
  auto& g = m.g;

  for (auto& op : g.ops) {
    using lunara::ir::OpKind;
    if (op.outputs.empty()) return lunara::Status::Error("shape_infer: op has no outputs");

    if (op.kind == OpKind::Add || op.kind == OpKind::Mul) {
      if (op.inputs.size() != 2) return lunara::Status::Error("shape_infer: add/mul expects 2 inputs");
      auto& a = g.value(op.inputs[0]);
      auto& b = g.value(op.inputs[1]);
      auto& out = g.value(op.outputs[0]);

      LUNARA_RETURN_IF_ERROR(require(a.type.dtype == b.type.dtype, "shape_infer: add/mul dtype mismatch"));
      LUNARA_RETURN_IF_ERROR(require(lunara::ir::same_shape(a.type, b.type), "shape_infer: add/mul shape mismatch"));

      out.type = a.type; // preserve
    }

    else if (op.kind == OpKind::Relu) {
      if (op.inputs.size() != 1) return lunara::Status::Error("shape_infer: relu expects 1 input");
      auto& a = g.value(op.inputs[0]);
      auto& out = g.value(op.outputs[0]);
      out.type = a.type;
    }

    else if (op.kind == OpKind::MatMul) {
      if (op.inputs.size() != 2) return lunara::Status::Error("shape_infer: matmul expects 2 inputs");
      auto& A = g.value(op.inputs[0]);
      auto& B = g.value(op.inputs[1]);
      auto& C = g.value(op.outputs[0]);

      LUNARA_RETURN_IF_ERROR(require(A.type.dtype == B.type.dtype, "shape_infer: matmul dtype mismatch"));
      LUNARA_RETURN_IF_ERROR(require(lunara::ir::rank(A.type) == 2 && lunara::ir::rank(B.type) == 2,
                                     "shape_infer: matmul requires rank-2"));

      auto M = A.type.shape.dims[0];
      auto K = A.type.shape.dims[1];
      auto K2 = B.type.shape.dims[0];
      auto N = B.type.shape.dims[1];

      // For v1: require known K and K2, and they match.
      LUNARA_RETURN_IF_ERROR(require(K >= 0 && K2 >= 0, "shape_infer: matmul requires known K"));
      LUNARA_RETURN_IF_ERROR(require(K == K2, "shape_infer: matmul K mismatch"));

      lunara::ir::TensorType tt;
      tt.dtype = A.type.dtype;
      tt.shape.dims = {M, N}; // M or N can be -1 if unknown
      C.type = tt;
    }

    // ignore other ops in v1
  }

  return lunara::Status::Ok();
}

}
