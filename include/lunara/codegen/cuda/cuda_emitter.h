#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lunara::codegen::cuda {

enum class ExprKind : std::uint8_t { InputRef, Add, Mul, Relu };

struct Expr {
  ExprKind kind;
  int input_index;
  std::unique_ptr<Expr> a;
  std::unique_ptr<Expr> b;

  static std::unique_ptr<Expr> input(int idx) {
    std::unique_ptr<Expr> e(new Expr());
    e->kind = ExprKind::InputRef;
    e->input_index = idx;
    return e;
  }
  static std::unique_ptr<Expr> add(std::unique_ptr<Expr> x, std::unique_ptr<Expr> y) {
    std::unique_ptr<Expr> e(new Expr());
    e->kind = ExprKind::Add;
    e->a = std::move(x);
    e->b = std::move(y);
    return e;
  }
  static std::unique_ptr<Expr> mul(std::unique_ptr<Expr> x, std::unique_ptr<Expr> y) {
    std::unique_ptr<Expr> e(new Expr());
    e->kind = ExprKind::Mul;
    e->a = std::move(x);
    e->b = std::move(y);
    return e;
  }
  static std::unique_ptr<Expr> relu(std::unique_ptr<Expr> x) {
    std::unique_ptr<Expr> e(new Expr());
    e->kind = ExprKind::Relu;
    e->a = std::move(x);
    return e;
  }
};

struct KernelIR {
  int num_inputs = 0;
  std::unique_ptr<Expr> out_expr;
};

} // namespace lunara::codegen::cuda