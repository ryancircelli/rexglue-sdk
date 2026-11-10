// Known-answer tests for the refxenos shader interpreter (shader/interp.*).
// Programs are assembled by hand with the ucode bit layouts of
// include/rex/graphics/format/ucode.h.

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/common.h"
#include "shader/interp.h"

using namespace refxenos;
using namespace rex::graphics;  // XE_GPU_REG_* register indices

namespace {

int g_failures = 0;
int g_checks = 0;

uint32_t Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float FromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

void CheckF(const char* what, float got, float want) {
  ++g_checks;
  bool ok = std::isnan(want) ? std::isnan(got) : Bits(got) == Bits(want);
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %s: got %.9g (0x%08X) want %.9g (0x%08X)\n", what, double(got), Bits(got),
                double(want), Bits(want));
  }
}
void CheckB(const char* what, bool got, bool want) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    std::printf("FAIL %s: got %d want %d\n", what, int(got), int(want));
  }
}

// --- Assembler ---

struct Src {
  bool temp = true;
  uint32_t reg = 0;  // temp: idx | 0x40 rel | 0x80 abs
  const char* swz = "xyzw";
  bool neg = false;
  bool rel = false;  // constants: addressed
};
Src R(uint32_t r, const char* s = "xyzw", bool neg = false) { Src x; x.reg = r; x.swz = s; x.neg = neg; return x; }
Src C(uint32_t c, const char* s = "xyzw", bool neg = false) { Src x; x.temp = false; x.reg = c; x.swz = s; x.neg = neg; return x; }

uint32_t CompIndex(char c) { return c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : 3; }
uint32_t AluSwz(const char* s) {
  uint32_t v = 0;
  for (uint32_t j = 0; j < 4; ++j) v |= ((CompIndex(s[j]) - j) & 3) << (2 * j);
  return v;
}
uint32_t Mask(const char* m) {
  uint32_t v = 0;
  for (const char* p = m; *p; ++p) v |= 1u << CompIndex(*p);
  return v;
}

namespace V { enum : uint32_t { add = 0, mul = 1, max = 2, min = 3, mad = 11, dp4 = 15, kill_eq = 24, maxa = 29 }; }
namespace S {
enum : uint32_t {
  adds = 0, adds_prev = 1, muls = 2, maxs = 5, exp = 14, logc = 15, log = 16, rcpc = 17, rcpf = 18,
  rcp = 19, rsqc = 20, rsqf = 21, rsq = 22, maxas = 23, setp_eq = 27, setp_gt = 29, sqrt = 40,
  retain_prev = 50
};
}

struct Alu {
  uint32_t vop = V::max, sop = S::retain_prev;
  uint32_t vdst = 0, sdst = 0, vmask = 0, smask = 0;
  bool vdst_rel = false, sdst_rel = false, vsat = false, ssat = false, exp = false;
  Src s1, s2, s3;
  bool pred = false, pred_cond = false, a0_rel = false;
  std::array<uint32_t, 3> Encode() const {
    uint32_t d0 = vdst | (vdst_rel << 6) | (sdst << 8) | (sdst_rel << 14) | (exp << 15) |
                  (vmask << 16) | (smask << 20) | (vsat << 24) | (ssat << 25) | (sop << 26);
    uint32_t d1 = AluSwz(s3.swz) | (AluSwz(s2.swz) << 8) | (AluSwz(s1.swz) << 16) |
                  (s3.neg << 24) | (s2.neg << 25) | (s1.neg << 26) | (pred_cond << 27) |
                  (pred << 28) | (a0_rel << 29);
    // const_0_rel_abs addresses the first constant operand, const_1 the next.
    const Src* ops[3] = {&s1, &s2, &s3};
    int nconst = 0;
    for (int i = 0; i < 3; ++i) {
      if (!ops[i]->temp && ops[i]->rel) d1 |= 1u << (nconst ? 30 : 31);
      if (!ops[i]->temp) ++nconst;
    }
    uint32_t d2 = s3.reg | (s2.reg << 8) | (s1.reg << 16) | (vop << 24) | (s3.temp << 29) |
                  (s2.temp << 30) | (s1.temp << 31);
    return {d0, d1, d2};
  }
};
// Vector-only instruction, scalar retain_prev without writes.
Alu VecOp(uint32_t op, uint32_t dst, const char* mask, Src a, Src b, Src c = Src()) {
  Alu x; x.vop = op; x.vdst = dst; x.vmask = Mask(mask); x.s1 = a; x.s2 = b; x.s3 = c; return x;
}
// Scalar-only instruction (vector max r0 with no write).
Alu ScaOp(uint32_t op, uint32_t dst, const char* mask, Src a) {
  Alu x; x.sop = op; x.sdst = dst; x.smask = Mask(mask); x.s1 = R(0); x.s2 = R(0); x.s3 = a; return x;
}
Alu Mov(uint32_t dst, Src a, const char* mask = "xyzw") { return VecOp(V::max, dst, mask, a, a); }
Alu Export(uint32_t reg, Src a, const char* mask = "xyzw") {
  Alu x = Mov(reg, a, mask); x.exp = true; return x;
}

uint32_t FetchDstSwz(const char* s) {
  uint32_t v = 0;
  for (uint32_t c = 0; c < 4; ++c) {
    uint32_t k = s[c] == '0' ? 4 : s[c] == '1' ? 5 : s[c] == '_' ? 7 : CompIndex(s[c]);
    v |= k << (3 * c);
  }
  return v;
}

struct VFetch {
  uint32_t dst = 1, src = 0, src_comp = 0, fetch_const = 0, format = 38, stride = 4;
  int32_t offset = 0, exp_adjust = 0;
  bool is_signed = false, normalized = true, mini = false, rounded = false, rf_nozero = false;
  const char* dst_swz = "xyzw";
  std::array<uint32_t, 3> Encode() const {
    uint32_t d0 = 0 | (src << 5) | (dst << 12) | (1u << 19) | ((fetch_const / 3) << 20) |
                  ((fetch_const % 3) << 25) | (src_comp << 30);
    uint32_t d1 = FetchDstSwz(dst_swz) | (is_signed << 12) | ((normalized ? 0u : 1u) << 13) |
                  (rf_nozero << 14) | (rounded << 15) | (format << 16) |
                  ((uint32_t(exp_adjust) & 63) << 24) | (mini << 30);
    uint32_t d2 = stride | ((uint32_t(offset) & 0x7FFFFF) << 8);
    return {d0, d1, d2};
  }
};

struct TFetch {
  uint32_t op = 1, dst = 2, src = 0, src_swz = 0 | (1 << 2), fetch_const = 0, dim = 1;
  bool comp_lod = true, reg_lod = false, reg_grads = false;
  const char* dst_swz = "xyzw";
  std::array<uint32_t, 3> Encode() const {
    uint32_t d0 = op | (src << 5) | (dst << 12) | (1u << 19) | (fetch_const << 20) | (src_swz << 26);
    uint32_t d1 = FetchDstSwz(dst_swz) | (3u << 12) | (3u << 14) | (3u << 16) | (7u << 18) |
                  (3u << 24) | (3u << 26) | (comp_lod << 28) | (reg_lod << 29);
    uint32_t d2 = reg_grads | (dim << 14);
    return {d0, d1, d2};
  }
};

// CF instruction as (w0, w1:16).
struct Cf {
  uint32_t w0 = 0, w1 = 0;
  int exec_block = -1;  // patched with the block's instruction address
};

struct Program {
  std::vector<Cf> cf;
  std::vector<std::vector<std::pair<std::array<uint32_t, 3>, bool>>> blocks;

  int Block(std::vector<std::pair<std::array<uint32_t, 3>, bool>> instrs) {
    blocks.push_back(std::move(instrs));
    return int(blocks.size() - 1);
  }
  // op: 1 exec, 2 exece, 3 cexec, 4 cexece, 5 cexec_pred, 6 cexece_pred, 13/14 pred_clean
  void Exec(uint32_t op, int block, uint32_t bool_addr = 0, bool cond = false) {
    // ucode.h ControlFlowExecInstruction::count is only 3 bits. Split
    // synthetic blocks so an 8-instruction test cannot silently encode zero.
    constexpr size_t kMaxExecCount = 6;
    if (blocks[size_t(block)].size() > kMaxExecCount) {
      auto instructions = blocks[size_t(block)];
      for (size_t first = 0; first < instructions.size(); first += kMaxExecCount) {
        size_t end = std::min(first + kMaxExecCount, instructions.size());
        int part = Block({instructions.begin() + first, instructions.begin() + end});
        uint32_t part_op = op;
        if (end != instructions.size() && (op == 2 || op == 4 || op == 6 || op == 14))
          part_op = op - 1;
        Exec(part_op, part, bool_addr, cond);
      }
      return;
    }
    Cf c;
    c.exec_block = block;
    c.w1 = (bool_addr << 2) | (uint32_t(cond) << 10) | (op << 12);
    cf.push_back(c);
  }
  void LoopStart(uint32_t loop_id, uint32_t skip_to, bool repeat = false) {
    Cf c; c.w0 = skip_to | (uint32_t(repeat) << 13) | (loop_id << 16); c.w1 = 7u << 12; cf.push_back(c);
  }
  void LoopEnd(uint32_t loop_id, uint32_t body, bool pred_break = false, bool cond = false) {
    Cf c; c.w0 = body | (loop_id << 16) | (uint32_t(pred_break) << 21);
    c.w1 = (uint32_t(cond) << 10) | (8u << 12); cf.push_back(c);
  }
  void Call(uint32_t addr) { Cf c; c.w0 = addr | (1u << 13); c.w1 = 9u << 12; cf.push_back(c); }
  void Ret() { Cf c; c.w1 = 10u << 12; cf.push_back(c); }
  void Jmp(uint32_t addr, bool predicated, bool cond) {
    Cf c; c.w0 = addr | (uint32_t(!predicated) << 13) | (uint32_t(predicated) << 14);
    c.w1 = (uint32_t(cond) << 10) | (11u << 12); cf.push_back(c);
  }
  uint32_t Size() const { return uint32_t(cf.size()); }

  ShaderProgram Build(bool ps) const {
    ShaderProgram p;
    p.type = ps ? xenos::ShaderType::kPixel : xenos::ShaderType::kVertex;
    uint32_t base = uint32_t((cf.size() + 1) / 2);
    std::vector<uint32_t> addr(blocks.size());
    uint32_t a = base;
    for (size_t b = 0; b < blocks.size(); ++b) { addr[b] = a; a += uint32_t(blocks[b].size()); }
    std::vector<Cf> cfs = cf;
    if (cfs.size() & 1) cfs.push_back(Cf());
    for (Cf& c : cfs) {
      if (c.exec_block < 0) continue;
      const auto& blk = blocks[size_t(c.exec_block)];
      uint32_t seq = 0;
      for (size_t i = 0; i < blk.size(); ++i) if (blk[i].second) seq |= 1u << (2 * i);
      c.w0 = addr[size_t(c.exec_block)] | (uint32_t(blk.size()) << 12) | (seq << 16);
    }
    for (size_t i = 0; i < cfs.size(); i += 2) {
      const Cf& x = cfs[i];
      const Cf& y = cfs[i + 1];
      p.ucode.push_back(x.w0);
      p.ucode.push_back((x.w1 & 0xFFFF) | (y.w0 << 16));
      p.ucode.push_back((y.w0 >> 16) | ((y.w1 & 0xFFFF) << 16));
    }
    for (const auto& blk : blocks)
      for (const auto& in : blk)
        for (uint32_t d : in.first) p.ucode.push_back(d);
    return p;
  }
};

std::pair<std::array<uint32_t, 3>, bool> A(const Alu& a) { return {a.Encode(), false}; }
std::pair<std::array<uint32_t, 3>, bool> F(const std::array<uint32_t, 3>& f) { return {f, true}; }

struct Env {
  std::unique_ptr<RegisterFile> regs = std::make_unique<RegisterFile>();
  std::unique_ptr<Memory> mem;
  Env() {
    std::memset(regs->values, 0, sizeof(regs->values));
    regs->values[XE_GPU_REG_SQ_VS_CONST] = 255u << 12;
    regs->values[XE_GPU_REG_SQ_PS_CONST] = 256u | (255u << 12);
  }
  void SetConst(bool ps, uint32_t i, float x, float y, float z, float w) {
    float v[4] = {x, y, z, w};
    std::memcpy(&regs->values[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * (i + (ps ? 256 : 0))], v, 16);
  }
  void SetLoop(uint32_t i, uint32_t count, uint32_t start, int32_t step) {
    regs->values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00 + i] = count | (start << 8) | ((uint32_t(step) & 0xFF) << 16);
  }
  void SetBool(uint32_t i, bool v) {
    uint32_t& w = regs->values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 + (i >> 5)];
    w = v ? (w | (1u << (i & 31))) : (w & ~(1u << (i & 31)));
  }
  Memory& M() { if (!mem) mem = std::make_unique<Memory>(); return *mem; }
};

std::unique_ptr<RawExports> RunVs(Env& env, const Program& prog, uint32_t index = 0,
                                  TextureSampler* sampler = nullptr) {
  auto e = std::make_unique<RawExports>();
  DrawStats stats;
  ShaderProgram sp = prog.Build(false);
  RunVertexShaderRaw(sp, *env.regs, env.M(), index, sampler, e.get(), stats, nullptr);
  return e;
}

// --- Tests ---

void TestLegacyMulAndMad() {
  Env env;
  float denorm = FromBits(0x00000001);
  env.SetConst(false, 0, 0.0f, -0.0f, denorm, 2.0f);
  env.SetConst(false, 1, INFINITY, 5.0f, INFINITY, 3.0f);
  env.SetConst(false, 2, 1.0f, -1.0f, 0.5f, 1.0f);
  env.SetConst(false, 3, 1e-20f, -1e-20f, 1e-30f, 0.0f);
  env.SetConst(false, 4, 1e-20f, 1e-20f, 1e-30f, NAN);
  Program p;
  int b = p.Block({A(VecOp(V::mul, 1, "xyzw", C(0), C(1))),
                   A(VecOp(V::mad, 2, "xyzw", C(0), C(1), C(2))),
                   A(VecOp(V::mul, 3, "xyzw", C(3), C(4))),
                   A(VecOp(V::add, 4, "xyzw", C(0, "zzzz"), C(0, "yyyy"))),
                   A(Export(0, R(1))), A(Export(1, R(2))), A(Export(2, R(3))), A(Export(3, R(4)))});
  p.Exec(2, b);
  auto e = RunVs(env, p);
  CheckF("mul 0*inf", e->reg[0][0], 0.0f);
  CheckF("mul -0*5 = +0", e->reg[0][1], 0.0f);
  CheckF("mul denorm*inf = +0", e->reg[0][2], 0.0f);
  CheckF("mul 2*3", e->reg[0][3], 6.0f);
  CheckF("mad 0*inf+1", e->reg[1][0], 1.0f);
  CheckF("mad -0*5+-1", e->reg[1][1], -1.0f);
  CheckF("mad denorm*inf+0.5", e->reg[1][2], 0.5f);
  CheckF("mad 2*3+1", e->reg[1][3], 7.0f);
  CheckF("mul result denormal flushed", e->reg[2][0], 0.0f);
  CheckF("mul -denormal flushed to -0", e->reg[2][1], -0.0f);
  CheckF("mul tiny underflow", e->reg[2][2], 0.0f);
  CheckF("mul 0*NaN = +0", e->reg[2][3], 0.0f);
  CheckF("add denorm + -0 flushed", e->reg[3][0], 0.0f);
}

void TestMaxMinNaN() {
  Env env;
  env.SetConst(false, 0, NAN, 1.0f, -0.0f, 0.0f);
  env.SetConst(false, 1, 1.0f, NAN, 0.0f, -0.0f);
  Program p;
  int b = p.Block({A(VecOp(V::max, 1, "xyzw", C(0), C(1))), A(VecOp(V::min, 2, "xyzw", C(0), C(1))),
                   A(Export(0, R(1))), A(Export(1, R(2)))});
  p.Exec(2, b);
  auto e = RunVs(env, p);
  CheckF("max(NaN,1) = 1", e->reg[0][0], 1.0f);
  CheckF("max(1,NaN) = NaN", e->reg[0][1], NAN);
  CheckF("max(-0,+0) = -0 (a>=b)", e->reg[0][2], -0.0f);
  CheckF("max(+0,-0) = +0", e->reg[0][3], 0.0f);
  CheckF("min(NaN,1) = 1", e->reg[1][0], 1.0f);
  CheckF("min(1,NaN) = NaN", e->reg[1][1], NAN);
  CheckF("min(-0,+0) = +0 (a<b false)", e->reg[1][2], 0.0f);
}

void TestTranscendentals() {
  Env env;
  float denorm = FromBits(0x00000010);
  env.SetConst(false, 0, 0.0f, -0.0f, INFINITY, -1.0f);
  env.SetConst(false, 1, denorm, 3.0f, -INFINITY, 1.0f);
  struct Case { uint32_t op; uint32_t c; const char* comp; float want; const char* name; };
  std::vector<Case> cases = {
      {S::rcp, 0, "xxxx", INFINITY, "rcp(+0)"},       {S::rcp, 0, "yyyy", -INFINITY, "rcp(-0)"},
      {S::rcp, 0, "zzzz", 0.0f, "rcp(inf)"},          {S::rcp, 1, "xxxx", INFINITY, "rcp(denorm)"},
      {S::rcp, 1, "yyyy", float(1.0 / 3.0), "rcp(3)"}, {S::rcpc, 0, "xxxx", FLT_MAX, "rcpc(+0)"},
      {S::rcpc, 0, "yyyy", -FLT_MAX, "rcpc(-0)"},     {S::rcpf, 0, "xxxx", 0.0f, "rcpf(+0)"},
      {S::rcpf, 0, "yyyy", -0.0f, "rcpf(-0)"},        {S::rsq, 0, "xxxx", INFINITY, "rsq(+0)"},
      {S::rsq, 0, "wwww", NAN, "rsq(-1)"},            {S::rsq, 0, "zzzz", 0.0f, "rsq(inf)"},
      {S::rsqc, 0, "xxxx", FLT_MAX, "rsqc(+0)"},      {S::rsqf, 0, "xxxx", 0.0f, "rsqf(+0)"},
      {S::log, 0, "xxxx", -INFINITY, "log(+0)"},      {S::log, 0, "wwww", NAN, "log(-1)"},
      {S::log, 0, "zzzz", INFINITY, "log(inf)"},      {S::logc, 0, "xxxx", -FLT_MAX, "logc(+0)"},
      {S::log, 1, "xxxx", -INFINITY, "log(denorm) = log(0)"},
      {S::exp, 1, "zzzz", 0.0f, "exp(-inf)"},         {S::exp, 0, "zzzz", INFINITY, "exp(inf)"},
      {S::exp, 1, "wwww", 2.0f, "exp(1)"},            {S::exp, 0, "xxxx", 1.0f, "exp(0)"},
      {S::sqrt, 0, "wwww", NAN, "sqrt(-1)"},          {S::sqrt, 0, "yyyy", -0.0f, "sqrt(-0)"},
      {S::sqrt, 1, "yyyy", float(std::sqrt(3.0)), "sqrt(3)"},
  };
  Program p;
  std::vector<std::pair<std::array<uint32_t, 3>, bool>> ins;
  for (size_t i = 0; i < cases.size(); ++i) {
    ins.push_back(A(ScaOp(cases[i].op, uint32_t(10 + i), "x", C(cases[i].c, cases[i].comp))));
  }
  // Export 4 results per interpolator: move rI.x into the needed component.
  for (size_t i = 0; i < cases.size(); i += 4) {
    for (size_t k = 0; k < 4 && i + k < cases.size(); ++k) {
      Alu m = Mov(uint32_t(i / 4), R(uint32_t(10 + i + k), "xxxx"), k == 0 ? "x" : k == 1 ? "y" : k == 2 ? "z" : "w");
      m.exp = true;
      ins.push_back(A(m));
    }
  }
  int b = p.Block(ins);
  p.Exec(2, b);
  auto e = RunVs(env, p);
  for (size_t i = 0; i < cases.size(); ++i) {
    CheckF(cases[i].name, e->reg[i / 4][i % 4], cases[i].want);
  }
}

void TestSaturateAndPrevScalar() {
  Env env;
  env.SetConst(false, 0, 2.0f, NAN, -3.0f, 0.0f);
  Program p;
  Alu s1 = ScaOp(S::maxs, 1, "x", C(0, "xxxx"));  // ps = 2, r1.x = sat(2) = 1
  s1.ssat = true;
  Alu v1 = VecOp(V::add, 2, "xyzw", C(0), C(0, "wwww"));
  v1.vsat = true;                                   // sat(2, NaN, -3, 0)
  Alu s2 = ScaOp(S::adds_prev, 3, "x", C(0, "wwww"));  // 0 + ps
  int b = p.Block({A(s1), A(v1), A(s2), A(Export(0, R(1))), A(Export(1, R(2))), A(Export(2, R(3)))});
  p.Exec(2, b);
  auto e = RunVs(env, p);
  CheckF("scalar sat", e->reg[0][0], 1.0f);
  CheckF("vector sat 2", e->reg[1][0], 1.0f);
  CheckF("vector sat NaN -> 0", e->reg[1][1], 0.0f);
  CheckF("vector sat -3", e->reg[1][2], 0.0f);
  CheckF("ps keeps unsaturated result", e->reg[2][0], 2.0f);
}

void TestPredication() {
  Env env;
  env.SetConst(false, 0, 0.0f, 5.0f, 7.0f, 1.0f);
  Program p;
  // p0 = (c0.x == 0) -> true; ps = 0
  Alu setp = ScaOp(S::setp_eq, 1, "x", C(0, "xxxx"));
  Alu on = Mov(2, C(0, "yyyy"), "x"); on.pred = true; on.pred_cond = true;
  Alu off = Mov(2, C(0, "zzzz"), "y"); off.pred = true; off.pred_cond = false;
  Alu off_s = ScaOp(S::maxs, 3, "x", C(0, "wwww")); off_s.pred = true; off_s.pred_cond = false;
  Alu prev = ScaOp(S::adds_prev, 4, "x", C(0, "xxxx"));  // ps unchanged by off_s: 0 + 0
  int b0 = p.Block({A(setp), A(on), A(off), A(off_s), A(prev)});
  int b1 = p.Block({A(Mov(5, C(0, "wwww"), "x"))});
  int b2 = p.Block({A(Mov(5, C(0, "zzzz"), "y"))});
  int b3 = p.Block({A(Export(0, R(2))), A(Export(1, R(3))), A(Export(2, R(4))), A(Export(3, R(5))),
                    A(Export(4, R(1)))});
  p.Exec(1, b0);
  p.Exec(5, b1, 0, true);   // cexec_pred p0 -> taken
  p.Exec(5, b2, 0, false);  // cexec_pred !p0 -> not taken
  p.Exec(2, b3);
  auto e = RunVs(env, p);
  CheckF("(p0) taken", e->reg[0][0], 5.0f);
  CheckF("(!p0) skipped", e->reg[0][1], 0.0f);
  CheckF("predicated-off scalar no write", e->reg[1][0], 0.0f);
  CheckF("predicated-off scalar keeps ps", e->reg[2][0], 0.0f);
  CheckF("cexec_pred p0 taken", e->reg[3][0], 1.0f);
  CheckF("cexec_pred !p0 skipped", e->reg[3][1], 0.0f);
  CheckF("setp_eq ps = 0", e->reg[4][0], 0.0f);
}

void TestBoolCallJmp() {
  Env env;
  env.SetConst(false, 0, 1.0f, 2.0f, 3.0f, 4.0f);
  env.SetBool(5, true);
  Program p;
  int b_a = p.Block({A(Mov(1, C(0, "xxxx"), "x"))});
  int b_b = p.Block({A(Mov(1, C(0, "yyyy"), "y"))});
  int b_sub = p.Block({A(VecOp(V::add, 2, "x", R(2), C(0, "xxxx")))});
  int b_out = p.Block({A(Export(0, R(1))), A(Export(1, R(2)))});
  p.Exec(3, b_a, 5, true);   // 0 cexec b5 == 1 -> taken
  p.Exec(3, b_b, 5, false);  // 1 cexec b5 == 0 -> skipped
  p.Call(5);                 // 2
  p.Call(5);                 // 3
  p.Jmp(7, false, false);    // 4 -> 7
  p.Exec(1, b_sub);          // 5 subroutine
  p.Ret();                   // 6
  p.Exec(2, b_out);          // 7
  auto e = RunVs(env, p);
  CheckF("cexec bool taken", e->reg[0][0], 1.0f);
  CheckF("cexec bool skipped", e->reg[0][1], 0.0f);
  CheckF("two calls", e->reg[1][0], 2.0f);
}

void TestLoopsAndRelative() {
  Env env;
  for (uint32_t i = 0; i < 16; ++i) env.SetConst(false, i, float(1u << i), 0, 0, 0);
  env.SetLoop(0, 4, 1, 2);  // aL = 1, 3, 5, 7
  env.SetLoop(1, 0, 0, 0);  // count 0 -> skipped
  Program p;
  Alu acc = VecOp(V::add, 1, "x", R(1), C(0, "xxxx"));
  acc.s2.rel = true;  // c[aL+0]
  Alu wr = Mov(10, C(0, "xxxx"), "x");
  wr.s1.rel = true; wr.s2.rel = true; wr.vdst_rel = true;  // r[aL+10] = c[aL]
  Alu cnt = VecOp(V::add, 3, "x", R(3), C(0, "xxxx"));  // + c0.x = 1 per iteration
  int body = p.Block({A(acc), A(wr), A(cnt)});
  int body2 = p.Block({A(VecOp(V::add, 4, "x", R(4), C(0, "xxxx")))});
  int out = p.Block({A(Export(0, R(1))), A(Export(1, R(11))), A(Export(2, R(17))), A(Export(3, R(3))),
                     A(Export(4, R(4)))});
  p.LoopStart(0, 3);  // 0
  p.Exec(1, body);    // 1
  p.LoopEnd(0, 1);    // 2
  p.LoopStart(1, 6);  // 3
  p.Exec(1, body2);   // 4
  p.LoopEnd(1, 4);    // 5
  p.Exec(2, out);     // 6
  auto e = RunVs(env, p);
  CheckF("loop sum c[aL]", e->reg[0][0], float(2 + 8 + 32 + 128));
  CheckF("r[aL+10] at aL=1", e->reg[1][0], 2.0f);
  CheckF("r[aL+10] at aL=7", e->reg[2][0], 128.0f);
  CheckF("loop count 4", e->reg[3][0], 4.0f);
  CheckF("loop count 0 skipped", e->reg[4][0], 0.0f);
}

void TestPredicatedBreak() {
  Env env;
  env.SetConst(false, 0, 1.0f, -3.0f, 0, 0);
  env.SetLoop(0, 10, 0, 1);
  Program p;
  // r1.x += 1; p0 = (r1.x - 3 > 0)... use setp_gt on (r1.x + c0.y).
  Alu inc = VecOp(V::add, 1, "x", R(1), C(0, "xxxx"));
  Alu tmp = VecOp(V::add, 2, "x", R(1), C(0, "yyyy"));
  Alu setp = ScaOp(S::setp_gt, 3, "x", R(2, "xxxx"));
  int body = p.Block({A(inc), A(tmp), A(setp)});
  int out = p.Block({A(Export(0, R(1)))});
  p.LoopStart(0, 3);
  p.Exec(1, body);
  p.LoopEnd(0, 1, true, true);
  p.Exec(2, out);
  auto e = RunVs(env, p);
  CheckF("predicated break after 4 iterations", e->reg[0][0], 4.0f);
}

void TestA0Relative() {
  Env env;
  for (uint32_t i = 0; i < 8; ++i) env.SetConst(false, i, float(i) * 10.0f, 0, 0, 0);
  env.SetConst(false, 20, 0, 0, 0, 1.5f);
  env.SetConst(false, 21, 0, 0, 0, 300.0f);
  env.SetConst(false, 22, 0, 0, 0, -0.5f);
  env.SetConst(false, 23, 0, 0, 0, 2.49f);
  Program p;
  auto load = [](uint32_t dst, uint32_t base) {
    Alu x = Mov(dst, C(base, "xxxx"), "x");
    x.s1.rel = true; x.s2.rel = true; x.a0_rel = true;
    return x;
  };
  int b = p.Block({A(VecOp(V::maxa, 9, "", C(20), C(20))), A(load(1, 2)),   // a0 = 2 -> c4
                   A(VecOp(V::maxa, 9, "", C(21), C(21))), A(load(2, 2)),   // a0 = 255 -> c257 > size -> 0
                   A(VecOp(V::maxa, 9, "", C(22), C(22))), A(load(3, 2)),   // a0 = floor(0) = 0 -> c2
                   A(VecOp(V::maxa, 9, "", C(23), C(23))), A(load(4, 1)),   // a0 = floor(2.99) = 2 -> c3
                   A(Export(0, R(1))), A(Export(1, R(2))), A(Export(2, R(3))), A(Export(3, R(4)))});
  p.Exec(2, b);
  env.regs->values[XE_GPU_REG_SQ_VS_CONST] = 255u << 12;
  auto e = RunVs(env, p);
  CheckF("maxa 1.5 -> a0 2", e->reg[0][0], 40.0f);
  CheckF("maxa 300 -> a0 255, out of range const reads 0", e->reg[1][0], 0.0f);
  CheckF("maxa -0.5 -> a0 0", e->reg[2][0], 20.0f);
  CheckF("maxa 2.49 -> a0 2", e->reg[3][0], 30.0f);
}

void WriteDword(Memory& m, uint32_t addr, uint32_t v, bool big_endian) {
  if (big_endian) v = __builtin_bswap32(v);
  m.Write(addr, &v, 4);
}

void TestVertexFetch() {
  Env env;
  Memory& m = env.M();
  const uint32_t base = 0x10000;
  // Buffer A (endian none): element stride 4 dwords; vertex 1 at dword 4.
  WriteDword(m, base + 16, 0x80FF4000u, false);          // 8888
  WriteDword(m, base + 20, 0x7F80C001u, false);          // 8888 signed
  WriteDword(m, base + 24, (0xC000u << 16) | 0x3C00u, false);  // half (1, -2)
  // 2_10_10_10: x=511, y=-512(0x200), z=1, w=3 (2 bits)
  WriteDword(m, base + 28, 511u | (0x200u << 10) | (1u << 20) | (3u << 30), false);
  // fetch constant 0: address, endian none, size 64 words
  env.regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 0] = 3u | base;
  env.regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 1] = 0u | (64u << 2);
  // Buffer B (8in32): float4 at vertex 1, stride 4
  const uint32_t base2 = 0x20000;
  float f4[4] = {1.5f, -2.25f, 1e30f, -0.0f};
  for (int i = 0; i < 4; ++i) WriteDword(m, base2 + 16 + 4 * i, Bits(f4[i]), true);
  env.regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 2] = 3u | base2;
  env.regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 3] = 2u | (8u << 2);  // 8 words

  VFetch unorm;  unorm.fetch_const = 0; unorm.format = 6; unorm.dst = 1;
  VFetch snorm = unorm; snorm.mini = true; snorm.offset = 1; snorm.is_signed = true; snorm.dst = 2;
  VFetch half = unorm; half.mini = true; half.offset = 2; half.format = 31; half.dst = 3; half.dst_swz = "xy01";
  VFetch p1010 = unorm; p1010.mini = true; p1010.offset = 3; p1010.format = 7; p1010.is_signed = true; p1010.dst = 4;
  VFetch fl = unorm; fl.fetch_const = 1; fl.format = 38; fl.dst = 5;
  VFetch oob = fl; oob.dst = 6; oob.offset = 4;  // words 8..11 >= size 8 -> 0
  VFetch intf = unorm; intf.mini = false; intf.normalized = false; intf.dst = 7; intf.exp_adjust = -1;
  Program p;
  int b = p.Block({F(unorm.Encode()), F(snorm.Encode()), F(half.Encode()), F(p1010.Encode()),
                   F(fl.Encode()), F(oob.Encode()), F(intf.Encode()),
                   A(Export(0, R(1))), A(Export(1, R(2))), A(Export(2, R(3))), A(Export(3, R(4))),
                   A(Export(4, R(5))), A(Export(5, R(6))), A(Export(6, R(7)))});
  p.Exec(2, b);
  auto e = RunVs(env, p, 1);
  const float u = 1.0f / 255.0f, s = 1.0f / 127.0f;
  CheckF("8888 unorm x", e->reg[0][0], 0.0f);
  CheckF("8888 unorm y", e->reg[0][1], 64.0f * u);
  CheckF("8888 unorm z", e->reg[0][2], 255.0f * u);
  CheckF("8888 unorm w", e->reg[0][3], 128.0f * u);
  CheckF("8888 snorm x", e->reg[1][0], 1.0f * s);
  CheckF("8888 snorm y", e->reg[1][1], -64.0f * s);
  CheckF("8888 snorm z (-128 -> -1)", e->reg[1][2], -1.0f);
  CheckF("8888 snorm w", e->reg[1][3], 127.0f * s);
  CheckF("16_16_FLOAT x", e->reg[2][0], 1.0f);
  CheckF("16_16_FLOAT y", e->reg[2][1], -2.0f);
  CheckF("dest swizzle 0", e->reg[2][2], 0.0f);
  CheckF("dest swizzle 1", e->reg[2][3], 1.0f);
  CheckF("2_10_10_10 snorm x", e->reg[3][0], 511.0f * (1.0f / 511.0f));
  CheckF("2_10_10_10 snorm y (min -> -1)", e->reg[3][1], -1.0f);
  CheckF("2_10_10_10 snorm z", e->reg[3][2], 1.0f * (1.0f / 511.0f));
  CheckF("2_10_10_10 snorm w (-1)", e->reg[3][3], -1.0f);
  for (int i = 0; i < 4; ++i) CheckF("32x4 float 8in32", e->reg[4][i], f4[i]);
  for (int i = 0; i < 4; ++i) CheckF("out of range words read 0", e->reg[5][i], 0.0f);
  CheckF("8888 integer exp_adjust -1 y", e->reg[6][1], 32.0f);
  CheckF("8888 integer exp_adjust -1 z", e->reg[6][2], 127.5f);
}

void TestQuadDivergenceAndGradients() {
  Env env;
  env.SetConst(true, 0, 100.0f, 0, 0, 0);
  Program p;
  // p0 = r1.x > 0 (lanes 2, 3)
  Alu setp = ScaOp(S::setp_gt, 5, "x", R(1, "xxxx"));
  Alu addx = VecOp(V::add, 0, "x", R(0), C(0));
  TFetch tf;
  TFetch gg; gg.op = 18; gg.dst = 3; gg.comp_lod = false;
  int b0 = p.Block({A(setp)});
  int b1 = p.Block({A(addx)});
  int b2 = p.Block({F(tf.Encode()), F(gg.Encode()), A(Export(0, R(2))), A(Export(1, R(3)))});
  p.Exec(1, b0);              // 0
  p.Jmp(3, true, true);       // 1: (p0) jmp 3
  p.Exec(1, b1);              // 2: lanes 0, 1 only
  p.Exec(2, b2);              // 3
  ShaderProgram sp = p.Build(true);
  float inputs[4][16][4] = {};
  const float u[4] = {1, 2, 5, 10}, v[4] = {3, 3.5f, 7, 8};
  for (int l = 0; l < 4; ++l) {
    inputs[l][0][0] = u[l];
    inputs[l][0][1] = v[l];
    inputs[l][1][0] = float(l) - 1.5f;
  }
  PixelQuad quad{};
  quad.x = 10; quad.y = 20;
  quad.coverage[0] = 1; quad.coverage[1] = 0; quad.coverage[2] = 1; quad.coverage[3] = 1;
  RecordingTextureSampler sampler;
  auto out = std::make_unique<RawExports[]>(4);
  DrawStats stats;
  RunPixelQuadRaw(sp, *env.regs, quad, inputs, &sampler, out.get(), stats, nullptr);
  CheckB("4 fetch requests", sampler.all.size() == 4, true);
  // r0.x after divergence: 101, 102, 5, 10.
  const float want_x[4] = {101, 102, 5, 10};
  for (size_t i = 0; i < sampler.all.size() && i < 4; ++i) {
    const TexFetchRequest& r = sampler.all[i];
    CheckF("tfetch coord x", r.coord[0], want_x[r.lane]);
    CheckF("tfetch grad_h x (TR-TL, coarse)", r.grad_h[0], 1.0f);
    CheckF("tfetch grad_v x (BL-TL)", r.grad_v[0], -96.0f);
    CheckF("tfetch grad_h y", r.grad_h[1], 0.5f);
    CheckF("tfetch grad_v y", r.grad_v[1], 4.0f);
    CheckB("tfetch has grads", r.has_grads, true);
    CheckB("tfetch computed lod", r.use_computed_lod, true);
    CheckB("tfetch helper flag", r.helper_lane, r.lane == 1);
  }
  for (int l = 0; l < 4; ++l) {
    CheckF("tfetch result = coord (test sampler)", out[l].reg[0][0], want_x[l]);
    CheckF("getGradients x = ddx(src.x)", out[l].reg[1][0], 1.0f);
    CheckF("getGradients y = ddy(src.x)", out[l].reg[1][1], -96.0f);
    CheckF("getGradients z = ddx(src.y)", out[l].reg[1][2], 0.5f);
    CheckF("getGradients w = ddy(src.y)", out[l].reg[1][3], 4.0f);
  }
}

void TestKill() {
  Env env;
  env.SetConst(true, 0, 1.0f, 2.0f, 3.0f, 4.0f);
  Program p;
  int b = p.Block({A(VecOp(V::kill_eq, 2, "", R(0), C(0))), A(Export(0, R(0)))});
  p.Exec(2, b);
  ShaderProgram sp = p.Build(true);
  float inputs[4][16][4] = {};
  inputs[0][0][0] = 1.0f;   // x equal -> kill
  inputs[1][0][3] = 4.0f;   // w equal -> kill
  inputs[2][0][0] = 1.5f;   // none
  inputs[3][0][1] = NAN;    // none (NaN != 2)
  PixelQuad quad{};
  for (int l = 0; l < 4; ++l) quad.coverage[l] = 1;
  auto out = std::make_unique<RawExports[]>(4);
  DrawStats stats;
  RunPixelQuadRaw(sp, *env.regs, quad, inputs, nullptr, out.get(), stats, nullptr);
  CheckB("kill_eq x", out[0].killed, true);
  CheckB("kill_eq w", out[1].killed, true);
  CheckB("kill_eq none", out[2].killed, false);
  CheckB("kill_eq NaN", out[3].killed, false);
  CheckF("killed pixel keeps executing", out[0].reg[0][0], 1.0f);
}

void TestTaint() {
  Env env;
  env.SetConst(false, 0, 4.0f, 2.0f, 0, 0);
  Program p;
  Alu r = ScaOp(S::rcp, 1, "x", C(0, "xxxx"));                      // taint 1
  Alu q = ScaOp(S::rsq, 1, "y", R(1, "xxxx"));                      // taint 2
  Alu m = VecOp(V::mul, 2, "xy", R(1), C(0));                        // x:1 y:2
  int b = p.Block({A(r), A(q), A(m), A(Export(0, R(2))), A(Export(1, C(0)))});
  p.Exec(2, b);
  auto e = RunVs(env, p);
  CheckB("taint rcp", e->taint[0][0] == 1, true);
  CheckB("taint rsq(rcp)", e->taint[0][1] == 2, true);
  CheckB("taint constant", e->taint[1][0] == 0, true);
  CheckF("rcp(4) * 4", e->reg[0][0], 1.0f);
}

}  // namespace

int main() {
  TestLegacyMulAndMad();
  TestMaxMinNaN();
  TestTranscendentals();
  TestSaturateAndPrevScalar();
  TestPredication();
  TestBoolCallJmp();
  TestLoopsAndRelative();
  TestPredicatedBreak();
  TestA0Relative();
  TestVertexFetch();
  TestQuadDivergenceAndGradients();
  TestKill();
  TestTaint();
  std::printf("test_interp: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
