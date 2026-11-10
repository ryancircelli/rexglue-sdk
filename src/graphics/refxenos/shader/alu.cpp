// Xenos ALU operations (vector and scalar) for the refxenos interpreter.
//
// Where Xenia's interpreter (shader_interpreter.cc) and its translators
// (dxbc_/spirv_shader_translator_alu.cc) disagree, the translators are
// followed: they are what runs the game correctly.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "shader/internal.h"

namespace refxenos {

RX_DEFINE_RULE(xenos_alu_denormal_flush, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator.cc:837 "
               "(denormals flushed on ALU inputs and outputs, sign preserved, "
               "\"similar to the real hardware\")",
               false, "");
RX_DEFINE_RULE(xenos_alu_legacy_mul, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:103; "
               "AMD R5xx Acceleration 8.7.5 Legacy multiply behavior (+-0 or "
               "denormal times anything, including inf/NaN, is +0)",
               false, "");
RX_DEFINE_RULE(xenos_alu_mad_unfused, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:94 "
               "(mad = legacy mul rounded, then add rounded; not fused)",
               false, "");
RX_DEFINE_RULE(xenos_alu_dot_order, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:196 "
               "(legacy products summed ((x+y)+z)+w, each rounded; dp2add "
               "(x+y)+c.x)",
               false, "");
RX_DEFINE_RULE(xenos_alu_max_min, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:132 "
               "(max = a >= b ? a : b, min = a < b ? a : b, so a NaN in either "
               "operand selects b); XDK microcode_alu_vector_max.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_max4_order, 1,
               "assumed: XDK microcode_alu_vector_max4.htm gives no NaN order; "
               "IPR2015-00325 sq_alu checks XYZW; implemented as the max rule "
               "chained x, y, z, w (dxbc_shader_translator_alu.cc:355 FIXME)",
               true,
               "PS: max4 r0, c0; oC0 = r0; c0 = (NaN, 1, 2, NaN) and every "
               "permutation of one/two NaNs among (NaN,1,2,3); render 1 pixel to "
               "32_32_32_32_FLOAT, read back r0.x");
RX_DEFINE_RULE(xenos_alu_cube, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:246 "
               "(operand 0 .z_xy; result (TC, SC, 2*MA signed, face id); "
               "XDK microcode_alu_vector_cube.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_setp_push, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:394 "
               "(p0 = a.w == 0 && cmp(b.w, 0); result.x = (a.x == 0 && cmp(b.x, "
               "0)) ? 0 : a.x + 1, replicated)",
               false, "");
RX_DEFINE_RULE(xenos_alu_vector_kill, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:508 "
               "(kill if the comparison holds for any component; result 1.0 if "
               "killed else 0.0, replicated); XDK microcode_alu_vector_kill_eq.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_dst, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:574 "
               "(1, legacy a.y*b.y, a.z, b.w)",
               false, "");
RX_DEFINE_RULE(xenos_alu_a0_rounding, 1,
               "XDK microcode_alu_scalar_maxas.htm, microcode_alu_scalar_movas.htm, "
               "microcode_alu_vector_maxa.htm (a0 = floor(x + 0.5) (maxasf/movasf: "
               "floor(x)); !(a0 >= -256) -> -256 (NaN too); > 255 -> 255). "
               "Differs from dxbc_shader_translator_alu.cc:916 which clamps "
               "maxas to [0, 255]; the XDK pseudocode is followed",
               false, "");
RX_DEFINE_RULE(xenos_alu_rcp, 1,
               "assumed: rcp = 1/x computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rcp r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_rcpc, 1,
               "assumed: rcpc = 1/x, +-inf results clamped to +-FLT_MAX computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rcpc r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_rcpf, 1,
               "assumed: rcpf = 1/x, +-inf results flushed to +-0 computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rcpf r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_rsq, 1,
               "assumed: rsq = 1/sqrt(x) computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rsq r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_rsqc, 1,
               "assumed: rsqc = 1/sqrt(x), +-inf results clamped to +-FLT_MAX computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rsqc r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_rsqf, 1,
               "assumed: rsqf = 1/sqrt(x), +-inf results flushed to +-0 computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`rsqf r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_sqrt, 1,
               "assumed: sqrt = sqrt(x) computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`sqrt r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_log, 1,
               "assumed: log = log2(x) computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`log r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_logc, 1,
               "assumed: logc = log2(x), -inf clamped to -FLT_MAX computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`logc r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_exp, 1,
               "assumed: exp = 2^x computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. xenia-canary 3a44f20c7 instead rounds to 21 mantissa bits "
               "(shader_interpreter.cc:285, \"isn't confirmed\"), available as "
               "InterpOptions::transcendental_mantissa_bits = 21",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`exp r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_sin, 1,
               "assumed: sin = sin(x radians) computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. Xenia does not reduce sin/cos precision",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`sin r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_cos, 1,
               "assumed: cos = cos(x radians) computed exactly (double) and rounded once "
               "to f32 (round to nearest even), then the XDK special cases, then "
               "the output denormal flush. Xenia does not reduce sin/cos precision",
               true,
               "sweep all 2^32 float32 inputs (or a stratified 2^24 subset incl. "
               "denormals, +-0, +-inf, NaNs, powers of two) through a ps that does "
               "`cos r0.x, c0.x` (input per pixel from a 32_FLOAT texture) and "
               "exports bits via a 32_32_32_32_FLOAT RT; compare bit-exact");
RX_DEFINE_RULE(xenos_alu_log_special, 1,
               "assumed: log/logc of +0 = -inf (logc -> -FLT_MAX), of a negative "
               "value = NaN, of +inf = +inf, as IEEE log2 "
               "(XDK microcode_alu_scalar_log.htm states no special cases)",
               true,
               "PS: r0.x = log(c0.x), r0.y = logc(c0.x) for c0.x in {+0, -0, -1, "
               "-inf, +inf, NaN}; export to 32_32_32_32_FLOAT and read bits");
RX_DEFINE_RULE(xenos_alu_clamped_transcendentals, 1,
               "XDK microcode_alu_scalar_logc.htm (-inf -> -MAX_FLOAT), "
               "microcode_alu_scalar_rcpc.htm/rsqc.htm (+-inf -> +-MAX_FLOAT), "
               "microcode_alu_scalar_rcpf.htm/rsqf.htm (+-inf -> +-0); "
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:847",
               false, "");
RX_DEFINE_RULE(xenos_alu_scalar_setp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:947 "
               "(setp_eq/ne/gt/ge: p0 = cmp(a, 0), ps = p0 ? 0 : 1; setp_inv: p0 "
               "= a == 1, ps = p0 ? 0 : (a == 0 ? 1 : a); setp_pop: t = a - 1, p0 "
               "= t <= 0, ps = p0 ? 0 : t; setp_clr: ps = FLT_MAX, p0 = 0; "
               "setp_rstr: p0 = a == 0, ps = p0 ? 0 : a); XDK "
               "microcode_alu_scalar_setp_*.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_scalar_kill, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:1011 "
               "(kills_eq/gt/ge/ne/one compare a with 0 (one: 1); ps = 1.0 if "
               "killed else 0.0); XDK microcode_alu_scalar_kills_eq.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_muls_prev2, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:755; "
               "XDK microcode_alu_scalar_muls_prev2.htm (ps == -MAX_FLOAT, ps "
               "not finite, b not finite or b <= 0 -> -MAX_FLOAT; else legacy "
               "a * ps)",
               false, "");
RX_DEFINE_RULE(xenos_alu_sincos_radians, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:1149 "
               "(sin/cos of the value in radians, no range reduction beyond "
               "IEEE); XDK microcode_alu_scalar_sin.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_frc_trunc_floor, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc "
               "(frc = a - floor(a) rounded once, trunc, floor as IEEE); XDK "
               "microcode_alu_vector_frc.htm",
               false, "");
RX_DEFINE_RULE(xenos_alu_saturate, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc "
               "(clamp via D3D10 _sat: NaN -> 0, result in [+0, 1]); "
               "XDK microcode_instruction_modifiers.htm",
               false, "");

namespace shader_internal {

uint32_t FloatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float BitsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

float FlushDenormal(float f) {
  uint32_t u = FloatBits(f);
  if ((u & 0x7F800000u) == 0 && (u & 0x007FFFFFu) != 0) {
    RX_USE(xenos_alu_denormal_flush);
    return BitsFloat(u & 0x80000000u);
  }
  return f;
}

float Saturate(float f) {
  RX_USE(xenos_alu_saturate);
  if (!(f > 0.0f)) return 0.0f;
  if (f > 1.0f) return 1.0f;
  return f;
}

float LegacyMul(float a, float b) {
  RX_USE(xenos_alu_legacy_mul);
  a = FlushDenormal(a);
  b = FlushDenormal(b);
  if (a == 0.0f || b == 0.0f) return 0.0f;
  return a * b;
}

namespace {

float ReduceMantissa(float f, uint32_t bits) {
  if (bits >= 23) return f;
  uint32_t u = FloatBits(f);
  if ((u & 0x7F800000u) == 0x7F800000u) return f;
  uint32_t drop = 23 - bits;
  uint32_t mask = (1u << drop) - 1;
  uint32_t r = (u & ~mask) + ((u & (1u << (drop - 1))) ? (1u << drop) : 0u);
  if ((r & 0x7F800000u) == 0x7F800000u) r = u & ~mask;
  return BitsFloat(r);
}

float RoundTranscendental(double d, bool reducible) {
  float f = float(d);
  if (reducible) {
    f = ReduceMantissa(f, GetInterpOptions().transcendental_mantissa_bits);
  }
  return f;
}

float Log2(float a) {
  RX_USE(xenos_alu_log_special);
  if (std::isnan(a)) return a;
  if (a == 0.0f) return -INFINITY;
  if (a < 0.0f) return std::numeric_limits<float>::quiet_NaN();
  return RoundTranscendental(std::log2(double(a)), true);
}

float Rcp(float a) {
  if (a == 0.0f) return std::signbit(a) ? -INFINITY : INFINITY;
  return RoundTranscendental(1.0 / double(a), true);
}

float Rsq(float a) {
  if (a == 0.0f) return std::signbit(a) ? -INFINITY : INFINITY;
  if (std::isnan(a)) return a;
  if (a < 0.0f) return std::numeric_limits<float>::quiet_NaN();
  return RoundTranscendental(1.0 / std::sqrt(double(a)), true);
}

float ClampInfToMax(float f) {
  RX_USE(xenos_alu_clamped_transcendentals);
  if (f == INFINITY) return FLT_MAX;
  if (f == -INFINITY) return -FLT_MAX;
  return f;
}

float ClampInfToZero(float f) {
  RX_USE(xenos_alu_clamped_transcendentals);
  if (f == INFINITY) return 0.0f;
  if (f == -INFINITY) return -0.0f;
  return f;
}

float Max(float a, float b) {
  RX_USE(xenos_alu_max_min);
  return a >= b ? a : b;
}
float Min(float a, float b) {
  RX_USE(xenos_alu_max_min);
  return a < b ? a : b;
}

float Frc(float a) {
  RX_USE(xenos_alu_frc_trunc_floor);
  return a - std::floor(a);
}

int32_t RoundA0(float x, bool floor_only) {
  RX_USE(xenos_alu_a0_rounding);
  float f = floor_only ? std::floor(x) : std::floor(x + 0.5f);
  if (!(f >= -256.0f)) return -256;
  if (f > 255.0f) return 255;
  return int32_t(f);
}

}  // namespace

bool IsTranscendentalScalarOp(ucode::AluScalarOpcode op) {
  using S = ucode::AluScalarOpcode;
  switch (op) {
    case S::kExp: case S::kLogc: case S::kLog: case S::kRcpc: case S::kRcpf:
    case S::kRcp: case S::kRsqc: case S::kRsqf: case S::kRsq: case S::kSqrt:
    case S::kSin: case S::kCos:
      return true;
    default:
      return false;
  }
}

void ExecuteVectorOp(ucode::AluVectorOpcode op, const float a[4],
                     const float b[4], const float c[4], VectorOpResult* r) {
  using V = ucode::AluVectorOpcode;
  float* v = r->v;
  auto replicate = [&](float x) { v[0] = v[1] = v[2] = v[3] = x; };
  switch (op) {
    case V::kAdd:
      for (int i = 0; i < 4; ++i) v[i] = a[i] + b[i];
      break;
    case V::kMul:
      for (int i = 0; i < 4; ++i) v[i] = LegacyMul(a[i], b[i]);
      break;
    case V::kMax:
      for (int i = 0; i < 4; ++i) v[i] = Max(a[i], b[i]);
      break;
    case V::kMin:
      for (int i = 0; i < 4; ++i) v[i] = Min(a[i], b[i]);
      break;
    case V::kSeq:
      for (int i = 0; i < 4; ++i) v[i] = a[i] == b[i] ? 1.0f : 0.0f;
      break;
    case V::kSgt:
      for (int i = 0; i < 4; ++i) v[i] = a[i] > b[i] ? 1.0f : 0.0f;
      break;
    case V::kSge:
      for (int i = 0; i < 4; ++i) v[i] = a[i] >= b[i] ? 1.0f : 0.0f;
      break;
    case V::kSne:
      for (int i = 0; i < 4; ++i) v[i] = a[i] != b[i] ? 1.0f : 0.0f;
      break;
    case V::kFrc:
      for (int i = 0; i < 4; ++i) v[i] = Frc(a[i]);
      break;
    case V::kTrunc:
      RX_USE(xenos_alu_frc_trunc_floor);
      for (int i = 0; i < 4; ++i) v[i] = std::trunc(a[i]);
      break;
    case V::kFloor:
      RX_USE(xenos_alu_frc_trunc_floor);
      for (int i = 0; i < 4; ++i) v[i] = std::floor(a[i]);
      break;
    case V::kMad:
      RX_USE(xenos_alu_mad_unfused);
      for (int i = 0; i < 4; ++i) {
        // Legacy product, rounded, flushed like any intermediate.
        float p = FlushDenormal(LegacyMul(a[i], b[i]));
        v[i] = p + c[i];
      }
      break;
    case V::kCndEq:
      for (int i = 0; i < 4; ++i) v[i] = a[i] == 0.0f ? b[i] : c[i];
      break;
    case V::kCndGe:
      for (int i = 0; i < 4; ++i) v[i] = a[i] >= 0.0f ? b[i] : c[i];
      break;
    case V::kCndGt:
      for (int i = 0; i < 4; ++i) v[i] = a[i] > 0.0f ? b[i] : c[i];
      break;
    case V::kDp4:
    case V::kDp3: {
      RX_USE(xenos_alu_dot_order);
      int n = op == V::kDp4 ? 4 : 3;
      float s = FlushDenormal(LegacyMul(a[0], b[0]));
      for (int i = 1; i < n; ++i) {
        s = FlushDenormal(s + FlushDenormal(LegacyMul(a[i], b[i])));
      }
      replicate(s);
    } break;
    case V::kDp2Add: {
      RX_USE(xenos_alu_dot_order);
      float s = FlushDenormal(FlushDenormal(LegacyMul(a[0], b[0])) +
                              FlushDenormal(LegacyMul(a[1], b[1])));
      replicate(s + c[0]);
    } break;
    case V::kCube: {
      RX_USE(xenos_alu_cube);
      // Operand 0 is .z_xy: [0] = z, [2] = x, [3] = y.
      float x = a[2], y = a[3], z = a[0];
      float tc, sc, ma, id;
      if (std::fabs(z) >= std::fabs(x) && std::fabs(z) >= std::fabs(y)) {
        tc = -y;
        sc = z < 0.0f ? -x : x;
        ma = z;
        id = z < 0.0f ? 5.0f : 4.0f;
      } else if (std::fabs(y) >= std::fabs(x)) {
        tc = y < 0.0f ? -z : z;
        sc = x;
        ma = y;
        id = y < 0.0f ? 3.0f : 2.0f;
      } else {
        tc = -y;
        sc = x < 0.0f ? z : -z;
        ma = x;
        id = x < 0.0f ? 1.0f : 0.0f;
      }
      v[0] = tc;
      v[1] = sc;
      v[2] = ma * 2.0f;
      v[3] = id;
    } break;
    case V::kMax4: {
      RX_USE(xenos_alu_max4_order);
      float m = a[0];
      for (int i = 1; i < 4; ++i) m = Max(a[i], m);
      replicate(m);
    } break;
    case V::kSetpEqPush:
    case V::kSetpNePush:
    case V::kSetpGtPush:
    case V::kSetpGePush: {
      RX_USE(xenos_alu_setp_push);
      auto cmp = [&](float x) {
        switch (op) {
          case V::kSetpEqPush: return x == 0.0f;
          case V::kSetpNePush: return x != 0.0f;
          case V::kSetpGtPush: return x > 0.0f;
          default: return x >= 0.0f;
        }
      };
      r->sets_predicate = true;
      r->predicate = a[3] == 0.0f && cmp(b[3]);
      replicate((a[0] == 0.0f && cmp(b[0])) ? 0.0f : a[0] + 1.0f);
    } break;
    case V::kKillEq:
    case V::kKillGt:
    case V::kKillGe:
    case V::kKillNe: {
      RX_USE(xenos_alu_vector_kill);
      bool k = false;
      for (int i = 0; i < 4; ++i) {
        switch (op) {
          case V::kKillEq: k |= a[i] == b[i]; break;
          case V::kKillGt: k |= a[i] > b[i]; break;
          case V::kKillGe: k |= a[i] >= b[i]; break;
          default: k |= a[i] != b[i]; break;
        }
      }
      r->kill = k;
      replicate(k ? 1.0f : 0.0f);
    } break;
    case V::kDst:
      RX_USE(xenos_alu_dst);
      v[0] = 1.0f;
      v[1] = LegacyMul(a[1], b[1]);
      v[2] = a[2];
      v[3] = b[3];
      break;
    case V::kMaxA:
      r->sets_a0 = true;
      r->a0 = RoundA0(a[3], false);
      for (int i = 0; i < 4; ++i) v[i] = Max(a[i], b[i]);
      break;
    default:
      replicate(0.0f);
      break;
  }
}

void ExecuteScalarOp(ucode::AluScalarOpcode op, float a, float b, float ps,
                     ScalarOpResult* r) {
  using S = ucode::AluScalarOpcode;
  float& v = r->v;
  auto setp = [&](bool p, float value) {
    RX_USE(xenos_alu_scalar_setp);
    r->sets_predicate = true;
    r->predicate = p;
    v = value;
  };
  auto kill = [&](bool k) {
    RX_USE(xenos_alu_scalar_kill);
    r->kill = k;
    v = k ? 1.0f : 0.0f;
  };
  switch (op) {
    case S::kAdds: v = a + b; break;
    case S::kAddsPrev: v = a + ps; break;
    case S::kMuls: v = LegacyMul(a, b); break;
    case S::kMulsPrev: v = LegacyMul(a, ps); break;
    case S::kMulsPrev2:
      RX_USE(xenos_alu_muls_prev2);
      if (ps == -FLT_MAX || !std::isfinite(ps) || !std::isfinite(b) ||
          !(b > 0.0f)) {
        v = -FLT_MAX;
      } else {
        v = LegacyMul(a, ps);
      }
      break;
    case S::kMaxs: v = Max(a, b); break;
    case S::kMins: v = Min(a, b); break;
    case S::kSeqs: v = a == 0.0f ? 1.0f : 0.0f; break;
    case S::kSgts: v = a > 0.0f ? 1.0f : 0.0f; break;
    case S::kSges: v = a >= 0.0f ? 1.0f : 0.0f; break;
    case S::kSnes: v = a != 0.0f ? 1.0f : 0.0f; break;
    case S::kFrcs: v = Frc(a); break;
    case S::kTruncs: RX_USE(xenos_alu_frc_trunc_floor); v = std::trunc(a); break;
    case S::kFloors: RX_USE(xenos_alu_frc_trunc_floor); v = std::floor(a); break;
    case S::kExp:
      RX_USE(xenos_alu_exp);
      if (std::isnan(a)) { v = a; break; }
      v = RoundTranscendental(std::exp2(double(a)), true);
      break;
    case S::kLogc: {
      RX_USE(xenos_alu_logc);
      float l = Log2(a);
      if (l == -INFINITY) {
        RX_USE(xenos_alu_clamped_transcendentals);
        l = -FLT_MAX;
      }
      v = l;
    } break;
    case S::kLog: RX_USE(xenos_alu_log); v = Log2(a); break;
    case S::kRcpc: RX_USE(xenos_alu_rcpc); v = ClampInfToMax(Rcp(a)); break;
    case S::kRcpf: RX_USE(xenos_alu_rcpf); v = ClampInfToZero(Rcp(a)); break;
    case S::kRcp: RX_USE(xenos_alu_rcp); v = Rcp(a); break;
    case S::kRsqc: RX_USE(xenos_alu_rsqc); v = ClampInfToMax(Rsq(a)); break;
    case S::kRsqf: RX_USE(xenos_alu_rsqf); v = ClampInfToZero(Rsq(a)); break;
    case S::kRsq: RX_USE(xenos_alu_rsq); v = Rsq(a); break;
    case S::kMaxAs:
      r->sets_a0 = true;
      r->a0 = RoundA0(a, false);
      v = Max(a, b);
      break;
    case S::kMaxAsf:
      r->sets_a0 = true;
      r->a0 = RoundA0(a, true);
      v = Max(a, b);
      break;
    case S::kSubs: v = a - b; break;
    case S::kSubsPrev: v = a - ps; break;
    case S::kSetpEq: { bool p = a == 0.0f; setp(p, p ? 0.0f : 1.0f); } break;
    case S::kSetpNe: { bool p = a != 0.0f; setp(p, p ? 0.0f : 1.0f); } break;
    case S::kSetpGt: { bool p = a > 0.0f; setp(p, p ? 0.0f : 1.0f); } break;
    case S::kSetpGe: { bool p = a >= 0.0f; setp(p, p ? 0.0f : 1.0f); } break;
    case S::kSetpInv: {
      bool p = a == 1.0f;
      setp(p, p ? 0.0f : (a == 0.0f ? 1.0f : a));
    } break;
    case S::kSetpPop: {
      float t = a - 1.0f;
      bool p = t <= 0.0f;
      setp(p, p ? 0.0f : t);
    } break;
    case S::kSetpClr: setp(false, FLT_MAX); break;
    case S::kSetpRstr: { bool p = a == 0.0f; setp(p, p ? 0.0f : a); } break;
    case S::kKillsEq: kill(a == 0.0f); break;
    case S::kKillsGt: kill(a > 0.0f); break;
    case S::kKillsGe: kill(a >= 0.0f); break;
    case S::kKillsNe: kill(a != 0.0f); break;
    case S::kKillsOne: kill(a == 1.0f); break;
    case S::kSqrt:
      RX_USE(xenos_alu_sqrt);
      if (std::isnan(a)) { v = a; break; }
      if (a == 0.0f) { v = a; break; }
      if (a < 0.0f) { v = std::numeric_limits<float>::quiet_NaN(); break; }
      v = RoundTranscendental(std::sqrt(double(a)), true);
      break;
    case S::kMulsc0:
    case S::kMulsc1: v = LegacyMul(a, b); break;
    case S::kAddsc0:
    case S::kAddsc1: v = a + b; break;
    case S::kSubsc0:
    case S::kSubsc1: v = a - b; break;
    case S::kSin:
      RX_USE(xenos_alu_sin);
      RX_USE(xenos_alu_sincos_radians);
      v = std::isfinite(a) ? RoundTranscendental(std::sin(double(a)), false)
                           : std::numeric_limits<float>::quiet_NaN();
      break;
    case S::kCos:
      RX_USE(xenos_alu_cos);
      RX_USE(xenos_alu_sincos_radians);
      v = std::isfinite(a) ? RoundTranscendental(std::cos(double(a)), false)
                           : std::numeric_limits<float>::quiet_NaN();
      break;
    case S::kRetainPrev: v = ps; break;
    default: v = 0.0f; break;
  }
}

std::string FormatFloat(float f) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%.9g(0x%08X)", double(f), FloatBits(f));
  return buf;
}

}  // namespace shader_internal
}  // namespace refxenos
