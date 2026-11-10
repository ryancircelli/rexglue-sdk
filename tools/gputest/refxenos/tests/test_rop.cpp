// refxenos output-merger known-answer tests.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "core/rop.h"

using namespace refxenos;
using namespace rex::graphics;  // XE_GPU_REG_* register indices
static int failures = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define EQ(a, b) do { ++checks; auto a_ = (a); auto b_ = (b); if (!(a_ == b_)) { \
  std::printf("FAIL %d: %s = 0x%llx, expected 0x%llx\n", __LINE__, #a, \
              (unsigned long long)a_, (unsigned long long)b_); ++failures; } } while (0)

template <typename R>
static void Set(RegisterFile& rf, R r) { rf[R::register_index] = r.value; }
static void SetF(RegisterFile& rf, uint32_t index, float value) { rf[index] = AsU32(value); }

static RegisterFile Base(xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X) {
  RegisterFile rf;
  std::memset(rf.values, 0, sizeof(rf.values));
  reg::RB_MODECONTROL mode{};
  mode.edram_mode = xenos::EdramMode::kColorDepth;
  Set(rf, mode);
  reg::RB_SURFACE_INFO surface{};
  surface.surface_pitch = 80;
  surface.msaa_samples = msaa;
  Set(rf, surface);
  reg::RB_COLOR_INFO color{};
  color.color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
  Set(rf, color);
  rf[XE_GPU_REG_RB_COLOR_MASK] = 0xFFFF;
  reg::RB_BLENDCONTROL blend{};
  blend.value = 0x00010001;
  for (auto index : reg::RB_BLENDCONTROL::rt_register_indices) rf[index] = blend.value;
  reg::RB_COLORCONTROL control{};
  control.alpha_func = xenos::CompareFunction::kAlways;
  Set(rf, control);
  reg::RB_DEPTH_INFO depth{};
  depth.depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  Set(rf, depth);
  return rf;
}

static PixelOut Output(float r, float g, float b, float a) {
  PixelOut out;
  out.color_written_mask = 1;
  out.color[0][0] = r;
  out.color[0][1] = g;
  out.color[0][2] = b;
  out.color[0][3] = a;
  return out;
}

static PixelQuad Quad(xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X) {
  PixelQuad q;
  uint32_t count = msaa == xenos::MsaaSamples::k4X ? 4 : (msaa == xenos::MsaaSamples::k2X ? 2 : 1);
  q.coverage[0] = (1u << count) - 1u;
  for (uint32_t s = 0; s < 4; ++s) q.sample_z[0][s] = 0.5;
  return q;
}

static uint32_t Pixel(Edram& edram, const RopState& state, uint32_t sample = 0) {
  return uint32_t(edram.ReadSample(state.color[0].surface, 0, 0, sample));
}

static void TestAlphaTest() {
  using C = xenos::CompareFunction;
  const C functions[] = {C::kNever, C::kLess, C::kEqual, C::kLessEqual,
                         C::kGreater, C::kNotEqual, C::kGreaterEqual, C::kAlways};
  const bool pass[] = {false, false, true, true, false, false, true, true};
  for (uint32_t i = 0; i < 8; ++i) {
    RegisterFile rf = Base();
    auto cc = rf.Get<reg::RB_COLORCONTROL>();
    cc.alpha_test_enable = 1;
    cc.alpha_func = functions[i];
    Set(rf, cc);
    SetF(rf, XE_GPU_REG_RB_ALPHA_REF, 0.5f);
    RopState state = DecodeRopState(rf);
    Edram edram;
    PixelQuad q = Quad();
    PixelOut out[4] = {Output(1, 0, 0, 0.5f), {}, {}, {}};
    DrawStats stats;
    RopQuad(state, edram, q, out, stats);
    EQ(Pixel(edram, state), pass[i] ? 0x800000FFu : 0u);
  }
  RegisterFile rf = Base();
  auto cc = rf.Get<reg::RB_COLORCONTROL>();
  cc.alpha_test_enable = 1;
  cc.alpha_func = C::kNotEqual;
  Set(rf, cc);
  RopState state = DecodeRopState(rf);
  Edram edram;
  PixelQuad q = Quad();
  PixelOut out[4] = {Output(1, 0, 0, std::nanf("")), {}, {}, {}};
  DrawStats stats;
  RopQuad(state, edram, q, out, stats);
  EQ(Pixel(edram, state), 0x000000FFu);
}

static void TestAlphaToCoverage() {
  RegisterFile rf = Base(xenos::MsaaSamples::k4X);
  auto cc = rf.Get<reg::RB_COLORCONTROL>();
  cc.alpha_to_mask_enable = 1;
  cc.alpha_to_mask_offset0 = 2;
  Set(rf, cc);
  RopState state = DecodeRopState(rf);
  Edram edram;
  PixelQuad q = Quad(xenos::MsaaSamples::k4X);
  PixelOut out[4] = {Output(1, 0, 0, 0.5f), {}, {}, {}};
  DrawStats stats;
  RopQuad(state, edram, q, out, stats);
  EQ(Pixel(edram, state, 0), 0u);
  EQ(Pixel(edram, state, 1), 0x800000FFu);
  EQ(Pixel(edram, state, 2), 0x800000FFu);
  EQ(Pixel(edram, state, 3), 0u);
  out[0].color_written_mask = 0;
  Edram unchanged;
  RopQuad(state, unchanged, q, out, stats);
  for (uint32_t s = 0; s < 4; ++s) EQ(Pixel(unchanged, state, s), 0u);
}

static void TestBlendAndMask() {
  RegisterFile rf = Base();
  reg::RB_BLENDCONTROL blend{};
  blend.color_srcblend = xenos::BlendFactor::kSrcAlpha;
  blend.color_destblend = xenos::BlendFactor::kOneMinusSrcAlpha;
  blend.color_comb_fcn = xenos::BlendOp::kAdd;
  blend.alpha_srcblend = xenos::BlendFactor::kOne;
  blend.alpha_destblend = xenos::BlendFactor::kZero;
  blend.alpha_comb_fcn = xenos::BlendOp::kAdd;
  Set(rf, blend);
  RopState state = DecodeRopState(rf);
  Edram edram;
  edram.WriteSample(state.color[0].surface, 0, 0, 0, 0xFF808040u);
  PixelQuad q = Quad();
  PixelOut out[4] = {Output(0.2f, 0.4f, 0.6f, 0.5f), {}, {}, {}};
  DrawStats stats;
  RopQuad(state, edram, q, out, stats);
  EQ(Pixel(edram, state), 0x808D733Au);

  blend.color_srcblend = xenos::BlendFactor::kConstantColor;
  blend.color_destblend = xenos::BlendFactor::kOne;
  blend.color_comb_fcn = xenos::BlendOp::kMin;
  Set(rf, blend);
  SetF(rf, XE_GPU_REG_RB_BLEND_RED, 0.5f);
  SetF(rf, XE_GPU_REG_RB_BLEND_GREEN, 0.5f);
  SetF(rf, XE_GPU_REG_RB_BLEND_BLUE, 0.5f);
  SetF(rf, XE_GPU_REG_RB_BLEND_ALPHA, 0.5f);
  state = DecodeRopState(rf);
  edram.WriteSample(state.color[0].surface, 0, 0, 0, 0xFFBFBFBFu);
  out[0] = Output(0.25f, 0.25f, 0.25f, 1.0f);
  RopQuad(state, edram, q, out, stats);
  EQ(Pixel(edram, state) & 0xFFu, 32u);

  rf = Base();
  rf[XE_GPU_REG_RB_COLOR_MASK] = 0x5;
  state = DecodeRopState(rf);
  edram.WriteSample(state.color[0].surface, 0, 0, 0, 0x44332211u);
  out[0] = Output(1, 1, 1, 1);
  RopQuad(state, edram, q, out, stats);
  EQ(Pixel(edram, state), 0x44FF22FFu);
}

static void TestAllBlendFactorsAndOps() {
  using F = xenos::BlendFactor;
  const F factors[] = {F::kZero, F::kOne, F::kSrcColor, F::kOneMinusSrcColor,
      F::kSrcAlpha, F::kOneMinusSrcAlpha, F::kDstColor, F::kOneMinusDstColor,
      F::kDstAlpha, F::kOneMinusDstAlpha, F::kConstantColor, F::kOneMinusConstantColor,
      F::kConstantAlpha, F::kOneMinusConstantAlpha, F::kSrcAlphaSaturate};
  const float dest_component = 64.0f / 255.0f;
  const float factor_values[] = {0, 1, 0.5f, 0.5f, 0.5f, 0.5f, dest_component,
      1.0f - dest_component, dest_component, 1.0f - dest_component, 0.75f, 0.25f,
      0.75f, 0.25f, 0.5f};
  for (uint32_t i = 0; i < 15; ++i) {
    RegisterFile rf = Base();
    reg::RB_BLENDCONTROL blend{};
    blend.color_srcblend = factors[i];
    blend.color_destblend = F::kZero;
    blend.color_comb_fcn = xenos::BlendOp::kAdd;
    blend.alpha_srcblend = F::kOne;
    blend.alpha_destblend = F::kZero;
    blend.alpha_comb_fcn = xenos::BlendOp::kAdd;
    Set(rf, blend);
    for (uint32_t index : {XE_GPU_REG_RB_BLEND_RED, XE_GPU_REG_RB_BLEND_GREEN,
                           XE_GPU_REG_RB_BLEND_BLUE, XE_GPU_REG_RB_BLEND_ALPHA})
      SetF(rf, index, 0.75f);
    RopState state = DecodeRopState(rf);
    Edram edram;
    edram.WriteSample(state.color[0].surface, 0, 0, 0, 0x40404040u);
    PixelQuad q = Quad();
    PixelOut out[4] = {Output(0.5f, 0.5f, 0.5f, 0.5f), {}, {}, {}};
    DrawStats stats;
    RopQuad(state, edram, q, out, stats);
    uint32_t expected = uint32_t(0.5f * factor_values[i] * 255.0f + 0.5f);
    EQ(Pixel(edram, state) & 0xFFu, expected);
  }
  using Op = xenos::BlendOp;
  const Op ops[] = {Op::kAdd, Op::kSubtract, Op::kMin, Op::kMax, Op::kRevSubtract};
  const uint32_t expected[] = {255, 127, 64, 191, 0};
  for (uint32_t i = 0; i < 5; ++i) {
    RegisterFile rf = Base();
    reg::RB_BLENDCONTROL blend{};
    blend.color_srcblend = blend.alpha_srcblend = F::kOne;
    blend.color_destblend = blend.alpha_destblend = F::kOne;
    blend.color_comb_fcn = blend.alpha_comb_fcn = ops[i];
    Set(rf, blend);
    RopState state = DecodeRopState(rf);
    Edram edram;
    edram.WriteSample(state.color[0].surface, 0, 0, 0, 0x40404040u);
    PixelQuad q = Quad();
    PixelOut out[4] = {Output(0.75f, 0.75f, 0.75f, 0.75f), {}, {}, {}};
    DrawStats stats;
    RopQuad(state, edram, q, out, stats);
    EQ(Pixel(edram, state) & 0xFFu, expected[i]);
  }
}

static void TestDepthStencil() {
  RegisterFile rf = Base();
  reg::RB_DEPTHCONTROL depth{};
  depth.z_enable = 1;
  depth.z_write_enable = 1;
  depth.zfunc = xenos::CompareFunction::kLess;
  depth.stencil_enable = 1;
  depth.stencilfunc = xenos::CompareFunction::kAlways;
  depth.stencilzpass = xenos::StencilOp::kReplace;
  Set(rf, depth);
  reg::RB_STENCILREFMASK stencil{};
  stencil.stencilref = 0x42;
  stencil.stencilmask = 0xFF;
  stencil.stencilwritemask = 0xFF;
  Set(rf, stencil);
  RopState state = DecodeRopState(rf);
  Edram edram;
  edram.WriteSample(state.depth_surface, 0, 0, 0, 0x80000001u);
  PixelQuad q = Quad();
  PixelOut out[4] = {Output(1, 0, 0, 1), {}, {}, {}};
  DrawStats stats;
  q.sample_z[0][0] = 0.25;
  RopQuad(state, edram, q, out, stats);
  EQ(edram.ReadSample(state.depth_surface, 0, 0, 0), 0x40000042u);
  q.sample_z[0][0] = 0.75;
  RopQuad(state, edram, q, out, stats);
  EQ(edram.ReadSample(state.depth_surface, 0, 0, 0), 0x40000042u);

  depth.zfunc = xenos::CompareFunction::kAlways;
  depth.stencilfunc = xenos::CompareFunction::kNever;
  depth.stencilfail = xenos::StencilOp::kIncrementClamp;
  Set(rf, depth);
  state = DecodeRopState(rf);
  edram.WriteSample(state.depth_surface, 0, 0, 0, 0x111111FFu);
  RopQuad(state, edram, q, out, stats);
  EQ(edram.ReadSample(state.depth_surface, 0, 0, 0), 0x111111FFu);
  edram.WriteSample(state.depth_surface, 0, 0, 0, 0x11111100u);
  RopQuad(state, edram, q, out, stats);
  EQ(edram.ReadSample(state.depth_surface, 0, 0, 0), 0x11111101u);

  using S = xenos::StencilOp;
  const S ops[] = {S::kKeep, S::kZero, S::kReplace, S::kIncrementClamp,
                   S::kDecrementClamp, S::kInvert, S::kIncrementWrap, S::kDecrementWrap};
  const uint8_t expected[] = {0x80, 0x00, 0x42, 0x81, 0x7F, 0x7F, 0x81, 0x7F};
  for (uint32_t i = 0; i < 8; ++i) {
    rf = Base();
    depth = {};
    depth.stencil_enable = 1;
    depth.stencilfunc = xenos::CompareFunction::kAlways;
    depth.stencilzpass = ops[i];
    Set(rf, depth);
    Set(rf, stencil);
    state = DecodeRopState(rf);
    edram.WriteSample(state.depth_surface, 0, 0, 0, 0x80000080u);
    RopQuad(state, edram, q, out, stats);
    EQ(uint8_t(edram.ReadSample(state.depth_surface, 0, 0, 0)), expected[i]);
  }

  rf = Base();
  depth = {};
  depth.stencil_enable = 1;
  depth.backface_enable = 1;
  depth.stencilfunc = depth.stencilfunc_bf = xenos::CompareFunction::kAlways;
  depth.stencilzpass = S::kReplace;
  depth.stencilzpass_bf = S::kZero;
  Set(rf, depth);
  reg::RB_STENCILREFMASK back = stencil;
  back.stencilref = 0x99;
  rf[XE_GPU_REG_RB_STENCILREFMASK_BF] = back.value;
  Set(rf, stencil);
  state = DecodeRopState(rf);
  q.front_face = false;
  RopQuad(state, edram, q, out, stats);
  EQ(uint8_t(edram.ReadSample(state.depth_surface, 0, 0, 0)), 0u);
}

int main() {
  TestAlphaTest();
  TestAlphaToCoverage();
  TestBlendAndMask();
  TestAllBlendFactorsAndOps();
  TestDepthStencil();
  std::printf("rop: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
