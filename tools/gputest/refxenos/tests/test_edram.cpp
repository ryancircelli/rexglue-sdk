// refxenos EDRAM addressing and format conversion known-answer tests.
#include <cmath>
#include <cstdio>
#include <cstdint>

#include "core/edram.h"

using namespace refxenos;
static int failures = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define EQ(a, b) do { ++checks; auto a_ = (a); auto b_ = (b); if (!(a_ == b_)) { \
  std::printf("FAIL %d: %s = 0x%llx, expected 0x%llx\n", __LINE__, #a, \
              (unsigned long long)a_, (unsigned long long)b_); ++failures; } } while (0)

static void TestAddressing() {
  EQ(EdramPitchTiles(80, xenos::MsaaSamples::k1X, false), 1u);
  EQ(EdramPitchTiles(81, xenos::MsaaSamples::k1X, false), 2u);
  EQ(EdramPitchTiles(80, xenos::MsaaSamples::k4X, false), 2u);
  EQ(EdramPitchTiles(40, xenos::MsaaSamples::k1X, true), 2u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k1X, 0, 0, 0), 0u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k1X, 79, 15, 0), 1279u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k1X, 80, 0, 0), 1280u);
  EQ(EdramOffset(0, 2, true, false, xenos::MsaaSamples::k1X, 1, 0, 0), 2u);
  EQ(EdramOffset(0, 2, true, false, xenos::MsaaSamples::k1X, 40, 0, 0), 1280u);
  EQ(EdramOffset(0, 1, false, true, xenos::MsaaSamples::k1X, 0, 0, 0), 40u);
  EQ(EdramOffset(0, 1, false, true, xenos::MsaaSamples::k1X, 40, 0, 0), 0u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k4X, 0, 0, 1), 2u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k4X, 0, 0, 2), 160u);
  EQ(EdramOffset(0, 1, false, false, xenos::MsaaSamples::k4X, 0, 0, 3), 162u);
  EQ(EdramOffset(2047, 1, false, false, xenos::MsaaSamples::k1X, 80, 0, 0), 0u);
  Edram edram;
  EdramSurface s;
  s.pitch_tiles = 2;
  s.is_64bpp = true;
  edram.WriteSample(s, 1, 0, 0, 0x1122334455667788ull);
  EQ(edram.ReadSample(s, 1, 0, 0), 0x1122334455667788ull);
  EQ(edram.dword(2), 0x55667788u);
  EQ(edram.dword(3), 0x11223344u);
}

static void TestColorFormats() {
  using F = xenos::ColorRenderTargetFormat;
  const float a[4] = {1.0f, 0.5f, 0.0f, 1.0f};
  EQ(PackColor(F::k_8_8_8_8, a), 0xFF0080FFu);
  const float gamma[4] = {64.0f / 1023.0f, 128.0f / 1023.0f, 512.0f / 1023.0f, 1.0f};
  EQ(PackColor(F::k_8_8_8_8_GAMMA, gamma), 0xFFC06040u);
  float out[4];
  UnpackColor(F::k_8_8_8_8, 0xFF0080FFu, out);
  EQ(out[0], 1.0f);
  EQ(out[1], 128.0f / 255.0f);
  EQ(out[2], 0.0f);
  EQ(out[3], 1.0f);
  const float b[4] = {1.0f, 0.5f, 1.0f, 2.0f / 3.0f};
  EQ(PackColor(F::k_2_10_10_10, b), 0xBFF803FFu);
  EQ(Float32To7e3(1.0f), 0x180u);
  EQ(Float32To7e3(31.875f), 0x3FFu);
  EQ(Float32To7e3(-1.0f), 0u);
  EQ(Float32To7e3(std::nanf("")), 0u);
  const float fixed[4] = {-32.0f, 32767.0f / 1024.0f, -0.5f / 1024.0f, -1.5f / 1024.0f};
  EQ(PackColor(F::k_16_16_16_16, fixed), 0xFFFF00007FFF8000ull);
  EQ(Float32To16Extended(1.0f), 0x3C00u);
  EQ(Float32To16Extended(65504.0f), 0x7BFFu);
  EQ(Float32To16Extended(131008.0f), 0x7FFFu);
  EQ(Float32To16Extended(std::nanf("")), 0u);
  EQ(AsU32(Float16ExtendedTo32(0x7C00)), AsU32(65536.0f));
  EQ(AsU32(Float16ExtendedTo32(1)), AsU32(std::ldexp(1.0f, -24)));
  EQ(Float32To16Extended(1.0f + std::ldexp(1.0f, -11)), 0x3C00u);
  EQ(Float32To16Extended(1.0f + 3.0f * std::ldexp(1.0f, -11)), 0x3C02u);
  EQ(Float32To16Extended(-0.0f), 0x8000u);
  for (uint32_t h = 0; h <= 0xFFFFu; ++h) {
    EQ(Float32To16Extended(Float16ExtendedTo32(uint16_t(h))), h);
  }
  for (uint32_t f = 0; f < 1024; ++f) {
    EQ(Float32To7e3(xenos::Float7e3To32(f)), f);
  }
  const float nan = std::nanf("");
  const float nans[4] = {nan, nan, nan, nan};
  EQ(PackColor(F::k_8_8_8_8, nans), 0u);
  EQ(PackColor(F::k_16_16, nans), 0u);
  const float f32[4] = {1.25f, -2.5f, 0, 0};
  uint64_t p32 = PackColor(F::k_32_32_FLOAT, f32);
  UnpackColor(F::k_32_32_FLOAT, p32, out);
  EQ(AsU32(out[0]), AsU32(1.25f));
  EQ(AsU32(out[1]), AsU32(-2.5f));
  EQ(AsU32(out[3]), AsU32(1.0f));
}

static void TestDepth() {
  using D = xenos::DepthRenderTargetFormat;
  EQ(PackDepthStencil(D::kD24S8, 0.5f, 0xA5), 0x800000A5u);
  EQ(PackDepthStencil(D::kD24S8, 1.0f, 0xFF), 0xFFFFFFFFu);
  EQ(PackDepthStencil(D::kD24S8, -1.0f, 0), 0u);
  EQ(PackDepthStencil(D::kD24S8, std::nanf(""), 0), 0u);
  EQ(UnpackDepth(D::kD24S8, 0xFFFFFF00u), 1.0f);
  EQ(PackDepthStencil(D::kD24FS8, 0.5f, 0x5A), 0xE000005Au);
  EQ(UnpackDepth(D::kD24FS8, 0xE0000000u), 0.5f);
  EQ(PackDepthStencil(D::kD24FS8, std::nanf(""), 0), 0u);
}

int main() {
  TestAddressing();
  TestColorFormats();
  TestDepth();
  std::printf("edram: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
