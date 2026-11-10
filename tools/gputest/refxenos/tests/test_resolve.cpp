// refxenos resolve known-answer tests.
#include <cstdio>
#include <cstring>

#include "core/resolve.h"
#include "texture/ref_texture.h"

using namespace refxenos;
using namespace rex::graphics;  // XE_GPU_REG_* register indices
static int failures = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define EQ(a, b) do { ++checks; auto a_ = (a); auto b_ = (b); if (!(a_ == b_)) { \
  std::printf("FAIL %d: %s = 0x%llx, expected 0x%llx\n", __LINE__, #a, \
              (unsigned long long)a_, (unsigned long long)b_); ++failures; } } while (0)

template <typename R>
static void Set(RegisterFile& rf, R r) { rf[R::register_index] = r.value; }
static void WriteBE32(Memory& memory, uint32_t address, uint32_t value) {
  uint8_t bytes[4] = {uint8_t(value >> 24), uint8_t(value >> 16), uint8_t(value >> 8), uint8_t(value)};
  memory.Write(address, bytes, 4);
}

static RegisterFile Base(Memory& memory, xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X) {
  RegisterFile rf;
  std::memset(rf.values, 0, sizeof(rf.values));
  reg::RB_SURFACE_INFO surface{};
  surface.surface_pitch = 32;
  surface.msaa_samples = msaa;
  Set(rf, surface);
  reg::RB_COLOR_INFO color{};
  color.color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
  Set(rf, color);
  reg::RB_COPY_CONTROL control{};
  control.copy_command = xenos::CopyCommand::kConvert;
  control.copy_sample_select = xenos::CopySampleSelect::k0;
  Set(rf, control);
  reg::RB_COPY_DEST_INFO dest{};
  dest.copy_dest_format = xenos::ColorFormat::k_8_8_8_8;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction;
  dest.copy_dest_endian = xenos::Endian128::k8in32;
  Set(rf, dest);
  reg::RB_COPY_DEST_PITCH pitch{};
  pitch.copy_dest_pitch = 32;
  pitch.copy_dest_height = 32;
  Set(rf, pitch);
  reg::PA_SC_WINDOW_SCISSOR_TL wtl{};
  wtl.window_offset_disable = 1;
  Set(rf, wtl);
  reg::PA_SC_WINDOW_SCISSOR_BR wbr{};
  wbr.br_x = wbr.br_y = 32;
  Set(rf, wbr);
  reg::PA_SC_SCREEN_SCISSOR_BR sbr{};
  sbr.br_x = sbr.br_y = 32;
  Set(rf, sbr);
  auto fetch = rf.GetVertexFetch(0);
  fetch.type = xenos::FetchConstantType::kVertex;
  fetch.address = 0x1000 / 4;
  fetch.size = 6;
  fetch.endian = xenos::Endian::k8in32;
  std::memcpy(&rf.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], &fetch, sizeof(fetch));
  const float vertices[6] = {0, 0, 8, 0, 0, 8};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, xenos::GpuSwap(AsU32(vertices[i]), fetch.endian));
  }
  rf[XE_GPU_REG_RB_COPY_DEST_BASE] = 0x100000;
  return rf;
}

static EdramSurface Source(const RegisterFile& rf) {
  auto si = rf.Get<reg::RB_SURFACE_INFO>();
  auto ci = rf.Get<reg::RB_COLOR_INFO>();
  EdramSurface s;
  s.pitch_tiles = EdramPitchTiles(si.surface_pitch, si.msaa_samples, false);
  s.msaa = si.msaa_samples;
  return s;
}

static void Test8888Resolve() {
  Memory memory;
  RegisterFile rf = Base(memory);
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF0080FFu);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  uint32_t address = 0x100000 + tex::TiledOffset2D(1, 0, 32, 4);
  const uint8_t* p = memory.At(address);
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x00u);
  EQ(p[2], 0x80u);
  EQ(p[3], 0xFFu);
  EQ(record.width, 8u);
  EQ(record.height, 8u);
  CHECK(record.dest_bytes >= 8 * 8 * 4);
  auto readback = ReadbackColorF32(edram, rf, 0, 1, 1);
  EQ(AsU32(readback[0]), AsU32(1.0f));
  EQ(AsU32(readback[1]), AsU32(128.0f / 255.0f));
  EQ(AsU32(readback[2]), AsU32(0.0f));
  EQ(AsU32(readback[3]), AsU32(1.0f));
}

static void TestAverageAndClear() {
  Memory memory;
  RegisterFile rf = Base(memory, xenos::MsaaSamples::k2X);
  auto cc = rf.Get<reg::RB_COPY_CONTROL>();
  cc.copy_sample_select = xenos::CopySampleSelect::k01;
  cc.color_clear_enable = 1;
  Set(rf, cc);
  rf[XE_GPU_REG_RB_COLOR_CLEAR] = 0x11223344;
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      edram.WriteSample(source, x, y, 0, 0xFF000000u);
      edram.WriteSample(source, x, y, 1, 0xFF0000FFu);
    }
  }
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint8_t* p = memory.At(0x100000);
  EQ(p[3], 128u);
  CHECK(record.cleared_color);
  EQ(edram.ReadSample(source, 0, 0, 0), 0x11223344u);
  EQ(edram.ReadSample(source, 7, 7, 1), 0x11223344u);
}

static void TestFloatReadbackResolve() {
  Memory memory;
  RegisterFile rf = Base(memory);
  auto dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_32_32_32_32_FLOAT;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kFloat;
  dest.copy_dest_endian = xenos::Endian128::kNone;
  Set(rf, dest);
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF0080FFu);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint8_t* p = memory.At(0x100000);
  float value;
  std::memcpy(&value, p, 4);
  EQ(AsU32(value), AsU32(1.0f));
  std::memcpy(&value, p + 4, 4);
  EQ(AsU32(value), AsU32(128.0f / 255.0f));
}

static void TestGammaExpAndDepth() {
  Memory memory;
  RegisterFile rf = Base(memory);
  auto color = rf.Get<reg::RB_COLOR_INFO>();
  color.color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  Set(rf, color);
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x)
      edram.WriteSample(source, x, y, 0, 0xFFC04080u);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint8_t* p = memory.At(0x100000);
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x81u);
  EQ(p[2], 0x10u);
  EQ(p[3], 0x40u);
  auto linear = ReadbackColorF32(edram, rf, 0, 1, 1);
  CHECK(linear[0] > 0.25f && linear[0] < 0.26f);
  CHECK(linear[1] > 0.062f && linear[1] < 0.063f);
  CHECK(linear[2] > 0.505f && linear[2] < 0.507f);

  rf = Base(memory);
  auto dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_32_32_32_32_FLOAT;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kFloat;
  dest.copy_dest_endian = xenos::Endian128::kNone;
  dest.copy_dest_exp_bias = -1;
  Set(rf, dest);
  edram.Fill(0);
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF0080FFu);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  float value;
  std::memcpy(&value, memory.At(0x100000), 4);
  EQ(AsU32(value), AsU32(0.5f));
  std::memcpy(&value, memory.At(0x100004), 4);
  EQ(AsU32(value), AsU32(64.0f / 255.0f));

  rf = Base(memory);
  auto control = rf.Get<reg::RB_COPY_CONTROL>();
  control.copy_src_select = 4;
  control.copy_sample_select = xenos::CopySampleSelect::k0123;
  Set(rf, control);
  auto depth = rf.Get<reg::RB_DEPTH_INFO>();
  depth.depth_format = xenos::DepthRenderTargetFormat::kD24FS8;
  Set(rf, depth);
  EdramSurface depth_surface;
  depth_surface.is_depth = true;
  depth_surface.pitch_tiles = 1;
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x)
      edram.WriteSample(depth_surface, x, y, 0, 0x600000A5u);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  p = memory.At(0x100000);
  EQ(p[0], 0x60u);
  EQ(p[3], 0xA5u);
  EQ(record.dest_format, uint32_t(xenos::TextureFormat::k_24_8_FLOAT));
  EQ(record.sample_select, uint32_t(xenos::CopySampleSelect::k0));
}

int main() {
  Test8888Resolve();
  TestAverageAndClear();
  TestFloatReadbackResolve();
  TestGammaExpAndDepth();
  std::printf("resolve: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
