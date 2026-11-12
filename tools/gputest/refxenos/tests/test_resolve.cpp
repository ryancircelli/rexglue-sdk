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
    // Guest memory stores the float in big-endian byte order. The GPU's
    // little-endian word load followed by the fetch's 8in32 swap restores it.
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  rf[XE_GPU_REG_RB_COPY_DEST_BASE] = 0x100000;
  return rf;
}

static EdramSurface Source(const RegisterFile& rf) {
  auto si = rf.Get<reg::RB_SURFACE_INFO>();
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
  CHECK(linear[2] > 0.504f && linear[2] < 0.505f);

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

static void FillColor(Edram& edram, const EdramSurface& surface, uint32_t sample,
                      uint64_t value) {
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) edram.WriteSample(surface, x, y, sample, value);
  }
}

static void TestFormatConversionAndEndian() {
  Memory memory;
  RegisterFile rf = Base(memory);
  Edram edram;
  EdramSurface source = Source(rf);
  FillColor(edram, source, 0, 0xFFC04080u);
  auto dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_6_5_5;
  dest.copy_dest_endian = xenos::Endian128::kNone;
  Set(rf, dest);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint8_t* p = memory.At(0x100000);
  EQ(p[0], 0x10u);
  EQ(p[1], 0x79u);

  const uint8_t expected32[6][4] = {
      {0x44, 0x33, 0x22, 0x11}, {0x33, 0x44, 0x11, 0x22},
      {0x11, 0x22, 0x33, 0x44}, {0x22, 0x11, 0x44, 0x33},
      {0x44, 0x33, 0x22, 0x11}, {0x44, 0x33, 0x22, 0x11}};
  for (uint32_t endian = 0; endian < 6; ++endian) {
    rf = Base(memory);
    dest = rf.Get<reg::RB_COPY_DEST_INFO>();
    dest.copy_dest_endian = xenos::Endian128(endian);
    Set(rf, dest);
    FillColor(edram, source, 0, 0x11223344u);
    CHECK(Resolve(rf, edram, memory, &record, stats));
    p = memory.At(0x100000);
    for (uint32_t i = 0; i < 4; ++i) EQ(p[i], expected32[endian][i]);
  }

  rf = Base(memory);
  dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_8;
  dest.copy_dest_endian = xenos::Endian128::k8in32;
  Set(rf, dest);
  FillColor(edram, source, 0, 0x00000080u);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(memory.At(0x100000)[0], 0x80u);

  rf = Base(memory);
  auto color = rf.Get<reg::RB_COLOR_INFO>();
  color.color_format = xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT;
  Set(rf, color);
  source = Source(rf);
  source.is_64bpp = true;
  source.pitch_tiles = EdramPitchTiles(32, xenos::MsaaSamples::k1X, true);
  const uint64_t extended = 0x7FFF7C007BFF3C00ull;
  FillColor(edram, source, 0, extended);
  dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_16_16_16_16_FLOAT;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kFloat;
  dest.copy_dest_endian = xenos::Endian128::kNone;
  Set(rf, dest);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint8_t expected64[8] = {0x00, 0x3C, 0xFF, 0x7B, 0x00, 0x7C, 0xFF, 0x7F};
  p = memory.At(0x100000);
  for (uint32_t i = 0; i < 8; ++i) EQ(p[i], expected64[i]);
  dest.copy_dest_endian = xenos::Endian128::k8in64;
  Set(rf, dest);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  p = memory.At(0x100000);
  for (uint32_t i = 0; i < 8; ++i) EQ(p[i], expected64[7 - i]);

  rf = Base(memory);
  color = rf.Get<reg::RB_COLOR_INFO>();
  color.color_format = xenos::ColorRenderTargetFormat::k_32_32_FLOAT;
  Set(rf, color);
  source = Source(rf);
  source.is_64bpp = true;
  source.pitch_tiles = EdramPitchTiles(32, xenos::MsaaSamples::k1X, true);
  FillColor(edram, source, 0,
            uint64_t(AsU32(2.0f)) << 32 | uint64_t(AsU32(1.0f)));
  dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_32_32_32_32_FLOAT;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kFloat;
  dest.copy_dest_endian = xenos::Endian128::k8in128;
  Set(rf, dest);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  const uint32_t words[4] = {AsU32(1.0f), AsU32(2.0f), 0, 0};
  p = memory.At(0x100000);
  for (uint32_t i = 0; i < 16; ++i) {
    uint32_t source_index = 15 - i;
    EQ(p[i], uint8_t(words[source_index / 4] >> (8 * (source_index & 3))));
  }
}

static void TestSampleSelectAndCombinedClear() {
  Memory memory;
  RegisterFile rf = Base(memory, xenos::MsaaSamples::k4X);
  auto dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_format = xenos::ColorFormat::k_32_32_32_32_FLOAT;
  dest.copy_dest_number = xenos::SurfaceNumberFormat::kFloat;
  dest.copy_dest_endian = xenos::Endian128::kNone;
  Set(rf, dest);
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t s = 0; s < 4; ++s) FillColor(edram, source, s, s * 64u);
  const float expected[] = {
      0.0f, 64.0f * (1.0f / 255.0f), 128.0f * (1.0f / 255.0f),
      192.0f * (1.0f / 255.0f),
      (0.0f + 64.0f * (1.0f / 255.0f)) * 0.5f,
      (128.0f * (1.0f / 255.0f) + 192.0f * (1.0f / 255.0f)) * 0.5f,
      ((0.0f + 64.0f * (1.0f / 255.0f)) + 128.0f * (1.0f / 255.0f) +
       192.0f * (1.0f / 255.0f)) * 0.25f};
  ResolveRecord record;
  DrawStats stats;
  for (uint32_t select = 0; select < 7; ++select) {
    auto control = rf.Get<reg::RB_COPY_CONTROL>();
    control.copy_sample_select = xenos::CopySampleSelect(select);
    Set(rf, control);
    CHECK(Resolve(rf, edram, memory, &record, stats));
    float value;
    std::memcpy(&value, memory.At(0x100000), 4);
    EQ(AsU32(value), AsU32(expected[select]));
    EQ(record.sample_select, select);
  }
  auto control = rf.Get<reg::RB_COPY_CONTROL>();
  control.copy_sample_select = xenos::CopySampleSelect(7);
  Set(rf, control);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(record.sample_select, uint32_t(xenos::CopySampleSelect::k0123));

  control.color_clear_enable = 1;
  control.depth_clear_enable = 1;
  Set(rf, control);
  rf[XE_GPU_REG_RB_COLOR_CLEAR] = 0x11223344;
  rf[XE_GPU_REG_RB_DEPTH_CLEAR] = 0xE000005A;
  EdramSurface depth;
  depth.is_depth = true;
  depth.msaa = xenos::MsaaSamples::k4X;
  depth.pitch_tiles = EdramPitchTiles(32, depth.msaa, false);
  CHECK(Resolve(rf, edram, memory, &record, stats));
  CHECK(record.cleared_color);
  CHECK(record.cleared_depth);
  for (uint32_t s = 0; s < 4; ++s) {
    EQ(edram.ReadSample(source, 7, 7, s), 0x11223344u);
    EQ(edram.ReadSample(depth, 7, 7, s), 0xE000005Au);
  }
}

// Regression tests for resolve vertex fetch byte order fix (commit 015dc27e).
// These tests exercise the exact bug scenario: GPU little-endian dword load
// followed by fetch 8in32 endian swap. The guest vertex buffer is stored in
// big-endian byte order (as the Xbox 360 CPU writes it), and the GPU's
// little-endian load + 8in32 swap must correctly reconstruct the float.

static void TestResolveVertexByteOrder_8in32_NoSwap() {
  // Test the core fix: 8in32 vertex fetch endian with no destination endian swap.
  // Vertex buffer at 0x1000, three float2 vertices forming a triangle at (10,20)-(30,20)-(10,40).
  // Rectangle expands to 8-pixel boundaries: x0=8, y0=16, x1=32, y1=32 -> w=24, h=16.
  Memory memory;
  RegisterFile rf = Base(memory);
  // Override destination to have no endian swap (kNone) to isolate vertex fetch.
  auto dest = rf.Get<reg::RB_COPY_DEST_INFO>();
  dest.copy_dest_endian = xenos::Endian128::kNone;
  Set(rf, dest);
  // Override vertex buffer with known coordinates (non-zero, non-trivial).
  const float vertices[6] = {10.0f, 20.0f, 30.0f, 20.0f, 10.0f, 40.0f};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  Edram edram;
  EdramSurface source = Source(rf);
  // Source color 0xFF0080FF is stored in EDRAM as A=0xFF, B=0x00, G=0x80, R=0xFF (ABGR order).
  // With kNone dest endian, output bytes are R,G,B,A = 0xFF, 0x80, 0x00, 0xFF.
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF0080FFu);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  // Rectangle after 8-pixel alignment: (8,16)-(32,32) -> width=24, height=16.
  EQ(record.x0, 8u);
  EQ(record.y0, 16u);
  EQ(record.width, 24u);
  EQ(record.height, 16u);
  // Check a pixel inside the rectangle (15, 25) has the source color.
  uint32_t address = 0x100000 + tex::TiledOffset2D(15, 25, 32, 4);
  const uint8_t* p = memory.At(address);
  // With kNone dest endian: R=0xFF, G=0x80, B=0x00, A=0xFF
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x80u);
  EQ(p[2], 0x00u);
  EQ(p[3], 0xFFu);
}

static void TestResolveVertexByteOrder_8in32_With8in32DestSwap() {
  // Test 8in32 vertex fetch with 8in32 destination endian (the common case).
  Memory memory;
  RegisterFile rf = Base(memory);
  const float vertices[6] = {10.0f, 20.0f, 30.0f, 20.0f, 10.0f, 40.0f};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF0080FFu);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(record.x0, 8u);
  EQ(record.y0, 16u);
  EQ(record.width, 24u);
  EQ(record.height, 16u);
  // With 8in32 dest endian, bytes within each 32-bit word are reversed: A,B,G,R.
  uint32_t address = 0x100000 + tex::TiledOffset2D(15, 25, 32, 4);
  const uint8_t* p = memory.At(address);
  EQ(p[0], 0xFFu);  // A
  EQ(p[1], 0x00u);  // B
  EQ(p[2], 0x80u);  // G
  EQ(p[3], 0xFFu);  // R
}

static void TestResolveVertexByteOrder_NonzeroDestBase() {
  // Test with nonzero destination base address (not 0x100000).
  // Vertices (5,5)-(15,5)-(5,15) -> rect (0,0)-(16,16) after 8-pixel align.
  Memory memory;
  RegisterFile rf = Base(memory);
  const float vertices[6] = {5.0f, 5.0f, 15.0f, 5.0f, 5.0f, 15.0f};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  rf[XE_GPU_REG_RB_COPY_DEST_BASE] = 0x200000;  // Nonzero dest base.
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 32; ++y)
    for (uint32_t x = 0; x < 32; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF884422u);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(record.x0, 0u);
  EQ(record.y0, 0u);
  EQ(record.width, 16u);
  EQ(record.height, 16u);
  uint32_t address = 0x200000 + tex::TiledOffset2D(7, 7, 32, 4);
  const uint8_t* p = memory.At(address);
  // Source 0xFF884422 = A=0xFF, B=0x88, G=0x44, R=0x22. k8in32 dest -> A,B,G,R.
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x88u);
  EQ(p[2], 0x44u);
  EQ(p[3], 0x22u);
}

static void TestResolveVertexByteOrder_WithScissor() {
  // Test with window scissor enabled (non-trivial scissor rectangle).
  // Triangle (0,0)-(32,0)-(0,32), window scissor (8,8)-(24,24).
  // Rect from vertices: (0,0)-(32,32). Clipped to scissor: (8,8)-(24,24).
  // 8-pixel align: (8,8)-(24,24) -> width=16, height=16.
  Memory memory;
  RegisterFile rf = Base(memory);
  const float vertices[6] = {0.0f, 0.0f, 32.0f, 0.0f, 0.0f, 32.0f};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  // Enable window scissor: clip to (8,8)-(24,24).
  auto wtl = rf.Get<reg::PA_SC_WINDOW_SCISSOR_TL>();
  wtl.window_offset_disable = 0;
  wtl.tl_x = 8;
  wtl.tl_y = 8;
  Set(rf, wtl);
  auto wbr = rf.Get<reg::PA_SC_WINDOW_SCISSOR_BR>();
  wbr.br_x = 24;
  wbr.br_y = 24;
  Set(rf, wbr);
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 32; ++y)
    for (uint32_t x = 0; x < 32; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF112233u);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(record.x0, 8u);
  EQ(record.y0, 8u);
  EQ(record.width, 16u);
  EQ(record.height, 16u);
  // Pixel at (10,10) should be inside scissor and triangle.
  // Source 0xFF112233 = A=0xFF, B=0x11, G=0x22, R=0x33. k8in32 dest -> A,B,G,R.
  uint32_t address = 0x100000 + tex::TiledOffset2D(10, 10, 32, 4);
  const uint8_t* p = memory.At(address);
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x11u);
  EQ(p[2], 0x22u);
  EQ(p[3], 0x33u);
  // Pixel at (5,5) should be outside scissor (not written).
  address = 0x100000 + tex::TiledOffset2D(5, 5, 32, 4);
  p = memory.At(address);
  EQ(p[0], 0x00u);
  EQ(p[1], 0x00u);
  EQ(p[2], 0x00u);
  EQ(p[3], 0x00u);
}

static void TestResolveVertexByteOrder_NegativeCoordinates() {
  // Test with negative vertex coordinates (should be clamped to 0).
  // Vertices (-10,-5),(10,-5),(-10,10) -> rect (0,0)-(16,16) after 8-pixel align.
  Memory memory;
  RegisterFile rf = Base(memory);
  const float vertices[6] = {-10.0f, -5.0f, 10.0f, -5.0f, -10.0f, 10.0f};
  for (uint32_t i = 0; i < 6; ++i) {
    WriteBE32(memory, 0x1000 + 4 * i, AsU32(vertices[i]));
  }
  Edram edram;
  EdramSurface source = Source(rf);
  for (uint32_t y = 0; y < 32; ++y)
    for (uint32_t x = 0; x < 32; ++x)
      edram.WriteSample(source, x, y, 0, 0xFF556677u);
  ResolveRecord record;
  DrawStats stats;
  CHECK(Resolve(rf, edram, memory, &record, stats));
  EQ(record.x0, 0u);
  EQ(record.y0, 0u);
  EQ(record.width, 16u);  // 8-pixel aligned from 0 to 10 -> 16
  EQ(record.height, 16u); // 8-pixel aligned from 0 to 10 -> 16
  uint32_t address = 0x100000 + tex::TiledOffset2D(5, 5, 32, 4);
  const uint8_t* p = memory.At(address);
  // Source 0xFF556677 = A=0xFF, B=0x55, G=0x66, R=0x77. k8in32 dest -> A,B,G,R.
  EQ(p[0], 0xFFu);
  EQ(p[1], 0x55u);
  EQ(p[2], 0x66u);
  EQ(p[3], 0x77u);
}

int main() {
  Test8888Resolve();
  TestAverageAndClear();
  TestFloatReadbackResolve();
  TestGammaExpAndDepth();
  TestFormatConversionAndEndian();
  TestSampleSelectAndCombinedClear();
  // Regression tests for vertex fetch byte order fix.
  TestResolveVertexByteOrder_8in32_NoSwap();
  TestResolveVertexByteOrder_8in32_With8in32DestSwap();
  TestResolveVertexByteOrder_NonzeroDestBase();
  TestResolveVertexByteOrder_WithScissor();
  TestResolveVertexByteOrder_NegativeCoordinates();
  std::printf("resolve: %d checks, %d failures\n", checks, failures);
  return failures != 0;
}
