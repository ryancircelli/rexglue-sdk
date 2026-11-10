// refxenos EDRAM model. See edram.h for the layout and sample convention.

#include "core/edram.h"

#include <algorithm>

namespace refxenos {

RX_DEFINE_RULE(edram_sample_layout, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:33-38,62-71", false,
               "");
RX_DEFINE_RULE(edram_tile_32bpp_80x16, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:76-81; "
               "include/rex/graphics/xenos.h:411-415 (kEdramTileWidthSamples 80, height 16, "
               "2048 tiles)",
               false, "");
RX_DEFINE_RULE(
    edram_tile_64bpp_40x16, 1,
    "assumed: xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:73-75 admits "
    "\"while the actual storage of 64bpp render targets in comparison to 32bpp is not "
    "known, storing 40x16 64bpp samples per tile for simplicity\"",
    true,
    "Clear a 1x 64bpp 16_16_16_16 RT (base tile 0, pitch 80 px) to 0, draw a single pixel "
    "at (x, 0) for x in {0, 1, 39, 40} with oC0 = (1,2,3,4)/1024 (packed words 0x00020001, "
    "0x00040003); then switch to a 1x 32bpp 8888 RT at base tile 0 / pitch 80 px with "
    "edram_mode kCopy and resolve the 80x16 rect raw (k_8_8_8_8, no swap) to memory. Read "
    "back the 80x16 u32 grid and record at which (u, v) the two words of each pixel land. "
    "Expected under this rule: pixel x -> u = 2x, 2x+1 of tile x/40 row 0.");
RX_DEFINE_RULE(edram_depth_half_tile_swap, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:82-88", false, "");
RX_DEFINE_RULE(edram_address_wrap, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:93-96 (EDRAM "
               "addressing is periodic modulo the EDRAM size)",
               false, "");
RX_DEFINE_RULE(edram_pitch_tiles, 1,
               "include/rex/graphics/xenos.h:423-431 GetSurfacePitchTiles (pitch in samples "
               "rounded up to 80, doubled for 64bpp); mirrors xenia-canary@cf206a3b "
               "src/xenia/gpu/xenos.h GetSurfacePitchTiles",
               false, "");

bool IsColorFormat64bpp(xenos::ColorRenderTargetFormat format) {
  return format == xenos::ColorRenderTargetFormat::k_16_16_16_16 ||
         format == xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT ||
         format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT;
}

uint32_t EdramPitchTiles(uint32_t surface_pitch_pixels, xenos::MsaaSamples msaa, bool is_64bpp) {
  RX_USE(edram_pitch_tiles);
  return xenos::GetSurfacePitchTiles(surface_pitch_pixels, msaa, is_64bpp);
}

uint32_t EdramOffset(uint32_t base_tile, uint32_t pitch_tiles, bool is_64bpp, bool is_depth,
                     xenos::MsaaSamples msaa, uint32_t x, uint32_t y, uint32_t sample) {
  RX_USE(edram_sample_layout);
  uint32_t u, v;
  if (msaa >= xenos::MsaaSamples::k4X) {
    u = ((x & ~1u) << 1) | (x & 1u) | ((sample & 1u) << 1);
    v = ((y & ~1u) << 1) | (y & 1u) | (sample & 2u);
  } else if (msaa == xenos::MsaaSamples::k2X) {
    u = (x & ~2u) | ((sample & 1u) << 1);
    v = ((y & ~1u) << 1) | (y & 1u) | (x & 2u);
  } else {
    u = x;
    v = y;
  }
  uint32_t ints_log2 = is_64bpp ? 1 : 0;
  RX_USE(edram_tile_32bpp_80x16);
  if (is_64bpp) {
    RX_USE(edram_tile_64bpp_40x16);
  }
  uint32_t tile_w = 80u >> ints_log2;
  uint32_t tile_h = 16u;
  uint32_t tx = u / tile_w, ty = v / tile_h;
  uint32_t tiles = base_tile + ty * pitch_tiles + tx;
  u -= tx * tile_w;
  v -= ty * tile_h;
  if (is_depth) {
    RX_USE(edram_depth_half_tile_swap);
    uint32_t half = tile_w >> 1;
    u = (u >= half) ? u - half : u + half;
  }
  uint64_t address = uint64_t(tiles) * kEdramTileDwords + (uint64_t(v * tile_w + u) << ints_log2);
  RX_USE(edram_address_wrap);
  return uint32_t(address % kEdramDwords);
}

uint64_t Edram::ReadSample(const EdramSurface& s, uint32_t x, uint32_t y, uint32_t sample) const {
  uint32_t o = s.Offset(x, y, sample);
  uint64_t lo = dword(o);
  if (!s.is_64bpp) return lo;
  return lo | (uint64_t(dword(o + 1)) << 32);
}

void Edram::WriteSample(const EdramSurface& s, uint32_t x, uint32_t y, uint32_t sample,
                        uint64_t value) {
  uint32_t o = s.Offset(x, y, sample);
  set_dword(o, uint32_t(value));
  if (s.is_64bpp) set_dword(o + 1, uint32_t(value >> 32));
}

std::vector<uint32_t> ExtractTiles(const Edram& edram, uint32_t base_tile, uint32_t tiles_per_row,
                                   uint32_t rows) {
  std::vector<uint32_t> out;
  out.reserve(size_t(rows) * tiles_per_row * kEdramTileDwords);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t t = 0; t < tiles_per_row; ++t) {
      uint32_t tile = (base_tile + r * tiles_per_row + t) % kEdramTiles;
      for (uint32_t i = 0; i < kEdramTileDwords; ++i) {
        out.push_back(edram.dword(tile * kEdramTileDwords + i));
      }
    }
  }
  return out;
}

}  // namespace refxenos
