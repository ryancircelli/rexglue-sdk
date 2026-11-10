// refxenos EDRAM model.
//
// ===========================================================================
// SAMPLE INDEX CONVENTION (shared by the rasterizer, the ROP and resolves)
// ===========================================================================
//
// A sample index s used anywhere in refxenos (PixelQuad::coverage bit s,
// PixelQuad::sample_z[pixel][s], Edram addressing, resolve sample select) is
// the Xenos GUEST sample index, exactly as Xenia's guest-facing code uses it:
//
//   1x : s = 0 only, at the pixel center (0, 0).
//   2x : s = 0 at (-4, +4)/16 px, s = 1 at (+4, -4)/16 px.
//   4x : s = 0 at (-2, -2)/16, s = 1 at (+2, +2)/16, s = 2 at (-6, +6)/16,
//        s = 3 at (+6, -6)/16.
//   Offsets are relative to the pixel center, +x right, +y down.
//   Source: xenia-canary@cf206a3b src/xenia/gpu/draw_util.h:150-152
//   (kXenosSamplePositions2x / kXenosSamplePositions4x, R6xx positions).
//
// Location of sample s of pixel (x, y) inside the EDRAM surface, in "sample
// columns/rows" (u, v) of the equivalent single-sampled view
// (xenia-canary@cf206a3b src/xenia/gpu/shaders/edram.xesli:33-38):
//   4x: u = ((x & ~1) << 1) | (x & 1) | ((s & 1) << 1)
//       v = ((y & ~1) << 1) | (y & 1) | (s & 2)
//   2x: u = (x & ~2) | ((s & 1) << 1)
//       v = ((y & ~1) << 1) | (y & 1) | (x & 2)
//   1x: u = x, v = y
// So for 4x, bit 0 of s steps +2 sample columns and bit 1 steps +2 sample
// rows; for 2x, sample 1 is +2 sample columns from sample 0. NOTE: the storage
// position of a sample is NOT its geometric position (4x sample 1 is stored
// to the right of sample 0 but sits below-right geometrically). Nothing in
// refxenos converts host sample orders: Xenia's host remaps (e.g. 0,3,2,1 for
// interlock) only translate host samples back to these guest indices.
//
// Resolve sample select (RB_COPY_CONTROL copy_sample_select k0..k3) names
// these same guest indices.
// ===========================================================================
//
// Layout: 2048 tiles of 5120 bytes. A tile holds 80x16 32bpp samples or 40x16
// 64bpp samples, row major. Surfaces start at a tile index (RB_COLOR_INFO
// color_base / RB_DEPTH_INFO depth_base) and are pitch_tiles tiles wide. Depth
// surfaces swap the two 40-sample halves of each tile row. Addressing wraps
// modulo the EDRAM size. Words are stored as host uint32 (EDRAM has no byte
// order visible to the guest except through resolves).

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common.h"

namespace refxenos {

constexpr uint32_t kEdramTiles = 2048;
constexpr uint32_t kEdramTileDwords = 80 * 16;
constexpr uint32_t kEdramDwords = kEdramTiles * kEdramTileDwords;  // 10 MiB / 4

// Dword index (0 .. kEdramDwords-1) of sample `sample` of pixel (x, y) of a
// surface. For 64bpp surfaces this is the index of the low dword; the high
// dword is at the returned index + 1.
//   base_tile    RB_COLOR_INFO.color_base / RB_DEPTH_INFO.depth_base
//   pitch_tiles  xenos::GetSurfacePitchTiles(surface_pitch, msaa, is_64bpp)
//   is_64bpp     16_16_16_16, 16_16_16_16_FLOAT, 32_32_FLOAT color formats
//   is_depth     depth/stencil surface (half-tile swap)
//   msaa         RB_SURFACE_INFO.msaa_samples
uint32_t EdramOffset(uint32_t base_tile, uint32_t pitch_tiles, bool is_64bpp, bool is_depth,
                     xenos::MsaaSamples msaa, uint32_t x, uint32_t y, uint32_t sample);

// Pitch in tiles for a surface (32bpp: depth or 32bpp color; 64bpp color).
uint32_t EdramPitchTiles(uint32_t surface_pitch_pixels, xenos::MsaaSamples msaa,
                         bool is_64bpp);

bool IsColorFormat64bpp(xenos::ColorRenderTargetFormat format);

// A view of a surface placed in EDRAM.
struct EdramSurface {
  uint32_t base_tile = 0;
  uint32_t pitch_tiles = 0;
  xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X;
  bool is_64bpp = false;
  bool is_depth = false;

  uint32_t Offset(uint32_t x, uint32_t y, uint32_t sample) const {
    return EdramOffset(base_tile, pitch_tiles, is_64bpp, is_depth, msaa, x, y, sample);
  }
};

class Edram {
 public:
  Edram() : data_(kEdramDwords, 0) {}

  uint32_t dword(uint32_t index) const { return data_[index % kEdramDwords]; }
  void set_dword(uint32_t index, uint32_t value) { data_[index % kEdramDwords] = value; }
  uint32_t* data() { return data_.data(); }
  const uint32_t* data() const { return data_.data(); }
  void Fill(uint32_t value) { std::fill(data_.begin(), data_.end(), value); }

  // Sample access. For 32bpp surfaces only the low 32 bits are used.
  uint64_t ReadSample(const EdramSurface& s, uint32_t x, uint32_t y, uint32_t sample) const;
  void WriteSample(const EdramSurface& s, uint32_t x, uint32_t y, uint32_t sample,
                   uint64_t value);

 private:
  std::vector<uint32_t> data_;
};

// Test / tool helpers.
// Raw dump of `rows` tile rows of `tiles_per_row` tiles starting at
// base_tile, as [row][tile][16][80] u32 (the physical EDRAM order).
std::vector<uint32_t> ExtractTiles(const Edram& edram, uint32_t base_tile, uint32_t tiles_per_row,
                                   uint32_t rows);

}  // namespace refxenos
