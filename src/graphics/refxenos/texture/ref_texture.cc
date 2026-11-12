#include "ref_texture.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ref_rules.h"

namespace refxenos::tex {

namespace {

uint32_t Bits(uint32_t v, uint32_t lo, uint32_t n) {
  return (v >> lo) & ((n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1));
}
int32_t SBits(uint32_t v, uint32_t lo, uint32_t n) {
  uint32_t u = Bits(v, lo, n);
  if (u & (1u << (n - 1))) return int32_t(u) - int32_t(1u << n);
  return int32_t(u);
}
uint32_t Log2Ceil(uint32_t v) {
  uint32_t r = 0;
  while ((1u << r) < v) ++r;
  return r;
}
uint32_t Log2Floor(uint32_t v) {
  uint32_t r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}
uint32_t NextPow2(uint32_t v) { return 1u << Log2Ceil(v); }
uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
uint32_t DivUp(uint32_t v, uint32_t d) { return (v + d - 1) / d; }

}  // namespace

void Diag::Use(const char* rule_id) {
  if (std::find(rules_used.begin(), rules_used.end(), rule_id) ==
      rules_used.end()) {
    rules_used.emplace_back(rule_id);
    std::sort(rules_used.begin(), rules_used.end());
  }
}

// ---------------------------------------------------------------------------
// Fetch constant (rule fetch.fields).

Fetch Fetch::Parse(const uint32_t d[6]) {
  Fetch f;
  std::memcpy(f.dw, d, sizeof(f.dw));
  f.type = Bits(d[0], 0, 2);
  for (int i = 0; i < 4; ++i) f.sign[i] = Bits(d[0], 2 + 2 * i, 2);
  for (int i = 0; i < 3; ++i) f.clamp[i] = Bits(d[0], 10 + 3 * i, 3);
  f.pitch_div32 = Bits(d[0], 22, 9);
  f.tiled = Bits(d[0], 31, 1);
  f.format = Bits(d[1], 0, 6);
  f.endian = Bits(d[1], 6, 2);
  f.request_size = Bits(d[1], 8, 2);
  f.stacked = Bits(d[1], 10, 1);
  f.clamp_policy = Bits(d[1], 11, 1);
  f.base_page = Bits(d[1], 12, 20);
  f.num_format_integer = Bits(d[3], 0, 1);
  for (int i = 0; i < 4; ++i) f.swizzle[i] = Bits(d[3], 1 + 3 * i, 3);
  f.exp_adjust = SBits(d[3], 13, 6);
  f.mag_filter = Bits(d[3], 19, 2);
  f.min_filter = Bits(d[3], 21, 2);
  f.mip_filter = Bits(d[3], 23, 2);
  f.aniso_filter = Bits(d[3], 25, 3);
  f.border_size = Bits(d[3], 31, 1);
  f.vol_mag_filter = Bits(d[4], 0, 1);
  f.vol_min_filter = Bits(d[4], 1, 1);
  f.min_mip_level = Bits(d[4], 2, 4);
  f.max_mip_level = Bits(d[4], 6, 4);
  f.mag_aniso_walk = Bits(d[4], 10, 1);
  f.min_aniso_walk = Bits(d[4], 11, 1);
  f.lod_bias = SBits(d[4], 12, 10);
  f.grad_exp_adjust_h = SBits(d[4], 22, 5);
  f.grad_exp_adjust_v = SBits(d[4], 27, 5);
  f.border_color = Bits(d[5], 0, 2);
  f.force_bcw_to_max = Bits(d[5], 2, 1);
  f.tri_clamp = Bits(d[5], 3, 2);
  f.aniso_bias = SBits(d[5], 5, 4);
  f.dimension = Bits(d[5], 9, 2);
  f.packed_mips = Bits(d[5], 11, 1);
  f.mip_page = Bits(d[5], 12, 20);
  // GPUTEXTURESIZE_* (dword 2): sizes are stored minus one.
  switch (f.dimension) {
    case kDim1D:
      f.width = Bits(d[2], 0, 24) + 1;
      f.height = 1;
      f.depth = 1;
      break;
    case kDim3D:
      f.width = Bits(d[2], 0, 11) + 1;
      f.height = Bits(d[2], 11, 11) + 1;
      f.depth = Bits(d[2], 22, 10) + 1;
      break;
    default:  // 2D / stacked / cube
      f.width = Bits(d[2], 0, 13) + 1;
      f.height = Bits(d[2], 13, 13) + 1;
      f.depth = Bits(d[2], 26, 6) + 1;
      break;
  }
  return f;
}

uint32_t Fetch::layers() const {
  if (dimension == kDimCube) return 6;
  if (dimension == kDim2D && stacked) return depth;
  return 1;
}

// ---------------------------------------------------------------------------
// Formats.

const FormatInfo& GetFormatInfo(uint32_t format) {
  static const FormatInfo kInfo[64] = {
      {"1_REVERSE", 1, 1, 1, true},
      {"1", 1, 1, 1, false},
      {"8", 1, 1, 8, true},
      {"1_5_5_5", 1, 1, 16, true},
      {"5_6_5", 1, 1, 16, true},
      {"6_5_5", 1, 1, 16, true},
      {"8_8_8_8", 1, 1, 32, true},
      {"2_10_10_10", 1, 1, 32, true},
      {"8_A", 1, 1, 8, true},
      {"8_B", 1, 1, 8, true},
      {"8_8", 1, 1, 16, true},
      {"Cr_Y1_Cb_Y0_REP", 2, 1, 32, false},
      {"Y1_Cr_Y0_Cb_REP", 2, 1, 32, false},
      {"16_16_EDRAM", 1, 1, 32, false},
      {"8_8_8_8_A", 1, 1, 32, true},
      {"4_4_4_4", 1, 1, 16, true},
      {"10_11_11", 1, 1, 32, true},
      {"11_11_10", 1, 1, 32, true},
      {"DXT1", 4, 4, 64, true},
      {"DXT2_3", 4, 4, 128, true},
      {"DXT4_5", 4, 4, 128, true},
      {"16_16_16_16_EDRAM", 1, 1, 64, false},
      {"24_8", 1, 1, 32, true},
      {"24_8_FLOAT", 1, 1, 32, true},
      {"16", 1, 1, 16, true},
      {"16_16", 1, 1, 32, true},
      {"16_16_16_16", 1, 1, 64, true},
      {"16_EXPAND", 1, 1, 16, true},
      {"16_16_EXPAND", 1, 1, 32, true},
      {"16_16_16_16_EXPAND", 1, 1, 64, true},
      {"16_FLOAT", 1, 1, 16, true},
      {"16_16_FLOAT", 1, 1, 32, true},
      {"16_16_16_16_FLOAT", 1, 1, 64, true},
      {"32", 1, 1, 32, true},
      {"32_32", 1, 1, 64, true},
      {"32_32_32_32", 1, 1, 128, true},
      {"32_FLOAT", 1, 1, 32, true},
      {"32_32_FLOAT", 1, 1, 64, true},
      {"32_32_32_32_FLOAT", 1, 1, 128, true},
      {"32_AS_8", 4, 1, 32, false},
      {"32_AS_8_8", 2, 1, 32, false},
      {"16_MPEG", 1, 1, 16, false},
      {"16_16_MPEG", 1, 1, 32, false},
      {"8_INTERLACED", 1, 1, 8, false},
      {"32_AS_8_INTERLACED", 4, 1, 32, false},
      {"32_AS_8_8_INTERLACED", 1, 1, 32, false},
      {"16_INTERLACED", 1, 1, 16, false},
      {"16_MPEG_INTERLACED", 1, 1, 16, false},
      {"16_16_MPEG_INTERLACED", 1, 1, 32, false},
      {"DXN", 4, 4, 128, true},
      {"8_8_8_8_AS_16_16_16_16", 1, 1, 32, true},
      {"DXT1_AS_16_16_16_16", 4, 4, 64, true},
      {"DXT2_3_AS_16_16_16_16", 4, 4, 128, true},
      {"DXT4_5_AS_16_16_16_16", 4, 4, 128, true},
      {"2_10_10_10_AS_16_16_16_16", 1, 1, 32, true},
      {"10_11_11_AS_16_16_16_16", 1, 1, 32, true},
      {"11_11_10_AS_16_16_16_16", 1, 1, 32, true},
      {"32_32_32_FLOAT", 1, 1, 96, false},
      {"DXT3A", 4, 4, 64, true},
      {"DXT5A", 4, 4, 64, true},
      {"CTX1", 4, 4, 64, true},
      {"DXT3A_AS_1_1_1_1", 4, 4, 64, true},
      {"8_8_8_8_GAMMA_EDRAM", 1, 1, 32, false},
      {"2_10_10_10_FLOAT_EDRAM", 1, 1, 32, false},
  };
  return kInfo[format & 63];
}

// Base format for the _AS_16_16_16_16 variants (rule format.as16).
static uint32_t BaseFormat(uint32_t format, Diag* diag) {
  uint32_t base = format;
  switch (format) {
    case k_8_8_8_8_AS_16_16_16_16: base = k_8_8_8_8; break;
    case k_DXT1_AS_16_16_16_16: base = k_DXT1; break;
    case k_DXT2_3_AS_16_16_16_16: base = k_DXT2_3; break;
    case k_DXT4_5_AS_16_16_16_16: base = k_DXT4_5; break;
    case k_2_10_10_10_AS_16_16_16_16: base = k_2_10_10_10; break;
    case k_10_11_11_AS_16_16_16_16: base = k_10_11_11; break;
    case k_11_11_10_AS_16_16_16_16: base = k_11_11_10; break;
    default: break;
  }
  if (base != format && diag) diag->Use("format.as16");
  return base;
}

// ---------------------------------------------------------------------------
// Memory.

void RangeMemory::Add(uint32_t phys, std::vector<uint8_t> bytes) {
  ranges_.push_back({phys, std::move(bytes)});
}

bool RangeMemory::Read(uint32_t phys, void* dst, uint32_t len) const {
  auto* out = static_cast<uint8_t*>(dst);
  // Bytes not covered by any range are an error (the caller must supply the
  // whole texture).
  for (uint32_t i = 0; i < len;) {
    uint32_t a = phys + i;
    bool found = false;
    for (const Range& r : ranges_) {
      if (a >= r.phys && a - r.phys < r.bytes.size()) {
        uint32_t n = std::min<uint32_t>(len - i, uint32_t(r.bytes.size() - (a - r.phys)));
        std::memcpy(out + i, r.bytes.data() + (a - r.phys), n);
        i += n;
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Addressing.

uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch_blocks,
                       uint32_t bytes_per_block) {
  // rule tile.2d. Written out from the UModel / crunch formula as used by
  // the SDK's GetTiledOffset2D; returns bytes.
  uint32_t bpb_log2 = Log2Floor(bytes_per_block);
  uint32_t pitch = AlignUp(pitch_blocks, 32);
  uint32_t macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (bpb_log2 + 7);
  uint32_t micro = ((x & 7) + ((y & 0xE) << 2)) << bpb_log2;
  uint32_t offset =
      macro + ((micro & ~0xFu) << 1) + (micro & 0xF) + ((y & 1) << 4);
  return ((offset & ~0x1FFu) << 3) + ((y & 16) << 7) +
         ((offset & 0x1C0) << 2) + ((((y & 8) >> 2) + (x >> 3)) & 3) * 64 +
         (offset & 0x3F);
}

uint32_t TiledOffset3D(uint32_t x, uint32_t y, uint32_t z,
                       uint32_t pitch_blocks, uint32_t height_blocks,
                       uint32_t bytes_per_block) {
  // rule tile.3d (SDK GetTiledOffset3D, from XGRAPHICS::TileVolume).
  uint32_t bpb_log2 = Log2Floor(bytes_per_block);
  uint32_t pitch = AlignUp(pitch_blocks, 32);
  uint32_t height = AlignUp(height_blocks, 32);
  uint32_t macro_outer = ((y >> 4) + (z >> 2) * (height >> 4)) * (pitch >> 5);
  uint32_t macro =
      ((((x >> 5) + macro_outer) << (bpb_log2 + 6)) & 0xFFFFFFF) << 1;
  uint32_t micro = (((x & 7) + ((y & 6) << 2)) << (bpb_log2 + 6)) >> 6;
  uint32_t offset_outer = ((y >> 3) + (z >> 2)) & 1;
  uint32_t offset1 = offset_outer + ((((x >> 3) + (offset_outer << 1)) & 3) << 1);
  uint32_t offset2 = ((macro + (micro & ~15u)) << 1) + (micro & 15) +
                     ((z & 3) << (bpb_log2 + 6)) + ((y & 1) << 4);
  uint32_t address = (offset1 & 1) << 3;
  address += (offset2 >> 6) & 7;
  address <<= 3;
  address += offset1 & ~1u;
  address <<= 2;
  address += offset2 & ~511u;
  address <<= 3;
  address += offset2 & 63;
  return address;
}

uint32_t PackedMipFirstLevel(uint32_t width, uint32_t height) {
  uint32_t log2_size = Log2Ceil(std::min(width, height));
  return log2_size > 4 ? log2_size - 4 : 0;
}

bool PackedMipOffset(uint32_t width, uint32_t height, uint32_t depth,
                     uint32_t format, uint32_t mip, uint32_t* x_blocks,
                     uint32_t* y_blocks, uint32_t* z) {
  // rule mip.packed (SDK GetPackedMipOffset). Offsets in texels first.
  uint32_t log2_w = Log2Ceil(width), log2_h = Log2Ceil(height);
  uint32_t log2_size = std::min(log2_w, log2_h);
  *x_blocks = *y_blocks = *z = 0;
  if (log2_size > 4 + mip) return false;
  uint32_t packed_base = log2_size > 4 ? log2_size - 4 : 0;
  uint32_t packed = mip - packed_base;
  uint32_t x = 0, y = 0, zz = 0;
  if (packed < 3) {
    if (log2_w > log2_h) {
      y = 16 >> packed;
    } else {
      x = 16 >> packed;
    }
  } else {
    uint32_t offset;
    if (log2_w > log2_h) {
      offset = (1u << (log2_w - packed_base)) >> (packed - 2);
      x = offset;
    } else {
      offset = (1u << (log2_h - packed_base)) >> (packed - 2);
      y = offset;
    }
    if (offset < 4) {
      uint32_t log2_d = Log2Ceil(depth);
      zz = (log2_d > 1 + packed) ? (log2_d - packed) * 4 : 4;
    }
  }
  const FormatInfo& fi = GetFormatInfo(format);
  *x_blocks = x / fi.block_w;
  *y_blocks = y / fi.block_h;
  *z = zz;
  return true;
}

LevelRange GetLevelRange(const Fetch& f, Diag* diag) {
  // rule mip.range (SDK GetSubresourcesFromFetchConstant).
  if (diag) diag->Use("mip.range");
  uint32_t longest = std::max(f.width, f.height);
  if (f.dimension == kDim3D) longest = std::max(longest, f.depth);
  uint32_t size_max = Log2Floor(longest);
  LevelRange r;
  uint32_t base_page = f.base_page & 0x1FFFF, mip_page = f.mip_page & 0x1FFFF;
  if (mip_page == 0) {
    r.min_level = r.max_level = 0;
  } else {
    r.min_level = std::min(f.min_mip_level, size_max);
    r.max_level = std::max(std::min(f.max_mip_level, size_max), r.min_level);
  }
  if (r.max_level != 0 && base_page == 0) r.min_level = std::max(r.min_level, 1u);
  return r;
}

bool GetLevelLocation(const Fetch& f, uint32_t level, LevelLocation* out,
                      Diag* diag) {
  *out = LevelLocation();
  const FormatInfo& fi = GetFormatInfo(f.format);
  if (diag) diag->Use("mip.layout");
  uint32_t bpb = fi.bytes_per_block();
  bool is3d = f.dimension == kDim3D;
  uint32_t depth = f.depth3d();
  uint32_t packed_level =
      f.packed_mips ? PackedMipFirstLevel(f.width, f.height) : 0xFFFFFFFFu;
  // The level whose storage holds this level.
  uint32_t storage_level = std::min(level, packed_level);

  auto level_strides = [&](uint32_t lvl, bool is_base, uint32_t* pitch_blocks,
                           uint32_t* row_bytes, uint32_t* slice_rows,
                           uint32_t* layer_stride) {
    uint32_t row_texels, rows_texels;
    if (is_base) {
      row_texels = f.pitch_div32 << 5;
      rows_texels = f.height;
    } else {
      row_texels = std::max(NextPow2(f.width) >> lvl, 1u);
      rows_texels = std::max(NextPow2(f.height) >> lvl, 1u);
    }
    *pitch_blocks = AlignUp(DivUp(row_texels, fi.block_w), 32);
    *row_bytes = fi.bits_per_block == 1 ? *pitch_blocks / 8
                                        : *pitch_blocks * bpb;
    if (!f.tiled && !is_base) *row_bytes = AlignUp(*row_bytes, 256);
    *slice_rows = f.dimension == kDim1D
                      ? 1
                      : AlignUp(DivUp(rows_texels, fi.block_h), 32);
    uint32_t stride = *row_bytes * *slice_rows;
    if (is3d) stride *= AlignUp(depth, 4);
    *layer_stride = AlignUp(stride, 4096);
  };

  bool is_base = level == 0;
  uint32_t pitch_blocks, row_bytes, slice_rows, layer_stride;
  if (is_base) {
    level_strides(0, true, &pitch_blocks, &row_bytes, &slice_rows,
                  &layer_stride);
    out->address = f.base_address();
  } else {
    // Mips are stored consecutively from the mip address, each padded to
    // next_pow2(base) >> level, for all layers.
    uint32_t addr = f.mip_address();
    uint32_t first_mip_storage = packed_level == 0 ? 0 : 1;
    for (uint32_t l = first_mip_storage; l < storage_level; ++l) {
      uint32_t pb, rb, sr, ls;
      level_strides(l, false, &pb, &rb, &sr, &ls);
      addr += ls * f.layers();
    }
    level_strides(storage_level, false, &pitch_blocks, &row_bytes, &slice_rows,
                  &layer_stride);
    out->address = addr;
  }
  out->present = true;
  out->width = std::max(f.width >> level, 1u);
  out->height = std::max(f.height >> level, 1u);
  out->depth = std::max(depth >> level, 1u);
  out->row_pitch_blocks = pitch_blocks;
  out->row_pitch_bytes = row_bytes;
  out->slice_rows_blocks = slice_rows;
  out->layer_stride_bytes = layer_stride;
  if (level >= packed_level) {
    if (diag) diag->Use("mip.packed");
    out->in_packed_tail = true;
    PackedMipOffset(f.width, f.height, depth, f.format, level,
                    &out->tail_x_blocks, &out->tail_y_blocks, &out->tail_z);
  }
  return true;
}

void EndianSwap(uint32_t endian, uint8_t* p, size_t len) {
  switch (endian) {
    case 1:  // 8in16
      for (size_t i = 0; i + 1 < len; i += 2) std::swap(p[i], p[i + 1]);
      break;
    case 2:  // 8in32
      for (size_t i = 0; i + 3 < len; i += 4) {
        std::swap(p[i], p[i + 3]);
        std::swap(p[i + 1], p[i + 2]);
      }
      break;
    case 3:  // 16in32
      for (size_t i = 0; i + 3 < len; i += 4) {
        std::swap(p[i], p[i + 2]);
        std::swap(p[i + 1], p[i + 3]);
      }
      break;
    default:
      break;
  }
}

// Reads one block of `bpb` bytes at guest address `addr`, applying the
// endian swap to the containing 32-bit memory words (rule endian).
static bool ReadBlock(const GuestMemory& mem, uint32_t addr, uint32_t bpb,
                      uint32_t endian, uint8_t* out) {
  uint32_t start = addr & ~3u;
  uint32_t end = AlignUp(addr + bpb, 4);
  uint8_t buf[32];
  if (end - start > sizeof(buf)) return false;
  if (!mem.Read(start, buf, end - start)) return false;
  EndianSwap(endian, buf, end - start);
  std::memcpy(out, buf + (addr - start), bpb);
  return true;
}

static bool BlockAddress(const Fetch& f, const LevelLocation& loc,
                         uint32_t layer, uint32_t bx, uint32_t by, uint32_t bz,
                         uint32_t* addr, Diag* diag) {
  const FormatInfo& fi = GetFormatInfo(f.format);
  uint32_t bpb = fi.bytes_per_block();
  uint32_t x = bx + loc.tail_x_blocks, y = by + loc.tail_y_blocks,
           z = bz + loc.tail_z;
  uint32_t a = loc.address + layer * loc.layer_stride_bytes;
  if (f.tiled) {
    if (f.dimension == kDim3D) {
      if (diag) diag->Use("tile.3d");
      a += TiledOffset3D(x, y, z, loc.row_pitch_blocks, loc.slice_rows_blocks,
                         bpb);
    } else {
      if (diag) diag->Use("tile.2d");
      a += TiledOffset2D(x, y, loc.row_pitch_blocks, bpb);
    }
  } else {
    if (diag) diag->Use("linear.address");
    a += (z * loc.slice_rows_blocks + y) * loc.row_pitch_bytes + x * bpb;
  }
  *addr = a;
  return true;
}

bool ReadLevelBlocks(const Fetch& f, const GuestMemory& mem, uint32_t level,
                     uint32_t layer, LevelBlocks* out, Diag* diag) {
  const FormatInfo& fi = GetFormatInfo(f.format);
  if (!fi.supported) {
    if (diag) diag->error = std::string("unsupported format ") + fi.name;
    return false;
  }
  LevelLocation loc;
  if (!GetLevelLocation(f, level, &loc, diag)) return false;
  uint32_t bpb = fi.bytes_per_block();
  out->w_blocks = DivUp(loc.width, fi.block_w);
  out->h_blocks = DivUp(loc.height, fi.block_h);
  out->depth = loc.depth;
  out->bytes.assign(size_t(out->w_blocks) * out->h_blocks * out->depth * bpb,
                    0);
  if (f.endian && diag) diag->Use("endian");
  for (uint32_t z = 0; z < out->depth; ++z) {
    for (uint32_t by = 0; by < out->h_blocks; ++by) {
      for (uint32_t bx = 0; bx < out->w_blocks; ++bx) {
        uint32_t addr;
        BlockAddress(f, loc, layer, bx, by, z, &addr, diag);
        uint8_t* dst =
            &out->bytes[((size_t(z) * out->h_blocks + by) * out->w_blocks +
                         bx) * bpb];
        if (!ReadBlock(mem, addr, bpb, f.endian, dst)) {
          if (diag) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "guest read failed at 0x%08X (level %u block %u,%u)",
                          addr, level, bx, by);
            diag->error = msg;
          }
          return false;
        }
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Decoding. A component is either a fixed-point field (raw bits + width) or
// a float.

namespace {

struct Comp {
  bool is_float = false;
  float f = 0.0f;
  uint32_t raw = 0;
  uint32_t bits = 0;
};

struct Texel {
  Comp c[4];
  uint32_t count = 4;  // components the format has
};

Comp Fixed(uint32_t raw, uint32_t bits) {
  Comp c;
  c.raw = raw;
  c.bits = bits;
  return c;
}
Comp Float(float f) {
  Comp c;
  c.is_float = true;
  c.f = f;
  return c;
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
  float v;
  if (exp == 0) {
    v = std::ldexp(float(man), -24);
  } else if (exp == 31) {
    uint32_t bits = (sign << 31) | 0x7F800000u | (man << 13);
    std::memcpy(&v, &bits, 4);
    return v;
  } else {
    v = std::ldexp(float(man | 0x400), int(exp) - 25);
  }
  return sign ? -v : v;
}

float Float20e4To32(uint32_t f24) {
  // SDK src/graphics/xenos.cpp:154-169, written out arithmetically.
  f24 &= 0xFFFFFF;
  uint32_t mantissa = f24 & 0xFFFFF, exponent = f24 >> 20;
  if (exponent == 0) return std::ldexp(float(mantissa), -20 - 14);
  return std::ldexp(float(mantissa | 0x100000), int(exponent) - 15 - 20);
}

uint16_t LE16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t LE32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

uint32_t Expand5(uint32_t v) { return (v << 3) | (v >> 2); }
uint32_t Expand6(uint32_t v) { return (v << 2) | (v >> 4); }

// DXT color block (8 bytes) -> RGBA8 per texel. dxt1: DXT1 mode selection.
void DecodeDxtColor(const uint8_t* b, bool dxt1, uint8_t out[16][4],
                    Diag* diag) {
  diag->Use("dxt.endpoints");
  diag->Use(dxt1 ? "dxt1.color" : "dxt23.color");
  uint32_t c0 = LE16(b), c1 = LE16(b + 2);
  uint32_t idx = LE32(b + 4);
  uint32_t e[2][3];
  for (int i = 0; i < 2; ++i) {
    uint32_t c = i ? c1 : c0;
    e[i][0] = Expand5((c >> 11) & 31);
    e[i][1] = Expand6((c >> 5) & 63);
    e[i][2] = Expand5(c & 31);
  }
  uint8_t pal[4][4];
  for (int ch = 0; ch < 3; ++ch) {
    pal[0][ch] = uint8_t(e[0][ch]);
    pal[1][ch] = uint8_t(e[1][ch]);
    if (c0 > c1 || !dxt1) {
      pal[2][ch] = uint8_t((2 * e[0][ch] + e[1][ch] + 1) / 3);
      pal[3][ch] = uint8_t((e[0][ch] + 2 * e[1][ch] + 1) / 3);
    } else {
      pal[2][ch] = uint8_t((e[0][ch] + e[1][ch]) / 2);
      pal[3][ch] = 0;
    }
  }
  pal[0][3] = pal[1][3] = pal[2][3] = 255;
  pal[3][3] = 255;
  if (dxt1 && c0 <= c1) {
    diag->Use("dxt1.transparent");
    pal[3][3] = 0;
  }
  for (int t = 0; t < 16; ++t) {
    uint32_t i = (idx >> (2 * t)) & 3;
    std::memcpy(out[t], pal[i], 4);
  }
}

// DXT4/5 alpha block (8 bytes) -> 16 alphas.
void DecodeDxt5Alpha(const uint8_t* b, uint8_t out[16], Diag* diag) {
  diag->Use("dxt5.alpha");
  uint32_t a0 = b[0], a1 = b[1];
  uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) bits |= uint64_t(b[2 + i]) << (8 * i);
  uint32_t pal[8];
  pal[0] = a0;
  pal[1] = a1;
  if (a0 > a1) {
    for (uint32_t k = 1; k <= 6; ++k)
      pal[1 + k] = ((7 - k) * a0 + k * a1 + 3) / 7;
  } else {
    for (uint32_t k = 1; k <= 4; ++k)
      pal[1 + k] = ((5 - k) * a0 + k * a1 + 2) / 5;
    pal[6] = 0;
    pal[7] = 255;
  }
  for (int t = 0; t < 16; ++t) out[t] = uint8_t(pal[(bits >> (3 * t)) & 7]);
}

// DXT2/3 explicit alpha block (8 bytes) -> 16 4-bit alphas.
void DecodeDxt3Alpha(const uint8_t* b, uint8_t out[16]) {
  for (int row = 0; row < 4; ++row) {
    uint32_t w = LE16(b + 2 * row);
    for (int x = 0; x < 4; ++x) out[row * 4 + x] = uint8_t((w >> (4 * x)) & 15);
  }
}

// Decodes one block into texels [16] (block_w x block_h, row-major).
bool DecodeBlock(uint32_t format, const uint8_t* b, Texel* t, uint32_t* n,
                 Diag* diag) {
  format = BaseFormat(format, diag);
  const FormatInfo& fi = GetFormatInfo(format);
  *n = fi.block_w * fi.block_h;
  auto fixed4 = [&](uint32_t v, const uint32_t (&w)[4], uint32_t count) {
    uint32_t shift = 0;
    t[0].count = count;
    for (uint32_t i = 0; i < count; ++i) {
      t[0].c[i] = Fixed(Bits(v, shift, w[i]), w[i]);
      shift += w[i];
    }
  };
  switch (format) {
    case k_8:
    case k_8_A:
    case k_8_B:
      diag->Use("format.layout");
      t[0].count = 1;
      t[0].c[0] = Fixed(b[0], 8);
      return true;
    case k_1_5_5_5:
      diag->Use("format.layout");
      fixed4(LE16(b), {5, 5, 5, 1}, 4);
      return true;
    case k_5_6_5:
      diag->Use("format.layout");
      fixed4(LE16(b), {5, 6, 5, 0}, 3);
      return true;
    case k_6_5_5:
      diag->Use("format.layout");
      fixed4(LE16(b), {5, 5, 6, 0}, 3);
      return true;
    case k_4_4_4_4:
      diag->Use("format.layout");
      fixed4(LE16(b), {4, 4, 4, 4}, 4);
      return true;
    case k_8_8:
      diag->Use("format.layout");
      fixed4(LE16(b), {8, 8, 0, 0}, 2);
      return true;
    case k_8_8_8_8:
    case k_8_8_8_8_A:
      diag->Use("format.layout");
      fixed4(LE32(b), {8, 8, 8, 8}, 4);
      return true;
    case k_2_10_10_10:
      diag->Use("format.layout");
      fixed4(LE32(b), {10, 10, 10, 2}, 4);
      return true;
    case k_10_11_11:
      diag->Use("format.layout");
      diag->Use("format.10_11_11");
      fixed4(LE32(b), {11, 11, 10, 0}, 3);
      return true;
    case k_11_11_10:
      diag->Use("format.layout");
      diag->Use("format.10_11_11");
      fixed4(LE32(b), {10, 11, 11, 0}, 3);
      return true;
    case k_16:
      diag->Use("format.layout");
      t[0].count = 1;
      t[0].c[0] = Fixed(LE16(b), 16);
      return true;
    case k_16_16:
      diag->Use("format.layout");
      fixed4(LE32(b), {16, 16, 0, 0}, 2);
      return true;
    case k_16_16_16_16:
      diag->Use("format.layout");
      t[0].count = 4;
      for (int i = 0; i < 4; ++i) t[0].c[i] = Fixed(LE16(b + 2 * i), 16);
      return true;
    case k_16_EXPAND:
    case k_16_FLOAT:
    case k_16_16_EXPAND:
    case k_16_16_FLOAT:
    case k_16_16_16_16_EXPAND:
    case k_16_16_16_16_FLOAT: {
      diag->Use("float16");
      uint32_t count = (format == k_16_EXPAND || format == k_16_FLOAT) ? 1
                       : (format == k_16_16_EXPAND || format == k_16_16_FLOAT)
                           ? 2
                           : 4;
      t[0].count = count;
      for (uint32_t i = 0; i < count; ++i)
        t[0].c[i] = Float(HalfToFloat(LE16(b + 2 * i)));
      return true;
    }
    case k_32:
    case k_32_32:
    case k_32_32_32_32: {
      uint32_t count = format == k_32 ? 1 : format == k_32_32 ? 2 : 4;
      t[0].count = count;
      for (uint32_t i = 0; i < count; ++i) t[0].c[i] = Fixed(LE32(b + 4 * i), 32);
      return true;
    }
    case k_32_FLOAT:
    case k_32_32_FLOAT:
    case k_32_32_32_32_FLOAT: {
      diag->Use("float32");
      uint32_t count = format == k_32_FLOAT ? 1 : format == k_32_32_FLOAT ? 2 : 4;
      t[0].count = count;
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t u = LE32(b + 4 * i);
        float v;
        std::memcpy(&v, &u, 4);
        t[0].c[i] = Float(v);
      }
      return true;
    }
    case k_24_8: {
      diag->Use("depth.24_8");
      t[0].count = 1;
      t[0].c[0] = Fixed(LE32(b) >> 8, 24);
      return true;
    }
    case k_24_8_FLOAT: {
      diag->Use("depth.24_8");
      t[0].count = 1;
      t[0].c[0] = Float(Float20e4To32(LE32(b) >> 8));
      return true;
    }
    case k_DXT1:
    case k_DXT2_3:
    case k_DXT4_5: {
      uint8_t rgba[16][4];
      uint8_t alpha[16];
      bool has_alpha_block = format != k_DXT1;
      DecodeDxtColor(b + (has_alpha_block ? 8 : 0), format == k_DXT1, rgba,
                     diag);
      if (format == k_DXT2_3) {
        diag->Use("dxt3.alpha");
        DecodeDxt3Alpha(b, alpha);
      } else if (format == k_DXT4_5) {
        DecodeDxt5Alpha(b, alpha, diag);
      }
      for (int i = 0; i < 16; ++i) {
        t[i].count = 4;
        for (int ch = 0; ch < 3; ++ch) t[i].c[ch] = Fixed(rgba[i][ch], 8);
        if (format == k_DXT2_3) {
          t[i].c[3] = Fixed(alpha[i], 4);
        } else if (format == k_DXT4_5) {
          t[i].c[3] = Fixed(alpha[i], 8);
        } else {
          t[i].c[3] = Fixed(rgba[i][3], 8);
        }
      }
      return true;
    }
    case k_DXN: {
      diag->Use("dxn");
      uint8_t x[16], y[16];
      DecodeDxt5Alpha(b, x, diag);
      DecodeDxt5Alpha(b + 8, y, diag);
      for (int i = 0; i < 16; ++i) {
        t[i].count = 2;
        t[i].c[0] = Fixed(x[i], 8);
        t[i].c[1] = Fixed(y[i], 8);
      }
      return true;
    }
    case k_DXT5A: {
      diag->Use("dxt5a");
      uint8_t x[16];
      DecodeDxt5Alpha(b, x, diag);
      for (int i = 0; i < 16; ++i) {
        t[i].count = 1;
        t[i].c[0] = Fixed(x[i], 8);
      }
      return true;
    }
    case k_DXT3A: {
      diag->Use("dxt3a");
      diag->Use("dxt3.alpha");
      uint8_t x[16];
      DecodeDxt3Alpha(b, x);
      for (int i = 0; i < 16; ++i) {
        t[i].count = 1;
        t[i].c[0] = Fixed(x[i], 4);
      }
      return true;
    }
    case k_DXT3A_AS_1_1_1_1: {
      diag->Use("dxt3a_1111");
      diag->Use("dxt3.alpha");
      uint8_t x[16];
      DecodeDxt3Alpha(b, x);
      for (int i = 0; i < 16; ++i) {
        t[i].count = 4;
        for (int ch = 0; ch < 4; ++ch) t[i].c[ch] = Fixed((x[i] >> ch) & 1, 1);
      }
      return true;
    }
    case k_CTX1: {
      diag->Use("ctx1");
      uint32_t g0 = b[0], r0 = b[1], g1 = b[2], r1 = b[3];
      uint32_t idx = LE32(b + 4);
      uint32_t cr[4] = {r0, r1, uint32_t(2.f / 3.f * r0 + 1.f / 3.f * r1),
                        uint32_t(1.f / 3.f * r0 + 2.f / 3.f * r1)};
      uint32_t cg[4] = {g0, g1, uint32_t(2.f / 3.f * g0 + 1.f / 3.f * g1),
                        uint32_t(1.f / 3.f * g0 + 2.f / 3.f * g1)};
      for (int i = 0; i < 16; ++i) {
        uint32_t s = (idx >> (2 * i)) & 3;
        t[i].count = 2;
        t[i].c[0] = Fixed(cr[s], 8);
        t[i].c[1] = Fixed(cg[s], 8);
      }
      return true;
    }
    default:
      diag->error = std::string("no decoder for format ") + fi.name;
      return false;
  }
}

bool IsCompressed(uint32_t base_format) {
  switch (base_format) {
    case k_DXT1: case k_DXT2_3: case k_DXT4_5: case k_DXN: case k_DXT3A:
    case k_DXT5A: case k_CTX1: case k_DXT3A_AS_1_1_1_1:
      return true;
    default:
      return false;
  }
}

float UNorm(uint32_t raw, uint32_t bits) {
  if (bits >= 32) return float(double(raw) / 4294967295.0);
  if (bits > 24) return float(double(raw) / double((1u << bits) - 1));
  // Both operands are exact in f32, so the f32 division is correctly rounded.
  return float(raw) / float((1u << bits) - 1);
}

float SNorm(uint32_t raw, uint32_t bits) {
  int64_t s = (raw & (1ull << (bits - 1))) ? int64_t(raw) - (int64_t(1) << bits)
                                            : int64_t(raw);
  float v = bits > 24 ? float(double(s) / double((1ull << (bits - 1)) - 1))
                      : float(s) / float((1u << (bits - 1)) - 1);
  return std::max(v, -1.0f);
}

}  // namespace

float PWLGammaToLinear(float gamma) {
  // rule sign.gamma: SDK src/graphics/xenos.cpp:27-68.
  gamma = std::min(std::max(gamma, 0.0f), 1.0f);
  float scale, offset;
  if (gamma >= 96.0f / 255.0f) {
    if (gamma >= 192.0f / 255.0f) {
      scale = 8.0f / 1024.0f;
      offset = -1024.0f;
    } else {
      scale = 4.0f / 1024.0f;
      offset = -256.0f;
    }
  } else {
    if (gamma >= 64.0f / 255.0f) {
      scale = 2.0f / 1024.0f;
      offset = -64.0f;
    } else {
      scale = 1.0f / 1024.0f;
      offset = 0.0f;
    }
  }
  float linear = gamma * ((255.0f * 1024.0f) * scale) + offset;
  linear += std::trunc(linear * scale);
  linear *= 1.0f / 1023.0f;
  return linear;
}

// Per-component value under the fetch's sign / number format.
static float ComponentValue(const Fetch& f, uint32_t base_format,
                            const Comp& c, uint32_t sign, Diag* diag) {
  if (c.is_float) return c.f;
  if (f.num_format_integer) {
    diag->Use("num.integer");
    if (sign == kSigned) {
      int64_t s = (c.raw & (1ull << (c.bits - 1)))
                      ? int64_t(c.raw) - (int64_t(1) << c.bits)
                      : int64_t(c.raw);
      return float(s);
    }
    return float(c.raw);
  }
  switch (sign) {
    case kSigned:
      diag->Use("norm.signed");
      if (IsCompressed(base_format)) diag->Use("compressed.signed");
      return SNorm(c.raw, c.bits);
    case kBias: {
      diag->Use("norm.unsigned");
      diag->Use("sign.bias");
      float u = UNorm(c.raw, c.bits);
      return u * 2.0f + -1.0f;
    }
    case kGamma:
      diag->Use("norm.unsigned");
      diag->Use("sign.gamma");
      return PWLGammaToLinear(UNorm(c.raw, c.bits));
    default:
      diag->Use("norm.unsigned");
      return UNorm(c.raw, c.bits);
  }
}

static bool DecodeReverseTexel(const Fetch& f, const GuestMemory& mem,
                               uint32_t level, uint32_t layer, uint32_t x,
                               uint32_t y, uint32_t z, uint32_t* raw,
                               Diag* diag) {
  LevelLocation loc;
  if (!GetLevelLocation(f, level, &loc, diag)) return false;
  if (f.tiled) {
    diag->error = "tiled 1_REVERSE textures are unsupported";
    return false;
  }
  diag->Use("linear.address");
  diag->Use("format.1_reverse");
  uint32_t addr = loc.address + layer * loc.layer_stride_bytes +
                  (z * loc.slice_rows_blocks + y) * loc.row_pitch_bytes + x / 8;
  uint8_t byte;
  if (!mem.Read(addr, &byte, 1)) {
    diag->error = "guest read failed for 1_REVERSE texture";
    return false;
  }
  *raw = (byte >> (7 - x % 8)) & 1;
  return true;
}

static bool DecodeLevelImpl(const Fetch& f, const GuestMemory& mem,
                            uint32_t level, uint32_t layer, bool apply_fetch,
                            DecodedTexels* out, Diag* diag) {
  diag->Use("fetch.fields");
  if (f.format == k_1_REVERSE) {
    uint32_t w = std::max(f.width >> level, 1u);
    uint32_t h = std::max(f.height >> level, 1u);
    uint32_t d = std::max(f.depth3d() >> level, 1u);
    out->width = w;
    out->height = h;
    out->depth = d;
    out->rgba.resize(size_t(w) * h * d * 4);
    for (uint32_t z = 0; z < d; ++z) {
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          uint32_t raw;
          if (!DecodeReverseTexel(f, mem, level, layer, x, y, z, &raw, diag)) {
            return false;
          }
          float data[4];
          for (uint32_t c = 0; c < 4; ++c) {
            Comp comp = Fixed(raw, 1);
            data[c] = apply_fetch
                          ? ComponentValue(f, f.format, comp, f.sign[c], diag)
                          : UNorm(raw, 1);
          }
          float* o = &out->rgba[((size_t(z) * h + y) * w + x) * 4];
          if (!apply_fetch) {
            std::memcpy(o, data, sizeof(data));
            continue;
          }
          diag->Use("format.replicate");
          diag->Use("swizzle");
          float scale = std::ldexp(1.0f, f.exp_adjust);
          for (uint32_t c = 0; c < 4; ++c) {
            uint32_t s = f.swizzle[c];
            float v = s < 4 ? data[s] : float(s & 1);
            if ((s & 0b101) != 4 && f.exp_adjust) v *= scale;
            o[c] = v;
          }
        }
      }
    }
    return true;
  }
  LevelBlocks blocks;
  if (!ReadLevelBlocks(f, mem, level, layer, &blocks, diag)) return false;
  const FormatInfo& fi = GetFormatInfo(f.format);
  uint32_t base_format = BaseFormat(f.format, diag);
  uint32_t w = std::max(f.width >> level, 1u), h = std::max(f.height >> level, 1u);
  out->width = w;
  out->height = h;
  out->depth = blocks.depth;
  out->rgba.assign(size_t(w) * h * blocks.depth * 4, 0.0f);
  uint32_t bpb = fi.bytes_per_block();
  Texel texels[16];
  for (uint32_t z = 0; z < blocks.depth; ++z) {
    for (uint32_t by = 0; by < blocks.h_blocks; ++by) {
      for (uint32_t bx = 0; bx < blocks.w_blocks; ++bx) {
        const uint8_t* b =
            &blocks.bytes[((size_t(z) * blocks.h_blocks + by) * blocks.w_blocks +
                           bx) * bpb];
        uint32_t n;
        if (!DecodeBlock(f.format, b, texels, &n, diag)) return false;
        for (uint32_t ty = 0; ty < fi.block_h; ++ty) {
          for (uint32_t tx = 0; tx < fi.block_w; ++tx) {
            uint32_t x = bx * fi.block_w + tx, y = by * fi.block_h + ty;
            if (x >= w || y >= h) continue;
            const Texel& t = texels[ty * fi.block_w + tx];
            float data[4];
            // rule format.replicate: missing components repeat the last.
            if (t.count < 4) diag->Use("format.replicate");
            for (uint32_t c = 0; c < 4; ++c) {
              const Comp& comp = t.c[std::min(c, t.count - 1)];
              uint32_t sign = apply_fetch ? f.sign[c] : kUnsigned;
              if (!apply_fetch) {
                data[c] = comp.is_float ? comp.f : UNorm(comp.raw, comp.bits);
              } else {
                data[c] = ComponentValue(f, base_format, comp, sign, diag);
              }
            }
            float* o = &out->rgba[((size_t(z) * h + y) * w + x) * 4];
            if (!apply_fetch) {
              std::memcpy(o, data, sizeof(data));
              continue;
            }
            diag->Use("swizzle");
            float scale = std::ldexp(1.0f, f.exp_adjust);
            if (f.exp_adjust) diag->Use("exp_adjust");
            for (uint32_t c = 0; c < 4; ++c) {
              uint32_t s = f.swizzle[c];
              float v;
              if (s < 4) {
                v = data[s];
              } else if (s & 1) {
                v = 1.0f;  // 5 = constant 1; 7 (undefined) as 1 (& 0b101)
              } else {
                v = 0.0f;  // 4 = constant 0; 6 (undefined) as 0 (& 0b101)
              }
              if ((s & 0b101) != 4 && f.exp_adjust) v *= scale;
              o[c] = v;
            }
          }
        }
      }
    }
  }
  return true;
}

// One texel, decoded from the block that holds it. Components before the
// swizzle: signed components as snorm (or integer), every other sign as the
// plain unsigned value (bias / gamma are applied by the sampler after
// filtering, rule sample.order).
bool DecodeTexelPreSwizzle(const Fetch& f, const GuestMemory& mem,
                           uint32_t level, uint32_t layer, uint32_t x,
                           uint32_t y, uint32_t z, float out[4], Diag* diag) {
  Diag local;
  if (!diag) diag = &local;
  const FormatInfo& fi = GetFormatInfo(f.format);
  if (!fi.supported) {
    diag->error = std::string("unsupported format ") + fi.name;
    return false;
  }
  if (f.format == k_1_REVERSE) {
    uint32_t raw;
    if (!DecodeReverseTexel(f, mem, level, layer, x, y, z, &raw, diag)) {
      return false;
    }
    diag->Use("format.replicate");
    Comp comp = Fixed(raw, 1);
    for (uint32_t c = 0; c < 4; ++c) {
      uint32_t sign = f.sign[c] == kSigned ? kSigned : kUnsigned;
      out[c] = ComponentValue(f, f.format, comp, sign, diag);
    }
    return true;
  }
  LevelLocation loc;
  if (!GetLevelLocation(f, level, &loc, diag)) return false;
  uint32_t bpb = fi.bytes_per_block();
  uint32_t addr;
  BlockAddress(f, loc, layer, x / fi.block_w, y / fi.block_h, z, &addr, diag);
  uint8_t b[16];
  if (f.endian) diag->Use("endian");
  if (!ReadBlock(mem, addr, bpb, f.endian, b)) {
    char msg[96];
    std::snprintf(msg, sizeof(msg), "guest read failed at 0x%08X", addr);
    diag->error = msg;
    return false;
  }
  Texel texels[16];
  uint32_t n;
  if (!DecodeBlock(f.format, b, texels, &n, diag)) return false;
  uint32_t base_format = BaseFormat(f.format, diag);
  const Texel& t = texels[(y % fi.block_h) * fi.block_w + (x % fi.block_w)];
  if (t.count < 4) diag->Use("format.replicate");
  for (uint32_t c = 0; c < 4; ++c) {
    const Comp& comp = t.c[std::min(c, t.count - 1)];
    uint32_t sign = f.sign[c] == kSigned ? kSigned : kUnsigned;
    out[c] = ComponentValue(f, base_format, comp, sign, diag);
  }
  return true;
}

bool DecodeLevel(const Fetch& f, const GuestMemory& mem, uint32_t level,
                 uint32_t layer, DecodedTexels* out, Diag* diag) {
  Diag local;
  return DecodeLevelImpl(f, mem, level, layer, true, out, diag ? diag : &local);
}

bool DecodeLevelUnsignedComponents(const Fetch& f, const GuestMemory& mem,
                                   uint32_t level, uint32_t layer,
                                   DecodedTexels* out, Diag* diag) {
  Diag local;
  return DecodeLevelImpl(f, mem, level, layer, false, out,
                         diag ? diag : &local);
}

}  // namespace refxenos::tex
