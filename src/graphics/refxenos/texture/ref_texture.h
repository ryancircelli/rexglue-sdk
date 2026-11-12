// refxenos texture reference: a slow, obvious, CPU-only model of how the
// Xenos (Xbox 360 GPU) fetches texels from guest memory.
//
// Pure C++17, no SDK dependencies, so it stays an independent opinion of the
// rules the Metal texture cache implements. Every rule has an id in
// ref_rules.cc with its source (XDK doc page, Xenia/SDK file:line at a pinned
// commit, or "assumed"); decode functions report the ids they used.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace refxenos::tex {

// Xenos GPUTEXTUREFORMAT (XDK GPUTEXTUREFORMAT.htm).
enum Format : uint32_t {
  k_1_REVERSE = 0,
  k_1 = 1,
  k_8 = 2,
  k_1_5_5_5 = 3,
  k_5_6_5 = 4,
  k_6_5_5 = 5,
  k_8_8_8_8 = 6,
  k_2_10_10_10 = 7,
  k_8_A = 8,
  k_8_B = 9,
  k_8_8 = 10,
  k_Cr_Y1_Cb_Y0_REP = 11,
  k_Y1_Cr_Y0_Cb_REP = 12,
  k_16_16_EDRAM = 13,
  k_8_8_8_8_A = 14,
  k_4_4_4_4 = 15,
  k_10_11_11 = 16,
  k_11_11_10 = 17,
  k_DXT1 = 18,
  k_DXT2_3 = 19,
  k_DXT4_5 = 20,
  k_16_16_16_16_EDRAM = 21,
  k_24_8 = 22,
  k_24_8_FLOAT = 23,
  k_16 = 24,
  k_16_16 = 25,
  k_16_16_16_16 = 26,
  k_16_EXPAND = 27,
  k_16_16_EXPAND = 28,
  k_16_16_16_16_EXPAND = 29,
  k_16_FLOAT = 30,
  k_16_16_FLOAT = 31,
  k_16_16_16_16_FLOAT = 32,
  k_32 = 33,
  k_32_32 = 34,
  k_32_32_32_32 = 35,
  k_32_FLOAT = 36,
  k_32_32_FLOAT = 37,
  k_32_32_32_32_FLOAT = 38,
  k_32_AS_8 = 39,
  k_32_AS_8_8 = 40,
  k_16_MPEG = 41,
  k_16_16_MPEG = 42,
  k_8_INTERLACED = 43,
  k_32_AS_8_INTERLACED = 44,
  k_32_AS_8_8_INTERLACED = 45,
  k_16_INTERLACED = 46,
  k_16_MPEG_INTERLACED = 47,
  k_16_16_MPEG_INTERLACED = 48,
  k_DXN = 49,
  k_8_8_8_8_AS_16_16_16_16 = 50,
  k_DXT1_AS_16_16_16_16 = 51,
  k_DXT2_3_AS_16_16_16_16 = 52,
  k_DXT4_5_AS_16_16_16_16 = 53,
  k_2_10_10_10_AS_16_16_16_16 = 54,
  k_10_11_11_AS_16_16_16_16 = 55,
  k_11_11_10_AS_16_16_16_16 = 56,
  k_32_32_32_FLOAT = 57,
  k_DXT3A = 58,
  k_DXT5A = 59,
  k_CTX1 = 60,
  k_DXT3A_AS_1_1_1_1 = 61,
  k_8_8_8_8_GAMMA_EDRAM = 62,
  k_2_10_10_10_FLOAT_EDRAM = 63,
};

enum Sign : uint32_t { kUnsigned = 0, kSigned = 1, kBias = 2, kGamma = 3 };
enum Dimension : uint32_t { kDim1D = 0, kDim2D = 1, kDim3D = 2, kDimCube = 3 };

// Field-by-field view of GPUTEXTURE_FETCH_CONSTANT (XDK
// GPUTEXTURE_FETCH_CONSTANT.htm). dw[] are the register values.
struct Fetch {
  uint32_t dw[6] = {};
  uint32_t type, sign[4], clamp[3], pitch_div32, tiled;
  uint32_t format, endian, request_size, stacked, clamp_policy, base_page;
  uint32_t width, height, depth;  // decoded sizes (not minus 1)
  uint32_t num_format_integer, swizzle[4];
  int32_t exp_adjust;
  uint32_t mag_filter, min_filter, mip_filter, aniso_filter, border_size;
  uint32_t vol_mag_filter, vol_min_filter, min_mip_level, max_mip_level;
  uint32_t mag_aniso_walk, min_aniso_walk;
  int32_t lod_bias, grad_exp_adjust_h, grad_exp_adjust_v;
  uint32_t border_color, force_bcw_to_max, tri_clamp;
  int32_t aniso_bias;
  uint32_t dimension, packed_mips, mip_page;

  static Fetch Parse(const uint32_t dwords[6]);
  uint32_t base_address() const { return base_page << 12; }
  uint32_t mip_address() const { return mip_page << 12; }
  // Array layers: stack depth for stacked 2D, 6 for cube, else 1.
  uint32_t layers() const;
  // Depth for 3D, else 1.
  uint32_t depth3d() const { return dimension == kDim3D ? depth : 1; }
};

struct FormatInfo {
  const char* name;
  uint32_t block_w, block_h, bits_per_block;
  bool supported;  // decodable by this reference
  uint32_t bytes_per_block() const { return bits_per_block / 8; }
};
const FormatInfo& GetFormatInfo(uint32_t format);

// Read-only view of guest physical memory.
class GuestMemory {
 public:
  virtual ~GuestMemory() = default;
  virtual bool Read(uint32_t phys, void* dst, uint32_t len) const = 0;
};

// Simple implementation: a set of (address, bytes) ranges.
class RangeMemory : public GuestMemory {
 public:
  void Add(uint32_t phys, std::vector<uint8_t> bytes);
  bool Read(uint32_t phys, void* dst, uint32_t len) const override;

 private:
  struct Range {
    uint32_t phys;
    std::vector<uint8_t> bytes;
  };
  std::vector<Range> ranges_;
};

struct Diag {
  std::vector<std::string> rules_used;  // ids from ref_rules.cc, sorted unique
  std::string error;
  void Use(const char* rule_id);
};

// Where one mip level lives. Levels are counted from the base (0).
struct LevelLocation {
  bool present = false;
  uint32_t width = 0, height = 0, depth = 0;  // texels of this level
  uint32_t address = 0;                       // start of the level (or of its mip tail)
  uint32_t row_pitch_blocks = 0;              // tile-aligned pitch used for addressing
  uint32_t row_pitch_bytes = 0;               // linear: bytes between block rows
  uint32_t slice_rows_blocks = 0;             // block rows per z slice (3D) / layer
  uint32_t layer_stride_bytes = 0;
  bool in_packed_tail = false;
  uint32_t tail_x_blocks = 0, tail_y_blocks = 0, tail_z = 0;
};

// Levels that the fetch constant makes addressable (mip_min..mip_max with
// the SDK's base/mip page rules, see rule "mip.range").
struct LevelRange {
  uint32_t min_level = 0, max_level = 0;
};
LevelRange GetLevelRange(const Fetch& fetch, Diag* diag);
bool GetLevelLocation(const Fetch& fetch, uint32_t level, LevelLocation* out, Diag* diag);

// Byte offset of block (x, y) inside a tiled 2D surface with the given pitch
// in blocks (rule "tile.2d"); and of (x, y, z) in a tiled 3D surface.
uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch_blocks, uint32_t bytes_per_block);
uint32_t TiledOffset3D(uint32_t x, uint32_t y, uint32_t z, uint32_t pitch_blocks,
                       uint32_t height_blocks, uint32_t bytes_per_block);
// Offset of a packed mip inside the tail, in blocks (rule "mip.packed").
bool PackedMipOffset(uint32_t width, uint32_t height, uint32_t depth, uint32_t format,
                     uint32_t level, uint32_t* x_blocks, uint32_t* y_blocks, uint32_t* z);
uint32_t PackedMipFirstLevel(uint32_t width, uint32_t height);

// Applies the fetch endian mode to a block (rule "endian").
void EndianSwap(uint32_t endian, uint8_t* data, size_t len);

// Raw blocks of one level/layer, untiled and endian-swapped, row-major by
// block, tightly packed (w_blocks * bytes_per_block per row).
struct LevelBlocks {
  uint32_t w_blocks = 0, h_blocks = 0, depth = 0;
  std::vector<uint8_t> bytes;
};
bool ReadLevelBlocks(const Fetch& fetch, const GuestMemory& mem, uint32_t level, uint32_t layer,
                     LevelBlocks* out, Diag* diag);

// Decoded format components (X, Y, Z, W of the data format) for every texel
// of a level, per sign interpretation. Raw integers for the bits
// (before normalisation) are kept so callers can run the sign rules.
struct DecodedTexels {
  uint32_t width = 0, height = 0, depth = 0;
  // f32 x4 per texel, [z][y][x][c]: the value of a point-filtered tfetch at
  // the texel centre: format decode, per-component sign / bias / gamma,
  // fetch swizzle, then exp_adjust (rule chain in ref_rules.cc).
  std::vector<float> rgba;
};
bool DecodeLevel(const Fetch& fetch, const GuestMemory& mem, uint32_t level, uint32_t layer,
                 DecodedTexels* out, Diag* diag);

// Same as DecodeLevel but without sign / swizzle / exp_adjust: the four data
// components as unsigned-normalised values (for cross-checking decoders).
bool DecodeLevelUnsignedComponents(const Fetch& fetch, const GuestMemory& mem, uint32_t level,
                                   uint32_t layer, DecodedTexels* out, Diag* diag);

// Xenos piecewise-linear gamma to linear (rule "sign.gamma").
// One texel of a level / layer (z = depth slice for 3D), components before
// the swizzle; signed components as snorm, all other signs unsigned (no bias
// or gamma). Used by the sampler (ref_sample.h).
bool DecodeTexelPreSwizzle(const Fetch& fetch, const GuestMemory& mem, uint32_t level,
                           uint32_t layer, uint32_t x, uint32_t y, uint32_t z, float out[4],
                           Diag* diag);

float PWLGammaToLinear(float gamma);

}  // namespace refxenos::tex
