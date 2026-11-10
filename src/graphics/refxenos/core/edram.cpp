// refxenos EDRAM model. See edram.h for the layout and sample convention.

#include "core/edram.h"

#include <algorithm>
#include <cmath>

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

RX_DEFINE_RULE(edram_pack_unorm, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:3086-3116,3180-3205 "
               "(clamp, scale, add 0.5, truncate)",
               false, "");
RX_DEFINE_RULE(edram_pwl_gamma, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/xenos.cc:23-100; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:3121-3172,3461-3491",
               false, "");
RX_DEFINE_RULE(edram_7e3_round_nearest_even, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:3812-3884",
               false, "");
RX_DEFINE_RULE(edram_fixed16_round_half_up, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/pixel_formats.xesli:196-213; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:3253-3298",
               false, "");
RX_DEFINE_RULE(edram_float16_extended_range, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:3745-3809; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:3301-3318 (round to nearest even, "
               "exponent 31 is finite, NaN becomes zero)",
               false, "");
RX_DEFINE_RULE(depth_d24s8_round_nearest_even, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:4135-4149; "
               "src/xenia/gpu/xenos.cc:151-167 for 20e4 decoding",
               false, "");
RX_DEFINE_RULE(depth_d24fs8_round_nearest_even, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:4128-4134; "
               "src/xenia/gpu/xenos.cc:126-167",
               false, "");

namespace {

float ClampNanZero(float value, float low, float high) {
  if (std::isnan(value)) return 0.0f;
  return std::min(std::max(value, low), high);
}

uint32_t PackUNorm(float value, uint32_t max) {
  return uint32_t(ClampNanZero(value, 0.0f, 1.0f) * float(max) + 0.5f);
}

uint32_t PackFixed16(float value) {
  float clamped = ClampNanZero(value, -32.0f, 32767.0f / 1024.0f);
  return uint32_t(int32_t(std::floor(clamped * 1024.0f + 0.5f))) & 0xFFFFu;
}

int16_t Signed16(uint32_t value) { return int16_t(value & 0xFFFFu); }

float RoundNearestEven(float value) {
  float floor_value = std::floor(value);
  float fraction = value - floor_value;
  if (fraction > 0.5f || (fraction == 0.5f && std::fmod(floor_value, 2.0f) != 0.0f)) {
    floor_value += 1.0f;
  }
  return floor_value;
}

}  // namespace

uint16_t Float32To16Extended(float value) {
  RX_USE(edram_float16_extended_range);
  if (std::isnan(value)) return 0;
  value = std::min(std::max(value, -131008.0f), 131008.0f);
  uint32_t bits = AsU32(value);
  uint16_t sign = uint16_t((bits >> 16) & 0x8000u);
  bits &= 0x7FFFFFFFu;
  uint32_t exponent = bits >> 23;
  uint32_t significand = (bits & 0x7FFFFFu) | 0x800000u;
  if (!bits) return sign;
  int32_t half_exponent = int32_t(exponent) - 127 + 15;
  if (half_exponent <= 0) {
    uint32_t shift = 126u - exponent;
    if (shift >= 25) return sign;
    uint32_t result = significand >> shift;
    uint32_t remainder = shift ? significand & ((1u << shift) - 1u) : 0;
    uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (result & 1u))) ++result;
    return sign | uint16_t(result);
  }
  uint32_t mantissa = significand >> 13;
  uint32_t remainder = significand & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (mantissa & 1u))) ++mantissa;
  if (mantissa == 0x800u) {
    mantissa = 0;
    ++half_exponent;
  }
  if (half_exponent > 31) return sign | 0x7BFFu;
  return sign | uint16_t((uint32_t(half_exponent) << 10) | (mantissa & 0x3FFu));
}

float Float16ExtendedTo32(uint16_t value) {
  RX_USE(edram_float16_extended_range);
  uint32_t exponent = (value >> 10) & 31u;
  uint32_t mantissa = value & 0x3FFu;
  float result;
  if (!exponent) {
    result = std::ldexp(float(mantissa), -24);
  } else {
    result = std::ldexp(float(1024u + mantissa), int(exponent) - 25);
  }
  return (value & 0x8000u) ? -result : result;
}

uint32_t Float32To7e3(float value) {
  RX_USE(edram_7e3_round_nearest_even);
  value = ClampNanZero(value, 0.0f, 31.875f);
  if (value == 0.0f) return 0;
  uint32_t bits = AsU32(value);
  if (bits < 0x3E800000u) {
    uint32_t shift = std::min(125u - (bits >> 23), 24u);
    bits = ((bits & 0x7FFFFFu) | 0x800000u) >> shift;
  } else {
    bits += 0xC2000000u;
  }
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (bits >> 16) & 0x3FFu;
}

uint64_t PackColor(xenos::ColorRenderTargetFormat format, const float color[4]) {
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
      RX_USE(edram_pack_unorm);
      float c[4] = {color[0], color[1], color[2], color[3]};
      if (format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        RX_USE(edram_pwl_gamma);
        for (uint32_t i = 0; i < 3; ++i) c[i] = xenos::LinearToPWLGamma(c[i]);
      }
      return PackUNorm(c[0], 255) | (uint64_t(PackUNorm(c[1], 255)) << 8) |
             (uint64_t(PackUNorm(c[2], 255)) << 16) |
             (uint64_t(PackUNorm(c[3], 255)) << 24);
    }
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
      RX_USE(edram_pack_unorm);
      return PackUNorm(color[0], 1023) | (uint64_t(PackUNorm(color[1], 1023)) << 10) |
             (uint64_t(PackUNorm(color[2], 1023)) << 20) |
             (uint64_t(PackUNorm(color[3], 3)) << 30);
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      RX_USE(edram_pack_unorm);
      return Float32To7e3(color[0]) | (uint64_t(Float32To7e3(color[1])) << 10) |
             (uint64_t(Float32To7e3(color[2])) << 20) |
             (uint64_t(PackUNorm(color[3], 3)) << 30);
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      RX_USE(edram_fixed16_round_half_up);
      return PackFixed16(color[0]) | (uint64_t(PackFixed16(color[1])) << 16) |
             (uint64_t(PackFixed16(color[2])) << 32) |
             (uint64_t(PackFixed16(color[3])) << 48);
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return Float32To16Extended(color[0]) | (uint64_t(Float32To16Extended(color[1])) << 16) |
             (uint64_t(Float32To16Extended(color[2])) << 32) |
             (uint64_t(Float32To16Extended(color[3])) << 48);
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      return AsU32(color[0]) | (uint64_t(AsU32(color[1])) << 32);
    default:
      return 0;
  }
}

void UnpackColor(xenos::ColorRenderTargetFormat format, uint64_t packed, float color[4]) {
  color[0] = color[1] = color[2] = 0.0f;
  color[3] = 1.0f;
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      for (uint32_t i = 0; i < 4; ++i) {
        color[i] = float((packed >> (8 * i)) & 255u) * (1.0f / 255.0f);
      }
      if (format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        RX_USE(edram_pwl_gamma);
        for (uint32_t i = 0; i < 3; ++i) color[i] = xenos::PWLGammaToLinear(color[i]);
      }
      break;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
      for (uint32_t i = 0; i < 3; ++i) {
        color[i] = float((packed >> (10 * i)) & 1023u) * (1.0f / 1023.0f);
      }
      color[3] = float((packed >> 30) & 3u) * (1.0f / 3.0f);
      break;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      RX_USE(edram_7e3_round_nearest_even);
      for (uint32_t i = 0; i < 3; ++i) {
        color[i] = xenos::Float7e3To32(uint32_t(packed >> (10 * i)) & 1023u);
      }
      color[3] = float((packed >> 30) & 3u) * (1.0f / 3.0f);
      break;
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16: {
      uint32_t count = format == xenos::ColorRenderTargetFormat::k_16_16 ? 2 : 4;
      for (uint32_t i = 0; i < count; ++i) {
        color[i] = float(Signed16(uint32_t(packed >> (16 * i)))) * (1.0f / 1024.0f);
      }
      break;
    }
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT: {
      uint32_t count = format == xenos::ColorRenderTargetFormat::k_16_16_FLOAT ? 2 : 4;
      for (uint32_t i = 0; i < count; ++i) {
        color[i] = Float16ExtendedTo32(uint16_t(packed >> (16 * i)));
      }
      break;
    }
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      color[0] = AsF32(uint32_t(packed));
      if (format == xenos::ColorRenderTargetFormat::k_32_32_FLOAT) {
        color[1] = AsF32(uint32_t(packed >> 32));
      }
      break;
    default:
      break;
  }
}

uint32_t PackDepthStencil(xenos::DepthRenderTargetFormat format, float depth, uint8_t stencil) {
  depth = ClampNanZero(depth, 0.0f, format == xenos::DepthRenderTargetFormat::kD24S8 ? 1.0f : 2.0f);
  uint32_t packed;
  if (format == xenos::DepthRenderTargetFormat::kD24FS8) {
    RX_USE(depth_d24fs8_round_nearest_even);
    packed = xenos::Float32To20e4(depth, true);
  } else {
    RX_USE(depth_d24s8_round_nearest_even);
    packed = uint32_t(RoundNearestEven(depth * 16777215.0f));
  }
  return (packed << 8) | stencil;
}

float UnpackDepth(xenos::DepthRenderTargetFormat format, uint32_t packed) {
  uint32_t depth = (packed >> 8) & 0xFFFFFFu;
  if (format == xenos::DepthRenderTargetFormat::kD24FS8) {
    RX_USE(depth_d24fs8_round_nearest_even);
    return xenos::Float20e4To32(depth);
  }
  RX_USE(depth_d24s8_round_nearest_even);
  return xenos::UNorm24To32(depth);
}

}  // namespace refxenos
