// refxenos EDRAM resolve. See core/edram.h for the guest sample order.

#include "core/resolve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "texture/ref_texture.h"

namespace refxenos {
using namespace rex::graphics;  // XE_GPU_REG_* register indices

namespace {
uint32_t ReadGuestWordLittleEndian(const Memory& memory, uint32_t address) {
  uint8_t bytes[4] = {};
  if (!memory.Read(address, bytes, sizeof(bytes))) return 0;
  return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
         (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}
}  // namespace

RX_DEFINE_RULE(resolve_rectangle_and_scissor, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:1226-1384 "
               "(three k_32_32_FLOAT vertices, top-left bounds, scissor, 8-pixel expansion)",
               false, "");
RX_DEFINE_RULE(resolve_sample_select_sanitize, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:1138-1178,1386-1402",
               false, "");
RX_DEFINE_RULE(resolve_tiled_destination, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/texture_address.xesli:15-71; "
               "src/xenia/gpu/shaders/resolve.xesli:176-207; "
               "src/xenia/gpu/draw_util.cc:1460-1563 (destination base and macro-phase adjustment)",
               false, "");
RX_DEFINE_RULE(resolve_endian_128, 2,
               "include/rex/graphics/registers.h:869-884; "
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/endian.xesli:24-74; "
               "src/xenia/gpu/shaders/resolve_full_8bpp.xesli:52-58 "
               "(destination-width-specific byte swaps)",
               false, "");
RX_DEFINE_RULE(resolve_color_unpack, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/resolve.xesli:285-365,379-551; "
               "src/xenia/gpu/shaders/pixel_formats.xesli:354-441",
               false, "");
RX_DEFINE_RULE(resolve_color_pack, 2,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/pixel_formats.xesli:102-213,264-352",
               false, "");
RX_DEFINE_RULE(resolve_exp_bias, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/resolve.xesli:150-152,737-790 "
               "(multiply by 2^copy_dest_exp_bias after averaging)",
               false, "");
RX_DEFINE_RULE(resolve_msaa_average, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/resolve.xesli:723-915 "
               "(sample order 0,1,2,3; pairwise groups k01 and k23)",
               false, "");
RX_DEFINE_RULE(resolve_pwl_gamma_average, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/resolve.xesli:645-704,778-790 "
               "(integer 10-bit linear codes, truncating average, PWL code quantization)",
               false, "");
RX_DEFINE_RULE(resolve_red_blue_swap, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shaders/resolve.xesli:792-795,939-953,1019-1026",
               false, "");
RX_DEFINE_RULE(resolve_clear, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.h:806-843; "
               "src/xenia/gpu/draw_util.cc:1607-1679",
               false, "");
RX_DEFINE_RULE(resolve_raw_copy, 2,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:1697-1755; "
               "src/xenia/gpu/shaders/resolve.xesli:1019-1063 (raw copy and packed R/B swap)",
               false, "");
RX_DEFINE_RULE(resolve_depth_raw, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:1386-1425,1607-1625; "
               "depth resolves preserve D24S8/D24FS8 words and disable averaging and exp bias",
               false, "");

namespace {

struct DestInfo {
  xenos::ColorFormat format;
  uint32_t bytes;
  uint32_t bytes_log2;
};

bool GetDestInfo(xenos::ColorFormat input, DestInfo& info) {
  switch (input) {
    case xenos::ColorFormat::k_8_A:
    case xenos::ColorFormat::k_8_B:
      input = xenos::ColorFormat::k_8;
      break;
    case xenos::ColorFormat::k_8_8_8_8_A:
      input = xenos::ColorFormat::k_8_8_8_8;
      break;
    case xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16:
      input = xenos::ColorFormat::k_8_8_8_8;
      break;
    case xenos::ColorFormat::k_2_10_10_10_AS_16_16_16_16:
      input = xenos::ColorFormat::k_2_10_10_10;
      break;
    case xenos::ColorFormat::k_10_11_11_AS_16_16_16_16:
      input = xenos::ColorFormat::k_10_11_11;
      break;
    case xenos::ColorFormat::k_11_11_10_AS_16_16_16_16:
      input = xenos::ColorFormat::k_11_11_10;
      break;
    default:
      break;
  }
  info.format = input;
  switch (input) {
    case xenos::ColorFormat::k_8: info.bytes = 1; break;
    case xenos::ColorFormat::k_1_5_5_5:
    case xenos::ColorFormat::k_5_6_5:
    case xenos::ColorFormat::k_6_5_5:
    case xenos::ColorFormat::k_8_8:
    case xenos::ColorFormat::k_4_4_4_4:
    case xenos::ColorFormat::k_16:
    case xenos::ColorFormat::k_16_FLOAT: info.bytes = 2; break;
    case xenos::ColorFormat::k_8_8_8_8:
    case xenos::ColorFormat::k_2_10_10_10:
    case xenos::ColorFormat::k_10_11_11:
    case xenos::ColorFormat::k_11_11_10:
    case xenos::ColorFormat::k_16_16:
    case xenos::ColorFormat::k_16_16_FLOAT:
    case xenos::ColorFormat::k_32_FLOAT: info.bytes = 4; break;
    case xenos::ColorFormat::k_16_16_16_16:
    case xenos::ColorFormat::k_16_16_16_16_FLOAT:
    case xenos::ColorFormat::k_32_32_FLOAT: info.bytes = 8; break;
    case xenos::ColorFormat::k_32_32_32_32_FLOAT: info.bytes = 16; break;
    default: return false;
  }
  info.bytes_log2 = info.bytes == 1 ? 0 : (info.bytes == 2 ? 1 : (info.bytes == 4 ? 2 :
                    (info.bytes == 8 ? 3 : 4)));
  return true;
}

bool ColorResolveNumberFormatMatches(xenos::ColorFormat format,
                                     xenos::SurfaceNumberFormat number) {
  switch (format) {
    case xenos::ColorFormat::k_16_FLOAT:
    case xenos::ColorFormat::k_16_16_FLOAT:
    case xenos::ColorFormat::k_16_16_16_16_FLOAT:
    case xenos::ColorFormat::k_32_FLOAT:
    case xenos::ColorFormat::k_32_32_FLOAT:
    case xenos::ColorFormat::k_32_32_32_32_FLOAT:
      return number == xenos::SurfaceNumberFormat::kFloat;
    default:
      return number == xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction;
  }
}

uint32_t SampleCount(xenos::MsaaSamples msaa) {
  return msaa == xenos::MsaaSamples::k4X ? 4 : (msaa == xenos::MsaaSamples::k2X ? 2 : 1);
}

xenos::CopySampleSelect SanitizeSampleSelect(xenos::CopySampleSelect select,
                                              xenos::MsaaSamples msaa, bool depth) {
  RX_USE(resolve_sample_select_sanitize);
  using S = xenos::CopySampleSelect;
  if (msaa >= xenos::MsaaSamples::k4X) {
    if (select > S::k0123) select = S::k0123;
    if (depth) {
      if (select == S::k01 || select == S::k0123) select = S::k0;
      if (select == S::k23) select = S::k2;
    }
  } else if (msaa == xenos::MsaaSamples::k2X) {
    if (select == S::k2) select = S::k0;
    if (select == S::k3) select = S::k1;
    if (select > S::k01) select = S::k01;
    if (depth && select == S::k01) select = S::k0;
  } else {
    select = S::k0;
  }
  return select;
}

void SelectedSamples(xenos::CopySampleSelect select, uint32_t indices[4], uint32_t& count) {
  using S = xenos::CopySampleSelect;
  if (select <= S::k3) {
    indices[0] = uint32_t(select);
    count = 1;
  } else if (select == S::k01 || select == S::k23) {
    indices[0] = select == S::k01 ? 0 : 2;
    indices[1] = indices[0] + 1;
    count = 2;
  } else {
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
    indices[3] = 3;
    count = 4;
  }
}

void UnpackResolveColor(xenos::ColorRenderTargetFormat format, uint64_t packed, float color[4]) {
  RX_USE(resolve_color_unpack);
  UnpackColor(format, packed, color);
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      color[3] = 0.0f;
      break;
    default:
      break;
  }
}

uint32_t GammaToLinearCode(uint32_t c) {
  uint32_t scale = c < 64 ? 1 : (c < 96 ? 2 : (c < 192 ? 4 : 8));
  uint32_t offset = c < 64 ? 0 : (c < 96 ? 64 : (c < 192 ? 256 : 1024));
  uint32_t n = c * scale - offset;
  return n + ((n * scale) >> 10);
}

float AverageGamma(float sum, float weight) {
  RX_USE(resolve_pwl_gamma_average);
  uint32_t m = uint32_t(sum * weight);
  uint32_t c = m < 64 ? m : (m < 128 ? (m + 64) >> 1 :
               (m < 512 ? (m + 256) >> 2 : (m + 1024) >> 3));
  c = std::min(c, 255u);
  uint32_t n = GammaToLinearCode(c);
  if (n == 1023) return 1.0f;
  return float((n << 14) | (n << 4) | (n >> 6)) * (1.0f / 16777216.0f);
}

void ResolveColorAt(const Edram& edram, const EdramSurface& surface,
                    xenos::ColorRenderTargetFormat format, uint32_t x, uint32_t y,
                    xenos::CopySampleSelect select, int32_t exp_bias, bool swap, bool scalar,
                    float result[4]) {
  uint32_t sample_indices[4], count;
  SelectedSamples(select, sample_indices, count);
  bool gamma = format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  result[0] = result[1] = result[2] = result[3] = 0.0f;
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t packed = edram.ReadSample(surface, x, y, sample_indices[i]);
    float sample[4];
    UnpackResolveColor(format, packed, sample);
    if (gamma) {
      RX_USE(resolve_pwl_gamma_average);
      for (uint32_t c = 0; c < 3; ++c) {
        sample[c] = float(GammaToLinearCode((uint32_t(packed) >> (8 * c)) & 255u));
      }
    }
    for (uint32_t c = 0; c < 4; ++c) result[c] += sample[c];
  }
  RX_USE(resolve_msaa_average);
  RX_USE(resolve_exp_bias);
  float weight = 1.0f / float(count);
  float exp_factor = std::ldexp(1.0f, exp_bias);
  for (uint32_t c = 0; c < 4; ++c) {
    result[c] = gamma && c < 3 ? AverageGamma(result[c], weight) * exp_factor
                               : result[c] * (weight * exp_factor);
  }
  if (swap) {
    RX_USE(resolve_red_blue_swap);
    if (scalar) {
      result[0] = result[3];
    } else {
      std::swap(result[0], result[2]);
    }
  }
}

float Clamp(float value, float low, float high) {
  if (std::isnan(value)) return 0.0f;
  return std::min(std::max(value, low), high);
}

uint32_t PackFixed(float value, uint32_t bits, xenos::SurfaceNumberFormat number) {
  RX_USE(resolve_color_pack);
  uint32_t mask = (1u << bits) - 1u;
  if (number == xenos::SurfaceNumberFormat::kSignedRepeatingFraction) {
    float max = float((1u << (bits - 1)) - 1u);
    float clamped = Clamp(value, -1.0f, 1.0f);
    int32_t packed = int32_t(clamped * max + (value >= 0.0f ? 0.5f : -0.5f));
    return uint32_t(packed) & mask;
  }
  if (number == xenos::SurfaceNumberFormat::kUnsignedInteger) {
    return uint32_t(Clamp(value, 0.0f, float(mask)) + 0.5f) & mask;
  }
  if (number == xenos::SurfaceNumberFormat::kSignedInteger) {
    int32_t low = -int32_t(1u << (bits - 1));
    int32_t high = int32_t((1u << (bits - 1)) - 1u);
    float clamped = Clamp(value, float(low), float(high));
    int32_t packed = int32_t(clamped + (value >= 0.0f ? 0.5f : -0.5f));
    return uint32_t(packed) & mask;
  }
  return uint32_t(Clamp(value, 0.0f, 1.0f) * float(mask) + 0.5f) & mask;
}

uint16_t Float32ToHalf(float value) {
  uint32_t bits = AsU32(value);
  uint16_t sign = uint16_t((bits >> 16) & 0x8000u);
  uint32_t exponent = (bits >> 23) & 255u;
  uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 255) return sign | 0x7C00u | (mantissa ? 0x200u : 0);
  if (!(bits & 0x7FFFFFFFu)) return sign;
  int32_t half_exponent = int32_t(exponent) - 127 + 15;
  uint32_t significand = mantissa | 0x800000u;
  if (half_exponent <= 0) {
    uint32_t shift = 126u - exponent;
    if (shift >= 25) return sign;
    uint32_t result = significand >> shift;
    uint32_t remainder = significand & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (result & 1u))) ++result;
    return sign | uint16_t(result);
  }
  uint32_t rounded = significand >> 13;
  uint32_t remainder = significand & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) ++rounded;
  if (rounded == 0x800u) {
    rounded = 0;
    ++half_exponent;
  }
  if (half_exponent > 30) return sign | 0x7C00u;
  return sign | uint16_t((uint32_t(half_exponent) << 10) | (rounded & 0x3FFu));
}

void Put16(std::array<uint8_t, 16>& bytes, uint32_t offset, uint16_t value) {
  bytes[offset] = uint8_t(value);
  bytes[offset + 1] = uint8_t(value >> 8);
}
void Put32(std::array<uint8_t, 16>& bytes, uint32_t offset, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) bytes[offset + i] = uint8_t(value >> (8 * i));
}

std::array<uint8_t, 16> PackDestination(const DestInfo& info, const float color[4],
                                         xenos::SurfaceNumberFormat number) {
  std::array<uint8_t, 16> bytes{};
  auto fixed4 = [&](uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
    return PackFixed(color[0], b0, number) | (PackFixed(color[1], b1, number) << b0) |
           (PackFixed(color[2], b2, number) << (b0 + b1)) |
           (PackFixed(color[3], b3, number) << (b0 + b1 + b2));
  };
  switch (info.format) {
    case xenos::ColorFormat::k_8: bytes[0] = uint8_t(PackFixed(color[0], 8, number)); break;
    case xenos::ColorFormat::k_1_5_5_5:
      Put16(bytes, 0, uint16_t(fixed4(5, 5, 5, 1)));
      break;
    case xenos::ColorFormat::k_5_6_5: {
      uint16_t value = uint16_t(PackFixed(color[0], 5, number) |
                                (PackFixed(color[1], 6, number) << 5) |
                                (PackFixed(color[2], 5, number) << 11));
      Put16(bytes, 0, value);
      break;
    }
    case xenos::ColorFormat::k_6_5_5: {
      uint16_t value = uint16_t(PackFixed(color[0], 5, number) |
                                (PackFixed(color[1], 5, number) << 5) |
                                (PackFixed(color[2], 6, number) << 11));
      Put16(bytes, 0, value);
      break;
    }
    case xenos::ColorFormat::k_8_8_8_8: Put32(bytes, 0, fixed4(8, 8, 8, 8)); break;
    case xenos::ColorFormat::k_2_10_10_10: Put32(bytes, 0, fixed4(10, 10, 10, 2)); break;
    case xenos::ColorFormat::k_8_8:
      Put16(bytes, 0, uint16_t(PackFixed(color[0], 8, number) |
                               (PackFixed(color[1], 8, number) << 8)));
      break;
    case xenos::ColorFormat::k_4_4_4_4: Put16(bytes, 0, uint16_t(fixed4(4, 4, 4, 4))); break;
    case xenos::ColorFormat::k_10_11_11:
      Put32(bytes, 0, PackFixed(color[0], 11, number) | (PackFixed(color[1], 11, number) << 11) |
                          (PackFixed(color[2], 10, number) << 22));
      break;
    case xenos::ColorFormat::k_11_11_10:
      Put32(bytes, 0, PackFixed(color[0], 10, number) | (PackFixed(color[1], 11, number) << 10) |
                          (PackFixed(color[2], 11, number) << 21));
      break;
    case xenos::ColorFormat::k_16:
      Put16(bytes, 0, uint16_t(PackFixed(color[0], 16, number)));
      break;
    case xenos::ColorFormat::k_16_16:
      Put32(bytes, 0, PackFixed(color[0], 16, number) | (PackFixed(color[1], 16, number) << 16));
      break;
    case xenos::ColorFormat::k_16_16_16_16:
      Put32(bytes, 0, PackFixed(color[0], 16, number) | (PackFixed(color[1], 16, number) << 16));
      Put32(bytes, 4, PackFixed(color[2], 16, number) | (PackFixed(color[3], 16, number) << 16));
      break;
    case xenos::ColorFormat::k_16_FLOAT:
      Put16(bytes, 0, Float32ToHalf(color[0]));
      break;
    case xenos::ColorFormat::k_16_16_FLOAT:
      Put16(bytes, 0, Float32ToHalf(color[0]));
      Put16(bytes, 2, Float32ToHalf(color[1]));
      break;
    case xenos::ColorFormat::k_16_16_16_16_FLOAT:
      for (uint32_t i = 0; i < 4; ++i) Put16(bytes, 2 * i, Float32ToHalf(color[i]));
      break;
    case xenos::ColorFormat::k_32_FLOAT: Put32(bytes, 0, AsU32(color[0])); break;
    case xenos::ColorFormat::k_32_32_FLOAT:
      Put32(bytes, 0, AsU32(color[0]));
      Put32(bytes, 4, AsU32(color[1]));
      break;
    case xenos::ColorFormat::k_32_32_32_32_FLOAT:
      for (uint32_t i = 0; i < 4; ++i) Put32(bytes, 4 * i, AsU32(color[i]));
      break;
    default:
      break;
  }
  return bytes;
}

void ApplyEndian(std::array<uint8_t, 16>& bytes, uint32_t size, xenos::Endian128 endian) {
  RX_USE(resolve_endian_128);
  auto reverse = [&](uint32_t count) { std::reverse(bytes.begin(), bytes.begin() + count); };
  switch (endian) {
    case xenos::Endian128::k8in16:
      if (size >= 2) {
        for (uint32_t i = 0; i < size; i += 2) std::swap(bytes[i], bytes[i + 1]);
      }
      break;
    case xenos::Endian128::k8in32:
      if (size >= 4) {
        for (uint32_t i = 0; i < size; i += 4) {
          std::reverse(bytes.begin() + i, bytes.begin() + i + 4);
        }
      }
      break;
    case xenos::Endian128::k16in32:
      if (size >= 4) {
        for (uint32_t i = 0; i < size; i += 4) {
          std::swap(bytes[i], bytes[i + 2]);
          std::swap(bytes[i + 1], bytes[i + 3]);
        }
      }
      break;
    case xenos::Endian128::k8in64:
      if (size >= 8) reverse(8);
      break;
    case xenos::Endian128::k8in128:
      if (size >= 16) reverse(16);
      break;
    case xenos::Endian128::kNone:
      break;
  }
}

uint32_t Align32(uint32_t value) { return (value + 31u) & ~31u; }

uint32_t TiledAddress(uint32_t x, uint32_t y, uint32_t z, bool array,
                      uint32_t pitch, uint32_t height, uint32_t bytes) {
  RX_USE(resolve_tiled_destination);
  return array ? tex::TiledOffset3D(x, y, z, pitch, height, bytes)
               : tex::TiledOffset2D(x, y, pitch, bytes);
}

void Note(DrawStats& stats, const char* text) { stats.notes.emplace_back(text); }

}  // namespace

bool Resolve(const RegisterFile& regs, Edram& edram, Memory& memory,
             ResolveRecord* record, DrawStats& stats) {
  ResolveRecord local;
  auto control = regs.Get<reg::RB_COPY_CONTROL>();
  auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto dest_info = regs.Get<reg::RB_COPY_DEST_INFO>();
  auto dest_pitch_reg = regs.Get<reg::RB_COPY_DEST_PITCH>();
  bool is_depth = control.copy_src_select >= 4;
  if (control.copy_command != xenos::CopyCommand::kRaw &&
      control.copy_command != xenos::CopyCommand::kConvert) {
    Note(stats, "resolve: unsupported copy command");
    return false;
  }
  if (surface_info.msaa_samples > xenos::MsaaSamples::k4X) {
    Note(stats, "resolve: unsupported MSAA mode");
    return false;
  }
  RX_USE(resolve_rectangle_and_scissor);
  auto fetch = regs.GetVertexFetch(0);
  if (fetch.type != xenos::FetchConstantType::kVertex || fetch.size != 6) {
    Note(stats, "resolve: vertex fetch is not three float2 vertices");
    return false;
  }
  float half_pixel = regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero
                         ? 0.5f
                         : 0.0f;
  int32_t fixed[6];
  for (uint32_t i = 0; i < 6; ++i) {
    uint32_t bits = ReadGuestWordLittleEndian(memory, fetch.address * 4 + i * 4);
    float value = AsF32(xenos::GpuSwap(bits, fetch.endian));
    fixed[i] = int32_t(std::lround((value + half_pixel) * 256.0f));
  }
  int32_t x0 = std::min({fixed[0], fixed[2], fixed[4]});
  int32_t y0 = std::min({fixed[1], fixed[3], fixed[5]});
  int32_t x1 = std::max({fixed[0], fixed[2], fixed[4]});
  int32_t y1 = std::max({fixed[1], fixed[3], fixed[5]});
  x0 = (x0 + 127) >> 8;
  y0 = (y0 + 127) >> 8;
  x1 = (x1 + 127) >> 8;
  y1 = (y1 + 127) >> 8;
  auto window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
  if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable) {
    x0 += window_offset.window_x_offset;
    x1 += window_offset.window_x_offset;
    y0 += window_offset.window_y_offset;
    y1 += window_offset.window_y_offset;
  }
  auto window_tl = regs.Get<reg::PA_SC_WINDOW_SCISSOR_TL>();
  auto window_br = regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>();
  int32_t sx0 = window_tl.tl_x, sy0 = window_tl.tl_y;
  int32_t sx1 = window_br.br_x, sy1 = window_br.br_y;
  if (!window_tl.window_offset_disable) {
    sx0 += window_offset.window_x_offset;
    sx1 += window_offset.window_x_offset;
    sy0 += window_offset.window_y_offset;
    sy1 += window_offset.window_y_offset;
  }
  auto screen_tl = regs.Get<reg::PA_SC_SCREEN_SCISSOR_TL>();
  auto screen_br = regs.Get<reg::PA_SC_SCREEN_SCISSOR_BR>();
  sx0 = std::max(sx0, int32_t(screen_tl.tl_x));
  sy0 = std::max(sy0, int32_t(screen_tl.tl_y));
  sx1 = std::min(sx1, int32_t(screen_br.br_x));
  sy1 = std::min(sy1, int32_t(screen_br.br_y));
  sx0 = std::max(sx0, 0);
  sy0 = std::max(sy0, 0);
  sx1 = std::max(sx1, sx0);
  sy1 = std::max(sy1, sy0);
  x0 = std::clamp(x0, sx0, sx1);
  x1 = std::clamp(x1, sx0, sx1);
  y0 = std::clamp(y0, sy0, sy1);
  y1 = std::clamp(y1, sy0, sy1);
  x0 &= ~7;
  y0 &= ~7;
  x1 = (x1 + 7) & ~7;
  y1 = (y1 + 7) & ~7;
  int32_t pitch_clamp = surface_info.surface_pitch & ~7u;
  x0 = std::min(x0, pitch_clamp);
  x1 = std::min(x1, pitch_clamp);
  if (y1 - y0 > int32_t(xenos::kMaxResolveSize)) y1 = y0 + xenos::kMaxResolveSize;

  EdramSurface source;
  source.msaa = surface_info.msaa_samples;
  source.is_depth = is_depth;
  xenos::ColorRenderTargetFormat color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
  xenos::DepthRenderTargetFormat depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  if (is_depth) {
    auto info = regs.Get<reg::RB_DEPTH_INFO>();
    depth_format = info.depth_format;
    source.base_tile = info.depth_base & (kEdramTiles - 1);
    source.pitch_tiles = EdramPitchTiles(surface_info.surface_pitch, source.msaa, false);
    local.src_format = uint32_t(depth_format);
  } else {
    auto info = regs.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[control.copy_src_select]);
    color_format = info.color_format;
    source.base_tile = info.color_base & (kEdramTiles - 1);
    source.is_64bpp = IsColorFormat64bpp(color_format);
    source.pitch_tiles = EdramPitchTiles(surface_info.surface_pitch, source.msaa,
                                         source.is_64bpp);
    local.src_format = uint32_t(color_format);
  }
  local.src_base_tile = source.base_tile;
  local.src_pitch_tiles = source.pitch_tiles;
  local.src_is_depth = is_depth;
  local.sample_select = uint32_t(SanitizeSampleSelect(control.copy_sample_select,
                                                       source.msaa, is_depth));
  local.x0 = uint32_t(x0);
  local.y0 = uint32_t(y0);
  local.width = x0 < x1 ? uint32_t(x1 - x0) : 0;
  local.height = y0 < y1 ? uint32_t(y1 - y0) : 0;
  local.exp_bias = is_depth ? 0 : dest_info.copy_dest_exp_bias;
  local.dest_pitch = Align32(dest_pitch_reg.copy_dest_pitch);
  local.dest_height = Align32(dest_pitch_reg.copy_dest_height);
  local.dest_tiled = true;
  local.clear_color_lo = regs[XE_GPU_REG_RB_COLOR_CLEAR_LO];
  local.clear_color_hi = regs[XE_GPU_REG_RB_COLOR_CLEAR];
  local.clear_depth = regs[XE_GPU_REG_RB_DEPTH_CLEAR];
  local.cleared_color = !is_depth && control.color_clear_enable;
  local.cleared_depth = control.depth_clear_enable;

  DestInfo dest;
  xenos::ColorFormat requested_dest = is_depth
      ? (depth_format == xenos::DepthRenderTargetFormat::kD24S8
             ? xenos::ColorFormat::k_32_FLOAT
             : xenos::ColorFormat::k_32_FLOAT)
      : xenos::ColorFormat(dest_info.copy_dest_format);
  if (is_depth) {
    dest.format = xenos::ColorFormat::k_32_FLOAT;
    dest.bytes = 4;
    dest.bytes_log2 = 2;
  } else if (!GetDestInfo(requested_dest, dest)) {
    Note(stats, "resolve: unsupported destination format");
    return false;
  }
  local.dest_format = is_depth
      ? uint32_t(depth_format == xenos::DepthRenderTargetFormat::kD24S8
                     ? xenos::TextureFormat::k_24_8
                     : xenos::TextureFormat::k_24_8_FLOAT)
      : uint32_t(requested_dest);
  uint32_t dest_base = regs[XE_GPU_REG_RB_COPY_DEST_BASE];
  uint32_t dest_addr_x0 = uint32_t(x0), dest_addr_y0 = uint32_t(y0);
  if (!dest_info.copy_dest_array) {
    uint32_t macro_bytes_log2 = 10 + dest.bytes_log2;
    uint32_t phase = (dest_base & (xenos::kTextureSubresourceAlignmentBytes - 1)) >>
                     macro_bytes_log2;
    dest_base -= phase << macro_bytes_log2;
    dest_addr_x0 += phase << 5;
  }
  uint32_t x_mask = (1u << xenos::GetTextureTiledXBaseGranularityLog2(
                        dest_info.copy_dest_array, dest.bytes_log2)) - 1u;
  uint32_t y_mask = (1u << xenos::GetTextureTiledYBaseGranularityLog2(
                        dest_info.copy_dest_array, dest.bytes_log2)) - 1u;
  uint32_t offset_x = dest_addr_x0 & x_mask, offset_y = dest_addr_y0 & y_mask;
  dest_base += TiledAddress(dest_addr_x0 & ~x_mask, dest_addr_y0 & ~y_mask, 0,
                            dest_info.copy_dest_array, local.dest_pitch,
                            local.dest_height, dest.bytes);
  bool raw_copy = !is_depth && !local.exp_bias &&
                  color_format != xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA &&
                  xenos::IsSingleCopySampleSelected(
                      xenos::CopySampleSelect(local.sample_select)) &&
                  xenos::IsColorResolveFormatBitwiseEquivalent(color_format, dest.format) &&
                  ColorResolveNumberFormatMatches(dest.format, dest_info.copy_dest_number);
  if (raw_copy && dest.bytes != (source.is_64bpp ? 8u : 4u)) {
    Note(stats, "resolve: raw copy element size differs from the source");
    return false;
  }
  uint32_t dest_min = UINT32_MAX, dest_max = 0;
  if (local.width && local.height) {
    for (uint32_t dy = 0; dy < local.height; ++dy) {
      for (uint32_t dx = 0; dx < local.width; ++dx) {
        uint32_t x = uint32_t(x0) + dx, y = uint32_t(y0) + dy;
        std::array<uint8_t, 16> bytes{};
        if (is_depth || raw_copy) {
          RX_USE(is_depth ? resolve_depth_raw : resolve_raw_copy);
          uint32_t sample = uint32_t(xenos::CopySampleSelect(local.sample_select));
          uint64_t packed = edram.ReadSample(source, x, y, sample);
          if (raw_copy && dest_info.copy_dest_swap) {
            uint32_t low = uint32_t(packed), high = uint32_t(packed >> 32);
            if (color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8) {
              low = (low & ~0xFF00FFu) | ((low & 0xFFu) << 16) | ((low >> 16) & 0xFFu);
            } else if (color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10 ||
                       color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT ||
                       color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10 ||
                       color_format == xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16) {
              low = (low & ~0x3FF003FFu) | ((low & 0x3FFu) << 20) |
                    ((low >> 20) & 0x3FFu);
            } else if (color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16 ||
                       color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT) {
              low = (low & 0xFFFF0000u) | (high & 0xFFFFu);
              high = (high & 0xFFFF0000u) | (packed & 0xFFFFu);
            }
            packed = uint64_t(low) | (uint64_t(high) << 32);
          }
          Put32(bytes, 0, uint32_t(packed));
          if (dest.bytes == 8) Put32(bytes, 4, uint32_t(packed >> 32));
        } else {
          float color[4];
          bool scalar = dest.format == xenos::ColorFormat::k_8;
          bool source_alpha = scalar && dest_info.copy_dest_swap;
          ResolveColorAt(edram, source, color_format, x, y,
                         xenos::CopySampleSelect(local.sample_select), local.exp_bias,
                         dest_info.copy_dest_swap, false, color);
          if (source_alpha) color[0] = color[3];
          bytes = PackDestination(dest, color, dest_info.copy_dest_number);
        }
        ApplyEndian(bytes, dest.bytes, dest_info.copy_dest_endian);
        uint32_t address = dest_base + TiledAddress(offset_x + dx, offset_y + dy,
                            dest_info.copy_dest_slice, dest_info.copy_dest_array,
                            local.dest_pitch, local.dest_height, dest.bytes);
        memory.Write(address, bytes.data(), dest.bytes);
        dest_min = std::min(dest_min, address);
        dest_max = std::max(dest_max, address + dest.bytes);
      }
    }
  }
  if (dest_max) {
    local.dest_base = dest_min;
    local.dest_bytes = dest_max - dest_min;
  }

  if (local.cleared_color || local.cleared_depth) {
    RX_USE(resolve_clear);
    for (uint32_t dy = 0; dy < local.height; ++dy) {
      for (uint32_t dx = 0; dx < local.width; ++dx) {
        uint32_t x = uint32_t(x0) + dx, y = uint32_t(y0) + dy;
        for (uint32_t s = 0; s < SampleCount(source.msaa); ++s) {
          if (local.cleared_color) {
            uint64_t value = source.is_64bpp
                ? (uint64_t(local.clear_color_lo) | (uint64_t(local.clear_color_hi) << 32))
                : local.clear_color_hi;
            edram.WriteSample(source, x, y, s, value);
          }
          if (local.cleared_depth) {
            EdramSurface depth = source;
            if (!is_depth) {
              auto info = regs.Get<reg::RB_DEPTH_INFO>();
              depth.base_tile = info.depth_base & (kEdramTiles - 1);
              depth.pitch_tiles = EdramPitchTiles(surface_info.surface_pitch, source.msaa, false);
              depth.is_depth = true;
              depth.is_64bpp = false;
            }
            edram.WriteSample(depth, x, y, s, local.clear_depth);
          }
        }
      }
    }
  }
  if (record) *record = local;
  return true;
}

std::vector<float> ReadbackColorF32(const Edram& edram, const RegisterFile& regs,
                                    uint32_t rt_index, uint32_t width, uint32_t height) {
  std::vector<float> result;
  if (rt_index >= 4) return result;
  result.resize(size_t(width) * height * 4);
  auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[rt_index]);
  EdramSurface surface;
  surface.base_tile = info.color_base & (kEdramTiles - 1);
  surface.msaa = surface_info.msaa_samples;
  surface.is_64bpp = IsColorFormat64bpp(info.color_format);
  surface.pitch_tiles = EdramPitchTiles(surface_info.surface_pitch, surface.msaa,
                                         surface.is_64bpp);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      ResolveColorAt(edram, surface, info.color_format, x, y, xenos::CopySampleSelect::k0,
                     0, false, false, &result[(size_t(y) * width + x) * 4]);
    }
  }
  return result;
}

std::vector<uint32_t> ReadbackDepthStencilRaw(const Edram& edram, const RegisterFile& regs,
                                              uint32_t width, uint32_t height) {
  std::vector<uint32_t> result(size_t(width) * height);
  auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto info = regs.Get<reg::RB_DEPTH_INFO>();
  EdramSurface surface;
  surface.base_tile = info.depth_base & (kEdramTiles - 1);
  surface.pitch_tiles = EdramPitchTiles(surface_info.surface_pitch, surface_info.msaa_samples,
                                         false);
  surface.msaa = surface_info.msaa_samples;
  surface.is_depth = true;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      result[size_t(y) * width + x] = uint32_t(edram.ReadSample(surface, x, y, 0));
    }
  }
  return result;
}

}  // namespace refxenos
