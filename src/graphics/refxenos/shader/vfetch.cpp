// Vertex fetch (vfetch_full / vfetch_mini) for the refxenos interpreter.

#include <cmath>
#include <cstring>

#include <rex/math.h>

#include "shader/internal.h"

namespace refxenos {

RX_DEFINE_RULE(xenos_vfetch_index, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:74 "
               "(index = floor(x) or, with is_index_rounded, floor(x + 0.5); "
               "converted with ftoi: NaN -> 0, saturating)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_mini_reuse, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_translator.cc:1002 and "
               "dxbc_shader_translator_fetch.cc:60 (vfetch_mini reuses the "
               "address (fetch constant, index register value at the time of the "
               "vfetch_full, stride, rounding) of the last vfetch_full; format, "
               "offset, signedness, normalization, exp_adjust are its own)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_address, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:120 "
               "(byte address = fetch.address*4 + (index*stride + offset + "
               "word)*4, 32-bit wrapping, upper bits dropped: & 0x1FFFFFFF)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_out_of_bounds, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:145 "
               "(words at or past the end of the fetch buffer (size in words) "
               "read as 0, unsigned compare against the exclusive end)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_endian, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:255 "
               "(each 32-bit word swapped per the fetch constant endian: 8in16, "
               "8in32, 16in32)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_packed_layout, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:212 "
               "(8_8_8_8 x@0; 2_10_10_10 10/10/10/2; 10_11_11 11@0 11@11 10@22; "
               "11_11_10 10@0 11@10 11@21; 16_16 / 16_16_16_16 low half first)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_normalize, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:370 "
               "(packed: x * float(1 / scale_inv), scale_inv = 2^(w-1)-1 signed "
               "(+0.5 for NoZero), 2^w-1 unsigned; signed ZeroClampMinusOne: "
               "max(r, -1); NoZero: + 0.5*scale; each step rounded to f32)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_normalize_32, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:307 "
               "(32-bit: float(x) * (1/2147483647) signed (max -1), "
               "* (1/2147483647.5) + (0.5/2147483647.5) NoZero, "
               "* (1/4294967295) unsigned)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_half, 1,
               "assumed: 16-bit float vertex components decode as Xenos extended "
               "range halves without inf/NaN, denormals flushed "
               "(rex::xenos_half_to_float as the SDK interpreter; "
               "spirv_shader_translator_fetch.cc:258 FIXME)",
               true,
               "VS: vfetch r1, r0.x, vf0 format 16_16_16_16_FLOAT with halves "
               "0x7C00, 0xFC00, 0x7E00, 0x0001, 0x7BFF; export r1 to an "
               "interpolator, PS writes it to a 32_32_32_32_FLOAT RT; read bits");
RX_DEFINE_RULE(xenos_vfetch_exp_adjust, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:514 "
               "(result * 2^exp_adjust for every format)",
               false, "");
RX_DEFINE_RULE(xenos_vfetch_missing_components, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:499 "
               "(components the format lacks read 0; destination swizzle 0/1 "
               "constants apply)",
               false, "");

namespace shader_internal {

namespace {

int32_t FtoI(float f) {
  if (std::isnan(f)) return 0;
  if (f >= 2147483648.0f) return INT32_MAX;
  if (f <= -2147483648.0f) return INT32_MIN;
  return int32_t(f);
}

uint32_t LoadWord(const Memory& memory, uint32_t byte_addr) {
  uint8_t b[4];
  memory.Read(byte_addr & 0x1FFFFFFFu, b, 4);
  // The GPU reads little-endian dwords; the fetch constant endian then swaps.
  return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) |
         (uint32_t(b[3]) << 24);
}

}  // namespace

VertexFetchAddress ComputeVertexFetchAddress(
    const ucode::VertexFetchInstruction& full, float index_value,
    const RegisterFile& regs) {
  VertexFetchAddress a;
  RX_USE(xenos_vfetch_index);
  float f = full.is_index_rounded() ? std::floor(index_value + 0.5f)
                                    : std::floor(index_value);
  a.index = FtoI(f);
  xenos::xe_gpu_vertex_fetch_t fc = regs.GetVertexFetch(full.fetch_constant_index());
  a.buffer_base = uint32_t(fc.address) << 2;
  a.buffer_end = a.buffer_base + (uint32_t(fc.size) << 2);
  a.endian = uint32_t(fc.endian);
  a.element_base = a.buffer_base + uint32_t(a.index) * full.stride() * 4u;
  a.valid = true;
  return a;
}

bool FetchVertex(const ucode::VertexFetchInstruction& op,
                 const VertexFetchAddress& addr, const Memory& memory,
                 float out[4], std::string* note) {
  using F = xenos::VertexFormat;
  out[0] = out[1] = out[2] = out[3] = 0.0f;
  RX_USE(xenos_vfetch_missing_components);
  F format = op.data_format();
  uint32_t word_count;
  switch (format) {
    case F::k_8_8_8_8: case F::k_2_10_10_10: case F::k_10_11_11:
    case F::k_11_11_10: case F::k_16_16: case F::k_16_16_FLOAT: case F::k_32:
    case F::k_32_FLOAT:
      word_count = 1;
      break;
    case F::k_16_16_16_16: case F::k_16_16_16_16_FLOAT: case F::k_32_32:
    case F::k_32_32_FLOAT:
      word_count = 2;
      break;
    case F::k_32_32_32_FLOAT:
      word_count = 3;
      break;
    case F::k_32_32_32_32: case F::k_32_32_32_32_FLOAT:
      word_count = 4;
      break;
    default:
      if (note) *note = "vfetch: unsupported vertex format " + std::to_string(uint32_t(format));
      return false;
  }
  RX_USE(xenos_vfetch_address);
  RX_USE(xenos_vfetch_out_of_bounds);
  RX_USE(xenos_vfetch_endian);
  uint32_t words[4] = {0, 0, 0, 0};
  for (uint32_t w = 0; w < word_count; ++w) {
    uint32_t byte_addr =
        addr.element_base + (uint32_t(op.offset()) + w) * 4u;
    if (byte_addr < addr.buffer_end) {
      words[w] = xenos::GpuSwap(LoadWord(memory, byte_addr),
                                xenos::Endian(addr.endian));
    }
  }

  bool is_signed = op.is_signed();
  bool normalized = op.is_normalized();
  bool no_zero =
      op.signed_rf_mode() == xenos::SignedRepeatingFractionMode::kNoZero;

  auto packed = [&](const int widths[4], const int offsets[4],
                    const uint32_t* src_words, int count) {
    RX_USE(xenos_vfetch_packed_layout);
    for (int i = 0; i < count; ++i) {
      uint32_t word = src_words[i];
      int w = widths[i];
      uint32_t raw = (word >> offsets[i]) & ((w == 32) ? 0xFFFFFFFFu : ((1u << w) - 1));
      float r;
      if (is_signed) {
        int32_t sv = int32_t(raw << (32 - w)) >> (32 - w);
        r = float(sv);
      } else {
        r = float(raw);
      }
      if (normalized) {
        RX_USE(xenos_vfetch_normalize);
        float scale_inv;
        if (is_signed) {
          scale_inv = float((uint32_t(1) << (w - 1)) - 1);
          if (no_zero) scale_inv += 0.5f;
        } else {
          scale_inv = float((uint32_t(1) << w) - 1);
        }
        float scale = 1.0f / scale_inv;
        r = r * scale;
        if (is_signed) {
          if (no_zero) {
            r = r + 0.5f * scale;
          } else {
            r = r >= -1.0f ? r : -1.0f;
          }
        }
      }
      out[i] = r;
    }
  };

  switch (format) {
    case F::k_8_8_8_8: {
      int wd[4] = {8, 8, 8, 8}, of[4] = {0, 8, 16, 24};
      uint32_t sw[4] = {words[0], words[0], words[0], words[0]};
      packed(wd, of, sw, 4);
    } break;
    case F::k_2_10_10_10: {
      int wd[4] = {10, 10, 10, 2}, of[4] = {0, 10, 20, 30};
      uint32_t sw[4] = {words[0], words[0], words[0], words[0]};
      packed(wd, of, sw, 4);
    } break;
    case F::k_10_11_11: {
      int wd[4] = {11, 11, 10, 0}, of[4] = {0, 11, 22, 0};
      uint32_t sw[4] = {words[0], words[0], words[0], 0};
      packed(wd, of, sw, 3);
    } break;
    case F::k_11_11_10: {
      int wd[4] = {10, 11, 11, 0}, of[4] = {0, 10, 21, 0};
      uint32_t sw[4] = {words[0], words[0], words[0], 0};
      packed(wd, of, sw, 3);
    } break;
    case F::k_16_16: {
      int wd[4] = {16, 16, 0, 0}, of[4] = {0, 16, 0, 0};
      uint32_t sw[4] = {words[0], words[0], 0, 0};
      packed(wd, of, sw, 2);
    } break;
    case F::k_16_16_16_16: {
      int wd[4] = {16, 16, 16, 16}, of[4] = {0, 16, 0, 16};
      uint32_t sw[4] = {words[0], words[0], words[1], words[1]};
      packed(wd, of, sw, 4);
    } break;
    case F::k_16_16_FLOAT:
    case F::k_16_16_16_16_FLOAT: {
      RX_USE(xenos_vfetch_half);
      int n = format == F::k_16_16_FLOAT ? 2 : 4;
      for (int i = 0; i < n; ++i) {
        uint16_t h = uint16_t(words[i >> 1] >> ((i & 1) * 16));
        out[i] = rex::xenos_half_to_float(h);
      }
    } break;
    case F::k_32:
    case F::k_32_32:
    case F::k_32_32_32_32: {
      for (uint32_t i = 0; i < word_count; ++i) {
        float r = is_signed ? float(int32_t(words[i])) : float(words[i]);
        if (normalized) {
          RX_USE(xenos_vfetch_normalize_32);
          if (is_signed) {
            if (no_zero) {
              r = r * (1.0f / 2147483647.5f) + (0.5f / 2147483647.5f);
            } else {
              r = r * (1.0f / 2147483647.0f);
              r = r >= -1.0f ? r : -1.0f;
            }
          } else {
            r = r * (1.0f / 4294967295.0f);
          }
        }
        out[i] = r;
      }
    } break;
    default:  // 32-bit floats
      for (uint32_t i = 0; i < word_count; ++i) out[i] = BitsFloat(words[i]);
      break;
  }

  int32_t e = op.exp_adjust();
  if (e) {
    RX_USE(xenos_vfetch_exp_adjust);
    float m = std::ldexp(1.0f, e);
    for (int i = 0; i < 4; ++i) out[i] *= m;
  }
  return true;
}

}  // namespace shader_internal
}  // namespace refxenos
