// refxenos render backend output merger. See core/edram.h for sample order.

#include "core/rop.h"

#include <algorithm>
#include <cmath>

namespace refxenos {
using namespace rex::graphics;  // XE_GPU_REG_* register indices

RX_DEFINE_RULE(rop_surface_decode, 1,
               "include/rex/graphics/registers.h:669-752,779-848; "
               "xenia-canary@cf206a3b src/xenia/gpu/render_target_cache.cc:698-705,741-749",
               false, "");
RX_DEFINE_RULE(rop_alpha_test_compare, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:519-625 "
               "(float alpha and RB_ALPHA_REF; not-equal and always pass NaN)",
               false, "");
RX_DEFINE_RULE(rop_alpha_rt0_written, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:472-517,627-666",
               false, "");
RX_DEFINE_RULE(rop_alpha_to_coverage_threshold, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:4184-4205,4245-4406",
               false, "");
RX_DEFINE_RULE(rop_alpha_before_exp_bias_and_depth_writes, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:594-628,707-757,813-825 "
               "and dxbc_shader_translator_om.cc:2200-2212",
               false, "");
RX_DEFINE_RULE(rop_color_exp_bias, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:813-825 "
               "(multiply all components by 2^RB_COLOR_INFO.color_exp_bias)",
               false, "");
RX_DEFINE_RULE(rop_blend_factors, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:3707-4059",
               false, "");
RX_DEFINE_RULE(rop_blend_min_max_factors, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:4062-4181 "
               "(unlike host Vulkan, factors are applied before min/max)",
               false, "");
RX_DEFINE_RULE(rop_blend_fixed_clamp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/render_target_cache.cc:228-354; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:3673-3705",
               false, "");
RX_DEFINE_RULE(rop_color_write_masks, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/render_target_cache.cc:228-354; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:1128-1151,1161-1271",
               false, "");
RX_DEFINE_RULE(depth_packed_compare, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:809-896",
               false, "");
RX_DEFINE_RULE(stencil_compare_and_ops, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:898-1167",
               false, "");
RX_DEFINE_RULE(stencil_two_sided, 1,
               "include/rex/graphics/registers.h:798-832; "
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:910-982",
               false, "");
RX_DEFINE_RULE(depth_stencil_failure_blocks_depth_not_stencil_op, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_om.cc:995-1045,1169-1187",
               false, "");
RX_DEFINE_RULE(rop_early_late_z_final_values, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator.cc:2655-2712; "
               "src/xenia/gpu/spirv_shader_translator_rb.cc:707-757; early tests defer writes "
               "until after kill/alpha/A2C, which is the final-value ordering used here",
               false, "");

namespace {

uint32_t SampleCount(xenos::MsaaSamples msaa) {
  return msaa == xenos::MsaaSamples::k4X ? 4 : (msaa == xenos::MsaaSamples::k2X ? 2 : 1);
}

bool CompareFloat(xenos::CompareFunction function, float value, float reference) {
  switch (function) {
    case xenos::CompareFunction::kNever: return false;
    case xenos::CompareFunction::kLess: return value < reference;
    case xenos::CompareFunction::kEqual: return value == reference;
    case xenos::CompareFunction::kLessEqual: return value <= reference;
    case xenos::CompareFunction::kGreater: return value > reference;
    case xenos::CompareFunction::kNotEqual: return !(value == reference);
    case xenos::CompareFunction::kGreaterEqual: return value >= reference;
    case xenos::CompareFunction::kAlways: return true;
  }
  return false;
}

bool CompareUint(xenos::CompareFunction function, uint32_t value, uint32_t reference) {
  switch (function) {
    case xenos::CompareFunction::kNever: return false;
    case xenos::CompareFunction::kLess: return value < reference;
    case xenos::CompareFunction::kEqual: return value == reference;
    case xenos::CompareFunction::kLessEqual: return value <= reference;
    case xenos::CompareFunction::kGreater: return value > reference;
    case xenos::CompareFunction::kNotEqual: return value != reference;
    case xenos::CompareFunction::kGreaterEqual: return value >= reference;
    case xenos::CompareFunction::kAlways: return true;
  }
  return false;
}

struct FormatBlendInfo {
  bool fixed_color;
  bool fixed_alpha;
  float color_min, color_max;
  float alpha_min, alpha_max;
};

FormatBlendInfo GetFormatBlendInfo(xenos::ColorRenderTargetFormat format) {
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
      return {true, true, 0.0f, 1.0f, 0.0f, 1.0f};
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      return {false, true, 0.0f, 31.875f, 0.0f, 1.0f};
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      return {true, true, -32.0f, 32767.0f / 1024.0f, -32.0f, 32767.0f / 1024.0f};
    default:
      return {false, false, 0.0f, 0.0f, 0.0f, 0.0f};
  }
}

float ClampBlend(float value, bool fixed, float low, float high) {
  if (!fixed) return value;
  if (std::isnan(value)) return 0.0f;
  return std::min(std::max(value, low), high);
}

float BlendFactorValue(xenos::BlendFactor factor, uint32_t component, bool alpha,
                       const float source[4], const float dest[4], const float constant[4]) {
  auto color_component = [&](const float value[4]) { return alpha ? value[3] : value[component]; };
  switch (factor) {
    case xenos::BlendFactor::kZero: return 0.0f;
    case xenos::BlendFactor::kOne: return 1.0f;
    case xenos::BlendFactor::kSrcColor: return color_component(source);
    case xenos::BlendFactor::kOneMinusSrcColor: return 1.0f - color_component(source);
    case xenos::BlendFactor::kSrcAlpha: return source[3];
    case xenos::BlendFactor::kOneMinusSrcAlpha: return 1.0f - source[3];
    case xenos::BlendFactor::kDstColor: return color_component(dest);
    case xenos::BlendFactor::kOneMinusDstColor: return 1.0f - color_component(dest);
    case xenos::BlendFactor::kDstAlpha: return dest[3];
    case xenos::BlendFactor::kOneMinusDstAlpha: return 1.0f - dest[3];
    case xenos::BlendFactor::kConstantColor: return color_component(constant);
    case xenos::BlendFactor::kOneMinusConstantColor: return 1.0f - color_component(constant);
    case xenos::BlendFactor::kConstantAlpha: return constant[3];
    case xenos::BlendFactor::kOneMinusConstantAlpha: return 1.0f - constant[3];
    case xenos::BlendFactor::kSrcAlphaSaturate: return std::fmin(source[3], 1.0f - dest[3]);
    default: return 1.0f;
  }
}

float ApplyBlendFactor(float value, xenos::BlendFactor factor, uint32_t component, bool alpha,
                       const FormatBlendInfo& info, const float source[4], const float dest[4],
                       const float constant[4]) {
  RX_USE(rop_blend_factors);
  if (factor == xenos::BlendFactor::kZero) return 0.0f;
  float f = BlendFactorValue(factor, component, alpha, source, dest, constant);
  float result = value * f;
  return ClampBlend(result, alpha ? info.fixed_alpha : info.fixed_color,
                    alpha ? info.alpha_min : info.color_min,
                    alpha ? info.alpha_max : info.color_max);
}

float BlendComponent(float source_value, float dest_value, uint32_t component, bool alpha,
                     xenos::BlendFactor source_factor, xenos::BlendFactor dest_factor,
                     xenos::BlendOp op, const FormatBlendInfo& info, const float source[4],
                     const float dest[4], const float constant[4]) {
  float s = ApplyBlendFactor(source_value, source_factor, component, alpha, info, source, dest,
                             constant);
  float d = ApplyBlendFactor(dest_value, dest_factor, component, alpha, info, source, dest,
                             constant);
  float result;
  switch (op) {
    case xenos::BlendOp::kSubtract: result = s - d; break;
    case xenos::BlendOp::kMin:
      RX_USE(rop_blend_min_max_factors);
      result = std::fmin(s, d);
      break;
    case xenos::BlendOp::kMax:
      RX_USE(rop_blend_min_max_factors);
      result = std::fmax(s, d);
      break;
    case xenos::BlendOp::kRevSubtract: result = d - s; break;
    case xenos::BlendOp::kAdd:
    default: result = s + d; break;
  }
  return ClampBlend(result, alpha ? info.fixed_alpha : info.fixed_color,
                    alpha ? info.alpha_min : info.color_min,
                    alpha ? info.alpha_max : info.color_max);
}

bool BlendEnabled(reg::RB_BLENDCONTROL blend) { return blend.value != 0x00010001u; }

void ColorKeepMask(xenos::ColorRenderTargetFormat format, uint32_t write_mask,
                   uint32_t& low, uint32_t& high) {
  low = high = 0;
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      for (uint32_t i = 0; i < 4; ++i) {
        if (!(write_mask & (1u << i))) low |= 0xFFu << (8 * i);
      }
      break;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      for (uint32_t i = 0; i < 3; ++i) {
        if (!(write_mask & (1u << i))) low |= 0x3FFu << (10 * i);
      }
      if (!(write_mask & 8u)) low |= 3u << 30;
      break;
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      if (!(write_mask & 1u)) low |= 0xFFFFu;
      if (!(write_mask & 2u)) low |= 0xFFFF0000u;
      break;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      if (!(write_mask & 1u)) low |= 0xFFFFu;
      if (!(write_mask & 2u)) low |= 0xFFFF0000u;
      if (!(write_mask & 4u)) high |= 0xFFFFu;
      if (!(write_mask & 8u)) high |= 0xFFFF0000u;
      break;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      if (!(write_mask & 1u)) low = UINT32_MAX;
      break;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      if (!(write_mask & 1u)) low = UINT32_MAX;
      if (!(write_mask & 2u)) high = UINT32_MAX;
      break;
    default:
      low = high = UINT32_MAX;
      break;
  }
  if (!write_mask) low = high = UINT32_MAX;
}

uint8_t ApplyStencilOp(xenos::StencilOp op, uint8_t old, uint8_t reference) {
  RX_USE(stencil_compare_and_ops);
  switch (op) {
    case xenos::StencilOp::kZero: return 0;
    case xenos::StencilOp::kReplace: return reference;
    case xenos::StencilOp::kIncrementClamp: return old == 255 ? 255 : old + 1;
    case xenos::StencilOp::kDecrementClamp: return old == 0 ? 0 : old - 1;
    case xenos::StencilOp::kInvert: return uint8_t(~old);
    case xenos::StencilOp::kIncrementWrap: return uint8_t(old + 1);
    case xenos::StencilOp::kDecrementWrap: return uint8_t(old - 1);
    case xenos::StencilOp::kKeep:
    default: return old;
  }
}

}  // namespace

RopState DecodeRopState(const RegisterFile& regs) {
  RX_USE(rop_surface_decode);
  RopState state;
  auto surface = regs.Get<reg::RB_SURFACE_INFO>();
  state.msaa = surface.msaa_samples;
  state.edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  state.color_write_mask = regs[XE_GPU_REG_RB_COLOR_MASK];
  state.color_control = regs.Get<reg::RB_COLORCONTROL>();
  state.alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  state.depth_control = regs.Get<reg::RB_DEPTHCONTROL>();
  state.stencil[0] = regs.Get<reg::RB_STENCILREFMASK>();
  state.stencil[1] = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  state.blend_constant[0] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
  state.blend_constant[1] = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
  state.blend_constant[2] = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
  state.blend_constant[3] = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  for (uint32_t i = 0; i < 4; ++i) {
    auto info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    auto& target = state.color[i];
    target.format = info.color_format;
    target.exp_bias = info.color_exp_bias;
    target.blend = regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[i]);
    target.surface.base_tile = info.color_base & (kEdramTiles - 1);
    target.surface.msaa = state.msaa;
    target.surface.is_64bpp = IsColorFormat64bpp(target.format);
    target.surface.pitch_tiles = EdramPitchTiles(surface.surface_pitch, state.msaa,
                                                  target.surface.is_64bpp);
  }
  auto depth = regs.Get<reg::RB_DEPTH_INFO>();
  state.depth_format = depth.depth_format;
  state.depth_surface.base_tile = depth.depth_base & (kEdramTiles - 1);
  state.depth_surface.pitch_tiles = EdramPitchTiles(surface.surface_pitch, state.msaa, false);
  state.depth_surface.msaa = state.msaa;
  state.depth_surface.is_depth = true;
  return state;
}

void RopQuad(const RopState& state, Edram& edram, const PixelQuad& quad,
             const PixelOut out[4], DrawStats& stats) {
  if (state.edram_mode != xenos::EdramMode::kColorDepth &&
      state.edram_mode != xenos::EdramMode::kDepthOnly) {
    return;
  }
  RX_USE(rop_early_late_z_final_values);
  uint32_t sample_count = SampleCount(state.msaa);
  for (uint32_t p = 0; p < 4; ++p) {
    uint32_t x = uint32_t(quad.x + int32_t(p & 1));
    uint32_t y = uint32_t(quad.y + int32_t(p >> 1));
    uint8_t coverage = quad.coverage[p] & ((1u << sample_count) - 1u);
    if (!coverage || out[p].killed) continue;
    if (state.edram_mode == xenos::EdramMode::kColorDepth &&
        (out[p].color_written_mask & 1u)) {
      RX_USE(rop_alpha_rt0_written);
      if (state.color_control.alpha_test_enable) {
        RX_USE(rop_alpha_test_compare);
        if (!CompareFloat(state.color_control.alpha_func, out[p].color[0][3], state.alpha_ref)) {
          stats.samples_alpha_test_failed += __builtin_popcount(coverage);
          continue;
        }
      }
      if (state.color_control.alpha_to_mask_enable) {
        RX_USE(rop_alpha_to_coverage_threshold);
        RX_USE(rop_alpha_before_exp_bias_and_depth_writes);
        uint32_t offset_index = (y & 1u) | ((x & 1u) << 1);
        uint32_t offset = (state.color_control.value >> (24 + 2 * offset_index)) & 3u;
        static constexpr float kBase4[4] = {0.75f, 0.25f, 0.5f, 1.0f};
        static constexpr float kBase2[2] = {0.5f, 1.0f};
        for (uint32_t s = 0; s < sample_count; ++s) {
          float base = sample_count == 4 ? kBase4[s]
                                         : (sample_count == 2 ? kBase2[s] : 1.0f);
          float scale = 1.0f / float(4 * sample_count);
          float threshold = base - float(offset) * scale;
          if (!(out[p].color[0][3] >= threshold)) coverage &= ~(1u << s);
        }
      }
    }
    if (!coverage) continue;

    uint8_t surviving = coverage;
    if (state.depth_control.z_enable || state.depth_control.stencil_enable) {
      bool back = state.depth_control.backface_enable && !quad.front_face;
      const auto& stencil = state.stencil[back ? 1 : 0];
      if (back) RX_USE(stencil_two_sided);
      for (uint32_t s = 0; s < sample_count; ++s) {
        if (!(coverage & (1u << s))) continue;
        uint32_t old = uint32_t(edram.ReadSample(state.depth_surface, x, y, s));
        float depth = out[p].depth_written ? out[p].depth : float(quad.sample_z[p][s]);
        uint32_t converted = PackDepthStencil(state.depth_format, depth, 0);
        uint32_t new_depth = converted >> 8;
        uint32_t old_depth = old >> 8;
        bool depth_pass = !state.depth_control.z_enable ||
                          CompareUint(state.depth_control.zfunc, new_depth, old_depth);
        if (state.depth_control.z_enable) RX_USE(depth_packed_compare);
        uint8_t old_stencil = uint8_t(old);
        uint8_t reference = uint8_t(stencil.stencilref);
        bool stencil_pass = !state.depth_control.stencil_enable ||
                            CompareUint(state.depth_control.stencilfunc,
                                        reference & stencil.stencilmask,
                                        old_stencil & stencil.stencilmask);
        if (!depth_pass) ++stats.samples_depth_failed;
        if (!stencil_pass) {
          ++stats.samples_stencil_failed;
          if (depth_pass) RX_USE(depth_stencil_failure_blocks_depth_not_stencil_op);
        }
        xenos::StencilOp stencil_op = xenos::StencilOp::kKeep;
        if (state.depth_control.stencil_enable) {
          RX_USE(stencil_compare_and_ops);
          if (!stencil_pass) {
            stencil_op = back ? state.depth_control.stencilfail_bf : state.depth_control.stencilfail;
          } else if (!depth_pass) {
            stencil_op = back ? state.depth_control.stencilzfail_bf : state.depth_control.stencilzfail;
          } else {
            stencil_op = back ? state.depth_control.stencilzpass_bf : state.depth_control.stencilzpass;
          }
        }
        uint8_t new_stencil = old_stencil;
        if (state.depth_control.stencil_enable) {
          new_stencil = ApplyStencilOp(stencil_op, old_stencil, reference);
          new_stencil = uint8_t((old_stencil & ~stencil.stencilwritemask) |
                                (new_stencil & stencil.stencilwritemask));
        }
        uint32_t stored_depth = (depth_pass && stencil_pass && state.depth_control.z_enable &&
                                 state.depth_control.z_write_enable)
                                    ? new_depth
                                    : old_depth;
        uint32_t result = (stored_depth << 8) | new_stencil;
        if (result != old) edram.WriteSample(state.depth_surface, x, y, s, result);
        if (!depth_pass || !stencil_pass) surviving &= ~(1u << s);
      }
    }

    if (state.edram_mode != xenos::EdramMode::kColorDepth) continue;
    for (uint32_t rt = 0; rt < 4; ++rt) {
      if (!(out[p].color_written_mask & (1u << rt))) continue;
      const auto& target = state.color[rt];
      uint32_t write_mask = (state.color_write_mask >> (4 * rt)) & 15u;
      if (!write_mask) continue;
      RX_USE(rop_color_exp_bias);
      float source[4];
      float exp_factor = std::ldexp(1.0f, target.exp_bias);
      for (uint32_t c = 0; c < 4; ++c) source[c] = out[p].color[rt][c] * exp_factor;
      uint32_t keep_low, keep_high;
      ColorKeepMask(target.format, write_mask, keep_low, keep_high);
      RX_USE(rop_color_write_masks);
      FormatBlendInfo blend_info = GetFormatBlendInfo(target.format);
      bool blending = BlendEnabled(target.blend);
      for (uint32_t s = 0; s < sample_count; ++s) {
        if (!(surviving & (1u << s))) continue;
        uint64_t old = edram.ReadSample(target.surface, x, y, s);
        uint64_t packed;
        if (blending) {
          RX_USE(rop_blend_fixed_clamp);
          float dest[4];
          UnpackColor(target.format, old, dest);
          float result[4];
          float clamped_source[4], clamped_constant[4];
          for (uint32_t c = 0; c < 4; ++c) {
            bool alpha = c == 3;
            clamped_source[c] = ClampBlend(source[c], alpha ? blend_info.fixed_alpha
                                                            : blend_info.fixed_color,
                                           alpha ? blend_info.alpha_min : blend_info.color_min,
                                           alpha ? blend_info.alpha_max : blend_info.color_max);
            clamped_constant[c] = ClampBlend(state.blend_constant[c],
                                             alpha ? blend_info.fixed_alpha
                                                   : blend_info.fixed_color,
                                             alpha ? blend_info.alpha_min : blend_info.color_min,
                                             alpha ? blend_info.alpha_max : blend_info.color_max);
          }
          for (uint32_t c = 0; c < 4; ++c) {
            bool alpha = c == 3;
            result[c] = BlendComponent(clamped_source[c], dest[c], c, alpha,
                                       alpha ? target.blend.alpha_srcblend
                                             : target.blend.color_srcblend,
                                       alpha ? target.blend.alpha_destblend
                                             : target.blend.color_destblend,
                                       alpha ? target.blend.alpha_comb_fcn
                                             : target.blend.color_comb_fcn,
                                       blend_info, clamped_source, dest, clamped_constant);
          }
          packed = PackColor(target.format, result);
        } else {
          packed = PackColor(target.format, source);
        }
        uint64_t keep = uint64_t(keep_low) | (uint64_t(keep_high) << 32);
        edram.WriteSample(target.surface, x, y, s, (old & keep) | (packed & ~keep));
      }
    }
    stats.samples_written += __builtin_popcount(surviving);
  }
}

}  // namespace refxenos
