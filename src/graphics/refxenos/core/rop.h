// refxenos render backend output merger.
#pragma once

#include "core/common.h"
#include "core/edram.h"

namespace refxenos {

struct RopColorTarget {
  EdramSurface surface;
  xenos::ColorRenderTargetFormat format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
  int32_t exp_bias = 0;
  reg::RB_BLENDCONTROL blend = {};
};

struct RopState {
  xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X;
  xenos::EdramMode edram_mode = xenos::EdramMode::kNoOperation;
  RopColorTarget color[4];
  uint32_t color_write_mask = 0;
  float blend_constant[4] = {};
  reg::RB_COLORCONTROL color_control = {};
  float alpha_ref = 0.0f;
  EdramSurface depth_surface;
  xenos::DepthRenderTargetFormat depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  reg::RB_DEPTHCONTROL depth_control = {};
  reg::RB_STENCILREFMASK stencil[2] = {};
};

RopState DecodeRopState(const RegisterFile& regs);
void RopQuad(const RopState& state, Edram& edram, const PixelQuad& quad,
             const PixelOut out[4], DrawStats& stats);

}  // namespace refxenos
