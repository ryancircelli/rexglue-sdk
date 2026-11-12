// refxenos texture sampling: one tfetch / getCompTexLOD / getWeights /
// getBCF, the slow obvious way. Every rule used is named in Diag::rules_used;
// rules and citations are in ref_rules.cc.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ref_texture.h"

namespace refxenos::tex {

enum class SampleOp : uint32_t {
  kTfetch,         // filtered texel value (after swizzle, sign, exp_adjust)
  kGetCompTexLOD,  // x = computed LOD (before clamping to the mip range)
  kGetWeights,     // x/y/z = filter fractions, w = mip fraction
  kGetBCF,         // border colour fraction: x/y/z = 1 where that axis is
                   // outside [0, 1) and the clamp mode uses the border
};

struct SampleRequest {
  SampleOp op = SampleOp::kTfetch;
  uint32_t fetch[6] = {};  // fetch constant dwords as registers
  // Instruction dimension: kDim1D, kDim2D, kDim3D (3D or stacked), kDimCube.
  Dimension dimension = kDim2D;
  // Coordinates as the instruction receives them. For kDimCube: face-local
  // u, v in [0, 1] and the resolved face in cube_face (the shader's
  // cube-vector + 1.5 handling is the caller's).
  float coord[3] = {};
  uint32_t cube_face = 0;
  bool unnormalized = false;
  // Texel offsets of the instruction (offset_x/y/z, half-texel steps).
  float offset[3] = {};
  // Gradients per pixel (ddx, ddy of u, v, w in the instruction's coordinate
  // space). Required when the LOD is computed.
  bool has_grads = false;
  float grad_h[3] = {};
  float grad_v[3] = {};
  // LOD sources (D3D order: register / computed LOD + fetch LOD bias +
  // instruction LOD bias).
  bool use_computed_lod = true;
  bool use_register_lod = false;
  float register_lod = 0.0f;
  float instr_lod_bias = 0.0f;
  // Instruction overrides; nullopt = "use fetch constant".
  std::optional<uint32_t> mag_filter, min_filter, mip_filter;
  std::optional<uint32_t> vol_mag_filter, vol_min_filter, aniso_filter;
};

std::array<float, 4> Sample(const SampleRequest& request, const GuestMemory& mem, Diag* diag);

}  // namespace refxenos::tex
