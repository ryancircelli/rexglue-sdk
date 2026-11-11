// refxenos rasterizer: primitive assembly, clipping, setup and rasterization.
//
// Input: the vertex shader outputs of one draw (VertexOut, in vertex-stream
// order, after index fetch) plus the register file. Output: PixelQuads,
// emitted in primitive order, then raster order (quad rows top to bottom,
// quads left to right inside a row). A quad is emitted when at least one of its
// pixels has a covered sample after scissor; its other pixels are helper
// pixels (coverage 0) that are still interpolated (for gradients).
//
// COORDINATE SPACE OF THE OUTPUT
//   PixelQuad::x/y are render-target pixel coordinates AFTER the window offset
//   (PA_SC_WINDOW_OFFSET applied to vertices when vtx_window_offset_enable,
//   and to the window scissor unless window_offset_disable). That is the same
//   space ps_param_gen XY and the scissor use. The EDRAM agent maps (x, y) to
//   EDRAM through the surface base/pitch only.
//
//   Internally the rasterizer works in "sample space": pixel (i, j) spans
//   [i, i+1) x [j, j+1) and its center is (i + 0.5, j + 0.5). With
//   PA_SU_VTX_CNTL.pix_center == kD3DZero (Direct3D 9 integer centers) the
//   vertices are shifted by +0.5 px instead (rule raster_half_pixel_offset), so
//   a guest vertex at integer x lands on the center of pixel x.
//   PixelQuad::center_x/y report that sample-space center (i + 0.5).
//
// SAMPLES
//   Sample index s is the guest index, see core/edram.h (kSamplePositions
//   below, 1/16 px from the pixel center, +x right, +y down).
//
// INTERPOLATION
//   All 16 interpolators are interpolated for every pixel (the shader glue
//   applies the PS input mask / zeroing). SQ_INTERPOLATOR_CNTL param_shade
//   is not treated as a flat mask (rule raster_flat_param_shade); centroid ones
//   are evaluated at the centroid location (see RasterCentroidMask).
//   PixelQuad::param_gen is always filled, encoded (the glue decides whether
//   the shader gets it, SQ_PROGRAM_CNTL param_gen + SQ_CONTEXT_MISC
//   param_gen_pos).
#pragma once

#include <cstdint>
#include <functional>
#include <span>

#include "core/common.h"

namespace refxenos {

struct RasterInput {
  xenos::PrimitiveType prim = xenos::PrimitiveType::kTriangleList;
  // One entry per vertex of the draw, in stream order.
  std::span<const VertexOut> vertices;
  // Optional (empty, or the same length as vertices): nonzero entry i means a
  // primitive reset happened before vertex i (the reset index itself is not in
  // `vertices`). Honoured for every primitive type: assembly restarts from
  // scratch at vertex i. The caller decides when reset applies
  // (PA_SU_SC_MODE_CNTL.multi_prim_ib_ena, VGT_MULTI_PRIM_IB_RESET_INDX).
  std::span<const uint8_t> restart_before;
};

using QuadEmitFn = std::function<void(const PixelQuad&)>;

// Rasterizes one draw. Calls `emit` for every quad, in order.
void RasterizeDraw(const RegisterFile& regs, const RasterInput& in, const QuadEmitFn& emit,
                   DrawStats& stats);

// Sample positions, 1/16 px from the pixel center, guest sample index order.
struct SamplePosition {
  int32_t x, y;
};
uint32_t RasterSampleCount(xenos::MsaaSamples msaa);
std::span<const SamplePosition> RasterSamplePositions(xenos::MsaaSamples msaa);

// Bit i set: interpolator i is sampled at the centroid (else at the center).
uint32_t RasterCentroidMask(const RegisterFile& regs);
// Bit i set: interpolator i is flat shaded (provoking vertex value). Currently
// always 0, see rule raster_flat_param_shade.
uint32_t RasterFlatMask(const RegisterFile& regs);

}  // namespace refxenos
