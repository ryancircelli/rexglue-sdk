// refxenos rasterizer. See raster.h for the API and the coordinate spaces.
//
// Model (each step is a rule below):
//   1. Primitive assembly from the vertex stream (raster_asm_*).
//   2. Vertex kill (oPts.z), NaN positions, user clip plane culling.
//   3. Clip-space position from oPos and the PA_CL_VTE_CNTL formats.
//   4. Clipping in homogeneous clip space: z (0..w or -w..w), w > 0, the
//      guard band in x/y, user clip planes. A clipped triangle becomes a fan.
//   5. Viewport transform in float32, then subpixel snapping (PA_SU_VTX_CNTL
//      quant_mode / round_mode). Coverage is computed exactly in integers on
//      the snapped positions (1/16 px units), top-left fill rule.
//   6. Face (shoelace area of the snapped polygon), cull, polygon mode.
//   7. Attributes and depth are evaluated from the ORIGINAL vertices with 2D
//      homogeneous barycentrics in double precision, so clipping and snapping
//      do not alter attribute values, then rounded to float once.
//   8. Depth: per-sample plane evaluation, polygon offset, clamp to [0, 1].
//   9. Scissor, quad emission (helper pixels included).
#include "core/raster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace refxenos {

namespace rg = rex::graphics;

// ---------------------------------------------------------------------------
// Rules.

RX_DEFINE_RULE(raster_sample_positions, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.h:150-152 (kXenosSamplePositions2x/4x, "
               "from the R6xx 3D register reference PA_SC_MODE_CNTL and Mesa r600)",
               false,
               "4x RT, draw a 1-pixel-wide vertical edge at x = 10 + k/16 for k = -8..8; per k resolve "
               "each sample separately (copy_sample_select 0..3) and record which samples are covered");
RX_DEFINE_RULE(raster_viewport_vte, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:447-456 (scale = *_scale_ena ? "
               "PA_CL_VPORT_*SCALE : 1, offset = *_offset_ena ? PA_CL_VPORT_*OFFSET : 0)",
               false, "");
RX_DEFINE_RULE(raster_window_offset_vertices, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:458-462 (PA_SC_WINDOW_OFFSET added to "
               "the viewport offset when PA_SU_SC_MODE_CNTL.vtx_window_offset_enable)",
               false, "");
RX_DEFINE_RULE(raster_half_pixel_offset, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:463-467 (+0.5 px when "
               "PA_SU_VTX_CNTL.pix_center == kD3DZero: guest integer coordinates are pixel centers)",
               false,
               "1x RT, rect list covering x in [10, 12) exactly (vertices at 10.0 and 12.0) once with "
               "pix_center=0 and once with pix_center=1: with 0 pixels 9..11 boundary moves by half a "
               "pixel (pixels 10,11 covered when centers are at .5; 10,11 shifted to 9.5.. in D3D mode)");
RX_DEFINE_RULE(raster_vte_formats, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/vulkan/vulkan_command_processor.cc:5185-5197 and "
               "spirv_shader_translator.cc:1960-2020 (w = vtx_w0_fmt ? W : 1/W; xy *= w if vtx_xy_fmt; "
               "z *= w if vtx_z_fmt)",
               false, "");
RX_DEFINE_RULE(raster_viewport_float_order, 1,
               "assumed: Xenia leaves the divide and viewport to the host rasterizer "
               "(draw_util.cc:296-470 GetHostViewportInfo); refxenos uses float32 rhw = 1/w, "
               "ndc = c * rhw, s = ndc * scale + (offset + window + half)",
               true,
               "1x RT, quant 1/16 round-to-even: triangles whose left edge vertex x_ndc is chosen so that "
               "the float32 results of (x*(1/w))*scale+offset and (x/w)*scale+offset differ by 1 ulp "
               "across a 1/32 px rounding boundary; check which pixel column is covered");
RX_DEFINE_RULE(raster_subpixel_snap, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/xenos.h:908-925 (VertexRounding/VertexQuantization, "
               "games use 1/16 and round-to-even); assumed: Xenia ignores PA_SU_VTX_CNTL quant_mode and "
               "round_mode (host subpixel precision), refxenos snaps x and y to the quant grid, "
               "kTruncate = floor, kRound = ties up, kRoundToEven/kRoundToOdd = ties to even/odd grid step",
               true,
               "1x RT, for each round_mode and quant_mode: a triangle with a vertical left edge at x = "
               "10.5 + k/256 px (k = -16..16) in OGL-center mode; record for each k whether pixel 10 is "
               "covered; the flip point gives the grid and the tie behaviour");
RX_DEFINE_RULE(raster_top_left_fill, 1,
               "sdk include/rex/graphics/registers.h:152-235 (SQ_CONTEXT_MISC comment: top-left rule "
               "like Direct3D); assumed: exact D3D10 definition (top = horizontal edge above the "
               "interior, left = edge with interior on its right going up in y-down screen space)",
               true,
               "1x and 4x RT, additive blend 1/255: two triangles sharing a horizontal, a vertical and a "
               "diagonal edge that pass exactly through pixel centers / sample points; no sample may be "
               "2/255 or 0; record which triangle owns the shared samples");
RX_DEFINE_RULE(raster_face_orientation, 1,
               "sdk include/rex/graphics/registers.h PA_SU_SC_MODE_CNTL.face (0: CCW front, 1: CW front); "
               "assumed: winding measured on the snapped screen polygon, y down, positive shoelace area = "
               "clockwise",
               true,
               "cull_back=1, face=0: triangle (0,0),(16,0),(0,16) in screen pixels (clockwise on screen) "
               "and its reverse; exactly one must draw; repeat with face=1");
RX_DEFINE_RULE(raster_cull_degenerate, 1,
               "assumed: a primitive whose snapped area is zero produces no pixels",
               true,
               "1x RT: triangle (10,10),(20,10.03),(30,10.06) collinear after 1/16 snapping; check no "
               "pixel written");
RX_DEFINE_RULE(raster_nonpolygonal_front, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.h:56-98 (IsPrimitivePolygonal: points, "
               "lines and rectangle lists are not polygonal: always front facing, never face culled)",
               false, "");
RX_DEFINE_RULE(raster_clip_planes, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:520-690 (clip_disable disables x/y/z "
               "clipping; dx_clip_space_def: 0 <= z <= w, else -w <= z <= w)",
               false, "");
RX_DEFINE_RULE(raster_clip_w_positive, 1,
               "assumed: geometry with w <= 0 is clipped away (plane w >= 2^-40) when clipping is enabled",
               true,
               "triangle with one vertex at w = 0 and z = 0 (dx clip space), the others in front; check "
               "the visible area matches the clipped fan");
RX_DEFINE_RULE(raster_guard_band, 1,
               "assumed: x/y are clipped only against the guard band |x| <= PA_CL_GB_HORZ_CLIP_ADJ * w, "
               "|y| <= PA_CL_GB_VERT_CLIP_ADJ * w (no clipping if the register is not a finite value > 0); "
               "viewport-edge clipping is the scissor's job (draw_util.cc:296-340 comment); discard adj "
               "only trivially rejects, no visible effect",
               true,
               "window scissor 64x64, viewport 32x32 at 0,0, GB adj = 1.0 then 4.0: a triangle crossing "
               "x_ndc = 1 draws pixels x >= 32 only when the guard band is wider than the viewport");
RX_DEFINE_RULE(raster_clip_fan, 1,
               "assumed: a clipped triangle is rasterized as the triangle fan (p0, pk, pk+1) of the "
               "clipped polygon, each with the top-left rule; fan triangles with inverted snapped "
               "winding are dropped",
               true,
               "near-plane-crossing triangle, 4x RT additive blend: check no sample is hit twice along "
               "the internal fan edges");
RX_DEFINE_RULE(raster_interp_unclipped, 1,
               "assumed: attributes and depth come from the original vertices through 2D homogeneous "
               "barycentrics (Olano-Greer); clipping and subpixel snapping do not change them",
               true,
               "32-bit float RT, near-plane-crossing triangle with o0 = (vertex index, 0, 0, 1) and a "
               "triangle with a vertex at x = 10 + 1/64 px: compare the exported float with the formula");
RX_DEFINE_RULE(raster_persp_interp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator.cc (interpolators are plain "
               "smooth host inputs, perspective-correct barycentrics are the host's job)",
               false, "");
RX_DEFINE_RULE(raster_interp_precision, 1,
               "assumed: barycentrics and sums in double, one rounding to float32 per component",
               true,
               "32-bit float RT, large triangle with o0 = (1e6, 1/3, -7, 0.1) at v0 and other values at "
               "v1/v2; compare the low mantissa bits against the double model");
RX_DEFINE_RULE(raster_persp_corr_dis, 1,
               "assumed: PA_SU_SC_MODE_CNTL.persp_corr_dis = 1 interpolates linearly in screen space "
               "(Xenia ignores the bit)",
               true,
               "triangle with w = 1, 4, 16 at its vertices and o0 = (u, 0, 0, 1); compare center pixel "
               "values with persp_corr_dis 0 and 1");
RX_DEFINE_RULE(raster_flat_param_shade, 1,
               "assumed: SQ_INTERPOLATOR_CNTL.param_shade bit i = 1 makes interpolator i flat (provoking "
               "vertex value); Xenia ignores param_shade",
               true,
               "D3DRS_SHADEMODE flat vs gouraud with a colour interpolant that differs per vertex; read "
               "back SQ_INTERPOLATOR_CNTL and the pixel values");
RX_DEFINE_RULE(raster_provoking_vertex, 1,
               "XDK GPU_interpolation.htm (nointerpolation takes the leading vertex: first vertex of each "
               "primitive; line strip v_i, tri strip v_i); assumed: provoking_vtx_last = 1 takes the last",
               true,
               "flat interpolant, triangle list / strip / line strip with distinct per-vertex values, "
               "provoking_vtx_last 0 and 1");
RX_DEFINE_RULE(raster_provoking_fan, 1,
               "assumed: triangle fan triangle i (v0, v_i+1, v_i+2) uses v_i+1 as first-vertex provoking "
               "vertex (D3D9/Vulkan convention); Xenia converts fans to lists and uses v0",
               true,
               "flat interpolant on a 4-vertex triangle fan with distinct per-vertex values; read both "
               "triangles");
RX_DEFINE_RULE(raster_strip_winding, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/primitive_processor.cc:378-420 (strips passed to the "
               "host unchanged: odd triangles have reversed winding)",
               false, "");
RX_DEFINE_RULE(raster_fan_order, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/primitive_processor.cc (fan triangle i = v0, v_i+1, "
               "v_i+2)",
               false, "");
RX_DEFINE_RULE(raster_line_loop, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/primitive_processor.cc (line loop = line strip plus "
               "the closing segment back to the first vertex)",
               false, "");
RX_DEFINE_RULE(raster_quad_list_split, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/primitive_processor.cc:219-228 (quad -> (v0,v1,v2), "
               "(v0,v2,v3), order marked TODO in Xenia)",
               true,
               "quad list with additive blend and a flat interpolant: check the diagonal ownership and "
               "the flat value of each half");
RX_DEFINE_RULE(raster_quad_strip_polygon, 1,
               "assumed: Xenia does not support kQuadStrip/kPolygon (primitive_processor.cc XELOGE); quad "
               "strip quad i = (v2i, v2i+1, v2i+3, v2i+2) split like a quad list, polygon = fan",
               true, "quad strip and polygon draws with additive blend and a flat interpolant");
RX_DEFINE_RULE(raster_unsupported_types, 1,
               "assumed: Xenia does not support kTriangleWithWFlags / 2D types; refxenos treats "
               "kTriangleWithWFlags as a triangle list, 2D copy/fill rect lists as rect lists, 2D line "
               "strip / tri strip as line / triangle strips",
               true, "a draw of each type with known vertices");
RX_DEFINE_RULE(raster_rect_fourth_vertex, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/vulkan/vulkan_pipeline_cache.cc:2261-2520 (longest "
               "edge is the diagonal; v3 = (v[i1] - v[i0]) + v[i2] for position and every interpolator)",
               false, "");
RX_DEFINE_RULE(raster_rect_edge_select, 1,
               "assumed: the rect-list diagonal is chosen from squared edge lengths of the float32 screen "
               "positions (Xenia uses host clip-space x/y, ties: 12 > 20 && 12 > 01 -> 0123, else 20 > "
               "01 -> 1203, else 2013)",
               true, "rect list with a right isoceles layout where two edges tie; check the covered area");
RX_DEFINE_RULE(raster_msaa_disable, 1,
               "assumed: with an MSAA RT and PA_SU_SC_MODE_CNTL.msaa_enable = 0, coverage and depth are "
               "evaluated at the pixel center and replicated to every sample",
               true,
               "4x RT, msaa_enable 0: triangle edge passing between samples of one pixel; resolve each "
               "sample index and compare colour and depth");
RX_DEFINE_RULE(raster_centroid, 1,
               "sdk include/rex/graphics/xenos.h:805-822 GetInterpolatorSamplingPattern (which "
               "interpolators use the centroid); XDK GPU_interpolation.htm (centroid = a covered sample); "
               "assumed: fully covered or helper pixel -> center, else the lowest-index covered sample "
               "(before scissor)",
               true,
               "4x RT, centroid interpolant o0 = screen position, triangle edges covering only sample 2, "
               "only samples 1+3, etc.; read o0 back");
RX_DEFINE_RULE(raster_depth_plane, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:2445-2510 (per-sample "
               "depth at the Xenos sample positions, z = viewport(z_c / w_c)); assumed: exact plane "
               "evaluation in double instead of center + coarse derivatives",
               true,
               "D24FS8 4x RT, steep triangle, resolve depth per sample, compare to the plane model");
RX_DEFINE_RULE(raster_depth_clamp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_rb.cc:2445-2510 (NClamp of "
               "the biased per-sample depth to [0, 1], NaN -> 0)",
               false, "");
RX_DEFINE_RULE(raster_poly_offset, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:217-246 (front/back regs by face for "
               "polygons, FRONT regs if poly_offset_para_enable otherwise), xenos.h:445 "
               "kPolygonOffsetScaleSubpixelUnit = 1/16, spirv_shader_translator_rb.cc:2395-2418 (offset = "
               "scale/16 * max(|dz/dx|, |dz/dy|) + offset, offset absolute for both depth formats)",
               false,
               "slope scale 16, offset 0 / 2^-20 on D24S8 and D24FS8 with a slanted triangle; compare depth");
RX_DEFINE_RULE(raster_scissor, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_util.cc:731-913 GetScissor (window scissor + "
               "window offset unless window_offset_disable, intersect screen scissor, clamp >= 0, BR "
               "exclusive)",
               false, "");
RX_DEFINE_RULE(raster_point_size, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/vulkan/vulkan_command_processor.cc:5331 (12.4 "
               "PA_SU_POINT_SIZE / MINMAX radius -> diameter * 2/16), spirv_shader_translator.cc:"
               "2183-2230 + 3535 (oPts.x used if its int bits >= 0, clamped as int to min/max); XDK "
               "point_sprites_enabling.htm (min/max clamp only the shader size)",
               false, "");
RX_DEFINE_RULE(raster_point_rect, 1,
               "assumed: a point sprite covers samples in [cx - w/2, cx + w/2) x [cy - h/2, cy + h/2) "
               "with corners computed in float32 and snapped like vertices",
               true,
               "1x RT, sprites of size 1, 2, 3, 2.5 px centered at integer and .5 positions; record the "
               "covered pixels");
RX_DEFINE_RULE(raster_point_coord, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator.cc:2183-2291 (corners TL BL TR "
               "BR with coords (0,0) (0,1) (1,0) (1,1), V = 0 at top) and dxbc_shader_translator.cc:"
               "745-760 (saturated into param_gen ZW); assumed: evaluated at the pixel center from the "
               "unsnapped corners",
               true,
               "32-bit float RT, 4 px sprite at x = 10.25: export param_gen ZW, compare");
RX_DEFINE_RULE(raster_point_clip, 1,
               "assumed: a point is discarded when its center fails z/w or user plane clipping; no x/y "
               "clipping (Xenia point expansion happens after host clipping decisions are unavailable)",
               true,
               "point sprite of size 8 with center just behind the near plane / just beyond x_ndc = 1");
RX_DEFINE_RULE(raster_vertex_kill, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator.cc:2109-2150 (oPts.z bits "
               "0:30 nonzero kills; PA_CL_CLIP_CNTL.vtx_kill_or: any vertex kills the primitive, else all "
               "must be killed)",
               false, "");
RX_DEFINE_RULE(raster_nan_position, 1,
               "assumed: a primitive with a NaN clip-space coordinate is discarded (xy/z/w_nan_retain "
               "ignored)",
               true, "triangle with x = NaN at one vertex, nan_retain bits 0 and 1");
RX_DEFINE_RULE(raster_line_width, 1,
               "assumed: line half-width = PA_SU_LINE_CNTL bits 0:15 in 12.4 px (like the point size "
               "registers); Xenia draws host 1 px lines",
               true,
               "line list with PA_SU_LINE_CNTL width 8, 16, 24: measure the covered height of a "
               "horizontal line at y = 10.5");
RX_DEFINE_RULE(raster_line_quad, 1,
               "assumed: a line is the parallelogram from v0 to v1 extended by +-half-width along the minor "
               "axis (x-major when |dx| >= |dy|), top-left rule, attributes constant across the width",
               true,
               "1x RT: lines at 0, 30, 45, 60 degrees with additive blend; record covered pixels and "
               "strip joints");
RX_DEFINE_RULE(raster_param_gen, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator.cc:702-820 (XY = floor(pixel "
               "position); point: X = |x|, Y = -|y|, ZW = point coord; else XY = |xy|, X negated on back "
               "faces of polygons, Z = line flag << 31, W = 0)",
               false, "");
RX_DEFINE_RULE(raster_param_gen_ogl_center, 1,
               "assumed: param_gen XY is the integer pixel index also with pix_center = kOGLHalf "
               "(Xenia floors in both modes; registers.h comment suggests .5 in OGL mode)",
               true, "pix_center = 1, PS exporting param_gen XY to a float RT");
RX_DEFINE_RULE(raster_ucp, 1,
               "assumed: user clip plane i keeps dot(PA_CL_UCP_i, clip position) >= 0; ucp_cull_only_ena "
               "culls a primitive whose vertices are all outside one plane and never clips; points and "
               "lines use the center/segment (ps_ucp_mode 0)",
               true, "triangle crossing a user plane with ucp_cull_only_ena 0 and 1");
RX_DEFINE_RULE(raster_polymode, 1,
               "assumed: PA_SU_SC_MODE_CNTL.poly_mode = kDualMode draws triangles of a face as points "
               "(vertices) or lines (edges, skipped when boundary_edge_flag_ena and oPts.y of the edge "
               "start is 0) per polymode_front/back_ptype; face polygon offset applies",
               true, "wireframe cube with flat colours, both ptypes");
RX_DEFINE_RULE(raster_fixed_range, 1,
               "assumed: primitives whose snapped coordinates leave +-2^24 px are dropped",
               true, "clip_disable triangle with a vertex at x = 1e8");
RX_DEFINE_RULE(raster_quad_order, 1,
               "assumed: quads of one primitive are emitted row by row top to bottom, left to right (a "
               "primitive covers each sample at most once, so the order has no visible effect)",
               true, "none needed");

namespace {

constexpr int64_t kSub = 16;                       // subpixel units per pixel
constexpr int64_t kMaxFixed = int64_t(1) << 28;    // +-2^24 px in 1/16 units
constexpr double kMinW = 1.0 / 1099511627776.0;    // 2^-40

const SamplePosition kSamples1x[1] = {{0, 0}};
const SamplePosition kSamples2x[2] = {{-4, 4}, {4, -4}};
const SamplePosition kSamples4x[4] = {{-2, -2}, {2, 2}, {-6, 6}, {6, -6}};

int64_t FloorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

struct Note {
  DrawStats& stats;
  std::set<std::string> seen;
  void operator()(const std::string& s) {
    if (seen.insert(s).second) stats.notes.push_back("raster: " + s);
  }
};

// Clip-space vertex in double (exact copies of the float32 values for
// original vertices).
struct CV {
  double c[4];
  bool original;  // true: unclipped input vertex, c are exact float values
};

struct Plane {
  double p[4];
};

// What the interpolation of one primitive is based on.
struct Setup {
  const VertexOut* v[3] = {};
  double C[3][3] = {};  // lambda_i(p) = C[i] . (x, y, 1), sample space px
  double W[3] = {};
  double zc[3] = {};
  double D = 0;  // sum W_i C_i[2] (= det of the homogeneous matrix)
  bool constant = false;  // points: everything from v[0]
  double z_const = 0;     // points: window z
};

struct Prim {
  Setup setup;
  const VertexOut* provoking = nullptr;
  uint32_t index = 0;
  bool front = true;
  bool polygonal = false;
  bool line = false;
  bool point = false;
  double point_rect[4] = {};  // L, T, R, B unsnapped, px (points only)
  double poly_offset = 0;     // added to every sample z
  bool has_poly_offset = false;
  double poly_scale = 0, poly_offset_reg = 0;
};

struct Ctx {
  const RegisterFile& regs;
  const QuadEmitFn& emit;
  DrawStats& stats;
  Note note;

  reg::PA_SU_SC_MODE_CNTL sc;
  reg::PA_CL_CLIP_CNTL clip;
  reg::PA_CL_VTE_CNTL vte;
  reg::PA_SU_VTX_CNTL vtx;
  xenos::MsaaSamples msaa;
  uint32_t sample_count = 1;
  std::span<const SamplePosition> samples;
  bool msaa_enable = false;
  float scale[3], offset[3];
  int32_t sc_l, sc_t, sc_r, sc_b;
  uint32_t centroid_mask, flat_mask;
  int64_t quant_step;
  std::vector<Plane> clip_planes;  // every plane that clips (zw + guard band + ucp)
  std::vector<Plane> center_planes;  // planes tested on point centers (zw + ucp)
  std::vector<Plane> ucp_cull;     // planes that only cull
  bool depth_float24 = false;
  float point_diam_const[2];  // x, y diameter px
  uint32_t point_min_bits, point_max_bits;
  int64_t line_half_width;    // 1/16 px

  Ctx(const RegisterFile& r, const QuadEmitFn& e, DrawStats& s) : regs(r), emit(e), stats(s), note{s, {}} {}
};

// ---------------------------------------------------------------------------
// Setup helpers.

float RegF(const RegisterFile& regs, uint32_t index) { return AsF32(regs[index]); }

void InitCtx(Ctx& ctx) {
  const RegisterFile& regs = ctx.regs;
  ctx.sc = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  ctx.clip = regs.Get<reg::PA_CL_CLIP_CNTL>();
  ctx.vte = regs.Get<reg::PA_CL_VTE_CNTL>();
  ctx.vtx = regs.Get<reg::PA_SU_VTX_CNTL>();
  ctx.msaa = regs.Get<reg::RB_SURFACE_INFO>().msaa_samples;
  ctx.sample_count = RasterSampleCount(ctx.msaa);
  ctx.samples = RasterSamplePositions(ctx.msaa);
  ctx.msaa_enable = ctx.sc.msaa_enable != 0;
  RX_USE(raster_sample_positions);
  if (ctx.sample_count > 1 && !ctx.msaa_enable) RX_USE(raster_msaa_disable);
  ctx.depth_float24 =
      regs.Get<reg::RB_DEPTH_INFO>().depth_format == xenos::DepthRenderTargetFormat::kD24FS8;

  // Viewport.
  RX_USE(raster_viewport_vte);
  const auto& vte = ctx.vte;
  ctx.scale[0] = vte.vport_x_scale_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_XSCALE) : 1.0f;
  ctx.scale[1] = vte.vport_y_scale_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f;
  ctx.scale[2] = vte.vport_z_scale_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_ZSCALE) : 1.0f;
  ctx.offset[0] = vte.vport_x_offset_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_XOFFSET) : 0.0f;
  ctx.offset[1] = vte.vport_y_offset_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0f;
  ctx.offset[2] = vte.vport_z_offset_ena ? RegF(regs, rg::XE_GPU_REG_PA_CL_VPORT_ZOFFSET) : 0.0f;
  auto window = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
  if (ctx.sc.vtx_window_offset_enable) {
    RX_USE(raster_window_offset_vertices);
    ctx.offset[0] += float(window.window_x_offset);
    ctx.offset[1] += float(window.window_y_offset);
  }
  if (ctx.vtx.pix_center == xenos::PixelCenter::kD3DZero) {
    RX_USE(raster_half_pixel_offset);
    ctx.offset[0] += 0.5f;
    ctx.offset[1] += 0.5f;
  }
  RX_USE(raster_vte_formats);
  RX_USE(raster_viewport_float_order);

  // Snapping.
  RX_USE(raster_subpixel_snap);
  uint32_t quant = uint32_t(ctx.vtx.quant_mode);
  if (quant > 4) {
    ctx.note("PA_SU_VTX_CNTL.quant_mode " + std::to_string(quant) + " unsupported, using 1/16");
    quant = 0;
  }
  ctx.quant_step = int64_t(1) << quant;

  // Scissor.
  RX_USE(raster_scissor);
  auto wtl = regs.Get<reg::PA_SC_WINDOW_SCISSOR_TL>();
  auto wbr = regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>();
  auto stl = regs.Get<reg::PA_SC_SCREEN_SCISSOR_TL>();
  auto sbr = regs.Get<reg::PA_SC_SCREEN_SCISSOR_BR>();
  int32_t l = int32_t(wtl.tl_x), t = int32_t(wtl.tl_y), r = int32_t(wbr.br_x), b = int32_t(wbr.br_y);
  if (!wtl.window_offset_disable) {
    l += window.window_x_offset;
    r += window.window_x_offset;
    t += window.window_y_offset;
    b += window.window_y_offset;
  }
  l = std::max(l, int32_t(stl.tl_x));
  t = std::max(t, int32_t(stl.tl_y));
  r = std::min(r, int32_t(sbr.br_x));
  b = std::min(b, int32_t(sbr.br_y));
  l = std::max(l, 0);
  t = std::max(t, 0);
  r = std::max(r, l);
  b = std::max(b, t);
  ctx.sc_l = l;
  ctx.sc_t = t;
  ctx.sc_r = r;
  ctx.sc_b = b;

  ctx.centroid_mask = RasterCentroidMask(regs);
  ctx.flat_mask = RasterFlatMask(regs);

  // Clip planes.
  if (!ctx.clip.clip_disable) {
    RX_USE(raster_clip_planes);
    RX_USE(raster_clip_w_positive);
    ctx.clip_planes.push_back({{0, 0, 0, 1}});  // w >= kMinW handled specially
    if (ctx.clip.dx_clip_space_def) {
      ctx.clip_planes.push_back({{0, 0, 1, 0}});
    } else {
      ctx.clip_planes.push_back({{0, 0, 1, 1}});
    }
    ctx.clip_planes.push_back({{0, 0, -1, 1}});
    ctx.center_planes = ctx.clip_planes;
    float gbx = RegF(regs, rg::XE_GPU_REG_PA_CL_GB_HORZ_CLIP_ADJ);
    float gby = RegF(regs, rg::XE_GPU_REG_PA_CL_GB_VERT_CLIP_ADJ);
    RX_USE(raster_guard_band);
    if (std::isfinite(gbx) && gbx > 0) {
      ctx.clip_planes.push_back({{-1, 0, 0, double(gbx)}});
      ctx.clip_planes.push_back({{1, 0, 0, double(gbx)}});
    }
    if (std::isfinite(gby) && gby > 0) {
      ctx.clip_planes.push_back({{0, -1, 0, double(gby)}});
      ctx.clip_planes.push_back({{0, 1, 0, double(gby)}});
    }
  }
  if (ctx.clip.ucp_ena) {
    RX_USE(raster_ucp);
    for (uint32_t i = 0; i < 6; ++i) {
      if (!(ctx.clip.ucp_ena & (1u << i))) continue;
      Plane p;
      for (uint32_t c = 0; c < 4; ++c) {
        p.p[c] = double(RegF(regs, rg::XE_GPU_REG_PA_CL_UCP_0_X + i * 4 + c));
      }
      if (ctx.clip.ucp_cull_only_ena) {
        ctx.ucp_cull.push_back(p);
      } else {
        ctx.clip_planes.push_back(p);
        ctx.center_planes.push_back(p);
      }
    }
    if (ctx.clip.ps_ucp_mode != 0) {
      ctx.note("PA_CL_CLIP_CNTL.ps_ucp_mode " + std::to_string(ctx.clip.ps_ucp_mode) +
               " treated as 0 (center/segment clipping)");
    }
  }

  // Points.
  auto psize = regs.Get<reg::PA_SU_POINT_SIZE>();
  auto pminmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
  ctx.point_diam_const[0] = float(psize.width) * (2.0f / 16.0f);
  ctx.point_diam_const[1] = float(psize.height) * (2.0f / 16.0f);
  ctx.point_min_bits = AsU32(float(pminmax.min_size) * (2.0f / 16.0f));
  ctx.point_max_bits = AsU32(float(pminmax.max_size) * (2.0f / 16.0f));
  // Lines.
  ctx.line_half_width = int64_t(regs[rg::XE_GPU_REG_PA_SU_LINE_CNTL] & 0xFFFF);
}

// Clip-space position (float32) from the exported position.
void ToClip(const VertexOut& v, const Ctx& ctx, float out[4]) {
  float w_in = v.position[3];
  float w = ctx.vte.vtx_w0_fmt ? w_in : 1.0f / w_in;
  out[0] = ctx.vte.vtx_xy_fmt ? v.position[0] * w : v.position[0];
  out[1] = ctx.vte.vtx_xy_fmt ? v.position[1] * w : v.position[1];
  out[2] = ctx.vte.vtx_z_fmt ? v.position[2] * w : v.position[2];
  out[3] = w;
}

// float32 viewport transform of a clip-space position.
void Project(const Ctx& ctx, const double c[4], float s[3]) {
  float cf[4] = {float(c[0]), float(c[1]), float(c[2]), float(c[3])};
  float rhw = 1.0f / cf[3];
  for (int i = 0; i < 3; ++i) {
    float ndc = cf[i] * rhw;
    float scaled = ndc * ctx.scale[i];
    s[i] = scaled + ctx.offset[i];
  }
}

bool RoundToGrid(const Ctx& ctx, float v, int64_t& out) {
  if (!std::isfinite(v)) return false;
  double t = double(v) * double(kSub) / double(ctx.quant_step);
  if (std::fabs(t) * double(ctx.quant_step) > double(kMaxFixed)) return false;
  double fl = std::floor(t);
  double frac = t - fl;
  int64_t r = int64_t(fl);
  switch (ctx.vtx.round_mode) {
    case xenos::VertexRounding::kTruncate:
      break;
    case xenos::VertexRounding::kRound:
      if (frac >= 0.5) ++r;
      break;
    case xenos::VertexRounding::kRoundToEven:
      if (frac > 0.5 || (frac == 0.5 && (r & 1))) ++r;
      break;
    case xenos::VertexRounding::kRoundToOdd:
      if (frac > 0.5 || (frac == 0.5 && !(r & 1))) ++r;
      break;
  }
  out = r * ctx.quant_step;
  return true;
}

double Dot(const Plane& p, const double c[4]) {
  return p.p[0] * c[0] + p.p[1] * c[1] + p.p[2] * c[2] + p.p[3] * c[3];
}

double PlaneDist(const Plane& p, const double c[4]) {
  double d = Dot(p, c);
  // The w plane keeps w >= kMinW.
  if (p.p[0] == 0 && p.p[1] == 0 && p.p[2] == 0 && p.p[3] == 1) d = c[3] - kMinW;
  return d;
}

// Sutherland-Hodgman against every clip plane. Returns false if empty.
bool ClipPolygon(const Ctx& ctx, std::vector<CV>& poly, bool& clipped) {
  for (const Plane& pl : ctx.clip_planes) {
    bool any_out = false;
    for (const CV& v : poly) any_out |= !(PlaneDist(pl, v.c) >= 0);
    if (!any_out) continue;
    clipped = true;
    std::vector<CV> out;
    size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
      const CV& a = poly[i];
      const CV& b = poly[(i + 1) % n];
      double da = PlaneDist(pl, a.c), db = PlaneDist(pl, b.c);
      bool ain = da >= 0, bin = db >= 0;
      if (ain) out.push_back(a);
      if (ain != bin) {
        double t = da / (da - db);
        CV m;
        m.original = false;
        for (int c = 0; c < 4; ++c) m.c[c] = a.c[c] + t * (b.c[c] - a.c[c]);
        out.push_back(m);
      }
    }
    poly.swap(out);
    if (poly.size() < 3) return false;
  }
  return true;
}

// Segment clipping (parametric).
bool ClipSegment(const Ctx& ctx, CV& a, CV& b, bool& clipped) {
  double t0 = 0, t1 = 1;
  for (const Plane& pl : ctx.clip_planes) {
    double da = PlaneDist(pl, a.c), db = PlaneDist(pl, b.c);
    if (da >= 0 && db >= 0) continue;
    if (da < 0 && db < 0) return false;
    if (!(da == da) || !(db == db)) return false;
    double t = da / (da - db);
    if (da < 0) {
      t0 = std::max(t0, t);
    } else {
      t1 = std::min(t1, t);
    }
    if (t0 > t1) return false;
  }
  if (t0 > 0 || t1 < 1) {
    clipped = true;
    CV na = a, nb = b;
    if (t0 > 0) {
      na.original = false;
      for (int c = 0; c < 4; ++c) na.c[c] = a.c[c] + t0 * (b.c[c] - a.c[c]);
    }
    if (t1 < 1) {
      nb.original = false;
      for (int c = 0; c < 4; ++c) nb.c[c] = a.c[c] + t1 * (b.c[c] - a.c[c]);
    }
    a = na;
    b = nb;
  }
  return true;
}

// User clip planes in cull-only mode: true if all vertices are outside one plane.
bool UcpCulled(const Ctx& ctx, const CV* v, size_t n) {
  for (const Plane& pl : ctx.ucp_cull) {
    bool all_out = true;
    for (size_t i = 0; i < n; ++i) all_out &= Dot(pl, v[i].c) < 0;
    if (all_out) return true;
  }
  return false;
}

bool HasNaN(const CV* v, size_t n) {
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < 4; ++c)
      if (std::isnan(v[i].c[c])) return true;
  return false;
}

// Homogeneous screen coordinates (sample space, px) of a clip-space vertex.
void Homogeneous(const Ctx& ctx, const double c[4], double h[3]) {
  h[0] = double(ctx.scale[0]) * c[0] + double(ctx.offset[0]) * c[3];
  h[1] = double(ctx.scale[1]) * c[1] + double(ctx.offset[1]) * c[3];
  h[2] = c[3];
}

bool BuildSetup(const Ctx& ctx, Setup& s, const VertexOut* v0, const VertexOut* v1,
                const VertexOut* v2, const double h[3][3], const double zc[3]) {
  s.v[0] = v0;
  s.v[1] = v1;
  s.v[2] = v2;
  for (int i = 0; i < 3; ++i) {
    const double* a = h[(i + 1) % 3];
    const double* b = h[(i + 2) % 3];
    s.C[i][0] = a[1] * b[2] - a[2] * b[1];
    s.C[i][1] = a[2] * b[0] - a[0] * b[2];
    s.C[i][2] = a[0] * b[1] - a[1] * b[0];
    s.W[i] = h[i][2];
    s.zc[i] = zc[i];
  }
  s.D = s.W[0] * s.C[0][2] + s.W[1] * s.C[1][2] + s.W[2] * s.C[2][2];
  (void)ctx;
  return std::isfinite(s.D) && s.D != 0;
}

void Bary(const Setup& s, double x, double y, bool persp, double b[3]) {
  double l[3];
  for (int i = 0; i < 3; ++i) {
    l[i] = s.C[i][0] * x + s.C[i][1] * y + s.C[i][2];
    if (!persp) l[i] *= s.W[i];
  }
  double sum = l[0] + l[1] + l[2];
  for (int i = 0; i < 3; ++i) b[i] = l[i] / sum;
}

// Window z at a sample-space position (before polygon offset).
double WindowZ(const Ctx& ctx, const Setup& s, double x, double y) {
  if (s.constant) return s.z_const;
  double n = 0;
  for (int i = 0; i < 3; ++i) n += s.zc[i] * (s.C[i][0] * x + s.C[i][1] * y + s.C[i][2]);
  double z_ndc = n / s.D;
  return z_ndc * double(ctx.scale[2]) + double(ctx.offset[2]);
}

void ZSlopes(const Ctx& ctx, const Setup& s, double& dzdx, double& dzdy) {
  if (s.constant) {
    dzdx = dzdy = 0;
    return;
  }
  double nx = 0, ny = 0;
  for (int i = 0; i < 3; ++i) {
    nx += s.zc[i] * s.C[i][0];
    ny += s.zc[i] * s.C[i][1];
  }
  dzdx = nx / s.D * double(ctx.scale[2]);
  dzdy = ny / s.D * double(ctx.scale[2]);
}

void ApplyPolyOffset(const Ctx& ctx, Prim& p) {
  // Face selection.
  float scale = 0, offset = 0;
  bool enabled = false;
  if (p.polygonal) {
    if (p.front && ctx.sc.poly_offset_front_enable) {
      enabled = true;
      scale = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
      offset = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
    } else if (!p.front && ctx.sc.poly_offset_back_enable) {
      enabled = true;
      scale = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE);
      offset = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET);
    }
  } else if (ctx.sc.poly_offset_para_enable) {
    enabled = true;
    scale = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
    offset = RegF(ctx.regs, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
  }
  p.has_poly_offset = enabled;
  p.poly_offset = 0;
  if (!enabled) return;
  RX_USE(raster_poly_offset);
  double dzdx, dzdy;
  ZSlopes(ctx, p.setup, dzdx, dzdy);
  double slope = std::max(std::fabs(dzdx), std::fabs(dzdy));
  p.poly_scale = scale;
  p.poly_offset_reg = offset;
  p.poly_offset = double(scale) * (1.0 / 16.0) * slope + double(offset);
}

double ClampDepth(double z) {
  // NClamp: NaN -> 0.
  if (!(z >= 0)) return 0.0;
  if (z > 1) return 1.0;
  return z;
}

// ---------------------------------------------------------------------------
// Coverage regions.

struct Edge {
  int64_t ax, ay, dx, dy;
  bool top_left;
};

struct Convex {
  std::vector<Edge> edges;
  int64_t minx, miny, maxx, maxy;
  bool Inside(int64_t x, int64_t y) const {
    for (const Edge& e : edges) {
      int64_t v = e.dx * (y - e.ay) - e.dy * (x - e.ax);
      if (v < 0 || (v == 0 && !e.top_left)) return false;
    }
    return true;
  }
};

int64_t Area2(const std::vector<int64_t>& xs, const std::vector<int64_t>& ys) {
  int64_t a = 0;
  size_t n = xs.size();
  for (size_t i = 0; i < n; ++i) {
    size_t j = (i + 1) % n;
    a += xs[i] * ys[j] - xs[j] * ys[i];
  }
  return a;
}

// Builds a convex region from snapped vertices; orientation normalized.
Convex MakeConvex(std::vector<int64_t> xs, std::vector<int64_t> ys) {
  if (Area2(xs, ys) < 0) {
    std::reverse(xs.begin(), xs.end());
    std::reverse(ys.begin(), ys.end());
  }
  Convex c;
  size_t n = xs.size();
  c.minx = c.maxx = xs[0];
  c.miny = c.maxy = ys[0];
  for (size_t i = 0; i < n; ++i) {
    size_t j = (i + 1) % n;
    Edge e{xs[i], ys[i], xs[j] - xs[i], ys[j] - ys[i], false};
    if (e.dx == 0 && e.dy == 0) continue;
    e.top_left = e.dy < 0 || (e.dy == 0 && e.dx > 0);
    c.edges.push_back(e);
    c.minx = std::min(c.minx, xs[i]);
    c.maxx = std::max(c.maxx, xs[i]);
    c.miny = std::min(c.miny, ys[i]);
    c.maxy = std::max(c.maxy, ys[i]);
  }
  return c;
}

// ---------------------------------------------------------------------------
// Quad emission.

std::string Hex8(uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%08X", v);
  return buf;
}

template <typename CoverFn>
void EmitRegion(Ctx& ctx, const Prim& p, int64_t minx, int64_t miny, int64_t maxx, int64_t maxy,
                const CoverFn& cover) {
  RX_USE(raster_quad_order);
  RX_USE(raster_top_left_fill);
  int64_t px0 = FloorDiv(minx, kSub) - 1, px1 = FloorDiv(maxx, kSub) + 1;
  int64_t py0 = FloorDiv(miny, kSub) - 1, py1 = FloorDiv(maxy, kSub) + 1;
  px0 = std::max<int64_t>(px0, ctx.sc_l);
  py0 = std::max<int64_t>(py0, ctx.sc_t);
  px1 = std::min<int64_t>(px1, int64_t(ctx.sc_r) - 1);
  py1 = std::min<int64_t>(py1, int64_t(ctx.sc_b) - 1);
  if (px0 > px1 || py0 > py1) return;
  px0 &= ~int64_t(1);
  py0 &= ~int64_t(1);
  const uint32_t ns = ctx.sample_count;
  const bool center_only = ns > 1 && !ctx.msaa_enable;
  const uint8_t full = uint8_t((1u << ns) - 1);
  const bool persp = !ctx.sc.persp_corr_dis;
  if (!persp) RX_USE(raster_persp_corr_dis);
  Probe& probe = Probe::Get();

  for (int64_t qy = py0; qy <= py1; qy += 2) {
    for (int64_t qx = px0; qx <= px1; qx += 2) {
      uint8_t prim_cov[4] = {};
      uint8_t cov[4] = {};
      bool any = false;
      for (int i = 0; i < 4; ++i) {
        int64_t x = qx + (i & 1), y = qy + (i >> 1);
        int64_t cx = x * kSub + kSub / 2, cy = y * kSub + kSub / 2;
        uint8_t bits = 0;
        if (center_only || ns == 1) {
          if (cover(cx, cy)) bits = full;
        } else {
          for (uint32_t s = 0; s < ns; ++s) {
            if (cover(cx + ctx.samples[s].x, cy + ctx.samples[s].y)) bits |= uint8_t(1u << s);
          }
        }
        prim_cov[i] = bits;
        bool in_scissor = x >= ctx.sc_l && x < ctx.sc_r && y >= ctx.sc_t && y < ctx.sc_b;
        cov[i] = in_scissor ? bits : 0;
        any |= cov[i] != 0;
      }
      if (!any) continue;

      PixelQuad q;
      q.x = int32_t(qx);
      q.y = int32_t(qy);
      q.front_face = p.front;
      q.primitive_index = p.index;
      for (int i = 0; i < 4; ++i) {
        int64_t x = qx + (i & 1), y = qy + (i >> 1);
        double fx = double(x) + 0.5, fy = double(y) + 0.5;
        q.coverage[i] = cov[i];
        q.center_x[i] = float(fx);
        q.center_y[i] = float(fy);

        // Interpolation location(s).
        double cenx = fx, ceny = fy;
        if (prim_cov[i] != 0 && prim_cov[i] != full && !center_only && ctx.centroid_mask) {
          RX_USE(raster_centroid);
          uint32_t s = 0;
          while (!(prim_cov[i] & (1u << s))) ++s;
          cenx = fx + ctx.samples[s].x / 16.0;
          ceny = fy + ctx.samples[s].y / 16.0;
        }
        double bc[3] = {1, 0, 0}, bcen[3] = {1, 0, 0};
        if (!p.setup.constant) {
          Bary(p.setup, fx, fy, persp, bc);
          Bary(p.setup, cenx, ceny, persp, bcen);
        }
        for (uint32_t r = 0; r < kMaxInterpolators; ++r) {
          for (int c = 0; c < 4; ++c) {
            float value;
            if (p.setup.constant) {
              value = p.setup.v[0]->interp[r][c];
            } else if (ctx.flat_mask & (1u << r)) {
              value = p.provoking->interp[r][c];
            } else {
              const double* b = (ctx.centroid_mask & (1u << r)) ? bcen : bc;
              double sum = b[0] * double(p.setup.v[0]->interp[r][c]) +
                           b[1] * double(p.setup.v[1]->interp[r][c]) +
                           b[2] * double(p.setup.v[2]->interp[r][c]);
              value = float(sum);
            }
            q.interp[i][r][c] = value;
          }
        }

        // Depth.
        for (uint32_t s = 0; s < ns; ++s) {
          double sx = fx, sy = fy;
          if (ns > 1 && !center_only) {
            sx += ctx.samples[s].x / 16.0;
            sy += ctx.samples[s].y / 16.0;
          }
          q.sample_z[i][s] = ClampDepth(WindowZ(ctx, p.setup, sx, sy) + p.poly_offset);
        }

        // ps_param_gen.
        float gx = float(x), gy = float(y);
        float* pg = q.param_gen[i];
        if (p.point) {
          pg[0] = std::fabs(gx);
          pg[1] = -std::fabs(gy);
          double w = p.point_rect[2] - p.point_rect[0], h = p.point_rect[3] - p.point_rect[1];
          double u = w > 0 ? (fx - p.point_rect[0]) / w : 0.0;
          double v = h > 0 ? (fy - p.point_rect[1]) / h : 0.0;
          pg[2] = float(std::clamp(u, 0.0, 1.0));
          pg[3] = float(std::clamp(v, 0.0, 1.0));
        } else {
          pg[0] = std::fabs(gx);
          pg[1] = std::fabs(gy);
          if (p.polygonal && !p.front) pg[0] = -pg[0];
          pg[2] = p.line ? AsF32(0x80000000u) : 0.0f;
          pg[3] = 0.0f;
        }

        if (probe.Wants(int32_t(x), int32_t(y))) {
          char buf[512];
          std::snprintf(buf, sizeof(buf),
                        "prim %u cov 0x%X (prim 0x%X) front %d bary_center (%.17g %.17g %.17g) "
                        "bary_centroid (%.17g %.17g %.17g) poly_offset %.17g",
                        p.index, cov[i], prim_cov[i], int(p.front), bc[0], bc[1], bc[2], bcen[0],
                        bcen[1], bcen[2], p.poly_offset);
          probe.Log(int32_t(x), int32_t(y), "raster", buf);
          std::string zs = "z";
          for (uint32_t s = 0; s < ns; ++s) {
            std::snprintf(buf, sizeof(buf), " s%u=%.17g", s, q.sample_z[i][s]);
            zs += buf;
          }
          probe.Log(int32_t(x), int32_t(y), "raster", zs);
          uint32_t mask = p.setup.v[0] ? p.setup.v[0]->interp_written_mask : 0;
          for (uint32_t r = 0; r < kMaxInterpolators; ++r) {
            if (!(mask & (1u << r))) continue;
            probe.Log(int32_t(x), int32_t(y), "raster",
                      "r" + std::to_string(r) + " " + FormatFloat4(q.interp[i][r]));
          }
          probe.Log(int32_t(x), int32_t(y), "raster", "param_gen " + FormatFloat4(pg));
        }
      }
      ++ctx.stats.quads;
      ctx.emit(q);
    }
  }
}

void EmitConvex(Ctx& ctx, const Prim& p, const Convex& c) {
  if (c.edges.size() < 3) return;
  EmitRegion(ctx, p, c.minx, c.miny, c.maxx, c.maxy,
             [&c](int64_t x, int64_t y) { return c.Inside(x, y); });
}

// ---------------------------------------------------------------------------
// Primitive kinds.

bool VertexKilled(const VertexOut& v) {
  return v.point_written && (AsU32(v.point_size_edge_kill[2]) & 0x7FFFFFFFu) != 0;
}

void LoadCV(const Ctx& ctx, const VertexOut& v, CV& out) {
  float c[4];
  ToClip(v, ctx, c);
  for (int i = 0; i < 4; ++i) out.c[i] = double(c[i]);
  out.original = true;
}

bool SnapCV(const Ctx& ctx, const CV& v, int64_t& x, int64_t& y, float s[3]) {
  Project(ctx, v.c, s);
  return RoundToGrid(ctx, s[0], x) && RoundToGrid(ctx, s[1], y);
}

void DrawPoint(Ctx& ctx, const VertexOut& v, uint32_t index, bool polygonal, bool front, bool point_flag);
void DrawLine(Ctx& ctx, const VertexOut& v0, const VertexOut& v1, const VertexOut& provoking,
              uint32_t index, bool polygonal, bool front);

bool CenterClipped(const Ctx& ctx, const CV& v) {
  for (const Plane& pl : ctx.center_planes) {
    if (!(PlaneDist(pl, v.c) >= 0)) return true;
  }
  return false;
}

void DrawPoint(Ctx& ctx, const VertexOut& v, uint32_t index, bool polygonal, bool front,
               bool point_flag) {
  RX_USE(raster_point_size);
  RX_USE(raster_point_rect);
  RX_USE(raster_point_coord);
  CV cv;
  LoadCV(ctx, v, cv);
  if (HasNaN(&cv, 1)) {
    RX_USE(raster_nan_position);
    ++ctx.stats.primitives_culled;
    return;
  }
  if (UcpCulled(ctx, &cv, 1) || CenterClipped(ctx, cv)) {
    RX_USE(raster_point_clip);
    ++ctx.stats.primitives_culled;
    return;
  }
  float s[3];
  Project(ctx, cv.c, s);
  float dx = ctx.point_diam_const[0], dy = ctx.point_diam_const[1];
  if (v.point_written && int32_t(AsU32(v.point_size_edge_kill[0])) >= 0) {
    int32_t bits = int32_t(AsU32(v.point_size_edge_kill[0]));
    bits = std::max(bits, int32_t(ctx.point_min_bits));
    bits = std::min(bits, int32_t(ctx.point_max_bits));
    dx = dy = AsF32(uint32_t(bits));
  }
  float hx = dx * 0.5f, hy = dy * 0.5f;
  float L = s[0] - hx, R = s[0] + hx, T = s[1] - hy, B = s[1] + hy;
  int64_t l, r, t, b;
  if (!RoundToGrid(ctx, L, l) || !RoundToGrid(ctx, R, r) || !RoundToGrid(ctx, T, t) ||
      !RoundToGrid(ctx, B, b)) {
    RX_USE(raster_fixed_range);
    ctx.note("point outside the fixed-point range dropped");
    ++ctx.stats.primitives_culled;
    return;
  }
  if (l >= r || t >= b) {
    ++ctx.stats.primitives_culled;
    return;
  }
  Prim p;
  p.setup.v[0] = p.setup.v[1] = p.setup.v[2] = &v;
  p.setup.constant = true;
  p.setup.z_const = double(s[2]);
  p.provoking = &v;
  p.index = index;
  p.front = front;
  p.polygonal = polygonal;
  p.point = point_flag;
  p.point_rect[0] = L;
  p.point_rect[1] = T;
  p.point_rect[2] = R;
  p.point_rect[3] = B;
  ApplyPolyOffset(ctx, p);
  EmitRegion(ctx, p, l, t, r, b,
             [=](int64_t x, int64_t y) { return x >= l && x < r && y >= t && y < b; });
}

void DrawLine(Ctx& ctx, const VertexOut& v0, const VertexOut& v1, const VertexOut& provoking,
              uint32_t index, bool polygonal, bool front) {
  RX_USE(raster_line_width);
  RX_USE(raster_line_quad);
  CV c[2];
  LoadCV(ctx, v0, c[0]);
  LoadCV(ctx, v1, c[1]);
  if (HasNaN(c, 2)) {
    RX_USE(raster_nan_position);
    ++ctx.stats.primitives_culled;
    return;
  }
  if (UcpCulled(ctx, c, 2)) {
    ++ctx.stats.primitives_culled;
    return;
  }
  double h[3][3], zc[3];
  Homogeneous(ctx, c[0].c, h[0]);
  Homogeneous(ctx, c[1].c, h[1]);
  zc[0] = c[0].c[2];
  zc[1] = c[1].c[2];
  CV a = c[0], b = c[1];
  bool clipped = false;
  if (!ClipSegment(ctx, a, b, clipped)) {
    ++ctx.stats.primitives_culled;
    return;
  }
  if (clipped) ++ctx.stats.primitives_clipped;
  int64_t x0, y0, x1, y1;
  float s0[3], s1[3];
  if (!SnapCV(ctx, a, x0, y0, s0) || !SnapCV(ctx, b, x1, y1, s1)) {
    RX_USE(raster_fixed_range);
    ctx.note("line outside the fixed-point range dropped");
    ++ctx.stats.primitives_culled;
    return;
  }
  int64_t dx = x1 - x0, dy = y1 - y0;
  int64_t hw = ctx.line_half_width;
  if ((dx == 0 && dy == 0) || hw == 0) {
    ++ctx.stats.primitives_culled;
    return;
  }
  bool xmajor = (dx < 0 ? -dx : dx) >= (dy < 0 ? -dy : dy);
  std::vector<int64_t> xs, ys;
  if (xmajor) {
    xs = {x0, x1, x1, x0};
    ys = {y0 - hw, y1 - hw, y1 + hw, y0 + hw};
  } else {
    xs = {x0 - hw, x0 + hw, x1 + hw, x1 - hw};
    ys = {y0, y0, y1, y1};
  }
  // Interpolation triangle: v0, v1, v0 shifted along the minor axis.
  h[2][0] = h[0][0] + (xmajor ? 0.0 : h[0][2]);
  h[2][1] = h[0][1] + (xmajor ? h[0][2] : 0.0);
  h[2][2] = h[0][2];
  zc[2] = zc[0];
  Prim p;
  if (!BuildSetup(ctx, p.setup, &v0, &v1, &v0, h, zc)) {
    ctx.note("line with a degenerate interpolation setup dropped");
    ++ctx.stats.primitives_culled;
    return;
  }
  p.provoking = &provoking;
  p.index = index;
  p.front = front;
  p.polygonal = polygonal;
  p.line = !polygonal;
  ApplyPolyOffset(ctx, p);
  EmitConvex(ctx, p, MakeConvex(xs, ys));
}

void DrawTriangle(Ctx& ctx, const VertexOut* v[3], const VertexOut& provoking, uint32_t index) {
  CV c[3];
  for (int i = 0; i < 3; ++i) LoadCV(ctx, *v[i], c[i]);
  if (HasNaN(c, 3)) {
    RX_USE(raster_nan_position);
    ++ctx.stats.primitives_culled;
    return;
  }
  if (UcpCulled(ctx, c, 3)) {
    ++ctx.stats.primitives_culled;
    return;
  }
  std::vector<CV> poly(c, c + 3);
  bool clipped = false;
  if (!ClipPolygon(ctx, poly, clipped) || poly.size() < 3) {
    ++ctx.stats.primitives_culled;
    return;
  }
  if (clipped) {
    RX_USE(raster_clip_fan);
    ++ctx.stats.primitives_clipped;
  }
  std::vector<int64_t> xs(poly.size()), ys(poly.size());
  for (size_t i = 0; i < poly.size(); ++i) {
    float s[3];
    if (!SnapCV(ctx, poly[i], xs[i], ys[i], s)) {
      RX_USE(raster_fixed_range);
      ctx.note("triangle outside the fixed-point range dropped");
      ++ctx.stats.primitives_culled;
      return;
    }
  }
  int64_t area = Area2(xs, ys);
  if (area == 0) {
    RX_USE(raster_cull_degenerate);
    ++ctx.stats.primitives_culled;
    return;
  }
  RX_USE(raster_face_orientation);
  bool cw = area > 0;
  bool front = ctx.sc.face ? cw : !cw;
  if ((front && ctx.sc.cull_front) || (!front && ctx.sc.cull_back)) {
    ++ctx.stats.primitives_culled;
    return;
  }

  if (ctx.sc.poly_mode == xenos::PolygonModeEnable::kDualMode) {
    xenos::PolygonType type = front ? ctx.sc.polymode_front_ptype : ctx.sc.polymode_back_ptype;
    if (type != xenos::PolygonType::kTriangles) {
      RX_USE(raster_polymode);
      if (type == xenos::PolygonType::kPoints) {
        for (int i = 0; i < 3; ++i) DrawPoint(ctx, *v[i], index, true, front, false);
      } else {
        for (int i = 0; i < 3; ++i) {
          const VertexOut& a = *v[i];
          const VertexOut& b = *v[(i + 1) % 3];
          if (ctx.clip.boundary_edge_flag_ena && a.point_written &&
              a.point_size_edge_kill[1] == 0.0f) {
            continue;
          }
          DrawLine(ctx, a, b, provoking, index, true, front);
        }
      }
      return;
    }
  }

  double h[3][3], zc[3];
  for (int i = 0; i < 3; ++i) {
    Homogeneous(ctx, c[i].c, h[i]);
    zc[i] = c[i].c[2];
  }
  Prim p;
  RX_USE(raster_interp_unclipped);
  RX_USE(raster_persp_interp);
  RX_USE(raster_interp_precision);
  RX_USE(raster_depth_plane);
  RX_USE(raster_depth_clamp);
  if (!BuildSetup(ctx, p.setup, v[0], v[1], v[2], h, zc)) {
    ctx.note("triangle with a degenerate homogeneous setup dropped");
    ++ctx.stats.primitives_culled;
    return;
  }
  p.provoking = &provoking;
  p.index = index;
  p.front = front;
  p.polygonal = true;
  ApplyPolyOffset(ctx, p);
  if (poly.size() == 3) {
    EmitConvex(ctx, p, MakeConvex(xs, ys));
    return;
  }
  for (size_t k = 1; k + 1 < poly.size(); ++k) {
    std::vector<int64_t> tx = {xs[0], xs[k], xs[k + 1]}, ty = {ys[0], ys[k], ys[k + 1]};
    int64_t a = Area2(tx, ty);
    if (a == 0 || (a > 0) != (area > 0)) continue;
    EmitConvex(ctx, p, MakeConvex(tx, ty));
  }
}

void DrawRect(Ctx& ctx, const VertexOut* v[3], const VertexOut& provoking, uint32_t index) {
  RX_USE(raster_rect_fourth_vertex);
  RX_USE(raster_rect_edge_select);
  RX_USE(raster_nonpolygonal_front);
  CV c[3];
  float cf[3][4];
  for (int i = 0; i < 3; ++i) {
    ToClip(*v[i], ctx, cf[i]);
    LoadCV(ctx, *v[i], c[i]);
  }
  if (HasNaN(c, 3)) {
    RX_USE(raster_nan_position);
    ++ctx.stats.primitives_culled;
    return;
  }
  // Diagonal selection from screen positions.
  float s[3][3];
  for (int i = 0; i < 3; ++i) Project(ctx, c[i].c, s[i]);
  auto len2 = [&](int a, int b) {
    double dx = double(s[a][0]) - double(s[b][0]), dy = double(s[a][1]) - double(s[b][1]);
    return dx * dx + dy * dy;
  };
  double e12 = len2(1, 2), e20 = len2(2, 0), e01 = len2(0, 1);
  int order[3];
  if (e12 > e20 && e12 > e01) {
    order[0] = 0, order[1] = 1, order[2] = 2;
  } else if (e20 > e01) {
    order[0] = 1, order[1] = 2, order[2] = 0;
  } else {
    order[0] = 2, order[1] = 0, order[2] = 1;
  }
  CV s0 = c[order[0]], s1 = c[order[1]], s2 = c[order[2]], s3;
  s3.original = true;  // computed in float32 like a vertex
  for (int k = 0; k < 4; ++k) {
    float d = cf[order[1]][k] - cf[order[0]][k];
    s3.c[k] = double(d + cf[order[2]][k]);
  }
  CV quad[4] = {s0, s1, s3, s2};
  if (UcpCulled(ctx, quad, 4)) {
    ++ctx.stats.primitives_culled;
    return;
  }
  std::vector<CV> poly(quad, quad + 4);
  bool clipped = false;
  if (!ClipPolygon(ctx, poly, clipped) || poly.size() < 3) {
    ++ctx.stats.primitives_culled;
    return;
  }
  if (clipped) ++ctx.stats.primitives_clipped;
  std::vector<int64_t> xs(poly.size()), ys(poly.size());
  for (size_t i = 0; i < poly.size(); ++i) {
    float sp[3];
    if (!SnapCV(ctx, poly[i], xs[i], ys[i], sp)) {
      RX_USE(raster_fixed_range);
      ctx.note("rectangle outside the fixed-point range dropped");
      ++ctx.stats.primitives_culled;
      return;
    }
  }
  int64_t area = Area2(xs, ys);
  if (area == 0) {
    RX_USE(raster_cull_degenerate);
    ++ctx.stats.primitives_culled;
    return;
  }
  double h[3][3], zc[3];
  const CV* sv[3] = {&s0, &s1, &s2};
  for (int i = 0; i < 3; ++i) {
    Homogeneous(ctx, sv[i]->c, h[i]);
    zc[i] = sv[i]->c[2];
  }
  Prim p;
  if (!BuildSetup(ctx, p.setup, v[order[0]], v[order[1]], v[order[2]], h, zc)) {
    ctx.note("rectangle with a degenerate homogeneous setup dropped");
    ++ctx.stats.primitives_culled;
    return;
  }
  RX_USE(raster_interp_unclipped);
  RX_USE(raster_depth_plane);
  RX_USE(raster_depth_clamp);
  p.provoking = &provoking;
  p.index = index;
  p.front = true;
  p.polygonal = false;
  ApplyPolyOffset(ctx, p);
  if (!clipped) {
    EmitConvex(ctx, p, MakeConvex(xs, ys));
    return;
  }
  RX_USE(raster_clip_fan);
  for (size_t k = 1; k + 1 < poly.size(); ++k) {
    std::vector<int64_t> tx = {xs[0], xs[k], xs[k + 1]}, ty = {ys[0], ys[k], ys[k + 1]};
    int64_t a = Area2(tx, ty);
    if (a == 0 || (a > 0) != (area > 0)) continue;
    EmitConvex(ctx, p, MakeConvex(tx, ty));
  }
}

// ---------------------------------------------------------------------------
// Primitive assembly.

enum class Kind { kPoint, kLine, kTri, kRect };

struct AsmPrim {
  Kind kind;
  uint32_t v[3];
  uint32_t provoking;
  uint32_t index;
};

void AssembleSegment(xenos::PrimitiveType type, uint32_t first, uint32_t n, bool last_provoking,
                     uint32_t& prim_index, std::vector<AsmPrim>& out, Note& note) {
  using PT = xenos::PrimitiveType;
  auto add = [&](Kind k, uint32_t a, uint32_t b, uint32_t c, uint32_t prov, bool new_prim = true) {
    if (new_prim) ++prim_index;
    out.push_back({k, {first + a, first + b, first + c}, first + prov, prim_index - 1});
  };
  switch (type) {
    case PT::kPointList:
      for (uint32_t i = 0; i < n; ++i) add(Kind::kPoint, i, i, i, i);
      break;
    case PT::kLineList:
      for (uint32_t i = 0; i + 1 < n; i += 2) add(Kind::kLine, i, i + 1, i + 1, last_provoking ? i + 1 : i);
      break;
    case PT::kLineStrip:
    case PT::k2DLineStrip:
      for (uint32_t i = 0; i + 1 < n; ++i) add(Kind::kLine, i, i + 1, i + 1, last_provoking ? i + 1 : i);
      break;
    case PT::kLineLoop:
      RX_USE(raster_line_loop);
      for (uint32_t i = 0; i + 1 < n; ++i) add(Kind::kLine, i, i + 1, i + 1, last_provoking ? i + 1 : i);
      if (n >= 2) add(Kind::kLine, n - 1, 0, 0, last_provoking ? 0 : n - 1);
      break;
    case PT::kTriangleList:
    case PT::kTriangleWithWFlags:
      for (uint32_t i = 0; i + 2 < n; i += 3) add(Kind::kTri, i, i + 1, i + 2, last_provoking ? i + 2 : i);
      break;
    case PT::kTriangleStrip:
    case PT::k2DTriStrip:
      RX_USE(raster_strip_winding);
      for (uint32_t i = 0; i + 2 < n; ++i) {
        uint32_t prov = last_provoking ? i + 2 : i;
        if (i & 1) {
          add(Kind::kTri, i + 1, i, i + 2, prov);
        } else {
          add(Kind::kTri, i, i + 1, i + 2, prov);
        }
      }
      break;
    case PT::kTriangleFan:
      RX_USE(raster_fan_order);
      RX_USE(raster_provoking_fan);
      for (uint32_t i = 0; i + 2 < n; ++i) add(Kind::kTri, 0, i + 1, i + 2, last_provoking ? i + 2 : i + 1);
      break;
    case PT::kPolygon:
      RX_USE(raster_quad_strip_polygon);
      for (uint32_t i = 0; i + 2 < n; ++i) add(Kind::kTri, 0, i + 1, i + 2, last_provoking ? i + 2 : 0);
      break;
    case PT::kRectangleList:
    case PT::k2DCopyRectListV0:
    case PT::k2DCopyRectListV1:
    case PT::k2DCopyRectListV2:
    case PT::k2DCopyRectListV3:
    case PT::k2DFillRectList:
      for (uint32_t i = 0; i + 2 < n; i += 3) add(Kind::kRect, i, i + 1, i + 2, last_provoking ? i + 2 : i);
      break;
    case PT::kQuadList:
      RX_USE(raster_quad_list_split);
      for (uint32_t i = 0; i + 3 < n; i += 4) {
        uint32_t prov = last_provoking ? i + 3 : i;
        add(Kind::kTri, i, i + 1, i + 2, prov);
        add(Kind::kTri, i, i + 2, i + 3, prov, false);
      }
      break;
    case PT::kQuadStrip:
      RX_USE(raster_quad_strip_polygon);
      for (uint32_t i = 0; i + 3 < n; i += 2) {
        uint32_t a = i, b = i + 1, c = i + 3, d = i + 2;
        uint32_t prov = last_provoking ? c : a;
        add(Kind::kTri, a, b, c, prov);
        add(Kind::kTri, a, c, d, prov, false);
      }
      break;
    default:
      note("primitive type " + std::to_string(uint32_t(type)) + " not rasterized");
      break;
  }
  switch (type) {
    case PT::kTriangleWithWFlags:
    case PT::k2DCopyRectListV0:
    case PT::k2DCopyRectListV1:
    case PT::k2DCopyRectListV2:
    case PT::k2DCopyRectListV3:
    case PT::k2DFillRectList:
    case PT::k2DLineStrip:
    case PT::k2DTriStrip:
      RX_USE(raster_unsupported_types);
      note("primitive type " + std::to_string(uint32_t(type)) + " rasterized by an assumed mapping");
      break;
    default:
      break;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API.

uint32_t RasterSampleCount(xenos::MsaaSamples msaa) {
  switch (msaa) {
    case xenos::MsaaSamples::k2X:
      return 2;
    case xenos::MsaaSamples::k4X:
      return 4;
    default:
      return 1;
  }
}

std::span<const SamplePosition> RasterSamplePositions(xenos::MsaaSamples msaa) {
  switch (msaa) {
    case xenos::MsaaSamples::k2X:
      return kSamples2x;
    case xenos::MsaaSamples::k4X:
      return kSamples4x;
    default:
      return kSamples1x;
  }
}

uint32_t RasterCentroidMask(const RegisterFile& regs) {
  auto misc = regs.Get<reg::SQ_CONTEXT_MISC>();
  auto icntl = regs.Get<reg::SQ_INTERPOLATOR_CNTL>();
  auto msaa = regs.Get<reg::RB_SURFACE_INFO>().msaa_samples;
  uint32_t center = xenos::GetInterpolatorSamplingPattern(msaa, misc.sc_sample_cntl,
                                                          icntl.sampling_pattern);
  return ~center & ((1u << kMaxInterpolators) - 1);
}

uint32_t RasterFlatMask(const RegisterFile& regs) {
  return regs.Get<reg::SQ_INTERPOLATOR_CNTL>().param_shade & ((1u << kMaxInterpolators) - 1);
}

void RasterizeDraw(const RegisterFile& regs, const RasterInput& in, const QuadEmitFn& emit,
                   DrawStats& stats) {
  Ctx ctx(regs, emit, stats);
  InitCtx(ctx);
  RX_USE(raster_provoking_vertex);
  if (ctx.flat_mask) RX_USE(raster_flat_param_shade);
  if (ctx.vtx.pix_center == xenos::PixelCenter::kOGLHalf) {
    RX_USE(raster_param_gen_ogl_center);
  }
  RX_USE(raster_param_gen);
  const bool last = ctx.sc.provoking_vtx_last != 0;

  // Assembly, segment by segment.
  std::vector<AsmPrim> prims;
  uint32_t prim_index = 0;
  uint32_t n = uint32_t(in.vertices.size());
  uint32_t seg_start = 0;
  for (uint32_t i = 1; i <= n; ++i) {
    bool cut = i == n || (i < in.restart_before.size() && in.restart_before[i]);
    if (!cut) continue;
    AssembleSegment(in.prim, seg_start, i - seg_start, last, prim_index, prims, ctx.note);
    seg_start = i;
  }
  stats.primitives_in += prim_index;

  if (Probe::Get().log_vertices) {
    for (const VertexOut& v : in.vertices) {
      CV cv;
      LoadCV(ctx, v, cv);
      float s[3];
      Project(ctx, cv.c, s);
      char buf[256];
      std::snprintf(buf, sizeof(buf), "clip (%.9g %.9g %.9g %.9g) screen (%.9g %.9g %.9g)",
                    cv.c[0], cv.c[1], cv.c[2], cv.c[3], s[0], s[1], s[2]);
      Probe::Get().LogVertex(v.vertex_index, "raster", buf);
    }
  }

  for (const AsmPrim& ap : prims) {
    const VertexOut* v[3] = {&in.vertices[ap.v[0]], &in.vertices[ap.v[1]], &in.vertices[ap.v[2]]};
    // Vertex kill.
    uint32_t nv = ap.kind == Kind::kPoint ? 1 : ap.kind == Kind::kLine ? 2 : 3;
    uint32_t killed = 0;
    for (uint32_t i = 0; i < nv; ++i) killed += VertexKilled(*v[i]) ? 1 : 0;
    if (killed) {
      RX_USE(raster_vertex_kill);
      if (ctx.clip.vtx_kill_or || killed == nv) {
        ++stats.primitives_culled;
        continue;
      }
    }
    const VertexOut& prov = in.vertices[ap.provoking];
    switch (ap.kind) {
      case Kind::kPoint:
        RX_USE(raster_nonpolygonal_front);
        DrawPoint(ctx, *v[0], ap.index, false, true, true);
        break;
      case Kind::kLine:
        RX_USE(raster_nonpolygonal_front);
        DrawLine(ctx, *v[0], *v[1], prov, ap.index, false, true);
        break;
      case Kind::kTri:
        DrawTriangle(ctx, v, prov, ap.index);
        break;
      case Kind::kRect:
        DrawRect(ctx, v, prov, ap.index);
        break;
    }
  }
}

}  // namespace refxenos
