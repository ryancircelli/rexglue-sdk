// refxenos raster unit tests: known answers for primitive assembly, fill rule,
// MSAA coverage, interpolation, flat shading, centroid, param_gen, points,
// rect lists, scissor, clipping, depth and polygon offset.
//
// Every scene uses PA_CL_VTE_CNTL = 0 with vtx_w0_fmt set, so positions are
// clip space with W taken as written; with W = 1 and no viewport scale the
// position is the window position. pix_center = kOGLHalf so no half-pixel
// shift is added (pixel (x, y) has its center at (x + 0.5, y + 0.5)).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "core/common.h"
#include "core/raster.h"

using namespace refxenos;
namespace rg = rex::graphics;
using PT = xenos::PrimitiveType;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond)                                                         \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)
#define CHECK_EQ(a, b)                                                                     \
  do {                                                                                     \
    ++g_checks;                                                                            \
    auto a_ = (a);                                                                         \
    auto b_ = (b);                                                                         \
    if (!(a_ == b_)) {                                                                     \
      std::printf("FAIL %s:%d: %s = %.17g, expected %.17g\n", __FILE__, __LINE__, #a,      \
                  double(a_), double(b_));                                                 \
      ++g_failures;                                                                        \
    }                                                                                      \
  } while (0)

template <typename R>
static void Set(RegisterFile& rf, R r) {
  rf[R::register_index] = r.value;
}
static void SetF(RegisterFile& rf, uint32_t idx, float f) { rf[idx] = AsU32(f); }

// Baseline state: 1x, no viewport scale/offset, W as written, clipping off,
// 64x64 window and screen scissor, OpenGL pixel centers, no culling.
static RegisterFile* Base(xenos::MsaaSamples msaa = xenos::MsaaSamples::k1X, bool msaa_enable = false) {
  auto* rf = new RegisterFile;
  std::memset(rf->values, 0, sizeof(rf->values));
  reg::PA_CL_VTE_CNTL vte{};
  vte.vtx_w0_fmt = 1;
  Set(*rf, vte);
  reg::PA_CL_CLIP_CNTL clip{};
  clip.clip_disable = 1;
  Set(*rf, clip);
  reg::PA_SU_VTX_CNTL vtx{};
  vtx.pix_center = xenos::PixelCenter::kOGLHalf;
  vtx.round_mode = xenos::VertexRounding::kTruncate;
  vtx.quant_mode = xenos::VertexQuantization::k_1_16th;
  Set(*rf, vtx);
  reg::PA_SU_SC_MODE_CNTL sc{};
  sc.msaa_enable = msaa_enable ? 1 : 0;
  Set(*rf, sc);
  reg::RB_SURFACE_INFO si{};
  si.surface_pitch = 64;
  si.msaa_samples = msaa;
  Set(*rf, si);
  reg::PA_SC_WINDOW_SCISSOR_TL wtl{};
  wtl.window_offset_disable = 1;
  Set(*rf, wtl);
  reg::PA_SC_WINDOW_SCISSOR_BR wbr{};
  wbr.br_x = 64;
  wbr.br_y = 64;
  Set(*rf, wbr);
  reg::PA_SC_SCREEN_SCISSOR_BR sbr{};
  sbr.br_x = 64;
  sbr.br_y = 64;
  Set(*rf, sbr);
  reg::SQ_CONTEXT_MISC misc{};
  misc.sc_sample_cntl = xenos::SampleControl::kCentersOnly;
  Set(*rf, misc);
  return rf;
}

static VertexOut V(float x, float y, float z = 0.5f, float w = 1.0f) {
  VertexOut v;
  v.position[0] = x;
  v.position[1] = y;
  v.position[2] = z;
  v.position[3] = w;
  v.position_written = true;
  v.interp_written_mask = 0xF;
  return v;
}

struct Pix {
  uint8_t cov = 0;
  int hits = 0;
  float interp[kMaxInterpolators][4];
  float param_gen[4];
  double z[kMaxSamples];
  bool front = true;
  uint32_t prim = 0;
};

struct Result {
  std::map<std::pair<int, int>, Pix> px;  // covered pixels only
  DrawStats stats;
  uint64_t quads = 0;
  int Covered() const { return int(px.size()); }
  const Pix* At(int x, int y) const {
    auto it = px.find({x, y});
    return it == px.end() ? nullptr : &it->second;
  }
  bool AllOnce() const {
    for (auto& [k, p] : px)
      if (p.hits != 1) return false;
    return true;
  }
};

static Result Draw(const RegisterFile& rf, PT prim, const std::vector<VertexOut>& verts,
                   const std::vector<uint8_t>& restart = {}) {
  Result r;
  RasterInput in;
  in.prim = prim;
  in.vertices = verts;
  in.restart_before = restart;
  RasterizeDraw(
      rf, in,
      [&](const PixelQuad& q) {
        ++r.quads;
        CHECK((q.x & 1) == 0 && (q.y & 1) == 0);
        for (int i = 0; i < 4; ++i) {
          if (!q.coverage[i]) continue;
          Pix& p = r.px[{q.x + (i & 1), q.y + (i >> 1)}];
          p.cov |= q.coverage[i];
          ++p.hits;
          std::memcpy(p.interp, q.interp[i], sizeof(p.interp));
          std::memcpy(p.param_gen, q.param_gen[i], sizeof(p.param_gen));
          std::memcpy(p.z, q.sample_z[i], sizeof(p.z));
          p.front = q.front_face;
          p.prim = q.primitive_index;
        }
      },
      r.stats);
  return r;
}

static bool CoversRect(const Result& r, int x0, int y0, int x1, int y1) {
  if (r.Covered() != (x1 - x0) * (y1 - y0)) return false;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x)
      if (!r.At(x, y)) return false;
  return true;
}

static int UlpDiff(float a, float b) {
  int32_t ia = int32_t(AsU32(a)), ib = int32_t(AsU32(b));
  if (ia < 0) ia = int32_t(0x80000000u) - ia;
  if (ib < 0) ib = int32_t(0x80000000u) - ib;
  return std::abs(ia - ib);
}

// Sources: xenia-canary@cf206a3b draw_util.h:128-152 (inferred Xenos
// guest sample order); raster.cpp raster_top_left_fill (D3D edge ownership)
// and raster_persp_interp / raster_interp_precision (smooth interpolation
// with explicitly assumed arithmetic precision).
// --- sample positions -------------------------------------------------------
static void TestSamplePositions() {
  CHECK_EQ(RasterSampleCount(xenos::MsaaSamples::k1X), 1u);
  CHECK_EQ(RasterSampleCount(xenos::MsaaSamples::k2X), 2u);
  CHECK_EQ(RasterSampleCount(xenos::MsaaSamples::k4X), 4u);
  auto s2 = RasterSamplePositions(xenos::MsaaSamples::k2X);
  const int e2[2][2] = {{-4, 4}, {4, -4}};
  for (int i = 0; i < 2; ++i) {
    CHECK_EQ(s2[i].x, e2[i][0]);
    CHECK_EQ(s2[i].y, e2[i][1]);
  }
  auto s4 = RasterSamplePositions(xenos::MsaaSamples::k4X);
  const int e4[4][2] = {{-2, -2}, {2, 2}, {-6, 6}, {6, -6}};
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(s4[i].x, e4[i][0]);
    CHECK_EQ(s4[i].y, e4[i][1]);
  }
}

// --- fill rule ----------------------------------------------------------------
// A square with its edges through pixel centers, split along the diagonal.
// Top-left rule: left and top edges own their centers, right and bottom do
// not, and the shared diagonal is owned by exactly one triangle.
static void TestFillRuleSquare() {
  RegisterFile* rf = Base();
  for (int winding = 0; winding < 2; ++winding) {
    std::vector<VertexOut> v;
    if (winding == 0) {
      v = {V(0.5f, 0.5f), V(4.5f, 0.5f), V(4.5f, 4.5f), V(0.5f, 0.5f), V(4.5f, 4.5f), V(0.5f, 4.5f)};
    } else {
      v = {V(0.5f, 0.5f), V(4.5f, 4.5f), V(4.5f, 0.5f), V(0.5f, 0.5f), V(0.5f, 4.5f), V(4.5f, 4.5f)};
    }
    Result r = Draw(*rf, PT::kTriangleList, v);
    CHECK(CoversRect(r, 0, 0, 4, 4));
    CHECK(r.AllOnce());
  }
  // The other diagonal.
  std::vector<VertexOut> v = {V(4.5f, 0.5f), V(4.5f, 4.5f), V(0.5f, 4.5f),
                              V(4.5f, 0.5f), V(0.5f, 4.5f), V(0.5f, 0.5f)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  CHECK(CoversRect(r, 0, 0, 4, 4));
  CHECK(r.AllOnce());
  delete rf;
}

// Shared diagonal through sample points: every guest sample is owned once,
// for either winding and either diagonal. Pixel hit counts cannot establish
// this under MSAA because different triangles can own disjoint samples.
static void TestSharedSamples() {
  for (auto msaa : {xenos::MsaaSamples::k1X, xenos::MsaaSamples::k2X,
                    xenos::MsaaSamples::k4X}) {
    auto* rf = Base(msaa, true);
    uint32_t ns = RasterSampleCount(msaa);
    for (int diagonal = 0; diagonal < 2; ++diagonal) {
      for (int reverse = 0; reverse < 2; ++reverse) {
        std::vector<VertexOut> v = diagonal == 0
          ? std::vector<VertexOut>{V(0,0), V(8,0), V(8,8), V(0,0), V(8,8), V(0,8)}
          : std::vector<VertexOut>{V(0,0), V(8,0), V(0,8), V(8,0), V(8,8), V(0,8)};
        if (reverse) { std::swap(v[0],v[2]); std::swap(v[3],v[5]); }
        uint32_t hits[8][8][4] = {};
        DrawStats stats;
        RasterizeDraw(*rf, {PT::kTriangleList, v, {}}, [&](const PixelQuad& q) {
          for (int p = 0; p < 4; ++p) {
            int x=q.x+(p&1), y=q.y+(p>>1);
            if (x<0 || x>=8 || y<0 || y>=8) { CHECK(q.coverage[p]==0); continue; }
            for (uint32_t sm=0; sm<ns; ++sm)
              hits[y][x][sm] += (q.coverage[p] >> sm) & 1;
          }
        }, stats);
        for (auto& row : hits) for (auto& pixel : row)
          for (uint32_t sm=0; sm<ns; ++sm) CHECK_EQ(pixel[sm], 1u);
      }
    }
    delete rf;
  }
}

// A fan of 8 triangles around a center pixel center: every pixel once.
static void TestFillRuleFan() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v = {V(8.5f, 8.5f)};
  const float ring[9][2] = {{2.5f, 2.5f}, {8.5f, 2.5f}, {14.5f, 2.5f}, {14.5f, 8.5f}, {14.5f, 14.5f},
                            {8.5f, 14.5f}, {2.5f, 14.5f}, {2.5f, 8.5f}, {2.5f, 2.5f}};
  for (auto& p : ring) v.push_back(V(p[0], p[1]));
  Result r = Draw(*rf, PT::kTriangleFan, v);
  CHECK(CoversRect(r, 2, 2, 14, 14));
  CHECK(r.AllOnce());
  CHECK_EQ(r.stats.primitives_culled, uint64_t(0));
  delete rf;
}

// Sub-pixel snapping: a vertex 1/32 px off a center rounds onto the 1/16 grid.
static void TestSnap() {
  RegisterFile* rf = Base();
  // Left edge at 0.5 + 1/64: truncate mode snaps it to 0.5, owning column 0.
  float e = 0.5f + 1.0f / 64.0f;
  std::vector<VertexOut> v = {V(e, 0.5f), V(4.5f, 0.5f), V(4.5f, 4.5f), V(e, 0.5f), V(4.5f, 4.5f), V(e, 4.5f)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  CHECK(CoversRect(r, 0, 0, 4, 4));
  delete rf;
}

// --- culling and face ---------------------------------------------------------
static void TestCullAndFace() {
  RegisterFile* rf = Base();
  // Clockwise on screen (y down): (0,0) (8,0) (0,8).
  std::vector<VertexOut> cw = {V(0, 0), V(8, 0), V(0, 8)};
  std::vector<VertexOut> ccw = {V(0, 0), V(0, 8), V(8, 0)};
  // face = 0: counter-clockwise is front.
  Result a = Draw(*rf, PT::kTriangleList, ccw);
  Result b = Draw(*rf, PT::kTriangleList, cw);
  CHECK(a.Covered() > 0 && a.At(1, 1) && a.At(1, 1)->front);
  CHECK(b.Covered() > 0 && b.At(1, 1) && !b.At(1, 1)->front);
  // Back face param_gen X is negated.
  CHECK(b.At(3, 2) && b.At(3, 2)->param_gen[0] == -3.0f && b.At(3, 2)->param_gen[1] == 2.0f);
  CHECK(a.At(3, 2) && a.At(3, 2)->param_gen[0] == 3.0f && a.At(3, 2)->param_gen[1] == 2.0f);
  CHECK(a.At(3, 2) && AsU32(a.At(3, 2)->param_gen[2]) == 0 && a.At(3, 2)->param_gen[3] == 0.0f);
  // cull_back drops the clockwise one.
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.cull_back = 1;
  Set(*rf, sc);
  CHECK(Draw(*rf, PT::kTriangleList, ccw).Covered() > 0);
  CHECK(Draw(*rf, PT::kTriangleList, cw).Covered() == 0);
  // face = 1 flips which winding is front.
  sc.face = 1;
  Set(*rf, sc);
  CHECK(Draw(*rf, PT::kTriangleList, ccw).Covered() == 0);
  CHECK(Draw(*rf, PT::kTriangleList, cw).Covered() > 0);
  // Degenerate triangles never draw.
  sc.cull_back = 0;
  sc.face = 0;
  Set(*rf, sc);
  Result d = Draw(*rf, PT::kTriangleList, {V(0, 0), V(4, 4), V(8, 8)});
  CHECK(d.Covered() == 0);
  CHECK_EQ(d.stats.primitives_culled, uint64_t(1));
  delete rf;
}

// Strip triangles alternate vertex order, so a flat strip keeps one facing.
static void TestStripWinding() {
  RegisterFile* rf = Base();
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.cull_back = 1;
  Set(*rf, sc);
  // Strip of 4: (0,0) (0,8) (8,0) (8,8): both triangles counter-clockwise.
  std::vector<VertexOut> v = {V(0, 0), V(0, 8), V(8, 0), V(8, 8)};
  Result r = Draw(*rf, PT::kTriangleStrip, v);
  CHECK(CoversRect(r, 0, 0, 8, 8));
  CHECK(r.AllOnce());
  // Restart splits the strip: 3 + 3 vertices, second strip triangle alone.
  std::vector<VertexOut> v2 = {V(0, 0), V(0, 8), V(8, 0), V(16, 0), V(16, 8), V(24, 0)};
  std::vector<uint8_t> restart = {0, 0, 0, 1, 0, 0};
  Result r2 = Draw(*rf, PT::kTriangleStrip, v2, restart);
  CHECK(r2.At(1, 1) && r2.At(17, 1));
  CHECK(!r2.At(9, 1));  // the bridging triangle (8,0)-(0,8)-(16,0) is not formed
  // Without the restart it is.
  CHECK(Draw(*rf, PT::kTriangleStrip, v2).At(9, 1));
  delete rf;
}

// Quad list: two triangles (v0,v1,v2), (v0,v2,v3) covering the quad once.
static void TestQuadList() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v = {V(0, 0), V(8, 0), V(8, 8), V(0, 8)};
  Result r = Draw(*rf, PT::kQuadList, v);
  CHECK(CoversRect(r, 0, 0, 8, 8));
  CHECK(r.AllOnce());
  delete rf;
}

// Rectangle list: the fourth corner is v1 - v0 + v2.
static void TestRectList() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v = {V(2, 2), V(10, 2), V(2, 6)};
  for (int i = 0; i < 3; ++i) {
    v[i].interp[0][0] = v[i].position[0];
    v[i].interp[0][1] = v[i].position[1];
  }
  Result r = Draw(*rf, PT::kRectangleList, v);
  CHECK(CoversRect(r, 2, 2, 10, 6));
  CHECK(r.AllOnce());
  // Interpolators extrapolate onto the fourth corner: value = pixel center.
  const Pix* p = r.At(9, 5);
  CHECK(p && p->interp[0][0] == 9.5f && p->interp[0][1] == 5.5f);
  delete rf;
}

// Line list: a horizontal line 1 px wide through row centers.
static void TestLine() {
  RegisterFile* rf = Base();
  rf->values[rg::XE_GPU_REG_PA_SU_LINE_CNTL] = 8;  // half width 8/16 px => 1 px wide
  std::vector<VertexOut> v = {V(2, 4.5f), V(10, 4.5f)};
  Result r = Draw(*rf, PT::kLineList, v);
  CHECK(r.Covered() > 0);
  // Every covered pixel sits on row 4 or straddles it.
  bool row4 = true;
  for (auto& [k, p] : r.px) {
    if (k.second != 4) row4 = false;
    // Lines set the param_gen Z sign bit and never negate X.
    CHECK(AsU32(p.param_gen[2]) == 0x80000000u);
    CHECK(p.param_gen[0] >= 0);
  }
  CHECK(row4);
  CHECK(r.At(5, 4));
  delete rf;
}

// --- MSAA -----------------------------------------------------------------------
// Left edge exactly on pixel 10's center column at x = 10.5.
static void TestMsaaCoverage() {
  for (int en = 0; en < 2; ++en) {
    RegisterFile* rf = Base(xenos::MsaaSamples::k4X, en != 0);
    std::vector<VertexOut> v = {V(10.5f, 0), V(20, 0), V(20, 8), V(10.5f, 0), V(20, 8), V(10.5f, 8)};
    Result r = Draw(*rf, PT::kTriangleList, v);
    const Pix* p = r.At(10, 3);
    CHECK(p != nullptr);
    if (p) {
      if (en) {
        // Samples at x offsets -2, +2, -6, +6 sixteenths: 1 and 3 are right of the edge.
        CHECK_EQ(int(p->cov), 0b1010);
      } else {
        // MSAA off: the center decides for all samples, top-left owns the tie.
        CHECK_EQ(int(p->cov), 0xF);
      }
    }
    const Pix* q = r.At(11, 3);
    CHECK(q && q->cov == 0xF);
    CHECK(!r.At(9, 3));
    delete rf;
  }
  // 2x: offsets -4, +4.
  RegisterFile* rf = Base(xenos::MsaaSamples::k2X, true);
  std::vector<VertexOut> v = {V(10.5f, 0), V(20, 0), V(20, 8), V(10.5f, 0), V(20, 8), V(10.5f, 8)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  CHECK(r.At(10, 3) && r.At(10, 3)->cov == 0b10);
  delete rf;
}

// Centroid: a partially covered pixel interpolates at its first covered
// sample; center interpolators stay at the pixel center.
static void TestCentroid() {
  RegisterFile* rf = Base(xenos::MsaaSamples::k4X, true);
  reg::SQ_CONTEXT_MISC misc{};
  misc.sc_sample_cntl = xenos::SampleControl::kCentroidsAndCenters;
  Set(*rf, misc);
  reg::SQ_INTERPOLATOR_CNTL ic{};
  ic.sampling_pattern = 0xFFFE;  // r0 centroid, r1.. center
  Set(*rf, ic);
  CHECK_EQ(RasterCentroidMask(*rf), 1u);
  std::vector<VertexOut> v = {V(10.5f, 0), V(20, 0), V(20, 8), V(10.5f, 0), V(20, 8), V(10.5f, 8)};
  for (auto& x : v) {
    x.interp[0][0] = x.position[0];
    x.interp[1][0] = x.position[0];
  }
  Result r = Draw(*rf, PT::kTriangleList, v);
  const Pix* p = r.At(10, 3);
  CHECK(p && p->cov == 0b1010);
  if (p) {
    CHECK_EQ(p->interp[1][0], 10.5f);                   // center
    CHECK_EQ(p->interp[0][0], 10.5f + 2.0f / 16.0f);    // sample 1 (+2/16)
  }
  const Pix* q = r.At(12, 3);  // fully covered: centroid = center
  CHECK(q && q->interp[0][0] == 12.5f);
  delete rf;
}

// --- interpolation ----------------------------------------------------------------
static void TestAffineInterp() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v = {V(0, 0), V(32, 0), V(0, 32)};
  for (auto& x : v) {
    x.interp[0][0] = x.position[0];
    x.interp[0][1] = x.position[1];
    x.interp[0][2] = 7.0f;
    x.interp[0][3] = x.position[0] + x.position[1];
  }
  Result r = Draw(*rf, PT::kTriangleList, v);
  bool ok = true;
  for (auto& [k, p] : r.px) {
    float ex = k.first + 0.5f, ey = k.second + 0.5f;
    if (p.interp[0][0] != ex || p.interp[0][1] != ey || p.interp[0][2] != 7.0f ||
        p.interp[0][3] != ex + ey)
      ok = false;
  }
  CHECK(ok);
  CHECK(r.Covered() > 400);
  delete rf;
}

// Perspective: left edge W = 1, right edge W = 3, u = 0 -> 1 across.
static void BuildPersp(std::vector<VertexOut>& v) {
  auto P = [](float sx, float sy, float w, float u) {
    VertexOut o = V(sx * w, sy * w, 0.5f * w, w);
    o.interp[0][0] = u;
    return o;
  };
  v = {P(0, 0, 1, 0), P(16, 0, 3, 1), P(16, 8, 3, 1), P(0, 0, 1, 0), P(16, 8, 3, 1), P(0, 8, 1, 0)};
}

static void TestPerspInterp() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v;
  BuildPersp(v);
  Result r = Draw(*rf, PT::kTriangleList, v);
  int worst = 0;
  for (auto& [k, p] : r.px) {
    double l = (k.first + 0.5) / 16.0;
    double u = (l / 3.0) / ((1.0 - l) + l / 3.0);
    worst = std::max(worst, UlpDiff(p.interp[0][0], float(u)));
  }
  CHECK(r.Covered() == 16 * 8);
  CHECK(worst <= 1);
  if (worst > 1) std::printf("  persp worst ulp %d\n", worst);
  // Depth stays linear in screen space: z/w = 0.5 everywhere.
  const Pix* p = r.At(8, 4);
  CHECK(p && std::fabs(p->z[0] - 0.5) < 1e-12);

  // persp_corr_dis: linear in screen space.
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.persp_corr_dis = 1;
  Set(*rf, sc);
  Result r2 = Draw(*rf, PT::kTriangleList, v);
  worst = 0;
  for (auto& [k, p2] : r2.px) {
    double l = (k.first + 0.5) / 16.0;
    worst = std::max(worst, UlpDiff(p2.interp[0][0], float(l)));
  }
  CHECK(worst <= 1);
  delete rf;
}

// Flat shading takes the provoking vertex: first, or last with provoking_vtx_last.
static void TestFlat() {
  RegisterFile* rf = Base();
  reg::SQ_INTERPOLATOR_CNTL ic{};
  ic.param_shade = 0b10;  // r1 flat
  Set(*rf, ic);
  std::vector<VertexOut> v = {V(0, 0), V(16, 0), V(0, 16)};
  for (int i = 0; i < 3; ++i) {
    v[i].interp[0][0] = float(i + 1);
    v[i].interp[1][0] = float(i + 1);
  }
  Result r = Draw(*rf, PT::kTriangleList, v);
  const Pix* p = r.At(3, 3);
  CHECK(p && p->interp[1][0] == 1.0f && p->interp[0][0] != 1.0f);
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.provoking_vtx_last = 1;
  Set(*rf, sc);
  r = Draw(*rf, PT::kTriangleList, v);
  p = r.At(3, 3);
  CHECK(p && p->interp[1][0] == 3.0f);
  delete rf;
}

// --- points -----------------------------------------------------------------------
static void TestPoint() {
  RegisterFile* rf = Base();
  reg::PA_SU_POINT_SIZE ps{};
  // 12.4 half-size: 32 -> diameter 32 * 2 / 16 = 4 px.
  ps.width = 32;
  ps.height = 32;
  Set(*rf, ps);
  reg::PA_SU_POINT_MINMAX mm{};
  mm.min_size = 0;
  mm.max_size = 0xFFFF;
  Set(*rf, mm);
  std::vector<VertexOut> v = {V(10, 10)};
  Result r = Draw(*rf, PT::kPointList, v);
  CHECK(CoversRect(r, 8, 8, 12, 12));
  const Pix* p = r.At(8, 9);
  CHECK(p && p->param_gen[0] == 8.0f && p->param_gen[1] == -9.0f);
  CHECK(p && p->param_gen[2] == 0.125f && p->param_gen[3] == 0.375f);
  const Pix* q = r.At(11, 11);
  CHECK(q && q->param_gen[2] == 0.875f && q->param_gen[3] == 0.875f);
  // Per-vertex size (oPts.x) overrides the register size, clamped by MINMAX.
  v[0].point_written = true;
  v[0].point_size_edge_kill[0] = 2.0f;
  r = Draw(*rf, PT::kPointList, v);
  CHECK(CoversRect(r, 9, 9, 11, 11));
  mm.min_size = 48;  // min diameter 6 px
  Set(*rf, mm);
  r = Draw(*rf, PT::kPointList, v);
  CHECK(CoversRect(r, 7, 7, 13, 13));
  delete rf;
}

// --- scissor, clipping, kill, depth -------------------------------------------------
static void TestScissor() {
  RegisterFile* rf = Base();
  reg::PA_SC_WINDOW_SCISSOR_TL tl{};
  tl.tl_x = 3;
  tl.tl_y = 2;
  tl.window_offset_disable = 1;
  Set(*rf, tl);
  reg::PA_SC_WINDOW_SCISSOR_BR br{};
  br.br_x = 7;
  br.br_y = 5;
  Set(*rf, br);
  std::vector<VertexOut> v = {V(0, 0), V(16, 0), V(16, 16), V(0, 0), V(16, 16), V(0, 16)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  CHECK(CoversRect(r, 3, 2, 7, 5));
  // Window offset moves the window scissor when window_offset_disable = 0.
  tl.window_offset_disable = 0;
  Set(*rf, tl);
  reg::PA_SC_WINDOW_OFFSET wo{};
  wo.window_x_offset = 4;
  wo.window_y_offset = 1;
  Set(*rf, wo);
  r = Draw(*rf, PT::kTriangleList, v);
  CHECK(CoversRect(r, 7, 3, 11, 6));
  // ... and vtx_window_offset_enable moves the vertices with it.
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.vtx_window_offset_enable = 1;
  Set(*rf, sc);
  std::vector<VertexOut> small = {V(0, 0), V(2, 0), V(2, 2), V(0, 0), V(2, 2), V(0, 2)};
  tl.tl_x = 0;
  tl.tl_y = 0;
  Set(*rf, tl);
  br.br_x = 64;
  br.br_y = 64;
  Set(*rf, br);
  r = Draw(*rf, PT::kTriangleList, small);
  CHECK(CoversRect(r, 4, 1, 6, 3));
  // Clockwise on screen with face = 0 is a back face, so param_gen X is negated.
  CHECK(r.At(4, 1) && !r.At(4, 1)->front && r.At(4, 1)->param_gen[0] == -4.0f &&
        r.At(4, 1)->param_gen[1] == 1.0f);
  delete rf;
}

static void TestHalfPixel() {
  RegisterFile* rf = Base();
  auto vtx = rf->Get<reg::PA_SU_VTX_CNTL>();
  vtx.pix_center = xenos::PixelCenter::kD3DZero;  // +0.5 to positions
  Set(*rf, vtx);
  // Square with corners on integers lands on pixel centers after +0.5, so the
  // top-left rule makes it cover [0,4) shifted by one only on the owned sides.
  std::vector<VertexOut> v = {V(0, 0), V(4, 0), V(4, 4), V(0, 0), V(4, 4), V(0, 4)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  CHECK(CoversRect(r, 0, 0, 4, 4));
  delete rf;
}

static void TestClip() {
  RegisterFile* rf = Base();
  reg::PA_CL_CLIP_CNTL clip{};
  clip.dx_clip_space_def = 1;
  Set(*rf, clip);
  // Huge guard band so only the near and far planes clip.
  SetF(*rf, rg::XE_GPU_REG_PA_CL_GB_HORZ_CLIP_ADJ, 1e6f);
  SetF(*rf, rg::XE_GPU_REG_PA_CL_GB_VERT_CLIP_ADJ, 1e6f);
  // Entirely in front of the near plane (z < 0): nothing.
  Result r = Draw(*rf, PT::kTriangleList, {V(0, 0, -0.5f), V(16, 0, -0.5f), V(0, 16, -0.5f)});
  CHECK(r.Covered() == 0);
  // Crossing z = 0 along x: z = x/8 - 1, so x >= 8 survives; attributes are
  // still the unclipped plane.
  std::vector<VertexOut> v = {V(0, 0, -1), V(16, 0, 1), V(16, 16, 1), V(0, 0, -1), V(16, 16, 1), V(0, 16, -1)};
  for (auto& x : v) x.interp[0][0] = x.position[0];
  r = Draw(*rf, PT::kTriangleList, v);
  CHECK(r.Covered() > 0);
  CHECK(!r.At(6, 4));
  CHECK(r.At(9, 4) && r.At(9, 4)->interp[0][0] == 9.5f);
  CHECK(r.stats.primitives_clipped > 0);
  delete rf;
}

static void TestVertexKill() {
  RegisterFile* rf = Base();
  std::vector<VertexOut> v = {V(0, 0), V(16, 0), V(0, 16)};
  v[1].point_written = true;
  v[1].point_size_edge_kill[2] = 1.0f;
  // One killed vertex: drawn unless vtx_kill_or.
  CHECK(Draw(*rf, PT::kTriangleList, v).Covered() > 0);
  auto clip = rf->Get<reg::PA_CL_CLIP_CNTL>();
  clip.vtx_kill_or = 1;
  Set(*rf, clip);
  CHECK(Draw(*rf, PT::kTriangleList, v).Covered() == 0);
  delete rf;
}

static void TestDepthAndOffset() {
  RegisterFile* rf = Base(xenos::MsaaSamples::k4X, true);
  // z = x / 64 (W = 1, z scale 1): per-sample depth on the exact plane.
  std::vector<VertexOut> v = {V(0, 0, 0), V(64, 0, 1), V(64, 64, 1), V(0, 0, 0), V(64, 64, 1), V(0, 64, 0)};
  Result r = Draw(*rf, PT::kTriangleList, v);
  const Pix* p = r.At(30, 20);
  CHECK(p != nullptr);
  const int sx[4] = {-2, 2, -6, 6};
  if (p) {
    for (int s = 0; s < 4; ++s) {
      double ex = (30.5 + sx[s] / 16.0) / 64.0;
      CHECK(std::fabs(p->z[s] - ex) < 1e-12);
    }
  }
  // Front polygon offset: scale in 1/16 subpixel units times max slope, plus offset.
  auto sc = rf->Get<reg::PA_SU_SC_MODE_CNTL>();
  sc.poly_offset_front_enable = 1;
  Set(*rf, sc);
  SetF(*rf, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE, 16.0f);
  SetF(*rf, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET, 0.25f);
  // These are clockwise on screen: back faces with face = 0, so no offset yet.
  Result b = Draw(*rf, PT::kTriangleList, v);
  const Pix* pb = b.At(30, 20);
  CHECK(pb && !pb->front && std::fabs(pb->z[0] - (30.5 - 2 / 16.0) / 64.0) < 1e-12);
  sc.face = 1;
  Set(*rf, sc);
  Result f = Draw(*rf, PT::kTriangleList, v);
  const Pix* pf = f.At(30, 20);
  CHECK(pf && pf->front);
  if (pf) {
    double ex = (30.5 - 2 / 16.0) / 64.0 + 16.0 / 16.0 * (1.0 / 64.0) + 0.25;
    CHECK(std::fabs(pf->z[0] - ex) < 1e-12);
  }
  // Clamp to [0, 1].
  SetF(*rf, rg::XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET, 2.0f);
  f = Draw(*rf, PT::kTriangleList, v);
  CHECK(f.At(30, 20) && f.At(30, 20)->z[0] == 1.0);
  delete rf;
}

int main() {
  TestSamplePositions();
  TestFillRuleSquare();
  TestSharedSamples();
  TestFillRuleFan();
  TestSnap();
  TestCullAndFace();
  TestStripWinding();
  TestQuadList();
  TestRectList();
  TestLine();
  TestMsaaCoverage();
  TestCentroid();
  TestAffineInterp();
  TestPerspInterp();
  TestFlat();
  TestPoint();
  TestScissor();
  TestHalfPixel();
  TestClip();
  TestVertexKill();
  TestDepthAndOffset();
  std::printf("raster: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
