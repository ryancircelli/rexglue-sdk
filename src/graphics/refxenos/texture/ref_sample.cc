// refxenos texture sampling. Slow and obvious: every texel a filter touches
// is decoded from guest memory on its own. Rule ids are listed in
// ref_rules.cc with their sources.

#include "ref_sample.h"

#include <algorithm>
#include <cmath>

namespace refxenos::tex {

namespace {

// Clamp modes (xenos.h ClampMode).
enum : uint32_t {
  kRepeat = 0,
  kMirroredRepeat = 1,
  kClampToEdge = 2,
  kMirrorClampToEdge = 3,
  kClampToHalfway = 4,
  kMirrorClampToHalfway = 5,
  kClampToBorder = 6,
  kMirrorClampToBorder = 7,
};
// Filters (xenos.h TextureFilter).
enum : uint32_t { kPoint = 0, kLinear = 1, kBaseMap = 2, kUseFetch = 3 };

bool UsesBorder(uint32_t clamp) {
  return clamp == kClampToBorder || clamp == kMirrorClampToBorder;
}

// Texel index after the address mode; -1 = border.
int32_t Address(int64_t i, uint32_t size, uint32_t clamp, Diag* diag) {
  int64_t n = size;
  switch (clamp) {
    case kRepeat:
      return int32_t(((i % n) + n) % n);
    case kMirroredRepeat: {
      int64_t m = ((i % (2 * n)) + 2 * n) % (2 * n);
      return int32_t(m < n ? m : 2 * n - 1 - m);
    }
    case kMirrorClampToEdge:
    case kMirrorClampToHalfway:
    case kMirrorClampToBorder:
      // rule sample.address: mirror once, then clamp to the edge (Xenia maps
      // all three mirror-clamp modes to mirror-clamp-to-edge).
      diag->Use("sample.address_mirror_clamp");
      if (i < 0) i = -i - 1;
      return int32_t(std::min<int64_t>(i, n - 1));
    case kClampToBorder:
      if (i < 0 || i >= n) return -1;
      return int32_t(i);
    case kClampToHalfway:
      diag->Use("sample.address_halfway");
      [[fallthrough]];
    default:
      return int32_t(std::clamp<int64_t>(i, 0, n - 1));
  }
}

void BorderColor(uint32_t border, float out[4]) {
  // Order of the returned components is pre-swizzle RGBA.
  switch (border) {
    case 1:  // k_ABGR_White
      out[0] = out[1] = out[2] = out[3] = 1.0f;
      break;
    case 2:  // k_ACBYCR_Black: (Cr, Y, Cb) = (0.5, 0, 0.5), alpha 0
      out[0] = 0.5f; out[1] = 0.0f; out[2] = 0.5f; out[3] = 0.0f;
      break;
    case 3:  // k_ACBCRY_Black: (Y, Cr, Cb) = (0, 0.5, 0.5), alpha 0
      out[0] = 0.0f; out[1] = 0.5f; out[2] = 0.5f; out[3] = 0.0f;
      break;
    default:  // k_ABGR_Black, alpha 0
      out[0] = out[1] = out[2] = out[3] = 0.0f;
      break;
  }
}

// rule sample.subtexel: texel-space coordinates are fixed point with 8
// fractional bits (truncated).
double Subtexel(double x) { return std::floor(x * 256.0) / 256.0; }

struct Axis {
  int64_t i0 = 0;     // first texel
  float frac = 0.0f;  // weight of the second texel (linear)
  bool linear = false;
  bool outside = false;  // coordinate outside [0, 1) (for getBCF)
};

Axis MakeAxis(double u, uint32_t size, bool linear) {
  Axis a;
  a.linear = linear;
  a.outside = !(u >= 0.0 && u < 1.0);
  double x = Subtexel(u * double(size));
  if (linear) {
    double xm = x - 0.5;
    double f = std::floor(xm);
    a.i0 = int64_t(f);
    a.frac = float(xm - f);
  } else {
    a.i0 = int64_t(std::floor(x));
  }
  return a;
}

struct Ctx {
  const Fetch* f;
  const GuestMemory* mem;
  Diag* diag;
  uint32_t clamp[3];
  bool ok = true;
};

// Pre-swizzle texel or the border colour.
void Texel(Ctx& c, uint32_t level, uint32_t layer, int32_t x, int32_t y,
           int32_t z, float out[4]) {
  if (x < 0 || y < 0 || z < 0) {
    c.diag->Use("sample.border");
    BorderColor(c.f->border_color, out);
    return;
  }
  if (!DecodeTexelPreSwizzle(*c.f, *c.mem, level, layer, uint32_t(x),
                             uint32_t(y), uint32_t(z), out, c.diag)) {
    c.ok = false;
    out[0] = out[1] = out[2] = out[3] = 0.0f;
  }
}

// Bi- or tri-linear (3D) filtered value of one level.
void FilterLevel(Ctx& c, uint32_t level, uint32_t layer, const double uvw[3],
                 bool linear_xy, bool linear_z, bool is3d, float out[4],
                 float weights_out[3]) {
  const Fetch& f = *c.f;
  uint32_t w = std::max(f.width >> level, 1u);
  uint32_t h = std::max(f.height >> level, 1u);
  uint32_t d = is3d ? std::max(f.depth >> level, 1u) : 1u;
  Axis ax = MakeAxis(uvw[0], w, linear_xy);
  Axis ay = MakeAxis(uvw[1], h, linear_xy);
  Axis az = is3d ? MakeAxis(uvw[2], d, linear_z) : Axis();
  if (weights_out) {
    weights_out[0] = ax.frac;
    weights_out[1] = ay.frac;
    weights_out[2] = az.frac;
  }
  for (int i = 0; i < 4; ++i) out[i] = 0.0f;
  int nz = (is3d && az.linear) ? 2 : 1;
  int ny = ay.linear ? 2 : 1;
  int nx = ax.linear ? 2 : 1;
  for (int kz = 0; kz < nz; ++kz) {
    float wz = nz == 1 ? 1.0f : (kz ? az.frac : 1.0f - az.frac);
    int32_t z = is3d ? Address(az.i0 + kz, d, c.clamp[2], c.diag) : 0;
    for (int ky = 0; ky < ny; ++ky) {
      float wy = ny == 1 ? 1.0f : (ky ? ay.frac : 1.0f - ay.frac);
      int32_t y = Address(ay.i0 + ky, h, c.clamp[1], c.diag);
      for (int kx = 0; kx < nx; ++kx) {
        float wx = nx == 1 ? 1.0f : (kx ? ax.frac : 1.0f - ax.frac);
        int32_t x = Address(ax.i0 + kx, w, c.clamp[0], c.diag);
        float t[4];
        Texel(c, level, layer, x, y, z, t);
        float wt = wx * wy * wz;
        for (int i = 0; i < 4; ++i) out[i] += wt * t[i];
      }
    }
  }
}

}  // namespace

std::array<float, 4> Sample(const SampleRequest& r, const GuestMemory& mem,
                            Diag* diag) {
  Diag local;
  if (!diag) diag = &local;
  diag->Use("fetch.fields");
  Fetch f = Fetch::Parse(r.fetch);
  Ctx c;
  c.f = &f;
  c.mem = &mem;
  c.diag = diag;
  // Clamp modes per axis (texture_util::GetClampModesForDimension): unused
  // axes and cube maps clamp to the edge.
  c.clamp[0] = c.clamp[1] = c.clamp[2] = kClampToEdge;
  switch (f.dimension) {
    case kDim3D: c.clamp[2] = f.clamp[2]; [[fallthrough]];
    case kDim2D: c.clamp[1] = f.clamp[1]; [[fallthrough]];
    case kDim1D: c.clamp[0] = f.clamp[0]; break;
    default: diag->Use("sample.cube_face_clamp"); break;
  }

  bool is3d = f.dimension == kDim3D;
  bool stacked = f.dimension == kDim2D && f.stacked && r.dimension == kDim3D;
  uint32_t size[3] = {f.width, f.height, is3d ? f.depth : f.layers()};

  // Coordinates: normalized; offsets in texels at the base level.
  double uvw[3];
  for (int i = 0; i < 3; ++i) {
    double v = r.coord[i];
    if (r.unnormalized && i < 2) {
      diag->Use("sample.unnormalized");
      v /= double(size[i]);
    }
    if (r.offset[i] != 0.0f && i < 2 + int(is3d)) {
      diag->Use("sample.offset");
      v += double(r.offset[i]) / double(size[i]);
    }
    uvw[i] = v;
  }
  if (r.dimension == kDim1D) uvw[1] = 0.5 / double(size[1]);

  // LOD (rule sample.lod): computed from the gradients scaled to texels of
  // the base level, plus fetch LOD bias / 32, plus instruction bias.
  double bias = double(f.lod_bias) / 32.0 + double(r.instr_lod_bias);
  diag->Use("sample.lod_bias");
  if (r.use_register_lod) bias += double(r.register_lod);
  double lod = bias;
  if (r.use_computed_lod) {
    diag->Use("sample.lod");
    double lh = 0.0, lv = 0.0;
    uint32_t dims = r.dimension == kDim1D ? 1 : (r.dimension == kDim3D && is3d) ? 3 : 2;
    double gh = std::ldexp(1.0, f.grad_exp_adjust_h);
    double gv = std::ldexp(1.0, f.grad_exp_adjust_v);
    if (f.grad_exp_adjust_h || f.grad_exp_adjust_v) diag->Use("sample.grad_exp_adjust");
    for (uint32_t i = 0; i < dims; ++i) {
      double sh = r.has_grads ? r.grad_h[i] : 0.0;
      double sv = r.has_grads ? r.grad_v[i] : 0.0;
      if (r.unnormalized) {
        sh /= double(size[i]);
        sv /= double(size[i]);
      }
      sh *= gh * size[i];
      sv *= gv * size[i];
      lh += sh * sh;
      lv += sv * sv;
    }
    double rho2 = std::max(lh, lv);
    double computed = rho2 > 0.0 ? 0.5 * std::log2(rho2) : -INFINITY;
    lod = computed + bias;
  }
  if (r.op == SampleOp::kGetCompTexLOD) {
    return {float(lod), 0.0f, 0.0f, 0.0f};
  }

  uint32_t mag = r.mag_filter.value_or(kUseFetch);
  uint32_t min = r.min_filter.value_or(kUseFetch);
  uint32_t mip = r.mip_filter.value_or(kUseFetch);
  uint32_t vmag = r.vol_mag_filter.value_or(kUseFetch);
  uint32_t vmin = r.vol_min_filter.value_or(kUseFetch);
  uint32_t aniso = r.aniso_filter.value_or(7);
  if (mag == kUseFetch) mag = f.mag_filter;
  if (min == kUseFetch) min = f.min_filter;
  if (mip == kUseFetch) mip = f.mip_filter;
  if (vmag == kUseFetch) vmag = f.vol_mag_filter;
  if (vmin == kUseFetch) vmin = f.vol_min_filter;
  if (aniso == 7) aniso = f.aniso_filter;
  if (aniso > 1) diag->Use("sample.aniso");  // filtered as trilinear

  // Mip range (rule sample.lod_clamp): [min_mip_level, max_mip_level] of the
  // fetch, base map = the minimum level.
  LevelRange lr = GetLevelRange(f, diag);
  diag->Use("sample.lod_clamp");
  bool magnify = !(lod > 0.0);  // rule sample.mag_min
  bool linear_xy = (magnify ? mag : min) == kLinear;
  bool linear_z = (magnify ? vmag : vmin) == kLinear;
  double lod_c = std::clamp(std::isfinite(lod) ? lod : (lod > 0 ? 1e9 : -1e9),
                            double(lr.min_level), double(lr.max_level));
  uint32_t level0 = lr.min_level, level1 = lr.min_level;
  float mip_frac = 0.0f;
  if (mip == kBaseMap || magnify) {
    level0 = level1 = lr.min_level;
  } else if (mip == kLinear) {
    diag->Use("sample.mip_linear");
    level0 = uint32_t(std::floor(lod_c));
    level1 = std::min(level0 + 1, lr.max_level);
    mip_frac = float(lod_c - std::floor(lod_c));
    // rule sample.subtexel also quantizes the mip fraction to 8 bits.
    mip_frac = float(std::floor(double(mip_frac) * 256.0) / 256.0);
  } else {
    diag->Use("sample.mip_point");
    level0 = level1 = uint32_t(std::floor(lod_c + 0.5));
    level0 = level1 = std::min(level0, lr.max_level);
  }

  if (r.op == SampleOp::kGetBCF) {
    std::array<float, 4> bcf = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
      bool outside = !(uvw[i] >= 0.0 && uvw[i] < 1.0);
      bcf[i] = (UsesBorder(c.clamp[i]) && outside) ? 1.0f : 0.0f;
    }
    return bcf;
  }

  // Layers: stacked 2D picks one or two layers from w (rule sample.stacked),
  // cube maps use the resolved face.
  float rgba[4] = {0, 0, 0, 0};
  float weights[3] = {0, 0, 0};
  auto filter_layer = [&](uint32_t level, float out[4]) {
    if (f.dimension == kDimCube) {
      diag->Use("sample.cube");
      FilterLevel(c, level, r.cube_face % 6, uvw, linear_xy, false, false, out,
                  weights);
      return;
    }
    if (stacked) {
      diag->Use("sample.stacked");
      uint32_t layers = f.layers();
      double z = Subtexel(uvw[2] * layers);
      float a[4], b[4];
      if (linear_z) {
        double zm = z - 0.5;
        int64_t z0 = int64_t(std::floor(zm));
        float fz = float(zm - std::floor(zm));
        weights[2] = fz;
        uint32_t l0 = uint32_t(std::clamp<int64_t>(z0, 0, layers - 1));
        uint32_t l1 = uint32_t(std::clamp<int64_t>(z0 + 1, 0, layers - 1));
        FilterLevel(c, level, l0, uvw, linear_xy, false, false, a, weights);
        FilterLevel(c, level, l1, uvw, linear_xy, false, false, b, nullptr);
        weights[2] = fz;
        for (int i = 0; i < 4; ++i) out[i] = a[i] * (1.0f - fz) + b[i] * fz;
      } else {
        uint32_t l = uint32_t(std::clamp<int64_t>(int64_t(std::floor(z)), 0,
                                                  layers - 1));
        FilterLevel(c, level, l, uvw, linear_xy, false, false, out, weights);
      }
      return;
    }
    FilterLevel(c, level, 0, uvw, linear_xy, linear_z, is3d, out, weights);
  };
  filter_layer(level0, rgba);
  if (level1 != level0 && mip_frac > 0.0f) {
    float b[4];
    float w_keep[3] = {weights[0], weights[1], weights[2]};
    filter_layer(level1, b);
    std::copy(w_keep, w_keep + 3, weights);
    for (int i = 0; i < 4; ++i)
      rgba[i] = rgba[i] * (1.0f - mip_frac) + b[i] * mip_frac;
  }

  if (r.op == SampleOp::kGetWeights) {
    diag->Use("sample.weights");
    return {weights[0], weights[1], weights[2], mip_frac};
  }
  if (!c.ok) return {NAN, NAN, NAN, NAN};

  // rule sample.order: filter, then swizzle, then bias / gamma per swizzled
  // component, then exp_adjust.
  diag->Use("sample.order");
  diag->Use("swizzle");
  std::array<float, 4> result;
  float scale = std::ldexp(1.0f, f.exp_adjust);
  if (f.exp_adjust) diag->Use("exp_adjust");
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t s = f.swizzle[i];
    float v;
    if (s < 4) {
      v = rgba[s];
      if (f.sign[s] == kBias) {
        diag->Use("sign.bias");
        v = v * 2.0f + -1.0f;
      } else if (f.sign[s] == kGamma) {
        diag->Use("sign.gamma");
        v = PWLGammaToLinear(v);
      }
    } else {
      v = (s & 1) ? 1.0f : 0.0f;
    }
    if ((s & 0b101) != 4 && f.exp_adjust) v *= scale;
    result[i] = v;
  }
  return result;
}

}  // namespace refxenos::tex
