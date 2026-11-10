// rexgpu-texconf: texture decode conformance of the Metal GPU plugin.
//
// Each case is a guest texture fetch constant plus the guest bytes of its base
// and mip levels. The case goes through the plugin's real texture load path
// (MetalTextureCache materialization: untiling, endian swap, format
// conversion, BC decompression or compute load shaders, packed mip tails,
// stacked / 3D / cube layouts), then the exact view a shader would bind
// (GetBindlessSRVIndexForBinding, with the host swizzle and the signed view)
// is sampled at every texel centre of every level and slice by a small
// compute kernel. The shader-side part of a tfetch (bias / gamma by sign,
// exp_adjust) is applied with the same arithmetic as the MSL translator
// (metal_msl_translator.cc:3187-3209, PWLGammaToLinear :2350-2370).
//
// The result is compared with the refxenos CPU reference decoder
// (src/graphics/refxenos/texture), which is written independently from the
// XDK documentation and the SDK's Xenia-derived rules, each rule cited.
//
// Inputs:
//   --synthetic             built-in cases (every format, tiled / linear,
//                           mips + packed tails, odd sizes, pitch padding,
//                           endian modes, signs, swizzles, exp_adjust)
//   --texpack FILE...       TXPK files (zone extraction / trace capture)
//   --golden-adapter CASE STORE OUTDIR
//   --batch LIST STORE OUTROOT
//                           golden store adapter (golden-store.md v1.1):
//                           runs texel_decode cases through Metal and writes
//                           OUTDIR/mip<N> (f32 [slices][h][w][4]) or
//                           OUTDIR/ERROR
//   --golden-write STORE    reference side: writes the selected cases as
//                           texel_decode cases with referee "refxenos"
//                           (CPU only, no Metal device needed)
//   --referee-version V     version recorded with --golden-write
// Options:
//   --only SUBSTR           cases whose name contains SUBSTR
//   --diff-dir DIR          diff PNGs for failing cases (default
//                           /tmp/bo2/texconf/diff)
//   --dump N                print up to N differing texels (reference vs
//                           Metal) with their exact bits, to classify the
//                           mismatch
//   --list                  print case names only
//   --bc-bitexact           BC decode regression: only the BC formats whose
//                           interpolation the XDK rounds to nearest, and they
//                           have to match the reference bit for bit (the
//                           signed compressed views are skipped).
//   -v                      per-level detail
//   --metal_force_bc_decompress=true and other cvars are passed through.
//
// Exit code: 0 all cases pass, 1 a case failed, 2 harness error.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rex/cvar.h"
#include "rex/graphics/metal/metal_command_processor.h"
#include "rex/graphics/metal/metal_graphics_system.h"
#include "rex/graphics/metal/metal_shared_memory.h"
#include "rex/graphics/metal/metal_test_hooks.h"
#include "rex/graphics/metal/metal_texture_cache.h"
#include "rex/graphics/pipeline/texture/util.h"
#include "rex/graphics/register_file.h"
#include "rex/graphics/xenos.h"
#include "rex/logging.h"
#include "metal_flags.h"
#include "rex/runtime.h"
#include "rex/system/xmemory.h"
#include "rex/thread.h"
#include "third_party/metal-cpp/Metal/Metal.hpp"

#include <nlohmann/json.hpp>

#include "golden_store.h"
#include "ref_rules.h"
#include "ref_texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace rex::graphics::metal {

// Friend of MetalCommandProcessor and MetalTextureCache (REX_METAL_TEST_FRIEND).
class TexConfAccess {
 public:
  using Plan = MetalTextureCache::TextureMaterializationPlan;

  TexConfAccess(MetalCommandProcessor* cp, RegisterFile* regs)
      : cp_(cp), regs_(regs) {}

  MetalTextureCache& tc() { return *cp_->texture_cache(); }
  MTL::Device* device() { return cp_->GetMetalDevice(); }
  MTL::CommandQueue* queue() { return cp_->GetMetalCommandQueue(); }

  // One draw's texture materialization for fetch slot 0 (as a draw does it).
  bool Materialize(const xenos::xe_gpu_texture_fetch_t& fetch) {
    std::memcpy(&regs_->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], &fetch,
                sizeof(fetch));
    if (!cp_->EnsureCommandBuffer()) return false;
    Plan plan;
    if (!tc().PrepareTextureMaterialization(*regs_, 1, plan)) return false;
    if (!plan.source_ranges.empty() &&
        !cp_->RequestSharedMemoryRanges(
            MetalCommandProcessor::SharedMemoryRequestReason::kUnknown,
            plan.source_ranges.data(), uint32_t(plan.source_ranges.size()))) {
      return false;
    }
    tc().BeginTextureUploadBatch();
    bool ok = tc().ExecuteTextureMaterialization(plan);
    ok &= tc().EndTextureUploadBatch();
    tc().RequestTexturesWithoutLoading(1, nullptr);
    return ok;
  }

  // The view a shader would bind for fetch constant 0.
  MTL::Texture* View(xenos::FetchOpDimension dimension, bool is_signed) {
    MTL::Texture* t = nullptr;
    tc().GetBindlessSRVIndexForBinding(0, dimension, is_signed, &t);
    return t;
  }

  void Flush() { cp_->FlushCommandBufferAndWait(5000000000ULL, "texconf"); }
  void Clear() { tc().ClearCache(); }

 private:
  MetalCommandProcessor* cp_;
  RegisterFile* regs_;
};

}  // namespace rex::graphics::metal

namespace {

namespace xenos = rex::graphics::xenos;
namespace ref = refxenos::tex;
using rex::graphics::metal::TexConfAccess;

rex::memory::Memory* g_mem = nullptr;
TexConfAccess* g_acc = nullptr;
bool g_verbose = false;
std::string g_diff_dir = "/tmp/bo2/texconf/diff";
bool g_adapter_mode = false;  // no diff PNGs, no per-case printing
// --bc-bitexact: regression mode for the BC decoder. Only the BC formats
// whose interpolation the XDK rounds to nearest are run, and they have to
// match the reference bit for bit. Signed compressed views are skipped: the
// reference models a snorm view of the decoded data, which Metal does not
// provide for compressed formats (rule "compressed.signed").
bool g_bc_bitexact = false;
// --dump N: print up to N differing texels (reference vs Metal) with their
// exact bits; used to classify a mismatch (rounding, sign, layout).
int g_dump_limit = 0;
int g_dump_count = 0;
// Runs a function on the command processor thread and waits for it.
std::function<void(std::function<void()>)> g_run_in_cp;

// ---------------------------------------------------------------------------
// Readback kernels: sample every texel centre of one level / slice with
// nearest filtering and an explicit level.

const char* kReadbackMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct P { uint w, h, d, level, slice, face, pad0, pad1; };
constexpr sampler smp(coord::normalized, filter::nearest,
                      mip_filter::nearest, address::clamp_to_edge);
kernel void rb_2d(texture2d<float> t [[texture(0)]],
                  device float4* o [[buffer(0)]], constant P& p [[buffer(1)]],
                  uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.w || g.y >= p.h) return;
  float2 uv = (float2(g.xy) + 0.5f) /
              float2(t.get_width(p.level), t.get_height(p.level));
  o[g.y * p.w + g.x] = t.sample(smp, uv, level(float(p.level)));
}
kernel void rb_2da(texture2d_array<float> t [[texture(0)]],
                   device float4* o [[buffer(0)]], constant P& p [[buffer(1)]],
                   uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.w || g.y >= p.h) return;
  float2 uv = (float2(g.xy) + 0.5f) /
              float2(t.get_width(p.level), t.get_height(p.level));
  o[g.y * p.w + g.x] = t.sample(smp, uv, p.slice, level(float(p.level)));
}
kernel void rb_3d(texture3d<float> t [[texture(0)]],
                  device float4* o [[buffer(0)]], constant P& p [[buffer(1)]],
                  uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.w || g.y >= p.h) return;
  float3 uvw = (float3(g.x, g.y, p.slice) + 0.5f) /
               float3(t.get_width(p.level), t.get_height(p.level),
                      t.get_depth(p.level));
  o[g.y * p.w + g.x] = t.sample(smp, uvw, level(float(p.level)));
}
float3 cube_dir(uint face, float2 uv) {
  float sc = uv.x * 2.0f - 1.0f, tc = uv.y * 2.0f - 1.0f;
  switch (face) {
    case 0: return float3(1.0f, -tc, -sc);
    case 1: return float3(-1.0f, -tc, sc);
    case 2: return float3(sc, 1.0f, tc);
    case 3: return float3(sc, -1.0f, -tc);
    case 4: return float3(sc, -tc, 1.0f);
    default: return float3(-sc, -tc, -1.0f);
  }
}
kernel void rb_cube(texturecube<float> t [[texture(0)]],
                    device float4* o [[buffer(0)]], constant P& p [[buffer(1)]],
                    uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.w || g.y >= p.h) return;
  float2 uv = (float2(g.xy) + 0.5f) /
              float2(t.get_width(p.level), t.get_height(p.level));
  o[g.y * p.w + g.x] = t.sample(smp, cube_dir(p.slice, uv),
                                level(float(p.level)));
}
kernel void rb_cubea(texturecube_array<float> t [[texture(0)]],
                     device float4* o [[buffer(0)]],
                     constant P& p [[buffer(1)]],
                     uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.w || g.y >= p.h) return;
  float2 uv = (float2(g.xy) + 0.5f) /
              float2(t.get_width(p.level), t.get_height(p.level));
  o[g.y * p.w + g.x] = t.sample(smp, cube_dir(p.slice % 6u, uv),
                                p.slice / 6u, level(float(p.level)));
}
)MSL";

struct Pipelines {
  MTL::ComputePipelineState* rb_2d = nullptr;
  MTL::ComputePipelineState* rb_2da = nullptr;
  MTL::ComputePipelineState* rb_3d = nullptr;
  MTL::ComputePipelineState* rb_cube = nullptr;
  MTL::ComputePipelineState* rb_cubea = nullptr;
} g_pipes;

bool CreatePipelines(MTL::Device* dev) {
  NS::Error* err = nullptr;
  MTL::Library* lib = dev->newLibrary(
      NS::String::string(kReadbackMsl, NS::UTF8StringEncoding), nullptr, &err);
  if (!lib) {
    std::printf("HARNESS: readback MSL compile failed: %s\n",
                err ? err->localizedDescription()->utf8String() : "?");
    return false;
  }
  auto make = [&](const char* name) -> MTL::ComputePipelineState* {
    MTL::Function* fn =
        lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* e = nullptr;
    MTL::ComputePipelineState* ps = dev->newComputePipelineState(fn, &e);
    fn->release();
    return ps;
  };
  g_pipes.rb_2d = make("rb_2d");
  g_pipes.rb_2da = make("rb_2da");
  g_pipes.rb_3d = make("rb_3d");
  g_pipes.rb_cube = make("rb_cube");
  g_pipes.rb_cubea = make("rb_cubea");
  lib->release();
  return g_pipes.rb_2d && g_pipes.rb_2da && g_pipes.rb_3d && g_pipes.rb_cube &&
         g_pipes.rb_cubea;
}

// Samples level `level`, slice / face / depth slice `slice` of `t` into
// w * h float4s.
bool SampleLevel(MTL::Texture* t, uint32_t level, uint32_t slice, uint32_t w,
                 uint32_t h, std::vector<float>& out, std::string& err) {
  MTL::ComputePipelineState* ps = nullptr;
  switch (t->textureType()) {
    case MTL::TextureType2D: ps = g_pipes.rb_2d; break;
    case MTL::TextureType2DArray: ps = g_pipes.rb_2da; break;
    case MTL::TextureType3D: ps = g_pipes.rb_3d; break;
    case MTL::TextureTypeCube: ps = g_pipes.rb_cube; break;
    case MTL::TextureTypeCubeArray: ps = g_pipes.rb_cubea; break;
    default:
      err = "unhandled MTLTextureType " + std::to_string(int(t->textureType()));
      return false;
  }
  if (level >= t->mipmapLevelCount()) {
    err = "view has " + std::to_string(t->mipmapLevelCount()) +
          " levels, level " + std::to_string(level) + " requested";
    return false;
  }
  MTL::Device* dev = g_acc->device();
  size_t bytes = size_t(w) * h * 16;
  MTL::Buffer* buf = dev->newBuffer(std::max<size_t>(bytes, 16),
                                    MTL::ResourceStorageModeShared);
  uint32_t params[8] = {w, h, 1, level, slice, 0, 0, 0};
  MTL::CommandBuffer* cb = g_acc->queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ps);
  enc->setTexture(t, 0);
  enc->setBuffer(buf, 0, 0);
  enc->setBytes(params, sizeof(params), 1);
  enc->dispatchThreads(MTL::Size(w, h, 1), MTL::Size(8, 8, 1));
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  bool ok = cb->status() == MTL::CommandBufferStatusCompleted;
  if (ok) {
    const float* p = static_cast<const float*>(buf->contents());
    out.assign(p, p + size_t(w) * h * 4);
  } else {
    err = "readback command buffer failed";
  }
  buf->release();
  return ok;
}

// ---------------------------------------------------------------------------
// Shader-side tfetch post-processing, as metal_msl_translator.cc emits it.

float PWLGammaToLinearMsl(float value) {
  // metal_msl_translator.cc:2350-2370 (from DxbcShaderTranslator).
  float s, o;
  if (value >= 96.0f / 255.0f) {
    bool t = value >= 192.0f / 255.0f;
    s = t ? 8.0f / 1024.0f : 4.0f / 1024.0f;
    o = t ? -1024.0f : -256.0f;
  } else {
    bool t = value >= 64.0f / 255.0f;
    s = t ? 2.0f / 1024.0f : 1.0f / 1024.0f;
    o = t ? -64.0f : 0.0f;
  }
  value = std::min(std::max(value, 0.0f), 1.0f);
  value = value * (255.0f * 1024.0f);
  value = std::fma(value, s, o);
  s = std::trunc(value * s);
  value = value + s;
  value = value * (1.0f / 1023.0f);
  return value;
}

// ---------------------------------------------------------------------------
// Guest memory.

struct Region {
  uint32_t va = 0, phys = 0, size = 0;
};

Region Alloc(uint32_t size) {
  size = std::max<uint32_t>((size + 0xFFFF) & ~0xFFFFu, 0x10000);
  Region r;
  r.va = g_mem->SystemHeapAlloc(size, 0x10000,
                                rex::memory::kSystemHeapPhysical);
  r.phys = g_mem->GetPhysicalAddress(r.va);
  r.size = size;
  return r;
}

void Free(const Region& r) {
  if (r.va) g_mem->SystemHeapFree(r.va);
}

// Host write plus the physical memory callbacks, so the texture cache and
// shared memory see the new contents.
void Write(const Region& r, uint32_t offset, const std::vector<uint8_t>& d) {
  if (d.empty()) return;
  std::memcpy(g_mem->TranslatePhysical(r.phys + offset), d.data(), d.size());
  g_mem->TriggerPhysicalMemoryCallbacks(
      rex::thread::global_critical_region::AcquireDirect(), r.va + offset,
      uint32_t(d.size()), true, true);
}

// ---------------------------------------------------------------------------
// Cases.

struct Case {
  std::string name;
  std::string source;
  uint32_t fetch[6] = {};   // as registers (host order dwords)
  std::vector<uint8_t> base;  // guest bytes at the base address
  std::vector<uint8_t> mips;  // guest bytes at the mip address
};

uint32_t FetchSet(uint32_t v, uint32_t lo, uint32_t n, uint32_t x) {
  uint32_t m = ((n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1)) << lo;
  return (v & ~m) | ((x << lo) & m);
}

// Per mip level accounting of one case.
struct LevelStat {
  uint64_t comps = 0;      // components compared
  uint64_t exact = 0;      // bit-exact
  uint64_t near = 0;       // different but within the tolerance
  uint64_t over = 0;       // different and over the tolerance
  uint64_t one_lsb = 0;    // different by exactly one 8-bit step (1/255)
  double max_err = 0.0;
};

struct Result {
  enum Kind { kPass, kFail, kError, kUnsupported } kind = kPass;
  std::string msg;
  uint64_t comps = 0, exact = 0;
  double max_err = 0.0;
  std::string worst;  // where the max error is
  std::vector<std::string> rules;
  std::map<uint32_t, LevelStat> levels;  // per mip level
};

// Tolerance per component: the BC interpolation must match the XDK
// formulas exactly (the Metal compute BC decoder rounds to nearest, like the
// refxenos reference); everything else must match within float rounding of
// the unorm conversions.
bool IsBCFormat(uint32_t base_format) {
  switch (base_format) {
    case ref::k_DXT1: case ref::k_DXT2_3: case ref::k_DXT4_5: case ref::k_DXN:
    case ref::k_DXT5A: case ref::k_DXT3A: case ref::k_CTX1:
    case ref::k_DXT3A_AS_1_1_1_1:
      return true;
    default:
      return false;
  }
}

// The BC formats whose colour / alpha interpolation the XDK rounds to
// nearest. These are the ones the Metal compute BC decoder has to reproduce
// bit for bit (see the --bc-bitexact regression mode).
bool IsRoundedBCFormat(uint32_t base_format) {
  switch (base_format) {
    case ref::k_DXT1: case ref::k_DXT2_3: case ref::k_DXT4_5: case ref::k_DXN:
    case ref::k_DXT5A:
      return true;
    default:
      return false;
  }
}

double Tolerance(uint32_t base_format) {
  if (g_bc_bitexact && IsRoundedBCFormat(base_format)) {
    // Regression mode: the BC decode has to be bit exact.
    return 0.0;
  }
  return IsBCFormat(base_format) ? 1.0 / 255.0 + 1e-6 : 2e-6;
}

uint32_t BaseFormatOf(uint32_t f) {
  switch (f) {
    case ref::k_8_8_8_8_AS_16_16_16_16: return ref::k_8_8_8_8;
    case ref::k_DXT1_AS_16_16_16_16: return ref::k_DXT1;
    case ref::k_DXT2_3_AS_16_16_16_16: return ref::k_DXT2_3;
    case ref::k_DXT4_5_AS_16_16_16_16: return ref::k_DXT4_5;
    case ref::k_2_10_10_10_AS_16_16_16_16: return ref::k_2_10_10_10;
    case ref::k_10_11_11_AS_16_16_16_16: return ref::k_10_11_11;
    case ref::k_11_11_10_AS_16_16_16_16: return ref::k_11_11_10;
    default: return f;
  }
}

void WriteDiffPng(const std::string& name, uint32_t level, uint32_t slice,
                  uint32_t w, uint32_t h, const std::vector<float>& refv,
                  const std::vector<float>& got) {
  std::error_code ec;
  std::filesystem::create_directories(g_diff_dir, ec);
  // Columns: reference | Metal | |diff| * 16. Rows: RGB, alpha.
  uint32_t W = w * 3, H = h * 2;
  std::vector<uint8_t> img(size_t(W) * H * 3, 0);
  auto put = [&](uint32_t x, uint32_t y, float r, float g, float b) {
    auto c = [](float v) {
      if (!(v == v)) return uint8_t(255);
      return uint8_t(std::lround(std::min(std::max(v, 0.0f), 1.0f) * 255.0f));
    };
    uint8_t* p = &img[(size_t(y) * W + x) * 3];
    p[0] = c(r); p[1] = c(g); p[2] = c(b);
  };
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const float* a = &refv[(size_t(y) * w + x) * 4];
      const float* b = &got[(size_t(y) * w + x) * 4];
      put(x, y, a[0], a[1], a[2]);
      put(w + x, y, b[0], b[1], b[2]);
      put(2 * w + x, y, std::fabs(a[0] - b[0]) * 16, std::fabs(a[1] - b[1]) * 16,
          std::fabs(a[2] - b[2]) * 16);
      put(x, h + y, a[3], a[3], a[3]);
      put(w + x, h + y, b[3], b[3], b[3]);
      float d = std::fabs(a[3] - b[3]) * 16;
      put(2 * w + x, h + y, d, d, d);
    }
  }
  std::string safe;
  for (char ch : name) safe += (std::isalnum((unsigned char)ch) || ch == '-' || ch == '_' || ch == '.') ? ch : '_';
  std::string path = g_diff_dir + "/" + safe + "_L" + std::to_string(level) +
                     "_s" + std::to_string(slice) + ".png";
  // Upscale small images so they can be looked at.
  uint32_t scale = std::max<uint32_t>(1, 256 / std::max(W, H));
  if (scale > 1) {
    std::vector<uint8_t> big(size_t(W) * scale * H * scale * 3);
    for (uint32_t y = 0; y < H * scale; ++y)
      for (uint32_t x = 0; x < W * scale; ++x)
        std::memcpy(&big[(size_t(y) * W * scale + x) * 3],
                    &img[(size_t(y / scale) * W + x / scale) * 3], 3);
    stbi_write_png(path.c_str(), int(W * scale), int(H * scale), 3, big.data(),
                   int(W * scale * 3));
  } else {
    stbi_write_png(path.c_str(), int(W), int(H), 3, img.data(), int(W * 3));
  }
}

// Outputs for the golden adapter: per level, f32x4 per texel for every slice.
using LevelOutputs = std::map<uint32_t, std::vector<float>>;

// Runs one case through Metal and the reference. If `metal_out` is set, the
// Metal values for every level are stored there (golden adapter).
Result RunCase(const Case& c, LevelOutputs* metal_out = nullptr,
               const std::vector<uint32_t>* only_levels = nullptr) {
  Result res;
  ref::Fetch rf = ref::Fetch::Parse(c.fetch);
  const ref::FormatInfo& fi = ref::GetFormatInfo(rf.format);
  uint32_t base_format = BaseFormatOf(rf.format);

  // Place the bytes in fresh guest memory and point the fetch at them.
  uint32_t base_size = (uint32_t(c.base.size()) + 0xFFFF) & ~0xFFFFu;
  Region region = Alloc(base_size + uint32_t(c.mips.size()) + 0x10000);
  Write(region, 0, c.base);
  Write(region, base_size, c.mips);
  uint32_t dw[6];
  std::memcpy(dw, c.fetch, sizeof(dw));
  bool has_base = !c.base.empty();
  bool has_mips = !c.mips.empty();
  dw[1] = FetchSet(dw[1], 12, 20, has_base ? (region.phys >> 12) : 0);
  dw[5] = FetchSet(dw[5], 12, 20, has_mips ? ((region.phys + base_size) >> 12) : 0);
  ref::Fetch f = ref::Fetch::Parse(dw);
  ref::RangeMemory mem;
  if (has_base) mem.Add(region.phys, c.base);
  if (has_mips) mem.Add(region.phys + base_size, c.mips);

  xenos::xe_gpu_texture_fetch_t xf;
  std::memcpy(&xf, dw, sizeof(xf));

  // Which views the shader would sample, per result component.
  bool need_unsigned = false, need_signed = false;
  uint32_t out_sign[4];
  bool constant[4];
  uint32_t all_signs = (dw[0] >> 2) & 0xFF;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t s = f.swizzle[i];
    constant[i] = (s & 4) != 0;
    out_sign[i] = constant[i] ? 0 : f.sign[s];
  }
  bool all_constant = constant[0] && constant[1] && constant[2] && constant[3];
  for (uint32_t i = 0; i < 4; ++i) {
    if (constant[i]) continue;
    if (out_sign[i] == ref::kSigned) need_signed = true; else need_unsigned = true;
  }
  if (all_constant) {
    // SwizzleSigns (util.cpp:537-547): all-signed fetch -> signed view.
    if (all_signs == 0x55) need_signed = true; else need_unsigned = true;
  }
  if (g_bc_bitexact && need_signed) {
    // Metal binds no separate signed view of a compressed texture, while the
    // reference models one (rule "compressed.signed"). Out of scope for the
    // BC decode regression.
    Free(region);
    res.kind = Result::kUnsupported;
    res.msg = "signed compressed view";
    return res;
  }

  xenos::FetchOpDimension dim;
  switch (f.dimension) {
    case ref::kDim1D: dim = xenos::FetchOpDimension::k1D; break;
    case ref::kDim3D: dim = xenos::FetchOpDimension::k3DOrStacked; break;
    case ref::kDimCube: dim = xenos::FetchOpDimension::kCube; break;
    default:
      dim = f.stacked ? xenos::FetchOpDimension::k3DOrStacked
                      : xenos::FetchOpDimension::k2D;
      break;
  }
  if (f.dimension == ref::kDim2D) dim = xenos::FetchOpDimension::k2D;

  bool materialized = false;
  MTL::Texture* view_u = nullptr;
  MTL::Texture* view_s = nullptr;
  {
    // Materialize and bind on the command processor thread.
    g_run_in_cp([&]() {
      materialized = g_acc->Materialize(xf);
      if (materialized) {
        if (need_unsigned) view_u = g_acc->View(dim, false);
        if (need_signed) view_s = g_acc->View(dim, true);
        if (view_u) view_u->retain();
        if (view_s) view_s->retain();
      }
      g_acc->Flush();
    });
  }

  ref::LevelRange lr = ref::GetLevelRange(f, nullptr);
  uint32_t slices = f.dimension == ref::kDim3D ? 0 : f.layers();

  auto finish = [&]() {
    if (view_u) view_u->release();
    if (view_s) view_s->release();
    Free(region);
  };

  if (!materialized) {
    res.kind = fi.supported ? Result::kError : Result::kUnsupported;
    res.msg = "materialization failed";
    finish();
    return res;
  }
  if ((need_unsigned && !view_u) || (need_signed && !view_s)) {
    res.kind = fi.supported ? Result::kError : Result::kUnsupported;
    res.msg = "no view bound (null texture)";
    finish();
    return res;
  }

  ref::Diag diag;
  double tol = Tolerance(base_format);
  for (uint32_t level = lr.min_level; level <= lr.max_level; ++level) {
    if (only_levels && std::find(only_levels->begin(), only_levels->end(),
                                 level) == only_levels->end()) {
      continue;
    }
    uint32_t w = std::max(f.width >> level, 1u);
    uint32_t h = std::max(f.height >> level, 1u);
    uint32_t nslices =
        f.dimension == ref::kDim3D ? std::max(f.depth >> level, 1u) : slices;
    std::vector<float>* lvl_out = nullptr;
    if (metal_out) {
      lvl_out = &(*metal_out)[level];
      lvl_out->clear();
    }
    for (uint32_t slice = 0; slice < nslices; ++slice) {
      // Reference.
      ref::DecodedTexels rt;
      uint32_t ref_layer = f.dimension == ref::kDim3D ? 0 : slice;
      if (!ref::DecodeLevel(f, mem, level, ref_layer, &rt, &diag)) {
        res.kind = fi.supported ? Result::kError : Result::kUnsupported;
        res.msg = "reference: " + diag.error;
        finish();
        return res;
      }
      size_t ref_off = f.dimension == ref::kDim3D ? size_t(slice) * w * h * 4 : 0;
      // Metal.
      std::vector<float> mu, ms;
      std::string err;
      if (view_u && !SampleLevel(view_u, level, slice, w, h, mu, err)) {
        res.kind = Result::kError;
        res.msg = "readback: " + err;
        finish();
        return res;
      }
      if (view_s && !SampleLevel(view_s, level, slice, w, h, ms, err)) {
        res.kind = Result::kError;
        res.msg = "readback (signed): " + err;
        finish();
        return res;
      }
      std::vector<float> got(size_t(w) * h * 4);
      for (size_t t = 0; t < size_t(w) * h; ++t) {
        for (uint32_t i = 0; i < 4; ++i) {
          bool from_signed = constant[i] ? (all_constant && need_signed)
                                         : out_sign[i] == ref::kSigned;
          float v = from_signed ? ms[t * 4 + i] : mu[t * 4 + i];
          bool nonzero = (f.swizzle[i] & 0b101) != 4;
          if (!constant[i]) {
            if (out_sign[i] == ref::kBias) v = std::fma(v, 2.0f, -1.0f);
            else if (out_sign[i] == ref::kGamma) v = PWLGammaToLinearMsl(v);
          }
          if (nonzero) {
            // ldexp(1, exp_adjust) built from the exponent bits like the MSL.
            float e = std::ldexp(1.0f, f.exp_adjust);
            v = v * e;
          }
          got[t * 4 + i] = v;
        }
      }
      if (lvl_out) lvl_out->insert(lvl_out->end(), got.begin(), got.end());
      // Compare.
      bool level_fail = false;
      LevelStat& ls = res.levels[level];
      for (size_t t = 0; t < size_t(w) * h; ++t) {
        for (uint32_t i = 0; i < 4; ++i) {
          float a = rt.rgba[ref_off + t * 4 + i];
          float b = got[t * 4 + i];
          ++res.comps;
          ++ls.comps;
          bool both_nan = (a != a) && (b != b);
          if (both_nan || std::memcmp(&a, &b, 4) == 0) {
            ++res.exact;
            ++ls.exact;
            continue;
          }
          double e = std::fabs(double(a) - double(b));
          if (a != a || b != b) e = INFINITY;
          if (e > res.max_err) {
            res.max_err = e;
            char buf[240];
            std::snprintf(buf, sizeof(buf),
                          "L%u s%u (%zu,%zu).%c ref=%.7g metal=%.7g", level,
                          slice, t % w, t / w, "xyzw"[i], a, b);
            res.worst = buf;
          }
          ls.max_err = std::max(ls.max_err, e);
          if (e > tol) {
            ++ls.over;
            level_fail = true;
          } else {
            ++ls.near;
          }
          // One 8-bit step: the classic BC interpolation rounding difference.
          if (std::fabs(e - 1.0 / 255.0) < 1e-5) ++ls.one_lsb;
          if (g_dump_limit > 0 && g_dump_count < g_dump_limit) {
            ++g_dump_count;
            uint32_t ab, bb;
            std::memcpy(&ab, &a, 4);
            std::memcpy(&bb, &b, 4);
            std::printf(
                "DUMP %s L%u s%u texel(%u,%u).%c ref=%.9g(0x%08X) "
                "metal=%.9g(0x%08X) diff=%.9g\n",
                c.name.c_str(), level, slice, uint32_t(t % w), uint32_t(t / w),
                "xyzw"[i], a, ab, b, bb, e);
          }
        }
      }
      if (level_fail) {
        res.kind = Result::kFail;
        std::vector<float> rv(rt.rgba.begin() + ref_off,
                              rt.rgba.begin() + ref_off + size_t(w) * h * 4);
        if (!g_adapter_mode) WriteDiffPng(c.name, level, slice, w, h, rv, got);
      }
      if (g_verbose) {
        std::printf("    L%u s%u %ux%u\n", level, slice, w, h);
      }
    }
  }
  res.rules = diag.rules_used;
  finish();
  return res;
}

// ---------------------------------------------------------------------------
// Synthetic cases.

struct SynthFormat {
  uint32_t format;
  uint32_t endian;  // natural guest endian for the format
};

const SynthFormat kSynthFormats[] = {
    {ref::k_8, 0},
    {ref::k_1_5_5_5, 1},
    {ref::k_5_6_5, 1},
    {ref::k_6_5_5, 1},
    {ref::k_8_8_8_8, 2},
    {ref::k_2_10_10_10, 2},
    {ref::k_8_A, 0},
    {ref::k_8_B, 0},
    {ref::k_8_8, 1},
    {ref::k_8_8_8_8_A, 2},
    {ref::k_4_4_4_4, 1},
    {ref::k_10_11_11, 2},
    {ref::k_11_11_10, 2},
    {ref::k_DXT1, 1},
    {ref::k_DXT2_3, 1},
    {ref::k_DXT4_5, 1},
    {ref::k_24_8, 2},
    {ref::k_24_8_FLOAT, 2},
    {ref::k_16, 1},
    {ref::k_16_16, 2},
    {ref::k_16_16_16_16, 1},
    {ref::k_16_EXPAND, 1},
    {ref::k_16_16_EXPAND, 2},
    {ref::k_16_16_16_16_EXPAND, 1},
    {ref::k_16_FLOAT, 1},
    {ref::k_16_16_FLOAT, 2},
    {ref::k_16_16_16_16_FLOAT, 1},
    {ref::k_32_FLOAT, 2},
    {ref::k_32_32_FLOAT, 2},
    {ref::k_32_32_32_32_FLOAT, 2},
    {ref::k_DXN, 1},
    {ref::k_8_8_8_8_AS_16_16_16_16, 2},
    {ref::k_DXT1_AS_16_16_16_16, 1},
    {ref::k_DXT2_3_AS_16_16_16_16, 1},
    {ref::k_DXT4_5_AS_16_16_16_16, 1},
    {ref::k_2_10_10_10_AS_16_16_16_16, 2},
    {ref::k_10_11_11_AS_16_16_16_16, 2},
    {ref::k_11_11_10_AS_16_16_16_16, 2},
    {ref::k_DXT3A, 1},
    {ref::k_DXT5A, 1},
    {ref::k_CTX1, 1},
    {ref::k_DXT3A_AS_1_1_1_1, 1},
};

struct SynthShape {
  const char* name;
  uint32_t dimension;  // ref::Dimension
  uint32_t w, h, d;
  bool stacked;
  bool tiled;
  uint32_t max_level;  // 0 = no mips
  bool packed;
  uint32_t extra_pitch_div32;  // pitch padding beyond the width
};

const SynthShape kSynthShapes[] = {
    {"256x256_tiled_mips_packed", ref::kDim2D, 256, 256, 1, false, true, 8, true, 0},
    {"128x64_tiled_mips_packed", ref::kDim2D, 128, 64, 1, false, true, 7, true, 0},
    {"64x64_tiled_mips_unpacked", ref::kDim2D, 64, 64, 1, false, true, 6, false, 0},
    {"16x16_tiled_packed_only", ref::kDim2D, 16, 16, 1, false, true, 4, true, 0},
    {"64x64_linear_mips_packed", ref::kDim2D, 64, 64, 1, false, false, 6, true, 0},
    {"37x23_linear_odd", ref::kDim2D, 37, 23, 1, false, false, 0, false, 0},
    {"100x60_tiled_odd_mips", ref::kDim2D, 100, 60, 1, false, true, 6, true, 0},
    {"40x24_linear_pitchpad", ref::kDim2D, 40, 24, 1, false, false, 0, false, 2},
    {"48x16_tiled_pitchpad", ref::kDim2D, 48, 16, 1, false, true, 0, false, 1},
    {"512x32_tiled_wide_mips", ref::kDim2D, 512, 32, 1, false, true, 9, true, 0},
    {"32x128_tiled_tall_mips", ref::kDim2D, 32, 128, 1, false, true, 7, true, 0},
    {"64x64x3_stacked_mips", ref::kDim2D, 64, 64, 3, true, true, 6, true, 0},
    {"32x32x8_3d_tiled_mips", ref::kDim3D, 32, 32, 8, false, true, 5, true, 0},
    {"32x32_cube_mips", ref::kDimCube, 32, 32, 1, false, true, 5, true, 0},
};

uint32_t Log2Floor(uint32_t v) {
  uint32_t r = 0;
  while (v > 1) { v >>= 1; ++r; }
  return r;
}

// Exact upper bounds for the synthetic guest-memory ranges, computed from the
// independent reference layout. This keeps the random corpus small while
// covering non-linear tiled address spans and packed mip tails.
std::pair<uint32_t, uint32_t> SynthAddressRangeSizes(const uint32_t d[6]) {
  uint32_t dw[6];
  std::memcpy(dw, d, sizeof(dw));
  constexpr uint32_t kBasePage = 0x1000;
  constexpr uint32_t kMipPage = 0x4000;
  dw[1] = FetchSet(dw[1], 12, 20, kBasePage);
  dw[5] = FetchSet(dw[5], 12, 20, kMipPage);
  ref::Fetch f = ref::Fetch::Parse(dw);
  ref::LevelRange levels = ref::GetLevelRange(f, nullptr);
  uint32_t ends[2] = {f.base_address(), f.mip_address()};
  const uint32_t base_addresses[2] = {f.base_address(), f.mip_address()};
  const ref::FormatInfo& fi = ref::GetFormatInfo(f.format);
  for (uint32_t level = levels.min_level; level <= levels.max_level; ++level) {
    bool is_base = level == 0;
    ref::LevelLocation loc;
    ref::Diag diag;
    if (!ref::GetLevelLocation(f, level, &loc, &diag)) return {0, 0};
    uint32_t range = is_base ? 0 : 1;
    uint32_t layers = f.dimension == ref::kDim3D ? 1 : f.layers();
    uint32_t blocks_x = (loc.width + fi.block_w - 1) / fi.block_w;
    uint32_t blocks_y = (loc.height + fi.block_h - 1) / fi.block_h;
    for (uint32_t layer = 0; layer < layers; ++layer) {
      for (uint32_t z = 0; z < (f.dimension == ref::kDim3D ? loc.depth : 1u);
           ++z) {
        for (uint32_t by = 0; by < blocks_y; ++by) {
          for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            uint32_t x = bx + loc.tail_x_blocks;
            uint32_t y = by + loc.tail_y_blocks;
            uint32_t zz = z + loc.tail_z;
            uint32_t address = loc.address + layer * loc.layer_stride_bytes;
            if (f.tiled) {
              address += f.dimension == ref::kDim3D
                  ? ref::TiledOffset3D(x, y, zz, loc.row_pitch_blocks,
                                       loc.slice_rows_blocks,
                                       fi.bytes_per_block())
                  : ref::TiledOffset2D(x, y, loc.row_pitch_blocks,
                                       fi.bytes_per_block());
            } else {
              address += (zz * loc.slice_rows_blocks + y) *
                             loc.row_pitch_bytes +
                         x * fi.bytes_per_block();
            }
            ends[range] = std::max(ends[range], address + fi.bytes_per_block());
          }
        }
      }
    }
  }
  auto padded = [&](uint32_t i) {
    uint32_t size = ends[i] - base_addresses[i];
    return (size + 4095u) & ~4095u;
  };
  return {padded(0), padded(1)};
}

bool MakeSynthCase(const SynthFormat& sf, const SynthShape& sh, uint32_t endian,
                   uint32_t sign_mode, uint32_t swizzle, int32_t exp_adjust,
                   uint32_t seed, const char* tag, Case* out) {
  const ref::FormatInfo& fi = ref::GetFormatInfo(sf.format);
  uint32_t d[6] = {};
  uint32_t pitch_div32 = (sh.w + 31) / 32 + sh.extra_pitch_div32;
  uint32_t signs = 0;
  switch (sign_mode) {
    case 0: signs = 0x00; break;  // unsigned
    case 1: signs = 0x55; break;  // signed
    case 2: signs = 0xAA; break;  // biased
    case 3: signs = 0xFF; break;  // gamma
    case 4: signs = 0x00 | (1 << 2) | (2 << 4) | (3 << 6); break;  // mixed
  }
  d[0] = 2u | (signs << 2) | (pitch_div32 << 22) | (sh.tiled ? 1u << 31 : 0);
  d[1] = sf.format | (endian << 6) | ((sh.stacked ? 1u : 0) << 10);
  switch (sh.dimension) {
    case ref::kDim3D:
      d[2] = (sh.w - 1) | ((sh.h - 1) << 11) | ((sh.d - 1) << 22);
      break;
    default:
      d[2] = (sh.w - 1) | ((sh.h - 1) << 13) | ((sh.d - 1) << 26);
      break;
  }
  d[3] = (swizzle << 1) | ((uint32_t(exp_adjust) & 63) << 13);
  uint32_t max_level = std::min(sh.max_level, Log2Floor(std::max(sh.w, std::max(sh.h, sh.dimension == ref::kDim3D ? sh.d : 1u))));
  d[4] = (0u << 2) | (max_level << 6);
  d[5] = (sh.dimension << 9) | ((sh.packed ? 1u : 0) << 11);
  auto [base_size, mip_size] = SynthAddressRangeSizes(d);
  if (!base_size || (max_level && !mip_size)) return false;
  out->base.resize(base_size);
  out->mips.resize(max_level ? mip_size : 0);
  std::mt19937 rng(seed);
  auto fill = [&](std::vector<uint8_t>& v) {
    for (auto& b : v) b = uint8_t(rng());
    // Float formats: avoid NaN payload comparisons dominating; keep a few.
    if (sf.format == ref::k_32_FLOAT || sf.format == ref::k_32_32_FLOAT ||
        sf.format == ref::k_32_32_32_32_FLOAT) {
      for (size_t i = 0; i + 3 < v.size(); i += 4) {
        // Byte order in memory is big-endian 8in32: exponent byte first.
        if ((v[i] & 0x7F) == 0x7F && (rng() & 15)) v[i] &= 0xBF;
      }
    }
  };
  fill(out->base);
  fill(out->mips);
  // DXT1: bias half of the blocks towards c0 <= c1 (punch-through alpha) with
  // index 3 used; the other half random.
  (void)fi;
  std::memcpy(out->fetch, d, sizeof(d));
  char name[256];
  std::snprintf(name, sizeof(name), "synth/%s/%s/e%u%s", fi.name, sh.name,
                endian, tag);
  out->name = name;
  out->source = "synthetic";
  return true;
}

std::vector<Case> SynthCases() {
  std::vector<Case> cases;
  uint32_t seed = 1;
  const uint32_t kXYZW = 0 | (1 << 3) | (2 << 6) | (3 << 9);
  const uint32_t kWZYX = 3 | (2 << 3) | (1 << 6) | (0 << 9);
  const uint32_t kX01Y = 0 | (4 << 3) | (5 << 6) | (1 << 9);
  for (const SynthFormat& sf : kSynthFormats) {
    for (const SynthShape& sh : kSynthShapes) {
      Case c;
      if (MakeSynthCase(sf, sh, sf.endian, 0, kXYZW, 0, seed++, "", &c))
        cases.push_back(std::move(c));
    }
    // Endian modes, signs, swizzles, exp_adjust on one tiled shape.
    const SynthShape& sh = kSynthShapes[1];
    for (uint32_t e = 0; e < 4; ++e) {
      if (e == sf.endian) continue;
      Case c;
      if (MakeSynthCase(sf, sh, e, 0, kXYZW, 0, seed++, "", &c))
        cases.push_back(std::move(c));
    }
    static const char* kSignTags[] = {"", "/signed", "/biased", "/gamma",
                                      "/mixedsign"};
    for (uint32_t s = 1; s < 5; ++s) {
      Case c;
      if (MakeSynthCase(sf, sh, sf.endian, s, kXYZW, 0, seed++, kSignTags[s],
                        &c))
        cases.push_back(std::move(c));
    }
    {
      Case c;
      if (MakeSynthCase(sf, sh, sf.endian, 0, kWZYX, 0, seed++, "/swz_wzyx", &c))
        cases.push_back(std::move(c));
    }
    {
      Case c;
      if (MakeSynthCase(sf, sh, sf.endian, 0, kX01Y, 3, seed++, "/swz_x01y_exp+3", &c))
        cases.push_back(std::move(c));
    }
    {
      Case c;
      if (MakeSynthCase(sf, sh, sf.endian, 1, kXYZW, -2, seed++, "/signed_exp-2", &c))
        cases.push_back(std::move(c));
    }
  }
  return cases;
}

// ---------------------------------------------------------------------------
// TXPK reader (zone extraction / trace capture).

// Records whose name does not contain `only` are skipped without reading
// their bytes (zone texpacks are hundreds of MB).
bool ReadTexpack(const std::string& path, std::vector<Case>& out,
                 const std::string& only = std::string()) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  auto u32 = [&](uint32_t& v) {
    uint8_t b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) return false;
    v = b[0] | (b[1] << 8) | (b[2] << 16) | (uint32_t(b[3]) << 24);
    return true;
  };
  char magic[4];
  if (!in.read(magic, 4) || std::memcmp(magic, "TXPK", 4)) return false;
  uint32_t version, count;
  if (!u32(version) || version != 1 || !u32(count)) return false;
  std::string stem = std::filesystem::path(path).stem().string();
  for (uint32_t i = 0; i < count; ++i) {
    Case c;
    uint32_t n;
    if (!u32(n)) return false;
    std::string name(n, '\0');
    in.read(name.data(), n);
    if (!u32(n)) return false;
    std::string source(n, '\0');
    in.read(source.data(), n);
    for (int k = 0; k < 6; ++k)
      if (!u32(c.fetch[k])) return false;
    uint32_t bl, ml;
    if (!u32(bl) || !u32(ml)) return false;
    if (!only.empty() &&
        (std::filesystem::path(path).stem().string() + "/" + name)
                .find(only) == std::string::npos) {
      in.seekg(std::streamoff(bl) + ml, std::ios::cur);
      if (!in) return false;
      continue;
    }
    c.base.resize(bl);
    c.mips.resize(ml);
    if (bl) in.read(reinterpret_cast<char*>(c.base.data()), bl);
    if (ml) in.read(reinterpret_cast<char*>(c.mips.data()), ml);
    if (!in) return false;
    c.name = stem + "/" + name;
    c.source = source;
    out.push_back(std::move(c));
  }
  return true;
}

// ---------------------------------------------------------------------------
// Golden store (research/golden-store.md v1.1), layer texel_decode.

namespace gs = refxenos::golden;
using json = nlohmann::json;

// 6 big-endian dwords with base_address and mip_address zeroed.
std::vector<uint8_t> FetchToGuestBytes(const uint32_t fetch[6]) {
  uint32_t d[6];
  std::memcpy(d, fetch, sizeof(d));
  d[1] = FetchSet(d[1], 12, 20, 0);
  d[5] = FetchSet(d[5], 12, 20, 0);
  std::vector<uint8_t> b(24);
  for (int i = 0; i < 6; ++i) {
    b[i * 4 + 0] = uint8_t(d[i] >> 24);
    b[i * 4 + 1] = uint8_t(d[i] >> 16);
    b[i * 4 + 2] = uint8_t(d[i] >> 8);
    b[i * 4 + 3] = uint8_t(d[i]);
  }
  return b;
}

// Reference decode of every level and slice. Base at kRefBase, mips at the
// next 64 KiB boundary after the base bytes (the same relocation RunCase
// uses for Metal).
constexpr uint32_t kRefBase = 0x08000000;

struct RefLevels {
  std::map<uint32_t, std::vector<float>> levels;  // [slices][h][w][4]
  std::map<uint32_t, std::array<uint32_t, 3>> shapes;  // slices, h, w
  std::vector<std::string> rules;
};

bool RefDecodeAll(const Case& c, RefLevels* out, std::string* err) {
  uint32_t base_size = (uint32_t(c.base.size()) + 0xFFFF) & ~0xFFFFu;
  uint32_t dw[6];
  std::memcpy(dw, c.fetch, sizeof(dw));
  dw[1] = FetchSet(dw[1], 12, 20, c.base.empty() ? 0 : (kRefBase >> 12));
  dw[5] = FetchSet(dw[5], 12, 20,
                   c.mips.empty() ? 0 : ((kRefBase + base_size) >> 12));
  ref::Fetch f = ref::Fetch::Parse(dw);
  ref::RangeMemory mem;
  if (!c.base.empty()) mem.Add(kRefBase, c.base);
  if (!c.mips.empty()) mem.Add(kRefBase + base_size, c.mips);
  ref::Diag diag;
  ref::LevelRange lr = ref::GetLevelRange(f, &diag);
  for (uint32_t level = lr.min_level; level <= lr.max_level; ++level) {
    uint32_t w = std::max(f.width >> level, 1u);
    uint32_t h = std::max(f.height >> level, 1u);
    std::vector<float>& v = out->levels[level];
    uint32_t nslices;
    if (f.dimension == ref::kDim3D) {
      ref::DecodedTexels rt;
      if (!ref::DecodeLevel(f, mem, level, 0, &rt, &diag)) {
        *err = diag.error;
        return false;
      }
      nslices = rt.depth;
      v = std::move(rt.rgba);
    } else {
      nslices = f.layers();
      for (uint32_t s = 0; s < nslices; ++s) {
        ref::DecodedTexels rt;
        if (!ref::DecodeLevel(f, mem, level, s, &rt, &diag)) {
          *err = diag.error;
          return false;
        }
        v.insert(v.end(), rt.rgba.begin(), rt.rgba.end());
      }
    }
    out->shapes[level] = {nslices, h, w};
  }
  out->rules = diag.rules_used;
  return true;
}

int WriteGoldens(std::vector<Case>& cases, const std::string& only,
                 const std::string& store_dir, const std::string& version) {
  gs::Store store(store_dir);
  size_t nrules = 0;
  const ref::RuleInfo* rules = ref::GetRules(&nrules);
  for (size_t i = 0; i < nrules; ++i) {
    store.RegisterRule(rules[i].id, rules[i].version, rules[i].source,
                       rules[i].status == ref::RuleStatus::kAssumed);
  }
  int written = 0, failed = 0, unsupported = 0;
  for (const Case& c : cases) {
    if (!only.empty() && c.name.find(only) == std::string::npos) continue;
    ref::Fetch f = ref::Fetch::Parse(c.fetch);
    if (!ref::GetFormatInfo(f.format).supported) {
      ++unsupported;
      continue;
    }
    RefLevels rl;
    std::string err;
    if (!RefDecodeAll(c, &rl, &err)) {
      std::printf("REFERROR %s  %s\n", c.name.c_str(), err.c_str());
      ++failed;
      continue;
    }
    json inputs = json::object();
    std::vector<uint8_t> fb = FetchToGuestBytes(c.fetch);
    inputs["fetch"] = store.PutObject(fb.data(), fb.size());
    inputs["base"] = store.PutObject(c.base.data(), c.base.size());
    if (!c.mips.empty())
      inputs["mips"] = store.PutObject(c.mips.data(), c.mips.size());
    json mips = json::array(), slices = json::array();
    json outputs = json::object(), ref_outputs = json::object();
    uint32_t max_slices = 0;
    for (auto& [level, v] : rl.levels) {
      mips.push_back(level);
      auto& sh = rl.shapes[level];
      max_slices = std::max(max_slices, sh[0]);
      std::string name = "mip" + std::to_string(level);
      outputs[name] = {{"elem", "f32"},
                       {"shape", {sh[0], sh[1], sh[2], 4}},
                       {"compare", "bits_nan_any"}};
      ref_outputs[name] = store.PutObject(v.data(), v.size() * sizeof(float));
    }
    for (uint32_t s = 0; s < max_slices; ++s) slices.push_back(s);
    json core = {{"layer", "texel_decode"},
                 {"game", "bo2"},
                 {"source", c.source + ":" + c.name},
                 {"inputs", inputs},
                 {"params", {{"mip", mips}, {"slice", slices}}},
                 {"outputs", outputs}};
    json rule_versions = json::object();
    for (const std::string& r : rl.rules) {
      const ref::RuleInfo* ri = ref::FindRule(r.c_str());
      rule_versions[r] = ri ? ri->version : 1;
    }
    json entry = {{"version", version},
                  {"rules", rule_versions},
                  {"outputs", ref_outputs}};
    std::string id, werr;
    if (!store.WriteCaseFromReferee(core, "refxenos", entry, &id, &werr)) {
      std::printf("STOREERROR %s  %s\n", c.name.c_str(), werr.c_str());
      ++failed;
      continue;
    }
    if (g_verbose) std::printf("%s %s\n", id.c_str(), c.name.c_str());
    ++written;
  }
  std::printf("golden-write: %d cases written, %d failed, %d unsupported\n",
              written, failed, unsupported);
  return failed ? 1 : 0;
}

void WriteFile(const std::filesystem::path& p, const void* data, size_t size) {
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  o.write(static_cast<const char*>(data), std::streamsize(size));
}

void WriteError(const std::filesystem::path& outdir, const std::string& msg) {
  std::error_code ec;
  std::filesystem::create_directories(outdir, ec);
  std::string m = msg + "\n";
  WriteFile(outdir / "ERROR", m.data(), m.size());
}

// One texel_decode case through Metal. Returns false only on harness
// failure (the case's ERROR file is written in either case).
void AdapterOne(const std::filesystem::path& case_json, const gs::Store& store,
                const std::filesystem::path& outdir) {
  std::error_code ec;
  std::filesystem::create_directories(outdir, ec);
  json m;
  try {
    std::ifstream in(case_json);
    m = json::parse(in);
  } catch (const std::exception& e) {
    WriteError(outdir, std::string("bad case json: ") + e.what());
    return;
  }
  std::string layer = m.value("layer", "");
  if (layer != "texel_decode") {
    WriteError(outdir, "layer " + layer + " not supported by rexgpu-texconf");
    return;
  }
  Case c;
  c.name = m.value("id", case_json.stem().string());
  const json& inputs = m["inputs"];
  std::vector<uint8_t> fb;
  if (!inputs.contains("fetch") ||
      !store.GetObject(inputs["fetch"].get<std::string>(), &fb) ||
      fb.size() != 24) {
    WriteError(outdir, "missing or bad fetch object");
    return;
  }
  for (int i = 0; i < 6; ++i) {
    c.fetch[i] = (uint32_t(fb[i * 4]) << 24) | (uint32_t(fb[i * 4 + 1]) << 16) |
                 (uint32_t(fb[i * 4 + 2]) << 8) | fb[i * 4 + 3];
  }
  if (inputs.contains("base") &&
      !store.GetObject(inputs["base"].get<std::string>(), &c.base)) {
    WriteError(outdir, "missing base object");
    return;
  }
  if (inputs.contains("mips") &&
      !store.GetObject(inputs["mips"].get<std::string>(), &c.mips)) {
    WriteError(outdir, "missing mips object");
    return;
  }
  std::vector<uint32_t> levels, slices;
  const json params = m.value("params", json::object());
  if (params.contains("mip"))
    for (auto& v : params["mip"]) levels.push_back(v.get<uint32_t>());
  if (params.contains("slice"))
    for (auto& v : params["slice"]) slices.push_back(v.get<uint32_t>());
  ref::Fetch f = ref::Fetch::Parse(c.fetch);
  if (levels.empty()) {
    ref::LevelRange lr = ref::GetLevelRange(f, nullptr);
    for (uint32_t l = lr.min_level; l <= lr.max_level; ++l) levels.push_back(l);
  }
  g_run_in_cp([&]() { g_acc->Clear(); });
  LevelOutputs outs;
  Result r = RunCase(c, &outs, &levels);
  if (r.kind == Result::kError || r.kind == Result::kUnsupported) {
    WriteError(outdir, r.msg);
    return;
  }
  for (uint32_t level : levels) {
    auto it = outs.find(level);
    if (it == outs.end()) {
      WriteError(outdir, "level " + std::to_string(level) + " not produced");
      return;
    }
    uint32_t w = std::max(f.width >> level, 1u);
    uint32_t h = std::max(f.height >> level, 1u);
    size_t slice_floats = size_t(w) * h * 4;
    size_t have = it->second.size() / slice_floats;
    std::vector<float> o;
    if (slices.empty()) {
      o = it->second;
    } else {
      for (uint32_t s : slices) {
        if (s >= have) continue;  // 3D: fewer depth slices at this level
        o.insert(o.end(), it->second.begin() + s * slice_floats,
                 it->second.begin() + (s + 1) * slice_floats);
      }
    }
    WriteFile(outdir / ("mip" + std::to_string(level)), o.data(),
              o.size() * sizeof(float));
  }
}

int RunCases(std::vector<Case>& cases, const std::string& only, bool list) {
  int fails = 0, errors = 0, passes = 0, unsupported = 0;
  std::map<std::string, std::array<uint64_t, 4>> per_format;  // pass,fail,err,unsup
  std::map<std::string, double> per_format_maxerr;
  std::map<std::string, std::pair<uint64_t, uint64_t>> per_format_exact;
  // Per (format, tiling) and per (format, tiling, mip) mismatch accounting.
  struct Key {
    std::string fmt;
    uint32_t tiled;
    uint32_t level;
    bool operator<(const Key& o) const {
      if (fmt != o.fmt) return fmt < o.fmt;
      if (tiled != o.tiled) return tiled < o.tiled;
      return level < o.level;
    }
  };
  std::map<Key, LevelStat> per_key;
  uint64_t total_comps = 0, total_exact = 0, total_near = 0, total_over = 0,
           total_one_lsb = 0;
  for (Case& c : cases) {
    if (!only.empty() && c.name.find(only) == std::string::npos) continue;
    if (g_bc_bitexact) {
      // Only the BC formats whose interpolation the XDK rounds to nearest.
      ref::Fetch bf = ref::Fetch::Parse(c.fetch);
      if (!IsRoundedBCFormat(BaseFormatOf(bf.format))) continue;
    }
    if (list) {
      std::printf("%s\n", c.name.c_str());
      continue;
    }
    g_run_in_cp([&]() { g_acc->Clear(); });
    Result r = RunCase(c);
    ref::Fetch f = ref::Fetch::Parse(c.fetch);
    std::string fmt = ref::GetFormatInfo(f.format).name;
    auto& pf = per_format[fmt];
    per_format_exact[fmt].first += r.exact;
    per_format_exact[fmt].second += r.comps;
    per_format_maxerr[fmt] = std::max(per_format_maxerr[fmt], r.max_err);
    total_comps += r.comps;
    total_exact += r.exact;
    for (auto& [level, ls] : r.levels) {
      LevelStat& agg = per_key[Key{fmt, f.tiled, level}];
      agg.comps += ls.comps;
      agg.exact += ls.exact;
      agg.near += ls.near;
      agg.over += ls.over;
      agg.one_lsb += ls.one_lsb;
      agg.max_err = std::max(agg.max_err, ls.max_err);
      total_near += ls.near;
      total_over += ls.over;
      total_one_lsb += ls.one_lsb;
    }
    switch (r.kind) {
      case Result::kPass:
        ++passes; ++pf[0];
        if (g_verbose)
          std::printf("PASS %s  bitexact %" PRIu64 "/%" PRIu64 " maxerr %.3g\n",
                      c.name.c_str(), r.exact, r.comps, r.max_err);
        break;
      case Result::kFail:
        ++fails; ++pf[1];
        std::printf("FAIL %s  maxerr %.6g at %s  bitexact %" PRIu64 "/%" PRIu64 "\n",
                    c.name.c_str(), r.max_err, r.worst.c_str(), r.exact, r.comps);
        break;
      case Result::kError:
        ++errors; ++pf[2];
        std::printf("ERROR %s  %s\n", c.name.c_str(), r.msg.c_str());
        break;
      case Result::kUnsupported:
        ++unsupported; ++pf[3];
        if (g_verbose) std::printf("UNSUPPORTED %s  %s\n", c.name.c_str(), r.msg.c_str());
        break;
    }
    std::fflush(stdout);
  }
  if (list) return 0;
  std::printf("\nper format: pass fail error unsupported | bit-exact components | max error\n");
  for (auto& [fmt, pf] : per_format) {
    auto& ex = per_format_exact[fmt];
    std::printf("  %-28s %4" PRIu64 " %4" PRIu64 " %4" PRIu64 " %4" PRIu64
                " | %" PRIu64 "/%" PRIu64 " | %.3g\n",
                fmt.c_str(), pf[0], pf[1], pf[2], pf[3], ex.first, ex.second,
                per_format_maxerr[fmt]);
  }
  std::printf(
      "\nmismatch by format / tiling / mip (tiled=1 tiled, 0 linear):\n"
      "  format                      tiled mip   components   not-bit-exact"
      "   1-LSB(1/255)   over-tol  max-error\n");
  for (auto& [k, ls] : per_key) {
    if (ls.near == 0 && ls.over == 0) continue;
    std::printf("  %-28s %4u %4u   %10" PRIu64 "   %10" PRIu64 "   %10" PRIu64
                "   %8" PRIu64 "   %.6g\n",
                k.fmt.c_str(), k.tiled, k.level, ls.comps, ls.near + ls.over,
                ls.one_lsb, ls.over, ls.max_err);
  }
  std::printf(
      "\ntotal components %" PRIu64 ", bit-exact %" PRIu64 " (%.4f%%), "
      "differ within tolerance %" PRIu64 ", differ over tolerance %" PRIu64
      ", differ by exactly 1/255 %" PRIu64 "\n",
      total_comps, total_exact,
      total_comps ? 100.0 * double(total_exact) / double(total_comps) : 0.0,
      total_near, total_over, total_one_lsb);
  std::printf("texconf: %d pass, %d fail, %d error, %d unsupported\n", passes,
              fails, errors, unsupported);
  std::fflush(stdout);
  if (g_bc_bitexact && !passes) {
    // The regression mode must not pass vacuously.
    std::printf("texconf: no BC cases ran\n");
    return 2;
  }
  return fails ? 1 : errors ? 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string only;
  bool synthetic = false, list = false;
  std::vector<std::string> texpacks;
  std::string adapter_case, adapter_list, adapter_store, adapter_out;
  std::string golden_write, referee_version = "unknown";
  std::vector<char*> cvar_argv = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-v") g_verbose = true;
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--synthetic") synthetic = true;
    else if (a == "--bc-bitexact") g_bc_bitexact = true;
    else if (a == "--list") list = true;
    else if (a == "--diff-dir" && i + 1 < argc) g_diff_dir = argv[++i];
    else if (a == "--dump" && i + 1 < argc)
      g_dump_limit = std::atoi(argv[++i]);
    else if ((a == "--golden-adapter" || a == "--batch") && i + 3 < argc) {
      (a == "--batch" ? adapter_list : adapter_case) = argv[++i];
      adapter_store = argv[++i];
      adapter_out = argv[++i];
      g_adapter_mode = true;
    } else if (a == "--golden-write" && i + 1 < argc) golden_write = argv[++i];
    else if (a == "--referee-version" && i + 1 < argc) referee_version = argv[++i];
    else if (a == "--texpack") {
      while (i + 1 < argc && argv[i + 1][0] != '-') texpacks.push_back(argv[++i]);
    } else if (a.rfind("--", 0) == 0 && a.find('=') != std::string::npos) {
      cvar_argv.push_back(argv[i]);
    } else {
      std::printf("unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  if (!synthetic && texpacks.empty() && !g_adapter_mode) synthetic = true;

  rex::cvar::Init(int(cvar_argv.size()), cvar_argv.data());
  rex::cvar::ApplyEnvironment();
  rex::cvar::FinalizeInit();
  {
    // Console sink: the harness is a CLI tool, and the Metal plugin reports
    // setup failures through the log (they abort the process).
    rex::LogConfig log_config;
    log_config.log_to_console = true;
    log_config.default_level =
        g_verbose ? spdlog::level::info : spdlog::level::warn;
    rex::InitLogging(log_config);
  }
  ApplyLegacyMetalEnv();

  std::vector<Case> cases;
  if (synthetic) {
    auto s = SynthCases();
    cases.insert(cases.end(), std::make_move_iterator(s.begin()),
                 std::make_move_iterator(s.end()));
  }
  for (const auto& p : texpacks) {
    if (!ReadTexpack(p, cases, only)) {
      std::printf("HARNESS: can't read texpack %s\n", p.c_str());
      return 2;
    }
  }
  if (list) return RunCases(cases, only, true);
  if (!golden_write.empty())
    return WriteGoldens(cases, only, golden_write, referee_version);

  auto* gs = new rex::graphics::metal::MetalGraphicsSystem();
  auto runtime = std::make_unique<rex::Runtime>(std::filesystem::path{});
  rex::RuntimeConfig config;
  config.graphics.reset(gs);
  if (runtime->Setup(std::move(config)) != 0) {
    std::printf("HARNESS: runtime setup failed\n");
    std::fflush(stdout);
    std::_Exit(2);
  }
  g_mem = runtime->memory();
  auto* cp = static_cast<rex::graphics::metal::MetalCommandProcessor*>(
      gs->command_processor());
  if (cp) {
    rex::thread::Fence ready;
    cp->CallInThread([&ready]() { ready.Signal(); });
    ready.Wait();
  }
  if (!cp || !cp->texture_cache() || !cp->shared_memory()) {
    std::printf("HARNESS: no Metal command processor / texture cache\n");
    std::fflush(stdout);
    std::_Exit(2);
  }
  TexConfAccess acc(cp, gs->register_file());
  g_acc = &acc;
  g_run_in_cp = [cp](std::function<void()> fn) {
    rex::thread::Fence fence;
    cp->CallInThread([&]() {
      fn();
      fence.Signal();
    });
    fence.Wait();
  };
  if (!CreatePipelines(cp->GetMetalDevice())) std::_Exit(2);

  if (g_adapter_mode) {
    gs::Store store(adapter_store);
    if (!adapter_case.empty()) {
      AdapterOne(adapter_case, store, adapter_out);
    } else {
      std::ifstream in(adapter_list);
      if (!in) {
        std::printf("HARNESS: can't read list %s\n", adapter_list.c_str());
        std::_Exit(2);
      }
      std::string line;
      size_t n = 0;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::filesystem::path p(line);
        std::string id = p.stem().string();
        AdapterOne(p, store, std::filesystem::path(adapter_out) / id);
        ++n;
      }
      std::printf("texconf adapter: %zu cases\n", n);
    }
    std::fflush(stdout);
    std::_Exit(0);
  }
  int rc = RunCases(cases, only, false);
  std::_Exit(rc);
}
