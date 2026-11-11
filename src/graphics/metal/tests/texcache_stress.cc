// Deterministic texture cache staleness stress test for the Metal GPU plugin.
//
// Builds the real MetalGraphicsSystem (real MetalCommandProcessor,
// MetalSharedMemory, MetalTextureCache and MTLDevice) on top of a real
// rex::Runtime with guest memory, then drives texture loads the way a draw
// does (PrepareTextureMaterialization + ExecuteTextureMaterialization on the
// command processor thread). Guest writes are injected at exact points inside
// the loads through the REX_METAL_TEST_HOOKS hooks (metal_test_hooks.h).
//
// After each scenario the texture the cache would sample is read back from the
// MTLTexture and compared with a reference: the same fetch constant pointed at
// a fresh copy of the current guest bytes (a brand new cache entry that has no
// history). A mismatch means the cache served stale bytes.
//
// Exit code: 0 all scenarios clean, 1 stale serve found, 2 harness error.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rex/cvar.h"
#include "rex/graphics/metal/metal_command_processor.h"
#include "rex/graphics/metal/metal_graphics_system.h"
#include "rex/graphics/metal/metal_shared_memory.h"
#include "rex/graphics/metal/metal_test_hooks.h"
#include "rex/graphics/metal/metal_texture_cache.h"
#include "rex/graphics/register_file.h"
#include "rex/graphics/xenos.h"
#include "rex/logging.h"
#include "metal_flags.h"
#include "rex/runtime.h"
#include "rex/system/xmemory.h"
#include "rex/thread.h"
#include "rex/thread/mutex.h"
#include "third_party/metal-cpp/Metal/Metal.hpp"
#include "xxhash.h"

namespace rex::graphics::metal {

namespace hooks = test_hooks;

// Friend of MetalCommandProcessor and MetalTextureCache (REX_METAL_TEST_FRIEND).
class TexCacheStressAccess {
 public:
  using Plan = MetalTextureCache::TextureMaterializationPlan;
  using Texture = MetalTextureCache::Texture;
  using TextureKey = MetalTextureCache::TextureKey;

  TexCacheStressAccess(MetalCommandProcessor* cp, RegisterFile* regs)
      : cp_(cp), regs_(regs) {}

  MetalTextureCache& tc() { return *cp_->texture_cache(); }
  MetalSharedMemory& sm() { return *cp_->shared_memory(); }

  void SetFetch(const xenos::xe_gpu_texture_fetch_t& fetch) {
    std::memcpy(&regs_->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], &fetch,
                sizeof(fetch));
  }

  // One draw's texture materialization for fetch slot 0.
  bool Materialize(const xenos::xe_gpu_texture_fetch_t& fetch) {
    SetFetch(fetch);
    if (!cp_->EnsureCommandBuffer()) return false;
    Plan plan;
    if (!tc().PrepareTextureMaterialization(*regs_, 1, plan)) return false;
    if (!plan.source_ranges.empty() &&
        !cp_->RequestSharedMemoryRanges(
            MetalCommandProcessor::SharedMemoryRequestReason::kUnknown,
            plan.source_ranges.data(),
            uint32_t(plan.source_ranges.size()))) {
      return false;
    }
    tc().BeginTextureUploadBatch();
    bool ok = tc().ExecuteTextureMaterialization(plan);
    ok &= tc().EndTextureUploadBatch();
    tc().RequestTexturesWithoutLoading(1, nullptr);
    return ok;
  }

  bool MakeResident(uint32_t start, uint32_t length) {
    if (!cp_->EnsureCommandBuffer()) return false;
    return cp_->RequestSharedMemoryRange(
        MetalCommandProcessor::SharedMemoryRequestReason::kUnknown, start,
        length);
  }

  void Flush() { cp_->FlushCommandBufferAndWait(5000000000ULL, "texcache-stress"); }
  // Completion processing (ProcessCompletedSubmissions: texture cache, then
  // the memexport readbacks), as the next submission would run it.
  void ProcessCompletions() { cp_->ProcessCompletedSubmissions(); }

  Texture* Find(const xenos::xe_gpu_texture_fetch_t& fetch) {
    TextureKey key;
    uint8_t signs;
    MetalTextureCache::BindingInfoFromFetchConstant(fetch, key, &signs);
    if (!key.is_valid) return nullptr;
    return tc().FindOrCreateTexture(key);
  }

  // Reads every mip level of slice 0 back into `levels`. Returns false with
  // `err` set if the texture can't be read back.
  bool Readback(Texture* texture, std::vector<std::vector<uint8_t>>& levels,
                std::string& err) {
    levels.clear();
    auto* mt = static_cast<MetalTextureCache::MetalTexture*>(texture);
    MTL::Texture* t = mt->metal_texture();
    if (!t) {
      err = "no MTLTexture";
      return false;
    }
    uint32_t block = 1, block_bytes = 0;
    switch (t->pixelFormat()) {
      case MTL::PixelFormatRGBA8Unorm:
      case MTL::PixelFormatBGRA8Unorm:
      case MTL::PixelFormatRGBA8Snorm:
      case MTL::PixelFormatR32Uint:
      case MTL::PixelFormatR32Float:
      case MTL::PixelFormatRG16Unorm:
        block_bytes = 4;
        break;
      case MTL::PixelFormatBC1_RGBA:
      case MTL::PixelFormatBC4_RUnorm:
        block = 4;
        block_bytes = 8;
        break;
      case MTL::PixelFormatBC2_RGBA:
      case MTL::PixelFormatBC3_RGBA:
      case MTL::PixelFormatBC5_RGUnorm:
        block = 4;
        block_bytes = 16;
        break;
      case MTL::PixelFormatRG8Unorm:
      case MTL::PixelFormatR16Unorm:
        block_bytes = 2;
        break;
      case MTL::PixelFormatR8Unorm:
        block_bytes = 1;
        break;
      default:
        err = "unhandled MTLPixelFormat " + std::to_string(uint32_t(t->pixelFormat()));
        return false;
    }
    Flush();
    MTL::Device* dev = cp_->GetMetalDevice();
    const uint32_t level_count = uint32_t(t->mipmapLevelCount());
    std::vector<size_t> offsets, sizes, bprs;
    size_t total = 0;
    for (uint32_t l = 0; l < level_count; ++l) {
      uint32_t w = std::max<uint32_t>(1, uint32_t(t->width()) >> l);
      uint32_t h = std::max<uint32_t>(1, uint32_t(t->height()) >> l);
      size_t bpr = size_t((w + block - 1) / block) * block_bytes;
      size_t size = bpr * ((h + block - 1) / block);
      offsets.push_back(total);
      sizes.push_back(size);
      bprs.push_back(bpr);
      total += (size + 255) & ~size_t(255);
    }
    MTL::Buffer* buf = dev->newBuffer(total, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cb = cp_->GetMetalCommandQueue()->commandBuffer();
    MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
    for (uint32_t l = 0; l < level_count; ++l) {
      uint32_t w = std::max<uint32_t>(1, uint32_t(t->width()) >> l);
      uint32_t h = std::max<uint32_t>(1, uint32_t(t->height()) >> l);
      blit->copyFromTexture(t, 0, l, MTL::Origin(0, 0, 0), MTL::Size(w, h, 1),
                            buf, offsets[l], bprs[l], sizes[l]);
    }
    blit->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    const uint8_t* p = static_cast<const uint8_t*>(buf->contents());
    for (uint32_t l = 0; l < level_count; ++l) {
      levels.emplace_back(p + offsets[l], p + offsets[l] + sizes[l]);
    }
    buf->release();
    return true;
  }

  uint64_t LoadCount() { return tc().loaded_texture_data_count(); }

#ifdef REX_METAL_3D_AS_2D_REFRESH
  // The 2D wrapper a draw sampling the 3D texture of `fetch` as 2D gets
  // (gpu_3d_to_2d_texture), created on first use.
  Texture* Wrapper3DAs2D(const xenos::xe_gpu_texture_fetch_t& fetch) {
    auto* t = static_cast<MetalTextureCache::MetalTexture*>(Find(fetch));
    if (!t || !cp_->EnsureCommandBuffer() ||
        !t->GetOrCreate3DAs2DView(xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
                                  xenos::FetchOpDimension::k2D, false)) {
      return nullptr;
    }
    return t->texture_3d_as_2d();
  }
#endif

  // A resolve / memexport style GPU write: valid with GPU-written data.
  void MarkGpuWritten(uint32_t start, uint32_t length) {
    MetalSharedMemory& s = sm();
    s.MakeRangeValid(start, length, true);
  }
  // clear_memory_page_state at the end of a frame.
  void EndOfFramePageReset() { sm().SetSystemPageBlocksValidWithGpuDataWritten(); }

  // A memexport draw (the draw's memexport range request, then IssueDraw's
  // memexport_ranges loop): the export lands in
  // the shared buffer and is copied back to guest RAM once this submission
  // completes. The caller flushes first so no earlier load of the open command
  // buffer sees the bytes written here.
  bool Memexport(uint32_t start, const std::vector<uint8_t>& bytes) {
    if (!cp_->EnsureCommandBuffer()) return false;
    MetalSharedMemory& s = sm();
    const uint32_t len = uint32_t(bytes.size());
    // The draw requests its memexport ranges first, like any other source.
    if (!cp_->RequestSharedMemoryRange(
            MetalCommandProcessor::SharedMemoryRequestReason::kMemexportStream,
            start, len)) {
      return false;
    }
    if (MetalSharedMemory::MemexportPreserveEnabled()) {
      s.NoteMemexportWrite(start, len);
    }
    if (MetalSharedMemory::MemexportReadbackEnabled()) {
      s.NoteMemexportReadback(start, len, cp_->submission_current_);
    }
    std::memcpy(static_cast<uint8_t*>(s.GetBuffer()->contents()) + start,
                bytes.data(), len);
    s.RangeWrittenByGpu(start, len);
    s.MarkGpuWrite(start & (SharedMemory::kBufferSize - 1), len,
                   cp_->submission_current_);
    return true;
  }

  // Same as Memexport, but the export bytes are written by the GPU in
  // command buffer order (a blit in the open command buffer standing in for
  // the memexport draw), so loads encoded before it see the old bytes and
  // loads encoded after it see the new ones, as with a real export.
  bool MemexportGpu(uint32_t start, const std::vector<uint8_t>& bytes) {
    if (!cp_->EnsureCommandBuffer()) return false;
    MetalSharedMemory& s = sm();
    const uint32_t len = uint32_t(bytes.size());
    if (!cp_->RequestSharedMemoryRange(
            MetalCommandProcessor::SharedMemoryRequestReason::kMemexportStream,
            start, len)) {
      return false;
    }
    MTL::Buffer* staging = cp_->GetMetalDevice()->newBuffer(
        bytes.data(), len, MTL::ResourceStorageModeShared);
    if (!staging) return false;
    MTL::BlitCommandEncoder* blit = cp_->GetSharedMemoryUploadBlitEncoder();
    if (!blit) {
      staging->release();
      return false;
    }
    blit->copyFromBuffer(staging, 0, s.GetBuffer(), start, len);
    cp_->EndSharedMemoryUploadBlitEncoder();
    cp_->GetCurrentCommandBuffer()->addCompletedHandler(
        [staging](MTL::CommandBuffer*) { staging->release(); });
    if (MetalSharedMemory::MemexportPreserveEnabled()) {
      s.NoteMemexportWrite(start, len);
    }
    if (MetalSharedMemory::MemexportReadbackEnabled()) {
      s.NoteMemexportReadback(start, len, cp_->submission_current_);
    }
    s.RangeWrittenByGpu(start, len);
    s.MarkGpuWrite(start & (SharedMemory::kBufferSize - 1), len,
                   cp_->submission_current_);
    return true;
  }

  // A GPU-ordered memexport draw whose declared export range is
  // [start, start + decl_len) but whose shader writes only `bytes` at
  // start + write_off (a copy shader exports only the elements it is given;
  // the eM stream constant size is often larger). Mirrors IssueDraw:
  //   1. the draw's range request (request == false: the prepare-time
  //      has_invalid_shared_memory was false, as when a CPU write invalidated
  //      the range after prepare; the execute-time re-check may still upload);
  //   2. `between` (the draw being encoded: a CPU write landing here comes
  //      after the request's upload and before the post-draw marking);
  //   3. the export (blit in command buffer order);
  //   4. IssueDraw's memexport_ranges loop over the declared range.
  bool MemexportGpuPartial(uint32_t start, uint32_t decl_len, uint32_t write_off,
                           const std::vector<uint8_t>& bytes, bool request,
                           const std::function<void()>& between) {
    if (!cp_->EnsureCommandBuffer()) return false;
    MetalSharedMemory& s = sm();
    // [memexport-exec-arm] as SubmitPreparedDraw: execute-time re-check and
    // readback snapshot (off with REX_METAL_MEMEXPORT_EXEC_ARM=0).
    const draw_util::MemExportRange mx(start >> 2, decl_len);
    const SharedMemory::Range dr{start, decl_len};
    const bool invalid_now =
        cp_->RecheckMemexportRangesForExecute(&mx, 1, &dr, 1);
    if ((request || invalid_now) &&
        !cp_->RequestSharedMemoryRange(
            MetalCommandProcessor::SharedMemoryRequestReason::kMemexportStream,
            start, decl_len)) {
      return false;
    }
    if (between) between();
    if (!cp_->EnsureCommandBuffer()) return false;
    const uint32_t len = uint32_t(bytes.size());
    MTL::Buffer* staging = cp_->GetMetalDevice()->newBuffer(
        bytes.data(), len, MTL::ResourceStorageModeShared);
    if (!staging) return false;
    MTL::BlitCommandEncoder* blit = cp_->GetSharedMemoryUploadBlitEncoder();
    if (!blit) {
      staging->release();
      return false;
    }
    blit->copyFromBuffer(staging, 0, s.GetBuffer(), start + write_off, len);
    cp_->EndSharedMemoryUploadBlitEncoder();
    cp_->GetCurrentCommandBuffer()->addCompletedHandler(
        [staging](MTL::CommandBuffer*) { staging->release(); });
    if (MetalSharedMemory::MemexportPreserveEnabled()) {
      s.NoteMemexportWrite(start, decl_len);
    }
    if (MetalSharedMemory::MemexportReadbackEnabled()) {
      s.NoteMemexportReadback(start, decl_len, cp_->submission_current_);
    }
    s.RangeWrittenByGpu(start, decl_len);
    s.MarkGpuWrite(start & (SharedMemory::kBufferSize - 1), decl_len,
                   cp_->submission_current_);
    return true;
  }

  // A memexport copy draw (a vs-only copy shader, as the game's texture
  // MOVE): it reads [src, src + copy_len) from the shared buffer (vfetch) and
  // exports it at dst + write_off, inside a declared export range
  // [dst, dst + decl_len). The draw's shared-memory ranges are the source
  // and the export range, as SubmitPreparedDraw's materialization_ranges.
  // request: the prepare-time has_invalid_shared_memory.
  bool MemexportGpuCopy(uint32_t src, uint32_t copy_len, uint32_t dst,
                        uint32_t decl_len, uint32_t write_off, bool request) {
    if (!cp_->EnsureCommandBuffer()) return false;
    MetalSharedMemory& s = sm();
    const draw_util::MemExportRange mx(dst >> 2, decl_len);
    const SharedMemory::Range dr[2] = {{src, copy_len}, {dst, decl_len}};
    const bool invalid_now =
        cp_->RecheckMemexportRangesForExecute(&mx, 1, dr, 2);
    if ((request || invalid_now) &&
        !cp_->RequestSharedMemoryRanges(
            MetalCommandProcessor::SharedMemoryRequestReason::kDrawMaterialization,
            dr, 2)) {
      return false;
    }
    if (!cp_->EnsureCommandBuffer()) return false;
    MTL::BlitCommandEncoder* blit = cp_->GetSharedMemoryUploadBlitEncoder();
    if (!blit) return false;
    blit->copyFromBuffer(s.GetBuffer(), src, s.GetBuffer(), dst + write_off,
                         copy_len);
    cp_->EndSharedMemoryUploadBlitEncoder();
    if (MetalSharedMemory::MemexportPreserveEnabled()) {
      s.NoteMemexportWrite(dst, decl_len);
    }
    if (MetalSharedMemory::MemexportReadbackEnabled()) {
      s.NoteMemexportReadback(dst, decl_len, cp_->submission_current_);
    }
    s.RangeWrittenByGpu(dst, decl_len);
    s.MarkGpuWrite(dst & (SharedMemory::kBufferSize - 1), decl_len,
                   cp_->submission_current_);
    return true;
  }

  // A 1x resolve into guest memory: the resolved bytes land in the shared
  // buffer, then the render target cache marks the range (rtt: the
  // [effects-native] MarkRangeAsResolvedUnscaledRtt route, else the base
  // MarkRangeAsResolved). The caller flushes first, as for Memexport.
  bool Resolve(uint32_t start, const std::vector<uint8_t>& bytes, bool rtt) {
    if (!cp_->EnsureCommandBuffer()) return false;
    const uint32_t len = uint32_t(bytes.size());
    std::memcpy(static_cast<uint8_t*>(sm().GetBuffer()->contents()) + start,
                bytes.data(), len);
    if (rtt) {
      tc().MarkRangeAsResolvedUnscaledRtt(start, len);
    } else {
      tc().MarkRangeAsResolved(start, len);
    }
    sm().MarkGpuWrite(start, len, cp_->submission_current_);
    return true;
  }

 private:
  MetalCommandProcessor* cp_;
  RegisterFile* regs_;
};

}  // namespace rex::graphics::metal

namespace {

using rex::graphics::metal::TexCacheStressAccess;
using rex::graphics::xenos::xe_gpu_texture_fetch_t;
namespace xenos = rex::graphics::xenos;
namespace hooks = rex::graphics::metal::test_hooks;

rex::memory::Memory* g_mem = nullptr;
TexCacheStressAccess* g_acc = nullptr;
int g_failures = 0;
int g_harness_errors = 0;
bool g_verbose = false;
// When set, Verify takes the base level bytes from here instead of guest RAM:
// for scenarios where guest RAM itself may hold the wrong bytes.
const std::vector<uint8_t>* g_expected_base = nullptr;

// ---------------------------------------------------------------------------
// Guest regions
// ---------------------------------------------------------------------------

struct Region {
  uint32_t va = 0;    // guest virtual (0xA/0xC/0xE physical view)
  uint32_t phys = 0;  // physical
  uint32_t size = 0;
};

Region Alloc(uint32_t size) {
  // 64 KB alignment keeps every region on its own 16 KB host pages, so the
  // page protection of one region never covers another.
  size = (size + 0xFFFF) & ~0xFFFFu;
  Region r;
  r.va = g_mem->SystemHeapAlloc(size, 0x10000, rex::memory::kSystemHeapPhysical);
  r.phys = g_mem->GetPhysicalAddress(r.va);
  r.size = size;
  return r;
}

void Fill(std::vector<uint8_t>& v, uint32_t seed) {
  uint32_t x = seed * 2654435761u + 0x9E3779B9u;
  for (auto& b : v) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    b = uint8_t(x);
  }
}

// A store from guest code: goes through the guest virtual view, so it faults
// on watched pages and runs the access watch callbacks.
void GuestWrite(const Region& r, uint32_t offset, const std::vector<uint8_t>& d) {
  volatile uint8_t* p = g_mem->TranslateVirtual<volatile uint8_t*>(r.va + offset);
  for (size_t i = 0; i < d.size(); ++i) p[i] = d[i];
}

// A host write that does not fault (file read DMA style): writes through the
// physical view. The caller triggers the callbacks afterwards (or never).
void HostWrite(const Region& r, uint32_t offset, const std::vector<uint8_t>& d) {
  std::memcpy(g_mem->TranslatePhysical(r.phys + offset), d.data(), d.size());
}
void HostWriteCallbacks(const Region& r, uint32_t offset, uint32_t length) {
  g_mem->TriggerPhysicalMemoryCallbacks(
      rex::thread::global_critical_region::AcquireDirect(), r.va + offset,
      length, true, true);
}

void ReadGuest(const Region& r, uint32_t offset, uint32_t length,
               std::vector<uint8_t>& out) {
  out.resize(length);
  std::memcpy(out.data(), g_mem->TranslatePhysical(r.phys + offset), length);
}
// Writes to a fresh (never loaded, never watched) region.
void Seed(const Region& r, uint32_t offset, const std::vector<uint8_t>& d) {
  std::memcpy(g_mem->TranslatePhysical(r.phys + offset), d.data(), d.size());
}

// ---------------------------------------------------------------------------
// Texture descriptions
// ---------------------------------------------------------------------------

struct TexDesc {
  xenos::TextureFormat format;
  uint32_t width, height;
  bool tiled;
  uint32_t mip_max_level;  // 0 = base only
  bool packed_mips;
  const char* name;
  xenos::DataDimension dimension = xenos::DataDimension::k2DOrStacked;
  uint32_t depth = 1;  // k3D only
};

xe_gpu_texture_fetch_t MakeFetch(const TexDesc& d, uint32_t base_phys,
                                 uint32_t mip_phys) {
  xe_gpu_texture_fetch_t f;
  std::memset(&f, 0, sizeof(f));
  f.type = xenos::FetchConstantType::kTexture;
  f.tiled = d.tiled ? 1 : 0;
  f.pitch = ((d.width + 31) / 32);
  f.format = d.format;
  f.endianness = xenos::Endian::kNone;
  f.base_address = base_phys >> 12;
  if (d.dimension == xenos::DataDimension::k3D) {
    f.size_3d.width = d.width - 1;
    f.size_3d.height = d.height - 1;
    f.size_3d.depth = d.depth - 1;
  } else {
    f.size_2d.width = d.width - 1;
    f.size_2d.height = d.height - 1;
  }
  f.swizzle = 0 | (1 << 3) | (2 << 6) | (3 << 9);
  f.mip_min_level = 0;
  f.mip_max_level = d.mip_max_level;
  f.dimension = d.dimension;
  f.packed_mips = d.packed_mips ? 1 : 0;
  f.mip_address = d.mip_max_level ? (mip_phys >> 12) : 0;
  return f;
}

// Guest extents of a texture, as the cache computes them.
struct Extents {
  uint32_t base = 0, mips = 0;
};
Extents GetExtents(const xe_gpu_texture_fetch_t& f) {
  Extents e;
  auto* t = g_acc->Find(f);
  if (t) {
    e.base = t->GetGuestBaseSize();
    e.mips = t->GetGuestMipsSize();
  }
  return e;
}

// ---------------------------------------------------------------------------
// Hook plumbing
// ---------------------------------------------------------------------------

struct Injection {
  hooks::Point point;
  uint32_t base_page;      // only fire for the texture whose base is here
  int remaining = 1;       // fire this many times
  std::function<void()> action;
  int fired = 0;
};
Injection* g_injection = nullptr;

void HookFn(hooks::Point point, uint32_t base_page) {
  Injection* inj = g_injection;
  if (!inj || inj->remaining <= 0 || point != inj->point) return;
  if (base_page != (inj->base_page & 0x1FFFF)) return;
  --inj->remaining;
  ++inj->fired;
  inj->action();
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

uint64_t HashLevels(const std::vector<std::vector<uint8_t>>& levels) {
  XXH3_state_t* s = XXH3_createState();
  XXH3_64bits_reset(s);
  for (auto& l : levels) XXH3_64bits_update(s, l.data(), l.size());
  uint64_t h = XXH3_64bits_digest(s);
  XXH3_freeState(s);
  return h;
}

const char* FormatName(xenos::TextureFormat f) {
  switch (f) {
    case xenos::TextureFormat::k_8_8_8_8: return "k_8_8_8_8";
    case xenos::TextureFormat::k_DXT1: return "k_DXT1";
    case xenos::TextureFormat::k_DXT2_3: return "k_DXT2_3";
    case xenos::TextureFormat::k_DXT4_5: return "k_DXT4_5";
    case xenos::TextureFormat::k_DXN: return "k_DXN";
    default: return "fmt?";
  }
}

// Reference: copy the texture's current guest bytes into a fresh twin region
// and load the same fetch constant from there (new cache entry, no history).
bool Verify(const char* scenario, const TexDesc& d, const Region& base,
            const Region* mips) {
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, mips ? mips->phys : 0);
  // The next draw: whatever the cache still believes is current gets served.
  if (!g_acc->Materialize(f)) {
    std::printf("HARNESS scenario %s: materialize failed\n", scenario);
    ++g_harness_errors;
    return false;
  }
  auto* tex = g_acc->Find(f);
  std::vector<std::vector<uint8_t>> served, ref;
  std::string err;
  if (!tex || !g_acc->Readback(tex, served, err)) {
    std::printf("HARNESS scenario %s: readback failed: %s\n", scenario, err.c_str());
    ++g_harness_errors;
    return false;
  }
  Extents e = GetExtents(f);
  Region tb = Alloc(base.size);
  Region tm;
  std::vector<uint8_t> bytes;
  if (g_expected_base) {
    bytes = *g_expected_base;
  } else {
    ReadGuest(base, 0, base.size, bytes);
  }
  Seed(tb, 0, bytes);
  if (mips) {
    tm = Alloc(mips->size);
    ReadGuest(*mips, 0, mips->size, bytes);
    Seed(tm, 0, bytes);
  }
  xe_gpu_texture_fetch_t rf = MakeFetch(d, tb.phys, mips ? tm.phys : 0);
  if (!g_acc->Materialize(rf)) {
    std::printf("HARNESS scenario %s: reference materialize failed\n", scenario);
    ++g_harness_errors;
    return false;
  }
  auto* rtex = g_acc->Find(rf);
  if (!rtex || !g_acc->Readback(rtex, ref, err)) {
    std::printf("HARNESS scenario %s: reference readback failed: %s\n", scenario,
                err.c_str());
    ++g_harness_errors;
    return false;
  }
  const uint64_t hs = HashLevels(served), hr = HashLevels(ref);
  if (hs != hr) {
    std::string bad;
    for (size_t l = 0; l < std::min(served.size(), ref.size()); ++l) {
      if (served[l] == ref[l]) continue;
      size_t first = 0, last = 0, n = std::min(served[l].size(), ref[l].size());
      while (first < n && served[l][first] == ref[l][first]) ++first;
      for (size_t i = first; i < n; ++i) {
        if (served[l][i] != ref[l][i]) last = i;
      }
      bad += (bad.empty() ? "" : ",") + std::to_string(l) + " (host bytes " +
             std::to_string(first) + ".." + std::to_string(last) + ")";
    }
    std::printf(
        "FAIL scenario %s: texture at guest 0x%08X (mips 0x%08X) fmt %s %ux%u "
        "mips %u%s served stale bytes (expected hash %016" PRIX64
        ", got %016" PRIX64 "; levels differing: %s; base %u B, mips %u B)\n",
        scenario, base.phys, mips ? mips->phys : 0, FormatName(d.format), d.width,
        d.height, d.mip_max_level, d.packed_mips ? " packed" : "", hr, hs,
        bad.c_str(), e.base, e.mips);
    ++g_failures;
    return false;
  }
  if (g_verbose) {
    std::printf("  ok %s (hash %016" PRIX64 ", %zu levels)\n", scenario, hs,
                served.size());
  }
  return true;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

const TexDesc kRgba64 = {xenos::TextureFormat::k_8_8_8_8, 64, 64, true, 0, false,
                         "rgba8 64x64 tiled"};
const TexDesc kDxt1_128 = {xenos::TextureFormat::k_DXT1, 128, 128, true, 0, false,
                           "dxt1 128x128 tiled"};
const TexDesc kDxt5_256m = {xenos::TextureFormat::k_DXT4_5, 256, 256, true, 8, true,
                            "dxt5 256x256 tiled mips packed"};
const TexDesc kDxt1_128m = {xenos::TextureFormat::k_DXT1, 128, 128, true, 7, true,
                            "dxt1 128x128 tiled mips packed"};

std::vector<uint8_t> Pattern(uint32_t size, uint32_t seed) {
  std::vector<uint8_t> v(size);
  Fill(v, seed);
  return v;
}

enum class Path { kCpuSource, kResident };

// Load the texture once from content A via `path`; while that load runs, a
// write of content B lands at `point` (through `writer`). Then the next draw
// must see B.
void RaceScenario(const char* name, const TexDesc& d, Path path,
                  hooks::Point point, bool host_write, bool callbacks_late,
                  bool write_mips_only = false) {
  const bool has_mips = d.mip_max_level != 0;
  Region base = Alloc(256 * 1024);
  Region mips = has_mips ? Alloc(256 * 1024) : Region{};
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, has_mips ? mips.phys : 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  const uint32_t mlen = (e.mips + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 1));
  if (has_mips) Seed(mips, 0, Pattern(mips.size, 2));
  if (path == Path::kResident) {
    g_acc->MakeResident(base.phys, blen);
    if (has_mips) g_acc->MakeResident(mips.phys, mlen);
  }
  const Region& target = write_mips_only ? mips : base;
  const uint32_t tlen = write_mips_only ? mlen : blen;
  // For the mip tail write only the last 4 KB (where the packed tail lives).
  const uint32_t woff = write_mips_only ? tlen - 0x1000 : 0;
  const uint32_t wlen = write_mips_only ? 0x1000 : tlen;
  auto content_b = Pattern(wlen, 3);
  Injection inj;
  inj.point = point;
  inj.base_page = base.phys >> 12;
  inj.action = [&]() {
    if (host_write) {
      HostWrite(target, woff, content_b);
      if (!callbacks_late) HostWriteCallbacks(target, woff, wlen);
    } else {
      GuestWrite(target, woff, content_b);
    }
  };
  g_injection = &inj;
  g_acc->Materialize(f);
  g_injection = nullptr;
  if (host_write && callbacks_late) HostWriteCallbacks(target, woff, wlen);
  if (!inj.fired) {
    std::printf("HARNESS scenario %s: hook point never reached (path not taken)\n",
                name);
    ++g_harness_errors;
    return;
  }
  Verify(name, d, base, has_mips ? &mips : nullptr);
}

// Same base address reused by textures of different format / size, as a
// streaming pool recycles a block. Each reuse writes new content first.
void ReuseScenario(const char* name, bool resident, bool race) {
  Region base = Alloc(512 * 1024);
  Region mips = Alloc(256 * 1024);
  const TexDesc seq[] = {kRgba64, kDxt1_128, kDxt5_256m, kRgba64, kDxt1_128m,
                         kDxt1_128};
  uint32_t seed = 10;
  for (size_t i = 0; i < sizeof(seq) / sizeof(seq[0]); ++i) {
    const TexDesc& d = seq[i];
    const bool has_mips = d.mip_max_level != 0;
    xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, mips.phys);
    Extents e = GetExtents(f);
    const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
    const uint32_t mlen = (e.mips + 0xFFF) & ~0xFFFu;
    // The streamer writes the new texture data into the recycled block.
    GuestWrite(base, 0, Pattern(blen, seed++));
    if (has_mips) GuestWrite(mips, 0, Pattern(mlen, seed++));
    if (resident) {
      g_acc->MakeResident(base.phys, blen);
      if (has_mips) g_acc->MakeResident(mips.phys, mlen);
    }
    Injection inj;
    auto late = Pattern(blen, seed++);
    if (race) {
      inj.point = resident ? hooks::Point::kResidentAfterCopy
                           : hooks::Point::kCpuSourceAfterCopy;
      inj.base_page = base.phys >> 12;
      inj.action = [&]() { GuestWrite(base, 0, late); };
      g_injection = &inj;
    }
    g_acc->Materialize(f);
    g_injection = nullptr;
    std::string step = std::string(name) + " step" + std::to_string(i) + " (" +
                       d.name + ")";
    Verify(step.c_str(), d, base, has_mips ? &mips : nullptr);
  }
}

// Streamed mip upgrade: base arrives first (fetch with no mips), mips arrive
// later; then only the packed mip tail is rewritten (with and without a race).
void MipStreamScenario(const char* name, const TexDesc& full, bool resident,
                       bool race_tail) {
  Region base = Alloc(512 * 1024);
  Region mips = Alloc(256 * 1024);
  TexDesc base_only = full;
  base_only.mip_max_level = 0;
  base_only.packed_mips = false;
  xe_gpu_texture_fetch_t fb = MakeFetch(base_only, base.phys, 0);
  xe_gpu_texture_fetch_t ff = MakeFetch(full, base.phys, mips.phys);
  Extents e = GetExtents(ff);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  const uint32_t mlen = (e.mips + 0xFFF) & ~0xFFFu;
  // Low-res placeholder in the mip region, base streamed in.
  Seed(base, 0, Pattern(base.size, 20));
  Seed(mips, 0, Pattern(mips.size, 21));
  if (resident) g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(fb);
  std::string s1 = std::string(name) + " base-only";
  Verify(s1.c_str(), base_only, base, nullptr);
  // Mips arrive.
  GuestWrite(mips, 0, Pattern(mlen, 22));
  if (resident) g_acc->MakeResident(mips.phys, mlen);
  g_acc->Materialize(ff);
  std::string s2 = std::string(name) + " mips-arrive";
  Verify(s2.c_str(), full, base, &mips);
  // Packed mip tail rewritten (the small mips live in the last tile).
  const uint32_t tail_off = mlen - 0x1000;
  if (race_tail) {
    if (resident) g_acc->MakeResident(mips.phys, mlen);
    // Make the texture outdated through its base so a load runs, and land the
    // tail write inside that load.
    GuestWrite(base, 0, Pattern(blen, 23));
    if (resident) g_acc->MakeResident(base.phys, blen);
    auto tail = Pattern(0x1000, 24);
    Injection inj;
    inj.point = resident ? hooks::Point::kResidentAfterCopy
                         : hooks::Point::kCpuSourceAfterCopy;
    inj.base_page = base.phys >> 12;
    inj.action = [&]() { GuestWrite(mips, tail_off, tail); };
    g_injection = &inj;
    g_acc->Materialize(ff);
    g_injection = nullptr;
  } else {
    GuestWrite(mips, tail_off, Pattern(0x1000, 25));
    if (resident) g_acc->MakeResident(mips.phys, mlen);
    g_acc->Materialize(ff);
  }
  std::string s3 = std::string(name) + " tail-rewrite";
  Verify(s3.c_str(), full, base, &mips);
  // And back to the base-only fetch (the streamer dropped the mips).
  GuestWrite(base, 0, Pattern(blen, 26));
  if (resident) g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(fb);
  std::string s4 = std::string(name) + " base-only-again";
  Verify(s4.c_str(), base_only, base, nullptr);
}

// clear_memory_page_state: the end-of-frame SetSystemPageBlocksValidWithGpuDataWritten
// drops the valid bits of CPU-uploaded pages without firing texture watches.
void PageResetScenario(const char* name, const TexDesc& d, int resets,
                       bool reload_between) {
  Region base = Alloc(256 * 1024);
  Region gpu = Alloc(64 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 30));
  g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  for (int i = 0; i < resets; ++i) {
    // Some GPU-written data this frame (a resolve), so the reset runs.
    g_acc->MarkGpuWritten(gpu.phys, 0x1000);
    g_acc->EndOfFramePageReset();
    if (reload_between) {
      g_acc->Materialize(f);
    }
  }
  GuestWrite(base, 0, Pattern(blen, 31));
  // Another frame reset before the draw.
  g_acc->MarkGpuWritten(gpu.phys, 0x1000);
  g_acc->EndOfFramePageReset();
  Verify(name, d, base, nullptr);
  // And a texture whose data the GPU wrote (resolve) then the CPU overwrote.
}

// GPU-written (resolve) pages followed by a CPU overwrite across frame resets.
void ResolveThenCpuScenario(const char* name, const TexDesc& d) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  // The "resolve": GPU data in the shared buffer and the guest copy agree.
  Seed(base, 0, Pattern(base.size, 40));
  g_acc->MakeResident(base.phys, blen);
  g_acc->MarkGpuWritten(base.phys, blen);
  g_acc->EndOfFramePageReset();
  g_acc->Materialize(f);
  g_acc->EndOfFramePageReset();
  GuestWrite(base, 0, Pattern(blen, 41));
  // Some other GPU write this frame so the end-of-frame reset runs.
  g_acc->MarkGpuWritten(base.phys + base.size - 0x1000, 0x1000);
  g_acc->EndOfFramePageReset();
  Verify(name, d, base, nullptr);
}

// File-read style host write: no fault, callbacks after the write.
void HostWriteScenario(const char* name, const TexDesc& d, bool resident,
                       bool with_callbacks) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 50));
  if (resident) g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  HostWrite(base, 0, Pattern(blen, 51));
  if (with_callbacks) HostWriteCallbacks(base, 0, blen);
  Verify(name, d, base, nullptr);
}

// clear_memory_page_state racing a CPU write. The texture's pages hold GPU
// (resolve) data; a game thread's write faults while the GPU thread is between
// the copy of the GPU-written valid bits and the buffer exchange in
// SetSystemPageBlocksValidWithGpuDataWritten. If the invalidation lands in the
// retiring buffer, the page stays valid with the old shared memory bytes: the
// fired watch makes the texture reload them (stale), and with next_frame the
// following swap drops the page without a watch, so it stays stale.
void PageSwapRaceScenario(const char* name, const TexDesc& d, bool next_frame) {
  Region base = Alloc(256 * 1024);
  Region gpu = Alloc(64 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 70));
  g_acc->MakeResident(base.phys, blen);
  g_acc->MarkGpuWritten(base.phys, blen);
  g_acc->EndOfFramePageReset();
  g_acc->Materialize(f);
  g_acc->Flush();
  // This frame's GPU write lands in the same valid-bit block (the page after
  // the texture), so the swap copies the texture's pages too.
  g_acc->MarkGpuWritten(base.phys + blen, 0x1000);

  std::mutex m;
  std::condition_variable cv;
  bool done = false;
  std::thread writer;
  Injection inj{hooks::Point::kPageSwapBeforeExchange, 0};
  inj.action = [&] {
    writer = std::thread([&] {
      GuestWrite(base, 0, Pattern(blen, 71));
      std::lock_guard<std::mutex> l(m);
      done = true;
      cv.notify_all();
    });
    // Unlocked swap: the write completes here. Locked swap: it waits for the
    // swap, so give up after a while and let the swap finish.
    std::unique_lock<std::mutex> l(m);
    cv.wait_for(l, std::chrono::milliseconds(250), [&] { return done; });
  };
  g_injection = &inj;
  g_acc->EndOfFramePageReset();
  g_injection = nullptr;
  if (writer.joinable()) writer.join();
  if (!inj.fired) {
    std::printf("HARNESS scenario %s: page swap hook did not fire\n", name);
    ++g_harness_errors;
    return;
  }
  if (next_frame) {
    g_acc->Materialize(f);
    g_acc->Flush();
    g_acc->MarkGpuWritten(gpu.phys, 0x1000);
    g_acc->EndOfFramePageReset();
  }
  Verify(name, d, base, nullptr);
}

// Memexport into a range a texture was loaded from. The export is copied back
// to guest RAM by ApplyCompletedMemexportReadbacks after its submission
// completes, through the unwatched host view.
//   kAfterCompletion: draw only after the readback.
//   kCpuWritesThenDraw: the CPU then writes every page of the texture outside
//     the export bytes (pages invalid, export hash unchanged) and a draw loads
//     the texture before the readback (CPU source path: guest RAM still holds
//     the pre-export bytes); verified at that draw.
//   kCpuWritesThenDrawThenLater: same, verified at a later draw (the readback
//     is applied when completions are processed, at that later draw; nothing
//     tells the texture, so it keeps the pre-export bytes).
//   kCpuOverwritesExport: the CPU also overwrites the export bytes (the CPU
//     wins, the readback is skipped).
enum class Mx { kAfterCompletion, kCpuWritesThenDraw, kCpuWritesThenDrawThenLater, kCpuOverwritesExport };
void MemexportScenario(const char* name, const TexDesc& d, bool resident, Mx mode) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 80));
  if (resident) g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  const uint32_t xoff = 0x100, xlen = 0x200;
  if (!g_acc->Memexport(base.phys + xoff, Pattern(xlen, 81))) {
    std::printf("HARNESS scenario %s: memexport failed\n", name);
    ++g_harness_errors;
    return;
  }
  if (mode != Mx::kAfterCompletion) {
    for (uint32_t off = 0; off < blen; off += 0x1000) {
      const uint32_t o = off ? off : xoff + xlen;
      GuestWrite(base, o, Pattern(0x100, 82 + off));
    }
    if (mode == Mx::kCpuOverwritesExport) {
      GuestWrite(base, xoff, Pattern(xlen, 90));
    }
  }
  auto dbg = [&](const char* step) {
    if (!g_verbose) return;
    std::vector<uint8_t> g;
    ReadGuest(base, xoff, xlen, g);
    const uint8_t* b = static_cast<const uint8_t*>(
        g_acc->sm().GetBuffer()->contents()) + base.phys + xoff;
    // No texture readback here: it would flush and apply the export early.
    std::printf("  [%s] %s: guest export bytes %016" PRIX64
                ", buffer export bytes %016" PRIX64 ", loads %" PRIu64 "\n",
                name, step, XXH3_64bits(g.data(), xlen), XXH3_64bits(b, xlen),
                g_acc->LoadCount());
  };
  dbg("after cpu writes");
  if (mode == Mx::kAfterCompletion) {
    g_acc->Flush();
  } else if (mode == Mx::kCpuWritesThenDrawThenLater) {
    g_acc->Materialize(f);
    dbg("after draw");
    g_acc->Flush();
    dbg("after flush");
  }
  Verify(name, d, base, nullptr);
}

// GPU-ordered memexport (the BO2 streamed-mip writer) into a texture's range,
// with the readback to guest RAM landing at command buffer completion. The
// expected bytes are explicit: the seed, then the export, then CPU writes in
// program order (the guest waited for the export before writing).
//   kLoadThenExport: texture loaded, export in the same command buffer.
//   kExportThenResidentLoad: export, then a load (shared memory) before the
//     readback lands.
//   kExportThenCpuPageWriteCpuLoad: export, a CPU write elsewhere on every
//     page of the texture (pages invalid), a load from guest RAM before the
//     readback lands (CPU source), verified at a later draw.
//   kExportThenCpuPageWriteReupload: same, but the pages are re-uploaded to
//     the shared buffer from guest RAM before the readback lands (the upload
//     must not overwrite the export bytes).
//   kExportThenCpuPartial: a CPU write into part of the export bytes before
//     the readback lands (game waited on the GPU fence, which the CP signals
//     before the Metal command buffer completes).
//   kExportThenFilePartial: the same through a non-faulting host write
//     (file read) plus callbacks.
enum class Gx {
  kLoadThenExport,
  kExportThenResidentLoad,
  kExportThenCpuPageWriteCpuLoad,
  kExportThenCpuPageWriteReupload,
  kExportThenCpuPartial,
  kExportThenFilePartial,
};
void MemexportGpuScenario(const char* name, const TexDesc& d, Gx mode) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  std::vector<uint8_t> expected = Pattern(base.size, 110);
  Seed(base, 0, expected);
  g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  const uint32_t xoff = 0x1000, xlen = 0x800;
  const std::vector<uint8_t> x = Pattern(xlen, 111);
  auto apply = [&](uint32_t off, const std::vector<uint8_t>& v) {
    std::copy(v.begin(), v.end(), expected.begin() + off);
  };
  if (mode == Gx::kLoadThenExport) {
    // Make the texture outdated so this draw's load is encoded first.
    const std::vector<uint8_t> w = Pattern(0x100, 112);
    GuestWrite(base, 0, w);
    apply(0, w);
    g_acc->Materialize(f);
  }
  if (!g_acc->MemexportGpu(base.phys + xoff, x)) {
    std::printf("HARNESS scenario %s: memexport failed\n", name);
    ++g_harness_errors;
    return;
  }
  apply(xoff, x);
  switch (mode) {
    case Gx::kLoadThenExport:
      break;
    case Gx::kExportThenResidentLoad:
      g_acc->Materialize(f);
      break;
    case Gx::kExportThenCpuPageWriteCpuLoad:
    case Gx::kExportThenCpuPageWriteReupload:
      for (uint32_t off = 0; off < blen; off += 0x1000) {
        const uint32_t o = off == xoff ? xoff + xlen : off;
        const std::vector<uint8_t> w = Pattern(0x100, 113 + off);
        GuestWrite(base, o, w);
        apply(o, w);
      }
      if (mode == Gx::kExportThenCpuPageWriteReupload) {
        g_acc->MakeResident(base.phys, blen);
      }
      g_acc->Materialize(f);
      break;
    case Gx::kExportThenCpuPartial:
    case Gx::kExportThenFilePartial: {
      const std::vector<uint8_t> w = Pattern(0x40, 114);
      if (mode == Gx::kExportThenCpuPartial) {
        GuestWrite(base, xoff + 0x100, w);
      } else {
        HostWrite(base, xoff + 0x100, w);
        HostWriteCallbacks(base, xoff + 0x100, uint32_t(w.size()));
      }
      apply(xoff + 0x100, w);
      break;
    }
  }
  g_acc->Flush();  // the export's command buffer completes; readback lands
  g_acc->Materialize(f);
  g_acc->Flush();
  // Guest RAM must hold the export too (CPU readers and later uploads).
  std::vector<uint8_t> g;
  ReadGuest(base, 0, blen, g);
  if (!std::equal(g.begin(), g.end(), expected.begin())) {
    uint32_t first = 0, last = 0;
    while (first < blen && g[first] == expected[first]) ++first;
    for (uint32_t i = first; i < blen; ++i) {
      if (g[i] != expected[i]) last = i;
    }
    std::printf("FAIL scenario %s: guest RAM at 0x%08X lost the memexport "
                "(bytes 0x%X..0x%X differ from export-then-CPU order)\n",
                name, base.phys, first, last);
    ++g_failures;
  }
  g_expected_base = &expected;
  Verify(name, d, base, nullptr);
  g_expected_base = nullptr;
}

// CPU store racing ApplyCompletedMemexportReadbacks.
// The export completes, then a CPU write hits the export range while the
// readback is being applied. Without the global lock, the readback's
// hash/merge/memcpy can race with the CPU write, causing lost updates or
// corruption.
void MemexportRaceCpuStoreScenario(const char* name, const TexDesc& d) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 200));
  g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  const uint32_t xoff = 0x100, xlen = 0x200;
  if (!g_acc->Memexport(base.phys + xoff, Pattern(xlen, 201))) {
    std::printf("HARNESS scenario %s: memexport failed\n", name);
    ++g_harness_errors;
    return;
  }
  // CPU writes to the export range after export but before readback applies.
  // This races with ApplyCompletedMemexportReadbacks when it processes the
  // completion. The global lock should serialize them.
  GuestWrite(base, xoff, Pattern(xlen, 202));
  g_acc->Flush();  // completion processed, readback applies
  Verify(name, d, base, nullptr);
}

// Overlapping later readback rebased.
// Two memexports to overlapping ranges complete. The first applies and rebases
// the second's hash and snapshot. Without the snapshot refresh, the second
// readback would merge against stale snapshot data.
void MemexportRebaseOverlapScenario(const char* name, const TexDesc& d) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 210));
  g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  // First export: range [0x100, 0x300)
  const uint32_t xoff1 = 0x100, xlen1 = 0x200;
  const std::vector<uint8_t> x1 = Pattern(xlen1, 211);
  if (!g_acc->Memexport(base.phys + xoff1, x1)) {
    std::printf("HARNESS scenario %s: memexport1 failed\n", name);
    ++g_harness_errors;
    return;
  }
  // Second export: overlapping range [0x200, 0x400)
  const uint32_t xoff2 = 0x200, xlen2 = 0x200;
  const std::vector<uint8_t> x2 = Pattern(xlen2, 212);
  if (!g_acc->Memexport(base.phys + xoff2, x2)) {
    std::printf("HARNESS scenario %s: memexport2 failed\n", name);
    ++g_harness_errors;
    return;
  }
  // CPU writes to the overlapping region [0x200, 0x300) after both exports.
  GuestWrite(base, 0x200, Pattern(0x100, 213));
  g_acc->Flush();  // both completions processed, readbacks apply
  // The first readback applies, rebases the second's snapshot.
  // The second readback should merge correctly using the refreshed snapshot.
  Verify(name, d, base, nullptr);
}

// Stale memexport readback (write-trace run 3, 2026-11-10): the game's copy
// routine wrote texture B's mips into a reused pool block, then a memexport
// draw's readback copied another texture's bytes over them. Texture A is
// loaded from the range and resident; the CPU rewrites the whole texture
// with texture B through the guest (watched) view; a memexport draw declares
// an export range over the texture but writes only part of it; the readback
// lands. Guest RAM and the texture must hold B outside the written bytes.
//   kBeforeDrawPendingUpload: CPU write, then the draw (its range request
//     encodes the upload blit of B just before the export).
//   kBeforeDrawUploaded: CPU write, B uploaded and completed in an earlier
//     command buffer, then the draw.
//   kBeforeDrawNoUpload: CPU write, then the draw with no upload (prepared
//     while the range was valid; the execute path trusts that flag).
//   kDuringEncode: the CPU write lands after the draw's range request and
//     before the post-draw marking (RangeWrittenByGpu, readback snapshot).
//     A guest-order race: the written bytes may hold the export or B.
//   kUnwatchedAlias: the CPU write goes through a view with no watch (host
//     write, no callbacks): the buffer stays valid with A. Not a readback bug
//     (the write is never seen), but the readback turns it into a guest RAM
//     overwrite. Extra, not in the default list.
// part_page: the declared range starts and ends inside pages (0x180..0x1C80)
// instead of covering the texture's whole page span.
enum class Sr {
  kBeforeDrawPendingUpload,
  kBeforeDrawUploaded,
  kBeforeDrawNoUpload,
  kDuringEncode,
  kUnwatchedAlias,
};
void StaleReadbackScenario(const char* name, const TexDesc& d, Sr mode,
                           bool part_page) {
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  std::vector<uint8_t> expected = Pattern(base.size, 300);  // texture A
  Seed(base, 0, expected);
  g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  const uint32_t doff = part_page ? 0x180 : 0;
  const uint32_t dlen = part_page ? blen - 0x380 : blen;
  const uint32_t woff = 0x400, wlen = 0x400;  // the bytes the draw writes
  const std::vector<uint8_t> x = Pattern(wlen, 301);
  const std::vector<uint8_t> b = Pattern(blen, 302);  // texture B
  auto cpu_write_b = [&]() {
    if (mode == Sr::kUnwatchedAlias) {
      HostWrite(base, 0, b);
    } else {
      GuestWrite(base, 0, b);
    }
  };
  // Guest order: B, then the export (the GPU ran the draw after the CPU
  // wrote, in every mode).
  std::copy(b.begin(), b.end(), expected.begin());
  std::copy(x.begin(), x.end(), expected.begin() + doff + woff);
  bool request = true;
  std::function<void()> between;
  switch (mode) {
    case Sr::kBeforeDrawPendingUpload:
    case Sr::kUnwatchedAlias:
      cpu_write_b();
      break;
    case Sr::kBeforeDrawUploaded:
      cpu_write_b();
      g_acc->MakeResident(base.phys, blen);
      g_acc->Flush();
      break;
    case Sr::kBeforeDrawNoUpload:
      cpu_write_b();
      request = false;
      break;
    case Sr::kDuringEncode:
      between = cpu_write_b;
      break;
  }
  if (!g_acc->MemexportGpuPartial(base.phys + doff, dlen, woff, x, request,
                                  between)) {
    std::printf("HARNESS scenario %s: memexport failed\n", name);
    ++g_harness_errors;
    return;
  }
  g_acc->Flush();  // the export's command buffer completes
  g_acc->ProcessCompletions();  // the readback lands
  std::vector<uint8_t> g;
  ReadGuest(base, 0, blen, g);
  if (mode == Sr::kDuringEncode) {
    // The CPU write lands after the draw was issued (the command processor
    // only sees a draw after the guest kicked it), racing the GPU in guest
    // order too. The readback merge decides per byte (CPU-changed bytes keep
    // the CPU data), so each written byte may hold the export or B.
    // Everything outside the written bytes must hold B either way, and the
    // texture must match guest RAM.
    bool ok = true;
    uint32_t kept_cpu = 0;
    for (uint32_t i = 0; i < blen && ok; ++i) {
      const bool in_w = i >= doff + woff && i < doff + woff + wlen;
      if (!in_w) {
        ok = g[i] == expected[i];
      } else if (g[i] != expected[i]) {
        ok = g[i] == b[i];
        ++kept_cpu;
      }
    }
    if (ok) {
      std::copy(g.begin(), g.end(), expected.begin());
      if (g_verbose) {
        std::printf("note %s: %u written bytes kept the CPU write\n", name,
                    kept_cpu);
      }
    }
  }
  if (!std::equal(g.begin(), g.end(), expected.begin())) {
    uint32_t first = 0, last = 0, n = 0, a_bytes = 0;
    while (first < blen && g[first] == expected[first]) ++first;
    const std::vector<uint8_t> a = Pattern(base.size, 300);
    for (uint32_t i = first; i < blen; ++i) {
      if (g[i] != expected[i]) {
        last = i;
        ++n;
        if (g[i] == a[i]) ++a_bytes;
      }
    }
    std::printf("FAIL scenario %s: guest RAM at 0x%08X holds stale bytes after "
                "the readback (0x%X..0x%X, %u bytes differ from B+export, %u "
                "of them equal texture A; declared 0x%X+0x%X, written 0x%X+0x%X)\n",
                name, base.phys, first, last, n, a_bytes, doff, dlen,
                doff + woff, wlen);
    ++g_failures;
  }
  g_expected_base = &expected;
  Verify(name, d, base, nullptr);
  g_expected_base = nullptr;
}

// Stale copy SOURCE (manager follow-up, 2026-11-10): a memexport copy draw
// (the game's texture MOVE) reads a source block whose guest bytes the CPU
// rewrote (S_old -> S_new) but which was never uploaded, and writes it into a
// valid, up-to-date destination holding loaded texture A. The draw was
// prepared before the CPU write (prepare-time has_invalid_shared_memory
// false). Guest RAM and the texture at the destination must hold S_new.
//   partial: the copy writes 0x800 bytes at +0x400 of an 8 KB declared range.
//   pending: control, prepared after the CPU write (the request uploads S_new).
// A resolve is not a case: its source is EDRAM, not the shared buffer.
void StaleSourceScenario(const char* name, const TexDesc& d, bool partial,
                         bool pending) {
  // A 512 KB spacer: a non-exact CPU-write invalidation widens up to its
  // 256 KB page block, and must not reach the destination (in the game the
  // source and destination pool blocks need not be neighbours).
  Region src = Alloc(256 * 1024);
  (void)Alloc(512 * 1024);
  Region dst = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, dst.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  std::vector<uint8_t> expected = Pattern(dst.size, 400);  // texture A
  Seed(dst, 0, expected);
  const std::vector<uint8_t> s_old = Pattern(blen, 401);
  Seed(src, 0, s_old);
  g_acc->MakeResident(dst.phys, blen);
  g_acc->MakeResident(src.phys, blen);  // the source is valid with S_old
  g_acc->Materialize(f);
  g_acc->Flush();
  const std::vector<uint8_t> s_new = Pattern(blen, 402);
  GuestWrite(src, 0, s_new);  // invalidates the source pages, not uploaded
  const uint32_t woff = partial ? 0x400 : 0;
  const uint32_t clen = partial ? 0x800 : blen;
  std::copy(s_new.begin(), s_new.begin() + clen, expected.begin() + woff);
  if (!g_acc->MemexportGpuCopy(src.phys, clen, dst.phys, blen, woff, pending)) {
    std::printf("HARNESS scenario %s: memexport copy failed\n", name);
    ++g_harness_errors;
    return;
  }
  g_acc->Flush();
  g_acc->ProcessCompletions();
  std::vector<uint8_t> g;
  ReadGuest(dst, 0, blen, g);
  if (!std::equal(g.begin(), g.end(), expected.begin())) {
    uint32_t first = 0, last = 0, n = 0, old_bytes = 0;
    while (first < blen && g[first] == expected[first]) ++first;
    for (uint32_t i = first; i < blen; ++i) {
      if (g[i] != expected[i]) {
        last = i;
        ++n;
        if (i >= woff && i < woff + clen && g[i] == s_old[i - woff]) {
          ++old_bytes;
        }
      }
    }
    std::printf("FAIL scenario %s: destination guest RAM at 0x%08X holds the "
                "stale source (0x%X..0x%X, %u bytes differ, %u of them equal "
                "the old source bytes)\n",
                name, dst.phys, first, last, n, old_bytes);
    ++g_failures;
  }
  g_expected_base = &expected;
  Verify(name, d, dst, nullptr);
  g_expected_base = nullptr;
}

// A 1x resolve (render target copy) overlapping a loaded texture's memory,
// then the texture is sampled. The resolved bytes are mirrored into guest RAM
// so the guest stays the oracle (the reference load sees the same bytes).
//   after 0: draw next. 1: an end-of-frame page reset first.
//   2: draw, then a CPU write into a resolved page, a page reset, then draw.
void ResolveAliasScenario(const char* name, const TexDesc& d, bool resident,
                          uint32_t roff, uint32_t rlen, bool rtt, int after) {
  Region base = Alloc(256 * 1024);
  Region gpu = Alloc(64 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 100));
  if (resident) g_acc->MakeResident(base.phys, blen);
  g_acc->Materialize(f);
  g_acc->Flush();
  if (!rlen) rlen = blen - roff;
  const std::vector<uint8_t> r = Pattern(rlen, 101);
  HostWrite(base, roff, r);
  if (!g_acc->Resolve(base.phys + roff, r, rtt)) {
    std::printf("HARNESS scenario %s: resolve failed\n", name);
    ++g_harness_errors;
    return;
  }
  if (after >= 1) g_acc->EndOfFramePageReset();
  if (after == 2) {
    g_acc->Materialize(f);
    g_acc->Flush();
    GuestWrite(base, roff + 0x40, Pattern(0x80, 102));
    g_acc->MarkGpuWritten(gpu.phys, 0x1000);
    g_acc->EndOfFramePageReset();
  }
  Verify(name, d, base, nullptr);
}

#ifdef REX_METAL_3D_AS_2D_REFRESH
// A 3D texture sampled as 2D is served from a 2D wrapper of its first slice
// that lives outside the texture map. After a CPU write the next draw must
// see the new data in the wrapper, not only in the 3D texture.
bool ReadWrapper(const char* name, const xe_gpu_texture_fetch_t& f,
                 std::vector<std::vector<uint8_t>>& levels) {
  std::string err;
  auto* w = g_acc->Wrapper3DAs2D(f);
  if (!w || !g_acc->Readback(w, levels, err)) {
    std::printf("HARNESS scenario %s: 3D-as-2D wrapper readback failed: %s\n",
                name, w ? err.c_str() : "no wrapper");
    ++g_harness_errors;
    return false;
  }
  return true;
}

void ThreeDAs2DScenario(const char* name) {
  const TexDesc d = {xenos::TextureFormat::k_8_8_8_8, 32, 32, true, 0, false,
                     "rgba8 32x32x4 tiled 3d", xenos::DataDimension::k3D, 4};
  Region base = Alloc(256 * 1024);
  xe_gpu_texture_fetch_t f = MakeFetch(d, base.phys, 0);
  Extents e = GetExtents(f);
  const uint32_t blen = (e.base + 0xFFF) & ~0xFFFu;
  Seed(base, 0, Pattern(base.size, 60));
  std::vector<std::vector<uint8_t>> served, ref;
  // Draw 1 samples it as 2D: the wrapper is made and loaded.
  if (!g_acc->Materialize(f) || !ReadWrapper(name, f, served)) return;
  // CPU write over the whole texture, then draw 2 samples it as 2D again.
  GuestWrite(base, 0, Pattern(blen, 61));
  if (!g_acc->Materialize(f) || !ReadWrapper(name, f, served)) return;
  // Reference: a fresh texture at a twin region with the same bytes.
  Region tb = Alloc(base.size);
  std::vector<uint8_t> bytes;
  ReadGuest(base, 0, base.size, bytes);
  Seed(tb, 0, bytes);
  xe_gpu_texture_fetch_t rf = MakeFetch(d, tb.phys, 0);
  if (!g_acc->Materialize(rf) || !ReadWrapper(name, rf, ref)) return;
  const uint64_t hs = HashLevels(served), hr = HashLevels(ref);
  if (hs != hr) {
    std::printf("FAIL scenario %s: 3D texture at guest 0x%08X sampled as 2D "
                "served stale bytes from its 2D wrapper (expected hash "
                "%016" PRIX64 ", got %016" PRIX64 "; base %u B)\n",
                name, base.phys, hr, hs, e.base);
    ++g_failures;
  } else if (g_verbose) {
    std::printf("  ok %s (hash %016" PRIX64 ")\n", name, hs);
  }
}
#endif

struct ScenarioEntry {
  const char* name;
  std::function<void(const char*)> run;
};

}  // namespace

int main(int argc, char** argv) {
  std::string only;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-v")) g_verbose = true;
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
  }
  const char* fake_argv[] = {"rexgpu-metal-texcache-stress"};
  rex::cvar::Init(1, const_cast<char**>(fake_argv));
  rex::cvar::ApplyEnvironment();
  rex::cvar::FinalizeInit();
  rex::InitLogging({}, g_verbose ? spdlog::level::info : spdlog::level::warn);
  ApplyLegacyMetalEnv();

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
  // The texture cache and shared memory are created by SetupContext on the
  // GPU worker thread; queued functions run only after it, so wait for one.
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
  TexCacheStressAccess acc(cp, gs->register_file());
  g_acc = &acc;
  hooks::g_hook = HookFn;

  using P = hooks::Point;
  std::vector<ScenarioEntry> scenarios = {
      // 1. CPU-source path (texture data not resident in shared memory).
      {"cpu_write_between_copy_and_watch",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kCpuSource, P::kCpuSourceAfterCopy, false, false); }},
      {"cpu_write_between_watch_and_copy",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kCpuSource, P::kCpuSourceBeforeCopy, false, false); }},
      {"cpu_mips_tail_write_during_copy",
       [](const char* n) { RaceScenario(n, kDxt5_256m, Path::kCpuSource, P::kCpuSourceAfterCopy, false, false, true); }},
      // 2. Resident path (data already valid in the shared memory buffer).
      {"resident_write_between_copy_and_watch",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kResident, P::kResidentAfterCopy, false, false); }},
      {"resident_write_before_copy",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kResident, P::kResidentBeforeCopy, false, false); }},
      {"resident_mips_tail_write_during_copy",
       [](const char* n) { RaceScenario(n, kDxt5_256m, Path::kResident, P::kResidentAfterCopy, false, false, true); }},
      // 3. Streaming pool recycling.
      {"reuse_cpu", [](const char* n) { ReuseScenario(n, false, false); }},
      {"reuse_resident", [](const char* n) { ReuseScenario(n, true, false); }},
      {"reuse_cpu_race", [](const char* n) { ReuseScenario(n, false, true); }},
      {"reuse_resident_race", [](const char* n) { ReuseScenario(n, true, true); }},
      // 4. Streamed mip upgrade / packed mip tail.
      {"mipstream_cpu", [](const char* n) { MipStreamScenario(n, kDxt5_256m, false, false); }},
      {"mipstream_resident", [](const char* n) { MipStreamScenario(n, kDxt5_256m, true, false); }},
      {"mipstream_cpu_tail_race", [](const char* n) { MipStreamScenario(n, kDxt1_128m, false, true); }},
      {"mipstream_resident_tail_race", [](const char* n) { MipStreamScenario(n, kDxt1_128m, true, true); }},
      // 5. clear_memory_page_state.
      {"page_reset_then_write", [](const char* n) { PageResetScenario(n, kDxt1_128, 1, false); }},
      {"page_reset_x3_reload_then_write", [](const char* n) { PageResetScenario(n, kDxt1_128, 3, true); }},
      {"resolve_then_cpu_overwrite", [](const char* n) { ResolveThenCpuScenario(n, kRgba64); }},
      // 6. Host writes that don't fault.
      {"hostwrite_callbacks_resident", [](const char* n) { HostWriteScenario(n, kDxt1_128, true, true); }},
      {"hostwrite_callbacks_cpu", [](const char* n) { HostWriteScenario(n, kDxt1_128, false, true); }},
      {"hostwrite_in_cpu_copy_window_callbacks_inside",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kCpuSource, P::kCpuSourceAfterCopy, true, false); }},
      {"hostwrite_in_cpu_copy_window_callbacks_after_load",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kCpuSource, P::kCpuSourceAfterCopy, true, true); }},
      {"hostwrite_in_resident_window_callbacks_inside",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kResident, P::kResidentAfterCopy, true, false); }},
      {"hostwrite_in_resident_window_callbacks_after_load",
       [](const char* n) { RaceScenario(n, kDxt1_128, Path::kResident, P::kResidentAfterCopy, true, true); }},
      // 7. clear_memory_page_state swap racing a CPU write (REX_METAL_PAGE_SWAP_LOCK).
      {"pageswap_write_mid_swap", [](const char* n) { PageSwapRaceScenario(n, kDxt1_128, false); }},
      {"pageswap_write_mid_swap_next_frame", [](const char* n) { PageSwapRaceScenario(n, kDxt1_128, true); }},
      {"pageswap_write_mid_swap_rgba", [](const char* n) { PageSwapRaceScenario(n, kRgba64, false); }},
      // 8. Memexport readback into a texture's range.
      {"memexport_into_loaded_resident", [](const char* n) { MemexportScenario(n, kDxt1_128, true, Mx::kAfterCompletion); }},
      {"memexport_into_loaded_cpu", [](const char* n) { MemexportScenario(n, kDxt1_128, false, Mx::kAfterCompletion); }},
      {"memexport_cpu_writes_then_draw", [](const char* n) { MemexportScenario(n, kDxt1_128, true, Mx::kCpuWritesThenDraw); }},
      {"memexport_cpu_writes_then_draw_later", [](const char* n) { MemexportScenario(n, kDxt1_128, true, Mx::kCpuWritesThenDrawThenLater); }},
      {"memexport_cpu_writes_then_draw_cpu_loaded", [](const char* n) { MemexportScenario(n, kRgba64, false, Mx::kCpuWritesThenDrawThenLater); }},
      {"memexport_cpu_overwrites_export", [](const char* n) { MemexportScenario(n, kDxt1_128, true, Mx::kCpuOverwritesExport); }},
      // 8b. GPU-ordered memexport (orderings a to d from the noise counts).
      // 8c. New deterministic scenarios for the readback fix.
      {"memexport_race_cpu_store", [](const char* n) { MemexportRaceCpuStoreScenario(n, kDxt1_128); }},
      {"memexport_rebase_overlap", [](const char* n) { MemexportRebaseOverlapScenario(n, kDxt1_128); }},
      {"mx_gpu_load_then_export", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kLoadThenExport); }},
      {"mx_gpu_export_then_resident_load", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kExportThenResidentLoad); }},
      {"mx_gpu_export_then_cpu_page_write_cpu_load", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kExportThenCpuPageWriteCpuLoad); }},
      {"mx_gpu_export_then_cpu_page_write_reupload", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kExportThenCpuPageWriteReupload); }},
      {"mx_gpu_export_then_cpu_partial_write", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kExportThenCpuPartial); }},
      {"mx_gpu_export_then_file_partial_write", [](const char* n) { MemexportGpuScenario(n, kDxt1_128, Gx::kExportThenFilePartial); }},
      // 8d. Stale memexport readback: CPU rewrites a texture, then a draw
      // exports into part of its declared range (write-trace run 3).
      {"stale_readback_pending_upload", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kBeforeDrawPendingUpload, false); }},
      {"stale_readback_uploaded", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kBeforeDrawUploaded, false); }},
      {"stale_readback_no_upload", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kBeforeDrawNoUpload, false); }},
      {"stale_readback_during_encode", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kDuringEncode, false); }},
      {"stale_readback_part_page_pending_upload", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kBeforeDrawPendingUpload, true); }},
      {"stale_readback_part_page_no_upload", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kBeforeDrawNoUpload, true); }},
      {"stale_readback_part_page_during_encode", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kDuringEncode, true); }},
      // 8e. Stale copy source: a memexport copy reads a CPU-rewritten,
      // not-uploaded source into a valid destination.
      {"stale_source_copy_full", [](const char* n) { StaleSourceScenario(n, kDxt1_128, false, false); }},
      {"stale_source_copy_partial", [](const char* n) { StaleSourceScenario(n, kDxt1_128, true, false); }},
      {"stale_source_copy_pending", [](const char* n) { StaleSourceScenario(n, kDxt1_128, false, true); }},
      // 9. Resolve (render target copy) overlapping a loaded texture.
      {"resolve_alias_full_rtt", [](const char* n) { ResolveAliasScenario(n, kRgba64, true, 0, 0, true, 0); }},
      {"resolve_alias_full_base", [](const char* n) { ResolveAliasScenario(n, kRgba64, true, 0, 0, false, 0); }},
      {"resolve_alias_full_cpu_loaded", [](const char* n) { ResolveAliasScenario(n, kRgba64, false, 0, 0, true, 0); }},
      {"resolve_alias_partial_dxt1", [](const char* n) { ResolveAliasScenario(n, kDxt1_128, true, 0x1000, 0, true, 0); }},
      {"resolve_alias_then_page_reset", [](const char* n) { ResolveAliasScenario(n, kRgba64, true, 0, 0, true, 1); }},
      {"resolve_alias_then_cpu_partial", [](const char* n) { ResolveAliasScenario(n, kDxt1_128, true, 0, 0x1000, true, 2); }},
#ifdef REX_METAL_3D_AS_2D_REFRESH
      // 7. A 3D texture sampled as 2D (needs the metal_3d_as_2d_refresh fix).
      {"3d_as_2d_after_cpu_write", [](const char* n) { ThreeDAs2DScenario(n); }},
#endif
  };

  // Extras: run only by name (--only), not part of the pass/fail gate.
  std::vector<ScenarioEntry> extras = {
      {"stale_readback_unwatched_alias", [](const char* n) { StaleReadbackScenario(n, kDxt1_128, Sr::kUnwatchedAlias, false); }},
  };
  if (!only.empty()) {
    for (auto& x : extras) scenarios.push_back(x);
  }
  int ran = 0;
  for (auto& s : scenarios) {
    if (!only.empty() && only != s.name) continue;
    ++ran;
    const int fails_before = g_failures, errs_before = g_harness_errors;
    rex::thread::Fence fence;
    cp->CallInThread([&]() {
      s.run(s.name);
      fence.Signal();
    });
    fence.Wait();
    std::printf("%s %s\n",
                g_failures != fails_before      ? "STALE"
                : g_harness_errors != errs_before ? "ERROR"
                                                  : "pass ",
                s.name);
    std::fflush(stdout);
  }
  hooks::g_hook = nullptr;
  std::printf("texcache-stress: %d scenarios, %d stale serves, %d harness errors\n",
              ran, g_failures, g_harness_errors);
  std::fflush(stdout);
  // Same as the gputest replay tool: skip static destructors of the GPU/runtime.
  std::_Exit(g_failures ? 1 : (g_harness_errors || !ran) ? 2 : 0);
}
