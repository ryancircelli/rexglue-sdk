// refxenos: pure-CPU reference Xenos renderer.
//
// Accuracy over speed. Every behaviour that is not plain arithmetic is a
// named Rule (see RX_DEFINE_RULE below) carrying its source:
//   "xenia-canary@cf206a3b src/xenia/gpu/<file>:<line>" (pinned commit
//       cf206a3bdc4ffdfb22dbcfda99f777ab8d286f1d, checkout in
//       /tmp/bo2/refxenos/xenia, sparse src/xenia/gpu + src/xenia/base),
//   "sdk@<sha> <path>:<line>" for ReXGlue code (Xenia-derived),
//   "XDK <page>.htm" for the XDK documentation pages,
//   "AMD <doc> <section>" for R5xx/R6xx documents,
//   "assumed: <why>" when nothing better exists.
// Assumed rules are the real-hardware test list (research/reference-renderer.md).
// Rule versions feed the golden store: bump the version whenever the rule's
// behaviour changes, so only goldens that used it are recomputed.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>

namespace refxenos {

namespace xenos = rex::graphics::xenos;
namespace reg = rex::graphics::reg;
using rex::graphics::RegisterFile;

// ---------------------------------------------------------------------------
// Rules

struct Rule {
  const char* id;
  int version;
  const char* source;  // citation, see header comment
  bool assumed;
  const char* hw_test;  // for assumed rules: the homebrew test that settles it
};

class RuleRegistry {
 public:
  static RuleRegistry& Get();
  void Register(const Rule* rule) { rules_[rule->id] = rule; }
  const std::map<std::string, const Rule*>& rules() const { return rules_; }

 private:
  std::map<std::string, const Rule*> rules_;
};

struct RuleRegistrar {
  explicit RuleRegistrar(const Rule* rule) { RuleRegistry::Get().Register(rule); }
};

// Rules used by the current unit of work (a draw, a resolve, a shader case).
// Thread local so batch modes can run units in parallel.
class RuleUsage {
 public:
  static RuleUsage& Current();
  void Use(const Rule* rule) { used_.insert(rule); }
  void Clear() { used_.clear(); }
  const std::set<const Rule*>& used() const { return used_; }

 private:
  std::set<const Rule*> used_;
};

#define RX_DEFINE_RULE(var, ver, src, assumed, hwtest)                \
  const ::refxenos::Rule var{#var, ver, src, assumed, hwtest};        \
  static const ::refxenos::RuleRegistrar var##_registrar_(&var)
#define RX_DECLARE_RULE(var) extern const ::refxenos::Rule var
#define RX_USE(var) ::refxenos::RuleUsage::Current().Use(&(var))

// ---------------------------------------------------------------------------
// Guest physical memory (512 MiB, contents in guest byte order, i.e. as the
// GPU sees them before any endian swap).

class Memory {
 public:
  static constexpr uint32_t kSize = 512u << 20;
  Memory() : data_(kSize, 0) {}
  uint8_t* At(uint32_t physical) { return data_.data() + (physical & (kSize - 1)); }
  const uint8_t* At(uint32_t physical) const { return data_.data() + (physical & (kSize - 1)); }
  bool Read(uint32_t physical, void* dst, uint32_t len) const {
    physical &= kSize - 1;
    if (uint64_t(physical) + len > kSize) return false;
    std::memcpy(dst, data_.data() + physical, len);
    return true;
  }
  void Write(uint32_t physical, const void* src, uint32_t len) {
    physical &= kSize - 1;
    if (uint64_t(physical) + len > kSize) len = kSize - physical;
    std::memcpy(data_.data() + physical, src, len);
  }
  uint32_t ReadU32BE(uint32_t physical) const {
    const uint8_t* p = At(physical);
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  }

 private:
  std::vector<uint8_t> data_;
};

// ---------------------------------------------------------------------------
// Pixel probe: dump intermediate values for chosen pixels.
// Stages append free-form "key: values" lines; the tool prints them per draw.

class Probe {
 public:
  static Probe& Get();
  void AddPixel(int32_t x, int32_t y) { pixels_.insert({x, y}); }
  bool Wants(int32_t x, int32_t y) const {
    return !pixels_.empty() && pixels_.count({x, y}) != 0;
  }
  bool any() const { return !pixels_.empty(); }
  void Log(int32_t x, int32_t y, const std::string& stage, const std::string& text);
  void LogVertex(uint32_t vertex, const std::string& stage, const std::string& text);
  std::vector<std::string>& lines() { return lines_; }
  bool log_vertices = false;

 private:
  std::set<std::pair<int32_t, int32_t>> pixels_;
  std::vector<std::string> lines_;
};

std::string FormatFloat4(const float* v);

// ---------------------------------------------------------------------------
// Pipeline data passed between stages.

constexpr uint32_t kMaxInterpolators = 16;
constexpr uint32_t kMaxSamples = 4;

// Vertex shader output, one per vertex (exports, not yet clipped/transformed).
struct VertexOut {
  float position[4] = {};      // oPos (clip space as exported)
  float interp[kMaxInterpolators][4] = {};  // o0..o15
  float point_size_edge_kill[4] = {};       // oPts: x point size, y edge flag, z kill vertex
  bool position_written = false;
  bool point_written = false;
  uint32_t interp_written_mask = 0;
  uint32_t vertex_index = 0;  // index value fetched (for probe/diagnostics)
};

// A 2x2 pixel quad produced by the rasterizer for one primitive.
// Pixels: 0 = (x, y), 1 = (x+1, y), 2 = (x, y+1), 3 = (x+1, y+1).
struct PixelQuad {
  int32_t x = 0, y = 0;  // top-left pixel, both even
  // Per pixel: bit s set if sample s is covered (after scissor). A pixel with
  // a zero mask is a helper pixel: the shader runs for gradients, the ROP
  // ignores it.
  uint8_t coverage[4] = {};
  // Per pixel per sample depth after viewport transform and polygon offset,
  // before the depth format conversion (the ROP rounds to D24 / D24FS8).
  float sample_z[4][kMaxSamples] = {};
  // Interpolated PS inputs r0..r15 per pixel (as the shader sees them).
  float interp[4][kMaxInterpolators][4] = {};
  // ps_param_gen register value per pixel (if enabled), already encoded.
  float param_gen[4][4] = {};
  bool front_face = true;
  uint32_t primitive_index = 0;
};

// Pixel shader results for one pixel.
struct PixelOut {
  float color[4][4] = {};  // oC0..oC3
  uint32_t color_written_mask = 0;  // bit n: oCn exported
  float depth = 0.0f;
  bool depth_written = false;
  bool killed = false;
};

// Per-unit statistics written to draws.jsonl.
struct DrawStats {
  uint64_t vertices_shaded = 0;
  uint64_t primitives_in = 0;
  uint64_t primitives_culled = 0;
  uint64_t primitives_clipped = 0;
  uint64_t quads = 0;
  uint64_t pixels_shaded = 0;
  uint64_t pixels_killed = 0;
  uint64_t samples_alpha_test_failed = 0;
  uint64_t samples_depth_failed = 0;
  uint64_t samples_stencil_failed = 0;
  uint64_t samples_written = 0;
  std::vector<std::string> notes;  // unsupported paths hit, etc.
};

// Float helpers on raw bits.
inline uint32_t AsU32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float AsF32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace refxenos
