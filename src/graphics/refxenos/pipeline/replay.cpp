// Trace -> PM4 -> vertex interpreter -> raster -> pixel interpreter -> ROP.
// Selected draws may be isolated for shader/raster diagnosis. Such runs carry
// an explicit missing-prior-EDRAM note; they are not whole-frame goldens.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "core/raster.h"
#if __has_include("core/rop.h") && __has_include("core/resolve.h")
#define REFXENOS_HAS_ROP 1
#include "core/rop.h"
#include "core/resolve.h"
#else
#define REFXENOS_HAS_ROP 0
#include "core/edram.h"
#endif
#include "cp/command_processor.h"
#include "output/replay_writer.h"
#include "pipeline/tex_rules_bridge.h"
#include "shader/interp.h"
#include "texture/ref_sample.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace refxenos {
namespace {
using namespace rex::graphics;

class Sampler final : public TextureSampler, public tex::GuestMemory {
 public:
  Sampler(const RegisterFile& regs, const Memory& memory, DrawStats& stats)
      : regs_(regs), memory_(memory), stats_(stats) {}
  bool ok() const { return ok_; }
  bool Read(uint32_t address, void* dst, uint32_t count) const override {
    return memory_.Read(address, dst, count);
  }
  void Fetch(const TexFetchRequest& r, float out[4], std::vector<std::string>* used) override {
    tex::SampleRequest q;
    switch (TexFetchOp(r.op)) {
      case TexFetchOp::kTextureFetch:
        q.op = tex::SampleOp::kTfetch;
        break;
      case TexFetchOp::kGetComputedLod:
        q.op = tex::SampleOp::kGetCompTexLOD;
        break;
      case TexFetchOp::kGetWeights:
        q.op = tex::SampleOp::kGetWeights;
        break;
      case TexFetchOp::kGetBorderColorFrac:
        q.op = tex::SampleOp::kGetBCF;
        break;
      default:
        Note("unsupported texture opcode " + std::to_string(r.op));
        std::fill_n(out, 4, 0.0f);
        return;
    }
    if (r.fetch_constant >= 32) {
      Note("texture constant outside 0..31");
      std::fill_n(out, 4, 0.0f);
      return;
    }
    std::copy_n(&regs_.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + r.fetch_constant * 6], 6,
                q.fetch);
    q.dimension = tex::Dimension(r.dimension);
    std::copy_n(r.coord, 3, q.coord);
    if (q.dimension == tex::kDimCube) {
      // Shader cube ALU emits SC/TC in [1,2), sampler API wants [0,1).
      q.coord[0] -= 1.0f;
      q.coord[1] -= 1.0f;
      q.cube_face = uint32_t(std::clamp(r.coord[2], 0.0f, 5.0f));
    }
    std::copy_n(r.offset, 3, q.offset);
    std::copy_n(r.grad_h, 3, q.grad_h);
    std::copy_n(r.grad_v, 3, q.grad_v);
    q.has_grads = r.has_grads;
    q.unnormalized = r.unnormalized_coordinates;
    q.use_computed_lod = r.use_computed_lod;
    q.use_register_lod = r.use_register_lod;
    q.register_lod = r.lod;
    q.instr_lod_bias = r.lod_bias_instr;
    q.mag_filter = r.mag_filter;
    q.min_filter = r.min_filter;
    q.mip_filter = r.mip_filter;
    q.vol_mag_filter = r.vol_mag_filter;
    q.vol_min_filter = r.vol_min_filter;
    q.aniso_filter = r.aniso_filter;
    tex::Diag diag;
    auto value = tex::Sample(q, *this, &diag);
    std::copy(value.begin(), value.end(), out);
    UseTexRules(diag.rules_used, &stats_);
    if (used)
      used->insert(used->end(), diag.rules_used.begin(), diag.rules_used.end());
    if (!diag.error.empty())
      Note("texture: " + diag.error);
  }

 private:
  void Note(const std::string& s) {
    ok_ = false;
    if (notes_.insert(s).second)
      stats_.notes.push_back(s);
  }
  const RegisterFile& regs_;
  const Memory& memory_;
  DrawStats& stats_;
  std::set<std::string> notes_;
  bool ok_ = true;
};

DrawRecordInfo Record(const DrawCall& d) {
  DrawRecordInfo r;
  r.draw = d.draw_index;
  r.frame = d.frame;
  r.packet = d.packet_index;
  r.packet_base = d.packet_base;
  r.prim_type = uint32_t(d.prim);
  r.source_select = uint32_t(d.source_select);
  r.index_count = d.num_indices;
  r.indexed = d.indexed;
  r.index_format = uint32_t(d.index_format);
  r.index_base = d.index_base;
  r.index_bytes = d.index_bytes;
  r.is_resolve = d.is_resolve;
  r.edram_mode = uint32_t(d.regs->Get<reg::RB_MODECONTROL>().edram_mode);
  if (d.vs)
    r.vs_hash = d.vs->hash;
  if (d.ps)
    r.ps_hash = d.ps->hash;
  return r;
}

void SavePng(const std::filesystem::path& path, uint32_t w, uint32_t h,
             const std::vector<float>& rgba) {
  std::vector<uint8_t> bytes(rgba.size());
  for (size_t i = 0; i < rgba.size(); ++i) {
    float x = rgba[i];
    if (!(x >= 0))
      x = 0;
    bytes[i] = uint8_t(std::clamp(x, 0.0f, 1.0f) * 255.0f + 0.5f);
  }
  if (!stbi_write_png(path.c_str(), int(w), int(h), 4, bytes.data(), int(w * 4)))
    throw std::runtime_error("cannot write " + path.string());
}

void Usage() {
  std::fprintf(
      stderr,
      "usage: refxenos replay TRACE -o DIR [--frame N] [--frames N] "
      "[--draws FIRST:COUNT] [--ps HEX] [--list] [--dump-shaders] [--rop shader|edram] [--full] "
      "[--probe X,Y] [--probe-out FILE]\n"
      "--rop shader is a diagnostic overwrite stub (no depth/blend/alpha test); "
      "edram executes the ROP. Filters skip prior drawing; initial EDRAM is zero.\n");
}
}  // namespace

int RunReplay(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
      Usage();
      return argc < 2 ? 2 : 0;
    }
    std::string trace_path = argv[1], out_path, probe_out;
    uint32_t frame = 0, frames = 0, first = 0, count = 0;
    bool frame_filter = false, list = false, shader_rop = true, full = false, dump_shaders = false;
    uint64_t ps_hash = 0;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&] {
        if (i + 1 >= argc)
          throw std::runtime_error("missing value for " + a);
        return std::string(argv[++i]);
      };
      if (a == "-o" || a == "--out")
        out_path = next();
      else if (a == "--frame") {
        frame = std::stoul(next());
        frame_filter = true;
      } else if (a == "--frames")
        frames = std::stoul(next());
      else if (a == "--ps")
        ps_hash = std::stoull(next(), nullptr, 16);
      else if (a == "--list")
        list = true;
      else if (a == "--dump-shaders")
        dump_shaders = true;
      else if (a == "--full")
        full = true;
      else if (a == "--rop") {
        auto v = next();
        if (v != "shader" && v != "edram")
          throw std::runtime_error("invalid ROP");
        shader_rop = v == "shader";
      } else if (a == "--draws") {
        auto v = next();
        auto c = v.find(':');
        first = std::stoul(v.substr(0, c));
        count = c == v.npos ? 1 : std::stoul(v.substr(c + 1));
        if (!count)
          throw std::runtime_error("draw count must be positive");
      } else if (a == "--probe") {
        auto v = next();
        auto c = v.find(',');
        if (c == v.npos)
          throw std::runtime_error("probe needs X,Y");
        Probe::Get().AddPixel(std::stoi(v.substr(0, c)), std::stoi(v.substr(c + 1)));
      } else if (a == "--probe-out")
        probe_out = next();
      else {
        Usage();
        return 2;
      }
    }
    if (out_path.empty()) {
      Usage();
      return 2;
    }
    TraceReader trace;
    if (!trace.Open(trace_path))
      throw std::runtime_error(trace.error());
#if !REFXENOS_HAS_ROP
    if (!shader_rop)
      throw std::runtime_error("EDRAM ROP module is not available in this checkout");
#endif
    Memory memory;
    Edram edram;
    CommandProcessor cp(memory);
    if (frame_filter)
      cp.SetStartFrame(frame);
    ReplayWriter writer;
    if (!writer.Open(out_path, trace_path, shader_rop ? "refxenos-shader-stub" : "refxenos",
                     trace.title_id(), full))
      throw std::runtime_error("cannot open output");
    std::filesystem::path dir(out_path);
    bool all_ok = true;
    uint32_t selected = 0;
    std::set<uint64_t> dumped;
    RuleUsage::Current().Clear();
    bool ok = cp.Run(
        trace,
        [&](const DrawCall& d) {
          if (frame_filter && d.frame > frame)
            return false;
          if (frames && d.frame >= frames)
            return false;
          if (count && uint64_t(d.draw_index) >= uint64_t(first) + count)
            return false;
          if (count && d.draw_index < first)
            return true;
          if (ps_hash && (!d.ps || d.ps->hash != ps_hash))
            return true;
          ++selected;
          DrawStats stats;
          auto info = Record(d);
          const auto& regs = *d.regs;
          if (dump_shaders) {
            std::filesystem::create_directories(dir / "shaders");
            for (const auto& blob : {d.vs, d.ps}) {
              if (!blob || !dumped.insert(blob->hash).second)
                continue;
              auto stem = std::format(
                  "{}_{:016X}", blob->type == xenos::ShaderType::kPixel ? "ps" : "vs", blob->hash);
              std::ofstream bin(dir / "shaders" / (stem + ".bin"), std::ios::binary);
              bin.write(reinterpret_cast<const char*>(blob->ucode.data()), blob->ucode.size() * 4);
              std::vector<uint32_t> be = blob->ucode;
              for (auto& dw : be)
                dw = __builtin_bswap32(dw);
              std::ofstream hashes(dir / "shaders" / (stem + ".hashes"));
              hashes << std::format(
                  "be_xxh3 {:016X} le_xxh3 {:016X} be_xxh64 {:016X} le_xxh64 {:016X}\n", blob->hash,
                  XXH3_64bits(blob->ucode.data(), blob->ucode.size() * 4),
                  XXH64(be.data(), be.size() * 4, 0),
                  XXH64(blob->ucode.data(), blob->ucode.size() * 4, 0));
            }
          }
          if (list) {
            writer.BeginDraw(info, regs);
            writer.EndDraw(RuleUsage::Current().used(), stats);
            RuleUsage::Current().Clear();
            std::printf("frame %u draw %u VS %016llX PS %016llX vertices %u%s\n", d.frame,
                        d.draw_index, (unsigned long long)(d.vs ? d.vs->hash : 0),
                        (unsigned long long)(d.ps ? d.ps->hash : 0), d.num_indices,
                        d.is_resolve ? " resolve" : "");
            return true;
          }
          if (frame_filter || count || ps_hash)
            stats.notes.push_back(
                "isolated selected draws: prior EDRAM and resolves not executed; not a whole-frame "
                "golden");
          if (shader_rop)
            stats.notes.push_back(
                "diagnostic shader ROP stub: overwrite surviving PS exports, any covered sample; "
                "no depth/stencil, blend, alpha test or format conversion");
          auto si = regs.Get<reg::RB_SURFACE_INFO>();
          uint32_t w =
              std::min<uint32_t>(regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>().br_x, si.surface_pitch);
          uint32_t h = regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>().br_y;
          // Same readback extent as frame_replay.cpp; bound by available EDRAM tiles.
          auto ci = regs.Get<reg::RB_COLOR_INFO>();
          uint32_t sx = si.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1;
          uint32_t sy = si.msaa_samples >= xenos::MsaaSamples::k2X ? 2 : 1;
          uint32_t tw = xenos::IsColorRenderTargetFormat64bpp(ci.color_format) ? 40 : 80;
          uint32_t pitch = (si.surface_pitch * sx + tw - 1) / tw;
          h = pitch && ci.color_base < 2048
                  ? std::min(h, ((2048 - ci.color_base) / pitch) * 16 / sy)
                  : 0;
          std::vector<float> color(size_t(w) * h * 4, 0.0f);
          std::vector<uint32_t> coverage(size_t(w) * h, 0);
          bool draw_ok = true;
          if (d.is_resolve) {
            if (shader_rop) {
              stats.notes.push_back("resolve skipped by shader stub");
              draw_ok = false;
            }
#if REFXENOS_HAS_ROP
            else {
              ResolveRecord rr;
              draw_ok = Resolve(regs, edram, memory, &rr, stats);
            }
#endif
          } else if (!d.vs || !d.ps) {
            stats.notes.push_back("missing shader");
            draw_ok = false;
          } else {
            Sampler sampler(regs, memory, stats);
            ShaderProgram vs{d.vs->type, d.vs->ucode}, ps{d.ps->type, d.ps->ucode};
            std::vector<VertexOut> vertices;
            vertices.reserve(d.vertex_indices.size());
            std::unordered_map<uint32_t, VertexOut> cache;
            for (auto index : d.vertex_indices) {
              auto it = cache.find(index);
              if (it == cache.end()) {
                VertexOut v;
                draw_ok =
                    RunVertexShader(vs, regs, memory, index, &sampler, &v, stats, &Probe::Get()) &&
                    draw_ok;
                it = cache.emplace(index, v).first;
              }
              vertices.push_back(it->second);
            }
            std::vector<uint8_t> restarts(vertices.size(), 0);
            for (auto n : d.restart_before)
              if (n < restarts.size())
                restarts[n] = 1;
#if REFXENOS_HAS_ROP
            RopState rop = DecodeRopState(regs);
#endif
            RasterizeDraw(
                regs, {d.prim, vertices, restarts},
                [&](const PixelQuad& q) {
                  PixelOut out[4];
                  draw_ok =
                      RunPixelQuad(ps, regs, q, &sampler, out, stats, &Probe::Get()) && draw_ok;
                  for (uint32_t p = 0; p < 4; ++p) {
                    int32_t x = q.x + (p & 1), y = q.y + (p >> 1);
                    if (x < 0 || y < 0 || uint32_t(x) >= w || uint32_t(y) >= h || !q.coverage[p] ||
                        out[p].killed)
                      continue;
                    size_t at = size_t(y) * w + x;
                    coverage[at] |= q.coverage[p];
                    if (shader_rop && (out[p].color_written_mask & 1)) {
                      std::copy_n(out[p].color[0], 4, &color[at * 4]);
                      stats.samples_written += std::popcount(uint32_t(q.coverage[p]));
                    }
                  }
#if REFXENOS_HAS_ROP
                  if (!shader_rop)
                    RopQuad(rop, edram, q, out, stats);
#endif
                },
                stats);
            draw_ok = draw_ok && sampler.ok();
#if REFXENOS_HAS_ROP
            if (!shader_rop && w && h)
              color = ReadbackColorF32(edram, regs, 0, w, h);
#endif
          }
          info.succeeded = draw_ok;
          all_ok &= draw_ok;
          writer.BeginDraw(info, regs);
          if (w && h && !d.is_resolve) {
            std::vector<uint32_t> bits(color.size());
            std::memcpy(bits.data(), color.data(), bits.size() * 4);
            auto key =
                std::format("c0_{}_{}_{}_{}", uint32_t(ci.color_base), uint32_t(ci.color_format),
                            uint32_t(si.surface_pitch), uint32_t(si.msaa_samples));
            writer.AddTarget("rt0", key, w, h, 4, bits);
            auto base = std::format("{:06}", d.draw_index);
            ReplayWriter::WriteNpy(dir / (base + "_coverage.npy"), "<u4", h, w, 1, coverage.data(),
                                   coverage.size() * 4);
            SavePng(dir / (base + "_rt0.png"), w, h, color);
          }
          writer.EndDraw(RuleUsage::Current().used(), stats);
          RuleUsage::Current().Clear();
          std::printf("frame %u draw %u: %llu vertices, %llu quads, %llu writes, %zu notes%s\n",
                      d.frame, d.draw_index, (unsigned long long)stats.vertices_shaded,
                      (unsigned long long)stats.quads, (unsigned long long)stats.samples_written,
                      stats.notes.size(), draw_ok ? "" : " FAILED");
          std::fflush(stdout);
          return true;
        },
        [](uint32_t) {});
    writer.Close(cp.frame(), ok && all_ok && selected > 0);
    std::ofstream(dir / "rules.json") << RulesJson();
    if (!probe_out.empty()) {
      std::ofstream f(probe_out);
      for (auto& line : Probe::Get().lines())
        f << line << '\n';
    }
    std::fprintf(stderr, "selected %u draws, CP %llu draws/%llu swaps, truncated %d\n", selected,
                 (unsigned long long)cp.stats().draws, (unsigned long long)cp.stats().swaps,
                 int(trace.truncated()));
    if (!ok)
      std::fprintf(stderr, "%s\n", cp.error().c_str());
    return ok && all_ok && selected > 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "refxenos: %s\n", e.what());
    return 2;
  }
}
}  // namespace refxenos
