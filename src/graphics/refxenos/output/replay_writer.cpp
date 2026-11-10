#include "replay_writer.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>

#define XXH_INLINE_ALL
#include "xxhash.h"

#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>

namespace refxenos {

using namespace rex::graphics;

namespace {

struct NamedRegister {
  uint32_t index;
  const char* name;
  bool is_float;
};
#define RX_REG(name) {XE_GPU_REG_##name, #name, false}
#define RX_REG_F(name) {XE_GPU_REG_##name, #name, true}
// Same list and order as kStateRegisters in src/graphics/trace/frame_replay.cpp,
// so replay_diff.py's register comparison sees identical keys.
const NamedRegister kStateRegisters[] = {
    RX_REG(RB_MODECONTROL),
    RX_REG(RB_SURFACE_INFO),
    RX_REG(RB_COLOR_INFO),
    RX_REG(RB_COLOR1_INFO),
    RX_REG(RB_COLOR2_INFO),
    RX_REG(RB_COLOR3_INFO),
    RX_REG(RB_DEPTH_INFO),
    RX_REG(RB_COLOR_MASK),
    RX_REG(RB_COLORCONTROL),
    RX_REG(RB_DEPTHCONTROL),
    RX_REG(RB_STENCILREFMASK),
    RX_REG(RB_STENCILREFMASK_BF),
    RX_REG(RB_BLENDCONTROL0),
    RX_REG(RB_BLENDCONTROL1),
    RX_REG(RB_BLENDCONTROL2),
    RX_REG(RB_BLENDCONTROL3),
    RX_REG_F(RB_BLEND_RED),
    RX_REG_F(RB_BLEND_GREEN),
    RX_REG_F(RB_BLEND_BLUE),
    RX_REG_F(RB_BLEND_ALPHA),
    RX_REG_F(RB_ALPHA_REF),
    RX_REG(RB_EDRAM_INFO),
    RX_REG(PA_SU_SC_MODE_CNTL),
    RX_REG_F(PA_SU_POLY_OFFSET_FRONT_SCALE),
    RX_REG_F(PA_SU_POLY_OFFSET_FRONT_OFFSET),
    RX_REG_F(PA_SU_POLY_OFFSET_BACK_SCALE),
    RX_REG_F(PA_SU_POLY_OFFSET_BACK_OFFSET),
    RX_REG(PA_SU_VTX_CNTL),
    RX_REG(PA_SU_POINT_SIZE),
    RX_REG(PA_SU_LINE_CNTL),
    RX_REG(PA_CL_VTE_CNTL),
    RX_REG(PA_CL_CLIP_CNTL),
    RX_REG_F(PA_CL_VPORT_XSCALE),
    RX_REG_F(PA_CL_VPORT_XOFFSET),
    RX_REG_F(PA_CL_VPORT_YSCALE),
    RX_REG_F(PA_CL_VPORT_YOFFSET),
    RX_REG_F(PA_CL_VPORT_ZSCALE),
    RX_REG_F(PA_CL_VPORT_ZOFFSET),
    RX_REG(PA_SC_WINDOW_OFFSET),
    RX_REG(PA_SC_WINDOW_SCISSOR_TL),
    RX_REG(PA_SC_WINDOW_SCISSOR_BR),
    RX_REG(PA_SC_SCREEN_SCISSOR_TL),
    RX_REG(PA_SC_SCREEN_SCISSOR_BR),
    RX_REG(PA_SC_AA_CONFIG),
    RX_REG(PA_SC_AA_MASK),
    RX_REG(PA_SC_LINE_CNTL),
    RX_REG(PA_SC_VIZ_QUERY),
    RX_REG(VGT_DRAW_INITIATOR),
    RX_REG(VGT_DMA_BASE),
    RX_REG(VGT_DMA_SIZE),
    RX_REG(VGT_INDX_OFFSET),
    RX_REG(VGT_MIN_VTX_INDX),
    RX_REG(VGT_MAX_VTX_INDX),
    RX_REG(VGT_MULTI_PRIM_IB_RESET_INDX),
    RX_REG(SQ_PROGRAM_CNTL),
    RX_REG(SQ_CONTEXT_MISC),
    RX_REG(RB_COPY_CONTROL),
    RX_REG(RB_COPY_DEST_BASE),
    RX_REG(RB_COPY_DEST_PITCH),
    RX_REG(RB_COPY_DEST_INFO),
    RX_REG(RB_DEPTH_CLEAR),
    RX_REG(RB_COLOR_CLEAR),
};
#undef RX_REG
#undef RX_REG_F

}  // namespace

std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (uint8_t(c) < 0x20) {
          out += std::format("\\u{:04x}", unsigned(uint8_t(c)));
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string JsonFloat(float f) {
  // Shortest round-trip form like fmt "{}" in frame_replay.cpp; non-finite
  // values as strings.
  if (!std::isfinite(f)) {
    return std::format("\"{}\"", f);
  }
  return std::format("{}", f);
}

bool ReplayWriter::WriteNpy(const std::filesystem::path& path, const char* descr,
                            uint32_t height, uint32_t width, uint32_t channels, const void* data,
                            size_t bytes) {
  std::string header =
      channels > 1
          ? std::format("{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
                        descr, height, width, channels)
          : std::format("{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}), }}", descr,
                        height, width);
  size_t total = 10 + header.size() + 1;
  size_t pad = (64 - total % 64) % 64;
  header.append(pad, ' ');
  header.push_back('\n');
  std::FILE* f = std::fopen(path.string().c_str(), "wb");
  if (!f) {
    return false;
  }
  const uint8_t magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  uint16_t header_len = uint16_t(header.size());
  std::fwrite(magic, 1, sizeof(magic), f);
  std::fwrite(&header_len, 1, 2, f);
  std::fwrite(header.data(), 1, header.size(), f);
  std::fwrite(data, 1, bytes, f);
  std::fclose(f);
  return true;
}

bool ReplayWriter::Open(const std::filesystem::path& out_dir, const std::string& trace_path,
                        const std::string& label, uint32_t title_id, bool full_images) {
  out_dir_ = out_dir;
  trace_path_ = trace_path;
  label_ = label;
  title_id_ = title_id;
  full_ = full_images;
  std::error_code ec;
  std::filesystem::create_directories(out_dir_ / "draws", ec);
  jsonl_ = std::fopen((out_dir_ / "draws.jsonl").string().c_str(), "wb");
  return jsonl_ != nullptr;
}

void ReplayWriter::BeginDraw(const DrawRecordInfo& d, const RegisterFile& regs) {
  current_draw_ = d.draw;
  json_ = std::format(
      "{{\"draw\":{},\"frame\":{},\"packet\":{},\"packet_base\":\"0x{:08X}\","
      "\"prim_type\":{},\"source_select\":{},\"index_count\":{},\"indexed\":{},"
      "\"index_format\":{},\"index_base\":\"0x{:08X}\",\"index_bytes\":{},\"succeeded\":{},"
      "\"edram_mode\":{},\"is_resolve\":{},",
      d.draw, d.frame, d.packet, d.packet_base, d.prim_type, d.source_select, d.index_count,
      d.indexed ? "true" : "false", d.index_format, d.index_base, d.index_bytes,
      d.succeeded ? "true" : "false", d.edram_mode, d.is_resolve ? "true" : "false");
  json_ += d.vs_hash ? std::format("\"vs\":\"{:016X}\",", *d.vs_hash) : "\"vs\":null,";
  json_ += d.ps_hash ? std::format("\"ps\":\"{:016X}\",", *d.ps_hash) : "\"ps\":null,";
  json_ += "\"regs\":{";
  bool first = true;
  for (const NamedRegister& r : kStateRegisters) {
    uint32_t v = regs.values[r.index];
    if (!first) json_ += ",";
    first = false;
    if (r.is_float) {
      json_ += std::format("\"{}\":{}", r.name, JsonFloat(std::bit_cast<float>(v)));
    } else {
      json_ += std::format("\"{}\":\"0x{:08X}\"", r.name, v);
    }
  }
  json_ += "},\"textures\":[";
  first = true;
  for (const auto& t : d.textures) {
    const uint32_t* f = &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + t.fetch * 6];
    xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(t.fetch);
    if (!first) json_ += ",";
    first = false;
    json_ += std::format(
        "{{\"stage\":\"{}\",\"fetch\":{},\"dwords\":[\"0x{:08X}\",\"0x{:08X}\",\"0x{:08X}\","
        "\"0x{:08X}\",\"0x{:08X}\",\"0x{:08X}\"],\"format\":{},\"base\":\"0x{:08X}\"}}",
        t.stage, t.fetch, f[0], f[1], f[2], f[3], f[4], f[5], uint32_t(fetch.format),
        uint32_t(fetch.base_address) << 12);
  }
  json_ += "],\"vertex_fetch\":[";
  first = true;
  for (const auto& v : d.vertex_fetch) {
    xenos::xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(v.fetch);
    if (!first) json_ += ",";
    first = false;
    json_ += std::format("{{\"fetch\":{},\"address\":\"0x{:08X}\",\"bytes\":{},\"stride\":{}}}",
                         v.fetch, uint32_t(fetch.address) << 2, uint32_t(fetch.size) << 2,
                         v.stride_bytes);
  }
  json_ += "],\"targets\":[";
  first_target_ = true;
}

void ReplayWriter::AddTarget(const std::string& label, const std::string& key, uint32_t width,
                             uint32_t height, uint32_t channels,
                             const std::vector<uint32_t>& data) {
  ++readbacks_;
  uint64_t hash = XXH3_64bits(data.data(), data.size() * sizeof(uint32_t));
  auto prev_it = previous_.find(label);
  bool have_prev = prev_it != previous_.end() && prev_it->second.key == key &&
                   prev_it->second.width == width && prev_it->second.height == height &&
                   prev_it->second.channels == channels;
  uint32_t x0 = 0, y0 = 0, x1 = width, y1 = height;
  bool changed = true;
  if (have_prev && !full_) {
    const auto& prev = prev_it->second.data;
    uint32_t bx0 = width, by0 = height, bx1 = 0, by1 = 0;
    size_t row = size_t(width) * channels;
    for (uint32_t y = 0; y < height; ++y) {
      const uint32_t* a = &data[y * row];
      const uint32_t* b = &prev[y * row];
      if (!std::memcmp(a, b, row * sizeof(uint32_t))) continue;
      by0 = std::min(by0, y);
      by1 = y + 1;
      for (uint32_t x = 0; x < width; ++x) {
        if (std::memcmp(a + x * channels, b + x * channels, channels * sizeof(uint32_t))) {
          bx0 = std::min(bx0, x);
          bx1 = std::max(bx1, x + 1);
        }
      }
    }
    if (by1 == 0) {
      changed = false;
      x0 = y0 = x1 = y1 = 0;
    } else {
      x0 = bx0;
      y0 = by0;
      x1 = bx1;
      y1 = by1;
    }
  }
  std::string file;
  if (changed) {
    uint32_t w = x1 - x0, h = y1 - y0;
    std::vector<uint32_t> crop(size_t(w) * h * channels);
    for (uint32_t y = 0; y < h; ++y) {
      std::memcpy(&crop[size_t(y) * w * channels], &data[(size_t(y + y0) * width + x0) * channels],
                  size_t(w) * channels * sizeof(uint32_t));
    }
    file = std::format("draws/{:06}_{}.npy", current_draw_, label);
    WriteNpy(out_dir_ / file, channels == 4 ? "<f4" : "<u4", h, w, channels, crop.data(),
             crop.size() * sizeof(uint32_t));
  }
  if (!first_target_) json_ += ",";
  first_target_ = false;
  json_ += std::format(
      "{{\"target\":\"{}\",\"key\":\"{}\",\"width\":{},\"height\":{},\"channels\":{},"
      "\"format\":\"{}\",\"hash\":\"{:016X}\",\"base\":{},\"file\":{},\"rect\":[{},{},{},{}]}}",
      label, key, width, height, channels, channels == 4 ? "rgba32f" : "depth_raw_u32", hash,
      have_prev && !full_ ? "\"delta\"" : "\"full\"",
      file.empty() ? std::string("null") : "\"" + file + "\"", x0, y0, x1 - x0, y1 - y0);
  Image& img = previous_[label];
  img.key = key;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.data = data;
}

void ReplayWriter::EndDraw(const std::set<const Rule*>& rules, const DrawStats& s) {
  json_ += "],\"refxenos\":{\"rules\":[";
  bool first = true;
  for (const Rule* r : rules) {
    if (!first) json_ += ",";
    first = false;
    json_ += std::format("\"{}@{}{}\"", r->id, r->version, r->assumed ? "*" : "");
  }
  json_ += std::format(
      "],\"stats\":{{\"vertices_shaded\":{},\"primitives_in\":{},\"primitives_culled\":{},"
      "\"primitives_clipped\":{},\"quads\":{},\"pixels_shaded\":{},\"pixels_killed\":{},"
      "\"samples_alpha_test_failed\":{},\"samples_depth_failed\":{},"
      "\"samples_stencil_failed\":{},\"samples_written\":{}}},\"notes\":[",
      s.vertices_shaded, s.primitives_in, s.primitives_culled, s.primitives_clipped, s.quads,
      s.pixels_shaded, s.pixels_killed, s.samples_alpha_test_failed, s.samples_depth_failed,
      s.samples_stencil_failed, s.samples_written);
  first = true;
  for (const auto& n : s.notes) {
    if (!first) json_ += ",";
    first = false;
    json_ += "\"" + JsonEscape(n) + "\"";
  }
  json_ += "]}}\n";
  std::fwrite(json_.data(), 1, json_.size(), jsonl_);
  std::fflush(jsonl_);
  ++draws_;
}

void ReplayWriter::Close(uint32_t frames, bool ok) {
  if (jsonl_) {
    std::fclose(jsonl_);
    jsonl_ = nullptr;
  }
  std::string summary = std::format(
      "{{\n  \"trace\": \"{}\",\n  \"backend\": \"{}\",\n  \"title_id\": \"{:08X}\",\n"
      "  \"frames\": {},\n  \"draws\": {},\n  \"readbacks\": {},\n  \"failed_readbacks\": 0,\n"
      "  \"scratch_address\": \"0x00000000\",\n  \"scratch_conflict\": false,\n  \"ok\": {}\n}}\n",
      JsonEscape(trace_path_), JsonEscape(label_), title_id_, frames, draws_, readbacks_,
      ok ? "true" : "false");
  std::ofstream(out_dir_ / "summary.json") << summary;
}

std::string RulesJson() {
  std::string out = "{\n";
  bool first = true;
  for (const auto& [id, r] : RuleRegistry::Get().rules()) {
    if (!first) out += ",\n";
    first = false;
    out += std::format("  \"{}\": {{\"version\": {}, \"assumed\": {}, \"source\": \"{}\"",
                       JsonEscape(id), r->version, r->assumed ? "true" : "false",
                       JsonEscape(r->source ? r->source : ""));
    if (r->assumed) {
      out += std::format(", \"hw_test\": \"{}\"", JsonEscape(r->hw_test ? r->hw_test : ""));
    }
    out += "}";
  }
  out += "\n}\n";
  return out;
}

std::string RulesMarkdown() {
  auto cell = [](const char* s) {
    std::string t = s ? s : "";
    std::string out;
    for (char c : t) {
      if (c == '|') {
        out += "\\|";
      } else if (c == '\n') {
        out += ' ';
      } else {
        out += c;
      }
    }
    return out;
  };
  size_t total = 0, assumed = 0;
  for (const auto& [id, r] : RuleRegistry::Get().rules()) {
    ++total;
    assumed += r->assumed;
  }
  std::string out = std::format(
      "<!-- generated by `refxenos --rules-md`; do not edit by hand -->\n"
      "{} rules registered, {} assumed.\n\n"
      "| Rule | Ver | Why assumed / best evidence | Hardware test that settles it |\n"
      "| --- | --- | --- | --- |\n",
      total, assumed);
  for (const auto& [id, r] : RuleRegistry::Get().rules()) {
    if (!r->assumed) continue;
    out += std::format("| `{}` | {} | {} | {} |\n", id, r->version, cell(r->source),
                       cell(r->hw_test));
  }
  return out;
}

}  // namespace refxenos
