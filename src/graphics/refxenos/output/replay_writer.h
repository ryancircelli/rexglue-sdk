// refxenos: writer for the rexgpu-replay output layout.
//
// refxenos writes the same directory layout as rexgpu-replay
// (src/graphics/trace/frame_replay.cpp: draws.jsonl, draws/NNNNNN_<label>.npy,
// summary.json) so replay_diff.py and oracle.py compare it with Metal, Vulkan
// and Xenia replays unchanged. Extra refxenos-only keys are added under
// "refxenos" in each draw record (rules used, stats, notes); the other tools
// ignore unknown keys.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <rex/graphics/register_file.h>

#include "../core/common.h"

namespace refxenos {

struct DrawRecordInfo {
  uint32_t draw = 0;
  uint32_t frame = 0;
  uint32_t packet = 0;
  uint32_t packet_base = 0;
  uint32_t prim_type = 0;
  uint32_t source_select = 0;
  uint32_t index_count = 0;
  bool indexed = false;
  uint32_t index_format = 0;
  uint32_t index_base = 0;
  uint32_t index_bytes = 0;
  bool succeeded = true;
  uint32_t edram_mode = 0;
  bool is_resolve = false;
  std::optional<uint64_t> vs_hash, ps_hash;
  struct Texture {
    const char* stage;  // "vs" / "ps"
    uint32_t fetch;
  };
  std::vector<Texture> textures;
  struct VertexFetch {
    uint32_t fetch;
    uint32_t stride_bytes;
  };
  std::vector<VertexFetch> vertex_fetch;
};

class ReplayWriter {
 public:
  bool Open(const std::filesystem::path& out_dir, const std::string& trace_path,
            const std::string& label, uint32_t title_id, bool full_images);
  // Starts a draw record; regs are the state at the draw.
  void BeginDraw(const DrawRecordInfo& info, const rex::graphics::RegisterFile& regs);
  // Adds a readback. data is HxWxchannels u32 (rgba32f bits or raw depth).
  // key identifies the surface as rexgpu-replay does ("c{rt}_{base}_{fmt}_{pitch}_{msaa}",
  // "d_{base}_{fmt}_{pitch}_{msaa}"); delta rectangles are relative to the previous
  // readback with the same label and key, exactly like frame_replay.cpp EmitTarget.
  void AddTarget(const std::string& label, const std::string& key, uint32_t width,
                 uint32_t height, uint32_t channels, const std::vector<uint32_t>& data);
  // Ends the record, appending refxenos extras.
  void EndDraw(const std::set<const Rule*>& rules, const DrawStats& stats);
  void Close(uint32_t frames, bool ok);

  uint32_t draws() const { return draws_; }
  uint32_t readbacks() const { return readbacks_; }

  static bool WriteNpy(const std::filesystem::path& path, const char* descr, uint32_t height,
                       uint32_t width, uint32_t channels, const void* data, size_t bytes);

 private:
  struct Image {
    std::string key;
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint32_t> data;
  };
  std::filesystem::path out_dir_;
  std::string trace_path_, label_;
  uint32_t title_id_ = 0;
  bool full_ = false;
  std::FILE* jsonl_ = nullptr;
  std::string json_;
  bool first_target_ = true;
  uint32_t current_draw_ = 0;
  uint32_t draws_ = 0, readbacks_ = 0;
  std::map<std::string, Image> previous_;
};

std::string JsonEscape(const std::string& s);
std::string JsonFloat(float f);

// Rule table exports.
std::string RulesJson();      // every registered rule, for golden rules.json
std::string RulesMarkdown();  // assumed-rules table for research/reference-renderer.md

}  // namespace refxenos
