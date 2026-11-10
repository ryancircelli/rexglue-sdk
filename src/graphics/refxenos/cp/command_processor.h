// refxenos: PM4 command processor and index fetch.
//
// Replays a Xenia-format .xtr trace the same way rexgpu-replay does
// (src/graphics/trace/frame_replay.cpp Replayer::Run): packet dwords are
// copied to their guest address at PacketStart and executed at PacketEnd;
// PacketEnds whose PacketStart carried zero dwords, or that close an
// INDIRECT_BUFFER whose contents were already executed packet by packet, are
// skipped. MemoryRead blocks are written to guest memory in trace order.
//
// Every draw the SDK's command processor reports to its trace hooks (DMA and
// auto-index draws not killed by PA_SC_VIZ_QUERY kill_pix_post_hi_z, resolves
// included) is delivered to on_draw synchronously, before the next trace
// command, with draw_index / packet_index numbered exactly like draws.jsonl.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common.h"
#include "cp/trace_reader.h"

namespace refxenos {

struct ShaderBlob {
  xenos::ShaderType type;
  std::vector<uint32_t> ucode;  // host-endian dwords as ucode.h expects
  // The SDK's Shader::ucode_data_hash(): XXH3_64bits over the ucode exactly as
  // it sits in guest memory (big-endian dwords), size_dwords * 4 bytes. Equal
  // to the vs/ps hash in rexgpu-replay's draws.jsonl.
  uint64_t hash;
  uint32_t guest_address;  // IM_LOAD address, or the ring address of the
                           // inline ucode for IM_LOAD_IMMEDIATE
};

struct DrawCall {
  uint32_t frame = 0;
  // Global, counting every draw incl. resolves, in trace order; same numbering
  // as rexgpu-replay's draws.jsonl "draw".
  uint32_t draw_index = 0;
  // Number of leaf packets executed so far including this one (draws.jsonl
  // "packet": the replayer increments packet_index_ after each executed leaf
  // packet and records draws after the increment).
  uint32_t packet_index = 0;
  uint32_t packet_base = 0;  // guest address of the executed packet

  const RegisterFile* regs = nullptr;  // live state at the draw
  std::shared_ptr<const ShaderBlob> vs, ps;

  bool is_resolve = false;  // RB_MODECONTROL edram_mode == kCopy
  xenos::PrimitiveType prim = xenos::PrimitiveType::kNone;
  uint32_t num_indices = 0;  // VGT_DRAW_INITIATOR num_indices
  bool indexed = false;      // source_select == kDMA
  bool major_mode_explicit = false;

  // draws.jsonl fields.
  xenos::SourceSelect source_select = xenos::SourceSelect::kDMA;
  xenos::IndexFormat index_format = xenos::IndexFormat::kInt16;  // kInt16 for auto draws
  xenos::Endian index_endian = xenos::Endian::kNone;  // VGT_DMA_SIZE swap_mode as written
  uint32_t index_base = 0;   // VGT_DMA_BASE & ~(index size - 1); 0 for auto draws
  uint32_t index_bytes = 0;  // VGT_DMA_SIZE num_words * index size; 0 for auto draws

  // Per index-buffer position: the vertex index the VS sees (r0.x) after
  // swap, 24-bit mask, offset and clamp. Reset entries are removed.
  std::vector<uint32_t> vertex_indices;
  // Positions in vertex_indices before which a primitive reset occurred
  // (strictly increasing; consecutive resets collapse into one entry; a reset
  // at the very start or end of the buffer is recorded as 0 / size()).
  std::vector<uint32_t> restart_before;

  bool viz_conditional = false;  // DRAW_INDX viz token bit 8
  uint32_t viz_token = 0;        // full token dword (id in bits 0:5); 0 for DRAW_INDX_2
};

// Inputs of index fetch, as latched by the draw packet.
struct IndexFetchParams {
  xenos::SourceSelect source_select = xenos::SourceSelect::kAutoIndex;
  xenos::IndexFormat index_format = xenos::IndexFormat::kInt16;
  uint32_t dma_base = 0;   // VGT_DMA_BASE
  uint32_t num_words = 0;  // VGT_DMA_SIZE num_words
  xenos::Endian swap_mode = xenos::Endian::kNone;
  uint32_t num_indices = 0;  // VGT_DRAW_INITIATOR num_indices
  bool reset_enabled = false;   // PA_SU_SC_MODE_CNTL multi_prim_ib_ena
  uint32_t reset_index = 0;     // VGT_MULTI_PRIM_IB_RESET_INDX (24-bit)
  uint32_t index_offset = 0;    // VGT_INDX_OFFSET (24-bit)
  uint32_t min_index = 0;       // VGT_MIN_VTX_INDX (24-bit)
  uint32_t max_index = 0;       // VGT_MAX_VTX_INDX (24-bit)

  static IndexFetchParams FromRegisters(const RegisterFile& regs);
};

// The hardware index pipeline: fetch, swap, 24-bit mask, reset compare,
// offset, 24-bit wrap, clamp. Auto-index draws produce 0..num_indices-1 put
// through offset and clamp.
void FetchVertexIndices(const Memory& memory, const IndexFetchParams& p,
                        std::vector<uint32_t>& vertex_indices,
                        std::vector<uint32_t>& restart_before);

// Anything the trace contained that refxenos does not model, and other
// counters. Kept going past all of these.
struct CpStats {
  uint64_t trace_commands = 0;
  uint64_t leaf_packets_executed = 0;  // == last draw's packet_index domain
  uint64_t packets_skipped_empty = 0;  // PacketEnd with no / zero-count PacketStart
  uint64_t type0 = 0, type1 = 0, type2 = 0, type3 = 0, bad_packets = 0;
  uint64_t predicated_skipped = 0;
  uint64_t draws = 0, resolves = 0, swaps = 0, xe_swaps = 0;
  uint64_t draws_immediate_dropped = 0;  // SourceSelect kImmediate / invalid
  uint64_t draws_viz_killed = 0;         // PA_SC_VIZ_QUERY ena && kill_pix_post_hi_z
  uint64_t draws_viz_conditional = 0;    // DRAW_INDX token bit 8 (rendered unconditionally)
  uint64_t memory_reads = 0, memory_read_bytes = 0;
  uint64_t memory_writes_ignored = 0, edram_snapshots_ignored = 0, gamma_ramps = 0;
  uint64_t decode_failures = 0;
  uint64_t wait_reg_mem_unsatisfied = 0;
  uint64_t zpd_reports_skipped = 0;      // EVENT_WRITE_ZPD with RB_SAMPLE_COUNT_ADDR != 0
  uint64_t viz_queries = 0;
  uint64_t indirect_buffers_from_memory = 0;  // IB executed from guest memory
  uint64_t count_overflows = 0;          // packet header count exceeds the packet
  std::map<uint32_t, uint64_t> opcodes;          // type-3 opcode histogram
  std::map<uint32_t, uint64_t> unknown_opcodes;  // reported, skipped by count
  std::vector<std::string> notes;
};

class CommandProcessor {
 public:
  explicit CommandProcessor(Memory& memory);
  ~CommandProcessor();

  // Replays the whole trace. on_draw returning false stops the replay (Run
  // then returns true). on_swap is called for each swap Event command with the
  // index of the frame that just ended. Returns false on a malformed trace.
  bool Run(const TraceReader& trace, const std::function<bool(const DrawCall&)>& on_draw,
           const std::function<void(uint32_t frame)>& on_swap);

  const RegisterFile& regs() const { return *regs_; }
  RegisterFile& regs() { return *regs_; }
  const CpStats& stats() const { return stats_; }
  const std::string& error() const { return error_; }
  uint32_t frame() const { return frame_; }

  // Executes `count` dwords of PM4 at guest address `ptr` (exposed for tests).
  void ExecutePacket(uint32_t ptr, uint32_t count);

 private:
  struct Reader;
  bool ExecuteOne(Reader& r);
  bool ExecuteType3(Reader& r, uint32_t packet);
  bool ExecuteDraw(Reader& r, uint32_t opcode, uint32_t count);
  void ExecuteIndirectBuffer(uint32_t ptr, uint32_t count);
  void WriteRegister(uint32_t index, uint32_t value);
  void LoadShader(xenos::ShaderType type, uint32_t guest_address, const uint8_t* be_ucode,
                  uint32_t size_dwords);
  void Note(std::string s);

  Memory& memory_;
  std::unique_ptr<RegisterFile> regs_;
  CpStats stats_;
  std::string error_;

  std::unordered_map<uint64_t, std::shared_ptr<const ShaderBlob>> shaders_;
  std::shared_ptr<const ShaderBlob> active_vs_, active_ps_;

  uint64_t bin_select_ = 0xFFFFFFFFull;
  uint64_t bin_mask_ = 0xFFFFFFFFull;
  uint32_t counter_ = 0;  // XE_SWAP count (EVENT_WRITE_SHD bit 31 source)
  uint32_t me_init_dwords_ = 0;
  int ib_depth_ = 0;

  // Replay bookkeeping (frame_replay.cpp numbering).
  uint32_t frame_ = 0;
  uint32_t draw_index_ = 0;
  uint32_t packet_index_ = 0;
  uint32_t current_packet_base_ = 0;
  const std::function<bool(const DrawCall&)>* on_draw_ = nullptr;
  bool stop_ = false;
};

}  // namespace refxenos
