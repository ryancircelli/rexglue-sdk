#include "cp/command_processor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace refxenos {

using namespace rex::graphics;  // XE_GPU_REG_* register enum

// ---------------------------------------------------------------------------
// Rules.

// Trace replay.
RX_DEFINE_RULE(cp_replay_packet_order, 1,
               "rexglue src/graphics/trace/frame_replay.cpp Replayer::Run (PacketStart copies "
               "dwords to base_ptr, PacketEnd executes, zero-count/unpaired PacketEnd skipped, "
               "draws numbered after ++packet_index_)",
               false, "");
RX_DEFINE_RULE(cp_replay_draw_numbering, 1,
               "rexglue src/graphics/command_processor.cpp:1350-1460 (draw event only for "
               "kDMA/kAutoIndex, not when PA_SC_VIZ_QUERY viz_query_ena && kill_pix_post_hi_z)",
               false, "");

// Packet decoding.
RX_DEFINE_RULE(cp_bad_packet_nop, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:268",
               false, "");
RX_DEFINE_RULE(cp_type0_write, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:328", false,
               "");
RX_DEFINE_RULE(cp_type1_write, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:357", false,
               "");
RX_DEFINE_RULE(cp_type2_nop, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:371", false,
               "");
RX_DEFINE_RULE(cp_type3_predicate, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:412", false,
               "");
RX_DEFINE_RULE(cp_type3_consumes_count, 1,
               "assumed: the CP advances exactly count dwords past a type-3 header regardless of "
               "how many the opcode consumes (Xenia only asserts this, "
               "pm4_command_processor_implement.h:598)",
               true,
               "Ring a type-3 NOP-able opcode (e.g. INTERRUPT with count 3) followed by a type-0 "
               "write of SCRATCH_REG2; read SCRATCH_REG2 back from the CPU to see whether the "
               "trailing dwords were skipped or executed as packets.");
RX_DEFINE_RULE(cp_unknown_opcode_skip, 1,
               "assumed: unknown type-3 opcodes are skipped by count and execution continues "
               "(Xenia HitUnimplementedOpcode pm4_command_processor_implement.h:625 skips count "
               "but aborts the buffer)",
               true,
               "Submit an IB holding an undocumented opcode (e.g. 0x70, count 1) followed by a "
               "type-0 SCRATCH_REG2 write and check whether the write lands.");
RX_DEFINE_RULE(cp_bin_mask_select, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:511-546",
               false, "");

// Register writes.
RX_DEFINE_RULE(cp_scratch_writeback, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/command_processor.cc:597", false, "");
RX_DEFINE_RULE(cp_coher_status_host, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/command_processor.cc:597 and :857", false, "");
RX_DEFINE_RULE(cp_constant_ranges, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/command_processor.cc:789-846 and "
               "pm4_command_processor_implement.h:1168",
               false, "");
RX_DEFINE_RULE(cp_set_constant2, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1211 and "
               ":1275",
               false, "");
RX_DEFINE_RULE(cp_load_alu_constant, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1222",
               false, "");
RX_DEFINE_RULE(cp_reg_rmw, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:780", false,
               "");

// Memory side effects.
RX_DEFINE_RULE(cp_mem_endian, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:806,828 "
               "(endian = address & 3, address & ~3, GpuSwap of a host-order load/store)",
               false, "");
RX_DEFINE_RULE(cp_wait_reg_mem, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:721",
               false, "");
RX_DEFINE_RULE(cp_wait_reg_mem_no_block, 1,
               "assumed: in a trace replay every wait is already satisfied on hardware; an "
               "unsatisfied compare is counted and execution continues",
               true,
               "None needed: a WAIT_REG_MEM that never passes would hang the real GPU and the "
               "capture would not exist.");
RX_DEFINE_RULE(cp_cond_write, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:845",
               false, "");
RX_DEFINE_RULE(cp_event_write, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:889",
               false, "");
RX_DEFINE_RULE(cp_event_write_shd, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:906",
               false, "");
RX_DEFINE_RULE(cp_event_write_ext, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:943",
               false, "");
RX_DEFINE_RULE(cp_xe_swap, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:666",
               false, "");
RX_DEFINE_RULE(cp_indirect_buffer, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:692",
               false, "");

// Shaders.
RX_DEFINE_RULE(cp_im_load, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1285 "
               "(type = addr_type & 3, address = addr_type & ~3, size = start_size & 0xFFFF)",
               false, "");
RX_DEFINE_RULE(cp_im_load_immediate, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1315",
               false, "");
RX_DEFINE_RULE(cp_im_load_start_ignored, 1,
               "assumed: IM_LOAD start (start_size >> 16) is 0 in practice and the program "
               "replaces the bound shader (Xenia asserts start == 0, "
               "pm4_command_processor_implement.h:1325)",
               true,
               "IM_LOAD a 2-instruction VS at start 0, then a different VS with start 64, draw "
               "with SQ_PROGRAM_CNTL vs base 0 and read back which program ran via memexport.");

// Draws.
RX_DEFINE_RULE(cp_draw_packet, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1024-1166 "
               "(DRAW_INDX: viz token then VGT_DRAW_INITIATOR; DRAW_INDX_2: initiator only; kDMA "
               "adds VGT_DMA_BASE, VGT_DMA_SIZE; leftovers skipped)",
               false, "");
RX_DEFINE_RULE(cp_draw_viz_conditional_unconditional, 1,
               "assumed: draws conditional on a VIZ_QUERY result (token bit 8) are rendered "
               "unconditionally by the CP; hardware skips them when the survey with the same id "
               "(bits 0:5) passed no samples (Xenia PrepareVIZDraw, command_processor.cc:1002), "
               "which needs the ROP's sample count and is left to the draw consumer "
               "(DrawCall::viz_conditional / viz_token)",
               true,
               "Draw a VIZ_QUERY survey fully hidden behind depth, then a DRAW_INDX with token "
               "bit 8 set and the same id; check whether the conditional draw writes pixels.");
RX_DEFINE_RULE(cp_draw_resolve_mode, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/command_processor.cc IssueDraw (RB_MODECONTROL "
               "edram_mode == kCopy makes the draw a resolve)",
               false, "");

// Index fetch.
RX_DEFINE_RULE(cp_index_dma_base_align, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/pm4_command_processor_implement.h:1070-1090 "
               "(guest_base = VGT_DMA_BASE & ~(index size - 1))",
               false, "");
RX_DEFINE_RULE(cp_index_swap16, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:118-123 and "
               "primitive_processor.cc:645 (16-bit indices: 8in32 acts as 8in16, 16in32 as none)",
               false, "");
RX_DEFINE_RULE(cp_index_mask24, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:183-184 (vertex "
               "index = GpuSwap(raw, swap_mode) & 0xFFFFFF)",
               false, "");
RX_DEFINE_RULE(cp_index_past_num_words_zero, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:174-186 (positions "
               "past VGT_DMA_SIZE num_words fetch index 0)",
               true,
               "Indexed draw with num_indices 6 and VGT_DMA_SIZE num_words 3, the buffer "
               "followed by known non-zero data; VS memexports r0.x per vertex.");
RX_DEFINE_RULE(cp_index_reset_before_offset, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/registers.h:340 and "
               "draw_extent_estimator.cc:188 (reset compare on the swapped, 24-bit-masked index "
               "against VGT_MULTI_PRIM_IB_RESET_INDX, before offset and clamp)",
               false, "");
RX_DEFINE_RULE(cp_index_reset_all_prims, 1,
               "XDK primitive_reset.htm (\"you can use this feature with any indexed primitive "
               "type\"); Xenia primitive_processor.cc:670-700 limits it to strips only for host "
               "API reasons",
               false, "");
RX_DEFINE_RULE(cp_index_offset_clamp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:196-198 "
               "(min(max_indx, max(min_indx, (index + indx_offset) & 0xFFFFFF)))",
               false, "");
RX_DEFINE_RULE(cp_index_auto, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:190-194 (auto-index "
               "draws: index = position, then offset and clamp)",
               false, "");
RX_DEFINE_RULE(cp_index_raw_load, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/draw_extent_estimator.cc:178-182 (indices "
               "are loaded as host-order words from guest memory, then GpuSwap'd)",
               false, "");

// ---------------------------------------------------------------------------
// Helpers.

namespace {

uint32_t LoadLE32(const Memory& m, uint32_t addr) {
  uint32_t v;
  std::memcpy(&v, m.At(addr), 4);
  return v;
}
uint16_t LoadLE16(const Memory& m, uint32_t addr) {
  uint16_t v;
  std::memcpy(&v, m.At(addr), 2);
  return v;
}
void StoreLE32(Memory& m, uint32_t addr, uint32_t v) { m.Write(addr, &v, 4); }

uint32_t bswap32(uint32_t v) {
  return (v >> 24) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | (v << 24);
}

// Guest-endian (host-order load + GpuSwap) access as Xenia does it.
uint32_t LoadSwapped(const Memory& m, uint32_t addr, xenos::Endian e) {
  return xenos::GpuSwap(LoadLE32(m, addr), e);
}
void StoreSwapped(Memory& m, uint32_t addr, uint32_t v, xenos::Endian e) {
  StoreLE32(m, addr, xenos::GpuSwap(v, e));
}

bool MatchValueAndRef(uint32_t value, uint32_t ref, uint32_t wait_info) {
  switch (wait_info & 0x7) {
    case 0: return false;
    case 1: return value < ref;
    case 2: return value <= ref;
    case 3: return value == ref;
    case 4: return value != ref;
    case 5: return value >= ref;
    case 6: return value > ref;
    default: return true;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Index fetch.

IndexFetchParams IndexFetchParams::FromRegisters(const RegisterFile& regs) {
  IndexFetchParams p;
  auto init = regs.Get<reg::VGT_DRAW_INITIATOR>();
  auto dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  p.source_select = init.source_select;
  p.index_format = init.index_size;
  p.num_indices = init.num_indices;
  p.dma_base = regs[XE_GPU_REG_VGT_DMA_BASE];
  p.num_words = dma_size.num_words;
  p.swap_mode = dma_size.swap_mode;
  p.reset_enabled = regs.Get<reg::PA_SU_SC_MODE_CNTL>().multi_prim_ib_ena != 0;
  p.reset_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
  p.index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  p.min_index = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
  p.max_index = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
  return p;
}

void FetchVertexIndices(const Memory& memory, const IndexFetchParams& p,
                        std::vector<uint32_t>& vertex_indices,
                        std::vector<uint32_t>& restart_before) {
  vertex_indices.clear();
  restart_before.clear();
  vertex_indices.reserve(p.num_indices);

  auto finish = [&](uint32_t v) {
    RX_USE(cp_index_offset_clamp);
    uint32_t x = (v + (p.index_offset & 0xFFFFFF)) & 0xFFFFFF;
    x = std::min(p.max_index & 0xFFFFFF, std::max(p.min_index & 0xFFFFFF, x));
    vertex_indices.push_back(x);
  };

  if (p.source_select == xenos::SourceSelect::kAutoIndex) {
    RX_USE(cp_index_auto);
    for (uint32_t i = 0; i < p.num_indices; ++i) finish(i);
    return;
  }
  if (p.source_select != xenos::SourceSelect::kDMA) return;

  RX_USE(cp_index_dma_base_align);
  RX_USE(cp_index_raw_load);
  RX_USE(cp_index_mask24);
  const bool is16 = p.index_format == xenos::IndexFormat::kInt16;
  const uint32_t size = is16 ? 2u : 4u;
  const uint32_t base = p.dma_base & ~(size - 1);
  xenos::Endian endian = p.swap_mode;
  if (is16) {
    RX_USE(cp_index_swap16);
    if (endian == xenos::Endian::k8in32) {
      endian = xenos::Endian::k8in16;
    } else if (endian == xenos::Endian::k16in32) {
      endian = xenos::Endian::kNone;
    }
  }
  const uint32_t reset_index = p.reset_index & 0xFFFFFF;
  bool pending_restart = false;
  for (uint32_t i = 0; i < p.num_indices; ++i) {
    uint32_t v = 0;
    if (i < p.num_words) {
      if (is16) {
        v = xenos::GpuSwap(LoadLE16(memory, base + i * 2), endian);
      } else {
        v = xenos::GpuSwap(LoadLE32(memory, base + i * 4), endian);
      }
      v &= 0xFFFFFF;
    } else {
      RX_USE(cp_index_past_num_words_zero);
    }
    if (p.reset_enabled) {
      RX_USE(cp_index_reset_before_offset);
      RX_USE(cp_index_reset_all_prims);
      if (v == reset_index) {
        pending_restart = true;
        continue;
      }
    }
    if (pending_restart) {
      uint32_t pos = uint32_t(vertex_indices.size());
      if (restart_before.empty() || restart_before.back() != pos) restart_before.push_back(pos);
      pending_restart = false;
    }
    finish(v);
  }
  if (pending_restart) {
    uint32_t pos = uint32_t(vertex_indices.size());
    if (restart_before.empty() || restart_before.back() != pos) restart_before.push_back(pos);
  }
}

// ---------------------------------------------------------------------------
// Ring reader over guest memory.

struct CommandProcessor::Reader {
  const Memory& mem;
  uint32_t ptr;
  uint32_t end;
  uint32_t remaining() const { return ptr < end ? (end - ptr) / 4 : 0; }
  uint32_t Read() {
    uint32_t v = mem.ReadU32BE(ptr);
    ptr += 4;
    return v;
  }
  void Skip(uint32_t dwords) { ptr += dwords * 4; }
};

CommandProcessor::CommandProcessor(Memory& memory)
    : memory_(memory), regs_(std::make_unique<RegisterFile>()) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::Note(std::string s) {
  if (stats_.notes.size() < 256) stats_.notes.push_back(std::move(s));
}

void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  if (index >= RegisterFile::kRegisterCount) {
    Note("register write out of range: 0x" + std::to_string(index));
    return;
  }
  regs_->values[index] = value;
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    RX_USE(cp_scratch_writeback);
    uint32_t i = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1u << i) & regs_->values[XE_GPU_REG_SCRATCH_UMSK]) {
      uint32_t addr = regs_->values[XE_GPU_REG_SCRATCH_ADDR] + i * 4;
      // store_and_swap: big-endian in guest memory.
      StoreLE32(memory_, addr, bswap32(value));
    }
  } else if (index == XE_GPU_REG_COHER_STATUS_HOST) {
    RX_USE(cp_coher_status_host);
    regs_->values[index] |= 0x80000000u;
  }
}

void CommandProcessor::LoadShader(xenos::ShaderType type, uint32_t guest_address,
                                  const uint8_t* be_ucode, uint32_t size_dwords) {
  uint64_t hash = XXH3_64bits(be_ucode, size_t(size_dwords) * 4);
  uint64_t key = hash ^ (uint64_t(type) << 63);
  auto it = shaders_.find(key);
  std::shared_ptr<const ShaderBlob> blob;
  if (it != shaders_.end()) {
    blob = it->second;
  } else {
    auto b = std::make_shared<ShaderBlob>();
    b->type = type;
    b->hash = hash;
    b->guest_address = guest_address;
    b->ucode.resize(size_dwords);
    for (uint32_t i = 0; i < size_dwords; ++i) {
      const uint8_t* q = be_ucode + i * 4;
      b->ucode[i] = (uint32_t(q[0]) << 24) | (uint32_t(q[1]) << 16) | (uint32_t(q[2]) << 8) | q[3];
    }
    blob = b;
    shaders_.emplace(key, blob);
  }
  if (type == xenos::ShaderType::kVertex) {
    active_vs_ = blob;
  } else {
    active_ps_ = blob;
  }
}

void CommandProcessor::ExecutePacket(uint32_t ptr, uint32_t count) {
  Reader r{memory_, ptr, ptr + count * 4};
  while (r.remaining() && !stop_) {
    if (!ExecuteOne(r)) break;
  }
}

void CommandProcessor::ExecuteIndirectBuffer(uint32_t ptr, uint32_t count) {
  RX_USE(cp_indirect_buffer);
  if (ib_depth_ >= 8) {
    Note("indirect buffer nesting too deep, skipped");
    return;
  }
  ++stats_.indirect_buffers_from_memory;
  ++ib_depth_;
  ExecutePacket(ptr, count);
  --ib_depth_;
}

bool CommandProcessor::ExecuteOne(Reader& r) {
  const uint32_t packet = r.Read();
  if (packet == 0 || packet == 0x0BADF00D) {
    RX_USE(cp_bad_packet_nop);
    ++stats_.bad_packets;
    return true;
  }
  switch (packet >> 30) {
    case 0: {
      RX_USE(cp_type0_write);
      ++stats_.type0;
      uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
      if (r.remaining() < count) {
        ++stats_.count_overflows;
        Note("type-0 count overflow");
        r.Skip(r.remaining());
        return false;
      }
      uint32_t base = packet & 0x7FFF;
      bool one_reg = (packet >> 15) & 1;
      for (uint32_t i = 0; i < count; ++i) {
        WriteRegister(one_reg ? base : base + i, r.Read());
      }
      return true;
    }
    case 1: {
      RX_USE(cp_type1_write);
      ++stats_.type1;
      if (r.remaining() < 2) {
        ++stats_.count_overflows;
        r.Skip(r.remaining());
        return false;
      }
      uint32_t reg1 = packet & 0x7FF;
      uint32_t reg2 = (packet >> 11) & 0x7FF;
      WriteRegister(reg1, r.Read());
      WriteRegister(reg2, r.Read());
      return true;
    }
    case 2:
      RX_USE(cp_type2_nop);
      ++stats_.type2;
      return true;
    default:
      ++stats_.type3;
      return ExecuteType3(r, packet);
  }
}

bool CommandProcessor::ExecuteType3(Reader& r, uint32_t packet) {
  const uint32_t opcode = (packet >> 8) & 0x7F;
  const uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (r.remaining() < count) {
    ++stats_.count_overflows;
    Note("type-3 count overflow, opcode 0x" + std::to_string(opcode));
    r.Skip(r.remaining());
    return false;
  }
  ++stats_.opcodes[opcode];
  const uint32_t data_end = r.ptr + count * 4;

  if (packet & 1) {
    RX_USE(cp_type3_predicate);
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    if (!any_pass || opcode == xenos::PM4_XE_SWAP) {
      ++stats_.predicated_skipped;
      r.ptr = data_end;
      return true;
    }
  }

  using namespace xenos;
  switch (opcode) {
    case PM4_ME_INIT:
      me_init_dwords_ = count;
      break;
    case PM4_NOP:
    case PM4_INTERRUPT:          // CPU interrupt only
    case PM4_INVALIDATE_STATE:   // host cache hint
    case PM4_CONTEXT_UPDATE:
    case PM4_WAIT_IB_PFD_COMPLETE:
    case PM4_WAIT_FOR_IDLE:
      break;
    case PM4_XE_SWAP:
      RX_USE(cp_xe_swap);
      ++stats_.xe_swaps;
      ++counter_;
      break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD: {
      uint32_t list_ptr = r.Read();
      uint32_t list_length = r.Read() & 0xFFFFF;
      r.ptr = data_end;
      ExecuteIndirectBuffer(list_ptr & 0x1FFFFFFF, list_length);
      return true;
    }
    case PM4_WAIT_REG_MEM: {
      RX_USE(cp_wait_reg_mem);
      uint32_t wait_info = r.Read();
      uint32_t poll_reg_addr = r.Read();
      uint32_t ref = r.Read();
      uint32_t mask = r.Read();
      (void)r.Read();  // wait
      uint32_t value;
      if (wait_info & 0x10) {
        value = LoadSwapped(memory_, poll_reg_addr & ~3u, Endian(poll_reg_addr & 3));
      } else {
        uint32_t reg = poll_reg_addr & 0xFFFF;
        if (reg == XE_GPU_REG_COHER_STATUS_HOST) {
          // MakeCoherent: the write-back completes immediately.
          if (regs_->values[reg] & 0x80000000u) regs_->values[reg] = 0;
        }
        value = reg < RegisterFile::kRegisterCount ? regs_->values[reg] : 0;
      }
      if (!MatchValueAndRef(value & mask, ref, wait_info)) {
        RX_USE(cp_wait_reg_mem_no_block);
        ++stats_.wait_reg_mem_unsatisfied;
      }
    } break;
    case PM4_REG_RMW: {
      RX_USE(cp_reg_rmw);
      uint32_t rmw_info = r.Read();
      uint32_t and_mask = r.Read();
      uint32_t or_mask = r.Read();
      uint32_t reg = rmw_info & 0x1FFF;
      uint32_t value = reg < RegisterFile::kRegisterCount ? regs_->values[reg] : 0;
      if ((rmw_info >> 31) & 1) {
        value &= regs_->values[and_mask & 0x1FFF];
      } else {
        value &= and_mask;
      }
      if ((rmw_info >> 30) & 1) {
        value |= regs_->values[or_mask & 0x1FFF];
      } else {
        value |= or_mask;
      }
      WriteRegister(reg, value);
    } break;
    case PM4_REG_TO_MEM: {
      RX_USE(cp_mem_endian);
      uint32_t reg = r.Read() & 0xFFFF;
      uint32_t mem_addr = r.Read();
      uint32_t value = reg < RegisterFile::kRegisterCount ? regs_->values[reg] : 0;
      StoreSwapped(memory_, mem_addr & ~3u, value, Endian(mem_addr & 3));
    } break;
    case PM4_MEM_WRITE: {
      RX_USE(cp_mem_endian);
      uint32_t write_addr = r.Read();
      Endian e = Endian(write_addr & 3);
      write_addr &= ~3u;
      for (uint32_t i = 0; i + 1 < count; ++i) {
        StoreSwapped(memory_, write_addr, r.Read(), e);
        write_addr += 4;
      }
    } break;
    case PM4_COND_WRITE: {
      RX_USE(cp_cond_write);
      uint32_t wait_info = r.Read();
      uint32_t poll_reg_addr = r.Read();
      uint32_t ref = r.Read();
      uint32_t mask = r.Read();
      uint32_t write_reg_addr = r.Read();
      uint32_t write_data = r.Read();
      uint32_t value;
      if (wait_info & 0x10) {
        value = LoadSwapped(memory_, poll_reg_addr & ~3u, Endian(poll_reg_addr & 3));
      } else {
        uint32_t reg = poll_reg_addr & 0xFFFF;
        value = reg < RegisterFile::kRegisterCount ? regs_->values[reg] : 0;
      }
      if (MatchValueAndRef(value & mask, ref, wait_info)) {
        if (wait_info & 0x100) {
          RX_USE(cp_mem_endian);
          StoreSwapped(memory_, write_reg_addr & ~3u, write_data, Endian(write_reg_addr & 3));
        } else {
          WriteRegister(write_reg_addr & 0xFFFF, write_data);
        }
      }
    } break;
    case PM4_EVENT_WRITE: {
      RX_USE(cp_event_write);
      uint32_t initiator = r.Read();
      WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
    } break;
    case PM4_EVENT_WRITE_SHD: {
      RX_USE(cp_event_write_shd);
      uint32_t initiator = r.Read();
      uint32_t address = r.Read();
      uint32_t value = r.Read();
      WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
      uint32_t data = ((initiator >> 31) & 1) ? counter_ : value;
      if (address > 0x1FFFFFFF) {
        // Xenia remaps writeback addresses through the CPU's 0x7F000000 view;
        // physical memory here, so fold to the backing.
        Note("EVENT_WRITE_SHD to writeback address folded to physical");
      }
      StoreSwapped(memory_, address & 0x1FFFFFFC, data, Endian(address & 3));
    } break;
    case PM4_EVENT_WRITE_EXT: {
      RX_USE(cp_event_write_ext);
      uint32_t initiator = r.Read();
      uint32_t address = r.Read();
      WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
      // Screen extents: {0, 8192 >> 3, 0, 8192 >> 3, 0, 1} as big-endian u16.
      const uint16_t extents[6] = {0, 8192 >> 3, 0, 8192 >> 3, 0, 1};
      uint8_t be[12];
      for (int i = 0; i < 6; ++i) {
        be[i * 2] = uint8_t(extents[i] >> 8);
        be[i * 2 + 1] = uint8_t(extents[i]);
      }
      memory_.Write(address & ~3u, be, sizeof(be));
    } break;
    case PM4_EVENT_WRITE_ZPD: {
      uint32_t initiator = r.Read();
      WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
      ++stats_.zpd_reports_skipped;
    } break;
    case PM4_DRAW_INDX:
    case PM4_DRAW_INDX_2:
      ExecuteDraw(r, opcode, count);
      r.ptr = data_end;
      return true;
    case PM4_SET_CONSTANT: {
      RX_USE(cp_constant_ranges);
      uint32_t offset_type = r.Read();
      uint32_t index = offset_type & 0x7FF;
      uint32_t type = (offset_type >> 16) & 0xFF;
      uint32_t base;
      switch (type) {
        case 0: base = 0x4000; break;  // ALU
        case 1: base = 0x4800; break;  // fetch
        case 2: base = 0x4900; break;  // bool
        case 3: base = 0x4908; break;  // loop
        case 4: base = 0x2000; break;  // registers
        default:
          Note("SET_CONSTANT unknown type " + std::to_string(type));
          r.ptr = data_end;
          return true;
      }
      for (uint32_t i = 0; i + 1 < count; ++i) WriteRegister(base + index + i, r.Read());
    } break;
    case PM4_SET_CONSTANT2:
    case PM4_SET_SHADER_CONSTANTS: {
      RX_USE(cp_set_constant2);
      uint32_t index = r.Read() & 0xFFFF;
      for (uint32_t i = 0; i + 1 < count; ++i) WriteRegister(index + i, r.Read());
    } break;
    case PM4_LOAD_ALU_CONSTANT: {
      RX_USE(cp_load_alu_constant);
      uint32_t address = r.Read() & 0x3FFFFFFF;
      uint32_t offset_type = r.Read();
      uint32_t size_dwords = r.Read() & 0xFFF;
      uint32_t index = offset_type & 0x7FF;
      uint32_t type = (offset_type >> 16) & 0xFF;
      uint32_t base;
      switch (type) {
        case 0: base = 0x4000; break;
        case 1: base = 0x4800; break;
        case 2: base = 0x4900; break;
        case 3: base = 0x4908; break;
        case 4: base = 0x2000; break;
        default:
          Note("LOAD_ALU_CONSTANT unknown type " + std::to_string(type));
          r.ptr = data_end;
          return true;
      }
      for (uint32_t i = 0; i < size_dwords; ++i) {
        WriteRegister(base + index + i, memory_.ReadU32BE(address + i * 4));
      }
    } break;
    case PM4_IM_LOAD: {
      RX_USE(cp_im_load);
      uint32_t addr_type = r.Read();
      uint32_t start_size = r.Read();
      ShaderType type = ShaderType(addr_type & 3);
      uint32_t addr = addr_type & ~3u;
      uint32_t size_dwords = start_size & 0xFFFF;
      if (start_size >> 16) {
        RX_USE(cp_im_load_start_ignored);
        Note("IM_LOAD with non-zero start");
      }
      if (type != ShaderType::kVertex && type != ShaderType::kPixel) {
        Note("IM_LOAD unknown shader type " + std::to_string(uint32_t(type)));
        break;
      }
      std::vector<uint8_t> bytes(size_t(size_dwords) * 4);
      memory_.Read(addr, bytes.data(), uint32_t(bytes.size()));
      LoadShader(type, addr & 0x1FFFFFFF, bytes.data(), size_dwords);
    } break;
    case PM4_IM_LOAD_IMMEDIATE: {
      RX_USE(cp_im_load_immediate);
      uint32_t dword0 = r.Read();
      uint32_t start_size = r.Read();
      ShaderType type = ShaderType(dword0);
      uint32_t size_dwords = start_size & 0xFFFF;
      if (start_size >> 16) {
        RX_USE(cp_im_load_start_ignored);
        Note("IM_LOAD_IMMEDIATE with non-zero start");
      }
      if (size_dwords + 2 > count) {
        ++stats_.count_overflows;
        Note("IM_LOAD_IMMEDIATE size exceeds packet");
        size_dwords = count - 2;
      }
      if (type != ShaderType::kVertex && type != ShaderType::kPixel) {
        Note("IM_LOAD_IMMEDIATE unknown shader type " + std::to_string(dword0));
        break;
      }
      std::vector<uint8_t> bytes(size_t(size_dwords) * 4);
      memory_.Read(r.ptr, bytes.data(), uint32_t(bytes.size()));
      LoadShader(type, r.ptr & 0x1FFFFFFF, bytes.data(), size_dwords);
    } break;
    case PM4_VIZ_QUERY: {
      ++stats_.viz_queries;
      uint32_t dword0 = r.Read();
      WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR,
                    (dword0 & 0x100) ? uint32_t(VIZQUERY_END) : uint32_t(VIZQUERY_START));
    } break;
    case PM4_SET_BIN_MASK_LO:
      RX_USE(cp_bin_mask_select);
      bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | r.Read();
      break;
    case PM4_SET_BIN_MASK_HI:
      RX_USE(cp_bin_mask_select);
      bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (uint64_t(r.Read()) << 32);
      break;
    case PM4_SET_BIN_SELECT_LO:
      RX_USE(cp_bin_mask_select);
      bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | r.Read();
      break;
    case PM4_SET_BIN_SELECT_HI:
      RX_USE(cp_bin_mask_select);
      bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (uint64_t(r.Read()) << 32);
      break;
    case PM4_SET_BIN_MASK: {
      RX_USE(cp_bin_mask_select);
      uint64_t hi = r.Read();
      uint64_t lo = r.Read();
      bin_mask_ = (hi << 32) | lo;
    } break;
    case PM4_SET_BIN_SELECT: {
      RX_USE(cp_bin_mask_select);
      uint64_t hi = r.Read();
      uint64_t lo = r.Read();
      bin_select_ = (hi << 32) | lo;
    } break;
    default:
      RX_USE(cp_unknown_opcode_skip);
      if (stats_.unknown_opcodes[opcode]++ == 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "unknown PM4 opcode 0x%02X (count %u)", opcode, count);
        Note(buf);
      }
      break;
  }
  RX_USE(cp_type3_consumes_count);
  r.ptr = data_end;
  return true;
}

bool CommandProcessor::ExecuteDraw(Reader& r, uint32_t opcode, uint32_t count) {
  RX_USE(cp_draw_packet);
  uint32_t remaining = count;
  uint32_t viz_token = 0;
  if (opcode == xenos::PM4_DRAW_INDX) {
    viz_token = r.Read();
    --remaining;
  }
  if (!remaining) {
    Note("draw packet too small for VGT_DRAW_INITIATOR");
    return false;
  }
  reg::VGT_DRAW_INITIATOR init;
  init.value = r.Read();
  --remaining;
  regs_->values[XE_GPU_REG_VGT_DRAW_INITIATOR] = init.value;

  bool ok = true;
  switch (init.source_select) {
    case xenos::SourceSelect::kDMA: {
      if (remaining < 2) {
        Note("draw packet too small for VGT_DMA_BASE/SIZE");
        return false;
      }
      WriteRegister(XE_GPU_REG_VGT_DMA_BASE, r.Read());
      WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, r.Read());
      remaining -= 2;
    } break;
    case xenos::SourceSelect::kAutoIndex:
      break;
    default:
      ok = false;
      break;
  }
  RX_USE(cp_replay_draw_numbering);
  if (!ok) {
    ++stats_.draws_immediate_dropped;
    return true;
  }
  auto viz = regs_->Get<reg::PA_SC_VIZ_QUERY>();
  if (viz.viz_query_ena && viz.kill_pix_post_hi_z) {
    ++stats_.draws_viz_killed;
    return true;
  }

  DrawCall d;
  d.frame = frame_;
  d.draw_index = draw_index_++;
  d.packet_index = packet_index_;
  d.packet_base = current_packet_base_;
  d.regs = regs_.get();
  d.vs = active_vs_;
  d.ps = active_ps_;
  RX_USE(cp_draw_resolve_mode);
  d.is_resolve = regs_->Get<reg::RB_MODECONTROL>().edram_mode == xenos::EdramMode::kCopy;
  d.prim = init.prim_type;
  d.num_indices = init.num_indices;
  d.source_select = init.source_select;
  d.indexed = init.source_select == xenos::SourceSelect::kDMA;
  d.major_mode_explicit = xenos::IsMajorModeExplicit(init.major_mode, init.prim_type);
  d.viz_token = viz_token;
  d.viz_conditional = (viz_token & 0x100) != 0;
  if (d.viz_conditional) {
    RX_USE(cp_draw_viz_conditional_unconditional);
    ++stats_.draws_viz_conditional;
  }
  if (d.indexed) {
    d.index_format = init.index_size;
    uint32_t size = d.index_format == xenos::IndexFormat::kInt16 ? 2u : 4u;
    auto dma_size = regs_->Get<reg::VGT_DMA_SIZE>();
    d.index_base = regs_->values[XE_GPU_REG_VGT_DMA_BASE] & ~(size - 1);
    d.index_bytes = dma_size.num_words * size;
    d.index_endian = dma_size.swap_mode;
  }
  FetchVertexIndices(memory_, IndexFetchParams::FromRegisters(*regs_), d.vertex_indices,
                     d.restart_before);

  ++stats_.draws;
  if (d.is_resolve) ++stats_.resolves;
  if (on_draw_ && *on_draw_ && !(*on_draw_)(d)) stop_ = true;
  return true;
}

// ---------------------------------------------------------------------------
// Trace replay.

bool CommandProcessor::Run(const TraceReader& trace,
                           const std::function<bool(const DrawCall&)>& on_draw,
                           const std::function<void(uint32_t frame)>& on_swap) {
  RX_USE(cp_replay_packet_order);
  on_draw_ = &on_draw;
  stop_ = false;
  using T = trace::TraceCommandType;
  const TraceCommand* pending = nullptr;
  std::vector<uint8_t> buf;
  for (const TraceCommand& cmd : trace.commands()) {
    if (stop_) break;
    ++stats_.trace_commands;
    switch (cmd.type) {
      case T::kPrimaryBufferStart:
      case T::kPrimaryBufferEnd:
      case T::kIndirectBufferStart:
      case T::kIndirectBufferEnd:
        break;
      case T::kPacketStart:
        memory_.Write(cmd.base_ptr, cmd.payload, cmd.payload_bytes);
        pending = &cmd;
        break;
      case T::kPacketEnd:
        if (!pending || pending->count == 0) {
          ++stats_.packets_skipped_empty;
          pending = nullptr;
          break;
        }
        current_packet_base_ = pending->base_ptr;
        ++packet_index_;
        ++stats_.leaf_packets_executed;
        ExecutePacket(pending->base_ptr, pending->count);
        pending = nullptr;
        break;
      case T::kMemoryRead: {
        std::string err;
        if (!TraceReader::Decode(cmd, buf, &err)) {
          ++stats_.decode_failures;
          Note("MemoryRead decode failed: " + err);
          break;
        }
        ++stats_.memory_reads;
        stats_.memory_read_bytes += buf.size();
        memory_.Write(cmd.base_ptr, buf.data(), uint32_t(buf.size()));
      } break;
      case T::kMemoryWrite:
        ++stats_.memory_writes_ignored;
        break;
      case T::kEdramSnapshot:
        ++stats_.edram_snapshots_ignored;
        break;
      case T::kEvent:
        if (cmd.event_type == 0) {
          ++stats_.swaps;
          if (on_swap) on_swap(frame_);
          ++frame_;
        } else {
          Note("unknown trace event " + std::to_string(cmd.event_type));
        }
        break;
      case T::kRegisters: {
        std::string err;
        if (!TraceReader::Decode(cmd, buf, &err)) {
          ++stats_.decode_failures;
          Note("Registers decode failed: " + err);
          break;
        }
        uint32_t first = cmd.first_register;
        uint32_t n = uint32_t(buf.size() / 4);
        if (first >= RegisterFile::kRegisterCount) break;
        n = std::min<uint32_t>(n, RegisterFile::kRegisterCount - first);
        std::memcpy(&regs_->values[first], buf.data(), size_t(n) * 4);
      } break;
      case T::kGammaRamp:
        ++stats_.gamma_ramps;
        break;
      default:
        error_ = "unknown trace command type " + std::to_string(uint32_t(cmd.type));
        on_draw_ = nullptr;
        return false;
    }
  }
  on_draw_ = nullptr;
  return true;
}

}  // namespace refxenos
