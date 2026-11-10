// refxenos cp test: index-fetch known answers, then a full replay of the
// 415608C3_4685 main-menu trace compared draw by draw against rexgpu-replay's
// draws.jsonl when one is available.
//
// Usage: refxenos_test_cp [trace.xtr] [draws.jsonl|-] [start_frame]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/common.h"
#include "cp/command_processor.h"
#include "cp/trace_reader.h"

using namespace refxenos;
using namespace rex::graphics;  // XE_GPU_REG_*

static int g_failures = 0;
#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

static std::string Join(const std::vector<uint32_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s;
}

#define CHECK_VEC(v, ...)                                                          \
  do {                                                                             \
    std::vector<uint32_t> expect_ = {__VA_ARGS__};                                 \
    if ((v) != expect_) {                                                          \
      std::printf("FAIL %s:%d: %s = [%s], expected [%s]\n", __FILE__, __LINE__, #v, \
                  Join(v).c_str(), Join(expect_).c_str());                         \
      ++g_failures;                                                                \
    }                                                                              \
  } while (0)

static void Put16BE(Memory& m, uint32_t a, uint16_t v) {
  uint8_t b[2] = {uint8_t(v >> 8), uint8_t(v)};
  m.Write(a, b, 2);
}
static void Put32BE(Memory& m, uint32_t a, uint32_t v) {
  uint8_t b[4] = {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
  m.Write(a, b, 4);
}
static void Put32LE(Memory& m, uint32_t a, uint32_t v) { m.Write(a, &v, 4); }

static IndexFetchParams Dma(xenos::IndexFormat fmt, uint32_t base, uint32_t n, xenos::Endian e) {
  IndexFetchParams p;
  p.source_select = xenos::SourceSelect::kDMA;
  p.index_format = fmt;
  p.dma_base = base;
  p.num_words = n;
  p.num_indices = n;
  p.swap_mode = e;
  p.min_index = 0;
  p.max_index = 0xFFFFFF;
  return p;
}

static void IndexFetchTests(Memory& m) {
  std::vector<uint32_t> vi, rb;
  const uint32_t A = 0x00100000;

  // 16-bit, big-endian in memory, swap 8in16 (the D3D default).
  Put16BE(m, A + 0, 1);
  Put16BE(m, A + 2, 2);
  Put16BE(m, A + 4, 0x1234);
  auto p = Dma(xenos::IndexFormat::kInt16, A, 3, xenos::Endian::k8in16);
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 1, 2, 0x1234);
  CHECK(rb.empty());

  // 16-bit with 8in32 behaves as 8in16; 16in32 as none.
  p.swap_mode = xenos::Endian::k8in32;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 1, 2, 0x1234);
  p.swap_mode = xenos::Endian::kNone;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0x0100, 0x0200, 0x3412);
  p.swap_mode = xenos::Endian::k16in32;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0x0100, 0x0200, 0x3412);

  // 16-bit base is aligned down to 2 bytes.
  p.swap_mode = xenos::Endian::k8in16;
  p.dma_base = A + 1;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 1, 2, 0x1234);

  // 32-bit: 8in32 on big-endian data, 24-bit mask.
  const uint32_t B = 0x00200000;
  Put32BE(m, B + 0, 7);
  Put32BE(m, B + 4, 0xAB000005);  // top byte dropped
  Put32BE(m, B + 8, 0x00FFFFFF);
  p = Dma(xenos::IndexFormat::kInt32, B + 2, 3, xenos::Endian::k8in32);  // base aligned to 4
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 7, 5, 0xFFFFFF);

  // 32-bit 16in32 and 8in16 on data stored as host-order words.
  const uint32_t C = 0x00300000;
  Put32LE(m, C, 0x00030002);
  p = Dma(xenos::IndexFormat::kInt32, C, 1, xenos::Endian::k16in32);
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0x00020003);
  p.swap_mode = xenos::Endian::k8in16;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0x00000200 | (0x0300u << 16 & 0xFFFFFF));
  p.swap_mode = xenos::Endian::kNone;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0x030002);

  // Offset (24-bit wrap) then clamp.
  p = Dma(xenos::IndexFormat::kInt16, A, 3, xenos::Endian::k8in16);
  p.index_offset = 10;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 11, 12, 0x1234 + 10);
  p.min_index = 12;
  p.max_index = 100;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 12, 12, 100);
  p.min_index = 0;
  p.max_index = 0xFFFFFF;
  p.index_offset = 0xFFFFFF;  // -1 in 24 bits
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0, 1, 0x1233);
  p.index_offset = 0;
  p.min_index = 5;  // clamp applies after offset, so index 0 + offset 0 -> 5
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 5, 5, 0x1234);

  // Reset: compared after swap, before offset/clamp; reset entries removed.
  const uint32_t D = 0x00400000;
  const uint16_t strip[] = {0, 1, 2, 0xFFFF, 3, 4, 5, 0xFFFF, 0xFFFF, 6};
  for (int i = 0; i < 10; ++i) Put16BE(m, D + i * 2, strip[i]);
  p = Dma(xenos::IndexFormat::kInt16, D, 10, xenos::Endian::k8in16);
  p.reset_enabled = true;
  p.reset_index = 0xFFFF;
  p.index_offset = 100;
  p.max_index = 104;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 100, 101, 102, 103, 104, 104, 104);  // 0xFFFF was not offset/clamped
  CHECK_VEC(rb, 3, 6);                                 // consecutive resets collapse
  // Disabled: the 0xFFFF entries are ordinary indices (offset then clamped).
  p.reset_enabled = false;
  FetchVertexIndices(m, p, vi, rb);
  CHECK(vi.size() == 10);
  CHECK(vi[3] == 104);
  CHECK(rb.empty());
  // Reset index compared to the swapped value: with swap none the stored
  // 0xFFFF still reads 0xFFFF but 0x0001 reads 0x0100.
  p.reset_enabled = true;
  p.reset_index = 0x0100;
  p.swap_mode = xenos::Endian::kNone;
  p.index_offset = 0;
  p.max_index = 0xFFFFFF;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 0, 0x0200, 0xFFFF, 0x0300, 0x0400, 0x0500, 0xFFFF, 0xFFFF, 0x0600);
  CHECK_VEC(rb, 1);
  // 16-bit indices never match a reset index above 0xFFFF.
  p.swap_mode = xenos::Endian::k8in16;
  p.reset_index = 0x1FFFF;
  FetchVertexIndices(m, p, vi, rb);
  CHECK(vi.size() == 10);
  CHECK(rb.empty());
  // Reset at the start and the end.
  Put16BE(m, D, 0xFFFF);
  p.reset_index = 0xFFFF;
  p.num_indices = p.num_words = 8;  // ends on the 0xFFFF pair at positions 7..8? -> 7 is 0xFFFF
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 1, 2, 3, 4, 5);
  CHECK_VEC(rb, 0, 2, 5);
  Put16BE(m, D, 0);
  // 32-bit reset compares the 24-bit masked value.
  Put32BE(m, B + 4, 0x7F00FFFF);
  p = Dma(xenos::IndexFormat::kInt32, B, 3, xenos::Endian::k8in32);
  p.reset_enabled = true;
  p.reset_index = 0x00FFFF;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 7, 0xFFFFFF);
  CHECK_VEC(rb, 1);

  // Positions past num_words read 0.
  p = Dma(xenos::IndexFormat::kInt16, A, 2, xenos::Endian::k8in16);
  p.num_indices = 4;
  p.index_offset = 3;
  FetchVertexIndices(m, p, vi, rb);
  CHECK_VEC(vi, 4, 5, 3, 3);

  // Auto-index: position, offset, clamp; reset never applies.
  IndexFetchParams a;
  a.source_select = xenos::SourceSelect::kAutoIndex;
  a.num_indices = 5;
  a.index_offset = 2;
  a.min_index = 3;
  a.max_index = 5;
  a.reset_enabled = true;
  a.reset_index = 1;
  FetchVertexIndices(m, a, vi, rb);
  CHECK_VEC(vi, 3, 3, 4, 5, 5);
  CHECK(rb.empty());
}

// Minimal PM4 checks through ExecutePacket.
static void PacketTests() {
  auto mem = std::make_unique<Memory>();
  CommandProcessor cp(*mem);
  const uint32_t R = 0x01000000;
  auto ring = [&](std::vector<uint32_t> dw) {
    for (size_t i = 0; i < dw.size(); ++i) Put32BE(*mem, R + uint32_t(i) * 4, dw[i]);
    cp.ExecutePacket(R, uint32_t(dw.size()));
  };
  auto t3 = [](uint32_t op, uint32_t count) { return 0xC0000000u | ((count - 1) << 16) | (op << 8); };

  // Type 0 sequential and one-reg.
  ring({0x00010000u | XE_GPU_REG_VGT_INDX_OFFSET, 5, 6});
  CHECK(cp.regs()[XE_GPU_REG_VGT_INDX_OFFSET] == 5);
  CHECK(cp.regs()[XE_GPU_REG_VGT_INDX_OFFSET + 1] == 6);
  ring({0x00018000u | 0x2100, 7, 8});
  CHECK(cp.regs()[0x2100] == 8);
  // SET_CONSTANT type 4 (registers, +0x2000) and type 0 (ALU, +0x4000).
  ring({t3(xenos::PM4_SET_CONSTANT, 3), (4u << 16) | 0x101, 0xAA, 0xBB});
  CHECK(cp.regs()[0x2101] == 0xAA && cp.regs()[0x2102] == 0xBB);
  ring({t3(xenos::PM4_SET_CONSTANT, 2), 0x10, 0x3F800000});
  CHECK(cp.regs()[0x4010] == 0x3F800000);
  // MEM_WRITE with 8in32.
  ring({t3(xenos::PM4_MEM_WRITE, 2), 0x00500000u | 2, 0x11223344});
  CHECK(mem->ReadU32BE(0x00500000) == 0x11223344);
  // Predicated packet skipped when bin select & mask == 0.
  ring({t3(xenos::PM4_SET_BIN_MASK_LO, 1), 0, t3(xenos::PM4_SET_BIN_MASK_HI, 1), 0,
        t3(xenos::PM4_SET_CONSTANT, 2) | 1, 0x20, 0x1234});
  CHECK(cp.regs()[0x4020] == 0);
  ring({t3(xenos::PM4_SET_BIN_MASK_LO, 1), 1, t3(xenos::PM4_SET_CONSTANT, 2) | 1, 0x20, 0x1234});
  CHECK(cp.regs()[0x4020] == 0x1234);
  // Unknown opcode skipped by count, execution continues.
  ring({t3(0x70, 2), 1, 2, 0x00000000u | XE_GPU_REG_VGT_INDX_OFFSET, 9});
  CHECK(cp.regs()[XE_GPU_REG_VGT_INDX_OFFSET] == 9);
  CHECK(cp.stats().unknown_opcodes.count(0x70) == 1);
  // IM_LOAD_IMMEDIATE binds a shader with host-endian ucode.
  // DRAW_INDX_2 auto-index triggers no callback outside Run, but must parse.
  ring({t3(xenos::PM4_IM_LOAD_IMMEDIATE, 4), 0, 2, 0xDEADBEEF, 0x01020304,
        t3(xenos::PM4_DRAW_INDX_2, 1), (3u << 16) | (2u << 6) | 4u});
  CHECK(cp.stats().draws == 1);
  CHECK(cp.regs().Get<reg::VGT_DRAW_INITIATOR>().num_indices == 3);
}

struct OracleDraw {
  uint32_t draw = 0, frame = 0, packet = 0, prim = 0, source_select = 0, index_count = 0;
  uint32_t index_format = 0, index_bytes = 0;
  uint32_t index_base = 0, packet_base = 0;
  bool indexed = false, is_resolve = false;
  std::string vs, ps;
};

static bool JsonField(const std::string& line, const char* key, std::string& out) {
  std::string k = std::string("\"") + key + "\":";
  size_t p = line.find(k);
  if (p == std::string::npos) return false;
  p += k.size();
  if (line[p] == '"') {
    size_t e = line.find('"', p + 1);
    out = line.substr(p + 1, e - p - 1);
  } else {
    size_t e = line.find_first_of(",}", p);
    out = line.substr(p, e - p);
  }
  return true;
}
static uint32_t U(const std::string& s) { return uint32_t(std::strtoul(s.c_str(), nullptr, 0)); }

static std::vector<OracleDraw> LoadOracle(const std::string& path) {
  std::vector<OracleDraw> v;
  std::ifstream f(path);
  std::string line, s;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    OracleDraw d;
    if (JsonField(line, "draw", s)) d.draw = U(s);
    if (JsonField(line, "frame", s)) d.frame = U(s);
    if (JsonField(line, "packet", s)) d.packet = U(s);
    if (JsonField(line, "packet_base", s)) d.packet_base = U(s);
    if (JsonField(line, "prim_type", s)) d.prim = U(s);
    if (JsonField(line, "source_select", s)) d.source_select = U(s);
    if (JsonField(line, "index_count", s)) d.index_count = U(s);
    if (JsonField(line, "indexed", s)) d.indexed = s == "true";
    if (JsonField(line, "index_format", s)) d.index_format = U(s);
    if (JsonField(line, "index_base", s)) d.index_base = U(s);
    if (JsonField(line, "index_bytes", s)) d.index_bytes = U(s);
    if (JsonField(line, "is_resolve", s)) d.is_resolve = s == "true";
    if (JsonField(line, "vs", s)) d.vs = s;
    if (JsonField(line, "ps", s)) d.ps = s;
    v.push_back(d);
  }
  return v;
}

static std::string Hex16(const std::shared_ptr<const ShaderBlob>& b) {
  if (!b) return "0000000000000000";
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)b->hash);
  return buf;
}

int main(int argc, char** argv) {
  const char* home = std::getenv("HOME");
  std::string trace_path = argc > 1 ? argv[1]
                                    : std::string(home ? home : "") +
                                          "/Games/Xbox360/bo2-build/gputest/captures/415608C3_4685.xtr";
  std::string oracle_path =
      argc > 2 ? argv[2] : "/tmp/bo2/oracle/det/menu4685_fixed_final/run1/draws.jsonl";

  auto mem = std::make_unique<Memory>();
  IndexFetchTests(*mem);
  PacketTests();
  std::printf("index fetch + packet unit tests: %s\n", g_failures ? "FAILED" : "ok");

  if (argc > 1 && std::string(argv[1]) == "--unit-only") return g_failures ? 1 : 0;

  TraceReader trace;
  if (!trace.Open(trace_path)) {
    std::printf("FAIL: %s\n", trace.error().c_str());
    return 1;
  }
  std::printf("trace %s: title %08X, build '%s', %zu commands\n", trace_path.c_str(),
              trace.title_id(), trace.build_sha().c_str(), trace.commands().size());

  if (trace.truncated()) std::printf("note: %s\n", trace.error().c_str());
  uint32_t start_frame = argc > 3 ? U(argv[3]) : 0;
  std::vector<OracleDraw> oracle;
  if (oracle_path != "-") oracle = LoadOracle(oracle_path);
  std::printf("oracle %s: %zu draws\n", oracle_path.c_str(), oracle.size());

  mem = std::make_unique<Memory>();
  CommandProcessor cp(*mem);
  cp.SetStartFrame(start_frame);
  size_t delivered = 0;
  uint32_t first_delivered = UINT32_MAX;
  uint32_t mismatches = 0, total_indices = 0, restarts = 0, ib_bytes_checked = 0;
  bool ok = cp.Run(
      trace,
      [&](const DrawCall& d) {
        if (first_delivered == UINT32_MAX) first_delivered = d.draw_index;
        if (d.draw_index < first_delivered + 10) {
          std::printf("  draw %2u pkt %3u prim %-2u %s indices %u%s%s vs %s ps %s\n", d.draw_index,
                      d.packet_index, uint32_t(d.prim), d.indexed ? "dma " : "auto",
                      d.num_indices, d.is_resolve ? " RESOLVE" : "",
                      d.viz_conditional ? " VIZ" : "", Hex16(d.vs).c_str(), Hex16(d.ps).c_str());
        }
        total_indices += uint32_t(d.vertex_indices.size());
        restarts += uint32_t(d.restart_before.size());
        if (d.draw_index < oracle.size()) {
          const OracleDraw& o = oracle[d.draw_index];
          bool same = o.draw == d.draw_index && o.frame == d.frame && o.packet == d.packet_index &&
                      o.packet_base == d.packet_base && o.prim == uint32_t(d.prim) &&
                      o.source_select == uint32_t(d.source_select) &&
                      o.index_count == d.num_indices && o.indexed == d.indexed &&
                      o.is_resolve == d.is_resolve && o.vs == Hex16(d.vs) && o.ps == Hex16(d.ps);
          if (d.indexed) {
            same = same && o.index_format == uint32_t(d.index_format) &&
                   o.index_base == d.index_base && o.index_bytes == d.index_bytes;
          }
          if (!same) {
            ++mismatches;
            std::printf(
                "  MISMATCH draw %u: ours pkt %u base %08X prim %u ss %u n %u idx %d fmt %u ib "
                "%08X/%u res %d vs %s ps %s | oracle pkt %u base %08X prim %u ss %u n %u idx %d "
                "fmt %u ib %08X/%u res %d vs %s ps %s\n",
                d.draw_index, d.packet_index, d.packet_base, uint32_t(d.prim),
                uint32_t(d.source_select), d.num_indices, d.indexed, uint32_t(d.index_format),
                d.index_base, d.index_bytes, d.is_resolve, Hex16(d.vs).c_str(),
                Hex16(d.ps).c_str(), o.packet, o.packet_base, o.prim, o.source_select,
                o.index_count, o.indexed, o.index_format, o.index_base, o.index_bytes,
                o.is_resolve, o.vs.c_str(), o.ps.c_str());
          }
        }
        if (d.indexed) {
          ib_bytes_checked += d.index_bytes;
          uint32_t maxv = 0;
          for (uint32_t v : d.vertex_indices) maxv = std::max(maxv, v);
          if (d.draw_index >= 29 && d.draw_index <= 31) {
            std::printf("  draw %u indices[0..%zu): %s (max %u, restarts %zu)\n", d.draw_index,
                        std::min<size_t>(d.vertex_indices.size(), 12),
                        Join(std::vector<uint32_t>(
                                 d.vertex_indices.begin(),
                                 d.vertex_indices.begin() +
                                     std::min<size_t>(d.vertex_indices.size(), 12)))
                            .c_str(),
                        maxv, d.restart_before.size());
          }
        }
        ++delivered;
        return true;
      },
      [&](uint32_t frame) {
        if (frame < 3 || frame + 1 >= start_frame) std::printf("  swap after frame %u\n", frame);
      });
  if (!ok) {
    std::printf("FAIL: replay error: %s\n", cp.error().c_str());
    return 1;
  }

  const CpStats& s = cp.stats();
  std::printf("draws %llu, resolves %llu, swaps %llu, leaf packets %llu (skipped empty %llu)\n",
              (unsigned long long)s.draws, (unsigned long long)s.resolves,
              (unsigned long long)s.swaps, (unsigned long long)s.leaf_packets_executed,
              (unsigned long long)s.packets_skipped_empty);
  std::printf("type0 %llu type1 %llu type2 %llu type3 %llu bad %llu predicated-skipped %llu\n",
              (unsigned long long)s.type0, (unsigned long long)s.type1,
              (unsigned long long)s.type2, (unsigned long long)s.type3,
              (unsigned long long)s.bad_packets, (unsigned long long)s.predicated_skipped);
  std::printf(
      "memory reads %llu (%llu bytes), decode failures %llu, immediate dropped %llu, viz killed "
      "%llu, viz conditional %llu, wait_reg_mem unsatisfied %llu, IB from memory %llu, count "
      "overflows %llu\n",
      (unsigned long long)s.memory_reads, (unsigned long long)s.memory_read_bytes,
      (unsigned long long)s.decode_failures, (unsigned long long)s.draws_immediate_dropped,
      (unsigned long long)s.draws_viz_killed, (unsigned long long)s.draws_viz_conditional,
      (unsigned long long)s.wait_reg_mem_unsatisfied,
      (unsigned long long)s.indirect_buffers_from_memory, (unsigned long long)s.count_overflows);
  std::printf("frames %u, start frame %u, draws delivered %zu, fast-forwarded %llu\n", cp.frame(),
              start_frame, delivered, (unsigned long long)s.draws_fast_forwarded);
  std::printf("indices fetched %u, restarts %u\n", total_indices, restarts);
  std::printf("opcodes:");
  for (auto& [op, n] : s.opcodes) std::printf(" %02X:%llu", op, (unsigned long long)n);
  std::printf("\n");
  for (auto& [op, n] : s.unknown_opcodes) {
    std::printf("unknown opcode 0x%02X x%llu\n", op, (unsigned long long)n);
  }
  for (auto& n : s.notes) std::printf("note: %s\n", n.c_str());

  std::printf("rules used:");
  for (const Rule* r : RuleUsage::Current().used()) {
    if (std::strncmp(r->id, "cp_", 3) == 0) std::printf(" %s%s", r->id, r->assumed ? "*" : "");
  }
  std::printf("\n");

  if (!oracle.empty()) {
    CHECK(oracle.size() == size_t(s.draws));
    std::printf("draw count: refxenos %llu, rexgpu-replay %zu -> %s; per-draw mismatches %u\n",
                (unsigned long long)s.draws, oracle.size(),
                oracle.size() == size_t(s.draws) ? "match" : "DIFFER",
                mismatches);
    CHECK(mismatches == 0);
  } else {
    std::printf("no oracle draws.jsonl, count not compared\n");
  }
  CHECK(s.decode_failures == 0);
  std::printf("%s\n", g_failures ? "TEST FAILED" : "TEST PASSED");
  return g_failures ? 1 : 0;
}
