// End-to-end known answer: PM4 -> real vfetch/VS -> triangle setup -> PS ->
// shader ROP stub -> NPY/PNG. Ucode bit layout: ucode.h CF exec/alloc and ALU
// export; coverage: raster_top_left_fill (strict ownership of the hypotenuse).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/common.h"
#include "cp/trace_reader.h"

namespace refxenos {
int RunReplay(int, char**);
}
using namespace refxenos;
using namespace rex::graphics;

namespace {
std::vector<uint8_t> bytes;
template <class T>
void Append(const T& x) {
  const auto* p = reinterpret_cast<const uint8_t*>(&x);
  bytes.insert(bytes.end(), p, p + sizeof(x));
}
void Packet(const std::vector<uint32_t>& words) {
  Append(trace::PacketStartCommand{trace::TraceCommandType::kPacketStart, 0x10000,
                                   uint32_t(words.size())});
  for (auto dw : words)
    Append(__builtin_bswap32(dw));
  Append(trace::PacketEndCommand{trace::TraceCommandType::kPacketEnd});
}
void Shader(uint32_t type, std::vector<uint32_t> code) {
  std::vector<uint32_t> words = {0xC0000000u | ((uint32_t(code.size()) + 1) << 16) | (0x2Bu << 8),
                                 type, uint32_t(code.size())};
  words.insert(words.end(), code.begin(), code.end());
  Packet(words);
}
int failures = 0;
#define CHECK(c)                                                       \
  do {                                                                 \
    if (!(c)) {                                                        \
      ++failures;                                                      \
      std::fprintf(stderr, "FAIL replay line %d: %s\n", __LINE__, #c); \
    }                                                                  \
  } while (0)
}  // namespace

int main() {
  const auto dir = std::filesystem::current_path() / "refxenos_test_replay_output";
  std::filesystem::create_directories(dir);
  trace::TraceHeader header{};
  header.version = 1;
  header.title_id = 0x5A5A0001;
  Append(header);
  RegisterFile regs{};
  regs[XE_GPU_REG_RB_SURFACE_INFO] = 8;
  regs[XE_GPU_REG_RB_COLOR_MASK] = 15;
  regs[XE_GPU_REG_RB_MODECONTROL] = uint32_t(xenos::EdramMode::kColorDepth);
  regs[XE_GPU_REG_VGT_MAX_VTX_INDX] = 0xFFFFFF;
  regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL] = 0x80000000;
  regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR] = 8 | (8 << 16);
  regs[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR] = 8 | (8 << 16);
  reg::PA_CL_VTE_CNTL vte{};
  vte.vtx_w0_fmt = 1;
  regs[vte.register_index] = vte.value;
  reg::PA_CL_CLIP_CNTL clip{};
  clip.clip_disable = 1;
  regs[clip.register_index] = clip.value;
  reg::PA_SU_VTX_CNTL vtx{};
  vtx.pix_center = xenos::PixelCenter::kOGLHalf;
  regs[vtx.register_index] = vtx.value;
  regs[XE_GPU_REG_SQ_VS_CONST] = 255u << 12;
  regs[XE_GPU_REG_SQ_PS_CONST] = 256u | (255u << 12);
  regs[XE_GPU_REG_SHADER_CONSTANT_256_X] = AsU32(1.0f);
  regs[XE_GPU_REG_SHADER_CONSTANT_256_W] = AsU32(1.0f);
  regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0] = 0x100003;
  regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_1] = 2 | (12 << 2);
  trace::RegistersCommand rc{};
  rc.type = trace::TraceCommandType::kRegisters;
  rc.register_count = RegisterFile::kRegisterCount;
  rc.encoded_length = rc.register_count * 4;
  Append(rc);
  const auto* raw = reinterpret_cast<const uint8_t*>(regs.values);
  bytes.insert(bytes.end(), raw, raw + rc.encoded_length);
  std::vector<uint32_t> vertex;
  for (const auto& v :
       std::vector<std::vector<float>>{{0, 0, .5f, 1}, {8, 0, .5f, 1}, {0, 8, .5f, 1}})
    for (float f : v)
      vertex.push_back(__builtin_bswap32(AsU32(f)));
  trace::MemoryCommand mc{};
  mc.type = trace::TraceCommandType::kMemoryRead;
  mc.base_ptr = 0x100000;
  mc.encoded_length = mc.decoded_length = vertex.size() * 4;
  Append(mc);
  raw = reinterpret_cast<const uint8_t*>(vertex.data());
  bytes.insert(bytes.end(), raw, raw + mc.encoded_length);
  Shader(0, {0x11002, 0x1200, 0xc2000000, 0x1003, 0x2200, 0, 0x80000, 0x260688, 4, 0xc80f803e, 0,
             0xc2000000});
  Shader(1, {0, 0x1001c400, 0x22000000, 0xc80f8000, 0, 0x02000000});
  Packet({0xC0003600, 4 | (2 << 6) | (3 << 16)});
  Append(trace::EventCommand{trace::TraceCommandType::kEvent, trace::EventCommand::Type::kSwap});
  const auto path = dir / "triangle.xtr";
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  std::vector<std::string> args = {"replay",  path.string(), "-o",    (dir / "render").string(),
                                   "--draws", "0:1",         "--full"};
  std::vector<char*> argv;
  for (auto& s : args)
    argv.push_back(s.data());
  CHECK(RunReplay(int(argv.size()), argv.data()) == 0);
  std::ifstream f(dir / "render/draws/000000_rt0.npy", std::ios::binary);
  std::vector<uint8_t> npy((std::istreambuf_iterator<char>(f)), {});
  CHECK(npy.size() >= 10 + 8 * 8 * 16);
  if (npy.size() >= 10 + 8 * 8 * 16) {
    uint32_t offset = 10 + npy[8] + (uint32_t(npy[9]) << 8), covered = 0;
    CHECK(npy.size() == offset + 8 * 8 * 16);
    if (npy.size() == offset + 8 * 8 * 16)
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          uint32_t rgba[4];
          std::memcpy(rgba, npy.data() + offset + (y * 8 + x) * 16, 16);
          bool inside = x + y < 7;
          covered += inside;
          CHECK(rgba[0] == (inside ? AsU32(1) : 0));
          CHECK(rgba[1] == 0);
          CHECK(rgba[2] == 0);
          CHECK(rgba[3] == (inside ? AsU32(1) : 0));
        }
    CHECK(covered == 28);
  }
  CHECK(std::filesystem::file_size(dir / "render/000000_rt0.png") > 20);
  std::ifstream records(dir / "render/draws.jsonl");
  std::string record((std::istreambuf_iterator<char>(records)), {});
  CHECK(record.find("\"quads\":10") != record.npos);
  CHECK(record.find("cp_draw_packet@") != record.npos);
  CHECK(record.find("diagnostic shader ROP stub") != record.npos);
  std::printf("replay integration: %d failures\n", failures);
  return failures ? 1 : 0;
}
