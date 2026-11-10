// refxenos shader job runner: the same job inputs and output format as
// `rexgpu-shadertool interp` (tools/gputest/shadertool/shadertool.cpp
// CommandInterp), so tools/gputest/shaderdiff.py can run the refxenos
// interpreter as an arm.
//
//   refxenos_test_shaderjob interp <ucode> --type vs|ps --consts F --inputs F
//       --count N --out F [--big-endian] [--patch-vfetch] [--taint-out F]
//       [--allow-tfetch] [--mantissa-bits 21|23] [--notes]
//
// Output: count x float[64][4] exports, then count x uint64 written masks.
// --taint-out writes count x uint8[64][4] transcendental taint counts.
// Pixel invocations run as quads of 4 consecutive invocations (no texture
// fetch -> no cross-lane dependence). Exit 2 if the shader uses texture
// fetch results (unless --allow-tfetch: coordinates are returned).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <rex/graphics/format/ucode.h>

#include "core/common.h"
#include "shader/interp.h"

using namespace refxenos;
using namespace rex::graphics;  // XE_GPU_REG_* register indices
namespace ucode = rex::graphics::ucode;

namespace {

bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

// Same as shadertool.cpp PatchVertexFetches (k-th format-0 vfetch becomes a
// 32_32_32_32_FLOAT element at offset 4k, stride 16 dwords).
void PatchVertexFetches(std::vector<uint32_t>& dwords) {
  if (dwords.size() < 3) return;
  uint32_t cf_end = uint32_t(dwords.size());
  int32_t element = -1;
  for (uint32_t pair = 0; pair * 3 + 3 <= cf_end; ++pair) {
    ucode::ControlFlowInstruction cf[2];
    ucode::UnpackControlFlowInstructions(dwords.data() + pair * 3, cf);
    for (auto& c : cf) {
      if (!ucode::IsControlFlowOpcodeExec(c.opcode())) continue;
      uint32_t address = c.exec.address(), count = c.exec.count(), seq = c.exec.sequence();
      cf_end = std::min(cf_end, address * 3);
      for (uint32_t i = 0; i < count; ++i) {
        if (!((seq >> (i * 2)) & 1)) continue;
        size_t at = size_t(address + i) * 3;
        if (at + 3 > dwords.size()) break;
        uint32_t d0 = dwords[at], d1 = dwords[at + 1], d2 = dwords[at + 2];
        if ((d0 & 0x1F) != 0) continue;
        if (((d1 >> 16) & 0x3F) != 0) continue;
        ++element;
        uint32_t el = uint32_t(element) & 3;
        d1 = (d1 & ~(0x3Fu << 16)) | (38u << 16);
        d2 = (d2 & ~0x7FFFFFFFu) | 16u | ((el * 4u) << 8);
        dwords[at + 1] = d1;
        dwords[at + 2] = d2;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || std::string(argv[1]) != "interp") {
    std::fprintf(stderr,
                 "usage: %s interp <ucode> --type vs|ps --consts F --inputs F --count N --out F "
                 "[--big-endian] [--patch-vfetch] [--taint-out F] [--allow-tfetch] "
                 "[--mantissa-bits N] [--notes] [--rules-out F]\n",
                 argv[0]);
    return 1;
  }
  std::string ucode_path = argv[2], type = "vs", consts_path, inputs_path, out_path, taint_path, rules_path;
  uint32_t count = 0;
  bool big_endian = false, patch = false, allow_tfetch = false, print_notes = false;
  InterpOptions opts = GetInterpOptions();
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--type") type = next();
    else if (a == "--consts") consts_path = next();
    else if (a == "--inputs") inputs_path = next();
    else if (a == "--count") count = uint32_t(std::stoul(next()));
    else if (a == "--out") out_path = next();
    else if (a == "--taint-out") taint_path = next();
    else if (a == "--rules-out") rules_path = next();
    else if (a == "--big-endian") big_endian = true;
    else if (a == "--patch-vfetch") patch = true;
    else if (a == "--allow-tfetch") allow_tfetch = true;
    else if (a == "--notes") print_notes = true;
    else if (a == "--mantissa-bits") opts.transcendental_mantissa_bits = uint32_t(std::stoul(next()));
    else if (a == "--interp-mask") next();
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 1;
    }
  }
  SetInterpOptions(opts);

  std::vector<uint8_t> bytes;
  if (!ReadFile(ucode_path, bytes) || bytes.empty() || bytes.size() % 4) {
    std::fprintf(stderr, "bad ucode file %s\n", ucode_path.c_str());
    return 1;
  }
  ShaderProgram prog;
  prog.type = type == "ps" ? xenos::ShaderType::kPixel : xenos::ShaderType::kVertex;
  prog.ucode.resize(bytes.size() / 4);
  std::memcpy(prog.ucode.data(), bytes.data(), bytes.size());
  if (big_endian)
    for (auto& d : prog.ucode) d = __builtin_bswap32(d);
  if (patch) PatchVertexFetches(prog.ucode);
  if (!allow_tfetch && ShaderUsesTextureFetchResults(prog)) {
    std::fprintf(stderr, "shader uses texture fetch results; not interpretable\n");
    return 2;
  }

  std::vector<uint8_t> consts, inputs;
  constexpr size_t kConstsSize = 256 * 16 + 8 * 4 + 32 * 4 + 192 * 4;
  if (!ReadFile(consts_path, consts) || consts.size() < kConstsSize) {
    std::fprintf(stderr, "bad consts file\n");
    return 1;
  }
  if (!ReadFile(inputs_path, inputs)) {
    std::fprintf(stderr, "bad inputs file\n");
    return 1;
  }
  bool is_vs = prog.type == xenos::ShaderType::kVertex;
  auto regs = std::make_unique<RegisterFile>();
  std::memset(regs->values, 0, sizeof(regs->values));
  uint32_t const_base = is_vs ? 0 : 256;
  std::memcpy(&regs->values[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * const_base], consts.data(), 256 * 16);
  std::memcpy(&regs->values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], consts.data() + 4096, 32);
  std::memcpy(&regs->values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00], consts.data() + 4128, 128);
  std::memcpy(&regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], consts.data() + 4256, 768);
  regs->values[XE_GPU_REG_SQ_VS_CONST] = 0 | (255u << 12);
  regs->values[XE_GPU_REG_SQ_PS_CONST] = 256 | (255u << 12);

  std::unique_ptr<Memory> memory;
  if (is_vs) {
    memory = std::make_unique<Memory>();
    if (inputs.size() >= 8) {
      uint32_t addr, size;
      std::memcpy(&addr, inputs.data(), 4);
      std::memcpy(&size, inputs.data() + 4, 4);
      size = std::min<uint32_t>(size, uint32_t(inputs.size() - 8));
      memory->Write(addr & 0x1FFFFFFF, inputs.data() + 8, size);
    }
  } else if (inputs.size() < size_t(count) * 256) {
    std::fprintf(stderr, "pixel inputs too small for count\n");
    return 1;
  }

  std::vector<float> out(size_t(count) * 256, 0.0f);
  std::vector<uint64_t> written(count, 0);
  std::vector<uint8_t> taint(size_t(count) * 256, 0);
  RuleUsage::Current().Clear();
  DrawStats stats;
  CoordTextureSampler sampler;
  TextureSampler* smp = allow_tfetch ? &sampler : nullptr;
  bool all_ok = true;
  auto store = [&](uint32_t n, const RawExports& e) {
    std::memcpy(&out[size_t(n) * 256], e.reg, sizeof(e.reg));
    std::memcpy(&taint[size_t(n) * 256], e.taint, sizeof(e.taint));
    written[n] = e.written;
  };
  if (is_vs) {
    auto e = std::make_unique<RawExports>();
    for (uint32_t n = 0; n < count; ++n) {
      all_ok &= RunVertexShaderRaw(prog, *regs, *memory, n, smp, e.get(), stats, nullptr);
      store(n, *e);
    }
  } else {
    auto e = std::make_unique<RawExports[]>(4);
    auto in = std::make_unique<float[][16][4]>(4);
    for (uint32_t q = 0; q * 4 < count; ++q) {
      PixelQuad quad{};
      quad.x = int32_t((q % 64) * 2);
      quad.y = int32_t((q / 64) * 2);
      for (uint32_t l = 0; l < 4; ++l) {
        uint32_t n = q * 4 + l;
        quad.coverage[l] = n < count ? 1 : 0;
        if (n < count) std::memcpy(in[l], inputs.data() + size_t(n) * 256, 256);
        else std::memset(in[l], 0, 256);
      }
      all_ok &= RunPixelQuadRaw(prog, *regs, quad, in.get(), smp, e.get(), stats, nullptr);
      for (uint32_t l = 0; l < 4 && q * 4 + l < count; ++l) store(q * 4 + l, e[l]);
    }
  }
  if (print_notes || !all_ok) {
    for (const auto& n : stats.notes) std::fprintf(stderr, "note: %s\n", n.c_str());
  }
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
  f.write(reinterpret_cast<const char*>(written.data()), std::streamsize(written.size() * 8));
  if (!taint_path.empty()) {
    std::ofstream t(taint_path, std::ios::binary);
    t.write(reinterpret_cast<const char*>(taint.data()), std::streamsize(taint.size()));
  }
  if (!rules_path.empty()) {
    // {"rule_id": version, ...} for every rule applied in this job.
    std::ofstream r(rules_path);
    r << "{";
    bool first = true;
    for (const Rule* rule : RuleUsage::Current().used()) {
      r << (first ? "" : ", ") << "\"" << rule->id << "\": " << rule->version;
      first = false;
    }
    r << "}\n";
  }
  return f ? 0 : 1;
}
