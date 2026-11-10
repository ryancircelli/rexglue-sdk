/**
 * rexgpu-shadertool: offline Xenos shader translation, reflection and CPU
 * reference execution for the GPU test harness (tools/gputest).
 *
 * Commands:
 *   translate <ucode> --type vs|ps [--interp-mask HEX] [--out PREFIX]
 *       Analyzes the ucode, translates it with the SDK SPIR-V translator and
 *       writes PREFIX.spv and PREFIX.json (shader info, constant maps,
 *       interpolator linkage, system constant layout, matching DXBC/MSL
 *       modification for the Metal plugin's translator).
 *   interp <ucode> --type vs|ps --consts FILE --inputs FILE --count N
 *          --out FILE [--interp-mask HEX]
 *       Runs the SDK ShaderInterpreter (Xenos semantics on the CPU) over N
 *       invocations and writes all exports. See tools/gputest/README.md for
 *       the binary layouts.
 *
 * Ucode files are little-endian host dwords (as dumped by the Metal plugin's
 * shader dump and rexgpu-metal-msl-xlate) unless --big-endian is given.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <rex/graphics/pipeline/shader/dxbc_translator.h>
#include <rex/graphics/pipeline/shader/interpreter.h>
#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/string/buffer.h>
#include <rex/system/xmemory.h>

using namespace rex;
using namespace rex::graphics;

namespace {

struct Args {
  std::string command;
  std::string ucode_path;
  std::string type = "";
  std::string out;
  std::string consts_path;
  std::string inputs_path;
  uint32_t count = 0;
  int64_t interp_mask = -1;
  bool big_endian = false;
  // Vertex fetch formats are patched into Xbox 360 vertex shaders at bind
  // time (from the vertex declaration), so ucode dumped from the game's
  // shader blobs has format 0 (undefined) in every vfetch. With this flag
  // each undefined vfetch gets a synthetic FLOAT32x4 element: stride 16
  // dwords, element k at offset 4*k, so fetched data is exercised.
  bool patch_vfetch = false;
};

bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(f), {});
  return true;
}

bool WriteFile(const std::string& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char*>(data), std::streamsize(size));
  return bool(f);
}

std::string JsonEscape(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      r += '\\';
      r += c;
    } else if (c == '\n') {
      r += "\\n";
    } else if (uint8_t(c) < 0x20) {
      r += ' ';
    } else {
      r += c;
    }
  }
  return r;
}

// Interpolators the pixel shader reads: Xenos copies interpolants into the
// first registers, bounded by the register count, so the reference linkage is
// "register i gets interpolator i". When not given explicitly, the vertex
// shader uses what it writes and the pixel shader uses all registers below
// min(register count, 16).
uint32_t DefaultInterpolatorMask(const Shader& shader, uint32_t register_count) {
  if (shader.type() == xenos::ShaderType::kVertex) {
    return shader.writes_interpolators();
  }
  uint32_t n = std::min<uint32_t>(register_count, xenos::kMaxInterpolators);
  return n >= 32 ? ~0u : ((1u << n) - 1);
}

// Returns the number of patched vertex fetch instructions.
uint32_t PatchVertexFetches(std::vector<uint32_t>& dwords, bool big_endian) {
  auto rd = [&](size_t i) { return big_endian ? __builtin_bswap32(dwords[i]) : dwords[i]; };
  auto wr = [&](size_t i, uint32_t v) { dwords[i] = big_endian ? __builtin_bswap32(v) : v; };
  if (dwords.size() < 3) {
    return 0;
  }
  // Control flow comes first, two instructions per three dwords; it ends
  // where the first exec's instructions begin.
  uint32_t cf_end_dwords = uint32_t(dwords.size());
  uint32_t patched = 0;
  int32_t element = -1;
  for (uint32_t pair = 0; pair * 3 + 3 <= cf_end_dwords; ++pair) {
    uint32_t raw[3] = {rd(pair * 3), rd(pair * 3 + 1), rd(pair * 3 + 2)};
    ucode::ControlFlowInstruction cf[2];
    ucode::UnpackControlFlowInstructions(raw, cf);
    for (auto& c : cf) {
      if (!ucode::IsControlFlowOpcodeExec(c.opcode())) {
        continue;
      }
      uint32_t address = c.exec.address(), count = c.exec.count();
      uint32_t sequence = c.exec.sequence();
      cf_end_dwords = std::min(cf_end_dwords, address * 3);
      for (uint32_t i = 0; i < count; ++i) {
        if (!((sequence >> (i * 2)) & 1)) {
          continue;  // ALU
        }
        size_t at = size_t(address + i) * 3;
        if (at + 3 > dwords.size()) {
          break;
        }
        uint32_t d0 = rd(at), d1 = rd(at + 1), d2 = rd(at + 2);
        if ((d0 & 0x1F) != 0) {
          continue;  // texture fetch or other
        }
        uint32_t format = (d1 >> 16) & 0x3F;
        if (format != 0) {
          continue;
        }
        ++element;
        uint32_t el = uint32_t(element) & 3;
        d1 = (d1 & ~(0x3Fu << 16)) | (uint32_t(xenos::VertexFormat::k_32_32_32_32_FLOAT) << 16);
        d2 = (d2 & ~0x7FFFFFFFu) | 16u | ((el * 4u) << 8);
        wr(at + 1, d1);
        wr(at + 2, d2);
        ++patched;
      }
    }
  }
  return patched;
}

struct LoadedShader {
  uint32_t patched_vfetches = 0;
  std::vector<uint32_t> dwords;
  std::unique_ptr<SpirvShader> shader;
  string::StringBuffer disasm;
};

bool LoadShader(const Args& args, LoadedShader& ls) {
  std::vector<uint8_t> bytes;
  if (!ReadFile(args.ucode_path, bytes) || bytes.empty() || bytes.size() % 4) {
    fmt::print(stderr, "bad ucode file {}\n", args.ucode_path);
    return false;
  }
  ls.dwords.resize(bytes.size() / 4);
  std::memcpy(ls.dwords.data(), bytes.data(), bytes.size());
  if (args.patch_vfetch) {
    ls.patched_vfetches = PatchVertexFetches(ls.dwords, args.big_endian);
  }
  xenos::ShaderType type =
      args.type == "ps" ? xenos::ShaderType::kPixel : xenos::ShaderType::kVertex;
  ls.shader = std::make_unique<SpirvShader>(
      type, 0, ls.dwords.data(), ls.dwords.size(),
      args.big_endian ? std::endian::big : std::endian::little);
  ls.shader->AnalyzeUcode(ls.disasm);
  return true;
}

// The register count the SPIR-V translator will use is only known after
// translation; the static address bound is a good proxy before it.
uint32_t GuessRegisterCount(const Shader& shader) {
  return std::max<uint32_t>(1, shader.register_static_address_bound());
}

std::string ConstantMapJson(const Shader& shader) {
  const Shader::ConstantRegisterMap& map = shader.constant_register_map();
  std::string s = "{\"float_count\": " + std::to_string(map.float_count) +
                  ", \"float_dynamic_addressing\": " +
                  (map.float_dynamic_addressing ? "true" : "false") + ", \"float_map\": [";
  bool first = true;
  for (uint32_t i = 0; i < 256; ++i) {
    if (!(map.float_bitmap[i >> 6] & (uint64_t(1) << (i & 63)))) {
      continue;
    }
    s += fmt::format("{}[{}, {}]", first ? "" : ", ", i, map.GetPackedFloatConstantIndex(i));
    first = false;
  }
  s += fmt::format("], \"loop_bitmap\": {}, \"bool_bitmap\": [{}, {}, {}, {}, {}, {}, {}, {}]",
                   map.loop_bitmap, map.bool_bitmap[0], map.bool_bitmap[1], map.bool_bitmap[2],
                   map.bool_bitmap[3], map.bool_bitmap[4], map.bool_bitmap[5], map.bool_bitmap[6],
                   map.bool_bitmap[7]);
  s += fmt::format(", \"vertex_fetch_bitmap\": [{}, {}, {}]}}", map.vertex_fetch_bitmap[0],
                   map.vertex_fetch_bitmap[1], map.vertex_fetch_bitmap[2]);
  return s;
}

std::string VertexBindingsJson(const Shader& shader) {
  std::string s = "[";
  bool first_b = true;
  for (const Shader::VertexBinding& b : shader.vertex_bindings()) {
    s += fmt::format("{}{{\"fetch_constant\": {}, \"stride_words\": {}, \"attributes\": [",
                     first_b ? "" : ", ", b.fetch_constant, b.stride_words);
    first_b = false;
    bool first_a = true;
    for (const auto& a : b.attributes) {
      const auto& at = a.fetch_instr.attributes;
      s += fmt::format("{}{{\"format\": {}, \"offset_words\": {}, \"signed\": {}, \"integer\": {}, "
                       "\"exp_adjust\": {}}}",
                       first_a ? "" : ", ", uint32_t(at.data_format), at.offset,
                       at.is_signed ? "true" : "false", at.is_integer ? "true" : "false",
                       at.exp_adjust);
      first_a = false;
    }
    s += "]}";
  }
  return s + "]";
}

#define SYSOFS(field) \
  fmt::format("\"{}\": {}", #field, offsetof(SpirvShaderTranslator::SystemConstants, field))

std::string SpirvSystemLayoutJson() {
  std::vector<std::string> f = {
      SYSOFS(flags),
      SYSOFS(vertex_index_load_address),
      SYSOFS(vertex_index_endian),
      SYSOFS(line_loop_closing_index),
      SYSOFS(vertex_base_index),
      SYSOFS(vertex_index_reset),
      SYSOFS(user_clip_planes),
      SYSOFS(ndc_scale),
      SYSOFS(point_vertex_diameter_min),
      SYSOFS(ndc_offset),
      SYSOFS(point_vertex_diameter_max),
      SYSOFS(point_constant_diameter),
      SYSOFS(point_screen_diameter_to_ndc_radius),
      SYSOFS(textures_resolution_scaled),
      SYSOFS(texture_swizzled_signs),
      SYSOFS(texture_swizzles),
      SYSOFS(alpha_test_reference),
      SYSOFS(alpha_to_mask),
      SYSOFS(color_exp_bias),
      SYSOFS(edram_poly_offset_front_scale),
      SYSOFS(edram_blend_constant),
      SYSOFS(vertex_index_min),
      SYSOFS(vertex_index_max),
  };
  std::string s = fmt::format("{{\"size\": {}", sizeof(SpirvShaderTranslator::SystemConstants));
  for (const auto& e : f) {
    s += ", " + e;
  }
  return s + "}";
}

// Flag bits the harness sets identically in both ABIs: alpha test always
// passes, and the W / divided-by-W conventions.
template <typename T>
std::string SysFlagsJson() {
  return fmt::format(
      "{{\"alpha_always\": {}, \"xy_divided_by_w\": {}, \"z_divided_by_w\": {}, "
      "\"w_not_reciprocal\": {}, \"primitive_polygonal\": {}}}",
      uint32_t(T::kSysFlag_AlphaPassIfLess | T::kSysFlag_AlphaPassIfEqual |
               T::kSysFlag_AlphaPassIfGreater),
      uint32_t(T::kSysFlag_XYDividedByW), uint32_t(T::kSysFlag_ZDividedByW),
      uint32_t(T::kSysFlag_WNotReciprocal), uint32_t(T::kSysFlag_PrimitivePolygonal));
}

std::string DxbcSystemLayoutJson() {
  using SC = DxbcShaderTranslator::SystemConstants;
  return fmt::format(
      "{{\"size\": {}, \"flags\": {}, \"vertex_index_endian\": {}, \"vertex_index_offset\": {}, "
      "\"vertex_index_min\": {}, \"vertex_index_max\": {}, \"ndc_scale\": {}, \"ndc_offset\": {}, "
      "\"color_exp_bias\": {}, \"alpha_test_reference\": {}, \"textures_resolution_scaled\": {}, "
      "\"texture_swizzled_signs\": {}}}",
      sizeof(SC), offsetof(SC, flags), offsetof(SC, vertex_index_endian),
      offsetof(SC, vertex_index_offset), offsetof(SC, vertex_index_min),
      offsetof(SC, vertex_index_max), offsetof(SC, ndc_scale), offsetof(SC, ndc_offset),
      offsetof(SC, color_exp_bias), offsetof(SC, alpha_test_reference),
      offsetof(SC, textures_resolution_scaled), offsetof(SC, texture_swizzled_signs));
}

int CommandTranslate(const Args& args) {
  LoadedShader ls;
  if (!LoadShader(args, ls)) {
    return 1;
  }
  SpirvShader& shader = *ls.shader;
  bool is_vs = shader.type() == xenos::ShaderType::kVertex;
  uint32_t interp_mask = args.interp_mask >= 0
                             ? uint32_t(args.interp_mask)
                             : DefaultInterpolatorMask(shader, GuessRegisterCount(shader));

  SpirvShaderTranslator translator(SpirvShaderTranslator::Features(true), false, false, false);
  // Dynamic addressing needs a register count; use the maximum the guest
  // could have (SQ_PROGRAM_CNTL num_reg is 6 bits).
  uint32_t dyn_regs = shader.GetDynamicAddressableRegisterCount(63);
  SpirvShaderTranslator::Modification mod(
      is_vs ? translator.GetDefaultVertexShaderModification(dyn_regs)
            : translator.GetDefaultPixelShaderModification(dyn_regs));
  if (is_vs) {
    mod.vertex.interpolator_mask = interp_mask & shader.writes_interpolators();
  } else {
    mod.pixel.interpolator_mask = interp_mask;
    mod.pixel.interpolators_centroid = 0;
  }
  Shader::Translation* translation = shader.GetOrCreateTranslation(mod.value);
  bool ok = translator.TranslateAnalyzedShader(*translation) && translation->is_valid();

  DxbcShaderTranslator::Modification dmod(0);
  if (is_vs) {
    dmod.vertex.interpolator_mask = interp_mask & shader.writes_interpolators();
    dmod.vertex.dynamic_addressable_register_count = dyn_regs;
    dmod.vertex.host_vertex_shader_type = Shader::HostVertexShaderType::kVertex;
  } else {
    dmod.pixel.interpolator_mask = interp_mask;
    dmod.pixel.dynamic_addressable_register_count = dyn_regs;
  }

  std::string prefix = args.out.empty() ? args.ucode_path : args.out;
  if (ok) {
    const auto& bin = translation->translated_binary();
    WriteFile(prefix + ".spv", bin.data(), bin.size());
  }
  std::string tex = "[";
  for (size_t i = 0; i < shader.GetTextureBindingsAfterTranslation().size(); ++i) {
    const auto& t = shader.GetTextureBindingsAfterTranslation()[i];
    tex += fmt::format("{}[{}, {}, {}]", i ? ", " : "", uint32_t(t.fetch_constant),
                       uint32_t(t.dimension), uint32_t(t.is_signed));
  }
  tex += "]";
  std::string smp = "[";
  for (size_t i = 0; i < shader.GetSamplerBindingsAfterTranslation().size(); ++i) {
    const auto& t = shader.GetSamplerBindingsAfterTranslation()[i];
    smp += fmt::format("{}[{}, {}, {}, {}]", i ? ", " : "", uint32_t(t.fetch_constant),
                       uint32_t(t.mag_filter), uint32_t(t.min_filter), uint32_t(t.mip_filter));
  }
  smp += "]";
  std::string errors;
  for (const auto& e : translation->errors()) {
    errors += (errors.empty() ? "\"" : ", \"") + JsonEscape(e.message) + "\"";
  }
  std::string json = fmt::format(
      "{{\"ok\": {}, \"type\": \"{}\", \"ucode_dwords\": {}, \"register_static_address_bound\": {}, "
      "\"uses_dynamic_addressing\": {}, \"writes_interpolators\": {}, \"writes_color_targets\": {}, "
      "\"writes_depth\": {}, \"kills_pixels\": {}, \"uses_tfetch_results\": {}, "
      "\"memexport\": {}, \"writes_point_size_edge_flag_kill_vertex\": {}, "
      "\"interpolator_mask\": {}, \"spirv_modification\": \"{:016X}\", "
      "\"dxbc_modification\": \"{:016X}\", \"constants\": {}, \"vertex_bindings\": {}, "
      "\"spirv_textures\": {}, \"spirv_samplers\": {}, \"spirv_system_layout\": {}, "
      "\"dxbc_system_layout\": {}, \"spirv_sys_flags\": {}, \"dxbc_sys_flags\": {}, "
      "\"patched_vfetches\": {}, \"errors\": [{}]}}\n",
      ok ? "true" : "false", is_vs ? "vs" : "ps", ls.dwords.size(),
      shader.register_static_address_bound(),
      shader.uses_register_dynamic_addressing() ? "true" : "false", shader.writes_interpolators(),
      shader.writes_color_targets(), shader.writes_depth() ? "true" : "false",
      shader.kills_pixels() ? "true" : "false",
      shader.uses_texture_fetch_instruction_results() ? "true" : "false",
      shader.memexport_eM_written() ? "true" : "false",
      shader.writes_point_size_edge_flag_kill_vertex(), interp_mask, mod.value, dmod.value,
      ConstantMapJson(shader), VertexBindingsJson(shader), tex, smp, SpirvSystemLayoutJson(),
      DxbcSystemLayoutJson(), SysFlagsJson<SpirvShaderTranslator>(),
      SysFlagsJson<DxbcShaderTranslator>(), ls.patched_vfetches, errors);
  WriteFile(prefix + ".json", json.data(), json.size());
  if (args.patch_vfetch) {
    WriteFile(prefix + ".ucode.bin" + (args.type == "ps" ? ".frag" : ".vert"), ls.dwords.data(),
              ls.dwords.size() * 4);
  }
  if (getenv("SHADERTOOL_DISASM")) {
    std::string d = shader.ucode_disassembly();
    WriteFile(prefix + ".ucode.txt", d.data(), d.size());
  }
  return ok ? 0 : 2;
}

// Constant block layout shared with the Python driver (all little-endian):
//   float   c[256][4]      float constants for this stage (c0 = first)
//   uint32  bool[8]        256 bool constants
//   uint32  loop[32]       loop constants
//   uint32  fetch[96 * 2]  fetch constants (32 x 6 dwords == 96 x 2)
constexpr size_t kConstsSize = 256 * 16 + 8 * 4 + 32 * 4 + 192 * 4;
// Vertex shaders also get guest memory: the inputs file is
//   uint32  memory_address  guest physical address of the blob
//   uint32  memory_size
//   bytes   blob
// Pixel shaders get per-invocation inputs: float r[16][4] interpolants.

class CollectSink : public ShaderInterpreter::ExportSink {
 public:
  float out[64][4];
  uint64_t written = 0;
  void Reset() {
    std::memset(out, 0, sizeof(out));
    written = 0;
  }
  void Export(ucode::ExportRegister reg, const float* value, uint32_t mask) override {
    uint32_t r = uint32_t(reg) & 63;
    for (uint32_t i = 0; i < 4; ++i) {
      if (mask & (1u << i)) {
        out[r][i] = value[i];
      }
    }
    written |= uint64_t(1) << r;
  }
};

int CommandInterp(const Args& args) {
  LoadedShader ls;
  if (!LoadShader(args, ls)) {
    return 1;
  }
  Shader& shader = *ls.shader;
  if (!ShaderInterpreter::CanInterpretShader(shader)) {
    fmt::print(stderr, "shader uses texture fetch results; not interpretable\n");
    return 2;
  }
  std::vector<uint8_t> consts, inputs;
  if (!ReadFile(args.consts_path, consts) || consts.size() < kConstsSize) {
    fmt::print(stderr, "bad consts file\n");
    return 1;
  }
  if (!ReadFile(args.inputs_path, inputs)) {
    fmt::print(stderr, "bad inputs file\n");
    return 1;
  }
  bool is_vs = shader.type() == xenos::ShaderType::kVertex;
  auto regs = std::make_unique<RegisterFile>();
  std::memset(regs->values, 0, sizeof(regs->values));
  uint32_t const_base = is_vs ? 0 : 256;
  std::memcpy(&(*regs)[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * const_base], consts.data(),
              256 * 16);
  std::memcpy(&(*regs)[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], consts.data() + 256 * 16, 32);
  std::memcpy(&(*regs)[XE_GPU_REG_SHADER_CONSTANT_LOOP_00], consts.data() + 256 * 16 + 32, 128);
  std::memcpy(&(*regs)[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], consts.data() + 256 * 16 + 160,
              192 * 4);
  // SQ_VS_CONST / SQ_PS_CONST: base in bits 0-8, size (minus 1) in 12-20.
  (*regs)[XE_GPU_REG_SQ_VS_CONST] = 0 | (255u << 12);
  (*regs)[XE_GPU_REG_SQ_PS_CONST] = 256 | (255u << 12);

  auto memory = std::make_unique<memory::Memory>();
  if (is_vs) {
    if (!memory->Initialize()) {
      fmt::print(stderr, "guest memory init failed\n");
      return 1;
    }
    if (inputs.size() >= 8) {
      uint32_t addr, size;
      std::memcpy(&addr, inputs.data(), 4);
      std::memcpy(&size, inputs.data() + 4, 4);
      size = std::min<uint32_t>(size, uint32_t(inputs.size() - 8));
      std::memcpy(memory->physical_membase() + (addr & 0x1FFFFFFF), inputs.data() + 8, size);
    }
  } else if (inputs.size() < size_t(args.count) * 256) {
    fmt::print(stderr, "pixel inputs too small for count\n");
    return 1;
  }

  ShaderInterpreter interp(*regs, *memory);
  CollectSink sink;
  interp.SetExportSink(&sink);
  interp.SetShader(shader);
  std::vector<float> out(size_t(args.count) * 64 * 4);
  std::vector<uint64_t> written(args.count);
  for (uint32_t n = 0; n < args.count; ++n) {
    float* r = interp.temp_registers();
    std::memset(r, 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
    if (is_vs) {
      // Xenos loads the vertex index into r0.x.
      r[0] = float(n);
    } else {
      std::memcpy(r, inputs.data() + size_t(n) * 256, 256);
    }
    sink.Reset();
    interp.Execute();
    std::memcpy(&out[size_t(n) * 256], sink.out, sizeof(sink.out));
    written[n] = sink.written;
  }
  std::ofstream f(args.out, std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
  f.write(reinterpret_cast<const char*>(written.data()), std::streamsize(written.size() * 8));
  return f ? 0 : 1;
}

void Usage() {
  fmt::print(stderr,
             "usage:\n"
             "  rexgpu-shadertool translate <ucode> --type vs|ps [--interp-mask HEX] "
             "[--out PREFIX] [--big-endian]\n"
             "  rexgpu-shadertool interp <ucode> --type vs|ps --consts F --inputs F --count N "
             "--out F [--big-endian]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    Usage();
    return 1;
  }
  Args args;
  args.command = argv[1];
  args.ucode_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--type") {
      args.type = next();
    } else if (a == "--out") {
      args.out = next();
    } else if (a == "--interp-mask") {
      args.interp_mask = int64_t(std::strtoull(next().c_str(), nullptr, 16));
    } else if (a == "--consts") {
      args.consts_path = next();
    } else if (a == "--inputs") {
      args.inputs_path = next();
    } else if (a == "--count") {
      args.count = uint32_t(std::atoi(next().c_str()));
    } else if (a == "--patch-vfetch") {
      args.patch_vfetch = true;
    } else if (a == "--big-endian") {
      args.big_endian = true;
    } else {
      fmt::print(stderr, "unknown argument {}\n", a);
      return 1;
    }
  }
  if (args.type.empty()) {
    // Infer from the Metal dump naming (.frag = pixel) or a ps_ prefix.
    std::string base = args.ucode_path.substr(args.ucode_path.find_last_of('/') + 1);
    args.type = (args.ucode_path.find(".frag") != std::string::npos || base.rfind("ps_", 0) == 0)
                    ? "ps"
                    : "vs";
  }
  if (args.command == "translate") {
    return CommandTranslate(args);
  }
  if (args.command == "interp") {
    return CommandInterp(args);
  }
  Usage();
  return 1;
}
