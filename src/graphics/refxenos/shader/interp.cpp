// refxenos shader interpreter: control flow, lanes, operands, exports,
// fetch front ends, trace.

#include "shader/interp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include <rex/math.h>

#include "shader/internal.h"

namespace refxenos {

using namespace shader_internal;

// ---------------------------------------------------------------------------
// Rules

RX_DEFINE_RULE(xenos_cf_pair_layout, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:515 "
               "(UnpackControlFlowInstructions: two 48-bit CF instructions per "
               "three dwords at the start of the program)",
               false, "");
RX_DEFINE_RULE(xenos_cf_exec, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.cc:60 "
               "(exec runs count instructions from address; sequence bit 2i set = "
               "fetch; cond_exec tests a bool constant bit, cond_exec_pred the "
               "predicate; *_end ends the shader only when the exec is taken)",
               false, "");
RX_DEFINE_RULE(xenos_cf_pred_clean, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_translator.cc:774 "
               "(cond_exec_pred_clean(_end) is a bool constant cond_exec; the "
               "\"clean\" flag is a predicate-clean hint without effect on results)",
               false, "");
RX_DEFINE_RULE(xenos_cf_loop, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator.cc:1888 "
               "(loop_start: count = loop constant bits 0:7, count 0 skips to "
               "address without pushing; aL = start (bits 8:15) or, is_repeat, "
               "kept from the outer loop; loop_end: count -= 1, break when 0 or "
               "(predicated break and p0 == condition), else aL += step (signed "
               "bits 16:23) and jump to address)",
               false, "");
RX_DEFINE_RULE(xenos_cf_loop_depth, 1,
               "assumed: loop and call stacks are 4 deep (xenia-canary@cf206a3b "
               "src/xenia/gpu/shader_interpreter.cc:125, :173); a loop_start or "
               "call beyond depth 4 is skipped, ret at depth 0 is a no-op "
               "(XDK microcode_sq_ret.htm)",
               true,
               "VS: 5 nested loops each count 1 incrementing r0.x by 1 inside the "
               "innermost; 5 nested calls each adding 1; export r0 to o0 and read "
               "it back through a PS to 32_32_32_32_FLOAT");
RX_DEFINE_RULE(xenos_cf_aL_clamp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.h:88 "
               "(aL = start + iterator*step clamped to [-256, 256], IPR2015-00325 "
               "sequencer specification)",
               false, "");
RX_DEFINE_RULE(xenos_cf_call_jmp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.cc:166 "
               "(call/jmp: unconditional, predicated (p0 == condition) or bool "
               "constant; call pushes the next CF index)",
               false, "");
RX_DEFINE_RULE(xenos_cf_lanes_independent, 1,
               "assumed: refxenos runs each pixel of a quad with its own control "
               "flow (lead requirement). XDK microcode_predicate_logic.htm says a "
               "predicated exec/call runs if ANY thread of the 64-wide vector "
               "passes and a predicated jmp/endloop break needs ALL; lane results "
               "are identical when each lane only executes what applies to it",
               true,
               "PS: pixels of one quad take different branches on a predicate from "
               "the pixel X coordinate; each branch writes a distinct constant to "
               "oC0; read back all 4 pixels");
RX_DEFINE_RULE(xenos_alu_operand_swizzle, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:930 (ALU component j "
               "= ((swizzle >> 2j) + j) & 3; abs applied before negate (XDK "
               "microcode_registers_source_modifiers.htm); scalar a = W, b = X, "
               "or Z when the vector op has 3 operands (shader_translator.cc:1424))",
               false, "");
RX_DEFINE_RULE(xenos_alu_scalar_const_temp, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_translator.cc:1433 "
               "(mulsc/addsc/subsc: a = c[src3].W swizzled, b = r[temp].X "
               "swizzled; src3 negate and abs_constants apply to both)",
               false, "");
RX_DEFINE_RULE(xenos_alu_const_addressing, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.cc:255 "
               "(index + (a0 or aL); index < 0 or > SQ_*_CONST.size reads 0; "
               "+ base; >= 512 reads 0); per-lane a0 makes constant "
               "waterfalling transparent",
               false, "");
RX_DEFINE_RULE(xenos_alu_const_oob_zero, 1,
               "assumed: a float constant read whose index (after a0/aL) is < 0, "
               "> SQ_*_CONST.size or whose final index is >= 512 returns (0, 0, 0, 0) "
               "(XDK leaves out-of-range constant reads unspecified; Xenia "
               "shader_interpreter.cc:255 returns 0)",
               true,
               "VS or PS, c_i = i + 1000 for all 256 constants of the stage, "
               "SQ_*_CONST.size = 255: for a0 in {-1, -256} (maxas a0, c0.x with "
               "c0.x = -1.0 / -256.0) read c[a0 + 255] and c[a0 + 0] and export "
               "them to a 32_32_32_32_FLOAT target; read back the 4 floats for each "
               "(0 if out-of-range reads give 0; c[254] = 1254 for a0 = -1, k = 255)");
RX_DEFINE_RULE(xenos_alu_temp_relative_oob, 1,
               "assumed: relative temp index outside [0, 63] reads 0 and drops "
               "writes (D3D indexable temp out of range, as Xenia's x# array)",
               true,
               "PS: loop with aL from -2 to 66 reading r[aL] into an accumulator "
               "with known r# contents; export to oC0, read back");
RX_DEFINE_RULE(xenos_alu_masks, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:1978 (vector mask, "
               "scalar mask; exports: overlapping vector+scalar components write "
               "1, scalar-dest-relative exports write 0 to unmasked components; "
               "vector stored before scalar, shader_translator.cc:1355)",
               false, "");
RX_DEFINE_RULE(xenos_alu_predication, 1,
               "XDK microcode_predicate_logic.htm; xenia-canary@cf206a3b "
               "src/xenia/gpu/shader_interpreter.cc:110 (a predicated ALU/fetch "
               "whose condition != p0 is skipped entirely: no ps, p0, a0, kill)",
               false, "");
RX_DEFINE_RULE(xenos_alu_vector_then_scalar, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:1162 "
               "(both ops read sources first; vector result then scalar result "
               "stored; scalar p0/a0 updates win)",
               false, "");
RX_DEFINE_RULE(xenos_alu_ps_unsaturated, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.cc:960 and "
               "src/xenia/gpu/dxbc_shader_translator_alu.cc:1196 (the previous "
               "scalar register keeps the unsaturated result; scalar_clamp only "
               "affects the stored destination)",
               false, "");
RX_DEFINE_RULE(xenos_kill_continues, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_alu.cc:27 "
               "(a killed pixel keeps executing; its exports are discarded); XDK "
               "microcode_alu_vector_kill_eq.htm (kill cannot be undone)",
               false, "");
RX_DEFINE_RULE(xenos_vs_r0_index, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator.cc:360 "
               "(all GPRs zeroed, r0.x = float(vertex index) (unsigned); the "
               "index offset / 24-bit mask / clamp are applied by the caller)",
               false, "");
RX_DEFINE_RULE(xenos_ps_inputs, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator.cc:700 "
               "(r_i = interpolator i for i < 16 written by the VS, other GPRs 0; "
               "the rasterizer supplies 0 for interpolators the VS did not "
               "export; ps_param_gen goes to r[SQ_CONTEXT_MISC.param_gen_pos] when "
               "SQ_PROGRAM_CNTL.param_gen and the position < register count)",
               false, "");
RX_DEFINE_RULE(xenos_ps_helper_lanes, 1,
               "assumed: helper pixels (coverage 0) run the whole shader and supply "
               "gradients; their exports are discarded (D3D quad semantics; "
               "Xenia runs host helper invocations)",
               true,
               "PS with getGradients on the interpolated X into oC0 on a triangle "
               "covering 1 pixel of a quad; read the covered pixel");
RX_DEFINE_RULE(xenos_vfetch_dest_swizzle, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:671 (fetch destination "
               "swizzle per component: 0-3 result, 4 = 0, 5 = 1, 7 = keep; 6 "
               "treated as 0 like the SDK interpreter (assumed))",
               false, "");
RX_DEFINE_RULE(xenos_fetch_source_swizzle, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_translator.cc:1026,1137 "
               "(fetch source component j = (swizzle >> 2j) & 3, not "
               "ALU-relative; source register + aL when relative)",
               false, "");
RX_DEFINE_RULE(tfetch_gradients_coarse, 1,
               "assumed: implicit gradients are coarse per quad: ddx = TR - TL, "
               "ddy = BL - TL for all 4 pixels (XDK microcode_fetch_getGradients.htm "
               "\"only the top-left, top-right and bottom-left pixels\"; "
               "xenia-canary@cf206a3b src/xenia/gpu/dxbc_shader_translator_fetch.cc:1822 "
               "deriv_rtx_coarse); lanes that diverged or ended contribute their "
               "current register values",
               true,
               "PS: tfetch2D on a 2x2 quad whose coordinates are (x*x, y*y) per pixel "
               "on a mip chain with distinct colours per level; read all 4 pixels, "
               "also with one pixel branching past the coordinate computation");
RX_DEFINE_RULE(tfetch_getgradients, 1,
               "XDK microcode_fetch_getGradients.htm (x = ddx(src.x), y = "
               "ddy(src.x), z = ddx(src.y), w = ddy(src.y), same result in all "
               "pixels; difference rounded to fp16 and expanded); "
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:640",
               false, "");
RX_DEFINE_RULE(tfetch_gradient_fp16, 1,
               "assumed: XDK microcode_fetch_getGradients.htm says the f32 difference "
               "is \"truncated to fit into a 16-bit floating point value\"; taken "
               "as round toward zero to IEEE half (overflow -> max half, "
               "denormal halves kept); Xenia does not do this",
               true,
               "PS: getGradients of an interpolant with per-pixel deltas 1/3, 1e-6, "
               "70000, -1/3; write to a 32_32_32_32_FLOAT target and read bits");
RX_DEFINE_RULE(tfetch_register_gradients_shared, 1,
               "XDK microcode_fetch_setGradientH.htm (one gradient register set per "
               "quad, taken from the first live pixel in TL, TR, BL, BR order; "
               "values truncated to fp16 (tfetch_gradient_fp16))",
               false, "");
RX_DEFINE_RULE(tfetch_lod_fields, 1,
               "XDK microcode_fetch_tfetch2D.htm (LOD = computed (UseComputedLOD) + "
               "register (UseRegisterLOD) + LODBias + fetch constant bias); "
               "xenia-canary@cf206a3b src/xenia/gpu/spirv_shader_translator_fetch.cc:666 "
               "(computed LOD only in pixel shaders or with register gradients)",
               false, "");
RX_DEFINE_RULE(tfetch_vs_zero_gradients, 1,
               "assumed: vertex shader implicit-LOD fetches have zero gradients "
               "(no quads); the computed-LOD term is dropped "
               "(spirv_shader_translator_fetch.cc:666)",
               true,
               "VS: tfetch2D with UseComputedLOD from a mip chain with distinct "
               "level colours; export to an interpolator; read back via PS");
RX_DEFINE_RULE(tfetch_cube_coords, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:560 (tfetchCube takes "
               "SC, TC in [1, 2) and the face index from the cube ALU sequence as "
               "x, y, z; register gradients are in cube space)",
               false, "");
RX_DEFINE_RULE(tfetch_predicated_skip, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/shader_interpreter.cc:96 "
               "(predicated fetch with condition != p0 does nothing); XDK "
               "microcode_predicate_logic.htm",
               false, "");
RX_DEFINE_RULE(xenos_memexport_recorded, 1,
               "xenia-canary@cf206a3b src/xenia/gpu/ucode.h:1819 (eA = export 32, "
               "eM0-4 = 33-37); refxenos records memexport writes in DrawStats "
               "notes and does not write guest memory",
               false, "");
RX_DEFINE_RULE(xenos_transcendental_taint, 1,
               "refxenos bookkeeping (lead): per component count of transcendental "
               "ops on the dataflow path, max over inputs + 1, capped at 255",
               false, "");

// ---------------------------------------------------------------------------
// Options

static InterpOptions g_options;
void SetInterpOptions(const InterpOptions& o) { g_options = o; }
const InterpOptions& GetInterpOptions() { return g_options; }

namespace {

uint8_t TaintAdd(uint8_t t, uint32_t add) {
  uint32_t v = uint32_t(t) + add;
  return uint8_t(v > 255 ? 255 : v);
}

struct Lane {
  float r[64][4];
  uint8_t rt[64][4];
  float ps = 0.0f;
  uint8_t ps_t = 0;
  bool p = false;
  uint8_t p_t = 0;
  int32_t a0 = 0;
  uint8_t a0_t = 0;

  uint32_t cf = 0;
  bool ended = false;
  uint64_t steps = 0;

  int loop_depth = 0;
  uint32_t loop_count[4] = {};
  int32_t loop_step[4] = {};
  int32_t aL[4] = {};
  int call_depth = 0;
  uint32_t call_ret[4] = {};

  bool killed = false;

  ucode::VertexFetchInstruction last_full{};
  VertexFetchAddress last_addr;

  float tex_lod = 0.0f;
  uint8_t tex_lod_t = 0;
  float reg_grad_h[3] = {}, reg_grad_v[3] = {};
  uint8_t reg_grad_t = 0;

  RawExports* exports = nullptr;
  bool helper = false;
  bool trace = false;
  int32_t x = 0, y = 0;

  int32_t GetAL() const {
    RX_USE(xenos_cf_aL_clamp);
    if (!loop_depth) return 0;
    return std::clamp(aL[loop_depth - 1], -256, 256);
  }
};

struct Ctx {
  const ShaderProgram* prog = nullptr;
  const RegisterFile* regs = nullptr;
  const Memory* memory = nullptr;
  TextureSampler* sampler = nullptr;
  DrawStats* stats = nullptr;
  Probe* probe = nullptr;
  bool is_ps = false;
  Lane* lanes = nullptr;
  int lane_count = 0;
  uint32_t const_base = 0, const_size_minus_1 = 0;
  uint32_t memexport_writes = 0;
  bool ok = true;
  std::vector<std::string> notes;
  uint32_t vertex_index = 0;

  void Note(const std::string& s) {
    if (std::find(notes.begin(), notes.end(), s) == notes.end()) notes.push_back(s);
  }
};

void Trace(Ctx& ctx, Lane& lane, const std::string& text) {
  if (!lane.trace || !ctx.probe) return;
  if (ctx.is_ps) {
    ctx.probe->Log(lane.x, lane.y, "ps", text);
  } else {
    ctx.probe->LogVertex(ctx.vertex_index, "vs", text);
  }
}

std::string Vec4Str(const float v[4]) {
  std::string s = "(";
  for (int i = 0; i < 4; ++i) {
    if (i) s += ", ";
    s += FormatFloat(v[i]);
  }
  return s + ")";
}

// --- Program access ---

bool GetCf(const ShaderProgram& prog, uint32_t cf, ucode::ControlFlowInstruction* out) {
  RX_USE(xenos_cf_pair_layout);
  size_t base = size_t(cf >> 1) * 3;
  if (base + 3 > prog.ucode.size()) return false;
  ucode::ControlFlowInstruction pair[2];
  ucode::UnpackControlFlowInstructions(prog.ucode.data() + base, pair);
  *out = pair[cf & 1];
  return true;
}

const uint32_t* GetInstr(const ShaderProgram& prog, uint32_t address) {
  size_t base = size_t(address) * 3;
  if (base + 3 > prog.ucode.size()) return nullptr;
  return prog.ucode.data() + base;
}

bool BoolConstant(const RegisterFile& regs, uint32_t bool_address) {
  uint32_t w = regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 + (bool_address >> 5)];
  return (w >> (bool_address & 31)) & 1;
}

// --- Operands ---

bool ReadTemp(const Lane& lane, int32_t idx, float v[4], uint8_t t[4]) {
  if (idx < 0 || idx > 63) {
    RX_USE(xenos_alu_temp_relative_oob);
    for (int i = 0; i < 4; ++i) { v[i] = 0.0f; t[i] = 0; }
    return false;
  }
  for (int i = 0; i < 4; ++i) { v[i] = lane.r[idx][i]; t[i] = lane.rt[idx][i]; }
  return true;
}

void ReadConst(const Ctx& ctx, int32_t idx, float v[4]) {
  RX_USE(xenos_alu_const_addressing);
  for (int i = 0; i < 4; ++i) v[i] = 0.0f;
  if (idx < 0 || uint32_t(idx) > ctx.const_size_minus_1) { RX_USE(xenos_alu_const_oob_zero); return; }
  uint32_t full = uint32_t(idx) + ctx.const_base;
  if (full >= 512) { RX_USE(xenos_alu_const_oob_zero); return; }
  std::memcpy(v, &ctx.regs->values[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * full], 16);
}

// Reads the raw (unswizzled) register of ALU source i and its abs flag.
void ReadAluSourceRegister(const Ctx& ctx, const Lane& lane,
                           const ucode::AluInstruction& op, uint32_t i,
                           float v[4], uint8_t t[4], bool* abs,
                           std::string* name) {
  uint32_t reg = op.src_reg(i);
  if (op.src_is_temp(i)) {
    int32_t idx = int32_t(reg & 0x3F);
    bool rel = (reg & 0x40) != 0;
    if (rel) idx += lane.GetAL();
    *abs = (reg & 0x80) != 0;
    ReadTemp(lane, idx, v, t);
    if (name) *name = (rel ? "r[aL+" + std::to_string(reg & 0x3F) + "]" : "r" + std::to_string(reg & 0x3F));
  } else {
    int32_t idx = int32_t(reg);
    uint8_t at = 0;
    if (op.src_const_is_addressed(i)) {
      if (op.is_const_address_register_relative()) {
        idx += lane.a0;
        at = lane.a0_t;
      } else {
        idx += lane.GetAL();
      }
    }
    *abs = op.abs_constants();
    ReadConst(ctx, idx, v);
    for (int k = 0; k < 4; ++k) t[k] = at;
    if (name) {
      *name = op.src_const_is_addressed(i)
                  ? std::string("c[") + (op.is_const_address_register_relative() ? "a0" : "aL") + "+" + std::to_string(reg) + "]"
                  : "c" + std::to_string(reg);
    }
  }
}

float ApplyMods(float x, bool abs, bool neg) {
  if (abs) x = std::fabs(x);
  if (neg) x = -x;
  return FlushDenormal(x);
}

void ReadVectorOperand(const Ctx& ctx, const Lane& lane, const ucode::AluInstruction& op,
                       uint32_t i, float out[4], uint8_t tout[4], std::string* name) {
  RX_USE(xenos_alu_operand_swizzle);
  float v[4];
  uint8_t t[4];
  bool abs;
  ReadAluSourceRegister(ctx, lane, op, i, v, t, &abs, name);
  uint32_t swz = op.src_swizzle(i);
  bool neg = op.src_negate(i);
  for (uint32_t j = 0; j < 4; ++j) {
    uint32_t c = ((swz >> (2 * j)) + j) & 3;
    out[j] = ApplyMods(v[c], abs, neg);
    tout[j] = t[c];
  }
}

// --- Writes ---

void WriteTemp(Lane& lane, int32_t idx, uint32_t mask, const float v[4],
               const uint8_t t[4]) {
  if (idx < 0 || idx > 63) {
    RX_USE(xenos_alu_temp_relative_oob);
    return;
  }
  for (int i = 0; i < 4; ++i) {
    if (mask & (1u << i)) {
      lane.r[idx][i] = v[i];
      lane.rt[idx][i] = t[i];
    }
  }
}

void WriteExport(Ctx& ctx, Lane& lane, uint32_t reg, uint32_t mask,
                 const float v[4], const uint8_t t[4]) {
  if (!mask) return;
  reg &= 63;
  if (reg >= 32 && reg <= 37) {
    RX_USE(xenos_memexport_recorded);
    ++ctx.memexport_writes;
    if (lane.trace) {
      Trace(ctx, lane, "memexport " + std::string(reg == 32 ? "eA" : "eM" + std::to_string(reg - 33)) +
                           " = " + Vec4Str(v));
    }
  }
  RawExports& e = *lane.exports;
  for (int i = 0; i < 4; ++i) {
    if (mask & (1u << i)) {
      e.reg[reg][i] = v[i];
      e.taint[reg][i] = t[i];
      e.comp_written[reg] |= uint8_t(1u << i);
    }
  }
  e.written |= uint64_t(1) << reg;
}

// --- ALU instruction ---

void ExecuteAlu(Ctx& ctx, Lane& lane, const uint32_t* dwords) {
  const auto& op = *reinterpret_cast<const ucode::AluInstruction*>(dwords);
  if (op.is_predicated() && op.predicate_condition() != lane.p) {
    RX_USE(xenos_alu_predication);
    if (lane.trace) Trace(ctx, lane, DisassembleAlu(dwords) + "  ; skipped (predicate)");
    return;
  }
  uint8_t pred_t = op.is_predicated() ? lane.p_t : 0;
  using V = ucode::AluVectorOpcode;
  using S = ucode::AluScalarOpcode;
  V vop = op.vector_opcode();
  S sop = op.scalar_opcode();
  const auto& vinfo = ucode::GetAluVectorOpcodeInfo(vop);
  const auto& sinfo = ucode::GetAluScalarOpcodeInfo(sop);
  uint32_t vcount = vinfo.GetOperandCount();

  // Vector operands.
  float vs[3][4] = {};
  uint8_t vt[3][4] = {};
  std::string vnames[3];
  for (uint32_t i = 0; i < vcount; ++i) {
    ReadVectorOperand(ctx, lane, op, i + 1, vs[i], vt[i], lane.trace ? &vnames[i] : nullptr);
  }
  // Scalar operands.
  float sa = 0.0f, sb = 0.0f;
  uint8_t sa_t = 0, sb_t = 0;
  std::string sname;
  if (sinfo.operand_count == 1) {
    RX_USE(xenos_alu_operand_swizzle);
    float v[4];
    uint8_t t[4];
    bool abs;
    ReadAluSourceRegister(ctx, lane, op, 3, v, t, &abs, lane.trace ? &sname : nullptr);
    uint32_t swz = op.src_swizzle(3);
    bool neg = op.src_negate(3);
    uint32_t ca = ((swz >> 6) + 3) & 3;
    uint32_t bj = vcount == 3 ? 2 : 0;
    uint32_t cb = ((swz >> (2 * bj)) + bj) & 3;
    sa = ApplyMods(v[ca], abs, neg);
    sb = ApplyMods(v[cb], abs, neg);
    sa_t = t[ca];
    sb_t = sinfo.single_operand_is_two_component ? t[cb] : 0;
  } else if (sinfo.operand_count == 2) {
    RX_USE(xenos_alu_scalar_const_temp);
    uint32_t swz = op.src_swizzle(3);
    bool neg = op.src_negate(3);
    bool abs = op.abs_constants();
    int32_t cidx = int32_t(op.src_reg(3));
    uint8_t at = 0;
    if (op.src_const_is_addressed(3)) {
      if (op.is_const_address_register_relative()) {
        cidx += lane.a0;
        at = lane.a0_t;
      } else {
        cidx += lane.GetAL();
      }
    }
    float c[4];
    ReadConst(ctx, cidx, c);
    float tv[4];
    uint8_t tt[4];
    uint32_t treg = op.scalar_const_reg_op_src_temp_reg();
    ReadTemp(lane, int32_t(treg), tv, tt);
    uint32_t ca = ((swz >> 6) + 3) & 3;
    uint32_t cb = (swz + 0) & 3;
    sa = ApplyMods(c[ca], abs, neg);
    sb = ApplyMods(tv[cb], abs, neg);
    sa_t = at;
    sb_t = tt[cb];
    if (lane.trace) sname = "c" + std::to_string(op.src_reg(3)) + ", r" + std::to_string(treg);
  }

  VectorOpResult vr;
  ExecuteVectorOp(vop, vs[0], vs[1], vs[2], &vr);
  // Vector result taint.
  uint8_t vres_t[4];
  {
    bool componentwise = true;
    switch (vop) {
      case V::kDp4: case V::kDp3: case V::kDp2Add: case V::kCube: case V::kMax4:
      case V::kSetpEqPush: case V::kSetpNePush: case V::kSetpGtPush:
      case V::kSetpGePush: case V::kKillEq: case V::kKillGt: case V::kKillGe:
      case V::kKillNe: case V::kDst:
        componentwise = false;
        break;
      default:
        break;
    }
    uint8_t all = pred_t;
    for (uint32_t i = 0; i < vcount; ++i)
      for (int j = 0; j < 4; ++j) all = std::max(all, vt[i][j]);
    for (int j = 0; j < 4; ++j) {
      if (componentwise) {
        uint8_t m = pred_t;
        for (uint32_t i = 0; i < vcount; ++i) m = std::max(m, vt[i][j]);
        vres_t[j] = m;
      } else {
        vres_t[j] = all;
      }
    }
  }

  ScalarOpResult sr;
  ExecuteScalarOp(sop, sa, sb, lane.ps, &sr);
  uint8_t sres_t = std::max({sa_t, sb_t, pred_t});
  switch (sop) {
    case S::kAddsPrev: case S::kMulsPrev: case S::kMulsPrev2: case S::kSubsPrev:
    case S::kRetainPrev:
      sres_t = std::max(sres_t, lane.ps_t);
      break;
    default:
      break;
  }
  if (IsTranscendentalScalarOp(sop)) {
    RX_USE(xenos_transcendental_taint);
    sres_t = TaintAdd(sres_t, 1);
  }

  // Output flush and saturate.
  float vout[4];
  for (int j = 0; j < 4; ++j) {
    vout[j] = FlushDenormal(vr.v[j]);
    if (op.vector_clamp()) vout[j] = Saturate(vout[j]);
  }
  float sraw = FlushDenormal(sr.v);
  float sout = op.scalar_clamp() ? Saturate(sraw) : sraw;

  // State updates (vector first, then scalar).
  RX_USE(xenos_alu_vector_then_scalar);
  if (vr.sets_predicate) { lane.p = vr.predicate; lane.p_t = vres_t[0]; }
  if (vr.sets_a0) { lane.a0 = vr.a0; lane.a0_t = vres_t[3]; }
  if (vr.kill && ctx.is_ps) { RX_USE(xenos_kill_continues); lane.killed = true; }
  if (sop != S::kRetainPrev) {
    RX_USE(xenos_alu_ps_unsaturated);
    lane.ps = sraw;
    lane.ps_t = sres_t;
  }
  if (sr.sets_predicate) { lane.p = sr.predicate; lane.p_t = sres_t; }
  if (sr.sets_a0) { lane.a0 = sr.a0; lane.a0_t = sres_t; }
  if (sr.kill && ctx.is_ps) { RX_USE(xenos_kill_continues); lane.killed = true; }
  if ((vr.kill || sr.kill) && !ctx.is_ps) ctx.Note("kill instruction in a vertex shader ignored");

  // Stores.
  RX_USE(xenos_alu_masks);
  uint32_t vmask = op.GetVectorOpResultWriteMask();
  uint32_t smask = op.GetScalarOpResultWriteMask();
  uint32_t c0mask = op.GetConstant0WriteMask();
  uint32_t c1mask = op.GetConstant1WriteMask();
  float sv4[4] = {sout, sout, sout, sout};
  uint8_t st4[4] = {sres_t, sres_t, sres_t, sres_t};
  if (op.is_export()) {
    uint32_t reg = op.vector_dest();
    WriteExport(ctx, lane, reg, vmask, vout, vres_t);
    WriteExport(ctx, lane, reg, smask, sv4, st4);
    float zeros[4] = {0, 0, 0, 0}, ones[4] = {1, 1, 1, 1};
    uint8_t zt[4] = {pred_t, pred_t, pred_t, pred_t};
    WriteExport(ctx, lane, reg, c0mask, zeros, zt);
    WriteExport(ctx, lane, reg, c1mask, ones, zt);
  } else {
    int32_t vd = int32_t(op.vector_dest());
    if (op.is_vector_dest_relative()) vd += lane.GetAL();
    WriteTemp(lane, vd, vmask, vout, vres_t);
    int32_t sd = int32_t(op.scalar_dest());
    if (op.is_scalar_dest_relative()) sd += lane.GetAL();
    WriteTemp(lane, sd, smask, sv4, st4);
  }

  if (lane.trace) {
    std::string s = DisassembleAlu(dwords) + "  ;";
    for (uint32_t i = 0; i < vcount; ++i) s += " " + vnames[i] + "=" + Vec4Str(vs[i]);
    if (sinfo.operand_count) s += " | " + sname + " a=" + FormatFloat(sa) + " b=" + FormatFloat(sb);
    s += " -> v=" + Vec4Str(vout) + " s=" + FormatFloat(sout);
    if (vr.sets_predicate || sr.sets_predicate) s += std::string(" p0=") + (lane.p ? "1" : "0");
    if (vr.sets_a0 || sr.sets_a0) s += " a0=" + std::to_string(lane.a0);
    if (lane.killed) s += " killed";
    Trace(ctx, lane, s);
  }
}

// --- Fetch ---

void StoreFetchResult(Lane& lane, const ucode::FetchInstruction& f, const float v[4],
                      const uint8_t t[4]) {
  RX_USE(xenos_vfetch_dest_swizzle);
  int32_t d = int32_t(f.dest());
  if (f.is_dest_relative()) d += lane.GetAL();
  float out[4];
  uint8_t ot[4];
  uint32_t mask = 0;
  for (uint32_t c = 0; c < 4; ++c) {
    uint32_t s = ucode::GetFetchDestinationComponentSwizzle(f.dest_swizzle(), c);
    if (s == 7) continue;
    mask |= 1u << c;
    if (s < 4) { out[c] = v[s]; ot[c] = t[s]; }
    else if (s == 5) { out[c] = 1.0f; ot[c] = 0; }
    else { out[c] = 0.0f; ot[c] = 0; }
  }
  WriteTemp(lane, d, mask, out, ot);
}

// Reads 3 fetch source components of lane (fetch swizzle).
void ReadFetchSource(const Lane& lane, uint32_t src, bool rel, uint32_t swz, uint32_t n,
                     float out[3], uint8_t t[3]) {
  RX_USE(xenos_fetch_source_swizzle);
  int32_t idx = int32_t(src);
  if (rel) idx += lane.GetAL();
  float v[4];
  uint8_t vt[4];
  ReadTemp(lane, idx, v, vt);
  for (uint32_t j = 0; j < 3; ++j) {
    uint32_t c = j < n ? (swz >> (2 * j)) & 3 : 0;
    out[j] = j < n ? v[c] : 0.0f;
    t[j] = j < n ? vt[c] : 0;
  }
}

// IEEE half rounding toward zero (tfetch_gradient_fp16).
float TruncateToHalf(float f) {
  RX_USE(tfetch_gradient_fp16);
  if (std::isnan(f)) return f;
  uint32_t u = FloatBits(f);
  uint32_t sign = u & 0x80000000u;
  float a = std::fabs(f);
  if (a >= 65504.0f) return BitsFloat(sign | FloatBits(65504.0f));
  if (a < 0x1p-24f) return BitsFloat(sign);
  if (a < 0x1p-14f) {
    // Denormal half: multiples of 2^-24.
    float q = std::trunc(a * 0x1p24f) * 0x1p-24f;
    return BitsFloat(sign | FloatBits(q));
  }
  // Normal: keep 10 mantissa bits.
  return BitsFloat(u & ~0x1FFFu);
}

struct QuadSnapshot {
  float v[4][3];
  uint8_t t[4][3];
};

void ExecuteFetch(Ctx& ctx, Lane& lane, const uint32_t* dwords, const QuadSnapshot* quad) {
  const auto& f = *reinterpret_cast<const ucode::FetchInstruction*>(dwords);
  if (f.is_predicated() && f.predicate_condition() != lane.p) {
    RX_USE(tfetch_predicated_skip);
    if (lane.trace) Trace(ctx, lane, DisassembleFetch(dwords) + "  ; skipped (predicate)");
    return;
  }
  ucode::FetchOpcode opc = f.opcode();
  if (opc == ucode::FetchOpcode::kVertexFetch) {
    const auto& vf = f.vertex_fetch();
    if (!vf.is_mini_fetch()) {
      float s[3];
      uint8_t st[3];
      ReadFetchSource(lane, vf.src(), vf.is_src_relative(), vf.src_swizzle(), 1, s, st);
      lane.last_full = vf;
      lane.last_addr = ComputeVertexFetchAddress(vf, s[0], *ctx.regs);
    } else {
      RX_USE(xenos_vfetch_mini_reuse);
    }
    float out[4] = {0, 0, 0, 0};
    std::string note;
    if (!ctx.memory) {
      ctx.Note("vfetch without guest memory (pixel shader) returns 0");
    } else if (!lane.last_addr.valid) {
      ctx.Note("vfetch_mini without a preceding vfetch_full returns 0");
    } else if (!FetchVertex(vf, lane.last_addr, *ctx.memory, out, &note)) {
      ctx.Note(note);
      ctx.ok = false;
    }
    uint8_t zt[4] = {0, 0, 0, 0};
    StoreFetchResult(lane, f, out, zt);
    if (lane.trace) {
      Trace(ctx, lane, DisassembleFetch(dwords) + "  ; index=" + std::to_string(lane.last_addr.index) +
                           " addr=0x" + [&] { char b[16]; std::snprintf(b, 16, "%08X", lane.last_addr.element_base); return std::string(b); }() +
                           " -> " + Vec4Str(out));
    }
    return;
  }

  const auto& tf = f.texture_fetch();
  float src[3];
  uint8_t src_t[3];
  uint32_t dim = uint32_t(tf.dimension());
  uint32_t ncomp = dim == 0 ? 1 : dim == 1 ? 2 : 3;
  uint8_t zt4[4] = {0, 0, 0, 0};

  switch (opc) {
    case ucode::FetchOpcode::kSetTextureLod: {
      ReadFetchSource(lane, tf.src(), tf.is_src_relative(), tf.src_swizzle(), 1, src, src_t);
      lane.tex_lod = src[0];
      lane.tex_lod_t = src_t[0];
      if (lane.trace) Trace(ctx, lane, DisassembleFetch(dwords) + "  ; lod=" + FormatFloat(src[0]));
      return;
    }
    case ucode::FetchOpcode::kSetTextureGradientsHorz:
    case ucode::FetchOpcode::kSetTextureGradientsVert: {
      ReadFetchSource(lane, tf.src(), tf.is_src_relative(), tf.src_swizzle(), 3, src, src_t);
      float* g = opc == ucode::FetchOpcode::kSetTextureGradientsHorz ? lane.reg_grad_h : lane.reg_grad_v;
      for (int i = 0; i < 3; ++i) {
        g[i] = TruncateToHalf(src[i]);
        lane.reg_grad_t = std::max(lane.reg_grad_t, src_t[i]);
      }
      if (lane.trace) {
        float g4[4] = {g[0], g[1], g[2], 0};
        Trace(ctx, lane, DisassembleFetch(dwords) + "  ; " + Vec4Str(g4));
      }
      return;
    }
    case ucode::FetchOpcode::kGetTextureGradients: {
      RX_USE(tfetch_getgradients);
      float out[4] = {0, 0, 0, 0};
      uint8_t t = 0;
      if (ctx.is_ps && quad) {
        RX_USE(tfetch_gradients_coarse);
        out[0] = TruncateToHalf(quad->v[1][0] - quad->v[0][0]);
        out[1] = TruncateToHalf(quad->v[2][0] - quad->v[0][0]);
        out[2] = TruncateToHalf(quad->v[1][1] - quad->v[0][1]);
        out[3] = TruncateToHalf(quad->v[2][1] - quad->v[0][1]);
        for (int l = 0; l < 3; ++l) t = std::max({t, quad->t[l][0], quad->t[l][1]});
      } else {
        RX_USE(tfetch_vs_zero_gradients);
      }
      uint8_t t4[4] = {t, t, t, t};
      StoreFetchResult(lane, f, out, t4);
      if (lane.trace) Trace(ctx, lane, DisassembleFetch(dwords) + "  ; -> " + Vec4Str(out));
      return;
    }
    default:
      break;
  }

  // tfetch, getBCF, getCompTexLOD, getWeights.
  ReadFetchSource(lane, tf.src(), tf.is_src_relative(), tf.src_swizzle(), ncomp, src, src_t);
  TexFetchRequest req;
  req.op = uint32_t(opc);
  req.fetch_constant = tf.fetch_constant_index();
  req.dimension = dim;
  req.coord_count = ncomp;
  for (uint32_t i = 0; i < 3; ++i) req.coord[i] = src[i];
  req.coord[3] = 0.0f;
  if (dim == 3) RX_USE(tfetch_cube_coords);
  RX_USE(tfetch_lod_fields);
  req.use_register_lod = tf.use_register_lod();
  req.lod = tf.use_register_lod() ? lane.tex_lod : 0.0f;
  req.lod_bias_instr = tf.lod_bias();
  req.offset[0] = tf.offset_x();
  req.offset[1] = tf.offset_y();
  req.offset[2] = tf.offset_z();
  req.unnormalized_coordinates = tf.unnormalized_coordinates();
  req.fetch_valid_only = tf.fetch_valid_only();
  if (tf.has_mag_filter()) req.mag_filter = uint32_t(tf.mag_filter());
  if (tf.has_min_filter()) req.min_filter = uint32_t(tf.min_filter());
  if (tf.has_mip_filter()) req.mip_filter = uint32_t(tf.mip_filter());
  if (tf.has_aniso_filter()) req.aniso_filter = uint32_t(tf.aniso_filter());
  if (tf.has_vol_mag_filter()) req.vol_mag_filter = uint32_t(tf.vol_mag_filter());
  if (tf.has_vol_min_filter()) req.vol_min_filter = uint32_t(tf.vol_min_filter());
  req.arbitrary_filter_raw = (dwords[1] >> 21) & 7;
  req.sample_location = uint32_t(tf.sample_location());
  req.is_pixel_shader = ctx.is_ps;
  req.helper_lane = lane.helper;
  req.lane = uint32_t(&lane - ctx.lanes);
  req.x = uint32_t(lane.x);
  req.y = uint32_t(lane.y);
  for (int i = 0; i < 3; ++i) req.instr_dwords[i] = dwords[i];

  uint8_t t = std::max({src_t[0], src_t[1], src_t[2]});
  if (req.use_register_lod) t = std::max(t, lane.tex_lod_t);
  bool wants_computed = tf.use_computed_lod() || opc == ucode::FetchOpcode::kGetTextureComputedLod;
  if (tf.use_register_gradients()) {
    RX_USE(tfetch_register_gradients_shared);
    const Lane* src_lane = &lane;
    if (ctx.is_ps) {
      for (int l = 0; l < ctx.lane_count; ++l) {
        if (!ctx.lanes[l].helper) { src_lane = &ctx.lanes[l]; break; }
      }
    }
    for (int i = 0; i < 3; ++i) {
      req.grad_h[i] = src_lane->reg_grad_h[i];
      req.grad_v[i] = src_lane->reg_grad_v[i];
    }
    req.has_grads = true;
    req.grads_from_registers = true;
    req.use_computed_lod = wants_computed;
    t = std::max(t, src_lane->reg_grad_t);
  } else if (ctx.is_ps && quad) {
    RX_USE(tfetch_gradients_coarse);
    for (uint32_t i = 0; i < 3; ++i) {
      req.grad_h[i] = i < ncomp ? quad->v[1][i] - quad->v[0][i] : 0.0f;
      req.grad_v[i] = i < ncomp ? quad->v[2][i] - quad->v[0][i] : 0.0f;
    }
    req.has_grads = true;
    req.use_computed_lod = wants_computed;
    if (wants_computed) {
      for (int l = 0; l < 3; ++l)
        for (uint32_t i = 0; i < ncomp; ++i) t = std::max(t, quad->t[l][i]);
    }
  } else {
    RX_USE(tfetch_vs_zero_gradients);
    req.has_grads = false;
    req.use_computed_lod = false;
  }
  req.lod_is_absolute = !req.use_computed_lod;

  float out[4] = {0, 0, 0, 0};
  if (ctx.sampler) {
    std::vector<std::string> used;
    ctx.sampler->Fetch(req, out, &used);
  } else {
    ctx.Note("texture fetch without a sampler returns 0");
  }
  uint8_t t4[4] = {t, t, t, t};
  StoreFetchResult(lane, f, out, t4);
  if (lane.trace) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  ; coord=(%s, %s, %s) gh=(%g, %g, %g) gv=(%g, %g, %g) lod=%g -> ",
                  FormatFloat(req.coord[0]).c_str(), FormatFloat(req.coord[1]).c_str(),
                  FormatFloat(req.coord[2]).c_str(), double(req.grad_h[0]), double(req.grad_h[1]),
                  double(req.grad_h[2]), double(req.grad_v[0]), double(req.grad_v[1]),
                  double(req.grad_v[2]), double(req.lod));
    Trace(ctx, lane, DisassembleFetch(dwords) + buf + Vec4Str(out));
  }
  (void)zt4;
}

bool FetchNeedsQuad(const uint32_t* dwords) {
  const auto& f = *reinterpret_cast<const ucode::FetchInstruction*>(dwords);
  switch (f.opcode()) {
    case ucode::FetchOpcode::kTextureFetch:
    case ucode::FetchOpcode::kGetTextureBorderColorFrac:
    case ucode::FetchOpcode::kGetTextureComputedLod:
    case ucode::FetchOpcode::kGetTextureWeights:
    case ucode::FetchOpcode::kGetTextureGradients:
      return true;
    default:
      return false;
  }
}

void SnapshotQuad(Ctx& ctx, const uint32_t* dwords, QuadSnapshot* snap) {
  const auto& tf = reinterpret_cast<const ucode::FetchInstruction*>(dwords)->texture_fetch();
  uint32_t dim = uint32_t(tf.dimension());
  uint32_t n = tf.opcode() == ucode::FetchOpcode::kGetTextureGradients ? 2
               : dim == 0 ? 1 : dim == 1 ? 2 : 3;
  for (int l = 0; l < 4; ++l) {
    if (l < ctx.lane_count) {
      ReadFetchSource(ctx.lanes[l], tf.src(), tf.is_src_relative(), tf.src_swizzle(), n,
                      snap->v[l], snap->t[l]);
    } else {
      for (int i = 0; i < 3; ++i) { snap->v[l][i] = 0; snap->t[l][i] = 0; }
    }
  }
}

// --- Control flow ---

void ExecuteExec(Ctx& ctx, const std::vector<Lane*>& lanes, const ucode::ControlFlowInstruction& cf) {
  const auto& ex = cf.exec;
  uint32_t address = ex.address();
  uint32_t count = ex.count();
  uint32_t seq = ex.sequence();
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t* instr = GetInstr(*ctx.prog, address + i);
    if (!instr) {
      ctx.Note("exec past the end of the program");
      ctx.ok = false;
      return;
    }
    bool is_fetch = (seq >> (2 * i)) & 1;
    if (is_fetch) {
      QuadSnapshot snap;
      const QuadSnapshot* sp = nullptr;
      if (ctx.is_ps && FetchNeedsQuad(instr)) {
        SnapshotQuad(ctx, instr, &snap);
        sp = &snap;
      }
      for (Lane* l : lanes) ExecuteFetch(ctx, *l, instr, sp);
    } else {
      for (Lane* l : lanes) ExecuteAlu(ctx, *l, instr);
    }
  }
}

std::string CfName(ucode::ControlFlowOpcode op) {
  static const char* kNames[] = {"cnop", "exec", "exece", "cexec", "cexece",
                                 "cexec_pred", "cexece_pred", "loop", "endloop",
                                 "ccall", "ret", "cjmp", "alloc",
                                 "cexec_pred_clean", "cexece_pred_clean",
                                 "vfetch_done"};
  return kNames[uint32_t(op) & 15];
}

// Runs one CF instruction at pc for the given lanes (all at the same pc).
void StepCf(Ctx& ctx, std::vector<Lane*>& lanes, uint32_t pc) {
  ucode::ControlFlowInstruction cf;
  if (!GetCf(*ctx.prog, pc, &cf)) {
    for (Lane* l : lanes) l->ended = true;
    ctx.Note("control flow ran past the end of the program");
    ctx.ok = false;
    return;
  }
  using C = ucode::ControlFlowOpcode;
  C op = cf.opcode();
  const RegisterFile& regs = *ctx.regs;
  for (Lane* l : lanes) {
    l->cf = pc + 1;
    if (++l->steps > GetInterpOptions().max_cf_steps) {
      l->ended = true;
      ctx.Note("control flow step limit reached");
      ctx.ok = false;
    }
  }
  switch (op) {
    case C::kExec:
    case C::kExecEnd:
    case C::kCondExec:
    case C::kCondExecEnd:
    case C::kCondExecPred:
    case C::kCondExecPredEnd:
    case C::kCondExecPredClean:
    case C::kCondExecPredCleanEnd: {
      RX_USE(xenos_cf_exec);
      std::vector<Lane*> taken;
      for (Lane* l : lanes) {
        if (l->ended) continue;
        bool take = true;
        if (op == C::kCondExec || op == C::kCondExecEnd || op == C::kCondExecPredClean ||
            op == C::kCondExecPredCleanEnd) {
          if (op == C::kCondExecPredClean || op == C::kCondExecPredCleanEnd) RX_USE(xenos_cf_pred_clean);
          take = BoolConstant(regs, cf.cond_exec.bool_address()) == cf.cond_exec.condition();
        } else if (op == C::kCondExecPred || op == C::kCondExecPredEnd) {
          take = l->p == cf.cond_exec_pred.condition();
        }
        if (l->trace) {
          Trace(ctx, *l, "cf " + std::to_string(pc) + ": " + CfName(op) + " addr " +
                             std::to_string(cf.exec.address()) + " count " +
                             std::to_string(cf.exec.count()) + (take ? "" : "  ; not taken"));
        }
        if (take) taken.push_back(l);
      }
      if (!taken.empty()) ExecuteExec(ctx, taken, cf);
      if (ucode::DoesControlFlowOpcodeEndShader(op)) {
        for (Lane* l : taken) l->ended = true;
      }
    } break;
    case C::kLoopStart: {
      RX_USE(xenos_cf_loop);
      const auto& ls = cf.loop_start;
      xenos::LoopConstant lc;
      lc.value = regs.values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00 + ls.loop_id()];
      for (Lane* l : lanes) {
        if (l->trace) Trace(*&ctx, *l, "cf " + std::to_string(pc) + ": loop i" + std::to_string(ls.loop_id()) +
                                           " count " + std::to_string(lc.count) + " start " +
                                           std::to_string(lc.start) + " step " + std::to_string(lc.step));
        if (lc.count == 0) { l->cf = ls.address(); continue; }
        if (l->loop_depth >= 4) {
          RX_USE(xenos_cf_loop_depth);
          ctx.Note("loop nesting deeper than 4 skipped");
          l->cf = ls.address();
          continue;
        }
        int32_t aL = ls.is_repeat() && l->loop_depth ? l->aL[l->loop_depth - 1] : int32_t(lc.start);
        if (ls.is_repeat() && !l->loop_depth) aL = 0;
        l->loop_count[l->loop_depth] = lc.count;
        l->loop_step[l->loop_depth] = lc.step;
        l->aL[l->loop_depth] = aL;
        ++l->loop_depth;
      }
    } break;
    case C::kLoopEnd: {
      RX_USE(xenos_cf_loop);
      const auto& le = cf.loop_end;
      for (Lane* l : lanes) {
        if (!l->loop_depth) { ctx.Note("loop_end without loop_start"); continue; }
        int d = l->loop_depth - 1;
        --l->loop_count[d];
        bool brk = l->loop_count[d] == 0 ||
                   (le.is_predicated_break() && l->p == le.condition());
        if (l->trace) Trace(ctx, *l, "cf " + std::to_string(pc) + ": endloop" + (brk ? "  ; exit" : "  ; repeat"));
        if (brk) {
          --l->loop_depth;
        } else {
          l->aL[d] += l->loop_step[d];
          l->cf = le.address();
        }
      }
    } break;
    case C::kCondCall: {
      RX_USE(xenos_cf_call_jmp);
      const auto& cc = cf.cond_call;
      for (Lane* l : lanes) {
        bool take = cc.is_unconditional() ||
                    (cc.is_predicated() ? l->p == cc.condition()
                                        : BoolConstant(regs, cc.bool_address()) == cc.condition());
        if (l->trace) Trace(ctx, *l, "cf " + std::to_string(pc) + ": call " + std::to_string(cc.address()) + (take ? "" : "  ; not taken"));
        if (!take) continue;
        if (l->call_depth >= 4) {
          RX_USE(xenos_cf_loop_depth);
          ctx.Note("call nesting deeper than 4 skipped");
          continue;
        }
        l->call_ret[l->call_depth++] = pc + 1;
        l->cf = cc.address();
      }
    } break;
    case C::kReturn:
      for (Lane* l : lanes) {
        if (l->trace) Trace(ctx, *l, "cf " + std::to_string(pc) + ": ret");
        if (!l->call_depth) { RX_USE(xenos_cf_loop_depth); continue; }
        l->cf = l->call_ret[--l->call_depth];
      }
      break;
    case C::kCondJmp: {
      RX_USE(xenos_cf_call_jmp);
      const auto& cj = cf.cond_jmp;
      for (Lane* l : lanes) {
        bool take = cj.is_unconditional() ||
                    (cj.is_predicated() ? l->p == cj.condition()
                                        : BoolConstant(regs, cj.bool_address()) == cj.condition());
        if (l->trace) Trace(ctx, *l, "cf " + std::to_string(pc) + ": jmp " + std::to_string(cj.address()) + (take ? "" : "  ; not taken"));
        if (take) l->cf = cj.address();
      }
    } break;
    case C::kAlloc:
      for (Lane* l : lanes) {
        if (l->trace) Trace(ctx, *l, "cf " + std::to_string(pc) + ": alloc type " +
                                       std::to_string(uint32_t(cf.alloc.alloc_type())) + " size " +
                                       std::to_string(cf.alloc.size()));
      }
      break;
    case C::kNop:
    case C::kMarkVsFetchDone:
    default:
      break;
  }
}

// SIMT scheduler: repeatedly run the lanes at the minimum pc.
void RunLanes(Ctx& ctx) {
  RX_USE(xenos_cf_lanes_independent);
  for (;;) {
    uint32_t min_pc = UINT32_MAX;
    for (int i = 0; i < ctx.lane_count; ++i) {
      if (!ctx.lanes[i].ended) min_pc = std::min(min_pc, ctx.lanes[i].cf);
    }
    if (min_pc == UINT32_MAX) break;
    std::vector<Lane*> group;
    for (int i = 0; i < ctx.lane_count; ++i) {
      if (!ctx.lanes[i].ended && ctx.lanes[i].cf == min_pc) group.push_back(&ctx.lanes[i]);
    }
    StepCf(ctx, group, min_pc);
  }
}

void InitLane(Lane& l) {
  std::memset(l.r, 0, sizeof(l.r));
  std::memset(l.rt, 0, sizeof(l.rt));
}

void InitExports(RawExports& e) {
  std::memset(&e, 0, sizeof(e));
  e.ok = true;
}

void SetupConstants(Ctx& ctx, bool ps) {
  uint32_t v = ctx.regs->values[ps ? XE_GPU_REG_SQ_PS_CONST : XE_GPU_REG_SQ_VS_CONST];
  ctx.const_base = v & 0x1FF;
  ctx.const_size_minus_1 = (v >> 12) & 0x1FF;
}

void FinishNotes(Ctx& ctx, DrawStats& stats) {
  if (ctx.memexport_writes) {
    ctx.Note("memexport: " + std::to_string(ctx.memexport_writes) +
             " eA/eM export writes recorded, guest memory not written");
  }
  for (auto& n : ctx.notes) {
    if (std::find(stats.notes.begin(), stats.notes.end(), n) == stats.notes.end()) {
      stats.notes.push_back(n);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry points

bool RunVertexShaderRaw(const ShaderProgram& prog, const RegisterFile& regs,
                        const Memory& memory, uint32_t vertex_index,
                        TextureSampler* sampler, RawExports* out, DrawStats& stats,
                        Probe* probe) {
  auto lane = std::make_unique<Lane>();
  InitLane(*lane);
  InitExports(*out);
  Ctx ctx;
  ctx.prog = &prog;
  ctx.regs = &regs;
  ctx.memory = &memory;
  ctx.sampler = sampler;
  ctx.stats = &stats;
  ctx.probe = probe;
  ctx.is_ps = false;
  ctx.lanes = lane.get();
  ctx.lane_count = 1;
  ctx.vertex_index = vertex_index;
  SetupConstants(ctx, false);
  RX_USE(xenos_vs_r0_index);
  lane->r[0][0] = float(vertex_index);
  lane->exports = out;
  lane->trace = probe && probe->log_vertices;
  RunLanes(ctx);
  out->ok = ctx.ok;
  FinishNotes(ctx, stats);
  ++stats.vertices_shaded;
  return ctx.ok;
}

bool RunVertexShader(const ShaderProgram& prog, const RegisterFile& regs,
                     const Memory& memory, uint32_t vertex_index,
                     TextureSampler* sampler, VertexOut* out, DrawStats& stats,
                     Probe* probe) {
  auto e = std::make_unique<RawExports>();
  bool ok = RunVertexShaderRaw(prog, regs, memory, vertex_index, sampler, e.get(), stats, probe);
  *out = VertexOut();
  out->vertex_index = vertex_index;
  if (e->written & (uint64_t(1) << 62)) {
    out->position_written = true;
    for (int i = 0; i < 4; ++i) {
      out->position[i] = e->reg[62][i];
      out->position_taint[i] = e->taint[62][i];
    }
  }
  if (e->written & (uint64_t(1) << 63)) {
    out->point_written = true;
    for (int i = 0; i < 4; ++i) {
      out->point_size_edge_kill[i] = e->reg[63][i];
      out->point_size_edge_kill_taint[i] = e->taint[63][i];
    }
  }
  for (uint32_t r = 0; r < kMaxInterpolators; ++r) {
    if (e->written & (uint64_t(1) << r)) {
      out->interp_written_mask |= 1u << r;
      for (int i = 0; i < 4; ++i) {
        out->interp[r][i] = e->reg[r][i];
        out->interp_taint[r][i] = e->taint[r][i];
      }
    }
  }
  return ok;
}

bool RunPixelQuadRaw(const ShaderProgram& prog, const RegisterFile& regs,
                     const PixelQuad& quad, const float (*inputs)[16][4],
                     TextureSampler* sampler, RawExports out[4], DrawStats& stats,
                     Probe* probe) {
  auto lanes = std::make_unique<Lane[]>(4);
  Ctx ctx;
  ctx.prog = &prog;
  ctx.regs = &regs;
  ctx.memory = nullptr;
  ctx.sampler = sampler;
  ctx.stats = &stats;
  ctx.probe = probe;
  ctx.is_ps = true;
  ctx.lanes = lanes.get();
  ctx.lane_count = 4;
  SetupConstants(ctx, true);
  auto cntl = regs.Get<reg::SQ_PROGRAM_CNTL>();
  auto misc = regs.Get<reg::SQ_CONTEXT_MISC>();
  RX_USE(xenos_ps_inputs);
  RX_USE(xenos_ps_helper_lanes);
  for (int l = 0; l < 4; ++l) {
    Lane& lane = lanes[l];
    InitLane(lane);
    InitExports(out[l]);
    lane.exports = &out[l];
    lane.x = quad.x + (l & 1);
    lane.y = quad.y + (l >> 1);
    lane.helper = quad.coverage[l] == 0;
    lane.trace = probe && probe->Wants(lane.x, lane.y);
    for (int r = 0; r < 16; ++r) {
      for (int c = 0; c < 4; ++c) {
        lane.r[r][c] = inputs ? inputs[l][r][c] : quad.interp[l][r][c];
      }
    }
    if (!inputs && cntl.param_gen && misc.param_gen_pos < 16) {
      for (int c = 0; c < 4; ++c) lane.r[misc.param_gen_pos][c] = quad.param_gen[l][c];
    }
  }
  RunLanes(ctx);
  for (int l = 0; l < 4; ++l) {
    out[l].killed = lanes[l].killed;
    out[l].ok = ctx.ok;
  }
  FinishNotes(ctx, stats);
  ++stats.quads;
  return ctx.ok;
}

bool RunPixelQuad(const ShaderProgram& prog, const RegisterFile& regs,
                  const PixelQuad& quad, TextureSampler* sampler, PixelOut out[4],
                  DrawStats& stats, Probe* probe) {
  auto e = std::make_unique<RawExports[]>(4);
  bool ok = RunPixelQuadRaw(prog, regs, quad, nullptr, sampler, e.get(), stats, probe);
  for (int l = 0; l < 4; ++l) {
    out[l] = PixelOut();
    const RawExports& x = e[l];
    for (int c = 0; c < 4; ++c) {
      if (x.written & (uint64_t(1) << c)) {
        out[l].color_written_mask |= 1u << c;
        for (int i = 0; i < 4; ++i) {
          out[l].color[c][i] = x.reg[c][i];
          out[l].color_taint[c][i] = x.taint[c][i];
        }
      }
    }
    if (x.written & (uint64_t(1) << 61)) {
      out[l].depth_written = true;
      out[l].depth = x.reg[61][0];
      out[l].depth_taint = x.taint[61][0];
    }
    out[l].killed = x.killed;
    if (quad.coverage[l]) {
      ++stats.pixels_shaded;
      if (x.killed) ++stats.pixels_killed;
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Disassembly

std::string DisassembleAlu(const uint32_t* dwords) {
  const auto& op = *reinterpret_cast<const ucode::AluInstruction*>(dwords);
  const auto& vi = ucode::GetAluVectorOpcodeInfo(op.vector_opcode());
  const auto& si = ucode::GetAluScalarOpcodeInfo(op.scalar_opcode());
  auto src = [&](uint32_t i) {
    std::string s = op.src_negate(i) ? "-" : "";
    uint32_t reg = op.src_reg(i);
    if (op.src_is_temp(i)) {
      s += (reg & 0x40) ? "r[aL+" + std::to_string(reg & 0x3F) + "]" : "r" + std::to_string(reg & 0x3F);
      if (reg & 0x80) s += "_abs";
    } else {
      if (op.src_const_is_addressed(i)) {
        s += std::string("c[") + (op.is_const_address_register_relative() ? "a0" : "aL") + "+" + std::to_string(reg) + "]";
      } else {
        s += "c" + std::to_string(reg);
      }
      if (op.abs_constants()) s += "_abs";
    }
    uint32_t swz = op.src_swizzle(i);
    s += ".";
    for (uint32_t j = 0; j < 4; ++j) s += "xyzw"[((swz >> (2 * j)) + j) & 3];
    return s;
  };
  auto mask = [](uint32_t m) {
    std::string s;
    for (int i = 0; i < 4; ++i) s += (m & (1u << i)) ? "xyzw"[i] : '_';
    return s;
  };
  std::string s;
  if (op.is_predicated()) s += op.predicate_condition() ? "(p0) " : "(!p0) ";
  std::string vdest = op.is_export() ? "export" + std::to_string(op.vector_dest())
                                     : (op.is_vector_dest_relative() ? "r[aL+" + std::to_string(op.vector_dest()) + "]"
                                                                     : "r" + std::to_string(op.vector_dest()));
  s += std::string(vi.name) + (op.vector_clamp() ? "_sat " : " ") + vdest + "." +
       mask(op.GetVectorOpResultWriteMask());
  for (uint32_t i = 0; i < vi.GetOperandCount(); ++i) s += ", " + src(i + 1);
  std::string sdest = op.is_export() ? "export" + std::to_string(op.vector_dest())
                                     : (op.is_scalar_dest_relative() ? "r[aL+" + std::to_string(op.scalar_dest()) + "]"
                                                                     : "r" + std::to_string(op.scalar_dest()));
  s += " + " + std::string(si.name) + (op.scalar_clamp() ? "_sat " : " ") + sdest + "." +
       mask(op.GetScalarOpResultWriteMask());
  if (si.operand_count == 1) s += ", " + src(3);
  if (si.operand_count == 2) {
    s += ", c" + std::to_string(op.src_reg(3)) + ", r" + std::to_string(op.scalar_const_reg_op_src_temp_reg());
  }
  return s;
}

std::string DisassembleFetch(const uint32_t* dwords) {
  const auto& f = *reinterpret_cast<const ucode::FetchInstruction*>(dwords);
  std::string s;
  if (f.is_predicated()) s += f.predicate_condition() ? "(p0) " : "(!p0) ";
  static const char* kDims[] = {"1D", "2D", "3D", "Cube"};
  char buf[160];
  auto dsw = [&] {
    std::string d;
    for (uint32_t c = 0; c < 4; ++c) d += "xyzw01?_"[ucode::GetFetchDestinationComponentSwizzle(f.dest_swizzle(), c)];
    return d;
  };
  switch (f.opcode()) {
    case ucode::FetchOpcode::kVertexFetch: {
      const auto& v = f.vertex_fetch();
      std::snprintf(buf, sizeof(buf), "%s r%u.%s, r%u.%c, vf%u format=%u stride=%u offset=%d%s%s exp=%d",
                    v.is_mini_fetch() ? "vfetch_mini" : "vfetch_full", f.dest(), dsw().c_str(), v.src(),
                    "xyzw"[v.src_swizzle() & 3], v.fetch_constant_index(), uint32_t(v.data_format()),
                    v.stride(), v.offset(), v.is_signed() ? " signed" : "", v.is_normalized() ? " norm" : "",
                    v.exp_adjust());
      return s + buf;
    }
    default: {
      const auto& t = f.texture_fetch();
      const char* name = "tfetch";
      switch (f.opcode()) {
        case ucode::FetchOpcode::kGetTextureBorderColorFrac: name = "getBCF"; break;
        case ucode::FetchOpcode::kGetTextureComputedLod: name = "getCompTexLOD"; break;
        case ucode::FetchOpcode::kGetTextureGradients: name = "getGradients"; break;
        case ucode::FetchOpcode::kGetTextureWeights: name = "getWeights"; break;
        case ucode::FetchOpcode::kSetTextureLod: name = "setTexLOD"; break;
        case ucode::FetchOpcode::kSetTextureGradientsHorz: name = "setGradientH"; break;
        case ucode::FetchOpcode::kSetTextureGradientsVert: name = "setGradientV"; break;
        default: break;
      }
      uint32_t sw = t.src_swizzle();
      std::snprintf(buf, sizeof(buf), "%s%s r%u.%s, r%u.%c%c%c, tf%u%s%s%s%s bias=%g off=(%g,%g,%g)",
                    name, kDims[uint32_t(t.dimension())], f.dest(), dsw().c_str(), t.src(),
                    "xyzw"[sw & 3], "xyzw"[(sw >> 2) & 3], "xyzw"[(sw >> 4) & 3],
                    t.fetch_constant_index(), t.use_computed_lod() ? " complod" : "",
                    t.use_register_lod() ? " reglod" : "", t.use_register_gradients() ? " reggrad" : "",
                    t.unnormalized_coordinates() ? " unnorm" : "", double(t.lod_bias()),
                    double(t.offset_x()), double(t.offset_y()), double(t.offset_z()));
      return s + buf;
    }
  }
}

std::string DisassembleCf(const ShaderProgram& prog, uint32_t cf_index) {
  ucode::ControlFlowInstruction cf;
  if (!GetCf(prog, cf_index, &cf)) return "<out of range>";
  return CfName(cf.opcode());
}

bool ShaderUsesTextureFetchResults(const ShaderProgram& prog) {
  // Walk the CF region: it ends at the lowest exec address.
  uint32_t min_addr = uint32_t(prog.ucode.size() / 3);
  for (uint32_t cf = 0; (cf >> 1) < min_addr; ++cf) {
    ucode::ControlFlowInstruction c;
    if (!GetCf(prog, cf, &c)) break;
    if (!ucode::IsControlFlowOpcodeExec(c.opcode())) continue;
    uint32_t a = c.exec.address();
    min_addr = std::min(min_addr, a);
    for (uint32_t i = 0; i < c.exec.count(); ++i) {
      if (!((c.exec.sequence() >> (2 * i)) & 1)) continue;
      const uint32_t* in = GetInstr(prog, a + i);
      if (!in) continue;
      const auto& f = *reinterpret_cast<const ucode::FetchInstruction*>(in);
      switch (f.opcode()) {
        case ucode::FetchOpcode::kTextureFetch:
        case ucode::FetchOpcode::kGetTextureBorderColorFrac:
        case ucode::FetchOpcode::kGetTextureComputedLod:
        case ucode::FetchOpcode::kGetTextureGradients:
        case ucode::FetchOpcode::kGetTextureWeights:
          if (f.dest_swizzle() != 0xFFF) return true;
          break;
        default:
          break;
      }
    }
  }
  return false;
}

}  // namespace refxenos
