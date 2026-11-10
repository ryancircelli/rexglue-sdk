// refxenos shader execution: vertex fetch, vertex and pixel shader
// interpreters and the texture fetch instruction front end.
//
// Every non-trivial behaviour is a Rule (RX_DEFINE_RULE in the .cpp files of
// this directory). Pixel shaders run a 2x2 quad as four lanes with fully
// independent state; control flow is scheduled SIMT style (the active lanes at
// the minimum control flow pc execute the next control flow instruction
// together), so every lane computes exactly what it would compute alone, while
// gradient instructions can read the other lanes' current registers.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/common.h"

namespace refxenos {

struct ShaderProgram {
  xenos::ShaderType type;
  std::vector<uint32_t> ucode;  // host-endian dwords
};

// Operation the sampler is asked to perform. Values are ucode::FetchOpcode.
enum class TexFetchOp : uint32_t {
  kTextureFetch = 1,
  kGetBorderColorFrac = 16,  // getBCF
  kGetComputedLod = 17,      // getCompTexLOD
  kGetWeights = 19,          // getWeights
};

struct TexFetchRequest {
  uint32_t op = 1;              // TexFetchOp
  uint32_t fetch_constant = 0;  // tf# index 0..31
  uint32_t dimension = 0;       // xenos::FetchOpDimension of the instruction
  // Coordinates exactly as the instruction supplies them (source swizzle
  // applied, nothing else). 1D: [0]; 2D: [0..1]; 3D/stacked: [0..2];
  // cube: [0]=SC, [1]=TC (expected in [1,2) after the cube ALU sequence),
  // [2]=face index from the cube ALU. [3] is always 0.
  float coord[4] = {0, 0, 0, 0};
  uint32_t coord_count = 0;
  // Raw coordinate differences (not scaled by texture size), coarse:
  // grad_h = lane TR - lane TL, grad_v = lane BL - lane TL for every lane of
  // the quad (rule tfetch_gradients_coarse), or the register gradients set by
  // setGradientH/V when use_register_gradients. For cube maps these are in
  // the SC/TC/face space the instruction supplied (register gradients are in
  // cube space per ucode comments). Zero when has_grads is false.
  float grad_h[3] = {0, 0, 0};
  float grad_v[3] = {0, 0, 0};
  bool has_grads = false;
  bool grads_from_registers = false;
  // Effective UseComputedLOD (instruction bit, forced off in vertex shaders
  // without register gradients: rule tfetch_vs_computed_lod).
  bool use_computed_lod = false;
  // UseRegisterLOD: lod holds the setTexLOD value. XDK: total LOD is additive
  // (computed if use_computed_lod) + (register if use_register_lod) + bias
  // (instruction) + bias (fetch constant).
  bool use_register_lod = false;
  float lod = 0.0f;
  // True when the LOD is not computed from gradients (lod is then the base:
  // register LOD or 0).
  bool lod_is_absolute = true;
  float lod_bias_instr = 0.0f;  // instruction LODBias, already * 1/16
  float offset[3] = {0, 0, 0};  // instruction OffsetX/Y/Z, already * 0.5
  bool unnormalized_coordinates = false;
  bool fetch_valid_only = false;
  // Instruction overrides; nullopt = "use fetch constant" (keep).
  std::optional<uint32_t> mag_filter;      // xenos::TextureFilter
  std::optional<uint32_t> min_filter;      // xenos::TextureFilter
  std::optional<uint32_t> mip_filter;      // xenos::TextureFilter
  std::optional<uint32_t> aniso_filter;    // xenos::AnisoFilter
  std::optional<uint32_t> vol_mag_filter;  // xenos::TextureFilter
  std::optional<uint32_t> vol_min_filter;  // xenos::TextureFilter
  uint32_t arbitrary_filter_raw = 7;       // xenos::ArbitraryFilter, 7 = keep
  uint32_t sample_location = 1;            // xenos::SampleLocation
  // Shader context.
  bool is_pixel_shader = false;
  bool helper_lane = false;   // pixel lane with coverage 0
  uint32_t lane = 0;          // 0..3 quad order, 0 for vertex shaders
  uint32_t x = 0, y = 0;      // pixel position (pixel shaders)
  uint32_t instr_dwords[3] = {0, 0, 0};  // the raw fetch instruction
};

class TextureSampler {
 public:
  virtual ~TextureSampler() = default;
  virtual void Fetch(const TexFetchRequest&, float out[4],
                     std::vector<std::string>* rules_used) = 0;
};

// Test sampler: returns coord[0..3] (for getCompTexLOD/getWeights/getBCF it
// still returns the coordinates).
class CoordTextureSampler : public TextureSampler {
 public:
  void Fetch(const TexFetchRequest& r, float out[4],
             std::vector<std::string>*) override {
    for (int i = 0; i < 4; ++i) out[i] = r.coord[i];
    last = r;
    ++count;
  }
  TexFetchRequest last;
  uint32_t count = 0;
};

class RecordingTextureSampler : public CoordTextureSampler {
 public:
  void Fetch(const TexFetchRequest& r, float out[4],
             std::vector<std::string>* u) override {
    CoordTextureSampler::Fetch(r, out, u);
    all.push_back(r);
  }
  std::vector<TexFetchRequest> all;
};

struct InterpOptions {
  // Mantissa bits kept for exp/log/logc/rcp*/rsq*/sqrt results (rule
  // xenos_alu_transcendental_rounding). 23 = correctly rounded f32 (default,
  // per the refxenos lead); 21 = xenia-canary 3a44f20c7 behaviour.
  uint32_t transcendental_mantissa_bits = 23;
  // Safety limit on executed control flow steps per lane.
  uint64_t max_cf_steps = 1u << 24;
};
void SetInterpOptions(const InterpOptions&);
const InterpOptions& GetInterpOptions();

// Raw export state of one invocation (for tools/tests). reg[r] is export
// register r (ucode::ExportRegister & 63); written bit r set when any
// component of it was written. Components not written stay 0.
struct RawExports {
  float reg[64][4];
  uint8_t taint[64][4];  // transcendental taint per component
  uint64_t written = 0;
  uint8_t comp_written[64];
  bool killed = false;
  bool ok = true;
};

bool RunVertexShader(const ShaderProgram&, const RegisterFile&, const Memory&,
                     uint32_t vertex_index, TextureSampler*, VertexOut* out,
                     DrawStats&, Probe*);
bool RunPixelQuad(const ShaderProgram&, const RegisterFile&,
                  const PixelQuad& quad, TextureSampler*, PixelOut out[4],
                  DrawStats&, Probe*);

// Lower level entry points used by RunVertexShader/RunPixelQuad and the
// shadertool-compatible runner. For pixel quads, `inputs` (if not null)
// replaces the interpolant/param_gen loading: inputs[lane][r] -> r#.
bool RunVertexShaderRaw(const ShaderProgram&, const RegisterFile&,
                        const Memory&, uint32_t vertex_index, TextureSampler*,
                        RawExports* out, DrawStats&, Probe*);
bool RunPixelQuadRaw(const ShaderProgram&, const RegisterFile&,
                     const PixelQuad& quad, const float (*inputs)[16][4],
                     TextureSampler*, RawExports out[4], DrawStats&, Probe*);

// Disassembly of one ALU (3 dwords) or fetch instruction, and CF instruction.
std::string DisassembleAlu(const uint32_t* dwords);
std::string DisassembleFetch(const uint32_t* dwords);
std::string DisassembleCf(const ShaderProgram&, uint32_t cf_index);

// Static analysis helpers.
bool ShaderUsesTextureFetchResults(const ShaderProgram&);

}  // namespace refxenos
