// Internal helpers shared by the refxenos shader interpreter files.
#pragma once

#include <cstdint>
#include <string>

#include <rex/graphics/format/ucode.h>

#include "shader/interp.h"

namespace refxenos {
// Defined in vfetch.cpp, applied in interp.cpp (extern so the const has
// external linkage).
RX_DECLARE_RULE(xenos_vfetch_mini_reuse);
}  // namespace refxenos

namespace refxenos::shader_internal {

namespace ucode = rex::graphics::ucode;

uint32_t FloatBits(float f);
float BitsFloat(uint32_t u);

// Denormal flush (sign preserving), applied to ALU inputs and outputs.
float FlushDenormal(float f);
// [0, 1] clamp of the ALU result (NaN -> 0).
float Saturate(float f);
// D3D9 legacy multiply: +-0 (after flush) times anything is +0.
float LegacyMul(float a, float b);

struct VectorOpResult {
  float v[4] = {0, 0, 0, 0};
  bool sets_predicate = false;
  bool predicate = false;
  bool kill = false;
  bool sets_a0 = false;
  int32_t a0 = 0;
};
// a, b, c: operands after swizzle/abs/negate/flush.
void ExecuteVectorOp(ucode::AluVectorOpcode op, const float a[4],
                     const float b[4], const float c[4], VectorOpResult* r);

struct ScalarOpResult {
  float v = 0.0f;
  bool sets_predicate = false;
  bool predicate = false;
  bool kill = false;
  bool sets_a0 = false;
  int32_t a0 = 0;
};
// a = first (W-derived) operand, b = second; ps = previous scalar.
void ExecuteScalarOp(ucode::AluScalarOpcode op, float a, float b, float ps,
                     ScalarOpResult* r);
bool IsTranscendentalScalarOp(ucode::AluScalarOpcode op);

// Vertex fetch. The element address is computed by vfetch_full (from its
// source register, fetch constant, stride and index rounding) and reused by
// the following vfetch_mini instructions.
struct VertexFetchAddress {
  uint32_t element_base = 0;  // byte address of the element (unwrapped)
  uint32_t buffer_base = 0;   // byte address of the buffer
  uint32_t buffer_end = 0;    // exclusive end of the buffer in bytes
  uint32_t endian = 0;        // xenos::Endian
  int32_t index = 0;
  bool valid = false;
};
VertexFetchAddress ComputeVertexFetchAddress(
    const ucode::VertexFetchInstruction& full, float index_value,
    const RegisterFile& regs);
// Returns false and sets *note for unsupported formats (result then 0).
bool FetchVertex(const ucode::VertexFetchInstruction& op,
                 const VertexFetchAddress& addr, const Memory& memory,
                 float out[4], std::string* note);

std::string FormatFloat(float f);

}  // namespace refxenos::shader_internal
