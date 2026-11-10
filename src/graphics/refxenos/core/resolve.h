// refxenos EDRAM resolve and render-target readback.
#pragma once

#include <vector>

#include "core/common.h"
#include "core/edram.h"

namespace refxenos {

struct ResolveRecord {
  uint32_t dest_base = 0, dest_bytes = 0;
  uint32_t dest_format = 0;
  bool dest_tiled = false;
  uint32_t dest_pitch = 0, dest_height = 0;
  int32_t exp_bias = 0;
  uint32_t src_base_tile = 0, src_pitch_tiles = 0;
  bool src_is_depth = false;
  uint32_t src_format = 0;
  uint32_t x0 = 0, y0 = 0, width = 0, height = 0;
  uint32_t sample_select = 0;
  bool cleared_color = false, cleared_depth = false;
  uint32_t clear_color_lo = 0, clear_color_hi = 0, clear_depth = 0;
};

bool Resolve(const RegisterFile& regs, Edram& edram, Memory& memory,
             ResolveRecord* record, DrawStats& stats);

std::vector<float> ReadbackColorF32(const Edram& edram, const RegisterFile& regs,
                                    uint32_t rt_index, uint32_t width, uint32_t height);
std::vector<uint32_t> ReadbackDepthStencilRaw(const Edram& edram, const RegisterFile& regs,
                                              uint32_t width, uint32_t height);

}  // namespace refxenos
