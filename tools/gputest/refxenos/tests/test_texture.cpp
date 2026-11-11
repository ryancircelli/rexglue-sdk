#include "texture/ref_texture.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace refxenos::tex;

int main() {
  const FormatInfo& info = GetFormatInfo(k_1_REVERSE);
  if (!info.supported || info.block_w != 8 || info.block_h != 1 ||
      info.bits_per_block != 8) {
    std::fprintf(stderr, "1_REVERSE format geometry is not byte-packed 1bpp\n");
    return 1;
  }

  // A 32-texel linear row. Endpoints in each byte distinguish MSB-first
  // (reverse) texel order from the ordinary LSB-first bit order.
  RangeMemory mem;
  mem.Add(0x1000, {0x80, 0x01, 0x00, 0x00});
  uint32_t d[6] = {};
  d[0] = 1u << 22;  // pitch = 32 texels, linear
  d[1] = k_1_REVERSE | (1u << 12);  // base page 0x1000
  d[2] = 15u;  // 16 x 1
  d[3] = 0;  // XXXX swizzle
  d[5] = 1u << 9;  // 2D
  Fetch fetch = Fetch::Parse(d);
  DecodedTexels texels;
  Diag diag;
  if (!DecodeLevel(fetch, mem, 0, 0, &texels, &diag)) {
    std::fprintf(stderr, "1_REVERSE decode failed: %s\n", diag.error.c_str());
    return 1;
  }
  const float expected[16] = {1, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 1};
  if (texels.width != 16 || texels.height != 1 || texels.rgba.size() != 64) {
    std::fprintf(stderr, "1_REVERSE decode dimensions are incorrect\n");
    return 1;
  }
  for (uint32_t x = 0; x < 16; ++x) {
    for (uint32_t c = 0; c < 4; ++c) {
      if (texels.rgba[x * 4 + c] != expected[x]) {
        std::fprintf(stderr, "1_REVERSE texel %u component %u: got %g expected %g\n",
                     x, c, texels.rgba[x * 4 + c], expected[x]);
        return 1;
      }
    }
  }
  std::printf("1_REVERSE: format metadata and 16 reverse-order fetches passed\n");
  return 0;
}
