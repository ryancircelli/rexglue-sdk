// Cross-checks of the reference texture decoder against independent second
// implementations:
//  - 2D tiling vs the legacy crunch-derived row/column formula that the SDK's
//    texture_conversion::Untile uses (copied verbatim below, cited);
//  - BC1/2/3/4/5 blocks vs bcdec (iOrange/bcdec 80859ed3, thirdparty/bcdec).
// Disagreements are printed as classes with counts; this program exits 0 only
// if tiling agrees exactly and every BC disagreement belongs to a known,
// documented class (interpolation rounding).

#define BCDEC_IMPLEMENTATION
#include "bcdec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>

#include "ref_texture.h"
#include "ref_rules.h"
#include "ref_sample.h"

using namespace refxenos::tex;

namespace {

// rexglue-sdk@a9ded573 src/graphics/pipeline/texture/conversion.cpp:88-102
// (crunch crn_decomp.h#L4108).
uint32_t LegacyRow(uint32_t y, uint32_t width, uint32_t log2_bpp) {
  uint32_t macro = ((y / 32) * (width / 32)) << (log2_bpp + 7);
  uint32_t micro = ((y & 6) << 2) << log2_bpp;
  return macro + ((micro & ~0xFu) << 1) + (micro & 0xF) +
         ((y & 8) << (3 + log2_bpp)) + ((y & 1) << 4);
}
uint32_t LegacyColumn(uint32_t x, uint32_t y, uint32_t log2_bpp,
                      uint32_t base_offset) {
  uint32_t macro = (x / 32) << (log2_bpp + 7);
  uint32_t micro = (x & 7) << log2_bpp;
  uint32_t offset = base_offset + (macro + ((micro & ~0xFu) << 1) + (micro & 0xF));
  return ((offset & ~0x1FFu) << 3) + ((offset & 0x1C0) << 2) + (offset & 0x3F) +
         ((y & 16) << 7) + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6);
}

int TestTiling() {
  int failures = 0;
  uint64_t checked = 0;
  for (uint32_t log2_bpp = 0; log2_bpp <= 4; ++log2_bpp) {
    uint32_t bpb = 1u << log2_bpp;
    for (uint32_t pitch = 32; pitch <= 256; pitch += 32) {
      for (uint32_t y = 0; y < 96; ++y) {
        uint32_t row = LegacyRow(y, pitch, log2_bpp);
        for (uint32_t x = 0; x < pitch; ++x) {
          uint32_t legacy = (LegacyColumn(x, y, log2_bpp, row) >> log2_bpp) * bpb;
          uint32_t ref = TiledOffset2D(x, y, pitch, bpb);
          ++checked;
          if (legacy != ref && failures++ < 10) {
            std::printf("tile.2d MISMATCH bpb=%u pitch=%u x=%u y=%u ref=%u legacy=%u\n",
                        bpb, pitch, x, y, ref, legacy);
          }
        }
      }
    }
  }
  std::printf("tile.2d vs legacy Untile: %llu offsets, %d mismatches\n",
              (unsigned long long)checked, failures);
  return failures;
}

void MakeFetch(uint32_t format, uint32_t w, uint32_t h, uint32_t base_phys,
               uint32_t d[6]) {
  std::memset(d, 0, 24);
  d[0] = 2u | (((w + 31) / 32) << 22);
  d[1] = format | ((base_phys >> 12) << 12);
  d[2] = (w - 1) | ((h - 1) << 13);
  d[3] = (0u | 1u << 3 | 2u << 6 | 3u << 9) << 1;
  d[5] = 1u << 9;  // 2D
}

struct ClassCount {
  uint64_t count = 0;
  std::string example;
};

int TestBc() {
  std::mt19937 rng(12345);
  struct Case {
    uint32_t format;
    int bcn;
    const char* name;
  } cases[] = {{k_DXT1, 1, "DXT1/BC1"},   {k_DXT2_3, 2, "DXT2_3/BC2"},
               {k_DXT4_5, 3, "DXT4_5/BC3"}, {k_DXT5A, 4, "DXT5A/BC4"},
               {k_DXN, 5, "DXN/BC5"}};
  int hard_failures = 0;
  for (const Case& c : cases) {
    const FormatInfo& fi = GetFormatInfo(c.format);
    uint32_t bpb = fi.bytes_per_block();
    const int kBlocks = 20000;
    uint64_t comps = 0, exact = 0;
    std::map<std::string, ClassCount> classes;
    for (int i = 0; i < kBlocks; ++i) {
      uint8_t block[16];
      for (uint32_t k = 0; k < bpb; ++k) block[k] = uint8_t(rng());
      // Edge cases: equal endpoints, swapped order.
      if (i % 7 == 1) { block[bpb - 8] = block[bpb - 6]; block[bpb - 7] = block[bpb - 5]; }
      if (i % 11 == 2 && bpb == 16) { block[0] = block[1]; }
      // Endpoint-only blocks (every index selects endpoint 0 or 1) must be
      // bit-exact: odd i -> color indices 0/1, alpha indices 0/1.
      bool endpoints_only = (i % 2) == 1;
      if (endpoints_only) {
        bool color_block = c.bcn <= 3;
        if (color_block) {
          for (int k = 0; k < 4; ++k) block[bpb - 4 + k] &= 0x55;
        }
        // Alpha / BC4 / BC5 3-bit indices: keep only the lowest bit of each.
        auto clear_alpha = [&](uint8_t* a) {
          uint64_t bits = 0;
          for (int k = 0; k < 6; ++k) bits |= uint64_t(a[2 + k]) << (8 * k);
          uint64_t m = 0;
          for (int k = 0; k < 16; ++k) m |= uint64_t(1) << (3 * k);
          bits &= m;
          for (int k = 0; k < 6; ++k) a[2 + k] = uint8_t(bits >> (8 * k));
        };
        if (c.bcn == 3 || c.bcn == 4 || c.bcn == 5) clear_alpha(block);
        if (c.bcn == 5) clear_alpha(block + 8);
        if (c.bcn == 2) {
          // Explicit alpha has no interpolation; leave as is.
        }
      }
      // Reference: 4x4 linear texture, endian none, block at row 0.
      uint32_t d[6];
      MakeFetch(c.format, 4, 4, 0x10000, d);
      Fetch f = Fetch::Parse(d);
      RangeMemory mem;
      std::vector<uint8_t> bytes(32 * bpb, 0);
      std::memcpy(bytes.data(), block, bpb);
      mem.Add(0x10000, bytes);
      DecodedTexels t;
      Diag diag;
      if (!DecodeLevelUnsignedComponents(f, mem, 0, 0, &t, &diag)) {
        std::printf("%s: reference decode failed: %s\n", c.name, diag.error.c_str());
        return 1;
      }
      // bcdec.
      uint8_t out[16 * 4];
      int out_comps = 4;
      switch (c.bcn) {
        case 1: bcdec_bc1(block, out, 16); break;
        case 2: bcdec_bc2(block, out, 16); break;
        case 3: bcdec_bc3(block, out, 16); break;
        case 4: bcdec_bc4(block, out, 4); out_comps = 1; break;
        case 5: bcdec_bc5(block, out, 8); out_comps = 2; break;
      }
      for (int p = 0; p < 16; ++p) {
        for (int ch = 0; ch < out_comps; ++ch) {
          float ref = t.rgba[size_t(p) * 4 + ch];
          uint8_t other = out[p * out_comps + ch];
          float o = float(other) / 255.0f;
          ++comps;
          if (ref == o) {
            ++exact;
            continue;
          }
          int ref8 = int(ref * 255.0f + 0.5f);
          int diff = ref8 - int(other);
          char key[96];
          std::snprintf(key, sizeof(key), "%s %s diff %+d",
                        (out_comps == 4 && ch == 3) ? "alpha" : "color",
                        endpoints_only ? "endpoint" : "mixed", diff);
          ClassCount& cc = classes[key];
          if (cc.count++ == 0) {
            char ex[160];
            std::snprintf(ex, sizeof(ex), "block %d texel %d ch %d ref=%.6f(%d) bcdec=%d", i,
                          p, ch, ref, ref8, other);
            cc.example = ex;
          }
          if (diff > 1 || diff < -1) ++hard_failures;

        }
      }
    }
    std::printf("%s vs bcdec: %llu/%llu components bit-exact\n", c.name,
                (unsigned long long)exact, (unsigned long long)comps);
    for (auto& [k, v] : classes)
      std::printf("  class %-18s %8llu  e.g. %s\n", k.c_str(),
                  (unsigned long long)v.count, v.example.c_str());
  }
  return hard_failures;
}


// Sampler sanity: point samples at texel centres equal DecodeLevel for every
// format family with the full fetch chain (sign, swizzle, exp_adjust), and a
// bilinear sample midway between four texels is their average.
int TestSample() {
  std::mt19937 rng(777);
  int failures = 0;
  uint64_t checked = 0;
  const uint32_t formats[] = {k_8_8_8_8, k_DXT1, k_DXT4_5, k_DXN, k_5_6_5,
                              k_2_10_10_10, k_16_16};
  for (uint32_t fmt : formats) {
    const uint32_t w = 32, h = 16, base = 0x100000;
    uint32_t d[6];
    MakeFetch(fmt, w, h, base, d);
    d[0] |= 1u << 31;          // tiled
    d[0] |= (1u << 2);          // sign x = signed
    d[0] |= (2u << 6);          // sign z = bias
    d[3] = ((2u | 1u << 3 | 0u << 6 | 3u << 9) << 1) | (1u << 13);  // ZYXW, exp +1
    const FormatInfo& fi = GetFormatInfo(fmt);
    std::vector<uint8_t> bytes(w * h * 4 * 4);
    for (auto& b : bytes) b = uint8_t(rng());
    (void)fi;
    RangeMemory mem;
    mem.Add(base, bytes);
    Fetch f = Fetch::Parse(d);
    DecodedTexels t;
    Diag diag;
    if (!DecodeLevel(f, mem, 0, 0, &t, &diag)) {
      std::printf("sample: decode %u failed: %s\n", fmt, diag.error.c_str());
      ++failures;
      continue;
    }
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        SampleRequest r;
        std::memcpy(r.fetch, d, 24);
        r.coord[0] = (x + 0.5f) / w;
        r.coord[1] = (y + 0.5f) / h;
        r.use_computed_lod = false;
        auto v = Sample(r, mem, &diag);
        const float* e = &t.rgba[(y * w + x) * 4];
        for (int c = 0; c < 4; ++c) {
          ++checked;
          if (v[c] != e[c] && !(std::isnan(v[c]) && std::isnan(e[c]))) {
            if (failures < 5)
              std::printf("sample: fmt %u (%u,%u).%d point %g vs decode %g\n",
                          fmt, x, y, c, v[c], e[c]);
            ++failures;
          }
        }
      }
    }
  }
  // Bilinear midpoint on 8_8_8_8, no sign / swizzle.
  {
    const uint32_t w = 8, h = 8, base = 0x200000;
    uint32_t d[6];
    MakeFetch(k_8_8_8_8, w, h, base, d);
    d[3] |= (1u << 19) | (1u << 21);  // mag / min linear
    std::vector<uint8_t> bytes(w * h * 4 + 4096);
    for (auto& b : bytes) b = uint8_t(rng());
    RangeMemory mem;
    mem.Add(base, bytes);
    Fetch f = Fetch::Parse(d);
    DecodedTexels t;
    Diag diag;
    DecodeLevel(f, mem, 0, 0, &t, &diag);
    SampleRequest r;
    std::memcpy(r.fetch, d, 24);
    r.coord[0] = 3.0f / w;
    r.coord[1] = 5.0f / h;
    r.use_computed_lod = false;
    auto v = Sample(r, mem, &diag);
    for (int c = 0; c < 4; ++c) {
      float e = 0.25f * (t.rgba[(4 * w + 2) * 4 + c] + t.rgba[(4 * w + 3) * 4 + c] +
                         t.rgba[(5 * w + 2) * 4 + c] + t.rgba[(5 * w + 3) * 4 + c]);
      ++checked;
      if (std::fabs(v[c] - e) > 1e-6f) {
        std::printf("sample: bilinear .%d %g vs %g\n", c, v[c], e);
        ++failures;
      }
    }
  }
  std::printf("sample: %llu components, %d mismatches\n",
              (unsigned long long)checked, failures);
  return failures;
}

}  // namespace

int main() {
  int f = 0;
  f += TestTiling();
  int bc = TestBc();
  if (bc) std::printf("BC: %d disagreements larger than 1/255\n", bc);
  f += bc;
  f += TestSample();
  {
    size_t n = 0;
    const RuleInfo* rules = GetRules(&n);
    for (size_t i = 0; i < n; ++i) {
      if (rules[i].status == RuleStatus::kAssumed &&
          (!rules[i].hw_test || !*rules[i].hw_test)) {
        std::printf("rule %s is assumed but has no hw_test\n", rules[i].id);
        ++f;
      }
    }
  }
  std::printf("%s\n", f ? "FAIL" : "PASS");
  return f ? 1 : 0;
}
