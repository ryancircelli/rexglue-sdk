#include "ref_rules.h"

#include <cstring>

namespace refxenos::tex {

// SDK citations are rexglue-sdk at a9ded573 (Xenia-derived code).
// XDK citations are pages of the Xbox 360 SDK documentation (xbox360sdk.chm,
// version 21256.3).
static const RuleInfo kRules[] = {
    {"fetch.fields", 1, RuleStatus::kDocumented,
     "Fetch constant bit layout, LSB-first per dword.",
     "XDK GPUTEXTURE_FETCH_CONSTANT.htm; SDK include/rex/graphics/xenos.h "
     "xe_gpu_texture_fetch_t"},
    {"mip.range", 1, RuleStatus::kXenia,
     "Addressable levels: mip page 0 -> base only; min/max clamped to "
     "log2(longest axis); base page 0 with mips -> min level 1.",
     "SDK src/graphics/pipeline/texture/util.cpp:20-108 "
     "GetSubresourcesFromFetchConstant"},
    {"mip.layout", 1, RuleStatus::kXenia,
     "Base pitch = fetch pitch*32 texels; mips padded to next_pow2(base)>>L, "
     "32x32 block tiles; linear mip rows 256-byte aligned; layer stride 4 KB "
     "aligned; mips stored consecutively from the mip address.",
     "SDK src/graphics/pipeline/texture/util.cpp:202-421 "
     "GetGuestTextureLayout; include/rex/graphics/xenos.h:1186-1200"},
    {"mip.packed", 1, RuleStatus::kXenia,
     "Levels whose shorter side is <=16 share one 32x32 tail tile; offsets "
     "by the SDK's GetPackedMipOffset.",
     "SDK src/graphics/pipeline/texture/util.cpp:110-200; "
     "include/rex/graphics/pipeline/texture/util.h:34 GetPackedMipLevel; "
     "XDK XGGetMipTailLevelOffset.htm (existence only)"},
    {"tile.2d", 1, RuleStatus::kXenia,
     "2D tiled address (UModel / crunch formula).",
     "SDK src/graphics/pipeline/texture/util.cpp:424-436 GetTiledOffset2D; "
     "XDK XGAddress2DTiledOffset.htm (signature only)"},
    {"tile.3d", 1, RuleStatus::kXenia,
     "3D tiled address (XGRAPHICS::TileVolume disassembly).",
     "SDK src/graphics/pipeline/texture/util.cpp:438-459 GetTiledOffset3D"},
    {"linear.address", 1, RuleStatus::kXenia,
     "Linear address = row * row_pitch_bytes + x * bytes_per_block.",
     "SDK src/graphics/pipeline/texture/util.cpp:290-320"},
    {"endian", 1, RuleStatus::kDocumented,
     "Endian mode swaps bytes of each 16/32-bit memory word "
     "(8in16, 8in32, 16in32).",
     "XDK GPUENDIAN.htm"},
    {"format.layout", 1, RuleStatus::kXenia,
     "Packed formats: X in the least significant bits after the endian "
     "swap; the name lists fields from the most significant (W) down.",
     "SDK src/graphics/vulkan/texture_cache.cpp:109-330 host format "
     "conversions (e.g. R5G6B5ToB5G6R5, R11G11B10ToRGBA16)"},
    {"format.replicate", 1, RuleStatus::kXenia,
     "Components a format lacks repeat its last one: 1 component -> XXXX, "
     "2 -> XYYY, 3 -> XYZZ.",
     "SDK src/graphics/vulkan/texture_cache.cpp:109-330 swizzles "
     "RRRR/RGGG/RGBB"},
    {"norm.unsigned", 1, RuleStatus::kAssumed,
     "Unsigned n-bit fraction = v / (2^n - 1), correctly rounded to f32.",
     "assumed (standard unorm; precision of the Xenos texture pipe "
     "unknown)",
     "Upload a linear 256x1 k_8 texture with texel x = x; point-fetch each texel centre in a pixel shader writing to a k_32_32_32_32_FLOAT render target; resolve and compare each value with x/255 exactly (also k_16 with 65536 values against x/65535)."},
    {"norm.signed", 1, RuleStatus::kXenia,
     "Signed n-bit fraction = max(v / (2^(n-1) - 1), -1) (zero-clamp-minus-"
     "one; the most negative code also maps to -1).",
     "SDK include/rex/graphics/xenos.h:1176-1182 (R400 doc change 133990)"},
    {"sign.bias", 2, RuleStatus::kDocumented,
     "GPUSIGN_BIAS: 2 * x - 1 on the unsigned value.",
     "XDK GPUSIGN.htm; SDK spirv_translator_fetch.cpp:1995-2001"},
    {"sign.gamma", 2, RuleStatus::kXenia,
     "GPUSIGN_GAMMA: piecewise-linear gamma to linear on the unsigned value.",
     "SDK src/graphics/xenos.cpp:27-68 PWLGammaToLinear; "
     "spirv_translator_fetch.cpp:2002-2008"},
    {"num.integer", 1, RuleStatus::kAssumed,
     "GPUNUMFORMAT_INTEGER returns the raw (sign-extended if signed) "
     "integer instead of a fraction.",
     "XDK GPUNUMFORMAT.htm (meaning only; Xenia ignores the bit)",
     "Upload k_8_8_8_8 and k_16 textures with NumFormat=integer; point-fetch every value 0..255 / 0..65535 into a k_32_32_32_32_FLOAT target; record whether the result is the raw integer or a fraction."},
    {"swizzle", 1, RuleStatus::kDocumented,
     "Destination selects X/Y/Z/W/0/1; signs belong to data components "
     "before the swizzle. Undefined selectors 6/7 act as 0/1 (value & 0b101).",
     "XDK GPUSWIZZLE.htm; SDK src/graphics/pipeline/texture/cache.cpp:357-361 "
     "GuestToHostSwizzle (6/7 handling); SDK util.cpp:518-536 SwizzleSigns"},
    {"exp_adjust", 1, RuleStatus::kXenia,
     "Result components other than constant 0 are multiplied by "
     "2^exp_adjust (fetch constant dword 3 bits 13:18, signed), constant 1 "
     "included.",
     "SDK spirv_translator_fetch.cpp:2034-2050"},
    {"dxt.endpoints", 1, RuleStatus::kAssumed,
     "565 endpoints expand to 8 bits by bit replication before "
     "interpolation.",
     "assumed (D3D9 reference behaviour; XDK formulas do not state the "
     "precision)",
     "Upload one DXT1 block per 565 endpoint value (c0=v, c1=0, all indices 0) for all 32 red, 64 green, 32 blue codes; point-fetch into a float target; record the 8-bit value per code (bit replication vs rounding)."},
    {"dxt1.color", 1, RuleStatus::kDocumented,
     "c0>c1: c2=(2c0+c1+1)/3, c3=(c0+2c1+1)/3; else c2=(c0+c1)/2, "
     "c3=transparent black. Integer math per 8-bit channel.",
     "XDK Opaque_and_1_Bit_Alpha_Textures.htm"},
    {"dxt1.transparent", 1, RuleStatus::kAssumed,
     "The DXT1 transparent texel reads (0,0,0,0).",
     "XDK Opaque_and_1_Bit_Alpha_Textures.htm recommends zero RGB; value "
     "the hardware returns is assumed",
     "Upload a DXT1 block with c0<=c1 and all indices 3; point-fetch into a float target; record RGB and alpha of the transparent texel (expect 0,0,0,0 or c-colour with a=0)."},
    {"dxt23.color", 1, RuleStatus::kDocumented,
     "DXT2/3/4/5 color blocks always use 4-color mode.",
     "XDK Textures_with_Alpha_Channels.htm"},
    {"dxt3.alpha", 1, RuleStatus::kDocumented,
     "Explicit 4-bit alpha, word0 bits 3:0 = texel [0][0]; value = a/15.",
     "XDK Textures_with_Alpha_Channels.htm"},
    {"dxt5.alpha", 1, RuleStatus::kDocumented,
     "a0>a1: 8 alphas with (k*a0+(7-k)*a1+3)/7; else 6 alphas with "
     "(k*a0+(5-k)*a1+2)/5, 0, 255.",
     "XDK Textures_with_Alpha_Channels.htm"},
    {"dxn", 1, RuleStatus::kDocumented,
     "DXN = two DXT5A blocks, first is X, second is Y.",
     "XDK Supported_Compressed_Texture_Formats.htm; SDK vulkan "
     "texture_cache.cpp k_DXN -> BC5"},
    {"dxt3a", 1, RuleStatus::kDocumented,
     "DXT3A = one DXT2/3 alpha block as a single component.",
     "XDK Supported_Compressed_Texture_Formats.htm"},
    {"dxt3a_1111", 1, RuleStatus::kAssumed,
     "DXT3A_AS_1_1_1_1: nibble bit i -> component i (bit 0 -> X).",
     "XDK Supported_Compressed_Texture_Formats.htm says each bit becomes a "
     "channel; the bit order is assumed",
     "Upload a DXT3A_AS_1_1_1_1 block with a single alpha nibble bit set per texel (bits 0..3 cycling); point-fetch into a float target; record which output channel each bit lands in."},
    {"dxt5a", 1, RuleStatus::kDocumented,
     "DXT5A = one DXT4/5 alpha block as a single component.",
     "XDK Supported_Compressed_Texture_Formats.htm"},
    {"ctx1", 1, RuleStatus::kXenia,
     "CTX1 block: bytes g0 r0 g1 r1 then 2-bit indices; X=r, Y=g; "
     "c2=trunc(2/3*e0+1/3*e1), c3=trunc(1/3*e0+2/3*e1) in f32.",
     "SDK src/graphics/pipeline/texture/conversion.cpp:47-78 "
     "ConvertTexelCTX1ToR8G8; XDK Supported_Compressed_Texture_Formats.htm"},
    {"compressed.signed", 1, RuleStatus::kAssumed,
     "Signed compressed formats: decoded 8-bit values reinterpreted as "
     "two's complement (snorm view of the decoded data).",
     "assumed (matches a snorm view of decompressed data; BC4/5 snorm "
     "interpolation not modelled)",
     "Upload DXN / DXT5A / CTX1 blocks covering endpoint bytes 0x00, 0x7F, 0x80, 0x81, 0xFF with every index; fetch with SignX/Y=signed into a float target; record the values (two's complement snorm vs biased)."},
    {"depth.24_8", 1, RuleStatus::kXenia,
     "k_24_8: X = (v >> 8) / (2^24 - 1); k_24_8_FLOAT: X = 20e4 float in "
     "bits 31:8.",
     "SDK vulkan texture_cache.cpp:116-128 DepthUnorm/DepthFloat; "
     "src/graphics/xenos.cpp:154-169 Float20e4To32"},
    {"float16", 1, RuleStatus::kXenia,
     "_FLOAT and _EXPAND 16-bit formats are IEEE half floats.",
     "SDK vulkan texture_cache.cpp:149-178 (EXPAND copied as SFLOAT)"},
    {"float32", 1, RuleStatus::kDocumented, "_FLOAT 32-bit formats are IEEE "
     "single floats.", "XDK GPUTEXTUREFORMAT.htm"},
    {"format.as16", 1, RuleStatus::kXenia,
     "_AS_16_16_16_16 formats hold the same data as their base format.",
     "SDK src/graphics/pipeline/texture/cache.cpp:972 GetBaseFormat"},
    {"format.10_11_11", 1, RuleStatus::kXenia,
     "10_11_11 / 11_11_10 are fixed-point fractions (not floats).",
     "SDK vulkan texture_cache.cpp R11G11B10ToRGBA16 / R10G11B11ToRGBA16"},
    {"sample.address_mirror_clamp", 1, RuleStatus::kAssumed,
     "Mirror-clamp modes (to edge / halfway / border) mirror once, then clamp to the edge texel.",
     "SDK vulkan/texture_cache.cpp:846-862 (Xenia maps all three to mirror-clamp-to-edge)",
     "Upload an 8x1 k_8 ramp; for each mirror-clamp mode (edge, halfway, border) point-fetch u from -2.0 to 3.0 in 1/64 steps; record the texel index chosen."},
    {"sample.address_halfway", 1, RuleStatus::kAssumed,
     "Clamp-to-halfway addresses like clamp-to-edge.",
     "SDK vulkan/texture_cache.cpp:846-862",
     "Upload an 8x1 k_8 ramp; bilinear-fetch with ClampToHalfway at u in [-0.25, 1.25] in 1/256 steps; compare with clamp-to-edge."},
    {"sample.border", 1, RuleStatus::kAssumed,
     "Border texels take the fetch border colour (0 black a=0, 1 white, 2 (0.5,0,0.5,0), 3 (0,0.5,0.5,0)) before the swizzle.",
     "SDK vulkan/texture_cache.cpp:886-925; XDK GPUBORDERCOLOR",
     "Upload a 4x4 k_8_8_8_8 texture with ClampToBorder and each BorderColor 0..3 and swizzles XYZW, WZYX; fetch at u=-0.5 and u=1.5; record the returned value per channel."},
    {"sample.subtexel", 1, RuleStatus::kAssumed,
     "Texel-space coordinates and the mip fraction are truncated to 8 fractional bits.",
     "SDK pipeline/shader/spirv_translator_fetch.cpp:678-681 (8-bit subtexel precision comment)",
     "Upload a 2x1 k_8 texture (0, 255); bilinear-fetch at 4096 evenly spaced u between the texel centres; record the distinct weights (expect 256 steps if 8-bit) and whether they truncate or round."},
    {"sample.unnormalized", 1, RuleStatus::kXenia,
     "Unnormalized coordinates are divided by the level-0 size.",
     "SDK pipeline/shader/spirv_translator_fetch.cpp (unnormalized path)"},
    {"sample.offset", 1, RuleStatus::kXenia,
     "Instruction offsets are in level-0 texels (half-texel steps).",
     "SDK pipeline/shader/spirv_translator_fetch.cpp:665-720"},
    {"sample.lod_bias", 1, RuleStatus::kXenia,
     "LOD = computed (or register) LOD + fetch LODBias/32 + instruction bias.",
     "SDK pipeline/shader/spirv_translator_fetch.cpp:1427-1452; XDK GPUTEXTURE_FETCH_CONSTANT LODBias"},
    {"sample.lod", 1, RuleStatus::kAssumed,
     "Computed LOD = log2 of the larger of |ddx*size| and |ddy*size| (Euclidean).",
     "D3D9 / OpenGL isotropic LOD formula; Xenos precision unknown",
     "Draw a quad with a 256x256 mip-chain texture whose levels are solid distinct colours, using getCompTexLOD with explicit gradients over a grid of (ddx, ddy) magnitudes and angles; record the returned LOD (isotropic formula and precision)."},
    {"sample.grad_exp_adjust", 1, RuleStatus::kAssumed,
     "GradExpAdjustH/V scale the gradients by 2^n before the LOD.",
     "XDK GPUTEXTURE_FETCH_CONSTANT GradExpAdjustH/V; Xenia ignores them",
     "Repeat the getCompTexLOD gradient grid with GradExpAdjustH/V = +1, -1; record whether the LOD shifts by exactly +-1."},
    {"sample.lod_clamp", 1, RuleStatus::kXenia,
     "LOD is clamped to [min_mip_level, max_mip_level]; base map filter uses min_mip_level.",
     "SDK vulkan/texture_cache.cpp:870-880"},
    {"sample.mag_min", 1, RuleStatus::kAssumed,
     "LOD <= 0 uses the mag filter and the base level, LOD > 0 the min filter.",
     "D3D9 / OpenGL convention",
     "With MagFilter=point, MinFilter=linear on a 2x2 texture, fetch with LOD bias -0.01, 0, +0.01; record which filter applies at LOD 0."},
    {"sample.mip_linear", 1, RuleStatus::kXenia,
     "Linear mip filter blends floor(LOD) and the next level by the LOD fraction.",
     "D3D9 convention; SDK vulkan sampler mipmapMode"},
    {"sample.mip_point", 1, RuleStatus::kAssumed,
     "Point mip filter picks floor(LOD + 0.5).",
     "OpenGL nearest-mipmap rule",
     "Use a mip chain with solid colours per level and MipFilter=point; fetch with register LOD from 0 to 3 in 1/64 steps; record the level-switch points (expect x.5 if round-to-nearest)."},
    {"sample.aniso", 1, RuleStatus::kAssumed,
     "Anisotropic filtering is evaluated as the trilinear sample.",
     "Not modelled; no source for the Xenos anisotropic footprint",
     "Draw a strongly anisotropic quad (16:1 gradients) with AnisoFilter=max 16:1 on a striped texture; compare the result with the trilinear fetch at the same LOD."},
    {"sample.cube", 1, RuleStatus::kAssumed,
     "Cube maps sample within the resolved face; no filtering across faces.",
     "Not modelled; cube faces resolved by the caller",
     "Upload a cube map with distinct face colours; bilinear-fetch directions grazing each face edge; record whether the result blends across faces."},
    {"sample.cube_face_clamp", 1, RuleStatus::kXenia,
     "Cube faces clamp to the edge on every axis.",
     "SDK pipeline/texture/util.cpp:559-579 GetClampModesForDimension"},
    {"sample.stacked", 1, RuleStatus::kAssumed,
     "Stacked 2D with a 3D fetch picks layer floor(w*layers); a linear volume filter blends two layers.",
     "SDK pipeline/texture/cache.cpp stacked binding; blend assumed",
     "Upload a 4-layer stacked 2D texture with solid colours per layer; fetch with a 3D coordinate w from 0 to 1 in 1/64 steps with VolMinFilter point and linear; record the layer choice and blend."},
    {"sample.weights", 1, RuleStatus::kXenia,
     "getWeights returns the filter fractions (x, y, z) and the mip fraction (w).",
     "XDK getWeights; SDK spirv_translator_fetch.cpp kGetTextureWeights"},
    {"sample.order", 1, RuleStatus::kXenia,
     "tfetch: filter, swizzle, then bias / gamma per swizzled component, then exp_adjust.",
     "SDK pipeline/shader/spirv_translator_fetch.cpp:1940-2050; metal_msl_translator.cc:3187-3209"},
};

const RuleInfo* GetRules(size_t* count) {
  *count = sizeof(kRules) / sizeof(kRules[0]);
  return kRules;
}

const RuleInfo* FindRule(const char* id) {
  for (const RuleInfo& rule : kRules) {
    if (!std::strcmp(rule.id, id)) return &rule;
  }
  return nullptr;
}

const char* RuleStatusName(RuleStatus status) {
  switch (status) {
    case RuleStatus::kDocumented:
      return "documented";
    case RuleStatus::kXenia:
      return "xenia";
    case RuleStatus::kAssumed:
      return "assumed";
  }
  return "?";
}

}  // namespace refxenos::tex
