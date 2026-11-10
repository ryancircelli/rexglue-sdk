# GPU test harness

Tools that find GPU and shader translation bugs in ReXGlue recomps without
playing the game, and name the exact draw or shader. Game agnostic: game data
(captured frames, dumped shaders) is passed in by path and never committed.

Build (macOS, Apple silicon):

```sh
cmake --preset mac-arm64 -DREXGLUE_BUILD_GPUTEST=ON
cmake --build out/build/mac-arm64 --config Release \
  --target rexgpu-shadertool rexgpu-metal-runner rexgpu-replay rexgpu-xenos
```

Needs `spirv-val`, `spirv-cross` (Homebrew) and Xcode's `xcrun metal`. The
native Metal arm also needs `rexgpu-metal-msl-xlate` from the Metal plugin
branch (default path `~/dev/rexglue-sdk-metal/out/mac-arm64/`, override with
`--msl-xlate`; the arm is skipped if missing).

| piece | what it does |
|---|---|
| `replay/` | Frame capture (Xenia `.xtr`), headless replay through any GPU plugin with per-draw readback, and `replay_diff.py` that names the first divergent draw with its VS/PS hashes and state. See `replay/README.md`. |
| `shaderdiff.py` | Per-shader differential test of every shader translation against each other and a CPU reference. |
| `static_checks.py` | Fast CI checks: spirv-val, MSL compiles, reflection and interpolator linkage consistency. |
| `shadertool/shadertool.cpp` | `rexgpu-shadertool`: SDK translation and reflection JSON for one ucode file; runs the SDK shader interpreter. |
| `shadertool/metal_runner.mm` | `rexgpu-metal-runner`: runs JSON-described Metal compute and render jobs (buffers, argument buffer patching, textures, samplers) and dumps raw outputs. |

## Shader input

A directory of Xenos microcode, one shader per file, little-endian dwords:
`ps_<hash>.ucode.bin.frag` and `vs_<hash>.ucode.bin.vert` (the layout the Metal
plugin's shader dump and `rexgpu-metal-msl-xlate` use). Pass `--big-endian` to
`rexgpu-shadertool` for raw guest-memory dumps.

## shaderdiff.py

```sh
python3 tools/gputest/shaderdiff.py --ucode DIR --out OUT [--filter vs_] [--limit N] \
  [--seeds 2] [--arms interp,spirv,native] [--mask-known sdk_exp_bias_word] \
  [--tex gradient|flat] [--tex-filter linear|nearest] [--inject const_offset|interp_swap]
```

Arms, all run on the same random inputs per seed:

- `interp`: SDK `ShaderInterpreter` on the CPU (reference when the shader has
  no texture fetches; it does not model texture fetch results).
- `spirv`: SDK `SpirvShaderTranslator`, SPIRV-Cross to MSL, run on Metal. This
  is the code the Xenos path on MoltenVK runs.
- `native`: the Metal plugin's native MSL translator.

Inputs: random float, bool, loop and fetch constants; vertex shaders read a
random vertex buffer through patched vertex fetch constants (vertex fetch
formats that the game patches at bind time are filled in by
`rexgpu-shadertool --patch-vfetch`) and are run as a compute kernel that
captures every export; pixel shaders draw a 16x16 grid of cells with random
interpolants per cell into RGBA32F targets (plus depth when written), with
gradient textures (every texel, mip, slice and face distinct) and random
texture sign modes bound identically for both ABIs. Fragment inputs are fed by
the linkage contract (native `user(texcoordN)` carries register N, SPIR-V
`user(locnK)` carries the K-th bit of the interpolator mask), so a shader that
reads the wrong slot fails.

Output: `OUT/summary.md` (failure classes with counts and example shaders),
`OUT/results.jsonl` (per shader: classes, first mismatch, suspects), and
`OUT/work/<shader>/` (SPIR-V, both MSLs, jobs). Classes:

- `<a>_vs_<b>:value_mismatch:oN|rtN`, `position_mismatch`, `sign_mismatch`,
  `nan_mismatch`, `kill_mismatch`, `depth_mismatch`, `precision`
- `<a>_vs_<b>:constant_sensitivity`: suspects list the float constant register
  one arm reads and the other does not (found by perturbing each register)
- `<a>_vs_<b>:interpolator_sensitivity`: same for PS input registers
- `*_msl_compile`, `*_harness`, `native_fallback`, `spirv_val*`

Negative controls: `--inject const_offset` shifts the first constant read in
the native MSL by one float (the float2 packing bug class), `--inject
interp_swap` swaps two fragment input slots. Both should light up most
shaders; run them after changing the harness.

Known-bug masks (`--mask-known`) keep a found bug out of the random inputs so
the run shows the next one. `sdk_exp_bias_word` zeroes the texture exponent
bias and LOD bias bit ranges (see Findings).

## static_checks.py

```sh
python3 tools/gputest/static_checks.py --ucode DIR [--msl DIR ...] --out OUT [-j N] \
  [--allow spirv_val_needs_scalar_block_layout,native_fallback]
```

Exit status 1 if a class outside `--allow` fails. About one second per shader
per core. Classes: `spirv_translate`, `spirv_val`,
`spirv_val_needs_scalar_block_layout`, `spirv_cross`, `spirv_msl_compile`,
`native_xlate`, `native_fallback`, `native_msl_compile`,
`reflect_interpolator_mask`, `reflect_color_targets`, `reflect_depth_write`,
`reflect_float_map`, `reflect_texture_fetch`, `interp_linkage`, `msl_compile`
(for `--msl` directories).

## Findings so far (Black Ops II shader set)

1. SDK SPIR-V system constants are not std140: `user_clip_planes` lands at
   offset 28, so every SDK SPIR-V module fails `spirv-val --target-env
   vulkan1.2` unless `--scalar-block-layout`. MoltenVK accepts it (SPIRV-Cross
   packs the struct), but a Vulkan driver without scalarBlockLayout may not.
2. SDK SPIR-V texture exponent bias reads the wrong fetch constant dword:
   `spirv_translator_fetch.cpp` applies `exp_adjust` from dword 4 bits 13..18
   (part of `lod_bias`), the DXBC translator and the native Metal translator
   use dword 3 bits 13..18 (`xenos.h`). Textures with an exponent bias are
   unscaled and textures with a LOD bias get scaled on the Vulkan path.
3. Native Metal translator `maxas`/`maxasf` (scalar) clamp the address
   register to 0..255 instead of -256..255, so negative relative constant
   indexing reads the wrong register. 15 of the 265 bo2 vertex shaders
   (all of them use `c[N+a0]`); `--inject fix_maxa` applies the -256 bound to
   the native MSL and all 15 then match the interpreter.
4. Open: 5 of the first 200 bo2 pixel shaders produce NaN on a few pixels in
   the native arm where SDK SPIR-V is finite (for example ps_0380eef8f8a2b730
   with `--seeds 8`). Not texture inputs or signs (`SHADERDIFF_SIMPLE_FETCH`
   does not change it) and not `fast::` math (`--inject precise_math` does
   not change it); no CPU reference for textured shaders, so which arm is
   right is not settled.
5. Fixed here (951de84c): `ShaderInterpreter` ignored `abs_constants` on
   constant operands, so the CPU oracle disagreed with both GPU arms on
   vs_30c3fbe39a6b58a2.

`SHADERDIFF_METAL_FLAGS` adds flags to every metal compile (for example
`-ffp-contract=off` to rule out fused multiply-add).

Random inputs are seeded from crc32(shader name, seed), so a run is
reproducible; `results.jsonl` is written per batch, so a killed run keeps its
finished batches.
