# CPU Xenos replay

Build the standalone tool through the shared build queue:

```sh
cmake -S tools/gputest/refxenos -B out/refxenos -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
~/Games/Xbox360/bo2-build/runq/bin/buildq -- ninja -C out/refxenos
ctest --test-dir out/refxenos --output-on-failure
```

List draws in a captured frame and locate a runtime pixel-shader hash:

```sh
out/refxenos/refxenos replay capture.xtr -o /tmp/bo2/refxenos-example/draw-list --frame 200 --list --dump-shaders
out/refxenos/refxenos replay capture.xtr -o /tmp/bo2/refxenos-example/fence-list --frame 200 --ps 1902C1FF51DA679D --list
```

`--dump-shaders` saves host-endian microcode and both byte orders' XXH3/XXH64
hashes. A filter with no matching draws returns failure and writes `ok:false`.
Runtime hashes in `draws.jsonl` use XXH3 on guest big-endian microcode.

Render a selected draw through the diagnostic overwrite stub:

```sh
out/refxenos/refxenos replay capture.xtr -o /tmp/bo2/refxenos-example/draw --frame 200 --draws 27388:1 --rop shader --full
```

The driver executes captured PM4 state and memory updates, runs vertex fetch,
VS, triangle rasterization, texture decode/sampling, and quad PS execution. It
writes `draws.jsonl`, `draws/NNNNNN_rt0.npy` (raw RGBA32F),
`NNNNNN_coverage.npy` (guest sample bits for surviving pixels), a clamped
`NNNNNN_rt0.png`, and the used rule versions. `--probe X,Y --probe-out FILE`
records shader/raster diagnostics. The readback extent follows the SDK replay
writer's scissor, pitch, and EDRAM capacity limits.

**This stub is a pipeline diagnostic, not an EDRAM reference.** It overwrites
RT0 with surviving PS exports wherever any sample is covered. It skips depth,
stencil, blending, alpha testing, and render-target conversion. Selection
filters also skip earlier draws and resolves; resolved textures and prior
EDRAM contents may therefore be missing. These limitations are recorded in
every selected draw. Compare it with Metal only with those differences
accounted for. A nearly black preview can be a linear intermediate target;
use the raw NPY to inspect its values.

`--rop edram` uses `DecodeRopState`, `RopQuad`, and `Resolve` when that module
is present in the checkout. The overwrite stub also builds without the
optional ROP/resolve files. Full reference status additionally requires
validation of that module and propagation of vertex taint through raster.

Raster coverage uses integer edge equations, D3D top-left ownership,
perspective-correct interpolants, and guest sample order. Arithmetic precision
and Xenos's inferred MSAA sample positions have explicit assumed rules and
hardware tests; inspect `refxenos --rules-json` for citations. Unit tests cover
per-sample shared edges at 1x/2x/4x, both windings/diagonals, clipping, viewport,
scissor, centroid/flat interpolation, and depth. The replay integration test
uses hand-assembled ucode and checks a red triangle's 28 covered pixels exactly.
