#!/usr/bin/env python3
"""Extract raw Xenos microcode from Xbox 360 XDK shader containers.

Usage: xdk_extract_ucode.py IN_DIR OUT_DIR [--skip-existing]

IN_DIR holds containers named ps_<hash>.bin / vs_<hash>.bin (big-endian magic
0x102A1100 pixel, 0x102A1101 vertex). Writes <name>.ucode.bin.frag (pixel) or
.ucode.bin.vert (vertex) as host-endian dwords, the input of shaderdiff.py,
static_checks.py and rexgpu-metal-msl-xlate.
"""
import os, struct, sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = "--skip-existing" in sys.argv
    if len(args) != 2:
        sys.exit(__doc__)
    src, dst = args
    os.makedirs(dst, exist_ok=True)
    ok = bad = skipped = 0
    for n in sorted(os.listdir(src)):
        if not n.endswith(".bin") or n[:3] not in ("ps_", "vs_"):
            continue
        out = os.path.join(dst, n[:-4] + ".ucode.bin" + (".frag" if n.startswith("ps_") else ".vert"))
        if skip and os.path.exists(out):
            skipped += 1
            continue
        d = open(os.path.join(src, n), "rb").read()
        try:
            # header: magic/flags, vertex-data size, program size, ..., offset of the program descriptor
            _, vsize, _, _, _, _, desc = struct.unpack_from(">7I", d, 0)
            poff, size = struct.unpack_from(">2I", d, desc)
            code = d[vsize + poff:vsize + poff + size]
            if size == 0 or size % 4 or len(code) != size:
                raise ValueError("bad program size %d" % size)
            words = struct.unpack(">%dI" % (size // 4), code)
            open(out, "wb").write(struct.pack("<%dI" % len(words), *words))
            ok += 1
        except Exception as e:  # noqa: BLE001
            print("bad", n, e)
            bad += 1
    print("extracted %d bad %d skipped %d" % (ok, bad, skipped))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
