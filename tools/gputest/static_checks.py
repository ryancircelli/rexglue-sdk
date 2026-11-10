#!/usr/bin/env python3
"""Fast static shader checks for CI (no GPU work, no running game).

For each Xenos ucode file (vs_*.bin / *.ucode.bin.vert, ps_*.bin /
*.ucode.bin.frag) in --ucode:

  spirv_translate        SDK SpirvShaderTranslator fails or emits nothing
  spirv_val              spirv-val --target-env vulkan1.2 (the scalar block
                         layout failure is reported as its own class,
                         spirv_val_needs_scalar_block_layout)
  spirv_cross            SPIRV-Cross to MSL fails
  spirv_msl_compile      xcrun metal on the SPIRV-Cross MSL fails
  native_xlate           native Metal translator (rexgpu-metal-msl-xlate)
                         error; a fallback is reported as native_fallback
  native_msl_compile     xcrun metal on the native MSL fails
  reflect_*              native sidecar disagrees with the SDK shader analysis:
                         interpolator mask, color target mask, depth write,
                         float constant map (register -> packed index),
                         texture fetch constant set
  interp_linkage         the PS reads an interpolator register the matching
                         mask does not provide (native user(texcoordN) or
                         SPIR-V location count)

Also, for directories of ready MSL (--msl DIR, repeatable), compiles every
.metal file and reports failures (useful for the bo2 native-metal output).

Output: <out>/static.jsonl, <out>/static.md. Exit code 1 if any check other
than the allow-listed classes (--allow) failed.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shaderdiff  # noqa: E402  (shares discovery, translation helpers)


def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def first_error(text):
    for line in text.splitlines():
        if "error" in line:
            return line.strip()[:300]
    return text.strip()[-300:]


AIR_DIR = None


def metal_compile(src):
    # The .air goes to our own output directory: --msl inputs may belong to
    # another tool or agent and are never written to.
    air = os.path.join(AIR_DIR, "%x.air" % (hash(src) & 0xFFFFFFFFFFFF))
    rc, _, err = run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-fno-fast-math",
                      "-Wno-unused-variable", "-c", src, "-o", air])
    try:
        os.remove(air)
    except OSError:
        pass
    return None if rc == 0 else first_error(err)


def check_shader(args, path):
    sh = shaderdiff.Shader(path, os.path.join(args.out, "work"))
    name, typ = sh.name, sh.type
    work = sh.work
    os.makedirs(work, exist_ok=True)
    res = {"shader": name, "type": typ, "fail": {}}

    def fail(cls, msg):
        res["fail"][cls] = msg

    pre = os.path.join(work, typ)
    cmd = [args.shadertool, "translate", path, "--type", typ, "--out", pre]
    if typ == "vs" and not args.no_patch_vfetch:
        cmd.append("--patch-vfetch")
    rc, out, err = run(cmd)
    if rc != 0 or not os.path.exists(pre + ".json"):
        fail("spirv_translate", first_error(err or out))
        return res
    info = json.load(open(pre + ".json"))
    if not info.get("ok") or not os.path.exists(pre + ".spv"):
        fail("spirv_translate", "; ".join(info.get("errors", [])) or "no SPIR-V")
        return res
    ucode = pre + ".ucode.bin.vert" if os.path.exists(pre + ".ucode.bin.vert") else path

    rc, out, err = run(["spirv-val", "--target-env", "vulkan1.2", pre + ".spv"])
    if rc != 0:
        msg = first_error(err or out)
        rc2, _, _ = run(["spirv-val", "--target-env", "vulkan1.2", "--scalar-block-layout", pre + ".spv"])
        fail("spirv_val_needs_scalar_block_layout" if rc2 == 0 else "spirv_val", msg)

    msl = pre + ".spirv.metal"
    rc, out, err = run(["spirv-cross", "--msl", "--msl-version", "20300", pre + ".spv", "--output", msl])
    if rc != 0:
        fail("spirv_cross", first_error(err or out))
    else:
        e = metal_compile(msl)
        if e:
            fail("spirv_msl_compile", e)

    spirv_locations = 0
    if os.path.exists(msl):
        spirv_locations = len(set(re.findall(r"\[\[user\(locn(\d+)\)\]\]", open(msl).read())))

    if args.msl_xlate:
        nat = pre + ".native.metal"
        sh.info = info
        mod = shaderdiff.native_modification(sh)
        rc, out, err = run([args.msl_xlate, ucode if typ == "vs" else path, mod, nat])
        text = (out + err).strip()
        if rc == 2:
            fail("native_fallback", text.splitlines()[-1][:300] if text else "fallback")
        elif rc != 0 or not os.path.exists(nat):
            fail("native_xlate", first_error(text))
        else:
            e = metal_compile(nat)
            if e:
                fail("native_msl_compile", e)
            side = json.load(open(nat + ".json")) if os.path.exists(nat + ".json") else None
            if side:
                reflect(info, side, typ, fail)
                linkage(info, open(nat).read(), typ, fail)
    if typ == "ps" and spirv_locations:
        want = bin(int(info["interpolator_mask"])).count("1")
        if spirv_locations > want:
            fail("interp_linkage", "SPIR-V PS reads %d locations, mask provides %d" % (spirv_locations, want))
    return res


def reflect(info, side, typ, fail):
    if int(side.get("interpolator_mask", 0)) != int(info["interpolator_mask"]):
        fail("reflect_interpolator_mask", "native %x sdk %x" % (int(side["interpolator_mask"]),
                                                                int(info["interpolator_mask"])))
    if typ == "ps":
        if "colors" in side and int(side["colors"]) & 0xF != int(info["writes_color_targets"]) & 0xF:
            fail("reflect_color_targets", "native %x sdk %x" % (int(side["colors"]), int(info["writes_color_targets"])))
        if "depth" in side and bool(side["depth"]) != bool(info["writes_depth"]):
            fail("reflect_depth_write", "native %s sdk %s" % (side["depth"], info["writes_depth"]))
    sdk_map = {int(a): int(b) for a, b in info["constants"].get("float_map", [])}
    nat_map = {int(a): int(b) for a, b in side.get("float_map", [])}
    if info["constants"].get("float_dynamic_addressing"):
        # Dynamic addressing binds the whole register file; only check that
        # statically read registers resolve to themselves on both sides.
        bad = [r for r in sdk_map if nat_map.get(r, r) != r]
    else:
        bad = [r for r in set(sdk_map) | set(nat_map) if sdk_map.get(r) != nat_map.get(r)]
    if bad:
        r = sorted(bad)[0]
        fail("reflect_float_map", "%d registers differ, first c%d: sdk %s native %s" %
             (len(bad), r, sdk_map.get(r), nat_map.get(r)))
    sdk_tf = {int(t[0]) for t in info.get("spirv_textures", [])}
    nat_tf = {int(t[2]) for t in side.get("textures", [])}
    if sdk_tf != nat_tf:
        fail("reflect_texture_fetch", "sdk %s native %s" % (sorted(sdk_tf), sorted(nat_tf)))


def linkage(info, text, typ, fail):
    mask = int(info["interpolator_mask"])
    if typ == "ps":
        regs = {int(n) for n in re.findall(r"\[\[user\(texcoord(\d+)\)\]\]", text)}
        extra = [r for r in regs if not (mask >> r) & 1]
        if extra:
            fail("interp_linkage", "native PS reads texcoord%s outside mask %x" % (sorted(extra), mask))
    else:
        regs = {int(n) for n in re.findall(r"\[\[user\(texcoord(\d+)\)\]\]", text)}
        missing = [r for r in range(16) if (mask >> r) & 1 and r not in regs]
        if missing:
            fail("interp_linkage", "native VS does not write texcoord%s from mask %x" % (missing, mask))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucode", help="directory of Xenos ucode")
    ap.add_argument("--msl", action="append", default=[], help="directory of .metal files to compile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bin", default=shaderdiff.DEFAULT_BIN)
    ap.add_argument("--msl-xlate", default=os.path.expanduser(
        "~/dev/rexglue-sdk-metal/out/mac-arm64/rexgpu-metal-msl-xlate"))
    ap.add_argument("--no-patch-vfetch", action="store_true")
    ap.add_argument("--allow", default="spirv_val_needs_scalar_block_layout,native_fallback",
                    help="comma list of classes that do not fail the run")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    args = ap.parse_args()
    args.shadertool = os.path.join(args.bin, "rexgpu-shadertool")
    if args.msl_xlate and not os.path.exists(args.msl_xlate):
        print("note: %s missing, native checks skipped" % args.msl_xlate)
        args.msl_xlate = None
    os.makedirs(args.out, exist_ok=True)
    global AIR_DIR
    AIR_DIR = os.path.join(args.out, "air")
    os.makedirs(AIR_DIR, exist_ok=True)

    results = []
    with cf.ThreadPoolExecutor(args.j) as ex:
        futs = []
        if args.ucode:
            files = sorted(f for f in os.listdir(args.ucode)
                           if ".ucode.bin" in f and not f.endswith((".json", ".metal")) and args.filter in f)
            if args.limit:
                files = files[:args.limit]
            paths = [os.path.join(args.ucode, f) for f in files]
            futs += [ex.submit(check_shader, args, p) for p in paths]
        for d in args.msl:
            files = sorted(f for f in os.listdir(d) if f.endswith(".metal") and args.filter in f)
            if args.limit:
                files = files[:args.limit]
            for f in files:
                def one(p=os.path.join(d, f)):
                    e = metal_compile(p)
                    return {"shader": p, "type": "msl", "fail": {"msl_compile": e} if e else {}}
                futs.append(ex.submit(one))
        for f in cf.as_completed(futs):
            results.append(f.result())

    allow = set(filter(None, args.allow.split(",")))
    by = {}
    for r in results:
        for c in r["fail"]:
            by.setdefault(c, []).append(r["shader"])
    with open(os.path.join(args.out, "static.jsonl"), "w") as f:
        for r in sorted(results, key=lambda r: r["shader"]):
            f.write(json.dumps(r) + "\n")
    lines = ["# static shader checks", "", "%d inputs" % len(results), "",
             "| class | count | allowed | examples |", "|---|---|---|---|"]
    for c, names in sorted(by.items(), key=lambda kv: -len(kv[1])):
        names = sorted(names)
        lines.append("| %s | %d | %s | %s |" % (c, len(names), "yes" if c in allow else "no",
                                                ", ".join(os.path.basename(n) for n in names[:6])))
    lines += ["", "## first message per class", ""]
    for c, names in sorted(by.items()):
        r = next(r for r in results if r["shader"] == sorted(names)[0])
        lines.append("- %s (%s): %s" % (c, os.path.basename(r["shader"]), r["fail"][c]))
    open(os.path.join(args.out, "static.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + len(by)]))
    bad = [c for c in by if c not in allow]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
