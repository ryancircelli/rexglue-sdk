#!/usr/bin/env python3
"""Golden store shader layers (shader_vs, shader_ps): Metal adapter and a
CPU-interpreter seeder.

  golden_shader.py adapt --batch LIST STORE OUTROOT
  golden_shader.py adapt CASE_JSON STORE OUTDIR
      Runs each case through the native Metal translator (rexgpu-metal-msl-xlate)
      and writes the outputs named in the manifest. Translations and metallibs
      are cached per (ucode, interpolator mask, translator binary) in
      GOLDEN_CACHE (default ~/Games/Xbox360/bo2-build/gputest/golden-cache).

  golden_shader.py seed --ucode DIR --store STORE [--filter S] [--limit N] [--seeds K] [-j N]
      Writes cases for shaders the CPU interpreter can run (no texture fetch
      results, no kill, no memexport) with referee "interp". These cases are
      "single" until refxenos adds its entry; they exercise the pipeline and
      give an information-only mismatch rate.

Case encoding (research/golden-store.md, layers shader_vs / shader_ps):
  inputs.ucode     little-endian dwords (VS: vertex fetch formats patched)
  inputs.consts    f32 c[256][4], u32 bool[8], u32 loop[32], u32 fetch[192]
  inputs.vertices  VS: u32 address, u32 size, big-endian blob
  inputs.interp    PS: f32 r[16][4] per pixel, 2x2 quad order
  params           count, interp_mask, sys {flags: [names], texture_signs: [8 u32]},
                   out_masks {output: component mask}
  outputs          VS: pos, o<N> (f32 [count, 4]); PS: oC<N> (f32 [count, 4]),
                   depth (f32 [count]), killed (u8 [count])
Components outside out_masks are written as 0 by every producer and adapter.

Environment: GOLDEN_MSL_XLATE (translator), GOLDEN_BIN (shadertool + runner dir).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_store as gs  # noqa: E402
import shaderdiff as sd  # noqa: E402

BIN = os.environ.get("GOLDEN_BIN", sd.DEFAULT_BIN)
SHADERTOOL = os.path.join(BIN, "rexgpu-shadertool")
RUNNER = os.path.join(BIN, "rexgpu-metal-runner")
XLATE = os.path.expanduser(os.environ.get(
    "GOLDEN_MSL_XLATE", "~/dev/rexglue-sdk-metal-nan/out/mac-arm64/Release/rexgpu-metal-msl-xlate"))
CACHE = os.path.expanduser(os.environ.get("GOLDEN_CACHE", "~/Games/Xbox360/bo2-build/gputest/golden-cache"))
HARNESS_VERSION = "1"
QW = 16                      # quads per row in the pixel harness
DEFAULT_FLAGS = ["alpha_always", "primitive_polygonal", "w_not_reciprocal"]


def run(cmd, timeout=600):
    return sd.run(cmd, timeout=timeout)


# ------------------------------------------------------------ pixel harness

def make_ps_golden(src, out_path):
    """Turns the native fragment entry into a plain function and adds an entry
    xe_gold_ps that feeds it per-pixel interpolants from a buffer (exact
    values, no rasterizer interpolation) in 2x2 quad order, and marks pixels
    that survive (kill detection). Returns {"inputs", "outputs", "entry"}."""
    e = sd.parse_entry(src, "fragment")
    if not e:
        return None, "no fragment entry"
    body_params, call_args, entry_params = [], [], []
    stage_type = None
    for p in e["params"]:
        decl = re.sub(r"\[\[.*?\]\]", "", p).strip()
        name = re.findall(r"(\w+)\s*$", decl)[0]
        body_params.append(decl)
        if "[[stage_in]]" in p:
            stage_type = decl.split()[0]
            call_args.append("xe_gin_v")
        elif "[[position]]" in p:
            call_args.append("xe_gfc")
        elif "[[front_facing]]" in p:
            call_args.append("true")
        elif "[[sample_id]]" in p:
            call_args.append("0u")
        else:
            call_args.append(name)
            entry_params.append(p)
    head = "static %s xe_gold_body(%s)" % (e["ret"], ", ".join(body_params))
    body = src[:e["span"][0]] + head + src[e["span"][1]:]
    inmap, assigns = {}, []
    if stage_type:
        for mbr in sd.parse_struct(src, stage_type):
            n, a = mbr["name"], mbr["attr"]
            if "position" in a:
                assigns.append("  xe_gin_v.%s = xe_gfc;" % n)
                continue
            if "front_facing" in a:
                assigns.append("  xe_gin_v.%s = true;" % n)
                continue
            um = re.search(r"user\(texcoord(\d+)\)", a)
            if not um:
                return None, "unsupported fragment input %s %s" % (n, a)
            reg = int(um.group(1))
            cast = {"float4": "", "float3": ".xyz", "float2": ".xy", "float": ".x"}.get(mbr["type"])
            if cast is None:
                return None, "unsupported input type %s" % mbr["type"]
            inmap[n] = reg
            assigns.append("  xe_gin_v.%s = xe_gin[li * 16u + %du]%s;" % (n, reg, cast))
    ep = ["float4 xe_gfc [[position]]"] + entry_params + [
        "const device float4* xe_gin [[buffer(29)]]", "device uint* xe_galive [[buffer(28)]]",
        "constant uint& xe_gcount [[buffer(27)]]"]
    entry = ("\nfragment %s xe_gold_ps(%s) {\n"
             "  uint2 p = uint2(xe_gfc.xy);\n"
             "  uint idx = ((p.y >> 1) * %du + (p.x >> 1)) * 4u + (p.y & 1u) * 2u + (p.x & 1u);\n"
             "  uint li = min(idx, xe_gcount - 1u);\n"
             "%s%s\n"
             "  %s o = xe_gold_body(%s);\n"
             "  if (idx < xe_gcount) { xe_galive[idx] = 1u; }\n"
             "  return o;\n}\n") % (
        e["ret"], ", ".join(ep), QW, ("  %s xe_gin_v;\n" % stage_type) if stage_type else "",
        "\n".join(assigns), e["ret"], ", ".join(call_args))
    vs = ("\nstruct XeGoldV { float4 pos [[position]]; };\n"
          "vertex XeGoldV xe_gold_vs(uint vid [[vertex_id]]) {\n"
          "  XeGoldV o;\n  float2 uv = float2((vid << 1) & 2u, vid & 2u);\n"
          "  o.pos = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.5f, 1.0f);\n  return o;\n}\n")
    with open(out_path, "w") as f:
        f.write(body + entry + vs)
    return {"inputs": inmap, "outputs": sd.parse_struct(src, e["ret"])}, None


def ps_dims(count):
    quads = (count + 3) // 4
    rows = (quads + QW - 1) // QW
    return QW * 2, max(2, rows * 2)


def ps_pixel(idx, width):
    q, s = idx // 4, idx % 4
    return (q % QW) * 2 + (s & 1), (q // QW) * 2 + (s >> 1)


# --------------------------------------------------------- translation cache

def xlate_identity():
    st = os.stat(XLATE)
    return "%s:%d:%d" % (os.path.realpath(XLATE), st.st_size, int(st.st_mtime))


def prepare(ucode, stype, interp_mask, ident):
    """Translates and compiles once per (ucode, mask, translator). Returns a
    dict with paths and metadata, or {"error": ...}."""
    key = hashlib.sha256(ucode + ("%s:%x:%s:%s" % (stype, interp_mask, ident, HARNESS_VERSION)).encode()).hexdigest()
    d = os.path.join(CACHE, key[:2], key)
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            return json.load(f)
    tmp = tempfile.mkdtemp(dir=os.path.join(CACHE), prefix=".tmp-")
    meta = {"dir": d}
    try:
        ext = ".ucode.bin.frag" if stype == "ps" else ".ucode.bin.vert"
        up = os.path.join(tmp, stype + ext)
        with open(up, "wb") as f:
            f.write(ucode)
        pre = os.path.join(tmp, stype)
        run([SHADERTOOL, "translate", up, "--type", stype, "--out", pre, "--interp-mask", "%x" % interp_mask])
        if not os.path.exists(pre + ".json"):
            meta["error"] = "shadertool translate failed"
        else:
            info = json.load(open(pre + ".json"))
            sh = sd.Shader(up, tmp)
            sh.type, sh.info = stype, info
            mod = sd.native_modification(sh)
            msl = pre + ".native.metal"
            rc, out, err = run([XLATE, up, mod, msl])
            if rc == 2:
                meta["error"] = "native_fallback: " + (out.strip().splitlines() or [""])[-1]
            elif rc != 0 or not os.path.exists(msl):
                meta["error"] = "native_translate_fail: " + (out + err).strip()[-200:]
            else:
                side = json.load(open(msl + ".json"))
                text = open(msl).read()
                if stype == "vs":
                    cap, err = sd.make_vs_capture(text, msl + ".cap.metal")
                    lib = msl + ".cap.metal"
                    meta.update(cap=cap)
                else:
                    cap, err = make_ps_golden(text, msl + ".gold.metal")
                    lib = msl + ".gold.metal"
                    meta.update(ps=cap)
                if not cap:
                    meta["error"] = "harness: " + err
                else:
                    err = sd.compile_metallib(lib)
                    if err:
                        meta["error"] = "msl_compile: " + err
                    else:
                        meta.update(metallib=os.path.join(d, os.path.basename(lib) + ".metallib"),
                                    side=side, dxbc_sys_flags=info["dxbc_sys_flags"])
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        try:
            os.rename(tmp, d)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)  # another worker won the race
        with open(meta_p) as f:
            return json.load(f)
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ adapter

def unpack_consts(blob):
    c = [list(struct.unpack_from("<4f", blob, i * 16)) for i in range(256)]
    o = 4096
    return {"c": c, "bool": list(struct.unpack_from("<8I", blob, o)),
            "loop": list(struct.unpack_from("<32I", blob, o + 32)),
            "fetch": list(struct.unpack_from("<192I", blob, o + 160))}


def build_job(m, prep, store, jdir, n):
    stype = "vs" if m["layer"] == "shader_vs" else "ps"
    p = m["params"]
    k = unpack_consts(store.get(m["inputs"]["consts"]))
    sh = sd.Shader("x" + (".frag" if stype == "ps" else ".vert"), jdir)
    sh.type, sh.native_side = stype, prep["side"]
    flags = 0
    for name in p.get("sys", {}).get("flags", DEFAULT_FLAGS):
        flags |= prep["dxbc_sys_flags"].get(name, 0)
    layout = prep["side"]["sys"]
    sysb = bytearray(sd.sys_block(layout, flags, None))
    signs = p.get("sys", {}).get("texture_signs")
    if signs and layout.get("signs") is not None:
        struct.pack_into("<8I", sysb, layout["signs"], *signs)
    stage = "compute" if stype == "vs" else "fragment"
    bufs = sd.native_buffers(sh, k, bytes(sysb), stage)
    count = int(p["count"])
    shared_size = sd.SHARED_SIZE
    vdata = None
    if stype == "vs" and "vertices" in m["inputs"]:
        v = store.get(m["inputs"]["vertices"])
        addr, size = struct.unpack_from("<II", v, 0)
        vdata = (addr, v[8:8 + size])
        shared_size = max(shared_size, addr + size + 4096)
    shared = bytearray(shared_size)
    if vdata:
        shared[vdata[0]:vdata[0] + len(vdata[1])] = vdata[1]
    bufs.insert(0, {"name": "shared", "data": bytes(shared), "patch": []})
    for b in bufs:
        if "data" in b:
            path = os.path.join(jdir, "%d.%s.bin" % (n, b["name"]))
            with open(path, "wb") as f:
                f.write(b.pop("data"))
            b["file"] = path
        elif b.get("file") is None:
            b.pop("file", None)
    if stype == "vs":
        slots = prep["cap"]["slots"]
        bufs.append({"name": "capture", "size": count * slots * 16, "bind": [["compute", 30]]})
        return {"mode": "compute", "source": prep["metallib"], "function": "xe_vs_capture",
                "threads": count, "dump": ["capture"], "buffers": bufs}
    interp = store.get(m["inputs"]["interp"]) if "interp" in m["inputs"] else bytes(count * 256)
    ip = os.path.join(jdir, "%d.interp.bin" % n)
    with open(ip, "wb") as f:
        f.write(interp)
    cp = os.path.join(jdir, "%d.count.bin" % n)
    with open(cp, "wb") as f:
        f.write(struct.pack("<I", count))
    bufs += [{"name": "gin", "file": ip, "bind": [["fragment", 29]]},
             {"name": "alive", "size": max(4, count * 4), "bind": [["fragment", 28]]},
             {"name": "gcount", "file": cp, "bind": [["fragment", 27]]}]
    W, H = ps_dims(count)
    colors = 0
    for o in prep["ps"]["outputs"]:
        mm = re.search(r"color\((\d+)\)", o["attr"])
        if mm:
            colors |= 1 << int(mm.group(1))
    depth = any("depth" in o["attr"] for o in prep["ps"]["outputs"])
    return {"mode": "render", "vertex_source": prep["metallib"], "vertex_function": "xe_gold_vs",
            "fragment_source": prep["metallib"], "fragment_function": "xe_gold_ps", "vertex_count": 3,
            "width": W, "height": H, "colors": colors, "depth": depth, "dump": ["alive"],
            "buffers": bufs, "_colors": colors, "_depth": depth}


def mask4(vals, mask):
    return [v if (mask >> (i % 4)) & 1 else 0.0 for i, v in enumerate(vals)]


def extract(m, prep, job, data):
    """{output name: bytes} from the runner output."""
    p = m["params"]
    count = int(p["count"])
    masks = p.get("out_masks", {})
    res = {}
    if m["layer"] == "shader_vs":
        cap = prep["cap"]
        slots = cap["slots"]
        cols = {}
        for si, lay in enumerate(cap["layout"]):
            attr, name = lay["attr"], lay["name"]
            if "position" in attr:
                key = "pos"
            else:
                mm = re.search(r"texcoord(\d+)", attr)
                key = "o%s" % mm.group(1) if mm else None
            if key:
                cols[key] = si
        for name in m["outputs"]:
            if name not in cols:
                continue
            vals = []
            for n in range(count):
                vals += mask4(struct.unpack_from("<4f", data, (n * slots + cols[name]) * 16), masks.get(name, 15))
            res[name] = struct.pack("<%df" % len(vals), *vals)
        return res
    W, H = ps_dims(count)
    per = W * H * 16
    off = 0
    rts = {}
    for i in range(8):
        if job["_colors"] & (1 << i):
            rts[i] = data[off:off + per]
            off += per
    dbuf = None
    if job["_depth"]:
        dbuf = data[off:off + W * H * 4]
        off += W * H * 4
    alive = struct.unpack_from("<%dI" % count, data, off)
    for name in m["outputs"]:
        mm = re.match(r"oC(\d)$", name)
        vals = []
        if mm and int(mm.group(1)) in rts:
            rt = rts[int(mm.group(1))]
            for idx in range(count):
                x, y = ps_pixel(idx, W)
                v = struct.unpack_from("<4f", rt, (y * W + x) * 16) if alive[idx] else (0.0,) * 4
                vals += mask4(v, masks.get(name, 15))
            res[name] = struct.pack("<%df" % len(vals), *vals)
        elif name == "depth" and dbuf is not None:
            for idx in range(count):
                x, y = ps_pixel(idx, W)
                vals.append(struct.unpack_from("<f", dbuf, (y * W + x) * 4)[0] if alive[idx] else 0.0)
            res[name] = struct.pack("<%df" % count, *vals)
        elif name == "killed":
            res[name] = bytes(0 if a else 1 for a in alive)
    return res


def adapt(cases, store_root, outroot):
    store = gs.Store(store_root)
    os.makedirs(CACHE, exist_ok=True)
    ident = xlate_identity()
    jdir = tempfile.mkdtemp(prefix="golden-shader-", dir=outroot)
    jobs, metas = [], []

    def err(cid, msg):
        d = os.path.join(outroot, cid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ERROR"), "w") as f:
            f.write(msg.replace("\n", " ")[:400] + "\n")

    for cp in cases:
        with open(cp) as f:
            m = json.load(f)
        if m["layer"] not in ("shader_vs", "shader_ps"):
            err(m["id"], "not a shader case")
            continue
        try:
            stype = "vs" if m["layer"] == "shader_vs" else "ps"
            prep = prepare(store.get(m["inputs"]["ucode"]), stype, int(m["params"].get("interp_mask", 0)), ident)
            if "error" in prep:
                err(m["id"], prep["error"])
                continue
            if m["inputs"].get("textures"):
                err(m["id"], "textured shader cases are not supported by this adapter yet")
                continue
            job = build_job(m, prep, store, jdir, len(jobs))
            job["out"] = os.path.join(jdir, "%d.out" % len(jobs))
            jobs.append(job)
            metas.append((m, prep))
        except Exception as e:  # one bad case must not stop the batch
            err(m["id"], "adapter exception: %r" % e)
    if jobs:
        clean = [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs]
        jp = os.path.join(jdir, "jobs.json")
        with open(jp, "w") as f:
            json.dump(clean, f)
        rc, out, errs = run([RUNNER, jp], timeout=3600)
        ok = {}
        for line in out.splitlines():
            mm = re.match(r"JOB (\d+) (ok|error)\s*(.*)", line)
            if mm:
                ok[int(mm.group(1))] = None if mm.group(2) == "ok" else mm.group(3)
        for i, (m, prep) in enumerate(metas):
            if i not in ok:
                err(m["id"], "runner gave no result (rc %d) %s" % (rc, errs[-200:]))
                continue
            if ok[i] is not None:
                err(m["id"], "runner: " + ok[i])
                continue
            res = extract(m, prep, jobs[i], open(jobs[i]["out"], "rb").read())
            d = os.path.join(outroot, m["id"])
            os.makedirs(d, exist_ok=True)
            for name, b in res.items():
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b)
    shutil.rmtree(jdir, ignore_errors=True)
    return 0


# ------------------------------------------------------------------- seeder

def seed_one(path, store, nseeds, work):
    base = os.path.basename(path)
    name = base.split(".")[0]
    stype = "ps" if (".frag" in base or base.startswith("ps_")) else "vs"
    w = os.path.join(work, name)
    os.makedirs(w, exist_ok=True)
    pre = os.path.join(w, stype)
    args = [SHADERTOOL, "translate", path, "--type", stype, "--out", pre]
    if stype == "vs":
        args.append("--patch-vfetch")
    run(args)
    if not os.path.exists(pre + ".json"):
        return name, "translate failed"
    info = json.load(open(pre + ".json"))
    if info["uses_tfetch_results"] or info["memexport"] or info["kills_pixels"]:
        return name, "skipped (tfetch, memexport or kill: interpreter cannot run it)"
    up = pre + ".ucode.bin.vert" if stype == "vs" and os.path.exists(pre + ".ucode.bin.vert") else path
    ucode = open(up, "rb").read()
    mask = int(info["interpolator_mask"])
    if stype == "vs":
        outs = ["pos"] + ["o%d" % r for r in range(16) if (mask >> r) & 1]
        count = sd.VS_COUNT
    else:
        outs = ["oC%d" % r for r in range(4) if (info["writes_color_targets"] >> r) & 1]
        if info["writes_depth"]:
            outs.append("depth")
        count = 256
    if not outs:
        return name, "skipped (no outputs)"
    n_ok = 0
    for s in range(nseeds):
        rng = random.Random(zlib.crc32(("%s:%d" % (name, s)).encode()))
        k = sd.random_consts(rng, info)
        consts = sd.pack_canonical(k)
        inputs = {"ucode": store.put(ucode), "consts": store.put(consts)}
        if stype == "vs":
            vblob = sd.vertex_blob(rng)
            vin = struct.pack("<II", sd.VB_ADDR, len(vblob)) + vblob
            inputs["vertices"] = store.put(vin)
            ib = vin
        else:
            ib = b"".join(struct.pack("<4f", *[sd.rand_float(rng) for _ in range(4)])
                          for _ in range(count * 16))
            inputs["interp"] = store.put(ib)
        kp, ip, op = pre + ".consts", pre + ".in", pre + ".out"
        open(kp, "wb").write(consts)
        open(ip, "wb").write(ib)
        rc, out, err = run([SHADERTOOL, "interp", up, "--type", stype, "--consts", kp, "--inputs", ip,
                            "--count", str(count), "--out", op])
        if rc != 0:
            return name, "interp failed: " + err.strip()[-200:]
        data = open(op, "rb").read()
        exp = [struct.unpack_from("<256f", data, n * 1024) for n in range(count)]
        results, spec = {}, {}
        for o in outs:
            if o == "pos":
                reg = 62
            elif o == "depth":
                reg = 61
            else:
                reg = int(o[2:] if o.startswith("oC") else o[1:])
            if o == "depth":
                vals = [e[reg * 4] for e in exp]
                spec[o] = {"elem": "f32", "shape": [count], "compare": "bits"}
            else:
                vals = [v for e in exp for v in e[reg * 4:reg * 4 + 4]]
                spec[o] = {"elem": "f32", "shape": [count, 4], "compare": "bits"}
            results[o] = struct.pack("<%df" % len(vals), *vals)
        params = {"count": count, "interp_mask": mask,
                  "sys": {"flags": DEFAULT_FLAGS, "texture_signs": k["signs"]},
                  "out_masks": {o: 15 for o in outs if o != "depth"}}
        store.add_referee("shader_" + stype, inputs, params, spec, "interp",
                          "rexgpu-shadertool interp", results,
                          source="golden_shader seed %s seed %d" % (name, s))
        n_ok += 1
    return name, "ok %d" % n_ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "adapt":
        a = sys.argv[2:]
        if a and a[0] == "--batch":
            lst, store, outroot = a[1:4]
            cases = [l.strip() for l in open(lst) if l.strip()]
            os.makedirs(outroot, exist_ok=True)
            return adapt(cases, store, outroot)
        case, store, outdir = a[:3]
        cid = json.load(open(case))["id"]
        tmp = tempfile.mkdtemp(prefix="golden-one-")
        adapt([case], store, tmp)
        os.makedirs(outdir, exist_ok=True)
        for n in os.listdir(os.path.join(tmp, cid)):
            shutil.copy(os.path.join(tmp, cid, n), os.path.join(outdir, n))
        shutil.rmtree(tmp, ignore_errors=True)
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seed"])
    ap.add_argument("--ucode", required=True)
    ap.add_argument("--store", default="~/Games/Xbox360/bo2-build/gputest/golden")
    ap.add_argument("--filter", default="")
    ap.add_argument("--list", default="", help="file with shader names, one per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    store = gs.Store(args.store)
    files = sorted(os.path.join(args.ucode, f) for f in os.listdir(args.ucode) if ".ucode.bin" in f)
    if args.filter:
        files = [f for f in files if args.filter in f]
    if args.list:
        want = set(l.strip() for l in open(args.list) if l.strip() and not l.startswith("#"))
        files = [f for f in files if os.path.basename(f).split(".")[0] in want]
    if args.limit:
        files = files[:args.limit]
    work = tempfile.mkdtemp(prefix="golden-seed-")
    stats = {}
    with cf.ThreadPoolExecutor(args.j) as ex:
        for name, st in ex.map(lambda f: seed_one(f, store, args.seeds, work), files):
            stats[st.split(":")[0].split(" (")[0]] = stats.get(st.split(":")[0].split(" (")[0], 0) + 1
    shutil.rmtree(work, ignore_errors=True)
    print("seeded %d shaders: %s" % (len(files), ", ".join("%s %d" % kv for kv in sorted(stats.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
