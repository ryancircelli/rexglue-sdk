#!/usr/bin/env python3
"""Per-shader differential tests for Xenos shader translations.

For every Xenos ucode shader in a directory this runs the same randomized
constants and inputs through several "arms" and compares the outputs:

  interp  SDK ShaderInterpreter on the CPU (Xenos semantics; ALU and vertex
          fetch only, no texture fetch results). The oracle when available.
  spirv   SDK SpirvShaderTranslator -> SPIRV-Cross MSL on Metal. This is what
          the SDK Vulkan backend runs through MoltenVK.
  native  The Metal plugin's native MSL translator (rexgpu-metal-msl-xlate
          from the metal-renderer branch), on Metal.

Pixel shaders are rendered over a grid of cells with flat (per-cell constant)
interpolants; vertex shaders are wrapped as compute kernels so every output
(position, interpolators) is captured without rasterization.

For failing shaders it then localizes: each used float constant and (for pixel
shaders) each interpolator register is perturbed alone; a register that changes
the output in one arm but not in the other is reported (misplaced constant
offset, wrong interpolator linkage).

Outputs (under --out): per-shader work files, results.jsonl, summary.md with
failures grouped by class.

usage: shaderdiff.py --ucode DIR --out DIR [--filter SUBSTR] [--limit N]
                     [--arms interp,spirv,native] [--seeds N] [-j N]
"""

import argparse
import concurrent.futures as cf
import copy
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_BIN = os.path.join(SDK, "out", "mac-arm64", "Release")

GRID = 16          # cells per side for pixel shaders
CELL = 4           # pixels per cell side
W = H = GRID * CELL
NCELLS = GRID * GRID
VS_COUNT = 64      # vertex shader invocations per seed
VB_ADDR = 0x00100000   # guest physical address of the random vertex data
VB_DWORDS = 0x4000
SHARED_SIZE = VB_ADDR + VB_DWORDS * 4 + 4096
SENTINEL = -7.0    # clear color: a killed pixel keeps it
HEAP_SLOTS = 64

# ---------------------------------------------------------------- utilities


def run(cmd, timeout=120, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def f2u(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def rand_float(rng):
    r = rng.random()
    if r < 0.05:
        return 0.0
    if r < 0.08:
        return 1.0
    if r < 0.10:
        return -1.0
    if r < 0.12:
        return 0.5
    return rng.uniform(-1.5, 1.5)


def close(a, b, rel=2e-5, abs_=2e-6):
    if a != a and b != b:
        return True
    if a != a or b != b:
        return False
    if a == b:
        return True
    if abs(a) == float("inf") or abs(b) == float("inf"):
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

# ---------------------------------------------------------- MSL parsing


def split_params(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_entry(src, kind):
    m = re.search(r"\b%s\s+(\w+)\s+(\w+)\s*\(" % kind, src)
    if not m:
        return None
    start = m.end()
    depth, i = 1, start
    while depth:
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
        i += 1
    params = split_params(src[start:i - 1])
    return {"ret": m.group(1), "name": m.group(2), "params": params,
            "span": (m.start(), i)}


def parse_struct(src, name):
    m = re.search(r"struct\s+%s\s*\{(.*?)\};" % re.escape(name), src, re.S)
    if not m:
        return []
    members = []
    for line in m.group(1).split(";"):
        line = line.strip()
        mm = re.match(r"(float4|float3|float2|float|uint4|uint|int)\s+(\w+)\s*(\[\[.*?\]\])?\s*(\[(\d+)\])?$", line)
        if mm:
            members.append({"type": mm.group(1), "name": mm.group(2), "attr": mm.group(3) or "",
                            "array": int(mm.group(5)) if mm.group(5) else 0})
    return members


def buffer_params(entry):
    """[(index, type_decl, name)] for [[buffer(N)]] params."""
    res = []
    for p in entry["params"]:
        m = re.search(r"\[\[buffer\((\d+)\)\]\]", p)
        if m:
            decl = re.sub(r"\[\[.*?\]\]", "", p).strip()
            name = re.findall(r"(\w+)\s*$", decl)[0]
            res.append((int(m.group(1)), decl, name))
    return res


def resource_params(entry):
    """{"textures": [(index, decl, name)], "samplers": [...]}."""
    res = {"textures": [], "samplers": []}
    for p in entry["params"]:
        for kind, attr in (("textures", "texture"), ("samplers", "sampler")):
            m = re.search(r"\[\[%s\((\d+)\)\]\]" % attr, p)
            if m:
                decl = re.sub(r"\[\[.*?\]\]", "", p).strip()
                name = re.findall(r"(\w+)\s*$", decl)[0]
                res[kind].append((int(m.group(1)), decl, name))
    return res


# FetchOpDimension: 0 1D, 1 2D (and 1D) as 2D array, 2 3D or stacked, 3 cube.
TEX_TYPES = {"1d": "2d_array", "2d": "2d_array", "3d": "3d", "cube": "cube"}
NATIVE_DIM = {0: "2d", 1: "2d", 2: "3d", 3: "cube"}


TEX_PATTERN = "gradient"
MASK_KNOWN = set()
TEX_LINEAR = True


def texture_set(rng):
    """One texture per (fetch, dimension, signed), shared by both arms so a
    divergence is the translator, not the data. Gradient pattern so wrong
    coordinates, LOD, faces or bindings show up."""
    texs = {}
    for fetch in range(32):
        for dim in ("2d", "3d", "cube"):
            for signed in (0, 1):
                if signed:
                    fill = [rng.uniform(-1, 1) for _ in range(4)]
                    fmt = "rgba8snorm"
                else:
                    fill = [rng.uniform(0.2, 1) for _ in range(4)]
                    fmt = "rgba16f"
                size = [32, 32, 8 if dim == "3d" else 1]
                texs[(fetch, dim, signed)] = {"type": TEX_TYPES[dim], "format": fmt, "size": size, "mips": 6,
                                              "fill": fill}
                if TEX_PATTERN == "gradient":
                    texs[(fetch, dim, signed)]["pattern"] = "gradient"
    return texs


def add_textures(job, sh, arm, texs):
    """Binds the textures/sampler each arm's MSL declares."""
    used = {}
    if arm == "native":
        side = sh.native_side
        vh = next(b for b in job["buffers"] if b["name"] == "viewheap")
        sm = next(b for b in job["buffers"] if b["name"] == "smpheap")
        vh.setdefault("patch", [])
        sm.setdefault("patch", [])
        for slot, dim, fetch, signed in side.get("textures", []):
            key = (fetch, NATIVE_DIM.get(dim, "2d"), 1 if signed else 0)
            name = "tex_%d_%s_%d" % key
            used[name] = key
            vh["patch"].append([slot * 24 + 8, "texture", name])
        for smp in side.get("samplers", []):
            sm["patch"].append([smp[0] * 24, "sampler", "smp"])
        binds = {}
    else:
        rp = (sh.spirv_ps if sh.type == "ps" else sh.spirv_cap).get("resources", {"textures": [], "samplers": []})
        binds = {}
        stage = "fragment" if sh.type == "ps" else "compute"
        for idx, decl, name in rp["textures"]:
            m = re.match(r"xe_texture(\d+)_(\w+?)_([us])$", name)
            if not m:
                return "unknown spirv texture %s" % name
            key = (int(m.group(1)), m.group(2), 1 if m.group(3) == "s" else 0)
            tname = "tex_%d_%s_%d" % key
            used[tname] = key
            binds.setdefault(tname, []).append([stage, idx])
        if rp["samplers"]:
            job.setdefault("samplers", []).append(
                {"name": "smp", "linear": TEX_LINEAR, "bind": [[stage, i] for i, _, _ in rp["samplers"]]})
    if arm == "native":
        job.setdefault("samplers", []).append({"name": "smp", "linear": TEX_LINEAR, "bind": []})
    for tname, key in used.items():
        t = dict(texs[key])
        t["name"] = tname
        t["bind"] = binds.get(tname, [])
        job.setdefault("textures", []).append(t)
    return None


def make_vs_capture(src, out_path):
    """Turn a vertex function into a plain function plus a kernel that writes
    every output member as float4 slots. Returns the slot layout."""
    e = parse_entry(src, "vertex")
    if not e:
        return None, "no vertex entry"
    members = parse_struct(src, e["ret"])
    call_args, kparams, fparams = [], [], []
    for p in e["params"]:
        decl = re.sub(r"\[\[.*?\]\]", "", p).strip()
        name = re.findall(r"(\w+)\s*$", decl)[0]
        if "[[vertex_id]]" in p:
            call_args.append("xe_cap_tid")
        elif "[[base_vertex]]" in p or "[[instance_id]]" in p or "[[base_instance]]" in p:
            call_args.append("0u")
        elif "[[stage_in]]" in p:
            return None, "stage_in vertex input"
        else:
            call_args.append(name)
            kparams.append(p)
        fparams.append(decl)
    head = "static %s %s(%s)" % (e["ret"], e["name"], ", ".join(fparams))
    body = src[:e["span"][0]] + head + src[e["span"][1]:]
    layout, lines = [], []
    for mbr in members:
        n = mbr["array"] or 1
        for k in range(n):
            ref = "o.%s%s" % (mbr["name"], "[%d]" % k if mbr["array"] else "")
            t = mbr["type"]
            if t == "float4":
                v = ref
            elif t in ("float3", "float2"):
                v = "float4(%s, %s)" % (ref, "0.0f" if t == "float3" else "0.0f, 0.0f")
            elif t == "float":
                v = "float4(%s, 0.0f, 0.0f, 0.0f)" % ref
            else:
                v = "as_type<float4>(uint4(%s))" % ref if t == "uint4" else \
                    "float4(as_type<float>(uint(%s)), 0.0f, 0.0f, 0.0f)" % ref
            lines.append("  xe_cap_out[xe_cap_tid * {S}U + %dU] = %s;" % (len(layout), v))
            layout.append({"name": mbr["name"], "elem": k, "attr": mbr["attr"]})
    slots = max(1, len(layout))
    kp = ", ".join(["uint xe_cap_tid [[thread_position_in_grid]]"] + kparams +
                   ["device float4* xe_cap_out [[buffer(30)]]"])
    kernel = "\nkernel void xe_vs_capture(%s) {\n  %s o = %s(%s);\n%s\n}\n" % (
        kp, e["ret"], e["name"], ", ".join(call_args),
        "\n".join(l.replace("{S}", str(slots)) for l in lines))
    with open(out_path, "w") as f:
        f.write(body + kernel)
    return {"slots": slots, "layout": layout, "buffers": buffer_params(e)}, None


def make_ps_harness(src, out_path, interp_mask=0xFFFF):
    """Writes a pass-through vertex function feeding the fragment function's
    stage_in members from a vertex buffer of [pos, r0..r15] float4s. Returns
    the input map {member: register}.

    Inputs are mapped by the cross-stage linkage contract, not by member
    name, so a fragment shader that reads the wrong slot is caught:
    user(texcoordN) carries register N (native ABI), user(locnK) carries the
    K-th set bit of the interpolator mask (SDK SPIR-V compacted locations)."""
    mask_regs = [i for i in range(16) if (interp_mask >> i) & 1]
    e = parse_entry(src, "fragment")
    if not e:
        return None, "no fragment entry"
    stage_type = None
    for p in e["params"]:
        if "[[stage_in]]" in p:
            stage_type = re.sub(r"\[\[.*?\]\]", "", p).split()[0]
    members = parse_struct(src, stage_type) if stage_type else []
    inmap, outm, assigns = {}, [], []
    for mbr in members:
        if "position" in mbr["attr"]:
            continue
        um = re.search(r"user\((\w+)\)", mbr["attr"])
        if not um:
            if "front_facing" in mbr["attr"] or "sample" in mbr["attr"]:
                continue
            return None, "unsupported fragment input %s %s" % (mbr["name"], mbr["attr"])
        sem = um.group(1)
        tm = re.match(r"texcoord(\d+)$", sem)
        lm = re.match(r"locn(\d+)$", sem)
        if tm:
            reg = int(tm.group(1))
        elif lm and int(lm.group(1)) < len(mask_regs):
            reg = mask_regs[int(lm.group(1))]
        else:
            rm = re.search(r"(\d+)$", mbr["name"])
            if not rm:
                return None, "cannot map input %s to a register" % mbr["name"]
            reg = int(rm.group(1))
        inmap[mbr["name"]] = reg
        interp = ""
        if "flat" in mbr["attr"]:
            interp = ", flat"
        outm.append("  %s %s [[user(%s)%s]];" % (mbr["type"], mbr["name"], um.group(1), interp))
        cast = {"float4": "", "float3": ".xyz", "float2": ".xy", "float": ".x"}.get(mbr["type"], None)
        if cast is None:
            return None, "unsupported input type %s" % mbr["type"]
        assigns.append("  o.%s = xe_vb[vid * 17u + %du]%s;" % (mbr["name"], 1 + reg, cast))
    hs = "#include <metal_stdlib>\nusing namespace metal;\nstruct XeHarnessOut {\n  float4 xe_pos [[position]];\n%s\n};\n" \
         "vertex XeHarnessOut xe_harness_vs(uint vid [[vertex_id]], const device float4* xe_vb [[buffer(0)]]) {\n" \
         "  XeHarnessOut o;\n  o.xe_pos = xe_vb[vid * 17u];\n%s\n  return o;\n}\n" % ("\n".join(outm), "\n".join(assigns))
    with open(out_path, "w") as f:
        f.write(hs)
    outs = parse_struct(src, e["ret"])
    return {"inputs": inmap, "outputs": outs, "buffers": buffer_params(e), "resources": resource_params(e),
            "entry": e["name"]}, None

# ------------------------------------------------------- constant blocks


def random_consts(rng, info):
    c = [[rand_float(rng) for _ in range(4)] for _ in range(256)]
    bools = [rng.getrandbits(32) for _ in range(8)]
    # Loop constants: count 0..4, start 0..7, step -1..2 (keeps aL in range).
    loops = [(rng.randrange(5)) | (rng.randrange(8) << 8) | (((rng.randrange(4) - 1) & 0xFF) << 16)
             for _ in range(32)]
    fetch = [0] * 192
    for t in range(32):
        # Texture fetch constant: type 2, 32x32 2D, random swizzle/sign/exp bias.
        w = [rng.getrandbits(32) for _ in range(6)]
        w[0] = (w[0] & ~3) | 2
        w[2] = 31 | (31 << 13)
        # Identity swizzle (XYZW, bits 1..12 of dword 3): the SDK SPIR-V
        # translator expects the host image view to apply the swizzle
        # (imageViewFormatSwizzle), so a random one would differ by ABI, not
        # by translator bug.
        w[3] = (w[3] & ~(0xFFF << 1)) | (0x688 << 1)
        w[5] = (w[5] & ~(3 << 9)) | (1 << 9)
        if os.environ.get("SHADERDIFF_SIMPLE_FETCH") in ("1", "words"):
            keep = os.environ.get("SHADERDIFF_FETCH_KEEP", "")
            simple = [2, 0, 31 | (31 << 13), 0x688 << 1, 0, 1 << 9]
            for i in range(6):
                if str(i) not in keep:
                    w[i] = simple[i]
        for spec in filter(None, os.environ.get("SHADERDIFF_FETCH_AND", "").split(",")):
            i, m = spec.split(":")
            w[int(i)] &= int(m, 16)
        if "sdk_exp_bias_word" in MASK_KNOWN:
            # Known SDK SPIR-V bug: result exponent bias read from dword 4
            # bits 13:18 instead of dword 3 (exp_adjust). Zero both fields so
            # every textured shader does not fail on this one bug.
            w[3] &= ~(0x3F << 13)
            w[4] &= ~(0x3F << 13)
        fetch[t * 6:t * 6 + 6] = w
    for b in info.get("vertex_bindings", []):
        fc = b["fetch_constant"]
        # Vertex fetch constant: type 3, address in dwords, endian 8in32,
        # size in dwords.
        fetch[fc * 2] = 3 | ((VB_ADDR >> 2) << 2)
        fetch[fc * 2 + 1] = 2 | (VB_DWORDS << 2)
    # Texture signs: 2 bits per component per fetch constant (0 unsigned,
    # 1 signed, 2 biased, 3 gamma); same bytes go into both ABIs.
    signs = [rng.getrandbits(32) for _ in range(8)]
    if os.environ.get("SHADERDIFF_SIMPLE_FETCH") in ("1", "signs"):
        signs = [0] * 8
    return {"c": c, "bool": bools, "loop": loops, "fetch": fetch, "signs": signs,
            "textures": texture_set(rng)}


def pack_canonical(k):
    b = b"".join(struct.pack("<4f", *v) for v in k["c"])
    b += struct.pack("<8I", *k["bool"]) + struct.pack("<32I", *k["loop"])
    b += struct.pack("<192I", *k["fetch"])
    return b


def vertex_blob(rng):
    """Random big-endian (8in32) dwords: floats in a sane range."""
    words = [f2u(rand_float(rng)) for _ in range(VB_DWORDS)]
    return struct.pack(">%dI" % VB_DWORDS, *words)


def sys_block(layout, flags, endian_fields):
    size = layout["size"]
    b = bytearray(size)

    def put(field, fmt, *vals):
        if field in layout:
            struct.pack_into(fmt, b, layout[field], *vals)
    put("flags", "<I", flags)
    put("ndc_scale", "<3f", 1.0, 1.0, 1.0)
    put("ndc_offset", "<3f", 0.0, 0.0, 0.0)
    put("color_exp_bias", "<4f", 1.0, 1.0, 1.0, 1.0)
    put("vertex_index_min", "<I", 0)
    put("vertex_index_max", "<I", 0xFFFFFF)
    put("point_vertex_diameter_max", "<f", 256.0)
    return bytes(b)

# ------------------------------------------------------------ per shader


class Shader:
    def __init__(self, path, work):
        self.path = path
        base = os.path.basename(path)
        self.name = base.split(".")[0]
        self.type = "ps" if (".frag" in base or base.startswith("ps_")) else "vs"
        self.work = os.path.join(work, self.name)
        self.result = {"shader": self.name, "type": self.type, "classes": [], "notes": []}
        self.info = None
        self.arms = {}

    def fail(self, cls, note=None):
        self.result["classes"].append(cls)
        if note:
            self.result["notes"].append("%s: %s" % (cls, note[:400]))


def prepare(sh, tools, args):
    """Translation + static checks (runs in a worker thread)."""
    os.makedirs(sh.work, exist_ok=True)
    pre = os.path.join(sh.work, sh.type)
    patch = sh.type == "vs" and not args.no_patch_vfetch
    rc, out, err = run([tools["shadertool"], "translate", sh.path, "--type", sh.type, "--out", pre]
                       + (["--interp-mask", args.interp_mask] if args.interp_mask else [])
                       + (["--patch-vfetch"] if patch else []))
    sh.ucode = pre + ".ucode.bin.vert" if patch and os.path.exists(pre + ".ucode.bin.vert") else sh.path
    if not os.path.exists(pre + ".json"):
        sh.fail("shadertool_crash", err.strip()[-300:])
        return sh
    sh.info = json.load(open(pre + ".json"))
    sh.result["uses_tfetch"] = sh.info["uses_tfetch_results"]
    sh.result["memexport"] = sh.info["memexport"]
    if not sh.info["ok"]:
        sh.fail("spirv_translate_fail", "; ".join(sh.info["errors"]))
    else:
        # Static: spirv-val.
        rc, out, err = run([tools["spirv-val"], "--target-env", "vulkan1.2", pre + ".spv"])
        if rc != 0:
            msg = (out + err).strip().splitlines()[0] if (out + err).strip() else ""
            rc2, _, _ = run([tools["spirv-val"], "--target-env", "vulkan1.2", "--scalar-block-layout", pre + ".spv"])
            sh.fail("spirv_val_needs_scalar_block_layout" if rc2 == 0 else "spirv_val", msg)
        rc, out, err = run([tools["spirv-cross"], pre + ".spv", "--msl", "--msl-version", "20300",
                            "--output", pre + ".spirv.metal"])
        if rc != 0:
            sh.fail("spirv_cross_fail", err.strip()[-300:])
        else:
            sh.arms["spirv"] = pre + ".spirv.metal"
    if tools.get("msl-xlate") and "native" in args.arms:
        rc, out, err = run([tools["msl-xlate"], sh.ucode,
                            native_modification(sh), pre + ".native.metal"])
        if rc == 2:
            sh.fail("native_fallback", out.strip().splitlines()[-1] if out.strip() else "")
        elif rc != 0 or not os.path.exists(pre + ".native.metal"):
            sh.fail("native_translate_fail", (out + err).strip()[-300:])
        else:
            sh.arms["native"] = pre + ".native.metal"
            if args.inject:
                inject_bug(pre + ".native.metal", args.inject)
            sh.native_side = json.load(open(pre + ".native.metal.json"))
    # Static: offline MSL compile of each arm (catches compile errors with
    # the real compiler and file/line, independent of the runtime path).
    # The metallibs are reused by the runner (compiling in parallel here is
    # much faster than serial runtime compiles in one process).
    for arm, src in list(sh.arms.items()):
        err = compile_metallib(src)
        if err:
            sh.fail("%s_msl_compile" % arm, err)
            del sh.arms[arm]
    # Static: interpolator linkage / bindings reflection for each arm.
    for arm, src in list(sh.arms.items()):
        text = open(src).read()
        if sh.type == "vs":
            cap, err = make_vs_capture(text, src + ".cap.metal")
            if not cap:
                sh.fail("%s_harness" % arm, err)
                del sh.arms[arm]
                continue
            setattr(sh, arm + "_cap", cap)
            err = compile_metallib(src + ".cap.metal")
            if err:
                sh.fail("%s_harness" % arm, "capture kernel: " + err)
                del sh.arms[arm]
        else:
            hv, err = make_ps_harness(text, src + ".harness.metal", int(sh.info["interpolator_mask"]))
            if not hv:
                sh.fail("%s_harness" % arm, err)
                del sh.arms[arm]
                continue
            setattr(sh, arm + "_ps", hv)
            err = compile_metallib(src + ".harness.metal")
            if err:
                sh.fail("%s_harness" % arm, "harness vertex: " + err)
                del sh.arms[arm]
                continue
            regs = sorted(hv["inputs"].values())
            want = [i for i in range(16) if (int(sh.info["interpolator_mask"]) >> i) & 1]
            if regs and regs != want[:len(regs)] and set(regs) - set(want):
                sh.fail("%s_interp_linkage" % arm, "reads registers %s, mask %s" % (regs, want))
    if "interp" in args.arms and not sh.info["uses_tfetch_results"]:
        sh.arms["interp"] = True
    return sh

def native_modification(sh):
    """The Metal plugin extends DxbcShaderTranslator::Modification: pixel
    depth_stencil_mode grows to 3 bits (46..48) and adds
    rt0_blend_rgb_factor_for_premult (49..53) and rt0_blend_a_factor_for_premult
    (54..58). Their zero value is BlendFactor::kZero, which zeroes the color
    output, so set both to kOne (1) like the plugin's pipeline cache does for
    non-MIN/MAX blending. The upstream layout is otherwise identical."""
    m = int(sh.info["dxbc_modification"], 16)
    if sh.type == "ps":
        m |= (1 << 49) | (1 << 54)
    return "%016x" % m


def compile_metallib(src):
    """xcrun metal source -> src.metallib with safe math (matches the
    runtime MTLMathModeSafe). Returns an error line or None."""
    rc, out, err = run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-fno-fast-math",
                        "-Wno-unused-variable", src, "-o", src + ".metallib"], timeout=600)
    if rc != 0:
        first = [l for l in err.splitlines() if "error" in l][:1]
        return first[0] if first else err[-300:]
    return None


def inject_bug(path, kind):
    """Negative controls: plant a known translator bug in the native MSL so
    the harness can prove it catches that class."""
    src = open(path).read()
    if kind == "const_offset":
        # Misplaced constant offset: first constant read lands one float late
        # (the float2 packing class of bug).
        src, n = re.subn(r"xe_float\[(\w+)\]", r"((constant float4*)((constant float*)xe_float + 1))[\1]", src, count=1)
    elif kind == "interp_swap":
        # Wrong interpolator linkage: swap the first two fragment inputs.
        regs = re.findall(r"\[\[user\((texcoord\d+)\)", src)
        n = 0
        if "fragment" in src and len(regs) >= 2:
            a, b = regs[0], regs[1]
            src = src.replace("user(%s)" % a, "user(XE_TMP)").replace("user(%s)" % b, "user(%s)" % a).replace("user(XE_TMP)", "user(%s)" % b)
            n = 1
    elif kind == "precise_math":
        # Probe for NaN classes: swap fast:: math for precise:: in the native MSL.
        src, n = re.subn(r"\bfast::", "precise::", src)
    elif kind == "fix_maxa":
        # Positive control for the native scalar maxas/maxasf clamp: apply
        # the -256 lower bound the SDK uses and check the mismatch goes away.
        src, n = re.subn(r"(a0 = int\(fast::min\(fast::max\(fast::floor\([^;]*\)), 0\.0f\)", r"\1, -256.0f)", src)
    else:
        raise SystemExit("unknown --inject %s" % kind)
    if n:
        open(path, "w").write(src)


# ---------------------------------------------------- ABI specific jobs


def native_buffers(sh, k, sysb, stage):
    """XeTop-based argument buffers for the native MSL ABI."""
    side = sh.native_side
    fl = bytearray(max(1, side["float_count"]) * 16)
    for reg, packed in side["float_map"]:
        struct.pack_into("<4f", fl, packed * 16, *k["c"][reg])
    bl = struct.pack("<8I", *k["bool"]) + struct.pack("<32I", *k["loop"])
    fe = struct.pack("<192I", *k["fetch"])
    desc = struct.pack("<%dI" % HEAP_SLOTS, *range(HEAP_SLOTS))
    sysd = bytearray(side["sys"]["size"])
    sysd[:] = sysb
    files = {"float": bytes(fl), "boolloop": bl, "fetch": fe, "desc": desc, "sys": bytes(sysd)}
    pre = "vs" if sh.type == "vs" else "ps"
    top_patch = [[10 * 8, "buffer", "sys"], [12 * 8, "buffer", "boolloop"],
                 [0, "buffer", "srv"], [5 * 8, "buffer", "srv"],
                 [8, "buffer", "viewheap"], [16, "buffer", "viewheap"], [24, "buffer", "viewheap"],
                 [9 * 8, "buffer", "smpheap"]]
    if pre == "vs":
        top_patch += [[11 * 8, "buffer", "float"], [13 * 8, "buffer", "fetch"], [14 * 8, "buffer", "desc"]]
    else:
        top_patch += [[21 * 8, "buffer", "float"], [22 * 8, "buffer", "fetch"], [23 * 8, "buffer", "desc"]]
    srv = struct.pack("<QQQ", 0, 0, SHARED_SIZE)
    bufs = [{"name": "top", "size": 32 * 8, "bind": [[stage, 2]], "patch": top_patch},
            {"name": "srv", "file": None, "data": srv, "patch": [[0, "buffer", "shared"]]},
            {"name": "viewheap", "size": HEAP_SLOTS * 24}, {"name": "smpheap", "size": HEAP_SLOTS * 24}]
    for n, d in files.items():
        bufs.append({"name": n, "data": d})
    return bufs


SPIRV_NAMES = {
    "system": re.compile(r"system", re.I),
    "float": re.compile(r"float", re.I),
    "boolloop": re.compile(r"bool|loop", re.I),
    "fetch": re.compile(r"fetch", re.I),
    "shared": re.compile(r"shared", re.I),
}


def spirv_buffers(sh, k, sysb, params, stage):
    """Buffers for the SPIRV-Cross MSL of the SDK SPIR-V, bound by the
    [[buffer(N)]] parameters the entry point declares."""
    cm = sh.info["constants"]
    fl = bytearray(max(1, cm["float_count"]) * 16)
    for reg, packed in cm["float_map"]:
        struct.pack_into("<4f", fl, packed * 16, *k["c"][reg])
    data = {
        "system": sysb,
        "float": bytes(fl),
        "boolloop": struct.pack("<8I", *k["bool"]) + struct.pack("<32I", *k["loop"]),
        "fetch": struct.pack("<192I", *k["fetch"]),
    }
    bufs, unknown = [], []
    for idx, decl, name in params:
        kind = None
        for kname, rx in SPIRV_NAMES.items():
            if rx.search(decl):
                kind = kname
                break
        if kind == "shared":
            bufs.append({"alias": "shared", "bind": [[stage, idx]]})
        elif kind:
            bufs.append({"name": "%s_%d" % (kind, idx), "data": data[kind], "bind": [[stage, idx]]})
        else:
            unknown.append(decl)
    return bufs, unknown


class JobBuilder:
    def __init__(self, dirpath):
        self.dir = dirpath
        self.jobs = []
        self.meta = []
        self.nfiles = 0

    def blob(self, data):
        self.nfiles += 1
        p = os.path.join(self.dir, "b%05d.bin" % self.nfiles)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def add(self, job, meta):
        for b in job.get("buffers", []):
            if "data" in b:
                d = b.pop("data")
                b["file"] = self.blob(d)
            b.pop("file", None) if b.get("file") is None else None
        job["out"] = os.path.join(self.dir, "o%05d.bin" % len(self.jobs))
        self.jobs.append(job)
        self.meta.append(meta)


def gpu_job(sh, arm, k, vb_cells, vblob, flags):
    """Builds one runner job for `arm` with constants `k`."""
    if arm == "native":
        layout = sh.native_side["sys"]
        f = flags["dxbc"]
    else:
        layout = sh.info["spirv_system_layout"]
        f = flags["spirv"]
    sysb = bytearray(sys_block(layout, f, None))
    soff = layout.get("texture_swizzled_signs", layout.get("signs"))
    if soff is not None and "signs" in k:
        struct.pack_into("<8I", sysb, soff, *k["signs"])
    sysb = bytes(sysb)
    shared = {"name": "shared", "size": SHARED_SIZE, "patch": []}
    stage = "compute" if sh.type == "vs" else "fragment"
    if arm == "native":
        bufs = native_buffers(sh, k, sysb, stage)
    else:
        params = (sh.spirv_cap if sh.type == "vs" else sh.spirv_ps)["buffers"]
        bufs, unknown = spirv_buffers(sh, k, sysb, params, stage)
        if unknown:
            return None, "unknown spirv buffer params %s" % unknown
    # The shared memory buffer carries the random vertex data.
    shared_bind = []
    out_bufs = []
    for b in bufs:
        if b.get("alias") == "shared":
            shared_bind += b["bind"]
        else:
            out_bufs.append(b)
    shared_data = bytearray(SHARED_SIZE)
    shared_data[VB_ADDR:VB_ADDR + len(vblob)] = vblob
    shared["data"] = bytes(shared_data)
    shared["bind"] = shared_bind
    out_bufs.insert(0, shared)
    if sh.type == "vs":
        cap = getattr(sh, arm + "_cap")
        out_bufs.append({"name": "capture", "size": VS_COUNT * cap["slots"] * 16, "bind": [["compute", 30]]})
        job = {"mode": "compute", "source": sh.arms[arm] + ".cap.metal.metallib", "function": "xe_vs_capture",
               "threads": VS_COUNT, "dump": ["capture"], "buffers": out_bufs}
    else:
        ps = getattr(sh, arm + "_ps")
        out_bufs.append({"name": "vb", "data": vb_cells, "bind": [["vertex", 0]]})
        colors = 0
        for o in ps["outputs"]:
            m = re.search(r"color\((\d+)\)", o["attr"])
            if m:
                colors |= 1 << int(m.group(1))
        depth = any("depth" in o["attr"] for o in ps["outputs"])
        job = {"mode": "render", "vertex_source": sh.arms[arm] + ".harness.metal.metallib",
               "vertex_function": "xe_harness_vs", "fragment_source": sh.arms[arm] + ".metallib",
               "fragment_function": ps["entry"], "vertex_count": NCELLS * 6, "width": W, "height": H,
               "colors": colors, "depth": depth, "buffers": out_bufs}
    if "textures" in k:
        err = add_textures(job, sh, arm, k["textures"])
        if err:
            return None, err
    return job, None


def cell_inputs(rng, flat_values=None):
    """Per-cell interpolants: [cell][reg][4] plus the vertex buffer."""
    vals = flat_values or [[[rand_float(rng) for _ in range(4)] for _ in range(16)] for _ in range(NCELLS)]
    vb = bytearray()
    for cy in range(GRID):
        for cx in range(GRID):
            cell = vals[cy * GRID + cx]
            for c in (0, 1, 2, 2, 1, 3):
                x = (cx + (c & 1)) / GRID * 2 - 1
                y = 1 - (cy + (c >> 1)) / GRID * 2
                vb += struct.pack("<4f", x, y, 0.5, 1.0)
                for r in range(16):
                    vb += struct.pack("<4f", *cell[r])
    return vals, bytes(vb)


def read_ps_output(path, colors, depth):
    data = open(path, "rb").read()
    per = W * H * 16
    res = {}
    off = 0
    for i in range(8):
        if colors & (1 << i):
            rt = data[off:off + per]
            off += per
            vals = []
            for cy in range(GRID):
                for cx in range(GRID):
                    px = (cy * CELL + CELL // 2) * W + cx * CELL + CELL // 2
                    vals.append(struct.unpack_from("<4f", rt, px * 16))
            res["rt%d" % i] = vals
    if depth:
        d = data[off:off + W * H * 4]
        res["depth"] = [struct.unpack_from("<f", d, ((cy * CELL + CELL // 2) * W + cx * CELL + CELL // 2) * 4)
                        for cy in range(GRID) for cx in range(GRID)]
    return res


def normalize_vs(sh, arm, data):
    """{"pos": [...], "o<reg>": [...]} from a capture dump."""
    cap = getattr(sh, arm + "_cap")
    slots = cap["slots"]
    res = defaultdict(list)
    for n in range(VS_COUNT):
        for si, lay in enumerate(cap["layout"]):
            v = struct.unpack_from("<4f", data, (n * slots + si) * 16)
            attr, name = lay["attr"], lay["name"]
            if "position" in attr:
                key = "pos"
            else:
                m = re.search(r"(\d+)$", name)
                key = ("o%s" % m.group(1)) if m and "user" in attr else name + ("[%d]" % lay["elem"] if lay["elem"] else "")
            res[key].append(v)
    return res


def compare(a, b):
    """Returns list of (key, cell, comp, va, vb) mismatches on common keys."""
    mism = []
    for key in sorted(set(a) & set(b)):
        for i, (va, vb) in enumerate(zip(a[key], b[key])):
            if isinstance(va, float):
                va, vb = (va,), (vb,)
            for c in range(len(va)):
                if not close(va[c], vb[c]):
                    mism.append((key, i, c, va[c], vb[c]))
    return mism


def classify(mism, a, b, keys_only_a, keys_only_b):
    classes = set()
    def rel(va, vb):
        return abs(va - vb) / max(1e-6, abs(va), abs(vb))
    if not (keys_only_a or keys_only_b) and all(
            va == va and vb == vb and SENTINEL not in (va, vb) and rel(va, vb) < 1e-3
            for _, _, _, va, vb in mism):
        # Small relative error only: transcendental precision or a different
        # but legal operation order. Reported, but low severity.
        return {"precision"}
    if keys_only_a or keys_only_b:
        classes.add("output_set_differs")
    for key, i, c, va, vb in mism:
        if va == SENTINEL or vb == SENTINEL:
            classes.add("kill_mismatch")
        elif va != va or vb != vb:
            classes.add("nan_mismatch")
        elif key == "depth":
            classes.add("depth_mismatch")
        elif key == "pos":
            classes.add("position_mismatch")
        elif abs(va + vb) <= 1e-6 * max(1, abs(va)):
            classes.add("sign_mismatch")
        else:
            classes.add("value_mismatch:" + re.sub(r"\d+$", "N", key) if key.startswith(("rt", "o")) else "value_mismatch")
    return classes

# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ucode", required=True, help="dir of *.ucode.bin.vert / *.ucode.bin.frag (little-endian dwords)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="interp,spirv,native")
    ap.add_argument("--reference", default="", help="arm to compare others against (default: interp if available, else spirv)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--interp-mask", default="", help="hex; default VS: written, PS: all registers")
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--msl-xlate", default=os.path.expanduser("~/dev/rexglue-sdk-metal/out/mac-arm64/rexgpu-metal-msl-xlate"))
    ap.add_argument("--inject", default="", help="edit the native MSL: const_offset | interp_swap plant a bug (negative controls); fix_maxa applies the address register clamp fix (positive control); precise_math replaces fast:: with precise::")
    ap.add_argument("--no-patch-vfetch", action="store_true", help="keep vertex fetch formats as dumped (undefined)")
    ap.add_argument("--no-localize", action="store_true")
    ap.add_argument("--batch", type=int, default=40, help="shaders per runner process")
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--tex", choices=["gradient", "flat"], default="gradient",
                    help="texture contents: gradient (catches coordinate/LOD/binding bugs) or flat "
                         "(isolates non-texture ALU differences)")
    ap.add_argument("--tex-filter", choices=["linear", "nearest"], default="linear")
    ap.add_argument("--mask-known", default="",
                    help="comma list of known bugs to keep out of the random inputs so other bugs "
                         "are visible: sdk_exp_bias_word")
    args = ap.parse_args()
    global TEX_PATTERN, TEX_LINEAR, MASK_KNOWN
    MASK_KNOWN = set(filter(None, args.mask_known.split(",")))
    TEX_PATTERN = args.tex
    TEX_LINEAR = args.tex_filter == "linear"
    args.arms = args.arms.split(",")

    tools = {
        "shadertool": os.path.join(args.bin, "rexgpu-shadertool"),
        "runner": os.path.join(args.bin, "rexgpu-metal-runner"),
        "spirv-val": shutil.which("spirv-val"),
        "spirv-cross": shutil.which("spirv-cross"),
        "msl-xlate": args.msl_xlate if ("native" in args.arms and os.path.exists(args.msl_xlate)) else None,
    }
    for t in ("shadertool", "runner", "spirv-val", "spirv-cross"):
        if not tools[t] or not os.path.exists(tools[t]):
            sys.exit("missing tool %s (%s)" % (t, tools[t]))
    if "native" in args.arms and not tools["msl-xlate"]:
        print("note: native arm disabled, %s not found" % args.msl_xlate)
        args.arms.remove("native")

    files = sorted(f for f in os.listdir(args.ucode) if ".ucode.bin" in f and not f.endswith((".json", ".metal")))
    files = [f for f in files if args.filter in f]
    if args.limit:
        files = files[:args.limit]
    work = os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    shaders = [Shader(os.path.join(args.ucode, f), work) for f in files]
    print("preparing %d shaders (-j %d)" % (len(shaders), args.j), flush=True)
    with cf.ThreadPoolExecutor(args.j) as ex:
        def safe_prepare(s):
            try:
                return prepare(s, tools, args)
            except Exception as e:  # keep going; one bad shader must not stop the run
                import traceback
                s.fail("harness_exception", "%s: %s" % (type(e).__name__, traceback.format_exc().splitlines()[-3:]))
                s.arms = {}
                return s
        list(ex.map(safe_prepare, shaders))
    print("prepared in %.0fs" % (time.time() - t0), flush=True)

    # Flag sets (identical semantics in both ABIs).
    def flags_for(sh):
        sf, df = sh.info["spirv_sys_flags"], sh.info["dxbc_sys_flags"]
        return {"spirv": sf["alpha_always"] | sf["primitive_polygonal"] | sf["w_not_reciprocal"],
                "dxbc": df["alpha_always"] | df["primitive_polygonal"] | df["w_not_reciprocal"]}

    results = open(os.path.join(args.out, "results.jsonl"), "w")
    written = set()
    runnable = [s for s in shaders if s.info and len(s.arms) >= 2]
    for s in shaders:
        if s not in runnable:
            if s.info and len(s.arms) < 2 and not s.result["classes"]:
                s.result["notes"].append("fewer than two arms available (%s)" % ",".join(s.arms))
                s.result["skipped"] = True
    for bstart in range(0, len(runnable), args.batch):
        batch = runnable[bstart:bstart + args.batch]
        jdir = os.path.join(args.out, "jobs", "%05d" % bstart)
        shutil.rmtree(jdir, ignore_errors=True)
        os.makedirs(jdir)
        jb = JobBuilder(jdir)
        interp_jobs = []
        for sh in batch:
            fl = flags_for(sh)
            ref = args.reference or ("interp" if "interp" in sh.arms else "spirv")
            if ref not in sh.arms:
                ref = sorted(sh.arms)[0]
            sh.ref = ref
            variants = []
            for seed in range(args.seeds):
                rng = random.Random(zlib.crc32(("%s:%d" % (sh.name, seed)).encode()))
                variants.append(("seed%d" % seed, random_consts(rng, sh.info), cell_inputs(rng), vertex_blob(rng)))
            sh.variants = variants
            # Localization variants: perturb one float constant / one input
            # register at a time (run lazily only for failing shaders; we
            # still schedule them now so everything stays in one runner pass).
            if not args.no_localize:
                base = variants[0]
                regs = [r for r, _ in sh.info["constants"]["float_map"]]
                for r in regs[:64]:
                    k2 = copy.deepcopy(base[1])
                    k2["c"][r] = [v + 0.37 if v != 0 else 0.73 for v in k2["c"][r]]
                    variants.append(("c%d" % r, k2, base[2], base[3]))
                if sh.type == "ps":
                    for r in range(16):
                        vals = [[list(x) for x in cell] for cell in base[2][0]]
                        for cell in vals:
                            cell[r] = [v + 0.37 for v in cell[r]]
                        variants.append(("r%d" % r, base[1], cell_inputs(None, vals), base[3]))
            for vname, k, (cellvals, vbcells), vblob in variants:
                for arm in sh.arms:
                    if arm == "interp":
                        interp_jobs.append((sh, vname, k, cellvals, vblob))
                        continue
                    job, err = gpu_job(sh, arm, k, vbcells, vblob, fl)
                    if not job:
                        sh.fail("%s_abi" % arm, err)
                        continue
                    jb.add(job, (sh, arm, vname))
        with open(os.path.join(jdir, "jobs.json"), "w") as f:
            json.dump(jb.jobs, f)
        outputs = {}
        rc, out, err = run([tools["runner"], os.path.join(jdir, "jobs.json")], timeout=3600)
        for line in out.splitlines():
            m = re.match(r"JOB (\d+) (ok|error)\s*(.*)", line)
            if not m:
                continue
            sh, arm, vname = jb.meta[int(m.group(1))]
            if m.group(2) == "ok":
                outputs[(sh.name, arm, vname)] = jb.jobs[int(m.group(1))]["out"]
            elif vname == "seed0":
                msg = m.group(3)
                cls = "%s_pipeline" % arm if msg.startswith("pipeline") else "%s_runtime" % arm
                sh.fail(cls, msg)
        if rc not in (0, 3):
            for sh in batch:
                sh.fail("runner_crash", err[-300:])

        # CPU oracle.
        def do_interp(item):
            sh, vname, k, cellvals, vblob = item
            d = os.path.join(sh.work, "interp_%s" % vname)
            kp, ip, op = d + ".consts", d + ".in", d + ".out"
            with open(kp, "wb") as f:
                f.write(pack_canonical(k))
            with open(ip, "wb") as f:
                if sh.type == "vs":
                    f.write(struct.pack("<II", VB_ADDR, len(vblob)) + vblob)
                else:
                    for cell in cellvals:
                        for r in range(16):
                            f.write(struct.pack("<4f", *cell[r]))
            count = VS_COUNT if sh.type == "vs" else NCELLS
            rc, out, err = run([tools["shadertool"], "interp", sh.ucode, "--type", sh.type, "--consts", kp,
                                "--inputs", ip, "--count", str(count), "--out", op])
            return item, (op if rc == 0 else None), err
        with cf.ThreadPoolExecutor(args.j) as ex:
            for (sh, vname, *_), op, err in ex.map(do_interp, interp_jobs):
                if op:
                    outputs[(sh.name, "interp", vname)] = op
                elif vname == "seed0":
                    sh.fail("interp_fail", err.strip()[-200:])

        # Compare.
        for sh in batch:
            def load(arm, vname):
                p = outputs.get((sh.name, arm, vname))
                if not p:
                    return None
                data = open(p, "rb").read()
                if arm == "interp":
                    count = VS_COUNT if sh.type == "vs" else NCELLS
                    res = {}
                    exp = [struct.unpack_from("<256f", data, n * 1024) for n in range(count)]
                    written = struct.unpack_from("<%dQ" % count, data, count * 1024)
                    if sh.type == "vs":
                        res["pos"] = [tuple(e[62 * 4:63 * 4]) for e in exp]
                        for r in range(16):
                            if (int(sh.info["interpolator_mask"]) >> r) & 1:
                                res["o%d" % r] = [tuple(e[r * 4:r * 4 + 4]) for e in exp]
                    else:
                        for r in range(4):
                            if (sh.info["writes_color_targets"] >> r) & 1:
                                res["rt%d" % r] = [tuple(e[r * 4:r * 4 + 4]) if not sh.info["kills_pixels"] else
                                                   tuple(e[r * 4:r * 4 + 4]) for e in exp]
                        if sh.info["writes_depth"]:
                            res["depth"] = [e[61 * 4] for e in exp]
                    return res
                if sh.type == "vs":
                    return normalize_vs(sh, arm, data)
                ps = getattr(sh, arm + "_ps")
                colors = sum(1 << int(re.search(r"color\((\d+)\)", o["attr"]).group(1))
                             for o in ps["outputs"] if "color(" in o["attr"])
                return read_ps_output(p, colors, any("depth" in o["attr"] for o in ps["outputs"]))

            ref = sh.ref
            arms = [a for a in sh.arms if a != ref]
            sh.result["reference"] = ref
            sh.result["arms"] = sorted(sh.arms)
            for arm in arms:
                worst = []
                for seed in range(args.seeds):
                    ra, rb = load(ref, "seed%d" % seed), load(arm, "seed%d" % seed)
                    if ra is None or rb is None:
                        break
                    if sh.type == "ps" and ref == "interp" and sh.info["kills_pixels"]:
                        # The interpreter does not model kill; drop killed cells.
                        for key in rb:
                            if key in ra:
                                ra[key] = [va if not (isinstance(vb, tuple) and vb[0] == SENTINEL) else vb
                                           for va, vb in zip(ra[key], rb[key])]
                    mism = compare(ra, rb)
                    common = sorted(set(ra) & set(rb))
                    sh.result.setdefault("compared_%s" % arm, sum(len(ra[k]) for k in common))
                    sh.result.setdefault("keys_%s" % arm, common)
                    if os.environ.get("SHADERDIFF_DUMP"):
                        for k in common[:3]:
                            print(sh.name, arm, k, "ref", ra[k][:2], "arm", rb[k][:2])
                    if not common:
                        sh.fail("%s_vs_%s:no_common_outputs" % (arm, ref), "%s vs %s" % (sorted(ra), sorted(rb)))
                    if mism:
                        cls = classify(mism, ra, rb, set(ra) - set(rb), set(rb) - set(ra))
                        for c in cls:
                            sh.fail("%s_vs_%s:%s" % (arm, ref, c))
                        worst = mism
                        break
                if worst:
                    k0 = worst[0]
                    sh.result["first_mismatch_%s" % arm] = {"key": k0[0], "index": k0[1], "comp": "xyzw"[k0[2]],
                                                            ref: k0[3], arm: k0[4], "count": len(worst)}
                    # Localization: sensitivity of each arm to each perturbed
                    # register vs the seed0 baseline.
                    if not args.no_localize:
                        base_r, base_a = load(ref, "seed0"), load(arm, "seed0")
                        suspects = []
                        for vname, *_ in sh.variants[args.seeds:]:
                            vr, va = load(ref, vname), load(arm, vname)
                            if vr is None or va is None or base_r is None or base_a is None:
                                continue
                            sr = bool(compare(base_r, vr))
                            sa = bool(compare(base_a, va))
                            if sr != sa:
                                suspects.append("%s(read by %s only)" % (vname, ref if sr else arm))
                        if suspects:
                            sh.result["suspects_%s" % arm] = suspects
                            kinds = {"c": "constant", "r": "interpolator"}
                            for s_ in suspects:
                                sh.fail("%s_vs_%s:%s_sensitivity" % (arm, ref, kinds[s_[0]]))
            sh.result["classes"] = sorted(set(sh.result["classes"]))
        for sh in batch:
            # Stream results so a killed run keeps finished batches.
            results.write(json.dumps(sh.result) + "\n")
            written.add(sh.name)
        results.flush()
        print("batch %d-%d done (%.0fs)" % (bstart, bstart + len(batch), time.time() - t0), flush=True)
        if not os.environ.get("SHADERDIFF_KEEP_JOBS"):
            shutil.rmtree(jdir, ignore_errors=True)

    for sh in shaders:
        sh.result["classes"] = sorted(set(sh.result["classes"]))
        if sh.name not in written:
            results.write(json.dumps(sh.result) + "\n")
    results.close()
    write_summary(args, shaders, time.time() - t0)


def write_summary(args, shaders, secs):
    groups = defaultdict(list)
    passed = 0
    skipped = 0
    for sh in shaders:
        if sh.result.get("skipped"):
            skipped += 1
        if not sh.result["classes"]:
            if not sh.result.get("skipped"):
                passed += 1
            continue
        for c in sh.result["classes"]:
            groups[c].append(sh)
    lines = ["# shaderdiff summary", "",
             "- shaders: %d, passed: %d, with findings: %d, skipped (fewer than two arms): %d, %.0fs" %
             (len(shaders), passed, sum(1 for s in shaders if s.result["classes"]), skipped, secs),
             "- arms: %s, seeds: %d, ucode: %s" % (",".join(args.arms), args.seeds, args.ucode), "",
             "| class | shaders | examples |", "|---|---|---|"]
    for c, shs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append("| %s | %d | %s |" % (c, len(shs), ", ".join(s.name for s in shs[:6])))
    lines += ["", "## details (first 200)", ""]
    n = 0
    for sh in shaders:
        if not sh.result["classes"] or n >= 200:
            continue
        n += 1
        lines.append("### %s (%s)" % (sh.name, sh.type))
        for k, v in sh.result.items():
            if k.startswith(("first_mismatch", "suspects")) or k == "notes":
                if v:
                    lines.append("- %s: %s" % (k, v))
        lines.append("- classes: %s" % ", ".join(sh.result["classes"]))
        lines.append("")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
