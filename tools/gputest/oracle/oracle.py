#!/usr/bin/env python3
"""Reference-image oracle for rexgpu-replay outputs.

Works on replay directories (draws.jsonl + draws/*.npy, see
tools/gputest/replay/README.md) written by rexgpu-replay on any backend, and
by the Xenia Canary trace dump patched with the same per-draw readback
(tools/gputest/oracle/xenia/).

  oracle.py determinism --trace T.xtr --replay BIN --plugin LIB [-n 3] -o DIR
      Replays T N times through one plugin and reports bit-exact agreement of
      every run with run 1, per draw and per frame. Exit 1 when any run
      differs. Races (speckle, stale textures) show up as differences on
      identical input.

  oracle.py exact A_DIR B_DIR [-o DIR]
      Bit-exact pixel agreement between two replays: identical pixels / total
      per draw (whole target and pixels either side wrote) and per frame.

  oracle.py vote --ref NAME=DIR [--ref NAME=DIR ...] --test NAME=DIR -o DIR
                 [--tol 0] [--depth-tol 0]
      Majority vote. Where all references agree bit-exactly on a pixel, that
      pixel is reference truth. The first draw whose test output differs from
      truth (on what the draw itself changed, so inherited damage is not
      re-reported) is the first divergent draw. Draws where the references
      disagree with each other are listed separately as referee
      disagreements.

Requires numpy (and Pillow for images).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))
from replay_diff import Replay, to_rgb8  # noqa: E402

try:
    from PIL import Image
except ImportError:  # images are optional
    Image = None


# ---------------------------------------------------------------------------
# Bit-exact helpers
# ---------------------------------------------------------------------------

def bits(image):
    """uint32 view: (h, w, c) for color, (h, w) for depth."""
    if image.dtype == np.uint32:
        return image
    return np.ascontiguousarray(image).view(np.uint32)


def pixel_neq(a, b):
    """Mask of pixels whose bits differ in any channel."""
    ba, bb = bits(a), bits(b)
    if ba.ndim == 3:
        return np.any(ba != bb, axis=-1)
    return ba != bb


def pixel_changed(before, after):
    if before is None or before.shape != after.shape:
        return np.ones(after.shape[:2], bool)
    return pixel_neq(before, after)


def load_states(path, frame=None):
    """List of (draw record, {target: (key, after, before)}) for one replay,
    optionally restricted to one frame. Materialises images lazily via
    Replay.states (each yielded image is a fresh array or a reference to the
    previous one, never mutated afterwards)."""
    r = Replay(path)
    out = []
    for d, targets in r.states():
        if frame is not None and d.get("frame") != frame:
            continue
        out.append((d, targets))
    return r, out


def draw_id(d):
    # Draws are matched by trace packet index (global, one DRAW packet per
    # draw), so implementations that skip different draws (VIZ query
    # predication, failed draws) still line up. Older replays without the
    # field fall back to the draw counter.
    if d.get("packet") is not None:
        return (d.get("frame"), "p", d["packet"])
    return (d.get("frame"), d.get("draw"))


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------

def exact_compare(path_a, path_b):
    """Returns dict with per-draw and per-frame bit-exact stats."""
    ra, sa = load_states(path_a)
    rb, sb = load_states(path_b)
    per_draw = []
    frames = {}  # frame -> {target: (key, img_a, img_b)} final state
    structural = []
    n = min(len(sa), len(sb))
    if len(sa) != len(sb):
        structural.append(f"draw count differs: {len(sa)} vs {len(sb)}")
    for i in range(n):
        da, ta = sa[i]
        db, tb = sb[i]
        if (da.get("vs"), da.get("ps")) != (db.get("vs"), db.get("ps")):
            structural.append(f"draw {i}: shader hashes differ ({da.get('vs')}/{da.get('ps')} vs {db.get('vs')}/{db.get('ps')})")
        rec = {"draw": da.get("draw"), "frame": da.get("frame"), "vs": da.get("vs"), "ps": da.get("ps"), "targets": []}
        for label in sorted(set(ta) | set(tb)):
            if label not in ta or label not in tb:
                rec["targets"].append({"target": label, "missing": "a" if label not in ta else "b"})
                continue
            ka, a, bef_a = ta[label]
            kb, b, bef_b = tb[label]
            if ka != kb or a.shape != b.shape:
                rec["targets"].append({"target": label, "key_mismatch": [ka, kb], "shape": [list(a.shape), list(b.shape)]})
                continue
            neq = pixel_neq(a, b)
            wrote = pixel_changed(bef_a, a) | pixel_changed(bef_b, b)
            total = int(neq.size)
            rec["targets"].append({
                "target": label, "key": ka, "total": total,
                "exact": total - int(neq.sum()),
                "written": int(wrote.sum()),
                "written_exact": int(wrote.sum()) - int((neq & wrote).sum()),
            })
            frames.setdefault(da.get("frame"), {})[label] = (ka, a, b)
        per_draw.append(rec)
    per_frame = []
    for f in sorted(frames):
        tot = ex = 0
        tgts = []
        for label, (key, a, b) in sorted(frames[f].items()):
            neq = pixel_neq(a, b)
            tgts.append({"target": label, "key": key, "total": int(neq.size), "exact": int(neq.size - neq.sum())})
            tot += neq.size
            ex += int(neq.size - neq.sum())
        per_frame.append({"frame": f, "total": int(tot), "exact": int(ex), "ratio": ex / tot if tot else 1.0, "targets": tgts})
    all_tot = sum(t["total"] for r in per_draw for t in r["targets"] if "total" in t)
    all_ex = sum(t["exact"] for r in per_draw for t in r["targets"] if "total" in t)
    w_tot = sum(t["written"] for r in per_draw for t in r["targets"] if "total" in t)
    w_ex = sum(t["written_exact"] for r in per_draw for t in r["targets"] if "total" in t)
    first = None
    for r in per_draw:
        for t in r["targets"]:
            if "total" not in t or t["exact"] != t["total"]:
                first = {"draw": r["draw"], "frame": r["frame"], "vs": r["vs"], "ps": r["ps"], "target": t}
                break
        if first:
            break
    return {
        "a": path_a, "b": path_b,
        "draws": n,
        "structural": structural,
        "draw_pixels": {"total": all_tot, "exact": all_ex, "ratio": all_ex / all_tot if all_tot else 1.0},
        "written_pixels": {"total": w_tot, "exact": w_ex, "ratio": w_ex / w_tot if w_tot else 1.0},
        "frames": per_frame,
        "bit_exact": not structural and all_ex == all_tot,
        "first_difference": first,
        "per_draw": per_draw,
    }


def fmt_ratio(ex, tot):
    if tot == 0:
        return "n/a"
    return f"{ex}/{tot} ({100.0 * ex / tot:.6f}%)"


def print_exact(res, name_a="A", name_b="B", out=sys.stdout):
    print(f"{name_a} vs {name_b}: {res['draws']} draws", file=out)
    for s in res["structural"][:5]:
        print(f"  STRUCTURAL: {s}", file=out)
    dp, wp = res["draw_pixels"], res["written_pixels"]
    print(f"  per-draw pixels bit-exact: {fmt_ratio(dp['exact'], dp['total'])}", file=out)
    print(f"  written pixels bit-exact:  {fmt_ratio(wp['exact'], wp['total'])}", file=out)
    for f in res["frames"]:
        print(f"  frame {f['frame']} final targets bit-exact: {fmt_ratio(f['exact'], f['total'])}", file=out)
    if res["first_difference"]:
        fd = res["first_difference"]
        print(f"  first difference: draw {fd['draw']} (frame {fd['frame']}) vs {fd['vs']} ps {fd['ps']} {fd['target']}", file=out)
    else:
        print("  BIT-EXACT", file=out)


def run_exact(args):
    res = exact_compare(args.a, args.b)
    print_exact(res, os.path.basename(os.path.normpath(args.a)), os.path.basename(os.path.normpath(args.b)))
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "exact.json"), "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["bit_exact"] else 1


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

SCRATCH_SIZE = 32 << 20
DEFAULT_SCRATCH = 0x1C000000


def used_pages(draws_jsonl):
    """1 MiB physical pages the trace's draws touch: texture bases (+4 MiB for
    mips), vertex and index buffers, and the destinations of the game's own
    resolves (+8 MiB). Resolve destinations are GPU writes, so the replay's
    own scratch-conflict check (trace memory records only) misses them."""
    used = set()

    def mark(addr, size):
        addr &= 0x1FFFFFFF
        for p in range(addr >> 20, ((addr + max(size, 1) - 1) >> 20) + 1):
            used.add(p)

    with open(draws_jsonl) as f:
        for line in f:
            d = json.loads(line)
            for t in d.get("textures") or []:
                if t.get("base"):
                    mark(int(t["base"], 16), 4 << 20)
            for v in d.get("vertex_fetch") or []:
                mark(int(v["address"], 16), v.get("bytes") or 1)
            if d.get("indexed") and d.get("index_bytes"):
                mark(int(d["index_base"], 16), d["index_bytes"])
            r = d.get("regs") or {}
            if d.get("is_resolve") and "RB_COPY_DEST_BASE" in r:
                v = r["RB_COPY_DEST_BASE"]
                mark(int(v, 16) if isinstance(v, str) else v, 8 << 20)
    return used


def scratch_conflicts(used, base):
    return sorted(p for p in range(base >> 20, (base + SCRATCH_SIZE) >> 20) if p in used)


def pick_scratch(used):
    """Highest 32 MiB-aligned-to-1 MiB window no draw touches, else None."""
    need = SCRATCH_SIZE >> 20
    for start in range(512 - need, -1, -1):
        if start < 16:
            break
        if not any(p in used for p in range(start, start + need)):
            return start << 20
    return None


def run_probe(args, probe):
    """Replay with no readbacks to collect draw metadata."""
    if os.path.exists(os.path.join(probe, "draws.jsonl")):
        return
    cmd = [args.replay, args.trace, "-o", probe, "--backend", args.backend,
           "--log-level", "warn", "--draws", "999999999:1"]
    if args.plugin:
        cmd += ["--plugin", args.plugin]
    if args.frames:
        cmd += ["--frames", str(args.frames)]
    with open(probe + ".log", "w") as log:
        subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def resolve_scratch(args):
    """--scratch auto: move the readback scratch out of anything the game
    uses. The default 0x1C000000 overlaps BO2's render-to-texture resolve
    targets in gameplay (shadow/post buffers at 0x1C3-0x1DC), so per-draw
    readbacks clobber textures later draws sample."""
    if not args.scratch:
        return None
    if args.scratch != "auto":
        return int(args.scratch, 0)
    probe = args.probe or os.path.join(args.output, "probe")
    run_probe(args, probe)
    used = used_pages(os.path.join(probe, "draws.jsonl"))
    bad = scratch_conflicts(used, DEFAULT_SCRATCH)
    base = DEFAULT_SCRATCH if not bad else pick_scratch(used)
    if base is None:
        raise SystemExit("no free 32 MiB window for the readback scratch")
    print(f"scratch: default 0x{DEFAULT_SCRATCH:08X} overlaps {len(bad)} used MiB pages; "
          f"using 0x{base:08X}" if bad else f"scratch: default 0x{base:08X} is free", flush=True)
    return base


def run_determinism(args):
    os.makedirs(args.output, exist_ok=True)
    scratch = resolve_scratch(args)
    if args.final_only:
        # Per-draw readback waits for the GPU after every draw, which hides
        # CPU/GPU and inter-submission races. Read back only the last draw
        # so the frame runs unserialised; the race damage lands in the
        # final targets.
        probe = os.path.join(args.output, "probe")
        cmd = [args.replay, args.trace, "-o", probe, "--backend", args.backend, "--log-level", "warn"]
        if args.plugin:
            cmd += ["--plugin", args.plugin]
        if args.frames:
            cmd += ["--frames", str(args.frames)]
        with open(probe + ".log", "w") as log:
            subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        # Readbacks only happen on colour/depth (non-resolve) draws, so pick
        # the last draw the probe actually read back.
        last = max(d["draw"] for d in Replay(probe).draws if d.get("targets"))
        args.draws = f"{last}:1"
        print(f"final-only: reading back draw {last} only", flush=True)
    runs = []
    for i in range(args.n):
        d = os.path.join(args.output, f"run{i + 1}")
        cmd = [args.replay, args.trace, "-o", d, "--label", f"{args.label}_run{i + 1}",
               "--backend", args.backend, "--log-level", "warn"]
        if args.plugin:
            cmd += ["--plugin", args.plugin]
        for c in args.cvar or []:
            cmd += ["--cvar", c]
        if args.draws:
            cmd += ["--draws", args.draws]
        if args.frames:
            cmd += ["--frames", str(args.frames)]
        if scratch is not None:
            cmd += ["--scratch", f"0x{scratch:08X}"]
        t0 = time.time()
        with open(d + ".log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"run {i + 1}: rc={rc} {time.time() - t0:.1f}s -> {d}", flush=True)
        if rc != 0:
            print(f"replay failed, see {d}.log")
            return 2
        runs.append(d)
    summary = {"trace": args.trace, "plugin": args.plugin, "runs": runs, "pairs": []}
    ok = True
    for d in runs[1:]:
        res = exact_compare(runs[0], d)
        print_exact(res, "run1", os.path.basename(d))
        ok &= res["bit_exact"]
        res.pop("per_draw")
        summary["pairs"].append(res)
    summary["bit_exact"] = ok
    with open(os.path.join(args.output, "determinism.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("DETERMINISTIC (bit-exact on every run)" if ok else "NONDETERMINISTIC")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# vote
# ---------------------------------------------------------------------------

def tol_neq(a, b, tol, depth_tol):
    if tol == 0 and depth_tol == 0:
        return pixel_neq(a, b)
    if a.ndim == 2:
        return np.abs((a >> 8).astype(np.int64) - (b >> 8).astype(np.int64)) > depth_tol
    fa, fb = a.astype(np.float64), b.astype(np.float64)
    nan = np.any(np.isnan(fa) != np.isnan(fb), axis=-1)
    err = np.abs(np.nan_to_num(fa) - np.nan_to_num(fb)).max(axis=-1)
    return (err > tol) | nan


def save_vote_image(path, refs, test, consensus, bad):
    if Image is None:
        return
    tiles = [to_rgb8(img) for img in refs] + [to_rgb8(test)]
    m = np.zeros(test.shape[:2] + (3,), np.uint8)
    m[..., 1] = 80 * consensus
    m[bad] = (255, 0, 0)
    tiles.append(m)
    h = max(t.shape[0] for t in tiles)
    row = []
    for t in tiles:
        if t.shape[0] < h:
            t = np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)))
        row += [t, np.full((h, 4, 3), 255, np.uint8)]
    Image.fromarray(np.concatenate(row[:-1], axis=1)).save(path)


def parse_named(s):
    if "=" in s:
        n, p = s.split("=", 1)
        return n, p
    return os.path.basename(os.path.normpath(s)), s


def run_vote(args):
    refs = [parse_named(s) for s in args.ref]
    tname, tpath = parse_named(args.test)
    loaded = [(n, load_states(p, args.frame)[1]) for n, p in refs]
    test = load_states(tpath, args.frame)[1]
    n = min([len(test)] + [len(s) for _, s in loaded])
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)
    divergent, disagreements, structural = [], [], []
    shown = 0
    for i in range(n):
        dt, tt = test[i]
        rec_refs = [(nm, s[i]) for nm, s in loaded]
        for nm, (dr, _) in rec_refs:
            if (dr.get("vs"), dr.get("ps")) != (dt.get("vs"), dt.get("ps")):
                structural.append(f"draw {i}: {nm} shaders {dr.get('vs')}/{dr.get('ps')} vs {tname} {dt.get('vs')}/{dt.get('ps')}")
        for label in sorted(tt):
            key, timg, tbef = tt[label]
            rimgs = []
            for nm, (dr, tr) in rec_refs:
                if label in tr and tr[label][0] == key and tr[label][1].shape == timg.shape:
                    rimgs.append((nm, tr[label][1], tr[label][2]))
            if len(rimgs) != len(rec_refs):
                continue
            # Consensus: pixels on which every reference agrees bit-exactly.
            agree = np.ones(timg.shape[:2], bool)
            for _, img, _ in rimgs[1:]:
                agree &= ~pixel_neq(rimgs[0][1], img)
            # Only judge what this draw changed in any implementation, so
            # damage inherited from an earlier draw is not re-reported.
            wrote = pixel_changed(tbef, timg)
            for _, img, bef in rimgs:
                wrote |= pixel_changed(bef, img)
            judged = agree & wrote
            if len(rimgs) > 1:
                dis = int((wrote & ~agree).sum())
                if dis:
                    disagreements.append({"index": i, "draw": dt.get("draw"), "frame": dt.get("frame"),
                                          "target": label, "pixels": dis, "written": int(wrote.sum()),
                                          "vs": dt.get("vs"), "ps": dt.get("ps")})
            # Pixels that already differed before this draw are inherited
            # damage (blending carries it forward); the first draw to break a
            # pixel is the one reported.
            bad = judged & tol_neq(timg, rimgs[0][1], args.tol, args.depth_tol)
            if tbef is not None and rimgs[0][2] is not None and tbef.shape == timg.shape:
                bad &= ~tol_neq(tbef, rimgs[0][2], args.tol, args.depth_tol)
            nbad = int(bad.sum())
            if nbad > args.max_pixels:
                entry = {"index": i, "draw": dt.get("draw"), "frame": dt.get("frame"), "target": label, "key": key,
                         "bad_pixels": nbad, "judged_pixels": int(judged.sum()), "written": int(wrote.sum()),
                         "vs": dt.get("vs"), "ps": dt.get("ps"),
                         "textures": dt.get("textures"), "regs": dt.get("regs")}
                ys, xs = np.nonzero(bad)
                entry["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                if shown < args.images:
                    p = os.path.join(img_dir, f"draw{dt.get('draw'):06d}_{label}.png")
                    save_vote_image(p, [im for _, im, _ in rimgs], timg, agree, bad)
                    entry["image"] = p
                    shown += 1
                divergent.append(entry)
    result = {"refs": dict(refs), "test": {tname: tpath}, "draws": n, "tol": args.tol,
              "structural": structural[:50], "first_divergent": divergent[0] if divergent else None,
              "divergent": divergent, "referee_disagreements": disagreements}
    with open(os.path.join(args.output, "vote.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"vote: refs={[n for n, _ in refs]} test={tname} draws={n}")
    for s in structural[:5]:
        print(f"  STRUCTURAL: {s}")
    print(f"  referee disagreements: {len(disagreements)} draw-targets"
          + (f" (first: draw {disagreements[0]['draw']} {disagreements[0]['target']} {disagreements[0]['pixels']} px)" if disagreements else ""))
    if divergent:
        fd = divergent[0]
        print(f"  FIRST DIVERGENT DRAW: {fd['draw']} (frame {fd['frame']}) target {fd['target']} "
              f"vs {fd['vs']} ps {fd['ps']} bad {fd['bad_pixels']}/{fd['judged_pixels']} px bbox {fd['bbox']}")
        print(f"  {len(divergent)} divergent draw-targets total")
    else:
        print(f"  {tname} matches reference consensus on every judged pixel")
    return 1 if divergent else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("determinism")
    p.add_argument("--trace", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--plugin")
    p.add_argument("--backend", default="any")
    p.add_argument("--label", default="det")
    p.add_argument("--cvar", action="append")
    p.add_argument("--draws")
    p.add_argument("--frames", type=int)
    p.add_argument("--scratch", help="readback scratch address, or 'auto' to pick a window "
                   "no draw touches (probe replay first)")
    p.add_argument("--probe", help="existing no-readback replay dir to use for --scratch auto")
    p.add_argument("--final-only", action="store_true",
                   help="read back only the last draw (no per-draw GPU sync)")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=run_determinism)
    p = sub.add_parser("exact")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=run_exact)
    p = sub.add_parser("vote")
    p.add_argument("--ref", action="append", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--frame", type=int)
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--depth-tol", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=0)
    p.add_argument("--images", type=int, default=12)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=run_vote)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
