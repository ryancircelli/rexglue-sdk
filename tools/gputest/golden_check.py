#!/usr/bin/env python3
"""Fast side of the golden store: run the Metal adapters on every stored case
in parallel and compare bit-exact against the referees' goldens.

usage: golden_check.py --store DIR [--layers a,b] [-j N] [--max-ppm 1]
                       [--adapter LAYER=CMD ...] [--out DIR] [--limit N]
                       [--gate-single]

Store format: bo2-mac-port research/golden-store.md (v1.1).

Adapters follow the batch contract: `CMD --batch LIST STORE OUTROOT`, LIST =
one case manifest path per line, outputs in OUTROOT/<case_id>/<name>, a case
that cannot run writes OUTROOT/<case_id>/ERROR. Built-in: shader_vs and
shader_ps (golden_shader.py, native Metal translator). Others come from
--adapter or the environment: GOLDEN_ADAPTER_<LAYER>="cmd ..." (for example
GOLDEN_ADAPTER_TEXEL_DECODE).

Per layer it reports agreed cases (two or more referees, identical outputs)
and single-referee cases apart; stale, disputed and empty cases are counted
but never compared. Exit 1 if any layer's agreed mismatches per million are
above --max-ppm (with --gate-single, single cases too), 77 if nothing was
compared (regress.sh SKIP), else 0.
"""

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_store as gs  # noqa: E402

BUILTIN = {
    "shader_vs": [sys.executable, os.path.join(HERE, "golden_shader.py"), "adapt"],
    "shader_ps": [sys.executable, os.path.join(HERE, "golden_shader.py"), "adapt"],
}


def adapter_for(layer, overrides):
    if layer in overrides:
        return overrides[layer]
    env = os.environ.get("GOLDEN_ADAPTER_" + layer.upper())
    if env:
        return shlex.split(env)
    return BUILTIN.get(layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="~/Games/Xbox360/bo2-build/gputest/golden")
    ap.add_argument("--layers", default="")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-ppm", type=float, default=1.0)
    ap.add_argument("--adapter", action="append", default=[], help="LAYER=CMD")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="cases per layer (0 = all)")
    ap.add_argument("--gate-single", action="store_true", help="single-referee cases also gate")
    ap.add_argument("--keep", action="store_true", help="keep adapter outputs")
    args = ap.parse_args()

    store = gs.Store(args.store)
    out = args.out or os.path.join(os.environ.get("REGRESS_OUT", "/tmp/bo2/gputest"), "golden-check")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    overrides = {}
    for a in args.adapter:
        k, v = a.split("=", 1)
        overrides[k] = shlex.split(v)
    layers = set(filter(None, args.layers.split(",")))
    rules = store.rules()

    t0 = time.time()
    counts = defaultdict(lambda: defaultdict(int))
    todo = defaultdict(list)  # layer -> [(path, manifest)]
    for path, m in store.cases(layers or None):
        st = gs.update_status(m, rules)
        counts[m["layer"]][st] += 1
        if st in ("agreed", "single"):
            if args.limit and len(todo[m["layer"]]) >= args.limit:
                continue
            todo[m["layer"]].append((path, m))

    # Run adapters: per layer, one list per worker, all layers' workers in
    # one pool so every core is busy.
    tasks, no_adapter = [], set()
    for layer, items in todo.items():
        cmd = adapter_for(layer, overrides)
        if not cmd:
            no_adapter.add(layer)
            continue
        n = max(1, min(args.j, len(items)))
        for w in range(n):
            part = items[w::n]
            lst = os.path.join(out, "%s.%03d.list" % (layer, w))
            with open(lst, "w") as f:
                f.write("".join(p + "\n" for p, _ in part))
            tasks.append((layer, cmd, lst, os.path.join(out, "res")))

    def run(task):
        layer, cmd, lst, root = task
        os.makedirs(root, exist_ok=True)
        log = lst + ".log"
        with open(log, "w") as lf:
            rc = subprocess.call(cmd + ["--batch", lst, store.root, root], stdout=lf, stderr=subprocess.STDOUT)
        return layer, rc, log

    bad_runs = []
    with cf.ThreadPoolExecutor(max(1, args.j)) as ex:
        for layer, rc, log in ex.map(run, tasks):
            if rc != 0:
                bad_runs.append((layer, rc, log))

    # Compare.
    stats = defaultdict(lambda: {"cases": 0, "values": 0, "mism": 0, "bad_cases": 0, "first": None, "max_ulp": None})
    worst = open(os.path.join(out, "mismatches.jsonl"), "w")
    for layer, items in todo.items():
        if layer in no_adapter:
            continue
        for path, m in items:
            kind = "agreed" if m["status"] == "agreed" else "single"
            s = stats[(layer, kind)]
            s["cases"] += 1
            d = os.path.join(out, "res", m["id"])
            err = None
            if os.path.exists(os.path.join(d, "ERROR")):
                err = open(os.path.join(d, "ERROR")).read().strip()
            case_mism, firsts = 0, []
            for name, h in m["golden"].items():
                spec = m["outputs"][name]
                want = store.get(h)
                gp = os.path.join(d, name)
                got = open(gp, "rb").read() if (not err and os.path.exists(gp)) else None
                total, mism, first, ulp = gs.compare_blob(got, want, spec["elem"], spec.get("compare", "bits"))
                s["values"] += total
                s["mism"] += mism
                case_mism += mism
                if ulp is not None:
                    s["max_ulp"] = max(s["max_ulp"] or 0, ulp)
                if first:
                    i, a, b = first
                    firsts.append({"output": name, "index": i, "got": a if isinstance(a, str) else "0x%x" % a,
                                   "want": b if isinstance(b, str) else "0x%x" % b, "mismatches": mism,
                                   "values": total, "max_ulp": ulp})
            if case_mism:
                s["bad_cases"] += 1
                rec = {"layer": layer, "kind": kind, "case": m["id"], "source": m.get("source", ""),
                       "referees": sorted(m["referees"]), "error": err, "outputs": firsts}
                worst.write(json.dumps(rec) + "\n")
                if s["first"] is None:
                    s["first"] = rec
    worst.close()
    if not args.keep:
        shutil.rmtree(os.path.join(out, "res"), ignore_errors=True)

    # Report.
    fail = False
    checked = 0
    lines = []
    all_layers = sorted(set(counts) | set(todo))
    for layer in all_layers:
        c = counts[layer]
        other = ", ".join("%s %d" % (k, c[k]) for k in ("stale", "disputed", "empty") if c.get(k))
        if layer in no_adapter:
            lines.append("%-14s NOT CHECKED (no Metal adapter): agreed %d, single %d%s" %
                         (layer, c.get("agreed", 0), c.get("single", 0), (", " + other) if other else ""))
            continue
        for kind in ("agreed", "single"):
            s = stats.get((layer, kind))
            if not s or not s["cases"]:
                continue
            checked += s["cases"]
            ppm = s["mism"] * 1e6 / s["values"] if s["values"] else 0.0
            gate = kind == "agreed" or args.gate_single
            over = ppm > args.max_ppm
            if gate and over:
                fail = True
            lines.append("%-14s %-6s %6d cases %12d values %10d mismatches %10.3f ppm  %d bad cases%s%s%s" % (
                layer, kind, s["cases"], s["values"], s["mism"], ppm, s["bad_cases"],
                ("  max %d ulp" % s["max_ulp"]) if s["max_ulp"] else "",
                "  FAIL" if gate and over else ("" if gate else "  (info)"),
                ("  [" + other + "]") if other and kind == "agreed" else ""))
            if s["first"]:
                f = s["first"]
                o = f["outputs"][0] if f["outputs"] else {}
                lines.append("    first: case %s (%s) %s%s[%s] got %s want %s" % (
                    f["case"][:16], f["source"][:60], ("ERROR %s; " % f["error"][:80]) if f["error"] else "",
                    o.get("output", "?"), o.get("index", "?"), o.get("got", "?"), o.get("want", "?")))
    for layer, rc, log in bad_runs:
        lines.append("adapter for %s exited %d (log %s)" % (layer, rc, log))
        fail = True
    print("golden store %s (%.0fs, -j %d, details %s)" % (store.root, time.time() - t0, args.j, out))
    for l in lines:
        print("  " + l)
    rc = 0
    if fail:
        summary = "FAIL: above %g ppm or adapter failure" % args.max_ppm
        rc = 1
    elif not checked:
        summary = "SKIP: no agreed or single cases with a Metal adapter"
        rc = 77
    else:
        tot = sum(s["values"] for (l, k), s in stats.items() if k == "agreed")
        mis = sum(s["mism"] for (l, k), s in stats.items() if k == "agreed")
        stot = sum(s["values"] for (l, k), s in stats.items() if k == "single")
        smis = sum(s["mism"] for (l, k), s in stats.items() if k == "single")
        summary = "PASS: agreed %d/%d values mismatched (%.3f ppm); single-referee %d/%d (info)" % (
            mis, tot, mis * 1e6 / tot if tot else 0.0, smis, stot)
        if not tot and not args.gate_single:
            # Nothing gated: report as SKIP so the ship gate does not read
            # a single-referee-only store as a pass.
            summary = "SKIP: no agreed goldens yet; single-referee %d/%d values mismatched (%.0f ppm, info)" % (
                smis, stot, smis * 1e6 / stot if stot else 0.0)
            rc = 77
    print(summary)
    return rc


if __name__ == "__main__":
    sys.exit(main())
