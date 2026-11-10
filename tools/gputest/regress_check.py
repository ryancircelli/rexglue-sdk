#!/usr/bin/env python3
"""Turn a shaderdiff or static_checks results.jsonl into pass/fail.

Usage: regress_check.py RESULTS.jsonl [--known FILE] [--allow CLASSES]

A finding fails the run unless its class is allowed (comma list, matched as a
substring) or the pair "shader class" is listed in the known file (one per
line, '#' comments; a class of '*' covers every class of that shader).
Known pairs that no longer show up are printed as "now passing" so the list
can be trimmed once a fix lands. Exit 0 on pass, 1 on new findings.
"""
import argparse, json, sys

DEFAULT_ALLOW = "spirv_val_needs_scalar_block_layout,native_fallback,precision"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--known", default="")
    ap.add_argument("--allow", default=DEFAULT_ALLOW)
    a = ap.parse_args()
    allow = [x for x in a.allow.split(",") if x]
    known = set()
    if a.known:
        for line in open(a.known):
            line = line.split("#", 1)[0].split()
            if len(line) >= 2:
                known.add((line[0], line[1]))
    seen, new = set(), []
    n = 0
    for line in open(a.results):
        r = json.loads(line)
        n += 1
        name = r.get("shader") or r.get("name") or "?"
        for c in r.get("classes", []):
            if any(x in c for x in allow):
                continue
            if (name, c) in known or (name, "*") in known:
                seen.add((name, c) if (name, c) in known else (name, "*"))
                continue
            arm = c.split("_vs_")[0]
            detail = []
            fm = r.get("first_mismatch_" + arm)
            if fm:
                detail.append("first mismatch " + ", ".join("%s=%s" % kv for kv in fm.items()))
            if r.get("suspects_" + arm):
                detail.append("suspects " + " ".join(r["suspects_" + arm][:4]))
            new.append((name, c, detail or r.get("notes", "")))
    for name, c, notes in new:
        if isinstance(notes, str):
            notes = [notes]
        hit = ([x for x in notes if c.split(":")[0] in str(x)]
               or [x for x in notes if not any(str(x).startswith(a) for a in allow)]
               or notes)
        note = "; ".join(map(str, hit))
        print("NEW  %s %s %s" % (name, c, note[:200]))
    for k in sorted(known - seen):
        print("now passing (drop from known list): %s %s" % k)
    print("%d results, %d new findings, %d known still present"
          % (n, len(new), len(seen)))
    return 1 if new or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
