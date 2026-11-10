"""Content-addressed golden store (format: bo2-mac-port research/golden-store.md v1.1).

store/
  objects/<h[0:2]>/<h>            raw immutable blobs, h = sha256 hex
  cases/<layer>/<case_id>.json    manifests
  disputes/<layer>/<case_id>.json referee disagreements
  rules.json                      {rule_id: {version, source, assumed}}

Library for producers and for golden_check.py. Python 3 standard library only.
"""

import fcntl
import hashlib
import json
import os
import struct
import tempfile

ELEM = {"f32": ("f", 4), "f16": ("e", 2), "u32": ("I", 4), "i32": ("i", 4),
        "u16": ("H", 2), "u8": ("B", 1)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def case_id(layer, inputs, params):
    return sha(canonical({"layer": layer, "inputs": inputs, "params": params}).encode())


class Store:
    def __init__(self, root):
        self.root = os.path.expanduser(root)

    def obj_path(self, h):
        return os.path.join(self.root, "objects", h[:2], h)

    def put(self, data):
        h = sha(data)
        p = self.obj_path(h)
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".tmp-")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        return h

    def get(self, h):
        with open(self.obj_path(h), "rb") as f:
            return f.read()

    def case_path(self, layer, cid, disputed=False):
        return os.path.join(self.root, "disputes" if disputed else "cases", layer, cid + ".json")

    def rules(self):
        p = os.path.join(self.root, "rules.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    def cases(self, layers=None):
        """Yields (path, manifest) for every case (not disputes)."""
        base = os.path.join(self.root, "cases")
        if not os.path.isdir(base):
            return
        for layer in sorted(os.listdir(base)):
            if layers and layer not in layers:
                continue
            d = os.path.join(base, layer)
            for n in sorted(os.listdir(d)):
                if n.endswith(".json"):
                    p = os.path.join(d, n)
                    with open(p) as f:
                        yield p, json.load(f)

    def add_referee(self, layer, inputs, params, outputs_spec, referee, version, results,
                    rules=None, game="bo2", source=""):
        """Merges one referee's result (results: {name: bytes}) into the case.
        Returns the case id. Locks the manifest so producers can run in
        parallel."""
        cid = case_id(layer, inputs, params)
        p = self.case_path(layer, cid)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        hashes = {n: self.put(b) for n, b in results.items()}
        lock_dir = os.path.join(self.root, "locks", layer)
        os.makedirs(lock_dir, exist_ok=True)
        with open(os.path.join(lock_dir, cid[:2]), "a") as lk:  # 256 shared lock files per layer
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
            else:
                m = {"schema": 1, "layer": layer, "id": cid, "game": game, "source": source,
                     "inputs": inputs, "params": params, "outputs": outputs_spec, "referees": {}}
            for n, spec in outputs_spec.items():
                m["outputs"].setdefault(n, spec)
            m["referees"][referee] = {"version": version, "outputs": hashes, "rules": rules or {}}
            update_status(m, self.rules())
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".tmp-")
            with os.fdopen(fd, "w") as f:
                json.dump(m, f, indent=1, sort_keys=True)
            os.replace(tmp, p)
        return cid


def stale(entry, rules):
    for rid, ver in entry.get("rules", {}).items():
        if rid in rules and int(ver) < int(rules[rid]["version"]):
            return True
    return False


def update_status(m, rules):
    """Sets status/golden from the non-stale referees. Compared outputs only
    (compare != none). Returns the status."""
    compared = [n for n, s in m["outputs"].items() if s.get("compare", "bits") != "none"]
    live = {r: e for r, e in m["referees"].items() if not stale(e, rules)}
    m.pop("golden", None)
    if not live:
        m["status"] = "stale" if m["referees"] else "empty"
        return m["status"]
    first = next(iter(live.values()))["outputs"]
    agree = all(all(e["outputs"].get(n) == first.get(n) for n in compared) for e in live.values())
    if not agree:
        m["status"] = "disputed"
    else:
        m["status"] = "agreed" if len(live) >= 2 else "single"
        m["golden"] = {n: first[n] for n in compared if n in first}
    return m["status"]


def unpack(data, elem):
    fmt, size = ELEM[elem]
    n = len(data) // size
    return struct.unpack("<%d%s" % (n, fmt), data[:n * size])


def raw_words(data, elem):
    """Bit patterns as ints (for bit-exact compares of float data)."""
    fmt, size = ELEM[elem]
    ifmt = {1: "B", 2: "H", 4: "I"}[size]
    n = len(data) // size
    return struct.unpack("<%d%s" % (n, ifmt), data[:n * size])


def is_nan_bits(v, elem):
    if elem == "f32":
        return (v & 0x7F800000) == 0x7F800000 and (v & 0x7FFFFF) != 0
    if elem == "f16":
        return (v & 0x7C00) == 0x7C00 and (v & 0x3FF) != 0
    return False


def compare_blob(got, want, elem, mode):
    """Returns (values, mismatches, first, max_ulp) where first is
    (index, got, want) and max_ulp is the largest f32 ULP distance among
    finite mismatching values (None for other element types)."""
    if mode == "none":
        return 0, 0, None, None
    w = raw_words(want, elem)
    g = raw_words(got, elem) if got is not None else ()
    total = len(w)
    mism, first = 0, None
    if len(g) != total:
        # Wrong size counts as all values wrong.
        return total, total, (-1, "size %d" % len(g), "size %d" % total), None
    if got == want:
        return total, 0, None, None
    max_ulp = 0 if elem == "f32" else None
    for i, (a, b) in enumerate(zip(g, w)):
        if a == b:
            continue
        if mode == "bits_nan_any" and is_nan_bits(a, elem) and is_nan_bits(b, elem):
            continue
        mism += 1
        if first is None:
            first = (i, a, b)
        if max_ulp is not None and not is_nan_bits(a, elem) and not is_nan_bits(b, elem):
            max_ulp = max(max_ulp, abs(f32_ordinal(a) - f32_ordinal(b)))
    return total, mism, first, max_ulp


def f32_ordinal(bits):
    """Maps f32 bits to a monotonic integer so ULP distance is a subtraction."""
    return bits & 0x7FFFFFFF if not bits & 0x80000000 else -(bits & 0x7FFFFFFF)
