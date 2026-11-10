#!/usr/bin/env python3
"""Per-op ULP of Metal fast:: and precise:: transcendentals (and of the SDK
CPU interpreter's float libm) against the correctly rounded result, plus a
rough GPU cost of each.

usage: msl_math_ulp.py [--work DIR] [--runner rexgpu-metal-runner] [--n LOG2]

The reference is the double-precision result rounded once to f32 (correct
rounding except in vanishingly rare double-rounding cases). Overflowing and
denormal results are skipped. Timing runs a dependent 8192-op chain on 16M
threads; numbers include process start (~0.1 s) and are noisy on a busy Mac.
"""
import argparse, array, collections, ctypes, json, math, os, random, struct, subprocess, time

OPS = ["rsqrt", "rcp", "sqrt", "exp2", "log2", "sin", "cos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/msl-math-ulp")
    ap.add_argument("--runner", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "../../out/mac-arm64/Release/rexgpu-metal-runner"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--no-perf", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.chdir(a.work)
    N, rng = 1 << a.n, random.Random(1)
    gen = {
        "rsqrt": lambda: 2.0 ** rng.uniform(-60, 60),
        "rcp": lambda: rng.choice((-1, 1)) * 2.0 ** rng.uniform(-60, 60),
        "sqrt": lambda: 2.0 ** rng.uniform(-60, 60),
        "exp2": lambda: rng.uniform(-60, 60),
        "log2": lambda: 2.0 ** rng.uniform(-60, 60) if rng.random() < .7 else rng.uniform(0.5, 2),
        "sin": lambda: rng.uniform(-math.pi, math.pi) if rng.random() < .8 else rng.uniform(-100, 100),
        "cos": lambda: rng.uniform(-math.pi, math.pi) if rng.random() < .8 else rng.uniform(-100, 100),
    }
    ref = {"rsqrt": lambda x: 1 / math.sqrt(x), "rcp": lambda x: 1 / x, "sqrt": math.sqrt,
           "exp2": lambda x: 2.0 ** x, "log2": math.log2, "sin": math.sin, "cos": math.cos}
    f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
    xs = array.array("f", (f32(gen[o]()) for o in OPS for _ in range(N)))
    open("in.bin", "wb").write(xs.tobytes())
    expr = lambda o, ns: "%s::divide(1.0f, x)" % ns if o == "rcp" else "%s::%s(x)" % (ns, o)
    body = ["  { float x = in[%d*N+t]; out[(%d*2+%d)*N+t] = %s; }" % (i, i, j, expr(o, ns))
            for i, o in enumerate(OPS) for j, ns in enumerate(("fast", "precise"))]
    hdr = "#include <metal_stdlib>\nusing namespace metal;\n"
    open("k.metal", "w").write(hdr + "constant uint N=%d;\nkernel void k(device const float* in [[buffer(0)]], "
                               "device float* out [[buffer(1)]], uint t [[thread_position_in_grid]]) {\n%s\n}\n"
                               % (N, "\n".join(body)))
    fix = {"rsqrt": "x = x*0.5f+1.0f;", "rcp": "x = x+1.5f;", "sqrt": "x = x+1.0f;", "exp2": "x = x*0.01f;",
           "log2": "x = x+2.0f;", "sin": "", "cos": ""}
    perf = ["kernel void p_%s_%s(device float* out [[buffer(1)]], uint t [[thread_position_in_grid]]) {\n"
            "  float x = 1.0f + t * 1e-7f; for (int i = 0; i < 8192; i++) { x = %s; %s }\n  out[t] = x;\n}"
            % (o, ns, expr(o, ns), fix[o]) for o in OPS for ns in ("fast", "precise")]
    open("p.metal", "w").write(hdr + "\n".join(perf) + "\n")
    for k in ("k", "p"):
        subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-fno-fast-math",
                               k + ".metal", "-o", k + ".metallib"])
    json.dump([{"mode": "compute", "out": "out.bin", "source": "k.metallib", "function": "k", "threads": N,
                "dump": ["o"], "buffers": [{"name": "i", "file": "in.bin", "bind": [["compute", 0]]},
                                           {"name": "o", "size": N * len(OPS) * 8, "bind": [["compute", 1]]}]}],
              open("acc.json", "w"))
    subprocess.check_call([a.runner, "acc.json"], stdout=subprocess.DEVNULL)
    out = array.array("f"); out.frombytes(open("out.bin", "rb").read())
    lm = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    for fn in ("exp2f", "log2f", "sinf", "cosf", "sqrtf"):
        getattr(lm, fn).restype = ctypes.c_float
        getattr(lm, fn).argtypes = [ctypes.c_float]
    interp = {"rsqrt": lambda x: f32(1.0 / lm.sqrtf(x)), "rcp": lambda x: f32(1.0 / x), "sqrt": lm.sqrtf,
              "exp2": lm.exp2f, "log2": lm.log2f, "sin": lm.sinf, "cos": lm.cosf}

    def ordn(v):
        b = struct.unpack("<i", struct.pack("<f", v))[0]
        return b if b >= 0 else -(b & 0x7FFFFFFF)
    step = max(1, N >> 18)
    print("op      arm      max_ulp  exact%%  <=1ulp%%  <=2ulp%%   (vs correctly rounded, %d inputs)" % (N // step))
    for i, o in enumerate(OPS):
        for j, arm in enumerate(("fast", "precise", "interp")):
            h, mx = collections.Counter(), 0
            for t in range(0, N, step):
                x = xs[i * N + t]
                try:
                    r = f32(ref[o](x))
                except (OverflowError, ValueError):
                    continue
                if math.isinf(r) or (r != 0 and abs(r) < 1.1754944e-38):
                    continue
                g = out[(i * 2 + j) * N + t] if j < 2 else interp[o](x)
                u = abs(ordn(g) - ordn(r)) if not (math.isnan(g) or math.isnan(r)) else 1 << 31
                h[min(u, 3)] += 1
                mx = max(mx, u)
            tot = sum(h.values())
            print("%-7s %-8s %7d  %6.2f  %7.2f  %7.2f" % (o, arm, mx, 100 * h[0] / tot,
                  100 * (h[0] + h[1]) / tot, 100 * (h[0] + h[1] + h[2]) / tot))
    if a.no_perf:
        return
    print("op      fast_s  precise_s   (best of 3, includes ~0.1 s process start)")
    for o in OPS:
        row = []
        for ns in ("fast", "precise"):
            jp = "perf_%s_%s.json" % (o, ns)
            json.dump([{"mode": "compute", "out": "/dev/null", "source": "p.metallib", "function": "p_%s_%s" % (o, ns),
                        "threads": 1 << 24, "dump": [], "buffers": [{"name": "o", "size": (1 << 24) * 4,
                                                                     "bind": [["compute", 1]]}]}], open(jp, "w"))
            best = 1e9
            for _ in range(3):
                t0 = time.time()
                subprocess.check_call([a.runner, jp], stdout=subprocess.DEVNULL)
                best = min(best, time.time() - t0)
            row.append(best)
        print("%-7s %6.3f  %9.3f" % (o, row[0], row[1]))


if __name__ == "__main__":
    main()
