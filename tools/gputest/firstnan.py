#!/usr/bin/env python3
"""Find the first native MSL line that makes a register NaN.

Usage (after a shaderdiff run with SHADERDIFF_KEEP_JOBS=1):
  firstnan.py WORK_DIR JOBS_DIR JOB_INDEX PIXEL RUNNER
WORK_DIR is <out>/work/<shader>, JOBS_DIR <out>/jobs/NNNNN, JOB_INDEX the
native job in jobs.json, PIXEL the linear rt0 pixel index, RUNNER the
rexgpu-metal-runner binary. Writes dbg.metal next to the shader and prints
the line number and the lines before it.
"""
# becomes non-finite; write (line, value) to c0; rerun the kept job.
import re, sys, json, subprocess, struct, math
W, J, job_idx, pix = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
src = open(W + '/ps.native.metal').read().split('\n')
out = []
inside = False
pat = re.compile(r'^\s+([a-zA-Z_]\w*)(\.[xyzw]+)? = ')
for i, l in enumerate(src, 1):
    out.append(l)
    if l.startswith('fragment XeOut xe_native_ps'):
        inside = True
        out.append('  float xe_dbg = 0.0f; float xe_dbgv = 0.0f;')
        continue
    if not inside: continue
    m = pat.match(l)
    if m and l.rstrip().endswith(';') and not l.lstrip().startswith(('float4 ', 'uint', 'int', 'bool', 'float ')):
        v = m.group(1)
        out.append('    if (xe_dbg == 0.0f && any(isnan(float4(%s)))) { xe_dbg = %d.0f; xe_dbgv = float4(%s).x; }' % (v, i, v))
    if l.strip().startswith('xe_out.c0 = xe_oc0;'):
        out.append('  xe_out.c0 = float4(xe_dbg, xe_dbgv, 0.0f, 0.0f);')
p = W + '/dbg.metal'
open(p, 'w').write('\n'.join(out))
r = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-std=metal3.1', '-fno-fast-math', '-Wno-unused-variable', p, '-o', p + '.metallib'], capture_output=True, text=True)
if r.returncode: print(r.stderr[:2000]); sys.exit(1)
jobs = json.load(open(J + '/jobs.json'))
j = jobs[job_idx]; j['fragment_source'] = p + '.metallib'; j['out'] = W + '/dbg.out'
json.dump([j], open(W + '/dbg.json', 'w'))
r = subprocess.run([sys.argv[5], W + '/dbg.json'], capture_output=True, text=True); print(r.stdout.strip()[-200:])
d = open(W + '/dbg.out', 'rb').read()
ln, v = struct.unpack_from('2f', d, 16 * pix)
print('pixel', pix, 'first non-finite line', int(ln), 'value', v)
if ln: print('\n'.join(src[int(ln) - 6:int(ln)]))
