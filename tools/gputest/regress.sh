#!/usr/bin/env bash
# Fast GPU regression: texture cache stress test, shader diff on a curated
# subset, static checks on the same subset, and trace replays. Minutes, not
# hours. Prints PASS/FAIL per stage and exits nonzero if any stage fails.
#
#   tools/gputest/regress.sh CONFIG [--only stress,shaders,static,replay,<extra>] [--out DIR]
#
# CONFIG is a bash file the game supplies (bo2: ~/dev/bo2-mac-port/patches/gputest/
# regress-bo2.conf). It may set:
#   UCODE        dir of *.ucode.bin.vert / *.ucode.bin.frag
#   SHADER_LIST  file of shader names (one per line, '#' comments)
#   KNOWN        file of known findings "shader class" (see regress_check.py)
#   MSL_XLATE    rexgpu-metal-msl-xlate binary of the Metal plugin under test
#   MASK_KNOWN   shaderdiff --mask-known value
#   STRESS       array: command line(s) of the texture cache stress test
#   REPLAY       array: command line(s), each exits nonzero on a divergence
#   OUT          output dir (default /tmp/gputest-regress)
#   JOBS         parallel jobs (default 4)
#   EXTRA        array of "name|command" stages owned by other harnesses; each
#                passes when its command exits 0 (log in OUT/<name>.log)
# Drop-in files CONFIG.d/*.conf (for bo2: regress-bo2.conf.d/) are sourced after
# CONFIG, in name order; other harnesses append their stages to EXTRA there so
# that nobody edits this script or the main config to add one.
# A stage with nothing configured is reported as SKIP, not PASS.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
[ $# -ge 1 ] || { sed -n 2,26p "$0"; exit 2; }
conf="$1"; shift
only=""
# Environment values win over the config when the config uses ${VAR:-default}.
UCODE="${UCODE:-}"; SHADER_LIST="${SHADER_LIST:-}"; KNOWN="${KNOWN:-}"; MSL_XLATE="${MSL_XLATE:-}"
MASK_KNOWN="${MASK_KNOWN:-}"; OUT="${OUT:-}"; JOBS="${JOBS:-4}"; STRESS=(); REPLAY=()
# shellcheck disable=SC1090
source "$conf"
if [ -d "$conf.d" ]; then
  for f in "$conf.d"/*.conf; do
    # shellcheck disable=SC1090
    [ -e "$f" ] && source "$f"
  done
fi
while [ $# -gt 0 ]; do
  case "$1" in
    --only) only="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
if [ -z "$only" ]; then
  only="stress,shaders,static,replay"
  for e in "${EXTRA[@]+"${EXTRA[@]}"}"; do only="$only,${e%%|*}"; done
fi
BIN="${BIN:-$root/out/mac-arm64/Release}"
OUT="${OUT:-/tmp/gputest-regress}"
mkdir -p "$OUT"
declare -a summary=()
fail=0
t0=$SECONDS
want() { [[ ",$only," == *",$1,"* ]]; }
record() {  # stage status seconds detail
  summary+=("$(printf '%-8s %-4s %4ss  %s' "$1" "$2" "$3" "$4")")
  [ "$2" = FAIL ] && fail=1
  return 0
}

if want stress; then
  s=$SECONDS
  if [ ${#STRESS[@]} -eq 0 ]; then record stress SKIP 0 "no STRESS command"
  else
    st=PASS; i=0
    for cmd in "${STRESS[@]}"; do
      i=$((i+1))
      if ! bash -c "$cmd" > "$OUT/stress.$i.log" 2>&1; then st=FAIL; fi
    done
    record stress $st $((SECONDS-s)) "$OUT/stress.*.log"
  fi
fi

subset=""
if { want shaders || want static; } && [ -n "$UCODE" ] && [ -n "$SHADER_LIST" ]; then
  subset="$OUT/ucode"
  rm -rf "$subset"; mkdir -p "$subset"
  missing=0
  while read -r name _; do
    [ -z "$name" ] || [[ "$name" == \#* ]] && continue
    found=0
    for f in "$UCODE/$name".ucode.bin.*; do
      [ -e "$f" ] && { ln -sf "$f" "$subset/"; found=1; }
    done
    [ $found = 1 ] || missing=$((missing+1))
  done < "$SHADER_LIST"
  [ $missing = 0 ] || echo "warning: $missing listed shaders not in $UCODE" >&2
fi

xl=()
[ -n "$MSL_XLATE" ] && xl=(--msl-xlate "$MSL_XLATE")
kn=()
[ -n "$KNOWN" ] && kn=(--known "$KNOWN")

if want shaders; then
  s=$SECONDS
  if [ -z "$subset" ]; then record shaders SKIP 0 "no UCODE/SHADER_LIST"
  else
    rm -rf "$OUT/shaderdiff"
    mk=()
    [ -n "$MASK_KNOWN" ] && mk=(--mask-known "$MASK_KNOWN")
    python3 "$here/shaderdiff.py" --ucode "$subset" --out "$OUT/shaderdiff" --seeds 1 \
      --bin "$BIN" "${xl[@]}" "${mk[@]}" -j "$JOBS" > "$OUT/shaderdiff.log" 2>&1
    if python3 "$here/regress_check.py" "$OUT/shaderdiff/results.jsonl" "${kn[@]}" \
        > "$OUT/shaders.check" 2>&1; then st=PASS; else st=FAIL; fi
    record shaders $st $((SECONDS-s)) "$(tail -1 "$OUT/shaders.check")"
  fi
fi

if want static; then
  s=$SECONDS
  if [ -z "$subset" ]; then record static SKIP 0 "no UCODE/SHADER_LIST"
  else
    rm -rf "$OUT/static"
    if python3 "$here/static_checks.py" --ucode "$subset" --out "$OUT/static" \
        --bin "$BIN" "${xl[@]}" > "$OUT/static.log" 2>&1; then st=PASS; else st=FAIL; fi
    record static $st $((SECONDS-s)) "$(grep -m1 -E "inputs$" "$OUT/static/static.md" 2>/dev/null || tail -1 "$OUT/static.log")"
  fi
fi

if want replay; then
  s=$SECONDS
  if [ ${#REPLAY[@]} -eq 0 ]; then record replay SKIP 0 "no REPLAY command"
  else
    st=PASS; i=0
    for cmd in "${REPLAY[@]}"; do
      i=$((i+1))
      if ! bash -c "$cmd" > "$OUT/replay.$i.log" 2>&1; then
        st=FAIL; grep -m3 -iE "stale|wrong|diverg|FAIL" "$OUT/replay.$i.log" | sed 's/^/    /'
      fi
    done
    record replay $st $((SECONDS-s)) "$OUT/replay.*.log"
  fi
fi

for e in "${EXTRA[@]+"${EXTRA[@]}"}"; do
  name="${e%%|*}"; cmd="${e#*|}"
  want "$name" || continue
  s=$SECONDS
  if bash -c "$cmd" > "$OUT/$name.log" 2>&1; then st=PASS; else st=FAIL; fi
  record "$name" $st $((SECONDS-s)) "$(tail -1 "$OUT/$name.log" | cut -c1-120)"
done

echo "gputest regress ($((SECONDS-t0))s, logs in $OUT)"
for l in "${summary[@]}"; do echo "  $l"; done
[ -f "$OUT/shaders.check" ] && grep -E "^NEW|^now passing" "$OUT/shaders.check" | head -20 | sed 's/^/  /'
if [ $fail = 0 ]; then echo "RESULT PASS"; else echo "RESULT FAIL"; fi
exit $fail
