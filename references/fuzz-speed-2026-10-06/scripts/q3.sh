#!/usr/bin/env bash
# Q3: 1 x (x10) versus 10 x (x1, independent seeds) on the same test node ids, in the private copy.
# usage: q3.sh <conc> <nodeid>...   conc = how many of the 10 x1 processes run at once
# every arm: fuzz profile, observability without coverage, its own HYPOTHESIS_STORAGE_DIRECTORY
set -u
cd /c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed/tree
PY=C:/Users/user/anaconda3/envs/intervals/python.exe
OUT=/c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed/q3
conc=$1; shift
mkdir -p "$OUT"
unset MULTIINTERVAL_BACKEND
export HYPOTHESIS_PROFILE=fuzz HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER=1
ms() { date +%s%N | cut -c1-13; }

one() {  # one pytest process: <label> <multiplier> [extra args]
  local label=$1 mult=$2; shift 2
  local s=$(ms)
  FUZZ_MULTIPLIER=$mult HYPOTHESIS_STORAGE_DIRECTORY="$(cygpath -w "$OUT/$label")" \
    $PY -m pytest -q -p no:cacheprovider --hypothesis-show-statistics --durations=0 "$@" > "$OUT/$label.log" 2>&1
  local rc=$?
  echo -e "$label\tmult=$mult\trc=$rc\t$(( $(ms) - s )) ms\t$(grep -E ' in [0-9.]+s' "$OUT/$label.log" | tail -1)" >> "$OUT/times.tsv"
}

echo "# $(date '+%F %T') start, conc=$conc, tests: $*" >> "$OUT/times.tsv"
# arm A: one process at x10
s=$(ms); one A_x10 10 "$@"; echo -e "ARM A (1 x x10, serial)\twall $(( $(ms) - s )) ms" >> "$OUT/times.tsv"
# arm B: ten processes at x1, conc at a time
s=$(ms)
pids=()
for i in $(seq 1 10); do
  one B_x1_$i 1 "$@" &
  pids+=($!)
  if (( ${#pids[@]} >= conc )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
echo -e "ARM B (10 x x1, $conc at once)\twall $(( $(ms) - s )) ms" >> "$OUT/times.tsv"
# arm C: xdist at x10, one worker per test
s=$(ms)
PYTHONPATH="$(cygpath -w /c/Users/user/PycharmProjects/intervals/.scratch/fuzz-speed/site)" one C_xdist_x10 10 -n $# --dist worksteal "$@"
echo -e "ARM C (xdist -n $# at x10)\twall $(( $(ms) - s )) ms" >> "$OUT/times.tsv"
echo "# $(date '+%F %T') end" >> "$OUT/times.tsv"
