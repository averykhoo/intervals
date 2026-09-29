#!/usr/bin/env bash
# the local check before a push: the fuzz job of .github/workflows/fuzz.yml, run on this machine
# against the committed tree. HYPOTHESIS_PROFILE=fuzz runs every test (so it covers the gate) with
# each hypothesis test randomized at FUZZ_MULTIPLIER (default 10) times its examples, in two calls
# as the gate is (tests/itf1788, then the rest). the local fuzz database (.hypothesis/examples)
# replays anything a previous local run found. logs go to .scratch/prepush/<sha>/; the last line
# of each log is `rc=<n>`, and the script exits non-zero if either call was red.
#
#   tools/prepush.sh                 # x10, as CI
#   FUZZ_MULTIPLIER=1 tools/prepush.sh
#   PYTHON=/path/to/python tools/prepush.sh
#
# it refuses a dirty tree: the run must be of the commit that gets pushed.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
PYTHON=${PYTHON:-C:/Users/user/anaconda3/envs/intervals/python.exe}
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "prepush: the tree has uncommitted changes; commit first" >&2
    exit 2
fi
sha=$(git rev-parse --short HEAD)
out=.scratch/prepush/$sha
mkdir -p "$out"
export HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=${FUZZ_MULTIPLIER:-10}
echo "prepush: $sha, x$FUZZ_MULTIPLIER, logs in $out ($(date '+%Y-%m-%d %H:%M'))"
"$PYTHON" -m pytest -q -p no:cacheprovider tests/itf1788 > "$out/itf.log" 2>&1
itf=$?
echo "rc=$itf" >> "$out/itf.log"
"$PYTHON" -m pytest -q -p no:cacheprovider --ignore=tests/itf1788 > "$out/rest.log" 2>&1
rest=$?
echo "rc=$rest" >> "$out/rest.log"
for part in itf rest; do
    echo "prepush: $part: $(grep -E '(passed|failed|error)' "$out/$part.log" | tail -n 1) $(tail -n 1 "$out/$part.log")"
done
[ "$itf" -eq 0 ] && [ "$rest" -eq 0 ]
