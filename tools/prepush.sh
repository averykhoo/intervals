#!/usr/bin/env bash
# the local check before a push: the fuzz job of .github/workflows/fuzz.yml, run on this machine
# against the committed tree. HYPOTHESIS_PROFILE=fuzz runs every test (so it covers the gate) with
# each hypothesis test randomized at FUZZ_MULTIPLIER (default 10) times its examples, in two calls
# as the gate is (tests/itf1788, then the rest). the local fuzz database (.hypothesis/examples)
# replays anything a previous local run found. logs go to .scratch/prepush/<sha>/; the last line
# of each log is `rc=<n>`, and the script exits non-zero if either call was red.
#
# docs only (owner, 2026-09-30): when every file changed since origin/master is markdown (`*.md`) or
# under `references/`, the fuzz run cannot catch anything, so it is skipped; the changed README.md
# files are still run, since pytest collects them as doctests (pyproject.toml). any other file, even
# one with no code in it, takes the full run.
#
#   tools/prepush.sh                 # x10, as CI (or the docs-only check)
#   FUZZ_MULTIPLIER=1 tools/prepush.sh
#   PYTHON=/path/to/python tools/prepush.sh
#   PREPUSH_FULL=1 tools/prepush.sh  # the full run even for docs
#   PREPUSH_DRY=1 tools/prepush.sh   # print the decision and the changed files, run nothing
#   PREPUSH_BASE=<ref> ...           # compare with <ref> instead of origin/master (for checking the script)
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

base=${PREPUSH_BASE:-}
if [ -z "$base" ]; then
    git fetch -q origin master 2>/dev/null || echo "prepush: fetch failed, comparing with the last fetched origin/master" >&2
    base=origin/master
fi
changed=$(git diff --name-only "$base...HEAD") || exit 2
docs_only=yes
[ -n "$changed" ] || docs_only=empty
while IFS= read -r path; do
    [ -n "$path" ] || continue
    case "$path" in
        *.md | references/*) ;;
        *) docs_only=no ;;
    esac
done <<< "$changed"
[ "${PREPUSH_FULL:-}" = 1 ] && [ "$docs_only" != empty ] && docs_only=no
echo "prepush: $sha against $base: $(printf '%s\n' "$changed" | grep -c .) file(s) changed, docs only: $docs_only"
if [ "${PREPUSH_DRY:-}" = 1 ]; then
    printf '%s\n' "$changed" | sed 's/^/  /'
    exit 0
fi
if [ "$docs_only" = empty ]; then
    echo "prepush: nothing to push"
    exit 0
fi
if [ "$docs_only" = yes ]; then
    readmes=$(printf '%s\n' "$changed" | grep -E '(^|/)README\.md$' || true)
    if [ -z "$readmes" ]; then
        echo "prepush: docs only, no README.md changed: nothing to run"
        exit 0
    fi
    # shellcheck disable=SC2086
    "$PYTHON" -m pytest -q -p no:cacheprovider $readmes
    rc=$?
    echo "prepush: docs only, the README doctests: rc=$rc"
    exit $rc
fi

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
