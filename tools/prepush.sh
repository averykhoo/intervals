#!/usr/bin/env bash
# the local check before a push, decided by the run ledger (tools/gate.py): it runs only what the
# code about to be pushed has not already passed, and its exit status is the ledger's verdict.
#
#   src changed since origin/master, no green fuzz at x10 on this src: the fuzz job of
#     .github/workflows/fuzz.yml, run here (HYPOTHESIS_PROFILE=fuzz, x10) in two calls as the gate is
#     (fuzz-x10:itf, fuzz-x10:rest). every test runs, so it covers the gate. about 83 min here
#   only a README.md changed since that (or since origin/master): its doctests (the docs phase), seconds
#   only markdown or references/ changed (owner, 2026-09-30), or everything already green: nothing
#
# a fuzz run made before `git commit`, or by an earlier prepush on the same code, still counts: the
# ledger keys a verdict by the content, not the commit. `$PY tools/gate.py status` shows the same
# decision with its reasons, without running anything. logs: .gate-runs/ (gitignored).
#
#   tools/prepush.sh                 # x10, as CI
#   PREPUSH_DRY=1 tools/prepush.sh   # print the decision, run nothing
#   PREPUSH_FULL=1 tools/prepush.sh  # the fuzz run even if the ledger says it is not needed
#   FUZZ_MULTIPLIER=20 tools/prepush.sh   # above CI's x10 counts for a push; below it does not
#   PYTHON=/path/to/python tools/prepush.sh
#   PREPUSH_BASE=<ref> ...           # compare with <ref> instead of origin/master
#
# it refuses uncommitted changes (untracked files too) outside markdown and references/: the run
# must be of the commit that gets pushed.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
PYTHON=${PYTHON:-C:/Users/user/anaconda3/envs/intervals/python.exe}
n=${FUZZ_MULTIPLIER:-10}

base=${PREPUSH_BASE:-}
if [ -z "$base" ]; then
    git fetch -q origin master 2>/dev/null || echo "prepush: fetch failed, comparing with the last fetched origin/master" >&2
    base=origin/master
fi
plan=$("$PYTHON" tools/gate.py plan --base "$base")
rc=$?
[ "$rc" -eq 0 ] || { echo "prepush: $plan: commit first" >&2; exit 2; }
if [ "${PREPUSH_FULL:-}" = 1 ] && [ "$plan" != nothing ]; then
    plan=fuzz
fi
echo "prepush: $(git rev-parse --short HEAD) against $base: $plan"
[ "${PREPUSH_DRY:-}" = 1 ] && exit 0

case "$plan" in
    nothing) ;;
    docs) "$PYTHON" tools/gate.py run docs ;;
    fuzz)
        echo "prepush: x$n, $(date '+%Y-%m-%d %H:%M')"
        "$PYTHON" tools/gate.py run "fuzz-x$n:itf"
        "$PYTHON" tools/gate.py run "fuzz-x$n:rest"
        ;;
    *) echo "prepush: unexpected plan '$plan'" >&2; exit 2 ;;
esac
"$PYTHON" tools/gate.py status --require push --base "$base"
