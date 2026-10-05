---
name: testing
description: How to run, read and extend this repo's tests: the run ledger of what has run on the current code (tools/gate.py), the gate, the fuzz profile, the pre-push run (tools/prepush.sh), watching CI after a push (tools/ci_watch.sh), reproducing and pinning a hypothesis failure from CI, sabotage checks that prove a test can fail, the exhaustive differential harnesses, the itf1788 census and the gmpy2 backend runs. Use before running any test here, when a test or a fuzz run is red, when adding a property or a pin, and around every push.
---

# testing in this repo

the rules (when each run is required, who may push) are in `CLAUDE.md`; this is how to do each one.
python is always `C:/Users/user/anaconda3/envs/intervals/python.exe` (written `$PY` below), from the
repo root. timings are from this shared laptop under load, 2026-09-30.

## which run, when

| moment | run | time |
|---|---|---|
| session start, before a commit or push | `$PY tools/gate.py status` | under a second |
| while editing | the touched test file(s): `$PY -m pytest -q tests/test_x.py` | seconds to minutes |
| before a commit | the gate, in two recorded calls (below) | ~1 min + ~10-14 min |
| before a push | `bash tools/prepush.sh`, in the background | ~83 min; docs only: seconds; + ~10-15 min if a backend file changed |
| after a push | a babysitter agent on `bash tools/ci_watch.sh <sha>` | ci ~12 min, fuzz ~30-56 min |
| CI only | the exhaustive harnesses (below) | 20 s to 12 min each on CI |

## the gate

`$PY -m pytest -q` is the whole gate, but past the 10-minute tool limit here, so two calls, each
through the ledger (it captures pytest's rc itself, so no pipe can hide it):

    $PY tools/gate.py run gate:itf      # ~1 min
    $PY tools/gate.py run gate:rest     # ~10-14 min: in the background

then `$PY tools/gate.py status --require commit` (exit 0: commit). a bare `$PY -m pytest` is fine
while iterating but is not recorded, so it never counts as the gate.

the gate also collects `README.md` and `tests/itf1788/README.md` as doctests and every module's
docstrings (`pyproject.toml`), so a prose edit to a README can break it. 33710 items (2026-10-02 at `233fdd4`, CI and local).
the library's warnings are errors inside the suite: a test that provokes one says so with
`pytest.warns` or a `filterwarnings` mark.

## the fuzz profile

`HYPOTHESIS_PROFILE=fuzz` (`tests/conftest.py`) randomizes every hypothesis test at
`FUZZ_MULTIPLIER` (default 10) times its own `max_examples`, no deadline, with an example database
in `.hypothesis/examples` (on CI too, since 2026-09-29: `tests/test_fuzz_profile.py`). one file:

    HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 $PY -m pytest -q -p no:cacheprovider tests/test_reverse.py

the local database replays what earlier local runs found; delete `.hypothesis/` only on purpose.

## the run ledger: tools/gate.py

every recorded run appends a row to `.gate-runs/ledger.tsv` (gitignored) and keeps its whole output
beside it. a row names the code by two content ids, not by the commit:

* `s:` (src): every file but `*.md` and `references/`, the paths fuzz.yml skips. a fuzz verdict is
  keyed by it.
* `c:` (code): src plus the `README.md` files pytest runs as doctests. gate and docs verdicts are
  keyed by it.

so a run made before `git commit` still counts after it; a HANDOFF or plan edit stales nothing; a
README edit stales only the doctests (`$PY tools/gate.py run docs`, seconds). phases: `gate:itf`,
`gate:rest`, `docs`, `fuzz-x<N>:itf`, `fuzz-x<N>:rest` (only N >= 10, fuzz.yml's, clears a push),
`gate:gmpy2` (below).

    $PY tools/gate.py status                   # each phase on this code, the commit and push verdicts
    $PY tools/gate.py status --require commit  # exit 1 unless the gate is green on this code
    $PY tools/gate.py status --require push    # exit 1 unless a push needs nothing more
    $PY tools/gate.py plan                     # nothing / dirty, or docs|fuzz and gmpy2 joined by +: what prepush reads

statuses: PASSED (rc 0 and a passing pytest summary), FAILED, INCONSISTENT (rc 0 without one),
MOVED (the code changed while it ran: it counts for nothing; do not edit sources during a run),
INTERRUPTED. a log with no row is a run killed or still running. every phase but `gate:gmpy2` removes
`MULTIINTERVAL_BACKEND` (CI's gate and fuzz run the pure path); all set the fuzz variables from the phase name.

`gate:gmpy2` (owner, Q16(e), 2026-10-03) is the whole suite, one call, with `MULTIINTERVAL_BACKEND=gmpy2`
set whatever your environment says: what ci.yml's `gate-gmpy2` job runs (python 3.13, ubuntu, the
PyPI wheel's MPFR). forced, `import multiinterval` raises without gmpy2, so the suite cannot pass on the
pure path (one test file alone can: a failed package import leaves leaf modules importable; the CI
job asserts `backend.name()` first). it is keyed by src like the fuzz, never covers the gate (a commit
needs the pure path), and a push needs it only when `tools/gate.py::BACKEND_FILES` (`backend`,
`_gmpy2`, `elementary`, `ops`) changed since the base. about 10-15 min: in the background. there is no
gmpy2 fuzz job.
the ids do not see `.hypothesis/` or the installed packages (recorded in each row, not matched).
`tests/test_gate_ledger.py` pins all of it; its sabotage table (13 breaks, each red) is in
`v2-implementation-plan.md` §2 "run ledger".

## before a push: tools/prepush.sh

refuses uncommitted changes outside markdown and `references/` (untracked files too); compares with
`origin/master` (fetched); runs what `tools/gate.py plan` says is missing: the fuzz at x10 if the
src changed and no green x10 run on it is recorded, the docs phase if only READMEs changed since,
nothing if only prose did; and besides, `gate:gmpy2` if a backend file changed and no green run of it
on this src is recorded. logs in `.gate-runs/`. its exit is `tools/gate.py status --require push`.
`PREPUSH_DRY=1` prints the decision, `PREPUSH_FULL=1` forces the fuzz run. push only on exit 0.

## after a push: tools/ci_watch.sh

    bash tools/ci_watch.sh <sha> [n]   # n runs to wait for: 2 on master (ci, fuzz)

waits until every run of the commit is done, prints one line each, exits 1 if any was red, and for a
red run saves `failed.log` and the artifact (`fuzz.log`, `.hypothesis/`) to `.scratch/ci/<run id>/`,
printing the failing tests and falsifying examples. it reads the GitHub token from the git credential
store each call. give it to a babysitter agent in the background; have the agent write notes as it
goes and never push.

## a red fuzz run: reproduce, decide, pin

1. **reproduce** in a throwaway worktree at the pushed commit (`git worktree add --detach
   ../intervals-repro <sha>`), two ways:
   * the database: `rm -rf .hypothesis` in the worktree FIRST, then `cp -r <artifact>/.hypothesis
     .hypothesis`, then run the failing test unchanged; it fails on the saved example. copying onto an
     existing `.hypothesis/` nests it as `.hypothesis/.hypothesis` and the test passes as if nothing
     were saved (this happened, 2026-09-29). run a control without the database too
   * the example: paste the log's "Falsifying example" as an `@example`. an `st.randoms()` argument
     prints as `HypothesisRandom(...)`: give it `random.Random(0)` and check it still fails
   * `@reproduce_failure` works only with CI's exact hypothesis version (CI installs the newest; this
     laptop had 6.167.1 when CI had 6.168.3). to replay its choices under the local version: decode
     the database entry with `hypothesis.database.choices_from_bytes`, then set
     `test._hypothesis_internal_use_reproduce_failure = (hypothesis.__version__,
     hypothesis.core.encode_failure(choices))` and call the test
2. **decide** library or test oracle, with the exact values (what the exact class, the outward class
   and the nearest class each return; what python itself does). a babysitter's root cause is a lead,
   not a finding: check that its fix would make the failing assertion pass.
3. **pin**: keep the `@example` on the test that found it, then show it red on the old code and green
   with the fix (sabotage, below). a decision about semantics is the owner's (fuzz-rev-inf became D26).
4. **record**: a subsection in `v2-implementation-plan.md` §2 like "fuzz-symmetry",
   "fuzz-floordiv-overflow", "fuzz-rev-inf" (found, diagnosis, fix and pins), and `HANDOFF.md`.

## sabotage: a check that cannot fail proves nothing

* for a pin or a new check, break what it guards and watch it go red. three ways:
  * edit the source: `tools/sabotage.py` (below). do not hand-roll the loop: a same-size break restored
    within the same second left python running the broken `.pyc` (M16c), and a hand loop in the live
    tree breaks files another agent is reading
  * in process: replace the function with a no-op (`module.fn = lambda ...: ...`) and call the test's
    inner function on each example, `test.hypothesis.inner_test(*args)`, then restore it
  * against an old commit: `git archive <rev> intervals | tar -x -C .scratch/<name>`, the test file
    beside it, and an empty `pytest.ini` there, run with `-c pytest.ini`. without it pytest finds the
    repo's `pyproject.toml`, whose `pythonpath = ["."]` imports the live package, and the old code is
    never run: the check passes vacuously (2026-10-04; print `multiinterval.__file__` from a conftest)
* check each example on its own: hypothesis stops at the first failing explicit example, so one red
  run says nothing about the others
* watch for vacuous checks: a "not empty" assertion passed under sabotage when other pieces kept the
  result non-empty (the trig D26 example); a per-piece check was needed
* never stop another process by name or command line: two sessions' harnesses have shared a name. stop
  the PID you started

### tools/sabotage.py: a break table on a private copy

    $PY tools/sabotage.py run TABLE --name NAME              # breaks a `git archive HEAD` copy
    $PY tools/sabotage.py run TABLE --name NAME --ref REV    # another commit
    $PY tools/sabotage.py run TABLE --name NAME --worktree   # uncommitted work (tracked + untracked, not ignored)
    $PY tools/sabotage.py run TABLE --name NAME --dry-run    # each row's match count, no pytest
    $PY tools/sabotage.py stop NAME                          # kills that run's recorded PID tree, removes its copy

the table is TOML (or `.json`); keep it in `.scratch/`, it is per task:

    select = ["tests/test_fmt.py"]        # pytest args: node ids, files, "-k", "expr"
    timeout = 600                          # seconds per pytest run
    [[break]]
    id = "fmt-swap-bracket"
    target = "multiinterval/fmt.py"
    old = '''{"]" if hi_closed else ")"}'''   # must occur exactly once (overlaps counted)
    new = '''{")" if hi_closed else "]"}'''
    # optional: select = [...], timeout = 60, expect = "green" (a placebo the selection must not see)

it snapshots into `.scratch/sabotage/<name>/tree`, never the live tree; runs the selection on the intact
copy first (control) and last (closing control); per row writes the break, clears `__pycache__` and
`.hypothesis`, runs `pytest -x` with `PYTHONDONTWRITEBYTECODE=1`, restores with `copy2` and checks the
bytes. a guard plugin in every run fails it if a module came from outside the copy. one line per row in
`<name>/verdicts.tsv` as it finishes, pytest's output in `<name>/logs/<id>.log`:

* RED caught. TIMEOUT caught (hung; its process tree killed). ERROR caught weakly: a collection or
  import error, a crash; it proves nothing about the check
* GREEN survived: exit 1 (unless `expect = "green"`)
* NOMATCH / AMBIGUOUS: `old` found 0 / 2+ times; the row did not run: exit 1. LEAK: imported from
  outside the copy: exit 1
* exit 2: the run proves nothing (control or closing control red, a restore that did not verify, a
  live run of that name, a bad table)

`--name` is yours: it refuses to start while that name's PID is alive, and `stop` checks the PID's
creation time, so a reused PID is never killed. `tests/test_sabotage_tool.py` pins the engine (toy
repos, ~25 s); its docstring holds the engine's own sabotage table, run through the engine itself

## the exhaustive harnesses (CI's `exhaustive` jobs, not the gate)

    $PY -m tests.exhaustive_ops            # every 1-2 piece set over a 9-point grid; ~265k checks
    $PY -m tests.exhaustive_ops --float    # the grid as floats; ~178k
    $PY -m tests.exhaustive_ops --sabotage # breaks the applicator 3 ways; each count must not be 0
    $PY -m tests.exhaustive_modulo

each docstring has its measured run time: the ops runs 11-18 minutes (2026-09-25), modulo about 105k
boxes at a few minutes per 10k. `--sample N` checks N random cases. run them in the background, and
only when `ops`, the applicator or `modulo` changed.

## CORE-MATH worst cases: tools/coremath.py (manual, local only)

    $PY tools/coremath.py status             # pin vs upstream, last full check, scalar changes since
    $PY tools/coremath.py check --jobs 4     # every input of the pinned files, DOWN/NEAREST/UP vs MPFR
    $PY tools/coremath.py sample --check     # the vendored rows are what the files and MPFR give
    $PY tools/coremath.py pin [<commit>]     # move to a newer CORE-MATH, then `sample` and a full check

the gate's part is `tests/test_coremath.py`: about 29k vendored rows (`tests/coremath/*.tsv`, every row
of a small block, 200 seeded rows of a big one), 12 s. it catches a Ziv loop that stops or computes
wrongly past its first precision (sabotaged 2026-10-02: 24 of 24 functions red), not a slightly loose
error bound, which the loop absorbs (`references/test-vector-sources.md` §3h). `check` is never in CI
or prepush: at the end of a session that changed the scalar evaluator, ask the owner whether to run
it (`CLAUDE.md`; 2547 s at 4 jobs, 2026-10-02). it appends its verdict to `references/coremath-runs.tsv`;
commit that row. the files live in `.scratch/coremath-cache/` (kept; `fetch` restores it).

## other tools

* `$PY tools/itf1788_census.py`: the itf1788 counts quoted in the docs (vectors, ops, divergence keys
  by category), read from the adapter, so a count in a doc can be regenerated rather than copied
* `MULTIINTERVAL_BACKEND=gmpy2 $PY -m pytest -q ...`: a file on the gmpy2/MPFR backend (opt-in; the whole
  suite is the `gate:gmpy2` phase above and ci.yml's `gate-gmpy2` job; every pure job also runs
  `tests/test_backend.py`'s differential); `$PY tools/backend_speed.py [--bound]` for its speed
* `tests/itf1788/`: the vendored ITF1788 vectors (unmodified; `tests/itf1788/README.md`), the parser
  `itl.py` and the adapter `test_itf1788.py`, whose docstring explains how a vector is compared; a new
  difference from 1788 is a row under a named category, never a skip

## shared-laptop hygiene

long runs go to the background; a `tail -f` monitor outlives its session, so stop it by PID when
done; `.scratch/` is gitignored and lost, so a result worth keeping goes into a tracked file the same
hour.
