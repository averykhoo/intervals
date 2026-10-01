---
name: testing
description: How to run, read and extend this repo's tests: the gate, the fuzz profile, the pre-push run (tools/prepush.sh), watching CI after a push (tools/ci_watch.sh), reproducing and pinning a hypothesis failure from CI, sabotage checks that prove a test can fail, the exhaustive differential harnesses, the itf1788 census and the gmpy2 backend runs. Use before running any test here, when a test or a fuzz run is red, when adding a property or a pin, and around every push.
---

# testing in this repo

the rules (when each run is required, who may push) are in `CLAUDE.md`; this is how to do each one.
python is always `C:/Users/user/anaconda3/envs/intervals/python.exe` (written `$PY` below), from the
repo root. timings are from this shared laptop under load, 2026-09-30.

## which run, when

| moment | run | time |
|---|---|---|
| while editing | the touched test file(s): `$PY -m pytest -q tests/test_x.py` | seconds to minutes |
| before a commit | the gate, in two calls (below) | ~1 min + ~10-14 min |
| before a push | `bash tools/prepush.sh`, in the background | ~83 min; docs only: seconds |
| after a push | a babysitter agent on `bash tools/ci_watch.sh <sha>` | ci ~12 min, fuzz ~30-56 min |
| CI only | the exhaustive harnesses (below) | 20 s to 12 min each on CI |

## the gate

`$PY -m pytest -q` is the whole gate, but past the 10-minute tool limit here, so two calls, each with
its rc captured into the log (a pipe through `tail` reports the pipe's status, not pytest's):

    $PY -m pytest -q -p no:cacheprovider tests/itf1788 > .scratch/gate/itf.log 2>&1; echo "rc=$?" >> .scratch/gate/itf.log
    $PY -m pytest -q -p no:cacheprovider --ignore=tests/itf1788 > .scratch/gate/rest.log 2>&1; echo "rc=$?" >> .scratch/gate/rest.log

the gate also collects `README.md` and `tests/itf1788/README.md` as doctests and every module's
docstrings (`pyproject.toml`), so a prose edit to a README can break it. 33406 items (2026-09-30).
the library's warnings are errors inside the suite: a test that provokes one says so with
`pytest.warns` or a `filterwarnings` mark.

## the fuzz profile

`HYPOTHESIS_PROFILE=fuzz` (`tests/conftest.py`) randomizes every hypothesis test at
`FUZZ_MULTIPLIER` (default 10) times its own `max_examples`, no deadline, with an example database
in `.hypothesis/examples` (on CI too, since 2026-09-29: `tests/test_fuzz_profile.py`). one file:

    HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 $PY -m pytest -q -p no:cacheprovider tests/test_reverse.py

the local database replays what earlier local runs found; delete `.hypothesis/` only on purpose.

## before a push: tools/prepush.sh

refuses an uncommitted tree; compares with `origin/master` (fetched). if every changed file is `*.md`
or under `references/`, it runs only the changed READMEs' doctests; otherwise the whole suite under
the fuzz profile at x10, logs in `.scratch/prepush/<sha>/`. `PREPUSH_DRY=1` prints the decision,
`PREPUSH_FULL=1` forces the full run. push only on exit 0.

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

* for a pin or a new check, break what it guards and watch it go red. two ways:
  * edit the source: copy it aside, replace one exact string that must match once, clear
    `__pycache__`, run with `PYTHONDONTWRITEBYTECODE=1`, restore with `cp`, confirm with `cmp`. a
    same-size break restored within the same second can leave python running the broken `.pyc`
  * in process: replace the function with a no-op (`module.fn = lambda ...: ...`) and call the test's
    inner function on each example, `test.hypothesis.inner_test(*args)`, then restore it
* check each example on its own: hypothesis stops at the first failing explicit example, so one red
  run says nothing about the others
* watch for vacuous checks: a "not empty" assertion passed under sabotage when other pieces kept the
  result non-empty (the trig D26 example); a per-piece check was needed
* never stop another process by name or command line: two sessions' harnesses have shared a name. stop
  the PID you started

## the exhaustive harnesses (CI's `exhaustive` jobs, not the gate)

    $PY -m tests.exhaustive_ops            # every 1-2 piece set over a 9-point grid; ~265k checks
    $PY -m tests.exhaustive_ops --float    # the grid as floats; ~178k
    $PY -m tests.exhaustive_ops --sabotage # breaks the applicator 3 ways; each count must not be 0
    $PY -m tests.exhaustive_modulo

each docstring has its measured run time: the ops runs 11-18 minutes (2026-09-25), modulo about 105k
boxes at a few minutes per 10k. `--sample N` checks N random cases. run them in the background, and
only when `ops`, the applicator or `modulo` changed.

## other tools

* `$PY tools/itf1788_census.py`: the itf1788 counts quoted in the docs (vectors, ops, divergence keys
  by category), read from the adapter, so a count in a doc can be regenerated rather than copied
* `INTERVALS_BACKEND=gmpy2 $PY -m pytest -q ...`: the suite on the gmpy2/MPFR backend (opt-in; CI runs
  only `tests/test_backend.py`'s differential); `$PY tools/backend_speed.py [--bound]` for its speed
* `tests/itf1788/`: the vendored ITF1788 vectors (unmodified; `tests/itf1788/README.md`), the parser
  `itl.py` and the adapter `test_itf1788.py`, whose docstring explains how a vector is compared; a new
  difference from 1788 is a row under a named category, never a skip

## shared-laptop hygiene

long runs go to the background; a `tail -f` monitor outlives its session, so stop it by PID when
done; `.scratch/` is gitignored and lost, so a result worth keeping goes into a tracked file the same
hour.
