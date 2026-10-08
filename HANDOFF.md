# handoff

what is true now: the banner, ranked open items, open questions for the owner, what is still owed, and the
newest session-log entries. volatile by design. elsewhere: every decision in `docs/decisions.md` (the one
decisions log; never here), records of finished work in `docs/records.md`, the v2 plans (design, milestones,
every record up to v2) archived in `docs/archive/v2/`, older session-log entries in `docs/session-log.md`.
a task the owner assigns overrides the ranking. at the end of a session:

* add a session-log entry (newest first) and refresh the banner. the banner is the current state only, a few
  bullets, never history: what happened goes in the session log
* edit rows in place and sweep closed ids: a finished item leaves the table, its record goes in
  `docs/records.md`, a one-line entry in the session log; nothing is listed as open and done at once
* an answered question leaves "open questions": the answer goes in `docs/decisions.md`, not here
* **rotate the session log**: keep the newest 8 entries here; move the older ones, unchanged, to the top of
  `docs/session-log.md`'s list
* list anything skipped as "Still owed:"

## banner (2026-10-08)

* **v2 is built** (a read-only audit, 2026-10-08: M1-M16 and every named item after them done, each record
  checked and a code symbol of each spot-checked). its plans are archived in `docs/archive/v2/`, its decisions
  in `docs/decisions.md`. what is left before 2.0.0 is row H1 and questions Q25-Q26
* **pushed: `origin/master` at `1330182`**, CI and fuzz green (2026-10-06). `master` is ahead by `a0a8939`..HEAD,
  not pushed: the 1788 survey, `tools/pyintval_check.py` (src), the fuzz-speed study, this archive. the push
  needs the x10 prepush (src changed) and the owner's go
* **gate** green on this code: 27798 + 7143 passed (2026-10-08, `tools/gate.py status`)
* the CORE-MATH full check last ran 2026-10-02 at `3044197`; 5 scalar-evaluator commits since (`tools/coremath.py
  status`, which until 2026-10-08 hid the ones before the rename and said 1). the owner's call, worth running
  before 2.0

## open items (ranked)

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | fuzz-xdist | run the x10 fuzz (and `gate:gmpy2`) under pytest-xdist: 21 min at `-n 8` against 75-125 min serial, the same 34941 passed (2026-10-06). needs `phase_spec` to pass `-n K --dist worksteal`, `workers=K` recorded per row, `tests/test_gate_ledger.py`'s argument pins, pytest-xdist in `[test]` and the env, and a look at the four wall-clock tests (one failed under load at `-n 4`) | ready; the study is done, the build is not | `references/fuzz-speed-2026-10-06/README.md` |
| 2 | m14b-open | what M14-breadth found and left (2026-10-02): number-type quirks with no wrong value (a 0 end exact among float operands: `abs(M(-1.0, 1.0))` is `[0, 1.0]`, still so 2026-10-08; trunc's non-negative side ints; a one-point domain clip takes its low cut's type) | ready, small; it changes result types, so cheaper before 2.0 (Q26) | `docs/archive/v2/v2-implementation-plan.md` §2 "M14-breadth" (left open) |
| 3 | vectors-ext | (c) only: cuinterval's `custom.itl` (26 vectors, MIT), probably covered by `test_domain_ends_and_limits`. (a) done 2026-10-05, (b) closed 2026-10-03 | low value | `references/test-vector-sources.md` |
| 4 | pown-ziv | an exact corner of about 2M bits within about 2 ** -(its size) of a rounding breakpoint, past `EXACT_RESULT_LIMIT`, runs ziv past 120 s where the old code built the power in milliseconds (`O(3 + 2 ** -2100000) ** 2`, 2026-10-04); pow the same at 2M bits. follow-ups: a near-1 shortcut in `rounded_pow`, or the exact build when ziv passes a precision cap and the build is affordable | ready, not scheduled; extreme sizes only | `docs/archive/v2/v2-implementation-plan.md` §2 "owner-answers"; `references/owner-questions-2026-10-03/streams/pown.md` step 6 |
| 5 | evaluate-box | a pure speed change: `applicator.evaluate_box` evaluates a float corner's exact value three times under `OUTWARD`; passing `fn`'s value into the hook would cut it to one, maybe worth as much for arithmetic as the backend, with no dependency | idea, not scheduled (M16e, 2026-09-28) | `docs/archive/v2/v2-implementation-plan.md` §2 M16e |
| 6 | later | not in v2.0, each "consider", optional or "if asked". from the owner's answers (2026-10-03): `Root`/`RootBox` as frozen dataclasses if a third state appears; an `rtol` beside `tol`; outward fma, `%`, hypot, `cancel_minus` typed per corner (tighter); a to-nearest `MultiInterval.rounded()`; rootn on cbrt's worst-case inputs; a numpy hook for the 1788 layer; an `AllenMatrix` class or a public `allen_pairs`; a strategies module after 2.0; Q20's optional pin. from the design's "later": a direction tag on a degenerate zero piece (if a solver needs `1/(1/[inf]) == [inf]`); an interval array type (the array API); the backend's non-dyadic part (an mpfr ziv loop for the points it declines, until a workload measures them); allen's composition table for the cut reading; vector-mode autodiff (if a measured solve is too slow); 1788's recommended ops not in the layer (`exp2m1`, `exp10m1`, `log2p1`, `log10p1`, `compoundm1`, `rsqrt`, the `*Pi` functions, text and interchange conversions, inf-sup types but binary64). from M8: pandas past `to_pandas()` (an `IntervalIndex` of several pieces, `IntervalArray`) | not scheduled | `references/owner-questions-2026-10-03/`; `docs/archive/v2/v2-plan.md` "later (not in v2.0)"; `docs/archive/v2/v2-implementation-plan.md` §2 M8 |
| 7 | H1 | release 2.0.0 (`pyproject.toml` is `2.0.0.dev0`, no tag). before it: Q25 (licence and package metadata: no LICENSE, no `license`/`readme`/`authors`/`urls`/classifiers in `pyproject.toml`; no publish workflow, no changelog), Q26 (m14b-open first?), release notes (from the v1 -> v2 surface map, `docs/archive/v2/v2-implementation-plan.md` §4), the CORE-MATH full check, claiming `multiinterval` on PyPI (the first upload) | waiting on Q25, Q26 (owner, 2026-09-26: release "when everything is fully done") | `docs/decisions.md` D5, D17, H1 in "owner answers to the open questions" (2026-09-26) |

## open questions for the owner

* **Q25** (2026-10-08, raised by the audit): the licence and package metadata for 2.0.0. the repo has no licence
  of its own (D15 noted it); `pyproject.toml` has no `license`, `readme`, `authors`, `urls` or classifiers; there
  is no publish workflow and no changelog. which licence (the vendored itf1788 files are Apache 2.0, LGPL-2.1+ or
  all-permissive per file, test-only; CORE-MATH's rows are MIT), and should a publish workflow (a tag push
  uploading to PyPI) and a changelog come with it?
* **Q26** (2026-10-08): fix m14b-open (row 2) before 2.0.0? it changes the number type of some ends (no value
  changes), so after 2.0 it is a behaviour change of a released version

every earlier question (Q1-Q24) is answered: `docs/decisions.md`.

## still owed

* fuzz gaps a census left (2026-10-03; the rest of its shortlist is built): no @given test for `DecoratedInterval.log(base)` (random base), the decorated reflected ops and divmod (examples only), the slow decorated functions (pow, hypot, trig) on float operands, exact-operand equality of the outward and nearest classes for about 25 more functions, `Builder` (low value)
* the run ledger (2026-10-01) knows local runs only: a push whose src is unchanged since `origin/master`
  trusts that master's fuzz run was green on CI (every push is watched to the end), it does not check.
  count floors (zanzibar's `MIN_TESTS_ALL`) are recorded in each row, not enforced: a gate that
  collected fewer tests would still read green
* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (archive plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (archive plan §2 M13d sabotage). recorded, not scheduled
* M16's tests: the long double cases (`tests/test_numpy_compat.py::test_longdouble_is_exact`, the `longdouble`
  rows of test 2) discriminate only where `np.finfo(np.longdouble).nmant > 52` (CI's linux, never this laptop).
  the local env is python 3.13, so a 3.12-only hole would reach CI unseen (as the 3.11 `Fraction ** interval`
  one did, D25)
* M16a: a constructed n = 3 system with two zeros costs more than 120 s, so n = 3 is covered by one
  constructed zero and the sphere only, with no random n = 3 test; whether a faster jacobian
  (Q12(a)) or a tighter form of `F` would change that is not measured. the natural path to a split
  of a box already proved unique was not found (pinned by a monkeypatched `_krawczyk` only). a
  continuum costs up to 2n + 1 output boxes per box of width `tol` (review F3); whether to output it
  differently (one box per connected unproved region) was never asked. `newton`'s own `width <=
  piece.wid() / 2` is left as M15 wrote it (archive plan §2 M16a)
* M16d/M16e: the critique's r = `2.0 ** 60` is pinned with u = -1 only, because `O(u) ** 2 ** 60` did not finish
  for u in {2, 1e300}; since D28 it answers at once (2026-10-08), so the pin can take those u now
* sabotage rows red in their first run were not re-run after the closing tests were added (M16a:
  the four closing tests and the review's, which only add red paths); M15's table was not re-checked
  for the stale-bytecode hazard M16c found (`tools/sabotage.py` could re-run either now)
* T1 (`tools/sabotage.py`): the `finally` in `Run.pytest` that kills the child when the wait itself is
  interrupted is untested; the linux kill paths ran but were not sabotaged; on macOS `stop` trusts the PID
  alone (no creation time)
* M16b: the layer's per-call `warnings.catch_warnings` is not thread-safe on python 3.11-3.13 (as
  `decorated.py::_quietly`); recorded, not addressed. the pass imports the adapter a second time as
  `tests.itf1788.test_itf1788` (the vectors parsed twice, 7.5 s cold, 2026-09-27); accepted
* M16c: the other relations over cut tuples in `relations.py` (`before`, `adjoins`, ...) also read
  normalized operands and do not assert it; only `allen_relations`, whose wrong answer would be
  silent and partial, does. the methods are unaffected
* M16e: the whole suite under `MULTIINTERVAL_BACKEND=gmpy2` runs in CI's `gate-gmpy2` job since 2026-10-04
  (Q16(e)) and locally as `gate:gmpy2` when a backend file changed. the backend is verified only with gmpy2 2.3.1 / MPFR
  4.2.2 on windows (python 3.13); CI's linux jobs run `tests/test_backend.py` with the PyPI wheel, no
  other MPFR build has been run. free-threaded builds untested (the three contexts are shared module
  objects; `backend._use` is a global). the speed numbers were taken beside four other streams' runs:
  a quiet-machine re-measure (`tools/backend_speed.py`) is owed before any number goes into the
  README. gmpy2 3: `auto` and `[test]` stop below it; when it ships, run the differential against it
  before `backend.CEILING` and the `[test]` pin move together
  (`tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` keeps them equal)
* pown-huge: `ops._check_marker_premises` refuses at import an `EXACT_POWER_LIMIT` below the
  marker proof's floor (36550 bits), but a limit lowered at run time, after import, still stalls
  (`O(0.5, 1.0) ** 1074` past a 60 s timeout; review SAB-3). recorded, not guarded

## session log (newest first; older entries in `docs/session-log.md`)

* **2026-10-08** housekeeping and the v2 archive (the owner: "v2 is more or less done (it is right? help me check
  that) so we should archive the prd / plans", one decisions log, not in HANDOFF, "get all the housekeeping
  done"). a read-only opus audit checked every milestone's record and spot-checked a symbol of each: v2 built,
  H1 open, nothing marked done unmet (its stale lines listed in `docs/archive/v2/README.md`). the plans moved to
  `docs/archive/v2/`; their decision sections (the D table, `v2-plan.md`'s log) moved verbatim to
  `docs/decisions.md`, with Q5's and Q6's superseded markers; records of new work go to `docs/records.md`;
  what was live only in the plans moved here (row "later", "still owed"); pointers in `CLAUDE.md`, README,
  the testing skill (plus the CI `ci`-profile rule, and `git archive <rev> multiinterval`), the package
  docstring. `tools/coremath.py status` fixed to see commits before the rename (said 1, is 5). housekeeping:
  the 2026-10-06 fuzz-speed study kept as `references/fuzz-speed-2026-10-06/` (`96a784c`); `.scratch/ci`,
  `push`, `fuzz-speed` and two killed runs' ledger logs to the Recycle Bin; seven orphaned `tail -f` from
  2026-09-26..10-04 stopped by PID (cwd checked); the banner cut to the current state and the session log
  rotated (older entries and the retired banner in `docs/session-log.md`). new row fuzz-xdist (first: the old banner's "next"), row H1 gained what 2.0 needs;
  new Q25, Q26. gate 27798 + 7143 (2026-10-08). Still owed: the push (x10 prepush; the owner's go); the CORE-MATH full check (the
  owner's call)
* **2026-10-07** (no entry by its session; written 2026-10-08 from the commits) the 1788 libraries on PyPI
  (`a0a8939`, `references/python-1788-libraries-2026-10-07.md`), the context-framework audit's doc fixes recorded
  as owed (`b77c2e8`; done 2026-10-08), and `tools/pyintval_check.py` (`25c2f1a`), the `ieee1788` layer against
  pyintval, local only (owner): 768,000 comparisons over three seeds, nothing found in the library (that file's
  last section). gate 27798 + 7143 on its code (2026-10-07). not pushed
* **2026-10-06** Q24 (the owner: refuse trailing and doubled separators, "okay yes refuse both"; asked whether to
  drop the refusal of juxtaposed items, took the session's recommendation to keep it). recorded (`8c92770`); an opus
  agent built it in `../intervals-q24` (plan §2 "Q24 built"); the session re-probed and re-sabotaged;
  fast-forwarded to `086b3a6`, worktree and branch removed. gate 27798 + 7143 (2026-10-06). Still owed: the push (x10 prepush and
  `gate:gmpy2`; the owner's go)

* **2026-10-06** Q23 (the owner: refuse numbers with no separator and non-ASCII digits, "each number should be
  something python can parse, split by a character that's not a valid part of the number"; hex only if needed, it
  is, D28; then, asked: refuse `- 5`, keep `1 / 3`, accept `_` as python does, keep `∞`; then "should we require
  commas or semicolons as separators": yes). recorded in `v2-plan.md`'s decision log (`f49bafd`); an opus agent
  built it in `../intervals-q23` (plan §2 "M14-breadth", "Q23 built"); the session re-probed old against new and
  re-sabotaged; fast-forwarded to `8edc643`, worktree and branch removed. gate 27798 + 6746 (2026-10-06; a first gate:rest died at 71% with no row). new Q24. Still owed: the push (x10
  prepush and `gate:gmpy2`; the owner's go)

* **2026-10-05/06** open items by subagents (the owner: "do some of the open tasks, but get subagents to do them").
  first the rename: the previous context's gate run went MOVED (README edited mid-run), re-run green (27795 + 6509)
  and committed (`c6a4cca`). then three opus agents, one worktree each off `c6a4cca`: vectors-ext (a), m14b-open's
  parse and sampling items, T1. the session re-ran a sabotage row of each: the first two held; T1's first commit
  had its central guard unpinned (a surviving break not failing the run passed all 35 tests), sent back, pinned
  with 17 more self-sabotage rows and a race in `stop` fixed (plan §2 T1). rebased and fast-forwarded: `a3db14c`,
  `281922b`, `076ad32`, `bf1ec7c`; worktrees and branches removed. gate 27798 + 6609 (2026-10-06). new Q23 (two
  parse leftovers). Still owed: the push (x10 prepush and `gate:gmpy2`; the owner's go); nothing here touched the scalar
  evaluator, so the CORE-MATH full check stays the owner's call from trig-rev-far (`tools/coremath.py status`)

* **2026-10-05** the name (the owner: "I need a good name for this library that isn't already on pypi", then
  `multiinterval`, one word, "do the rename in this repo now"). candidates checked against PyPI's JSON API (404 =
  no project): `multiinterval`, `multi-interval`, `atoll`, `attained`, `realsets` free; `intervalset`, `dedekind`,
  `archipelago`, `enclosure` taken. `git mv intervals multiinterval` and a scripted byte-level pass (CRLF kept)
  over every tracked file but `references/`: imports, `intervals.x` and `intervals/x` pointers, quoted names,
  `INTERVALS_*` variables; then by hand the bare `intervals` identifiers in four tests, `tools/coremath.py`'s import
  walker paths and the directory mentions in the docs. the ledger's `ALGO` id kept (a new one would orphan every row).
  Still owed: the push (x10 prepush and `gate:gmpy2`); claiming the name on PyPI (the first upload)
* **2026-10-05** trig-rev-far (the owner: "get a subagent to do that"). an opus agent in the worktree `../intervals-trig-rev-far`
  measured (branch count, not cost per branch: ~ulp(X)/(2 pi) branches round onto x's open end to nearest, onto ±inf past the
  doubles), fixed with identical results, pinned (a differential hypothesis test against the old walk; 18 far cases under a
  10000-branch cap) and sabotaged (9 breaks red). the session re-checked: the diff, tan 1e21 3.4 s -> 0.004 s with the same
  answer, sin/cos at 1e21 equal to master's, 1e300 and 10**400 under 1 s, the far pins red on a `git archive` of `913bd5a`
  (15 of 15 before a 500 s cap). fast-forwarded `master` to `8e709b3`; worktree, branch and `.scratch/trig-rev-far/` removed.
  gate 27795 + 6509. Q21 walked through with the owner, who took every recommendation (`v2-plan.md` decision log). Still owed: the push (x10 prepush and `gate:gmpy2`); the CORE-MATH
  full check is the owner's call (`elementary.py` changed: a cache of the inverse-trig enclosure)
* **2026-10-04** Q22 and H4 (the owner: "A - yes just make it strict and refuse", "B - copy it into references",
  the tests fixed, "then we can complete the task about removing v1", then the full gate, push and babysit CI).
  strict flags (`cuts.flag`) and pins for the audit's unrecorded time differences, 10 sabotage breaks all red (plan
  §2 "q22-h4"); `references/v1-readme.md`; `archive/v1/` deleted, with its differentials in `tests/test_kernel.py`
  and `tests/test_modulo.py` (the second missing from H4's list) and the `pythonpath` entry. gate 27795 + 6489;
  prepush green (x10: 27795 in 63 s + 6489 in 5344 s); pushed `5888c6e..a984e26`; the babysitter (haiku): CI run
  37213772220 green, fuzz run 37213772177 red on a test oracle (a DST gap; plan §2 "fuzz-dst-gap"), reproduced by
  the session, fixed and pinned, sabotaged red. gate 27795 + 6490; prepush green; pushed `a984e26..48631f5`; CI run
  37223604634 and fuzz run 37223604624 both green (34285 passed). the first babysitter's "reproduced locally" rested
  on a script importing a name that does not exist (`from multiinterval import Interval`); the session reproduced the
  case itself. Still owed: Q21; the CORE-MATH full check is the owner's call (`cuts.py` and `kernel.py` changed: the
  flag check in constructors only)
