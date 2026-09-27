# handoff

what is true now: ranked open items, open questions for the owner, loose ends, and a dated session
log. volatile by design. the spec of anything not yet built stays in `v2-implementation-plan.md`
(milestones, D rows, exits) and the design in `v2-plan.md` ("current design"); the rows below only
point at them. a task the owner assigns overrides the ranking. at the end of a session: add a
session-log entry, refresh the banner, edit rows in place, sweep closed ids (a finished item leaves
the open-items table: its record goes in the plan's milestone section, with a one-line entry in the
session log below; nothing is listed as open and done at once), and list anything skipped as
"Still owed:".

## banner (2026-09-27)

* branch `v2`: M13 finished and merged 2026-09-27 (branches `m13e`, `m13g`, merged in `m13-merge`,
  then fast-forwarded into `v2`); **not pushed**. `origin/v2` is at `e5dc315`; everything after it
  is local. the last CI run is 36219282601 at `2f3a895` (all 8 jobs green, 2026-09-26): nothing of
  M13e or M13g has run on python 3.11, 3.12 or 3.14 yet
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root; on
  this shared laptop it runs past the 10-min tool limit, so run it as two calls (`tests/itf1788` and
  `--ignore=tests/itf1788`). last recorded 2026-09-27 at M13's close: 18246 passed in 40 s + 3920 in 510 s (22166; 17346 at M13d)
* M13 done: every statement of the 19 itf1788 files runs (9542 vectors of 111 ops, 0 skipped, pinned
  by `tests/itf1788/test_itf1788.py::test_nothing_is_skipped`); 185 divergence keys, 0 unknown
  failures (2026-09-27; regenerate with `tools/itf1788_census.py`). M14: fuzz job and flint oracle
  built, never run on GitHub

## open items (ranked)

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | H2' | push `v2` (M13e, M13g, the merge, D18) and read the CI run: the first run of the reverse ops and the decorated type on python 3.11-3.14 | needs the owner's go-ahead to push | plan §1 (CI) |
| 2 | M14-run | the fuzz workflow's first green run on GitHub; record its example count and time, dated. `multiplier=10` is the cheap first `workflow_dispatch`; ×100 was extrapolated to 2.2-3.4 h before M13e/g, under the 350-min timeout, and M13e/g grew the gate from 17346 to 22166 items (the non-vector ones, where the property tests are, from 3351 to 3920; 2026-09-27), so re-extrapolate from the ×10 run before trying ×100 | caveat: `.github/workflows/fuzz.yml` is not on `origin/master` (no workflows there at all), and GitHub runs `schedule` and `workflow_dispatch` only for workflows on the default branch (GitHub docs; not tried here), so a push to `v2` alone may not make it runnable | plan §2 M14 "exit" and "**the fuzz job, built 2026-09-26**" |
| 3 | M14-breadth | fuzz where it is thin: `tests/test_extreme_floats.py` extended to the functions, `minimum`/`maximum`/`fma`, `%`, `//` and `OutwardMultiInterval`; more `@given` in `test_outward`, `test_steps`, `test_fmt`, `test_applicator` | ready | plan §2 M14 "**breadth where fuzz is thin**" |
| 4 | Q6-shift | port v1's `<<` and `>>` (owner 2026-09-26: "for sure"); choose the meaning on real sets when built (`A * 2**n`; `>>` exact or floored) | ready | plan §4 (the `<<`, `>>` row); `v2-plan.md` "2026-09-26 revision: owner answers" |
| 5 | T1 | a reusable sabotage engine in `tools/sabotage.py`: M13's sub-tasks wrote the same ~30-line loop nine times (copy the file, apply one replacement that must match exactly once, clear `.hypothesis`, run pytest with a timeout for hangs, restore, `filecmp`, log a line), each with its own table of breaks. the break tables are per task and not worth keeping; the engine is | idea, not scheduled (from the `.scratch/m13` audit, 2026-09-27) | plan §2 intro (sabotage rule) |
| 6 | Q6-rest | `random_multi_interval`, a public `apply()`: to-do, undecided whether to port | owner's call, later | plan §4 (their rows) |
| 7 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | when everything is fully done (owner 2026-09-26) | plan §2 M11; D5, D17 |
| 8 | M8 | the time layer on the v2 class | on hold, no rush (owner 2026-09-26); D4 recommends (a), Fraction seconds under a thin wrapper | plan §2 "M8 `time_interval.py`"; D4 |
| 9 | H3 | solver stack: direction tag on a degenerate zero (only if a solver needs `1/(1/[inf])` back), thin `ieee1788.py`, autodiff, Newton as a test (buildable: the functions exist since M12), numpy interop (today `__array_ufunc__ = None`), per-piece Allen matrix, gmpy2/mpfr backend | not v2.0; numpy and gmpy2/mpfr recorded, not now (owner 2026-09-26); suggested first pick when it starts: Newton with autodiff | `v2-plan.md` "later (not in v2.0)"; the Allen matrix and `ieee1788.py`: `v2-plan.md` decision log "v2 consolidated decisions (2026-08-16)", "comparisons" and "ieee 1788 conformance: test adapter, not a runtime flag" |
| 10 | H4 | delete `archive/v1/` | after v2 is stable (owner 2026-09-26) | plan §2 M10 (last bullet before "done") |

## open questions for the owner

* **Q9 `mulRevToPair`'s decoration.** 1788 decorates the pair's first interval as the decorated
  division `c / b` where `0 ∉ b` (6 com, 41 dac, 5 def in `libieeep1788_mul_rev.itl`), but its
  `mulRev`, the hull of the same set, trv. ours is one op, `mul_rev`, always trv (sound: trv claims
  nothing), so 52 vectors are rows under "decoration expectations" on the decoration alone (the set
  must match, `tests/itf1788/test_itf1788.py::DECORATION_ONLY`). add a pair op with 1788's
  decoration, or keep the rows? built as the conservative reading (plan §2 M13 "exit for M13")
* **Q10 the constructors' outward pass.** the 201 interval-valued vectors of the four 1788
  constructors (`b-`/`d-textToInterval` 91 each, `b-numsToInterval` 10, `d-numsToInterval` 9) run in
  the plain pass only: they have no interval operand, so an outward item would repeat the plain
  call. give the constructors a class argument (`OutwardMultiInterval`, 1788's binary64 hull; new
  API), or is the plain pass enough? kept as built (plan §2 M13g "review")

Q1-Q8 answered 2026-09-26 (`v2-plan.md` "2026-09-26 revision: owner answers to the open
questions"); D18 (M13's two proposed categories, the exact-com rows, `set_dec`) answered 2026-09-27
(`v2-plan.md` "2026-09-27 revision: owner answers on M13's proposed categories (D18)").

## still owed

* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (plan §2 M13d sabotage). recorded, not scheduled

## session log (newest first)

* **2026-09-27** M13 finished, as an orchestrated build: M13e (reverse ops, `intervals/reverse.py`:
  sqr, abs, pown, cosh, mul, sin, cos, tan, pow_rev1, pow_rev2) and M13g (`DecoratedInterval`,
  `intervals/literals.py`'s 1788 text syntax, the four constructors, `set_dec`, the signals
  `UndefinedOperationError` and `PossiblyUndefinedOperationWarning`, propagation through every op)
  were built in parallel worktrees, each by sequential builders with properties and sabotage, then
  three independent reviewers each (math differential, sabotage audit, spec) and a fixer that
  reproduced each finding first. one library bug found and fixed: M13e's rational `log` rewrite was
  cubic in the operand's size (pinned). then merged, the decorated reverse vectors wired (481, 174
  pairs), the exit test added (`::test_nothing_is_skipped`; dropping an op from `OPS` turns it red).
  the owner approved both proposed categories and kept the exact-com rows and `set_dec`'s demotion
  (D18). the adapter's `::is_decorated` missed a decoration on the result alone (0 vectors run
  differently; the census undercounted 1521 for 1624), fixed and pinned. `.scratch/m13` audited
  (174 files): 2 notes' unrecorded review results transcribed into the M13e record, then the
  directory deleted. records: plan §2 M13e, M13g, "exit for M13"; `v2-plan.md` 2026-09-27 revisions

* **2026-09-26** the owner answered Q3-Q7 and H1-H5 (`v2-plan.md` "2026-09-26 revision: owner
  answers to the open questions"); then, after 1788's context, Q1 (`UndefinedOperation` and `IntvlPartOfNaI` raise,
  `PossiblyUndefinedOperation` warns), Q2 (keep `ValueError`) and a new Q8 (no NaI at all), so
  M13g is unblocked. Q6's `<<`/`>>`
  became an open item; the v1 README's leftovers moved to `references/todo-from-v1-readme.md`
  (H5 done). gate green (17346 in 380 s), then `v2` pushed (H2 done); CI run 36219282601 at
  `2f3a895` all 8 jobs green, the first CI on M12 and M13
* **2026-09-26** M13d: 1788 `pow` through `**` (D11), `__rpow__`, and expm1, log1p, cbrt,
  rootn, hypot, cot, sec, csc, acot, coth, csch, sech, acoth, correctly rounded in pure python, with
  the decimal and arb oracles and set-level properties. the 1939 vectors of those 14 ops, already
  vendored at M13a, now run: all match in both passes, no new row. a sabotage harness
  (throwaway, results in the plan's record) turned 23 of 25 breaks red; the two green ones only save ziv iterations.
  three first-thin catches were thickened with an example or a property. records: plan §2 M13d,
  `v2-plan.md` "2026-09-26 revision: M13d"

* **2026-09-26** HANDOFF.md created; open items and owner questions moved here from both plans.
  the census script behind the itf1788 counts moved from `.scratch/` to `tools/itf1788_census.py`
* **2026-09-26** (`814b302`..`bcee889`) M13a (vendored all 19 files of oheim/ITF1788, new
  parser and adapter), M14's fuzz job and flint oracle, then M13h, M13b, M13c, M13f; a review pinned
  cancel's vacuous `-inf`. fuzz found a test-oracle bug (mixed pairs rounded twice), and the gate an
  `exp10` test bug; the library had none. records: plan §2 M13a, b, c, f, h and M14
* **2026-09-25** (`62cc80c`..`c8d843d`) M13 and M14 planned; the owner settled D9-D17. before that
  (`53400d4`..`4e98bdc`) M12: the elementary and step functions, min/max/fma, outward rounding, 5
  more itf1788 files; exhaustive modulo re-run, 0 mismatches. records: plan §2 M12, M13, M14
* **2026-09-25** (`c6cfe14`..`d232b78`, pushed) M10 (v1 archived, new README, M11 backlog),
  M11's restored harnesses, CI; `d897d77` (not pushed) records the first CI run. records: plan §1,
  §2 M10, M11. M1-M9: 2026-09-23/24, see plan §2
