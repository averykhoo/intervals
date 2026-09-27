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

* **M15 (H3's first part) built on `v2`, committed, not pushed**: forward-mode autodiff
  (`intervals/autodiff.py`, `Dual`, `derivative`) and interval newton (`intervals/solver.py`,
  `newton`, `Root`), exported from `intervals`. the choices the build made are D19, owner question
  Q11. record: plan §2 M15; design: `v2-plan.md` "the solver stack"

* branch `v2`: M13 finished and merged 2026-09-27 (branches `m13e`, `m13g`, merged in `m13-merge`,
  then fast-forwarded into `v2`); pushed. `origin/v2` is at `a1d45a9`, whose CI run 36305984327
  is green (all 8 jobs, 22166 passed on each of python 3.11-3.14, 2026-09-27): M13e and M13g have
  now run on every supported python
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root; on
  this shared laptop it runs past the 10-min tool limit, so run it as two calls (`tests/itf1788` and
  `--ignore=tests/itf1788`). last recorded 2026-09-27 at M15: 18246 passed in 67 s + 4088 in 482 s (22334) (22166 at the H2' fix)
* M13 done: every statement of the 19 itf1788 files runs (9542 vectors of 111 ops, 0 skipped, pinned
  by `tests/itf1788/test_itf1788.py::test_nothing_is_skipped`); 185 divergence keys, 0 unknown
  failures (2026-09-27; regenerate with `tools/itf1788_census.py`). M14: fuzz job and flint oracle
  built, never run on GitHub

## open items (ranked)

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | M14-run | the fuzz workflow's first green run on GitHub; record its example count and time, dated. **the default is ×10 since 2026-09-27** (the owner's choice, to keep runs cheap; `fuzz.yml` and `tests/conftest.py`, `timeout-minutes: 180`), because ×100, the old default, does not fit the 350-min timeout. measured locally 2026-09-27 at `dbec908` (the workflow's command, `FUZZ_MULTIPLIER=10`, python 3.13, shared laptop under another session's load): `22166 passed in 5036.97s` (1 h 24 min), no failures; it did not find the H2' example. linear fit through the ×1 gate (691 s non-vector, same day) and ×10 (4971 s), itf1788's 66 s held fixed: ×100 ≈ 47800 s (13.3 h); ×1 fuzz of `test_reverse.py`/`test_orders.py` scaled 12.0× to ×10 (per test 9.1-23.6×), so slightly superlinear, ≈ 16 h. with 25% headroom (≤ 262 min) ×25 fits (≈ 204 min), ×30 is the edge. laptop timings, not the runner's; the old 2.2-3.4 h estimate is withdrawn | blocked: GitHub has only `ci` registered (checked 2026-09-27, `gh workflow list --all`); `fuzz.yml` must reach `master` (the default branch) before `schedule` or `workflow_dispatch` can run it — the owner's call | plan §2 M14 "exit" and "**the fuzz job, built 2026-09-26**" |
| 2 | M14-breadth | fuzz where it is thin: `tests/test_extreme_floats.py` extended to the functions, `minimum`/`maximum`/`fma`, `%`, `//` and `OutwardMultiInterval`; more `@given` in `test_outward`, `test_steps`, `test_fmt`, `test_applicator` | ready | plan §2 M14 "**breadth where fuzz is thin**" |
| 3 | Q6-shift | port v1's `<<` and `>>` (owner 2026-09-26: "for sure"); choose the meaning on real sets when built (`A * 2**n`; `>>` exact or floored) | ready | plan §4 (the `<<`, `>>` row); `v2-plan.md` "2026-09-26 revision: owner answers" |
| 4 | T1 | a reusable sabotage engine in `tools/sabotage.py`: M13's sub-tasks wrote the same ~30-line loop nine times (copy the file, apply one replacement that must match exactly once, clear `.hypothesis`, run pytest with a timeout for hangs, restore, `filecmp`, log a line), each with its own table of breaks. the break tables are per task and not worth keeping; the engine is | idea, not scheduled (from the `.scratch/m13` audit, 2026-09-27) | plan §2 intro (sabotage rule) |
| 5 | Q6-rest | `random_multi_interval`, a public `apply()`: to-do, undecided whether to port | owner's call, later | plan §4 (their rows) |
| 6 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | when everything is fully done (owner 2026-09-26) | plan §2 M11; D5, D17 |
| 7 | M8 | the time layer on the v2 class | on hold, no rush (owner 2026-09-26); D4 recommends (a), Fraction seconds under a thin wrapper | plan §2 "M8 `time_interval.py`"; D4 |
| 8 | H3 | the rest of the solver stack: a solver in several variables (a `Dual` carries one derivative; a gradient, a jacobian, krawczyk), direction tag on a degenerate zero (only if a solver needs `1/(1/[inf])` back; M15 did not), thin `ieee1788.py`, numpy interop (today `__array_ufunc__ = None`), per-piece Allen matrix, gmpy2/mpfr backend. **built 2026-09-27 as M15: forward-mode autodiff and interval newton** (the first pick) | not v2.0; numpy and gmpy2/mpfr recorded, not now (owner 2026-09-26) | `v2-plan.md` "the solver stack" and "later (not in v2.0)"; the Allen matrix and `ieee1788.py`: `v2-plan.md` decision log "v2 consolidated decisions (2026-08-16)", "comparisons" and "ieee 1788 conformance: test adapter, not a runtime flag" |
| 9 | H4 | delete `archive/v1/` | after v2 is stable (owner 2026-09-26) | plan §2 M10 (last bullet before "done") |

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

* **Q11 M15's choices (D19)**, built as the session's defaults when the owner said "do h3 first":
  (a) `Dual`, `derivative`, `newton` and `Root` are public and exported from `intervals` (the
  2025-12 sketch named `autodiff.py` and `solver.py`), not newton "as a test" only; (b) newton's
  step runs only where decorations prove `f` C¹, else the piece is pruned and bisected; (c) the
  step is `mul_rev`, never `/`; (d) one variable only; (e) `tol=1e-10` absolute, `max_steps=10_000`.
  keep, rename, or narrow the public surface? (plan §0 D19; `v2-plan.md` "2026-09-27 revision: M15")

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

* **2026-09-27** M15, H3's first part: the owner asked for H3 first; built its suggested first pick,
  forward-mode autodiff over sets (`Dual`, chain rules over every elementary method, arb's taylor
  series as the oracle) and interval newton over multi-intervals (`newton`: the step is `mul_rev`,
  so a derivative set holding 0 splits a piece in one step; C¹ proved by decorations; uniqueness
  proofs; exponent splits for wide pieces). 30 breaks sabotaged, 23 red at once, the 7 green ones
  (3 gaps, 4 cost rules) closed with tests and re-run red. the gate's randomized run found an old
  test-oracle bug in `tests/test_kernel.py::test_normalize_is_canonical` (a midpoint underflowing
  onto an end), fixed and pinned. gate 18246 passed in 67 s + 4088 in 482 s (22334). not pushed. records: plan
  §0 D19, §2 M15; `v2-plan.md` "the solver stack" and its 2026-09-27 revision; new question Q11

* **2026-09-27** (`69a667a`..`a1d45a9`, pushed) H2' done: `v2` pushed at `dbec908`, CI red on one
  test-oracle gap, fixed in `d7e46c2`, CI green at `a1d45a9` (plan §1). fuzz ×10 measured locally,
  ×100 does not fit, the default is now ×10 (M14-run). `.scratch/m13` still not recycled (the
  VisualBasic recycle call refused it; `.scratch/fuzz` went through)
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
  directory left in place: neither Recycle Bin route worked from the session, and the audit found
  nothing else in it untracked, so `.scratch/m13/` can go to the Recycle Bin by hand. records: plan §2 M13e, M13g, "exit for M13"; `v2-plan.md` 2026-09-27 revisions

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
