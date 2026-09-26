# handoff

what is true now: ranked open items, open questions for the owner, loose ends, and a dated session
log. volatile by design. the spec of anything not yet built stays in `v2-implementation-plan.md`
(milestones, D rows, exits) and the design in `v2-plan.md` ("current design"); the rows below only
point at them. a task the owner assigns overrides the ranking. at the end of a session: add a
session-log entry, refresh the banner, edit rows in place, sweep closed ids (a finished item leaves
the open-items table: its record goes in the plan's milestone section, with a one-line entry in the
session log below; nothing is listed as open and done at once), and list anything skipped as
"Still owed:".

## banner (2026-09-26)

* branch `v2`, last commit the M13d commit (2026-09-26, after `ed2a300`). `origin/v2` is at
  `d232b78` (2026-09-25): the 17 commits `d897d77` to M13d (M12, the M13/M14 plan, M13a/b/c/d/f/h,
  M14's fuzz job and flint oracle, this file) are **not pushed**, so CI has not run on any of them.
  the last CI run is 36091651163 at `d232b78`, all 8 jobs green (plan section 1)
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root.
  last recorded 17346 passed in 373 s (2026-09-26, at M13d; plan M13d "evidence"). the laptop is
  shared with other repos' jobs: the same gate took 6-10 min on the night of 2026-09-26, so a slow
  run is load, not a regression
* M13: a, b, c, d, f, h done; e, g open. M14: fuzz job and flint oracle built, never run on GitHub.
  itf1788: 7314 vectors of 83 ops, 2228 statements of 28 ops still skipped (2026-09-26, M13d;
  regenerate with `tools/itf1788_census.py`)

## open items (ranked)

the plan's suggested order (2026-09-25): one session each for M13e, then M13g starting with Q1,
each ending with a green gate and a commit.

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | M13e | reverse ops in a new `intervals/reverse.py` (1955 statements) | ready | plan §2 M13 "**M13e reverse ops**"; D12 |
| 2 | M13g | decorated wrapper type, NaI, 1788 constructors, signals; the 52 generated `[nai]` rows go stale | blocked on Q1 | plan §2 M13 "**M13g decorations, NaI, constructors and signals**"; D16 |
| 3 | M13-exit | `SKIPPED` empty and asserted (`tests/itf1788/test_itf1788.py::SKIPPED`) | after M13e, g | plan §2 M13 "**exit for M13**" and "**every sub-task**" |
| 4 | H2 | push `v2` (17 commits), so CI runs on M12 and M13 | owner's permission | plan §1 (CI) |
| 5 | M14-run | the fuzz workflow's first green run on GitHub; record its example count and time, dated. `multiplier=10` is the cheap first `workflow_dispatch`; ×100 was extrapolated to 2.2-3.4 h, under the 350-min timeout | after H2. caveat: `.github/workflows/fuzz.yml` is not on `origin/master` (no workflows there at all), and GitHub runs `schedule` and `workflow_dispatch` only for workflows on the default branch (GitHub docs; not tried here), so a push to `v2` alone may not make it runnable | plan §2 M14 "exit" and "**the fuzz job, built 2026-09-26**" |
| 6 | M14-breadth | fuzz where it is thin: `tests/test_extreme_floats.py` extended to the functions, `minimum`/`maximum`/`fma`, `%`, `//` and `OutwardMultiInterval`; more `@given` in `test_outward`, `test_steps`, `test_fmt`, `test_applicator` | ready | plan §2 M14 "**breadth where fuzz is thin**" |
| 7 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | owner's call; not blocked by M13, no hurry (D17) | plan §2 M11; D5, D17 |
| 8 | M8 | the time layer on the v2 class | deferred (D4); Q5 | plan §2 "M8 `time_interval.py`"; D4 |
| 9 | H3 | solver stack: direction tag on a degenerate zero (only if a solver needs `1/(1/[inf])` back), thin `ieee1788.py`, autodiff, Newton as a test (buildable: the functions exist since M12), numpy interop (today `__array_ufunc__ = None`), per-piece Allen matrix, gmpy2/mpfr backend | not v2.0; owner's call | `v2-plan.md` "later (not in v2.0)"; the Allen matrix and `ieee1788.py`: `v2-plan.md` decision log "v2 consolidated decisions (2026-08-16)", "comparisons" and "ieee 1788 conformance: test adapter, not a runtime flag" |
| 10 | H4 | delete `archive/v1/` | owner decision only, once v2 works: after release, and after M8 if the time layer is wanted | plan §2 M10 (last bullet before "done") |
| 11 | H5 | old README leftovers: the reading list (arxiv 1111.0167, Hickey) and "redo illustrations with negative and positive bits" (modulo) | if still wanted | `archive/v1/README.md` |

## open questions for the owner

Q2-Q4 and Q7 were built on the conservative reading, each recorded in its decision-log entry; only
Q1 blocks a sub-task. Q5 and Q6 are scope calls with nothing built.

* **Q1** (blocks M13g) 1788's signals `UndefinedOperation`, `PossiblyUndefinedOperation`,
  `IntvlPartOfNaI` (`ieee1788-exceptions.itl`, the constructors' `signal` clauses): `IntervalWarning`
  subclasses or exceptions? no default taken. recorded: plan §0 D16, M13g; `v2-plan.md` decision
  log "2026-09-25 revision: M13 and M14 planned (not built)"
* **Q2** (M13h) where 1788 answers NaN (a `nan` operand, `inf + -inf`, `0 * inf` in `dot`) the
  reductions raise `ValueError`. returning `nan`, as 1788 and python's float `sum` do, would be a
  looser, compatible change. recorded: `v2-plan.md` "2026-09-26 revision: M13h, the reductions, built"
* **Q3** (M13c) two readings: "equal infinite ends count" as 1788 writes it (two starts at -inf or
  two ends at +inf, so `[inf]` is not strictly less than itself); `.interior` in the reals
  (`[5, inf]` gives `(5, inf)`, not `(5, inf]`). recorded: `v2-plan.md` "2026-09-26 revision: M13c"
* **Q4** (M13f) rounding is outward in `OutwardMultiInterval` and to nearest in `MultiInterval`
  (the vectors require it), so `B + X ⊆ A` can miss by one ulp on float operands. an inward
  variant (a method or type giving a certified inner answer) is not built; today exact operands
  (`Fraction(f)`) give one. recorded: `v2-plan.md` "2026-09-26 revision: M13f"; plan M13f record
* **Q5** (M8) whether and when to build the time layer; D4 recommends (a), Fraction seconds under a
  thin wrapper. recorded: plan §0 D4, §2 M8
* **Q6** v1 surface with no v2 counterpart: port or record as gone: `<<` / `>>`,
  `random_multi_interval` (the tests use hypothesis instead), a public `apply()` (`applicator` and
  `OpDescriptor` are not exported). recorded: plan §4 (the three "open" rows)
* **Q7** (M13d) the readings D11 left open: `0 ** y` for y <= 0 dropped as outside the domain
  (`[0, 1] ** [-1]` = `[1, inf)`, where pown's `[0, 1] ** -1` = `[1, inf]` attains the limit);
  `1 ** ±inf` and `inf ** 0` indeterminate (empty and a warning, as atan2's); an integral Fraction
  exponent is pown; `acot` continuous in (0, pi) rather than `atan(1/x)`. recorded: `v2-plan.md`
  "2026-09-26 revision: M13d"; plan M13d record

## still owed

* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* CI has not run since `d232b78`: M12's and M13's commits are green locally only (H2)
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (plan §2 M13d sabotage). recorded, not scheduled

## session log (newest first)

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
