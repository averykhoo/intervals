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

* branch `v2`, pushed 2026-09-26 with the owner's permission (H2): the 18 commits `d897d77` to
  the owner-answers commit (M12, the M13/M14 plan, M13a/b/c/d/f/h, M14's fuzz job and flint
  oracle, this file). CI run 36219282601 at `2f3a895`: all 8 jobs green (gate on python
  3.11-3.14, 2m38s-3m48s; the three exhaustive grids and sabotage, longest 8m23s; 2026-09-26)
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root.
  last recorded 17346 passed in 380 s (2026-09-26, at `db52ad7`, M13d, the pre-push run). the laptop is
  shared with other repos' jobs: the same gate took 6-10 min on the night of 2026-09-26, so a slow
  run is load, not a regression
* M13: a, b, c, d, f, h done; e, g open. M14: fuzz job and flint oracle built, never run on GitHub.
  itf1788: 7314 vectors of 83 ops, 2228 statements of 28 ops still skipped (2026-09-26, M13d;
  regenerate with `tools/itf1788_census.py`)

## open items (ranked)

the plan's suggested order (2026-09-25): one session each for M13e, then M13g, each ending with a
green gate and a commit.

| # | id | what | status / blocker | spec |
|---|---|---|---|---|
| 1 | M13e | reverse ops in a new `intervals/reverse.py` (1955 statements) | ready | plan §2 M13 "**M13e reverse ops**"; D12 |
| 2 | M13g | decorated wrapper type, NaI, 1788 constructors, signals; the 52 generated `[nai]` rows go stale | ready: signals settled 2026-09-26 (D16); `IntvlPartOfNaI` as a warning is the owner's lean, may switch to raising before it is built | plan §2 M13 "**M13g decorations, NaI, constructors and signals**"; D16 |
| 3 | M13-exit | `SKIPPED` empty and asserted (`tests/itf1788/test_itf1788.py::SKIPPED`) | after M13e, g | plan §2 M13 "**exit for M13**" and "**every sub-task**" |
| 4 | M14-run | the fuzz workflow's first green run on GitHub; record its example count and time, dated. `multiplier=10` is the cheap first `workflow_dispatch`; ×100 was extrapolated to 2.2-3.4 h, under the 350-min timeout | H2 done (pushed 2026-09-26). caveat: `.github/workflows/fuzz.yml` is not on `origin/master` (no workflows there at all), and GitHub runs `schedule` and `workflow_dispatch` only for workflows on the default branch (GitHub docs; not tried here), so a push to `v2` alone may not make it runnable | plan §2 M14 "exit" and "**the fuzz job, built 2026-09-26**" |
| 5 | M14-breadth | fuzz where it is thin: `tests/test_extreme_floats.py` extended to the functions, `minimum`/`maximum`/`fma`, `%`, `//` and `OutwardMultiInterval`; more `@given` in `test_outward`, `test_steps`, `test_fmt`, `test_applicator` | ready | plan §2 M14 "**breadth where fuzz is thin**" |
| 6 | Q6-shift | port v1's `<<` and `>>` (owner 2026-09-26: "for sure"); choose the meaning on real sets when built (`A * 2**n`; `>>` exact or floored) | ready | plan §4 (the `<<`, `>>` row); `v2-plan.md` "2026-09-26 revision: owner answers" |
| 7 | Q6-rest | `random_multi_interval`, a public `apply()`: to-do, undecided whether to port | owner's call, later | plan §4 (their rows) |
| 8 | H1 | release 2.0.0 (`pyproject.toml` is now `2.0.0.dev0`) | when everything is fully done (owner 2026-09-26) | plan §2 M11; D5, D17 |
| 9 | M8 | the time layer on the v2 class | on hold, no rush (owner 2026-09-26); D4 recommends (a), Fraction seconds under a thin wrapper | plan §2 "M8 `time_interval.py`"; D4 |
| 10 | H3 | solver stack: direction tag on a degenerate zero (only if a solver needs `1/(1/[inf])` back), thin `ieee1788.py`, autodiff, Newton as a test (buildable: the functions exist since M12), numpy interop (today `__array_ufunc__ = None`), per-piece Allen matrix, gmpy2/mpfr backend | not v2.0; numpy and gmpy2/mpfr recorded, not now (owner 2026-09-26); suggested first pick when it starts: Newton with autodiff | `v2-plan.md` "later (not in v2.0)"; the Allen matrix and `ieee1788.py`: `v2-plan.md` decision log "v2 consolidated decisions (2026-08-16)", "comparisons" and "ieee 1788 conformance: test adapter, not a runtime flag" |
| 11 | H4 | delete `archive/v1/` | after v2 is stable (owner 2026-09-26) | plan §2 M10 (last bullet before "done") |

## open questions for the owner

none (2026-09-26): Q1-Q7 answered, recorded in `v2-plan.md` "2026-09-26 revision: owner answers to
the open questions". the one soft spot: `IntvlPartOfNaI` as a warning, which the owner may switch
to raising before M13g builds it (the case is in that entry).

## still owed

* the M12 taylor loops' error bounds are argued in docstrings, not pinned: removing them keeps every
  test green (plan §2 M12 "evidence", sabotage bullet). recorded, not scheduled
* M13d's extra working bits near 0 for expm1 and log1p (`elementary.py::_tiny_bits`) are a speed
  measure only: removing them keeps every test green, since ziv then doubles the precision itself
  (plan §2 M13d sabotage). recorded, not scheduled

## session log (newest first)

* **2026-09-26** the owner answered Q3-Q7 and H1-H5 (`v2-plan.md` "2026-09-26 revision: owner
  answers to the open questions"); then, after 1788's context, Q1 (`UndefinedOperation` raises,
  the other two warn) and Q2 (keep `ValueError`), so M13g is unblocked. Q6's `<<`/`>>`
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
