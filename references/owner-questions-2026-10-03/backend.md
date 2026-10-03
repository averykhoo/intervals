# Q16 M16e's choices (D24): the gmpy2/mpfr backend — options and recommendations

written 2026-10-03 by a read-only advisory agent. sources: `v2-implementation-plan.md` §0 D24 and §2 M16e;
`v2-plan.md` "elementary and step functions" (the backend bullets) and "2026-09-28 revision: M16e";
`intervals/backend.py`, `intervals/_gmpy2.py`, `intervals/elementary.py`, `intervals/ops.py`;
`tests/test_backend.py`; `pyproject.toml`; `.github/workflows/ci.yml`, `fuzz.yml`; `tools/gate.py`,
`tools/prepush.sh`; `HANDOFF.md` Q16, "still owed" M16e bullet, open item 8 `evaluate-box`. probes under
`.scratch/owner-questions/probes-backend/` (`speed_probe.py`, `identity_probe.py`, `newton_rerun.py`, their `.out`).

status of sections: all written.

## 0. what is built today (common to all sub-questions)

verified by reading `intervals/backend.py`, `intervals/_gmpy2.py`, the four dispatch sites and `pyproject.toml`
(2026-10-03):

* selection: `backend.py::_select` reads `INTERVALS_BACKEND` once at `import intervals`. unset/`''`/`python`:
  pure, gmpy2 never imported. `gmpy2`: forced, `ImportError` naming why if gmpy2 is missing or below
  `backend.FLOOR` (2.3) / `backend.MPFR_FLOOR` (4.2). `auto`: gmpy2 iff importable and `2.3 <= version < backend.CEILING`
  (3), else pure, silently. anything else `ValueError`. `backend.name()` says which; not exported from `intervals`.
  `backend._use` is the tests' switch (module global, not thread safe).
* the contract (`_gmpy2.py` module docstring; `v2-plan.md` "elementary and step functions", the backend bullets):
  the backend answers only "which double", after every decision that is not a rounding. dispatch sites:
  `elementary.py::rounded` (after `_beyond`), `::rounded_pow` (after the two range shortcuts), `::rounded_angle`
  (after `q == 0 and m == 0`), `::rounded_inverse_trig` (after the exact k = 0 case), and `ops.py::outward` keyed on
  the five descriptor objects (`ops._FAST_OPS`). each reads `backend.fast` at call time; None falls through to the
  pure `_ziv` loop or `round_rational`.
* what it declines (pure): non-dyadic x (except atan/acot via `atan2` of two ints, and the hook's mixed operands
  as an exact `mpq`), `log` to a base, `acoth`, `rootn` with `n <= 0` or `n >= 2**31` (`_gmpy2.ROOTN_LIMIT`), k != 0
  in `k pi + f(v)`, any operand past `_gmpy2.BOUND` (2**20) bits, the `pow{n}` descriptors.
* guards: every mpfr names a context (`_gmpy2._CONTEXTS`, `_WIDE`); `+ 0.0` last (no `-0.0`); `_gmpy2._value` raises
  `ArithmeticError` on nan and on a ternary value of 0 for an elementary result (a missed exact case).
* packaging: `pyproject.toml` `fast = ["gmpy2>=2.3"]` (no ceiling), `test = [..., "gmpy2>=2.3,<3"]`;
  `tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` keeps the `[test]` pin equal to
  `FLOOR`/`CEILING`. `ci.yml`: three gate jobs (3.12/3.13/3.14, ubuntu) install `.[test]` + numpy and run the suite on
  the default (pure) path; `fuzz.yml`: one x10 job, same install, 180-min timeout. no job sets `INTERVALS_BACKEND`.
* the run ledger: `tools/gate.py::run_phase` **removes `INTERVALS_BACKEND`** from the environment ("a verdict on
  INTERVALS_BACKEND=gmpy2 must not stand for one on the default"), and `::phase_spec` knows only `gate:itf`,
  `gate:rest`, `docs`, `fuzz-x<N>:itf`, `fuzz-x<N>:rest`. so a forced-gmpy2 whole-suite run is not recordable today;
  the one that exists (2026-09-28) is in the plan's prose only.
* installed here: python 3.13.15, gmpy2 2.3.1, MPFR 4.2.2, GMP 6.3.0 (probe, 2026-10-03). PyPI has only two releases in
  `auto`'s window, 2.3.0 and 2.3.1 (`pip index versions gmpy2`, 2026-10-03); every 2.2.x and below is refused by
  the floor. so `auto` in an environment with an older gmpy2 (conda envs made before mid-2025 are a likely case;
  inferred, not surveyed) is silently the pure path.
* verified MPFR builds: this laptop's gmpy2 2.3.1 / MPFR 4.2.2 (windows; the full suite forced once, 2026-09-28) and
  the PyPI linux wheel in CI, but only `tests/test_backend.py`'s differential there, never the whole suite
  (`HANDOFF.md` "still owed", M16e bullet).
* a stale citation found on the way: `tools/backend_speed.py`'s docstring cites `h3-records/gmpy2.md`; no such path
  is in the tree or in `git ls-files` (checked 2026-10-03). the tables it means are in `v2-implementation-plan.md`
  §2 M16e, "speed". doc-only; not fixed here (read-only session).

## 1. the design's claim "same doubles under both backends": checked

the claim (`backend.py` docstring: "both give the same doubles and the same flags"; README "a faster backend"):

* **why it should hold** (inferred from the code): MPFR is correctly rounded in every direction by specification,
  `gmpy2.ieee(64)` is binary64 with subnormals applied through the ternary value, and the input is always an mpfr
  *equal* to x (built at x's own bit length in `_WIDE`), so one MPFR call is one rounding of the exact value, which
  is what the pure `_ziv` loop computes. every non-rounding decision (exactness, flags, attainment, `_beyond`) stays
  pure, so a disagreement could only be "a different double for the same exact real", i.e. an MPFR or gmpy2 bug.
* **what pins it** (tracked): `tests/test_backend.py::test_rounded_matches_python` and kin (three-way, sign bit
  included), `::EDGES` (179 cases), `::HARD` (78 points within 2**-17..2**-22 half-ulps of a rounding boundary),
  `::test_set_level_matches_unary`, `::test_set_level_matches_binary`, `::test_set_level_matches_newton` (the
  `repr` under both). the M16e review's soundness lens checked ~41k primitive answers against arb, ~118k hook
  answers exactly, 1200 set-level reprs: 0 wrong (`v2-implementation-plan.md` §2 M16e, the review paragraph).
* **probe, 2026-10-03** (`.scratch/owner-questions/probes-backend/identity_probe.py`, seed 20261003): 400 random
  draws x 2 classes x 25 operations (17 unary methods, `+ * /`, `** 2.5`, `** -2`, `atan2`, `pow`, `hypot`), random
  float pieces including signed zeros, subnormals, 1e22, 1e308, 1-3 pieces: **20000 cases, 0 differences** in
  `repr` (which distinguishes `-0.0`) and in exception type/text; 10.4 s. VERIFIED on this machine's MPFR build.
* **the residual**: the identity is verified for gmpy2 2.3.1 / MPFR 4.2.2 (windows) and, for the primitives only,
  the linux PyPI wheel. it is *argued*, not tested, for any other MPFR build a user's `auto` might pick up (a
  distro's or conda-forge's MPFR). MPFR's correct rounding is its core contract and the same source builds
  everywhere, so the risk is small, but it is exactly the risk a pure default avoids and a gmpy2 CI job (Q16(e))
  partly measures. the failure mode if it ever broke: a double off by one ulp, silently, which in
  `OutwardMultiInterval` could miss the true value (unsound) and in `MultiInterval` is a wrong nearest. only the
  "missed exact case" class is caught by the `rc == 0` guard.
* **reproducibility across machines**: with the pure default, every machine gives identical sets by construction
  (rational arithmetic; `v2-plan.md` "values" bullet). with `auto` as default, two machines agree iff both MPFR
  builds are correctly rounded, which MPFR promises; the project cannot test this for builds it does not have.

## 2. timing probe, pure vs gmpy2 (2026-10-03 19:51, shared loaded laptop; ratios, not absolutes)

`.scratch/owner-questions/probes-backend/speed_probe.py` (the rows of `tools/backend_speed.py`, back to back under
`backend._use`, best of 5). python 3.13.15, gmpy2 2.3.1 / MPFR 4.2.2, Windows 11. other agents were running:
absolute times moved 2x between repeats of one row, so read the ratio column only.

| call | pure us | gmpy2 us | ratio | record 2026-09-28 |
|---|---|---|---|---|
| `rounded('exp', 0.7, DOWN)` | 25.1 | 7.95 | 3.2 | 4.8 |
| `rounded('exp', 1/3, DOWN)` (declined) | 26.9 | 30.1 | 0.9 | 0.96 |
| `rounded('log', 0.7, DOWN)` | 32.5 | 8.08 | 4.0 | 3.6 |
| `rounded('sin', 0.7, DOWN)` | 39.2 | 7.71 | 5.1 | 4.0 |
| `rounded('atan', 2**-30, DOWN)` | 48.7 | 6.49 | 7.5 | 6.3 |
| `rounded_pow(2, 1/2, UP)` | 48.5 | 16.3 | 3.0 | 2.9 |
| hook `add(0.1, 0.2)` down | 7.21 | 1.07 | 6.7 | 6.2 |
| hook `div(1.0, 3.0)` down | 6.68 | 1.2 | 5.5 | 5.6 |

| op | pure ms | gmpy2 ms | ratio | record |
|---|---|---|---|---|
| `O` 3 float pieces `.exp()` | 0.237 | 0.0971 | 2.4 | 2.3 |
| same `.log()` | 0.266 | 0.0937 | 2.8 | 3.5 |
| same `.sin()` (`floor_over_pi` pure) | 0.371 | 0.27 | 1.4 | 1.1 |
| same `.atan()` | 0.284 | 0.0975 | 2.9 | 2.8 |
| `M` 2 float pieces `.exp()` | 0.151 | 0.0658 | 2.3 | 2.1 |
| `A + B` outward (3 x 2) | 1.0 | 0.946 | 1.1 | 1.2 |
| `A * B` outward | 1.28 | 1.02 | 1.3 | 1.2 |
| `newton(t**2 - 2)` | 13.8 | 11.9 | 1.2 | 1.2 |
| `newton(sin t - t/3)` | 70.5 | 123 | 0.57 (noise) | 1.4 |

the last row re-run alone three times (`newton_rerun.py`, 19:52): 1.32, 1.26, 1.35 (absolute times doubled between
rep 0 and rep 1: load). so the record's picture stands: **3-7.5x per elementary primitive at a float, 2.3-2.9x at set
level for exp/log/atan, 1.1-1.4x for outward arithmetic and newton**, 1x where declined. the set-level gain is
bounded by python the backend does not touch (the applicator, `floor_over_pi`, the kernel), which is why open
item 8 `evaluate-box` bears on Q16(a): `applicator.py::evaluate_box` calls `desc.fn(*args)` (the exact value) and
then `desc.rounded[0]` and `[1]`, each of which under the pure path recomputes `exact(*args)`
(`ops.py::outward`, the inner `rounded`): three exact evaluations per float corner; under gmpy2 the two roundings
skip `exact`. passing the value in would give the pure path most of the hook's arithmetic gain with no dependency
(inferred from reading; not measured).

## Q16(a) automatic or opt-in

**the question.** what a user who never heard of `INTERVALS_BACKEND` gets. **built:** opt-in; unset is the pure
path (`backend.py::_select`; pinned by `tests/test_backend.py::test_env_var[None-...]`). the design had
automatic-when-importable and the critique reversed it (`v2-plan.md` "2026-09-28 revision: M16e", first bullet).

**the fact that frames every option:** the backend promises identical sets (§1). so the default is not a semantic
commitment, it is a *provenance* commitment: which code computed the double a user is looking at. flipping it later
changes no documented result, which makes this one of the cheaper choices to revisit after 2.0, provided §1 keeps
holding.

### option A1: opt-in (as built)

* meaning: `pip install intervals` runs pure python everywhere; `INTERVALS_BACKEND=gmpy2|auto` is the user's act.
* pros: the pure path is the reference, and the gate, the fuzz, itf1788 and CORE-MATH all run on the path users
  run; results are identical on every machine by construction, not by MPFR's promise; no native library a user did
  not ask for ever decides a double; a bug report needs no backend question (today, `backend.name()` is `python`
  unless they set something); the soundness story in the README stays one sentence.
* cons: nearly nobody discovers it, so the speed (2-3x at set level for elementary functions, ~1.2x arithmetic,
  §2) stays unused; a user whose environment already has gmpy2 (sympy and mpmath environments; mpmath itself
  auto-uses gmpy2 when importable, `MPMATH_NOGMPY` to refuse: background knowledge, not verified here) gets
  nothing from it.
* when better: a library whose selling point is "the same answer everywhere, provably" rather than throughput;
  a project whose CI cannot run every MPFR build its users have (this one); any time before a gmpy2 whole-suite CI
  job exists (Q16(e)).

### option A2: `auto` as the default

* meaning: unset behaves as `auto`: gmpy2 when importable and `2.3 <= version < 3`, else pure, silently.
  `python` would remain the explicit refusal.
* pros: free speed for anyone who already has gmpy2 2.3.x; the precedent (mpmath) shows users accept it; the
  version window and the hostile-context tests (`tests/test_backend.py::test_hostile_global_context`) already
  bound the exposure to gmpy2 2.3.0/2.3.1 and to the user's own gmpy2 settings.
* cons: a user's distribution's MPFR build decides their doubles unasked, and the project has verified one build
  (§0, §1 residual); two users comparing output must first learn which backend each ran; a bug report must carry
  `backend.name()` (so (b) changes); the gate as run here and in CI (pure) would no longer be the path a sizeable
  fraction of users run unless (e) adds the gmpy2 job; `tools/gate.py::run_phase` would have to stop stripping the
  variable or the ledger would record nothing about the path those users run; "silently" cuts both ways: the
  user who installed gmpy2 2.2.x for sympy gets pure and never knows. the window being two PyPI releases wide
  (§0) also makes the benefit narrower than "everyone with gmpy2".
* when better: when the owner's own workloads are elementary-function-heavy over many pieces and the owner
  controls the environment; when CI runs the whole suite on gmpy2 on linux, macos and windows wheels; when the
  speed is the headline rather than the soundness argument.

### option A3: `auto` default plus a one-time notice

* meaning: as A2, but `import intervals` emits a `warnings.warn(..., category=ImportWarning)` or a log line
  naming the backend taken.
* pros: provenance made visible.
* cons: import-time warnings are hated and filtered; `ImportWarning` is ignored by default so it buys nothing;
  the suite's `filterwarnings = error` would need an exemption. not recommended in any case.

### option A4 (not in the plan): keep opt-in, but offer a programmatic opt-in beside the variable

* meaning: `intervals.backend.select('gmpy2')` (or `use`) callable before the first evaluation, same rules as the
  variable; still off by default. this is really a (b) option; listed here because it is the usual answer to "the
  variable is awkward in a notebook or from a dependent library".
* pros: a dependent package can turn the backend on for its own users without touching their environment.
* cons: a public mutable global; the plan's "no ambient mode" line (`v2-plan.md` backend bullet: "rounding is a
  property of the type, never an ambient mode") is about results, which a backend switch does not change, so the
  objection is weaker than stated, but a setter still has thread-safety and "when does it take effect" semantics
  to promise. additive: can be added after 2.0 with no break. see (b).

### option A5 (not in the plan): staged: opt-in for 2.0, `auto` considered for a 2.x minor once (e) runs

* meaning: ship A1; add the gmpy2 CI job (e); after a few releases with it green across the linux wheel (and,
  if a macos/windows CI job is ever added, those), decide A2 with evidence of more than one build. because the
  default changes no documented result, it can be a minor-version change with a release-note line, not a 3.0.
* pros: gets the evidence before the exposure; nothing is foreclosed.
* cons: a default flip is still a behaviour change some users notice (a traceback now goes through `_gmpy2`);
  the window `< 3` means an eventual gmpy2 3 flips many users back to pure silently until `CEILING` moves.

### recommendation

**A1 now, as A5** (opt-in for 2.0; revisit only after Q16(e)'s gmpy2 job has run green for a while, and after
open item 8 `evaluate-box` has been tried, since it narrows the arithmetic gap with no dependency). confidence
**high (~80%)**. the deciding facts: the identity is verified on one MPFR build; the ledger and CI run the pure
path and would both have to change for `auto` to be an honest default; the measured gain is 2-3x on elementary
functions at set level and ~1.2x on arithmetic (§2), real but not transformative; the window is two releases.

what would change my mind: a gmpy2 whole-suite job green in CI on two or more wheels for several releases;
`evaluate-box` done and the backend still the dominant gain in a workload the owner cares about; or evidence
that the typical user already has gmpy2 2.3.x (then A2's benefit is wide and its risk measured).

cost of changing later: **before 2.0** zero. **after 2.0** low: identical results, so a minor-version flip
with a release note; the only breakage is for someone who relied on `backend.name() == 'python'` by default
(nobody should) or whose gmpy2 misbehaves (which is the risk itself).

## Q16(b) public surface

**the question.** what is API. **built:** the variable (`python|gmpy2|auto|unset`), the `[fast]` extra,
`intervals.backend.name()` importable but not exported from `intervals`; no setter (`backend.py` docstring;
`backend._use` is test-only). the README's "a faster backend" paragraph documents the variable and the extra; it
does not name `backend.name()`, and the README has no bug-report guidance at all (grep "bug report", 2026-10-03:
no hits).

### option B1: as built

* pros: smallest surface, nothing to deprecate; the module path `intervals.backend.name()` is already stable
  enough to put in a bug-report sentence.
* cons: nobody writing a bug report knows to call it; a dependent library cannot turn the backend on
  programmatically.
* when better: the default stays opt-in (then `name()` is almost always `python`).

### option B2: export a top-level `intervals.backend_name()` (or re-export `backend.name`)

* pros: discoverable; one line in a bug template: "paste `intervals.backend_name()`".
* cons: one more top-level name kept forever; it duplicates `intervals.backend.name()`. a cheaper equivalent:
  document `intervals.backend.name()` in the README under a short "reporting a result" sentence; the symbol is
  already public in all but `__all__`.
* when better: Q16(a) goes `auto` (then every report needs it, and discoverability is worth a name).

### option B3: a public setter / context manager (`intervals.backend.use('gmpy2')`)

* meaning: promote `_use` or add a one-shot `select()` taking effect for later calls.
* pros: notebooks and dependent libraries; benchmarks; tests of user code under both.
* cons: public mutable global state; `_use` is explicitly not thread safe (`backend.py` docstring: "a module
  global, not thread safe"), and a public one must say what a concurrent switch means; the forced-vs-auto
  semantics of the variable would need a programmatic twin. results are unaffected by a mid-flight switch
  (same doubles), so the harm is confusion and a thread-safety promise, not a wrong set.
* when better: the owner has a dependent package wanting speed for its users, or notebook users complain about
  setting an environment variable before import. additive, so it can wait for the demand.

### option B4 (not in the plan): a diagnostics helper `intervals.about()` / `intervals.backend.about()`

* meaning: a string with python, intervals, backend name, gmpy2 and MPFR versions (as `tools/gate.py::_versions`
  already assembles for the ledger).
* pros: the right unit for a bug report, more than the backend name alone.
* cons: scope; a second thing to keep accurate.
* when better: only if the owner wants a bug template at all; then prefer this to B2.

### recommendation

**B1, plus one README sentence** naming `intervals.backend.name()` for anyone reporting a result (doc only, no
new symbol). confidence **medium-high (~70%)**. do not add a setter for 2.0: additive later, and the demand is
unproven; the "ambient mode" objection is weaker than the plan states (results do not change), so if demand
appears, add `select()`/`use()` without guilt, documented as "speed only, not thread safe".

what would change my mind: Q16(a) flips to `auto` (then export the name, B2 or B4); a dependent package appears.

cost of changing later: adding names is free after 2.0; removing an exported name needs a deprecation cycle. so
under-export now.

## Q16(c) the non-dyadic points

**the question.** whether to build the backend's second part, an MPFR ziv loop for the inputs it declines today.
**built:** declined to the pure path: non-dyadic x (`Fraction(1, 3)`), `log` to a base, `pow_rev2`'s `log_t v`,
`acoth`, `rootn` with `n <= 0`, the periodic reverse ops' `k pi + f(v)`; natively handled where one call is one
rounding anyway (atan/acot and the angles via `atan2` of two ints, the hook's mixed operands via `mpq`). a declined
call costs one `bit_length` comparison and a few isinstance checks (`_gmpy2._ratio`; §2: 0.9-0.96x, noise).

### option C1: as built (defer indefinitely; measure demand first)

* pros: the correctness argument stays "one MPFR call is one rounding", checkable by reading `_gmpy2.py`; the
  differential's table (`tests/test_backend.py::declines_rounded` and kin) stays a complete statement of what the
  backend touches; no second code path to review. non-dyadic inputs are rare in float-heavy use: a `Fraction`
  operand, a user-chosen log base, `acoth`, negative roots, and the reverse trig ops at k != 0 are the whole list.
* cons: those calls stay at pure speed (25-150 us/call here; worse at wide operands); `sin_rev`/`cos_rev` over wide
  x with many periods call `rounded_inverse_trig` at k != 0 for every piece end, so a hot loop of reverse trig
  gets nothing from the backend.
* when better: always, until someone measures a workload bound by those calls.

### option C2: an MPFR ziv loop (the plan's "second part", `v2-plan.md` "later (not in v2.0)")

* meaning: for a monotone f and a bracket `[x_lo, x_hi]` of the exact x at precision p (or for a composite like
  `log x / log b`, `k pi + f(v)`, `1 / rootn(x, n)`), compute the result at precision p with directed rounding in
  both directions in MPFR, round each bound to a double in the requested direction, return when they agree,
  else double p; the pure `_ziv` loop's structure (`elementary.py::_ziv`) with MPFR as the arithmetic.
* pros: covers everything declined except the bits bound; the MPFR arithmetic is 10-50x faster than the
  fixed-point python the pure `_enclose` runs.
* cons: a *second kind* of correctness argument: per composite, the bound's sign and monotonicity must be
  right (two roundings in the same direction bound a monotone composition only with care about signs of
  intermediate values, e.g. `k pi + f(v)` with k < 0, or `log x / log b` with `b < 1`); a wrong bracket is a wrong
  double silently, which the present design makes impossible by construction; it needs its own differential and
  sabotage table; and the gain is **unmeasured** (`HANDOFF.md` Q16(c): "not measured"), on inputs that are rare.
  the record also shows that of a 30 us backend call the pure prelude (`exact`, `_beyond`) is already a third
  (`v2-implementation-plan.md` §2 M16e, "the design's probes"), so even a perfect second part cannot reach the
  per-call ratios of the first.
* when better: a user with exact-rational intervals (Fractions as ends) doing elementary functions in volume, or
  reverse trig over many periods in a solver loop, *and* a measurement showing those calls dominate.

### option C3 (not in the plan): the two cheap composites only

* meaning: handle only `rootn` with n < 0 (as `rootn` then reciprocal with a guard-bit precision and both
  directions, agree-or-fall-back) and `k pi + f(v)` with |k| small; leave non-dyadic x and `log` to a base pure.
* pros: smaller surface than C2; the reverse trig case is the one with a plausible hot loop.
* cons: still a second argument kind; still unmeasured; the differential table gets two conditional rows.
* when better: a measured reverse-trig workload, nothing else.

### recommendation

**C1.** confidence **high (~85%)**. the backend's whole value as built is that it cannot be wrong where it answers;
C2 trades that for a speedup nobody has measured on inputs few users have. keep the item under "later" with the
condition "measured first".

what would change my mind: a profile of a real workload showing `rounded_inverse_trig` at k != 0 or a
non-dyadic `rounded` on the critical path.

cost of changing later: zero either side of 2.0; purely internal, the differential grows rows.

## Q16(d) gmpy2 in `[test]`

**the question.** who installs gmpy2 for the tests, and how it is pinned. **built:** `test` has `gmpy2>=2.3,<3` so
`tests/test_backend.py` never skips, and the pin equals `auto`'s window
(`::test_the_test_extra_installs_what_auto_takes`, red without the pin: review V1). `fast = ["gmpy2>=2.3"]`, no
ceiling. the three CI gate jobs (3.12, 3.13, 3.14) and the fuzz job all install `.[test]`, so the differential runs
in every job; CI was green at `912558b` (`HANDOFF.md` banner), so gmpy2 2.3.1 wheels exist for 3.14 on ubuntu.

### option D1: as built

* pros: the differential never passes by skipping; CI installs exactly what `auto` would take; the pin and the
  code's window cannot drift (the test reads `pyproject.toml`).
* cons: a contributor on a platform without a gmpy2 wheel (a brand-new CPython before gmpy2 ships wheels, or an
  exotic OS) cannot install `[test]` at all: pip falls to a source build needing GMP/MPFR headers. for CI this
  bites exactly when a new python is added to the matrix before gmpy2's wheels exist; the fix is then "wait" or a
  per-job marker. with the window two releases wide, a conda env with gmpy2 2.2.x fails `[test]` resolution too
  unless pip upgrades it (pip will, in a pip-managed env; in a conda env that is the usual mixed-manager mess).
* when better: the project values "a test that cannot skip" over "anyone can run the tests anywhere" (this
  project's stated stance, `HANDOFF.md` Q16(d): "a test that passes by skipping").

### option D2: gmpy2 only in the CI jobs that ask; `tests/test_backend.py` skips without it

* pros: `[test]` installs everywhere; a new-python job works before gmpy2 wheels exist.
* cons: locally the differential silently skips on a machine without gmpy2, and a green local gate then says
  nothing about the backend; the repo's own doctrine ("an assurance step that fails by passing") rules it out.
* when better: a contributor base on many platforms; not this project's situation.

### option D3: pin `[fast]` to the same window (`gmpy2>=2.3,<3`)

* meaning: the user-facing extra installs what `auto` takes; `backend.CEILING` and both pins move together when
  gmpy2 3 is verified; extend `::test_the_test_extra_installs_what_auto_takes` to assert `[fast]`'s pin too.
* pros: coherence: today `pip install intervals[fast]` + `INTERVALS_BACKEND=auto` could, once gmpy2 3 is on PyPI,
  install gmpy2 3 and then silently run pure, which is the one combination the extra exists to prevent; with
  `INTERVALS_BACKEND=gmpy2` it would run an unverified series (likely an `AttributeError` at the first context
  call if the API moved, so loud rather than wrong, but unplanned). a pinned extra makes "I installed `[fast]`"
  mean "the backend is on if I ask".
* cons: narrows a user's environment: a project needing gmpy2 3 for something else cannot also install
  `intervals[fast]` until the pin moves; a pin the project must remember to move (the test makes forgetting
  loud).
* when better: whenever `auto` has a ceiling, which it does. the alternative that also restores coherence is to
  drop `auto`'s ceiling, which the owner should not: the series is what was verified.

### recommendation

**D1 + D3**: keep gmpy2 in `[test]` pinned as built, and pin `[fast]` to the same window, with the existing test
extended to cover both extras. confidence **medium-high (~70%)** on D3, **high** on keeping D1. the one real
cost of D1, a new CPython before gmpy2 wheels, is a CI-matrix timing issue the owner controls.

what would change my mind on D3: a known consumer needing gmpy2 >= 3 alongside; or a decision to make `auto`
accept any version at the floor (then no pin is coherent, and I would still keep `[test]`'s for CI determinism).

cost of changing later: a pin in an extra is a packaging change shippable in a patch release either way; low
before and after 2.0.

## Q16(e) CI and fuzz

**the question.** whether any CI job runs the whole suite on gmpy2, and whether the fuzz does. **built:** no
workflow change. every gate job and the fuzz job run the suite on the pure path and `tests/test_backend.py`'s
differential in-process (the primitives at drawn points, 15 edge classes, the set-level `repr` of every method,
newton; `v2-plan.md` "the backend differential"). the forced whole suite ran once, locally, 2026-09-28
(`v2-implementation-plan.md` §2 M16e, "measured 2026-09-28"); `HANDOFF.md` "still owed" names it. the ledger strips
the variable (`tools/gate.py::run_phase`), so no local forced run is recordable today. fuzz at x10 on the whole
suite: 4899 s locally (CLAUDE.md, 2026-09-29) against a 180-min (10800 s) timeout; `tests/test_backend.py` alone at
x10 was 387.5 s (2026-09-28, loaded).

### option E1: as built

* pros: zero CI cost; the differential already fuzzes every primitive at drawn points in every job, and the
  set-level `repr` comparison covers the 30 methods, the binary ops and newton in both classes.
* cons: the integration paths not in the set-level differential (the itf1788 vectors, `DecoratedInterval`, the
  reverse ops' full surface, the solver in n variables, `ieee1788.py`, numpy interop, the README doctests) run on
  gmpy2 only in one local run from 2026-09-28; a regression in the dispatch wiring that the differential's own
  draws do not reach stays unseen until a user with the variable set finds it. the "forced never falls back" rule
  (`backend.py::_load`) was designed to make such a job meaningful, and no job uses it.
* when better: the backend stays opt-in and nobody reports using it; CI minutes are scarce.

### option E2: one gate job with `INTERVALS_BACKEND=gmpy2` (python 3.13, ubuntu)

* meaning: a fourth matrix entry (or a separate job) with `env: INTERVALS_BACKEND: gmpy2`, `pip install -e
  ".[test]" numpy`, `python -m pytest -q`; a selection assert is free: forced raises `ImportError` at import if
  gmpy2 is not taken (`::test_forced_gmpy2_never_falls_back`), so the job cannot pass on the pure path; a
  one-line step `python -c "import intervals.backend as b; assert b.name() == 'gmpy2'"` makes the intent visible.
  `tests/test_backend.py::test_use_switches` and `::test_use_restores` read `before`/`saved` and restore to it, so
  they pass under a gmpy2 default (the 2026-09-28 forced run passed the whole suite: 4805 + 18246).
* cost: one more parallel job, ~12-15 runner-minutes per push (the pure gate is ~12 min, `CLAUDE.md` push §3);
  wall-clock unchanged (`strategy.fail-fast: false`, jobs run in parallel). the hypothesis `ci` profile is
  derandomized, so this job sees the same examples as the pure one: a pure differential, not a second fuzz.
* pros: the whole suite on a second MPFR build (the linux PyPI wheel) on every push; `HANDOFF.md`'s "only one
  local whole-suite run" closes for good; a prerequisite for ever answering Q16(a) with `auto`.
* cons: a red there is a red for a non-default path; the owner's rule is that a push is watched to the end
  (`CLAUDE.md` push), so one more verdict to read. locally: `tools/gate.py` would need a `gate:gmpy2` phase (or
  a `backend=gmpy2` flag on `phase_spec`) that *keeps* the variable and records under a distinct phase name, so
  that the ledger can say whether this code was run forced; `tools/prepush.sh` need not run it before every push
  (the ledger could require it only when `intervals/elementary.py`, `ops.py`, `_gmpy2.py` or `backend.py`
  changed since the last forced green, as the CORE-MATH rule gates on the evaluator's files). about 15 min of
  laptop per such push.
* when better: whenever the backend is shipped at all (Q16(f) yes), and mandatory if Q16(a) ever goes `auto`.

### option E3: fuzz under gmpy2 too (a second fuzz job, or a backend matrix in `fuzz.yml`)

* cost: ~80 runner-minutes per push to master (the x10 run is ~80 min), within the 180-min timeout as a
  separate job; and, by the owner's rule that the same fuzz runs locally before the push (`tools/prepush.sh`),
  another ~80 min of laptop per push, doubling prepush.
* benefit: low beyond E2. the fuzz's new inputs test the *oracles* against the library; under gmpy2 the
  differential's x10 draws already fuzz the primitives against the pure path in the existing job. a whole-suite
  gmpy2 fuzz would find only a wiring fault at a random point that neither the derandomized gmpy2 gate nor the
  differential's draws reach: a thin slice for a doubled prepush.
* when better: only if Q16(a) goes `auto` and the owner wants the fuzz to run the default path users get; even
  then, alternating the backend between pushes (one job, `INTERVALS_BACKEND` chosen by run parity) keeps the
  cost flat.

### option E4: a scheduled or dispatch-only gmpy2 job

* ruled out by policy: nobody reads a scheduled run's mail (`fuzz.yml` header, owner 2026-09-29).

### recommendation

**E2, not E3**: one forced-gmpy2 gate job on 3.13 in `ci.yml`, with a `gate:gmpy2` phase in `tools/gate.py` so
the ledger can record a local forced run, required before a push only when the evaluator's files changed.
confidence **medium-high (~75%)**. the cost is ~15 runner-minutes in parallel and nothing in wall-clock; the
benefit is the one thing the HANDOFF says is owed and that Q16(a) cannot be reconsidered without.

what would change my mind: CI minutes being metered and scarce (then E1 and a manual forced run at each
release); or Q16(a) going `auto` (then E2 is mandatory and E3's alternating form becomes worth it).

cost of changing later: workflow and tooling only, no API; the same before and after 2.0. (CLAUDE.md: keep CI
config changes out of feature branches unless that is the task; this would be its own commit.)

## Q16(f) 2.0 or later

**the question.** whether the backend is part of the 2.0 release. the item sat under `v2-plan.md` "later (not in
v2.0)"; it was built as M16e when the owner said "get the rest of h3 done" (2026-09-27). **built:** opt-in, merged,
735 tests, documented in the README ("a faster backend, optional") and the `[fast]` extra exists.

### option F1: ship in 2.0 as the opt-in it is, in the release notes

* pros: it is merged, reviewed (three lenses, 0 wrong doubles), sabotaged (every row red), documented; it changes
  nothing unless selected. what 2.0 then commits to is small and already the right shape: the variable's name and
  its four values (unset/`python`, `gmpy2` forced, `auto` silent fallback), the `[fast]` extra name,
  `intervals.backend.name()`, and the promise "same doubles and flags". each of those is what one would design
  again.
* cons: the README's speed sentence ("a few times for the elementary functions at a float, less at set level")
  has no quiet-machine number behind it (`HANDOFF.md` "still owed": re-measure before any number goes in);
  today's probe (§2, loaded) supports the qualitative sentence as written. the `auto` value is a commitment to
  silent fallback semantics; fine, but worth one sentence in the notes.
* when better: always, given opt-in. if Q16(a) were `auto`, F1 would need E2 first.

### option F2: ship the code, keep it out of the release notes until Q16(a) is answered

* pros: no public promise yet about the default.
* cons: the README already documents it, so "out of the notes" is a fiction; the default question does not need
  answering to ship an opt-in (§Q16(a): flipping later changes no result). half-measures confuse more than
  either clean answer.

### option F3: hold it out of 2.0 (revert or branch)

* pros: a smaller 2.0 surface.
* cons: work to remove a merged, tested stream; the `[test]` extra and the differential go with it; users with
  the variable set get nothing; nothing is gained in soundness because the default is already pure.
* when better: only if the owner distrusts the identity argument itself; §1's evidence says there is no reason.

### recommendation

**F1.** confidence **high (~85%)**. release-note the opt-in with the qualitative speed sentence only (no number
until `tools/backend_speed.py` is run on a quiet machine and date-stamped), and the `auto` fallback semantics in
one line.

what would change my mind: a disagreement found between the backends on any build before the release (none in
~41k + 118k + 1200 reviewed answers, 20000 probed here, every gate run of the differential since 2026-09-28).

cost of changing later: the names (`INTERVALS_BACKEND`, its values, `[fast]`, `backend.name()`) become
deprecation-cycle items after 2.0; before it, free. the default (Q16(a)) is *not* among them.

## summary table

| sub-question | built | recommendation | confidence | change cost after 2.0 |
|---|---|---|---|---|
| (a) default | opt-in | keep opt-in for 2.0; reconsider `auto` only after (e) runs green a while and `evaluate-box` is tried (A5) | high ~80% | low: identical results, a minor-version note |
| (b) surface | var + `[fast]`; `backend.name()` unexported; no setter | as built, plus one README sentence naming `intervals.backend.name()` for reports; no setter until demand | medium-high ~70% | adding names free; removing needs deprecation |
| (c) non-dyadic | pure | keep pure; "later", conditioned on a measured workload | high ~85% | zero (internal) |
| (d) `[test]` pin | `gmpy2>=2.3,<3` in test; `[fast]` unpinned | keep; also pin `[fast]` to the same window and extend the pin test to it | high / medium-high ~70% | low (patch release) |
| (e) CI | no gmpy2 job | one forced-gmpy2 gate job (3.13) + a `gate:gmpy2` ledger phase; no gmpy2 fuzz job | medium-high ~75% | none (tooling) |
| (f) 2.0 | in, opt-in | ship in 2.0, notes with the qualitative speed sentence only | high ~85% | names become deprecation items |

cross-cutting facts behind the table: both backends gave identical sets in 20000 probed cases (2026-10-03) and in
every tracked differential; the gain is 3-7.5x per elementary primitive, 2.3-2.9x for exp/log/atan at set level,
1.1-1.4x for arithmetic and newton (2026-10-03, loaded laptop, ratios only); `auto`'s window holds two PyPI
releases; the whole suite has run forced once, locally, and the ledger cannot record such a run today
(`tools/gate.py::run_phase` strips the variable).

found on the way, for the owner (not fixed, read-only): `tools/backend_speed.py`'s docstring cites
`h3-records/gmpy2.md`, which is not in the tree.
