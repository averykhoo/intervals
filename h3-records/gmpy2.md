# M16e, H3's second part: the gmpy2/mpfr backend (stream record, 2026-09-28)

the orchestrator merges this file's sections into `v2-plan.md`, `v2-implementation-plan.md`,
`HANDOFF.md` and `README.md`; nothing here edits them. ids: M16e, D24, Q16. branch `h3-gmpy2`, based
on `v2` at `04946af`.

## design

text for `v2-plan.md` "current design". three places change and one bullet is new.

**"arithmetic", the rounding bullet**: replace "that is the tightest float enclosure, so gmpy2/mpfr
would only be faster, not tighter." with:

> that is the tightest float enclosure. the same doubles can come from gmpy2/mpfr, faster (the
> backend, "elementary and step functions" below); it never makes one tighter or looser.

**"elementary and step functions", a new bullet after the values bullet** (the backend):

* **the backend** (M16e, 2026-09-28): which code picks the rounded double, the pure path
  (`python`) or gmpy2/mpfr (`gmpy2`, `intervals/_gmpy2.py`). **the default is the pure path**; the
  environment variable `INTERVALS_BACKEND`, read once at `import intervals`
  (`intervals/backend.py`), selects: unset, `''` or `python` the pure path (gmpy2 never imported);
  `gmpy2` gmpy2, and an `ImportError` at import if it is missing or below the floor (gmpy2 2.3 with
  MPFR 4.2); `auto` gmpy2 if it imports and `2.3 <= version < 3`, else the pure path, silently;
  anything else a `ValueError`. no public setter (rounding is a property of the type, never an
  ambient mode); `intervals.backend.name()` says which, and is not exported from `intervals`
* **the backend's contract: the same doubles and the same flags, faster.** it answers only "which
  double": every decision that is not a rounding (`elementary.exact`, `exact_pow`, `_beyond` and
  the range shortcuts of `rounded_pow`, the pi limits at ±inf, every flag and attainment in
  `functions`, `reverse` and the applicator) runs first and stays pure. it replaces the `_ziv`
  loop of `elementary.rounded`, `rounded_pow`, `rounded_angle` (the `(q, m)` of `functions._angle`)
  and `rounded_inverse_trig` (k = 0), and the rounding of the five `ops.OUTWARD` descriptors (add sub
  mul div reciprocal, keyed on the descriptor object). each answers a float or None (then the pure
  path runs), so a partial backend is correct by construction
* **why one MPFR call is the correctly rounded double**: MPFR is correctly rounded in every direction,
  and a `gmpy2.ieee(64)` context is binary64 exactly (the subnormals applied with the ternary
  value). so the input must be exact: an MPFR function is called only on an mpfr equal to x, built
  at x's own bit length in a private context (x dyadic: every float, every int, `Fraction(3, 8)`).
  declined, hence pure: a non-dyadic x (`1/3`), `log` to a base, `acoth`, `rootn` with `n <= 0` or
  `n >= 2**31` (MPFR's n is a C `unsigned long`, 32 bits on windows), `k pi + ...` with k != 0, the
  `pow{n}` descriptors, an operand past `2**20` bits in its numerator or denominator (past MPFR's
  exponent range, `2**30` on windows, a dyadic flushes to 0 or inf with a ternary value of 0,
  silently). native at a non-dyadic rational: atan, acot and the angles of atan2, as atan2 of two
  exact ints; the hook's mixed operands, as an exact mpq rounded once
* **the backend's own rules**: every mpfr it builds names a context (a bare `mpfr(x)` reads the
  user's global context); `+ 0.0` is every function's last operation, after a sign (MPFR gives
  `-0.0` where the pure path gives `0.0`); a ternary value of 0 on an elementary result is a missed
  exact case and raises `ArithmeticError` as the pure loop does, in every direction (the pure loop
  answers where both ends of its enclosure round alike); a nan (an argument outside the domain)
  raises. untested on free-threaded builds; `backend._use`, the tests' switch, is a module global

**"package layout"**, two lines after `rounding.py`:

        backend.py         which code picks a rounded double: INTERVALS_BACKEND, python (default),
                           gmpy2 or auto; imports _gmpy2 only when selected (M16e)
        _gmpy2.py          the gmpy2/mpfr backend: the same doubles as elementary.py and the
                           outward hook, faster, or None (then the pure path) (M16e)

**"testing"**, a new bullet:

* **the backend differential** (M16e, `tests/test_backend.py`): at every point it draws, each
  primitive three ways, the pure path under `backend._use('python')`, `_gmpy2`'s function directly
  (the same double and sign bit, and None exactly where the module's table says,
  `::declines_rounded` and its kin, written from this design), and the dispatch under
  `_use('gmpy2')` (which sees an argument dropped on the way); 15 edge classes (`::EDGES`, `::HARD`,
  `::_bound_cases`), the whole list again under a hostile gmpy2 global context; the `repr` of every
  set-level method under both backends; `::test_use_switches` guards that the two really ran
  different code. gmpy2 is in `[test]`, so nothing skips. the rest of the suite runs on the pure
  path (the default); the build ran the whole gate once more forced to gmpy2

**"later (not in v2.0)"**: drop ", gmpy2/mpfr as a faster backend for `elementary.py` and the
outward hook (not a tighter one)" and the "owner 2026-09-26: recorded, not now" that follows it (the
numpy clause keeps its own record, M16d).

## decision-log revision

### 2026-09-28 revision: M16e, the gmpy2/mpfr backend (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now" (`HANDOFF.md` H3; plan §2 M15: "gmpy2/mpfr stay out"). built as M16e,
one of M16's five streams. the choices, each the build's default, open for the owner (D24, Q16):
* **the pure path is the default**; gmpy2 only with `INTERVALS_BACKEND=gmpy2` (forced) or `auto`.
  the design had automatic-when-importable; its critique held that the conservative reading wins:
  the pure path is the reference, the local gate and itf1788 then keep checking it, and a user's
  gmpy2 (linked to whatever MPFR their distribution ships) never changes code paths unasked
* **the backend only picks the double** (the contract above), so it is correct by construction
  wherever it declines, and bit-identical where it answers (the differential)
* **declined where one MPFR call is not one rounding**: non-dyadic inputs (but atan2 of two ints and
  the hook's exact mpq), `log` to a base, `acoth`, `rootn` with n < 0, `k pi + ...`; and where MPFR's
  limits bite: `rootn` from `n = 2**31`, an operand past `2**20` bits
* **`gmpy2>=2.3` in `[test]`** (the differential never skips) and a new extra `[fast]`; no CI change:
  every gate job installs `.[test]`, so `tests/test_backend.py` runs everywhere, and the rest of the
  suite runs on the pure path there as here
* **`auto`'s window is `2.3 <= version < 3`** (the series verified); forced takes any gmpy2 at the floor

## D24

the §0 table row:

| D24 | **decided in the build 2026-09-28 (the session's defaults), open for the owner: `HANDOFF.md` Q16.** the gmpy2/mpfr backend (M16e): (a) the default is the pure path; `INTERVALS_BACKEND=gmpy2` forces gmpy2 (ImportError if missing or below 2.3 / MPFR 4.2), `auto` takes it if importable and `2.3 <= version < 3`; (b) public surface: the env var and the `[fast]` extra only; `intervals.backend.name()` not exported, no setter; (c) non-dyadic points stay pure (no mpfr ziv loop), but atan, acot, atan2's angles and the hook's mixed operands; (d) `gmpy2>=2.3` in `[test]`; (e) CI unchanged: the whole suite on the pure path, `tests/test_backend.py` compares both in every job; no gmpy2 fuzz job; (f) ships in 2.0 as an opt-in, or waits under "later" | as built | M16e |

what it blocks: nothing. every answer keeps today's behaviour (the pure path) unless a user sets the
variable; Q16(a) (default automatic) and Q16(f) (in 2.0 or not) are the only ones whose answer
would change what a user without the variable sees, and neither blocks H1.

## M16e

### M16e the gmpy2/mpfr backend: `backend.py`, `_gmpy2.py` (H3's second part; done 2026-09-28)

the owner, 2026-09-27: "get the rest of h3 done" (H3), superseding 2026-09-26's "recorded, not
now". H3's rest is M16, five streams; this is M16e. the design is `v2-plan.md` "elementary and step
functions" (the backend bullets); the choices are D24, open as Q16. here the spec, the exit and the
record.

* **`intervals/backend.py`**: `INTERVALS_BACKEND` read at import (`::_select`: unset, `''`,
  `python` → pure; `gmpy2` forced; `auto`; else `ValueError`), `NAME`, `fast`, `name()`, the version
  floor `::_supported` (gmpy2 `>= 2.3`, a pre-release counting as just before its release; MPFR
  `>= 4.2`; `auto` also `< 3`; an unparsable string unsupported), `::_load` (lazy import of
  `_gmpy2`), `::_use` (the tests' switch)
* **`intervals/_gmpy2.py`**: `rounded`, `rounded_pow`, `rounded_angle`, `rounded_inverse_trig`,
  `outward`; each a float or None. the exact input (`::_operand`, `::_ratio`, `::_int`), the two
  guards (`::_value`: nan, and a ternary value of 0 on an elementary result), `BOUND = 2**20`,
  `ROOTN_LIMIT = 2**31`
* **the dispatches**: `elementary.rounded` after `_beyond`, `rounded_pow` after its two range
  shortcuts, `rounded_angle` after `q == 0 and m == 0`, `rounded_inverse_trig` after its exact k = 0
  case; `ops.outward` keyed on the descriptor object (`ops._FAST_OPS`). each reads `backend.fast` at
  call time. `pyproject.toml`: `fast = ["gmpy2>=2.3"]`, `gmpy2>=2.3` in `test`
* exit: every primitive bit-identical to the pure path (value and sign bit) at drawn points and over
  15 edge classes, None exactly where the table says; the set-level `repr` of every method equal
  under both; the switch, the env var and the version floor pinned; every new property sabotaged
  once and seen red; the gate green on the default (pure) backend, and once more with
  `INTERVALS_BACKEND=gmpy2`

record (2026-09-28):
* **what the build found on its way**, each fixed before the record:
    * the critique's seven blocking items were built as fixes before any code (B1 the `rootn` bound,
      B2 the bound on bits, B4 no fixture: each example computes both under `_use`, B5 the pure
      default, C1 `log(b)`/`rootn(n)`/`pow_rev2` at set level, C3 hard points by construction, C4
      `+ 0.0` last)
    * **the class-15 test at the real bound hung the first run**: the pure path at a tiny x past
      `2**20` bits grows about quadratically for sin, exp, atan and pow (at `2**16` bits: 3.3 s for
      sin, 5.5 s for exp, 2026-09-28, `.scratch` timing), so a million bits would take ~15 min. the
      real-bound test keeps only the calls whose pure path stays cheap (atan and log of a huge int,
      acot of a tiny one, the hook, an angle; 0.12 s at `2**20`); every call runs at a bound
      monkeypatched to `2**12` (`::_bound_cases`, its `cheap` flag). the backend's check itself is
      one `bit_length` comparison
    * **a hostile global context changed by `set_context` does not reach a backend that captured
      the global context object**: the class-13 helper `::hostile_global_context` mutates the global
      context in place instead, so both a bare `mpfr(x)` (S3) and a captured global (S3b) go red
    * `sys.modules['gmpy2'] = None` (the "gmpy2 missing" stub) leaves the key in `sys.modules`: the
      env-var test's expected line says so
    * the backend is NOT faster everywhere at set level (measured below): `.sin()` over a
      multi-interval and newton on a polynomial gain nothing, since `floor_over_pi`, the applicator
      and the kernel dominate there
* **tests** (`tests/test_backend.py`, 717 tests, and `tests/test_elementary.py::test_a_missed_exact_case_raises_instead_of_looping`
  now under `backend._use('python')`):
    * the three-way check per primitive, `::check_rounded`, `::check_pow`, `::check_angle`,
      `::check_inverse_trig`, `::check_outward`, against the table `::declines_rounded`,
      `::declines_pow`, `::declines_angle`, `::declines_inverse_trig`, `::declines_outward`
    * drawn: `::test_rounded_matches_python` (all 30 names over `tests/test_oracle_flint.py::points`),
      `::test_log_base_matches_python`, `::test_rootn_matches_python` (n up to `2**64 + 1`, and
      negative), `::test_pow_matches_python`, `::test_angle_matches_python` (m in 0, ±1, ±2, ±3),
      `::test_inverse_trig_matches_python` (k in 0, ±1, 5), `::test_outward_matches_python`,
      `::test_outward_reciprocal_matches_python`
    * the edge classes `::EDGES` (1 dyadic floats over the range, 2 subnormal results, 3 the `-0.0`
      class incl. `csch(-746)` and C4's `atan(5e-324)` with sign -1, 4 overflow, 5 wide exact inputs,
      6 non-dyadic, 7 infinite x, 8 every `(q, m)` of `functions._angle` and some it never makes, 9
      inverse trig, 10 rootn incl. `2**31 - 1`, `2**31`, `2**32`, `2**64 + 1`, 11 pow, 12 mixed hook
      operands; 174 cases) in `::test_edge_class`, all again in `::test_hostile_global_context` (13, guarded by
      `::test_the_hostile_context_is_hostile`), 14 `::HARD` in `::test_hard_point` (78 points, each
      asserted answered by the backend), 15 `::test_past_the_bound_is_declined_at_a_small_bound` and
      `::test_past_the_bound_is_declined_at_the_real_bound`; `test_oracle_flint.py::EXTREMES` in
      `::test_extreme_point`
    * coverage: `::test_backend_answers_where_it_should` (a point per row of the table),
      `::test_backend_declines_where_it_should`, `::test_power_descriptors_decline`
    * rules: `::test_the_shortcuts_run_before_the_backend`, `::test_the_hook_is_keyed_on_the_descriptor`,
      `::test_missed_exact_case_raises` (all three directions), `::test_a_domain_slip_raises_instead_of_returning_nan`,
      `::test_inputs_it_does_not_know_are_declined`
    * set level: `::test_set_level_matches_unary` (the 30 methods, `reciprocal`, `** 3`, `** -2`,
      `** 2.5`, `rootn` 2 3 -2 -3, `log` to 1/2, 0.25, 3, 2.5), `::test_set_level_matches_binary` (+ -
      * /, atan2, pow, hypot, `sin_rev`, `cos_rev`, `tan_rev`, `pow_rev2`),
      `::test_set_level_matches_newton`, `::test_set_level_matches_newton_sin`; both classes
    * the switch: `::test_use_switches`, `::test_env_var` (11 cases in a subprocess, the variable
      removed from the inherited env), `::test_forced_gmpy2_never_falls_back`, `::test_version_floor`
      (17 cases)
* **measured 2026-09-28** (this laptop, python 3.13, gmpy2 2.3.1 / MPFR 4.2.2; four other M16 streams
  running, so absolute times are loaded): `tests/test_backend.py` 717 passed in 46-49 s
  (`python -m pytest -q tests/test_backend.py`). the gate, two runs (`.scratch/gate.sh`: three
  calls, `tests/itf1788`, then `tests --ignore=tests/itf1788 --ignore-glob=tests/test_[o-z]*.py`,
  then `tests/test_[o-z]*.py intervals README.md`; 3251 + 1554 = 4805 items, the non-vector half):
    * **default (pure), 2026-09-28 08:30-08:48**: `tests/itf1788` 18246 passed in 86.6 s; the
      first group 3251 passed in 540.9 s; the second 1554 passed in 455.7 s; so 4805 in 996.6 s
      (pytest's own times; M15's one call was 482 s on an unloaded laptop, this one ran beside four
      other streams)
    * **`INTERVALS_BACKEND=gmpy2`, 2026-09-28 08:48-09:06**: `tests/itf1788` 18246 passed in 110.8 s;
      3251 passed in 477.4 s; 1554 passed in 464.3 s; so 4805 in 941.7 s. the evidence that the
      backend passes the whole suite (the forced setting raises at import if gmpy2 is not taken).
      the two runs' times are not a speed comparison (load)
    * after the last edit (a docstring), `tests/test_backend.py` re-run: 717 passed in 43.4 s, and
      717 in 39.1 s with `INTERVALS_BACKEND=gmpy2`
* **speed** (`.scratch/speed.py`, pure then gmpy2 back to back under `backend._use`, best of 5 per
  call; 2026-09-28 09:07, after this stream's gate, with other streams' runs on the laptop, so ratios,
  not absolute times. a first run beside the gate had `.sin()` 0.86x and newton on `t**2 - 2`
  0.82x; the second run below has them 1.1x and 1.2x: noise at that size):

| call | pure µs | gmpy2 µs | ratio |
|---|---|---|---|
| `rounded('exp', 0.7, DOWN)` | 85.7 | 17.8 | 4.8 |
| `rounded('exp', 1/3, DOWN)` (declined: pure in both) | 76.4 | 79.7 | 0.96 |
| `rounded('log', 0.7, DOWN)` | 138 | 38.1 | 3.6 |
| `rounded('sin', 0.7, DOWN)` | 145 | 36.3 | 4.0 |
| `rounded('sin', 1e22, DOWN)` | 119 | 15.5 | 7.7 |
| `rounded('atan', 1/3, DOWN)` (atan2 of the ints) | 83.7 | 20.1 | 4.2 |
| `rounded('atan', 2**-30, DOWN)` | 129 | 20.6 | 6.3 |
| `rounded_pow(2, 1/2, UP)` | 107 | 36.7 | 2.9 |
| `rounded_angle(1/3, 1, UP)` | 83.7 | 8.38 | 10 |
| hook `add(0.1, 0.2)` down | 17.1 | 2.76 | 6.2 |
| hook `div(1.0, 3.0)` down | 13.4 | 2.38 | 5.6 |
| hook `add(0.1, 1/3)` down (the mpq route) | 18.4 | 6.76 | 2.7 |

| op | pure ms | gmpy2 ms | ratio |
|---|---|---|---|
| `OutwardMultiInterval`, 3 float pieces, `.exp()` | 0.42 | 0.185 | 2.3 |
| same, `.log()` | 0.699 | 0.198 | 3.5 |
| same, `.sin()` (`floor_over_pi` stays pure) | 0.911 | 0.828 | 1.1 |
| same, `.atan()` | 1.27 | 0.453 | 2.8 |
| `MultiInterval`, 2 float pieces, `.exp()` | 0.561 | 0.261 | 2.1 |
| A + B (3 x 2 float pieces, outward) | 5.77 | 4.8 | 1.2 |
| A * B | 5.75 | 4.66 | 1.2 |
| A / B | 6.55 | 5.15 | 1.3 |
| `newton(t**2 - 2, Outward(-10.0, 10.0))` | 65.3 | 53.1 | 1.2 |
| `newton(sin(t) - t/3, Outward(-10.0, 10.0))` | 345 | 250 | 1.4 |

  so: an elementary function at a float point 3.6-7.7x (10x for an angle of atan2), a declined
  point 1x (one `bit_length` test), the outward hook about 6x per corner (2.7x on the mpq route); at
  set level 2-3.5x for exp, log and atan over a multi-interval, 1.1x for sin, 1.2-1.3x for outward
  `+ * /` and 1.2-1.4x for newton, where the applicator's and the kernel's own python dominate

* **sabotage** (a throwaway harness: each break alone, `.hypothesis` cleared,
  `tests/test_backend.py` and `tests/test_elementary.py::test_a_missed_exact_case_raises_instead_of_looping`
  with `-x` and a 900 s timeout, the files restored and compared; 2026-09-28). the last column is
  the first test to fail under `-x`:

| break | first run | final run: red by |
|---|---|---|
| S1 drop `+ 0.0` (rounded) | red | red: `tests/test_backend.py::test_edge_class[3-rounded-'sin'--5e-324]` |
| S1b drop `+ 0.0` (hook, two dyadic operands) | red | red: `tests/test_backend.py::test_edge_class[3-outward-'mul'--1e-300-1e-300]` |
| S2 a non-dyadic handed to MPFR as mpq | red | red: `tests/test_backend.py::test_edge_class[6-rounded-'exp'-Fraction(1, 3)]` |
| S3 a float built with a bare `mpfr(x)` | red | red: `tests/test_backend.py::test_hostile_global_context[3-outward-'mul'--1e-300-1e-300]` |
| S3b the wide context is the global one | red | red: `tests/test_backend.py::test_hostile_global_context[1-rounded-'log'-5e-324]` |
| S4 a dyadic Fraction built at 53 bits | red | red: `tests/test_backend.py::test_edge_class[5-rounded-'log'-Fraction(...)]` (`(2**100 + 1) / 2**100`) |
| S4b an int built at 53 bits | red | red: `tests/test_backend.py::test_edge_class[5-rounded-'sin'-1152921504606846977]` |
| S5 subnormalize off | red | red: `tests/test_backend.py::test_edge_class[2-rounded-'sinh'-5e-324]` |
| S6 angle m = 1 as `atan2(n, -d)` | red | red: `tests/test_backend.py::test_edge_class[8-angle-Fraction(1, 3)-1]` |
| S7 angle accepts m = 2 with q > 0 | red | red: `tests/test_backend.py::test_edge_class[8-angle-Fraction(1, 3)-2]` |
| S7b angle at q = 0: direction not reversed for m < 0 | red | red: `tests/test_backend.py::test_edge_class[8-angle-0--1]` |
| S8 hook, mixed operands: each rounded to 53 bits first | red | red: `tests/test_backend.py::test_edge_class[6-outward-'add'-0.1-Fraction(1, 3)]` |
| S9 `rounded` always declines | red | red: `tests/test_backend.py::test_edge_class[1-rounded-'exp'-0.7]`; alone, `::test_backend_answers_where_it_should[rounded-args0]` |
| S10 rootn n < 0 as `1 / rootn(m, -n)` | red | red: `tests/test_backend.py::test_edge_class[10-rounded-'rootn'-2.0--2]` |
| S11 inverse trig: sign < 0 keeps the direction | red | red: `tests/test_backend.py::test_edge_class[3-inverse-'atan'-5e-324--1-0]` |
| S12 acot as `atan2(1, n/d rounded)` | red | red: `tests/test_backend.py::test_edge_class[6-rounded-'acot'-Fraction(1, 3)]` |
| S13 drop the `rc == 0` guard | red | red: `tests/test_backend.py::test_missed_exact_case_raises` |
| S14 `_use` a no-op | red | red: `tests/test_backend.py::test_the_hook_is_keyed_on_the_descriptor`; alone, `::test_use_switches` |
| S15 the hook captures the backend at build | red | red: `tests/test_backend.py::test_the_hook_is_keyed_on_the_descriptor`; alone, `::test_use_switches` |
| S16 forced gmpy2 falls back when missing | red | red: `tests/test_backend.py::test_forced_gmpy2_never_falls_back[...-gmpy2]` (blocked) |
| S16b forced gmpy2 falls back below the floor | red | red: `tests/test_backend.py::test_forced_gmpy2_never_falls_back[...-2.2.9]` |
| S17 the version floor compares strings | red | red: `tests/test_backend.py::test_version_floor[2.10.0-MPFR 4.2.2-True-True]` |
| S17b a pre-release counts as its release | red | red: `tests/test_backend.py::test_version_floor[2.3.0rc1-MPFR 4.2.2-False-False]` |
| S17c no ceiling for auto | red | red: `tests/test_backend.py::test_env_var[auto-...3.0.0...]` |
| S17d no MPFR floor | red | red: `tests/test_backend.py::test_env_var[auto-...MPFR 4.1.1...]` |
| S17e an unparsable version accepted | red | red: `tests/test_backend.py::test_forced_gmpy2_never_falls_back[...-two point three]` |
| S17f auto takes gmpy2 3 (auto read as forced) | red | red: `tests/test_backend.py::test_env_var[auto-sys.modules['gmpy2'] = None-...]` |
| S18 the dispatch moved before `_beyond` | **green** | red: `tests/test_backend.py::test_the_shortcuts_run_before_the_backend` |
| C1 the dispatch drops `base` | red | red: `tests/test_backend.py::test_edge_class[10-rounded-'rootn'-2.0-2]`; set level alone, `::test_set_level_matches_unary[log base 0.25]` and `::test_set_level_matches_binary[pow_rev2]` |
| C2 decline NEAREST | red | red: `tests/test_backend.py::test_edge_class[1-rounded-'exp'-0.7]`; alone, `::test_backend_answers_where_it_should[rounded-args0]` |
| C4 `+ 0.0` before the sign | red | red: `tests/test_backend.py::test_edge_class[3-inverse-'atan'-5e-324--1-0]` |
| B1 drop the rootn n bound | red | red: `tests/test_backend.py::test_edge_class[10-rounded-'rootn'-2.0-2147483648]` |
| B2 drop the bound on bits | red | red: `tests/test_backend.py::test_past_the_bound_is_declined_at_a_small_bound` |
| N1 the nan guard dropped | red | red: `tests/test_backend.py::test_a_domain_slip_raises_instead_of_returning_nan` |
| N2 any input taken as a Fraction | red (see below) | red: `tests/test_backend.py::test_inputs_it_does_not_know_are_declined` |
| N3 `''` not read as unset | red | red: `tests/test_backend.py::test_env_var[--python python False False]` |
| N4 the hook keyed on the name | red | red: `tests/test_backend.py::test_the_hook_is_keyed_on_the_descriptor` |
| N5 inverse trig answers k != 0 | red | red: `tests/test_backend.py::test_edge_class[9-inverse-'acos'--1-1-0]` |
| N6 the hook: a zero divisor not declined | red | red: `tests/test_backend.py::test_inputs_it_does_not_know_are_declined` |
| N7 `_gmpy2` imported eagerly | red | red: `tests/test_backend.py::test_env_var[None--python python False False]` |
| N8 the pure missed-exact guard dropped | red | red: `tests/test_elementary.py::test_a_missed_exact_case_raises_instead_of_looping` |

the one green in the first run was a gap, as the design predicted: moving the dispatch before
`_beyond` changes no value (MPFR agrees with `_beyond`, which is why no differential sees it), only
cost and provenance. closed by `::test_the_shortcuts_run_before_the_backend` (a spy on
`_gmpy2.rounded` and `_gmpy2.rounded_pow`, never called for `exact`'s values, `_beyond`'s, the pi
limits at ±inf or `rounded_pow`'s range shortcuts), and the break re-run red. N2's first run was red
for the wrong reason (the replacement left a syntax error: red at collection); rewritten as intended
and re-run red by its test. S9, S14, S15, C1 and C2 were re-run against their named guard alone,
since under `-x` an earlier test caught them first; each went red there too.

## readme

text for `README.md`, "what it does", a bullet after **rounding** (doctested with the file; the
example is deterministic and prints no backend name, so it holds under either backend):

* **a faster backend, optional** (M16e): `pip install intervals[fast]` adds gmpy2, and
  `INTERVALS_BACKEND=gmpy2` (or `auto`: gmpy2 when it imports) picks it at `import intervals`. it
  computes the same doubles as the default pure-python path, only faster (a few times for the
  elementary functions at a float, less at set level), and changes no flag, so every result is
  the same either way:

  ```python
  >>> from intervals import OutwardMultiInterval
  >>> OutwardMultiInterval(0.5, 2.0).exp()
  OutwardMultiInterval.parse('(1.648721270700128, 7.38905609893065)')
  >>> OutwardMultiInterval(0.1) + 0.2
  OutwardMultiInterval.parse('(0.3, 0.30000000000000004)')
  ```

and in "layout", after `solver`: "`backend` and `_gmpy2` (the optional gmpy2 backend)". in "tests":
"needs `pytest`, `hypothesis`, `python-flint` and `gmpy2`".

## Q16

owner questions, each with the default built (D24):

* **Q16(a) automatic or opt-in.** built: opt-in. unset is the pure path; `INTERVALS_BACKEND=auto`
  takes gmpy2 when it imports (`2.3 <= version < 3`), `gmpy2` forces it. alternative: `auto` as the
  default, which gives every user who has gmpy2 (sympy's and mpmath's often do) the speed, and puts
  their MPFR build on the path unasked
* **Q16(b) public surface.** built: the variable and the `[fast]` extra (a new name) only;
  `intervals.backend.name()` importable, not exported from `intervals`; no setter (it would read like
  the ambient rounding mode the plan rules out). alternative: export `backend_name()` for bug reports
* **Q16(c) the non-dyadic points.** built: pure (a user's `Fraction(1, 3)`, `log` to a base,
  `pow_rev2`'s `log_t v`, `acoth`, `rootn` with n < 0, the periodic reverse ops' `k pi + f(v)`),
  except atan, acot, atan2's angles and the hook's mixed operands (native, one rounding).
  alternative: a second part, an mpfr ziv loop for a monotone f at a bracketed x; not measured
* **Q16(d) gmpy2 in `[test]`.** built: yes, so the differential never skips. alternative: only in CI
  jobs that ask for it, with `tests/test_backend.py` skipping locally (a test that passes by skipping)
* **Q16(e) CI and fuzz.** built: no workflow change; every gate job runs the whole suite on the pure
  path and `tests/test_backend.py`'s differential in-process, `fuzz.yml` likewise (so its ×10 run
  fuzzes the differential too). alternative: a gate job with `INTERVALS_BACKEND=gmpy2` (the whole
  suite on MPFR in CI, as this build ran it once locally), and a selection assert in each job
* **Q16(f) 2.0 or later.** the item sat under `v2-plan.md` "later (not in v2.0)". built as an opt-in
  that changes nothing unless selected, so it can ship in 2.0 (H1) as is; alternative: keep it out
  of the 2.0 release notes until Q16(a) is answered
* not questions, recorded: `round_rational` has no gmpy2 path (the design measured 1.0-2.1x on a call
  of a few µs, 2026-09-27); `floor_over_pi` and `compare` stay pure (they decide integers and signs),
  which is why `.sin()` gains nothing at set level

## still owed

* the whole suite under `INTERVALS_BACKEND=gmpy2` exists only as this build's local run (dated above);
  no CI job runs it (Q16(e))
* the backend is verified only with gmpy2 2.3.1 / MPFR 4.2.2 on windows (python 3.13). CI's linux jobs
  run `tests/test_backend.py` with the PyPI wheel; no other MPFR build has been run
* free-threaded builds: untested (the three contexts are shared module objects; `backend._use` is a
  global)
* the pure change the design noted (`applicator.evaluate_box` evaluates a float corner's exact value
  three times under `OUTWARD`; passing `fn`'s value into the hook would cut it to one) is not built;
  it may be worth as much for arithmetic as the backend, with no dependency
* the speed numbers were taken beside four other streams' runs; a quiet-machine re-measure is owed
  before any number goes into the README
