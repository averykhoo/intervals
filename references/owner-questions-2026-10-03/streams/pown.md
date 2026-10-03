# owner answers: pown build (Q17 (c)/(e), Q18 (ii), m14b-open hex, CORE-MATH pown sample)

> **changed at the merge (2026-10-04, `87e6ea3`)**: this stream built Q17's over-limit result as rounded to
> nearest in `MultiInterval`. the accepted report's option (c) is the tightest open float enclosure in both
> classes, so the session gave the nearest pown descriptor rounding hooks (a float corner to nearest, an exact
> corner past the limit outward): `M(2) ** 2 ** 60` is `(MAX, inf)`, as `M(2) ** M(2 ** 60)`. where this record
> says "to nearest the value rounded to nearest" for an exact corner, read the enclosure.

agent worktree: C:\Users\user\PycharmProjects\intervals\.claude\worktrees\agent-ad7ab118e02eb5794
branch: worktree-agent-ad7ab118e02eb5794 (fast-forwarded from 912558b to master 09435ca on 2026-10-03,
since the report commit was not in the worktree's starting point)

## step 0, 2026-10-03: reading
* read CLAUDE.md, testing skill, references/owner-questions-2026-10-03/pown.md and README.md
* probe (2026-10-03): every exact-operand rounding in the library today is the open enclosure in BOTH
  classes: `M(1000).exp()`, `M(100001).exp2()`, `M(2) ** M(2 ** 60)` are all `(MAX, inf)` in M and O;
  `M(2).sqrt()` is the open one-ulp piece. the brief (and the report's (c) text, and the README summary
  "(nearest: rounded)") says pown of exact operands past the limit is ROUNDED TO NEAREST in
  MultiInterval. that contradicts (1) the plan's "an exact operand never loses its true value" clause
  (lines 323-325), which the report itself cites as the invariant (c) restores, and (2) the report's own
  proposed pin `A ** n == A ** M(n)` on exact points, in the nearest class: `M(2) ** 2 ** 60` would be
  `[inf]` while `M(2) ** M(2 ** 60)` is `(MAX, inf)`. SURPRISE for the owner. built as the brief says
  (nearest in M); the pin `A ** n == A ** M(n)` on exact points is therefore an outward-class pin.

## step 1, 2026-10-03: ops._NotADouble's proof re-derived for a general Fraction corner (before the build)

setting: x = ±num/den in lowest terms (num, den > 0 coprime), n a nonzero int, the shared bound
B = |n| max(bitlen(num), bitlen(den)) > L, v = |x| ** n. claim: v is not a rounding breakpoint, so the
rounding hooks' ziv loop (`elementary.rounded_pow`) ends, no rounded end (no double, no ±inf) equals v,
and attainment against the marker (equal to nothing) is the exact answer.

breakpoints. directed rounding (DOWN, UP) changes value only at doubles (0 and MAX included; above MAX,
DOWN is MAX and UP inf throughout). nearest changes value at the midpoints of adjacent doubles, at
2**-1075 (between 0 and the least subnormal) and at 2**1024 - 2**970 (between MAX and the overflow).
every nonzero breakpoint b is dyadic, `b = M 2**e` with M odd, `M < 2**54` (a double's odd part is
< 2**53; a midpoint inside a binade is `(2k+1) 2**(e-1)` with 2k+1 < 2**54; across a binade boundary
`(2**54 - 1) 2**(e-54)`; the overflow threshold `(2**54 - 1) 2**970`), and `2**-1075 <= |b| < 2**1024`.

excluded before the marker: x = 0 (exact_pow answers 0; pown answers 0 or the pole), |x| = 1 (answers
±1 for every n). so v is a nonzero rational, `v = num**n / den**n` (n > 0) or `den**k / num**k` (n < 0,
k = |n|), still in lowest terms (powers of coprime ints are coprime).

case A, v not dyadic (its denominator in lowest terms has an odd prime factor): v is no breakpoint, all
of which are dyadic. this is n > 0 with den not a power of two, or n < 0 with num not a power of two.

case B1, v a power of two: then |x| = 2**e, e != 0, and max(bitlen) = |e| + 1 <= 2|e|, so
`B <= 2 |e n|`, so `|e n| >= B/2 > L/2`. with L >= 2150, |e n| > 1075: v >= 2**1076 or v < 2**-1075,
outside the breakpoints' range. (this is the docstring's first premise, `limit >= 2 * 1075`.)

case B2, v dyadic with odd part P >= 3. write m for the odd part of the numerator side (num for n > 0,
den for n < 0), m >= 3 (m = 1 would be B1), P = m**k. suppose v were a breakpoint: then P < 2**54,
so k (bitlen(m) - 1) < 54 (as m**k >= 2**(k (bitlen(m)-1))), so k < 54 (bitlen(m) >= 2) and
k bitlen(m) < 108. two shapes of x (lowest terms forbids a factor 2 on both sides):
* the dyadic side is a power of two times m on the numerator side only, |x| = m 2**t or 1/(m 2**t),
  t >= 0: v = m**k 2**(±t k), and v < 2**1024 or v >= 2**-1075 forces t k <= 1024 + 54 (the odd
  part's bits count toward the exponent the other way), so B = k (bitlen(m) + t) < 108 + 1078 < 1200;
* |x| = m / 2**d or 2**d / m, d >= 1, m odd: v = m**±k 2**(∓d k); v in range forces
  d k <= 1075 + 54, so B = k max(bitlen(m), d + 1) < max(108, 1129 + 54) < 1200.
either way B < 1200 <= L: a contradiction. so v is no breakpoint.

so the proof holds for every rational corner, whatever its bit length, and needs only L >= 2150
(B1) and L >= 1200 (B2). the docstring's floor of 36550 came from bounding a float's max bitlen by 1075
and asking 3 ** (L/1075 + 1) > 2 ** 54; for a general Fraction that bound is not available, but the
joint use of the range (v must lie in [2**-1075, 2**1024)) replaces it and gives the weaker floor
above. the existing import check (36550) is stronger than needed and is kept; it now also checks
EXACT_RESULT_LIMIT. the injectivity sentence ("on one side of zero, or an odd n > 0, x ** n is
injective on a box") holds for any reals.

what the proof does NOT give (a SURPRISE, recorded for the owner): termination within
`elementary._MAX_PRECISION` (2**22 bits). ziv must separate v from the nearest breakpoint, and for a
rational v with an S-bit denominator that distance can be as small as about 2**-(S + 1130) relative.
for a float corner S <= |n| 1075 but hard cases sit within ~2**-128 (Lefevre / CORE-MATH), so the loop
ends at 128-256 bits. for an exact corner past EXACT_RESULT_LIMIT the operand itself can be crafted
close to a breakpoint (an exact Newton iterate `3 + 2**-3000000` squared sits 2**-2999997 from 9.0):
ziv then needs millions of bits (slow beyond use, and past 2**22 bits it raises ArithmeticError), where
the old exact build answered in milliseconds. measured below after the build.

## step 2, 2026-10-03: the build (Q17 (e) + Q18 (ii)), library side
* `intervals/elementary.py`: `EXACT_RESULT_LIMIT = 1 << 22` (exact operands; pown, pow, exp2/exp10
  share it) beside `EXACT_POWER_LIMIT = 100000` (float corners, a speed choice). new
  `::exact_power_bits(x, n)` (|n| * max bitlen of num/den; 0 for x in 0, ±1), the one measure all
  limits use. `::exact_pow(x, y, limit=EXACT_POWER_LIMIT, too_long=None)`. `::exact(name, x, base,
  limit=EXACT_RESULT_LIMIT)`: exp2/exp10's limit is now in result bits (`exact_power_bits(2 or 10, x)`),
  no longer `|x| > 100000`; `::EXP_BASES`. `::rounded` calls `exact` with EXACT_POWER_LIMIT (it is only
  reached after the caller's own `exact` said None, so it never builds a long power for nothing).
* `intervals/errors.py::PowerLimitWarning` (IntervalWarning), ignored by default (filter appended like
  EmptySetPropagation/DomainClipped); exported from `intervals`.
* `intervals/applicator.py::Unbuilt` (marker base) and `::evaluate_box`: a corner whose `fn` value is an
  `Unbuilt` goes through the rounding hooks even with exact operands. only pown returns one.
* `intervals/ops.py`: `::_NotADouble(Unbuilt)` docstring restates the proof for any rational corner;
  `::_check_marker_premises(limit, name)` now runs on both limits; `::_too_long`; `::_power_descriptor`
  one rule: float corner exact (cached `float_exact`) within EXACT_POWER_LIMIT then `round_rational`,
  else `rounded_pow`; exact corner exact within EXACT_RESULT_LIMIT else the marker (outward: hooks give
  the open enclosure) / `rounded_pow(.., NEAREST)` (nearest). `_FLOAT_EXACT_EXPONENT` and the libm
  `float ** int` route for floats are gone from `_power_descriptor` (`_exact_power_descriptor` keeps its
  float branch, used now only by the tests' oracle `ops.outward(ops._exact_power_descriptor(n))`).
  `::power` emits PowerLimitWarning once per call when an exact finite end is past the limit.
* `intervals/functions.py::pow_` / `::_power_box` / `::_power_corner`: exact operands build with
  EXACT_RESULT_LIMIT (was 100000), a float operand keeps EXACT_POWER_LIMIT; a rational-but-too-long
  corner of exact operands is collected and `pow_` warns once. `::_Function.end` passes the limit by
  operand kind; exp2/exp10 of an exact int past the limit sets `too_long` and `apply` warns once.
  reverse `pow_rev1` reuses `_power_box` (no `too_long` list): it shares the new limit, silently.
* probes (2026-10-03, .scratch/pown/probe1.py, all < 1 ms unless noted): `M(2) ** 2 ** 60` = `[inf]`,
  `O(2) ** 2 ** 60` = `(MAX, inf)`, `O(0.5, 2) ** 2 ** 40` = `(0.0, inf)`, `O(Fraction(1, 3)) ** 2 ** 40` =
  `(0.0, 5e-324)`, `M(2) ** 1e300` = `[inf]`, each with the warning; `M(3) ** M(70000)` now the exact
  110948-bit int (was `(MAX, inf)`), `M(2) ** M(50001)` and `M(10) ** M(25001)` exact ints; at the shared
  boundary `M(2 ** 21).exp2()`, `M(2) ** 2 ** 21` exact (2097153 bits, ~20 ms) and `M(2 ** 21 + 1).exp2()`,
  `M(2) ** M(2 ** 21 + 1)` = `(MAX, inf)`, `M(2) ** (2 ** 21 + 1)` = `[inf]` (nearest), `O(..)` = `(MAX, inf)`;
  `M(3) ** 2 ** 21` (3.3M bits) 0.73 s; `M(1.0026606152364441) ** 13` = `[1.03514557277232]` (libm gave
  ...7723202) and `M(1.3811118839148833) ** 26` = `[4425.378458811315]` (libm ...314).

## step 3, 2026-10-03/04: fmt hex, Cut repr, coremath pown, docs, tests
* `intervals/fmt.py`: `::format_value` writes an int part python refuses in decimal (try `str`, on
  ValueError `hex`), a Fraction as `p/q` with each part so; `::_NUMBER` reads `0x[0-9a-f]+` (case-blind,
  never followed by `.`, so `0x1.8p1` is refused rather than read as `0x1` then `.8`; the grammar takes
  two bare numbers as a piece), alone or as either part of `p/q`; `::parse_value` / `::_parse_int`:
  hex via `int(s, 16)`; a decimal past python's limit still raises ValueError (python's guard kept,
  per the report's (4) and not (2)), now naming the hex form. `intervals/cuts.py::Cut.__repr__` via
  `::_value_repr` (hex past the limit, still a python literal).
* `tests/conftest.py::_pretty_fraction`: hypothesis writes an example's arguments eagerly with its own
  printer, which writes huge ints in hex already but a Fraction by `repr` (raises): registered a
  Fraction printer. hypothesis also reprs strategies, so test_fmt makes huge values by `.map` only.
  SURPRISE worth recording: without the printer, `@example(None, Fraction(-1, 10 ** 4300))` fails
  before the test body runs.
* `tools/coremath.py`: `DERIVED = {'pown': 'pow'}`, `FILES` (the pinned files: fetch/pin iterate it),
  `FUNCTIONS` gains pown; `inside('pown')`: n integral, n != 0, x != 0; `ours('pown')`: the nearest
  descriptor's `fn` / the outward descriptor's hooks; `CACHE` honours `INTERVALS_COREMATH_CACHE` (read
  the main checkout's cache read-only from a worktree). `tests/coremath/pown.tsv`: 2857 rows (731
  negative bases, |n| from 1 to past 1e308), `sample --check` rc 0, the 24 other .tsv files unchanged by
  the resample (git status). measured 2026-10-03: 0 wrong on the new code, 54 calls past 64 bits
  (DEEP['pown'] = 27, half, the file's convention), and python's `x ** n` differs from MPFR's nearest on
  243 of the 2857 rows on this laptop.
* README (values, power, functions, warnings, new "text form" bullet, the 1788 table's numbers and
  rounding rows), v2-plan "current design" (rounding's "never rounded", power, values, representation:
  text); no decision-log entry, no D row, HANDOFF untouched.
* existing tests changed: `test_applicator::test_package_exports_unchanged` (+PowerLimitWarning);
  `test_elementary::test_exact` (exp2 row moved to the new limit, exp10 row added),
  `::test_exact_pow` (+1 row), new `::test_the_two_power_limits`; two `exact` monkeypatches take
  `limit` (test_elementary, test_backend); `test_oracle_flint::_check_point` ignores PowerLimitWarning
  around the exact-operand set call (hypothesis drew x = 2097153.0 = 2**21 + 1, exp2 of its exact
  value is now past the limit and warns) and `::test_pow_against_arb`'s oracle uses
  EXACT_RESULT_LIMIT; `test_extreme_floats::test_nearest_class_rounds_the_exact_result_to_nearest`:
  pow's carve-out removed (Q18).
* full suite on the build before these test edits (2026-10-04, plain pytest, 886 s): 7 failed, 33825
  passed; the 7 were exactly the rows above (exports, 2 monkeypatch signatures, exp2 limit row, 3 flint).

## step 4, 2026-10-04: sabotage of the new code

`.scratch/sabotage.py` (worktree): one exact string replaced, `__pycache__` cleared, PYTHONDONTWRITEBYTECODE=1, restored by copy and byte-compared after each break.

| break | pin run | verdict | pytest summary |
|---|---|---|---|
| S1 power never warns | `tests/test_pown_huge.py::test_the_power_limit_warning` | RED | 2 failed, 4 passed in 0.20s |
| S1 power never warns | `tests/test_pown_huge.py::test_exact_corners_at_the_limit` | RED | 1 failed in 0.23s |
| S2 power always warns | `tests/test_pown_huge.py::test_no_power_limit_warning` | RED | 3 failed, 5 passed in 0.32s |
| S3 hex token may be followed by a point | `tests/test_fmt.py::test_hex_floats_and_bad_hex_are_refused` | RED | 1 failed, 5 passed in 0.11s |
| S4 an Unbuilt corner skips the hooks | `tests/test_pown_huge.py::test_exact_corners_at_the_limit` | RED | 1 failed in 0.27s |
| S4 an Unbuilt corner skips the hooks | `tests/test_pown_huge.py::test_an_exact_corner_past_the_limit_inside_the_float_range` | RED | 1 failed in 0.58s |
| S5 the limit test off by one | `tests/test_pown_huge.py::test_exact_corners_at_the_limit` | RED | 1 failed in 0.13s |
| S5 the limit test off by one | `tests/test_pown_huge.py::test_one_limit_for_pown_pow_exp2_exp10` | RED | 1 failed in 0.21s |
| S6 pow keeps the float limit for exact operands | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points` | RED | 4 failed, 6 passed in 0.19s |
| S6 pow keeps the float limit for exact operands | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points` | RED | 4 failed, 1 passed in 0.32s |
| S7 exp2/exp10 limit on the exponent again | `tests/test_pown_huge.py::test_one_limit_for_pown_pow_exp2_exp10` | RED | 1 failed in 0.18s |
| S7 exp2/exp10 limit on the exponent again | `tests/test_elementary.py::test_the_two_power_limits` | RED | 1 failed in 0.13s |
| S8 nearest float corner from libm again | `tests/test_pown_huge.py::test_nearest_is_correctly_rounded` | RED | 4 failed in 0.16s |
| S8 nearest float corner from libm again | `tests/test_coremath.py::test_the_sample_is_correctly_rounded[pown]` | RED | 1 failed in 0.31s |
| S8 nearest float corner from libm again | `tests/test_pown_huge.py::test_nearest_against_the_exact_power` | RED | 1 failed in 0.24s |
| S9 Cut repr plain | `tests/test_fmt.py::test_past_the_int_str_limit` | RED | 1 failed in 0.80s |
| S10 format_value decimal only | `tests/test_fmt.py::test_past_the_int_str_limit` | RED | 1 failed in 0.15s |
| S10 format_value decimal only | `tests/test_fmt.py::test_round_trip_extreme` | RED | 1 failed in 0.30s |
| S10 format_value decimal only | `tests/test_fmt.py::test_repr_round_trip` | RED | 2 failed in 0.40s |
| S10 format_value decimal only | `tests/test_fmt.py::test_parse_value_round_trip` | RED | 1 failed in 0.27s |
| S11 parse reads no hex | `tests/test_fmt.py::test_parse_hex` | RED | 6 failed in 0.31s |
| S11 parse reads no hex | `tests/test_fmt.py::test_spellings` | RED | 1 failed, 1 warning in 2.37s |
| S12 the warning not ignored by default | `tests/test_pown_huge.py::test_the_power_limit_warning_is_ignored_by_default_and_can_be_an_error` | RED | 1 failed in 0.32s |

12 breaks, each restored ok. not sabotaged: the import check on EXACT_RESULT_LIMIT (an import-time refusal; `test_the_marker_proof_premises` has its row).

## step 5, 2026-10-04: the new pins on the OLD code (09435ca)

old tree = `git archive 09435ca` + the new test files, `tests/coremath/pown.tsv`, the new `tools/coremath.py`, and name-only scaffolding so the files import (`errors.PowerLimitWarning`, exported; `elementary.EXACT_RESULT_LIMIT`, unused). each id alone, 90 s hard timeout (`.scratch/run_ids.py`; a timeout is a hang, red).

| verdict | time | test id | first error |
|---|---|---|---|
| RED | 61.4s | `tests/test_pown_huge.py::test_reproductions_finish` | E               subprocess.TimeoutExpired: Command '['C:\\Users\\user\\anaconda3\\envs\\intervals\\python.exe', '-c', '\nimport math, warnings\nwarnin |
| RED | 1.2s | `tests/test_pown_huge.py::test_exact_corners_at_the_limit` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 2.5s | `tests/test_pown_huge.py::test_an_exact_corner_past_the_limit_inside_the_float_range` | E           assert (<[ValueError...d9f880>, True) == (1.0001260168...150614, False) |
| RED | 1.3s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[3-70000]` | E       AssertionError: (3, 70000) |
| RED | 1.3s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[2-50001]` | E       AssertionError: (2, 50001) |
| RED | 1.4s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[10-25001]` | E       AssertionError: (10, 25001) |
| RED | 1.0s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[2-2097152]` | E       AssertionError: (2, 2097152) |
| GREEN | 1.1s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[x4--40000]` |  |
| RED | 1.0s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[2-2097153]` | E       AssertionError: (2, 2097153) |
| RED(timeout) | 90.0s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[x6-1099511627776]` | no answer in 90 s |
| RED(timeout) | 90.1s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[x7--1073741824]` | no answer in 90 s |
| RED(timeout) | 90.0s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[7-1000000000000000000000000000000]` | no answer in 90 s |
| RED(timeout) | 90.0s | `tests/test_pown_huge.py::test_pown_matches_pow_on_exact_points[x9-1099511627777]` | no answer in 90 s |
| RED | 0.7s | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points[3-70000]` | E       AssertionError: (3, 70000) |
| RED | 0.8s | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points[2-50001]` | E       AssertionError: (2, 50001) |
| RED | 1.0s | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points[10-25001]` | E       AssertionError: (10, 25001) |
| RED | 0.8s | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points[2-2097152]` | E       AssertionError: (2, 2097152) |
| GREEN | 0.6s | `tests/test_pown_huge.py::test_nearest_pown_matches_pow_on_exact_points[x4--40000]` |  |
| RED | 0.8s | `tests/test_pown_huge.py::test_one_limit_for_pown_pow_exp2_exp10` | E               AssertionError: ('exp2', 2097152) |
| RED(timeout) | 90.2s | `tests/test_pown_huge.py::test_the_power_limit_warning[M(2) ** 2 ** 60-pow<a 61-bit int>: .* rounded to nearest]` | no answer in 90 s |
| RED(timeout) | 90.2s | `tests/test_pown_huge.py::test_the_power_limit_warning[O(2, 3) ** -(2 ** 40)-pow-1099511627776: .* tightest float enclosure]` | no answer in 90 s |
| RED | 0.9s | `tests/test_pown_huge.py::test_the_power_limit_warning[M(2) ** M(2 ** 60)-pow: .* tightest float enclosure]` | E       Failed: DID NOT WARN. No warnings of type (<class 'intervals.errors.PowerLimitWarning'>,) were emitted. |
| RED | 0.9s | `tests/test_pown_huge.py::test_the_power_limit_warning[O(Fraction(1, 3)) ** O(2 ** 40)-pow: ]` | E       Failed: DID NOT WARN. No warnings of type (<class 'intervals.errors.PowerLimitWarning'>,) were emitted. |
| RED | 0.9s | `tests/test_pown_huge.py::test_the_power_limit_warning[M(2 ** 60).exp2()-exp2: ]` | E       Failed: DID NOT WARN. No warnings of type (<class 'intervals.errors.PowerLimitWarning'>,) were emitted. |
| RED | 0.9s | `tests/test_pown_huge.py::test_the_power_limit_warning[O(-(2 ** 60)).exp10()-exp10: ]` | E       Failed: DID NOT WARN. No warnings of type (<class 'intervals.errors.PowerLimitWarning'>,) were emitted. |
| RED | 61.2s | `tests/test_pown_huge.py::test_the_power_limit_warning_is_ignored_by_default_and_can_be_an_error` | E               subprocess.TimeoutExpired: Command '['C:\\Users\\user\\anaconda3\\envs\\intervals\\python.exe', '-c', 'import warnings\nfrom intervals |
| RED | 0.7s | `tests/test_pown_huge.py::test_nearest_is_correctly_rounded[1.0287349703833546-10-1.3275015385484197]` | E           AssertionError: (1, 1.0287349703833546, 10) |
| RED | 0.7s | `tests/test_pown_huge.py::test_nearest_is_correctly_rounded[1.0026606152364441-13-1.03514557277232]` | E           AssertionError: (1, 1.0026606152364441, 13) |
| RED | 0.7s | `tests/test_pown_huge.py::test_nearest_is_correctly_rounded[2.1095375758280437e-154-2-4.450148783830459e-308]` | E           AssertionError: (1, 2.1095375758280437e-154, 2) |
| RED | 0.7s | `tests/test_pown_huge.py::test_nearest_is_correctly_rounded[1.3811118839148833-26-4425.378458811315]` | E           AssertionError: (1, 1.3811118839148833, 26) |
| RED | 0.8s | `tests/test_pown_huge.py::test_nearest_against_the_exact_power` | E       AssertionError: (1.0026606152364441, 13) |
| RED | 0.8s | `tests/test_pown_huge.py::test_nearest_huge_exponent_against_mpfr` | E       AssertionError: (1.0026606152364441, 13) |
| RED | 0.8s | `tests/test_fmt.py::test_round_trip_extreme` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 0.9s | `tests/test_fmt.py::test_repr_round_trip[MultiInterval]` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 0.9s | `tests/test_fmt.py::test_repr_round_trip[OutwardMultiInterval]` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 2.1s | `tests/test_fmt.py::test_spellings` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 0.8s | `tests/test_fmt.py::test_parse_value_round_trip` | 1 failed in 0.14s |
| RED | 0.7s | `tests/test_fmt.py::test_past_the_int_str_limit` | E       ValueError: Exceeds the limit (4300 digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit |
| RED | 0.7s | `tests/test_fmt.py::test_parse_hex[0x1f-31]` | E       ValueError: invalid literal for int() with base 10: '0x1f' |
| RED | 0.7s | `tests/test_fmt.py::test_parse_hex[-0X1F--31]` | E       ValueError: invalid literal for int() with base 10: '0x1f' |
| RED | 0.8s | `tests/test_fmt.py::test_parse_hex[0x1f/0x3-value2]` | E                   ValueError: Invalid literal for Fraction: '0x1f/0x3' |
| RED | 0.8s | `tests/test_fmt.py::test_parse_hex[1/0x10-value3]` | E                   ValueError: Invalid literal for Fraction: '1/0x10' |
| RED | 0.7s | `tests/test_fmt.py::test_parse_hex[0x10 / 3-value4]` | E                   ValueError: Invalid literal for Fraction: '0x10/3' |
| RED | 0.7s | `tests/test_fmt.py::test_parse_hex[0x0-0]` | E       ValueError: invalid literal for int() with base 10: '0x0' |
| RED | 0.8s | `tests/test_coremath.py::test_the_sample_is_correctly_rounded[pown]` | E       AssertionError: 239 wrong |
| RED | 0.7s | `tests/test_elementary.py::test_the_two_power_limits` | E       AttributeError: module 'intervals.elementary' has no attribute 'exact_power_bits' |

46 ids: 44 red on the old code (6 of them hangs, killed at 90 s), 2 green: the `(Fraction(2, 3), -40000)`
rows of the two pown-vs-pow tests, a point under both old limits (80000 bits), kept as a guard, not a
change pin. the not-a-change guards (`test_no_power_limit_warning`, `test_hex_floats_and_bad_hex_are_refused`,
`test_the_two_power_limits`'s irrational row) are shown able to fail by the in-tree sabotage above.

## step 6, 2026-10-04: the ziv pathology, measured (SURPRISE for the owner, not fixed)

`.scratch/pown/probe3.py`, `probe4.py` (worktree), 120 s hard timeout each, this laptop:
* new code, pown, exact corner past EXACT_RESULT_LIMIT and close to a breakpoint:
  `O(3 + Fraction(1, 2 ** 1400000)) ** 3` (B = 4.2M bits), `O(3 + 2 ** -2100000) ** 2`,
  `O(3 + 2 ** -3000000) ** 2`: each killed at 120 s (ziv must separate the value from a double it sits
  2**-1.4M..2**-3M from; it would pass 2**22 bits and raise ArithmeticError if it lived that long). the
  old code built each exact power in milliseconds. `** 1` of the same 2.1M-bit operand: 0.03 s (bits
  2.1M, under the limit).
* the same class already existed in `pow_` (exact operands), at a far smaller size: OLD code
  `O(3 + 2 ** -60000) ** O(2)` (120004 bits, past pow's old 100000) and `e = 200000`: both killed at 120 s;
  NEW code: exact Fractions in 0.00 s / 0.01 s (now under the shared 2**22 limit).
* so the build moves the pathology for pow from 50k-bit operands to 2M-bit ones, and gives it to pown at
  the same 2M-bit size. it needs an operand of more than 2**22 / |n| bits within about 2**-(its size) of
  a value whose power is a breakpoint (an exact newton iterate is exactly that).
* possible follow-ups (owner's call, not built): (1) a near-1 shortcut in `rounded_pow` (`|y ln x|`
  bracketed inside (0, 2**-55) rounds like 'above 1' / 'below 1'), which covers x near 1 with huge n,
  where no exact build is affordable; (2) for an exact corner, when ziv passes some precision cap and
  the exact build is affordable (v in range forces |n| log2|x| <= ~1075, so B is a bounded multiple of
  the operand's own size unless x is near 1), fall back to the exact build.

## step 7, 2026-10-04: final runs on the finished tree (plain pytest, not tools/gate.py)
* whole suite (`python -m pytest -q`, every test file, the module doctests and README): 33891 passed,
  0 failed, 740 s.
* `python -m tests.exhaustive_ops --sample 20000`: 20000 checks, 0 failures, 49 s.
* `tools/coremath.py sample --check` (cache from the main checkout via INTERVALS_COREMATH_CACHE): rc 0.
* earlier per-file runs: test_pown_huge 88 passed (12 s); test_fmt + fmt/cuts doctests 74 passed;
  test_extreme_floats + test_elementary + test_backend + test_applicator + test_oracle_flint 1567 passed
  (73 s); test_coremath 51 passed (15 s).
* owed by the orchestrator: the gate through tools/gate.py after the merge; the session changed the
  scalar evaluator (`elementary.exact`, `exact_pow`, `rounded`), so per CLAUDE.md ask the owner whether
  to run `tools/coremath.py check --jobs 4` (it now covers pown too: pow.wc's integral-exponent rows).

## commit
* `60a05a9` on branch `worktree-agent-ad7ab118e02eb5794` (parent 09435ca), 2026-10-04.
* the worktree .scratch/ (gitignored) holds the probe and sabotage scripts (`.scratch/pown/`, `sabotage.py`, `run_ids.py`, logs); nothing in it is needed once this file is transcribed.
