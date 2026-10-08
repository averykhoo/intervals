# records

what was built after v2 (2026-10-08 on), one section per finished item, newest first: what changed, the
evidence (gate counts with dates, the pins and their red-on-old check, sabotage), and what it left. the
successor of `docs/archive/v2/v2-implementation-plan.md` §2, which holds every record up to v2.

an open item's spec is its row in `HANDOFF.md`; when it is done its record goes here, its decisions in
`docs/decisions.md`, and a one-line entry in `HANDOFF.md`'s session log.

## m14b-open: number types, precision first (D31, D32), built 2026-10-08

the owner's D32 (precision first: what is known exactly is exact) built as its "what m14b-open builds" bullet says,
plus what the pins found it needed. every example below was probed on the built code, 2026-10-08.

**what changed**

* **the kernel's tie rule** (`kernel.normalize`, `kernel.sweep`): where two cuts of equal value meet, an exact one
  and a float of the same number, the exact one is kept, whichever operand came first (`M(-1, 1.0) | M(-1.0, 1)` and
  swapped are both `[-1, 1]`; were `[-1, 1.0]` and `[-1.0, 1]`). a point whose two cuts are one exact and one float
  (`[2.0, 2]`) is the exact point; `kernel.is_valid` now refuses such a point, so `from_cuts` raises on it, as on
  any tuple `normalize` would not give (the fuzz-symmetry examples that built one by hand were rewritten)
* **`kernel.restrict(a, region)`**, the library's clip of an operand to a domain or a part of the line: `a ∩ region`
  with `a`'s own cut kept on a tie, a clip point the region's (exact). the first build used `intersection` for the
  domain clip, and `tests/test_oracle_flint.py` caught it: `asin([1.0])` became the open one-ulp piece around pi/2,
  since the operand's 1.0 tied with the domain's exact 1, against D32's own line that `acos([-1.0])` stays
  `[3.141592653589793]` "where the user gave the float point". `restrict` serves `functions.apply`'s domain clip,
  `pow_`'s, the sign/part splits (`_sign_parts`, `_tagged_parts`) and `modulo.mod`'s dividend and divisor clips;
  `reverse.py`'s intersections stay `intersection` (their `x` is the user's set; its tests were green either way)
* **abs** (`ops.absolute`) is the applicator's set intersected with `[0, inf]`, which changes no point and gives an
  attained 0 the constant's exact type: the owner's identity `abs(X) = (X ∪ -X) ∩ [0, inf]`, types included. the
  pin found a case D32's probe had not: an operand end at `0.0` (`abs(M(0.0, 1))` was `[0.0, 1]`, is `[0, 1]`;
  `abs((-inf, 0.0))` keeps its open `0.0`, not reached). `ops._abs` is unchanged
* **sin, cos, csc, sec**: an extremum inside the piece is the exact ±1, from a float piece too (`cos(M(-1.0, 1.0))`
  `[0.5403023058681398, 1]`, `sin(M(0.0, 2.0))` `[0.0, 1]`); csc and sec are a build default (D33): D32 named sin
  and cos, and its rule (a) covers every attained extremum. `sin(O(0.0, h))`, h the double below pi/2, stays
  `[0.0, 1.0)` (no extremum inside), as D32 pins
* **the step functions** list ints on a float piece (`floor(M(-2.5, 3.0))` `{[-3], ..., [3]}`, `trunc(M(-2.5, 3))`
  all ints, `sign` `-1, 0, 1`, a hull's ends too); `round(A, ndigits)` with `ndigits > 0` on a float piece still
  gives floats, its grid values (0.12) being no doubles, as python's `round(2.675, 2)` (D33). past 2 ** 53 the
  outward class lists the exact ints (`ceil(O(2.0 ** 53, 2.0 ** 53 + 2))` is three exact points), where it enclosed
  each one that is no double in the gap around it
* the acos clip and abs's own code unchanged, as D32 said; README "values" and its `math.floor` example, the
  modules' docstrings

**pins** (every one red on a break, below): `tests/test_kernel.py::test_a_tie_goes_to_the_exact_cut` (union,
intersection, symmetric difference: the same types in either order, an exact cut wherever an operand has one, no
mixed point), `::test_normalize_tie_table`, `::test_restrict_keeps_the_operands_own_cuts`;
`tests/test_ops_properties.py::test_abs_is_the_union_with_the_negation_on_the_half_line` (`repr(abs(X)) ==
repr((X | -X) & C(0, inf))`, both classes, 200 examples a run over values in both types);
`tests/test_functions.py::test_an_attained_extremum_is_exact` (D32's `h` pins), `::test_a_domain_clip_is_exact_
and_an_operands_end_its_own`; `tests/test_steps.py::test_the_integers_are_exact`, and its oracle (`_rounds`,
`_image`, the hull, the decimal check) now compares types (`repr`) as well as values, which `==` never did
(`[3] == [3.0]`).

**sabotage** (`tools/sabotage.py`, copies of the working tree, 2026-10-08), every break RED, every control green.
on the final kernel (8): normalize's end tie, its start tie, its point check on append and `_exact_point`'s body,
complement's exact points, the sweep's tie, `restrict` as `intersection`, `is_valid` admitting a mixed point. on
lines unchanged since (6): abs without the half line, the domain clip by `intersection`, sin/cos's and csc/sec's
float ±1, the steps' float ints, the steps' exact fine grid.

**speed**: the first kernel cost `normalize` about 1.65x (a second pass over the pieces, a call per sweep event);
rewritten to check a type before a value and a mixed point only where one can arise, about 1.2x on `normalize`
alone and level with HEAD on whole operations (`|`, `&`, `+`, `*`, `sin`, `abs` over random float sets; this
laptop's noise is about 20 %), 2026-10-08.

**gate**: see the session log (2026-10-08). the first full run (before the oracle updates) failed 42, every one an
old type expectation, the asin/acos clip above, or a tokenize error from editing a test file under a running
pytest. the first recorded gate:rest failed 1: `tests/test_extreme_floats_functions.py`'s sharpness oracle
clipped by `intersection` and so asked asin of `[0.9999999999999999, 1.0]` for the exact pi/2; it clips by
`restrict` now, as the library, and the case is pinned (`test_ends_are_sharp_where_the_fuzz_looked`, with acos's
mirror; both red under the old oracle).

**left**: `x ** 2` and cosh at an operand end `0.0` are still `[0.0, ...]`, `[1.0, ...]` where `abs(x) ** 2` and
`cosh(abs(x))` are `[0, ...]`, `[1, ...]` (Q27); `reverse.py` not moved to `restrict`.
