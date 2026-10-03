# small owner answers (agent-aece9c8239fc8e041), 2026-10-03

worktree branch `worktree-agent-aece9c8239fc8e041`, fast-forwarded to master `09435ca` first (it was at `912558b`,
without the reports).

## topic 1: ieee1788 (commit 354f12d)

changed:
* `intervals/ieee1788.py::_number`, `::mid_rad`: the empty set gives `nan` / `(nan, nan)` (explicit `if not s`, no
  exception caught); `::_reduction` (new) wraps `sum_`, `sum_abs`, `sum_square`, `dot`: library call, `nan` on
  ValueError, but `_reductions._direction(rounding)` checked first and `dot` of different lengths passed straight
  through, so those still raise. `ieee1788.sum_ is intervals.sum_` no longer holds (pin removed). module docstring:
  numbers bullet, new numpy bullet, "where 1788 defines another answer" clause, a doctest `mid(Interval())`, `sum_`.
* `tests/itf1788/test_ieee1788.py`: NAN_READ, NUMBERS, REDUCTIONS, `_expects_nan` and the `except ValueError` branch
  of `read` removed; docstring "three readings" -> "two readings"; `test_the_nan_reading_is_narrow` ->
  `test_nothing_is_read_as_nan`. NOTE: the orchestrator's prompt said the clause is in `test_itf1788.py`; it is in
  `test_ieee1788.py` (the report names it there). the library adapter's NaN reading in `test_itf1788.py` stays,
  as the report says (the library still raises, D9).
* `tests/itf1788/test_itf1788.py`: module docstring paragraph naming the third pass and the second reading;
  `_PAIR_DECORATED_AS_DIVISION` names `ieee1788.mul_rev_to_pair` (Q9 closed as built); `REASONS` without
  'domain-clipped functions'; `test_divergence_rows` now also asserts every category has a row (new, mechanical).
* `tests/test_ieee1788_layer.py`: `test_numbers` (nan, and the library part still raises), `test_numbers_examples`
  (both flavours of empty), new `test_reductions[sum|sumAbs|sumSquare|dot]`.
* README: departures rows constructors (D21(d), Q10 closed), mulRevToPair (D21(c), Q9 closed), numbers of the empty
  set and reductions (`ieee1788`'s are NaN); the layer bullet; one numpy-section line ("the 1788 layer's `Interval`
  is a scalar to numpy: operators with numpy scalars work, ufuncs do not") appended to the numpy bullet (possible
  merge conflict with the numpy stream: that bullet's last line got a full stop).
* v2-plan.md: residual table (~614) without domain-clipped; "the 1788 layer" current design: first bullet, numbers,
  numpy rule (no longer "owed"), third-pass readings. line 2339 (2026-08-16 history) left alone.
* census (`tools/itf1788_census.py`) output: category counts unchanged (an empty category is not printed); no doc count
  changes.

for the orchestrator: D21 (b), (c), (d) rows to mark decided; README cites "D21(b)/(c)/(d) ... owner 2026-10-03"
in the recorded column; HANDOFF still-owed M16b bullet (two clauses) is paid; layer-numpy open item and
the departures-census 'domain-clipped' half are done; D27 (the four departures) not written by me.

sabotage (2026-10-03, `.scratch/sab/sab.py` in the worktree: replace once, clear pycache, PYTHONDONTWRITEBYTECODE,
restore, byte-compare):
| break | red |
|---|---|
| whole old `ieee1788.py` (HEAD) under the new tests | 27 failed: test_numbers, test_numbers_examples, test_reductions x4, test_nothing_is_read_as_nan, 20 third-pass vectors (num.itl 96..251 x12, reduction.itl 26..51 x8) |
| `mid_rad` empty raises again | test_numbers, test_numbers_examples, num.itl:152, :167, test_nothing_is_read_as_nan |
| reduction: drop the rounding check | test_reductions x4 |
| reduction: drop the length check | test_reductions[dot] |
| re-add 'domain-clipped functions' to REASONS | test_divergence_rows |

runs: `tests/itf1788 tests/test_ieee1788_layer.py intervals/ieee1788.py README.md`: 28094 passed in 72 s (2026-10-03).

left open (the report's "consider"/later): a hook of the layer's own for numpy; exporting the layer; a hull method in
the library (Q10's "if a second consumer appears").

## topic 2: allen / relations (commit d8fa24f)

changed:
* `intervals/relations.py::_normalized` (new helper) asserted (`assert _normalized(a, b), (a, b)`) in `lt`, `le`,
  `eq_pointwise`, `before`, `adjoins`, `disjoint`, `overlaps`, `contains`, `within`, `equals`, `weakly_less`,
  `strictly_less`, `allen` (its two pieces, after the length check), `allen_relations` (was `kernel.is_valid` inline);
  `gt`, `ge`, `after`, `certainly_*`, `possibly_*` delegate. `allen_matrix` does not assert (D1 kept: any order),
  each piece is checked through `allen`. module docstring states the rule.
* docstrings: `relations.allen_relations` and `MultiInterval.allen_relations`: the set is extensional, not a
  disjunction; `relations.allen_matrix`: `zip(*M)`, `np.array(M)` (and `()` loses m), the converse (the report's (a)
  docstring note, in its summary table as a recommendation).
* `v2-plan.md` "comparisons": the extensional sentence and the one rule (current design, not the decision log).
* tests: `tests/test_relations.py::test_every_relation_asserts_normalized_operands[24 names]` (functions found by
  inspection: public, module's own, params exactly (a, b)), `::test_the_relations_are_the_ones_listed`.
* cost: `allen_relations`' cut-comparison count in `::test_allen_relations_compares_cuts_linearly` was 360 of a bound
  800 before (measured 2026-10-03); allen's own check adds at most 2 per sweep call; the test stays green.

sabotage (2026-10-03):
| break | red |
|---|---|
| old `relations.py` (HEAD) under the new tests | 23 of 24 params (all but `allen_relations`, which asserted already) |
| drop `allen`'s assert only | `[allen]`, `[allen_matrix]` |

runs: `tests/test_relations.py tests/test_orders.py tests/test_multi_interval.py tests/test_propagation.py
tests/test_numpy_compat.py intervals/relations.py intervals/multi_interval.py README.md`: 741 passed in 106 s
(2026-10-03).

left open (the report's later/optional): an `AllenMatrix` class, a public sparse `allen_pairs`, the fill + sweep,
documenting `()` beyond the docstring note. HANDOFF still-owed M16c bullet is paid.

## topic 3: solver (commit 6b621f3)

changed (`intervals/solver.py`):
* `newton`: `width <= piece.wid() / 2` -> `2 * width <= piece.wid()` (newton-width, HANDOFF item 4). SURPRISE: the
  bug was live, not "not observed": `newton(lambda x: x ** 2 - 9 * 10 ** 800, M(10 ** 400, 10 ** 401),
  max_steps=1)` raised `OverflowError: integer division result too large for a float` at the first step (2026-10-04);
  HANDOFF's 2026-09-28 note said it "did not finish in 2 minutes" (max_steps=10: the exact ends grow; still true
  past max_steps 5 after the fix, a 200 s probe timed out at max_steps=10).
* `_magnitude_split(piece, below_one=False)`: with `below_one`, a piece inside [-1, 1] spanning > 16x in magnitude is
  split at +-2 ** mean exponent too (both signs by the existing mirror; a piece from 0 to <= +-1 still None).
  `_bisect(piece, below_one=False)` passes it on. ONLY newton passes `below_one=True`.
* newton's step gate: `wide = _magnitude_split(piece, below_one=True) is not None and piece.wid() > tol`; a piece
  wide in magnitude but within tol is still stepped (else a regression, below).
* docstrings: `newton` and `solve` say tol is absolute and max_steps counts pieces/boxes (measured 2026-10-04 with
  a call-counting f: newton 1 Dual call + 1 point call per piece, up to 2 more at output; solve n + 2 per stepped
  box, 1 per pruned box, up to n + 2 more at output, the last from v2-plan's known-limits bullet); module docstring
  bullet on the split; `_magnitude_split` docstring with the numbers.
* v2-plan.md "the solver stack" termination bullet: the split below 1, newton only, why, tol/max_steps sentence.

SURPRISES / deviations from the report, for the owner:
1. the report's variant applied to `_magnitude_split` as is (shared by solve's `_wide`/`_choose`) blew
   `tests/test_solve.py::test_evaluation_budgets`: the circle `(x**2+y**2-1, x-y)` on `[-1e300, 1e300]^2` went
   from 103 calls to 3433 (bound 160): gauss-seidel leaves components like `(-1.0, -3.7e-301)`, which the split
   took through every exponent in both coordinates. so the split below 1 is newton's only; solve's counts are
   identical to before on every probe (circle 103, `(x**2-1e-40, y-x)` 202, `(x-1e-25, y-2x)` 13).
2. the variant alone cost newton a proof: `(x - 3)(x - 1e-25)` on `[1e-30, 4]` had its small zero proved `[1e-25]`
   before and unproved `[1e-30, 3.6e-15]` after (likewise `x - 1e-25` on `[1e-30, 1]`): the exponent splits reach
   tol before any step runs. hence the gate change: a piece within tol gets the step. with it every probe is
   proved where the old code proved, at the same or fewer calls.
3. better than the report: `x ** 2 - 1e-40` on `[-1, 1]` at the default tol is now 28 calls with BOTH zeros PROVED
   (the report's variant: 11 calls, unproved; old: 132 unproved). so HANDOFF Q11(e)'s example (two unproved roots)
   no longer holds. `[1e-60, 1]`: 65 -> 22 calls, still unproved (wider piece, 8 steps past tol not enough).

probe numbers (2026-10-04, `.scratch/sab/count.py` in the worktree, new vs `git show HEAD` solver):
| case | old calls, result | new calls, result |
|---|---|---|
| newton x**2-1e-40 on [-1,1] | 132, 2 unproved (2.3e-14 wide) | 28, 2 proved |
| same, tol=1e-30 | 216, 2 proved | 48, 2 proved |
| newton x-1e-25 on [1e-30,1] / mirror | 7 proved / 7 proved | 7 proved / 7 proved |
| newton x**2-1e-40 on [1e-60,1] / mirror | 65 / 65 unproved | 22 / 23 unproved |
| newton (x-3)(x-1e-25) on [1e-30,4] | 30, both proved | 30, both proved |
| solve circle on [-1e300,1e300]^2 | 103 | 103 |

pins (tests/test_solver.py): `test_the_magnitude_split_goes_below_one[1|-1]` (both zeros proved, < 1e-30 wide,
<= 25 evaluations, unit checks of the split), `test_a_wide_piece_within_tol_is_still_stepped[1|-1]`,
`test_a_piece_wider_than_the_doubles`.

sabotage (2026-10-04):
| break | red |
|---|---|
| split below 1 off (`hi <= 1` -> None again) | below_one[1], [-1] (on `[r.unique ...] == [True, True]`, not just the unit check) |
| gate without `and piece.wid() > tol` | below_one x2, within_tol x2 |
| `width <= piece.wid() / 2` back | test_a_piece_wider_than_the_doubles |
| solve's `_wide` with below_one=True | tests/test_solve.py::test_evaluation_budgets |

runs: `tests/test_solver.py tests/test_solve.py tests/test_autodiff.py tests/test_gradient.py intervals/solver.py
README.md`: 301 passed in 108 s (2026-10-04).

left open (report's "consider"/later): Root/RootBox as frozen dataclasses (§10.1); `rtol`; solve accepting a bare set;
`roots` rename; off-centre bisection; merging continuum boxes.

## topic 4: Q19 and Q20 (commit 50497cd)

changed:
* `intervals/multi_interval.py::OutwardMultiInterval.rounded` (new, after `expand`): `self._wrap(rounding.float_cuts(
  self._cuts, outward=True))`; `from intervals import rounding` added to the module's imports. name `rounded()` is the
  report's suggestion ("name open"); placed on `OutwardMultiInterval` only: the report says the exact class "could"
  get a to-nearest one "for symmetry" -> left open. an exact end past the doubles goes to `inf` (open) / `MAX`, as
  `float_cuts` already does (`[1, 10**400]` -> `[1.0, inf)`).
* `OutwardMultiInterval` docstring: the isotonicity paragraph (within one grid; across grids within
  `f(B).rounded()`). README "rounding" bullet: one sentence + `rounded()`. v2-plan "current design" > arithmetic >
  **rounding**: the isotone-within-one-grid paragraph, the oracle, `rounded()`.
* Q20: v2-plan "flags at rounded ends": the crossed-piece sentence with the rootn example and the applicator's `*`
  example (`MultiInterval(1 - Fraction(1, 10 ** 30), 1.0) * Fraction(1, 3)` = `[0.3333333333333333,
  333333333333333333333333333333/1000000000000000000000000000000]`, re-run 2026-10-04). `_settled`'s docstring
  already said it (report (2)); no README change (the report names only the plan).
* tests (`tests/test_outward.py`): `test_rounded_examples[9 ids]`, `test_rounded_is_the_tightest_double_cover`
  (hypothesis over `any_operands`: equals the module's own `cover`, no package rounding; float ends; idempotent;
  double-ended input unchanged), `test_rounded_restores_isotonicity_across_grids` (Q19's `+` row and the steps'
  `round([0.0, 0.25], 1)` vs `[-1, 3/10]` row: premise `f(A) ⊄ f(B)` asserted, then `f(A) ⊆ f(B).rounded()` and
  `f(A.rounded()) ⊆ f(B.rounded())`).

sabotage (2026-10-04):
| break | red |
|---|---|
| `rounded()` returns self | 10 of 11 (all examples but 'empty', the cover property, the isotone test) |
| `rounded()` to nearest (`outward=False`) | 9 of 11 (examples but 'ints', 'empty'; cover; isotone) |
(the method absent entirely, the old code, fails all as AttributeError.)

runs: `tests/test_outward.py tests/test_propagation.py tests/test_multi_interval.py intervals/multi_interval.py
intervals/rounding.py README.md`: 405 passed in 44 s (2026-10-04).

left open: `MultiInterval.rounded()` to nearest (report: "could"); the optional fma/%/hypot/cancel_minus re-typing to
per-corner (report: "not needed for 2.0"); Q20's optional pin tying `_settled` and the applicator's `*` sliver.

## final (2026-10-04)

combined run on the final tree (`tests/itf1788 tests/test_ieee1788_layer.py tests/test_relations.py tests/test_orders.py
tests/test_solver.py tests/test_solve.py tests/test_outward.py tests/test_propagation.py tests/test_multi_interval.py`
+ doctests of ieee1788, relations, solver, multi_interval, rounding + README.md): 28738 passed in 344 s, rc 0.
not run: the full gate (tools/gate.py, the orchestrator's), the fuzz. the worktree's `.scratch/sab` probes deleted.
commits on `worktree-agent-aece9c8239fc8e041`: 354f12d ieee1788, d8fa24f relations, 6b621f3 solver, 50497cd Q19/Q20.
