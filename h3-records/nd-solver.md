# M16a, H3's second part: the solver in several variables (stream nd-solver), 2026-09-28

the orchestrator merges these sections into `v2-plan.md`, `v2-implementation-plan.md`, `HANDOFF.md`
and `README.md`; this stream edits none of them. design: `.scratch/h3b/design/nd-solver.md` and its
critique `nd-solver-critique.md` in the main checkout (gitignored; what the build took from them is
below). ids: M16a, D20, Q12 only.

## design

(for `v2-plan.md` "the solver stack", after M15's bullets; heading suggestion "the solver stack (M15,
2026-09-27; M16a, several variables, 2026-09-28)")

* **`gradient(f, xs)` and `jacobian(F, xs)`** (`intervals/autodiff.py`, below `derivative`; exported
  from `intervals`): n passes of `f`, pass j calling `f(c_1, ..., Dual.variable(x_j), ..., c_n)` with
  `c_k = Dual.constant(x_k)` and reading the derivatives: column j of the jacobian. `Dual` is not
  touched (no vector mode: a tangent tuple inside `Dual` would edit the 42 chain rules M15 pinned).
  `xs` is a list or a tuple of n sets or numbers, every set bare or every set decorated (checked up
  front: `f` need not combine the two, and `lambda x, y: x` would never meet the bare `y`); a number
  is a point of the first set's kind (a `MultiInterval` if every entry is a number). `f` takes n
  positional arguments and returns a `Dual` or a number (a constant: its partials `[0]`); `F` returns
  a list or a tuple of m of them, and `jacobian(F, xs)[i][j]` is `∂F_i/∂x_j`, rows as tuples, no
  matrix type. at n == 1 `gradient(f, [x]) == (derivative(f, x),)`, the same pass. decorated, the
  entries carry the C¹ proof in n variables, relative to the box: a value and every partial dac or
  better say every op and every op of its chain rule's formula was defined and continuous on the
  box, so `F` is C¹ on the box relative to the box, which is what the mean value theorem on the box
  needs (one-sided at its faces). not on an open set holding the box: `x ** 1.5` over `[0, 1]` is com
  with nothing below 0 in its domain, and `abs` over the point `[0]` has a dac derivative (review
  F1)
* **`solve(F, xs, *, tol=1e-10, max_steps=10_000)`** (`intervals/solver.py`, below `newton`) returns
  `RootBox(box, unique)`s (`box` a tuple of n connected `OutwardMultiInterval`s), pairwise disjoint in
  some coordinate, inside `xs`, sorted by the components' `sort_key`s, and every zero of `F` in `xs`
  is in one. `xs` as for `newton` per coordinate (a `DecoratedInterval` refused); n == 1 is `newton`
  exactly. a branch and prune over the product of the pieces of `xs`, per box: prune by range; a
  point is unique iff `F` is exactly `[0]` there; **the step** on a box with no wide component
  (unbounded, or spanning more than a factor of 16 in magnitude) where the n decorated passes over
  the box's **closed hull** `H` are all dac or better (krawczyk's theorem is for a compact box, and
  a box from `mul_rev` or a bisection has open ends): `J` over `H`, a point `m` (the float midpoint,
  else the exact one, so a component two doubles wide can still be stepped), `Y` the float inverse
  of `mid J` by gauss-jordan with partial pivoting (the identity where a midpoint is not a finite
  float or the matrix is singular: any real `Y` keeps the step valid), `Mx = Y J`, `b = Y F(m)`, then
  * **krawczyk's test**: every `K[i] = m[i] - b[i] + Σ_j (δ_ij - Mx[i][j]) (H[j] - m[j])` non-empty
    and inside `H[i]`'s interior proves exactly one zero in `H`, so in the box (`int H` is inside the
    box whatever its ends). `g(x) = x - Y F(x)` maps `H` into `K` by the mean value theorem row by
    row: brouwer gives a fixed point. uniqueness is argued on the exact `K*`, built from the closed
    range of the true partials over the compact `H`: `K* ⊆ K ⊆ int H` and `K*` compact make it
    strictly narrower than `H`, so `ρ(|I - Y A|) < 1` for every real `A` of the mean value theorem,
    `Y` is regular and `g` contracts. (the computed `K` may be as wide as `H`, its ends open: the
    strict inequality is not read off `K`)
  * **the narrowing, preconditioned interval gauss-seidel with `mul_rev`** (hansen and sengupta),
    on the box: row by row `Z[i] = Z[i] ∩ (m[i] + mul_rev(Mx[i][i], -b[i] - Σ_{j≠i} Mx[i][j] (Z[j] -
    m[j])))`, each row using the rows before it. `mul_rev`, not `/` (D7). where `0 ∈ Mx[i][i]` the row
    is two pieces: the sweep stops and the box splits there, the other components as narrowed so far.
    krawczyk proves and gauss-seidel narrows: krawczyk has no division, so a partial holding 0 teaches
    it nothing, which is where multi-intervals split
  * then `newton`'s rules: a unique box is narrowed while the step narrows it; an unproved one goes on
    while the step halves its width, past `tol` for at most `_PAST_TOL = 8` steps; else bisection: a
    wide component first, round robin from the last split coordinate + 1 (`x + y, x - y` on `REALS²`:
    15 calls, 2749 without), then the other components widest first, falling through past a component
    that cannot be split
* **before an unproved box is output**: (1) **its simplest point**, the simplest rational of each
  component's closed hull (`_simplest_between`, continued fractions), when inside the component (not
  on an unbounded one): if `F` is exactly `[0]` there it is output alone as a unique point, and the
  rest of the box goes back as up to 2n boxes. a zero at a simple rational sits on a split face or on
  a component a row made degenerate, where no `K` fits inside an interior. (2) else **krawczyk on the
  box inflated within its region**: each stack entry carries its region `R`, the part of the input
  it stands for, with every zero of `R` in the box. the work list starts with `R` = the box; a
  gauss-seidel narrowing keeps `R`; a split (gauss-seidel's or a bisection's) cuts the old `R` in the
  split coordinate only, between the neighbouring boxes, and the simplest point's rest boxes get `R`
  cut at the point. `H'' = ` the closed hull of (the box widened by twice its width, at least 1e-12
  relative, on each side) ∩ `R`, with its own jacobian and C¹ gate: `K(H'') ⊆ int H''` gives one zero
  in `H''`, in `int H'' ⊆ R`, so in the box, and the box ⊆ `H''` holds no other. why: one row that
  pins a coordinate (`y - 1/4`, `3y - 1`) makes the box degenerate or an ulp wide there at once, and
  krawczyk on the box can then never prove the textbook simple zero
* the warnings inside `F` are silenced as in `newton`. `F` is called with decorated `Dual`s, with boxes
  of `OutwardMultiInterval`s and with points
* known limits: a zero on a split face that is not a simple rational in every coordinate, and a
  singular zero, end as unproved boxes of width `tol`, often with unproved slivers beside them
  (sound, not proved). the cost is the library's arithmetic: a box costs n + 2 calls of `F` (the
  plain call, n decorated passes, the point), plus up to n + 2 when it is output unproved. a
  continuum of zeros is bisected to `tol` everywhere, and each of its boxes gives its simplest point
  and up to 2n rest boxes, so the output is up to 2n + 1 times the boxes of width `tol` (not a
  cascade: a rest box's closed hull holds the point, which is then its own simplest point and not
  in it). an exact end beyond the doubles (`[10 ** 400, 10 ** 401]`) makes the float
  preconditioner overflow `b` to `(1.8e308, inf)`, so such a box is bisected, not stepped (review
  F2, F3; numbers in M16a's record)
* **the direction tag stays under "later", now for n variables**: the step runs only on a box whose
  components are all bounded, where `F` is dac on the closed hull, so every real value and partial
  over it is bounded. the enclosures may still hold an open end at ±inf by overflow (`exp` over
  `[700, 720]` is `(1.0142320547350045e+304, inf)`, dac): every set op keeps every real point, so the
  real `A` and the real `F(m)` of the mean value theorem stay inside `J`, `Mx`, `b` and `K`, and no
  degenerate `[±inf]` arises from a finite real (overflow is open at inf, and D2's corner makes
  `(a, inf) * [0]` `[0]`). on a box holding ±inf as a point `F` is only evaluated for the range
  prune, where `1/(1/[inf])` is `∅` (D7) and the tag would make it `[inf]`: that changes which
  function `F` is at ±inf, not whether a zero of it is found (a zero is a point where `F` has the
  value 0, and with the tag `[inf] - 5` holds no 0 either). no test: a test that the solver behaves
  the same with and without a type that does not exist cannot be written;
  `tests/test_solve.py::test_overflow_box` pins the overflow case

"later (not in v2.0)": strike "a solver in several variables (a `Dual` carries one derivative)":
built as M16a. the direction tag's bullet: add "M16a: not needed in n variables either (the solver
stack, above)".

"package layout": `autodiff.py` gains `gradient`, `jacobian`; `solver.py` gains `solve`, `RootBox`.

"testing": an M16a bullet: `tests/test_gradient.py` (the jacobian against arb series, one column per
seeded variable) and `tests/test_solve.py` (constructed systems `A G(B x + c)` with every real zero
known: exact fractions, or `r + Σ a sqrt(q)` decided by arb).

## decision-log revision

### 2026-09-28 revision: M16a, the solver in several variables (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done". M16a is its stream nd-solver; the choices are the
build's defaults, open for the owner (Q12; D20):
* **n passes, `Dual` untouched**: a gradient or a jacobian is n evaluations of `F`, each with one
  variable seeded. vector mode (a tangent tuple inside `Dual`, one pass) edits M15's pinned chain
  rules; it stays the alternative if a measured solve is too slow (Q12(a))
* **`solve` and `RootBox`** (names, Q12(b)); `Root` stays one set for `newton`. n == 1 delegates to
  `newton`, so one variable keeps M15's behaviour exactly
* **krawczyk proves, gauss-seidel with `mul_rev` narrows**, on the closed hull, with a float
  preconditioner (identity fallback); uniqueness for n >= 2 by krawczyk only (hansen and sengupta's
  own test is not used: its proof was not checked)
* **the simplest point and the inflation, before an unproved box is output**, each measured to be
  needed: without the first, the cusp's `(0, 0)` and `(1, 1)` end unproved; without the second,
  `(x ** 2 - 2, y - 1/4)` proves none of its two zeros (critique B1). the inflation is clipped to the
  box's region, carried on the stack, without which it can claim a box holding no zero
* **the C¹ gate in n variables**: every value and every partial of the n decorated passes dac or
  better, over the closed hull
* **bisection**: wide components first, round robin; then the widest (kearfott's smear measured on
  the prototype, 148 against 146 calls, 345 against 340, not taken)
* **no direction tag** (the solver stack, above)

## D20

(the §0 table row)

| D20 | **decided in the build 2026-09-28 (the session's defaults), open for the owner: `HANDOFF.md` Q12.** the solver stack's second part (M16a): (a) a gradient or a jacobian is n passes of `F`, `Dual` untouched (not vector mode); (b) names `gradient`, `jacobian`, `solve`, `RootBox`, public and exported from `intervals`, `Root` unchanged; (c) `solve` at n == 1 is `newton`; (d) uniqueness by krawczyk only, on the closed hull, with a float preconditioner (identity fallback), narrowing by gauss-seidel with `mul_rev`; (e) before an unproved box is output, its simplest rational point (exactly `[0]`: a unique point) and then krawczyk on the box inflated within its region; (f) wide components bisected first, round robin, then the widest; (g) `tol=1e-10` absolute on the widest component, `max_steps=10_000` boxes; (h) no direction tag | as built | M16a |

## M16a

### spec

* **`intervals/autodiff.py`**: `gradient(f, xs)`, `jacobian(F, xs)` (and the private `_box`,
  `_entry`, `_passes`, `_sequence`), appended below `derivative`; nothing above it edited
* **`intervals/solver.py`**: `RootBox(box, unique)`, `solve(F, xs, *, tol, max_steps)`, and the
  private `_input_box`, `_outputs`, `_values`, `_jacobian`, `_points`, `_mid`, `_inverse`,
  `_combine`, `_precondition`, `_krawczyk`, `_gauss_seidel`, `_width`, `_wide`, `_choose`,
  `_simplest_between`, `_simplest_point`, `_finish_box`, `_regions`, `_inflate`, `_rounded`,
  `_inflated_unique`, appended below `_bisect`; M15's functions reused unchanged (`newton`,
  `_magnitude_split`, `_bisect`, `_PAST_TOL`) but `_point_in`, whose `float(mid)` now falls back to
  the exact midpoint where it overflows (review F2, which `newton` shared); the module docstring
  unchanged
* `intervals/__init__.py`: the four names imported and in `__all__`, under "M16: the solver stack's
  second part, several variables (H3)"
* exit: the jacobian against arb (soundness at points of boxes, sharpness at a point) and equal to
  `derivative` at n == 1; `solve` sound and its uniqueness claims true on constructed systems with
  every real zero known (rational and irrational), under any `tol` and `max_steps`, and with factors
  not C¹ under a budget (`max_steps` up to 30); the C¹ gate shown necessary by examples (the pole,
  the kink, a coupled kink, a jump: the random factors not C¹ do not detect it, review spec F1);
  simple zeros proved, irrational ones by
  krawczyk (pinned coordinates included), simple rational ones as exact points; n == 1 equal to
  `newton`; the budgets pinned; the gate green; every new property sabotaged once and seen red

### record (2026-09-28)

* **what the build found on its way**, each fixed before the record:
  * **critique B1's region was not enough.** the critique defined the region as "the box as last
    pushed by the work list, a bisection, a gauss-seidel split or step 6". built that way,
    `(x ** 2 + y ** 2 - 1, y - 1/4)` on `[-3, 3]²` still proved neither zero: y was pinned to `[0.25]`
    by a step *before* a later split, so the split's pieces, and their regions, were degenerate in y,
    where no inflated box has an interior. fix (`solver.py::_regions`): a split cuts the *old* region
    in the split coordinate only, between the neighbouring boxes (their open and closed ends
    respected); the other coordinates keep the region's extent. every zero of the new region is a
    zero of the old one, so in one of the boxes, and on its own box's side of the cut. the simplest
    point's rest boxes get the region cut at the point, the point's coordinates before k. pinned by
    `tests/test_solve.py::test_simple_zeros_are_proved_unique[pinned circle]` (bisection) and
    `[pinned, then split]` (a gauss-seidel split: `(x - 1/4, x ** 2 + y ** 2 - 4)`, whose first step
    pins x in row 0 and splits y in row 1; the `- 1` variant does not split, row 1's `c` holding 0)
  * **the n-dimensional loop at n == 1 gives `newton`'s boxes** on all four functions of
    `::test_n_equals_one_is_newton` (sabotage: green), at 1.4x to 2x the calls (49 against 32 for
    `x ** 2 - 2` on `[-10, 10]`, 145 against 93 for `sin`, 32 against 16 for the kink, 47 against 33
    for the pole; 2026-09-28, the review's `.scratch/h3b/review/nd-solver-spec/n1.py`, which execs
    `solve` with `if n == 1:` made `if False:`; the first two pairs were quoted as "about 1.5x" until
    review spec F4). the delegation is pinned by the call counts as well as the boxes
  * **cost** (2026-09-28, five streams sharing the laptop; `.scratch` timing scripts calling
    `tests/test_solve.py::system` and `::Counted`): unbudgeted constructed systems with two factors
    per coordinate have a tail (one of 11 draws: 4677 calls, 201 s); factors not C¹ at their zeros
    (`abs`, `cbrt`), unbudgeted with `tol=0.01`, cost 500 to 1500 calls, 50 to 90 s each (the gate
    keeps the step off every box across the kink line, so they are bisected); a constructed n = 3
    system with two zeros did not finish in 120 s (1464 calls in 66 s at `max_steps=400`). so the
    unbudgeted random test draws one factor per coordinate (its `@example`s keep two), the factors not
    C¹ are drawn only in the budgeted test, and `::test_three_variables` has one zero (0.4 s); the
    sphere is the n = 3 system with two zeros. so the `abs` and `cbrt` factors check soundness under
    a budget only: they do not detect the C¹ gate removed (review spec F1: forced off, the budgeted
    test and 25 draws at `max_steps` 100 to 400 stayed green); the gate is pinned by the examples
  * the gate found `tests/test_applicator.py::test_package_exports_unchanged` red: it pins
    `intervals.__all__`; the four names are added there
  * a split of a box already proved unique needs a step whose new preconditioner leaves 0 in a
    diagonal entry, which no example reached (sabotage "a split box keeps unique": green). pinned
    directly: `::test_a_split_box_is_unproved` makes `_krawczyk` claim the first box, which the step
    then splits, and with `max_steps=1` both pieces are output unproved
  * the "jacobian over the box, not its closed hull" break is seen by the kink, not only by the
    helper case the design planned (`::test_c1_is_decided_on_the_closed_hull`)
* **tests** (`tests/test_gradient.py`, `tests/test_solve.py`, and the doctests of `gradient`,
  `jacobian`, `solve`):
  * `tests/test_gradient.py::test_jacobian_encloses_the_partials` (random expression trees in x and
    y, `F = (e1, e2)` over drawn boxes, every entry against arb's partial at a drawn point, 60
    examples), `::test_jacobian_is_tight_at_a_point` (1e-10 relative; it cannot see a wrong seed,
    which is still tight at a point), `::test_gradient_is_the_jacobian_row`,
    `::test_one_variable_is_derivative` (set and decoration), `::test_seeds`,
    `::test_decorated_columns`, `::test_arguments` (the kind check with an `f` that ignores the
    bare argument, critique B2)
  * `tests/test_solve.py`, the oracle `::system` (`A G(B x + c)`, zeros exact or `r + Σ a sqrt(q)`
    decided by arb) and `::check_roots`: `::test_every_zero_is_enclosed` (one factor per coordinate,
    four input boxes: closed, open-ended, multi-piece, integer faces; `max_steps=2000`, critique N6;
    `@example`s: systems 7 and 3 of the prototype, system 3 on a box with zeros on its faces and
    corners, a double zero, zeros 1e-12 apart, irrational zeros),
    `::test_every_zero_is_enclosed_on_a_budget` (with `abs` and `cbrt` factors, `max_steps` 1..30,
    `tol` in {1e-3, 1e-10, 0}, 40 examples), `::test_three_variables`,
    `::test_simple_zeros_are_proved_unique` (12 systems: `(x ** 2 - 2, y - x)`, the circle and the
    line on `[-10, 10]²`, `REALS²` and `[-1e300, 1e300]²`, the sphere, `(exp x - y, x + y - 2)` (x =
    2 - W(e²) by arb), `(sin(x + y), x - 2y)`, and the pinned-coordinate systems of critique B1:
    `y - 1/4`, `3y - 1`, `y - 0.1`, the pinned circle, pinned then split; each zero in one unique box
    under 1e-12 wide), `::test_simple_rational_zeros_are_exact_points` (cusp, pole, kink, `sin` at
    0), `::test_the_first_step_splits_the_box`, `::test_not_c1_is_not_stepped` and
    `::test_not_c1_would_lose_a_zero` (the kink and the coupled kink),
    `::test_a_jump_is_caught_by_the_value_decoration`, `::test_c1_is_decided_on_the_closed_hull`;
    the helpers: `::test_krawczyk_proves_only_inside_the_interior`, `::test_gauss_seidel_step`,
    `::test_inverse`, `::test_precondition_falls_back_to_the_identity` (an exact midpoint beyond the
    doubles, critique B4), `::test_simplest_between`, `::test_the_simplest_point_needs_an_exact_zero`
    (the box case of critique B3), `::test_a_point_is_unique_only_when_f_is_exactly_zero`,
    `::test_inflation_is_clipped_to_the_region`, `::test_inflation_takes_its_own_jacobian` (a spy
    `F`), `::test_inflation_needs_the_c1_gate`, `::test_a_split_box_is_unproved`,
    `::test_a_bisected_box_is_unproved` (review S3),
    `::test_the_simplest_points_rest_is_its_own_region`,
    `::test_choose_falls_through_to_the_other_components` (critique N1, N2);
    `::test_n_equals_one_is_newton`, `::test_multi_piece_input`, `::test_unbounded_input`,
    `::test_constant_and_continuum_systems`, `::test_overflow_box` (critique B4),
    `::test_degenerate_input_component`, `::test_ends_beyond_the_doubles` (review F2),
    `::test_arguments_are_checked` (a decorated `xs` refused,
    one wording for a wrong-length `F` at n == 1 and n == 2, critique N7), `::test_warnings_stay_inside`,
    `::test_results_are_outward`, `::test_evaluation_budgets`
* **measured 2026-09-28** (five streams sharing the laptop):
  * `tests/test_gradient.py` 15 tests in 0.9 s; `tests/test_solve.py` 52 tests in 134 s
    (`::test_every_zero_is_enclosed` 104 s of it); after the review (its fixes in the tree):
    15 tests in 1.0 s and 54 tests in 66 s (`::test_every_zero_is_enclosed` 36 s; the laptop less
    loaded, not a speed-up); command
    `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q tests/test_solve.py --durations=8`
    (and the same for `tests/test_gradient.py`), `.hypothesis` cleared first
  * calls of `F` (every kind: plain, decorated, point), `tests/test_solve.py::_calls` with
    `max_steps=20000`, against the bounds of `::test_evaluation_budgets` (the five bounded rows
    re-measured after the review, 2026-09-28: unchanged):

| system | box | calls | bound | the break it catches |
|---|---|---|---|---|
| `(x + y, x - y)` | `REALS²` | 15 | 30 | round robin removed |
| circle and line | `REALS²` | 71 | 110 | `[-inf, inf]` not wide |
| circle and line | `[-1e300, 1e300]²` | 103 | 160 | magnitude split removed |
| cusp `(x ** 2 - y, y - x ** 3)` | `[-2, 2]²` | 371 | 560 | past-tol cap removed |
| system 7 | `[-6, 6]²` | 31 | 50 | (the exact midpoint: caught first by the pinned 1/3 system) |
| `(x ** 2 - 2, y - x)` | `[-10, 10]²` | 64 | — | |
| `(sin(x + y), x - 2y)` | `[-4, 4]²` | 148 | — | |
| kink | `[-1, 3] × [-1, 1]` | 102 | — | |
| `(exp x - y, x + y - 2)` | `[-5, 5]²` | 24 | — | |

  (the prototype's counts, `.scratch/h3b/nd-solver/probe13a.log`, were the same but the cusp's,
  359: the inflation costs 12 calls there)
  * the gate, from the worktree root, on the tree committed: `C:/Users/user/anaconda3/envs/intervals/python.exe
    -m pytest -q tests/itf1788`: 18246 passed in 40.3 s; the rest (`--ignore=tests/itf1788`: 4158
    collected) in four calls, each with `--ignore=tests/itf1788` and explicit paths: the four solver
    test files, `intervals` and `README.md`: 334 passed in 104.3 s; the first 14 other test files
    (alphabetical): 2361 passed in 192.3 s; the other 13: 1460 passed in 245.3 s; `tests/oracles.py`
    (its doctests): 3 passed in 0.2 s. sum 4158 passed in 542 s. the whole tree collects 22404 in
    one process (test basenames unique)
  * the gate after the review, from the worktree root, on the tree committed (2026-09-28, five
    streams sharing the laptop): `tests/itf1788`: 18246 passed in 48.8 s; the rest (4160 collected)
    in three calls, each with `--ignore=tests/itf1788` and explicit paths: the four solver test
    files, `intervals` and `README.md`: 336 passed in 115.6 s; the first 14 other test files
    (alphabetical): 2361 passed in 251.9 s; the other 13 and `tests/oracles.py`: 1463 passed in
    292.7 s. sum 4160 passed in 660 s. the whole tree collects 22406 in one process
    (`python -m pytest --collect-only -q`; basenames unique)
* **sabotage** (a throwaway harness, M15's shape: each break alone, the one replacement matching
  exactly once, `.hypothesis` cleared, `tests/test_gradient.py`, `tests/test_solve.py`,
  `tests/test_solver.py`, `tests/test_autodiff.py` and the two modules' doctests with `-x` and a
  900 s timeout, the file restored and compared; 2026-09-28). a no-op edit first: green, 235 passed
  in 248 s. the last column is the first test to fail under `-x`. the four green at first were
  closed by tests and re-run red; seven whose first run named only a hypothesis note were re-run to
  name the test; the other rows' final run is the first (the closing tests only add red paths):

| break | first run | final run: red by |
|---|---|---|
| jacobian transposed (columns as rows) | red | red: `tests/test_gradient.py::test_jacobian_encloses_the_partials` |
| column j read from pass j + 1 | red | red: `tests/test_gradient.py::test_jacobian_encloses_the_partials` |
| a constant coordinate seeded `[1]` | red | red: `tests/test_gradient.py::test_jacobian_encloses_the_partials` |
| a number beside decorated sets a bare point | red | red: `tests/test_gradient.py::test_arguments` |
| the up-front kind check removed (critique B2) | red | red: `tests/test_gradient.py::test_arguments` |
| C¹ gate removed | red | red: `tests/test_solve.py::test_simple_rational_zeros_are_exact_points[pole]` |
| C¹ gate reads the values only | red | red: `tests/test_solve.py::test_simple_rational_zeros_are_exact_points[kink]` |
| C¹ gate reads the partials only | red | red: `tests/test_solve.py::test_a_jump_is_caught_by_the_value_decoration` |
| jacobian over the box, not its closed hull | red | red: `tests/test_solve.py::test_simple_rational_zeros_are_exact_points[kink]` |
| krawczyk never proves in the step | red | red: `tests/test_solve.py::test_three_variables` |
| the inflation test removed (critique B1) | red | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[pinned 1/4]` |
| krawczyk inside `H`, not `int H` | red | red: `tests/test_solve.py::test_simple_rational_zeros_are_exact_points[kink]` |
| krawczyk ignores an empty K | red | red: `tests/test_solve.py::test_krawczyk_proves_only_inside_the_interior` |
| krawczyk on the box's own (open) ends | red | red: `tests/test_solve.py::test_krawczyk_proves_only_inside_the_interior` |
| gauss-seidel divides, not `mul_rev` | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| gauss-seidel not intersected with the box | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| gauss-seidel takes the hull of two pieces | red | red: `tests/test_solve.py::test_the_first_step_splits_the_box` |
| jacobi, not gauss-seidel | red | red: `tests/test_solve.py::test_gauss_seidel_step` |
| a split box keeps unique | green | red: `tests/test_solve.py::test_a_split_box_is_unproved` |
| preconditioner removed | red | red: `tests/test_solve.py::test_three_variables` |
| no pivoting in `_inverse` | red | red: `tests/test_solve.py::test_not_c1_would_lose_a_zero` (the coupled kink; `::test_inverse` comes later in the file) |
| range prune removed | red | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[circle and line, the reals]` |
| a point unique without `F == [0]` | red | red: `tests/test_solve.py::test_a_point_is_unique_only_when_f_is_exactly_zero` |
| simplest point removed | red | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[sin]` |
| simplest point accepts `0 ∈ F(p)` (critique B3) | red | red: `tests/test_solve.py::test_the_simplest_point_needs_an_exact_zero` |
| simplest point drops the rest of the box | red | red: `tests/test_solve.py::test_constant_and_continuum_systems` |
| exact midpoint fallback removed | red | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[pinned 1/3]` |
| round robin removed | red | red: `tests/test_solve.py::test_evaluation_budgets` |
| `[-inf, inf]` not wide (magnitude split only) | red | red: `tests/test_solve.py::test_evaluation_budgets` |
| magnitude split removed for bounded components | red | red: `tests/test_solve.py::test_evaluation_budgets` |
| past-tol cap removed | red | red: `tests/test_solve.py::test_evaluation_budgets` |
| tol ignored | red | red: `tests/test_solve.py::test_simple_rational_zeros_are_exact_points[cusp]` (after 735 s) |
| max_steps drops the stack | red | red: `tests/test_solve.py::test_every_zero_is_enclosed_on_a_budget` |
| bisection keeps one half | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| bisection drops the split face (M15's `_bisect`) | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| n == 1 not delegated to `newton` | green | red: `tests/test_solve.py::test_n_equals_one_is_newton[<lambda>-x0]` (the call counts) |
| work list: the first piece of each component | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| output not sorted | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` |
| inflation not clipped to the region (critique B1) | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` (a unique box with no zero) |
| inflation test with `J(B)`, not `J(H'')` | red | red: `tests/test_solve.py::test_inflation_takes_its_own_jacobian` |
| inflation test ignores the C¹ gate | red | red: `tests/test_solve.py::test_inflation_needs_the_c1_gate` |
| a gauss-seidel split's region is the piece | green (twice: the first closing case did not split) | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[pinned, then split]` |
| a bisection's region is the half | red | red: `tests/test_solve.py::test_simple_zeros_are_proved_unique[pinned circle]` |
| a split does not cut the region | red | red: `tests/test_solve.py::test_every_zero_is_enclosed` (a unique box with no zero) |
| the simplest point's rest keeps the whole region | green | red: `tests/test_solve.py::test_the_simplest_points_rest_is_its_own_region` |
| `_choose`: no fall-through past unsplittable wide components (critique N1) | red | red: `tests/test_solve.py::test_choose_falls_through_to_the_other_components` |
| simplest point on an unbounded component (critique N2) | red | red: `tests/test_solve.py::test_choose_falls_through_to_the_other_components` |
| krawczyk `m + b` (review S1) | green (the reviewer's run) | red: `tests/test_solve.py::test_krawczyk_proves_only_inside_the_interior` |
| the inflation of an unbounded box claims it (review S2) | green (the reviewer's run) | red: `tests/test_solve.py::test_choose_falls_through_to_the_other_components` |
| a bisected box keeps unique (review S3) | green (the reviewer's run) | red: `tests/test_solve.py::test_a_bisected_box_is_unproved` |
| the pass-length check removed (review S4, spec F2) | green (the reviewer's run) | red: `tests/test_gradient.py::test_arguments` |
| `solve` takes a bool as a number (review S5) | green (the reviewer's run, and ours with `match='got bool'`) | red: `tests/test_solve.py::test_arguments_are_checked` |
| `gradient` takes a bool as a number (review S5) | green (the reviewer's run, and ours with `match='got bool'`) | red: `tests/test_gradient.py::test_arguments` |
| `_combine` multiplies by an exact 0 (review S6) | green (the reviewer's run) | red: `tests/test_solve.py::test_precondition_falls_back_to_the_identity` |
| the step skips a `J` with an infinite end (review spec F5) | not run before | red: `tests/test_solve.py::test_overflow_box` |
| the simplest point not checked inside its component (review F3's claim) | red (the reviewer's run, by `::test_every_zero_is_enclosed`) | red: `tests/test_solve.py::test_the_simplest_points_rest_is_its_own_region` |
| `_point_in`'s `float(mid)` unguarded (review F2) | red (the crash itself) | red: `tests/test_solve.py::test_ends_beyond_the_doubles` |
| `solve`'s width halved by float division (review F2) | red (the next crash) | red: `tests/test_solve.py::test_ends_beyond_the_doubles` |

  47 breaks, 4 green at first, all red in the final runs; then the review's 11 (the rows from
  "krawczyk `m + b`" on; "first run" is the reviewer's harness over the stream's tests, or the
  crash before the fix; the final run is the closing test alone, `.scratch/fix/sab.py`,
  2026-09-28): 7 green at first and 1 not run before; 58 breaks in all, all red in the final runs. against the design's 36-row plan: its
  rows 1 and 3 are one break here ("a constant coordinate seeded"), row 1's place taken by the
  transposed jacobian; the rows for critique B1, B2, B3, N1, N2 and for the regions are new
* **the direction tag, not built**: the argument is in "design" above, corrected per critique B4 (an
  enclosure may hold an open end at inf by overflow, which keeps every real point; no degenerate
  `[±inf]` arises from a finite real), with the overflow box as a test (`::test_overflow_box`:
  `(exp x - y, x - 709.5)` over `[700, 720] × [1e307, 1.7e308]`, the zero enclosed)

## review

(2026-09-28, three read-only reviewers over `c8c9e08`, lenses soundness, sabotage-audit and
spec/regression; each finding reproduced on `c8c9e08` before any change, by
`.scratch/fix/repro.py` in the worktree (gitignored) or by the red run of its closing test; ids are
the reviewers', the soundness and spec lenses both using F1 to F3). **no wrong answer in the
solver**: every finding is a false claim in the text, a crash, a cost, or a property the tests did
not pin. found and fixed:

* **soundness F1 (blocking, a false claim)**: `autodiff.py::gradient`'s docstring and "design" above
  said dac or better makes `F` C¹ "on an open set holding the box". decorations are relative to the
  box: `gradient(lambda x, y: x ** 1.5 + y, [D(O(0, 1)), D(O(0, 1))])` is `[0.0, 1.5]_com
  [1.0]_com` with no point below 0 in the domain, and `abs(Dual.variable(D(O(0))))` is `[0]_com d
  [0]_dac`. both texts now say C¹ on the box relative to the box, which is what the mean value
  theorem on `H` needs (one-sided at its faces); the solver's argument never used an open set, so no
  answer changes
* **soundness F2 (minor, a crash, shared with `newton` at `04946af`)**: `solve(lambda x, y: (x - 3 *
  10 ** 400, y - x), [M(10 ** 400, 10 ** 401)] * 2)` and `newton(lambda x: x - 3 * 10 ** 400, M(10
  ** 400, 10 ** 401))` raised `OverflowError` in `solver.py::_point_in`'s `float(mid)`. it now falls
  back to the exact midpoint there; the next crash on the same input, `solve`'s `width <=
  _width(box) / 2` (int true division), is now `2 * width <= _width(box)`. pinned by
  `tests/test_solve.py::test_ends_beyond_the_doubles` (red before: `OverflowError`), n == 2 on a
  budget (`max_steps=10`) since the unbudgeted solve costs 13669 calls, 27 to 46 s (2026-09-28, loaded
  laptop; it ends with the exact point, unique): the float `Y` overflows `b` to `(1.8e308, inf)`, so
  the step is idle and the box is bisected. recorded under known limits. `newton`'s own `width <=
  piece.wid() / 2` is left as M15 wrote it (still owed)
* **soundness F3 (minor, a cost)**: `solve(lambda x, y: (0, 0), [M(0, 1)] * 2, tol=1e-2)` returns
  8056 boxes. measured (2026-09-28, `.scratch/fix/f3.py`): 8056 is `max_steps=10_000` cutting the
  bisection short; with `max_steps=10 ** 6` it is 64720 boxes (12996 unique points), 492580 calls,
  568 s; `tol` 0.25, 0.1, 0.05 give 34, 688, 3312 boxes. **the cascade the finding describes does not
  happen**: a rest box's closed hull holds the point, the simplest rational of a larger set, so the
  point is the rest box's simplest point too, and not inside it: `_simplest_point` is None there and
  no second point is drawn. the output is at most 2n + 1 times the boxes of width `tol`, which a
  continuum needs anyway. so the suggested fix (skip the point on rest boxes) would save no call of
  `F`; not built. recorded under known limits, and the no-cascade claim pinned in
  `tests/test_solve.py::test_the_simplest_points_rest_is_its_own_region`
* **sabotage S1 (blocking)**: krawczyk's `m - b` made `m + b` stayed green: every unit case had `m`
  at the midpoint, where the two mirror. pinned in
  `tests/test_solve.py::test_krawczyk_proves_only_inside_the_interior` with `m = (1/4, 1/4)`: `b =
  (-1/2, 0)` proves the zero `(3/4, 1/4)`, `b = (1/2, 0)` does not claim `(-1/4, 1/4)`
* **sabotage S2 (blocking)**: `_inflated_unique` returning True where the inflation is None
  stayed green: `tests/test_solve.py::test_choose_falls_through_to_the_other_components` now asserts
  that no root of `(1 / x, y - 0.5)` over `(MAX, inf] × [-1, 1]` is unique (as built: one root,
  unproved)
* **sabotage S3 (blocking)**: a bisected box keeping its `unique` flag stayed green: the new
  `tests/test_solve.py::test_a_bisected_box_is_unproved` makes `_krawczyk` claim the first box and
  `_gauss_seidel` narrow it to `[0.5, 20] × [-1, 1]` (wide in x), which the next pop bisects; with
  `max_steps=2` both halves are output unproved
* **sabotage S4 and spec F2 (minor, one finding)**: `_passes`'s "F returned sequences of different
  lengths" had no test: `tests/test_gradient.py::test_arguments` now calls `jacobian` with an `F`
  of two outputs on pass 0 and one on pass 1 (under the break: `IndexError`)
* **sabotage S5 (minor)**: the bool refusals in `solver.py::_input_box` and `autodiff.py::_box`
  dropped stayed green. the reviewer's closing `match='got bool'` stays green too, since
  `MultiInterval(True)` itself raises `TypeError: expected a real number, got bool: True`: the
  break changes the wording, not the refusal. the tests (`::test_arguments_are_checked`,
  `tests/test_gradient.py::test_arguments`) pin each function's own wording, `match='or a number,
  got bool'`
* **sabotage S6 (minor)**: `_combine`'s exact-0 skip made unconditional stayed green: `0.0 * (1,
  inf)` is already `[0]`, and only `0 * [inf]` is `{}`. pinned in
  `tests/test_solve.py::test_precondition_falls_back_to_the_identity` (`_combine((0.0, 1.0),
  (O(inf), O(2))) == O(2)`), and the docstring narrowed to an entry `[±inf]`
* **spec F1 (minor, an overstated exit)**: the random oracle's `abs` and `cbrt` factors run only
  under a budget and do not detect the C¹ gate removed. the exit line and the cost bullet above now
  say so; the gate stays pinned by the pole, kink, coupled kink and jump examples (the sabotage
  table)
* **spec F3 (minor)**: the README's package-layout line was missing from "readme"; added there
* **spec F4 (minor, a number)**: "about 1.5x the calls" at n == 1 held for two of the four
  functions; re-measured 2026-09-28 (49/32, 145/93, 32/16, 47/33), now "1.4x to 2x" with all four,
  in the record and in `tests/test_solve.py::test_n_equals_one_is_newton`'s docstring
* **spec F5 (minor, an unpinned docstring)**: `tests/test_solve.py::test_overflow_box` claimed the
  step runs with an infinite end in `J` but asserted only the enclosure; a spy on
  `solver._precondition` now asserts that some `J` it sees has one (red when the step skips such a
  `J`)

sabotage of the review's fixes: `.scratch/fix/sab.py` in the worktree (a throwaway harness, each
break alone, its closing test run alone with `-x`, `.hypothesis` cleared, the file restored and
compared; 2026-09-28). the rows are in the table above, from "krawczyk `m + b`" on

## readme

(for `README.md`: the example block, after the `newton` lines; deterministic, measured 2026-09-28)

```
>>> from intervals import gradient, solve
>>> print(*gradient(lambda x, y: x * y ** 2, [MI(1, 2), 3]))     # n passes, one variable seeded each
[9] [6, 12]
>>> for root in solve(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [MI(-10, 10), MI(-10, 10)]):
...     print(root.unique, *root.box)                            # a square system: krawczyk proves
True (-0.7071067811865476, -0.7071067811865475) (-0.7071067811865476, -0.7071067811865475)
True (0.7071067811865475, 0.7071067811865476) (0.7071067811865475, 0.7071067811865476)
```

and a bullet for "what it does", after "interval newton (M15)":

* **several variables** (M16a): `gradient(f, xs)` and `jacobian(F, xs)` are n passes of forward-mode
  autodiff, one variable seeded each. `solve(F, xs)` returns `RootBox(box, unique)`es holding every
  zero of a square system `F` in the box `xs`: gauss-seidel with `mul_rev` narrows (a partial holding
  0 splits the box in one step), krawczyk's test proves a zero unique, on the closed hull and, for a
  converged box, once more on the box inflated within the part of the input it stands for; a zero at
  a simple rational is output as that exact point. the step runs only where the decorations prove
  `F` C¹ on the box

and in the package layout (README.md, the `intervals/` line; review spec F3): `autodiff` (`Dual`)
becomes `autodiff` (`Dual`, `gradient`, `jacobian`), and `solver` (`newton`) becomes `solver`
(`newton`, `solve`, `RootBox`)

## Q12

(owner questions for `HANDOFF.md`, each built as the stated default meanwhile)

* **Q12(a)** a jacobian as n passes with `Dual` untouched (default), or vector mode (a tangent tuple
  inside `Dual`, one pass, M15's chain rules edited)? measured only indirectly: a box costs n + 2
  calls of `F` either way and the library's ops dominate
* **Q12(b)** names `gradient`, `jacobian`, `solve`, `RootBox` (default), or `newton_system` /
  `krawczyk` and a widened `Root`? folds into Q11 (keep, rename or narrow the public surface)
* **Q12(c)** zeros on split faces that are not simple rationals stay unproved, with unproved slivers
  beside proved ones (default); or bisect off-centre (measured on the prototype: 1 to 3 of 4 face
  zeros proved before the simplest point existed). noise, not error
* **Q12(d)** the simplest-point rule and the inflated krawczyk test are additions beyond H3's wording
  (a gradient, a jacobian, krawczyk), each measured to be needed (decision log above): keep (default)?
* not a question, recorded: hansen and sengupta's uniqueness test not used; smear, the mean value
  prune and "any component halved" measured on the prototype and left out

## still owed

* the orchestrator's merge of the sections above into `v2-plan.md`, `v2-implementation-plan.md` (§0
  D20, §2 M16a), `HANDOFF.md` (the H3 row, Q12, the banner, the session log) and `README.md` (the
  example block needs the `MI` import README's block already has, and a blank line before the
  closing fence; its output was doctested 2026-09-28 in a scratch copy)
* `tests/test_applicator.py::test_package_exports_unchanged` is edited here (the four names); other
  M16 streams that export names edit the same set: a merge conflict to resolve by union
* not pushed, so CI has not run the two new files; the local gate is the only evidence
* a constructed n = 3 system with two zeros costs more than 120 s: n = 3 is covered by one
  constructed zero and the sphere only, with no random n = 3 test. whether a faster jacobian
  (Q12(a)) or a tighter form of `F` would change that is not measured
* the natural path to a split of a box already proved unique was not found; the rule is pinned by a
  monkeypatched `_krawczyk` only
* sabotage rows red in the first run were not re-run after the four closing tests were added (those
  tests only add red paths), nor after the review's closing tests (the same reason)
* `newton`'s `width <= piece.wid() / 2` (M15) would raise `OverflowError` on an exact piece wider
  than the doubles once a step narrows without proving (int true division, as `solve`'s copy did
  before the fix; not observed in `newton`); not reached by a linear `f` (proved at the
  first step), and a nonlinear `f` there is impractical anyway (`x ** 2 - 9 * 10 ** 800` over
  `[10 ** 400, 10 ** 401]` with `max_steps=10` did not finish in 2 minutes, 2026-09-28: the exact
  fractions grow). `solve`'s copy of the line is fixed (review F2)
* the review's F3: a continuum costs up to 2n + 1 output boxes per box of width `tol`; whether to
  output a continuum differently (one box per connected unproved region) is not asked (Q12 has no
  item for it)
