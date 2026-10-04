# `MultiInterval` v2 implementation plan (sketch, 2026-09-23)

companion to `v2-plan.md`. that file says *what*; this one says *in what order*, with an exit
criterion per milestone. review findings that needed an owner decision are in section 0; D1–D7 are settled or deferred as
of 2026-09-23 and written into `v2-plan.md`'s "current design"; D8 was settled 2026-09-24. D9–D17
were settled 2026-09-25 for M13 and M14: they go into "current design" as each is built, and until
then they are in `v2-plan.md`'s decision log. D14 (the flint oracle) and D15 (vendoring) were built
with M13a and M14's first two items (2026-09-26) and are in "current design" now. D19 (M15) and
D20–D24 (M16, 2026-09-28) are the build's own defaults, written into "current design" as built and
open for the owner in `HANDOFF.md` (Q11 to Q16).

**open work and open questions live in `HANDOFF.md`** (since 2026-09-26): ranked items, questions for
the owner, loose ends, session log. this file keeps the spec (what to build, exits) and the records.

## 0. decisions (D1–D8 from the 2026-09-23 reviews; D9–D17 for M13 and M14, 2026-09-25; D18 for M13, D19 for M15, 2026-09-27; D20–D24 for M16, 2026-09-28; D27–D29 from the owner's answers, 2026-10-03)

| # | question | recommended default | blocks |
|---|---|---|---|
| D1 | **decided: recommended default.** closure at infinity: `1/(-1, 0)` is written as `[-inf, -1)`, but the involution claim and `1/[1, inf)` = `(0, 1]` both need the flag to *propagate*: `(-inf, -1)`. state the rule as "±inf are ordinary points; an infinite endpoint is closed iff attained; a pole at a **closed** zero endpoint attains ±inf by the piece's sign". drop the "closure over limits" wording | propagate flags; `1/(-1,0)` = `(-inf,-1)` | M6 |
| D2 | **decided: recommended default.** indeterminate corners: `[-inf,-1] * [0]` is written as the entire line, but `1/[-1,0]` already uses the sharp limit-along-the-box rule. the same rule for mul: at an indeterminate corner `(±inf, 0)` the corner contributes `0` if the infinite factor's interval is non-degenerate, and the signed infinity if the zero factor's interval is non-degenerate. so `[-inf]*[0,1]` = `[-inf]`, `[-inf,-1]*[0]` = `[0]`, `[-inf,-1]*[0,1]` = `[-inf,0]`, `[1,inf]/[1,inf]` = `[0,inf]`, all matching 1788 up to closure at inf. general form: an indeterminate corner contributes the limit along each non-degenerate edge that meets it, so for sub at `(inf, inf)` it contributes `-inf` if the minuend is non-degenerate and `+inf` if the subtrahend is (`[inf]-[1,inf]` = `[inf]`, `[1,inf]-[inf]` = `[-inf]`, `[1,inf]-[1,inf]` = entire, as 1788); add at `(inf, -inf)` likewise. a box that *is* the indeterminate point returns `∅` + warning (D7). through the itf1788 adapter's input rule (1788 unbounded → open at inf) the infinite corner is never in the box, so D2 does not change conformance — it only affects user-typed literal `[-inf, …]` bounds | sharp rule | M6 |
| D3 | **decided: recommended default**, plus integral Fractions normalize to int in `Cut`. `int / int` that is not integral: Fraction (exact, per "never rounded") or float (what users expect)? | Fraction; `fmt` prints `1/3`; float only if an operand is float | M6 |
| D4 | **deferred**: the time layer is not being rebuilt now; v1's is archived with the rest of v1 at M10 and comes back in M8 on top of the v2 class (see M8, M10). infinities for the time layer: v1 stores float unix seconds (loses sub-µs, dodges the question). v2 options: (a) Fraction seconds in the numeric kernel, thin wrapper; (b) native datetime cuts + two sentinel objects that compare below/above everything | (a) — reuses every kernel test unchanged, when M8 happens | M8 (deferred) |
| D5 | **decided: every sign combination before release**, as its own milestone (M7b); the recommended Q1-only v2.0 is rejected. modulo scope for v2.0: the v3 work covers A ≥ 0, B > 0 only; Q2 primitive and zero-crossing operands are underived (design notes §4) | full modulo, M7a then M7b | release |
| D6 | constructor default for an infinite bound: `MI(1, inf)` = `[1, inf]` (literal) or `[1, inf)` (1788 reading)? **settled (v2-plan.md current design): literal** — `[a, inf]` and `[a, inf)` are different sets and "a user-typed `[1, inf]` is taken literally". D2 removes the blow-up footgun that made this look open | literal | — |
| D7 | **decided 2026-09-23 by owner**: a box that *is* an indeterminate point (`1/[0]`, `[0]*[inf]`, `[inf]-[inf]`, `[0]/[0]`) returns `∅` + `IndeterminateResultWarning` (was `[-inf] ∪ [inf]` / entire). isotonicity forces it: `[0]` is inside `[-1,0]` and `[0,1]`, so `1/[0] ⊆ [-inf,-1] ∩ [1,inf] = ∅`; `[0]*[inf] ⊆ [0]*[5,inf] ∩ [0,1]*[inf]` = `∅` and `[inf]-[inf] ⊆ [inf]-[1,inf] ∩ [1,inf]-[inf]` = `∅` under D2. solvers need isotone ops; matches 1788's empty. cost: `1/(1/[inf])` = `∅`; `1/x` round-trips only on sets with no degenerate piece at `0`, `inf`, `-inf`; the "later" direction tag stays the recovery path. separately, `f(A ∪ B) == f(A) ∪ f(B)` fails for reciprocal with any `1/[0]` (`A=[-1,0)`, `B=[0]`), so that law is only `⊇` for reciprocal/div | `∅` + warning | M6 |
| D8 | **decided 2026-09-24 by owner: recommended default**, implemented in M7. modulo with infinite operands: python gives `inf % 3` = `nan`, `3 % inf` = `3`, `-3 % inf` = `inf`. a dividend of ±inf attains nothing, so it is dropped with `DomainClippedWarning`; a finite dividend mod an infinite divisor follows python's scalar result (also the limit along the box) | clip / follow python | M7b |
| D9 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13b), now in `v2-plan.md` "set operations and size".** 1788's numeric ops on a multi-interval. `mid`, `rad`, `wid` (and `midRad`) are **of the hull**: a midpoint outside the set (`mid([0,1] ∪ [9,10])` = 5) is still a valid bisection point, a per-piece form would return a tuple, and `size.length` already gives the width without the gaps. `mag` and `mig` are **of the set**, as `sup` and `inf` of `{abs(x) : x ∈ A}`: `mig([-3,-2] ∪ [2,3])` = 2, where the hull would give 0; on a connected set the two readings agree. unbounded operands follow 1788 (`mid` of entire is 0, of a half-bounded set ±max float; `rad` and `wid` are inf); the empty set raises `ValueError`, as `.inf` does today, and the adapter maps it to 1788's `NaN` | hull for mid/rad/wid, set for mag/mig | M13b |
| D10 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13c), now in `v2-plan.md` "comparisons" and "set operations and size".** names for 1788's interval orders, since `<` and `<=` are pointwise and return a `TruthSet` (`MI(1,3) < MI(2,4)` is `BOTH`): `A.weakly_less(B)` for `less` (both of A's ends ≤ B's; 1788's own wording, "weakly less than"), `A.strictly_less(B)` for `strictLess`. `interior` is not a method: a new property `B.interior` (the set with every end opened, a set operation in its own right) and the existing `A.within(B.interior)` | `weakly_less`, `strictly_less`, `.interior` | M13c |
| D11 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13d), now in `v2-plan.md` "arithmetic" and "elementary and step functions".** power. an `int` exponent, or a float with an integral value, is `pown` as today, like python's scalars (`(-3.0) ** 2.0` = 9.0; `MI(-3,1) ** 2.0` = `[0, 9]`). a non-integral float or a `MultiInterval` exponent is 1788's `pow`: domain `x > 0`, plus `x = 0` where `y > 0` (`0 ** y` = 0); negative bases are dropped with `DomainClippedWarning`. so `MI(-3,1) ** MI(2)` = `[0, 1]`, not `[0, 9]`: an interval exponent means `pow`, never `pown`. `b ** A` for a scalar base is `MultiInterval(b) ** A` through `__rpow__`. exact where the value is rational, as `log` is. 3-argument `pow(A, n, m)` is dropped (not 1788; v1 had it on integers only) | pown for integral, else 1788 pow; 3-arg dropped | M13d |
| D12 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13e), now in `v2-plan.md` "elementary and step functions", "empties and warnings" and "ieee 1788".** a reverse op whose answer has infinitely many or very many pieces (`sinRev`, `cosRev`, `tanRev` and their `*Bin` forms over an unbounded or wide `x`): the exact pieces up to the step functions' cap of 1000, past it their hull with a `HullWarning`, the rule `steps.py` already follows. a bounded `x` gets the exact union (`sinRev([0.5, 1], [0, 20])` has 4 pieces), which 1788 cannot give. ends are irrational, so each is its tightest float enclosure, open | exact to 1000 pieces, else hull + warning | M13e |
| D13 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13f), now in `v2-plan.md` "arithmetic" and "ieee 1788".** `cancelMinus(A, B)` is the **Minkowski difference**, the largest `X` with `B + X ⊆ A`, which is defined on any multi-intervals; `cancelPlus(A, B)` is `cancelMinus(A, -B)`. for connected `A` and `B` it is exactly 1788's answer whenever 1788 has one (`[a1 - b1, a2 - b2]` when `wid A ≥ wid B`). where 1788 has no answer it returns entire as a "no answer" signal, and ours is a real set: `cancelMinus([-inf,-1], [-1,5])` = `(-inf, -6]`, and `∅` when nothing fits. those vectors are divergence rows under a **new residual category, "cancellation as a Minkowski difference"**, approved with this decision | Minkowski difference; new divergence category | M13f |
| D14 | **decided 2026-09-25 by owner: recommended default.** the independent oracle for the elementary functions is **`python-flint`** (Arb: every result is a ball proven to contain the true value), a test-only dependency in the `[test]` extra, installed into the `intervals` env and in CI. `mpmath` was the alternative (pure python, but its values carry no proven bound). python-flint 0.9.0 has Windows wheels for 3.10 and later (abi3), 3.13 and 3.14 included (checked on PyPI 2026-09-25) | python-flint | M14 |
| D15 | **decided 2026-09-25 by owner: recommended default.** licence. `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` are LGPL-2.1-or-later, the two `ieee1788-*.itl` files carry an all-permissive notice, the rest Apache 2.0, and this repo has no licence of its own. vendor all 19 files of oheim/ITF1788 at `b6ee1e2` unmodified into `tests/itf1788/`, replacing nehmeier's 7, with the fork's `LICENSE`, `NOTICE` and `COPYING.LESSER` beside them. the wheel ships only `intervals/`, so no test file is distributed with the library. **corrected 2026-09-26 at M13a**, from every file's header: five files carry the all-permissive notice, not two (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`); the eleven `libieeep1788_*` are Apache 2.0 | vendor unmodified, with the licence files | M13a |
| D16 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13g), now in `v2-plan.md` "ieee 1788" (`DecoratedInterval`, `UndefinedOperationError`, `PossiblyUndefinedOperationWarning`).** decorations (com/dac/def/trv/ill), NaI and 1788's constructors go in a **separate decorated wrapper type**: the solver stack's (M11), brought forward. the core `MultiInterval` stays undecorated, so `v2-plan.md` "ieee 1788" ("decorations are not in the core") holds. 1788's signals, owner 2026-09-26 (`v2-plan.md` "2026-09-26 revision: owner answers"): `UndefinedOperation` **raises** (a `ValueError` subclass, so it reads like `MultiInterval(2, 1)`'s `ValueError`); `PossiblyUndefinedOperation` is an **`IntervalWarning` subclass** (the result is returned); names chosen when built. **no NaI** and no `ill` (owner 2026-09-26): its statements are rows under a new category, M13g | wrapper type; signals as the owner chose | M13g |
| D17 | **decided 2026-09-25 by owner**: M13 does **not** block the 2.0.0 release, and there is no hurry to release either ("I have zero users and this is a yak shaving pet project"). M13 only adds methods and a type and gives a meaning to exponents that raise `TypeError` today, so nothing that works now changes | release whenever; not blocked | — |
| D18 | **decided 2026-09-27 by owner**, on M13's proposed categories and choices: (a) **"tighter than the vector"** is an approved residual category (M13e): 11 keys, 18 vectors where 1788's expected hull is looser than the tightest double enclosure and ours is the tightest, checked with arb or exactly; the two grossly loose `pow_rev.itl:609`, `:642` stay in it. (b) **"exact parsing decides validity"** is approved (M13g): the 1788 text constructors read bounds exactly, so no `PossiblyUndefinedOperation` for a near-tie literal; 7 keys. (c) the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY` 12) stay under **decoration expectations**. (d) `set_dec` **demotes** as 1788's `setDec` does; only the `DecoratedInterval` constructor raises | both categories approved; rows stay; set_dec demotes | M13e, M13g |
| D19 | **decided in the build 2026-09-27 (the session's defaults); confirmed 2026-10-03 by owner (Q11, as built; `tol` documented as absolute, `max_steps` as boxes).** the solver stack's first part (M15): (a) `intervals/autodiff.py` (`Dual`, `derivative`) and `intervals/solver.py` (`newton`, `Root`) are public and exported from `intervals`, not newton as a test only; (b) newton's step runs only where `f` is proved C¹ by decorations (dac or better on the value and the derivative), else the piece is pruned and bisected; (c) the step is `mul_rev`, never `/` (D7); (d) one variable; (e) `tol=1e-10` absolute, `max_steps=10_000` | as built | M15 |
| D20 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q12, as built).** the solver stack's second part (M16a): (a) a gradient or a jacobian is n passes of `F`, `Dual` untouched (not vector mode); (b) names `gradient`, `jacobian`, `solve`, `RootBox`, public and exported from `intervals`, `Root` unchanged; (c) `solve` at n == 1 is `newton`; (d) uniqueness by krawczyk only, on the closed hull, with a float preconditioner (identity fallback), narrowing by gauss-seidel with `mul_rev`; (e) before an unproved box is output, its simplest rational point (exactly `[0]`: a unique point) and then krawczyk on the box inflated within its region; (f) wide components bisected first, round robin, then the widest; (g) `tol=1e-10` absolute on the widest component, `max_steps=10_000` boxes; (h) no direction tag | as built | M16a |
| D21 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q13) but (b): the layer's numbers of the empty set are `nan`, 1788's answer, not the library's `ValueError`; Q9 and Q10 closed as built.** the 1788 layer (M16b): (a) `intervals/ieee1788.py`, one `Interval` class for both flavours, 1788's names in snake_case (`NAMES` has the camelCase), not exported from `intervals`; (b) `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad` of the empty set raise `ValueError`, where 1788 says NaN; (c) Q9 answered by the layer: `ieee1788.mul_rev_to_pair` is 1788's pair with its decoration, the library's `mul_rev` unchanged; (d) Q10 answered by the layer: its pass runs the constructors in binary64, no class argument on the library's; (e) where 1788 defines another answer than the library's set (cancellation, overlap, attained infinities), the layer gives 1788's and the library keeps its own | as built | M16b |
| D22 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q14, as built; every cut-tuple relation asserts normalized operands).** the per-piece allen matrix (M16c): (a) `A.allen_matrix(B)`, a tuple of tuples of `Allen` (rows the pieces of `A`, columns those of `B`), and `relations.allen_matrix` over cut tuples; (b) `A.allen_relations(B)`, the `frozenset` of the relations holding between some pair of pieces, a second public name the H3 row did not list; (c) an empty operand gives `()` / one empty row per piece / `frozenset()`, no raise and no warning; (d) the matrix is the plain `n x m` loop over `allen()` (no dependence on normalized input; the design's ~2-3x faster fill + sweep not taken), the set view an `O(n + m)` sweep that never builds the matrix; (e) methods on `MultiInterval`, functions in `relations.py`, nothing at the top level, not on `DecoratedInterval`, the sparse `(i, j, relation)` view private | as built | nothing (additive: `allen()` and every existing name unchanged); M16c's record |
| D23 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q15): (a), (b), (d), (e), (h) as built; changed: (c) `==`/`!=` against an ndarray is elementwise, (f) `fmin`/`fmax` are `minimum`/`maximum`, (g) numpy is in `[test]`; and the methods follow (h)'s rule (a mixed method call returns the class the operators do).** numpy interop (M16d): (a) `__array_ufunc__` on `MultiInterval`, `DecoratedInterval`, `Dual` (`intervals/numpy_compat.py`), operator ufuncs as python's operators on our dunders only, the others the method of the same set image, the rest `TypeError`; `__array__` on `MultiInterval` only (a 0-d object array); the array API standard not built (the alternative: an interval-array type); (b) a foreign real is its exact value (a `Rational` by type, else where `float()` would round), alternatives refuse or keep `float()`; (c) an ndarray meeting ours is elementwise into an object array, `==`/`!=` never broadcast; (d) no numpy-named alias methods; (e) `np.invert` the complement; (f) `fmin`/`fmax` not mapped; (g) numpy not in `[test]`; (h) both operands ours in a method ufunc (`hypot minimum maximum arctan2`): the subclass decides, as for the operators | as built | M16d |
| D24 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q16): (a)-(c), (f) as built; changed: (d) `[fast]` pinned to `auto`'s window as `[test]` is, (e) one CI gate job on the forced gmpy2 backend.** the gmpy2/mpfr backend (M16e): (a) the default is the pure path; `INTERVALS_BACKEND=gmpy2` forces gmpy2 (ImportError if missing or below 2.3 / MPFR 4.2), `auto` takes it if importable and `2.3 <= version < 3`; (b) public surface: the env var and the `[fast]` extra only; `intervals.backend.name()` not exported, no setter; (c) non-dyadic points stay pure (no mpfr ziv loop), but atan, acot, atan2's angles and the hook's mixed operands; (d) `gmpy2>=2.3,<3` in `[test]`; (e) CI unchanged: the whole suite on the pure path, `tests/test_backend.py` compares both in every job; no gmpy2 fuzz job; (f) ships in 2.0 as an opt-in, or waits under "later" | as built | M16e |
| D25 | **decided 2026-09-28 by owner**: CPython 3.11's `Fraction.__pow__` rounds a Fraction base to a float before `MultiInterval.__rpow__` runs, so on 3.11 `Fraction(1, 3) ** OutwardMultiInterval(2)` misses 1/9, and nothing in the library can see it (the other Fraction operators defer correctly; 3.12 returns NotImplemented). drop 3.11, or keep it with `Fraction ** interval` documented as unsupported there? | **python >= 3.12**: `pyproject.toml` `requires-python`, CI's gate matrix 3.12-3.14; `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11 | — |
| D26 | **decided 2026-09-30 by owner** (fuzz-rev-inf): to nearest, a reverse op's exact preimage wholly past MAX squeezes to the point `[±inf]` (IEEE 754 rounds such a value to ±inf; `_widened` reads it as `[MAX, inf]`), and intersecting with `x` after that rounding lost it whenever `x` is open at that infinity: `pown_rev(c, -1, (-inf, -2))` for `c = (-2.2e-309, 0)` was `{}` though its exact answer `(-inf, -4.49e308)` is not empty; the same at a finite double (`sqr_rev([2, 2.0000000000000004], (1.4142135623730951, 2])` was `{}`). options weighed with the owner: (a) `x` meets the preimage before the rounding, 1788's order; (b) keep it and document it; (c) the nearest class saturates an overflow to `(MAX, inf)`, 1788's enclosure rule, no longer python's float | **(a)**, in the reverse ops only (`reverse._keep_squeezed`): a part of the exact answer inside `x` that rounds wholly onto one double is that double, as a point: an end `x` excludes (the case above; the one way a result leaves `x`), or a point of `x` where the rounding kept an end open (`mul_rev(10, (1, 2), [0.1])`, whose exact answer holds the double 0.1, was `{}`, now `[0.1]`; a known loss M13e's tests had worked around by leaving `x` out of their checks). the nearest class keeps IEEE 754 round-to-nearest (as python's float) everywhere; forward ops unchanged: `[inf] & (0, inf)` is still `{}` (documented, README "rounding"). not a 1788 divergence row: the vectors test the outward class, which was already right | `intervals/reverse.py`; `tests/test_reverse.py::test_float_operands` (two `@example`s), `::test_exactly_the_points_with_f_in_c` |
| D27 | **decided 2026-10-03 by owner** (the 1788 departures census, 2026-09-30, never asked before): four departures from 1788 that were build choices are deliberate: step functions are point sets (`floor([-1.5, 1.5])` is four points, 1788 `[-2, 1]`); an end that rounding moved is open (M12); the divergence categories "degenerate infinities" (D1, D6 and the domain-end rule) and "cut-based relations" (the 2026-08-16 principles). the stale category "domain-clipped functions" (no row since M13d) is removed from `tests/itf1788/test_itf1788.py::REASONS` and the current design | as stated | `references/owner-questions-2026-10-03/ieee1788.md` |
| D28 | **decided 2026-10-03 by owner** (Q17, Q18, m14b-open's 4300 digits): pown of exact operands past one exact-result limit of about 2 ** 22 bits, shared by pown, `pow_` and exp2/exp10, is the tightest open float enclosure in the outward class and the value rounded to nearest in the nearest class, with a default-ignored warning; `elementary.EXACT_POWER_LIMIT` stays the float-corner threshold. pown to nearest is correctly rounded for every n (the exact power rounded once, `rounded_pow` past the threshold), not libm's `float ** int`. `repr` does not raise past python's 4300-digit limit (hex past it; `parse` reads it) | as stated | `references/owner-questions-2026-10-03/pown.md`; §2 "owner-answers" |
| D29 | **decided 2026-10-03 by owner** (Q19, Q20): each end keeps its own number type. the outward class is isotone within one grid; across grids `f(A)` lies within the tightest double cover of `f(B)` (documented; a public method rounds every end onto the double grid, outward). the exact class's crossed piece is the piece between the two values, each end keeping its flag (`rootn((10 ** -30, 1.0000000000000003e-30], 5)` is `[1e-06, 1/1000000)`) | keep per-end typing | `references/owner-questions-2026-10-03/q19.md`, `q20.md`; §2 "fuzz-steps-isotone", "fuzz-rootn-crossed" |

implementability review (2026-09-23, second pass), written into "current design" and the
milestones below: infinite result endpoints always go through attainment (the corner-flag rule is
wrong at ±inf); mul splits at zero; ±inf is exact; `-` is arithmetic, set difference is a method;
`==` does not coerce; relations return bool; import-time filters `append=True`; itf1788 hulls the
expected value too; `Cut` normalizes in a subclass because `typing.NamedTuple` refuses `__new__`.

smaller gaps, accepted and written into "current design" 2026-09-23: `Size` has componentwise
`+`; `TruthSet.certainly`/`.possibly` on `{}` are True/False; `nan` in a constructor is a
`ValueError`; `x[a:b]` restricts to closed `[a, b]`; the import-time `'ignore'` filters are
process-global and overridden by `pytest -W` / `simplefilter`; itf1788 licence check before
vendoring.

## 1. environment and gate

* env exists: `C:/Users/user/anaconda3/envs/intervals/python.exe`, Python 3.13.15, pytest 9.1.1,
  hypothesis 6.167.1, numpy, pandas (measured 2026-09-23; `v2-plan.md`'s size section now records the executed v1
  reproduction); python-flint 0.9.0 since M14 (2026-09-26, D14), in the `[test]` extra as
  `python-flint>=0.9`, so CI's `pip install -e ".[test]"` picks it up
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root. add
  `pyproject.toml` (package metadata, `[tool.pytest.ini_options] testpaths = ["tests"]` and
  `pythonpath = ["."]` so the v1 modules at the root and the `intervals/` package import without
  relying on `python -m` putting cwd on `sys.path`, and a
  `filterwarnings` entry turning the library's own warnings into errors inside the suite once
  the warning classes exist)
* CI (added on `v2` 2026-09-25, by owner request): `.github/workflows/ci.yml` runs on every push and
  pull request. it runs the gate on Python 3.12 to 3.14 (3.11 to 3.14 until 2026-09-28, D25) and each exhaustive harness as its own job:
  `tests.exhaustive_ops` exact, `--float` and `--sabotage`, and `tests.exhaustive_modulo`.
  under GitHub Actions hypothesis loads its built-in `ci` profile (derandomized, no deadline).
  `tests/conftest.py` (M14, 2026-09-26) loads nothing unless `HYPOTHESIS_PROFILE` is set, so the
  gate still gets hypothesis's own choice (`default` locally, `ci` under Actions); set to `fuzz` it
  runs every hypothesis test randomized at `FUZZ_MULTIPLIER` (default 10; 100 until 2026-09-27) times its examples,
  which `.github/workflows/fuzz.yml` does on every push to `master` and on `workflow_dispatch`
  (weekly and never on push until 2026-09-29, `v2-plan.md` "2026-09-29 revision: fuzz on push"), and
  `tools/prepush.sh` locally before a push (M14).
  first run 2026-09-25 at `d232b78` (run 36091651163), all 8 jobs green:
  the gate took 71-94 s on each python, and on the runners the exhaustive jobs took 21 s
  (sabotage), 5 min (float), 6½ min (exact) and 12½ min (modulo). each harness exits nonzero on a
  failure. M13e/g's first run, 2026-09-27 at `dbec908` (run 36293351201): the 4 gate jobs each
  `1 failed, 22165 passed`, `tests/test_reverse.py::test_mul_rev_float_operands` at a `b` whose
  quotient passes the largest double, where to nearest gives the piece `[inf]` (as designed) and
  the oracle `_widened` skipped infinite ends; the local gate missed it (randomized profile; `ci`
  is derandomized). fixed in the oracle, the example pinned (`d7e46c2`); run 36305984327 at
  `a1d45a9` all 8 green, the gate 272-285 s on each python. M15 and M16's first, CI run 36402681261 at `3aaf8f4` (M15 and M16's first, 2026-09-28): 7 of 8 jobs green, the gate 33330 passed on python 3.12-3.14 in 365-429 s; on python 3.11 `1 failed, 33329 passed in 393 s`, `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[longdouble-pow]`: the oracle's `Fraction(2 ** 60 + 1, 2 ** 60) ** (-inf, -2)` gave `[1.0]`. first misread as numpy 2.4 comparing a long double as its double (`42234f0` made the oracle decide exactly, and run 36406179185 failed the same way); the cause is CPython 3.11's `Fraction.__pow__`, which answers any non-rational exponent with `float(a) ** b`, so a Fraction base is rounded before the library's `__rpow__` sees it: on 3.11 `Fraction(1, 3) ** O(2)` is `(0.11111111111111109, 0.1111111111111111)`, missing 1/9, through `Dual` too (checked with a local 3.11.15; `+ - * / // % divmod` stay exact; 3.12 returns NotImplemented). nothing in the library can see it (its `__rpow__` gets a float), so the owner set python >= 3.12 (D25); `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11. the exhaustive jobs 19 s to 8 min 54 s; pushed at `c1552b1`, CI run 36415649083 is green (all 7 jobs, 2026-09-28): the gate 33332 passed on each of python 3.12-3.14 in 355-410 s, the exhaustive jobs 23 s (sabotage) to 12 min 12 s (modulo), no 3.11 job
* v1 files stay in place, untouched, until M10, then move to `archive/v1/`. **no v1 file is ever
  deleted by this plan**: the archive is the reference until v2 works. v1 is the differential
  oracle for set ops and for `A % scalar`. the package is `intervals/`, so `import multi_interval`
  (v1) and `from intervals import MultiInterval` (v2) coexist, before and after the move
* since M10 (2026-09-25): v1 is in `archive/v1/`, pytest's `pythonpath` is `[".", "archive/v1"]`,
  and `testpaths` also collects `README.md` as a doctest. v1 is imported in one place,
  `tests/test_kernel.py::to_v1`, used by the set-op differential (`test_matches_v1`) and by the
  `A % scalar` one (`tests/test_modulo.py::test_matches_v1_mod_scalar`)

## 2. milestones

each milestone = one branch or one commit series, green gate at the end, sabotage check for
every new property test (flip one comparison, watch red, restore).

### M1 `errors.py` + `cuts.py` (½ day)
* `Side(IntEnum)`, `Cut` with `-0.0 → 0.0`, integral `Fraction → int` and `nan → ValueError` in `__new__`
  (`typing.NamedTuple` refuses a `__new__` override, so `Cut` subclasses a `_CutBase(NamedTuple)`
  with `__slots__ = ()` and normalizes there), `below(v)`, `above(v)`,
  `mirror(cut)`, `as_start(cut) -> (value, closed)`, `as_end(cut)`, `start_cut(value, closed)`,
  `end_cut(value, closed)`
* the four warning classes and the filter install (`append=True`, so earlier user filters win)
* tests: ordering table from the plan, mirror is an involution, `[a,b]` round-trips through
  `start_cut/as_start`, `Cut(-0.0, x) == Cut(0.0, x)` and prints `0.0`; `type(Cut(Fraction(6, 3), x).value) is int`

### M2 `kernel.py` (2 days) — the risky one
* `normalize(pairs) -> cuts`: sort, sweep, merge iff `next.start <= cur.end`, drop `start >= end`
* `union`, `intersection` (sweep with a depth counter; `n_overlaps` generalizes v1's `merge`),
  `complement` (prepend/append + role shift), `difference` = `A ∩ ~B`, `symmetric_difference`
* `contains_point`, `is_subset`, `hull`, `pieces` (iterate pairs), `size -> Size(rays, length, points)`
* `Builder`: collect, sort once, sweep (compare.py: bisect-insert for incremental, timsort for bulk)
* tests: hypothesis strategy `cut_tuples()` over int/Fraction/float/±inf; laws `~~A == A`,
  De Morgan, `A ∪ ~A == REALS`, `A ∩ ~A == ∅`, `A - B == A ∩ ~B`, idempotence, commutativity,
  associativity; size tiling `size(A) + size(B) == size(A ∪ B)` for disjoint A, B and the plan's
  table (`[1]`, `[1,2)`, `(1,2)`, `(1,inf]` = `Size(1,-1,0)`); differential vs v1 `union` /
  `intersection` / `difference` on random finite inputs (v1 is trusted there)

### M3 `fmt.py` (½ day)
* `format(cuts)`: v1 grammar (`{}`, bare piece, `{ A , B }`, `[x]`, `inf`); Fraction as `p/q`
* `parse(str)`: v1's three regexes at module level, plus `∪`/`|`/`,` as separators
* tests: round-trip under hypothesis; the example strings in v1's parser comments (v1 has no doctests;
  checked 2026-09-23). done 2026-09-23: the parser is strict (leftover text is a ValueError) and
  keeps v1's juxtaposition `[1,2)[3,4)`; floats print by `repr` so they round-trip

### M4 `multi_interval.py` — the class (1 day)
* frozen, `__slots__ = ('_cuts',)`; `MultiInterval(start=None, end=None, *, start_closed=True,
  end_closed=True)`, `from_cuts`, `from_pieces`, `parse` (explicit; strings are not coerced)
* `_coerce(other)`: Real → degenerate, MultiInterval → itself, else `NotImplemented`
* dunders: `| & ^ ~` for set algebra, `union intersection difference symmetric_difference
  complement` as methods (`-` is left for M6's arithmetic subtraction, as in v1),
  `__contains__` (scalar, subset alias), `__getitem__` (slice = restriction),
  `__bool__`, `__eq__/__hash__` structural (`NotImplemented` for non-MultiInterval, no coercion),
  `__iter__` over pieces, `__len__` = piece count,
  `__repr__` (fixes v1), `__str__`
* properties ported from v1: `is_empty is_contiguous is_degenerate is_finite is_integral
  is_positive is_negative is_non_negative is_non_positive finite positive negative inf sup
  inf_closed sup_closed degenerate_points hull pieces size`
* dropped from v1 (immutability): `add clear discard pop remove update *_update merge_adjacent
  abs/invert/mirror (mutating forms) expand(inplace=) copy`; `expand(d)` returns a new value
* tests: mostly delegation; `hash`/`==` consistency under hypothesis; `MI(5) != 5`;
  `sorted()` uses `sort_key`
* done 2026-09-23. choices made while building: `__getitem__` takes slices only (v1 also took a
  number or a MultiInterval; `in` and `&` cover those); `inf`/`sup` on the empty set raise
  `ValueError` like `min([])` (v1: `KeyError`); `hull` keeps the endpoint flags and `closed_hull`
  closes them (v1 had only `closed_hull`); `finite` keeps v1's meaning (pieces touching ±inf are
  dropped whole); `positive`/`negative` include ±inf; `expand` needs a finite distance; `nan in A`
  is False, a non-number raises `TypeError` (as v1)

### M5 `relations.py` (1 day)
* `TruthSet` (four states, `__bool__` raises on `{}` and `{T,F}`, `.certainly`, `.possibly`)
* pointwise `lt le gt ge` on cut tuples from the operands' extreme cuts (`A < B` is `{T}` iff
  every point of A is below every point of B, `{F}` iff no pair satisfies it, else `{T,F}`,
  `{}` if either operand is empty), then verified by sampling
* `equals_pointwise`, `before after adjoins disjoint overlaps contains within`, `certainly_*` /
  `possibly_*`, `allen(a, b)` on contiguous operands (raise otherwise), `sort_key`
* relations return `bool`; only `lt le gt ge` return a `TruthSet`
* tests: oracle = enumerate sampled pairs, compare the attained truth set; `[1,2) < [2,3]` is
  `{T}` and `[1,2] < [2,3]` is `{T,F}`; `before([1,2), [2,3])` is True and `before([1,2], [2,3])`
  is False; `before(A, B) == (A < B).certainly` for non-empty operands; the 13 Allen cases plus
  the two cut-refined ones
* done 2026-09-23. choices made while building: `bool()` of an ambiguous or empty `TruthSet`
  raises `ValueError` (numpy's choice); `adjoins` is symmetric (either end meets the other's
  start); an empty operand is `before`/`after`/`adjoins` nothing; the modal variants are
  `certainly_`/`possibly_` × `before`/`after`/`equal`. the comparison oracle samples two points
  per gap between endpoint values, which is exactly enough to realise `<`, `==` and `>`

### M6 `applicator.py` + `ops.py` (3 days) — needs D1, D2, D3, D7
* `OpDescriptor(fn, monotone=(dir_x, dir_y) | None, split_points=(...), attained=None,
  rounded=(fn_down, fn_up) | None)`
* `apply_binary(desc, A, B)`: split at descriptor points (zero for reciprocal, div **and mul**)
  → for each piece pair, corners on `(lo, lo_closed, hi, hi_closed)` treated closed → closure:
  corner-flag rule only for a finite result endpoint of an op injective in each argument there;
  every infinite result endpoint and every flat spot goes through `desc.attained(v, A, B)` on the
  full operands → union → normalize. `apply_unary` the same
* ±inf is exact: never rounded, does not make an interval float, attainment at ±inf decided
  symbolically
* `div` evaluates `a / b` at the corners directly (not `a * (1/b)`, which rounds twice on floats)
* indeterminate corner policy per D2, warnings per plan; empty propagation + warning
* exact division (D3): int/Fraction operands divide as Fraction; infinite corners are evaluated
  by the applicator, not by python (`Fraction(1) / inf` is `0.0`), so `1/[inf]` is `[0]` exactly
* ops: `add sub neg mul reciprocal div abs pos`, `pow` with int exponents only (fractional and
  negative-base cases stay `NotImplemented`, as v1); `exp log` via `apply_unary` if cheap, else
  defer to `functions.py`
* `tests/oracles.py`: `sample(A, n)` respecting open/closed and drawing a closed ±inf endpoint
  with positive probability, `attained_oracle(op, v, A, B)` for int/Fraction operands with ±inf
  handled symbolically (promoted from `modulo_v3_prototype.attained`)
* tests: soundness fuzz per op; attainment per endpoint on exact operands; isotonicity
  `A ⊆ B ⇒ f(A) ⊆ f(B)` for every op; `f(A ∪ B) == f(A) ∪ f(B)` for add, sub, neg, abs, mul only,
  and only `⊇` for reciprocal/div (counterexample `A=[-1,0)`, `B=[0]` under D1, see `v2-plan.md` testing);
  `1/(1/A) == A` for every A with no degenerate piece at `0`, `inf` or `-inf`; the plan's worked
  examples as a table (`1/[-1,0]`, `1/[-1,1]`, `1/[1,inf]`, `1/[1,inf)`, `[0,1]*(2,3)` = `[0,3)`,
  `1/[0]` = `∅` and warns, `[1]/[3]` = `[1/3]` as Fraction, `type` of `([6]/[3]).inf` is int);
  the D2 table (`[-inf,-1]*[0]` = `[0]`, `[-inf]*[0,1]` = `[-inf]`, `[-inf,-1]*[0,1]` = `[-inf,0]`,
  `[1,inf]/[1,inf]` = `[0,inf]`, `[1,inf]-[1,inf]` = entire, `[inf]-[inf]` = `∅`); the
  infinity-closure table (`[inf] + (1,2)` = `[inf]`, `[1,inf] + [0,1)` = `[1,inf]`,
  `[inf] * (1,2)` = `[inf]`, `(1,inf] + [0]` = `(1,inf]`); interior sharpness on exact operands
  (every sampled point of the result is attained; `[-1,1] * [inf]` = `[-inf] ∪ [inf]`)
* property tests that generate indeterminate or empty cases on purpose carry a
  `filterwarnings('ignore::...')` mark; each warning is pinned by its own `pytest.warns` test
* sabotage: the isotonicity property must go red if `1/[0]` is set back to `[-inf] ∪ [inf]` (the
  hypothesis strategy has to generate degenerate `[0]` inside `[-1,0]` / `[0,1]` often enough);
  soundness must go red if infinite endpoints are given back to the corner-flag rule;
  interior sharpness must go red if mul's zero split is removed
* done 2026-09-24, in one session. the whole semantics reduces to one rule: the result is the set of
  values the *defined* pairs attain (`0*±inf`, `inf-inf`, `±inf/±inf`, `0/0` have none), every
  endpoint closed iff attained, and `x/0` gives ±inf with the sign of x times the side of zero the
  divisor's piece extends to. D2's corner limits and D7's `∅` both follow from it. choices made while
  building:
    * closure is a face rule, decided per split box (same union as deciding against the full operands):
      an extreme is attained at an all-closed corner or along a flat edge whose fixed coordinate is a
      closed end. `OpDescriptor` gained a `pole(args, dirs)` field for `x/0`; `fn` returns None where
      the op has no value. ops are `neg pos absolute reciprocal power add sub mul div` over cut tuples
    * for `+ - * /` the D2 edge-limit block in `evaluate_box` is redundant for values (the far corner
      of the edge already gives it); it only makes an exact `0` win over `0.0` in
      `[0] * [2.5, inf]`. kept as the documented rule for later ops
    * negative powers are one step, `1 / x**|n|`, not `reciprocal(power(A, |n|))`: the same set on
      exact operands, but a float `x**|n|` underflowing to 0 would lose the pole's sign
    * a mixed exact/float pair is computed exactly and rounded once (`Fraction(1, 10**400) * 1e300`
      would otherwise be 0.0); a float piece that rounding squeezes to one point keeps it, closed
      (`[1] + (0, 1e-300)` = `[1]`, not `∅`); poles never reach the rounding hook
    * `A ** Fraction(2)` works (python's `Fraction.__rpow__` makes it `A ** 2`); `A ** Fraction(1, 2)`
      is a TypeError. `__array_ufunc__ = None` so numpy scalars on the left reach the reflected dunders
    * one `EmptySetPropagationWarning` / `IndeterminateResultWarning` per call, attributed to the first
      frame outside the package
    * **plan claim corrected by the tests**: `1/(1/A) == A` also needs A not to be unbounded at both
      ends while holding exactly one of ±inf (`A = (-inf, inf]`: `1/A` = `[-inf, inf]`, which maps to
      itself). `tests/test_ops_properties.py::test_reciprocal_involution` pins the exact condition;
      the "testing" section of `v2-plan.md` was corrected 2026-09-24 after owner review
  evidence (2026-09-24): the gate was 940 passed in 72 s. every sabotage was
  run in a separate worktree with hypothesis seeds default, 1 and 2, and all of them went red. the
  plan's three:
    * `1/[0]` set to `[-inf] ∪ [inf]` turned `test_isotone[div, reciprocal, pow]` red on 3/3 seeds,
      even with the pinned `@example`s disabled
    * infinite endpoints given back to the corner-flag rule were caught by the infinity-closure
      tables on every run, and by soundness or attainment on every seed. soundness *alone* went red
      on 2 of 3 seeds
    * removing mul's zero split turned `test_interior_sharpness[mul]` red on 3/3 seeds
  the others:
    * the pole sign taken from a sign bit
    * attainment checked at corners only
    * no `IndeterminateResultWarning`
    * float instead of Fraction for exact division
    * `finite/inf` returning `0.0`
    * the lo == hi collapse rule removed
    * the negative-power open-pole rule broken, which is now red on 5/5 seeds
    * an oracle bug in `attained('mul', 0, ...)`

  an exhaustive differential against `tests/oracles.py` covered every 1- and 2-piece operand over
  the exact grid `{-inf, -2, -1, -1/2, 0, 1/2, 1, 2, inf}` with all open/closed combinations: about
  230k binary and 32k unary checks, plus 179k on the same grid as floats, all with 0 mismatches.
  the harness was recovered from the Recycle Bin at M11 (2026-09-25) and is now
  `tests/exhaustive_ops.py` (see M11)

  recovered from the M6 review's scratch output at M10 (2026-09-25). the differential and the
  extreme-float fuzz scripts were recovered at M11 and are tracked now; the rest were not kept:
    * **outward rounding on extreme floats**: a fuzz over subnormals (5e-324, 1e-320,
      2.2250738585072014e-308), 1e308, `sys.float_info.max`, 1e±300, 1e154 and random `ldexp`
      values across the exponent range, mixed with ±inf and int/Fraction, checking soundness
      with the open/closed flags **as given** (the tracked
      `test_sound_float_outward_rounding` closes the result first, and `tests/strategies.py` draws
      floats from `[-20, 20]` only). 2026-09-24 at `ca667e2`: 22000 boxes, 0 unsound; a hook
      that did not round made add, sub, mul, div and reciprocal unsound. re-run 2026-09-25 at
      `c6cfe14`, 2 × 4000 boxes: 0 unsound, and the rounding hook never received an infinite or
      a float-free argument. in the gate since M11 as `tests/test_extreme_floats.py`
    * two hand-derived tables (84 distinct cases, 58 not in `tests/test_ops_examples.py`) matched
      the implementation and `oracles.attained` at every probe point, so the oracle and the
      implementation do not share a blind spot there. four disagreements were the reviewers' own
      derivation mistakes. two of those cases are now rows in `test_ops_examples.py`
      (`(0, inf] / (0, inf]`, `[-inf, 0] - [-inf]`)
    * findings rejected, so they are not raised again: `test_isotone[div]` is mostly trivial
      examples (by design; the union laws give about 27 non-trivial div isotonicity cases per
      run); `test_applicator` and `test_ops_examples` share 7 of 25 closure rows
      (`test_ops_examples` is the independent table derived from the plan); moving
      `applicator.warn` into `errors.py` (style)

### M7a `modulo.py`, Q1 port (2 days)
* port `modulo_v3_prototype.py` (P1 scalar mod, P2 scalar-mod-interval, far-edge union,
  attainment closure) onto `(lo, lo_closed, hi, hi_closed)` pieces; multi-piece operands test
  attainment against the full operands
* `floordiv = floor ∘ div` with `floor` as an enumerating unary op under a size cap, hull +
  warning above it; `divmod` from the two; `rmod` by symmetry of the entry point
* pass the generating `(edge, k)` into the attainment test (design notes §3c cost) or cap `k`
* tests: prototype's 112-case corner suite verbatim (its literals are exact binary floats); its
  4000-case fuzz with the same seed and ranges but operands as `Fraction(str(x))`, since
  attainment is checked on exact types only; degenerate operands
  table from design notes §3c; differential vs v1 `A % scalar` (trusted); `[1,2) // 1 == [1]`
* other sign combinations raise `NotImplementedError` until M7b; not a releasable state
* done 2026-09-24, together with M7b in one session, so no Q1-only state was ever committed

### M7b modulo, every sign combination (one full session; blocks release)
* derive the Q2 primitive pair (dividend < 0, divisor > 0) in closed form (design notes §2 Thm B,
  §4 next steps), with a proof note next to the existing ones in `references/modulo-derivations/`
* Q3/Q4 from Q1/Q2 by the antipodal identity; operands crossing zero split into sign-pure pieces via
  the descriptor's split points; a divisor touching zero drops 0 with `DomainClippedWarning`
* infinite operands per D8 (confirm with owner first)
* tests: extend the prototype's corner suite and fuzz to all four quadrants and zero-crossing
  operands; attainment oracle on exact operands; python's `%` sign convention (result takes the
  divisor's sign) as the scalar reference
* done 2026-09-24 (M7a and M7b together): `intervals/modulo.py` (`mod floor floordiv divmod_`), the class's
  `% // divmod` with their reflected forms and `floor()`, `HullWarning`, and `tests/test_modulo.py`.
  choices made while building:
    * **D8 is implemented as its recommended default, confirmed by the owner the same day**: `±inf mod y`
      is dropped and `x mod ±inf` follows python. it is isolated in `modulo._box` (the `[inf]` branch) and
      `modulo._attained` (the `has_inf` lines), so reversing it is local
    * one path for every quadrant: a negative divisor goes through the antipodal identity, and the
      Q2 left edge is `_scalar_mod_interval_negative` (closed form in its docstring; the proof is in
      `references/modulo-derivations/claude-fable/proof-all-quadrants.md`)
    * attainment is exact and O(1) (`_attained`, `_holds_multiple`): only the two extreme quotients
      need an exact check, any quotient strictly between them meets the interior. it covers k >= 1 and
      k <= -1 in one test, so it is quadrant-agnostic. this replaces the prototype's O(x1/y0) k loop
      (`[10**12, 10**12 + 1] % [1, 3/2]` is timed in the tests)
    * ends are decided per located piece, before the union. the prototype decided them after merging,
      which could hide an unattained value where two pieces touch; its own fuzz checked soundness and
      closure but not the interior, so it could not have seen that
    * attainment is decided per box, not against the full operands (design notes §3b): the union of
      per-box attained sets is the attained set of the union, so the result is the same
    * finite values are computed as Fractions and a float operand rounds the result once (the M6
      mixed-pair rule). for a mixed Fraction/float pair that differs from python, which rounds the
      Fraction first
    * `floor` enumerates up to `FLOOR_ENUMERATION_CAP = 1000` integers per call, then hulls with a
      `HullWarning`, which is new and shown by default (precision was lost). `floordiv` is `floor ∘ div`
      over the finite divisors, so it inherits div's poles (`[1] // [0, 1]` holds inf, where `mod` drops
      the 0). `divmod` is a pair of sets, one warning for an empty operand
    * **follow-up the same day (owner decision)**: at an infinite divisor `//` takes the limit, as python:
      `[-5] // [inf]` = `[-1]`, not `floor(div)`'s `[0]`. the oddity and the reasons are in `v2-plan.md`
      (arithmetic, floordiv) and the `modulo` docstring; the one rule behind it and D8 is "a pair's value
      at an infinite operand is its limit". the python comparison table added for it found a float bug
      that predated it: `floor(div)` floored the *rounded* quotient, so `[1] // [0.001]` gave `[1000.0]`
      where the true value is 999; the quotient is now exact and only the integers are made float.
      reverting that turns `test_scalar_floordiv_matches_python` and `test_floordiv_sound_float` red
    * **v1 is not sharp for `A % scalar`**: an open end at a multiple of m gives it a 0 nothing attains
      (`[0.25, 0.5) % 0.5` is `{ [0] , [0.25, 0.5) }`). on the quarter grid 0..10 with m in {1/4, 1/2,
      3/4, 5/4, 7/4}, 302 of 16605 boxes differed, all of them that 0, and ours matched the oracle
      every time (measured 2026-09-24). the differential test therefore checks ours ⊆ v1 and v1 − ours
      ⊆ {0}, and `test_v1_phantom_zero` pins the defect
    * nine expected values first written by hand for the example tables were wrong, all in Q2/Q4, an
      unbounded divisor or a clipped zero (`[-6, -3] mod [4, 5]` is `[0, 5)`, `[2, 3] mod [1, inf)`
      has a gap `[3/2, 2)`). each was re-derived by hand and checked against the oracle before it was
      changed; the table comments carry the derivations
  evidence (2026-09-24): the gate was 1603 passed in 87 s.
    * the derivation was checked and extended to every sign by a Claude (Fable) subagent:
      `references/modulo-derivations/claude-fable/proof-all-quadrants.md`, with its executable form
      `modulo_allquadrants_prototype.py`. that audit found the prototype's merge-before-closure hole
      independently; its harness reported 0 failures of every kind on 89,700 grid pairs and 6000
      fuzz cases, and its `--quick` mode was rerun here with 0 failures
    * `python -m tests.exhaustive_modulo` (every single-piece pair over `{-inf, -3, -2, -3/2, -1,
      -1/2, 0, 1/2, 1, 3/2, 2, 3, inf}`, all flags): 105,625 boxes, 0 mismatches against the
      brute-force oracle in `tests/oracles.py`, 1002 s
    * the Fable prototype and `intervals/modulo.py` were written independently; they agree on all
      105,625 of those pairs
    * sabotage, each run against `tests/test_modulo.py`: skipping the exact check at the two extreme
      quotients, dropping the Q2 left edge, keeping unattained degenerate pieces, reading every
      operand end as closed in the attainment test, not splitting the dividend at 0, and treating a
      negative divisor as positive all went red. with only the property tests selected, each
      property (soundness, closure, interior sharpness, isotonicity, union, antipodal, periodicity)
      went red under at least one of them. breaking floor's open-integer-end rule turned
      `test_floor_sound_and_sharp` red, and rounding the low end of a float result inward turned
      `test_sound_float` red

### M8 `time_interval.py` (1½ days) — next (owner 2026-10-04; was deferred, D4)
* starts from `archive/v1/time_interval.py`, ported onto the v2 class with whatever tweaks that
  needs; the archived copy stays until the port works, then `archive/v1/` goes (`HANDOFF.md` H4)
* before building, the owner's three choices (`HANDOFF.md` session log 2026-10-04): timezones (naive as
  wall clock? mixing aware and naive raises?), what an infinite end reads as, and whether the
  end-of-day snap (23:59:59.999999) stays or becomes a half-open next midnight
* `DateTimeInterval`, `TimeDeltaInterval` as thin wrappers over a numeric `MultiInterval` of
  exact seconds (D4a), with the v1 cross-type arithmetic table; keep the end-of-day snapping for
  `date` inputs (document it); fill v1's gaps (`__repr__`, slicing on both, item methods dropped
  with immutability)
* tests: the arithmetic table; pandas round-trips; `[-inf, t]` style open-ended ranges

### M9 `tests/itf1788/` (1½ days)
* vendor a subset of `.itl` files (licence check first), a parser for the used subset (`add`,
  `sub`, `mul`, `div`, `recip`, `neg`, `abs`, set ops, `sqr`/`pow` if implemented), input rule
  (1788 unbounded → open at inf), output rule (closed hull of **both** ours and the expected
  value), divergence table as a dict of
  `(op, inputs) -> reason`
* exit: every vector either matches through the adapter or is in the divergence table with a
  reason from the plan's list. `1/[0]` is not a divergence row (both give empty, D7)
* done 2026-09-24. `tests/itf1788/`: `libieeep1788_tests_elem.itl` and `libieeep1788_tests_set.itl`
  vendored unmodified from nehmeier/ITF1788 at `e0e0d7e` (Apache 2.0; `LICENSE`, `NOTICE` and a
  provenance `README.md` beside them), `itl.py` (parser; only used ops are parsed, anything
  unrecognised in one raises) and `test_itf1788.py` (adapter, one test per vector). the used subset
  is `pos neg abs add sub mul div recip sqr pown floor intersection convexHull`, 847 vectors
  (decorated ones included, decorations dropped). all 847 match; the divergence table is empty
  (measured 2026-09-24). choices made while building:
    * a third rule, **precision**: literals are read as their nearest double (as the C++ tests the
      files came from; `pown [13.1, 13.1] 2` expects a one-ulp result that an outward reading of 13.1
      cannot give), held as exact Fractions, and our exact result is rounded outward to doubles.
      so each vector is an exact soundness-and-sharpness check, not a tolerance
    * the library's warnings are ignored inside a vector; they are pinned elsewhere
    * sabotage, 2026-09-24: rounding to nearest instead of outward turned 97 vectors red, a flipped
      reciprocal pole 10, a flipped div pole 99, abs without its zero split 7, a wrong `sub`
      monotonicity 14; a parser dropping `_com` vectors turned the line-count check red, and a
      divergence row on a matching vector failed as stale. **two changes stayed green**: closing
      the infinite input bounds (the input rule is unobservable through a closed hull under D2, so
      it is pinned by `test_input_rule` alone) and mul without its zero split (interior sharpness,
      which a hull cannot see; `tests/test_ops_properties.py::test_interior_sharpness` goes red on that change)
    * not covered: the bool, num, overlap and reverse-op files (not in the plan's subset), and
      1788 ops not implemented (`sqrt`, `exp`, trig, `fma`, `ceil`, `trunc`, `sign`, `min`/`max`, ...)

### M10 archive v1 (½ day)
* `git mv` `interval.py`, `multi_interval.py`, `time_interval.py` and `compare.py` into
  `archive/v1/`, unchanged: nothing is deleted. v1 `time_interval.py` imports v1
  `multi_interval.py`, so they move together and stay runnable side by side
* add `archive/v1` to pytest's `pythonpath` so the differential tests against v1 keep running
* README rewritten around the package, with a short note that `archive/v1/` is the old
  implementation kept as reference; keep `references/`
* `v2-plan.md` "current design" updated for every decision changed during the build (D1–D7
  outcomes), dated
* the archive goes only by owner decision, once v2 works (release, and M8 if the time layer is
  wanted)
* done 2026-09-25. the four v1 modules and the old README moved unchanged to `archive/v1/` (v1
  still imports and runs from there); `pythonpath` gained `archive/v1`, and without it the v1
  differential fails with `ModuleNotFoundError`. the new README is collected as a doctest by the
  gate, and a wrong example in it turns the gate red. `v2-plan.md` "current design" was checked
  line by line against the build and corrected in 20 places (the face rule for attainment, the
  union laws, symmetric `adjoins`, modulo built for every quadrant, when exceptions are raised,
  `HullWarning`, outward rounding not yet built, layout). the baseline gate was red before any
  M10 change: `test_is_subset`'s oracle probed no point between two adjacent doubles, fixed in
  `tests/strategies.py::midpoint` and pinned by an `@example`

### M11 backlog: everything not yet built (listed 2026-09-25)
not a milestone with an exit criterion: the open work left after M10, from a sweep of both plans,
the old README (`archive/v1/README.md`), v1's public surface and the code (no TODO, FIXME, skip or
xfail in `intervals/` or `tests/` as of 2026-09-25). tags: **(a)** needs an owner decision first,
**(b)** ready to build, **(c)** housekeeping. items become milestones when picked up
* **release** (a, then c): merge `v2` into `master`, `version = "2.0.0"` in `pyproject.toml`, tag.
  D5's blocker (M7b) is met; D17: no hurry. status: `HANDOFF.md` H1
* **M8, the time layer** (a: whether and when; 1½ days): see M8; open, `HANDOFF.md` M8 and Q5
* **functions**, **rounding functions**, **the other 1788 ops** and **outward float rounding**, the
  four (b) items: built at M12 (below), done 2026-09-25
* **reverse ops** (a): `mulRevToPair` and friends, for the reverse-op itf1788 files and for a solver.
  owner 2026-09-25: build, as M13e
* **power beyond int exponents** (a: scope; owner 2026-09-25: build 1788's `pow`, as M13d; built
  2026-09-26, see M13d): `A ** 0.5`, `A ** B`, `2 ** A` were TypeError until then
  (`intervals/multi_interval.py::MultiInterval.__pow__`); v1 took an interval exponent on a
  positive base. 3-argument `pow(A, n, m)` (v1: integers only; old README "allow interval modulo
  for `__pow__()`"). settled by D11: integral exponents stay pown, others are 1788 pow, and
  3-argument `pow` is dropped
* **the 1788 ops still missing** (a: whether to add them; settled, see below): `less`, `strictLess`, `interior` (weak
  and strict interval orders, not v2's pointwise comparisons) and `mid`, `rad`, `wid`, `mag`, `mig`
  (for a multi-interval, of the hull or per piece?). their vectors are counted and skipped in
  `tests/itf1788/test_itf1788.py::SKIPPED`; `isNaI` has no counterpart (and gets none: no NaI, owner 2026-09-26). owner 2026-09-25: add
  every one, as M13b, M13c and M13g (D9, D10, D16)
* **solver stack** (a; `v2-plan.md` "later (not in v2.0)"): its decorated type was brought forward
  to M13g by D16; autodiff and interval newton built as M15 (2026-09-27); the rest is open, `HANDOFF.md`
  H3 (numpy and gmpy2/mpfr recorded, not now: owner 2026-09-26)
* **v1 surface with no v2 row in section 4** (a: port or record as gone): closed 2026-10-04 (the owner:
  shifts, `merge`'s k-overlap mode and parsing, `random_multi_interval` gone; `apply()` stays internal;
  `v2-plan.md` decision log, 2026-10-04)
* **smaller** (c): the old README's leftovers, `HANDOFF.md` H5
* **archive deletion** (a): `HANDOFF.md` H4
* done 2026-09-25, from the backlog: the M6 differential and the M6 review's extreme-float fuzz,
  restored from the Recycle Bin by the owner, are tracked again. evidence, at `6d4851f` plus the
  two files:
    * `tests/test_extreme_floats.py` (in the gate, about 8 s): 1000 boxes for each of two seeds, open/closed
      flags as given, with an audit that the hook never sees an infinity and always sees a float.
      sabotage, each red: a hook that does not round (the file's own
      `test_fuzz_catches_a_hook_that_does_not_round`, unsound for add, sub, mul, div and
      reciprocal); the exact hook rounding to nearest (6 ops unsound); `applicator._rounds` letting
      infinite or all-exact corners through, and the exact hook rounding inward (both red by an
      exception, not by the assertion)
    * `python -m tests.exhaustive_ops`: 0 failures in 1105 s. `--float`: 0 failures in 665 s. both
      ran the same check counts as at M6 (exact: 3213 each for neg, abs and reciprocal, 22491 pow,
      58165 add, 57871 sub, 58071 mul, 58241 div). `--sabotage`: 1072, 801 and 209 failures, the
      same as M6's `sabotage.py`
    * the gate: 2817 passed

### M12 the unblocked backlog: functions, step functions, min/max/fma, outward rounding (done 2026-09-25)

the four (b) items of M11, built in one session by owner request ("build all the things that are
unblocked, and add the relevant tests and reference vectors"). commits `53400d4` (the code and the
vectors), `1ed5e70` (the unit tests), `5d584c1` (atan2); the design is in `v2-plan.md` "current
design" (arithmetic, "elementary and step functions (M12)", ieee 1788) and its decision log entry
"2026-09-25 revision: M12"
* built:
    * `intervals/elementary.py`: sqrt, exp, exp2, exp10, log (any base), log2, log10, sin, cos, tan,
      asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh and the atan2 angle at one exact point,
      correctly rounded down, to nearest and up, in pure python (no libm)
    * `intervals/functions.py`: those functions and atan2 over sets; methods of the class
    * `intervals/steps.py`: floor, ceil, trunc, round (and `round(A, ndigits)`), round_ties_away,
      sign; `math.floor/ceil/trunc` and `round()` return sets; `modulo.floor` delegates here
    * `intervals/ops.py`: `minimum`, `maximum`, `fma`, and the `OUTWARD` descriptors;
      `intervals/rounding.py`; `outward=` on every rounding op, `mod` and `floordiv` included
    * `OutwardMultiInterval`, exported from `intervals`
    * itf1788: `libieeep1788_tests_bool.itl`, `_num.itl`, `_overlap.itl`, `_rec_bool.itl` and
      `atan2.itl` vendored unmodified (git blob hashes equal upstream's), the parser reads numbers,
      booleans and overlap states, and every interval-valued vector runs a second time through
      `OutwardMultiInterval` with float operands, compared without adapter rounding
* found while building, fixed: `MultiInterval(1e308) // MultiInterval(1e-308)` raised
  `OverflowError` (a quotient past the float range; `modulo.py::_to_float` called `float()` on it).
  it rounds to `inf` now, `(max float, inf)` outward
* evidence, measured 2026-09-25 at `5d584c1`:
    * the gate: 7979 passed in 206 s (2817 before M12)
    * `python -m tests.exhaustive_modulo` after `modulo.py` began rounding through
      `rounding.round_piece` and delegating floor to `steps.py`: 105625 boxes, 0 with a mismatch
      (1289 s, sharing the machine with a gate run)
    * itf1788: 2932 vectors of 54 ops from 7 files (847 before), 2438 interval-valued run twice. 18
      divergence rows, all anticipated by the plan's categories: 11 degenerate infinities (log,
      log2, log10 of an operand meeting the domain only at 0; atanh of one meeting it only at ±1)
      and 7 cut-based relations (overlap: 1788's meets is our overlaps). every function vector
      matches 1788's tightest enclosure in both passes, atan2's 375 included
    * `tests/test_elementary.py`: all 19 functions correctly rounded in the three directions against
      the decimal oracle on 60 random points each, plus 40 extreme points (`sin(1e22)`, exp near the
      float range, `log(10**-400)`, ...); the enclosures hold the value at 64, 100 and 180 bits.
      libm is not an oracle: this laptop's UCRT `acosh` was 2 ulp off near 1, where the decimal
      oracle agreed with `elementary`
    * sabotage, each red (the scripts were throwaway, the results are here): the vectors under
      elementary ignoring the direction, ops never rounding outward, functions rounding floats to
      nearest when outward, round ties away instead of to even, sin/cos ignoring interior extrema;
      the unit tests under cosh's direction flipped, a rounded end kept closed, the wrong end of sin
      taken as the lower, an extremum at a piece's end counted as inside, tan ignoring a pole,
      silent domain clipping, a wrong sin/cos quadrant, ln 2 a hair off, the error bounds dropped,
      round's ties going down, ceil ignoring an open start, sign of an open end at 0, min never
      attaining a flat end, min falling back to the face rule, fma rounding twice, the outward hook
      rounding to nearest (`tests/test_extreme_floats.py`), the subclass losing its reflected add,
      outward keeping flags at moved ends, and for atan2: the quadrant II corners swapped, no
      attainment along an infinite edge, the cut read as pi from below, the (+, -) angle missing its
      pi. two survived at first and were killed by new tests: an extremum at a piece's end (only cos
      at 0; pinned by examples) and dropped error bounds (pinned through the raw constant series).
      the taylor loops' error bounds still survive their removal, because the interval rounding
      around them is wider than what they bound; they are argued, not pinned
* choices made while building (also in `v2-plan.md`'s decision log): functions have their own
  evaluator, not the applicator (an irrational value has no exact `Value`, and sin/cos/tan split at
  irrational points); no libm; an irrational value of an exact operand is its tightest float
  enclosure, open at both ends; outward, a moved end is open; a domain's end is a point of it where
  the one-sided limit exists (`log(0)` = -inf); atan2 on the negative x axis is pi; `minimum` and
  `maximum` for the method names; `exp2`/`exp10` of an int past 100000 are rounded, not built
* not built, and why: `pow` with a real exponent (a: scope, M11); `less`, `strictLess`, `interior`,
  `mid`, `rad`, `wid`, `mag`, `mig` (a: see M11); reverse ops (a). all now M13

### M13 full itf1788: every vector vendored, every op built (done 2026-09-27: M13a to M13h, M13e and M13g merged; their two new divergence categories approved by the owner, D18)

owner request 2026-09-25: "implement all these ops and get all these tests vendored and passing".
this settles M11's "whether to add them" for every 1788 op, and D9–D17 (section 0) settle how,
D16's signals included (owner, 2026-09-26). sub-tasks M13a to M13h: **M13a goes
first**, the rest are independent of each other. the release is not waiting on any of it (D17)

**the source.** the vendored files today are 7 of the 12 at nehmeier/ITF1788 `e0e0d7e` (that
repo's HEAD). the maintained fork, **oheim/ITF1788 at `b6ee1e24d209c289f99a68ddc357839935799eae`**
(2018-09-22), has 19 `.itl` files under `itl/`: nehmeier's 12 renamed (`libieeep1788_tests_elem.itl`
is `libieeep1788_elem.itl`, and so on), plus `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` (vectors
converted from those libraries' own suites), `ieee1788-constructors.itl`, `ieee1788-exceptions.itl`,
`libieeep1788_class.itl` and `libieeep1788_reduction.itl`. raw files are at
`https://raw.githubusercontent.com/oheim/ITF1788/b6ee1e24d209c289f99a68ddc357839935799eae/itl/<name>.itl`,
and `LICENSE`, `NOTICE`, `COPYING.LESSER` at the repo root. the fork's versions of the 7 vendored
files are a superset in bare intervals: they correct decorations (`acos [entire]_def` becomes
`_dac`), decorate bare `[empty]`s, and add `[nai]`, `NaN` and empty cases. statement counts, fork
minus nehmeier (distinct texts): elem 3817 vs 3804, bool 392 vs 354, num 183 vs 122, rec_bool 137 vs
114, overlap 77 vs 76, set 20 vs 19, atan2 38 vs 38; 9490 distinct statements over all 19 files

**measured 2026-09-25**, the fork's 19 files downloaded to a temp dir (not kept) and run through
this repo's unchanged adapter (`tests/itf1788/test_itf1788.py::run`, `::run_outward`), ops as in
`OPS`:
* **already passing, only needing vendoring**: `fi_lib` 687/687 (outward 687/687), `mpfi` 980/980
  (900/900), `c-xsc` 126/126 (85/85), `atan2` and `libieeep1788_set` all. 1793 plain runs in all
* failing: 40 `[nai]` operands (bool, elem; M13g), 4 `isMember NaN ...` (the parser reads `NaN` as
  a word; the library itself answers `float('nan') in A` false, as 1788 does), and
  `atanh [1.0,1.0]_def = [empty]_trv`, the known atanh row under a different key, because
  `DIVERGENCES` is keyed on the text with its decorations
* **4764 statements of 57 ops not implemented**, all assigned to a sub-task below

**M13a vendoring and the adapter (done 2026-09-26).** no new op; lands the 1793 passing vectors
* vendor per D15: all 19 files, unmodified, from the pinned commit into `tests/itf1788/`, replacing
  nehmeier's 7, with the fork's `LICENSE`, `NOTICE` and `COPYING.LESSER`. verify each file's git blob
  hash against the fork's tree at the pin (`git hash-object` against the GitHub trees API), as M9 did.
  rewrite `tests/itf1788/README.md` for the new source, the licences per file and the hash check
* parser (`tests/itf1788/itl.py`): `NaN` as a number literal; quoted strings (`b-textToInterval
  "[1, 2]"`); a trailing `signal <Name>` clause (kept on the vector, checked from M13g on);
  two-value results (`mulRevToPair ... = [empty] [empty]`, `midRad ... = 0.0 infinity`)
* adapter (`tests/itf1788/test_itf1788.py`): `FILES` lists the 19 new names; `DIVERGENCES` keyed
  on the statement text with decorations stripped, so the atanh row above and the fork's
  re-decorated copies of today's 18 rows keep their keys; a `[nai]` operand, until M13g, is a
  divergence row under "decoration expectations" (a residual category the plan already has)
* exit: every vector of an op in `OPS` matches or is a row, 0 unknown failures, the 1793 in the
  gate, `test_parser_drops_nothing` and `test_every_file_is_used` covering the new files, sabotage
  per section 2 (a parser that drops quoted-string statements, a divergence key that keeps its
  decorations)
* done 2026-09-26, with M14's fuzz job and oracle (below). built:
    * vendored: all 19 `.itl` files and the fork's `LICENSE`, `NOTICE`, `COPYING.LESSER` from
      oheim/ITF1788 at `b6ee1e2`, byte-exact, replacing nehmeier's 7 (`LICENSE` and `NOTICE`
      replaced with the fork's). `git hash-object` equals the trees-API sha for all 22 files; the
      upstream files are pure LF, and a new `tests/itf1788/.gitattributes` marks them `-text` (in a
      throwaway repo with `core.autocrlf=true` the blobs came out identical even without it, so
      the attribute only keeps a Windows checkout from turning them CRLF on disk).
      `tests/itf1788/README.md` has the source, each file's licence (from its header: Apache 2.0 for
      the 11 `libieeep1788_*`, LGPL-2.1-or-later for `mpfi`, `fi_lib`, `c-xsc`, all-permissive for
      five, correcting D15's two) and a hash re-check snippet (run: 22 ok)
    * parser (`tests/itf1788/itl.py`), rewritten: a strict tokenizer (a character no token covers
      raises, and so does text outside a testcase); a testcase ends by brace counting, not at the
      first `}`; `NaN` is a number, quoted strings a new `itl.py::Text`, `{a, b}` lists tuples; a
      trailing `signal <Name>` is kept in `Vector.signal`, not checked until M13g; a two-value
      result is a plain tuple; `parse_file(path)` with no op list parses every statement
    * **a bug the old parser had**: its testcase regex ended a block at the first `}` of a `{...}`
      list, which in `libieeep1788_reduction.itl` silently dropped 11 statements, not even counted as
      skipped. that is why the 4764 skipped statements measured 2026-09-25 are 4775 (and M13h's
      reductions have more statements than "1 each")
    * adapter (`tests/itf1788/test_itf1788.py`): `Vector.text` keeps the raw statement and the
      divergence key is `test_itf1788.py::key`, `strip_decorations(v.text)`; the 18 old rows collapse
      to 15 bare keys. `[nai]` rows are generated in code under "decoration expectations": an operand
      the core has no counterpart for raises `test_itf1788.py::NoCounterpart`, which counts as not
      matching, so each row fails as stale once M13g makes its vector match. `same()` treats NaN as
      equal to NaN, so a row expecting `NaN` can go stale too; the 4 `isMember NaN` vectors match
    * tests: `test_parser_drops_nothing` checks per file that parsed vectors plus skipped statements
      equal an independent count of statement lines; the new `test_parser_reads_every_statement`
      parses every op; `test_every_file_is_used` asserts the folder's `*.itl` files are exactly
      `FILES`, that there are 19, and that each has statements
* evidence, measured 2026-09-26 (`tests/itf1788`: 8939 passed in 68.1 s; the census by importing
  the test module):
    * 19 files, 9542 statements (9484 distinct texts; an independent line regex also counts 9542)
    * 4767 vectors of 54 ops, every op in `OPS` with vectors; 4120 interval-valued and run twice,
      8887 vector test items. by file: elem 2390, set 20, bool 252, num 58, overlap 77, rec_bool
      139, atan2 38, c-xsc 126 (85 outward), fi_lib 687 (687), mpfi 980 (900); c-xsc + fi_lib +
      mpfi = 1793
    * 49 divergence keys, 0 unknown failures: degenerate infinities 10 keys (11 vectors, 11
      outward items), cut-based relations 5 (7 vectors), decoration expectations, the generated
      `[nai]` rows, 34 (34 vectors, 6 outward items; the 40 above)
    * skipped: 4775 statements of 57 ops; the largest pow 1431, powRev1 429, powRev2 375,
      mulRevToPair 347, pownRev 285, mulRev 182, cancelMinus 126, cancelPlus 116, csc 109, sec 109,
      b-/d-textToInterval 91 each. `SKIPPED` is not asserted empty yet: that is M13's exit
    * sabotage, each red, then restored and green: a parser dropping statements containing a quote
      turned 6 red (`test_parser_drops_nothing` and `test_parser_reads_every_statement` for
      `libieeep1788_class`, `ieee1788-constructors`, `ieee1788-exceptions`); a key keeping its
      decorations 4 (`test_vector` at `libieeep1788_elem.itl:4116`, atanh `_def`, and
      `libieeep1788_overlap.itl:98`, `:119`; `test_vector_outward` at `elem:4116`); the old brace
      behaviour fails collection loudly (`ValueError: text outside a testcase`, the reduction file)

**M13b numeric ops (done 2026-09-26)** (D9). `mid` 36, `rad` 19, `wid` 27, `mag` 27, `mig` 33, `midRad` 25
statements
* methods `mid()`, `rad()`, `wid()`, `mag()`, `mig()`, `mid_rad()`. `mid`, `rad`, `wid` of the
  hull; `mag`, `mig` of the set (`sup` and `inf` of the absolute values). exact operands give exact
  values; float results round as 1788 specifies (`mid` to nearest, `rad` and `wid` up) and the
  adapter's number rule compares them
* unbounded: `mid` of entire is 0, of a half-bounded set ±max float; `rad` and `wid` are inf.
  empty raises `ValueError`, like `.inf`; the adapter maps that to `NaN`
* done 2026-09-26. built:
    * `intervals/numeric.py::mid`, `::rad`, `::wid`, `::mag`, `::mig`, `::mid_rad` over cut
      tuples, bound as the methods `MultiInterval.mid()`, `.rad()`, `.wid()`, `.mag()`, `.mig()`,
      `.mid_rad()` (a pair). `OutwardMultiInterval` inherits them unchanged: a number is not an
      interval, and its direction is 1788's for the function, so both classes give the same numbers
      (pinned by `tests/test_numeric.py::test_outward_class_gives_the_same_numbers` and inside the
      float property)
    * **choices the plan left open** (conservative, flagged in the decision log):
        * "a float result" means an operand with a finite float end anywhere
          (`rounding.has_finite_float`, the package's definition of a float interval); then every
          number is a float, the exact value rounded once. an operand without one gives int or
          Fraction (integral Fractions as int), except `mid` of a half-bounded hull, ±max float as D9
          says. `[inf]` and `[-inf] | [inf]` count as exact
        * **`rad` is 1788's**: the smallest double `r` with `[mid - r, mid + r]` holding the hull,
          measured from the *rounded* midpoint, not half the width rounded up. the vectors require it:
          `rad [0X1P+0,0X1.0000000000003P+0] = 0X1P-51`, where half the width, `3 * 2**-53`, is a
          double but the midpoint rounds (a tie, to even) one ulp up. `mid_rad()` is `(mid(), rad())`
        * `mag` rounds up and `mig` down; on double ends both are exact, so the direction shows
          only for a mixed operand (`[1/3, 0.5]`: `mig` 0.3333333333333333)
        * a single point, `[inf]` too, is its own midpoint with radius and width 0; `mid` of
          `(-inf, inf)` is 0 (0.0 for a float operand); `mag` and `mig` of `[inf]` are inf
        * an exact end past max float: a float operand's rounded midpoint is kept inside the hull
          (`MI(1.0, 3 * 2**1024).mid()` is max float, not inf), and an exact half-bounded hull that
          starts past max float has that start as its midpoint (`MI(2**1030, inf).mid()` = `2**1030`)
        * the empty set raises `ValueError('the empty set has no midpoint')` (radius, width,
          magnitude, mignitude; `mid_rad` "midpoint or radius"), no warning
    * adapter (`tests/itf1788/test_itf1788.py`): the six ops in `OPS`, and a **numeric rule**
      (`::NUMERIC`, `::_numeric`, `::_mid_rad_1788`, `::round_nearest`): the first pass has exact
      operands, so our exact number is rounded as 1788 rounds it (`mid` to nearest, `wid`/`mag` up,
      `mig` down, `rad` around the rounded midpoint); a new second pass, `::test_vector_float` via
      `::run_float`, gives the 167 vectors float operands in a `MultiInterval` and an
      `OutwardMultiInterval` and compares the library's own numbers with no rounding by the adapter
      (so the float path is checked the way `test_vector_outward` checks outward rounding). a
      `ValueError` is `NaN` in both passes; `same()` compares a `midRad` pair item by item. no
      divergence row beyond the 6 generated `[nai]` ones
    * tests (`tests/test_numeric.py`, 26 items, 13-17 s, 2026-09-26): 5 `@given`: exact operands
      against the definitions, `mag`/`mig` through `abs(A)` (an independent path) and soundness at
      sampled points (`mig <= abs(x) <= mag`, `x` in `[mid - rad, mid + rad]`,
      `abs(x - y) <= wid`); float operands over the whole double range (subnormals, ±max, exact ends
      mixed in), each number checked from the definition of rounding on its neighbouring doubles
      (`tests/test_reductions.py::is_rounded`), `rad` as the smallest covering double, both classes
      equal, soundness; isotonicity on exact and on float operands; hull against set. `@example`s:
      the rad tie above, `midRad [0X1.FFFFFFFFFFFFFP+1022, max]`, the mpfi tie
      `mid [-0x1fffffffffffffp-53, 2.0] = 0.5`, subnormal ties, `[-max, max]`, D9's `mig`. under
      `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=20` the file ran green in 346 s (2026-09-26)
    * **a bug the float property found while building**: a point with an int and a float end,
      `(Cut(0, BELOW), Cut(0.0, ABOVE))`, gave `mid` the int 0 for a float operand; now 0.0, pinned
      as an `@example`
* evidence, measured 2026-09-26 (census by importing the test module): 167 vectors of the six ops
  (`libieeep1788_num` 126, `mpfi` 41), all matching in both passes except the 6 `[nai]` rows; all
  ops: 4949 vectors of 64 ops, 4120 interval-valued, 167 numeric run twice more with floats, 9403
  vector test items, 55 divergence keys (40 decoration expectations), 0 unknown failures; skipped
  4593 statements of 47 ops. the gate: 12225 passed in 415 s
* sabotage (section 2), each red, then the file restored from a copy and `cmp`-checked; targeted
  runs (`tests/test_numeric.py`, the two modules' doctests, the num and mpfi vectors):
  `mid` rounded up 23 red (the float property, `test_vector_float` at `num.itl:105`, `:106`,
  `mpfi.itl:1085`, `:1088`, `:1090`; the first pass cannot see it, its operands are exact);
  `mig` of the hull 7 (`test_examples` for D9's sets, both value properties, the `mig` doctest);
  `rad` as half the width 19 (`test_rad_is_from_the_rounded_midpoint`, `num.itl:135`, `:148` among
  them); `wid` to nearest 2, `mag` to nearest 1, `mig` up 2 (the float property, the mixed
  example); the half-bounded `mid` sign flipped 29; `mig` of the empty set 0 instead of raising 8
  (`test_empty_raises`, `num.itl:237`, `:251`, both passes); `rad` of a half-bounded hull not inf 18
  (`test_hull_and_set` among them); `wid` of the first piece 10 and the hull ends of the first
  piece 12 (both isotonicity properties among them); `mag` as the smaller end 63. the wiring: the
  adapter's `rad` rounded up naively 2 (`num.itl:135`, `:148`), its `mid` rounded up 7, the float
  pass not mapping `ValueError` to `NaN` 24. **not red, as expected**: `OPS['mig']` taking the
  hull's `mig`, since every vector is one interval, where hull and set agree; D9's set reading is
  held by the property tests only (the `mig`-of-the-hull sabotage above)

**M13c interval orders (done 2026-09-26)** (D10). `less` 88, `strictLess` 32, `interior` 64
* `A.weakly_less(B)`: `inf A ≤ inf B` and `sup A ≤ sup B`; `A.strictly_less(B)`: both strict,
  except that equal infinite ends count, as in 1788. on the ends, so the hull's, for a multi-interval
* `B.interior`: a property, the set with every end opened (the topological interior); `interior`
  in the vectors is `A.within(B.interior)`. 1788's unbounded ends arrive open at inf through the
  input rule, which is what makes `interior [1, infinity] [0, infinity]` true
* done 2026-09-26. built:
    * `intervals/relations.py::weakly_less`, `::strictly_less` over cut tuples, and the methods
      `MultiInterval.weakly_less(B)`, `.strictly_less(B)` (a number is coerced to a point, anything
      else is a `TypeError`, as for `within`). on the ends as values, so a multi-interval's are its
      hull's and open or closed does not matter (`[0, 2]` is weakly, not strictly, less than
      `[0, 2)`). `intervals/kernel.py::interior` and the property `MultiInterval.interior` (the
      class is kept, so an `OutwardMultiInterval`'s is one)
    * empty sets as 1788's vectors have them: two empty sets are weakly and strictly less than each
      other, an empty and a non-empty set are neither (`less [empty] [1.0,2.0] = false`); the
      interior of the empty set is empty, so `interior [empty] B` is true for every `B` and
      `interior A [empty]` false for a non-empty `A`
    * **choices the plan left open** (conservative, flagged in the decision log):
        * "equal infinite ends count" is read exactly as 1788 writes it: two starts at **-inf**,
          two ends at **+inf**. a start at +inf or an end at -inf is a point there (`[inf]`,
          `[-inf]`, which 1788 has no interval for), and is not strictly less than itself; the
          looser reading, any equal infinite ends, would make `[inf]` strictly less than `[inf]`
        * `.interior` opens the infinite ends too: the interior in the topology of the reals,
          not of the extended reals, so a closed end at inf is a point that drops out
          (`[5, inf]` → `(5, inf)`, `[inf]` → `∅`, `[-inf, inf]` → `(-inf, inf)`), as the plan's
          "every end opened" says. a degenerate piece drops out (`[2]` → `∅`), and pieces stay
          apart (`[0, 1) | (1, 2]` → `(0, 1) | (1, 2)`). the extended reals' interior would keep
          `(5, inf]` and make `[-inf, inf]` its own interior
    * adapter (`tests/itf1788/test_itf1788.py`): `less`, `strictLess`, `interior` in `OPS`, the
      last as `a.within(b.interior)`. no rule and no listed row: the 184 vectors match, except
      the 12 with a `[nai]` operand, generated rows under "decoration expectations"
    * tests (`tests/test_orders.py`, 57 items in 25 s, 2026-09-26): 7 `@given`: both orders
      against an oracle written from 1788's quantified definitions (every point of each operand
      has one of the other on the right side), decided by brute force on a half-integer grid over
      1788's reading of the hulls (`::_quantified`); the orders from the public ends, unchanged by
      the hull and the closed hull; `weakly_less` a preorder and `strictly_less` transitive;
      soundness at sampled points (exact and float, both classes: a point of either operand has
      one of the other's closure on the right side, strictly for the real points); the interior
      against its definition at probe points (a point is interior iff it is a real point of the
      set and not an end value, normalized pieces being maximal), every piece open; the laws
      (inside the set, idempotent, isotone, `int(A & B)` = `int A & int B`, the class kept); and
      `A.within(B.interior)` against a grid neighbourhood oracle (`::_interior_1788`).
      `@example`s: the empty cases, `[entire]`, `interior [1, infinity] [0, infinity]`,
      `interior [0.0,0.0] [-0.0,-0.0]`, mpfi's `less [0.0, 0.0] [0.0, +infinity]`, `[inf]` and
      `[-inf]` against themselves, `[5, inf]`, the int/float point `(Cut(0, BELOW), Cut(0.0, ABOVE))`
* evidence, measured 2026-09-26 (census by importing the test module): 184 vectors of the three
  ops (`libieeep1788_bool` 124, `mpfi` 32 `less`, `c-xsc` 28 `interior`), all matching except
  the 12 `[nai]` rows; all ops: 5133 vectors of 67 ops, 4120 interval-valued, 167 numeric run twice
  more with floats, 9587 vector test items, 67 divergence keys (52 decoration expectations), 0
  unknown failures; skipped 4409 statements of 44 ops. the gate: 12470 passed in 428 s
* sabotage (section 2), each red, then the file restored from a copy and `cmp`-checked; targeted
  runs (`tests/test_orders.py`, the doctests of `relations.py`, `kernel.py`, `multi_interval.py`,
  every itf1788 vector): `weakly_less` with `<` on the starts 37 red (the 1788-definitions and
  on-the-ends properties, 9 examples, 26 vectors, `mpfi.itl:902` among them); `strictly_less`
  with `or` for `and` 19 (the preorder and soundness properties among them, 6 vectors); without
  the rule for two starts at -inf 9 (`bool.itl:412`, `:437`); two empty sets not weakly less 6
  (`bool.itl:227`, `:265`); `weakly_less` comparing `inf A` with `sup B` 18 (preorder, soundness,
  12 vectors); the interior keeping a closed end at ±inf 9 and the interior of the hull 10 (the
  probe-point, neighbourhood and laws properties, the examples, both doctests). **no vector sees
  either interior sabotage, as expected**: the input rule never gives a closed inf, and each
  vector is one interval, where hull and set agree; these are held by the property tests only.
  the wiring: `interior` as `a.within(b)` 16 red (`bool.itl:368`, `c-xsc.itl:132` among them),
  `strictLess` as `weakly_less` 10, `less` with its operands swapped 34

**M13d power and the rest of the elementary functions (done 2026-09-26)** (D11). every value correctly rounded in
pure python, in `intervals/elementary.py` at one point and `intervals/functions.py` over sets, as at
M12; each checked against the D14 oracle (M14)
* `pow` 1431: `__pow__` as D11 (integral exponent → pown as today; non-integral float or
  `MultiInterval` exponent → 1788 pow, negative bases dropped with `DomainClippedWarning`,
  `0 ** y` = 0 for `y > 0`), `__rpow__` for a scalar base, exact where the value is rational.
  3-argument `pow` stays a `TypeError` (dropped, D11)
* `expm1` 38, `logp1` 37 (method `log1p`, python's name), `cbrt` 10, `rootn` 3 (`rootn(n)`),
  `hypot` 17 (`hypot(B)`), `csc` 109, `sec` 109, `cot` 49, `acot` 30, `coth` 46, `acoth` 30, `csch` 16,
  `sech` 14. python's name where python has one, 1788's otherwise. the poles of csc, sec, cot,
  coth, csch split a piece as tan's do; `acoth`'s domain is `|x| ≥ 1` with the M12 rule for a
  domain's end (the one-sided limit, so ±inf at ±1: expect degenerate-infinity rows like atanh's)
* done 2026-09-26. built:
    * `intervals/elementary.py`: the twelve functions at an exact point in `NAMES` (and `rootn`
      through the `base` argument, as its degree n), with their exact values (`exact`: 0 or 1 at
      the obvious points, the rational roots, the limits at ±inf and at a domain's end) and their
      enclosures (`_enclose`): `::_expm1_fractions` (exp at extra precision near 0, by
      `::_tiny_bits`), log1p as the log of the exact `1 + x`, `::_root_fractions` as
      `exp(ln|x| / n)`, cot/csc/sec from `_sin_cos`, acot as `atan(1/x)` (plus pi below 0),
      `::_reciprocal_hyperbolic` (coth and csch through expm1, sech through exp), acoth as
      `ln((x + 1)/(x - 1)) / 2`. `_beyond` knows where expm1, coth, csch and sech round like a value
      next to -1, ±1 or 0 or past the float range. `POLE_AT_ZERO` (cot, csc, coth, csch): `exact`
      raises `ValueError` there. `::exact_pow` (rational iff x is a b-th power for y = a/b, via
      `::_exact_root` and the integer root `::_iroot`; not built past `EXACT_POWER_LIMIT` bits)
      and `::rounded_pow` (`exp(y ln x)` through ziv, overflow and underflow decided first from
      `::_ln_bracket`, two rationals of one sign within a factor of 3.1 of `ln x`)
    * `intervals/functions.py`: the new names in `NAMES`, a public `::domain(name, base)`, the
      tables `MONOTONE` (expm1, log1p, cbrt, acot, sech with its new `'above 0'`, acoth on
      `|x| ≥ 1`), `RECIPROCAL_TRIG` (cot, csc, sec: the poles' and extrema's offsets in pi) and
      `POLE_AT_ZERO` (coth, csch). `_Function.reciprocal_trig` cuts a piece at its poles and
      extrema (`::_inside_k`, `floor_over_pi` as for sin) into monotone segments
      (`_Function.segment`); `_Function.pole_at_zero` for coth, csch and an odd negative root;
      `rootn` through `apply(name, a, base=n)` (`::_check_degree`). `::pow_` over boxes as atan2
      does (`::_power_box`, `::_power_corner`, `::_POW_EXTREMES`), and `::hypot` (exact sums of
      squares, one square root; `_Function`'s new `float_operands`)
    * `intervals/multi_interval.py`: the methods `expm1()`, `log1p()`, `cbrt()`, `rootn(n)`,
      `hypot(B)`, `cot()`, `sec()`, `csc()`, `acot()`, `coth()`, `csch()`, `sech()`, `acoth()`;
      `__pow__` per D11 (`::_is_integral` for a number's value) and `__rpow__`, overridden in
      `OutwardMultiInterval` as the other reflected dunders are
    * **choices the plan left open** (conservative, flagged in the decision log):
        * a Fraction with an integral value is pown, as D11 says of a float; any `MultiInterval`
          exponent is pow, `[2]` included
        * `0 ** y` for y <= 0 is outside pow's domain (D11's words), dropped with the
          `DomainClippedWarning`, not an `IndeterminateResultWarning`, and the one-sided limit
          `0 ** -1` = inf is not taken: `[0, 1] ** [-1]` is `[1, inf)`, pown's `[0, 1] ** -1` is
          `[1, inf]`
        * ±inf in pow are points where the power has a limit; `1 ** ±inf` and `inf ** 0` are
          indeterminate points, atan2's treatment (nothing and a warning for a box that is one,
          the other points' values for a larger box). a corner is attained iff in the box or on a
          closed infinite edge
        * the poles at 0 (cot, csc, coth, csch, an odd negative root) follow `reciprocal`: a piece
          ending at 0 takes the one-sided limit, closed iff it holds 0; `[0]` alone is empty with
          an `IndeterminateResultWarning`; a pole inside a piece gives both infinities, attained
        * acot is continuous, `pi/2 - atan x` in (0, pi) (fi_lib's; its vectors are all positive,
          where the conventions agree), not `atan(1/x)`
        * rootn for every int n other than 0 (1788's): n < 0 is the root of `1/x`, an even root's
          domain x >= 0 with `rootn(0, -2)` = inf at its end; `rootn(x, 0)` a `ValueError`
        * hypot rounded once from the exact sums of squares, in the receiver's class
    * adapter (`tests/itf1788/test_itf1788.py`): the 14 ops in `OPS` (`logp1` as `log1p()`,
      `rootn` with its int, `pow` as `a ** b`: an interval exponent, so 1788's pow). no rule and no
      row: all 1939 vectors match in both passes. 41 of them end at or are the pole 0 of cot, csc,
      coth or csch, where the closed hulls agree; no vector has acoth at ±1
    * tests:
        * `tests/test_elementary.py`: the decimal oracle for the eleven unary functions (working
          precision grown by the operand's distance from 1, `::_oracle_new`), so
          `test_correctly_rounded` and `test_enclosures_hold_the_value` cover them; 30 extreme
          points, 9 known roundings past the float range, 36 exact values, the poles raising;
          `::test_rootn_correctly_rounded` for 9 degrees, `::test_exact_rootn`,
          `::test_iroot_is_the_floor`; `::test_pow_correctly_rounded` (150 draws, three kinds, at
          least 90 checked), `::test_exact_pow`, `::test_known_pow_roundings`,
          `::test_ln_bracket_holds_ln`
        * `tests/test_oracle_flint.py`: the eleven through `_ARB` (arb's own expm1, log1p, root,
          cot, sec, csc, coth, csch, sech; acot and acoth through atan and atanh), log transforms
          where arb cannot hold the value (`::_shifted_logged_sign` for expm1 near -1 and coth near
          1, the log of expm1, csch, sech far out), 42 more extremes, 14 more ends and 8 more
          outside points; `::test_rootn_against_arb`, `::test_pow_against_arb` (through
          `log(x ** y) = y log x`, so past the float range too), `::test_hypot_against_arb`
        * `tests/test_functions.py`: the property tests over every name through `domain()`, the
          poles at 0 skipped where there is no value and the union law relaxed to the finite
          points there; `::test_m13d_examples` (46), `::test_a_pole_at_zero_has_no_value_there`,
          `::test_the_infinities_of_a_pole_at_zero` (±inf in the result iff the operand reaches 0
          from that side while holding it; coth, csch, rootn -3 and -1),
          rootn (examples, domain and poles, refusals, soundness and isotonicity over 12 degrees,
          `cbrt` = `rootn(3)`), pow (28 examples, 5 outward, domain and indeterminate warnings, and
          `@given`: soundness on exact sets, ends sharp, isotone and distributing over a union in
          each argument, outward soundness on floats, nearest holding the nearest power), hypot
          (examples, outward, soundness on exact and float sets, isotone and symmetric)
        * `tests/test_applicator.py`: D11's dispatch (`::test_pow_dispatch`), numbers refused,
          3-argument pow refused, `::test_rpow_is_pow_of_a_point` (the class on both sides)
    * **a bug found while building, in the tests' first draft, not the library**: my hand-written
      expected strings for six set examples were wrong (cot on `[1, 4]` taken on the wrong branch,
      five last digits); arb agreed with the library on each
* evidence, measured 2026-09-26 (census by importing the test module, `tools/itf1788_census.py`):
  1939 vectors of the 14 ops (by file: libieeep1788_elem 1428, all pow; mpfi 329; fi_lib 176; c-xsc
  6, pow 3 and rootn 3),
  all matching in both passes; all ops: 7314 vectors of 83 ops, 6301 interval-valued, 167
  numeric run twice more with floats, 13949 vector test items, 114 divergence keys (unchanged),
  0 unknown failures; skipped 2228 statements of 28 ops. the gate: 17346 passed in 373 s (13000 at M13f)
* sabotage (section 2): 25 breaks, each run against its targeted tests by a throwaway harness
  that restored the file from a copy and byte-compared it (`filecmp`), results appended as they
  landed. red: csc/sec extrema of the wrong sign 207 (19 unit, 188 mpfi vectors); the side of a
  pole flipped 7 (isotonicity, union, examples; **no vector**: a piece holding a pole hulls to
  entire either way); the pole 0 as a start giving -inf 13 (8 vectors); the whole range already
  at two poles 1 at first (only `sec [1, 5]`), 2 after adding `csc [3, 7]`; the infinity at the
  pole 0 always closed 1 at first (only `coth (0, 1]`), 3 after adding
  `::test_the_infinities_of_a_pole_at_zero` with `@example`s for an open 0 (**no vector**: the
  closed hull hides attainment); pow's corners of `(x < 1, y > 0)` swapped 1077 (1060 vectors);
  `0 ** y` = 0 for y <= 0 558 (554 vectors); no attainment on a closed infinite edge 4 (the
  properties; no vector, the input rule never closes an infinity); float corners to nearest when
  outward 164 (160 outward vectors); overflow decided at `ln > 80` 2; no rational root for a
  non-integral exponent 3 (`test_exact_pow`), and its other two targets **hung**, ziv climbing
  toward `_MAX_PRECISION` on a rational value it cannot settle, killed after 835 s and 140 s: a
  missed exact case costs minutes before the `ArithmeticError`, as M12 designed it; `ln 2 > 0.8`
  in the bracket 3; expm1's -1 from -30 2 at first (two mpfi vectors), 4 after adding the
  extreme points -37 and -35; coth's 1 from `|x| >= 10` 5; csch/sech's 0 from 700 6; acot
  without pi below 0 11 (no vector: fi_lib's are all positive); rootn not reciprocated for n < 0
  7; hypot's float operands as exact 3 (the examples; the vectors test outward, where the
  result is the same enclosure); an integral float exponent as pow 2; sech's direction reversed
  35 (26 vectors); cbrt losing the negative root 16. the wiring: `logp1` run as expm1 70 vectors,
  `pow` run as pown on a degenerate integral exponent 56 vectors. **green, as argued**: expm1's
  and log1p's extra bits near 0 (`_tiny_bits`) removed, 0 red: they save ziv iterations, and
  without them ziv doubles the precision until the ends agree, so the value is the same

**M13e reverse ops (done 2026-09-26)** (D12). 1955 statements
* a new module `intervals/reverse.py`, the functions exported from `intervals`, each taking the
  constraint first and the domain `x` last, defaulting to the whole line: `sqr_rev(c, x=REALS)`,
  `abs_rev(c, x=REALS)`, `pown_rev(c, n, x=REALS)`, `sin_rev`, `cos_rev`, `tan_rev`, `cosh_rev`
  (all `(c, x=REALS)`), `mul_rev(b, c, x=REALS)`, `pow_rev1(b, c, x=REALS)` (the bases x with
  `x ** y ∈ c` for some `y ∈ b`), `pow_rev2(a, c, y=REALS)` (the exponents). the `*Bin` vectors are
  the two-argument call and `mulRevTen` is `mul_rev` with `x` given
* each result is `{x ∈ X : f(x) ∈ C}` (for the binary ones, `∃` over the other operand) as an exact
  multi-interval: `mulRevToPair`'s two intervals are one value here, compared by the adapter as a
  union (our closed pieces against the pair's). ends that are irrational are tightest float
  enclosures, open (for exact operands; float operands round to nearest, closed, in `MultiInterval`
  and outward, open, in `OutwardMultiInterval`: part 1's choice, the library's rule)
* periodic answers per D12: exact pieces up to 1000, past that their hull with `HullWarning`
* **part 1 done 2026-09-26 (branch `m13e`): `sqr_rev`, `abs_rev`, `pown_rev`,
  `cosh_rev`** (476 statements); sin, cos, tan, mul and pow in parts 2 to 4 and the close-out
  below. built:
    * `intervals/reverse.py`: `::sqr_rev`, `::abs_rev`, `::pown_rev`, `::cosh_rev`, exported from
      `intervals` (`tests/test_applicator.py::test_package_exports_unchanged` lists them). the
      engine, for the later reverse ops (the design is in the module docstring): `::Branch` (a
      piece of f's domain where f is continuous and strictly monotone, given by its image, each end
      closed iff attained, and its inverse as `exact` and `rounded`), `::named` (a branch whose
      inverse is one of `elementary`'s correctly rounded functions: `sqrt`, `rootn` with its degree,
      `acosh`), `::branch_preimage` (the inverse applied piece by piece to `c ∩ image`, flags kept,
      ends swapped for a decreasing f), `::_end` (the rounding of one end, `functions._Function.end`'s
      rule), `::negate`, `::_even` (`P ∪ -P`), `::_odd` (`P(c) ∪ -P(-c)`), `::_reverse` (coercion,
      the class, the empty-operand warning, then `∩ x` after rounding). the four ops are one branch
      on `[0, inf]` each: identity (abs), sqrt (sqr), rootn(n) (pown; for n < 0 the image is
      `[0, inf)`, falling, 0 attained at inf), acosh on `[1, inf]` (cosh); `pown_rev(c, 0, x)` is
      `x` if `1 ∈ c`, else `∅`. a later op with many branches (sin: `k pi + (-1)^k asin`) gives each
      branch its own `exact`/`rounded` and unions `branch_preimage` over them, capped per D12
    * **choices the plan left open** (conservative, flagged in the decision log, `v2-plan.md`
      "2026-09-26 revision: M13e, first part"):
        * `x=REALS` is `[-inf, inf]` with f's own values at ±inf (`MultiInterval(inf) ** -2` is
          `[0]`), so `pown_rev([0], -2)` = `[-inf] ∪ [inf]`; the vectors this touches are rows
        * a point where f has no value (0 for n < 0) is in no preimage: the result is pointwise,
          although the set op `1/[-1, 1]` attains ±inf around 0
        * an empty operand gives `∅` and `EmptySetPropagationWarning`, as the functions do; an empty
          answer from non-empty operands warns nothing
        * the result is an `OutwardMultiInterval` if either operand is one (the dunders' rule); a
          number is a point; `n` an int, not bool (`TypeError` otherwise, `n = 0` allowed)
        * `∩ x` after rounding, so an end of `x` inside an irrational end's slack is kept and the
          result never leaves `x`
        * a float end produced by an earlier irrational enclosure is a float operand when fed back
          into a `MultiInterval` (rounded to nearest), the library-wide rule: `T ⊆ rev(f(T))` is
          promised for cosh only in `OutwardMultiInterval` (the property test runs in that class)
    * adapter (`tests/itf1788/test_itf1788.py`): `sqrRev`, `sqrRevBin`, `absRev`, `absRevBin`,
      `pownRev`, `pownRevBin`, `coshRev`, `coshRevBin` in `OPS`, one labelled block; the unary form is
      the call with `x` omitted, the `*Bin` form passes `x` through the input rule; the exponent, a
      1788 integer literal (a float in the outward pass), is made an int. **the comparison rule**:
      the output rule unchanged, closed hulls, since 1788's reverse op is by definition the hull of
      the preimage; the pieces inside the hull are held by the property tests. rows:
      `::_POWN_REV_ROWS` (26 keys, reason `::_POWN_REV_INF`, "degenerate infinities") and
      `::_POWN_REV_LOOSE_ROWS` (2 keys, `::_POWN_REV_LOOSE`) under **"tighter than the vector", a
      new category PROPOSED here, added to `REASONS` with a PROPOSED comment, awaiting the owner**
    * **a finding: two 1788 vectors are not tight.** `pownRev [0X0P+0,0X0.0000000000001P-1022] -7 =
      [0x1.588cea3f093bcp+153,infinity]` (`rev.itl:276`, its mirror `:277`, decorated `:477`, `:478`):
      the end is `2 ** (1074/7)` = 1.53674635563762978699...e46 (arb at 200 bits), strictly between
      the doubles 0x1.588cea3f093bdp+153 and 0x1.588cea3f093bep+153; ours is `(0x...bd, ...)`, 1788's
      one double lower. the infinity row would have hidden it: it was caught by the check that each
      row matches once `x` is 1788's entire
    * tests (`tests/test_reverse.py`, 50 items in 15.5 s, 2026-09-26): 9 `@given`, over the four ops
      and pown with n in {-8, -7, -3, -2, -1, 0, 1, 2, 3, 4, 7, 8}: `::test_exactly_the_points_with_f_in_c`
      (exact operands; at every end value, a point between each two, beyond each end, ±inf and a
      fraction of an ulp either side of every float end: soundness, every `t` of the result with
      `f(t) ∈ c` except inside an open rounded end's one-double slack, and tightness, the next
      double inward from a rounded end is a true point; `f(t)` from the definitions in the test,
      cosh through `elementary`), `::test_the_largest_set` (`T ⊆ rev(C)` whenever `f(T) ⊆ C`, the
      library's `f` on sets), `::test_the_image_of_the_preimage` (`f(rev(f(T))) = f(T)`, all but
      cosh), `::test_isotone` (in `c` and `x`), `::test_union_and_x` (distributes over `∪` of `c`;
      `rev(c, x) = rev(c) ∩ x`), `::test_symmetry` (even, odd), `::test_relations_between_the_ops`
      (`sqr_rev` = `pown_rev(., 2)`, sqrt against rootn 2; `pown_rev(., 1)` = `c ∩ x`;
      `pown_rev(., 0)`; `abs_rev`), `::test_float_operands` (outward holds the exact result of the
      same doubles and adds no double strictly inside what it adds; nearest within one double and
      inside the outward closure), `::test_sound_at_sampled_points`. every property's `rev` asserts
      an empty operand warns and gives `∅`, and nothing else warns. `@example`s: `rev.itl:35`, `:52`,
      `:189`, `:217`, `:224`, `:246`, `:261`, `:276`, `:289`, `:322`, `:760`, `:762`, `abs_rev.itl:29`,
      `:35`, closed inf in `c` for n < 0, an `x` end inside the slack; plus 35 parametrized
      examples, the enclosures, class and coercion, warnings, and
      `::test_the_rows_differ_only_at_the_infinities`, `::test_pown_rev_is_tighter_than_the_vector`
* evidence, measured 2026-09-26 (`tools/itf1788_census.py`): the 476 vectors of the eight ops
  (`libieeep1788_rev.itl` 452, `abs_rev.itl` 24; all interval-valued, none with `[nai]`): 420 match
  in both passes; 52 (26 keys) are degenerate infinities of the unary `pownRev` with n < 0 and 0 in
  `c`, each matching with 1788's entire as `x`; 4 (2 keys) the proposed category. all ops: 7790
  vectors of 91 ops, 6777 interval-valued, 142 divergence keys (36 degenerate infinities, 2
  proposed), 0 unknown failures; skipped 1752 statements of 20 ops. the gate, as two runs on
  2026-09-26: `tests/itf1788` 14953 passed in 33.6 s, the rest 3401 passed in 392.7 s (18354 in
  all; 17346 at M13d)
* sabotage (section 2), 2026-09-26: 22 breaks, each run by a throwaway harness against
  `tests/test_reverse.py`, the doctests of `intervals/reverse.py` and every itf1788 item matching
  `rev` (1018 items), the file restored from a copy and byte-compared (`filecmp`, all equal),
  results appended as they landed. red: an irrational end closed 6 (the enclosure, defining,
  sampled-soundness and arb tests, 2 doctests; **no vector**: the closed hull hides a flag); the
  ends' rounding directions swapped 252; a decreasing branch's ends not swapped 255; `_even`
  without its mirror 320; `_odd`'s negative side from `c` not `-c` 219; the n < 0 image closed at
  inf 4 (3 examples, the defining property; **no vector**: the input rule never closes inf); the
  n < 0 image open at 0 116 (the ±inf rows go stale); `x` not intersected 189; `n = 0` always the
  whole line 26; cosh's image from 0 13; the class from `c` only 1 (`test_class_and_coercion`; no
  vector, their operands share a class); outward ignored 123 (the outward vector items, the float
  property); no empty-operand warning 9 (every property through the test's `rev`); float ends
  directed in nearest 1 (the enclosure test; no vector, the plain pass is exact and the outward
  pass directed); parity swapped 350. the wiring: `pownRevBin` with `c` and `x` swapped 66,
  `sqrRev` run as `abs_rev` 16, the 26 infinity rows dropped 104, `coshRevBin` ignoring `x` 16,
  the 2 loose rows dropped 8. **first green, now pinned**: an outward end moved by rounding but
  kept closed (0 red: sound, so no soundness check sees it) and the squeeze rule removed (0 red:
  hypothesis never drew an open piece of `c` narrow enough). pinned by
  `test_irrational_ends_are_open_one_ulp_enclosures` (`pown_rev(O(3.0), -1)` open at both ends;
  `sqr_rev` of `(2, 2 + ulp)` to nearest is `±sqrt 2`), a new clause of `test_float_operands` (an
  outward end is closed only if it is a point of the exact result; the nearest result is not
  empty when the exact one is not) and two `@example`s on it; re-run, each now turns 2 red
* **part 2 done 2026-09-26 (branch `m13e`): `mul_rev`** (mulRev 182, mulRevTen 10, mulRevToPair
  347 statements). built:
    * `intervals/reverse.py::mul_rev(b, c, x=REALS)`, `{t ∈ x : t * y ∈ c for some y ∈ b}` with the
      library's `*`, exported from `intervals` (`test_package_exports_unchanged` lists it). `*` is
      the set of the values of the defined pairs, so `t` is in iff `{t} * b` meets `c`;
      `::_mul_preimage` takes the cases by the kind of `t` (the derivation is in `mul_rev`'s
      docstring): `t = 0` iff `0 ∈ c` and `b` has a finite point (`0 * ±inf` has no value); a
      finite `t != 0` from `ops.div(c ∩ R*, b ∩ R*)` (`R*` the finite nonzero reals, `::_NONZERO`),
      from `y = 0` (every finite `t` if `0 ∈ b` and `0 ∈ c`) and from `y = ±inf` (the finite `t` of
      the sign that makes an infinity of `c`, `::_FINITE_OF_SIGN`); `t = ±inf` where `c` holds the
      infinity it makes with a nonzero `y` of `b` (`::_SIDE`). no branch engine: `*` is not a
      function of one variable. `::_reverse` gained `given=()`, a binary op's other operand, so its
      coercion, class, empty-operand warning and `∩ x` after rounding are shared (pow_rev1/2 can
      reuse it)
    * **choices the plan left open** (conservative, flagged in `v2-plan.md` "2026-09-26 revision:
      M13e, second part"):
        * **the library's own `*` defines it**, so `mul_rev([inf], [0])` is `∅`, `mul_rev([0],
          [0])` is `(-inf, inf)` (not `[-inf, inf]`) and `mul_rev([1, inf], [3], [0, 10])` is `(0,
          3]`; 1788's reals have `0 * y = 0` for every `y`, but no 1788 vector has an infinite
          point in `b`, so none sees the difference
        * **rounding is the division's**: the quotients come from `ops.div`, so an end is exact
          for int and Fraction ends and rounded (to nearest, flags kept; outward, a moved end
          open) exactly where `c / w` would round it, and `mul_rev([w], c) = c / w` for a finite
          `w != 0` in both classes. the first draft rounded the whole result once when an operand
          had a finite float end, as `fma` and `cancel_minus` do; `test_mul_rev_by_a_point` found
          `mul_rev([3], (-2, 0.0))` = `(-0.6666666666666666, 0.0)` against `(-2, 0.0) / 3` =
          `(-2/3, 0.0)`, and parity with the division (and with part 1's per-end rule) won
        * `x` defaults to `[-inf, inf]`; the class, the coercion of a number, the warnings (an
          empty `b`, `c` or `x`: `EmptySetPropagationWarning`; nothing else, the `0 * inf` corner
          included) and `∩ x` after rounding are part 1's. the operand order is 1788's, `b` first
        * to nearest, flags kept, an end of `x` inside the half ulp a rounded end moved can be
          lost (`b = [10.0]`, `c = (1.0, 2.0)`, `x = [0.1]`: exactly `[0.1]`, since the double 0.1
          is above 1/10; to nearest `(0.1, 0.2) ∩ [0.1]` = `∅`), the library's nearest rule, not a
          promise; `OutwardMultiInterval` keeps it
    * adapter (`tests/itf1788/test_itf1788.py`): `mulRev`, `mulRevTen`, `mulRevToPair` in `OPS`
      (one labelled block), all `mul_rev`; **the pair rule** (`::PAIRS`, `::_pair`,
      `::_pair_of_expected`, hooks in `::run` and `::run_outward`): each of our pieces closed
      (rounded outward in the first pass; in the outward pass taken as they are, asserted doubles)
      and compared in order with the pair's non-empty intervals, **piece by piece**, stricter than
      the union the plan allowed. `INTERVAL_VECTORS` includes the pair vectors, so they run in
      the outward pass too. no rule changed, no new row beyond the 6 generated `[nai]` ones
    * tests (`tests/test_reverse.py`, its `mul_rev` section, 31 items in 75.5 s under load,
      2026-09-26): 7 `@given`, all with `deadline=None` (the shared laptop's load made the default
      200 ms deadline flake in the first sabotage run): `::test_mul_rev_is_exactly_the_points_that_fit`
      (300 examples; `t ∈ result` iff `t ∈ x` and `{t} * b` meets `c`, the library's `*`, at every
      quotient of an end of `c` by a nonzero end of `b`, every end of `x`, 0, a point between each
      two, one beyond each end and ±inf: soundness and maximality at once, as
      `tests/test_cancel.py::test_exactly_the_points_that_fit`), `::test_mul_rev_the_largest_set`
      (`T ⊆ mul_rev(B, T * B ∪ more)`, the `t` with `{t} * B` empty aside, and with `x`),
      `::test_mul_rev_isotone_and_distributive` (in `b`, `c`, `x`; over unions of `b` and of `c`;
      `x` only intersects), `::test_mul_rev_symmetry` (`-b`, `-c`, both classes),
      `::test_mul_rev_by_a_point` (`[w]` is `c / w`, `[0]` is every finite `t` or nothing, both
      classes), `::test_mul_rev_float_operands` (outward holds the exact result of the same
      doubles, adds no double strictly inside what it adds, closes only exact points; nearest,
      `x` omitted, within one double of the exact result and inside the outward closure, not
      empty when it is not; then `x` only intersects), `::test_mul_rev_sound_at_sampled_points`.
      `@example`s: `mul_rev.itl:32`, `:34`, `:35`, `:36`, `:42`, `:102`, `:106`, `:193`,
      `rev.itl:979`, `:980`, and the infinite points (`[inf]` against `[inf]`, `[0]`, `[-inf, 0]`;
      `[0] ∪ [inf]`; `[0]` against `{-inf, inf}`); plus 22 parametrized examples,
      `::test_mul_rev_1788_float_vector` (`mul_rev.itl:34`'s two doubles, open) and
      `::test_mul_rev_class_coercion_and_warnings`
* evidence, measured 2026-09-26 (`tools/itf1788_census.py`): the 539 vectors of the three ops
  (`libieeep1788_mul_rev.itl` 347, all `mulRevToPair`; `libieeep1788_rev.itl` 192, `mulRev` 182 and
  `mulRevTen` 10; all interval-valued, the pairs included): 533 match in both passes, the pairs
  piece by piece (298 one piece, 32 two, 14 empty, besides the 3 `[nai]` ones); 6 have a `[nai]`
  operand (generated rows, "decoration expectations", for M13g to move). no new row, no new
  category. all ops: 8329 vectors of 94 ops, 7316 interval-valued, 148 divergence keys (58
  decoration expectations), 0 unknown failures; skipped 1213 statements of 17 ops. the gate, as two
  runs on 2026-09-26: `tests/itf1788` 16031 passed in 83.9 s, the rest 3433 passed in 712.6 s under
  the shared laptop's load (19464 in all; 18354 at part 1)
* sabotage (section 2), 2026-09-26: 18 breaks by a throwaway harness, each file restored from a
  copy and byte-compared (`filecmp`, all equal), results appended as they landed. the library
  breaks against the `mul_rev` tests and `reverse.py`'s doctests (red counts from a second run
  with `deadline=None`; the first run's had deadline flakes), and against every itf1788 item
  matching `rev` (2038 items; deterministic): `t = 0` without a finite `y` 3 red (the defining
  property, `[inf]` by `[0]`, the warnings test; **no vector**: no 1788 `b` is infinite only);
  `t = 0` dropped 11 and 416 vector items; dividing by `b` with its 0 and infinities 10 (**no
  vector**: the closed hull hides the spurious ±inf and 0); `y = 0` dropped 9 and 198; `y = 0`
  without `0 ∈ c` 10 and 168; the finite `t` by `y = ±inf` with its sign flipped 6, dropped 8;
  `t = ±inf` with its sign flipped 9, dropped 10; `t = ±inf` also by `y = 0` 4 at first, the
  defining property not among them, **then pinned** by an `@example` (`b = [0]`, `c = {-inf,
  inf}`): 5; outward ignored 3 and 69 outward vector items; an empty `b` not caught 7; the class
  from `c` and `x` only 1 (`test_mul_rev_class_coercion_and_warnings`). **no vector sees any of
  the infinite-point clauses**: the input rule never gives `b` or `c` an infinite point, so
  these are held by the properties alone. **green, as argued**: dividing `c` with its 0 and
  infinities, 0 red (a 0 of `c` gives `t = 0`, which the `t = 0` clause gives too; an infinity
  of `c` gives `t = ±inf` with a finite `y`, which the `t = ±inf` clause gives; 3000 random
  pairs agreed, values and warnings). the wiring: `mulRev` with `b` and `c` swapped 134 red,
  `mulRevTen` ignoring `x` 20, the pair rule as one hull 64 (the 32 two-piece pairs, both
  passes), `mulRevToPair` without the pair rule 344
* **part 3 done 2026-09-26 (branch `m13e`): `sin_rev`, `cos_rev`, `tan_rev`** (sinRev 12, sinRevBin
  40, cosRev 12, cosRevBin 42, tanRev 10, tanRevBin 20 statements). built:
    * `intervals/reverse.py::sin_rev(c, x=REALS)`, `::cos_rev`, `::tan_rev`, exported from
      `intervals` (`test_package_exports_unchanged` lists them): `{t ∈ x : f(t) ∈ c}` for the
      library's sin, cos, tan at a point, which have no value at ±inf (`functions.domain`) and tan
      none at its poles. part 1's engine: `::_Periodic` names an op's branches and
      `::_trig_branch` makes the k-th, a `Branch` whose inverse is `m pi + sign g(v)`, g being
      `elementary`'s asin, acos or atan: sin `k pi + (-1)^k asin v` on `[k pi - pi/2, k pi +
      pi/2]`, rising for an even k; cos `k pi + acos v`, falling, for an even k and `(k + 1) pi -
      acos v`, rising, for an odd k; tan `k pi + atan v`, rising, its image `(-inf, inf)` open.
      `::_periodic_preimage` takes each piece of `x ∩ (-inf, inf)`, the branches from
      `elementary.floor_over_pi` at its ends plus one on each side, and unions `branch_preimage` over
      them. D12's cap follows `steps.step` (its `ENUMERATION_CAP` reused): a piece of `x` whose part
      would take the count past 1000 pieces, or that spans more than `::_BRANCH_LIMIT` branches, or
      is unbounded, gives its part's hull, and the call emits one `HullWarning`. the hull of a wide
      or unbounded piece comes from `::_periodic_hull`, walking inward from the piece's finite ends
      to the first branch with a solution (every branch has one). `::_BRANCH_LIMIT` is 2 × 1000 + 8:
      past it, a `c` that is not the whole image has more than 1000 pieces, since each period holds
      a solution and a point that is not one. `::_reverse` is reused with `given=(x,)`
      (`::_periodic`): the preimage needs `x` to know which branches to list
    * `intervals/elementary.py::rounded_inverse_trig(name, v, sign, k, direction)`: `k pi + sign
      f(v)` correctly rounded by ziv's loop, the working precision grown by k's bits so that a branch
      far out needs no extra doublings; `acos(-1)` = pi is folded into k, since `(k + sign) pi` is 0
      at `k = -sign`
    * **choices the plan left open** (conservative, flagged in `v2-plan.md` "2026-09-26 revision:
      M13e, third part"):
        * **a `c` holding sin's or cos's whole image `[-1, 1]` is not hulled**: every finite t is a
          solution, one piece whatever `x` is, so `sin_rev([-1, 1])` is `(-inf, inf)` with no
          warning (D12 hulls only an answer with too many pieces). tan has no such `c`: its poles
          leave a gap in every period, so `tan_rev((-inf, inf))` is `(-inf, inf)` with a `HullWarning`
        * **the poles are in no preimage**, part 1's rule for 0 in `pown_rev(c, -1)`, although the set
          op `tan` attains ±inf around a pole inside a piece: `tan_rev([inf])` is `∅`, no warning. a
          pole is irrational, so a piece around it cannot leave it out: the enclosures of the two
          branches' ends overlap and merge (`tan_rev([-inf, inf], [1, 2])` is `[1, 2]`)
        * **±inf are in no preimage**, so the default `x = [-inf, inf]` gives what 1788's entire gives
          (pinned by `tests/test_reverse.py::test_trig_rev_unary_forms_do_not_see_the_infinities`)
        * **the cap per piece of `x`**, as `steps.step` has it per piece of its operand: earlier
          pieces stay exact, the piece that would pass 1000 and every unbounded piece become their
          part's hull, one warning per call. a hull lies inside its piece of `x`, so the result never
          leaves `x`
        * a `c` whose gaps are single irrational points (`sin_rev([-1, 1))`, without 1 at `pi/2 + 2k
          pi`) has gaps, so a wide or unbounded `x` is hulled with the warning, although the ends'
          enclosures around each gap overlap and listing would give the same set
        * **one branch more on each side of a piece of `x`**, so that the result is the union of every
          branch's enclosure, then `∩ x`, even where `x` starts or ends inside the one-double slack of
          a neighbouring branch's end (just past or before a pole of tan). without it the result would
          depend on which branches were listed. pinned by two `@example`s
        * **far out, the doubles are coarser than a period** (their spacing passes pi above about
          1.8e16), so the enclosures of neighbouring solutions merge: `sin_rev([1/2], [1e20, 1e20 +
          7])` is `(1e20, 1e20 + 7]`, still an enclosure (pinned in
          `::test_trig_rev_far_out_against_arb`)
        * a float end of `c` is a float operand, rounded to nearest (flags kept) or outward (a moved end
          open), part 1's rule; the class, the coercion and the warnings are part 1's too
    * adapter (`tests/itf1788/test_itf1788.py`): `sinRev`, `sinRevBin`, `cosRev`, `cosRevBin`,
      `tanRev`, `tanRevBin` in `OPS`, one labelled block; the unary form is the call with `x` omitted,
      whose answer is the hull `(-inf, inf)` with a `HullWarning` (ignored in a vector), 1788's entire.
      rows: `::_TRIG_REV_LOOSE_ROWS` (7 keys, the undecorated `rev.itl:555` and its decorated copy
      `:595` differ by a space), reason `::_TRIG_REV_LOOSE`, under the **proposed category "tighter
      than the vector"** of part 1, not yet approved by the owner. no other row, no new category
    * **a finding: six more 1788 vectors are not tight** (12 with their decorated copies, all
      `*Bin` in `libieeep1788_rev.itl`): one end of 1788's hull is outside the tightest double
      enclosure of `k pi ± asin/acos/atan(v)`, the other end equal; ours is the tightest.
      `sinRevBin` `:555`/`:595` (pi/2, high end one double out), `cosRevBin` `:633`/`:675` (pi, high
      end one), `:642`/`:684` (high end one), `:643`/`:685` (low end one), `tanRevBin` `:711`/`:735`
      (-pi/2, low end two doubles out), `:713`/`:737` (-pi, low end one). checked with arb at 300 bits
      by `tests/test_reverse.py::test_trig_rev_is_tighter_than_the_vector`: the true end lies
      strictly between our double and the next one inward. a throwaway arb census over every
      bounded `*Bin` vector (2026-09-26) found ours the tightest hull for all of them
    * **a bug found while building**, by the fuzz profile at ×2: `rounded_inverse_trig('acos', -1,
      -1, 1, ...)` hung in ziv's loop, since the value `pi - pi` is 0, a rational it cannot settle.
      `cos_rev` never asks for it (its branches give `k pi` there, k odd), but the helper was wrong:
      acos(-1) is now folded into k, pinned by `::test_rounded_inverse_trig_exact_case` and an
      `@example`. the fuzz run also found a test-oracle weakness: arb's relative precision was too
      low to separate `asin(7.27e-245)` from its argument, so the oracle now raises its precision
      until the comparison is decided (an `@example` keeps that input)
    * **a test bug found by the first sabotage run**: `::test_trig_rev_float_operands` asked the
      nearest result with `x` to hold the exact one within a double, and hypothesis found `c =
      (-5.6e-24, 0.0)`, `x = (-inf, -5.6e-24)`: asin of the double -5.6e-24 rounds to nearest onto
      that double, so `x`'s open end cuts off the exact sliver between them, part 2's recorded
      nearest rule ("an end of `x` inside the half ulp a rounded end moved can be lost"). the
      assertion now takes the whole box as `x`, as mul_rev's does; the example is kept. that run's
      red counts all held this one replayed failure, so the harness was run again (below). the
      second run found another in `::test_trig_rev_symmetry`: the class's `-` rebuilds a point
      whose ends differ in type (`[0, 0.0]`, from the strategies) with one value, `[0, 0]`, so an end
      changed from float to exact and rounded differently. the test now mirrors cut by cut
      (`reverse.negate`, which keeps each end's type), with the point as an `@example`; the `-` on
      such a point is the library's own behaviour, not this op's, and is left as it is
    * tests (`tests/test_reverse.py`, its periodic section, 31 items in 66 s under the shared
      laptop's load, 2026-09-26): 8 `@given`, all with `deadline=None`:
      `::test_trig_rev_exactly_the_points_with_f_in_c` (150 examples, exact operands; over the bounded
      pieces of `x`, at every end, between each two, beyond, ±inf and a fraction of an ulp around
      every float end: soundness, the converse except inside a rounded end's one-double slack
      (`::trig_in_slack`), and tightness, the next double inward from a rounded end being a true
      point; over an unbounded piece, soundness and the part equal to the hull built from the exact
      result on a window of 7 at its finite end; ±inf never in it; `f(t)` from `elementary`, as for
      cosh; `::trev` requires a `HullWarning` exactly where `::trig_hulls` says, and nothing else to
      warn), `::test_trig_rev_the_largest_set` (`T ⊆ rev(f(T) ∪ more)`, tan's poles inside `T`
      included, in the outward class), `::test_trig_rev_isotone` (in `c` and `x`, hulls included),
      `::test_trig_rev_union_and_x` (over a bounded `x`, distributes over `∪` of `c`, and `x` only
      intersects), `::test_trig_rev_symmetry` (sin and tan odd, cos even, float operands and hulls
      included), `::test_trig_rev_float_operands` (outward holds the exact result of the same doubles,
      adds no double strictly inside, closes only exact points; nearest, over the whole box, within
      one double and inside the outward closure, then `x` only intersects, as mul_rev's test has
      it), `::test_trig_rev_sound_at_sampled_points`,
      `::test_rounded_inverse_trig_against_arb` (200 examples: asin, acos over `[-1, 1]`, atan over
      the reals and ±inf, k up to 2**400, each direction, arb deciding at a growing precision).
      `@example`s: `rev.itl:554`, `:555`, `:563`, `:569`, `:633`, `:644`, `:646`, `:647`, `:652`,
      `:708`, `:711`, `:715`, `:718`, `[inf]` for tan, `[-1, 1)` for sin, the two slack cases at tan's
      pole, `asin(-5.6e-24)`'s nearest rounding. plus 15 parametrized examples,
      `::test_d12_example`, `::test_trig_rev_hull_past_the_cap` (1000 pieces over `[0, 6280]`
      listed, 1001 over `[0, 6284]` hulled, equal to the hull of two halves; over `[0, 7] ∪ [8,
      6300]` the first piece listed with its gap, the second hulled alone), `::test_trig_rev_hull_of_a_wide_x` (`[0, 1e6]` and `[0, 1] ∪ [10, inf)`, all three
      ops, against windows), `::test_trig_rev_far_out_against_arb`,
      `::test_trig_rev_class_coercion_and_warnings`,
      `::test_trig_rev_unary_forms_do_not_see_the_infinities`,
      `::test_trig_rev_is_tighter_than_the_vector`, `::test_rounded_inverse_trig_exact_case`. under
      `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=3` the section ran green in 87 s (2026-09-26), after
      the two catches above
* evidence, measured 2026-09-26 (`tools/itf1788_census.py`): the 136 vectors of the six ops (all
  in `libieeep1788_rev.itl`, all interval-valued, none with `[nai]`): 124 match in both passes, the
  34 unary ones and the 16 `*Bin` ones over an unbounded `x` through the hull; 12 (7 keys) are rows
  under the proposed category. all ops: 8465 vectors of 100 ops, 7452 interval-valued, 155
  divergence keys (36 degenerate infinities, 9 proposed, 58 decoration expectations), 0 unknown
  failures; skipped 1077 statements of 11 ops. the gate, as two runs on 2026-09-26: `tests/itf1788`
  16303 passed in 55.4 s, the rest 3468 passed in 564.1 s under the shared laptop's load (19771 in
  all; 19464 at part 2)
* sabotage (section 2), 2026-09-26: 24 breaks by a throwaway harness, each file restored from a
  copy and byte-compared (`filecmp`, all equal), results appended as they landed; the library
  breaks against the periodic section of `tests/test_reverse.py` with the doctests of
  `reverse.py` and `elementary.py` (32 items), and against every itf1788 item matching `rev.itl`
  (2310). the counts are from the clean runs, after the two test bugs above (the first runs'
  counts each held a replayed failure of the buggy test). red: sin's branches all rising with
  `+asin` 8 and 44 vector items; sin's direction flags inverted 10 and 52; cos's odd branch shifted
  by `k pi` 7 and 40; cos's even branch rising 9 and 44; tan's image closed (a pole a solution for
  ±inf) 3 (**no vector**: the input rule never gives `c` an infinity); no whole-image case 8 (no
  vector: the hull of every finite t is the same set, only the warning differs); tan taken as whole
  2 (the warnings); ±inf of `x` kept 5 (no vector: the closed hull hides them); no extra branch
  below 1, above 1 (`::test_trig_rev_union_and_x`, the slack `@example`s, **added for this**:
  before them both were green); the cap off by one 1 (`::test_trig_rev_hull_past_the_cap`); no
  `HullWarning` 7; a wide `x`'s hull taking its low end for its high end 5 and 16 vector items;
  an unbounded hull closed at inf 4 (no vector); the branch limit at 10 (small `x` hulled) 6; no
  exact end (0 enclosed) 4 (no vector: the closed hull hides it); the hull over all of `x`, not per
  piece, 1 (`::test_trig_rev_hull_past_the_cap`'s `[0, 7] ∪ [8, 6300]`, **added for this**: the
  first run was green, since the final `∩ x` hid it for a first piece without a gap); atan(±inf)'s
  sign flipped 10 and 4 vector items; the sign of `f(v)` ignored 14 and 84; acos(-1) not folded
  into k **hung**, killed after 900 s (the new `::test_rounded_inverse_trig_exact_case` line; as at
  M13d, a missed exact case costs minutes before `_MAX_PRECISION`'s `ArithmeticError`). the
  wiring: `sinRevBin` run as `cos_rev` 56 vector items, `tanRevBin` ignoring `x` 28, the 7 loose
  rows dropped 24 (each vector in both passes). **green, as argued**: `rounded_inverse_trig`'s
  extra bits for k removed, 0 red: they save ziv doublings for a branch far out, and ziv doubles the
  precision itself without them
* **part 4 done 2026-09-26 (branch `m13e`): `pow_rev1`, `pow_rev2`** (powRev1 429, powRev2 375
  statements, all in `pow_rev.itl`, all with the domain given; none in `libieeep1788_rev.itl`).
  built:
    * `intervals/reverse.py::pow_rev1(b, c, x=REALS)`, the bases `{t ∈ x : t ** y ∈ c for some y ∈
      b}`, and `::pow_rev2(a, c, y=REALS)`, the exponents `{s ∈ y : t ** s ∈ c for some t ∈ a}`,
      `**` the library's pow (`functions.pow_`, D11) at a point; exported from `intervals`
      (`test_package_exports_unchanged` lists them). no branch engine, as for `mul_rev`: pow is not
      a function of one variable. `::_pow1_preimage` and `::_pow2_preimage` take a case per special
      point of pow (the derivations are in the two docstrings): a base 0 (`0 ** y` = 0 for y in `(0,
      inf]`), 1 (`1 ** y` = 1 for a finite y) or inf (inf for y > 0, 0 for y < 0), an exponent 0
      (`t ** 0` = 1 for a finite t > 0) or ±inf (0 or inf by the side of 1); the rest, a finite
      base in `(0, 1) ∪ (1, inf)` with a finite exponent other than 0, has values in `(0, 1) ∪ (1,
      inf)` and comes from boxes of `c` and the other operand cut at 1 and at 0, monotone in both
      variables, so each box's ends are two corners. the bases are `v ** (1/w)`, run through pow's
      own box rule, `functions._power_box`, on `(v, 1/w)` (`::_reciprocal` inverts a piece of `b`
      exactly, 1/0 the signed infinity), so an end is rounded exactly as `**` rounds `v ** (1/w)`;
      the exponents are `log_t v`, `::_log_box` and `::_log_corner` (the extreme corners by the
      signs of `ln t` and `ln v`, the limits at 0, 1 and inf, never 1 with 1 nor 0 or inf with 0 or
      inf), each end `elementary.rounded('log', v, direction, t)`. `::_reverse` with `given=(b,)`
      does the coercion, class, empty-operand warning and `∩` the domain after rounding
    * `intervals/elementary.py::_exact_log` rewritten, with `::_perfect_power` and `::_primes_below`:
      `log_b x` is rational iff x and b are powers of one rational, found by writing each as `r **
      h` with h as large as it can be (then `log_b x` = h/g iff the two roots agree). both helpers
      were replaced at the review (2026-09-27) by `::_log_ratio` and `::_divide_out`, below
    * **a bug found while building**: `MultiInterval(2).log(4)` hung (killed after 60 s): the old
      `_exact_log` searched only for an int k with `b ** k == x`, so `log_4 2` = 1/2 was taken as
      irrational and ziv's loop never settled. every `powRev2` vector's ends are rational (the file
      picks operands whose binary logs are exact), so `pow_rev2` needs the fix; pinned by
      `tests/test_elementary.py::test_exact_log_to_a_base_is_rational_where_it_is` (13 cases) and
      `::test_log_to_a_base_at_a_rational_value` (300 random powers of one root, and the two
      `MultiInterval` calls)
    * **choices the plan left open** (conservative, flagged in `v2-plan.md` "2026-09-26 revision:
      M13e, fourth part"):
        * **the library's own pow defines them**, ±inf as points, so `pow_rev1([-2], [0, 1])` is
          `[1, inf]` (`inf ** -2` = 0) and the points with no value (`0 ** y` for y <= 0, `1 **
          ±inf`, `inf ** 0`, a negative base) solve nothing; no 1788 vector sees it, since each gives
          the domain through the input rule and no operand has an infinite point
        * the operand order is 1788's, the domain last, named `x` for `pow_rev1` and `y` for
          `pow_rev2` (the plan's names), defaulting to `[-inf, inf]`
        * **the float rule is pow's, per operand** (any finite float end of either operand makes
          every end float), not part 1's per end of `c`: "built on the library's own pow". so
          `pow_rev1([n], c)` is `pown_rev(c, n)` on `[0, inf]` exactly when `c`'s ends are all
          exact or all float (`tests/test_pow_rev.py::test_pow_rev1_by_an_int_is_pown_rev_on_the_bases`
          draws only those)
        * to nearest, an end past the largest double rounds to inf and stays closed, the library's
          nearest rule (`MultiInterval(1e200) ** MultiInterval(1000.0)` is `[inf]`, and so is
          `pow_rev1([1e-3], [1e200])`); `OutwardMultiInterval` keeps it open. the float test widens
          through ±inf for it (`::widened`)
        * the class, the coercion, the warnings (only an empty operand warns) and `∩` the domain
          after rounding are part 1's
    * adapter (`tests/itf1788/test_itf1788.py`): `powRev1`, `powRev2` in `OPS`, one labelled
      block, the calls with the domain given. rows: `::_POW_REV_LOOSE_ROWS` (2 keys, reason
      `::_POW_REV_LOOSE`) under the **proposed category "tighter than the vector"** of part 1, not
      yet approved by the owner. no other row, no new category
    * **a finding: two 1788 vectors are far from tight.** `powRev2 [0.25, 0.5] [2.0, infinity]
      [entire] = [entire]` (`pow_rev.itl:609`) and `powRev2 [0.25, 1.0] [2.0, infinity] [entire] =
      [-infinity, 0.0]` (`:642`): for t in `[1/4, 1)`, `t ** s >= 2` iff `s <= log_t 2 <= -1/2`
      (`(1/4) ** -1/2` is exactly 2; `1 ** s` is never 2), so the tightest hull is `[-inf, -0.5]`,
      which ours is; the vectors just above them with `c = [2, 4]` (`:608`, `:640`) answer -0.5 at
      that end. decided exactly, no rounding, by `tests/test_pow_rev.py::test_pow_rev2_is_tighter_than_the_vector`
    * tests (`tests/test_pow_rev.py`, a new module, 50 items in 38.6 s under the shared laptop's
      load, 2026-09-27): 11 `@given`, all
      `deadline=None`. the oracle is written from the definitions (`::special`, `::cmp_value`,
      `::fits1`, `::fits2`): the other operand's pieces map monotonically onto intervals whose ends
      are compared with `c`'s, `t ** (p/q)` against `v` exactly as `t ** p` against `v ** q` where
      that is small, else by arb at a growing precision (`::cmp_pow`).
      `::test_pow_rev1_is_exactly_the_points_that_fit`, `::test_pow_rev2_is_exactly_the_points_that_fit`
      (150 examples each, exact non-empty operands; at the ends of the operands and of the result,
      one double either side of every float end, a float approximation of every `v ** (1/w)` or
      `log_t v` of the ends with its neighbours, a point between each two, one beyond, ±inf, -1, 0,
      1: a point fitting is in the result, a point in it fits or lies in a rounded end's slack, the
      double inward from a rounded end therefore fitting), `::test_pow_rev1_the_largest_set`,
      `::test_pow_rev2_the_largest_set` (`T ⊆ pow_rev1(B, T ** B ∪ more)` with the library's pow on
      sets, in the outward class, the t with `{t} ** B` empty aside; likewise `S`),
      `::test_pow_rev_isotone_and_distributive` (both ops: isotone in each operand, distributing
      over unions of each, the domain only intersecting), `::test_pow_rev_symmetry` (`pow_rev1(-B,
      1/C)` is `pow_rev1(B, C)` but at 0; `pow_rev2(A, 1/C)` is `-pow_rev2(A, C)` for A without 0),
      `::test_pow_rev1_by_an_int_is_pown_rev_on_the_bases` (both classes, rounding included),
      `::test_pow_rev2_of_a_point_is_the_log` (`pow_rev2([t], c)` is `c.log(t)`, both classes),
      `::test_pow_rev_float_operands` (outward holds the exact result of the same doubles, adds no
      double strictly inside, closes only exact points; nearest within one double, inside the
      outward closure, not empty when the exact result is not; then the domain only intersects),
      `::test_pow_rev_sound_at_sampled_points` (the library's `{t} ** B` and `A ** {s}` on sampled
      points, float operands and ±inf included: a point whose image surely meets `c` is in the exact
      and the outward results, a point of the exact result has an image meeting `c` or lies in a
      slack). `@example`s: `pow_rev.itl:35`, `:45`, `:47`, `:61`, `:86`, `:96`, `:107`, `:173`,
      `:504`, `:544`, `:554`, `:559`, `:573`, `:591`, `:608`, `:609`, `:642`, the special points (`[inf]`,
      `[-inf]`, `[0]` against `[0]`, `[1]`, `[0, inf]`), `log_3 2`, overflow past max float and
      underflow below the least subnormal, `log_8 2` = 1/3 from float operands and a squeezed
      piece (the last two added for sabotage, below). plus 37 parametrized examples,
      `::test_irrational_ends_are_open_one_ulp_enclosures`, `::test_class_coercion_and_warnings`,
      `::test_pow_rev2_is_tighter_than_the_vector`
* evidence, measured 2026-09-26 (`tools/itf1788_census.py`): the 804 vectors of the two ops (all in
  `pow_rev.itl`, all interval-valued, none with `[nai]`): 802 match in both passes; 2 (2 keys) are
  rows under the proposed category. all ops: 9269 vectors of 102 ops, 8256 interval-valued, 157
  divergence keys (36 degenerate infinities, 11 proposed, 58 decoration expectations), 0 unknown
  failures; skipped 273 statements of 9 ops, all M13g's (`b-textToInterval` and `d-textToInterval` 91
  each, `setDec` 22, `isNaI` 16, `intervalPart` 15, `newDec` 13, `b-numsToInterval` 10,
  `d-numsToInterval` 9, `decorationPart` 6). the gate, as two runs on 2026-09-27: `tests/itf1788`
  17911 passed in 68.7 s, the rest 3535 passed in 676.9 s under the shared laptop's load (21446 in
  all; 19771 at part 3)
* sabotage (section 2), 2026-09-26/27: 40 breaks by a throwaway harness, each file restored from a
  copy and byte-compared (`filecmp`, all equal), the hypothesis database cleared before every run
  (so no replayed failure of an earlier break counts), results appended as they landed. the
  library breaks run against `tests/test_pow_rev.py`, the log tests of `tests/test_elementary.py`
  (`-k 'pow_rev or log'`, 107 items) and the doctests of `reverse.py` and `elementary.py`, and
  against every itf1788 item matching `pow_rev` (1610); counts as lib / vector items. pow_rev1: `t
  = 0` for any finite y 7 / 40; `t = 0` dropped 5 / 12; `t = 1` dropped 7 / 12; `t = 1` by `y =
  ±inf` too 1 / 0 at first (the defining property's `@example`), then 3 with a pinning example
  (`pow_rev1([inf], [1])` is `∅`); `inf ** y` = 0 by y > 0 8 / 0; `t = inf` dropped 9 / 0; `y = 0`
  dropped 5 / 118; `y = 0` without `1 ∈ c` 6 / 138; `y = ±inf` with the sides of 1 swapped 5 / 0;
  `y = ±inf` dropped 6 / 0; `1/y` with its ends not swapped 9 / 518; `1/0` of the wrong sign 8 / 530;
  the float rule from `c` only 1 / 0 (`::test_irrational_ends_are_open_one_ulp_enclosures`); outward
  ignored 3 / 35; the main part dropped 15 / 296. pow_rev2: `s = 0` dropped 6 / 26; `s = 0` by `t =
  0` and inf too 4 / 2; `0 ** inf` dropped 4 / 0; `0 ** -inf` taken as inf 1 / 0; `t = 0` dropped 5 /
  18; `t = 1` dropped 4 / 124; `_log_box`'s u end chosen wrong 14 / 478, its r end 11 / 204; the
  corner at `r = 1` with its sign flipped 9 / 498, at `u = 0` 7 / 44; `log_r 1` taken as 1 4 / 30; an
  irrational end kept closed 4 / 0; outward ignored 2 / 0; the float rule from `c` only 1 / 0. the
  rational log: the fraction inverted 35 / 36; unrelated roots taken as one 22 / 0; one root per
  prime in `_perfect_power` 3 / 0; ints only, the old behaviour, **hung** in both runs (killed after
  600 s: every `powRev2` vector's ends are rational). the wiring: `powRev1` with `b` and `c` swapped
  418, `powRev2` run as `pow_rev1` 638, `powRev1` ignoring `x` 396, the 2 loose rows dropped 4.
  **no vector sees** any clause at an infinite point (the input rule gives no operand one) nor the
  float rule, the outward rule of the logs or a closed irrational end (the closed hull hides a
  flag): the properties hold those. **first green, now pinned**: `inf ** s` with its signs swapped
  (0 red: `[inf]` against `[0, inf]` gives the same set either way; now 2, `pow_rev2([inf], [inf])`
  is `(0, inf]` and against `[0]` it is `[-inf, 0)`), a moved rational log end kept closed outward
  (0 red: hypothesis never drew a rational non-double log from float operands; now 1, `log_8 2` from
  `[8.0]` and `[2.0]`), and `_log_box`'s squeeze rule removed (0 red: no drawn open piece of `c` was
  narrow enough; now 1, `log_3` of `(1e300, 1e300 + ulp)` is one double to nearest)
* **the close-out, done 2026-09-26 (branch `m13e`): M13e is complete**, all 1955 statements of the
  19 reverse ops run and none is skipped. built, over the four parts:
    * `intervals/reverse.py`: ten functions, exported from `intervals`, the constraint first and the
      domain last, defaulting to `[-inf, inf]`: `::sqr_rev`, `::abs_rev`, `::pown_rev`, `::cosh_rev`
      (part 1), `::mul_rev` (part 2), `::sin_rev`, `::cos_rev`, `::tan_rev` (part 3), `::pow_rev1`,
      `::pow_rev2` (part 4). the one-variable ops share the branch engine (`::Branch`,
      `::branch_preimage`, `::_trig_branch` for the periodic ones), the two-variable ones take a case
      per special point; all go through `::_reverse` (coercion, class, the empty-operand warning,
      `∩` the domain after rounding). two library helpers came with them:
      `intervals/elementary.py::rounded_inverse_trig` (part 3) and the rational `::_exact_log`
      (part 4, fixing a hang of `MultiInterval(2).log(4)`)
    * the close-out: the module docstring no longer calls the engine's later users "still to come";
      three reverse-op lines in the README's doctest block (`sqr_rev`, `mul_rev`, `sin_rev` over a
      bounded `x`, no warning); and a pin, below
* **choices the plan left open**: each part's, listed in its record above and in `v2-plan.md`'s four
  "2026-09-26 revision: M13e" entries; the close-out made none. still awaiting the owner: the
  category **"tighter than the vector", PROPOSED at part 1**, now 11 keys (18 vectors: `pownRev` 4,
  `sinRevBin` 2, `cosRevBin` 6, `tanRevBin` 4, `powRev2` 2), each where 1788's hull is looser than
  the tightest one, which ours is (arb, or exactly for `powRev2`)
* adapter (`tests/itf1788/test_itf1788.py`): the 19 ops in `OPS` in four labelled blocks, the pair
  rule for `mulRevToPair` (`::PAIRS`), and the rows `::_POWN_REV_ROWS`, `::_POWN_REV_LOOSE_ROWS`,
  `::_TRIG_REV_LOOSE_ROWS`, `::_POW_REV_LOOSE_ROWS`. the close-out adds
  `::test_only_m13g_ops_are_skipped` (with `::_REVERSE_OPS`, `::_M13G_OPS`): every reverse op is in
  `OPS` and every skipped statement belongs to one of M13g's 9 ops, so an op dropped from `OPS`, or
  a file gaining one, cannot quietly add skips before M13's exit asserts `SKIPPED` empty
* tests, measured 2026-09-26: `tests/test_reverse.py` 112 items (24 `@given`: 9 of part 1, 7 of
  part 2, 8 of part 3) and `tests/test_pow_rev.py` 50 (10 `@given`; part 4's record says 11, a
  miscount: the ten it lists are all there are), 162 passed in 122.0 s under the shared laptop's
  load; plus part 4's two rational-log tests in `tests/test_elementary.py`
* evidence, measured 2026-09-26 (`tools/itf1788_census.py`, and a throwaway per-op count running
  each vector through `::run` and `::run_outward`): the 1955 vectors of the 19 ops, all
  interval-valued, 0 unknown failures, 0 stale rows: 1879 match in both passes; 52 (26 keys, all
  unary `pownRev`) are degenerate infinities; 18 (11 keys) the proposed category; 6 (6 keys, 3
  `mulRev` and 3 `mulRevToPair`) have a `[nai]` operand, generated rows under "decoration
  expectations" that M13g moves. by op: `sqrRev` 20, `sqrRevBin` 22, `absRev` 18, `absRevBin` 38,
  `pownRev` 285, `pownRevBin` 73, `coshRev` 10, `coshRevBin` 10, `sinRev` 12, `sinRevBin` 40,
  `cosRev` 12, `cosRevBin` 42, `tanRev` 10, `tanRevBin` 20, `mulRev` 182, `mulRevTen` 10,
  `mulRevToPair` 347, `powRev1` 429, `powRev2` 375. all ops: 19 files, 9269 vectors of 102 ops,
  8256 interval-valued (17525 vector test items), 157 divergence keys (36 degenerate infinities, 5
  cut-based relations, 47 cancellations, 11 proposed, 58 decoration expectations), 0 unknown
  failures; skipped 273 statements of 9 ops, all M13g's. the gate, as two runs on 2026-09-26: `tests/itf1788` 17912
  passed in 67.4 s, the rest 3535 passed in 667.7 s under the shared laptop's load (21447 in all;
  21446 at part 4, the one more being the pin). **not updated here**:
  the shared totals in the README's "ieee 1788" bullet and `v2-plan.md`'s residual-divergence
  paragraph still carry M13d's (7314 vectors, 2228 skipped), left for the merge with the parallel
  M13g branch, as parts 1 to 4 left them
* sabotage (section 2): the four parts ran 22, 18, 24 and 40 breaks (104), recorded above. the
  close-out, 2026-09-26, by a throwaway harness, each file restored from a copy and `cmp`-checked
  against a reference copy (equal): `powRev1` dropped from `OPS` 1 red and `mid` dropped from `OPS`
  1 red, both only `::test_only_m13g_ops_are_skipped` (**0 red without it**: `test_parser_drops_nothing`
  counts a dropped op's statements as skips, and `test_every_op_has_vectors` checks `OPS` against
  the vectors, not the skips); the README's `sqr_rev` line expecting 1788's hull `[-2, 2]` 1 red, its
  `sin_rev` line with pi closed 1 red (the README doctest)
* review (2026-09-27, three read-only reviewers over c6a3b3d: math against an independent oracle,
  sabotage, spec), each finding reproduced on `m13e` before acting. **one library bug, a
  performance regression**: part 4's `_exact_log` took a root for every prime below the operand's
  bit length, so a large exact operand that is not a perfect power was very slow (measured
  2026-09-27 at c6a3b3d: `MultiInterval(3 ** 5000 + 1).log2()` 1.51 s, `.log10()` 1.69 s,
  `pow_rev2(M(2), M(3 ** 5000 + 1))` 3.08 s; the reviewers had 356 s for `3 ** 20000 + 1`). fixed:
  `intervals/elementary.py::_exact_log` now reduces to the numerators and the denominators, each by
  `::_log_ratio`, euclid on the exponents (`a = b ** t * a'`, `log_b a = t + 1 / log_a' b`, t by
  `::_divide_out` in binary); `_perfect_power` and `_primes_below` are gone. the same cases now
  0.00 s, `3 ** 20000 + 1` 0.01 s, `2 ** 60000 + 1` 0.02 s; on 19859 random operands the value and
  its type agree with c6a3b3d's. (`MultiInterval(10 ** 6000 + 1).log10()` still takes 27 s: ziv's
  loop at that size, not the exact test, and not new.) pinned by
  `tests/test_elementary.py::test_rational_log_of_large_operands_is_fast` (7 cases, 5 s each).
  **three unpinned clauses, now pinned**: to nearest, a moved end keeps its flag in
  `intervals/reverse.py::_end` (pinned by `tests/test_reverse.py::test_nearest_keeps_the_flag_of_a_moved_end`:
  `pown_rev(c, -1)` is `c ** -1`, closed, for 3 float c) and in `::_log_corner` (by
  `tests/test_pow_rev.py::test_pow_rev2_nearest_keeps_the_flag_of_a_moved_log`: `pow_rev2([a], c)`
  is `c.log(a)`, closed, 4 cases); D12's count running across the pieces of x in
  `::_periodic_preimage` (a case added to `tests/test_reverse.py::test_trig_rev_hull_past_the_cap`:
  `[0, 3000] ∪ [3001, 6300]` has 478 then 525 pieces, so the second piece is hulled). **one
  unpinned adapter clause, now pinned**: `INTERVAL_VECTORS`' `or v.op in PAIRS`
  (`tests/itf1788/test_itf1788.py::test_the_pair_vectors_run_outward`). sabotage, 2026-09-27, each
  file restored from a copy and `cmp`-checked (equal), the hypothesis database cleared first:
  the flag in `_end` dropped 3 red (0 before), in `_log_corner` 4 red (0 before), the running count
  dropped 1 red (0 before), `or v.op in PAIRS` dropped 1 red (0 before; 17565 passed, 347 items fewer); the
  rational log: c6a3b3d's `elementary.py` 1 red (5.4 s on `3 ** 10000 + 1`), the denominators' check
  dropped 4 red, `_divide_out` adding 1 per bit 33, the continued fraction folded without `1 / e`
  20, a swap not recorded 17, a denominator of 1 on one side accepted **hangs** (`_divide_out` by
  1; the guard keeps `_log_ratio`'s operands at 2 or more), `_divide_out`'s powers stopping one
  square short 0 red, an equivalent mutant (a leftover factor gives a 0 quotient, and `[..., t, 0,
  t', ...]` folds to `[..., t + t', ...]`). **docs**: the README's "ieee 1788" bullet and a
  "counts at M13e" bullet in `v2-plan.md` "ieee 1788" now carry this branch's census (they said
  the reverse ops were not built; the every-sub-task rule asks for them, and the merge with M13g
  re-measures them again); the README's layout names `reverse`; the spec line on irrational ends
  now says it is the rule for exact operands. **not changed**: the offsets `Fraction(-1, 2)` of
  `_SIN` and `_TAN` are an equivalent mutant (the one-branch padding absorbs a half-period shift),
  and `_operands`' `not isinstance(a, bool)` is redundant with `MultiInterval(True)` raising, left
  as a guard; the category "tighter than the vector" has its line in `v2-plan.md` "ieee 1788" (the
  reverse-op bullets) but is still PROPOSED, awaiting the owner; the M13 heading's status line,
  `HANDOFF.md` and the two skip pins (`::_REVERSE_OPS` is defined on both branches, with the same
  value) are the merge's. part 4's "11 `@given`" is a miscount the close-out already notes (10), and
  the close-out's measurements dated 2026-09-26 were taken after part 4's, on 2026-09-27 (+0800).
  the census after the fix (2026-09-27) is unchanged from the close-out's (9269 vectors, 157 keys,
  273 skipped). the gate, two runs on 2026-09-27: `tests/itf1788` 17913 passed in 47.3 s (one more,
  the pair pin), the rest 3543 passed in 464.6 s (8 more: the rational-log pin and the 7 flag cases)
* the reviewers' clean results, transcribed 2026-09-27 from their notes (gitignored, since deleted):
  the math lens's independent oracle (exact Fractions and python-flint arb, membership decided from
  the definitions with the library's forward ops) found no wrong result: random multi-piece `c`,
  `x`, `b` with 0, ±1, ±inf and rationals, exact operands (sqr 300, abs 300, pown 300 with n in
  -8..7, cosh 300, sin/cos/tan 150 each, mul 2000, pow_rev1 600, pow_rev2 600): 0 unsound, 0 not
  tight to the double, 0 wrong exact ends, 0 stray warnings; trig near k pi/2 (k up to 10**300/7),
  1536 cases: 0 problems; `OutwardMultiInterval` with float operands, all 10 ops: 0 problems (tan
  on 40 seeds only, the oracle being slow there); `MultiInterval` to nearest: no member farther than
  2 ulp from an end. the slow rational log grew about as the cube of the size in bits (5000 digits
  2.7 s, 10000 44 s, 20000 356 s, at c6a3b3d). the sabotage lens ran 30 mutations of its own
  beyond those above, all red (e.g. `_power_box`'s sv/sw swapped: 1 doctest, 6 library tests, 208
  vector items; the pair rule keeping 1788's empty second interval: 624 vector items; each `*Bin`
  adapter entry ignoring `x`: 22 to 378 vector items), and reproduced the builders' recorded counts
  exactly (416, 518, 2, 2, 1). one mutant, `c` not cut to sin's or tan's image, hangs the trig tests
  (900 s timeout) besides turning 2 doctests red

**M13f cancellation (done 2026-09-26)** (D13). `cancelPlus` 116, `cancelMinus` 126
* `A.cancel_minus(B)`: the largest `X` with `B + X ⊆ A` (the Minkowski difference);
  `A.cancel_plus(B)` is `A.cancel_minus(-B)`. for connected operands with `wid A ≥ wid B` it is
  `[a1 - b1, a2 - b2]`, 1788's answer
* where 1788 returns entire as "no answer" (A narrower than B, an unbounded operand) ours is a real
  set, often `∅`: those rows get the new residual category **"cancellation as a Minkowski
  difference"**, added to `REASONS` and to `v2-plan.md` "ieee 1788" when this lands
* done 2026-09-26. built:
    * `intervals/ops.py::cancel_minus`, `::cancel_plus` over cut tuples (helpers `::_fitting`,
      `::_piece_fits`), and the methods `MultiInterval.cancel_minus(B)`, `.cancel_plus(B)` (a
      number is coerced to a point, anything else is a `TypeError`; the result has the receiver's
      class, as `fma`). the derivation is in `cancel_minus`'s docstring: `+` is the set of values
      of the defined pairs, so the largest `X` is `{x : {x} + B ⊆ A}`. a finite `x` needs `B`'s
      infinite points in `A`, and each piece `q` of `B`'s reals shifted into one piece `p` of
      `A`'s reals (the connected components), which is `[p1 - q1, p2 - q2]` with an end closed
      unless `p` is open there and `q` closed, an unbounded side of `q` needing the same of `p`:
      an intersection over the `q` of a union over the `p`. `inf` fits iff `inf ∈ A` or
      `B = [-inf]` (`inf + -inf` has no value), `-inf` mirrored; an empty `B` gives `[-inf, inf]`
    * **choices the plan left open** (conservative, flagged in the decision log):
        * **rounding outward, not inward.** the task text suggested inward rounding, to keep
          `B + X ⊆ A` for floats; 1788 and its vectors round outward (`cancel.itl:218`: `[0x1.FFFFFFFFFFFFP+0]`
          minus the double 0.1 is the two doubles around the difference; `:221`: `[max]` minus
          `[-max]` is `[max, infinity]`), the plan requires the outward pass to match, and
          `OutwardMultiInterval` promises an enclosure of the exact result. so the difference is
          computed exactly and rounded once like `fma`: outward in `OutwardMultiInterval` (every
          `x` that fits is in it; `B + X ⊆ A` can fail by an ulp at a moved end), to nearest in
          `MultiInterval`. a certified inner answer comes from exact operands (`Fraction(f)`);
          an inward variant: open, `HANDOFF.md` Q4
        * `cancel_minus(∅, ∅)` is `[-inf, inf]`, not 1788's `∅`: every `X` fits the empty `B`, and
          "the largest `X`" leaves no choice. the two `[empty] [empty]` vectors are rows under the
          new category with their own reason (`test_itf1788.py::_CANCEL_EMPTY`)
        * the infinite points follow the library's `+`, so `[0, 1].cancel_minus([-inf])` is `[inf]`
          (`{inf} + [-inf]` is empty and fits vacuously); no warning is emitted for any operand,
          and `cancel_plus` negates a non-empty `B` only, since `neg(∅)` warns
    * adapter (`tests/itf1788/test_itf1788.py`): `cancelMinus`, `cancelPlus` in `OPS`; "cancellation
      as a Minkowski difference" in `REASONS`; rows `::_CANCELLATION_ROWS` (45 keys, reason
      `::_CANCEL`, every vector where 1788 answers entire and ours is not the whole line) and the
      two `[empty] [empty]` keys (`::_CANCEL_EMPTY`)
    * tests (`tests/test_cancel.py`, 42 items in 24.8 s, 2026-09-26): 10 `@given`: the defining
      property decided completely on exact operands, `x ∈ X` iff `{x} + B ⊆ A` with the library's
      `+` at every difference of an end of `A` and one of `B`, a point between each two, one
      beyond each end and ±inf (soundness and maximality at once; twice, the second with short
      `B`s against many-piece `A`s, where `X` has several pieces in about 10% of examples); `B + X
      ⊆ A` at set level; `C ⊆ (B + C).cancel_minus(B)`, equal for connected closed bounded `C`;
      isotone in `A`, antitone in `B`; a point `B` is subtraction; 1788's formula on connected
      closed operands; `cancel_plus(B)` = `cancel_minus(-B)` in both classes; float operands
      (outward encloses the exact `X` of the same doubles and adds no double strictly inside,
      nearest is `X` rounded once); soundness at sampled points (every sampled `x` of the exact
      `X` fits and is in the outward result). `@example`s: `cancel.itl:166`, `:167`, `:177`,
      `:183`, `:196`, `:204`, `:218`, `:219`, `:221`, `:223`, `:229`, `:231`, `:234`, `:235`, `:28`,
      `:48`, and the infinite-point cases (`[inf]` minus `[-inf]`, `[0, 1]` minus `[-inf]` and
      minus `[0] | [inf]`); plus 29 parametrized examples with open ends, several pieces and ±inf
* evidence, measured 2026-09-26 (census by importing the test module): 242 vectors of the two ops
  (`cancelPlus` 116, `cancelMinus` 126, all in `libieeep1788_cancel.itl`, all interval-valued): 148
  match in both passes, the 16 that need rounding (outward differs from nearest) included; 94
  (47 keys) are rows under the new category, 45 keys where 1788 answers entire as no answer
  (ours `∅` or a ray) and
  the 2 `[empty] [empty]`; no other category was needed. all ops: 5375 vectors of 69 ops, 4362
  interval-valued, 167 numeric run twice more with floats, 10071 vector test items, 114
  divergence keys (47 cancellation, 52 decoration expectations), 0 unknown failures; skipped
  4167 statements of 42 ops. the gate: 13000 passed in 450 s
* sabotage (section 2), each red, then the file restored from a copy and `cmp`-checked; targeted
  runs (`tests/test_cancel.py`, every itf1788 cancel vector, the doctests of `ops.py` and
  `multi_interval.py`): the start's side rule swapped (open against open missing, open against closed fitting) 11 red (the
  defining property both ways, `B + X ⊆ A`, the cancellation, subtraction and sampled-soundness
  properties, 5 examples); the end's 9 (the same kind, and the `ops.py` doctest); `p1 + q1` for
  `p1 - q1` 198 (every property, 22 non-vector items, 176 vector items, `cancel.itl:63` among them); `B`'s
  infinite points ignored for a finite `x` 6 (the defining property, `B + X ⊆ A`, sampled
  soundness, 3 examples); the vacuous `inf` for `B = [-inf]` dropped 4 (the defining property,
  the cancellation property, 2 examples; before those examples were added only the cancellation
  property saw it); an empty `B` giving `∅` 63 (56 vector items, the `[empty] [empty]` rows red
  as stale among them); an unbounded `q` accepted in a bounded `p` 5 (the defining property,
  isotonicity, sampled soundness, 2 examples); outward rounded to nearest 20 (the float and
  sampled-soundness properties, `test_1788_float_vectors`, the class doctest, 16 outward vector
  items, `cancel.itl:218`, `:221`, `:234` among them); `cancel_plus` without the negation 104. **no
  vector sees the side rules, the infinite points or an unbounded `q` in a bounded `p`**: the
  vectors' finite ends are closed and their unbounded ones rows, which only assert that ours
  differs; these are held by the properties alone. the wiring: `cancelMinus` run as
  `cancel_plus` 100 red; the 45 rows dropped 180 (each key's 2 vectors in both passes)
* review (2026-09-26, a read-only reviewer over the four sub-task commits): no library bug. one
  unpinned clause: the vacuous `-inf` for `B = [inf]` (`ops.py::_fitting`, `or b == _PLUS_INF`)
  could be dropped with all 42 cancel tests green, even under the fuzz profile at ×10, since the
  strategy almost never draws `[inf]` alone and `cancel_plus` routes both sides through the same
  clause. now pinned by a `test_examples` row and an `@example` on
  `tests/test_cancel.py::test_exactly_the_points_that_fit`; dropping the clause turns 2 red

**M13g decorations, constructors and signals (done 2026-09-26)** (D16; no NaI, owner 2026-09-26). 273 statements,
plus a decoration check on every decorated vector
* a decorated wrapper type around a `MultiInterval` (name chosen when built, recorded in the
  decision log): a decoration per 1788's com/dac/def/trv, propagated through every op the core
  has. the core class stays undecorated
* **no NaI and no `ill`** (owner 2026-09-26): once invalid input raises, nothing in the library
  makes NaI, and its remaining uses (a per-element "invalid" in batch work, reading another
  1788 library's `[nai]`) come with numpy or data import, if ever; add it back then. every
  statement that needs a NaI (`isNaI` 16, a `[nai]` operand, a `d-` constructor or `setDec`
  answering `[nai]`, `intervalPart [nai]`, text `"[nai]"`) is a row under a new divergence
  category, **"no NaI: invalid input raises"** (approved with the decision), added to `REASONS`
  and to `v2-plan.md` "ieee 1788" when built; the 52 generated `[nai]` rows under "decoration
  expectations" move to it. a vector whose expected `[nai]` comes with `signal
  UndefinedOperation` instead passes when the constructor raises (below)
* 1788 text and number constructors: `b-textToInterval` 91 and `b-numsToInterval` 10 give a bare
  `MultiInterval` (1788's input rule: an infinite end is open), `d-textToInterval` 91 and
  `d-numsToInterval` 9 the wrapper. a parser for 1788's text syntax, separate from
  `MultiInterval.parse`, whose syntax is ours. `setDec` 22, `newDec` 13, `intervalPart` 15,
  `decorationPart` 6 on the wrapper (`isNaI` 16: rows, above)
* the adapter stops dropping decorations: a decorated vector runs through the wrapper and its
  expected decoration is checked
* 1788's signals (`UndefinedOperation`, `PossiblyUndefinedOperation`, `IntvlPartOfNaI`, from
  `ieee1788-exceptions.itl` and the constructors' `signal` clauses), owner 2026-09-26:
  `UndefinedOperation` raises a `ValueError` subclass, so a 1788 constructor given invalid input
  stops (as `MultiInterval(2, 1)` does); `IntvlPartOfNaI` would raise too, but with no NaI it
  cannot arise;
  `PossiblyUndefinedOperation` is an `IntervalWarning` subclass and the result is returned. the
  adapter takes the raised error as a vector's expected result with `signal UndefinedOperation`
  (the `b-` flavour's `[empty]`, the `d-` flavour's `[nai]`), as the reduction rule takes
  `ValueError` as `NaN`
* **part 1 built 2026-09-26 (branch `m13g`): the signals, the 1788 text parser, the
  bare constructors, the new category.** left for parts 2 and 3 then (both built, below): the decorated wrapper type,
  `d-textToInterval`, `d-numsToInterval`, `setDec`, `newDec`, `intervalPart`, `decorationPart`, and
  the adapter checking decorations. built:
    * signals (`intervals/errors.py`): `UndefinedOperationError(ValueError)` and
      `PossiblyUndefinedOperationWarning(IntervalWarning)`, exported from `intervals`
      (`tests/test_applicator.py::test_package_exports_unchanged` lists them)
    * `intervals/literals.py`, a new module (not `ieee1788.py`, which `HANDOFF.md` H3 keeps for a
      later thin adapter): `::parse_literal` reads 1788's interval literal (1788-2015 §9.7) into a
      `::Literal` (exact `lo`, `hi`, `None` for empty, and the decoration if any); `::number` one
      number literal; `::text_to_interval` (1788's `b-textToInterval`) and `::nums_to_interval`
      (`b-numsToInterval`) give a bare `MultiInterval` under 1788's input rule (a finite end closed,
      an infinite one open, `::_bare`), both exported from `intervals`. the grammar, from the
      vectors of `libieeep1788_class.itl`, `ieee1788-constructors.itl`, `ieee1788-exceptions.itl`:
      `[l, u]`, `[x]`, `[l,]`, `[,u]`, `[,]`, `[ ]`, `[empty]`, `[entire]`; the uncertain form
      `m?r`, `m?`, `m??` with a direction `u`/`d` and an exponent `e`; decimal, hex (`0x...p...`),
      `p/q` and `inf`/`infinity` numbers; decorations `_com` `_dac` `_def` `_trv`; any letter case.
      invalid input, `[nai]` included, raises `UndefinedOperationError`
    * **choices the plan left open** (conservative):
        * names: 1788's with python's suffix, `UndefinedOperationError` and
          `PossiblyUndefinedOperationWarning`; the functions `text_to_interval`, `nums_to_interval`
          (1788's names in python's style, module-level as the reductions are, not methods beside
          our own `MultiInterval.parse`). the warning gets no import-time `'ignore'` filter, so it
          is shown, like `IndeterminateResultWarning`
        * **exact values**: a literal's numbers are the rationals they spell (`[0.1]` is the point
          `1/10`, `[1.0E+400]` is `10**400`), never rounded, as int and Fraction are exact everywhere
          in the package (D3). 1788's binary64 enclosure is what the adapter's precision rule makes
          of it. so validity is decided exactly too: `[1.0000000000000002,1.0000000000000001]` is
          invalid and raises, `[1.0000000000000001, 1.0000000000000002]` is valid and does not
          warn; `PossiblyUndefinedOperationWarning` is never emitted today (the owner foresaw it,
          `v2-plan.md` "2026-09-26 revision: owner answers", Q1)
        * the result is always a `MultiInterval`: no class argument, since an exact result has
          nothing for `OutwardMultiInterval` to round; the constructors are not in the outward pass
        * strict syntax: no white space outside the brackets, inside a number or anywhere in the
          uncertain form (`" [1, 2]"` raises); ASCII white space inside the brackets; a hex number
          needs its `p` exponent (IEEE 754's hex form); `1.` and `.5` are numbers; the sign of `p/q`
          goes before `p` only. a non-str argument is a `TypeError`, as are a `bool` or a non-real
          bound for `nums_to_interval`
        * a decoration is checked against the exact value: `com` needs a bounded non-empty
          interval, the empty set takes `trv` only, `ill` and an unknown word never fit (the
          vectors' `[ Empty ]_ill`, `[,]_com`, `_fooo`, `_da`). `[1.0E+400 ]_com` fits here
          (bounded as a rational) where 1788's binary64 hull is unbounded and gets `dac`
          (`libieeep1788_class.itl:165`): the decorated type's question. the bare constructor
          refuses every decorated literal, as 1788's does (`class.itl:50`)
        * `nums_to_interval(-inf, -inf)` and `(inf, inf)` raise, as 1788 says, although `[-inf]` is
          a legal point of `MultiInterval`; `nan` raises `UndefinedOperationError`, not the plain
          `ValueError` of `MultiInterval(nan)`
        * no limit on an exponent's size: `1e999999999` builds the exact power, slowly, as
          `Fraction('1e999999999')` does; the fuzz property assumes away exponents of 4 digits or
          more
        * `isNaI` is in `OPS` as an op with no counterpart (all 16 vectors rows), not as
          `lambda a: False`, which would match 15 of them with an op the package does not have
    * adapter (`tests/itf1788/test_itf1788.py`): `b-textToInterval`, `b-numsToInterval`, `isNaI`
      in `OPS`; the **signal rule** (`::SIGNALLED`, `::_signalled`: (closed hull, signal) pairs,
      a raise read as `[empty]` with `UndefinedOperation`, the warning as
      `PossiblyUndefinedOperation`), `::test_signals_are_checked` (every op whose vectors carry a
      signal is in `SIGNALLED`, so `d-textToInterval`, `setDec`, `intervalPart` join it when
      built), `::test_signalled_reads_both_signals` (the warning's branch, which no vector reaches);
      `REASONS` gains "no NaI: invalid input raises" (D16) and, **PROPOSED, not approved: an owner
      question**, "exact parsing decides validity" for the 4 `b-textToInterval` vectors that expect
      `PossiblyUndefinedOperation` (`::_EXACT_VALID`, `::_EXACT_INVALID`); `::_NAI` moved to the
      new category and `::_NO_IS_NAI` added. `tools/itf1788_census.py` matches a reason to its
      `REASONS` entry, since the new one holds a colon
    * tests (`tests/test_literals.py`, 122 items in 1.5 s, 2026-09-26): 8 `@given`:
      `nums_to_interval` against 1788's definition at probe points (raises iff invalid, else exactly
      the reals between the bounds: soundness and maximality); every spelling of a value (a
      float's exact decimal expansion, `float.hex`, `p/q`, `inf`) reads back exactly, through `[x]`
      and `[l, u]`, with an empty side as an infinity; the uncertain form against a
      `decimal.Decimal` oracle, with soundness (the midpoint and the radius's ends inside) and
      maximality (a tenth of a unit past an end outside); white space and case inside the brackets
      do not matter and outside they do; decorations read iff they fit and always refused by the
      bare constructor; any text over the literal alphabet is one interval under the input rule or
      raises `UndefinedOperationError`, nothing else. `@example`s: `class.itl:30`-`33`, `:50`,
      `:120`, `:123`, `:125`, `:127`, `:165`, `:227`, `exceptions.itl:16`, `constructors.itl:17`,
      `3.56?1`, `-10?u`, `0.0??u`, `2.500?5de-5`, `10?3e380`. plus 32 examples, 55 invalid texts,
      the fitting and unfitting decorations of `parse_literal`, and the signals' classes. under
      `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=20` the file ran green in 57 s (2026-09-26, 91
      items then)
* evidence, measured 2026-09-26 at part 1 (`tools/itf1788_census.py`): the 101 vectors of the two
  bare constructors (`libieeep1788_class.itl` 76, `ieee1788-constructors.itl` 22,
  `ieee1788-exceptions.itl` 3; 33 with a signal) all match with their signal except the 4 rows
  above; `isNaI` 16, all rows. all ops: 7431 vectors of 86 ops, 6301 interval-valued; 132
  divergence keys: 66 "no NaI: invalid input raises" (68 vectors: the 52 former "decoration
  expectations" rows, `isNaI [nai]` and 15 more `isNaI`), 4 "exact parsing decides validity"
  (PROPOSED), 47 cancellation, 10 degenerate infinities, 5 cut-based relations, 0 decoration
  expectations; 0 unknown failures; skipped 2111 statements of 25 ops. `tests/itf1788`: 14120
  passed in 46 s. the gate, in two runs: `tests/itf1788` 14120 and the rest 3471 in 450 s, 17591
  in all (2026-09-26)
* sabotage (section 2), 28 breaks by a throwaway harness, each file restored from a copy and
  byte-compared, results appended as they landed; targets `tests/test_literals.py`, the doctests
  of `literals.py` and the itf1788 vectors (all of them for an adapter break). red, library: `nums`
  accepting `lo > hi` 3 (`class.itl:31`); accepting `lo = +inf`/`hi = -inf` 3 (`:32`, `:33`); `nan`
  as a plain `ValueError` 2 (`:30`); text accepting `lo > hi` 8 (`:136`-`138`, the exact-parsing
  rows going stale); accepting `[inf, inf]` 4 (`:130`); an infinite point 5 (`:129`,
  `exceptions.itl:15`); infinite ends closed 15; an omitted radius a full unit 17; `u` and `d`
  swapped 21; the exponent not scaling the radius 5 (`:102`-`104`, `constructors.itl:34`); the unit
  from the whole digits 32; the bare constructor accepting a decoration 8 (`:50`, `:52`, `:55`,
  `:57`, `:60`); `[nai]` read as empty 3 (`:114`); hex fraction digits as 1 bit 8
  (`constructors.itl:31`); no zero-denominator check 2 (a `ZeroDivisionError` escapes); case
  sensitive 18; the decimal exponent's sign dropped 6 (`:104`, `constructors.itl:30`). **4 were
  thin, 1 red each, only `test_decorations` or the white-space property saw them**: the empty set
  decorated other than `trv`, `com` on an unbounded interval, an unknown decoration accepted,
  outer white space stripped. no vector can see the first three through the bare constructor, which
  refuses every decoration; `test_parse_literal_reads_a_fitting_decoration`,
  `test_parse_literal_refuses_a_decoration_that_does_not_fit` and four outer-space texts in
  `test_invalid` were added, and the four breaks then turned 3, 6, 3 and 5 red. adapter: the
  expected signal ignored 30; a raise read as no signal 30; the constructors not routed to the
  signal rule 33; `b-numsToInterval` out of `SIGNALLED` 16 (`test_signals_are_checked` among
  them); the `isNaI` rows dropped 15; `isNaI` answered `False` 15; **the warning never read 1,
  only `test_signalled_reads_both_signals`**: no vector reaches it, since the library never warns,
  which is why that test was added before the run
* **part 2 built 2026-09-26 (branch `m13g`): the decorated type, `d-textToInterval`,
  `d-numsToInterval`, `newDec`, `setDec`, `intervalPart`, `decorationPart`, and the adapter checking
  their decorations.** still open for M13g after it: **decorations propagated through the core's
  ops** (1788's decorated arithmetic and functions; the plan's "propagated through every op the
  core has"), and the adapter checking the decorations of those ops' vectors, which it still drops:
  1503 vectors of other ops carry a decoration (1022 of built ops: `libieeep1788_elem.itl` 493,
  `bool` 210, `cancel` 121, `num` 87, `rec_bool` 72, `overlap` 29, `set` 10; 481 of the reverse
  ops; counted with `tests/itf1788/itl.py::parse_file`, 2026-09-26). built:
    * `intervals/decorated.py`, a new module above `literals.py`: `::Decoration`, an enum `COM`,
      `DAC`, `DEF`, `TRV` (values the 1788 names) ordered `TRV < DEF < DAC < COM`, no `ILL`;
      `::DecoratedInterval(interval, decoration=None)`, immutable and hashable, with the properties
      `.interval` (1788's `intervalPart`) and `.decoration` (`decorationPart`); `::set_dec`
      (`setDec`); `::text_to_decorated_interval` (`d-textToInterval`, `literals.py::parse_literal`
      then the literal's decoration or newDec's) and `::nums_to_decorated_interval`
      (`d-numsToInterval`). all five exported from `intervals`
      (`tests/test_applicator.py::test_package_exports_unchanged` lists them)
    * **a core bug found and fixed**: `MultiInterval.is_finite` and `.finite` raised `OverflowError`
      on an exact end past the doubles (`MultiInterval(10**400).is_finite`), since `math.isfinite`
      converts to float; `newDec` of the literal `[1.0E+400]` hit it. they compare with ±inf now,
      pinned by `tests/test_multi_interval.py::test_finiteness_of_an_exact_end_past_the_doubles`
    * **choices the plan left open** (conservative, flagged in the decision log, `v2-plan.md`
      "2026-09-26 revision: M13g part 2"):
        * the name `DecoratedInterval`, after M8's `DateTimeInterval`/`TimeDeltaInterval` (wrappers
          of a `MultiInterval` named `...Interval` though multi-piece); `Decoration` is a plain
          enum, not a `str` one, so `Decoration.COM != 'com'` (`==` does not coerce, as in the core)
        * newDec is the constructor with no decoration and the two parts are properties, so only
          `set_dec` and the two constructors are functions (module-level, as `text_to_interval`)
        * **the constructor is strict, `set_dec` follows 1788**: `DecoratedInterval(x, d)` with a
          `d` that does not fit (anything but `trv` on ∅, `com` on an unbounded set) raises
          `UndefinedOperationError`, as the literal `"[1,]_com"` does; `set_dec` demotes as 1788
          defines `setDec`, with no signal (`libieeep1788_class.itl:283`-`288`: `[empty]_trv`,
          `_dac`), so its decoration is `min(d, newDec's)`, and raises only for `ill`
          (`:289`-`291`, which 1788 answers `[nai]` with `UndefinedOperation`). the task suggested
          raising for every unfitting `setDec`; that would turn six vectors 1788 answers without a
          signal into rows under a new category, so it was not taken. **an owner question**
        * a decoration argument is a `Decoration` or its exact lower-case name (`'COM'` raises: a
          python argument is not 1788 text, whose parser is case-insensitive); `ill` and any other
          name raise `UndefinedOperationError`; any other type, and a non-`MultiInterval` set, is a
          `TypeError`
        * **bounded is decided on the exact set**: com needs a non-empty set with no point at and
          no piece reaching ±inf, so `[1.0E+400]_com` keeps com where 1788's binary64 hull
          `[max, inf]` is demoted to dac; an attained infinity (`[1, inf]`) is unbounded; a
          bounded set of several pieces is com. the three vectors (`libieeep1788_class.itl:165`,
          `:201`, `:204`) are rows under the existing **"decoration expectations"**
          (`tests/itf1788/test_itf1788.py::_BOUNDED_EXACTLY`); **an owner question**: or under the
          PROPOSED "exact parsing decides validity", widened to "exact parsing"
        * equal iff both parts are; never equal to the bare set; the set keeps its class (an
          `OutwardMultiInterval` stays one); `str` is the set in our syntax then `_com` (it reads
          back as 1788 text for a bounded connected closed exact set only); `repr` evaluates back;
          no `__bool__`, no arithmetic (propagation is the open part)
    * adapter (`tests/itf1788/test_itf1788.py`): the six ops in `OPS` (`newDec` is
      `DecoratedInterval`, the parts `.interval`, `.decoration`); `::DECORATED`, `::CONSTRUCTORS`;
      `::to_ours` keeps a decorated operand's decoration for a decorated op (`decorated=`, via
      `::_args`); `::_ours` and `::_expected` compare a decorated value as (closed hull, decoration)
      and a `Decoration` by name; `::_nai_is_a_raise` and `::_RAISED`: a raise is the decorated
      flavour's `[nai]` with `UndefinedOperation`, so those vectors match and the generated NaI
      rows skip them; `SIGNALLED` gains `d-textToInterval`, `d-numsToInterval`, `setDec`,
      `intervalPart`; the outward pass now excludes only `CONSTRUCTORS`, so `newDec`, `setDec` and
      `intervalPart` run outward (`::_signalled(vector, outward=True)`, and `::_outward_hull`,
      factored out of `::run_outward`); `::test_decorated_ops_are_checked` pins both. rows: the
      three `d-` twins of the `_EXACT_INVALID` vectors (`class.itl:229`-`231`, PROPOSED category,
      pending the owner) and the three `_BOUNDED_EXACTLY`; no new category
    * tests (`tests/test_decorated.py`, 40 items, 2026-09-26): 12 `@given`: newDec against 1788's
      definition written out in the test (`::fits`, `::bounded` on the cuts), with maximality
      (every decoration up to newDec's fits and constructs, none past it does, over sets with
      several pieces, ±inf as points and unattained ends, and exact ends past the doubles);
      `set_dec` against 1788's definition, never promoting, the best fitting at or below `d`,
      `min(d, newDec's)`, idempotent, by member and by name; the order `trv < def < dac < com`;
      `text_to_decorated_interval` as `text_to_interval` plus a decoration over bracket literals,
      the uncertain form and any undecorated text over the literal alphabet (fitting kept,
      unfitting raises, none gives newDec's); `nums_to_decorated_interval` as `nums_to_interval`
      plus newDec's; round trips (`str` of a bounded connected closed exact set read back as 1788
      text, `repr` through `eval`, pickle, deepcopy); equality of both parts and hash; the set's
      class kept (`OutwardMultiInterval`); immutability. `@example`s: `libieeep1788_class.itl:37`,
      `:38`, `:40`, `:42`-`44`, `:143`, `:144`, `:147`, `:148`, `:152`, `:155`, `:165`, `:167`, `:168`,
      `:188`, `:198`, `:204`, `:208`-`211`, `:214`, `:216`, `:223`, `:225`, `:227`, `:229`, `:261`,
      `:263`, `:264`, `:276`, `:279`, `:280`, `:283`-`288`, `ieee1788-constructors.itl:25`, `:52`,
      `:57`, `:71`, `:73`. plus parametrized: `ill` and other names raise, non-decorations and
      non-sets are `TypeError`s, the constructor refuses each unfitting pair that `set_dec`
      demotes. under `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10` the file ran green, 40 items in
      307 s (2026-09-26, a loaded laptop)
* evidence, measured 2026-09-26 at part 2 (`tools/itf1788_census.py`, and the adapter imported
  for the decorated counts): the 156 vectors of the six decorated ops (`d-textToInterval` 91,
  `setDec` 22, `intervalPart` 15, `newDec` 13, `d-numsToInterval` 9, `decorationPart` 6; 35 with a
  signal) all match, decoration included, except 11 rows: 5 "no NaI" (`[nai]` as text or operand),
  3 PROPOSED "exact parsing decides validity", 3 "decoration expectations"; 30 of them match by a
  raise read as `[nai]` with `UndefinedOperation`. the 50 outward items of `newDec`, `setDec`,
  `intervalPart` match but for the 2 `intervalPart [nai]` rows. all ops: 7587 vectors of 92 ops,
  6351 interval-valued; 142 divergence keys: 70 "no NaI: invalid input raises" (73 vectors), 7
  "exact parsing decides validity" (PROPOSED), 3 decoration expectations, 47 cancellation, 10
  degenerate infinities, 5 cut-based relations; 0 unknown failures. **skipped: 1955 statements of 19
  ops, all M13e's reverse ops**; no other op is left for M13's exit. the gate, in two runs:
  `tests/itf1788` 14329 passed in 141 s; the rest 3516 passed in 953 s (the laptop shared and loaded; the same run took 450 s at part 1), 17845 in all (2026-09-26)
* sabotage (section 2), 31 breaks by a throwaway harness, each file restored from a copy and
  byte-compared, results appended as they landed; targets `tests/test_decorated.py`, the doctests
  of `decorated.py`, `tests/test_multi_interval.py` and the itf1788 vectors of the class,
  constructor, exception and bool files with the adapter's own tests (962 items then). red,
  library: newDec com for every non-empty set 28 (`class.itl:38`-`40`, `:147` ...); dac for every
  one 68; def for ∅ 17 (`:142`, `:264`, `:283`-`285`); set_dec not demoting 18 (`:283`-`288`);
  set_dec as `max` 35; `ill` read as trv 7 (`:289`-`291`); names case-insensitive 2; a non-str
  decoration passed through 5; a non-`MultiInterval` accepted 6; def and dac swapped in the order
  10; the text's decoration dropped 19; nums as def 6; `str` without the decoration 6; the core's
  `is_finite` through `math.isfinite` again 11 (`class.itl:165`, `:204` among them); the
  constructor refusing newDec's own decoration too: a collection error (a module-level
  `DecoratedInterval` in a parametrize), then 112 and 2 errors with
  `--continue-on-collection-errors`. **seen by one test each**, each the property that decides the
  clause: `__lt__` answering a str (`test_the_order_is_only_among_decorations`), `==` ignoring the
  decoration or the set (`test_equality_is_both_parts`), pickle dropping the decoration
  (`test_repr_pickle_and_copy_give_it_back`), `.finite` through `math.isfinite`
  (`test_finiteness_of_an_exact_end_past_the_doubles`). **thin, then thickened**: the
  constructor's fit check dropped, 1 (only newDec's maximality property; `set_dec` demotes and the
  literal parser checks the fit itself, so no vector reaches it): `test_the_constructor_refuses_what_does_not_fit`
  added, then 7. adapter: our decoration dropped 129; always com 74; the expected decoration
  dropped 129; a decorated raise read as empty 33; operand decorations dropped 31; `setDec` out of
  `SIGNALLED` 7; the decorated ops not run outward 1 (`test_decorated_ops_are_checked`); a
  `_BOUNDED_EXACTLY` row dropped 1 (`class.itl:204`). **green, then pinned**: `_nai_is_a_raise`
  always false turned 0 red, since the 30 vectors it matches then become generated NaI rows,
  which pass as rows: a rule that fails by passing. `test_undefined_operation_is_never_a_row`
  added (no vector with `signal UndefinedOperation`, 59 of them, is a row), then 1 red. the
  decorated outward pass run on `MultiInterval` turned 0 red, since `newDec` does no arithmetic:
  `test_outward_pass_of_a_decorated_op_is_outward` added (a spy on the operand's class), then 1
* **part 3 built 2026-09-26 (branch `m13g`): decoration propagation through every point function
  of the core, set operations trv, and the adapter checking the decoration of every decorated
  vector.** left for M13g after it: the reverse ops' decorated vectors, with M13e (a parallel
  branch; the hook below). built:
    * `intervals/decorated.py`: `DecoratedInterval` gains the core's point functions: `+ - * /` and
      their reflected forms, `%`, `//`, `divmod`, `**` (D11's dispatch: an integral real exponent is
      pown, `::_pown`, anything else pow, `::_pow`) and `__rpow__`, `-x`, `+x`, `abs`, `reciprocal`,
      `minimum`, `maximum`, `fma`, `hypot`, `atan2` (`::_atan2`), `log(base)`, `rootn(n)`, the 29
      other elementary functions (made by `::_function` from `::_FUNCTION_DOMAINS` and `::_POLES`),
      `floor`, `ceil`, `trunc`, `round(ndigits)`, `round_ties_away(ndigits)`, `sign` (`::_step`), and
      `math.floor`/`ceil`/`trunc` and `round()`; and the set operations `& | ^ ~`, `difference`,
      `complement`, `hull`, `closed_hull`, `interior`, `cancel_minus`, `cancel_plus`, all trv
      (`::_trivial`). each computes the core's set on the intervals (so the set, its class and the
      core's warnings are the core's) and decorates it with `::_propagate`: the local decoration on
      the box of the operands' sets is trv unless `defined` (the box inside 1788's domain of the op,
      a set of reals: `::_REALS`, `::_NON_ZERO`, `::_POSITIVE`, ...), def unless `restricted` (the op
      restricted to the box is continuous), dac unless `everywhere` (continuous at each point of the
      box) and every operand is bounded, else com; the result's decoration is `min(local, newDec of
      the result, each operand's)`. only four kinds of op have jumps inside their domain and compute
      `restricted` and `everywhere`: the step functions (`::_step`: constant on each piece, and no
      closed end a jump, `::_JUMPS`), atan2 (the negative x axis), `%` and `//`
      (`::_quotient_steps`: per pair of pieces, `floor(x / y)` one integer and `x / y` never reaching
      it). the poles of tan, sec, cot, csc are found exactly (`::_misses_poles`, through
      `functions.py::_inside_k` and `elementary.py::floor_over_pi`). every decision is made on the
      exact set (`::_exact`, `rounding.py::exact_cuts`), the core's ops it needs computed quietly
      (`::_quietly`)
    * **choices the plan left open** (conservative, flagged in the decision log, `v2-plan.md`
      "2026-09-26 revision: M13g part 3"):
        * **a multi-piece box is decided on the set, not on the hull** (the reading the task asked
          for): continuity on the set is continuity on each piece, since normalized pieces are
          apart, so `floor` on `[1/4, 1/2] ∪ [5/4, 3/2)` is com and on `[1/4, 1/2] ∪ [1, 3/2)` dac (a
          closed jump at 1), though it is constant on neither hull; `%` and `//` per pair of pieces
        * **an attained ±inf is outside every domain**: 1788's functions are functions of reals, so
          an operand holding ±inf as a point gets trv, even where the core gives it a limit
          (`exp([0, inf])` trv, `exp([0, inf))` dac); the number `inf` as an operand is such a point
        * **com needs the result bounded as returned**: exact for exact operands, rounded for float
          ones. exact operands keep com where 1788's binary64 result overflows: 12 vectors, rows in
          the plain pass only (below). `OutwardMultiInterval` rounds as 1788 does and matches them;
          `MultiInterval` rounds `2.0 + max` to nearest (max) and keeps com
        * continuity at a point is relative to the op's domain: `sqrt([0, 25])`, `acosh([1])` and
          `pow([0, 1/2], [1/10])` are com, as `libieeep1788_elem.itl` has them; `pown(x, 0)` is
          defined at 0 (`pown [-5.0,10.0]_com 0 = [1.0,1.0]_com`)
        * a step function is at best dac on a set holding a jump (`sign([0])` from com is dac, as
          `ceil [max,max]_com` is in 1788). `%`, `//`, `divmod` and `round(ndigits)`, which 1788
          lacks, follow the same rule from their definitions: defined where the divisor is not 0,
          jumps where `x / y` is an integer or at the half grid step
        * every set operation, `hull` and `interior` included, is trv, as 1788 decorates
          intersection, convexHull, cancelMinus and cancelPlus. the booleans, numbers and relations
          are not on the wrapper: `.interval` first, as 1788 defines them on the interval part
        * an operand is a `DecoratedInterval` or a real number (newDec's point, in the receiver's
          class); a bare `MultiInterval`, a `bool` or anything else is a `TypeError`, as 1788 has no
          implicit mix of bare and decorated intervals
        * the core's warnings reach the caller once, attributed as before; the decoration's own core
          calls warn nothing
    * **M13e hook**: 1788 decorates every reverse op's result trv (the 459 decorated results in the
      four reverse-op files are all `_trv`, counted 2026-09-26), so a decorated reverse op is
      `decorated.py::_trivial(<the reverse op on the intervals>)`, and in the adapter each reverse
      op joins `tests/itf1788/test_itf1788.py::PROPAGATED`. until it does,
      `::test_decorated_vectors_run_decorated` fails after the merge (an interval-valued op with
      decorated vectors outside `PROPAGATED`): the reminder is mechanical. **but not for a pair**
      (the M13g review, 2026-09-26): `mulRevToPair`'s 174 decorated vectors expect a pair, which
      that test does not see and `::_ours`/`::_expected` do not compare with decorations, so the
      merge also needs a pair-with-decoration case there; `::test_no_decorated_pair_goes_unchecked`
      fails until it is written
    * adapter (`tests/itf1788/test_itf1788.py`): `::PROPAGATED` (57 interval-valued ops) and
      `::BARE_PART` (the other 23 with interval operands: booleans, numbers, overlap); `::is_decorated`;
      `::_args` builds a decorated vector's operands as `DecoratedInterval`s (so each decoration must
      fit its set) and gives a `BARE_PART` op their interval parts; `::run`, `::run_outward` and
      `::run_float` all go through `::_args` and `::_ours`, and `::_expected` keeps the expected
      decoration for `PROPAGATED`. `::PLAIN_ONLY`: rows on a decoration alone, for the plain pass
      only, keyed on the statement with its decorations (`exp2 [1024.0,1024.0] = [max,infinity]`
      is also the key of its bare twin, which matches), read by `::row(vector, outward)` in `::check`;
      `::test_divergence_rows` checks each is one interval-valued vector under no other row, so its
      outward item must match. `::test_decorated_vectors_run_decorated` pins the wiring: every
      decorated interval-valued op in `PROPAGATED`, none in `BARE_PART`, and a decoration on both
      sides of a propagated result in both passes (an adapter dropping it on both sides passes every
      vector). `tools/itf1788_census.py` counts the decorated vectors and the plain-only rows
    * tests (`tests/test_propagation.py`, 115 items, 67 s on the loaded laptop, 2026-09-26): an
      oracle written out from 1788's definitions and decided by brute force (operands on a quarter
      grid, so the domains' ends and the jumps are grid points, and the eighth grid plus points just
      inside each end decides domain, constancy and jumps exactly; the poles against a 32-digit pi;
      atan2's cut and `%`'s jumps from their definitions, `x / y` from the corners). `@given`: every
      unary op (40) and binary op (11) against the oracle, over a mixed-op strategy with the itf1788
      examples and per op (parametrized, 40 examples each); pown, rootn, fma, `round(ndigits)` on a
      scaled grid; float operands, both classes and doubles up to ±max, decorate as the exact values
      of the same doubles but for newDec of the rounded result; the min law (`f(set_dec(x, d))` is
      `min(d, f(newDec x))`); antitone in the box (a non-empty sub-box never decorates worse). plus
      set operations trv, the class kept, numbers as points and a bare set refused, divmod, warnings
      once, the methods mirroring the core, `round(ndigits)` examples. `@example`s:
      `libieeep1788_elem.itl:110`, `:111`, `:113`, `:306`, `:676`-`678`, `:708`, `:709`, `:755`,
      `:756`, `:1403`, `:1405`, `:1588`, `:1589`, `:1596`-`1598`, `:3167`, `:3236`, `:3506`, `:3508`,
      `:3527`, `:3554`, `:4086`, `:4087`, `:4114`, `:4141`, `:4145`, `:4167`, `:4200`, `:4203`,
      `:4232`, `:4241`, `:4269`, `:4299`, `:4301`, `:4353`, the atan2 cut cases, the pow domain cases,
      `libieeep1788_set.itl:33`; and, added after the sabotage runs (below), the ends of rootn's and
      log1p's domains, `1.0 % 0.1` in floats and `x ** 2.0 == x ** 2`. the first strategy gave trv in about 95% of examples; rebalanced
      (mostly newDec's decoration, infinities mostly open, narrow pieces, ends at 0, ±1/2, ±1 often),
      per 200 examples of the earlier rebalance floor gave trv 99, def 71, dac 8, com 22 and atan2
      135, 20, 28, 17 (2026-09-26). under `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10` the file ran
      green, 111 items in 304 s (2026-09-26, before the 4 `round(ndigits)` examples)
    * **a test-oracle bug found by the gate run** (not the library): M13d's
      `tests/test_functions.py::_hypot_holds` required an irrational value's float bracket inside
      the result, which fails next to an exact end: `hypot((-inf, -2/3], [0, 1/2))` is `[2/3, inf)`,
      right, and `hypot(-2/3, 2**-27)` lies in it though its lower double is below 2/3. it now
      decides the bracket exactly on the squares, pinned by
      `tests/test_functions.py::test_the_hypot_oracle_next_to_an_exact_end`; flipping that
      comparison turned 2 red (restored, `cmp` ok)
* evidence, measured 2026-09-26 at part 3 (`tools/itf1788_census.py`, and the adapter imported): of
  the 1022 decorated vectors of the core's ops (`libieeep1788_elem.itl` 493, `bool` 210, `cancel`
  121, `num` 87, `rec_bool` 72, `overlap` 29, `set` 10), 624 are of 44 propagating ops and now
  checked with their decoration in both passes: 560 match in the plain pass and 572 outward; 52
  keep the rows their bare part already had (47 cancellation, 4 no NaI, 1 degenerate infinity,
  `atanh [1.0,1.0]_def`); **12 are new rows on the decoration alone, plain pass only**, under the
  existing **decoration expectations** (`::_OVERFLOWS_ONLY_ROUNDED`): the exact result is bounded,
  past the doubles, so com; 1788's binary64 result overflows and is dac. they are
  `libieeep1788_elem.itl:111`, `:112` (add), `:161`, `:162` (sub), `:305`, `:306` (mul), `:676`
  (div), `:731` (sqr), `:1404` (fma), `:1591`, `:1593` (pown), `:3167` (exp2, `2 ** 1024`); **an
  owner question**, as part 2's `_BOUNDED_EXACTLY`: this category, or one for exact values past the
  doubles. no decoration differs because of the multi-interval set semantics. the other 398 (of
  `BARE_PART` ops) take the interval part: 358 match, 40 keep their rows (38 no NaI, 2 cut-based
  relations). all ops: 7587 vectors of 92 ops, 6351 interval-valued; 142 keys (unchanged) plus the
  12 plain-only rows; 0 unknown failures; skipped 1955 statements of 19 ops, all M13e's.
  the gate, in two runs: `tests/itf1788` 14330 passed in 120 s and the rest 3636 passed in 863 s (a
  shared, loaded laptop), 17966 in all (2026-09-26); after the sabotage runs' examples, 14330 in
  35 s and 3636 in 411 s (2026-09-26)
* sabotage (section 2), 45 breaks by a throwaway harness, each file restored from a copy and
  byte-compared (all `cmp` ok), results appended as they landed; targets `tests/test_propagation.py`,
  `tests/test_decorated.py`, the doctests of `decorated.py` and the itf1788 vectors of the seven files
  with decorated vectors plus the adapter's own tests (2026-09-26). a first run stopped at break 31:
  under the load, hypothesis's 200 ms deadline failed unrelated atan2 and div properties, so every
  `@settings` in the file now has `deadline=None` and the run was repeated clean. red, library:
  `_propagate` without the operands' decorations 273 (every per-op property, `elem.itl:4352`,
  `:4353` ...); undefined read as def 329; restricted continuity ignored 73 (`elem.itl:4271`
  ...); continuity at every point ignored 23 (`:4211`, the `round(ndigits)`
  examples); the reals closed at ±inf 26; div defined at 0 5 (`:677` both passes); pow at `0 ** y`,
  `y <= 0` 56; pow on negative bases 5 (`:3081`, `:3092`); atan2 at the origin 118; atan2's cut from
  below missed 7 (`:3816`, `:3900`, `:3928`); atan2's cut ignored for com 4 (`:3942`, the atan2
  doctest); poles inside a piece ignored 25 (`:3501`, `:3509` ...); tan's poles at `k pi` 18;
  constancy of a step not checked 63; a closed jump not checked 16 (`:4167`, `:4211`); round's jumps
  at the integers 5 (`:4269`); `floor(x / y)` constancy not checked 4; the `%`/`//` domain 3; pown
  `n < 0` at 0 6 (`:1596`, `:1598`); acosh without 1 8 (`:4084`, `:4086`, `:4088`); reciprocal at 0 10
  (`:709`-`:712`); set operations newDec's instead of trv 193 (`test_set_operations_are_trv`, the
  decorated `cancel.itl` vectors); the result cap dropped with the strict constructor kept 36 (it raises: `:3167`,
  atanh, cot, coth, csc properties). **seen by examples only** (the random properties missed them in
  the run, so each is pinned by an `@example` or a direct test, several added after the first
  runs): a closed end at the pole 0 (cot, csc) 2; trunc jumping at 0 2; sign never jumping 1;
  `round(ndigits)`'s grid ignored 2 (0 before `test_round_to_ndigits_examples`, whose first example
  had been wrong); `x / y` reaching the integer not checked 2 (`test_divmod_is_both_decorated`; the
  first `%` example, meant for it, was def); an even root below 0 and an odd negative root at 0,
  1 each (0 and 1 before two rootn examples); log1p at -1 1 (0 before its example); decisions on the
  float set instead of the exact one 1 (0 before the example `1.0 % 0.1`, whose float ratio rounds to
  10); `**` of an integral float as pow 1 (0 before `x ** 2.0 == x ** 2`); a bare `MultiInterval`
  accepted 1, the core's calls not quieted 1, `__rsub__` not reflected 1. **green, as expected**: the
  cap replaced by `set_dec` 0, which demotes to newDec's the same way (equivalent code; the real break
  is the one above, 36); the operands' boundedness ignored for com 0, unreachable, since an unbounded
  operand is never com (the strict constructor), kept as 1788 states the rule. adapter: decorated
  vectors not detected 1132; the expected decoration dropped 1132; `BARE_PART` ops given the
  decorated operands 467; `PLAIN_ONLY` read in the outward pass 12, dropped 12. **both sides
  dropped: 13**, the 12 plain-only rows going stale and `::test_decorated_vectors_run_decorated`;
  `floor` out of `PROPAGATED`: 1, that test alone (every floor vector matches on the interval part).
  those two first ran green: the harness's `-k 'not vector'` had excluded the pin test by its name,
  so they were rerun with it
* **done 2026-09-26 (branch `m13g`; parts 1 to 3 above, then a close-out).** M13g's spec is met,
  but for the owner's decisions below (the PROPOSED category's 7 rows; the constructors' outward
  pass, from the review):
  every statement of its ops is a vector that matches or is a row, every decorated vector's
  decoration is checked, and nothing of M13g is skipped. built, over the three parts:
    * the signals, `intervals/errors.py::UndefinedOperationError` (a `ValueError`) and
      `::PossiblyUndefinedOperationWarning` (an `IntervalWarning`); 1788's literals,
      `intervals/literals.py::parse_literal`, and the bare constructors `::text_to_interval`,
      `::nums_to_interval` (part 1)
    * the decorated type, `intervals/decorated.py::Decoration` and `::DecoratedInterval` (newDec,
      `.interval`, `.decoration`), `::set_dec`, `::text_to_decorated_interval`,
      `::nums_to_decorated_interval` (part 2); decoration propagation through every point function of
      the core and trv for its set operations, `::_propagate`, `::_trivial` (part 3). the core
      `MultiInterval` stays undecorated (D16), with one fix, `is_finite`/`finite` on an exact end past
      the doubles (part 2)
    * the close-out: `tests/itf1788/test_itf1788.py::test_only_the_reverse_ops_are_skipped` (with
      `::_REVERSE_OPS`, the 19 reverse ops' names); the stale comment above `::DIVERGENCES` now
      names M13g's rows; README (a doctest-checked example of the constructors, the raise and
      propagation; a signals bullet; the `ieee 1788` counts re-measured); `v2-plan.md` "ieee 1788"
      (D16 as current design, the counts at M13g, the signal rule's list), "package layout",
      "testing", "later", and "2026-09-26 revision: M13g, decorations, constructors and signals,
      done"
* **choices the plan left open** (conservative; each part's own are in its record above and in its
  `v2-plan.md` revision entry):
    * the close-out pin checks that the skipped ops are a **subset** of the 19 reverse ops, named in
      the test rather than derived, so it holds before and after M13e's merge (which empties
      `SKIPPED`); M13's exit, `SKIPPED` empty and asserted, stays M13-exit's to add
    * the README and `v2-plan.md` counts are re-measured on this branch (M13e's merge changes them
      again, and they are to be re-measured then with `tools/itf1788_census.py`); the M13 heading's
      status line is left to the session that merges
    * three owner questions stay open, each built as the conservative reading: the PROPOSED category
      "exact parsing decides validity" (7 rows); the 15 rows where an exact value past the doubles
      keeps com (`::_BOUNDED_EXACTLY` 3, `::PLAIN_ONLY` 12) under "decoration expectations" or a
      category of their own; `set_dec` demoting as 1788's `setDec` does rather than raising
* M14: no new op landed in the close-out, so no new property; the M13g ops have theirs from their
  parts (`tests/test_literals.py`, `tests/test_decorated.py`, `tests/test_propagation.py`)
* tests, measured 2026-09-26: `tests/test_literals.py` 122 passed in 1.1 s,
  `tests/test_decorated.py` 40 in 4.7 s, `tests/test_propagation.py` 115 in 15.1 s; `tests/itf1788`
  14331 passed in 35.1 s (the one new pin added to part 3's 14330)
* evidence, measured 2026-09-26 at the close-out (`tools/itf1788_census.py`, and the adapter
  imported): the 273 statements of M13g's ops are all vectors: `b-textToInterval` and
  `b-numsToInterval` 101, the six decorated ops 156, `isNaI` 16. every decorated vector's
  decoration is checked: 1040 vectors carry one, 624 of 44 propagating ops (with it, both passes),
  398 of `BARE_PART` ops (each operand built as a `DecoratedInterval`, so its decoration must fit,
  then its interval part), 18 of the decorated ops, whose 156 vectors are all compared with their
  decoration or raise. all ops: 7587 vectors of 92 ops, 6351 interval-valued, 167 numeric run twice
  more, 14272 vector test items; 142 keys (70 no NaI with 73 vectors, 47 cancellation, 10 degenerate
  infinities, 7 exact parsing PROPOSED, 5 cut-based relations, 3 decoration expectations) plus 12
  plain-only rows; 0 unknown failures. skipped: 1955 statements of 19 ops, all M13e's reverse ops.
  the gate, in two runs: `tests/itf1788` 14331 passed in 35 s; the rest 3636 passed in 459 s;
  17967 in all (2026-09-26)
* sabotage (section 2) of the close-out, each file restored from a copy and `cmp`-checked: `isNaI`
  dropped from `OPS` 1 red, **only the new pin**: before it, its 16 statements would have become
  silent skips with every test green (`test_parser_drops_nothing` only checks that an op in `OPS`
  is not skipped), which is why the pin was added; `setDec` dropped from `OPS` 2
  (`::test_decorated_ops_are_checked` and the pin); the README's `_trv` example expected as `_dac`
  1 (the README doctest, so the example is collected). the parts' own: 28, 31 and 45 breaks, above
* review (2026-09-26, three read-only reviewers over `4956c86`, lenses math, sabotage and spec; each
  finding reproduced on this branch before any change): **no wrong result in the library** (the math
  lens's own brute-force oracle over 39 unary and 11 binary ops, pown, rootn, fma and 75000 float
  calls: 0 mismatches in set, decoration or warning). found and fixed:
    * **the wrapper lacked some of the core's set operations** (math and spec lenses): the named
      n-ary `union`, `intersection`, `symmetric_difference` were missing, `difference` took one
      operand, and `positive`, `negative`, `finite`, `expand` were missing. now on
      `decorated.py::DecoratedInterval`, n-ary as the core's and trv as every set operation; so is
      the restriction `x[a:b]` (`__getitem__`), with `__iter__ = None` so that the wrapper is not
      taken for a sequence. pinned by 15 new ops in `tests/test_propagation.py::test_set_operations_are_trv`
      (60 items) and by `::test_every_public_name_of_the_core_is_on_the_wrapper_or_asked_of_the_interval`:
      every public name of `MultiInterval` is on the wrapper or in `::NOT_ON_THE_WRAPPER` (the
      booleans, numbers, relations and structure, asked of `.interval`), so a name the core gains
      goes red until it is placed. **a choice the plan left open**: `expand`, which 1788 lacks, is
      trv as a set operation (the weakest claim), not propagated as a point function
    * **the literal regex backtracked in cubic time on invalid text** (math lens): a digit run
      matched `[0-9]+\.?[0-9]*` in as many ways as it has digits, and adjacent `\s*` split white
      space every way; measured 2026-09-26, two runs of 400 digits 5.4 s, two of 400 spaces 0.8 s,
      8x per doubling. `literals.py::_DECIMAL`, `::_HEX` and the uncertain form's mantissa now read
      `[0-9]+(?:\.[0-9]*)?` (the same language, one split) and `::_LITERAL`'s white space is
      possessive (`\s*+`, python >= 3.11, as `pyproject.toml` requires): those texts now take under
      1 ms. pinned by `tests/test_literals.py::test_invalid_text_is_refused_in_linear_time` (8 long
      invalid texts, under 1 s together; about 10 ms, 2026-09-26)
    * the strict constructor's message cited both rules; it now gives the one that applies
      (`tests/test_decorated.py::test_the_constructor_refuses_what_does_not_fit` matches it)
    * docs: README's "142 listed divergences" counted keys, and now says the 195 vectors under the
      142 keys; "every sub-task" below now names D16's category as approved too; the done line above
      is qualified by the owner's open decisions
    * **unpinned clauses, each now pinned** (the sabotage lens found each at 0 red under
      `HYPOTHESIS_PROFILE=ci`): the reflected `/ % // **` (`::test_reflected_operators_reflect`,
      values, not only the class); the attained-inf checks on pow's exponent, div's dividend, fma's
      addend and unary plus, fma's addend decoration, rootn's even negative domain on
      `{[-1, -1/2], [1, 4]}`, asin's lower end and acoth's two ends (`@example`s on the per-op
      properties); the step functions deciding constancy on the exact set
      (`::test_a_step_is_decided_on_the_exact_set`: `OutwardMultiInterval(0.12, 0.13)` and `(0.15)`,
      `round(1)` and `round_ties_away(1)`, com); `re.ASCII` (7 texts in `test_invalid`: white space
      ` `, `\xa0`, `　`, `\x1c`, and `ı`, which folds to `i` without it); the adapter's
      `::_signalled` reading only `UndefinedOperationError` as the signal
      (`tests/itf1788/test_itf1788.py::test_signalled_reads_only_undefined_operation`)
    * **the M13e hook missed pairs** (sabotage lens): see "M13e hook" above;
      `tests/itf1788/test_itf1788.py::test_no_decorated_pair_goes_unchecked` fails once an op with a
      decorated pair-valued vector joins `OPS` (its predicate over every statement of the 19 files,
      `itl.py::parse_file`, finds `mulRevToPair` 174, 2026-09-26)
* sabotage of the review's fixes (section 2), a throwaway harness, each file restored from a copy and
  `cmp`-checked (all ok); targets `tests/test_propagation.py`, `tests/test_decorated.py`,
  `tests/test_literals.py`, the doctests of `decorated.py` and `literals.py`, the adapter's own tests;
  `HYPOTHESIS_PROFILE=ci`, 2026-09-26. every mutation the sabotage lens reported green is now red:
  `__rtruediv__`, `__rmod__`, `__rfloordiv__`, `__rpow__` not reflected 1 each; pow's exponent
  check 1, div's dividend check 1, fma's addend decoration 1 and its inf check 1, unary plus 1; the
  step on the float set 4; rootn's `_POSITIVE` as `_NON_ZERO` 1; asin widened to -2 1; acoth closed
  at 1 1, at -1 1; the lax adapter (`except ValueError`) 1; `re.ASCII` dropped 6. the new clauses:
  newDec instead of trv for `union` 12, `symmetric_difference` 8, `positive` 4, `negative` 1,
  `finite` 3, `expand` 8, the restriction 8; `intersection` and `difference` on their first operand
  only 4 and 3 (`intersection` first ran green: its case `a.intersection(b, a)` equals `a ∩ b`, so
  it became `a.intersection(a, b)`); `__iter__` not refused 1; the generic message 6; the white space
  not possessive 1, the decimal, hex or mantissa run ambiguous again 1 each, the whole regex of
  `4956c86` 1 (each `test_invalid_text_is_refused_in_linear_time`). **equivalent, so not kept**: an
  atomic group around a number and possessive radius, exponent and decoration runs, 0 red: with one
  split per run nothing is left to retry
* **owner questions** from the review (the orchestrator carries them to `HANDOFF.md`, which this
  branch does not edit): (1) the PROPOSED category "exact parsing decides validity" (7 rows), in
  `REASONS` unapproved; (2) **the constructors' 201 interval-valued vectors run in the plain pass
  only** (spec lens: `b-textToInterval` 91, `d-textToInterval` 91, `b-numsToInterval` 10,
  `d-numsToInterval` 9), where the rule says both passes. they take text or numbers, not an interval,
  and return the exact set, so an outward item would repeat the plain call; a class argument (an
  `OutwardMultiInterval` result, 1788's binary64 hull) would give the pass something to check, and
  is new API. conservative, kept as built; (3) and (4) as in the close-out: the 15 rows where an
  exact value past the doubles keeps com, and `set_dec` demoting
* not a defect: a trial merge with `m13e` (spec lens) puts `m13e`'s `PAIRS` branch of `run_outward`
  inside `::_outward_hull`, which has no `vector`; the merger moves it back by hand. the other
  conflicts are insertion points
* tests and evidence, measured 2026-09-26 after the review: `tests/test_literals.py` 130 items,
  `tests/test_decorated.py` 40, `tests/test_propagation.py` 181; `tools/itf1788_census.py` unchanged
  (7587 vectors of 92 ops, 6351 interval-valued; 142 keys with 195 vectors plus 12 plain-only rows;
  1040 decorated vectors; skipped 1955 statements of 19 ops, all reverse ops). the gate, in two
  runs: `tests/itf1788` 14333 passed in 50 s; the rest 3710 passed in 529 s (a shared, loaded laptop); 18043 in all

**M13h reductions (done 2026-09-26)**. `sum_nearest`, `sum_abs_nearest`, `sum_sqr_nearest`,
`dot_nearest`, 1 each as counted 2026-09-25 by the old parser, which saw only the first statement of
each of the file's 4 testcases; M13a's parser reads the 11 it dropped (2026-09-26): 3, 3, 3 and 6
* `intervals/reductions.py`: `sum_`, `sum_abs`, `sum_sqr`, `dot` over sequences of numbers, the
  exact value through `Fraction` then rounded once, to nearest by default. point ops, not interval
  ops, so no M14 properties beyond a random differential against `Fraction` arithmetic
* done 2026-09-26. built:
    * `intervals/reductions.py::sum_`, `::sum_abs`, `::sum_sqr`, `::dot`, exported from `intervals`
      (M13e's precedent for the reverse ops; `tests/test_applicator.py::test_package_exports_unchanged`,
      which pins `intervals.__all__`, lists the four now). any iterable of real numbers (int, Fraction, float,
      mixed; `bool` and non-reals are a `TypeError`, as in `cuts.py::normalize_value`); each operand
      is held exactly, the value summed as a Fraction and rounded once by
      `rounding.round_rational`. the result is always a float, never `-0.0`; `sum_([])` is `0.0`
    * **choices the plan left open** (conservative, flagged in the decision log): the direction is a
      keyword-only `rounding='nearest'`, with `'down'` and `'up'` for the largest double below and
      the smallest above (a string, since no public API had a direction before; `rounding.py`'s
      `DOWN`/`NEAREST`/`UP` stay internal). ±inf are points: a sum reaching one infinity is that
      infinity in every direction, `sum_abs`/`sum_sqr` of anything infinite is `inf`. where 1788
      answers `NaN` (a `NaN` operand, `inf + -inf`, `0 * inf` in `dot`) ours **raises
      `ValueError`**, following D9's rule for `mid` of the empty set and the constructors' `nan`
      rule, rather than returning a float `nan`; unequal lengths in `dot` are a `ValueError` too.
      no warning is emitted
    * adapter (`tests/itf1788/test_itf1788.py`): the four ops in `OPS`, and a **reduction rule**
      (`::REDUCTIONS`, `::_reduce`): the result must already be a float and is compared as it is,
      not through the number rule's `round_up` (which would hide a wrong direction), and a
      `ValueError` from the op is 1788's `NaN`. no divergence row
    * tests (`tests/test_reductions.py`, 20 items in 13.0 s, 2026-09-26): 5 `@given`: the sums
      and `dot` against the exact value computed with Fraction, checked by `::is_rounded` from the
      definition on the result's neighbouring doubles (ties to even, ±inf at ±2**1024 for nearest),
      not by the package's rounding; float sums against `math.fsum`; special values (±inf, `nan`,
      0 against ±inf) for the sums and for `dot`, each error by its message. the 15 itf1788
      vectors are `@example`s, with the exact tie past `MAX`, `[0.1, 0.1, 0.1]` (a tie) and
      overflow by direction; plus the errors, keyword-only `rounding`, `-0.0`, iterables
* evidence, measured 2026-09-26 (census by importing the test module): the 15 reduction vectors
  all match, 0 rows; all ops: 4782 vectors of 58 ops, 4120 interval-valued, 8902 vector test items,
  49 divergence keys, 0 unknown failures; skipped 4760 statements of 53 ops
* sabotage (section 2), each red, then `reductions.py` restored from a copy and `cmp`-checked: the
  direction ignored (always nearest) 5 red (both differentials, both special-value properties,
  `test_overflow_by_direction`); each term rounded to a float before summing 7 (the differentials,
  fsum, `test_vector[libieeep1788_reduction.itl:45]`, the `2**104` dot); `inf + -inf` answered
  4 (`reduction.itl:27` among them); `0 * inf` answered 4 (`:50`, `:51`); `sum_abs` without `abs` 6
  (`:31`, `:33`); the `nan` check dropped 3 (both special-value properties and `test_errors`: the
  adapter alone does not see this one, since `Fraction(nan)` raises a `ValueError` of its own)

**every sub-task**
* its ops' vectors pass in both passes (plain and, if interval-valued, outward) or are divergence
  rows with a reason from `REASONS`; a new category needs an owner decision (D13's, D16's and D18's two are the only ones
  approved so far) and a line in `v2-plan.md` "ieee 1788"
* its ops get the M14 properties the day they land, sabotage per section 2
* the D rows it implements move into `v2-plan.md` "current design", the README's feature list and
  the `ieee 1788` counts are re-measured and dated, and this section records what was built, as
  M12's does

**exit for M13: no statement of the 19 files is skipped.** `SKIPPED` is empty and a test asserts
it, so a file that gains an op cannot quietly add skips

**exit (done 2026-09-27, on branch `m13-merge`: `m13e` at `6453703` merged with `m13g` at `7e6681c`)**
* the merge: the conflicts were insertion points, resolved keeping both sides, M13e's first
  (`intervals/__init__.py` and `tests/test_applicator.py::test_package_exports_unchanged`, the union
  of the exports; `tests/itf1788/test_itf1788.py`'s `OPS`, `REASONS`, `DIVERGENCES`, docstring;
  README and `v2-plan.md`). the one semantic conflict, as M13g's review predicted: the textual merge
  put M13e's `PAIRS` branch of `run_outward` inside `::_outward_hull`, where there is no `vector`;
  it is back in `::run_outward`. at the merge commit `tests/itf1788` had 2 red by design, M13g's
  reminders `::test_decorated_vectors_run_decorated` and `::test_no_decorated_pair_goes_unchecked`
* **the decorated reverse ops** (M13g's hook): given a `DecoratedInterval` operand, each reverse op
  is the core's set on the intervals, decorated trv (`intervals/reverse.py::_reverse`,
  `::_decorated`, `decorated.py::_trivial`); a bare `MultiInterval` beside a decorated operand is
  a `TypeError`, but for the omitted `x`. in the adapter `::REVERSE` joins `::PROPAGATED`, and
  `::_pair_outcome` compares a decorated pair as (pieces, decoration), 1788's decoration being its
  non-empty intervals'. every decorated reverse vector runs through the wrapper, its decoration
  compared in both passes: 481 (2026-09-27), 174 of them `mulRevToPair` pairs; the 4 with a `[nai]`
  operand are rows (D16). pinned by `::test_decorated_reverse_vectors_are_checked` and the rewritten
  `::test_no_decorated_pair_goes_unchecked` (172 checkable pairs, both passes)
* **new rows, 52, under "decoration expectations"** (approved category): 1788 decorates
  mulRevToPair's first interval as the decorated division `c / b` where `0 ∉ b` (6 com, 41 dac, 5
  def in `libieeep1788_mul_rev.itl`), while its mulRev, the same set's hull, is trv there
  (`libieeep1788_rev.itl:988`); M13g's "all 459 decorated results are trv" missed these. ours is
  one op, `mul_rev`, trv. the rows are on the decoration alone, in both passes
  (`::DECORATION_ONLY`, generated, keyed with the decorations since each bare twin matches);
  `::check` requires the set to match and only the decoration to differ, and
  `::test_divergence_rows` pins 52, each a pair vector with `0 ∉ b`. **owner question**: a pair op
  with 1788's decoration, or the rows as they are
* M14: `tests/test_propagation.py::test_each_reverse_op_is_trv` (the 10 ops, x omitted or given,
  decorated grid sets: the core's set, its class, trv), `::test_a_reverse_op_keeps_the_class_and_refuses_a_bare_set`,
  `::test_a_reverse_op_warns_once`; a doctest in `reverse.py`
* **the exit test**: `tests/itf1788/test_itf1788.py::test_nothing_is_skipped` asserts `SKIPPED`
  has no statement in any of the 19 files. it replaces the two interim pins, which the merge had
  combined silently (M13e's `test_only_m13g_ops_are_skipped` with `_M13G_OPS`, M13g's
  `test_only_the_reverse_ops_are_skipped`, each with its own `_REVERSE_OPS`); both removed
* one merge artefact outside the conflicts: `tests/test_reverse.py::test_trig_rev_is_tighter_than_the_vector`
  read the adapter's result for its 6 decorated copies as a hull, now (hull, decoration); it checks
  both trv and compares the hulls
* census, measured 2026-09-27 (`tools/itf1788_census.py`, which now also prints the
  decoration-only rows and the decorated reverse vectors): 19 files; 9542 vectors of 111 ops, 8306
  interval-valued (17848 vector test items, plus 334 float items); 1624 decorated vectors (1105 of
  61 propagating ops, 398 `BARE_PART`, 121 of the decorated ops; 1521 and 18 before the
  `is_decorated` fix of D18's day, which only changed this count); 185 keys (109 listed) with 271 vectors,
  0 unknown failures: 76 no NaI (79 vectors), 47 cancellations (94), 36 degenerate infinities
  (63), 11 tighter than the vector (18), 7 exact parsing decides validity (7),
  5 cut-based relations (7), 3 decoration expectations (3); plus 12 plain-only and 52
  decoration-only rows, all decoration expectations. skipped: 0 statements
* sabotage (section 2), 2026-09-27, a throwaway harness, each file restored from a copy and
  `cmp`-checked (equal), `HYPOTHESIS_PROFILE=ci`: the exit test: `powRev1` dropped from `OPS` 2 red
  (`::test_nothing_is_skipped`, and `::test_decorated_vectors_run_decorated`, as `PROPAGATED` holds
  `REVERSE`), `isNaI` dropped 1 and `mid` dropped 1 (`::test_nothing_is_skipped` alone). the
  decorated reverse ops: each of the 10 decorated newDec instead of trv 1 red (`mul_rev` 2), the
  dispatch in `_reverse` dropped 12, a bare set accepted 1, `x` ignored 7; the adapter: `REVERSE`
  dropped from `PROPAGATED` 107, a pair's decoration dropped on both sides 107
* the two categories proposed at M13e and M13g, "tighter than the vector" and "exact parsing decides
  validity", were approved by the owner on 2026-09-27 (D18), with the 15 exact-com rows kept under
  decoration expectations and `set_dec` demoting; the `(PROPOSED)` markers are gone from `REASONS`
  and the rows' reasons, and the historical records above keep the word as they were written
* the gate, two runs on 2026-09-27: `tests/itf1788` 18245 passed in 45.5 s; the rest 3920 passed
  in 498.2 s (22165 in all), on a shared, loaded laptop

the order of the remaining sub-tasks, and the owner's open questions on the built ones (M13c,
M13f, M13h): `HANDOFF.md`.

### M14 fuzzing (open, added 2026-09-25; the fuzz job and the flint oracle built 2026-09-26)

owner request 2026-09-25: "it would be great if we had fuzzing eg hypothesis". hypothesis is
already in the gate: 81 `@given` tests across 13 files (counted 2026-09-25), 25 to 300 examples
each (recounted 2026-09-26, below). the gaps are depth, independence and breadth. M14 does not wait for M13; its first two items
land with M13a so that every later M13 op arrives with them
* **a fuzz job, because no run explores new inputs in CI.** GitHub Actions loads hypothesis's `ci`
  profile, which is derandomized: every CI run replays the same examples, so new inputs are only
  ever tried by a local gate run
    * register a `fuzz` profile in a new `tests/conftest.py`: randomized, `max_examples` about 100
      times the gate's (a multiplier over each test's own `@settings`, or `settings(max_examples=...)`
      in the profile with the per-test caps lifted; decide when built), no deadline, selected by
      `HYPOTHESIS_PROFILE=fuzz`. section 1's "the suite needs no conftest" is updated when it lands
    * a separate workflow, `.github/workflows/fuzz.yml`, on a weekly `schedule` and
      `workflow_dispatch`, **not** on push and not in the gate. it caches `.hypothesis/` (the
      example database) between runs with `actions/cache`, and on a failure uploads the database and
      the log as an artifact
    * each failure found is pinned as an `@example` on the gate test that found it
* **an independent oracle for the functions** (D14: `python-flint`). today
  `tests/test_elementary.py` checks the 19 functions against `decimal` (correctly rounded for exp,
  ln, log10, sqrt) and taylor series written in this repo for the trig functions, on 60 random
  points each, so trig has no independent check
    * install `python-flint` into the `intervals` env and add it to the `[test]` extra in
      `pyproject.toml` (CI installs `.[test]`, so it follows)
    * a new `tests/test_oracle_flint.py`: hypothesis draws a float or exact point, arb evaluates
      the function at about 200 bits as a ball proven to contain the true value, and the test
      asserts the ball is inside our enclosure (soundness) and each end of ours is within one ulp
      outside the ball (sharpness). every function M12 built and every function M13d adds
* **breadth where fuzz is thin** (done 2026-10-02: the record is "M14-breadth" below): `tests/test_outward.py` has 1 `@given`, `tests/test_steps.py` 3,
  `tests/test_fmt.py` 1 (the parse/format round trip), `tests/test_applicator.py` 1.
  `tests/test_extreme_floats.py` covers add, sub, mul, div, reciprocal, neg, abs and pow only:
  extend it to the functions, `minimum`/`maximum`/`fma`, `%` and `//`, and `OutwardMultiInterval`
* **every M13 op as it lands**: soundness (`f(x) ∈ f(A)` at sampled `x ∈ A`, exact and float, under
  identity and outward rounding), isotonicity, interior sharpness, and for a reverse op its defining
  property (`x ∈ rev(C, X)` iff `x ∈ X` and `f(x) ∈ C`, at sampled points), with the op's itf1788
  vectors as `@example`s. `cancel_minus`: `B + X ⊆ A`, and no sampled point outside `X` fits
* exit: the fuzz workflow exists and has run green once (a `workflow_dispatch` run is enough), its
  example count and time recorded here with a date; the flint oracle in the gate; every new
  property sabotaged once and seen red (section 2)
* **the fuzz job, built 2026-09-26** (`tests/conftest.py`, `.github/workflows/fuzz.yml`):
    * mechanism: a test's own `@settings(max_examples=N)` overrides any profile, so a profile alone
      cannot raise the count. the conftest registers a `fuzz` profile (randomized, no deadline,
      `print_blob`, `too_slow` suppressed as in `ci`) and loads a profile only when
      `HYPOTHESIS_PROFILE` is set (another name goes to `settings.load_profile`, which refuses one it
      does not know). under `fuzz`, `conftest.py::pytest_collection_modifyitems` rewraps each
      hypothesis test's settings with `max_examples` times `FUZZ_MULTIPLIER` (default 100),
      `derandomize=False`, `deadline=None`, once per function: 18 tests combine `@given` with
      `parametrize`, and per item their multiplier would be squared. with `HYPOTHESIS_PROFILE` unset
      the conftest loads nothing and the hook returns at once
    * counts from the decorators, 2026-09-26: at `c8d843d` 81 `@given` tests in 13 files, 48
      pinning `max_examples` (25 to 300) and 33 on the default 100; with `tests/test_oracle_flint.py`
      87, 54 and 33
    * workflow: a weekly cron (Mon 03:23 UTC) and `workflow_dispatch` with a `multiplier` input, not
      on push; Python 3.13 and `ci.yml`'s install; `permissions: contents: read`, runs never cancel
      each other; `timeout-minutes: 350`, under the hosted runner's 360. `.hypothesis/` is restored
      by prefix and saved under a fresh `run_id` key with `if: always()`
      (`actions/cache/restore` + `cache/save`, since plain `actions/cache` saves only on success and
      would drop the failing example); pytest runs through `tee fuzz.log` under `pipefail`; on
      `failure() || cancelled()` (a timeout is a cancellation) it uploads `fuzz.log` and
      `.hypothesis/` with `include-hidden-files: true`. checked only that the YAML parses
    * proof the multiplier works (a throwaway probe loading the real conftest, counting calls of an
      unpinned test, one pinned to 7 and a parametrized one pinned to 5): 100/7/5/5 with no conftest,
      locally and with `GITHUB_ACTIONS=true`, and identical with the conftest and the variable
      unset; fuzz ×3 300/21/15/15; fuzz ×100 10000/700/500/500 (20.7 s). `tests/test_outward.py`
      (pins 60) under fuzz ×2 reports `max_examples=120` per parameter. sabotage, each red: the
      multiplier dropped (100/7/5/5 under ×3), the per-function guard removed (45 per parameter,
      ×3 squared), the fuzz profile loaded when unset (the probe passed this at first; a check of
      the loaded profile's name made it red, locally and under `GITHUB_ACTIONS=true`)
    * cost, local, shared 12-CPU laptop, 2026-09-26: `tests/test_ops_properties.py` 91 passed in
      89.9 s; the same at `FUZZ_MULTIPLIER=10` 1 failed, 90 passed in 1155 s (12.8×, shrinking
      included; slowest the failing div test, 51 s). extrapolated, the suite at ×100 is about 2.2 to
      3.4 hours: under 350 minutes but with little room as M13 adds tests. `multiplier=10` is the
      cheaper first `workflow_dispatch`
    * **found on its first run, and fixed the same session (2026-09-26): a bug in the test's
      oracle, not the library**. `tests/test_ops_properties.py::test_sound_float_identity_rounding[div]`
      went red on a = `[1/3, 1/2)`, b = `[-inf, 2.75]`: `MI(Fraction(1,3)) / 2.75` is
      `[0.12121212121212122]`, while python's `Fraction(1,3) / 2.75` is `0.1212121212121212`, one
      ulp lower, because python rounds 1/3 to a float first. v2-plan.md "arithmetic" says a mixed
      exact/float pair is computed exactly and rounded once, so the library is right. the oracle
      now rounds a mixed add, mul or div once (`tests/oracles.py::_once`; two floats still go
      through python's float op). the example cannot be an `@example` (the test draws through
      `st.data()`), so it is pinned as `tests/test_ops_properties.py::test_mixed_pair_rounded_once`.
      sabotage: `_once` rounding the operands first turned the replayed example red again
    * the same session's baseline gate (randomized locally) found a second test-only bug:
      `tests/test_functions.py::test_nearest_holds_the_nearest_value_of_every_float[exp10]` compared
      a to-nearest result with the exact value where it is rational (`exp10(-1.0)` = 1/10, below
      the double 0.1 that ends the result) instead of with the nearest double the docstring promises.
      it now always uses `rounded(..., NEAREST)`, which rounds the rational value itself
    * the exit's green GitHub run, **met 2026-09-30**: run 36654816589 at `697abdd` (`master`, on push),
      x10, python 3.13: `33406 passed in 1765.10s (0:29:25)`, the job 29 min 41 s; its cache restore
      took `fuzz-hypothesis-36580954134-1`, the database the red run before it saved (holding the
      fuzz-rev-inf example, which it replayed). the runs before it, all x10 on GitHub: 36507253782
      (branch `fuzz-run`, red: fuzz-symmetry, 53 min 36 s), 36540588320 (`fuzz-run`, red:
      fuzz-floordiv-overflow, a test oracle, 56 min 4 s; it also showed the fuzz profile kept no
      database on CI, fixed 2026-09-29), 36580954134 (`master`, the first on push, red: fuzz-rev-inf,
      D26, 49 min 5 s). since 2026-09-29 the job runs on every push to `master` after the same run
      locally (`tools/prepush.sh`), not weekly (`v2-plan.md` "2026-09-29 revision: fuzz on push")
* **the flint oracle, built 2026-09-26** (`tests/test_oracle_flint.py`, 9 test functions, 122
  tests parametrised; python-flint 0.9.0 in the env and `python-flint>=0.9` in the `[test]` extra;
  `intervals/` unchanged):
    * per function, all 19 of M12, at a drawn float, int or Fraction point: soundness (DOWN ≤ value
      ≤ UP); sharpness (no double strictly between either end and the value, so each end is the
      correctly rounded bound, which a 1-ulp outward error already fails); NEAREST the correct one
      of the two against the midpoint (ties to even); a rational value must come from `exact()`,
      overlap arb's ball and be one closed point at set level, and an exact arb ball where ours says
      irrational also fails; at set level `MultiInterval(exact x).f()` is the open one-ulp piece,
      `OutwardMultiInterval(float).f()` a sharp enclosure closed only where attained, and
      `MultiInterval(float).f()` the nearest double
    * also: `log` with a drawn base and of exact powers; about 60 fixed hard points (`sin(1e22)`,
      `sin(MAX)`, `tan` at the float pi/2, `exp` both sides of overflow and underflow, subnormals,
      `atanh`/`acosh`/`asin` by their domain ends); domain ends and limits at ±inf (`atan(±inf)`
      against arb's pi/2); outside the domain an empty set with `DomainClippedWarning`; `atan2` at
      set level, exact and outward; `elementary.rounded_angle` and `elementary.floor_over_pi`. not
      covered: `elementary.compare`, an internal helper
    * how: a float enters arb exactly, a Fraction through `fmpq` as a ball containing it, and arb's
      `<`/`>` hold only when every point of both balls agrees, so each comparison is proven or
      undecided. undecided retries at 200, 1000, then 4000 bits (`test_oracle_flint.py::PRECISIONS`)
      plus the operand's bit size, so huge arguments reduce correctly; past that the example is
      rejected by `assume` and counted in `test_oracle_flint.py::UNDECIDED`. where arb cannot hold
      the value (`exp(1e308)` is `[+/- inf]`, `exp(-1e308)` straddles 0, `1 - tanh(1e308)`
      underflows), `exp`/`exp2`/`exp10`/`sinh`/`cosh` with |x| > 600 are compared through the log of
      the value and `tanh` with x > 20 through `-log(1 - t)`, both increasing, so still exact. a
      plain import: missing python-flint fails loudly, not as a skip
    * measured 2026-09-26: the oracle with `tests/test_elementary.py` 337 passed in 20.59 s; the
      oracle alone 122 passed in 10.3 to 18.8 s across runs (machine shared), 11.33 s under
      `HYPOTHESIS_PROFILE=ci`, 148 s under fuzz ×10 with 0 failures. undecided: 0 at default settings
      and 0 at fuzz ×10. share of drawn points with a rational value: sqrt 62.5%, exp2 47.5%, log2
      41.5%, log10 40.5%, exp10 37.5%, acos 30%, acosh 25%, log 19%, the rest 7.5 to 10%. rejected
      examples are filter or `assume` rejections only: atanh 20, atan2 24 (y = x = 0),
      rounded_angle 10. no library bug found, so nothing is pinned as xfail. PyPI (0.9.0): Python ≥
      3.10, cp310-abi3 wheels for win_amd64, manylinux x86_64 and macOS, plus cp313 and cp314,
      covering CI's 3.11 to 3.14
    * sabotage, each red, then restored and green: a throwaway edit of `elementary.rounded` moving
      sin's DOWN 1 ulp inward (11 failed, "unsound") or 1 or 2 ulp outward (11 each, "not sharp"),
      exp's UP 1 ulp inward (11, soundness) or 1 or 2 outward (10 each, sharpness), red in
      `test_point_against_arb[fn]`, the extreme points and the matching `tests/test_elementary.py`
      tests; monkeypatches: an irrational set-level end kept closed (75 failed), `exact('log2')` one
      too high ("log2(16) is not the rational 5"), `rounded_angle` 1 ulp inward (atan2 and
      rounded_angle fail on soundness), `floor_over_pi` off by one (the floor test fails)
* what is still open in M14: `HANDOFF.md` (M14-run, M14-breadth; each remaining M13 sub-task brings
  its own properties)

### M15 the solver stack's first part: `autodiff.py`, `solver.py` (H3; done 2026-09-27)

the owner, 2026-09-27: "do h3 first". H3 is the solver stack of `v2-plan.md` "later (not in
v2.0)"; its suggested first pick (`v2-plan.md` "2026-09-26 revision: owner answers") was forward-mode
autodiff with newton's method, "the demonstration of what multi-intervals are for". numpy and
gmpy2/mpfr stay out (owner 2026-09-26, recorded, not now); the direction tag was not needed (see
the design). the choices the build made are D19, open for the owner as `HANDOFF.md` Q11. the
design is `v2-plan.md` "the solver stack"; here the spec, the exit and the record.

* **`intervals/autodiff.py`**: `Dual(value, derivative)`, two `MultiInterval`s (either class) or two
  `DecoratedInterval`s; `Dual.variable`, `Dual.constant`, `derivative(f, x)`; the arithmetic
  dunders, `reciprocal`, `abs`, `**` (number, `Dual` or set exponent, and `number ** Dual`) and
  every elementary method of `MultiInterval` with its chain rule
* **`intervals/solver.py`**: `newton(f, x, *, tol, max_steps)` and `Root(interval, unique)`: branch
  and prune, a newton step `piece ∩ (m + mul_rev(F', -f(m)))` where decorations prove `f` C¹ on a
  bounded piece, the uniqueness proof (`solver.py::_newton_step`), bisection (by exponent on a
  piece spanning more than a factor of 16), exact zeros at a point or a closed end
  (`solver.py::_finish`)
* exit: every op of `Dual` against an independent oracle (arb's taylor series) for soundness and
  sharpness; newton sound (every zero enclosed) and its uniqueness claims true on polynomials with
  known zeros, under any `tol` and `max_steps`; the C¹ gate shown necessary by a function it
  saves; the gate green; every new property sabotaged once and seen red

record (2026-09-27):
* **what the build found on its way**, each fixed before the record: the quotient rule as
  `(u' - (u / v) v') / v` gave `1 / x` over `[-1, 1]` the derivative `[-inf, -1] ∪ [1, inf]` (sound,
  but the sign lost across the pole); `(u' v - u v') / v ** 2` gives `[-inf, -1]`. newton from far
  off a zero crept by a constant factor a step (`x ** 2 - 2` on `[-inf, inf]`: 743 evaluations, 2 s);
  splitting by exponent before any step on a piece spanning more than a factor of 16 made it 57
  (0.04 s). at a double zero newton converges linearly (3/8 a step at 0 for `x ** 2`) and ran into
  the subnormals past `tol` (5 s for `x ** 2 (x - c)`); stopping it at `tol` then left a simple
  zero one step short of its uniqueness proof, so newton goes on past `tol` for at most 8 steps
  (`solver.py::_PAST_TOL`). a zero on a split point (`cbrt(x) - 1` on `[0, 8]` splits at 1) is a
  closed end, where no newton set fits inside the interior: `_finish` outputs such an end alone
  when `f` is exactly 0 there
* **the gate found an old test-oracle bug** (not M15's, not the library's): the local gate's
  randomized `tests/test_kernel.py::test_normalize_is_canonical` drew the piece `(0, 5e-324)`, which
  the test re-expresses split at `(lo + hi) / 2`, here `0.0`, adding the point 0. the kernel was
  right. fixed in the oracle (a split only at a midpoint strictly inside), the example pinned; the old
  oracle is red on it. CI never saw it, its profile being derandomized
* **tests** (`tests/test_autodiff.py`, `tests/test_solver.py`, and the two modules' doctests):
  each of the 42 rows of `test_autodiff.py::OPS` at drawn points of drawn intervals against
  `arb_series` (value and derivative, soundness), the derivative at a point within 1e-10 relative
  (sharpness), random expression trees over the ops (200 examples), every op dac inside its domain
  and trv or def where it is not differentiable; newton on polynomials from 1 to 4 drawn zeros
  (ints, fractions, floats; with doubles and close pairs), with and without a budget, 1 to 3 zeros
  `n + 1/sqrt 2` each proved unique to within 1e-12, `sin` on `[-10, 10]` (7 zeros, each unique,
  within 2e-15 of `k pi`), the first step's split, `_newton_step`'s three conditions one by one,
  the non-C¹ example and its sabotage as a test (`::test_not_c1_would_lose_a_zero`), poles,
  unbounded and multi-piece input, warnings kept inside
* **measured 2026-09-27** (shared laptop): `tests/test_autodiff.py` 144 tests in about 15 s,
  `tests/test_solver.py` 21 in about 24 s; the gate numbers are in `HANDOFF.md`'s banner
* **sabotage** (a throwaway harness: each break alone, `.hypothesis` cleared, the four files' tests
  with `-x` and a 600 s timeout, the file restored and compared; 2026-09-27). the last column is
  the first test to fail under `-x`:

| break | first run | final run: red by |
|---|---|---|
| cos derivative sign | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[cos]` |
| product rule term dropped | red | red: `tests/test_autodiff.py::test_expression_encloses_value_and_derivative` |
| quotient rule sign | red | red: `tests/test_autodiff.py::test_expression_encloses_value_and_derivative` |
| sqrt factor | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[sqrt]` |
| abs derivative 1 | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[abs]` |
| tan derivative | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[tan]` |
| exp derivative loose but sound | red | red: `tests/test_autodiff.py::test_op_derivative_is_tight_at_a_point[exp]` |
| pow 0 derivative | red | red: `tests/test_autodiff.py::test_pow_zero_is_the_constant_one` |
| pow dual: log term dropped | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[pow self]` |
| log base factor | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[log base 3]` |
| acos sign | red | red: `tests/test_autodiff.py::test_op_encloses_value_and_derivative[acos]` |
| C1 gate removed | red | red: `tests/test_solver.py::test_not_c1_is_bisected_not_stepped` |
| C1 gate at def | red | red: `tests/test_solver.py::test_not_c1_is_bisected_not_stepped` |
| C1 gate: value decoration ignored | green | red: `tests/test_solver.py::test_a_jump_is_caught_by_the_value_decoration` |
| uniqueness: 0 in slope allowed | red | red: `tests/test_solver.py::test_newton_step_proves_uniqueness_only_without_zero_slope` |
| uniqueness: empty image allowed | green | red: `tests/test_solver.py::test_newton_step_proves_uniqueness_only_without_zero_slope` |
| uniqueness: piece not interior | red | red: `tests/test_solver.py::test_newton_step_proves_uniqueness_only_without_zero_slope` |
| step not intersected with the piece | red | red: a hypothesis failure in `tests/test_solver.py` (several examples) |
| division instead of mul_rev | red | red: `tests/test_solver.py::test_every_zero_is_enclosed` |
| range prune removed | red | red: `tests/test_solver.py::test_sin_zeros` |
| degenerate always unique | green | red: `tests/test_solver.py::test_a_point_is_unique_only_when_f_is_exactly_zero` |
| exact-ends rule removed | red | red: `tests/test_solver.py::test_sin_zeros` |
| no newton past tol | red | red: `tests/test_solver.py::test_close_zeros` |
| past-tol cap removed (multiple zero runs on) | green | red: `tests/test_solver.py::test_evaluation_budgets` |
| tol ignored | green | red: `tests/test_solver.py::test_evaluation_budgets` |
| magnitude split before newton removed | green | red: `tests/test_solver.py::test_evaluation_budgets` |
| magnitude split across 0 removed | green | red: `tests/test_solver.py::test_evaluation_budgets` |
| budget drops the stack | red | red: `tests/test_solver.py::test_every_zero_is_enclosed_on_a_budget` |
| split keeps one piece | red | red: a hypothesis failure in `tests/test_solver.py` (several examples) |
| bisection drops the split point | red | red: a hypothesis failure in `tests/test_solver.py` (several examples) |

the seven green in the first run were gaps, each closed by a test added the same session and the
break re-run red: the value's decoration (every derivative formula of `Dual` already carries its
op's domain, so only a hand-made `Dual` shows it: `::test_a_jump_is_caught_by_the_value_decoration`),
the empty newton set and the exact point (`::test_newton_step_proves_uniqueness_only_without_zero_slope`,
`::test_a_point_is_unique_only_when_f_is_exactly_zero`), and four rules that cost time, not
soundness, pinned by evaluation counts (`::test_evaluation_budgets`, 57, 57, 30, 5 and 49
evaluations on 2026-09-27, against bounds of 80 and 10; without the magnitude split 743, without
the cap past tol 922)

### M16 the solver stack's second part and the rest of H3 (done 2026-09-28)

the owner, 2026-09-27: "get the rest of h3 done", which superseded 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now" (M15 above: "gmpy2/mpfr stay out"). H3's rest was built as M16, H3's
second part, in five streams on five branches, each off `v2` at `04946af`: M16a nd-solver
(`h3-nd-solver`: `gradient`, `jacobian`, `solve`, `RootBox`; D20, `HANDOFF.md` Q12), M16b ieee1788
(`h3-ieee1788`: `intervals/ieee1788.py`, the 1788 layer; D21, Q13), M16c allen-matrix
(`h3-allen-matrix`: `allen_matrix`, `allen_relations`; D22, Q14), M16d numpy (`h3-numpy`:
`intervals/numpy_compat.py`; D23, Q15) and M16e gmpy2 (`h3-gmpy2`: `intervals/backend.py`,
`intervals/_gmpy2.py`, opt-in, pure by default; D24, Q16). each stream was designed by one agent
and critiqued by an adversarial one, then built by a builder with properties and sabotage,
reviewed by three read-only reviewers (lenses soundness, sabotage audit, spec/regression), fixed by
a fixer that reproduced each finding first, and checked by a verifier, in its own worktree. the
five were merged into `v2` on `h3-merge`, without conflicts; the gate on the merged tree is in
`HANDOFF.md`'s banner. the designs are `v2-plan.md` "the solver stack", "the 1788 layer",
"comparisons", "numpy" and "elementary and step functions" (the backend); here each stream's spec,
exit and record. each stream's gate numbers below were measured on its own branch, with the five
sharing the laptop, so every time is loaded.

**M16a the solver in several variables: `gradient`, `jacobian`, `solve`, `RootBox` (done 2026-09-28)**
(D20). the design is `v2-plan.md` "the solver stack" (its M16a bullets); the choices are D20, open
as `HANDOFF.md` Q12.

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

record (2026-09-28):
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
  * **the rules measured on the prototype** (the design's, 2026-09-27/28, the throwaway
    `.scratch/h3b/nd-solver/proto.py` with one switch per rule, shared laptop; calls of `F` of every
    kind; the build re-measured the defaults above, not these). each rule off, the rest the design's
    final rules (the defaults: circle and line 71 calls on `REALS²` and 103 on `[-1e300, 1e300]²`, the
    cusp 359, system 7 31, the kink 102, `(sin(x + y), x - 2y)` 148):
    * the preconditioner off (`Y = I`): circle and line 225 and 248 calls, neither zero proved; the
      cusp 1344; `sin(x + y)` 564, 1 of 3 proved; `(exp x - y, x + y - 2)` 175, not proved
    * the magnitude split off for bounded components: `[-1e300, 1e300]²` 10250 calls in 158 s
    * the past-tol cap off: the cusp 5100 calls in 73 s
    * the exact midpoint off: system 7 163 calls and 3 boxes (1 unique), against 31 and 1. it was
      found on a proved box whose x had narrowed to two adjacent doubles, then bisected in y into two
      unproved halves; "a unique box is never bisected" closes that case too, but only for proved
      boxes, and with the exact midpoint it is unreachable, so it is not written
    * gauss-seidel with `/` for `mul_rev`: the cusp 5 calls and 1 box, a zero lost; the hull where
      `mul_rev` gives two pieces: sound, a cost only (the cusp 391)
    * the jacobian over the box, not its closed hull: every count the same but the kink's, 58 calls
      against 102, both zeros still proved. the closed hull is kept for krawczyk's theorem (a compact
      box), not for a measured gain, and is pinned directly (`::test_c1_is_decided_on_the_closed_hull`)
    * the design's first rule for zeros at simple rationals was a **vertex rule** (every closed vertex
      of an unproved box evaluated, up to 2 ** n calls): with it and no simplest point, the cusp had 0
      of its 2 zeros proved, the kink 1 of 2, the pole 1 of 2 (the other coordinate had open ends from
      a step); the simplest point, one call a box, proved 2, 2 and 2. the vertex rule was dropped
    * the mean value form as an extra prune (`F(m) + J (X - m)`): 889 calls against 897, 524 against
      524, 497 against 499 (three constructed systems, the prototype's earlier rules); going on while
      *any* component halves, not the widest: 883, 543, 489 against 897, 524, 499, and 1849 against
      1862. neither taken
    * bisecting off-centre (at `63/128` of the width), before the simplest point existed: constructed
      system 3 had 3 of its 4 zeros proved (943 calls) against 1 (897), the faces of the magnitude
      split at 0 and ±2 ** k remaining; with the simplest point and centred bisection all 4 are
      proved (988 calls). not taken (Q12(c))
    * a profile (2026-09-27, one constructed system, 426 calls of `F` in 95 s under cProfile): 81 s in
      `MultiInterval._binary` → `applicator._apply` (20784 binary ops, about 4 ms each: exact corner
      products in Fraction, then rounding), 54 s of the 95 in the decorated passes. the library's
      arithmetic is the cost, not the solver's bookkeeping (Q12(a); `HANDOFF.md` evaluate-box)
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
* **the direction tag, not built**: the argument is in `v2-plan.md` "the solver stack", corrected per critique B4 (an
  enclosure may hold an open end at inf by overflow, which keeps every real point; no degenerate
  `[±inf]` arises from a finite real), with the overflow box as a test (`::test_overflow_box`:
  `(exp x - y, x - 709.5)` over `[700, 720] × [1e307, 1.7e308]`, the zero enclosed)

review (2026-09-28, three read-only reviewers over `c8c9e08`, lenses soundness, sabotage-audit and
spec/regression; each finding reproduced on `c8c9e08` before any change, by
`.scratch/fix/repro.py` in the worktree (gitignored) or by the red run of its closing test; ids are
the reviewers', the soundness and spec lenses both using F1 to F3). **no wrong answer in the
solver**: every finding is a false claim in the text, a crash, a cost, or a property the tests did
not pin. found and fixed:

* **soundness F1 (blocking, a false claim)**: `autodiff.py::gradient`'s docstring and `v2-plan.md` "the solver stack"
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
  568 s; `tol` 0.25, 0.1, 0.05 give 34, 688, 3312 boxes (280, 5548, 25924 calls; 2.4 s and 28 s for the last two). **the cascade the finding describes does not
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
  (O(inf), O(2))) == O(2)`), and the docstring narrowed to an entry `[±inf]`; under the break the closing assertion goes red first by the `IndeterminateResultWarning` the suite turns into an error, before the `== O(2)`; the equality alone fails too (`{}` against `[2.0]`; the verifier, 2026-09-28)
* **spec F1 (minor, an overstated exit)**: the random oracle's `abs` and `cbrt` factors run only
  under a budget and do not detect the C¹ gate removed. the exit line and the cost bullet above now
  say so; the gate stays pinned by the pole, kink, coupled kink and jump examples (the sabotage
  table)
* **spec F3 (minor)**: the README's package-layout line was missing from the record's readme text; added (README "layout")
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

the reviewers' evidence beyond their findings (2026-09-28, over `c8c9e08`; their probes under
`.scratch/h3b/review/nd-solver-soundness/` and `nd-solver-sabotage/`, gitignored):
* soundness: the jacobian of a 3 x 3 system (`sin`, `exp`, `/`, `atan`, `sqrt`, `log`, `** 3`) over
  300 random boxes (bare, decorated, a number beside sets), 5 arb points each: 0 misses. 60 seeds of
  systems with zeros on and beside the split faces, close pairs 2 ** -8 to 2 ** -44 apart, coarse
  `tol`, rotated: 0 wrong, 75 unique boxes checked (405 s). 60 random 2-d systems `(P(x) + s Q(y),
  Q(y))` in rotated coordinates, every zero known by arb: 0 wrong, 65 unique boxes (1009 s).
  `(x ** 2 - 2, y ** 2 - 3, z - x y)` on `[-3, 3]² × [-4, 4]`: 4 unique boxes, each holding its
  zero. edge inputs (empty, points, unbounded, open, multi-piece, Fraction ends, subnormal, huge,
  tiny boxes, `tol=0`, `tol<0`, `max_steps=0`, a log outside its domain, a singular zero): no wrong
  answer. its own breaks, each red by `::test_every_zero_is_enclosed`: the simplest point not
  checked inside its component (the row above), the solver's jacobian transposed, `b = F(m)` for
  `Y F(m)`, `F(m)` to nearest, `J` to nearest
* sabotage audit: 7 rows of the table re-run with the builder's replacements (the four green at
  first, the inflation clip, `int H`, the exact zero), each red by the test the table names; 25
  breaks of its own, 18 red at once (the solver's jacobian transposed or seeding x_0 in every pass, a
  constant output's partial `[1]`, the gate accepting def, krawczyk dropping the identity or skipping
  the last row, gauss-seidel `+ b` or a split forgetting the rows before, `_inverse` keeping a
  non-finite inverse, `_choose` narrowest first, `max_steps` off by one or dropping the current box,
  a box with one degenerate component taken as a point, `_simplest_between` losing a negative sign,
  the simplest point unchecked inside its component, its rest boxes overlapping or the lower piece
  given the upper side's region, the inflation clipped below only) and 7 green: S1 to S6 above

**M16b the 1788 layer: `ieee1788.py` (done 2026-09-28)** (D21).

the owner, 2026-09-27: "get the rest of h3 done". the layer is H3's "thin `ieee1788.py`" and
Q9's pair op; the design is `v2-plan.md` "the 1788 layer", the choices D21 (owner question Q13).

* **`intervals/ieee1788.py`**: `Interval(lo, hi, decoration)`, `from_set`, `Overlap` (16 states),
  `NAMES` (104 names: 1788's 102 and itf1788's `d-numsToInterval`, `d-textToInterval`), and the
  104 functions: the constructors (`nums_to_interval`,
  `text_to_interval`, their decorated twins, `empty`, `entire`), `new_dec`, `set_dec`,
  `interval_part`, `decoration_part`, every forward op and elementary function the library has, the
  step functions, the reverse ops (`x` optional) and `mul_rev_to_pair`, `cancel_minus`,
  `cancel_plus`, `intersection`, `convex_hull`, the numbers, the booleans, `overlap`, and the
  library's reductions re-exported. no library module is edited
* **`tests/itf1788/test_ieee1788.py`**: the third conformance pass, exact, with its own rows (the
  adapter's under three categories)
* **`tests/test_ieee1788_layer.py`**: the properties
* exit: every vector matches through the layer or is a row of the three owner-approved categories
  (no row on a decoration alone, none for cancellation, relations or infinities); every function the
  library's op through the output rule, in 1788's form, on drawn operands of both flavours; the
  output rule against its definition; the gate green, one-process collection clean; every new
  property sabotaged once and seen red

record (2026-09-28):
* **what the build found on its way**:
  * measured before the build (2026-09-27, a throwaway prototype of the layer at `04946af`, every
    vector compared exactly, operands the literals' doubles): 9187 of the 9207 vectors with no
    adapter row matched (7877 keys), the other 20 a `ValueError` where 1788 says `NaN` (12 the
    numbers of `[empty]`, 8 the reductions), which the pass reads as `NaN`; every row of the
    adapter's cancellation (47 keys, 94 vectors), cut-based (5, 7), degenerate-infinity (36, 63) and
    decoration categories (52 `DECORATION_ONLY`, 12 `PLAIN_ONLY`, 3 `_BOUNDED_EXACTLY`) matched, and
    the 94 keys the pass keeps did not. each rule off alone brought its rows back (1788's "no
    answer" the 94 cancellation vectors, the overlap table the 7, the pair's decoration the 52),
    except the drop of attained infinities: the reverse ops' omitted `x` as 1788's entire dissolves
    the same 26 `pownRev` keys (52 vectors), so either alone suffices and only both off shows them,
    and a lone `[±inf]` result (the 10 `log`/`atanh` keys, 11 vectors) is emptied even then by the
    1788-form constructor (`OutwardMultiInterval(-inf, -inf, start_closed=False)` is empty). so the
    build drops explicitly and has no default-`x` rule, pinned by
    `tests/test_ieee1788_layer.py::test_the_drop_is_explicit`
  * before the build, the critique put 29 edge boxes of its own (poles, domain ends, step jumps,
    overflow: `pow`, `atan2`, `sign`, `floor`, `ceil`, `trunc`, the two roundings, `recip`, `log`,
    `atanh`, `tan`, `pown`, `exp`, `mul`, `div`, `sqrt`, `acosh`, `coth`, `rootn`) through the
    prototype layer, each against 1788's decoration worked out by hand: all 29 gave it (2026-09-27)
  * the pass matched on its first run with the module (9542 vectors; the rows exactly the 94 keys,
    104 vectors, the design's probe predicted), so the prototype's measurement carried over. the
    rows' count is pinned (`tests/itf1788/test_ieee1788.py::test_rows`) and regenerated by
    `tools/itf1788_census.py`
  * a number beside a decorated operand must become newDec's point **before** any op of the layer
    runs on it: `cancel_plus(x_dec, 2.0)` as `cancel_minus(a, neg(b))` with `neg(2.0)` bare would
    be a mixed call and a `TypeError`. so `cancel_plus` and `mul_rev_to_pair` take the flavour first
    (`ieee1788.py::_interval_of`), pinned by `tests/test_ieee1788_layer.py::test_flavours`
  * the critique's leak test is written against the filters, not with `pytest.warns`: `pytest.warns`
    records under its own `catch_warnings` with `simplefilter('always')`, which outranks any filter a
    leaking layer left behind. `::test_the_silencing_does_not_leak` compares `warnings.filters`
    before and after layer calls and lets the suite's `error::IntervalWarning` filter turn a direct
    library call's `DomainClippedWarning` into an error; its break ("silencing without
    catch_warnings") is red there
  * 1788's `pown` takes an integer, D11's `**` an integral real: the function `pown(x, 2.0)` is a
    `TypeError` (as the library's `pown_rev` refuses a float), the operator `x ** 2.0` is `pown(x,
    2)` and `x ** 0.5` `pow_(x, 0.5)`; the pass keeps an `int` exponent an `int` and makes a
    `Fraction` its float (critique n2)
  * the module's docstring examples were first written with outputs from the design's text; two
    were wrong (`Interval(1, 2, 'com') * Interval(-1, inf, 'dac')` is `[-2.0, inf]_dac`, and
    `mul_rev_to_pair([-2, -0.1]_dac, [-2.1, -0.4]_dac)`'s first member is `[0.2, 21.0]_dac`, both ends
    exact doubles) and were corrected to the computed values
* **tests**: `tests/itf1788/test_ieee1788.py::test_vector` (9542 items), `::test_rows` (94 keys by
  category, 104 vectors, none on a decoration alone), `::test_every_op_is_mapped` (every vector op
  but `isNaI` reaches `NAMES`, and `NAMES` less those is exactly `{'empty', 'entire'}`),
  `::test_the_comparison_is_exact` (one double wider at either end, a bare result for a decorated
  vector, an int end, an open finite end, a non-float number, a `midRad` member that is not a
  float, a set of another type than `OutwardMultiInterval`: each rejected), `::test_a_stale_row_fails`,
  `::test_an_escaped_warning_fails`, `::test_the_nan_reading_is_narrow`,
  `::test_operands_keep_int_exponents`; `tests/test_ieee1788_layer.py`:
  `::test_each_function_is_the_library_op_in_1788_form` (75 rows of `::LIBRARY`, every lifted
  function and each reverse op with `x` omitted and given, both flavours drawn),
  `::test_from_set_is_the_output_rule`, `::test_from_set_examples`, `::test_the_drop_is_explicit`,
  `::test_flavours`, `::test_a_call_of_numbers_alone_is_bare`, `::test_library_values_are_not_operands`, `::test_operators_refuse_library_values`,
  `::test_cancel_minus_is_1788s`, `::test_overlap_on_the_grid`, `::test_overlap_examples`,
  `::test_mul_rev_to_pair_bare`, `::test_mul_rev_to_pair_decorated`, `::test_mul_rev_to_pair_examples`,
  `::test_numbers`, `::test_the_sign_of_zero_is_1788s`, `::test_numbers_examples`, `::test_booleans`,
  `::test_booleans_examples`, `::test_is_member`, `::test_constructors`, `::test_constructor_refusals`,
  `::test_decorations`, `::test_set_dec_and_new_dec`, `::test_the_library_s_warnings_are_silent`,
  `::test_possibly_undefined_operation_reaches_the_caller`, `::test_the_silencing_does_not_leak`,
  `::test_names`, `::test_not_exported`, `::test_repr_str_and_to_set`, `::test_repr_and_str_examples`,
  `::test_operators_are_the_functions`, `::test_the_class`, `::test_exponents_are_ints`,
  `::test_rootn_of_degree_0_raises`; and the module's doctests
* **measured 2026-09-28, after the review's fixes** (shared laptop, five H3 streams running at
  once, so every time is loaded): the pass, `python -m pytest -q tests/itf1788/test_ieee1788.py`,
  9549 passed (9542 vectors and its seven rules) in 38 s; the properties and the module's doctests,
  `python -m pytest -q tests/test_ieee1788_layer.py intervals/ieee1788.py`, 292 passed in 22 s
  (287 at `027d8e6`, before the review). the pass's rows: 94 keys, 104 vectors (76 no NaI, 11
  tighter than the vector, 7 exact parsing), `python tools/itf1788_census.py` (its new last line,
  "layer pass rows"). the gate in three calls from the worktree root: `pytest -q tests/itf1788`
  27795 passed in 133 s (18246 at M15, plus the pass's 9549); then 14 files named one by one,
  `tests/test_applicator.py`, `test_autodiff.py`, `test_cancel.py`, `test_cuts.py`,
  `test_decorated.py`, `test_elementary.py`, `test_errors.py`, `test_extreme_floats.py`,
  `test_fmt.py`, `test_functions.py`, `test_ieee1788_layer.py`, `test_kernel.py`,
  `test_literals.py`, `test_minmax_fma.py`, 1748 passed in 337 s; and everything else,
  `pytest -q --ignore=tests/itf1788` with an `--ignore` for each of those 14 (so the rest of
  `tests/`, the modules' doctests and `README.md`), 2632 passed in 472 s: 32175 passed, 942 s
  (32169 at `027d8e6`), and `pytest --collect-only -q` over the whole tree in one process collects
  the same 32175 (no basename clash). re-run after the second look (2026-09-28, same three calls,
  each rc=0): 27795 passed in 151 s, 1748 in 366 s (and 328 s in a second, lone run of that call),
  2632 in 493 s: 32175 passed, 1010 s; the collection again 32175. a leftover gate run of the
  verifier's, still going beside this one and writing the same log files, ended its second call
  with rc=1, and its output was lost (its log was deleted and overwritten mid-run); it ran on this
  tree less the one test edit below, and both later runs of that call on the final tree are
  green, so it is recorded here and not explained; a likely cause, found after the merge: `tests/test_literals.py::test_any_text_is_an_interval_or_undefined`, in that call, asserted `result.is_contiguous`, false for `text_to_interval('[]')`, the empty literal, so it failed whenever hypothesis drew `'[]'` (pre-existing at `04946af`, a test-oracle bug; found by M16e's verifier). fixed 2026-09-28 at the merge: `@example('[]')`, red on the old assertion, and the assertion is now `result.is_empty or result.is_contiguous`
* **files**: `intervals/ieee1788.py`, `tests/itf1788/test_ieee1788.py`, `tests/test_ieee1788_layer.py`
  (new); `tools/itf1788_census.py` (prints the layer pass's rows). no library module and no existing
  test was edited, so the library's behaviour is unchanged
* **sabotage** (a throwaway harness copied from M15's: each break alone, `.hypothesis` cleared, the
  layer's files with `-x` (the module's doctests, `tests/test_ieee1788_layer.py` and the pass's own
  rules), then the pass's `::test_vector` with `-x`, the file restored and compared; 2026-09-28).
  the last column is the first test to fail under `-x`, the layer's files first:

| break | first run | final run: red by |
|---|---|---|
| from_set without step 1 (no drop of ±inf points) | red | red: `tests/test_ieee1788_layer.py::test_from_set_is_the_output_rule`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_rev.itl:217]` |
| from_set rounds to nearest, not outward | red | red: `tests/test_ieee1788_layer.py::test_from_set_is_the_output_rule`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_class.itl:73]` |
| from_set leaves a finite end open | red | red: the collection of `tests/test_ieee1788_layer.py` (the class's `__debug__` invariant assertion, building `::GRID`); the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:25]` |
| from_set without the newDec cap | red | red: `tests/test_ieee1788_layer.py::test_from_set_is_the_output_rule` (drawn; at the build first red at `::test_from_set_examples[s7-want7]`); the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_class.itl:165]` |
| cancellation without 1788's "no answer" | red | red: `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_cancel.itl:28]` |
| cancellation wid a > wid b for >= | red | red: `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_cancel.itl:67]` |
| cancellation: a empty, b unbounded gives empty | red | red: `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_cancel.itl:33]` |
| overlap: meets and metBy as the library's allen (overlaps) | red | red: `tests/test_ieee1788_layer.py::test_overlap_on_the_grid`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_overlap.itl:37]` |
| overlap: starts/startedBy swapped | red | red: `tests/test_ieee1788_layer.py::test_overlap_on_the_grid`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_overlap.itl:43]` |
| pair decorated trv always | red | red: `tests/test_ieee1788_layer.py::test_mul_rev_to_pair_decorated`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_mul_rev.itl:223]` |
| pair decorated as the operands also when 0 in b | red | red: `tests/test_ieee1788_layer.py::test_mul_rev_to_pair_decorated`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_mul_rev.itl:224]` |
| pair pieces in decreasing order | red | red: `tests/test_ieee1788_layer.py::test_mul_rev_to_pair_bare`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_mul_rev.itl:32]` |
| pair as one hull (1788's mulRev) | red | red: `tests/test_ieee1788_layer.py::test_mul_rev_to_pair_bare`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_mul_rev.itl:32]` |
| a bare operand beside a decorated one accepted | red | red: `tests/test_ieee1788_layer.py::test_flavours[cancel_minus]` (the pass green) |
| a number always made bare | red | red: `tests/test_ieee1788_layer.py::test_flavours[add]` (the pass green) |
| the layer's silencing removed | red | red: `tests/test_ieee1788_layer.py::test_each_function_is_the_library_op_in_1788_form[pos I]`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:26]` |
| silencing the IntervalWarning base class | red | red: `tests/test_ieee1788_layer.py::test_possibly_undefined_operation_reaches_the_caller` (the pass green) |
| silencing without catch_warnings (leaks) | red | red: `tests/test_ieee1788_layer.py::test_the_silencing_does_not_leak` (the pass green) |
| numbers not made float | red | red: `tests/test_ieee1788_layer.py::test_numbers`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_num.itl:97]` |
| inf/sup of empty raise | red | red: `tests/test_ieee1788_layer.py::test_numbers`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_num.itl:26]` |
| numbers of empty return nan | red | red: `tests/test_ieee1788_layer.py::test_numbers` (the pass green) |
| Interval(lo, hi, d) demotes instead of raising | red | red: `tests/test_ieee1788_layer.py::test_constructor_refusals[args1-UndefinedOperationError]` (the pass green) |
| text_to_interval hulls to nearest | red | red: `tests/test_ieee1788_layer.py::test_constructors`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_class.itl:73]` |
| a NAMES entry bound to the wrong function (sub -> add) | red | red: `tests/test_ieee1788_layer.py::test_each_function_is_the_library_op_in_1788_form[sub II]`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:132]` |
| a NAMES entry missing (hypot) | red | red: `tests/test_ieee1788_layer.py::test_each_function_is_the_library_op_in_1788_form[hypot II]`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[mpfi.itl:812]` |
| a NAMES entry no vector reaches (exp2m1) | red | red: `tests/test_ieee1788_layer.py::test_names` (the pass green) |
| to_set of a decorated interval returns the bare set | red | red: `tests/test_ieee1788_layer.py::test_from_set_is_the_output_rule`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:40]` |
| library values accepted as operands | red | red: `tests/test_ieee1788_layer.py::test_library_values_are_not_operands[multi-add]` (the pass green) |
| inf of a zero lower end is +0.0 | red | red: `tests/test_ieee1788_layer.py::test_the_sign_of_zero_is_1788s` (the pass green) |
| sup of a zero upper end is -0.0 | red | red: `tests/test_ieee1788_layer.py::test_the_sign_of_zero_is_1788s` (the pass green) |
| repr of an infinite end does not evaluate back | red | red: `tests/test_ieee1788_layer.py::test_repr_str_and_to_set` (the pass green) |
| hash disagrees with == | red | red: `tests/test_ieee1788_layer.py::test_the_class` (the pass green) |
| == ignores the decoration | red | red: `tests/test_ieee1788_layer.py::test_booleans_examples` (the pass green) |
| is_member(nan) true | red | red: `tests/test_ieee1788_layer.py::test_is_member[nan-False]`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_rec_bool.itl:138]` |
| Interval(None, hi) is empty | red | red: `tests/test_ieee1788_layer.py::test_constructor_refusals[args7-TypeError]` (the pass green) |
| set_dec takes a decorated interval | red | red: `tests/test_ieee1788_layer.py::test_library_values_are_not_operands[multi-set_dec]` (the pass green) |
| x ** 2.0 is pow, not pown (D11) | red | red: `tests/test_ieee1788_layer.py::test_operators_are_the_functions` (the pass green) |
| pown takes a float exponent | red | red: `tests/test_ieee1788_layer.py::test_exponents_are_ints[<lambda>0]` (the pass green) |
| new_dec: com for an unbounded interval | red | red: `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_class.itl:38]` |
| the pass: comparison allows one double of slack | red | red: `tests/itf1788/test_ieee1788.py::test_the_comparison_is_exact`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_rev.itl:276]` |
| the pass: decorations dropped on ours | red | red: `tests/itf1788/test_ieee1788.py::test_the_comparison_is_exact`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:40]` |
| the pass: form assertion removed | red | red: `tests/itf1788/test_ieee1788.py::test_the_comparison_is_exact` (the pass green) |
| the pass: pownRevBin operands in the wrong order | red | red: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_rev.itl:287]` (the layer's files green) |
| the pass: recorded warnings not asserted | red | red: `tests/itf1788/test_ieee1788.py::test_an_escaped_warning_fails` (the pass green) |
| the pass: NaN reading anywhere | red | red: `tests/itf1788/test_ieee1788.py::test_the_nan_reading_is_narrow` (the pass green) |
| the pass: int exponents made floats | red | red: `tests/itf1788/test_ieee1788.py::test_operands_keep_int_exponents`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_elem.itl:1409]` |
| cancellation compares widths in float (the review's F1, S6) | green in the layer's files, red in the pass | red: `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s`; the pass: `tests/itf1788/test_ieee1788.py::test_vector[libieeep1788_cancel.itl:86]` |
| a call of numbers alone decorated (`_flavour`: `decorated != {False}`; S1) | green | red: `tests/test_ieee1788_layer.py::test_a_call_of_numbers_alone_is_bare` (the pass green) |
| the pass: stale-row check dropped (S2) | green | red: `tests/itf1788/test_ieee1788.py::test_a_stale_row_fails` (the pass green) |
| the pass: the type assertion dropped (S3) | green | red: `tests/itf1788/test_ieee1788.py::test_the_comparison_is_exact` (the pass green) |
| the pass: `midRad`'s members not checked float (S4) | green | red: `tests/itf1788/test_ieee1788.py::test_the_comparison_is_exact` (the pass green) |
| `__delattr__` removed (S5) | green | red: `tests/test_ieee1788_layer.py::test_the_class` (the pass green) |
| `rootn(x, 0)` answered as `rootn(x, 1)` (F3) | new | red: `tests/test_ieee1788_layer.py::test_rootn_of_degree_0_raises[x0]` (the pass green) |
| `rootn(empty, 0)` answered empty (F3) | new | red: `tests/test_ieee1788_layer.py::test_rootn_of_degree_0_raises[x0]` and `[x2]`, the bare and the decorated (trv) empty set (the pass green) |

at the build no break stayed green in both runs; the review (below) found five that did (S1 to S5)
and one green in the layer's files alone (F1, S6), each now closed by a test and red (the last
eight rows: six the reviewers' breaks, whose first run is theirs, and two new for F3's pin; the
stale-row break was run by hand, since the harness names the pass's rule tests one by one and
`::test_a_stale_row_fails` is new). after the review's fixes the builder's 46 breaks were
re-run on the layer's files (2026-09-28, `.hypothesis` cleared): all red by the same test but one
drawn case, and the pass's column is unchanged (its module's `::test_vector` and the layer were not
edited). 27 breaks are green in the pass alone, each as expected: the vectors never mix flavours, never pass a library
value, never call with numbers alone, never read the sign of a zero, a `repr`, a hash, a deletion
or a leak, never take a root of degree 0, never match under a row, never yield another set type or
a non-float `midRad` member through a working layer, and every vector operand's decoration fits (so
a demoting constructor cannot show); the design's expectation that removing the
layer's silencing would be seen by the warnings test only was the critique's n1, and it is now red
in the pass as well (the pass asserts every recorded warning is a `PossiblyUndefinedOperationWarning`).
one break is green in the layer's files alone, as expected: `pownRevBin`'s operand order is the
pass's own map, seen by the vectors only. two of the design's §8.3 breaks were no-ops as worded and
were restated to be real breaks: "pair decorated as div also when 0 in b" changes nothing (a
division by a `b` holding 0 is trv already), so the break decorates with the operands' decorations;
"the pass's comparison rounds ours outward first" changes nothing on float ends, so the break lets
the comparison accept one double of slack
the sabotage reviewer (2026-09-28, at `027d8e6`) re-ran 10 of the 46 rows, each red by the test the
table names, and made 20 breaks of its own: 12 red in the layer's files (cancellation with `a` empty
and `b` bounded as entire; each of the silenced `HullWarning`, `IndeterminateResultWarning` and
`EmptySetPropagationWarning` let through; bare operands computed to nearest; `is_member` without its
real-number guard; `cancel_plus` negating before taking the flavour; `is_common_interval` true of
the empty set; the pair's `div` branch taken when 0 is an end of `b`; the newDec cap taken before
rounding; `__rpow__` swapped; `firstEmpty`/`secondEmpty` swapped), one red in the pass only (F1,
S6), five green in both (S1 to S5), and two equivalent. `mul_rev_to_pair` without its drop of
attained infinities changes nothing: on 1788-form operands the library's `mul_rev` never attains
±inf (the 27 x 27 grid probed, 0 differences), so the drop there is defensive and unpinned.
`Interval` checking only `lo`'s type also changes nothing, since `literals.nums_to_interval` already
refuses a non-real `hi`

* review (2026-09-28, three read-only reviewers over `027d8e6`, lenses soundness, sabotage audit
  and spec/regression; each finding reproduced on this branch before any change, by a probe at
  `027d8e6` or by its break): **no wrong answer in the layer**. 15 findings, 11 distinct (F1 and S6
  one gap; F2, S7 and m1 one wording): six test gaps, closed; one unrecorded choice, recorded and
  pinned; one inherited library hang, owed; three doc slips. found and fixed:
    * **the evidence behind "no wrong answer in the layer"**: the soundness reviewer's probes
      (throwaway, 2026-09-28, at `027d8e6`): 42 unary functions in both flavours against arb at 300
      bits, 66420 checks over four seeds (an oracle sabotaged on purpose gave 14 bad of 80, so the
      check is live); the binary ops, `fma`, `pown`/`rootn`, the reverse ops with and without `x`,
      `mul_rev_to_pair` and cancellation, 21672 checks over three seeds; decoration propagation 4437
      checks; the numbers against `Fraction` on 2980 drawn intervals and the booleans against 1788's
      definitions on 144 pairs in both flavours (11664); every `NAMES` function on 12 edge operands
      in every arity (12396 calls: no crash, empty always trv, com never unbounded); and the 11
      tighter-than-the-vector rows re-run through the layer against arb at 400 bits, each still an
      enclosure. 0 bad in all
    * **cancellation's exact width comparison was held by the pass alone** (F1, S6): with the
      widths compared in float, `cancel_minus(Interval(0.1, 1e17), Interval(0.0, 1e17))` is empty
      where 1788 (and the layer) say entire, and the layer's property stayed green, since
      `tests/test_ieee1788_layer.py::cancel_operands` never drew a width tie that floats round
      away. it now has a branch drawing widths equal as floats but not exactly (`big` in
      [2^54, 2^70], `small` in [tiny, 0.5], so `big ± small` rounds to `big`; hypothesis's `find`
      over the strategy reaches one within 300 examples), and `::test_cancel_minus_is_1788s` three
      `@example`s: F1's pair, S6's `1e16 + 1` pair and two widths that overflow in floats
    * **a call of numbers alone was unpinned** (S1): `_flavour` returning `decorated != {False}`
      made `add(1, 2)` com and stayed green. `::test_a_call_of_numbers_alone_is_bare` (`add`,
      `sqrt`, `fma`, `mul_rev_to_pair` on numbers: bare)
    * **three of the pass's own assertions were never exercised** (S2, S3, S4): the stale-row check
      (`tests/itf1788/test_ieee1788.py::test_a_stale_row_fails`, a matching vector put under `ROWS`
      with `monkeypatch`), `interval_form`'s type check (a to-nearest `MultiInterval(4.0, 6.0)`
      passes every other check) and `value_form`'s float check on `midRad`'s members (both
      appended to `::test_the_comparison_is_exact`)
    * **immutability was half pinned** (S5): `del x._set` is now refused in
      `tests/test_ieee1788_layer.py::test_the_class`
    * **`rootn(x, 0)` was an unrecorded choice** (F3): it raises `ValueError` for every `x`, the
      empty set included, the library's rule (`v2-plan.md`: "rootn(n) for every int n other than
      0"), while `pown_rev(c, 0)` answers. recorded in the design's numbers bullet as the default
      built, and pinned by `::test_rootn_of_degree_0_raises` (4 items: empty and `[1, 4]`, bare and
      com); no vector has degree 0. not made a Q13 item: the layer keeps a documented library rule
      where no 1788 vector answers
    * docs: `NAMES` read as 106 names (F2, S7, m1; it is 104, 1788's 102 and itf1788's two `d-`
      constructors, 104 distinct functions: `len(NAMES)`, and `::test_names`); the form invariant
      cited `Interval._make` (m2; it is `Interval._init`, which `__init__` and `_make` call); the
      gate's second call was described as "`tests/test_a*` to `tests/test_m*`" (m3; it named 14
      files, `tests/test_applicator.py` to `tests/test_minmax_fma.py`, and `tests/test_modulo.py`,
      `tests/test_multi_interval.py` ran in the third; M16b's measured bullet now names the split); the
      testing bullet said the pair's set law is drawn with com operands (m4; in
      `::test_mul_rev_to_pair_decorated` both sides of the law come from the layer's `div` branch,
      so it pins the decoration only; the law is drawn on bare operands by
      `::test_mul_rev_to_pair_bare`, and the test's docstring now says so)
* deferred (F4): the library's `pown` of a huge integral exponent is unbounded in time, inherited
  and reached through the layer's `pown` and `**`; `HANDOFF.md` "still owed"
* sabotage of the fixes: the last eight rows of the table above, all red
* second look (2026-09-28, a read-only verifier over `06d9876`): no defect. the gate's second and
  third calls had not finished when it reported, so the gate was re-run whole (M16b's measured
  bullet); F4's deferral judged sound; `rootn(empty, 0)`'s break is red at both empty items,
  `[x0]` and `[x2]` (the table now names both); and
  `::test_a_call_of_numbers_alone_is_bare` checked `.decoration is None` for `add` only, the rest
  by equality. it now asserts no decoration on all five results (`add`, `sqrt`, `fma` and both of
  the pair's); the S1 break is red there, as before (not a gap: `Interval`'s equality already
  compares decorations)

| id | lens | disposition | evidence |
|---|---|---|---|
| F1 | soundness | fixed (with S6) | probe at `027d8e6`: float widths agree for the pair, exact do not; break "cancellation compares widths in float" was green in the layer's files, now red at `tests/test_ieee1788_layer.py::test_cancel_minus_is_1788s` |
| F2 | soundness | fixed (with S7, m1) | `len(ieee1788.NAMES)` 104, 104 distinct values, the two `d-` keys among them; design and M16b spec reworded |
| F3 | soundness | fixed: recorded and pinned | `ieee1788.rootn(Interval(), 0)` raises `ValueError` at `027d8e6`; design numbers bullet; `::test_rootn_of_degree_0_raises`, red under two breaks |
| F4 | soundness | deferred: still owed | `pown(Interval(0.5, 1), 2 ** 31 - 1)` and `OutwardMultiInterval(0.5, 1) ** (2 ** 31 - 1)` past a 30 s timeout (2026-09-28); library, outside M16b |
| S1 | sabotage | fixed | `::test_a_call_of_numbers_alone_is_bare`, red under the break |
| S2 | sabotage | fixed | `tests/itf1788/test_ieee1788.py::test_a_stale_row_fails`, red under the break (by hand) |
| S3 | sabotage | fixed | `::test_the_comparison_is_exact` (a `MultiInterval` set), red under the break |
| S4 | sabotage | fixed | `::test_the_comparison_is_exact` (`midRad` with a `Fraction` member), red under the break |
| S5 | sabotage | fixed | `tests/test_ieee1788_layer.py::test_the_class` (`del x._set`), red under the break |
| S6 | sabotage | fixed (F1's gap) | as F1: three `@example`s and a generator branch |
| S7 | sabotage | fixed (F2's wording) | as F2 |
| m1 | spec | fixed (F2's wording) | as F2 |
| m2 | spec | fixed | `intervals/ieee1788.py::Interval._init` holds the `_is_1788_form` assertion; the design cites it |
| m3 | spec | fixed | the measured bullet names the gate's three calls by file |
| m4 | spec | fixed | testing bullet reworded; `::test_mul_rev_to_pair_decorated`'s docstring says what it pins |

**M16c the per-piece allen matrix (done 2026-09-28)** (D22).

the owner, 2026-09-27: "get the rest of h3 done". H3's row lists "per-piece Allen matrix"; its
spec pointer is `v2-plan.md` "v2 consolidated decisions (2026-08-16)" / "comparisons". the choices
the build made are D22, open for the owner as `HANDOFF.md` Q14; the design is `v2-plan.md`
"comparisons".

* **`intervals/relations.py`**: `allen_matrix(a, b)` (the plain loop), `allen_relations(a, b)`
  (the sweep and two corners), `_allen_pairs(pa, pb)` (the sweep, private; it calls `allen` as the
  module global, which the cost test counts); `allen_relations` asserts its operands normalized
  (the review, F1); two sentences in the module docstring
* **`intervals/multi_interval.py`**: `MultiInterval.allen_matrix`, `MultiInterval.allen_relations`,
  each coercing through `_coerce_or_raise`, each with doctests (the worked example and an empty
  operand)
* exit: every entry is `allen()` of its pair of pieces and the set view is the matrix's entries,
  both against the `n x m` loop over the pinned `allen()`; the converse, a set against itself and
  the set relations (overlaps, disjoint, before, after, within, contains, equals, adjoins) as
  identities over the matrix; the empty shapes; the set view `O(n + m)` in calls and in cut
  comparisons, never the matrix, and refusing out-of-order operands; the gate green; every new
  property sabotaged once and seen red

record (2026-09-28):
* **what the build found on its way**:
  * **the gate found a name the design missed.** `tests/test_propagation.py::test_every_public_name_of_the_core_is_on_the_wrapper_or_asked_of_the_interval`
    went red on `{'allen_matrix', 'allen_relations'}`: a name the core gains must be on
    `DecoratedInterval` or in `::NOT_ON_THE_WRAPPER`, on purpose. the design's "not on
    `DecoratedInterval`" is now written there, beside `allen` (the design listed the file as
    untouched). the guard worked as meant
  * **the sweep's tie rule was held by chance.** "a tie advances `i` only" is output-equivalent (one
    extra `allen()` per tie, an AFTER pair; still within `n + m - 1`), and the design accepted it
    unpinned. the repo's rule makes a green break a gap: the cost test now asserts the sweep visits
    exactly the pairs of cells that intersect (piece `i` owns the cuts after the end of piece
    `i - 1`, up to its own end; the sweep walks the two partitions' common refinement). that went
    red on a re-run, but green on the first final run: measured 2026-09-28, 19 of 20 seeded
    100-example runs of `operand_pairs` hit a tie followed by more pieces (0 to 15 examples a
    run). so the test carries an `@example` of a two-piece set against itself, where every end
    ties; red since then
  * **the sabotage harness ran stale bytecode.** the template harness (copy the file to `.orig`,
    write the break, run, move `.orig` back) restores a file whose mtime lies in the same second
    as the broken write; for a break of the same byte length ("allen_relations operands swapped")
    the restored source matched the broken `.pyc` (mtime in seconds, size), and python kept
    running the broken bytecode. the next two runs were red for that reason, not their own; both
    were re-run. the harness now clears `__pycache__` before and after each break, runs with
    `PYTHONDONTWRITEBYTECODE=1`, restores with `copy2`, and starts with a control row on the
    intact code. H3's template harness has the same hazard
  * **the conservative matrix is pinned**: the plain loop is a choice, so
    `::test_allen_matrix_does_not_need_normalized_operands` (two cut tuples laid end to end give
    the two matrices stacked) holds it; the design's fill + sweep goes red there
* **tests** (`tests/test_relations.py`, and the two methods' doctests in
  `intervals/multi_interval.py`), all at hypothesis's default settings (no `@settings`, so the
  fuzz profile multiplies them like the rest); written first, against stubs raising `NotImplementedError`: 43 failed (every new test) and 30 passed (the module's old ones) in 206 s, 2026-09-28:
  * operands: `::operand_pairs`, a mixture of two independent `::operands`
    (`exact_cut_tuples` or `::dense`, a grid with ±inf) and pairs where one is derived from the
    other (`::_derived`: itself, its complement, hull, gaps, interior, and itself with the
    complement of its hull), so shared cuts, MEETS and MET_BY are common (the census, on the design's prototype 2026-09-27, derandomized, matrix entries counted: two
    independent `exact_cut_tuples` over 3000 draws gave MEETS 19 and MET_BY 25 of 3012 entries;
    `operand_pairs` over 400 draws gave MEETS 31 and MET_BY 25 of 403, all 13 relations present,
    OVERLAPS the rarest at 1, which the independent third of the mixture supplies; not re-measured on
    the built strategy); oracle `::allen_loop`,
    the `n x m` loop over `relations.allen`; `::converse` takes the column count explicitly
  * `::test_allen_matrix_is_allen_of_each_pair_of_pieces` (and the method equals the function),
    `::test_allen_relations_are_the_matrix_entries`, `::test_allen_matrix_converse`,
    `::test_allen_matrix_of_a_set_with_itself`, `::test_allen_matrix_and_the_set_relations`
    (columns counted with `range(m)`), `::test_allen_matrix_on_single_pieces`, and two lines in
    `::test_allen_table` (on each of its 17 rows the matrix is `((relation,),)` and the set view
    `{relation}`)
  * `::test_allen_matrix_does_not_need_normalized_operands`,
    `::test_allen_matrix_of_unordered_overlapping_pieces` (the plain loop's contract)
  * `::test_allen_matrix_of_an_empty_operand` (no raise, no warning of any kind; `allen()` still
    raises), `::test_allen_matrix_table` (13 worked rows, both directions: a shared closed end,
    a piece that meets where the sets do not adjoin, a point filling a one-point gap, points
    against pieces, `[1, inf)` and `[1, inf]` against `[inf]`, `[-inf]`, both corners at once),
    `::test_a_piece_meets_where_the_sets_do_not_adjoin`,
    `::test_allen_matrix_coerces_as_every_relation` (a number, a string, nan, the two classes in
    both orders), `::test_allen_matrix_mixes_numeric_types_at_a_shared_cut` (`[0, 0.5)` MEETS
    `[1/2, 3]`, `[0, 1.0)` MEETS `[1, 2]`)
  * `::test_allen_relations_is_a_linear_sweep`, the cost pin: `_allen_pairs` directly (no pair
    twice, at most `n + m - 1`, every pair that is not BEFORE or AFTER, exactly the intersecting
    cells, each relation `allen()`'s), then `allen_relations` with `relations.allen_matrix` patched
    to raise (critique B1: flattening the matrix would keep the call count) and `relations.allen`
    spied: at most `n + m - 1` calls and at least one per pair that is neither BEFORE nor AFTER.
    its docstring says the lower bound pins an implementation detail on purpose (a spy that
    cannot pass at 0 calls): relax it knowingly, do not delete it
  * `::test_allen_relations_compares_cuts_linearly` (the review, SAB-1): the same cost in cut
    comparisons, which the call counts do not see. `::_CountingCut`, a `Cut` subclass counting
    `< <= == != > >=`, on 40 pieces of `a` all AFTER 40 of `b` (a quadratic scan for BEFORE cannot
    stop early): at most `10 (n + m)` = 800. measured 2026-09-28: 360 (158 of them the operands'
    `kernel.is_valid`); the review measured its `O(nm)` BEFORE pass at 1801, before the check
  * `::test_allen_relations_refuses_unnormalized_operands` (the review, F1): `[3, 4]` then `[0, 1]`
    against `[0, 1]`, whose matrix holds AFTER and EQUALS and whose sweep found only AFTER, raises
    `AssertionError` in both orders and against itself (skipped under `python -O`)
* **measured 2026-09-28** (shared laptop, four other streams running). every command runs from the
  worktree root with the env's interpreter, written `$PY` below: `PY=C:/Users/user/anaconda3/envs/intervals/python.exe`
  (bare `python` is a Microsoft Store stub on this laptop):
  * the stream: `$PY -m pytest -q -p no:cacheprovider tests/test_relations.py
    intervals/relations.py intervals/multi_interval.py`: 103 passed in 36.4 s at `9bc9e7c`; after
    the review's two tests, 105 passed in 26.1 s
  * the module three times each, `.hypothesis` cleared before every run, `$PY -m pytest -q -p
    no:cacheprovider <module>`: at `04946af` 47 tests in 21.1, 25.2, 26.6 s; at `9bc9e7c` 73 tests
    in 34.0, 38.8, 38.1 s: about 13 s added to the gate. the review's two tests are not
    hypothesis tests and ran in 0.41 s together (collection included). `too_slow` never fired
    (these six runs and every sabotage run), so no health check is suppressed
  * the trade, laptop numbers (the tests pin counts, not times): `n = m` pieces `[2k, 2k+1]`
    against `[2k+1/2, 2k+3/2]`, best of 3 by `time.perf_counter`. the probe is the block below
    (gitignored as a file, so it is written out here): save it as `.scratch/trade.py` and run
    `PYTHONPATH=. $PY .scratch/trade.py`. re-run 2026-09-28 after the review (the first run's
    table, same day at `9bc9e7c`: 4040 / 2060 / 27.5 / 4640 ms at 1000); the fill's gain stays
    about 2x at 1000 pieces, the design's ~3x on 2026-09-27

    ```
    import time
    from fractions import Fraction

    from intervals import kernel, relations
    from intervals.relations import Allen


    def fill_sweep(a, b):  # the design's matrix, not taken
        pa, pb = tuple(kernel.pairs(a)), tuple(kernel.pairs(b))
        rows = [[Allen.BEFORE if p[1] < q[0] else Allen.AFTER for q in pb] for p in pa]
        for i, j, r in relations._allen_pairs(pa, pb):
            rows[i][j] = r
        return tuple(tuple(row) for row in rows)


    def best(f, *args, k=3):
        out = []
        for _ in range(k):
            t = time.perf_counter()
            f(*args)
            out.append(time.perf_counter() - t)
        return min(out) * 1000


    for n in (10, 100, 300, 1000):
        a = kernel.normalize([kernel.piece(2 * k, 2 * k + 1) for k in range(n)])
        b = kernel.normalize([kernel.piece(2 * k + Fraction(1, 2), 2 * k + Fraction(3, 2)) for k in range(n)])
        assert relations.allen_matrix(a, b) == fill_sweep(a, b)
        flat = lambda a, b: frozenset(r for row in relations.allen_matrix(a, b) for r in row)  # noqa: E731
        assert relations.allen_relations(a, b) == flat(a, b)
        cols = [best(relations.allen_matrix, a, b), best(fill_sweep, a, b),
                best(relations.allen_relations, a, b), best(flat, a, b)]
        print(f'| {n} | ' + ' | '.join(f'{c:.2f} ms' for c in cols) + ' |', flush=True)
    ```

    | n = m | matrix, plain loop (built) | matrix, fill + sweep (not taken) | relations, sweep (built) | relations, matrix flattened |
    |---|---|---|---|---|
    | 10 | 0.28 ms | 0.27 ms | 0.20 ms | 0.31 ms |
    | 100 | 24.7 ms | 9.6 ms | 2.07 ms | 30.0 ms |
    | 300 | 246 ms | 156 ms | 6.55 ms | 262 ms |
    | 1000 | 3788 ms | 1942 ms | 22.6 ms | 3772 ms |

  * the gate at `9bc9e7c`, from the worktree root: `$PY -m pytest -q tests/itf1788`: 18246 passed in 81.3 s.
    `$PY -m pytest -q --ignore=tests/itf1788` in three calls (five streams at once would
    overrun one call): `tests/test_reverse.py tests/test_functions.py tests/test_propagation.py
    tests/test_oracle_flint.py tests/test_pow_rev.py` 1011 passed and the one red above in
    398.8 s, `tests/test_propagation.py` re-run after the fix 193 passed in 51.0 s; nine files
    (`test_elementary`, `test_modulo`, `test_ops_examples`, `test_ops_properties`,
    `test_relations`, `test_applicator`, `test_literals`, `test_oracles`, `test_autodiff`) 2467
    passed in 277.5 s; the rest (`--ignore` of those 14 files, with the doctests and `README.md`)
    637 passed in 296.8 s. 4116 items outside itf1788 (22362 collected in all, one process),
    973 s summed (1024 s with the re-run)
  * the gate after the review's fixes, the same calls and groups: `tests/itf1788` 18246 passed in
    78.8 s; the five files 1012 passed in 381.4 s; the nine files 2469 passed in 372.7 s; the rest
    637 passed in 424.2 s. 4118 items outside itf1788 in 1178.3 s summed (the laptop more loaded
    than at `9bc9e7c`); `$PY -m pytest --collect-only -q` over the whole tree in one process:
    22364 collected, no error (so every test file basename is unique)
* **sabotage** (a throwaway harness: each break alone, `.hypothesis` and `__pycache__` cleared, the
  stream's three files with `-x` and a 600 s timeout, the file restored and compared; 2026-09-28).
  the last column is the first test to fail under `-x`, in the whole table's re-run after the
  review's fixes (21 rows, the control 105 passed); two breaks now fall first to another test than
  in the build's final run (`sweep: compare starts` was red by
  `::test_allen_relations_are_the_matrix_entries`, `sweep: stops after two pairs` by the same):
  hypothesis draws afresh with `.hypothesis` cleared, and `-x` stops at the first:

| break | first run | final run: red by |
|---|---|---|
| none (control: the intact code) | green | green: 105 passed |
| sweep: advance the other pointer | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| sweep: compare starts, not ends | red | red: `tests/test_relations.py::test_allen_matrix_converse` |
| sweep: a tie advances `i` only | green | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| sweep: stops after two pairs | red | red: `tests/test_relations.py::test_allen_matrix_table[[0, 10]-[0, 1] \| [2, 3] \| [9, 10]-matrix12]` |
| sweep: `allen` bound locally (the spy reads 0) | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: BEFORE corner dropped | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: BEFORE corner `<=` (MEETS counted) | red | red: `tests/test_relations.py::test_allen_table[[1, 2)-[2, 3]-Allen.MEETS]` |
| relations: AFTER corner on the wrong pieces | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: flatten `allen_matrix` (critique B1) | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: `n m` calls to `allen()` | red | red: `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` |
| relations: an empty operand raises | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| relations: the empty guard removed | red | red: `tests/test_relations.py::test_allen_relations_are_the_matrix_entries` |
| matrix: each entry inverted | red | red: `tests/test_relations.py::test_allen_table[[1, 2]-[3, 4]-Allen.BEFORE]` |
| matrix: transposed | red | red: `tests/test_relations.py::test_allen_matrix_is_allen_of_each_pair_of_pieces` |
| matrix: an empty other gives `()` (shape lost) | red | red: `tests/test_relations.py::test_allen_matrix_is_allen_of_each_pair_of_pieces` |
| matrix: the design's fill + sweep (the choice not taken) | red | red: `tests/test_relations.py::test_allen_matrix_does_not_need_normalized_operands` |
| method: `allen_matrix` without `_coerce_or_raise` | red | red: `tests/test_relations.py::test_allen_matrix_coerces_as_every_relation` |
| method: `allen_relations` operands swapped | red | red: `tests/test_relations.py::test_allen_table[[1, 2]-[3, 4]-Allen.BEFORE]` |
| relations: BEFORE corner by an `O(nm)` comparison pass, `any(p[1] < q[0] for p in pa for q in pb)` (review SAB-1) | green (at `9bc9e7c`: 103 passed) | red: `tests/test_relations.py::test_allen_relations_compares_cuts_linearly` |
| relations: the normalized-operand assert removed (review F1) | red (the test written first, against `9bc9e7c`: DID NOT RAISE) | red: `tests/test_relations.py::test_allen_relations_refuses_unnormalized_operands` |

the two greens in a first run were gaps, each closed and re-run red: the tie rule
(`::test_allen_relations_is_a_linear_sweep`, the intersecting cells and the `@example` above) in the
build, and the `O(nm)` comparison pass (`::test_allen_relations_compares_cuts_linearly`) at the
review. the control row was added with the harness fix, so its first run is the final one's. two
runs between the build's first and final run were red for a stale `.pyc` (above) and are not in
the table.

review (2026-09-28, three read-only reviewers over `9bc9e7c`, lenses soundness, sabotage-audit and
spec/regression; each finding reproduced on this branch before any change): **no wrong result
through `MultiInterval`**. found and fixed:
* **F1 (soundness) and m2 (spec), one defect: the module docstring said `allen_matrix` and
  `allen_relations` "take any"**, and `relations.allen_relations` on a valid-per-piece but
  out-of-order cut tuple gave a strict subset of the matrix's entries, silently. reproduced at
  `9bc9e7c`: `a = [3, 4]` then `[0, 1]` against `b = [0, 1]` gave `{AFTER}`, the matrix
  `{AFTER, EQUALS}`. the methods could not reach it (`MultiInterval.from_cuts` validates and
  `MultiInterval._wrap` asserts). fixed twice over: the docstring now says the matrix takes any cut
  pairs in any order and the set view's sweep needs normalized cut tuples, as every relation there;
  and `relations.allen_relations` asserts `kernel.is_valid` of both operands under `__debug__`, as
  `_wrap`. pinned by `tests/test_relations.py::test_allen_relations_refuses_unnormalized_operands`,
  written first and red at `9bc9e7c` (DID NOT RAISE); its break (the assert removed) is a sabotage
  row. the cost: `is_valid` is `O(n + m)` comparisons, inside the set view's bound
* **SAB-1 (sabotage-audit): the set view's `O(n + m)` was pinned in calls to `allen()` only.** a
  BEFORE corner rewritten as `any(p[1] < q[0] for p in pa for q in pb)`, `n m` cut comparisons
  that call neither `allen()` nor `allen_matrix`, stayed green; reproduced at `9bc9e7c` with the
  harness (103 passed; critique B1 had named the residue). the reviewer's option (a) taken:
  `tests/test_relations.py::test_allen_relations_compares_cuts_linearly` counts cut comparisons
  through `::_CountingCut` (the reviewer's class, with `!=` counted too) and bounds them by
  `10 (n + m)`; 360 of 800 on the intact code, and the break is red there. a sabotage row
* **m1 (spec): the trade table named no command**, its probe only in the gitignored
  `.scratch/trade.py`, while the decision-log revision's and D22's ~2x rested on it. the probe is
  now written out in M16c's record with its command, and re-run 2026-09-28 after the fix (3788 ms
  against 1942 ms at 1000 pieces: still about 2x)
* **m3 (spec): the record's commands read bare `python`**, a Microsoft Store stub on this laptop.
  they now read `$PY`, defined once as the env's interpreter
* not changed: the record's "O(n + m) in calls" and "pinned by counts, not time", which the
  sabotage lens called not false; they now name both counts
* the reviewers' clean results, transcribed 2026-09-28 from their notes (gitignored): the
  soundness lens's independent point-set oracle (membership decided from the raw piece specs,
  never the library's cuts; components over sample points with ±inf; both classes; int, float and
  `Fraction` ends mixed, `0.5` against `1/2`) found no wrong entry of `allen_matrix` and no wrong
  `allen_relations` in 20000 random cases, nor in 20000 on a grid with ±inf, up to 9 raw pieces
  and `b` derived from `a` (itself, complement, hull, union with the gaps; 9510 with an EQUALS
  entry); on the 66 single pieces over `{-inf, 0, 0.5, 1, 2, inf}` with every closedness, all 4356
  pairs, `allen_matrix(a, b)[0][0]` at `9bc9e7c` is `allen(a, b)` at `04946af` and
  `allen_relations` is `{allen(a, b)}`. the three lenses ran 33 breaks of their own beyond the
  table (soundness 7, spec/regression 7, sabotage-audit 19; a few the same break, e.g. the AFTER
  corner `<=` and a tie advancing `j` only), all red but one, SAB-1 above: e.g. the sweep yielding
  `(j, i)`, or draining the rest of `a` against the last of `b`, by
  `::test_allen_relations_is_a_linear_sweep`; the BEFORE corner against the first of `b`, or taken
  only when the sweep found none, by `::test_allen_relations_are_the_matrix_entries`; the AFTER
  corner `<=` by `::test_allen_table[[2, 3]-[1, 2)-Allen.MET_BY]`; `other` an iterator used up by
  the first row, or the rows reversed, by `::test_allen_matrix_is_allen_of_each_pair_of_pieces`;
  `self` normalized first by `::test_allen_matrix_does_not_need_normalized_operands`; the flatten
  break kept for operands of at most 4 pairs by `::test_allen_relations_is_a_linear_sweep`. they
  re-ran 11 of the build's rows at `9bc9e7c`, each red by the test the build's record named. the
  verifier over `ce480a1`: F1's and SAB-1's breaks red (the comparison pass counted 1959 against
  the bound of 800), the 360 comparisons (158 of them `is_valid`) reproduced, and the gate green
  (`tests/itf1788` 18246 passed in 129.5 s; the rest 4118 passed in 1160.6 s summed; 22364
  collected)

**M16d numpy interop: `numpy_compat.py` (done 2026-09-28)** (D23).

the owner, 2026-09-27: "get the rest of h3 done". the design is `v2-plan.md` "numpy"; the choices
are D23, open as `HANDOFF.md` Q15. here the spec, the exit and the record.

* **`intervals/numpy_compat.py`**: `array_ufunc` (the `__array_ufunc__` of `MultiInterval`,
  `DecoratedInterval`, `Dual`) and `array` (`MultiInterval.__array__`); imports no numpy at load
* **`cuts.py::normalize_value`**: the foreign-real exact path (`cuts.py::_exact_value`)
* **integer arguments**: `functions.py::_check_degree`, `steps.py::step` (`ndigits`),
  `reverse.py::pown_rev` any `Integral` but bool; `functions.py::_check_base` any real but bool
  through `normalize_value`
* **`autodiff.py::Dual.__pow__`**: a number exponent an int or a point set before `- 1` (B1, an
  M15 hole); **`Dual.rootn`** `int(n)` first (B2)
* exit: every numpy scalar type in every operator the classes have, on both sides, the same outcome
  as the python number of the same value; every ufunc of the table equal to its method; the
  elementwise path equal to the scalar path per element; the foreign-real rule against a stub
  holding a `Fraction`; B1 against arb; numpy never imported at load; the gate green; every new
  property sabotaged once and seen red

record (2026-09-28):
* **what the build found on its way**, each fixed before the record:
  * **M15 shipped an unsoundness in `Dual ** r`** (the critique found it; confirmed on `04946af`
    with arb at 400 bits): `exponent * u ** (exponent - 1)` computed `r - 1` in r's own float
    arithmetic, rounded to nearest. `Dual.variable(O(u)) ** r` missed `r u^(r-1)` for 5 of 9
    pairs, r in {0.1, 1e-20, 0.3}, u in {2, 1e300, 1e-300}: 0.1 with 1e300 and 1e-300, 1e-20 with
    2, 0.3 with 1e300 and 1e-300 (0.1 with 1e300: the derivative
    `(9.999999999999845e-272, 9.999999999999849e-272)`), and the same 5 for
    `DecoratedInterval(O(u))` (the build first wrote "6 of 9", and pinned the bare class only: the
    review's catch, see the review below). through numpy the hole was not confined to large or tiny u: an `np.float32(0.1)` or `np.float16(0.1)` exponent had `r - 1` rounded to 24 or 11 bits, so `Dual.variable(O(2)) ** np.float32(0.1)` missed its derivative too (the critique's probe, 2026-09-27: 11 of 15 cases over python, float32 and float16 exponents at u in {2, 1e300, 1e-300}; `np.longdouble` reaches it on linux only). the build found the integral case too:
    `2.0 ** 60 - 1` rounds to `2.0 ** 60`, so `Dual.variable(O(-1)) ** 2.0 ** 60` had the
    derivative `+2 ** 60` where it is `-2 ** 60` (a sign; `np.float32(16777218)` the same). the
    solver builds its steps from that derivative. fixed: an integral r is `n = int(r)` and `n - 1`
    int arithmetic; any other r a point set `e` of u's kind and `e - 1` the library's subtraction
  * the fix changes the nearest and decorated classes too, not only the unsound outward cases
    (the review's catch): an integral float exponent's derivative is `n u ** (n - 1)` with the int
    n, exact where the value is (`Dual.variable(M.parse('[1/3, 3]')) ** 2.0`: the value `[1/9, 9]`,
    the derivative `[2/3, 6]` where M15 gave `[0.6666666666666666, 6.0]`; `** -3.0` at `M(3)`:
    `[-1/27]`, was `[-0.037037037037037035]`), and a non-integral r whose float `r - 1` is -1.0
    (r = 1e-20) is pow, as the value is, where M15 took pown(u, -1): over `M(-1, 1)` the derivative
    is `[1e-20, inf)` (was `{ [-inf, -1e-20] , [1e-20, inf] }`), over `M(-2, -1)` it is `{}` beside
    the empty value (was `[-1e-20, -5e-21]`). kept: each agrees with the value's own reading of r
    (pown for an integral r, else pow); nothing outward loses its value
  * with both operands ours, the method ufuncs took the first operand's method and class:
    `np.hypot(M(0.1), O(0.1))` was the nearest `[0.1414213562373095]`, which misses the true value,
    and `np.hypot(O(0.1), M(0.1))` outward (the review's catch). fixed: the operators' subclass rule
    (`numpy_compat.py::_subclass_first`), Q15(h)
  * the `except TypeError` round the table lookup in `numpy_compat.py::array_ufunc` guarded an
    unhashable ufunc numpy never passes, and nothing could pin it (the review's sabotage): deleted
  * `O(u) ** 2 ** 60` does not finish for u in {2, 2.0, 1e300} (outward pown evaluates
    `Fraction(u) ** n` exactly; a plain `M(2) ** 2 ** 60` too): pre-existing, not this stream's; so
    the critique's r = `2.0 ** 60` is pinned with u = -1, where the power is cheap (still owed)
  * `Dual.rootn(np.int64(-2 ** 63))` would compute `n - 1` in int64 (a wrap and a RuntimeWarning)
    once the core took numpy ints: `int(n)` first (B2)
  * the prototype looked dunders up with `hasattr(type(x), name)`, which finds `type.__or__` (PEP
    604) on every class: `Dual` has no `|` but `hasattr(Dual, '__or__')` is True. the hook walks
    the MRO dicts, as python's operator lookup does (`numpy_compat.py::_special`)
  * numpy reads the floating-point status flags after an object loop, and the library's own
    python float arithmetic leaves them set: `np.array([1.0]) / M.parse('(-inf, 2.2e-309)')` gave
    a numpy `RuntimeWarning: overflow` the scalar path never gives. the elementwise path runs under
    `np.errstate(all='ignore')` (found by
    `tests/test_numpy_compat.py::test_ndarray_and_interval_elementwise[divide]`); underflow too
    (`np.array([1e-308]) + M(0)` under a caller's `np.errstate(all='raise')`), which numpy ignores
    by default, so only the review's sabotage (`errstate(over=...)` stayed green) found it unpinned
  * the prototype's `__array__` cast with `astype(dtype)`; numpy casts the 0-d object array itself,
    so the cast was dead code with no test able to see it: dropped
  * the numpy-missing check by hand: `sys.modules['numpy'] = None` crashes hypothesis itself (it
    reads `sys.modules['numpy']`); a meta-path finder raising `ModuleNotFoundError` is the faithful
    stand-in
* **tests** (`tests/test_numpy_compat.py`; two in `tests/test_autodiff.py`):
  * `tests/test_numpy_compat.py::test_numpy_is_never_imported_at_load` (a subprocess imports every
    module; `pyproject.toml` has no numpy in `dependencies` or `[test]`)
  * `::test_numpy_scalar_operators_are_python_numbers`: 9 numpy scalar types (float64, float32,
    float16, longdouble with values off the doubles, int8, int64, uint64, bool_, complex128) x the
    operators **derived from the classes' reflected dunders** (`::REFLECTED`) plus `== != < <= > >=`
    and `in` of a list, both sides, over drawn `MultiInterval`, `OutwardMultiInterval`, decorated
    and `Dual` objects: the same result (type and repr) or exception type, and the same warning
    categories, attributed to the test file, as the python number of the same value
    (`::python_number`); `::test_numpy_scalar_operator_examples`, `::test_the_operators_are_derived`,
    `::test_a_missing_dunder_is_numpys_refusal` (a gap the sabotage found), `::test_numpy_exponent_of_a_dual` (B1)
  * `::test_unary_ufunc_is_the_method`, `::test_binary_ufunc_is_the_method_or_the_operator` (the
    other operand a number or one of ours, both orders; both ours is python's rule),
    `::test_both_ours_is_pythons_rule`, `::test_both_ours_examples` (B3),
    `::test_both_ours_methods_are_subclass_first` (Q15(h), the review), `::test_ufunc_pinned_examples`
    (`square [-1, 1]`, `rint [1/2, 5/2]`, `arcsin` against `acos`, `arctan2` both ways)
  * `::test_unmapped_ufuncs_and_forms_are_type_errors` (fmod with its reason, fmin/fmax, keywords,
    `reduce`/`outer`/`accumulate`, a `frompyfunc` ufunc, list operands),
    `::test_the_table_is_keyed_by_the_ufunc_object` (a stand-in named `'sin'`, B3) and its positive
    control `::test_the_table_answers_numpys_sin`
  * `::test_equality_never_broadcasts` (with `np.array([A]) == A` and `A in np.array([A])` pinned
    False), `::test_arrays_hold_intervals_as_elements`, `::test_object_array_loops_are_numpys`
    (`np.square(arr)` is `x * x`, `np.arcsin(arr)`, `np.round`, `np.around` TypeErrors)
  * `::test_ndarray_and_interval_elementwise` (float64 and int64 arrays of shape `(k,)` or
    `(2, k)`, every two-operand entry but `==`/`!=`, both orders, against the scalar path on each
    python element; an element that raises raises the op), `::test_elementwise_examples` (a nan
    element, bool and complex arrays), `::test_elementwise_warning_is_the_callers`,
    `::test_elementwise_leaves_numpys_float_flags_alone` (a gap the sabotage found),
    `::test_elementwise_leaves_every_float_flag_alone` (underflow; the review)
  * `::test_foreign_reals_are_exact` (stubs `::Wide`, registered `numbers.Real`, and `::Rat`,
    registered `numbers.Rational`, over drawn fractions, ones past the doubles and x86-64's long
    double nearest 1/3), `::test_foreign_real_specials`,
    `::test_a_foreign_real_without_a_ratio_is_its_float` (`::Plain`; the review),
    `::test_float32_is_the_double_it_holds`
    (every float32 bit pattern drawn), `::test_longdouble_is_exact` (discriminates on linux CI),
    `::test_gmpy2_values_are_exact` (skips where gmpy2 is absent, as on CI; since the merge with
    M16e gmpy2 is in `[test]`, so it runs in every CI job)
  * `::test_integer_arguments_take_numpy_ints` (rootn, round, round_ties_away, pown_rev, the
    decorated and `Dual` forms, the log base; bool refused), `::test_numpy_ndigits_past_int64` and
    `::test_numpy_pown_rev_degrees` (the `int()` conversions; the review),
    `::test_dual_rootn_takes_the_degree_as_an_int` (B2)
  * `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses` (B1, arb at 400 bits,
    r in {0.1, 1e-20, 0.3} x u in {2, 1e300, 1e-300} x {`O(u)`, `DecoratedInterval(O(u))`}),
    `::test_pow_integral_exponent_derivative_is_exact` (r in {2.0 ** 60, 2 ** 60, Fraction(2 ** 60)},
    u = -1), `::test_pow_any_zero_is_the_constant_one` (0, 0.0, -0.0, `Fraction(0)` over four u;
    the review), `::test_pow_number_exponent_nearest_examples` (the nearest-class change; the review)
* **measured 2026-09-28** (laptop shared with four other streams' runs, so loaded):
  * `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q tests/test_numpy_compat.py`: 280 passed
    in 36.96 s (numpy 2.5.2, python 3.13, windows; after the review's fixes)
  * the numpy-missing path, by hand: a meta-path finder raising `ModuleNotFoundError` for numpy,
    then `pytest.main(['-q', 'tests/test_numpy_compat.py', 'tests/test_cuts.py',
    'tests/test_applicator.py'])`: 140 passed, 276 skipped, nothing errored, numpy never imported
    (8.57 s; after the review's fixes)
  * the gate after the review's fixes, from the worktree root, the second call split in three by
    file for the tool's time limit (`python` is the env's): `python -m pytest -q tests/itf1788`
    18246 passed in 80.64 s; `python -m pytest -q tests/test_[a-l]*.py` 1475 passed in 312.48 s,
    `python -m pytest -q tests/test_[m-z]*.py` 2829 passed in 690.57 s, and `python -m pytest -q
    intervals README.md tests/conftest.py tests/exhaustive_modulo.py tests/exhaustive_ops.py
    tests/oracles.py tests/strategies.py` 102 passed in 0.60 s: the three are what
    `--ignore=tests/itf1788` collects (4406, checked with `--collect-only`), 4406 passed in 1003.65
    s (482 s at M15 unloaded; `--collect-only` over the whole tree: 22652)
  * the hook's cost, `timeit` best of 5 x 2000 on `A = MultiInterval(1, 2)`, the package at
    `04946af` against this build: `np.float64(2) == A` 0.24 µs to 4.70 µs (numpy's override
    machinery now runs, then the identity fallback: it matters for `x in list_of_sets` with numpy
    numbers), `np.float64(2) < A` 8.5 to 13.9 µs; `np.float64(2) + A` 60.6 to 68.6 µs, inside the
    noise (`2.0 + A`, untouched, 53.3 to 73.2 µs in the same runs); `normalize_value(0.1)` 3.0 to
    2.4 µs (the float path: one `isinstance` more, noise); `normalize_value(np.float32(0.1))` 2.0 to
    12.3 µs (a `Fraction` built per foreign value)
  * the no-change and enclosure censuses (probes in `.scratch/`, not kept, so these numbers cannot
    be regenerated from the tree): the design's prototype against `04946af`, 5292 cases (14 numpy
    scalars x 9 objects x 21 operators incl. `@`, `<<`, `>>`, `in`, both sides): 0 differ in result,
    exception type or warning categories (3110 results and 2182 exceptions at `04946af`;
    2026-09-27). the soundness review against `e7bb3fb` (2026-09-28): 11 numpy scalars x 12 objects
    x 19 operators, both sides, identical to `04946af` except `Dual ** <numpy scalar>` (B1 and its
    nearest-class change); `Dual.variable(O(u)) ** r` and `DecoratedInterval(O(u))`'s, 19 exponents
    x 11 bases, value and derivative against arb at 400 bits: `04946af` misses 54 of 418, the build
    0; 1636 ufunc enclosure checks against arb at 300 bits (`hypot arctan2 minimum power add
    subtract multiply divide` with numbers, numpy scalars and float32/float16 arrays, both orders;
    `sqrt exp log sin arctan cbrt expm1 log1p exp2 reciprocal square` on `O` and `D(O)`): 0 bad
    (both-ours method pairs aside: F2); the elementwise path equals the scalar path for float32,
    float16, float64, longdouble, int8 and uint64 arrays over 8 ufuncs; `log(x, base)` and
    `Dual.log(base)` over 14 bases x 8 sets unchanged from `04946af`
* **sabotage** (`.scratch/sabotage.py` in the worktree, the M15 harness adapted, and
  `.scratch/fix/sabotage2.py` for the review: each break alone, `.hypothesis` cleared,
  `tests/test_numpy_compat.py tests/test_autodiff.py tests/test_cuts.py tests/test_applicator.py`
  with `-x` and a 900 s timeout, the file restored and compared; 2026-09-28). the review's rows ran
  first against an export of the build's commit `e7bb3fb` (every one green: 567 passed), and the
  final run is all 47 against a copy of the fixed tree (the `arctan2` swap re-anchored on the new
  line). the last column is the first test to fail under `-x`:

| break | first run | final run: red by |
|---|---|---|
| `import numpy` at the top of numpy_compat | red | red: `tests/test_numpy_compat.py::test_numpy_is_never_imported_at_load` |
| `bitwise_or` dropped from the table | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-or]` |
| identity fallback of `equal` removed | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-eq]` |
| `less` reflected as `__lt__` | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[longdouble-lt]` |
| reflected call made with the forward dunder | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-divmod]` |
| 0-d arrays not unwrapped | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-ge]` |
| both ours: forward dunder first, no subclass rule (B3) | red | red: `tests/test_numpy_compat.py::test_binary_ufunc_is_the_method_or_the_operator[add]` |
| table keyed by `ufunc.__name__` (B3) | red | red: `tests/test_numpy_compat.py::test_the_table_is_keyed_by_the_ufunc_object` |
| dunder lookup through the metaclass (`type.__or__`) | green | red: `tests/test_numpy_compat.py::test_a_missing_dunder_is_numpys_refusal` |
| `arcsin` mapped to `acos` | red | red: `tests/test_numpy_compat.py::test_unary_ufunc_is_the_method[arcsin]` |
| `square` as `x * x` | red | red: `tests/test_numpy_compat.py::test_unary_ufunc_is_the_method[square]` |
| `rint` as `round_ties_away` | red | red: `tests/test_numpy_compat.py::test_ufunc_pinned_examples` |
| `arctan2` with swapped arguments | red | red: `tests/test_numpy_compat.py::test_binary_ufunc_is_the_method_or_the_operator[arctan2]` |
| `invert` dropped | red | red: `tests/test_numpy_compat.py::test_unary_ufunc_is_the_method[invert]` |
| `fmod` mapped to `%` | red | red: `tests/test_numpy_compat.py::test_unmapped_ufuncs_and_forms_are_type_errors` |
| `fmin`/`fmax` mapped to `minimum`/`maximum` | red | red: `tests/test_numpy_compat.py::test_unmapped_ufuncs_and_forms_are_type_errors` |
| keywords accepted (`out=` ignored) | red | red: `tests/test_numpy_compat.py::test_unmapped_ufuncs_and_forms_are_type_errors` |
| method `reduce` accepted | red | red: `tests/test_numpy_compat.py::test_unmapped_ufuncs_and_forms_are_type_errors` |
| a list taken for an array | red | red: `tests/test_numpy_compat.py::test_unmapped_ufuncs_and_forms_are_type_errors` |
| elementwise `equal` (array path) | red | red: `tests/test_numpy_compat.py::test_equality_never_broadcasts` |
| elementwise path returns NotImplemented | red | red: `tests/test_numpy_compat.py::test_ndarray_and_interval_elementwise[add]` |
| elementwise element with swapped operands | red | red: `tests/test_numpy_compat.py::test_ndarray_and_interval_elementwise[divide]` |
| `errstate` removed around the object loop | green | red: `tests/test_numpy_compat.py::test_elementwise_leaves_numpys_float_flags_alone` |
| `__array__` removed | red | red: `tests/test_numpy_compat.py::test_equality_never_broadcasts` |
| `__array__` returns an array for `copy=False` | red | red: `tests/test_numpy_compat.py::test_arrays_hold_intervals_as_elements` |
| `warn` stops at numpy_compat frames | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-add]` |
| `normalize_value` back to `float()` for foreign reals | red | red: `tests/test_numpy_compat.py::test_foreign_reals_are_exact` |
| exact path taken for doubles too | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float32-add]` |
| a foreign Rational by the value rule, not exact by type | red | red: `tests/test_numpy_compat.py::test_foreign_reals_are_exact` |
| `_check_degree` back to `int` only | red | red: `tests/test_numpy_compat.py::test_integer_arguments_take_numpy_ints` |
| `ndigits` back to `int` only | red | red: `tests/test_numpy_compat.py::test_integer_arguments_take_numpy_ints` |
| `pown_rev` back to `int` only | red | red: `tests/test_numpy_compat.py::test_integer_arguments_take_numpy_ints` |
| `_check_base` back to int, float, Fraction | red | red: `tests/test_numpy_compat.py::test_foreign_real_specials` |
| `_check_base` without `normalize_value` | red | red: `tests/test_numpy_compat.py::test_foreign_real_specials` |
| B1: restore `exponent - 1` (non-integral) | red | red: `tests/test_numpy_compat.py::test_numpy_exponent_of_a_dual` |
| B1: restore `exponent - 1` (integral) | red | red: `tests/test_autodiff.py::test_pow_integral_exponent_derivative_is_exact[1.152921504606847e+18]` |
| B2: drop `int(n)` in `Dual.rootn` | red | red: `tests/test_numpy_compat.py::test_dual_rootn_takes_the_degree_as_an_int` |
| B1 broken in the decorated class only (review F4) | green | red: `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses[0.1-1e+300-decorated outward]` |
| `Dual ** 0.0` not the constant-1 case (review SAB-R21) | green | red: `tests/test_autodiff.py::test_pow_any_zero_is_the_constant_one[u0-0.0]` |
| `int(ndigits)` dropped in `step` (review SAB-R19) | green | red: `tests/test_numpy_compat.py::test_numpy_ndigits_past_int64` |
| `int(n)` dropped in `pown_rev` (review SAB-R20) | green | red: `tests/test_numpy_compat.py::test_numpy_pown_rev_degrees` |
| `errstate(all=...)` narrowed to `over=` (review SAB-R9) | green | red: `tests/test_numpy_compat.py::test_elementwise_leaves_every_float_flag_alone` |
| a real without `as_integer_ratio` refused (review SAB-R16) | green | red: `tests/test_numpy_compat.py::test_a_foreign_real_without_a_ratio_is_its_float` |
| integral exponent: multiplier the float, not the int n (review F3) | green | red: `tests/test_autodiff.py::test_pow_number_exponent_nearest_examples` |
| nearest class keeps M15's float `r - 1` (review F3) | green | red: `tests/test_autodiff.py::test_pow_number_exponent_nearest_examples` |
| both ours: `hypot minimum maximum` without the subclass rule (review F2/F5) | red | red: `tests/test_numpy_compat.py::test_binary_ufunc_is_the_method_or_the_operator[maximum]` |
| both ours: `arctan2` without the subclass rule (review F2/F5) | red | red: `tests/test_numpy_compat.py::test_both_ours_methods_are_subclass_first` |

the two green in the first run were gaps, each closed by a test added the same session and the break
re-run red: the metaclass lookup (every class answers `hasattr(cls, '__or__')` through
`type.__or__`, so the break only changed which TypeError numpy raised:
`::test_a_missing_dunder_is_numpys_refusal` pins numpy's refusal), and `errstate` (the property test
had found the numpy RuntimeWarning by chance once and then drew past it:
`::test_elementwise_leaves_numpys_float_flags_alone` pins the example). the first run's B1
(non-integral) row was red by `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses`
alone: the property test 2 did not draw a `Dual` with a float32 exponent in its examples, so
`tests/test_numpy_compat.py::test_numpy_exponent_of_a_dual` pins B1 through numpy too (the final run's first red).
the eight green first runs marked "review" were the review's (see the review below): each closed by a test
and re-run red. the two subclass-rule rows are new code (Q15(h)); their test,
`tests/test_numpy_compat.py::test_both_ours_methods_are_subclass_first`, was red against the
build's `numpy_compat.py` before the fix (with `::test_binary_ufunc_is_the_method_or_the_operator[maximum]`); that `[maximum]` is a hypothesis draw: run alone under the break, `tests/test_numpy_compat.py::test_binary_ufunc_is_the_method_or_the_operator` stayed green (21 passed; the verifier, 2026-09-28), so the pin is `::test_both_ours_methods_are_subclass_first`, red under both subclass-rule breaks, and the property test's oracle `::_binary_oracle` is a second, chance catch

* review (2026-09-28, three read-only reviewers over `e7bb3fb`, lenses soundness, sabotage-audit
  and spec/regression; each finding reproduced on this branch before any change, probes in the
  worktree's `.scratch/fix/`). **no unsound result in the build**: the M15 hole is closed in the
  bare and decorated outward classes (0 of 9 pairs miss at 400 bits, both kinds). found and fixed
  (id, lens, disposition, evidence):
    * **soundness F1 = spec F1** (fixed): the record and the docstring of
      `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses` said M15 missed "6 of
      9" pairs; the test's own oracle at 400 bits against `04946af` misses 5 (0.1 with 1e300 and
      1e-300, 1e-20 with 2, 0.3 with 1e300 and 1e-300), the build 0. both now say 5 and name them
    * **soundness F4** (fixed): the B1 pin covered the bare outward class only; a B1 break in the
      decorated class alone stayed green (567 passed on `e7bb3fb`), and `04946af` missed the same 5
      pairs there. `::test_pow_number_exponent_derivative_encloses` now runs both kinds (18 items);
      the break is red by its `[0.1-1e+300-decorated outward]`
    * **sabotage SAB-R21** (fixed): the zero exponent's constant-1 case narrowed to an int zero
      stayed green, and `Dual.variable(M(0)) ** 0.0` then has the derivative `{}` where it is 0.
      pinned by `tests/test_autodiff.py::test_pow_any_zero_is_the_constant_one` (0, 0.0, -0.0,
      `Fraction(0)` over `M(0)`, `O(0)`, `M(-1, 1)`, `D(M(0))`; 16 items)
    * **sabotage SAB-R19** (fixed): dropping `int(ndigits)` in `steps.py::step` stayed green; an
      int64 `10 ** ndigits` wraps past 10 ** 18. pinned by
      `tests/test_numpy_compat.py::test_numpy_ndigits_past_int64` (19, 20, 30 digits)
    * **sabotage SAB-R20** (fixed): dropping `int(n)` in `reverse.py::pown_rev` stayed green
      (`pown_rev(M(1, 4), np.int64(3))` an `OverflowError`). pinned by
      `tests/test_numpy_compat.py::test_numpy_pown_rev_degrees`
    * **sabotage SAB-R9** (fixed): `np.errstate(all=...)` narrowed to `over=` stayed green; the
      library leaves underflow set too, which only a caller's `np.errstate(all='raise')` shows.
      pinned by `::test_elementwise_leaves_every_float_flag_alone`
    * **sabotage SAB-R16** (fixed): the fallback of `cuts.py::_exact_value` (a real with no
      `as_integer_ratio()`) was untested, and the docstring of `cuts.py` and this record's design
      said "never `float()` of it", false there. pinned by
      `tests/test_numpy_compat.py::test_a_foreign_real_without_a_ratio_is_its_float` (`::Plain`);
      both texts now say a real with no ratio is `float()` of it, as before
    * **sabotage SAB-R10** (fixed by deletion): the `except TypeError` round the table lookup in
      `numpy_compat.py::array_ufunc` guarded an unhashable ufunc numpy never passes (`KeyError` in
      its place stayed green); deleted, so there is no line left to pin
    * **soundness F2 = spec F5** (fixed, Q15(h)): with both operands ours, `hypot minimum maximum
      arctan2` took the first operand's method, so `np.hypot(M(0.1), O(0.1))` was the nearest
      `[0.1414213562373095]`, missing the true value, while the operators put the subclass first.
      `numpy_compat.py::_subclass_first` now gives the method ufuncs the operators' rule; pinned by
      `tests/test_numpy_compat.py::test_both_ours_methods_are_subclass_first` (red against the build's hook) and by the
      property test's oracle `::_binary_oracle`, which takes the same rule; two sabotage rows
    * **soundness F3 = spec F2** (fixed, documented and pinned; the change kept): the B1 fix also
      changes the nearest and decorated classes (an exact derivative for an integral float
      exponent; pow, not pown, for r = 1e-20). kept, because each agrees with the value's own
      reading of r (the value of `Dual.variable(M.parse('[1/3, 3]')) ** 2.0` is the exact `[1/9,
      9]`, and over `M(-2, -1)` the value of `** 1e-20` is empty, where M15 gave a derivative);
      stated under "what the build found" and pinned by
      `tests/test_autodiff.py::test_pow_number_exponent_nearest_examples`: both of the reviewer's
      alternatives (the float multiplier; M15's `r - 1` in the nearest class) stayed green on
      `e7bb3fb` and are red now
    * **spec F3** (fixed): two bare `::` citations named the wrong file under the record's
      convention; both are written in full as `tests/test_numpy_compat.py::...`
    * **spec F4** (deferred, as the build recorded it): r = `2.0 ** 60` is pinned at u = -1 only,
      because `O(u) ** 2 ** 60` does not finish for u in {2, 1e300, 1e-300} (an exact power,
      pre-existing at `04946af`); it stays under `HANDOFF.md` "still owed"
    * **coverage, not findings**: the sabotage lens re-ran 10 of the build's 37 rows on an export of
      `e7bb3fb` (each red by the test the table names) and 30 breaks of its own (24 red). of the 6
      green, R9, R10, R16, R19, R20 and R21 are the gaps above, and two are equivalent mutants: 0-d
      arrays sent down the elementwise path (R6: `frompyfunc` over a 0-d input gives the same scalar
      object and the same errors) and `square` as `x ** 2.0` (R26: pown either way; `M(3) ** 2` and
      `** 2.0` have the same repr). the soundness lens's own breaks were red (S1: the outward
      exponent's point set made in the nearest class; S3: a foreign real kept exact only past the
      doubles; S4: `arctan2`'s number y coerced to `MultiInterval`), except S5, which is F4

**M16e the gmpy2/mpfr backend: `backend.py`, `_gmpy2.py` (done 2026-09-28)** (D24).

the owner, 2026-09-27: "get the rest of h3 done" (H3), superseding 2026-09-26's "recorded, not
now". H3's rest is M16, five streams; this is M16e. the design is `v2-plan.md` "elementary and step
functions" (the backend bullets); the choices are D24, open as Q16. here the spec, the exit and the
record. what it blocks: nothing. every answer keeps today's behaviour (the pure path) unless a user sets the
variable; Q16(a) (default automatic) and Q16(f) (in 2.0 or not) are the only ones whose answer
would change what a user without the variable sees, and neither blocks H1.

* **`intervals/backend.py`**: `INTERVALS_BACKEND` read at import (`::_select`: unset, `''`,
  `python` → pure; `gmpy2` forced; `auto`; else `ValueError`), `NAME`, `fast`, `name()`, the version
  floor `::_supported` (gmpy2 `>= 2.3`, a pre-release counting as just before its release, a
  `+local` label read on a release or a pre-release; MPFR `>= 4.2`; `auto` also `< 3`; an
  unparsable string unsupported), `::_load` (lazy import of
  `_gmpy2`), `::_use` (the tests' switch)
* **`intervals/_gmpy2.py`**: `rounded`, `rounded_pow`, `rounded_angle`, `rounded_inverse_trig`,
  `outward`; each a float or None. the exact input (`::_operand`, `::_ratio`, `::_int`), the two
  guards (`::_value`: nan, and a ternary value of 0 on an elementary result), `BOUND = 2**20`,
  `ROOTN_LIMIT = 2**31`
* **the dispatches**: `elementary.rounded` after `_beyond`, `rounded_pow` after its two range
  shortcuts, `rounded_angle` after `q == 0 and m == 0`, `rounded_inverse_trig` after its exact k = 0
  case; `ops.outward` keyed on the descriptor object (`ops._FAST_OPS`). each reads `backend.fast` at
  call time. `pyproject.toml`: `fast = ["gmpy2>=2.3"]`, `gmpy2>=2.3,<3` in `test`
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
    * **the design's probes** (2026-09-27/28, a throwaway prototype patched in): the hardest of
      200000 random dyadic x per function was only `2**-17.4` (exp), `2**-21.4` (log), `2**-22.1`
      (sin) and `2**-19.7` (atan) half-ulps from a double or a midpoint, so class 14 drawn at random
      would have been empty (C3); those four x close `::HARD`. a `-0.0` slip is invisible to the
      rest of the suite: the prototype with `+ 0.0` dropped passed `test_oracle_flint`,
      `test_elementary`, `test_functions`, `test_outward`, `test_extreme_floats` and `test_reverse`
      (1191 passed, 55716 answers from the backend), since `-0.0 == 0.0` and `cuts.py` normalizes it
      at set level; only the sign-bit check of `tests/test_backend.py` sees it (S1). of the
      prototype's 30 µs for `rounded('exp', 0.7)`, `exact` took 8, `_beyond` 3.4, building the mpfr
      2.7, the MPFR call 2.7 and `float()` 1.8: the pure prelude the contract keeps is most of a
      backend call. `round_rational` through `mpfr(mpq, 53, ctx)` was bit-identical over 60000
      probes, so leaving it pure (1.0-2.1x) is a speed call only
    * **the class-15 test at the real bound hung the first run**: the pure path at a tiny x past
      `2**20` bits grows about quadratically for sin, exp, atan and pow (at `2**16` bits: 3.3 s for
      sin, 5.5 s for exp, 2026-09-28, the build's timing; 3.46 s and 1.46 s, 5.94 s for the whole
      list, at 10:03 the same day, loaded, `tools/backend_speed.py --bound`), so a million bits would
      take ~15 min. the
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
* **tests** (`tests/test_backend.py`, 735 tests since the review, 717 at the build, and `tests/test_elementary.py::test_a_missed_exact_case_raises_instead_of_looping`
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
      class incl. `csch(-746)`, C4's `atan(5e-324)` with sign -1, the angle's atan2 route at
      `(-5e-324, 0)` and the hook's mpq route at `mul(-5e-324, 1/3)`, 4 overflow, 5 wide exact inputs,
      6 non-dyadic, 7 infinite x, 8 every `(q, m)` of `functions._angle` and some it never makes, 9
      inverse trig, 10 rootn incl. `2**31 - 1`, `2**31`, `2**32`, `2**64 + 1`, 11 pow, 12 mixed hook
      operands; 179 cases) in `::test_edge_class`, all again in `::test_hostile_global_context` (13, guarded by
      `::test_the_hostile_context_is_hostile`), 14 `::HARD` in `::test_hard_point` (78 points, each
      asserted answered by the backend), 15 `::test_past_the_bound_is_declined_at_a_small_bound` and
      `::test_past_the_bound_is_declined_at_the_real_bound`; `test_oracle_flint.py::EXTREMES` in
      `::test_extreme_point`
    * coverage: `::test_backend_answers_where_it_should` (a point per row of the table),
      `::test_backend_declines_where_it_should`, `::test_power_descriptors_decline`
    * rules: `::test_the_shortcuts_run_before_the_backend`, `::test_the_hook_is_keyed_on_the_descriptor`,
      `::test_missed_exact_case_raises` (all three directions) and its twin
      `::test_missed_exact_case_raises_in_pow_and_inverse_trig` (`rounded_pow(4, 1/2)`,
      `asin(0)`, `atan(0)` with sign -1), `::test_a_domain_slip_raises_instead_of_returning_nan`,
      `::test_inputs_it_does_not_know_are_declined`
    * set level: `::test_set_level_matches_unary` (the 30 methods, `reciprocal`, `** 3`, `** -2`,
      `** 2.5`, `rootn` 2 3 -2 -3, `log` to 1/2, 0.25, 3, 2.5), `::test_set_level_matches_binary` (+ -
      * /, atan2, pow, hypot, `sin_rev`, `cos_rev`, `tan_rev`, `pow_rev2`),
      `::test_set_level_matches_newton`, `::test_set_level_matches_newton_sin`; both classes
    * the switch: `::test_use_switches`, `::test_env_var` (11 cases in a subprocess, the variable
      removed from the inherited env), `::test_forced_gmpy2_never_falls_back`, `::test_version_floor`
      (20 cases), `::test_use_restores` (nested and on an exception: the rest of the suite runs after
      this file in one process), `::test_the_test_extra_installs_what_auto_takes` (`[test]`'s pin)
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
    * **after the review's fixes, default (pure), 2026-09-28 10:14-10:30** (the same three calls;
      the x10 fuzz below ran beside the first two): `tests/itf1788` 18246 passed in 77.1 s; 3269
      passed in 442.2 s; 1554 passed in 420.6 s; so 4823 in 862.8 s. `tests/test_backend.py` alone:
      735 passed in 43.4 s, and 735 in 41.6 s with `INTERVALS_BACKEND=gmpy2`. the forced whole gate
      was not re-run: the review changed the library only in `backend.py::_PRE` (a version string)
      and comments. `pytest --collect-only -q` over the whole tree in one process: 23069 items
    * **the x10 fuzz of the differential** (`HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10 python -m
      pytest -q tests/test_backend.py`, 2026-09-28 10:14-10:20, beside the gate): 735 passed in
      387.5 s; the spec review had 579.5 s for the 717 at `adeeb97`, loaded. `fuzz.yml` runs it with
      the rest (the last whole x10 run was 5037 s, `HANDOFF.md` M14-run), so about 6-10 min more
      against its 180-min timeout
* **speed** (`tools/backend_speed.py`, tracked since the review; the build ran the same loop from
  `.scratch`: pure then gmpy2 back to back under `backend._use`, best of 5 per call; 2026-09-28 09:07, after this stream's gate, with other streams' runs on the laptop, so ratios,
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
  set level 2-3.5x for exp, log and atan over a multi-interval, 1.1x for sin, about 1.0-1.3x for
  outward `+ * /` and 1.2-1.4x for newton, where the applicator's and the kernel's own python
  dominate. the spec review's re-run (2026-09-28 09:28, loaded) matched the per-call ratios roughly
  and had `A + B` at 0.72x once and 1.0-1.4x over six repeats: at set level the arithmetic gain is
  within the noise of a loaded laptop

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
| S1c drop `+ 0.0` (hook, the mpq route; review T1) | **green** | red: `tests/test_backend.py::test_edge_class[3-outward-'mul'--5e-324-Fraction(1, 3)]` |
| S1d drop `+ 0.0` (angle, the atan2 route; review G4) | **green** | red: `tests/test_backend.py::test_edge_class[3-angle--5e-324-0]` |
| S13b `rounded_pow` without the `rc == 0` guard (review G1) | **green** | red: `tests/test_backend.py::test_missed_exact_case_raises_in_pow_and_inverse_trig[exact_pow-<lambda>]` |
| S13c `rounded_inverse_trig` without the guard (review G3) | **green** | red: `tests/test_backend.py::test_missed_exact_case_raises_in_pow_and_inverse_trig[exact-<lambda>0]` |
| S13d `rounded_angle`'s atan2 without the guard (review G2) | green | green: equivalent, not kept (unreachable, see the review below) |
| S13e `rounded_angle`'s pi without the guard | green | green: equivalent, not kept (pi is irrational) |
| S14b `_use` never restores (review R11) | **green** | red: `tests/test_backend.py::test_use_restores` |
| S17g `dev` dropped from `_PRE` (review R7) | **green** | red: `tests/test_backend.py::test_version_floor[2.3.1.dev1-MPFR 4.2.2-True-True]` |
| S17h `+local` dropped from `_FINAL` (review R8) | **green** | red: `tests/test_backend.py::test_version_floor[2.3.1+local-MPFR 4.2.2-True-True]` |
| S17i `+local` dropped from `_PRE` (new with the fix) | red (the row, on `adeeb97`) | red: `tests/test_backend.py::test_version_floor[2.3.1rc1+local-MPFR 4.2.2-True-True]` |
| B2b the bound off by one, `> BOUND + 1` (review R9) | **green** | red: `tests/test_backend.py::test_past_the_bound_is_declined_at_a_small_bound` |
| V1 `[test]` without the `<3` pin (new with the fix) | red (the test, on `adeeb97`) | red: `tests/test_backend.py::test_the_test_extra_installs_what_auto_takes` |

the one green in the first run was a gap, as the design predicted: moving the dispatch before
`_beyond` changes no value (MPFR agrees with `_beyond`, which is why no differential sees it), only
cost and provenance. closed by `::test_the_shortcuts_run_before_the_backend` (a spy on
`_gmpy2.rounded` and `_gmpy2.rounded_pow`, never called for `exact`'s values, `_beyond`'s, the pi
limits at ±inf or `rounded_pow`'s range shortcuts), and the break re-run red. N2's first run was red
for the wrong reason (the replacement left a syntax error: red at collection); rewritten as intended
and re-run red by its test. S9, S14, S15, C1 and C2 were re-run against their named guard alone,
since under `-x` an earlier test caught them first; each went red there too. the rows from S1c on
come from the review (2026-09-28): their first run is the break on `adeeb97` (a git-archive copy,
718 passed for each green one), their final run the same harness on the fixed tree (736 items
unbroken; S13d and S13e 736 passed, as they must)

review: three read-only reviewers over `adeeb97` (2026-09-28; lenses soundness, sabotage-audit and
spec/regression). the sabotage lens re-ran 10 of the table's rows (S1b S4b S7b S8 S13 S16b S18 C4 B2
N5): each red on the test the table names, so no row was false. their own breaks that went red at once on `adeeb97` (2026-09-28, from their notes): the sabotage lens 12 of its 20 (R5 the hook's mpq route not subnormalized, R6 the inverse-trig dispatch before its exact k = 0 case, R10 the bound on the numerator only, R12 acot of a negative x as `atan(1/x)`, R13 `ROOTN_LIMIT = 2**32`, R14 a strict MPFR floor, R15 the hook's reciprocal as `x / 1`, R16 dyadic Fractions declined, R17 the ieee contexts with a wide exponent range, R18 the angle's pi at `(0, 2)` rounded the other way, R19 `auto` with no floor, R20 an int past `2**53` built at 53 bits; its 8 greens are the findings below), the soundness lens 5 (T3 the inverse-trig dispatch before `exact`, T8 and T9 the hook raising on an exact result on the mpq and on the dyadic route, T10 `_ratio`'s finiteness check dropped, T18 the bound on the numerator only). the spec lens found the set-level differential not vacuous: 40 draws per method of `::UNARY` and `::BINARY`, none raising under either backend. **no wrong double, flag or sign bit
in the backend as built**: the soundness lens checked about 41k primitive answers against arb, about
118k hook answers exactly and 1200 set-level `repr`s under both backends, 0 wrong; the spec lens ran
`tests/itf1788`, `tests/test_elementary.py` and `tests/test_backend.py` under
`INTERVALS_BACKEND=gmpy2`, 19354 passed (both 2026-09-28, from their notes). every finding below was reproduced on `adeeb97` first (a git-archive
copy, the break alone, `tests/test_backend.py` and the pure twin under `-x`, 2026-09-28: 718
passed for each green one), then fixed with a test seen red, or rejected with evidence. ids are the
reviewers' own, prefixed by lens, since three lenses reused `F1`:

| id | lens | finding | disposition | evidence |
|---|---|---|---|---|
| soundness F1, sabotage F3, spec G4G5 | all three | `+ 0.0` (C4) unpinned on the hook's mpq route and on the angle's atan2 route: MPFR gives `-0.0` at `outward('mul', (-5e-324, 1/3), UP)` and at `rounded_angle(-5e-324, 0, UP)` | fixed | reproduced: T1 and G4 green (718 passed each). five class-3 `::EDGES` cases added (`angle (-5e-324, 0)`, `angle (-1/2**1100, 0)`, `mul(-5e-324, 1/3)`, `mul(-1/10**400, 1e-300)`, `add(-5e-324, 2/(3 * 2**1074))`); T1 now red by `::test_edge_class[3-outward-'mul'--5e-324-Fraction(1, 3)]`, G4 by `::test_edge_class[3-angle--5e-324-0]` |
| soundness F2, sabotage F1, spec G1, spec G3 | all three | the missed-exact guard (rc 0 raises) pinned only on `rounded`; dropped at `rounded_pow` or `rounded_inverse_trig`, nothing went red | fixed | reproduced: G1 and G3 green. `::test_missed_exact_case_raises_in_pow_and_inverse_trig` (`exact_pow` or `exact` patched to None: `rounded_pow(4, 1/2)`, `asin(0)`, `atan(0)` with sign -1, every direction); G1 and G3 now red by it. the angle's guard (spec G2) is **rejected as a gap**: G2 and G2b green before and after, since it cannot be reached (`rounded_angle` calls atan2 only with y != 0, and atan of a nonzero rational plus a multiple of pi is irrational, as is pi); the guard stays as defence |
| sabotage F2 | sabotage | `_use` never restoring (`finally: pass`) stayed green: `::test_use_switches` reads `before` after earlier tests left gmpy2 on, so every file after `test_backend.py` in the gate's one process could run on gmpy2 unseen | fixed | reproduced: R11 green. `::test_use_restores` (nested both ways, and on an exception); R11 now red by it |
| sabotage F4 | sabotage | a dev build of a supported release and a `+local` label unpinned (`dev` dropped from `_PRE`, `+local` from `_FINAL`: green); `2.3.1rc1+local` read as unsupported | fixed | reproduced: R7, R8 green; `::test_version_floor[2.3.1rc1+local-...]` red on `adeeb97`. `backend.py::_PRE` takes a `+local` label too; three rows added (`2.3.1.dev1`, `2.3.1+local`, `2.3.1rc1+local`: 20 cases). R7, R8 and S17i (the new group dropped) now red by their rows |
| sabotage F5 | sabotage | the bound's exact edge untested: the "just past" operands had b + 2 bits, so `> BOUND + 1` stayed green | fixed | reproduced: R9 green. `::_bound_cases` now uses `2**b` and `1/2**b` (b + 1 bits); R9 (B2b) now red by `::test_past_the_bound_is_declined_at_a_small_bound` |
| soundness F3 | soundness | the bound's rationale said a dyadic past MPFR's range "flushes to 0 or inf with a ternary value of 0" | fixed (wording) | reproduced (gmpy2 2.3.1 / MPFR 4.2.2, 2026-09-28): `mpfr(2**(2**30 + 1), 2, context())` is `inf` with rc 1; `mpfr(mpq(1, 2**(2**30 + 1)), 2, context())` is `0.0` with rc 0. now "to 0 with a ternary value of 0 (to inf with 1), silently" in `_gmpy2.py`'s docstring, `tests/test_backend.py`'s class-15 comment and `v2-plan.md` "elementary and step functions"; the conclusion (decline) stands |
| sabotage F6 | sabotage | the rootn bound's reason: gmpy2 takes n as a C `unsigned long`, accepts `2**31` and `2**32 - 1`, and raises `OverflowError` only from `2**32` (windows); `ROOTN_LIMIT`'s comment said "a C long" | fixed (wording) | reproduced (same versions, 2026-09-28): `ieee(64).rootn(mpfr(2), 2**31)` is 1.0000000003227718, `2**32` raises. `2**31` kept as a margin that holds on every platform; `_gmpy2.py`, the class-10 comment, `v2-plan.md` "elementary and step functions" and its decision-log revision say so |
| spec R1 | spec | the readme example had no blank line before the closing fence, so doctest read the fence as expected output | fixed | reproduced: the section extracted and run with `python -m doctest` failed under both backends (1 of 3). blank line added; 3 passed under `INTERVALS_BACKEND=python` and `gmpy2` |
| spec V1 | spec | `[test]` had `gmpy2>=2.3` with no ceiling while `auto` takes only `< 3`: a gmpy2 3 on PyPI turns `test_env_var`'s auto row red in every CI job | fixed | `[test]` now `gmpy2>=2.3,<3`; `::test_the_test_extra_installs_what_auto_takes` reads `pyproject.toml` and asserts the pin equals `backend.FLOOR`/`CEILING` and that the installed gmpy2 is in the window: red on `adeeb97`, and red again with the pin removed (the table's V1). `[fast]` stays unpinned (Q16(d)) |
| spec F1 | spec | the record says `fuzz.yml`'s x10 run fuzzes the differential, with no cost | fixed (a number) | the spec lens measured `tests/test_backend.py` under `HYPOTHESIS_PROFILE=fuzz FUZZ_MULTIPLIER=10` at 579.47 s (717 passed, loaded); re-measured on the fixed tree 2026-09-28 10:14-10:20, beside this fix's gate: 735 passed in 387.51 s. with the last whole x10 run at 5037 s (`HANDOFF.md` M14-run, 2026-09-27) that is about 5400-5600 s against `fuzz.yml`'s 180 min; in "measured" above, `HANDOFF.md` Q16(e) and "still owed" |
| spec S1 | spec | the speed table and the class-15 cost cited gitignored `.scratch` scripts | fixed | the loop is now tracked, `tools/backend_speed.py` (and `--bound` for the class-15 cost; run 2026-09-28 10:03, loaded: at `2**16` bits sin 3.46 s, exp 1.46 s, 5.94 s for the list; `2**20` cheap 0.15 s). the set-level `+ * /` claim softened to about 1.0-1.3x, with the spec lens's 0.72x-1.4x re-runs |

### pown-huge: pown with a huge integral exponent (done 2026-09-29)

HANDOFF row 2 (found by the M16b review, F4, and by M16d's build). `A ** n` for an integral n
(`ops.power` -> `ops._power_descriptor`, reached through `MultiInterval`/`OutwardMultiInterval`
`__pow__`, `DecoratedInterval`, `Dual`, `ieee1788.pown` and `Interval.__pow__`, numpy's `power`)
formed the exact `Fraction(x) ** n` of every float corner in the outward class (`ops.outward`'s
`exact`), so `O(0.5) ** (2 ** 31 - 1)`, `O(2.0) ** 2 ** 60`, `O(0.5) ** 1e20` (an integral float
exponent is the int) never finished; and a base just above 1 never saturates
(`O(1.0000000000000002) ** 10 ** 9`), so no range check alone was the fix. two designs, one
critique, then this build, all 2026-09-29, in the worktree `intervals-pown-huge` (branch `pown-huge`
off `v2` at `7288e81`).

**the designs.** A, directed binary powering (squarings cut to p bits toward the bound, ziv doubling
p), and B, `elementary.rounded_pow` (1788's pow route: `exact_pow` while the power is at most
`EXACT_POWER_LIMIT` bits, then `_ln_bracket`'s range shortcuts, then ziv over `exp(n ln |x|)`) with a
marker for attainment. the critique (read-only, adversarial) found no unsound end and no wrong flag
in either: each agreed with MPFR on 63000 random points (bases over the whole range, subnormals,
1 ± ulps, negatives; n from 1 to 2**64, 2**k ± 2, 10**20 to 10**300, both signs) and with each other
on all 54 outward/1788 cases of a 58-case adversarial sweep; on the 30 cases today's code finished
they equalled today in values and flags. B was chosen:
* A's cost grows about cubically in bitlen(n), since it checks the range only after powering (C2,
  one corner: 0.105 s at `O(1.5) ** 2 ** 4000`, 0.64 s at `2 ** 8000`, 2.89 s at `2 ** 14000`,
  38.3 s at `2 ** 30000`, past 120 s at `2 ** 60000`; B 0.4 to 3 ms on the same, 2026-09-29, loaded)
* A's own unsound-direction mutants went red only in white-box checks, so its new numerics could
  only ever be pinned by internal tests; B adds none (`rounded_pow` is pinned by `pow_`, the itf1788
  vectors and the MPFR tests, and has a backend)
* B saturates in O(1) through `_ln_bracket`, and the designer's full gate with the prototype
  installed as a plugin was 27795 + 5537 = 33332 passed (2026-09-29)

three grafts from the critique: C4 (the nearest fix at a zero base, below), C3 (a bounded
descriptor name, below) and, from A, its O(1) `is_double` and its exact-double rows as test oracles
only. the critique also refuted "extend the M16d pin to u in {2, 1e300, 1e-300}" twice (C1): `O(2)`
holds an exact int (the pin would hang under both designs; it must be `O(2.0)`), and at those three
bases `u ** (2 ** 60 - 1)` and `u ** 2 ** 60` give the same saturated enclosure, so the B1 break
stays green there; only a negative base or one near 1 tells them apart.

**what was built** (`intervals/ops.py` only; the applicator, `elementary`, `_gmpy2` and the class
layers are unchanged):
* `_exact_power_descriptor(n)` is the old body (fn, pole, split points byte for byte);
  `_power_descriptor(n, rounds_outward)` (still `lru_cache(64)`) dispatches
* outward: `base._replace(fn=exact, rounded=(hook(DOWN), hook(UP)))`. a finite float corner's
  `exact` is a per-descriptor `lru_cache(maxsize=4)` `float_exact(x)`: `s * elementary.exact_pow(|x|, n)`
  (s = -1 iff x < 0 and the int n is odd), or the marker `_NOT_A_DOUBLE` where `exact_pow` declines.
  a hook rounds the cached value with `round_rational` (today's value, built once instead of four
  times a corner), and for the marker is `s * rounded_pow(|x|, n, d * s) + 0.0` (negating swaps DOWN
  and UP). int, Fraction and ±inf corners go to `base.fn` as before
* `_NOT_A_DOUBLE` equals nothing, and its docstring carries the proof: once `exact_pow` declines
  (`|n| max(bitlen(num), bitlen(den)) > 100000`), `x ** n` is neither a double nor a midpoint. x = ±1
  never declines; a power of two 2**e has `|e n| > 50000`, outside the float range; any other double
  has an odd m >= 3 and at most 1075 bits, so `|n| >= 94` and the odd part `m ** n > 2 ** 54` (n > 0)
  or the value is not dyadic (n < 0). so the marker's flags are the exact value's, and the hooks'
  ziv loop meets no breakpoint at the value. `ops.outward`'s "fn is exact" now says pown is built there
* N1 (found by the designer): to nearest, python's `float ** int` converts n to a double, so past
  2**53 the parity was lost (`M(-1.0) ** (2 ** 60 + 1)` = `[1.0]`,
  `M(-1.0000000000000002) ** (2 ** 53 + 1)` = `[7.389056098930649]`, wrong sign) and past about 1.8e308 the
  conversion raised OverflowError, read as an overflow (`M(0.5) ** 10 ** 400` = `[inf]`). for
  `|n| > 2 ** 53` a float corner is now `s * rounded_pow(|x|, n, NEAREST) + 0.0`; up to 2**53 the
  descriptor is the old object. C4 (the critique): a zero base with n > 0 is 0.0 there (B's first
  N1 sent it to python, `M(0.0) ** 10 ** 400` = `[inf]` and `M(0.0, 0.5) ** 10 ** 400` = `[0.0, inf]`, a
  wrong set); n < 0 keeps None and the pole rule
* C3 (the critique, pre-existing in every class): the name `f'pow{n}'` raised python's 4300-digit
  ValueError for `|n| >= 10 ** 4300` (`O(1.5) ** 2 ** 20000`, `M(0.5) ** 2 ** 20000`, the `Dual`
  form). `_power_name(n)` is `pow{n}` while `|n| < 10 ** 18`, else `pow[-]<a {bit_length}-bit int>`

**what the build found**
* a DEGENERATE result is closed whatever attainment says (`applicator.evaluate_box` keeps the point
  of a squeezed piece), so `O(0.5) ** 1074` = `[5e-324]` does not observe the flag: a marker returned
  for a representable power still gives it. the exact-double rows therefore take pieces with width
  too (`O(0.5, 1.0) ** 1074` = `[5e-324, 1.0]`, `** 1075` = `(0.0, 1.0]`, `O(2.0, 4.0) ** 500` closed
  at both ends), and they are what catches "marker whenever |n| > 93" (S3 below); the marker-boundary
  property alone stays green there (S3b)
* `O(0.0, 0.5) ** -(2 ** 61)`, `O(-1.0, 0.5) ** (2 ** 1000 + 1)` and `O(1.0, 1.0000000000000002) ** 10 ** 30`
  hang on the old code too, so they are in the subprocess list, not in-process rows
* the u = 1.0000000000000002 derivative of the M16d pin checked against MPFR: ieee(64) RoundDown/RoundUp
  of `u ** (2 ** 60 - 1)` times 2**60 are 1.7425574576408943e+129 and 1.7425574576408946e+129, the pinned ends
* `pown_rev` with a huge n was never part of the hang (it is a root, not a power):
  `pown_rev(O(0.5, 1), 2 ** 31 - 1)` = `(0.9999999996772282, 1]`, and `pown_rev(O(0.5, 1), 2 ** 61)`,
  `pown_rev(O(0.5, 1), -(2 ** 61 + 1))`, `pown_rev(O(1e300), 2 ** 61 + 1)` and
  `ieee1788.pown_rev(Interval(0.5, 1), 2 ** 31 - 1)` each take about a millisecond, before and after
  the build (the critique, 2026-09-29; re-measured at `4f51e86`, 0.7-1.3 ms under a gate's load,
  2026-09-29)

**tests** (`tests/test_pown_huge.py`, new, unless noted):
* `::test_reproductions_finish`: 23 expressions in one subprocess, `timeout=60` (the pattern of
  `tests/test_backend.py::_run`): the outward class through `**`, `ieee1788.pown`, `Interval ** 1e300`,
  `DecoratedInterval` (COM and TRV), `Dual` (value and derivative), `np.power`, and the pole rows.
  red on the old code: `TimeoutExpired` after 60 s
* `::test_exact_double_boundaries`: 18 rows (A's plus the width rows), green on the old code (they are its values)
* `::test_marker_boundary`: `@given` x and n with `100000 < |n| bits(x) <= 400000`: the piece is the
  exact power rounded both ways, closed iff it is a double, and A's `_is_double` is False wherever it is open
* `::test_identity_with_the_exact_construction`: random 1-3 piece outward sets (float, int, Fraction
  and ±inf ends, random flags), three n an example in ±1..400 and ±1800..4000: `ops.power(a, n, True)`
  equals `apply_unary(ops.outward(ops._exact_power_descriptor(n)), a)` in values, types and flags
* `::test_huge_exponent_against_mpfr`: points against MPFR ieee(64) RoundDown/RoundUp, n an exact-width
  mpfr, closed iff the ternary value is 0; n from 2**31..2**40, 1..2**64, 2**k ± 2, int(10.0**j), both signs
* `::test_pown_matches_pow`: `A ** n` and `A ** O(n)` (1788's pow) agree on positive float points and one-ulp pieces
* `::test_nearest_past_2_53` (12 rows), `::test_nearest_zero_to_a_huge_negative_power_is_empty`,
  `::test_nearest_huge_exponent_against_mpfr` (MPFR RoundToNearest, |n| > 2**53); the nearest rows
  were red on commit 1's library: 10 of 12 rows and the property (the other two are python's own
  value at 2**53 and a pole row, the same before and after)
* `::test_an_exponent_past_4300_digits`, red on commit 2's library with the 4300-digit ValueError
* `tests/test_backend.py::test_huge_pown_is_the_same_on_both_backends`, `::test_huge_pown_reaches_the_backend`
  (a spy: `_gmpy2.rounded_pow` is not called for `O(0.5) ** (2 ** 31 - 1)`, a range shortcut, and is
  called once a direction for `O(1.0000000000000002) ** 10 ** 9`); `::test_power_descriptors_decline`'s docstring amended
* `tests/test_autodiff.py::test_pow_integral_exponent_derivative_is_exact`: u over `O(-1)` and the
  floats 2.0, 1e300, 1e-300 (they finish), -2.0, -1e-300, 1.0000000000000002 (they tell `n - 1` from `n`), times r

**sabotage** (plan §2's rule, 2026-09-29: a throwaway harness under `.scratch/pown-huge/build/`,
`__pycache__` cleared before and after each break, `PYTHONDONTWRITEBYTECODE=1`, one exact
replacement matching once, restored with `copy2` and checked with `filecmp`, a control row on the
intact code first, each targeted run in its own subprocess stopped by its own PID). every control
green; every break below red unless marked:

| id | break | red by |
|---|---|---|
| S1 | the outward descriptor back to `outward(_exact_power_descriptor(n))` (the old code) | `::test_reproductions_finish`, `TimeoutExpired` at 60 s |
| S2 | the marker's `__eq__` True | `::test_marker_boundary` |
| S3 | the marker whenever `abs(n) > 93` instead of where `exact_pow` declines | `::test_exact_double_boundaries[... O(0.5, 1.0) ** 1074]`, a width row |
| S3b | S3, run against `::test_marker_boundary` alone | **green, as expected**: at ±1.0 the result is degenerate and closed anyway, and a power of two past the limit is out of range; the width rows are the guard |
| S4 | the cached exact value without its sign | `::test_identity_with_the_exact_construction` |
| S5 | the cache at module level, keyed on x alone | `::test_identity_with_the_exact_construction` (three n an example) |
| S6 | the rounded path's direction not flipped for a negative result | `::test_huge_exponent_against_mpfr` (ValueError, start after end) |
| S7 | the rounded path's sign dropped | `::test_huge_exponent_against_mpfr` |
| S8 | the parity from `float(n)` | `::test_huge_exponent_against_mpfr` (the `@example` at `2 ** 60 + 1`) |
| S9 | the two hooks swapped | `::test_pown_matches_pow` |
| S10 | the backend's `_gmpy2._int` at 53 bits | `tests/test_backend.py::test_huge_pown_is_the_same_on_both_backends` |
| S11 | autodiff's `n - 1` in float (`int(float(n) - 1)`) | `tests/test_autodiff.py::test_pow_integral_exponent_derivative_is_exact`, 12 of 21: u = `O(-1)`, -2.0, -1e-300, 1.0000000000000002 at each r; **green at 2.0, 1e300, 1e-300, as expected** (saturated: C1) |
| N1 | the nearest threshold at 2**63 | `::test_nearest_past_2_53` (the 2**60 + 1 and 2**53 + 1 rows) and `::test_nearest_huge_exponent_against_mpfr` |
| N2 | a zero base sent to python (the design's first N1) | `::test_nearest_past_2_53`, the three zero rows (C4) |
| N3 | the zero guard removed (zero reaches `rounded_pow`, whose `exact_pow` answers 0 for any n) | the pole row `M(-2.0, 0.0) ** -(10 ** 400 + 1)` and `::test_nearest_zero_to_a_huge_negative_power_is_empty` |
| N4 | the nearest sign dropped | `::test_nearest_past_2_53` (the negative rows) and the MPFR property |
| C3 | the name back to `f'pow{n}'` | `::test_an_exponent_past_4300_digits` (the 4300-digit ValueError) |

**gate** (2026-09-29, loaded shared laptop, `tests/itf1788` alone and the rest in the background,
rc captured without a pipe): commit 1's tree 27795 passed in 50 s + 5580 in 513 s = 33375 (the 33332
before, plus 43 new items); the final tree (commits 1-3) 27795 in 51 s + 5596 in 504 s = 33391. commit 2's
tree alone had a targeted run (the pown, backend, autodiff, class, ops, outward, extreme-float, numpy
and 1788-layer files and `ops.py`'s doctests: 2121 passed in 119 s); it differs from the final tree
only in the descriptor's name. not pushed.

**review** (2026-09-29, three read-only lenses on `a82545d`: soundness, sabotage, spec; then a fixer).
no blocking finding, and no soundness defect: the soundness lens re-derived `_NotADouble`'s proof and
`_ln_bracket`'s saturation bounds, and checked the branch against an independent oracle (binary powering
on (mantissa, exponent) int pairs cut to bitlen(n) + 96 bits, floor and ceil, never the power; itself
0 mismatches against exact rounding and MPFR on 600 + 600 cases): 2500 points on the pure backend and
2540 under gmpy2 (the outward class, `D`, `np.power`, `ieee1788.pown`, `Interval ** n`, nearest past
2**53, `Dual` values and derivatives), 4 x 2500 + 2 x 2500 random one-piece sets (optimal enclosure
and flags, 74k sampled points, identity with the exact construction below 20000), and about 100
hand-read rows, 0 mismatches; that probe went red under in-process breaks of the direction and of the
marker's `__eq__` (53 and 278 reds). the sabotage lens re-ran S1-S11, N1-N4 and C3 red on the final
tree (S1 and S7 with re-anchored texts, the recorded ones having moved), seven new value/flag/hang
breaks red, and the equivalent greens explained (`+ 0.0`, the marker's `__hash__`/`__ne__`, nearest's
`<=` at 2**53). the spec lens found no value regression (79 expressions, same reprs and warnings on
`v2` and the branch), no fast case slower (outward small n about 2x faster; the only slower path,
nearest past 2**53, was wrong before), and every named reproduction finishing in about 0.1 s. the
findings, each reproduced on the branch first by the fixer:

| id | lens | finding | outcome | pin |
|---|---|---|---|---|
| SND-1 | soundness | exact int operands still hang: `O(2) ** 2 ** 60`, `M(2) ** 2 ** 60`, `M(2) ** 1e300`, `O(0.5, 2) ** 2 ** 40` (the 2 an int) | **deferred**: owner question Q-exact below, recorded since the build; not a regression. reproduced: each a 10 s timeout on the branch. row 2 closes only with Q-exact carried forward | — |
| SND-2 | soundness | the cache comment said a value is at most `EXACT_POWER_LIMIT` bits (12.5 KB); `exact_pow` bounds numerator and denominator each, so about 25 KB (`exact_pow(Fraction(1 - 2.0 ** -53), 1851)`: 98103 + 98104 bits) | fixed, comment only | none possible (the code is unchanged; memory stays bounded, 4 values x 64 descriptors) |
| SAB-1 | sabotage | the bounded name was pinned only at `2 ** 20000` (6021 digits): `_power_name`'s threshold could move to `10 ** 6000` unseen | fixed | `::test_an_exponent_past_4300_digits` also at `10 ** 4300`, the smallest n python refuses to write: red under the threshold at `10 ** 6000`, where the old test stayed green |
| SAB-2 | sabotage | nothing pinned "an int corner is the exact descriptor's" in the nearest class past 2**53 | fixed | `::test_nearest_past_2_53` + 4 exact-int rows (`M(1) ** 10 ** 400` = `[1]`, `M(-1) ** ±(2 ** 60 + 1)` = `[-1]`, `M(-1, 0.5) ** (2 ** 60 + 1)` = `[-1, 0.0]`) and a check of the end types: the 4 rows red with int corners sent to `rounded_pow`, the old file green |
| SAB-3 | sabotage | `_NotADouble`'s proof rests on `EXACT_POWER_LIMIT`, unchecked; at 2000 the hooks' ziv loop stalls (no per-test timeout, so a stall, not a red) | fixed, a mechanical refusal | `ops._check_marker_premises`, run at import: `RuntimeError` unless `limit >= 2 * 1075` and `3 ** (limit // 1075 + 1) > 2 ** 54` (at least 36550, the floor of the proof as written; the lens's finer floor of about 2150 is not relied on). `::test_the_marker_proof_premises`; with the limit at 2000 the suite fails at collection in 0.4 s. a limit lowered at run time, after import, still stalls (`O(0.5, 1.0) ** 1074`, 60 s timeout): not guarded |
| SAB-4, F1 | sabotage, spec | the comment "each hung before the fix" is false for `O(-1.0) ** (2 ** 60 + 1)`: `v2` gives `[-1.0]` in 0.08-0.17 s | fixed, comment only | none possible; the row stays as a parity pin |
| SAB-5 | sabotage | "each corner's power is built once" (`float_exact`'s cache) was untested; without it about 5x slower near the limit | fixed | `::test_a_corner_power_is_built_once`: a spy on `elementary.exact_pow` over `O(1.2, 1.3) ** 1879` sees one build a corner; red with the cache at `maxsize=0`, the old file green |
| F2 | spec | `@example((0.5, 100000))  # just under the limit` is on the marker side (2 bits x 100000 = twice the limit); no example sat on the built side | fixed | `@example`s `(0.5, 50000)` (the last built) and `(0.5, 50001)` (the first marker) and `::test_the_marker_boundary_of_one_half` against `exact_pow` directly |
| F3 | spec | `HANDOFF.md` row 2 still says ready; Q-exact and Q-nearest-libm are not under its owner questions | **deferred**: this branch does not edit `HANDOFF.md`; owed when the branch is merged | — |
| F4 | spec | `_exact_power_descriptor`'s docstring read as if python's `float ** int` were correctly rounded for any n | fixed, docstring only: libm's pow, not promised correctly rounded, n rounded to a double past 2**53, and `_power_descriptor` never sends a float there | none possible |

each "red" above is the fixer's own run (2026-09-29, `.scratch/pown-huge/fix/sab.py`: one exact
replacement, `__pycache__` cleared, restored with `copy2` and checked with `filecmp`, a control on the
intact tree first), with the new `tests/test_pown_huge.py` and, beside it, the file as committed at
`a82545d`, which stayed green under every break but the limit's (there the import refuses). gate on
the fixed tree (2026-09-29, loaded shared laptop, rc captured without a pipe): `tests/itf1788` 27795
passed in 55 s, the rest 5607 passed in 567 s (the 5596 before plus 11 new items) = 33402. not pushed.

**left open** (owner questions, not built):
* **Q-exact**, pown of exact int/Fraction operands: `M(2) ** 2 ** 60` and `M(2) ** 1e300` still
  never finish (their exact value does not fit in memory), and exact ends are common inside outward
  intervals: `O(0.5, 2)` and `O.parse('[0.5, 2]')` store the 2 as an int, so `O(0.5, 2) ** 2 ** 40`
  still hangs, and `O(0.5, 3) ** 2 ** 22` returns an interval whose repr raises the 4300-digit
  error. `functions.pow_` already rounds exact operands past `EXACT_POWER_LIMIT`
  (`M(2) ** M(2 ** 60)` = `(MAX, inf)`). options: (a) keep exact and document the limit, (b) raise
  OverflowError up front past a bit budget, (c) past a limit shared with pow_, the tightest open
  float enclosure, (d) (c) in the outward class only with (a) or (b) in the exact class. both
  designers recommend (c); (d) is the minimum that makes `O(0.5, 2) ** 2 ** 40` finish. a separate
  commit whichever is chosen. pown and 1788's pow already answer the same exact point in different
  kinds: `M(3) ** 70000` is the exact 110948-bit int (about 9 ms under load), `M(3) ** M(70000)` is
  `(MAX, inf)` (2026-09-29 at `4f51e86`), so (c) at today's limit also changes cheap exact results
  that finish now (int ends become float ends). the exp/log designer recommended (c) with ONE limit
  shared by pown and pow, raised well above what builds quickly (about 2 ** 20 to 2 ** 24 bits
  instead of 100000), which changes pow's answers between the old and the new limit; (b)'s budget was
  sketched at about 2 ** 26 bits and would need a rule for `DecoratedInterval` and the 1788 layer (an
  error, or `[entire]`/NaI). the 1788 layer is outside the question: `ieee1788.Interval` stores every
  end as a float, so `ieee1788.pown(Interval(2, 3), 2 ** 40)` = `[MAX, inf]` in about a millisecond
* **Q-nearest-libm**: the nearest class keeps python's `float ** int` (libm's `pow`) for
  `|n| <= 2 ** 53`; libm's pow is not promised correctly rounded (3000 of 3000 random near-1 bases
  agreed with `rounded_pow(.., NEAREST)` on this laptop, 2026-09-29). is correctly rounded pown a
  promise of the nearest class? no change made

### fuzz-symmetry: `-` of a mixed point (done 2026-09-29)

**found** by M14's first GitHub fuzz run (run 36507253782, branch `fuzz-run` = `v2` at `7288e81` plus
a temporary trigger, ×10, 2026-09-29: `1 failed, 33331 passed in 3216.59s`):
`tests/test_reverse.py::test_symmetry` red on `op=('pown', -7)`, `c` the point 1/2 as
`(Cut(Fraction(1, 2), BELOW), Cut(0.5, ABOVE))` (`@reproduce_failure('6.168.3',
b'AEEEAUEAQQRBAyg/4AAAAAAAAAEBAA==')`). `pown_rev(c, -7)` was `[1.1040895136738123,
1.1040895136738125)` (the float end to nearest, closed; the exact end directed, open), but
`pown_rev(-c, -7)` was `(-1.1040895136738125, -1.1040895136738123)`, so `rev(-c) != -rev(c)`.
reproduced locally at `7288e81`, before pown-huge.

**diagnosis**: the library, not the test. the class's `-` went through the applicator, which reads
a point by its low cut alone (`applicator._ends`: `lo == hi` gives one end), so `-c` came back as
`[-1/2]` with both cuts exact: the float end lost its type. every other forward op reads a point so
too (`c ** -7`, `sqrt`, `c * 1` of `[1/2, 0.5]` all see an exact 1/2, and of `[0.5, 1/2]` a float
0.5); the reverse ops read each end by its own type (`reverse._end`), so a mixed point's preimage
has one float end and one exact end, and the mirror of that is not the preimage of `-c` as the
class built it. the trig reverse ops met the same point before (`[0, 0.0]`) and
`tests/test_reverse.py::test_trig_rev_symmetry` worked around it by mirroring with `reverse.negate`,
its docstring naming the class's `-` as the cause. a probe of every reverse op on six mixed points,
in both classes (`rev(mixed)` against `rev(the point at its low cut)`), found 45 disagreements, the
two-variable ops included: the per-end reading is the reverse engine's throughout, and sound.

**options weighed**: (a) the reverse ops read a point at its low cut too (a canonicalization in
`reverse._reverse`, tried: the probe's 45 went to 0 and `test_symmetry` passed, but
`test_trig_rev_symmetry`'s own `@example`, the cut mirror of `[0, 0.0]`, would then fail: the cut
mirror and the low-cut reading are not compatible); (b) the test mirrors with `reverse.negate`, as
the trig test does, leaving `-` inconsistent between a point and a piece; (c) **chosen**: `-` is the
cut mirror and `+` the identity, each cut keeping its type (`ops.neg`, `ops.pos`, no longer through
the applicator; the empty operand still warns). this is what `-` already did for a piece with two
ends, makes `-` an involution on the representation, and makes both symmetry tests hold with the
class's `-`. the other forward ops keep the low-cut reading, the reverse ops the per-end one; a mixed
point is sound either way, only its rounding differs.

**pins**: `tests/test_reverse.py::test_symmetry`'s `@example(op=('pown', -7),
cut_tuples_c=one(Fraction(1, 2), 0.5))`; `tests/test_ops_properties.py::test_neg_and_pos_keep_each_cuts_type`
(`-` is the typed cut mirror and an involution, `+` the identity; two mixed-point `@example`s). both
red with `intervals/ops.py` as at `548ac78` (`__pycache__` cleared, `PYTHONDONTWRITEBYTECODE=1`),
green with the fix. `test_trig_rev_symmetry`'s docstring updated (its `negate` stays).

**gate**: green, 2026-09-29 on the fixed tree (shared laptop, rc captured without a pipe): `tests/itf1788` 27795 passed in 57 s, the rest 5608 passed in 591 s (5607 before plus the new neg/pos test) = 33403; `test_trig_rev_symmetry`'s docstring was edited during the second call (text only). the fuzz rerun on `fuzz-run` (run 36540588320, 2026-09-29) passed `test_symmetry` (on its `@example`: the saved cache held no examples) and found fuzz-floordiv-overflow below
(`HANDOFF.md` row M14-run).

### fuzz-floordiv-overflow: a test-oracle overflow (done 2026-09-29)

**found** by the fuzz rerun (run 36540588320, branch `fuzz-run` = `master` at `8a4cc2f` plus the
trigger, ×10, the first run's `.hypothesis` cache restored but holding no examples, as the first
run's did not: the profile had no database on CI, `v2-plan.md` "2026-09-29 revision: fuzz on push";
`test_symmetry` passed on its `@example`, not a replay; 2026-09-29: `1 failed, 33402 passed in
3364.07s`, the job 56 min 25 s; `test_symmetry` passed): `tests/test_modulo.py::test_floordiv_sound_float`
raised `OverflowError: int too large to convert to float` on `a = [1/2, 1)`, `b = (-inf,
2.2250738585e-313]` (`@reproduce_failure('6.168.3',
b'AXicY3RkcGQBYlZGBgZGIM3gyKzBwMDAVRN1pIiBkcGRXMgIAABYC3Q=')`).

**diagnosis**: the test, not the library. `floordiv(a, b)` is `[-inf, -1.0] ∪ {inf}`: over the
divisor's positive part the exact floor is a finite int past 2.2e312 (1038 bits), whose nearest
double is inf, as python's own `0.5 // 2.2250738585e-313` is. the oracle checked the rounded value
with `float(exact)`, which raises past MAX instead of rounding to ±inf (`tests/oracles.py::_once`
already maps the overflow to ±inf).

**fix and pin**: `tests/test_modulo.py::_nearest` (float, past MAX ±inf), used by the property and
by `_python_floordiv`'s mixed path; the example pinned as an `@example` on `test_floordiv_sound_float`,
red with the old `float(exact)` (`OverflowError`), green with the fix; `tests/test_modulo.py` 1003
passed (2026-09-29). gate green on the fixed tree: `tests/itf1788` 27795 passed in 55 s, the rest
5608 in 559 s = 33403 (an `@example` adds no item), 2026-09-29. not pushed.

### fuzz-rev-inf: a squeezed reverse result lost to `x` (done 2026-09-30)

**found** by the first push-triggered fuzz run (run 36580954134 at `97d9824`, ×10, 2026-09-29: `1 failed,
33405 passed in 2945.29s`; the babysitter's `tools/ci_watch.sh`, the first run whose saved `.hypothesis`
replays the failure unchanged): `tests/test_reverse.py::test_float_operands` on `op=('pown', -1)`,
`c = (-2.225073858507203e-309, 0)`, `x = (-inf, -2)`, its last assert ("nothing of the exact result
vanishes to nearest").

**diagnosis**: the library's nearest class, by its order of operations. the exact result `(-inf,
-4.49e308)` lies wholly past -MAX; outward gave `(-inf, -MAX)`; to nearest the preimage squeezes to
`[-inf]` (both ends round to -inf, `branch_preimage`'s squeeze), and `_reverse` then intersected with an
`x` open at -inf. the babysitter read the closed `-inf` as unsound and proposed not squeezing; checked by
the session, that empties the result for every `x`, so it was not the fix. the same loss at a finite
double: `sqr_rev([2, 2.0000000000000004], (1.4142135623730951, 2])` was `{}`.

**decision**: D26 (a), the owner, 2026-09-30, after weighing 1788 (which intersects with `x` first and
then encloses, never producing an infinite point) against python's float rounding.

**fix and pins**: `reverse._keep_squeezed`, run by `_reverse` for the nearest class unless `x` is the
default `[-inf, inf]`: a piece of the outward preimage inside `x` that the rounded result does not meet
holds a part of the exact answer that rounds only onto that piece's ends, and each end in the closure of
the rounded preimage (before `x` cuts it; for the periodic ops, computed over a one-double neighbourhood
of `x`) is kept as a point. a first version took every end outside `x` and missed the second way (an end
of `x` it holds, at an open rounded end: `mul_rev(10, (1, 2), [0.1])` stayed `{}`), found by the rewritten
mul_rev test. the tests: `test_float_operands` gains the fuzz case and a finite one (`sqr_rev([2,
2.0000000000000004], (1.4142135623730951, 2])`); the float tests of mul_rev and the trig ops, which pinned
"x only intersects, after the rounding" (`== nearest_all & x`), now check D26 (`meets_x_as_d26`: the old
answer kept, anything more a closed point of `x`'s closure within one double of the exact result, and no
piece of the outward result in `x` without a point of the result, `no_piece_vanishes`); the trig test's
`sin` sliver example (`x = (-inf, -5.6e-24)`) and mul_rev's `[0.1]` example now expect the point. each of
the four red with `_keep_squeezed` a no-op and green with it (2026-09-30); the per-piece check was added
after the `sin` example stayed green when sabotaged (its other pieces kept the result non-empty).
`test_exactly_the_points_with_f_in_c` allows exactly the one way out of `x` (a closed point on `x`'s
closure) and still refuses a stray point (checked with a planted `[5.0]`).

### M14-breadth: fuzz where it was thin (done 2026-10-02)

**the owner**, 2026-10-02: "lets do M14-breadth finished then run fuzzing" (the push's prepush was paused
for it). the spec is M14's bullet "**breadth where fuzz is thin**". six builders, one per file, each in its
own worktree off `fa59944` (sabotage edits library files: T1's hazard), each writing properties from the
design, measuring them under the default profile and the fuzz profile at x10, and sabotaging each; a red
property on the intact library was reported, not fixed. the session verified every library finding
first-hand, fixed and pinned it, and removed the builders' narrowings.

| stream | file | properties | sabotage | default / fuzz x10 (2026-10-02, loaded laptop) |
|---|---|---|---|---|
| extreme-functions | `tests/test_extreme_floats_functions.py` (new) | every function of `functions.py::NAMES`, `log(base)`, `rootn`, `atan2`, `hypot`, `**` on extreme operands against arb: values in the result, ends sharp where monotone | 26 breaks, all red | 8.0 s / 101 s |
| extreme-ops | `tests/test_extreme_floats.py` | both classes end to end (operators, reflected, scalars) for `+ - * /`, reciprocal, pow, `%`, `//`, divmod, fma, neg, abs, minimum, maximum: sound, tightest, nearest is the exact result rounded | 23, all red | +12 s / 106 s |
| outward | `tests/test_outward.py` | exact ⊆ outward ⊆ the tightest cover (16 ops), nearest inside outward's closure, the class closed, pickle/repr, `==` and hash across classes | 23, all red | +16 s / 234 s |
| steps | `tests/test_steps.py` | every double against python, `ndigits` against decimal, float and mixed operands, the hull past the cap, isotone, unions, idempotent | 15, all red | +12 s / 268 s |
| fmt | `tests/test_fmt.py` | round trip over extremes with types, repr in both classes, every spelling, `parse_value`, any text parses or raises ValueError | 15, all red | +10.5 s / 127 s |
| applicator | `tests/test_applicator.py` | split is a partition, `evaluate_box` against the oracle, the hook sees every finite float corner and nothing else, number types, warnings once, results canonical | 20, all red | +20 s / 239 s |

**library bugs found and fixed** (each pin red on the old code, green on the fix, 2026-10-02):

* **outward floor, ceil, trunc rounded to nearest** (soundness): `MultiInterval.floor/ceil/trunc` never passed
  `outward` to the step engine, so `OutwardMultiInterval(2.0 ** 53, 2.0 ** 53 + 2).ceil()` was `{ [2 ** 53] ,
  [2 ** 53 + 2] }`, missing 2 ** 53 + 1 (round was right). `steps.floor/ceil/trunc` and `modulo.floor` take
  `outward`. pin `tests/test_steps.py::test_outward_lists_what_no_double_holds`
* **the cap counted a shared value twice**: `ceil({ [0, 1/2] , [7/10, 999] })` (1000 values) and trunc's 0 on
  both sides of its split hulled at 999. `steps.step` counts distinct values (the functions are
  non-decreasing, the pieces in order). pin `::test_the_cap_counts_distinct_values`
* **nearest pown with n < 0 rounded twice**: `M(5.155830884225402) ** -3` was `1 / x ** 3`, an ulp below the
  nearest double; it is python's `x ** n` now, as the design says (`ops._exact_power_descriptor`). pin
  `tests/test_outward.py::test_a_negative_power_rounds_once`
* **fmt**: a zero denominator raised `ZeroDivisionError` (`parse('[1/0]')`), and a separator after a leading
  empty item was refused (`parse('[] , [1]')`). pins: `@example`s of
  `tests/test_fmt.py::test_any_text_parses_or_raises_value_error`; `::test_spellings` red on the old parser

**a test-oracle bug found**: `tests/test_pow_rev.py::test_pow_rev_float_operands` still read "x after the
rounding" after D26; a randomized local run drew `pow_rev1([-2.86e-115, 0.0], (-inf, 0.5), (-inf, inf))`,
`[inf]`, which D26 keeps. it now checks `meets_x_as_d26`, with the example pinned (red with
`reverse._keep_squeezed` a no-op).

**the first fuzz x10 run of the merged tree** (`fuzz-x10:rest` at `6f9fd09`, 2026-10-02: `4 failed, 5910 passed`
in 6559 s; the prepush refused the push) found four more, each fixed and pinned red-on-old:

* **library**: a point whose two cuts differ in type (`[1.0, 1]`, a float and an int end) formatted as `[1.0]`,
  so `repr` lost a type and the round trip failed (`parse('[0E0-0]')`; an outward `(2.0, 2)` through `repr`).
  `fmt.format_piece` writes such a point with both numbers. pins: `@example`s of
  `tests/test_fmt.py::test_any_text_parses_or_raises_value_error` and
  `tests/test_outward.py::test_pickle_copy_and_repr_round_trip`
* **test oracle, older than M14-breadth**: `tests/oracles.py` computed a float `x ** -n` as `1 / x ** n`,
  the double rounding just removed from the library, so `test_ops_properties.py::test_sound_float_identity_rounding[pow]`
  went red on `0.6 ** -2` (nearest double 2.777777777777778, the oracle's 2.7777777777777777). the oracle now
  takes python's `x ** n`. pin `tests/test_oracles.py::test_a_float_negative_power_is_python_s_value`
* **test oracle, new**: `test_extreme_floats.py::test_nearest_class_rounds_the_exact_result_to_nearest` zipped
  hull ends in order, but to nearest rounding can carry one end past the other (`M((1/3, 1.0)) + 10 ** 20`: the
  float corner rounds to 1e20, below the exact corner 10 ** 20 + 1/3), and then a hull end can be another
  corner's value (`(-TINY, 0) - 1/3` ends at the exact -1/3: found by the next x10 run, after a first fix that
  only allowed the two ends to swap). each end is now checked to lie between the exact end and its nearest
  double (every corner lies beyond the exact end, rounding is monotone), exactly the nearest double where
  mod, `//` and fma round once; both cases pinned by `@example`s, and `_pick` always outward still turns it red
  on add, sub, mul, div and reciprocal (as in the builder's table)

**the third x10 run** (`fuzz-x10:rest` on `s:c967ae7f2f46`, 2026-10-02: `2 failed, 5913 passed` in 6581 s), two
more test oracles, both new: `tests/test_steps.py::_nearest` caught the overflow of a value past the doubles and
then called `math.copysign(INF, q)`, which converts q and overflows again (`round(1.7976931348623155e308, -293)`;
pinned by an `@example`); and the applicator's hook property expected the hook at a float corner that is a split
point (`abs` of `[0, 0.0]`), which the applicator reads as the split point, an int, so nothing rounds: the type
quirk below, no wrong value. that property now requires every other float corner (still red when the hook skips
mixed corners).

**left open, not fixed** (rows in `HANDOFF.md`): an int past python's 4300-digit `str()` limit cannot be formatted
(`repr(MultiInterval(10 ** 4300))` raises; with Q17); `parse(' ' * 30000 + 'x')` takes 37 s (the tokenizer's
regex is quadratic on leading whitespace, the answer right); number-type quirks with no wrong value (a 0 end
exact among float operands, `abs(M(-1.0, 1.0))` is `[0, 1.0]`; trunc's non-negative side ints, `trunc([-2.5,
3])`; a one-point domain clip takes its low cut's type, `M(-1.0000000000000002, -1.0).acos()` open);
`parse_value('+-5')` is 5 and `'1 2'` is 12; `tests/test_extreme_floats.py::_float_samples` raises
`OverflowError` on an exact piece wider than the doubles.

**the parse fixed (2026-10-04, `6450502`)**: the cause was `fmt.py::_TOKEN`'s leading `\s*` (and `_NUMBER`'s inner
`[+-]?\s*`): with no token after a white-space run, the engine gave the run back one space at a time and at each
retried `_NUMBER`, which re-read the rest of the run, O(n ** 2). every `\s*` there is now possessive `\s*+`, which
cannot change a match (nothing after any of those runs begins with white space). measured before, per doubling of n
from 1000 to 16000: 0.017, 0.068, 0.28, 1.10, 4.45 s for `' ' * n + 'x'` and its `'+x'` variant, the only
superlinear shapes of the ones tried (trailing, inner and between-token white space, long or malformed numbers, many
separators, unbalanced brackets, `parse_value`, `literals.parse_literal`, all linear already; `_LITERAL` was already
possessive since M13g); after, 0.0002 s at 30000 and 0.001 s at 200000. an old-vs-new differential on 250019
inputs (random over the grammar's alphabet, white-space-sprinkled valid texts, `'+-5'`, `'1 2'`, `'- 5'`): 0
differences in tokens, values or exception types and messages. pinned by
`tests/test_fmt.py::test_white_space_runs_parse_in_linear_time` (8 shapes at n = 100000, < 10 s each); on the
old `fmt.py` 2 of them failed (186.8 s and 229.9 s), the other six were never quadratic. gate on the branch:
27795 + 6242 passed (2026-10-04)

### fuzz-steps-isotone: outward isotonicity across number types (done 2026-10-03)

**found** by a local x50 fuzz run (2026-10-03, replayed by the session): `tests/test_steps.py::test_isotone[round]`
and `[round_ties_away]`, in the outward class. `round_ties_away((0.0, 1/10), 1)` = `{ [0.0] ,
(0.09999999999999999, 0.1) }` is not inside `round_ties_away((-inf, 1/10), 1)` = `(-inf, 1/10]`; and `round((-1.0,
-1/20), 1)`, which lists `(-0.1, -0.09999999999999999)`, is not inside `round((-inf, -1/20), 1)` = `(-inf, -1/10]`.

**diagnosis**: the test, not the library. the session's lead was "the hull path keeps an exact end while the listed
path rounds"; checked, `steps.step` rounds a hull outward too, for a float piece. B's piece has no finite float end
(±inf is exact, `rounding.is_float`), so the outward class computes f(B) exactly, as its contract says
(`tests/test_outward.py`: "on exact operands it is the same as MultiInterval"), while A's piece has a float end, so
its values that are not doubles are listed as the open gap around them. each is within the contract; together
`A ⊆ B` does not give `f(A) ⊆ f(B)`, and no rule by number type can make it: an exact listed `1/10` in f(B) cannot
hold any rounding of A's `1/10`. hulls play no part: `round([0.0, 0.25], 1)` against `round([-1, 3/10], 1)` and
`ceil([2.0 ** 53, 2.0 ** 53 + 2])` against `ceil([2 ** 53 - 1/2, 2 ** 53 + 3])` fail the same way with everything
listed, and outward `+` too: `O([1.0, 2]) + 1/3` = `(1.3333333333333333, 7/3]` is not inside `O([1 - 10 ** -30, 2])
+ 1/3`. every other isotonicity property in `tests/` runs on exact operands. the lead's fix (round an outward hull
whatever the piece) turns `tests/test_outward.py::test_exact_operands_give_the_same_set[round]`,
`[round_ties_away]` and `tests/test_steps.py::test_hull_past_the_cap[round]`, `[round_ties_away]` red.

**fix and pins**: `test_isotone`'s check (`_assert_isotone`) keeps `f(A) ⊆ f(B)` in the outward class where each
float piece of A lies in a float piece of B (outward rounding is monotone, and an exact value lies in its own open
gap), and elsewhere checks `f(A) ⊆` the tightest double-ended cover of f(B) (`_cover`), which the contract does
give. pins: `tests/test_steps.py::test_isotone_float_piece_in_an_exact_one`, the two fuzz rows and the two listed
ones (an `@example` cannot feed `st.data()`), each also asserting outward f(B) equals MultiInterval's. sabotage
(2026-10-03): the old check, all four rows red; the lead's fix, rows 1-2 red (and the four tests above); outward
listing rounded to nearest, `test_isotone[round_ties_away]` and rows 1 and 3 red. the library is unchanged.

### fuzz-mixed-points: two oracles that assumed a float stays a float (done 2026-10-03)

**found** by the prepush's x10 fuzz at `e5cb396` (2026-10-03, `fuzz-x10:rest`, replayed from the local database):
`tests/test_applicator.py::test_rounding_hook_sees_every_finite_float_corner_and_nothing_else[mul]` on `(-inf, -1.0)
* [0, 0.0]` and `[abs]` on `[2, 2.0]` and `[1, 1.0]`; `tests/test_ops_properties.py::test_sound_float_identity_rounding
[sub]` on `(0, 5.0533628082320924e-138) - (-inf, 1/5]`. the same run's other 27 failures and 24 errors were the
laptop, not the code: every one a child process or `git` that exited `3221225794` (`0xC0000142`, a DLL failing to
initialise, under another session's concurrent hypothesis run), and the ledger read the run as MOVED.

**diagnosis**: the tests, not the library. (1) a one-point piece whose cuts differ in type is evaluated at its exact
low cut: `abs` of `[2, 2.0]` calls no hook and gives `[2]`, `(-inf, -1.0) * [2, 2.0]` rounds the corner `(-1.0, 2)`
and gives `(-inf, -2.0)`, in both classes; the values are right, the type quirk of HANDOFF row m14b-open. the hook
oracle allowed it only at a split point (`abs` of `[0, 0.0]`, M14-breadth). (2) python computes `2.5266814041160462e-138
- Fraction(1, 5)` as `float(1/5)` then a float difference, -0.2, below the exact open end -1/5 that the exact class
computes from the corner `0 - 1/5`; no exact end can stop a rounded value, so "python's float value is in the result"
holds only next to float corners.

**fix and pins**: the hook oracle (`check_rounding_hook`) lets a float corner coordinate be read as an equal exact
twin (`exact_twins`: the exact end of a mixed one-point piece, or a split point), the corner then seen with the twin
or, if no float is left, not at all; the identity-rounding oracle (`check_float_identity_rounding`) allows a python
float outside the result only where the pair's exact value is inside. pins:
`tests/test_applicator.py::test_rounding_hook_reads_a_mixed_point_as_its_exact_twin` (the three fuzz rows, the earlier
`[0, 0.0]`, and `(-inf, -1.0) * [2, 2.0]` off the split point) and
`tests/test_ops_properties.py::test_float_identity_rounding_past_an_exact_end` (asserting the old oracle's claim
fails on it). sabotage (2026-10-03, in process): the old hook oracle, 4 of the 5 rows red (`[0, 0.0]` was its own
exemption); the applicator skipping the hook on every negative float, the new hook oracle red on `mul`, `abs` and
the mixed-point `mul` (`add` takes the monotone fast path, outside that clause); the `sub` result moved to `(-1/10,
inf]` or emptied, the new identity oracle red. the library is unchanged.

### fuzz-rootn-crossed: a float end rounded to nearest past an exact one (done 2026-10-03)

**found** by fuzz run 37098878528 on GitHub at `8e33d7e` (2026-10-03, x10; CI run 37098878518 green), a babysitter on
`tools/ci_watch.sh`, reproduced locally from the artifact's database and by the seed:
`tests/test_extreme_floats_functions.py::test_every_value_is_in_the_result[rootn]`, `rng=Random(94807)`: `rootn(., 5)`
of an operand with the piece `(10 ** -30, 1.0000000000000003e-30]` raised `ValueError: interval start Fraction(1,
1000000) is after end 1e-06` in the exact class (`MultiInterval`), from `kernel.piece`.

**diagnosis**: the library's. `functions._Function.end` types each end on its own: the exact end gives `rootn(10 **
-30, 5)` = 1/10 ** 6 exactly, the float end rounds `(1.0000000000000003e-30) ** (1/5)` (about 1/10 ** 6 + 6e-23) to
nearest, onto the double `1e-06`, which is below 1/10 ** 6. `_settled` kept a piece squeezed to one point but not one
whose ends crossed. `reverse.branch_preimage` has the same per-end rule (`reverse._end`), so `pown_rev((10 ** -30,
1.0000000000000003e-30], 5)` raised the same. outward rounding never crosses (each end moves away from the other).
a probe over every function and `rootn`/`log` degree and base, at 15 exact points with float partners one to five
doubles away, found only these (2026-10-03).

**fix and pins**: a crossed piece is the piece between the two values, each keeping its flag, as the applicator's
ends are the least and greatest corner values: `[1e-06, 1/1000000)`, whose closure holds the nearest double of every
value of the image (`functions._settled`, `reverse.branch_preimage`). pins:
`tests/test_functions.py::test_rootn_ends_crossed_by_rounding`, `tests/test_reverse.py::test_pown_rev_ends_crossed_by_rounding`
(each asserting the outward result unchanged), and `@example(rng=random.Random(94807))` on the test that found it.
sabotage (2026-10-03): `functions.py` reverted, the rootn pin and the example red; `reverse.py` reverted, the pown_rev
pin red. the outward class is unchanged; what the exact class gives for a crossed piece is the session's choice, by
the applicator's rule (HANDOFF Q20).

### owner-answers: the build of the owner's 2026-10-03 answers (done 2026-10-04)

the owner accepted every recommendation of `references/owner-questions-2026-10-03/` (D27-D29; `v2-plan.md`
"2026-10-03 revision: owner answers"). four streams, each in its own worktree off `09435ca`, merged into
`master` (`87c9319`, `9896284` with a README conflict resolved by keeping both, `87e6ea3`); each stream's
record (what changed by `file::symbol`, its pins and sabotage tables, its runs) is
`references/owner-questions-2026-10-03/streams/<stream>.md`.

* **pown** (D28; `streams/pown.md`): `elementary.EXACT_RESULT_LIMIT = 2 ** 22` bits for exact operands, shared by
  pown, `functions.pow_` and exp2/exp10 (exp2/exp10 now measured in result bits); `EXACT_POWER_LIMIT` stays
  the float-corner threshold. past the limit an exact corner is its tightest open float enclosure in both
  classes, with `errors.PowerLimitWarning` (ignored by default, exported). pown to nearest is correctly
  rounded for every n (`ops._power_descriptor`: the exact power rounded once, else `rounded_pow`); libm's
  `float ** int` and the `2 ** 53` seam are gone. `ops._NotADouble`'s proof re-derived for any rational
  corner before the build (`streams/pown.md` step 1). ints past python's 4300-digit limit print in hex and
  `parse` reads `0x` (`fmt`, `cuts`). `tests/coremath/pown.tsv`: 2857 rows from pow.wc's integral exponents;
  `tools/coremath.py` reads a cache elsewhere through `INTERVALS_COREMATH_CACHE`. **changed at the merge**:
  the stream built "rounded to nearest" for `MultiInterval` past the limit, from a misread summary; the
  report's (c) is the enclosure in both classes (one class only was (d), rejected), so the nearest descriptor
  got rounding hooks (a float corner to nearest, an exact corner outward). sabotage: the stream's 44 of 46
  new pin ids red on `09435ca` (6 by hanging), its 12 breaks each red; the merge's change: 4 of
  `tests/test_pown_huge.py`'s tests red on the stream's `ops.py`, green after (2026-10-04). **left open**: an
  exact corner of about 2M bits within about 2 ** -(its size) of a breakpoint now runs ziv past 120 s where
  the old code built the power in milliseconds (`O(3 + 2 ** -2100000) ** 2`); pow had the same at 60k-bit
  operands before and has it at 2M bits now (`streams/pown.md` step 6, two follow-ups)
* **numpy, methods, shifts** (D23 as answered; `streams/numpy.md`): `multi_interval.py::_subclass_decides` on
  the eleven methods taking another set (union, intersection, difference, symmetric_difference, minimum,
  maximum, fma, cancel_minus, cancel_plus, hypot, atan2), so a mixed call is outward as the operators are;
  `DecoratedInterval`'s `& | ^` had the same defect (they called `MultiInterval.__and__`), fixed;
  `numpy_compat._subclass_first` removed (a ufunc is the method). `==`/`!=` against an ndarray give a bool
  array; `fmin`/`fmax` are `minimum`/`maximum`; numpy in `[test]`, the README numpy section doctests, the
  trailing `numpy` gone from the workflows. `<<`/`>>` (Q6-shift) on `MultiInterval`, `DecoratedInterval`,
  `Dual`, with `left_shift`/`right_shift` in `_OPERATORS` (dropped 2026-10-04, owner: `0513109` reverted,
  `tests/test_multi_interval.py::test_no_shifts`). every new or flipped pin red on `09435ca`; nine
  breaks of the new code each red
* **backend and CI** (D24 as answered; `streams/backend.md`): `[fast]` pinned `gmpy2>=2.3,<3`
  (`tests/test_backend.py::test_the_fast_extra_installs_what_auto_takes`); CI job `gate-gmpy2` (python 3.13,
  `INTERVALS_BACKEND=gmpy2`, asserts `backend.name() == 'gmpy2'` first: a forced but missing gmpy2 fails
  collection for most files but not all, `tests/test_cuts.py` passes, so the assert is the guard, pinned);
  ledger phase `gate:gmpy2`, keyed by src, never part of the commit verdict, required for a push only when a
  file of `tools/gate.py::BACKEND_FILES` changed since the base (`tools/prepush.sh` runs it; `CLAUDE.md`
  push). the whole suite on gmpy2 locally: 33833 passed in 777 s (2026-10-04, on the stream's tree). 13
  sabotage rows red. README: `intervals.backend.name()` for bug reports; `tools/backend_speed.py` cites §2
  M16e
* **the rest** (D21, D22, D20, D29; `streams/small.md`): the 1788 layer's numbers of the empty set are `nan`
  and its four reductions return `nan` for a nan operand, `inf + -inf` and `0 * inf` (the library keeps D9);
  the third pass's NaN-reading clause gone (`tests/itf1788/test_ieee1788.py`); Q9/Q10 clauses paid; the stale
  "domain-clipped functions" category removed and `test_divergence_rows` now wants a row per category.
  every public cut-tuple relation asserts normalized operands (`allen_matrix` takes pieces in any order and
  checks each through `allen`). solver: `newton-width` fixed (it was live: `newton(x ** 2 - 9 * 10 ** 800,
  [10 ** 400, 10 ** 401])` raised OverflowError at the first step); the split inside (0, 1] in `newton` only
  (in `solve` it took the circle on `[-1e300, 1e300] ** 2` from 103 calls to 3433), and not for a piece
  already within `tol`; `x ** 2 - 1e-40` on `[-1, 1]` proves both zeros in 28 calls (was 132, unproved).
  `OutwardMultiInterval.rounded()` (`rounding.float_cuts`, outward) and Q19's rule documented; Q20's
  sentence in `v2-plan.md` "flags at rounded ends". every pin red on the old code or a targeted break

### run ledger: what has run on this code (done 2026-10-01)

**why**: the owner, 2026-10-01: "i need some machinery to know whats run and not on the current code ...
so we know if we need to rerun before a push". until then the answer was memory and `.scratch/` logs
named by commit; `tools/prepush.sh` re-ran the 83-minute fuzz on every non-docs push, even when the same
code had just passed it.

**built**, adapted from the sibling repo graph-reachability-zanzibar-index (`scripts/gate_status.py` and
the ledger block of `formal/verify.sh`, 2026-08-16..09-10): `tools/gate.py`, recorder and reader in one
file. `run <phase>` (`gate:itf`, `gate:rest`, `docs`, `fuzz-x<N>:itf`, `fuzz-x<N>:rest`) runs pytest
with the phase's environment, keeps the output in `.gate-runs/` and appends a row (verdict, counts,
versions, and two content ids taken at the start and checked again at the end). `status` reports each
phase on the current code and the commit and push verdicts; `plan` is what `tools/prepush.sh` reads.
taken from zanzibar: the content address over tracked and untracked-not-ignored bytes (it survives
`git commit`; a committed deletion hashes as the pending one, their 2026-09-05b hole); refusing to guess
an id when git fails; verdicts keyed by (phase, id); per-phase scopes; a killed run is a log without a
row. two scopes here: `src` (all but `*.md` and `references/`, fuzz.yml's `paths-ignore`, pinned equal)
keys a fuzz verdict, `code` (src plus every `README.md`, the doctest glob, pinned) keys a gate or docs
verdict, so the owner's docs-only rule (2026-09-30) became a property of the ids: a README edit after a
fuzz run needs only the docs phase. not taken: zanzibar's run lock (its defect was two runs sharing one
fixed log path; here each run has its own log and its own row); count floors (recorded, not enforced).
added here: a run whose ids moved while it ran is MOVED and counts for nothing (agents edit this tree
in parallel); `INTERVALS_BACKEND` is removed for a run; a fuzz run below fuzz.yml's x10 does not clear
a push (before, `FUZZ_MULTIPLIER=1 tools/prepush.sh` exited 0).

**the scope survey** (2026-10-01; an exclusion is a fail-open surface): the suite reads `pyproject.toml`,
`tests/itf1788`'s data and `archive/v1` (pythonpath), and collects exactly `README.md` and
`tests/itf1788/README.md` as doctests; nothing under `tests/`, `intervals/` or `archive/` names another
markdown file or `references/`. `tests/test_gate_ledger.py::test_no_source_names_a_prose_path` re-runs
that survey on every gate (red on a probe file naming `HANDOFF.md`, 2026-10-01).

**sabotage** (each break alone in `tools/gate.py`, `tests/test_gate_ledger.py` run, 2026-10-01; control
45 passed): README classed as prose (9 red); no MOVED (1); keyed by phase only (2); any multiplier clears
a push (1); a deletion hashed as absent (1); rc 0 alone passes (3); the backend variable kept (1); a git
failure ignored (1); a README change needs nothing (2); untracked files not hashed (2); the multiplier
drifting from fuzz.yml's (1); the log header parsed as output (1); prose counted as dirty (1). the first
round had "backend kept" green, and that was a bug in the tool, not a gap in the test: the summary parser
read the log's header, which echoes the command, so a command whose text held "3 passed in 0.1s" was
recorded PASSED. fixed (only the run's own output is parsed) and pinned
(`::test_the_command_line_is_not_the_verdict`).

## 3. order and parallelism

M1 → M2 → M3 → M4 → M5 → M6 → {M7a → M7b, M9} → M10, all done by 2026-09-25; M8 deferred; M11 is
the backlog, and M12 built its (b) items the same day. M13 (full itf1788) and M14 (fuzzing): M13a
first, then M13b to M13h in any order, each with its M14 properties; each sub-task's record says
whether it is done (the M13 and M14 headings list them); what is open, and in what order, is in
`HANDOFF.md`. M4 depends on M3 (the class's
`parse`, `__str__` and `__repr__` come from `fmt`); M7a and M9 are independent after M6. total ≈ 12
working days (the per-milestone sum without M8) plus the M7b session. the first internally usable
point is after M5 (set algebra, formatting, comparisons); arithmetic lands at M6; release needs M7b.
M15 (H3's first part, 2026-09-27) came after M13, on the owner's call to take H3 first.
M16 (H3's second part, 2026-09-28) came after M15: its five streams, M16a to M16e, are
independent of each other and were built in parallel worktrees, then merged on `h3-merge`.

## 4. v1 → v2 surface map (for the M10 README and for not forgetting anything)

"gone" below means gone from v2's API; the v1 code itself is archived at M10, not deleted.

| v1 | v2 |
|---|---|
| `merge(*args, n_overlaps=)` classmethod, parses strings | `union(*)`, `intersection(*)`; `MultiInterval.parse(str)`, `from_pieces`. the k-overlap mode and the mixed-input parsing gone (owner 2026-10-04: artifacts of v1's code, not features) |
| `update/intersection_update/...`, `add/discard/pop/remove/clear` | gone (immutable); `Builder` for incremental construction |
| `merge_adjacent(distance=)`, `expand(d, inplace=)` | `expand(d)` pure; no distance rule anywhere (cuts make it exact) |
| `cardinality -> (half_rays, length, half_points)` | `size -> Size(rays, length, points)` |
| `overlapping(or_adjacent=)`, `overlaps` | `relations.overlaps/adjoins`, `A & B` for the overlap itself; v1's `overlapping` returned the whole pieces of self meeting other, now `[p for p in A if p.overlaps(B)]` |
| no set operators (`\|` between two MultiIntervals raises); `~` = complement; `-` = subtraction | `\| & ^ ~` set algebra; `difference()` named; `-` still subtraction |
| `__eq__` coerces scalars, unhashable | structural, no coercion, hashable |
| `A[0:5]` reads a 0 bound as missing (`item.start or -inf`) | a 0 bound is a bound |
| `∅ in ∅` is False; `repr` raises | True; `repr` is `MultiInterval.parse('...')` and evals back |
| `contiguous_intervals`, `infimum`/`supremum`(`_is_closed`) | `pieces` / iteration, `inf`/`sup`(`_closed`) |
| `__lt__` etc. comparing endpoint lists | pointwise `TruthSet`; `sort_key` for the old structural order |
| `reciprocal` → whole line at zero | split at zero, direction from the sign of the piece |
| `__floordiv__` floors endpoints | `floor ∘ div` (exact quotient), enumerating; the limit at an infinite divisor |
| `apply_monotonic_{unary,binary}_function` | `applicator.apply_{unary,binary}(descriptor, ...)` |
| `INFINITY_IS_NOT_FINITE`, `CONSISTENCY_CHECK` | deleted; `if __debug__` check in the class |
| `interval.py` (`Interval`, `MultipleInterval`) | archived in `archive/v1/`; `tests/oracles.py` does its job |
| `time_interval.py` (`DateTimeInterval`, `TimeDeltaInterval`) | archived in `archive/v1/`; comes back at M8 |
| `exp()`, `log(base)` | `exp()`, `log(base=None)`, and the rest of `functions.py` (M12) |
| `__round__`, `__trunc__`, `__floor__`, `__ceil__` (endpoint-wise) | the same dunders, returning the set of values attained (`steps.py`, M12) |
| `**` with an interval exponent on a positive base; `pow(A, n, m)` on integers | an integral number exponent is pown; any other real or interval exponent is 1788 `pow`, and `b ** A` works (M13d, D11); `pow(A, n, m)` dropped (D11) |
| `<<`, `>>` | gone, a TypeError (owner 2026-10-04: built 2026-10-03 as exact scaling, then dropped, no use case); `* 2 ** n`, `// 2 ** n` |
| `random_multi_interval` | gone (owner 2026-10-04: a v1 test helper); the tests use hypothesis strategies |
| public `apply()` | not now (owner 2026-10-04); `applicator` and `OpDescriptor` are not exported |
