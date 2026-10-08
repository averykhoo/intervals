# decisions

the one decisions log of `multiinterval` (since 2026-10-08): what was decided, by whom, why, and what it
superseded. nothing else in the repo records a decision; `HANDOFF.md`, the code and the archived v2 plans
(`docs/archive/v2/`) point here.

* **a new decision** is an entry at the top of "the log", headed `### YYYY-MM-DD: Dnn, <title>` with the next
  D-id (**D33 is next**): who decided (the owner, or a build's default awaiting the owner), what, why, and the
  ids or entries it supersedes. an earlier entry it changes gets a one-line `> **superseded YYYY-MM-DD**`
  marker and is otherwise left as it was.
* **owner questions** (Q1-Q26 asked so far; the next is Q27) are asked in `HANDOFF.md`; the answer is recorded here and `HANDOFF.md`
  keeps no copy of it. their wordings are in `HANDOFF.md`'s git history (Q9-Q20 also in
  `references/owner-questions-2026-10-03/README.md`). not to be confused with the modulo quadrants "Q1"-"Q4"
  of D5 and M7a/M7b, which are not questions.
* the two sections below came from the v2 plans on 2026-10-08, verbatim: the D table was
  `v2-implementation-plan.md` §0, the log was `v2-plan.md`'s "decision log". inside them, "current design",
  "§2 M13e" and the like point into those plans, now `docs/archive/v2/`, and "`HANDOFF.md` Qnn" into the
  HANDOFF of that date (git history).

## the D table (D1-D30, v2; from `v2-implementation-plan.md` §0)

| # | question | recommended default | blocks |
|---|---|---|---|
| D1 | **decided: recommended default.** closure at infinity: `1/(-1, 0)` is written as `[-inf, -1)`, but the involution claim and `1/[1, inf)` = `(0, 1]` both need the flag to *propagate*: `(-inf, -1)`. state the rule as "±inf are ordinary points; an infinite endpoint is closed iff attained; a pole at a **closed** zero endpoint attains ±inf by the piece's sign". drop the "closure over limits" wording | propagate flags; `1/(-1,0)` = `(-inf,-1)` | M6 |
| D2 | **decided: recommended default.** indeterminate corners: `[-inf,-1] * [0]` is written as the entire line, but `1/[-1,0]` already uses the sharp limit-along-the-box rule. the same rule for mul: at an indeterminate corner `(±inf, 0)` the corner contributes `0` if the infinite factor's interval is non-degenerate, and the signed infinity if the zero factor's interval is non-degenerate. so `[-inf]*[0,1]` = `[-inf]`, `[-inf,-1]*[0]` = `[0]`, `[-inf,-1]*[0,1]` = `[-inf,0]`, `[1,inf]/[1,inf]` = `[0,inf]`, all matching 1788 up to closure at inf. general form: an indeterminate corner contributes the limit along each non-degenerate edge that meets it, so for sub at `(inf, inf)` it contributes `-inf` if the minuend is non-degenerate and `+inf` if the subtrahend is (`[inf]-[1,inf]` = `[inf]`, `[1,inf]-[inf]` = `[-inf]`, `[1,inf]-[1,inf]` = entire, as 1788); add at `(inf, -inf)` likewise. a box that *is* the indeterminate point returns `∅` + warning (D7). through the itf1788 adapter's input rule (1788 unbounded → open at inf) the infinite corner is never in the box, so D2 does not change conformance — it only affects user-typed literal `[-inf, …]` bounds | sharp rule | M6 |
| D3 | **decided: recommended default**, plus integral Fractions normalize to int in `Cut`. `int / int` that is not integral: Fraction (exact, per "never rounded") or float (what users expect)? | Fraction; `fmt` prints `1/3`; float only if an operand is float | M6 |
| D4 | **decided 2026-10-04 by owner: (a)**, with (b)'s two sentinels as read-outs of an infinite end only (D30). was deferred: the time layer is not being rebuilt now; v1's is archived with the rest of v1 at M10 and comes back in M8 on top of the v2 class (see M8, M10). infinities for the time layer: v1 stores float unix seconds (loses sub-µs, dodges the question). v2 options: (a) Fraction seconds in the numeric kernel, thin wrapper; (b) native datetime cuts + two sentinel objects that compare below/above everything | (a) — reuses every kernel test unchanged, when M8 happens | M8 |
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
| D15 | **decided 2026-09-25 by owner: recommended default.** licence. `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` are LGPL-2.1-or-later, the two `ieee1788-*.itl` files carry an all-permissive notice, the rest Apache 2.0, and this repo has no licence of its own. vendor all 19 files of oheim/ITF1788 at `b6ee1e2` unmodified into `tests/itf1788/`, replacing nehmeier's 7, with the fork's `LICENSE`, `NOTICE` and `COPYING.LESSER` beside them. the wheel ships only `multiinterval/`, so no test file is distributed with the library. **corrected 2026-09-26 at M13a**, from every file's header: five files carry the all-permissive notice, not two (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`); the eleven `libieeep1788_*` are Apache 2.0 | vendor unmodified, with the licence files | M13a |
| D16 | **decided 2026-09-25 by owner: recommended default; built 2026-09-26 (M13g), now in `v2-plan.md` "ieee 1788" (`DecoratedInterval`, `UndefinedOperationError`, `PossiblyUndefinedOperationWarning`).** decorations (com/dac/def/trv/ill), NaI and 1788's constructors go in a **separate decorated wrapper type**: the solver stack's (M11), brought forward. the core `MultiInterval` stays undecorated, so `v2-plan.md` "ieee 1788" ("decorations are not in the core") holds. 1788's signals, owner 2026-09-26 (`v2-plan.md` "2026-09-26 revision: owner answers"): `UndefinedOperation` **raises** (a `ValueError` subclass, so it reads like `MultiInterval(2, 1)`'s `ValueError`); `PossiblyUndefinedOperation` is an **`IntervalWarning` subclass** (the result is returned); names chosen when built. **no NaI** and no `ill` (owner 2026-09-26): its statements are rows under a new category, M13g | wrapper type; signals as the owner chose | M13g |
| D17 | **decided 2026-09-25 by owner**: M13 does **not** block the 2.0.0 release, and there is no hurry to release either ("I have zero users and this is a yak shaving pet project"). M13 only adds methods and a type and gives a meaning to exponents that raise `TypeError` today, so nothing that works now changes | release whenever; not blocked | — |
| D18 | **decided 2026-09-27 by owner**, on M13's proposed categories and choices: (a) **"tighter than the vector"** is an approved residual category (M13e): 11 keys, 18 vectors where 1788's expected hull is looser than the tightest double enclosure and ours is the tightest, checked with arb or exactly; the two grossly loose `pow_rev.itl:609`, `:642` stay in it. (b) **"exact parsing decides validity"** is approved (M13g): the 1788 text constructors read bounds exactly, so no `PossiblyUndefinedOperation` for a near-tie literal; 7 keys. (c) the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY` 12) stay under **decoration expectations**. (d) `set_dec` **demotes** as 1788's `setDec` does; only the `DecoratedInterval` constructor raises | both categories approved; rows stay; set_dec demotes | M13e, M13g |
| D19 | **decided in the build 2026-09-27 (the session's defaults); confirmed 2026-10-03 by owner (Q11, as built; `tol` documented as absolute, `max_steps` as boxes).** the solver stack's first part (M15): (a) `multiinterval/autodiff.py` (`Dual`, `derivative`) and `multiinterval/solver.py` (`newton`, `Root`) are public and exported from `multiinterval`, not newton as a test only; (b) newton's step runs only where `f` is proved C¹ by decorations (dac or better on the value and the derivative), else the piece is pruned and bisected; (c) the step is `mul_rev`, never `/` (D7); (d) one variable; (e) `tol=1e-10` absolute, `max_steps=10_000` | as built | M15 |
| D20 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q12, as built).** the solver stack's second part (M16a): (a) a gradient or a jacobian is n passes of `F`, `Dual` untouched (not vector mode); (b) names `gradient`, `jacobian`, `solve`, `RootBox`, public and exported from `multiinterval`, `Root` unchanged; (c) `solve` at n == 1 is `newton`; (d) uniqueness by krawczyk only, on the closed hull, with a float preconditioner (identity fallback), narrowing by gauss-seidel with `mul_rev`; (e) before an unproved box is output, its simplest rational point (exactly `[0]`: a unique point) and then krawczyk on the box inflated within its region; (f) wide components bisected first, round robin, then the widest; (g) `tol=1e-10` absolute on the widest component, `max_steps=10_000` boxes; (h) no direction tag | as built | M16a |
| D21 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q13) but (b): the layer's numbers of the empty set are `nan`, 1788's answer, not the library's `ValueError`; Q9 and Q10 closed as built.** the 1788 layer (M16b): (a) `multiinterval/ieee1788.py`, one `Interval` class for both flavours, 1788's names in snake_case (`NAMES` has the camelCase), not exported from `multiinterval`; (b) `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad` of the empty set raise `ValueError`, where 1788 says NaN; (c) Q9 answered by the layer: `ieee1788.mul_rev_to_pair` is 1788's pair with its decoration, the library's `mul_rev` unchanged; (d) Q10 answered by the layer: its pass runs the constructors in binary64, no class argument on the library's; (e) where 1788 defines another answer than the library's set (cancellation, overlap, attained infinities), the layer gives 1788's and the library keeps its own | as built | M16b |
| D22 | **decided in the build 2026-09-28 (the session's defaults); confirmed 2026-10-03 by owner (Q14, as built; every cut-tuple relation asserts normalized operands).** the per-piece allen matrix (M16c): (a) `A.allen_matrix(B)`, a tuple of tuples of `Allen` (rows the pieces of `A`, columns those of `B`), and `relations.allen_matrix` over cut tuples; (b) `A.allen_relations(B)`, the `frozenset` of the relations holding between some pair of pieces, a second public name the H3 row did not list; (c) an empty operand gives `()` / one empty row per piece / `frozenset()`, no raise and no warning; (d) the matrix is the plain `n x m` loop over `allen()` (no dependence on normalized input; the design's ~2-3x faster fill + sweep not taken), the set view an `O(n + m)` sweep that never builds the matrix; (e) methods on `MultiInterval`, functions in `relations.py`, nothing at the top level, not on `DecoratedInterval`, the sparse `(i, j, relation)` view private | as built | nothing (additive: `allen()` and every existing name unchanged); M16c's record |
| D23 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q15): (a), (b), (d), (e), (h) as built; changed: (c) `==`/`!=` against an ndarray is elementwise, (f) `fmin`/`fmax` are `minimum`/`maximum`, (g) numpy is in `[test]`; and the methods follow (h)'s rule (a mixed method call returns the class the operators do).** numpy interop (M16d): (a) `__array_ufunc__` on `MultiInterval`, `DecoratedInterval`, `Dual` (`multiinterval/numpy_compat.py`), operator ufuncs as python's operators on our dunders only, the others the method of the same set image, the rest `TypeError`; `__array__` on `MultiInterval` only (a 0-d object array); the array API standard not built (the alternative: an interval-array type); (b) a foreign real is its exact value (a `Rational` by type, else where `float()` would round), alternatives refuse or keep `float()`; (c) an ndarray meeting ours is elementwise into an object array, `==`/`!=` never broadcast; (d) no numpy-named alias methods; (e) `np.invert` the complement; (f) `fmin`/`fmax` not mapped; (g) numpy not in `[test]`; (h) both operands ours in a method ufunc (`hypot minimum maximum arctan2`): the subclass decides, as for the operators | as built | M16d |
| D24 | **decided in the build 2026-09-28 (the session's defaults); answered 2026-10-03 by owner (Q16): (a)-(c), (f) as built; changed: (d) `[fast]` pinned to `auto`'s window as `[test]` is, (e) one CI gate job on the forced gmpy2 backend.** the gmpy2/mpfr backend (M16e): (a) the default is the pure path; `MULTIINTERVAL_BACKEND=gmpy2` forces gmpy2 (ImportError if missing or below 2.3 / MPFR 4.2), `auto` takes it if importable and `2.3 <= version < 3`; (b) public surface: the env var and the `[fast]` extra only; `multiinterval.backend.name()` not exported, no setter; (c) non-dyadic points stay pure (no mpfr ziv loop), but atan, acot, atan2's angles and the hook's mixed operands; (d) `gmpy2>=2.3,<3` in `[test]`; (e) CI unchanged: the whole suite on the pure path, `tests/test_backend.py` compares both in every job; no gmpy2 fuzz job; (f) ships in 2.0 as an opt-in, or waits under "later" | as built | M16e |
| D25 | **decided 2026-09-28 by owner**: CPython 3.11's `Fraction.__pow__` rounds a Fraction base to a float before `MultiInterval.__rpow__` runs, so on 3.11 `Fraction(1, 3) ** OutwardMultiInterval(2)` misses 1/9, and nothing in the library can see it (the other Fraction operators defer correctly; 3.12 returns NotImplemented). drop 3.11, or keep it with `Fraction ** interval` documented as unsupported there? | **python >= 3.12**: `pyproject.toml` `requires-python`, CI's gate matrix 3.12-3.14; `tests/test_outward.py::test_a_fraction_base_stays_exact` is red on 3.11 | — |
| D26 | **decided 2026-09-30 by owner** (fuzz-rev-inf): to nearest, a reverse op's exact preimage wholly past MAX squeezes to the point `[±inf]` (IEEE 754 rounds such a value to ±inf; `_widened` reads it as `[MAX, inf]`), and intersecting with `x` after that rounding lost it whenever `x` is open at that infinity: `pown_rev(c, -1, (-inf, -2))` for `c = (-2.2e-309, 0)` was `{}` though its exact answer `(-inf, -4.49e308)` is not empty; the same at a finite double (`sqr_rev([2, 2.0000000000000004], (1.4142135623730951, 2])` was `{}`). options weighed with the owner: (a) `x` meets the preimage before the rounding, 1788's order; (b) keep it and document it; (c) the nearest class saturates an overflow to `(MAX, inf)`, 1788's enclosure rule, no longer python's float | **(a)**, in the reverse ops only (`reverse._keep_squeezed`): a part of the exact answer inside `x` that rounds wholly onto one double is that double, as a point: an end `x` excludes (the case above; the one way a result leaves `x`), or a point of `x` where the rounding kept an end open (`mul_rev(10, (1, 2), [0.1])`, whose exact answer holds the double 0.1, was `{}`, now `[0.1]`; a known loss M13e's tests had worked around by leaving `x` out of their checks). the nearest class keeps IEEE 754 round-to-nearest (as python's float) everywhere; forward ops unchanged: `[inf] & (0, inf)` is still `{}` (documented, README "rounding"). not a 1788 divergence row: the vectors test the outward class, which was already right | `multiinterval/reverse.py`; `tests/test_reverse.py::test_float_operands` (two `@example`s), `::test_exactly_the_points_with_f_in_c` |
| D27 | **decided 2026-10-03 by owner** (the 1788 departures census, 2026-09-30, never asked before): four departures from 1788 that were build choices are deliberate: step functions are point sets (`floor([-1.5, 1.5])` is four points, 1788 `[-2, 1]`); an end that rounding moved is open (M12); the divergence categories "degenerate infinities" (D1, D6 and the domain-end rule) and "cut-based relations" (the 2026-08-16 principles). the stale category "domain-clipped functions" (no row since M13d) is removed from `tests/itf1788/test_itf1788.py::REASONS` and the current design | as stated | `references/owner-questions-2026-10-03/ieee1788.md` |
| D28 | **decided 2026-10-03 by owner** (Q17, Q18, m14b-open's 4300 digits): pown of exact operands past one exact-result limit of about 2 ** 22 bits, shared by pown, `pow_` and exp2/exp10, is the tightest open float enclosure in the outward class and the value rounded to nearest in the nearest class, with a default-ignored warning; `elementary.EXACT_POWER_LIMIT` stays the float-corner threshold. pown to nearest is correctly rounded for every n (the exact power rounded once, `rounded_pow` past the threshold), not libm's `float ** int`. `repr` does not raise past python's 4300-digit limit (hex past it; `parse` reads it) | as stated | `references/owner-questions-2026-10-03/pown.md`; §2 "owner-answers" |
| D29 | **decided 2026-10-03 by owner** (Q19, Q20): each end keeps its own number type. the outward class is isotone within one grid; across grids `f(A)` lies within the tightest double cover of `f(B)` (documented; a public method rounds every end onto the double grid, outward). the exact class's crossed piece is the piece between the two values, each end keeping its flag (`rootn((10 ** -30, 1.0000000000000003e-30], 5)` is `[1e-06, 1/1000000)`) | keep per-end typing | `references/owner-questions-2026-10-03/q19.md`, `q20.md`; §2 "fuzz-steps-isotone", "fuzz-rootn-crossed" |
| D30 | **decided 2026-10-04 by owner: as recommended** (M8's three choices and the smaller ones with them; `references/m8-choices-2026-10-04/`). (a) a naive datetime is exact wall-clock seconds by subtraction from naive 1970-01-01, never `timestamp()`; an aware one its exact UTC seconds; mixing naive and aware raises `TypeError`; aware ends in different zones allowed, the left operand's zone kept for display only. (b) an infinite end reads out as one of two sentinels ordered below/above every datetime, date, timedelta and pandas type, taken back by the constructors; storage stays (a)'s ±inf, closed or open as written. (c) no end-of-day snap: a `date` is the half-open day `[d 00:00, d+1 00:00)`, a datetime an exact instant (v1's hour/minute/second snaps of datetime ends dropped too). with them: a non-microsecond end raises on read-out (a raw Fraction accessor beside it); comparisons return `TruthSet`; foreign `==` is `NotImplemented`, the wrappers hashable; `NaT`/nan raises | as decided | §2 M8 |

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


## the log (newest first; D31 on written here, the rest from `v2-plan.md`'s decision log)

### 2026-10-08: D32, precision first: what is known exactly is exact (revises D31)

the owner, 2026-10-08, the same day as D31: "abs can produce an exact end from abs [-1.0, 1.0] since 0 is
included, let's prioritize precision where it's possible"; then, asked, the step functions give ints and the acos
clip stays exact (both the session's recommendations). D31's first bullet (types from the operands, python's
tower, `/` as D3), its tie rule and its reasons stand; three of its bullets are superseded (marked there).

* **exact where the value is known exactly, not only carried**: two kinds of result value are exact (an `int` or
  a `Fraction`) whatever the operands' types. (a) the **constants a function reaches**: its interior extrema
  and its domain ends. abs's and `x ** 2`'s 0 and cosh's 1 already are (`abs(M(-1.0, 1.0))` is `[0, 1.0]`,
  `cosh(M(-1.0, 1.0))` `[1, 1.5430806348152437]`, unchanged); sin's and cos's ±1 become so
  (`cos(M(-1.0, 1.0))` was `[0.5403023058681398, 1.0]`, `sin(M(0.0, 2.0))` `[0.0, 1.0]`: their 1 becomes `1`).
  (b) the **integers the step functions list**: `floor`, `ceil`, `trunc`, `round` and the rest of
  `multiinterval.steps` give ints on a float piece, as python's `math.floor` does (`floor(M(-2.5, 3.0))` was
  `{[-3.0], [-2.0], ..., [3.0]}`, becomes `{[-3], [-2], ..., [3]}`)
* **everything else computed from a float operand stays a float** (D29, D31), even when its value happens to be
  exact: `M(1.0) + 1` is `[2.0]`, an end that is a float operand's own value (`abs`'s `1.0` from `-1.0`) is a
  float. the line is what the value depends on: a float operand's value carried through arithmetic or a
  continuous function is float; an integer by definition or a constant of the function is exact
* **a domain clip is exact**: the clip point of `M(-1.0000000000000002, -1.0)` against acos's domain is both the
  operand's `-1.0` and the domain's `-1`, a tie, so exact `-1` (D31's tie rule), and
  `M(-1.0000000000000002, -1.0).acos()` stays `(3.141592653589793, 3.1415926535897936)`, which contains pi,
  as today. it differs from `M(-1.0).acos()`, `[3.141592653589793]` to nearest, where the user gave the float
  point
* **ties as D31**: `abs(M(-1, 1.0))` and `abs(M(-1.0, 1))` are both `[0, 1]` (were `[0, 1]` and `[0, 1.0]`)
* **what m14b-open builds**: sin's and cos's extrema exact; the step functions' ints; the tie rule. abs and the
  acos clip are unchanged
* **the test for exact, confirmed by the owner** (2026-10-08: "only choose the exact type if we know for sure it
  has no ulp contamination, and the endpoint is the exact value. we could end up at 1.0 by many paths and not all
  guarantee that 1 is at the endpoint"): an end is exact only when its value is proved to be that exact number
  and attained there, decided on exact values as attainment already is; a value that merely rounds to it stays a
  float. pins for the build, probed 2026-10-08 (`h = 1.5707963267948966`, the double nearest pi/2, below it):
  `sin(O(0.0, h))` is `[0.0, 1.0)` (pi/2 outside the piece, the max below 1) and stays a float, `[0.0, 1.0]` to
  nearest; `sin(O(0.0, nextafter(h, 2)))` and `sin(O(0.0, 2.0))` hold pi/2, today `[0.0, 1.0]`, and become
  `[0.0, 1]` in both classes

### 2026-10-08: D31, a result's number type says where its value came from (Q26)

the owner, 2026-10-08, answering Q26 and the question under it: prefer one type per result ("all float, all
int, all frac") or the most precise type ("int over frac over float")? neither: the session's recommendation,
accepted ("okay with everything, record it"). it states the rule D3 and D29 already follow and fixes the cases
m14b-open found (`HANDOFF.md` row m14b-open; the fix is built before 2.0.0).

* **types come only from the operands.** `int` and `Fraction` mean exact, never rounded; `float` means on the
  double grid, so the next operation on it rounds. per end (D29), python's numeric tower: exact op exact stays
  exact, a float operand makes a float; `/` of exact operands is an `int` when integral, else a `Fraction`,
  never a float (D3, unchanged). a float is never upgraded to exact (`0.1` is not
  `3602879701896397/36028797018963968`) and an exact value is never silently made a float
* **a value inside a piece takes the piece's type**: a result end that is no operand's end (abs's 0, `x ** 2`'s
  0) is a float if the piece has a finite float end, as the step functions already decide per piece. so
  `abs(M(-1.0, 1.0))` is `[0.0, 1.0]` (was `[0, 1.0]`). `0.0` has no rounding error; the type records the grid,
  so `0.0 + 1/3` rounds where `0 + 1/3` does not (the owner: "whether a float can ever produce an int ... 0.0
  is right")
  > **superseded 2026-10-08 by D32**: a constant the function reaches (abs's 0) is exact
* **the step functions are per piece, all of them**: `trunc(M(-2.5, 3))` is all floats, as `floor` is (was
  `{[-2.0], [-1.0], [0.0], [1], [2], [3]}`)
  > **superseded 2026-10-08 by D32**: the step functions give ints
* **a domain constant contributes no type**: a clip at a domain end takes the operand's value there.
  `M(-1.0000000000000002, -1.0).acos()` clips to the point `[-1.0]` and is `[3.141592653589793]` in the nearest
  class, as `M(-1.0).acos()` (was the open `(3.141592653589793, 3.1415926535897936)`: the clip left `[-1, -1.0]`
  and acos ran on its exact cut). the outward class is unchanged (open around pi either way)
  > **superseded 2026-10-08 by D32**: the clip is exact, the result stays open around pi
* **a tie goes to the exact type**: an end reached by an exact and a float value of the same number is exact,
  so `abs(M(-1, 1.0))` and `abs(M(-1.0, 1))` are both `[0.0, 1]` (`[0, 1]` under D32) (were `[0, 1]` and `[0, 1.0]`: the order of the
  operand's ends decided). ends from different values keep their own types: `abs(M(-1, 2.0))` and
  `abs(M(-2.0, 1))` are `[0.0, 2.0]`, `abs(M(-1.0, 2))` is `[0.0, 2]` (the owner: "this is fine and I guess it
  makes sense")
* **why not one type per result**: it rounds free exact information away (`O([1.0, 2]) + Fraction(1, 3)` is
  `(1.3333333333333333, 7/3]`; all-float would open and widen the exact end, and every later step inherits it),
  and it is not inclusion-isotone either (Q19's report). **why not the most precise type**: only among exact
  types (an integral `Fraction` is an `int`, D3); a float made exact claims a precision nobody had
* **Fractions in results**: correct and visible rather than silently rounded; a user who wants floats gives one
  float operand (`M(1.0) / 3`). the README should say so where `/` is introduced, and the to-nearest
  `MultiInterval.rounded()` (row "later") would make the way back one call. long exact iterations grow their
  denominators: the docs should point heavy numerical work at floats


### 2026-10-06 revision: separators exactly between items (Q24)

the owner, 2026-10-06, answering Q24: a trailing or a doubled separator is refused ("okay yes refuse both"):
`[1,]`, `{1,}`, `{1,,2}` were read as if it were not there; a leading one was already refused. and items with
nothing between them (`[1,2)[3,4)`, v1's form, refused by Q23's build) stay refused, on the session's
recommendation: brackets cannot be part of a number, so the form is unambiguous, but allowing it makes a second
rule ("a separator between numbers, but brackets may touch") and lets white space alone part two pieces again
(`[1,2) [3,4)`), while nothing writes it and v1 is deleted. the rule is one sentence: two items always have
exactly one of `,` `;` `|` `∪` between them (`,` or `;` between a piece's two numbers)

### 2026-10-06 revision: a number is what python reads (Q23)

the owner, 2026-10-06, answering Q23: "each number should be something python can parse, split by a character
that's not a valid part of the number". the text syntax of `fmt.py` (`parse`, `parse_value`; not the 1788
literals, whose grammar is the standard's):
* two numbers with no separator are refused (`[0.1.2]`, `[-2-1]`, `{1-2}` were split into two);
* ASCII digits only (python's `\d`, `int` and `float` take other scripts': `[١٢]` was 12);
* a sign is attached to its number: `- 5` and `- inf` are refused (`float('- 5')` fails);
* white space around `/` stays (`Fraction(' 1 / 3 ')` reads it); `1/-3` stays refused;
* python's digit separators are read by python's rules: `1_000`, `1_0.5`, `1e1_0`; the malformed ones refused;
* `inf`, `infinity` and `∞` stay; hex stays (D28: `repr` writes an int past 4300 digits in hex); no binary or octal;
* the owner, the same day: "if spaces can be part of a fraction then should we require commas or semicolons as
  separators": yes, white space alone no longer separates two items (`[1 2]`, `{1 2}`, `1 2` were read); between
  a set's items `,` `;` `|` `∪` stay (the session's choice: explicit, documented, never white space).
the owner also said the string syntax is a side quest ("we could totally just not allow parsing strings and be
strict about creation"), kept because it is a natural way to write a set. reverses: yesterday's `1_000` refusal
(m14b-open), and the spaced sign accepted since M14-breadth (the v1 parity audit's `[- 5, 5]` row is history)

### 2026-10-05 revision: the package is `multiinterval`

the owner, 2026-10-05: the library needs a name not taken on PyPI (`intervals` is another project's, so the two
could not be installed side by side). chosen: `multiinterval`, one word, singular, the same spelling for
`pip install` and `import` (the stdlib's and PEP 8's style for packages; it names the main class). the package
directory `intervals/` is now `multiinterval/`, every import, path and doc pointer follows (the dated snapshots
in `references/` keep the old name), and the backend's variable is `MULTIINTERVAL_BACKEND` (was
`INTERVALS_BACKEND`; never released), likewise `MULTIINTERVAL_COREMATH_CACHE`. unchanged: the class names, the
repo folder, the conda env `intervals`, and the run ledger's format id `intervals-gate-ledger/1`. `multiinterval`
was free on PyPI on 2026-10-05 (no project under the name; claimed only by the first upload)

### 2026-10-05 revision: M8's choices confirmed (Q21)

the owner accepted the session's recommendation on each choice the time layer's build made beyond D30
(`v2-implementation-plan.md` §2 M8, its done-record and review round), all as built, nothing changes:
* (a) the sentinels are `NEG_INF`/`POS_INF`; (b) `td / td` and `td // td` a `MultiInterval`, `td % td` a
  `TimeDeltaInterval` (python's `timedelta` types), `td // real` refused (`MultiInterval // n` floors to whole seconds,
  python's `timedelta // n` to microseconds); (c) `repr` the constructor call, no time `parse`; (d) `tz=` for dates
* (e) a float factor is its exact value (`td * 0.1` is exact and its read-out raises; python rounds to the
  microsecond): the only inexact step would be a silent rounding; `Fraction(1, 10)` reads out
* (f) bounds ordered on their readings by `MultiInterval`'s rule (`D(tue, mon)` empty, `D(wed, mon)` raises), one rule
  over a "looks reversed" special case
* (g) aware `dt - dt` and `dt + td` are elapsed time between instants, consistent with `==` and `<` (and with pandas'
  aware `Timestamp`), not python's same-tzinfo wall clock
* (h) `pd.Timedelta % A` and `divmod(pd.Timedelta, A)` sound but wider than exact, `TimeDeltaInterval(x) % A` the
  documented workaround (no pandas hook defers it); (i) `degenerate_points` a tuple (a DST fold's two instants compare
  equal)

### 2026-10-04 revision: strict flags; v1 deleted (Q22)

the owner, on the v1 parity audit's two questions (`v2-implementation-plan.md` §4):
* **(a) a flag is a bool**: `start_closed`/`end_closed` (and `from_pieces`' and `Builder.add_piece`'s flags) take
  python's bool or numpy's; anything else is a TypeError (`cuts.flag`). they were read by truthiness, so
  `MultiInterval(0, 1, start_closed='no')` was `[0, 1]`, silently; `0`, `1` and `None` are refused too. the same in
  the time classes, for the empty set's flags as well. internal callers pass computed bools and are not checked
* **(b)** the v1 README is kept whole as `references/v1-readme.md` (its "notes:" and "Geminis feedback" were kept
  nowhere else); then H4: `archive/v1/` deleted, with its two differential tests and the `pythonpath` entry

### 2026-10-04 revision: M8's review round (three reviews, one fixer)

three read-only reviews of the build (soundness and D30, spec and API, sabotage); the session decided the
fixes, a fixer built them on branch `m8` (the record, with each finding's pin and the sabotage re-run,
`v2-implementation-plan.md` §2 M8 "review round"). behaviour changed, all in "the time layer (M8)" above:
* numpy's `timedelta64` (a `numbers.Integral` to numpy) and `datetime64` are no numbers: refused
  (TypeError) by the numeric class wherever it takes a number or an int argument, and by every scalar
  slot of the time layer (`MultiInterval(np.timedelta64(3, 'ns'))` was `[3]`, `td_iv * np.timedelta64(3,
  'ns')` was `td_iv * 3`)
* the order of two bounds is checked on their readings by `MultiInterval`'s rule, not on the values as
  written; so `D(tue, mon)` (from Tuesday through Monday) is now empty, not a `ValueError`, and
  `D(mon, noon_mon, start_closed=False)` a `ValueError`, not empty
* `degenerate_points` is a tuple sorted by instant (a set lost one instant of a DST fold)
* aware read-outs no longer pass through the UTC datetime (near the range's ends they overflowed) and
  read a tzinfo whose `dst()` is None
* `NaT` is a `ValueError` in every arithmetic operator; `to_pandas()` past pandas' range a `ValueError`
* documented, not changed: aware `dt + td` adds elapsed time; `pd.Timedelta % A` is pandas' own, wider
  remainder (no hook of pandas 3 lets the interval answer)

### 2026-10-04 revision: M8, the time layer, built

D30 built as decided, on branch `m8`; the design is now "the time layer (M8)" in current design, the record
(choices among the defaults, the build's own, the sabotage table) `v2-implementation-plan.md` §2 M8. no
deviation from D30; one consequence written down: `dt - dt` of aware ends is the elapsed time between the
instants also when both share a tzinfo (python's `-` then gives the wall-clock difference). the build's
choices for the owner to confirm: the sentinels' names `NEG_INF`/`POS_INF`; a sentinel is no arithmetic
operand; a float factor is exact (`td * 0.1` is not a whole microsecond); `td // real` refused; `tz=` built

### 2026-10-04 revision: the time layer's choices (M8, D30)

the owner accepted the recommendations of `references/m8-choices-2026-10-04/` (a read-only agent's report;
the session re-ran its claims). to be built in M8; the design goes into "current design" when it is:
* **time as exact seconds, wall clock for naive**: `DateTimeInterval` and `TimeDeltaInterval` wrap a numeric
  `MultiInterval` of exact Fraction seconds (D4's (a)). a naive datetime counts from naive 1970-01-01 by
  subtraction, never `timestamp()` (machine-dependent, DST-dependent, and `OSError` on windows before
  1970-01-02 and for `datetime.min`/`max`); this is pandas' reading of a naive `Timestamp`. an aware
  datetime is its UTC instant; naive and aware do not mix (`TypeError`, as python and pandas); aware ends
  in different zones do, the left operand's zone kept for display, never part of `==`
* **infinite ends**: two sentinels, ordered below / above every datetime, date, timedelta and pandas type
  and taken back by the constructors, are what an infinite end reads out as; the numeric value underneath
  is ±inf, closed or open as written, as everywhere in v2. `math.inf` does not order against a datetime
  and `datetime.max` is a finite instant, so neither can stand in
* **no end-of-day snap**: a `date` is the half-open day `[d 00:00, d+1 00:00)`, its flag saying whether
  the day is in; a datetime is an exact instant (v1 also stretched `10:00` to `10:59:59.999999`). with
  exact seconds the snap leaves a gap between adjacent days, misses `23:59:59.9999995` and makes a day
  shorter than 86400 s; the half-open day tiles, as `[0, 1) | [1, 2)` does
* with them: an end that is no whole number of microseconds raises when read out as a datetime (a raw
  Fraction accessor beside it; never rounded silently); comparisons return `TruthSet`; `==` with a
  foreign type is `NotImplemented`; the wrappers are immutable and hashable; `NaT` and nan are refused

### 2026-10-04 revision: v1's leftovers settled; shifts dropped

the owner, going through what v1 had that v2 lacks (the implementation plan's §4 surface map):
* **shifts dropped**, reversing Q6-shift (built 2026-10-03, `0513109`; reverted). no use case: `<<` on a
  set only saves writing `* 2 ** n`, and the one plausible use, running integer or fixed-point code over
  sets (`sample >> 8`), wants python's int floor (`3 >> 1` is 1, `-3 >> 1` is -2), which the exact
  `>>` (`M(3) >> 1` is `[3/2]`) did not give. integer ranges are not supported, and would differ
  from the reals in kind (no open ends); 1788 has no integer interval type either. `A * 2 ** n` and
  `A // 2 ** n` remain
* **v1's `merge`**: `union(*)` and `intersection(*)` cover it. the "exactly / at least k overlaps" mode
  and its mixed-input parsing (numbers, sets, lists, tuples, loose strings) were artifacts of how v1
  was written, not features: gone
* **`random_multi_interval`**: v1's test helper, gone (hypothesis strategies do its job)
* **a public `apply()`**: not now; `applicator` stays internal

### 2026-10-03 revision: owner answers to Q9-Q20 and the owner's-call items

the owner accepted every recommendation of `references/owner-questions-2026-10-03/` on 2026-10-03
("i'll accept everything fable said"): nine reports, one per group of questions, each option's pros,
cons and when it is the better choice (`README.md` there is the table). where a report said
"consider", "optional" or "later if asked", nothing was decided beyond that. in short:
* **kept as built, now owner-confirmed**: D19 (Q11, M15) and D20 (Q12, M16a) whole; D21 (Q13, the
  1788 layer) but (b); D22 (Q14, allen) whole; D23 (Q15, numpy) (a), (b), (d), (e); D24 (Q16, the
  backend) (a)-(c), (f). Q9 and Q10 close as built in the 1788 layer (`ieee1788.mul_rev_to_pair`; the
  constructors' binary64 pass); `layer-numpy`: the layer keeps `__array_ufunc__ = None`
* **Q20 (D29)**: a crossed piece in the exact class stays the piece between the two values, each end
  keeping its flag (`[1e-06, 1/1000000)`), as plain `*` already gives a float end beside an exact one
* **Q19 (D29)**: the outward class stays typed per end, so it is isotone within one grid and, across
  grids, `f(A)` lies within the tightest double cover of `f(B)`; documented, and a public method puts
  every end on the double grid, outward, for a user who needs isotonicity. rejected: doubles only in
  the outward class (drops exact points such as the solver's `1/6`), and rounding whenever a float is
  mixed in (not isotone either: a purely exact `B` stays exact)
* **Q17 and Q18 (D28)**: pown of exact operands past one exact-result limit of about 2 ** 22 bits,
  shared by pown, pow and exp2/exp10, is the tightest open float enclosure (outward) or the value
  rounded to nearest, with a default-ignored warning; `EXACT_POWER_LIMIT` stays the float-corner
  threshold. pown to nearest is correctly rounded for every n (libm's `float ** int` was off by an ulp
  on this laptop; it was the one libm value in the library). `repr` must not raise past python's
  4300-digit limit: hex past it, which `parse` reads
* **Q13(b)**: the 1788 layer's numbers of the empty set are `nan`, 1788's answer; the library keeps
  D9's `ValueError`
* **Q15(c), (f), (g), (h)**: `==`/`!=` against an ndarray is elementwise; `fmin`/`fmax` are
  `minimum`/`maximum`; numpy joins `[test]` and the README numpy section becomes doctests; a method
  mixing the two classes returns the class the operators do (a defect: `M(0.1).hypot(O(0.1))` was a
  `MultiInterval`, against README "rounding")
* **Q16(d), (e)**: `[fast]` pinned to the window `auto` takes, as `[test]`; one CI gate job on the
  forced gmpy2 backend, no gmpy2 fuzz
* **the four 1788 departures never asked (D27)**: confirmed deliberate; the stale "domain-clipped
  functions" category goes
* **Q6-shift**: `A << n` is `A * 2 ** n` and `A >> n` is `A * 2 ** -n`, exact, through `*`; negative n
  allowed; `//` stays the floor. **Q6-rest closed**: neither `random_multi_interval` nor a public
  `apply()` in 2.0. **vectors-ext (b) closed**: glibc's rows are conformance inputs, not hard cases;
  pown takes CORE-MATH's pow rows with integral exponents instead
* small, from the same reports: `allen_relations`' set is documented as extensional and every
  cut-tuple relation asserts normalized operands; the solver documents `tol` as absolute and
  `max_steps` as boxes, splits symmetrically inside (0, 1], and `newton`'s width test cannot overflow
  (open item `newton-width`)
* left open, as the reports said: `Root`/`RootBox` as dataclasses if a third state appears; an `rtol`;
  the outward fma, `%`, hypot and `cancel_minus` per corner (tighter, optional); rootn run on cbrt's
  worst-case inputs (optional); a public strategies module after 2.0 if asked

### 2026-09-30 revision: to nearest, a reverse op meets `x` before rounding (D26)

* the fuzz job found `pown_rev(c, -1, (-inf, -2))` empty to nearest for `c = (-2.2e-309, 0)`, whose
  exact answer `(-inf, -4.49e308)` is wholly past -MAX: the preimage rounds to the point `[-inf]` and
  `x`, open at -inf, dropped it (plan §2 "fuzz-rev-inf"). 1788 intersects with `x` first and then
  encloses (`[-inf, -MAX]` there, no infinite point); python's float rounds such a value to -inf
* chosen (owner): 1788's order in the reverse ops, python's rounding kept: `x` meets the preimage
  before the rounding, so a part of it inside `x` that rounds wholly onto one double is that double,
  as a point: `[-inf]` here, an end `x` excludes (the one case where a result leaves `x`); or a point of
  `x` at an end the rounding kept open (`mul_rev(10, (1, 2), [0.1])` is `[0.1]`, was `{}`). the nearest class stays IEEE round-to-nearest everywhere, so forward ops
  are unchanged (`(M(1e308) * 10) & M.parse('(0, inf)')` is `{}`: `[inf]` is the point inf).
  rejected: saturating an overflow to `(MAX, inf)` in the nearest class (1788's enclosure rule; it
  would stop matching python's float), and keeping the loss as a documented rule
### 2026-09-29 revision: fuzz on push, not on a schedule

* the owner: the fuzz runs "fully autonomously or not at all"; nobody reads a scheduled run's
  failure email. so `fuzz.yml` runs on every push to `master` (and `workflow_dispatch`), not weekly;
  the same fuzz run happens locally before each push (`tools/prepush.sh`), and the session that
  pushed watches both workflows to the end (`tools/ci_watch.sh`, a babysitter agent), debugging a red
  run from its log and artifact. the push procedure is in `CLAUDE.md`
* the fuzz database was never kept on CI: hypothesis loads its `ci` profile at import under
  GitHub Actions, whose database is None, and the `fuzz` profile, registered without one, inherited
  it, so the first two GitHub runs saved no example and the carried `.hypothesis` held only
  hypothesis's constants cache (their artifacts have no `examples/`). found 2026-09-29 by replaying
  the second run's artifact locally, which replayed nothing. the profile now names its database;
  `tests/test_fuzz_profile.py` pins it (red without it, under a simulated CI)

### 2026-09-29 revision: `-` and `+` keep each cut's type (fuzz-symmetry)

* a point can hold one value in two types: `[0, 1/2] & [0.5, 1]` is the point with a float 0.5 below
  and an exact 1/2 above, and the fuzz drew the other way round. the applicator reads a point by its
  low cut alone (`applicator._ends`), so `-` of the fuzz's point gave `[-1/2]` exact, while the reverse ops read each end by its own
  type (`reverse._end`): `pown_rev(-c, -7)` was not `-pown_rev(c, -7)`, found by M14's first GitHub
  fuzz run (plan §2 "fuzz-symmetry")
* `-` is now the cut mirror and `+` the identity, each cut keeping its type, as `-` already did for a
  piece with two ends: exact, an involution, and the `-` `test_trig_rev_symmetry` had to build for
  itself (`reverse.negate`). the other forward ops still read a point at its low cut; the reverse ops
  still read each end by its own type. a mixed point is sound either way; only its rounding differs

### 2026-09-29 revision: pown-huge, built

* pown (`A ** n` for an integral n, and every path to it: `ieee1788.pown`, `Interval ** n`,
  `DecoratedInterval`, `Dual`, `np.power`) of a float corner with a huge n never finished: the outward
  descriptor built the exact `Fraction(x) ** n`. it now builds it only within `EXACT_POWER_LIMIT` bits
  and otherwise rounds with `elementary.rounded_pow`, the route pow already took, so pown and pow give
  the same doubles; the design, its proof and the tests are plan §2 "pown-huge". "rounding"'s "fn is
  exact" holds for pown up to that limit; past it attainment is against a marker equal to nothing,
  which the proof shows is the exact value's answer
* the nearest class past `|n| = 2 ** 53` is `rounded_pow` to nearest: python's `float ** int` lost the
  parity (a wrong sign) or read an int past the double range as an overflow (`M(0.5) ** 10 ** 400` was
  `[inf]`). up to 2**53 it is python's `float ** int` as before (whether that promises correct
  rounding is open, Q-nearest-libm)
* the descriptor's name is bounded (`pow<a 20001-bit int>`): `f'pow{n}'` raised python's 4300-digit
  ValueError in every class
* exact int/Fraction operands are unchanged and still huge by nature (Q-exact, for the owner)

### 2026-09-28 revision: python 3.12 minimum (owner, D25)

* CI's first run of M15 and M16 (36402681261) was red on python 3.11 alone: CPython 3.11's `Fraction.__pow__` answers a non-rational exponent with `float(a) ** b`, so `Fraction(1, 3) ** A` reached `A.__rpow__` as `0.3333333333333333 ** A`, unsound in `OutwardMultiInterval` (1/9 missed) and indistinguishable, inside the library, from a float base. the owner chose python >= 3.12 over documenting `Fraction ** interval` as unsupported on 3.11 (`pyproject.toml`, `ci.yml`); pinned by `tests/test_outward.py::test_a_fraction_base_stays_exact`

### 2026-09-28 revision: M16e, the gmpy2/mpfr backend (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now" (`HANDOFF.md` H3; plan §2 M15: "gmpy2/mpfr stay out"). built as M16e,
one of M16's five streams. the choices, each the build's default, open for the owner (D24, Q16):
* **the pure path is the default**; gmpy2 only with `MULTIINTERVAL_BACKEND=gmpy2` (forced) or `auto`.
  the design had automatic-when-importable; its critique held that the conservative reading wins:
  the pure path is the reference, the local gate and itf1788 then keep checking it, and a user's
  gmpy2 (linked to whatever MPFR their distribution ships) never changes code paths unasked
* **the backend only picks the double** (the contract above), so it is correct by construction
  wherever it declines, and bit-identical where it answers (the differential)
* **declined where one MPFR call is not one rounding**: non-dyadic inputs (but atan2 of two ints and
  the hook's exact mpq), `log` to a base, `acoth`, `rootn` with n < 0, `k pi + ...`; and short of
  where MPFR's limits bite: `rootn` from `n = 2**31` (a margin: gmpy2 raises from `2**32` on
  windows), an operand past `2**20` bits
* **`gmpy2>=2.3,<3` in `[test]`** (the differential never skips, and CI installs the series `auto`
  takes) and a new extra `[fast]` (`gmpy2>=2.3`, unpinned); no CI change:
  every gate job installs `.[test]`, so `tests/test_backend.py` runs everywhere, and the rest of the
  suite runs on the pure path there as here
* **`auto`'s window is `2.3 <= version < 3`** (the series verified); forced takes any gmpy2 at the floor

### 2026-09-28 revision: M16d, numpy interop (H3's second part), built

the owner asked (2026-09-27) for the rest of H3; numpy was one of its five streams. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q15; D23 in
v2-implementation-plan.md):
* **`__array_ufunc__`, not the array API**: the owner's 2025-12 line names "numpy compat via
  data-apis.org/array-api or `__array_ufunc__` or the interoperability page"; a multi-interval is an
  element, not an array, so the hook; the array API would be a new interval-array type (Q15(a))
* **a foreign real is exact** (Q15(b)): a `numbers.Rational` by type, any other real by value where
  `float()` would round it; fixes the outward class for `np.longdouble` on linux and gmpy2's `mpq`
  and `mpfr` operands; a double stays the float it is
* **an ndarray meeting ours is elementwise into an object array**, `==`/`!=` never broadcast
  (Q15(c))
* **no numpy-named alias methods** (`arcsin`, `rint`, ...) on the classes (Q15(d)), so numpy's
  object loops and `np.round` refuse them
* **`np.invert(A)` is the complement `~A`** (Q15(e)); `fmin`/`fmax` not mapped (Q15(f))
* **numpy not in the `[test]` extra** (Q15(g)); the README's numpy section is prose
* **both operands ours in a method ufunc: the subclass decides** (Q15(h)), as for the operators, so
  `np.hypot(M, O)` is outward (the review found the first operand's class decided)
* found on the way and fixed (not choices): M15's `Dual ** r` computed `r - 1` in r's own
  arithmetic, unsound in the outward class with plain python floats (`Dual.variable(O(1e300)) ** 0.1`
  missed its derivative, bare and decorated; `Dual.variable(O(-1)) ** 2.0 ** 60` had its sign
  flipped); a number exponent is now an int (integral) or a point set of u's kind first, which
  also changes the nearest class (an exact derivative for an integral float exponent; pow, not
  pown, for an r like 1e-20 whose float `r - 1` is -1.0)

### 2026-09-28 revision: M16c, the per-piece allen matrix (H3's second part), built

the owner, 2026-09-27: "get the rest of h3 done" (`HANDOFF.md` H3). the per-piece allen matrix
was built as M16c, one of five streams. the choices the build made, each the session's default,
open for the owner (`HANDOFF.md` Q14; D22 in `v2-implementation-plan.md`):
* **two views**, the 2026-08-16 note's "per-piece relation matrix, or the set of relations
  holding between any piece pair": `A.allen_matrix(B)` (nested tuples of `Allen`, not a matrix
  class) and `A.allen_relations(B)` (a `frozenset`). the set view is a second public name the H3
  row did not list; built because the note motivates it and it is the `O(n + m)` form. functions
  over cut tuples in `relations.py`, methods on `MultiInterval`
* **an empty operand gives no rows or empty rows and `frozenset()`**, not `allen()`'s
  `ValueError`: a matrix owes one entry per pair and there are none. matches `before`, `after`,
  `adjoins` of an empty operand (False) and the pointwise comparisons (`NEITHER`)
* **the matrix is the plain `n x m` loop over `allen()`**, the conservative choice: it does not
  depend on the operands being normalized. the design's alternative filled every entry BEFORE or
  AFTER with one cut comparison and overwrote the pairs the sweep visits: faster (~3x at 1000
  pieces the designer measured 2026-09-27; ~2x measured 2026-09-28 on a loaded laptop, M16c's
  record), but wrong on a cut tuple whose pieces are out of order. not taken; the trade is recorded
* **the set view is the merge sweep** (`relations.py::_allen_pairs`) plus two corners, and never
  builds the matrix; pinned by counts (calls to `allen()` and cut comparisons), not time. it needs
  normalized operands and asserts them under `__debug__` (the review, 2026-09-28); the matrix
  takes any cut pairs
* **not on `DecoratedInterval`** (through `.interval`, as `allen`); the sparse `(i, j, relation)`
  view private; nothing at the top level
* not built: allen's composition table over the cut reading ("later")

### 2026-09-28 revision: M16b, the 1788 layer (H3's second part), built

the owner asked (2026-09-27) "get the rest of h3 done", which supersedes 2026-09-26's "numpy and
gmpy2/mpfr recorded, not now"; H3's rest was built as M16 in five streams, this one M16b. the
2026-08-16 decision ("if real conformance is ever needed, it's a thin wrapper class in
`ieee1788.py`, never a mode on MultiInterval") is now built, as designed there. the choices the
build made, each the session's default, open for the owner (`HANDOFF.md` Q13; D21 in
`v2-implementation-plan.md`):
* **shape**: `multiinterval/ieee1788.py`, one class `Interval` for both flavours, snake_case 1788 names
  with 1788's camelCase in `NAMES`, not exported from `multiinterval`
* **numbers of the empty set raise** (`ValueError`, D9's answer, as the owner chose for the
  reductions, Q2) where 1788 says NaN; `inf`/`sup` of it are `±inf` either way
* **Q9 and Q10 answered by the layer, pending Q13 (c), (d)**: 1788's pair with its decoration is
  `ieee1788.mul_rev_to_pair` and the library's `mul_rev` stays one op, trv; the constructors' binary64
  run is the layer's pass, no class argument on the library's constructors. the adapter's 52
  `DECORATION_ONLY` rows and its reasons are left as they are (true of the library).
  why the library's `mul_rev` stays trv rather than decorated as the division: in the library ±inf
  are points, so where 0 is not in `b` the two sets differ (`mul_rev([1, inf], [inf])` is `(0,
  inf]`, `[inf] / [1, inf]` is `[inf]`; measured 2026-09-27, re-checked 2026-09-28), and a
  decoration copied from the division would claim com or dac for a set that is not the division's; a
  multi-piece `b` gives more than two pieces, so a pair has no meaning on the core; and 1788
  decorates its own `mulRev`, the hull of the same set, trv (`libieeep1788_rev.itl:988`), so one
  library op cannot match both
* **where 1788 and the library disagree, the layer answers 1788's way** (cancellation's "no answer",
  `meets`, attained infinities dropped), the library its own
* **the sign of zero follows 1788 in the layer**: `inf` gives `-0.0` for a lower end of 0 and `sup`
  `+0.0` for an upper end of 0 (1788-2015's rule; the critique's n7: "recorded, not an owner
  question" did not fit the layer's principle of answering 1788's way). no vector can see it
  (`tests/itf1788/itl.py::parse_number` reads `-0.0` as `Fraction(0)`); pinned by
  `tests/test_ieee1788_layer.py::test_the_sign_of_zero_is_1788s`. the package's one zero (2026-09-22
  revision) is unchanged: the ends of an `Interval` are never `-0.0`, and nothing re-enters the
  library from `inf`/`sup`

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

### 2026-09-27 revision: M15, autodiff and interval newton (H3's first part), built

the owner asked for H3 first; its suggested first pick (2026-09-26 revision below) was built. the
choices the build made, each the session's default, open for the owner (`HANDOFF.md` Q11; D19 in
v2-implementation-plan.md):
* **public, in the package**: `multiinterval/autodiff.py` (`Dual`, `derivative`) and
  `multiinterval/solver.py` (`newton`, `Root`), exported from `multiinterval`, as the 2025-12 layout sketch
  named them (`autodiff.py`, `solver.py`), rather than newton "as a test" only
* **C¹ is proved by decorations** (dac or better on the value and on the derivative), the decorated
  type's use in the solver that the "later" list kept back; where it fails the piece is only pruned
  and bisected. the alternative, trusting the caller that `f` is differentiable, is the silent
  failure `tests/test_solver.py::test_not_c1_would_lose_a_zero` shows
* **the step is `mul_rev`**, the reverse op, never `/`: `[0] / [0]` is `∅` (D7)
* **one variable**: a `Dual` carries one derivative; several variables (a gradient, a jacobian and
  a krawczyk or newton step in n dimensions) are not built. **superseded 2026-09-28**: built as M16a (the 2026-09-28
  revision above), `Dual` untouched
* `tol=1e-10` absolute, `max_steps=10_000`; float midpoints; the magnitude split at a factor of 16

### 2026-09-27 revision: owner answers on M13's proposed categories (D18)

* "tighter than the vector" (M13e: 11 keys, 18 vectors, the two grossly loose `pow_rev.itl:609`,
  `:642` included) and "exact parsing decides validity" (M13g: 7 keys) are approved residual
  categories; the `(PROPOSED)` markers are gone from `REASONS` and from the rows' reasons
* the 15 rows where an exact value past the doubles keeps com (`_BOUNDED_EXACTLY` 3, `PLAIN_ONLY`
  12) stay under "decoration expectations"; `set_dec` keeps demoting as 1788's `setDec` does
* the adapter's `tests/itf1788/test_itf1788.py::is_decorated` unpacked an `Interval` result (a
  NamedTuple) into its fields, so a decoration on the result alone went unseen; 0 vectors were
  affected in what runs (the 103 it missed are all of the six decorated ops, which take the decorated
  path by op), but the census undercounted: 1624 vectors carry a decoration, not 1521 (121 of the
  decorated ops, not 18; measured 2026-09-27). fixed and pinned by
  `::test_is_decorated_sees_the_result_alone` (the old line: 1 red)

### 2026-09-27 revision: M13 merged, M13's exit met

M13e (the reverse ops) and M13g (decorations, constructors and signals), built in parallel on
branches `m13e` and `m13g`, are merged on `m13-merge` (details in v2-implementation-plan.md, "exit
for M13"). M13 is done: every statement of the 19 files is a vector of an op in `OPS`, and
`tests/itf1788/test_itf1788.py::test_nothing_is_skipped` asserts `SKIPPED` empty. what the merge
itself decided, each the conservative reading:
* **the reverse ops take decorated operands** ("ieee 1788" above, decorated reverse ops): the
  core's set, decorated trv, as 1788 decorates a reverse op's result
  (`multiinterval/reverse.py::_decorated`); a bare `MultiInterval` beside a `DecoratedInterval` is a
  `TypeError`, as for the wrapper's other ops
* **mulRevToPair's better decoration is not built**: 1788 decorates the pair's first interval as
  the decorated division `c / b` where `0 ∉ b`, but its mulRev, the same set's hull, trv. ours is
  one op, `mul_rev`, which cannot be both; it is trv (sound: trv claims nothing). 52 vectors are
  rows under "decoration expectations", on the decoration alone (the set must match). whether to
  add a pair op with 1788's decoration is an owner question
* the counts at M13e and at M13g are replaced by one re-measurement ("counts at M13's merge");
  the two categories proposed at M13e and M13g were approved the same day (next revision up)

### 2026-09-26 revision: owner answers to the open questions

the owner answered `HANDOFF.md`'s questions and items on 2026-09-26:
* **Q3 and Q7 confirmed**: the readings of M13c (equal infinite ends as 1788 writes them;
  `.interior` in the reals) and M13d (`0 ** y` for y <= 0 outside the domain, `1 ** ±inf` and
  `inf ** 0` indeterminate, an integral Fraction exponent is pown, acot continuous) stand as
  recorded in their entries below
* **Q4 closed, no inward variant.** 1788 has none either: `cancelMinus` at level 2 is the hull of
  the exact level-1 answer, an outer enclosure, which `OutwardMultiInterval` already gives.
  reopen if a solver needs a certified inner answer; exact operands give one today
* **Q5 on hold** (the time layer, M8): no rush
  > **superseded 2026-10-04**: answered by D4 (a) and D30, the time layer built (M8)
* **Q6: `<<` and `>>` will be ported**; `random_multi_interval` and a public `apply()` are kept as
  to-dos, undecided. v1 applied python's int shifts endpoint-wise (`archive/v1/multi_interval.py::__lshift__`),
  so floats raised; the meaning on real sets (`A << n` as `A * 2**n`, and `>>` as exact division
  or as python's floor) is chosen when built
  > **superseded 2026-10-04**: the shifts were built, then dropped ("v1's leftovers settled; shifts dropped" above);
  > `random_multi_interval` a test helper, `apply()` not public
* **H1**: release 2.0.0 when everything is fully done, which narrows D17's "whenever"
* **H4**: delete `archive/v1/` after v2 is stable
* **H5**: the v1 README's reading list and illustration to-do are kept, moved to
  `references/todo-from-v1-readme.md` so they outlive `archive/v1/`
* **H2**: push `v2` approved
* **Q1, after 1788's context** (in 1788 every signal is a flag, like IEEE 754's: the result is
  returned, empty for bare and NaI for decorated, and the program goes on, since the result itself
  carries the news: empty and NaI propagate). the owner's choice for python, where errors raise:
    * **`UndefinedOperation` raises**, a `ValueError` subclass: a 1788 constructor given invalid
      input (`numsToInterval(2, 1)`, `"[ foo ]"`) stops, as `MultiInterval(2, 1)` already does.
      so NaI, which 1788 makes only from such input, arises only when made on purpose
    * **`PossiblyUndefinedOperation` warns** (an `IntervalWarning` subclass) and returns 1788's
      result. an exact parser can always decide validity, so it may never fire; the one vector
      that expects it (`ieee1788-exceptions.itl:18`) is settled when M13g is built
    * **`IntvlPartOfNaI` raises** (owner, 2026-09-26, after first leaning to a warning): NaI
      propagates like NaN through decorated ops, but `intervalPart` is the one place it stops,
      turning "invalid" into `∅`, which then reads as a real "no values"
    * **no NaI at all** (owner, 2026-09-26, Q8): with both raising, nothing in the library makes
      NaI. its only other uses, a per-element "invalid" in batch work and reading another 1788
      library's `[nai]`, come with numpy or data import, if ever, and it can be added back then.
      the decorated type has com/dac/def/trv, no `ill`, so `IntvlPartOfNaI` cannot arise; the
      statements needing a NaI become rows under a new divergence category, "no NaI: invalid
      input raises", approved with the decision (details: v2-implementation-plan.md M13g)
* **Q2 settled: the reductions keep raising `ValueError`** where 1788 answers the float `nan`
  (they are operations on floats, not intervals, so an empty interval is no answer either);
  `math.fsum` raises on `inf + -inf` too
* **H3**: numpy interop and a gmpy2/mpfr backend are recorded, not built now ("later (not in
  v2.0)" above). the session's suggested first pick when the solver stack starts: Newton's
  method with forward-mode autodiff, the demonstration of what multi-intervals are for. **superseded
  2026-09-27** by the owner's "get the rest of h3 done": numpy interop and the gmpy2/mpfr backend
  were built as M16d and M16e (the 2026-09-28 revisions above)

### 2026-09-26 revision: M13e, the reverse ops, built

M13e is complete (2026-09-26; details in v2-implementation-plan.md, M13e, parts 1 to 4 and the
close-out). D12 is now in "current design" whole: elementary and step functions (the reverse ops,
their exact pieces and D12's cap for the periodic ones), empties and warnings (`HullWarning` for
the periodic ones past the cap or over an unbounded piece of `x`) and ieee 1788 (the reverse ops,
reverse multiplication, the periodic and the power reverse ops, and all of them together). the
choices the plan left open are the four parts' entries below; the close-out made none. in short:
* **ten functions in `multiinterval/reverse.py`**, exported from `multiinterval`, each the exact set
  `{t ∈ x : f(t) ∈ c}` (for `mul_rev`, `pow_rev1`, `pow_rev2`, `∃` over the other operand) with the
  library's own f, ±inf points like any other, where 1788 answers its hull; an irrational end is its
  tightest float enclosure, open, and the domain is intersected after rounding
* **the itf1788 vectors**: all 1955 of the 19 ops run in both passes; 1879 match, 52 are
  degenerate infinities (the unary `pownRev` with n < 0 and 0 in `c`), 6 `[nai]` rows move with
  M13g, and 18 (11 keys) are under **"tighter than the vector", still PROPOSED and awaiting the
  owner**: 1788's hull is looser than the tightest, which ours is
* **a pin until M13's exit**: `tests/itf1788/test_itf1788.py::test_only_m13g_ops_are_skipped`, since
  dropping a reverse op from `OPS` turned no test red before it (its statements only become skips)
* itf1788 now 9269 vectors of 102 ops; 273 statements of 9 ops skipped, all M13g's (2026-09-26,
  `tools/itf1788_census.py`). the totals in "ieee 1788" above are M13d's until the merge with
  M13g's branch re-measures them

### 2026-09-26 revision: M13e, fourth part (pow_rev1, pow_rev2), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 4). `pow_rev1` and
`pow_rev2` moved into "current design" (elementary and step functions: reverse ops; ieee 1788: power
reverse ops). the choices the plan left open, each the most conservative reading, are now current
design too:
* **the library's own pow defines them**, ±inf as points: `pow_rev1([-2], [0, 1])` is `[1, inf]`
  (`inf ** -2` = 0), `pow_rev1([inf], [0])` is `[0, 1)` and `pow_rev2([0], [0])` is `(0, inf]` (`0 **
  inf` = 0); `0 ** y` for y <= 0, `1 ** ±inf` and `inf ** 0` have no value, so they solve nothing.
  1788 has no infinite points and every 1788 vector gives the domain through the input rule, so no
  vector sees the difference
* **the operand order is 1788's** (`b` or `a` first, then `c`, then the domain), and the domain is
  named `x` for the bases and `y` for the exponents, defaulting to `[-inf, inf]`
* **the float rule is pow's, per operand**: a finite float end in either operand makes every end
  float (to nearest, flags kept; outward, a moved end open), as `**` does; part 1's ops take each
  end of `c` on its own (the functions' rule). so `pow_rev1([n], c)` equals `pown_rev(c, n)` on `[0,
  inf]` when `c`'s ends are all exact or all float, not when they mix
* to nearest, an end past the largest double is inf, closed, the library's nearest rule (`M(1e200)
  ** M(1000.0)` is `[inf]` too); `OutwardMultiInterval` keeps it open
* the class, the coercion of a number, the warnings (an empty operand only; a negative base or a
  point with no value warns nothing) and `∩` the domain after rounding are part 1's
* **a bug found while building**: `MultiInterval(2).log(4)` hung, since `elementary._exact_log`
  looked only for an int k with `4 ** k == 2` and ziv's loop never settles on the rational 1/2.
  `log_b x` is now exact wherever rational, through each operand's perfect-power decomposition
  (`elementary._perfect_power`; since M13e's review, 2026-09-27, euclid on the exponents,
  `elementary._log_ratio`, as the root search took minutes on `3 ** 20000 + 1`); `pow_rev2` needs it,
  as every `powRev2` vector's ends are rational
* **a proposed category gains rows**: two `powRev2` vectors expect a hull far looser than the
  tightest (`[entire]` and `[-infinity, 0.0]` where it is `[-inf, -0.5]`); they are rows under
  "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 804 vectors: 802 match in both passes, 2 are those rows. itf1788 now 9269 vectors of 102 ops;
  273 statements of 9 ops skipped, all M13g's (2026-09-26, `tools/itf1788_census.py`)

### 2026-09-26 revision: M13e, third part (sin_rev, cos_rev, tan_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 3). D12 for the
periodic ops moved into "current design" (elementary and step functions: reverse ops; ieee 1788:
periodic reverse ops). the choices the plan left open, each the most conservative reading, are now
current design too:
* **a `c` holding `[-1, 1]` is not hulled** for sin and cos: every finite t is a solution, one
  piece, so `sin_rev([-1, 1])` is `(-inf, inf)` with no warning; D12 hulls only answers with too
  many pieces. tan has no such `c` (its poles), so `tan_rev((-inf, inf))` warns
* **the poles and ±inf are in no preimage**, part 1's rule for a point with no value, although the
  set op `tan` attains ±inf around a pole inside a piece: `tan_rev([inf])` is `∅`. a piece around a
  pole keeps it, since it is irrational and the enclosures of the two branches' ends overlap
* **the cap per piece of `x`**, as `steps.step` has it: earlier pieces stay exact, the piece that
  would pass 1000 pieces and every unbounded piece are hulled, one `HullWarning` per call; the
  hull stays inside its piece of `x`
* **the result is the union of every branch's enclosure, then `∩ x`**, so one branch more is listed
  on each side of a piece of `x`, for an `x` that starts inside a neighbour's rounding slack
* a `c` whose gaps are single irrational points (`[-1, 1)` for sin) counts as having gaps: a wide
  `x` is hulled with the warning even though listing would give the same set
* far out, where the doubles are coarser than a period (above about 1.8e16), neighbouring
  enclosures merge into one piece; still an enclosure
* **a proposed category gains rows**: six more `*Bin` vectors (12 with decorated copies) expect an
  end one or two doubles outside the tightest enclosure; ours is the tightest (arb). they are rows
  under "tighter than the vector", PROPOSED at part 1 and still awaiting the owner
* the 136 vectors: 124 match in both passes, 12 are those rows. itf1788 now 8465 vectors of 100
  ops; 1077 statements of 11 ops skipped (`powRev1`, `powRev2`, and M13g's)

### 2026-09-26 revision: M13e, second part (mul_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, part 2). `mul_rev` moved
into "current design" (elementary and step functions: reverse ops; ieee 1788: the pair rule). the
choices the plan left open, each the most conservative reading, are now current design too:
* **the library's own `*` defines it**: `t` is in `mul_rev(b, c, x)` iff `{t} * b` meets `c` and
  `t ∈ x`, so `0 * ±inf` (no value) contributes nothing: `mul_rev([inf], [0])` is `∅` and
  `mul_rev([1, inf], [3], [0, 10])` is `(0, 3]`, 0 excluded. 1788 has no infinite points, so no
  1788 vector sees the difference
* **rounding is the division's**: every finite end other than 0 is a quotient of an end of `c` by one
  of `b`, computed by `ops.div`, so it is exact for int and Fraction ends and rounded (to nearest,
  flags kept, or outward, a moved end open) exactly where the division would round it. the other
  candidate, `fma`'s rule (all ends rounded once if any operand has a finite float end), would
  make `mul_rev([3], (-2, 0.0))` differ from `(-2, 0.0) / 3` in its exact end `-2/3`
* **x defaults to `[-inf, inf]`**, as for the other reverse ops; no 1788 vector is touched, since
  the input rule never gives `c` an infinite point and ±inf join a preimage only through one
* **`mulRevToPair` is `mul_rev` under a pair rule** in the adapter, piece by piece (stricter than
  as a union, which the plan allowed); the one exported function serves all three 1788 ops
* the 539 vectors: all match in both passes but the 6 `[nai]` ones (generated rows under
  "decoration expectations", which M13g moves). no new category

### 2026-09-26 revision: M13e, first part (sqr_rev, abs_rev, pown_rev, cosh_rev), built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13e, in progress). D12's
reading for these four moved into "current design" (elementary and step functions: reverse ops;
ieee 1788: the reverse ops rule). the choices the plan left open, each the most conservative
reading, are now current design too:
* **`x` defaults to `[-inf, inf]` with f's own values at ±inf** (the plan's `x=REALS`), so the
  infinities are points of a preimage like any other: `pown_rev([0], -2)` is `[-inf] ∪ [inf]`,
  1788's `[empty]`. the 1788 vectors this touches are rows under "degenerate infinities"
* **a point where f has no value is in no preimage**: 0 for n < 0, where the library's `1/[0]` is
  empty, even though the set op `1/[-1, 1]` attains ±inf around it. so the result is pointwise,
  and `T ⊆ rev(f(T))` holds for every `T` without such points
* **an empty operand warns** (`EmptySetPropagationWarning`, off by default), as the functions do;
  an empty answer from non-empty operands does not warn: "no solution here" is a normal answer
* **the result class is `OutwardMultiInterval` if either operand is one**, as for the dunders;
  the functions are module-level (`multiinterval.sqr_rev`), not methods, as the plan's signatures say
* **the adapter compares the closed hulls** (the output rule unchanged): 1788's reverse op is the
  hull of the preimage by definition
* **a proposed new residual category, "tighter than the vector" (not yet approved)**: two
  `pownRev` vectors (`rev.itl:276`, `:277`) expect an end one double outside the tightest enclosure
  of `2 ** (1074/7)`; ours is the tightest (checked against arb). they are rows under it, and the
  owner decides whether the category stands
* the 476 vectors: 420 match in both passes, 52 (26 keys) are degenerate infinities of the unary
  `pownRev`, 4 (2 keys) the proposed category. itf1788 now 7790 vectors of 91 ops; 1752
  statements of 20 ops skipped (the rest of M13e, and M13g)

### 2026-09-26 revision: M13g, decorations, constructors and signals, done

M13g is done on branch `m13g` (2026-09-26); its record is v2-implementation-plan.md M13g, "done",
and parts 1 to 3 below are its steps. **D16 is now current design** ("ieee 1788" above): the core
`MultiInterval` stays undecorated, and 1788's decorations live in the wrapper `DecoratedInterval`,
with `Decoration` com/dac/def/trv, 1788's constructors (`text_to_interval`, `nums_to_interval` and
their decorated twins, `set_dec`) and propagation through every point function and set operation of
the core. 1788's signals are python's: `UndefinedOperationError` (a `ValueError`) raises,
`PossiblyUndefinedOperationWarning` (an `IntervalWarning`) would warn and is never emitted, since
the exact parser decides validity. no NaI and no `ill` (Q8): every statement that needs one is a
row under "no NaI: invalid input raises". every itf1788 statement of an M13g op is a vector (matching,
or a row), every decorated vector runs through the decorated type with its decoration checked, and
the only statements still skipped are the 19 reverse ops' (M13e), pinned by
`tests/itf1788/test_itf1788.py::test_only_the_reverse_ops_are_skipped`. **still open for the owner**
(each built as the conservative reading, flagged in the part entries): the PROPOSED category "exact
parsing decides validity" (7 rows); whether the 15 rows where an exact value past the doubles keeps
com (`_BOUNDED_EXACTLY`, 3; `PLAIN_ONLY`, 12) stay under "decoration expectations" or get a category
of their own; and `set_dec` demoting as 1788's `setDec` does rather than raising. the reverse ops
join the decorated type with M13e, all trv (`decorated.py::_trivial`)

### 2026-09-26 revision: M13g part 3, decoration propagation, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part
3") and "ieee 1788" above. `DecoratedInterval` now propagates 1788's decorations through every point
function of the core and decorates set operations trv; the adapter checks the decoration of every
decorated vector of those ops. the choices the plan left open, each the conservative reading, flagged:
* **a multi-piece box is decided on the set, not on the hull**: continuity on the set is continuity
  on each piece, the pieces being apart (the owner's task text asked for this reading)
* **an attained ±inf is outside every domain**: 1788's functions are functions of reals, so an
  operand holding ±inf as a point gets trv even where the core takes a limit (`exp([0, inf])` is trv,
  `exp([0, inf))` dac)
* **com needs the result bounded as returned**: the exact result for exact operands (so
  `[1,2]_com + [5,max]_com` stays com here, 12 plain-pass rows under "decoration expectations"), the
  rounded one for float operands (`OutwardMultiInterval` gives `[6.0, inf)`, dac, as 1788;
  `MultiInterval` rounds `2.0 + max` to nearest, max, and keeps com). the owner question of part 2
  (`_BOUNDED_EXACTLY`: this category, or an exact-values one) covers these rows too
* continuity at a point is relative to the op's domain (`sqrt` at 0, `acosh` at 1, `pow` at `x = 0`
  with `y > 0` are com), as the vectors require
* `%`, `//`, `divmod` and `round(ndigits)`, not in 1788, get the same rule from their definitions;
  `sign` and every step function at a jump point is dac at best
* booleans, numbers and relations stay off the wrapper (`.interval` first); a bare `MultiInterval`
  operand is a `TypeError`, a real number newDec's point
* the core's warnings reach the caller once; the decoration's own core calls are silenced

### 2026-09-26 revision: M13g part 2, the decorated type, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 2").
the decorated wrapper of D16 is `DecoratedInterval` (`multiinterval/decorated.py`, "ieee 1788" above),
named after M8's `DateTimeInterval`, with a `Decoration` enum and no `ill`. the choices the plan left
open, each the conservative reading, flagged:
* **the constructor is strict, `set_dec` is 1788's**: `DecoratedInterval(x, d)` raises
  `UndefinedOperationError` for a decoration that does not fit, as the literal `"[1,]_com"` does;
  `set_dec` demotes as 1788 defines it, with no signal (`libieeep1788_class.itl:283`-`288` expect
  `[empty]_trv` and `_dac`), and raises only for `ill`. raising for every unfitting `setDec` would
  have turned six matching vectors into rows needing a new category, for input 1788 calls valid
* **bounded is decided on the exact set**, so `[1.0E+400]_com` stays com: three vectors are rows
  under the existing "decoration expectations" (owner question: or the PROPOSED exact-parsing
  category, widened)
* a decoration name is exact lower case (`'COM'` raises): a python argument is not 1788 text
* **a core bug fixed**: `MultiInterval.is_finite` and `.finite` raised `OverflowError` on an exact end
  past the doubles (`MultiInterval(10**400)`), since `math.isfinite` converts to float; they compare
  with ±inf now (`tests/test_multi_interval.py::test_finiteness_of_an_exact_end_past_the_doubles`)
* not built: decorations propagated through the core's ops (1788's decorated arithmetic); every
  other op's vectors still drop their decorations

### 2026-09-26 revision: M13g part 1, signals and bare constructors, built

built and measured 2026-09-26 on branch `m13g`; details in v2-implementation-plan.md (M13g, "part 1").
the signals as the owner chose them (Q1): `UndefinedOperationError(ValueError)` raises,
`PossiblyUndefinedOperationWarning(IntervalWarning)` would warn and return. `multiinterval/literals.py`
reads 1788's interval literals, a syntax separate from `MultiInterval.parse`, **exactly**: a decimal
is the rational it spells, so validity is decided exactly and the warning is never emitted today.
`text_to_interval` and `nums_to_interval` give a bare `MultiInterval` with an infinite end open.
"no NaI: invalid input raises" is in `REASONS` (D16) with the 52 former `[nai]` rows and every
`isNaI`; **a new category, "exact parsing decides validity", is PROPOSED for the owner** (the 4
vectors expecting `PossiblyUndefinedOperation`, "ieee 1788" above). the decorated type is still to
come

### 2026-09-26 revision: M13d, power and the rest of the elementary functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13d). D11 moved into "current
design" (arithmetic: power; elementary and step functions: pow, hypot and the twelve functions;
ieee 1788: the counts). the choices D11 and the plan left open, each the most conservative
reading, are now current design too:
* **a Fraction with an integral value is pown**, as D11 says for a float: `A ** Fraction(4, 2)` is
  `A ** 2`. any `MultiInterval` exponent is pow, even a degenerate integral one, as D11 says
* **`0 ** y` for y <= 0 is outside the domain**, dropped with the `DomainClippedWarning` negative
  bases get, not an `IndeterminateResultWarning`: D11 names the domain as x > 0 plus x = 0 where
  y > 0. the one-sided limit `0 ** -1` = inf, which the M12 rule for a domain's end would allow, is
  not taken, since 1788's domain excludes those points outright; so `[0, 1] ** [-1]` is `[1, inf)`
  while pown's `[0, 1] ** -1` is `[1, inf]`
* **±inf in pow are points where the power has a limit** (`inf ** 2` = inf, `(1/2) ** inf` = 0),
  and `1 ** ±inf`, `inf ** 0` are indeterminate points with atan2's treatment: a box that is one of
  them contributes nothing and warns, a larger box keeps the values of its other points
* **the poles at 0 of cot, csc, coth, csch and an odd negative root follow `reciprocal`** (a
  one-sided limit at a piece's end, closed iff the piece holds 0; the point 0 alone empty with an
  `IndeterminateResultWarning`), the plan's "split a piece as tan's do" for the poles inside
* **acot is continuous**, `pi/2 - atan x` with values in (0, pi), fi_lib's reading (its 30 vectors
  are all on positive operands, where the two readings agree); `atan(1/x)`, with a jump at 0 and
  values in (-pi/2, pi/2], is the other convention
* **rootn(n) for every int n other than 0**, 1788's: n < 0 is the root of `1/x`; an even root's
  domain is x >= 0 with `rootn(0, -2n)` = inf at the domain's end; an odd negative root has the
  two-sided pole at 0; `rootn(x, 0)` is a `ValueError`, a non-int degree a `TypeError`
* **hypot is rounded once** from the exact sums of squares, like `fma`; its class is the receiver's
* the 1939 vectors of the 14 ops match in both passes with no divergence row. 41 of them end at or
  are the pole 0 of cot, csc, coth or csch: `[0]` is empty in both, and a piece ending at 0 has the
  same closed hull whether the infinity is attained or not. `0 ** y` for y <= 0 is outside both
  domains (`pow [0.0,0.0] [0.0,0.0] = [empty]`); no vector has `acoth` at ±1, and the input rule
  never gives a closed infinity. itf1788 now 7314 vectors of 83 ops; 2228 statements of 28 ops
  skipped (M13e, M13g)

### 2026-09-26 revision: M13f, cancellation, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13f). D13 moved into "current
design" (arithmetic, ieee 1788, testing), and its new residual category, "cancellation as a
Minkowski difference", is in the divergence table. the choices D13 and the plan left open, each
the most conservative reading, are now current design too:
* **float operands round outward in `OutwardMultiInterval`**, to the tightest enclosure of the exact
  `X`, and to nearest in `MultiInterval`, computed exactly and rounded once like `fma`. the vectors
  require it (`cancel.itl:218`, `:221`: 1788's answer is the hull of the exact difference) and it
  is that class's promise, every `x` that fits is in the result. inward rounding would certify
  `B + X ⊆ A` but break the outward pass of 16 matching vectors and the class's meaning; it is
  **open to the owner** as a separate method or type if a certified inner answer is wanted.
  today exact operands (a float as `Fraction(f)`) give it
* **an empty `B` gives `[-inf, inf]`, for `A = ∅` too**: every `X` has `∅ + X = ∅ ⊆ A`, so the
  largest is the whole line. 1788 answers `∅` for `cancelMinus [empty] [empty]` (and entire for a
  non-empty `A`, which matches); the two `[empty] [empty]` vectors are rows under the new category
  with their own reason. special-casing `∅ ⊖ ∅ = ∅` would contradict "the largest `X`"
* **the infinite points follow the library's `+`**: `inf + -inf` has no value, so `{inf} + [-inf]`
  is empty and `inf` fits any `A` when `B = [-inf]` (`[0, 1].cancel_minus([-inf])` = `[inf]`)
* **methods on the receiver's class**, as `fma`: `OutwardMultiInterval` only when `self` is one; a
  number is coerced to a point; no warning, not even for an empty operand
* the 242 vectors: 148 match in both passes; 94 (47 keys) are rows under the new category, 45 where
  1788 has no answer and 2 for `[empty] [empty]`. itf1788 now 5375 vectors of 69 ops, 4167
  statements of 42 ops skipped

### 2026-09-26 revision: M13c, the interval orders and the interior, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13c). D10 moved into "current
design" (comparisons, set operations and size, testing). the choices D10 and the plan left open,
each the most conservative reading, are now current design too:
* **"equal infinite ends count" exactly as 1788 writes it**: two starts at -inf, two ends at +inf.
  a start at +inf or an end at -inf (`[inf]`, `[-inf]`, points 1788 has no interval for) is not
  strictly less than itself; the looser "any equal infinite ends" would make `[inf]` strictly less
  than `[inf]`
* **`.interior` is the interior in the reals**: the infinite ends are opened too, so a closed end
  at ±inf drops out (`[5, inf]` → `(5, inf)`, `[inf]` → `∅`, `[-inf, inf]` → `(-inf, inf)`), the
  plan's "every end opened". the extended reals' interior would keep `(5, inf]`. degenerate
  pieces drop out; the class is kept (`OutwardMultiInterval`)
* **open or closed ends do not matter to the orders** (they are on the ends as values, the hull's),
  so `[0, 2]` is weakly but not strictly less than `[0, 2)`; the orders coerce a number to a
  point and raise `TypeError` on anything else, as `within` does
* **empty sets follow the vectors**: two empty sets are weakly and strictly less than each other,
  an empty and a non-empty one neither; `∅.within(B.interior)` is True
* the 184 vectors match with no rule and no listed row (12 generated `[nai]` rows); itf1788 now
  5133 vectors of 67 ops, 4409 statements of 44 ops skipped

### 2026-09-26 revision: M13b, the numeric functions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13b). D9 moved into "current
design" (set operations and size, ieee 1788, testing). the choices D9 and the plan left open, each
the most conservative reading, are now current design too:
* **methods, not properties**: `mid()`, `rad()`, `wid()`, `mag()`, `mig()`, `mid_rad()` (the plan's
  spelling; `inf` and `sup` stay properties)
* **"float" is the operand's kind**: any finite float end makes every number a float, rounded once;
  otherwise int or Fraction. the one exception is D9's ±max float for a half-bounded `mid`
* **`rad` is 1788's radius around the rounded midpoint**, not half the width rounded up, because the
  vectors require it (`rad [1, 1 + 3ulp]` = `2ulp`); `mag` rounds up and `mig` down (exact on
  double ends); both classes give the same numbers, the direction being the function's
* **edge cases no source covered**: a point, `[inf]` included, is its own midpoint with radius and
  width 0; an exact end past max float keeps the midpoint inside the hull; the empty set raises
  `ValueError` naming the missing quantity, with no warning
* **the adapter gained a float pass for the six ops** (`test_vector_float`, both classes, compared
  as they are): the first pass's exact operands would never exercise the library's own rounding
* the 167 vectors match, no divergence row beyond the 6 generated `[nai]` ones; itf1788 now 4949
  vectors of 64 ops, 4593 statements of 47 ops skipped

### 2026-09-26 revision: M13h, the reductions, built

built and measured 2026-09-26; details in v2-implementation-plan.md (M13h). no D row covered it, so
the choices below were made while building, each the most conservative reading of the plan, and
are now "current design" (arithmetic, ieee 1788, testing):
* **exported from `multiinterval`**: `sum_`, `sum_abs`, `sum_sqr`, `dot`, following M13e's plan for
  the reverse ops (`sum_` keeps its underscore so it never shadows the builtin)
* **the direction is a keyword-only string**, `rounding='nearest'` by default, `'down'`, `'up'`:
  no public API took a direction before, and `rounding.py`'s constants stay internal
* **the result is always a float**, also for int and Fraction operands (the plan's "rounded once",
  an exception to "int and Fraction are never rounded", since a reduction is 1788's float op)
* **no value raises**: a `nan` operand, `inf + -inf`, `0 * inf` raise `ValueError` where 1788
  answers `NaN`, as `nan` does in a constructor and as D9 decided for `mid` of the empty set; the
  adapter reads the error as `NaN`. no warning. open to the owner: returning `nan` instead (1788,
  ieee 754 and python's float `sum` do) would be a looser, compatible change later
* the 15 reduction vectors match with no divergence row; itf1788 now 4782 vectors of 58 ops, 4760
  statements of 53 ops skipped

### 2026-09-25 revision: M13a and M14's fuzz job and oracle built

built across 2026-09-25 and 26 (measured 2026-09-26), the first session M13 suggests; the details
and numbers are in v2-implementation-plan.md (M13a, M14). D14 and D15 moved into "current design"
(ieee 1788, testing):
* **vendored** all 19 files of oheim/ITF1788 at `b6ee1e2` and its three licence files, byte-exact,
  blob hashes checked; reading every header corrected D15: five files carry the all-permissive
  notice (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`)
* **the parser reads every statement** (9542): a strict tokenizer, brace-counted testcases,
  `NaN`, quoted text, `{...}` lists, `signal` clauses, two-value results. the old one ended a
  testcase at a list's `}` and silently dropped 11 statements of `libieeep1788_reduction.itl`
* **divergence keys drop decorations**, so the 18 old rows are 15 keys; the `[nai]` operands are
  rows generated in code under "decoration expectations" until M13g, each failing as stale once
  its vector matches. 4767 vectors, 49 keys, 0 unknown failures
* **fuzzing is a multiplier, not a profile's count**: a test's own `max_examples` overrides any
  profile, so under `HYPOTHESIS_PROFILE=fuzz` the conftest multiplies each test's own, once per
  function; the weekly `fuzz.yml` saves the example database even on failure. its first local run
  found `test_sound_float_identity_rounding[div]` red on mixed Fraction and float operands:
  python rounds `Fraction(1,3) / 2.75` twice, the library once, as "arithmetic" says it should. the
  fault was the test oracle's, which now rounds a mixed pair once (`tests/oracles.py::_once`)
* **the flint oracle requires correct rounding**, not a tolerance: no double may lie between an end
  and the value, decided by arb's proven comparisons, more bits when undecided; values arb cannot
  hold near the float range are compared through their logs. it found no bug

### 2026-09-25 revision: M13 and M14 planned (not built)

owner requests: vendor the whole itf1788 suite (the maintained fork, oheim/ITF1788 at `b6ee1e2`,
19 files) and implement every op in it, as M13; and fuzzing, as M14. the owner accepted every
recommended default, recorded as D9–D17 in v2-implementation-plan.md section 0 with the details.
none of this is "current design" yet: each point moves up into it when its sub-task is built.
in short:
* **D9** `mid`, `rad`, `wid` are of the hull; `mag`, `mig` of the set (`mig([-3,-2] ∪ [2,3])` = 2)
* **D10** `weakly_less`, `strictly_less` for 1788's `less`, `strictLess` (`<` stays pointwise); a
  `.interior` property, so 1788's `interior(A, B)` is `A.within(B.interior)`
* **D11** `**`: an integral exponent is pown as today; a non-integral or interval exponent is 1788's
  `pow`, negative bases dropped with `DomainClippedWarning` (so `MI(-3,1) ** MI(2)` = `[0, 1]`);
  `b ** A` through `__rpow__`; 3-argument `pow` dropped
* **D12** reverse ops with periodic answers: exact pieces up to 1000, else the hull + `HullWarning`
  (built 2026-09-26 at M13e; now in "current design", elementary and step functions, empties and
  warnings, ieee 1788)
* **D13** `cancel_minus` is the Minkowski difference (the largest `X` with `B + X ⊆ A`); where 1788
  answers "no answer" with entire, ours is a real set, under a new residual category
  "cancellation as a Minkowski difference"
* **D14** `python-flint` (Arb) is the independent oracle for the elementary functions, test-only
  (built 2026-09-26; now in "current design", testing)
* **D15** the fork's LGPL-2.1+ files (`mpfi`, `fi_lib`, `c-xsc`) are vendored unmodified as test
  data with their licence files; the wheel ships only `multiinterval/` (built 2026-09-26; now in
  "current design", ieee 1788. five files are all-permissive, not the two `ieee1788-*` alone)
* **D16** decorations, NaI and 1788's constructors in a separate decorated wrapper type, brought
  forward from "later"; the core stays undecorated. open: 1788's signals as warnings or exceptions
* **D17** M13 does not block 2.0.0, and the release is in no hurry

### 2026-09-25 revision: M12, the unblocked backlog built

the (b) items of the M11 backlog, built in one session by owner request ("build all the things that
are unblocked"); the choices made while building are recorded here and in
v2-implementation-plan.md (M12), and written into "current design" above:

* **functions do not go through the applicator**: an irrational value has no exact `Value` for the
  descriptor's `fn` to return, and sin/cos/tan split at irrational points. `functions.py` has its
  own small evaluator, as `modulo.py` does; it reuses `applicator.split_pieces` and `warn`
* **no libm**: every value comes from a pure-python, correctly rounded evaluator (`elementary.py`),
  so results are the same on every platform and a directed rounding is a true bound. this replaces
  "±1 ulp around trig is pragmatic, not rigorous — document"
* **an irrational value of an exact operand is its tightest float enclosure** (`sqrt([2])` is the
  open one-ulp piece), not a nearest float: an exact operand never loses its true value
* **a moved end is open, outward**: attainment is decided on exact values, so the ends that directed
  rounding moved are open; to nearest, flags stay conservative
* **the end of a function's domain is a point of it** where the one-sided limit exists
  (`log(0)` = -inf, `atanh(±1)` = ±inf); 1788 drops those points, and the 11 vectors where that
  shows are divergence rows (degenerate infinities)
* **atan2 on the negative x axis is pi** (there is no -0); `(0, 0)` and `(±inf, ±inf)` have no value
* **Allen relations stay cut-based** (`[1, 2]` and `[2, 3]` overlap): a new divergence category,
  "cut-based relations", for the 7 overlap vectors where 1788 says meets
* **the step functions share one engine and one cap** (`steps.py`); `modulo.floor` delegates to it
* names: `minimum`/`maximum` (numpy's names for the pointwise min and max; builtin `min` needs a
  bool from `<`), `round_ties_away` (1788's roundTiesToAway), `OutwardMultiInterval`

### 2026-09-23 revision: implementability review

a review of whether the plan could be built as written. the design stands; these holes would have
been built in, two of them past the planned tests. all are written into "current design".

* **corner-flag rule at infinity** (unsound): add/sub/mul are flat at ±inf, so "closed iff both
  corners closed" excluded attained infinities — `[inf] + (1, 2)` came out `∅`, `[1, inf] + [0, 1)`
  came out `[1, inf)`. infinite result endpoints now always go through the attainment predicate.
  the planned attainment tests ran "on int/Fraction only", and `inf` is a float, so they would not
  have seen it
* **mul splits at zero** (not sharp): `[-1, 1] * [inf]` hulled to `[-inf, inf]` although only
  `[-inf] ∪ [inf]` is attained. sound, so the fuzz passed, and endpoint checks cannot see an
  interior point. an interior-sharpness test is added
* **±inf is exact** whatever its python type, so the exact/float split and the rounding hook
  have a rule for it
* **`-` is arithmetic subtraction** (as v1, `multi_interval.py:1188`); set difference is a named
  method. the 2026-08 dunder list had given `-` to both
* **`==` does not coerce** scalars (hash contract)
* **relations return bool**; only pointwise comparisons return a `TruthSet`
* **warnings**: import-time filters use `append=True`; property tests that hit indeterminate cases
  opt out of the suite's warnings-as-errors
* **modulo with infinite operands** (D8): clip an infinite dividend, follow python for an infinite
  divisor
* **itf1788 output rule** hulls the expected value too
* **division** computed directly, not as `a * (1/b)` (double rounding on floats)
* implementation detail, in the implementation plan: `typing.NamedTuple` refuses a `__new__`
  override (`AttributeError: Cannot overwrite NamedTuple attribute __new__`, executed 2026-09-23 on
  python 3.13.15), so `Cut` normalizes in a subclass of the named tuple

### 2026-09-23 revision: D1–D6 settled

owner decisions on the review's open list (v2-implementation-plan.md section 0).

* **D1, closure at infinity**: flags propagate; an infinite endpoint is closed iff attained.
  "closure over attainable values and their limits" is gone: it made `1/(-1, 0)` = `[-inf, -1)`,
  which contradicted both `1/[1, inf)` = `(0, 1]` and the involution (`1/(1/(-1, 0))` came back
  `(-1, 0]`)
* **D2, indeterminate corners**: limit along the box, as reciprocal already did at zero. replaces
  "`[-inf, -1] * [0]` is the entire line". this is what forces D7's `∅` for `[0] * [inf]` and
  `[inf] - [inf]`; under the entire-line rule `∅` was allowed but not forced
* **D3**: exact division as Fraction, integral Fractions normalized to int
* **D4**: time layer deferred; v1's `time_interval.py` (and v1 `multi_interval.py` under it) stay
* **D5**: modulo for every sign combination is required before release, as its own milestone; not
  shipped Q1-only
* **D6**: `[a, inf]` literal — already the design, recorded as settled
* **v1 files** (owner, later the same day): archived to `archive/v1/`, never deleted, until v2
  works. this includes `time_interval.py`, which returns on top of the v2 class; the D4 bullet's
  "stay" means "stay in the archive"

### 2026-09-23 revision: degenerate indeterminate boxes return ∅

> **amended same day**: D1 and D2 were decided after this entry; see the entry above.

owner decision (D7 in v2-implementation-plan.md). a box that *is* an indeterminate point — `1/[0]`,
`[0] * [inf]`, `[inf] - [inf]`, `[0] / [0]` — returns `∅` and emits `IndeterminateResultWarning`,
where the 2026-09-22 design gave `[-inf] ∪ [inf]` or the entire line.

**why.** inclusion isotonicity (`A ⊆ B ⇒ f(A) ⊆ f(B)`) forces it: `[0]` is inside both `[-1, 0]` and
`[0, 1]`, so `1/[0] ⊆ [-inf, -1] ∩ [1, inf] = ∅`; `[0] * [inf] ⊆ [0] * [5, inf] ∩ [0, 1] * [inf]`,
empty under the sharp corner rule. solvers (bisection, forward-backward contraction) need isotone
ops; it matches 1788's empty and removes the `1/[0]` row from the itf1788 divergence table.

**cost.** `1/(1/[inf])` = `∅`, and `1/x` round-trips only for sets with no degenerate piece at `0`,
`inf` or `-inf` (`[0] ∪ [1, 2]` → `[1/2, 1]` → `[1, 2]`). the direction tag under "later" stays the
recovery path.

**separately.** `f(A ∪ B) == f(A) ∪ f(B)` is false for reciprocal under the direction-from-the-piece
rule with *any* value of `1/[0]`: `A = [-1, 0)`, `B = [0]` gives `1/(A ∪ B)` = `[-inf, -1]` but
`1/A ∪ 1/B` = `(-inf, -1]` (with D1's flag propagation). that law is only `⊇` for reciprocal/div.

### 2026-09-22 revision: signed zero dropped

**how it got in.** v1's reciprocal returns the whole line for anything touching zero → fix by
splitting at zero → the pieces must reach infinity, so ±inf became points → `1/0` evaluated at an
endpoint needs a sign to pick `±inf` → `-0` imported from ieee 754 as a distinct point → six-state and
four-state enums, then three boundaries at zero, glued/strict modes, `same_set()`, `config.py`.

**why it is out.** the fourth step assumes `1/x` is evaluated at a scalar. an interval library
evaluates sets, and the set already carries the direction: `1/[-1, 0]` is `[-inf, -1]` because the
piece is negative, which is also Hickey's and 1788's answer. the sign is informative only for a
degenerate `[0]`, where it records how the zero was produced — dependency tracking, which interval
arithmetic gives up everywhere else. outward rounding never manufactures a signed zero from underflow
on a non-degenerate interval either, since the bound facing away from zero rounds away. as a position
in the order it also cost real correctness: `[-1, 0]` necessarily contains `-0`, so `1/[-1, 0]` grew a
phantom `+inf` under "signs pick the branch"; `(-0.0, 1)` contained `0.0`; ints cannot spell `-0`;
`[-0]`, `[+0]` and `[-0, 0]` were `==` and hash-equal yet gave different reciprocals.

> **superseded 2026-09-23**: `1/[0]` is now `∅`, so `1/(1/[inf])` = `∅` and the involution also
> fails at a degenerate zero piece. see the 2026-09-23 entry.

**what is lost.** `1/(1/[inf])` is `[-inf] ∪ [inf]` instead of `[inf]` — Kahan's involution argument,
the one real case for the sign bit. it breaks at both infinities symmetrically and nowhere else, and
is recoverable later as metadata (see "later").

> **superseded 2026-09-23**: "now" column, rows `1/[0]` and `[0] * [inf]` — both are `∅` + warn

| case          | pure math      | ieee 754 scalar | python         | ieee 1788                     | cset (Hickey)         | 2026-08 plan         | now                     |
|---------------|----------------|-----------------|----------------|-------------------------------|-----------------------|----------------------|-------------------------|
| `1/[0]`       | undefined      | ±inf by sign    | raises         | empty                         | `{-inf, +inf}`        | `[+inf]` or `[-inf]` | `[-inf] ∪ [inf]` + warn |
| `1/[-1, 0]`   | undefined at 0 | n/a             | n/a            | `[-inf, -1]`, -inf unattained | `[-inf, -1]`          | `[-inf,-1] ∪ [+inf]` | `[-inf, -1]`            |
| `1/[-1, 1]`   |                |                 |                | entire                        | `[-inf,-1] ∪ [1,inf]` | same                 | same                    |
| `[0] * [inf]` | undefined      | nan             | nan            | not expressible               | entire                | `[0, inf]` by signs  | entire + warn           |
| `-0.0` input  | same number    | distinct, `==`  | distinct, `==` | absent                        | absent                | distinct point       | normalized to 0         |

**other changes in the same revision**: the cut side is an `IntEnum`, not "epsilon", and there is no
third "exact" side; the truth set has four states so empty operands raise instead of being vacuously
TRUE; `size` in points rather than half-points, and the v1 ray bug noted; rounding is a hook with an
identity default enabled by type; no context managers of any kind; 1788 decorations deferred to the
solver; the itf1788 adapter gets an input rule for infinity; mixins replaced by kernel functions plus
one class file; relations defined on cuts.

### 2025-12 wishlist and negative-zero enums

* the range will be the affine extended real numbers, meaning support for ±inf along with negative zero, i.e.:
  `[-inf] + (-inf, 0) + [-0, 0] + (0, inf) + [inf]`
* divide by zero is supported, and there will be warnings for indeterminism
* newton's method solver as a test
* generalized function applicator as long as its continuous and differentiable
* forward mode autodiff
* numpy compat via https://data-apis.org/array-api/latest/ or `__array_ufunc__` or https://numpy.org/doc/stable/user/basics.interoperability.html
* consider ieee 1788 compatible decorations, although multi intervals actually support a bit more so not sure if it matters

### negative zero

> **superseded 2026-09-22**: there is no signed zero in v2. see "current design" and the 2026-09-22 log entry.

* note that while we accept that `-0 == 0`, `-0` will be stored as a separate value, and they interact as follows:
    * `(..., 0)` & `[-0]` will merge to form `(..., -0]`
    * `[0]` & `(-0, ...)` will merge to form `[0, ...)`
    * `(..., -0)` & `[0]` will merge to form `(..., 0]` (but this is somewhat questionable)
    * `[-0]` & `(0, ...)` will **not** merge
    * `[-0]` & `[0]` will merge to form `[-0, 0]`
* we can think of `-0` as a hyperreal number of the form `(0, -1ε)`, while plain `0` is `(0, 0ε)`
* we need `-0` to handle infinities:
    * `(0, inf]/0 == [-inf, -0)/-0 == inf`
    * `[-inf, -0)/0 == (0, inf]/-0 == -inf`
    * `0/0 == -0/-0 == inf * 0 == -inf * -0 == [0, inf]` (raises `IndeterminateResultWarning`)
    * `-0/0 == 0/-0 == -inf * 0 == inf * -0 == [-inf, -0]`(raises `IndeterminateResultWarning`)

maybe this enum, but this introduces a `(-0, ...)` that just feels wrong:

* -3 open end neg zero
* -2 open end
* -1 closed neg zero
* 0 closed
* 1 open start neg zero
* 2 open start

or a simplified version that might be more intuitive, but this introduces a weird symmetry break at 
`[-0]|(-0,...)` that cannot exist and can never merge, and `(...,-0)|[0]` that merges but introduces a 
spurious `-0`, so taking the reciprocal now produces a `-inf` that should not have been there:

* -2 open end
* -1 closed neg zero
* 0 closed
* 2 open start

### update

found that there's an ieee spec that does something similar

### v2 consolidated decisions (2026-08-16)

> **partly superseded 2026-09-22**: cuts, applicator, retirements and the 1788-as-adapter principle survive; signed zero, zero modes, config.py, mixins and the `measure` name do not. each subsection is marked.

supersedes the enum discussion above (kept for history). derivation: chat sessions 2026-08-15/16.

#### representation: cuts (boundaries), not endpoints

* store each bound as the *boundary where the set stops*, not "point + open/closed flag"
* every real x has two boundaries: just below = `(x, -1)`, just above = `(x, +1)`; a piece is a pair of cuts
    * `[a, b]` = `(a,-1),(b,+1)` — `(a, b)` = `(a,+1),(b,-1)` — `[a, b)` = `(a,-1),(b,-1)` — `[x]` = `(x,-1),(x,+1)`
    * empty iff start_cut >= end_cut, so `[1,1)` and `(1,1)` normalize to empty; `[2,1]` is still ValueError
* sort = plain lexicographic tuple compare; a normalized multi-interval's flat cut list is *strictly* increasing
* **there is no merge rule**: pieces merge iff `next.start_cut <= current.end_cut`
    * `[1,2] | (2,3]` tiles exactly (both cuts `(2,+1)`) -> merge; `[1,2) | (2,3]` has `(2,-1) < (2,+1)` -> the point 2 is missing -> gap
    * no `diff <= k`, no adjacency table, no special cases
* complement reuses the same cut tokens with start/end roles swapped (no epsilon negation); mirror = `(-v, -s)`
* the hyperreal story is the *documentation*, not the implementation: `(v,+1)` is the boundary at v+ε; only order matters, so tuples suffice
* v1's `{-1, 0, 1}` epsilon had a redundant state: closed-start-at-x and closed-end-at-x are different boundaries; cuts never conflate them

#### signed zero: keep it

> **superseded**: dropped 2026-09-22 (see log entry for why).

* -0 is direction info on a zero boundary, not a user-facing number. IEEE hands us the sign bit anyway (`-1.0 * 0.0 == -0.0`), and it's exactly what keeps reciprocal sharp under our closure-of-limits semantics (`1/[-5,-0] = [-inf, -0.2]`)
* zero has *three* boundaries instead of two: `(0,-1)` below -0, `(0,0)` between -0 and +0, `(0,+1)` above +0. side 0 is only legal at value 0 (one constructor check)
* the merge rules from the section above now fall out of plain comparison, nothing written down:
    * `(..., 0) | [-0]` -> merges -> `(..., -0]`
    * `[0] | (-0, ...)` -> merges -> `[0, ...)` (in cuts `(-0, ...)` *is* `[+0, ...)` — same boundary, the "feels wrong" state never exists)
    * `[-0] | (0, ...)` -> does NOT merge (+0 genuinely missing)
    * `[-0] | [+0]` -> merges -> the full zero
    * deviation from the old plan: `(..., -0) | [0]` does NOT merge (-0 genuinely missing). this was the case marked "somewhat questionable" above — merging it is what manufactured the spurious -0 and the phantom -inf under reciprocal
* the seam at zero is irreducible: -0 and +0 are two *adjacent* points in an otherwise dense order — an order-theoretic fact no encoding can hide. its entire footprint is one extra token at value 0. dropping -0 entirely is the only way to a perfectly homogeneous line (and would cost sharp division)

#### zero gluing: float semantics by default

> **superseded**: no zero modes, no context managers.

model the UX on how python/IEEE already treat -0.0 (equal, same hash, sign preserved through arithmetic, visible only via copysign / division):

* `zero_mode='glued'` (default): `==`/`hash`/membership/merging treat -0 == +0; sign tags on zero endpoints are preserved until an actual cross-zero merge consumes them, so `1/[-5,-0]` stays sharp even in glued mode. gluing only ever *widens* by the other zero point -> sound, never wrong
* `zero_mode='strict'`: no gluing; `[-0] | (0,...)` stays two pieces; for solvers and branch cuts (1/x, log, sqrt). toggle via context manager (same pattern as the warnings design)
* repr always faithful (prints -0 when present, like python floats); str likewise

#### comparisons

> **amended**: the truth set has four states (empty operands raise); relations are defined on cuts, not sup/inf.

* `< <= > >=` return a tri-state truth set (TRUE / FALSE / BOTH, since "every a op every b" can be both); `__bool__` raises on BOTH (numpy precedent), so `if a < b:` either works or fails loudly — never guesses
* `==` and `__hash__` are structural set equality (required for dicts/tests); pointwise equality is a method, and is BOTH for any non-degenerate `a == a` (document this, it surprises everyone)
* explicit `sort_key` (lex on cuts) for structural ordering; `sorted()` raising on ambiguous intervals is a feature
* relation vocabulary (the adjacent/adjoining/intersecting/overlapping todo): named set-level predicates — disjoint, adjoins (end cut == start cut), overlaps, contains, within, equals, before (sup A < inf B), after — plus certainly_/possibly_ modal variants
* allen's 13 relations are only JEPD for *contiguous* intervals; expose `allen(a, b)` restricted to contiguous pieces (raise otherwise, or caller passes hulls explicitly). optional: per-piece relation matrix, or the set of relations holding between any piece pair (allen's algebra natively reasons over relation sets). note cuts make the taxonomy *finer* than classical allen: tiling-without-sharing (`[1,2) meets [2,3]`) vs sharing-one-point (`[1,2] ∩ [2,3] = {2}`) are distinguishable

#### division semantics vs ieee 1788 (they are not wrong, just different)

> **amended**: direction comes from the set, not a sign bit; `1/[0]` = `[-inf] ∪ [inf]`.
> **superseded 2026-09-23**: `1/[0]` = `∅` + warning, and results are attained values with flags propagating through infinity, not a closure over limits (see the 2026-09-23 entries).

* ours: closure over attainable values/limits (cset-flavored, cf. hickey/van emden paper in README todo) -> zero denominators contribute their limit infinities, signed zeros pick the branch
* 1788: division is the *inverse relation of multiplication* over the reals ({z : x = z·y}); y=0 contributes no z because z·0=1 has no solution -> `1/[-5,0] = [-inf,-0.2]` with decoration dropped to `trv` (partiality is recorded, not ignored). solvers get the split result via `mulRevToPair` (two-interval extended division)
* consequence: itf1788 vectors will disagree on division-by-zero cases *by design*; keep a documented divergence table rather than chasing exact matches
* naming: **ieee 1788-2015** = the standard (1788.1-2017 = simplified subset); **itf1788** = community test framework for it (test files in a small DSL, "itl"; used by IntervalArithmetic.jl et al)

#### housekeeping

> **amended**: `measure` renamed again to `size` (in points, not half-points); rounding is a hook with identity default, enabled by type.

* rename `cardinality` -> `measure`: it's a lex-graded size (rays, open length, closed endpoint count), not cardinality. the ω/ε gloss is fine intuition; we only compare these tuples, never do arithmetic on them
* delete `INFINITY_IS_NOT_FINITE`: affine extended reals, `[inf]` degenerate allowed; cuts encode `[a, inf]` vs `[a, inf)` naturally, and a mutable global that changes set semantics is a footgun
* v2 core type immutable + hashable; incremental building via a small builder (bisect-insert is 82x faster than re-sort per compare.py)
* directed rounding moves FIRST, before newton/autodiff: `math.nextafter` outward rounding behind a rounding-policy hook, gmpy2/mpfr later as tight mode. every arithmetic op touches endpoint computation — retrofitting means touching everything twice. (libm is not correctly rounded, so ±1 ulp around trig is pragmatic-not-rigorous; document)
* floordiv: enumerate integer points below a size cap, else return hull + warning — never silently drop openness
* testing: random fuzz vs sampling oracle for soundness (`op(x,y) ∈ op(A,B)` for sampled x,y) + attainment checks for closure — the pattern already validated by the modulo v3 work; itf1788 as conformance suite with the divergence table above

#### ieee 1788 conformance: test adapter, not a runtime flag

> **amended**: the adapter also needs an input rule (1788 unbounded → our open-at-inf).

* 1788 is further away than a flag: closed intervals only (no open bounds exist in the standard), connected only (no multi-intervals — everything hulls), ±inf never attained, decorations everywhere. a semantics flag would be `INFINITY_IS_NOT_FINITE` again ×10, and every flag multiplies the test matrix (`zero_mode × 1788_mode × rounding`)
* instead: a conformance adapter in the *test suite* — parse itf1788 vectors, run our ops, closed-hull the result, compare, consult a small divergence table. hulling absorbs most divergence automatically (`[1,2]/[-1,1]`: 1788 says entire, we say `[-inf,-1] ∪ [1,inf]`, hull = entire → match). residual table: empty-vs-degenerate-infinity division cases, domain-clipped functions, decoration expectations
* principle: flags change what existing objects mean; wrapper types add meanings. if real conformance is ever needed, it's a thin wrapper class in `ieee1788.py`, never a mode on MultiInterval

#### package architecture

> **superseded**: kernel functions over cut tuples plus one class file; no mixins, no config.py, no zero_mode.

split by domain, layered so imports only point downward (no cycles):

    multiinterval/
        __init__.py      public API assembly; constants (EMPTY, REALS, ...)
        errors.py        warning/exception classes                 -- layer 0
        config.py        zero_mode + rounding policy               -- layer 0
        cuts.py          cut encoding: (value, side) helpers,      -- layer 1
                         zero-sign logic, negation, ordering
        core.py          MultiInterval: immutable flat cut tuple,  -- layer 2
                         normalization sweep, eq/hash/bool,
                         Builder for incremental construction
        sets.py          union/intersection/difference/complement, -- layer 3
                         membership, subset, adjoins/overlaps
        relations.py     TruthSet tri-state, certainly_/possibly_, -- layer 3
                         sort_key, allen() (contiguous only)
        rounding.py      outward-rounding eval (nextafter now,     -- layer 3
                         gmpy2/mpfr later) — the single hook
        apply.py         generic applicator (see below)            -- layer 4
        arithmetic.py    + - * / reciprocal as op descriptors      -- layer 5
        modulo.py        v3 far-edge mod/divmod/floordiv           -- layer 5
        fmt.py           str/repr grammar + parser                 -- layer 5
        functions.py     sqrt/log/exp/trig via apply         (later)
        autodiff.py, solver.py, numpy_compat.py             (later)
        time_interval/   DateTime/TimeDelta layers over core (port)
    tests/
        oracles.py       sampling + attainment oracles — promote out of
                         modulo_v3_prototype; they're the library-wide
                         test pattern, not a modulo tool
        itf1788/         vector runner + divergence table
        test_<module>.py

* one public class assembled from mixins: `sets.py` / `relations.py` / `arithmetic.py` each export a mixin, `core.py` composes `MultiInterval(SetOps, Relations, Arithmetic, Base)`. domain split without circular imports, dunders stay on one discoverable type
* **refinement of the zero-gluing section above**: `zero_mode` is a *construction-time* policy, not an ambient switch — eq/hash must not change when a context flag flips or dicts corrupt. normalization glues (or not) under the mode active at construction; after that the object is just data. `==`/`hash` are ALWAYS glued (float precedent: `0.0 == -0.0` unconditionally); strict structural comparison is a named method (`same_set()`). the "context manager" from above scopes which policy new objects are built with
* `_consistency_check` goes under `if __debug__:` (stripped by `-O`) — v1 runs an O(n) scan at the top of nearly every public method; that's the real hot cost, not corner counts. correctness lives in the oracle tests instead

#### generic applicator: keep it, but shape-then-attainment

* keep the generic endpoint applicator — not to save code, but because it's the ONE place where directed rounding, closure decisions, and domain splitting get woven into endpoint computation. per-op code would reimplement all three per op, forever. 4-vs-2 evals is noise next to interpreter overhead
* v1's epsilon propagation through corners is UNSOUND, not just inelegant: `[0,1] * (2,3)` — min corner `0*2=0` gets eps `0 or 1` = open, v1 returns `(0,3)`, but 0 is attained (`0 * 2.5`); true result `[0,3)`. the flat spot at zero makes the result independent of the open operand. same disease the modulo v3 work diagnosed (604 closure faults from epsilon propagation), here it excludes attainable values → containment violation
* restructure around the modulo lesson:
    1. locations first, all endpoints treated closed (corner min/max — correct for coordinatewise-monotone and bilinear ops)
    2. closure per endpoint separately: strictly-monotone ops → corner-flag rule is provably fine (fast path, zero cost); flat-spot ops (mul at 0, pow, min/max-like) → attainment check
    3. driven by a small op descriptor per operation: monotonicity directions (gives add/sub a 2-corner fast path for free), flat-spot predicate, rounded-eval pair from `rounding.py`

#### more v1 retirements

* empty operands in arithmetic propagate: `A / ∅ = ∅` (vacuous union over no divisors), same for `∅ / B`, `A + ∅`, etc. — this matches ieee 1788 (empty in → empty out) and the other interval libraries, and mirrors nan propagation in floats, so silent propagation is the standard-aligned default. **note (owner):** silent empty propagation is effectively implicit nan/null propagation and can be annoying to debug later — so emit a dedicated `EmptySetPropagationWarning` on division (maybe all arithmetic) with a default `'ignore'` filter installed at import; solver code opts into `warnings.simplefilter('error', EmptySetPropagationWarning)` as a tripwire. exceptions stay reserved for malformed construction (`[2,1]`)
* parsing moves out of `merge()` into `fmt.py`, regexes compiled at module level (hot-path compile in v1)
* one `_coerce(other)` in core instead of per-method isinstance ladders
* the `inplace=` dual API disappears with immutability — large chunk of v1 surface gone for free
* retire `interval.py` (the alternative debug implementation) — the sampling oracle does that job better, and a second implementation is a maintenance tax
* keep `__getitem__` slicing (`x[0:5]` as restriction); `in` = scalar membership + documented subset alias; subset stays a named method since `<=` is the tri-state comparator
* `__bool__` = non-empty, explicitly (set precedent) — distinct from tri-state comparisons, which raise on ambiguity
