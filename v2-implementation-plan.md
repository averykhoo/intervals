# `MultiInterval` v2 implementation plan (sketch, 2026-09-23)

companion to `v2-plan.md`. that file says *what*; this one says *in what order*, with an exit
criterion per milestone. review findings that needed an owner decision are in section 0; D1–D7 are settled or deferred as
of 2026-09-23 and written into `v2-plan.md`'s "current design"; D8 was settled 2026-09-24. D9–D17
were settled 2026-09-25 for M13 and M14: they go into "current design" as each is built, and until
then they are in `v2-plan.md`'s decision log. D14 (the flint oracle) and D15 (vendoring) were built
with M13a and M14's first two items (2026-09-26) and are in "current design" now.

**open work and open questions live in `HANDOFF.md`** (since 2026-09-26): ranked items, questions for
the owner, loose ends, session log. this file keeps the spec (what to build, exits) and the records.

## 0. decisions (D1–D8 from the 2026-09-23 reviews; D9–D17 for M13 and M14, 2026-09-25)

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
| D19 | **decided in the build 2026-09-27 (the session's defaults), open for the owner: `HANDOFF.md` Q11.** the solver stack's first part (M15): (a) `intervals/autodiff.py` (`Dual`, `derivative`) and `intervals/solver.py` (`newton`, `Root`) are public and exported from `intervals`, not newton as a test only; (b) newton's step runs only where `f` is proved C¹ by decorations (dac or better on the value and the derivative), else the piece is pruned and bisected; (c) the step is `mul_rev`, never `/` (D7); (d) one variable; (e) `tol=1e-10` absolute, `max_steps=10_000` | as built | M15 |

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
  pull request. it runs the gate on Python 3.11 to 3.14 and each exhaustive harness as its own job:
  `tests.exhaustive_ops` exact, `--float` and `--sabotage`, and `tests.exhaustive_modulo`.
  under GitHub Actions hypothesis loads its built-in `ci` profile (derandomized, no deadline).
  `tests/conftest.py` (M14, 2026-09-26) loads nothing unless `HYPOTHESIS_PROFILE` is set, so the
  gate still gets hypothesis's own choice (`default` locally, `ci` under Actions); set to `fuzz` it
  runs every hypothesis test randomized at `FUZZ_MULTIPLIER` (default 10; 100 until 2026-09-27) times its examples,
  which `.github/workflows/fuzz.yml` does weekly and on `workflow_dispatch`, never on push (M14).
  first run 2026-09-25 at `d232b78` (run 36091651163), all 8 jobs green:
  the gate took 71-94 s on each python, and on the runners the exhaustive jobs took 21 s
  (sabotage), 5 min (float), 6½ min (exact) and 12½ min (modulo). each harness exits nonzero on a
  failure. M13e/g's first run, 2026-09-27 at `dbec908` (run 36293351201): the 4 gate jobs each
  `1 failed, 22165 passed`, `tests/test_reverse.py::test_mul_rev_float_operands` at a `b` whose
  quotient passes the largest double, where to nearest gives the piece `[inf]` (as designed) and
  the oracle `_widened` skipped infinite ends; the local gate missed it (randomized profile; `ci`
  is derandomized). fixed in the oracle, the example pinned (`d7e46c2`); run 36305984327 at
  `a1d45a9` all 8 green, the gate 272-285 s on each python
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

### M8 `time_interval.py` (1½ days) — deferred (D4); not part of this plan's schedule
* starts from `archive/v1/time_interval.py`, ported onto the v2 class with whatever tweaks that
  needs; the archived copy stays until the port works
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
* **v1 surface with no v2 row in section 4** (a: port or record as gone): open, `HANDOFF.md` Q6
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
* **breadth where fuzz is thin**: `tests/test_outward.py` has 1 `@given`, `tests/test_steps.py` 3,
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
    * the exit's green GitHub run: not yet, `HANDOFF.md` M14-run
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

## 4. v1 → v2 surface map (for the M10 README and for not forgetting anything)

"gone" below means gone from v2's API; the v1 code itself is archived at M10, not deleted.

| v1 | v2 |
|---|---|
| `merge(*args, n_overlaps=)` classmethod, parses strings | `MultiInterval.parse(str)`; `union(*)`; `kernel.overlap_count(cuts_list, n)` |
| `update/intersection_update/...`, `add/discard/pop/remove/clear` | gone (immutable); `Builder` for incremental construction |
| `merge_adjacent(distance=)`, `expand(d, inplace=)` | `expand(d)` pure; no distance rule anywhere (cuts make it exact) |
| `cardinality -> (half_rays, length, half_points)` | `size -> Size(rays, length, points)` |
| `overlapping(or_adjacent=)`, `overlaps` | `relations.overlaps/adjoins`, `A & B` for the overlap itself |
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
| `<<`, `>>` | to port (owner 2026-09-26, `HANDOFF.md` Q6-shift); the meaning on real sets is chosen when built |
| `random_multi_interval` | to-do, undecided (owner 2026-09-26, `HANDOFF.md` Q6); the tests use hypothesis strategies instead |
| public `apply()` | to-do, undecided (owner 2026-09-26, `HANDOFF.md` Q6); `applicator` and `OpDescriptor` are not exported |
