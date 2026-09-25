# `MultiInterval` v2 implementation plan (sketch, 2026-09-23)

companion to `v2-plan.md`. that file says *what*; this one says *in what order*, with an exit
criterion per milestone. review findings that needed an owner decision are in section 0; D1–D7 are settled or deferred as
of 2026-09-23 and written into `v2-plan.md`'s "current design"; D8 was settled 2026-09-24. D9–D17
were settled 2026-09-25 for M13 and M14: they go into "current design" as each is built, and until
then they are in `v2-plan.md`'s decision log. D14 (the flint oracle) and D15 (vendoring) were built
with M13a and M14's first two items (2026-09-26) and are in "current design" now.

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
| D9 | **decided 2026-09-25 by owner: recommended default.** 1788's numeric ops on a multi-interval. `mid`, `rad`, `wid` (and `midRad`) are **of the hull**: a midpoint outside the set (`mid([0,1] ∪ [9,10])` = 5) is still a valid bisection point, a per-piece form would return a tuple, and `size.length` already gives the width without the gaps. `mag` and `mig` are **of the set**, as `sup` and `inf` of `{abs(x) : x ∈ A}`: `mig([-3,-2] ∪ [2,3])` = 2, where the hull would give 0; on a connected set the two readings agree. unbounded operands follow 1788 (`mid` of entire is 0, of a half-bounded set ±max float; `rad` and `wid` are inf); the empty set raises `ValueError`, as `.inf` does today, and the adapter maps it to 1788's `NaN` | hull for mid/rad/wid, set for mag/mig | M13b |
| D10 | **decided 2026-09-25 by owner: recommended default.** names for 1788's interval orders, since `<` and `<=` are pointwise and return a `TruthSet` (`MI(1,3) < MI(2,4)` is `BOTH`): `A.weakly_less(B)` for `less` (both of A's ends ≤ B's; 1788's own wording, "weakly less than"), `A.strictly_less(B)` for `strictLess`. `interior` is not a method: a new property `B.interior` (the set with every end opened, a set operation in its own right) and the existing `A.within(B.interior)` | `weakly_less`, `strictly_less`, `.interior` | M13c |
| D11 | **decided 2026-09-25 by owner: recommended default.** power. an `int` exponent, or a float with an integral value, is `pown` as today, like python's scalars (`(-3.0) ** 2.0` = 9.0; `MI(-3,1) ** 2.0` = `[0, 9]`). a non-integral float or a `MultiInterval` exponent is 1788's `pow`: domain `x > 0`, plus `x = 0` where `y > 0` (`0 ** y` = 0); negative bases are dropped with `DomainClippedWarning`. so `MI(-3,1) ** MI(2)` = `[0, 1]`, not `[0, 9]`: an interval exponent means `pow`, never `pown`. `b ** A` for a scalar base is `MultiInterval(b) ** A` through `__rpow__`. exact where the value is rational, as `log` is. 3-argument `pow(A, n, m)` is dropped (not 1788; v1 had it on integers only) | pown for integral, else 1788 pow; 3-arg dropped | M13d |
| D12 | **decided 2026-09-25 by owner: recommended default.** a reverse op whose answer has infinitely many or very many pieces (`sinRev`, `cosRev`, `tanRev` and their `*Bin` forms over an unbounded or wide `x`): the exact pieces up to the step functions' cap of 1000, past it their hull with a `HullWarning`, the rule `steps.py` already follows. a bounded `x` gets the exact union (`sinRev([0.5, 1], [0, 20])` has 4 pieces), which 1788 cannot give. ends are irrational, so each is its tightest float enclosure, open | exact to 1000 pieces, else hull + warning | M13e |
| D13 | **decided 2026-09-25 by owner: recommended default.** `cancelMinus(A, B)` is the **Minkowski difference**, the largest `X` with `B + X ⊆ A`, which is defined on any multi-intervals; `cancelPlus(A, B)` is `cancelMinus(A, -B)`. for connected `A` and `B` it is exactly 1788's answer whenever 1788 has one (`[a1 - b1, a2 - b2]` when `wid A ≥ wid B`). where 1788 has no answer it returns entire as a "no answer" signal, and ours is a real set: `cancelMinus([-inf,-1], [-1,5])` = `(-inf, -6]`, and `∅` when nothing fits. those vectors are divergence rows under a **new residual category, "cancellation as a Minkowski difference"**, approved with this decision | Minkowski difference; new divergence category | M13f |
| D14 | **decided 2026-09-25 by owner: recommended default.** the independent oracle for the elementary functions is **`python-flint`** (Arb: every result is a ball proven to contain the true value), a test-only dependency in the `[test]` extra, installed into the `intervals` env and in CI. `mpmath` was the alternative (pure python, but its values carry no proven bound). python-flint 0.9.0 has Windows wheels for 3.10 and later (abi3), 3.13 and 3.14 included (checked on PyPI 2026-09-25) | python-flint | M14 |
| D15 | **decided 2026-09-25 by owner: recommended default.** licence. `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` are LGPL-2.1-or-later, the two `ieee1788-*.itl` files carry an all-permissive notice, the rest Apache 2.0, and this repo has no licence of its own. vendor all 19 files of oheim/ITF1788 at `b6ee1e2` unmodified into `tests/itf1788/`, replacing nehmeier's 7, with the fork's `LICENSE`, `NOTICE` and `COPYING.LESSER` beside them. the wheel ships only `intervals/`, so no test file is distributed with the library. **corrected 2026-09-26 at M13a**, from every file's header: five files carry the all-permissive notice, not two (`ieee1788-constructors`, `ieee1788-exceptions`, `atan2`, `abs_rev`, `pow_rev`); the eleven `libieeep1788_*` are Apache 2.0 | vendor unmodified, with the licence files | M13a |
| D16 | **decided 2026-09-25 by owner: recommended default.** decorations (com/dac/def/trv/ill), NaI and 1788's constructors go in a **separate decorated wrapper type**: the solver stack's (M11), brought forward. the core `MultiInterval` stays undecorated, so `v2-plan.md` "ieee 1788" ("decorations are not in the core") holds. still open, for M13g: whether 1788's signals (`UndefinedOperation`, `PossiblyUndefinedOperation`, `IntvlPartOfNaI`) become `IntervalWarning` subclasses or exceptions | wrapper type; signals open | M13g |
| D17 | **decided 2026-09-25 by owner**: M13 does **not** block the 2.0.0 release, and there is no hurry to release either ("I have zero users and this is a yak shaving pet project"). M13 only adds methods and a type and gives a meaning to exponents that raise `TypeError` today, so nothing that works now changes | release whenever; not blocked | — |

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
  runs every hypothesis test randomized at `FUZZ_MULTIPLIER` (default 100) times its examples,
  which `.github/workflows/fuzz.yml` does weekly and on `workflow_dispatch`, never on push (M14).
  first run 2026-09-25 at `d232b78` (run 36091651163), all 8 jobs green:
  the gate took 71-94 s on each python, and on the runners the exhaustive jobs took 21 s
  (sabotage), 5 min (float), 6½ min (exact) and 12½ min (modulo). each harness exits nonzero on a
  failure
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
* **release** (a, then c): merge `v2` into `master` (`v2` pushed 2026-09-25), `version =
  "2.0.0"` in `pyproject.toml`, tag. D5's blocker (M7b) is met. CI exists since 2026-09-25 and
  is green on `v2` (section 1). owner 2026-09-25 (D17): no hurry, and M13 does not block it
* **M8, the time layer** (a: whether and when; 1½ days): see M8. D4 is still open, (a) recommended
* **functions**, **rounding functions**, **the other 1788 ops** and **outward float rounding**, the
  four (b) items: built at M12 (below), done 2026-09-25
* **reverse ops** (a): `mulRevToPair` and friends, for the reverse-op itf1788 files and for a solver.
  owner 2026-09-25: build, as M13e
* **power beyond int exponents** (a: scope; owner 2026-09-25: build 1788's `pow`, as M13d): `A ** 0.5`, `A ** B`, `2 ** A` are TypeError today
  (`intervals/multi_interval.py::MultiInterval.__pow__`); v1 took an interval exponent on a
  positive base. 3-argument `pow(A, n, m)` (v1: integers only; old README "allow interval modulo
  for `__pow__()`"). settled by D11: integral exponents stay pown, others are 1788 pow, and
  3-argument `pow` is dropped
* **the 1788 ops still missing** (a: whether to add them; settled, see below): `less`, `strictLess`, `interior` (weak
  and strict interval orders, not v2's pointwise comparisons) and `mid`, `rad`, `wid`, `mag`, `mig`
  (for a multi-interval, of the hull or per piece?). their vectors are counted and skipped in
  `tests/itf1788/test_itf1788.py::SKIPPED`; `isNaI` has no counterpart. owner 2026-09-25: add
  every one, as M13b, M13c and M13g (D9, D10, D16)
* **solver stack** (a; `v2-plan.md` "later (not in v2.0)"): the direction tag on a degenerate zero
  piece (only if a solver needs `1/(1/[inf])` back), a decorated type (com/dac/def/trv/ill; brought
  forward to M13g by D16) and an
  optional thin `ieee1788.py`, forward-mode autodiff, Newton's method as a test (b: the functions
  exist since M12), numpy interop (array API vs `__array_ufunc__`; today `__array_ufunc__ = None`), the
  optional per-piece Allen matrix
* **v1 surface with no v2 row in section 4** (a: port or record as gone): `<<` / `>>`,
  `random_multi_interval`, a public `apply()` (the applicator and `OpDescriptor` are not exported
  from `intervals`). the rows are added to section 4 as "open (M11)"
* **smaller** (c): the old README's reading list (arxiv 1111.0167) and its "redo the modulo
  illustrations" item, if still wanted
* **archive deletion** (a): `archive/v1/` goes only by owner decision, after release and M8
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

### M13 full itf1788: every vector vendored, every op built (open, added 2026-09-25; M13a done 2026-09-26)

owner request 2026-09-25: "implement all these ops and get all these tests vendored and passing".
this settles M11's "whether to add them" for every 1788 op, and D9–D17 (section 0) settle how;
the one question still open is D16's signals, for M13g. sub-tasks M13a to M13h: **M13a goes
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

**M13b numeric ops** (D9). `mid` 36, `rad` 19, `wid` 27, `mag` 27, `mig` 33, `midRad` 25
statements
* methods `mid()`, `rad()`, `wid()`, `mag()`, `mig()`, `mid_rad()`. `mid`, `rad`, `wid` of the
  hull; `mag`, `mig` of the set (`sup` and `inf` of the absolute values). exact operands give exact
  values; float results round as 1788 specifies (`mid` to nearest, `rad` and `wid` up) and the
  adapter's number rule compares them
* unbounded: `mid` of entire is 0, of a half-bounded set ±max float; `rad` and `wid` are inf.
  empty raises `ValueError`, like `.inf`; the adapter maps that to `NaN`

**M13c interval orders** (D10). `less` 88, `strictLess` 32, `interior` 64
* `A.weakly_less(B)`: `inf A ≤ inf B` and `sup A ≤ sup B`; `A.strictly_less(B)`: both strict,
  except that equal infinite ends count, as in 1788. on the ends, so the hull's, for a multi-interval
* `B.interior`: a property, the set with every end opened (the topological interior); `interior`
  in the vectors is `A.within(B.interior)`. 1788's unbounded ends arrive open at inf through the
  input rule, which is what makes `interior [1, infinity] [0, infinity]` true

**M13d power and the rest of the elementary functions** (D11). every value correctly rounded in
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

**M13e reverse ops** (D12). 1955 statements
* a new module `intervals/reverse.py`, the functions exported from `intervals`, each taking the
  constraint first and the domain `x` last, defaulting to the whole line: `sqr_rev(c, x=REALS)`,
  `abs_rev(c, x=REALS)`, `pown_rev(c, n, x=REALS)`, `sin_rev`, `cos_rev`, `tan_rev`, `cosh_rev`
  (all `(c, x=REALS)`), `mul_rev(b, c, x=REALS)`, `pow_rev1(b, c, x=REALS)` (the bases x with
  `x ** y ∈ c` for some `y ∈ b`), `pow_rev2(a, c, y=REALS)` (the exponents). the `*Bin` vectors are
  the two-argument call and `mulRevTen` is `mul_rev` with `x` given
* each result is `{x ∈ X : f(x) ∈ C}` (for the binary ones, `∃` over the other operand) as an exact
  multi-interval: `mulRevToPair`'s two intervals are one value here, compared by the adapter as a
  union (our closed pieces against the pair's). ends that are irrational are tightest float
  enclosures, open
* periodic answers per D12: exact pieces up to 1000, past that their hull with `HullWarning`

**M13f cancellation** (D13). `cancelPlus` 116, `cancelMinus` 126
* `A.cancel_minus(B)`: the largest `X` with `B + X ⊆ A` (the Minkowski difference);
  `A.cancel_plus(B)` is `A.cancel_minus(-B)`. for connected operands with `wid A ≥ wid B` it is
  `[a1 - b1, a2 - b2]`, 1788's answer
* where 1788 returns entire as "no answer" (A narrower than B, an unbounded operand) ours is a real
  set, often `∅`: those rows get the new residual category **"cancellation as a Minkowski
  difference"**, added to `REASONS` and to `v2-plan.md` "ieee 1788" when this lands

**M13g decorations, NaI, constructors and signals** (D16). 273 statements, plus the 40 `[nai]`
operands of implemented ops and a decoration check on every decorated vector
* a decorated wrapper type around a `MultiInterval` (name chosen when built, recorded in the
  decision log): a decoration per 1788 (com/dac/def/trv/ill), propagated through every op the core
  has, and NaI as the `ill` value. the core class stays undecorated
* 1788 text and number constructors: `b-textToInterval` 91 and `b-numsToInterval` 10 give a bare
  `MultiInterval` (1788's input rule: an infinite end is open), `d-textToInterval` 91 and
  `d-numsToInterval` 9 the wrapper. a parser for 1788's text syntax, separate from
  `MultiInterval.parse`, whose syntax is ours. `setDec` 22, `newDec` 13, `intervalPart` 15,
  `decorationPart` 6, `isNaI` 16 on the wrapper
* the adapter stops dropping decorations: a decorated vector runs through the wrapper and its
  expected decoration is checked
* **open, ask the owner before building**: 1788's signals (`UndefinedOperation`,
  `PossiblyUndefinedOperation`, `IntvlPartOfNaI`, from `ieee1788-exceptions.itl` and the
  constructors' `signal` clauses): `IntervalWarning` subclasses or exceptions

**M13h reductions**. `sum_nearest`, `sum_abs_nearest`, `sum_sqr_nearest`, `dot_nearest`, 1 each
as counted 2026-09-25 by the old parser, which saw only the first statement of each of the file's
4 testcases; M13a's parser reads the 11 it dropped (2026-09-26), so recount when this is built
* `intervals/reductions.py`: `sum_`, `sum_abs`, `sum_sqr`, `dot` over sequences of numbers, the
  exact value through `Fraction` then rounded once, to nearest by default. point ops, not interval
  ops, so no M14 properties beyond a random differential against `Fraction` arithmetic

**every sub-task**
* its ops' vectors pass in both passes (plain and, if interval-valued, outward) or are divergence
  rows with a reason from `REASONS`; a new category needs an owner decision (D13's is the only one
  approved so far) and a line in `v2-plan.md` "ieee 1788"
* its ops get the M14 properties the day they land, sabotage per section 2
* the D rows it implements move into `v2-plan.md` "current design", the README's feature list and
  the `ieee 1788` counts are re-measured and dated, and this section records what was built, as
  M12's does

**exit for M13: no statement of the 19 files is skipped.** `SKIPPED` is empty and a test asserts
it, so a file that gains an op cannot quietly add skips

**suggested sessions** (a guide, not a rule; each ends with a green gate and a commit): (1) M13a
with M14's fuzz job and oracle; (2) M13b, M13c, M13f and M13h, the small ones; (3) M13d; (4) M13e;
(5) M13g, starting with the signals question

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
    * **still owed for the exit: the workflow has never run on GitHub** (nothing is pushed), so no
      green `workflow_dispatch` run, example count or time is recorded yet
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
* still open in M14: the breadth items (`tests/test_extreme_floats.py` not yet extended) and every
  M13 op's properties as it lands

## 3. order and parallelism

M1 → M2 → M3 → M4 → M5 → M6 → {M7a → M7b, M9} → M10, all done by 2026-09-25; M8 deferred; M11 is
the backlog, and M12 built its (b) items the same day. M13 (full itf1788) and M14 (fuzzing) are
open: M13a is done (2026-09-26), then M13b to M13h in any order, each with its M14 properties;
M14's fuzz job and flint oracle are built (2026-09-26), the job's first green GitHub run still
owed. M4 depends on M3 (the class's
`parse`, `__str__` and `__repr__` come from `fmt`); M7a and M9 are independent after M6. total ≈ 12
working days (the per-milestone sum without M8) plus the M7b session. the first internally usable
point is after M5 (set algebra, formatting, comparisons); arithmetic lands at M6; release needs M7b.

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
| `**` with an interval exponent on a positive base; `pow(A, n, m)` on integers | int exponents only; interval and real exponents planned as 1788 `pow` (M13d, D11); `pow(A, n, m)` dropped (D11) |
| `<<`, `>>` | open (M11): port or record as gone |
| `random_multi_interval` | open (M11): the tests use hypothesis strategies instead |
