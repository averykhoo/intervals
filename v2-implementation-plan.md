# `MultiInterval` v2 implementation plan (sketch, 2026-09-23)

companion to `v2-plan.md`. that file says *what*; this one says *in what order*, with an exit
criterion per milestone. review findings that needed an owner decision are in section 0; D1–D7 are settled or deferred as
of 2026-09-23 and written into `v2-plan.md`'s "current design"; D8 was settled 2026-09-24.

## 0. decisions (from the 2026-09-23 reviews; D1–D8 settled or deferred)

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
  reproduction)
* gate: `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q` from the repo root. add
  `pyproject.toml` (package metadata, `[tool.pytest.ini_options] testpaths = ["tests"]` and
  `pythonpath = ["."]` so the v1 modules at the root and the `intervals/` package import without
  relying on `python -m` putting cwd on `sys.path`, and a
  `filterwarnings` entry turning the library's own warnings into errors inside the suite once
  the warning classes exist)
* CI (added on `v2` 2026-09-25, by owner request): `.github/workflows/ci.yml` runs on every push and
  pull request. it runs the gate on Python 3.11 to 3.14 and each exhaustive harness as its own job:
  `tests.exhaustive_ops` exact, `--float` and `--sabotage`, and `tests.exhaustive_modulo`.
  under GitHub Actions hypothesis loads its built-in `ci` profile (derandomized, no deadline), so
  the suite needs no conftest. first run 2026-09-25 at `d232b78` (run 36091651163), all 8 jobs green:
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
  is green on `v2` (section 1)
* **M8, the time layer** (a: whether and when; 1½ days): see M8. D4 is still open, (a) recommended
* **functions**, **rounding functions**, **the other 1788 ops** and **outward float rounding**, the
  four (b) items: built at M12 (below), done 2026-09-25
* **reverse ops** (a): `mulRevToPair` and friends, for the reverse-op itf1788 files and for a solver.
  owner 2026-09-25: build, as M13e
* **power beyond int exponents** (a: scope; owner 2026-09-25: build 1788's `pow`, as M13d): `A ** 0.5`, `A ** B`, `2 ** A` are TypeError today
  (`intervals/multi_interval.py::MultiInterval.__pow__`); v1 took an interval exponent on a
  positive base. 3-argument `pow(A, n, m)` (v1: integers only; old README "allow interval modulo
  for `__pow__()`")
* **the 1788 ops still missing** (a: whether to add them): `less`, `strictLess`, `interior` (weak
  and strict interval orders, not v2's pointwise comparisons) and `mid`, `rad`, `wid`, `mag`, `mig`
  (for a multi-interval, of the hull or per piece?). their vectors are counted and skipped in
  `tests/itf1788/test_itf1788.py::SKIPPED`; `isNaI` has no counterpart. owner 2026-09-25: add
  every one, as M13b, M13c and M13g
* **solver stack** (a; `v2-plan.md` "later (not in v2.0)"): the direction tag on a degenerate zero
  piece (only if a solver needs `1/(1/[inf])` back), a decorated type (com/dac/def/trv/ill) and an
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

### M13 full itf1788: every vector vendored, every op built (open, added 2026-09-25)

owner request 2026-09-25: "implement all these ops and get all these tests vendored and passing".
this settles M11's "whether to add them" for every 1788 op; the (a) tags left below are on *how*,
not whether. sub-tasks M13a to M13h; M13a goes first, the rest are independent of each other
* **the full set**: the vendored files are 7 of the 12 at nehmeier/ITF1788 `e0e0d7e` (that repo's
  HEAD). the maintained fork, **oheim/ITF1788 at `b6ee1e24d209c289f99a68ddc357839935799eae`**
  (2018-09-22), has 19 files: nehmeier's 12 (renamed `libieeep1788_*.itl`, the `_tests` dropped),
  plus `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` (vectors converted from those libraries' suites),
  `ieee1788-constructors.itl`, `ieee1788-exceptions.itl`, `libieeep1788_class.itl` and
  `libieeep1788_reduction.itl`. its versions of the 7 vendored files are a superset in bare
  intervals: they fix decorations (`acos [entire]_def` becomes `_dac`), decorate bare `[empty]`s
  and add `[nai]`, `NaN` and empty cases
* **measured 2026-09-25**, the fork's 19 files downloaded to a temp dir and run through this repo's
  unchanged adapter (`tests/itf1788/test_itf1788.py::run`, `::run_outward`), ops as in `OPS`:
    * **already passing, only needing vendoring**: `fi_lib` 687/687 (outward 687/687), `mpfi`
      980/980 (900/900), `c-xsc` 126/126 (85/85), `atan2` and `libieeep1788_set` all
    * failing: 40 `[nai]` operands (bool, elem; no counterpart), 4 `isMember NaN ...` (the parser
      reads `NaN` as a word; the library itself answers `nan in A` false, as 1788 does), and
      `atanh [1.0,1.0]_def = [empty]_trv`, the known atanh row under a different key, because
      `DIVERGENCES` is keyed on the text with decorations
    * **4764 statements of 57 ops not implemented**, grouped into the sub-tasks below
* **M13a vendoring and the adapter** (b; licence settled by owner 2026-09-25: the recommendation
  below, as written). the three library-derived files are
  **LGPL-2.1-or-later** (Inria / Karlsruhe / Wuppertal, converted by O. Heimlich), the two
  `ieee1788-*` files carry an all-permissive notice, the rest Apache 2.0; this repo has no licence
  of its own. recommended: vendor all 19 unmodified into `tests/itf1788/`, replacing nehmeier's 7
  (one source, one pin), with the fork's `LICENSE`, `NOTICE` and `COPYING.LESSER`; the wheel ships
  only `intervals/`, so no test file is distributed with the library. then in the adapter: key
  `DIVERGENCES` on the text with decorations stripped; parse `NaN`, quoted strings
  (`b-textToInterval "[1, 2]"`), `signal <Name>` clauses, and two-interval results
  (`mulRevToPair ... = [empty] [empty]`); every op's `NaI` input either maps (M13g) or is a
  divergence row. exit: the 1793 vectors above in the gate, 0 unknown failures, the new files in
  `test_parser_drops_nothing` and `test_every_file_is_used`
* **M13b numeric ops** (a: of the hull or per piece; hull recommended, since 1788's answer is the
  hull's and a per-piece form can be a separate method): `mid` 36, `rad` 19, `wid` 27, `mag` 27,
  `mig` 33, `midRad` 25 statements. `mid` and `rad` round to nearest and outward as 1788 specifies
* **M13c interval orders** (a: names, since `<` and `<=` are already pointwise and return a
  `TruthSet`): `less` 88, `strictLess` 32, `interior` 64. for a multi-interval `less` is on the
  hull's ends, `interior` is `A ⊆ int(B)` and generalises as it is
* **M13d power and the rest of the elementary functions**, all correctly rounded in pure python
  like `intervals/elementary.py`: `pow` (real exponent, domain `x > 0`, or `x = 0` with `y > 0`)
  1431; `expm1` 38, `logp1` 37, `cbrt` 10, `rootn` 3, `hypot` 17, `csc` 109, `sec` 109, `cot` 49,
  `acot` 30, `coth` 46, `acoth` 30, `csch` 16, `sech` 14. (a) what `A ** 0.5`, `A ** B` and `2 ** A`
  mean (M11 "power beyond int exponents"; `**` with an int is `pown` today), and 3-argument `pow`
* **M13e reverse ops** (a: signatures; `sqr_rev(c, x=REALS)` recommended, the `*Bin` vectors being
  the two-argument form): `powRev1` 429, `powRev2` 375, `mulRevToPair` 347, `pownRev` 285,
  `mulRev` 182, `mulRevTen` 10, `sqrRev` 20, `absRev` 18, `sinRev` 12, `cosRev` 12, `tanRev` 10,
  `coshRev` 10, and the `*Bin` forms (`pownRevBin` 73, `cosRevBin` 42, `sinRevBin` 40,
  `absRevBin` 38, `sqrRevBin` 22, `tanRevBin` 20, `coshRevBin` 10). a multi-interval holds
  `mulRevToPair`'s two pieces as one value, so the adapter compares the pair as a union.
  `sinRev`/`cosRev`/`tanRev` return infinitely many pieces over an unbounded `x`: (a) the hull
  with a `HullWarning`, as the step functions do past 1000 values
* **M13f cancellation**: `cancelPlus` 116, `cancelMinus` 126. (a) meaning on a multi-interval;
  1788 defines it for connected operands only
* **M13g decorations, NaI, constructors and signals** (settled by owner 2026-09-25: a separate
  decorated wrapper type, the M11 solver stack's, built now rather than with the solver. the core
  class stays undecorated, so `v2-plan.md` "ieee 1788" still holds): `b-textToInterval` 91, `d-textToInterval` 91,
  `b-numsToInterval` 10, `d-numsToInterval` 9, `setDec` 22, `newDec` 13, `intervalPart` 15,
  `decorationPart` 6, `isNaI` 16, the 40 `[nai]` operands of implemented ops, and a decoration
  check on every decorated vector, which the adapter drops today. `ieee1788-exceptions.itl` expects
  signals (`UndefinedOperation`, `PossiblyUndefinedOperation`, `IntvlPartOfNaI`): (a) map them to
  `IntervalWarning` subclasses or exceptions
* **M13h reductions**: `sum_nearest`, `sum_abs_nearest`, `sum_sqr_nearest`, `dot_nearest`, 1 each.
  correctly rounded sums of float vectors (exact through `Fraction`), point ops, not interval ops
* every sub-task: its ops' vectors pass in both passes (plain and outward) or are divergence rows
  with a reason from the plan's categories; a new category needs an owner decision and a line in
  `v2-plan.md` "ieee 1788"; its ops get the M14 properties the day they land; sabotage per
  section 2
* exit for M13: **no statement of the 19 files is skipped.** `SKIPPED` is empty and a test asserts
  it, so a file that gains an op cannot quietly add skips. `v2-plan.md` "ieee 1788" and the README
  counts re-measured, dated. (a) whether M13 blocks the 2.0.0 release

### M14 fuzzing (open, added 2026-09-25)

owner request 2026-09-25: "it would be great if we had fuzzing eg hypothesis". hypothesis is
already in the gate: 81 `@given` tests across 13 files (counted 2026-09-25), 30 to 300 examples
each. the gaps are depth, independence and breadth:
* **no run explores new inputs in CI.** GitHub Actions loads hypothesis's `ci` profile, which is
  derandomized: every CI run replays the same examples, so new inputs are only ever tried by a
  local gate run. add a `fuzz` profile (randomized, `max_examples` about 100 times the gate's, no
  deadline) and a CI job on a schedule and `workflow_dispatch`, **not** in the gate and not on every
  push. cache the example database between runs, upload it with the log on a failure, and pin each
  failure found as an `@example` in the gate test that found it. (b)
* **an independent oracle for the functions.** `tests/test_elementary.py` checks the 19 functions
  against `decimal` (and taylor series in decimal for the trig functions) on 60 random points each.
  add a hypothesis-driven differential against an arbitrary-precision library that shares no code
  with ours: the true value (a ball at about 200 bits) inside our enclosure, and each end of ours
  within one ulp outside it (sharpness). (a) the test-only dependency: `mpmath` (pure python) or
  `python-flint` (arb, rigorous balls; recommended for the rigour)
* **breadth where fuzz is thin**: `tests/test_outward.py` has 1 `@given`, `tests/test_steps.py` 3,
  `tests/test_fmt.py` 1 (the parse/format round trip), `tests/test_applicator.py` 1.
  `tests/test_extreme_floats.py` covers add, sub, mul, div, reciprocal, neg, abs and pow only:
  extend it to the functions, `minimum`/`maximum`/`fma`, `%` and `//`, and the outward class
* **every M13 op as it lands**: soundness (`f(x) ∈ f(A)` at sampled `x ∈ A`, exact and float, under
  identity and outward rounding), isotonicity, interior sharpness, and for a reverse op its defining
  property (`x ∈ rev(C, X)` iff `x ∈ X` and `f(x) ∈ C`, at sampled points), with the op's itf1788
  vectors as `@example`s
* exit: the fuzz job exists and has run green once, its example count and time recorded here with a
  date; every new property sabotaged once and seen red (section 2)

## 3. order and parallelism

M1 → M2 → M3 → M4 → M5 → M6 → {M7a → M7b, M9} → M10, all done by 2026-09-25; M8 deferred; M11 is
the backlog, and M12 built its (b) items the same day. M13 (full itf1788) and M14 (fuzzing) are
open: M13a first, then M13b to M13h in any order, each with its M14 properties; M14's CI job and
oracle do not wait for M13. M4 depends on M3 (the class's
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
| `**` with an interval exponent on a positive base; `pow(A, n, m)` on integers | open (M11): int exponents only |
| `<<`, `>>` | open (M11): port or record as gone |
| `random_multi_interval` | open (M11): the tests use hypothesis strategies instead |
