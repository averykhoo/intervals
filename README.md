# intervals

*A GLORIOUS EXERCISE IN YAK-SHAVING*

`MultiInterval`: a finite union of disjoint intervals over the affine extended reals, each piece open
or closed at either end, with set algebra, pointwise comparisons and arithmetic that returns exactly
the set of values attained.

```python
>>> from intervals import MultiInterval as MI
>>> x = MI(0, 1, end_closed=False) | MI(2, 3, start_closed=False)
>>> x
MultiInterval.parse('{ [0, 1) , (2, 3] }')
>>> print(1 / MI(-2, 2))           # split at zero, the gap is kept
{ [-inf, -1/2] , [1/2, inf] }
>>> print(MI(1) / MI(3))           # int and Fraction stay exact
[1/3]
>>> print(MI(0, 10) // 4)          # floor of the exact quotient, enumerated
{ [0] , [1] , [2] }
>>> print(MI(1, 3) < MI(2, 4))     # comparisons are pointwise
BOTH
>>> MI(1, 3).strictly_less(MI(2, 4))   # 1788's interval order, on the ends
True
>>> print(MI(0, 1).interior)           # every end opened
(0, 1)
>>> x.size
Size(rays=0, length=2, points=0)
>>> y = MI(-3, -2) | MI(2, 3)
>>> y.mid(), y.wid(), y.mig()      # mid and wid of the hull, mig of the set
(0, 6, 2)
>>> print(MI(0, 10).cancel_minus(MI(1, 3)))   # the largest X with [1, 3] + X inside [0, 10]
[-1, 7]
>>> print(MI(0, 3).sqrt())           # sqrt(3) is irrational: its enclosure's upper end, open
[0, 1.7320508075688774)
>>> print(MI(1, 8).log(2))           # exact where the value is rational
[0, 3]
>>> print(MI(1, 2).tan())            # a pole inside the piece: both sides, both infinities
{ [-inf, -2.185039863261519) , (1.557407724654902, inf] }
>>> from intervals import sqr_rev, mul_rev, sin_rev
>>> print(sqr_rev(MI(1, 4)))         # reverse ops: the t with t ** 2 in [1, 4], not 1788's hull
{ [-2, -1] , [1, 2] }
>>> print(mul_rev(MI(-1, 1), MI(1, 2)))   # the t with t * y in [1, 2] for some y in [-1, 1]
{ (-inf, -1] , [1, inf) }
>>> print(sin_rev(MI(0), MI(-1, 7)))  # the t in [-1, 7] with sin t = 0: 0 exact, pi and 2 pi enclosed
{ [0] , (3.141592653589793, 3.1415926535897936) , (6.283185307179586, 6.283185307179587) }
>>> import math
>>> print(math.floor(MI(-1.5, 1.5)))
{ [-2.0] , [-1.0] , [0.0] , [1.0] }
>>> from intervals import OutwardMultiInterval as OMI
>>> print(OMI(0.1) + 0.2)            # outward rounding: the exact sum is strictly between
(0.3, 0.30000000000000004)
>>> from intervals import text_to_interval, text_to_decorated_interval, DecoratedInterval
>>> print(text_to_interval('[0.1, infinity]'))   # 1788's literals, read exactly; an infinite end open
[1/10, inf)
>>> text_to_interval('[2, 1]')                   # 1788's UndefinedOperation raises
Traceback (most recent call last):
    ...
intervals.errors.UndefinedOperationError: invalid 1788 interval literal '[2, 1]': the lower bound exceeds the upper
>>> d = text_to_decorated_interval('[1, 4]_com')
>>> print(d.sqrt())                              # decorations propagate as 1788's do
[1, 2]_com
>>> print(d / DecoratedInterval(MI(-1, 1)))      # 1/0 is outside the domain: trv
{ [-inf, -1] , [1, inf] }_trv

```

## what it does

* **values**: int and Fraction are exact and never rounded; float endpoints go through a rounding
  hook (identity by default). `-inf` and `inf` are ordinary points, so `[1, inf]` and `[1, inf)` are
  different sets, and `[inf]` is a legal degenerate interval
* **set algebra**: `| & ^ ~`, `difference()`, `issubset()`, `in`, slicing `x[a:b]` (restricts to
  `[a, b]`), `hull`, `interior` (every end opened), `expand()`, `size` (rays, length, isolated
  points, ordered lexicographically)
* **relations**: `< <= > >=` and `eq_pointwise()` return a `TruthSet` (`TRUE`, `FALSE`, `BOTH`,
  empty); `before after adjoins overlaps contains within` return bool; `allen()` gives the Allen
  relation; `weakly_less()` and `strictly_less()` are 1788's interval orders, on the ends (the
  hull's), and return bool. `==` is structural and `MultiInterval` is hashable and immutable
* **arithmetic**: `+ - * /`, `reciprocal()`, `abs`, `**`, `%`, `//`, `divmod`,
  `minimum()`, `maximum()`, `fma()`, for every sign combination including zero-crossing and
  infinite operands. a result is
  the set of values attained: an infinite endpoint is closed iff it is attained, a pole at a closed
  zero attains the infinity of its piece's sign, and a box that *is* an indeterminate point
  (`1/[0]`, `[0]*[inf]`, `[inf]-[inf]`) is empty with an `IndeterminateResultWarning`
* **power**: a number exponent with an integral value is 1788's pown, over every base
  (`MI(-3, 1) ** 2` is `[0, 9]`); any other real exponent, and every `MultiInterval` one, is 1788's
  pow, over the bases x > 0 and x = 0 where y > 0, the rest dropped with a `DomainClippedWarning`
  (`MI(-3, 1) ** MI(2)` is `[0, 1]`). `2 ** A` is `MI(2) ** A`; 3-argument `pow` is refused
* **cancellation**: `A.cancel_minus(B)` is the Minkowski difference, the largest `X` with
  `B + X ⊆ A`, for any two sets (`∅` when nothing fits); `A.cancel_plus(B)` is
  `A.cancel_minus(-B)`. 1788's `cancelMinus`/`cancelPlus` where 1788 has an answer, a real set
  where it answers entire as "no answer". exact for exact operands, outward an enclosure of `X`
* **functions**: `sqrt`, `exp`, `exp2`, `exp10`, `expm1`, `log` (any base), `log2`, `log10`,
  `log1p`, `cbrt`, `rootn(n)`, `hypot`, `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`,
  `atan`, `acot`, `atan2`, `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, `asinh`, `acosh`,
  `atanh`, `acoth`, as methods; a pole inside a piece gives both infinities, as `1/x` does. values are correctly rounded by a pure-python evaluator (no libm), so they are the same on
  every platform; an irrational value of an exact operand is its tightest float enclosure
* **step functions**: `floor()`, `ceil()`, `trunc()`, `round(ndigits)`, `round_ties_away()`,
  `sign()`, and `math.floor/ceil/trunc` and `round()` on a set: the values attained, listed up to
  1000 of them, else their hull with a `HullWarning`
* **numbers of a set** (1788's numeric functions): `mid()`, `rad()`, `wid()` and `mid_rad()` of the
  hull, `mag()` and `mig()` of the set. exact for an exact operand; for a float one rounded as 1788
  specifies (`mid` to nearest, `rad`, `wid` and `mag` up, `mig` down), in both classes alike
* **reductions**: `sum_()`, `sum_abs()`, `sum_sqr()`, `dot()` over sequences of numbers (1788's
  reductions): the exact value, rounded once to a float, to nearest by default or
  `rounding='down'` / `'up'`, so the order of the operands never matters
* **reverse ops** (1788's reverse-mode functions): `sqr_rev(c, x)`, `abs_rev(c, x)`,
  `pown_rev(c, n, x)`, `cosh_rev(c, x)` are the set `{t ∈ x : f(t) ∈ c}` as an exact union, not
  1788's hull (`sqr_rev(MI(1, 4))` is `[-2, -1] ∪ [1, 2]`); `x` defaults to `[-inf, inf]`, and an
  irrational end is its tightest float enclosure, open. their 476 ITF1788 vectors run through the
  adapter below (M13e, 2026-09-26)
* **reverse multiplication**: `mul_rev(b, c, x)` is `{t ∈ x : t * y ∈ c for some y ∈ b}`, the
  values that solve `t * b ∋ c`, as an exact union (`mul_rev(MI(-1, 1), MI(1, 2))` is
  `(-inf, -1] ∪ [1, inf)`, where 1788's `mulRev` gives entire and `mulRevToPair` the two pieces);
  `0 * inf` has no value, as in `*`. its 539 ITF1788 vectors run through the adapter (M13e,
  2026-09-26)
* **periodic reverse ops**: `sin_rev(c, x)`, `cos_rev(c, x)`, `tan_rev(c, x)` are `{t ∈ x : f(t) ∈
  c}`, the exact pieces over a bounded `x` (`sin_rev(MI(Fraction(1, 2), 1), MI(0, 20))` has 4);
  past 1000 pieces, or over an unbounded `x` (the default), their hull with a `HullWarning`. ±inf
  and tan's poles have no value, so they are in no preimage. their 136 ITF1788 vectors run through
  the adapter (M13e, 2026-09-26)
* **power reverse ops**: `pow_rev1(b, c, x)` is the bases `{t ∈ x : t ** y ∈ c for some y ∈ b}` and
  `pow_rev2(a, c, y)` the exponents `{s ∈ y : t ** s ∈ c for some t ∈ a}`, with the library's pow
  (`pow_rev1(MI(-1, 1), MI(2))` is `(0, 1/2] ∪ [2, inf)`, `pow_rev2(MI(4), MI(2))` is `[1/2]`);
  exact where rational, else the tightest float enclosure, open. their 804 ITF1788 vectors run
  through the adapter (M13e, 2026-09-26)
* **1788 constructors** (M13g): `text_to_interval()` reads 1788's interval literals (`[1, 2]`,
  `[1,]`, `[entire]`, `3.56?1e2`, hex and `p/q` numbers), a syntax separate from
  `MultiInterval.parse`; `nums_to_interval()` takes two bounds. both give the exact set with an
  infinite end open, as 1788 reads one, and raise `UndefinedOperationError` (a `ValueError`) on
  invalid input, where 1788 signals `UndefinedOperation`
* **decorated intervals** (M13g): `DecoratedInterval(x)` is a set with 1788's best decoration for it
  (`Decoration.COM`, `DAC`, `DEF`, `TRV`; no NaI and no `ill`); `set_dec()` sets one as 1788 does,
  demoting it where it cannot fit; `.interval` and `.decoration` are its parts;
  `text_to_decorated_interval()` (`"[1, 2]_def"`) and `nums_to_decorated_interval()` are the
  decorated constructors. the core `MultiInterval` stays undecorated; a `DecoratedInterval`'s own
  arithmetic, functions, step functions, `%`, `//` and set operations compute the core's set and
  propagate the decoration as 1788 does (the weakest of the operands' and the op's own on the
  operands' sets: `DecoratedInterval(MI(1, 2)) / DecoratedInterval(MI(0, 1))` is `[1, inf]_trv`)
* **rounding**: `MultiInterval` rounds a float result to nearest; `OutwardMultiInterval` rounds it
  outward to the tightest float enclosure of the exact result, and an end that rounding moved is
  open. mixing the two gives an `OutwardMultiInterval`
* **warnings**: every lossy or surprising step warns with a subclass of `IntervalWarning`
  (`DomainClippedWarning`, `IndeterminateResultWarning`, `HullWarning`,
  `EmptySetPropagationWarning`)
* **ieee 1788**: not a runtime mode. the test suite runs 9269 vectors of 102 ops from all 19 files
  of the ITF1788 suite through an adapter, the 8256 interval-valued ones a second time through
  `OutwardMultiInterval`, the 167 numeric ones twice more with float operands, and all of them
  match except 157 listed divergences where the semantics
  differ on purpose or the vector needs decorations (`tests/itf1788/`, measured 2026-09-27, M13e). the
  273 statements of ops not built yet (the text constructors, decorations) are counted
  and skipped
* **1788's signals** (M13g): `UndefinedOperation` raises `UndefinedOperationError`, a `ValueError`,
  so a 1788 constructor or `DecoratedInterval` given invalid input stops, as `MI(2, 1)` does; hence
  there is no NaI. `PossiblyUndefinedOperation` would be `PossiblyUndefinedOperationWarning`, an
  `IntervalWarning`, with the result returned; the exact parser can always decide validity, so it
  is never emitted today
* **ieee 1788**: not a runtime mode. the test suite runs 7587 vectors of 92 ops from all 19 files
  of the ITF1788 suite through an adapter, the 6351 interval-valued ones a second time through
  `OutwardMultiInterval`, the 167 numeric ones twice more with float operands, a decorated vector
  through `DecoratedInterval` with its decoration checked, and all of them match except the 195
  vectors under 142 listed divergences where the semantics differ on purpose or the vector needs a
  NaI, and 12 on a decoration alone in the exact pass (`tests/itf1788/`, measured 2026-09-26 at M13g). the
  statements of ops not built yet (the reverse ops) are counted and skipped

## layout

* `intervals/` — the package: `cuts` (the representation), `kernel` (set algebra on cut tuples),
  `fmt` (printing and parsing), `multi_interval` (the two classes), `relations`, `applicator` and
  `ops` (arithmetic), `modulo`, `steps` (floor, ceil, round, sign), `functions` and `elementary`
  (the elementary functions over sets, and at one point), `numeric` (midpoint, radius, width,
  magnitude, mignitude), `reductions` (sums and dot products of numbers), `reverse` (the reverse
  ops), `literals` (1788's interval literals and constructors), `decorated` (1788's decorated
  type), `rounding`, `errors`
* `tests/` — the suite; `tests/oracles.py` holds the brute-force reference the arithmetic is checked
  against, `tests/itf1788/` the vendored conformance vectors (Apache 2.0, LGPL-2.1-or-later or
  all-permissive per file; see its README)
* `v2-plan.md` — the design. its "current design" section is normative: where it and the code
  disagree, one of them is a bug
* `v2-implementation-plan.md` — milestones (each one's spec and, once built, its record), decisions
  D1–D17
* `HANDOFF.md` — what is open now: ranked items, questions for the owner, a session log
* `references/` — papers and the modulo derivations
* `archive/v1/` — the previous implementation, kept unchanged as a reference: `multi_interval.py`,
  `interval.py`, `time_interval.py` (`DateTimeInterval`, `TimeDeltaInterval`), `compare.py`, and
  the old README with its notes and TODO list. `tests/test_kernel.py` still uses v1 as a
  differential oracle for set operations, so pytest puts `archive/v1` on the path

## status

`2.0.0.dev0`, on the `v2` branch. there is no time layer yet: v1's is archived and comes back on top
of the v2 class later (M8 in the implementation plan).

## tests

```
C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q
```

needs `pytest`, `hypothesis` and `python-flint` (`pip install -e .[test]`). the library's own
warnings are errors inside the suite. `HYPOTHESIS_PROFILE=fuzz` runs every hypothesis test
randomized at `FUZZ_MULTIPLIER` (default 100) times its examples, as the weekly
`.github/workflows/fuzz.yml` does.
