# multiinterval

*A GLORIOUS EXERCISE IN YAK-SHAVING*

`MultiInterval`: a finite union of disjoint intervals over the affine extended reals, each piece open
or closed at either end, with set algebra, pointwise comparisons and arithmetic that returns exactly
the set of values attained.

```python
>>> from multiinterval import MultiInterval as MI
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
>>> A = MI(0, 1) | MI(3, 5)
>>> [[r.name for r in row] for row in A.allen_matrix(MI(1, 4))]   # allen() of each pair of pieces
[['OVERLAPS'], ['OVERLAPPED_BY']]
>>> sorted(r.name for r in A.allen_relations(MI(2, 6) | MI(8, 9)))   # the relations holding
['BEFORE', 'DURING']
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
>>> from multiinterval import sqr_rev, mul_rev, sin_rev
>>> print(sqr_rev(MI(1, 4)))         # reverse ops: the t with t ** 2 in [1, 4], not 1788's hull
{ [-2, -1] , [1, 2] }
>>> print(mul_rev(MI(-1, 1), MI(1, 2)))   # the t with t * y in [1, 2] for some y in [-1, 1]
{ (-inf, -1] , [1, inf) }
>>> print(sin_rev(MI(0), MI(-1, 7)))  # the t in [-1, 7] with sin t = 0: 0 exact, pi and 2 pi enclosed
{ [0] , (3.141592653589793, 3.1415926535897936) , (6.283185307179586, 6.283185307179587) }
>>> import math
>>> print(math.floor(MI(-1.5, 1.5)))  # the integers it attains, exact, as python's math.floor gives ints
{ [-2] , [-1] , [0] , [1] }
>>> from multiinterval import OutwardMultiInterval as OMI
>>> print(OMI(0.1) + 0.2)            # outward rounding: the exact sum is strictly between
(0.3, 0.30000000000000004)
>>> from multiinterval import text_to_interval, text_to_decorated_interval, DecoratedInterval
>>> print(text_to_interval('[0.1, infinity]'))   # 1788's literals, read exactly; an infinite end open
[1/10, inf)
>>> text_to_interval('[2, 1]')                   # 1788's UndefinedOperation raises
Traceback (most recent call last):
    ...
multiinterval.errors.UndefinedOperationError: invalid 1788 interval literal '[2, 1]': the lower bound exceeds the upper
>>> d = text_to_decorated_interval('[1, 4]_com')
>>> print(d.sqrt())                              # decorations propagate as 1788's do
[1, 2]_com
>>> print(d / DecoratedInterval(MI(-1, 1)))      # 1/0 is outside the domain: trv
{ [-inf, -1] , [1, inf] }_trv
>>> from multiinterval import derivative, newton
>>> print(derivative(lambda t: t ** 3 - 2 * t, MI(-1, 2)))   # forward-mode autodiff over a set
[-2, 10]
>>> for root in newton(lambda t: t ** 2 - 2, MI(-10, 10)):   # every zero, each proved unique
...     print(root.unique, root.interval)
True (-1.4142135623730951, -1.414213562373095)
True (1.414213562373095, 1.4142135623730951)
>>> from multiinterval import gradient, solve
>>> print(*gradient(lambda x, y: x * y ** 2, [MI(1, 2), 3]))     # n passes, one variable seeded each
[9] [6, 12]
>>> for root in solve(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [MI(-10, 10), MI(-10, 10)]):
...     print(root.unique, *root.box)                            # a square system: krawczyk proves
True (-0.7071067811865476, -0.7071067811865475) (-0.7071067811865476, -0.7071067811865475)
True (0.7071067811865475, 0.7071067811865476) (0.7071067811865475, 0.7071067811865476)
>>> from multiinterval import ieee1788                 # 1788's own answers, as a thin layer
>>> from multiinterval.ieee1788 import Interval
>>> Interval(1, 2) / 10                            # binary64, rounded outward
Interval(0.09999999999999999, 0.2)
>>> ieee1788.cancel_minus(Interval(0, 1), Interval(0, 2))   # 1788's "no answer"
Interval(float('-inf'), float('inf'))
>>> ieee1788.overlap(Interval(1, 2), Interval(2, 3))
<Overlap.MEETS: 'meets'>
>>> print(ieee1788.sqrt(Interval(-1, 4, 'com')), ieee1788.NAMES['mulRevToPair'](Interval(-1, 1), Interval(1, 2)))
[0.0, 2.0]_trv (Interval(float('-inf'), -1.0), Interval(1.0, float('inf')))

```

## what it does

* **values**: int and Fraction are exact and never rounded, but for a power too long to build (past
  2**22 bits, about 1.26M digits: pown, pow, `exp2`, `exp10`), which becomes a float like an
  irrational value, with a `PowerLimitWarning` (ignored by default; make it an error to forbid it);
  float endpoints go through a rounding hook (identity by default). each end keeps its own type: a value
  computed from a float is a float, but what is known exactly stays exact, from float operands too: the
  constants a function reaches (`abs(MI(-1.0, 1.0))` is `[0, 1.0]`, `MI(-1.0, 1.0).cos()`
  `[0.5403023058681398, 1]`) and the integers a step function lists; an end reached by an exact and a float
  value of one number is exact. `-inf` and `inf` are ordinary points, so `[1, inf]` and `[1, inf)` are
  different sets, and `[inf]` is a legal degenerate interval
* **set algebra**: `| & ^ ~`, `difference()`, `issubset()`, `in`, slicing `x[a:b]` (restricts to
  `[a, b]`), `hull`, `interior` (every end opened), `expand()`, `size` (rays, length, isolated
  points, ordered lexicographically)
* **relations**: `< <= > >=` and `eq_pointwise()` return a `TruthSet` (`TRUE`, `FALSE`, `BOTH`,
  empty); `before after adjoins overlaps contains within` return bool; `allen()` gives the Allen
  relation of two contiguous sets, and `allen_matrix()` and `allen_relations()` that of every pair
  of pieces, as a matrix or as the set of relations holding; `weakly_less()` and `strictly_less()` are 1788's interval orders, on the ends (the
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
  (`MI(-3, 1) ** MI(2)` is `[0, 1]`). `2 ** A` is `MI(2) ** A`; 3-argument `pow` is refused.
  both are correctly rounded with no libm, pown to nearest too (python's `float ** int` is libm's
  `pow` and can be an ulp off). exact operands give an exact power up to 2**22 bits, the same limit
  for both: `MI(3) ** 70000` and `MI(3) ** MI(70000)` are the same 110948-bit int; past it pown and pow
  give the tightest float enclosure, open, in both classes (`MI(2) ** 2 ** 60` and `MI(2) ** MI(2 ** 60)`
  are `(MAX, inf)`; the float `MI(2.0) ** 2 ** 60` is `[inf]`, rounded to nearest)
* **cancellation**: `A.cancel_minus(B)` is the Minkowski difference, the largest `X` with
  `B + X ⊆ A`, for any two sets (`∅` when nothing fits); `A.cancel_plus(B)` is
  `A.cancel_minus(-B)`. 1788's `cancelMinus`/`cancelPlus` where 1788 has an answer, a real set
  where it answers entire as "no answer". exact for exact operands, outward an enclosure of `X`
* **functions**: `sqrt`, `exp`, `exp2`, `exp10`, `expm1`, `log` (any base), `log2`, `log10`,
  `log1p`, `cbrt`, `rootn(n)`, `hypot`, `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`,
  `atan`, `acot`, `atan2`, `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, `asinh`, `acosh`,
  `atanh`, `acoth`, as methods; a pole inside a piece gives both infinities, as `1/x` does. values are correctly rounded by a pure-python evaluator (no libm, pown included), so they are the same on
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
  operands' sets: `DecoratedInterval(MI(1, 2)) / DecoratedInterval(MI(0, 1))` is `[1, inf]_trv`).
  the reverse ops take `DecoratedInterval` operands too and decorate the result trv, as 1788 does
* **autodiff** (M15): `Dual.variable(X)` and the ops on it (`+ - * / **`, `abs`, `reciprocal`, the
  elementary functions) carry a derivative beside the value, each a set, by the chain rule over the
  library's own ops; `derivative(f, X)` encloses `f'` over `X`. with `DecoratedInterval` parts the
  decorations prove `f` C¹ on `X` (dac or better on both), which an enclosure of `f'` alone does not
* **interval newton** (M15): `newton(f, X)` returns `Root(interval, unique)`s holding every zero of
  `f` in `X` (any multi-interval, unbounded included), `unique` when exactly one zero is proved. its
  step is `m + mul_rev(F', -f(m))`, so where the derivative's set holds 0 one step cuts the piece in
  two, where a connected interval type would get the hull; the step runs only where the
  decorations prove `f` C¹, elsewhere the pieces are pruned by range and bisected. it computes in
  `OutwardMultiInterval`, so every root encloses
* **several variables** (M16a): `gradient(f, xs)` and `jacobian(F, xs)` are n passes of forward-mode
  autodiff, one variable seeded each. `solve(F, xs)` returns `RootBox(box, unique)`es holding every
  zero of a square system `F` in the box `xs`: gauss-seidel with `mul_rev` narrows (a partial holding
  0 splits the box in one step), krawczyk's test proves a zero unique, on the closed hull and, for a
  converged box, once more on the box inflated within the part of the input it stands for; a zero at
  a simple rational is output as that exact point. the step runs only where the decorations prove
  `F` C¹ on the box
* **numpy** (M16d, optional: the library never imports it; the `[test]` extra installs it for the
  gate): a numpy scalar is a python number to every op (`np.float32(0.1)` is the double it holds; an
  `np.longdouble` wider than a double is exact, as any foreign real); ufuncs on a set are its methods
  (`np.sin(A)` is `A.sin()`, `np.arcsin(A)` is `A.asin()`, `np.square(A)` is `A ** 2`, `np.fmin` is
  `minimum`, a nan refused as everywhere) or python's operators (`np.add(M, O)` is `M + O`), anything
  else a `TypeError`; an ndarray meeting a set is elementwise into an object array, and `==` into a
  bool array, as numpy compares any element type (each element structurally); `np.array([A, B])`
  holds the sets as elements. `np.asarray(x, dtype=float)` rounds each point to nearest, in both
  classes. the 1788 layer's `Interval` is a scalar to numpy: operators with numpy scalars work,
  ufuncs do not. object arrays of sets run numpy's own loops, which look for numpy's names
  (`np.arcsin(arr)` and `np.round(A)` are TypeErrors; `np.frompyfunc` reaches the method):

  ```python
  >>> import numpy as np
  >>> np.float32(0.1) + MI(0)                        # the double a float32 holds, exactly
  MultiInterval.parse('[0.10000000149011612]')
  >>> np.arcsin(MI(0, 1)) == MI(0, 1).asin()
  True
  >>> np.hypot(MI(0.1), OMI(0.1)) == MI(0.1).hypot(OMI(0.1)) == OMI(0.1).hypot(MI(0.1))
  True
  >>> print(np.fmin(MI(1, 2), 1.5))
  [1, 1.5]
  >>> print(*(np.linspace(0, 1, 3) + MI(1, 2)))       # elementwise, an object array
  [1.0, 2.0] [1.5, 2.5] [2.0, 3.0]
  >>> arr = np.array([MI(1, 2), MI(3)])              # two elements, not their pieces
  >>> arr.shape, arr == MI(3), MI(3) in arr
  ((2,), array([False,  True]), True)
  >>> print(*np.frompyfunc(MI.asin, 1, 1)(np.array([MI(0), MI(1)])))
  [0] (1.5707963267948966, 1.5707963267948968)

  ```
* **time** (M8): `DateTimeInterval` and `TimeDeltaInterval` are sets of instants and of durations,
  immutable and hashable, thin wrappers over a `MultiInterval` of exact seconds with its set algebra,
  relations and `TruthSet` comparisons, and datetime and timedelta arithmetic (`dt - dt` is a
  `TimeDeltaInterval`, `td / td` a `MultiInterval`). a naive datetime is wall-clock time (never
  `timestamp()`), an aware one its UTC instant, and the two never mix (`TypeError`); aware ends in
  different zones do, one zone kept for display, and aware arithmetic is in elapsed time (across a DST
  change `t + 1 day` is 24 h later, where python's aware `+` keeps the wall time). a `date` is the
  half-open day `[d 00:00, d+1 00:00)`, so days tile and a day is 86400 s; a datetime is an exact instant;
  two bounds are ordered as read (`DTI(noon, day)` is noon through that day). `NEG_INF` and `POS_INF` are
  the infinite ends, ordered against every time type. nothing is rounded: an end that is no whole number
  of microseconds raises when read out as a datetime, and `inf_seconds`, `sup_seconds` and `seconds` give
  it exactly. pandas' `Timestamp` and `Timedelta` are read exactly in their unit, `NaT` is refused (a
  `ValueError`, in every operator too), numpy's `datetime64`/`timedelta64` are a `TypeError`, and
  `to_pandas()` / `from_pandas()` convert one bounded piece to and from a `pd.Interval`; the library
  never imports pandas otherwise. one limit: with a pandas `Timedelta` on the left of `%` or `divmod`,
  pandas computes `x - (x // A) * A` itself, a sound but wider set; write `TDI(x) % A`, which is exact:

  ```python
  >>> import datetime
  >>> from zoneinfo import ZoneInfo
  >>> from multiinterval import DateTimeInterval as DTI, TimeDeltaInterval as TDI, NEG_INF
  >>> mon, tue = datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
  >>> print(DTI(mon, tue))                       # a closed date end: through that day
  [2024-01-01 00:00:00, 2024-01-03 00:00:00)
  >>> DTI(mon) | DTI(tue) == DTI(mon, tue), DTI(mon).total_duration   # days tile
  (True, datetime.timedelta(days=1))
  >>> work = DTI(datetime.datetime(2024, 1, 1, 9), datetime.datetime(2024, 1, 1, 17))
  >>> work - datetime.datetime(2024, 1, 1)
  TimeDeltaInterval(datetime.timedelta(seconds=32400), datetime.timedelta(seconds=61200))
  >>> print(work < datetime.datetime(2024, 1, 1, 12))      # pointwise, a TruthSet
  BOTH
  >>> until_noon = DTI(NEG_INF, datetime.datetime(2024, 1, 1, 12))
  >>> until_noon.inf, until_noon.inf < datetime.datetime.min
  (-inf, True)
  >>> DTI(datetime.datetime(2024, 1, 1, 8, tzinfo=ZoneInfo('Asia/Singapore'))) == \
  ...     DTI(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc))   # the same instant
  True
  >>> TDI(datetime.timedelta(hours=1), datetime.timedelta(hours=3)) / datetime.timedelta(minutes=30)
  MultiInterval.parse('[2, 6]')
  >>> third = TDI(datetime.timedelta(seconds=1)) / 3
  >>> third.inf
  Traceback (most recent call last):
  ValueError: 1/3 s is not a whole number of microseconds, so it has no datetime or timedelta; `inf_seconds` gives it exactly
  >>> third.inf_seconds
  Fraction(1, 3)

  ```
* **rounding**: `MultiInterval` rounds a float result to nearest; `OutwardMultiInterval` rounds it
  outward to the tightest float enclosure of the exact result, and an end that rounding moved is
  open. mixing the two gives an `OutwardMultiInterval`, by an operator or by a method taking another
  set (`MI(0.1).hypot(OMI(0.1))` is outward, as `OMI(0.1).hypot(MI(0.1))`). an exact end stays
  exact, so the outward class is isotone within one grid; across grids (a float piece of `A` inside
  an exact piece of `B`), `f(A)` lies within the tightest double cover of `f(B)`, which
  `f(B).rounded()` gives: round the inputs first (`A.rounded()`: every end a double, outward) and
  `A ⊆ B` gives `f(A) ⊆ f(B)`. to nearest, as python's float, a value past the
  largest double is `inf` (`MultiInterval(1e308) * 10` is `[inf]`, the point, so `& (0, inf)` leaves
  nothing); the outward class keeps it as `(MAX, inf)`. the reverse ops meet `x` before rounding, as
  1788 does: a part of the answer inside `x` that rounds wholly onto one double is that double, even
  an end `x` excludes (D26)
* **a faster backend, optional** (M16e): `pip install multiinterval[fast]` adds gmpy2, and
  `MULTIINTERVAL_BACKEND=gmpy2` (or `auto`: gmpy2 when it imports, else the pure path, silently) picks
  it at `import multiinterval`. it computes the same doubles as the default pure-python path, only
  faster (a few times for the elementary functions at a float, less over a whole set, little for
  arithmetic), and changes no flag, so every result is the same either way:

  ```python
  >>> from multiinterval import OutwardMultiInterval
  >>> OutwardMultiInterval(0.5, 2.0).exp()
  OutwardMultiInterval.parse('(1.648721270700128, 7.38905609893065)')
  >>> OutwardMultiInterval(0.1) + 0.2
  OutwardMultiInterval.parse('(0.3, 0.30000000000000004)')

  ```

  reporting a result, say which backend computed it: `multiinterval.backend.name()` is `'python'` or
  `'gmpy2'`
* **warnings**: every lossy or surprising step warns with a subclass of `IntervalWarning`
  (`DomainClippedWarning`, `IndeterminateResultWarning`, `HullWarning`,
  `EmptySetPropagationWarning`, `PowerLimitWarning`)
* **text form**: `repr` evaluates back and `str` is what `MultiInterval.parse` reads. an int too long
  for python to write in decimal (past `sys.get_int_max_str_digits()`, 4300 digits by default) is
  written in hex (`0x...`), which `parse` reads, so `repr` never raises; `parse` keeps python's limit
  on a decimal literal. a number in that text is one python's `int`, `float` or `Fraction` reads, in
  ASCII digits (`-5`, `1_000`, `.5`, `1e-05`, `1 / 3`, `inf`; also `∞` and hex), and two items need
  `,` `;` `|` or `∪` between them: white space only pads, so `[1 2]`, `- 5` and `[0.1.2]` are
  `ValueError`s. a separator stands only between two items or two numbers, so a trailing or
  doubled one is a `ValueError` too (`[1,]`, `{1,}`, `{1,,2}`), while `{}` and `[]` are the empty
  set (`multiinterval.fmt` has the grammar)
* **1788's signals** (M13g): `UndefinedOperation` raises `UndefinedOperationError`, a `ValueError`,
  so a 1788 constructor or `DecoratedInterval` given invalid input stops, as `MI(2, 1)` does; hence
  there is no NaI. `PossiblyUndefinedOperation` would be `PossiblyUndefinedOperationWarning`, an
  `IntervalWarning`, with the result returned; the exact parser can always decide validity, so it
  is never emitted today
* **ieee 1788**: not a runtime mode. the test suite runs every statement of all 19 files of the
  ITF1788 suite, 9542 vectors of 111 ops, through an adapter, the 8306 interval-valued ones a second
  time through `OutwardMultiInterval`, the 167 numeric ones twice more with float operands, and the
  1226 with a decorated operand or result through `DecoratedInterval` with their decoration checked
  (398 more are booleans or numbers of a decorated interval, which take its interval part); all of them match
  except the 271 vectors under 185 listed divergences where the semantics differ on purpose or the
  vector needs a NaI, and 64 on a decoration alone (12 in the exact pass only; 52 `mulRevToPair`
  pairs whose set matches) (`tests/itf1788/`, measured 2026-09-27 at M13's merge). no statement is
  skipped
* **the 1788 layer** (M16b): `from multiinterval import ieee1788` gives 1788's inf-sup binary64
  intervals, bare and decorated, as one class, `ieee1788.Interval`, over the library: every set is
  the library's, converted in by 1788's input rule and out by its output rule (attained infinities
  dropped, the hull rounded outward to doubles, an infinite end open), with 1788's answer where it
  defines another (cancellation's "no answer" is entire, touching intervals `meets`,
  `mul_rev_to_pair` decorated as the division, NaN for the numbers of the empty set and a reduction
  with no value). 1788's names in snake_case, and `ieee1788.NAMES` in
  1788's own spelling. a third conformance pass runs every vector through it and compares exactly:
  all match but 104 vectors under 94 rows (no NaI, tighter than the vector, exact parsing;
  2026-09-28)

## departures from ieee 1788

where the library answers otherwise than 1788, on purpose, and where each choice is recorded: `D` rows
and headings are `docs/decisions.md`'s, `Q` items the owner's questions, answered there (`v2-plan.md`'s
headings are in `docs/archive/v2/`). 1788's own answers are in `multiinterval.ieee1788`, the thin layer. the itf1788 adapter
(`tests/itf1788/test_itf1788.py`) names the rows each departure produces; most produce none, since the
adapter compares closed hulls in binary64 (`docs/archive/v2/v2-plan.md` "ieee 1788"). surveyed 2026-09-30.

| | 1788 | this library | recorded |
|---|---|---|---|
| sets | connected intervals; a hull (`1/[-1, 1]` is entire) | finite unions (`[-inf, -1] ∪ [1, inf]`); reverse ops give the union | "ieee 1788"; "division semantics vs ieee 1788" |
| ends | closed; infinity never attained | open or closed; ±inf are points, `[inf]` is legal | D1, D6; "domain and semantics" |
| numbers | a floating-point format | int and Fraction exact, never rounded (a power past 2**22 bits is a float) | D3 |
| indeterminate points | not writable | `[0] * [inf]`, `1/[0]` are empty, with a warning | D2, D7 |
| a domain end with no value | dropped (`log([0])` empty) | its limit, a point (`log([0])` is `[-inf]`) | "elementary and step functions"; rows: degenerate infinities |
| rounded ends | closed | outward, a moved end is open | "arithmetic" (flags at rounded ends) |
| rounding | every result encloses | `MultiInterval` rounds to nearest as python's float (overflow is the point `inf`); `OutwardMultiInterval` encloses; pown to nearest is correctly rounded, not libm's `pow` | "arithmetic" (rounding) |
| reverse ops and `x`, to nearest | `x` first, then enclose | `x` first, then round to nearest: a part rounding onto one double is kept, even an end `x` excludes | D26 |
| periodic reverse ops | the hull | exact pieces; the hull past 1000 or over an unbounded `x` | D12 |
| step functions | `floor([-1.5, 1.5])` is `[-2, 1]` | the points `{-2, -1, 0, 1}`; the hull past 1000 | "elementary and step functions" (no decision of its own) |
| cancellation | entire as "no answer" | the Minkowski difference | D13; rows: cancellation as a Minkowski difference |
| relations | `overlap([1, 2], [2, 3])` is meets | overlaps: they share the point 2 | "comparisons"; rows: cut-based relations |
| interval orders | `less`, `strictLess` | `weakly_less()`, `strictly_less()`; `<` is pointwise, a `TruthSet` | D10 |
| NaI and signals | NaI; signals are flags | no NaI; UndefinedOperation raises, PossiblyUndefined warns | D16; "2026-09-26 revision: owner answers to the open questions" (Q1, Q8); rows: no NaI |
| parsing | may round first | exact: validity decided on the exact bounds | D18(b); rows: exact parsing decides validity |
| constructors | the binary64 hull | the exact set (`[0.1, infinity]` is `[1/10, inf)`); the hull is `ieee1788.text_to_interval` | "2026-09-26 revision: M13g part 1"; D21(d) (Q10 closed as built, owner 2026-10-03) |
| tightness | some vectors 1-2 doubles loose | the tightest enclosure | D18(a); rows: tighter than the vector |
| decorations | on every op; decided in binary64 | only on `DecoratedInterval`; decided on the exact set, per piece | D16, D18(c), D18(d); "2026-09-26 revision: M13g part 3" |
| mulRevToPair's decoration | the first interval as `c / b` | `mul_rev` is one op, trv; 1788's pair is `ieee1788.mul_rev_to_pair` | D21(c) (Q9 closed as built, owner 2026-10-03); rows: decoration expectations |
| numbers of the empty set | NaN | `ValueError`; `mig`/`mag` of the set, not the hull; `ieee1788`'s are NaN | D9; the layer: D21(b) (Q13(b), owner 2026-10-03) |
| reductions | NaN for nan, `inf + -inf`, `0 * inf` | `ValueError`; `ieee1788`'s are NaN | "2026-09-26 revision: owner answers to the open questions" (Q2); the layer: Q13(b) |
| zero | signed | one zero | "2026-09-22 revision: signed zero dropped" |
| warnings | none | `EmptySetPropagationWarning`, `DomainClippedWarning`, `IndeterminateResultWarning`, `HullWarning` | "empties and warnings" |

not departures but additions with no 1788 counterpart: `%`, `//`, `divmod`, `round(ndigits)`, the
Allen relations and matrices, `TruthSet` comparisons.

## layout

* `multiinterval/` — the package: `cuts` (the representation), `kernel` (set algebra on cut tuples),
  `fmt` (printing and parsing), `multi_interval` (the two classes), `relations`, `applicator` and
  `ops` (arithmetic), `modulo`, `steps` (floor, ceil, round, sign), `functions` and `elementary`
  (the elementary functions over sets, and at one point), `numeric` (midpoint, radius, width,
  magnitude, mignitude), `reductions` (sums and dot products of numbers), `reverse` (the reverse
  ops), `literals` (1788's interval literals and constructors), `decorated` (1788's decorated
  type), `autodiff` (`Dual`, `gradient`, `jacobian`), `solver` (`newton`, `solve`, `RootBox`),
  `time_interval` (`DateTimeInterval`, `TimeDeltaInterval`, `NEG_INF`, `POS_INF`; pandas imported only by
  `to_pandas()`),
  `numpy_compat` (numpy's hooks, numpy imported only when numpy calls them), `backend` and `_gmpy2`
  (the optional gmpy2 backend), `rounding`, `errors`; and `ieee1788` (1788's intervals over the
  library, not imported by `multiinterval`)
* `tests/` — the suite; `tests/oracles.py` holds the brute-force reference the arithmetic is checked
  against, `tests/itf1788/` the vendored conformance vectors (Apache 2.0, LGPL-2.1-or-later or
  all-permissive per file; see its README)
* `docs/decisions.md` — every decision, D1–D30 and a dated log
* `docs/records.md` — what was built after v2, each item's evidence
* `docs/archive/v2/` — the v2 plans, read-only: the design (`v2-plan.md`, its "current design" the fullest
  statement of the semantics as built) and the milestones with their records (`v2-implementation-plan.md`)
* `HANDOFF.md` — what is open now: ranked items, questions for the owner, a session log (older entries in
  `docs/session-log.md`)
* `references/` — papers and the modulo derivations; `v1-readme.md` is the previous implementation's
  README, kept for its notes. v1 itself was deleted on 2026-10-04 once v2 did everything it did
  (`references/v1-parity-2026-10-04/`); git history has it (`git show 22e16f8:archive/v1/multi_interval.py`)

## status

`2.0.0.dev0`, on the `master` branch. the time layer is back on the v2 class (M8, 2026-10-04).

## tests

```
C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q
```

needs `pytest`, `hypothesis`, `python-flint`, `gmpy2`, `numpy` and `pandas` (`pip install -e .[test]`);
the library itself needs none of them. the library's own
warnings are errors inside the suite. `HYPOTHESIS_PROFILE=fuzz` runs every hypothesis test
randomized at `FUZZ_MULTIPLIER` (default 10; 100 until 2026-09-27) times its examples, as
`.github/workflows/fuzz.yml` does on every push to `master` and `tools/prepush.sh` does locally.
