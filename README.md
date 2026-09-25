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
>>> x.size
Size(rays=0, length=2, points=0)
>>> print(MI(0, 3).sqrt())           # sqrt(3) is irrational: its enclosure's upper end, open
[0, 1.7320508075688774)
>>> print(MI(1, 8).log(2))           # exact where the value is rational
[0, 3]
>>> print(MI(1, 2).tan())            # a pole inside the piece: both sides, both infinities
{ [-inf, -2.185039863261519) , (1.557407724654902, inf] }
>>> import math
>>> print(math.floor(MI(-1.5, 1.5)))
{ [-2.0] , [-1.0] , [0.0] , [1.0] }
>>> from intervals import OutwardMultiInterval as OMI
>>> print(OMI(0.1) + 0.2)            # outward rounding: the exact sum is strictly between
(0.3, 0.30000000000000004)

```

## what it does

* **values**: int and Fraction are exact and never rounded; float endpoints go through a rounding
  hook (identity by default). `-inf` and `inf` are ordinary points, so `[1, inf]` and `[1, inf)` are
  different sets, and `[inf]` is a legal degenerate interval
* **set algebra**: `| & ^ ~`, `difference()`, `issubset()`, `in`, slicing `x[a:b]` (restricts to
  `[a, b]`), `hull`, `expand()`, `size` (rays, length, isolated points, ordered lexicographically)
* **relations**: `< <= > >=` and `eq_pointwise()` return a `TruthSet` (`TRUE`, `FALSE`, `BOTH`,
  empty); `before after adjoins overlaps contains within` return bool; `allen()` gives the Allen
  relation. `==` is structural and `MultiInterval` is hashable and immutable
* **arithmetic**: `+ - * /`, `reciprocal()`, `abs`, `**` with int exponents, `%`, `//`, `divmod`,
  `minimum()`, `maximum()`, `fma()`, for every sign combination including zero-crossing and
  infinite operands. a result is
  the set of values attained: an infinite endpoint is closed iff it is attained, a pole at a closed
  zero attains the infinity of its piece's sign, and a box that *is* an indeterminate point
  (`1/[0]`, `[0]*[inf]`, `[inf]-[inf]`) is empty with an `IndeterminateResultWarning`
* **functions**: `sqrt`, `exp`, `exp2`, `exp10`, `log` (any base), `log2`, `log10`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, as
  methods. values are correctly rounded by a pure-python evaluator (no libm), so they are the same on
  every platform; an irrational value of an exact operand is its tightest float enclosure
* **step functions**: `floor()`, `ceil()`, `trunc()`, `round(ndigits)`, `round_ties_away()`,
  `sign()`, and `math.floor/ceil/trunc` and `round()` on a set: the values attained, listed up to
  1000 of them, else their hull with a `HullWarning`
* **rounding**: `MultiInterval` rounds a float result to nearest; `OutwardMultiInterval` rounds it
  outward to the tightest float enclosure of the exact result, and an end that rounding moved is
  open. mixing the two gives an `OutwardMultiInterval`
* **warnings**: every lossy or surprising step warns with a subclass of `IntervalWarning`
  (`DomainClippedWarning`, `IndeterminateResultWarning`, `HullWarning`,
  `EmptySetPropagationWarning`)
* **ieee 1788**: not a runtime mode. the test suite runs 4767 vectors of 54 ops from all 19 files
  of the ITF1788 suite through an adapter, the 4120 interval-valued ones a second time through
  `OutwardMultiInterval`, and all of them match except 49 listed divergences where the semantics
  differ on purpose or the vector needs decorations (`tests/itf1788/`, measured 2026-09-26). the
  statements of ops not built yet (`pow` with a real exponent, reverse ops, cancellation, ...) are
  counted and skipped

## layout

* `intervals/` — the package: `cuts` (the representation), `kernel` (set algebra on cut tuples),
  `fmt` (printing and parsing), `multi_interval` (the two classes), `relations`, `applicator` and
  `ops` (arithmetic), `modulo`, `steps` (floor, ceil, round, sign), `functions` and `elementary`
  (the elementary functions over sets, and at one point), `rounding`, `errors`
* `tests/` — the suite; `tests/oracles.py` holds the brute-force reference the arithmetic is checked
  against, `tests/itf1788/` the vendored conformance vectors (Apache 2.0, LGPL-2.1-or-later or
  all-permissive per file; see its README)
* `v2-plan.md` — the design. its "current design" section is normative: where it and the code
  disagree, one of them is a bug
* `v2-implementation-plan.md` — milestones, decisions D1–D17, and what is still open
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
