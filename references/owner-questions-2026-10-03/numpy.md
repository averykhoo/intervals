# Q15: M16d's numpy choices (D23) -- options walk-through and recommendations

written 2026-10-03 by a read-only advisory agent. sources: `HANDOFF.md` Q15(a)-(h); plan §0 D23 and
§2 "M16d numpy interop" (`v2-implementation-plan.md`); `v2-plan.md` "numpy (M16d, ...)" and
"2026-09-28 revision: M16d"; `intervals/numpy_compat.py`; `tests/test_numpy_compat.py`;
`cuts.py::normalize_value`. "probed" = run today in the `intervals` env (numpy version noted in §0);
"inferred" = from reading code/docs; "from memory" = my knowledge of other libraries, not checked.

## 0. what is built (common ground)

env probed 2026-10-03: numpy 2.5.2, python 3.13.15, gmpy2 2.3.1, pandas installed; `np.longdouble` is
64 bits on this windows box (so the wide-longdouble case is reachable only on linux CI).

* `intervals/numpy_compat.py::array_ufunc` is the `__array_ufunc__` of `MultiInterval` (so also
  `OutwardMultiInterval`), `DecoratedInterval` and `Dual` (`multi_interval.py:57`, `decorated.py:207`,
  `autodiff.py:53`). the 1788 layer's `Interval` sets `__array_ufunc__ = None` (`ieee1788.py:143`):
  numpy defers to its reflected dunders, no table.
* four tables: `_OPERATORS` (17 binary operator ufuncs -> dunder, reflected dunder, python operator),
  `_UNARY` (`negative positive absolute fabs invert`), `_METHODS` (24 unary set-image methods, numpy
  name -> library name), `_SYMMETRIC` (`minimum maximum hypot`), `_OTHERS` (`square`, `arctan2`). the
  table is keyed by ufunc *objects* (`_ufunc_names`). anything else, every method but `__call__`, every
  kwarg: `NotImplemented` -> numpy raises `TypeError`.
* `_operator` reruns python's protocol on our dunders only; `equal`/`not_equal` fall back to identity.
  `_subclass_first` makes `hypot/minimum/maximum/arctan2` pick the subclass operand (Q15(h)).
* the elementwise path (`array_ufunc`, the `np.frompyfunc` call under `np.errstate(all='ignore')`): an
  ndarray operand of ndim > 0 with any ufunc of the table except `equal`/`not_equal` -> object array.
* `numpy_compat.py::array` is `MultiInterval.__array__`: a 0-d object array (so `np.array([A, B])` is
  shape (2,)); `copy=False` a ValueError.
* `cuts.py::normalize_value` + `cuts.py::_exact_value`: a foreign real (`numbers.Real` that is not
  int/float/Fraction) is exact by type if `numbers.Rational`, else exact via `as_integer_ratio()` only
  where `float()` would move it; a real without `as_integer_ratio` is `float()` of it.
* tests: `tests/test_numpy_compat.py` (782 lines, 2026-10-03; skips without numpy); numpy is installed in
  CI beside the `[test]` extra (`.github/workflows/ci.yml:31`, `fuzz.yml:46`), not in `pyproject.toml`
  `[project.optional-dependencies] test`.
* README.md:188-195 documents numpy as prose (not doctests; README is collected by pytest, so doctests
  there would need numpy for the gate).
* open item 5 `Q6-shift` (HANDOFF.md:91): when `<<`/`>>` land, `_OPERATORS` needs `left_shift`/
  `right_shift` or `::test_the_operators_are_derived` goes red (the test derives the operator list
  from the classes' reflected dunders).

### how other element types meet numpy (from memory unless marked probed)

* `Fraction`, `Decimal`: no `__array_ufunc__`; numpy wraps them as object arrays. a float array `+ Fraction`
  works elementwise via python's dunders inside numpy's object loop; `==` is elementwise bool (numpy's
  convention); unary ufuncs on object arrays call a *method of the ufunc's name* on each element
  (`Decimal.sqrt` works, `Decimal.ln` is not `log`, so `np.log` fails). (probed below.)
* gmpy2 `mpfr`/`mpq`: no numpy hook either; same object-loop behaviour (probed below).
* mpmath `mpf`: no hook; `np.sin(np.array([mpf(1)]))` fails for lack of a `.sin` method (the
  documented idiom is `np.vectorize(mpmath.sin)` or `mpmath.matrix`). not installed here.
* sympy: same shape; `sympy.lambdify` with the numpy module is the bridge, not object arrays. not installed.
* `uncertainties`: `ufloat` is a scalar with no array hook; `uncertainties.unumpy` supplies its own
  `sin`, `exp`, ... that work on object arrays of `ufloat`s and a `umatrix`. that is exactly the
  "alias names" route turned into a parallel namespace (Q15(d), option 3 there).
* `pint` `Quantity` and `astropy` `Quantity`: array *containers* (a Quantity wraps an ndarray) that
  implement `__array_ufunc__` *and* `__array_function__` with a per-ufunc table like `_OPERATORS`
  here. that is the "array type" shape of Q15(a), with the cost: pint's table is hundreds of lines and
  tracks numpy releases.
* the Array API standard (data-apis.org): a namespace of functions over arrays of a fixed-size dtype,
  with elementwise bool comparisons and IEEE special cases; implemented by numpy, cupy, torch, jax,
  dask, ndonnx. no scalar-element library implements it; it is for array containers.

the pattern across them: scalar-like element types either (i) rely on numpy's object loops (names must
match numpy's), or (ii) ship a parallel namespace (`unumpy`), or (iii) add `__array_ufunc__` as a
scalar (this library; rare -- most scalar types predate the protocol). none of the scalar types
override `==`'s elementwise meaning in object arrays; this library is the one that does, because its
`__array_ufunc__` intercepts `np.equal`.

## Q15(a) the array API or the hook

**the question.** should numpy interop be `__array_ufunc__` on the scalar classes (built), or an
interval *array* type exposing the array API standard's namespace (the plan's alternative)?

**what is built.** the hook on `MultiInterval`, `DecoratedInterval`, `Dual` (`numpy_compat.py::array_ufunc`)
plus `MultiInterval.__array__` (0-d object array). the 1788 layer's `Interval` has `__array_ufunc__ = None`
(`ieee1788.py:143`); `v2-plan.md` line ~881 notes "the rule for the layer is owed". probed 2026-10-03:
`np.linspace(0, 1, 10000) + A` 335 ms vs the list comprehension 314 ms; `np.sin` over a 10000-element
object array 1353 ms. the elementwise path buys convenience, not speed.

### option 1: `__array_ufunc__` on the scalar classes (built)
* meaning: a multi-interval is one element; numpy's ufuncs on it or on arrays meeting it go through
  the table; arrays *of* intervals are object arrays running numpy's own object loops.
* pros: small (200 lines), keyed by ufunc objects, every answer is "the method python would have
  reached", so nothing new to specify or fuzz beyond "ufunc == method"; numpy stays optional and
  unimported; pandas, masked arrays, `np.sum/mean/cumsum/sort/max` of object arrays all work today
  (probed); the `Q6-shift` follow-up is two `_OPERATORS` rows.
* cons: object arrays are python-speed; numpy's object loops use numpy's *names* (`arcsin`, `rint`) and
  numpy's *meanings* (`np.square(arr)` is `x*x`, `np.sign(np.array([M(-1,1)]))` is a ValueError via
  `bool(TruthSet)`), so the array path and the scalar path can disagree (Q15(d)); `np.round`, `np.clip`,
  `np.isclose`, `np.isnan` (array functions, not ufuncs) are unreachable: `np.round(A)` TypeError,
  `np.clip(M(1,2), 0, 1)` ValueError (probed).
* when better: the users are "I have an interval or a few in a numpy/pandas pipeline", not "I have a
  million intervals". that is this library's stated shape (ragged multi-intervals, exact ends, sets).

### option 2: an interval-array type with the array API namespace
* meaning: a new container (`IntervalArray`, fixed shape, lo/hi (and open/closed) planes as float
  arrays), implementing `__array_ufunc__`/`__array_function__` or the array-API namespace, with
  vectorized bound arithmetic. the shape pint/astropy `Quantity` and numpy-2 user DTypes
  (`numpy-user-dtypes`: `quaddtype`, `unytdtype`) take (from memory).
* pros: real speed for many intervals; `np.clip`, `np.where`, reductions by design; fits downstream
  array-API consumers.
* cons: a multi-interval is ragged (a variable number of pieces), so a fixed-size element can hold a
  single interval only -- the type would be a *different* object from `MultiInterval`, with its own
  arithmetic kernels (vectorized outward rounding needs `nextafter` planes, not python's exact
  Fractions), its own `==` (elementwise bool in the standard vs structural here), its own special
  cases (nan allowed in the standard, refused here). pint's dispatch table is hundreds of lines and
  tracks numpy releases; it would be the largest module in the package. "a new type, not interop"
  (HANDOFF's own words) is right.
* when better: a user base doing vectorized interval Newton/branch-and-bound over large boxes, or a
  downstream wanting an array-API-compatible interval dtype. none of that is in `v2-plan.md`'s goals.

### option 3 (not in the plan): keep the hook, add a *narrow* `__array_function__` on the scalars
* meaning: `__array_function__` dispatches `np.round`, `np.clip`, `np.isclose`, `np.around` etc. for an
  object that defines it, even a scalar; a 5-row table could map `np.round -> .round`, `np.clip ->
  .maximum(lo).minimum(hi)`, `np.isnan -> False`, the rest `NotImplemented`.
* pros: fixes the two TypeErrors users hit first (`np.round(A)`, `np.clip(A, lo, hi)`), still no new type.
* cons: `__array_function__` is all-or-nothing per call -- once defined, *every* `np.*` function with
  A among its arguments asks it first, and an unmapped one is a TypeError where today numpy's own
  fallback sometimes works (`np.sum(arr)`, `np.sort`); the table would grow by request forever (the
  slope the plan's "every other ufunc is a TypeError" avoids). numpy docs also recommend against it
  for non-array types (from memory).
* when better: only if owner feedback shows `np.round(A)` is the common complaint; then prefer a
  `rint` alias (Q15(d)) first, which costs one line.

### option 4: both, sequenced
* the hook now (built), an interval-array type later as its own module or package if a user asks.
  this is what `v2-plan.md` "the array API standard and `__array_function__` are not built" already
  records; it costs nothing to keep open.

**recommendation: option 1 as built, with option 4's note kept (array type "later, if ever").
confidence: high.** what would change my mind: a concrete user needing >10^5 intervals per operation
(then the type is a separate project anyway, not a change to this hook). owed, not a choice: a one-line
rule for the layer's `Interval` (`__array_ufunc__ = None` today: numpy scalars on the left use the
reflected dunders, so it works; document it or give it the same hook).

**cost of changing later.** before 2.0: none to keep. after 2.0: adding an array type is additive
(no break); removing the hook would break `np.float64(2) + A`'s result type for anyone relying on it,
so the hook is the commitment, and it is a cheap one.

## Q15(b) foreign reals

**the question.** a `numbers.Real` that is not int/float/Fraction (`np.longdouble` wider than a
double, gmpy2 `mpfr`, `mpq`, user stand-ins): exact value, `float()`, refuse, or value-rule-for-all?

**what is built.** `cuts.py::normalize_value`: `Rational` by type -> exact `Fraction`/int; any other
foreign real -> `cuts.py::_exact_value` via `as_integer_ratio()`, kept exact only where `float()` would
move it, else the float; no `as_integer_ratio` -> `float()`. tested: `tests/test_numpy_compat.py::test_foreign_reals_are_exact`,
`::test_longdouble_is_exact` (discriminates on linux only), `::test_gmpy2_values_are_exact`.
probed 2026-10-03: `M(mpfr('0.1', 60))` end is `Fraction(922337203685477581, 9223372036854775808)`;
`M(mpfr('0.1', 100)) * 3.0` collapses to float ends `(0.3, 0.30000000000000004)` (one float op later
nothing exact remains, so no lingering Fraction cost: 1000 iterations from a 100-bit start 163 ms vs
165 ms from a double); `M(mpfr(0.5, 100))` end is `float` (a double value stays a float); `M(mpq(1,3))`
is `[1/3]`; `M(Decimal('0.1'))` is a TypeError (Decimal is not `numbers.Real`); `np.longdouble` here is
64-bit so `np.longdouble('1e4000')` is `inf` on windows with a numpy RuntimeWarning.

### option 1: exact value where `float()` would round; `Rational` exact by type (built)
* meaning: the operand's mathematical value enters the set. a double-valued foreign real is the float.
* pros: the only option under which `OutwardMultiInterval`'s promise ("every result holds the exact
  result of its operands") survives a wide operand: `q in O(0) + Wide(q)` is pinned. zero cost on the
  hot path (one `isinstance(value, float)`); exactness dies after the first rounded op anyway (probed).
  matches how the library already treats `Fraction` (exact) and `int` (exact).
* cons: a user handing a 200-bit mpfr gets Fraction ends they may not expect (`repr` is a ratio, and
  for >4300-digit ints python's `int -> str` limit makes `repr` raise -- probed, pre-existing for
  `M(10**5000)` too, not this rule's). a Rational "by type" and a Real "by value" are two rules
  (`mpq(1,2)` stays `1/2`, `mpfr(0.5)` becomes `0.5`), which the HANDOFF flags.
* when better: any user of the outward class; anyone on linux with `np.longdouble` data; gmpy2
  backend users (M16e) handing mpfr values back in.

### option 2: refuse a foreign real that is not a double (TypeError)
* pros: nothing silent; the user converts explicitly (`Fraction(*x.as_integer_ratio())` or `float(x)`).
* cons: a double-valued longdouble passes and a wide one refuses, so linux-vs-windows behaviour differs
  by *data*, a flaky kind of error; refuses gmpy2's `mpfr` at 100 bits where the library can represent
  it exactly for free.
* when better: a library that wants no Fraction ends to appear unless the user typed one.

### option 3: keep `float()` for every foreign real (pre-M16d)
* pros: one rule, float ends only.
* cons: unsound for `OutwardMultiInterval` (an operand's value outside the result: the review's
  finding), and `np.longdouble('1e4000')` -> `inf` silently. the one option that violates a documented
  invariant; not defensible for the outward class.

### option 4: the value rule for Rationals too (`mpq(1,2)` -> the float 0.5)
* pros: one rule ("exact only where float() moves it").
* cons: `Fraction(1,2)` stays a Fraction today (`normalize_value` keeps non-integral Fractions), so
  `mpq(1,2)` would then differ from `Fraction(1,2)` -- the opposite inconsistency. a Rational *is* a
  Fraction's kind; by-type is the right rule for it.

### option 5 (not in the plan): also accept `Decimal` (any object with `as_integer_ratio`, not only `numbers.Real`)
* `Decimal` deliberately is not `numbers.Real`; accepting it would mean accepting non-Real numbers by
  duck typing, and `Decimal('NaN')`/`'Infinity'` handling. not recommended; mentioned only because the
  `as_integer_ratio` machinery is already there. a user converts with `Fraction(d)`.

**recommendation: option 1 as built. confidence: high.** what would change my mind: evidence that a
foreign real's exact `as_integer_ratio()` is slow or wrong for some type (mpfr's is exact and fast,
probed); or an owner preference for "no Fraction unless typed", which option 2 serves.

**cost of changing later.** before 2.0: free. after 2.0: a change here silently moves numeric results
(ends change type and value) -- the most expensive kind of change. decide now; this is already decided
correctly.

## Q15(c) an ndarray meeting ours; `==`/`!=`

**the question.** `np.linspace(0,1,3) + A` is elementwise into an object array (built). should `==`/`!=`
also be elementwise (numpy's convention), stay scalar identity (built), or should the whole
ndarray-meets-ours case be a TypeError (pre-M16d)?

**what is built.** `array_ufunc` sends every table ufunc with an ndim>0 operand through the elementwise
path *except* `equal`/`not_equal`, which take the scalar path: `_operator` runs `A.__eq__(arr)` ->
NotImplemented -> identity `arr is A` -> `False`. pinned in `::test_equality_never_broadcasts`.

**probed 2026-10-03 (today):**

| expression | today |
|---|---|
| `np.array([A]) == A` | `False` (scalar) |
| `A in np.array([A])` | `False` |
| `np.array([A]) == np.array([A])` | `array([True])` -- numpy's object loop, hook not consulted |
| `np.array([A, M(3)]) == np.array([M(3), M(3)])` | `array([False, True])` |
| `np.array(A) == A` (0-d) | `True` |
| `pd.Series([A, M(3)]) == M(3)` | `Series([False, True])` (pandas broadcasts itself) |
| `np.isin(np.array([A, M(3)]), [M(3)])` | `array([False, True])` |
| `M(3) in [A, M(3)]`, `in {A, M(3)}` | `True` |
| `np.array([Fraction(1,2)]) == Fraction(1,2)`, same for `Decimal`, `mpq`, `mpfr`, `frozenset` | `array([True])` |
| `np.array([2.0, 3.0]) == M(2)` | `False` (scalar) |

so today the *only* place where equality of an array against one of ours is not elementwise is "a
>=1-d ndarray directly against a bare scalar of ours": `arr == arr2`, `pd.Series == A`, 0-d, `isin`,
lists and sets are all elementwise/structural. the identity `False` is the odd one out, and it is
wrong in the plain sense that the array *does* hold `A`.

**probed, the elementwise alternative** (hook patched in-process to send `equal`/`not_equal` down the
elementwise path and cast to bool; `.scratch/owner-questions/probes-numpy/probe2.py`):

| expression | alternative |
|---|---|
| `np.array([A]) == A` | `array([True])` |
| `A in np.array([A])` | `True` |
| `np.array([A, M(3)]) == M(3)` | `array([False, True])`; `np.where(...)` finds index 1 |
| `np.array([2.0, 3.0]) == M(2)` | `array([False, False])` (each float vs A: identity False, no raise) |
| `M(2) in np.array([2.0, 3.0])` | `False` -- **unchanged** |
| `np.array([2.0, 3.0]) != M(2)` | `array([True, True])` |
| `np.float64(2) == M(2)` | `False` -- unchanged (scalar path) |
| `pd.Series([A]) == A` | `Series([True])` -- unchanged |

a nan element gives `False`, not a raise (identity fallback), so no new exception surface.

### option 1: elementwise for every table ufunc but `==`/`!=` identity (built)
* pros: "`==` is structural and never broadcasts" is one sentence; `f == A` is a scalar `False` like
  `2.0 == A`.
* cons: `arr == A` is `False` while `arr == np.array([A])` is `array([True])` and `pd.Series([A]) == A`
  is `True` -- the hook *introduces* the asymmetry (a hook-less element type gets numpy's elementwise
  answer, probed with Fraction/Decimal/mpq/mpfr); `A in arr` lies; `np.where(arr == A)` cannot find
  A. the HANDOFF's cost claim "changes what `f == A` and `A in f` mean today" is half right: `A in f`
  does not change (probed), only `f == A`'s *shape* does (scalar False -> all-False bool array).
* when better: a user who writes `if f == A:` with a float array f and expects a scalar. that is
  already un-numpy-like (`if arr == x` raises for arrays elsewhere).

### option 2: elementwise `==`/`!=` too (numpy's convention, every other element type's behaviour)
* meaning: `equal`/`not_equal` take the elementwise path when an ndim>0 array is present; result a
  bool array (the probe did `astype(bool)`). the scalar path (`np.float64(2) == M(2)`, `np.equal(A, A)`)
  unchanged.
* pros: consistent with `arr == arr2`, pandas, `np.isin`, lists/sets, and with how Fraction/Decimal/
  gmpy2 elements behave in numpy; `A in arr` becomes true; three lines in `array_ufunc` (drop the
  `name in ('equal', 'not_equal')` clause, cast to bool) and one test edit (`::test_equality_never_broadcasts`
  flips its last two asserts). structural `==` semantics are untouched: the elements still compare by
  cuts.
* cons: `f == A` becomes `array([False, ...])` instead of `False`, a shape change for a rarely-written
  expression; the README sentence "except `==`, which stays structural" (README.md:193) needs rewording
  ("`==` is structural per element; an array meeting a set compares elementwise, as numpy does").
* when better: any user who stores intervals in object arrays or DataFrames and looks them up -- the
  realistic numpy use of this library (`arr == A`, `np.where`, `A in arr`).

### option 3: an ndarray meeting ours is a TypeError (pre-M16d)
* pros: nothing elementwise to specify.
* cons: throws away the useful `np.linspace(0, 1, 3) + A` and pandas Series arithmetic (both work
  and are pinned); numpy's own object loops would still make `arr + 1` work, so the refusal would be
  only for "float array meets our scalar", an odd hole. no user favours it.

**recommendation: option 2 (elementwise `==`/`!=` with arrays; identity on the scalar path stays).
confidence: medium-high.** what would change my mind: an owner rule that a *comparison* against
an array must never produce an array because `<`/`<=` already produce object arrays of `TruthSet`s --
but those are elementwise today too (`np.array([2.0, 1.0]) < M(2)` is `array([FALSE, TRUE])`, probed),
so `==` is currently the only comparison that does not broadcast, which weakens the case for keeping it.

**cost of changing later.** before 2.0: three lines plus one test and a README sentence. after 2.0:
a return-type change (`bool` -> `ndarray[bool]`) for `f == A` -- breaking for anyone who wrote
`if f == A`. decide before 2.0.

## Q15(d) numpy's names as methods

**the question.** should the classes grow numpy-spelled aliases (`arcsin arccos arctan arcsinh arccosh
arctanh rint arctan2`) so numpy's *object loops* (which call a method named after the ufunc on each
element) and `np.round` work on arrays of intervals?

**what is built.** no aliases. the table maps `np.arcsin(A)` -> `A.asin()` for a *scalar* A; but
`np.arcsin(np.array([A]))` is numpy's object loop, which looks for `A.arcsin` and raises
`TypeError: ... no callable arcsin method` (probed). same for `np.rint(arr)` (wants `rint`; `np.rint(A)`
scalar works via the table), `np.arctan2(arr, 1)` (`AttributeError: 'MultiInterval' object has no
attribute 'arctan2'`, probed). `np.round(A)` and `np.round(arr)` are TypeErrors because `np.round` is
not a ufunc: it tries `A.round(decimals=0, out=None)` (our `round(ndigits)` rejects the kwargs), then
falls back to an object array and the `rint` object loop. python's `round(A)` works.

probed 2026-10-03, which of numpy's unary ufunc names the class already has: `acos acosh asin asinh atan
atan2 atanh cbrt ceil cos cosh exp exp2 expm1 floor hypot log log10 log1p log2 maximum minimum negative
positive reciprocal sign sin sinh sqrt tan tanh trunc` -- so `np.sqrt(arr)`, `np.exp2(arr)`, `np.cbrt`,
`np.expm1`, `np.log1p`, `np.hypot(arr, 1)`, `np.floor`, `np.reciprocal`, `np.abs` (uses builtin `abs`)
all work on object arrays today. missing: `arcsin arccos arctan arcsinh arccosh arctanh rint arctan2`
(the HANDOFF's eight; `fabs`, `square` are not method lookups in numpy's loops: `np.square(arr)` is
`x*x`, which no alias can fix). comparison: `Decimal` has `sqrt`/`exp` but `ln` not `log`, so
`np.log(np.array([Decimal(4)]))` fails the same way (probed); mpmath's `mpf` and gmpy2's `mpfr` have
none of the method names (`np.sqrt(np.array([mpfr(4)]))` TypeError, probed), and their users reach for
`np.vectorize`/`np.frompyfunc` or `uncertainties.unumpy`-style parallel namespaces (from memory).

a subclass with the aliases (`probe2.py::M2`) made `np.arcsin(arr)`, `np.rint(arr)`, `np.round(M2(...))`
and `np.around(arr)` work (probed); `np.round(M2(1.5, 2.5), 1)` worked too but through numpy's own
`rint(x*10)/10` object recipe, *not* `A.round(1)`: a nearest-rounded `x*10` can differ from the
library's exact decimal rounding. that is the "numpy's loop, not the table" caveat in miniature.

### option 1: no aliases (built)
* pros: one vocabulary (1788's); `dir(A)` stays clean; nothing on an object array silently takes
  numpy's recipe where the library's method would be tighter or exact (`np.round(x, n)` above,
  `np.square` already does); the honest story is "object arrays run numpy's loops; the scalar table
  is the library's".
* cons: `np.arcsin(arr)` TypeError where `np.arcsin(A)` works -- the array path and the scalar path
  disagree on eight names; `np.round(A)`, numpy's most-typed rounding call, is a TypeError while
  `round(A)` and `np.rint(A)` work. users of pandas columns of intervals hit this.
* when better: the owner's stated priority of one name per operation and "TypeError over a guess".

### option 2: the eight aliases on the three classes
* meaning: `arcsin = asin`, ..., `rint = round`, `arctan2 = atan2` as class attributes (and on
  `DecoratedInterval` and `Dual` where the method exists), documented as "numpy's spellings".
* pros: `np.arcsin(arr)`, `np.rint(arr)`, `np.arctan2(arr, x)`, `np.round(A)`/`np.round(arr)` work;
  the table and the loops then agree everywhere except `square` (numpy's `x*x`) and `np.round(x, n>0)`
  (numpy's `rint(x*10^n)/10^n`).
* cons: 8 x 3 = up to 24 attributes that are pure duplicates; `rint = round` is a *semantic* alias
  (1788 `round` is ties-to-even -- matches `rint`, but numpy's `np.round(A, 2)` then succeeds with
  numpy's recipe, a quiet looseness); the 1788 names stop being "the" names.
* when better: users living in pandas/numpy who store intervals as elements and want
  `df['x'].apply(np.arcsin)`-style code to just work; teaching material that switches between
  numpy floats and intervals by changing one import.

### option 3: `rint` only (not in the plan)
* `rint` fixes `np.round(A)`, `np.round(arr)`, `np.around`, `np.rint(arr)` at one attribute per class;
  the six inverse-trig names stay 1788's (users write `np.frompyfunc(M.asin, 1, 1)(arr)` or
  `np.vectorize`). cons: the `np.round(A, n)` looseness comes with it; a half-measure that still has
  two vocabularies.

### option 4: a parallel namespace (`intervals.unumpy`-like) -- not recommended
* `uncertainties.unumpy` ships `sin`, `exp`, ... that vectorize over object arrays. here it would be
  `np.frompyfunc` wrappers of the methods under numpy's names -- a module nobody asked for; the
  scalar table already does this for scalars, and `np.vectorize(M.asin)` is one line.

**recommendation: option 1 (no aliases), documented with the one-line recipe
`np.frompyfunc(MultiInterval.asin, 1, 1)(arr)`, which `v2-plan.md` "numpy" already states.
confidence: medium.** what would change my mind: a user report that `np.round(A)` is the first thing
tried -- then option 3 (`rint` only) at one line, accepting the `np.round(A, n)` caveat in the README.

**cost of changing later.** additive: aliases can be added after 2.0 without breaking anything; they
cannot be *removed* after 2.0. so "none now, add on demand" is the cheap direction.

## Q15(e) `np.invert(A)`

**the question.** `np.invert` is numpy's `~` ufunc. map it to the complement `~A` (built), or make the
explicit unary call a TypeError?

**what is built.** `_UNARY['invert'] = '__invert__'`; `np.invert(A)`, `np.bitwise_not(A)` (an alias
of the same ufunc object) are `~A`, the complement (probed: `{ [-inf, 1) , (2, inf] }`). `np.invert(D(A))`
is the decorated complement (TRV); `np.invert(Dual)` a TypeError because `Dual` has no `__invert__`
(`numpy_compat.py::_special` finds none); `np.logical_not(A)` a TypeError (not mapped). numpy's object
loop also calls `~`: `np.invert(np.array([A]))` is the complement elementwise regardless of the table
(probed), and `np.invert(np.array([Fraction(1)]))` raises python's `~` TypeError the same way.

### option 1: the complement (built)
* pros: `invert` *is* `~` in numpy (its object loop calls `PyNumber_Invert`), and `bitwise_and/or/xor`
  are already `& | ^` -- set algebra -- so `np.int64(1) | A` is union (probed `{ [1] , [2, 3] }`),
  `np.int64(2) & A` intersection, `np.int64(2) ^ A` symmetric difference. the unary completes the
  boolean algebra consistently; the object-array path agrees with the scalar path with no alias.
* cons: a reader of `np.invert(x)` in float code expects a bit flip, which has no meaning here; the
  complement over the extended reals is the only sensible reading, but a generic numpy pipeline that
  reaches `np.invert` probably had ints in mind.
* when better: anyone who already uses `~A` and the `& | ^` operators with numpy scalars on the left.

### option 2: TypeError for the explicit unary call only
* pros: nothing "bitwise" is answered with a set.
* cons: inconsistent three ways: `~A` works, `np.invert(np.array([A]))` works (numpy's loop, out of
  the table's hands), `np.bitwise_or(np.int64(1), A)` works, but `np.invert(A)` would not. there is no
  principled line between the binary bitwise ufuncs and the unary one.

**recommendation: option 1 as built. confidence: high.** what would change my mind: an owner decision
to drop `bitwise_and/or/xor` from `_OPERATORS` too (then `np.int64(1) | A` becomes a TypeError, and
`invert` goes with them for consistency). I would not: `np.int64(1) | A` is a reflected-operator call
python made before the hook existed, and refusing it would be a regression.

**cost of changing later.** before 2.0: one dict entry. after 2.0: changing a working call into a
TypeError (or the reverse) is a behaviour break either way; small blast radius.

## Q15(f) `fmin`/`fmax`

**the question.** `np.fmin`/`np.fmax` are `minimum`/`maximum` except that a nan operand is ignored
(the other operand is returned). not mapped (built, TypeError), or mapped to `minimum`/`maximum`?

**what is built.** not in any table: `np.fmin(A, 1.5)` is numpy's generic `TypeError: operand type(s)
all returned NotImplemented from __array_ufunc__` (probed) -- no reason in the message. `np.minimum(A, 1.5)`
is `[1, 1.5]`. on an object array numpy's own `fmin` loop runs a comparison and `bool(TruthSet)`, so
`np.fmin(np.array([A]), 1.5)` is a ValueError "true for some points and false for others" (probed) --
exactly as `np.minimum(np.array([A]), 1.5)` is. nan: `A.minimum(np.nan)` is `ValueError: nan is not a
point of the extended reals`; `np.minimum(np.array([1.0, np.nan]), A)` the same ValueError (elementwise
path), `np.fmin(...)` the generic TypeError. for hook-less types numpy's `fmin` works (`np.fmin(Fraction(1),
Fraction(2))` is `Fraction(1)`, `np.fmin(mpfr(1), mpfr(2))` is `mpfr('1.0')`, probed).

### option 1: not mapped (built)
* pros: the one point of `fmin` over `minimum` is nan-skipping, which the library refuses, so a
  TypeError says "you asked for a nan semantics that does not exist here".
* cons: the error is numpy's generic one, not that sentence; a user who wrote `np.fmin` because their
  pipeline has nans gets an opaque TypeError on *non-nan* data too, where `minimum` would have
  answered; numpy's own fallback for a hook-less scalar (`np.fmin(Fraction(1), Fraction(2))` works)
  is friendlier than the hook's refusal.
* when better: the owner wants every ufunc whose *definition* differs from the library's to refuse
  (as `fmod`, whose non-nan values differ, rightly does).

### option 2: map to `minimum`/`maximum`; a nan operand is refused as everywhere (ValueError)
* meaning: `_SYMMETRIC['fmin'] = 'minimum'`, `_SYMMETRIC['fmax'] = 'maximum'` (two entries; the
  subclass rule of Q15(h) comes for free).
* pros: follows the table's own rule ("the method computing the set image of the ufunc's pointwise
  function" -- on the extended reals without nan, fmin's pointwise function *is* min); the error on a
  nan becomes the library's clear `ValueError: nan is not a point of the extended reals`, better than
  the TypeError today; `np.fmin(A, 1.5)` and `np.minimum(A, 1.5)` agree as they do for every other
  element type without nans.
* cons: a user who wrote `fmin` for nan-skipping learns it does not skip only when a nan arrives
  (but then with the precise message). `fmod` stays refused, so "some differently-named ufuncs are
  mapped and others not" needs one sentence: mapped where the non-nan values agree.
* when better: numpy/pandas users on real data; anyone porting numpy float code.

### option 3: map with nan-skipping (`np.fmax(A, nan) is A`)
* pros: numpy's exact semantics.
* cons: the one place in the library where a nan is accepted and silently dropped; the elementwise
  path would then return `A` for a nan element while `np.minimum` raises -- two nan policies in one
  module; for `Dual` the derivative of "the other operand" is ill-defined at a nan input. breaks the
  "nan is refused" rule stated in `v2-plan.md` (constructors: "`nan` in a constructor is a `ValueError`").

**recommendation: option 2. confidence: medium.** what would change my mind: an owner rule "a ufunc
whose definition differs from ours anywhere is refused, full stop" -- then option 1, but add the
reason to the message by mapping `fmin`/`fmax` to an explicit raise with text, as the docstring does for
`fmod`.

**cost of changing later.** 1 -> 2 is additive (a TypeError becomes an answer): safe after 2.0 too.
2 -> 3 or 3 -> 2 changes results on nan inputs: before 2.0 only.

## Q15(g) numpy in the `[test]` extra

**the question.** `pyproject.toml` `[project.optional-dependencies] test` has `pytest hypothesis
python-flint gmpy2` and not numpy; CI installs numpy beside the extra (`.github/workflows/ci.yml:31`,
`fuzz.yml:46`); `tests/test_numpy_compat.py` skips without numpy; README's numpy section (README.md:188-195)
is prose because README is doctest-collected (`pyproject.toml:24`, `--doctest-glob=README.md`) and
`>>> import numpy` would fail without it.

### option 1: not in `[test]`; README prose (built)
* pros: `pip install -e .[test]` stays numpy-free (numpy is ~20 MB and the library does not need it);
  the gate on a numpy-less machine still runs.
* cons: on a numpy-less machine the gate is **green with the numpy tests skipped** -- a silently
  weaker gate. the repo's own reasoning for putting gmpy2 in `[test]` ("so that `tests/test_backend.py`'s
  differential never skips", `pyproject.toml:14`) applies verbatim to numpy. the owner's CLAUDE.md
  names this trap ("an assurance step that fails by PASSING"). README's numpy examples are not
  executed, so they can drift (two of them already describe `np.arcsin(arr)` and `np.round(A)` as
  TypeErrors; if Q15(d) changes, the prose must be edited by hand).
* when better: a contributor base on exotic platforms without numpy wheels. not this repo's case
  (the dev env has it; CI has it).

### option 2: add numpy to `[test]`; README numpy section as doctests
* pros: the gate cannot silently skip numpy; `[test]` means "everything the gate needs", as for gmpy2;
  README examples become pinned behaviour (and the `tools/prepush.sh` README-doctest step covers them).
  CI's `pip install -e ".[test]" numpy` loses the trailing `numpy` (one token in two workflows).
* cons: numpy becomes required to run the suite at all (`import numpy` in README doctests fails
  loudly rather than skipping). `tests/test_numpy_compat.py::test_numpy_is_never_imported_at_load`
  asserts "`pyproject.toml` has no numpy in `dependencies` or `[test]`" (plan §2 M16d record) -- that
  test needs its `[test]` half dropped (keep the `dependencies` half, which is the real invariant).
* when better: the gate is the product's assurance (it is here: ledger, prepush, CI watch).

### option 3: add numpy to `[test]` but keep README prose
* the gate point of option 2 without touching README. cheaper, but leaves README examples unpinned.

### option 4: `[test]` without numpy, README doctests that skip without it
* `>>> np = pytest.importorskip('numpy')` at the top of the README section skips the rest of that
  doctest file when numpy is missing (pytest honours `Skipped` inside doctests -- inferred from pytest
  docs, not probed here). cons: README would show a pytest line to users; a skipped README section is
  the same silent hole as option 1.

**recommendation: option 2. confidence: medium-high.** what would change my mind: an owner decision
that `[test]` is "the minimum to run the pure-python tests" rather than "the gate's dependencies" --
but gmpy2 (a C extension heavier to build than numpy) is already there for the opposite reason.

**cost of changing later.** packaging metadata only; free before or after 2.0 (the `[test]` extra is
not a user-facing API). the README-doctest half is editorial.

## Q15(h) both operands ours in a method ufunc

**the question.** `np.hypot(M, O)`, `np.minimum/maximum(M, O)`, `np.arctan2(M, O)`: when both operands
are ours and one's class is a proper subclass of the other's, whose method runs?

**what is built.** `numpy_compat.py::_subclass_first`: the subclass decides, as python's reflected-
operand rule does for the operators (`M + O` is `O.__radd__`), so `np.hypot(M(0.1), O(0.1))` and
`np.hypot(O(0.1), M(0.1))` are both the outward `(0.1414213562373095, 0.14142135623730953)`;
`np.arctan2(M(1), O(1))` promotes y to `OutwardMultiInterval` first. pinned in
`::test_both_ours_methods_are_subclass_first`. with no subclass relation (`M` and `DecoratedInterval`,
`M` and `Dual`) the first operand's method, which is a TypeError for `hypot` either way (probed; `M + Dual`
works through `Dual`'s dunders but `M.hypot(Dual)` does not).

**probed 2026-10-03: the methods called directly do NOT follow the subclass rule, and that contradicts
the README.** README.md:198: "mixing the two gives an `OutwardMultiInterval`". but:

| call | result |
|---|---|
| `M(0.1) + O(0.1)` | `OutwardMultiInterval` (operator: subclass rule) |
| `M(0.1).hypot(O(0.1))` | `MultiInterval.parse('[0.1414213562373095]')` -- nearest, a point that misses sqrt(0.02) |
| `O(0.1).hypot(M(0.1))` | `OutwardMultiInterval.parse('(0.1414213562373095, 0.14142135623730953)')` |
| `M(0.1).minimum(O(0.3))`, `.maximum`, `.fma(O, ..)`, `.cancel_minus(O)` | `MultiInterval` |
| `M(1).atan2(O(1))` | `MultiInterval` |
| `M(1).union(O(2))`, `.intersection(O(1))` | `MultiInterval`; `M(1) | O(2)`, `M(1) & O(1)` are `OutwardMultiInterval` |
| `M(5).minimum(O(0.1) + 0.2)` | `MultiInterval.parse('(0.3, 0.30000000000000004)')` -- an M carrying O's open rounded ends |

so the hook's rule (h) is right, and the inconsistency the HANDOFF notes ("`M.hypot(O)`, called
directly, is still M's") is a library defect against README.md:198, not a numpy question: every
binary *method* on `MultiInterval` that takes another set (`minimum maximum hypot atan2 fma
cancel_minus union intersection difference symmetric_difference`) ignores the operand's subclass while
every *operator* honours it.

### option 1: hook follows the subclass rule; methods unchanged (built)
* pros: `np.hypot(M, O)` is sound in either order; no library change.
* cons: `np.hypot(M, O) != M.hypot(O)` -- the one place where "the ufunc is the method" is false;
  the README promise is kept by the hook and broken by the method it stands for.

### option 2: hook takes the first operand's method and class (as the method does today)
* pros: "the ufunc is the method" holds literally.
* cons: `np.hypot(M(0.1), O(0.1))` returns a nearest point that misses the true value while the
  operand `O` promised an enclosure -- the review's finding; order-dependent soundness. rejected by the
  review for cause.

### option 3 (not in the plan, recommended): fix the methods so the subclass decides there too; the hook then matches trivially
* meaning: in `MultiInterval`, when `other` is an instance of a proper subclass of `type(self)`,
  the method computes in `other`'s class: `self._wrap` -> `type(other)._wrap`, `outward=type(other)._outward`.
  one helper (`_result_class(self, other)`) used by the ~10 binary methods; symmetric ones can
  simply delegate (`return other.hypot(self)`), asymmetric ones promote `self`
  (`type(other).from_cuts(self.cuts).atan2(other)`, the pattern `_subclass_first` already uses for
  `arctan2`). `numpy_compat.py::_subclass_first` then becomes redundant and can go (or stay as a
  no-op safety).
* pros: README.md:198 becomes true for methods; `np.hypot(M, O) == M.hypot(O)` -- the ufunc *is* the
  method again; `M(5).minimum(O(...))` no longer yields an `M` with outward-rounded open ends; set
  algebra methods and operators agree on the result type (`M.union(O)` vs `M | O`).
* cons: a behaviour change in `MultiInterval` methods (result *type* changes from `M` to `O` when
  mixing, and float ends widen outward) -- exactly the kind of change to make before 2.0, not after;
  touches ~10 methods and their doctests; `DecoratedInterval` wraps `_interval` so it inherits the fix
  (`decorated.py:340` calls `self._interval.hypot(other._interval)`).
* when better: always, given README.md:198; the only reason not to is scope (it is an M16-core change
  filed under a numpy question).

**recommendation: option 3, and keep (h)'s hook rule meanwhile (it is what option 3 makes true).
confidence: high** that the methods contradict README.md:198 (probed); **medium-high** that fixing the
methods is the right response rather than rewording the README to "mixing the two *with an operator*
gives an `OutwardMultiInterval`; a method rounds in its own class". what would change my mind: an owner
statement that `M.hypot(O)` returning `M` is intended ("the receiver's class decides for methods") --
then reword README.md:198 and keep the hook as built, accepting `np.hypot(M, O) != M.hypot(O)` as the
documented exception.

**cost of changing later.** before 2.0: ~10 methods, their doctests, a README sentence, one hypothesis
property ("a method mixing M and O equals the operator's class"). after 2.0: a result-type change in
core methods -- breaking. decide before 2.0.

## summary table

| sub-q | built | recommendation | confidence | change before 2.0? |
|---|---|---|---|---|
| (a) hook vs array API | hook on 3 classes | keep the hook; array type "later, if ever"; document the 1788 layer's `__array_ufunc__ = None` | high | nothing to change |
| (b) foreign reals | exact where `float()` rounds; `Rational` by type | keep | high | nothing; after 2.0 this is frozen |
| (c) `==`/`!=` vs an ndarray | identity (scalar `False`) | **change**: elementwise bool array, as numpy, pandas, lists and every other element type; scalar path unchanged | medium-high | yes: 3 lines + 1 test + README:193 |
| (d) numpy-name aliases | none | keep none; add `rint` only if `np.round(A)` complaints arrive | medium | additive, can wait |
| (e) `np.invert` | complement `~A` | keep | high | nothing |
| (f) `fmin`/`fmax` | TypeError | **change**: map to `minimum`/`maximum`; a nan is the library's ValueError | medium | additive, safe either side |
| (g) numpy in `[test]` | no; README prose | **change**: add numpy to `[test]` (gmpy2's own rationale), README numpy section as doctests, drop the `[test]` half of `::test_numpy_is_never_imported_at_load` | medium-high | packaging only, any time |
| (h) both ours, method ufunc | subclass decides in the hook | keep the hook rule; **fix the methods** (`M.hypot(O)` etc. return `M` today, against README.md:198) so the ufunc equals the method again | high (defect) / medium-high (fix) | yes: core result-type change |

probe artefacts: `.scratch/owner-questions/probes-numpy/probe1.py`, `probe2.py` and their `.out`
files (2026-10-03). env: numpy 2.5.2, python 3.13.15, gmpy2 2.3.1, pandas 3.0.6, windows (longdouble 64-bit).
