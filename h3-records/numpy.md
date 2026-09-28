# H3's second part, stream M16d: numpy interop (record, 2026-09-28)

the orchestrator merges these sections into `v2-plan.md`, `v2-implementation-plan.md`, `HANDOFF.md`
and `README.md`; the stream edits none of them. the owner, 2026-09-27: "get the rest of h3 done",
which supersedes 2026-09-26's "numpy and gmpy2/mpfr recorded, not now" (`HANDOFF.md` H3 row,
`v2-plan.md` "later (not in v2.0)"). ids: milestone M16d, decision D23, owner question Q15.

## design

text for `v2-plan.md` "current design", a new section after "the solver stack":

### numpy (M16d, H3's second part, 2026-09-28)

numpy is optional: never imported at load (`intervals/numpy_compat.py` imports numpy inside the two
hooks, which only numpy calls), not in `[project]` dependencies nor the `[test]` extra (CI's jobs
install it beside the extra). a multi-interval is a *scalar* to numpy, one value of a number-like
type, never an array of numbers, so interop is four rules and one refusal:

* **a numpy scalar is a python number** to every op: numpy registers `np.floating` as
  `numbers.Real` and `np.integer` as `numbers.Integral`, so a numpy scalar goes through `_coerce`
  and `cuts.py::normalize_value` like any python number. `np.float64` is a `float`, rounded to
  nearest in `MultiInterval` and outward in `OutwardMultiInterval`; `np.float32` and `np.float16`
  are the doubles they hold, exactly (`np.float32(0.1)` is `0.10000000149011612`); the numpy ints
  are ints; `np.bool_` and `np.complex*` are refused, as python's bool and complex
* **a foreign real is its exact value** (`cuts.py::normalize_value`), never `float()` of it. a
  foreign real is a `numbers.Real` that is no int, float or Fraction: numpy's scalars, gmpy2's
  numbers. a `numbers.Rational` (gmpy2's `mpq`) is exact by type, as a `Fraction` is; any other
  real is the float it equals where it is a double (so nothing a double can hold changes type), else
  its exact `as_integer_ratio()` as a `Fraction` or an int. so an `np.longdouble` wider than a
  double (x86-64 linux, where CI runs: 64-bit significand) and a wide `mpfr` are exact, where
  `float()` rounded them to nearest and an `OutwardMultiInterval` result did not hold its operand
  (and `np.longdouble('1e4000')` became `inf`). the float path pays one `isinstance(value, float)`
* **integer arguments take any `Integral` but bool**, as `ops.py::power` does: `rootn(n)`,
  `round(ndigits)`, `round_ties_away(ndigits)`, `pown_rev(c, n)` (`functions.py::_check_degree`,
  `steps.py::step`, `reverse.py::pown_rev`), and the `log` base takes any real but bool, through
  `normalize_value` (`functions.py::_check_base`). `Dual.rootn` takes `int(n)` before `n - 1`
* **ufuncs** (`MultiInterval`, `OutwardMultiInterval`, `DecoratedInterval`, `Dual`:
  `__array_ufunc__ = numpy_compat.array_ufunc`): a numpy scalar on the left of an operator reaches
  numpy's ufunc and so the hook, which runs python's protocol **on our dunders only**
  (`np.float64(2) + A` is `A.__radd__(np.float64(2))`, the call python made before the hook
  existed; both operands ours, python's own operator, subclass first, so `np.add(M, O)` is `M + O`,
  an `OutwardMultiInterval`); `==` and `!=` fall back to identity, as python's do
  (`np.float64(2) == M(2)` is False). the other ufuncs are the method computing the set image of
  the ufunc's pointwise function: `sqrt cbrt exp exp2 expm1 log log2 log10 log1p sin cos tan sinh
  cosh tanh floor ceil trunc sign reciprocal` the method of that name, `arcsin ... arctanh` the
  1788 names `asin ... atanh`, `rint` `round` (ties to even, as numpy's), `square` `x ** 2` (pown,
  not `x * x`), `minimum maximum hypot` the method of whichever operand is ours, `arctan2(y, x)`
  `y.atan2(x)`, and the unary operators (`negative positive absolute fabs`, `invert` the complement
  `~`). a class without the method (`Dual` has no `floor`) is a TypeError. **not mapped**, so a
  TypeError: `fmod` (C's truncated remainder, `fmod(-7, 2)` is -1 where `-7 % M(2)` is `[1]`),
  `fmin`/`fmax` (numpy's point is to ignore a nan operand; a nan is refused here), `float_power`,
  `deg2rad` and the like, `isnan` and the other predicates of a point, `matmul`, every other ufunc,
  every ufunc method but `__call__` (`reduce`, `outer`, `accumulate`, `at`) and every keyword
  (`out=`, `where=`, `dtype=`, `casting=`). the table is keyed by the ufunc *objects*, so a foreign
  ufunc sharing a name (scipy's) is not numpy's
* **an ndarray meeting one of ours** (`np.linspace(0, 1, 3) + A`, either order, any ufunc of the
  table with two operands) is elementwise into an object array of the array's shape, each element a
  python number (as `tolist()` gives it) meeting the object on the scalar path; an element with no
  answer raises (a nan element: `ValueError`, as `nan + A`; a bool or complex array: `TypeError`).
  `==` and `!=` never broadcast: `f == A` is False and `A in f` False, as before. a list is not an
  array here: `np.add([1, 2], A)` is a TypeError
* **arrays hold a `MultiInterval` as one element** (`MultiInterval.__array__ = numpy_compat.array`,
  a 0-d object array): `np.array([A, B])` has shape `(2,)` (before, numpy took A for the sequence
  of its pieces: a 64-deep array or a ValueError). `np.asarray(A, copy=False)` is a ValueError.
  `DecoratedInterval` and `Dual` have no `__len__` and were scalars to numpy already. a float dtype
  is numpy's cast, `float()` of each element, **to nearest in both classes**:
  `np.asarray(O(Fraction(1, 3)), dtype=float)` is the double nearest 1/3, as `float(O(...))` is,
  so a degenerate outward set flows into float code rounded to nearest
* **object arrays run numpy's own loops, not the table**: numpy calls the python operator or a
  method named after the ufunc on each element, so on `arr = np.array([A, B])` `arr + 1`,
  `np.sum(arr)`, `np.sin(arr)` work, `np.arcsin(arr)` is a TypeError (no method `arcsin`),
  `np.square(arr)` is `x * x` (looser than `np.square(A)`), and `arr == A` is False and `A in arr`
  False although `arr` holds `A` (identity: `==` is structural and does not broadcast).
  `np.round(A)` and `np.around(A)` are TypeErrors (not ufuncs: numpy's fallback looks for `rint`).
  `np.frompyfunc(MultiInterval.asin, 1, 1)(arr)` or an operand of ours reaches the table
* **the array API standard and `__array_function__` are not built**: the standard is a namespace
  for arrays of fixed-size numbers, with elementwise `bool` comparisons and float special cases;
  ours are ragged sets with structural `==`, `TruthSet` comparisons and set images without nan.
  an interval *array* type is the D23 alternative, "later" if ever

"package layout": `autodiff.py, solver.py, numpy_compat.py` loses "(later)". "later (not in v2.0)":
the numpy line goes; it gains "an interval *array* type (the array API standard's namespace), if
ever; not the numpy interop, which is built (M16d)".

## decision-log revision

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
* found on the way and fixed (not choices): M15's `Dual ** r` computed `r - 1` in r's own
  arithmetic, unsound in the outward class with plain python floats (`Dual.variable(O(1e300)) ** 0.1`
  missed its derivative; `Dual.variable(O(-1)) ** 2.0 ** 60` had its sign flipped); a number
  exponent is now an int (integral) or a point set of u's kind first

## D23

the §0 table row:

| D23 | **decided in the build 2026-09-28 (the session's defaults), open for the owner: `HANDOFF.md` Q15.** numpy interop (M16d): (a) `__array_ufunc__` on `MultiInterval`, `DecoratedInterval`, `Dual` (`intervals/numpy_compat.py`), operator ufuncs as python's operators on our dunders only, the others the method of the same set image, the rest `TypeError`; `__array__` on `MultiInterval` only (a 0-d object array); the array API standard not built (the alternative: an interval-array type); (b) a foreign real is its exact value (a `Rational` by type, else where `float()` would round), alternatives refuse or keep `float()`; (c) an ndarray meeting ours is elementwise into an object array, `==`/`!=` never broadcast; (d) no numpy-named alias methods; (e) `np.invert` the complement; (f) `fmin`/`fmax` not mapped; (g) numpy not in `[test]` | as built | M16d |

## M16d

### M16d numpy interop: `numpy_compat.py` (H3's second part; done 2026-09-28)

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
    arithmetic, rounded to nearest. `Dual.variable(O(u)) ** r` missed `r u^(r-1)` for 6 of 9
    pairs, r in {0.1, 1e-20, 0.3}, u in {2, 1e300, 1e-300} (0.1 with 1e300: the derivative
    `(9.999999999999845e-272, 9.999999999999849e-272)`). the build found the integral case too:
    `2.0 ** 60 - 1` rounds to `2.0 ** 60`, so `Dual.variable(O(-1)) ** 2.0 ** 60` had the
    derivative `+2 ** 60` where it is `-2 ** 60` (a sign; `np.float32(16777218)` the same). the
    solver builds its steps from that derivative. fixed: an integral r is `n = int(r)` and `n - 1`
    int arithmetic; any other r a point set `e` of u's kind and `e - 1` the library's subtraction
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
    `np.errstate(all='ignore')` (found by `::test_ndarray_and_interval_elementwise[divide]`)
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
    `::test_both_ours_is_pythons_rule`, `::test_both_ours_examples` (B3), `::test_ufunc_pinned_examples`
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
    `::test_elementwise_leaves_numpys_float_flags_alone` (a gap the sabotage found)
  * `::test_foreign_reals_are_exact` (stubs `::Wide`, registered `numbers.Real`, and `::Rat`,
    registered `numbers.Rational`, over drawn fractions, ones past the doubles and x86-64's long
    double nearest 1/3), `::test_foreign_real_specials`, `::test_float32_is_the_double_it_holds`
    (every float32 bit pattern drawn), `::test_longdouble_is_exact` (discriminates on linux CI),
    `::test_gmpy2_values_are_exact` (skips where gmpy2 is absent, as on CI)
  * `::test_integer_arguments_take_numpy_ints` (rootn, round, round_ties_away, pown_rev, the
    decorated and `Dual` forms, the log base; bool refused), `::test_dual_rootn_takes_the_degree_as_an_int`
    (B2)
  * `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses` (B1, arb at 400 bits,
    r in {0.1, 1e-20, 0.3} x u in {2, 1e300, 1e-300}), `::test_pow_integral_exponent_derivative_is_exact`
    (r in {2.0 ** 60, 2 ** 60, Fraction(2 ** 60)}, u = -1)
* **measured 2026-09-28** (laptop shared with four other streams' runs, so loaded):
  * `C:/Users/user/anaconda3/envs/intervals/python.exe -m pytest -q tests/test_numpy_compat.py`: 275 passed
    in 49.49 s (numpy 2.5.2, python 3.13, windows)
  * the numpy-missing path, by hand: a meta-path finder raising `ModuleNotFoundError` for numpy,
    then `pytest.main(['-q', 'tests/test_numpy_compat.py', 'tests/test_cuts.py',
    'tests/test_applicator.py'])`: 139 passed, 272 skipped, nothing errored, numpy never imported
    (8.55 s)
  * the gate, from the worktree root, the second call split in two by file for the tool's time
    limit: `python -m pytest -q tests/itf1788` 18246 passed in 78.29 s; `python -m pytest -q
    tests/test_[a-l]*.py` 1449 passed in 287.10 s and `python -m pytest -q --ignore=tests/itf1788`
    with those files ignored 2926 passed in 685.34 s: `--ignore=tests/itf1788` is 4375 passed in
    972.44 s (482 s at M15 unloaded; `--collect-only` over the whole tree: 22621)
  * the hook's cost, `timeit` best of 5 x 2000 on `A = MultiInterval(1, 2)`, the package at
    `04946af` against this build: `np.float64(2) == A` 0.24 µs to 4.70 µs (numpy's override
    machinery now runs, then the identity fallback: it matters for `x in list_of_sets` with numpy
    numbers), `np.float64(2) < A` 8.5 to 13.9 µs; `np.float64(2) + A` 60.6 to 68.6 µs, inside the
    noise (`2.0 + A`, untouched, 53.3 to 73.2 µs in the same runs); `normalize_value(0.1)` 3.0 to
    2.4 µs (the float path: one `isinstance` more, noise); `normalize_value(np.float32(0.1))` 2.0 to
    12.3 µs (a `Fraction` built per foreign value)
* **sabotage** (`.scratch/sabotage.py` in the worktree, the M15 harness adapted: each break alone,
  `.hypothesis` cleared, `tests/test_numpy_compat.py tests/test_autodiff.py tests/test_cuts.py
  tests/test_applicator.py` with `-x` and a 900 s timeout, the file restored and compared;
  2026-09-28). the last column is the first test to fail under `-x`:

| break | first run | final run: red by |
|---|---|---|
| `import numpy` at the top of numpy_compat | red | red: `tests/test_numpy_compat.py::test_numpy_is_never_imported_at_load` |
| `bitwise_or` dropped from the table | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-or]` |
| identity fallback of `equal` removed | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-eq]` |
| `less` reflected as `__lt__` | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float32-lt]` |
| reflected call made with the forward dunder | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-divmod]` |
| 0-d arrays not unwrapped | red | red: `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers[float64-ge]` |
| both ours: forward dunder first, no subclass rule (B3) | red | red: `tests/test_numpy_compat.py::test_binary_ufunc_is_the_method_or_the_operator[remainder]` |
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

the two green in the first run were gaps, each closed by a test added the same session and the break
re-run red: the metaclass lookup (every class answers `hasattr(cls, '__or__')` through
`type.__or__`, so the break only changed which TypeError numpy raised:
`::test_a_missing_dunder_is_numpys_refusal` pins numpy's refusal), and `errstate` (the property test
had found the numpy RuntimeWarning by chance once and then drew past it:
`::test_elementwise_leaves_numpys_float_flags_alone` pins the example). the first run's B1
(non-integral) row was red by `tests/test_autodiff.py::test_pow_number_exponent_derivative_encloses`
alone: the property test 2 did not draw a `Dual` with a float32 exponent in its examples, so
`::test_numpy_exponent_of_a_dual` pins B1 through numpy too (the final run's first red)

## readme

text for `README.md`: one bullet for "what it does", after "interval newton"; prose, no `>>>`,
because the README is one doctest and numpy is not in the `[test]` extra (Q15(g)):

* **numpy** (M16d, optional): a numpy scalar is a python number to every op (`np.float32(0.1)` is
  the double it holds; an `np.longdouble` wider than a double is exact, as any foreign real);
  ufuncs on a set are its methods (`np.sin(A)` is `A.sin()`, `np.arcsin(A)` is `A.asin()`,
  `np.square(A)` is `A ** 2`) or python's operators (`np.add(M, O)` is `M + O`), anything else a
  `TypeError`; an ndarray meeting a set is elementwise into an object array
  (`np.linspace(0, 1, 3) + A`), except `==`, which stays structural; `np.array([A, B])` holds the
  sets as elements. `np.asarray(x, dtype=float)` rounds each point to nearest, in both classes.
  object arrays of sets run numpy's own loops (`np.arcsin(arr)` and `np.round(A)` are TypeErrors)

and in "layout", `numpy_compat` after `solver` in the module list; in "tests": "the numpy tests
skip without numpy; CI installs it beside the extra".

## Q15

owner questions, each built as the stated default (D23):

* **Q15(a) the array API or the hook.** default **`__array_ufunc__` on the three classes**
  (a multi-interval is an element, not an array); alternative: an interval-array type exposing the
  array API standard's namespace, whose dtypes, elementwise bool `==` and float special cases all
  collide with the library's choices (a new type, not interop)
* **Q15(b) foreign reals** (`np.longdouble` on linux, gmpy2's `mpq`/`mpfr`): default **the exact
  value**, a `Rational` exact by type (`mpq(1, 2)` is `Fraction(1, 2)`, as `Fraction(1, 2)` is),
  any other real exact where `float()` would round (a double stays the float); alternatives: refuse
  a foreign real that is not a double (`TypeError`), or keep `float()` (unsound for
  `OutwardMultiInterval`), or the value rule for rationals too (`mpq(1, 2)` a float)
* **Q15(c) an ndarray meeting ours**: default **elementwise into an object array**, `==`/`!=`
  identity as before (`f == A` False; `np.array([A]) == A` False and `A in np.array([A])` False
  although the array holds `A`); alternatives: `TypeError` as before; elementwise `==` (numpy's
  convention, which changes what `f == A` and `A in f` mean today)
* **Q15(d) numpy's names as methods**: default **no aliases** (the 1788 names are the library's),
  so `np.arcsin(object_array)`, `np.round(A)` and `np.around(A)` are TypeErrors; alternative:
  eight aliases (`arcsin arccos arctan arcsinh arccosh arctanh rint arctan2`) on the three classes,
  after which numpy's loops and the table agree except `square`
* **Q15(e) `np.invert(A)`**: default **the complement `~A`** (numpy's `invert` is the `~` ufunc,
  and `np.bitwise_and/or/xor` must mean `& | ^` for `np.int64(1) | A`); alternative: TypeError for
  the explicit unary call only
* **Q15(f) `fmin`/`fmax`**: default **not mapped** (TypeError): their point is a nan operand, which
  the library refuses; alternative: `minimum`/`maximum` with `fmax(A, nan) is A`
* **Q15(g) numpy in the `[test]` extra**: default **no** (CI installs numpy beside it; the numpy
  tests skip without it; the README section is prose); alternative: add it, and write the README
  section as doctests

## still owed

* `O(u) ** n` and `M(u) ** n` for a huge int n (2 ** 60) do not finish (exact `Fraction ** n`),
  pre-existing; the B1 pin for r = `2.0 ** 60` uses u = -1 for that reason. a pown that returns the
  overflow or underflow without the exact power would let the pin take u in {2, 1e300, 1e-300}
* the long double half of `tests/test_numpy_compat.py::test_longdouble_is_exact` discriminates
  only where `np.finfo(np.longdouble).nmant > 52`: CI's linux, never this windows laptop; its first
  CI run is its first real run
* `ieee1788.py`'s new `Interval` (stream M16b) was designed with `__array_ufunc__ = None`; after
  M16d the three core classes have the hook: the orchestrator states the rule for the 1788 layer
* HANDOFF Q6-shift: when `<<`/`>>` land, `tests/test_numpy_compat.py::test_numpy_scalar_operators_are_python_numbers`
  derives them and goes red until `numpy_compat.py::_OPERATORS` gains `left_shift`/`right_shift`
