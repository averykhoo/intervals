# owner answers 2026-10-03: numpy.md (Q15 c f g h) + misc.md §3 (Q6-shift) -- implementing agent's record

worktree: .claude/worktrees/agent-ae88ba38db93402e1, branch worktree-agent-ae88ba38db93402e1,
fast-forwarded to master 09435ca at start (it was at 912558b, without the references/ reports).

## plan (2026-10-03)

* Q15(h): every MultiInterval method taking other sets (union intersection difference symmetric_difference
  minimum maximum fma cancel_minus cancel_plus hypot atan2) computes in the class the operators would pick:
  a decorator `multi_interval.py::_subclass_decides` promotes the receiver to an operand's proper subclass
  and calls that class's method. this REVERSES a documented rule: v2-plan "methods on the receiver's class"
  (decision-log entries M13f 2026-09-26 line ~1955 and M13d ~1928, current design "cancellation" ~374),
  pinned by tests/test_outward.py::test_the_class_is_closed (receiver list asserted M) and
  tests/test_functions.py:507 (`type(M(1).hypot(O)) is M`). those pins flip. decision-log entries are left
  for the orchestrator.
* DecoratedInterval: its methods call the core's, so they inherit; but `_set_operation` called
  `MultiInterval.__and__/__or__/__xor__` UNBOUND on (a._interval, b._interval), bypassing python's
  reflected rule: `D(M) & D(O)` was M's. same defect, fixed with operator.and_/or_/xor.
* Dual: no binary methods; its operators use the parts' operators (python's rule). nothing to fix.

## step 1: Q15(h) built (2026-10-03)

changed:
* `intervals/multi_interval.py::_subclass_decides` (new decorator) on `MultiInterval.union intersection difference
  symmetric_difference minimum maximum fma cancel_minus cancel_plus hypot atan2`; it looks at positional AND keyword
  operands, promotes the receiver (`cls._wrap(self._cuts)`) and calls the deciding class's own method. docstrings:
  module, `union` (new doctest), `fma` (new doctest), `cancel_minus`, `hypot` (new doctest), `OutwardMultiInterval`.
* `intervals/decorated.py::DecoratedInterval.__and__/__rand__/__or__/__ror__/__xor__/__rxor__`: `operator.and_/or_/xor`
  instead of `MultiInterval.__and__` etc. unbound (the same defect: `D(M) & D(O)` held an M).
* `intervals/numpy_compat.py::_subclass_first` removed: the method now does it, so the hook is literally the
  method (`np.hypot(M, O)` is `M.hypot(O)`); module docstring reworded.
pins (new or flipped), each red on the old code (new tests run in a detached worktree of 09435ca, `.scratch/old`):
| pin | old code |
|---|---|
| tests/test_outward.py::test_the_class_is_closed (receiver list flipped M->O, plus value == the method on the receiver read as O; `MIXED_METHODS` incl. fma by keyword) | red |
| tests/test_outward.py::test_a_mixed_method_is_the_outward_method (new; hypot value, union/intersection vs `| &`, M(5).minimum(O(0.1)+0.2)) | red |
| tests/test_functions.py::test_methods_keep_the_class (line 507 flipped: `M(1).hypot(O)` is O) | red |
| tests/test_propagation.py::test_mixing_the_classes_is_the_cores_rule[19 params] (new) | red for `& | ^` and all 12 methods (15); green for `+ - * /` (4), which were right already |
| tests/test_numpy_compat.py::test_both_ours_method_ufunc_is_the_method (renamed from ::test_both_ours_methods_are_subclass_first; ufunc == direct method, both O) | red |
| tests/test_numpy_compat.py::_binary_oracle: symmetric/arctan2 oracle is now the plain method (no subclass swap) | a chance catch only (hypothesis draw) |
sabotage of the new code (in `.scratch/old` with the new files copied in):
| break | result |
|---|---|
| S2: deciding wraps the receiver's own result in the subclass (`cls._wrap(method(self, ...)._cuts)`: right class, nearest rounding) | red: test_the_class_is_closed, test_a_mixed_method_is_the_outward_method, test_both_ours_method_ufunc_is_the_method; propagation test green (it compares D with the core, both broken alike) |
| S3: deciding ignores keyword operands | red: test_the_class_is_closed (the `fma(addend=x, factor=s)` lambda) |
runs: tests/test_outward.py tests/test_functions.py tests/test_propagation.py tests/test_numpy_compat.py + doctests of
multi_interval.py decorated.py numpy_compat.py: 1101 passed (140 s, 2026-10-03).

## step 2: Q15(c) and Q15(f) built (2026-10-03/04)

changed: `intervals/numpy_compat.py::array_ufunc` (equal/not_equal take the elementwise path with an ndim>0
array, result `.astype(bool)`; the scalar path unchanged), `::_SYMMETRIC` (+ `fmin`->minimum, `fmax`->maximum),
module docstring. `A in arr` follows (ndarray.__contains__ is `(arr == A).any()`); `A in f` for a float array
stays False.
tests (tests/test_numpy_compat.py): `::test_equality_never_broadcasts` replaced by
`::test_equality_against_an_array_is_elementwise`; `ELEMENTWISE` now includes equal/not_equal (expected cast to
bool); `BINARY_METHODS` += fmin/fmax (so the method property and the elementwise property cover them);
fmin/fmax removed from `::test_unmapped_ufuncs_and_forms_are_type_errors`; new `::test_fmin_fmax_are_minimum_maximum`.
red on the pre-step hook (step-1 numpy_compat): 8 failed / 280 passed: binary_ufunc[fmin], [fmax],
test_fmin_fmax_are_minimum_maximum, test_equality_against_an_array_is_elementwise, elementwise[equal],
[not_equal], [fmin], [fmax].
sabotage S4 (drop the bool cast, object array of bools): red: test_equality_against_an_array_is_elementwise,
elementwise[equal], elementwise[not_equal].
run: tests/test_numpy_compat.py + numpy_compat doctests 288 passed (30 s).

## step 3: Q15(g) built (2026-10-04)

changed: `pyproject.toml` `[project.optional-dependencies] test` (only that line, now a 4-line array with a
comment; the `fast = [...]` line untouched), `.github/workflows/ci.yml:31` and `fuzz.yml:46` lose the trailing
` numpy` (ci.yml:55, the exhaustive job's plain `.[test]` install, now gets numpy too), README "numpy" bullet
rewritten with a doctest block (np.float32, arcsin, hypot both orders == the method, fmin, linspace elementwise,
`arr == MI(3)` bool array and `MI(3) in arr`, frompyfunc recipe), README "rounding": methods mix too.
tests: `tests/test_numpy_compat.py::test_numpy_is_never_imported_at_load` keeps only the `dependencies` half
(helper `::_project`); new `::test_the_gate_needs_numpy` (numpy in `[test]`); module docstring.
red on old: `::test_the_gate_needs_numpy` with the old pyproject (assert []); README's new doctests on the
original library: red at the hypot line (the method was M's).
runs (2026-10-04): every test file using the changed methods (test_applicator backend cancel extreme_floats
extreme_floats_functions ieee1788_layer kernel minmax_fma multi_interval oracle_flint pow_rev relations reverse
solver itf1788 decorated autodiff solve): 20483 passed (462 s). README + packaging checks: 3 passed.

docs (step 1-3): v2-plan.md current design: "arithmetic" rounding bullet (methods mix, `_subclass_decides`),
"cancellation" rounding (class is the operators', not the receiver's), "numpy" (test extra, fmin/fmax, method rule,
== elementwise, object-array bullet, pandas bullet), "testing" numpy bullet. decision-log entries M13d (~1928
"hypot ... its class is the receiver's") and M13f (~1955 "methods on the receiver's class") now describe a
superseded rule: left untouched for the orchestrator's revision entry. v2-implementation-plan.md (D23 row,
M16d records) untouched likewise.
run before commit (2026-10-04): README + test_numpy_compat + test_outward + test_propagation + test_functions +
every intervals/ module's doctests: 1180 passed (153 s).
COMMIT f847334 "Q15(c,f,g,h): ..." on worktree-agent-ae88ba38db93402e1.

## step 4: Q6-shift built (2026-10-04)

changed:
* `intervals/multi_interval.py::MultiInterval.__lshift__/__rshift__` (`self._binary(scale, ops.mul)`, so class,
  openness, rounding and warnings are `*`'s: `M() << 3` warns EmptySetPropagationWarning as `M() * 8` does),
  `::__rlshift__/__rrshift__` raise TypeError via `::_refuse_shift_count`; `::_power_of_two(n, sign)` (Integral but
  bool, `int(n)` so numpy ints; `1 << n` or `Fraction(1, 1 << -n)`; no cap: `A << 2**63` is python's MemoryError,
  fast, as `1 << 2**63`). OutwardMultiInterval needs no override (the reflected forms only refuse).
* `intervals/decorated.py::DecoratedInterval.__lshift__/__rshift__/_scaled/__rlshift__/__rrshift__`: decorated as the
  product with a point: `_propagate(result, (self,), _everywhere(self))` (trv if inf attained, com iff bounded).
* `intervals/autodiff.py::Dual.__lshift__/__rshift__/__rlshift__/__rrshift__`: `Dual(u << n, u' << n)`. NOTE: not
  identical to `x * 2**n` in decoration past the doubles: the product rule computes `u' 2**n + u 0`, whose
  `inf + 0` is trv where the scaling alone is dac (same sets). the test pins the scaling definition and the
  product's sets.
* `intervals/numpy_compat.py::_OPERATORS` += `left_shift`, `right_shift`.
SURPRISE 1: tests/test_numpy_compat.py's derivation regex `__r[a-z]+__` matched the FORWARD `__rshift__` (name
'shift', no `operator.shift`) and crashed collection; fixed: a reflected dunder needs its forward one in dir(cls).
SURPRISE 2: `::test_numpy_scalar_operators_are_python_numbers[*-lshift/rshift]` does NOT go red without the
`_OPERATORS` rows (measured: 19 passed): the reflected shifts refuse, so `np.int64(3) << A` is a TypeError either
way (numpy's message vs ours), and `A << np.int64(3)` runs `A.__lshift__` first. the table rows are pinned by
`::test_binary_ufunc_is_the_method_or_the_operator[left_shift/right_shift]` (BINARY_OPERATORS += the two),
`::test_ndarray_and_interval_elementwise[left_shift/right_shift]` and `::test_integer_arguments_take_numpy_ints`.
also `::test_numpy_scalar_operators_are_python_numbers` skips shifts by a numpy int past 64 (as pow): a 2**63-bit scale.
tests added: tests/test_multi_interval.py::test_shift_examples (13 rows from misc.md §3),
::test_shift_of_the_empty_set_warns_as_a_product, ::test_shift_is_scaling_by_a_power_of_two (both classes, any cuts,
n in -80..80 + -1100 -1075 -1074 1024 1100; type, repr and warnings vs `*`), ::test_shift_of_exact_ends_moves_each_cut
(independent oracle: each cut value * 2**n, and `(A << n) >> n == A`), ::test_shift_refusals;
tests/test_propagation.py::test_a_shift_is_decorated_as_the_product (+ @example inf attained),
::test_shift_examples_and_refusals; tests/test_autodiff.py::test_a_shift_is_the_product;
tests/test_numpy_compat.py::test_the_operators_are_derived (+lshift rshift), BINARY_OPERATORS, integer_arguments rows.
sabotage (sandbox `.scratch/old` with the new code, one break at a time, `.scratch/breaks.py`):
| break | red |
|---|---|
| Sh1 `>>` floors (`modulo.floordiv`, R2) | 24: shift_examples x6, scaling x2, exact cuts x2, both propagation tests, autodiff x12 |
| Sh2 negative count refused for `<<` | 9: examples[3], scaling x2, exact x2, propagation property, autodiff[-3] x3 |
| Sh3 bool count accepted | 16: refusals (core, decorated), autodiff x12, numpy python_numbers[bool_-lshift/rshift] |
| Sh4 `__rlshift__` answers (returns self) | 3: test_shift_refusals, propagation refusals, numpy integer_arguments |
| Sh5 `_OPERATORS` rows dropped | 5: binary_ufunc[left_shift/right_shift], elementwise[left_shift/right_shift], integer_arguments |
| Sh6 decorated shift ignores the domain | first only `::test_shift_examples_and_refusals` (the property drew no attained inf in 30 examples): added the `@example`; then both red |
| Sh7 Dual derivative not scaled | 9: autodiff[-3, 2, 1100] x3 (n=0 green: scale 1) |

docs (step 4): README top doctest `print(MI(3) >> 1, MI(3) // 2)` -> `[3/2] [1]` beside "int and Fraction stay
exact"; README new "shifts" bullet after "arithmetic"; README autodiff bullet lists `<< >>`; v2-plan current design
"arithmetic": new "shifts" bullet after "power"; "numpy" ufuncs bullet: left_shift/right_shift sentence;
decorated.py module docstring lists `<< >>`. docstrings: MultiInterval.__lshift__ (doctests), __rshift__,
DecoratedInterval.__lshift__ (doctest), Dual.__lshift__/__rshift__.
README + doctests of multi_interval decorated autodiff numpy_compat: 47 passed (2026-10-04).

## final (2026-10-04)

full suite `python -m pytest -q -p no:cacheprovider` on the step-4 tree (before one docstring-only rewording in
decorated.py, re-run after: 12 passed): 33919 passed, rc 0, 873 s. NOT through tools/gate.py (per instructions:
the orchestrator runs the recorded gate after merging).
COMMITS on worktree-agent-ae88ba38db93402e1 (based on master 09435ca): f847334 (Q15 c f g h), 0513109 (Q6-shift).
sandbox worktree .scratch/old removed.
left for the orchestrator: HANDOFF.md; decision-log entries in v2-plan.md (and the superseded "methods on the
receiver's class" entries, M13d ~1928 / M13f ~1955, which now describe a reversed rule); D rows in
v2-implementation-plan.md §0 (D23 row says (c) never broadcast, (f) not mapped, (g) not in [test], (h) hook only);
the M16d records in v2-implementation-plan.md §2 cite `::test_both_ours_methods_are_subclass_first`,
`::test_equality_never_broadcasts` and `numpy_compat.py::_subclass_first`, all renamed or removed now.
not done: nothing asked was skipped. not touched: ieee1788.Interval (its own rule, open item 6); pandas untested.
