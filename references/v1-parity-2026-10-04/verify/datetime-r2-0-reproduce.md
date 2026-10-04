# verify (reproduce and find a way): datetime, truthiness bool(DateTimeInterval) / bool(TimeDeltaInterval)

claimed: UNDOCUMENTED_DIFFERENCE. result: **survives (not refuted), UNDOCUMENTED_DIFFERENCE, low severity**, with two corrections.

## reproduction (own probe)
probe: `.scratch/v1-parity/verify/datetime-r2-0/probe_truthiness_r2.py`
command: `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/datetime-r2-0/probe_truthiness_r2.py`
* v1 DTI/TDI define neither `__bool__` nor `__len__` (`vars(cls)` and `hasattr`): object default, always True.
* `D()`, `T()`, disjoint intersection, `A.difference(A)`, `T` disjoint intersection, `D() + td`, `D() - dt`, `T() + td`:
  v1 bool True with is_empty True; v2 bool False with is_empty True.
* seeded sweep (seed 20261004), 492 random DTI set ops (intersection/difference/union, open/closed ends):
  is_empty disagreements 0, v1 bool False 0, v2 `bool != not is_empty` 0, v2 False 118,
  **v1 `bool(x.interval)` != v2 `bool(x)`: 0**.
* sabotage (expect v2 bool == v1 bool) caught on 5 hand cases.
* note: v2 `D - D` is `dt - dt` arithmetic (a TimeDeltaInterval), as v1's `__sub__`; set difference is `.difference` on
  both sides (a first draft of my probe mixed them and showed 11 spurious disagreements).

## finding a way
* v1's result (constant True) needs no spelling: `x is not None`. v1's only emptiness test, `x.is_empty`, is EQUAL.
* v1's own `bool(x.interval)` (the v1 MultiInterval inside, `multi_interval.py:1849` `not self.is_empty`) equals v2's
  `bool(x)` on every case: v2 lifts v1's core truthiness to the wrapper. no capability is lost.
* v1 is not "wrong" in the exact sense (no unsound answer, no crash); its always-True is an omission, inconsistent
  with v1's own MultiInterval (the DTI docstring: "refer to MultiInterval for more detailed explanations").

## corrections to the claim
1. "no test pins it" is wrong: `tests/test_time_interval.py::test_tz_dates_property` uses `a.tz is (tz if a else None)`
   and `if a:`. sabotage `.scratch/v1-parity/verify/datetime-r2-0/sabotage_bool_pin.py` (in-process patch
   `_TimeInterval.__bool__ = lambda self: True`, tracked files untouched): unpatched PASS, patched FAIL AssertionError.
   the pin is incidental (DateTimeInterval only, tz dates), not a named pin of truthiness.
2. documentation: still nothing for the time layer. `v2-plan.md:2688` "`__bool__` = non-empty, explicitly (set
   precedent)" and `:111` are the core class; the time layer section (`v2-plan.md` "the time layer (M8...)") lists set
   ops, relations, comparisons and arithmetic as the numeric class's, iteration/pieces, but not bool/len; plan §4's
   time_interval row and HANDOFF Q21 (a)-(i) do not mention it. `len()` (v1 TypeError, v2 number of pieces) is an
   addition, also unrecorded.

## minimal reproduction
`bool(time_interval.DateTimeInterval())` v1 True; `bool(intervals.DateTimeInterval())` v2 False (same for TimeDeltaInterval).
a v1 `if spans:` on an empty result takes the other branch. fix: one line in §4's time row (or the time layer's
"thin wrapper" bullet) plus an explicit pin.
