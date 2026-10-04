# verify datetime-40 (docs lens): `__getitem__` slice with a step

claim: v1 `time_interval.py:224-225` raises ValueError on a slice step; v2 raises TypeError('slice step is not supported').
claimed verdict UNDOCUMENTED_DIFFERENCE.

## probe
file: .scratch/v1-parity/verify/datetime-40-step-probe.py
cmd: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/datetime-40-step-probe.py
    v1 DTI step ValueError slice(datetime(2024,1,1), datetime(2024,1,5), 1)
    v1 MI step TypeError slice(0, 5, 1)          <- v1's own MultiInterval (multi_interval.py:712-713) used TypeError
    v2 DTI step TypeError slice step is not supported
    v2 TDI step TypeError slice step is not supported
    v2 MI step TypeError slice step is not supported
    SANITY: deliberate wrong expectation (ValueError) caught -> TypeError
v2 code: intervals/time_interval.py:457-458 and intervals/multi_interval.py:240-241 (same message).

## doc search (no record names the step or its exception class)
searched v2-implementation-plan.md (D rows, M4 done-record, M8 + review round, M10, §4), v2-plan.md, README.md,
HANDOFF.md (Q21 a-i), references/** (incl. m8-choices-2026-10-04), docstrings in intervals/.
* adjacent only: v2-implementation-plan.md:4907 (§4 row) "`__getitem__` with an interval ... or a scalar ...: slicing
  only, as `MultiInterval`" -- covers the non-slice arguments, not a step.
* adjacent only: v2-plan.md:1241 "slicing reads its bounds the same way"; v2-implementation-plan.md:394 "a missing slice
  bound is closed, as `MultiInterval.__getitem__` (v1 opened it)" -- bounds, not step.
* the `_TimeInterval.__getitem__` docstring (time_interval.py:450-454) does not mention a step.
* Q21 (HANDOFF.md:123-132) lists nothing about slicing.

## verdict
NOT refuted on the docs lens: UNDOCUMENTED_DIFFERENCE (trivial). no capability is lost: v1 refused a step too; only the
exception class changed, and v2's TypeError matches v1's own MultiInterval.__getitem__ (TypeError), i.e. v2 unified an
inconsistency in v1. a one-line note in §4's time_interval row would close it.
