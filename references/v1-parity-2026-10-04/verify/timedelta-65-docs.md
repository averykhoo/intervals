# verify timedelta-65 (docs lens): truthiness bool(T())

claim: v1 TimeDeltaInterval has no __bool__ (object default True); v2 bool(A) == not A.is_empty. UNDOCUMENTED_DIFFERENCE.

## probe (re-run, confirms the claim)
file: .scratch/v1-parity/verify/timedelta-65-probe.py
cmd: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/timedelta-65-probe.py
    v1 TDI defines __bool__/__len__: False False DTI: False False
    v1 empty TDI: bool True is_empty True inner bool False
    v1 empty MultiInterval bool False
    v2 empty TDI: bool False is_empty True
    zero point: v1 True v2 True
    v1 empty DTI bool True v2 False
(the assert bool(v2 empty) is False would catch a wrong expectation.)
same difference for DateTimeInterval.

## docs searched
grep -i '__bool__|truth|bool(|falsy|truthy' over all *.md (v2-plan.md, v2-implementation-plan.md, README.md, HANDOFF.md,
references/**), intervals/time_interval.py docstrings, tests/test_time_interval.py.

adjacent records (general, about MultiInterval, not the time classes):
* v2-plan.md:111 (set operations and size): "`__bool__` = non-empty (set precedent)"
* v2-plan.md:2688 (more v1 retirements): "`__bool__` = non-empty, explicitly (set precedent)"
* v2-implementation-plan.md:143 (M5 dunders): "`__bool__`, ..."
* v2-plan.md:1204 (time layer M8): "every set operation, relation, comparison and arithmetic op is the numeric class's on those seconds"
  (does not name truthiness)

nothing names the time classes' truthiness:
* v2-implementation-plan.md §2 M8 done-record (~l.427-440) lists renames, not-ported and added-from-v2 surface; no __bool__/__len__.
* §4 surface map row `time_interval.py` (l.4907) lists renames and the review-round drops; no truthiness.
* references/m8-choices-2026-10-04/report.md: only TruthSet.__bool__ of comparisons (l.237-239), not the wrapper's own.
* HANDOFF.md Q21 (a)-(i): no truthiness item.
* tests/test_time_interval.py: no bool() of a wrapper pinned.

## verdict
refuted: false. corrected_verdict: UNDOCUMENTED_DIFFERENCE (low severity).
v2's behaviour is consistent with the general recorded design (empty set falsy, set precedent) and with v1's own core
(v1 MultiInterval.__bool__ = not is_empty, multi_interval.py:1849; v1's empty TDI's .interval is falsy), so one could
argue V1_BUG_FIXED (v1's wrappers simply omitted __bool__ and __len__). but no record names the change for the time
classes, and v1's `if span:` was always True, so a v1 caller silently changes behaviour. fix: one line in the §4 row of
`time_interval.py` ("__bool__ = non-empty, as MultiInterval; v1's wrappers were always truthy").
