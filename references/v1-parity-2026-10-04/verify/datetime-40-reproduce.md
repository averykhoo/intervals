# datetime-40: `__getitem__` slice with a step (claim: UNDOCUMENTED_DIFFERENCE) -- REFUTED

probe: .scratch/v1-parity/verify/datetime-40-probe.py
command: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/datetime-40-probe.py

key output:
    v1 DT step=1 ValueError ValueError(slice(datetime(2024,1,1), datetime(2024,1,5), 1))
    v2 DT step=1 TypeError TypeError('slice step is not supported')
    (same for step=timedelta(days=1), step=0)
    v1 DT step=None RESULT [2024-01-01 to 2024-01-05 00:00]   / v2 same set
    v1 TD step TypeError "'TimeDeltaInterval' object is not subscriptable"   (v1 TD had no __getitem__ at all)
    v2 TD step TypeError 'slice step is not supported'
    v1 MI step TypeError TypeError(slice(0, 3, 1))   (archive/v1/multi_interval.py:712-713)
    v2 MI step TypeError 'slice step is not supported'
    sabotage: a ValueError-expecting check against v2 fails (v2 raises TypeError) -- probe can fail.

reproduced: yes, the exception class changed ValueError -> TypeError for DateTimeInterval only.

why it is not a gap:
* no capability: v1 never did anything with a step on any class; both versions refuse every step
  (including step=0, step=1, a timedelta step). there is no v1 result to reproduce.
* v1 was inconsistent with itself: v1 MultiInterval.__getitem__ raised TypeError for a step
  (multi_interval.py:712-713), v1 DateTimeInterval raised ValueError (time_interval.py:224-225), v1
  TimeDeltaInterval raised TypeError (not subscriptable). v2 unifies all three on TypeError, the same
  text in intervals/multi_interval.py:240-241 and intervals/time_interval.py:457-458.
* docs: v2-implementation-plan.md §4 time_interval row: "`__getitem__` with an interval ... or a scalar
  ...: slicing only, as `MultiInterval`" -- the time layer's slicing is defined as MultiInterval's, which
  refuses a step with TypeError in both v1 and v2. not an explicit line about the step's exception class.

verdict: refuted. corrected_verdict EQUIVALENT_RENAMED (the refusal is kept; the exception class aligns with
v1's own MultiInterval). at most a cosmetic exception-class note, not a lost capability.
