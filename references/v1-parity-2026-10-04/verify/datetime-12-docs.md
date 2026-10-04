# verify datetime-12 (docs lens): DTI(None, t) = point t in v1, ValueError in v2

verdict: NOT refuted. corrected_verdict UNDOCUMENTED_DIFFERENCE (minor; workaround DTI(t) exists and agrees).

## run
probe: .scratch/v1-parity/verify/datetime-12-probe.py, datetime-12-probe2.py
cmd: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/datetime-12-probe.py
    v1 DTI(None,t) ok [2024-01-01 10:00]
    v1 MI(None,1) raise ValueError
    v2 DTI(None,t) raise ValueError an end without a start
    v2 MI(None,1) raise ValueError an end without a start
    v2 DTI(t) ok [2024-01-01 10:00:00]
probe2: v1 DTI(None,t).interval == v1 DTI(t).interval -> True ([1704074400.0]); so v2's DTI(t) reproduces it.

## why v1 did it
archive/v1/time_interval.py:59-63: `if pd.isna(start): start, end = end, None`. pd.isna(None) is True, so a None
start takes the same swap branch as NaT/nan. v1's own MultiInterval (archive/v1/multi_interval.py:151-153) raised a
bare ValueError for an end without a start; v2 applies that MultiInterval rule to the time layer
(intervals/multi_interval.py:95-97, intervals/time_interval.py:779-781, 1010-1012).

## records searched
* v2-implementation-plan.md:411-412 (M8): "`NaT` or nan in the constructor raises (v1 dropped it:
  `DateTimeInterval(NaT, t)` was the point t)". ADJACENT: same v1 lines and the same swap mechanism, but it names
  NaT/nan, not None.
* references/m8-choices-2026-10-04/report.md:244-246: "`pd.NaT`/`nan` in the constructor. v1 silently dropped them
  (`pd.isna(end)` -> `None`, so `DateTimeInterval(NaT, t)` became the point `t`, lines 59-63) ... Recommend raising."
  ADJACENT, same: NaT/nan only.
* intervals/time_interval.py DateTimeInterval docstring (~757): "`DateTimeInterval()` is empty, `DateTimeInterval(t)`
  the point t ... and `DateTimeInterval(a, b, ...)` one piece". lists the accepted forms, does not mention (None, t).
* v2-implementation-plan.md §4 surface map, time_interval row (line 4885+23): renames and the M8-review drops
  (td - dt, __getitem__ with interval/scalar, inf/sup of empty); nothing on a None start.
* README.md time section (227-240): NaT refused; nothing on None.
* HANDOFF.md Q21 (a)-(i): nothing on the constructor's None start.
* v2-plan.md: only `Interval(None, hi)` (line 881), the 1788 layer, not the time layer.
* grep "end without|without a start|None, t|isna" over *.md: no further hit.

## conclusion
No record names DTI(None, t). The NaT/nan decision is the nearest (same v1 branch), and v2 is consistent with v1's
own MultiInterval rule, so the difference is defensible and almost certainly intended, but it is not recorded.
Fix would be one clause in plan §4's time row or in the M8 NaT line: "a None start with an end raises, as
MultiInterval's (v1 swapped it to the point t via pd.isna)".
