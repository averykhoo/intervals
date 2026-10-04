# timedelta-65: truthiness bool(T()) -- reproduce and refute

claim: UNDOCUMENTED_DIFFERENCE, bool(empty TimeDeltaInterval) v1 True / v2 False.

## reproduction (own probe)
probe: .scratch/v1-parity/verify/td65/probe_bool.py
cmd:   timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/td65/probe_bool.py  (rc=0)
key output:
    v1 TD has __bool__: False __len__: False
    v1 DT has __bool__: False __len__: False
    v1 MI __bool__ on empty: False on [1,2]: True
    T1() is_empty True bool True | v1 inner MI bool False
    T1 [1,2]&[3,4] is_empty True bool True | v1 inner MI bool False
    T1 (1,1] is_empty True bool True | v1 inner MI bool False
    T2() is_empty True bool False
    D1() bool True is_empty True | D2() bool False is_empty True
    sweep n=400: v1 bool!=nonempty 110, v2 bool!=nonempty 0, v1/v2 bool differ 110
    sanity ok: caught: v1 bool(empty) != v2 bool(empty)
The difference is real: v1 time wrappers are always truthy (object default; neither class defines
__bool__ or __len__), v2's are truthy iff non-empty.

## who is right
v1's own core: archive/v1/multi_interval.py:1849 `def __bool__(self): return not self.is_empty`.
The v1 TimeDeltaInterval is a wrapper holding `.interval: MultiInterval` and exposing `is_empty` and a
python-set API (isdisjoint/issubset/clear/pop/...). In every one of the 110 empty sweep cases v1's
`bool(T)` disagrees with v1's own `bool(T.interval)` and with `not T.is_empty`: the wrapper simply
failed to forward its core's __bool__. v1's answer carries no information (constant True), so there
is no v1 capability lost: v1's result is spelled `True` in v2; emptiness truthiness is `bool(A)` /
`not A.is_empty`. Same for DateTimeInterval.

## docs
* v2-plan.md:2688 "`__bool__` = non-empty, explicitly (set precedent)"; v2-plan.md:111 "`__bool__` = non-empty (set precedent)"
* v2-plan.md:1204 time layer: "a thin wrapper over exact seconds ... every set operation, relation,
  comparison and arithmetic op is the numeric class's on those seconds"; intervals/time_interval.py:535
  `def __bool__(self): return bool(self._mi)`.
* not named in the M8 behaviour-change list (v2-plan.md:1594) nor the §4 time_interval row; a one-line
  note for v1 time callers ("`if spans:` was always True in v1") would be a doc nicety, not a gap.

## verdict
refuted: true. corrected_verdict: V1_BUG_FIXED (v1's time wrappers dropped v1's own MultiInterval
__bool__; v2 restores the documented set-precedent truthiness).
