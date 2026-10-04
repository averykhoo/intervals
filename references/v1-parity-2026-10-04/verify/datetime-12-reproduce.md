# datetime-12: DTI(None, t) = point t -- reproduce and find a way

claim: UNDOCUMENTED_DIFFERENCE (v1 DTI(None, t) is the point t; v2 raises ValueError 'an end without a start').

probe: .scratch/v1-parity/verify/datetime_12_none_end.py
command: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/datetime_12_none_end.py

key output (2026-10-04):
    v1 DTI(None,t) -> [2024-01-01 10:00]
    v2 DTI(None,t) -> raise ValueError an end without a start
    v2 DTI(t) -> DateTimeInterval(datetime.datetime(2024, 1, 1, 10, 0))
    v1 MI(None,5) -> raise ValueError            <- v1's OWN numeric class refused an end without a start
    v2 MI(None,5) -> raise ValueError an end without a start
    v1 DTI(None,t,start_closed=False) -> raise ValueError ; v2 DTI(t,start_closed=False) -> raise ValueError
    sweep 1200 checks 0 diffs   (300 seeded random t, membership at t, t-1us, t+1us, t+1h: v1 DTI(None,t) vs v2 DTI(t))
    sabotage caught: True       (a point 1 us off is detected)

reproduced: yes, the spelling DTI(None, t) is refused by v2.

why v1 accepted it: archive/v1/time_interval.py:60-63 is the NaT/nan branch,
`if pd.isna(end): end = None` / `if pd.isna(start): start, end = end, None`. pd.isna(None) is True, so None
rides the same swap as NaT. It is not a designed "None start" feature: v1's MultiInterval(None, 5) raises
ValueError, so v2's DTI matches v1's numeric-class rule.

documentation: v2-implementation-plan.md:411-412 (M8, D30's decided-with): "`NaT` or nan in the constructor
raises (v1 dropped it: `DateTimeInterval(NaT, t)` was the point t)". That records dropping exactly this v1
branch (lines 62-63), though it names NaT and not None. v2's docstring (intervals/time_interval.py, class
DateTimeInterval): "`DateTimeInterval()` is empty, `DateTimeInterval(t)` the point t (a `date`: its day)".

way in v2: DateTimeInterval(t) -- ran, agrees with v1 DTI(None, t) on every membership probe. (A date t gives
v2's half-open day per D30(c), the documented no-snap departure, not this row's issue.)

verdict: REFUTED as a gap. corrected: DROPPED_DOCUMENTED (the swap branch is recorded as not ported; the
point is still DateTimeInterval(t), verified). Residual nit: the plan line names NaT only; adding "or None"
would make it literal.
