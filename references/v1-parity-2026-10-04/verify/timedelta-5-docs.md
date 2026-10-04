# verify timedelta-5 (DOCS lens): T(start_closed=False) with no bounds

claim: v1 raises ValueError, v2 returns the empty set; UNDOCUMENTED_DIFFERENCE.

## code
- v1 archive/v1/time_interval.py:420-444 TimeDeltaInterval.__init__ forwards flags to MultiInterval;
  archive/v1/multi_interval.py:151-155: `if start is None: ... if start_closed != end_closed: raise ValueError; self.endpoints = []`
  (so only MISMATCHED flags raise; both False is empty in v1 too).
- v2 intervals/time_interval.py:1009-1014 (TimeDeltaInterval), :779-783 (DateTimeInterval), intervals/multi_interval.py:95-98:
  start None -> only `end is not None` raises; flags never inspected.

## probe
.scratch/v1-parity/verify/timedelta_5_probe.py; `timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/verify/timedelta_5_probe.py`
  T {'start_closed': False} v1: raise ValueError v2: True
  T {'end_closed': False} v1: raise ValueError v2: True
  T {'start_closed': False, 'end_closed': False} v1: True v2: True
  (same for D and the numeric MultiInterval)   sabotage caught

## docs searched
v2-implementation-plan.md (D rows incl. D30, M4 :137-155, M8 :380-425, M8 review F3 :560-567, §4 surface map :4885+),
v2-plan.md (start_closed hits :1238-1239, :1601 only), README.md, HANDOFF.md Q21, references/*.md, references/m8-choices-2026-10-04/,
docstrings in intervals/, tests/test_time_interval.py.
- adjacent only: v2-implementation-plan.md:565 (M8 review F3) "`D(d, start_closed=False, end_closed=False)` (after d, before d) raises (was empty)"
  and tests/test_time_interval.py:303-311 (T(HOUR, start_closed=False) raises) -- both about a GIVEN start, not the no-bounds call.
- M4 :137 only gives the signature; the docstrings say "`TimeDeltaInterval()` is empty" (time_interval.py:995) without flags.
- nothing names flags-with-no-bounds. not a Q21 item.

## verdict
UNDOCUMENTED_DIFFERENCE survives (not refuted). trivial: no set is lost (v2 is more permissive: a flag on a nonexistent end
is ignored rather than refused). the difference is library-wide (MultiInterval too), not specific to the time layer.
