import sys, warnings; sys.path[:0] = ['.', 'archive/v1']
import time_interval as v1t, multi_interval as v1
from intervals.time_interval import TimeDeltaInterval as T, DateTimeInterval as D
from intervals import MultiInterval as M
cases = [dict(start_closed=False), dict(end_closed=False), dict(start_closed=False, end_closed=False), {}]
for kw in cases:
    for name, a, b in (('T', v1t.TimeDeltaInterval, T), ('D', v1t.DateTimeInterval, D), ('M', v1.MultiInterval, M)):
        try: r1 = repr(a(**kw).is_empty) if name != 'M' else repr(a(**kw).is_empty)
        except Exception as e: r1 = 'raise ' + type(e).__name__
        try: r2 = repr(b(**kw).is_empty)
        except Exception as e: r2 = 'raise ' + type(e).__name__
        print(name, kw, 'v1:', r1, 'v2:', r2)
# a probe that can fail: wrong expectation must be caught
try:
    assert T(start_closed=False).is_empty is False
    print('SABOTAGE NOT CAUGHT')
except AssertionError: print('sabotage caught')
