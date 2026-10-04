"""hand cases of td / T at zero, both sides"""
import sys
sys.path[:0] = ['.', 'archive/v1']
import datetime as dt, warnings
import time_interval as v1t, multi_interval as v1m
from intervals.time_interval import TimeDeltaInterval as T2
T1 = v1t.TimeDeltaInterval
H, Z = dt.timedelta(hours=1), dt.timedelta(0)
for lab, t, a, b, sc, ec in [('1h / [0]', H, Z, Z, True, True), ('0 / [-1h, 1h]', Z, -H, H, True, True),
                             ('0 / (0, 1h]', Z, Z, H, False, True), ('1h / [0, 1h]', H, Z, H, True, True),
                             ('1h / (0, 1h]', H, Z, H, False, True), ('-1h / [-1h, 2h]', -H, -H, 2 * H, True, True)]:
    try:
        r1 = str(v1m.MultiInterval(t.total_seconds()) / T1(a, b, start_closed=sc, end_closed=ec).interval)
    except Exception as e:
        r1 = f'raise {type(e).__name__} {e}'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r2 = t / T2(a, b, start_closed=sc, end_closed=ec)
    print(f'{lab:18} v1 {r1:40} | v2 {r2!r} {[type(x.message).__name__ + ": " + str(x.message)[:60] for x in w]}')
