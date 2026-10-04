import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import time_interval as v1t
import multi_interval as v1
from intervals.time_interval import DateTimeInterval as D2, TimeDeltaInterval as T2
import intervals as v2

a, b = dt.datetime(2024,1,1), dt.datetime(2024,1,5)
def run(label, f):
    try:
        r = f(); print(label, 'RESULT', r)
    except Exception as e:
        print(label, type(e).__name__, repr(e)[:120])
x1 = v1t.DateTimeInterval(a, b)
x2 = D2(a, b)
for step in (1, dt.timedelta(days=1), 0, None):
    run(f'v1 DT step={step!r}', lambda: x1[a:b:step])
    run(f'v2 DT step={step!r}', lambda: x2[a:b:step])
td1 = getattr(v1t, 'TimeDeltaInterval', None)
if td1:
    run('v1 TD step', lambda: td1(dt.timedelta(1), dt.timedelta(3))[dt.timedelta(1):dt.timedelta(2):1])
run('v2 TD step', lambda: T2(dt.timedelta(1), dt.timedelta(3))[dt.timedelta(1):dt.timedelta(2):1])
run('v1 MI step', lambda: v1.MultiInterval(0, 5)[0:3:1])
run('v2 MI step', lambda: v2.MultiInterval(0, 5)[0:3:1])
# sanity: a deliberately wrong expectation must be caught
try:
    x2[a:b:1]; print('SABOTAGE: v2 accepted a step (unexpected)')
except ValueError:
    print('SABOTAGE FAIL: expected ValueError would have been wrong-caught')
except TypeError:
    print('sabotage check OK: v2 raises TypeError, not ValueError, so a ValueError-expecting probe fails')
