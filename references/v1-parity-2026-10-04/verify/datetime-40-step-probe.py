import sys, datetime; sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1mi, time_interval as v1t
import intervals as v2
from intervals.time_interval import DateTimeInterval as D2, TimeDeltaInterval as T2
d0, d1 = datetime.datetime(2024,1,1), datetime.datetime(2024,1,5)
def tryit(name, f):
    try: r = f(); print(name, 'OK', r)
    except Exception as e: print(name, type(e).__name__, e)
tryit('v1 DTI step', lambda: v1t.DateTimeInterval(d0, d1)[d0:d1:1])
tryit('v1 MI step', lambda: v1mi.MultiInterval(0, 5)[0:5:1])
tryit('v2 DTI step', lambda: D2(d0, d1)[d0:d1:1])
tryit('v2 TDI step', lambda: T2(datetime.timedelta(0), datetime.timedelta(1))[datetime.timedelta(0):datetime.timedelta(1):1])
tryit('v2 MI step', lambda: v2.MultiInterval(0, 5)[0:5:1])
# wrong-expectation sanity: assert v2 raises ValueError should fail
try:
    D2(d0, d1)[d0:d1:1]
except ValueError: print('SANITY: unexpected ValueError')
except TypeError: print('SANITY: deliberate wrong expectation (ValueError) caught -> TypeError')
