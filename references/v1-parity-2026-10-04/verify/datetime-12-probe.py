import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime
import time_interval as v1t, multi_interval as v1
from intervals import DateTimeInterval as DTI, MultiInterval as MI
t = datetime.datetime(2024, 1, 1, 10)
for label, f in [('v1 DTI(None,t)', lambda: v1t.DateTimeInterval(None, t)),
                 ('v1 MI(None,1)', lambda: v1.MultiInterval(None, 1)),
                 ('v2 DTI(None,t)', lambda: DTI(None, t)),
                 ('v2 MI(None,1)', lambda: MI(None, 1)),
                 ('v2 DTI(t)', lambda: DTI(t))]:
    try: r = f(); print(label, 'ok', r)
    except Exception as e: print(label, 'raise', type(e).__name__, e)
# sanity: v1 point equals v2 DTI(t) as a set
a = v1t.DateTimeInterval(None, t)
print('v1 point is degenerate at t:', a == v1t.DateTimeInterval(t, t))
assert DTI(t) == DTI(t, t)
assert not (DTI(t) == DTI(t, t + datetime.timedelta(seconds=1)))  # a wrong expectation is caught
