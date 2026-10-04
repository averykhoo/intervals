import sys; sys.path[:0] = ['.', 'archive/v1']
import time, datetime as dt, warnings
print(time.tzname, time.timezone, time.daylight)
print(dt.datetime(2024,1,1).timestamp(), dt.datetime(2024,7,1).timestamp() - dt.datetime(2024,1,1).timestamp() - 182*86400)
from intervals.time_interval import TimeDeltaInterval as T2
H = dt.timedelta(hours=1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r = T2(H, 2*H) / 0
    print(repr(r), [str(x.message)[:100] for x in w])
    r = T2(-H, 2*H) / 0
    print(repr(r), [str(x.message)[:100] for x in w])
