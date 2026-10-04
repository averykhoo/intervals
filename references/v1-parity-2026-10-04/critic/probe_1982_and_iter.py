import sys, datetime as dt, itertools, time
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import time_interval as v1t
import intervals as v2
import intervals.time_interval as v2t
print('tzname', time.tzname)
a, b = dt.datetime(1981, 12, 31, 23, 0, 0, 1), dt.datetime(1982, 1, 1, 1, 0, 0, 1)
x1 = v1t.DateTimeInterval(a, b); x2 = v2t.DateTimeInterval(a, b)
print('v1 total_seconds', x1.total_seconds, 'v1 total_duration', x1.total_duration)
d2 = x2.total_seconds
print('v2 total_seconds', d2)
print('v1 (b-a) via subtract:', end=' ')
try:
    print(v1t.DateTimeInterval(b) - v1t.DateTimeInterval(a))
except Exception as e: print(repr(e))
# iteration protocol: v1 has __getitem__ but no __iter__
A1 = v1.MultiInterval(1, 2)
print('v1 first 5 from iter():', [str(p) for p in itertools.islice(iter(A1), 5)])
print('v2 list(A):', [str(p) for p in v2.MultiInterval(1, 2)])
assert [str(p) for p in itertools.islice(iter(A1), 2)] != [str(p) for p in v2.MultiInterval(1, 2)], "sabotage: v1 iteration must differ"
