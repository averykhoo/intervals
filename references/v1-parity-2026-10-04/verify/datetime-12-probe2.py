import sys; sys.path[:0] = ['.', 'archive/v1']
import datetime
import time_interval as v1t
t = datetime.datetime(2024, 1, 1, 10)
a = v1t.DateTimeInterval(None, t); b = v1t.DateTimeInterval(t, t); c = v1t.DateTimeInterval(t)
print(a.interval, b.interval, c.interval, a == b, a == c, a.interval == c.interval)
