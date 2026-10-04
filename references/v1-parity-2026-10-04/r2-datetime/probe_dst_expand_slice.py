"""expand() and slicing across a DST change, naive and aware, v1 vs v2. RUN IN WSL with TZ=Europe/London."""
import os, sys, datetime as dt
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'stub'), '.', 'archive/v1']
import time_interval as v1t
del sys.modules['pandas']
import intervals.time_interval as v2t

Z = ZoneInfo(os.environ['TZ'])
UTC = dt.timezone.utc
US = dt.timedelta(microseconds=1)
H2 = dt.timedelta(hours=2)
a, b = dt.datetime(2024, 3, 31, 2, 30, 0, 5), dt.datetime(2024, 3, 31, 4, 0, 0, 5)   # after the 01:00 GMT -> 02:00 BST jump
e1 = v1t.DateTimeInterval(a, b).expand(H2)
e2 = v2t.DateTimeInterval(a, b).expand(H2)
print(f'naive [{a}, {b}].expand(2h): v1 [{e1.infimum}, {e1.supremum}]  v2 [{e2.inf}, {e2.sup}]  python wall [{a - H2}, {b + H2}]')
ok_naive = e2.inf == a - H2 and e2.sup == b + H2 and (e1.infimum, e1.supremum) == (a, b)   # v1 expand is a no-op (known, datetime/report.md)
aa, ab = a.replace(tzinfo=Z), b.replace(tzinfo=Z)
f1 = v1t.DateTimeInterval(aa, ab).expand(H2)
f2 = v2t.DateTimeInterval(aa, ab).expand(H2)
print(f'aware expand(2h): v1 [{f1.infimum.astimezone(UTC)}, {f1.supremum.astimezone(UTC)}]  v2 [{f2.inf.astimezone(UTC)}, {f2.sup.astimezone(UTC)}]')
ok_aware = f2.inf == aa.astimezone(UTC) - H2 and f2.sup == ab.astimezone(UTC) + H2   # v2 aware expand is elapsed
# slicing with naive bounds across the change
big1 = v1t.DateTimeInterval(dt.datetime(2024, 3, 30, 0, 0, 0, 5), dt.datetime(2024, 4, 1, 0, 0, 0, 5))
big2 = v2t.DateTimeInterval(dt.datetime(2024, 3, 30, 0, 0, 0, 5), dt.datetime(2024, 4, 1, 0, 0, 0, 5))
lo, hi = dt.datetime(2024, 3, 31, 0, 30, 0, 5), dt.datetime(2024, 3, 31, 2, 30, 0, 5)
s1, s2 = big1[lo:hi], big2[lo:hi]
print(f'naive slice [{lo}:{hi}]: v1 total {s1.total_seconds} [{s1.infimum}, {s1.supremum}]   v2 total {float(s2.total_seconds)} [{s2.inf}, {s2.sup}]')
print('naive expand: v2 wall, v1 no-op:', ok_naive, '  aware expand: v2 elapsed:', ok_aware)
assert ok_naive and ok_aware
# sabotage: a naive slice claimed equal in span must be caught
print('SABOTAGE (expect naive slice spans equal) caught:', abs(s1.total_seconds - float(s2.total_seconds)) > 1)
