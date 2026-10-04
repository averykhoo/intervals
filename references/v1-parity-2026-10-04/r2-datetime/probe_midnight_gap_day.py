"""a local day whose midnight falls in a gap (Asia/Singapore 1981-12-31 23:30 +07:30 -> 1982-01-01 00:00 +08:00).
RUN IN WSL with TZ=Asia/Singapore. v1 reads a date as [combine(d, time.min), combine(d, time.max)] through
timestamp(); time.max of the 31st is a nonexistent wall time, so its timestamp lands 30 min into the next day.
exact reference: the day 1981-12-31 in Singapore is [1981-12-30 16:30Z, 1981-12-31 16:00Z) = 84600 s."""
import os, sys, datetime as dt
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'stub'), '.', 'archive/v1']
import time_interval as v1t
del sys.modules['pandas']
import intervals.time_interval as v2t

Z = ZoneInfo(os.environ['TZ'])
UTC = dt.timezone.utc
d31, d1 = dt.date(1981, 12, 31), dt.date(1982, 1, 1)
ref_lo, ref_hi = dt.datetime(1981, 12, 30, 16, 30, tzinfo=UTC), dt.datetime(1981, 12, 31, 16, 0, tzinfo=UTC)
a1, b1 = v1t.DateTimeInterval(d31), v1t.DateTimeInterval(d1)
a2, b2 = v2t.DateTimeInterval(d31, tz=Z), v2t.DateTimeInterval(d1, tz=Z)
print('reference day 31st (UTC):', ref_lo, '->', ref_hi, (ref_hi - ref_lo).total_seconds(), 's')
print('v1 DTI(31st):', a1.infimum.astimezone(UTC), '->', a1.supremum.astimezone(UTC), a1.total_seconds)
print('v2 DTI(31st, tz=Z):', a2.inf.astimezone(UTC), '->', a2.sup.astimezone(UTC), float(a2.total_seconds), 'sup_closed', a2.sup_closed)
ov1 = a1.intersection(b1)
ov2 = a2 & b2
print('v1 DTI(31st) & DTI(1st) empty?', ov1.is_empty, '' if ov1.is_empty else f'overlap {ov1.infimum} .. {ov1.supremum} ({ov1.total_seconds} s)')
print('v2 DTI(31st, tz) & DTI(1st, tz) empty?', ov2.is_empty, ' union one piece?', len(a2 | b2) == 1)
probe = dt.datetime(1981, 12, 31, 16, 15, tzinfo=UTC)          # 1982-01-01 00:15 +08:00, a moment of the 1st
print(f'{probe.astimezone(Z)} in v1 day 31st: {probe in a1}   in v2 day 31st: {probe in a2}   in v2 day 1st: {probe in b2}')
assert a2.inf == ref_lo and a2.sup == ref_hi and not a2.sup_closed
assert not ov1.is_empty and ov2.is_empty
print('SABOTAGE (expect v1 31st to end at the reference) caught:', a1.supremum.astimezone(UTC) != ref_hi)
