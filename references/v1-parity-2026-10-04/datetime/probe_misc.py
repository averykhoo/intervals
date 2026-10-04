"""`.interval` vs `.seconds`, building from raw seconds, flags passed as None (v1's Optional[bool])"""
from common import *
D = dt.datetime
OFFSET = 28800  # this machine: UTC+8, no DST; v1 seconds = wall-clock seconds - OFFSET

# SELFTEST: a wrong offset must be caught
a1, a2 = V1D(D(2024, 1, 1, 9, 0, 0, 1)), V2D(D(2024, 1, 1, 9, 0, 0, 1))
selftest(lambda: Fraction(a1.interval.endpoints[0][0]) + 3600 == Fraction(a2.seconds.inf) and False or
         abs(a1.interval.endpoints[0][0] + 3600 - float(a2.seconds.inf)) < 1e-3)
r = random.Random(3)
for i in range(300):
    t = rand_dt(r, lo=D(1990, 1, 1), span_days=15000)
    x1, x2 = V1D(t), V2D(t)
    check(abs(x1.interval.endpoints[0][0] + OFFSET - float(x2.seconds.inf)) < 1e-3, f'offset {t}')
print('v1 .interval', a1.interval, ' v2 .seconds', a2.seconds, ' v2 has .interval?', hasattr(a2, 'interval'))
# raw construction: v1 by assigning .interval, v2 from_seconds
raw1 = V1D(); raw1.interval = v1m.MultiInterval(1704070800.5 - OFFSET, 1704074400.25 - OFFSET)
raw2 = V2D.from_seconds(M2(Fraction(17040708005, 10), Fraction(170407440025, 100)))
print('raw: v1', raw1, ' v2', raw2, ' same', same_set(raw1, raw2))
check(same_set(raw1, raw2)[0], 'raw seconds')
print('v2 assigning .seconds:', safe(lambda: setattr(raw2, 'seconds', None)))
for sc in (None, 0, 1):
    print(f'start_closed={sc!r}: v1', safe(lambda: V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1), start_closed=sc)),
          '| v2', safe(lambda: V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1), start_closed=sc)))
report('misc')
