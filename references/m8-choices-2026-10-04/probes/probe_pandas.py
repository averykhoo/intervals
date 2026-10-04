import datetime as dt, sys
import pandas as pd, numpy as np
print("pandas", pd.__version__)
# 1. tz-naive vs aware comparisons / arithmetic
n = pd.Timestamp('2024-03-10 02:30')
a = pd.Timestamp('2024-03-10 02:30', tz='UTC')
for op, f in [('==', lambda: n == a), ('<', lambda: n < a), ('-', lambda: n - a)]:
    try: print('naive', op, 'aware ->', f())
    except Exception as e: print('naive', op, 'aware -> RAISES', type(e).__name__, e)
# python datetime
pn = dt.datetime(2024,3,10,2,30); pa = dt.datetime(2024,3,10,2,30,tzinfo=dt.timezone.utc)
for op, f in [('==', lambda: pn == pa), ('<', lambda: pn < pa), ('-', lambda: pn - pa)]:
    try: print('py naive', op, 'aware ->', f())
    except Exception as e: print('py naive', op, 'aware -> RAISES', type(e).__name__, e)
# 2. pd.Interval of Timestamps mixed tz
try: print(pd.Interval(n, pd.Timestamp('2024-03-11')))
except Exception as e: print('Interval naive RAISES', e)
try: print(pd.Interval(a, pd.Timestamp('2024-03-11', tz='UTC')))
except Exception as e: print('Interval aware RAISES', e)
try: print(pd.Interval(n, pd.Timestamp('2024-03-11', tz='UTC')))
except Exception as e: print('Interval mixed RAISES', type(e).__name__, e)
try: print(pd.Interval(a, pd.Timestamp('2024-03-11', tz='Asia/Singapore')))
except Exception as e: print('Interval two zones RAISES', type(e).__name__, e)
# aware in two zones but same instant compare equal?
b = pd.Timestamp('2024-03-10 10:30', tz='Asia/Singapore')
print('UTC 02:30 == SGT 10:30:', a == b, '| hash equal:', hash(a)==hash(b))
print('Interval(UTC, SGT) ->', end=' ')
try: print(pd.Interval(a, b + pd.Timedelta('1D')))
except Exception as e: print('RAISES', type(e).__name__, e)
# 3. pd.Timestamp min/max, NaT, infinite
print('Timestamp.min', pd.Timestamp.min, repr(pd.Timestamp.min.value))
print('Timestamp.max', pd.Timestamp.max)
print('Timedelta.min/max', pd.Timedelta.min, pd.Timedelta.max)
print('NaT < ts:', pd.NaT < n, '| NaT == NaT:', pd.NaT == pd.NaT, '| isna:', pd.isna(pd.NaT))
try: print(pd.Timestamp(float('inf')))
except Exception as e: print('Timestamp(inf) RAISES', type(e).__name__, e)
try: print(pd.Timestamp(np.inf, unit='s'))
except Exception as e: print('Timestamp(inf,s) RAISES', type(e).__name__, e)
try: print(pd.Interval(pd.Timestamp.min, n))
except Exception as e: print('Interval(min, n) RAISES', e)
try: print(pd.Interval(pd.NaT, n))
except Exception as e: print('Interval(NaT, n) RAISES', type(e).__name__, e)
# 4. Timestamp resolution / exactness
t = pd.Timestamp('2024-01-01 00:00:00.123456789')
print('ns ts', t, 'unit', t.unit, 'value', t.value, 'to_pydatetime ->', t.to_pydatetime())
print('py datetime.max', dt.datetime.max, 'min', dt.datetime.min)
print('datetime.max.timestamp() ->', end=' ')
try: print(dt.datetime.max.timestamp())
except Exception as e: print('RAISES', type(e).__name__, e)
print('datetime.max - epoch (naive) ->', (dt.datetime.max - dt.datetime(1970,1,1)).total_seconds())
print('fromtimestamp(1e18) ->', end=' ')
try: print(dt.datetime.fromtimestamp(1e18))
except Exception as e: print('RAISES', type(e).__name__, e)
print('timedelta.max', dt.timedelta.max, 'secs', dt.timedelta.max.total_seconds())
# 5. float timestamp precision
x = dt.datetime(2024, 1, 1, 12, 0, 0, 1)
print('timestamp float exact?', x.timestamp(), dt.datetime.fromtimestamp(x.timestamp()) == x)
y = dt.datetime(2024, 1, 1, 12, 0, 0, 999999)
print('999999 roundtrip', dt.datetime.fromtimestamp(y.timestamp()) == y, repr(y.timestamp()))
# local zone, DST probe: what is the machine's zone
print('local tz', dt.datetime.now().astimezone().tzinfo)
# 6. naive wall clock difference across DST in a DST zone
import zoneinfo
ny = zoneinfo.ZoneInfo('America/New_York')
d1 = dt.datetime(2024,3,10,1,30, tzinfo=ny); d2 = dt.datetime(2024,3,10,3,30, tzinfo=ny)
print('NY 1:30 -> 3:30 on 2024-03-10 (gap): aware diff', d2 - d1, '| via UTC', d2.astimezone(dt.timezone.utc) - d1.astimezone(dt.timezone.utc))
f1 = dt.datetime(2024,11,3,1,30, tzinfo=ny, fold=0); f2 = dt.datetime(2024,11,3,1,30, tzinfo=ny, fold=1)
print('NY fold: same wall 1:30 fold0/fold1 ==', f1 == f2, '| utc', f1.astimezone(dt.timezone.utc).time(), f2.astimezone(dt.timezone.utc).time(), '| timestamps', f1.timestamp(), f2.timestamp())
# 7. pandas Interval date-like closed
print('pd.Interval closed default:', pd.Interval(0, 1).closed)
print(pd.interval_range(pd.Timestamp('2024-01-01'), periods=2, freq='D'))
# pandas naive timestamp -> seconds since epoch (which epoch?)
print('naive ts .timestamp()', pd.Timestamp('1970-01-01 00:00:01').timestamp(), '| py naive', dt.datetime(1970,1,1,0,0,1).timestamp())
