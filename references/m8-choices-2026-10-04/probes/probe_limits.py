import datetime as dt
from fractions import Fraction
import pandas as pd, numpy as np
EPOCH = dt.datetime(1970, 1, 1)
def wall_seconds(d: dt.datetime) -> Fraction:
    td = d - EPOCH
    return Fraction(td.days * 86400 + td.seconds) + Fraction(td.microseconds, 10**6)
def from_wall(s: Fraction) -> dt.datetime:
    return EPOCH + dt.timedelta(microseconds=int(s * 10**6))  # int(): truncation; check exactness separately
# pandas naive timestamp() reads naive as UTC
print('pd naive 1970-01-01 00:00:01 .timestamp() =', pd.Timestamp('1970-01-01 00:00:01').timestamp())
print('pd naive 2024-01-01 .timestamp() =', pd.Timestamp('2024-01-01').timestamp(), '| py local .timestamp() =', dt.datetime(2024,1,1).timestamp(), '| wall =', float(wall_seconds(dt.datetime(2024,1,1))))
# python's windows range for timestamp()/fromtimestamp()
for d in [dt.datetime(1970,1,1), dt.datetime(1970,1,2), dt.datetime(1969,12,31), dt.datetime(1900,1,1), dt.datetime.min, dt.datetime.max, dt.datetime(3001,1,1)]:
    try: r = d.timestamp()
    except Exception as e: r = f'RAISES {type(e).__name__}'
    print(f'py local .timestamp() of {d} -> {r}')
for s in [-1, 0, 86400*365*200, 32536799999, 32536850400, 253402300799]:
    try: r = dt.datetime.fromtimestamp(s)
    except Exception as e: r = f'RAISES {type(e).__name__}'
    print(f'fromtimestamp({s}) -> {r}')
# utc fromtimestamp
for s in [-1, -62135596800, 253402300799]:
    try: r = dt.datetime.fromtimestamp(s, dt.timezone.utc)
    except Exception as e: r = f'RAISES {type(e).__name__}'
    print(f'fromtimestamp({s}, utc) -> {r}')
# wall-seconds roundtrip at the extremes (no OS call)
for d in [dt.datetime.min, dt.datetime.max, dt.datetime(2024,3,10,2,30,0,1)]:
    s = wall_seconds(d); print(f'wall_seconds({d}) = {s} ; back == d: {from_wall(s) == d}')
# pd.Timestamp exact ns -> Fraction
t = pd.Timestamp('2024-01-01 00:00:00.123456789')
s = Fraction(t.value, 10**9); print('pd ns ->', s, '-> back', pd.Timestamp(int(s*10**9), unit='ns') == t)
# pd.Timestamp aware: .value is UTC ns
a = pd.Timestamp('2024-01-01 08:00', tz='Asia/Singapore'); u = pd.Timestamp('2024-01-01 00:00', tz='UTC')
print('aware SGT .value == UTC .value:', a.value == u.value, '| naive 00:00 .value:', pd.Timestamp('2024-01-01 00:00').value == u.value)
# half microsecond: 23:59:59.9999995 as a Fraction; is it representable in datetime?
half = wall_seconds(dt.datetime(2024,1,1,23,59,59,999999)) + Fraction(1, 2*10**6)
print('23:59:59.9999995 wall =', half, '| next midnight =', wall_seconds(dt.datetime(2024,1,2)), '| half < midnight:', half < wall_seconds(dt.datetime(2024,1,2)))
# pd.Timestamp from Fraction seconds (ns resolution only)
try: print('pd.Timestamp(half ns):', pd.Timestamp(int(half*10**9), unit='ns'))
except Exception as e: print('RAISES', e)
# timedelta exactness: total_seconds float vs exact
td = dt.timedelta(days=10**6, microseconds=1)
print('timedelta total_seconds float exact?', td.total_seconds(), '| exact', Fraction(td.days*86400+td.seconds)+Fraction(td.microseconds,10**6))
# datetime.min/max as a would-be infinite end: does constructing from it give the same? (it is finite)
print('datetime.max wall secs', wall_seconds(dt.datetime.max), 'is finite; datetime.min', wall_seconds(dt.datetime.min))
# pd.Timestamp.min .value, pd.NaT value
print('pd.Timestamp.min.value', pd.Timestamp.min.value, '| NaT.value', pd.NaT.value, '| np.datetime64 NaT', np.datetime64('NaT'))
# date vs datetime isinstance
print('isinstance(datetime, date):', isinstance(dt.datetime(2024,1,1), dt.date), '| pd.Timestamp is datetime:', isinstance(pd.Timestamp('2024-01-01'), dt.datetime))
# dateutil-style zones: tzinfo object identity vs utcoffset
import zoneinfo
ny = zoneinfo.ZoneInfo('America/New_York')
x = dt.datetime(2024,3,10,2,30, tzinfo=ny)  # nonexistent (gap)
print('NY gap 02:30 utcoffset', x.utcoffset(), '-> utc', x.astimezone(dt.timezone.utc), '| fold=1 ->', x.replace(fold=1).astimezone(dt.timezone.utc))
print('NY 02:30 == same utc instant via astimezone round trip:', x.astimezone(dt.timezone.utc).astimezone(ny))
