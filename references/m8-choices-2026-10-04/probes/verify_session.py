import math, time
from datetime import datetime, date, timedelta, timezone
from fractions import Fraction
import pandas as pd
from intervals import MultiInterval as M

def tryit(label, f):
    try: print(label, '->', repr(f()))
    except Exception as e: print(label, '-> RAISES', type(e).__name__, e)

print('tz', time.tzname, 'pandas', pd.__version__)
# Q1
for d in [datetime(1970,1,1), datetime(1970,1,2), datetime(1970,1,2,9), datetime(1900,1,1), datetime.min, datetime.max, datetime(3001,1,1)]:
    tryit(f'timestamp {d}', d.timestamp)
tryit('fromtimestamp(-1)', lambda: datetime.fromtimestamp(-1))
tryit('pd naive ts', lambda: pd.Timestamp('2024-01-01').timestamp())
tryit('py local ts', lambda: datetime(2024,1,1).timestamp())
def wall(d): 
    td = d - datetime(1970,1,1); return td.days*86400 + td.seconds + Fraction(td.microseconds, 10**6)
tryit('wall 2024', lambda: wall(datetime(2024,1,1)))
tryit('Fraction(ts.value,1e9)', lambda: Fraction(pd.Timestamp('2024-01-01').value, 10**9))
tryit('wall min/max', lambda: (wall(datetime.min), wall(datetime.max)))
tryit('naive<aware', lambda: datetime(2024,1,1) < datetime(2024,1,1,tzinfo=timezone.utc))
tryit('naive==aware', lambda: datetime(2024,1,1) == datetime(2024,1,1,tzinfo=timezone.utc))
tryit('pd naive<aware', lambda: pd.Timestamp('2024-01-01') < pd.Timestamp('2024-01-01', tz='UTC'))
tryit('pd.Interval naive,aware', lambda: pd.Interval(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02', tz='UTC')))
tryit('pd.Interval two zones', lambda: pd.Interval(pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2024-01-02', tz='Asia/Singapore')))
tryit('total_seconds loss', lambda: timedelta(days=10**6, microseconds=1).total_seconds())
# Q2
tryit('inf > datetime', lambda: math.inf > datetime(2024,1,1))
tryit('inf > Timestamp', lambda: math.inf > pd.Timestamp('2024-01-01'))
tryit('None < datetime', lambda: None < datetime(2024,1,1))
tryit('Timestamp.min/max', lambda: (pd.Timestamp.min, pd.Timestamp.max))
tryit('Timestamp(datetime.max).unit', lambda: pd.Timestamp(datetime.max).unit)
tryit('Timestamp(inf)', lambda: pd.Timestamp(math.inf))
tryit('M(-inf,5)', lambda: (M(-math.inf, 5).inf, M(-math.inf,5).inf_closed, M(-math.inf,5).size))
class _Top:
    def __lt__(s,o): return False
    def __gt__(s,o): return o is not s
    def __le__(s,o): return o is s
    def __ge__(s,o): return True
    def __repr__(s): return 'inf'
T=_Top()
tryit('sentinel > datetime', lambda: T > datetime(2024,1,1))
tryit('Timestamp < sentinel', lambda: pd.Timestamp('2024-01-01') < T)
tryit('datetime < sentinel', lambda: datetime(2024,1,1) < T)
tryit('date < sentinel', lambda: date(2024,1,1) < T)
tryit('sorted', lambda: sorted([pd.Timestamp('2024-01-01'), T, datetime(2023,1,1)]))
# other decisions
tryit('timedelta(seconds=1/3)', lambda: timedelta(seconds=Fraction(1,3)))
tryit('timedelta(us=1/2)', lambda: timedelta(microseconds=Fraction(1,2)))
tryit('M lt type', lambda: type(M(0,1) < M(2,3)).__name__)
# Q3
D=86400
tryit('closed snap days union', lambda: M(0, Fraction(D*10**6-1,10**6)) | M(D, Fraction(2*D*10**6-1,10**6)))
tryit('half-open days union', lambda: M.parse('[0, 86400)') | M.parse('[86400, 172800)'))
tryit('half-open size', lambda: M.parse('[0, 86400)').size)
tryit('snap size', lambda: M(0, Fraction(D*10**6-1,10**6)).size)
tryit('member snap', lambda: Fraction(D*10**7-5, 10**7) in M(0, Fraction(D*10**6-1,10**6)))
tryit('member halfopen', lambda: Fraction(D*10**7-5, 10**7) in M.parse('[0, 86400)'))
tryit('nan', lambda: M(float('nan'), 1))
tryit('M(datetime)', lambda: M(datetime(2024,1,1)))
