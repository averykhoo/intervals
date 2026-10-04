import datetime as dt, math, functools
from fractions import Fraction
import pandas as pd, numpy as np
D = dt.datetime(2024,1,1); TS = pd.Timestamp('2024-01-01'); TD = dt.timedelta(1); PTD = pd.Timedelta('1D'); DATE = dt.date(2024,1,1)
def t(label, f):
    try: print(f'{label} -> {f()!r}')
    except Exception as e: print(f'{label} -> RAISES {type(e).__name__}: {e}')
print('--- float inf as a read-out')
t('math.inf > datetime', lambda: math.inf > D)
t('math.inf > Timestamp', lambda: math.inf > TS)
t('-math.inf < timedelta', lambda: -math.inf < TD)
t('None < datetime', lambda: None < D)
print('--- datetime.min/max as a read-out')
t('pd.Timestamp(datetime.max)', lambda: (pd.Timestamp(dt.datetime.max), pd.Timestamp(dt.datetime.max).unit))
t('pd.Timestamp("9999-12-31").unit', lambda: pd.Timestamp('9999-12-31').unit)
t('pd.Timestamp("2262-04-12")', lambda: (pd.Timestamp('2262-04-12'), pd.Timestamp('2262-04-12').unit))
t('pd.Timestamp.max.to_pydatetime()', lambda: pd.Timestamp.max.to_pydatetime())
t('datetime.max > Timestamp.max', lambda: dt.datetime.max > pd.Timestamp.max)
t('pd.Interval(Timestamp.min, Timestamp.max)', lambda: pd.Interval(pd.Timestamp.min, pd.Timestamp.max))
t('pd.Interval(Timestamp(datetime.min), Timestamp(datetime.max))', lambda: pd.Interval(pd.Timestamp(dt.datetime.min), pd.Timestamp(dt.datetime.max)))
print('--- NaT')
t('NaT < datetime', lambda: pd.NaT < D); t('NaT > datetime', lambda: pd.NaT > D)
print('--- a sentinel pair')
@functools.total_ordering
class _Inf:
    __slots__ = ('sign',)
    def __init__(self, sign): self.sign = sign
    def __repr__(self): return '-inf' if self.sign < 0 else 'inf'
    def __eq__(self, other): return isinstance(other, _Inf) and other.sign == self.sign
    def __hash__(self): return hash(('time-inf', self.sign))
    def __lt__(self, other):
        if isinstance(other, _Inf): return self.sign < other.sign
        return self.sign < 0
NEG, POS = _Inf(-1), _Inf(1)
for name, x in [('datetime', D), ('Timestamp', TS), ('date', DATE), ('timedelta', TD), ('pd.Timedelta', PTD), ('float', 1.0), ('NaT', pd.NaT)]:
    t(f'NEG < {name}', lambda: NEG < x); t(f'{name} < POS', lambda: x < POS); t(f'{name} > NEG', lambda: x > NEG); t(f'{name} == POS', lambda: x == POS)
t('sorted([TS, POS, D, NEG])', lambda: sorted([TS, POS, D, NEG]))
t('max(D, POS)', lambda: max(D, POS))
print('--- datetime/timedelta from a Fraction')
t('timedelta(seconds=Fraction(1,3))', lambda: dt.timedelta(seconds=Fraction(1,3)))
t('timedelta(microseconds=Fraction(1,2))', lambda: dt.timedelta(microseconds=Fraction(1,2)))
t('timedelta(seconds=Fraction(3,2))', lambda: dt.timedelta(seconds=Fraction(3,2)))
t('epoch + timedelta(seconds=Fraction(7, 3))', lambda: dt.datetime(1970,1,1) + dt.timedelta(seconds=Fraction(7,3)))
t('pd.Timestamp(Fraction ns)', lambda: pd.Timestamp(int(Fraction(7,3)*10**9), unit='ns'))
print('--- v1 date-end snap details')
# v1 snaps a datetime end on the hour too (lines 102-109)
e = dt.datetime(2024,1,1,10,0)
print('v1 would snap datetime end 10:00 to', e.replace(minute=59, second=59, microsecond=999999))
