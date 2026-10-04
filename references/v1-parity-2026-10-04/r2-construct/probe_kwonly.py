"""start_closed / end_closed are keyword-only on both sides; a positional third argument is a TypeError.
also the accepted values of the flags (None, 0/1)."""
import sys, warnings, datetime as dt
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import time_interval as v1t
import intervals as v2
import intervals.time_interval as v2t

def outcome(f):
    try:
        r = f(); return 'ok ' + str(r)
    except Exception as e:
        return type(e).__name__

d0, d1 = dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)
t0, t1 = dt.timedelta(0), dt.timedelta(hours=1)
cases = [
    ('MI(1,2,False)', lambda: v1.MultiInterval(1, 2, False), lambda: v2.MultiInterval(1, 2, False)),
    ('MI(1,2,False,False)', lambda: v1.MultiInterval(1, 2, False, False), lambda: v2.MultiInterval(1, 2, False, False)),
    ('MI(1,2,True)', lambda: v1.MultiInterval(1, 2, True), lambda: v2.MultiInterval(1, 2, True)),
    ('MI(1,None,True)', lambda: v1.MultiInterval(1, None, True), lambda: v2.MultiInterval(1, None, True)),
    ('DTI(d0,d1,False)', lambda: v1t.DateTimeInterval(d0, d1, False), lambda: v2t.DateTimeInterval(d0, d1, False)),
    ('DTI(d0,d1,False,False)', lambda: v1t.DateTimeInterval(d0, d1, False, False), lambda: v2t.DateTimeInterval(d0, d1, False, False)),
    ('TDI(t0,t1,False)', lambda: v1t.TimeDeltaInterval(t0, t1, False), lambda: v2t.TimeDeltaInterval(t0, t1, False)),
    ('TDI(t0,t1,True,True)', lambda: v1t.TimeDeltaInterval(t0, t1, True, True), lambda: v2t.TimeDeltaInterval(t0, t1, True, True)),
    # keyword spelling works on both
    ('MI(1,2,start_closed=False)', lambda: v1.MultiInterval(1, 2, start_closed=False), lambda: v2.MultiInterval(1, 2, start_closed=False)),
    ('MI(1,2,end_closed=False)', lambda: v1.MultiInterval(1, 2, end_closed=False), lambda: v2.MultiInterval(1, 2, end_closed=False)),
    ('DTI(d0,d1,start_closed=False)', lambda: v1t.DateTimeInterval(d0, d1, start_closed=False), lambda: v2t.DateTimeInterval(d0, d1, start_closed=False)),
    ('TDI(t0,t1,end_closed=False)', lambda: v1t.TimeDeltaInterval(t0, t1, end_closed=False), lambda: v2t.TimeDeltaInterval(t0, t1, end_closed=False)),
    # flag values other than bool
    ('MI(1,2,start_closed=None)', lambda: v1.MultiInterval(1, 2, start_closed=None), lambda: v2.MultiInterval(1, 2, start_closed=None)),
    ('MI(1,2,start_closed=0)', lambda: v1.MultiInterval(1, 2, start_closed=0), lambda: v2.MultiInterval(1, 2, start_closed=0)),
    ('MI(1,2,end_closed=1)', lambda: v1.MultiInterval(1, 2, end_closed=1), lambda: v2.MultiInterval(1, 2, end_closed=1)),
    ('MI(1,2,end_closed="")', lambda: v1.MultiInterval(1, 2, end_closed=''), lambda: v2.MultiInterval(1, 2, end_closed='')),
    ('TDI(t0,t1,start_closed=None)', lambda: v1t.TimeDeltaInterval(t0, t1, start_closed=None), lambda: v2t.TimeDeltaInterval(t0, t1, start_closed=None)),
    ('DTI(d0,d1,end_closed=None)', lambda: v1t.DateTimeInterval(d0, d1, end_closed=None), lambda: v2t.DateTimeInterval(d0, d1, end_closed=None)),
    # unknown keyword
    ('MI(1,2,closed=False)', lambda: v1.MultiInterval(1, 2, closed=False), lambda: v2.MultiInterval(1, 2, closed=False)),
]
diff = 0
for name, f1, f2 in cases:
    o1, o2 = outcome(f1), outcome(f2)
    flag = '' if o1 == o2 else '   <-- differs'
    diff += bool(flag)
    print(f'{name:34s} v1: {o1:40s} v2: {o2}{flag}')
print('differing:', diff)
# membership check of the None-flag case at the ends, exactly
for v, M in (('v1', v1.MultiInterval), ('v2', v2.MultiInterval)):
    try:
        A = M(1, 2, start_closed=None); print(v, 'start_closed=None: 1 in A', 1 in A, '1.5 in A', 1.5 in A, '2 in A', 2 in A)
    except Exception as e: print(v, 'start_closed=None:', type(e).__name__, e)
# sabotage: the outcome comparison must see a difference where one exists
assert outcome(lambda: v1.MultiInterval(1, 2, end_closed=False)) != outcome(lambda: v2.MultiInterval(1, 2)), 'sabotage: blind compare'
print('sabotage caught')
# the two DTI rows differ only in str: compare membership exactly
mid = d0 + (d1 - d0) / 2; eps = dt.timedelta(microseconds=1)
for kw in ({'start_closed': False}, {'end_closed': None}, {'start_closed': False, 'end_closed': False}):
    a, b = v1t.DateTimeInterval(d0, d1, **kw), v2t.DateTimeInterval(d0, d1, **kw)
    pts = [d0 - eps, d0, d0 + eps, mid, d1 - eps, d1, d1 + eps]
    m1 = [p in a for p in pts]; m2 = [p in b for p in pts]
    print('DTI', kw, 'v1', m1, 'v2', m2, 'same' if m1 == m2 else 'DIFF')
for f in (lambda: v1.MultiInterval(1, 2, False), lambda: v2.MultiInterval(1, 2, False),
          lambda: v1t.DateTimeInterval(d0, d1, False), lambda: v2t.DateTimeInterval(d0, d1, False),
          lambda: v1t.TimeDeltaInterval(t0, t1, False), lambda: v2t.TimeDeltaInterval(t0, t1, False)):
    try: f()
    except TypeError as e: print('msg:', e)
