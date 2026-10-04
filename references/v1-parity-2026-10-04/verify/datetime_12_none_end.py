import sys, datetime as dt, random
sys.path[:0] = ['.', 'archive/v1']
import time_interval as v1t
import multi_interval as v1
import intervals as v2
from intervals.time_interval import DateTimeInterval as D2
D1 = v1t.DateTimeInterval
t = dt.datetime(2024, 1, 1, 10)
for name, f in [('v1 DTI(None,t)', lambda: D1(None, t)), ('v1 DTI(t)', lambda: D1(t)),
                ('v2 DTI(None,t)', lambda: D2(None, t)), ('v2 DTI(t)', lambda: D2(t)),
                ('v1 DTI(None,date)', lambda: D1(None, t.date())), ('v2 DTI(date)', lambda: D2(t.date())),
                ('v1 MI(None,5)', lambda: v1.MultiInterval(None, 5)), ('v2 MI(None,5)', lambda: v2.MultiInterval(None, 5)),
                ('v1 DTI(None,t,start_closed=False)', lambda: D1(None, t, start_closed=False)),
                ('v1 DTI(None,t,end_closed=False)', lambda: D1(None, t, end_closed=False)),
                ('v2 DTI(t,start_closed=False)', lambda: D2(t, start_closed=False)),
                ]:
    try: print(name, '->', repr(f()))
    except Exception as e: print(name, '-> raise', type(e).__name__, e)
# membership sweep: v1 DTI(None,t) vs v2 DTI(t), at t, t+-1us, t+-1h
rng = random.Random(12); bad = 0; n = 0
for _ in range(300):
    t = dt.datetime(2000,1,1) + dt.timedelta(seconds=rng.randrange(10**9), microseconds=rng.choice([0, rng.randrange(10**6)]))
    a, b = D1(None, t), D2(t)
    for p in [t, t - dt.timedelta(microseconds=1), t + dt.timedelta(microseconds=1), t + dt.timedelta(hours=1)]:
        n += 1
        if (p in a) != bool(p in b): bad += 1; print('DIFF', t, p, p in a, p in b)
print('sweep', n, 'checks', bad, 'diffs')
# sensitivity: a deliberately wrong expectation must be caught
wrong = D2(t + dt.timedelta(microseconds=1))
print('sabotage caught:', (t in D1(None, t)) != bool(t in wrong))
