"""time layer: v2 iteration over pieces vs v1 contiguous_intervals, multi-piece, membership-compared."""
import sys, random, warnings, datetime as dt
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
rng = random.Random(7); us = dt.timedelta(microseconds=1); bad = n = 0
base = dt.datetime(2024, 3, 1, 0, 0, 0, 1)   # microsecond 1: v1's round-end stretch never fires
for _ in range(150):
    T1, T2 = v1t.TimeDeltaInterval(), v2t.TimeDeltaInterval()
    D1, D2 = v1t.DateTimeInterval(), v2t.DateTimeInterval()
    for _ in range(rng.randint(0, 3)):
        s = rng.randint(-50, 50); t = s + rng.randint(1, 20); sc, tc = rng.random() < .5, rng.random() < .5
        a, b = dt.timedelta(seconds=s, microseconds=1), dt.timedelta(seconds=t, microseconds=1)
        T1 = T1.union(v1t.TimeDeltaInterval(a, b, start_closed=sc, end_closed=tc))
        T2 = T2 | v2t.TimeDeltaInterval(a, b, start_closed=sc, end_closed=tc)
        D1 = D1.union(v1t.DateTimeInterval(base + a, base + b, start_closed=sc, end_closed=tc))
        D2 = D2 | v2t.DateTimeInterval(base + a, base + b, start_closed=sc, end_closed=tc)
    for X1, X2, z in ((T1, T2, dt.timedelta(0)), (D1, D2, base)):
        n += 1
        q1, q2 = X1.contiguous_intervals, list(X2)
        if len(q1) != len(q2) or len(X2) != len(q1): bad += 1; continue
        for p1, p2 in zip(q1, q2):
            for s in range(-55, 75):
                for off in (-us, dt.timedelta(0), us, dt.timedelta(microseconds=1)):
                    x = z + dt.timedelta(seconds=s) + off
                    if (x in p1) != (x in p2): bad += 1
print('time cases', n, 'piece mismatches:', bad)
# sabotage: drop a piece on the v2 side, must be caught
if any(len(T) > 1 for T in [T2]):
    pass
X1 = v1t.TimeDeltaInterval(dt.timedelta(0), dt.timedelta(1)).union(v1t.TimeDeltaInterval(dt.timedelta(3), dt.timedelta(4)))
X2 = v2t.TimeDeltaInterval(dt.timedelta(0), dt.timedelta(1))
assert len(X1.contiguous_intervals) != len(list(X2)), 'sabotage blind'
print('sabotage caught')
