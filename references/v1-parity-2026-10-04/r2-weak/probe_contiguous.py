"""contiguous_intervals (v1) vs pieces / iteration (v2), DateTimeInterval and TimeDeltaInterval, with checks that fail:
(1) random multi-piece sets with sub-second ends (no v1 snap): piece count equal and each piece's ends within 1 us and
    flags equal (v2 built aware in the local zone so its seconds are POSIX, as v1's); asserted per set.
(2) whole-second ends: v1 rebuilds each piece through its constructor, whose end snap inflates it; checked by
    membership 1 s past the end: in v1's piece, not in v1's set; v2's pieces' union equals the set.
`sab`: a v2 piece end shifted 2 us in one set, must be caught."""
import sys, random, warnings, datetime as dt
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
SAB = len(sys.argv) > 1
def st1(m): e = m.endpoints; return [(F(e[i][0]), e[i][1] == 0, F(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]
def st2(m): return [(F(p.inf), bool(p.inf_closed), F(p.sup), bool(p.sup_closed)) for p in m]
def close(a, b, sab=False):
    if sab: b = [(b[0][0] + F(2, 10**6),) + b[0][1:]] + b[1:]
    return len(a) == len(b) and all(abs(x[0] - y[0]) < F(1, 10**6) and abs(x[2] - y[2]) < F(1, 10**6) and x[1] == y[1] and x[3] == y[3] for x, y in zip(a, b))
rng = random.Random(180); c = dict(dti=0, dti_ok=0, tdi=0, tdi_ok=0)
for k in range(300):
    ts = sorted({dt.datetime(2000, 1, 1) + dt.timedelta(seconds=rng.randrange(10**8), microseconds=rng.randrange(1, 10**6)) for _ in range(2 * rng.randint(1, 4))})
    a1, a2 = v1t.DateTimeInterval(), v2t.DateTimeInterval()
    for i in range(0, len(ts) - 1, 2):
        lc, hc = rng.random() < .5, rng.random() < .5
        a1 = a1.union(v1t.DateTimeInterval(ts[i], ts[i + 1], start_closed=lc, end_closed=hc))
        a2 = a2 | v2t.DateTimeInterval(ts[i].astimezone(), ts[i + 1].astimezone(), start_closed=lc, end_closed=hc)
    p1 = [st1(p.interval)[0] for p in a1.contiguous_intervals]; p2 = [st2(p.seconds)[0] for p in a2.pieces]
    assert list(a2) == list(a2.pieces)
    c['dti'] += 1; c['dti_ok'] += close(p1, p2, SAB and k == 5)
    ds = sorted({dt.timedelta(seconds=rng.randrange(-10**7, 10**7), microseconds=rng.randrange(1, 10**6)) for _ in range(2 * rng.randint(1, 4))})
    b1, b2 = v1t.TimeDeltaInterval(), v2t.TimeDeltaInterval()
    for i in range(0, len(ds) - 1, 2):
        lc, hc = rng.random() < .5, rng.random() < .5
        b1 = b1.union(v1t.TimeDeltaInterval(ds[i], ds[i + 1], start_closed=lc, end_closed=hc))
        b2 = b2 | v2t.TimeDeltaInterval(ds[i], ds[i + 1], start_closed=lc, end_closed=hc)
    c['tdi'] += 1; c['tdi_ok'] += close([st1(p.interval)[0] for p in b1.contiguous_intervals], [st2(p.seconds)[0] for p in b2.pieces])
print(c)
# (2) whole-second ends
t = dt.datetime(2024, 1, 1, 11)
cases = {'point at 11:00': (v1t.DateTimeInterval(t), v2t.DateTimeInterval(t)),
         'A - [10:30, 11:00] leaves a 10:30:00 end': (v1t.DateTimeInterval(dt.datetime(2024, 1, 1, 9, 0, 0, 1), dt.datetime(2024, 1, 1, 11, 59, 59, 1)).difference(v1t.DateTimeInterval(dt.datetime(2024, 1, 1, 10, 30), dt.datetime(2024, 1, 1, 11, 0, 0, 5))),
                                                      v2t.DateTimeInterval(dt.datetime(2024, 1, 1, 9, 0, 0, 1), dt.datetime(2024, 1, 1, 11, 59, 59, 1)).difference(v2t.DateTimeInterval(dt.datetime(2024, 1, 1, 10, 30), dt.datetime(2024, 1, 1, 11, 0, 0, 5))))}
inflated = 0
for name, (s1, s2) in cases.items():
    sup1 = max(p.interval.endpoints[1][0] for p in s1.contiguous_intervals)
    probe = dt.datetime.fromtimestamp(s1.interval.endpoints[1][0]) + dt.timedelta(seconds=1)
    in_piece1 = any(probe in p for p in s1.contiguous_intervals); in_set1 = probe in s1
    union2 = v2t.DateTimeInterval()
    for p in s2: union2 = union2 | p
    inflated += in_piece1 and not in_set1
    print(f'{name}: probe {probe}: in a v1 piece {in_piece1}, in the v1 set {in_set1};  v2 pieces union == set {union2 == s2}, probe in v2 {probe in s2}')
    assert union2 == s2 and not (probe in s2)
assert c['dti_ok'] == c['dti'] and c['tdi_ok'] == c['tdi'] and inflated == 2
