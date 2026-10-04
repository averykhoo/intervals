"""does v2 still reach v1's naive-local (elapsed-time) semantics? RUN IN WSL with TZ=<zone>.
v1 read a naive datetime as the process's local time (`timestamp()`), so naive arithmetic was elapsed time
and the two readings of a repeated wall time (fold 0/1) were two instants. v2 reads naive as wall clock
(D30 (a)). the v2 spelling tried here: make each naive input aware in the local zone first,
`L(d) = d.replace(tzinfo=ZoneInfo(TZ))` (fold respected; a gap time takes the offset before the change,
as `timestamp()` does), and read results back as naive local `R(x) = x.astimezone(ZoneInfo(TZ)).replace(tzinfo=None)`.
`DTI(d, tz=ZoneInfo(TZ))` stands for v1's local day.
"""
import os, sys, random, datetime as dt
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'stub'), '.', 'archive/v1']
import time_interval as v1t
del sys.modules['pandas']
import intervals.time_interval as v2t

ZONE = os.environ['TZ']
Z = ZoneInfo(ZONE)
UTC = dt.timezone.utc
US = dt.timedelta(microseconds=1)
L = lambda d: d.replace(tzinfo=Z)
R = lambda x: x.astimezone(Z).replace(tzinfo=None)
YEARS = {'Asia/Singapore': (1981, 1982)}.get(ZONE, (2024, 2024))


def transitions(zone, years):
    out = []
    t = dt.datetime(years[0], 1, 1, tzinfo=UTC) - dt.timedelta(days=1)
    end = dt.datetime(years[1], 12, 31, tzinfo=UTC)
    prev = t.astimezone(zone).utcoffset()
    while t < end:
        t2 = t + dt.timedelta(minutes=15)
        off = t2.astimezone(zone).utcoffset()
        if off != prev:
            out.append((t2, prev, off))
            prev = off
        t = t2
    return out


TR = transitions(Z, YEARS)
rng = random.Random(4242)
n = bad = 0
counts = dict(span=0, plus=0, minus=0, member=0, raise_both=0, fold=0)
naive_differs = 0
for _ in range(500):
    u, ob, oa = rng.choice(TR)
    wall_tr = (u + ob).replace(tzinfo=None)
    a = wall_tr + dt.timedelta(seconds=rng.randint(-4 * 3600, 3 * 3600), microseconds=rng.randint(1, 999999))
    a = a.replace(fold=rng.randint(0, 1))
    b = a + dt.timedelta(seconds=rng.randint(0, 5 * 3600), microseconds=rng.randint(1, 999999))
    b = b.replace(fold=rng.randint(0, 1))
    if b.microsecond == 0:
        b += US
    td = dt.timedelta(seconds=rng.randint(0, 5 * 3600), microseconds=rng.randint(1, 999999))
    try:
        x1 = v1t.DateTimeInterval(a, b)
        e1 = None
    except ValueError as e:
        e1 = 'ValueError'
    try:
        x2 = v2t.DateTimeInterval(L(a), L(b))
        e2 = None
    except ValueError as e:
        e2 = 'ValueError'
    n += 1
    if e1 or e2:
        ok = e1 == e2
        counts['raise_both'] += ok
        bad += not ok
        if not ok:
            print('  constructor mismatch', a, b, e1, e2)
        continue
    ok = abs(x1.total_seconds - float(x2.total_seconds)) < 1e-5
    counts['span'] += ok
    s1, s2 = x1 + td, x2 + td
    o = s1.infimum == R(s2.inf) and s1.supremum == R(s2.sup)
    counts['plus'] += o
    ok &= o
    d1, d2 = v1t.DateTimeInterval(b) - x1, v2t.DateTimeInterval(L(b)) - x2
    o = d1.infimum == d2.inf and d1.supremum == d2.sup
    counts['minus'] += o
    ok &= o
    p = (a + dt.timedelta(seconds=rng.randint(-3600, 6 * 3600))).replace(fold=rng.randint(0, 1))
    o = (p in x1) == (L(p) in x2)
    counts['member'] += o
    ok &= o
    # naive v2 (no workaround) on the same case, to show the probe can fail
    y2 = v2t.DateTimeInterval(a, b) + td
    naive_differs += not (s1.infimum == y2.inf and s1.supremum == y2.sup)
    bad += not ok
    if not ok:
        print('  mismatch', a, b, td, p)
print(f'TZ {ZONE}: {n} cases, mismatches {bad}; agreements {counts}')
print(f'  same cases WITHOUT the workaround (naive v2 + td vs v1): {naive_differs} differ   <- sabotage: must be > 0')
# the local day
for u, ob, oa in TR:
    d = (u + ob).date()
    v1d, v2d = v1t.DateTimeInterval(d), v2t.DateTimeInterval(d, tz=Z)
    print(f'  day {d}: v1 DTI(d) total {v1d.total_seconds} [{v1d.infimum} .. {v1d.supremum}]   '
          f'v2 DTI(d, tz=Z) total {float(v2d.total_seconds)} [{R(v2d.inf)} .. {R(v2d.sup)}) '
          f'  v2 naive DTI(d) total {float(v2t.DateTimeInterval(d).total_seconds)}')
assert bad == 0 and naive_differs > 0
