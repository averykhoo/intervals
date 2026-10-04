"""v1 vs v2 time arithmetic across DST and historical offset changes. RUN IN WSL with TZ=<zone>
(the windows zone of this laptop has no DST and no history). v1 reads every datetime through
`timestamp()` (naive = the process's local time, aware = UTC) and reads out with `fromtimestamp()`
(naive local); v2 reads naive as wall clock and aware as UTC, and keeps the zone (D30 (a)).
references computed independently of both libraries:
  wall(a, b)    = b - a on naive datetimes (python's naive arithmetic)
  elapsed(a, b) = b.timestamp() - a.timestamp() (seconds between the local instants)
"""
import os, sys, time, random, datetime as dt
from fractions import Fraction
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'stub'), '.', 'archive/v1']
import time_interval as v1t
del sys.modules['pandas']          # v2 then runs as on a machine without pandas (v1 keeps its own `pd`)
import intervals.time_interval as v2t

ZONE = os.environ['TZ']
LOCAL = ZoneInfo(ZONE)
UTC = dt.timezone.utc
print('TZ', ZONE, 'tzname', time.tzname)
US = dt.timedelta(microseconds=1)
FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print('  !! UNEXPECTED:', msg)


def transitions(zone, years):
    """UTC instants where zone's offset changes, with (off_before, off_after)"""
    out = []
    t = dt.datetime(years[0], 1, 1, tzinfo=UTC) - dt.timedelta(days=1)
    end = dt.datetime(years[1], 12, 31, tzinfo=UTC)
    prev = t.astimezone(zone).utcoffset()
    step = dt.timedelta(minutes=15)
    while t < end:
        t2 = t + step
        off = t2.astimezone(zone).utcoffset()
        if off != prev:
            out.append((t2, prev, off))
            prev = off
        t = t2
    return out


YEARS = {'Asia/Singapore': (1981, 1982)}.get(ZONE, (2024, 2024))
TR = transitions(LOCAL, YEARS)
print('transitions:', [(u.isoformat(), str(b), str(a)) for u, b, a in TR])
check(len(TR) > 0, 'zone has no transition in the probed years: the probe could not fail')


def elapsed(a, b):
    return b.timestamp() - a.timestamp()


def us_fraction(td):
    return Fraction(td // US, 10 ** 6)


# ---------------- NAIVE ----------------
print('\n=== naive, hand-picked, one per transition ===')
for u, ob, oa in TR:
    wall_tr = (u + ob).replace(tzinfo=None)                  # wall clock at the transition (old offset)
    a = wall_tr - dt.timedelta(hours=1) + 5 * US
    b = wall_tr + dt.timedelta(hours=2) + 5 * US
    x1, x2 = v1t.DateTimeInterval(a, b), v2t.DateTimeInterval(a, b)
    print(f'[{a} , {b}] total_seconds v1 {x1.total_seconds}  v2 {float(x2.total_seconds)}  '
          f'wall {(b - a).total_seconds()}  elapsed {elapsed(a, b)}')
    check(abs(x1.total_seconds - elapsed(a, b)) < 1e-5, 'v1 naive span is elapsed time')
    check(Fraction(x2.total_seconds) == us_fraction(b - a), 'v2 naive span is wall time')
    d1 = v1t.DateTimeInterval(b) - v1t.DateTimeInterval(a)
    d2 = v2t.DateTimeInterval(b) - v2t.DateTimeInterval(a)
    print(f'  DTI(b) - DTI(a): v1 {d1.infimum}  v2 {d2.inf}  python b - a {b - a}')
    check(d2.inf == b - a, 'v2 naive dt - dt == python naive b - a')
    td = dt.timedelta(hours=3)
    s1, s2 = v1t.DateTimeInterval(a) + td, v2t.DateTimeInterval(a) + td
    el = dt.datetime.fromtimestamp(a.timestamp() + td.total_seconds())
    print(f'  DTI(a) + 3h: v1 {s1.infimum}  v2 {s2.inf}  python a + 3h {a + td}  elapsed-reading {el}')
    check(s2.inf == a + td, 'v2 naive + == python naive +')
    check(s1.infimum == el, 'v1 naive + is elapsed then local read-out')
    day = wall_tr.date()
    print(f'  DTI({day}).total_seconds v1 {v1t.DateTimeInterval(day).total_seconds}  '
          f'v2 {float(v2t.DateTimeInterval(day).total_seconds)}')
    delta = oa - ob
    if delta > dt.timedelta(0):          # spring forward: [wall_tr, wall_tr + delta) never happens
        mid = wall_tr + delta / 2 + 7 * US
        r1, r2 = v1t.DateTimeInterval(mid).infimum, v2t.DateTimeInterval(mid).inf
        print(f'  gap wall time {mid}: v1 reads back {r1}   v2 reads back {r2}')
        check(r2 == mid, 'v2 gap time reads back as given')
        later = wall_tr + delta + dt.timedelta(minutes=1) + 9 * US     # just after the gap, wall order mid < later
        try:
            y1 = v1t.DateTimeInterval(mid, later)
            r = f'ok total {y1.total_seconds}'
        except Exception as e:
            r = f'{type(e).__name__}: {e}'
        y2 = v2t.DateTimeInterval(mid, later)
        print(f'  DTI({mid.time()}, {later.time()}): v1 {r}   v2 total {float(y2.total_seconds)} '
              f'(wall {(later - mid).total_seconds()}, elapsed {elapsed(mid, later)})')
    else:                                 # fall back: [wall_tr - |delta|, wall_tr) happens twice
        rep = wall_tr - (-delta) / 2 + 7 * US
        p0, p1 = rep.replace(fold=0), rep.replace(fold=1)
        e1 = v1t.DateTimeInterval(p0) == v1t.DateTimeInterval(p1)
        e2 = v2t.DateTimeInterval(p0) == v2t.DateTimeInterval(p1)
        print(f'  repeated wall time {rep} fold 0 vs fold 1: v1 DTI(p0) == DTI(p1) {e1}   v2 {e2}')
        lo = wall_tr + delta + 3 * US
        w1, w2 = v1t.DateTimeInterval(lo, p0), v2t.DateTimeInterval(lo, p0)
        print(f'  p1 (second {rep.time()}) in [{lo.time()}, p0 (first {rep.time()})]: v1 {p1 in w1}   v2 {p1 in w2}')
        sp1 = v1t.DateTimeInterval(a, p1).total_seconds
        sp2 = float(v2t.DateTimeInterval(a, p1).total_seconds)
        print(f'  DTI(a, p1).total_seconds: v1 {sp1}  v2 {sp2}  wall {(p1 - a).total_seconds()}  elapsed {elapsed(a, p1)}')
        pts1 = v1t.DateTimeInterval(p0).union(v1t.DateTimeInterval(p1)).degenerate_points
        print(f'  naive DTI(p0) | DTI(p1) degenerate_points: v1 {sorted(pts1)}   '
              f'v2 {(v2t.DateTimeInterval(p0) | v2t.DateTimeInterval(p1)).degenerate_points}')

print('\n=== naive, seeded sweep near each transition ===')
rng = random.Random(1982)
agree = diff = diff_ok = 0
mem_diff = mem_n = v1_raise = 0
for _ in range(600):
    u, ob, oa = rng.choice(TR)
    wall_tr = (u + ob).replace(tzinfo=None)
    a = wall_tr + dt.timedelta(seconds=rng.randint(-4 * 3600, 3 * 3600)) + rng.randint(1, 999999) * US
    td = dt.timedelta(seconds=rng.randint(0, 5 * 3600), microseconds=rng.randint(1, 999999))
    s1, s2 = v1t.DateTimeInterval(a) + td, v2t.DateTimeInterval(a) + td
    if s1.infimum == s2.inf:
        agree += 1
    else:
        diff += 1
        if s2.inf == a + td and s1.infimum == dt.datetime.fromtimestamp(a.timestamp() + td.total_seconds()):
            diff_ok += 1
    check(s2.inf == a + td, f'v2 naive + == python, a={a} td={td}')
    b = a + td
    if b.microsecond == 0:
        b += US
    p = a + dt.timedelta(seconds=rng.randint(-3600, 6 * 3600), microseconds=rng.randint(0, 999999))
    try:
        m1 = p in v1t.DateTimeInterval(a, b)
    except ValueError:
        m1 = 'raise'
        v1_raise += 1
    m2 = p in v2t.DateTimeInterval(a, b)
    mem_n += 1
    if m1 != m2:
        mem_diff += 1
    check(m2 == (a <= p <= b), f'v2 naive membership is wall order {a} {b} {p}')
    if m1 != 'raise':
        check(m1 == (a.timestamp() <= p.timestamp() <= b.timestamp()), f'v1 membership is local-timestamp order {a} {b} {p}')
print(f'naive a + td over 600: agree {agree}  differ {diff}  (of which v2 == python naive + and v1 == elapsed: {diff_ok})')
print(f'naive membership over {mem_n}: differ {mem_diff}, v1 constructor raised {v1_raise}')
check(diff == diff_ok, 'every naive difference is wall vs elapsed')

# ---------------- AWARE ----------------
others = {'Europe/London': 'Australia/Lord_Howe', 'Australia/Lord_Howe': 'Europe/London',
          'Asia/Singapore': 'Australia/Lord_Howe'}
zones = [LOCAL, ZoneInfo(others[ZONE]), UTC]


def inst(x):   # v1 read-out (naive local) -> UTC instant; an aware one converts
    return x.astimezone(UTC)


print('\n=== aware, hand-picked across each transition of each zone ===')
for Z in zones[:-1]:
    for u, ob, oa in transitions(Z, YEARS):
        a = (u - dt.timedelta(hours=2) + 5 * US).astimezone(Z)
        one_day = dt.timedelta(days=1)
        s1, s2 = v1t.DateTimeInterval(a) + one_day, v2t.DateTimeInterval(a) + one_day
        print(f'{Z} a={a}: a+1day v1 {s1.infimum} (tz {s1.infimum.tzinfo})  v2 {s2.inf}  python aware {a + one_day}')
        check(inst(s1.infimum) == inst(s2.inf), 'aware + : v1 and v2 the same instant')
        check(inst(s2.inf) == inst(a) + one_day, 'aware + is elapsed')
        b = (u + dt.timedelta(hours=3) + 9 * US).astimezone(Z)
        d1 = v1t.DateTimeInterval(b) - v1t.DateTimeInterval(a)
        d2 = v2t.DateTimeInterval(b) - v2t.DateTimeInterval(a)
        print(f'  b={b}  b - a: v1 {d1.infimum}  v2 {d2.inf}  python same-tzinfo b - a {b - a}  UTC diff {inst(b) - inst(a)}')
        check(d1.infimum == d2.inf == inst(b) - inst(a), 'aware dt - dt: v1 == v2 == elapsed')
        t1, t2 = v1t.DateTimeInterval(a, b).total_seconds, v2t.DateTimeInterval(a, b).total_seconds
        check(abs(t1 - float(t2)) < 1e-5, 'aware span equal')
        if oa < ob:   # fall back in Z: a wall time with two instants
            first = (u - (ob - oa) / 2).astimezone(Z)          # inside the repeated stretch, first pass
            rep = first.replace(tzinfo=None, microsecond=7)
            p0, p1 = rep.replace(tzinfo=Z, fold=0), rep.replace(tzinfo=Z, fold=1)
            u1 = v1t.DateTimeInterval(p0).union(v1t.DateTimeInterval(p1))
            u2 = v2t.DateTimeInterval(p0) | v2t.DateTimeInterval(p1)
            print(f'  fold pair {rep} ({inst(p0).time()}Z / {inst(p1).time()}Z): v1 pieces {len(u1.contiguous_intervals)} '
                  f'degenerate_points {sorted(u1.degenerate_points)} (len {len(u1.degenerate_points)})   '
                  f'v2 pieces {len(u2)} degenerate_points {u2.degenerate_points} (len {len(u2.degenerate_points)})')

print('\n=== aware, seeded sweep ===')
rng = random.Random(7530)
n = bad = 0
for _ in range(600):
    Z = rng.choice(zones)
    trs = transitions(Z, YEARS) if Z is not UTC else TR
    u = rng.choice(trs)[0]
    a = (u + dt.timedelta(seconds=rng.randint(-5 * 3600, 3 * 3600), microseconds=rng.randint(1, 999999))).astimezone(Z)
    td = dt.timedelta(seconds=rng.randint(0, 30 * 3600), microseconds=rng.randint(1, 999999))
    b = (inst(a) + td).astimezone(Z)  # the instant td later
    if b.microsecond == 0:
        b += US
    x1, x2 = v1t.DateTimeInterval(a, b), v2t.DateTimeInterval(a, b)
    ok = (inst(x1.infimum) == inst(x2.inf) and inst(x1.supremum) == inst(x2.sup)
          and abs(x1.total_seconds - float(x2.total_seconds)) < 1e-5)
    s1, s2 = x1 + td, x2 + td
    ok &= inst(s1.infimum) == inst(s2.inf) and inst(s1.supremum) == inst(s2.sup)
    d1, d2 = v1t.DateTimeInterval(b) - x1, v2t.DateTimeInterval(b) - x2
    ok &= d1.infimum == d2.inf and d1.supremum == d2.sup
    for k in range(4):
        p = (inst(a) + dt.timedelta(seconds=rng.randint(-3600, 40 * 3600), microseconds=rng.randint(0, 999999))).astimezone(rng.choice(zones))
        ok &= (p in x1) == (p in x2) == (inst(a) <= inst(p) <= inst(b))
    n += 1
    bad += not ok
    if not ok:
        print('  mismatch', Z, a, td)
print(f'aware sweep: {n} cases, {bad} mismatches (instants, spans, + td, dt - dt, membership x4)')
check(bad == 0, 'aware sweep agrees')

print('\nUNEXPECTED count:', len(FAIL))
# sabotage: claiming v1 naive arithmetic is wall clock must be caught (needs a real transition)
u, ob, oa = TR[0]
a = (u + ob).replace(tzinfo=None) - dt.timedelta(hours=1) + 5 * US
sab = (v1t.DateTimeInterval(a) + dt.timedelta(hours=3)).infimum == a + dt.timedelta(hours=3)
print('SABOTAGE (expect v1 naive + == python naive +) caught:', not sab)
