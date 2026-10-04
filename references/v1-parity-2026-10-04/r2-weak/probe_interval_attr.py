"""v1 `.interval` (MultiInterval of float POSIX seconds) vs v2 `.seconds` (exact MultiInterval of seconds):
(1) DateTimeInterval: v1 DTI(naive t...).interval vs v2 DTI(t.astimezone()...).seconds, multi-piece sets, ends
    within 1 us (v1 is a float) and flags exact; naive v2 .seconds differs by the local offset (D30 (a)).
(2) TimeDeltaInterval: v1 .interval vs v2 .seconds, same check.
(3) the reverse (v1's way of building from raw seconds: assign .interval): v2 DateTimeInterval.from_seconds(mi, tz=local)
    and TimeDeltaInterval.from_seconds(mi); membership of the original instants compared.
`sab`: one v2 end shifted by 2 us, must be caught."""
import sys, random, warnings, datetime as dt, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m, time_interval as v1t
import intervals as v2, intervals.time_interval as v2t
SAB = len(sys.argv) > 1
LOCAL = dt.datetime.now().astimezone().tzinfo
OFF = F(int(LOCAL.utcoffset(None).total_seconds()))
def st1(m): e = m.endpoints; return [(F(e[i][0]), e[i][1] == 0, F(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]
def st2(m): return [(F(p.inf), bool(p.inf_closed), F(p.sup), bool(p.sup_closed)) for p in m]
def close(s1, s2, shift=F(0)):
    return len(s1) == len(s2) and all(abs(a[0] - b[0] - shift) < F(1, 10**6) and abs(a[2] - b[2] - shift) < F(1, 10**6)
                                      and a[1] == b[1] and a[3] == b[3] for a, b in zip(s1, s2))
def to_v2(m):   # the same float ends as a v2 MultiInterval (v1's MultiInterval is a different class)
    e = m.endpoints; return v2.MultiInterval.from_pieces([(e[i][0], e[i + 1][0], e[i][1] == 0, e[i + 1][1] == 0) for i in range(0, len(e), 2)])
rng = random.Random(55)
def rdt(): return dt.datetime(1990, 1, 1) + dt.timedelta(days=rng.randrange(15000), seconds=rng.randrange(86400), microseconds=rng.randrange(1, 10**6))
def rtd(): return dt.timedelta(days=rng.randrange(-500, 500), seconds=rng.randrange(86400), microseconds=rng.randrange(1, 10**6))
c = dict(dti=0, dti_aware_ok=0, dti_naive_eq=0, dti_naive_shift_ok=0, tdi=0, tdi_ok=0, back_dt_ok=0, back_td_ok=0)
for k in range(400):
    ts = sorted({rdt() for _ in range(2 * rng.randint(1, 3))}); ds = sorted({rtd() for _ in range(2 * rng.randint(1, 3))})
    a1 = v1t.DateTimeInterval(); a2n = v2t.DateTimeInterval(); a2a = v2t.DateTimeInterval()
    for i in range(0, len(ts) - 1, 2):
        lc, hc = rng.random() < .5, rng.random() < .5
        a1 = a1.union(v1t.DateTimeInterval(ts[i], ts[i + 1], start_closed=lc, end_closed=hc))
        a2n = a2n | v2t.DateTimeInterval(ts[i], ts[i + 1], start_closed=lc, end_closed=hc)
        a2a = a2a | v2t.DateTimeInterval(ts[i].astimezone(), ts[i + 1].astimezone(), start_closed=lc, end_closed=hc)
    s1, s2a, s2n = st1(a1.interval), st2(a2a.seconds), st2(a2n.seconds)
    if SAB and k == 3: s2a[0] = (s2a[0][0] + F(2, 10**6),) + s2a[0][1:]
    c['dti'] += 1; c['dti_aware_ok'] += close(s1, s2a); c['dti_naive_eq'] += close(s1, s2n); c['dti_naive_shift_ok'] += close(s1, s2n, -OFF)
    b1 = v1t.TimeDeltaInterval(); b2 = v2t.TimeDeltaInterval()
    for i in range(0, len(ds) - 1, 2):
        lc, hc = rng.random() < .5, rng.random() < .5
        b1 = b1.union(v1t.TimeDeltaInterval(ds[i], ds[i + 1], start_closed=lc, end_closed=hc))
        b2 = b2 | v2t.TimeDeltaInterval(ds[i], ds[i + 1], start_closed=lc, end_closed=hc)
    c['tdi'] += 1; c['tdi_ok'] += close(st1(b1.interval), st2(b2.seconds))
    # reverse: raw seconds -> set. v1 assigns .interval; v2 from_seconds. compare membership of the original instants,
    # 1 us inside and outside each end (v1's float seconds are within 1 us; probe 2 us away to stay decisive)
    r1 = v1t.DateTimeInterval(); r1.interval = a1.interval
    r2 = v2t.DateTimeInterval.from_seconds(to_v2(a1.interval), tz=LOCAL)
    pts = [t + dt.timedelta(microseconds=d) for t in ts for d in (-2, 2)]
    c['back_dt_ok'] += [p in r1 for p in pts] == [p.astimezone() in r2 for p in pts]
    q1 = v1t.TimeDeltaInterval(); q1.interval = b1.interval
    q2 = v2t.TimeDeltaInterval.from_seconds(to_v2(b1.interval))
    pts = [x + dt.timedelta(microseconds=d) for x in ds for d in (-2, 2)]
    c['back_td_ok'] += [p in q1 for p in pts] == [p in q2 for p in pts]
print('local offset', OFF, c)
try: print('read-out of a from_seconds end built from a v1 float:', v2t.DateTimeInterval.from_seconds(to_v2(v1t.DateTimeInterval(ts[0]).interval), tz=LOCAL).inf)
except Exception as e: print('read-out of a from_seconds end built from a v1 float raises:', type(e).__name__, str(e)[:100])
assert c['dti_aware_ok'] == c['dti'] and c['tdi_ok'] == c['tdi'] and c['back_dt_ok'] == c['dti'] and c['back_td_ok'] == c['tdi']
assert (c['dti_naive_eq'] == 0) == (OFF != 0) and c['dti_naive_shift_ok'] == c['dti']
