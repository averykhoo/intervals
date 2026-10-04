"""str() of a non-negative TimeDeltaInterval: random unions of microsecond-valued pieces (0 to ~1000 days), each
library's text checked against a formatter of its own rule (v1: python's text; v2: python's text minus the comma),
and v1 == v2 checked to hold exactly when no end reaches one day. `sab`: the v2 formatter keeps the comma."""
import sys, random, warnings, datetime as dt
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import time_interval as v1t
import intervals.time_interval as v2t
SAB = len(sys.argv) > 1
td = dt.timedelta
def fmt(pieces, comma):
    t = lambda x: str(x) if comma else str(x).replace(', ', ' ')
    texts = [f'[{t(lo)}]' if lo == hi else f'{"[" if lc else "("}{t(lo)}, {t(hi)}{"]" if hc else ")"}' for lo, hi, lc, hc in pieces]
    return '{}' if not texts else texts[0] if len(texts) == 1 else '{ ' + ' , '.join(texts) + ' }'
def rtd(rng):
    k = rng.random()
    if k < .4: return td(microseconds=rng.randrange(0, 86400 * 10**6))                     # under a day
    if k < .5: return td(days=1) - td(microseconds=rng.randrange(0, 3))                      # at the edge
    if k < .8: return td(days=rng.randrange(1, 1000), seconds=rng.randrange(86400), microseconds=rng.choice([0, rng.randrange(10**6)]))
    return td(seconds=rng.randrange(0, 3 * 86400))
rng = random.Random(684); n = ok1 = ok2 = 0; same = same_expected = 0; mism = []
cases = [[(td(hours=1), td(hours=1), True, True)], [(td(days=1, hours=2), td(days=3), True, True)], []]
for _ in range(600):
    ends = sorted({rtd(rng) for _ in range(2 * rng.randint(1, 3))})
    ps = []
    for i in range(0, len(ends) - 1, 2):
        lo, hi = ends[i], ends[i + 1]
        if rng.random() < .15: hi = lo
        ps.append((lo, hi, True if lo == hi else rng.random() < .5, True if lo == hi else rng.random() < .5))
    cases.append(ps)
for ps in cases:
    # drop touching pieces that the libraries would merge: keep a strict gap
    keep = []
    for p in ps:
        if keep and p[0] <= keep[-1][1]: continue
        keep.append(p)
    a1, a2 = v1t.TimeDeltaInterval(), v2t.TimeDeltaInterval()
    for lo, hi, lc, hc in keep:
        a1 = a1.union(v1t.TimeDeltaInterval(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else v1t.TimeDeltaInterval(lo))
        a2 = a2 | (v2t.TimeDeltaInterval(lo, hi, start_closed=lc, end_closed=hc) if lo != hi else v2t.TimeDeltaInterval(lo))
    s1, s2 = str(a1), str(a2)
    e1, e2 = fmt(keep, True), fmt(keep, SAB)
    n += 1; ok1 += s1 == e1; ok2 += s2 == e2
    big = any(x >= td(days=1) for p in keep for x in p[:2])
    same += s1 == s2; same_expected += (s1 == s2) == (not big)
    if s1 != e1 or s2 != e2: mism.append((s1, e1, s2, e2))
print('cases', n, ' v1 == its rule', ok1, ' v2 == its rule', ok2, ' v1 text == v2 text', same,
      ' (v1 == v2) iff (no end >= 1 day):', same_expected)
for m in mism[:4]: print('  mismatch', m)
print('e.g.', repr(str(v1t.TimeDeltaInterval(td(days=1, hours=2), td(days=3)))), '|', repr(str(v2t.TimeDeltaInterval(td(days=1, hours=2), td(days=3)))))
assert ok1 == ok2 == same_expected == n
