"""`date in A` (v1: whole day a subset). sets built from the SAME instants on both sides: v1 via its MultiInterval of
POSIX seconds (so no end snap), v2 via the constructor. every answer checked against an exact brute-force oracle of
each library's own day: v1 closed [d 00:00, d 23:59:59.999999], v2 half-open [d 00:00, d+1 00:00) (D30 (c)).
`sab` arg: the v1 oracle is given v2's day, so it must fail."""
import sys, random, warnings, datetime as dt
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m, time_interval as v1t
import intervals.time_interval as v2t
SAB = len(sys.argv) > 1
D, US = dt.datetime, dt.timedelta(microseconds=1)
EPOCH = D(1970, 1, 1)
def ws(t): return F((t - EPOCH) // US, 10**6)              # exact wall seconds (a local zone without DST: UTC+8 here)
d0 = dt.date(2024, 3, 5); day0 = D(2024, 3, 5); day1 = day0 + dt.timedelta(days=1)
GRID = [day0 - dt.timedelta(hours=3), day0 - US, day0, day0 + US, day0 + dt.timedelta(hours=12),
        day1 - 2 * US, day1 - US, day1, day1 + US, day1 + dt.timedelta(hours=3)]
V1DAY = (ws(day0), ws(day1 - US)); V2DAY = (ws(day0), ws(day1))
def merged(ps):
    ps = sorted(ps); out = []
    for lo, hi, lc, hc in ps:
        if out and (lo < out[-1][1] or (lo == out[-1][1] and (lc or out[-1][3]))):
            plo, phi, plc, phc = out[-1]; plc = plc or (lo == plo and lc)
            if hi > phi or (hi == phi and hc): phi, phc = hi, hc
            out[-1] = (plo, phi, plc, phc)
        else: out.append((lo, hi, lc, hc))
    return out
def subset(day, closed_end, ps):     # [day0, day1(closed_end?)] subset of union(ps)
    a, b = day
    for lo, hi, lc, hc in merged(ps):
        if (lo < a or (lo == a and lc)) and (b < hi or (b == hi and (hc or not closed_end))): return True
    return False
rng = random.Random(30); n = 0; bad1 = bad2 = 0; diffs = {}
cases = [[(day0, day1 - US, True, True)], [(day0, day1, True, False)], [(day0, day1, True, True)],
         [(day0, day1 - US, True, False)], [(day0 - US, day1 - US, False, True)], [(day0, day0 + US, True, False), (day0 + US, day1, True, False)]]
for _ in range(600):
    ps = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, b = sorted(rng.sample(GRID, 2)); ps.append((a, b, rng.random() < .6, rng.random() < .6))
    cases.append(ps)
for ps in cases:
    a1 = v1t.DateTimeInterval()
    for lo, hi, lc, hc in ps:
        p = v1t.DateTimeInterval(); p.interval = v1m.MultiInterval(lo.timestamp(), hi.timestamp(), start_closed=lc, end_closed=hc)
        a1 = a1.union(p)
    a2 = v2t.DateTimeInterval()
    for lo, hi, lc, hc in ps: a2 = a2 | v2t.DateTimeInterval(lo, hi, start_closed=lc, end_closed=hc)
    fps = [(ws(lo), ws(hi), lc, hc) for lo, hi, lc, hc in ps]
    r1, r2 = d0 in a1, d0 in a2
    o1 = subset(V2DAY if SAB else V1DAY, not SAB, fps); o2 = subset(V2DAY, False, fps)
    n += 1; bad1 += r1 != o1; bad2 += r2 != o2
    if r1 != o1 or r2 != o2: print("  ORACLE?", [(str(lo), str(hi), lc, hc) for lo, hi, lc, hc in ps], r1, o1, r2, o2)
    if r1 != r2: diffs[(r1, r2)] = diffs.get((r1, r2), 0) + 1
    if r1 != r2 and len(diffs) and diffs[(r1, r2)] == 1: print('  example differ:', [(str(lo), str(hi), lc, hc) for lo, hi, lc, hc in ps], 'v1', r1, 'v2', r2)
print('cases', n, ' v1 vs its oracle (closed day to 23:59:59.999999) wrong:', bad1, ' v2 vs its oracle (half-open day) wrong:', bad2)
print('v1 != v2 by (v1, v2):', diffs)
assert bad1 == 0 and bad2 == 0, 'oracle mismatch'
assert set(diffs) <= {(True, False)}, 'a difference other than the day-edge kind'
