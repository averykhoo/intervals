"""v1 `merge_adjacent()` (distance 0, sort=True, and sort=False on a pre-sorted list) on RAW unmerged endpoint lists
(the only input where it does anything), vs v2 `MultiInterval.from_pieces` on the same pieces, vs a brute-force
membership oracle. grid includes +-inf (open), points, touching open/open, closed/open, duplicates, nesting.
`sab`: the oracle treats an open/open touch as merged, which must be caught."""
import sys, random, warnings, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m
import intervals as v2
SAB = len(sys.argv) > 1
INF = math.inf
GRID = [-INF, F(-3), F(-1), F(-1, 3), F(0), F(1, 7), F(1), F(2), F(5, 2), INF]
def raw_v1(pieces):
    a = v1m.MultiInterval()
    for lo, hi, lc, hc in pieces: a.endpoints += [(lo, 0 if lc else 1), (hi, 0 if hc else -1)]
    return a
def member(pieces, x):
    return any((lo < x or (lo == x and lc)) and (x < hi or (x == hi and hc)) for lo, hi, lc, hc in pieces)
def n_components(pieces, pts):     # brute-force count of maximal runs: between consecutive probe points too
    m = [member(pieces, x) for x in pts]
    return sum(1 for i, v in enumerate(m) if v and (i == 0 or not m[i - 1]))
fin = [g for g in GRID if not (isinstance(g, float) and math.isinf(g))]
PTS = sorted({g + d for g in fin for d in (F(-1, 1000), 0, F(1, 1000))} | {F(-100), F(100)})
if SAB: PTS = PTS  # sabotage happens in the oracle below
rng = random.Random(912); stats = dict(cases=0, sorted_false=0, v1_v2_struct=0, v1_v2_member=0, v2_brute=0, v1_brute=0)
cases = [[(F(0), F(1), True, False), (F(1), F(2), False, True)],            # open/open touch: two pieces
         [(F(0), F(1), True, False), (F(1), F(2), True, True)],             # closed/open touch: one
         [(F(1), F(1), True, True), (F(0), F(1), True, False)],             # a point closing an open end
         [(F(0), F(2), True, True), (F(0), F(2), False, False)],            # duplicate with different flags
         [(-INF, F(0), False, False), (F(0), INF, False, False)],           # everything but 0
         [(-INF, F(0), False, True), (F(0), INF, True, False)]]             # duplicate touching point
for _ in range(1500):
    ps = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < .15:
            g = rng.choice(fin); ps.append((g, g, True, True)); continue
        lo, hi = sorted(rng.sample(GRID, 2))
        lc = rng.random() < .5 and not math.isinf(lo); hc = rng.random() < .5 and not math.isinf(hi)
        ps.append((lo, hi, lc, hc))
    rng.shuffle(ps); cases.append(ps)
for ps in cases:
    for sort in (True, False):
        src = ps if sort else sorted(ps, key=lambda p: (p[0], 0 if p[2] else 1, p[1], 0 if p[3] else -1))
        if not sort: stats['sorted_false'] += 1
        a1 = raw_v1(src); r = a1.merge_adjacent(sort=sort)
        assert r is a1                                   # v1: in place, returns self
        a2 = v2.MultiInterval.from_pieces(src)
        st1 = [(e[i][0], e[i][1] == 0, e[i + 1][0], e[i + 1][1] == 0) for e in [a1.endpoints] for i in range(0, len(e), 2)]
        st2 = [(p.inf, bool(p.inf_closed), p.sup, bool(p.sup_closed)) for p in a2]
        oracle_ps = [(lo, hi, True, True) if (SAB and not lc and not hc and lo != hi) else (lo, hi, lc, hc) for lo, hi, lc, hc in src] if SAB else src
        m1 = [x in a1 for x in PTS]; m2 = [x in a2 for x in PTS]; mb = [member(oracle_ps, x) for x in PTS]
        stats['cases'] += 1
        stats['v1_v2_struct'] += st1 != st2; stats['v1_v2_member'] += m1 != m2
        stats['v2_brute'] += m2 != mb; stats['v1_brute'] += m1 != mb
print(stats)
assert stats['v1_v2_struct'] == stats['v1_v2_member'] == stats['v2_brute'] == stats['v1_brute'] == 0
