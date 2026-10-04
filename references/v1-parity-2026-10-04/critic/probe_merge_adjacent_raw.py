"""re-check setops row 'merge_adjacent() (distance 0, sort=) [EQUIVALENT_RENAMED]': the sweep compared v1
merge_adjacent() on already-merged public sets (identity) with the v2 set itself, which cannot fail. here: raw,
unmerged, unsorted endpoint lists (the only input where merge_adjacent does anything) vs v2 from_pieces."""
import sys, random, warnings, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m
import intervals as v2
SAB = len(sys.argv) > 1
GRID = [F(-2), F(-1), F(-1, 2), F(0), F(1, 3), F(1), F(3, 2), F(2), F(3)]
def raw_v1(pieces):
    a = v1m.MultiInterval()
    for lo, hi, lc, hc in pieces:
        a.endpoints += [(lo, 0 if lc else 1), (hi, 0 if hc else -1)]
    return a
def member(pieces, x):
    return any((lo < x or (lo == x and lc)) and (x < hi or (x == hi and hc)) for lo, hi, lc, hc in pieces)
rng = random.Random(77); n = bad = bad_brute = 0
for _ in range(500):
    ps = []
    for _ in range(rng.randint(1, 5)):
        lo, hi = sorted(rng.sample(GRID, 2)) if rng.random() < .85 else (rng.choice(GRID),) * 2
        lc, hc = (True, True) if lo == hi else (rng.random() < .5, rng.random() < .5)
        ps.append((lo, hi, lc, hc))
    rng.shuffle(ps)
    a1 = raw_v1(ps).merge_adjacent()          # default sort=True
    a2 = v2.MultiInterval.from_pieces(ps)
    if SAB and n == 3: a2 = a2 | v2.MultiInterval(F(5))
    pts = sorted({g + d for g in GRID for d in (F(-1, 100), 0, F(1, 100))} | {F(5)})
    m1 = [x in a1 for x in pts]; m2 = [x in a2 for x in pts]; mb = [member(ps, x) for x in pts]
    st1 = [(e[i][0], e[i][1] == 0, e[i + 1][0], e[i + 1][1] == 0) for e in [a1.endpoints] for i in range(0, len(e), 2)]
    st2 = [(p.inf, p.inf_closed, p.sup, p.sup_closed) for p in a2]
    n += 1; bad += (m1 != m2) or (st1 != st2); bad_brute += m2 != mb
print('cases', n, 'v1 merge_adjacent vs v2 from_pieces mismatches', bad, ' v2 vs brute', bad_brute)
assert bad == 0
