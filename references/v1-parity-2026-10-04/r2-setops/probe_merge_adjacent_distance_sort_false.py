"""v1 merge_adjacent(d, sort=False) on pair-sorted RAW (unmerged) lists vs the v2 gap-fill composition recorded by
setops/report.md row 47, applied to v2 from_pieces(ps). run from repo root; `sab` arg flips one v2 result."""
import sys, random, warnings, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-setops']
warnings.simplefilter('ignore')
import multi_interval as v1m
import intervals as v2
from probe_merge_adjacent_sort_false import eps, raw_v1, rand_pieces, structure, v2_struct, PTS  # noqa (reruns that probe's prints)

def gapfill(A, d):
    if A.is_empty:
        return A
    gaps = [g for g in A.hull.difference(A) if g.sup - g.inf < d or (g.sup - g.inf == d and not (g.inf_closed and g.sup_closed))]
    return A.union(*gaps)

SAB = len(sys.argv) > 1 and sys.argv[-1] == 'sab2'
rng = random.Random(99); n = bad = 0
for d in (0, F(1, 4), F(1, 3), F(1, 2), F(2, 3), 1, 2, math.inf):
    for _ in range(100):
        ps = sorted(rand_pieces(rng), key=eps)
        r1 = raw_v1(ps).merge_adjacent(d, sort=False)
        r2 = gapfill(v2.MultiInterval.from_pieces(ps), d)
        if SAB and n == 9:
            r2 = r2 | v2.MultiInterval(F(5))
        n += 1
        bad += structure(r1) != v2_struct(r2) or [x in r1 for x in PTS] != [x in r2 for x in PTS]
print(f'[distance d, sort=False, pair-sorted raw] cases={n} v1 vs v2 gap-fill(from_pieces) mismatches={bad}')
assert bad == (1 if SAB else 0)
print('ok')
