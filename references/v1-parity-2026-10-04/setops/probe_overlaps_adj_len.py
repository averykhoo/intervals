"""v1 overlaps(or_adjacent=True) vs the v2 composition len(A | B) < len(A) + len(B), and nan as other"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random
rng = random.Random(7)
t = Tally('overlaps_adj_len'); t2 = Tally('overlaps_adj_any_adjoins')
for _ in range(1500):
    pa, pb = rand_pieces(rng), rand_pieces(rng)
    a1, a2 = twin(pa); b1, b2 = twin(pb)
    r1 = a1.overlaps(b1, or_adjacent=True)
    t.check((pa, pb), None if r1 == (len(a2 | b2) < len(a2) + len(b2)) else f'v1 {r1}')
    t2.check((pa, pb), None if r1 == (a2.overlaps(b2) or any(p.adjoins(q) for p in a2 for q in b2)) else f'v1 {r1}')
t.report(); t2.report()
# hand case: adjacency to an inner piece
A1, A2 = twin([(1, 2, False, False)]); B1, B2 = twin([(-5, -4, True, True), (0, 1, True, True), (5, 6, True, True)])
print('inner-adjacent: v1', A1.overlaps(B1, or_adjacent=True), 'v2 len', len(A2 | B2) < len(A2) + len(B2), 'v2 naive A.adjoins(B)', A2.overlaps(B2) or A2.adjoins(B2))
for f in [lambda: M1(0, 1).overlaps(float('nan')), lambda: M2(0, 1).overlaps(float('nan'))]:
    try: print('overlaps(nan):', f())
    except Exception as e: print('overlaps(nan): raise', type(e).__name__, e)
s = Tally('sabotage'); s.check('x', None if True == (len(A2 | B2) == len(A2) + len(B2)) else 'caught'); assert s.report(0) == 1
