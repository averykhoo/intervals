"""stress: up to 8 pieces per operand, the two-pointer walks of v1 (contains, overlapping, difference_update) vs v2"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
import common
from common import *
import random
common.GRID[:] = list(range(-6, 7)) + [Fraction(k, 3) for k in range(-10, 11)] + [0.5, -1.5, 2.25]
rng = random.Random(8)
names = ['contains', 'issubset', 'overlaps', 'overlapping', 'overlapping_adj', 'difference', 'intersection', 'union', 'symdiff', 'invert', 'expand']
T = {k: Tally(k) for k in names}
for _ in range(2000):
    pa, pb = rand_pieces(rng, 8), rand_pieces(rng, 8)
    if rng.random() < 0.3 and pa:
        pb = rng.sample(pa, rng.randint(1, len(pa)))
    a1, a2 = twin(pa); b1, b2 = twin(pb)
    if a1.is_empty or b1.is_empty: pass
    T['contains'].check((pa, pb), None if (b1 in a1) == (b2 in a2) or (b1.is_empty and a1.is_empty) else f'{b1 in a1} {b2 in a2}')
    T['issubset'].check((pa, pb), None if a1.issubset(b1) == a2.issubset(b2) else 'x')
    T['overlaps'].check((pa, pb), None if a1.overlaps(b1) == a2.overlaps(b2) else 'x')
    T['overlapping'].check((pa, pb), same_set_v1_v2(a1.overlapping(b1), M2().union(*[p for p in a2 if p.overlaps(b2)])))
    T['overlapping_adj'].check((pa, pb), same_set_v1_v2(a1.overlapping(b1, or_adjacent=True), M2().union(*[p for p in a2 if p.overlaps(b2) or any(p.adjoins(q) for q in b2)])))
    T['difference'].check((pa, pb), same_set_v1_v2(a1.difference(b1), a2.difference(b2)))
    T['intersection'].check((pa, pb), same_set_v1_v2(a1.intersection(b1), a2 & b2))
    T['union'].check((pa, pb), same_set_v1_v2(a1.union(b1), a2 | b2))
    T['symdiff'].check((pa, pb), same_set_v1_v2(a1.symmetric_difference(b1), a2 ^ b2))
    T['invert'].check(pa, same_set_v1_v2(~a1, ~a2 & M2(-inf, inf, start_closed=False, end_closed=False)))
    e = rng.choice([0, Fraction(1, 3), 0.5, 1])
    T['expand'].check((pa, e), same_set_v1_v2(a1.expand(e), a2.expand(e)))
for t in T.values(): t.report(2)
s = Tally('sabotage'); a1, a2 = twin([(0, 1, True, True), (2, 3, True, True)]); s.check('x', same_set_v1_v2(a1.overlapping(M1(0.5)), a2)); assert s.report(0) == 1
