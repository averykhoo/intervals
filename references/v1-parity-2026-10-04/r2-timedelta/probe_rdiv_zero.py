"""td / T where 0 is in T or t == 0: tally v1's composition result (MultiInterval(t) / A.interval) by shape,
and check v1's answer is a superset of v2's exact one (soundness) and whether it adds points no ratio reaches"""
import sys
sys.path[:0] = ['.', 'archive/v1', '.scratch/v1-parity/r2-timedelta']
import io, contextlib, random, warnings
from fractions import Fraction
from collections import Counter
with contextlib.redirect_stdout(io.StringIO()):
    import probe_rdiv_neg as P   # reuse builders (its own prints suppressed)
import multi_interval as v1m

rng = random.Random(4242)
shapes = Counter()
v1_superset = v1_not_superset = extra_points = 0
ex = []
for i in range(500):
    pieces = P.rand_pieces(rng)
    t = Fraction(rng.randint(-8, 8), 2)
    num = P.td(t)
    rng.choice(['td', 'pd'])
    sp = [(Fraction(a), sc, Fraction(b), ec) for a, b, sc, ec in pieces]
    has0 = P.member(sp, 0)
    if not (has0 or t == 0):
        continue
    a1, a2 = P.build1(pieces), P.build2(pieces)
    r1, _ = P.try_(lambda: v1m.MultiInterval(num.total_seconds()) / a1.interval)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r2 = num / a2
    c2 = P.canon_mi2(r2)
    key = ('0inA' if has0 else '0notinA') + (' t=0' if t == 0 else ' t!=0')
    if isinstance(r1, tuple):
        shapes[(key, 'v1 raises ' + r1[1])] += 1
        continue
    c1 = P.canon_mi1(r1)
    shapes[(key, 'v1 ' + ('whole line' if c1 == ((float('-inf'), False, float('inf'), False),) else 'nan ends' if any(x != x for p in c1 for x in (p[0], p[2])) else 'other'))] += 1
    if any(x != x for p in c1 for x in (p[0], p[2])):
        continue
    grid = [Fraction(k, 8) for k in range(-200, 201)]
    sup = all(P.member(c1, q) for q in grid if P.member(c2, q))
    v1_superset += sup
    v1_not_superset += not sup
    if P.member(c1, 0) and not P.member(c2, 0):
        extra_points += 1
        if len(ex) < 3:
            ex.append((pieces, str(t), c1, c2, [str(x.message)[:60] for x in w]))
for k, v in sorted(shapes.items()):
    print(k, v)
print(f'v1 superset of v2 (grid of 1/8): {v1_superset}, not: {v1_not_superset}; v1 holds 0 where v2 (exact) does not: {extra_points}')
for e in ex:
    print('  e.g.', e)
