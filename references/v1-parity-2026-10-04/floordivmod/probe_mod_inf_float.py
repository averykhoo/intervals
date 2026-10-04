"""v1 mod paths that touch infinity or floats: B = [b, inf) (open at inf: v1 cannot close it), m = inf (Real),
and float operands (A and m/B floats), v1 vs v2 as sets. exact truth for B=[b,inf): x mod y over y in B; for
y > x >= 0, x mod y = x, so the finite oracle on B clipped to [b, max(A)+1] plus A itself (if B reaches past A) decides."""
from common import *
import sys
SAB = '--sabotage' in sys.argv
rng = random.Random(31337)
stats = {}; shown = {}
def note(k, e=None):
    stats[k] = stats.get(k, 0) + 1
    if k not in shown and e is not None: shown[k] = e
for _ in range(300):
    A = rand_pieces(rng, lo=0, hi=12)
    if A[0][0] == 0 and A[0][1]: A[0] = (A[0][0], False) + A[0][2:]
    A = canon([p for p in A if nonempty(*p)])
    if not A: continue
    b = Fraction(rng.randint(1, 40), 4); bc = rng.random() < 0.5
    B = [(b, bc, INF, False)]
    s1, r1 = run(lambda: mk1(A) % mk1(B)); s2, r2 = run(lambda: mk2(A) % mk2(B))
    if s1 != 'ok' or s2 != 'ok': note('B=[b,inf): raised', (show(A), show(B), r1, r2)); continue
    p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
    if SAB: p2 = canon(v2_pieces(mk2(A) % mk2([(b + 1, bc, INF, False)])))
    if p1 == p2: note('B=[b,inf): equal'); continue
    # exact: values from y in B∩[b, top] plus x itself when some y in B exceeds x (always: B unbounded)
    top = max(hi for _, _, hi, _ in A) + 1
    Bc = [(b, bc, top, True)]
    bad = []
    for x in probe_points(p1, p2):
        if x < 0: continue
        i1, i2 = contains(p1, x), contains(p2, x)
        if i1 != i2:
            truth = attained_mod(x, A, Bc) or contains(A, x)
            bad.append((str(x), i1, i2, truth))
    k = 'B=[b,inf): differ, v2 right at every differing point' if all(t == i2 for _, _, i2, t in bad) else 'B=[b,inf): differ, v1 right somewhere'
    note(k, (show(A), show(B), show(p1), show(p2), bad[:4]))
# Real inf divisor
for _ in range(200):
    A = rand_pieces(rng, lo=0, hi=12)
    s1, r1 = run(lambda: mk1(A) % INF); s2, r2 = run(lambda: mk2(A) % INF)
    if s1 != 'ok' or s2 != 'ok': note('A % inf: raised', (show(A), r1, r2)); continue
    p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
    note('A % inf: equal (both == A)' if p1 == p2 == A else 'A % inf: differ', None if p1 == p2 == A else (show(A), show(p1), show(p2)))
# floats: A float pieces, m float; compare as sets of exact rationals; judge differing points by exact oracle
def fl(v): return float(v)
for _ in range(300):
    A = [(fl(lo + Fraction(rng.randint(0, 9), 10)), lc, fl(hi + Fraction(rng.randint(0, 9), 10)), hc) for lo, lc, hi, hc in rand_pieces(rng, lo=0, hi=12)]
    A = [p for p in A if p[0] < p[2] or (p[0] == p[2] and p[1] and p[3])]
    A = [p for i, p in enumerate(A) if all(q[2] < p[0] for q in A[:i])]
    if not A: continue
    m = rng.choice([0.1, 0.3, 0.7, 1.1, 2.5, 3.3, 0.2])
    for form in ('scalar', 'set'):
        if form == 'scalar':
            f1, f2 = (lambda: mk1(A) % m), (lambda: mk2(A) % m)
            Bx = [(Fraction(m), True, Fraction(m), True)]
        else:
            B = [(m, True, m * 3, rng.random() < 0.5)]
            if A[0][0] == 0: continue
            f1, f2 = (lambda: mk1(A) % mk1(B)), (lambda: mk2(A) % mk2(B))
            Bx = [(Fraction(B[0][0]), True, Fraction(B[0][2]), B[0][3])]
        s1, r1 = run(f1); s2, r2 = run(f2)
        tag = f'float A % {form}'
        if s1 != 'ok' or s2 != 'ok': note(f'{tag}: raised', (A, m, r1, r2)); continue
        p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
        if p1 == p2: note(f'{tag}: equal'); continue
        Ax = canon([(Fraction(lo), lc, Fraction(hi), hc) for lo, lc, hi, hc in A])
        bad = []
        for x in probe_points(p1, p2):
            if x < 0: continue
            i1, i2 = contains(p1, x), contains(p2, x)
            if i1 != i2: bad.append((str(x), i1, i2, attained_mod(x, Ax, Bx)))
        v1_unsound = any(t and not i1 for _, i1, _, t in bad)
        v2_unsound = any(t and not i2 for _, _, i2, t in bad)
        note(f'{tag}: differ (v1 misses attained: {v1_unsound}, v2 misses attained: {v2_unsound})', (A, m, show(p1), show(p2), bad[:3]))
for k in sorted(stats): print(f'{k}: {stats[k]}', shown.get(k, ''))
