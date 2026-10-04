"""A % m, A non-negative finite, m > 0 scalar (v1's Real branch), v1 vs v2 vs exact oracle; int, Fraction, float m"""
from common import *
import sys, time
SAB = '--sabotage' in sys.argv
rng = random.Random(7)
cases = []
hand = [
    ([(Fraction(1, 4), True, Fraction(1, 2), False)], Fraction(1, 2)),
    ([(0, True, 5, True)], 2),
    ([(0, True, 5, False)], 5),
    ([(1, True, 3, False)], 3),
    ([(1, False, 3, True)], 3),
    ([(3, True, 3, True)], 3),
    ([(2, True, 7, True)], 2),
    ([(2, False, 4, False)], 2),
    ([(12, True, Fraction(37, 2), True)], Fraction(15, 2)),
    ([(1, True, 2, True), (4, True, 5, True)], 3),
    ([(0, True, 0, True)], 3),
]
for a, m in hand: cases.append((canon(a), m))
for _ in range(500):
    a = rand_pieces(rng, lo=0, hi=12)
    m = rng.choice([Fraction(rng.randint(1, 24), rng.choice((1, 2, 4))), rng.randint(1, 6)])
    cases.append((a, m))
n = agree = 0; diffs = []; t0 = time.time()
for a, m in cases:
    n += 1
    s1, r1 = run(lambda: mk1(a) % m)
    s2, r2 = run(lambda: mk2(a) % m)
    if s1 != 'ok' or s2 != 'ok':
        diffs.append(('raise', show(a), m, r1 if s1 != 'ok' else show(v1_pieces(r1)), r2 if s2 != 'ok' else show(v2_pieces(r2)))); continue
    p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
    if SAB: p2 = canon(v2_pieces(mk2(a) % (m + 1)))
    if p1 == p2: agree += 1; continue
    bad = []
    for x in probe_points(p1, p2):
        if x < 0: continue
        i1, i2 = contains(p1, x), contains(p2, x)
        if i1 != i2: bad.append((str(x), i1, i2, attained_mod(x, a, [(Fraction(m), True, Fraction(m), True)])))
    diffs.append(('set', show(a), m, show(p1), show(p2), bad[:4]))
print(f'cases {n}, agree {agree}, differ {len(diffs)}  ({time.time()-t0:.1f}s)')
kinds = {}
for d in diffs:
    if d[0] == 'set':
        k = 'v1 extra only, all at 0, oracle says unattained' if all(x == '0' and i1 and not i2 and not o for x, i1, i2, o in d[5]) else 'OTHER'
    else:
        k = 'raise'
    kinds.setdefault(k, []).append(d)
for k, v in kinds.items():
    print(k, len(v))
    for d in v[:4]: print('   ', d)
