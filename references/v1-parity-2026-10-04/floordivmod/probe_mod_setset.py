"""A % B, both strictly positive finite sets (v1's __modulo path), v1 vs v2 vs an exact oracle"""
from common import *
import sys
SAB = '--sabotage' in sys.argv
rng = random.Random(20261004)
cases = []
# hand-picked
H = lambda *p: [tuple(x) for x in p]
hand = [
    ([(1, True, 2, True)], [(3, True, 4, True)]),
    ([(3, True, Fraction(79, 10), True)], [(Fraction(79, 10), True, Fraction(126, 10), True)]),
    ([(3, True, 8, True)], [(8, True, 12, True)]),
    ([(3, True, 8, True)], [(8, False, 12, True)]),
    ([(1, True, 2, True)], [(3, False, 4, False)]),
    ([(3, True, 7, True)], [(4, True, 4, True), (5, True, 5, True)]),
    ([(12, True, Fraction(37, 2), True)], [(Fraction(15, 2), True, Fraction(15, 2), True)]),
    ([(0, False, 5, True)], [(2, True, 3, True)]),
    ([(1, False, 9, False)], [(0, False, 2, False)]),
    ([(5, True, 5, True)], [(2, True, 3, True)]),
    ([(6, True, 6, True)], [(2, False, 3, False)]),
    ([(6, True, 6, True)], [(2, True, 3, True)]),
    ([(1, True, 2, True), (5, True, 6, False)], [(2, True, 2, True), (7, False, 8, True)]),
    ([(Fraction(1, 4), True, Fraction(1, 2), False)], [(Fraction(1, 2), True, Fraction(1, 2), True)]),
    ([(10, True, 11, True)], [(3, True, 5, True)]),
]
for a, b in hand:
    cases.append((canon(a), canon(b)))
for _ in range(400):
    a = rand_pieces(rng, lo=0, hi=12)
    b = rand_pieces(rng, lo=0, hi=8)
    # v1 needs both strictly positive: drop a closed 0
    if a[0][0] == 0 and a[0][1]: a[0] = (a[0][0], False) + a[0][2:]
    if b[0][0] == 0 and b[0][1]: b[0] = (b[0][0], False) + b[0][2:]
    a, b = canon([p for p in a if nonempty(*p)]), canon([p for p in b if nonempty(*p)])
    if a and b: cases.append((a, b))

n = agree = 0; diffs = []
for a, b in cases:
    n += 1
    s1, r1 = run(lambda: mk1(a) % mk1(b))
    s2, r2 = run(lambda: mk2(a) % mk2(b))
    if s1 != 'ok' or s2 != 'ok':
        diffs.append(('raise', show(a), show(b), r1 if s1 != 'ok' else show(v1_pieces(r1)), r2 if s2 != 'ok' else show(v2_pieces(r2))))
        continue
    p1, p2 = canon(v1_pieces(r1)), canon(v2_pieces(r2))
    if SAB:
        p2 = canon(v2_pieces(mk2(a) % mk2(canon([(lo + 1, lc, hi + 1, hc) for lo, lc, hi, hc in b]))))
    if p1 == p2:
        agree += 1
        continue
    # decide by oracle at probe points
    bad = []
    for x in probe_points(p1, p2):
        if x < 0: continue
        i1, i2 = contains(p1, x), contains(p2, x)
        if i1 != i2:
            o = attained_mod(x, a, b)
            bad.append((x, i1, i2, o))
    diffs.append(('set', show(a), show(b), show(p1), show(p2), bad[:6]))
print(f'cases {n}, agree {agree}, differ {len(diffs)}')
for d in diffs[:30]:
    print(d)
# oracle self-check on agreeing cases: every probe point of a few agreeing results must match oracle
chk = 0; wrong = 0
for a, b in cases[:60]:
    r2 = canon(v2_pieces(mk2(a) % mk2(b)))
    for x in probe_points(r2):
        if x < 0: continue
        chk += 1
        if contains(r2, x) != attained_mod(x, a, b):
            wrong += 1
            if wrong < 5: print('ORACLE/V2 DISAGREE', show(a), show(b), x, contains(r2, x))
print(f'oracle cross-check points {chk}, disagreements {wrong}')
