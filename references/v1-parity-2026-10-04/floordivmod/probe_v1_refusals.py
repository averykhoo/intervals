"""inputs v1 refused (NotImplemented -> TypeError, ZeroDivisionError, ValueError): record v1's outcome, then check
v2's answer against the exact oracle at every probe point. forms: A % B any signs / zero-crossing, A % m (m < 0),
m % A (rmod), divmod(A, B), divmod(m, A)"""
from common import *
import sys
SAB = '--sabotage' in sys.argv
rng = random.Random(8080)
v1_out = {}; v2_ok = {}; v2_bad = {}; shown = {}
def shift(p, d): return canon([(lo + d, lc, hi + d, hc) for lo, lc, hi, hc in p])
def check(tag, res, A, B):
    p = canon(v2_pieces(res))
    if SAB: p = shift(p, Fraction(1, 2))
    bad = [(str(x), contains(p, x)) for x in probe_points(p, extra=[Fraction(0)]) if contains(p, x) != attained_mod(x, A, B)]
    if bad:
        v2_bad[tag] = v2_bad.get(tag, 0) + 1
        shown.setdefault(tag, (show(A), show(B), show(p), bad[:3]))
    else:
        v2_ok[tag] = v2_ok.get(tag, 0) + 1
def v1rec(tag, f):
    s, r = run(f)
    k = (tag, r.split(':')[0] if s == 'err' else 'ok')
    v1_out[k] = v1_out.get(k, 0) + 1
for _ in range(250):
    A = shift(rand_pieces(rng, lo=0, hi=12), -6)
    B = shift(rand_pieces(rng, lo=0, hi=8), -4)
    if not [1 for lo, _, hi, _ in B if not (lo == hi == 0)]: continue
    m = rng.choice([Fraction(rng.randint(-12, 12), rng.choice((1, 2, 4))), rng.randint(-5, 5)])
    tagAB = 'A % B (any signs)'
    v1rec(tagAB, lambda: mk1(A) % mk1(B))
    check(tagAB, mk2(A) % mk2(B), A, B)
    if m != 0:
        mp = [(Fraction(m), True, Fraction(m), True)]
        if m < 0:
            v1rec('A % m (m < 0)', lambda: mk1(A) % m)
            check('A % m (m < 0)', mk2(A) % m, A, mp)
        else:
            v1rec('A % m (m > 0, A any sign)', lambda: mk1(A) % m)
            check('A % m (m > 0, A any sign)', mk2(A) % m, A, mp)
    v1rec('m % B (rmod)', lambda: m % mk1(B))
    check('m % B (rmod)', m % mk2(B), [(Fraction(m), True, Fraction(m), True)], B)
    # divmod: the pair equals (//, %) in v2; v1 refuses
    v1rec('divmod(A, B)', lambda: divmod(mk1(A), mk1(B)))
    v1rec('divmod(m, B)', lambda: divmod(m, mk1(B)))
    q, r = divmod(mk2(A), mk2(B))
    ok = (q == mk2(A) // mk2(B)) and (r == mk2(A) % mk2(B))
    q2, r2 = divmod(m, mk2(B))
    ok2 = (q2 == m // mk2(B)) and (r2 == m % mk2(B))
    if SAB: ok = q == mk2(A) % mk2(B)
    for tag, good in (('divmod(A, B) == (A // B, A % B)', ok), ('divmod(m, B) == (m // B, m % B)', ok2)):
        (v2_ok if good else v2_bad)[tag] = (v2_ok if good else v2_bad).get(tag, 0) + 1
print('v1 outcomes:'); [print(f'  {k}: {v}') for k, v in sorted(v1_out.items())]
print('v2 vs exact oracle:')
for t in sorted(set(v2_ok) | set(v2_bad)): print(f'  {t}: ok {v2_ok.get(t, 0)}, wrong {v2_bad.get(t, 0)}', shown.get(t, ''))
