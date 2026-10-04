"""positive base ** (non-integral number | interval) exponent: v1 corner rule vs v2 1788 pow vs an exact oracle.
ends chosen so every corner value is rational (perfect-square bases, half-integer exponents)."""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random
from fractions import Fraction as F
from common import *

rng = random.Random(7)
BASES = [F(1, 16), F(1, 9), F(1, 4), F(1), F(4), F(9), F(16)]
EXPS = [F(-2), F(-3, 2), F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(3, 2), F(2)]

def ipow(x, y):
    """x ** y exactly for x a rational square and y a half-integer"""
    if y.denominator == 1:
        return x ** int(y)
    r = F(math.isqrt(x.numerator), math.isqrt(x.denominator))
    assert r * r == x
    return r ** int(2 * y)

def rand_pieces(pool, n):
    vals = sorted(rng.sample(pool, 2 * n))
    out = []
    for i in range(n):
        lo, hi = vals[2 * i], vals[2 * i + 1]
        if rng.random() < 0.2:
            out.append((lo, lo, True, True))
        else:
            out.append((lo, hi, rng.random() < 0.5, rng.random() < 0.5))
    return out

def truth(bs, es):
    out = []
    for xl, xh, xlc, xhc in bs:
        for yl, yh, ylc, yhc in es:
            corners = [(ipow(x, y), xc and yc) for x, xc in ((xl, xlc), (xh, xhc)) for y, yc in ((yl, ylc), (yh, yhc))]
            vals = [v for v, _ in corners]
            lo, hi = min(vals), max(vals)
            one = (xl < 1 < xh) or (xl == 1 and xlc) or (xh == 1 and xhc) or (yl < 0 < yh) or (yl == 0 and ylc) or (yh == 0 and yhc)
            loc = any(v == lo and c for v, c in corners) or (lo == 1 and one)
            hic = any(v == hi and c for v, c in corners) or (hi == 1 and one)
            out.append((lo, hi, loc, hic))
    return out

def classify(diffs, t):
    ends = {e for p in t for e in p[:2]}
    kinds = set()
    for x, _, _ in diffs:
        if any(F(x) == e for e in ends):
            kinds.add('flag-at-exact-end')
        elif any(abs(F(x) - e) <= abs(e) * F(1, 10 ** 12) for e in ends):
            kinds.add('rounding-noise')
        else:
            kinds.add('STRUCTURAL')
    return kinds

stats, ex = {}, {}
N = 0
for trial in range(600):
    bs = rand_pieces(BASES, rng.randint(1, 2))
    number_exp = trial % 3 == 0
    if number_exp:
        y = rng.choice([e for e in EXPS if e.denominator == 2])
        es = [(y, y, True, True)]
        e1, e2 = float(y), (float(y) if trial % 2 else y)
    else:
        es = rand_pieces(EXPS, rng.randint(1, 2))
        e1, e2 = mk1(es), mk2(es)
    b1, b2 = mk1(bs), mk2(bs)
    N += 1
    r1, x1 = run(lambda: b1 ** e1)
    r2, x2 = run(lambda: b2 ** e2)
    t = truth(bs, es)
    if x1 or (r1 is not None and not well_formed1(r1)):
        k1 = {'v1 ' + (x1.split(':')[0] if x1 else 'MALFORMED')}
    else:
        k1 = classify(compare(pieces1(r1), t), t)
    k2 = {'v2 ' + x2.split(':')[0]} if x2 else classify(compare(pieces2(r2), t), t)
    key = f'{"num" if number_exp else "ivl"} v1:{sorted(k1)} v2:{sorted(k2)}'
    stats[key] = stats.get(key, 0) + 1
    ex.setdefault(key, (bs, es, x1 or s1(r1), x2 or str(r2), [(str(a), str(b), c, d) for a, b, c, d in t]))
print('cases', N)
for k, v in sorted(stats.items()):
    print(f'{v:5d}  {k}')
for k, v in ex.items():
    if 'STRUCTURAL' in k or 'flag' in k or 'raise' in k or 'Err' in k or 'MAL' in k:
        print(k, '\n   ', v)
# sabotage: claim [1,4] ** [1/2] has an open top -> must be caught
assert compare(pieces2(mk2([(F(1), F(4), True, True)]) ** mk2([(F(1, 2),) * 2 + (True, True)])), [(F(1), F(2), True, False)])
print('sabotage caught')
