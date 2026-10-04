"""seeded sweep of __pow__ on positive bases (v1's only working branch besides [0] and [0, b]).
exact oracle (Fraction) for integral exponents; v1 vs v2 vs truth."""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random
from fractions import Fraction as F
from common import *

rng = random.Random(20261004)

def rand_end(kind):
    if kind == 'int':
        return rng.randint(1, 6)
    if kind == 'frac':
        return F(rng.randint(1, 30), rng.randint(1, 8))
    return rng.choice([0.1, 0.3, 0.5, 1.1, 1.5, 2.0, 2.7, 3.3, 7.25])

def rand_base(kind):
    """1-2 disjoint positive pieces, ends of `kind`, maybe open, sometimes from 0 open or to inf"""
    n = rng.randint(1, 2)
    vals = sorted({rand_end(kind) for _ in range(2 * n + 2)})
    while len(vals) < 2 * n:
        vals = sorted({rand_end(kind) for _ in range(2 * n + 2)})
    vals = vals[:2 * n]
    ps = []
    for i in range(n):
        lo, hi = vals[2 * i], vals[2 * i + 1]
        lc, hc = rng.random() < 0.5, rng.random() < 0.5
        if rng.random() < 0.15:
            hi, hc = lo, True
            lc = True
        ps.append([lo, hi, lc, hc])
    if rng.random() < 0.15:
        ps[0][0], ps[0][2] = 0, False
    if rng.random() < 0.15 and ps[-1][0] != ps[-1][1]:
        ps[-1][1], ps[-1][3] = math.inf, False
    return [tuple(p) for p in ps]

def truth_pown(ps, n):
    out = []
    for lo, hi, lc, hc in ps:
        if n == 0:
            out.append((F(1), F(1), True, True)); continue
        def p(v, closed):
            if math.isinf(v):
                return (math.inf if n > 0 else F(0)), closed
            if v == 0:
                return (F(0) if n > 0 else math.inf), closed
            return F(v) ** n, closed
        a, ac = p(lo, lc)
        b, bc = p(hi, hc)
        if n > 0:
            out.append((a, b, ac, bc))
        else:
            out.append((b, a, bc, ac))
    return out

def noise(x, *pss):
    for ps in pss:
        for p in ps:
            for e in p[:2]:
                if not math.isinf(e) and abs(F(x) - F(e)) <= abs(F(e)) * F(1, 10 ** 12) + F(1, 10**300):
                    return True
    return False

stats = {}
def tally(key):
    stats[key] = stats.get(key, 0) + 1

examples = {}
N = 0
for kind in ('int', 'frac', 'float'):
    for _ in range(150):
        base = rand_base(kind)
        b1, b2 = mk1(base), mk2(base)
        for n in (-3, -2, -1, 0, 1, 2, 3, 2.0, -1.0, F(4, 2)):
            N += 1
            r1, e1 = run(lambda: b1 ** n)
            r2, e2 = run(lambda: b2 ** n)
            if e2:
                tally('v2 raises'); examples.setdefault('v2 raises', (base, n, e2)); continue
            if e1:
                tally(f'v1 raises {e1.split(":")[0]} n={n!r}'); examples.setdefault(f'v1 raises {e1.split(":")[0]}', (base, n, e1, r2)); continue
            if not well_formed1(r1):
                tally(f"{kind}: v1 MALFORMED n={n!r}"); examples.setdefault(f"{kind}: v1 MALFORMED", (base, n, r1.endpoints, str(r2))); continue
            p1, p2 = pieces1(r1), pieces2(r2)
            t = truth_pown(base, int(n))
            d12 = compare(p1, p2)
            if not d12:
                tally(f'{kind}: agree'); continue
            d1t, d2t = compare(p1, t), compare(p2, t)
            if kind == 'float':
                # nearest class rounds; only structural (non-noise) differences matter
                s1 = [d for d in d1t if not noise(d[0], t)]
                s2 = [d for d in d2t if not noise(d[0], t)]
                key = f'float: v1 vs v2 differ; structural v1 wrong={bool(s1)} v2 wrong={bool(s2)}'
            else:
                key = f'{kind}: v1 vs v2 differ n={n!r}; v1 wrong={bool(d1t)} v2 wrong={bool(d2t)}'
            tally(key); examples.setdefault(key, (base, n, s1(r1), str(r2), d12[:3]))
print('cases', N)
for k, v in sorted(stats.items()):
    print(f'{v:5d}  {k}')
print()
for k, v in examples.items():
    print(k, '\n   ', v)
# sabotage: a wrong oracle (n+1) must be caught by compare
base = [(1, 2, True, True)]
assert compare(pieces2(mk2(base) ** 2), truth_pown(base, 3)), 'sabotage not caught'
print('sabotage caught')
