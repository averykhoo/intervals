"""A // m (v1: apply_monotonic_binary_function(truediv) then floor ends) and m // A (v1: m * A.reciprocal() then floor
ends) vs v2, judged by the exact set of floor values; int, Fraction, float scalars of both signs"""
from common import *
import sys
SAB = '--sabotage' in sys.argv
rng = random.Random(4242)

def floor_set_div(A, m, reflected):
    """exact floor values of x/m (or m/x), A finite, 0 not in A when reflected"""
    out = set()
    m = ex(m)
    for lo, lc, hi, hc in A:
        if reflected:
            if lo <= 0 <= hi: raise ValueError('0 in A')
            qs = [(m / lo, lc), (m / hi, hc)]
        else:
            qs = [(lo / m, lc), (hi / m, hc)]
        qlo, qhi = min(q[0] for q in qs), max(q[0] for q in qs)
        qlc = any(c for q, c in qs if q == qlo); qhc = any(c for q, c in qs if q == qhi)
        for n in range(math.floor(qlo) - 1, math.floor(qhi) + 2):
            if inter(qlo, qlc, qhi, qhc, Fraction(n), True, Fraction(n + 1), False):
                out.add(n)
    return out

def ints_in(pcs, lo, hi):
    return {n for n in range(lo, hi + 1) if contains(pcs, Fraction(n))}

stats = {}
ex_shown = {}
def note(k, e):
    stats[k] = stats.get(k, 0) + 1
    if k not in ex_shown: ex_shown[k] = e
N = 0
for _ in range(600):
    A = canon([(lo - 6, lc, hi - 6, hc) for lo, lc, hi, hc in rand_pieces(rng, lo=0, hi=12)])
    kind = rng.choice(['int', 'frac', 'float'])
    if kind == 'int': m = rng.choice([1, 2, 3, 5, -1, -2, -3])
    elif kind == 'frac': m = Fraction(rng.choice([1, 3, 5, 7, -1, -3, -7]), rng.choice([2, 3, 4]))
    else: m = rng.choice([0.5, 0.25, 1.5, -0.75, 2.5, 0.1, 0.3, -0.3])
    for reflected in (False, True):
        if reflected and any(lo <= 0 <= hi for lo, _, hi, _ in A):
            A2 = canon([(lo, lc, hi, hc) for lo, lc, hi, hc in A if lo > 0 or hi < 0])
            if not A2: continue
        else:
            A2 = A
        N += 1
        truth = floor_set_div(A2, m, reflected)
        f1 = (lambda: m // mk1(A2)) if reflected else (lambda: mk1(A2) // m)
        f2 = (lambda: m // mk2(A2)) if reflected else (lambda: mk2(A2) // m)
        s1, r1 = run(f1); s2, r2 = run(f2)
        lo, hi = min(truth) - 3, max(truth) + 3
        tag = ('m // A' if reflected else 'A // m') + f' [{kind}]'
        if s2 != 'ok': note(f'{tag}: v2 raised', (show(A2), m, r2)); continue
        p2 = canon(v2_pieces(r2))
        if SAB: p2 = canon(v2_pieces(mk2(A2) // (m * 2)))
        i2 = ints_in(p2, lo, hi)
        is_float = kind == 'float'
        # with a float, python's x // m differs from the exact floor at most where x/m is within an ulp of an integer;
        # report separately rather than hide it
        if i2 == truth and not any(a < b for a, _, b, _ in p2): note(f'{tag}: v2 == exact', None)
        else: note(f'{tag}: v2 != exact', (show(A2), m, show(p2), sorted(truth)))
        if s1 != 'ok': note(f'{tag}: v1 raised', (show(A2), m, r1)); continue
        p1 = canon(v1_pieces(r1)); i1 = ints_in(p1, lo, hi)
        if i1 == truth: note(f'{tag}: v1 ints == exact', None)
        elif i1 > truth: note(f'{tag}: v1 extra ints', (show(A2), m, show(p1), sorted(truth)))
        else: note(f'{tag}: v1 MISSES ints', (show(A2), m, show(p1), sorted(truth)))
print('cases', N)
for k in sorted(stats): print(f'  {k}: {stats[k]}', '' if ex_shown[k] is None else ex_shown[k])
