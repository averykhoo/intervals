"""A // B and A // m and m // A: v1 (floor the endpoints of truediv) vs v2 (floor . div, enumerating) vs exact oracle"""
from common import *
import sys
SAB = '--sabotage' in sys.argv
rng = random.Random(99)

def quot_box(x, y):
    """exact image of x/y over one box (y piece excludes 0): (lo, lc, hi, hc)"""
    xlo, xlc, xhi, xhc = x; ylo, ylc, yhi, yhc = y
    corners = [(xa / ya, xc and yc) for xa, xc in ((xlo, xlc), (xhi, xhc)) for ya, yc in ((ylo, ylc), (yhi, yhc))]
    lo = min(c[0] for c in corners); hi = max(c[0] for c in corners)
    return (lo, any(c[1] for c in corners if c[0] == lo), hi, any(c[1] for c in corners if c[0] == hi))

def floor_values(A, B):
    out = set()
    for x in A:
        for y in B:
            lo, lc, hi, hc = quot_box(x, y)
            for n in range(math.floor(lo) - 1, math.floor(hi) + 2):
                if inter(lo, lc, hi, hc, Fraction(n), True, Fraction(n + 1), False):
                    out.add(n)
    return out

def ints_in(pcs, lo, hi):
    return {n for n in range(lo, hi + 1) if contains(pcs, Fraction(n))}

def nonint_member(pcs):
    """does the set hold a non-integer? (any piece with lo < hi)"""
    return any(lo < hi for lo, _, hi, _ in pcs)

cases = []
hand = [
    ([(1, True, 2, False)], [(1, True, 1, True)]),
    ([(1, True, 2, True)], [(1, True, 1, True)]),
    ([(0, False, 1, False)], [(1, True, 1, True)]),
    ([(-5, True, 5, True)], [(2, True, 3, True)]),
    ([(-3, True, -1, True)], [(-2, True, -1, False)]),
    ([(7, True, 7, True)], [(2, True, 2, True)]),
    ([(-7, True, -7, True)], [(2, True, 2, True)]),
    ([(1, True, 3, True)], [(-4, True, -1, True)]),
]
for a, b in hand: cases.append((canon(a), canon(b)))
for _ in range(400):
    a = canon([(lo - 6, lc, hi - 6, hc) for lo, lc, hi, hc in rand_pieces(rng, lo=0, hi=12)])
    s = rng.choice((1, -1))
    b = canon([(lo + Fraction(1, 2), lc, hi + Fraction(1, 2), hc) for lo, lc, hi, hc in rand_pieces(rng, lo=0, hi=5)])
    if s < 0: b = canon([(-hi, hc, -lo, lc) for lo, lc, hi, hc in b])
    cases.append((a, b))
n = 0; stats = {'v2 == oracle': 0, 'v2 != oracle': 0, 'v1 ints == oracle': 0, 'v1 ints superset': 0, 'v1 ints miss some': 0,
                'v1 holds non-integers': 0, 'v1 raised': 0}
examples = []
for a, b in cases:
    n += 1
    truth = floor_values(a, b)
    s1, r1 = run(lambda: mk1(a) // mk1(b))
    s2, r2 = run(lambda: mk2(a) // mk2(b))
    p2 = canon(v2_pieces(r2)) if s2 == 'ok' else None
    if SAB and p2 is not None: p2 = canon([(lo + 1, lc, hi + 1, hc) for lo, lc, hi, hc in p2])
    lo, hi = min(truth) - 3, max(truth) + 3
    if p2 is not None and ints_in(p2, lo, hi) == truth and not nonint_member(p2):
        stats['v2 == oracle'] += 1
    else:
        stats['v2 != oracle'] += 1; examples.append(('v2', show(a), show(b), r2 if p2 is None else show(p2), sorted(truth)))
    if s1 != 'ok':
        stats['v1 raised'] += 1; examples.append(('v1 raised', show(a), show(b), r1)); continue
    p1 = canon(v1_pieces(r1))
    i1 = ints_in(p1, lo, hi)
    if i1 == truth: stats['v1 ints == oracle'] += 1
    elif i1 > truth: stats['v1 ints superset'] += 1; examples.append(('v1 extra ints', show(a), show(b), show(p1), sorted(truth)))
    else: stats['v1 ints miss some'] += 1; examples.append(('v1 MISSES', show(a), show(b), show(p1), sorted(truth)))
    if nonint_member(p1): stats['v1 holds non-integers'] += 1
print(f'cases {n}'); [print(' ', k, v) for k, v in stats.items()]
seen = set()
for e in examples:
    if e[0] in seen and e[0] != 'v1 MISSES': continue
    seen.add(e[0]); print(e)
