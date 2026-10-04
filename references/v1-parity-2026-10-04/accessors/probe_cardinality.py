"""cardinality (v1) vs size (v2): every component, rays included"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *
from collections import Counter


def oracle(b):
    """independent: count from the pieces. rays from the origin; length the finite remainder; points = (closed-open)/2"""
    rays, length, hp = 0, Fraction(0), 0
    for p in b.pieces:
        lo, hi = p.inf, p.sup
        hp += (1 if p.inf_closed else -1) + (1 if p.sup_closed else -1)
        if lo == -INF and hi == INF:
            rays += 2
        elif lo == -INF:
            rays += 1; length += Fraction(hi)          # (-inf, hi) = (-inf, 0) + [0, hi)
        elif hi == INF:
            rays += 1; length -= Fraction(lo)          # (lo, inf) = [0, inf) - [0, lo)
        else:
            length += Fraction(hi) - Fraction(lo)
    return rays, length, Fraction(hp, 2)


hand = [[], [(0, 0, True, True)], [(1, 2, True, False)], [(1, 2, True, True)], [(1, 2, False, False)],
        [(1, INF, False, False)], [(-INF, 1, False, True)], [(-INF, INF, False, False)],
        [(-INF, -10, False, False), (-3, -2, True, True)], [(-INF, -1, False, False), (1, INF, False, False)],
        [(2, 3, True, True), (5, INF, True, False)], [(-1, 1, True, True)], [(0.5, 2.5, True, False)]]
rng = random.Random(1788)
cases = hand + [rand_pieces(rng) for _ in range(600)]
cmp = Counter(); ex = {}
n = 0
for pcs in cases:
    try:
        a, b = build(pcs)
    except Exception:
        continue
    n += 1
    s = b.size
    o = oracle(b)
    exact = not isinstance(s.length, float)
    len_ok = (Fraction(s.length) == o[1]) if exact else abs(s.length - float(o[1])) <= 1e-12 * max(1, abs(float(o[1])))
    check('v2 vs oracle', s.rays == o[0] and s.points == o[2] and len_ok, (str(b), s, o))
    if not exact: cmp['v2 length is float (a float end in the set)'] += 1
    try:
        c = a.cardinality
    except Exception as e:
        cmp['v1 raises ' + type(e).__name__] += 1; ex.setdefault('v1 raises ' + type(e).__name__, (str(b), repr(e))); continue
    has_ray = s.rays > 0
    k = ('ray' if has_ray else 'bounded')
    r_ok = c[0] == s.rays
    l_ok = c[1] == s.length
    if not l_ok and not has_ray and isinstance(c[1], float) and abs(c[1] - float(o[1])) <= 1e-12 * max(1, abs(float(o[1]))):
        cmp['bounded length: float summation order only (both within 1e-12 of exact)'] += 1; l_ok = True
    p_ok = c[2] == 2 * s.points
    for name, ok in [('rays', r_ok), ('length', l_ok), ('points(v1 = 2*v2)', p_ok)]:
        key = f'{k} {name} {"agree" if ok else "DIFFER"}'
        cmp[key] += 1
        if not ok: ex.setdefault(key, (str(b), 'v1', c, 'v2', s, 'oracle', o))
# docstring example of v1 and v2
print('v1 docstring says (1, inf] -> (1, -1, 0); v2 doc says Size(1, -1, 0)')
print('v2 (1, inf):', v2.MultiInterval.parse('(1, inf)').size, ' v2 (1, inf]:', v2.MultiInterval.parse('(1, inf]').size)
print('v1 (1, inf):', v1.MultiInterval(1, INF, start_closed=False, end_closed=False).cardinality)
# invariants (v1 docstring) on v2: move, extract point, split, join, mirror
P = v2.MultiInterval.parse
for x, y in [('[1, 2)', '[2, 3)'), ('[1, 2)', '{ [1] , (1, 2) }'), ('(1, inf]', '{ (1, 2) , [2, inf] }'), ('[1, 2)', '(-2, -1]'), ('(-inf, 0)', '(0, inf)')]:
    check('invariant ' + x + ' ~ ' + y, P(x).size == P(y).size, (P(x).size, P(y).size))
# sabotage
before = len(FAILS); check('SABOTAGE', P('[1, 2]').size == P('[1, 2)').size); assert len(FAILS) == before + 1; FAILS.pop()
print(n, 'sets')
for k_, v in sorted(cmp.items()):
    print(f'{k_}: {v}', ex.get(k_, ''))
report('cardinality', n)
