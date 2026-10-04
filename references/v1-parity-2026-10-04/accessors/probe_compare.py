"""v1 __lt__ __le__ __eq__ __ne__ __ge__ __gt__ (endpoint-list order) vs v2 sort_key; scalars; non-numbers; hash"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *
import operator
from collections import Counter

OPS = [operator.lt, operator.le, operator.eq, operator.ne, operator.ge, operator.gt]
rng = random.Random(1788)
sets = [build(p) for p in ([[], [(0, 0, True, True)], [(0, 1, True, True)], [(0, 1, False, True)], [(0, 1, True, False)],
                              [(0, 1, False, False)], [(0, 0, True, True), (1, 1, True, True)], [(-INF, INF, False, False)],
                              [(-INF, 0, False, True)], [(0, INF, True, False)], [(1, 1, True, True)], [(1.0, 1.0, True, True)],
                              [(Fraction(1, 3), 0.5, True, False)]])]
while len(sets) < 260:
    try:
        sets.append(build(rand_pieces(rng)))
    except Exception:
        pass
stats = Counter(); ex = {}
n = 0
for a1, a2 in sets:
    for b1, b2 in sets:
        for op in OPS:
            n += 1
            r1 = op(a1, b1)
            r2 = op(a2.sort_key, b2.sort_key)
            if r1 != r2:
                stats[f'set-set {op.__name__} DIFFER'] += 1; ex.setdefault(op.__name__, (a1.endpoints, b1.endpoints, r1, r2))
            else:
                stats[f'set-set {op.__name__} {r1}'] += 1
# scalars: v1 coerces a Real to a degenerate point; v2 spelling: A.sort_key OP MultiInterval(x).sort_key
scalars = [-INF + 0 if False else -3, -1, 0, Fraction(1, 3), 0.5, 1, 1.0, 2.5, 3]
for a1, a2 in sets:
    for x in scalars:
        for op in OPS:
            n += 1
            r1 = op(a1, x)
            r2 = op(a2.sort_key, v2.MultiInterval(x).sort_key)
            if r1 != r2:
                stats[f'set-scalar {op.__name__} DIFFER'] += 1; ex.setdefault('s' + op.__name__, (a1.endpoints, x, r1, r2))
            else:
                stats[f'set-scalar {op.__name__} {r1}'] += 1
# sorted order of a list
L1 = [s[0] for s in sets]; L2 = [s[1] for s in sets]
idx = list(range(len(sets)))
o1 = sorted(idx, key=lambda i: L1[i].endpoints)  # v1 has no key; its < is the same list compare
o1b = sorted(idx, key=lambda i: L1[i]) if True else None
o2 = sorted(idx, key=lambda i: L2[i].sort_key)
print('sorted(v1 objects) == sorted(v2 by sort_key):', [L1[i].endpoints for i in o1b] == [L1[i].endpoints for i in o2])
# direct behaviours
A1, A2 = build([(3, 3, True, True)])
for expr in ['A1 == 3', 'A2 == 3', 'A2 == v2.MultiInterval(3)', 'A1 < 4', 'A2 < 4', "A1 == 'x'", "A2 == 'x'", 'A1 == None', 'A2 == None',
             'hash(A1)', 'hash(A2) == hash(v2.MultiInterval(3))', 'v2.MultiInterval(1) == v2.MultiInterval(1.0)', 'v1.MultiInterval(1) == v1.MultiInterval(1.0)',
             'hash(v2.MultiInterval(1)) == hash(v2.MultiInterval(1.0))', 'A2.sort_key < 4']:
    try:
        r = eval(expr)
    except Exception as e:
        r = f'RAISES {type(e).__name__}: {e}'
    print(f'  {expr} -> {r!r}')
# sabotage: compare v1 lt against v2 gt must be caught at least once
a1, a2 = sets[2]; b1, b2 = sets[10]
assert operator.lt(a1, b1) != operator.gt(a2.sort_key, b2.sort_key), 'sabotage not caught'
print(n, 'comparisons')
for k, v in sorted(stats.items()):
    print(f'{k}: {v}')
print('examples', ex)
