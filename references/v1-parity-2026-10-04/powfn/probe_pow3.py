"""3-argument pow(A, n, m): v1 enumerates integer points; v2 drops it (D11). check the workaround."""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import random
from common import *

def points1(ps):
    out = V1()
    for p in ps:
        out = out.union(V1(p))
    return out

def points2(ps):
    return V2.from_pieces((p, p) for p in ps)

def workaround(A, E, M):
    """v2 composition: enumerate the degenerate pieces (an integral set is all points)"""
    pts = lambda S: [p.inf for p in S if p.inf == p.sup and float(p.inf).is_integer()]
    return V2.from_pieces((pow(int(b), int(e), int(m)),) * 2 for b in pts(A) for e in pts(E) for m in pts(M))

rng = random.Random(3)
agree = differ = 0
for _ in range(300):
    bs = rng.sample(range(-20, 40), rng.randint(1, 4))
    es = rng.sample(range(0, 12), rng.randint(1, 3))
    ms = rng.sample([m for m in range(-9, 15) if m != 0], rng.randint(1, 3))
    r1, x1 = run(lambda: pow(points1(bs), points1(es), points1(ms)))
    w = workaround(points2(bs), points2(es), points2(ms))
    truth = sorted({pow(b, e, m) for b in bs for e in es for m in ms})
    if x1:
        print('v1 raised', bs, es, ms, x1); differ += 1; continue
    ok1 = pieces1(r1) == [(t, t, True, True) for t in truth]
    okw = pieces2(w) == [(t, t, True, True) for t in truth]
    if ok1 and okw:
        agree += 1
    else:
        differ += 1
        print('DIFF', bs, es, ms, s1(r1), w)
print('agree', agree, 'differ', differ)

for label, f1, f2 in [
    ('pow(A, 2, 5) A={2,3}', lambda: pow(points1([2, 3]), 2, 5), lambda: pow(points2([2, 3]), 2, 5)),
    ('pow([2,3], 2, 5) non-degenerate', lambda: pow(V1(2, 3), 2, 5), lambda: pow(V2(2, 3), 2, 5)),
    ('pow({3}, -1, 7) negative exp', lambda: pow(V1(3), -1, 7), lambda: pow(V2(3), -1, 7)),
    ('pow({3.0}, 2, 7) float points', lambda: pow(V1(3.0), 2, 7), lambda: pow(V2(3.0), 2, 7)),
    ('pow({2**60+1}, 1, 10**30) big int', lambda: pow(V1(2 ** 60 + 1), 1, 10 ** 30), lambda: pow(V2(2 ** 60 + 1), 1, 10 ** 30)),
    ('pow({3}, 2, inf)', lambda: pow(V1(3), 2, float('inf')), lambda: pow(V2(3), 2, float('inf'))),
]:
    r1, x1 = run(f1); r2, x2 = run(f2)
    print(f'{label:36s} v1: {x1 or s1(r1)!s:34s} v2: {x2 or r2}')
print('big int truth', pow(2 ** 60 + 1, 1, 10 ** 30), 'workaround', workaround(points2([2 ** 60 + 1]), points2([1]), points2([10 ** 30])))
# sabotage: a wrong truth is caught
assert pieces2(workaround(points2([2]), points2([2]), points2([5]))) != [(3, 3, True, True)]
print('sabotage caught')
