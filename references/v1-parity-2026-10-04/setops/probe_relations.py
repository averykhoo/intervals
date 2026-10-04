"""isdisjoint, issubset, issuperset, __contains__, overlaps, overlapping(or_adjacent), _intervals_intersect"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random

def brute_subset(a2, b2):
    return all((p in b2) for p in points_for(a2, b2) if p in a2)

def brute_overlap(a2, b2):
    return any((p in a2) and (p in b2) for p in points_for(a2, b2))

rng = random.Random(20261004)
T = {k: Tally(k) for k in ['isdisjoint', 'issubset', 'issuperset', 'contains_MI', 'contains_num',
                           'overlaps', 'overlaps_adj', 'overlapping', 'overlapping_adj', 'overlapping_adj_naive',
                           'issubset_num', 'issuperset_num', 'isdisjoint_num', 'overlapping_num', 'overlapping_num_adj',
                           '_intervals_intersect']}
v1_wrong = {k: [] for k in T}

def adj_or_overlap(p, B):
    return p.overlaps(B) or any(p.adjoins(q) for q in B)

def v2_overlapping(A, B, adj=False):
    keep = [p for p in A if (adj_or_overlap(p, B) if adj else p.overlaps(B))]
    return M2().union(*keep)

def v2_overlapping_naive(A, B):
    return M2().union(*[p for p in A if p.overlaps(B) or p.adjoins(B)])

def brute_overlapping(A, B, adj):
    # pieces of A whose union with some piece of B is contiguous (adj) / which share a point (not adj)
    out = []
    for p in A:
        ok = False
        for q in B:
            if adj:
                ok = ok or len(p | q) == 1
            else:
                ok = ok or brute_overlap(p, q)
        if ok:
            out.append(p)
    return M2().union(*out)

N = 600
for i in range(N):
    pa, pb = rand_pieces(rng), rand_pieces(rng)
    if rng.random() < 0.2:
        pb = pa[:rng.randint(0, len(pa))]  # subsets more often
    a1, a2 = twin(pa); b1, b2 = twin(pb)
    lab = (pa, pb)
    # oracles from brute membership on v2 sets (same sets as v1, checked by common)
    sub = brute_subset(a2, b2); sup = brute_subset(b2, a2); ov = brute_overlap(a2, b2)
    for name, f1, f2, truth in [
        ('isdisjoint', lambda: a1.isdisjoint(b1), lambda: a2.isdisjoint(b2), not ov),
        ('issubset', lambda: a1.issubset(b1), lambda: a2.issubset(b2), sub),
        ('issuperset', lambda: a1.issuperset(b1), lambda: a2.issuperset(b2), sup),
        ('contains_MI', lambda: b1 in a1, lambda: b2 in a2, sup),
        ('overlaps', lambda: a1.overlaps(b1), lambda: a2.overlaps(b2), ov),
    ]:
        try: r1 = f1()
        except Exception as e: r1 = f'raise {type(e).__name__}'
        r2 = f2()
        T[name].check(lab, None if r1 == r2 else f'v1 {r1} v2 {r2} truth {truth}')
        if r1 != truth: v1_wrong[name].append((lab, r1, truth))
        assert r2 == truth, (name, lab, r2, truth)
    # overlaps(or_adjacent=True): v2 spelling A.overlaps(B) or any piece adjoins
    r1 = a1.overlaps(b1, or_adjacent=True)
    r2 = any(adj_or_overlap(p, b2) for p in a2)
    truth = any(len(p | q) == 1 for p in a2 for q in b2)
    T['overlaps_adj'].check(lab, None if r1 == r2 else f'v1 {r1} v2 {r2} truth {truth}')
    if r1 != truth: v1_wrong['overlaps_adj'].append((lab, r1, truth))
    # overlapping
    for adj, key in [(False, 'overlapping'), (True, 'overlapping_adj')]:
        r1 = a1.overlapping(b1, or_adjacent=adj)
        r2 = v2_overlapping(a2, b2, adj)
        tr = brute_overlapping(a2, b2, adj)
        T[key].check(lab, same_set_v1_v2(r1, r2))
        if from_v1(r1) != tr: v1_wrong[key].append((lab, str(r1), str(tr)))
        assert r2 == tr, (key, lab)
    r2n = v2_overlapping_naive(a2, b2)
    T['overlapping_adj_naive'].check(lab, None if r2n == brute_overlapping(a2, b2, True) else f'naive {r2n} truth {brute_overlapping(a2, b2, True)}')
    # numbers on the right
    x = rng.choice(GRID + [-inf, inf])
    for name, f1, f2 in [
        ('issubset_num', lambda: a1.issubset(x), lambda: a2.issubset(x)),
        ('issuperset_num', lambda: a1.issuperset(x), lambda: a2.issuperset(x)),
        ('isdisjoint_num', lambda: a1.isdisjoint(x), lambda: a2.isdisjoint(x)),
        ('contains_num', lambda: x in a1, lambda: x in a2),
    ]:
        try: r1 = f1()
        except Exception as e: r1 = f'raise {type(e).__name__}: {e}'
        try: r2 = f2()
        except Exception as e: r2 = f'raise {type(e).__name__}: {e}'
        T[name].check((pa, x), None if r1 == r2 else f'v1 {r1} v2 {r2}')
    for adj, key in [(False, 'overlapping_num'), (True, 'overlapping_num_adj')]:
        try: r1 = a1.overlapping(x, or_adjacent=adj)
        except Exception as e: r1 = f'raise {type(e).__name__}'
        r2 = v2_overlapping(a2, M2(x), adj)
        T[key].check((pa, x), same_set_v1_v2(r1, r2) if not isinstance(r1, str) else f'v1 {r1} v2 {r2}')
    # _intervals_intersect on two single pieces
    p, q = rand_piece(rng), rand_piece(rng)
    r1 = v1._intervals_intersect(p[0], p[2], p[1], p[3], q[0], q[2], q[1], q[3])
    r2 = M2.from_pieces([p]).overlaps(M2.from_pieces([q]))
    T['_intervals_intersect'].check((p, q), None if r1 == r2 else f'v1 {r1} v2 {r2}')

bad = 0
for t in T.values():
    bad += t.report()
print('v1 disagreeing with brute truth:', {k: len(v) for k, v in v1_wrong.items() if v})
for k, v in v1_wrong.items():
    for e in v[:3]: print('  v1wrong', k, e)

# sabotage: a wrong expectation is caught
s = Tally('sabotage')
a1, a2 = twin([(0, 1, True, False)])
s.check('x', None if (1 in a1) == (not (1 in a2)) else 'caught')
assert s.report(0) == 1, 'sabotage not caught'
