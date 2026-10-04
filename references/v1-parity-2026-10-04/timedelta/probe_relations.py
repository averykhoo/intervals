from common import *

def outcome(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__)

def adj_or_overlap_oracle(p, q):
    """two canonical pieces overlap or are adjacent: the union of the two is one piece (grid-free, exact)"""
    lo1, lc1, hi1, hc1 = p; lo2, lc2, hi2, hc2 = q
    # disjoint with a gap?  p entirely before q
    def before(x, y):
        lo1, lc1, hi1, hc1 = x; lo2, lc2, hi2, hc2 = y
        return hi1 < lo2 or (hi1 == lo2 and not hc1 and not lc2)
    return not (before(p, q) or before(q, p))

rng = random.Random(2)
for i in range(500):
    sa, sb = rand_pieces(rng), rand_pieces(rng)
    a1, b1, a2, b2 = build1(sa), build1(sb), build2(sa), build2(sb)
    A, B = oracle_set(canon2(a2)), oracle_set(canon2(b2))
    # structural comparisons: v1 lexicographic on endpoints vs v2 sort_key
    for op in ('lt', 'le', 'gt', 'ge'):
        import operator as O
        check(('cmp-sortkey', op, sa, sb), getattr(O, op)(a1, b1), getattr(O, op)(a2.sort_key, b2.sort_key))
    check(('eq', sa, sb), a1 == b1, a2 == b2)
    check(('ne', sa, sb), a1 != b1, a2 != b2)
    check(('eq oracle', sa, sb), a2 == b2, A == B)
    # set relations
    check(('isdisjoint', sa, sb), a1.isdisjoint(b1), a2.isdisjoint(b2))
    check(('isdisjoint oracle', sa, sb), a2.isdisjoint(b2), not (A & B))
    check(('issubset', sa, sb), a1.issubset(b1), a2.issubset(b2))
    check(('issubset oracle', sa, sb), a2.issubset(b2), A <= B)
    check(('issuperset', sa, sb), a1.issuperset(b1), a2.issuperset(b2))
    check(('issuperset oracle', sa, sb), a2.issuperset(b2), A >= B)
    check(('contains iv', sa, sb), b1 in a1, b2 in a2)
    # overlaps / overlapping
    check(('overlaps', sa, sb), a1.overlaps(b1), a2.overlaps(b2))
    check(('overlaps oracle', sa, sb), a2.overlaps(b2), bool(A & B))
    ov1 = a1.overlapping(b1)
    ov2 = T2().union(*[p for p in a2 if p.overlaps(b2)])
    check(('overlapping', sa, sb), canon1(ov1), canon2(ov2))
    exp_adj = any(adj_or_overlap_oracle(p, q) for p in canon2(a2) for q in canon2(b2))
    check(('overlaps adj', sa, sb), a1.overlaps(b1, or_adjacent=True), exp_adj)
    check(('overlaps adj v2 composition', sa, sb), a2.overlaps(b2) or any(p.adjoins(q) for p in a2 for q in b2), exp_adj)
    check(('overlaps adj v2 len composition', sa, sb), len(a2 | b2) < len(a2) + len(b2), exp_adj)
    ova1 = a1.overlapping(b1, or_adjacent=True)
    ova2 = T2().union(*[p for p in a2 if any(adj_or_overlap_oracle(canon2(p)[0], q) for q in canon2(b2))])
    ova2c = T2().union(*[p for p in a2 if len(p | b2) <= len(b2)])
    check(('overlapping adj', sa, sb), canon1(ova1), canon2(ova2))
    check(('overlapping adj v2 composition', sa, sb), canon2(ova2c), canon2(ova2))
    # scalar operand: membership of points (grid, as timedelta and pd.Timedelta)
    for x in rng.sample(GRID, 6):
        t = td(x)
        m = member(canon2(a2), x)
        check(('in td', sa, x), t in a1, t in a2)
        check(('in td oracle', sa, x), t in a2, m)
        check(('in pd.Timedelta', sa, x), pd.Timedelta(t) in a1, pd.Timedelta(t) in a2)
        check(('issuperset td', sa, x), a1.issuperset(t), a2.issuperset(t))
        check(('isdisjoint td', sa, x), a1.isdisjoint(t), a2.isdisjoint(t))
        check(('overlaps td', sa, x), a1.overlaps(t), a2.overlaps(t))
        check(('issubset td', sa, x), a1.issubset(t), a2.issubset(t))
        check(('eq td', sa, x), a1 == t, a2 == T2(t))
        check(('lt td sortkey', sa, x), a1 < t, a2.sort_key < T2(t).sort_key)
from collections import Counter
print(Counter(f[0][0] for f in FAILS))
seen = set()
for f in FAILS:
    if f[0][0] not in seen:
        seen.add(f[0][0]); print('  first', f)
# fixed cases
E1, E2 = T1(), T2()
H = dt.timedelta(hours=1)
print('empty issuperset empty: v1', E1.issuperset(T1()), 'v2', E2.issuperset(T2()))
print('empty in empty: v1', T1() in T1(), 'v2', T2() in T2())
print('empty in [1h]: v1', T1() in T1(H), 'v2', T2() in T2(H))
print('scalar ==: v1', T1(H) == H, 'v2', T2(H) == H, '| v1 == 5', outcome(lambda: T1(H) == 5), 'v2 == 5', outcome(lambda: T2(H) == 5))
print('v2 < is TruthSet:', repr(T2(H, 2*H) < T2(3*H)), outcome(lambda: bool(T2(H, 3*H) < T2(2*H))))
print('v1 < bool:', T1(H, 2*H) < T1(3*H), T1(H, 3*H) < T1(2*H))
print('foreign operand: v1 issubset(5)', outcome(lambda: T1(H).issubset(5)), 'v2', outcome(lambda: T2(H).issubset(5)))
print('5 in: v1', outcome(lambda: 5 in T1(H)), 'v2', outcome(lambda: 5 in T2(H)))
print('NaT in: v1', outcome(lambda: pd.NaT in T1(H)), 'v2', outcome(lambda: pd.NaT in T2(H)))
print('v2 hash', outcome(lambda: hash(T2(H))), 'v1 hash', outcome(lambda: hash(T1(H))))
assert not check('sabotage', T1(H) in T1(H), False)
FAILS.pop()
report('probe_relations')
others = [f for f in FAILS if not (f[0][1] == [] and f[0][2] == [])]
print('mismatches not of the empty-empty pair:', len(others), others[:3])
