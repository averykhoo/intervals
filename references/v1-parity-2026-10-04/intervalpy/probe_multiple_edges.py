from common import *
from probe_arith_helpers import run
sys.path.insert(0, 'tests')
import oracles  # noqa: E402

# overlapping(or_adjacent=True) when a piece of self adjoins an INNER piece of other
a = MI(I(0, False, 1, True), I(5, False, 6, True))
b = MI(I(-5, False, -4, True), I(1, True, 2, True), I(10, False, 11, True))
A, B = mi_to_v2(a), mi_to_v2(b)
v1r = a.overlapping(b, or_adjacent=True)
naive = M().union(*[p for p in A if p.overlaps(B) or p.adjoins(B)])
per_piece = M().union(*[p for p in A if any(p.overlaps(q) or p.adjoins(q) for q in B)])
contig = M().union(*[p for p in A if any((p | q).is_contiguous for q in B)])
print('v1 overlapping(or_adjacent=True):', v1r)
print('v2 p.overlaps(B) or p.adjoins(B):', naive)
print('v2 per piece of B:', per_piece, '| via (p | q).is_contiguous:', contig)
check('or_adjacent per-piece composition', mi_to_v2(v1r) == per_piece)
check('or_adjacent naive composition (expected to differ)', mi_to_v2(v1r) == naive)
print('A.adjoins(B) for a multi-piece B:', A.adjoins(B), '| M(0,1).adjoins(M(1,2,start_closed=False)):', M(0, 1).adjoins(M(1, 2, start_closed=False)))

# reciprocal: the cases where v1 is neither equal nor a superset
rng = random.Random(21)
shown = 0
for _ in range(3000):
    ivs = [rand_interval(rng) for _ in range(rng.randint(1, 3))]
    mi = MI(*ivs)
    Am = mi_to_v2(mi)
    r1 = run(lambda: mi.reciprocal())
    R = Am.reciprocal()
    if r1[0] != 'ok':
        if shown < 8:
            print('v1 reciprocal raised', mi, r1, '| v2', R)
            shown += 1
        continue
    pts = probe_points(*ends_of(R), *ends_of(r1[1]))
    bad = [q for q in pts if (q in r1[1]) != (q in R)]
    if bad and not all(q in r1[1] for q in bad):
        missing = [q for q in bad if q in R and q not in r1[1]]
        # is v2 right at those points? exact oracle
        truth = [oracles.attained('reciprocal', q, Am.cuts) for q in missing]
        if shown < 8:
            print('v1 misses', [str(q) for q in missing[:3]], 'of', mi, '-> v1', r1[1], '| v2', R, '| oracle attained:', truth[:3])
            shown += 1
        check('v2 reciprocal right where v1 misses', all(truth))
# length
print('length of {[0,1] ; [2,4]}: v1', MI(I(0, False, 1, True), I(2, False, 4, True)).length, '| v2 size.length', (M(0, 1) | M(2, 4)).size.length)
print('length with a ray: v1', MI(I(-INF, True, 0, True)).length, '| v2', M(-INF, 0, start_closed=False).size)
# from_multi_interval
def v1m_of(pieces):
    out = v1m.MultiInterval()
    for lo, hi, lc, hc in pieces:
        out = out.union(v1m.MultiInterval(lo, hi, start_closed=lc, end_closed=hc))
    return out
for pieces in [[(0, 1, True, True), (2, 3, False, False)], [(0, 0, True, True)], [], [(-INF, 0, False, True)], [(1, INF, True, False)], [(1, INF, True, True)]]:
    r = run(lambda: v1m_of(pieces))
    if r[0] != 'ok':
        print('v1 MultiInterval build failed', pieces, r)
        continue
    vm = r[1]
    f1 = run(lambda: MI.from_multi_interval(vm))
    v2_spell = M.from_pieces([(p.infimum, p.supremum, p.infimum_is_closed, p.supremum_is_closed) for p in vm.contiguous_intervals])
    print('from_multi_interval', pieces, '->', f1, '| v2 from_pieces:', v2_spell, '| v2 parse(str(v1)):', run(lambda: M.parse(str(vm))))
    if f1[0] == 'ok':
        same_reals('from_multi_interval ' + str(pieces), f1[1], v2_spell)
# in-place ops alias semantics
x = MI(I(0, False, 1, True))
alias = x
x.update(I(2, False, 3, True))
print('v1 update mutates alias:', alias)
X = M(0, 1)
Y = X
X = X | M(2, 3)
print('v2 rebinding leaves alias:', Y)
# symmetric_difference_update with a single Interval / Real
print('v1 symdiff_update(Interval):', run(lambda: MI(I(0, False, 2, True)).symmetric_difference_update(I(1, False, 3, True))), '| v2', M(0, 2) ^ M(1, 3))
print('v1 symdiff_update(Real):', run(lambda: MI(I(0, False, 2, True)).symmetric_difference_update(1)), '| v2', M(0, 2) ^ 1)
print('v1 union(b, c):', run(lambda: MI(I(0, False, 1, True)).union(MI(I(2, False, 3, True)), MI(I(4, False, 5, True)))), '| v2', M(0, 1).union(M(2, 3), M(4, 5)))
print('v1 update(Real):', MI(I(0, False, 1, True)).update(1), MI(I(0, True, 1, False)).update(1), '| v2', M(0, 1, start_closed=False, end_closed=False) | 1)
print('v1 empty MultipleInterval:', MI(), '| repr', repr(MI()), '| v2', M(), repr(M()))
report_end(__file__)
