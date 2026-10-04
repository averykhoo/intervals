"""finite positive negative infimum(_is_closed) supremum(_is_closed) degenerate_points closed_hull contiguous_intervals"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *
from collections import Counter

REALS = v2.MultiInterval(-INF, INF, start_closed=False, end_closed=False)
diffs = Counter()
examples = {}


def same_set(a1, b2, label, pcs):
    """compare v1 set a1 with v2 set b2 by membership at every relevant point"""
    if a1 is None or b2 is None:
        ok = a1 is None and b2 is None
        if not ok:
            diffs[label] += 1; examples.setdefault(label, (pcs, a1 if a1 is None else a1.endpoints, str(b2)))
        return ok
    for x in probe_points(b2, v2.MultiInterval.from_pieces([(e[0], e[0]) for e in a1.endpoints])):
        if v1_contains(a1, x) != (x in b2):
            diffs[label] += 1; examples.setdefault(label, (pcs, a1.endpoints, str(b2), x))
            return False
    return True


hand = [[], [(0, 0, True, True)], [(-INF, INF, False, False)], [(-INF, 0, False, True)], [(0, INF, True, False)],
        [(-INF, -1, False, False), (1, 2, True, True), (3, INF, False, False)], [(-1, 1, True, True)],
        [(-1, 0, False, False), (0, 0, True, True), (Fraction(1, 3), 0.5, True, False)], [(5, 5, True, True), (7, 7, True, True)],
        [(-2, -1, False, True), (1, 1, True, True), (2, 3, False, False)]]
rng = random.Random(1788)
cases = hand + [rand_pieces(rng) for _ in range(600)]
n = 0
for pcs in cases:
    try:
        a, b = build(pcs)
    except Exception as e:
        continue
    n += 1
    # finite / positive / negative: as sets
    same_set(a.finite, b.finite, 'finite', pcs)
    same_set(a.positive, b.positive, 'positive', pcs)
    same_set(a.negative, b.negative, 'negative', pcs)
    # inf/sup and flags
    if a.is_empty:
        for n1, n2 in [('infimum', 'inf'), ('infimum_is_closed', 'inf_closed'), ('supremum', 'sup'), ('supremum_is_closed', 'sup_closed')]:
            try: getattr(a, n1); r1 = 'no raise'
            except Exception as e: r1 = type(e).__name__
            try: getattr(b, n2); r2 = 'no raise'
            except Exception as e: r2 = type(e).__name__
            diffs[f'{n1} on empty: v1 {r1} v2 {r2}'] += 1
    else:
        for n1, n2 in [('infimum', 'inf'), ('infimum_is_closed', 'inf_closed'), ('supremum', 'sup'), ('supremum_is_closed', 'sup_closed')]:
            r1, r2 = getattr(a, n1), getattr(b, n2)
            if not (r1 == r2 and type(r1) is type(r2) if not isinstance(r1, bool) else r1 == r2):
                diffs[n1] += 1; examples.setdefault(n1, (pcs, r1, r2, type(r1), type(r2)))
        # oracle: inf is in the set iff inf_closed (brute membership)
        check('inf_closed oracle', (b.inf in b) == b.inf_closed or abs(b.inf) == INF, pcs)
        check('sup_closed oracle', (b.sup in b) == b.sup_closed or abs(b.sup) == INF, pcs)
    # degenerate points
    d1, d2 = a.degenerate_points, b.degenerate_points
    if d1 != d2:
        diffs['degenerate_points'] += 1; examples.setdefault('degenerate_points', (pcs, d1, d2))
    # closed hull: v1 is open at ±inf, v2 closed. v1's meaning = v2 closed_hull & (-inf, inf)
    ch1 = a.closed_hull
    ch2 = b.closed_hull
    if a.is_empty:
        diffs[f'closed_hull of empty: v1 {ch1!r} v2 {ch2}'] += 1
    else:
        if not same_set(ch1, ch2, 'closed_hull_raw', pcs):
            pass
        same_set(ch1, ch2 & REALS, 'closed_hull_via_&_reals', pcs)
    # contiguous_intervals vs pieces
    c1, c2 = a.contiguous_intervals, b.pieces
    if len(c1) != len(c2):
        diffs['contiguous_intervals len'] += 1
    else:
        for p1, p2 in zip(c1, c2):
            same_set(p1, p2, 'contiguous_intervals', pcs)
    # list(b) and pieces agree
    check('iter==pieces', tuple(b) == c2)

# sabotage: positive vs negative must be caught as different
a, b = build([(-1, 1, True, True)])
before = sum(diffs.values())
same_set(a.positive, b.negative, 'SABOTAGE', None)
assert sum(diffs.values()) == before + 1, 'sabotage missed'
del diffs['SABOTAGE']; examples.pop('SABOTAGE')
print(n, 'sets')
for k, v in sorted(diffs.items()):
    print(f'DIFF {k}: {v}', examples.get(k, ''))
report('derived', n)
