from common import *
import sys as _sys

def outcome(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__)

rng = random.Random(1)
n = 0
for i in range(400):
    pieces = rand_pieces(rng)
    a, b = build1(pieces), build2(pieces)
    n += 1
    check(('struct', pieces), canon1(a), canon2(b))
    check(('is_empty', pieces), a.is_empty, b.is_empty)
    check(('is_contiguous', pieces), a.is_contiguous, b.is_contiguous)
    check(('is_degenerate', pieces), a.is_degenerate, b.is_degenerate)
    if a.is_empty:
        check(('inf empty', pieces), (a.infimum, a.supremum, a.closed_hull), (None, None, None))
        check(('v2 inf empty', pieces), outcome(lambda: b.inf), ('raise', 'ValueError'))
        check(('v2 sup empty', pieces), outcome(lambda: b.sup), ('raise', 'ValueError'))
        check(('v2 closed_hull empty', pieces), canon2(b.closed_hull), ())
    else:
        check(('infimum', pieces), a.infimum, b.inf)
        check(('supremum', pieces), a.supremum, b.sup)
        check(('closed_hull', pieces), canon1(a.closed_hull), canon2(b.closed_hull))
    check(('degenerate_points', pieces), a.degenerate_points, set(b.degenerate_points))
    check(('degenerate_points sorted', pieces), list(b.degenerate_points), sorted(b.degenerate_points))
    l1, p1 = a.cardinality
    s = b.size
    check(('cardinality', pieces), (round_us(l1), p1), (s.length, 2 * s.points))
    check(('cardinality rays', pieces), s.rays, 0)
    check(('total_seconds', pieces), round_us(l1), b.total_seconds)
    check(('contiguous_intervals', pieces), [canon1(x) for x in a.contiguous_intervals], [canon2(x) for x in b.pieces])
    check(('iter', pieces), [canon2(x) for x in b], [canon2(x) for x in b.pieces])
    check(('pieces type', pieces), all(type(x) is T2 for x in b.pieces), True)
    c = a.copy()
    check(('copy', pieces), canon1(c), canon1(a))
    c.clear()
    check(('copy independent', pieces), canon1(a), canon2(b))
    check(('str', pieces), str(a), str(b), ok=(str(a).replace(', ', ' ') == str(b).replace(', ', ' ').replace(' ,', ',')) or None)
print('degenerate_points types: v1', type(T1(dt.timedelta(1)).degenerate_points).__name__, 'v2', type(T2(dt.timedelta(1)).degenerate_points).__name__)
print('v1 sizeof', T1(dt.timedelta(1)).__sizeof__(), 'v2 sizeof (object default)', T2(dt.timedelta(1)).__sizeof__(), 'getsizeof', _sys.getsizeof(T2(dt.timedelta(1))))
print('v1 repr', repr(T1(dt.timedelta(1)))[:60], '| v2 repr', repr(T2(dt.timedelta(1))))
# sabotage
assert not check('sabotage', a.cardinality, (1, 2))
FAILS.pop()
report('probe_props')
from collections import Counter
print(Counter(f[0][0] if isinstance(f[0], tuple) else f[0] for f in FAILS))
neg_free = [f for f in FAILS if f[0][0] == 'str' and '-' not in f[1]]
print('str mismatches with no negative value:', len(neg_free), neg_free[:3])
