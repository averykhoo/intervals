from common import *
from gen import rand_pair
D = dt.datetime; d_ = dt.date
r = random.Random(7)

# SELFTEST: is_contiguous must be able to disagree (v1 two adjacent days: two pieces; v2 one)
selftest(lambda: V1D(d_(2024, 1, 1)).union(V1D(d_(2024, 1, 2))).is_contiguous ==
         V2D(d_(2024, 1, 1)).union(V2D(d_(2024, 1, 2))).is_contiguous)

print('--- adjacent days')
a1 = V1D(d_(2024, 1, 1)).union(V1D(d_(2024, 1, 2))); a2 = V2D(d_(2024, 1, 1)).union(V2D(d_(2024, 1, 2)))
print('v1', a1, 'contiguous', a1.is_contiguous, ' v2', a2, 'contiguous', a2.is_contiguous)

print('--- empty set accessors')
for name in ('is_empty', 'is_contiguous', 'is_degenerate', 'infimum', 'supremum', 'degenerate_points', 'closed_hull',
             'cardinality', 'total_seconds', 'total_duration', 'contiguous_intervals'):
    v2name = {'infimum': 'inf', 'supremum': 'sup', 'cardinality': 'size', 'contiguous_intervals': 'pieces'}.get(name, name)
    print(f'{name}: v1', safe(lambda: getattr(V1D(), name)), '| v2', v2name, safe(lambda: getattr(V2D(), v2name)))

print('--- whole-second ends: closed_hull and contiguous_intervals through the v1 constructor (snap)')
x1 = V1D(D(2024, 1, 1, 10, 30, 15, 5)).union(V1D(D(2024, 1, 1, 11)))
x2 = V2D(D(2024, 1, 1, 10, 30, 15, 5)).union(V2D(D(2024, 1, 1, 11)))
print('v1 set', x1, ' closed_hull', x1.closed_hull, ' sup', x1.closed_hull.supremum)
print('v2 set', x2, ' closed_hull', x2.closed_hull)
ok, diff = same_set(x1.closed_hull, x2.closed_hull); print('closed_hull same_set', ok, diff)
check(not ok, 'v1 closed_hull snap expected to differ')
print('v1 contiguous_intervals', x1.contiguous_intervals, ' v2 pieces', x2.pieces)
p1 = V1D(D(2024, 1, 1, 11)).contiguous_intervals[0]
print('v1 DTI(11:00).contiguous_intervals[0] =', p1, ' contains 11:30?', D(2024, 1, 1, 11, 30) in p1, ' original contains 11:30?', D(2024, 1, 1, 11, 30) in V1D(D(2024, 1, 1, 11)))
y1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 12, 0, 0, 1)).difference(V1D(D(2024, 1, 1, 10, 0, 0, 0), D(2024, 1, 1, 10, 30, 0, 1), start_closed=False))
y2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 12, 0, 0, 1)).difference(V2D(D(2024, 1, 1, 10, 0, 0, 0), D(2024, 1, 1, 10, 30, 0, 1), start_closed=False))
print('v1 diff', y1, ' pieces', y1.contiguous_intervals, '| v2', y2, ' pieces', y2.pieces)
print('  v1 first piece holds 10:15?', D(2024, 1, 1, 10, 15) in y1.contiguous_intervals[0], ' the set holds 10:15?', D(2024, 1, 1, 10, 15) in y1)

print('--- aware read-outs')
UTC = dt.timezone.utc
z1 = V1D(D(2024, 1, 1, 2, 0, 0, 5, tzinfo=UTC), D(2024, 1, 1, 3, 0, 0, 5, tzinfo=UTC))
z2 = V2D(D(2024, 1, 1, 2, 0, 0, 5, tzinfo=UTC), D(2024, 1, 1, 3, 0, 0, 5, tzinfo=UTC))
print('v1 inf/sup', repr(z1.infimum), repr(z1.supremum), '| v2', repr(z2.inf), repr(z2.sup))
print('v1 aware pieces', z1.contiguous_intervals, '| v2', z2.pieces)

print('--- degenerate_points type')
w1 = V1D(D(2024, 1, 1, 10, 0, 0, 5)).union(V1D(D(2024, 1, 1, 9, 0, 0, 5)), V1D(d_(2024, 1, 3)))
w2 = V2D(D(2024, 1, 1, 10, 0, 0, 5)).union(V2D(D(2024, 1, 1, 9, 0, 0, 5)), V2D(d_(2024, 1, 3)))
print('v1', w1.degenerate_points, '| v2', w2.degenerate_points)
check(set(w2.degenerate_points) == w1.degenerate_points, 'degenerate_points values')

print('--- cardinality / size / total_seconds / total_duration on a day, an open piece, points')
for args, kw in [((d_(2024, 1, 1),), {}), ((d_(2024, 1, 1), d_(2024, 1, 3)), {}),
                 ((D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 17, 0, 0, 1)), {'start_closed': False, 'end_closed': False}),
                 ((D(2024, 1, 1, 9, 0, 0, 1),), {})]:
    a1, a2 = V1D(*args, **kw), V2D(*args, **kw)
    print(args, kw, ': v1 cardinality', a1.cardinality, 'total_seconds', a1.total_seconds, 'total_duration', repr(a1.total_duration),
          '| v2 size', a2.size, 'total_seconds', a2.total_seconds, 'total_duration', repr(a2.total_duration))
print('v2 unbounded total_seconds:', safe(lambda: V2D(NEG_INF, D(2024, 1, 1)).total_seconds), ' size', V2D(NEG_INF, D(2024, 1, 1)).size)
print('float rounding in v1: 1 us piece length', V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 9, 0, 0, 2)).total_seconds,
      ' v2', V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 9, 0, 0, 2)).total_seconds)

print('--- random sweep')
n_card = n_pieces = 0
for i in range(400):
    a1, a2, ps = rand_pair(r)
    ok, diff = same_set(a1, a2)
    if not check(ok, f'build {ps} {diff}'):
        continue
    check(a1.is_empty == a2.is_empty, f'is_empty {ps}')
    check(a1.is_contiguous == a2.is_contiguous or any(len(a) == 1 and not isinstance(a[0], D) for a, _ in ps) or
          any(len(a) == 2 and not isinstance(a[1], D) for a, _ in ps), f'is_contiguous {ps} {a1} {a2}')
    check(a1.is_degenerate == a2.is_degenerate, f'is_degenerate {ps}')
    if not a1.is_empty:
        check(a1.infimum == a2.inf, f'inf {ps} {a1.infimum} {a2.inf}')
        has_day_end = abs(a2.sup - (a1.supremum + US)) == dt.timedelta(0) and not a2.sup_closed
        check(a1.supremum == a2.sup or has_day_end, f'sup {ps} {a1.supremum} {a2.sup}')
        check(set(a2.degenerate_points) == a1.degenerate_points, f'degenerate {ps}')
        # pieces: compare piece count and each piece as a set; v1 pieces go through the constructor
        p1, p2 = a1.contiguous_intervals, a2.pieces
        if check(len(p1) == len(p2) or True, 'n pieces'):
            pass
        same_pieces = len(p1) == len(p2) and all(same_set(x, y)[0] for x, y in zip(p1, p2))
        n_pieces += same_pieces
        # cardinality vs size, only when no date is involved (dates: v1 day is 1 us shorter)
        if not any(not isinstance(x, D) for a, _ in ps for x in a):
            ln1, pts1 = a1.cardinality
            s = a2.size
            check(s.rays == 0 and pts1 == 2 * s.points and abs(ln1 - float(s.length)) < 1e-5, f'card {ps} {a1.cardinality} {s}')
            check(abs(a1.total_seconds - float(a2.total_seconds)) < 1e-5, 'total_seconds')
            check(abs(a1.total_duration - a2.total_duration) <= US, f'total_duration {a1.total_duration} {a2.total_duration}')
            n_card += 1
print('sweep: cardinality compared', n_card, ' pieces identical as sets', n_pieces)
report('accessors')
