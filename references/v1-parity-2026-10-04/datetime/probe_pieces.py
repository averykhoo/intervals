"""contiguous_intervals (v1) vs pieces / iteration (v2): where do they differ, and why"""
from common import *
from gen import rand_pair
D = dt.datetime
r = random.Random(7)


def piece_sets_equal(p1, p2):
    return len(p1) == len(p2) and all(same_set(x, y)[0] for x, y in zip(p1, p2))


# SELFTEST: a shifted piece list must be caught
selftest(piece_sets_equal, V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1)).contiguous_intervals,
         V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 2)).pieces)

nonempty = same = 0
reasons = {}
examples = {}
for i in range(400):
    a1, a2, ps = rand_pair(r)
    if a1.is_empty:
        continue
    nonempty += 1
    p1, p2 = a1.contiguous_intervals, a2.pieces
    if piece_sets_equal(p1, p2):
        same += 1
        continue
    # union of v1's pieces vs v1's set: is v1's piece list even a partition of its own set?
    u1 = V1D().union(*p1)
    own, _ = same_set_any(u1, a1)
    # are v2 pieces the merge of v1 pieces (adjacent days tiling)?
    merged = V2D().union(*[V2D.from_seconds(v2.MultiInterval(0)) for _ in ()]) if False else None
    key = ('v1 pieces != v1 set (constructor snap)' if not own else
           'v1 keeps adjacent days apart, v2 tiles them' if len(p1) > len(p2) else 'other')
    reasons[key] = reasons.get(key, 0) + 1
    examples.setdefault(key, (ps, a1, p1, a2))
print('nonempty', nonempty, 'identical piece lists', same, 'differences', reasons)
for k, (ps, a1, p1, a2) in examples.items():
    print('EXAMPLE', k)
    print('   v1 set   ', a1)
    print('   v1 pieces', p1)
    print('   v2 set   ', a2)

# v1's snap through contiguous_intervals: whole-second ends that arise from set ops
cases = [
    ('point at a whole hour', V1D(D(2024, 1, 1, 11)), V2D(D(2024, 1, 1, 11))),
    ('difference leaves a whole-minute end', V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 12, 0, 0, 1)).difference(V1D(D(2024, 1, 1, 10, 30, 0, 0), D(2024, 1, 1, 10, 45, 0, 1), start_closed=False)),
     V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 12, 0, 0, 1)).difference(V2D(D(2024, 1, 1, 10, 30, 0, 0), D(2024, 1, 1, 10, 45, 0, 1), start_closed=False))),
    ('midnight point', V1D(D(2024, 1, 1)), V2D(D(2024, 1, 1))),
]
for label, a1, a2 in cases:
    p1 = a1.contiguous_intervals
    u1 = V1D().union(*p1)
    own, diff = same_set_any(u1, a1)
    check(not own, label + ' (v1 snap expected)')
    ok2 = V2D().union(*a2.pieces) == a2 and list(a2) == list(a2.pieces)
    check(ok2, label + ' v2 pieces partition')
    print(f'{label}: v1 pieces union == v1 set? {own} {diff} | v2 pieces union == set? {ok2}  v1 pieces {p1}  v2 pieces {a2.pieces}')
report('pieces')
