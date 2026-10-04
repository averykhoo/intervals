"""isdisjoint/issubset/issuperset: explain every v1/v2 disagreement (empty set; the snapped day's sub-us gap)"""
from common import *
from gen import rand_pair
D = dt.datetime
r = random.Random(29)


def has_date(a2):
    return any(not p.sup_closed and p.sup == p.sup.replace(hour=0, minute=0, second=0, microsecond=0) for p in a2)


def brute_subset(a2, b2):
    pts = candidates(*v2_points(a2), *v2_points(b2))
    return all((p not in a2) or (p in b2) for p in pts)


# SELFTEST: brute_subset must be able to say False
selftest(brute_subset, V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 11, 0, 0, 1)), V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1)))

counts = {}
for i in range(3000):
    a1, a2, _ = rand_pair(r, max_pieces=3, days=3)
    b1, b2, _ = rand_pair(r, max_pieces=3, days=3)
    for rel in ('isdisjoint', 'issubset', 'issuperset'):
        x1, x2 = getattr(a1, rel)(b1), getattr(a2, rel)(b2)
        if x1 == x2:
            why = 'agree'
        elif a1.is_empty and b1.is_empty:
            why = 'both empty'
        elif has_date(a2) or has_date(b2):
            why = 'a date piece (v1 day ends at 23:59:59.999999)'
        else:
            why = 'UNEXPLAINED'
            check(False, f'{rel} {a1} {b1} v1={x1} v2={x2}')
        # v2 against brute force at us grid (date-free only: sub-us gaps are not on the grid)
        if rel == 'issubset' and not has_date(a2) and not has_date(b2):
            check(x2 == brute_subset(a2, b2), f'v2 issubset oracle {a2} {b2}')
        counts[(rel, why)] = counts.get((rel, why), 0) + 1
for k, v in sorted(counts.items()):
    print(k, v)
report('relations')
