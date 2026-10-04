# v1's __main__ blocks are its only worked usage examples: multi_interval.py:2001-2019, time_interval.py:727-732.
# translate each line to v2 and compare on random_multi_interval inputs (v1's own generator, seeded)
from common import *
import datetime as dt
import time_interval as v1t
from intervals import DateTimeInterval as DTI
warnings.simplefilter('ignore')
random.seed(2026)
def mem_ok(r2, truth, pts):
    return all((p in r2) == truth(p) for p in pts)
stats = {}; bad = []
def rec(k, ok, info=None):
    stats[k] = stats.get(k, 0) + bool(ok)
    if not ok and len(bad) < 12: bad.append((k,) + (info or ()))
N = 300
for _ in range(N):
    i = v1.random_multi_interval(-100, 100, random.randint(0, 5), 0)
    j = v1.random_multi_interval(-100, 100, random.randint(0, 5), 0)
    I, J = conv(i), conv(j)
    pts = probes_of(I, J) | {0, Fraction(1, 2), -Fraction(1, 2)}
    # closed_hull
    ch1 = i.closed_hull; rec('closed_hull', (MI() if ch1 is None else conv(ch1)) == I.closed_hull, (str(I), str(ch1), str(I.closed_hull)))
    # reciprocal: v1 is known wrong at zero; compare only where 0 not in I
    if 0 not in I: rec('reciprocal(0 not in i)', conv(i.reciprocal()) == I.reciprocal(), (str(I), str(i.reciprocal()), str(I.reciprocal())))
    for name, f1, f2, truth in [
        ('union', lambda: i.union(j), lambda: I | J, lambda p: p in I or p in J),
        ('intersection', lambda: i.intersection(j), lambda: I & J, lambda p: p in I and p in J),
        ('difference', lambda: i.difference(j), lambda: I.difference(J), lambda p: p in I and p not in J),
        ('symmetric_difference', lambda: i.symmetric_difference(j), lambda: I ^ J, lambda p: (p in I) != (p in J)),
        ('abs', lambda: abs(i), lambda: abs(I), lambda p: (p in I) or (-p in I) if p >= 0 else False),
        ('i[-50:50]', lambda: i[-50:50], lambda: I[-50:50], lambda p: p in I and -50 <= p <= 50),
        ('j[4:]', lambda: j[4:], lambda: J[4:], lambda p: p in J and p >= 4),
        ('j[:]', lambda: j[:], lambda: J[:], lambda p: p in J),
        ('i[:0]', lambda: i[:0], lambda: I[:0], lambda p: p in I and p <= 0),
    ]:
        r1 = conv(f1()); r2 = f2()
        rec(name + ' v2 exact', mem_ok(r2, truth, pts), (str(I), str(J), str(r2)))
        rec(name + ' v1==v2', r1 == r2, (str(I), str(J), str(r1), str(r2)))
    # the `in` (subset) checks of the block
    U = I | J; X = I & J; D = I.difference(J)
    rec('in-subset', [i.union(j) in j, j in i.union(j), i.intersection(j) in i, i.intersection(j) in j, i.difference(j) in i, i.difference(j) in j]
        == [U in J, J in U, X in I, X in J, D in I, D in J], (str(I), str(J)))
    # overlapping: the whole pieces of i meeting j
    o2 = MI().union(*[p for p in I if p.overlaps(J)])
    rec('overlapping', conv(i.overlapping(j)) == o2, (str(I), str(J), str(i.overlapping(j)), str(o2)))
print(f'N={N}'); [print(f'  {k:32s} {v}') for k, v in stats.items()]
for b in bad: print('  BAD', b)
# sabotage: a wrong truth table is caught
I = MI(0, 1); assert not mem_ok(I, lambda p: 0 < p <= 1, probes_of(I))

print('--- time_interval.py __main__')
x1 = v1t.DateTimeInterval(dt.date(2018, 9, 1)); x2 = DTI(dt.date(2018, 9, 1))
print('x        v1:', x1, '| v2:', x2)
x1.update(x1 + dt.timedelta(999)); x2 = x2 | (x2 + dt.timedelta(999))
print('update   v1:', x1, '| v2:', x2)
print('slice    v1:', x1[dt.date(2018, 1, 2):dt.date(2018, 8, 9)], '| v2:', x2[dt.date(2018, 1, 2):dt.date(2018, 8, 9)])
print('inter    v1:', x1.intersection(v1t.DateTimeInterval(dt.date(2018, 9, 1), dt.date(2019, 5, 30))), '| v2:', x2 & DTI(dt.date(2018, 9, 1), dt.date(2019, 5, 30)))
