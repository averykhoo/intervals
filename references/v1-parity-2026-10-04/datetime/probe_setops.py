from common import *
from gen import rand_pair
D = dt.datetime; d_ = dt.date
r = random.Random(23)


def brute(op, sets, p):
    """membership of p in op(sets) from membership in each (v2 sets as the oracle of the pieces)"""
    ms = [p in s for s in sets]
    if op == 'union':
        return any(ms)
    if op == 'intersection':
        return all(ms)
    if op == 'difference':
        return ms[0] and not any(ms[1:])
    if op == 'symmetric_difference':
        return sum(ms) % 2 == 1


# SELFTEST: the brute oracle must catch a wrong op (union vs intersection)
x = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1)); y = V2D(D(2024, 1, 1, 9, 30, 0, 1), D(2024, 1, 1, 11, 0, 0, 1))
selftest(lambda: all(brute('intersection', [x, y], p) == (p in x.union(y)) for p in candidates(*v2_points(x), *v2_points(y))))

stats = {}
mism = {}
for i in range(600):
    k = r.choice([0, 1, 1, 2, 2, 3])
    pairs = [rand_pair(r, max_pieces=3, days=4) for _ in range(k + 1)]
    s1 = [p[0] for p in pairs]; s2 = [p[1] for p in pairs]
    if not all(same_set(a, b)[0] for a, b in zip(s1, s2)):
        check(False, 'build')
        continue
    for op in ('union', 'intersection', 'difference', 'symmetric_difference'):
        x1 = safe(lambda: getattr(s1[0], op)(*s1[1:]))
        x2 = safe(lambda: getattr(s2[0], op)(*s2[1:]))
        if x1[0] != 'ok' or x2[0] != 'ok':
            check(False, f'{op} raised v1={x1} v2={x2}')
            continue
        pts = candidates(*[q for s in s2 for q in v2_points(s)], *v1_points(x1[1]))
        ok12, diff = same_set(x1[1], x2[1], extra=pts)
        ok_or = all((p in x2[1]) == brute(op, s2, p) for p in pts)
        ok_v1 = all((p in x1[1]) == brute(op, s1, p) for p in pts)
        key = (op, k)
        stats.setdefault(key, [0, 0, 0, 0])
        stats[key][0] += 1; stats[key][1] += ok12; stats[key][2] += ok_or; stats[key][3] += ok_v1
        check(ok_or, f'v2 {op} vs oracle')
        if not ok12:
            mism.setdefault(key, (s1, x1[1], x2[1], diff, ok_v1))
    # relations, k >= 1
    if k >= 1:
        a1, b1, a2, b2 = s1[0], s1[1], s2[0], s2[1]
        for rel in ('isdisjoint', 'issubset', 'issuperset'):
            r1 = safe(lambda: getattr(a1, rel)(b1)); r2 = safe(lambda: getattr(a2, rel)(b2))
            key = (rel,)
            stats.setdefault(key, [0, 0, 0, 0]); stats[key][0] += 1; stats[key][1] += (r1 == r2)
            if r1 != r2:
                mism.setdefault((rel, a1.is_empty, b1.is_empty), (a1, b1, r1, r2))
print('op, n_others: [cases, v1==v2 as sets, v2==oracle, v1==oracle]')
for k_, v in sorted(stats.items(), key=str):
    print('  ', k_, v)
for k_, v in mism.items():
    print('MISMATCH', k_, v)

print('--- relations with empty sets (exact reasoning: the empty set is a subset of everything)')
E1, E2 = V1D(), V2D()
X1, X2 = V1D(D(2024, 1, 1, 9)), V2D(D(2024, 1, 1, 9))
for rel in ('isdisjoint', 'issubset', 'issuperset'):
    for (l1, l2, a, b) in (('E', 'E', (E1, E2), (E1, E2)), ('E', 'X', (E1, E2), (X1, X2)), ('X', 'E', (X1, X2), (E1, E2))):
        print(f'{l1}.{rel}({l2}): v1', safe(lambda: getattr(a[0], rel)(b[0])), ' v2', safe(lambda: getattr(a[1], rel)(b[1])))
print('--- scalar arguments')
A1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 3, 9, 0, 0, 1)); A2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 3, 9, 0, 0, 1))
for label, x in (('dt', D(2024, 1, 5, 10)), ('date', d_(2024, 1, 5)), ('Timestamp', pd.Timestamp('2024-01-05 10:00'))):
    for op in ('union', 'intersection', 'difference', 'symmetric_difference'):
        x1 = getattr(A1, op)(x); x2 = getattr(A2, op)(x)
        ok, diff = same_set(x1, x2, extra=[D(2024, 1, 5, 23, 59, 59, 999999), D(2024, 1, 6)])
        check(ok, f'{op}({label})')
    for rel in ('isdisjoint', 'issubset', 'issuperset'):
        check(getattr(A1, rel)(x) == getattr(A2, rel)(x), f'{rel}({label})')
print('scalar args checked; foreign: v1', safe(lambda: A1.union(5)), ' v2', safe(lambda: A2.union(5)), ' str v2', safe(lambda: A2.union('2024-01-01')))
print('--- in-place: v1 mutates and returns self; v2 immutable, rebinding operators')
B1 = V1D(D(2024, 1, 2, 9, 0, 0, 1), D(2024, 1, 4, 9, 0, 0, 1)); B2 = V2D(D(2024, 1, 2, 9, 0, 0, 1), D(2024, 1, 4, 9, 0, 0, 1))
for m1, opsym in (('update', '|'), ('intersection_update', '&'), ('difference_update', '-'), ('symmetric_difference_update', '^')):
    c1 = A1.copy(); ret = getattr(c1, m1)(B1)
    c2 = A2
    if opsym == '|': c2 |= B2
    elif opsym == '&': c2 &= B2
    elif opsym == '-': c2 = c2.difference(B2)
    else: c2 ^= B2
    ok, diff = same_set(c1, c2)
    check(ok and ret is c1, m1)
    print(f'{m1}: v1 returns self {ret is c1}, mutated {c1} | v2 rebinding {c2} same={ok}; v2 has {m1}? {hasattr(A2, m1)}')
print('v2 A - B for difference?', safe(lambda: A2 - B2))
print('v1 operators | & ^ ~ exist?', [safe(lambda: eval(f'A1 {o} B1'))[0] for o in '|&^'], ' v2', [safe(lambda: eval(f'A2 {o} B2'))[0] for o in '|&^'])
report('setops')
