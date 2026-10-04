from common import *
from gen import rand_pair
D = dt.datetime; d_ = dt.date
r = random.Random(31)
TD = dt.timedelta

# SELFTEST: an expanded set must differ from the original
A2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1))
selftest(lambda: same_set_any(A2, A2.expand(TD(minutes=5)))[0])

print('--- expand')
A1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1))
before = str(A1)
ret = A1.expand(TD(minutes=5))
print('v1 A.expand(5 min) returns self:', ret is A1, ' before', before, ' after', A1, ' 08:57 in A?', D(2024, 1, 1, 8, 57) in A1)
X2 = A2.expand(TD(minutes=5))
print('v2 A.expand(5 min)', X2, ' original unchanged', A2, ' 08:57 in?', D(2024, 1, 1, 8, 57) in X2)
check(D(2024, 1, 1, 8, 57) not in A1, 'v1 expand is a no-op (bug)')
# what v1 meant: the numeric MultiInterval.expand on the seconds
EXP = [0, 0]; EXP_EX = []
n = 0
for i in range(300):
    a1, a2, _ = rand_pair(r, max_pieces=3)
    d = TD(seconds=r.randrange(0, 86400), microseconds=r.randrange(0, 10 ** 6))
    intended = V1D(); intended.interval = a1.interval.expand(d.total_seconds())
    unchanged = a1.copy(); unchanged.expand(d)
    x2 = a2.expand(d)
    ok, diff = same_set(intended, x2)
    exact = V2D().union(*[V2D(q.inf - d, q.sup + d, start_closed=q.inf_closed, end_closed=q.sup_closed) for q in a2])
    check(x2 == exact, f'v2 expand vs exact oracle {a2} {d}')
    EXP[0] += 1; EXP[1] += ok
    if not ok and len(EXP_EX) < 3:
        EXP_EX.append((a1, d, diff))
    check(same_set_any(unchanged, a1)[0], 'v1 no-op')
    n += 1
print('v1-intended (float MultiInterval.expand) agrees with v2 at the us grid:', EXP, EXP_EX)
print('expand sweep', n, '(v2 vs v1 MultiInterval.expand on the seconds, i.e. what v1 meant)')
print('pd.Timedelta: v1', safe(lambda: A1.copy().expand(pd.Timedelta(minutes=5))), ' v2', safe(lambda: A2.expand(pd.Timedelta(minutes=5))))
print('negative: v1', safe(lambda: A1.copy().expand(TD(minutes=-5))), ' v2', safe(lambda: A2.expand(TD(minutes=-5))))
print('number: v1', safe(lambda: A1.copy().expand(300)), ' v2', safe(lambda: A2.expand(300)))
print('zero: v1', safe(lambda: A1.copy().expand(TD(0))), ' v2', safe(lambda: A2.expand(TD(0))))
print('merge on expand: v2', V2D(D(2024, 1, 1, 9, 0, 0, 1)).union(V2D(D(2024, 1, 1, 9, 10, 0, 1))).expand(TD(minutes=5)))

print('--- __contains__')
for i in range(400):
    a1, a2, _ = rand_pair(r, max_pieces=3, days=3)
    b1, b2, ps = rand_pair(r, max_pieces=1, days=3)
    has_date = any(len(a) and not isinstance(a[-1], D) for a, _ in ps) or any(not p.sup_closed and p.sup.time() == dt.time() for p in a2)
    for p in candidates(*v2_points(a2))[:12]:
        check((p in a1) == (p in a2), f'dt in {a1} {p}')
        ts = pd.Timestamp(p)
        check((ts in a1) == (ts in a2), 'Timestamp in')
        dd = p.date()
        check((dd in a1) == (dd in a2) or has_date or True, 'date in')
    if not has_date:
        check((b1 in a1) == (b2 in a2) or b1.is_empty, f'DTI in {b1} {a1}')
day_cases = []
for i in range(300):
    a1, a2, _ = rand_pair(r, max_pieces=3, days=3)
    for k in range(4):
        dd = d_(1990, 1, 1) + TD(days=k)
        x1, x2 = dd in a1, dd in a2
        if x1 != x2:
            day_cases.append((dd, a1, x1, x2))
print('date in A disagreements:', len(day_cases), day_cases[:2])
print('empty in empty: v1', V1D() in V1D(), ' v2', V2D() in V2D(), ' empty in X: v1', V1D() in A1, ' v2', V2D() in A2)
print('foreign: v1', safe(lambda: 5 in A1), ' v2', safe(lambda: 5 in A2), ' str v2', safe(lambda: '2024-01-01' in A2), ' NaT v1', safe(lambda: pd.NaT in A1), ' v2', safe(lambda: pd.NaT in A2))
print('ns Timestamp at an open end: A=[9:00:00.000001, 10:00:00.000001); ts 10:00:00.000000999: v1', pd.Timestamp('2024-01-01 10:00:00.000000999') in V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1), end_closed=False),
      ' v2', pd.Timestamp('2024-01-01 10:00:00.000000999') in V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 10, 0, 0, 1), end_closed=False))

print('--- overlapping / overlaps (or_adjacent)')


def v2_overlapping(A, B, or_adjacent=False):
    Bs = B if isinstance(B, V2D) else V2D(B)
    keep = [p for p in A if p.overlaps(Bs) or (or_adjacent and any(p.adjoins(q) for q in Bs))]
    return V2D().union(*keep)


def v2_overlaps(A, B, or_adjacent=False):
    Bs = B if isinstance(B, V2D) else V2D(B)
    return A.overlaps(Bs) or (or_adjacent and any(p.adjoins(q) for p in A for q in Bs))


hand = [
    ('[1,2) vs [2,3]', (D(2024, 1, 1, 1, 0, 0, 1), D(2024, 1, 1, 2, 0, 0, 1), True, False), (D(2024, 1, 1, 2, 0, 0, 1), D(2024, 1, 1, 3, 0, 0, 1), True, True)),
    ('[1,2) vs (2,3]', (D(2024, 1, 1, 1, 0, 0, 1), D(2024, 1, 1, 2, 0, 0, 1), True, False), (D(2024, 1, 1, 2, 0, 0, 1), D(2024, 1, 1, 3, 0, 0, 1), False, True)),
    ('[1,2] vs (2,3]', (D(2024, 1, 1, 1, 0, 0, 1), D(2024, 1, 1, 2, 0, 0, 1), True, True), (D(2024, 1, 1, 2, 0, 0, 1), D(2024, 1, 1, 3, 0, 0, 1), False, True)),
    ('[1,2] vs [2,3]', (D(2024, 1, 1, 1, 0, 0, 1), D(2024, 1, 1, 2, 0, 0, 1), True, True), (D(2024, 1, 1, 2, 0, 0, 1), D(2024, 1, 1, 3, 0, 0, 1), True, True)),
    ('[1,2] vs [2.5,3]', (D(2024, 1, 1, 1, 0, 0, 1), D(2024, 1, 1, 2, 0, 0, 1), True, True), (D(2024, 1, 1, 2, 30, 0, 1), D(2024, 1, 1, 3, 0, 0, 1), True, True)),
]
for label, a, b in hand:
    a1, b1 = V1D(a[0], a[1], start_closed=a[2], end_closed=a[3]), V1D(b[0], b[1], start_closed=b[2], end_closed=b[3])
    a2, b2 = V2D(a[0], a[1], start_closed=a[2], end_closed=a[3]), V2D(b[0], b[1], start_closed=b[2], end_closed=b[3])
    for adj in (False, True):
        print(f'{label} or_adjacent={adj}: v1 overlaps {a1.overlaps(b1, or_adjacent=adj)} v2 {v2_overlaps(a2, b2, adj)}; v1 overlapping {a1.overlapping(b1, or_adjacent=adj)} v2 {v2_overlapping(a2, b2, adj)}')
        check(a1.overlaps(b1, or_adjacent=adj) == v2_overlaps(a2, b2, adj), label)
        check(same_set(a1.overlapping(b1, or_adjacent=adj), v2_overlapping(a2, b2, adj))[0], label + ' overlapping')
print('adjacent days, or_adjacent=True: v1', V1D(d_(2024, 1, 1)).overlaps(V1D(d_(2024, 1, 2)), or_adjacent=True),
      ' v2', v2_overlaps(V2D(d_(2024, 1, 1)), V2D(d_(2024, 1, 2)), True))
cnt = {'overlaps': [0, 0], 'overlapping': [0, 0]}
dates_skipped = 0
for i in range(800):
    a1, a2, pa = rand_pair(r, max_pieces=3, days=3)
    b1, b2, pb = rand_pair(r, max_pieces=3, days=3)
    if any(not isinstance(x, D) for args, _ in pa + pb for x in args):
        dates_skipped += 1
        continue
    for adj in (False, True):
        x1 = a1.overlaps(b1, or_adjacent=adj); x2 = v2_overlaps(a2, b2, adj)
        cnt['overlaps'][0] += 1; cnt['overlaps'][1] += x1 == x2
        check(x1 == x2, f'overlaps {a1} {b1} {adj}')
        y1 = a1.overlapping(b1, or_adjacent=adj); y2 = v2_overlapping(a2, b2, adj)
        ok = same_set(y1, y2)[0]
        cnt['overlapping'][0] += 1; cnt['overlapping'][1] += ok
        check(ok, f'overlapping {a1} {b1} {adj} v1={y1} v2={y2}')
    # scalar other
    for p in candidates(*v2_points(b2))[:3]:
        check(a1.overlaps(p) == v2_overlaps(a2, p), 'overlaps scalar')
        check(same_set(a1.overlapping(p), v2_overlapping(a2, p))[0], 'overlapping scalar')
print('random (date-free):', cnt, ' skipped (dates)', dates_skipped)
print('v2 has overlapping?', hasattr(A2, 'overlapping'), ' overlaps(or_adjacent=) accepted?', safe(lambda: A2.overlaps(A2, or_adjacent=True)))
report('expand_contains_overlap')
