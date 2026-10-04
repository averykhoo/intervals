from common import *
from gen import rand_pair
import zoneinfo
D = dt.datetime; d_ = dt.date
r = random.Random(53)

# SELFTEST: v2's repr round-trip must detect a different set
selftest(lambda: eval(repr(V2D(D(2024, 1, 1, 10))), {'datetime': dt, 'DateTimeInterval': V2D}) == V2D(D(2024, 1, 1, 11)))

cases = [
    ('empty', (), {}), ('point', (D(2024, 1, 1, 10, 30, 15, 5),), {}), ('day', (d_(2024, 1, 1),), {}),
    ('week', (d_(2024, 1, 1), d_(2024, 1, 7)), {}), ('half-open', (D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 17, 0, 0, 1)), {'end_closed': False}),
    ('open', (D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 17, 0, 0, 1)), {'start_closed': False, 'end_closed': False}),
    ('aware', (D(2024, 1, 1, 9, 0, 0, 1, tzinfo=dt.timezone.utc),), {}),
]
for label, args, kw in cases:
    a1, a2 = V1D(*args, **kw), V2D(*args, **kw)
    print(f'{label}: v1 str {str(a1)!r} repr {repr(a1)!r} | v2 str {str(a2)!r} repr {repr(a2)!r}')
m1 = V1D(D(2024, 1, 1, 9, 0, 0, 1)).union(V1D(d_(2024, 1, 3)), V1D(D(2024, 1, 5, 1, 0, 0, 1), D(2024, 1, 5, 2, 0, 0, 1), start_closed=False))
m2 = V2D(D(2024, 1, 1, 9, 0, 0, 1)).union(V2D(d_(2024, 1, 3)), V2D(D(2024, 1, 5, 1, 0, 0, 1), D(2024, 1, 5, 2, 0, 0, 1), start_closed=False))
print('multi: v1', str(m1), '| v2', str(m2))
print('lossy v1: two different points print the same:', str(V1D(D(2024, 1, 1, 10, 0, 0, 5))), str(V1D(D(2024, 1, 1, 10, 0, 30))),
      '| v2', str(V2D(D(2024, 1, 1, 10, 0, 0, 5))), str(V2D(D(2024, 1, 1, 10, 0, 30))))
ns = {'datetime': dt, 'DateTimeInterval': V2D, 'NEG_INF': NEG_INF, 'POS_INF': POS_INF, 'zoneinfo': zoneinfo, 'MultiInterval': M2}
v1_raised = v2_raised = rt = collide1 = collide2 = n = 0
seen1, seen2 = {}, {}
for i in range(400):
    a1, a2, _ = rand_pair(r, max_pieces=3, days=3)
    n += 1
    s1 = safe(lambda: str(a1)); s2 = safe(lambda: str(a2))
    v1_raised += s1[0] != 'ok'; v2_raised += s2[0] != 'ok'
    back = eval(repr(a2), ns)
    rt += back == a2
    check(back == a2, f'repr round trip {a2!r}')
    if s1[0] == 'ok':
        if s1[1] in seen1 and seen1[s1[1]] != a2:
            collide1 += 1
        seen1.setdefault(s1[1], a2)
    if s2[1] in seen2 and seen2[s2[1]] != a2:
        collide2 += 1
    seen2.setdefault(s2[1], a2)
print(f'random {n}: str raised v1 {v1_raised} v2 {v2_raised}; v2 repr round-trips {rt}; distinct sets with equal str: v1 {collide1} v2 {collide2}')
print('v1 str of a sub-us end (ns Timestamp):', str(V1D(pd.Timestamp('2024-01-01 10:00:00.000000001'))), '| v2', str(V2D(pd.Timestamp('2024-01-01 10:00:00.000000001'))))
report('str')
