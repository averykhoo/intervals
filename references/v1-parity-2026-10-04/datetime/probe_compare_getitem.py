from common import *
from gen import rand_pair
import copy as copymod
import sys as _sys
D = dt.datetime; d_ = dt.date
r = random.Random(11)

# SELFTEST: sort_key order must be able to disagree with v1's order (reverse it)
a1, a2 = V1D(D(2024, 1, 1, 9, 0, 0, 1)), V2D(D(2024, 1, 1, 9, 0, 0, 1))
b1, b2 = V1D(D(2024, 1, 1, 10, 0, 0, 1)), V2D(D(2024, 1, 1, 10, 0, 0, 1))
selftest(lambda: (a1 < b1) == (b2.sort_key < a2.sort_key))

print('--- comparisons: v1 bool on endpoint lists; v2 TruthSet; sort_key for v1 order')
print('v1 a<b', a1 < b1, ' v2 a<b', repr(a2 < b2), ' bool:', safe(lambda: bool(a2 < b2)))
w1, w2 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 11, 0, 0, 1)), V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 1, 11, 0, 0, 1))
print('overlapping: v1 w<b', w1 < b1, ' v2 w<b', repr(w2 < b2), ' bool:', safe(lambda: bool(w2 < b2)))
print('scalar: v1 w < dt', w1 < D(2024, 1, 1, 10), ' v2', repr(w2 < D(2024, 1, 1, 10)))
print('scalar: v1 w < date', safe(lambda: w1 < d_(2024, 1, 2)), ' v2', safe(lambda: w2 < d_(2024, 1, 2)))
print('scalar: v1 w < Timestamp', safe(lambda: w1 < pd.Timestamp('2024-01-02')), ' v2', safe(lambda: w2 < pd.Timestamp('2024-01-02')))
print('foreign: v1 w < 5', safe(lambda: w1 < 5), ' v2', safe(lambda: w2 < 5))
n = 0
for i in range(500):
    x1, x2, _ = rand_pair(r, whole=False)
    y1, y2, _ = rand_pair(r, whole=False)
    # date-free sets only (a date's end differs below a microsecond)
    if any(isinstance(p, V1D) for p in ()):
        pass
    for op in ('__lt__', '__le__', '__gt__', '__ge__'):
        v1r = getattr(x1, op)(y1)
        v2r = getattr(x2.sort_key, op)(y2.sort_key)
        if not same_set(x1, x2)[0] or not same_set(y1, y2)[0]:
            continue
        has_date = any(p.sup == p.sup.replace(hour=0, minute=0, second=0, microsecond=0) and not p.sup_closed for p in list(x2) + list(y2))
        if has_date:
            continue
        check(v1r == v2r, f'{op} {x1} {y1} v1={v1r} v2={v2r}')
        n += 1
    e1 = (x1 == y1); e2 = (x2 == y2)
    check(e1 == e2 or has_date, f'== {x1} {y1} {e1} {e2}')
    check((x1 != y1) == (x2 != y2) or has_date, f'!=')
    check(x1 == x1.union(V1D()) and x2 == x2.union(V2D()), 'self eq')
print('sort_key order compared', n)
print('--- == with scalars and foreign types')
p1, p2 = V1D(D(2024, 1, 1, 9)), V2D(D(2024, 1, 1, 9))
print('v1 DTI(t) == t', safe(lambda: p1 == D(2024, 1, 1, 9)), ' v2', safe(lambda: p2 == D(2024, 1, 1, 9)), ' v2 workaround DTI(t) == DTI(t)', p2 == V2D(D(2024, 1, 1, 9)))
print('v1 DTI(d) == d', safe(lambda: V1D(d_(2024, 1, 1)) == d_(2024, 1, 1)), ' v2', safe(lambda: V2D(d_(2024, 1, 1)) == d_(2024, 1, 1)))
print('v1 == "x"', safe(lambda: p1 == 'x'), ' v2', safe(lambda: p2 == 'x'))
UTC = dt.timezone.utc
print('naive local 10:00 vs aware 02:00Z: v1 ==', V1D(D(2024, 1, 1, 10)) == V1D(D(2024, 1, 1, 2, tzinfo=UTC)),
      ' v2 ==', V2D(D(2024, 1, 1, 10)) == V2D(D(2024, 1, 1, 2, tzinfo=UTC)))
print('hash: v1', safe(lambda: hash(p1)), ' v2', safe(lambda: hash(p2)))

print('--- copy / __sizeof__')
c1 = p1.copy(); c1.update(V1D(D(2024, 1, 1, 10)))
print('v1 copy independent of original:', p1, c1)
print('v2 copy attr:', hasattr(p2, 'copy'), ' copy.copy:', safe(lambda: copymod.copy(p2)), ' deepcopy:', safe(lambda: copymod.deepcopy(p2) == p2))
print('v2 mutate attempt:', safe(lambda: setattr(p2, '_mi', None)))
print('sizeof: v1', safe(lambda: _sys.getsizeof(p1)), p1.__sizeof__(), ' v2', safe(lambda: _sys.getsizeof(p2)))

print('--- __getitem__')
A1 = V1D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 5, 17, 0, 0, 1))
A2 = V2D(D(2024, 1, 1, 9, 0, 0, 1), D(2024, 1, 5, 17, 0, 0, 1))
def g(label, k, expect=True, extra=()):
    x1 = safe(lambda: A1[k]); x2 = safe(lambda: A2[k])
    if x1[0] == 'ok' and x2[0] == 'ok':
        ok, diff = same_set(x1[1], x2[1], extra)
        print(f'{label}: same_set={ok} v1={x1[1]} v2={x2[1]} {diff or ""}')
        if expect is not None:
            check(ok == expect, label)
    else:
        print(f'{label}: v1 {x1} | v2 {x2}')
g('[dt:dt]', slice(D(2024, 1, 2, 10, 0, 0, 3), D(2024, 1, 3, 10, 0, 0, 3)))
g('[dt:dt] whole-hour stop', slice(D(2024, 1, 2, 10), D(2024, 1, 3, 10)))
g('[date:date]', slice(d_(2024, 1, 2), d_(2024, 1, 3)), extra=[D(2024, 1, 3, 23, 59, 59, 999999), D(2024, 1, 4)])
g('[:dt]', slice(None, D(2024, 1, 3, 10, 0, 0, 3)))
g('[dt:]', slice(D(2024, 1, 3, 10, 0, 0, 3), None))
g('[:]', slice(None, None))
g('[Timestamp:Timestamp]', slice(pd.Timestamp('2024-01-02 10:00:00.000003'), pd.Timestamp('2024-01-03 10:00:00.000003')))
g('[date:dt]', slice(d_(2024, 1, 2), D(2024, 1, 3, 10, 0, 0, 3)))
g('[dt:date]', slice(D(2024, 1, 2, 10, 0, 0, 3), d_(2024, 1, 3)), extra=[D(2024, 1, 3, 23, 59, 59, 999999), D(2024, 1, 4)])
g('reversed [b:a]', slice(D(2024, 1, 3, 10, 0, 0, 3), D(2024, 1, 2, 10, 0, 0, 3)), expect=None)
g('step', slice(D(2024, 1, 2), D(2024, 1, 3), 1), expect=None)
g('int bound', slice(0, 5), expect=None)
g('NEG_INF bound', slice(NEG_INF, D(2024, 1, 3, 10, 0, 0, 3)), expect=None)
g('NaT bound', slice(pd.NaT, D(2024, 1, 3, 10, 0, 0, 3)), expect=None)
# open-ended slice of a set whose end is at the slice bound: v1 open infinite end vs v2 closed (irrelevant for finite sets)
g('[DTI]', V1D(D(2024, 1, 2, 10, 0, 0, 3), D(2024, 1, 3)) if False else None, expect=None)
k1 = V1D(D(2024, 1, 2, 10, 0, 0, 3), D(2024, 1, 3, 10, 0, 0, 3)); k2 = V2D(D(2024, 1, 2, 10, 0, 0, 3), D(2024, 1, 3, 10, 0, 0, 3))
x1 = A1[k1]; x2 = safe(lambda: A2[k2]); x2w = A2 & k2
print('item DTI: v1', x1, '| v2', x2, '| workaround A & B', x2w, same_set(x1, x2w))
check(same_set(x1, x2w)[0], 'getitem DTI workaround')
for label, item in (('item dt inside', D(2024, 1, 2, 10)), ('item dt outside', D(2024, 1, 9, 10)), ('item date inside', d_(2024, 1, 3)),
                    ('item date partial', d_(2024, 1, 1)), ('item Timestamp', pd.Timestamp('2024-01-02 10:00'))):
    x1 = A1[item]
    x2 = safe(lambda: A2[item])
    w = V2D(item) if item in A2 else V2D()
    w_and = A2 & item
    print(f'{label}: v1 {x1} | v2 A[x] {x2} | workaround (DTI(x) if x in A else DTI()) {w} {same_set(x1, w)[0]} | A & x {w_and} {same_set(x1, w_and)[0]}')
    check(same_set(x1, w, extra=[D(2024, 1, 4), D(2024, 1, 3, 23, 59, 59, 999999)])[0], label)
report('compare_getitem')
