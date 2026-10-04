"""truthiness of the time classes, v1 vs v2.
v1 DateTimeInterval/TimeDeltaInterval define no __bool__ and no __len__ (object default: always True);
v1's own MultiInterval.__bool__ is `not is_empty`. v2's wrappers delegate to MultiInterval.__bool__.
seeded sweep over intervals built the same way on both sides, incl. empties made by set ops."""
import sys, random, datetime as dt
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import time_interval as v1t
import intervals as v2
import intervals.time_interval as v2t

print('v1 DTI has __bool__:', '__bool__' in vars(v1t.DateTimeInterval), '__len__:', '__len__' in vars(v1t.DateTimeInterval))
print('v1 TDI has __bool__:', '__bool__' in vars(v1t.TimeDeltaInterval), '__len__:', '__len__' in vars(v1t.TimeDeltaInterval))
print('v1 MultiInterval bool(empty):', bool(v1.MultiInterval()), ' v2 MultiInterval bool(empty):', bool(v2.MultiInterval()))

base = dt.datetime(2024, 5, 1, 10, 0, 0, 5)
H = dt.timedelta(hours=1)
rng = random.Random(20261004)
def mk(mod, cls_name, kind, a, b, sc, ec):
    C = getattr(mod, cls_name)
    return C(a, b, start_closed=sc, end_closed=ec)

stats = {}
def rec(label, a1, a2):
    b1, b2 = bool(a1), bool(a2)
    e1, e2 = a1.is_empty, a2.is_empty
    key = (label, b1, b2, e1 == e2, b2 == (not e2))
    stats[key] = stats.get(key, 0) + 1
    return b1, b2, e1, e2

hand = [
    ('DTI()', v1t.DateTimeInterval(), v2t.DateTimeInterval()),
    ('TDI()', v1t.TimeDeltaInterval(), v2t.TimeDeltaInterval()),
    ('DTI point', v1t.DateTimeInterval(base), v2t.DateTimeInterval(base)),
    ('DTI disjoint &', v1t.DateTimeInterval(base, base + H).intersection(v1t.DateTimeInterval(base + 2 * H, base + 3 * H)),
                       v2t.DateTimeInterval(base, base + H) & v2t.DateTimeInterval(base + 2 * H, base + 3 * H)),
    ('DTI A - A', v1t.DateTimeInterval(base, base + H).difference(v1t.DateTimeInterval(base, base + H)),
                  v2t.DateTimeInterval(base, base + H) - v2t.DateTimeInterval(base, base + H) if False else
                  v2t.DateTimeInterval(base, base + H).difference(v2t.DateTimeInterval(base, base + H))),
    ('DTI(empty) + td', (lambda e: e + H)(v1t.DateTimeInterval()), v2t.DateTimeInterval() + H),
    ('empty DTI - dt (-> TDI)', v1t.DateTimeInterval() - base, v2t.DateTimeInterval() - base),
]
for label, a1, a2 in hand:
    b1, b2, e1, e2 = rec(label, a1, a2)
    print(f'{label:26s} v1 bool {b1!s:5} is_empty {e1!s:5} | v2 bool {b2!s:5} is_empty {e2!s:5}')

# seeded sweep: random pieces, random flags, random set op (union/intersection/difference/symdiff)
n = 0
for _ in range(400):
    for cls_name, unit, origin in (('DateTimeInterval', dt.timedelta(minutes=1), base), ('TimeDeltaInterval', dt.timedelta(minutes=1), dt.timedelta(0))):
        def piece():
            a = rng.randint(0, 20); b = a + rng.randint(1, 4)  # v1 raises on (t, t) with an open flag
            sc, ec = rng.random() < .5, rng.random() < .5
            x = origin + a * unit + dt.timedelta(microseconds=5); y = origin + b * unit + dt.timedelta(microseconds=5)
            return (x, y, sc, ec)
        p, q = piece(), piece()
        op = rng.choice(['union', 'intersection', 'difference', 'symmetric_difference'])
        a1 = getattr(mk(v1t, cls_name, None, *p), op)(mk(v1t, cls_name, None, *q))
        a2 = getattr(mk(v2t, cls_name, None, *p), op)(mk(v2t, cls_name, None, *q))
        rec(f'sweep {cls_name}', a1, a2); n += 1
print('sweep cases', n)
for k, v in sorted(stats.items(), key=str):
    print(' ', k, v)

# the claims, each able to fail:
mism_empty = sum(v for k, v in stats.items() if not k[3])
print('is_empty disagreements v1 vs v2:', mism_empty)
v1_false = sum(v for k, v in stats.items() if k[1] is False)
print('v1 bool False count:', v1_false)
v2_not_nonempty = sum(v for k, v in stats.items() if not k[4])
print('v2 bool != (not is_empty) count:', v2_not_nonempty)
empties = sum(v for k, v in stats.items() if k[2] is False)
print('cases where v2 bool False (empty):', empties)
assert mism_empty == 0 and v1_false == 0 and v2_not_nonempty == 0 and empties > 0
# sabotage: a wrong expectation "v1 and v2 truthiness always agree" must be caught
caught = any(k[1] != k[2] for k in stats)
print('SABOTAGE (expect v1 bool == v2 bool everywhere) caught:', caught)
assert caught
