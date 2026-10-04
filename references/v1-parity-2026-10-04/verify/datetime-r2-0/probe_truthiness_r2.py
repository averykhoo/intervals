import sys, random, datetime as dt
sys.path[:0] = ['.', 'archive/v1']
import time_interval as v1t
import multi_interval as v1m
import intervals as v2
from intervals.time_interval import DateTimeInterval as D2, TimeDeltaInterval as T2
D1, T1 = v1t.DateTimeInterval, v1t.TimeDeltaInterval

print('v1 DTI has __bool__/__len__:', '__bool__' in vars(D1), '__len__' in vars(D1), hasattr(D1, '__len__'))
print('v1 TDI has __bool__/__len__:', '__bool__' in vars(T1), '__len__' in vars(T1), hasattr(T1, '__len__'))

B = dt.datetime(2024, 1, 1, 12)
H = dt.timedelta(hours=1)
cases = {
  'D()': (D1(), D2()),
  'T()': (T1(), T2()),
  'disjoint &': (D1(B, B+H).intersection(D1(B+2*H, B+3*H)), D2(B, B+H) & D2(B+2*H, B+3*H)),
  'A - A': (D1(B, B+H).difference(D1(B, B+H)), D2(B, B+H).difference(D2(B, B+H))),
  'T disjoint &': (T1(H, 2*H).intersection(T1(3*H, 4*H)), T2(H, 2*H) & T2(3*H, 4*H)),
  'nonempty D': (D1(B, B+H), D2(B, B+H)),
  'point D': (D1(B, B), D2(B, B)),
}
for k, (a, b) in cases.items():
    print(f'{k:14s} v1 bool {bool(a)!s:5s} v1 is_empty {a.is_empty!s:5s} v1 bool(.interval) {bool(a.interval)!s:5s} | v2 bool {bool(b)!s:5s} v2 is_empty {b.is_empty}')

# v2 spellings of v1's always-True: none needed; `x is not None` / True. v2 spelling of v1's own non-empty test: not x.is_empty
# seeded sweep: random pieces, unions/intersections/differences
rng = random.Random(20261004)
def rnd_pair():
    a = rng.randint(0, 20); b = a + rng.randint(-2, 6)
    sc, ec = rng.random() < .5, rng.random() < .5
    return a, b, sc, ec
n = disagree_empty = v1_false = v2_vs_not_empty = v2_false = v1interval_vs_v2 = 0
raised = 0
for _ in range(800):
    ops = []
    try:
        a, b, sc, ec = rnd_pair(); c, d, sc2, ec2 = rnd_pair()
        if b < a or d < c:
            continue
        x1 = D1(B + a*H, B + b*H + dt.timedelta(microseconds=1), start_closed=sc, end_closed=ec)  # avoid v1 snap ambiguity
        y1 = D1(B + c*H, B + d*H + dt.timedelta(microseconds=1), start_closed=sc2, end_closed=ec2)
        x2 = D2(B + a*H, B + b*H + dt.timedelta(microseconds=1), start_closed=sc, end_closed=ec)
        y2 = D2(B + c*H, B + d*H + dt.timedelta(microseconds=1), start_closed=sc2, end_closed=ec2)
    except Exception as e:
        raised += 1; continue
    op = rng.choice(['&', '-', '|'])
    r1 = {'&': x1.intersection, '-': x1.difference, '|': x1.union}[op](y1); r2 = {'&': x2.intersection, '-': x2.difference, '|': x2.union}[op](y2)
    n += 1
    disagree_empty += r1.is_empty != r2.is_empty
    v1_false += not bool(r1)
    v2_vs_not_empty += bool(r2) != (not r2.is_empty)
    v2_false += not bool(r2)
    v1interval_vs_v2 += bool(r1.interval) != bool(r2)
print(f'sweep n={n} skipped(raised)={raised} is_empty disagree={disagree_empty} v1 False={v1_false} '
      f'v2 bool != not is_empty={v2_vs_not_empty} v2 False={v2_false} v1 bool(.interval) != v2 bool={v1interval_vs_v2}')
# sabotage: the wrong expectation "v2 bool == v1 bool" must be caught
assert v2_false > 0, 'sabotage not caught: no empties in sweep'
mismatch = sum(1 for k, (a, b) in cases.items() if bool(a) != bool(b))
print('sabotage (expect v2 bool == v1 bool) mismatches on hand cases:', mismatch, '-> caught' if mismatch else '-> NOT caught')
# arithmetic empties (claim: empty + td, empty DTI - dt)
for k, a, b in [('empty D + td', D1() + H, D2() + H), ('empty D - dt', D1() - B, D2() - B), ('empty T + td', T1() , T2() + H)]:
    print(f'{k:14s} v1 bool {bool(a)!s:5s} is_empty {a.is_empty!s:5s} | v2 bool {bool(b)!s:5s} is_empty {b.is_empty} type {type(b).__name__}')
# v2 composition reproducing v1's (constant) truthiness: any object test; and v1's own MultiInterval rule in v2
print('v2 `x is not None` on empty:', D2() is not None, '| v1 MultiInterval bool(empty):', bool(v1m.MultiInterval()), '| v2 MultiInterval bool(empty):', bool(v2.MultiInterval()))
print('len: v1 DTI', end=' ')
try: print(len(D1(B, B+H)))
except TypeError as e: print('TypeError', e)
print('len: v2 DTI', len(D2(B, B+H)), len(D2()))
