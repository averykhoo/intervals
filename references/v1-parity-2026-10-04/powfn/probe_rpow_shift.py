"""b ** A (__rpow__) and the four shifts, v1 vs v2's spelling"""
import sys; sys.path.insert(0, '.scratch/v1-parity/powfn')
import math, random
from fractions import Fraction as F
from common import *

def show(label, f1, f2):
    r1, e1 = run(f1); r2, e2 = run(f2)
    print(f'{label:40s} v1: {e1 or s1(r1)!s:42s} v2: {e2 or r2!s}')
    return r1, r2

print('--- rpow')
show('2 ** [1,3]', lambda: 2 ** V1(1, 3), lambda: 2 ** V2(1, 3))
show('V1(2) ** [1,3] vs 2 ** [1,3]', lambda: V1(2) ** V1(1, 3), lambda: 2 ** V2(1, 3))
show('0.5 ** [1,3]', lambda: V1(0.5) ** V1(1, 3), lambda: 0.5 ** V2(1, 3))
show('F(1,2) ** [1,3]', lambda: V1(F(1, 2)) ** V1(1, 3), lambda: F(1, 2) ** V2(1, 3))
show('0 ** [1,3]', lambda: V1(0) ** V1(1, 3), lambda: 0 ** V2(1, 3))
show('-2 ** [1,3] (as (-2)**A)', lambda: V1(-2) ** V1(1, 3), lambda: (-2) ** V2(1, 3))
show('2 ** True-int(1)', lambda: V1(1, 2) ** 1, lambda: V2(1, 2) ** int(True))

rng = random.Random(11)
agree = noisy = struct = 0
for _ in range(300):
    b = rng.choice([F(1, 4), F(1, 2), F(1), F(2), F(3), F(10)])
    lo = F(rng.randint(-6, 6), rng.choice([1, 2]))
    hi = lo + F(rng.randint(0, 6), rng.choice([1, 2]))
    lc, hc = (True, True) if lo == hi else (rng.random() < .5, rng.random() < .5)
    A1 = V1(lo) if lo == hi else V1(start=lo, end=hi, start_closed=lc, end_closed=hc)
    A2 = V2(lo) if lo == hi else V2(lo, hi, start_closed=lc, end_closed=hc)
    r1, e1 = run(lambda: V1(b) ** A1)
    r2, e2 = run(lambda: b ** A2)
    if e1 or e2 or not well_formed1(r1):
        print('ERR', b, A2, e1, e2); struct += 1; continue
    d = compare(pieces1(r1), pieces2(r2))
    ends = [e for p in pieces2(r2) for e in p[:2] if not math.isinf(e)]
    if not d:
        agree += 1
    elif all(any(abs(F(x) - F(e)) <= abs(F(e)) * F(1, 10 ** 12) for e in ends) for x, _, _ in d):
        noisy += 1
    else:
        struct += 1; print('STRUCT', b, A2, s1(r1), r2, d[:3])
print(f'rpow sweep: agree {agree}, rounding-only {noisy}, structural {struct}')

print('--- shifts')
for label, f1, f2 in [
    ('[3,5] << 1', lambda: V1(3, 5) << 1, lambda: V2(3, 5) << 1),
    ('[3,5] << 1 vs * 2**1', lambda: V1(3, 5) << 1, lambda: V2(3, 5) * 2 ** 1),
    ('[3,5] >> 1 vs // 2**1', lambda: V1(3, 5) >> 1, lambda: V2(3, 5) // 2 ** 1),
    ('(3,5) >> 1 vs // 2', lambda: V1(start=3, end=5, start_closed=False, end_closed=False) >> 1, lambda: V2(3, 5, start_closed=False, end_closed=False) // 2),
    ('{-3} >> 1 vs // 2', lambda: V1(-3) >> 1, lambda: V2(-3) // 2),
    ('{3} << -1 vs * 2**-1', lambda: V1(3) << -1, lambda: V2(3) * 2 ** -1),
    ('[3.0] << 1', lambda: V1(3.0) << 1, lambda: V2(3.0) * 2),
    ('2 << [1,3] vs 2 * 2**A', lambda: 2 << V1(1, 3), lambda: 2 * 2 ** V2(1, 3)),
    ('2 << {1,3} vs 2 * 2**A', lambda: 2 << V1(1).union(V1(3)), lambda: 2 * 2 ** (V2(1) | V2(3))),
    ('16 >> [1,3] vs 16 // 2**A', lambda: 16 >> V1(1, 3), lambda: 16 // 2 ** V2(1, 3)),
    ('16 >> {1,3} vs 16 // 2**A', lambda: 16 >> V1(1).union(V1(3)), lambda: 16 // 2 ** (V2(1) | V2(3))),
    ('[1,3] << [1,2] vs A * 2**B', lambda: V1(1, 3) << V1(1, 2), lambda: V2(1, 3) * 2 ** V2(1, 2)),
    ('[1, inf) << 1', lambda: V1(start=1, end=math.inf, end_closed=False) << 1, lambda: V2(1, math.inf, end_closed=False) * 2),
]:
    show(label, f1, f2)
for op in ('<<', '>>'):
    for expr in (f'V2(1) {op} 1', f'1 {op} V2(1)'):
        r, e = run(lambda: eval(expr))
        print(f'v2 {expr}: {e or r}')

# integer-point sweep: v1 shift on point sets vs v2 workaround vs python's int shift (the truth)
rng = random.Random(12)
cnt = {'v1 ok': 0, 'v1 bad': 0, 'v2 ok': 0, 'v2 bad': 0}
for _ in range(300):
    xs = rng.sample(range(-40, 41), rng.randint(1, 4))
    n = rng.randint(0, 5)
    op = rng.choice(['<<', '>>'])
    truth = sorted({(x << n) if op == '<<' else (x >> n) for x in xs})
    A1 = V1()
    for x in xs:
        A1 = A1.union(V1(x))
    A2 = V2.from_pieces((x, x) for x in xs)
    r1, e1 = run(lambda: (A1 << n) if op == '<<' else (A1 >> n))
    r2, e2 = run(lambda: (A2 * 2 ** n) if op == '<<' else (A2 // 2 ** n))
    want = [(t, t, True, True) for t in truth]
    cnt['v1 ok' if not e1 and pieces1(r1) == want else 'v1 bad'] += 1
    cnt['v2 ok' if not e2 and pieces2(r2) == want else 'v2 bad'] += 1
print('integer-point shift sweep', cnt)
assert [(1, 1, True, True)] != pieces2(V2(3) // 2 ** 0), 'sabotage'
print('sabotage caught')
