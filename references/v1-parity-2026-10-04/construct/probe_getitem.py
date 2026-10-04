"""MultiInterval.__getitem__: slice / MultiInterval / scalar, v1 vs v2"""
from common import *
import random
inf = math.inf
VALS = [-5, -3, -1, 0, 1, 2, 4, F(1, 2), F(-7, 3), 1.5, -0.25, 0.0]

def rand_v1():
    m = M1()
    for _ in range(random.randint(0, 4)):
        a, b = sorted(random.sample(VALS, 2))
        if a == b or random.random() < .2:
            b = a; lc = hc = True
        else:
            lc, hc = random.random() < .5, random.random() < .5
        if random.random() < .15 and a != b:
            a, lc = -inf, False
        if random.random() < .15 and a != b:
            b, hc = inf, False
        m = m.union(M1(a, b, start_closed=lc, end_closed=hc))
    return m

def brute_slice(A1, lo, hi):
    lo = -inf if lo is None else lo
    hi = inf if hi is None else hi
    return lambda x: v1_contains(A1, x) and lo <= x <= hi

random.seed(2024)
t_slice, t_v1_brute, t_v2_brute = Tally('slice v1 vs v2'), Tally('slice v1 vs brute'), Tally('slice v2 vs brute')
zero_bound_diffs, raise_diffs = 0, []
BOUNDS = [None, None, -inf, inf, 0, 0.0, F(0)] + VALS + [10, -10]
for _ in range(800):
    A1 = rand_v1(); A2 = v1_to_v2(A1)
    lo, hi = random.choice(BOUNDS), random.choice(BOUNDS)
    o1 = outcome(lambda: A1[lo:hi]); o2 = outcome(lambda: A2[lo:hi])
    if o1[0] != o2[0]:
        raise_diffs.append((str(A1), lo, hi, o1[1] if o1[0] == 'raise' else str(o1[1]), o2[1] if o2[0] == 'raise' else str(o2[1])))
        continue
    if o1[0] == 'raise':
        continue
    pts = probe_points([v for v in VALS] + [x for x in (lo, hi) if x is not None])
    r1, r2 = v1_to_v2(o1[1]), o2[1]
    same = r1 == r2
    t_slice.check(same, (str(A1), lo, hi, str(r1), str(r2)))
    has_zero_bound = (lo is not None and lo == 0) or (hi is not None and hi == 0)
    if not same and has_zero_bound:
        zero_bound_diffs += 1
    t_v1_brute.check(not same_set_by_membership(brute_slice(A1, lo, hi), lambda p: p in r1, pts), (str(A1), lo, hi, str(r1)))
    t_v2_brute.check(not same_set_by_membership(brute_slice(A1, lo, hi), lambda p: p in r2, pts), (str(A1), lo, hi, str(r2)))
t_slice.report(3); t_v1_brute.report(3); t_v2_brute.report(3)
print('slice mismatches with a 0 bound:', zero_bound_diffs, 'of', len(t_slice.bad))
print('raise/no-raise differences:', len(raise_diffs))
for r in raise_diffs[:6]:
    print('   ', r)

print('=== same sweep with no 0 bound and no reversed slice')
random.seed(77)
t2 = Tally('slice v1 vs v2 (nonzero bounds)')
NZ = [None, -inf, inf] + [v for v in VALS if v != 0] + [10, -10]
for _ in range(800):
    A1 = rand_v1(); A2 = v1_to_v2(A1)
    lo, hi = random.choice(NZ), random.choice(NZ)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    o1 = outcome(lambda: A1[lo:hi]); o2 = outcome(lambda: A2[lo:hi])
    t2.check(o1[0] == o2[0] == 'ok' and v1_to_v2(o1[1]) == o2[1], (str(A1), lo, hi, o1, o2))
t2.report(5)

print('=== edge table')
A1 = M1.merge('{ [-3, -1) , [0] , (1, 4] }'); A2 = v1_to_v2(A1)
for label, f1, f2 in [
        ('A[0:2] (0 start)', lambda: A1[0:2], lambda: A2[0:2]),
        ('A[-2:0] (0 stop)', lambda: A1[-2:0], lambda: A2[-2:0]),
        ('A[2:1] reversed', lambda: A1[2:1], lambda: A2[2:1]),
        ('A[::2] step', lambda: A1[::2], lambda: A2[::2]),
        ('A["a":2]', lambda: A1['a':2], lambda: A2['a':2]),
        ('A[:-inf]', lambda: A1[:-inf], lambda: A2[:-inf]),
        ('A[inf:]', lambda: A1[inf:], lambda: A2[inf:]),
        ('A[math.nan:2]', lambda: A1[math.nan:2], lambda: A2[math.nan:2]),
        ('A[True:3]', lambda: A1[True:3], lambda: A2[True:3]),
        ('A[M(0, 2)] (interval item)', lambda: A1[M1(0, 2)], lambda: A2[M2(0, 2)]),
        ('  v2 way: A & M(0, 2)', lambda: A1[M1(0, 2)], lambda: A2 & M2(0, 2)),
        ('A[0] (scalar member)', lambda: A1[0], lambda: A2[0]),
        ('  v2 way: A & 0', lambda: A1[0], lambda: A2 & 0),
        ('A[2] (scalar member)', lambda: A1[2], lambda: A2 & 2),
        ('A[1] (scalar, not member)', lambda: A1[1], lambda: A2 & 1),
        ('A[F(5,2)]', lambda: A1[F(5, 2)], lambda: A2 & F(5, 2)),
        ('A["x"]', lambda: A1['x'], lambda: A2['x']),
        ('A[None]', lambda: A1[None], lambda: A2[None]),
]:
    o1, o2 = outcome(f1), outcome(f2)
    print(f'  {label:30s} v1={str(o1[1]) if o1[0] == "ok" else "RAISE " + o1[1]:40s} v2={str(o2[1]) if o2[0] == "ok" else "RAISE " + o2[1]}')

print('=== random sweep: v1 A[B] / A[x] vs v2 A & B / A & x')
random.seed(11)
t3 = Tally('A[B] == A & B'); t4 = Tally('A[x] == A & x')
for _ in range(400):
    A1, B1 = rand_v1(), rand_v1()
    t3.check(v1_to_v2(A1[B1]) == (v1_to_v2(A1) & v1_to_v2(B1)), (str(A1), str(B1)))
    x = random.choice(VALS + [F(v) + F(1, 10**6) for v in VALS if not isinstance(v, float)])
    t4.check(v1_to_v2(A1[x]) == (v1_to_v2(A1) & x), (str(A1), x, str(A1[x]), str(v1_to_v2(A1) & x)))
t3.report(); t4.report()

print('=== sabotage: wrong expectation (open restriction) must be caught')
random.seed(3)
s = Tally('sabotage')
for _ in range(100):
    A1 = rand_v1(); A2 = v1_to_v2(A1)
    s.check(v1_to_v2(A1[-1:2]) == (A2 & M2(-1, 2, start_closed=False, end_closed=False)), '')
assert s.bad, 'sabotage not caught'
print('sabotage caught', len(s.bad))
