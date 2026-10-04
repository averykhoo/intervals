"""v1 with INFINITY_IS_NOT_FINITE = False (closed ends at +-inf allowed) vs v2 (always allowed)"""
from common import *
import random
inf = math.inf
v1.INFINITY_IS_NOT_FINITE = False
print('flag now', v1.INFINITY_IS_NOT_FINITE)

random.seed(31)
pool = [-3, 0, 1, F(1, 3), 2.5, inf, -inf]
t_init, outcomes = Tally('init, flag off: v1 vs v2'), {}
for _ in range(500):
    a, b = random.choice(pool), random.choice(pool)
    sc, ec = random.random() < .5, random.random() < .5
    args = (a,) if random.random() < .2 else (a, b)
    kw = dict(start_closed=sc, end_closed=ec)
    o1, o2 = outcome(lambda: M1(*args, **kw)), outcome(lambda: M2(*args, **kw))
    if o1[0] == o2[0] == 'ok':
        pts = probe_points([x for x in args])
        t_init.check(v1_to_v2(o1[1]) == o2[1] and not same_set_by_membership(lambda p: v1_contains(o1[1], p), lambda p: p in o2[1], pts), (args, kw, str(o1[1]), str(o2[1])))
    elif o1[0] != o2[0]:
        k = f'v1 {o1[0]} / v2 {o2[0]}'
        outcomes.setdefault(k, []).append((args, kw, o1[1] if o1[0] == 'raise' else str(o1[1]), o2[1] if o2[0] == 'raise' else str(o2[1])))
t_init.report()
for k, v in outcomes.items():
    print(' ', k, len(v), 'e.g.', v[:3])

print('=== union / merge / str / slice / membership of closed-inf sets, flag off')
cases = [((1, inf), {}), ((-inf, 0), {}), ((inf,), {}), ((-inf,), {}), ((-inf, inf), {})]
for args, kw in cases:
    a1, a2 = M1(*args, **kw), M2(*args, **kw)
    print(f'  {str(args):14s} v1 str={str(a1):12s} v2 str={str(a2):12s} inf in: v1 {inf in a1} v2 {inf in a2} | -inf in: v1 {-inf in a1} v2 {-inf in a2}'
          f' | A[0:] v1 {str(a1[0.5:])} v2 {str(a2[0.5:])}')
u1 = M1(1, inf).union(M1(-inf, 0)); u2 = M2(1, inf) | M2(-inf, 0)
print('  union [1,inf] | [-inf,0]: v1', u1, 'v2', u2, 'same', v1_to_v2(u1) == u2)
print('  merge("[1, inf]") v1:', outcome(lambda: str(M1.merge('[1, inf]'))), ' v2 parse:', M2.parse('[1, inf]'))
print('  merge([1, inf]) list v1:', outcome(lambda: str(M1.merge([1, inf]))))
print('  v1 random_multi_interval with flag off, seeded:')
random.seed(5)
for _ in range(3):
    r = v1.random_multi_interval(-10, 10, 3, 0)
    print('     ', r, '-> v2 reads its str:', M2.parse(str(r)) == v1_to_v2(r))
v1.INFINITY_IS_NOT_FINITE = True
print('=== sabotage'); s = Tally('sab'); s.check((inf in M2(1, inf, end_closed=False)) == True, 'open at inf holds inf?'); assert s.report() == 1
