"""merge_adjacent(distance), abs(), invert(), mirror(), expand(distance, inplace)"""
import sys; sys.path.insert(0, '.scratch/v1-parity/setops')
from common import *
import random

OPEN_REALS = M2(-inf, inf, start_closed=False, end_closed=False)

def v2_close_gaps(A, d):
    """v1 merge_adjacent(distance=d) on a normalized set: fill every gap g with wid < d, or wid == d unless g is closed at both ends"""
    gaps = A.hull.difference(A)
    fill = [g for g in gaps if (g.sup - g.inf) < d or ((g.sup - g.inf) == d and not (g.inf_closed and g.sup_closed))]
    return A.union(*fill)

def brute_abs(A, p):  # p in |A| iff p >= 0 and (p in A or -p in A)
    return p >= 0 and ((p in A) or (-p in A))

DIST = [0, 0.5, 1, Fraction(1, 2), Fraction(3, 4), 2, 0.25, 3, inf]
rng = random.Random(31337)
T = {k: Tally(k) for k in ['merge_adjacent(d) vs gap-fill composition', 'merge_adjacent(0) no-op', 'abs() inplace', '__abs__', 'abs brute',
                           'invert() vs ~A & (-inf,inf)', 'invert() vs ~A (raw)', '__invert__ vs composition', 'mirror()', '__neg__',
                           'expand(d)', 'expand(d, inplace=True)', 'expand brute', 'abs mutates self in v1']}
for _ in range(500):
    pa = rand_pieces(rng); a1, a2 = twin(pa)
    d = rng.choice(DIST)
    r1 = a1.copy().merge_adjacent(distance=d)
    T['merge_adjacent(d) vs gap-fill composition'].check((pa, d), same_set_v1_v2(r1, v2_close_gaps(a2, d)))
    T['merge_adjacent(0) no-op'].check(pa, same_set_v1_v2(a1.copy().merge_adjacent(), a2))
    x1 = a1.copy(); r1 = x1.abs()
    T['abs() inplace'].check(pa, same_set_v1_v2(r1, abs(a2)))
    T['abs mutates self in v1'].check(pa, None if r1 is x1 else 'not self')
    T['__abs__'].check(pa, same_set_v1_v2(abs(a1), abs(a2)))
    T['abs brute'].check(pa, None if all(((p in abs(a2)) == brute_abs(a2, p)) for p in points_for(a2, abs(a2)) if not (isinstance(p, float) and math.isinf(p))) else 'brute')
    T['invert() vs ~A & (-inf,inf)'].check(pa, same_set_v1_v2(a1.copy().invert(), ~a2 & OPEN_REALS))
    T['invert() vs ~A (raw)'].check(pa, same_set_v1_v2(a1.copy().invert(), ~a2))
    T['__invert__ vs composition'].check(pa, same_set_v1_v2(~a1, a2.complement() & OPEN_REALS))
    T['mirror()'].check(pa, same_set_v1_v2(a1.copy().mirror(), -a2))
    T['__neg__'].check(pa, same_set_v1_v2(-a1, -a2))
    e = rng.choice(DIST[:-1])
    T['expand(d)'].check((pa, e), same_set_v1_v2(a1.expand(e), a2.expand(e)))
    T['expand(d, inplace=True)'].check((pa, e), same_set_v1_v2(a1.copy().expand(e, inplace=True), a2.expand(e)))
    # brute: p in expand(A, e) iff some q in A with |p - q| <= e  -> v2 expand equals A + [-e, e]
    T['expand brute'].check((pa, e), None if a2.expand(e) == a2 + M2(-e, e) else f'{a2.expand(e)} vs {a2 + M2(-e, e)}')
for t in T.values(): t.report(3)

def run(label, f1, f2):
    try: r1 = str(f1())
    except Exception as ex: r1 = f'raise {type(ex).__name__}: {str(ex)[:30]}'
    try: r2 = str(f2())
    except Exception as ex: r2 = f'raise {type(ex).__name__}: {str(ex)[:30]}'
    print(f'{label:50s} v1={r1:40.40s} v2={r2:40.40s}{"" if r1 == r2 else "   <-- DIFF"}')
G1 = M1(0, 1).union(M1(2, 3)); G2 = M2(0, 1) | M2(2, 3)
run('[0,1]u[2,3].merge_adjacent(1)', lambda: G1.copy().merge_adjacent(1), lambda: v2_close_gaps(G2, 1))
run('[0,1)u(2,3].merge_adjacent(1) (gap [1,2])', lambda: M1(0, 1, end_closed=False).union(M1(2, 3, start_closed=False)).merge_adjacent(1),
    lambda: v2_close_gaps(M2(0, 1, end_closed=False) | M2(2, 3, start_closed=False), 1))
run('merge_adjacent(-1)', lambda: G1.copy().merge_adjacent(-1), lambda: 'n/a')
run('merge_adjacent("1")', lambda: G1.copy().merge_adjacent('1'), lambda: 'n/a')
run('merge_adjacent(inf)', lambda: G1.copy().merge_adjacent(inf), lambda: v2_close_gaps(G2, inf))
run('merge_adjacent(1, sort=False)', lambda: G1.copy().merge_adjacent(1, sort=False), lambda: v2_close_gaps(G2, 1))
run('invert of empty', lambda: ~M1(), lambda: ~M2() & OPEN_REALS)
run('invert of empty, raw ~', lambda: ~M1(), lambda: ~M2())
run('invert of (-inf,inf)', lambda: ~M1(-inf, inf, start_closed=False, end_closed=False), lambda: ~M2(-inf, inf, start_closed=False, end_closed=False))
run('invert of [0,1]', lambda: ~M1(0, 1), lambda: ~M2(0, 1))
run('invert twice of [0,1] (involution)', lambda: ~~M1(0, 1), lambda: ~~M2(0, 1))
run('abs of (-1,0)', lambda: abs(M1(-1, 0, start_closed=False, end_closed=False)), lambda: abs(M2(-1, 0, start_closed=False, end_closed=False)))
run('abs of [-2,-1]u(0.5,1]', lambda: abs(M1(-2, -1).union(M1(0.5, 1, start_closed=False))), lambda: abs(M2(-2, -1) | M2(0.5, 1, start_closed=False)))
run('abs of (-inf,-1)', lambda: abs(M1(-inf, -1, start_closed=False, end_closed=False)), lambda: abs(M2(-inf, -1, start_closed=False, end_closed=False)))
run('abs of empty', lambda: abs(M1()), lambda: abs(M2()))
run('mirror of [0.0, 1]', lambda: M1(0.0, 1).mirror(), lambda: -M2(0.0, 1))
run('0.0 in mirror([0.0,1]) / -0.0', lambda: (0.0 in M1(0.0, 1).mirror(), -0.0 in M1(0.0, 1).mirror()), lambda: (0.0 in -M2(0.0, 1), -0.0 in -M2(0.0, 1)))
run('mirror of (-inf, 0]', lambda: M1(-inf, 0, start_closed=False).mirror(), lambda: -M2(-inf, 0, start_closed=False))
run('expand(0.5) [0,1)u(1,2]', lambda: M1(0, 1, end_closed=False).union(M1(1, 2, start_closed=False)).expand(0.5), lambda: (M2(0, 1, end_closed=False) | M2(1, 2, start_closed=False)).expand(0.5))
run('expand(0) [0,1)u(1,2]', lambda: M1(0, 1, end_closed=False).union(M1(1, 2, start_closed=False)).expand(0), lambda: (M2(0, 1, end_closed=False) | M2(1, 2, start_closed=False)).expand(0))
run('expand(1/3) [0]', lambda: M1(0).expand(Fraction(1, 3)), lambda: M2(0).expand(Fraction(1, 3)))
run('expand(0.1) [0] float', lambda: M1(0).expand(0.1), lambda: M2(0).expand(0.1))
run('expand(1) (-inf,0)', lambda: M1(-inf, 0, start_closed=False, end_closed=False).expand(1), lambda: M2(-inf, 0, start_closed=False, end_closed=False).expand(1))
run('expand(inf) [0,1]', lambda: M1(0, 1).expand(inf), lambda: M2(0, 1).expand(inf))
run('  ...v1 expand(inf) membership of -inf', lambda: -inf in M1(0, 1).expand(inf), lambda: 'n/a')
run('expand(-1) [0,3]', lambda: M1(0, 3).expand(-1), lambda: M2(0, 3).expand(-1))
run('expand(True)', lambda: M1(0, 1).expand(True), lambda: M2(0, 1).expand(True))
run('expand("1")', lambda: M1(0, 1).expand('1'), lambda: M2(0, 1).expand('1'))
run('expand(nan)', lambda: M1(0, 1).expand(float('nan')), lambda: M2(0, 1).expand(float('nan')))
run('expand(1) of empty', lambda: M1().expand(1), lambda: M2().expand(1))
x = M1(0, 1); x.expand(1); run('v1 expand() default leaves self', lambda: x, lambda: 'n/a')
x = M1(0, 1); x.expand(1, inplace=True); run('v1 expand(inplace=True) mutates', lambda: x, lambda: M2(0, 1).expand(1))
s = Tally('sabotage'); s.check('x', same_set_v1_v2(~M1(0, 1), ~M2(0, 1))); assert s.report(0) == 1
