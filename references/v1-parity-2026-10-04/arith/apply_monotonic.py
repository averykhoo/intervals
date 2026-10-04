"""
v1 apply_monotonic_unary_function / apply_monotonic_binary_function vs the two v2 routes:
 (a) internal: intervals.applicator.apply_unary/apply_binary with a user OpDescriptor (not exported)
 (b) public composition over A.pieces (inf, sup, inf_closed, sup_closed) and MultiInterval.from_pieces
"""
import sys, os, random, warnings, math, collections, itertools
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from intervals import applicator
from intervals.applicator import OpDescriptor
warnings.simplefilter('ignore')

def v2_internal_unary(A, f):
    return V2.from_cuts(applicator.apply_unary(OpDescriptor('user', f), A.cuts))

def v2_internal_binary(A, B, f):
    return V2.from_cuts(applicator.apply_binary(OpDescriptor('user', f), A.cuts, B.cuts))

def v2_public_unary(A, f):
    """public API only: each piece's ends through f, a closed end stays closed (strictly monotone f)"""
    out = []
    for p in A.pieces:
        a, b = (f(p.inf), p.inf_closed), (f(p.sup), p.sup_closed)
        (lo, lc), (hi, hc) = sorted([a, b], key=lambda t: t[0])
        out.append((lo, hi, lc, hc) if lo != hi else (lo, lo, True, True))
    return V2.from_pieces(out)

def v2_public_binary(A, B, f):
    out = []
    for p in A.pieces:
        for q in B.pieces:
            corners = [(f(x, y), xc and yc) for (x, xc) in {(p.inf, p.inf_closed), (p.sup, p.sup_closed)}
                       for (y, yc) in {(q.inf, q.inf_closed), (q.sup, q.sup_closed)}]
            lo = min(v for v, _ in corners); hi = max(v for v, _ in corners)
            lc = any(c for v, c in corners if v == lo); hc = any(c for v, c in corners if v == hi)
            out.append((lo, hi, lc, hc) if lo != hi else (lo, lo, True, True))
    return V2.from_pieces(out)

UNARY = {
    '3x+1 (exact, increasing)': lambda x: 3 * x + 1,
    '-x/2 (exact, decreasing)': lambda x: -x / 2,
    'x**3 (exact, increasing)': lambda x: x ** 3,
    'atan (float)': lambda x: math.atan(x),
    '-atan (float, decreasing)': lambda x: -math.atan(x),
}
BINARY = {
    'x+2y': lambda x, y: x + 2 * y,
    'x-3y': lambda x, y: x - 3 * y,
    '2x-y/2': lambda x, y: 2 * x - y / 2,
    'atan(x)+atan(y) (float)': lambda x, y: math.atan(x) + math.atan(y),
}
FINITE_ONLY = {'x**3 (exact, increasing)'}

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1788)
stats = collections.Counter(); ex = collections.defaultdict(list)
for i in range(300):
    pa = rand_pieces(rng); pb = rand_pieces(rng)
    a1, a2, b1, b2 = mk1(pa), mk2(pa), mk1(pb), mk2(pb)
    for name, f in UNARY.items():
        if name in FINITE_ONLY and any(math.isinf(e) for p in pa for e in p[:2]):
            continue
        r1 = a1.apply_monotonic_unary_function(f)
        for route, g in (('internal', v2_internal_unary), ('public', v2_public_unary)):
            try:
                r2 = g(a2, f)
            except Exception as e:
                stats[f'unary {route} {name}: v2 raises'] += 1; ex[f'unary {route} raises'].append(f'{fmt_pieces(pa)}: {type(e).__name__} {e}'); continue
            d = finite_diff(r1, r2) + inf_diff(r1, r2)
            k = f'unary {route} {name}: ' + ('agree' if not d else 'DIFFER')
            stats[k] += 1
            if d: ex[k].append(f'{fmt_pieces(pa)}: v1 {r1} v2 {r2} {d[:3]}')
    for name, f in BINARY.items():
        try:
            r1 = a1.apply_monotonic_binary_function(f, b1); r1._consistency_check()
        except Exception as e:
            stats[f'binary {name}: v1 raises {type(e).__name__}'] += 1; continue
        for route, g in (('internal', v2_internal_binary), ('public', v2_public_binary)):
            try:
                r2 = g(a2, b2, f)
            except Exception as e:
                stats[f'binary {route} {name}: v2 raises'] += 1; ex[f'binary {route} raises'].append(f'{fmt_pieces(pa)} , {fmt_pieces(pb)}: {type(e).__name__} {e}'); continue
            d = finite_diff(r1, r2) + inf_diff(r1, r2)
            k = f'binary {route} {name}: ' + ('agree' if not d else 'DIFFER')
            stats[k] += 1
            if d: ex[k].append(f'{fmt_pieces(pa)} , {fmt_pieces(pb)}: v1 {r1} v2 {r2} {d[:3]}')
# right_hand_side= and scalar other in v1
A1, A2 = V1(F(1), F(2), end_closed=False), V2(F(1), F(2), end_closed=False)
f = lambda x, y: x - 3 * y
print('rhs: v1', A1.apply_monotonic_binary_function(f, 5, right_hand_side=True), ' v2 internal', v2_internal_binary(V2(5), A2, f), ' public', v2_public_binary(V2(5), A2, f))
print('scalar other: v1', A1.apply_monotonic_binary_function(f, 5), ' v2', v2_internal_binary(A2, V2(5), f))
for label, call in [('v1 unary on empty', lambda: V1().apply_monotonic_unary_function(math.atan)),
                    ('v1 binary empty first', lambda: V1().apply_monotonic_binary_function(f, A1)),
                    ('v1 binary empty second', lambda: A1.apply_monotonic_binary_function(f, V1())),
                    ('v2 internal empty second', lambda: v2_internal_binary(A2, V2(), f)),
                    ('v1 binary other=str', lambda: A1.apply_monotonic_binary_function(f, '1')),
                    ('v1 inplace=True returns self', lambda: (lambda m: (m.apply_monotonic_unary_function(math.atan, inplace=True) is m, m))(V1(0, 1))),
                    ('v1 flat fn max(x,0) on (-2,-1)', lambda: V1(-2, -1, start_closed=False, end_closed=False).apply_monotonic_unary_function(lambda x: max(x, 0))),
                    ('v2 internal flat fn max(x,0) on (-2,-1)', lambda: v2_internal_unary(V2(-2, -1, start_closed=False, end_closed=False), lambda x: max(x, 0))),
                    ('v1 min(x,1) on [0,2) (true [0,1])', lambda: V1(0, 2, end_closed=False).apply_monotonic_unary_function(lambda x: min(x, 1))),
                    ('v2 internal min(x,1) on [0,2)', lambda: v2_internal_unary(V2(0, 2, end_closed=False), lambda x: min(x, 1))),
                    ('v1 x+max(y,2) on [0,1]x(2,3) (true (2,4))', lambda: V1(0, 1).apply_monotonic_binary_function(lambda x, y: x + max(y, 2), V1(2, 3, start_closed=False, end_closed=False))),
                    ('v2 internal x+max(y,2) on [0,1]x(2,3)', lambda: v2_internal_binary(V2(0, 1), V2(2, 3, start_closed=False, end_closed=False), lambda x, y: x + max(y, 2))),
                    ('v2 public x+max(y,2) on [0,1]x(2,3)', lambda: v2_public_binary(V2(0, 1), V2(2, 3, start_closed=False, end_closed=False), lambda x, y: x + max(y, 2))),
                    ]:
    try:
        print(f'{label}: {call()}')
    except Exception as e:
        print(f'{label}: RAISES {type(e).__name__}: {e}')
for k in sorted(stats):
    print(f'{stats[k]:5d}  {k}')
for k in sorted(ex):
    print('==', k)
    for e in ex[k][:4]:
        print('   ', e)
# self-check: a wrong function on the v2 side must be caught
r1 = V1(0, 1).apply_monotonic_unary_function(lambda x: 3 * x + 1); r2 = v2_public_unary(V2(0, 1), lambda x: 3 * x + 2)
print('self-check (wrong fn caught):', bool(finite_diff(r1, r2)))
