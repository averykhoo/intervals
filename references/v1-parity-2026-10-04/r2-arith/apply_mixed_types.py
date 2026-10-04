"""v1 apply_monotonic_unary/binary_function with a USER function that mixes Fraction and float in one call, vs v2's two
routes: (a) internal applicator.apply_unary/apply_binary + OpDescriptor (not exported), (b) the public composition over
A.pieces / from_pieces (same as r1 arith/apply_monotonic.py). inputs mix int, Fraction and float ends, open/closed, +-inf.
compared: exact end values (==, Fraction vs float compare exactly), end flags, and membership at test points."""
import sys, os, random, collections, math
sys.path.insert(0, os.path.dirname(__file__))
from oracle import *
from intervals import applicator
from intervals.applicator import OpDescriptor
warnings.simplefilter('ignore')
SAB = len(sys.argv) > 2

def v2_int_u(A, f): return V2.from_cuts(applicator.apply_unary(OpDescriptor('user', f), A.cuts))
def v2_int_b(A, B, f): return V2.from_cuts(applicator.apply_binary(OpDescriptor('user', f), A.cuts, B.cuts))
def v2_pub_u(A, f):
    out = []
    for p in A.pieces:
        (lo, lc), (hi, hc) = sorted([(f(p.inf), p.inf_closed), (f(p.sup), p.sup_closed)], key=lambda t: t[0])
        out.append((lo, hi, lc, hc) if lo != hi else (lo, lo, True, True))
    return V2.from_pieces(out)
def v2_pub_b(A, B, f):
    out = []
    for p in A.pieces:
        for q in B.pieces:
            cs = [(f(x, y), xc and yc) for (x, xc) in {(p.inf, p.inf_closed), (p.sup, p.sup_closed)}
                  for (y, yc) in {(q.inf, q.inf_closed), (q.sup, q.sup_closed)}]
            lo = min(v for v, _ in cs); hi = max(v for v, _ in cs)
            lc = any(c for v, c in cs if v == lo); hc = any(c for v, c in cs if v == hi)
            out.append((lo, hi, lc, hc) if lo != hi else (lo, lo, True, True))
    return V2.from_pieces(out)

POOL = [-INF, -2.5, F(-1, 3), -1, 0, 0.1, F(1, 2), 1, 2.75, F(7, 3), 1e-300, INF]
def rand_pieces(rng, allow_inf):
    pool = POOL if allow_inf else [v for v in POOL if not (isinstance(v, float) and math.isinf(v))]
    out = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, b = sorted(rng.sample(pool, 2))
        if rng.random() < .15 and not math.isinf(a): out.append((a, a, True, True)); continue
        out.append((a, b, rng.random() < .5 and not math.isinf(a), rng.random() < .5 and not math.isinf(b)))
    return out

UNARY = {   # each mixes a Fraction and a float in one evaluation (or returns either type by input)
    'F(1,3)*x + 0.25': (lambda x: F(1, 3) * x + 0.25, True),
    '-(x*0.5) + F(2,7)': (lambda x: -(x * 0.5) + F(2, 7), True),
    'x/3 + F(1,7) (Fraction for exact x, float for float x)': (lambda x: x / 3 + F(1, 7), True),
    'F(x) + 0.1 (F() of the end, then a float)': (lambda x: F(x) + 0.1, False),
    'x**3 * F(1,2) - 1e-3': (lambda x: x ** 3 * F(1, 2) - 1e-3, False),
    'atan(x) + F(1,3)': (lambda x: math.atan(x) + F(1, 3), True),
    'x if Fraction else x*1.0 + F(0) (type flips by input)': (lambda x: F(x) * 2 if isinstance(x, (int, F)) else x * 2.0, False),
}
BINARY = {
    'x + 0.5*y': (lambda x, y: x + 0.5 * y, True),
    'F(1,3)*x - y*0.25': (lambda x, y: F(1, 3) * x - y * 0.25, True),
    'x - F(2,3)*y + 0.1': (lambda x, y: x - F(2, 3) * y + 0.1, True),
    'F(x) + F(y)/3 + 1e-9': (lambda x, y: F(x) + F(y) / 3 + 1e-9, False),
    'atan(x) + F(1,2)*y': (lambda x, y: math.atan(x) + F(1, 2) * y, True),
}

def compare(r1, r2):
    s1 = [(a, eps) for a, eps in r1.endpoints]
    s2 = []
    for p in r2.pieces:
        s2 += [(p.inf, 0 if p.inf_closed else 1), (p.sup, 0 if p.sup_closed else -1)]
    struct_eq = len(s1) == len(s2) and all(a == b and e == g for (a, e), (b, g) in zip(s1, s2))
    pts = test_points([a for a, _ in s1], [a for a, _ in s2])
    mem_d = [z for z in pts if v1_mem(r1, z) != (z in r2)]
    return struct_eq, mem_d

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
st = collections.Counter(); ex = collections.defaultdict(list)
types_seen = collections.Counter()
for i in range(250):
    for name, (f, inf_ok) in UNARY.items():
        pa = rand_pieces(rng, inf_ok)
        a1, a2 = mk1(pa), mk2(pa)
        try: r1 = a1.apply_monotonic_unary_function(f); r1._consistency_check(); e1 = None
        except Exception as e: r1, e1 = None, f'{type(e).__name__}: {e}'
        for route, g in (('internal', v2_int_u), ('public', v2_pub_u)):
            try: r2 = g(a2, f); e2 = None
            except Exception as e: r2, e2 = None, f'{type(e).__name__}: {e}'
            if SAB and r2 is not None and i == 3: r2 = r2 | V2(F(1, 10**7))
            if e1 or e2:
                k = f'unary {route}: v1 raises={bool(e1)} v2 raises={bool(e2)}'; st[k] += 1; ex[k].append(f'{name} on {fmt(pa)}: v1 {e1 or r1} v2 {e2 or r2}'); continue
            for p in r2.pieces: types_seen[(type(p.inf).__name__, type(p.sup).__name__)] += 1
            seq, md = compare(r1, r2)
            k = f'unary {route}: ' + ('same structure' if seq else 'same set, other structure' if not md else 'DIFFERENT SET')
            st[k] += 1
            if not seq: ex[k].append(f'{name} on {fmt(pa)}: v1 {r1.endpoints} v2 {r2!r} diff@{[str(z) for z in md[:3]]}')
    for name, (f, inf_ok) in BINARY.items():
        pa, pb = rand_pieces(rng, inf_ok), rand_pieces(rng, inf_ok)
        a1, a2, b1, b2 = mk1(pa), mk2(pa), mk1(pb), mk2(pb)
        try: r1 = a1.apply_monotonic_binary_function(f, b1); r1._consistency_check(); e1 = None
        except Exception as e: r1, e1 = None, f'{type(e).__name__}: {e}'
        for route, g in (('internal', v2_int_b), ('public', v2_pub_b)):
            try: r2 = g(a2, b2, f); e2 = None
            except Exception as e: r2, e2 = None, f'{type(e).__name__}: {e}'
            if e1 or e2:
                k = f'binary {route}: v1 raises={bool(e1)} v2 raises={bool(e2)}'; st[k] += 1; ex[k].append(f'{name} on {fmt(pa)} x {fmt(pb)}: v1 {e1 or r1} v2 {e2 or r2}'); continue
            seq, md = compare(r1, r2)
            k = f'binary {route}: ' + ('same structure' if seq else 'same set, other structure' if not md else 'DIFFERENT SET')
            st[k] += 1
            if not seq: ex[k].append(f'{name} on {fmt(pa)} x {fmt(pb)}: v1 {r1.endpoints} v2 {r2!r} diff@{[str(z) for z in md[:3]]}')
for k in sorted(st): print(f'{st[k]:5d}  {k}')
print('v2 end types (internal+public, unary):', dict(types_seen))
for k in sorted(ex):
    print('==', k)
    for e in ex[k][:5]: print('   ', e[:400])
if SAB: assert any('DIFFERENT' in k for k in st), 'sabotage missed'; print('SABOTAGE CAUGHT')
