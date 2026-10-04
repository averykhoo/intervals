"""float rounding of + - * / on float-ended sets, A op scalar and scalar op A, against an EXACT oracle: corners in
Fraction, extreme + attained-ness (a corner with both ends attained, or a closed 0 factor), rounded once to nearest
(float(Fraction) is correctly rounded). both v1 and v2 are scored against the oracle. `sab`: oracle nudges one end."""
import sys, math, random, warnings, operator, itertools
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m
import intervals as v2
SAB = len(sys.argv) > 1
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
def rfloat(rng):
    k = rng.random()
    if k < .3: return round(rng.uniform(-5, 5), 1)
    if k < .5: return rng.uniform(-1e3, 1e3)
    if k < .6: return rng.choice([0.0, 1.0, -1.0, 0.5, 3.0, 0.1])
    if k < .7: return rng.uniform(-1, 1) * 1e-200
    return rng.uniform(-10, 10)
def rpiece(rng, nozero=False):
    while True:
        a, b = sorted((rfloat(rng), rfloat(rng)))
        if rng.random() < .1: b = a
        lc, hc = (True, True) if a == b else (rng.random() < .5, rng.random() < .5)
        if nozero and ((a < 0 < b) or (a == 0 and lc) or (b == 0 and hc) or a == b == 0): continue
        if nozero and a == 0 and b == 0: continue
        return (a, b, lc, hc)
def oracle(op, A, B):
    (a, b, ac, bc), (c, d, cc, dc) = A, B
    vals = []
    for (x, xa), (y, ya) in itertools.product([(a, ac), (b, bc)], [(c, cc), (d, dc)]):
        if op == '/' and y == 0: return None          # open zero end of the divisor: skip (pole)
        vals.append((OPS[op](F(x), F(y)), xa and ya))
    lo = min(v for v, _ in vals); hi = max(v for v, _ in vals)
    def att(e):
        if any(v == e and t for v, t in vals): return True
        if e == 0 and op in '*/' and ((a == 0 and ac) or (b == 0 and bc)): return True
        if e == 0 and op == '*' and ((c == 0 and cc) or (d == 0 and dc)): return True
        return False
    MAX = F(sys.float_info.max)
    if abs(lo) > MAX or abs(hi) > MAX: return None
    flo, fhi = float(lo), float(hi)
    if flo == fhi and lo != hi: return None                # collapses on rounding: skip the closedness rule there
    allx = all(F(float(v)) == v for v, _ in vals)       # every corner a double: no rounding anywhere
    exact = (allx, allx)
    return ([(flo, att(lo), fhi, att(hi))] if lo != hi else [(flo, True, fhi, True)]), exact
def st1(m):
    e = m.endpoints; return [(float(e[i][0]), e[i][1] == 0, float(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]
def st2(m): return [(float(p.inf), bool(p.inf_closed), float(p.sup), bool(p.sup_closed)) for p in m]
def mk1(p): a, b, lc, hc = p; return v1m.MultiInterval(a) if a == b else v1m.MultiInterval(a, b, start_closed=lc, end_closed=hc)
def mk2(p): return v2.MultiInterval.from_pieces([p])
rng = random.Random(1788)
forms = ['A op B', 'A op x', 'x op A']
score = {(o, f): [0, 0, 0, 0, 0, 0] for o in OPS for f in forms}   # n, v1 ok, v2 ok, v1 end-value wrong, v1 closedness wrong
ex = {}
for i in range(3000):
    for o in OPS:
        for f in forms:
            A = rpiece(rng, nozero=(o == '/' and f == 'x op A'))
            B = rpiece(rng, nozero=(o == '/'))
            if f == 'A op B': L, R, l1, r1, l2, r2 = A, B, mk1(A), mk1(B), mk2(A), mk2(B)
            elif f == 'A op x': x = B[0]; L, R, l1, r1, l2, r2 = A, (x, x, True, True), mk1(A), x, mk2(A), x
            else:
                x = B[0]
                if o == '/' and x == 0: x = 1.5
                L, R, l1, r1, l2, r2 = (x, x, True, True), A, x, mk1(A), x, mk2(A)
            if o == '/' and f == 'A op x' and x == 0: continue
            want = oracle(o, L, R)
            if want is None: continue
            want, exact = want
            if SAB and i == 7: want = [(want[0][0], want[0][1], math.nextafter(want[0][2], math.inf), want[0][3])]
            try: g1 = st1(OPS[o](l1, r1))
            except Exception as e: g1 = repr(e)
            g2 = st2(OPS[o](l2, r2))
            def ok(g):   # end values always; a flag only where its end is exact (an inexact end's flag is the
                         # float image's, not the exact set's: checked v1 against v2 instead, column 'flag v1!=v2')
                if isinstance(g, str) or len(g) != 1: return False, False
                vals = (g[0][0], g[0][2]) == (want[0][0], want[0][2])     # -0.0 == 0.0: a set of reals has no signed zero
                flags = all(g[0][k] == want[0][k] for k, e in ((1, exact[0]), (3, exact[1])) if e)
                return vals, flags
            s = score[(o, f)]; s[0] += 1
            v1v, v1f = ok(g1); v2v, v2f = ok(g2)
            s[1] += v1v and v1f; s[2] += v2v and v2f; s[3] += not v1v; s[4] += v1v and not v1f
            s[5] += (not isinstance(g1, str)) and len(g1) == 1 and len(g2) == 1 and (g1[0][1], g1[0][3]) != (g2[0][1], g2[0][3])
            if not v1v: ex.setdefault((o, f, 'v1 value'), (L, R, g1, want))
            if v1v and not v1f: ex.setdefault((o, f, 'v1 flag'), (L, R, g1, want))
            if not (v2v and v2f): ex.setdefault((o, f, 'v2'), (L, R, g2, want))
for k, s in score.items(): print(f'{k[0]} {k[1]:7s} n {s[0]:5d}  v1 ok {s[1]:5d}  v2 ok {s[2]:5d}   v1 wrong end value {s[3]:4d}  v1 wrong exact-end flag {s[4]:3d}  flag v1!=v2 {s[5]:3d}')
for k, v in list(ex.items())[:12]: print('  ex', k, v)
assert all(s[2] == s[0] for s in score.values()), 'v2 disagrees with the exact oracle'
