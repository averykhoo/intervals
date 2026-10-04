"""exact membership oracle for A op B over the affine extended reals (Fractions; +-inf as floats), and shared helpers.
semantics: z in A op B iff z = x op y for some x in A, y in B with x op y DEFINED (inf-inf, 0*inf, inf/inf, x/0 undefined).
pole=True adds v2's documented convention: x / 0 for x != 0 attains sign(x)*side*inf when 0 is a closed member of a piece
of B extending to `side` of 0 (README.md "a pole at a closed zero attains the infinity of its piece's sign")."""
import math, sys, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
V1, V2 = v1.MultiInterval, v2.MultiInterval
INF = math.inf

def fx(x):
    return x if isinstance(x, float) and math.isinf(x) else F(x)

def pieces2(m):
    return [(fx(p.inf), fx(p.sup), bool(p.inf_closed), bool(p.sup_closed)) for p in m.pieces]

def mem_p(p, x):
    lo, hi, lc, hc = p
    return lo < x < hi or (x == lo and lc) or (x == hi and hc)

def mem(ps, x):
    return any(mem_p(p, x) for p in ps)

def v1_mem(m, x):
    e = m.endpoints
    return any(e[i] <= (x, 0) <= e[i + 1] for i in range(0, len(e), 2))

def nonempty(I):
    lo, hi, lc, hc = I
    return lo < hi or (lo == hi and lc and hc)

def inter(I, J):
    a, b, ac, bc = I; c, d, cc, dc = J
    if a > c: lo, lc = a, ac
    elif c > a: lo, lc = c, cc
    else: lo, lc = a, ac and cc
    if b < d: hi, hc = b, bc
    elif d < b: hi, hc = d, dc
    else: hi, hc = b, bc and dc
    return (lo, hi, lc, hc)

def fin(p):
    lo, hi, lc, hc = p
    return (lo, hi, lc and not math.isinf(lo), hc and not math.isinf(hi))

POS = (F(0), INF, False, False)
NEG = (-INF, F(0), False, False)

def has(ps, pred_I):
    return any(nonempty(inter(p, pred_I)) for p in ps)

def has_finite(ps):
    return any(nonempty(fin(p)) for p in ps)

def has_pos(ps):  # some member > 0 (incl. +inf)
    return any(nonempty(inter(p, (F(0), INF, False, True))) for p in ps)

def has_neg(ps):
    return any(nonempty(inter(p, (-INF, F(0), True, False))) for p in ps)

def has_finite_pos(ps): return has(ps, POS)
def has_finite_neg(ps): return has(ps, NEG)

def has_other_than(ps, v):
    return any(nonempty(p) and not (p[0] == p[1] == v) for p in ps)

def sgn(x): return (x > 0) - (x < 0)

def recip_image(sub, z):
    """image of a sub-interval of (0,inf) or (-inf,0) (finite part, 0 excluded) under t -> z/t, z != 0 finite"""
    a, b, ac, bc = sub
    def f(t):
        if t == 0: return INF * sgn(z) * (1 if b > 0 else -1)  # limit from the sub's side
        if math.isinf(t): return F(0)
        return z / t
    fa, fb = f(a), f(b)
    ac = ac and a != 0 and not math.isinf(a); bc = bc and b != 0 and not math.isinf(b)
    if z > 0: return (fb, fa, bc, ac)
    return (fa, fb, ac, bc)

def lin_image(I, z):
    a, b, ac, bc = I
    m = lambda t: t * z if not math.isinf(t) else t * sgn(z)
    if z > 0: return (m(a), m(b), ac, bc)
    return (m(b), m(a), bc, ac)

def neg_I(I):
    a, b, ac, bc = I
    return (-b, -a, bc, ac)

def shift(I, z):
    a, b, ac, bc = I
    return (a + z, b + z, ac, bc)

def member(op, A, B, z, pole=True):
    """A, B: piece lists (pieces2). z: Fraction or +-inf"""
    if not A or not B: return False
    if math.isinf(z):
        s = 1 if z > 0 else -1
        if op == '+':
            return (mem(A, z) and has_other_than(B, -z)) or (mem(B, z) and has_other_than(A, -z))
        if op == '-':
            return (mem(A, z) and has_other_than(B, z)) or (mem(B, -z) and has_other_than(A, -z))
        if op == '*':
            for X, Y in ((A, B), (B, A)):
                if mem(X, INF) and (has_pos(Y) if s > 0 else has_neg(Y)): return True
                if mem(X, -INF) and (has_neg(Y) if s > 0 else has_pos(Y)): return True
            return False
        if op == '/':
            if mem(A, INF) and (has_finite_pos(B) if s > 0 else has_finite_neg(B)): return True
            if mem(A, -INF) and (has_finite_neg(B) if s > 0 else has_finite_pos(B)): return True
            if pole and mem(B, F(0)):
                for q in B:
                    if not mem_p(q, F(0)): continue
                    right = q[1] > 0; left = q[0] < 0
                    # x/y, y -> 0 from the right: sign(x)*inf; from the left: -sign(x)*inf
                    if right and ((s > 0 and has_pos(A)) or (s < 0 and has_neg(A))): return True
                    if left and ((s > 0 and has_neg(A)) or (s < 0 and has_pos(A))): return True
            return False
    # finite z
    for P in A:
        for Q in B:
            fP, fQ = fin(P), fin(Q)
            if op == '+':
                if nonempty(inter(fP, shift(neg_I(fQ), z))): return True
            elif op == '-':
                if nonempty(inter(fP, shift(fQ, z))): return True
            elif op == '*':
                if z == 0:
                    if (mem_p(P, F(0)) and nonempty(fQ)) or (mem_p(Q, F(0)) and nonempty(fP)): return True
                else:
                    for S in (POS, NEG):
                        sub = inter(fP, S)
                        if nonempty(sub) and nonempty(inter(recip_image(sub, z), Q)): return True
            elif op == '/':
                if z == 0:
                    if mem_p(P, F(0)) and (has_pos([Q]) or has_neg([Q])): return True
                    if nonempty(fP) and (mem_p(Q, INF) or mem_p(Q, -INF)): return True
                else:
                    for S in (POS, NEG):
                        sub = inter(fQ, S)
                        if nonempty(sub) and nonempty(inter(lin_image(sub, z), P)): return True
    return False

def recip_member(A, z, pole=True):
    if math.isinf(z):
        if not pole or not mem(A, F(0)): return False
        side = (lambda q: q[1] > 0) if z > 0 else (lambda q: q[0] < 0)
        return any(mem_p(q, F(0)) and side(q) for q in A)
    if z == 0: return mem(A, INF) or mem(A, -INF)
    return mem(A, 1 / z)

def test_points(*lists):
    pts = set()
    for v in lists:
        for x in v:
            if isinstance(x, float) and (math.isinf(x) or math.isnan(x)): continue
            x = F(x); pts.add(x)
            for d in (F(1, 10**6), F(1, 10**12)): pts.add(x + d); pts.add(x - d)
    xs = sorted(pts)
    for a, b in zip(xs, xs[1:]): pts.add((a + b) / 2)
    pts |= {F(10**9), F(-10**9), F(0), F(10**30), F(-10**30), F(1, 10**30), F(-1, 10**30)}
    return sorted(pts) + [INF, -INF]

def ends1(m): return [p for p, _ in m.endpoints]
def ends2(m): return [x for p in m.pieces for x in (p.inf, p.sup)]

def mk1(ps):
    out = V1()
    for lo, hi, lc, hc in ps:
        out = out.union(V1(lo) if lo == hi else V1(lo, hi, start_closed=lc, end_closed=hc))
    return out

def mk2(ps):
    return V2.from_pieces(ps)

def fmt(ps):
    return ' u '.join(f"[{lo}]" if lo == hi else f"{'[' if lc else '('}{lo}, {hi}{']' if hc else ')'}" for lo, hi, lc, hc in ps) or 'empty'
