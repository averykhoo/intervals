"""shared helpers for the arith parity probes (v1 archive vs v2 package)"""
import math, random, sys, warnings
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1m
import intervals as v2
V1 = v1m.MultiInterval
V2 = v2.MultiInterval
INF = math.inf

def mk1(pieces):
    """v1 set from (lo, hi, lo_closed, hi_closed) pieces"""
    out = V1()
    for lo, hi, lc, hc in pieces:
        if lo == hi:
            p = V1(lo)
        else:
            p = V1(lo, hi, start_closed=lc, end_closed=hc)
        out = out.union(p)
    return out

def mk2(pieces):
    return V2.from_pieces([(lo, hi, lc, hc) if lo != hi else (lo, lo, True, True) for lo, hi, lc, hc in pieces])

VALS = [-INF, F(-3), F(-2), F(-1), F(-1, 2), F(0), F(1, 2), F(1), F(2), F(3), INF]

def rand_pieces(rng, n=None, allow_inf=True, allow_points=True):
    n = rng.choice([1, 1, 1, 2, 2, 3]) if n is None else n
    vals = VALS if allow_inf else VALS[1:-1]
    out = []
    for _ in range(n):
        a, b = sorted(rng.sample(vals, 2))
        if allow_points and rng.random() < 0.15 and not math.isinf(a):
            out.append((a, a, True, True))
            continue
        lc = rng.random() < 0.5 and not math.isinf(a)
        hc = rng.random() < 0.5 and not math.isinf(b)
        out.append((a, b, lc, hc))
    return out

def v1_ends(m):
    return [p for p, _ in m.endpoints]

def v2_ends(m):
    out = []
    for p in m.pieces:
        out += [p.inf, p.sup]
    return out

def probe_points(*vals_lists):
    pts = set()
    for v in vals_lists:
        for x in v:
            if isinstance(x, float) and math.isinf(x):
                continue
            x = F(x)
            pts.add(x)
            for d in (F(1, 10**6), F(1, 10**12)):
                pts.add(x + d); pts.add(x - d)
    xs = sorted(pts)
    for a, b in zip(xs, xs[1:]):
        pts.add((a + b) / 2)
    pts |= {F(10**9), F(-10**9), F(0)}
    return sorted(pts)

def finite_diff(m1, m2, extra=()):
    """points (finite) where membership differs: list of (x, in_v1, in_v2)"""
    pts = probe_points(v1_ends(m1), v2_ends(m2), extra)
    return [(x, x in m1, x in m2) for x in pts if (x in m1) != (x in m2)]

def inf_diff(m1, m2):
    return [(x, x in m1, x in m2) for x in (-INF, INF) if (x in m1) != (x in m2)]

def fmt_pieces(ps):
    def f(x):
        return str(x)
    return ' u '.join(f"{'[' if lc else '('}{f(lo)}, {f(hi)}{']' if hc else ')'}" if lo != hi else f'[{f(lo)}]' for lo, hi, lc, hc in ps) or 'empty'
