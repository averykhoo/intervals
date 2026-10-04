import sys, math, random, warnings
from fractions import Fraction
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import intervals as v2
MI = v2.MultiInterval
V1 = v1.MultiInterval

def conv(a):
    """v1 MultiInterval -> v2 MultiInterval, by its endpoints"""
    e = a.endpoints
    ps = []
    for i in range(0, len(e), 2):
        (lo, le), (hi, he) = e[i], e[i + 1]
        ps.append((lo, hi, le == 0, he == 0))
    return MI.from_pieces(ps)

def tov1(b):
    """v2 -> v1 by its pieces"""
    out = V1()
    for p in b.pieces:
        out.update(V1(p.inf, p.sup, start_closed=p.inf_closed, end_closed=p.sup_closed))
    return out

def probes_of(*sets):
    """ends, just inside/outside, midpoints of every end in the sets"""
    pts = set()
    for s in sets:
        for p in s.pieces:
            for x in (p.inf, p.sup):
                pts.add(x)
                if math.isfinite(x):
                    for d in (Fraction(1, 10**6), Fraction(1, 7)):
                        pts.add(Fraction(x) + d); pts.add(Fraction(x) - d)
            if math.isfinite(p.inf) and math.isfinite(p.sup):
                pts.add((Fraction(p.inf) + Fraction(p.sup)) / 2)
    return pts

FAILS = []
def check(name, cond, detail=''):
    if not cond:
        FAILS.append((name, detail))
    return cond

def report(tag):
    print(f'[{tag}] failures: {len(FAILS)}')
    for f in FAILS[:15]:
        print('   ', f)
