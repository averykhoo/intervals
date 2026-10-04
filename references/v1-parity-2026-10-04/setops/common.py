import sys, os, random, math, warnings
from fractions import Fraction
ROOT = 'C:/Users/user/PycharmProjects/intervals'
os.chdir(ROOT)
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import intervals as v2
from intervals import kernel
inf = math.inf
M1 = v1.MultiInterval
M2 = v2.MultiInterval

GRID = [-3, -2, -1, 0, 1, 2, 3, Fraction(1, 2), Fraction(-5, 2), 0.5, 1.5, -0.25]


def rand_piece(rng, allow_inf=True):
    """a piece (lo, hi, lo_closed, hi_closed) legal in v1 (inf ends open, degenerate closed)"""
    a, b = sorted([rng.choice(GRID), rng.choice(GRID)])
    if allow_inf and rng.random() < 0.15:
        a = -inf
    if allow_inf and rng.random() < 0.15:
        b = inf
    if a == b:
        return (a, a, True, True)
    lc = rng.random() < 0.5 if a != -inf else False
    hc = rng.random() < 0.5 if b != inf else False
    return (a, b, lc, hc)


def rand_pieces(rng, kmax=4, allow_inf=True):
    return [rand_piece(rng, allow_inf) for _ in range(rng.randint(0, kmax))]


def v1_from(pieces):
    out = M1()
    for lo, hi, lc, hc in pieces:
        if lo == hi:
            out = out.union(M1(lo))
        else:
            out = out.union(M1(lo, hi, start_closed=lc, end_closed=hc))
    return out


def v2_from(pieces):
    return M2.from_pieces(pieces)


def twin(pieces):
    return v1_from(pieces), v2_from(pieces)


def from_v1(x):
    """v1 MultiInterval -> v2 MultiInterval, same set (v1 eps 0 = closed)"""
    it = iter(x.endpoints)
    return M2.from_pieces((lo, hi, le == 0, he == 0) for (lo, le), (hi, he) in zip(it, it))


def points_for(*sets2):
    """membership grid: every end value, +-1/8 around it, midpoints, +-inf, a few far points"""
    vals = set([-inf, inf, -100, 100, 0])
    for s in sets2:
        for lo, lc, hi, hc in kernel.pieces(s.cuts):
            for v in (lo, hi):
                vals.add(v)
    fin = sorted(Fraction(v) for v in vals if not (isinstance(v, float) and math.isinf(v)))
    out = set(vals)
    for v in fin:
        out.add(v + Fraction(1, 8)); out.add(v - Fraction(1, 8))
    for a, b in zip(fin, fin[1:]):
        out.add((a + b) / 2)
    return sorted(out, key=lambda v: float(v))


def same_set_v1_v2(r1, r2, extra=()):
    """compare a v1 result and a v2 result as sets: structure via from_v1, and membership on a grid.
    returns None if equal, else a description."""
    c = from_v1(r1)
    if c != r2:
        return f'structure: v1={r1} v2={r2}'
    for p in points_for(c, r2, *extra):
        a = p in r1 if not (isinstance(p, float) and math.isinf(p)) else _v1_in_inf(r1, p)
        b = p in r2
        if a != b:
            return f'membership at {p}: v1 {a} v2 {b}; v1={r1} v2={r2}'
    return None


def _v1_in_inf(r1, p):
    return p in r1


class Tally:
    def __init__(self, name):
        self.name = name; self.n = 0; self.bad = []

    def check(self, label, diff):
        self.n += 1
        if diff is not None:
            self.bad.append((label, diff))

    def report(self, show=5):
        print(f'[{self.name}] cases={self.n} mismatches={len(self.bad)}')
        for label, d in self.bad[:show]:
            print('   ', label, '->', d)
        return len(self.bad)
