"""shared helpers for the construct-slice probes (v1 vs v2)"""
import sys, os, math, warnings
ROOT = 'C:/Users/user/PycharmProjects/intervals'
sys.path[:0] = [ROOT, ROOT + '/archive/v1']
os.chdir(ROOT)
from fractions import Fraction as F
import multi_interval as v1
import intervals as v2
M1, M2 = v1.MultiInterval, v2.MultiInterval
warnings.simplefilter('ignore')

def v1_to_v2(a):
    """structural conversion of a v1 MultiInterval to v2 (epsilon 0 = closed)"""
    e = a.endpoints
    return M2.from_pieces((e[i][0], e[i + 1][0], e[i][1] == 0, e[i + 1][1] == 0) for i in range(0, len(e), 2))

def v1_contains(a, x):
    """independent v1 membership straight from its endpoint list (no v1 method)"""
    e = a.endpoints
    for i in range(0, len(e), 2):
        (s, se), (t, te) = e[i], e[i + 1]
        lo_ok = x > s or (x == s and se == 0)
        hi_ok = x < t or (x == t and te == 0)
        if lo_ok and hi_ok:
            return True
    return False

def probe_points(values):
    """every value, just inside/outside (exact Fractions), midpoints, and +-inf"""
    vals = sorted({v for v in values if not (isinstance(v, float) and math.isinf(v))})
    pts = set(vals) | {-math.inf, math.inf}
    d = F(1, 10 ** 6)
    for v in vals:
        pts |= {F(v) - d, F(v) + d}
    for a, b in zip(vals, vals[1:]):
        pts.add((F(a) + F(b)) / 2)
    return sorted(pts)

def same_set_by_membership(member_a, member_b, points):
    """list of points where the two membership predicates disagree"""
    return [p for p in points if member_a(p) != member_b(p)]

class Tally:
    def __init__(self, name):
        self.name, self.n, self.bad = name, 0, []
    def check(self, ok, info):
        self.n += 1
        if not ok:
            self.bad.append(info)
    def report(self, show=5):
        print(f'[{self.name}] cases={self.n} mismatches={len(self.bad)}')
        for b in self.bad[:show]:
            print('   ', b)
        return len(self.bad)

def outcome(f):
    """('ok', value) or ('raise', ExceptionTypeName)"""
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__ + ': ' + str(e)[:60])
