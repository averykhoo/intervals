"""shared helpers: build the same set in v1 and v2 from a piece list, random piece lists, brute membership"""
import sys, os, math, random, warnings
from fractions import Fraction
ROOT = 'C:/Users/user/PycharmProjects/intervals'
os.chdir(ROOT)
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import intervals as v2

INF = math.inf
FAILS = []


def build(pieces):
    """pieces: list of (lo, hi, lo_closed, hi_closed). returns (v1 obj, v2 obj)"""
    a = v1.MultiInterval()
    for lo, hi, lc, hc in pieces:
        if lo == hi:
            p = v1.MultiInterval(lo)
        else:
            p = v1.MultiInterval(lo, hi, start_closed=lc, end_closed=hc)
        a = a.union(p)
    b = v2.MultiInterval.from_pieces([(lo, hi, lc, hc) if lo != hi else (lo, lo, True, True) for lo, hi, lc, hc in pieces])
    return a, b


GRID = sorted([-INF, -3, Fraction(-5, 2), -2, -1.5, -1, Fraction(-1, 3), 0, 0.5, Fraction(1, 3), 1, 2, 2.5, 3, INF])


def rand_pieces(rng, n=None):
    n = rng.randint(0, 4) if n is None else n
    out = []
    for _ in range(n):
        a, b = sorted(rng.sample(range(len(GRID)), 2)) if rng.random() < .8 else (None, None)
        if a is None:
            # a degenerate point (finite)
            x = rng.choice([g for g in GRID if abs(g) != INF])
            out.append((x, x, True, True))
            continue
        lo, hi = GRID[a], GRID[b]
        lc = rng.random() < .5 and lo != -INF
        hc = rng.random() < .5 and hi != INF
        out.append((lo, hi, lc, hc))
    return out


def v1_contains(a, x):
    """v1 membership straight from the endpoint list (independent of v1's __contains__)"""
    e = a.endpoints
    for i in range(0, len(e), 2):
        (s, se), (t, te) = e[i], e[i + 1]
        lo_ok = s < x or (s == x and se == 0)
        hi_ok = x < t or (t == x and te == 0)
        if lo_ok and hi_ok:
            return True
    return False


def probe_points(*objs):
    pts = set([0, 1, -1])
    for o in objs:
        for c in o.cuts:
            v = c.value
            if abs(v) != INF:
                pts |= {v, v - Fraction(1, 10**6), v + Fraction(1, 10**6)}
            else:
                pts.add(v)
    pts = sorted(pts)
    mids = [(p + q) / 2 for p, q in zip(pts, pts[1:]) if abs(p) != INF and abs(q) != INF]
    return pts + mids


def check(label, cond, detail=''):
    if not cond:
        FAILS.append((label, detail))
    return cond


def report(name, n):
    print(f'{name}: {n} cases, {len(FAILS)} mismatches')
    for f in FAILS[:15]:
        print('  MISMATCH', f)
