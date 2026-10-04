"""shared helpers for the floordivmod parity probes (v1 archive vs v2 package)"""
import sys, os, math, warnings, random
from fractions import Fraction
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'archive', 'v1')]
import multi_interval as v1
import intervals as v2
from intervals import kernel
from intervals.fmt import format_cuts
warnings.simplefilter('ignore')

INF = math.inf

def ex(v):
    """exact value (float -> Fraction it denotes), inf kept"""
    if isinstance(v, float) and math.isfinite(v):
        return Fraction(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return Fraction(v)
    return v

def v1_pieces(m):
    e = m.endpoints
    return [(ex(e[i][0]), e[i][1] == 0, ex(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]

def v2_pieces(m):
    return [(ex(lo), lc, ex(hi), hc) for lo, lc, hi, hc in kernel.pieces(m.cuts)]

def canon(pcs):
    """normalize a piece list through v2's kernel so equal sets compare equal"""
    cuts = kernel.normalize(kernel.piece(lo, hi, lc, hc) for lo, lc, hi, hc in pcs)
    return [(ex(lo), lc, ex(hi), hc) for lo, lc, hi, hc in kernel.pieces(cuts)]

def show(pcs):
    if not pcs:
        return '{}'
    return ' u '.join(f"{'[' if lc else '('}{lo}, {hi}{']' if hc else ')'}" for lo, lc, hi, hc in pcs)

def mk1(pcs):
    out = v1.MultiInterval()
    for lo, lc, hi, hc in pcs:
        out = out.union(v1.MultiInterval(lo, hi, start_closed=lc, end_closed=hc))
    return out

def mk2(pcs):
    return v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in pcs]) if pcs else v2.MultiInterval()

def contains(pcs, x):
    return any((lo < x < hi) or (x == lo and lc) or (x == hi and hc) for lo, lc, hi, hc in pcs)

def run(f):
    """call f, return ('ok', value) or ('err', ExceptionTypeName: msg)"""
    try:
        return 'ok', f()
    except Exception as e:  # noqa
        return 'err', f'{type(e).__name__}: {e}'

def rand_pieces(rng, n_max=3, lo=0, hi=12, den=(1, 2, 4), allow_point=True, closed_only=False):
    """random disjoint exact pieces within [lo, hi]"""
    n = rng.randint(1, n_max)
    pts = sorted({Fraction(rng.randint(lo * 4, hi * 4), rng.choice(den)) for _ in range(2 * n + 2)})
    pts = [p for p in pts if lo <= p <= hi]
    out = []
    i = 0
    while i + 1 < len(pts) and len(out) < n:
        a, b = pts[i], pts[i + 1]
        if allow_point and rng.random() < 0.15:
            out.append((a, True, a, True)); i += 2; continue
        lc = True if closed_only else rng.random() < 0.6
        hc = True if closed_only else rng.random() < 0.6
        out.append((a, lc, b, hc)); i += 2
    return canon(out) if out else [(Fraction(lo), True, Fraction(lo), True)]


def attained_mod(r, A, B):
    """independent exact oracle (finite pieces, any sign): is r == x mod y (python floor-mod) for x in A, y in B?
    r = x - k*y with integer k and r on y's side: 0 <= r < y for y > 0, y < r <= 0 for y < 0.
    for an x-piece and an admissible y-piece the reals k = (x - r) / y form an interval whose ends are among the
    four corner quotients, so only integers near those quotients (and one interior one) need an exact check."""
    r = Fraction(r)
    for blo, blc, bhi, bhc in B:
        for sign in (1, -1):
            if sign == 1:
                if r < 0: continue
                (ylo, ylc), (yhi, yhc) = max_lo((blo, blc), (max(r, 0), False)), (bhi, bhc)
            else:
                if r > 0: continue
                (ylo, ylc), (yhi, yhc) = (blo, blc), min_hi((bhi, bhc), (min(r, 0), False))
            if not nonempty(ylo, ylc, yhi, yhc):
                continue
            for alo, alc, ahi, ahc in A:
                if contains([(alo, alc, ahi, ahc)], r):
                    return True  # k = 0
                touches0 = (ylo == 0) if sign == 1 else (yhi == 0)
                if touches0:  # only when r == 0: x / k lands in Y for any x != 0 and |k| large
                    if alo != 0 or ahi != 0:
                        return True
                    continue
                qs = [(x - r) / y for x in (alo, ahi) for y in (ylo, yhi)]
                cand = set()
                for q in qs:
                    f = math.floor(q)
                    cand.update(range(f - 2, f + 3))
                cand.add(math.floor((min(qs) + max(qs)) / 2))
                for k in cand:
                    if k == 0: continue
                    if k > 0:
                        lo, lc, hi, hc = (alo - r) / k, alc, (ahi - r) / k, ahc
                    else:
                        lo, lc, hi, hc = (ahi - r) / k, ahc, (alo - r) / k, alc
                    if inter(lo, lc, hi, hc, ylo, ylc, yhi, yhc):
                        return True
    return False


def max_lo(a, b):
    if a[0] > b[0]: return a
    if b[0] > a[0]: return b
    return (a[0], a[1] and b[1])

def min_hi(a, b):
    if a[0] < b[0]: return a
    if b[0] < a[0]: return b
    return (a[0], a[1] and b[1])

def nonempty(lo, lc, hi, hc):
    return lo < hi or (lo == hi and lc and hc)

def inter(lo1, lc1, hi1, hc1, lo2, lc2, hi2, hc2):
    lo, lc = max_lo((lo1, lc1), (lo2, lc2))
    hi, hc = min_hi((hi1, hc1), (hi2, hc2))
    return nonempty(lo, lc, hi, hc)


def probe_points(*piece_lists, extra=()):
    ends = set(extra)
    for pcs in piece_lists:
        for lo, _, hi, _ in pcs:
            for v in (lo, hi):
                if isinstance(v, Fraction) or isinstance(v, int):
                    ends.add(Fraction(v))
    ends = sorted(ends)
    pts = set(ends)
    eps = Fraction(1, 10 ** 9)
    for e in ends:
        pts.add(e - eps); pts.add(e + eps)
    for a, b in zip(ends, ends[1:]):
        pts.add((a + b) / 2)
    return sorted(pts)
