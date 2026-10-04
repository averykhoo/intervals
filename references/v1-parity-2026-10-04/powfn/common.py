"""shared helpers for the powfn parity probes. run probes from the repo root."""
import math
import sys
import warnings
from fractions import Fraction

sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1  # noqa: E402
import intervals as v2  # noqa: E402

V1 = v1.MultiInterval
V2 = v2.MultiInterval


def mk1(pieces):
    """v1 set from [(lo, hi, lo_closed, hi_closed)]"""
    out = V1()
    for lo, hi, lc, hc in pieces:
        if lo == hi:
            out = out.union(V1(lo)) if hasattr(out, 'union') else out
        else:
            out = out.union(V1(start=lo, end=hi, start_closed=lc, end_closed=hc))
    return out


def mk2(pieces):
    return V2.from_pieces([(lo, hi, lc, hc) for lo, hi, lc, hc in pieces])


def pieces1(a):
    e = a.endpoints
    return [(e[i][0], e[i + 1][0], e[i][1] == 0, e[i + 1][1] == 0) for i in range(0, len(e), 2)]


def pieces2(a):
    return [(p.inf, p.sup, p.inf_closed, p.sup_closed) for p in a]


def contains_pieces(ps, x):
    for lo, hi, lc, hc in ps:
        if (lo < x or (lc and lo == x)) and (x < hi or (hc and x == hi)):
            return True
    return False


def probe_points(*piece_lists):
    """ends, just inside/outside, midpoints, of every piece of every list"""
    pts = set()
    vals = set()
    for ps in piece_lists:
        for lo, hi, _, _ in ps:
            vals.add(lo)
            vals.add(hi)
    vals = sorted(v for v in vals if not (isinstance(v, float) and math.isnan(v)))
    for v in vals:
        pts.add(v)
        if not math.isinf(v):
            fv = Fraction(v)
            for d in (Fraction(1, 10 ** 9), Fraction(1, 10 ** 30)):
                pts.add(fv - d)
                pts.add(fv + d)
            fl = float(v)
            pts.add(math.nextafter(fl, math.inf))
            pts.add(math.nextafter(fl, -math.inf))
    fin = [Fraction(v) for v in vals if not math.isinf(v)]
    for a, b in zip(fin, fin[1:]):
        pts.add((a + b) / 2)
    pts.add(0)
    pts.add(1)
    pts.add(-1)
    return pts


def compare(ps1, ps2, extra=()):
    """list of (x, in_v1, in_v2) where membership differs"""
    diffs = []
    for x in sorted(probe_points(ps1, ps2) | set(extra), key=lambda t: float(t)):
        a = contains_pieces(ps1, x)
        b = contains_pieces(ps2, x)
        if a != b:
            diffs.append((x, a, b))
    return diffs


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **k)


def run(fn):
    """(result, None) or (None, 'ExcType: msg')"""
    try:
        return quiet(fn), None
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'


def well_formed1(a):
    e = a.endpoints
    return all(e[i][1] in (0, 1) and e[i + 1][1] in (0, -1) and e[i] <= e[i + 1] for i in range(0, len(e), 2))


def s1(a):
    try:
        return str(a)
    except Exception:  # noqa: BLE001
        return f'MALFORMED{a.endpoints}'
