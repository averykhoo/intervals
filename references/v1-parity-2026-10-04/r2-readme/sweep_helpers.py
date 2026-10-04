# helpers copied from probe_unsorted_sweeps.py (generated)

import copy
import math
import random
import sys
import warnings
from fractions import Fraction

sys.path[:0] = ['.', 'archive/v1']
import compare as v1  # noqa: E402
import intervals as v2  # noqa: E402
from intervals import kernel  # noqa: E402

MI = v2.MultiInterval
INF = math.inf


# ---- the shared reading of a record --------------------------------------------------------------

def closed_start(eps):
    return eps != 2


def closed_end(eps):
    return eps != -2


def rec_covers(rec, x):
    (s, se), (e, ee) = rec
    lo_ok = s < x or (s == x and closed_start(se))
    hi_ok = x < e or (x == e and closed_end(ee))
    return lo_ok and hi_ok


def oracle(records, x):
    """brute force: x is in the union of the input records"""
    return any(rec_covers(r, x) for r in records)


def v1_member(out, x):
    return any(rec_covers(r, x) for r in out)


def v2_pieces(records):
    return [(s, e, closed_start(se), closed_end(ee)) for (s, se), (e, ee) in records]


def v2_from_pieces(records):
    return MI.from_pieces(v2_pieces(records))


def v2_builder(records):
    b = kernel.Builder()
    for p in v2_pieces(records):
        b.add_piece(*p)
    return MI.from_cuts(b.build())


def test_points(records):
    vals = set()
    for (s, _), (e, _) in records:
        vals.add(s)
        vals.add(e)
    vals = sorted(vals)
    pts = list(vals)
    for a, b in zip(vals, vals[1:]):
        if math.isinf(a) or math.isinf(b):
            pts.append(b - 1 if math.isinf(a) else a + 1)
        else:
            pts.append((Fraction(a) + Fraction(b)) / 2)
    finite = [v for v in vals if not math.isinf(v)]
    if finite:
        pts.append(min(finite) - 1)
        pts.append(max(finite) + 1)
    else:
        pts.append(0)
    return pts


def v1_canonical(out):
    """v1 output pieces as (lo, hi, lo_closed, hi_closed), empty records dropped"""
    canon = []
    for (s, se), (e, ee) in out:
        lc, hc = closed_start(se), closed_end(ee)
        if s < e or (s == e and lc and hc):
            canon.append((s, e, lc, hc))
    return canon


def v2_canonical(mi):
    return [(lo, hi, lc, hc) for lo, lc, hi, hc in kernel.pieces(mi.cuts)]


def same_number(a, b):
    return a == b



# ---- random records -------------------------------------------------------------------------

POOL = [-3, -2, -1, 0, 1, 2, 3, Fraction(1, 2), Fraction(-5, 3), 0.5, 2.0, -0.0, 0.25, -INF, INF]
EPS = [-2, -1, 0, 2]


def rand_records(rng, n):
    recs = []
    for _ in range(n):
        a, b = rng.choice(POOL), rng.choice(POOL)
        if b < a:
            a, b = b, a
        recs.append(((a, rng.choice(EPS)), (b, rng.choice(EPS))))
    rng.shuffle(recs)
    return recs


