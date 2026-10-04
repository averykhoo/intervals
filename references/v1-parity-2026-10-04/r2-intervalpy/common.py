"""shared helpers for the r2-intervalpy probes: v1/v2 import, exact membership comparison"""
import sys
sys.path[:0] = ['.', 'archive/v1']
import math
from fractions import Fraction

import interval as v1i            # v1 Interval, MultipleInterval
import intervals as v2            # v2

V1 = v1i.Interval
MI = v2.MultiInterval

MISMATCHES = []


def check(label, ok, detail=''):
    if not ok:
        MISMATCHES.append((label, detail))
    return ok


def v1_to_v2(iv):
    """v1 Interval -> v2 single piece, through the keyword constructor"""
    return MI(iv.start, iv.end, start_closed=not iv.start_open, end_closed=iv.end_closed)


def test_points(values):
    """ends, just inside/outside, midpoints, exactly (Fraction), plus infinities"""
    pts = set()
    finite = sorted({Fraction(v) for v in values if not math.isinf(v)})
    for v in finite:
        for d in (0, Fraction(1, 10 ** 9), -Fraction(1, 10 ** 9)):
            pts.add(v + d)
    for a, b in zip(finite, finite[1:]):
        pts.add((a + b) / 2)
    if finite:
        pts.add(finite[0] - 1000)
        pts.add(finite[-1] + 1000)
    return sorted(pts)


def v1_members(obj, pts):
    return tuple(p in obj for p in pts)


def v2_members(A, pts):
    return tuple(p in A for p in pts)


def report(name):
    print(f'== {name}: {len(MISMATCHES)} mismatches')
    for m in MISMATCHES[:15]:
        print('  MISMATCH', m)
