"""shared helpers for the interval.py parity probes (v1 Interval / MultipleInterval vs v2 MultiInterval)"""
import math
import os
import random
import sys
import warnings
from fractions import Fraction as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = 'C:/Users/user/PycharmProjects/intervals'
os.chdir(ROOT)
sys.path[:0] = ['.', 'archive/v1']
import interval as v1i  # noqa: E402
import multi_interval as v1m  # noqa: E402
import intervals as v2  # noqa: E402
from intervals import MultiInterval as M  # noqa: E402

I = v1i.Interval
MI = v1i.MultipleInterval
warnings.simplefilter('ignore')
INF = math.inf

FAILS = []
CHECKS = [0]


def check(name, ok, detail=''):
    CHECKS[0] += 1
    if not ok:
        FAILS.append((name, detail))
        print('MISMATCH', name, detail)


def to_v2(i):
    """v1 Interval -> v2 MultiInterval (same set of reals; v1 never holds a point at infinity)"""
    return M(i.start, i.end, start_closed=i.start_closed, end_closed=i.end_closed)


def mi_to_v2(mi):
    out = M()
    for i in mi:
        out = out | to_v2(i)
    return out


def probe_points(*vals):
    """endpoints, midpoints, just inside / outside (exact), and far points; finite only"""
    fin = sorted({F(v) for v in vals if v is not None and not (isinstance(v, float) and math.isinf(v))})
    pts = set(fin)
    for a, b in zip(fin, fin[1:]):
        pts.add((a + b) / 2)
    for a in fin:
        pts.add(a + F(1, 10**9))
        pts.add(a - F(1, 10**9))
    lo = (fin[0] if fin else 0) - 1000
    hi = (fin[-1] if fin else 0) + 1000
    pts.update({lo, hi, F(-10**30), F(10**30)})
    return sorted(pts)


def ends_of(x):
    """endpoints of a v1 Interval, a v1 MultipleInterval, a v2 MultiInterval or a number"""
    if isinstance(x, I):
        return [x.start, x.end]
    if isinstance(x, MI):
        return [e for i in x for e in (i.start, i.end)]
    if isinstance(x, M):
        return [c.value for c in x.cuts]
    return [x]


def v1_member(x, p):
    return p in x


def same_reals(name, v1obj, v2obj, extra=()):
    """compare as sets of finite reals; v1obj None means the empty set"""
    pts = probe_points(*ends_of(v1obj) if v1obj is not None else [], *ends_of(v2obj), *extra)
    for p in pts:
        a = False if v1obj is None else (p in v1obj)
        b = p in v2obj
        if a != b:
            check(name, False, f'point {p}: v1 {a} v2 {b}; v1={v1obj} v2={v2obj}')
            return False
    check(name, True)
    return True


def rand_val(rng, allow_inf=False):
    r = rng.random()
    if r < 0.4:
        return rng.randint(-6, 6)
    if r < 0.7:
        return F(rng.randint(-30, 30), rng.randint(1, 7))
    if r < 0.9:
        return rng.choice([0, 1, -1, 2, F(1, 2)])
    return rng.randint(-6, 6) + 0.5


def rand_interval(rng, allow_inf=True, allow_point=True):
    while True:
        a, b = sorted([rand_val(rng), rand_val(rng)])
        so, ec = rng.random() < 0.5, rng.random() < 0.5
        if allow_inf and rng.random() < 0.15:
            a, so = -INF, True
        if allow_inf and rng.random() < 0.15:
            b, ec = INF, False
        if a == b:
            if not allow_point:
                continue
            so, ec = False, True
        try:
            return I(a, so, b, ec)
        except ValueError:
            continue


def report_end(fname):
    print(f'{fname}: {CHECKS[0]} checks, {len(FAILS)} mismatches')
