"""shared helpers for the timedelta parity probes"""
import sys
sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import random
import warnings
from fractions import Fraction
import pandas as pd
import time_interval as v1t
import multi_interval as v1m
from intervals import kernel
from intervals.multi_interval import MultiInterval as M2
from intervals.time_interval import TimeDeltaInterval as T2, DateTimeInterval as D2, NEG_INF, POS_INF
T1 = v1t.TimeDeltaInterval
D1 = v1t.DateTimeInterval
US = 10 ** 6
FAILS = []
CHECKS = [0]


def td(s):
    """timedelta of s seconds (s a Fraction/int/float with whole microseconds)"""
    return dt.timedelta(microseconds=int(Fraction(s) * US))


def round_us(x):
    return Fraction(round(Fraction(x) * US), US)


def canon1(t, exact=False):
    """v1 TimeDeltaInterval/DateTimeInterval/MultiInterval -> tuple of (lo, lo_closed, hi, hi_closed)"""
    mi = t.interval if hasattr(t, 'interval') else t
    e = mi.endpoints
    f = (lambda x: Fraction(x)) if exact else round_us
    return tuple((f(e[i][0]), e[i][1] == 0, f(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2))


def canon2(t):
    mi = t.seconds if hasattr(t, 'seconds') else t
    return tuple((Fraction(lo), lc, Fraction(hi), hc) for lo, lc, hi, hc in kernel.pieces(mi.cuts))


def check(label, got1, got2, ok=None):
    CHECKS[0] += 1
    good = (got1 == got2) if ok is None else ok
    if not good:
        FAILS.append((label, got1, got2))
    return good


def report(name):
    print(f'{name}: {CHECKS[0]} checks, {len(FAILS)} mismatches')
    for f in FAILS[:15]:
        print('  MISMATCH', f)


def rand_pieces(rng, n=None, lo=-6, hi=6, step=Fraction(1, 2)):
    """a list of (a, b, sc, ec) with a <= b, half-second grid (float-exact)"""
    n = rng.randint(0, 3) if n is None else n
    out = []
    for _ in range(n):
        a = rng.randint(int(lo / step), int(hi / step)) * step
        b = a + rng.randint(0, 6) * step
        sc, ec = rng.random() < .5, rng.random() < .5
        if a == b:
            sc = ec = True
        out.append((a, b, sc, ec))
    return out


def build1(pieces):
    t = T1()
    for a, b, sc, ec in pieces:
        t.update(T1(td(a), td(b), start_closed=sc, end_closed=ec))
    return t


def build2(pieces):
    t = T2()
    for a, b, sc, ec in pieces:
        t = t | T2(td(a), td(b), start_closed=sc, end_closed=ec)
    return t


GRID = [Fraction(k, 4) for k in range(-40, 70)]


def member(pieces, x):
    return any((lo < x or (lc and lo == x)) and (x < hi or (hc and x == hi)) for lo, lc, hi, hc in pieces)


def oracle_set(pieces):
    """the canonical pieces' membership over GRID"""
    return frozenset(x for x in GRID if member(pieces, x))


def spec_pieces(spec):
    """raw spec list (a, b, sc, ec) -> pieces tuple for member()"""
    return tuple((Fraction(a), sc, Fraction(b), ec) for a, b, sc, ec in spec)
