"""shared helpers for the datetime parity probes: import v1 and v2 side by side, compare as sets"""
import sys, os, warnings, datetime as dt, random
from fractions import Fraction
ROOT = 'C:/Users/user/PycharmProjects/intervals'
sys.path[:0] = [ROOT, ROOT + '/archive/v1']
os.chdir(ROOT)
warnings.simplefilter('ignore')
import pandas as pd
import time_interval as v1t          # v1
import multi_interval as v1m
import intervals as v2
import intervals.time_interval as v2t
from intervals import MultiInterval as M2

V1D, V1T = v1t.DateTimeInterval, v1t.TimeDeltaInterval
V2D, V2T = v2t.DateTimeInterval, v2t.TimeDeltaInterval
NEG_INF, POS_INF = v2t.NEG_INF, v2t.POS_INF
US = dt.timedelta(microseconds=1)

FAILS = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(msg)
    return cond


def selftest(fn_compare, *args):
    """a deliberately wrong expectation must be caught by the comparison"""
    assert not fn_compare(*args), 'SELFTEST: comparison could not fail'
    print('SELFTEST ok: a wrong expectation was caught')


def report(name):
    print(f'{name}: {CHECKS[0]} checks, {len(FAILS)} failures')
    for f in FAILS[:25]:
        print('  FAIL', f)


def safe(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__ + ': ' + str(e)[:120])


def v1_points(a):
    """v1 endpoints as naive datetimes (local reading)"""
    return [dt.datetime.fromtimestamp(float(x)) for x, _ in a.interval.endpoints if abs(x) != float('inf')]


def v2_points(a):
    out = []
    for p in a:
        for v in (p.inf, p.sup):
            if isinstance(v, dt.datetime):
                out.append(v.replace(tzinfo=None) if v.tzinfo is None else v)
    return out


def candidates(*pts):
    """each endpoint, +-1 us, +-1 s, and midpoints of consecutive ones"""
    base = sorted(set(pts))
    out = set()
    for p in base:
        for d in (dt.timedelta(0), US, -US, dt.timedelta(seconds=1), -dt.timedelta(seconds=1)):
            out.add(p + d)
    for a, b in zip(base, base[1:]):
        out.add(a + (b - a) // 2)
    return sorted(out)


def members(a, pts):
    return [p in a for p in pts]


def same_set(a1, a2, extra=()):
    """v1 and v2 agree on membership at every candidate microsecond instant"""
    pts = candidates(*v1_points(a1), *v2_points(a2), *extra)
    m1, m2 = members(a1, pts), members(a2, pts)
    if m1 != m2:
        diff = [(p, x, y) for p, x, y in zip(pts, m1, m2) if x != y]
        return False, diff[:4]
    return True, None


def rand_dt(r, lo=dt.datetime(1990, 1, 1), span_days=30, whole=False):
    d = lo + dt.timedelta(days=r.randrange(span_days), seconds=r.randrange(86400))
    if not whole:
        d += dt.timedelta(microseconds=r.randrange(1, 10 ** 6))
    return d


def pts_of(a):
    return v1_points(a) if isinstance(a, V1D) else v2_points(a)


def same_set_any(a, b, extra=()):
    pts = candidates(*pts_of(a), *pts_of(b), *extra)
    m1, m2 = members(a, pts), members(b, pts)
    if m1 != m2:
        return False, [(p, x, y) for p, x, y in zip(pts, m1, m2) if x != y][:4]
    return True, None
