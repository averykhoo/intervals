"""the two pairings v1 never defined on TimeDeltaInterval: `timedelta / T` (no __rtruediv__) and `-T` (no __neg__),
plus their siblings `+T`, `abs(T)`, `td // T`, `td % T`, `divmod(td, T)` (no __pos__/__abs__/__rfloordiv__/__rmod__).

v1 direct use: expected TypeError. v1 composition (what a v1 user could write):
  td / A      ->  v1m.MultiInterval(td.total_seconds()) / A.interval     (a MultiInterval of ratios)
  -A          ->  A * -1   (TimeDeltaInterval.__mul__)   or   out.interval = -A.interval
v2: `td / A2`, `-A2`. compared exactly on structure (v1 floats -> Fraction, tolerance only where v1's float division
rounds), and both against a brute-force membership oracle.
"""
import sys
sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import math
import random
import warnings
from fractions import Fraction

import pandas as pd
import time_interval as v1t
import multi_interval as v1m
from intervals import kernel
from intervals.multi_interval import MultiInterval as M2
from intervals.time_interval import TimeDeltaInterval as T2

T1 = v1t.TimeDeltaInterval
US = 10 ** 6
FAILS, CHECKS = [], [0]


def check(label, ok, *info):
    CHECKS[0] += 1
    if not ok:
        FAILS.append((label,) + info)
    return ok


def td(s):
    return dt.timedelta(microseconds=int(Fraction(s) * US))


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


def canon_mi1(mi):
    e = mi.endpoints
    return tuple((e[i][0], e[i][1] == 0, e[i + 1][0], e[i + 1][1] == 0) for i in range(0, len(e), 2))


def canon_mi2(mi):
    return tuple((lo, lc, hi, hc) for lo, lc, hi, hc in kernel.pieces(mi.cuts))


def close(x, y):
    """v1 float vs v2 exact end"""
    if isinstance(x, float) and math.isinf(x) or isinstance(y, float) and math.isinf(y):
        return x == y
    x, y = Fraction(x), Fraction(y)
    return abs(x - y) <= abs(y) * Fraction(1, 10 ** 12) + Fraction(1, 10 ** 15)


def same_structure(c1, c2):
    return len(c1) == len(c2) and all(
        close(a1, a2) and l1 == l2 and close(b1, b2) and h1 == h2 for (a1, l1, b1, h1), (a2, l2, b2, h2) in zip(c1, c2))


def member(pieces, x):
    return any((lo < x or (lc and lo == x)) and (x < hi or (hc and x == hi)) for lo, lc, hi, hc in pieces)


def rand_pieces(rng):
    out = []
    for _ in range(rng.randint(1, 3)):
        a = Fraction(rng.randint(-12, 12), 2)
        b = a + Fraction(rng.randint(0, 6), 2)
        sc, ec = rng.random() < .5, rng.random() < .5
        if a == b:
            sc = ec = True
        out.append((a, b, sc, ec))
    return out


def try_(f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = f()
        return r, [str(x.message)[:70] for x in w]
    except Exception as e:
        return ('raise', type(e).__name__, str(e)[:90]), []


H = dt.timedelta(hours=1)
A1, A2 = T1(H, 3 * H), T2(H, 3 * H)

print('== direct v1 use')
for label, f1, f2 in [
    ('td / T', lambda: H / A1, lambda: H / A2),
    ('pd.Timedelta / T', lambda: pd.Timedelta(hours=1) / A1, lambda: pd.Timedelta(hours=1) / A2),
    ('-T', lambda: -A1, lambda: -A2),
    ('+T', lambda: +A1, lambda: +A2),
    ('abs(T)', lambda: abs(A1), lambda: abs(T2(-H, H))),
    ('td // T', lambda: H // A1, lambda: H // A2),
    ('td % T', lambda: H % A1, lambda: H % A2),
    ('divmod(td, T)', lambda: divmod(H, A1), lambda: divmod(H, A2)),
    ('T / T (v1 has __truediv__ by Real only)', lambda: A1 / A1, lambda: A2 / A2),
    ('5 / T', lambda: 5 / A1, lambda: 5 / A2),
    ('np.timedelta64 / T', lambda: __import__('numpy').timedelta64(3600, 's') / A1,
     lambda: __import__('numpy').timedelta64(3600, 's') / A2),
]:
    r1, w1 = try_(f1)
    r2, w2 = try_(f2)
    print(f'{label:40} v1 {r1!r:70.70} | v2 {r2!r} {w2 if w2 else ""}')

print('== td / T : v1 composition MultiInterval(td.total_seconds()) / A.interval vs v2 td / A')
rng = random.Random(4242)
n_same = n_diff = n_zero = 0
diff_examples = []
oracle_bad2 = 0
for i in range(500):
    pieces = rand_pieces(rng)
    t = Fraction(rng.randint(-8, 8), 2)
    tdv = td(t)
    a1, a2 = build1(pieces), build2(pieces)
    scal = rng.choice(['td', 'pd'])
    num = tdv if scal == 'td' else pd.Timedelta(tdv)
    has0 = any(member([(Fraction(a), sc, Fraction(b), ec)], 0) for a, b, sc, ec in pieces)
    r1, _ = try_(lambda: v1m.MultiInterval(num.total_seconds()) / a1.interval)
    r2, w2 = try_(lambda: num / a2)
    check('v2 type', isinstance(r2, M2), r2)
    c2 = canon_mi2(r2)
    # brute-force oracle: q in t/A iff exists a in A with t/a == q  (a != 0). check sampled a's land in r2,
    # and sampled q's in r2 have a preimage a = t/q in A (q != 0), q == 0 iff t == 0 and some a != 0 in A
    sp = [(Fraction(a), sc, Fraction(b), ec) for a, b, sc, ec in pieces]
    for a, sc, b, ec in sp:
        for x in {a, b, (a + b) / 2, a + (b - a) / 7, b - (b - a) / 9}:
            if member(sp, x) and x != 0:
                ok = r2.contains(t / x) if hasattr(r2, 'contains') else (t / x) in r2
                oracle_bad2 += not check('v2 oracle: t/a in t/A', bool((t / x) in r2), pieces, t, x, c2)
    for lo, lc, hi, hc in c2:
        for q in {lo, hi, (lo + hi) / 2 if not (math.isinf(lo) or math.isinf(hi)) else None}:
            if q is None or (isinstance(q, float) and math.isinf(q)):
                continue
            q = Fraction(q)
            if not member([(lo, lc, hi, hc)], q):
                continue
            if q == 0:
                ok = t == 0 and any(member(sp, x) for x in [Fraction(k, 4) for k in range(-60, 61) if k])
            else:
                ok = t != 0 and member(sp, t / q)
            oracle_bad2 += not check('v2 oracle: q in t/A has a preimage', ok, pieces, t, q, c2)
    if isinstance(r1, tuple) and r1[0] == 'raise':
        same = False
        c1 = r1
    else:
        c1 = canon_mi1(r1)
        same = same_structure(c1, c2)
    if same:
        n_same += 1
    else:
        n_diff += 1
        n_zero += has0
        if len(diff_examples) < 6:
            diff_examples.append((pieces, str(t), 'v1', c1, 'v2', c2, w2))
        if not has0:
            check('td/T differs with 0 not in A', False, pieces, t, c1, c2)
print(f'500 cases: same {n_same}, differ {n_diff} (of which 0 in A: {n_zero}); v2 oracle failures {oracle_bad2}')
for e in diff_examples:
    print('  DIFF', e)

print('== -T : v1 A * -1 (and -A.interval) vs v2 -A')
rng = random.Random(777)
n_bad = 0
for i in range(500):
    pieces = rand_pieces(rng) if rng.random() < .95 else []
    a1, a2 = build1(pieces), build2(pieces)
    r2 = -a2
    check('v2 -A type', type(r2) is T2, type(r2))
    c2 = canon_mi2(r2.seconds)
    m1 = T1()
    m1.interval = -a1.interval
    for lab, f in (('A * -1', lambda: a1 * -1), ('-A.interval', lambda: m1)):
        r1, _ = try_(f)
        if isinstance(r1, tuple):
            check(f'v1 {lab} raised', pieces == [], pieces, r1)  # tolerated for the empty set only
            continue
        n_bad += not check(f'-T vs v1 {lab}', same_structure(canon_mi1(r1.interval), c2), pieces, canon_mi1(r1.interval), c2)
    # oracle: x in -A iff -x in A
    sp = [(Fraction(a), sc, Fraction(b), ec) for a, b, sc, ec in pieces]
    for k in range(-30, 31):
        x = Fraction(k, 4)
        check('-A oracle', member(c2, x) == member(sp, -x), pieces, x)
    check('-A == A * -1 in v2', r2 == a2 * -1, pieces)
    check('-(-A) == A', -r2 == a2, pieces)
print(f'500 cases; -T mismatches vs v1 compositions: {n_bad}')
r, w = try_(lambda: T1() * -1)
print('v1 T() * -1:', r if isinstance(r, tuple) else str(r), '| v2 -T():', str(-T2()))

# deliberately wrong expectation must be caught: -A == A for a non-symmetric A
print('sabotaged expectation caught:', not ((-A2) == A2))
print(f'probe_rdiv_neg: {CHECKS[0]} checks, {len(FAILS)} mismatches')
for f in FAILS[:12]:
    print('  MISMATCH', f)
