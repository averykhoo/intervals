"""str() of TimeDeltaInterval ends that are not whole microseconds: v1 vs v2, against the exact value.

both sides hold the SAME exact ends: dyadic floats (k / 2**m s), which v1 stores as floats and v2 takes exactly
(from_seconds of a MultiInterval of Fractions). each side's text is decoded back to a number and compared with
the exact end.
"""
import sys
sys.path[:0] = ['.', 'archive/v1']
import datetime as dt
import random
import re
import warnings
from fractions import Fraction

import pandas as pd
import time_interval as v1t
import multi_interval as v1m
from intervals import kernel
from intervals.multi_interval import MultiInterval as M2
from intervals.time_interval import TimeDeltaInterval as T2

warnings.simplefilter('ignore')
T1 = v1t.TimeDeltaInterval
US = 10 ** 6
FAILS, CHECKS = [], [0]


def check(label, ok, *info):
    CHECKS[0] += 1
    if not ok:
        FAILS.append((label,) + info)
    return ok


# ---- builders: the same exact pieces on both sides
def build1(pieces):
    t = T1()
    for a, b, sc, ec in pieces:
        p = T1()
        p.interval = v1m.MultiInterval(start=float(a), end=float(b), start_closed=sc, end_closed=ec)
        t.update(p)
    return t


def build2(pieces):
    t = T2()
    for a, b, sc, ec in pieces:
        t = t | T2.from_seconds(M2(Fraction(a), Fraction(b), start_closed=sc, end_closed=ec))
    return t


def canon2(t):
    return tuple((Fraction(lo), lc, Fraction(hi), hc) for lo, lc, hi, hc in kernel.pieces(t.seconds.cuts))


def canon1(t):
    e = t.interval.endpoints
    return tuple((Fraction(e[i][0]), e[i][1] == 0, Fraction(e[i + 1][0]), e[i + 1][1] == 0)
                 for i in range(0, len(e), 2))


# ---- text decoders
V2_END = re.compile(r'^(-?)(?:(\d+) days? )?(\d+):(\d\d):(\d\d)(?:\.(\d{6}))?(?:\+(\d+)(?:/(\d+))?us)?$')
V1_END = re.compile(r'^(?:(-?\d+) days?, )?(\d+):(\d\d):(\d\d)(?:\.(\d{6}))?$')


def dec2(s, reading='signed-whole-plus-rest'):
    """v2 end text -> Fraction seconds. reading 'signed-whole-plus-rest': (-W) + r  (what the code does);
    'sign-over-all': -(W + r)  (the 'signed as a whole' reading applied to the whole text)"""
    m = V2_END.match(s)
    assert m, s
    sign, d, h, mi, se, f, rn, rd = m.groups()
    whole = int(d or 0) * 86400 + int(h) * 3600 + int(mi) * 60 + int(se) + Fraction(int(f or 0), US)
    rest = Fraction(int(rn), int(rd or 1)) / US if rn else Fraction(0)
    if reading == 'signed-whole-plus-rest':
        return (-whole if sign else whole) + rest
    return -(whole + rest) if sign else whole + rest


def dec1(s):
    m = V1_END.match(s)
    assert m, s
    d, h, mi, se, f = m.groups()
    return int(d or 0) * 86400 + int(h) * 3600 + int(mi) * 60 + int(se) + Fraction(int(f or 0), US)


PIECE = re.compile(r'^([\[(])(.*?)(?:, (.*?))?([\])])$')


def split_v2(text):
    """'{ [a, b) , [c] }' -> list of (open_char, a, b, close_char); v2 end texts have no ', '"""
    if text == '{}':
        return []
    body = text[2:-2].split(' , ') if text.startswith('{ ') else [text]
    out = []
    for p in body:
        m = PIECE.match(p)
        assert m, p
        o, a, b, c = m.groups()
        out.append((o, a, a if b is None else b, c))
    return out


def split_v1(text):
    """v1 end texts may contain ', ' (negative days), so split on the end pattern"""
    if text == '{}':
        return []
    body = text[2:-2].split(' , ') if text.startswith('{ ') else [text]
    end = r'(?:-?\d+ days?, )?\d+:\d\d:\d\d(?:\.\d{6})?'
    out = []
    for p in body:
        m = re.match(rf'^([\[(])({end})(?:, ({end}))?([\])])$', p)
        assert m, p
        o, a, b, c = m.groups()
        out.append((o, a, a if b is None else b, c))
    return out


def round_half_even_us(x):
    return Fraction(round(x * US), US)  # Fraction.__round__ is half-even


# ---- hand-picked
print('== hand-picked')
hand = [
    ('T(1s)/3', T1(dt.timedelta(seconds=1)) / 3, T2(dt.timedelta(seconds=1)) / 3, Fraction(1, 3)),
    ('T(-1s)/3', T1(dt.timedelta(seconds=-1)) / 3, T2(dt.timedelta(seconds=-1)) / 3, Fraction(-1, 3)),
    ('T(1us)/2', T1(dt.timedelta(microseconds=1)) / 2, T2(dt.timedelta(microseconds=1)) / 2, Fraction(1, 2 * US)),
    ('T(-1us)/2', T1(dt.timedelta(microseconds=-1)) / 2, T2(dt.timedelta(microseconds=-1)) / 2, Fraction(-1, 2 * US)),
    ('T(3us)/2', T1(dt.timedelta(microseconds=3)) / 2, T2(dt.timedelta(microseconds=3)) / 2, Fraction(3, 2 * US)),
    ('T(1day)/7', T1(dt.timedelta(days=1)) / 7, T2(dt.timedelta(days=1)) / 7, Fraction(86400, 7)),
    ('T(-3days)/7', T1(dt.timedelta(days=-3)) / 7, T2(dt.timedelta(days=-3)) / 7, Fraction(-3 * 86400, 7)),
    ('pd 1500ns', T1(pd.Timedelta(1500, 'ns')), T2(pd.Timedelta(1500, 'ns')), Fraction(1500, 10 ** 9)),
    ('pd -1500ns', T1(pd.Timedelta(-1500, 'ns')), T2(pd.Timedelta(-1500, 'ns')), Fraction(-1500, 10 ** 9)),
    ('pd 1ns', T1(pd.Timedelta(1, 'ns')), T2(pd.Timedelta(1, 'ns')), Fraction(1, 10 ** 9)),
    ('T(1h)*0.1', T1(dt.timedelta(hours=1)) * 0.1, T2(dt.timedelta(hours=1)) * 0.1, Fraction(0.1) * 3600),
]
for label, a1, a2, exact in hand:
    s1, s2 = str(a1), str(a2)
    e2 = split_v2(s2)[0][1]
    v2_ok = dec2(e2) == exact
    alt = dec2(e2, 'sign-over-all')
    print(f'{label:12} v1 {s1!r:34} v2 {s2!r}')
    print(f'{"":12} exact {exact} | v2 decoded == exact: {v2_ok} | v2 read as -(W+r): {"== exact" if alt == exact else f"{float(alt)!r} (off by {float(alt - exact) * US:.6g} us)"}'
          f' | v1 decoded {dec1(split_v1(s1)[0][1])} (|err| {float(abs(dec1(split_v1(s1)[0][1]) - exact)) * US:.4g} us)')
    check(f'hand v2 {label}', v2_ok, s2, exact)

# hand: a non-degenerate interval whose two ends v1 prints identically, and a multi-piece one
lossy_pieces = [(Fraction(1, 2 ** 22), Fraction(3, 2 ** 22), True, False)]  # 0.238 us .. 0.715 us
print('lossy one piece: v1', str(build1(lossy_pieces)), '| v2', str(build2(lossy_pieces)))
lossy_pieces = [(Fraction(1, 2 ** 23), Fraction(1, 2 ** 22), True, True)]  # 0.119 us .. 0.238 us
print('lossy one piece: v1', str(build1(lossy_pieces)), '| v2', str(build2(lossy_pieces)))
multi = [(Fraction(1, 2 ** 22), Fraction(1, 2 ** 21), False, True), (Fraction(3, 2 ** 21), Fraction(3, 2 ** 21), True, True),
         (Fraction(-5, 2 ** 21), Fraction(-1, 2 ** 21), True, False)]
print('multi: v1', str(build1(multi)), '| v2', str(build2(multi)))
# big magnitude with a sub-us rest
big = [(Fraction(10 ** 9 * 86400 - 1) + Fraction(1, 2), Fraction(10 ** 9 * 86400 - 1) + Fraction(1, 2), True, True)]
try:
    s1 = str(build1(big))
except Exception as e:
    s1 = f'raise {type(e).__name__}: {e}'
print('past timedelta.max: v1', s1, '| v2', str(build2(big)))
near_max = Fraction(dt.timedelta.max.days * 86400 + 86399) + Fraction(1, 4)  # inside range, float holds .25 exactly
try:
    s1 = str(build1([(near_max, near_max, True, True)]))
except Exception as e:
    s1 = f'raise {type(e).__name__}: {e}'
print('near timedelta.max +1/4 s: v1', s1, '| v2', str(build2([(near_max, near_max, True, True)])))
# a value whose floor-us lands at timedelta.min: negative near the bottom
near_min = -Fraction(dt.timedelta.max.days * 86400 + 86399) - Fraction(1, 4)
try:
    s1 = str(build1([(near_min, near_min, True, True)]))
except Exception as e:
    s1 = f'raise {type(e).__name__}: {e}'
print('near -timedelta.max -1/4 s: v1', s1, '| v2', str(build2([(near_min, near_min, True, True)])))

# ---- random sweep
print('== random sweep')
rng = random.Random(20261004)
n_cases = 600
n_v1_text_differs = n_v1_lossy_equal_ends = n_v1_err = n_alt_misread = n_neg_sub = 0
n_v1_round_ok = 0
examples = []
for i in range(n_cases):
    m = rng.choice([18, 20, 21, 22, 23, 24, 26])  # 2**-m s: sub-microsecond granularity
    span = rng.choice([4, 64, 2 ** 12, 2 ** 30, 2 ** 40])
    k = rng.randint(1, 3)
    pieces = []
    for _ in range(k):
        a = Fraction(rng.randint(-span, span), 2 ** m)
        b = a + Fraction(rng.randint(0, span // 2 + 1), 2 ** m)
        sc, ec = rng.random() < .5, rng.random() < .5
        if a == b:
            sc = ec = True
        pieces.append((a, b, sc, ec))
    a1, a2 = build1(pieces), build2(pieces)
    c1, c2 = canon1(a1), canon2(a2)
    check('structure', c1 == c2, pieces, c1, c2)
    s1, s2 = str(a1), str(a2)
    p1, p2 = split_v1(s1), split_v2(s2)
    check('piece count', len(p1) == len(p2) == len(c2), s1, s2)
    for (o1, x1, y1, z1), (o2, x2, y2, z2), (lo, lc, hi, hc) in zip(p1, p2, c2):
        check('brackets', (o1, z1) == (o2, z2) == ('[' if lc else '(', ']' if hc else ')'), s1, s2)
        for t1, t2, exact in ((x1, x2, lo), (y1, y2, hi)):
            v2v = dec2(t2)
            check('v2 text decodes to the exact end', v2v == exact, t2, exact)
            v1v = dec1(t1)
            n_v1_round_ok += v1v == round_half_even_us(exact)
            if v1v != exact:
                n_v1_err += 1
            if t1 != t2:
                n_v1_text_differs += 1
            if exact < 0 and exact * US != int(exact * US):
                n_neg_sub += 1
                if dec2(t2, 'sign-over-all') != exact:
                    n_alt_misread += 1
                    if len(examples) < 3:
                        examples.append((t2, exact, float(dec2(t2, 'sign-over-all') - exact) * US))
        if lo != hi and dec1(x1) == dec1(y1):
            n_v1_lossy_equal_ends += 1
total_ends = 2 * sum(len(canon2(build2([]))) for _ in [0])  # placeholder
print(f'{n_cases} random sets; v2 every end decoded == exact unless listed below')
print(f'v1 end text != exact value: {n_v1_err} ends; == round-half-even to the us: {n_v1_round_ok} ends')
print(f'v1 non-degenerate pieces whose two ends print the SAME text: {n_v1_lossy_equal_ends}')
print(f'end texts differing v1 vs v2: {n_v1_text_differs}')
print(f'negative sub-us ends: {n_neg_sub}; of them, misread under "-(whole+rest)": {n_alt_misread}; e.g. {examples}')

# ---- the probe can fail: a deliberately wrong expectation (v2 text == the half-even rounding) must be caught
wrong = [dec2(split_v2(str(T2(dt.timedelta(seconds=1)) / 3))[0][1]) == round_half_even_us(Fraction(1, 3))]
print('sabotaged expectation caught:', not wrong[0])
print(f'probe_str_subus: {CHECKS[0]} checks, {len(FAILS)} mismatches')
for f in FAILS[:10]:
    print('  MISMATCH', f)
