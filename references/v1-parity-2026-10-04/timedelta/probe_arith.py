from common import *
import numpy as np


def _s(e):
    try:
        return str(e)[:70]
    except Exception:
        return '<str failed>'


def outcome(f):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__, _s(e))


BASE = dt.datetime(2024, 1, 1)
BASE_TS = BASE.timestamp()          # v1 reads a naive datetime through the local zone (UTC+8 here, no DST)
WALL = Fraction(int((BASE - dt.datetime(1970, 1, 1)).total_seconds()))  # v2 reads it as wall clock


def dti1(spec):
    out = D1()
    for a, b, sc, ec in spec:
        out.interval.update(v1m.MultiInterval(float(BASE_TS + a), float(BASE_TS + b), start_closed=sc, end_closed=ec))
    return out


def dti2(spec):
    out = D2()
    for a, b, sc, ec in spec:
        out = out | D2(BASE + td(a), BASE + td(b), start_closed=sc, end_closed=ec)
    return out


def dcanon1(d):
    """v1 DateTimeInterval -> pieces in seconds relative to BASE"""
    return tuple((round_us(Fraction(lo) - Fraction(BASE_TS)), lc, round_us(Fraction(hi) - Fraction(BASE_TS)), hc)
                 for lo, lc, hi, hc in canon1(d, exact=True))


def dcanon2(d):
    return tuple((lo - WALL, lc, hi - WALL, hc) for lo, lc, hi, hc in canon2(d))


def res(r, canon):
    return canon(r[1]) if r[0] == 'ok' else r[:2]


rng = random.Random(5)
for i in range(300):
    sa, sb = rand_pieces(rng), rand_pieces(rng)
    a1, a2, b1, b2 = build1(sa), build2(sa), build1(sb), build2(sb)
    # A. duration +- duration (interval, timedelta, pd.Timedelta, both sides)
    x = rng.choice(GRID)
    t = td(x)
    pt = pd.Timedelta(t)
    for lab, f1, f2 in [
            ('T+T', lambda: a1 + b1, lambda: a2 + b2), ('T-T', lambda: a1 - b1, lambda: a2 - b2),
            ('T+td', lambda: a1 + t, lambda: a2 + t), ('td+T', lambda: t + a1, lambda: t + a2),
            ('T-td', lambda: a1 - t, lambda: a2 - t), ('td-T', lambda: t - a1, lambda: t - a2),
            ('T+pdTd', lambda: a1 + pt, lambda: a2 + pt), ('pdTd+T', lambda: pt + a1, lambda: pt + a2),
            ('T-pdTd', lambda: a1 - pt, lambda: a2 - pt), ('pdTd-T', lambda: pt - a1, lambda: pt - a2)]:
        r1, r2 = outcome(f1), outcome(f2)
        check((lab, sa, sb, x), res(r1, canon1), res(r2, canon2))
        if r1[0] == 'ok':
            check((lab + ' v1 type', sa), type(r1[1]).__name__, 'TimeDeltaInterval')
        if r2[0] == 'ok':
            check((lab + ' v2 type', sa), type(r2[1]).__name__, 'TimeDeltaInterval')
            # oracle: the sum/difference set by brute force over the grid pieces (exact, interval endpoints)
    # B. duration + instant -> DateTimeInterval (scalar datetime, Timestamp, interval; both sides)
    dt_ = BASE + t
    ts_ = pd.Timestamp(dt_)
    sd = rand_pieces(rng)
    d1, d2 = dti1(sd), dti2(sd)
    for lab, f1, f2 in [
            ('T+dt', lambda: a1 + dt_, lambda: a2 + dt_), ('dt+T', lambda: dt_ + a1, lambda: dt_ + a2),
            ('T+Timestamp', lambda: a1 + ts_, lambda: a2 + ts_), ('Timestamp+T', lambda: ts_ + a1, lambda: ts_ + a2),
            ('T+D', lambda: a1 + d1, lambda: a2 + d2), ('D+T', lambda: d1 + a1, lambda: d2 + a2)]:
        r1, r2 = outcome(f1), outcome(f2)
        check((lab, sa, sd, x), res(r1, dcanon1), res(r2, dcanon2))
        if r2[0] == 'ok':
            check((lab + ' v2 type', sa), type(r2[1]).__name__, 'DateTimeInterval')
    # C. duration - instant (v1: a DateTimeInterval; v2: dropped)
    r1, r2 = outcome(lambda: a1 - dt_), outcome(lambda: a2 - dt_)
    check(('T-dt v1 outcome', bool(sa)), (r1[0], type(r1[1]).__name__ if r1[0] == 'ok' else r1[1]), ('ok', 'DateTimeInterval'))
    check(('T-dt v2', sa), r2[:2], ('raise', 'TypeError'))
    # D. scaling
    k = rng.choice([0, 1, 2, -1, -3, Fraction(1, 2), Fraction(-3, 4)])
    for lab, f1, f2 in [('T*k', lambda: a1 * k, lambda: a2 * k), ('k*T', lambda: k * a1, lambda: k * a2)]:
        check((lab, sa, k), res(outcome(f1), canon1), res(outcome(f2), canon2))
    if k != 0:
        check(('T/k', sa, k), res(outcome(lambda: a1 / k), canon1), res(outcome(lambda: a2 / k), canon2))
    for k in (np.int64(3), np.float64(0.5)):
        check(('T*np', sa, k), res(outcome(lambda: a1 * k), canon1), res(outcome(lambda: a2 * k), canon2))
        check(('np*T', sa, k), res(outcome(lambda: k * a1), canon1), res(outcome(lambda: k * a2), canon2))
    # float factor that is not dyadic: v1 float, v2 exact. compare within 1e-9 s
    f = rng.choice([0.1, 0.3, -0.7, 1.1])
    r1, r2 = outcome(lambda: a1 * f), outcome(lambda: a2 * f)
    if r1[0] == r2[0] == 'ok':
        c1, c2 = canon1(r1[1], exact=True), canon2(r2[1])
        close = len(c1) == len(c2) and all(
            p[1] == q[1] and p[3] == q[3] and abs(p[0] - q[0]) < Fraction(1, 10 ** 9) and abs(p[2] - q[2]) < Fraction(1, 10 ** 9)
            for p, q in zip(c1, c2))
        check(('T*float', sa, f), close, True)
        # v2 exactly f * the input ends
        ends_in = sorted({p[0] for p in canon2(a2)} | {p[2] for p in canon2(a2)})
        ends_out = sorted({q[0] for q in c2} | {q[2] for q in c2})
        check(('T*float v2 exact', sa, f), set(ends_out) <= {Fraction(f) * e for e in ends_in}, True)
    else:
        check(('T*float outcome', sa, f), r1[:2], r2[:2])

from collections import Counter
print(Counter(f[0][0] for f in FAILS))
seen = set()
for fl in FAILS:
    if fl[0][0] not in seen:
        seen.add(fl[0][0])
        print('  first', fl)
H = dt.timedelta(hours=1)
A1, A2 = T1(H, 2 * H), T2(H, 2 * H)
print('--- hand-picked')
for lab, f1, f2 in [
        ('T + date', lambda: str(A1 + dt.date(2024, 1, 1)), lambda: str(A2 + dt.date(2024, 1, 1))),
        ('date + T', lambda: str(dt.date(2024, 1, 1) + A1), lambda: str(dt.date(2024, 1, 1) + A2)),
        ('T - dt', lambda: str(A1 - BASE), lambda: A2 - BASE),
        ('dt - T', lambda: str(BASE - A1), lambda: str(BASE - A2)),
        ('T * True', lambda: str(A1 * True), lambda: A2 * True),
        ('T * MultiInterval', lambda: str(A1 * v1m.MultiInterval(1, 2)), lambda: str(A2 * M2(1, 2))),
        ('T / MultiInterval', lambda: str(A1 / v1m.MultiInterval(1, 2)), lambda: str(A2 / M2(1, 2))),
        ('T * T', lambda: A1 * A1, lambda: A2 * A2),
        ('T * td', lambda: A1 * H, lambda: A2 * H),
        ('T / td', lambda: A1 / H, lambda: repr(A2 / H)),
        ('T / 0', lambda: A1 / 0, lambda: A2 / 0),
        ('(T / 0).infimum', lambda: (A1 / 0).infimum, lambda: (A2 / 0).is_empty),
        ('str(T / 0)', lambda: str(A1 / 0), lambda: str(A2 / 0)),
        ('T / 0.0', lambda: A1 / 0.0, lambda: A2 / 0.0),
        ('T / 3 str', lambda: str(A1 / 3), lambda: str(A2 / 3)),
        ('(T/3).supremum', lambda: (A1 / 3).supremum, lambda: (A2 / 3).sup),
        ('T * 0.1 inf', lambda: (A1 * 0.1).infimum, lambda: (A2 * 0.1).inf),
        ('T + 5', lambda: A1 + 5, lambda: A2 + 5),
        ('T + "1h"', lambda: A1 + '1h', lambda: A2 + '1h'),
        ('T + NaT', lambda: A1 + pd.NaT, lambda: A2 + pd.NaT),
        ('T * nan', lambda: str(A1 * float('nan')), lambda: A2 * float('nan')),
        ('T * inf', lambda: str(A1 * float('inf')), lambda: str(A2 * float('inf'))),
        ('empty + td', lambda: str(T1() + H), lambda: str(T2() + H)),
        ('td + empty', lambda: str(H + T1()), lambda: str(H + T2())),
        ('T + empty', lambda: str(A1 + T1()), lambda: str(A2 + T2())),
        ('empty * 2', lambda: str(T1() * 2), lambda: str(T2() * 2)),
        ('T + np.timedelta64', lambda: A1 + np.timedelta64(1, 'h'), lambda: A2 + np.timedelta64(1, 'h')),
        ('T * np.timedelta64', lambda: A1 * np.timedelta64(3, 'ns'), lambda: A2 * np.timedelta64(3, 'ns')),
        ('1us + 10**6 days', lambda: str(T1(dt.timedelta(microseconds=1)) + dt.timedelta(days=10 ** 6)),
         lambda: str(T2(dt.timedelta(microseconds=1)) + dt.timedelta(days=10 ** 6))),
        ('(10**6 days + 1us).inf', lambda: (T1(dt.timedelta(days=10 ** 6)) + dt.timedelta(microseconds=1)).infimum,
         lambda: (T2(dt.timedelta(days=10 ** 6)) + dt.timedelta(microseconds=1)).inf),
        ('(1e9 s + 1us) - 1e9 s', lambda: ((T1(dt.timedelta(seconds=10 ** 9)) + dt.timedelta(microseconds=1)) - dt.timedelta(seconds=10 ** 9)).infimum,
         lambda: ((T2(dt.timedelta(seconds=10 ** 9)) + dt.timedelta(microseconds=1)) - dt.timedelta(seconds=10 ** 9)).inf),
]:
    print(f'{lab}: v1 {outcome(f1)} | v2 {outcome(f2)}')
assert not check('sabotage', canon1(A1 + H), canon2(A2 + 2 * H))
FAILS.pop()
report('probe_arith')
print('=== triage')
def empty_operand(f):
    lab = f[0]
    return (lab[1] == [] or (len(lab) > 2 and isinstance(lab[2], list) and lab[2] == [])) and f[1][:2] == ('raise', 'ValueError')
def close_values(f):
    a, b = f[1], f[2]
    return isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and all(
        isinstance(p, tuple) and len(p) == 4 and p[1] == q[1] and p[3] == q[3] and abs(p[0] - q[0]) <= Fraction(1, US) and abs(p[2] - q[2]) <= Fraction(1, US) for p, q in zip(a, b))
rest = [f for f in FAILS if not empty_operand(f) and not close_values(f)]
print('empty-operand v1 ValueError:', sum(1 for f in FAILS if empty_operand(f)), Counter(f[0][0] for f in FAILS if empty_operand(f)))
print('within 1us (v1 float vs v2 exact):', sum(1 for f in FAILS if not empty_operand(f) and close_values(f)), Counter(f[0][0] for f in FAILS if not empty_operand(f) and close_values(f)))
print('remaining:', len(rest), Counter(f[0][0] for f in rest))
seen = set()
for fl in rest:
    if fl[0][0] not in seen:
        seen.add(fl[0][0]); print('  R', fl)
