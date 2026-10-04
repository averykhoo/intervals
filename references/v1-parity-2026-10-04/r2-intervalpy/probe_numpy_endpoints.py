"""numpy nan / inf / narrow-float scalars as endpoints: v1 Interval.__post_init__ vs the v2 constructor"""
import random
import warnings
from collections import Counter
import numpy as np
warnings.simplefilter('ignore', RuntimeWarning)

from common import *  # noqa

NANS = [float('nan'), np.nan, np.float64('nan'), np.float32('nan'), np.float16('nan'), np.longdouble('nan'),
        -np.float64('nan')]
POS_INFS = [math.inf, np.inf, np.float64(np.inf), np.float32('inf'), np.float16('inf'), np.longdouble('inf')]
NEG_INFS = [-x for x in POS_INFS]


def run(ctor, *a, **k):
    try:
        return 'ok', ctor(*a, **k)
    except Exception as ex:  # noqa
        return type(ex).__name__, str(ex)


print('== nan endpoints (start, then end; every flag pair)')
for x in NANS:
    for so in (False, True):
        for ec in (False, True):
            for where in ('start', 'end'):
                s, e = (x, 1) if where == 'start' else (0, x)
                r1 = run(V1, s, so, e, ec)
                r2 = run(MI, s, e, start_closed=not so, end_closed=ec)
                check('nan-v1-raises', r1[0] == 'ValueError', (type(x).__name__, where, r1))
                check('nan-v2-raises', r2[0] == 'ValueError', (type(x).__name__, where, r2))
    r2p = run(MI, x)  # point
    check('nan-v2-point', r2p[0] == 'ValueError', r2p)
    print(f'  {type(x).__name__:10} v1: {run(V1, x, False, 1, True)}  v2: {run(MI, x, 1)}  v2 point: {r2p}')
# nan both ends
print('  both nan: v1', run(V1, np.float32('nan'), False, np.float32('nan'), True), ' v2', run(MI, np.float32('nan'), np.float32('nan')))

print('== infinite endpoints, open (v1-legal)')
pts = [-math.inf, -1e308, -5, -1, 0, Fraction(1, 3), 1, 5, 1e308, math.inf, 10 ** 400, Fraction(-10 ** 400, 3)]


def ex_(x):
    """exact value of a real (numpy scalars through the double they hold; longdouble is a double on win64)"""
    if isinstance(x, (int, Fraction)) and not isinstance(x, bool):
        return Fraction(x)
    assert np.finfo(np.longdouble).nmant == 52
    return Fraction(float(x))


def mem(obj, q):
    try:
        return q in obj
    except Exception as ex:  # noqa
        return type(ex).__name__


def truth_of(lo, lo_closed, hi, hi_closed, q):
    if math.isinf(q) if isinstance(q, float) else False:
        return (q == lo and lo_closed) or (q == hi and hi_closed)
    L = -math.inf if math.isinf(lo) else ex_(lo)
    H = math.inf if math.isinf(hi) else ex_(hi)
    Q = ex_(q)
    return (L < Q or (lo_closed and L == Q)) and (Q < H or (hi_closed and Q == H))


INF_STATS = Counter()
INF_V1_WRONG = []
for ni in NEG_INFS:
    for pi in POS_INFS:
        cases = [(ni, pi, True, False)]
        for fin in (-1, 0, Fraction(1, 3), 2.5, np.float32(2.5), np.int64(4)):
            cases += [(ni, fin, True, True), (fin, pi, False, False)]
        for s_, e_, so, ec in cases:
            r1 = run(V1, s_, so, e_, ec)
            r2 = run(MI, s_, e_, start_closed=not so, end_closed=ec)
            check('inf-both-ok', r1[0] == 'ok' and r2[0] == 'ok', (s_, e_, r1, r2))
            if r1[0] != 'ok' or r2[0] != 'ok':
                continue
            for q in pts:
                t = truth_of(s_, not so, e_, ec, q)
                a, b = mem(r1[1], q), mem(r2[1], q)
                check('inf-v2-exact', b == t, (type(s_).__name__, s_, type(e_).__name__, e_, q, b, t))
                if a == t:
                    INF_STATS['v1 right'] += 1
                else:
                    INF_STATS['v1 wrong/raises'] += 1
                    INF_V1_WRONG.append((type(s_).__name__, str(s_), type(e_).__name__, str(e_), so, ec, q if not isinstance(q, int) or abs(q) < 1e300 else '10**400', a, t))
print('  membership vs exact truth, v1:', dict(INF_STATS))
seen = set()
for w in INF_V1_WRONG:
    key = (w[0], w[2], str(w[6])[:12], w[7])
    if key in seen:
        continue
    seen.add(key)
    if len(seen) <= 14:
        print('    v1', w)
print('  open infinities of every numpy type: compared (mismatches below)')
# v2 stores a numpy inf as math.inf? (type of the cut value)
A = MI(np.float32('-inf'), np.float16('inf'), start_closed=False, end_closed=False)
print('  v2 stored types:', [type(c.value).__name__ for c in A.cuts], repr(A),
      ' v1 stored:', type(V1(np.float32('-inf'), True, np.float16('inf'), False).start).__name__)

print('== infinite endpoints, closed or reversed (v1 refuses)')
for x in (np.float32('inf'), np.float64('-inf'), np.longdouble('inf')):
    for args in ((x, False, x, True), (-abs(x), False, 0, True), (0, False, abs(x), True), (abs(x), True, abs(x), False),
                 (-abs(x), True, -abs(x), False)):
        r1 = run(V1, *args)
        s, so, e, ec = args
        r2 = run(MI, s, e, start_closed=not so, end_closed=ec)
        print(f'  {type(x).__name__:9} {str(args):60} v1 {r1[0]:10} {"" if r1[0] == "ok" else r1[1]:45} v2 {r2[0]} {r2[1] if r2[0] != "ok" else repr(r2[1])}')

print('== narrow floats and numpy ints as endpoints (exact meaning)')
f32 = np.float32(0.1)
exact = Fraction(float(f32))   # 0.100000001490116119384765625
iv = V1(f32, False, 1, True)
A = MI(f32, 1)
for q in (0.1, Fraction(1, 10), float(f32), exact, exact - Fraction(1, 10 ** 20), np.float32(0.1), np.float64(0.1)):
    truth = exact <= ex_(q) <= 1
    try:
        m1 = q in iv
    except Exception as ex:  # noqa
        m1 = type(ex).__name__
    m2 = q in A
    print(f'  {type(q).__name__:9} {str(q):32} exact {truth!s:5} v1 {m1!s:5} v2 {m2!s:5}')
    check('f32-v2-exact', m2 == truth, q)
# seeded sweep: random float32/float16/longdouble/int64 endpoints, membership of exact neighbours
rng = random.Random(3)
st = {'agree': 0, 'v1 wrong': 0, 'v2 wrong': 0}
v1_wrong_examples = []
for _ in range(400):
    t = rng.choice([np.float32, np.float16, np.float64, np.longdouble, np.int64, np.int16])
    raw = sorted([rng.uniform(-100, 100), rng.uniform(-100, 100)])
    lo, hi = t(raw[0]), t(raw[1])
    if not lo <= hi:
        continue
    so, ec = rng.random() < .5, rng.random() < .5
    if lo == hi:
        so, ec = False, True
    r1 = run(V1, lo, so, hi, ec)
    r2 = run(MI, lo, hi, start_closed=not so, end_closed=ec)
    if r1[0] != 'ok' or r2[0] != 'ok':
        check('sweep-build', r1[0] == r2[0], (t, lo, hi, r1, r2))
        continue
    L, H = ex_(lo), ex_(hi)
    for q in (L, H, L - Fraction(1, 10 ** 12), H + Fraction(1, 10 ** 12), (L + H) / 2, float(L), float(H), float(L) - 1e-9):
        Q = ex_(q)
        truth = (L < Q or (not so and L == Q)) and (Q < H or (ec and Q == H))
        a, b = mem(r1[1], q), q in r2[1]
        if b != truth:
            st['v2 wrong'] += 1
            check('sweep-v2', False, (t, lo, hi, so, ec, q))
        elif a != truth:
            st['v1 wrong'] += 1
            v1_wrong_examples.append((t.__name__, float(lo), float(hi), so, ec, q, a, truth))
        else:
            st['agree'] += 1
print('  sweep (400 random numpy-typed intervals x 8 exact probes):', st)
for x in v1_wrong_examples[:6]:
    print('    v1 wrong:', x)

print('== nan as a membership query')
for n in (np.float64('nan'), np.float32('nan'), float('nan')):
    print(f'  {type(n).__name__:8} v1 {run(lambda: n in V1(0, False, 1, True))}  v2 {run(lambda: n in MI(0, 1))}')

print('== other inputs: bool and np.bool_ endpoints; non-bool flags')
for args, kw in (((True, False, 2, True), {}), ((np.bool_(True), False, 2, True), {}), ((0, np.bool_(False), 1, True), {}),
                 ((0, 0, 1, 1), {}), ((0, 'no', 1, 'no'), {})):
    r1 = run(V1, *args)
    s, so, e, ec = args
    r2 = run(MI, s, e, start_closed=(not so) if isinstance(so, (bool, np.bool_)) else so, end_closed=ec)
    print(f'  {str(args):40} v1 {r1[0]} {r1[1] if r1[0] == "ok" else r1[1]!s:30} v2 {r2[0]} {r2[1]!r}')

before = len(MISMATCHES)
check('SABOTAGE', (0.1 in MI(np.float32(0.1), 1)) is True)
check('SABOTAGE2', run(MI, np.float16('nan'), 1)[0] == 'ok')
print('sabotage caught', len(MISMATCHES) - before, 'of 2'); del MISMATCHES[before:]
report('probe_numpy_endpoints')
