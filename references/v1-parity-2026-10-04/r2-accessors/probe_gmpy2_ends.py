"""probe (b): every v1 accessor on gmpy2 mpz/mpq/mpfr ends (incl. 200-bit mpfr) and np.longdouble ends, mixed with
int/Fraction/float. v1 with INFINITY_IS_NOT_FINITE = True unless --flag-off."""
import sys, math, random, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
FLAG_OFF = '--flag-off' in sys.argv
v1.INFINITY_IS_NOT_FINITE = not FLAG_OFF
import intervals as v2
from fractions import Fraction
import numpy as np
import gmpy2
from gmpy2 import mpz, mpq, mpfr
INF = math.inf
SABOTAGE = '--sabotage' in sys.argv
MPZ, MPQ = type(mpz(0)), type(mpq(1, 2))


def ex(x):
    """exact value: an int, a Fraction, or +-inf"""
    if isinstance(x, (int, MPZ)):
        return int(x)
    if isinstance(x, (Fraction, MPQ)):
        return Fraction(int(x.numerator), int(x.denominator))
    if math.isinf(float(x)):
        return float(x)
    n, d = x.as_integer_ratio()
    return Fraction(int(n), int(d))


def v1_pieces(m):
    e = m.endpoints
    return [(ex(e[i][0]), e[i][1] == 0, ex(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]


def v2_pieces(m):
    return [(ex(p.inf), p.inf_closed, ex(p.sup), p.sup_closed) for p in m]


def mk(pieces):
    a = v1.MultiInterval.merge(*[v1.MultiInterval(lo, hi, start_closed=lc, end_closed=hc) if not (ex(lo) == ex(hi))
                                 else v1.MultiInterval(lo) for lo, lc, hi, hc in pieces]) if pieces else v1.MultiInterval()
    b = v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in pieces])
    return a, b


def get(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__, str(e)[:60])


stats, diffs, types = {}, {}, {}


def record(cap, same, case, a, b):
    stats.setdefault(cap, [0, 0])
    stats[cap][0 if same else 1] += 1
    if not same:
        diffs.setdefault(cap, []).append((case, a, b))


def compare(case, a, b):
    record('construct-same-set', v1_pieces(a) == v2_pieces(b), case, v1_pieces(a), v2_pieces(b))
    for c1, c2 in [('infimum', 'inf'), ('supremum', 'sup')]:
        r1 = get(lambda: getattr(a, c1)); r2 = get(lambda: getattr(b, c2))
        if r1[0] == 'ok' and r2[0] == 'ok':
            same = ex(r1[1]) == ex(r2[1])
            if SABOTAGE and c1 == 'infimum':
                same = ex(r1[1]) == ex(r2[1]) + 1
            k = (c1, type(r1[1]).__name__, type(r2[1]).__name__)
            types[k] = types.get(k, 0) + 1
        else:
            same = r1[0] == r2[0] == 'raise'
        record(c1, same, case, r1, r2)
    for c1, c2 in [('infimum_is_closed', 'inf_closed'), ('supremum_is_closed', 'sup_closed'),
                   ('is_finite', 'is_finite'), ('is_empty', 'is_empty'), ('is_contiguous', 'is_contiguous'),
                   ('is_degenerate', 'is_degenerate'), ('is_integral', 'is_integral'), ('is_positive', 'is_positive'),
                   ('is_negative', 'is_negative'), ('is_non_negative', 'is_non_negative'),
                   ('is_non_positive', 'is_non_positive')]:
        r1 = get(lambda: getattr(a, c1)); r2 = get(lambda: getattr(b, c2))
        same = r1 == r2 or (r1[0] == r2[0] == 'raise')
        record(c1, same, case, r1, r2)
    r1 = get(lambda: {ex(x) for x in a.degenerate_points}); r2 = get(lambda: {ex(x) for x in b.degenerate_points})
    record('degenerate_points', r1 == r2, case, r1, r2)
    r1 = get(lambda: a.degenerate_points)
    if r1[0] == 'ok':
        for x in r1[1]:
            k = ('degenerate_points', type(x).__name__, None)
            types[k] = types.get(k, 0) + 1
    for c1, c2 in [('closed_hull', 'closed_hull'), ('finite', 'finite'), ('positive', 'positive'), ('negative', 'negative')]:
        r1 = get(lambda: getattr(a, c1)); r2 = get(lambda: getattr(b, c2))
        p1 = ('ok', v1_pieces(r1[1]) if r1[1] is not None else []) if r1[0] == 'ok' else r1
        p2 = ('ok', v2_pieces(r2[1])) if r2[0] == 'ok' else r2
        record(c1, p1 == p2, case, p1, p2)
    r1 = get(lambda: [v1_pieces(x) for x in a.contiguous_intervals]); r2 = get(lambda: [v2_pieces(x) for x in b.pieces])
    record('contiguous_intervals', r1 == r2, case, r1, r2)
    r1 = get(lambda: a.cardinality); r2 = get(lambda: tuple(b.size))
    if r1[0] == 'ok' and r2[0] == 'ok':
        record('cardinality.rays+points', r1[1][0] == r2[1][0] and r1[1][2] == 2 * r2[1][2], case, r1, r2)
        key = 'cardinality.length(ray)' if r2[1][0] > 0 else 'cardinality.length(no ray)'
        record(key, ex(r1[1][1]) == ex(r2[1][1]), case, (r1[1][1], type(r1[1][1]).__name__),
               (r2[1][1], type(r2[1][1]).__name__))
    else:
        record('cardinality', r1[0] == r2[0], case, r1, r2)


with gmpy2.local_context(precision=200):  # gmpy2 arithmetic rounds to the context precision
    wide = mpfr(1) + mpfr(2) ** -150
    negwide = -wide          # 1 + 2**-150, not a double
wide_tenth = mpfr('0.1', 200)                        # nearest 200-bit to 0.1, not a double
with gmpy2.local_context(precision=200):
    big = mpfr(2) ** 70 + 1                         # 2**70 + 1, an integer, not a double
pool = [mpz(-3), mpz(0), mpz(2), mpq(1, 3), mpq(-5, 2), mpq(4, 2), mpfr('0.1'), mpfr('1.5'), mpfr(-2),
        wide, negwide, wide_tenth, big, np.longdouble('0.1'), np.longdouble(2), np.longdouble(-1.25),
        -3, 0, 1, Fraction(1, 3), 0.5, 2.0]
for _w in (wide, negwide, wide_tenth, big):
    assert _w.precision == 200 and float(_w) != _w, ('not wide', _w)
if FLAG_OFF:
    pool += [mpfr('inf'), mpfr('-inf'), np.longdouble('inf'), -INF, INF]

hand = [
    [(wide, True, wide, True)], [(wide_tenth, True, wide_tenth, True)],
    [(mpq(1, 3), True, wide, False)], [(negwide, False, wide, True)],
    [(mpz(2), True, mpz(2), True)], [(mpq(4, 2), True, mpq(4, 2), True)], [(mpfr(-2), True, mpfr(-2), True)],
    [(np.longdouble('0.1'), True, np.longdouble(2), False)], [(mpz(0), False, mpfr('1.5'), True)],
    [(1, True, wide, True)],
    [(wide_tenth, True, mpfr('0.1'), True)] if ex(wide_tenth) < ex(mpfr('0.1')) else [(mpfr('0.1'), True, wide_tenth, True)],
    [(big, True, big, True)],
    [(mpz(0), True, mpz(0), True), (mpq(1, 3), True, mpq(1, 3), True)],
    [(mpq(1, 3), True, wide_tenth * 10, True)],
]
if FLAG_OFF:
    hand += [[(mpfr('-inf'), True, mpfr('inf'), True)], [(mpfr('inf'), True, mpfr('inf'), True)],
             [(wide, False, mpfr('inf'), True)], [(np.longdouble('-inf'), True, np.longdouble(0), True)]]
else:
    hand += [[(mpfr('-inf'), False, mpfr('inf'), False)], [(wide, False, mpfr('inf'), False)],
             [(np.longdouble('-inf'), False, np.longdouble(0), True)]]
for i, ps in enumerate(hand):
    try:
        a, b = mk(ps)
    except Exception as e:
        print('hand construct raised', i, ps, repr(e)); continue
    compare(('hand', i, ps), a, b)

rng = random.Random(4242)
n_rand = 0
v2_raise = v1_raise = 0
v1_raise_kinds = {}
for _ in range(500):
    ps = []
    for _ in range(rng.randint(1, 4)):
        lo, hi = sorted(rng.sample(pool, 2), key=ex)
        if ex(lo) == ex(hi) or rng.random() < 0.2:
            ps.append((lo, True, lo, True)); continue
        lc, hc = rng.random() < 0.5, rng.random() < 0.5
        if ex(lo) == INF and not lc or ex(hi) == -INF and not hc:
            continue
        ps.append((lo, lc, hi, hc))
    try:
        b = v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in ps])
    except Exception as e:
        print('V2 construct raised', ps, repr(e)); v2_raise += 1; continue
    try:
        a, b = mk(ps)
    except Exception as e:
        v1_raise += 1; v1_raise_kinds[str(e)[:70]] = v1_raise_kinds.get(str(e)[:70], 0) + 1
        if not any(isinstance(x, np.longdouble) for p_ in ps for x in (p_[0], p_[2])):
            print('v1 raised without a longdouble end', ps, repr(e))
        continue
    n_rand += 1
    compare(('rand', ps), a, b)

print('v1 construct raised', v1_raise, 'v2 construct raised', v2_raise)
for k, v in v1_raise_kinds.items():
    print('   v1 raise:', v, k)
print('flag_off', FLAG_OFF, 'random cases', n_rand, 'gmpy2 precision', gmpy2.get_context().precision)
for cap, (s, d) in stats.items():
    print(f'{cap:28s} same={s} diff={d}')
print('types (v1, v2):')
for k, v in sorted(types.items(), key=str):
    print('   ', k, v)
for cap, lst in diffs.items():
    print('==', cap, len(lst))
    for x in lst[:5]:
        print('   ', x)


def xkey(e):  # exact order: float() would merge values 2**-150 apart
    return (0, 0) if e == -INF else (2, 0) if e == INF else (1, e)


# ---- who is right? brute-force exact membership of the input pieces vs each side's structure ----
def member(pcs, x):
    return any((lo < x or (lo == x and lc)) and (x < hi or (x == hi and hc)) for lo, lc, hi, hc in pcs)


def truth_check(ps, got):
    """ps: input pieces (raw values); got: exact pieces of one side. True iff membership agrees at every probe point"""
    exact = [(ex(lo), lc, ex(hi), hc) for lo, lc, hi, hc in ps]
    ends = sorted({e for lo, _, hi, _ in exact for e in (lo, hi)} | {e for lo, _, hi, _ in got for e in (lo, hi)}, key=xkey)
    fin = [e for e in ends if not isinstance(e, float) or not math.isinf(e)]
    pts = list(ends) + [(a + b) / 2 for a, b in zip(fin, fin[1:])]
    if fin:
        pts += [fin[0] - 1, fin[-1] + 1]
    return all(member(exact, x) == member(got, x) for x in pts)


tally = {}
for cap in ('construct-same-set',):
    for case, p1, p2 in diffs.get(cap, []):
        ps = case[-1]
        k = (truth_check(ps, p1), truth_check(ps, p2))
        tally[k] = tally.get(k, 0) + 1
        if k != (False, True):
            print('UNEXPECTED', k, case, p1, p2)
print('construct diffs (v1 right, v2 right):', tally)
wide_involved = sum(1 for case, _, _ in diffs.get('construct-same-set', [])
                    if any(getattr(x, 'precision', 53) > 53 for p_ in case[-1] for x in (p_[0], p_[2])))
print('construct diffs with a >53-bit mpfr end:', wide_involved, 'of', len(diffs.get('construct-same-set', [])))
# accessor diffs on sets whose construction agreed (so not inherited from v1's merge)
agree_cases = {repr(c) for c, _, _ in diffs.get('construct-same-set', [])}
for cap, lst in diffs.items():
    if cap == 'construct-same-set':
        continue
    own = [(c, a, b) for c, a, b in lst if repr(c) not in agree_cases]
    if own:
        print('-- diff on a set both sides built alike:', cap, len(own))
        for c, a, b in own[:3]:
            print('     ', str((c, a, b))[:400])


# ---- in the construct diffs where both sides hold the same set, who answers is_contiguous / degenerate_points right? ----
def truth_contiguous_and_isolated(ps):
    exact = [(ex(lo), lc, ex(hi), hc) for lo, lc, hi, hc in ps]
    ends = sorted({e for lo, _, hi, _ in exact for e in (lo, hi)}, key=xkey)
    fin = [e for e in ends if not isinstance(e, float) or not math.isinf(e)]
    mids = [(a + b) / 2 for a, b in zip(fin, fin[1:])]
    outer = [fin[0] - 1, fin[-1] + 1] if fin else [0]  # a sample point between +-inf and the finite ends
    pts = sorted(set(ends) | set(mids) | set(outer), key=xkey)
    mem = [member(exact, x) for x in pts]
    first, last = mem.index(True), len(mem) - 1 - mem[::-1].index(True)
    contiguous = all(mem[first:last + 1])
    isolated = set()
    for i, x in enumerate(pts):  # an end is an isolated point iff it is a member and both neighbouring sample points are not
        if x in ends and mem[i] and (i == 0 or not mem[i - 1]) and (i == len(pts) - 1 or not mem[i + 1]):
            isolated.add(x)
    return contiguous, isolated


sc = {'v1 is_contiguous right': 0, 'v1 is_contiguous wrong': 0, 'v2 is_contiguous wrong': 0,
      'v1 degenerate_points wrong': 0, 'v2 degenerate_points wrong': 0, 'cases': 0}
for case, p1, p2 in diffs.get('construct-same-set', []):
    ps = case[-1]
    a, b = mk(ps)
    tc, ti = truth_contiguous_and_isolated(ps)
    sc['cases'] += 1
    try:
        sc['v1 is_contiguous right' if a.is_contiguous == tc else 'v1 is_contiguous wrong'] += 1
    except TypeError:
        sc['v1 raises'] = sc.get('v1 raises', 0) + 1; sc['v2 is_contiguous wrong'] += b.is_contiguous != tc; continue
    sc['v2 is_contiguous wrong'] += b.is_contiguous != tc
    sc['v1 degenerate_points wrong'] += {ex(x) for x in a.degenerate_points} != ti
    sc['v2 degenerate_points wrong'] += {ex(x) for x in b.degenerate_points} != ti
    if {ex(x) for x in b.degenerate_points} != ti:
        print('V2 DEGEN?', ps, b, {ex(x) for x in b.degenerate_points}, ti)
print('construct-diff cases vs brute-force truth:', sc)
