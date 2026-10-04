"""verify (accessors r2-17): v1 accessors vs v2 on WIDE np.longdouble ends (x86-64 linux, 64-bit significand).
run from the repo root. `--sabotage` swaps the exact oracle for float() of the end, which must go red on a wide build."""
import sys, math, random, warnings
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
import intervals as v2
from fractions import Fraction
import numpy as np
import platform

SABOTAGE = '--sabotage' in sys.argv
LD = np.longdouble
fi = np.finfo(LD)
print('platform', platform.machine(), platform.system(), 'python', sys.version.split()[0], 'numpy', np.__version__)
print('longdouble nmant', fi.nmant, 'max', fi.max)
WIDE = fi.nmant > 52
print('WIDE build:', WIDE)


def ex(x):
    """exact value of an end: int, Fraction, or +-inf (float)"""
    if isinstance(x, int):
        return x
    if isinstance(x, Fraction):
        return x
    if np.isinf(x):
        return math.inf if x > 0 else -math.inf
    if SABOTAGE:
        f = float(x)  # deliberately wrong oracle: rounds the long double
        return f if math.isinf(f) else Fraction(f)
    n, d = x.as_integer_ratio()
    return Fraction(int(n), int(d))


one = LD(1)
e60 = LD(2) ** -60
vals = {
    '1+2^-60': one + e60,
    '-(1+2^-60)': -(one + e60),
    '1-2^-63': one - LD(2) ** -63,
    'ld(0.1)': LD('0.1'),
    'ld(1/3)': one / LD(3),
    '2^64-1': LD(2) ** 64 - one,
    '2^63+0.5': LD(2) ** 62 + LD(0.5),
    '1e4000': LD('1e4000'),
    '-1e4000': LD('-1e4000'),
    '1e-4000': LD('1e-4000'),
    '0.5': LD(0.5),
    '3': LD(3),
}
for k, x in vals.items():
    print(f'  {k:12s} exact={ex(x) if abs(ex(x)) < 10**30 and abs(ex(x)) > Fraction(1, 10**30) else "<huge/tiny>"} '
          f'is_double={Fraction(float(x)) == ex(x) if np.isfinite(float(x)) else False}')

fails = []
LEN_SHOWN = 0
stats = {}


def check(cap, label, a, b):
    stats.setdefault(cap, [0, 0])
    ok = a == b
    stats[cap][0 if ok else 1] += 1
    if not ok:
        fails.append((cap, label, a, b))


def get(f):
    try:
        return ('ok', f())
    except Exception as e:
        return ('raise', type(e).__name__, str(e)[:70])


def v1_pieces(m):
    e = m.endpoints
    return [(ex(e[i][0]), e[i][1] == 0, ex(e[i + 1][0]), e[i + 1][1] == 0) for i in range(0, len(e), 2)]


def noinf(ps):
    # v1 flag-on leaves +-inf open in closed_hull: documented (round-1 DIFFERS_DOCUMENTED); compare finite ends only
    return [(lo, lc if not math.isinf(lo) else None, hi, hc if not math.isinf(hi) else None) for lo, lc, hi, hc in ps]


def v2_pieces(m):
    return [(ex(p.inf), p.inf_closed, ex(p.sup), p.sup_closed) for p in m]


def accessors(a, b, label):
    """compare every v1 accessor with v2's spelling, exactly"""
    pairs = [
        ('infimum', lambda: ex(a.infimum), lambda: ex(b.inf)),
        ('supremum', lambda: ex(a.supremum), lambda: ex(b.sup)),
        ('infimum_is_closed', lambda: a.infimum_is_closed, lambda: b.inf_closed),
        ('supremum_is_closed', lambda: a.supremum_is_closed, lambda: b.sup_closed),
        ('is_empty', lambda: a.is_empty, lambda: b.is_empty),
        ('is_contiguous', lambda: a.is_contiguous, lambda: b.is_contiguous),
        ('is_degenerate', lambda: a.is_degenerate, lambda: b.is_degenerate),
        ('is_finite', lambda: a.is_finite, lambda: b.is_finite),
        ('is_integral', lambda: a.is_integral, lambda: b.is_integral),
        ('is_positive', lambda: a.is_positive, lambda: b.is_positive),
        ('is_negative', lambda: a.is_negative, lambda: b.is_negative),
        ('is_non_negative', lambda: a.is_non_negative, lambda: b.is_non_negative),
        ('is_non_positive', lambda: a.is_non_positive, lambda: b.is_non_positive),
        ('degenerate_points', lambda: sorted(ex(p) for p in a.degenerate_points),
         lambda: sorted(ex(p) for p in b.degenerate_points)),
        ('finite', lambda: v1_pieces(a.finite), lambda: v2_pieces(b.finite)),
        ('positive', lambda: v1_pieces(a.positive), lambda: v2_pieces(b.positive)),
        ('negative', lambda: v1_pieces(a.negative), lambda: v2_pieces(b.negative)),
        ('closed_hull', lambda: noinf(v1_pieces(a.closed_hull)) if a.closed_hull is not None else [], lambda: noinf(v2_pieces(b.closed_hull))),
        ('contiguous_intervals', lambda: [v1_pieces(p) for p in a.contiguous_intervals],
         lambda: [v2_pieces(p) for p in b.pieces]),
        ('cardinality.rays+points', lambda: (a.cardinality[0], a.cardinality[2]),
         lambda: (b.size.rays, 2 * b.size.points)),
    ]
    for cap, f1, f2 in pairs:
        if a.is_empty and cap in ('infimum', 'supremum', 'infimum_is_closed', 'supremum_is_closed'):
            continue
        check(cap, label, get(f1), get(f2))
    # cardinality length: v1's own number vs exact length (v1 ray bug is documented; skip rays)
    if not a.is_empty and not any(math.isinf(ex(e[0])) for e in a.endpoints):
        global LEN_SHOWN
        exact_len = sum((ex(a.endpoints[i + 1][0]) - ex(a.endpoints[i][0]) for i in range(0, len(a.endpoints), 2)), 0)
        r1 = get(lambda: ex(a.cardinality[1]) if not isinstance(a.cardinality[1], (int, Fraction)) else a.cardinality[1])
        r2 = get(lambda: ex(b.size.length) if not isinstance(b.size.length, (int, Fraction)) else b.size.length)
        if not any(isinstance(p.inf, float) or isinstance(p.sup, float) for p in b):
            check('cardinality.length vs exact (v2, no float end)', label, ('ok', exact_len), r2)
        stats.setdefault('cardinality.length v1 == exact', [0, 0])
        stats['cardinality.length v1 == exact'][0 if r1 == ('ok', exact_len) else 1] += 1
        if r1 != ('ok', exact_len) and LEN_SHOWN < 3:
            LEN_SHOWN += 1
            print('  v1 length inexact:', label, v1_pieces(a), 'v1', a.cardinality[1], type(a.cardinality[1]).__name__,
                  ex(a.cardinality[1]) if not isinstance(a.cardinality[1], (int, Fraction)) else '', 'exact', exact_len, 'v2', b.size.length)


def membership(b, pieces, label):
    """brute force: v2 holds each end exactly as asked, and a long double just inside/outside"""
    if len(pieces) != 1:
        return
    for lo, lc, hi, hc in pieces:
        for x, want in ((lo, lc), (hi, hc)):
            if math.isinf(ex(x)):
                continue
            check('membership at end', label, ('ok', want), get(lambda: ex(x) in b))
        if ex(lo) < ex(hi) and not math.isinf(ex(lo)) and not math.isinf(ex(hi)):
            mid = (ex(lo) + ex(hi)) / 2
            check('membership mid', label, ('ok', True), get(lambda: mid in b))
        if not math.isinf(ex(lo)):
            below = np.nextafter(lo, LD(-np.inf))
            # just below lo is outside unless an earlier piece covers it (single-piece sets only here)
            if len(pieces) == 1:
                check('membership just below', label, ('ok', False), get(lambda: ex(below) in b))


def build(pieces):
    pieces = [q for q in pieces if ex(q[0]) != ex(q[2]) or (q[1] and q[3])]
    a = v1.MultiInterval.merge(*[v1.MultiInterval(lo, hi, start_closed=lc, end_closed=hc)
                                 if ex(lo) != ex(hi) else v1.MultiInterval(lo) for lo, lc, hi, hc in pieces])
    b = v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in pieces])
    return a, b


# hand cases
hand = [
    [(one, True, one + e60, False)],
    [(one, True, one + e60, True)],
    [(one + e60, True, one + e60, True)],
    [(-(one + e60), False, -one, True)],
    [(LD('0.1'), True, LD(1), True)],
    [(LD(0), True, one / LD(3), False)],
    [(LD(2) ** 64 - one, True, LD(2) ** 64 - one, True)],  # integral wide point
    [(LD(2) ** 62 + LD(0.5), True, LD(2) ** 62 + LD(0.5), True)],  # non-integral wide point (a double: 2^62)
    [(LD('-1e4000'), True, LD('1e4000'), True)],
    [(LD('1e4000'), True, LD('1e4000'), True)],
    [(LD(0), False, LD('1e-4000'), True)],
    [(-(one + e60), True, -one, False), (one, True, one + e60, True)],
    [(one, True, one + e60, False), (one + e60, True, LD(2), True)],  # touching: contiguous [1,2]
    [(one, True, one + e60, False), (one + LD(2) ** -59, True, LD(2), True)],  # gap of 2^-60
    [(LD(-np.inf), False, one + e60, True)],
    [(one - LD(2) ** -63, False, LD(np.inf), False)],
]
construct_raise = {'v1': [], 'v2': []}
for i, pieces in enumerate(hand):
    label = ('hand', i)
    pieces = [q for q in pieces if ex(q[0]) != ex(q[2]) or (q[1] and q[3])]
    try:
        a, b = build(pieces)
    except Exception as e:
        try:
            v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in pieces])
            construct_raise['v1'].append((label, type(e).__name__, str(e)[:70]))
        except Exception as e2:
            construct_raise['v2'].append((label, type(e2).__name__, str(e2)[:70]))
        continue
    check('construct (exact pieces)', label, v1_pieces(a), v2_pieces(b))
    accessors(a, b, label)
    membership(b, v2_pieces(b) and [(p.inf, p.inf_closed, p.sup, p.sup_closed) for p in b] and pieces, label)

# seeded sweep: ends drawn from a grid of long doubles that are mostly NOT doubles
rng = random.Random(20261004)
grid = sorted({ex(v): v for v in [one + LD(k) * e60 for k in range(-3, 4)] + [LD(k) / LD(3) for k in range(-4, 5)]
               + [LD('0.1') * LD(k) for k in range(-3, 4)] + [LD(2) ** 64 + LD(k) for k in (-3, -1, 0)]
               + [LD(-np.inf), LD(np.inf)]}.items())
grid = [v for _, v in grid]
for n in range(400):
    k = rng.randint(1, 3)
    idx = sorted(rng.sample(range(len(grid)), 2 * k))
    pieces = []
    for j in range(k):
        lo, hi = grid[idx[2 * j]], grid[idx[2 * j + 1]]
        lc = rng.random() < 0.5 and not np.isinf(lo)
        hc = rng.random() < 0.5 and not np.isinf(hi)
        pieces.append((lo, lc, hi, hc))
    if rng.random() < 0.2:
        p = grid[rng.randrange(1, len(grid) - 1)]
        pieces.append((p, True, p, True))
    label = ('rand', n)
    try:
        a, b = build(pieces)
    except Exception as e:
        construct_raise['v1'].append((label, type(e).__name__, str(e)[:70]))
        continue
    check('construct (exact pieces)', label, v1_pieces(a), v2_pieces(b))
    accessors(a, b, label)

print('construct raised: v1', len(construct_raise['v1']), 'v2', len(construct_raise['v2']))
for r in construct_raise['v1'][:10]:
    print('   v1 raise', r)
for r in construct_raise['v2'][:10]:
    print('   v2 raise', r)
print('per capability (same, diff):')
for cap, (s, d) in stats.items():
    print(f'  {cap:36s} same={s} diff={d}')
print('first diffs:')
seen = {}
for cap, label, a, b in fails:
    seen[cap] = seen.get(cap, 0) + 1
    if seen[cap] <= 3:
        print('  ', cap, label, '\n      v1/oracle:', str(a)[:200], '\n      v2       :', str(b)[:200])
print('TOTAL diffs', len(fails))
# type of what v2 hands back for a wide end
b = v2.MultiInterval(one, one + e60)
print('v2 sup type/value:', type(b.sup).__name__, b.sup, '== exact', b.sup == ex(one + e60), '; float(end) =', float(one + e60))
a = v1.MultiInterval(one, one + e60)
print('v1 supremum type/value:', type(a.supremum).__name__, repr(a.supremum))
print('v2 repr:', repr(b))
# v2 membership of the raw long double (not its Fraction), and v1's
c = v2.MultiInterval.from_pieces([(one, one + e60, True, False)])
print('v2 raw ld: 1+2^-60 in [1, 1+2^-60):', get(lambda: (one + e60) in c), '; 1+2^-61 in it:', get(lambda: (one + LD(2) ** -61) in c), '; c.sup exact:', get(lambda: c.sup == ex(one + e60)), repr(c))
d = v1.MultiInterval(one, one + e60, end_closed=False)
print('v1 raw ld: 1+2^-60 in [1, 1+2^-60):', get(lambda: (one + e60) in d), '; 1+2^-61 in it:', get(lambda: (one + LD(2) ** -61) in d))

# the finite long doubles past the doubles: v1 refuses them (math.isinf(float(x))), v2 keeps them exact
big = LD('1e4000')
print('float(ld 1e4000) =', float(big), '; math.isinf ->', math.isinf(big), '; np.isinf ->', np.isinf(big))
B = get(lambda: v2.MultiInterval(-big, big))
print('v2 [-1e4000, 1e4000]:', B[0], 'inf == exact', get(lambda: B[1].inf == ex(-big)), 'is_finite', get(lambda: B[1].is_finite),
      'type', get(lambda: type(B[1].sup).__name__), '1e300*10**3000 in it', get(lambda: 10 ** 3999 in B[1]))
print('v1 [-1e4000, 1e4000]:', get(lambda: v1.MultiInterval(-big, big)))
print('v1 [1e4000]:', get(lambda: v1.MultiInterval(big)))
print('v2 [1e4000]:', get(lambda: v2.MultiInterval(big).is_degenerate))
