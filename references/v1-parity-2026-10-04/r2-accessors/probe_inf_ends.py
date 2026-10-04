"""probe (a): accessors on sets with closed +-inf ends / [+-inf] points; v1 with INFINITY_IS_NOT_FINITE=False"""
import sys, math, random, warnings, itertools
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1
v1.INFINITY_IS_NOT_FINITE = False
import intervals as v2
from fractions import Fraction
INF = math.inf
SABOTAGE = '--sabotage' in sys.argv

def v1_pieces(m):
    e = m.endpoints
    return [(e[i][0], e[i][1] == 0, e[i+1][0], e[i+1][1] == 0) for i in range(0, len(e), 2)]

def v2_pieces(m):
    out = []
    for p in m:
        out.append((p.inf, p.inf_closed, p.sup, p.sup_closed))
    return out

def mk(pieces):
    a = v1.MultiInterval.merge(*[v1.MultiInterval(lo, hi, start_closed=lc, end_closed=hc) if lo != hi
                                 else v1.MultiInterval(lo) for lo, lc, hi, hc in pieces]) if pieces else v1.MultiInterval()
    b = v2.MultiInterval.from_pieces([(lo, hi, lc, hc) for lo, lc, hi, hc in pieces])
    return a, b

def get(f):
    try:
        return ('ok', f())
    except Exception as ex:
        return ('raise', type(ex).__name__)

stats = {}
diffs = {}
def record(cap, same, case, a, b):
    stats.setdefault(cap, [0, 0])
    stats[cap][0 if same else 1] += 1
    if not same:
        diffs.setdefault(cap, []).append((case, a, b))

def compare(case, a, b):
    record('construct-same-set', v1_pieces(a) == v2_pieces(b), case, v1_pieces(a), v2_pieces(b))
    for c1, c2 in [('infimum', 'inf'), ('supremum', 'sup'), ('infimum_is_closed', 'inf_closed'),
                   ('supremum_is_closed', 'sup_closed'), ('is_finite', 'is_finite'), ('is_empty', 'is_empty'),
                   ('is_contiguous', 'is_contiguous'), ('is_degenerate', 'is_degenerate'),
                   ('is_integral', 'is_integral'), ('is_positive', 'is_positive'), ('is_negative', 'is_negative'),
                   ('is_non_negative', 'is_non_negative'), ('is_non_positive', 'is_non_positive'),
                   ('degenerate_points', 'degenerate_points')]:
        r1 = get(lambda: getattr(a, c1)); r2 = get(lambda: getattr(b, c2))
        if r1[0] == 'raise' and r2[0] == 'raise':
            same = True  # KeyError vs ValueError on empty: documented
        else:
            same = r1 == r2 and (r1[0] != 'ok' or type(r1[1]) == type(r2[1]) or c1.startswith('is_') or c1 == 'degenerate_points')
            if SABOTAGE and c1 == 'supremum':
                same = r1 == ('ok', 12345)
        record(c1, same, case, r1, r2)
    for c1, c2 in [('closed_hull', 'closed_hull'), ('finite', 'finite'), ('positive', 'positive'), ('negative', 'negative')]:
        r1 = get(lambda: getattr(a, c1)); r2 = get(lambda: getattr(b, c2))
        p1 = ('ok', v1_pieces(r1[1]) if r1[1] is not None else None) if r1[0] == 'ok' else r1
        p2 = ('ok', v2_pieces(r2[1])) if r2[0] == 'ok' else r2
        if c1 == 'closed_hull' and b.is_empty:
            same = p1 == ('ok', None) and p2 == ('ok', [])   # v1 returns None on empty, v2 the empty set
        else:
            same = p1 == p2
        record(c1, same, case, p1, p2)
    # contiguous_intervals -> pieces
    r1 = get(lambda: [v1_pieces(x) for x in a.contiguous_intervals]); r2 = get(lambda: [v2_pieces(x) for x in b.pieces])
    record('contiguous_intervals', r1 == r2, case, r1, r2)
    # cardinality -> size  (v1 counts half-points)
    r1 = get(lambda: a.cardinality); r2 = get(lambda: tuple(b.size))
    if r1[0] == 'ok' and r2[0] == 'ok':
        rays_pts = r1[1][0] == r2[1][0] and r1[1][2] == 2 * r2[1][2]
        record('cardinality.rays+points', rays_pts, case, r1, r2)
        record('cardinality.length', r1[1][1] == r2[1][1], case, r1, r2)
    else:
        record('cardinality.rays+points', False, case, r1, r2)

hand = [
    [(-INF, True, INF, True)],
    [(-INF, False, INF, True)], [(-INF, True, INF, False)], [(-INF, False, INF, False)],
    [(1, False, INF, True)], [(1, True, INF, False)], [(1, True, INF, True)],
    [(-INF, True, 1, True)], [(-INF, False, 1, True)],
    [(INF, True, INF, True)], [(-INF, True, -INF, True)],
    [(-INF, True, -INF, True), (INF, True, INF, True)],
    [(-INF, True, -INF, True), (0, True, 0, True), (INF, True, INF, True)],
    [(-INF, True, -INF, True), (2, True, 3, False), (INF, True, INF, True)],
    [(-INF, True, 0, False), (0, False, INF, True)],
    [(-INF, True, -1, True), (1, True, INF, True)],
    [(0, True, 0, True), (INF, True, INF, True)],
    [(-INF, True, -INF, True), (Fraction(1, 3), True, 2.5, True)],
    [(5, True, 5, True), (INF, True, INF, True)],
    [(-INF, True, -INF, True), (-5, True, -5, True)],
    [],
]
for i, ps in enumerate(hand):
    a, b = mk(ps)
    compare(('hand', i, ps), a, b)

rng = random.Random(20261004)
vals = [-INF, -3, -2, -1, 0, 1, 2, 3, INF, Fraction(1, 2), 1.5]
n_rand = 0
for _ in range(600):
    k = rng.randint(0, 4)
    ps = []
    for _ in range(k):
        lo, hi = sorted(rng.sample(vals, 2), key=float)
        if rng.random() < 0.25:
            hi = lo
            ps.append((lo, True, lo, True)); continue
        lc, hc = rng.random() < 0.5, rng.random() < 0.5
        if lo == INF and not lc or hi == -INF and not hc:
            continue
        ps.append((lo, lc, hi, hc))
    if not any(math.isinf(float(p[0])) or math.isinf(float(p[2])) for p in ps) and rng.random() < 0.7:
        continue
    try:
        a, b = mk(ps)
    except Exception as ex:
        print('construct raised', ps, repr(ex)); continue
    n_rand += 1
    compare(('rand', ps), a, b)

print('random cases', n_rand)
for cap, (s, d) in stats.items():
    print(f'{cap:28s} same={s} diff={d}')
for cap, lst in diffs.items():
    print('==', cap, len(lst))
    for x in lst[:6]:
        print('   ', x)

# classify the cardinality.length diffs: every one must involve a ray piece, and v1's length must be non-finite there
bad = []
for case, r1, r2 in diffs.get('cardinality.length', []):
    ps = case[-1]
    has_ray = any(lo != hi and (math.isinf(float(lo)) or math.isinf(float(hi))) for lo, lc, hi, hc in ps)
    if not has_ray or math.isfinite(r1[1][1]):
        bad.append((case, r1, r2))
print('cardinality.length diffs without a ray piece or with finite v1 length:', len(bad), bad[:5])
nray = sum(1 for case, *_ in [(d[0],) for d in diffs.get('cardinality.length', [])])
# do ray-free sets ever differ? count ray-carrying cases that AGREE (v1 right by luck?)
