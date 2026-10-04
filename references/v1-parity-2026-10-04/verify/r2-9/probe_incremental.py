"""readme-r2-9: incremental insert cost/correctness, v1 vs v2 (own probe)."""
import sys, time, random, bisect
from fractions import Fraction
sys.path[:0] = ['.', 'archive/v1']
import multi_interval as v1
import compare as v1c
import intervals as v2
from intervals import kernel
M = v2.MultiInterval
print(sys.version)

def bisect_insert(mi, lo, hi, lo_closed=True, hi_closed=True):
    """public-API composition: O(log n) locate + O(n) C-level tuple splice, then from_cuts"""
    s, e = kernel.piece(lo, hi, lo_closed, hi_closed)
    if s >= e:
        return mi
    c = mi.cuts
    starts, ends = c[0::2], c[1::2]
    i = bisect.bisect_left(ends, s)     # first pair that could merge on the left (end >= s)
    j = bisect.bisect_right(starts, e)  # pairs i..j-1 overlap or tile the new piece
    if i < j:
        s = min(s, starts[i]); e = max(e, ends[j - 1])
    return M.from_cuts(c[:2 * i] + (s, e) + c[2 * j:])

# ---- correctness: composition == mi | piece, and == v1 add, on a seeded sweep ----
rng = random.Random(9)
vals = [0, 1, 2, 3, 5, Fraction(7, 2), 2.5, 4, 6, float('inf'), float('-inf')]
fails = 0; n = 0; v1diff = 0
def members(m, pts):
    return [p in m for p in pts]
pts = sorted({Fraction(x) for x in range(-1, 8)} | {Fraction(x, 2) for x in range(-2, 16)} | {Fraction(x, 4) for x in range(-4, 30)})
for trial in range(600):
    mi = M.from_pieces([])
    v1mi = v1.MultiInterval()
    for _ in range(rng.randint(1, 6)):
        a, b = sorted(rng.sample([x for x in vals if abs(x) != float('inf')], 2))
        lc, hc = rng.random() < .5, rng.random() < .5
        if a == b: lc = hc = True
        ref = mi | M(a, b, start_closed=lc, end_closed=hc)
        got = bisect_insert(mi, a, b, lc, hc)
        n += 1
        if got != ref:
            fails += 1
            if fails < 5: print('MISMATCH', mi, a, b, lc, hc, got, ref)
        mi = got
        # v1 add: closed pieces only (v1 had openness via epsilons; use closed to compare)
    # v1 closed-only comparison on a separate run
for trial in range(300):
    mi = M.from_pieces([]); v1mi = v1.MultiInterval()
    for _ in range(rng.randint(1, 6)):
        a, b = sorted(rng.sample(range(0, 12), 2))
        mi = bisect_insert(mi, a, b)
        v1mi.add(v1.MultiInterval(a, b))
    fl = [float(p) for p in pts] + [x / 4 for x in range(0, 50)]
    if [x in mi for x in fl] != [x in v1mi for x in fl]:
        v1diff += 1
print(f'composition vs mi|piece: {n} inserts, {fails} mismatches; vs v1 add (closed, membership): {v1diff} diffs / 300')
# deliberate wrong expectation must be caught
bad = bisect_insert(M(0, 1), 2, 3)
assert bad != (M(0, 1) | M(2, 4)), 'probe cannot fail'
print('sabotage check: wrong expectation caught:', bad != (M(0, 1) | M(2, 4)))
# edge: tiling [0,1) + [1,2] merges; (0,1) + (1,2) stays two; point fills gap
print(bisect_insert(M(0, 1, end_closed=False), 1, 2), bisect_insert(M(0, 1, start_closed=False, end_closed=False), 1, 2, False, False),
      bisect_insert(M(0, 1, start_closed=False, end_closed=False) | M(1, 2, start_closed=False, end_closed=False), 1, 1))

# ---- cost at 50000 pieces, 10 inserts each followed by a query ----
N, K = 50000, 10
base = [(2 * i, 2 * i + 1) for i in range(N)]
news = [(rng.randrange(0, 2 * N) + 0.25, None) for _ in range(K)]
news = [(x, x + 0.5) for x, _ in news]
q = 777.5
def timed(label, fn):
    t = time.perf_counter(); r = fn(); dt = time.perf_counter() - t
    print(f'{label:55s} {dt:8.4f}s  ({dt / K * 1000:.2f} ms/insert)'); return r

v1base = v1.MultiInterval(); v1base.endpoints = [e for a, b in base for e in ((a, 0), (b, 0))]
def v1_add():
    m = v1.MultiInterval(); m.endpoints = list(v1base.endpoints)
    for a, b in news:
        m.add(v1.MultiInterval(a, b)); q in m
    return m
r1 = timed('v1 library MultiInterval.add (its real incremental path)', v1_add)
tup = [((a, 0), (b, 0)) for a, b in base]
timed('v1 compare.run_incremental_bisect (unmerged list, no set)', lambda: v1c.run_incremental_bisect(tup, [((a, 0), (b, 0)) for a, b in news]))
mbase = M.from_pieces(base)
def v2_or():
    m = mbase
    for a, b in news:
        m = m | M(a, b); q in m
    return m
r2 = timed('v2 mi | piece', v2_or)
def v2_builder():
    bld = v2.Builder()
    for a, b in base: bld.add_piece(a, b)
    t = time.perf_counter()
    for a, b in news:
        bld.add_piece(a, b); m = M.from_cuts(bld.build()); q in m
    return m
r3 = timed('v2 Builder.add_piece + build each (incl. initial fill)', v2_builder)
def v2_bis():
    m = mbase
    for a, b in news:
        m = bisect_insert(m, a, b); q in m
    return m
r4 = timed('v2 composition bisect_insert (cuts + from_cuts)', v2_bis)
print('final sets equal:', r2 == r3 == r4, ' v1 vs v2 membership on 2000 pts:',
      all((x in r1) == (x in r4) for x in [i * 0.25 for i in range(0, 2 * N * 4, 1000)]))
def bisect_insert_wrap(mi, lo, hi):
    s, e = kernel.piece(lo, hi)
    c = mi.cuts; starts, ends = c[0::2], c[1::2]
    i = bisect.bisect_left(ends, s); j = bisect.bisect_right(starts, e)
    if i < j: s = min(s, starts[i]); e = max(e, ends[j - 1])
    return M._wrap(c[:2 * i] + (s, e) + c[2 * j:])
def v2_bis_wrap():
    m = mbase
    for a, b in news:
        m = bisect_insert_wrap(m, a, b); q in m
    return m
r5 = timed('v2 splice via private _wrap (no O(n) is_valid; __debug__=' + str(__debug__) + ')', v2_bis_wrap)
print('r5 == r4:', r5 == r4)
