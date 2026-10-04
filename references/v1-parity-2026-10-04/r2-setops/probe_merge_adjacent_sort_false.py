"""v1 merge_adjacent(sort=False) on hand-built raw endpoint lists, vs v2 MultiInterval.from_pieces and brute force.

parts:
  A. precondition HOLDS: pieces sorted by (start, end) pair but unmerged (overlapping, nested, touching, points,
     duplicates). v1 sort=False must equal v2 from_pieces and brute force.
  B. precondition VIOLATED: pieces shuffled until NOT sorted. classify each v1 outcome:
     correct / silently wrong but consistent / raises (in merge or in a later membership test).
  C. is_contiguous on raw lists (v1 calls merge_adjacent(sort=False) internally) vs v2 from_pieces(...).is_contiguous.
  D. distance > 0 with sort=False on sorted raw lists vs sort=True (same v1 result expected).
  E. hand-picked cases, printed.
run: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe .scratch/v1-parity/r2-setops/probe_merge_adjacent_sort_false.py [sab]
"""
import sys, random, warnings, math
from fractions import Fraction as F
sys.path[:0] = ['.', 'archive/v1']
warnings.simplefilter('ignore')
import multi_interval as v1m
import intervals as v2

SAB = len(sys.argv) > 1
GRID = [F(-2), F(-1), F(-1, 2), F(0), F(1, 3), F(1), F(3, 2), F(2), F(3)]
PTS = sorted({g + d for g in GRID for d in (F(-1, 100), 0, F(1, 100))} | {F(5)}
             | {(a + b) / 2 for a, b in zip(GRID, GRID[1:])})


def eps(p):
    lo, hi, lc, hc = p
    return ((lo, 0 if lc else 1), (hi, 0 if hc else -1))


def raw_v1(pieces):
    a = v1m.MultiInterval()
    for p in pieces:
        a.endpoints += list(eps(p))
    return a


def member(pieces, x):
    return any((lo < x or (lo == x and lc)) and (x < hi or (x == hi and hc)) for lo, hi, lc, hc in pieces)


def v1_members(a):
    """membership via v1's own __contains__ (runs _consistency_check); None if it raises"""
    try:
        return [x in a for x in PTS]
    except AssertionError:
        return None


def structure(a):
    e = a.endpoints
    return [(e[i][0], e[i][1] == 0, e[i + 1][0], e[i + 1][1] == 0) for i in range(0, len(e), 2)]


def v2_struct(m):
    return [(p.inf, p.inf_closed, p.sup, p.sup_closed) for p in m]


def rand_pieces(rng):
    ps = []
    for _ in range(rng.randint(2, 6)):
        lo, hi = sorted(rng.sample(GRID, 2)) if rng.random() < .85 else (rng.choice(GRID),) * 2
        lc, hc = (True, True) if lo == hi else (rng.random() < .5, rng.random() < .5)
        ps.append((lo, hi, lc, hc))
    return ps


def is_sorted_pairs(ps):
    keys = [eps(p) for p in ps]
    return keys == sorted(keys)


caught = 0
rng = random.Random(2026)

# ---- A. sorted (by v1's own pair key) but unmerged
nA = badA = bruteA = 0
for _ in range(600):
    ps = sorted(rand_pieces(rng), key=eps)
    a1 = raw_v1(ps).merge_adjacent(sort=False)
    a1t = raw_v1(ps).merge_adjacent()
    a2 = v2.MultiInterval.from_pieces(ps)
    if SAB and nA == 5:
        a2 = a2 | v2.MultiInterval(F(5))
    m1, m2, mb = v1_members(a1), [x in a2 for x in PTS], [member(ps, x) for x in PTS]
    nA += 1
    bad = m1 != m2 or structure(a1) != v2_struct(a2) or structure(a1) != structure(a1t)
    badA += bad
    bruteA += m2 != mb
    if bad and SAB and nA == 6:
        caught += 1
print(f'[A sorted-unmerged, sort=False] cases={nA} v1-vs-v2 mismatches={badA} v2-vs-brute={bruteA}')

# ---- B. unsorted (precondition violated)
nB = correct = wrong_consistent = raises_merge = raises_later = wrong_inconsistent_nocheck = bruteB = 0
examples = {}
while nB < 600:
    ps = rand_pieces(rng)
    rng.shuffle(ps)
    if is_sorted_pairs(ps):
        continue
    nB += 1
    a2 = v2.MultiInterval.from_pieces(ps)
    mb = [member(ps, x) for x in PTS]
    m2 = [x in a2 for x in PTS]
    bruteB += m2 != mb
    try:
        a1 = raw_v1(ps).merge_adjacent(sort=False)
    except Exception as exc:  # noqa
        raises_merge += 1
        examples.setdefault('raises_merge', (ps, repr(exc)))
        continue
    m1 = v1_members(a1)
    if m1 is None:
        raises_later += 1
        # what would membership say with the check off? use a brute evaluation of v1's raw output pieces
        examples.setdefault('raises_later', (ps, structure(a1)))
        continue
    if m1 == mb and structure(a1) == v2_struct(a2):
        correct += 1
    else:
        wrong_consistent += 1
        examples.setdefault('wrong_consistent', (ps, structure(a1), v2_struct(a2),
                                                 [x for x, u, v in zip(PTS, m1, mb) if u != v][:4]))
print(f'[B unsorted, sort=False] cases={nB} v1 correct={correct} v1 silently-wrong (passes its own '
      f'consistency check)={wrong_consistent} v1 output fails consistency check on next use={raises_later} '
      f'v1 raises in merge={raises_merge}; v2 from_pieces vs brute mismatches={bruteB}')
for k, v in examples.items():
    print('   example', k, v)

# ---- C. is_contiguous on raw lists. v1's copy() runs _consistency_check, which asserts the FLAT endpoint list is
# sorted, so with the default CONSISTENCY_CHECK=True only flat-sorted raw lists (touching, not overlapping: the
# docstring's "{ [1] , (1, 2] }") get through. C1: those. C2: any raw list, sorted by pair or not, with the check on
# and off (v1: "may be set to True for production use", i.e. turned off).
def flat_sorted(ps):
    e = [x for p in ps for x in eps(p)]
    return e == sorted(e)

def touching_pieces(rng):
    # consecutive pieces sharing an end value, random closedness, sometimes a gap; flat-sorted by construction
    # unless an open end meets an open start (then the pair is still flat-sorted: (v,-1) <= (v,1))
    vals = sorted(rng.sample(GRID, rng.randint(3, 6)))
    ps = []
    for lo, hi in zip(vals, vals[1:]):
        if rng.random() < .2:
            continue
        ps.append((lo, hi, rng.random() < .5, rng.random() < .5))
    if rng.random() < .3:
        v = rng.choice(vals)
        ps.append((v, v, True, True))
        ps.sort(key=eps)
    return ps

nC1 = badC1 = skippedC1 = 0
for i in range(600):
    ps = touching_pieces(rng)
    if not ps or not flat_sorted(ps):
        skippedC1 += 1
        continue
    truth = v2.MultiInterval.from_pieces(ps).is_contiguous
    ms = [member(ps, x) for x in PTS]
    idx = [j for j, m in enumerate(ms) if m]
    assert (bool(idx) and all(ms[idx[0]:idx[-1] + 1])) == truth, ps
    if SAB and nC1 == 7:
        truth = not truth
    c1 = raw_v1(ps).is_contiguous
    nC1 += 1
    if c1 != truth:
        badC1 += 1
        if SAB and nC1 == 8:
            caught += 1
print(f'[C1 is_contiguous, flat-sorted raw (touching) lists, check on] cases={nC1} (skipped {skippedC1}) '
      f'v1-vs-v2 mismatches={badC1}; v2 vs brute 0 (asserted)')

for check in (True, False):
    v1m.CONSISTENCY_CHECK = check
    cnt = {}
    for i in range(600):
        ps = rand_pieces(rng)
        if i % 2:
            ps.sort(key=eps)
        else:
            rng.shuffle(ps)
        truth = v2.MultiInterval.from_pieces(ps).is_contiguous
        try:
            c1 = raw_v1(ps).is_contiguous
            r = 'agree' if c1 == truth else 'WRONG'
        except AssertionError:
            r = 'AssertionError'
        key = ('pair-sorted' if is_sorted_pairs(ps) else 'unsorted', 'flat-sorted' if flat_sorted(ps) else 'overlapping', r)
        cnt[key] = cnt.get(key, 0) + 1
        if r == 'WRONG':
            examples.setdefault(f'is_contiguous_wrong_check{check}', (ps, c1, truth))
    print(f'[C2 is_contiguous, random raw lists, CONSISTENCY_CHECK={check}]', dict(sorted(cnt.items())))
v1m.CONSISTENCY_CHECK = True
for k in [k for k in examples if k.startswith('is_contiguous_wrong')]:
    print('   example', k, examples[k])

# ---- D. distance > 0 with sort=False on sorted raw lists (must equal sort=True)
nD = badD = 0
for d in (F(1, 4), F(1, 2), 1, 2, math.inf):
    for _ in range(100):
        ps = sorted(rand_pieces(rng), key=eps)
        a = structure(raw_v1(ps).merge_adjacent(d, sort=False))
        b = structure(raw_v1(ps).merge_adjacent(d))
        nD += 1
        badD += a != b
print(f'[D distance>0, sort=False vs sort=True on sorted raw] cases={nD} mismatches={badD}')

# ---- E. hand-picked
def show(ps, **kw):
    try:
        r1 = structure(raw_v1(ps).merge_adjacent(sort=False, **kw))
    except Exception as exc:  # noqa
        r1 = repr(exc)
    r2 = v2_struct(v2.MultiInterval.from_pieces(ps))
    print('   ', ps, '\n      v1 sort=False ->', r1, '\n      v2 from_pieces ->', r2)

T, Fl = True, False
print('[E hand-picked]')
show([(5, 6, T, T), (0, 1, T, T)])                       # reversed disjoint
show([(0, 10, T, T), (1, 2, T, T), (3, 4, T, T)])        # nested, sorted
show([(3, 4, T, T), (1, 2, T, T), (0, 10, T, T)])        # nested, reversed
show([(1, 2, Fl, T), (1, 1, T, T)])                      # (1,2] before [1]: unsorted by pair key
show([(2, 3, T, T), (0, 1, T, Fl), (1, 2, T, Fl)])       # tiling, shuffled
show([(-math.inf, 0, Fl, Fl), (5, math.inf, T, Fl), (1, 2, T, T)])
show([(2, 2, T, T), (2, 2, T, T)])                       # duplicate point (sorted)
print('v2 Builder exported:', hasattr(v2, 'Builder'), ' kernel.Builder exists:',
      hasattr(__import__('intervals.kernel').kernel, 'Builder'))
try:
    v2.MultiInterval.from_cuts(v2.MultiInterval.from_pieces([(5, 6)]).cuts + v2.MultiInterval.from_pieces([(0, 1)]).cuts)
    print('from_cuts accepted unsorted cuts')
except ValueError as exc:
    print('from_cuts(unsorted) ->', 'ValueError:', str(exc)[:70])

if SAB:
    print('sabotage caught', caught, 'of 2')
    assert caught == 2
else:
    assert badA == 0 and bruteA == 0 and bruteB == 0 and badC1 == 0 and badD == 0
print('done')
