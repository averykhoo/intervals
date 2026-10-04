"""is_empty is_contiguous is_degenerate is_finite is_integral is_positive/negative/non_negative/non_positive"""
import sys; sys.path.insert(0, 'C:/Users/user/PycharmProjects/intervals/.scratch/v1-parity/accessors')
from common import *

PREDS = ['is_empty', 'is_contiguous', 'is_degenerate', 'is_finite', 'is_integral',
         'is_positive', 'is_negative', 'is_non_negative', 'is_non_positive']


def oracle(pieces_norm, name):
    """exact oracle from the normalized v2 pieces (lo, lo_closed, hi, hi_closed)"""
    P = pieces_norm
    if name == 'is_empty': return not P
    if name == 'is_contiguous': return len(P) == 1
    if name == 'is_degenerate': return bool(P) and all(lo == hi for lo, _, hi, _ in P)
    if name == 'is_finite': return all(abs(lo) != INF and abs(hi) != INF for lo, _, hi, _ in P)
    if name == 'is_integral': return bool(P) and all(lo == hi and abs(lo) != INF and Fraction(lo).denominator == 1 for lo, _, hi, _ in P)
    if name == 'is_positive': return bool(P) and (P[0][0] > 0 or (P[0][0] == 0 and not P[0][1]))
    if name == 'is_negative': return bool(P) and (P[-1][2] < 0 or (P[-1][2] == 0 and not P[-1][3]))
    if name == 'is_non_negative': return not P or P[0][0] >= 0
    if name == 'is_non_positive': return not P or P[-1][2] <= 0


def pieces_of(b):
    out = []
    for p in b.pieces:
        out.append((p.inf, p.inf_closed, p.sup, p.sup_closed))
    return out


hand = [
    [], [(0, 0, True, True)], [(0, 1, False, True)], [(0, 1, True, True)], [(-1, 0, True, False)], [(-1, 0, True, True)],
    [(-INF, 0, False, False)], [(0, INF, False, False)], [(-INF, INF, False, False)], [(1, 1, True, True), (2, 2, True, True)],
    [(2.0, 2.0, True, True)], [(Fraction(4, 2), Fraction(4, 2), True, True)], [(0.5, 0.5, True, True)],
    [(1, 1, True, True), (2, 3, True, True)], [(-2, -1, True, True), (1, 2, True, True)], [(0, 0, True, True), (1, 2, False, False)],
    [(Fraction(-1, 3), 0, False, False)], [(-0.0, 1, True, True)], [(10**400, 10**400, True, True)],
    [(-10**400, 10**400, True, True)], [(Fraction(10**400, 3), 10**401, True, False)], [(-5, -5, True, True), (5, 5, True, True)],
]
rng = random.Random(1788)
cases = hand + [rand_pieces(rng) for _ in range(600)]
n = 0
errs = []
for pcs in cases:
    try:
        a, b = build(pcs)
    except Exception as e:
        errs.append(('build', pcs, repr(e)));
        continue
    # builder sanity: same set
    for x in probe_points(b):
        check('builder', v1_contains(a, x) == (x in b), (pcs, x))
    P = pieces_of(b)
    for name in PREDS:
        n += 1
        try:
            r1 = getattr(a, name)
        except Exception as e:
            r1 = f'RAISES {type(e).__name__}: {e}'
        r2 = getattr(b, name)
        o = oracle(P, name)
        if not (r1 == r2 == o):
            FAILS.append((name, str(b), 'v1', r1, 'v2', r2, 'oracle', o))

# sabotage: a deliberately wrong expectation must be caught
before = len(FAILS)
a, b = build([(0, 1, True, True)])
check('sabotage', b.is_positive == oracle(pieces_of(b), 'is_non_negative') and False)
assert len(FAILS) == before + 1, 'sabotage not caught'
FAILS.pop()
print('build errors', errs[:5])
report('predicates', n)
