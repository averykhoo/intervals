"""MultipleInterval equality on equal sets built differently (split pieces, shuffled, overlapping duplicates) vs v2"""
import random
from collections import Counter
from common import *  # noqa
from probe_pickle_hash import MV1, rand_mv1, mv1_to_v2, pts, rng as _r  # reuses the pool

rng = random.Random(7)
st = Counter()
for trial in range(400):
    a = rand_mv1()
    pieces = []
    for iv in a.intervals:
        if not iv.is_degenerate and rng.random() < 0.7:
            lo = iv.start if not math.isinf(iv.start) else (iv.end - 5 if not math.isinf(iv.end) else 0)
            hi = iv.end if not math.isinf(iv.end) else lo + 5
            mid = Fraction(lo) + (Fraction(hi) - Fraction(lo)) / 3
            if rng.random() < 0.5:   # split, closed-left half-open
                pieces += [V1(iv.start, iv.start_open, mid, False), V1(mid, False, iv.end, iv.end_closed)]
            else:                    # overlapping halves plus a duplicate point
                pieces += [V1(iv.start, iv.start_open, mid, True), V1(mid, False, iv.end, iv.end_closed),
                           V1(mid, False, mid, True)]
        else:
            pieces.append(iv)
    rng.shuffle(pieces)
    try:
        b = MV1(*pieces)
    except Exception as ex:  # noqa
        st['v1 build raised ' + type(ex).__name__] += 1
        print('v1 raised', type(ex).__name__, ex, pieces)
        continue
    A, B = mv1_to_v2(a), mv1_to_v2(b)
    same_set = v1_members(a, pts + [Fraction(p) / 7 for p in range(-30, 30)]) == v1_members(b, pts + [Fraction(p) / 7 for p in range(-30, 30)])
    st[f'v1 eq={a == b} v2 eq={A == B} sameset={same_set}'] += 1
    check('eq-v1', (a == b) == same_set, (a, b))
    check('eq-v2', (A == B) == same_set, (A, B))
    check('hash-v2', A != B or hash(A) == hash(B))
print(dict(st))
before = len(MISMATCHES)
check('SABOTAGE', (MV1(V1(0, False, 1, True)) == MV1(V1(0, False, Fraction(1, 2), False), V1(Fraction(1, 2), False, 1, True))) is False)
print('sabotage caught', len(MISMATCHES) - before, 'of 1'); del MISMATCHES[before:]
report('probe_mv_equal_sets')
