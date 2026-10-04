"""pickling/copy of v1 Interval and MultipleInterval vs v2 MultiInterval; MultipleInterval hashability and equality"""
import copy
import itertools
import pickle
import random
from collections import Counter

from common import *  # noqa

MV1 = v1i.MultipleInterval
rng = random.Random(41)
POOL = [-math.inf, -3, -2, -1, 0, Fraction(1, 3), 0.5, 1, 1.0, 2, 3, math.inf]
valid = []
for s, so, e, ec in itertools.product(POOL, (False, True), POOL, (False, True)):
    try:
        valid.append(V1(s, so, e, ec))
    except (ValueError, TypeError):
        pass
pts = test_points(POOL)


def mv1_to_v2(m):
    return MI.from_pieces([(iv.start, iv.end, not iv.start_open, iv.end_closed) for iv in m.intervals])


def rand_mv1():
    return MV1(*[rng.choice(valid) for _ in range(rng.randint(0, 5))])


# ---- pickling / copy / deepcopy / repr of v1 Interval vs v2 ----
st = Counter()
for iv in valid:
    A = v1_to_v2(iv)
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        b1 = pickle.loads(pickle.dumps(iv, proto))
        b2 = pickle.loads(pickle.dumps(A, proto))
        check('pickle-v1', b1 == iv and type(b1) is V1 and hash(b1) == hash(iv), (iv, proto))
        check('pickle-v2', b2 == A and type(b2) is MI and hash(b2) == hash(A), (A, proto))
        # endpoint types survive (Fraction stays Fraction, float stays float)
        check('pickle-v2-types', [type(c.value) for c in b2.cuts] == [type(c.value) for c in A.cuts], A)
        st['pickled'] += 1
    for f in (copy.copy, copy.deepcopy):
        check('copy-v1', f(iv) == iv)
        check('copy-v2', f(A) == A)
    check('repr-eval-v1', eval(repr(iv), {'Interval': V1, 'inf': math.inf, 'Fraction': Fraction}) == iv, repr(iv))
    check('repr-eval-v2', eval(repr(A), {'MultiInterval': MI}) == A, repr(A))
print('Interval pickle/copy/deepcopy/repr-eval:', dict(st))
print('  v1 repr', repr(V1(-math.inf, True, Fraction(1, 3), True)), '| v2 repr', repr(MI(-math.inf, Fraction(1, 3), start_closed=False)))
print('  v2 copy.copy is identity:', copy.copy(MI(1, 2)) is MI(1, 2), '(fresh objects)')

# ---- MultipleInterval: hashability ----
m = MV1(V1(0, False, 1, True), V1(2, False, 3, True))
try:
    hash(m)
    print('v1 MultipleInterval hash OK')
    check('mv1-unhashable', False)
except TypeError as ex:
    print('v1 MultipleInterval hash: TypeError', ex)
print('  v1 MultipleInterval.__hash__ is', MV1.__hash__)
try:
    {m}
except TypeError as ex:
    print('  {MultipleInterval}: TypeError', ex)
A = mv1_to_v2(m)
print('v2 hash', hash(A), ' in a set:', {A, mv1_to_v2(MV1(V1(2, False, 3, True), V1(0, False, 1, True)))})
print('v1 MultiInterval (the real class) __hash__:', v1i.MultiInterval.__hash__)

# ---- MultipleInterval: equality, membership, pickling, copy vs v2 ----
st = Counter()
for trial in range(400):
    a, b = rand_mv1(), rand_mv1()
    A, B = mv1_to_v2(a), mv1_to_v2(b)
    check('mv-members', v1_members(a, pts) == v2_members(A, pts), (a, A))
    eq1, eq2 = a == b, A == B
    st[f'eq v1={eq1} v2={eq2}'] += 1
    check('mv-eq', eq1 == eq2, (a, b))
    # hash consistency on v2 for equal sets
    if eq2:
        check('mv-hash', hash(A) == hash(B))
    # v1 equality really is set equality? brute force
    check('mv-eq-is-set-eq', eq1 == (v1_members(a, pts) == v1_members(b, pts)), (a, b))
    # pickling / copy
    for proto in (0, 2, pickle.HIGHEST_PROTOCOL):
        pa = pickle.loads(pickle.dumps(a, proto))
        check('mv-pickle', pa == a and type(pa) is MV1 and pa.intervals == a.intervals, (a, proto))
        check('v2-pickle', pickle.loads(pickle.dumps(A, proto)) == A)
    ca = a.copy()
    check('mv-copy', ca == a and ca is not a and ca.intervals is not a.intervals)
    check('mv-deepcopy', copy.deepcopy(a) == a and copy.deepcopy(A) == A)
    # dedup via the v2 hash vs v1 pairwise equality (the v1 way, O(n^2))
    xs = [rand_mv1() for _ in range(rng.randint(1, 6))] + [a, a.copy()]
    uniq1 = []
    for x in xs:
        if not any(x == y for y in uniq1):
            uniq1.append(x)
    check('dedup', len(uniq1) == len({mv1_to_v2(x) for x in xs}), xs)
print('MultipleInterval 400 random pairs:', dict(st))
# v1 MultipleInterval vs a bare number and vs v1 MultiInterval
print('  v1 MV(point 1) == 1:', MV1(V1(1, False, 1, True)) == 1, ' v2 MI(1) == 1:', MI(1) == 1)
print('  v1 MV() == MV():', MV1() == MV1(), ' v2 EMPTY == MI():', v2.EMPTY == MI())

# sabotage
before = len(MISMATCHES)
a = MV1(V1(0, False, 1, True))
check('SABOTAGE', v1_members(a, pts) == v2_members(MI(0, 1, start_closed=False), pts))
check('SABOTAGE2', (MV1(V1(0, False, 1, True)) == MV1(V1(0, False, 2, True))) == (MI(0, 1) == MI(0, 1)))
print('sabotage caught:', len(MISMATCHES) - before, 'of 2')
del MISMATCHES[before:]
report('probe_pickle_hash')
