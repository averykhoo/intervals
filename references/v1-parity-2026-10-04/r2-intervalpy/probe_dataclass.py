"""v1 Interval's dataclass surface vs v2: keyword construction, replace, asdict/astuple/fields, frozen, order, hash/eq"""
import copy
import dataclasses
import itertools
import random
from collections import Counter

from common import *  # noqa

rng = random.Random(20261004)
POOL = [-math.inf, -2, -1, 0, Fraction(1, 3), 0.5, Fraction(1, 2), 1, 1.0, 2, math.inf]
FLAGS = [False, True]


def outcome_v1(**kw):
    try:
        return 'ok', V1(**kw)
    except Exception as e:  # noqa
        return type(e).__name__, str(e)


def outcome_v2(start, start_open, end, end_closed):
    try:
        return 'ok', MI(start=start, end=end, start_closed=not start_open, end_closed=end_closed)
    except Exception as e:  # noqa
        return type(e).__name__, str(e)


def classify_v1_refusal(s, so, e, ec, msg):
    if (math.isinf(s) and not so) or (math.isinf(e) and ec):
        return 'v1 refuses a closed infinite end (v2: affine extended reals)'
    if s == math.inf or e == -math.inf:
        return 'v1 refuses start=+inf / end=-inf'
    if s == e and (so or not ec):
        return 'v1 refuses a half-open/open degenerate (empty) interval'
    return 'OTHER: ' + msg


# ---- A. keyword construction, every combination of the pool ----
stats = Counter()
v1_refusals = Counter()
pts_all = test_points([v for v in POOL])
for s, so, e, ec in itertools.product(POOL, FLAGS, POOL, FLAGS):
    r1 = outcome_v1(start=s, start_open=so, end=e, end_closed=ec)
    r2 = outcome_v2(s, so, e, ec)
    if r1[0] == 'ok' and r2[0] == 'ok':
        stats['both ok'] += 1
        check('kw-membership', v1_members(r1[1], pts_all) == v2_members(r2[1], pts_all), (s, so, e, ec))
        # a v1 Interval never contains +-inf; v2 with open infinite ends neither
        check('kw-inf', (math.inf in r2[1]) == False and (-math.inf in r2[1]) == False, (s, so, e, ec))
    elif r1[0] != 'ok' and r2[0] == 'ok':
        why = classify_v1_refusal(s, so, e, ec, r1[1])
        v1_refusals[why] += 1
        stats['v1 raises, v2 ok'] += 1
        check('kw-v1-refusal-classified', not why.startswith('OTHER'), (s, so, e, ec, r1))
        if why.startswith('v1 refuses a half-open'):
            check('kw-empty', r2[1] == v2.EMPTY, (s, so, e, ec, r2))
    elif r1[0] == 'ok' and r2[0] != 'ok':
        stats['v1 ok, v2 raises'] += 1
        check('kw-v2-refuses', False, (s, so, e, ec, r2))
    else:
        stats['both raise'] += 1
        check('kw-both-raise-same-type', r1[0] == r2[0], (s, so, e, ec, r1, r2))
print('A keyword construction:', dict(stats))
print('  v1 refusals where v2 builds a set:', dict(v1_refusals))

# positional v1 construction == keyword
iv = V1(1, True, 2, False)
check('kw==pos', V1(start=1, start_open=True, end=2, end_closed=False) == iv)
# v2 keyword names: start/end/start_closed/end_closed
check('v2-kw', MI(start=1, end=2, start_closed=False, end_closed=False) == MI.parse('(1, 2)'))


# ---- B. dataclasses.replace vs v2 composition ----
def v2_replace(A, start=None, start_open=None, end=None, end_closed=None):
    """v2 spelling of dataclasses.replace on a one-piece set"""
    s = A.inf if start is None else start
    sc = A.inf_closed if start_open is None else not start_open
    e = A.sup if end is None else end
    ec = A.sup_closed if end_closed is None else end_closed
    return MI(s, e, start_closed=sc, end_closed=ec)


stats = Counter()
valid = []
for s, so, e, ec in itertools.product(POOL, FLAGS, POOL, FLAGS):
    r = outcome_v1(start=s, start_open=so, end=e, end_closed=ec)
    if r[0] == 'ok':
        valid.append(r[1])
for _ in range(600):
    iv = rng.choice(valid)
    field = rng.choice(['start', 'start_open', 'end', 'end_closed'])
    val = rng.choice(FLAGS) if field.endswith(('open', 'closed')) else rng.choice(POOL)
    A = v1_to_v2(iv)
    try:
        r1 = ('ok', dataclasses.replace(iv, **{field: val}))
    except Exception as ex:  # noqa
        r1 = (type(ex).__name__, str(ex))
    try:
        r2 = ('ok', v2_replace(A, **{field: val}))
    except Exception as ex:  # noqa
        r2 = (type(ex).__name__, str(ex))
    if r1[0] == 'ok' and r2[0] == 'ok':
        stats['both ok'] += 1
        check('replace-membership', v1_members(r1[1], pts_all) == v2_members(r2[1], pts_all), (iv, field, val))
    elif r1[0] != 'ok' and r2[0] == 'ok':
        stats['v1 raises (post_init re-validates), v2 builds'] += 1
        new = dataclasses.asdict(iv) | {field: val}
        check('replace-v1-refusal-classified',
              not classify_v1_refusal(new['start'], new['start_open'], new['end'], new['end_closed'], r1[1]).startswith('OTHER'),
              (iv, field, val, r1))
    elif r1[0] == 'ok':
        check('replace-v2-raises', False, (iv, field, val, r2))
    else:
        stats['both raise'] += 1
        check('replace-same-exc', r1[0] == r2[0], (iv, field, val, r1, r2))
print('B dataclasses.replace (600 random):', dict(stats))
# copy.replace (python 3.13) on each
print('  copy.replace(v1, end=5):', copy.replace(V1(1, False, 2, True), end=5))
try:
    copy.replace(MI(1, 2), end=5)
    print('  copy.replace(v2) works')
except TypeError as ex:
    print('  copy.replace(v2) TypeError:', ex)
try:
    dataclasses.replace(MI(1, 2), end=5)
except TypeError as ex:
    print('  dataclasses.replace(v2) TypeError:', ex)


# ---- C. asdict / astuple / fields vs v2 read-out ----
def v2_asdict(A):
    return {'start': A.inf, 'start_open': not A.inf_closed, 'end': A.sup, 'end_closed': A.sup_closed}


n = 0
for iv in valid:
    d1 = dataclasses.asdict(iv)
    d2 = v2_asdict(v1_to_v2(iv))
    check('asdict', d1 == d2, (iv, d1, d2))
    check('astuple', dataclasses.astuple(iv) == tuple(d2.values()), iv)
    # round trip from the dict
    check('from-dict', V1(**d1) == iv)
    check('v2-from-dict', MI(d2['start'], d2['end'], start_closed=not d2['start_open'], end_closed=d2['end_closed'])
          == v1_to_v2(iv))
    n += 1
print('C asdict/astuple on', n, 'valid intervals; v1 fields:', [f.name for f in dataclasses.fields(V1)])
print('  is_dataclass v1/v2:', dataclasses.is_dataclass(V1), dataclasses.is_dataclass(MI))
# value types: v1 keeps the given type, v2 normalizes (Fraction(2,1) -> 2, -0.0 -> 0.0)
print('  types: v1', dataclasses.asdict(V1(Fraction(2, 1), False, 3, True)), dataclasses.asdict(V1(-0.0, False, 3, True)),
      ' v2', v2_asdict(MI(Fraction(2, 1), 3)), v2_asdict(MI(-0.0, 3)))
print('  empty: v2 inf of EMPTY ->', end=' ')
try:
    print(v2.EMPTY.inf)
except Exception as ex:  # noqa
    print(type(ex).__name__, ex)

# ---- D. frozen ----
iv, A = V1(1, False, 2, True), MI(1, 2)
for obj, attr in ((iv, 'start'), (iv, 'end_closed'), (iv, 'new_attr'), (A, '_cuts'), (A, 'new_attr')):
    try:
        setattr(obj, attr, 5)
        print('  setattr OK (not frozen)', type(obj).__name__, attr)
        check('frozen', False, (type(obj).__name__, attr))
    except AttributeError as ex:
        print(f'  setattr {type(obj).__name__}.{attr}: {type(ex).__name__} (AttributeError subclass) {ex}')
for obj, attr in ((iv, 'start'), (A, '_cuts')):
    try:
        delattr(obj, attr)
        check('frozen-del', False, attr)
    except AttributeError as ex:
        print(f'  delattr {type(obj).__name__}.{attr}: {type(ex).__name__}')
check('v1-frozen-exc', issubclass(dataclasses.FrozenInstanceError, AttributeError))
check('v2-no-dict', not hasattr(A, '__dict__'))
# v1 has a __dict__? (dataclass without slots)
print('  v1 has __dict__:', hasattr(iv, '__dict__'), ' v2 has __dict__:', hasattr(A, '__dict__'))

# ---- E. order=True (lexicographic on fields) vs v2 sort_key ----
for trial in range(300):
    xs = [rng.choice(valid) for _ in range(rng.randint(2, 12))]
    s1 = sorted(xs)
    s2 = sorted((v1_to_v2(x) for x in xs), key=lambda a: a.sort_key)
    check('order-sorted', [v1_to_v2(x) for x in s1] == s2, xs)
    a, b = rng.choice(valid), rng.choice(valid)
    for op in ('__lt__', '__le__', '__gt__', '__ge__'):
        check('order-' + op, getattr(a, op)(b) == getattr(v1_to_v2(a).sort_key, op)(v1_to_v2(b).sort_key), (a, b, op))
print('E order: 300 random lists and pairs vs sort_key')
try:
    print('  v1 Interval < 3:', V1(1, False, 2, True) < 3)
except TypeError as ex:
    print('  v1 Interval < 3: TypeError', ex)
print('  v2 MI(1,2) < MI(3):', MI(1, 2) < MI(3), ' (pointwise TruthSet, not structural)')

# ---- F. hash / eq across numeric types ----
for trial in range(300):
    xs = [rng.choice(valid) for _ in range(rng.randint(1, 15))]
    check('hash-dedup', len(set(xs)) == len({v1_to_v2(x) for x in xs}), xs)
    a, b = rng.choice(valid), rng.choice(valid)
    check('eq', (a == b) == (v1_to_v2(a) == v1_to_v2(b)), (a, b))
    if a == b:
        check('hash-v1', hash(a) == hash(b))
        check('hash-v2', hash(v1_to_v2(a)) == hash(v1_to_v2(b)))
print('F hash/eq: 300 random lists and pairs;',
      'v1 [1/2,1] == [0.5,1.0]:', V1(Fraction(1, 2), False, 1, True) == V1(0.5, False, 1.0, True),
      ' v2:', MI(Fraction(1, 2), 1) == MI(0.5, 1.0))
print('  v1 Interval(1,F,1,T) == 1:', V1(1, False, 1, True) == 1, ' v2 MI(1) == 1:', MI(1) == 1)

# ---- sabotage: a deliberately wrong expectation must be caught ----
before = len(MISMATCHES)
bad = V1(0, False, 1, True)
check('SABOTAGE', v1_members(bad, pts_all) == v2_members(MI(0, 1, end_closed=False), pts_all))
check('SABOTAGE-order', sorted([V1(1, True, 2, True), V1(1, False, 2, True)])[0].start_open is True)
caught = len(MISMATCHES) - before
print('sabotage caught:', caught, 'of 2')
del MISMATCHES[before:]
report('probe_dataclass')
