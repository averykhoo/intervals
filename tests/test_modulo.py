"""
modulo, floor, floordiv and divmod (multiinterval.modulo) against the brute-force oracle in tests.oracles

the oracle enumerates the quotient k = floor(x / y) instead of solving for it, so it checks the
far-edge shapes and the O(1) attainment test rather than restating them. soundness, endpoint
attainment and interior sharpness together pin a result to the attained set exactly; the tables pin
the prototype's suites (references/modulo-derivations/claude-fable/modulo_v3_prototype.py), the design
notes' degenerate cases and python's scalar `%` (v1's `A % scalar` too, until v1 was deleted on 2026-10-04).

`tests/exhaustive_modulo.py` is the slow exhaustive differential over an exact grid; it is not part of
the gate.
"""
import math
import random
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import MultiInterval
from multiinterval import modulo
from multiinterval import ops
from multiinterval.errors import DomainClippedWarning
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import HullWarning
from multiinterval.errors import IndeterminateResultWarning
from multiinterval.fmt import format_cuts
from multiinterval.fmt import parse
from multiinterval.kernel import EMPTY
from multiinterval.kernel import contains_point
from multiinterval.kernel import intersection
from multiinterval.kernel import is_subset
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import union
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import probe_points
from tests.test_ops_properties import exact_sets
from tests.test_ops_properties import special_point_sets

INF = math.inf
P = MultiInterval.parse

# the operand strategies clip +-inf dividends and 0 divisors on purpose; each warning is pinned by its
# own pytest.warns test below
pytestmark = [
    pytest.mark.filterwarnings('ignore::multiinterval.errors.DomainClippedWarning'),
    pytest.mark.filterwarnings('ignore::multiinterval.errors.EmptySetPropagationWarning'),
]


def one(lo, lo_closed, hi, hi_closed):
    return normalize([piece(lo, hi, lo_closed, hi_closed)])


def show(cuts):
    return list(pieces(cuts))


# STRATEGIES: every sign, zero crossings, +-inf, and the values where quotients change

mod_values = st.one_of(
    st.sampled_from([-INF, -3, -2, Fraction(-3, 2), -1, Fraction(-1, 2), 0, Fraction(1, 2), 1,
                     Fraction(3, 2), 2, 3, INF]),
    st.integers(-12, 12),
    st.fractions(min_value=-12, max_value=12, max_denominator=4),
)


@st.composite
def mod_pieces(draw):
    lo, hi = sorted((draw(mod_values), draw(mod_values)))
    if lo == hi:
        return piece(lo, hi)
    return piece(lo, hi, draw(st.booleans()), draw(st.booleans()))


mod_operands = st.lists(mod_pieces(), min_size=1, max_size=3).map(normalize)


def mod_probes(a, b, result):
    """result and operand ends, their small quotients (the z-points x/k) and residues, and gaps"""
    ends = {v for c in (a, b) for lo, _, hi, _ in pieces(c) for v in (lo, hi)}
    extra = set()
    for x in ends:
        if math.isfinite(x):
            extra.update(Fraction(x) / k for k in range(1, 9))
            extra.update(-Fraction(x) / k for k in range(1, 9))
        for y in ends:
            extra.update(pointwise('mod', x, y))
    extra_cuts = normalize(piece(v, v) for v in extra)
    return set(probe_points(result, extra_cuts)) | {0}


# THE PROTOTYPE'S SUITES (M7a)

# modulo_v3_prototype.py's corner / zero-touch geometries, verbatim: f = (x0, x1, y0, y1 closed)
GEOMS = [
    ('G1', (3, 4.5, 2.75, 3), lambda f: [(0, f[0] and f[3], 1.75, f[1] and f[2])]),
    ('G2', (3, 4, 2, 3), lambda f: [(0, (f[0] and f[3]) or (f[1] and f[2]), 2, False)]),
    ('G3', (3, 6, 2, 3), lambda f: [(0, True, 3, False)]),
    ('G4', (1, 2, 3, 4), lambda f: [(1, f[0], 2, f[1])]),
    ('G5', (1, 3, 3, 4), lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(1, f[0], 3, f[1])]),
    ('G6', (3.5, 4, 2, 3),
     lambda f: ([(0, True, 0, True)] if (f[1] and f[2]) else []) + [(0.5, f[0] and f[3], 2, False)]),
    ('G7', (3, 4, 2.75, 4), lambda f: [(0, True, 1.25, f[1] and f[2]), (3, f[0], 4, False)]),
]


@pytest.mark.parametrize('name, box, expect', GEOMS, ids=[g[0] for g in GEOMS])
@pytest.mark.parametrize('flags', range(16))
def test_prototype_corner_suite(name, box, expect, flags):
    """the prototype's 112 hand-derived cases (its literals are exact binary floats)"""
    x0, x1, y0, y1 = box
    f = tuple(bool(flags & bit) for bit in (1, 2, 4, 8))
    got = modulo.mod(one(x0, f[0], x1, f[1]), one(y0, f[2], y1, f[3]))
    assert show(got) == expect(f)


def _antipodal(expected):
    """a result under (x, y) -> (-x, -y), the one exact sign identity: negated, flags on their ends"""
    return [(-hi, hc, -lo, lc) for lo, lc, hi, hc in reversed(expected)]


@pytest.mark.parametrize('name, box, expect', GEOMS, ids=[g[0] for g in GEOMS])
@pytest.mark.parametrize('flags', range(16))
def test_prototype_corner_suite_antipodal(name, box, expect, flags):
    """Q3: the same 112 cases through (-x) mod (-y) = -(x mod y), with the flags following their ends"""
    x0, x1, y0, y1 = box
    f = tuple(bool(flags & bit) for bit in (1, 2, 4, 8))
    got = modulo.mod(one(-x1, f[1], -x0, f[0]), one(-y1, f[3], -y0, f[2]))
    assert show(got) == _antipodal(expect(f))


def _prototype_cases():
    """the prototype's fuzz: same seed and ranges; its 2-decimal floats are read as exact decimals"""
    r = random.Random(5)
    for _ in range(4000):
        x0 = round(r.uniform(0, 20), 2)
        x1 = round(x0 + r.uniform(0, 20), 2)
        y0 = round(r.uniform(0.3, 9), 2)
        y1 = round(y0 + r.uniform(0, 8), 2)
        flags = [r.random() < .5 for _ in range(4)]
        a = (Fraction(str(x0)), flags[0], Fraction(str(x1)), flags[1])
        b = (Fraction(str(y0)), flags[2], Fraction(str(y1)), flags[3])
        if a[0] == a[2] and not (a[1] and a[3]) or b[0] == b[2] and not (b[1] and b[3]):
            continue
        for _ in range(160):  # the prototype's 80 sampled (x, y) pairs, so later cases match its own
            r.uniform(0, 1)
        yield a, b


def test_prototype_fuzz():
    """
    every endpoint closed iff the oracle attains it, and sampled pairs land inside. the prototype
    printed `3997 cases`; the count here is pinned so the generator cannot silently shrink
    """
    rng = random.Random(0)
    cases = 0
    for (x0, x0c, x1, x1c), (y0, y0c, y1, y1c) in _prototype_cases():
        cases += 1
        a, b = one(x0, x0c, x1, x1c), one(y0, y0c, y1, y1c)
        result = modulo.mod(a, b)
        for lo, lo_closed, hi, hi_closed in pieces(result):
            assert lo_closed == attained('mod', lo, a, b), (show(a), show(b), show(result), 'lo')
            assert hi_closed == attained('mod', hi, a, b), (show(a), show(b), show(result), 'hi')
        for x in sample(a, 4, rng):
            for y in sample(b, 4, rng):
                assert contains_point(result, pointwise('mod', x, y)[0]), (x, y, show(result))
    assert cases == 3997


# the design notes' degenerate-operand table (§3c): none of these needs special-casing
@pytest.mark.parametrize('a, b, expected', [
    ('[5]', '[2, 3]', '{ [0, 1] , [2, 5/2) }'),
    ('[3, 7]', '[5]', '{ [0, 2] , [3, 5) }'),
    ('[7]', '[3]', '[1]'),
    ('[6]', '[3]', '[0]'),
    ('[2]', '[5, 9]', '[2]'),
    ('[4]', '[4, 7]', '{ [0] , [4] }'),
])
def test_degenerate_operands(a, b, expected):
    assert format_cuts(modulo.mod(parse(a), parse(b))) == expected


# the design notes' pre-guard failures, kept there as the regression target for negative operands
@pytest.mark.parametrize('a, b, expected', [
    ('[-7, -3]', '[2, 5]', '[0, 5)'),  # was ZeroDivisionError
    ('[3, 7]', '[-5, -2]', '(-5, 0]'),  # was silently empty; -1.8 is 3.2 mod -5
    ('[-3, 7]', '[2, 5]', '[0, 5)'),
])
def test_design_notes_negative_regressions(a, b, expected):
    assert format_cuts(modulo.mod(parse(a), parse(b))) == expected


def test_design_notes_zero_crossing_divisor():
    # was (0, 5): the negative half of the divisor was dropped; -0.99 is 3.01 mod -2
    result = modulo.mod(parse('[3, 7]'), parse('[-2, 5]'))
    assert format_cuts(result) == '(-2, 5)'
    assert contains_point(result, Fraction(-99, 100))


# WORKED EXAMPLES, every quadrant

@pytest.mark.parametrize('a, b, expected', [
    # Q1
    ('[1, 2]', '[3, 4]', '[1, 2]'),
    ('[3, 6]', '[4, 5]', '{ [0, 2] , [3, 5) }'),  # the near edges would miss [4, 5)
    ('[3, 8]', '(8, 12]', '[3, 8]'),
    ('(0, 1)', '[1]', '(0, 1)'),
    ('[0, 10]', '[3]', '[0, 3)'),
    # Q2: x <= 0 < y
    ('[-6, -3]', '[4, 5]', '[0, 5)'),  # [-6, -5) mod 5 is [4, 5), [-5, -3] mod 5 is [0, 2], -6 mod y is [2, 4]
    ('[-1]', '[2, 3]', '[1, 2]'),
    ('[-2]', '[2, 3]', '[0, 1]'),  # 0 at y = 2, y - 2 above it
    ('(-1, 0]', '[1]', '[0, 1)'),
    ('[-4]', '(1, 2]', '[0, 2)'),  # 3y - 4 on (4/3, 2) rises towards 2
    ('[-5]', '[3, 4]', '[1, 3]'),  # one sector: 2y - 5
    ('[-5]', '[2, 3]', '[0, 5/2)'),  # 3y - 5 on [2, 5/2), 2y - 5 on [5/2, 3]
    ('[-9]', '[4, 5]', '{ [0, 1] , [3, 9/2) }'),  # 2y - 9 up to y = 5, and 3y - 9 on [4, 9/2)
    # Q3 and Q4 by the antipodal identity
    ('[-6, -3]', '[-5, -4]', '{ (-5, -3] , [-2, 0] }'),
    ('[3, 6]', '[-5, -4]', '(-5, 0]'),
    ('[9]', '[-5, -4]', '{ (-9/2, -3] , [-1, 0] }'),
    # one-point holes where two located pieces touch at a value neither attains: the v3 prototype
    # merged before deciding ends and returned [0, 5/4) etc. (proof-all-quadrants.md §1.2a)
    ('(2, 5/2)', '(1, 3/2)', '{ [0, 1/2) , (1/2, 5/4) }'),
    ('(5/2, 7/2)', '[1]', '{ [0, 1/2) , (1/2, 1) }'),
    ('[5/2]', '(1, 2)', '{ [0, 1/2) , (1/2, 5/4) }'),
    # zero-crossing dividend
    ('[-1, 1]', '[3]', '{ [0, 1] , [2, 3) }'),
    ('[-1, 1]', '[-3]', '{ (-3, -2] , [-1, 0] }'),
])
def test_examples(a, b, expected):
    assert format_cuts(modulo.mod(parse(a), parse(b))) == expected


# SCALARS: python's % is the reference, sign convention included

scalars = [-7, -6, -Fraction(7, 2), -3, -1, -Fraction(1, 3), 0, Fraction(1, 3), 1, 3, Fraction(7, 2), 6, 7,
           -2.5, 0.75, 1e-3, -1e300, INF, -INF]


def _python_mod(x, y):
    """
    python's `x % y`, except that a mixed Fraction/float pair is computed exactly and rounded once
    (python rounds the Fraction to a float first), and an infinity never makes the result float
    """
    if math.isinf(y):
        return x if (x >= 0) == (y > 0) or x == 0 else y
    if isinstance(x, float) != isinstance(y, float) and (isinstance(x, Fraction) or isinstance(y, Fraction)):
        return float(Fraction(x) % Fraction(y))
    return x % y


@pytest.mark.parametrize('x', scalars)
@pytest.mark.parametrize('y', [v for v in scalars if v != 0])
def test_scalar_matches_python(x, y):
    result = modulo.mod(one(x, True, x, True), one(y, True, y, True))
    if math.isinf(x):
        assert result == EMPTY  # python: nan
        return
    expected = _python_mod(x, y)
    assert show(result) == [(expected, True, expected, True)], (x, y, expected, show(result))
    assert type(show(result)[0][0]) is type(show(one(expected, True, expected, True))[0][0])


def test_float_result_is_float_and_exact_is_exact():
    assert show(modulo.mod(parse('[7/2]'), parse('[1]'))) == [(Fraction(1, 2), True, Fraction(1, 2), True)]
    got = show(modulo.mod(parse('[3.5]'), parse('[1]')))
    assert got == [(0.5, True, 0.5, True)] and type(got[0][0]) is float
    # a mixed pair is computed exactly and rounded once: 0.1 is not 1/10
    expected = float(Fraction(0.1) % Fraction(1, 30))
    assert show(modulo.mod(one(0.1, True, 0.1, True), parse('[1/30]'))) == [(expected, True, expected, True)]


def test_no_phantom_zero():
    """an open end at a multiple of m attains no 0: `[0.25, 0.5) % 0.5` is `[0.25, 0.5)`. v1 gave `{ [0] , [0.25, 0.5) }`
    (pinned against v1 itself until v1 was deleted, 2026-10-04)"""
    assert modulo.mod(one(0.25, True, 0.5, False), parse('[0.5]')) == normalize([piece(0.25, 0.5, True, False)])


# PROPERTIES over every sign

@settings(max_examples=150, deadline=None)
@given(a=mod_operands, b=mod_operands, rng=st.randoms(use_true_random=False))
@example(a=parse('[-3, 7]'), b=parse('[-2, 5]'), rng=random.Random(0))
@example(a=parse('[-inf, -1]'), b=parse('[2, inf]'), rng=random.Random(0))
def test_sound(a, b, rng):
    result = modulo.mod(a, b)
    for x in sample(a, 8, rng):
        for y in sample(b, 8, rng):
            for v in pointwise('mod', x, y):
                assert contains_point(result, v), (x, y, v, show(result))


def _closed(cuts):
    return normalize(piece(lo, hi) for lo, _, hi, _ in pieces(cuts))


def _exact_mod(x, y):
    """the exact value of a pair, floats read as the Fractions they denote"""
    as_exact = [Fraction(v) if isinstance(v, float) and math.isfinite(v) else v for v in (x, y)]
    return pointwise('mod', *as_exact)


@settings(max_examples=150, deadline=None)
@given(a=cut_tuples(max_pieces=2), b=cut_tuples(max_pieces=2), rng=st.randoms(use_true_random=False))
@example(a=one(-1e-20, True, -1e-20, True), b=parse('[1.0]'), rng=random.Random(0))  # python: 1.0
def test_sound_float(a, b, rng):
    """
    a float result is the exact one rounded once, so the rounded exact value of every sampled pair is
    in the result read with every end closed (a rounded end's flag is conservative, not a promise)
    """
    result = _closed(modulo.mod(a, b))
    for x in sample(a, 6, rng):
        for y in sample(b, 6, rng):
            for v in _exact_mod(x, y):
                rounded = float(v) if math.isfinite(v) else v
                assert contains_point(result, rounded) or contains_point(result, v), (x, y, v, show(result))


@settings(max_examples=150, deadline=None)
@given(a=mod_operands, b=mod_operands)
@example(a=parse('(-inf, -1]'), b=parse('[2, inf]'))
@example(a=parse('[0, inf)'), b=parse('(0, 1]'))
def test_endpoints_closed_iff_attained(a, b):
    result = modulo.mod(a, b)
    for lo, lo_closed, hi, hi_closed in pieces(result):
        assert lo_closed == attained('mod', lo, a, b), ('lo', lo, show(result))
        assert hi_closed == attained('mod', hi, a, b), ('hi', hi, show(result))


@settings(max_examples=150, deadline=None)
@given(a=mod_operands, b=mod_operands)
@example(a=parse('[3, 4]'), b=parse('(2, 3)'))  # the corner-touch zeros of G2, both corners open
@example(a=parse('[-4]'), b=parse('(1, 2]'))
def test_interior_sharpness(a, b):
    """on probes through every gap, a value is in the result iff the oracle attains it"""
    result = modulo.mod(a, b)
    for p in mod_probes(a, b, result):
        assert contains_point(result, p) == attained('mod', p, a, b), (p, show(result))


@settings(max_examples=100, deadline=None)
@given(a=mod_operands, b=mod_operands, a_sub=st.one_of(mod_operands, special_point_sets),
       b_sub=st.one_of(mod_operands, special_point_sets))
def test_isotone(a, b, a_sub, b_sub):
    small_a, small_b = intersection(a, a_sub), intersection(b, b_sub)
    assert is_subset(modulo.mod(small_a, small_b), modulo.mod(a, b))


@settings(max_examples=100, deadline=None)
@given(a=mod_operands, a2=mod_operands, b=mod_operands, b2=mod_operands)
def test_distributes_over_union(a, a2, b, b2):
    """mod is pointwise with no pole, so the image of a union is the union of images, in each argument"""
    assert modulo.mod(union(a, a2), b) == union(modulo.mod(a, b), modulo.mod(a2, b))
    assert modulo.mod(a, union(b, b2)) == union(modulo.mod(a, b), modulo.mod(a, b2))


def _neg(cuts):
    return normalize(piece(-hi, -lo, hc, lc) for lo, lc, hi, hc in pieces(cuts))


@settings(max_examples=100, deadline=None)
@given(a=mod_operands, b=mod_operands)
def test_antipodal(a, b):
    """(-x) mod (-y) = -(x mod y), the one exact sign identity"""
    assert modulo.mod(_neg(a), _neg(b)) == _neg(modulo.mod(a, b))


@settings(max_examples=100, deadline=None)
@given(a=exact_sets(min_pieces=1), k=st.integers(-3, 3), m=st.sampled_from([1, 2, Fraction(3, 2), 5]))
def test_periodic(a, k, m):
    shifted = normalize(piece(lo + k * m, hi + k * m, lc, hc) for lo, lc, hi, hc in pieces(a))
    point = one(m, True, m, True)
    assert modulo.mod(shifted, point) == modulo.mod(a, point)


# INFINITIES (D8) AND ZERO

@pytest.mark.parametrize('a, b, expected', [
    ('[3]', '[inf]', '[3]'),
    ('[-3]', '[inf]', '[inf]'),
    ('[0]', '[inf]', '[0]'),
    ('[3]', '[-inf]', '[-inf]'),
    ('[-3]', '[-inf]', '[-3]'),
    ('[-3, 2]', '[inf]', '{ [0, 2] , [inf] }'),
    ('[2, 3]', '[1, inf)', '{ [0, 3/2) , [2, 3] }'),  # 3/2 <= x - y < 2 would need y <= 3/2 < x - y
    ('[2, 3]', '[4, inf]', '[2, 3]'),
    ('[-3]', '[2, inf)', '[0, inf)'),  # 2y - 3 on [2, 3), y - 3 from y = 3 on
    ('[-3]', '[4, inf)', '[1, inf)'),
    ('[-3]', '[4, inf]', '[1, inf]'),
    ('[0, inf)', '[2, 3]', '[0, 3)'),
    ('(-inf, -1]', '[2, inf]', '[0, inf]'),
    ('(-inf, -1]', '[2, inf)', '[0, inf)'),
    ('[1, inf)', '[1, inf)', '[0, inf)'),
    ('[1, inf)', '[-3, -2]', '(-3, 0]'),  # Q4: -((-inf, -1] mod [2, 3])
    ('[-3, -2]', '[inf]', '[inf]'),
    ('[2, 3]', '[1, inf]', '{ [0, 3/2) , [2, 3] }'),  # a closed inf adds x mod inf = x, already there
])
def test_infinite_operands(a, b, expected):
    assert format_cuts(modulo.mod(parse(a), parse(b))) == expected


def test_infinite_dividend_is_clipped_with_a_warning():
    with pytest.warns(DomainClippedWarning, match='infinite points'):
        assert format_cuts(modulo.mod(parse('[1, inf]'), parse('[2]'))) == '[0, 2)'
    with pytest.warns(DomainClippedWarning):
        assert modulo.mod(parse('[inf]'), parse('[2]')) == EMPTY


def test_zero_divisor_is_clipped_with_a_warning():
    with pytest.warns(DomainClippedWarning, match='x mod 0'):
        # 5 mod y for y in (0, 2]: 5 - 2y below y = 2, then [0, y) for every smaller sector
        assert format_cuts(modulo.mod(parse('[5]'), parse('[0, 2]'))) == '[0, 5/3)'
    with pytest.warns(DomainClippedWarning):
        assert modulo.mod(parse('[5]'), parse('[0]')) == EMPTY


def test_no_warning_when_nothing_is_clipped():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        modulo.mod(parse('(0, inf)'), parse('{ [-inf, 0) , (0, inf] }'))


def test_empty_operand():
    with pytest.warns(EmptySetPropagationWarning):
        assert modulo.mod(EMPTY, parse('[1]')) == EMPTY
    with pytest.warns(EmptySetPropagationWarning):
        assert modulo.mod(parse('[1]'), EMPTY) == EMPTY


def test_attainment_is_constant_time():
    """the design notes measured 173 ms for `[1e6, 1e6+1] % [1, 1.5]` with the k loop"""
    import time
    start = time.perf_counter()
    result = modulo.mod(parse('[1000000000000, 1000000000001]'), parse('[1, 3/2]'))
    assert time.perf_counter() - start < 0.05
    assert format_cuts(result) == '[0, 3/2)'


# FLOOR, FLOORDIV, DIVMOD

@pytest.mark.parametrize('a, expected', [
    ('[1, 2)', '[1]'),
    ('[1, 2]', '{ [1] , [2] }'),
    ('(1, 2)', '[1]'),
    ('(-1/2, 1/2)', '{ [-1] , [0] }'),
    ('[inf]', '[inf]'),
    ('[-inf]', '[-inf]'),
    ('{ [5/2] , [3] }', '{ [2] , [3] }'),
    ('[2.5, 4.0)', '{ [2.0] , [3.0] }'),
])
def test_floor(a, expected):
    assert format_cuts(modulo.floor(parse(a))) == expected


@pytest.mark.parametrize('a, expected', [
    ('[0, inf)', '[0, inf)'),
    ('[0, inf]', '[0, inf]'),
    ('(-inf, 1/2]', '(-inf, 0]'),
    ('[-inf, inf]', '[-inf, inf]'),
    ('[0, 5000]', '[0, 5000]'),
])
def test_floor_hulls_with_a_warning(a, expected):
    with pytest.warns(HullWarning):
        assert format_cuts(modulo.floor(parse(a))) == expected


def test_floor_cap_counts_across_pieces():
    half = modulo.FLOOR_ENUMERATION_CAP // 2
    a = parse(f'{{ [0, {half - 1}] , [{10 * half}, {11 * half - 1}] }}')
    assert len(list(pieces(modulo.floor(a)))) == 2 * half
    with pytest.warns(HullWarning):
        modulo.floor(parse(f'{{ [0, {half}] , [{10 * half}, {11 * half}] }}'))


def _meets_unit(a, n):
    """does a hold some x with floor(x) == n?"""
    return bool(intersection(a, one(n, True, n + 1, False)))


@settings(max_examples=150, deadline=None)
@given(a=exact_sets(min_pieces=1).filter(bool), rng=st.randoms(use_true_random=False))
def test_floor_sound_and_sharp(a, rng):
    """sampled points floor into the result; without a hull, an integer is in it iff it is attained"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = modulo.floor(a)
    for x in sample(a, 20, rng):
        assert contains_point(result, math.floor(x) if math.isfinite(x) else x)
    if not caught:
        for n in range(-8, 9):
            assert contains_point(result, n) == _meets_unit(a, n), (n, show(a), show(result))
        for v in (-INF, INF):
            assert contains_point(result, v) == contains_point(a, v)


@pytest.mark.parametrize('a, b, expected', [
    ('[1, 2)', '[1]', '[1]'),
    ('[7]', '[2]', '[3]'),
    ('[-7]', '[2]', '[-4]'),
    ('[7, 9]', '[2, 3]', '{ [2] , [3] , [4] }'),
    ('[1]', '[3]', '[0]'),
    ('[-5]', '[inf]', '[-1]'),  # the limit, as python; floor(div) would give 0
    ('[-5, 5]', '[-inf]', '{ [-1] , [0] }'),
    ('[inf]', '[3]', '[inf]'),  # python: nan
])
def test_floordiv(a, b, expected):
    assert format_cuts(modulo.floordiv(parse(a), parse(b))) == expected


def _python_floordiv(x, y):
    """
    python's `x // y`, with the library's number rules: exact pairs stay exact, a mixed Fraction/float
    pair is computed exactly and rounded once, an infinity never makes the result float, and an
    infinite dividend over a finite divisor is the limit, a signed infinity (python: nan)
    """
    if math.isinf(x):
        return x if y > 0 else -x
    if math.isinf(y):
        q = -1 if (x > 0 and y < 0) or (x < 0 and y > 0) else 0
        return float(q) if isinstance(x, float) else q
    q = math.floor(Fraction(x) / Fraction(y))
    return _nearest(q) if isinstance(x, float) or isinstance(y, float) else q


def _nearest(q):
    """an exact floor rounded to the nearest double, past MAX an infinity (python's `float` raises)"""
    try:
        return float(q)
    except OverflowError:
        return math.inf if q > 0 else -math.inf


@pytest.mark.parametrize('x', scalars)
@pytest.mark.parametrize('y', [v for v in scalars if v != 0])
def test_scalar_floordiv_matches_python(x, y):
    if math.isinf(x) and math.isinf(y):
        with pytest.warns(IndeterminateResultWarning):
            assert modulo.floordiv(one(x, True, x, True), one(y, True, y, True)) == EMPTY
        return
    if not (math.isinf(x) or isinstance(x, Fraction) or isinstance(y, Fraction)):
        assert _python_floordiv(x, y) == x // y  # the reference is python's own value where it has one
    expected = _python_floordiv(x, y)
    result = modulo.floordiv(one(x, True, x, True), one(y, True, y, True))
    assert show(result) == [(expected, True, expected, True)], (x, y, expected, show(result))
    assert type(show(result)[0][0]) is type(show(one(expected, True, expected, True))[0][0])


@settings(max_examples=150, deadline=None)
@given(a=cut_tuples(max_pieces=2), b=cut_tuples(max_pieces=2), rng=st.randoms(use_true_random=False))
@example(a=parse('[1]'), b=one(0.001, True, 0.001, True), rng=random.Random(0))  # float 1 / 0.001 is 1000.0
# fuzz (run 36540588320): a floor past MAX rounds to inf, where python's float() raises
@example(a=one(Fraction(1, 2), True, 1, False), b=one(-math.inf, False, 2.2250738585e-313, True),
         rng=random.Random(0))
@pytest.mark.filterwarnings('ignore::multiinterval.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::multiinterval.errors.HullWarning')
def test_floordiv_sound_float(a, b, rng):
    """the floor of the EXACT quotient of every sampled pair is in the result (as a float if rounded)"""
    result = _closed(modulo.floordiv(a, b))
    for x in sample(a, 6, rng):
        for y in sample(b, 6, rng):
            if y == 0 or (math.isinf(x) and math.isinf(y)):
                continue  # a pole (div's rule) and an indeterminate pair
            exact = _python_floordiv(*(Fraction(v) if isinstance(v, float) and math.isfinite(v) else v
                                       for v in (x, y)))
            assert contains_point(result, exact) or contains_point(result, _nearest(exact)), (x, y, exact, show(result))


def test_floordiv_is_not_floor_div_at_an_infinite_divisor():
    """the documented oddity: -5 / inf is 0, a point with no side, so floor(div) loses the limit's -1"""
    a, b = parse('[-5]'), parse('[inf]')
    assert modulo.floor(ops.div(a, b)) == parse('[0]')
    assert modulo.floordiv(a, b) == parse('[-1]')
    # the limit keeps the infinite point continuous with its finite neighbours
    assert modulo.floordiv(a, parse('[1, inf]')) == modulo.floordiv(a, parse('[1, inf)'))
    assert format_cuts(modulo.floordiv(a, parse('[1, inf]'))) == '{ [-5] , [-4] , [-3] , [-2] , [-1] }'
    # and divmod's two parts are limits of the same finite pairs, as in python
    assert divmod(P('[-5]'), INF) == (P('[-1]'), P('[inf]'))
    assert divmod(-5, INF) == (-1.0, INF)


def test_floordiv_indeterminate_only_for_an_isolated_infinity_pair():
    with pytest.warns(IndeterminateResultWarning):
        assert modulo.floordiv(parse('{ [1] , [inf] }'), parse('[inf]')) == parse('[0]')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # the box [inf] // [5, inf] has values (inf // 5), so nothing warns
        assert modulo.floordiv(parse('[inf]'), parse('[5, inf]')) == parse('[inf]')


def test_floordiv_at_a_zero_divisor_follows_div():
    with pytest.warns(HullWarning):
        assert format_cuts(modulo.floordiv(parse('[1]'), parse('[0, 1]'))) == '[1, inf]'


def test_class_dunders():
    a = P('[3, 7]')
    assert a % 5 == P('{ [0, 2] , [3, 5) }')
    assert 7 % P('[3]') == P('[1]')
    assert P('[1, 2)') // 1 == P('[1]')
    assert 7 // P('[2]') == P('[3]')
    assert divmod(P('[7]'), 2) == (P('[3]'), P('[1]'))
    assert divmod(7, P('[2]')) == (P('[3]'), P('[1]'))
    assert P('[-1/2, 2)').floor() == P('{ [-1] , [0] , [1] }')
    assert P('[1]').__mod__('x') is NotImplemented
    with pytest.raises(TypeError):
        divmod(P('[1]'), 'x')


def test_divmod_warns_once_on_an_empty_operand():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert divmod(MultiInterval(), 2) == (MultiInterval(), MultiInterval())
    assert [w.category for w in caught] == [EmptySetPropagationWarning]
