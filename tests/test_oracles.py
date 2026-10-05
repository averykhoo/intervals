"""
self-tests for tests/oracles.py, the independent reference the arithmetic tests are checked against

three angles: the plan's worked examples as exact attained / not-attained facts, brute force over
small exact grids (complete where the grid provably is), and the sampler's contract
"""
import math
import random
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval.fmt import parse
from multiinterval.kernel import EMPTY
from multiinterval.kernel import contains_point
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from tests import oracles
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import power_image
from tests.oracles import sample
from tests.oracles import witness
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import probe_points

P = parse
inf = math.inf
half = Fraction(1, 2)

# every half from -2 to 2, plus the infinities: operand endpoints for the brute-force checks
GRID = [-inf, *(Fraction(k, 2) for k in range(-4, 5)), inf]
# a finer exact grid of candidate values and points
FINE = [-inf, *(Fraction(k, 4) for k in range(-24, 25)), inf]
# quarters over a range wide enough to make the add/sub search complete (see that test)
SEARCH = [-inf, *(Fraction(k, 4) for k in range(-32, 33)), inf]
EXPONENTS = [-3, -2, -1, 0, 1, 2, 3]

grid_sets = cut_tuples(values=st.sampled_from(GRID), max_pieces=2).filter(bool)
point_sets = st.lists(st.sampled_from(GRID), min_size=1, max_size=4).map(
    lambda xs: normalize(piece(x, x) for x in xs))


def points_of(cuts, rng, n=12):
    """grid points of the set plus sampled ones"""
    return [x for x in FINE if contains_point(cuts, x)][::3] + sample(cuts, n, rng)


# WORKED EXAMPLES
# (op, A, B, result): the oracle's attained set must equal `result` at every probe point, which
# includes every endpoint, so this pins both locations and closedness

EXAMPLES = [
    # direction comes from the set, flags propagate through infinity (D1)
    ('reciprocal', '[-1, 0]', None, '[-inf, -1]'),
    ('reciprocal', '[-1, 1]', None, '{[-inf, -1], [1, inf]}'),
    ('reciprocal', '[1, inf]', None, '[0, 1]'),
    ('reciprocal', '[1, inf)', None, '(0, 1]'),
    ('reciprocal', '(-1, 0)', None, '(-inf, -1)'),
    ('reciprocal', '[inf]', None, '[0]'),
    ('reciprocal', '{[0], [1, 2]}', None, '[1/2, 1]'),
    ('reciprocal', '[-1, 0)', None, '(-inf, -1]'),
    ('reciprocal', '{[-1, 0), [0]}', None, '[-inf, -1]'),
    ('reciprocal', '[0, 1]', None, '[1, inf]'),
    ('reciprocal', '(0, 1]', None, '[1, inf)'),
    # shape then attainment: v1 got (0, 3)
    ('mul', '[0, 1]', '(2, 3)', '[0, 3)'),
    # exact division
    ('div', '[1]', '[3]', '[1/3]'),
    ('div', '[6]', '[3]', '[2]'),
    # D2: indeterminate corners of a non-degenerate box take the limit along the box
    ('mul', '[-inf, -1]', '[0]', '[0]'),
    ('mul', '[-inf]', '[0, 1]', '[-inf]'),
    ('mul', '[-inf, -1]', '[0, 1]', '[-inf, 0]'),
    ('div', '[1, inf]', '[1, inf]', '[0, inf]'),
    ('sub', '[1, inf]', '[1, inf]', '[-inf, inf]'),
    ('sub', '[inf]', '[1, inf]', '[inf]'),
    ('sub', '[1, inf]', '[inf]', '[-inf]'),
    # D7: a box that is an indeterminate point gives nothing
    ('sub', '[inf]', '[inf]', '{}'),
    ('add', '[inf]', '[-inf]', '{}'),
    ('mul', '[0]', '[inf]', '{}'),
    ('div', '[0]', '[0]', '{}'),
    ('reciprocal', '[0]', None, '{}'),
    ('div', '[inf]', '[-inf]', '{}'),
    ('div', '[3]', '[0]', '{}'),
    # an infinite endpoint is closed iff attained
    ('add', '[inf]', '(1, 2)', '[inf]'),
    ('add', '[1, inf]', '[0, 1)', '[1, inf]'),
    ('mul', '[inf]', '(1, 2)', '[inf]'),
    ('add', '(1, inf]', '[0]', '(1, inf]'),
    ('add', '(1, inf)', '[0, 1]', '(1, inf)'),
    # interior sharpness: mul splits at zero
    ('mul', '[-1, 1]', '[inf]', '{[-inf], [inf]}'),
    ('mul', '[-1, 1]', '[-inf, inf]', '[-inf, inf]'),
    # poles in the denominator of div
    ('div', '[1, 2]', '[-1, 1]', '{[-inf, -1], [1, inf]}'),
    ('div', '[1, 2]', '[0, 1]', '[1, inf]'),
    ('div', '[-2, -1]', '[0, 1]', '[-inf, -1]'),
    ('div', '[0, 1]', '[0, 1]', '[0, inf]'),
    ('div', '[inf]', '[0, 1]', '[inf]'),
    ('div', '[-1, 1]', '[inf]', '[0]'),
    ('div', '(0, 1]', '{[-1], [0]}', '[-1, 0)'),
    ('div', '[-1, 1]', '[2, 4]', '[-1/2, 1/2]'),
    # unary
    ('neg', '(1, inf]', None, '[-inf, -1)'),
    ('pos', '(1, inf]', None, '(1, inf]'),
    ('abs', '[-2, 1)', None, '[0, 2]'),
    ('abs', '[-inf, -1)', None, '(1, inf]'),
    ('abs', '{[-3, -2), (2, 3]}', None, '(2, 3]'),
    # pow: B is the exponent
    ('pow', '[-2, 1)', 2, '[0, 4]'),
    ('pow', '(-2, 1)', 2, '[0, 4)'),
    ('pow', '[-inf, -1]', 3, '[-inf, -1]'),
    ('pow', '[-inf, -1]', 2, '[1, inf]'),
    ('pow', '[-1, 2]', -1, '{[-inf, -1], [1/2, inf]}'),
    ('pow', '[-1, 1]', -2, '[1, inf]'),
    ('pow', '[-1, 1]', -3, '{[-inf, -1], [1, inf]}'),
    ('pow', '[0]', -1, '{}'),
    ('pow', '[inf]', 0, '[1]'),
    ('pow', '[2, 3]', 0, '[1]'),
    ('pow', '{}', 0, '{}'),
    # empties propagate
    ('add', '{}', '[1, 2]', '{}'),
    ('div', '[1, 2]', '{}', '{}'),
    ('reciprocal', '{}', None, '{}'),
]


@pytest.mark.parametrize('op, a, b, result', EXAMPLES)
def test_worked_example(op, a, b, result):
    a, r = P(a), P(result)
    b = P(b) if isinstance(b, str) else b
    operands = [a, b] if isinstance(b, tuple) else [a]
    extra = [Fraction(1, 3), Fraction(-1, 3), Fraction(1, 4), 5, -5]
    for v in probe_points(r, *operands) + extra:
        assert attained(op, v, a, b) == contains_point(r, v), (op, v)


@pytest.mark.parametrize('op, v, a, b, expected', [
    ('reciprocal', -inf, '[-1, 0]', None, True),
    ('reciprocal', inf, '[-1, 0]', None, False),
    ('reciprocal', -inf, '(-1, 0)', None, False),
    ('add', inf, '[1, inf]', '[0, 1)', True),
    ('add', inf, '[1, inf)', '[0, 1)', False),
    ('mul', 0, '[inf]', '(0, 1]', False),
    ('mul', inf, '[inf]', '(0, 1]', True),
    ('mul', 0, '[0, 1]', '(2, 3)', True),
    ('mul', 3, '[0, 1]', '(2, 3)', False),
    ('sub', 0, '[inf]', '[inf]', False),
    ('div', 0, '[0]', '[0]', False),
    ('div', inf, '[1]', '[0]', False),
    ('div', -inf, '[1]', '[0]', False),
])
def test_endpoint_fact(op, v, a, b, expected):
    assert attained(op, v, P(a), P(b) if b else None) is expected


def test_witnesses_are_exact():
    assert witness('div', Fraction(1, 3), P('[1]'), P('[3]')) == (1, 3)
    x, y = witness('mul', 2, P('[0, 1]'), P('(2, 3)'))
    assert x * y == 2 and not isinstance(x, float) and not isinstance(y, float)
    assert witness('reciprocal', inf, P('[0, 1]')) == (0,)
    assert witness('add', inf, P('[inf]'), P('(1, 2)'))[0] == inf


def test_float_operands_are_read_exactly():
    # 0.1 is not 1/10; the oracle decides on the Fraction the float denotes
    assert attained('pos', Fraction(0.1), P('[0.1]'))
    assert not attained('pos', Fraction(1, 10), P('[0.1]'))
    assert attained('add', 0.30000000000000004, P('[0.1]'), P('[0.2]')) is False
    assert attained('add', Fraction(0.1) + Fraction(0.2), P('[0.1]'), P('[0.2]'))


# POINTWISE TABLE

@pytest.mark.parametrize('op, x, y, expected', [
    ('add', inf, -inf, []),
    ('add', -inf, inf, []),
    ('add', inf, inf, [inf]),
    ('add', inf, -5, [inf]),
    ('sub', inf, inf, []),
    ('sub', inf, -inf, [inf]),
    ('mul', 0, inf, []),
    ('mul', -inf, 0, []),
    ('mul', -inf, -2, [inf]),
    ('mul', -inf, half, [-inf]),
    ('div', inf, inf, []),
    ('div', -inf, 2, [-inf]),
    ('div', 3, -inf, [0]),
    ('div', 1, 3, [Fraction(1, 3)]),
    ('div', 6, 3, [2]),
    ('pos', inf, None, [inf]),
    ('neg', -inf, None, [inf]),
    ('abs', -inf, None, [inf]),
])
def test_pointwise_table(op, x, y, expected):
    out = pointwise(op, x, y)
    assert out == expected
    assert [type(v) for v in out] == [type(v) for v in expected]


def test_pointwise_exact_zero_at_infinity():
    # python gives Fraction(1) / inf == 0.0; the table gives an exact 0
    assert type(pointwise('div', Fraction(1, 3), inf)[0]) is int
    assert type(pointwise('reciprocal', inf, a=P('[inf]'))[0]) is int


@pytest.mark.parametrize('b, expected', [
    ('[-1, 0]', [-inf]),
    ('[0, 1]', [inf]),
    ('(-1, 1)', [-inf, inf]),
    ('{[-1, 0), [0]}', [-inf]),
    ('{[-1, 0), (0, 1]}', None),  # 0 is not in b
    ('[0]', []),
])
def test_pointwise_pole_direction(b, expected):
    if expected is None:
        with pytest.raises(ValueError):
            pointwise('div', 2, 0, b=P(b))
        return
    assert pointwise('div', 2, 0, b=P(b)) == expected
    assert pointwise('div', -2, 0, b=P(b)) == [-v for v in expected]
    assert pointwise('div', inf, 0, b=P(b)) == expected
    assert pointwise('div', 0, 0, b=P(b)) == []
    assert pointwise('reciprocal', 0, a=P(b)) == expected


def test_pointwise_pow():
    assert pointwise('pow', -inf, 2) == [inf]
    assert pointwise('pow', -inf, 3) == [-inf]
    assert pointwise('pow', Fraction(-1, 2), 3) == [Fraction(-1, 8)]
    assert pointwise('pow', inf, 0) == [1]
    assert pointwise('pow', 0, -2, a=P('[-1, 1]')) == [inf]  # the pole sits in [0, 1] = [-1, 1] ** 2
    assert pointwise('pow', 0, -1, a=P('[-1, 1]')) == [-inf, inf]
    assert pointwise('pow', 0, -1, a=P('[0]')) == []
    assert pointwise('pow', 2, -2, a=P('[2]')) == [Fraction(1, 4)]
    with pytest.raises(TypeError):
        pointwise('pow', 2, True)
    with pytest.raises(TypeError):
        attained('pow', 4, P('[2]'), 2.0)


def test_power_image():
    assert power_image(P('[-2, 1)'), 2) == P('[0, 4]')
    assert power_image(P('(-2, 1)'), 2) == P('[0, 4)')
    assert power_image(P('(-inf, -1]'), 2) == P('[1, inf)')
    assert power_image(P('[-inf, 2)'), 3) == P('[-inf, 8)')
    assert power_image(P('{(-2, -1), (1, 2)}'), 2) == P('(1, 4)')


# BRUTE FORCE

@settings(max_examples=80, deadline=None)
@given(grid_sets, grid_sets, st.sampled_from(oracles.BINARY), st.randoms(use_true_random=False))
def test_every_sampled_binary_value_is_attained(a, b, op, rng):
    """no false negatives on the points the arithmetic tests will actually draw"""
    for x in points_of(a, rng):
        for y in points_of(b, rng):
            for v in pointwise(op, x, y, a, b):
                assert attained(op, v, a, b), (x, y, v)


@settings(max_examples=120, deadline=None)
@given(grid_sets, st.sampled_from(oracles.UNARY + ('pow',)), st.sampled_from(EXPONENTS),
       st.randoms(use_true_random=False))
def test_every_sampled_unary_value_is_attained(a, op, n, rng):
    y = n if op == 'pow' else None
    for x in points_of(a, rng, n=40):
        for v in pointwise(op, x, y, a):
            assert attained(op, v, a, y), (x, v)


@settings(max_examples=120, deadline=None)
@given(grid_sets, grid_sets, st.sampled_from(oracles.BINARY + oracles.UNARY))
def test_every_witness_is_a_defined_pair(a, b, op):
    """no false positives: a True answer always comes with a checkable pair"""
    for v in FINE:
        found = witness(op, v, a, b if op in oracles.BINARY else None)
        if found is None:
            continue
        assert all(not isinstance(p, float) or math.isinf(p) for p in found), found
        if op in oracles.BINARY:
            x, y = found
            assert contains_point(a, x) and contains_point(b, y), found
            assert v in pointwise(op, x, y, a, b), (found, v)
        else:
            assert contains_point(a, found[0]), found
            assert v in pointwise(op, found[0], a=a), (found, v)


@settings(max_examples=200, deadline=None)
@given(point_sets, point_sets, st.sampled_from(oracles.OPS), st.sampled_from(EXPONENTS))
def test_point_sets_agree_with_enumeration(a, b, op, n):
    """on finite sets the brute enumeration is complete, so attained must match it exactly"""
    xs = [lo for lo, _, _, _ in pieces(a)]
    ys = [lo for lo, _, _, _ in pieces(b)]
    if op in oracles.BINARY:
        values = {v for x in xs for y in ys for v in pointwise(op, x, y, a, b)}
        arg = b
    else:
        arg = n if op == 'pow' else None
        values = {v for x in xs for v in pointwise(op, x, arg, a)}
    for v in set(FINE) | values | {Fraction(1, 3), Fraction(-2, 3), 9, 27, -8}:
        assert attained(op, v, a, arg) == (v in values), v


@settings(max_examples=150, deadline=None)
@given(grid_sets, grid_sets, st.sampled_from(['add', 'sub']))
def test_add_sub_agree_with_a_complete_grid_search(a, b, op):
    """
    for half-grid operands and a half-grid v, the set of witnesses x is an interval with half-grid
    ends in [-6, 6]: a point of the half grid, or wide enough to hold a quarter-grid point of
    [-8, 8] (an open ray from 6 holds 25/4). so a search over SEARCH is complete for these v
    """
    xs = [x for x in SEARCH if contains_point(a, x)]
    ys = [y for y in SEARCH if contains_point(b, y)]
    found = {v for x in xs for y in ys for v in pointwise(op, x, y, a, b)}
    for v in GRID + [Fraction(k, 2) for k in range(-8, 9)]:
        assert attained(op, v, a, b) == (v in found), v


@settings(max_examples=150, deadline=None)
@given(grid_sets, st.sampled_from([1, 2, 3]))
def test_pow_agrees_with_exact_roots(a, n):
    """v = r ** n for a quarter-grid r: the only candidate bases are r and, for even n, -r"""
    for r in FINE:
        v = pointwise('pow', r, n)[0]
        expected = contains_point(a, r) or (n % 2 == 0 and contains_point(a, -r))
        assert attained('pow', v, a, n) == expected, r


@settings(max_examples=100, deadline=None)
@given(exact_cut_tuples, exact_cut_tuples, st.sampled_from(oracles.BINARY), st.randoms(use_true_random=False))
def test_general_exact_operands(a, b, op, rng):
    """the same two checks on the shared exact strategy, whose endpoints are off the half grid"""
    for x in sample(a, 8, rng):
        for y in sample(b, 8, rng):
            for v in pointwise(op, x, y, a, b):
                assert attained(op, v, a, b)
                wx, wy = witness(op, v, a, b)
                assert contains_point(a, wx) and contains_point(b, wy)
                assert v in pointwise(op, wx, wy, a, b)


def test_empty_operands_attain_nothing():
    for op in oracles.OPS:
        b = 2 if op == 'pow' else P('[1, 2]')
        assert not attained(op, 1, EMPTY, b)
    for op in oracles.BINARY:
        assert not attained(op, 1, P('[1, 2]'), EMPTY)


# SAMPLER

@settings(max_examples=100, deadline=None)
@given(cut_tuples(), st.randoms(use_true_random=False))
def test_sample_stays_in_the_set(cuts, rng):
    points = sample(cuts, 30, rng)
    assert len(points) == (30 if cuts else 0)
    for p in points:
        assert contains_point(cuts, p), p


@settings(max_examples=100, deadline=None)
@given(exact_cut_tuples, st.randoms(use_true_random=False))
def test_sample_of_exact_set_is_exact(cuts, rng):
    for p in sample(cuts, 30, rng):
        assert not isinstance(p, float) or math.isinf(p), p


@pytest.mark.parametrize('text, must, never', [
    ('[-inf, 0]', {-inf, 0}, {inf}),
    ('{[-inf, -1), (1, inf]}', {-inf, inf}, {-1, 1}),
    ('(-inf, inf)', set(), {-inf, inf}),
    ('[inf]', {inf}, set()),
    ('{[-inf], (0, 1)}', {-inf}, {0, 1}),
    ('[1/3, 2.5]', {Fraction(1, 3), 2.5}, set()),
])
def test_sample_draws_closed_endpoints_and_never_open_ones(text, must, never):
    points = sample(P(text), 300, random.Random(0))
    assert must <= set(points)
    assert not never & set(points)


def test_sample_favours_points_near_the_ends():
    rng = random.Random(1)
    points = sample(P('(0, 1)'), 300, rng)
    assert min(points) < Fraction(1, 1000) and max(points) > 1 - Fraction(1, 1000)
    far = sample(P('(-inf, 0)'), 300, rng)
    assert min(far) < -10 ** 6 and max(far) > -Fraction(1, 1000)


def test_a_float_negative_power_is_python_s_value():
    """a float x ** -n is python's `x ** -n`, one rounding: `1 / x ** n` rounds twice and gave 2.7777777777777777
    for 0.6 ** -2, whose nearest double is 2.777777777777778 (M14-breadth's fuzz x10, 2026-10-02)"""
    a = normalize([piece(0.6, 0.6)])
    assert pointwise('pow', 0.6, -2, a=a) == [0.6 ** -2] == [2.777777777777778]
