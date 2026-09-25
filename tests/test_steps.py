"""
the step functions: ceil, trunc, round (ties to even), round_ties_away and sign (intervals.steps; floor
has its own tests in tests/test_modulo.py, which now run through the same engine)

the oracle is each function's preimages, written out independently: grid value n is in the result
iff the operand meets the preimage of n. on single points the functions must agree with python's
`math.ceil`, `math.trunc`, `round` and, for round_ties_away, the decimal module's ROUND_HALF_UP.
"""
import math
import warnings
from decimal import ROUND_HALF_UP
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import steps
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.fmt import format_cuts
from intervals.fmt import parse
from intervals.kernel import EMPTY
from intervals.kernel import contains_point
from intervals.kernel import intersection
from intervals.kernel import normalize
from intervals.kernel import piece
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples

INF = math.inf
HALF = Fraction(1, 2)
NAMES = ('floor', 'ceil', 'trunc', 'round', 'round_ties_away', 'sign')


def preimage(name: str, n: int):
    """the points f maps to n, as a cut tuple"""
    if name == 'floor':
        p = piece(n, n + 1, True, False)
    elif name == 'ceil':
        p = piece(n - 1, n, False, True)
    elif name == 'trunc':
        p = piece(n, n + 1, True, False) if n > 0 else piece(n - 1, n, False, True) if n < 0 else piece(-1, 1, False, False)
    elif name == 'round':
        p = piece(n - HALF, n + HALF, n % 2 == 0, n % 2 == 0)
    elif name == 'round_ties_away':
        p = piece(n - HALF, n + HALF, n > 0, n < 0) if n else piece(-HALF, HALF, False, False)
    else:  # sign
        p = {-1: piece(-INF, 0, True, False), 0: piece(0, 0), 1: piece(0, INF, False, True)}[n]
    return normalize([p])


def show(cuts) -> str:
    return format_cuts(cuts)


# EXAMPLES

@pytest.mark.parametrize('name, a, expected', [
    ('ceil', '(1, 3]', '{ [2] , [3] }'),
    ('ceil', '[1, 3)', '{ [1] , [2] , [3] }'),
    ('ceil', '(-1/2, 1/2)', '{ [0] , [1] }'),
    ('ceil', '[inf]', '[inf]'),
    ('ceil', '{ [-inf] , [5/2] }', '{ [-inf] , [3] }'),
    ('trunc', '(-2, 2)', '{ [-1] , [0] , [1] }'),
    ('trunc', '[-2, 2]', '{ [-2] , [-1] , [0] , [1] , [2] }'),
    ('trunc', '(-1, 1)', '[0]'),
    ('trunc', '[-5/2, -2)', '[-2]'),
    ('round', '[1/2, 5/2]', '{ [0] , [1] , [2] }'),
    ('round', '(1/2, 5/2)', '{ [1] , [2] }'),
    ('round', '(1/2, 3/2)', '[1]'),
    ('round', '[3/2]', '[2]'),
    ('round', '[-3/2]', '[-2]'),
    ('round', '(-1/2, 1/2)', '[0]'),
    ('round_ties_away', '[1/2, 5/2]', '{ [1] , [2] , [3] }'),
    ('round_ties_away', '[-5/2, -1/2]', '{ [-3] , [-2] , [-1] }'),
    ('round_ties_away', '(-1/2, 1/2)', '[0]'),
    ('round_ties_away', '[1/2, 3/2)', '[1]'),
    ('sign', '[-2, 0]', '{ [-1] , [0] }'),
    ('sign', '(0, inf]', '[1]'),
    ('sign', '[-inf, inf]', '{ [-1] , [0] , [1] }'),
    ('sign', '(-inf, 0)', '[-1]'),
    ('sign', '[0]', '[0]'),
    ('sign', '[-inf]', '[-1]'),
    ('sign', '[-2.5, 3.0]', '{ [-1.0] , [0.0] , [1.0] }'),
    ('ceil', '[2.5, 4.0]', '{ [3.0] , [4.0] }'),
])
def test_examples(name, a, expected):
    assert show(steps.step(name, parse(a))) == expected


@pytest.mark.parametrize('name, a, expected', [
    ('ceil', '[0, inf)', '[0, inf)'),
    ('ceil', '(-inf, 1/2]', '(-inf, 1]'),
    ('trunc', '[-inf, inf]', '[-inf, inf]'),
    ('round', '[0, 5000]', '[0, 5000]'),
    ('round_ties_away', '(-inf, 0]', '(-inf, 0]'),
])
def test_hulls_with_a_warning(name, a, expected):
    with pytest.warns(HullWarning):
        assert show(steps.step(name, parse(a))) == expected


def test_sign_never_hulls():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert show(steps.sign(parse('[-inf, inf]'))) == '{ [-1] , [0] , [1] }'


def test_cap_counts_across_pieces():
    half = steps.ENUMERATION_CAP // 2
    a = parse(f'{{ [0, {half - 1}] , [{10 * half}, {11 * half - 1}] }}')
    assert len(steps.ceil(a)) // 2 == 2 * half
    with pytest.warns(HullWarning):
        steps.ceil(parse(f'{{ [0, {half}] , [{10 * half}, {11 * half}] }}'))


def test_empty_warns():
    with pytest.warns(EmptySetPropagationWarning):
        assert steps.sign(EMPTY) == EMPTY


def test_bad_arguments():
    with pytest.raises(ValueError):
        steps.step('cbrt', parse('[1]'))
    with pytest.raises(TypeError):
        steps.step('ceil', parse('[1]'), ndigits=2)
    with pytest.raises(TypeError):
        steps.step('round', parse('[1]'), ndigits=2.0)


# NDIGITS

@pytest.mark.parametrize('a, ndigits, expected', [
    ('[1/8, 3/8]', 1, '{ [1/10] , [1/5] , [3/10] , [2/5] }'),
    ('[0.125, 0.135]', 2, '{ [0.12] , [0.13] , [0.14] }'),
    ('[1250, 1350]', -2, '{ [1200] , [1300] , [1400] }'),
    ('[150]', -2, '[200]'),
    ('[250]', -2, '[200]'),
])
def test_ndigits(a, ndigits, expected):
    assert show(steps.round_(parse(a), ndigits)) == expected


@pytest.mark.parametrize('x', [2.675, 0.125, 0.375, -1.005, 12345.678, 1e-5, 2.5, 3.5])
@pytest.mark.parametrize('ndigits', [None, 0, 1, 2, 3, -1])
def test_round_matches_python_on_a_float_point(x, ndigits):
    """python rounds the float's exact value (2.675 is below 2.675, so it goes to 2.67)"""
    expected = round(x, ndigits) if ndigits is not None else float(round(x))
    assert MultiInterval(x).round(ndigits) == MultiInterval(float(expected))


# AGREEMENT WITH PYTHON ON POINTS

def _points():
    return st.one_of(st.integers(-50, 50), st.fractions(min_value=-20, max_value=20, max_denominator=8),
                     st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))


def _half_away(x) -> int:
    """decimal's ROUND_HALF_UP is ties away from zero; the division is exact for these small operands"""
    q = Fraction(x)
    return int((Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


@given(x=_points())
def test_points_agree_with_python(x):
    typed = float if isinstance(x, float) else int
    a = MultiInterval(x)
    assert a.ceil() == MultiInterval(typed(math.ceil(x)))
    assert a.trunc() == MultiInterval(typed(math.trunc(x)))
    assert a.round() == MultiInterval(typed(round(x)))
    assert a.round_ties_away() == MultiInterval(typed(_half_away(x)))
    assert a.sign() == MultiInterval(typed((x > 0) - (x < 0)))
    assert math.floor(a) == a.floor() and math.ceil(a) == a.ceil()
    assert math.trunc(a) == a.trunc() and round(a) == a.round() and round(a, 1) == a.round(1)


# SOUND AND SHARP

@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=150, deadline=None)
@given(a=exact_cut_tuples.filter(bool), rng=st.randoms(use_true_random=False))
def test_sound_and_sharp(name, a, rng):
    """without a hull, n is in the result iff the operand meets n's preimage"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = steps.step(name, a)
    for x in sample(a, 20, rng):
        if x in (-INF, INF):
            v = x if name != 'sign' else (1 if x > 0 else -1)
        else:
            v = {'floor': math.floor, 'ceil': math.ceil, 'trunc': math.trunc, 'round': round,
                 'round_ties_away': _half_away, 'sign': lambda y: (y > 0) - (y < 0)}[name](x)
        assert contains_point(result, v), (name, show(a), x, v, show(result))
    if not caught:
        for n in (range(-1, 2) if name == 'sign' else range(-8, 9)):
            assert contains_point(result, n) == bool(intersection(a, preimage(name, n))), (name, n, show(a), show(result))
        if name != 'sign':
            for v in (-INF, INF):
                assert contains_point(result, v) == contains_point(a, v)


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=50, deadline=None)
@given(a=cut_tuples(max_pieces=3))
def test_float_pieces_give_floats(name, a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = steps.step(name, a)
    float_input = any(isinstance(c.value, float) and math.isfinite(c.value) for c in a)
    if not float_input:
        assert not any(isinstance(c.value, float) and math.isfinite(c.value) for c in result)


# THE CLASS

def test_dunders_return_sets():
    a = MultiInterval.parse('[-5/2, 7/2)')
    assert math.floor(a) == MultiInterval.parse('{ [-3] , [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert math.ceil(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] , [4] }')
    assert math.trunc(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert round(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert round(MultiInterval.parse('[1.25, 1.35]'), 1) == MultiInterval.parse('{ [1.2] , [1.3] , [1.4] }')


def test_outward_round_to_digits_encloses_the_grid_point():
    """a grid point that is not a double becomes the open piece between its neighbours, outward"""
    assert OutwardMultiInterval(0.25).round(1) == OutwardMultiInterval.parse('(0.19999999999999998, 0.2)')
    assert MultiInterval(0.25).round(1) == MultiInterval.parse('[0.2]')
