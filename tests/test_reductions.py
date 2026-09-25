"""
the reductions (intervals.reductions): a random differential against Fraction arithmetic

each result must be the exact value, computed here with Fraction, rounded once in the asked
direction. "rounded" is checked from its definition on the result's neighbouring doubles
(`is_rounded`), not by calling the package's rounding, and `math.fsum` (correctly rounded to nearest)
is a second, independent oracle for float sums. the itf1788 vectors of the four ops are the
`@example`s.
"""
import math
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import dot
from intervals import sum_
from intervals import sum_abs
from intervals import sum_sqr

INF = math.inf
NAN = math.nan
MAX = 1.7976931348623157e308
TOP = Fraction(2) ** 1024  # where inf sits when rounding to nearest: one ulp past MAX


# THE ORACLE

def _place(f: float) -> Fraction:
    return TOP if f == INF else -TOP if f == -INF else Fraction(f)


def _even(f: float) -> bool:
    """the last significand bit is 0 (ties go there); ±inf counts as even, like 2 ** 1024"""
    if math.isinf(f) or f == 0:
        return True
    return (Fraction(f) / Fraction(math.ulp(f))).numerator % 2 == 0


def is_rounded(r: float, exact: Fraction, rounding: str) -> bool:
    """r is `exact` rounded to a double: the largest below (down), the smallest above (up), or the
    nearest, ties to even, with ±inf standing at ±2 ** 1024 (nearest)"""
    below, above = math.nextafter(r, -INF), math.nextafter(r, INF)
    if rounding == 'down':
        if r == -INF:
            return exact < -MAX
        return r != INF and Fraction(r) <= exact and (above == INF or exact < Fraction(above))
    if rounding == 'up':
        if r == INF:
            return exact > MAX
        return r != -INF and Fraction(r) >= exact and (below == -INF or exact > Fraction(below))
    distance = abs(exact - _place(r))
    for n in {below, above} - {r}:
        other = abs(exact - _place(n))
        if other < distance or (other == distance and not _even(r)):
            return False
    return True


def test_is_rounded():
    tenth = Fraction(1, 10)
    assert is_rounded(0.1, tenth, 'nearest') and is_rounded(0.1, tenth, 'up')
    assert is_rounded(0.09999999999999999, tenth, 'down') and not is_rounded(0.1, tenth, 'down')
    assert not is_rounded(0.09999999999999999, tenth, 'nearest')
    assert is_rounded(0.0, Fraction(math.ulp(0.0)) / 2, 'nearest')  # a tie, to the even 0
    assert not is_rounded(math.ulp(0.0), Fraction(math.ulp(0.0)) / 2, 'nearest')
    assert is_rounded(INF, TOP - Fraction(2) ** 970, 'nearest')  # the tie past MAX goes to inf
    assert is_rounded(MAX, TOP - Fraction(2) ** 970 - 1, 'nearest')
    assert is_rounded(MAX, TOP * 2, 'down') and is_rounded(INF, TOP * 2, 'up')
    assert is_rounded(-INF, -TOP * 2, 'down') and not is_rounded(-MAX, -TOP * 2, 'down')


# THE DIFFERENTIAL

finite = st.one_of(
    st.integers(),
    st.fractions(max_denominator=10 ** 6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(min_value=-4, max_value=4, allow_nan=False),
)
roundings = st.sampled_from(['nearest', 'down', 'up'])

SUMS = {
    'sum_': (sum_, lambda x: x),
    'sum_abs': (sum_abs, abs),
    'sum_sqr': (sum_sqr, lambda x: x * x),
}


@settings(max_examples=300, deadline=None)
@given(name=st.sampled_from(sorted(SUMS)), xs=st.lists(finite, max_size=12), rounding=roundings)
@example(name='sum_', xs=[1.0, 2.0, 3.0], rounding='nearest')  # itf1788: 6.0
@example(name='sum_abs', xs=[1.0, -2.0, 3.0], rounding='nearest')  # 6.0
@example(name='sum_sqr', xs=[1.0, 2.0, 3.0], rounding='nearest')  # 14.0
@example(name='sum_', xs=[1e100, 1.0, -1e100], rounding='nearest')
@example(name='sum_', xs=[0.1, 0.1, 0.1], rounding='nearest')  # a tie, to the even side
@example(name='sum_sqr', xs=[1e200, 1e200], rounding='down')  # past the largest double
@example(name='sum_', xs=[MAX, 2.0 ** 970], rounding='nearest')  # exactly the tie past MAX
@example(name='sum_', xs=[], rounding='up')
def test_sums_round_the_exact_value_once(name, xs, rounding):
    op, term = SUMS[name]
    r = op(xs, rounding=rounding)
    assert isinstance(r, float)
    assert is_rounded(r, sum((term(Fraction(x)) for x in xs), Fraction(0)), rounding), (r, xs)


@settings(max_examples=300, deadline=None)
@given(pairs=st.lists(st.tuples(finite, finite), max_size=12), rounding=roundings)
@example(pairs=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], rounding='nearest')  # itf1788: 14.0
@example(pairs=[(float(2 ** 52 + 1), float(2 ** 52 - 1)), (2.0 ** 104, -1.0)], rounding='nearest')  # -1.0
@example(pairs=[(2.0 ** 1000, 2.0 ** 24), (-(2.0 ** 1000), 2.0 ** 24), (3, Fraction(1, 3))], rounding='up')
def test_dot_rounds_the_exact_value_once(pairs, rounding):
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    r = dot(xs, ys, rounding=rounding)
    assert isinstance(r, float)
    assert is_rounded(r, sum((Fraction(x) * Fraction(y) for x, y in pairs), Fraction(0)), rounding), (r, pairs)


@settings(max_examples=300, deadline=None)
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=12))
@example(xs=[1.0, 2.0, 3.0])
@example(xs=[1e100, 1.0, -1e100])
@example(xs=[1e16, 1.0, 1.0])
def test_sum_agrees_with_fsum(xs):
    for op, terms in ((sum_, xs), (sum_abs, [abs(x) for x in xs])):
        try:
            expected = math.fsum(terms)
        except OverflowError:  # fsum overflowed on the way; nothing to compare
            continue
        assert op(xs) == expected, (op.__name__, xs)


# SPECIAL VALUES: ±inf are points; nan and a sum or product without a value raise

special = st.one_of(finite, st.sampled_from([INF, -INF, NAN, 0, 0.0]))


def _infinite(x) -> bool:
    return x in (INF, -INF)


NAN_OPERAND = 'nan is not a point'
NO_SUM = r'inf \+ -inf has no value'
NO_PRODUCT = r'0 \* inf has no value'


def _raises_or_equals(call, expected):
    """`expected` is a float, or the message of the ValueError expected"""
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            call()
    else:
        assert call() == expected


def _infinite_sum(terms):
    """the value by the rule, from the infinite terms: no value for both signs, else that inf"""
    infinities = set(terms)
    return NO_SUM if len(infinities) == 2 else infinities.pop()


@settings(max_examples=300, deadline=None)
@given(name=st.sampled_from(sorted(SUMS)), xs=st.lists(special, max_size=8), rounding=roundings)
@example(name='sum_', xs=[1.0, 2.0, NAN, 3.0], rounding='nearest')  # itf1788: NaN
@example(name='sum_', xs=[1.0, -INF, 2.0, INF, 3.0], rounding='nearest')  # NaN
@example(name='sum_abs', xs=[1.0, -2.0, NAN, 3.0], rounding='nearest')  # NaN
@example(name='sum_abs', xs=[1.0, -INF, 2.0, INF, 3.0], rounding='nearest')  # infinity
@example(name='sum_sqr', xs=[1.0, 2.0, NAN, 3.0], rounding='nearest')  # NaN
@example(name='sum_sqr', xs=[1.0, -INF, 2.0, INF, 3.0], rounding='nearest')  # infinity
@example(name='sum_', xs=[-INF, MAX, MAX], rounding='up')
def test_sums_of_special_values(name, xs, rounding):
    op, term = SUMS[name]
    if any(isinstance(x, float) and math.isnan(x) for x in xs):
        expected = NAN_OPERAND
    elif any(_infinite(x) for x in xs):
        expected = _infinite_sum([term(x) for x in xs if _infinite(x)])
    else:
        assert is_rounded(op(xs, rounding=rounding), sum((term(Fraction(x)) for x in xs), Fraction(0)), rounding)
        return
    _raises_or_equals(lambda: op(xs, rounding=rounding), expected)


@settings(max_examples=300, deadline=None)
@given(pairs=st.lists(st.tuples(special, special), max_size=8), rounding=roundings)
@example(pairs=[(1.0, 1.0), (2.0, 2.0), (NAN, 3.0), (3.0, 4.0)], rounding='nearest')  # itf1788: NaN
@example(pairs=[(1.0, 1.0), (2.0, 2.0), (3.0, NAN), (4.0, 3.0)], rounding='nearest')  # NaN
@example(pairs=[(1.0, 1.0), (2.0, 2.0), (0.0, INF), (4.0, 3.0)], rounding='nearest')  # NaN
@example(pairs=[(1.0, 1.0), (2.0, 2.0), (-INF, 0.0), (4.0, 3.0)], rounding='nearest')  # NaN
@example(pairs=[(-INF, -2), (INF, 3)], rounding='down')
@example(pairs=[(-INF, -2), (INF, -3)], rounding='down')
def test_dot_of_special_values(pairs, rounding):
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    values = xs + ys
    if any(isinstance(v, float) and math.isnan(v) for v in values):
        expected = NAN_OPERAND
    elif any((_infinite(x) and y == 0) or (_infinite(y) and x == 0) for x, y in pairs):
        expected = NO_PRODUCT
    elif any(_infinite(v) for v in values):
        expected = _infinite_sum([INF if (x > 0) == (y > 0) else -INF
                                  for x, y in pairs if _infinite(x) or _infinite(y)])
    else:
        exact = sum((Fraction(x) * Fraction(y) for x, y in pairs), Fraction(0))
        assert is_rounded(dot(xs, ys, rounding=rounding), exact, rounding)
        return
    _raises_or_equals(lambda: dot(xs, ys, rounding=rounding), expected)


# THE REST OF THE CONTRACT

def test_result_is_a_float_and_never_negative_zero():
    for r in (sum_([]), sum_([-0.0]), sum_([1, -1]), dot([-0.0], [1]), sum_([Fraction(-1, 3), Fraction(1, 3)])):
        assert r == 0 and isinstance(r, float) and math.copysign(1, r) == 1
    assert sum_([2, 3]) == 5.0 and isinstance(sum_([2, 3]), float)


def test_overflow_by_direction():
    big = 10 ** 400
    assert (sum_([big]), sum_([big], rounding='down'), sum_([big], rounding='up')) == (INF, MAX, INF)
    assert (sum_([-big]), sum_([-big], rounding='down'), sum_([-big], rounding='up')) == (-INF, -INF, -MAX)
    assert sum_([big, -big, 1]) == 1.0


def test_infinity_is_exact_in_every_direction():
    for rounding in ('nearest', 'down', 'up'):
        assert sum_([INF, 1], rounding=rounding) == INF
        assert sum_([-INF, 1], rounding=rounding) == -INF
        assert dot([INF], [-2], rounding=rounding) == -INF


def test_any_iterable_and_any_real():
    assert sum_(x for x in (1, 2.5)) == 3.5
    assert dot(iter([1, 2]), (x for x in [3, 4])) == 11.0
    assert sum_abs([Fraction(-1, 2), -1]) == 1.5


@pytest.mark.parametrize('call, error, message', [
    (lambda: sum_(['1']), TypeError, 'expected a real number'),
    (lambda: sum_([True]), TypeError, 'expected a real number'),
    (lambda: sum_([1 + 2j]), TypeError, 'expected a real number'),
    (lambda: sum_([1], rounding='zero'), ValueError, 'rounding must be'),
    (lambda: sum_([1], rounding=None), ValueError, 'rounding must be'),
    (lambda: sum_([NAN]), ValueError, NAN_OPERAND),
    (lambda: sum_([INF, -INF]), ValueError, NO_SUM),
    (lambda: dot([INF], [0]), ValueError, NO_PRODUCT),
    (lambda: dot([1, 2], [1]), ValueError, 'different lengths, 2 and 1'),
])
def test_errors(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_rounding_is_keyword_only():
    with pytest.raises(TypeError):
        sum_([1], 'up')
