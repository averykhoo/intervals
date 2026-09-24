"""
worked examples for M6 arithmetic: exact expected results, derived by hand from the pointwise table
(v2-plan.md "domain and semantics" / "arithmetic", decisions D1, D2, D3, D7) and not from the code.

every operand and expected value is written in `fmt`'s grammar and compared with structural `==`.
a case with no expected warning runs with the library's warnings as errors, so a spurious warning
fails it; a case with a warning asserts exactly one warning, of exactly that class.
"""
import math
import operator
import warnings
from fractions import Fraction

import pytest

from intervals import MultiInterval
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import IntervalWarning

P = MultiInterval.parse
BINARY = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
UNARY = {
    'neg': operator.neg,
    'pos': operator.pos,
    'abs': abs,
    '1/': lambda a: 1 / a,
    'reciprocal': lambda a: a.reciprocal(),
}


def quiet(thunk):
    """run `thunk` with every library warning turned into an error"""
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntervalWarning)
        return thunk()


def warns_once(category, thunk):
    """run `thunk`, require exactly one library warning, of exactly `category`"""
    with pytest.warns(category) as record:
        result = thunk()
    got = [w.category.__name__ for w in record if issubclass(w.category, IntervalWarning)]
    assert got == [category.__name__]
    return result


def binary_cases(rows):
    return [pytest.param(a, op, b, expected, id=f'{a} {op} {b} = {expected}') for a, op, b, expected in rows]


# BINARY OPS, NO WARNING

ADD = [
    ('[1, 2]', '+', '[3, 4]', '[4, 6]'),
    ('(1, 2)', '+', '(3, 4)', '(4, 6)'),
    ('[1, 2)', '+', '(3, 4]', '(4, 6)'),
    ('[1, 2)', '+', '[3, 4]', '[4, 6)'),
    ('[-1, 1]', '+', '[-1, 1]', '[-2, 2]'),
    ('[1, 2]', '+', '[1/2]', '[3/2, 5/2]'),
    # multi-piece operands
    ('{[1, 2], [4, 5]}', '+', '[0, 1]', '{[1, 3], [4, 6]}'),
    ('{[1, 2], [4, 5]}', '+', '[0, 2]', '[1, 7]'),
    ('{[1, 2), (2, 3]}', '+', '[0]', '{[1, 2), (2, 3]}'),
    ('{[1, 2), (2, 3]}', '+', '[0, 1]', '[1, 4]'),
    ('{1, 3}', '+', '{10, 20}', '{11, 13, 21, 23}'),
    # infinity closure: add is flat at +-inf, so an infinite endpoint is closed iff attained
    ('[inf]', '+', '(1, 2)', '[inf]'),
    ('[1, inf]', '+', '[0, 1)', '[1, inf]'),
    ('(1, inf]', '+', '[0]', '(1, inf]'),
    ('(1, inf)', '+', '[0, 1]', '(1, inf)'),
    ('[-inf]', '+', '(1, 2)', '[-inf]'),
    ('[-inf, 0]', '+', '[1]', '[-inf, 1]'),
    ('(-inf, 0]', '+', '[1]', '(-inf, 1]'),
    ('[inf]', '+', '[inf]', '[inf]'),
    # D2: the indeterminate corner (inf, -inf) of a non-degenerate box contributes its limits
    ('[inf]', '+', '[-inf, 0]', '[inf]'),
    ('[-inf, 0]', '+', '[inf]', '[inf]'),
    ('[-inf, inf]', '+', '[inf]', '[inf]'),
    ('(-inf, inf)', '+', '[inf]', '[inf]'),
    ('(1, inf)', '+', '[-inf]', '[-inf]'),
    ('[1, inf]', '+', '[-inf, -1]', '[-inf, inf]'),
    ('[-inf, inf]', '+', '[-inf, inf]', '[-inf, inf]'),
]

SUB = [
    ('[1, 2]', '-', '[3, 4]', '[-3, -1]'),
    ('[1, 2)', '-', '(3, 4]', '[-3, -1)'),
    ('[1, 2]', '-', '[1, 2]', '[-1, 1]'),  # `-` is arithmetic, not set difference
    ('{[1, 2], [4, 5]}', '-', '[0, 1]', '{[0, 2], [3, 5]}'),
    ('[3]', '-', '[1/2]', '[5/2]'),
    # D2
    ('[1, inf]', '-', '[1, inf]', '[-inf, inf]'),
    ('[inf]', '-', '[1, inf]', '[inf]'),
    ('[1, inf]', '-', '[inf]', '[-inf]'),
    # infinity closure
    ('[inf]', '-', '(1, 2)', '[inf]'),
    ('(1, 2)', '-', '[inf]', '[-inf]'),
    ('[1, inf]', '-', '[0, 1)', '(0, inf]'),
    ('[0]', '-', '(1, inf]', '[-inf, -1)'),
    ('[inf]', '-', '[-inf]', '[inf]'),
    ('[-inf]', '-', '[inf]', '[-inf]'),
]

MUL = [
    ('[2, 3]', '*', '[4, 5]', '[8, 15]'),
    ('(1, 2]', '*', '[3, 4)', '(3, 8)'),
    ('[-3, -2]', '*', '[-5, -4]', '[8, 15]'),
    ('[-2, 3]', '*', '[-1, 4]', '[-8, 12]'),
    ('(-2, 3]', '*', '[-1, 4)', '(-8, 12)'),
    # the flat spot at zero: v1's `(0, 3)` excluded the attained 0 (0 * 2.5)
    ('[0, 1]', '*', '(2, 3)', '[0, 3)'),
    ('(0, 1]', '*', '(2, 3)', '(0, 3)'),
    ('[-1, 0)', '*', '(2, 3)', '(-3, 0)'),
    ('(-1, 1)', '*', '(2, 3)', '(-3, 3)'),
    ('[-1, 1]', '*', '(0, 1)', '(-1, 1)'),
    ('[0]', '*', '(2, 3)', '[0]'),
    # multi-piece operands
    ('{[-2, -1], [1, 2]}', '*', '[0, 1]', '[-2, 2]'),
    ('{[-2, -1], [1, 2]}', '*', '(0, 1]', '{[-2, 0), (0, 2]}'),
    ('{1, 3}', '*', '{-1, 2}', '{-3, -1, 2, 6}'),
    # D2
    ('[-inf, -1]', '*', '[0]', '[0]'),
    ('[-inf]', '*', '[0, 1]', '[-inf]'),
    ('[-inf, -1]', '*', '[0, 1]', '[-inf, 0]'),
    ('[0, 1]', '*', '[inf]', '[inf]'),
    ('[inf]', '*', '[-1, 0)', '[-inf]'),
    ('[0]', '*', '[1, inf]', '[0]'),
    ('[0]', '*', '[-inf, inf]', '[0]'),
    ('[-inf, inf]', '*', '[0]', '[0]'),
    ('[0, inf]', '*', '[0, inf]', '[0, inf]'),
    ('[-inf, 0]', '*', '[0, inf]', '[-inf, 0]'),
    ('[-1, 1]', '*', '[-inf, inf]', '[-inf, inf]'),
    # infinity closure
    ('[inf]', '*', '(1, 2)', '[inf]'),
    ('[2, 3]', '*', '[1, inf]', '[2, inf]'),
    ('(2, 3)', '*', '(1, inf]', '(2, inf]'),
    ('(2, 3)', '*', '(1, inf)', '(2, inf)'),
    ('(0, inf)', '*', '(0, inf)', '(0, inf)'),
    ('[-inf]', '*', '[-inf]', '[inf]'),
    ('[-inf]', '*', '[inf]', '[-inf]'),
    ('[-inf]', '*', '[-1]', '[inf]'),
    # interior sharpness: mul splits at zero, so 0 * inf never fills the gap
    ('[-1, 1]', '*', '[inf]', '{[-inf], [inf]}'),
    ('(-1, 1)', '*', '[inf]', '{[-inf], [inf]}'),
]

DIV = [
    ('[1]', '/', '[3]', '[1/3]'),
    ('[6]', '/', '[3]', '[2]'),
    ('[1, 2]', '/', '[4, 8]', '[1/8, 1/2]'),
    ('[-1, 1]', '/', '[2]', '[-1/2, 1/2]'),
    ('{[1, 2], [4, 5]}', '/', '[2]', '{[1/2, 1], [2, 5/2]}'),
    ('[1, 2]', '/', '{[-2, -1], [1, 2]}', '{[-2, -1/2], [1/2, 2]}'),
    # a denominator touching zero: the pole's direction comes from the piece containing 0
    ('[1]', '/', '[-1, 0]', '[-inf, -1]'),
    ('[1]', '/', '[0, 1]', '[1, inf]'),
    ('[1]', '/', '(0, 1]', '[1, inf)'),
    ('[1, 2]', '/', '[0, 1]', '[1, inf]'),
    ('[1, 2]', '/', '[0, 1)', '(1, inf]'),
    ('[1, 2]', '/', '(0, 1)', '(1, inf)'),
    ('[-2, -1]', '/', '[0, 1]', '[-inf, -1]'),
    ('[1, 2]', '/', '[-1, 1]', '{[-inf, -1], [1, inf]}'),
    ('[1, 2]', '/', '(-1, 1)', '{[-inf, -1), (1, inf]}'),
    ('[-1, 1]', '/', '[-1, 1]', '[-inf, inf]'),
    ('[0, 1]', '/', '[0, 1]', '[0, inf]'),  # 0/0 is dropped, but its box is not a point
    ('[0]', '/', '[0, 1]', '[0]'),
    ('[0]', '/', '[1, 2]', '[0]'),
    # infinite operands
    ('[0]', '/', '[inf]', '[0]'),
    ('[1, 2]', '/', '[inf]', '[0]'),
    ('[-1, 1]', '/', '[inf]', '[0]'),
    ('[-inf, inf]', '/', '[inf]', '[0]'),
    ('[1, 2]', '/', '[1, inf]', '[0, 2]'),
    ('[1, 2]', '/', '[1, inf)', '(0, 2]'),
    ('[1, inf]', '/', '[1, inf]', '[0, inf]'),  # D2
    ('[inf]', '/', '[2]', '[inf]'),
    ('[inf]', '/', '[-2]', '[-inf]'),
    ('[inf]', '/', '[0, 1]', '[inf]'),
    ('[inf]', '/', '[-1, 1]', '{[-inf], [inf]}'),
    ('[inf]', '/', '[1, inf]', '[inf]'),
    ('[inf]', '/', '[-inf, -1]', '[-inf]'),
]


@pytest.mark.parametrize('a, op, b, expected', binary_cases(ADD + SUB + MUL + DIV))
def test_binary(a, op, b, expected):
    result = quiet(lambda: BINARY[op](P(a), P(b)))
    assert isinstance(result, MultiInterval)
    assert result == P(expected), str(result)


# UNARY OPS, NO WARNING

NEG = [
    ('[1, 2)', '(-2, -1]'),
    ('(-1, 2]', '[-2, 1)'),
    ('[0]', '[0]'),
    ('[-inf, 0)', '(0, inf]'),
    ('(-inf, inf)', '(-inf, inf)'),
    ('[inf]', '[-inf]'),
    ('{[-inf], [1, 2)}', '{(-2, -1], [inf]}'),
]

POS = [
    ('[1, 2)', '[1, 2)'),
    ('{[-inf], (0, 1]}', '{[-inf], (0, 1]}'),
    ('[-1/3]', '[-1/3]'),
]

ABS = [
    ('[-2, 1)', '[0, 2]'),
    ('(-2, 1)', '[0, 2)'),
    ('(-1, 2)', '[0, 2)'),
    ('(-1, 1)', '[0, 1)'),
    ('(-3, 0)', '(0, 3)'),
    ('[-3, -1)', '(1, 3]'),
    ('(1, 3]', '(1, 3]'),
    ('[0]', '[0]'),
    ('(-inf, 2]', '[0, inf)'),
    ('[-inf, 2]', '[0, inf]'),
    ('[-inf, inf]', '[0, inf]'),
    ('[-inf]', '[inf]'),
    ('{[-inf], [inf]}', '[inf]'),
    ('{(-3, -2), [2]}', '[2, 3)'),
    ('{[-2, -1], [1, 2]}', '[1, 2]'),
    ('{(-2, -1), [1]}', '[1, 2)'),
    ('{[-5], (0, 1)}', '{(0, 1), [5]}'),
]

RECIPROCAL = [
    # v2-plan.md "direction comes from the set" and D1
    ('[-1, 0]', '[-inf, -1]'),
    ('[-1, 1]', '{[-inf, -1], [1, inf]}'),
    ('[1, inf]', '[0, 1]'),
    ('[1, inf)', '(0, 1]'),
    ('(-1, 0)', '(-inf, -1)'),
    ('[-1, 0)', '(-inf, -1]'),  # the D7 union counterexample's A
    ('(-1, 0]', '[-inf, -1)'),
    ('[0, 1]', '[1, inf]'),
    ('(0, 1]', '[1, inf)'),
    ('(0, 1)', '(1, inf)'),
    ('[-1, 2]', '{[-inf, -1], [1/2, inf]}'),
    ('[2]', '[1/2]'),
    ('[-2]', '[-1/2]'),
    ('[-2, -1]', '[-1, -1/2]'),
    ('[2, 4]', '[1/4, 1/2]'),
    ('{[-2, -1], [1, 2]}', '{[-1, -1/2], [1/2, 1]}'),
    ('[inf]', '[0]'),
    ('[-inf]', '[0]'),
    ('[-inf, -1]', '[-1, 0]'),
    ('[0, inf]', '[0, inf]'),
    ('(0, inf)', '(0, inf)'),
    ('[-inf, 0)', '(-inf, 0]'),
    ('(-inf, 0]', '[-inf, 0)'),
    ('[-inf, inf]', '[-inf, inf]'),
    ('(-inf, inf)', '{[-inf, 0), (0, inf]}'),
    ('{[1, 2], [inf]}', '{[0], [1/2, 1]}'),
]


def unary_cases():
    rows = [('neg', a, e) for a, e in NEG] + [('pos', a, e) for a, e in POS] + [('abs', a, e) for a, e in ABS]
    rows += [(spelling, a, e) for a, e in RECIPROCAL for spelling in ('1/', 'reciprocal')]
    return [pytest.param(name, a, expected, id=f'{name} {a} = {expected}') for name, a, expected in rows]


@pytest.mark.parametrize('name, a, expected', unary_cases())
def test_unary(name, a, expected):
    result = quiet(lambda: UNARY[name](P(a)))
    assert isinstance(result, MultiInterval)
    assert result == P(expected), str(result)


# POWER, NO WARNING

POWER = [
    # n == 0: [1] for any non-empty operand
    ('[-2, 1)', 0, '[1]'),
    ('(-inf, inf)', 0, '[1]'),
    ('[inf]', 0, '[1]'),
    ('[0]', 0, '[1]'),
    # n == 1: identity
    ('[-2, 1)', 1, '[-2, 1)'),
    ('(0, 2)', 1, '(0, 2)'),
    ('[-inf]', 1, '[-inf]'),
    ('[-inf, inf]', 1, '[-inf, inf]'),
    # n == 2: even, flat at 0, (+-inf)**2 = inf
    ('[-2, 1)', 2, '[0, 4]'),
    ('(-2, 1)', 2, '[0, 4)'),
    ('(-1, 2]', 2, '[0, 4]'),
    ('(-1, 2)', 2, '[0, 4)'),
    ('(0, 2)', 2, '(0, 4)'),
    ('[-3, -2)', 2, '(4, 9]'),
    ('{[-2, -1], [1, 2]}', 2, '[1, 4]'),
    ('[2]', 2, '[4]'),
    ('[1/2]', 2, '[1/4]'),
    ('[-inf, -1]', 2, '[1, inf]'),
    ('(-inf, -1]', 2, '[1, inf)'),
    ('[-inf, inf]', 2, '[0, inf]'),
    ('(-inf, inf)', 2, '[0, inf)'),
    ('[-inf]', 2, '[inf]'),
    ('[inf]', 2, '[inf]'),
    # n == 3: odd, monotone, (-inf)**3 = -inf
    ('[-2, 1)', 3, '[-8, 1)'),
    ('(-1, 2]', 3, '(-1, 8]'),
    ('(0, 2)', 3, '(0, 8)'),
    ('[-3, -2)', 3, '[-27, -8)'),
    ('[-inf, -1]', 3, '[-inf, -1]'),
    ('(-inf, 2]', 3, '(-inf, 8]'),
    ('[-inf, 2]', 3, '[-inf, 8]'),
    ('[-inf, inf]', 3, '[-inf, inf]'),
    ('[-inf]', 3, '[-inf]'),
    # n == -1: reciprocal(A)
    ('[-2, 1)', -1, '{[-inf, -1/2], (1, inf]}'),
    ('[-1, 2]', -1, '{[-inf, -1], [1/2, inf]}'),
    ('[-1, 0]', -1, '[-inf, -1]'),
    ('[0, 1]', -1, '[1, inf]'),
    ('(0, 2)', -1, '(1/2, inf)'),
    ('(1, 2]', -1, '[1/2, 1)'),
    ('[2]', -1, '[1/2]'),
    ('[inf]', -1, '[0]'),
    ('[-inf, -1]', -1, '[-1, 0]'),
    ('[-inf, inf]', -1, '[-inf, inf]'),
    ('(-inf, inf)', -1, '{[-inf, 0), (0, inf]}'),
    # n == -2: reciprocal(A**2); the pole's direction comes from A**2, which is >= 0
    ('[-2, 1)', -2, '[1/4, inf]'),
    ('(-2, 1)', -2, '(1/4, inf]'),
    ('[-1, 1]', -2, '[1, inf]'),
    ('[-1, 2]', -2, '[1/4, inf]'),
    ('(0, 2)', -2, '(1/4, inf)'),
    ('(1, 2]', -2, '[1/4, 1)'),
    ('[2]', -2, '[1/4]'),
    ('[-inf, -1]', -2, '[0, 1]'),
    ('(-inf, -1]', -2, '(0, 1]'),
    ('[-inf]', -2, '[0]'),
    ('[-inf, inf]', -2, '[0, inf]'),
    ('(-inf, inf)', -2, '(0, inf]'),
]


@pytest.mark.parametrize('a, n, expected', [pytest.param(a, n, e, id=f'{a} ** {n} = {e}') for a, n, e in POWER])
def test_power(a, n, expected):
    result = quiet(lambda: P(a) ** n)
    assert isinstance(result, MultiInterval)
    assert result == P(expected), str(result)
    assert quiet(lambda: pow(P(a), n)) == result


# 1/(1/A) == A for every A with no degenerate piece at 0, inf or -inf

INVOLUTION = [
    '[-1, 0]', '(-1, 0)', '[1, inf]', '[1, inf)', '[-1, 1]', '(-1, 2]', '[2, 4]', '(0, inf)', '[-inf, 0)',
    '[-inf, inf]', '(-inf, inf)', '{[-2, -1], (1, 2]}', '{[1, 2], [4, 5]}',
]


@pytest.mark.parametrize('a', INVOLUTION)
def test_reciprocal_involution(a):
    assert quiet(lambda: 1 / (1 / P(a))) == P(a)


# NUMBER TYPES (D3)

TYPES = [
    # int op int stays int for + - *; int or Fraction division is exact
    ('[1]', '+', '[2]', 'inf', 3, int),
    ('[7]', '-', '[2]', 'inf', 5, int),
    ('[3]', '*', '[4]', 'inf', 12, int),
    ('[1]', '/', '[3]', 'inf', Fraction(1, 3), Fraction),
    ('[1]', '/', '[2]', 'inf', Fraction(1, 2), Fraction),
    ('[6]', '/', '[3]', 'inf', 2, int),  # integral Fractions normalize to int
    ('[1/2]', '+', '[1/2]', 'inf', 1, int),
    ('[1/3]', '*', '[3]', 'inf', 1, int),
    ('[3]', '-', '[1/2]', 'inf', Fraction(5, 2), Fraction),
    ('[1, 2]', '/', '[4, 8]', 'inf', Fraction(1, 8), Fraction),
    # python's Fraction(1) / inf is 0.0: the applicator must give an exact 0
    ('[1]', '/', '[inf]', 'inf', 0, int),
    ('[1/3]', '/', '[-inf]', 'inf', 0, int),
    ('[1, 2]', '/', '[inf]', 'sup', 0, int),
    ('[1, inf]', '/', '[1, inf]', 'inf', 0, int),
    ('[-inf, -1]', '*', '[0]', 'inf', 0, int),
    # a result corner is float iff a finite float operand fed it; inf never makes it float
    ('[2.0]', '/', '[inf]', 'inf', 0.0, float),
    ('[1]', '/', '[2.0]', 'inf', 0.5, float),
    ('[1.5]', '+', '[1]', 'inf', 2.5, float),
    ('[1, 2.5]', '+', '[1]', 'inf', 2, int),
    ('[1, 2.5]', '+', '[1]', 'sup', 3.5, float),
]


@pytest.mark.parametrize('a, op, b, attr, value, kind', [
    pytest.param(*row, id=f'{row[0]} {row[1]} {row[2]} .{row[3]} is {row[5].__name__}') for row in TYPES])
def test_result_types(a, op, b, attr, value, kind):
    result = quiet(lambda: BINARY[op](P(a), P(b)))
    assert getattr(result, attr) == value
    assert type(getattr(result, attr)) is kind


@pytest.mark.parametrize('thunk, value, kind', [
    pytest.param(lambda: 1 / P('[inf]'), 0, int, id='1/[inf] is exact 0'),
    pytest.param(lambda: P('[-inf]').reciprocal(), 0, int, id='reciprocal [-inf] is exact 0'),
    pytest.param(lambda: 1 / P('[4]'), Fraction(1, 4), Fraction, id='1/[4]'),
    pytest.param(lambda: P('[2]') ** -1, Fraction(1, 2), Fraction, id='[2]**-1'),
    pytest.param(lambda: P('[2]') ** 3, 8, int, id='[2]**3'),
    pytest.param(lambda: P('[1/2]') ** 2, Fraction(1, 4), Fraction, id='[1/2]**2'),
    pytest.param(lambda: P('[-inf]') ** -2, 0, int, id='[-inf]**-2 is exact 0'),
    pytest.param(lambda: abs(P('[-3]')), 3, int, id='abs [-3]'),
    pytest.param(lambda: -P('[1/2]'), Fraction(-1, 2), Fraction, id='-[1/2]'),
    pytest.param(lambda: 1 / P('[2.0]'), 0.5, float, id='1/[2.0]'),
])
def test_unary_result_types(thunk, value, kind):
    result = quiet(thunk)
    assert result.inf == value
    assert type(result.inf) is kind


def test_float_endpoints_use_plain_float_arithmetic():
    # the rounding hook is the identity by default
    assert quiet(lambda: P('[0.1]') + P('[0.2]')) == MultiInterval(0.1 + 0.2)
    assert quiet(lambda: P('[0.1]') * P('[3]')) == MultiInterval(0.1 * 3)
    assert quiet(lambda: P('[1]') / P('[3.0]')) == MultiInterval(1 / 3.0)


# SCALAR COERCION, BOTH SIDES

@pytest.mark.parametrize('thunk, expected', [
    pytest.param(lambda: P('[0, 1]') + 1, '[1, 2]', id='MI + 1'),
    pytest.param(lambda: 1 + P('[0, 1]'), '[1, 2]', id='1 + MI'),
    pytest.param(lambda: P('[0, 1)') - 2, '[-2, -1)', id='MI - 2'),
    pytest.param(lambda: 2 - P('[0, 1)'), '(1, 2]', id='2 - MI'),
    pytest.param(lambda: 3 * P('(1, 2]'), '(3, 6]', id='3 * MI'),
    pytest.param(lambda: P('(1, 2]') * -1, '[-2, -1)', id='MI * -1'),
    pytest.param(lambda: P('[1, 3]') / 2, '[1/2, 3/2]', id='MI / 2'),
    pytest.param(lambda: 1 / P('[2, 4]'), '[1/4, 1/2]', id='1 / MI'),
    pytest.param(lambda: 1 / P('[-1, 1]'), '{[-inf, -1], [1, inf]}', id='1 / MI across zero'),
    pytest.param(lambda: 1 / P('(0, inf)'), '(0, inf)', id='1 / (0, inf)'),
    pytest.param(lambda: Fraction(1, 3) * P('[3, 6]'), '[1, 2]', id='Fraction * MI'),
    pytest.param(lambda: 0.5 * P('[2, 4]'), '[1, 2]', id='float * MI'),
    pytest.param(lambda: math.inf + P('[1, 2]'), '[inf]', id='inf + MI'),
    pytest.param(lambda: P('[1, inf]') - math.inf, '[-inf]', id='MI - inf'),
    pytest.param(lambda: 2 - P('[1, inf]'), '[-inf, 1]', id='2 - [1, inf]'),
    pytest.param(lambda: -math.inf * P('[0, 1]'), '[-inf]', id='-inf * [0, 1]'),
    pytest.param(lambda: 0 * P('[1, inf]'), '[0]', id='0 * [1, inf]'),
])
def test_scalar_coercion(thunk, expected):
    result = quiet(thunk)
    assert isinstance(result, MultiInterval)
    assert result == P(expected), str(result)


def test_reciprocal_method_matches_one_over():
    for text in ('[-1, 1]', '(0, 2]', '{[-3, -1), [1/2]}', '[1, inf)'):
        assert quiet(lambda: P(text).reciprocal()) == quiet(lambda: 1 / P(text))


# INDETERMINATE RESULTS (D7): the box is an indeterminate point -> that box gives nothing, one warning

INDETERMINATE = [
    ('[inf]', '-', '[inf]', '{}'),
    ('[-inf]', '-', '[-inf]', '{}'),
    ('[inf]', '+', '[-inf]', '{}'),
    ('[-inf]', '+', '[inf]', '{}'),
    ('[0]', '*', '[inf]', '{}'),
    ('[inf]', '*', '[0]', '{}'),
    ('[0]', '*', '[-inf]', '{}'),
    ('[0]', '/', '[0]', '{}'),
    ('[1]', '/', '[0]', '{}'),
    ('[-3]', '/', '[0]', '{}'),
    ('[inf]', '/', '[0]', '{}'),
    ('[inf]', '/', '[inf]', '{}'),
    ('[-inf]', '/', '[inf]', '{}'),
    ('[inf]', '/', '[-inf]', '{}'),
    # every pair is 0/0 or a pole without a direction (the zero piece of B is degenerate)
    ('[-1, 1]', '/', '[0]', '{}'),
    ('[-inf, inf]', '/', '[0]', '{}'),
    # one piece pair is an indeterminate point, the others are not: still exactly one warning
    ('{[0], [1, 2]}', '*', '[inf]', '[inf]'),
    ('[inf]', '*', '{[0], [1, 2]}', '[inf]'),
    ('{[1], [inf]}', '-', '[inf]', '[-inf]'),
    ('[1, 2]', '/', '{[0], [4]}', '[1/4, 1/2]'),
    ('{[0], [1, 2]}', '/', '{[0], [1]}', '{[0], [1, 2]}'),
]


@pytest.mark.parametrize('a, op, b, expected', binary_cases(INDETERMINATE))
def test_indeterminate_binary_warns(a, op, b, expected):
    result = warns_once(IndeterminateResultWarning, lambda: BINARY[op](P(a), P(b)))
    assert result == P(expected), str(result)


@pytest.mark.parametrize('thunk, expected', [
    pytest.param(lambda: 1 / P('[0]'), '{}', id='1/[0]'),
    pytest.param(lambda: P('[0]').reciprocal(), '{}', id='reciprocal [0]'),
    pytest.param(lambda: 1 / (1 / P('[inf]')), '{}', id='1/(1/[inf])'),
    pytest.param(lambda: 1 / P('{[0], [1, 2]}'), '[1/2, 1]', id='1/{[0], [1, 2]}'),
    pytest.param(lambda: 1 / P('{[0], [1/2, 1]}'), '[1, 2]', id='1/{[0], [1/2, 1]}'),
    pytest.param(lambda: 1 / P('{[0], [inf]}'), '[0]', id='1/{[0], [inf]}'),
    pytest.param(lambda: P('[0]') ** -1, '{}', id='[0]**-1'),
    pytest.param(lambda: P('[0]') ** -2, '{}', id='[0]**-2'),
    pytest.param(lambda: P('{[0], [2]}') ** -1, '[1/2]', id='{[0], [2]}**-1'),
    pytest.param(lambda: P('{[-2], [0]}') ** -2, '[1/4]', id='{[-2], [0]}**-2'),
    pytest.param(lambda: 1 / (1 / P('{[1, 2], [inf]}')), '[1, 2]', id='1/(1/{[1, 2], [inf]})'),
    # scalar operands coerce to degenerate pieces
    pytest.param(lambda: 0 * P('[inf]'), '{}', id='0 * [inf]'),
    pytest.param(lambda: P('[0]') * math.inf, '{}', id='[0] * inf'),
    pytest.param(lambda: math.inf - P('[inf]'), '{}', id='inf - [inf]'),
    pytest.param(lambda: P('[1, 2]') / 0, '{}', id='[1, 2] / 0'),
])
def test_indeterminate_other_warns(thunk, expected):
    result = warns_once(IndeterminateResultWarning, thunk)
    assert result == P(expected), str(result)


# EMPTY OPERANDS: the image of an empty set, one warning per call

EMPTY_CASES = [
    ('∅ + A', lambda e, a: e + a),
    ('A + ∅', lambda e, a: a + e),
    ('∅ + ∅', lambda e, a: e + e),
    ('∅ - A', lambda e, a: e - a),
    ('A - ∅', lambda e, a: a - e),
    ('∅ * A', lambda e, a: e * a),
    ('A * ∅', lambda e, a: a * e),
    ('∅ / A', lambda e, a: e / a),
    ('A / ∅', lambda e, a: a / e),
    ('∅ / ∅', lambda e, a: e / e),
    ('∅ + 1', lambda e, a: e + 1),
    ('1 - ∅', lambda e, a: 1 - e),
    ('2 * ∅', lambda e, a: 2 * e),
    ('1 / ∅', lambda e, a: 1 / e),
    ('∅ / 2', lambda e, a: e / 2),
    ('[0] / ∅', lambda e, a: P('[0]') / e),  # empty wins: no indeterminate warning
    ('∅ * [inf]', lambda e, a: e * P('[inf]')),
    ('-∅', lambda e, a: -e),
    ('+∅', lambda e, a: +e),
    ('abs ∅', lambda e, a: abs(e)),
    ('∅.reciprocal()', lambda e, a: e.reciprocal()),
] + [(f'∅ ** {n}', (lambda n: lambda e, a: e ** n)(n)) for n in (0, 1, 2, 3, -1, -2)]


@pytest.mark.parametrize('thunk', [pytest.param(thunk, id=name) for name, thunk in EMPTY_CASES])
def test_empty_operand_propagates_with_one_warning(thunk):
    result = warns_once(EmptySetPropagationWarning, lambda: thunk(MultiInterval(), P('[1, 2]')))
    assert isinstance(result, MultiInterval)
    assert result == MultiInterval()


# NOT SUPPORTED (yet or at all)

@pytest.mark.parametrize('thunk', [
    pytest.param(lambda a: a + 'a', id='MI + str'),
    pytest.param(lambda a: 'a' + a, id='str + MI'),
    pytest.param(lambda a: a - [1], id='MI - list'),
    pytest.param(lambda a: a * None, id='MI * None'),
    pytest.param(lambda a: a / 'x', id='MI / str'),
    pytest.param(lambda a: a + True, id='MI + bool'),  # _coerce rejects bool
    pytest.param(lambda a: a ** 0.5, id='MI ** 0.5'),
    pytest.param(lambda a: a ** 2.0, id='MI ** 2.0'),
    pytest.param(lambda a: a ** Fraction(1, 2), id='MI ** Fraction(1, 2)'),
    pytest.param(lambda a: a ** a, id='MI ** MI'),
    pytest.param(lambda a: a ** True, id='MI ** True'),
    pytest.param(lambda a: 2 ** a, id='2 ** MI'),
    pytest.param(lambda a: a // 2, id='MI // 2'),
    pytest.param(lambda a: 2 // a, id='2 // MI'),
    pytest.param(lambda a: a % 2, id='MI % 2'),
    pytest.param(lambda a: 2 % a, id='2 % MI'),
    pytest.param(lambda a: divmod(a, 2), id='divmod(MI, 2)'),
])
def test_type_errors(thunk):
    with pytest.raises(TypeError):
        quiet(lambda: thunk(P('[1, 2]')))


def test_integral_fraction_exponent_is_an_int_exponent():
    """
    `__pow__` returns NotImplemented for a Fraction, and python then calls `Fraction.__rpow__`, which
    turns an integral Fraction into its int (`a ** Fraction(2)` is `a ** 2`). that is the pointwise
    answer and D3's "an integral Fraction is an int", so it is pinned rather than refused
    """
    assert P('[1, 2]') ** Fraction(2) == P('[1, 4]')
    assert P('[1, 2]') ** Fraction(-1) == P('[1/2, 1]')
