"""
applicator mechanics: splitting, corners, D2 limits, closure, warnings, number types, rounding hook,
and the arithmetic dunders on the class. worked-example tables and the laws live in
test_ops_examples.py and test_ops_properties.py
"""
import math
import sys
import warnings
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings

import intervals
from intervals import EMPTY
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import kernel
from intervals import ops
from intervals.applicator import OpDescriptor
from intervals.applicator import apply_binary
from intervals.applicator import apply_unary
from intervals.applicator import evaluate_box
from intervals.applicator import split_pieces
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.fmt import format_cuts
from intervals.fmt import parse
from tests.strategies import exact_cut_tuples

inf = math.inf
P = MultiInterval.parse


def run(op, *texts):
    return format_cuts(op(*map(parse, texts)))


# SPLITTING

@pytest.mark.parametrize('pieces, expected', [
    ([(-1, True, 1, False)], [(-1, True, 0, True), (0, True, 1, False)]),
    ([(-1, False, 0, False)], [(-1, False, 0, False)]),  # touching zero: no split
    ([(0, True, 2, True)], [(0, True, 2, True)]),
    ([(0, True, 0, True)], [(0, True, 0, True)]),  # a degenerate [0] stays whole
    ([(-inf, True, inf, True)], [(-inf, True, 0, True), (0, True, inf, True)]),
    ([(-2, True, -1, True), (1, False, 2, True)], [(-2, True, -1, True), (1, False, 2, True)]),
])
def test_split_at_zero(pieces, expected):
    assert split_pieces(pieces, (0,)) == expected


def test_split_at_several_points():
    assert split_pieces([(0, False, 3, False)], (1, 2)) == [
        (0, False, 1, True), (1, True, 2, True), (2, True, 3, False)]


def test_mul_splits_both_operands():
    # without the split, [-1, 1] * [inf] would hull to the entire line
    assert run(ops.mul, '[-1, 1]', '[inf]') == '{ [-inf] , [inf] }'
    assert run(ops.mul, '[inf]', '[-1, 1]') == '{ [-inf] , [inf] }'
    assert run(ops.div, '[-1, 1]', '[0, 1]') == '[-inf, inf]'


# CORNERS AND D2 LIMITS

@pytest.mark.parametrize('desc, box, expected', [
    # a defined box: min and max over the corners
    (ops.MUL, ((1, True, 2, True), (-3, True, -1, False)), (-6, True, -1, False)),
    # (inf, 0) has no value; the non-degenerate factor's edge gives the limit (D2)
    (ops.MUL, ((-inf, True, -1, True), (0, True, 0, True)), (0, True, 0, True)),
    (ops.MUL, ((-inf, True, -inf, True), (0, True, 1, True)), (-inf, True, -inf, True)),
    (ops.MUL, ((-inf, True, -1, True), (0, True, 1, True)), (-inf, True, 0, True)),
    (ops.SUB, ((1, True, inf, True), (1, True, inf, True)), (-inf, True, inf, True)),
    (ops.SUB, ((inf, True, inf, True), (1, True, inf, True)), (inf, True, inf, True)),
    (ops.DIV, ((1, True, inf, True), (1, True, inf, True)), (0, True, inf, True)),
    (ops.DIV, ((0, True, 1, True), (0, True, 1, True)), (0, True, inf, True)),
    # a pole takes its sign from the side of zero the divisor's piece lies on
    (ops.DIV, ((1, True, 2, True), (-1, True, 0, True)), (-inf, True, -1, True)),
    (ops.DIV, ((-2, True, -1, True), (-1, True, 0, True)), (1, True, inf, True)),
    (ops.RECIPROCAL, ((0, True, 1, False),), (1, False, inf, True)),
    (ops.RECIPROCAL, ((-1, True, 0, False),), (-inf, False, -1, True)),
])
def test_evaluate_box(desc, box, expected):
    assert evaluate_box(desc, box) == expected


@pytest.mark.parametrize('desc, box', [
    (ops.MUL, ((0, True, 0, True), (inf, True, inf, True))),
    (ops.SUB, ((inf, True, inf, True), (inf, True, inf, True))),
    (ops.ADD, ((-inf, True, -inf, True), (inf, True, inf, True))),
    (ops.DIV, ((0, True, 0, True), (0, True, 0, True))),
    (ops.DIV, ((inf, True, inf, True), (-inf, True, -inf, True))),
    (ops.DIV, ((1, True, 2, True), (0, True, 0, True))),  # a pole with no side, along a whole edge
    (ops.RECIPROCAL, ((0, True, 0, True),)),
])
def test_a_box_with_no_value_anywhere_is_none(desc, box):
    assert evaluate_box(desc, box) is None


@settings(max_examples=150, deadline=None)
@given(exact_cut_tuples, exact_cut_tuples)
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_monotone_fast_path_matches_every_corner(a, b):
    for desc in (ops.ADD, ops.SUB):
        assert apply_binary(desc, a, b) == apply_binary(desc._replace(monotone=None), a, b)


def test_div_is_evaluated_at_the_corners_not_through_the_reciprocal():
    calls = []

    def spy(x, y):
        calls.append((x, y))
        return ops.DIV.fn(x, y)

    assert apply_binary(ops.DIV._replace(fn=spy), parse('[1, 2]'), parse('[3, 4]')) == parse('[1/4, 2/3]')
    assert (1, 4) in calls and (2, 3) in calls


# CLOSURE

@pytest.mark.parametrize('op, a, b, expected', [
    # finite endpoint of an injective op: the corner-flag rule
    (ops.add, '[1, 2)', '(3, 4]', '(4, 6)'),
    (ops.mul, '[1, 2)', '[3, 4]', '[3, 8)'),
    (ops.div, '(1, 2]', '[3, 4)', '(1/4, 2/3]'),
    # a flat spot at 0: 0 * y is 0 along the whole edge, attained with both y ends open
    (ops.mul, '[0, 1]', '(2, 3)', '[0, 3)'),
    (ops.mul, '(0, 1]', '(2, 3)', '(0, 3)'),
    (ops.div, '[0]', '(2, 3)', '[0]'),
    (ops.div, '(1, 2)', '[inf]', '[0]'),  # x / inf is 0 for every finite x
    (ops.div, '(1, 2)', '(3, inf)', '(0, 2/3)'),
    # infinite endpoints: attained through a partner, never by the corner flags alone
    (ops.add, '[inf]', '(1, 2)', '[inf]'),
    (ops.add, '[1, inf]', '[0, 1)', '[1, inf]'),
    (ops.add, '(1, inf]', '[0]', '(1, inf]'),
    (ops.add, '(1, inf)', '[0]', '(1, inf)'),
    (ops.mul, '[inf]', '(1, 2)', '[inf]'),
    (ops.mul, '(0, 1)', '[inf]', '[inf]'),
    (ops.mul, '[0, 1)', '(0, inf)', '[0, inf)'),
    (ops.div, '(1, 2)', '[0, 1]', '(1, inf]'),  # the pole at the closed 0 attains inf
    (ops.div, '(1, 2)', '(0, 1]', '(1, inf)'),
    (ops.div, '[1, inf)', '[1, inf)', '(0, inf)'),
])
def test_closure(op, a, b, expected):
    assert run(op, a, b) == expected


@pytest.mark.parametrize('op, a, expected', [
    (ops.reciprocal, '[1, inf)', '(0, 1]'),
    (ops.reciprocal, '(-1, 0)', '(-inf, -1)'),
    (ops.reciprocal, '[-1, 0]', '[-inf, -1]'),
    (ops.absolute, '(-2, 1]', '[0, 2)'),
    (ops.absolute, '[-inf, -1)', '(1, inf]'),
    (ops.neg, '[1, inf)', '(-inf, -1]'),
    (ops.pos, '(1, 2] | [3]', '{ (1, 2] , [3] }'),
])
def test_unary_closure(op, a, expected):
    assert run(op, a) == expected


def test_attained_override_is_used():
    always_open = ops.ADD._replace(attained=lambda v, box: False)
    assert format_cuts(apply_binary(always_open, parse('[1, 2]'), parse('[3]'))) == '(4, 5)'


# WARNINGS

def test_empty_operand_warns_once():
    for op, args in [(ops.add, ((), parse('[1, 2]'))), (ops.div, (parse('[1, 2]'), ())),
                     (ops.neg, ((),)), (ops.reciprocal, ((),)), (ops.power, ((), -2))]:
        with pytest.warns(EmptySetPropagationWarning) as record:
            assert op(*args) == ()
        assert len(record) == 1


@pytest.mark.parametrize('op, args', [
    (ops.sub, ('[inf]', '[inf]')),
    (ops.mul, ('[0]', '[inf]')),
    (ops.div, ('[0]', '[0]')),
    (ops.div, ('[1, 2]', '[0]')),
    (ops.reciprocal, ('[0]',)),
])
def test_indeterminate_box_warns_and_gives_empty(op, args):
    with pytest.warns(IndeterminateResultWarning) as record:
        assert op(*map(parse, args)) == ()
    assert len(record) == 1


def test_indeterminate_warns_once_per_call_even_with_a_non_empty_result():
    # two indeterminate boxes ([0] x [inf] and [0] x [-inf]) and two defined ones
    with pytest.warns(IndeterminateResultWarning) as record:
        assert run(ops.mul, '[0] | [1]', '[-inf] | [inf]') == '{ [-inf] , [inf] }'
    assert len(record) == 1
    with pytest.warns(IndeterminateResultWarning):
        assert run(ops.reciprocal, '[0] | [1, 2]') == '[1/2, 1]'


@pytest.mark.parametrize('op, a, b', [
    (ops.mul, '[-inf, -1]', '[0]'),
    (ops.mul, '[-inf]', '[0, 1]'),
    (ops.sub, '[1, inf]', '[1, inf]'),
    (ops.sub, '[inf]', '[1, inf]'),
    (ops.div, '[1, inf]', '[1, inf]'),
    (ops.div, '[0, 1]', '[0, 1]'),
    (ops.add, '[-inf, 5]', '[inf]'),
])
def test_indeterminate_corner_of_a_larger_box_does_not_warn(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert op(parse(a), parse(b))


def test_warnings_point_at_the_caller():
    with pytest.warns(IndeterminateResultWarning) as record:
        _ = 1 / P('[0]')
    assert record[0].filename == __file__
    with pytest.warns(EmptySetPropagationWarning) as record:
        ops.add((), parse('[1]'))
    assert record[0].filename == __file__


# NUMBER TYPES (D3)

def types(cuts):
    return [type(cut.value) for cut in cuts]


@pytest.mark.parametrize('op, a, b, expected', [
    (ops.add, '[1, 2]', '[3]', [int, int]),
    (ops.sub, '[1, 2]', '[3]', [int, int]),
    (ops.mul, '[1, 2]', '[3]', [int, int]),
    (ops.div, '[6]', '[3]', [int, int]),  # an integral Fraction comes back as int
    (ops.div, '[1]', '[3]', [Fraction, Fraction]),
    (ops.div, '[1/2, 1]', '[3]', [Fraction, Fraction]),
    (ops.add, '[1/2]', '[1/2]', [int, int]),
    (ops.div, '[1]', '[inf]', [int, int]),  # not python's Fraction(1) / inf == 0.0
    (ops.div, '[1/3]', '[-inf, -1]', [Fraction, int]),
    (ops.mul, '[2]', '[1, inf]', [int, float]),  # the inf is exact; the 2 stays int
    (ops.add, '[1.5]', '[1]', [float, float]),
    (ops.div, '[1.5]', '[inf]', [float, float]),  # a finite float operand fed it
    (ops.mul, '[0]', '[2.5, inf]', [int, int]),  # the 0 from the edge limit is exact
])
def test_number_types(op, a, b, expected):
    assert types(op(parse(a), parse(b))) == expected


def test_equal_candidates_prefer_the_exact_value():
    # 0 * 2.5 is 0.0 and 0 * 3 is 0: the exact candidate wins the tie
    assert types(ops.mul(parse('[0]'), parse('[2.5, 3]'))) == [int, int]


def test_pow_types():
    assert types(ops.power(parse('[2, 3]'), 2)) == [int, int]
    assert types(ops.power(parse('[1/2]'), 3)) == [Fraction, Fraction]
    assert types(ops.power(parse('[2]'), -2)) == [Fraction, Fraction]
    assert types(ops.power(parse('[2.0]'), 2)) == [float, float]
    assert ops.power(parse('[1e200]'), 2) == parse('[inf]')  # float ** int overflow, as float * float


# ROUNDING HOOK

def test_rounding_hook_only_on_finite_float_corners():
    calls = []

    def down(x, y):
        calls.append((x, y))
        return math.nextafter(x + y, -inf)

    def up(x, y):
        calls.append((x, y))
        return math.nextafter(x + y, inf)

    outward = ops.ADD._replace(rounded=(down, up))

    # exact operands: never rounded, flags from attainment
    assert apply_binary(outward, parse('[1, 2]'), parse('[1/2]')) == parse('[3/2, 5/2]')
    assert calls == []
    # infinite corners are exact too
    assert apply_binary(outward, parse('[1, inf]'), parse('[inf]')) == parse('[inf]')
    assert calls == []

    # finite float corners go down at the bottom and up at the top; a rounded endpoint is attained
    # by nothing, so it comes out open
    result = apply_binary(outward, parse('[1.5, 2.5]'), parse('[1]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(2.5, -inf), math.nextafter(3.5, inf),
                                                    False, False)])
    assert set(calls) == {(1.5, 1), (2.5, 1)}

    # a float corner at the bottom, an infinite one at the top: only the bottom is rounded
    calls.clear()
    result = apply_binary(outward, parse('[1.5, inf]'), parse('[1]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(2.5, -inf), inf, False, True)])
    assert calls == [(1.5, 1), (1.5, 1)]


def test_rounding_hook_on_a_mixed_corner_beyond_float_range():
    # fn runs before the hook; neither may raise on an exact operand that has no float
    outward = ops.ADD._replace(rounded=(lambda x, y: math.nextafter(ops._add(x, y), -inf),
                                        lambda x, y: math.nextafter(ops._add(x, y), inf)))
    result = apply_binary(outward, MultiInterval(10 ** 400).cuts, parse('[0.5]'))
    assert result == kernel.normalize([kernel.piece(sys.float_info.max, inf, False, True)])


def test_rounding_hook_unary():
    outward = ops.NEG._replace(rounded=(lambda x: math.nextafter(-x, -inf), lambda x: math.nextafter(-x, inf)))
    result = apply_unary(outward, parse('[1, 2.0]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(-2.0, -inf), -1, False, True)])


# THE CLASS

def test_dunders_bind_the_ops():
    a, b = P('[1, 2]'), P('[3, 4)')
    assert a + b == P('[4, 6)')
    assert a - b == P('(-3, -1]')
    assert a * b == P('[3, 8)')
    assert a / b == P('(1/4, 2/3]')
    assert -a == P('[-2, -1]')
    assert +b == b
    assert abs(P('[-3, 1)')) == P('[0, 3]')
    assert a ** 2 == P('[1, 4]')
    assert a ** -1 == P('[1/2, 1]')
    assert a.reciprocal() == P('[1/2, 1]')


def test_scalars_coerce_on_both_sides():
    a = P('[1, 2]')
    assert a + 1 == 1 + a == P('[2, 3]')
    assert a - 1 == P('[0, 1]')
    assert 1 - a == P('[-1, 0]')
    assert 2 * a == a * 2 == P('[2, 4]')
    assert a / 2 == P('[1/2, 1]')
    assert 2 / a == P('[1, 2]')
    assert Fraction(1, 2) * a == P('[1/2, 1]')
    assert a * 0.5 == P('[0.5, 1.0]')


def test_numpy_scalars_run_the_reflected_dunders():
    np = pytest.importorskip('numpy')
    a = P('[1, 2]')
    assert np.float64(0.5) * a == P('[0.5, 1.0]')
    assert np.int64(3) - a == P('[1, 2]')
    assert np.float64(1) / a == P('[0.5, 1.0]')
    assert (np.float64(0.5) < a) == (0.5 < a)
    assert a ** np.float64(2) == P('[1, 4]')  # an integral value: pown (D11)
    assert a ** np.float64(0.5) == P('[1.0, 2.0]').sqrt()  # any other: 1788's pow, rounded as a float
    for thunk in (lambda: a + np.bool_(True), lambda: np.bool_(True) + a):
        with pytest.raises(TypeError):  # bool is refused, as in _coerce
            thunk()


def test_minus_is_arithmetic_not_set_difference():
    assert P('[0, 3]') - P('[1, 2]') == P('[-2, 2]')
    assert P('[0, 3]').difference(P('[1, 2]')) == P('[0, 1) | (2, 3]')


@pytest.mark.parametrize('exponent', [True, '2', None, [2]])
def test_pow_refuses_non_numbers(exponent):
    with pytest.raises(TypeError):
        _ = P('[1, 2]') ** exponent


@pytest.mark.parametrize('exponent, expected', [
    # D11: a number with an integral value is pown, over every base, as python's numbers do
    (2, '[0, 9]'), (2.0, '[0, 9]'), (Fraction(4, 2), '[0, 9]'), (-1, '{ [-inf, -1/3] , [1, inf] }'),
    # a non-integral number, or any MultiInterval, is 1788's pow: the negative bases are dropped
    (P('[2]'), '[0, 1]'), (0.5, '[0.0, 1.0]'), (Fraction(1, 2), '[0, 1]'),
    (P('[1/2, 2]'), '[0, 1]'), (math.inf, '[0]'),
])
def test_pow_dispatch(exponent, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        warnings.simplefilter('ignore', IndeterminateResultWarning)
        assert str(P('[-3, 1]') ** exponent) == expected


def test_pow_zero_and_three_argument_pow():
    assert P('[-inf, 0)') ** 0 == P('[1]')
    with pytest.raises(TypeError):
        pow(P('[1, 2]'), 2, 3)  # dropped (D11): not 1788, and v1 had it on integers only
    with pytest.raises(TypeError):
        pow(P('[1, 2]'), P('[2]'), 3)
    with pytest.raises(TypeError):
        ops.power(parse('[1]'), True)


@pytest.mark.parametrize('other', ['1', None, [1], (1, 2)])
def test_non_numbers_are_refused(other):
    a = P('[1, 2]')
    for op in (lambda: a + other, lambda: other + a, lambda: a - other, lambda: other - a,
               lambda: a * other, lambda: other * a, lambda: a / other, lambda: other / a):
        with pytest.raises(TypeError):
            op()


def test_rpow_is_pow_of_a_point():
    # M13d (D11): `b ** A` for a real b is `MultiInterval(b) ** A`, so 1788's pow
    assert 2 ** P('[1, 3]') == P('[2, 8]')
    assert Fraction(1, 4) ** P('[1/2]') == P('[1/2]')
    assert 4.0 ** P('[1/2]') == P('[2.0]')
    assert type(2 ** OutwardMultiInterval(1.0, 3.0)) is OutwardMultiInterval
    # a subclass's reflected method runs first, so mixing keeps the outward class on either side
    assert type(P('[2]') ** OutwardMultiInterval(0.5)) is OutwardMultiInterval
    assert type(OutwardMultiInterval(2.0) ** P('[1/2]')) is OutwardMultiInterval
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        assert (-2) ** P('[1, 2]') == EMPTY  # a negative base is outside pow's domain
    for base in (True, '2', None):
        with pytest.raises(TypeError):
            _ = base ** P('[1, 2]')


def test_package_exports_unchanged():
    # the ops, the applicator, modulo, steps and functions stay in their submodules; M7 adds only
    # HullWarning, M12 only OutwardMultiInterval, M13h only the reductions (point functions, exported
    # as M13e plans for the reverse ops)
    assert set(intervals.__all__) == {
        'MultiInterval', 'OutwardMultiInterval', 'EMPTY', 'REALS', 'Size', 'Builder', 'TruthSet', 'Allen',
        'Cut', 'Side', 'IntervalWarning', 'EmptySetPropagationWarning', 'DomainClippedWarning',
        'IndeterminateResultWarning', 'HullWarning', 'sum_', 'sum_abs', 'sum_sqr', 'dot',
        # M13e: the reverse ops (intervals/reverse.py)
        'sqr_rev', 'abs_rev', 'pown_rev', 'cosh_rev',
        'mul_rev',
        'sin_rev', 'cos_rev', 'tan_rev',
        'pow_rev1', 'pow_rev2',
        # M13g: ieee 1788's signals and bare constructors
        'UndefinedOperationError', 'PossiblyUndefinedOperationWarning', 'text_to_interval', 'nums_to_interval',
        # M13g: ieee 1788's decorated type and its constructors
        'DecoratedInterval', 'Decoration', 'set_dec', 'text_to_decorated_interval', 'nums_to_decorated_interval',
        # M15: autodiff and interval newton (intervals/autodiff.py, intervals/solver.py)
        'Dual', 'derivative', 'newton', 'Root',
        # M16a: several variables (intervals/autodiff.py, intervals/solver.py)
        'gradient', 'jacobian', 'solve', 'RootBox'}
    for name in ('add', 'mul', 'OpDescriptor', 'apply_binary', 'mod', 'floordiv', 'floor', 'sqrt', 'sign',
                 'fma', 'apply', 'step'):
        assert not hasattr(intervals, name), name
