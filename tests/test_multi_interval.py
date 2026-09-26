import copy
import math
import pickle
from fractions import Fraction

import pytest
from hypothesis import given

from intervals import EMPTY
from intervals import REALS
from intervals import MultiInterval
from intervals import Size
from intervals import kernel
from tests.strategies import cut_tuples

inf = math.inf
P = MultiInterval.parse
multi_intervals = cut_tuples().map(MultiInterval.from_cuts)


# CONSTRUCTION

@pytest.mark.parametrize('made, text', [
    (MultiInterval(), '{}'),
    (MultiInterval(5), '[5]'),
    (MultiInterval(1, 2, end_closed=False), '[1, 2)'),
    (MultiInterval(1, 2, start_closed=False, end_closed=False), '(1, 2)'),
    (MultiInterval(1, inf), '[1, inf]'),  # literal: inf is a member
    (MultiInterval(1, 1, end_closed=False), '{}'),  # [1, 1) is empty
    (MultiInterval(5, start_closed=False, end_closed=False), '{}'),
    (MultiInterval(inf), '[inf]'),
    (MultiInterval(Fraction(4, 2)), '[2]'),
    (MultiInterval(-0.0), '[0.0]'),
    (MultiInterval.from_pieces([(3, 4), (1, 2, False, True)]), '{ (1, 2] , [3, 4] }'),
])
def test_constructors(made, text):
    assert str(made) == text


@pytest.mark.parametrize('args, kwargs, error', [
    ((2, 1), {}, ValueError),
    ((5,), {'start_closed': False}, ValueError),
    ((None, 3), {}, ValueError),
    ((math.nan,), {}, ValueError),
    (('1',), {}, TypeError),
    ((True,), {}, TypeError),
])
def test_constructor_errors(args, kwargs, error):
    with pytest.raises(error):
        MultiInterval(*args, **kwargs)


def test_from_cuts_validates():
    with pytest.raises(ValueError):
        MultiInterval.from_cuts(REALS.cuts[::-1])
    with pytest.raises(ValueError):
        MultiInterval.from_cuts((1, 2))


@given(multi_intervals)
def test_repr_round_trips(a):
    assert eval(repr(a), {'MultiInterval': MultiInterval}) == a


# EQUALITY AND HASHING

@given(multi_intervals, multi_intervals)
def test_eq_and_hash_are_structural(a, b):
    assert (a == b) == (a.cuts == b.cuts)
    assert (a != b) == (a.cuts != b.cuts)
    if a == b:
        assert hash(a) == hash(b)


def test_equal_across_number_types():
    assert MultiInterval(2) == MultiInterval(2.0) == MultiInterval(Fraction(2))
    assert len({MultiInterval(2), MultiInterval(2.0), MultiInterval(Fraction(2))}) == 1


def test_eq_does_not_coerce():
    assert MultiInterval(5) != 5
    assert not (MultiInterval(5) == 5)
    assert {MultiInterval(5): 'a'}.get(5) is None


def test_immutable():
    a = MultiInterval(1, 2)
    with pytest.raises(AttributeError):
        a._cuts = ()
    with pytest.raises(AttributeError):
        a.anything = 1
    with pytest.raises(AttributeError):
        del a._cuts


@given(multi_intervals)
def test_pickle_and_copy(a):
    assert pickle.loads(pickle.dumps(a)) == a
    assert copy.copy(a) == a and copy.deepcopy(a) == a


def test_sorted_by_sort_key():
    xs = [P('[3]'), P('[1, 2)'), P('(1, 2)'), P('{}'), P('[1, 2]')]
    assert [str(x) for x in sorted(xs, key=lambda x: x.sort_key)] == ['{}', '[1, 2)', '[1, 2]', '(1, 2)', '[3]']


# SET ALGEBRA DELEGATES TO THE KERNEL

@given(multi_intervals, multi_intervals)
def test_set_operators(a, b):
    assert (a | b).cuts == kernel.union(a.cuts, b.cuts) == a.union(b).cuts
    assert (a & b).cuts == kernel.intersection(a.cuts, b.cuts) == a.intersection(b).cuts
    assert (a ^ b).cuts == kernel.symmetric_difference(a.cuts, b.cuts) == a.symmetric_difference(b).cuts
    assert a.difference(b).cuts == kernel.difference(a.cuts, b.cuts)
    assert (~a).cuts == kernel.complement(a.cuts) == a.complement().cuts
    assert a.issubset(b) == kernel.is_subset(a.cuts, b.cuts) == (a in b)
    assert b.issuperset(a) == a.issubset(b)
    assert a.isdisjoint(b) == (not (a & b))


def test_scalars_coerce_in_set_operations():
    assert P('[1, 2]') | 5 == 5 | P('[1, 2]') == P('{[1, 2], [5]}')
    assert P('[1, 2]') & 2 == 2 & P('[1, 2]') == P('[2]')
    assert P('[1, 2]').union(5, P('[7]')) == P('{[1, 2], [5], [7]}')
    assert P('[1, 2]').difference(1) == P('(1, 2]')


def test_minus_is_not_set_difference():
    # `-` is arithmetic (M6); it must never silently mean set difference
    assert P('[1, 2]') - P('[1]') == P('[0, 1]')
    assert P('[1, 2]').difference(P('[1]')) == P('(1, 2]')
    with pytest.raises(TypeError):
        P('[1, 2]') | 'x'
    with pytest.raises(TypeError):
        P('[1, 2]').union('x')


# MEMBERSHIP AND SLICING

def test_contains():
    a = P('[1, 2) | [3]')
    assert 1 in a and 1.5 in a and 3 in a
    assert 2 not in a and 0 not in a and inf not in a
    assert math.nan not in a
    assert P('[1, 1.5]') in a and P('[1, 2]') not in a
    assert EMPTY in EMPTY  # v1 said False
    assert inf in MultiInterval(1, inf)
    with pytest.raises(TypeError):
        'a' in a


@pytest.mark.parametrize('sliced, expected', [
    (lambda a: a[0:5], '{ [0, 2) , [3, 5] }'),  # a 0 bound is a bound (v1 read it as missing)
    (lambda a: a[:1], '[-5, 1]'),
    (lambda a: a[3:], '[3, inf]'),
    (lambda a: a[2:2], '{}'),
    (lambda a: a[3:3], '[3]'),
])
def test_slicing_restricts_to_closed(sliced, expected):
    a = P('[-5, 2) | [3, inf]')
    assert str(sliced(a)) == expected


def test_slicing_errors():
    a = P('[1, 2]')
    with pytest.raises(TypeError):
        a[1]
    with pytest.raises(TypeError):
        a[0:5:1]
    with pytest.raises(ValueError):
        a[5:0]


def test_container_protocol():
    a = P('{ (1, 2] , [3] }')
    assert len(a) == 2 and bool(a) and not EMPTY and len(EMPTY) == 0
    assert list(a) == [P('(1, 2]'), P('[3]')] == list(a.pieces)


# PROPERTIES

def test_predicates():
    assert EMPTY.is_empty and not EMPTY.is_contiguous and not EMPTY.is_degenerate
    assert P('[1]').is_contiguous and P('[1]').is_degenerate and P('{1, 2}').is_degenerate
    assert not P('{1, 2}').is_contiguous and not P('[1, 2]').is_degenerate
    assert EMPTY.is_finite and P('[1, 2]').is_finite and not P('[inf]').is_finite
    assert P('{1, 2.0}').is_integral and not P('{1, 1/2}').is_integral and not P('[1, 2]').is_integral
    assert not EMPTY.is_integral and not P('[inf]').is_integral
    assert P('(0, 1]').is_positive and not P('[0, 1]').is_positive and not EMPTY.is_positive
    assert P('[-1, 0)').is_negative and not P('[-1, 0]').is_negative
    assert P('[0, 1]').is_non_negative and EMPTY.is_non_negative and not P('[-1, 1]').is_non_negative
    assert P('[-1, 0]').is_non_positive and not P('(-1, 1)').is_non_positive


def test_finiteness_of_an_exact_end_past_the_doubles():
    # M13g: math.isfinite overflowed on 10**400, so is_finite and finite raised OverflowError
    big = MultiInterval(-Fraction(10 ** 400, 3), 10 ** 400)
    assert big.is_finite and big.finite == big
    assert not (big | P('[inf]')).is_finite and (big | P('[inf]')).finite == big
    assert not MultiInterval(10 ** 400, math.inf).is_finite and MultiInterval(10 ** 400, math.inf).finite == EMPTY


def test_derived_sets():
    a = P('{ [-inf, -3] , [-1, 1) , [2] , [5, inf] }')
    assert a.finite == P('{ [-1, 1) , [2] }')  # pieces touching infinity are dropped whole, as v1
    assert a.positive == P('{ (0, 1) , [2] , [5, inf] }')
    assert a.negative == P('{ [-inf, -3] , [-1, 0) }')
    assert a.hull == P('[-inf, inf]') and P('(1, 2)').hull == P('(1, 2)')
    assert P('{ (1, 2) , (3, 4) }').closed_hull == P('[1, 4]')
    assert P('(1, inf)').closed_hull == P('[1, inf]')
    assert EMPTY.hull == EMPTY.closed_hull == EMPTY
    assert a.degenerate_points == {2}
    assert P('(1, inf]').size == Size(1, -1, 0)


def test_bounds():
    a = P('{ (1, 2] , [3, 4) }')
    assert (a.inf, a.inf_closed, a.sup, a.sup_closed) == (1, False, 4, False)
    with pytest.raises(ValueError):
        EMPTY.inf
    with pytest.raises(ValueError):
        EMPTY.sup_closed


def test_expand():
    assert P('{ (1, 2] , [4] }').expand(1) == P('(0, 5]')
    assert P('{ (1, 2) , (4, 5) }').expand(1) == P('{ (0, 3) , (3, 6) }')  # the point 3 is still missing
    assert P('{ (1, 2) , [4, 5) }').expand(1) == P('(0, 6)')
    assert P('[-inf, 0]').expand(Fraction(1, 2)) == P('[-inf, 1/2]')
    with pytest.raises(ValueError):
        P('[1]').expand(-1)
    with pytest.raises(ValueError):
        P('[1]').expand(inf)


def test_conversions():
    assert float(P('[1/2]')) == 0.5 and int(P('[3]')) == 3 and complex(P('[1]')) == 1
    with pytest.raises(ValueError):
        float(P('[1, 2]'))
    with pytest.raises(ValueError):
        int(P('{1, 2}'))
