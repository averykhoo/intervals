import copy
import math
import pickle
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import EMPTY
from intervals import REALS
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import Size
from intervals import kernel
from tests.strategies import cut_tuples
from tests.strategies import endpoint_values
from tests.strategies import probe_points
from tests.strategies import values

inf = math.inf
P = MultiInterval.parse
multi_intervals = cut_tuples().map(MultiInterval.from_cuts)
# the two classes: these set operations round nothing, so the outward class must give the same set
classes = pytest.mark.parametrize('cls', [MultiInterval, OutwardMultiInterval])
# half the default examples for each of the two, so a test at FUZZ_MULTIPLIER=50 takes about 30 s
# (2 x 2500 examples, 2026-10-03)
per_class = settings(max_examples=50)
# for the predicates: endpoints at 0 and +-inf often (the rays' ends), endpoints past the doubles
# (M13g: math.isfinite overflowed on them; probe_points cannot take a midpoint between 10**400 and a
# float, so not elsewhere), and a few added points, which become degenerate pieces beside the others.
# without the bias 50 examples rarely held a set starting at [0 or a point beside a piece, and
# sabotaging is_positive or is_degenerate (`all` -> `any`) passed (2026-10-03)
_wide_points = st.sampled_from([-inf, -1, 0, -0.0, Fraction(1, 2), 2.0, inf, 10 ** 400, Fraction(-10 ** 400, 3)])


@st.composite
def wide_cut_tuples(draw):
    cuts = draw(cut_tuples(values=st.one_of(endpoint_values, st.sampled_from([0, -inf, inf]), _wide_points)))
    points = draw(st.lists(_wide_points, max_size=3))
    return kernel.union(cuts, *(kernel.normalize([kernel.piece(p, p)]) for p in points))


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


@classes
@per_class
@given(cut_tuples(), st.one_of(st.none(), values, endpoint_values), st.one_of(st.none(), values, endpoint_values))
def test_slicing_is_the_closed_restriction(cls, cuts, start, stop):
    """`A[a:b]` against the kernel's `A & [a, b]` (a missing bound is +-inf, both ends closed, inf
    included) and against membership of the probe points; `a > b` is a ValueError"""
    a = cls.from_cuts(cuts)
    lo = -inf if start is None else start
    hi = inf if stop is None else stop
    if lo > hi:
        with pytest.raises(ValueError):
            a[start:stop]
        return
    restricted = a[start:stop]
    window = kernel.normalize([kernel.piece(lo, hi)])
    assert type(restricted) is cls
    assert restricted.cuts == kernel.intersection(cuts, window), (a, start, stop, restricted)
    for x in probe_points(cuts, window):
        assert (x in restricted) == (x in a and lo <= x <= hi), (a, start, stop, x)


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


def _ray(lo, hi, lo_closed, hi_closed):
    return kernel.normalize([kernel.piece(lo, hi, lo_closed, hi_closed)])


POSITIVE_RAY = _ray(0, inf, False, True)  # (0, inf]
NEGATIVE_RAY = _ray(-inf, 0, True, False)  # [-inf, 0)
NON_NEGATIVE_RAY = _ray(0, inf, True, True)  # [0, inf]
NON_POSITIVE_RAY = _ray(-inf, 0, True, True)  # [-inf, 0]
FINITE_LINE = _ray(-inf, inf, False, False)  # (-inf, inf)


def _bounded(cuts):
    """the closure of every piece is a `kernel.is_subset` of (-inf, inf): no point at, and no piece
    reaching, +-inf"""
    closure = kernel.normalize([kernel.piece(lo, hi) for lo, _, hi, _ in kernel.pieces(cuts)])
    return kernel.is_subset(closure, FINITE_LINE)


@classes
@per_class
@given(wide_cut_tuples())
def test_sign_predicates_are_subsets_of_the_rays(cls, cuts):
    """each sign predicate against `kernel.is_subset` of its ray: is_positive in (0, inf] and
    is_negative in [-inf, 0) (both False when empty), is_non_negative in [0, inf] and
    is_non_positive in [-inf, 0] (both True when empty)"""
    a = cls.from_cuts(cuts)
    assert a.is_positive == (bool(cuts) and kernel.is_subset(cuts, POSITIVE_RAY)), a
    assert a.is_negative == (bool(cuts) and kernel.is_subset(cuts, NEGATIVE_RAY)), a
    assert a.is_non_negative == kernel.is_subset(cuts, NON_NEGATIVE_RAY), a
    assert a.is_non_positive == kernel.is_subset(cuts, NON_POSITIVE_RAY), a


@classes
@per_class
@given(wide_cut_tuples())
def test_shape_predicates_against_the_kernel(cls, cuts):
    """is_empty, is_contiguous (non-empty and its own `kernel.hull`), degenerate_points (the pieces
    whose cut pair is `kernel.piece(v, v)`), is_degenerate (non-empty, every piece such a point),
    is_finite (`_bounded`: its closure in (-inf, inf)) and is_integral (degenerate, finite, every point
    an integer), each against the kernel"""
    a = cls.from_cuts(cuts)
    points = {start.value for start, end in kernel.pairs(cuts)
              if (start, end) == kernel.piece(start.value, start.value)}
    degenerate = bool(cuts) and len(points) == len(cuts) // 2
    finite = _bounded(cuts)
    assert a.is_empty == (cuts == kernel.EMPTY), a
    assert a.is_contiguous == (bool(cuts) and kernel.hull(cuts) == cuts), a
    assert a.degenerate_points == points, a
    assert a.is_degenerate == degenerate, a
    assert a.is_finite == finite, a
    assert a.is_integral == (degenerate and finite and all(Fraction(v).denominator == 1 for v in points)), a


@classes
@per_class
@given(cut_tuples())
def test_positive_and_negative_are_the_rays(cls, cuts):
    """`positive` and `negative` against `kernel.intersection` with (0, inf] and [-inf, 0), and
    against membership of the probe points (`x > 0`, `x < 0`, inf and -inf included)"""
    a = cls.from_cuts(cuts)
    positive, negative = a.positive, a.negative
    assert type(positive) is type(negative) is cls
    assert positive.cuts == kernel.intersection(cuts, POSITIVE_RAY), (a, positive)
    assert negative.cuts == kernel.intersection(cuts, NEGATIVE_RAY), (a, negative)
    for x in probe_points(cuts):
        assert (x in positive) == (x in a and x > 0), (a, x)
        assert (x in negative) == (x in a and x < 0), (a, x)


@classes
@per_class
@given(wide_cut_tuples())
def test_finite_keeps_the_pieces_off_infinity(cls, cuts):
    """`finite` against the `kernel.union` of the pieces of A whose closure is a `kernel.is_subset`
    of (-inf, inf): a piece reaching or touching +-inf is dropped whole (as v1), so `finite` lies in
    A & (-inf, inf), and is A exactly when A is_finite"""
    a = cls.from_cuts(cuts)
    finite = a.finite
    kept = [pair for pair in kernel.pairs(cuts) if _bounded(pair)]
    assert type(finite) is cls
    assert finite.cuts == kernel.union(*kept), (a, finite)
    assert kernel.is_subset(finite.cuts, kernel.intersection(cuts, FINITE_LINE)), (a, finite)
    assert (finite == a) == _bounded(cuts) == a.is_finite, (a, finite)


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


def test_no_shifts():
    """`<<` and `>>` are not defined (owner, 2026-10-04: dropped; scaling is `* 2 ** n`, the floor `// 2 ** n`)"""
    from intervals import DecoratedInterval, Dual
    for x in (P('[1, 3]'), OutwardMultiInterval(0.1), DecoratedInterval(P('[1, 3]')), Dual.variable(P('[1, 3]'))):
        for shift in (lambda: x << 1, lambda: x >> 1, lambda: 1 << x, lambda: 1 >> x):
            with pytest.raises(TypeError):
                shift()
