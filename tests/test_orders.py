"""
the interval orders and the interior (M13c, D10): `weakly_less`, `strictly_less`, `.interior`

* the orders are **on the ends**, so a multi-interval's are its hull's, and open or closed does not
  matter; against an oracle written from 1788's own quantified definitions (`less`: every point of
  each operand has a point of the other on the right side, `<=`; `strictLess` the same with `<`),
  decided by brute force on a grid, over 1788's reading of the hulls
* `.interior` is every end opened: against the definition (a point is interior iff it is a real
  number of the set that is not an end value, the pieces being maximal), on probe points; and
  1788's `interior(A, B)`, `A.within(B.interior)`, against a grid oracle for `A ⊆ int B`
* soundness at sampled points (exact and float operands, both classes), and the laws: the interior
  is open, inside the set, idempotent, isotone and distributes over `&`
* the itf1788 vectors of the three ops run in tests/itf1788; some are `@example`s here
"""
import math
import random
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import EMPTY
from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval.kernel import contains_point
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import union
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import probe_points

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf


def one(lo, hi, lo_closed=True, hi_closed=True):
    return normalize([piece(lo, hi, lo_closed, hi_closed)])


def p(text: str) -> MultiInterval:
    return M.parse(text)


# EXAMPLES

@pytest.mark.parametrize('a, b, weakly, strictly', [
    ('{}', '{}', True, True),  # 1788: less [empty] [empty] = true, strictLess too
    ('[1, 2]', '{}', False, False),  # less [1.0,2.0] [empty] = false
    ('{}', '[1, 2]', False, False),  # less [empty] [1.0,2.0] = false
    ('(-inf, inf)', '(-inf, inf)', True, True),  # strictLess [entire] [entire] = true
    ('[-inf, inf]', '[-inf, inf]', True, True),  # closed at ±inf: the same ends
    ('[1, 2]', '(-inf, inf)', False, False),
    ('[1, 2]', '[1, 2]', True, False),
    ('[1, 3.5]', '[3, 4]', True, True),
    ('[1, 4]', '[3, 4]', True, False),
    ('[0, 2]', '[0, 2)', True, False),  # on the ends: an open end is still that end
    ('[0, 2)', '(0, 2]', True, False),
    ('(-inf, 1]', '(-inf, 2)', True, True),  # two starts at -inf count
    ('[-inf, 1]', '(-inf, 2]', True, True),
    ('[1, inf]', '[2, inf)', True, True),  # two ends at inf count
    ('[inf]', '[inf]', True, False),  # a start at inf is a point, not less than itself
    ('[-inf]', '[-inf]', True, False),  # an end at -inf likewise
    ('[-inf]', '[inf]', True, True),
    ('[-inf, 0]', '[-inf]', False, False),
    ('[5, inf]', '[inf]', True, True),
    ('[0, 1] | [5, 6]', '[2, 6]', True, False),  # the hull [0, 6]
    ('[0, 1] | [5, 6]', '[2, 3] | [7]', True, True),  # hulls [0, 6] and [2, 7]
    ('[3]', '[1, 2] | [4]', False, False),  # inf [3] > inf B = 1
])
def test_order_examples(a, b, weakly, strictly):
    assert p(a).weakly_less(p(b)) is weakly
    assert p(a).strictly_less(p(b)) is strictly
    assert O.parse(a).strictly_less(O.parse(b)) is strictly and p(a).weakly_less(O.parse(b)) is weakly


def test_orders_coerce_numbers_only():
    assert M(1, 2).weakly_less(2) and M(1).strictly_less(Fraction(3, 2)) and not M(1, 2).strictly_less(2)
    with pytest.raises(TypeError):
        M(1, 2).weakly_less('[2, 3]')
    with pytest.raises(TypeError):
        M(1, 2).strictly_less(None)


def test_orders_are_not_pointwise():
    """D10: `<` and `<=` stay pointwise; the orders are the named methods"""
    a, b = M(1, 3), M(2, 4)
    assert repr(a < b) == 'BOTH' and repr(a <= b) == 'BOTH'
    assert a.weakly_less(b) and a.strictly_less(b)


@pytest.mark.parametrize('a, interior', [
    ('{}', '{}'),
    ('[2]', '{}'),  # a degenerate piece has no interior
    ('[0, 1] | [2] | [3]', '(0, 1)'),
    ('[0, 1)', '(0, 1)'),
    ('[0, 1) | (1, 2]', '{ (0, 1) , (1, 2) }'),  # the missing point stays missing
    ('[5, inf]', '(5, inf)'),  # a closed end at inf is a point with no real neighbourhood
    ('[5, inf)', '(5, inf)'),
    ('[-inf, inf]', '(-inf, inf)'),
    ('(-inf, inf)', '(-inf, inf)'),
    ('[inf]', '{}'),
    ('[-inf] | [0, 1] | [inf]', '(0, 1)'),
    ('[-inf, -1] | [1, inf]', '{ (-inf, -1) , (1, inf) }'),
])
def test_interior_examples(a, interior):
    assert p(a).interior == p(interior)
    assert type(O.parse(a).interior) is O and O.parse(a).interior == O.parse(interior)


def test_interior_of_a_point_with_an_int_and_a_float_end():
    """`(Cut(0, BELOW), Cut(0.0, ABOVE))` is the point 0, so its interior is empty"""
    assert M.from_cuts(normalize([piece(0, 0.0)])).interior == EMPTY


@pytest.mark.parametrize('a, b, want', [
    ('{}', '{}', True),  # 1788: interior [empty] [empty] = true
    ('{}', '[0, 4]', True),
    ('[0, 4]', '{}', False),
    ('(-inf, inf)', '(-inf, inf)', True),  # the input rule: entire is open at both ends
    ('[-inf, inf]', '[-inf, inf]', False),  # ours, closed at ±inf: ±inf are not interior
    ('[0, 4]', '[0, 4]', False),
    ('[1, 2]', '[0, 4]', True),
    ('[0]', '[-2, 4]', True),
    ('[0]', '[0]', False),  # interior [0.0,0.0] [-0.0,-0.0] = false
    ('[1, inf)', '[0, inf)', True),  # interior [1, infinity] [0, infinity] under the input rule
    ('[1, inf]', '[0, inf]', False),  # ours: inf is a point of A and not interior to B
    ('[1, 2] | [5]', '[0, 3) | (4, 6)', True),
    ('[1, 3]', '[0, 3) | (3, 6)', False),
])
def test_1788_interior(a, b, want):
    assert p(a).within(p(b).interior) is want


# THE ORACLES

# a grid of halves: with integer ends in [-5, 5] a set is the same on every point strictly between
# two integers, so every order and neighbourhood question is decided on the grid exactly. the
# quantified side runs over [-6, 6], the witnesses over [-8, 8], so an unbounded operand always has
# a witness beyond every finite end; a neighbourhood is a quarter on each side
GRID = [Fraction(k, 2) for k in range(-16, 17)]
ALL = [x for x in GRID if abs(x) <= 6]
QUARTER = Fraction(1, 4)

grid_ends = st.one_of(st.integers(-5, 5), st.integers(-5, 5), st.sampled_from([-INF, INF]))
grid_cut_tuples = cut_tuples(values=grid_ends, max_pieces=4)


def _reading_1788(cuts):
    """1788's reading of the hull: finite ends closed, infinite ends open; None for a hull that is
    a single point at ±inf (1788 has no such interval), `()` for the empty set"""
    if not cuts:
        return ()
    lo, hi = cuts[0].value, cuts[-1].value
    if lo == hi and math.isinf(lo):
        return None
    return one(lo, hi, not math.isinf(lo), not math.isinf(hi))


def _points(cuts, grid):
    return [x for x in grid if contains_point(cuts, x)]


def _quantified(a, b, rel) -> bool:
    """1788's definition: every x of a has a y of b with rel(x, y), and every y of b an x of a"""
    return (all(any(rel(x, y) for y in _points(b, GRID)) for x in _points(a, ALL))
            and all(any(rel(x, y) for x in _points(a, GRID)) for y in _points(b, ALL)))


def _interior_1788(a, b) -> bool:
    """`a ⊆ int b`: no point of a at ±inf, and each grid point of a has a neighbourhood in b"""
    if contains_point(a, INF) or contains_point(a, -INF):
        return False
    return all(contains_point(b, x - QUARTER) and contains_point(b, x) and contains_point(b, x + QUARTER)
               for x in _points(a, ALL))


# THE ORDERS

@settings(max_examples=300, deadline=None)
@given(a=grid_cut_tuples, b=grid_cut_tuples)
@example(a=(), b=())  # less [empty] [empty] = true
@example(a=one(1, 2), b=())  # less [1.0,2.0] [empty] = false
@example(a=one(-INF, INF, False, False), b=one(-INF, INF, False, False))  # strictLess [entire] [entire]
@example(a=one(1, 2), b=one(-INF, INF, False, False))  # less [1.0,2.0] [entire] = false
@example(a=one(0, 0), b=one(0, INF, True, False))  # mpfi: less [0.0, 0.0] [0.0, +infinity] = true
@example(a=one(-INF, 5, False), b=one(0, 0))  # mpfi: less [-infinity, 5.0] [0.0, 0.0] = false
@example(a=one(1, 4), b=one(3, 4))  # strictLess [1.0,4.0] [3.0,4.0] = false
@example(a=one(-3, -1), b=one(-2, -1))  # strictLess [-3.0,-1.5] [-2.0,-1.0] (integer ends here)
@example(a=union(one(0, 1), one(5, 5)), b=one(2, 5, True, False))  # the hulls [0, 5] and [2, 5)
def test_orders_match_1788_definitions(a, b):
    ra, rb = _reading_1788(a), _reading_1788(b)
    if ra is None or rb is None:
        return
    A, B = M.from_cuts(a), M.from_cuts(b)
    assert A.weakly_less(B) is _quantified(ra, rb, lambda x, y: x <= y), (A, B)
    assert A.strictly_less(B) is _quantified(ra, rb, lambda x, y: x < y), (A, B)


@settings(max_examples=300, deadline=None)
@given(a=cut_tuples(), b=cut_tuples())
@example(a=one(INF, INF), b=one(INF, INF))  # a point at inf: weakly, not strictly, less than itself
@example(a=one(-INF, -INF), b=one(-INF, -INF))
@example(a=one(-INF, INF), b=one(-INF, INF, False, False))  # flags do not matter at ±inf either
@example(a=union(one(0, 1), one(5, 6)), b=one(2, 6))
def test_orders_are_on_the_ends(a, b):
    """the plan's definition, from the public ends; the hull and the closed hull change nothing"""
    A, B = M.from_cuts(a), M.from_cuts(b)
    weakly, strictly = A.weakly_less(B), A.strictly_less(B)
    assert weakly is A.hull.weakly_less(B.hull) is A.closed_hull.weakly_less(B.closed_hull)
    assert strictly is A.hull.strictly_less(B.hull) is A.closed_hull.strictly_less(B.closed_hull)
    if not A or not B:
        assert weakly is strictly is (not A and not B)
        return
    assert weakly is (A.inf <= B.inf and A.sup <= B.sup)
    lo = A.inf < B.inf or A.inf == B.inf == -INF
    hi = A.sup < B.sup or A.sup == B.sup == INF
    assert strictly is (lo and hi)
    assert not strictly or weakly
    assert A.weakly_less(A)
    assert A.strictly_less(A) is (A.inf == -INF and A.sup == INF)


@settings(max_examples=200, deadline=None)
@given(a=cut_tuples(), b=cut_tuples(), c=cut_tuples())
def test_weakly_less_is_a_preorder_on_the_ends(a, b, c):
    A, B, C = M.from_cuts(a), M.from_cuts(b), M.from_cuts(c)
    if A.weakly_less(B) and B.weakly_less(C):
        assert A.weakly_less(C)
    if A.strictly_less(B) and B.strictly_less(C):
        assert A.strictly_less(C)
    if A.weakly_less(B) and B.weakly_less(A):
        assert A.hull.closed_hull == B.hull.closed_hull


@settings(max_examples=300, deadline=None)
@given(a=cut_tuples(), b=cut_tuples(), cls=st.sampled_from([M, O]), rng=st.randoms(use_true_random=False))
@example(a=one(5, INF), b=one(INF, INF), cls=M, rng=random.Random(0))  # inf in A, ends equal at inf
@example(a=one(-INF, 0), b=one(-INF, -INF), cls=M, rng=random.Random(0))
def test_orders_sound_at_sampled_points(a, b, cls, rng):
    """weakly less: a sampled point of either has a point of the other's closure on the right side.
    strictly less: likewise with `<` for the real points (1788's intervals hold no infinity)"""
    A, B = cls.from_cuts(a), cls.from_cuts(b)
    xs, ys = sample(a, 12, rng), sample(b, 12, rng)
    if A.weakly_less(B):
        assert all((x <= B.closed_hull).possibly for x in xs), (A, B)
        assert all((A.closed_hull <= y).possibly for y in ys), (A, B)
    if A.strictly_less(B):
        assert all((x < B).possibly for x in xs if math.isfinite(x)), (A, B)
        assert all((A < y).possibly for y in ys if math.isfinite(y)), (A, B)


# THE INTERIOR

@settings(max_examples=400, deadline=None)
@given(a=cut_tuples())
@example(a=one(2, 2))  # the interior of [2] is empty
@example(a=one(INF, INF))
@example(a=one(5, INF))  # (5, inf)
@example(a=one(-INF, INF))  # (-inf, inf)
@example(a=union(one(0, 1, True, False), one(1, 2, False)))  # (0, 1) | (1, 2)
@example(a=normalize([piece(0, 0.0)]))  # an int and a float end: still a point
def test_interior_is_every_end_opened(a):
    """at every probe point, `x` is interior iff it is a real point of the set and no end value of it
    (normalized pieces are maximal, so a point of the set at an end value has a gap beside it)"""
    A = M.from_cuts(a)
    ends = {v for lo, _, hi, _ in pieces(a) for v in (lo, hi)}
    interior = A.interior
    for x in probe_points(a):
        want = x in A and math.isfinite(x) and x not in ends
        assert (x in interior) is want, (A, x)
    for lo, lo_closed, hi, hi_closed in pieces(interior.cuts):
        assert not lo_closed and not hi_closed and lo < hi


@settings(max_examples=300, deadline=None)
@given(a=cut_tuples(), b=cut_tuples(), cls=st.sampled_from([M, O]))
def test_interior_laws(a, b, cls):
    A, B = cls.from_cuts(a), cls.from_cuts(b)
    assert type(A.interior) is cls
    assert A.interior.within(A)
    assert A.interior.interior == A.interior
    assert (A & B).interior == A.interior & B.interior
    if A.within(B):
        assert A.interior.within(B.interior)  # isotone
    assert A.hull.interior.issuperset(A.interior)


@settings(max_examples=300, deadline=None)
@given(a=grid_cut_tuples, b=grid_cut_tuples)
@example(a=(), b=())  # interior [empty] [empty] = true
@example(a=one(0, 4), b=())  # interior [0.0,4.0] [empty] = false
@example(a=one(0, 0), b=one(0, 0))  # interior [0.0,0.0] [-0.0,-0.0] = false
@example(a=one(1, INF, True, False), b=one(0, INF, True, False))  # interior [1, infinity] [0, infinity]
@example(a=one(-INF, INF, False, False), b=one(-INF, INF, False, False))  # [entire] [entire] = true
@example(a=one(-INF, INF, False, False), b=one(0, 4))  # interior [entire] [0.0,4.0] = false
@example(a=one(-2, 2), b=one(-2, 4))  # interior [-2.0,2.0] [-2.0,4.0] = false
@example(a=one(0, 0), b=one(-2, 2))  # c-xsc: interior [0.0, 0.0] [-2.0, 2.0] = true
@example(a=one(1, INF), b=one(0, INF))  # ours, closed at inf: inf is not interior
@example(a=one(1, 1), b=union(one(0, 1, True, False), one(1, 2, False)))  # the missing point
def test_1788_interior_matches_the_neighbourhood_definition(a, b):
    A, B = M.from_cuts(a), M.from_cuts(b)
    assert A.within(B.interior) is _interior_1788(a, b), (A, B)
