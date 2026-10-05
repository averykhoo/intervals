import math
import random

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import strategies as st

from multiinterval.cuts import above
from multiinterval.cuts import below
from multiinterval.kernel import EMPTY
from multiinterval.kernel import REALS
from multiinterval.kernel import Builder
from multiinterval.kernel import Size
from multiinterval.kernel import complement
from multiinterval.kernel import contains_point
from multiinterval.kernel import difference
from multiinterval.kernel import hull
from multiinterval.kernel import intersection
from multiinterval.kernel import is_subset
from multiinterval.kernel import is_valid
from multiinterval.kernel import normalize
from multiinterval.kernel import overlap_count
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import size
from multiinterval.kernel import symmetric_difference
from multiinterval.kernel import union
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import piece_pairs
from tests.strategies import probe_points

inf = math.inf


def mi(*specs):
    """test shorthand: each spec is `(lo, hi, lo_closed, hi_closed)` or a bare point"""
    return normalize(piece(s, s) if not isinstance(s, tuple) else piece(*s) for s in specs)


def naive_contains(cuts, x):
    """membership read straight off the pieces, independent of the bisect in the kernel"""
    return any((lo < x or (lo == x and lo_closed)) and (x < hi or (x == hi and hi_closed))
               for lo, lo_closed, hi, hi_closed in pieces(cuts))


def assert_matches(result, predicate, *operands):
    assert is_valid(result)
    for x in probe_points(result, *operands):
        assert naive_contains(result, x) == predicate(x), x


# NORMALIZATION

@given(st.lists(piece_pairs(), max_size=6))
def test_normalize_is_valid_and_covers_the_same_points(pairs_):
    result = normalize(pairs_)
    assert_matches(result, lambda x: any(naive_contains(p, x) for p in pairs_ if p[0] < p[1]),
                   *[p for p in pairs_ if p[0] < p[1]])


@given(cut_tuples(), st.randoms(use_true_random=False))
@example((above(0), below(5e-324)), random.Random(0))  # (0 + 5e-324) / 2 is 0.0
def test_normalize_is_canonical(cuts, rnd):
    # re-express the same set as split, duplicated and shuffled pieces. a piece is split only at a
    # midpoint strictly inside it: between adjacent floats the midpoint rounds onto an end, and the
    # split would add that end as a point (the oracle's bug found 2026-09-27, not the kernel's)
    parts = []
    for lo, lo_closed, hi, hi_closed in pieces(cuts):
        mid = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else None
        if mid is not None and lo < mid < hi:
            parts += [piece(lo, mid, lo_closed, False), piece(mid, hi, True, hi_closed)]
        else:
            parts.append(piece(lo, hi, lo_closed, hi_closed))
    parts += rnd.sample(parts, len(parts) // 2)
    rnd.shuffle(parts)
    assert normalize(parts) == cuts


@pytest.mark.parametrize('pieces_, expected', [
    # equal cuts tile: [1,2) | [2,3] = [1,3]
    ([(1, 2, True, False), (2, 3)], mi((1, 3))),
    # [1,2] | (2,3] = [1,3]
    ([(1, 2), (2, 3, False, True)], mi((1, 3))),
    # [1,2) | (2,3]: the point 2 is missing
    ([(1, 2, True, False), (2, 3, False, True)], (below(1), below(2), above(2), above(3))),
    # empty pieces drop out
    ([(1, 1, True, False), (1, 1, False, False)], EMPTY),
    ([(-inf, inf)], REALS),
])
def test_normalize_table(pieces_, expected):
    assert normalize(piece(*p) for p in pieces_) == expected


def test_reversed_values_are_an_error():
    with pytest.raises(ValueError):
        piece(2, 1)


# SET ALGEBRA

@given(cut_tuples(), cut_tuples())
def test_union(a, b):
    assert_matches(union(a, b), lambda x: naive_contains(a, x) or naive_contains(b, x), a, b)


@given(cut_tuples(), cut_tuples())
def test_intersection(a, b):
    assert_matches(intersection(a, b), lambda x: naive_contains(a, x) and naive_contains(b, x), a, b)


@given(cut_tuples(), cut_tuples())
def test_difference(a, b):
    assert_matches(difference(a, b), lambda x: naive_contains(a, x) and not naive_contains(b, x), a, b)


@given(cut_tuples(), cut_tuples(), cut_tuples())
def test_symmetric_difference(a, b, c):
    assert_matches(symmetric_difference(a, b, c),
                   lambda x: (naive_contains(a, x) + naive_contains(b, x) + naive_contains(c, x)) % 2 == 1,
                   a, b, c)


@given(cut_tuples())
def test_complement(a):
    assert_matches(complement(a), lambda x: not naive_contains(a, x), a)


@given(st.lists(cut_tuples(), min_size=1, max_size=4), st.integers(1, 4))
def test_overlap_count(operands, n):
    assert_matches(overlap_count(operands, n),
                   lambda x: sum(naive_contains(c, x) for c in operands) >= n, *operands)


@given(cut_tuples(), cut_tuples())
# adjacent doubles: the gap between them has no float midpoint, found by the suite 2026-09-25
@example(a=mi((-20.0, -2, False, False)), b=mi((-19.999999999999996, -2, True, False)))
def test_is_subset(a, b):
    expected = all(naive_contains(b, x) for x in probe_points(a, b) if naive_contains(a, x))
    assert is_subset(a, b) == expected


@given(cut_tuples(), st.one_of(st.sampled_from([-inf, -1, 0, 0.5, 1, inf]), st.fractions(-3, 3)))
def test_contains_point(a, x):
    assert contains_point(a, x) == naive_contains(a, x)


@given(cut_tuples())
def test_complement_is_an_involution(a):
    assert complement(complement(a)) == a


@given(cut_tuples(), cut_tuples())
def test_de_morgan(a, b):
    assert complement(union(a, b)) == intersection(complement(a), complement(b))
    assert complement(intersection(a, b)) == union(complement(a), complement(b))


@given(cut_tuples())
def test_excluded_middle(a):
    assert union(a, complement(a)) == REALS
    assert intersection(a, complement(a)) == EMPTY
    assert union(a, a) == intersection(a, a) == a


@given(cut_tuples(), cut_tuples(), cut_tuples())
def test_commutative_and_associative(a, b, c):
    assert union(a, b) == union(b, a)
    assert intersection(a, b) == intersection(b, a)
    assert union(union(a, b), c) == union(a, union(b, c)) == union(a, b, c)
    assert intersection(intersection(a, b), c) == intersection(a, intersection(b, c)) == intersection(a, b, c)
    assert difference(a, b) == intersection(a, complement(b))


def test_complement_table():
    assert complement(EMPTY) == REALS
    assert complement(REALS) == EMPTY
    # (-inf, inf) leaves both infinities
    assert complement(mi((-inf, inf, False, False))) == mi(-inf, inf)
    assert complement(mi((1, 2))) == mi((-inf, 1, True, False), (2, inf, False, True))


def test_hull():
    assert hull(EMPTY) == EMPTY
    assert hull(mi((1, 2, False, True), 5)) == mi((1, 5, False, True))


def test_builder():
    built = Builder().add_piece(3, 4).add_point(1).add(mi((0, 1, True, False))).add_piece(4, 5, False, False).build()
    assert built == mi((0, 1), (3, 5, True, False))


# SIZE

@pytest.mark.parametrize('cuts, expected', [
    (mi(1), Size(0, 0, 1)),
    (mi((1, 2, True, False)), Size(0, 1, 0)),
    (mi((1, 2)), Size(0, 1, 1)),
    (mi((1, 2, False, False)), Size(0, 1, -1)),
    (mi((1, inf, False, True)), Size(1, -1, 0)),
    (mi((1, inf, True, False)), Size(1, -1, 0)),
    (mi((-inf, 1)), Size(1, 1, 1)),
    (mi((-inf, -1)), Size(1, -1, 1)),
    (REALS, Size(2, 0, 1)),
    (mi((-inf, inf, False, False)), Size(2, 0, -1)),
    (mi(inf), Size(0, 0, 1)),
    (EMPTY, Size(0, 0, 0)),
])
def test_size_table(cuts, expected):
    assert size(cuts) == expected


def test_size_tiling_table():
    assert size(mi(1)) + size(mi((1, 2, False, False))) + size(mi(2)) == size(mi((1, 2)))
    assert size(mi((0, 1, True, False))) + size(mi((1, 2, True, False))) == size(mi((0, 2, True, False)))
    assert size(mi((0, inf, True, False))) + size(mi(inf)) == size(mi((0, inf)))
    assert size(mi((-inf, 0, True, False))) + size(mi((0, inf))) == size(REALS)


@given(exact_cut_tuples, exact_cut_tuples)
def test_size_is_additive_on_disjoint_sets(a, b):
    b = difference(b, a)
    assert size(a) + size(b) == size(union(a, b))


def test_size_is_lex_ordered():
    assert size(mi((0, inf, True, False))) > size(mi((0, 10 ** 9)))
    assert size(mi((0, 1))) > size(mi((0, 1, True, False))) > size(mi((0, 1, False, False)))
