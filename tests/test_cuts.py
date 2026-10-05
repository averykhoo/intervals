import math
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from multiinterval.cuts import Cut
from multiinterval.cuts import Side
from multiinterval.cuts import above
from multiinterval.cuts import as_end
from multiinterval.cuts import as_start
from multiinterval.cuts import below
from multiinterval.cuts import end_cut
from multiinterval.cuts import mirror
from multiinterval.cuts import start_cut
from tests.strategies import cuts
from tests.strategies import values


# the plan's table: the same cut reads differently as a start and as an end
@pytest.mark.parametrize('cut, start, end', [
    (below(3), (3, True), (3, False)),  # `[3` / `3)`
    (above(3), (3, False), (3, True)),  # `(3` / `3]`
])
def test_reading_table(cut, start, end):
    assert as_start(cut) == start
    assert as_end(cut) == end


@pytest.mark.parametrize('lo, lo_closed, hi, hi_closed, expected', [
    (1, True, 2, True, (below(1), above(2))),  # [1, 2]
    (1, False, 2, False, (above(1), below(2))),  # (1, 2)
    (1, True, 2, False, (below(1), below(2))),  # [1, 2)
    (1, True, 1, True, (below(1), above(1))),  # [1]
])
def test_piece_encoding(lo, lo_closed, hi, hi_closed, expected):
    assert (start_cut(lo, lo_closed), end_cut(hi, hi_closed)) == expected


def test_ordering_at_one_value():
    # just below 2 < just above 2 < just below 3
    assert below(2) < above(2) < below(3)
    # a closed end at 2 and a closed start at 2 are different boundaries
    assert end_cut(2, True) != start_cut(2, True)
    # an open end at 2 and a closed start at 2 are the same boundary: [1,2) | [2,3] tiles
    assert end_cut(2, False) == start_cut(2, True)
    # a closed end at 2 and an open start at 2 are the same boundary: [1,2] | (2,3] tiles
    assert end_cut(2, True) == start_cut(2, False)


def test_ordering_with_infinities():
    assert below(-math.inf) < above(-math.inf) < below(-10 ** 9) < above(10 ** 9) < below(math.inf) < above(math.inf)


@given(values, values, st.sampled_from(list(Side)), st.sampled_from(list(Side)))
def test_order_follows_value_then_side(a, b, side_a, side_b):
    assert (Cut(a, side_a) < Cut(b, side_b)) == ((a, side_a) < (b, side_b))


@given(values, st.booleans())
def test_start_round_trip(value, closed):
    assert as_start(start_cut(value, closed)) == (value, closed)


@given(values, st.booleans())
def test_end_round_trip(value, closed):
    assert as_end(end_cut(value, closed)) == (value, closed)


@given(cuts)
def test_mirror_is_an_involution(cut):
    assert mirror(mirror(cut)) == cut
    assert type(mirror(cut).side) is Side


@given(cuts, cuts)
def test_mirror_reverses_order(a, b):
    assert (a < b) == (mirror(b) < mirror(a))


@given(values, st.booleans())
def test_mirror_turns_a_start_into_an_end(value, closed):
    assert as_end(mirror(start_cut(value, closed))) == (-value, closed)


def test_negative_zero_is_normalized():
    cut = Cut(-0.0, Side.BELOW)
    assert cut == Cut(0.0, Side.BELOW) == Cut(0, Side.BELOW)
    assert math.copysign(1, cut.value) == 1
    assert repr(cut) == 'Cut(0.0, BELOW)'
    assert repr(mirror(Cut(0.0, Side.BELOW))) == 'Cut(0.0, ABOVE)'


def test_integral_fraction_becomes_int():
    assert type(Cut(Fraction(6, 3), Side.BELOW).value) is int
    assert type(Cut(Fraction(1, 3), Side.BELOW).value) is Fraction


def test_equal_values_of_different_types_are_equal_cuts():
    cuts_ = [Cut(1, Side.ABOVE), Cut(1.0, Side.ABOVE), Cut(Fraction(1), Side.ABOVE)]
    assert len(set(cuts_)) == 1


def test_numpy_scalars_become_python_numbers():
    np = pytest.importorskip('numpy')
    assert type(Cut(np.int64(3), Side.BELOW).value) is int
    zero = Cut(np.float64(-0.0), Side.BELOW).value
    assert type(zero) is float and math.copysign(1, zero) == 1


@pytest.mark.parametrize('value, error', [
    (math.nan, ValueError),
    ('1', TypeError),
    (True, TypeError),
    (1j, TypeError),
    (None, TypeError),
])
def test_bad_values(value, error):
    with pytest.raises(error):
        Cut(value, Side.BELOW)


def test_bad_side():
    with pytest.raises(ValueError):
        Cut(1, 0)
