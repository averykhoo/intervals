"""
the time layer (M8, D30): `intervals/time_interval.py`. the pins name the D30 point each guards; the hypothesis
properties check that the wrapper is a homomorphism onto the numeric class (the seconds of `A op B` are
`A.seconds op B.seconds`), that datetimes and timedeltas round-trip exactly over their whole range, and that an
aware reading does not depend on the zone. pandas is in `tests/test_time_pandas.py`.
"""
import copy
import datetime
import math
import operator
import os
import pickle
import subprocess
import sys
import warnings
import zoneinfo
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import intervals
from intervals import NEG_INF
from intervals import POS_INF
from intervals import DateTimeInterval
from intervals import IndeterminateResultWarning
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import Size
from intervals import TimeDeltaInterval
from intervals import TruthSet
from intervals.relations import BOTH
from intervals.relations import FALSE
from intervals.relations import NEITHER
from intervals.relations import TRUE
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples

ROOT = Path(__file__).resolve().parent.parent
D, T, M = DateTimeInterval, TimeDeltaInterval, MultiInterval
inf = math.inf
dt, td, date = datetime.datetime, datetime.timedelta, datetime.date
UTC = datetime.timezone.utc
SGT = zoneinfo.ZoneInfo('Asia/Singapore')
NY = zoneinfo.ZoneInfo('America/New_York')
DAY = td(days=1)
HOUR = td(hours=1)
JAN1 = date(2024, 1, 1)
NOON = dt(2024, 1, 1, 12)
JAN1_SECONDS = 1704067200  # 2024-01-01 00:00 since 1970-01-01 00:00, wall clock
EVAL_NAMESPACE = {'datetime': datetime, 'zoneinfo': zoneinfo, **{n: getattr(intervals, n) for n in intervals.__all__}}

# exact seconds sets: the numeric class's strategies (int, Fraction and +-inf ends)
naive_sets = exact_cut_tuples.map(lambda cuts: D.from_seconds(M.from_cuts(cuts)))
duration_sets = exact_cut_tuples.map(lambda cuts: T.from_seconds(M.from_cuts(cuts)))
# whole microseconds around 2024 and the infinities: sets every read-out can show
microseconds = st.integers(-10 ** 6, 10 ** 6).map(lambda k: Fraction(JAN1_SECONDS * 10 ** 6 + k * 997, 10 ** 6))
us_cut_tuples = cut_tuples(values=st.one_of(st.sampled_from([-inf, inf, JAN1_SECONDS]), microseconds))
factors = st.one_of(st.integers(-5, 5), st.fractions(-5, 5, max_denominator=6), st.floats(-5, 5),
                    st.sampled_from([-inf, inf]))
DISPLAY_ZONES = [UTC, SGT, NY]


def datetime_sets(aware: bool):
    """exact seconds sets, naive or aware in one of three display zones (the review's F8: aware sets too)"""
    zones = st.sampled_from(DISPLAY_ZONES) if aware else st.none()
    return st.builds(lambda cuts, tz: D.from_seconds(M.from_cuts(cuts), tz=tz), exact_cut_tuples, zones)


# a pair of the same kind, naive or aware (each aware one in its own zone)
datetime_set_pairs = st.booleans().flatmap(lambda aware: st.tuples(datetime_sets(aware), datetime_sets(aware)))


def has_finite_end(x) -> bool:
    return any(-inf < c.value < inf for c in x.seconds.cuts)


def display_zone(result, *operands):
    """the zone a result shows: the first operand's that has one, None when the result has no finite end"""
    return next((o.tz for o in operands if o.tz is not None), None) if has_finite_end(result) else None


def seconds_of_timedelta(x: td) -> Fraction:
    return Fraction(x // td(microseconds=1), 10 ** 6)


def seconds_of_naive(d: dt) -> Fraction:
    """an oracle independent of the library's subtraction: whole microseconds since datetime.min"""
    return Fraction((d - dt.min) // td(microseconds=1), 10 ** 6) - Fraction((dt(1970, 1, 1) - dt.min) // td(microseconds=1), 10 ** 6)


def seconds_of_aware(d: dt) -> Fraction:
    """the wall clock less the offset, in exact seconds (no datetime arithmetic, which overflows at the range ends)"""
    return seconds_of_naive(d.replace(tzinfo=None)) - seconds_of_timedelta(d.utcoffset())


def wall_of_instant(d: dt) -> dt:
    """the naive wall time `d`'s instant reads as in its zone: `d`'s own, unless a DST gap skips it (`2024-03-10 02:30`
    in New York is 07:30 UTC, which reads as 03:30). where the instant has no datetime (near the range ends) `d`'s
    own, so a gap there would fail loudly"""
    try:
        return d.astimezone(datetime.timezone.utc).astimezone(d.tzinfo).replace(tzinfo=None)
    except OverflowError:
        return d.replace(tzinfo=None)




def quiet(fn, *args):
    """an op whose warnings (empty, indeterminate, hull) the test compares rather than raises"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args)


# CONSTRUCTION AND READINGS (D30 (a))

def test_naive_is_wall_clock_not_timestamp():
    """D30 (a): naive seconds count from naive 1970-01-01 by subtraction; `timestamp()` is local time, raises
    on windows before 1970 and for datetime.min/max, and would make a DST day 23 or 25 hours"""
    assert D(dt(1970, 1, 1)).inf_seconds == 0
    assert D(NOON).inf_seconds == JAN1_SECONDS + 12 * 3600
    assert D(dt.min).inf_seconds == -62135596800
    assert D(dt.max).sup_seconds == Fraction(253402300799999999, 10 ** 6)
    assert D(dt.min).inf == dt.min and D(dt.max).sup == dt.max
    # the US spring-forward and fall-back days are 86400 s, as every naive day
    for day in (date(2024, 3, 10), date(2024, 11, 3)):
        assert D(day).total_seconds == 86400


def test_aware_is_utc_seconds():
    assert D(dt(1970, 1, 1, tzinfo=UTC)).inf_seconds == 0
    assert D(dt(1970, 1, 1, 7, 30, tzinfo=SGT)).inf_seconds == 0  # singapore was utc+7:30 then
    assert D(dt(2024, 1, 1, 8, tzinfo=SGT)).inf_seconds == JAN1_SECONDS
    assert D(dt(2024, 3, 10, 3, tzinfo=NY)).inf_seconds - D(dt(2024, 3, 10, 1, tzinfo=NY)).inf_seconds == 3600


def test_timedelta_reading_keeps_the_microsecond():
    """D30 (a): `days*86400 + seconds + microseconds/10**6`; `total_seconds()` drops it at 10**6 days"""
    x = td(days=10 ** 6, microseconds=1)
    assert x.total_seconds() == 86400000000.0  # python's float lost it
    assert T(x).inf_seconds == 86400000000 + Fraction(1, 10 ** 6)
    assert T(x).inf == x
    assert T(td.min).inf == td.min and T(td.max).sup == td.max


@pytest.mark.parametrize('make', [
    lambda: D(NOON, dt(2024, 1, 2, tzinfo=SGT)),
    lambda: D(dt(2024, 1, 2, tzinfo=SGT), NOON),
    lambda: D(JAN1, dt(2024, 1, 2, tzinfo=SGT)),
    lambda: D(NOON, tz=SGT),
    lambda: D(NOON) | D(dt(2024, 1, 2, tzinfo=SGT)),
    lambda: D(NOON).union(D(NOON), D(dt(2024, 1, 2, tzinfo=SGT))),
    lambda: D(NOON) & dt(2024, 1, 2, tzinfo=SGT),
    lambda: dt(2024, 1, 2, tzinfo=SGT) | D(NOON),
    lambda: D(NOON).difference(D(dt(2024, 1, 2, tzinfo=SGT))),
    lambda: D(NOON) < D(dt(2024, 1, 2, tzinfo=SGT)),
    lambda: D(NOON) >= dt(2024, 1, 2, tzinfo=SGT),
    lambda: D(NOON).before(dt(2024, 1, 2, tzinfo=SGT)),
    lambda: D(NOON) - D(dt(2024, 1, 2, tzinfo=SGT)),
    lambda: dt(2024, 1, 2, tzinfo=SGT) - D(NOON),
    lambda: dt(2024, 1, 2, tzinfo=SGT) in D(NOON),
    lambda: JAN1 in D(dt(2024, 1, 1, tzinfo=SGT), dt(2024, 1, 3, tzinfo=SGT)),
    lambda: D(dt(2024, 1, 1, tzinfo=SGT), dt(2024, 1, 3, tzinfo=SGT))[JAN1:],
    lambda: D(NOON).astimezone(UTC),
])
def test_naive_and_aware_never_mix(make):
    """D30 (a): mixing naive and aware raises TypeError, as python and pandas do"""
    with pytest.raises(TypeError):
        make()


def test_naive_and_aware_are_unequal():
    """`==` stays an answer (False), as python's `naive == aware` is"""
    assert D(dt(1970, 1, 1)) != D(dt(1970, 1, 1, tzinfo=UTC))
    assert D(dt(1970, 1, 1)).seconds == D(dt(1970, 1, 1, tzinfo=UTC)).seconds


def test_zones_differ_but_compare_as_instants():
    """D30 (a): aware ends in different zones are one set of instants; the left operand's zone is kept for
    display only and is never part of `==` or the hash"""
    a = D(dt(2024, 1, 1, 8, tzinfo=SGT))
    b = D(dt(2024, 1, 1, 0, tzinfo=UTC))
    assert a == b and hash(a) == hash(b)
    assert a.tz is SGT and b.tz is UTC
    assert (a | b).tz is SGT and (b | a).tz is UTC
    mixed = D(dt(2024, 1, 1, 9, tzinfo=SGT), dt(2024, 1, 1, 9, tzinfo=NY))
    assert mixed.tz is SGT
    assert mixed.sup == dt(2024, 1, 1, 9, tzinfo=NY) and mixed.sup.tzinfo is SGT
    assert mixed.total_seconds == 13 * 3600  # 01:00 UTC to 14:00 UTC
    assert (mixed < dt(2024, 1, 1, 23, tzinfo=SGT)) == TRUE  # 15:00 UTC
    assert D(dt(2024, 1, 1, tzinfo=SGT), tz=NY).tz is NY


def test_astimezone_keeps_the_instants():
    a = D(dt(2024, 1, 1, 8, tzinfo=SGT), dt(2024, 1, 1, 9, tzinfo=SGT))
    b = a.astimezone(NY)
    assert b == a and b.seconds == a.seconds and b.tz is NY
    assert b.inf == dt(2023, 12, 31, 19, tzinfo=NY) and b.inf.tzinfo is NY
    assert D(NEG_INF, POS_INF).astimezone(NY) == D(NEG_INF, POS_INF)  # no finite end: no zone to show
    with pytest.raises(TypeError):
        a.astimezone('UTC')


def test_tz_reads_a_date_in_a_zone():
    """a date is naive; `tz=` makes it that day in a zone (aware), and is the display zone"""
    day = D(JAN1, tz=SGT)
    assert day.tz is SGT and day.inf == dt(2024, 1, 1, tzinfo=SGT)
    assert day.inf_seconds == JAN1_SECONDS - 8 * 3600 and day.total_seconds == 86400
    assert D(date(2024, 3, 10), tz=NY).total_seconds == 23 * 3600  # an aware day is physical
    assert D(JAN1, dt(2024, 1, 2, 12, tzinfo=UTC), tz=SGT).sup == dt(2024, 1, 2, 20, tzinfo=SGT)
    with pytest.raises(TypeError):
        D(JAN1, tz='Asia/Singapore')


def test_nan_and_foreign_bounds_refused():
    """D30: nan (and any number) is no time value; v1 read nan as a missing end"""
    for bad in (float('nan'), 1.0, 0, '2024-01-01', Fraction(1), M(1), D(NOON)):
        with pytest.raises(TypeError):
            D(bad)
        with pytest.raises(TypeError):
            T(bad)
    with pytest.raises(TypeError):
        D(td(1))
    with pytest.raises(TypeError):
        T(NOON)
    with pytest.raises(ValueError):
        D(None, NOON)  # an end without a start, as MultiInterval; v1 read a None start as NaT, a point


@pytest.mark.parametrize('bad', ['no', None, 0, 1])
def test_flags_are_bools(bad):
    """as MultiInterval (owner, Q22(a), 2026-10-04), the empty set's flags too"""
    for make in (lambda: D(NOON, NOON + HOUR, start_closed=bad), lambda: D(JAN1, end_closed=bad, start_closed=bad),
                 lambda: D(start_closed=bad), lambda: T(-HOUR, HOUR, end_closed=bad), lambda: T(end_closed=bad)):
        with pytest.raises(TypeError, match='must be a bool'):
            make()


def test_empty_is_falsy():
    """as MultiInterval and python's sets; v1's time classes had no `__bool__`, so an empty one was True
    (the v1 parity audit, 2026-10-04: plan §4)"""
    assert not D() and not T() and not (D(NOON) & D(NOON + HOUR)) and not T(HOUR) - T(HOUR) & T()
    assert D(NOON) and T(td(0)) and D(NEG_INF, POS_INF)


def test_no_bounds_with_any_flags_is_empty():
    """`T(start_closed=False)` is the empty set, as `MultiInterval(start_closed=False)` (v1 raised; plan §4)"""
    for flags in ({'start_closed': False}, {'end_closed': False}, {'start_closed': False, 'end_closed': False}):
        assert D(**flags) == D() and T(**flags) == T() and MultiInterval(**flags) == MultiInterval()


def test_slice_step_is_a_type_error():
    """slicing restricts to a closed range; a step means nothing (v1 raised ValueError; plan §4)"""
    for a, lo, hi in ((D(NEG_INF, POS_INF), NOON, NOON + HOUR), (T(NEG_INF, POS_INF), -HOUR, HOUR)):
        with pytest.raises(TypeError, match='step'):
            a[lo:hi:HOUR]


def test_end_or_end_not_ported():
    """D30: v1 wrote `end or _end`, so a falsy end (the epoch, 0 s) was replaced; here it is an end"""
    a = D(date(1969, 12, 31), dt(1970, 1, 1))
    assert a.sup_seconds == 0 and a.sup == dt(1970, 1, 1) and a.sup_closed
    assert T(-HOUR, td(0)).sup_seconds == 0


# DATES: THE HALF-OPEN DAY (D30 (c))

def test_a_date_is_the_half_open_day():
    day = D(JAN1)
    assert day.inf == dt(2024, 1, 1) and day.inf_closed
    assert day.sup == dt(2024, 1, 2) and not day.sup_closed
    assert not day.is_degenerate


def test_a_days_size_is_86400():
    """D30 (c): a day is 86400 s, no point short (v1's snapped day was 86399.999999 s plus a point)"""
    assert D(JAN1).size == Size(0, 86400, 0)
    assert D(JAN1).total_duration == DAY
    assert D(JAN1, date(2024, 1, 7)).total_duration == 7 * DAY


def test_adjacent_days_merge_into_one_piece():
    """D30 (c): with the snap a gap of exact seconds stayed between two days"""
    two = D(JAN1) | D(date(2024, 1, 2))
    assert two.is_contiguous and len(two) == 1
    assert two == D(JAN1, date(2024, 1, 2))
    week = D().union(*(D(JAN1 + k * DAY) for k in range(7)))
    assert len(week) == 1 and week.total_duration == 7 * DAY
    # v1's wish: [Tue, Sat) == [Tue, Fri] == (Mon, Fri]
    tue, sat = date(2024, 1, 2), date(2024, 1, 6)
    assert D(tue, sat, end_closed=False) == D(tue, date(2024, 1, 5)) == D(JAN1, date(2024, 1, 5), start_closed=False)


def test_last_half_microsecond_is_in_its_day():
    """D30 (c): 23:59:59.9999995 is in the day (the snapped day ended at 23:59:59.999999)"""
    late = JAN1_SECONDS + 86400 - Fraction(1, 2 * 10 ** 6)
    assert D.from_seconds(late) in D(JAN1)
    assert D.from_seconds(late) not in D(date(2024, 1, 2))
    assert D.from_seconds(JAN1_SECONDS + 86400) not in D(JAN1)


@pytest.mark.parametrize('start_closed, end_closed, lo, hi', [
    (True, True, dt(2024, 1, 1), dt(2024, 1, 4)),   # Mon through Wed
    (True, False, dt(2024, 1, 1), dt(2024, 1, 3)),  # Mon, before Wed
    (False, True, dt(2024, 1, 2), dt(2024, 1, 4)),  # after Mon, through Wed
    (False, False, dt(2024, 1, 2), dt(2024, 1, 3)),  # after Mon, before Wed: Tue only
])
def test_date_flags_say_whether_the_day_is_in(start_closed, end_closed, lo, hi):
    a = D(JAN1, date(2024, 1, 3), start_closed=start_closed, end_closed=end_closed)
    assert (a.inf, a.inf_closed, a.sup, a.sup_closed) == (lo, True, hi, False)


def test_bounds_are_ordered_as_read():
    """the order of two bounds is checked on their readings (cuts), by MultiInterval's rule: a start read after
    the end raises, equal readings with an open flag are empty. the build checked the values as written (a date as
    its 00:00), so it refused noon-through-the-day and let after-the-2nd-until-its-noon be empty (review F3)"""
    tue, tue_noon = date(2024, 1, 2), dt(2024, 1, 2, 12)
    for reversed_ in (lambda: D(date(2024, 1, 3), JAN1), lambda: D(NOON, JAN1, end_closed=False),
                      lambda: D(POS_INF, NEG_INF),
                      lambda: D(tue, tue_noon, start_closed=False),  # after the 2nd, until its noon
                      lambda: D(JAN1, NOON, start_closed=False),
                      lambda: D(JAN1, start_closed=False, end_closed=False),  # after Mon, before Mon
                      lambda: D(JAN1, start_closed=False),  # half-open single bound, as MultiInterval
                      lambda: D(tue_noon, tue, end_closed=False)):  # noon, before the 2nd
        with pytest.raises(ValueError):
            reversed_()
    assert D(tue_noon, tue) == D(tue_noon, dt(2024, 1, 3), end_closed=False)  # noon, through the day
    assert D(JAN1, JAN1, start_closed=False) == D()  # after Mon, through Mon: equal readings, as M(1, 1, start_closed=False)
    assert D(tue, JAN1) == D()  # from Tue through Mon: [Tue 00:00, Tue 00:00), empty (the build raised: written reversed)
    assert D(JAN1, dt(2024, 1, 2), start_closed=False) == D(dt(2024, 1, 2))  # after Mon, through Tue 00:00
    week = D(JAN1, date(2024, 1, 7))
    assert week[tue_noon:tue] == D(tue_noon, dt(2024, 1, 3), end_closed=False)  # slicing reads the same way
    assert week[tue:JAN1] == D()
    with pytest.raises(ValueError):
        week[date(2024, 1, 3):JAN1]


@pytest.mark.parametrize('make', [
    lambda: T(HOUR, -HOUR), lambda: T(HOUR, start_closed=False), lambda: T(HOUR, end_closed=False),
    lambda: T(POS_INF, td(0)), lambda: T(NEG_INF, POS_INF)[HOUR:-HOUR],
])
def test_durations_reversed_or_half_open_single_raise(make):
    """as MultiInterval and DateTimeInterval (the sabotage review's R15, R16: pinned for D only)"""
    with pytest.raises(ValueError):
        make()


def test_a_datetime_is_an_instant_no_snap():
    """D30 (c): v1 stretched a datetime end at midnight to the end of its day, and 10:00 to 10:59:59.999999"""
    a = D(dt(2024, 1, 1, 8), dt(2024, 1, 2))
    assert a.sup == dt(2024, 1, 2) and a.sup_closed
    assert D(dt(2024, 1, 1, 8), dt(2024, 1, 1, 10)).sup == dt(2024, 1, 1, 10)
    assert D(dt(2024, 1, 2)).is_degenerate
    assert dt(2024, 1, 2, 0, 0, 1) not in a


def test_date_membership_is_the_whole_day():
    a = D(dt(2024, 1, 1, 8), dt(2024, 1, 3, 8))
    assert date(2024, 1, 2) in a
    assert JAN1 not in a and date(2024, 1, 3) not in a
    assert dt(2024, 1, 1, 9) in a and dt(2024, 1, 1, 7) not in a
    assert D(dt(2024, 1, 2, 3)) in a and a in D(JAN1, date(2024, 1, 3))


# THE INFINITE ENDS (D30 (b))

ORDERED_VALUES = [dt.min, dt.max, date.min, date.max, td.min, td.max, td(0), NOON, dt(2024, 1, 1, tzinfo=SGT)]


@pytest.mark.parametrize('value', ORDERED_VALUES, ids=repr)
def test_sentinels_order_against_every_type(value):
    """D30 (b): below / above every datetime, date and timedelta, both ways round (pandas: test_time_pandas)"""
    assert NEG_INF < value and value > NEG_INF and NEG_INF <= value and not NEG_INF >= value
    assert POS_INF > value and value < POS_INF and POS_INF >= value and not POS_INF <= value
    assert NEG_INF != value and POS_INF != value
    assert sorted([POS_INF, value, NEG_INF]) == [NEG_INF, value, POS_INF]
    assert max(value, POS_INF) is POS_INF and min(NEG_INF, value) is NEG_INF


def test_sentinels_order_against_each_other_and_nothing_else():
    assert NEG_INF < POS_INF and POS_INF > NEG_INF and NEG_INF <= NEG_INF and not NEG_INF < NEG_INF
    for x in (NEG_INF, POS_INF):  # each against itself (R13)
        assert x <= x and x >= x and not x < x and not x > x
    assert not POS_INF <= NEG_INF and not NEG_INF >= POS_INF
    assert NEG_INF == NEG_INF and NEG_INF != POS_INF
    assert -NEG_INF is POS_INF and -POS_INF is NEG_INF
    for foreign in (0, inf, -inf, 'x', None):
        with pytest.raises(TypeError):
            _ = NEG_INF < foreign
    assert NEG_INF != -inf and POS_INF != inf  # not math.inf, which does not order against a datetime


def test_sentinels_hash_print_pickle_and_copy():
    assert repr(NEG_INF) == str(NEG_INF) == '-inf' and repr(POS_INF) == str(POS_INF) == 'inf'
    assert len({NEG_INF, POS_INF, NEG_INF}) == 2
    for s in (NEG_INF, POS_INF):
        assert pickle.loads(pickle.dumps(s)) is s and copy.copy(s) is s and copy.deepcopy(s) is s
    with pytest.raises(AttributeError):
        NEG_INF._sign = 1


def test_infinite_ends_round_trip():
    """D30 (b): an infinite end reads out as a sentinel and the constructors take it back, closed or open as
    written; the storage is +-inf in the numeric class"""
    for cls, finite in ((D, NOON), (T, HOUR)):
        for a in (cls(NEG_INF, finite), cls(NEG_INF, finite, start_closed=False),
                  cls(finite, POS_INF, end_closed=False), cls(NEG_INF, POS_INF), cls(POS_INF)):
            assert cls(a.inf, a.sup, start_closed=a.inf_closed, end_closed=a.sup_closed) == a
            assert not a.is_finite
        assert cls(NEG_INF, finite).inf is NEG_INF and cls(NEG_INF, finite).inf_closed
        assert cls(NEG_INF, finite).seconds.inf == -inf
        assert cls(finite, POS_INF, end_closed=False).sup is POS_INF
        assert NEG_INF in cls(NEG_INF, finite) and NEG_INF not in cls(NEG_INF, finite, start_closed=False)
    assert D(NEG_INF, NOON).size == Size(1, JAN1_SECONDS + 12 * 3600, 1)


def test_open_ended_ranges():
    """`[-inf, t]` style ranges through set ops, relations and arithmetic"""
    until = D(NEG_INF, NOON)
    since = D(NOON, POS_INF, start_closed=False)
    assert until | since == D(NEG_INF, POS_INF)
    assert (until & since).is_empty and until.adjoins(since) and until.before(since)
    assert ~until == since
    assert until & D(JAN1) == D(dt(2024, 1, 1), NOON)
    assert until + HOUR == D(NEG_INF, NOON + HOUR)
    assert NOON - until == T(td(0), POS_INF)
    assert (until < dt(2025, 1, 1)) == TRUE and (until < NOON) == BOTH
    assert until.sup == NOON and until[dt(2023, 1, 1):].inf == dt(2023, 1, 1)
    assert D(NEG_INF, POS_INF).tz is None and D(NEG_INF, POS_INF) | D(NOON) == D(NEG_INF, POS_INF)


def test_kind_is_a_function_of_the_set():
    """a set with no finite end (empty, the infinities) is neither naive nor aware and meets either"""
    only_infinity = D(NEG_INF, NOON) & D(NEG_INF)
    assert only_infinity == D(NEG_INF) and only_infinity.tz is None
    aware = D(dt(2024, 1, 1, tzinfo=SGT))
    assert (only_infinity | aware).tz is SGT and (aware | only_infinity).tz is SGT
    assert (D(NOON) & D(JAN1 + 9 * DAY)) | aware == aware
    assert D(NEG_INF, POS_INF) == D(NEG_INF, POS_INF, tz=SGT) and D(NEG_INF, POS_INF, tz=SGT).tz is None


def test_missing_slice_bound_is_closed():
    """D30 (b): as MultiInterval.__getitem__ (v1 opened it)"""
    everything = D(NEG_INF, POS_INF)
    assert everything[:NOON] == D(NEG_INF, NOON) and everything[:NOON].inf_closed
    assert everything[NOON:] == D(NOON, POS_INF)
    assert everything[:] == everything
    assert everything[JAN1:JAN1] == D(JAN1)  # a date stop is through that day
    assert T(NEG_INF, POS_INF)[:td(0)] == T(NEG_INF, td(0))
    with pytest.raises(TypeError):
        everything[NOON]
    with pytest.raises(TypeError):
        everything[JAN1:NOON:2]


# READ-OUTS: NEVER ROUNDED (D30)

def test_non_microsecond_read_out_raises_and_the_raw_accessor_works():
    third = T(HOUR, 2 * HOUR) / 7  # 3600/7 s
    with pytest.raises(ValueError, match='inf_seconds'):
        third.inf
    with pytest.raises(ValueError, match='sup_seconds'):
        third.sup
    assert third.inf_seconds == Fraction(3600, 7) and third.sup_seconds == Fraction(7200, 7)
    assert third.seconds == M(Fraction(3600, 7), Fraction(7200, 7))
    with pytest.raises(ValueError, match='total_seconds'):
        third.total_duration
    assert third.total_seconds == Fraction(3600, 7)
    point = D.from_seconds(Fraction(1, 3))
    with pytest.raises(ValueError, match='degenerate_points'):
        point.degenerate_points
    assert point.seconds.degenerate_points == {Fraction(1, 3)}
    assert str(point) == '[1970-01-01 00:00:00.333333+1/3us]'  # str never raises
    assert eval(repr(point), EVAL_NAMESPACE) == point


def test_out_of_range_read_out_raises_overflow():
    late = D(dt.max) + DAY
    with pytest.raises(OverflowError, match='sup_seconds'):
        late.sup
    assert late.sup_seconds == D(dt.max).sup_seconds + 86400
    assert str(late).endswith(' s]') and eval(repr(late), EVAL_NAMESPACE) == late
    with pytest.raises(OverflowError):
        (T(td(0), td.max) * 2).total_duration
    with pytest.raises(ValueError, match='size'):
        D(NEG_INF, NOON).total_seconds  # unbounded


def test_float_factor_is_exact():
    """a float factor is its exact value, never rounded to a microsecond (python's `td * 0.1` rounds)"""
    tenth = T(DAY) * 0.1
    assert tenth.seconds == M(86400 * Fraction(0.1))
    with pytest.raises(ValueError):
        tenth.inf
    assert (T(DAY) * Fraction(1, 10)).inf == td(hours=2, minutes=24)
    assert (T(DAY) * M(0.5, 1)).seconds == M(43200, 86400)
    assert (T(DAY) * OutwardMultiInterval(0.1)).seconds == M(86400 * Fraction(0.1))


# EQUALITY, HASH, COMPARISONS (D30)

def test_comparisons_return_truthsets():
    a = D(NOON, NOON + 2 * HOUR)
    assert (a < NOON + 3 * HOUR) == TRUE
    assert (a < NOON + HOUR) == BOTH
    assert (a > NOON + 3 * HOUR) == FALSE
    assert (NOON - HOUR < a) == TRUE  # reflected through datetime's NotImplemented
    with pytest.raises(ValueError):
        bool(a < NOON + HOUR)
    assert (T(HOUR) <= T(HOUR, 2 * HOUR)) == TRUE
    assert a.eq_pointwise(NOON) == BOTH
    assert (D() < NOON) == NEITHER and isinstance(a < NOON, TruthSet)
    with pytest.raises(TypeError):
        _ = a < T(HOUR)
    with pytest.raises(TypeError):
        _ = a < 5


def test_foreign_equality_is_not_implemented_and_hashable():
    a = D(NOON)
    assert a.__eq__(NOON) is NotImplemented and a.__ne__(NOON) is NotImplemented
    assert a != NOON and not (a == NOON) and a != M(JAN1_SECONDS + 43200)
    assert T(HOUR).__eq__(HOUR) is NotImplemented and T(HOUR) != HOUR
    assert T(HOUR) != T(2 * HOUR) and not (T(HOUR) != T(HOUR)) and T(HOUR) == T(HOUR)  # R14
    assert D(NOON) != D(JAN1) and not (D(NOON) != D(NOON))
    assert D.from_seconds(3600) != T.from_seconds(3600)
    assert len({D(NOON), D(NOON), D(JAN1), T(HOUR), T(HOUR)}) == 3
    assert {D(dt(2024, 1, 1, 8, tzinfo=SGT)): 1}[D(dt(2024, 1, 1, tzinfo=UTC))] == 1


def test_immutable_and_pickles():
    a = D(dt(2024, 1, 1, 8, tzinfo=SGT), POS_INF)
    for x in (a, T(HOUR, 2 * HOUR), D(), D(NOON)):
        with pytest.raises(AttributeError):
            x._mi = M()
        y = pickle.loads(pickle.dumps(x))
        assert y == x and type(y) is type(x)
    assert pickle.loads(pickle.dumps(a)).tz == SGT


# REPR AND STR

REPR_CASES = [
    D(), D(JAN1), D(NOON), D(NEG_INF, NOON), D(NOON, POS_INF, start_closed=False), D(POS_INF), D(NEG_INF, POS_INF),
    D(dt(2024, 1, 1, 8, tzinfo=SGT), dt(2024, 1, 1, 9, tzinfo=NY)), D(JAN1, tz=UTC), D(JAN1) | D(NOON + 3 * DAY),
    D(dt(2024, 11, 3, 1, 30, fold=1, tzinfo=NY)), D(dt(2024, 1, 1, 0, 0, 0, 1), dt(2024, 1, 1, 0, 0, 1)),
    D(JAN1, tz=datetime.timezone(td(hours=-3))),
    D.from_seconds(M(Fraction(1, 3), 2)), D.from_seconds(M(Fraction(1, 3), 2), tz=SGT),
    T(), T(HOUR), T(-HOUR, 2 * HOUR, end_closed=False), T(NEG_INF, td(0)), T(DAY) / 7, T(td.max) * 2,
]


@pytest.mark.parametrize('x', REPR_CASES, ids=str)
def test_repr_round_trips(x):
    back = eval(repr(x), EVAL_NAMESPACE)
    assert back == x and type(back) is type(x) and back.seconds == x.seconds
    assert getattr(back, 'tz', None) == getattr(x, 'tz', None)


@settings(max_examples=60)
@given(us_cut_tuples, st.sampled_from([None, UTC, SGT, NY]))
def test_repr_round_trips_property(cuts, tz):
    for x in (D.from_seconds(M.from_cuts(cuts), tz=tz), T.from_seconds(quiet(M.__sub__, M.from_cuts(cuts), JAN1_SECONDS))):
        back = eval(repr(x), EVAL_NAMESPACE)
        assert back == x and getattr(back, 'tz', None) == getattr(x, 'tz', None)
        str(x)


@pytest.mark.parametrize('x, text', [
    (D(), '{}'),
    (D(JAN1), '[2024-01-01 00:00:00, 2024-01-02 00:00:00)'),
    (D(NEG_INF, NOON, start_closed=False), '(-inf, 2024-01-01 12:00:00]'),
    (D(NOON) | D(JAN1 + DAY), '{ [2024-01-01 12:00:00] , [2024-01-02 00:00:00, 2024-01-03 00:00:00) }'),
    (D(dt(2024, 1, 1, 8, tzinfo=SGT)), '[2024-01-01 08:00:00+08:00]'),
    (D.from_seconds(Fraction(1, 3), tz=SGT), '[1970-01-01 07:30:00.333333+1/3us+07:30]'),
    (T(-HOUR, 2 * DAY + HOUR), '[-1:00:00, 2 days 1:00:00]'),
    (T(td(microseconds=-1)) / 3, '[-0:00:00.000001+2/3us]'),
    (T(NEG_INF, POS_INF), '[-inf, inf]'),
])
def test_str(x, text):
    assert str(x) == text


# ARITHMETIC: THE TABLE

ARITHMETIC_TABLE = [
    # (expression, result class, seconds of the result)
    (lambda: D(NOON) + HOUR, D, M(JAN1_SECONDS + 46800)),
    (lambda: D(NOON) + T(td(0), HOUR), D, M(JAN1_SECONDS + 43200, JAN1_SECONDS + 46800)),
    (lambda: D(NOON) - HOUR, D, M(JAN1_SECONDS + 39600)),
    (lambda: D(NOON) - T(td(0), HOUR), D, M(JAN1_SECONDS + 39600, JAN1_SECONDS + 43200)),
    (lambda: NOON + T(HOUR), D, M(JAN1_SECONDS + 46800)),
    (lambda: NOON - T(HOUR), D, M(JAN1_SECONDS + 39600)),
    (lambda: JAN1 + T(HOUR), D, M(JAN1_SECONDS + 3600, JAN1_SECONDS + 3600 + 86400, end_closed=False)),
    (lambda: HOUR + D(NOON), D, M(JAN1_SECONDS + 46800)),
    (lambda: T(HOUR) + D(NOON), D, M(JAN1_SECONDS + 46800)),
    (lambda: T(HOUR) + NOON, D, M(JAN1_SECONDS + 46800)),
    (lambda: D(NOON) - D(JAN1), T, M(43200 - 86400, 43200, start_closed=False)),
    (lambda: D(NOON) - NOON, T, M(0)),
    (lambda: NOON - D(JAN1), T, M(43200 - 86400, 43200, start_closed=False)),
    (lambda: T(HOUR) + T(HOUR), T, M(7200)),
    (lambda: T(HOUR) + HOUR, T, M(7200)),
    (lambda: HOUR + T(HOUR), T, M(7200)),
    (lambda: T(HOUR) - 2 * HOUR, T, M(-3600)),
    (lambda: 2 * HOUR - T(HOUR), T, M(3600)),
    (lambda: T(HOUR) * 3, T, M(10800)),
    (lambda: 3 * T(HOUR), T, M(10800)),
    (lambda: Fraction(1, 2) * T(HOUR), T, M(1800)),
    (lambda: T(HOUR) * M(1, 2), T, M(3600, 7200)),
    (lambda: M(1, 2) * T(HOUR), T, M(3600, 7200)),
    (lambda: T(HOUR) / 3, T, M(1200)),
    (lambda: T(HOUR) / M(1, 2), T, M(1800, 3600)),
    (lambda: T(HOUR, 3 * HOUR) / HOUR, M, M(1, 3)),
    (lambda: T(HOUR, 3 * HOUR) / T(HOUR), M, M(1, 3)),
    (lambda: 3 * HOUR / T(HOUR), M, M(3)),
    (lambda: T(HOUR, 3 * HOUR) // HOUR, M, M.from_pieces([(1, 1), (2, 2), (3, 3)])),
    (lambda: 5 * HOUR // T(2 * HOUR), M, M(2)),
    (lambda: T(5 * HOUR) % (2 * HOUR), T, M(3600)),
    (lambda: T(-5 * HOUR) % T(2 * HOUR), T, M(3600)),
    (lambda: 5 * HOUR % T(2 * HOUR), T, M(3600)),
    (lambda: -T(HOUR, 2 * HOUR), T, M(-7200, -3600)),
    (lambda: +T(HOUR), T, M(3600)),
    (lambda: abs(T(-HOUR, HOUR / 2)), T, M(0, 3600)),
]


@pytest.mark.parametrize('expression, cls, seconds', ARITHMETIC_TABLE)
def test_arithmetic_table(expression, cls, seconds):
    result = expression()
    assert type(result) is cls
    assert (result if cls is M else result.seconds) == seconds


def test_arithmetic_matches_python_on_points():
    """each row of the table on points is python's own scalar op"""
    a, b, x = dt(2024, 1, 1, 9, 30), dt(2023, 6, 1, 1, 2, 3, 4), td(hours=5, microseconds=7)
    y = td(minutes=-17, seconds=3)
    assert (D(a) + x).inf == a + x and (D(a) - x).inf == a - x and (x + D(a)).inf == x + a
    assert (D(a) - D(b)).inf == a - b
    assert (T(x) + T(y)).inf == x + y and (T(x) - T(y)).inf == x - y
    assert (T(x) * 4).inf == x * 4 and (T(x) / 4).inf_seconds == seconds_of_timedelta(x) / 4
    assert (T(x) / T(y)).inf == Fraction(seconds_of_timedelta(x), seconds_of_timedelta(y))
    assert (T(x) // T(y)).inf == x // y and (T(x) % T(y)).inf == x % y
    q, r = divmod(T(x), T(y))
    assert (q.inf, r.inf) == divmod(x, y)
    assert divmod(x, T(y))[1].inf == x % y


@pytest.mark.parametrize('expression', [
    lambda: D(NOON) + D(NOON),
    lambda: D(NOON) + NOON,
    lambda: NOON + D(NOON),
    lambda: D(NOON) * 2,
    lambda: 2 * D(NOON),
    lambda: D(NOON) / 2,
    lambda: -D(NOON),
    lambda: abs(D(NOON)),
    lambda: D(NOON) + 3600,
    lambda: D(NOON) - 3600,
    lambda: D(NOON) + M(1),
    lambda: HOUR - D(NOON),
    lambda: T(HOUR) - D(NOON),
    lambda: T(HOUR) - NOON,
    lambda: T(HOUR) * T(HOUR),
    lambda: T(HOUR) * HOUR,
    lambda: T(HOUR) * True,
    lambda: T(HOUR) * '2',
    lambda: 2 / T(HOUR),
    lambda: T(HOUR) // 2,
    lambda: T(HOUR) % 2,
    lambda: 2 % T(HOUR),
    lambda: T(HOUR) + 1,
    lambda: T(HOUR) - 1,
    lambda: 1 - T(HOUR),
    lambda: 1 + T(HOUR),
    lambda: 3600 - D(NOON),
    lambda: 3600 + D(NOON),
    lambda: T(HOUR) - M(1),
    lambda: M(1) - T(HOUR),
    lambda: M(1) + D(NOON),
    lambda: D(NOON) - M(1),
    lambda: T(HOUR) ** 2,
    lambda: D(NOON) + NEG_INF,
    lambda: T(HOUR) + POS_INF,
    lambda: D(NOON) - POS_INF,
    lambda: D(NOON) | T(HOUR),
    lambda: D(NOON) | M(1),
    lambda: M(1) & T(HOUR),
    lambda: D(NOON) + T(HOUR) * T(HOUR),
])
def test_arithmetic_outside_the_table_is_a_type_error(expression):
    with pytest.raises(TypeError):
        expression()


def test_arithmetic_passes_the_numeric_warnings_through():
    with pytest.warns(IndeterminateResultWarning):
        assert T(HOUR) / T(td(0)) == M()


def test_numpy_scalars_meet_the_operators():
    np = pytest.importorskip('numpy')  # numpy is in the [test] extra, so this runs in the gate
    assert np.float64(2) * T(HOUR) == T(2 * HOUR)
    assert T(HOUR) * np.int64(3) == T(3 * HOUR)
    assert T(HOUR) / np.float32(0.5) == T(2 * HOUR)


def test_numpy_times_are_refused():
    """numpy registers `timedelta64` as `numbers.Integral`, so `_factor` took it as a number in its own unit:
    `X * np.timedelta64(3, 'ns')` was `X * 3`, `X / np.timedelta64(3, 'ns')` a duration (the review's F6). every
    place the time layer takes a scalar refuses numpy's timedelta64 and datetime64 (TypeError)"""
    np = pytest.importorskip('numpy')
    x = T(2 * HOUR, 4 * HOUR)
    for t in (np.timedelta64(3, 'ns'), np.timedelta64(1, 'h'), np.timedelta64(3, 'Y'), np.datetime64('2024-01-01')):
        for thunk in (lambda: x * t, lambda: t * x, lambda: x / t, lambda: t / x, lambda: x // t, lambda: t // x,
                      lambda: x % t, lambda: t % x, lambda: divmod(x, t), lambda: x + t, lambda: t + x, lambda: x - t,
                      lambda: t - x, lambda: D(NOON) + t, lambda: t + D(NOON), lambda: D(NOON) - t, lambda: t - D(NOON),
                      lambda: D(t), lambda: T(t), lambda: D(NOON, t), lambda: T.from_seconds(t),
                      lambda: D.from_seconds(t), lambda: x.expand(t), lambda: x | t, lambda: t in x, lambda: x < t,
                      lambda: NEG_INF < t, lambda: POS_INF > t, lambda: t < POS_INF, lambda: t > NEG_INF,
                      lambda: NEG_INF <= t, lambda: t >= NEG_INF):
            with pytest.raises(TypeError):
                thunk()


# HOMOMORPHISM ONTO THE NUMERIC CLASS (properties)

SET_OPS = ['__or__', '__and__', '__xor__', 'difference', 'union', 'intersection', 'symmetric_difference']


@settings(max_examples=50)
@given(datetime_set_pairs, st.sampled_from(SET_OPS))
def test_set_ops_are_the_numeric_ones(pair, name):
    """on naive and aware sets (each aware one in its own zone): the seconds are the numeric op's, the kind is
    kept and the display zone is the first operand's that has one"""
    a, b = pair
    result = getattr(D, name)(a, b)
    assert type(result) is D and result.seconds == getattr(M, name)(a.seconds, b.seconds)
    assert result.tz is display_zone(result, a, b)
    durations = getattr(T, name)(T.from_seconds(a.seconds), T.from_seconds(b.seconds))
    assert type(durations) is T and durations.seconds == result.seconds
    assert (~a).seconds == ~a.seconds and a.hull.seconds == a.seconds.hull
    assert a.interior.seconds == a.seconds.interior and a.closed_hull.seconds == a.seconds.closed_hull
    assert [p.seconds for p in a] == list(a.seconds) and len(a) == len(a.seconds)
    for x in (~a, a.hull, a.interior, a.closed_hull, *a):  # unary results keep the kind and zone (R25, R26)
        assert type(x) is D and x.tz is display_zone(x, a)
        if has_finite_end(x) and has_finite_end(a):
            assert x == D.from_seconds(x.seconds, tz=a.tz)


RELATIONS = ['issubset', 'issuperset', 'isdisjoint', 'before', 'after', 'adjoins', 'overlaps', 'contains', 'within',
             'weakly_less', 'strictly_less', 'allen_relations', 'allen_matrix', 'eq_pointwise']


@settings(max_examples=50)
@given(datetime_set_pairs)
def test_relations_and_comparisons_are_the_numeric_ones(pair):
    """naive and aware sets, the aware ones in different zones (the review's F8)"""
    a, b = pair
    for name in RELATIONS:
        assert getattr(a, name)(b) == getattr(a.seconds, name)(b.seconds), name
    for op in (operator.lt, operator.le, operator.gt, operator.ge):
        assert op(a, b) == op(a.seconds, b.seconds)
    assert (b in a) == (b.seconds in a.seconds)
    assert (a == b) == (a.seconds == b.seconds)
    assert (hash(a) == hash(b)) or a != b
    if a.is_contiguous and b.is_contiguous:
        assert a.allen(b) == a.seconds.allen(b.seconds)


@settings(max_examples=50)
@given(st.booleans().flatmap(datetime_sets), duration_sets, duration_sets, factors)
def test_arithmetic_is_the_numeric_one(a, x, y, f):
    exact_f = Fraction(f) if isinstance(f, float) and math.isfinite(f) else f
    cases = [
        (lambda: a + x, D, lambda: a.seconds + x.seconds),
        (lambda: x + a, D, lambda: x.seconds + a.seconds),
        (lambda: a - x, D, lambda: a.seconds - x.seconds),
        (lambda: a - a, T, lambda: a.seconds - a.seconds),
        (lambda: x + y, T, lambda: x.seconds + y.seconds),
        (lambda: x - y, T, lambda: x.seconds - y.seconds),
        (lambda: x * f, T, lambda: x.seconds * exact_f),
        (lambda: f * x, T, lambda: exact_f * x.seconds),
        (lambda: x / f, T, lambda: x.seconds / exact_f),
        (lambda: x / y, M, lambda: x.seconds / y.seconds),
        (lambda: x // y, M, lambda: x.seconds // y.seconds),
        (lambda: x % y, T, lambda: x.seconds % y.seconds),
        (lambda: -x, T, lambda: -x.seconds),
        (lambda: abs(x), T, lambda: abs(x.seconds)),
    ]
    for wrapped, cls, numeric in cases:
        result, expected = quiet(wrapped), quiet(numeric)
        assert type(result) is cls
        assert (result if cls is M else result.seconds) == expected
        if cls is D:
            assert result.tz is display_zone(result, a)


# ROUND TRIPS AND ZONE INVARIANCE (properties)

@settings(max_examples=100)
@given(st.datetimes())
def test_naive_datetime_round_trips_over_the_whole_range(d):
    a = D(d)
    assert a.inf_seconds == seconds_of_naive(d) and a.inf == d and a.sup == d
    assert D.from_seconds(a.inf_seconds).inf == d
    assert D(d.date()).inf == dt.combine(d.date(), datetime.time())


@settings(max_examples=100)
@given(st.one_of(st.datetimes(timezones=st.timezones()),
                 st.datetimes(timezones=st.timezones(), min_value=dt(9999, 12, 30)),
                 st.datetimes(timezones=st.timezones(), max_value=dt(1, 1, 2)),
                 st.datetimes(timezones=st.sampled_from([datetime.timezone(td(hours=h)) for h in (-23, -5, 5, 23)]))))
@example(dt(1986, 4, 27, 2, 0, tzinfo=zoneinfo.ZoneInfo('America/Inuvik')))  # in a DST gap: fuzz run 37213772177
def test_aware_datetime_round_trips(d):
    """over datetime's whole range: an aware datetime near the ends reads back although its UTC instant has no
    datetime (`9999-12-31 23:00-05:00`; the review's F2: the build bounded this property to 1-01-02..9999-12-30)"""
    a = D(d)
    assert a.inf_seconds == seconds_of_aware(d)
    assert a.inf.tzinfo is d.tzinfo and a.tz is d.tzinfo
    assert a.inf.replace(tzinfo=None) == wall_of_instant(d) and seconds_of_aware(a.inf) == seconds_of_aware(d)
    assert D(a.inf) == a and D.from_seconds(a.seconds, tz=d.tzinfo) == a and a.sup == a.inf
    assert eval(repr(a), EVAL_NAMESPACE) == a


@settings(max_examples=100)
@given(st.datetimes(min_value=dt(1, 1, 2), max_value=dt(9999, 12, 30), timezones=st.timezones()),
       st.datetimes(min_value=dt(1, 1, 2), max_value=dt(9999, 12, 30), timezones=st.timezones()),
       st.timezones())
def test_aware_readings_are_zone_invariant(d, e, zone):
    lo, hi = sorted((d, e), key=seconds_of_aware)
    a = D(lo, hi)
    moved = D(lo.astimezone(zone), hi.astimezone(zone))
    assert moved == a and moved.seconds == a.seconds and hash(moved) == hash(a)
    assert a.astimezone(zone) == a and a.astimezone(zone).tz is zone
    assert (D(lo) < D(hi)) == (D(lo.astimezone(zone)) < D(hi.astimezone(UTC)))
    # the elapsed time between the instants (python's own `-` ignores a shared tzinfo: the wall-clock difference)
    assert (D(hi) - D(lo)).inf == hi.astimezone(UTC) - lo.astimezone(UTC)


@settings(max_examples=100)
@given(st.timedeltas())
def test_timedelta_round_trips_over_the_whole_range(x):
    a = T(x)
    assert a.inf_seconds == seconds_of_timedelta(x) and a.inf == x
    assert T.from_seconds(a.inf_seconds).inf == x


# PANDAS STAYS OPTIONAL

def test_pandas_is_never_imported():
    """the library imports pandas only for to_pandas(): reading datetimes, timedeltas, ops and str do not"""
    code = '\n'.join([
        'import sys, datetime',
        'import intervals',
        'from intervals import DateTimeInterval as D, TimeDeltaInterval as T, NEG_INF',
        'a = D(datetime.date(2024, 1, 1)) | D(NEG_INF, datetime.datetime(2023, 1, 1))',
        't = T(datetime.timedelta(1)) / 3',
        'r = (repr(a), str(a), repr(t), str(t), a - datetime.datetime(2024, 1, 1), NEG_INF < datetime.date.min)',
        'assert "pandas" not in sys.modules, sorted(m for m in sys.modules if "pandas" in m)',
    ])
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


# THE REVIEW ROUND (M8's three reviews, 2026-10-04): each pin names its finding

def test_degenerate_points_keep_both_instants_of_a_fold():
    """a tuple sorted by instant, not a set: python compares and hashes two datetimes of one zone by wall clock,
    so the fold pair 01:30 EDT / 01:30 EST was one element of a set (review F7)"""
    a0 = dt(2024, 11, 3, 1, 30, tzinfo=NY)
    a1 = a0.replace(fold=1)
    points = (D(a1) | D(a0)).degenerate_points
    assert type(points) is tuple and len(points) == 2
    assert [p.fold for p in points] == [0, 1] and seconds_of_aware(points[1]) - seconds_of_aware(points[0]) == 3600
    assert len(set(points)) == 1  # why it is not a set


def test_degenerate_points_values():
    """R7: their values, sorted, for both classes and the sentinels (the build tested only the raise)"""
    a = D(NOON) | D(JAN1 + 2 * DAY) | D(NOON - DAY) | D(dt(2024, 1, 5), dt(2024, 1, 6))
    assert a.degenerate_points == (NOON - DAY, NOON)
    assert (T(HOUR) | T(-HOUR) | T(2 * HOUR, 3 * HOUR)).degenerate_points == (-HOUR, HOUR)
    assert (D(POS_INF) | D(NEG_INF) | D(NOON)).degenerate_points == (NEG_INF, NOON, POS_INF)
    assert D(JAN1).degenerate_points == () and D().degenerate_points == ()
    assert D(dt(2024, 1, 1, 8, tzinfo=SGT)).degenerate_points == (dt(2024, 1, 1, 8, tzinfo=SGT),)


def test_fold_is_read_in():
    """R3: an aware datetime with fold=1 is the second instant of its wall time (NY: 3600 s later)"""
    a0 = dt(2024, 11, 3, 1, 30, tzinfo=NY)
    assert D(a0.replace(fold=1)).inf_seconds - D(a0).inf_seconds == 3600
    assert D(a0, a0.replace(fold=1)).total_seconds == 3600
    assert D(a0.replace(fold=1)).inf.fold == 1 and D(a0).inf.fold == 0


@pytest.mark.parametrize('d', [
    dt(9999, 12, 31, 23, tzinfo=datetime.timezone(td(hours=-5))),
    dt(9999, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.timezone(-td(hours=23, minutes=59))),
    dt(9999, 12, 31, 20, tzinfo=NY),
    dt(1, 1, 1, 1, tzinfo=datetime.timezone(td(hours=5))),
    dt(1, 1, 1, tzinfo=zoneinfo.ZoneInfo('Asia/Tokyo')),
    dt(1, 1, 1, tzinfo=datetime.timezone(td(hours=23, minutes=59))),
], ids=str)
def test_aware_ends_near_the_range_read_back(d):
    """review F2: `_from_us` built the UTC datetime before `astimezone`, so these raised OverflowError"""
    a = D(d)
    assert a.inf == d and a.inf.tzinfo is d.tzinfo and a.sup == d
    assert str(a) == f'[{d.isoformat(sep=" ")}]' and eval(repr(a), EVAL_NAMESPACE) == a
    assert a.degenerate_points == (d,)


def test_an_error_only_when_the_local_datetime_is_past_the_range():
    late = D(dt(9999, 12, 31, 23, tzinfo=datetime.timezone(td(hours=-5)))) + HOUR  # 10000-01-01 00:00-05:00
    with pytest.raises(OverflowError, match='sup_seconds'):
        late.sup
    assert str(late).endswith(' s]') and eval(repr(late), EVAL_NAMESPACE) == late
    early = D(dt(1, 1, 1, tzinfo=UTC)) - td(microseconds=1)
    with pytest.raises(OverflowError, match='inf_seconds'):
        early.inf


class _NoDst(datetime.tzinfo):
    """a tzinfo whose dst() is None, as the protocol allows; `astimezone` (fromutc) refuses it"""

    def utcoffset(self, d):
        return td(hours=1)

    def dst(self, d):
        return None

    def tzname(self, d):
        return 'NODST'


def test_a_wall_time_in_a_dst_gap_reads_as_its_instant():
    """a wall time a DST gap skips is the instant python's `utcoffset()` gives it (fold=0: the offset before the
    gap), read back as that instant's real wall time: `02:30` on New York's spring-forward night is 07:30 UTC, 03:30"""
    gap = dt(2024, 3, 10, 2, 30, tzinfo=NY)
    a = D(gap)
    assert a.inf_seconds == seconds_of_aware(gap) == seconds_of_aware(dt(2024, 3, 10, 7, 30, tzinfo=UTC))
    assert a.inf.replace(tzinfo=None) == dt(2024, 3, 10, 3, 30) and a.inf.utcoffset() == -4 * HOUR
    assert a == D(dt(2024, 3, 10, 3, 30, tzinfo=NY)) and gap in a


def test_a_tzinfo_without_dst_reads_out():
    """review F5: the read-outs and `str` raised `ValueError: fromutc: non-None dst() result required`"""
    z = _NoDst()
    d = dt(2024, 1, 1, tzinfo=z)
    a = D(d, d + HOUR)
    assert a.inf_seconds == JAN1_SECONDS - 3600
    assert a.inf == d and a.inf.tzinfo is z and a.sup == d + HOUR
    assert str(a) == '[2024-01-01 00:00:00+01:00, 2024-01-01 01:00:00+01:00]'
    assert a.degenerate_points == () and D(d).degenerate_points == (d,)


def test_aware_plus_a_timedelta_adds_elapsed_time():
    """review F4: aware `dt + td` adds elapsed time (instants), as `dt - dt` gives it; python's aware `+` is wall
    clock. NY springs forward on 2024-03-10, so a day after noon on the 9th is 13:00 on the 10th"""
    t = dt(2024, 3, 9, 12, tzinfo=NY)
    assert (D(t) + DAY).inf == dt(2024, 3, 10, 13, tzinfo=NY) != t + DAY
    assert (D(t) + DAY).inf_seconds - D(t).inf_seconds == 86400
    assert (D(t) + DAY) - D(t) == T(DAY)
    assert (D(dt(2024, 3, 9, 12)) + DAY).inf == dt(2024, 3, 10, 12)  # naive: wall clock, as python
    assert D(date(2024, 3, 9), tz=NY) + DAY != D(date(2024, 3, 10), tz=NY)  # not the next day in tz


@pytest.mark.parametrize('name', RELATIONS + ['allen', 'union', 'intersection', 'difference', 'symmetric_difference',
                                              '__or__', '__and__', '__xor__', '__lt__', '__le__', '__gt__', '__ge__'])
def test_naive_and_aware_never_mix_in_any_relation(name):
    """R10: the mixing check, pinned for `before` only among the relations"""
    naive, aware = D(NOON), D(dt(2024, 1, 2, tzinfo=SGT))
    for a, b in ((naive, aware), (aware, naive), (naive, dt(2024, 1, 2, tzinfo=SGT)), (aware, NOON), (aware, JAN1)):
        with pytest.raises(TypeError):
            getattr(a, name)(b)


@pytest.mark.parametrize('start_closed, end_closed, lo, hi', [
    (True, True, dt(2024, 3, 9, tzinfo=NY), dt(2024, 3, 12, tzinfo=NY)),
    (True, False, dt(2024, 3, 9, tzinfo=NY), dt(2024, 3, 11, tzinfo=NY)),
    (False, True, dt(2024, 3, 10, tzinfo=NY), dt(2024, 3, 12, tzinfo=NY)),
    (False, False, dt(2024, 3, 10, tzinfo=NY), dt(2024, 3, 11, tzinfo=NY)),  # after the 9th, before the 11th: the 10th
])
def test_tz_date_flags(start_closed, end_closed, lo, hi):
    """R11: `tz=` with an open start or end (pinned for closed/closed only); the 10th is 23 h in NY"""
    a = D(date(2024, 3, 9), date(2024, 3, 11), start_closed=start_closed, end_closed=end_closed, tz=NY)
    assert (a.inf, a.inf_closed, a.sup, a.sup_closed) == (lo, True, hi, False) and a.tz is NY
    assert a.total_seconds == seconds_of_aware(hi) - seconds_of_aware(lo)


def test_tz_date_at_the_end_of_the_range():
    """R2: date.max has no next date to name, so its day ends 86400 s after it starts"""
    for tz in (UTC, SGT, datetime.timezone(td(hours=-5))):
        last = D(date.max, tz=tz)
        assert last.total_seconds == 86400 and last.inf == dt(9999, 12, 31, tzinfo=tz)
        assert last.sup_seconds == seconds_of_aware(dt(9999, 12, 31, tzinfo=tz)) + 86400
        with pytest.raises(OverflowError, match='sup_seconds'):
            last.sup  # 10000-01-01 00:00 has no datetime


def test_reflected_set_ops_keep_the_left_zone():
    """R5: a scalar on the left of `| & ^` is the left operand: its zone is the display zone"""
    left, right = dt(2024, 1, 1, 20, tzinfo=SGT), D(dt(2024, 1, 1, tzinfo=NY), dt(2024, 1, 2, tzinfo=NY))
    for result in (left | right, left & right, left ^ right):  # 20:00 SGT is 07:00 NY, inside: all three non-empty
        assert result and result.tz is SGT
    for result in (right | left, right & left, right ^ left):
        assert result.tz is NY


def test_expand():
    """R6: `expand` had no test at all"""
    assert D(NOON).expand(HOUR) == D(NOON - HOUR, NOON + HOUR)
    assert (D(NOON) | D(NOON + 2 * HOUR)).expand(HOUR) == D(NOON - HOUR, NOON + 3 * HOUR)  # the pieces meet
    assert T(td(0), HOUR).expand(td(minutes=30)) == T(-td(minutes=30), td(minutes=90))
    assert D(NOON, POS_INF).expand(HOUR) == D(NOON - HOUR, POS_INF)
    aware = D(dt(2024, 1, 1, 8, tzinfo=SGT)).expand(HOUR)
    assert aware.tz is SGT and aware.total_seconds == 7200
    assert D(NOON).expand(td(0)) == D(NOON) and D().expand(HOUR) == D()
    with pytest.raises(TypeError):
        D(NOON).expand(3600)
    with pytest.raises(ValueError):
        D(NOON).expand(-HOUR)


def test_from_seconds_takes_a_number_exactly_and_a_tzinfo_only():
    """R12: a float number is its exact value; R23: tz must be a tzinfo"""
    assert D.from_seconds(0.1).inf_seconds == Fraction(0.1) != Fraction(1, 10)
    assert T.from_seconds(0.1).inf_seconds == Fraction(0.1)
    # a float compares equal to its exact Fraction, so check the type and an op that would round a float
    assert type(D.from_seconds(0.1).inf_seconds) is Fraction and type(T.from_seconds(0.1).inf_seconds) is Fraction
    assert (T.from_seconds(0.1) * 3).inf_seconds == 3 * Fraction(0.1) != 0.1 * 3
    assert D.from_seconds(Fraction(1, 10)).inf_seconds == Fraction(1, 10) and D.from_seconds(3).inf_seconds == 3
    for bad in ('UTC', 8, SGT.key):
        with pytest.raises(TypeError):
            D.from_seconds(0, tz=bad)
    for bad in (True, '1', td(1), None):
        with pytest.raises(TypeError):
            T.from_seconds(bad)


def test_reflected_divmod():
    """R17: the quotient of `divmod(timedelta, T)` (only the remainder was checked)"""
    q, r = divmod(5 * HOUR, T(2 * HOUR))
    assert q == M(2) and r == T(HOUR)
    x = T(2 * HOUR, 3 * HOUR)
    q, r = divmod(5 * HOUR, x)
    assert q == (5 * HOUR) // x == M.from_pieces([(1, 1), (2, 2)]) and r == (5 * HOUR) % x


def test_arithmetic_keeps_the_aware_zone():
    """R18: the display zone through every aware arithmetic row (all the table's rows are naive)"""
    t = dt(2024, 1, 1, 8, tzinfo=SGT)
    for result in (T(HOUR) + t, HOUR + D(t), D(t) + HOUR, D(t) - HOUR, t - T(HOUR), t + T(HOUR), D(t) + T(HOUR),
                   T(HOUR) + D(t), D(t) - T(HOUR)):
        assert type(result) is D and result.tz is SGT and result.inf.tzinfo is SGT
    assert (D(t) - dt(2024, 1, 1, tzinfo=UTC)) == T(td(0))


def test_astimezone_of_a_set_with_no_finite_end():
    """a set with no finite end has no kind and no zone: `astimezone` gives it back, its tz stays None"""
    for x in (D(), D(NEG_INF, POS_INF), D(POS_INF)):
        assert x.astimezone(SGT) is x and x.astimezone(SGT).tz is None


TZ_ZONES = [UTC, SGT, NY, zoneinfo.ZoneInfo('America/Havana'), zoneinfo.ZoneInfo('America/Santiago'),
            zoneinfo.ZoneInfo('Asia/Beirut'), zoneinfo.ZoneInfo('Australia/Lord_Howe'), zoneinfo.ZoneInfo('Pacific/Apia'),
            datetime.timezone(td(hours=-3, minutes=-30, seconds=-17))]


def midnight_seconds(day: date, tz) -> Fraction:
    """the oracle: the wall clock of `day 00:00` less the zone's offset there"""
    return seconds_of_aware(dt.combine(day, datetime.time(), tzinfo=tz))


@settings(max_examples=60)
@given(st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31)), st.integers(0, 3), st.booleans(),
       st.booleans(), st.sampled_from(TZ_ZONES))
def test_tz_dates_property(first, days, start_closed, end_closed, tz):
    """review F8: a `tz=` date range against midnights read by an independent oracle, every flag, zones with
    midnight DST changes (Havana, Santiago, Beirut), a 30-minute DST (Lord_Howe), a skipped day (Apia 2011)"""
    last = first + td(days)
    lo = midnight_seconds(first if start_closed else first + DAY, tz)
    hi = midnight_seconds(last + DAY if end_closed else last, tz)
    if lo > hi:
        with pytest.raises(ValueError):
            D(first, last, start_closed=start_closed, end_closed=end_closed, tz=tz)
        return
    a = D(first, last, start_closed=start_closed, end_closed=end_closed, tz=tz)
    assert a.seconds == M(lo, hi, end_closed=False)
    assert a.tz is (tz if a else None)
    if a:
        assert a.inf.tzinfo is tz and seconds_of_aware(a.inf) == lo
