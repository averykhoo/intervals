"""
the time layer with pandas (M8, D30): exact readings of `Timestamp` and `Timedelta` in their own unit, `NaT`
refused, the sentinels against pandas' types, and the `pd.Interval` round trips. pandas is in the `[test]` extra
(pyproject.toml), so nothing here skips; the library itself never imports it to read a value
(`tests/test_time_interval.py::test_pandas_is_never_imported`).
"""
import datetime
import operator
import warnings
import zoneinfo
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import NEG_INF
from multiinterval import POS_INF
from multiinterval import DateTimeInterval as D
from multiinterval import MultiInterval as M
from multiinterval import TimeDeltaInterval as T

dt, td = datetime.datetime, datetime.timedelta
SGT = zoneinfo.ZoneInfo('Asia/Singapore')
JAN1_SECONDS = 1704067200
NS = Fraction(1, 10 ** 9)


# READINGS

def test_timestamp_read_exactly_in_its_unit():
    """D30 (a): a naive Timestamp is wall clock (pandas' own reading), read in its unit, nanoseconds kept"""
    t = pd.Timestamp('2024-01-01 00:00:00.000000001')
    assert t.unit == 'ns'
    assert D(t).inf_seconds == JAN1_SECONDS + NS
    assert D(pd.Timestamp('2024-01-01')).inf_seconds == JAN1_SECONDS  # unit us in pandas 3
    assert D(pd.Timestamp('2024-01-01').as_unit('s')).inf_seconds == JAN1_SECONDS
    assert D(pd.Timestamp('2024-01-01')) == D(dt(2024, 1, 1))


def test_timestamp_past_the_nanosecond_range():
    """`.value` is nanoseconds and overflows past 2262; the reading takes the Timestamp's own unit"""
    t = pd.Timestamp('9999-12-31')
    with pytest.raises(OverflowError):
        t.value
    assert D(t).inf == dt(9999, 12, 31)


def test_aware_timestamp_is_utc():
    t = pd.Timestamp('2024-01-01 08:00', tz='Asia/Singapore')
    assert D(t).inf_seconds == JAN1_SECONDS
    assert D(t) == D(pd.Timestamp('2024-01-01', tz='UTC')) == D(dt(2024, 1, 1, 8, tzinfo=SGT))
    assert D(t).tz == t.tzinfo
    with pytest.raises(TypeError):
        D(t, pd.Timestamp('2024-01-02'))


def test_timedelta_read_exactly():
    assert T(pd.Timedelta(5, 'ns')).inf_seconds == 5 * NS
    assert T(pd.Timedelta(days=3)).inf == td(days=3)
    assert T(pd.Timedelta(td(days=10 ** 6, microseconds=1))).inf == td(days=10 ** 6, microseconds=1)


def test_nat_refused():
    """D30: NaT raises (v1 dropped it: `DateTimeInterval(NaT, t)` was the point t)"""
    t = pd.Timestamp('2024-01-01')
    for make in (lambda: D(pd.NaT), lambda: D(pd.NaT, t), lambda: D(t, pd.NaT), lambda: T(pd.NaT),
                 lambda: T(pd.NaT, pd.Timedelta(1)), lambda: D(t) | pd.NaT, lambda: pd.NaT in D(t),
                 lambda: T(pd.Timedelta(1)) | pd.NaT):
        with pytest.raises(ValueError, match='NaT'):
            make()


def test_nanosecond_end_in_its_day():
    """D30 (c): 23:59:59.9999995 is in its day; here from a Timestamp with nanoseconds"""
    late = pd.Timestamp('2024-01-01 23:59:59.999999500')
    assert late in D(datetime.date(2024, 1, 1))
    assert late not in D(datetime.date(2024, 1, 2))
    with pytest.raises(ValueError, match='inf_seconds'):
        D(late).inf  # not a whole microsecond: no datetime
    assert D(late).inf_seconds == JAN1_SECONDS + 86400 - Fraction(1, 2 * 10 ** 6)


# THE SENTINELS AGAINST PANDAS' TYPES (D30 (b))

@pytest.mark.parametrize('value', [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01', tz='UTC'),
                                   pd.Timestamp.min, pd.Timestamp.max, pd.Timedelta(0), pd.Timedelta.min,
                                   pd.Timedelta.max], ids=repr)
def test_sentinels_order_against_pandas(value):
    assert NEG_INF < value and value > NEG_INF and NEG_INF <= value and not NEG_INF > value
    assert POS_INF > value and value < POS_INF and POS_INF >= value and not POS_INF < value
    assert sorted([POS_INF, value, NEG_INF]) == [NEG_INF, value, POS_INF]


def test_nat_does_not_order():
    for s in (NEG_INF, POS_INF):
        assert not (s < pd.NaT) and not (s > pd.NaT) and not (s <= pd.NaT) and not (s >= pd.NaT)
        assert s != pd.NaT


# ARITHMETIC WITH PANDAS SCALARS

def test_pandas_scalars_in_the_table():
    t, h = pd.Timestamp('2024-01-01 12:00'), pd.Timedelta(hours=1)
    assert t + T(h) == D(dt(2024, 1, 1, 13)) and T(h) + t == D(dt(2024, 1, 1, 13))
    assert h + D(t) == D(dt(2024, 1, 1, 13)) and D(t) - h == D(dt(2024, 1, 1, 11))
    assert t - D(dt(2024, 1, 1)) == T(td(hours=12)) and D(t) - t == T(td(0))
    assert pd.Timedelta(days=1) / T(h) == M(24) and T(h) * 2 == T(2 * h)
    assert (pd.Timestamp('2024-01-01 00:00:00.000000001') - D(dt(2024, 1, 1))).inf_seconds == NS


# THE pd.Interval ROUND TRIPS

INTERVALS = [
    pd.Interval(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'), closed='left'),
    pd.Interval(pd.Timestamp('2024-01-01 00:00:00.000000001'), pd.Timestamp('2024-01-02'), closed='both'),
    pd.Interval(pd.Timestamp('2024-01-01', tz='Asia/Singapore'), pd.Timestamp('2024-01-03', tz='Asia/Singapore'),
                closed='neither'),
    pd.Interval(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01'), closed='both'),
    pd.Interval(pd.Timedelta(5, 'ns'), pd.Timedelta(hours=1), closed='right'),
    pd.Interval(pd.Timedelta(-1, 'D'), pd.Timedelta(0), closed='left'),
]


@pytest.mark.parametrize('interval', INTERVALS, ids=str)
def test_pandas_interval_round_trips(interval):
    cls = T if isinstance(interval.left, pd.Timedelta) else D
    x = cls.from_pandas(interval)
    assert x.is_contiguous
    assert x.to_pandas() == interval and x.to_pandas().closed == interval.closed
    assert cls.from_pandas(x.to_pandas()) == x


def test_to_pandas_converts_to_the_display_zone():
    x = D(pd.Timestamp('2024-01-01', tz='Asia/Singapore'), pd.Timestamp('2024-01-01 10:00', tz='UTC'))
    p = x.to_pandas()
    assert p.left.tz == p.right.tz == x.tz
    assert p == pd.Interval(pd.Timestamp('2024-01-01', tz=SGT), pd.Timestamp('2024-01-01 18:00', tz=SGT), closed='both')


@pytest.mark.parametrize('x', [
    D(), T(),
    D(dt(2024, 1, 1)) | D(dt(2024, 1, 3)),
    D(NEG_INF, dt(2024, 1, 1)), D(dt(2024, 1, 1), POS_INF, end_closed=False), T(NEG_INF, td(0)),
    D.from_seconds(Fraction(1, 3)), T(td(1)) / 7,
], ids=str)
def test_to_pandas_refuses_what_cannot_round_trip(x):
    """an empty set, several pieces, an infinite end (pandas has none) or an end that is no whole number of
    nanoseconds has no pd.Interval"""
    with pytest.raises(ValueError):
        x.to_pandas()


def test_from_pandas_takes_only_an_interval_of_its_type():
    for bad in (pd.Interval(1, 2), (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'))):
        with pytest.raises(TypeError):
            D.from_pandas(bad)
    with pytest.raises(TypeError):
        T.from_pandas(INTERVALS[0])
    with pytest.raises(TypeError):
        D.from_pandas(INTERVALS[-1])


nanoseconds = st.integers(-10 ** 12, 10 ** 12)


@settings(max_examples=60)
@given(nanoseconds, nanoseconds, st.sampled_from(['both', 'left', 'right', 'neither']), st.booleans())
def test_pandas_round_trip_property(a, b, closed, aware):
    a, b = sorted((a, b))
    base = pd.Timestamp('2024-01-01', tz='Asia/Singapore' if aware else None).as_unit('ns')
    interval = pd.Interval(base + pd.Timedelta(a, 'ns'), base + pd.Timedelta(b, 'ns'), closed=closed)
    x = D.from_pandas(interval)
    start = JAN1_SECONDS - (8 * 3600 if aware else 0)  # 2024-01-01 00:00 in singapore is 16:00 utc the day before
    assert x.seconds == M(start + a * NS, start + b * NS, start_closed=interval.closed_left,
                          end_closed=interval.closed_right)
    if x:
        assert x.to_pandas() == interval
    durations = pd.Interval(pd.Timedelta(a, 'ns'), pd.Timedelta(b, 'ns'), closed=closed)
    y = T.from_pandas(durations)
    assert y.seconds == M(a * NS, b * NS, start_closed=durations.closed_left, end_closed=durations.closed_right)
    if y:
        assert y.to_pandas() == durations

# THE REVIEW ROUND (M8's three reviews, 2026-10-04): each pin names its finding

UNIT_CASES = [('s', '2024-01-01 00:00:07', Fraction(7)), ('ms', '2024-01-01 00:00:07.123', Fraction(7123, 10 ** 3)),
              ('us', '2024-01-01 00:00:07.123456', Fraction(7123456, 10 ** 6)),
              ('ns', '2024-01-01 00:00:07.123456789', Fraction(7123456789, 10 ** 9))]


@pytest.mark.parametrize('unit, text, seconds', UNIT_CASES)
def test_every_pandas_unit_is_read_exactly(unit, text, seconds):
    """R1: a Timestamp and a Timedelta in each of pandas' units (only s, us and ns were pinned; 'ms' read as
    1e-4 s stayed green)"""
    t = pd.Timestamp(text).as_unit(unit)
    assert t.unit == unit
    assert D(t).inf_seconds == JAN1_SECONDS + seconds
    assert D(t.tz_localize('Asia/Singapore')).inf_seconds == JAN1_SECONDS - 8 * 3600 + seconds
    x = (t - pd.Timestamp('2024-01-01').as_unit(unit)).as_unit(unit)
    assert x.unit == unit and T(x).inf_seconds == seconds and T(-x).inf_seconds == -seconds


def test_to_pandas_units_and_range():
    """R8: a whole-microsecond end goes through datetime (unit 'us', to year 9999), a nanosecond one is unit 'ns'
    (1677 to 2262); past either range is a ValueError, as the other refusals (the review: it was an OverflowError,
    or pandas' OutOfBoundsDatetime)"""
    far = D(dt(3000, 1, 1), dt(3000, 1, 2)).to_pandas()
    assert far.left == pd.Timestamp('3000-01-01') and far.left.unit == far.right.unit == 'us'
    near = D(pd.Timestamp('2024-01-01 00:00:00.000000001'), dt(2024, 1, 2)).to_pandas()
    assert near.left.unit == 'ns' and near.right.unit == 'us'
    big = T(td(days=200000), td(days=200001)).to_pandas()  # past the nanosecond Timedelta's 106751 days
    assert big.left == pd.Timedelta(td(days=200000)) and big.left.unit == 'us'
    for x in (D.from_seconds(M(0, 253402300800 + 86400)),  # a microsecond end past 9999
              D.from_seconds(M(0, 10 ** 10 + NS)),  # a nanosecond end in 2286
              D.from_seconds(M(-10 ** 10 + NS, 0)),  # and in 1653
              T.from_seconds(M(0, 10 ** 13 + NS)),  # a nanosecond duration past pandas' 106751 days
              D.from_seconds(M(0, 253402300800 + 86400), tz=SGT)):
        with pytest.raises(ValueError, match='range'):
            x.to_pandas()


ARITHMETIC_OPS = [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod, divmod]
NAT_CASES = ([(op, x) for x in (D(dt(2024, 1, 1)), D(dt(2024, 1, 1, tzinfo=SGT))) for op in ARITHMETIC_OPS[:2]]
             + [(op, T(td(1))) for op in ARITHMETIC_OPS])


@pytest.mark.parametrize('op, x', NAT_CASES, ids=str)
def test_nat_is_refused_by_every_operator(op, x):
    """NaT on either side of `+ - * / // % divmod` is a ValueError, as in the constructors (the review: `D + NaT`
    was a TypeError, `D - NaT` a ValueError, `T + NaT` a ValueError, `T - NaT` and the rest TypeErrors). a
    `DateTimeInterval` has `+` and `-` only: its `*` is a TypeError whatever the operand"""
    for thunk in (lambda: op(x, pd.NaT), lambda: op(pd.NaT, x)):
        with pytest.raises(ValueError, match='NaT'):
            thunk()
    if isinstance(x, D):
        with pytest.raises(TypeError):
            x * pd.NaT


FLOOR_MOD_CASES = [(pd.Timedelta(hours=1), T(td(hours=1), td(hours=2))),
                   (pd.Timedelta(hours=5), T(td(hours=1), td(hours=3), start_closed=False)),
                   (pd.Timedelta(minutes=-50), T(td(minutes=20), td(minutes=30)))]


@pytest.mark.parametrize('x, a', FLOOR_MOD_CASES, ids=str)
def test_pandas_timedelta_on_the_left_of_mod_is_sound_and_the_workaround_exact(x, a):
    """review F1: `pd.Timedelta % T` never reaches `T.__rmod__`: pandas computes `x - (x // T) * T` itself, and
    no hook of pandas 3 (`__pandas_priority__`, `__array_priority__`, `_typ`) makes the scalar defer. the result
    holds the exact set (sound, wider: the dependency is lost); `T(x) % A` is exact, as the docs say"""
    exact = x.to_pytimedelta() % a
    assert T(x) % a == exact and divmod(T(x), a)[1] == exact
    assert exact.seconds.issubset((x % a).seconds)
    q, r = divmod(x, a)
    assert q == x.to_pytimedelta() // a and exact.seconds.issubset(r.seconds)


@settings(max_examples=40)
@given(st.integers(-10 ** 9, 10 ** 9), st.integers(-10 ** 9, 10 ** 9), st.integers(1, 10 ** 9), st.booleans(),
       st.sampled_from([np.float64(2.5), np.int64(-3), np.float32(0.1), np.int32(7)]))
def test_pandas_and_numpy_scalars_on_the_left_are_the_numeric_ones(lo, hi, h, aware, f):
    """review F8: pandas and numpy scalars on the LEFT of the table's operators: the seconds are the numeric op's
    on the scalar's own reading, the kind and zone kept (`%` and `divmod` with a pandas Timedelta on the left
    excepted: F1)"""
    lo, hi = sorted((lo, hi))
    x = T(pd.Timedelta(lo, 'us'), pd.Timedelta(hi, 'us'))
    s = pd.Timedelta(h, 'us')
    t = pd.Timestamp('2024-01-01', tz='Asia/Singapore' if aware else None) + pd.Timedelta(h, 'us')
    a = D(t, t + pd.Timedelta(hi - lo, 'us'))
    ss, ts = T(s).seconds, D(t).seconds
    exact_f = M(Fraction(float(f))) if isinstance(f, np.floating) else M(int(f))
    cases = [(lambda: s + x, T, lambda: ss + x.seconds), (lambda: s - x, T, lambda: ss - x.seconds),
             (lambda: s / x, M, lambda: ss / x.seconds), (lambda: s // x, M, lambda: ss // x.seconds),
             (lambda: s + a, D, lambda: ss + a.seconds), (lambda: t + x, D, lambda: ts + x.seconds),
             (lambda: t - x, D, lambda: ts - x.seconds), (lambda: t - a, T, lambda: ts - a.seconds),
             (lambda: f * x, T, lambda: exact_f * x.seconds)]
    for thunk, cls, numeric in cases:
        with warnings.catch_warnings():  # a ratio by a set holding 0 warns, as the numeric op does
            warnings.simplefilter('ignore')
            result, seconds = thunk(), numeric()
        assert type(result) is cls
        assert (result if cls is M else result.seconds) == seconds
        if cls is D:
            assert result.tz == (t.tzinfo if aware else None)
