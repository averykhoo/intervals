"""
the time layer (M8, D30): `DateTimeInterval` and `TimeDeltaInterval`, immutable and hashable, each a thin
wrapper over an exact `MultiInterval` of seconds (D4's (a)); every set operation, relation and arithmetic
op is the numeric class's on those seconds.

**readings** (how a value becomes exact seconds; nothing is rounded):

* a naive `datetime` is wall-clock time, `d - datetime(1970, 1, 1)` as an exact number, never
  `timestamp()` (machine- and DST-dependent, and an `OSError` on windows before 1970). an aware one is its
  exact UTC instant. naive and aware never mix (`TypeError`, as python and pandas); aware ends in
  different zones do, the set is of instants, and one zone is kept for display only: the constructor's
  `tz=`, else its start's zone, else its end's; of an op, the left operand's. it is never part of `==`
* a `date` is the half-open day `[d 00:00, d+1 00:00)`, and its flag says whether the day is in: a
  closed start is `d 00:00`, an open start "after d" (`d+1 00:00`, closed), a closed end "through d"
  (`d+1 00:00`, open), an open end "before d" (`d 00:00`, open). so adjacent days tile into one piece and
  a day is 86400 s. a date is naive; `tz=` reads it as that day in a zone (aware). a datetime, midnight
  included, is an exact instant: no end-of-day snap of any kind (v1 snapped both)
* a `timedelta` is `days * 86400 + seconds + microseconds / 10 ** 6` (`total_seconds()` drops a
  microsecond at 10 ** 6 days). a pandas `Timestamp` or `Timedelta` is read exactly in its own unit (its
  `asm8`; `.value` is nanoseconds and overflows past them): a naive `Timestamp` as wall clock, which is
  pandas' own reading, an aware one as UTC
* `NaT` is refused (`ValueError`), and anything that is not a time value (a number, nan) is a `TypeError`;
  so are numpy's `datetime64` and `timedelta64`, as bounds and as factors (numpy registers `timedelta64`
  as an integer, so the numeric class refuses it too): read one with pandas first (`pd.Timedelta(x)`)
* the order of two bounds is checked on these readings, by `MultiInterval`'s rule: a start read after
  the end is a `ValueError`, equal readings with an open flag are empty. so `DateTimeInterval(
  datetime(2024, 1, 2, 12), date(2024, 1, 2))` is `[01-02 12:00, 01-03 00:00)` (noon through the day),
  `DateTimeInterval(date(2024, 1, 2), datetime(2024, 1, 2, 12), start_closed=False)` (after the 2nd,
  until its noon) raises, and `DateTimeInterval(d, d, start_closed=False)` (after d, through d) is empty

**infinite ends**: `NEG_INF` and `POS_INF` order below and above every datetime, date, timedelta, pandas
`Timestamp` and `Timedelta` and each other, hash, print as `-inf` / `inf` and are taken back by the
constructors and by slicing. they are the read-out of an infinite end, stored as ±inf in the numeric
class, closed or open as written (`DateTimeInterval(NEG_INF, t)` holds the point -inf, as
`MultiInterval(-inf, 5)` does); a missing slice bound is closed, as in `MultiInterval.__getitem__`.
`math.inf` cannot stand in (it does not order against a datetime), nor `datetime.max` (a finite instant).

**read-outs**: `inf` and `sup` give datetimes (timedeltas), `degenerate_points` a tuple of them sorted by
instant, `total_duration` a timedelta. an end that is no whole number of microseconds (a nanosecond
`Timestamp`, `td / 3`) has no datetime, so reading it raises `ValueError` naming the raw accessor beside
it (`inf_seconds`, `sup_seconds`, `seconds`, `total_seconds`, `size`), which gives the exact seconds and
never fails; an instant whose datetime in the display zone is outside datetime's range raises
`OverflowError` the same way (every datetime that was an input reads back, also `9999-12-31 23:00-05:00`,
whose UTC instant has no datetime). iteration gives the pieces as intervals of the same class and never
fails.

**arithmetic** (any other pairing is a `TypeError`):

    | left           | op           | right          | result             |
    |----------------|--------------|----------------|--------------------|
    | datetime       | + -          | timedelta      | `DateTimeInterval` |
    | timedelta      | +            | datetime       | `DateTimeInterval` |
    | datetime       | -            | datetime       | `TimeDeltaInterval`|
    | timedelta      | + -  %       | timedelta      | `TimeDeltaInterval`|
    | timedelta      | * /          | real           | `TimeDeltaInterval`|
    | real           | *            | timedelta      | `TimeDeltaInterval`|
    | timedelta      | / //         | timedelta      | `MultiInterval`    |

each side an interval or a scalar of its kind (a `date` is its day); a real factor is an int, Fraction,
float or `MultiInterval`, a float taken at its exact value (`td * 0.1` is exact, so it is not a whole
number of microseconds: use `Fraction(1, 10)`). a sentinel is not an arithmetic operand: write
`TimeDeltaInterval(POS_INF)`. `-`, `+` and `abs` of a `TimeDeltaInterval` are its own class; `divmod` of
two is `(td // td, td % td)`. `NaT` on either side of an operator is a `ValueError`, as in the
constructors.

an aware `dt + td` adds elapsed time, as `dt - dt` gives it (the set is of instants): across a DST change
`t + timedelta(days=1)` is 24 hours later, not the same wall time on the next day as python's aware `+`
gives, and `DateTimeInterval(d, tz=tz) + timedelta(days=1)` is not the next day in tz. a naive interval's
`+` is wall clock, as python's.

with a pandas `Timedelta` on the LEFT of `%` or `divmod`, pandas computes `x - (x // A) * A` itself (no
hook of pandas 3 makes the scalar defer), which loses the dependency between the two x: a sound but wider
set (`pd.Timedelta(hours=1) % TimeDeltaInterval(hour, 2 * hour)` holds -1:00:00 to 0:00:00 besides 0:00:00
and 1:00:00). put the interval on the left: `TimeDeltaInterval(x) % A` is exact. the quotient, `+ - / //`,
and a python `timedelta` on the left are exact.

**comparisons** `< <= > >=` are v2's pointwise `TruthSet` (`.certainly`, `.possibly`; `bool` raises on
`BOTH`), so a v1 caller's `if a < b:` can raise. `==` is structural and the same set of instants: a
foreign type is `NotImplemented`, the display zone never counts. `date in A` asks whether the day is a
subset; a datetime is a point.

**pandas** stays optional: the library never imports it to read a value, it looks for it in
`sys.modules` (a `Timestamp` exists only once its user has imported pandas); only `to_pandas()` imports
it. `to_pandas()` of one bounded piece is a `pd.Interval` and `from_pandas()` takes it back; an empty
set, several pieces, an unbounded end, an end that is no whole number of nanoseconds and one past
pandas' range have no `pd.Interval` and raise `ValueError`.

>>> import datetime
>>> d = datetime.date(2024, 1, 1)
>>> week = DateTimeInterval(d, d + datetime.timedelta(6))
>>> print(week)
[2024-01-01 00:00:00, 2024-01-08 00:00:00)
>>> week.total_duration
datetime.timedelta(days=7)
>>> DateTimeInterval(d) | DateTimeInterval(d + datetime.timedelta(1))   # adjacent days tile
DateTimeInterval(datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 3, 0, 0), end_closed=False)
>>> DateTimeInterval(NEG_INF, datetime.datetime(2024, 1, 1, 12)).inf
-inf
"""
import datetime as _dt
import functools
import math
import sys
from fractions import Fraction
from numbers import Real
from typing import Iterator
from typing import Tuple

from intervals import kernel
from intervals.cuts import Cut
from intervals.cuts import Value
from intervals.cuts import below
from intervals.cuts import end_cut
from intervals.cuts import is_numpy_time
from intervals.cuts import normalize_value
from intervals.cuts import start_cut
from intervals.kernel import Size
from intervals.multi_interval import MultiInterval
from intervals.relations import TruthSet

_EPOCH = _dt.datetime(1970, 1, 1)
_EPOCH_UTC = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_DAY = 86400
_US = 10 ** 6
_NS = 10 ** 9
_UNITS = {'s': 1, 'ms': 10 ** 3, 'us': _US, 'ns': _NS}
_CLOSED = {(True, True): 'both', (True, False): 'left', (False, True): 'right', (False, False): 'neither'}


def _pandas():
    """pandas if it is loaded already, else None: the library never imports it to read a value"""
    return sys.modules.get('pandas')


def _is_nat(x) -> bool:
    pd = _pandas()
    return pd is not None and x is pd.NaT


# THE INFINITE ENDS

class _Infinity:
    """
    the read-out of an infinite end: below (`NEG_INF`) or above (`POS_INF`) every datetime, date,
    timedelta, pandas `Timestamp` and `Timedelta`, and each other. `NaT` does not order (every comparison
    False, as nan); any other type is a TypeError, numpy's `datetime64` and `timedelta64` among them

    >>> import datetime
    >>> NEG_INF < datetime.date.min, POS_INF > datetime.timedelta.max, -POS_INF is NEG_INF
    (True, True, True)
    """
    __slots__ = ('_sign', '_name')
    # numpy hands a comparison back to python: `np.datetime64(...) < POS_INF` is a TypeError, not numpy's False
    __array_ufunc__ = None

    def __init__(self, sign: int, name: str):
        object.__setattr__(self, '_sign', sign)
        object.__setattr__(self, '_name', name)

    def __setattr__(self, name, value):
        raise AttributeError('the infinite ends are immutable')

    def _compare(self, other):
        """the sign of `self - other`, None against NaT, NotImplemented for a foreign type"""
        if isinstance(other, _Infinity):
            return (self._sign > other._sign) - (self._sign < other._sign)
        if _is_nat(other):
            return None
        if isinstance(other, (_dt.date, _dt.timedelta)):  # datetime, Timestamp and Timedelta among them
            return self._sign
        return NotImplemented

    def __lt__(self, other):
        c = self._compare(other)
        return c if c is NotImplemented else c is not None and c < 0

    def __le__(self, other):
        c = self._compare(other)
        return c if c is NotImplemented else c is not None and c <= 0

    def __gt__(self, other):
        c = self._compare(other)
        return c if c is NotImplemented else c is not None and c > 0

    def __ge__(self, other):
        c = self._compare(other)
        return c if c is NotImplemented else c is not None and c >= 0

    __eq__ = object.__eq__  # each is equal to itself only
    __hash__ = object.__hash__

    def __neg__(self) -> '_Infinity':
        return POS_INF if self is NEG_INF else NEG_INF

    def __repr__(self) -> str:
        return '-inf' if self._sign < 0 else 'inf'

    def __reduce__(self):
        return self._name  # pickled and copied as the module's singleton


NEG_INF = _Infinity(-1, 'NEG_INF')
POS_INF = _Infinity(1, 'POS_INF')


# READINGS: a value to exact seconds

def _pandas_seconds(x) -> Value:
    """a `Timestamp` (UTC if aware, wall clock if naive) or `Timedelta`, exactly, in its own unit"""
    return normalize_value(Fraction(int(x.asm8.view('i8')), _UNITS[getattr(x, 'unit', 'ns')]))


def _timedelta_seconds(td: _dt.timedelta) -> Value:
    """exact seconds of a timedelta or a pandas Timedelta (never `total_seconds()`)"""
    pd = _pandas()
    if pd is not None and isinstance(td, pd.Timedelta):
        return _pandas_seconds(td)
    return normalize_value(td.days * _DAY + td.seconds + Fraction(td.microseconds, _US))


def _is_aware(d: _dt.datetime) -> bool:
    return d.tzinfo is not None and d.utcoffset() is not None


def _datetime_seconds(d: _dt.datetime) -> Value:
    """exact seconds of a datetime: wall clock since naive 1970-01-01 if naive, UTC if aware"""
    pd = _pandas()
    if pd is not None and isinstance(d, pd.Timestamp):
        return _pandas_seconds(d)
    return _timedelta_seconds(d - (_EPOCH_UTC if _is_aware(d) else _EPOCH))


def _day_bounds(d: _dt.date, tz) -> Tuple[Value, Value]:
    """the seconds of `d 00:00` and of `d+1 00:00`, naive or in the zone tz"""
    if tz is None:
        start = (d.toordinal() - _EPOCH.toordinal()) * _DAY
        return start, start + _DAY
    start = _datetime_seconds(_dt.datetime.combine(d, _dt.time(), tzinfo=tz))
    try:
        stop = _datetime_seconds(_dt.datetime.combine(d + _dt.timedelta(1), _dt.time(), tzinfo=tz))
    except OverflowError:  # date.max: no next date to name
        stop = start + _DAY
    return start, stop


def _exact(mi: MultiInterval) -> MultiInterval:
    """the same set in the exact class, every finite float end its exact Fraction"""
    return MultiInterval.from_cuts(
        Cut(Fraction(c.value), c.side) if isinstance(c.value, float) and math.isfinite(c.value) else c
        for c in mi.cuts)


def _factor(x):
    """a real factor or divisor of a duration as an exact MultiInterval, or NotImplemented"""
    if isinstance(x, MultiInterval):
        return _exact(x)
    if is_numpy_time(x):  # numpy registers timedelta64 as an Integral: `int()` of it is its count in its unit
        raise TypeError(f'a numpy {type(x).__name__} is a time value, not a real factor; '
                        f'read it with pandas first (pd.Timedelta)')
    if isinstance(x, bool) or not isinstance(x, Real):
        return NotImplemented
    value = normalize_value(x)  # nan: ValueError
    if isinstance(value, float) and math.isfinite(value):
        value = normalize_value(Fraction(value))
    return MultiInterval(value)


def _as_seconds(seconds) -> MultiInterval:
    if isinstance(seconds, MultiInterval):
        return _exact(seconds)
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise TypeError(f'expected a MultiInterval of seconds or a real number, got {type(seconds).__name__}')
    return _exact(MultiInterval(seconds))


def _has_finite_end(mi: MultiInterval) -> bool:
    return any(-math.inf < c.value < math.inf for c in mi.cuts)


def _nat_refused(method):
    """an arithmetic operator: `NaT` on either side is a ValueError, as in the constructors (pandas' `NaT`
    is a `datetime` but no `timedelta`, so without this the error type depended on the operator)"""
    @functools.wraps(method)
    def operator(self, other):
        if _is_nat(other):
            raise ValueError('NaT is not a point of time or a duration')
        return method(self, other)
    return operator


def _whole_microseconds(v: Value, accessor: str) -> int:
    us = Fraction(v) * _US
    if us.denominator != 1:
        raise ValueError(f'{v} s is not a whole number of microseconds, so it has no datetime or timedelta; '
                         f'`{accessor}` gives it exactly')
    return int(us)


def _fraction_text(v: Value) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'


class _Bound:
    """one constructor bound read: its cut, kind and zone"""
    __slots__ = ('cut', 'aware', 'tz')

    def __init__(self, cut, aware, tz):
        self.cut, self.aware, self.tz = cut, aware, tz


def _infinite_bound(x: _Infinity, is_start: bool, closed: bool) -> _Bound:
    v = math.inf if x is POS_INF else -math.inf
    return _Bound((start_cut if is_start else end_cut)(v, closed), None, None)


def _piece_cuts(s: _Bound, e: _Bound, start, end) -> tuple:
    """the cuts of one piece, the order checked on the READ bounds by `MultiInterval`'s rule (`kernel.piece`):
    a start read after the end is a ValueError, equal readings with an open flag are empty. a date bound is a
    midnight, its own or the next by its flag, so a datetime and a date (which python does not order) compare"""
    if s.cut.value > e.cut.value:
        dates = any(isinstance(x, _dt.date) and not isinstance(x, _dt.datetime) for x in (start, end))
        raise ValueError(f'interval start {start!r} is after end {end!r}'
                         + (' as read (a date bound is a midnight: its own, or the next by its flag)' if dates else ''))
    return (s.cut, e.cut) if s.cut < e.cut else ()


# THE COMMON PART

class _TimeInterval:
    """the wrapper both classes share: set algebra, relations, pieces, read-outs, over `self._mi`"""
    __slots__ = ('_mi',)
    # numpy hands an operator back to python (`np.float64(2) * td` is `td.__rmul__`), never an object array
    __array_ufunc__ = None

    # subclass hooks
    def _coerce(self, other):
        raise NotImplementedError

    def _check_kinds(self, *others) -> None:
        pass

    def _with(self, mi: MultiInterval, *others):
        raise NotImplementedError

    def _from_us(self, us: int):
        raise NotImplementedError

    def _str_value(self, value, rest: str = '') -> str:
        raise NotImplementedError

    # IMMUTABILITY

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name):
        raise AttributeError(f'{type(self).__name__} is immutable')

    @property
    def seconds(self) -> MultiInterval:
        """the raw set: an exact `MultiInterval` of seconds (since naive 1970-01-01, or UTC if aware)"""
        return self._mi

    # OPERANDS

    def _operand(self, other):
        """other as this class (an interval or a scalar of it), kinds checked; NotImplemented if foreign"""
        coerced = self._coerce(other)
        if coerced is not NotImplemented:
            self._check_kinds(coerced)
        return coerced

    def _operand_or_raise(self, other):
        coerced = self._operand(other)
        if coerced is NotImplemented:
            raise TypeError(f'expected a {type(self).__name__} or a scalar of it, got {type(other).__name__}')
        return coerced

    # SET ALGEBRA

    def union(self, *others):
        os_ = [self._operand_or_raise(o) for o in others]
        return self._with(self._mi.union(*(o._mi for o in os_)), *os_)

    def intersection(self, *others):
        os_ = [self._operand_or_raise(o) for o in others]
        return self._with(self._mi.intersection(*(o._mi for o in os_)), *os_)

    def difference(self, *others):
        os_ = [self._operand_or_raise(o) for o in others]
        return self._with(self._mi.difference(*(o._mi for o in os_)), *os_)

    def symmetric_difference(self, *others):
        os_ = [self._operand_or_raise(o) for o in others]
        return self._with(self._mi.symmetric_difference(*(o._mi for o in os_)), *os_)

    def complement(self):
        return self._with(self._mi.complement())

    def __invert__(self):
        return self.complement()

    def _set_op(self, other, op, reflected=False):
        o = self._operand(other)
        if o is NotImplemented:
            return NotImplemented
        a, b = (o, self) if reflected else (self, o)
        return a._with(op(a._mi, b._mi), b)

    def __or__(self, other):
        return self._set_op(other, MultiInterval.__or__)

    def __ror__(self, other):
        return self._set_op(other, MultiInterval.__or__, reflected=True)

    def __and__(self, other):
        return self._set_op(other, MultiInterval.__and__)

    def __rand__(self, other):
        return self._set_op(other, MultiInterval.__and__, reflected=True)

    def __xor__(self, other):
        return self._set_op(other, MultiInterval.__xor__)

    def __rxor__(self, other):
        return self._set_op(other, MultiInterval.__xor__, reflected=True)

    def issubset(self, other) -> bool:
        return self._mi.issubset(self._operand_or_raise(other)._mi)

    def issuperset(self, other) -> bool:
        return self._mi.issuperset(self._operand_or_raise(other)._mi)

    def isdisjoint(self, other) -> bool:
        return self._mi.isdisjoint(self._operand_or_raise(other)._mi)

    def __contains__(self, item) -> bool:
        """
        a scalar instant (duration): membership; a `date`: its whole day is a subset; an interval: subset

        >>> import datetime, pandas
        >>> late = pandas.Timestamp('2024-01-01 23:59:59.999999500')   # nanoseconds, no datetime
        >>> late in DateTimeInterval(datetime.date(2024, 1, 1)), late in DateTimeInterval(datetime.date(2024, 1, 2))
        (True, False)
        """
        return self._operand_or_raise(item)._mi.issubset(self._mi)

    def __getitem__(self, item: slice):
        """
        `A[a:b]` is the restriction to the closed `[a, b]` (a `date` stop: through that day; the bounds
        read and ordered as the constructor's); a missing bound is the closed infinity. a date is naive, so
        an aware interval refuses a date bound (TypeError), as it refuses a naive datetime
        """
        if not isinstance(item, slice):
            raise TypeError(f'{type(self).__name__} supports slicing only, e.g. A[a:b]; use `in` for membership')
        if item.step is not None:
            raise TypeError('slice step is not supported')
        start = NEG_INF if item.start is None else item.start
        stop = POS_INF if item.stop is None else item.stop
        return self & type(self)(start, stop)

    def expand(self, distance):
        """widen every piece by a finite, non-negative `timedelta` (or pandas `Timedelta`) on both sides"""
        if not isinstance(distance, _dt.timedelta):
            raise TypeError(f'expand() takes a timedelta, got {type(distance).__name__}')
        return self._with(self._mi.expand(_timedelta_seconds(distance)))

    # POINTWISE COMPARISONS (a TruthSet)

    def _compare(self, other, op):
        o = self._operand(other)
        return NotImplemented if o is NotImplemented else op(self._mi, o._mi)

    def __lt__(self, other):
        return self._compare(other, MultiInterval.__lt__)

    def __le__(self, other):
        return self._compare(other, MultiInterval.__le__)

    def __gt__(self, other):
        return self._compare(other, MultiInterval.__gt__)

    def __ge__(self, other):
        return self._compare(other, MultiInterval.__ge__)

    def eq_pointwise(self, other) -> TruthSet:
        return self._mi.eq_pointwise(self._operand_or_raise(other)._mi)

    # SET-LEVEL RELATIONS (bool; see MultiInterval)

    def before(self, other) -> bool:
        return self._mi.before(self._operand_or_raise(other)._mi)

    def after(self, other) -> bool:
        return self._mi.after(self._operand_or_raise(other)._mi)

    def adjoins(self, other) -> bool:
        return self._mi.adjoins(self._operand_or_raise(other)._mi)

    def overlaps(self, other) -> bool:
        return self._mi.overlaps(self._operand_or_raise(other)._mi)

    def contains(self, other) -> bool:
        return self._mi.contains(self._operand_or_raise(other)._mi)

    def within(self, other) -> bool:
        return self._mi.within(self._operand_or_raise(other)._mi)

    def allen(self, other):
        return self._mi.allen(self._operand_or_raise(other)._mi)

    def allen_matrix(self, other):
        return self._mi.allen_matrix(self._operand_or_raise(other)._mi)

    def allen_relations(self, other):
        return self._mi.allen_relations(self._operand_or_raise(other)._mi)

    def weakly_less(self, other) -> bool:
        return self._mi.weakly_less(self._operand_or_raise(other)._mi)

    def strictly_less(self, other) -> bool:
        return self._mi.strictly_less(self._operand_or_raise(other)._mi)

    # CONTAINER PROTOCOL

    def __hash__(self):
        return hash(self._mi)

    @property
    def sort_key(self):
        """structural order, for `sorted(xs, key=lambda x: x.sort_key)`"""
        return self._mi.sort_key

    def __bool__(self) -> bool:
        return bool(self._mi)

    def __len__(self) -> int:
        """the number of pieces"""
        return len(self._mi)

    def __iter__(self) -> Iterator:
        """the pieces, each an interval of this class"""
        for piece in self._mi:
            yield self._with(piece)

    @property
    def pieces(self) -> tuple:
        return tuple(self)

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return self._mi.is_empty

    @property
    def is_contiguous(self) -> bool:
        return self._mi.is_contiguous

    @property
    def is_degenerate(self) -> bool:
        return self._mi.is_degenerate

    @property
    def is_finite(self) -> bool:
        """no end at, or piece reaching, an infinity (the empty set is finite)"""
        return self._mi.is_finite

    @property
    def hull(self):
        return self._with(self._mi.hull)

    @property
    def closed_hull(self):
        return self._with(self._mi.closed_hull)

    @property
    def interior(self):
        return self._with(self._mi.interior)

    @property
    def size(self) -> Size:
        """the raw `Size(rays, length, points)` of the seconds (see `MultiInterval.size`); never fails"""
        return self._mi.size

    @property
    def total_seconds(self) -> Value:
        """the exact length in seconds, the gaps left out; ValueError when unbounded"""
        size = self._mi.size
        if size.rays:
            raise ValueError(f'{self} is unbounded: it has no total length; `size` gives its rays')
        return size.length

    @property
    def total_duration(self) -> _dt.timedelta:
        """`total_seconds` as a timedelta; raises if it is no whole number of microseconds"""
        us = _whole_microseconds(self.total_seconds, 'total_seconds')
        try:
            return _dt.timedelta(microseconds=us)
        except OverflowError:
            raise OverflowError(f'{self.total_seconds} s is past timedelta; `total_seconds` gives it exactly') \
                from None

    # READ-OUTS

    def _out(self, v: Value, accessor: str):
        """an end's value as a datetime (timedelta) or a sentinel; raises naming the raw accessor"""
        if v == math.inf:
            return POS_INF
        if v == -math.inf:
            return NEG_INF
        us = _whole_microseconds(v, accessor)
        try:
            return self._from_us(us)
        except OverflowError:
            raise OverflowError(f'{v} s is outside the range of {self._scalar_name} (in the display zone); '
                                f'`{accessor}` gives it exactly') from None
        except ValueError as e:  # a tzinfo under which no datetime reads as this instant
            raise ValueError(f'{e}; `{accessor}` gives it exactly') from None

    @property
    def inf_seconds(self) -> Value:
        """the infimum's exact seconds (±inf for an infinite end); ValueError when empty"""
        return self._mi.inf

    @property
    def sup_seconds(self) -> Value:
        return self._mi.sup

    @property
    def inf(self):
        """the infimum, a datetime (timedelta) or `NEG_INF`/`POS_INF`; ValueError when empty"""
        return self._out(self._mi.inf, 'inf_seconds')

    @property
    def sup(self):
        return self._out(self._mi.sup, 'sup_seconds')

    @property
    def inf_closed(self) -> bool:
        return self._mi.inf_closed

    @property
    def sup_closed(self) -> bool:
        return self._mi.sup_closed

    @property
    def degenerate_points(self) -> tuple:
        """
        the single-point pieces, sorted by instant: a tuple, not a set as `MultiInterval`'s, because python
        compares and hashes two datetimes of one zone by wall clock, so the two instants of a DST fold (fold=0
        and fold=1) would be one element of a set
        """
        return tuple(self._out(v, 'seconds.degenerate_points') for v in sorted(self._mi.degenerate_points))

    # PANDAS

    @classmethod
    def from_pandas(cls, interval):
        """a `pd.Interval` of `Timestamp`s (`Timedelta`s), its ends and `closed` kept exactly"""
        pd = _pandas()
        if pd is None or not isinstance(interval, pd.Interval):
            raise TypeError(f'from_pandas() takes a pandas Interval, got {type(interval).__name__}')
        return cls(interval.left, interval.right, start_closed=interval.closed_left, end_closed=interval.closed_right)

    def to_pandas(self):
        """
        one bounded piece as a `pd.Interval` (imports pandas). an empty set, several pieces (`[p.to_pandas()
        for p in A]`), an infinite end (pandas has none), an end that is no whole number of
        nanoseconds and one past pandas' range raise ValueError
        """
        import pandas as pd
        if len(self._mi) != 1:
            raise ValueError(f'a pd.Interval is one piece; {self} has {len(self._mi)}: [p.to_pandas() for p in A]')
        if not self._mi.is_finite:
            raise ValueError(f'{self} has an infinite end, and pandas has no infinite {self._pandas_name}')
        (lo, lo_closed, hi, hi_closed), = kernel.pieces(self._mi.cuts)
        return pd.Interval(self._pandas_end(pd, lo), self._pandas_end(pd, hi), closed=_CLOSED[lo_closed, hi_closed])

    def _pandas_end(self, pd, v: Value):
        ns = Fraction(v) * _NS
        if ns.denominator != 1:
            raise ValueError(f'{v} s is not a whole number of nanoseconds, so it has no pandas '
                             f'{self._pandas_name}; `seconds` gives it exactly')
        try:
            if ns % 1000 == 0:  # a whole microsecond: through datetime, unit 'us' (to year 9999)
                return self._pandas_scalar(pd, self._out(v, 'seconds'))
            return self._pandas_ns(pd, int(ns))  # unit 'ns': 1677 to 2262
        except (OverflowError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
            raise ValueError(f'{v} s is past the range of a pandas {self._pandas_name}; `seconds` gives it '
                             f'exactly') from None

    # FORMATTING

    def __repr__(self) -> str:
        """the constructor call (pieces joined by `|`), which evaluates back in a namespace holding
        `datetime`, `zoneinfo` and the package's names; `from_seconds(...)` where an end has no datetime"""
        name = type(self).__name__
        if not self._mi:
            return f'{name}()'
        try:
            return ' | '.join(self._piece_repr(*p) for p in kernel.pieces(self._mi.cuts))
        except (ValueError, OverflowError):
            return f'{name}.from_seconds({self._mi!r}{self._tz_arg()})'

    def _tz_arg(self) -> str:
        return ''

    def _piece_repr(self, lo, lo_closed, hi, hi_closed) -> str:
        name = type(self).__name__
        if lo == hi:
            return f'{name}({self._value_repr(lo)})'
        flags = ('' if lo_closed else ', start_closed=False') + ('' if hi_closed else ', end_closed=False')
        return f'{name}({self._value_repr(lo)}, {self._value_repr(hi)}{flags})'

    def _value_repr(self, v) -> str:
        out = self._out(v, 'seconds')
        if out is NEG_INF:
            return 'NEG_INF'
        if out is POS_INF:
            return 'POS_INF'
        return repr(out)

    def __str__(self) -> str:
        """readable and never raising: `[a, b)`, `[a]`, `{ [a, b) , [c] }`; an end with no datetime
        prints its whole microseconds plus the exact rest, as `...00.333333+1/3us`"""
        texts = []
        for lo, lo_closed, hi, hi_closed in kernel.pieces(self._mi.cuts):
            if lo == hi:
                texts.append(f'[{self._end_text(lo)}]')
            else:
                texts.append(f'{"[" if lo_closed else "("}{self._end_text(lo)}, '
                             f'{self._end_text(hi)}{"]" if hi_closed else ")"}')
        if not texts:
            return '{}'
        return texts[0] if len(texts) == 1 else '{ ' + ' , '.join(texts) + ' }'

    def _end_text(self, v: Value) -> str:
        if v == math.inf:
            return 'inf'
        if v == -math.inf:
            return '-inf'
        us = Fraction(v) * _US
        whole = math.floor(us)
        try:
            value = self._from_us(whole)
        except (OverflowError, ValueError):  # past the range; a tzinfo that cannot convert
            return f'{_fraction_text(v)} s'
        rest = us - whole
        return self._str_value(value, f'+{_fraction_text(rest)}us' if rest else '')


# DATETIMES

class DateTimeInterval(_TimeInterval):
    """
    a set of instants: zero or more disjoint pieces, each with open or closed ends (see the module
    docstring for the readings, the sentinels and the arithmetic)

    `DateTimeInterval()` is empty, `DateTimeInterval(t)` the point t (a `date`: its day), and
    `DateTimeInterval(a, b, start_closed=..., end_closed=..., tz=None)` one piece. each bound is a
    datetime, a date, a pandas `Timestamp` or a sentinel; `tz` reads a date bound as that day in a zone,
    and is the display zone. aware arithmetic is in elapsed time, as `dt - dt` is: across a DST change
    `A + timedelta(days=1)` is 24 hours later, not python's same wall time on the next day

    >>> import datetime
    >>> A = DateTimeInterval(datetime.datetime(2024, 1, 1, 9), datetime.datetime(2024, 1, 1, 17))
    >>> datetime.datetime(2024, 1, 1, 12) in A, datetime.date(2024, 1, 1) in A
    (True, False)
    >>> A - datetime.datetime(2024, 1, 1)
    TimeDeltaInterval(datetime.timedelta(seconds=32400), datetime.timedelta(seconds=61200))
    """
    __slots__ = ('_aware', '_tz')

    def __init__(self, start=None, end=None, *, start_closed: bool = True, end_closed: bool = True, tz=None):
        if tz is not None and not isinstance(tz, _dt.tzinfo):
            raise TypeError(f'tz must be a datetime.tzinfo, got {type(tz).__name__}')
        if start is None:
            if end is not None:
                raise ValueError('an end without a start')
            self._init(MultiInterval(), None, None)
            return
        if end is None:
            if start_closed != end_closed:
                raise ValueError(f'half-open degenerate interval at {start!r}')
            end = start
        s = _datetime_bound(start, True, start_closed, tz)
        e = _datetime_bound(end, False, end_closed, tz)
        if s.aware is not None and e.aware is not None and s.aware != e.aware:
            raise TypeError(f'cannot mix naive and aware ends: {start!r}, {end!r}')
        aware = s.aware if s.aware is not None else e.aware
        cuts = _piece_cuts(s, e, start, end)
        self._init(MultiInterval.from_cuts(cuts), aware, tz if tz is not None else s.tz or e.tz)

    def _init(self, mi: MultiInterval, aware, tz):
        if not _has_finite_end(mi):
            aware = tz = None  # no finite end, no kind: combines with naive and aware alike
        elif not aware:
            tz = None
        object.__setattr__(self, '_mi', mi)
        object.__setattr__(self, '_aware', aware)
        object.__setattr__(self, '_tz', tz)

    @classmethod
    def _make(cls, mi: MultiInterval, aware, tz) -> 'DateTimeInterval':
        out = object.__new__(cls)
        out._init(mi, aware, tz)
        return out

    @classmethod
    def from_seconds(cls, seconds, tz=None) -> 'DateTimeInterval':
        """
        the raw constructor: a `MultiInterval` (or a number) of seconds, naive (wall clock since
        1970-01-01) if `tz` is None, else UTC seconds displayed in `tz`. a float end is taken exactly

        >>> DateTimeInterval.from_seconds(MultiInterval(0, 86400)).sup
        datetime.datetime(1970, 1, 2, 0, 0)
        """
        if tz is not None and not isinstance(tz, _dt.tzinfo):
            raise TypeError(f'tz must be a datetime.tzinfo, got {type(tz).__name__}')
        return cls._make(_as_seconds(seconds), tz is not None, tz)

    def __reduce__(self):
        return type(self)._make, (self._mi, self._aware, self._tz)

    _scalar_name = 'datetime'
    _pandas_name = 'Timestamp'

    @property
    def tz(self):
        """the display zone of an aware interval; None if naive or without a finite end"""
        return self._tz

    def astimezone(self, tz) -> 'DateTimeInterval':
        """the same instants displayed in `tz`; TypeError for a naive interval (it has no zone). an interval
        with no finite end (empty, `[-inf, inf]`) has no kind and no zone: itself, `tz` stays None"""
        if not isinstance(tz, _dt.tzinfo):
            raise TypeError(f'tz must be a datetime.tzinfo, got {type(tz).__name__}')
        if self._aware is None:
            return self
        if not self._aware:
            raise TypeError('a naive interval has no zone to convert from')
        return self._make(self._mi, True, tz)

    def _coerce(self, other):
        if isinstance(other, DateTimeInterval):
            return other
        if isinstance(other, (_Infinity, _dt.date)):
            return DateTimeInterval(other)
        return NotImplemented

    def _check_kinds(self, *others) -> None:
        kinds = {o._aware for o in (self, *others)} - {None}
        if len(kinds) > 1:
            raise TypeError('cannot mix naive and aware datetimes')

    def _with(self, mi: MultiInterval, *others) -> 'DateTimeInterval':
        """a result: the kind of the operands, the display zone of the first aware one"""
        for o in (self, *others):
            if o._aware is not None:
                return self._make(mi, o._aware, o._tz)
        return self._make(mi, None, None)

    def __eq__(self, other):
        if not isinstance(other, DateTimeInterval):
            return NotImplemented
        return self._mi == other._mi and self._aware == other._aware

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = _TimeInterval.__hash__

    def _from_us(self, us: int) -> _dt.datetime:
        if self._aware:
            return _aware_datetime(us, self._tz)
        return _EPOCH + _dt.timedelta(microseconds=us)

    def _str_value(self, value: _dt.datetime, rest: str = '') -> str:
        """iso 8601 with a space; a sub-microsecond rest goes before the utc offset"""
        naive = value.replace(tzinfo=None).isoformat(sep=' ')
        return naive + rest + value.isoformat(sep=' ')[len(naive):]

    def _tz_arg(self) -> str:
        return f', tz={self._tz!r}' if self._aware else ''

    @staticmethod
    def _pandas_scalar(pd, value):
        return pd.Timestamp(value)

    def _pandas_ns(self, pd, ns: int):
        if self._aware:
            return pd.Timestamp(ns, unit='ns', tz='UTC').tz_convert(self._tz)
        return pd.Timestamp(ns, unit='ns')

    # ARITHMETIC

    @_nat_refused
    def __add__(self, other):
        """`+ timedelta`: the instants shifted by every duration (elapsed time: an aware `t + 1 day` is 24 h
        later, across a DST change too)"""
        o = _as_timedelta_interval(other)
        if o is NotImplemented:
            return NotImplemented
        return self._with(self._mi + o._mi)

    __radd__ = __add__

    @_nat_refused
    def __sub__(self, other):
        """`- timedelta` is a `DateTimeInterval`, `- datetime` the `TimeDeltaInterval` of differences"""
        o = _as_timedelta_interval(other)
        if o is not NotImplemented:
            return self._with(self._mi - o._mi)
        o = _as_datetime_interval(other)
        if o is NotImplemented:
            return NotImplemented
        self._check_kinds(o)
        return TimeDeltaInterval._make(self._mi - o._mi)

    @_nat_refused
    def __rsub__(self, other):
        o = _as_datetime_interval(other)
        if o is NotImplemented:
            return NotImplemented
        self._check_kinds(o)
        return TimeDeltaInterval._make(o._mi - self._mi)


def _aware_datetime(us: int, tz: _dt.tzinfo) -> _dt.datetime:
    """
    the instant `us` microseconds after 1970-01-01 UTC as a datetime in tz. `astimezone` builds the UTC datetime
    first, which is past datetime's range near its ends although the local one may not be (`9999-12-31
    23:00-05:00` is 10000-01-01 04:00 UTC), and refuses a tzinfo whose `dst()` is None (`fromutc` needs it);
    there the wall time w with `w - w.utcoffset()` the instant is found from the zone's offsets. OverflowError
    when that wall time is itself outside datetime's range
    """
    delta = _dt.timedelta(microseconds=us)
    try:
        return (_EPOCH_UTC + delta).astimezone(tz)
    except (OverflowError, ValueError):
        pass
    try:  # a first offset: the zone's at the UTC wall time, or at the range's end nearest it
        near = _EPOCH + delta
    except OverflowError:
        near = _dt.datetime.max if us > 0 else _dt.datetime.min
    offset = near.replace(tzinfo=tz).utcoffset()
    for _ in range(4):  # an offset that changes near w converges in a step or two
        if offset is None:
            break
        wall = _EPOCH + (delta + offset)  # OverflowError: the local datetime is outside the range
        for fold in (0, 1):
            d = wall.replace(tzinfo=tz, fold=fold)
            if d.utcoffset() == offset:
                return d
        offset = wall.replace(tzinfo=tz).utcoffset()
    raise ValueError(f'no datetime in {tz!r} reads as the instant {Fraction(us, _US)} s')


def _datetime_bound(x, is_start: bool, closed: bool, tz) -> _Bound:
    if isinstance(x, _Infinity):
        return _infinite_bound(x, is_start, closed)
    if _is_nat(x):
        raise ValueError('NaT is not a point of time')
    if isinstance(x, _dt.datetime):
        aware = _is_aware(x)
        if tz is not None and not aware:
            raise TypeError(f'cannot mix naive and aware ends: tz={tz!r} with the naive {x!r}')
        v = _datetime_seconds(x)
        return _Bound((start_cut if is_start else end_cut)(v, closed), aware, x.tzinfo if aware else None)
    if isinstance(x, _dt.date):
        day_start, next_day = _day_bounds(x, tz)
        # every date bound is a cut below a midnight: the day's own (a closed start, an open end) or the next's
        own_midnight = closed if is_start else not closed
        return _Bound(below(day_start if own_midnight else next_day), tz is not None, tz)
    raise TypeError(f'expected a datetime, date, pandas Timestamp, NEG_INF or POS_INF, got {type(x).__name__}')


def _as_datetime_interval(other):
    """an interval or a scalar of datetimes, for arithmetic (a sentinel is not one)"""
    if isinstance(other, DateTimeInterval):
        return other
    if isinstance(other, _dt.date):
        return DateTimeInterval(other)
    return NotImplemented


# TIMEDELTAS

class TimeDeltaInterval(_TimeInterval):
    """
    a set of durations: zero or more disjoint pieces, each with open or closed ends (see the module
    docstring). `TimeDeltaInterval()` is empty, `TimeDeltaInterval(d)` the point d, and
    `TimeDeltaInterval(a, b, start_closed=..., end_closed=...)` one piece; each bound a timedelta, a pandas
    `Timedelta` or a sentinel. with a pandas `Timedelta` x on the left, `x % A` and `divmod(x, A)` are
    pandas' own `x - (x // A) * A`, a sound but wider set: write `TimeDeltaInterval(x) % A`

    >>> import datetime
    >>> hour = datetime.timedelta(hours=1)
    >>> print(TimeDeltaInterval(-hour, 2 * hour) / 3)
    [-0:20:00, 0:40:00]
    >>> TimeDeltaInterval(hour, 3 * hour) / hour
    MultiInterval.parse('[1, 3]')
    """
    __slots__ = ()

    def __init__(self, start=None, end=None, *, start_closed: bool = True, end_closed: bool = True):
        if start is None:
            if end is not None:
                raise ValueError('an end without a start')
            object.__setattr__(self, '_mi', MultiInterval())
            return
        if end is None:
            if start_closed != end_closed:
                raise ValueError(f'half-open degenerate interval at {start!r}')
            end = start
        s = _timedelta_bound(start, True, start_closed)
        e = _timedelta_bound(end, False, end_closed)
        object.__setattr__(self, '_mi', MultiInterval.from_cuts(_piece_cuts(s, e, start, end)))

    @classmethod
    def _make(cls, mi: MultiInterval) -> 'TimeDeltaInterval':
        out = object.__new__(cls)
        object.__setattr__(out, '_mi', mi)
        return out

    @classmethod
    def from_seconds(cls, seconds) -> 'TimeDeltaInterval':
        """
        the raw constructor: a `MultiInterval` (or a number) of seconds, a float end taken exactly

        >>> TimeDeltaInterval.from_seconds(MultiInterval(0, 90)).sup
        datetime.timedelta(seconds=90)
        """
        return cls._make(_as_seconds(seconds))

    def __reduce__(self):
        return type(self)._make, (self._mi,)

    _scalar_name = 'timedelta'
    _pandas_name = 'Timedelta'

    def _coerce(self, other):
        if isinstance(other, TimeDeltaInterval):
            return other
        if isinstance(other, (_Infinity, _dt.timedelta)) or _is_nat(other):
            return TimeDeltaInterval(other)
        return NotImplemented

    def _with(self, mi: MultiInterval, *others) -> 'TimeDeltaInterval':
        return self._make(mi)

    def __eq__(self, other):
        if not isinstance(other, TimeDeltaInterval):
            return NotImplemented
        return self._mi == other._mi

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = _TimeInterval.__hash__

    def _from_us(self, us: int) -> _dt.timedelta:
        return _dt.timedelta(microseconds=us)

    def _str_value(self, value: _dt.timedelta, rest: str = '') -> str:
        """python's text, signed as a whole (`-0:20:00`, not `-1 day, 23:40:00`) and without the comma"""
        if value < _dt.timedelta(0):
            return '-' + str(-value).replace(', ', ' ') + rest
        return str(value).replace(', ', ' ') + rest

    @staticmethod
    def _pandas_scalar(pd, value):
        return pd.Timedelta(value)

    @staticmethod
    def _pandas_ns(pd, ns: int):
        return pd.Timedelta(ns, unit='ns')

    # ARITHMETIC

    @_nat_refused
    def __add__(self, other):
        """`+ timedelta` is a `TimeDeltaInterval`, `+ datetime` a `DateTimeInterval`"""
        o = _as_timedelta_interval(other)
        if o is not NotImplemented:
            return self._make(self._mi + o._mi)
        o = _as_datetime_interval(other)
        if o is NotImplemented:
            return NotImplemented
        return o._with(o._mi + self._mi)

    __radd__ = __add__

    @_nat_refused
    def __sub__(self, other):
        o = _as_timedelta_interval(other)
        if o is NotImplemented:
            return NotImplemented
        return self._make(self._mi - o._mi)

    @_nat_refused
    def __rsub__(self, other):
        """`timedelta - self` is a `TimeDeltaInterval`, `datetime - self` a `DateTimeInterval`"""
        o = _as_timedelta_interval(other)
        if o is not NotImplemented:
            return self._make(o._mi - self._mi)
        o = _as_datetime_interval(other)
        if o is NotImplemented:
            return NotImplemented
        return o._with(o._mi - self._mi)

    @_nat_refused
    def __mul__(self, other):
        """times a real number or a `MultiInterval`, exactly"""
        f = _factor(other)
        return NotImplemented if f is NotImplemented else self._make(self._mi * f)

    __rmul__ = __mul__

    @_nat_refused
    def __truediv__(self, other):
        """by a duration: the `MultiInterval` of ratios; by a real number or a `MultiInterval`: a duration"""
        o = _as_timedelta_interval(other)
        if o is not NotImplemented:
            return self._mi / o._mi
        f = _factor(other)
        return NotImplemented if f is NotImplemented else self._make(self._mi / f)

    @_nat_refused
    def __rtruediv__(self, other):
        o = _as_timedelta_interval(other)
        return NotImplemented if o is NotImplemented else o._mi / self._mi

    @_nat_refused
    def __floordiv__(self, other):
        """by a duration: the integers `floor(a / b)`, a `MultiInterval` (as python's `timedelta // timedelta`)"""
        o = _as_timedelta_interval(other)
        return NotImplemented if o is NotImplemented else self._mi // o._mi

    @_nat_refused
    def __rfloordiv__(self, other):
        o = _as_timedelta_interval(other)
        return NotImplemented if o is NotImplemented else o._mi // self._mi

    @_nat_refused
    def __mod__(self, other):
        """by a duration: python's floor-mod over every pair, a duration (as `timedelta % timedelta`)"""
        o = _as_timedelta_interval(other)
        return NotImplemented if o is NotImplemented else self._make(self._mi % o._mi)

    @_nat_refused
    def __rmod__(self, other):
        o = _as_timedelta_interval(other)
        return NotImplemented if o is NotImplemented else self._make(o._mi % self._mi)

    @_nat_refused
    def __divmod__(self, other):
        o = _as_timedelta_interval(other)
        if o is NotImplemented:
            return NotImplemented
        q, r = divmod(self._mi, o._mi)
        return q, self._make(r)

    @_nat_refused
    def __rdivmod__(self, other):
        o = _as_timedelta_interval(other)
        if o is NotImplemented:
            return NotImplemented
        q, r = divmod(o._mi, self._mi)
        return q, self._make(r)

    def __neg__(self) -> 'TimeDeltaInterval':
        return self._make(-self._mi)

    def __pos__(self) -> 'TimeDeltaInterval':
        return self

    def __abs__(self) -> 'TimeDeltaInterval':
        return self._make(abs(self._mi))


def _timedelta_bound(x, is_start: bool, closed: bool) -> _Bound:
    if isinstance(x, _Infinity):
        return _infinite_bound(x, is_start, closed)
    if _is_nat(x):
        raise ValueError('NaT is not a duration')
    if isinstance(x, _dt.timedelta):
        v = _timedelta_seconds(x)
        return _Bound((start_cut if is_start else end_cut)(v, closed), None, None)
    raise TypeError(f'expected a timedelta, pandas Timedelta, NEG_INF or POS_INF, got {type(x).__name__}')


def _as_timedelta_interval(other):
    """an interval or a scalar of durations, for arithmetic (a sentinel is not one)"""
    if isinstance(other, TimeDeltaInterval):
        return other
    if isinstance(other, _dt.timedelta):
        return TimeDeltaInterval(other)
    return NotImplemented


__all__ = ['DateTimeInterval', 'TimeDeltaInterval', 'NEG_INF', 'POS_INF']
