import datetime
import math
import operator
from numbers import Real
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd

from multi_interval import MultiInterval

DATETIME_LIKE = Union['DateTimeInterval', datetime.datetime, datetime.date, pd.Timestamp]
TIMEDELTA_LIKE = Union['TimeDeltaInterval', datetime.timedelta, pd.Timedelta]


def _datetime_interval(dti: DATETIME_LIKE):
    if isinstance(dti, DateTimeInterval):
        return dti.interval
    elif isinstance(dti, (datetime.datetime, datetime.date, pd.Timestamp)):
        return DateTimeInterval(dti).interval
    else:
        raise TypeError(dti)


def _timedelta_interval(tdi: TIMEDELTA_LIKE):
    if isinstance(tdi, TimeDeltaInterval):
        return tdi.interval
    elif isinstance(tdi, (datetime.timedelta, pd.Timedelta)):
        return TimeDeltaInterval(tdi).interval
    else:
        raise TypeError(tdi)


class DateTimeInterval:
    """
    multiple contiguous time intervals
    possibly degenerate but never infinite
    refer to MultiInterval for more detailed explanations
    """
    interval: MultiInterval

    def __init__(self,
                 start: Optional[Union[datetime.datetime, datetime.date]] = None,
                 end: Optional[Union[datetime.datetime, datetime.date]] = None,
                 *,
                 start_closed: Optional[bool] = True,
                 end_closed: Optional[bool] = True,
                 # day_first: bool = True,  # WARNING: may produce unexpected results
                 # year_first: bool = False,  # WARNING: may produce unexpected results
                 ):
        # using strings to input dates is possible but NOT recommended
        # if dates are provided as a string, it should be "dd/mm/yyyy HH:MM" or "dd mmm yyyy II:MM am/pm"
        # if an iso8601-formatted date is to be parsed, set day_first=False and year_first=True
        # strings are always parsed into datetimes, never into dates, so the interval will not span a full day

        # handle nan
        if pd.isna(end):
            end = None
        if pd.isna(start):
            start, end = end, None

        # for printing in error message below
        start_str = str(start)
        end_str = str(end)

        # convert start to unix datetime
        _end = None
        if start is not None:
            # convert to datetime
            if isinstance(start, pd.Timestamp):
                start = start.to_pydatetime()
            # elif isinstance(start, str):
            #     start = dateutil.parser.parse(start, dayfirst=day_first, yearfirst=year_first)

            # already a datetime / timestamp, do nothing
            if isinstance(start, datetime.datetime):
                start = start.timestamp()

            # start (and end) of day
            elif isinstance(start, datetime.date):
                _end = datetime.datetime.combine(start, datetime.time.max).timestamp()  # end of day
                start = datetime.datetime.combine(start, datetime.time.min).timestamp()  # start of day
            else:
                raise TypeError(start)

        # convert end to unix datetime
        if end is not None:
            # convert to datetime
            if isinstance(end, pd.Timestamp):
                end = end.to_pydatetime()
            # elif isinstance(end, str):
            #     end = dateutil.parser.parse(end, dayfirst=day_first, yearfirst=year_first)

            if isinstance(end, datetime.datetime):
                # not the most elegant way of doing this, but probably the most obvious
                # also, an end date that is open doesn't really make sense with this method
                # because that only excludes the last microsecond of the day
                # logically speaking, [Tuesday to Saturday) == [Tuesday to Friday] == (Monday to Friday]
                if end.hour == 0 and end.minute == 0 and end.second == 0 and end.microsecond == 0:
                    end = end.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
                elif end.minute == 0 and end.second == 0 and end.microsecond == 0:
                    end = end.replace(minute=59, second=59, microsecond=999999).timestamp()
                elif end.second == 0 and end.microsecond == 0:
                    end = end.replace(second=59, microsecond=999999).timestamp()
                elif end.microsecond == 0:
                    end = end.replace(microsecond=999999).timestamp()
                else:
                    end = end.timestamp()

            # end of day
            elif isinstance(end, datetime.date):
                end = datetime.datetime.combine(end, datetime.time.max).timestamp()

            # wrong type
            else:
                raise TypeError(end)

            # flipped interval
            if end < start:
                raise ValueError(f'start ({start_str}) cannot be after end ({end_str})')

        self.interval = MultiInterval(start=start,
                                      end=end or _end,
                                      start_closed=start_closed,
                                      end_closed=end_closed)

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return self.interval.is_empty

    @property
    def is_contiguous(self) -> bool:
        return self.interval.is_contiguous

    @property
    def is_degenerate(self) -> bool:
        return self.interval.is_degenerate

    @property
    def infimum(self) -> Optional[datetime.datetime]:
        if not self.is_empty:
            return datetime.datetime.fromtimestamp(float(self.interval.infimum))

    @property
    def supremum(self) -> Optional[datetime.datetime]:
        if not self.is_empty:
            return datetime.datetime.fromtimestamp(float(self.interval.supremum))

    @property
    def degenerate_points(self) -> Set[datetime.datetime]:
        return set([datetime.datetime.fromtimestamp(float(point)) for point in self.interval.degenerate_points])

    @property
    def closed_hull(self) -> Optional['DateTimeInterval']:
        if not self.is_empty:
            return DateTimeInterval(start=datetime.datetime.fromtimestamp(float(self.interval.infimum)),
                                    end=datetime.datetime.fromtimestamp(float(self.interval.supremum)))

    @property
    def cardinality(self) -> Tuple[float, int]:
        _rays, _len, _points = self.interval.cardinality
        assert _rays == 0
        return _len, _points

    @property
    def total_seconds(self) -> float:
        _len, _points = self.cardinality
        return _len

    @property
    def total_duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.total_seconds)

    @property
    def contiguous_intervals(self) -> List['DateTimeInterval']:
        out = []
        for idx in range(0, len(self.interval.endpoints), 2):
            out.append(DateTimeInterval(start=datetime.datetime.fromtimestamp(float(self.interval.endpoints[idx][0])),
                                        end=datetime.datetime.fromtimestamp(float(self.interval.endpoints[idx + 1][0])),
                                        start_closed=self.interval.endpoints[idx][1] == 0,
                                        end_closed=self.interval.endpoints[idx + 1][1] == 0))
        return out

    # COMPARISONS

    def __lt__(self, other: DATETIME_LIKE) -> bool:
        return operator.lt(self.interval, _datetime_interval(other))

    def __le__(self, other: DATETIME_LIKE) -> bool:
        return operator.le(self.interval, _datetime_interval(other))

    def __eq__(self, other: DATETIME_LIKE) -> bool:
        return operator.eq(self.interval, _datetime_interval(other))

    def __ne__(self, other: DATETIME_LIKE) -> bool:
        return operator.ne(self.interval, _datetime_interval(other))

    def __ge__(self, other: DATETIME_LIKE) -> bool:
        return operator.ge(self.interval, _datetime_interval(other))

    def __gt__(self, other: DATETIME_LIKE) -> bool:
        return operator.gt(self.interval, _datetime_interval(other))

    # UTILITY

    def copy(self) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.copy()
        return out

    def __sizeof__(self) -> int:
        return self.interval.__sizeof__()  # probably correct?

    def __getitem__(self, item: Union[slice, DATETIME_LIKE]) -> 'DateTimeInterval':
        if isinstance(item, DateTimeInterval):
            return self.intersection(item)

        elif isinstance(item, slice):
            if item.step is not None:
                raise ValueError(item)

            if item.start is None:
                start = -math.inf
            elif isinstance(item.start, (datetime.datetime, datetime.date, pd.Timestamp)):
                start = DateTimeInterval(item.start).infimum.timestamp()
            else:
                raise TypeError(item.start)

            if item.stop is None:
                end = math.inf
            elif isinstance(item.stop, (datetime.datetime, datetime.date, pd.Timestamp)):
                end = DateTimeInterval(item.stop).supremum.timestamp()  # gotta round up datetime.date
            else:
                raise TypeError(item.stop)

            out = DateTimeInterval()
            out.interval = self.interval.intersection(MultiInterval(start=start,
                                                                    end=end,
                                                                    start_closed=not math.isinf(start),
                                                                    end_closed=not math.isinf(end)))
            return out

        elif isinstance(item, (datetime.datetime, datetime.date, pd.Timestamp)):
            if item in self:
                return DateTimeInterval(item)
            else:
                return DateTimeInterval()

        else:
            raise TypeError

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: DATETIME_LIKE) -> bool:
        return self.interval.isdisjoint(_datetime_interval(other))

    def issubset(self, other: DATETIME_LIKE) -> bool:
        return self.interval.issubset(_datetime_interval(other))

    def issuperset(self, other: DATETIME_LIKE) -> bool:
        return self.interval.issuperset(_datetime_interval(other))

    # SET: BOOLEAN ALGEBRA (INPLACE)

    def update(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.update(*[_datetime_interval(_other) for _other in other])
        return self

    def intersection_update(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.intersection_update(*[_datetime_interval(_other) for _other in other])
        return self

    def difference_update(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.difference_update(*[_datetime_interval(_other) for _other in other])
        return self

    def symmetric_difference_update(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.symmetric_difference_update(*[_datetime_interval(_other) for _other in other])
        return self

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        return self.copy().update(*other)

    def intersection(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        return self.copy().intersection_update(*other)

    def difference(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        return self.copy().difference_update(*other)

    def symmetric_difference(self, *other: DATETIME_LIKE) -> 'DateTimeInterval':
        return self.copy().symmetric_difference_update(*other)

    # INTERVAL OPERATIONS (INPLACE)

    def expand(self, distance: Union[datetime.timedelta, pd.Timedelta]) -> 'DateTimeInterval':
        if isinstance(distance, (datetime.timedelta, pd.Timedelta)):
            self.interval.expand(distance.total_seconds())
            return self
        else:
            raise TypeError(distance)

    # INTERVAL OPERATIONS

    def __contains__(self, other: DATETIME_LIKE) -> bool:
        return _datetime_interval(other) in self.interval

    def overlapping(self, other: DATETIME_LIKE, or_adjacent: bool = False) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.overlapping(_datetime_interval(other), or_adjacent=or_adjacent)
        return out

    def overlaps(self, other: DATETIME_LIKE, or_adjacent: bool = False) -> bool:
        return self.interval.overlaps(_datetime_interval(other), or_adjacent=or_adjacent)

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: TIMEDELTA_LIKE) -> 'DateTimeInterval':
        if isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = DateTimeInterval()
            out.interval = self.interval + _timedelta_interval(other)
            return out
        else:
            raise TypeError(other)

    def __radd__(self, other: TIMEDELTA_LIKE) -> 'DateTimeInterval':
        if isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = DateTimeInterval()
            out.interval = _timedelta_interval(other) + self.interval
            return out
        else:
            raise TypeError(other)

    def __sub__(self, other: Union[DATETIME_LIKE, TIMEDELTA_LIKE]) -> Union['DateTimeInterval', 'TimeDeltaInterval']:
        if isinstance(other, (DateTimeInterval, datetime.datetime, datetime.date, pd.Timestamp)):
            out = TimeDeltaInterval()
            out.interval = self.interval - _datetime_interval(other)
            return out
        elif isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = DateTimeInterval()
            out.interval = self.interval - _timedelta_interval(other)
            return out
        else:
            raise TypeError(other)

    def __rsub__(self, other: DATETIME_LIKE) -> 'TimeDeltaInterval':
        if isinstance(other, (DateTimeInterval, datetime.datetime, datetime.date, pd.Timestamp)):
            out = TimeDeltaInterval()
            out.interval = _datetime_interval(other) - self.interval
            return out
        else:
            raise TypeError(other)

    def __str__(self) -> str:
        def _interval_to_str(start_tuple, end_tuple):
            assert start_tuple <= end_tuple, (start_tuple, end_tuple)

            # unpack
            start_unix_timestamp, start_epsilon = start_tuple
            end_unix_timestamp, end_epsilon = end_tuple
            assert start_epsilon in {1, 0}, (start_tuple, end_tuple)
            assert end_epsilon in {0, -1}, (start_tuple, end_tuple)
            assert not math.isinf(start_unix_timestamp)
            assert not math.isinf(end_unix_timestamp)

            start: datetime.datetime = datetime.datetime.fromtimestamp(start_unix_timestamp)
            end: datetime.datetime = datetime.datetime.fromtimestamp(end_unix_timestamp)
            if start.time() == datetime.time.min and end.date() > start.date():
                _start = start.strftime("%Y-%m-%d")
            else:
                _start = start.strftime("%Y-%m-%d %H:%M")

            if end.time() == datetime.time.max and end.date() > start.date():
                _end = end.strftime("%Y-%m-%d")
            else:
                _end = end.strftime("%Y-%m-%d %H:%M")

            # degenerate interval
            if start == end:
                assert start_epsilon == 0
                assert end_epsilon == 0

            # possibly non-degenerate (e.g. full day)
            if _start == _end and start_epsilon == 0 and end_epsilon == 0:
                return f'[{_start}]'

            return f'{"(" if start_epsilon else "["}{_start} to {_end}{")" if end_epsilon else "]"}'

        # null set: {}
        if self.is_empty:
            return '{}'

        # single contiguous interval: [x, y)
        elif len(self.interval.endpoints) == 2:
            return _interval_to_str(*self.interval.endpoints)  # handles degenerate intervals too

        # multiple intervals: { [x, y) | [z] | (a, b) }
        else:
            str_intervals = []
            for idx in range(0, len(self.interval.endpoints), 2):
                str_intervals.append(_interval_to_str(self.interval.endpoints[idx], self.interval.endpoints[idx + 1]))

            if len(str_intervals) == 1:
                return str_intervals[0]

            return f'{{ {" ; ".join(str_intervals)} }}'

    __repr__ = __str__


class TimeDeltaInterval:
    interval: MultiInterval

    def __init__(self,
                 start: Optional[Union[datetime.timedelta, pd.Timedelta]] = None,
                 end: Optional[Union[datetime.timedelta, pd.Timedelta]] = None,
                 *,
                 start_closed: Optional[bool] = True,
                 end_closed: Optional[bool] = True
                 ):

        if start is not None:
            # already a datetime / timestamp, do nothing
            if isinstance(start, (datetime.timedelta, pd.Timedelta)):
                start = start.total_seconds()
            else:
                raise TypeError(start)
        if end is not None:
            # already a datetime / timestamp, do nothing
            if isinstance(end, (datetime.timedelta, pd.Timedelta)):
                end = end.total_seconds()
            else:
                raise TypeError(end)

        self.interval = MultiInterval(start=start,
                                      end=end,
                                      start_closed=start_closed,
                                      end_closed=end_closed)

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return self.interval.is_empty

    @property
    def is_contiguous(self) -> bool:
        return self.interval.is_contiguous

    @property
    def is_degenerate(self) -> bool:
        return self.interval.is_degenerate

    @property
    def infimum(self) -> Optional[datetime.timedelta]:
        if not self.is_empty:
            return datetime.timedelta(seconds=float(self.interval.infimum))

    @property
    def supremum(self) -> Optional[datetime.timedelta]:
        if not self.is_empty:
            return datetime.timedelta(seconds=float(self.interval.supremum))

    @property
    def degenerate_points(self) -> Set[datetime.timedelta]:
        return set([datetime.timedelta(seconds=float(point)) for point in self.interval.degenerate_points])

    @property
    def closed_hull(self) -> Optional['TimeDeltaInterval']:
        if not self.is_empty:
            return TimeDeltaInterval(start=datetime.timedelta(seconds=float(self.interval.infimum)),
                                     end=datetime.timedelta(seconds=float(self.interval.supremum)))

    @property
    def cardinality(self) -> Tuple[float, int]:
        _rays, _len, _points = self.interval.cardinality
        assert _rays == 0
        return _len, _points

    @property
    def contiguous_intervals(self) -> List['TimeDeltaInterval']:
        out = []
        for idx in range(0, len(self.interval.endpoints), 2):
            out.append(TimeDeltaInterval(start=datetime.timedelta(seconds=float(self.interval.endpoints[idx][0])),
                                         end=datetime.timedelta(seconds=float(self.interval.endpoints[idx + 1][0])),
                                         start_closed=self.interval.endpoints[idx][1] == 0,
                                         end_closed=self.interval.endpoints[idx + 1][1] == 0))
        return out

    # COMPARISONS

    def __lt__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.lt(self.interval, _timedelta_interval(other))

    def __le__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.le(self.interval, _timedelta_interval(other))

    def __eq__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.eq(self.interval, _timedelta_interval(other))

    def __ne__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.ne(self.interval, _timedelta_interval(other))

    def __ge__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.ge(self.interval, _timedelta_interval(other))

    def __gt__(self, other: TIMEDELTA_LIKE) -> bool:
        return operator.gt(self.interval, _timedelta_interval(other))

    # UTILITY

    def copy(self) -> 'TimeDeltaInterval':
        out = TimeDeltaInterval()
        out.interval = self.interval.copy()
        return out

    def __sizeof__(self) -> int:
        return self.interval.__sizeof__()  # probably correct?

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: TIMEDELTA_LIKE) -> bool:
        return self.interval.isdisjoint(_timedelta_interval(other))

    def issubset(self, other: TIMEDELTA_LIKE) -> bool:
        return self.interval.issubset(_timedelta_interval(other))

    def issuperset(self, other: TIMEDELTA_LIKE) -> bool:
        return self.interval.issuperset(_timedelta_interval(other))

    # SET: ITEMS (INPLACE)

    def add(self, other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.add(_timedelta_interval(other))
        return self

    def clear(self) -> 'TimeDeltaInterval':
        self.interval.clear()
        return self

    def discard(self, other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.discard(_timedelta_interval(other))
        return self

    def pop(self) -> 'TimeDeltaInterval':
        out = TimeDeltaInterval()
        out.interval = self.interval.pop()
        return out

    def remove(self, other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.remove(_timedelta_interval(other))
        return self

    # SET: BOOLEAN ALGEBRA (INPLACE)

    def update(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.update(*[_timedelta_interval(_other) for _other in other])
        return self

    def intersection_update(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.intersection_update(*[_timedelta_interval(_other) for _other in other])
        return self

    def difference_update(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.difference_update(*[_timedelta_interval(_other) for _other in other])
        return self

    def symmetric_difference_update(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        self.interval.symmetric_difference_update(*[_timedelta_interval(_other) for _other in other])
        return self

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        return self.copy().update(*other)

    def intersection(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        return self.copy().intersection_update(*other)

    def difference(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        return self.copy().difference_update(*other)

    def symmetric_difference(self, *other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        return self.copy().symmetric_difference_update(*other)

    # INTERVAL OPERATIONS (INPLACE)

    def expand(self, distance: Union[datetime.timedelta, pd.Timedelta], inplace=False) -> 'TimeDeltaInterval':
        if not inplace:
            return self.copy().expand(distance, inplace=True)

        if isinstance(distance, (datetime.timedelta, pd.Timedelta)):
            self.interval.expand(distance.total_seconds(), inplace=True)
            return self
        else:
            raise TypeError(distance)

    # INTERVAL OPERATIONS

    def __contains__(self, other: TIMEDELTA_LIKE) -> bool:
        return _timedelta_interval(other) in self.interval

    def overlapping(self, other: TIMEDELTA_LIKE, or_adjacent: bool = False) -> 'TimeDeltaInterval':
        out = TimeDeltaInterval()
        out.interval = self.interval.overlapping(_timedelta_interval(other), or_adjacent=or_adjacent)
        return out

    def overlaps(self, other: TIMEDELTA_LIKE, or_adjacent: bool = False) -> bool:
        return self.interval.overlaps(_timedelta_interval(other), or_adjacent=or_adjacent)

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: Union[DATETIME_LIKE, TIMEDELTA_LIKE]) -> Union['DateTimeInterval', 'TimeDeltaInterval']:
        if isinstance(other, (DateTimeInterval, datetime.datetime, datetime.date, pd.Timestamp)):
            out = DateTimeInterval()
            out.interval = self.interval + _datetime_interval(other)
            return out
        elif isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = TimeDeltaInterval()
            out.interval = self.interval + _timedelta_interval(other)
            return out
        else:
            raise TypeError(other)

    def __radd__(self, other: Union[DATETIME_LIKE, TIMEDELTA_LIKE]) -> Union['DateTimeInterval', 'TimeDeltaInterval']:
        if isinstance(other, (DateTimeInterval, datetime.datetime, datetime.date, pd.Timestamp)):
            out = DateTimeInterval()
            out.interval = _datetime_interval(other) + self.interval
            return out
        elif isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = TimeDeltaInterval()
            out.interval = _timedelta_interval(other) + self.interval
            return out
        else:
            raise TypeError(other)

    def __sub__(self, other: Union[DATETIME_LIKE, TIMEDELTA_LIKE]) -> Union['DateTimeInterval', 'TimeDeltaInterval']:
        if isinstance(other, (DateTimeInterval, datetime.datetime, datetime.date, pd.Timestamp)):
            out = DateTimeInterval()
            out.interval = self.interval - _datetime_interval(other)
            return out
        elif isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = TimeDeltaInterval()
            out.interval = self.interval - _timedelta_interval(other)
            return out
        else:
            raise TypeError(other)

    def __rsub__(self, other: TIMEDELTA_LIKE) -> 'TimeDeltaInterval':
        if isinstance(other, (TimeDeltaInterval, datetime.timedelta, pd.Timedelta)):
            out = TimeDeltaInterval()
            out.interval = _timedelta_interval(other) - self.interval
            return out
        else:
            raise TypeError(other)

    def __mul__(self, other: Real) -> 'TimeDeltaInterval':
        if not isinstance(other, Real):
            raise TypeError(other)
        out = TimeDeltaInterval()
        out.interval = self.interval * other
        return out

    def __rmul__(self, other: Real) -> 'TimeDeltaInterval':
        if not isinstance(other, Real):
            raise TypeError(other)
        out = TimeDeltaInterval()
        out.interval = other * self.interval
        return out

    def __truediv__(self, other: Real) -> 'TimeDeltaInterval':
        if not isinstance(other, Real):
            raise TypeError(other)
        out = TimeDeltaInterval()
        out.interval = self.interval / other
        return out

    def __str__(self) -> str:
        def _interval_to_str(start_tuple, end_tuple):
            assert start_tuple <= end_tuple, (start_tuple, end_tuple)

            # unpack
            start, start_epsilon = start_tuple
            end, end_epsilon = end_tuple
            assert start_epsilon in {1, 0}, (start_tuple, end_tuple)
            assert end_epsilon in {0, -1}, (start_tuple, end_tuple)
            assert not math.isinf(start)
            assert not math.isinf(end)

            # degenerate interval
            if start == end:
                assert start_epsilon == 0
                assert end_epsilon == 0
                return f'[{datetime.timedelta(seconds=start)}]'

            return f'{"(" if start_epsilon else "["}' \
                   f'{datetime.timedelta(seconds=start)}, ' \
                   f'{datetime.timedelta(seconds=end)}' \
                   f'{")" if end_epsilon else "]"}'

        # null set: {}
        if self.is_empty:
            return '{}'

        # single contiguous interval: [x, y)
        elif len(self.interval.endpoints) == 2:
            return _interval_to_str(*self.interval.endpoints)  # handles degenerate intervals too

        # multiple intervals: { [x, y) | [z] | (a, b) }
        else:
            str_intervals = []
            for idx in range(0, len(self.interval.endpoints), 2):
                str_intervals.append(_interval_to_str(self.interval.endpoints[idx], self.interval.endpoints[idx + 1]))

            if len(str_intervals) == 1:
                return str_intervals[0]

            return f'{{ {" , ".join(str_intervals)} }}'


if __name__ == '__main__':
    x = DateTimeInterval(datetime.date(2018, 9, 1))
    print(x)
    print(x.update(x + datetime.timedelta(999)))  # update in-place
    print(x[datetime.date(2018, 1, 2):datetime.date(2018, 8, 9)])
    print(x.intersection(DateTimeInterval(datetime.date(2018, 9, 1), datetime.date(2019, 5, 30))))
