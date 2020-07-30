import datetime
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
    elif isinstance(dti, (datetime.date, pd.Timestamp)):
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
    interval: MultiInterval

    def __init__(self,
                 start: Optional[Union[datetime.datetime, datetime.date]] = None,
                 end: Optional[Union[datetime.datetime, datetime.date]] = None,
                 *,
                 start_closed: Optional[bool] = True,
                 end_closed: Optional[bool] = True
                 ):

        _end = None
        if start is not None:
            # already a datetime / timestamp, do nothing
            if isinstance(start, pd.Timestamp):
                start = start.to_pydatetime()
            if isinstance(start, datetime.datetime):
                start = start.timestamp()

            # start (and end) of day
            elif isinstance(start, datetime.date):
                _end = datetime.datetime.combine(start, datetime.time.max).timestamp()  # end of day
                start = datetime.datetime.combine(start, datetime.time.min).timestamp()  # start of day
            else:
                raise TypeError(start)

        if end is not None:
            # round up to end of day / hour / min / etc
            if isinstance(end, pd.Timestamp):
                end = end.to_pydatetime()
            if isinstance(end, datetime.datetime):
                # not the most elegant way of doing this, but probably the most obvious
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
        raise NotImplementedError

    @property
    def supremum(self) -> Optional[datetime.datetime]:
        raise NotImplementedError

    @property
    def degenerate_points(self) -> Set[datetime.datetime]:
        raise NotImplementedError

    @property
    def closed_hull(self) -> Optional['DateTimeInterval']:
        raise NotImplementedError

    @property
    def cardinality(self) -> Tuple[float, int]:
        _rays, _len, _points = self.interval.cardinality
        assert _rays == 0
        return _len, _points

    @property
    def contiguous_intervals(self) -> List['DateTimeInterval']:
        raise NotImplementedError

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
        raise NotImplementedError

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: DATETIME_LIKE) -> bool:
        return self.interval.isdisjoint(_datetime_interval(other))

    def issubset(self, other: DATETIME_LIKE) -> bool:
        return self.interval.issubset(_datetime_interval(other))

    def issuperset(self, other: DATETIME_LIKE) -> bool:
        return self.interval.issuperset(_datetime_interval(other))

    # SET: ITEMS (INPLACE)

    def add(self, other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.add(_datetime_interval(other))
        return self

    def clear(self) -> 'DateTimeInterval':
        self.interval.clear()
        return self

    def discard(self, other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.discard(_datetime_interval(other))
        return self

    def pop(self) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.pop()
        return out

    def remove(self, other: DATETIME_LIKE) -> 'DateTimeInterval':
        self.interval.remove(_datetime_interval(other))
        return self

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

    def expand(self, distance: Union[datetime.timedelta]) -> 'DateTimeInterval':
        raise NotImplementedError

    # INTERVAL OPERATIONS

    def __contains__(self, other: DATETIME_LIKE) -> bool:
        raise NotImplementedError

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

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


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
        raise NotImplementedError

    @property
    def supremum(self) -> Optional[datetime.timedelta]:
        raise NotImplementedError

    @property
    def degenerate_points(self) -> Set[datetime.timedelta]:
        raise NotImplementedError

    @property
    def closed_hull(self) -> Optional['TimeDeltaInterval']:
        raise NotImplementedError

    @property
    def cardinality(self) -> Tuple[float, int]:
        _rays, _len, _points = self.interval.cardinality
        assert _rays == 0
        return _len, _points

    @property
    def contiguous_intervals(self) -> List['TimeDeltaInterval']:
        raise NotImplementedError

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

    def __getitem__(self, item: Union[slice, TIMEDELTA_LIKE]) -> 'TimeDeltaInterval':
        raise NotImplementedError

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

    def expand(self, distance: Union[datetime.timedelta]) -> 'TimeDeltaInterval':
        raise NotImplementedError

    # INTERVAL OPERATIONS

    def __contains__(self, other: TIMEDELTA_LIKE) -> bool:
        raise NotImplementedError

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

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
