import datetime
import operator
from collections import Callable
from numbers import Real
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd

from multi_interval import MultiInterval

DATE_LIKE = Union['DateTimeInterval', datetime.datetime, datetime.date]


def _get_interval(dti: DATE_LIKE):
    if isinstance(dti, datetime.date):
        dti = DateTimeInterval(dti)
    if not isinstance(dti, DateTimeInterval):
        raise TypeError(dti)

    return dti.interval


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

    # GENERIC

    def __compare(self, func: Callable, other: DATE_LIKE) -> bool:
        if isinstance(other, DateTimeInterval):
            return func(self.interval, other.interval)
        elif isinstance(other, datetime.date):
            return func(self.interval, DateTimeInterval(other))
        else:
            raise TypeError(other)

    def apply_monotonic_unary_function(self, func: Callable, inplace: bool = False) -> 'DateTimeInterval':
        # by default, do this to a copy
        if not inplace:
            return self.copy().apply_monotonic_unary_function(func, inplace=True)

        self.interval.apply_monotonic_unary_function(func, inplace=True)
        return self

    def apply_monotonic_binary_function(self,
                                        func: Callable,
                                        other: DATE_LIKE,
                                        right_hand_side: bool = False,
                                        inplace: bool = False,
                                        allowed_other_type=Real
                                        ) -> 'DateTimeInterval':

        # by default, do this to a copy
        if not inplace:
            return self.copy().apply_monotonic_binary_function(func,
                                                               other,
                                                               right_hand_side=right_hand_side,
                                                               inplace=True,
                                                               allowed_other_type=allowed_other_type)

        if not isinstance(other, allowed_other_type):
            raise TypeError(other)

        self.interval.apply_monotonic_binary_function(func, other, right_hand_side=right_hand_side, inplace=True)
        return self

    # COMPARISONS

    def __lt__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.lt, other)

    def __le__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.le, other)

    def __eq__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.eq, other)

    def __ne__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.ne, other)

    def __ge__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.ge, other)

    def __gt__(self, other: DATE_LIKE) -> bool:
        return self.__compare(operator.gt, other)

    # UTILITY

    def copy(self) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.copy()
        return out

    def __sizeof__(self) -> int:
        return self.interval.__sizeof__()  # probably correct?

    def __getitem__(self, item: Union[slice, DATE_LIKE]) -> 'DateTimeInterval':
        raise NotImplementedError

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: DATE_LIKE) -> bool:
        return self.interval.isdisjoint(_get_interval(other))

    def issubset(self, other: DATE_LIKE) -> bool:
        return self.interval.issubset(_get_interval(other))

    def issuperset(self, other: DATE_LIKE) -> bool:
        return self.interval.issuperset(_get_interval(other))

    # SET: ITEMS (INPLACE)

    def add(self, other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.add(_get_interval(other))
        return self

    def clear(self) -> 'DateTimeInterval':
        self.interval.clear()
        return self

    def discard(self, other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.discard(_get_interval(other))
        return self

    def pop(self) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.pop()
        return out

    def remove(self, other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.remove(_get_interval(other))
        return self

    # SET: BOOLEAN ALGEBRA (INPLACE)

    def update(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.update(*[_get_interval(_other) for _other in other])
        return self

    def intersection_update(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.intersection_update(*[_get_interval(_other) for _other in other])
        return self

    def difference_update(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.difference_update(*[_get_interval(_other) for _other in other])
        return self

    def symmetric_difference_update(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        self.interval.symmetric_difference_update(*[_get_interval(_other) for _other in other])
        return self

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        return self.copy().update(*other)

    def intersection(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        return self.copy().intersection_update(*other)

    def difference(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        return self.copy().difference_update(*other)

    def symmetric_difference(self, *other: DATE_LIKE) -> 'DateTimeInterval':
        return self.copy().symmetric_difference_update(*other)

    # INTERVAL OPERATIONS (INPLACE)

    def expand(self, distance: Union[datetime.timedelta]) -> 'DateTimeInterval':
        raise NotImplementedError

    # INTERVAL OPERATIONS

    def __contains__(self, other: DATE_LIKE) -> bool:
        raise NotImplementedError

    def overlapping(self, other: DATE_LIKE, or_adjacent: bool = False) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.overlapping(_get_interval(other), or_adjacent=or_adjacent)
        return out

    def overlaps(self, other: DATE_LIKE, or_adjacent: bool = False) -> bool:
        return self.interval.overlaps(_get_interval(other), or_adjacent=or_adjacent)

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: None) -> 'DateTimeInterval':
        raise NotImplementedError

    def __radd__(self, other: None) -> 'DateTimeInterval':
        raise NotImplementedError

    def __sub__(self, other: None) -> 'DateTimeInterval':
        raise NotImplementedError

    def __rsub__(self, other: None) -> 'DateTimeInterval':
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
