import datetime
import operator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd

from multi_interval import MultiInterval


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

    def __lt__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.lt(self.interval, other.interval)

    def __le__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.le(self.interval, other.interval)

    def __eq__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.eq(self.interval, other.interval)

    def __ne__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.ne(self.interval, other.interval)

    def __ge__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.ge(self.interval, other.interval)

    def __gt__(self, other: 'DateTimeInterval') -> bool:
        if not isinstance(other, DateTimeInterval):
            raise TypeError(other)
        return operator.gt(self.interval, other.interval)

        # UTILITY

    def copy(self) -> 'DateTimeInterval':
        out = DateTimeInterval()
        out.interval = self.interval.copy()
        return out

    def __sizeof__(self) -> int:
        return self.interval.__sizeof__()  # probably correct?

    def __getitem__(self, item: Union[slice, 'DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    # SET: BINARY RELATIONS

    def isdisjoint(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> bool:
        raise NotImplementedError

    def issubset(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> bool:
        raise NotImplementedError

    def issuperset(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> bool:
        raise NotImplementedError

    # SET: ITEMS (INPLACE)

    def add(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def clear(self) -> 'DateTimeInterval':
        raise NotImplementedError

    def discard(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def pop(self) -> 'DateTimeInterval':
        raise NotImplementedError

    def remove(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    # SET: BOOLEAN ALGEBRA (INPLACE)

    def update(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def intersection_update(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def difference_update(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def symmetric_difference_update(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    # SET: BOOLEAN ALGEBRA

    def union(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        return self.copy().update(*other)

    def intersection(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        return self.copy().intersection_update(*other)

    def difference(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        return self.copy().difference_update(*other)

    def symmetric_difference(self, *other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        return self.copy().symmetric_difference_update(*other)

    # INTERVAL OPERATIONS (INPLACE)

    def expand(self, distance: Union[datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    # INTERVAL OPERATIONS

    def __contains__(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> bool:
        raise NotImplementedError

    def overlapping(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date], or_adjacent: bool = False) -> 'DateTimeInterval':
        raise NotImplementedError

    def overlaps(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date], or_adjacent: bool = False) -> bool:
        raise NotImplementedError

    # INTERVAL ARITHMETIC: BINARY

    def __add__(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def __radd__(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def __sub__(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def __rsub__(self, other: Union['DateTimeInterval', datetime.datetime, datetime.date]) -> 'DateTimeInterval':
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
