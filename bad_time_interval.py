import datetime
from typing import List
from typing import Optional
from typing import Union

import pandas as pd


class Interval:
    from_date: datetime.datetime
    to_date: datetime.datetime

    @property
    def duration(self):
        return self.to_date - self.from_date

    def __init__(self,
                 from_date: Union[datetime.date, datetime.datetime, pd.Timestamp],
                 to_date: Union[datetime.date, datetime.datetime, pd.Timestamp]):

        # maybe convert to immutable (frozen), hashable, comparable (order) dataclass?

        if isinstance(from_date, pd.Timestamp):
            self.from_date = from_date.to_pydatetime()  # convert pandas to datetime
        elif isinstance(from_date, datetime.datetime):
            self.from_date = from_date
        elif isinstance(from_date, datetime.date):
            self.from_date = datetime.datetime.combine(from_date, datetime.time.min)  # convert date to datetime
        else:
            raise TypeError(from_date)

        if isinstance(to_date, pd.Timestamp):
            self.to_date = to_date.to_pydatetime()
        elif isinstance(to_date, datetime.datetime):
            self.to_date = to_date
        elif isinstance(to_date, datetime.date):
            self.to_date = datetime.datetime.combine(to_date, datetime.time.max)  # include full day
        else:
            raise TypeError(to_date)

        if self.from_date > self.to_date:
            raise ValueError(f'duration cannot be negative: {self.from_date} to {self.to_date} -> {self.duration}')

    def __hash__(self):
        return hash((self.from_date, self.to_date))

    def __eq__(self, other: 'Interval') -> bool:
        if isinstance(other, Interval):
            return self.from_date == other.from_date and self.to_date == other.to_date

        return False

    def __contains__(self, item: Union[datetime.date, datetime.datetime, pd.Timestamp, 'Interval']) -> bool:
        # contains an instant in time
        if isinstance(item, pd.Timestamp):
            return self.from_date <= item.to_pydatetime() <= self.to_date

        elif isinstance(item, datetime.datetime):
            return self.from_date <= item <= self.to_date

        # contains an interval
        elif isinstance(item, Interval):
            return self.from_date <= item.from_date and item.to_date <= self.to_date

        # contains a full day (from 00:00:00.000 to 23:59:59.999)
        elif isinstance(item, datetime.date):
            return self.from_date <= datetime.datetime.combine(item, datetime.time.min) and \
                   self.to_date >= datetime.datetime.combine(item, datetime.time.max)

        # unknown type
        else:
            raise TypeError(item)

    def overlaps(self, other: Union[datetime.date, datetime.datetime, pd.Timestamp, 'Interval']) -> bool:
        # contains an instant in time
        if isinstance(other, pd.Timestamp):
            return other in self
        elif isinstance(other, datetime.datetime):
            return other in self

        # overlaps an interval
        elif isinstance(other, Interval):
            return self.from_date <= other.to_date and other.from_date <= self.to_date

        # overlaps a day == contains a date
        elif isinstance(other, datetime.date):
            return self.from_date.date() <= other <= self.to_date.date()

        # unknown type
        else:
            raise TypeError(other)

    def touches(self, other: Union[datetime.date, datetime.datetime, pd.Timestamp, 'Interval']) -> bool:
        tmp = Interval(self.from_date - datetime.timedelta(microseconds=1),
                       self.to_date + datetime.timedelta(microseconds=1))
        return tmp.overlaps(other)

    def intersection(self,
                     other: Union[datetime.date, datetime.datetime, pd.Timestamp, 'Interval']
                     ) -> Optional['Interval']:

        # contains an instant in time
        if isinstance(other, pd.Timestamp):
            _other = other.to_pydatetime()
            if _other in self:
                return Interval(_other, _other)
        elif isinstance(other, datetime.datetime):
            if other in self:
                return Interval(other, other)

        # overlaps an interval
        elif isinstance(other, Interval):
            if self.overlaps(other):
                return Interval(max(self.from_date, other.from_date), min(self.to_date, other.to_date))

        # overlaps a day == contains a date
        elif isinstance(other, datetime.date):
            return self.intersection(Interval(other, other))

        # unknown type
        else:
            raise TypeError(other)

    def union(self, other: Union['Interval', datetime.date]) -> Optional['Interval']:
        # overlaps an interval
        if isinstance(other, Interval):
            if self.overlaps(other):
                return Interval(min(self.from_date, other.from_date), max(self.to_date, other.to_date))

            # possibly subsequent days, 1 microsecond apart: self day precedes other day
            elif other.from_date - self.to_date == datetime.timedelta(microseconds=1):
                return Interval(self.from_date, other.to_date)

            # other day precedes self day
            elif self.from_date - other.to_date == datetime.timedelta(microseconds=1):
                return Interval(other.from_date, self.to_date)

        # overlaps a day == contains a date
        elif isinstance(other, datetime.date):
            return self.union(Interval(other, other))

        # unknown type
        else:
            raise TypeError(other)

    def __repr__(self):
        return f'Interval({repr(self.from_date)}, {repr(self.to_date)})'


class IntervalUnion:
    intervals: List[Interval]

    @property
    def duration(self):
        return sum((interval.duration for interval in self.intervals), datetime.timedelta(0))

    @property
    def min_date(self):
        if len(self.intervals) > 0:
            return min(interval.from_date for interval in self.intervals)

    @property
    def max_date(self):
        if len(self.intervals) > 0:
            return max(interval.to_date for interval in self.intervals)

    def __init__(self):
        self.intervals = []

    def __iter__(self):
        return iter(self.intervals)

    def __contains__(self, item):
        for interval in self.intervals:
            if item in interval:
                return True
        return False

    def add(self, other: Interval):
        _other = Interval(other.from_date, other.to_date)

        if isinstance(other, Interval):
            _overlapping = []
            _remainder = []

            # sort intervals into overlapping and disjoint
            for interval in self.intervals:
                if interval.touches(other):
                    _overlapping.append(interval)
                else:
                    _remainder.append(interval)

            # combine all overlapping intervals
            for interval in _overlapping:
                if other.union(interval) is None:
                    print(_other.touches(interval))
                    print(other.touches(interval))
                    print(_other)
                    print(other)
                    print(interval)
                    print('-' * 100)
                    print(self.intervals)
                    for _interval in _overlapping:
                        print(_interval)
                other = other.union(interval)

            # sort the intervals before saving
            _remainder.append(other)
            self.intervals = sorted(_remainder, key=lambda x: (x.from_date, x.to_date))

        else:
            raise TypeError

    def union(self, other: 'IntervalUnion'):
        if isinstance(other, IntervalUnion):
            _new = IntervalUnion()
            for interval in self.intervals:
                _new.add(interval)
            for interval in other.intervals:
                _new.add(interval)

            return _new

        raise TypeError(other)

    def intersection(self, other: 'IntervalUnion'):
        if isinstance(other, IntervalUnion):
            _new = IntervalUnion()
            for interval_1 in self.intervals:
                for interval_2 in other.intervals:
                    if interval_1.overlaps(interval_2):
                        _new.add(interval_1.intersection(interval_2))
                    elif interval_2.from_date > interval_1.to_date:
                        break

            return _new

        raise TypeError(other)

    def __repr__(self):
        return repr(self.intervals)
