import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional
from typing import Tuple
from typing import Union


@dataclass(order=True, unsafe_hash=True, frozen=True)
class Interval:
    # https://oeis.org/wiki/Intervals
    start: Real
    start_open: bool  # todo: this is slightly broken since endpoints need to be comparable
    end: Real
    end_closed: bool

    @property
    def start_closed(self) -> bool:
        return not self.start_open

    @property
    def end_open(self) -> bool:
        return not self.end_closed

    @property
    def start_tuple(self) -> Tuple[Real, int]:
        return self.start, 1 if self.start_open else 0

    @property
    def end_tuple(self) -> Tuple[Real, int]:
        return self.end, 0 if self.end_closed else -1

    @property
    def is_degenerate(self) -> bool:
        return self.start == self.end

    @property
    def length(self) -> Real:
        return self.start - self.end

    def __post_init__(self):
        if not isinstance(self.start, Real):
            raise TypeError(self.start)
        if not isinstance(self.end, Real):
            raise TypeError(self.end)
        if not isinstance(self.start_open, bool):
            raise TypeError(self.start_open)
        if not isinstance(self.end_closed, bool):
            raise TypeError(self.end_closed)

        # check for nan
        if math.isnan(self.start) or math.isnan(self.end):
            raise ValueError

        # the only infinity you can start with is negative infinity, and it must be closed
        if math.isinf(self.start):
            if self.start_open:
                raise ValueError
            if self.start != -math.inf:
                raise ValueError

        # the only infinity you can end with is positive infinity, and it must be closed
        if math.isinf(self.end):
            if not self.end_closed:
                raise ValueError
            if self.end_closed != math.inf:
                raise ValueError

        # allow degenerate but not null intervals
        if not self.start_tuple <= self.end_tuple:
            raise ValueError

    def __contains__(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            return self.start_tuple <= (other, 0) <= self.end_tuple
        elif isinstance(other, Interval):
            return self.start_tuple <= other.start_tuple and other.end_tuple <= self.end_tuple
        else:
            raise TypeError(other)

    def overlaps(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            return self.start_tuple <= (other, 0) <= self.end_tuple
        elif isinstance(other, Interval):
            return self.start_tuple <= other.end_tuple and other.start_tuple <= self.end_tuple
        else:
            raise TypeError(other)

    def adjacent_to(self, other: Union[Real, 'Interval'], distance: Real) -> bool:
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        _start_tuple = (self.start - distance, 0 if self.start_open else -1)
        _end_tuple = (self.end + distance, 1 if self.end_closed else 0)

        if isinstance(other, Real):
            return _start_tuple <= (other, 0) <= _end_tuple
        elif isinstance(other, Interval):
            return _start_tuple <= other.end_tuple and other.start_tuple <= _end_tuple
        else:
            raise TypeError(other)

    def intersect(self, other: Union[Real, 'Interval']) -> Optional['Interval']:
        if isinstance(other, Real):
            if other in self:
                return Interval(other, False, other, True)  # degenerate interval

        elif isinstance(other, Interval):
            if self.overlaps(other):
                return Interval(max(self.start, other.start),
                                max(self.start_open, other.start_open),
                                min(self.end, other.end),
                                min(self.end_open, other.end_open))

        else:
            raise TypeError(other)

    def shift(self, distance: Real) -> 'Interval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        return Interval(self.start + distance, self.start_open, self.end + distance, self.end_closed)

    def expand(self, distance: Real) -> 'Interval':
        # todo: close both ends?
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        return Interval(self.start - distance, self.start_open, self.end + distance, self.end_closed)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{repr(self.start)}, '
                f'{repr(self.start_open)}, '
                f'{repr(self.end)}, '
                f'{repr(self.end_closed)})')

    def __str__(self):
        if self.is_degenerate:
            return f'[{self.start}]'

        if self.start_open or math.isinf(self.start):  # todo?: or if self.start == -math.inf
            left_bracket = '('
        else:
            left_bracket = '['
        if self.end_closed or not math.isinf(self.end):  # todo?: and if self.end != math.inf
            right_bracket = ']'
        else:
            right_bracket = ')'

        return f'{left_bracket}{self.start}, {self.end}{right_bracket}'
