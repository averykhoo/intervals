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
    start_open: bool
    end: Real
    end_closed: bool

    @property
    def start_closed(self) -> bool:
        return not self.start_open

    @property
    def end_open(self) -> bool:
        return not self.end_closed

    @property
    def is_degenerate(self) -> bool:
        return self.start == self.end

    @property
    def length(self) -> Real:
        return self.start - self.end

    @property
    def start_tuple(self) -> Tuple[Real, int]:
        return self.start, 1 if self.start_open else 0

    @property
    def end_tuple(self) -> Tuple[Real, int]:
        return self.end, 0 if self.end_closed else -1

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
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        return Interval(self.start - distance, self.start_open, self.end + distance, self.end_closed)

    def closed(self):
        return Interval(self.start, False, self.end, True)

    def __apply_operator(self, func, other):
        if isinstance(other, Real):
            if self.is_degenerate:
                return Interval(func(self.start, other), False, func(self.start, other), True)
            else:
                _start_tuple = min((func(self.start, other), self.start_open),
                                   (func(self.end, other), self.end_open))
                _end_tuple = max((func(self.start, other), self.start_closed),
                                 (func(self.end, other), self.end_closed))
                return Interval(*_start_tuple, *_end_tuple)

        elif isinstance(other, Interval):
            _start_tuple = min((func(self.start, other.start), self.start_open or other.start_open),
                               (func(self.start, other.end), self.start_open or other.end_open),
                               (func(self.end, other.start), self.end_open or other.start_open),
                               (func(self.end, other.end), self.end_open or other.end_open))
            _end_tuple = max((func(self.start, other.start), self.start_closed and other.start_closed),
                             (func(self.start, other.end), self.start_closed and other.end_closed),
                             (func(self.end, other.start), self.end_closed and other.start_closed),
                             (func(self.end, other.end), self.end_closed and other.end_closed))
            return Interval(*_start_tuple, *_end_tuple)

        else:
            raise TypeError

    def __add__(self, other: Union[Real, 'Interval']):
        if isinstance(other, Real):
            return Interval(self.start + other, self.start_open, self.end + other, self.end_closed)

        elif isinstance(other, Interval):
            return Interval(self.start + other.start,
                            self.start_open or other.start_open,
                            self.end + other.end,
                            self.end_closed and other.end_closed)

        else:
            raise TypeError

    def __sub__(self, other: Union[Real, 'Interval']):
        if isinstance(other, Real):
            return Interval(self.start - other, self.start_open, self.end - other, self.end_closed)

        elif isinstance(other, Interval):
            return Interval(self.start - other.end,
                            self.start_open or not other.end_closed,
                            self.end - other.start,
                            self.end_closed and not other.start_open)

    def __mul__(self, other: Union[Real, 'Interval']):
        if isinstance(other, Real):
            return Interval(self.start * other, self.start_open, self.end * other, self.end_closed)

        elif isinstance(other, Interval):
            tmp = [self.start * other.start, self.start * other.end, self.end * other.start, self.end * other.end]
            return Interval(min(tmp),
                            self.start_open or other.start_open,  # todo: probably wrong
                            max(tmp),
                            self.end_closed and other.end_closed)  # todo: probably wrong

        else:
            raise TypeError

    def reciprocal(self):
        if 0 not in self:
            return Interval(1 / self.end,
                            not self.end_closed,  # todo: check
                            1 / self.start,
                            not self.start_open)  # todo: check

        elif self.is_degenerate:
            return Interval(-math.inf, False, math.inf, True)
        elif self.start_tuple == (0, 0):
            return Interval(1 / self.end,
                            not self.end_closed,  # todo: check
                            math.inf,
                            True)
        elif self.end_tuple == (0, 0):
            return Interval(-math.inf,
                            False,  # todo: check
                            1 / self.start,
                            not self.start_open)  # todo: check
        else:
            return Interval(-math.inf, False, math.inf, True)

    def __truediv__(self, other):
        if isinstance(other, Real):
            if other > 0:
                return Interval(self.start / other, self.start_open, self.end / other, self.end_closed)
            elif other < 0:
                return Interval(self.end / other, not self.end_closed, self.start / other, not self.start_open)
            else:
                Interval(-math.inf, False, math.inf, True)

        elif isinstance(other, Interval):
            return self * other.reciprocal()

        else:
            raise TypeError

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{repr(self.start)}, '
                f'{repr(self.start_open)}, '
                f'{repr(self.end)}, '
                f'{repr(self.end_closed)})')

    def __str__(self):
        if self.is_degenerate:
            return f'[{self.start}]'

        if self.start == -math.inf:
            _start = '-∞'
            _left_bracket = '('
        else:
            _start = self.start
            _left_bracket = '(' if self.start_open else '['

        if self.end == math.inf:
            _end = '∞'
            _right_bracket = ')'
        else:
            _end = self.end
            _right_bracket = ']' if self.end_closed and self.end != math.inf else ')'

        return f'{_left_bracket}{_start}, {_end}{_right_bracket}'
