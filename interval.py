import math
import operator
from dataclasses import dataclass
from numbers import Real
from typing import Callable
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
        if self.start == self.end:
            assert self.start_closed and self.end_closed
            return True
        return False

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
            raise ValueError('Interval cannot start or end with NaN')

        # the only infinity you can start with is negative infinity, and it must be closed
        if math.isinf(self.start):
            if self.start != -math.inf:
                raise ValueError('Interval cannot start with positive infinity')
            if self.start_open:
                raise ValueError('There is no real value just after negative infinity')

        # the only infinity you can end with is positive infinity, and it must be closed
        if math.isinf(self.end):
            if self.end != math.inf:
                raise ValueError('Interval cannot end with negative infinity')
            if self.end_open:
                raise ValueError('There is no real value just before positive infinity')

        # strictly one way only
        if self.start > self.end:
            raise ValueError('Interval cannot go backwards')

        # allow degenerate but not null intervals
        if self.start_tuple > self.end_tuple:
            raise ValueError('Interval cannot be null')

    def __contains__(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            return self.start_tuple <= (other, 0) <= self.end_tuple
        elif isinstance(other, Interval):
            return self.start_tuple <= other.start_tuple and other.end_tuple <= self.end_tuple
        else:
            raise TypeError(other)

    def expand(self, distance: Real) -> 'Interval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        return Interval(self.start - distance, self.start_open, self.end + distance, self.end_closed)

    def as_closed_interval(self):
        if self.start_closed and self.end_closed:
            return self
        else:
            return Interval(self.start, False, self.end, True)

    def overlaps(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            return self.start_tuple <= (other, 0) <= self.end_tuple
        elif isinstance(other, Interval):
            return self.start_tuple <= other.end_tuple and other.start_tuple <= self.end_tuple
        else:
            raise TypeError(other)

    def adjacent_to(self, other: Union[Real, 'Interval']) -> bool:
        _start_tuple = (self.start, 0 if self.start_open else -1)
        _end_tuple = (self.end, 1 if self.end_closed else 0)

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
                                min(self.end_closed, other.end_closed))

        else:
            raise TypeError(other)

    def _apply_monotonic_unary_function(self, func: Callable) -> 'Interval':
        _start_tuple = min((func(self.start), self.start_open),
                           (func(self.end), self.end_open))
        _end_tuple = max((func(self.start), self.start_closed),
                         (func(self.end), self.end_closed))
        return Interval(*_start_tuple, *_end_tuple)

    def _apply_monotonic_binary_function(self,
                                         func: Callable,
                                         other: Union[Real, 'Interval'],
                                         right_hand_side: bool = False
                                         ) -> 'Interval':

        if isinstance(other, Real):
            if self.is_degenerate:
                if right_hand_side:
                    return Interval(func(other, self.start), False, func(other, self.start), True)
                else:
                    return Interval(func(self.start, other), False, func(self.start, other), True)

            else:
                if right_hand_side:
                    _start_tuple = min((func(other, self.start), self.start_open),
                                       (func(other, self.end), self.end_open))
                    _end_tuple = max((func(other, self.start), self.start_closed),
                                     (func(other, self.end), self.end_closed))
                else:
                    _start_tuple = min((func(self.start, other), self.start_open),
                                       (func(self.end, other), self.end_open))
                    _end_tuple = max((func(self.start, other), self.start_closed),
                                     (func(self.end, other), self.end_closed))
                return Interval(*_start_tuple, *_end_tuple)

        elif isinstance(other, Interval):
            if right_hand_side:
                x, y = other, self
            else:
                x, y = self, other

            _start_tuple = min((func(x.start, y.start), x.start_open or y.start_open),
                               (func(x.start, y.end), x.start_open or y.end_open),
                               (func(x.end, y.start), x.end_open or y.start_open),
                               (func(x.end, y.end), x.end_open or y.end_open))
            _end_tuple = max((func(x.start, y.start), x.start_closed and y.start_closed),
                             (func(x.start, y.end), x.start_closed and y.end_closed),
                             (func(x.end, y.start), x.end_closed and y.start_closed),
                             (func(x.end, y.end), x.end_closed and y.end_closed))

            return Interval(*_start_tuple, *_end_tuple)

        else:
            raise TypeError

    def __add__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.add, other)

    def __radd__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.add, other, True)

    def __sub__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.sub, other)

    def __rsub__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.sub, other, True)

    def __mul__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.mul, other)

    def __rmul__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.mul, other, True)

    def __lshift__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.lshift, other)

    def __rlshift__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.lshift, other, True)

    def __rshift__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.rshift, other)

    def __rrshift__(self, other: Union[Real, 'Interval']) -> 'Interval':
        return self._apply_monotonic_binary_function(operator.rshift, other, True)

    def __pow__(self, other: Union[Real, 'Interval']) -> 'Interval':
        if float(self.start) > 0.0:
            return self._apply_monotonic_binary_function(operator.pow, other)
        else:
            raise NotImplementedError

    def __float__(self) -> float:
        if self.is_degenerate:
            return float(self.start)
        else:
            raise ValueError

    def __int__(self) -> int:
        if self.is_degenerate:
            return int(float(self.start))
        else:
            raise ValueError

    def __abs__(self) -> 'Interval':
        if 0 in self:
            _start_tuple = (0, False)
        else:
            _start_tuple = min((abs(self.start), self.start_open),
                               (abs(self.end), self.end_open))
        _end_tuple = max((abs(self.start), self.start_closed),
                         (abs(self.end), self.end_closed))
        return Interval(*_start_tuple, *_end_tuple)

    def reciprocal(self):
        if 0 in self:
            return Interval(-math.inf, False, math.inf, True)

        elif self.start == 0:
            return Interval(1 / self.end,
                            self.end_open,
                            math.inf,
                            True)

        elif self.end == 0:
            return Interval(-math.inf,
                            False,
                            1 / self.start,
                            self.start_closed)

        else:
            return Interval(1 / self.end,
                            self.end_open,
                            1 / self.start,
                            self.start_closed)

    def __truediv__(self, other):
        if isinstance(other, Real):
            if other == 0:
                return Interval(-math.inf, False, math.inf, True)
            else:
                return self._apply_monotonic_binary_function(operator.truediv, other)

        elif isinstance(other, Interval):
            return self * other.reciprocal()

        else:
            raise TypeError

    def __rtruediv__(self, other):
        if isinstance(other, Real):
            if other == 0:
                return Interval(0, False, 0, True)
            else:
                return other * self.reciprocal()

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


if __name__ == '__main__':
    print(math.inf / Interval(1, False, 2, True))
