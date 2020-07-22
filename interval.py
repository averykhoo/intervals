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
        if isinstance(other, Interval):
            return self.start_tuple <= other.start_tuple and other.end_tuple <= self.end_tuple
        elif isinstance(other, Real):
            return self.start_tuple <= (other, 0) <= self.end_tuple
        else:
            raise TypeError(other)

    def expand(self, distance: Real) -> 'Interval':
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)  # todo: allow negative numbers as long as it returns a non-empty interval

        return Interval(self.start - distance, self.start_open, self.end + distance, self.end_closed)

    def as_closed_interval(self):
        if self.start_closed and self.end_closed:
            return self
        else:
            return Interval(self.start, False, self.end, True)

    def overlaps(self, other: Union[Real, 'Interval'], or_adjacent=False) -> bool:
        if or_adjacent:
            _start_tuple = (self.start, 0 if self.start_open else -1)
            _end_tuple = (self.end, 1 if self.end_closed else 0)
        else:
            _start_tuple = self.start_tuple
            _end_tuple = self.end_tuple

        if isinstance(other, Interval):
            return _start_tuple <= other.end_tuple and other.start_tuple <= _end_tuple
        elif isinstance(other, Real):
            return _start_tuple <= (other, 0) <= _end_tuple
        else:
            raise TypeError(other)

    def intersect(self, other: Union[Real, 'Interval']) -> Optional['Interval']:
        if isinstance(other, Interval) and not other.is_degenerate:
            if self.overlaps(other):
                _start_tuple = max(self.start_tuple, other.start_tuple)
                _end_tuple = min(self.end_tuple, other.end_tuple)
                return Interval(*_start_tuple, *_end_tuple)
            else:
                raise ValueError(f'{str(other)} does not overlap {str(self)}, so the intersection is a null set')

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real
            if other in self:
                return Interval(other, False, other, True)  # degenerate interval
            else:
                raise ValueError(f'[{other}] is not in {str(self)}, so the intersection is a null set')

        else:
            raise TypeError(other)

    def union(self, other: Union[Real, 'Interval']) -> Optional['Interval']:
        if isinstance(other, Interval) and not other.is_degenerate:
            if self.overlaps(other, or_adjacent=True):
                _start_tuple = min(self.start_tuple, other.start_tuple)
                _end_tuple = max(self.end_tuple, other.end_tuple)
                return Interval(*_start_tuple, *_end_tuple)
            else:
                raise ValueError(f'{str(other)} is not adjacent to {str(self)}, so the union comprises two Intervals')

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

            if other in self:
                return self
            elif other == self.start:
                return Interval(self.start, False, self.end, self.end_closed)
            elif other == self.end:
                return Interval(self.start, self.start_open, self.end, True)
            else:
                raise ValueError(f'[{other}] is not adjacent to {str(self)}, so the union comprises two Intervals')

        else:
            raise TypeError(other)

    def difference(self, other: Union[Real, 'Interval']) -> Optional['Interval']:
        if isinstance(other, Interval) and not other.is_degenerate:
            if not self.overlaps(other):
                return self
            raise NotImplementedError  # todo

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

            if self.is_degenerate:
                if float(self) == other:
                    raise ValueError('difference is the null set')
                else:
                    return self

            if other not in self:
                return self
            elif self.start == other:
                return Interval(self.start, True, self.end, self.end_closed)
            elif self.end == other:
                return Interval(self.start, self.start_open, self.end, False)
            else:
                raise ValueError(f'{str(self)} difference [{other}] comprises two Intervals')

        else:
            raise TypeError(other)

    def symmetric_difference(self, other: Union[Real, 'Interval']) -> Optional['Interval']:
        if isinstance(other, Interval) and not other.is_degenerate:
            if self.start_tuple == other.start_tuple:
                if self.end_tuple < other.end_tuple:
                    return Interval(*self.end_tuple, *other.end_tuple)
                else:
                    return Interval(*other.end_tuple, *self.end_tuple)
            elif self.end_tuple == other.end_tuple:
                if self.start_tuple < other.start_tuple:
                    return Interval(*self.start_tuple, *other.start_tuple)
                else:
                    return Interval(*other.start_tuple, *self.start_tuple)

            else:
                raise ValueError(f'{str(self)} symmetric difference {str(other)} comprises two Intervals')

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

            if self.is_degenerate:
                if float(self) == other:
                    raise ValueError('symmetric difference is the null set')
                else:
                    raise ValueError('symmetric difference comprises two intervals')

            if self.start_tuple == (other, False):
                return Interval(self.start, True, self.end, self.end_closed)
            elif self.end_tuple == (other, True):
                return Interval(self.start, self.start_open, self.end, False)
            else:
                raise ValueError(f'{str(self)} symmetric difference [{other}] comprises two Intervals')

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
        if isinstance(other, Interval) and not other.is_degenerate:
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

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

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

        else:
            raise TypeError(other)

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
        if isinstance(other, Interval) and not other.is_degenerate:
            if self.is_degenerate:
                return self.start ** other
            elif float(self.start) > 0.0:
                return self._apply_monotonic_binary_function(operator.pow, other)
            elif float(self.start) < 0.0:
                raise ValueError('non-even powers of negative numbers are complex')

            # self.start == 0.0 < self.end, other.start > 0
            elif float(other.start) > 0:
                return self._apply_monotonic_binary_function(operator.pow, other)
            else:
                # todo: when self.start == 0.0 and 0.0 in other, you might get a multi-interval
                raise NotImplementedError

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

            if self.is_degenerate:
                return Interval(self.start ** other, False, self.start ** other, True)

            elif float(other) == 0.0:
                return Interval(1, False, 1, True)

            elif float(self.start) > 0:
                return self._apply_monotonic_binary_function(operator.pow, other)
            elif float(self.start) < 0:
                if float(other) % 2 == 0:
                    return abs(self) ** other
                else:
                    raise ValueError('non-even powers of negative numbers are complex')

            # self.start == 0 < self.end
            elif float(other) > 0:
                return self._apply_monotonic_binary_function(operator.pow, other)

            # self.start == 0 < self.end, other < 0
            else:
                return Interval(self.end ** other, self.end_open, math.inf, True)

        else:
            raise TypeError(other)

    def __rpow__(self, other: Union[Real, 'Interval']) -> 'Interval':
        if isinstance(other, Interval) and not other.is_degenerate:
            raise NotImplementedError  # handled in __pow__

        elif isinstance(other, (Real, Interval)):
            if isinstance(other, Interval):
                other = other.start  # treat degenerate interval as Real

            if float(other) > 0:
                return self._apply_monotonic_binary_function(operator.pow, other, True)
            elif float(other) == 0:
                if 0 not in self:
                    return Interval(0, False, 0, True)
                elif self.is_degenerate and float(self) == 0:
                    return Interval(1, False, 1, True)
                else:
                    raise ValueError(f'0.0 ** {self.__str__()} does not produce a single Interval')

            elif self.is_degenerate and float(self) % 2 == 0:
                return Interval(other ** float(self), False, other ** float(self), True)

            else:
                raise ValueError('non-even powers of negative numbers are complex')

        else:
            raise TypeError(other)

    def __float__(self) -> float:
        if self.is_degenerate:
            return float(self.start)
        else:
            raise ValueError('Interval that is not degenerate cannot be coerced to float')

    def __int__(self) -> int:
        if self.is_degenerate:
            return int(float(self.start))
        else:
            raise ValueError('Interval that is not degenerate cannot be coerced to int')

    def __neg__(self):
        # return Interval(-self.end, self.end_open, -self.start, self.start_closed)
        return self._apply_monotonic_unary_function(operator.neg)

    def __pos__(self):
        # return Interval(self.start, self.start_open, self.end, self.end_closed)  # or `return self`
        return self._apply_monotonic_unary_function(operator.pos)

    def __abs__(self) -> 'Interval':
        if 0 in self:
            _start_tuple = (0, False)
        else:
            _start_tuple = min((abs(self.start), self.start_open),
                               (abs(self.end), self.end_open))
        _end_tuple = max((abs(self.start), self.start_closed),
                         (abs(self.end), self.end_closed))
        return Interval(*_start_tuple, *_end_tuple)

    def __invert__(self):
        if math.isinf(self.start) and not math.isinf(self.end):
            return Interval(self.end, self.end_closed, math.inf, True)
        elif not math.isinf(self.start) and math.isinf(self.end):
            return Interval(-math.inf, False, self.start, self.start_open)
        elif math.isinf(self.start) and math.isinf(self.end):
            raise ValueError(f'the inversion of {str(self)} is the null set, which is not a valid Interval')
        else:
            raise ValueError(f'the inversion of {str(self)} comprises two Intervals')

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
        if isinstance(other, Interval) and not other.is_degenerate:
            return self * other.reciprocal()

        elif isinstance(other, (Real, Interval)):
            if float(other) == 0:
                return Interval(-math.inf, False, math.inf, True)
            else:
                return self._apply_monotonic_binary_function(operator.truediv, other)

        else:
            raise TypeError

    def __rtruediv__(self, other):
        if isinstance(other, Interval) and not other.is_degenerate:
            return self * other.reciprocal()

        elif isinstance(other, (Real, Interval)):

            if float(other) == 0:
                return Interval(0, False, 0, True)
            elif math.isinf(float(other)):
                raise ValueError('infinity divided by an Interval does not produce a valid Interval')
            else:
                return other * self.reciprocal()

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
