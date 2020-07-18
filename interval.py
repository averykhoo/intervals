from dataclasses import dataclass
from numbers import Real
from typing import Tuple
from typing import Union


@dataclass(order=True, unsafe_hash=True, frozen=True)
class Interval:
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
        if self.start > self.end:
            raise ValueError((self.start, self.end))
        if self.start == self.end:
            if self.start_open or not self.end_closed:
                raise ValueError(f'both left and right bound must be closed for a degenerate interval')
        if self.start_tuple >= self.end_tuple:
            raise ValueError('something else went wrong')

    def __contains__(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            other = Interval(other, False, other, True)
        elif not isinstance(other, Interval):
            raise TypeError(other)

        return self.start_tuple <= other.start_tuple and other.end_tuple <= self.end_tuple

    def overlaps(self, other: Union[Real, 'Interval']) -> bool:
        if isinstance(other, Real):
            other = Interval(other, False, other, True)
        elif not isinstance(other, Interval):
            raise TypeError(other)

        return self.start_tuple <= other.end_tuple and other.start_tuple <= self.end_tuple

    def expand(self, distance: Real) -> 'Interval':
        # todo: close both ends?
        if not isinstance(distance, Real):
            raise TypeError(distance)
        if distance < 0:
            raise ValueError(distance)

        return Interval(self.start - distance, self.start_open, self.end + distance, self.end_closed)

    def nearby(self, other: Union[Real, 'Interval'], distance: Real) -> bool:
        # todo: is [0, 1) within 2 of (3, 4] or not?
        return self.expand(distance).overlaps(other)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{repr(self.start)}, '
                f'{repr(self.start_open)}, '
                f'{repr(self.end)}, '
                f'{repr(self.end_open)})')

    def __str__(self):
        if self.start_open:
            left_bracket = '('
        else:
            left_bracket = '['
        if self.end_closed:
            right_bracket = ']'
        else:
            right_bracket = ')'

        if self.start == self.end:
            return f'{left_bracket}{self.start}{right_bracket}'
        else:
            return f'{left_bracket}{self.start}, {self.end}{right_bracket}'
