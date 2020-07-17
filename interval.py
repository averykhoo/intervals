from dataclasses import dataclass
from numbers import Real


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
            if self.start_open:
                raise ValueError(f'left bound must be closed for a degenerate interval')
            if not self.end_closed:
                raise ValueError(f'right bound must be closed for a degenerate interval')
