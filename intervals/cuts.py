"""
cuts: the boundaries between points

a bound is a cut, a boundary *between* points: `Cut(value, side)` sits just below or just above
`value`. the same cut reads differently as the start or the end of a piece:

    | cut           | as start | as end |
    |---------------|----------|--------|
    | `(v, BELOW)`  | `[v`     | `v)`   |
    | `(v, ABOVE)`  | `(v`     | `v]`   |

so `[a, b]` = `(a, BELOW), (b, ABOVE)` and `[x]` = `(x, BELOW), (x, ABOVE)`. ordering is plain tuple
comparison, which is why a closed end at 2 and a closed start at 2 are different cuts: `[1, 2) | [2, 3]`
tiles exactly because the end `(2, BELOW)` equals the start `(2, BELOW)`.

values are int, Fraction or float, including `-inf` and `inf`. the constructor normalizes them:
`-0.0` becomes `0.0` (there is one zero), a Fraction with denominator 1 becomes int, and `nan` is
rejected.
"""
import math
from enum import IntEnum
from fractions import Fraction
from numbers import Integral
from numbers import Real
from typing import NamedTuple
from typing import Tuple
from typing import Union

Value = Union[int, Fraction, float]


class Side(IntEnum):
    BELOW = -1
    ABOVE = 1


def normalize_value(value) -> Value:
    """
    coerce a number to int, Fraction or float; `-0.0` -> `0.0`, integral Fraction -> int

    >>> normalize_value(Fraction(6, 3))
    2
    >>> normalize_value(-0.0)
    0.0
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'expected a real number, got {type(value).__name__}: {value!r}')
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else value
    value = float(value)
    if math.isnan(value):
        raise ValueError('nan is not a point of the extended reals')
    if value == 0:
        return 0.0  # also turns -0.0 into 0.0
    return value


class _CutBase(NamedTuple):
    value: Value
    side: Side


class Cut(_CutBase):
    """
    a boundary just below or just above `value`; see the module docstring

    (typing.NamedTuple refuses a `__new__` override, hence the base class)
    """
    __slots__ = ()

    def __new__(cls, value, side: Side) -> 'Cut':
        return super().__new__(cls, normalize_value(value), Side(side))

    def __repr__(self) -> str:
        return f'Cut({self.value!r}, {self.side.name})'


def below(value) -> Cut:
    """the cut just below `value`: a closed start or an open end"""
    return Cut(value, Side.BELOW)


def above(value) -> Cut:
    """the cut just above `value`: an open start or a closed end"""
    return Cut(value, Side.ABOVE)


def start_cut(value, closed: bool) -> Cut:
    return Cut(value, Side.BELOW if closed else Side.ABOVE)


def end_cut(value, closed: bool) -> Cut:
    return Cut(value, Side.ABOVE if closed else Side.BELOW)


def as_start(cut: Cut) -> Tuple[Value, bool]:
    """read a cut as the start of a piece: `(value, closed)`"""
    return cut.value, cut.side is Side.BELOW


def as_end(cut: Cut) -> Tuple[Value, bool]:
    """read a cut as the end of a piece: `(value, closed)`"""
    return cut.value, cut.side is Side.ABOVE


def mirror(cut: Cut) -> Cut:
    """
    the cut's image under `x -> -x`: just below v becomes just above -v

    (a lookup rather than `-cut.side`, because negating an IntEnum member gives a plain int)
    """
    return Cut(-cut.value, _OTHER_SIDE[cut.side])


_OTHER_SIDE = {Side.BELOW: Side.ABOVE, Side.ABOVE: Side.BELOW}
