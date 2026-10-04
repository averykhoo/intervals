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
rejected. a *foreign* real (a `numbers.Real` that is no int, float or Fraction: numpy's scalars, gmpy2's
numbers) is its exact value where it has one to give: a `numbers.Rational` is exact by type, as a
Fraction is; any other real is the float it equals where it is a double (so `np.float32(0.1)` is the
double it holds, as today), else its exact `as_integer_ratio()` (an `np.longdouble` wider than a
double, a wide `mpfr`), where `float()` would have rounded it and an outward result would not hold it;
a real with no `as_integer_ratio()` is `float()` of it, as before (M16d, 2026-09-28). a numpy `timedelta64`
is no number although numpy registers it as `numbers.Integral` (its `int()` is its count in its own unit):
a TypeError, as a `datetime64` is (`is_numpy_time`; M8's review, 2026-10-04).
"""
import math
from enum import IntEnum
from fractions import Fraction
from numbers import Integral
from numbers import Rational
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
        if type(value) is not int and is_numpy_time(value):
            raise TypeError(f'expected a real number, got {type(value).__name__}: {value!r} (a duration, not a number)')
        return int(value)
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else value
    if not isinstance(value, float):  # a foreign real (the float path pays this one check only)
        if isinstance(value, Rational):  # gmpy2's mpq: exact by type, as a Fraction
            exact = Fraction(int(value.numerator), int(value.denominator))
            return int(exact) if exact.denominator == 1 else exact
        exact = _exact_value(value)  # exact where float() would round
        if exact is not None:
            try:
                rounded = float(exact)  # correctly rounded
            except OverflowError:  # past the doubles
                rounded = None
            if rounded != exact:  # float() would have moved it: keep it exact
                return int(exact) if exact.denominator == 1 else exact
            value = rounded  # a double: the float, as ever
    value = float(value)
    if math.isnan(value):
        raise ValueError('nan is not a point of the extended reals')
    if value == 0:
        return 0.0  # also turns -0.0 into 0.0
    return value


def is_numpy_time(value) -> bool:
    """
    a numpy `timedelta64` or `datetime64` (dtype kind 'm' or 'M'). numpy registers `timedelta64` as a
    `numbers.Integral`, so a real-number check takes it, and `int()` of it is its count in its own unit
    (`np.timedelta64(3, 'ns')` would be 3): a duration is no number, so the numeric class refuses it
    (TypeError), as it refuses a `timedelta`. the time layer reads neither (`intervals.time_interval`)
    """
    return getattr(getattr(value, 'dtype', None), 'kind', None) in ('m', 'M')


def _exact_value(value) -> Union[Fraction, None]:
    """a foreign real's exact value, its `as_integer_ratio()`; None for a nan or an infinity (which
    raise there) or a real without one"""
    ratio = getattr(value, 'as_integer_ratio', None)
    if ratio is None:
        return None
    try:
        numerator, denominator = ratio()
    except (ValueError, OverflowError):
        return None
    return Fraction(int(numerator), int(denominator))


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
        return f'Cut({_value_repr(self.value)}, {self.side.name})'


def _value_repr(v) -> str:
    """
    `repr(v)`, with an int part past python's int-str limit in hex (a python literal still), so the repr
    of a cut never raises (python's own `repr(10 ** 4300)` does)

    >>> _value_repr(Fraction(1, 3)), _value_repr(-(2 ** 15000))[:9]
    ('Fraction(1, 3)', '-0x100000')
    """
    try:
        return repr(v)
    except ValueError:
        if isinstance(v, Fraction):
            return f'Fraction({_value_repr(v.numerator)}, {_value_repr(v.denominator)})'
        return hex(v)


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


def flag(value, name: str) -> bool:
    """
    an open/closed flag a user passed: python's bool or numpy's, anything else a TypeError (owner, Q22(a),
    2026-10-04: a flag read by truthiness turned `start_closed='no'` into a closed end, silently)

    >>> flag(False, 'start_closed')
    False
    >>> flag('no', 'start_closed')
    Traceback (most recent call last):
    ...
    TypeError: start_closed must be a bool, not str: 'no'
    """
    if value is True or value is False:
        return value
    kind = type(value)
    if kind.__module__ == 'numpy' and kind.__name__ in ('bool', 'bool_'):  # `bool` from numpy 2, `bool_` before
        return bool(value)
    raise TypeError(f'{name} must be a bool, not {kind.__name__}: {value!r}')


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
