"""
the reductions of ieee 1788 (after ieee 754's): `sum_`, `sum_abs`, `sum_sqr` and `dot` over sequences
of numbers, each the exact value through `Fraction`, rounded once to a double

they are point operations, not interval ones. the operands are real numbers (int, Fraction or float,
mixed freely, ±inf included) and the result is one float: the exact value rounded to nearest, ties to
even, or `rounding='down'` / `'up'` for the largest double below it or the smallest above. nothing in
between is rounded, so the order of the operands never matters, as it does for float arithmetic:

>>> (1e100 + 1.0) - 1e100, sum_([1e100, 1.0, -1e100])
(0.0, 1.0)
>>> tenths = [Fraction(1, 10)] * 3
>>> sum_(tenths), sum_(tenths, rounding='down'), sum_(tenths, rounding='up')
(0.3, 0.3, 0.30000000000000004)
>>> dot([2 ** 52 + 1, 2 ** 104], [2 ** 52 - 1, -1])  # float products would cancel to 0
-1.0

±inf are ordinary points, as everywhere in the package: a sum reaching one infinity is that infinity
in every rounding. an operand that is `nan`, and a value that has none (`inf + -inf`, `0 * inf`),
raise `ValueError`, as `nan` does in a constructor; 1788 answers `NaN` there, and the itf1788 adapter
reads the error as that `NaN`

>>> sum_abs([1, -INF, 2, INF]), sum_sqr([3, 4.0]), sum_([])
(inf, 25.0, 0.0)
>>> sum_([1, -INF, 2, INF])
Traceback (most recent call last):
ValueError: inf + -inf has no value
>>> dot([0], [INF])
Traceback (most recent call last):
ValueError: 0 * inf has no value
"""
import math
from fractions import Fraction
from numbers import Integral
from numbers import Rational
from numbers import Real
from typing import Iterable
from typing import List
from typing import Union

from multiinterval.rounding import DOWN
from multiinterval.rounding import NEAREST
from multiinterval.rounding import UP
from multiinterval.rounding import is_infinite
from multiinterval.rounding import round_rational

INF = math.inf

_DIRECTIONS = {'nearest': NEAREST, 'down': DOWN, 'up': UP}

Exact = Union[int, Fraction, float]  # a float only when it is ±inf


def _direction(rounding: str) -> int:
    try:
        return _DIRECTIONS[rounding]
    except (KeyError, TypeError):
        raise ValueError(f"rounding must be 'nearest', 'down' or 'up', got {rounding!r}") from None


def _exact(x) -> Exact:
    """an operand held exactly: an int or a Fraction, or ±inf"""
    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError(f'expected a real number, got {type(x).__name__}: {x!r}')
    if isinstance(x, Integral):
        return int(x)
    if isinstance(x, Rational):
        return Fraction(x)
    x = float(x)
    if math.isnan(x):
        raise ValueError('nan is not a point of the extended reals')
    return x if is_infinite(x) else Fraction(x)


def _operands(xs: Iterable) -> List[Exact]:
    return [_exact(x) for x in xs]


def _rounded(terms: Iterable[Exact], rounding: str) -> float:
    """the exact sum of the terms, rounded once"""
    direction = _direction(rounding)
    terms = list(terms)
    infinities = {t for t in terms if is_infinite(t)}
    if len(infinities) == 2:
        raise ValueError('inf + -inf has no value')
    if infinities:
        return float(infinities.pop())
    return round_rational(sum(terms, 0), direction)


def _product(a: Exact, b: Exact) -> Exact:
    if is_infinite(a) or is_infinite(b):
        if a == 0 or b == 0:
            raise ValueError('0 * inf has no value')
        return INF if (a > 0) == (b > 0) else -INF
    return a * b


def sum_(xs: Iterable, *, rounding: str = 'nearest') -> float:
    """
    the sum of the numbers, exact, rounded once (1788's `sum`)

    >>> sum_([1, Fraction(1, 3), 0.5])  # 11/6
    1.8333333333333333
    """
    return _rounded(_operands(xs), rounding)


def sum_abs(xs: Iterable, *, rounding: str = 'nearest') -> float:
    """
    the sum of the numbers' absolute values, exact, rounded once (1788's `sumAbs`)

    >>> sum_abs([1, -2.5, Fraction(-1, 2)])
    4.0
    """
    return _rounded((abs(x) for x in _operands(xs)), rounding)


def sum_sqr(xs: Iterable, *, rounding: str = 'nearest') -> float:
    """
    the sum of the numbers' squares, exact, rounded once (1788's `sumSquare`)

    >>> sum_sqr([1e200, 1e200]), sum_sqr([1e200, 1e200], rounding='down')  # past the largest double
    (inf, 1.7976931348623157e+308)
    """
    return _rounded((x * x for x in _operands(xs)), rounding)


def dot(xs: Iterable, ys: Iterable, *, rounding: str = 'nearest') -> float:
    """
    the dot product of two sequences of numbers of the same length, exact, rounded once (1788's `dot`)

    >>> dot([1, 2, 3], [4, 5, 6.0])
    32.0
    >>> dot([1, 2], [3])
    Traceback (most recent call last):
    ValueError: dot of sequences of different lengths, 2 and 1
    """
    xs, ys = _operands(xs), _operands(ys)
    if len(xs) != len(ys):
        raise ValueError(f'dot of sequences of different lengths, {len(xs)} and {len(ys)}')
    return _rounded(map(_product, xs, ys), rounding)
