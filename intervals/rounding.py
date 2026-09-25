"""
rounding an exact value to a double: to nearest, down or up

int and Fraction are exact and never rounded by the package on their own; a value is rounded when a
float is involved, or when the exact value is irrational (a function's value, see `intervals.elementary`).
the rounding direction is a property of the type that asks for it (`MultiInterval` rounds to nearest,
`OutwardMultiInterval` outward), never an ambient mode.
"""
import math
import sys
from fractions import Fraction

from intervals import kernel

DOWN, NEAREST, UP = -1, 0, 1

MAX = sys.float_info.max
INF = math.inf


def is_infinite(x) -> bool:
    return x == INF or x == -INF


def is_float(x) -> bool:
    """a finite float: the values that are rounded (±inf is exact whatever its python type)"""
    return isinstance(x, float) and not is_infinite(x)


def round_rational(v, direction: int) -> float:
    """
    the double nearest to v (NEAREST, ties to even), the largest double <= v (DOWN) or the smallest
    double >= v (UP), for an exact finite v; ±inf past the float range where the direction allows it

    >>> round_rational(Fraction(1, 10), DOWN), round_rational(Fraction(1, 10), UP)
    (0.09999999999999999, 0.1)
    >>> round_rational(2 ** 1024, NEAREST), round_rational(2 ** 1024, DOWN)
    (inf, 1.7976931348623157e+308)
    """
    try:
        f = float(v)  # int / int and int -> float are correctly rounded in CPython
    except OverflowError:
        positive = v > 0
        if direction == NEAREST or (direction == UP) == positive:
            return INF if positive else -INF
        return MAX if positive else -MAX
    if direction == DOWN and Fraction(f) > v:
        f = math.nextafter(f, -INF)
    elif direction == UP and Fraction(f) < v:
        f = math.nextafter(f, INF)
    return f + 0.0  # no -0.0


def round_value(v, direction: int):
    """`round_rational` for a finite value; ±inf and a float as they are"""
    if is_infinite(v) or isinstance(v, float):
        return v
    return round_rational(v, direction)


def round_piece(p, outward: bool):
    """
    a piece's finite ends made float. outward, the low end goes down and the high end up, and an end
    that moved is open (nothing attains it). to nearest, the flags stay: conservative, not a promise.
    a piece that rounding squeezes to one point keeps that point, closed
    """
    lo, lo_closed, hi, hi_closed = p
    if outward:
        lo_rounded, hi_rounded = round_value(lo, DOWN), round_value(hi, UP)
        lo_closed = lo_closed and lo_rounded == lo
        hi_closed = hi_closed and hi_rounded == hi
    else:
        lo_rounded, hi_rounded = round_value(lo, NEAREST), round_value(hi, NEAREST)
    if lo_rounded == hi_rounded:
        return lo_rounded, True, hi_rounded, True
    return lo_rounded, lo_closed, hi_rounded, hi_closed


def has_finite_float(cuts) -> bool:
    return any(is_float(cut.value) for cut in cuts)


def exact_cuts(cuts):
    """the same set with every finite float end as the Fraction it denotes"""
    return kernel.normalize(kernel.piece(_exact(lo), _exact(hi), lo_closed, hi_closed)
                            for lo, lo_closed, hi, hi_closed in kernel.pieces(cuts))


def float_cuts(cuts, outward: bool):
    """every finite end rounded to a float (see `round_piece`)"""
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed)
                            for lo, lo_closed, hi, hi_closed in (round_piece(p, outward) for p in kernel.pieces(cuts)))


def _exact(v):
    return Fraction(v) if is_float(v) else v
