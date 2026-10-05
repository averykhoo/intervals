"""
the numeric functions of ieee 1788 over cut tuples: `mid`, `rad`, `wid`, `mag`, `mig`, `mid_rad`

`mid`, `rad` and `wid` are of the hull: a midpoint outside the set is still a valid bisection
point, and `size().length` already gives the width without the gaps. `mag` and `mig` are of the set,
the supremum and the infimum of `{abs(x) : x in A}`, so a gap around zero counts. open and closed
ends do not matter: these are infima and suprema (D9).

an operand with no finite float end is exact and its values are exact (int or Fraction). one with a
finite float end gives floats, the exact value rounded once as 1788 specifies: `mid` to nearest (ties
to even), `rad` the smallest double `r` with `[mid - r, mid + r]` holding the hull (so measured from
the rounded midpoint, which can make it larger than half the width rounded up), `wid` and `mag` up,
`mig` down. the direction is the function's, not the class's: `MultiInterval` and
`OutwardMultiInterval` give the same numbers.

unbounded operands follow 1788: the midpoint of `(-inf, inf)` is 0 and of a half-bounded set ±max
float (a float even for an exact operand, as in 1788); `rad` and `wid` are inf. a single point, `[inf]`
too, has itself as midpoint and 0 as radius and width. the empty set has none of these and raises
`ValueError`, as `inf` and `sup` do; 1788 answers `NaN` there.
"""
import math
from fractions import Fraction
from typing import Tuple

from multiinterval.cuts import Value
from multiinterval.cuts import normalize_value
from multiinterval.kernel import Cuts
from multiinterval.kernel import pieces
from multiinterval.rounding import DOWN
from multiinterval.rounding import MAX
from multiinterval.rounding import NEAREST
from multiinterval.rounding import UP
from multiinterval.rounding import has_finite_float
from multiinterval.rounding import is_infinite
from multiinterval.rounding import is_float
from multiinterval.rounding import round_rational
from multiinterval.rounding import round_value

INF = math.inf


def _hull(cuts: Cuts, what: str) -> Tuple[Value, Value]:
    if not cuts:
        raise ValueError(f'the empty set has no {what}')
    return cuts[0].value, cuts[-1].value


def _exact(v):
    return Fraction(v) if is_float(v) else v


def mid(cuts: Cuts) -> Value:
    """the midpoint of the hull"""
    lo, hi = _hull(cuts, 'midpoint')
    return _mid(lo, hi, has_finite_float(cuts))


def _mid(lo, hi, rounded: bool) -> Value:
    if lo == hi:
        return round_value(lo, NEAREST) if rounded else lo  # `[0, 0.0]` holds one float end
    if is_infinite(lo) and is_infinite(hi):
        return 0.0 if rounded else 0
    if is_infinite(lo) or is_infinite(hi):
        m = -MAX if lo == -INF else MAX
    else:
        m = Fraction(_exact(lo) + _exact(hi), 2)
        if not rounded:
            return normalize_value(m)
        m = round_rational(m, NEAREST)
    # only an exact end past the float range can leave m outside the hull (rounding to nearest
    # overflowed, or a half-bounded hull starts beyond max float): take the hull's end then
    if m < lo:
        return round_value(lo, UP) if rounded else lo
    if m > hi:
        return round_value(hi, DOWN) if rounded else hi
    return m


def rad(cuts: Cuts) -> Value:
    """the radius of the hull, measured from `mid`"""
    return _mid_rad(cuts, 'radius')[1]


def mid_rad(cuts: Cuts) -> Tuple[Value, Value]:
    """`(mid, rad)`"""
    return _mid_rad(cuts, 'midpoint or radius')


def _mid_rad(cuts: Cuts, what: str) -> Tuple[Value, Value]:
    lo, hi = _hull(cuts, what)
    rounded = has_finite_float(cuts)
    m = _mid(lo, hi, rounded)
    if lo == hi:
        return m, 0.0 if rounded else 0
    if is_infinite(lo) or is_infinite(hi):
        return m, INF
    r = max(_exact(m) - _exact(lo), _exact(hi) - _exact(m))
    return m, round_rational(r, UP) if rounded else normalize_value(r)


def wid(cuts: Cuts) -> Value:
    """the width of the hull, `sup - inf`"""
    lo, hi = _hull(cuts, 'width')
    rounded = has_finite_float(cuts)
    if lo == hi:
        return 0.0 if rounded else 0
    if is_infinite(lo) or is_infinite(hi):
        return INF
    w = _exact(hi) - _exact(lo)
    return round_rational(w, UP) if rounded else normalize_value(w)


def mag(cuts: Cuts) -> Value:
    """the magnitude, `sup {abs(x) : x in A}`"""
    lo, hi = _hull(cuts, 'magnitude')
    m = max(abs(lo), abs(hi))
    return round_value(m, UP) if has_finite_float(cuts) else m


def mig(cuts: Cuts) -> Value:
    """the mignitude, `inf {abs(x) : x in A}`: of the set, so a gap around zero counts"""
    if not cuts:
        raise ValueError('the empty set has no mignitude')
    m = min(0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) for lo, _, hi, _ in pieces(cuts))
    return round_value(m, DOWN) if has_finite_float(cuts) else m
