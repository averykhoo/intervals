"""
elementary functions over cut tuples: sqrt, exp, log, trig, hyperbolic and their inverses, and atan2

`apply(name, a)` is the set `{f(x) : x in a}`, with ±inf as ordinary points where f has a limit there
(`exp(-inf)` = 0, `atan(inf)` = pi/2, `tanh(inf)` = 1). every endpoint is closed iff attained.

* **domain**: points where f has no value are dropped with one `DomainClippedWarning`: `sqrt` and the
  logs below 0, `asin` `acos` `atanh` outside [-1, 1], `acosh` below 1, and sin, cos and tan at ±inf
  (no limit). the end of a domain is a point of it wherever f has a limit there, so
  `log([0, 1])` = `[-inf, 0]`, `sqrt([-1, 4])` = `[0, 2]` + warning, and `atanh([1])` = `[inf]`: the
  domain reaches 0 and 1 from one side only, so there is no second limit to disagree with
* **shape**: each function is monotone on each piece once split where its direction changes (cosh at
  0); a monotone continuous function maps a piece to the piece between its ends' images, each end
  closed iff the piece's end is. sin and cos also attain ±1 at every extremum inside a piece, and
  tan maps a piece holding a pole to both sides of it with both infinities attained, as `1/x` does
  at a zero inside a piece
* **values**: `intervals.elementary` computes each end, so a result is the same on every platform. an
  exact end (int, Fraction, ±inf) gives an exact result where the value is rational (`sqrt([9/4])` =
  `[3/2]`, `exp([0])` = `[1]`) and otherwise its tightest float enclosure, so an exact operand never
  loses the true value: `sqrt([2])` is the open one-ulp piece around the square root of 2, open
  because neither double is attained. a float end gives a float, rounded to nearest (flags kept: a
  conservative reading, not a promise), or outward with `outward=True` (`OutwardMultiInterval`),
  where an end that rounding moved is open as well

>>> from intervals.fmt import format_cuts, parse
>>> format_cuts(apply('sqrt', parse('[1/4, 9]')))
'[1/2, 3]'
>>> format_cuts(apply('exp', parse('[-inf, 0)')))
'[0, 1)'
>>> format_cuts(apply('tan', parse('[1, 2]')))
'{ [-inf, -2.185039863261519) , (1.557407724654902, inf] }'
>>> format_cuts(apply('sin', parse('[0, 4]')))
'(-0.7568024953079283, 1]'
"""
import math
from fractions import Fraction
from itertools import product
from typing import List
from typing import Tuple

from intervals import elementary
from intervals import fmt
from intervals import kernel
from intervals.applicator import split_pieces
from intervals.applicator import warn
from intervals.cuts import Value
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.kernel import Cuts
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import is_float
from intervals.rounding import is_infinite
from intervals.rounding import round_rational

INF = math.inf

Piece = Tuple[Value, bool, Value, bool]

_ALL = kernel.REALS
_FINITE = kernel.normalize([kernel.piece(-INF, INF, False, False)])
_NON_NEGATIVE = kernel.normalize([kernel.piece(0, INF)])
_UNIT = kernel.normalize([kernel.piece(-1, 1)])
_FROM_ONE = kernel.normalize([kernel.piece(1, INF)])

# name -> (domain, where it is decreasing: 'never', 'always' or 'below 0')
MONOTONE = {
    'sqrt': (_NON_NEGATIVE, 'never'),
    'exp': (_ALL, 'never'),
    'exp2': (_ALL, 'never'),
    'exp10': (_ALL, 'never'),
    'log': (_NON_NEGATIVE, 'never'),
    'log2': (_NON_NEGATIVE, 'never'),
    'log10': (_NON_NEGATIVE, 'never'),
    'asin': (_UNIT, 'never'),
    'acos': (_UNIT, 'always'),
    'atan': (_ALL, 'never'),
    'sinh': (_ALL, 'never'),
    'cosh': (_ALL, 'below 0'),
    'tanh': (_ALL, 'never'),
    'asinh': (_ALL, 'never'),
    'acosh': (_FROM_ONE, 'never'),
    'atanh': (_UNIT, 'never'),
}
PERIODIC = ('sin', 'cos', 'tan')
NAMES = tuple(MONOTONE) + PERIODIC


def apply(name: str, a: Cuts, outward: bool = False, base=None) -> Cuts:
    """
    `{f(x) : x in a}` for f named by `name`; `base` is the logarithm's (a real > 0 other than 1)

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(apply('log', parse('[1/8, 4]'), base=2))
    '[-3, 2]'
    """
    if name not in NAMES:
        raise ValueError(f'unknown function {name!r}')
    if base is not None:
        if name != 'log':
            raise TypeError(f'{name}() takes no base')
        base = _check_base(base)
    if not a:
        warn(EmptySetPropagationWarning, f'{name}: the operand is empty, so the result is empty')
        return kernel.EMPTY
    domain = _FINITE if name in PERIODIC else MONOTONE[name][0]
    inside = kernel.intersection(a, domain)
    if inside != a:
        warn(DomainClippedWarning, f'{name}: points outside its domain {fmt.format_cuts(domain)} were dropped')
    fn = _Function(name, outward, base)
    out = []
    for p in kernel.pieces(inside):
        out.extend(fn.periodic(p) if name in PERIODIC else fn.monotone(p))
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)


def _check_base(base) -> Value:
    if isinstance(base, bool) or not isinstance(base, (int, float, Fraction)):
        raise TypeError(f'the base must be a real number, got {type(base).__name__}')
    if not (0 < base < INF) or base == 1:
        raise ValueError(f'a logarithm needs a finite base > 0 other than 1, got {base!r}')
    return base


class _Function:
    def __init__(self, name: str, outward: bool, base):
        self.name = name
        self.outward = outward
        self.base = base

    # ONE END

    def end(self, x, closed: bool, want: int) -> Tuple[Value, bool]:
        """
        a result end from a piece's end x, as `(f(x), closed)`: `want` is DOWN for a result's low end
        and UP for its high end. an exact x gives the exact value, or the enclosure's end when the value
        is irrational; a float x gives a float, to nearest or (outward) in the `want` direction. an end
        that a directed rounding moved is open, since nothing attains it
        """
        as_float = is_float(x)
        direction = (want if self.outward else NEAREST) if as_float else want
        exact_x = Fraction(x) if as_float else x
        value = elementary.exact(self.name, exact_x, self.base)
        if value is None:  # irrational
            return elementary.rounded(self.name, exact_x, direction, self.base), closed and direction == NEAREST
        if is_infinite(value) or not as_float:
            return value, closed
        rounded = round_rational(value, direction)
        return rounded, closed and (direction == NEAREST or rounded == value)

    def point(self, x) -> Piece:
        lo, lo_closed = self.end(x, True, DOWN)
        hi, hi_closed = self.end(x, True, UP)
        return _settled(lo, lo_closed, hi, hi_closed)

    def between(self, low_end, high_end) -> Piece:
        """the piece from f at one `(x, closed)` end to f at another"""
        lo, lo_closed = self.end(*low_end, DOWN)
        hi, hi_closed = self.end(*high_end, UP)
        return _settled(lo, lo_closed, hi, hi_closed)

    def as_float(self, p: Piece) -> bool:
        return is_float(p[0]) or is_float(p[2])

    # MONOTONE FUNCTIONS

    def increasing(self, p: Piece) -> bool:
        where = MONOTONE[self.name][1]
        if where == 'below 0':
            return p[0] >= 0
        increasing = where == 'never'
        if self.base is not None and self.base < 1:
            increasing = not increasing
        return increasing

    def monotone(self, p: Piece) -> List[Piece]:
        split = (0,) if MONOTONE[self.name][1] == 'below 0' else ()
        return [self.monotone_piece(q) for q in split_pieces([p], split)]

    def monotone_piece(self, p: Piece) -> Piece:
        lo, lo_closed, hi, hi_closed = p
        if lo == hi:
            return self.point(lo)
        ends = (lo, lo_closed), (hi, hi_closed)
        return self.between(*(ends if self.increasing(p) else ends[::-1]))

    # SIN, COS, TAN

    def periodic(self, p: Piece) -> List[Piece]:
        lo, lo_closed, hi, hi_closed = p
        if lo == hi:
            return [self.point(lo)]
        unbounded = is_infinite(lo) or is_infinite(hi)
        if self.name == 'tan':
            return self.tan(p, unbounded)
        one = self.typed(1, p)
        if unbounded:
            return [(-one, True, one, True)]
        # the extrema: sin at pi/2 + k pi, cos at k pi; a maximum for even k
        first, last = self.critical_inside(lo, hi)
        if last - first >= 1:
            return [(-one, True, one, True)]
        ends = [(lo, lo_closed), (hi, hi_closed)]
        if first > last:  # monotone: after a maximum it falls
            rising = (first - 1) % 2 == 1
            return [self.between(*(ends if rising else ends[::-1]))]
        # one extremum inside: it is one end of the result, the lower of the piece's ends the other
        if first % 2 == 0:
            lo, lo_closed = self.end(*self.lower_end(ends), DOWN)
            return [_settled(lo, lo_closed, one, True)]
        hi, hi_closed = self.end(*self.lower_end(ends, sign=-1), UP)
        return [_settled(-one, True, hi, hi_closed)]

    def lower_end(self, ends, sign: int = 1):
        """the end with the lower value (sign=1) or the higher one (sign=-1); closed if either is, on a tie"""
        (a, a_closed), (b, b_closed) = ends
        if self.name == 'cos' and a == -b:  # cos is even: the only way two points share a value
            return a, a_closed or b_closed
        return (a, a_closed) if elementary.compare(self.name, a, b) * sign < 0 else (b, b_closed)

    def critical_inside(self, lo, hi) -> Tuple[int, int]:
        """the k with the critical point strictly inside (lo, hi), as a range `first..last`"""
        offset = Fraction(0) if self.name == 'cos' else Fraction(1, 2)  # cos: k pi, sin and tan: pi/2 + k pi
        k, _ = elementary.floor_over_pi(lo, offset)
        first = k + 1
        k, exact = elementary.floor_over_pi(hi, offset)
        last = k - 1 if exact else k
        return first, last

    def tan(self, p: Piece, unbounded: bool) -> List[Piece]:
        lo, lo_closed, hi, hi_closed = p
        if unbounded:
            return [(-INF, True, INF, True)]
        first, last = self.critical_inside(lo, hi)
        if last - first >= 1:  # a whole branch between two poles
            return [(-INF, True, INF, True)]
        if first > last:
            return [self.between((lo, lo_closed), (hi, hi_closed))]
        # one pole: both infinities attained there, as at a zero inside a divisor
        below, below_closed = self.end(hi, hi_closed, UP)
        above, above_closed = self.end(lo, lo_closed, DOWN)
        return [(-INF, True, below, below_closed), (above, above_closed, INF, True)]

    def typed(self, v: int, p: Piece):
        return float(v) if self.as_float(p) else v


def _settled(lo, lo_closed: bool, hi, hi_closed: bool) -> Piece:
    """a piece rounding squeezed to one point keeps it, closed (as in the applicator)"""
    if lo == hi:
        return lo, True, hi, True
    return lo, lo_closed, hi, hi_closed


# ATAN2

def atan2(y: Cuts, x: Cuts, outward: bool = False) -> Cuts:
    """
    `{atan2(v, u) : v in y, u in x}`, the angle of the point (u, v), in [-pi, pi]

    pointwise, with ±inf as ordinary points where the angle has a limit: `atan2(0, u)` is 0 for u > 0
    and pi for u < 0 (there is no -0, so the negative u axis is at pi, and points just below it are
    near -pi), `atan2(v, 0)` is ±pi/2, `atan2(±inf, u)` is ±pi/2 for a finite u, `atan2(v, inf)` is 0
    and `atan2(v, -inf)` is pi for v >= 0 and -pi for v < 0 (python's values, and the limits).
    `atan2(0, 0)` and `atan2(±inf, ±inf)` have no value: a box that is one of those points gives
    nothing and one `IndeterminateResultWarning`, and a larger box takes the limits along its edges
    there, as the arithmetic does at `0 * inf`.

    both operands split into negative, zero and positive parts, so each box lies in one open quadrant
    or on one half-axis. there the angle is continuous and monotone in each coordinate, so its ends
    are two corners' values, each closed iff its corner is in the box, or it is reached along an edge
    at ±inf, where the angle is constant.

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(atan2(parse('[0, 1]'), parse('[1]')))
    '[0, 0.7853981633974484)'
    """
    if not y or not x:
        warn(EmptySetPropagationWarning, 'atan2: an operand is empty, so the result is empty')
        return kernel.EMPTY
    out, indeterminate = [], []
    for box in product(kernel.pieces(y), kernel.pieces(x)):
        # a box warns only if no point of it has an angle: (0, 0) or (±inf, ±inf) itself
        found = [p for py in _sign_parts(_cuts(box[0])) for px in _sign_parts(_cuts(box[1]))
                 for p in (_angle_box(py, px, outward),) if p is not None]
        if not found:
            indeterminate.append(box)
        out.extend(found)
    if indeterminate:
        shown = ', '.join(fmt.format_piece(*kernel.piece(lo, hi, lo_closed, hi_closed))
                          for lo, lo_closed, hi, hi_closed in indeterminate[0])
        warn(IndeterminateResultWarning, f'atan2({shown}) has no value at any point, so that part contributes nothing')
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)


_NEGATIVE = kernel.normalize([kernel.piece(-INF, 0, True, False)])
_ZERO = kernel.normalize([kernel.piece(0, 0)])
_POSITIVE = kernel.normalize([kernel.piece(0, INF, False, True)])


def _cuts(p: Piece) -> Cuts:
    return kernel.normalize([kernel.piece(p[0], p[2], p[1], p[3])])


def _sign_parts(cuts: Cuts) -> List[Piece]:
    """each piece cut into its negative part, [0] and its positive part (the parts open at 0)"""
    return [q for part in (_NEGATIVE, _ZERO, _POSITIVE) for q in kernel.pieces(kernel.intersection(cuts, part))]


def _side(p: Piece) -> int:
    return 0 if p[0] == p[2] == 0 else 1 if p[0] >= 0 else -1


def _angle(v, u, sv: int, su: int):
    """
    the angle at (u, v) as `(q, m)`, meaning atan(q) + m pi/2, where a 0 end of a positive or negative
    part is the limit from inside the part's quadrant; None at (±inf, ±inf) and (0, 0)
    """
    if sv == 0:
        return None if su == 0 else (0, 0) if su > 0 else (0, 2)
    if su == 0:
        return 0, sv
    if is_infinite(v):
        return None if is_infinite(u) else (0, sv)
    if u == INF:
        return 0, 0
    if u == -INF:
        return 0, 2 * sv
    if v == 0:
        return (0, 0) if su > 0 else (0, 2 * sv)
    if u == 0:
        return 0, sv
    return Fraction(v) / Fraction(u), 0 if su > 0 else 2 * sv


# the corners holding the least and the greatest angle of an open quadrant, as (y end, x end) with
# 0 for a piece's low end and 1 for its high end. the angle rises with y where x > 0 and falls where
# x < 0, and rises with x where y < 0 and falls where y > 0
_EXTREMES = {
    (1, 1): ((0, 1), (1, 0)),
    (1, -1): ((1, 1), (0, 0)),
    (-1, -1): ((1, 0), (0, 1)),
    (-1, 1): ((0, 0), (1, 1)),
}


def _stand_in(p: Piece):
    """a finite point strictly inside a non-degenerate piece of one sign"""
    lo, _, hi, _ = p
    if is_infinite(lo) and is_infinite(hi):
        return 1
    if is_infinite(lo):
        return hi - 1 if hi <= 0 else Fraction(hi) / 2
    if is_infinite(hi):
        return lo + 1 if lo >= 0 else Fraction(lo) / 2
    return (Fraction(lo) + Fraction(hi)) / 2


def _has_finite(p: Piece) -> bool:
    return not (p[0] == p[2] and is_infinite(p[0]))


def _angle_value(a) -> float:
    """an angle's approximate value, only to order two candidates"""
    q, m = a
    return math.atan(float(q)) + m * math.pi / 2


def _angle_box(py: Piece, px: Piece, outward: bool):
    """the angles of one box (one quadrant or half-axis) as a piece; None if no point has one"""
    sv, su = _side(py), _side(px)
    as_float = any(is_float(v) for v in (py[0], py[2], px[0], px[2]))
    if sv == 0 or su == 0:  # on an axis: the same angle at every point
        a = _angle(0 if sv == 0 else py[0], 0 if su == 0 else px[0], sv, su)
        if a is None:
            return None
        lo, lo_moved = _rounded_angle(a, DOWN, as_float, outward)
        hi, hi_moved = _rounded_angle(a, UP, as_float, outward)
        return _settled(lo, not lo_moved, hi, not hi_moved)
    ends = []
    for index, (y_end, x_end) in enumerate(_EXTREMES[sv, su]):
        yv, y_closed = (py[0], py[1]) if y_end == 0 else (py[2], py[3])
        xv, x_closed = (px[0], px[1]) if x_end == 0 else (px[2], px[3])
        a = _angle(yv, xv, sv, su)
        if a is not None:
            # along an edge at ±inf the angle is constant, so a closed infinite end attains it too
            attained = (y_closed and x_closed) or (is_infinite(yv) and y_closed and _has_finite(px)) or \
                (is_infinite(xv) and x_closed and _has_finite(py))
        else:  # (±inf, ±inf): the limits along the edges leaving it, each constant along its edge
            limits = []
            if py[0] != py[2]:
                limits.append((_angle(_stand_in(py), xv, sv, su), x_closed))
            if px[0] != px[2]:
                limits.append((_angle(yv, _stand_in(px), sv, su), y_closed))
            if not limits:
                return None
            pick = min if index == 0 else max
            a, attained = pick(limits, key=lambda item: _angle_value(item[0]))
        ends.append((a, attained))
    (low, low_attained), (high, high_attained) = ends
    lo, lo_moved = _rounded_angle(low, DOWN, as_float, outward)
    hi, hi_moved = _rounded_angle(high, UP, as_float, outward)
    return _settled(lo, low_attained and not lo_moved, hi, high_attained and not hi_moved)


def _rounded_angle(a, want: int, as_float: bool, outward: bool) -> Tuple[Value, bool]:
    """`(value, moved)`: the angle 0 is exact (0.0 when float); any other is rounded"""
    q, m = a
    if q == 0 and m == 0:
        return (0.0 if as_float else 0), False
    direction = (want if outward else NEAREST) if as_float else want
    return elementary.rounded_angle(q, m, direction), direction != NEAREST
