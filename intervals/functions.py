"""
elementary functions over cut tuples: sqrt, exp, log, trig, hyperbolic and their inverses

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
