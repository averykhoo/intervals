"""
elementary functions over cut tuples: sqrt, exp, log, trig, hyperbolic, their reciprocals and
inverses, roots, and the two-argument atan2, pow and hypot

`apply(name, a)` is the set `{f(x) : x in a}`, with ±inf as ordinary points where f has a limit there
(`exp(-inf)` = 0, `atan(inf)` = pi/2, `tanh(inf)` = 1). every endpoint is closed iff attained.

* **domain**: points where f has no value are dropped with one `DomainClippedWarning`: `sqrt` and the
  logs below 0, `log1p` below -1, `asin` `acos` `atanh` outside [-1, 1], `acoth` inside (-1, 1),
  `acosh` below 1, an even root below 0, and the six trigonometric functions at ±inf (no limit).
  the end of a domain is a point of it wherever f has a limit there, so `log([0, 1])` = `[-inf, 0]`,
  `sqrt([-1, 4])` = `[0, 2]` + warning, and `atanh([1])` = `[inf]`: the domain reaches 0 and 1 from
  one side only, so there is no second limit to disagree with
* **shape**: each function is monotone on each piece once split where its direction changes (cosh
  and sech at 0); a monotone continuous function maps a piece to the piece between its ends' images,
  each end closed iff the piece's end is. sin, cos, csc and sec also attain their extrema (±1) inside
  a piece, and tan, cot, csc and sec map a piece holding a pole to both sides of it with both
  infinities attained, as `1/x` does at a zero inside a piece. coth, csch and an odd negative root
  have a pole at 0 with a side each way, as do cot and csc: a piece ending at 0 takes the one-sided
  limit there (closed iff the piece holds 0, as `1/[0, 1]` = `[1, inf]`), and the point 0 alone has
  no value (`IndeterminateResultWarning`, as `1/[0]`)
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
from numbers import Integral
from numbers import Real
from typing import List
from typing import Optional
from typing import Tuple

from intervals import elementary
from intervals import fmt
from intervals import kernel
from intervals import ops
from intervals.applicator import split_pieces
from intervals.applicator import warn
from intervals.cuts import Value
from intervals.cuts import is_numpy_time
from intervals.cuts import normalize_value
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import PowerLimitWarning
from intervals.kernel import Cuts
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import exact_cuts
from intervals.rounding import has_finite_float
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
_FROM_MINUS_ONE = kernel.normalize([kernel.piece(-1, INF)])
_OUTSIDE_UNIT = kernel.normalize([kernel.piece(-INF, -1), kernel.piece(1, INF)])

# name -> (domain, where it is decreasing: 'never', 'always', 'below 0' or 'above 0')
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
    'expm1': (_ALL, 'never'),
    'log1p': (_FROM_MINUS_ONE, 'never'),
    'cbrt': (_ALL, 'never'),
    'acot': (_ALL, 'always'),
    'sech': (_ALL, 'above 0'),
    'acoth': (_OUTSIDE_UNIT, 'always'),
}
PERIODIC = ('sin', 'cos', 'tan')
# periodic with a pole in each period: name -> (the poles' offset, the extrema's, as multiples of pi)
RECIPROCAL_TRIG = {'cot': (Fraction(0), None), 'csc': (Fraction(0), Fraction(1, 2)), 'sec': (Fraction(1, 2), Fraction(0))}
# falling on each side of a pole at 0 with a side each way: -inf just below 0, +inf just above
POLE_AT_ZERO = ('coth', 'csch')
NAMES = tuple(MONOTONE) + PERIODIC + tuple(RECIPROCAL_TRIG) + POLE_AT_ZERO


def domain(name: str, base=None) -> Cuts:
    """
    the points of the reals where f has a value or a one-sided limit (`base` is rootn's n)

    >>> from intervals.fmt import format_cuts
    >>> format_cuts(domain('acoth')), format_cuts(domain('rootn', -2))
    ('{ [-inf, -1] , [1, inf] }', '[0, inf]')
    """
    if name == 'rootn':
        return _ALL if base % 2 else _NON_NEGATIVE
    if name in PERIODIC or name in RECIPROCAL_TRIG:
        return _FINITE
    if name in POLE_AT_ZERO:
        return _ALL
    return MONOTONE[name][0]


def apply(name: str, a: Cuts, outward: bool = False, base=None) -> Cuts:
    """
    `{f(x) : x in a}` for f named by `name`, one of `NAMES` or `rootn`; `base` is the logarithm's (a
    real > 0 other than 1), and for `rootn` its degree n (an int other than 0), `rootn(x, n)` being
    the real n-th root of x (of `1/x` for n < 0), which needs x >= 0 for an even n

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(apply('log', parse('[1/8, 4]'), base=2))
    '[-3, 2]'
    >>> format_cuts(apply('rootn', parse('[-8, 27]'), base=3))
    '[-2, 3]'
    >>> format_cuts(apply('coth', parse('[-1, 1]')))
    '{ [-inf, -1.3130352854993312) , (1.3130352854993312, inf] }'
    """
    if name not in NAMES and name != 'rootn':
        raise ValueError(f'unknown function {name!r}')
    if name == 'rootn':
        base = _check_degree(base)
    elif base is not None:
        if name != 'log':
            raise TypeError(f'{name}() takes no base')
        base = _check_base(base)
    if not a:
        warn(EmptySetPropagationWarning, f'{name}: the operand is empty, so the result is empty')
        return kernel.EMPTY
    where = domain(name, base)
    inside = kernel.intersection(a, where)
    if inside != a:
        warn(DomainClippedWarning, f'{name}: points outside its domain {fmt.format_cuts(where)} were dropped')
    fn = _Function(name, outward, base)
    out = []
    for p in kernel.pieces(inside):
        if name in PERIODIC:
            out.extend(fn.periodic(p))
        elif name in RECIPROCAL_TRIG:
            out.extend(fn.reciprocal_trig(p))
        elif fn.where == 'pole':
            out.extend(fn.pole_at_zero(p))
        else:
            out.extend(fn.monotone(p))
    if fn.poles_hit:
        warn(IndeterminateResultWarning, f'{name}([0]) has no value (a pole with a side each way), so that part '
                                         f'contributes nothing')
    if fn.too_long:
        warn(PowerLimitWarning, f'{name}: the power of an exact operand is longer than '
                                f'{elementary.EXACT_RESULT_LIMIT} bits, so its tightest float enclosure was returned')
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)


def _check_base(base) -> Value:
    """any real but bool, as the python number `normalize_value` makes it (so a numpy float never
    reaches the functions with numpy's own arithmetic)"""
    if isinstance(base, bool) or not isinstance(base, Real):
        raise TypeError(f'the base must be a real number, got {type(base).__name__}')
    try:
        value = normalize_value(base)
    except ValueError:  # nan
        value = None
    if value is None or not (0 < value < INF) or value == 1:
        raise ValueError(f'a logarithm needs a finite base > 0 other than 1, got {base!r}')
    return value


def _check_degree(n) -> int:
    """any `Integral` but bool, as an int (numpy's ints included, as `ops.power` takes them)"""
    if isinstance(n, bool) or not isinstance(n, Integral) or is_numpy_time(n):  # a timedelta64 is numpy's Integral
        raise TypeError(f'the degree of a root must be an int, got {type(n).__name__}')
    n = int(n)
    if n == 0:
        raise ValueError('rootn(x, 0) has no value: the degree must be an int other than 0')
    return n


class _Function:
    def __init__(self, name: str, outward: bool, base, float_operands: Optional[bool] = None):
        """`float_operands` forces the float rules on every end (hypot: exact ends of float operands)"""
        self.name = name
        self.outward = outward
        self.base = base
        self.float_operands = float_operands
        self.poles_hit = False  # a piece that is the point 0 of a pole with a side each way
        self.too_long = False  # exp2/exp10 of an exact int past elementary.EXACT_RESULT_LIMIT
        if name == 'rootn':
            self.where = 'never' if base > 0 else 'always' if base % 2 == 0 else 'pole'
        elif name in POLE_AT_ZERO:
            self.where = 'pole'
        else:
            self.where = MONOTONE[name][1] if name in MONOTONE else None

    # ONE END

    def end(self, x, closed: bool, want: int) -> Tuple[Value, bool]:
        """
        a result end from a piece's end x, as `(f(x), closed)`: `want` is DOWN for a result's low end
        and UP for its high end. an exact x gives the exact value, or the enclosure's end when the value
        is irrational, or (exp2, exp10) rational past `elementary.EXACT_RESULT_LIMIT`; a float x gives a
        float, to nearest or (outward) in the `want` direction. an end that a directed rounding moved is
        open, since nothing attains it
        """
        as_float = is_float(x) if self.float_operands is None else self.float_operands and not is_infinite(x)
        direction = (want if self.outward else NEAREST) if as_float else want
        exact_x = Fraction(x) if as_float else x
        limit = elementary.EXACT_POWER_LIMIT if as_float else elementary.EXACT_RESULT_LIMIT
        value = elementary.exact(self.name, exact_x, self.base, limit)
        if value is None:  # irrational, or a power too long to build
            if not as_float and self.name in elementary.EXP_BASES and not is_infinite(x) \
                    and Fraction(x).denominator == 1:
                self.too_long = True
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
        if self.where == 'below 0':
            return p[0] >= 0
        if self.where == 'above 0':
            return p[2] <= 0
        increasing = self.where == 'never'
        if self.name == 'log' and self.base is not None and self.base < 1:
            increasing = not increasing
        return increasing

    def monotone(self, p: Piece) -> List[Piece]:
        split = (0,) if self.where in ('below 0', 'above 0') else ()
        return [self.monotone_piece(q) for q in split_pieces([p], split)]

    # A POLE AT 0 WITH A SIDE EACH WAY (coth, csch, an odd negative root: falling on each side)

    def pole_at_zero(self, p: Piece) -> List[Piece]:
        lo, lo_closed, hi, hi_closed = p
        if lo == hi == 0:
            self.poles_hit = True
            return []
        if lo >= 0 or hi <= 0:
            if lo == 0 or hi == 0:  # a piece ending at the pole: the one limit on its side
                if lo == 0:
                    below, below_closed = self.end(hi, hi_closed, DOWN)
                    return [_settled(below, below_closed, INF, lo_closed)]
                above, above_closed = self.end(lo, lo_closed, UP)
                return [_settled(-INF, hi_closed, above, above_closed)]
            return [self.monotone_piece(p)]
        # the pole inside: both infinities attained, as at a zero inside a divisor
        above, above_closed = self.end(lo, lo_closed, UP)
        below, below_closed = self.end(hi, hi_closed, DOWN)
        return [_settled(-INF, True, above, above_closed), _settled(below, below_closed, INF, True)]

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
        return _inside_k(lo, hi, offset)

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

    # COT, CSC, SEC

    def reciprocal_trig(self, p: Piece) -> List[Piece]:
        """
        the piece cut at its poles and extrema into monotone segments, each mapped between the values
        at its ends: a piece's end gives f there (the one-sided limit at the pole 0 of cot and csc,
        closed iff the end is), a pole inside the piece its two infinities and an extremum its ±1,
        both attained. three poles inside hold a whole period, so the whole range
        """
        lo, lo_closed, hi, hi_closed = p
        pole_offset, extremum_offset = RECIPROCAL_TRIG[self.name]
        pole_at_zero = pole_offset == 0
        if lo == hi:
            if lo == 0 and pole_at_zero:
                self.poles_hit = True
                return []
            return [self.point(lo)]
        one = self.typed(1, p)
        whole = [(-INF, True, INF, True)] if self.name == 'cot' else [(-INF, True, -one, True), (one, True, INF, True)]
        if is_infinite(lo) or is_infinite(hi):
            return whole
        first, last = _inside_k(lo, hi, pole_offset)
        if last - first >= 2:
            return whole
        # the breaks inside the piece in order, each as (where, its value from the left, from the right)
        breaks = [(k + pole_offset, -self.after_pole(k), self.after_pole(k)) for k in range(first, last + 1)]
        if extremum_offset is not None:
            e_first, e_last = _inside_k(lo, hi, extremum_offset)
            breaks += [(k + extremum_offset, one if k % 2 == 0 else -one, one if k % 2 == 0 else -one)
                       for k in range(e_first, e_last + 1)]
        breaks.sort(key=lambda b: b[0])
        # a piece's end: None means f at it, to be rounded; the pole 0 gives its one-sided limit
        start = (INF, lo_closed) if lo == 0 and pole_at_zero else None
        stop = (-INF, hi_closed) if hi == 0 and pole_at_zero else None
        lefts = [start] + [(b[2], True) for b in breaks]  # the value at each segment's left end
        rights = [(b[1], True) for b in breaks] + [stop]
        out = []
        for i, (left, right) in enumerate(zip(lefts, rights)):
            left_x = (lo, lo_closed) if i == 0 else None
            right_x = (hi, hi_closed) if i == len(breaks) else None
            out.append(self.segment(left, left_x, right, right_x))
        return out

    def after_pole(self, k: int) -> float:
        """f's limit just after its k-th pole, ±inf (cot and csc: at k pi; sec: at pi/2 + k pi)"""
        if self.name == 'cot':
            return INF
        after = 1 if (k % 2 == 0) == (self.name == 'csc') else -1
        return after * INF

    def segment(self, left, left_x, right, right_x) -> Piece:
        """
        one monotone segment: `left` and `right` are `(value, closed)` where known (a pole's limit, an
        extremum, the pole 0), else None, and then `left_x`/`right_x` is the piece's `(end, closed)`
        """
        if left is not None and right is not None:
            (a, a_closed), (b, b_closed) = sorted((left, right), key=lambda v: v[0])
            return _settled(a, a_closed, b, b_closed)
        if left is None and right is None:
            ends = [left_x, right_x]
            low_end = self.lower_end(ends)
            high_end = right_x if low_end[0] == left_x[0] else left_x
            return self.between(low_end, high_end)
        known, x_end = (left, right_x) if right is None else (right, left_x)
        v, v_closed = known
        # +inf, and -1 (the top of a negative branch), sit above the rest of their segment
        if v == INF or v == -1:
            lo, lo_closed = self.end(*x_end, DOWN)
            return _settled(lo, lo_closed, v, v_closed)
        hi, hi_closed = self.end(*x_end, UP)
        return _settled(v, v_closed, hi, hi_closed)

    def typed(self, v: int, p: Piece):
        return float(v) if self.as_float(p) else v


def _inside_k(lo, hi, offset: Fraction) -> Tuple[int, int]:
    """the k with `(k + offset) pi` strictly inside (lo, hi), as a range `first..last`"""
    k, _ = elementary.floor_over_pi(lo, offset)
    first = k + 1
    k, exact = elementary.floor_over_pi(hi, offset)
    last = k - 1 if exact else k
    return first, last


def _settled(lo, lo_closed: bool, hi, hi_closed: bool) -> Piece:
    """
    a piece rounding squeezed to one point keeps it, closed; one whose ends rounding crossed is the piece between
    the two values, each keeping its flag (both as in the applicator, whose ends are the least and greatest corner
    values). ends cross only to nearest, an exact end beside a float end rounded past it: `rootn((10 ** -30,
    1.0000000000000003e-30], 5)` is `[1e-06, 1/1000000)` (fuzz x10 on CI, 2026-10-03; a reversed piece raised)
    """
    if lo == hi:
        return lo, True, hi, True
    if lo > hi:
        return hi, hi_closed, lo, lo_closed
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


# POW (ieee 1788's: the real power, defined for x > 0, and for x = 0 where y > 0)

_ZERO_BASE = kernel.normalize([kernel.piece(0, 0)])
_BELOW_ONE = kernel.normalize([kernel.piece(0, 1, False, False)])
_ONE = kernel.normalize([kernel.piece(1, 1)])
_ABOVE_ONE = kernel.normalize([kernel.piece(1, INF, False, True)])

# the corners holding the least and the greatest power in a box, by (the base's side of 1, the
# exponent's sign), as (x end, y end) with 0 for a piece's low end and 1 for its high end: x ** y
# rises with x where y > 0 and falls where y < 0, and rises with y where x > 1 and falls where x < 1
_POW_EXTREMES = {
    (1, 1): ((0, 0), (1, 1)),
    (1, -1): ((1, 0), (0, 1)),
    (-1, 1): ((0, 1), (1, 0)),
    (-1, -1): ((1, 1), (0, 0)),
}
_OUTSIDE = object()  # a box outside pow's domain
_TOO_LONG = object()  # a rational power longer than its limit


def pow_(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    `{x ** y : x in a, y in b}`, ieee 1788's pow: defined for x > 0, and for x = 0 where y > 0
    (`0 ** y` = 0). the pairs outside that are dropped with one `DomainClippedWarning` (a negative
    base, or `0 ** y` for y <= 0). ±inf are points where the power has a limit: `inf ** y` is inf for
    y > 0 and 0 for y < 0, `x ** inf` is 0 for x < 1 and inf for x > 1 (the reverse at -inf);
    `1 ** ±inf` and `inf ** 0` have none, so a box that is one of those points gives nothing and one
    `IndeterminateResultWarning`, as `atan2` does at its own.

    the base splits at 0 and 1 and the exponent at 0, so x lies in [0], (0, 1), [1] or (1, inf] and y
    has one sign or is 0. the power is 0 at x = 0 and 1 at x = 1 or y = 0; in the other boxes it is
    monotone in each coordinate, so its ends are two corners' values (the limit where the corner is
    not in the box), each closed iff the corner is in the box or on a closed infinite edge, along
    which the power is constant. exact operands give an exact end where it is rational
    (`[4] ** [1/2]` = `[2]`) and the tightest float enclosure where not; float operands round once,
    to nearest, or outward with `outward=True`

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(pow_(parse('[1/4, 4]'), parse('[1/2]')))
    '[1/2, 2]'
    >>> format_cuts(pow_(parse('[0, 2]'), parse('[1/2, 1]')))
    '[0, 2]'
    >>> format_cuts(pow_(parse('[2]'), parse('[1/2]')))
    '(1.414213562373095, 1.4142135623730951)'
    """
    if not a or not b:
        warn(EmptySetPropagationWarning, 'pow: an operand is empty, so the result is empty')
        return kernel.EMPTY
    inside = kernel.intersection(a, _NON_NEGATIVE)
    clipped = inside != a
    as_float = has_finite_float(a) or has_finite_float(b)
    out, indeterminate, too_long = [], [], []
    for box in product(kernel.pieces(inside), kernel.pieces(b)):
        found, undefined = [], False
        for sx, px in _tagged_parts(box[0], ((0, _ZERO_BASE), (-1, _BELOW_ONE), (None, _ONE), (1, _ABOVE_ONE))):
            for sy, py in _tagged_parts(box[1], ((-1, _NEGATIVE), (0, _ZERO), (1, _POSITIVE))):
                p = _power_box(px, py, sx, sy, as_float, outward, too_long)
                if p is _OUTSIDE:
                    clipped = True
                elif p is None:
                    undefined = True
                else:
                    found.append(p)
        if undefined and not found:
            indeterminate.append(box)
        out.extend(found)
    if clipped:
        warn(DomainClippedWarning, 'pow: pairs outside its domain (x > 0, or x = 0 with y > 0) were dropped')
    if indeterminate:
        shown = ', '.join(fmt.format_piece(*kernel.piece(lo, hi, lo_closed, hi_closed))
                          for lo, lo_closed, hi, hi_closed in indeterminate[0])
        warn(IndeterminateResultWarning, f'pow({shown}) has no value at any point, so that part contributes nothing')
    if too_long:
        warn(PowerLimitWarning, f'pow: the power of exact operands is longer than {elementary.EXACT_RESULT_LIMIT} '
                                f'bits, so its tightest float enclosure was returned')
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)


def _tagged_parts(p: Piece, parts) -> List[Tuple[object, Piece]]:
    """the piece cut into its parts, each with the part's tag"""
    return [(tag, q) for tag, part in parts for q in kernel.pieces(kernel.intersection(_cuts(p), part))]


def _power_box(px: Piece, py: Piece, sx, sy: int, as_float: bool, outward: bool, too_long: Optional[list] = None):
    """
    the powers of one box as a piece; None if no point has one, `_OUTSIDE` if it is outside the domain.
    a corner of exact operands whose power is too long to build appends to `too_long`, if given
    """
    if sx == 0:
        return _OUTSIDE if sy <= 0 else _point(0, as_float)
    if sx is None:  # 1 ** y
        return _point(1, as_float) if _has_finite(py) else None
    if sy == 0:  # x ** 0
        return _point(1, as_float) if _has_finite(px) else None
    ends = []
    for want, (x_end, y_end) in zip((DOWN, UP), _POW_EXTREMES[sx, sy]):
        xv, x_closed = (px[0], px[1]) if x_end == 0 else (px[2], px[3])
        yv, y_closed = (py[0], py[1]) if y_end == 0 else (py[2], py[3])
        # along a closed infinite edge the power is constant, so any point of it attains the corner's value
        attained = (x_closed and y_closed) or (xv == INF and x_closed) or (is_infinite(yv) and y_closed)
        ends.append(_power_corner(xv, yv, sx, sy, want, attained, as_float, outward, too_long))
    (lo, lo_closed), (hi, hi_closed) = ends
    return _settled(lo, lo_closed, hi, hi_closed)


def _point(v: int, as_float: bool) -> Piece:
    v = float(v) if as_float else v
    return v, True, v, True


def _power_corner(x, y, sx: int, sy: int, want: int, attained: bool, as_float: bool, outward: bool,
                  too_long: Optional[list] = None):
    """
    `(value, closed)` at a corner of a box with x on side `sx` of 1 and y of sign `sy`. exact operands
    build a rational power while it is at most `elementary.EXACT_RESULT_LIMIT` bits, the limit pown and
    exp2/exp10 share, and past it round it like an irrational one (appending to `too_long`); with a float
    operand the limit is `elementary.EXACT_POWER_LIMIT`, a speed choice: the double is the same
    """
    direction = (want if outward else NEAREST) if as_float else want
    if x == INF:
        value = INF if sy > 0 else 0
    elif is_infinite(y):
        value = INF if (y > 0) == (sx > 0) else 0
    elif x == 0:  # the limit at an open end of (0, 1)
        value = 0 if sy > 0 else INF
    elif x == 1 or y == 0:  # the limit at an open end of a part
        value = 1
    else:
        limit = elementary.EXACT_POWER_LIMIT if as_float else elementary.EXACT_RESULT_LIMIT
        value = elementary.exact_pow(Fraction(x), Fraction(y), limit, _TOO_LONG)
        if value is _TOO_LONG:  # rational, but too long to build
            value = None
            if too_long is not None and not as_float:
                too_long.append((x, y))
        if value is None:  # irrational, or too long
            return elementary.rounded_pow(Fraction(x), Fraction(y), direction), attained and direction == NEAREST
    if is_infinite(value) or not as_float:
        return value, attained
    rounded = round_rational(value, direction)
    return rounded, attained and (direction == NEAREST or rounded == value)


# HYPOT

def hypot(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    `{sqrt(x**2 + y**2) : x in a, y in b}`: the sums of squares as an exact set (a float operand as the
    rational it is), then one square root, exact where rational for exact operands and else the
    tightest float enclosure; with a float operand it is rounded once, to nearest or (outward) outward

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(hypot(parse('[3, 5]'), parse('[-4, 0]')))
    '[3, 6.403124237432849)'
    """
    if not a or not b:
        warn(EmptySetPropagationWarning, 'hypot: an operand is empty, so the result is empty')
        return kernel.EMPTY
    squares = ops.add(ops.power(exact_cuts(a), 2), ops.power(exact_cuts(b), 2))
    fn = _Function('sqrt', outward, None, float_operands=has_finite_float(a) or has_finite_float(b))
    out = [q for p in kernel.pieces(squares) for q in fn.monotone(p)]
    return kernel.normalize(kernel.piece(lo, hi, lo_closed, hi_closed) for lo, lo_closed, hi, hi_closed in out)
