"""
modulo, floor and floor division over cut tuples

`mod(A, B)` is the set of values `x mod y = x - y * floor(x / y)` over x in A and y in B: python's
floor-mod, so a result has the divisor's sign. every endpoint is closed iff attained. pointwise, with
±inf as ordinary points (D8):

* `±inf mod y` has no value (python gives nan), and neither has `x mod 0`. an operand that loses
  points this way emits one `DomainClippedWarning`
* a finite `x mod inf` is x for x >= 0 and inf for x < 0; `x mod -inf` is x for x <= 0 and -inf for
  x > 0. these are python's values, and the limits along the box (`-3 mod y = y - 3` for y >= 3)

one rule is behind both, and behind `//` below: at an infinite operand, a pair's value is the limit of
the values of its finite neighbours, and a pair with no limit has no value (`inf mod y` oscillates).
python agrees except where its float arithmetic gives nan for a limit that exists (`inf // 3`).

the algorithm is the far-edge one derived in `references/modulo-derivations/claude-fable/`:

1. split the dividend at 0 (0 in both halves) and drop 0 from the divisor, so every box of one
   dividend piece and one divisor piece lies in one closed quadrant. a box with a negative divisor is
   the antipodal image of one with a positive divisor: `(-x) mod (-y) = -(x mod y)`
2. locate the box's image with every end treated closed. the image of a box is the image of its two
   edges furthest from the origin (right and top for a dividend >= 0, left and top for a dividend
   <= 0), and each edge is an interval-mod-scalar or scalar-mod-interval primitive in closed form
3. close each end of each located piece iff some point of the box attains it, by an exact test that
   costs O(1) whatever the operands' size; a degenerate piece whose value is not attained is dropped.
   ends are decided per located piece, before the union, so an unattained value where two pieces
   meet stays a hole
4. union the boxes and normalize

finite values are computed exactly (a float as the Fraction it denotes). a float operand makes the
result float, rounded once at the end (to nearest, or outward with `outward=True`), and a rounded
end's flag is conservative, not a promise.

`floor` enumerates the integers a set holds, up to `FLOOR_ENUMERATION_CAP` of them; above that, or
for an unbounded piece, it returns their hull with a `HullWarning` (`intervals.steps`, which also has
ceil, trunc, round and sign).

`floordiv` is `floor(div(A, B))` over the finite divisors, so it follows `div` at a zero divisor
(`[1] // [0, 1]` holds inf) and gives `inf // 3` = inf where python gives nan. an infinite divisor
takes the limit instead, as python does: `x // inf` is -1 for x < 0 and 0 for x >= 0, `x // -inf` is -1
for x > 0 and 0 for x <= 0. **so `//` is not `floor(div)` at an infinite divisor**: `-5 / inf` is 0, a
point with no side, and `floor([-5] / [inf])` is `[0]`, while `[-5] // [inf]` is `[-1]`, the limit of
`floor(-5 / y)` = -1 as y grows. the limit is what keeps the infinite point continuous with its
neighbours (`[-5] // [1, inf]` = `[-5] // [1, inf)`, where floor(div) would add a stray 0) and keeps
`divmod(-5, inf)` = `(-1, inf)`, both parts limits of the same finite pairs. it is the `1/[0]` story
again: the direction a value was approached from is lost at a degenerate point, here the 0 of
`-5 / inf`. `x = q * y + r` itself cannot hold at y = inf (`-1 * inf + inf` has no value).
`divmod` is the pair of the two sets, which does not remember which quotient went with which remainder.

>>> from intervals.fmt import format_cuts, parse
>>> format_cuts(mod(parse('[12, 37/2]'), parse('[15/2]')))
'{ [0, 7/2] , [9/2, 15/2) }'
>>> format_cuts(mod(parse('[3, 7]'), parse('{ [4] , [5] }')))
'[0, 5)'
>>> format_cuts(mod(parse('[-7, -3]'), parse('[2, 5]')))
'[0, 5)'
>>> format_cuts(floor(parse('[1, 2)')))
'[1]'
"""
import math
from fractions import Fraction
from itertools import product
from typing import List
from typing import Tuple

from intervals import kernel
from intervals import ops
from intervals import steps
from intervals.applicator import is_infinite
from intervals.applicator import split_pieces
from intervals.applicator import warn
from intervals.cuts import Value
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.kernel import Cuts
from intervals.rounding import exact_cuts
from intervals.rounding import float_cuts
from intervals.rounding import has_finite_float
from intervals.rounding import round_piece

INF = math.inf

Piece = Tuple[Value, bool, Value, bool]
Span = Tuple[Value, Value]  # a located piece, both ends treated closed

FLOOR_ENUMERATION_CAP = steps.ENUMERATION_CAP

_FINITE: Cuts = kernel.normalize([kernel.piece(-INF, INF, False, False)])
_NONZERO: Cuts = kernel.complement(kernel.normalize([kernel.piece(0, 0)]))
_POSITIVE: Piece = (0, False, INF, False)


# MOD

def mod(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    `{x mod y : x in a, y in b}`; see the module docstring

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(mod(parse('[3, 8]'), parse('[8, 12]')))  # 8 mod 8 is 0
    '{ [0] , [3, 8] }'
    >>> format_cuts(mod(parse('[3, 8]'), parse('(8, 12]')))
    '[3, 8]'
    >>> format_cuts(mod(parse('[1, 2]'), parse('(3, 4)')))
    '[1, 2]'
    >>> format_cuts(mod(parse('[-3]'), parse('[inf]')))
    '[inf]'
    """
    if not a or not b:
        warn(EmptySetPropagationWarning, 'mod: an operand is empty, so the result is empty')
        return kernel.EMPTY
    dividend = kernel.intersection(a, _FINITE)
    divisor = kernel.intersection(b, _NONZERO)
    if dividend != a:
        warn(DomainClippedWarning, 'mod: ±inf mod y has no value, so the dividend\'s infinite points were dropped')
    if divisor != b:
        warn(DomainClippedWarning, 'mod: x mod 0 has no value, so 0 was dropped from the divisor')
    rounded = has_finite_float(a) or has_finite_float(b)
    xs = split_pieces([_exact(p) for p in kernel.pieces(dividend)], (0,))
    ys = [_exact(p) for p in kernel.pieces(divisor)]
    out = []
    for x, y in product(xs, ys):
        for p in _box(x, y):
            lo, lo_closed, hi, hi_closed = round_piece(p, outward) if rounded else p
            out.append(kernel.piece(lo, hi, lo_closed, hi_closed))
    return kernel.normalize(out)


def _box(x: Piece, y: Piece) -> List[Piece]:
    """the image of one box: x on one side of 0 (or [0]), y on one side of 0 and not holding it"""
    if y[2] <= 0:  # 0 itself was dropped, so this piece is negative
        return [_negated(p) for p in _box(_negated(x), _negated(y))]
    if y[0] == INF:  # y is [inf]
        out = [x] if x[0] >= 0 else [(INF, True, INF, True)]
        if x[0] < 0 and x[2] == 0 and x[3]:
            out.append((0, True, 0, True))
        return out
    out = []
    for lo, hi in _shape(x, y):
        lo_closed, hi_closed = _attained(lo, x, y), _attained(hi, x, y)
        if lo < hi or (lo_closed and hi_closed):
            out.append((lo, lo_closed, hi, hi_closed))
    return out


# SHAPE: the image of a box with every end treated closed, y > 0

def _shape(x: Piece, y: Piece) -> List[Span]:
    x0, _, x1, _ = x
    y0, _, y1, _ = y
    if x0 >= 0:
        if x1 == INF:  # every residue below any y of B
            return [(0, y1)]
        return _interval_mod_scalar(x0, x1, y1) + _scalar_mod_interval_nonnegative(x1, y0, y1)
    if x0 == -INF:
        return [(0, y1)]
    return _interval_mod_scalar(x0, x1, y1) + _scalar_mod_interval_negative(x0, y0, y1)


def _interval_mod_scalar(x0, x1, m) -> List[Span]:
    """`[x0, x1] mod m` for one-signed x and m > 0 (the top edge); m may be inf"""
    if m == INF:
        if x0 >= 0:
            return [(x0, x1)]
        return [(INF, INF)] + ([(0, 0)] if x1 == 0 else [])
    n = math.floor(x1 / m) - math.floor(x0 / m)
    r0, r1 = _fmod(x0, m), _fmod(x1, m)
    if n == 0:
        return [(r0, r1)]
    if n == 1:
        return [(0, r1), (r0, m)]
    return [(0, m)]


def _scalar_mod_interval_nonnegative(c, y0, y1) -> List[Span]:
    """
    `c mod [y0, y1]` for c >= 0 (the right edge). with a = floor(c / y1) and b = floor(c / y0) the
    quotient runs from a to b as y falls; each quotient k is one sector, where `c - k * y` rises from 0
    to `c / k` as y falls, so the sectors past the first two are covered by the one next to y1
    """
    if c == 0:
        return [(0, 0)]
    a = 0 if y1 == INF else math.floor(c / y1)
    b = INF if y0 == 0 else math.floor(c / y0)
    if a == b:
        return [(_fmod(c, y1), _fmod(c, y0))]
    right = (c, c) if a == 0 else (_fmod(c, y1), c / (a + 1))
    if b == a + 1:
        return [(0, _fmod(c, y0)), right]
    return [(0, c / (a + 2)), right]


def _scalar_mod_interval_negative(c, y0, y1) -> List[Span]:
    """
    `c mod [y0, y1]` for c < 0 (the left edge). on the sector floor(c / y) = -m, `c + m * y` rises
    with y, from 0 at y = -c/m to `-c / (m - 1)` (unbounded for m = 1). with m_lo = -floor(c / y1)
    and m_hi = -floor(c / y0) the sector m_lo holds y1, the sector m_lo + 1 is the largest of the
    ones below it, and every sector below that gives a subset of it
    """
    m_lo = 1 if y1 == INF else -math.floor(c / y1)
    m_hi = INF if y0 == 0 else -math.floor(c / y0)
    top = _fmod(c, y1)
    if m_lo == m_hi:
        return [(_fmod(c, y0), top)]
    if m_hi == m_lo + 1:
        return [(0, top), (_fmod(c, y0), -c / m_lo)]
    return [(0, top), (0, -c / m_lo)]


def _fmod(x, y):
    """x mod y for finite x and y > 0, exact; y may be inf"""
    if y == INF:
        return x if x >= 0 else INF
    return x - y * math.floor(x / y)


# ATTAINMENT: exact, y > 0

def _attained(v, x: Piece, y: Piece) -> bool:
    """
    does some point of the box give v? with y > 0 that needs 0 <= v < y and `x = v + k * y` for an
    integer k: k = 0 is v in x, k >= 1 puts a multiple of y in `x - v`, k <= -1 one in `v - x`
    """
    y0, y0_closed, y1, y1_closed = y
    has_inf = y1 == INF and y1_closed
    if v == INF:
        return has_inf and x[0] < 0
    if v < 0:
        return False
    if has_inf and _contains(x, v):
        return True
    beyond = _meet((y0, y0_closed, y1, y1_closed and y1 != INF), (v, False, INF, False))
    if not _nonempty(beyond):
        return False
    if _contains(x, v):
        return True
    x0, x0_closed, x1, x1_closed = x
    return (_holds_multiple((x0 - v, x0_closed, x1 - v, x1_closed), beyond)
            or _holds_multiple((v - x1, x1_closed, v - x0, x0_closed), beyond))


def _holds_multiple(c: Piece, j: Piece) -> bool:
    """
    is `k * y` in c for some integer k >= 1 and y in j, where j is a non-empty piece of finite
    positive values? the k with `k * j` meeting c's hull run from ceil(inf c / sup j) to
    floor(sup c / inf j), and any k strictly between those two meets c's interior, so only the two
    ends need an exact check
    """
    c = _meet(c, _POSITIVE)
    if not _nonempty(c):
        return False
    c0, _, c1, _ = c
    j0, j0_closed, j1, j1_closed = j
    if j0 == 0 or c1 == INF:  # an arbitrarily small y, or an arbitrarily large multiple
        return True
    if j1 == INF:  # k * j lies inside j for every k, so k = 1 is enough
        return _nonempty(_meet(c, j))
    k_min, k_max = max(1, math.ceil(c0 / j1)), math.floor(c1 / j0)
    if k_max - k_min >= 2:
        return True
    return any(_nonempty(_meet(c, (k * j0, j0_closed, k * j1, j1_closed))) for k in range(k_min, k_max + 1))


# PIECES

def _exact(p) -> Piece:
    lo, lo_closed, hi, hi_closed = p
    return _exact_value(lo), lo_closed, _exact_value(hi), hi_closed


def _exact_value(v):
    return v if is_infinite(v) else Fraction(v)


def _negated(p: Piece) -> Piece:
    lo, lo_closed, hi, hi_closed = p
    return -hi, hi_closed, -lo, lo_closed


def _contains(p: Piece, v) -> bool:
    lo, lo_closed, hi, hi_closed = p
    return lo < v < hi or (v == lo and lo_closed) or (v == hi and hi_closed)


def _nonempty(p: Piece) -> bool:
    lo, lo_closed, hi, hi_closed = p
    return lo < hi or (lo == hi and lo_closed and hi_closed)


def _meet(p: Piece, q: Piece) -> Piece:
    (a, a_closed, b, b_closed), (c, c_closed, d, d_closed) = p, q
    lo, lo_closed = (a, a_closed) if a > c else (c, c_closed) if c > a else (a, a_closed and c_closed)
    hi, hi_closed = (b, b_closed) if b < d else (d, d_closed) if d < b else (b, b_closed and d_closed)
    return lo, lo_closed, hi, hi_closed


# FLOOR, FLOORDIV, DIVMOD

def floor(a: Cuts) -> Cuts:
    """
    `{floor(x) : x in a}`, with floor(±inf) = ±inf; a float keeps its type (`floor(2.5)` is 2.0)

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(floor(parse('{ (-1, 1/2] , (2, 3) }')))
    '{ [-1] , [0] , [2] }'
    """
    return steps.floor(a)


def floordiv(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    `floor(div(a, b))` over the finite divisors; an infinite divisor gives the limit (module docstring)

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(floordiv(parse('[1, 2)'), parse('[1]')))
    '[1]'
    >>> format_cuts(floordiv(parse('[-5, 5]'), parse('[inf]')))  # floor(div) would give [0]
    '{ [-1] , [0] }'
    """
    if not a or not b:
        warn(EmptySetPropagationWarning, 'floordiv: an operand is empty, so the result is empty')
        return kernel.EMPTY
    # the quotient is taken exactly and only the integers are made float: a rounded quotient can
    # cross an integer, and the floor turns that ulp into a whole unit (`1 // 0.001` is 999, but the
    # float `1 / 0.001` is 1000.0)
    as_float = has_finite_float(a) or has_finite_float(b)
    a, b = exact_cuts(a), exact_cuts(b)
    parts = []
    finite_divisor = kernel.intersection(b, _FINITE)
    if finite_divisor:
        parts.append(floor(ops.div(a, finite_divisor)))
    finite_dividend = kernel.intersection(a, _FINITE)
    for y in (-INF, INF):
        if kernel.contains_point(b, y):
            parts.append(_floordiv_by_infinity(finite_dividend, y))
    if _isolated_infinities(a) and _isolated_infinities(b):
        # the box `[±inf] // [±inf]` has no value, as in div
        warn(IndeterminateResultWarning, 'floordiv(±inf, ±inf) has no value at any point (an indeterminate '
                                         'form), so that part contributes nothing')
    out = kernel.union(*parts) if parts else kernel.EMPTY
    return float_cuts(out, outward) if as_float else out


def _floordiv_by_infinity(finite_dividend: Cuts, y) -> Cuts:
    """the limit of floor(x / t) as t -> y = ±inf: 0 where x has y's sign or is 0, -1 where it has the other"""
    same = kernel.piece(0, INF, True, False) if y > 0 else kernel.piece(-INF, 0, False, True)
    other = kernel.piece(-INF, 0, False, False) if y > 0 else kernel.piece(0, INF, False, False)
    return kernel.normalize(kernel.piece(n, n) for side, n in ((same, 0), (other, -1))
                            if kernel.intersection(finite_dividend, side))


def _isolated_infinities(cuts: Cuts) -> bool:
    return any(lo == hi and is_infinite(lo) for lo, _, hi, _ in kernel.pieces(cuts))


def divmod_(a: Cuts, b: Cuts, outward: bool = False) -> Tuple[Cuts, Cuts]:
    """`(floordiv(a, b), mod(a, b))`, with one EmptySetPropagationWarning for an empty operand"""
    if not a or not b:
        warn(EmptySetPropagationWarning, 'divmod: an operand is empty, so the result is empty')
        return kernel.EMPTY, kernel.EMPTY
    return floordiv(a, b, outward), mod(a, b, outward)
