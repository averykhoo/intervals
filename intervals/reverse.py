"""
reverse ops (ieee 1788's reverse-mode elementary functions; M13e, D12): the preimage of a set

`sqr_rev(c, x)`, `abs_rev(c, x)`, `pown_rev(c, n, x)` and `cosh_rev(c, x)` are each

    {t in x : f(t) has a value and f(t) in c}

for f the library's own function (`A ** 2`, `abs`, `A ** n`, `A.cosh()`) at a point, as an exact
multi-interval. `x` defaults to the whole of the affine extended reals, so ±inf are points like any
other, with f's value there: `inf ** 2` = inf, `inf ** -2` = 0 (so `pown_rev([0], -2)` is `[-inf] ∪
[inf]`), `cosh(-inf)` = inf. a point where f has no value (0 for a negative n, where `1/[0]` is
empty) is in no preimage. ieee 1788 answers the hull, `[-2, 2]` for `sqr_rev([1, 4])`; ours is the
union `[-2, -1] ∪ [1, 2]`, whose hull is 1788's answer (up to 1788 having no infinite points).

an end that is irrational (`sqrt 2`) is its tightest float enclosure, open, so an exact operand never
loses a true point; a float operand gives float ends, rounded to nearest in a `MultiInterval` (flags
kept) and outward in an `OutwardMultiInterval` (a moved end open), the rules of the functions
(`intervals.functions`). the result is an `OutwardMultiInterval` if either operand is one. a number
is taken as the point it is; an empty operand gives the empty set and an
`EmptySetPropagationWarning`, as the functions do. an empty result from non-empty operands is an
ordinary answer (no solution there), with no warning

**design, for the reverse ops still to come** (sin, cos, tan, mul, pow; plan §2 M13e):

* a **branch** is a piece of f's domain on which f is continuous and strictly monotone, described
  from the value side as `Branch(image, exact, rounded, increasing)`: `image` is the set of values f
  takes there (each end closed iff attained), and `exact`/`rounded` evaluate the inverse g on it.
  g maps the image onto the branch's piece of the domain preserving order (reversing it for a
  decreasing f), so the preimage of `c` in the branch is g applied piece by piece to `c ∩ image`,
  each end closed iff it was: `branch_preimage`. that is the whole algorithm: no case analysis per
  op beyond naming its branches
* g at a piece's end is exact where rational (`exact` returns int, Fraction or ±inf) and otherwise
  rounded down for a low end and up for a high end, open (`_end`, the rule of
  `functions._Function.end`); `named(name, image, ...)` makes a branch whose inverse is one of
  `intervals.elementary`'s correctly rounded functions (`sqrt`, `rootn`, `acosh` here; `asin`,
  `acos`, `atan` shifted by k pi for the periodic ones would be written the same way, with their
  own `exact` and `rounded`)
* an op is the union of its branches' preimages, **then** the intersection with `x`, after the
  rounding: an end of `x` inside an enclosure's slack is kept as it is, so the result never leaves `x`
  and is never looser than rounding then intersecting would make it
* the four ops here are even or odd functions of one monotone branch on `[0, inf]`: an even one's
  preimage is `P ∪ -P` and an odd one's `P(c) ∪ -P(-c)`, P being the branch's (`_even`, `_odd`)
* D12's cap (1000 pieces, then the hull and a `HullWarning`) is for the periodic ones, which have
  infinitely many branches; none of the four here has more than two pieces per piece of `c`

>>> from intervals import MultiInterval as M
>>> sqr_rev(M(1, 4))
MultiInterval.parse('{ [-2, -1] , [1, 2] }')
>>> sqr_rev(M(1, 4), M(0, 10))
MultiInterval.parse('[1, 2]')
>>> print(sqr_rev(M(2)))
{ (-1.4142135623730951, -1.414213562373095) , (1.414213562373095, 1.4142135623730951) }
>>> pown_rev(M(0), -2)
MultiInterval.parse('{ [-inf] , [inf] }')
"""
import math
from fractions import Fraction
from numbers import Real
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from intervals import elementary
from intervals import kernel
from intervals import ops
from intervals.applicator import warn
from intervals.cuts import Value
from intervals.cuts import mirror
from intervals.errors import EmptySetPropagationWarning
from intervals.kernel import Cuts
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import OutwardMultiInterval
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import is_float
from intervals.rounding import is_infinite
from intervals.rounding import round_rational

INF = math.inf

_REALS = MultiInterval(-INF, INF)  # the default x: the affine extended reals (intervals.REALS)


class Branch(NamedTuple):
    """
    a piece of f's domain where f is continuous and strictly monotone, from the value side: the values
    f takes there (`image`, each end closed iff attained), and its inverse g on them, as `exact(v)`
    (g(v) as an int, Fraction or ±inf where it is one, else None) and `rounded(v, direction)` (g(v)
    rounded DOWN, NEAREST or UP, for an exact v where `exact` gave None)
    """
    image: Cuts
    exact: Callable[[Value], Optional[Value]]
    rounded: Callable[[Value, int], float]
    increasing: bool


def named(name: str, image: Cuts, increasing: bool = True, base=None) -> Branch:
    """the branch whose inverse is `intervals.elementary`'s `name` (`base`: rootn's degree)"""
    return Branch(image, lambda v: elementary.exact(name, v, base),
                  lambda v, direction: elementary.rounded(name, v, direction, base), increasing)


def _one(lo, hi, lo_closed: bool = True, hi_closed: bool = True) -> Cuts:
    return kernel.normalize([kernel.piece(lo, hi, lo_closed, hi_closed)])


_NON_NEGATIVE = _one(0, INF)
_IDENTITY = Branch(_NON_NEGATIVE, lambda v: v, round_rational, True)


# THE ENGINE

def _end(branch: Branch, v, closed: bool, want: int, outward: bool) -> Tuple[Value, bool]:
    """
    g at one end v of a piece of values, as `(g(v), closed)`: `want` is DOWN for the preimage's low
    end and UP for its high end. an exact v gives g(v) exactly where rational, else its enclosure's
    end, open; a float v gives a float, to nearest (flags kept) or, outward, in the `want` direction,
    open where rounding moved it (`functions._Function.end`)
    """
    as_float = is_float(v)
    direction = (want if outward else NEAREST) if as_float else want
    exact_v = Fraction(v) if as_float else v
    value = branch.exact(exact_v)
    if value is None:  # irrational
        return branch.rounded(exact_v, direction), closed and direction == NEAREST
    if is_infinite(value) or not as_float:
        return value, closed
    rounded = round_rational(value, direction)
    return rounded, closed and (direction == NEAREST or rounded == value)


def branch_preimage(c: Cuts, branch: Branch, outward: bool) -> Cuts:
    """
    `{t in the branch : f(t) in c}`: g applied to each piece of `c ∩ image`, whose ends map to the
    preimage's (swapped for a decreasing f), each closed iff it was and not moved by rounding

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(branch_preimage(parse('[-1, 9/4]'), named('sqrt', _NON_NEGATIVE), outward=False))
    '[0, 3/2]'
    """
    out = []
    for lo, lo_closed, hi, hi_closed in kernel.pieces(kernel.intersection(c, branch.image)):
        if not branch.increasing:
            lo, lo_closed, hi, hi_closed = hi, hi_closed, lo, lo_closed
        a, a_closed = _end(branch, lo, lo_closed, DOWN, outward)
        b, b_closed = _end(branch, hi, hi_closed, UP, outward)
        if a == b:  # rounding to nearest squeezed the piece to one point: keep it, closed
            a_closed = b_closed = True
        out.append(kernel.piece(a, b, a_closed, b_closed))
    return kernel.normalize(out)


def negate(cuts: Cuts) -> Cuts:
    """`{-t : t in cuts}`, exact (a double's negation is a double), with no warning"""
    return tuple(mirror(cut) for cut in reversed(cuts))


def _even(c: Cuts, branch: Branch, outward: bool) -> Cuts:
    """f(-t) = f(t), with `branch` on [0, inf]: the branch's preimage and its mirror"""
    p = branch_preimage(c, branch, outward)
    return kernel.union(p, negate(p))


def _odd(c: Cuts, branch: Branch, outward: bool) -> Cuts:
    """f(-t) = -f(t), with `branch` on [0, inf]: t <= 0 has f(t) in c iff -t has f(-t) in -c"""
    return kernel.union(branch_preimage(c, branch, outward), negate(branch_preimage(negate(c), branch, outward)))


# THE OPS

def _operands(name: str, *operands):
    """the operands as MultiIntervals (a real number is a point), and the result's class"""
    out = []
    for a in operands:
        if isinstance(a, MultiInterval):
            out.append(a)
        elif isinstance(a, Real) and not isinstance(a, bool):
            out.append(MultiInterval(a))
        else:
            raise TypeError(f'{name}: expected a MultiInterval or a real number, got {type(a).__name__}')
    cls = OutwardMultiInterval if any(isinstance(a, OutwardMultiInterval) for a in out) else MultiInterval
    return out, cls


def _reverse(name: str, c, x, preimage: Callable[..., Cuts], given=()) -> MultiInterval:
    """`preimage(*given, c, outward) ∩ x` on cut tuples; `given` holds a binary op's other operand (mul_rev's b)"""
    (*given, c, x), cls = _operands(name, *given, c, x)
    if not c or not x or not all(given):
        warn(EmptySetPropagationWarning, f'{name}: an operand is empty, so the result is empty')
        return cls()
    return cls.from_cuts(kernel.intersection(preimage(*(g.cuts for g in given), c.cuts, cls._outward), x.cuts))


def sqr_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : t ** 2 in c}` (ieee 1788's sqrRev; with `x`, sqrRevBin)

    >>> from intervals import MultiInterval as M
    >>> sqr_rev(M(0.0, 25.0), M(-4.1, 6.0))
    MultiInterval.parse('[-4.1, 5.0]')
    >>> sqr_rev(M(-10, -1))
    MultiInterval.parse('{}')
    """
    return _reverse('sqr_rev', c, x, lambda cuts, outward: _even(cuts, _SQRT, outward))


_SQRT = named('sqrt', _NON_NEGATIVE)


def abs_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : abs(t) in c}` (ieee 1788's absRev; with `x`, absRevBin)

    >>> from intervals import MultiInterval as M
    >>> abs_rev(M.parse('[-1, 1) | (2, 3]'))
    MultiInterval.parse('{ [-3, -2) , (-1, 1) , (2, 3] }')
    """
    return _reverse('abs_rev', c, x, lambda cuts, outward: _even(cuts, _IDENTITY, outward))


def pown_rev(c, n: int, x=_REALS) -> MultiInterval:
    """
    `{t in x : t ** n in c}` for an int n (ieee 1788's pownRev; with `x`, pownRevBin). `t ** 0` is 1
    everywhere, ±inf included; for n < 0, 0 has no value (`1/[0]` is empty) and `(±inf) ** n` is 0

    >>> from intervals import MultiInterval as M
    >>> pown_rev(M(-8, 27), 3)
    MultiInterval.parse('[-2, 3]')
    >>> pown_rev(M(1, 4), -2)
    MultiInterval.parse('{ [-1, -1/2] , [1/2, 1] }')
    >>> pown_rev(M.parse('[0, inf)'), -1)
    MultiInterval.parse('{ [-inf] , (0, inf] }')
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f'pown_rev: the exponent must be an int, got {type(n).__name__}')
    if n == 0:
        return _reverse('pown_rev', c, x, lambda cuts, outward: kernel.REALS if kernel.contains_point(cuts, 1) else ())
    if n > 0:  # t ** n rises from 0 to inf on [0, inf]
        branch = named('rootn', _NON_NEGATIVE, base=n)
    else:  # falls from inf (not attained: 0 has no value) to 0 (at inf) on (0, inf]
        branch = named('rootn', _one(0, INF, True, False), increasing=False, base=n)
    parity = _even if n % 2 == 0 else _odd
    return _reverse('pown_rev', c, x, lambda cuts, outward: parity(cuts, branch, outward))


def cosh_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : cosh(t) in c}` (ieee 1788's coshRev; with `x`, coshRevBin)

    >>> from intervals import MultiInterval as M
    >>> cosh_rev(M(1, 2), M(0, 10))
    MultiInterval.parse('[0, 1.3169578969248168)')
    """
    return _reverse('cosh_rev', c, x, lambda cuts, outward: _even(cuts, _ACOSH, outward))


_ACOSH = named('acosh', _one(1, INF))


# MULTIPLICATION (M13e: ieee 1788's mulRev, mulRevTen and mulRevToPair)

def mul_rev(b, c, x=_REALS) -> MultiInterval:
    """
    `{t in x : t * y in c for some y in b}`, `*` being the library's (ieee 1788's mulRev; with `x`,
    mulRevTen; mulRevToPair's two intervals are the pieces of this one set)

    `*` is the set of the values of the defined pairs (`intervals.ops`), so `t` is in the result iff
    `{t} * b` meets `c`. `0 * ±inf` has no value and `±inf * y` is the signed infinity for `y != 0`,
    so, by the kind of `t` (the derivation, for `_mul_preimage`):

    * `t = 0`: `0 * y = 0` for every finite `y`, so 0 is in iff `0 ∈ c` and `b` has a finite point
    * finite `t != 0`: from a finite `y != 0`, `t = v / y` for a finite `v != 0` of `c` (`v = 0`
      gives `t = 0`); from `y = 0`, every such `t` if `0 ∈ b` and `0 ∈ c`; from `y = ±inf`, every
      `t` of the sign that makes `t * y` an infinity of `c`
    * `t = ±inf`: `t * y` is an infinity for every `y != 0` (none for `y = 0`), so `t` is in iff `c`
      holds the infinity it makes with a nonzero point of `b`

    ieee 1788 answers the hull (or two intervals, mulRevToPair) and has no infinite points: `0 * y =
    0` for every `y` there. every end is 0, ±inf or a quotient `v / w` of an end of `c` by one of `b`,
    computed by the library's own division (`intervals.ops.div`), so it is rounded exactly where the
    division `c / w` would round it: exact for int and Fraction ends, and where a float is involved
    to nearest in a `MultiInterval` (flags kept) or outward in an `OutwardMultiInterval` (a moved end
    open). a point `b = [w]`, `w` finite and nonzero, gives `c / w` in both classes. then `∩ x`

    >>> from intervals import MultiInterval as M
    >>> mul_rev(M(2, 4), M(1, 8))
    MultiInterval.parse('[1/4, 4]')
    >>> mul_rev(M(-1, 1), M(1, 2))  # 1788 answers entire (mulRev) or the two pieces (mulRevToPair)
    MultiInterval.parse('{ (-inf, -1] , [1, inf) }')
    >>> mul_rev(M(0), M(0))  # t * 0 = 0 for every finite t; inf * 0 has no value
    MultiInterval.parse('(-inf, inf)')
    >>> mul_rev(M.parse('[1, inf]'), M(3), M(0, 10))  # 0 * inf has no value, so 0 is not in
    MultiInterval.parse('(0, 3]')
    """
    return _reverse('mul_rev', c, x, _mul_preimage, given=(b,))


_REAL_LINE = _one(-INF, INF, False, False)  # (-inf, inf)
_NEGATIVE = _one(-INF, 0, False, False)
_POSITIVE = _one(0, INF, False, False)
_NONZERO = kernel.union(_NEGATIVE, _POSITIVE)  # the finite t != 0, and the finite y != 0
_SIDE = {1: _one(0, INF, False, True), -1: _one(-INF, 0, True, False)}  # the y != 0 of each sign, ±inf included
_FINITE_OF_SIGN = {1: _POSITIVE, -1: _NEGATIVE}


def _mul_preimage(b: Cuts, c: Cuts, outward: bool) -> Cuts:
    """`{t : t * y in c for some y in b}` for non-empty `b` and `c` (the cases: `mul_rev`); only the
    quotients round, in `ops.div`"""
    has = kernel.contains_point
    parts = []
    if has(c, 0) and kernel.intersection(b, _REAL_LINE):  # t = 0
        parts.append(_one(0, 0))
    b_nonzero, c_nonzero = kernel.intersection(b, _NONZERO), kernel.intersection(c, _NONZERO)
    if b_nonzero and c_nonzero:  # finite t != 0 and finite y != 0: t = v / y, never 0 nor ±inf
        parts.append(ops.div(c_nonzero, b_nonzero, outward))
    if has(b, 0) and has(c, 0):  # finite t != 0 and y = 0
        parts.append(_NONZERO)
    for c_sign in (1, -1):  # an infinity of c, made by t * y with y = ±inf or t = ±inf
        if not has(c, c_sign * INF):
            continue
        for y_sign in (1, -1):
            if has(b, y_sign * INF):  # every finite t != 0 of the sign that makes c_sign * inf
                parts.append(_FINITE_OF_SIGN[c_sign * y_sign])
            if kernel.intersection(b, _SIDE[y_sign]):  # t = ±inf with a nonzero y of this sign
                parts.append(_one(c_sign * y_sign * INF, c_sign * y_sign * INF))
    return kernel.union(*parts)
