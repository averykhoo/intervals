"""
reverse ops (ieee 1788's reverse-mode elementary functions; M13e, D12): the preimage of a set

`sqr_rev(c, x)`, `abs_rev(c, x)`, `pown_rev(c, n, x)` and `cosh_rev(c, x)` (and `sin_rev`, `cos_rev`,
`tan_rev`, `mul_rev`, `pow_rev1`, `pow_rev2`, below) are each

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

**design** (plan §2 M13e): the engine of the one-variable ops (these four and the periodic ones below;
`mul_rev`, `pow_rev1` and `pow_rev2` have two variables and take cases instead):

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
  `acos`, `atan` shifted by k pi for the periodic ones are written the same way, with their own
  `exact` and `rounded`: `_trig_branch`)
* an op is the union of its branches' preimages, **then** the intersection with `x`, after the
  rounding: an end of `x` inside an enclosure's slack is kept as it is, so the result never leaves `x`
  and is never looser than rounding then intersecting would make it
* the four ops here are even or odd functions of one monotone branch on `[0, inf]`: an even one's
  preimage is `P ∪ -P` and an odd one's `P(c) ∪ -P(-c)`, P being the branch's (`_even`, `_odd`)
* D12's cap (1000 pieces, then the hull and a `HullWarning`) is for the periodic ones, which have
  infinitely many branches; none of the four here has more than two pieces per piece of `c`

**the periodic ops** (`sin_rev`, `cos_rev`, `tan_rev`, M13e part 3) are built so: one branch per k
(`_Periodic`), whose inverse `k pi ± asin/acos/atan` is rounded by `elementary.rounded_inverse_trig`;
the branches meeting each piece of `x` (one more on each side) are listed, and D12's cap applies per
piece of `x` as `steps.step` applies it per piece of its operand (`_periodic_preimage`). sin, cos and
tan have no value at ±inf, nor tan at its poles, so neither is in any preimage

**the power's reverse ops** (`pow_rev1`, `pow_rev2`, M13e part 4) are two-variable, like `mul_rev`: a
case per special point of the library's pow (a base 0, 1 or inf; an exponent 0 or ±inf), and for the
rest the set of `v ** (1/w)` (the bases) or of `log_t v` (the exponents) over boxes of `c` and the other
operand, each box monotone in both, so its ends are two corners (`_power_box`'s rule; `_log_box`)

**decorated** (M13's merge of M13e and M13g): given a `DecoratedInterval` operand, each op is 1788's
decorated reverse op: the same set, computed on the operands' intervals, decorated trv, as 1788
decorates every reverse op's result (the decoration says nothing about a preimage). every interval
operand is then a `DecoratedInterval` or a real number; a bare `MultiInterval` is a `TypeError`, as
for the wrapper's other ops. mulRevToPair is the one op 1788 decorates better (its first interval as
the decorated division `c / b` where `0 ∉ b`); ours is one set, `mul_rev`'s, trv

>>> from intervals import DecoratedInterval as D, MultiInterval as M
>>> print(sqr_rev(D(M(1, 4))), mul_rev(D(M(2, 4)), D(M(1, 8)), D(M(0, 1))))
{ [-2, -1] , [1, 2] }_trv [1/4, 1]_trv

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
from intervals import functions
from intervals import kernel
from intervals import ops
from intervals.applicator import warn
from intervals.cuts import Value
from intervals.cuts import mirror
from intervals.decorated import DecoratedInterval
from intervals.decorated import _trivial
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.kernel import Cuts
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import OutwardMultiInterval
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import has_finite_float
from intervals.rounding import is_float
from intervals.rounding import is_infinite
from intervals.rounding import round_rational
from intervals.steps import ENUMERATION_CAP

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
    if any(isinstance(a, DecoratedInterval) for a in (*given, c, x)):
        return _decorated(name, c, x, preimage, given)
    (*given, c, x), cls = _operands(name, *given, c, x)
    if not c or not x or not all(given):
        warn(EmptySetPropagationWarning, f'{name}: an operand is empty, so the result is empty')
        return cls()
    return cls.from_cuts(kernel.intersection(preimage(*(g.cuts for g in given), c.cuts, cls._outward), x.cuts))


def _decorated(name: str, c, x, preimage: Callable[..., Cuts], given) -> DecoratedInterval:
    """the decorated reverse op (M13's merge of M13e and M13g): the op on the operands' intervals,
    decorated trv as 1788 decorates a reverse op's result (`decorated.py::_trivial`). an operand is a
    DecoratedInterval or a real number (a point); a bare MultiInterval is refused, as the wrapper's
    ops refuse it, but for the omitted `x` (the default, the affine extended reals)"""
    def interval(a):
        if isinstance(a, DecoratedInterval):
            return a.interval
        if isinstance(a, MultiInterval) and a is not _REALS:
            raise TypeError(f'{name}: expected a DecoratedInterval or a real number, got {type(a).__name__}')
        return a  # a number, or the default x; anything else is refused by `_operands`
    return _trivial(_reverse(name, interval(c), interval(x), preimage, tuple(map(interval, given))))


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


# PERIODIC FUNCTIONS (M13e: ieee 1788's sinRev, cosRev, tanRev and their *Bin forms; D12)

class _Periodic(NamedTuple):
    """
    a periodic f as its branches: the k-th holds the finite t with `floor(t / pi - offset) = k` (its
    ends aside), and `branch(k)` is it, with the inverse `m pi + sign * g(v)` for g `elementary`'s
    asin, acos or atan
    """
    name: str
    image: Cuts
    offset: Fraction
    branch: Callable[[int], Branch]
    gapless: bool  # whether a c holding the whole image has every finite t in its preimage (no poles)


def _trig_branch(g: str, image: Cuts, sign: int, m: int, increasing: bool) -> Branch:
    """the branch whose inverse is `m pi + sign * g(v)`: rational only at m = 0 where g(v) = 0"""
    return Branch(image, lambda v: 0 if m == 0 and elementary.exact(g, v) == 0 else None,
                  lambda v, direction: elementary.rounded_inverse_trig(g, v, sign, m, direction), increasing)


_UNIT = _one(-1, 1)
# sin on [k pi - pi/2, k pi + pi/2]: `k pi + (-1)**k asin v`, rising for an even k
_SIN = _Periodic('sin_rev', _UNIT, Fraction(-1, 2),
                 lambda k: _trig_branch('asin', _UNIT, 1 if k % 2 == 0 else -1, k, k % 2 == 0), True)
# cos on [k pi, (k + 1) pi]: `k pi + acos v`, falling, for an even k; `(k + 1) pi - acos v`, rising, for an odd k
_COS = _Periodic('cos_rev', _UNIT, Fraction(0),
                 lambda k: _trig_branch('acos', _UNIT, 1, k, False) if k % 2 == 0
                 else _trig_branch('acos', _UNIT, -1, k + 1, True), True)
# tan on (k pi - pi/2, k pi + pi/2): `k pi + atan v`, rising over all the reals; the poles between the
# branches have no value (as 0 for `1/x`), so they are in no preimage, and ±inf is not in the image
_TAN = _Periodic('tan_rev', _REAL_LINE, Fraction(-1, 2),
                 lambda k: _trig_branch('atan', _REAL_LINE, 1, k, True), False)

# over more branches than this, the preimage of a c that is not the whole image has more than
# ENUMERATION_CAP pieces: each period (two branches) holds a solution and a point that is not one
_BRANCH_LIMIT = 2 * ENUMERATION_CAP + 8


def _periodic_preimage(fn: _Periodic, x: Cuts, c: Cuts, outward: bool) -> Cuts:
    """
    `{t in x : f(t) in c}` as the union of the branches' preimages meeting each piece of x, D12's cap
    as `steps.step` has it: a piece of x whose part would take the count of pieces past
    `ENUMERATION_CAP`, or holds infinitely many (unbounded), gives its part's hull, and one
    `HullWarning` is emitted. f has no value at ±inf, so they are in no preimage
    """
    c = kernel.intersection(c, fn.image)
    x = kernel.intersection(x, _REAL_LINE)
    if not c or not x:
        return kernel.EMPTY
    if fn.gapless and kernel.is_subset(fn.image, c):  # every finite t: one piece, however wide x is
        return x
    out, count, hulled = [], 0, False
    for lo, lo_closed, hi, hi_closed in kernel.pieces(x):
        part = _one(lo, hi, lo_closed, hi_closed)
        # one branch more on each side: x may start inside the one-double slack of the enclosure of the
        # branch before it (just past a pole or an extremum), which the union of every branch holds
        first = None if lo == -INF else elementary.floor_over_pi(lo, fn.offset)[0] - 1
        last = None if hi == INF else elementary.floor_over_pi(hi, fn.offset)[0] + 1
        if first is not None and last is not None and last - first < _BRANCH_LIMIT:
            p = kernel.intersection(
                kernel.union(*(branch_preimage(c, fn.branch(k), outward) for k in range(first, last + 1))), part)
            if count + len(p) // 2 <= ENUMERATION_CAP:
                count += len(p) // 2
                out.append(p)
                continue
            out.append(kernel.hull(p))
        else:
            out.append(_periodic_hull(fn, c, part, first, last, outward))
        hulled = True
    if hulled:
        warn(HullWarning, f'{fn.name}: more than {ENUMERATION_CAP} pieces, or infinitely many, so their hull '
                          f'was returned')
    return kernel.union(*out)


def _periodic_hull(fn: _Periodic, c: Cuts, part: Cuts, first: Optional[int], last: Optional[int],
                   outward: bool) -> Cuts:
    """
    the hull of the preimage in one piece of x spanning more than `_BRANCH_LIMIT` branches or unbounded
    (`first`/`last` None): every branch holds a solution, so the preimage is unbounded where the piece
    is (never reaching ±inf, open there), and elsewhere its end is in one of the first branches met
    walking inward from the piece's end
    """
    def walk(k: int, step: int) -> Cuts:
        while True:
            found = kernel.intersection(branch_preimage(c, fn.branch(k), outward), part)
            if found:
                return found
            k += step

    lo, lo_closed = (-INF, False) if first is None else next(kernel.pieces(walk(first, 1)))[:2]
    hi, hi_closed = (INF, False) if last is None else next(kernel.pieces(walk(last, -1)[-2:]))[2:]
    return _one(lo, hi, lo_closed, hi_closed)


def sin_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : sin(t) in c}` (ieee 1788's sinRev; with `x`, sinRevBin), sin having no value at ±inf

    the exact pieces over a bounded `x`, where 1788 answers their hull: every end is `k pi ± asin v`,
    irrational but where it is 0, so its tightest float enclosure, open (a float `c` rounds as the
    other reverse ops do). past `steps.ENUMERATION_CAP` pieces, or infinitely many (an unbounded `x`,
    the default among them), their hull and a `HullWarning` (D12). a `c` holding [-1, 1] needs no
    hull: every finite t is a solution

    >>> from intervals import MultiInterval as M
    >>> len(sin_rev(M(0.5, 1), M(0, 20)).pieces)  # D12's example
    4
    >>> sin_rev(M(0), M(-1, 4))
    MultiInterval.parse('{ [0] , (3.141592653589793, 3.1415926535897936) }')
    >>> sin_rev(M(-1, 1))
    MultiInterval.parse('(-inf, inf)')
    >>> sin_rev(M(2))
    MultiInterval.parse('{}')
    """
    return _reverse('sin_rev', c, x, _periodic(_SIN), given=(x,))


def cos_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : cos(t) in c}` (ieee 1788's cosRev; with `x`, cosRevBin): as `sin_rev`, the ends
    `2k pi ± acos v`

    >>> from intervals import MultiInterval as M
    >>> cos_rev(M(1), M(-1, 7))
    MultiInterval.parse('{ [0] , (6.283185307179586, 6.283185307179587) }')
    """
    return _reverse('cos_rev', c, x, _periodic(_COS), given=(x,))


def tan_rev(c, x=_REALS) -> MultiInterval:
    """
    `{t in x : tan(t) in c}` (ieee 1788's tanRev; with `x`, tanRevBin): as `sin_rev`, the ends
    `k pi + atan v`. the poles `pi/2 + k pi` have no value, so they are in no preimage (as 0 is in none
    of `pown_rev(c, -1)`), even for a `c` holding ±inf, which is therefore no solution at all; since
    every branch has one, any `c` with a finite point over an unbounded `x` gives a hull

    >>> from intervals import MultiInterval as M
    >>> tan_rev(M(1), M(0, 4))
    MultiInterval.parse('{ (0.7853981633974483, 0.7853981633974484) , (3.9269908169872414, 3.926990816987242) }')
    >>> tan_rev(M.parse('[inf]'))
    MultiInterval.parse('{}')
    """
    return _reverse('tan_rev', c, x, _periodic(_TAN), given=(x,))


def _periodic(fn: _Periodic) -> Callable[[Cuts, Cuts, bool], Cuts]:
    """the preimage function `_reverse` calls: x comes first, as the periodic ops' `given`, since which
    branches to list depends on it"""
    return lambda x, c, outward: _periodic_preimage(fn, x, c, outward)


# POWER (M13e: ieee 1788's powRev1 and powRev2, the reverse ops of pow, D11)
#
# the library's pow (`functions.pow_`) at a point: `0 ** y` = 0 for y in (0, inf], and nothing for
# y <= 0; `1 ** y` = 1 for a finite y, nothing at ±inf; `inf ** y` = inf for y in (0, inf], 0 for y in
# [-inf, 0), nothing at 0; `x ** 0` = 1 for a finite x > 0; for x in (0, 1) ∪ (1, inf), `x ** inf` is 0
# below 1 and inf above, `x ** -inf` the reverse; a negative base has no value. every other `x ** y`
# (x in (0, 1) ∪ (1, inf) finite, y finite and not 0) is finite, positive and not 1

_OPEN_UNIT = _one(0, 1, False, False)  # (0, 1)
_ABOVE_ONE = _one(1, INF, False, False)  # (1, inf)
_BASES = kernel.union(_OPEN_UNIT, _ABOVE_ONE)  # the finite x > 0 but 1, and the values x ** y there
_BASE_PARTS = ((-1, _OPEN_UNIT), (1, _ABOVE_ONE))  # by the side of 1 (ln's sign)
_EXPONENT_PARTS = ((-1, _NEGATIVE), (1, _POSITIVE))  # the finite y != 0, by sign
_UP_TO_INF = _one(0, INF, False, True)  # (0, inf]: the y with `0 ** y` = 0 and `inf ** y` = inf
_FROM_MINUS_INF = _one(-INF, 0, True, False)  # [-inf, 0): the y with `inf ** y` = 0
_BELOW_ONE_FROM_0 = _one(0, 1, True, False)  # [0, 1): the x with `x ** inf` = 0
_ABOVE_ONE_TO_INF = _one(1, INF, False, True)  # (1, inf]: the x with `x ** inf` = inf and `x ** -inf` = 0


def pow_rev1(b, c, x=_REALS) -> MultiInterval:
    """
    `{t in x : t ** y in c for some y in b}`, `**` being the library's pow (`functions.pow_`, D11:
    ieee 1788's pow, with ±inf as points where it has a limit); ieee 1788's powRev1

    `t` is in the result iff `{t} ** b` meets `c`. by the kind of `t` (the derivation, for `_pow1_preimage`):

    * `t = 0`: iff `0 ∈ c` and `b` meets `(0, inf]`; `t = 1`: iff `1 ∈ c` and `b` has a finite point;
      `t = inf`: iff `inf ∈ c` and `b` meets `(0, inf]`, or `0 ∈ c` and `b` meets `[-inf, 0)`
    * a finite `t > 0` other than 1: from `y = 0`, all of them if `0 ∈ b` and `1 ∈ c`; from `y = inf`,
      those below 1 if `0 ∈ c` and those above 1 if `inf ∈ c` (`y = -inf` the other way round); from a
      finite `y != 0`, `t = v ** (1/y)` for `v` in `c ∩ ((0, 1) ∪ (1, inf))`: the library's pow of those
      `v` by those `1/y`, box by box (`functions._power_box`), so an end is exact where rational and
      rounded where not exactly as `**` rounds `v ** (1/y)` (to nearest in a `MultiInterval` with a
      float operand, flags kept; outward in an `OutwardMultiInterval`, a moved end open; for exact
      operands, an irrational end is its tightest float enclosure, open)

    1788 answers the hull and has no infinite points. then `∩ x`

    >>> from intervals import MultiInterval as M
    >>> pow_rev1(M(2), M(4, 9))
    MultiInterval.parse('[2, 3]')
    >>> pow_rev1(M(-1, 1), M(2))  # t = 2 ** (1/y): [2, inf) for y in (0, 1], (0, 1/2] for y in [-1, 0)
    MultiInterval.parse('{ (0, 1/2] , [2, inf) }')
    >>> pow_rev1(M(0), M(1))  # t ** 0 = 1 for a finite t > 0; 0 ** 0 and inf ** 0 have no value
    MultiInterval.parse('(0, inf)')
    >>> pow_rev1(M(-2), M(0, 1))  # inf ** -2 = 0; 0 ** -2 has no value
    MultiInterval.parse('[1, inf]')
    >>> print(pow_rev1(M(2), M(2)))
    (1.414213562373095, 1.4142135623730951)
    """
    return _reverse('pow_rev1', c, x, _pow1_preimage, given=(b,))


def _pow1_preimage(b: Cuts, c: Cuts, outward: bool) -> Cuts:
    """`{t : t ** y in c for some y in b}` for non-empty `b` and `c` (the cases: `pow_rev1`)"""
    has, meets = kernel.contains_point, _meets
    parts = []
    if has(c, 0) and meets(b, _UP_TO_INF):  # 0 ** y = 0 for y > 0
        parts.append(_one(0, 0))
    if has(c, 1) and meets(b, _REAL_LINE):  # 1 ** y = 1 for a finite y
        parts.append(_one(1, 1))
    if (has(c, INF) and meets(b, _UP_TO_INF)) or (has(c, 0) and meets(b, _FROM_MINUS_INF)):  # inf ** y
        parts.append(_one(INF, INF))
    if has(b, 0) and has(c, 1):  # t ** 0 = 1
        parts.append(_BASES)
    for y, small, large in ((INF, 0, INF), (-INF, INF, 0)):  # t ** ±inf: by the side of 1, 0 or inf
        if has(b, y):
            if has(c, small):
                parts.append(_OPEN_UNIT)
            if has(c, large):
                parts.append(_ABOVE_ONE)
    as_float = has_finite_float(b) or has_finite_float(c)
    for sv, v_part in _BASE_PARTS:  # finite t and y != 0: t = v ** (1/y), v = t ** y in (0, 1) ∪ (1, inf)
        for sw, w_part in _EXPONENT_PARTS:
            for v in kernel.pieces(kernel.intersection(c, v_part)):
                for w in kernel.pieces(kernel.intersection(b, w_part)):
                    lo, lo_closed, hi, hi_closed = functions._power_box(v, _reciprocal(w, sw), sv, sw, as_float, outward)
                    parts.append(_one(lo, hi, lo_closed, hi_closed))
    return kernel.union(*parts)


def _reciprocal(w, sign: int):
    """`{1/y : y in w}` for a piece w of one sign, exactly: its ends swapped, 1/0 the signed infinity"""
    lo, lo_closed, hi, hi_closed = w

    def inverse(y):
        return sign * INF if y == 0 else 0 if is_infinite(y) else 1 / Fraction(y)
    return inverse(hi), hi_closed, inverse(lo), lo_closed


def _meets(cuts: Cuts, part: Cuts) -> bool:
    return bool(kernel.intersection(cuts, part))


def pow_rev2(a, c, y=_REALS) -> MultiInterval:
    """
    `{s in y : t ** s in c for some t in a}`, `**` being the library's pow (`functions.pow_`, D11);
    ieee 1788's powRev2

    `s` is in the result iff `a ** {s}` meets `c`. by the kind of `s` (the derivation, for `_pow2_preimage`):

    * `s = 0`: iff `1 ∈ c` and `a` meets `(0, inf)`; `s = inf`: iff `0 ∈ c` and `a` meets `[0, 1)`, or
      `inf ∈ c` and `a` meets `(1, inf]`; `s = -inf`: iff `inf ∈ c` and `a` meets `(0, 1)`, or `0 ∈ c` and
      `a` meets `(1, inf]`
    * a finite `s != 0`: from `t = 0`, every `s > 0` if `0 ∈ a` and `0 ∈ c`; from `t = 1`, every `s` if
      `1 ∈ a` and `1 ∈ c`; from `t = inf`, every `s > 0` if `inf ∈ c` and every `s < 0` if `0 ∈ c`;
      from a finite `t` in `(0, 1) ∪ (1, inf)`, `s = log_t v` for `v` in `c ∩ ((0, 1) ∪ (1, inf))`,
      box by box (`_log_box`), each end `elementary`'s correctly rounded `log(v, t)`: exact where
      rational (`log_4 2` = 1/2), else rounded as `pow_rev1` rounds

    1788 answers the hull and has no infinite points. then `∩ y`

    >>> from intervals import MultiInterval as M
    >>> pow_rev2(M(2), M(4, 8))
    MultiInterval.parse('[2, 3]')
    >>> pow_rev2(M(1, 4), M(2))  # log_t 2 for t in (1, 4]: [1/2, inf); 1 ** s is never 2
    MultiInterval.parse('[1/2, inf)')
    >>> pow_rev2(M(0), M(0))  # 0 ** s = 0 for s in (0, inf]
    MultiInterval.parse('(0, inf]')
    >>> print(pow_rev2(M(3), M(2)))
    (0.6309297535714574, 0.6309297535714575)
    """
    return _reverse('pow_rev2', c, y, _pow2_preimage, given=(a,))


def _pow2_preimage(a: Cuts, c: Cuts, outward: bool) -> Cuts:
    """`{s : t ** s in c for some t in a}` for non-empty `a` and `c` (the cases: `pow_rev2`)"""
    has, meets = kernel.contains_point, _meets
    parts = []
    if has(c, 1) and meets(a, _POSITIVE):  # t ** 0 = 1 for a finite t > 0
        parts.append(_one(0, 0))
    if (has(c, 0) and meets(a, _BELOW_ONE_FROM_0)) or (has(c, INF) and meets(a, _ABOVE_ONE_TO_INF)):  # t ** inf
        parts.append(_one(INF, INF))
    if (has(c, INF) and meets(a, _OPEN_UNIT)) or (has(c, 0) and meets(a, _ABOVE_ONE_TO_INF)):  # t ** -inf
        parts.append(_one(-INF, -INF))
    if has(a, 0) and has(c, 0):  # 0 ** s = 0 for s > 0
        parts.append(_POSITIVE)
    if has(a, 1) and has(c, 1):  # 1 ** s = 1
        parts.append(_NONZERO)
    if has(a, INF):  # inf ** s: inf for s > 0, 0 for s < 0
        if has(c, INF):
            parts.append(_POSITIVE)
        if has(c, 0):
            parts.append(_NEGATIVE)
    as_float = has_finite_float(a) or has_finite_float(c)
    for st, t_part in _BASE_PARTS:  # finite t and s != 0: s = log_t v, v = t ** s in (0, 1) ∪ (1, inf)
        for sv, v_part in _BASE_PARTS:
            for t in kernel.pieces(kernel.intersection(a, t_part)):
                for v in kernel.pieces(kernel.intersection(c, v_part)):
                    lo, lo_closed, hi, hi_closed = _log_box(v, t, sv, st, as_float, outward)
                    parts.append(_one(lo, hi, lo_closed, hi_closed))
    return kernel.union(*parts)


def _log_box(v, t, sv: int, st: int, as_float: bool, outward: bool):
    """
    `{log_r u : u in v, r in t}` as a piece, for pieces v and t on sides `sv` and `st` of 1: `log_r u =
    ln u / ln r` rises with u iff `st > 0` and with r iff `sv < 0`, so the least value is at v's low end
    iff `st > 0` and t's low end iff `sv < 0`, the greatest at the other two
    """
    ends = []
    for want, other in ((DOWN, False), (UP, True)):
        v_hi, t_hi = (st < 0) != other, (sv > 0) != other
        u, u_closed = (v[2], v[3]) if v_hi else (v[0], v[1])
        r, r_closed = (t[2], t[3]) if t_hi else (t[0], t[1])
        ends.append(_log_corner(u, r, sv, st, want, u_closed and r_closed, as_float, outward))
    (lo, lo_closed), (hi, hi_closed) = ends
    if lo == hi:  # rounding to nearest squeezed the piece to one point: keep it, closed
        return lo, True, hi, True
    return lo, lo_closed, hi, hi_closed


def _log_corner(u, r, sv: int, st: int, want: int, attained: bool, as_float: bool, outward: bool):
    """
    `(log_r u, closed)` at a corner, the limit where the corner is an open end at 0, 1 or inf (an extreme
    corner never pairs 1 with 1, nor 0 or inf with 0 or inf), rounded as `functions._power_corner` rounds
    a power
    """
    direction = (want if outward else NEAREST) if as_float else want
    if u == 1:
        value = 0
    elif r == 1:  # ln r -> 0 from st's side
        value = sv * st * INF
    elif u == 0:
        value = -st * INF
    elif u == INF:
        value = st * INF
    elif r == 0 or r == INF:
        value = 0
    else:
        value = elementary.exact('log', Fraction(u), Fraction(r))
        if value is None:  # irrational
            return elementary.rounded('log', Fraction(u), direction, Fraction(r)), attained and direction == NEAREST
    if is_infinite(value) or not as_float:
        return value, attained
    rounded = round_rational(value, direction)
    return rounded, attained and (direction == NEAREST or rounded == value)
