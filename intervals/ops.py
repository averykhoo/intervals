"""
arithmetic over cut tuples: the pointwise ops as descriptors, and one function per op

the result of an op is the set of values its *defined* pairs attain, with ±inf as ordinary points;
every endpoint, finite or infinite, is closed iff attained. pointwise:

* `x + y`, `x - y`: `inf - inf` (and `inf + -inf`) has no value; `inf + y` is `inf` otherwise
* `x * y`: `0 * ±inf` has no value; `±inf * y` is the signed infinity for `y != 0`
* `x / y`: `finite / ±inf` is 0, `±inf / ±inf` has no value, `±inf / finite` is signed infinity.
  `x / 0` for `x != 0` is `±inf` with the sign of x times the side of zero the divisor's piece lies
  on (both, if it crosses zero; nothing for a degenerate `[0]`), and `0 / 0` has no value
* `reciprocal(A)` is `div([1], A)`; `neg`, `pos`, `abs` are pointwise
* `power(A, n)` for an int n: `x ** n` for `n >= 1`, `[1]` for `n == 0` and `reciprocal(power(A, -n))`
  below that (evaluated as `1 / x ** -n` in one step, which is the same set on exact operands)
* `minimum(A, B)`, `maximum(A, B)`: the pointwise min and max. they are flat where the other operand
  is out of reach (`min(1, y)` is 1 for every y >= 1), so they decide attainment themselves
* `fma(A, B, C)`: `x * y + z`, computed exactly and rounded once when an operand is float

a box of the operands with no defined point at all (`[0] * [inf]`, `[inf] - [inf]`, `[0] / [0]`,
`1 / [0]`) contributes nothing and the call emits one `IndeterminateResultWarning`; an empty operand
gives `∅` and an `EmptySetPropagationWarning`.

int and Fraction are exact: `int / int` is a Fraction (an integral one becomes int in `Cut`), and
python's float leaks at infinity (`0 * inf`, `inf - inf`, `Fraction(1) / inf`) are evaluated
symbolically here, so an infinite operand never makes a result float. a float corner is rounded to
nearest, or with `outward=True` computed exactly and rounded down for a low end and up for a high
one (the `OUTWARD` descriptors), which gives the tightest float enclosure of the exact result.

>>> from intervals.fmt import format_cuts, parse
>>> format_cuts(reciprocal(parse('[-1, 0]')))
'[-inf, -1]'
>>> format_cuts(reciprocal(parse('(-1, 0)')))
'(-inf, -1)'
>>> format_cuts(mul(parse('[-inf, -1]'), parse('[0, 1]')))
'[-inf, 0]'
>>> format_cuts(sub(parse('[inf]'), parse('[1, inf]')))
'[inf]'
>>> format_cuts(mul(parse('[-1, 1]'), parse('[inf]')))
'{ [-inf] , [inf] }'
>>> format_cuts(div(parse('[1]'), parse('[3]')))
'[1/3]'
"""
import math
import operator
from fractions import Fraction
from functools import lru_cache
from numbers import Integral

from intervals import kernel
from intervals.applicator import OpDescriptor
from intervals.applicator import apply_binary
from intervals.applicator import apply_unary
from intervals.applicator import is_infinite
from intervals.applicator import sign
from intervals.applicator import signed_inf
from intervals.applicator import warn
from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.cuts import above
from intervals.cuts import below
from intervals.errors import EmptySetPropagationWarning
from intervals.kernel import Cuts
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import exact_cuts
from intervals.rounding import float_cuts
from intervals.rounding import has_finite_float
from intervals.rounding import is_float
from intervals.rounding import round_rational
from intervals import backend
from intervals import elementary


# POINTWISE (None where the op has no value)

def _add(x, y):
    if is_infinite(x):
        return None if is_infinite(y) and y != x else x
    if is_infinite(y):
        return y
    return _finite(operator.add, x, y)


def _sub(x, y):
    return _add(x, -y)


def _mul(x, y):
    if is_infinite(x) or is_infinite(y):
        s = sign(x) * sign(y)
        return signed_inf(s) if s else None
    return _finite(operator.mul, x, y)


def _div(x, y):
    if y == 0:
        return None
    if is_infinite(y):
        if is_infinite(x):
            return None
        return 0.0 if isinstance(x, float) else 0
    if is_infinite(x):
        return signed_inf(sign(x) * sign(y))
    if isinstance(x, float) or isinstance(y, float):
        return _finite(operator.truediv, x, y)
    return Fraction(x) / y


def _finite(op, x, y):
    """
    `op(x, y)` for finite x and y. python turns the exact one of a mixed pair into a float first,
    which overflows (`10**400 + 0.5`), makes a nonzero divisor 0.0 (`1.0 / Fraction(1, 10**400)`)
    or silently flushes it to zero (`Fraction(1, 10**400) * 1e300` would be 0.0), so a mixed pair
    is computed exactly and rounded once, to the signed infinity if it overflows
    """
    if isinstance(x, float) == isinstance(y, float):
        return op(x, y)
    exact = op(Fraction(x), Fraction(y))
    try:
        return float(exact)
    except OverflowError:
        return signed_inf(sign(exact))


def _div_pole(args, dirs):
    x, y = args
    if y == 0 and x != 0 and dirs[1]:
        return signed_inf(sign(x) * dirs[1])
    return None


def _reciprocal(x):
    return _div(1, x)


def _reciprocal_pole(args, dirs):
    return _div_pole((1, *args), (0, *dirs))


def _neg(x):
    return -x


def _pos(x):
    return x


def _abs(x):
    return abs(x)


def _holds(p, v) -> bool:
    lo, lo_closed, hi, hi_closed = p
    return lo < v < hi or (v == lo and lo_closed) or (v == hi and hi_closed)


def _reaches(p, v, upward: bool) -> bool:
    """does piece p hold a point >= v (upward) or <= v?"""
    lo, lo_closed, hi, hi_closed = p
    if upward:
        return hi > v or (hi == v and hi_closed)
    return lo < v or (lo == v and lo_closed)


def _min_attained(v, box) -> bool:
    """min(x, y) == v: one of them is v and the other is at least v"""
    a, b = box
    return (_holds(a, v) and _reaches(b, v, True)) or (_holds(b, v) and _reaches(a, v, True))


def _max_attained(v, box) -> bool:
    a, b = box
    return (_holds(a, v) and _reaches(b, v, False)) or (_holds(b, v) and _reaches(a, v, False))


ADD = OpDescriptor('add', _add, monotone=(1, 1))
SUB = OpDescriptor('sub', _sub, monotone=(1, -1))
MUL = OpDescriptor('mul', _mul, split_points=(0,))
DIV = OpDescriptor('div', _div, split_points=(0,), pole=_div_pole)
RECIPROCAL = OpDescriptor('reciprocal', _reciprocal, split_points=(0,), pole=_reciprocal_pole)
NEG = OpDescriptor('neg', _neg)
POS = OpDescriptor('pos', _pos)
ABS = OpDescriptor('abs', _abs, split_points=(0,))
MIN = OpDescriptor('min', min, monotone=(1, 1), attained=_min_attained)
MAX = OpDescriptor('max', max, monotone=(1, 1), attained=_max_attained)


_FAST_OPS = ((ADD, 'add'), (SUB, 'sub'), (MUL, 'mul'), (DIV, 'div'), (RECIPROCAL, 'reciprocal'))


def outward(desc: OpDescriptor) -> OpDescriptor:
    """
    the descriptor with a directed rounding hook: a float corner is evaluated exactly (each float as
    the Fraction it denotes) and rounded down for a result's low end, up for its high end. `fn` is
    exact too, so attainment is decided on exact values: an end that rounding moved is open (pown's
    outward descriptor is built by `_power_descriptor` instead: its `fn` is exact while the power is
    short enough to build, and past that a marker equal to nothing, `_NOT_A_DOUBLE`). the five
    arithmetic descriptors (keyed on the object, not its name) ask the backend for the double first
    (`intervals._gmpy2.outward`), read at each call; any other, and a None, keeps the pure rounding
    """
    fast_op = next((op for d, op in _FAST_OPS if d is desc), None)

    def exact(*args):
        # an infinite corner is exact already (the pointwise functions treat ±inf symbolically), and
        # evaluating it on the floats keeps a float operand's result float (`2.5 / inf` is 0.0)
        if any(is_infinite(x) for x in args):
            return desc.fn(*args)
        return desc.fn(*(Fraction(x) if is_float(x) else x for x in args))

    def rounding(direction):
        def rounded(*args):
            fast = backend.fast
            if fast is not None and fast_op is not None and (
                    answer := fast.outward(fast_op, args, direction)) is not None:
                return answer
            return round_rational(exact(*args), direction)
        return rounded
    return desc._replace(fn=exact, rounded=(rounding(DOWN), rounding(UP)))


# the descriptors that round (neg, pos, abs, min and max are exact on floats)
OUTWARD = {desc.name: outward(desc) for desc in (ADD, SUB, MUL, DIV, RECIPROCAL)}


def _pick(desc: OpDescriptor, outward_rounding: bool) -> OpDescriptor:
    return OUTWARD[desc.name] if outward_rounding else desc


class _NotADouble:
    """
    the exact value of `x ** n` for a finite float x whose power is too long to build
    (`elementary.exact_pow` declined: more than `EXACT_POWER_LIMIT` bits). it equals nothing, which is
    the truth: such a power is neither a double nor the midpoint of two, so no rounded end (and no
    ±inf) is attained by it, and no other corner of the box has the same value

    the proof, |x| = num/den in lowest terms, `|n| max(bitlen(num), bitlen(den)) > EXACT_POWER_LIMIT`
    (100000): x = ±1 never gets here (`exact_pow` answers 1 for every n). |x| = 2**e with e != 0 has
    max bitlen `|e| + 1 <= 2 |e|`, so `|e n| > 50000` and `2 ** (e n)` lies outside
    `[2**-1075, 2**1024]`, where there is no double and no midpoint. otherwise |x| = m 2**e with m odd
    >= 3 and max bitlen <= 1075 (a subnormal's denominator is 2**1074), so `|n| > 93`: for n > 0 the
    odd part of the power is `m ** n >= 3 ** 94 > 2 ** 54`, and for n < 0 its denominator has the odd
    factor `m ** |n|`, not dyadic at all, while a double's odd part is below 2**53 and a midpoint's
    below 2**54. so the rounding hooks' ziv loop meets no breakpoint at the value either, and ends
    (`elementary.rounded_pow`). on one side of zero, or for an odd n > 0, `x ** n` is injective on
    a box, so no other corner equals it
    """
    __slots__ = ()

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    __hash__ = object.__hash__

    def __repr__(self):
        return '<a power longer than EXACT_POWER_LIMIT bits>'


_NOT_A_DOUBLE = _NotADouble()


# python's `float ** int` converts the int to a double, which is exact up to here
_FLOAT_EXACT_EXPONENT = 2 ** 53


def _power_sign(x, n: int) -> int:
    """the sign of `x ** n` for x != 0: the int n's own parity, never a float's"""
    return -1 if x < 0 and n % 2 else 1


def _exact_power_descriptor(n: int) -> OpDescriptor:
    """
    `x ** n` for `n != 0`: monotone on each side of zero, so split there for even n and for n < 0.
    n < 0 is `1 / x ** -n` in one step: the same set as `reciprocal(power(A, -n))` (a piece of A
    at 0 and its image at 0 lie on the same side), but a float `x ** -n` that underflows to 0 keeps
    the sign of its pole instead of rounding to a zero with no side. exact on int and Fraction, python's
    `float ** int` (rounded to nearest) on a float
    """
    k = abs(n)

    def fn(x):
        if is_infinite(x):
            return math.inf if k % 2 == 0 else x
        try:
            return x ** k
        except OverflowError:  # float ** int raises where float * float gives inf
            return signed_inf(1 if x > 0 or k % 2 == 0 else -1)

    if n > 0:
        return OpDescriptor(f'pow{n}', fn, split_points=(0,) if n % 2 == 0 else ())

    def fn_negative(x):
        if x == 0:
            return None
        p = fn(x)
        if is_infinite(p):
            return 0.0 if isinstance(x, float) and not is_infinite(x) else 0
        if p == 0:  # a float underflow: the true value is a signed infinity after overflow
            return signed_inf(sign(x) ** k)
        return _div(1, p)

    def pole(args, dirs):
        return signed_inf(dirs[0] ** k) if args[0] == 0 and dirs[0] else None

    return OpDescriptor(f'pow{n}', fn_negative, split_points=(0,), pole=pole)


@lru_cache(maxsize=64)
def _power_descriptor(n: int, rounds_outward: bool = False) -> OpDescriptor:
    """
    `x ** n` for `n != 0` (`_exact_power_descriptor`), and its outward form. the outward form never
    builds a float corner's power past `elementary.EXACT_POWER_LIMIT` bits: `O(0.5) ** (2 ** 31 - 1)`
    and a base just above 1, which never saturates (`O(1.0000000000000002) ** 10 ** 9`), used to build
    it exactly and never finish (pown-huge). a finite float corner's exact power is built once, by
    `elementary.exact_pow`, where it is short, and rounded down and up from there, exactly as
    `outward` would; past that each end is `elementary.rounded_pow` (the range shortcuts or ziv over
    `exp(n ln |x|)`, 1788's pow route), the same double `round_rational` gives, and attainment sees
    `_NOT_A_DOUBLE`. int, Fraction and ±inf corners are the exact descriptor's

    to nearest, a float corner is python's `float ** int` while `|n| <= 2 ** 53`. past that python
    rounds n to a double (the parity lost: `(-1.0) ** (2 ** 60 + 1)` is 1.0) or cannot convert it
    (`0.5 ** 10 ** 400` raises OverflowError, which read as an overflow gave inf), so a float corner
    is `rounded_pow` to nearest with the int n's sign, and a zero base with n > 0 is 0.0
    """
    base = _exact_power_descriptor(n)
    if not rounds_outward:
        if abs(n) <= _FLOAT_EXACT_EXPONENT:
            return base

        def nearest(x):
            if not is_float(x):
                return base.fn(x)
            if x == 0:
                return 0.0 if n > 0 else base.fn(x)  # n < 0: None, and the pole decides
            s = _power_sign(x, n)
            return s * elementary.rounded_pow(abs(Fraction(x)), n, NEAREST) + 0.0
        return base._replace(fn=nearest)

    # a box asks for a corner's value up to four times (fn, both hooks, attainment): build it once.
    # a box has at most two corners, and a value is at most EXACT_POWER_LIMIT bits (12.5 KB). the
    # cache is this descriptor's, keyed on x: the value depends on n too
    @lru_cache(maxsize=4)
    def float_exact(x: float):
        if x == 0:
            return base.fn(Fraction(0))
        v = elementary.exact_pow(abs(Fraction(x)), n)
        return _NOT_A_DOUBLE if v is None else _power_sign(x, n) * v

    def exact(x):
        return float_exact(x) if is_float(x) else base.fn(x)

    def hook(direction):
        def rounded(x):  # a finite float corner with a value (x != 0 when n < 0)
            v = float_exact(x)
            if v is not _NOT_A_DOUBLE:
                return round_rational(v, direction)
            s = _power_sign(x, n)  # round(s v, d) is s round(v, s d): negating swaps DOWN and UP
            return s * elementary.rounded_pow(abs(Fraction(x)), n, direction * s) + 0.0
        return rounded
    return base._replace(fn=exact, rounded=(hook(DOWN), hook(UP)))


# OPS OVER CUT TUPLES

def neg(a: Cuts) -> Cuts:
    return apply_unary(NEG, a)


def pos(a: Cuts) -> Cuts:
    return apply_unary(POS, a)


def absolute(a: Cuts) -> Cuts:
    return apply_unary(ABS, a)


def reciprocal(a: Cuts, outward: bool = False) -> Cuts:
    return apply_unary(_pick(RECIPROCAL, outward), a)


def add(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    return apply_binary(_pick(ADD, outward), a, b)


def sub(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    return apply_binary(_pick(SUB, outward), a, b)


def mul(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    return apply_binary(_pick(MUL, outward), a, b)


def div(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    return apply_binary(_pick(DIV, outward), a, b)


def minimum(a: Cuts, b: Cuts) -> Cuts:
    """
    `{min(x, y) : x in a, y in b}`

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(minimum(parse('[3]'), parse('(1, 5)')))  # 3 = min(3, 4)
    '(1, 3]'
    """
    return apply_binary(MIN, a, b)


def maximum(a: Cuts, b: Cuts) -> Cuts:
    return apply_binary(MAX, a, b)


def fma(a: Cuts, b: Cuts, c: Cuts, outward: bool = False) -> Cuts:
    """
    `{x * y + z}`: `add(mul(a, b), c)` computed exactly, then rounded once if an operand is float

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(fma(parse('[0.1]'), parse('[10]'), parse('[-1]')))  # 0.1 is a hair above 1/10
    '[5.551115123125783e-17]'
    """
    if not a or not b or not c:
        warn(EmptySetPropagationWarning, 'fma: an operand is empty, so the result is empty')
        return kernel.EMPTY
    product = mul(exact_cuts(a), exact_cuts(b))
    if not product:  # an indeterminate product, which mul has warned about
        return kernel.EMPTY
    result = add(product, exact_cuts(c))
    if any(has_finite_float(x) for x in (a, b, c)):
        return float_cuts(result, outward)
    return result


def power(a: Cuts, n: int, outward: bool = False) -> Cuts:
    """
    `a ** n` for an int n (bool is refused)

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(power(parse('[-2, 1)'), 2))
    '[0, 4]'
    >>> format_cuts(power(parse('[-2, 1)'), -1))
    '{ [-inf, -1/2] , (1, inf] }'
    """
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError(f'the exponent must be an int, got {type(n).__name__}')
    n = int(n)
    if not a:
        warn(EmptySetPropagationWarning, 'pow: an operand is empty, so the result is empty')
        return kernel.EMPTY
    if n == 0:
        return below(1), above(1)
    return apply_unary(_power_descriptor(n, outward), a)


# CANCELLATION (ieee 1788's cancelMinus and cancelPlus, D13)

_REAL_LINE: Cuts = (above(-math.inf), below(math.inf))  # (-inf, inf): the finite points
_MINUS_INF: Cuts = (below(-math.inf), above(-math.inf))  # [-inf]
_PLUS_INF: Cuts = (below(math.inf), above(math.inf))  # [inf]


def cancel_minus(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    the Minkowski difference: the largest `X` with `b + X ⊆ a`, where `+` is `add` above

    **the definition.** `add` is the set of values of the defined pairs, so `b + X` is the union of
    the `{x} + b` over `x ∈ X`, and the largest `X` is exactly the set of the `x` that fit,
    `X = {x : {x} + b ⊆ a}`, which is `∩_{y ∈ b} (a - y)` over the defined pairs. an `X` like that
    exists for any two multi-intervals, so there is no "no answer" case: nothing fits gives `∅`, and
    an empty `b` (every `x` fits, `∅ ⊆ a`) gives the whole line `[-inf, inf]`. for connected,
    bounded, closed operands with `wid a ≥ wid b` it is ieee 1788's `[a1 - b1, a2 - b2]`; where
    1788 answers entire as "no answer" (`a` narrower than `b`, an unbounded operand), ours is a real
    set, often `∅`, and 1788's `cancelMinus [empty] [empty] = [empty]` is ours `[-inf, inf]`

    **the derivation**, finite `x` and the two infinite points apart, since `x + inf` is `inf` for
    every finite `x`, and `inf + -inf` has no value (so that pair drops out of `{x} + b`):

    * finite `x`: `{x} + b` is the real part of `b` shifted by `x`, plus `b`'s infinite points
      unchanged. so a finite `x` fits iff `inf ∈ b ⇒ inf ∈ a`, `-inf ∈ b ⇒ -inf ∈ a`, and
      `x + b_R ⊆ a_R`, the real parts `b_R = b ∩ (-inf, inf)`, `a_R` likewise
    * `x + b_R ⊆ a_R` holds iff it holds for every piece `q` of `b_R`, so the finite `x` are the
      intersection over the pieces `q`. `x + q` is connected, and the pieces `p` of the normalized
      `a_R` are its connected components (two pieces always have a missing point between them), so
      `x + q ⊆ a_R` iff `x + q ⊆ p` for one `p`: a union over the pieces `p`
    * one piece in one piece, `x + q ⊆ p`, as cuts: `start(p) <= start(q) + x` and
      `end(q) + x <= end(p)`, a cut shifted by `x` keeping its side. so `x >= p1 - q1`, where
      `x = p1 - q1` itself fits unless `p` is open there and `q` closed (a closed start of `q` would
      land on the missing point `p1`); likewise `x <= p2 - q2`, closed unless `p` is open there and
      `q` closed. an unbounded side of `q` needs the same side of `p` unbounded and then bounds
      nothing, as does an unbounded side of `p` against a bounded `q`. `p1 - q1 > p2 - q2`
      (`q` wider than `p`) or equal with an open side gives no `x`
    * `x = inf`: `{inf} + b` is `{inf}` unless `b` is `[-inf]` (nothing defined, so it fits), so
      `inf` fits iff `inf ∈ a` or `b = [-inf]`. `x = -inf` mirrors it

    **rounding.** the difference is computed exactly (each float as the Fraction it denotes) and, if
    an operand has a finite float end, rounded once like `fma`: to nearest, or with `outward=True`
    down at a low end and up at a high one, an end that moved open. outward is the tightest float
    enclosure of the exact `X`, which is ieee 1788's answer (`cancelMinus` of `[0x1.FFFFFFFFFFFFP+0]`
    and the double 0.1 is the two doubles around the difference, and of `[max]` and `[-max]` is
    `[max, inf]`) and the promise of `OutwardMultiInterval`: every `x` that fits is in it. it is not
    a certificate that `b + X ⊆ a`, which an outward end can break by an ulp; the exact classes are:
    int and Fraction operands (a float as `Fraction(f)`) give `X` exactly

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(cancel_minus(parse('[0, 10]'), parse('[1, 3]')))
    '[-1, 7]'
    >>> format_cuts(cancel_minus(parse('[0, 10)'), parse('[1, 3]')))  # 10 is missing: x + 3 < 10
    '[-1, 7)'
    >>> format_cuts(cancel_minus(parse('[0, 1]'), parse('[0, 2]')))  # nothing fits
    '{}'
    >>> format_cuts(cancel_minus(parse('[0, 1] | [10, 11]'), parse('[0] | [10]')))
    '[0, 1]'
    >>> format_cuts(cancel_minus(parse('(-inf, -1]'), parse('[-1, 5]')))
    '(-inf, -6]'
    >>> format_cuts(cancel_minus(parse('[3]'), parse('{}')))  # every x fits
    '[-inf, inf]'
    """
    x = _fitting(exact_cuts(a), exact_cuts(b))
    if has_finite_float(a) or has_finite_float(b):
        return float_cuts(x, outward)
    return x


def cancel_plus(a: Cuts, b: Cuts, outward: bool = False) -> Cuts:
    """
    `cancel_minus(a, neg(b))`: the largest `X` with `X - b ⊆ a`, ieee 1788's `cancelPlus`

    >>> from intervals.fmt import format_cuts, parse
    >>> format_cuts(cancel_plus(parse('[0, 10]'), parse('[1, 3]')))
    '[3, 11]'
    """
    return cancel_minus(a, neg(b) if b else b, outward)  # neg would warn of an empty operand


def _fitting(a: Cuts, b: Cuts) -> Cuts:
    """`{x : {x} + b ⊆ a}` for exact operands (see `cancel_minus` for the derivation)"""
    if not b:
        return kernel.REALS
    has = kernel.contains_point
    parts = []
    if (has(a, math.inf) or not has(b, math.inf)) and (has(a, -math.inf) or not has(b, -math.inf)):
        finite = _REAL_LINE
        a_real = kernel.intersection(a, _REAL_LINE)
        for q in kernel.pairs(kernel.intersection(b, _REAL_LINE)):
            finite = kernel.intersection(finite, kernel.normalize(
                fits for fits in (_piece_fits(p, q) for p in kernel.pairs(a_real)) if fits))
            if not finite:
                break
        parts.append(finite)
    if has(a, math.inf) or b == _MINUS_INF:
        parts.append(_PLUS_INF)
    if has(a, -math.inf) or b == _PLUS_INF:
        parts.append(_MINUS_INF)
    return kernel.union(*parts)


def _piece_fits(p, q):
    """
    the cut pair of the finite `x` with `x + q ⊆ p`, for pieces `p`, `q` of the finite points, or
    None when an unbounded side of `q` meets a bounded one of `p`. the pair may come out empty or
    reversed (`q` wider than `p`), which `kernel.normalize` drops
    """
    (p_start, p_end), (q_start, q_end) = p, q
    if q_start.value == -math.inf:  # q unbounded below: so must p be, and then nothing is bounded
        if p_start.value != -math.inf:
            return None
        start = p_start
    elif p_start.value == -math.inf:
        start = p_start
    else:  # x = p1 - q1 fits unless p is open there and q closed
        missed = p_start.side is Side.ABOVE and q_start.side is Side.BELOW
        start = Cut(p_start.value - q_start.value, Side.ABOVE if missed else Side.BELOW)
    if q_end.value == math.inf:
        if p_end.value != math.inf:
            return None
        end = p_end
    elif p_end.value == math.inf:
        end = p_end
    else:  # x = p2 - q2 fits unless p is open there and q closed
        missed = p_end.side is Side.BELOW and q_end.side is Side.ABOVE
        end = Cut(p_end.value - q_end.value, Side.BELOW if missed else Side.ABOVE)
    return start, end
