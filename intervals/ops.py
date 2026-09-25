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
from intervals.cuts import above
from intervals.cuts import below
from intervals.errors import EmptySetPropagationWarning
from intervals.kernel import Cuts
from intervals.rounding import DOWN
from intervals.rounding import UP
from intervals.rounding import exact_cuts
from intervals.rounding import float_cuts
from intervals.rounding import has_finite_float
from intervals.rounding import is_float
from intervals.rounding import round_rational


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


def outward(desc: OpDescriptor) -> OpDescriptor:
    """
    the descriptor with a directed rounding hook: a float corner is evaluated exactly (each float as
    the Fraction it denotes) and rounded down for a result's low end, up for its high end. `fn` is
    exact too, so attainment is decided on exact values: an end that rounding moved is open
    """
    def exact(*args):
        # an infinite corner is exact already (the pointwise functions treat ±inf symbolically), and
        # evaluating it on the floats keeps a float operand's result float (`2.5 / inf` is 0.0)
        if any(is_infinite(x) for x in args):
            return desc.fn(*args)
        return desc.fn(*(Fraction(x) if is_float(x) else x for x in args))

    def rounding(direction):
        def rounded(*args):
            return round_rational(exact(*args), direction)
        return rounded
    return desc._replace(fn=exact, rounded=(rounding(DOWN), rounding(UP)))


# the descriptors that round (neg, pos, abs, min and max are exact on floats)
OUTWARD = {desc.name: outward(desc) for desc in (ADD, SUB, MUL, DIV, RECIPROCAL)}


def _pick(desc: OpDescriptor, outward_rounding: bool) -> OpDescriptor:
    return OUTWARD[desc.name] if outward_rounding else desc


@lru_cache(maxsize=64)
def _power_descriptor(n: int, rounds_outward: bool = False) -> OpDescriptor:
    """
    `x ** n` for `n != 0`: monotone on each side of zero, so split there for even n and for n < 0.
    n < 0 is `1 / x ** -n` in one step: the same set as `reciprocal(power(A, -n))` (a piece of A
    at 0 and its image at 0 lie on the same side), but a float `x ** -n` that underflows to 0 keeps
    the sign of its pole instead of rounding to a zero with no side
    """
    k = abs(n)

    def fn(x):
        if is_infinite(x):
            return math.inf if k % 2 == 0 else x
        try:
            return x ** k
        except OverflowError:  # float ** int raises where float * float gives inf
            return signed_inf(1 if x > 0 or k % 2 == 0 else -1)

    if rounds_outward:
        return outward(_power_descriptor(n))
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
