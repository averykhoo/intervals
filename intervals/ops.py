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
  below that (evaluated as `1 / x ** -n` in one step, which is the same set)

a box of the operands with no defined point at all (`[0] * [inf]`, `[inf] - [inf]`, `[0] / [0]`,
`1 / [0]`) contributes nothing and the call emits one `IndeterminateResultWarning`; an empty operand
gives `∅` and an `EmptySetPropagationWarning`.

int and Fraction are exact: `int / int` is a Fraction (an integral one becomes int in `Cut`), and
python's float leaks at infinity (`0 * inf`, `inf - inf`, `Fraction(1) / inf`) are evaluated
symbolically here, so an infinite operand never makes a result float.

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


# POINTWISE (None where the op has no value)

def _add(x, y):
    if is_infinite(x):
        return None if is_infinite(y) and y != x else x
    if is_infinite(y):
        return y
    return x + y


def _sub(x, y):
    return _add(x, -y)


def _mul(x, y):
    if is_infinite(x) or is_infinite(y):
        s = sign(x) * sign(y)
        return signed_inf(s) if s else None
    return x * y


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
        return x / y
    return Fraction(x) / y


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


ADD = OpDescriptor('add', _add, monotone=(1, 1))
SUB = OpDescriptor('sub', _sub, monotone=(1, -1))
MUL = OpDescriptor('mul', _mul, split_points=(0,))
DIV = OpDescriptor('div', _div, split_points=(0,), pole=_div_pole)
RECIPROCAL = OpDescriptor('reciprocal', _reciprocal, split_points=(0,), pole=_reciprocal_pole)
NEG = OpDescriptor('neg', _neg)
POS = OpDescriptor('pos', _pos)
ABS = OpDescriptor('abs', _abs, split_points=(0,))


@lru_cache(maxsize=64)
def _power_descriptor(n: int) -> OpDescriptor:
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


def reciprocal(a: Cuts) -> Cuts:
    return apply_unary(RECIPROCAL, a)


def add(a: Cuts, b: Cuts) -> Cuts:
    return apply_binary(ADD, a, b)


def sub(a: Cuts, b: Cuts) -> Cuts:
    return apply_binary(SUB, a, b)


def mul(a: Cuts, b: Cuts) -> Cuts:
    return apply_binary(MUL, a, b)


def div(a: Cuts, b: Cuts) -> Cuts:
    return apply_binary(DIV, a, b)


def power(a: Cuts, n: int) -> Cuts:
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
    return apply_unary(_power_descriptor(n), a)
