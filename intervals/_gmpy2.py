"""
the gmpy2/mpfr backend (`INTERVALS_BACKEND=gmpy2` or `auto`, see `intervals.backend`): which double

each function here answers one question the pure path answers in `intervals.elementary` and in the
outward hook (`intervals.ops.outward`), "which double is f(x) rounded down, to nearest or up", with
a float, or None for "no one-call answer here", and then the pure path runs as before. it is called
only after every decision that is not a rounding: `exact` (is the value rational, and which),
`exact_pow`, `_beyond` and the other shortcuts past the float range, and every flag and attainment
in `functions`, `reverse` and the applicator. so it cannot change a flag, an exactness or an
attainment, only how fast a double is found, and the answer is the pure path's double, sign bit
included (`tests/test_backend.py` compares the two at every point it draws).

why one MPFR call is the correctly rounded double: MPFR's functions are correctly rounded in every
direction, and a context `gmpy2.ieee(64)` is binary64 exactly (53 bits, the subnormals applied with
the ternary value, so no double rounding). so the input must be exact: an MPFR function is called
only on an mpfr equal to x, which exists when x is dyadic (every float, every int, `Fraction(3, 8)`),
built at x's own bit length in a private wide context. a non-dyadic rational (`1/3`) is declined,
but where the function is atan2 of two ints (atan, acot, the angles of atan2) or where the value is
a rational (the hook's mixed operands: rounded from an exact mpq). the rules the probes found
(`v2-implementation-plan.md` M16e):

* every mpfr built here names a context: a bare `mpfr(x)` reads gmpy2's global context, which is
  the user's (at 10 bits it rounds the input; with traps on it raises)
* no `-0.0`: MPFR gives one for a negative value rounded to 0, and `x - x` rounded down; the pure
  path never does, so `+ 0.0` is every function's last operation (after a sign, not before it)
* an operand past `BOUND` bits (numerator or denominator) is declined: MPFR's exponent range is
  `2**30` on windows, past which a dyadic flushes to 0 with a ternary value of 0 (to inf with 1),
  silently
* `rootn` only for `0 < n < 2**31`: gmpy2 takes n as a C `unsigned long` (32 bits on windows, where
  it raises `OverflowError` from `2**32`); `2**31` is a margin that holds on every platform
* a ternary value of 0 on a function's result means the value is a double, so rational, which
  `exact` said it is not: an exact case was missed, and this raises as the pure loop does (MPFR
  would return the double, and the caller would mark the end open: a point lost). a nan (an argument
  outside the domain) raises too, never a nan end

the three contexts are module objects whose flags MPFR writes after each call; nothing reads them,
so this is safe under the GIL. untested on free-threaded builds.
"""
import math
from fractions import Fraction
from typing import Optional
from typing import Tuple

import gmpy2
from gmpy2 import mpfr
from gmpy2 import mpq

from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP

BOUND = 1 << 20  # the most bits of an operand's numerator or denominator taken (class 15)
ROOTN_LIMIT = 1 << 31  # rootn's n below this: a C unsigned long is 32 bits on windows, 64 on linux
_MISSED = 'the enclosure never narrowed to one double: an exact case was missed'

# the elementary functions that are a context method of the same name
NATIVE = frozenset((
    'sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'expm1', 'log1p', 'cbrt', 'cot', 'sec', 'csc',
    'coth', 'csch', 'sech'))


def _context(rounding) -> gmpy2.context:
    context = gmpy2.ieee(64)
    context.round = rounding
    return context


_CONTEXTS = {DOWN: _context(gmpy2.RoundDown), NEAREST: _context(gmpy2.RoundToNearest), UP: _context(gmpy2.RoundUp)}
_WIDE = gmpy2.context()  # a fresh default (53 bits, MPFR's widest exponent range), never the global one
_OPERATIONS = ('add', 'sub', 'mul', 'div')


# THE INPUT: EXACT, OR DECLINED

def _ratio(x) -> Optional[Tuple[int, int]]:
    """`(numerator, denominator)` of a finite float, an int or a Fraction within `BOUND`, else None"""
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        n, d = x.as_integer_ratio()
    elif isinstance(x, int):
        n, d = int(x), 1
    elif isinstance(x, Fraction):
        n, d = x.numerator, x.denominator
    else:
        return None
    if max(abs(n).bit_length(), d.bit_length()) > BOUND:
        return None
    return n, d


def _int(n: int) -> mpfr:
    """an int as the mpfr equal to it"""
    return mpfr(n, max(2, abs(n).bit_length()), _WIDE)


def _operand(x) -> Optional[mpfr]:
    """the mpfr equal to x, or None where there is none (x not dyadic) or x is declined"""
    if isinstance(x, float) and math.isfinite(x):
        return mpfr(x, 53, _WIDE)
    ratio = _ratio(x)
    if ratio is None:
        return None
    n, d = ratio
    if d & (d - 1):
        return None
    if d == 1:
        return _int(n)
    return mpfr(mpq(n, d), max(2, abs(n).bit_length()), _WIDE)


def _value(r: mpfr, missed_if_exact: bool = True) -> float:
    """the double r is (a float, maybe -0.0), after the two guards"""
    if r.is_nan():
        raise ArithmeticError('MPFR answered not a number: an argument outside the domain reached the backend')
    if missed_if_exact and r.rc == 0:
        raise ArithmeticError(_MISSED)
    return float(r)


# THE ELEMENTARY FUNCTIONS

def _native(name: str, x, direction: int, base=None) -> Optional[mpfr]:
    context = _CONTEXTS[direction]
    if name in ('atan', 'acot'):
        # atan(n/d) = atan2(n, d) and acot(n/d) = pi/2 - atan(n/d) = atan2(d, n), for d > 0: two exact
        # ints, so any rational x, dyadic or not
        ratio = _ratio(x)
        if ratio is None:
            return None
        n, d = ratio
        return context.atan2(_int(n), _int(d)) if name == 'atan' else context.atan2(_int(d), _int(n))
    if name == 'rootn':
        if not 0 < base < ROOTN_LIMIT:  # n < 0 is a reciprocal: two roundings
            return None
        m = _operand(x)
        return None if m is None else context.rootn(m, base)
    if name not in NATIVE or (name == 'log' and base is not None):  # log to a base, acoth: quotients
        return None
    m = _operand(x)
    return None if m is None else getattr(context, name)(m)


def rounded(name: str, x, direction: int, base=None) -> Optional[float]:
    """`elementary.rounded`'s double, for an x where `exact` is None and `_beyond` is None; or None"""
    r = _native(name, x, direction, base)
    return None if r is None else _value(r) + 0.0


def rounded_pow(x, y, direction: int) -> Optional[float]:
    """`elementary.rounded_pow`'s double for dyadic x > 0 and y, past its shortcuts; or None"""
    mx, my = _operand(x), _operand(y)
    if mx is None or my is None:
        return None
    return _value(_CONTEXTS[direction].pow(mx, my)) + 0.0


def rounded_angle(q, m: int, direction: int) -> Optional[float]:
    """
    `elementary.rounded_angle`'s double, `atan(q) + m pi/2`, for the (q, m) of `functions._angle`: with
    q = n/d, d > 0, it is atan2 of two ints by quadrant; at q = 0 a multiple of pi/2, pi rounded
    in the direction (reversed for m < 0) and scaled by a power of 2, exactly. None for any other m
    """
    ratio = _ratio(q)
    if ratio is None:
        return None
    n, d = ratio
    if n == 0:
        if m not in (1, -1, 2, -2):
            return None
        pi = _value(_CONTEXTS[direction if m > 0 else -direction].const_pi())
        return m * pi / 2 + 0.0
    if m == 0:
        y, x = n, d
    elif m == 1:
        y, x = d, -n
    elif m == -1:
        y, x = -d, n
    elif (m == 2 and n < 0) or (m == -2 and n > 0):  # the side that stays inside [-pi, pi]
        y, x = -n, -d
    else:
        return None
    return _value(_CONTEXTS[direction].atan2(_int(y), _int(x))) + 0.0


def rounded_inverse_trig(name: str, v, sign: int, k: int, direction: int) -> Optional[float]:
    """
    `elementary.rounded_inverse_trig`'s double for k = 0: `sign * f(v)`, f rounded in the direction
    reversed where sign < 0. None for k != 0 (`k pi + ...` is a sum: two roundings), ±inf (pi/2, pure)
    """
    if k != 0 or (name == 'acos' and v == -1):
        return None
    r = _native(name, v, direction if sign > 0 else -direction)
    return None if r is None else sign * _value(r) + 0.0


# THE OUTWARD HOOK

def outward(op: str, args, direction: int) -> Optional[float]:
    """
    `ops.outward`'s rounded corner for the descriptor `op` (add sub mul div reciprocal): the exact
    result rounded once. two dyadic operands are one MPFR operation; otherwise the exact value as an
    mpq, rounded by the context. no guard on an exact result here: `0.5 + 0.25` is a double
    """
    if op == 'reciprocal':
        (b,), a, op = args, 1, 'div'
    else:
        a, b = args
    if op not in _OPERATIONS or (op == 'div' and b == 0):
        return None
    context = _CONTEXTS[direction]
    ma, mb = _operand(a), _operand(b)
    if ma is not None and mb is not None:
        return _value(getattr(context, op)(ma, mb), False) + 0.0
    ra, rb = _ratio(a), _ratio(b)
    if ra is None or rb is None:
        return None
    qa, qb = mpq(*ra), mpq(*rb)
    exact = qa + qb if op == 'add' else qa - qb if op == 'sub' else qa * qb if op == 'mul' else qa / qb
    return _value(mpfr(exact, 53, context), False) + 0.0
