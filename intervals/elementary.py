"""
elementary functions at one exact point, correctly rounded

`rounded(name, x, direction)` is `f(x)` rounded to a double to nearest, down or up, for an exact x
(int or Fraction, or ±inf where f has a limit there); `exact(name, x)` is `f(x)` itself where it is
rational (`sqrt(9/4)`, `exp(0)`, `log2(1/8)`, `tanh(inf)`) and None where it is not. `name` is one
of `NAMES`; `log` also takes a `base`.

nothing here calls libm. a value is enclosed between two dyadic rationals at a working precision of
p bits, every rounding error bounded, and p is doubled until both ends of the enclosure round to the
same double (ziv's strategy). that terminates because an irrational number is never a double nor the
midpoint of two, and the rational values are exactly the ones `exact` returns: for a rational x the
values of exp, log, sin, atan, sinh and their kin are transcendental unless x is the obvious point
(lindemann-weierstrass, gelfond-schneider), `2 ** x` and `10 ** x` are irrational unless x is an int,
and `sqrt(x)` unless x is a square. so a directed result is a true bound, a nearest result is the
correctly rounded one, and both are the same on every platform.

the enclosures are fixed-point intervals: a pair of ints `(lo, hi)` standing for
`[lo / 2**p, hi / 2**p]`, where every operation rounds its low end down and its high end up.
"""
import math
from fractions import Fraction
from functools import lru_cache
from typing import Optional
from typing import Tuple

from intervals.rounding import DOWN
from intervals.rounding import MAX
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import is_infinite
from intervals.rounding import round_rational

INF = math.inf

NAMES = ('sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos',
         'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh')

_START_PRECISION = 64
# `2 ** x` and `10 ** x` for an int x beyond this are not built exactly (2**100000 has 100001 bits);
# they are rounded like any irrational value, which past the float range means [max float, inf]
EXACT_POWER_LIMIT = 100000
_MAX_PRECISION = 1 << 22  # a guard against a missed exact case, which would otherwise loop forever

Fix = Tuple[int, int]  # a fixed-point enclosure at an implied precision p


class _Retry(Exception):
    """the enclosure at this precision cannot answer (a divisor or a sign straddles zero)"""


# FIXED-POINT INTERVAL ARITHMETIC

def _fix(x, p: int) -> Fix:
    """the enclosure of an exact rational"""
    x = Fraction(x)
    n, d = x.numerator << p, x.denominator
    return n // d, -(-n // d)


def _one(p: int) -> Fix:
    return 1 << p, 1 << p


def _fractions(a: Fix, p: int) -> Tuple[Fraction, Fraction]:
    return Fraction(a[0], 1 << p), Fraction(a[1], 1 << p)


def _add(a: Fix, b: Fix) -> Fix:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Fix, b: Fix) -> Fix:
    return a[0] - b[1], a[1] - b[0]


def _neg(a: Fix) -> Fix:
    return -a[1], -a[0]


def _widen(a: Fix, e: int) -> Fix:
    return a[0] - e, a[1] + e


def _mul(a: Fix, b: Fix, p: int) -> Fix:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products) >> p, -(-max(products) >> p)


def _div(a: Fix, b: Fix, p: int) -> Fix:
    if b[0] <= 0 <= b[1]:
        raise _Retry
    return (min((x << p) // y for x in a for y in b),
            max(-(-(x << p) // y) for x in a for y in b))


def _scale(a: Fix, n: int) -> Fix:
    """times an exact int"""
    return (a[0] * n, a[1] * n) if n >= 0 else (a[1] * n, a[0] * n)


def _div_int(a: Fix, n: int) -> Fix:
    """divided by an int n > 0"""
    return a[0] // n, -(-a[1] // n)


def _shift(a: Fix, k: int) -> Fix:
    """times 2**k"""
    if k >= 0:
        return a[0] << k, a[1] << k
    return a[0] >> -k, -(-a[1] >> -k)


def _magnitude(a: Fix) -> int:
    return max(abs(a[0]), abs(a[1]))


# CONSTANTS

def _series_atan_inv(n: int, p: int, alternating: bool) -> Fix:
    """
    atan(1/n) (alternating) or atanh(1/n) for an int n >= 2: sum of ±1 / ((2k+1) n**(2k+1)). each term
    is two floors of an exact quotient, so it is low by less than 2 units; the omitted tail is below
    one unit for atan (alternating, decreasing) and below 4/3 of one for atanh (geometric, n >= 2)
    """
    power = (1 << p) // n
    n2 = n * n
    total = k = 0
    while power:
        term = power // (2 * k + 1)
        total += -term if alternating and k % 2 else term
        power //= n2
        k += 1
    return _widen((total, total), 2 * k + 2)


def _guard(p: int) -> int:
    return p.bit_length() + 8


def _cached(compute):
    """a constant computed at the next multiple of 64 bits and cached there"""
    at = lru_cache(maxsize=16)(compute)

    def constant(p: int) -> Fix:
        top = (p + 63) // 64 * 64
        return _shift(at(top), p - top)
    constant.__doc__ = compute.__doc__
    return constant


@_cached
def _pi(p: int) -> Fix:
    """machin: pi = 16 atan(1/5) - 4 atan(1/239)"""
    q = p + _guard(p)
    a = _scale(_series_atan_inv(5, q, True), 16)
    b = _scale(_series_atan_inv(239, q, True), 4)
    return _shift(_sub(a, b), p - q)


@_cached
def _ln2(p: int) -> Fix:
    """ln 2 = 2 atanh(1/3)"""
    q = p + _guard(p)
    return _shift(_scale(_series_atan_inv(3, q, False), 2), p - q)


@_cached
def _ln10(p: int) -> Fix:
    """ln 10 = 3 ln 2 + ln(5/4) = 3 ln 2 + 2 atanh(1/9)"""
    q = p + _guard(p)
    return _shift(_add(_scale(_ln2(q), 3), _scale(_series_atan_inv(9, q, False), 2)), p - q)


# CORE SERIES (each returns an enclosure at precision p)

def _exp_fix(x: Fix, p: int) -> Tuple[Fraction, Fraction]:
    """
    exp over an enclosure x (|x| < 2**12), as Fractions: x = k ln 2 + r, then exp(r / 2**s) by taylor
    and s squarings. the squarings double the relative error s times, which the guard bits pay for
    """
    s = 12
    mid = float(Fraction(x[0] + x[1], 2 << p))  # an estimate: any k that leaves |r| small is correct
    k = round(mid / math.log(2))
    q = p + s + 24
    xq = _shift(x, q - p)
    r = _shift(_sub(xq, _scale(_ln2(q), k)), -s)
    one = _one(q)
    total = term = one
    n = 0
    while True:
        n += 1
        term = _div_int(_mul(term, r, q), n)
        total = _add(total, term)
        m = _magnitude(term)
        if m <= 1:
            break
    # |r| < 1/2, so the omitted terms add up to less than twice the last one
    e = _widen(total, 2 * m + 2)
    for _ in range(s):
        e = _mul(e, e, q)
    lo, hi = _fractions(e, q)
    scale = Fraction(2) ** k
    return lo * scale, hi * scale


def _log_rational(x: Fraction, p: int) -> Fix:
    """ln x for an exact x > 0: x = 2**e y with y in [2/3, 4/3], ln y = 2 atanh((y - 1) / (y + 1))"""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    y = x / Fraction(2) ** e
    if y > Fraction(4, 3):
        y, e = y / 2, e + 1
    elif y < Fraction(2, 3):
        y, e = y * 2, e - 1
    q = p + _guard(p) + abs(e).bit_length()
    t = (y - 1) / (y + 1)  # |t| <= 1/5
    tt = _fix(t, q)
    t2 = _mul(tt, tt, q)
    total = power = tt
    k = 0
    while True:
        k += 1
        power = _mul(power, t2, q)
        term = _div_int(power, 2 * k + 1)
        total = _add(total, term)
        m = _magnitude(term)
        if m <= 1:
            break
    atanh = _widen(total, 2 * m + 2)
    return _shift(_add(_scale(atanh, 2), _scale(_ln2(q), e)), p - q)


def _log_fractions(lo: Fraction, hi: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    """ln over [lo, hi], 0 < lo <= hi, as Fractions (ln is increasing)"""
    if lo <= 0:
        raise _Retry
    return _fractions(_log_rational(lo, p), p)[0], _fractions(_log_rational(hi, p), p)[1]


def _atan_rational(x: Fraction, p: int) -> Fix:
    """
    atan x for an exact x: odd; pi/2 - atan(1/x) above 1; below 1 the angle is halved three times
    (`t -> t / (1 + sqrt(1 + t**2))`) before the series
    """
    if x < 0:
        return _neg(_atan_rational(-x, p))
    if x == 0:
        return 0, 0
    if x > 1:
        return _sub(_shift(_pi(p), -1), _atan_rational(1 / x, p))
    q = p + _guard(p) + 4
    one = _one(q)
    t = _fix(x, q)
    for _ in range(3):
        root = _sqrt_fix(_add(_mul(t, t, q), one), q)
        t = _div(t, _add(one, root), q)
    t2 = _mul(t, t, q)
    total = power = t
    k = 0
    while True:
        k += 1
        power = _neg(_mul(power, t2, q))
        term = _div_int(power, 2 * k + 1)
        total = _add(total, term)
        m = _magnitude(term)
        if m <= 1:
            break
    return _shift(_scale(_widen(total, 2 * m + 2), 8), p - q)


def _atan_fractions(lo: Fraction, hi: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    """atan over [lo, hi] (increasing)"""
    return _fractions(_atan_rational(lo, p), p)[0], _fractions(_atan_rational(hi, p), p)[1]


def _sqrt_fix(a: Fix, p: int) -> Fix:
    """sqrt over an enclosure of a value >= 0"""
    lo = math.isqrt(max(a[0], 0) << p)
    h = max(a[1], 0) << p
    r = math.isqrt(h)
    return lo, r if r * r == h else r + 1


def _sqrt_fractions(x: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    """sqrt of an exact x > 0 with p bits of relative precision: sqrt(n / d) = sqrt(n d) / d"""
    n = x.numerator * x.denominator
    k = max(0, p - n.bit_length() // 2)
    s = math.isqrt(n << (2 * k))
    d = x.denominator << k
    return Fraction(s, d), Fraction(s + 1, d)


def _sin_cos(x: Fraction, p: int) -> Tuple[Fix, Fix]:
    """(sin x, cos x) for an exact x: x = k pi/2 + r with |r| <= pi/4 (plus a hair), then taylor"""
    size = max(0, abs(x.numerator).bit_length() - x.denominator.bit_length()) + 2
    w = p + size + _guard(p)
    half_pi = _shift(_pi(w), -1)
    xw = _fix(x, w)
    k = round(Fraction(xw[0] + xw[1], half_pi[0] + half_pi[1]))
    q = p + _guard(p)
    r = _shift(_sub(xw, _scale(half_pi, k)), q - w)
    r2 = _mul(r, r, q)
    s = _series_sin_cos(r, r2, q, 1)
    c = _series_sin_cos(_one(q), r2, q, 0)
    quadrant = k % 4
    sin, cos = ((s, c), (c, _neg(s)), (_neg(s), _neg(c)), (_neg(c), s))[quadrant]
    return _shift(sin, p - q), _shift(cos, p - q)


def _series_sin_cos(first: Fix, r2: Fix, q: int, odd: int) -> Fix:
    """sum of (-1)**k r**(2k+odd) / (2k+odd)!; |r| < 0.8, so the terms shrink at least tenfold"""
    total = term = first
    n = odd
    while True:
        term = _neg(_mul(term, r2, q))
        term = _div_int(term, (n + 1) * (n + 2))
        n += 2
        total = _add(total, term)
        m = _magnitude(term)
        if m <= 1:
            break
    return _widen(total, 2 * m + 2)


# ENCLOSURES OF THE FUNCTIONS (x exact and finite, inside the domain, not an exact case)

def _enclose(name: str, x: Fraction, p: int, base=None) -> Tuple[Fraction, Fraction]:
    if name == 'sqrt':
        return _sqrt_fractions(x, p)
    if name == 'exp':
        return _exp_fix(_fix(x, p), p)
    if name == 'exp2':
        q = p + 16
        return _exp_fix(_mul(_fix(x, q), _ln2(q), q), q)
    if name == 'exp10':
        q = p + 16
        return _exp_fix(_mul(_fix(x, q), _ln10(q), q), q)
    if name == 'log':
        if base is not None:
            return _fractions(_div(_log_rational(x, p), _log_rational(Fraction(base), p), p), p)
        return _fractions(_log_rational(x, p), p)
    if name == 'log2':
        return _fractions(_div(_log_rational(x, p), _ln2(p), p), p)
    if name == 'log10':
        return _fractions(_div(_log_rational(x, p), _ln10(p), p), p)
    if name in ('sin', 'cos', 'tan'):
        s, c = _sin_cos(x, p)
        if name == 'sin':
            return _fractions(s, p)
        if name == 'cos':
            return _fractions(c, p)
        return _fractions(_div(s, c, p), p)
    if name == 'atan':
        return _fractions(_atan_rational(x, p), p)
    if name == 'asin':
        if abs(x) == 1:
            lo, hi = _fractions(_shift(_pi(p), -1), p)
            return (lo, hi) if x > 0 else (-hi, -lo)
        # asin x = atan(x / sqrt(1 - x**2)), |x| < 1
        root = _sqrt_fix(_fix(1 - x * x, p), p)
        return _atan_fractions(*_fractions(_div(_fix(x, p), root, p), p), p)
    if name == 'acos':
        if x == -1:
            return _fractions(_pi(p), p)
        # acos x = 2 atan(sqrt((1 - x) / (1 + x))), |x| < 1: no cancellation near 1
        lo, hi = _atan_fractions(*_fractions(_sqrt_fix(_fix((1 - x) / (1 + x), p), p), p), p)
        return 2 * lo, 2 * hi
    if name in ('sinh', 'cosh', 'tanh'):
        return _hyperbolic(name, x, p)
    if name == 'asinh':
        if x < 0:
            lo, hi = _enclose('asinh', -x, p)
            return -hi, -lo
        # asinh x = ln(x + sqrt(x**2 + 1))
        v = _add(_fix(x, p), _sqrt_fix(_fix(x * x + 1, p), p))
        return _log_fractions(*_fractions(v, p), p)
    if name == 'acosh':
        # acosh x = ln(x + sqrt(x**2 - 1)), x > 1
        v = _add(_fix(x, p), _sqrt_fix(_fix(x * x - 1, p), p))
        return _log_fractions(*_fractions(v, p), p)
    if name == 'atanh':
        # atanh x = ln((1 + x) / (1 - x)) / 2, |x| < 1
        lo, hi = _fractions(_log_rational((1 + x) / (1 - x), p), p)
        return lo / 2, hi / 2
    raise ValueError(f'unknown function {name!r}')


def _hyperbolic(name: str, x: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    if name == 'cosh':
        # (e + 1/e) / 2 rises with e for e >= 1, and e = exp|x| >= 1
        lo, hi = _exp_fix(_fix(abs(x), p), p)
        lo = max(lo, Fraction(1))
        return (lo + 1 / lo) / 2, (hi + 1 / hi) / 2
    if x < 0:
        lo, hi = _hyperbolic(name, -x, p)
        return -hi, -lo
    if name == 'sinh':
        # (e - 1/e) / 2 rises with e
        lo, hi = _exp_fix(_fix(x, p), p)
        if lo <= 0:
            raise _Retry
        return (lo - 1 / lo) / 2, (hi - 1 / hi) / 2
    # tanh x = 1 - 2 / (exp(2x) + 1), rising with exp(2x)
    lo, hi = _exp_fix(_fix(2 * x, p), p)
    if lo <= -1:
        raise _Retry
    return 1 - 2 / (lo + 1), 1 - 2 / (hi + 1)


# EXACT VALUES

def exact(name: str, x, base=None):
    """
    `f(x)` where it is rational or ±inf, None where it is irrational (then `rounded` is needed). x is
    an exact value (int, Fraction, ±inf) inside f's domain, the domain's ends included

    >>> exact('sqrt', Fraction(9, 4)), exact('log2', Fraction(1, 8)), exact('tanh', -INF)
    (Fraction(3, 2), -3, -1)
    >>> exact('exp', 1) is None
    True
    """
    if is_infinite(x):
        return _AT_INFINITY[name](x, base)
    x = Fraction(x)
    if name == 'sqrt':
        return _exact_sqrt(x)
    if name in ('sinh', 'tanh', 'asinh', 'sin', 'tan', 'asin', 'atan'):
        return 0 if x == 0 else None
    if name in ('exp', 'cos', 'cosh'):
        return 1 if x == 0 else None
    if name == 'atanh':
        return 0 if x == 0 else INF if x == 1 else -INF if x == -1 else None
    if name in ('exp2', 'exp10'):
        if x.denominator != 1 or abs(x) > EXACT_POWER_LIMIT:
            return None
        return Fraction(2 if name == 'exp2' else 10) ** int(x)
    if name == 'log':
        if x == 0:
            return -INF if base is None or base > 1 else INF
        if base is None:
            return 0 if x == 1 else None
        return _exact_log(x, Fraction(base))
    if name == 'log2':
        return -INF if x == 0 else _exact_log(x, Fraction(2))
    if name == 'log10':
        return -INF if x == 0 else _exact_log(x, Fraction(10))
    if name == 'acos':
        return 0 if x == 1 else None
    if name == 'acosh':
        return 0 if x == 1 else None
    raise ValueError(f'unknown function {name!r}')


def _exact_sqrt(x: Fraction) -> Optional[Fraction]:
    n, d = x.numerator, x.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    return Fraction(rn, rd) if rn * rn == n and rd * rd == d else None


def _exact_log(x: Fraction, b: Fraction) -> Optional[int]:
    """the int k with b**k == x, if there is one (b > 0, b != 1, x > 0)"""
    if x == 1:
        return 0
    # b**k in lowest terms is num(b)**k / den(b)**k (or its inverse), and one of those is >= 2, so an
    # exact k is no longer than x's numerator or denominator in bits
    bound = max(x.numerator.bit_length(), x.denominator.bit_length()) + 1
    p = _START_PRECISION
    while True:
        try:
            lo, hi = _fractions(_div(_log_rational(x, p), _log_rational(b, p), p), p)
        except _Retry:
            p *= 2
            continue
        if lo > bound or hi < -bound:
            return None
        if hi - lo < 2:
            break
        p *= 2
    for k in range(math.floor(lo), math.ceil(hi) + 1):
        if b ** k == x:
            return k
    return None


def _log_at_infinity(x, base):
    if base is not None and base < 1:
        return -x
    return x


_AT_INFINITY = {
    'sqrt': lambda x, base: x,
    'exp': lambda x, base: x if x > 0 else 0,
    'exp2': lambda x, base: x if x > 0 else 0,
    'exp10': lambda x, base: x if x > 0 else 0,
    'log': _log_at_infinity,
    'log2': lambda x, base: x,
    'log10': lambda x, base: x,
    'sinh': lambda x, base: x,
    'cosh': lambda x, base: INF,
    'tanh': lambda x, base: 1 if x > 0 else -1,
    'asinh': lambda x, base: x,
    'acosh': lambda x, base: x,
    'atan': lambda x, base: None,  # ±pi/2
}


# ROUNDING

def rounded(name: str, x, direction: int, base=None) -> float:
    """
    f(x) rounded to a double (DOWN, NEAREST or UP) for an exact x inside the domain

    >>> rounded('exp', 1, DOWN), rounded('exp', 1, UP)
    (2.718281828459045, 2.7182818284590455)
    >>> rounded('atan', INF, NEAREST) == math.pi / 2
    True
    """
    value = exact(name, x, base)
    if value is not None:
        return value if is_infinite(value) else round_rational(value, direction)
    if is_infinite(x):  # atan(±inf) = ±pi/2
        sign = 1 if x > 0 else -1
        return sign * _ziv(lambda p: _fractions(_shift(_pi(p), -1), p), direction * sign)
    x = Fraction(x)
    outside = _beyond(name, x, base)
    if outside is not None:
        return _round_outside(outside, direction)
    return _ziv(lambda p: _enclose(name, x, p, base), direction)


def _ziv(enclose, direction: int) -> float:
    p = _START_PRECISION
    while p <= _MAX_PRECISION:
        try:
            lo, hi = enclose(p)
        except _Retry:
            p *= 2
            continue
        a, b = round_rational(lo, direction), round_rational(hi, direction)
        if a == b:
            return a
        p *= 2
    raise ArithmeticError('the enclosure never narrowed to one double: an exact case was missed')


def _beyond(name: str, x: Fraction, base):
    """
    where the value is known to round like a value just past a double, without computing it: past the
    float range (`'overflow'` / `'-overflow'`), just above 0 or just below ±1. None elsewhere
    """
    if name == 'exp':
        return 'overflow' if x >= 710 else 'above 0' if x <= -746 else None
    if name == 'exp2':
        return 'overflow' if x >= 1024 else 'above 0' if x <= -1076 else None
    if name == 'exp10':
        return 'overflow' if x >= 309 else 'above 0' if x <= -324 else None
    if name in ('sinh', 'cosh') and abs(x) >= 711:
        return 'overflow' if x > 0 or name == 'cosh' else '-overflow'
    if name == 'tanh' and abs(x) >= 20:
        # 1 - tanh x = 2 / (exp(2x) + 1) < 2**-54 there, so it rounds like a value just below 1
        return 'below 1' if x > 0 else 'above -1'
    return None


def _round_outside(where: str, direction: int) -> float:
    below_one = math.nextafter(1.0, 0.0)
    return {
        'overflow': {DOWN: MAX, NEAREST: INF, UP: INF},
        '-overflow': {DOWN: -INF, NEAREST: -INF, UP: -MAX},
        'above 0': {DOWN: 0.0, NEAREST: 0.0, UP: math.ulp(0.0)},
        'below 1': {DOWN: below_one, NEAREST: 1.0, UP: 1.0},
        'above -1': {DOWN: -1.0, NEAREST: -1.0, UP: -below_one},
    }[where][direction]


# ANGLES, FOR ATAN2

def rounded_angle(q, m: int, direction: int) -> float:
    """
    `atan(q) + m * pi/2` rounded to a double, for an exact q and an int m: every value of atan2 is one
    (`atan2(y, x)` = atan(y/x) + {0, pi, -pi} by quadrant, and ±pi/2 on the axis). it is rational only
    at q = 0, m = 0, so ziv's loop ends everywhere else

    >>> rounded_angle(0, 2, DOWN), rounded_angle(-1, 0, NEAREST)
    (3.141592653589793, -0.7853981633974483)
    """
    q = Fraction(q)
    if q == 0 and m == 0:
        return 0.0

    def enclose(p):
        lo, hi = _fractions(_add(_atan_rational(q, p), _scale(_shift(_pi(p), -1), m)), p)
        return lo, hi
    return _ziv(enclose, direction)


# PI, FOR THE PERIODIC FUNCTIONS

def floor_over_pi(x, offset: Fraction) -> Tuple[int, bool]:
    """
    `(floor(x / pi - offset), exact)` for an exact finite x, where `exact` says `x / pi - offset` is
    that integer itself. it is irrational unless x == 0, so the loop ends
    """
    x = Fraction(x)
    if x == 0:
        v = -offset
        return math.floor(v), v.denominator == 1
    p = _START_PRECISION + max(0, abs(x.numerator).bit_length() - x.denominator.bit_length())
    while True:
        lo, hi = _fractions(_pi(p), p)
        a, b = sorted((x / hi, x / lo))
        if math.floor(a - offset) == math.floor(b - offset):
            return math.floor(a - offset), False
        p *= 2


def compare(name: str, x, y) -> int:
    """the sign of f(x) - f(y) for exact x, y where the two values are known to differ"""
    p = _START_PRECISION
    while p <= _MAX_PRECISION:
        try:
            a, b = _point_enclosure(name, x, p), _point_enclosure(name, y, p)
        except _Retry:
            p *= 2
            continue
        if a[1] < b[0]:
            return -1
        if b[1] < a[0]:
            return 1
        p *= 2
    raise ArithmeticError('the two values never separated: they are equal')


def _point_enclosure(name: str, x, p: int) -> Tuple[Fraction, Fraction]:
    value = exact(name, x)
    if value is not None:
        return Fraction(value), Fraction(value)
    return _enclose(name, Fraction(x), p)
