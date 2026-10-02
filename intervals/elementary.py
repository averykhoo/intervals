"""
elementary functions at one exact point, correctly rounded

`rounded(name, x, direction)` is `f(x)` rounded to a double to nearest, down or up, for an exact x
(int or Fraction, or ±inf where f has a limit there); `exact(name, x)` is `f(x)` itself where it is
rational (`sqrt(9/4)`, `exp(0)`, `log2(1/8)`, `tanh(inf)`) and None where it is not. `name` is one
of `NAMES`, or `rootn`; `log` also takes a `base`, and `rootn` its degree n (an int other than 0) in
the same argument. `rounded_pow(x, y, direction)` and `exact_pow(x, y)` are the same for `x ** y`.

nothing here calls libm. a value is enclosed between two dyadic rationals at a working precision of
p bits, every rounding error bounded, and p is doubled until both ends of the enclosure round to the
same double (ziv's strategy). that terminates because an irrational number is never a double nor the
midpoint of two, and the rational values are exactly the ones `exact` returns: for a rational x the
values of exp, log, sin, atan, sinh and their kin are transcendental unless x is the obvious point
(lindemann-weierstrass, gelfond-schneider), `2 ** x` and `10 ** x` are irrational unless x is an int,
`sqrt(x)` unless x is a square (and the n-th root unless x is an n-th power, so `x ** (a/b)` in lowest
terms unless x is a b-th power). so a directed result is a true bound, a nearest result is the
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
from intervals import backend

INF = math.inf

NAMES = ('sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos',
         'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
         'expm1', 'log1p', 'cbrt', 'cot', 'sec', 'csc', 'acot', 'coth', 'csch', 'sech', 'acoth')
# the poles at 0, where `exact` and `rounded` have no value to give (the set level takes the limits)
POLE_AT_ZERO = ('cot', 'csc', 'coth', 'csch')

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
    if name == 'expm1':
        return _expm1_fractions(x, p)
    if name == 'log1p':
        # ln(1 + x) of the exact 1 + x; the series' error is absolute, so a small x needs more bits
        q = p + _tiny_bits(x)
        return _fractions(_log_rational(1 + x, q), q)
    if name == 'cbrt':
        return _root_fractions(x, 3, p)
    if name == 'rootn':
        return _root_fractions(x, base, p)
    if name in ('cot', 'csc', 'sec'):
        s, c = _sin_cos(x, p)
        top, bottom = {'cot': (c, s), 'csc': (_one(p), s), 'sec': (_one(p), c)}[name]
        return _fractions(_div(top, bottom, p), p)
    if name == 'acot':
        # pi/2 - atan x, continuous and falling from pi to 0: atan(1/x) above 0, pi + atan(1/x) below
        if x == 0:
            return _fractions(_shift(_pi(p), -1), p)
        angle = _atan_rational(1 / x, p)
        return _fractions(angle if x > 0 else _add(_pi(p), angle), p)
    if name in ('coth', 'csch', 'sech'):
        return _reciprocal_hyperbolic(name, x, p)
    if name == 'acoth':
        # acoth x = atanh(1/x) = ln((x + 1) / (x - 1)) / 2, |x| > 1
        lo, hi = _fractions(_log_rational((x + 1) / (x - 1), p), p)
        return lo / 2, hi / 2
    raise ValueError(f'unknown function {name!r}')


def _tiny_bits(x: Fraction) -> int:
    """how far below 1 |x| is, in bits (0 for |x| >= 1/2): the extra precision a value near x needs"""
    return max(0, x.denominator.bit_length() - abs(x.numerator).bit_length()) + 8


def _expm1_fractions(x: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    """exp(x) - 1 for an exact x (|x| < 2**12), with p bits relative to the value even near 0"""
    q = p + _tiny_bits(x)
    lo, hi = _exp_fix(_fix(x, q), q)
    return lo - 1, hi - 1


def _root_fractions(x: Fraction, n: int, p: int) -> Tuple[Fraction, Fraction]:
    """the n-th root of an exact x != 0 (n odd if x < 0; n < 0 for its reciprocal) as exp(ln|x| / n)"""
    if x < 0:
        lo, hi = _root_fractions(-x, n, p)
        return -hi, -lo
    q = p + 16
    ln = _log_rational(x, q)
    return _exp_fix(_div_int(ln, n) if n > 0 else _neg(_div_int(ln, -n)), q)


def _reciprocal_hyperbolic(name: str, x: Fraction, p: int) -> Tuple[Fraction, Fraction]:
    """coth, csch (x != 0) and sech, from exp and expm1 so that nothing cancels near 0"""
    if name == 'sech':
        # 2 / (e + 1/e) falls as e = exp|x| >= 1 rises
        lo, hi = _exp_fix(_fix(abs(x), p), p)
        lo = max(lo, Fraction(1))
        return 2 / (hi + 1 / hi), 2 / (lo + 1 / lo)
    if x < 0:  # both odd
        lo, hi = _reciprocal_hyperbolic(name, -x, p)
        return -hi, -lo
    if name == 'coth':
        # 1 + 2 / expm1(2x), falling as expm1(2x) > 0 rises
        lo, hi = _expm1_fractions(2 * x, p)
        if lo <= 0:
            raise _Retry
        return 1 + 2 / hi, 1 + 2 / lo
    # csch x = 2 / (expm1(x) - expm1(-x)), the difference being 2 sinh x > 0
    a_lo, a_hi = _expm1_fractions(x, p)
    b_lo, b_hi = _expm1_fractions(-x, p)
    if a_lo - b_hi <= 0:
        raise _Retry
    return 2 / (a_hi - b_lo), 2 / (a_lo - b_hi)


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

# each function's domain where it is not the whole extended line, its ends included (each holds a value or a
# one-sided limit): the set layer's (`functions.domain`), which clips before it calls here. pinned to it by
# tests/test_elementary.py::test_the_scalar_domain_is_the_set_layers
_DOMAIN = {'sqrt': (0, INF), 'log': (0, INF), 'log2': (0, INF), 'log10': (0, INF), 'log1p': (-1, INF),
           'asin': (-1, 1), 'acos': (-1, 1), 'acosh': (1, INF), 'atanh': (-1, 1)}
_FINITE_ONLY = ('sin', 'cos', 'tan', 'cot', 'sec', 'csc')


def _check_domain(name: str, x, base) -> None:
    """a ValueError where x is outside f's domain: past it the series and ziv's loop never settle"""
    if name == 'rootn':
        outside = base % 2 == 0 and x < 0
    elif name == 'acoth':
        outside = -1 < x < 1
    elif name in _FINITE_ONLY:
        outside = is_infinite(x)
    else:
        low, high = _DOMAIN.get(name, (-INF, INF))
        outside = not low <= x <= high
    if outside:
        raise ValueError(f'{name} has no value at {x}, outside its domain')


def exact(name: str, x, base=None):
    """
    `f(x)` where it is rational or ±inf, None where it is irrational (then `rounded` is needed). x is
    an exact value (int, Fraction, ±inf) inside f's domain, the domain's ends included

    >>> exact('sqrt', Fraction(9, 4)), exact('log2', Fraction(1, 8)), exact('tanh', -INF)
    (Fraction(3, 2), -3, -1)
    >>> exact('exp', 1) is None
    True
    """
    _check_domain(name, x, base)
    if name == 'rootn':
        return _exact_rootn(x, base)
    if is_infinite(x):
        return _AT_INFINITY[name](x, base)
    x = Fraction(x)
    if name in POLE_AT_ZERO and x == 0:
        raise ValueError(f'{name} has a pole at 0, so no value there')
    if name == 'sqrt':
        return _exact_sqrt(x)
    if name == 'cbrt':
        return _exact_root(x, 3)
    if name in ('expm1', 'cot', 'csc', 'coth', 'csch', 'acot'):
        return 0 if x == 0 and name == 'expm1' else None
    if name in ('sec', 'sech'):
        return 1 if x == 0 else None
    if name == 'log1p':
        return 0 if x == 0 else -INF if x == -1 else None
    if name == 'acoth':
        return INF if x == 1 else -INF if x == -1 else None
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


def _iroot(n: int, k: int) -> int:
    """the floor of the k-th root of an int n >= 0 (newton's method from above)"""
    if n < 2 or k == 1:
        return n
    if k >= n.bit_length():
        return 1
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _exact_root(x: Fraction, k: int) -> Optional[Fraction]:
    """the rational k-th root of x (k >= 1, odd if x < 0), or None if there is none"""
    if x < 0:
        r = _exact_root(-x, k)
        return None if r is None else -r
    n, d = x.numerator, x.denominator
    rn, rd = _iroot(n, k), _iroot(d, k)
    return Fraction(rn, rd) if rn ** k == n and rd ** k == d else None


def _exact_rootn(x, n: int):
    """rootn(x, n) where rational: at ±inf its limit, at 0 for n < 0 the limit from above (+inf)"""
    if is_infinite(x):
        return 0 if n < 0 else x
    x = Fraction(x)
    if x == 0:
        return 0 if n > 0 else INF
    r = _exact_root(x, abs(n))
    return None if r is None else r if n > 0 else 1 / r


def _exact_log(x: Fraction, b: Fraction):
    """
    `log_b x` where it is rational (an int or a Fraction), None where it is not (b > 0, b != 1, x > 0).
    with x, b > 1 (an operand below 1 is inverted and the sign flipped), `log_b x` = e > 0 is rational
    iff `x ** q == b ** p` for e = p/q, which in lowest terms is `num(x) ** q == num(b) ** p` and
    `den(x) ** q == den(b) ** p`: so e is the rational log of the two numerators (`::_log_ratio`),
    and the two denominators must have the same one (or both be 1). so `log_4 2` = 1/2 and
    `log_8 1/4` = -2/3, which a search for an int k with b**k == x missed (M13e: ziv's loop then never
    settled, `MultiInterval(2).log(4)` hung)

    >>> _exact_log(Fraction(2), Fraction(4)), _exact_log(Fraction(1, 4), Fraction(8)), _exact_log(Fraction(3), Fraction(2))
    (Fraction(1, 2), Fraction(-2, 3), None)
    """
    if x == 1:
        return 0
    sign = 1
    if x < 1:
        x, sign = 1 / x, -sign
    if b < 1:
        b, sign = 1 / b, -sign
    e = _log_ratio(x.numerator, b.numerator)
    if e is None:
        return None
    if x.denominator != 1 or b.denominator != 1:
        if x.denominator == 1 or b.denominator == 1 or _log_ratio(x.denominator, b.denominator) != e:
            return None
    k = sign * e
    return k.numerator if k.denominator == 1 else k


def _log_ratio(a: int, b: int) -> Optional[Fraction]:
    """
    the rational e with `a ** q == b ** p` (e = p/q), for ints a, b >= 2, or None if there is none.
    euclid on the exponents: if a = u ** h and b = u ** g, the larger is the smaller times a power of
    the smaller (`::_divide_out`) times `u ** (h mod g)`, so `a = b ** t * a'` and `log_b a = t +
    log_b a'`, down to a' = 1; a step where the larger is not divisible by the smaller shows there is
    no such u. each step is a few big-int divisions and multiplications, O(log) steps in all (M13e's
    review: the per-prime root search it replaces took minutes on `3 ** 20000 + 1`)
    """
    # log_{b} a as a continued fraction: a = b ** t0 * a1, b = a1 ** t1 * a2, ... until some a_i is 1
    quotients = []
    while True:
        if a == b:
            quotients.append(1)
            break
        if a < b:
            a, b = b, a
            quotients.append(0)
        rest, t = _divide_out(a, b)
        if t == 0:
            return None
        quotients.append(t)
        if rest == 1:
            break
        a, b = b, rest
    # fold the continued fraction [q0; q1, q2, ...] back up (a 0 swaps: log_b a = 1 / log_a b)
    e = Fraction(quotients[-1])
    for q in reversed(quotients[:-1]):
        e = q + 1 / e
    return e


def _divide_out(a: int, b: int) -> Tuple[int, int]:
    """`(a', t)` with `a == b ** t * a'` and b not dividing a', for ints a >= 1, b >= 2 (t in binary)"""
    powers = [b]
    while powers[-1] ** 2 <= a:
        powers.append(powers[-1] ** 2)
    t = 0
    for i in reversed(range(len(powers))):  # b ** t | a with t < 2 ** len(powers), so each bit once
        q, r = divmod(a, powers[i])
        if r == 0:
            a, t = q, t + (1 << i)
    return a, t


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
    'expm1': lambda x, base: x if x > 0 else -1,
    'log1p': lambda x, base: x,
    'cbrt': lambda x, base: x,
    'acot': lambda x, base: 0 if x > 0 else None,  # pi at -inf
    'coth': lambda x, base: 1 if x > 0 else -1,
    'csch': lambda x, base: 0,
    'sech': lambda x, base: 0,
    'acoth': lambda x, base: 0,
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
    if is_infinite(x):  # atan(±inf) = ±pi/2, acot(-inf) = pi
        if name == 'acot':
            return _ziv(lambda p: _fractions(_pi(p), p), direction)
        sign = 1 if x > 0 else -1
        return sign * _ziv(lambda p: _fractions(_shift(_pi(p), -1), p), direction * sign)
    x = Fraction(x)
    outside = _beyond(name, x, base)
    if outside is not None:
        return _round_outside(outside, direction)
    fast = backend.fast  # read at each call: the backend only picks the double (intervals._gmpy2)
    if fast is not None and (answer := fast.rounded(name, x, direction, base)) is not None:
        return answer
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
    if name == 'expm1':
        # exp(x) < 2**-57 below -40, so exp(x) - 1 rounds like a value just above -1
        return 'overflow' if x >= 710 else 'above -1' if x <= -40 else None
    if name == 'coth' and abs(x) >= 20:
        # coth x - 1 = 2 / (exp(2x) - 1) < 2**-56 there
        return 'above 1' if x > 0 else 'below -1'
    if name in ('csch', 'sech') and abs(x) >= 747:
        # |f(x)| < 2 exp(-|x|) < 2**-1076, under half the least subnormal
        return 'below 0' if name == 'csch' and x < 0 else 'above 0'
    return None


def _round_outside(where: str, direction: int) -> float:
    below_one = math.nextafter(1.0, 0.0)
    return {
        'overflow': {DOWN: MAX, NEAREST: INF, UP: INF},
        '-overflow': {DOWN: -INF, NEAREST: -INF, UP: -MAX},
        'above 0': {DOWN: 0.0, NEAREST: 0.0, UP: math.ulp(0.0)},
        'below 0': {DOWN: -math.ulp(0.0), NEAREST: 0.0, UP: 0.0},
        'below 1': {DOWN: below_one, NEAREST: 1.0, UP: 1.0},
        'above 1': {DOWN: 1.0, NEAREST: 1.0, UP: math.nextafter(1.0, 2.0)},
        'above -1': {DOWN: -1.0, NEAREST: -1.0, UP: -below_one},
        'below -1': {DOWN: -math.nextafter(1.0, 2.0), NEAREST: -1.0, UP: -1.0},
    }[where][direction]


# POWERS, FOR 1788'S POW

def exact_pow(x, y):
    """
    `x ** y` where it is rational, None where it is not (then `rounded_pow` is needed), for exact
    finite x > 0, or x = 0 with y > 0. with y = a/b in lowest terms it is rational iff x is a b-th
    power. a value longer than `EXACT_POWER_LIMIT` bits is not built; ziv's loop still ends for it,
    since a value that long in lowest terms is neither a double nor the midpoint of two

    >>> exact_pow(Fraction(9, 4), Fraction(3, 2)), exact_pow(8, Fraction(-2, 3)), exact_pow(2, Fraction(1, 2))
    (Fraction(27, 8), Fraction(1, 4), None)
    """
    x, y = Fraction(x), Fraction(y)
    if x == 0:
        return Fraction(0)
    if y == 0 or x == 1:
        return Fraction(1)
    root = x if y.denominator == 1 else _exact_root(x, y.denominator)
    if root is None:
        return None
    a = y.numerator
    if abs(a) * max(root.numerator.bit_length(), root.denominator.bit_length()) > EXACT_POWER_LIMIT:
        return None
    return root ** a


def rounded_pow(x, y, direction: int) -> float:
    """
    `x ** y` rounded to a double, for the operands of `exact_pow`: exp(y ln x) enclosed, or known to
    round like a value past the float range

    >>> rounded_pow(2, Fraction(1, 2), DOWN), rounded_pow(2, Fraction(1, 2), UP)
    (1.414213562373095, 1.4142135623730951)
    """
    value = exact_pow(x, y)
    if value is not None:
        return round_rational(value, direction)
    x, y = Fraction(x), Fraction(y)
    lo, hi = sorted(y * b for b in _ln_bracket(x))
    # ln(x ** y) is known within a factor of 3.1, so an undecided one is under 2500 and exp's argument
    # stays small; e**800 > 2**1154 is past the float range, and e**-800 under half a subnormal
    if lo > 800:
        return _round_outside('overflow', direction)
    if hi < -800:
        return _round_outside('above 0', direction)
    fast = backend.fast
    if fast is not None and (answer := fast.rounded_pow(x, y, direction)) is not None:
        return answer
    extra = max(0, abs(y.numerator).bit_length() - y.denominator.bit_length()) + 16

    def enclose(p):
        q = p + extra
        return _exp_fix(_mul(_log_rational(x, q), _fix(y, q), q), q)
    return _ziv(enclose, direction)


def _ln_bracket(x: Fraction) -> Tuple[Fraction, Fraction]:
    """two rationals of one sign holding ln x (x > 0, x != 1), within a factor of 3.1 of each other"""
    if Fraction(1, 2) <= x <= 2:
        # ln x = 2 atanh(t) with |t| <= 1/3, and atanh(t) lies between t and t / (1 - t**2)
        t = (x - 1) / (x + 1)
        return tuple(sorted((2 * t, 2 * t / (1 - t * t))))
    # x lies in (2**(e-1), 2**(e+1)), and ln 2 in (0.69, 0.7)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    low, high = Fraction(69, 100), Fraction(7, 10)
    if x > 2:
        return max((e - 1) * low, low), (e + 1) * high
    return (e - 1) * high, min((e + 1) * low, -low)


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
    fast = backend.fast
    if fast is not None and (answer := fast.rounded_angle(q, m, direction)) is not None:
        return answer

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


def rounded_inverse_trig(name: str, v, sign: int, k: int, direction: int) -> float:
    """
    `k pi + sign * f(v)` rounded to a double (DOWN, NEAREST or UP), f one of asin, acos, atan at an exact
    v in its domain (atan also at ±inf, where it is ±pi/2), sign ±1 and k an int: the ends of the
    periodic reverse ops' branches (`sin_rev`'s k-th is `k pi + (-1)**k asin`). `acos(-1)` is pi, taken
    into k; then the value is rational only where k = 0 and f(v) = 0, so ziv's loop ends everywhere
    else: `k pi + r` for a rational r != 0 would make v the transcendental sin, cos or tan of a
    rational, and asin(±1), atan(±inf) are odd multiples of pi/2. the working precision grows with k's
    bits, since the value's size does

    >>> rounded_inverse_trig('asin', 1, 1, 0, UP) == math.nextafter(math.pi / 2, INF)
    True
    >>> rounded_inverse_trig('atan', INF, 1, -1, NEAREST) == -math.pi / 2
    True
    >>> rounded_inverse_trig('acos', 1, -1, 10 ** 20, DOWN)  # 1e20 pi, below it
    3.141592653589793e+20
    """
    if name == 'acos' and v == -1:  # pi: missed, the value (k + sign) pi = 0 would never settle
        v, k = 1, k + sign
    value = None if is_infinite(v) else exact(name, v)
    if value is not None and k == 0:
        return round_rational(sign * value, direction)
    fast = backend.fast
    if fast is not None and (answer := fast.rounded_inverse_trig(name, v, sign, k, direction)) is not None:
        return answer
    extra = abs(k).bit_length() + 4

    def enclose(p):
        q = p + extra
        if value is not None:
            lo = hi = Fraction(sign * value)
        elif is_infinite(v):  # atan(±inf) = ±pi/2
            lo, hi = _fractions(_shift(_pi(q), -1), q)
            if sign * v < 0:
                lo, hi = -hi, -lo
        else:
            lo, hi = _enclose(name, Fraction(v), q)
            if sign < 0:
                lo, hi = -hi, -lo
        k_lo, k_hi = _fractions(_scale(_pi(q), k), q)
        return lo + k_lo, hi + k_hi
    return _ziv(enclose, direction)


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
