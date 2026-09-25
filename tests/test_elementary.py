"""
intervals.elementary against an independent oracle: python's decimal module

decimal's exp, ln, log10 and sqrt are correctly rounded at any precision; the trig functions here are
plain taylor series in decimal with pi from the decimal docs' recipe. at 90 digits the oracle is far
past a double's 17, so it decides the three roundings of each value exactly, unless the value lies
within 1e-70 (relative) of a double or of a midpoint between two, which a sample is then skipped for.
the checks: DOWN <= value <= UP with UP the next double after DOWN, NEAREST the nearer of the two, and
every rational value returned exactly.
"""
import math
import random
from decimal import Decimal
from decimal import localcontext
from fractions import Fraction

import pytest

from intervals import elementary
from intervals.elementary import compare
from intervals.elementary import exact
from intervals.elementary import floor_over_pi
from intervals.elementary import rounded
from intervals.rounding import DOWN
from intervals.rounding import MAX
from intervals.rounding import NEAREST
from intervals.rounding import UP

INF = math.inf
PREC = 90


# THE ORACLE (decimal, PREC digits)

def _pi(prec: int = PREC) -> Decimal:
    """the decimal docs' recipe"""
    with localcontext() as ctx:
        ctx.prec = prec + 10
        three = Decimal(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
        return +s


def _series_sin(x: Decimal) -> Decimal:
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    return s


def _series_cos(x: Decimal) -> Decimal:
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    return s


def _reduced(x: Decimal, prec: int) -> Decimal:
    """x mod 2 pi, into (-pi, pi]; pi to enough digits for |x| (only |x| < 1e25 is used)"""
    two_pi = 2 * _pi(prec)
    k = (x / two_pi).to_integral_value()
    return x - k * two_pi


def _atan(x: Decimal, prec: int = PREC) -> Decimal:
    if x < 0:
        return -_atan(-x, prec)
    if x > 1:
        return _pi(prec) / 2 - _atan(1 / x, prec)
    for _ in range(4):
        x = x / (1 + (1 + x * x).sqrt())
    total, term, k = x, x, 0
    while True:
        k += 1
        term = -term * x * x
        step = term / (2 * k + 1)
        if abs(step) < Decimal(10) ** -(prec + 5):
            break
        total += step
    return 16 * total


def oracle(name: str, x: Fraction, prec: int = PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 20
        d = Decimal(x.numerator) / Decimal(x.denominator)
        ln2, ln10 = Decimal(2).ln(), Decimal(10).ln()
        if name == 'sqrt':
            v = d.sqrt()
        elif name == 'exp':
            v = d.exp()
        elif name == 'exp2':
            v = (d * ln2).exp()
        elif name == 'exp10':
            v = (d * ln10).exp()
        elif name == 'log':
            v = d.ln()
        elif name == 'log2':
            v = d.ln() / ln2
        elif name == 'log10':
            v = d.log10()
        elif name in ('sin', 'cos', 'tan'):
            r = _reduced(d, prec)
            s, c = _series_sin(r), _series_cos(r)
            v = {'sin': s, 'cos': c, 'tan': s / c if c else None}[name]
        elif name == 'atan':
            v = _atan(d, prec)
        elif name == 'asin':
            v = _atan(d / (1 - d * d).sqrt(), prec)
        elif name == 'acos':
            v = _pi(prec) / 2 - _atan(d / (1 - d * d).sqrt(), prec)
        elif name == 'sinh':
            v = (d.exp() - (-d).exp()) / 2
        elif name == 'cosh':
            v = (d.exp() + (-d).exp()) / 2
        elif name == 'tanh':
            v = 1 - 2 / ((2 * d).exp() + 1)
        elif name == 'asinh':
            v = (abs(d) + (d * d + 1).sqrt()).ln().copy_sign(d)
        elif name == 'acosh':
            v = (d + (d * d - 1).sqrt()).ln()
        elif name == 'atanh':
            v = ((1 + d) / (1 - d)).ln() / 2
        else:
            raise ValueError(name)
        return +v


def roundings(v: Decimal):
    """(DOWN, NEAREST, UP) of v, or None where v is within 1e-70 of a double or a midpoint"""
    nearest = float(v)  # decimal -> float is correctly rounded
    if math.isinf(nearest):
        return None
    below = nearest if Decimal(nearest) <= v else math.nextafter(nearest, -INF)
    above = math.nextafter(below, INF) if Decimal(below) != v else below
    width = abs(Decimal(above) - Decimal(below)) or Decimal(1)
    middle = (Decimal(below) + Decimal(above)) / 2
    if min(abs(v - Decimal(below)), abs(v - Decimal(above)), abs(v - middle)) < width * Decimal('1e-60'):
        return None
    return below, nearest, above


# THE DRAWS

def _draw(name: str, rng):
    """an exact point: a float's value, or a small Fraction"""
    r = rng.random()
    if name in ('asin', 'acos', 'atanh'):
        x = rng.uniform(-1, 1)
    elif name == 'acosh':
        x = 1 + math.ldexp(rng.random(), rng.randrange(-50, 30))
    elif name in ('sqrt', 'log', 'log2', 'log10'):
        x = math.ldexp(rng.random(), rng.randrange(-1074, 1024))
    elif name in ('exp', 'sinh', 'cosh'):
        x = rng.uniform(-700, 700) if r < 0.7 else rng.uniform(-3, 3)
    elif name == 'exp2':
        x = rng.uniform(-1070, 1020)
    elif name == 'exp10':
        x = rng.uniform(-300, 300)
    elif name == 'tanh':
        x = rng.uniform(-25, 25)
    elif name in ('sin', 'cos', 'tan'):
        x = rng.choice((-1, 1)) * math.ldexp(rng.random(), rng.randrange(-30, 70))
    else:  # atan, asinh
        x = rng.choice((-1, 1)) * math.ldexp(rng.random(), rng.randrange(-60, 200))
    if r < 0.2 and name not in ('sqrt', 'log', 'log2', 'log10', 'acosh'):
        x = Fraction(rng.randrange(-999, 1000), rng.randrange(1, 1000))
        if name in ('asin', 'acos', 'atanh') and abs(x) >= 1:
            x = 1 / (x + (1 if x >= 0 else -1) * 2)
    return Fraction(x)


@pytest.mark.parametrize('name', elementary.NAMES)
def test_correctly_rounded(name):
    rng = random.Random(name)
    checked = 0
    for _ in range(60):
        x = _draw(name, rng)
        if exact(name, x) is not None:
            continue
        expected = roundings(oracle(name, x))
        if expected is None:
            continue
        got = tuple(rounded(name, x, d) for d in (DOWN, NEAREST, UP))
        assert got == expected, (name, x, got, expected)
        checked += 1
    assert checked >= 40, checked


@pytest.mark.parametrize('base, xs', [
    (3, [Fraction(1, 7), Fraction(10), Fraction(81, 2)]),
    (Fraction(1, 2), [Fraction(3), Fraction(1, 5), Fraction(1000)]),
    (2.5, [Fraction(7), Fraction(1, 3)]),
])
def test_log_to_a_base(base, xs):
    for x in xs:
        with localcontext() as ctx:
            ctx.prec = PREC
            v = (Decimal(x.numerator) / x.denominator).ln() / (Decimal(Fraction(base).numerator)
                                                               / Fraction(base).denominator).ln()
        expected = roundings(v)
        assert tuple(rounded('log', x, d, base) for d in (DOWN, NEAREST, UP)) == expected


def _decimal(q: Fraction) -> Decimal:
    return Decimal(q.numerator) / Decimal(q.denominator)


@pytest.mark.parametrize('name', elementary.NAMES)
def test_enclosures_hold_the_value(name):
    """
    the error bounds themselves: at every working precision the enclosure holds the value. correct
    rounding alone would not show a missing bound except on a value next to a rounding boundary
    """
    rng = random.Random(f'enclose {name}')
    checked = 0
    with localcontext() as ctx:
        ctx.prec = 250
        for _ in range(25):
            x = _draw(name, rng)
            if exact(name, x) is not None or elementary._beyond(name, x, None) is not None:
                continue
            v = oracle(name, x, prec=220)  # sqrt's enclosure is as fine as the operand is long
            for p in (64, 100, 180):
                try:
                    lo, hi = elementary._enclose(name, x, p)
                except elementary._Retry:
                    continue
                assert _decimal(lo) <= v <= _decimal(hi), (name, x, p)
                checked += 1
    assert checked >= 30, checked


@pytest.mark.parametrize('constant, value', [
    (elementary._pi, lambda: _pi()),
    (elementary._ln2, lambda: Decimal(2).ln()),
    (elementary._ln10, lambda: Decimal(10).ln()),
])
@pytest.mark.parametrize('p', [8, 64, 65, 200])
def test_constants_hold_the_value(constant, value, p):
    with localcontext() as ctx:
        ctx.prec = PREC + 30
        lo, hi = constant(p)
        v = value()
        assert Decimal(lo) / 2 ** p <= v <= Decimal(hi) / 2 ** p
        assert hi - lo <= 2 ** 8  # and the enclosure is narrow


@pytest.mark.parametrize('n', [2, 3, 5, 9, 239])
@pytest.mark.parametrize('p', [8, 64, 201])
@pytest.mark.parametrize('alternating', [True, False])
def test_constant_series_hold_the_value(n, p, alternating):
    """
    the raw series behind pi, ln 2 and ln 10, before any outward shift: its stated error bound is all
    that separates the enclosure from a single point. (the taylor loops' bounds are mostly covered by
    the slack of the interval rounding around them, so no sampled value shows them missing; they are
    argued in their docstrings, not pinned here)
    """
    with localcontext() as ctx:
        ctx.prec = PREC + 30
        v = _atan(1 / Decimal(n)) if alternating else ((1 + 1 / Decimal(n)) / (1 - 1 / Decimal(n))).ln() / 2
        lo, hi = elementary._series_atan_inv(n, p, alternating)
        assert Decimal(lo) / 2 ** p <= v <= Decimal(hi) / 2 ** p


# EXTREME POINTS: each checked against the oracle, or a documented value

@pytest.mark.parametrize('name, x', [
    ('exp', Fraction(709.78)), ('exp', Fraction(-745.1)), ('exp', Fraction(-708.5)),
    ('exp', Fraction(1, 10 ** 30)), ('exp', Fraction(-1, 10 ** 30)),
    ('log', Fraction(1, 10 ** 400)), ('log', Fraction(10 ** 400 + 7)), ('log', 1 + Fraction(1, 2 ** 200)),
    ('log2', Fraction(3, 2 ** 1100)), ('log10', Fraction(10 ** 50 + 1)),
    ('sqrt', Fraction(10 ** 401)), ('sqrt', Fraction(2, 10 ** 401)), ('sqrt', Fraction(2) ** -1073),
    ('sin', Fraction(10 ** 22)), ('cos', Fraction(10 ** 22)), ('tan', Fraction(10 ** 22)),
    ('sin', Fraction(355)), ('tan', Fraction(355, 226)), ('cos', Fraction(math.pi / 2)),
    ('sin', Fraction(math.pi)), ('sin', Fraction(1, 10 ** 40)),
    ('atan', Fraction(10 ** 300)), ('atan', Fraction(1, 10 ** 300)),
    ('asin', 1 - Fraction(1, 2 ** 52)), ('acos', 1 - Fraction(1, 2 ** 52)), ('acos', -1 + Fraction(1, 2 ** 52)),
    ('sinh', Fraction(1, 10 ** 20)), ('sinh', Fraction(710)), ('cosh', Fraction(-710)),
    ('tanh', Fraction(19.99)), ('tanh', Fraction(-1, 10 ** 30)),
    ('asinh', Fraction(-10 ** 100)), ('asinh', Fraction(1, 10 ** 25)),
    ('acosh', 1 + Fraction(1, 10 ** 30)), ('acosh', Fraction(10 ** 150)),
    ('atanh', 1 - Fraction(1, 10 ** 40)), ('atanh', Fraction(-1, 10 ** 30)),
    ('exp2', Fraction(-1074.5)), ('exp10', Fraction(-323.5)), ('exp10', Fraction(308.25)),
])
def test_extreme_points(name, x):
    with localcontext() as ctx:
        ctx.prec = 200
        expected = roundings(oracle(name, x))
    assert expected is not None
    assert tuple(rounded(name, x, d) for d in (DOWN, NEAREST, UP)) == expected


def test_sin_of_ten_to_the_22():
    """the classic hard argument reduction (k. c. ng, 1992): sin(1e22) = -0.8522008497671888017727..."""
    assert rounded('sin', Fraction(10 ** 22), NEAREST) == -0.8522008497671888


@pytest.mark.parametrize('name, x, down, nearest, up', [
    # past the float range, without computing the value
    ('exp', 710, MAX, INF, INF),
    ('exp', 10 ** 100, MAX, INF, INF),
    ('exp', -746, 0.0, 0.0, 5e-324),
    ('exp', -10 ** 100, 0.0, 0.0, 5e-324),
    ('exp2', 10 ** 6, MAX, INF, INF),
    ('exp10', Fraction(-10 ** 9, 3), 0.0, 0.0, 5e-324),
    ('sinh', -800, -INF, -INF, -MAX),
    ('cosh', -800, MAX, INF, INF),
    ('tanh', 20, math.nextafter(1.0, 0), 1.0, 1.0),
    ('tanh', -10 ** 300, -1.0, -1.0, math.nextafter(-1.0, 0)),
    # pi/2 at an infinity, and pi at the domain's end
    ('atan', INF, 1.5707963267948966, 1.5707963267948966, 1.5707963267948968),
    ('atan', -INF, -1.5707963267948968, -1.5707963267948966, -1.5707963267948966),
    ('asin', -1, -1.5707963267948968, -1.5707963267948966, -1.5707963267948966),
    ('acos', -1, 3.141592653589793, 3.141592653589793, 3.1415926535897936),
])
def test_known_roundings(name, x, down, nearest, up):
    assert (rounded(name, x, DOWN), rounded(name, x, NEAREST), rounded(name, x, UP)) == (down, nearest, up)


# EXACT VALUES

@pytest.mark.parametrize('name, x, value', [
    ('sqrt', Fraction(9, 4), Fraction(3, 2)), ('sqrt', 0, 0), ('sqrt', INF, INF), ('sqrt', Fraction(2), None),
    ('sqrt', Fraction(5e-324), Fraction(2) ** -537), ('sqrt', Fraction(2) ** -1073, None),
    ('exp', 0, 1), ('exp', -INF, 0), ('exp', INF, INF), ('exp', 1, None),
    ('exp2', -3, Fraction(1, 8)), ('exp2', 1024, 2 ** 1024), ('exp2', Fraction(1, 2), None),
    ('exp2', elementary.EXACT_POWER_LIMIT + 1, None),
    ('exp10', -2, Fraction(1, 100)), ('exp10', 0, 1), ('exp10', Fraction(1, 2), None),
    ('log', 1, 0), ('log', 0, -INF), ('log', INF, INF), ('log', 2, None),
    ('log2', Fraction(1, 8), -3), ('log2', Fraction(2 ** -1074), -1074), ('log2', 3, None), ('log2', 0, -INF),
    ('log10', Fraction(10 ** 30), 30), ('log10', Fraction(1, 1000), -3), ('log10', 20, None),
    ('sin', 0, 0), ('cos', 0, 1), ('tan', 0, 0), ('sin', 1, None),
    ('asin', 0, 0), ('asin', 1, None), ('acos', 1, 0), ('acos', 0, None), ('atan', 0, 0), ('atan', INF, None),
    ('sinh', 0, 0), ('sinh', -INF, -INF), ('cosh', 0, 1), ('cosh', -INF, INF),
    ('tanh', 0, 0), ('tanh', INF, 1), ('tanh', -INF, -1),
    ('asinh', 0, 0), ('asinh', INF, INF), ('acosh', 1, 0), ('acosh', INF, INF),
    ('atanh', 0, 0), ('atanh', 1, INF), ('atanh', -1, -INF), ('atanh', Fraction(1, 2), None),
])
def test_exact(name, x, value):
    assert exact(name, x) == value


@pytest.mark.parametrize('x, base, value', [
    (Fraction(81), 3, 4), (Fraction(1, 27), 3, -3), (Fraction(8), Fraction(1, 2), -3), (0, Fraction(1, 2), INF),
    (0, 10, -INF), (INF, Fraction(1, 3), -INF), (Fraction(25, 4), Fraction(5, 2), 2), (Fraction(7), 3, None),
    (Fraction(2), 1 + Fraction(1, 2 ** 100), None),
])
def test_exact_log_to_a_base(x, base, value):
    assert exact('log', x, base) == value


def test_exact_values_round_like_any_other():
    assert rounded('exp10', -1, DOWN) == 0.09999999999999999 and rounded('exp10', -1, UP) == 0.1
    assert rounded('exp2', -1075, NEAREST) == 0.0 and rounded('exp2', -1075, UP) == 5e-324
    assert rounded('exp2', 1024, NEAREST) == INF and rounded('exp2', 1024, DOWN) == MAX


def test_a_missed_exact_case_raises_instead_of_looping(monkeypatch):
    """ziv's loop cannot narrow onto a double that the value IS; the precision cap turns that into an error"""
    monkeypatch.setattr(elementary, '_MAX_PRECISION', 1 << 12)
    monkeypatch.setattr(elementary, 'exact', lambda name, x, base=None: None)
    assert rounded('sqrt', 4, DOWN) == 2.0  # both ends of [2, 2 + tiny] round down to 2
    with pytest.raises(ArithmeticError):
        rounded('sqrt', 4, UP)


# THE HELPERS FOR SIN, COS, TAN

@pytest.mark.parametrize('x, offset, expected', [
    (0, Fraction(1, 2), (-1, False)), (0, Fraction(0), (0, True)),
    (Fraction(3), Fraction(0), (0, False)), (Fraction(4), Fraction(0), (1, False)),
    (Fraction(1.5707963267948966), Fraction(1, 2), (-1, False)),  # the double just below pi/2
    (Fraction(1.5707963267948968), Fraction(1, 2), (0, False)),  # and just above
    (Fraction(-4), Fraction(0), (-2, False)), (Fraction(10 ** 22), Fraction(0), (3183098861837906715377, False)),
])
def test_floor_over_pi(x, offset, expected):
    assert floor_over_pi(x, offset) == expected


@pytest.mark.parametrize('name, x, y, sign', [
    ('sin', 1, 2, -1), ('sin', 2, 1, 1), ('cos', 0, 1, 1),
    # the double below pi/2 is nearer the maximum (6e-17 against 1.6e-16) than the double above
    ('sin', Fraction(1.5707963267948966), Fraction(1.5707963267948968), 1),
    ('tan', 1, Fraction(3, 2), -1),
])
def test_compare(name, x, y, sign):
    assert compare(name, x, y) == sign
