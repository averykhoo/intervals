"""
the step functions: ceil, trunc, round (ties to even), round_ties_away and sign (multiinterval.steps; floor
has its own tests in tests/test_modulo.py, which now run through the same engine)

the oracle is each function's preimages, written out independently: grid value n is in the result
iff the operand meets the preimage of n. on single points the functions must agree with python's
`math.ceil`, `math.trunc`, `round` and, for round_ties_away, the decimal module's ROUND_HALF_UP.

float and mixed operands, from subnormals to the largest double with ±inf, in both classes: a listed
result is the preimages' set with each value of a float piece made a double (to nearest, or outward
the open gap between the doubles around it, from `int / int` and `nextafter`, not the package's
rounding); `round(A, ndigits)` against the decimal module's exact grid value; past the cap (or for
a piece reaching ±inf) the warning and exactly the documented hull, from the least value attained to
the greatest; sign never hulls. and the laws of a pointwise image where nothing is hulled:
isotone, distributing over unions, idempotent. two library bugs they found are fixed and pinned
(M14-breadth, 2026-10-02): the outward class's floor, ceil and trunc rounded their values to nearest
(`test_outward_lists_what_no_double_holds`), and the cap counted a value two pieces share twice
(`test_the_cap_counts_distinct_values`). isotone holds in the outward class only where each float
piece of A lies in a float piece of B; a float piece inside an exact one gets f(A) inside the
tightest double-ended cover of f(B) (fuzz-steps-isotone, 2026-10-03: a test oracle, not the library).
"""
import math
import sys
import warnings
from decimal import ROUND_HALF_EVEN
from decimal import ROUND_HALF_UP
from decimal import Decimal
from decimal import localcontext
from fractions import Fraction

import pytest
from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval import steps
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import HullWarning
from multiinterval.fmt import format_cuts
from multiinterval.fmt import parse
from multiinterval.kernel import EMPTY
from multiinterval.kernel import contains_point
from multiinterval.kernel import intersection
from multiinterval.kernel import is_subset
from multiinterval.kernel import normalize
from multiinterval.kernel import pairs
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import union
from tests.oracles import _exact_cuts
from tests.oracles import _finite_float
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import infinities

INF = math.inf
MAX = sys.float_info.max
HALF = Fraction(1, 2)
CAP = steps.ENUMERATION_CAP
NAMES = ('floor', 'ceil', 'trunc', 'round', 'round_ties_away', 'sign')
STEPS = NAMES[1:]  # floor's tests are in tests/test_modulo.py
DIGITS = ('round', 'round_ties_away')  # the two that take ndigits
CLASSES = (MultiInterval, OutwardMultiInterval)


def preimage(name: str, n: int):
    """the points f maps to n, as a cut tuple"""
    if name == 'floor':
        p = piece(n, n + 1, True, False)
    elif name == 'ceil':
        p = piece(n - 1, n, False, True)
    elif name == 'trunc':
        p = piece(n, n + 1, True, False) if n > 0 else piece(n - 1, n, False, True) if n < 0 else piece(-1, 1, False, False)
    elif name == 'round':
        p = piece(n - HALF, n + HALF, n % 2 == 0, n % 2 == 0)
    elif name == 'round_ties_away':
        p = piece(n - HALF, n + HALF, n > 0, n < 0) if n else piece(-HALF, HALF, False, False)
    else:  # sign
        p = {-1: piece(-INF, 0, True, False), 0: piece(0, 0), 1: piece(0, INF, False, True)}[n]
    return normalize([p])


def show(cuts) -> str:
    return format_cuts(cuts)


# EXAMPLES

@pytest.mark.parametrize('name, a, expected', [
    ('ceil', '(1, 3]', '{ [2] , [3] }'),
    ('ceil', '[1, 3)', '{ [1] , [2] , [3] }'),
    ('ceil', '(-1/2, 1/2)', '{ [0] , [1] }'),
    ('ceil', '[inf]', '[inf]'),
    ('ceil', '{ [-inf] , [5/2] }', '{ [-inf] , [3] }'),
    ('trunc', '(-2, 2)', '{ [-1] , [0] , [1] }'),
    ('trunc', '[-2, 2]', '{ [-2] , [-1] , [0] , [1] , [2] }'),
    ('trunc', '(-1, 1)', '[0]'),
    ('trunc', '[-5/2, -2)', '[-2]'),
    ('round', '[1/2, 5/2]', '{ [0] , [1] , [2] }'),
    ('round', '(1/2, 5/2)', '{ [1] , [2] }'),
    ('round', '(1/2, 3/2)', '[1]'),
    ('round', '[3/2]', '[2]'),
    ('round', '[-3/2]', '[-2]'),
    ('round', '(-1/2, 1/2)', '[0]'),
    ('round_ties_away', '[1/2, 5/2]', '{ [1] , [2] , [3] }'),
    ('round_ties_away', '[-5/2, -1/2]', '{ [-3] , [-2] , [-1] }'),
    ('round_ties_away', '(-1/2, 1/2)', '[0]'),
    ('round_ties_away', '[1/2, 3/2)', '[1]'),
    ('sign', '[-2, 0]', '{ [-1] , [0] }'),
    ('sign', '(0, inf]', '[1]'),
    ('sign', '[-inf, inf]', '{ [-1] , [0] , [1] }'),
    ('sign', '(-inf, 0)', '[-1]'),
    ('sign', '[0]', '[0]'),
    ('sign', '[-inf]', '[-1]'),
    ('sign', '[-2.5, 3.0]', '{ [-1.0] , [0.0] , [1.0] }'),
    ('ceil', '[2.5, 4.0]', '{ [3.0] , [4.0] }'),
])
def test_examples(name, a, expected):
    assert show(steps.step(name, parse(a))) == expected


@pytest.mark.parametrize('name, a, expected', [
    ('ceil', '[0, inf)', '[0, inf)'),
    ('ceil', '(-inf, 1/2]', '(-inf, 1]'),
    ('trunc', '[-inf, inf]', '[-inf, inf]'),
    ('round', '[0, 5000]', '[0, 5000]'),
    ('round_ties_away', '(-inf, 0]', '(-inf, 0]'),
])
def test_hulls_with_a_warning(name, a, expected):
    with pytest.warns(HullWarning):
        assert show(steps.step(name, parse(a))) == expected


def test_sign_never_hulls():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert show(steps.sign(parse('[-inf, inf]'))) == '{ [-1] , [0] , [1] }'


def test_cap_counts_across_pieces():
    half = steps.ENUMERATION_CAP // 2
    a = parse(f'{{ [0, {half - 1}] , [{10 * half}, {11 * half - 1}] }}')
    assert len(steps.ceil(a)) // 2 == 2 * half
    with pytest.warns(HullWarning):
        steps.ceil(parse(f'{{ [0, {half}] , [{10 * half}, {11 * half}] }}'))


def test_empty_warns():
    with pytest.warns(EmptySetPropagationWarning):
        assert steps.sign(EMPTY) == EMPTY


def test_bad_arguments():
    with pytest.raises(ValueError):
        steps.step('cbrt', parse('[1]'))
    with pytest.raises(TypeError):
        steps.step('ceil', parse('[1]'), ndigits=2)
    with pytest.raises(TypeError):
        steps.step('round', parse('[1]'), ndigits=2.0)


# NDIGITS

@pytest.mark.parametrize('a, ndigits, expected', [
    ('[1/8, 3/8]', 1, '{ [1/10] , [1/5] , [3/10] , [2/5] }'),
    ('[0.125, 0.135]', 2, '{ [0.12] , [0.13] , [0.14] }'),
    ('[1250, 1350]', -2, '{ [1200] , [1300] , [1400] }'),
    ('[150]', -2, '[200]'),
    ('[250]', -2, '[200]'),
])
def test_ndigits(a, ndigits, expected):
    assert show(steps.round_(parse(a), ndigits)) == expected


@pytest.mark.parametrize('x', [2.675, 0.125, 0.375, -1.005, 12345.678, 1e-5, 2.5, 3.5])
@pytest.mark.parametrize('ndigits', [None, 0, 1, 2, 3, -1])
def test_round_matches_python_on_a_float_point(x, ndigits):
    """python rounds the float's exact value (2.675 is below 2.675, so it goes to 2.67)"""
    expected = round(x, ndigits) if ndigits is not None else float(round(x))
    assert MultiInterval(x).round(ndigits) == MultiInterval(float(expected))


# AGREEMENT WITH PYTHON ON POINTS

def _points():
    return st.one_of(st.integers(-50, 50), st.fractions(min_value=-20, max_value=20, max_denominator=8),
                     st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))


def _half_away(x) -> int:
    """decimal's ROUND_HALF_UP is ties away from zero; the division is exact for these small operands"""
    q = Fraction(x)
    return int((Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


@given(x=_points())
def test_points_agree_with_python(x):
    typed = float if isinstance(x, float) else int
    a = MultiInterval(x)
    assert a.ceil() == MultiInterval(typed(math.ceil(x)))
    assert a.trunc() == MultiInterval(typed(math.trunc(x)))
    assert a.round() == MultiInterval(typed(round(x)))
    assert a.round_ties_away() == MultiInterval(typed(_half_away(x)))
    assert a.sign() == MultiInterval(typed((x > 0) - (x < 0)))
    assert math.floor(a) == a.floor() and math.ceil(a) == a.ceil()
    assert math.trunc(a) == a.trunc() and round(a) == a.round() and round(a, 1) == a.round(1)


# SOUND AND SHARP

@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=150, deadline=None)
@given(a=exact_cut_tuples.filter(bool), rng=st.randoms(use_true_random=False))
def test_sound_and_sharp(name, a, rng):
    """without a hull, n is in the result iff the operand meets n's preimage"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = steps.step(name, a)
    for x in sample(a, 20, rng):
        if x in (-INF, INF):
            v = x if name != 'sign' else (1 if x > 0 else -1)
        else:
            v = {'floor': math.floor, 'ceil': math.ceil, 'trunc': math.trunc, 'round': round,
                 'round_ties_away': _half_away, 'sign': lambda y: (y > 0) - (y < 0)}[name](x)
        assert contains_point(result, v), (name, show(a), x, v, show(result))
    if not caught:
        for n in (range(-1, 2) if name == 'sign' else range(-8, 9)):
            assert contains_point(result, n) == bool(intersection(a, preimage(name, n))), (name, n, show(a), show(result))
        if name != 'sign':
            for v in (-INF, INF):
                assert contains_point(result, v) == contains_point(a, v)


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=50, deadline=None)
@given(a=cut_tuples(max_pieces=3))
def test_float_pieces_give_floats(name, a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = steps.step(name, a)
    float_input = any(isinstance(c.value, float) and math.isfinite(c.value) for c in a)
    if not float_input:
        assert not any(isinstance(c.value, float) and math.isfinite(c.value) for c in result)


# THE CLASS

def test_dunders_return_sets():
    a = MultiInterval.parse('[-5/2, 7/2)')
    assert math.floor(a) == MultiInterval.parse('{ [-3] , [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert math.ceil(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] , [4] }')
    assert math.trunc(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert round(a) == MultiInterval.parse('{ [-2] , [-1] , [0] , [1] , [2] , [3] }')
    assert round(MultiInterval.parse('[1.25, 1.35]'), 1) == MultiInterval.parse('{ [1.2] , [1.3] , [1.4] }')


def test_outward_round_to_digits_encloses_the_grid_point():
    """a grid point that is not a double becomes the open piece between its neighbours, outward"""
    assert OutwardMultiInterval(0.25).round(1) == OutwardMultiInterval.parse('(0.19999999999999998, 0.2)')
    assert MultiInterval(0.25).round(1) == MultiInterval.parse('[0.2]')


# THE ORACLE FOR FLOAT RESULTS

def _nearest(v) -> float:
    """the double nearest the exact v, ties to even (int / int is correctly rounded); past the largest, ±inf"""
    q = Fraction(v)
    try:
        return q.numerator / q.denominator
    except OverflowError:  # int / int returns MAX where the value rounds to it: this is past it
        return INF if q > 0 else -INF  # not copysign, which converts q and overflows again (fuzz x10)


def _down(v) -> float:
    f = _nearest(v)
    return math.nextafter(f, -INF) if f > v else f


def _up(v) -> float:
    f = _nearest(v)
    return math.nextafter(f, INF) if f < v else f


def _enclosure(v, outward: bool):
    """a grid value of a float piece: the double nearest it, or outward the open gap between the doubles around it"""
    if not outward:
        return piece(_nearest(v), _nearest(v))
    lo, hi = _down(v), _up(v)
    return piece(lo, hi) if lo == hi else piece(lo, hi, False, False)


def _rounded_hull(lo, lo_closed, hi, hi_closed, outward: bool):
    """a float piece's hull with its finite ends made doubles: outward a moved end is open; one point stays closed"""
    if outward:
        rlo, rhi = (lo if lo in (-INF, INF) else _down(lo)), (hi if hi in (-INF, INF) else _up(hi))
        lo_closed, hi_closed = lo_closed and rlo == lo, hi_closed and rhi == hi
    else:
        rlo, rhi = (lo if lo in (-INF, INF) else _nearest(lo)), (hi if hi in (-INF, INF) else _nearest(hi))
    return piece(rlo, rhi) if rlo == rhi else piece(rlo, rhi, lo_closed, hi_closed)


def _decimal_round(q: Fraction, ndigits: int, rounding) -> Fraction:
    """q to a multiple of 10 ** -ndigits by the decimal module (exact for these operands: 2000 digits)"""
    with localcontext() as ctx:
        ctx.prec = 2000
        d = Decimal(q.numerator) / Decimal(q.denominator)
        return Fraction(d.quantize(Decimal(1).scaleb(-ndigits), rounding=rounding))


def _at(name: str, x, ndigits=None):
    """f at one point, from python (Fraction's round, math's ceil and trunc) and decimal (ties away)"""
    if x in (-INF, INF):
        return (1 if x > 0 else -1) if name == 'sign' else x
    q = Fraction(x)
    if name == 'ceil':
        return math.ceil(q)
    if name == 'trunc':
        return math.trunc(q)
    if name == 'sign':
        return (q > 0) - (q < 0)
    if name == 'round':
        return round(q) if ndigits is None else round(q, ndigits)
    return _decimal_round(q, ndigits or 0, ROUND_HALF_UP)


def _unit(ndigits) -> Fraction:
    return Fraction(1) if ndigits is None else Fraction(10) ** -ndigits


def _grid_preimage(name: str, n: int, unit: Fraction):
    """`preimage` on the grid of `unit` (round and round_ties_away to ndigits)"""
    if unit == 1:
        return preimage(name, n)
    (lo, lo_closed, hi, hi_closed), = pieces(preimage(name, n))
    return normalize([piece(lo * unit, hi * unit, lo_closed, hi_closed)])


def _image(name: str, a, ndigits=None, outward=False):
    """
    f(a) listed from the preimages, as the class must give it: a piece with a finite float end has its values made
    doubles (`_enclosure`); `(cuts, whether some such value is not a double)`, or None where f(a) must be a hull (a
    piece reaching ±inf, or wider than the cap)
    """
    unit = _unit(ndigits)
    out, enclosed = [], False
    for lo, lo_closed, hi, hi_closed in pieces(a):
        as_float = _finite_float(lo) or _finite_float(hi)
        exact = _exact_cuts(normalize([piece(lo, hi, lo_closed, hi_closed)]))
        if name == 'sign':
            window = (-1, 0, 1)
        else:
            out += [piece(end, end) for end, closed in ((lo, lo_closed), (hi, hi_closed))
                    if closed and end in (-INF, INF)]
            if lo == hi and lo in (-INF, INF):
                continue
            if lo == -INF or hi == INF or (Fraction(hi) - Fraction(lo)) / unit > CAP + 3:
                return None
            window = range(math.floor(Fraction(lo) / unit) - 1, math.ceil(Fraction(hi) / unit) + 2)
        for n in window:
            if intersection(exact, _grid_preimage(name, n, unit)):
                v = n * unit
                enclosed = enclosed or (as_float and _nearest(v) != v)
                out.append(_enclosure(v, outward) if as_float else piece(v, v))
    return normalize(out), enclosed


def _call(cls, name: str, a, ndigits=None):
    """f(a) through the class, as cuts, and whether it warned that it hulled"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        x = cls.from_cuts(a)
        result = getattr(x, name)() if ndigits is None else getattr(x, name)(ndigits)
    return result.cuts, any(issubclass(w.category, HullWarning) for w in caught)


def _closed_hull(cuts):
    return normalize([piece(cuts[0].value, cuts[-1].value)])


# subnormals, the smallest normal, where the doubles stop holding every integer (2 ** 53), the largest double
FLOAT_BASES = [0.0, 0.5, -2.5, 0.1, 1e-5, 12345.678, 5e-324, -5e-324, 1e-320, 2.2250738585072014e-308, 2.0 ** 52 + 0.5,
               2.0 ** 53, -2.0 ** 53 - 2, 2.0 ** 60, -2.0 ** 62, 1e17, 1e300, -1e300, MAX, -MAX]


@st.composite
def float_operands(draw, ndigits=None, floats_only=False, max_pieces=3):
    """
    pieces around one base double drawn from the whole range: its neighbouring doubles, half grid steps from it as a
    double or (a mixed piece) exactly, ±inf, small floats and small exact values
    """
    unit = _unit(ndigits)
    base = draw(st.one_of(st.sampled_from(FLOAT_BASES), st.floats(allow_nan=False, allow_infinity=False),
                          st.floats(-1e6, 1e6, allow_nan=False), st.floats(-30, 30, allow_nan=False)))

    def end():
        kind = draw(st.integers(0, 6 if floats_only else 9))
        if kind <= 1:
            x = base
            for _ in range(draw(st.integers(0, 3))):
                x = math.nextafter(x, draw(infinities))
            return x
        if kind <= 3:
            return _nearest(Fraction(base) + draw(st.integers(-6, 6)) * unit / 2)
        if kind == 4:
            return draw(infinities)
        if kind <= 6:
            return draw(st.floats(-20, 20, allow_nan=False))
        if kind <= 8:
            return Fraction(base) + draw(st.integers(-6, 6)) * unit / 2
        return draw(st.sampled_from([-2, -1, 0, HALF, 1, 3]))

    ps = []
    for _ in range(draw(st.integers(1, max_pieces))):
        lo, hi = sorted((end(), end()))
        ps.append(piece(lo, hi, draw(st.booleans()), draw(st.booleans())))
    return normalize(ps)


def _ndigits(data, name: str, lo=-3, hi=3):
    return data.draw(st.none() | st.integers(1, hi) | st.integers(lo, 0), label='ndigits') if name in DIGITS else None


# FLOAT POINTS

@settings(max_examples=200, deadline=None)
@given(x=st.one_of(st.floats(allow_nan=False), st.sampled_from(FLOAT_BASES)))
def test_every_double_agrees_with_python(x):
    """the whole double range, subnormals to ±inf, both classes: python's math.ceil, math.trunc and round, and the
    decimal module's ROUND_HALF_UP; `f(±inf)` = ±inf, sign's ±1"""
    for cls in CLASSES:
        a = cls(x)
        for name in STEPS:
            v = _at(name, x)
            assert getattr(a, name)() == cls(float(v)), (cls.__name__, name, x)
        if math.isfinite(x):
            assert a.ceil() == cls(float(math.ceil(x))) and a.trunc() == cls(float(math.trunc(x)))
            assert a.round() == cls(float(round(x)))


@settings(max_examples=150, deadline=None)
@given(x=st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.floats(-1e3, 1e3), st.sampled_from(FLOAT_BASES),
                   st.fractions(max_denominator=1000), st.integers(-10 ** 30, 10 ** 30)),
       ndigits=st.integers(-4, 4) | st.integers(-330, 330))
@example(x=1.7976931348623155e+308, ndigits=-293)  # the grid value 1.797693134862316e308 overflows (fuzz x10)
def test_round_to_digits_against_decimal(x, ndigits):
    """
    one point to ndigits against the decimal module's exact grid value g (ROUND_HALF_EVEN, ROUND_HALF_UP): an exact x
    gives g, and Fraction's round agrees; a double gives, in MultiInterval, the double nearest g, as python's
    `round(x, ndigits)` does (whose OverflowError is the point ±inf), and in OutwardMultiInterval g if it is a double,
    else the open gap between the doubles around it
    """
    for name, rounding in (('round', ROUND_HALF_EVEN), ('round_ties_away', ROUND_HALF_UP)):
        g = _decimal_round(Fraction(x), ndigits, rounding)
        for cls in CLASSES:
            got = getattr(cls(x), name)(ndigits).cuts
            want = normalize([_enclosure(g, cls is OutwardMultiInterval) if isinstance(x, float) else piece(g, g)])
            assert got == want, (name, cls.__name__, x, ndigits, g, show(got))
        if name == 'round' and isinstance(x, float):
            try:
                assert round(x, ndigits) == _nearest(g)
            except OverflowError:
                assert _nearest(g) in (-INF, INF)
        elif name == 'round':
            assert round(Fraction(x), ndigits) == g


# FLOAT AND MIXED OPERANDS

@pytest.mark.parametrize('name', STEPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_sound_and_sharp_on_float_operands(name, data, rng):
    """
    float and mixed operands across the double range, both classes: listed, the result is the preimage oracle's set
    (`_image`), a float piece's values made doubles, to nearest or outward; hulled, it holds that set inside its
    closed hull; a piece reaching ±inf or wider than the cap is hulled; sign never is; every sampled point's value is
    in the result (or, to nearest, the double nearest it)
    """
    ndigits = _ndigits(data, name)
    a = data.draw(float_operands(ndigits).filter(bool), label='a')
    for cls in CLASSES:
        outward = cls is OutwardMultiInterval
        result, hulled = _call(cls, name, a, ndigits)
        expected = _image(name, a, ndigits, outward)
        assert not (name == 'sign' and hulled)
        if expected is None:
            assert hulled, (cls.__name__, show(a), show(result))
        elif hulled:
            assert is_subset(expected[0], result) and is_subset(result, _closed_hull(expected[0])), (
                cls.__name__, show(a), show(result), show(expected[0]))
        else:
            assert result == expected[0], (cls.__name__, show(a), show(result), show(expected[0]))
        for x in sample(a, 8, rng):
            v = _at(name, x, ndigits)
            rounded = not outward and v not in (-INF, INF) and contains_point(result, _nearest(v))
            assert contains_point(result, v) or rounded, (
                cls.__name__, show(a), x, v, show(result))


# THE HULL PAST THE CAP

def _first_and_last(name: str, a, unit: Fraction):
    """the least and the greatest grid value a single piece attains (±inf where it has no end), from the preimages"""
    (lo, _, hi, _), = pieces(a)
    exact = _exact_cuts(a)
    meets = [n for n in range(math.floor(Fraction(lo) / unit) - 1, math.floor(Fraction(lo) / unit) + 3)
             if intersection(exact, _grid_preimage(name, n, unit))] if lo != -INF else []
    first = -INF if lo == -INF else min(meets)
    meets = [n for n in range(math.ceil(Fraction(hi) / unit) - 2, math.ceil(Fraction(hi) / unit) + 2)
             if intersection(exact, _grid_preimage(name, n, unit))] if hi != INF else []
    return first, INF if hi == INF else max(meets)


@pytest.mark.parametrize('name', ('ceil', 'trunc', 'round', 'round_ties_away'))
@settings(max_examples=30, deadline=None)
@given(data=st.data())
def test_hull_past_the_cap(name, data):
    """
    one piece of about the cap's width, exact, float or mixed, either end possibly ±inf: it is listed iff it is
    bounded and attains at most ENUMERATION_CAP grid values, else the result is exactly the documented hull with a
    HullWarning: from the least value it attains to the greatest, closed, an infinite end open unless the piece
    holds it, the ends made doubles as a float piece's are
    """
    ndigits = _ndigits(data, name, -2, 2)
    unit = _unit(ndigits)
    start = data.draw(st.integers(-3000, 2000) | st.fractions(-3000, 2000, max_denominator=4), label='start') * unit
    width = data.draw(st.integers(CAP - 3, CAP + 3) | st.integers(0, 3 * CAP), label='width')
    end = start + (width + data.draw(st.sampled_from([0, HALF, Fraction(1, 3), -HALF]))) * unit
    ends = [data.draw(st.sampled_from([v, _nearest(v), inf]), label='end') for v, inf in ((start, -INF), (end, INF))]
    lo, hi = sorted(ends)
    a = normalize([piece(lo, hi, data.draw(st.booleans()), data.draw(st.booleans()))])
    assume(a)
    (lo, lo_closed, hi, hi_closed), = pieces(a)
    # trunc splits a piece at 0 and hulls each half on its own (`trunc((-inf, 997))` is `(-inf, 0]` and the listed
    # 1..996), tighter than the one hull this oracle models: such a piece is left out here; the count of a value both
    # halves share is `test_the_cap_counts_distinct_values`'s
    assume(name != 'trunc' or lo >= 0 or hi < 0 or hi == 0 and not hi_closed)
    first, last = _first_and_last(name, a, unit)
    must_hull = first == -INF or last == INF or last - first + 1 > CAP
    as_float = _finite_float(lo) or _finite_float(hi)
    for cls in CLASSES:
        outward = cls is OutwardMultiInterval
        result, hulled = _call(cls, name, a, ndigits)
        assert hulled == must_hull, (cls.__name__, show(a), first, last)
        if must_hull:
            hull = (-INF if first == -INF else first * unit, first != -INF,
                    INF if last == INF else last * unit, last != INF)
            points = [piece(e, e) for e, closed in ((lo, lo_closed), (hi, hi_closed)) if closed and e in (-INF, INF)]
            want = normalize([_rounded_hull(*hull, outward) if as_float else piece(hull[0], hull[2], hull[1], hull[3]),
                              *points])
        else:  # a connected piece attains every grid value from its first to its last
            want = normalize(_enclosure(n * unit, outward) if as_float else piece(n * unit, n * unit)
                             for n in range(first, last + 1))
        assert result == want, (cls.__name__, show(a), show(result), show(want))



def test_outward_lists_what_no_double_holds():
    """past 2 ** 53 a float piece meets integers that are not doubles: outward, floor, ceil and trunc enclose
    each in the open gap around it, as round does; they rounded it to nearest, so `ceil` of [2 ** 53, 2 ** 53 + 2]
    missed 2 ** 53 + 1 (M14-breadth, 2026-10-02)"""
    a = OutwardMultiInterval(2.0 ** 53, 2.0 ** 53 + 2)
    for f in (a.floor, a.ceil, a.trunc, a.round):
        assert f() == a, f
    assert (-a).trunc() == -a


@pytest.mark.parametrize('name, text', [('ceil', '{ [0, 1/2] , [7/10, 999] }'), ('trunc', '[-1, 1997/2)'),
                                        ('floor', '{ [0, 1/2] , [3/4, 999] }')])
def test_the_cap_counts_distinct_values(name, text):
    """1000 distinct values are listed, not hulled: two pieces sharing a value (ceil's 1, floor's 0) or trunc's
    split sharing 0 counted it twice and hulled at 999 (M14-breadth, 2026-10-02)"""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = getattr(MultiInterval.parse(text), name)()
    assert len(result.cuts) == 2 * CAP


# LAWS

@pytest.mark.parametrize('name', STEPS)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_isotone(name, data):
    """
    A ⊆ B gives f(A) ⊆ f(B) unless f(A) is a hull: in the exact class on exact operands, and in the outward class on
    float and mixed ones where each float piece of A lies in a float piece of B (to nearest it cannot hold: an exact
    piece of A keeps a value B's float piece rounds). a float piece of A inside an exact piece of B gets only
    f(A) ⊆ the tightest double-ended cover of f(B): B's piece is exact, so the outward class computes it exactly
    (as MultiInterval), while A's lists each value that is not a double as the open gap around it
    (`test_isotone_float_piece_in_an_exact_one`)
    """
    ndigits = _ndigits(data, name)
    b = data.draw(float_operands(ndigits).filter(bool), label='b')
    a = intersection(b, data.draw(cut_tuples(), label='c')) or normalize([next(pairs(b))])
    _assert_isotone(name, ndigits, a, b)


def _assert_isotone(name, ndigits, a, b):
    for cls, (sub, sup) in ((MultiInterval, (_exact_cuts(a), _exact_cuts(b))), (OutwardMultiInterval, (a, b))):
        fa, a_hulled = _call(cls, name, sub, ndigits)
        fb, _ = _call(cls, name, sup, ndigits)
        if a_hulled:
            continue
        if cls is OutwardMultiInterval and not _float_pieces_in_float_pieces(sub, sup):
            fb = _cover(fb)
        assert is_subset(fa, fb), (cls.__name__, show(sub), show(sup), show(fa), show(fb))


def _is_float_piece(p) -> bool:
    return _finite_float(p[0]) or _finite_float(p[2])


def _float_pieces_in_float_pieces(a, b) -> bool:
    """each piece of a with a finite float end lies in a piece of b with one (b ⊇ a, so each lies in one)"""
    def cuts(p):
        return piece(p[0], p[2], p[1], p[3])
    return all(any(_is_float_piece(q) for q in pieces(b) if is_subset(cuts(p), cuts(q)))
               for p in pieces(a) if _is_float_piece(p))


def _cover(cuts):
    """the tightest double-ended cover: each end that is not a double moved to the next double outward, and opened"""
    return normalize(_rounded_hull(*p, outward=True) for p in pieces(cuts))


@pytest.mark.parametrize('name, ndigits, a, b', [
    ('round_ties_away', 1, '(0.0, 1/10)', '(-inf, 1/10)'),  # x50 fuzz, 2026-10-03: B's hull keeps 1/10
    ('round', 1, '(-1.0, -1/20)', '(-inf, -1/20)'),  # x50 fuzz, 2026-10-03
    ('round', 1, '[0.0, 0.25]', '[-1, 3/10]'),  # nothing hulled: f(B) lists 1/10 and 1/5 exactly
    ('ceil', None, '[9007199254740992.0, 9007199254740994.0]', '[18014398509481983/2, 9007199254740995]'),
])
def test_isotone_float_piece_in_an_exact_one(name, ndigits, a, b):
    """
    the x50 fuzz (2026-10-03) found test_isotone claiming f(A) ⊆ f(B) in the outward class for a float piece of A
    inside an exact piece of B: `round_ties_away((0.0, 1/10), 1)` = { [0.0] , (0.09999999999999999, 0.1) } is not
    inside `round_ties_away((-inf, 1/10), 1)` = (-inf, 1/10]. the library is right and the oracle was wrong: B has
    no finite float end, so the class computes f(B) exactly, as MultiInterval does (tests/test_outward.py: "on
    exact operands it is the same as MultiInterval"), and no rounding of A's values can fit inside B's exact 1/10.
    nothing here is about hulls (rows 3 and 4 list everything), and outward `+` is the same: `O([1.0, 2]) + 1/3`
    = (1.3333333333333333, 7/3] is not inside `O([1 - 10 ** -30, 2]) + 1/3`. what does hold: f(A) is inside the
    tightest double-ended cover of f(B), and f(B) is the exact class's
    """
    a, b = parse(a), parse(b)
    _assert_isotone(name, ndigits, a, b)
    assert _call(OutwardMultiInterval, name, b, ndigits) == _call(MultiInterval, name, b, ndigits)


@pytest.mark.parametrize('name', STEPS)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_union_distributes(name, data):
    """
    f(A ∪ B) = f(A) ∪ f(B) where nothing is hulled, as for any pointwise image: exact operands in the exact class,
    and operands whose every end is a double or ±inf (so that a merged piece is a float piece too) in both classes
    """
    ndigits = _ndigits(data, name)
    exact_a, exact_b = data.draw(exact_cut_tuples, label='exact a'), data.draw(exact_cut_tuples, label='exact b')
    float_a = data.draw(float_operands(ndigits, floats_only=True), label='float a')
    float_b = data.draw(float_operands(ndigits, floats_only=True), label='float b')
    for cls, a, b in ((MultiInterval, exact_a, exact_b), (MultiInterval, float_a, float_b),
                      (OutwardMultiInterval, float_a, float_b)):
        if not a or not b:
            continue
        fa, ha = _call(cls, name, a, ndigits)
        fb, hb = _call(cls, name, b, ndigits)
        fab, hab = _call(cls, name, union(a, b), ndigits)
        assert ha or hb or hab or fab == union(fa, fb), (cls.__name__, show(a), show(b), show(fab), show(union(fa, fb)))


@pytest.mark.parametrize('name', STEPS)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_idempotent(name, data):
    """
    f(f(A)) = f(A) where f(A) is listed, its values being grid points: in the exact class (ndigits too), in
    MultiInterval on float operands on the integer grid (a rounded integer is an integer), and in the outward class
    where every value is a double; outward otherwise f(A) ⊆ f(f(A)), the enclosures only growing
    """
    ndigits = _ndigits(data, name)
    exact = _exact_cuts(data.draw(float_operands(ndigits).filter(bool), label='exact a'))
    floats = data.draw(float_operands().filter(bool), label='float a')
    for cls, a, nd in ((MultiInterval, exact, ndigits), (MultiInterval, floats, None),
                       (OutwardMultiInterval, floats, ndigits)):
        fa, hulled = _call(cls, name, a, nd)
        if hulled:
            continue
        ffa, again = _call(cls, name, fa, nd)
        listed = _image(name, a, nd, cls is OutwardMultiInterval)
        if cls is OutwardMultiInterval and (listed is None or listed[1]):
            assert again or is_subset(fa, ffa), (show(a), show(fa), show(ffa))
        else:
            assert not again and ffa == fa, (cls.__name__, show(a), show(fa), show(ffa))
