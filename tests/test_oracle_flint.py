"""
intervals.elementary and intervals.functions against an independent oracle: arb, through python-flint

arb evaluates f at a point as a ball proven to contain the true value (D14). the operand goes in
exactly where it can (a float is an exact arb) and otherwise as a ball holding it (a Fraction through
fmpq), so the ball holds f(x) either way. arb's `<` and `>` hold only when every point of both balls
agrees, so each comparison below is a proof or an "undecided", never a rounded guess; an undecided
one is retried at a higher precision, and past 4000 bits (plus the operand's size) the example is
rejected with `assume` and counted in `UNDECIDED`.

the checks, for each function at a drawn float or exact point:
* soundness: our DOWN <= f(x) <= our UP
* sharpness: no double lies strictly between an end and f(x), so each end is within one ulp outside
  the value: DOWN and UP are the correctly rounded bounds, and NEAREST is the nearer of the two
* a rational value (`sqrt(9/4)`, `log2(1/8)`, `exp10(3)`) comes back exact: `exact` returns it, arb's
  ball overlaps it at every precision tried, and the set-level result is that one closed point
* the set-level methods on a degenerate interval: an exact operand's irrational value is the open
  one-ulp piece around it, `OutwardMultiInterval` of a float encloses it the same way (closed only
  where attained), and `MultiInterval` of a float gives the nearest double
where arb cannot hold the value itself (exp(1e308) overflows its ball, 1 - tanh(1e308) underflows
it), the comparisons go through an increasing function of it that arb can hold: the log of exp,
sinh and cosh, and -log(1 - t) for tanh near 1.
"""
import math
from fractions import Fraction

import pytest
from flint import arb
from flint import ctx
from flint import fmpq
from hypothesis import assume
from hypothesis import event
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import DomainClippedWarning
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import elementary
from intervals import kernel
from intervals.rounding import DOWN
from intervals.rounding import MAX
from intervals.rounding import NEAREST
from intervals.rounding import UP

INF = math.inf
PRECISIONS = (200, 1000, 4000)
# how many drawn examples no precision could decide, by test; reported, not asserted
UNDECIDED = {}


class _Undecided(Exception):
    """a ball is too wide to settle a comparison at this precision"""


# THE ORACLE

def _arb(q) -> arb:
    """q as a ball: exact for a float or a short dyadic, otherwise holding q at the working precision"""
    if isinstance(q, float):
        return arb(q)
    q = Fraction(q)
    return arb(fmpq(q.numerator, q.denominator))


def _cmp(a: arb, b: arb) -> int:
    """the sign of a - b, when every point of the two balls agrees on it"""
    if a < b:
        return -1
    if a > b:
        return 1
    if a.is_exact() and b.is_exact() and a == b:
        return 0
    raise _Undecided


_ARB = {
    'sqrt': arb.sqrt,
    'exp': arb.exp,
    'exp2': lambda x: (x * arb.const_log2()).exp(),
    'exp10': lambda x: (x * arb(10).log()).exp(),
    'log': arb.log,
    'log2': lambda x: x.log() / arb.const_log2(),
    'log10': lambda x: x.log() / arb(10).log(),
    'sin': arb.sin,
    'cos': arb.cos,
    'tan': arb.tan,
    'asin': arb.asin,
    'acos': arb.acos,
    'atan': arb.atan,
    'sinh': arb.sinh,
    'cosh': arb.cosh,
    'tanh': arb.tanh,
    'asinh': arb.asinh,
    'acosh': arb.acosh,
    'atanh': arb.atanh,
}


def _ball_sign(ball: arb):
    """`sign(q)` = the sign of q - v for the value v in `ball`"""
    def sign(q) -> int:
        if q in (INF, -INF):
            if not ball.is_finite():
                raise _Undecided
            return 1 if q > 0 else -1
        return _cmp(_arb(q), ball)
    return sign


def _logged_sign(log_v: arb):
    """`sign(q)` for a value v > 0 known only as log v"""
    def sign(q) -> int:
        if q <= 0:
            return -1
        if q == INF:
            return 1
        return _cmp(_arb(q).log(), log_v)
    return sign


def _near_one_sign(g_v: arb):
    """`sign(q)` for a value v in (0, 1) known only as -log(1 - v), which rises with v"""
    def sign(q) -> int:
        if q >= 1:
            return 1
        if q <= 0:
            return -1
        return _cmp(-(-_arb(q)).log1p(), g_v)
    return sign


def _oracle(name: str, x, base=None):
    """`sign(q)` = the sign of q - f(x), at the working precision; x a float, int or Fraction"""
    if name in ('sinh', 'tanh') and x < 0:  # odd: sign(q - f(x)) = -sign(-q - f(-x))
        odd = _oracle(name, -x)
        return lambda q: -odd(-q)
    b = _arb(x)
    if name in ('exp', 'exp2', 'exp10') and abs(x) > 600:
        return _logged_sign(b * {'exp': arb(1), 'exp2': arb.const_log2(), 'exp10': arb(10).log()}[name])
    if name in ('sinh', 'cosh') and abs(x) > 600:
        # log(cosh x) = |x| - log 2 + log1p(exp(-2|x|)), and sinh has the minus sign (x > 0 here)
        tail = (-2 * abs(b)).exp()
        return _logged_sign(abs(b) - arb.const_log2() + (tail if name == 'cosh' else -tail).log1p())
    if name == 'tanh' and x > 20:
        # 1 - tanh x = 2 / (exp(2x) + 1), so -log(1 - tanh x) = 2x + log1p(exp(-2x)) - log 2
        return _near_one_sign(2 * b + (-2 * b).exp().log1p() - arb.const_log2())
    if name == 'log' and base is not None:
        return _ball_sign(b.log() / _arb(base).log())
    return _ball_sign(_ARB[name](b))


def _exact_sign(r: Fraction):
    def sign(q) -> int:
        if q in (INF, -INF):
            return 1 if q > 0 else -1
        return (Fraction(q) > r) - (Fraction(q) < r)
    return sign


def _bits(*xs) -> int:
    """extra precision for a large operand: arb needs it to reduce sin(10**40 / 3) mod pi"""
    return max(max(0, Fraction(x).numerator.bit_length() - Fraction(x).denominator.bit_length()) for x in xs)


def _decided(test: str, extra: int, run):
    """`run()` at rising precision until no comparison in it is undecided"""
    for prec in PRECISIONS:
        with ctx.workprec(prec + extra):
            try:
                return run()
            except _Undecided:
                continue
    UNDECIDED[test] = UNDECIDED.get(test, 0) + 1
    event(f'undecided at {PRECISIONS[-1]} bits')
    assume(False)


# THE CHECKS

def _next_up(v: float) -> float:
    return math.nextafter(v, INF)


def _next_down(v: float) -> float:
    return math.nextafter(v, -INF)


def _midpoint(down: float, up: float) -> Fraction:
    """the point where rounding to nearest switches from down to up (past max float: to inf)"""
    if up == INF:
        return Fraction(MAX) + Fraction(math.ulp(MAX)) / 2
    if down == -INF:
        return -Fraction(MAX) - Fraction(math.ulp(MAX)) / 2
    return (Fraction(down) + Fraction(up)) / 2


def _check_rounding(sign, down: float, near: float, up: float, where: str):
    s_down, s_up = sign(down), sign(up)
    assert s_down <= 0 <= s_up, f'unsound: {where} is not inside [{down!r}, {up!r}]'
    if s_down < 0:
        assert sign(_next_up(down)) > 0, f'not sharp: a double above {down!r} is still below {where}'
    if s_up > 0:
        assert sign(_next_down(up)) < 0, f'not sharp: a double below {up!r} is still above {where}'
    if s_down == 0:
        assert near == down == up, f'{where} is the double {down!r}, but nearest gave {near!r}'
        return
    s_mid = sign(_midpoint(down, up))
    if s_mid == 0:  # a rational value on a tie goes to the even double
        expected = down if math.frexp(down)[0] * 2 ** 53 % 2 == 0 else up
    else:
        expected = up if s_mid < 0 else down
    assert near == expected, f'nearest of {where} is {expected!r}, got {near!r}'


def _check_piece(sign, piece, where: str):
    """one set-level piece encloses the value, closed only where it is the value, no double to spare"""
    lo, lo_closed, hi, hi_closed = piece
    s_lo, s_hi = sign(lo), sign(hi)
    assert s_lo <= 0 <= s_hi, f'unsound: {where} is not inside {piece}'
    assert not lo_closed or s_lo == 0, f'{piece} is closed at {lo!r}, which is not {where}'
    assert not hi_closed or s_hi == 0, f'{piece} is closed at {hi!r}, which is not {where}'
    if s_lo < 0:
        assert sign(_next_up(lo)) > 0, f'not sharp: {piece} could start one double higher'
    if s_hi > 0:
        assert sign(_next_down(hi)) < 0, f'not sharp: {piece} could end one double lower'


def _pieces(m: MultiInterval):
    return list(kernel.pieces(m.cuts))


def _method(m: MultiInterval, name: str, base=None) -> MultiInterval:
    return m.log(base) if base is not None else getattr(m, name)()


def _check_point(test: str, name: str, x, base=None):
    """every check above for f at one finite point x inside f's domain"""
    exact_x = Fraction(x)
    where = f'{name}({x!r}' + (f', base={base!r})' if base is not None else ')')
    r = elementary.exact(name, exact_x, base)
    assert r not in (INF, -INF), f'{where} = {r!r} inside the domain'
    down, near, up = (elementary.rounded(name, exact_x, d, base) for d in (DOWN, NEAREST, UP))
    exact_result = _method(MultiInterval(exact_x), name, base)
    outward = _method(OutwardMultiInterval(x), name, base) if isinstance(x, float) else None
    nearest = _method(MultiInterval(x), name, base) if isinstance(x, float) else None

    def run():
        ball_sign = _oracle(name, x, base)
        if r is not None:
            # arb's ball must meet the claimed rational at every precision: if it missed it, f(x) != r
            assert _overlaps(name, x, base, r), f'{where} is not the rational {r!r}'
            sign = _exact_sign(Fraction(r))
        else:
            sign = ball_sign
            ball = _ARB[name](_arb(x)) if base is None else None
            assert ball is None or not ball.is_exact(), f'{where} is the dyadic {ball}, but exact() missed it'
        _check_rounding(sign, down, near, up, where)
        pieces = _pieces(exact_result)
        if r is not None:
            assert pieces == [(r, True, r, True)] and not isinstance(r, float), \
                f'{where} over a set is {pieces}, not the exact [{r}]'
        else:
            assert len(pieces) == 1 and not pieces[0][1] and not pieces[0][3], f'{where}: {pieces}'
            _check_piece(sign, pieces[0], where)
        if outward is not None:
            pieces = _pieces(outward)
            assert len(pieces) == 1, f'{where} outward: {pieces}'
            _check_piece(sign, pieces[0], where + ' outward')
            assert _pieces(nearest) == [(near, True, near, True)], f'{where} to nearest: {_pieces(nearest)}'
    _decided(test, _bits(x), run)


def _overlaps(name: str, x, base, r) -> bool:
    b = _arb(x)
    v = b.log() / _arb(base).log() if base is not None else _ARB[name](b)
    r = Fraction(r)
    return (v * r.denominator).overlaps(arb(r.numerator)) if v.is_finite() else True


# THE POINTS

def _domain(name: str):
    """(lowest, highest, whether the ends are excluded) of the finite points drawn for f"""
    if name == 'sqrt':
        return 0, None, False
    if name in ('log', 'log2', 'log10'):
        return 0, None, True
    if name in ('asin', 'acos'):
        return -1, 1, False
    if name == 'atanh':
        return -1, 1, True
    if name == 'acosh':
        return 1, None, False
    return None, None, False


def _inside(name: str, x) -> bool:
    lo, hi, open_ends = _domain(name)
    if open_ends:
        return (lo is None or x > lo) and (hi is None or x < hi)
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def _rational_points(name: str):
    """points where f is rational, so the exact path is drawn often"""
    if name == 'sqrt':
        return st.fractions(min_value=0, max_denominator=10 ** 6).map(lambda q: q * q)
    if name in ('exp2', 'exp10'):
        return st.integers(-400, 400)
    if name in ('log2', 'log10'):
        b = 2 if name == 'log2' else 10
        return st.integers(-400, 400).map(lambda k: Fraction(b) ** k)
    return st.sampled_from([x for x in (-1, 0, 1) if _inside(name, x) and elementary.exact(name, x) is not None])


def points(name: str):
    lo, hi, open_ends = _domain(name)
    floats = st.floats(min_value=lo, max_value=hi, exclude_min=open_ends and lo is not None,
                       exclude_max=open_ends and hi is not None, allow_nan=False, allow_infinity=False)
    step = 1 if open_ends else 0
    ints = st.integers(-10 ** 30 if lo is None else lo + step, 10 ** 30 if hi is None else hi - step)
    exact = st.one_of(ints, st.fractions(min_value=lo, max_value=hi, max_denominator=10 ** 12))
    return st.one_of(floats, exact.filter(lambda x: _inside(name, x)), _rational_points(name))


NAMES = elementary.NAMES


# THE TESTS

@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_point_against_arb(name, data):
    x = data.draw(points(name), label='x')
    event('rational value' if elementary.exact(name, Fraction(x)) is not None else 'irrational value')
    _check_point('test_point_against_arb', name, x)


# the extremes: past the float range, subnormal, a huge argument to reduce, next to a pole or a domain end
EXTREMES = [
    ('sin', 1e22), ('cos', 1e22), ('tan', 1e22), ('sin', MAX), ('cos', -MAX), ('tan', MAX),
    ('tan', math.pi / 2), ('tan', -math.pi / 2), ('cos', math.pi / 2), ('sin', math.pi),
    ('sin', Fraction(10 ** 40 + 1, 3)), ('tan', 5e-324), ('sin', 5e-324), ('atan', MAX),
    ('exp', 709.782712893384), ('exp', 709.7827128933841), ('exp', -745.1332191019411), ('exp', -745.1332191019412),
    ('exp', MAX), ('exp', -MAX), ('exp2', 1023.9999999999999), ('exp2', -1074.5), ('exp10', 308.2547155599167),
    ('exp10', -323.5), ('log', 5e-324), ('log', MAX), ('log2', 5e-324), ('log10', MAX), ('log', 1 + 2 ** -52),
    ('log', 1 - 2 ** -53), ('sqrt', 5e-324), ('sqrt', MAX), ('sqrt', 2), ('sinh', 710.5), ('sinh', -MAX),
    ('cosh', 710.5), ('cosh', -MAX), ('sinh', 5e-324), ('tanh', 19.0625), ('tanh', 20), ('tanh', -MAX),
    ('tanh', 5e-324), ('asinh', MAX), ('asinh', -5e-324), ('acosh', 1 + 2 ** -52), ('acosh', MAX),
    ('atanh', 1 - 2 ** -53), ('atanh', -1 + 2 ** -53), ('atanh', 5e-324), ('asin', 1 - 2 ** -53),
    ('asin', 1.0), ('asin', -1), ('acos', -1.0), ('acos', 1 - 2 ** -53), ('acos', -1 + 2 ** -53),
    ('exp10', 22), ('exp10', -3), ('log10', Fraction(1, 1000)), ('log2', 2.0 ** -1074), ('sqrt', Fraction(9, 4)),
]


@pytest.mark.parametrize('name, x', EXTREMES)
def test_extreme_point_against_arb(name, x):
    _check_point('test_extreme_point_against_arb', name, x)


BASES = [2, 10, 3, Fraction(1, 2), Fraction(7, 3), 0.1, 1e-300, 1e300, 1 + 2 ** -52]


@settings(max_examples=60, deadline=None)
@given(x=points('log'), base=st.one_of(st.sampled_from(BASES), st.floats(min_value=5e-324, allow_infinity=False).filter(lambda b: b != 1)))
@example(x=8, base=2)
@example(x=Fraction(1, 9), base=Fraction(1, 3))
@example(x=1000.0, base=10)
@example(x=0.001, base=0.1)
def test_log_base_against_arb(x, base):
    _check_point('test_log_base_against_arb', 'log', x, base)


@settings(max_examples=40, deadline=None)
@given(k=st.integers(-400, 400), base=st.sampled_from([2, 3, 10, Fraction(1, 2), Fraction(2, 3)]))
def test_log_of_a_power_is_exact(k, base):
    _check_point('test_log_of_a_power_is_exact', 'log', Fraction(base) ** k, base)


# THE DOMAIN'S ENDS, AND ±INF

def _pi_over_2(sign: int):
    return lambda q: _cmp(_arb(q), sign * arb.pi() / 2)


# (name, x, base, the value: exact, or a sign function for an irrational one)
ENDS = [
    ('sqrt', 0, None, 0), ('sqrt', INF, None, INF), ('exp', INF, None, INF), ('exp', -INF, None, 0),
    ('exp2', -INF, None, 0), ('exp10', INF, None, INF), ('log', 0, None, -INF), ('log', INF, None, INF),
    ('log', 0, Fraction(1, 2), INF), ('log', INF, Fraction(1, 2), -INF), ('log2', 0, None, -INF),
    ('log10', 0, None, -INF), ('atanh', 1, None, INF), ('atanh', -1, None, -INF), ('acosh', 1, None, 0),
    ('acos', 1, None, 0), ('sinh', -INF, None, -INF), ('cosh', -INF, None, INF), ('tanh', INF, None, 1),
    ('tanh', -INF, None, -1), ('asinh', -INF, None, -INF), ('acosh', INF, None, INF),
    ('atan', INF, None, 'pi/2'), ('atan', -INF, None, '-pi/2'),
]


@pytest.mark.parametrize('name, x, base, value', ENDS)
def test_domain_ends_and_limits(name, x, base, value):
    m = _method(MultiInterval(x), name, base)
    if isinstance(value, str):  # atan(±inf) = ±pi/2, irrational: the open one-ulp piece, checked by arb
        s = -1 if value.startswith('-') else 1
        [piece] = _pieces(m)
        down, near, up = (elementary.rounded(name, x, d) for d in (DOWN, NEAREST, UP))
        with ctx.workprec(200):
            _check_rounding(_pi_over_2(s), down, near, up, f'{name}({x})')
            _check_piece(_pi_over_2(s), piece, f'{name}({x})')
        return
    assert elementary.exact(name, x, base) == value
    assert _pieces(m) == [(value, True, value, True)]


# points outside the domain, and sin, cos, tan at ±inf, where there is no limit: dropped, with a warning
OUTSIDE = [('sqrt', -1), ('sqrt', -5e-324), ('log', -1), ('log2', -INF), ('log10', Fraction(-1, 3)),
           ('asin', 1 + 2 ** -52), ('acos', -2), ('atanh', 1.5), ('atanh', -INF), ('acosh', 1 - 2 ** -53),
           ('acosh', -INF), ('sin', INF), ('cos', -INF), ('tan', INF)]


@pytest.mark.parametrize('name, x', OUTSIDE)
def test_outside_the_domain(name, x):
    with pytest.warns(DomainClippedWarning):
        assert _pieces(getattr(MultiInterval(x), name)()) == []


# ATAN2, ITS ANGLES, AND FLOOR(X / PI)

def _operand():
    return st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers(-10 ** 20, 10 ** 20),
                     st.fractions(max_denominator=10 ** 9), st.just(0))


@settings(max_examples=100, deadline=None)
@given(y=_operand(), x=_operand())
@example(y=0, x=-1)
@example(y=5e-324, x=-MAX)
@example(y=-5e-324, x=-1.0)
@example(y=1, x=0)
@example(y=-MAX, x=0.0)
@example(y=0, x=3)
def test_atan2_against_arb(y, x):
    assume(y != 0 or x != 0)
    where = f'atan2({y!r}, {x!r})'
    exact_result = MultiInterval(Fraction(y)).atan2(MultiInterval(Fraction(x)))
    floats = isinstance(y, float) and isinstance(x, float)
    outward = OutwardMultiInterval(y).atan2(OutwardMultiInterval(x)) if floats else None

    def run():
        if y == 0 and x > 0:  # the one rational angle
            assert _pieces(exact_result) == [(0, True, 0, True)]
            return
        sign = _ball_sign(arb.atan2(_arb(y), _arb(x)))
        [piece] = _pieces(exact_result)
        assert not piece[1] and not piece[3], f'{where}: {piece}'
        _check_piece(sign, piece, where)
        if outward is not None:
            [piece] = _pieces(outward)
            _check_piece(sign, piece, where + ' outward')
    _decided('test_atan2_against_arb', _bits(y, x), run)


@settings(max_examples=60, deadline=None)
@given(q=st.one_of(st.fractions(max_denominator=10 ** 12), st.floats(allow_nan=False, allow_infinity=False)),
       m=st.integers(-4, 4))
def test_rounded_angle_against_arb(q, m):
    assume(q != 0 or m != 0)
    down, near, up = (elementary.rounded_angle(Fraction(q), m, d) for d in (DOWN, NEAREST, UP))

    def run():
        sign = _ball_sign(_arb(q).atan() + m * arb.pi() / 2)
        _check_rounding(sign, down, near, up, f'atan({q!r}) + {m} pi/2')
    _decided('test_rounded_angle_against_arb', _bits(q), run)


@settings(max_examples=60, deadline=None)
@given(x=st.one_of(st.fractions(max_denominator=10 ** 12), st.floats(allow_nan=False, allow_infinity=False)),
       offset=st.sampled_from([Fraction(0), Fraction(1, 2), Fraction(-1, 2)]))
def test_floor_over_pi_against_arb(x, offset):
    k, on_it = elementary.floor_over_pi(Fraction(x), offset)

    def run():
        v = _arb(x) / arb.pi() - _arb(offset)
        if x == 0:
            assert (k, on_it) == (math.floor(-offset), (-offset).denominator == 1)
            return
        assert not on_it
        assert _cmp(arb(k), v) < 0 < _cmp(arb(k + 1), v), f'floor({x!r} / pi - {offset}) is not {k}'
    _decided('test_floor_over_pi_against_arb', _bits(x), run)
