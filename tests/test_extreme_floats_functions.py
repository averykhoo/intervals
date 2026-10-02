"""
the elementary functions at the edges of the float range, against arb

`tests/test_extreme_floats.py` fuzzes the arithmetic on operands from the extremes of the float range;
`tests/test_functions.py` checks the functions' set logic on operands in [-20, 20] against
`intervals.elementary`, and `tests/test_oracle_flint.py` checks `elementary` against arb one point at a
time. here every function of the class (the thirty of `functions.NAMES`, `log` to a base, `rootn`, and
the two-argument `atan2`, `hypot` and `**`) gets multi-interval operands drawn from the pools of
`tests/test_extreme_floats.py` (subnormals, 1e308 and the max float, `ldexp` across the exponent range,
±inf, ints and Fractions), with each function's own edges mixed in: its domain's ends and the doubles
beside them, the arguments where exp, sinh or tanh leave the float range, the doubles around pi/2 and
pi, an argument hard to reduce mod pi/2, and pieces a few ulps wide. an operand is all floats, mixed,
or all exact (the same pools, each float as the Fraction it denotes).

every value comes from arb (python-flint) as a decided comparison, never from the library: `sign(q)`
is the sign of `q - f(x)`, retried at a higher precision while a ball is too wide (a point no
precision settles is skipped, with an `event`). `elementary.exact` only says where a value is
rational, and arb's ball must meet that rational. at ±inf and at a domain's end the value is the
limit the plan's table names (`exp(-inf)` = 0, `atan(inf)` = pi/2, `log(0)` = -inf, `atanh(1)` = inf).

* soundness: outward, the value at every sampled point of the operand is in the result, flags read
  as given; to nearest the same for an exact operand (an exact operand never loses its true value, in
  either class), and for a float operand the nearest double to each value is in the result's closure
  (a flag at a rounded end is conservative there, not a promise: v2-plan "arithmetic", rounding)
* sharpness, on one piece where the function is monotone (no extremum or pole strictly inside, the
  quadrant decided by arb): the result is one piece between the values at the piece's two ends, each
  end the tightest bound outside its value, closed iff the piece's end is closed and the value is that
  number (outward, and at an exact or infinite end in both classes); to nearest a float end is the
  nearest double to its value. a pole strictly inside a piece (at 0 for coth, csch and an odd negative
  root; tan's, cot's, sec's or csc's with no other multiple of pi/2 inside) gives two pieces, each from
  an end's value to the infinity on its side, both infinities closed; a pole's point 0 alone gives nothing
"""
import math
import warnings
from fractions import Fraction

import pytest
from flint import arb
from flint import ctx
from flint import fmpq
from hypothesis import event
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.control import currently_in_test_context

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import elementary
from intervals.functions import NAMES
from intervals.kernel import intersection
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from intervals.rounding import DOWN
from intervals.rounding import UP
from tests.oracles import _exact
from tests.oracles import sample
from tests.test_extreme_floats import MAX
from tests.test_extreme_floats import _float_samples
from tests.test_extreme_floats import _value
from tests.test_oracle_flint import PRECISIONS
from tests.test_oracle_flint import _ARB
from tests.test_oracle_flint import _Undecided
from tests.test_oracle_flint import _arb
from tests.test_oracle_flint import _ball_sign
from tests.test_oracle_flint import _bits
from tests.test_oracle_flint import _midpoint
from tests.test_oracle_flint import _oracle
from tests.test_oracle_flint import _pow_sign
from tests.test_oracle_flint import _root_ball

INF = math.inf
TRIG = ('sin', 'cos', 'tan', 'cot', 'sec', 'csc')
RISING = ('sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'asin', 'atan', 'sinh', 'tanh', 'asinh',
          'acosh', 'atanh', 'expm1', 'log1p', 'cbrt')
FALLING = ('acos', 'acot', 'acoth', 'coth', 'csch')  # coth and csch on each side of their pole
# the quadrants k (mod 4), [k pi/2, (k + 1) pi/2], where each trigonometric function rises
RISING_QUADRANTS = {'sin': (0, 3), 'cos': (2, 3), 'tan': (0, 1, 2, 3), 'cot': (), 'sec': (0, 1), 'csc': (1, 2)}
DEGREES = [-4, -3, -2, -1, 1, 2, 3, 5, 1000, -1001]
BASES = [2, 10, 3, Fraction(1, 2), Fraction(7, 3), 0.1, 1e-300, 1e300, 1 + 2 ** -52, 5e-324, MAX]
CASES = list(NAMES) + ['rootn', 'log-base']
MODES = ('float', 'float', 'mixed', 'exact')


# THE DOMAINS AND THE LIMITS (v2-plan "elementary and step functions")

_ALL = normalize([piece(-INF, INF)])
_FINITE = normalize([piece(-INF, INF, False, False)])
_FROM_ZERO = normalize([piece(0, INF)])
_UNIT = normalize([piece(-1, 1)])
DOMAINS = {'sqrt': _FROM_ZERO, 'log': _FROM_ZERO, 'log2': _FROM_ZERO, 'log10': _FROM_ZERO,
           'log1p': normalize([piece(-1, INF)]), 'asin': _UNIT, 'acos': _UNIT, 'atanh': _UNIT,
           'acosh': normalize([piece(1, INF)]), 'acoth': normalize([piece(-INF, -1), piece(1, INF)]),
           **{name: _FINITE for name in TRIG}}

# f(-inf) and f(inf): a number, an irrational multiple of pi by name, or None where f has no value
LIMITS = {'sqrt': (None, INF), 'exp': (0, INF), 'exp2': (0, INF), 'exp10': (0, INF), 'log': (None, INF),
          'log2': (None, INF), 'log10': (None, INF), 'asin': (None, None), 'acos': (None, None),
          'atan': ('-pi/2', 'pi/2'), 'sinh': (-INF, INF), 'cosh': (INF, INF), 'tanh': (-1, 1),
          'asinh': (-INF, INF), 'acosh': (None, INF), 'atanh': (None, None), 'expm1': (-1, INF),
          'log1p': (None, INF), 'cbrt': (-INF, INF), 'acot': ('pi', 0), 'coth': (-1, 1), 'csch': (0, 0),
          'sech': (0, 0), 'acoth': (0, 0), **{name: (None, None) for name in TRIG}}

# a domain's finite end where f has an infinite limit from inside, a point of the domain
INFINITE_ENDS = {('log', 0): -INF, ('log2', 0): -INF, ('log10', 0): -INF, ('log1p', -1): -INF,
                 ('atanh', -1): -INF, ('atanh', 1): INF, ('acoth', -1): -INF, ('acoth', 1): INF}


def _domain(name, base):
    if name == 'rootn':
        return _ALL if base % 2 else _FROM_ZERO
    return DOMAINS.get(name, _ALL)


def _limits(name, base):
    if name == 'rootn':
        if base > 0:
            return (-INF if base % 2 else None), INF
        return (0 if base % 2 else None), 0
    if name == 'log' and base is not None and base < 1:
        return None, -INF
    return LIMITS[name]


def _pole(name, base) -> bool:
    """a pole at 0 with a side each way: only a one-sided limit there, and no value"""
    return name in ('cot', 'csc', 'coth', 'csch') or (name == 'rootn' and base < 0 and base % 2 == 1)


# THE VALUES, FROM ARB

def _exact_sign(r):
    """`sign(q)` for an exact value r: a rational or ±inf"""
    def sign(q) -> int:
        if q == r:
            return 0
        if r in (INF, -INF) or q in (INF, -INF):
            return 1 if q > r else -1
        return (Fraction(q) > r) - (Fraction(q) < r)
    return sign


_PI = {'pi/2': lambda: arb.pi() / 2, '-pi/2': lambda: -arb.pi() / 2, 'pi': lambda: arb.pi(), '-pi': lambda: -arb.pi()}


def _constant(v):
    return _ball_sign(_PI[v]()) if isinstance(v, str) else _exact_sign(v)


def _ball(name, x, base) -> arb:
    if name == 'rootn':
        return _root_ball(x, base)
    if name == 'log' and base is not None:
        return _arb(x).log() / _arb(base).log()
    return _ARB[name](_arb(x))


def _rational(r, ball: arb, where: str):
    """the rational `elementary.exact` named, once arb's ball is seen to meet it"""
    r = Fraction(r)
    assert not ball.is_finite() or (ball * r.denominator).overlaps(arb(r.numerator)), f'{where} is not the rational {r}'
    return _exact_sign(r)


def _sign_of(name, base, x, side: int = 0):
    """
    `sign(q)` = the sign of q - f(x) at the working precision, x a point of f's domain (a float, int,
    Fraction or ±inf); None where f has no value. `side` is the side a piece meets a pole at 0 from (1
    above, -1 below): its one-sided limit there, ±inf
    """
    if x in (INF, -INF):
        v = _limits(name, base)[x > 0]
        return None if v is None else _constant(v)
    if x == 0 and _pole(name, base):
        return _exact_sign(side * INF) if side else None
    if name == 'rootn' and base < 0 and x == 0:  # an even negative root: the domain's end
        return _exact_sign(INF)
    if (name, x) in INFINITE_ENDS:
        v = INFINITE_ENDS[name, x]
        return _exact_sign(-v if name == 'log' and base is not None and base < 1 else v)
    r = elementary.exact(name, Fraction(x), base)
    if r is not None:
        return _rational(r, _ball(name, x, base), f'{name}({x!r})')
    if name == 'rootn':
        return _ball_sign(_root_ball(x, base))
    return _oracle(name, x, base)


def _sign_of_pair(name, v, u):
    """`sign(q)` of q - f(v, u) for atan2(v, u), hypot(v, u) and pow (v ** u); None where it has no value"""
    inf_v, inf_u = v in (INF, -INF), u in (INF, -INF)
    if name == 'atan2':
        if (v == 0 and u == 0) or (inf_v and inf_u):
            return None
        if inf_v:
            return _constant('pi/2' if v > 0 else '-pi/2')
        if u == INF:
            return _exact_sign(0)
        if u == -INF:
            return _constant('pi' if v >= 0 else '-pi')
        if v == 0:  # no -0: the negative axis is at pi
            return _exact_sign(0) if u > 0 else _constant('pi')
        return _ball_sign(arb.atan2(_arb(v), _arb(u)))
    if name == 'hypot':
        if inf_v or inf_u:
            return _exact_sign(INF)
        s = Fraction(v) ** 2 + Fraction(u) ** 2
        root = math.isqrt(s.numerator), math.isqrt(s.denominator)
        if root[0] ** 2 == s.numerator and root[1] ** 2 == s.denominator:
            return _exact_sign(Fraction(*root))
        return _ball_sign(arb(fmpq(s.numerator, s.denominator)).sqrt())
    # pow, 1788's: x > 0, and 0 ** y = 0 for y > 0
    x, y = v, u
    if x < 0 or (x == 0 and y <= 0):
        return None
    if x == 0:
        return _exact_sign(0)
    if x == 1:
        return None if inf_u else _exact_sign(1)
    if y == 0:
        return None if x == INF else _exact_sign(1)
    if x == INF:
        return _exact_sign(INF if y > 0 else 0)
    if inf_u:
        return _exact_sign(INF if (y > 0) == (x > 1) else 0)
    r = elementary.exact_pow(Fraction(x), Fraction(y))
    if r is not None:
        if r > 0 and abs(y) < 10 ** 6:
            assert (_arb(y) * _arb(x).log()).overlaps(_arb(r).log()), f'pow({x!r}, {y!r}) is not the rational {r}'
        return _exact_sign(Fraction(r))
    return _pow_sign(x, y)


def _note(what: str):
    """a hypothesis `event`, and nothing in a plain test"""
    if currently_in_test_context():
        event(what)


def _settle(run, *xs) -> bool:
    """`run()` at rising precision until no comparison in it is undecided; False (an event) if none settles"""
    extra = max([_bits(x) for x in xs if x not in (INF, -INF)], default=0)
    for prec in PRECISIONS:
        with ctx.workprec(prec + extra):
            try:
                run()
                return True
            except _Undecided:
                continue
    _note(f'undecided at {PRECISIONS[-1]} bits')
    return False


# THE OPERANDS

def _around(*xs):
    """each value with the doubles on either side of it"""
    return [y for x in xs for y in (math.nextafter(x, -INF), x, math.nextafter(x, INF))]


LN2 = math.log(2)
UNIT_EDGES = _around(1.0, -1.0, 0.0)
EXP_EDGES = _around(math.log(MAX), -1075 * LN2, -1022 * LN2)
TRIG_EDGES = _around(math.pi / 2, -math.pi / 2, math.pi, -math.pi, 2 * math.pi) + [
    math.ldexp(6381956970095103, 797), -math.ldexp(6381956970095103, 797),  # within 2**-60 of a multiple of pi/2
    1e22, 2.0 ** 1023, 0.0]
EDGES = {
    'exp': EXP_EDGES, 'expm1': EXP_EDGES + _around(-54 * LN2),
    'exp2': _around(1024.0, -1022.0, -1074.0, -1075.0), 'exp10': _around(math.log10(MAX), -1075 * math.log10(2)),
    'sinh': _around(math.log(MAX) + LN2, -math.log(MAX) - LN2), 'cosh': _around(math.log(MAX) + LN2, 0.0),
    # past 55 ln2 / 2 tanh rounds to nearest onto ±1, past 27 ln2 coth does; between there and 20 they are computed
    'tanh': _around(55 * LN2 / 2, -55 * LN2 / 2, 19.0, 20.0, 0.0),
    'coth': _around(27 * LN2, -27 * LN2, 19.0, 20.0, 0.0),
    'csch': _around(1076 * LN2, -1076 * LN2, 0.0), 'sech': _around(1076 * LN2, math.log(MAX) + LN2, 0.0),
    'atan': _around(2.0 ** 53, -2.0 ** 53, 0.0), 'acot': _around(2.0 ** 53, -2.0 ** 53, 0.0),
    'log1p': _around(-1.0, 0.0), **{name: TRIG_EDGES for name in TRIG},
    'hypot': _around(math.sqrt(MAX), math.sqrt(2.2250738585072014e-308), 0.0) + [MAX, 5e-324],
    'atan2': UNIT_EDGES + [MAX, -MAX],
    'pow': _around(1.0, 0.0, 2.0) + [0.5, 10.0, 1074.0, -1074.0, 1e19, -1e19, 0.5, -0.5, 2.0 ** -1074],
}


def _number(rng, mode, edges):
    """an end: the pools of tests/test_extreme_floats.py or one of the function's edges"""
    v = rng.choice(edges) if rng.random() < 0.3 else _value(rng, mode == 'float')
    return _exact(v) if mode == 'exact' else v


def _beside(x, rng, mode):
    """a point a few ulps above x, or a small relative step above it"""
    y = float(x)
    if rng.random() < 0.5 or not math.isfinite(y) or y == 0:
        for _ in range(rng.randrange(1, 4)):
            y = math.nextafter(y, INF)
    else:
        y = y + abs(y) * 2.0 ** -rng.randrange(1, 60)
    return _exact(y) if mode == 'exact' else y


def _piece(rng, mode, edges, narrow=0.35):
    """a point, a piece `narrow` of the time a few ulps (or a small relative step) wide, else two ends"""
    lo = _number(rng, mode, edges)
    r = rng.random()
    if r < 0.15:
        return piece(lo, lo)
    hi = _beside(lo, rng, mode) if r < 0.15 + narrow else _number(rng, mode, edges)
    lo, hi = sorted((lo, hi))
    if lo == hi:
        return piece(lo, lo)
    return piece(lo, hi, rng.random() < 0.6, rng.random() < 0.6)


def _operand(rng, edges, max_pieces=3):
    mode = rng.choice(MODES)
    return normalize([_piece(rng, mode, edges) for _ in range(rng.randrange(1, max_pieces + 1))])


def _samples(cuts, rng) -> list:
    """`_float_samples` piece by piece: on an exact piece wider than the float range (`[-Fraction(MAX),
    Fraction(MAX)]`) its spread point overflows, and `oracles.sample`'s exact points stand in for it"""
    out = []
    for p in pieces(cuts):
        one = normalize([piece(p[0], p[2], p[1], p[3])])
        try:
            out += _float_samples(one, rng)
        except OverflowError:
            out += [end for end, closed in ((p[0], p[1]), (p[2], p[3])) if closed] + sample(one, 6, rng)
    return out


def _function(case, rng):
    if case == 'rootn':
        return 'rootn', rng.choice(DEGREES)
    if case == 'log-base':
        return 'log', rng.choice(BASES)
    return case, None


def _apply(cls, name, base, cuts):
    m = cls.from_cuts(cuts)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if name == 'rootn':
            return m.rootn(base)
        if base is not None:
            return m.log(base)
        return getattr(m, name)()


def _finite_float(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _kinds(cuts):
    """(every finite end a float, no end a finite float)"""
    finite = [c.value for c in cuts if c.value not in (INF, -INF)]
    return all(isinstance(v, float) for v in finite), not any(isinstance(v, float) for v in finite)


# THE CHECKS

def _holds(result, sign) -> bool:
    """the value is in a piece of the result, each end's flag read as given"""
    for lo, lo_closed, hi, hi_closed in pieces(result.cuts):
        s_lo, s_hi = sign(lo), sign(hi)
        if (s_lo <= 0 if lo_closed else s_lo < 0) and (s_hi >= 0 if hi_closed else s_hi > 0):
            return True
    return False


def _next_up(v):
    return math.nextafter(v, INF)


def _next_down(v):
    return math.nextafter(v, -INF)


def _rounds_onto(sign, d) -> bool:
    """the nearest double to the value is d (a tie may go either way): the value lies between d's midpoints"""
    above_low = d == -INF or sign(_midpoint(_next_down(d), d)) <= 0
    below_high = d == INF or sign(_midpoint(d, _next_up(d))) >= 0
    return above_low and below_high


def _nearest_holds(result, sign) -> bool:
    """the nearest double to the value is in the closure of a piece of the result (its ends are doubles)"""
    for lo, _, hi, _ in pieces(result.cuts):
        above_lo = lo == -INF or sign(_midpoint(_next_down(lo), lo)) <= 0
        if above_lo and (hi == INF or sign(_midpoint(hi, _next_up(hi))) >= 0):
            return True
    return False


def _check_end(sign, end, closed, x, x_closed, want, nearest: bool, flags: bool, where: str):
    """
    one end of a monotone piece's image, from the operand end x: to nearest at a finite float x the
    nearest double to f(x); otherwise the tightest bound outside it (`want` DOWN for a low end), closed
    iff x is closed and the end is f(x)
    """
    if nearest and _finite_float(x):
        assert _rounds_onto(sign, end), f'{where}: {end!r} is not the nearest double to f({x!r})'
        return
    s = sign(end)
    if want == DOWN:
        assert s <= 0, f'{where}: the low end {end!r} is above f({x!r})'
        assert s == 0 or sign(_next_up(end)) > 0, f'{where}: not sharp, a double above {end!r} is still below f({x!r})'
    else:
        assert s >= 0, f'{where}: the high end {end!r} is below f({x!r})'
        assert s == 0 or sign(_next_down(end)) < 0, f'{where}: not sharp, a double below {end!r} is still above f({x!r})'
    if flags:
        assert closed == (x_closed and s == 0), f'{where}: the end {end!r} is {"closed" if closed else "open"}, x={x!r}'


def _floor_quarter_turns(x) -> int:
    """floor(2x / pi), decided by arb"""
    k = (2 * _arb(x) / arb.pi()).floor().unique_fmpz()
    if k is None:
        raise _Undecided
    return int(k)


def _quadrants(lo, hi):
    """
    `(first, last)`: the piece lies in the quadrants `[k pi/2, (k + 1) pi/2]` for k from first to last, so
    the multiples of pi/2 strictly inside are `(first + 1) pi/2` to `last pi/2`; None for an unbounded piece
    """
    if lo in (INF, -INF) or hi in (INF, -INF):
        return None
    first = 0 if lo == 0 else _floor_quarter_turns(lo)
    last = -1 if hi == 0 else _floor_quarter_turns(hi)  # 0 is the only multiple of pi/2 an end can be
    return first, last


def _trig_pole(name, m: int) -> bool:
    """m pi/2 is a pole: tan's and sec's at odd m, cot's and csc's at even m"""
    return {'tan': 1, 'sec': 1, 'cot': 0, 'csc': 0}.get(name) == m % 2


def _rising(name, base, lo, hi, quadrant) -> bool:
    """the direction on a piece with no extremum or pole strictly inside"""
    if name == 'rootn':
        return base > 0
    if name == 'log' and base is not None:
        return base > 1
    if name == 'cosh':
        return lo >= 0
    if name == 'sech':
        return hi <= 0
    if name in TRIG:
        return quadrant % 4 in RISING_QUADRANTS[name]
    assert name in RISING or name in FALLING, name
    return name in RISING


def _show(cuts):
    return list(pieces(cuts))


# THE TESTS

@pytest.mark.parametrize('case', CASES)
@settings(max_examples=50, deadline=None)
@given(rng=st.randoms(use_true_random=True))
def test_every_value_is_in_the_result(case, rng):
    """
    outward, the value at every sampled point is in the result; to nearest the same for an exact
    operand, and the nearest double to it is in the closure for a float operand
    """
    name, base = _function(case, rng)
    a = _operand(rng, EDGES.get(name, UNIT_EDGES))
    floats, exacts = _kinds(a)
    outward, nearest = _apply(OutwardMultiInterval, name, base, a), _apply(MultiInterval, name, base, a)
    for x in _samples(intersection(a, _domain(name, base)), rng):
        where = f'{name}{"" if base is None else f"[{base!r}]"} of {_show(a)} at {x!r}'

        def run():
            sign = _sign_of(name, base, x)
            if sign is None:
                return
            assert _holds(outward, sign), f'outward {where}: {_show(outward.cuts)}'
            if exacts:
                assert _holds(nearest, sign), f'nearest, exact operand, {where}: {_show(nearest.cuts)}'
            if floats:
                assert _nearest_holds(nearest, sign), f'nearest {where}: {_show(nearest.cuts)}'
        _settle(run, x)


@pytest.mark.parametrize('case', CASES)
@settings(max_examples=50, deadline=None)
@given(rng=st.randoms(use_true_random=True))
def test_ends_are_sharp_where_monotone(case, rng):
    """one piece with no extremum or pole strictly inside maps to the one piece between its ends' values"""
    name, base = _function(case, rng)
    mode, edges = rng.choice(MODES), EDGES.get(name, UNIT_EDGES)
    r = rng.random()
    if r < 0.15:  # a piece ending at 0: a pole's one-sided limit, a domain's end, cosh's minimum
        zero = 0.0 if mode == 'float' else 0
        other = _number(rng, mode, edges) if rng.random() < 0.5 else rng.choice((-1, 1)) * _beside(zero, rng, mode)
        lo, hi = sorted((zero, other))
        p = piece(lo, lo) if lo == hi else piece(lo, hi, rng.random() < 0.5, rng.random() < 0.5)
    elif r < 0.35:  # a few ulps across an edge: a pole (pi's doubles, 0), a domain's end, a threshold
        e = rng.choice(edges)
        lo, hi = -_beside(-e, rng, mode), _beside(e, rng, mode)
        p = piece(lo, hi, rng.random() < 0.5, rng.random() < 0.5)
    else:  # the trigonometric functions are monotone only between multiples of pi/2: mostly narrow pieces
        p = _piece(rng, mode, edges, 0.75 if name in TRIG else 0.35)
    _check_sharp(name, base, normalize([p]))


# what the fuzz found (2026-10-02), each an oracle bug fixed above: to nearest csch's two sides meet at 0.0,
# and a domain's end clipped onto a float end is a point with an exact cut and a float one
@pytest.mark.parametrize('name, a', [
    ('csch', normalize([piece(-1e154, 1e154)])),
    ('asin', normalize([piece(1.0, INF)])),
    ('acos', normalize([piece(-1.0000000000000002, -1.0)])),
])
def test_ends_are_sharp_where_the_fuzz_looked(name, a):
    _check_sharp(name, None, a)


def _check_sharp(name, base, a):
    results =[(_apply(cls, name, base, a), cls is MultiInterval) for cls in (OutwardMultiInterval, MultiInterval)]
    parts = list(pieces(intersection(a, _domain(name, base))))
    where = f'{name}{"" if base is None else f"[{base!r}]"} of {_show(a)}'
    if not parts or (parts[0][0] == parts[0][2] == 0 and _pole(name, base)):
        got = [_show(r.cuts) for r, _ in results]
        assert got == [[], []], f'{where}: nothing has a value, got {got}'
        return
    if len(parts) > 1:
        _note('two pieces in the domain')
        return
    [(lo, lo_closed, hi, hi_closed)] = parts

    def two_sided(below, above):
        """
        a pole strictly inside: the part reaching -inf ends at f of the operand end `below`, the part
        from +inf starts at f of `above`, both infinities attained (as `1/x` at a zero inside a divisor)
        """
        _note('a pole inside')
        s_below, s_above = _sign_of(name, base, below[0]), _sign_of(name, base, above[0])
        for result, nearest in results:
            ps = list(pieces(result.cuts))
            if nearest and ps == [(-INF, True, INF, True)]:
                # to nearest both sides may round onto 0.0, flags kept, and meet: csch(±1e154)
                assert _rounds_onto(s_below, 0.0) and _rounds_onto(s_above, 0.0), f'nearest {where}: {ps}'
                continue
            assert len(ps) == 2 and ps[0][:2] == (-INF, True) and ps[1][2:] == (INF, True), f'{where}: {ps}'
            _check_end(s_below, ps[0][2], ps[0][3], *below, UP, nearest, True, where)
            _check_end(s_above, ps[1][0], ps[1][1], *above, DOWN, nearest, True, where)

    def run():
        if lo < 0 < hi and (name in ('cosh', 'sech') or (_pole(name, base) and name not in TRIG)):
            if name in ('cosh', 'sech') or lo == -INF or hi == INF:
                _note('an extremum, or a pole with an infinite end, inside')
                return
            two_sided((lo, lo_closed), (hi, hi_closed))  # coth, csch, an odd negative root: falling each side
            return
        quadrant = None
        if name in TRIG and lo != hi:
            turns = _quadrants(lo, hi)
            if turns is not None and turns[1] == turns[0] + 1 and _trig_pole(name, turns[1]):
                # one pole inside, nothing else: rising on both sides (tan, csc at an odd multiple of pi,
                # sec at pi/2 mod 2 pi) leaves the low end's value going up to +inf and the high end's
                # coming from -inf; falling, the other way round
                rising = _rising(name, base, lo, hi, turns[0])
                assert rising == _rising(name, base, lo, hi, turns[1]), (name, turns)
                ends = (hi, hi_closed), (lo, lo_closed)
                two_sided(*(ends if rising else ends[::-1]))
                return
            if turns is None or turns[0] != turns[1]:
                _note('a multiple of pi/2 inside')
                return
            quadrant = turns[0]
        ends = [(lo, lo_closed, 1), (hi, hi_closed, -1)]
        low, high = ends if lo == hi or _rising(name, base, lo, hi, quadrant) else ends[::-1]
        s_low, s_high = _sign_of(name, base, low[0], low[2]), _sign_of(name, base, high[0], high[2])
        for result, nearest in results:
            if nearest and lo == hi and _finite_float(lo) != _finite_float(hi):
                # a domain's end clipped onto a float end (asin of [1.0, inf]) is a point with a float cut and an
                # exact one; to nearest either cut's rule is a fair reading of it
                _note('to nearest, a point with a float cut and an exact cut')
                continue
            ps = list(pieces(result.cuts))
            assert len(ps) == 1, f'{where}: {ps}, not one piece'
            r_lo, r_lo_closed, r_hi, r_hi_closed = ps[0]
            flags = not (nearest and r_lo == r_hi)  # to nearest a piece squeezed onto one double is that point, closed
            _check_end(s_low, r_lo, r_lo_closed, low[0], low[1], DOWN, nearest, flags, where)
            _check_end(s_high, r_hi, r_hi_closed, high[0], high[1], UP, nearest, flags, where)
    _settle(run, lo, hi)


def _pair_samples(cuts, rng, n=6):
    xs = _samples(cuts, rng)
    return rng.sample(xs, min(n, len(xs)))


@pytest.mark.parametrize('name', ['atan2', 'hypot', 'pow'])
@settings(max_examples=60, deadline=None)
@given(rng=st.randoms(use_true_random=True))
def test_two_argument_values_are_in_the_result(name, rng):
    """the soundness above for atan2(A, B), hypot(A, B) and A ** B (1788's pow), at sampled pairs"""
    a, b = _operand(rng, EDGES[name]), _operand(rng, EDGES[name])
    floats, exacts = (all(k) for k in zip(_kinds(a), _kinds(b)))
    results = []
    for cls in (OutwardMultiInterval, MultiInterval):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x, y = cls.from_cuts(a), cls.from_cuts(b)
            results.append(x ** y if name == 'pow' else getattr(x, name)(y))
    outward, nearest = results
    first = intersection(a, _FROM_ZERO) if name == 'pow' else a
    for v in _pair_samples(first, rng):
        for u in _pair_samples(b, rng):
            where = f'{name} of {_show(a)} and {_show(b)} at ({v!r}, {u!r})'

            def run():
                sign = _sign_of_pair(name, v, u)
                if sign is None:
                    return
                assert _holds(outward, sign), f'outward {where}: {_show(outward.cuts)}'
                if exacts:
                    assert _holds(nearest, sign), f'nearest, exact operands, {where}: {_show(nearest.cuts)}'
                if floats:
                    assert _nearest_holds(nearest, sign), f'nearest {where}: {_show(nearest.cuts)}'
            _settle(run, v, u)
