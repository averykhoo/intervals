"""
ieee 1788's decoration propagation on `DecoratedInterval` (M13g part 3; 1788-2015 §11)

the rule, written out here from 1788's definitions and decided by brute force, independently of
`intervals.decorated`: an op's local decoration on the box of its operands' sets is trv unless every
point of the box is in the op's domain (a set of reals, so an attained ±inf is never in it), def
unless the op restricted to the box is continuous, dac unless it is also continuous at every point
of the box and every operand is bounded, else com; the result's decoration is the min of that, each
operand's decoration and the best the result can have (trv for ∅, dac at most for an unbounded
result). a multi-piece box is decided on the set: its pieces are apart, so on each piece.

* the oracle: operands on a quarter-integer grid (ends in [-3, 3], open or closed, several pieces,
  ±inf as points and as open ends). the domains' boundaries (0, ±1) and the step functions' jumps
  (integers, half-integers) are grid points, so probing the eighth-integer grid of each piece, plus
  far points on an unbounded one, decides domain membership, constancy on a piece and jumps inside
  it exactly. the poles of tan, sec, cot and csc against a 30-digit pi (a grid end is never within
  1e-3 of a pole but 0). atan2's cut and `%`'s jumps from their definitions (corners with an open one
  nudged inward, as `x / y` is monotone in each coordinate on a box off y = 0). equality with the
  oracle is soundness and maximality at once
* float operands decorate as the exact values of the same doubles, but for the result's boundedness,
  which is the rounded result's (`min(exact decoration, newDec of the float result)`)
* laws: the min structure (`f(set_dec(x, d))` is `min(d, f(newDec x))`), antitone in the box (a
  non-empty sub-box never decorates worse), the set is always the core op's (and keeps its class)
* set operations and cancellation are trv; mixing a bare `MultiInterval` is a `TypeError`; the core's
  warnings reach the caller once and the decoration's own work warns nothing
* the reverse ops (M13's merge): the core's set on the intervals, trv, as 1788 decorates them
* `@example`s: the itf1788 vectors that pin each rule (`libieeep1788_elem.itl`, `set.itl`, `cancel.itl`)
"""
import math
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import abs_rev
from intervals import cos_rev
from intervals import cosh_rev
from intervals import mul_rev
from intervals import pow_rev1
from intervals import pow_rev2
from intervals import pown_rev
from intervals import set_dec
from intervals import sin_rev
from intervals import sqr_rev
from intervals import tan_rev
from intervals.errors import DomainClippedWarning
from intervals.errors import HullWarning
from intervals.errors import IntervalWarning
from intervals.rounding import exact_cuts
from tests.strategies import cut_tuples
from tests.test_decorated import bounded

M = MultiInterval
INF = math.inf
TRV, DEF, DAC, COM = Decoration.TRV, Decoration.DEF, Decoration.DAC, Decoration.COM
WORST_TO_BEST = [TRV, DEF, DAC, COM]
MAX = 1.7976931348623157e308
PI = Fraction(314159265358979323846264338327950, 10 ** 32)  # pi to 32 digits, error < 1e-31

quiet = pytest.mark.filterwarnings('ignore::intervals.errors.IntervalWarning')


def newdec(x: MultiInterval) -> Decoration:
    """1788's newDec on the set, written out: trv for ∅, com if bounded, else dac"""
    return TRV if not x else COM if bounded(x) else DAC


# THE ORACLE

@st.composite
def grid_sets(draw, unit=Fraction(1), max_pieces=3):
    """sets whose finite ends are quarter units in [-3, 3] units, and ±inf (mostly as an open end, as
    1788 has it; attained, an infinity is outside every domain and the rest of the box is moot).
    mostly one piece, sometimes none, often narrow (inside one step, off a pole)"""
    # the points where domains end and steps jump (0, ±1/2, ±1) often
    quarter = st.one_of(st.integers(-12, 12).map(lambda k: Fraction(k, 4) * unit),
                        st.sampled_from([0, Fraction(1, 2), -Fraction(1, 2), 1, -1]).map(lambda v: v * unit))
    end = st.one_of(*[quarter] * 6, st.sampled_from([-INF, INF]))
    pieces = []
    for _ in range(draw(st.sampled_from([1, 1, 1, 1, 2, 2, max_pieces, 0]))):
        if draw(st.booleans()):
            a = draw(quarter)
            b = a + draw(st.sampled_from([0, Fraction(1, 4), Fraction(1, 2)])) * unit
        else:
            a, b = sorted((draw(end), draw(end)))
        closed = [draw(st.booleans()) if _real(v) else draw(st.sampled_from([False] * 4 + [True])) for v in (a, b)]
        pieces.append((a, b, *closed))
    return M.from_pieces(pieces)


def decorated(sets):
    """a decorated set: mostly newDec's decoration, else any up to it (set_dec)"""
    decorations = st.one_of(*[st.none()] * 3, st.sampled_from(WORST_TO_BEST))
    return st.tuples(sets, decorations).map(lambda p: DecoratedInterval(p[0]) if p[1] is None else set_dec(*p))


def probes(x: MultiInterval, unit=Fraction(1)):
    """the eighth-unit grid points of x in [-5, 5] units, far points on an unbounded piece, each
    piece's finite ends and points just inside them (for an example off the grid), and ±inf where x
    holds them"""
    points = [Fraction(k, 8) * unit for k in range(-40, 41)] + [s * f * unit for s in (-1, 1) for f in (10, 1000)]
    tiny = Fraction(1, 10 ** 9) * unit
    for p in x:
        for end, inward in ((p.inf, tiny), (p.sup, -tiny)):
            if _real(end):
                points += [Fraction(end), Fraction(end) + inward]
    return [p for p in points if p in x] + [v for v in (-INF, INF) if v in x]


def _real(v) -> bool:
    return -INF < v < INF


POINT_DOMAINS = {
    'sqrt': lambda x: x >= 0, 'log': lambda x: x > 0, 'log2': lambda x: x > 0, 'log10': lambda x: x > 0,
    'log1p': lambda x: x > -1, 'asin': lambda x: -1 <= x <= 1, 'acos': lambda x: -1 <= x <= 1,
    'atanh': lambda x: -1 < x < 1, 'acosh': lambda x: x >= 1, 'acoth': lambda x: abs(x) > 1,
    'reciprocal': lambda x: x != 0, 'coth': lambda x: x != 0, 'csch': lambda x: x != 0,
    'div': lambda x, y: y != 0, 'mod': lambda x, y: y != 0, 'floordiv': lambda x, y: y != 0,
    'pow': lambda x, y: x > 0 or (x == 0 and y > 0), 'atan2': lambda y, x: not (y == 0 and x == 0),
}
POLES = {'tan': Fraction(1, 2), 'sec': Fraction(1, 2), 'cot': Fraction(0), 'csc': Fraction(0)}
STEP_POINT = {
    'floor': lambda v, u: math.floor(v / u), 'ceil': lambda v, u: math.ceil(v / u),
    'trunc': lambda v, u: int(v / u), 'round': lambda v, u: round(v / u),
    'round_ties_away': lambda v, u: int(abs(v / u) + Fraction(1, 2)) * (1 if v >= 0 else -1),
    'sign': lambda v, u: (v > 0) - (v < 0),
}
STEP_JUMP = {
    'floor': lambda k: k.denominator == 1, 'ceil': lambda k: k.denominator == 1,
    'trunc': lambda k: k.denominator == 1 and k != 0, 'sign': lambda k: k == 0,
    'round': lambda k: (k - Fraction(1, 2)).denominator == 1,
    'round_ties_away': lambda k: (k - Fraction(1, 2)).denominator == 1,
}


def _has_pole(x: MultiInterval, offset: Fraction) -> bool:
    for p in x:
        if not (_real(p.inf) and _real(p.sup)):
            return True
        near = range(math.floor(Fraction(p.inf) / PI) - 1, math.floor(Fraction(p.sup) / PI) + 2)
        if any((k + offset) * PI in p for k in near if k + offset != 0):
            return True
        if offset == 0 and 0 in p:
            return True
    return False


def defined(name: str, *sets, extra=()) -> bool:
    """every point of the box is a real point in the op's domain"""
    if any(v in s for s in sets for v in (-INF, INF)):
        return False
    if name in POLES:
        return not _has_pole(sets[0], POLES[name])
    if name == 'pown':
        return extra[0] >= 0 or 0 not in sets[0]
    if name == 'rootn':
        n = extra[0]
        return all((v >= 0 if n > 0 else v > 0) if n % 2 == 0 else (n > 0 or v != 0) for v in probes(sets[0]))
    test = POINT_DOMAINS.get(name)
    if test is None:
        return True
    if len(sets) == 1:
        return all(test(v) for v in probes(sets[0]))
    return all(test(u, v) for u in probes(sets[0]) for v in probes(sets[1]))


def _step_continuity(name, x: MultiInterval, unit):
    """(restricted, everywhere) of a step function on the set, by brute force per piece"""
    restricted = everywhere = True
    for p in x:
        if len({STEP_POINT[name](v, unit) for v in probes(p, unit)}) != 1:
            return False, False
        if name != 'sign' and not bounded(p):
            everywhere = False  # an unbounded piece holds infinitely many jumps
            continue
        # every half unit from just below the piece to just above it (sign: only 0 is a jump)
        lo, hi = (Fraction(-1), Fraction(1)) if not bounded(p) else (Fraction(p.inf) / unit, Fraction(p.sup) / unit)
        halves = (Fraction(k, 2) for k in range(math.floor(2 * lo) - 1, math.floor(2 * hi) + 2))
        if any(STEP_JUMP[name](k) and k * unit in p for k in halves):
            everywhere = False
    return restricted, everywhere


def _atan2_continuity(y: MultiInterval, x: MultiInterval):
    """atan2 is pi on the negative x axis and near -pi just below it"""
    on_cut = 0 in y and any(v < 0 for v in probes(x))
    if not on_cut:
        return True, True
    return Fraction(-1, 8) not in y, False  # from below: the piece holding 0 reaches below it


def _corner_ratios(p: MultiInterval, q: MultiInterval):
    """x / y at the corners of a box off y = 0, an open corner nudged inward, an unbounded side far out
    (x / y is monotone in each coordinate there, so these are its extremes, one sidedly)"""
    eps = Fraction(1, 10 ** 6)

    def ends(r):
        lo = -Fraction(10 ** 9) if r.inf == -INF else r.inf if r.inf_closed else r.inf + eps
        hi = Fraction(10 ** 9) if r.sup == INF else r.sup if r.sup_closed else r.sup - eps
        return lo, hi
    return [Fraction(a) / Fraction(b) for a in ends(p) for b in ends(q)]


def _quotient_continuity(x: MultiInterval, y: MultiInterval):
    """`%` and `//` jump wherever x / y is an integer: continuous on a pair of pieces iff floor(x / y)
    is one integer there, at every point iff moreover x / y never reaches it"""
    everywhere = True
    for p in x:
        for q in y:
            ratios = _corner_ratios(p, q)
            if len({math.floor(r) for r in ratios}) != 1:
                return False, False
            everywhere = everywhere and all(r.denominator != 1 for r in ratios)
    return True, everywhere


def local(name: str, sets, extra=(), unit=Fraction(1)) -> Decoration:
    if not defined(name, *sets, extra=extra):
        return TRV
    restricted = everywhere = True
    if name in STEP_POINT:
        restricted, everywhere = _step_continuity(name, sets[0], unit)
    elif name == 'atan2':
        restricted, everywhere = _atan2_continuity(*sets)
    elif name in ('mod', 'floordiv'):
        restricted, everywhere = _quotient_continuity(*sets)
    if not restricted:
        return DEF
    return COM if everywhere and all(bounded(s) for s in sets) else DAC


def expected(name, operands, result: MultiInterval, extra=(), unit=Fraction(1)) -> Decoration:
    sets = [M.from_cuts(exact_cuts(o.interval.cuts)) for o in operands]
    if not all(sets):
        return TRV
    return min(local(name, sets, extra, unit), newdec(result), *(o.decoration for o in operands))


# THE OPS: name -> the decorated call, and the core's on the intervals

def _method(name):
    return lambda a: getattr(a, name)()


UNARY = {
    **{n: _method(n) for n in (
        'sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'expm1', 'log1p', 'cbrt', 'cot', 'sec', 'csc',
        'acot', 'coth', 'csch', 'sech', 'acoth', 'reciprocal', 'floor', 'ceil', 'trunc', 'round',
        'round_ties_away', 'sign')},
    'neg': lambda a: -a, 'pos': lambda a: +a, 'abs': abs,
}
BINARY = {
    'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b, 'div': lambda a, b: a / b,
    'mod': lambda a, b: a % b, 'floordiv': lambda a, b: a // b, 'min': lambda a, b: a.minimum(b),
    'max': lambda a, b: a.maximum(b), 'hypot': lambda a, b: a.hypot(b), 'atan2': lambda a, b: a.atan2(b),
    'pow': lambda a, b: a ** b,
}
# the slow ones (irrational values through ziv's loop, or many pieces) run fewer examples
SLOW = {'pow', 'hypot', 'tan', 'cot', 'sec', 'csc', 'sin', 'cos'}


def _run(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return fn(*args)


def _check(name, fn, operands, extra=(), unit=Fraction(1)):
    ours = _run(fn, *operands)
    core = _run(fn, *(o.interval for o in operands))
    assert isinstance(ours, DecoratedInterval)
    assert ours.interval == core and type(ours.interval) is type(core)  # the set is the core's
    assert ours.decoration is expected(name, operands, core, extra, unit), (name, operands, ours)


@quiet
@settings(max_examples=40, deadline=None)
@given(st.sampled_from(sorted(UNARY)), decorated(grid_sets()))
@example('floor', set_dec(M(Fraction(11, 10), 2), COM))  # elem.itl:4200: floor [1.1,2.0]_com = _def
@example('ceil', set_dec(M(Fraction(11, 10), 2), COM))  # :4167: ceil [1.1,2.0]_com = [2.0,2.0]_dac
@example('floor', set_dec(M(Fraction(-12, 10), Fraction(-11, 10)), COM))  # :4203: _com
@example('round', set_dec(M(Fraction(-16, 10), Fraction(-15, 10)), COM))  # :4269: roundTiesToEven, _dac
@example('round_ties_away', set_dec(M(Fraction(25, 10), Fraction(26, 10)), COM))  # :4301: _dac
@example('round_ties_away', set_dec(M(Fraction(19, 10), Fraction(22, 10)), COM))  # :4299: _com
@example('trunc', set_dec(M(Fraction(11, 10), Fraction(19, 10)), COM))  # :4232: _com
@example('trunc', set_dec(M(MAX), COM))  # :4241: trunc [max,max]_com = _dac (max is an integer)
@example('sign', set_dec(M(-1, 0), COM))  # :4141: _def
@example('sign', set_dec(M(0), DAC))  # :4145: sign [0.0,0.0]_dac = _dac
@example('sign', set_dec(M(0), COM))  # sign is not continuous at 0: dac
@example('sign', set_dec(M(0, 1, start_closed=False), COM))  # (0, 1]: com
@example('trunc', set_dec(M(0, Fraction(1, 2)), COM))  # trunc is continuous at 0: com
@example('trunc', set_dec(M(Fraction(-1, 2), Fraction(1, 2)), COM))  # com
@example('csc', set_dec(M(-1, 0), COM))  # a pole at 0, a closed end: trv
@example('reciprocal', set_dec(M(0), COM))  # :708: recip [0.0,0.0]_com = [empty]_trv
@example('reciprocal', set_dec(M(-10, 0), COM))  # :709: _trv
@example('sqrt', set_dec(M(-5, 25), COM))  # :755: _trv
@example('sqrt', set_dec(M(0, 25), DEF))  # :756: _def
@example('log', set_dec(M(0, 1), COM))  # :3236: [-infinity,0.0]_trv (log 0 is -inf here, still undefined)
@example('acosh', set_dec(M(1), COM))  # :4086: acosh [1.0,1.0]_com = [0.0,0.0]_com
@example('log1p', set_dec(M(-1, 0), COM))  # log1p(-1) is -inf here, and undefined: trv
@example('log1p', set_dec(M(-1, 0, start_closed=False), COM))  # off -1: (-inf, 0], dac
@example('acosh', set_dec(M(Fraction(9, 10), 1), COM))  # :4087: _trv
@example('atanh', set_dec(M(-1, 1), COM))  # :4114: [entire]_trv
@example('asin', set_dec(M.parse('[0, inf)'), DAC))  # :3554: _trv
@example('tan', set_dec(M(0, float.fromhex('0x1.921FB54442D18P+0')), DAC))  # :3506: just below pi/2, _dac
@example('tan', set_dec(M(0, float.fromhex('0x1.921FB54442D19P+0')), TRV))  # :3508: just past it, [entire]_trv
@example('tan', set_dec(M(float.fromhex('0x1.4E18E147AE148P+12'), float.fromhex('0x1.546028F5C28F6P+12')), DEF))  # :3527: _trv
@example('cot', set_dec(M(0, 1), COM))  # a pole at 0, a closed end
@example('cot', set_dec(M(0, 1, start_closed=False), COM))  # open there: com
@example('floor', set_dec(M.parse('[inf]'), DAC))  # an attained infinity is outside every domain
@example('exp', set_dec(M.parse('[0, inf)'), DAC))  # unbounded: dac
@example('sign', set_dec(M.parse('(0, inf)'), DAC))  # constant on an unbounded piece
@example('floor', set_dec(M.parse('{ [1/4, 1/2] , [5/4, 3/2) }'), COM))  # constant on each piece: com
@example('floor', set_dec(M.parse('{ [1/4, 1/2] , [1, 3/2) }'), COM))  # a closed jump at 1: dac
# M13g review: domain ends and attained infinities no random run is sure to reach (each pins a clause)
@example('pos', set_dec(M.parse('[inf]'), DAC))  # +x is a function of reals too: trv
@example('asin', set_dec(M(Fraction(-3, 2), 0), COM))  # past asin's domain: trv
@example('acoth', set_dec(M(1, 2), COM))  # acoth(1) is undefined: trv
@example('acoth', set_dec(M(-2, -1), COM))  # and acoth(-1): trv
@example('acoth', set_dec(M(1, 2, start_closed=False), COM))  # off 1: com
def test_unary_ops_decorate_as_1788(name, x):
    _check(name, UNARY[name], (x,))


@quiet
@settings(max_examples=40, deadline=None)
@pytest.mark.parametrize('name', sorted(UNARY))
@given(x=decorated(grid_sets()))
def test_each_unary_op(name, x):
    """the same, with examples of its own for every op"""
    _check(name, UNARY[name], (x,))


@quiet
@settings(max_examples=40, deadline=None)
@pytest.mark.parametrize('name', sorted(BINARY))
@given(a=decorated(grid_sets(max_pieces=2)), b=decorated(grid_sets(max_pieces=2)))
def test_each_binary_op(name, a, b):
    _check(name, BINARY[name], (a, b))


@quiet
@settings(max_examples=40, deadline=None)
@given(st.sampled_from(sorted(set(BINARY) - SLOW)), decorated(grid_sets()), decorated(grid_sets()))
@example('add', set_dec(M(1, 2), COM), set_dec(M(5, 7), DEF))  # elem.itl:110: _def
@example('add', set_dec(M(1, 2), TRV), set_dec(M(), TRV))  # :113: [empty]_trv
@example('div', set_dec(M(-2, -1), COM), set_dec(M(0, 10), COM))  # :677: _trv
@example('div', set_dec(M(1, 3), DEF), set_dec(M.parse('(-inf, -10]'), DAC))  # :678: _def
@example('min', set_dec(M(-7, 0), DAC), set_dec(M(2, 4), DEF))  # :4353: _def
@example('atan2', set_dec(M(-2, 0), COM), set_dec(M(-2, Fraction(-1, 10)), DAC))  # atan2.itl: [-PI,PI]_def
@example('atan2', set_dec(M(0, 1), COM), set_dec(M(-2, Fraction(-1, 10)), COM))  # _dac
@example('atan2', set_dec(M(0), COM), set_dec(M(Fraction(1, 10), 1), COM))  # [0.0,0.0]_com
@example('atan2', set_dec(M(0), COM), set_dec(M(0, 1), COM))  # the origin: [0.0,0.0]_trv
@example('atan2', set_dec(M.parse('{ [-2, -1] , [0] }'), COM), set_dec(M(-2, -1), COM))  # apart: dac
@example('mod', set_dec(M(Fraction(1, 4), Fraction(3, 4)), COM), set_dec(M(1), COM))  # inside a step: com
@example('mod', set_dec(M(1, Fraction(3, 2)), COM), set_dec(M(1), COM))  # a closed jump at 1: dac
@example('floordiv', set_dec(M(-3, -2), COM), set_dec(M(2), COM))  # x / y reaches -1: dac
@example('mod', set_dec(M(Fraction(1, 4), 2), COM), set_dec(M(1), COM))  # across one: def
@example('floordiv', set_dec(M(1, 2), COM), set_dec(M(0, 1), COM))  # the divisor holds 0: trv
@example('div', set_dec(M.parse('[0, inf]'), DAC), set_dec(M(1, 2), COM))  # M13g review: an attained inf dividend, trv
def test_binary_ops_decorate_as_1788(name, a, b):
    _check(name, BINARY[name], (a, b))


@quiet
@settings(max_examples=15, deadline=None)
@given(st.sampled_from(sorted(SLOW & set(BINARY))), decorated(grid_sets(max_pieces=2)),
       decorated(grid_sets(max_pieces=2)))
@example('pow', set_dec(M(0, 1), DEF), set_dec(M(0, Fraction(5, 2)), DAC))  # elem.itl pow: (0, 0) inside, _trv
@example('pow', set_dec(M(0), COM), set_dec(M.parse('[1, inf)'), DAC))  # [0.0,0.0]_dac
@example('pow', set_dec(M(0), COM), set_dec(M(Fraction(-5, 2), Fraction(1, 10)), COM))  # _trv
@example('pow', set_dec(M(0, Fraction(1, 2)), COM), set_dec(M(Fraction(1, 10)), COM))  # x = 0 with y > 0: com
@example('pow', set_dec(M(-1, Fraction(-1, 10)), DAC), set_dec(M(Fraction(-1, 10), Fraction(5, 2)), COM))  # _trv
@example('pow', set_dec(M(1, 2), COM), set_dec(M.parse('[0, inf]'), DAC))  # M13g review: an attained inf exponent, trv
def test_slow_binary_ops_decorate_as_1788(name, a, b):
    _check(name, BINARY[name], (a, b))


@quiet
@settings(max_examples=40, deadline=None)
@given(decorated(grid_sets()), st.integers(-4, 4))
@example(set_dec(M(-5, 10), COM), 0)  # elem.itl:1588: pown [-5.0,10.0]_com 0 = [1.0,1.0]_com
@example(set_dec(M.parse('(-inf, 15]'), DAC), 0)  # :1589: _dac
@example(set_dec(M(-5, 3), COM), -2)  # :1596: _trv
@example(set_dec(M(3, 5), DAC), -3)  # :1597: _dac
@example(set_dec(M(-3, 5), COM), -3)  # :1598: [entire]_trv
def test_pown_decorates_as_1788(x, n):
    _check('pown', lambda a: a ** n, (x,), extra=(n,))


@quiet
@settings(max_examples=60, deadline=None)
@given(decorated(grid_sets()), st.integers(-4, 4).filter(bool))
@example(set_dec(M(0, 16), COM), 4)  # an even root from 0: com
@example(set_dec(M(-1, 4), COM), 2)  # below 0: trv
@example(set_dec(M(-1, 1), COM), -3)  # an odd negative root at 0: trv
@example(set_dec(M(0, 16), COM), -2)  # rootn(0, -2) has no value: trv
@example(set_dec(M(-8, 27), COM), 3)
@example(set_dec(M(-8, -1), COM), -3)
@example(set_dec(M.parse('{ [-1, -1/2] , [1, 4] }'), COM), -2)  # M13g review: below 0, and 0 not in it: trv
def test_rootn_decorates_as_1788(x, n):
    _check('rootn', lambda a: a.rootn(n), (x,), extra=(n,))


@quiet
@settings(max_examples=40, deadline=None)
@given(st.sampled_from(['round', 'round_ties_away']), st.sampled_from([-1, 0, 1, 2]), st.data())
def test_round_to_ndigits_decorates_as_1788(name, ndigits, data):
    unit = Fraction(10) ** -ndigits
    x = data.draw(decorated(grid_sets(unit)))
    _check(name, lambda a: getattr(a, name)(ndigits), (x,), unit=unit)


@pytest.mark.parametrize('name, ndigits, x, decoration', [
    ('round', 1, M(Fraction(3, 20), Fraction(1, 5)), DAC),  # round(0.15, 1) = 0.2 = round(0.2, 1); 0.15 jumps
    ('round', 1, M(Fraction(3, 20), Fraction(1, 5), start_closed=False), COM),
    ('round_ties_away', -1, M(25, 30), DAC),  # 25 rounds away to 30, and jumps
    ('round', 2, M(Fraction(1, 1000), Fraction(4, 1000)), COM),  # inside one step of 0.01
])
def test_round_to_ndigits_examples(name, ndigits, x, decoration):
    result = getattr(DecoratedInterval(x), name)(ndigits)
    assert result.decoration is decoration
    _check(name, lambda a: getattr(a, name)(ndigits), (DecoratedInterval(x),), unit=Fraction(10) ** -ndigits)


@quiet
@settings(max_examples=40, deadline=None)
@given(decorated(grid_sets()), decorated(grid_sets()), decorated(grid_sets()))
@example(set_dec(M(1, 2), COM), set_dec(M(1, 2), COM), set_dec(M(2, 5), COM))  # elem.itl:1405: _com
@example(set_dec(M(Fraction(-1, 2), Fraction(-1, 10)), COM), set_dec(M.parse('(-inf, 3]'), DAC),
         set_dec(M(Fraction(-1, 10), Fraction(1, 10)), COM))  # :1403: _dac
# M13g review: the addend's decoration and its attained inf count
@example(set_dec(M(1, 2), COM), set_dec(M(1, 2), COM), set_dec(M(0, 1), DEF))  # [1, 5]_def
@example(set_dec(M(1, 2), COM), set_dec(M(1, 2), COM), set_dec(M.parse('[0, inf]'), DAC))  # trv
def test_fma_decorates_as_1788(a, b, c):
    _check('fma', lambda x, y, z: x.fma(y, z), (a, b, c))


# FLOAT OPERANDS

float_sets = cut_tuples(max_pieces=3).map(M.from_cuts)
big = st.sampled_from([MAX, -MAX, MAX / 2, 1.0, -2.0, 0.5])
big_sets = cut_tuples(values=big, max_pieces=2).map(M.from_cuts)


def _as_class(x: DecoratedInterval, cls) -> DecoratedInterval:
    return set_dec(cls.from_cuts(x.interval.cuts), x.decoration)


def _as_exact(x: DecoratedInterval) -> DecoratedInterval:
    return set_dec(M.from_cuts(exact_cuts(x.interval.cuts)), x.decoration)


@quiet
@settings(max_examples=60, deadline=None)
@given(st.sampled_from(sorted(set(BINARY) - SLOW)), decorated(st.one_of(float_sets, big_sets)),
       decorated(st.one_of(float_sets, big_sets)), st.sampled_from([M, OutwardMultiInterval]))
@example('add', set_dec(M(1.0, 2.0), COM), set_dec(M(5.0, MAX), COM), OutwardMultiInterval)  # elem.itl:111
@example('add', set_dec(M(1.0, 2.0), COM), set_dec(M(5.0, MAX), COM), M)  # rounds to max: com
@example('mul', set_dec(M(-MAX, 2.0), COM), set_dec(M(-1.0, 5.0), COM), OutwardMultiInterval)  # :306: _dac
@example('div', set_dec(M(-200.0, -1.0), COM), set_dec(M(5e-324, 10.0), COM), OutwardMultiInterval)  # :676
@example('mod', set_dec(M(1.0), COM), set_dec(M(0.1), COM), M)  # 1.0 / 0.1 is 9.99..., 10.0 rounded: com
@example('floordiv', set_dec(M(1.0), COM), set_dec(M(0.1), COM), OutwardMultiInterval)
def test_float_operands_decorate_as_their_exact_values(name, a, b, cls):
    """the same decision on the same doubles; only com needs the rounded result bounded"""
    fn = BINARY[name]
    a, b = _as_class(a, cls), _as_class(b, cls)
    ours = _run(fn, a, b)
    exact = _run(fn, _as_exact(a), _as_exact(b))
    assert type(ours.interval) is cls and bool(ours.interval) == bool(exact.interval)
    assert ours.decoration is min(exact.decoration, newdec(ours.interval))


@quiet
@settings(max_examples=60, deadline=None)
@given(st.sampled_from(sorted(set(UNARY) - SLOW)), decorated(st.one_of(float_sets, big_sets)),
       st.sampled_from([M, OutwardMultiInterval]))
@example('exp2', set_dec(M(1024.0), COM), OutwardMultiInterval)  # elem.itl:3167: [max,infinity]_dac
@example('exp2', set_dec(M(1024.0), COM), M)  # 2 ** 1024 rounds to inf here too
@example('sqrt', set_dec(M(-MAX, MAX), COM), OutwardMultiInterval)
def test_float_operand_of_a_function(name, x, cls):
    x = _as_class(x, cls)
    ours = _run(UNARY[name], x)
    exact = _run(UNARY[name], _as_exact(x))
    assert type(ours.interval) is cls
    assert ours.decoration is min(exact.decoration, newdec(ours.interval))


# LAWS

ALL = {**{(n, 1): f for n, f in UNARY.items() if n not in SLOW}, **{(n, 2): f for n, f in BINARY.items() if n not in SLOW}}


@quiet
@settings(max_examples=60, deadline=None)
@given(st.sampled_from(sorted(ALL)), grid_sets(), grid_sets(), st.sampled_from(WORST_TO_BEST), st.booleans())
def test_the_result_is_the_min_over_the_inputs(op, x, y, d, first):
    """f(set_dec(x, d)) is min(d, f(newDec x)): an operand's decoration only ever caps the result"""
    name, arity = op
    fn = ALL[op]
    best = [DecoratedInterval(x), DecoratedInterval(y)][:arity]
    capped = list(best)
    i = 0 if first or arity == 1 else 1
    capped[i] = set_dec(capped[i].interval, d)
    assert _run(fn, *capped).decoration is min(capped[i].decoration, _run(fn, *best).decoration)


@quiet
@settings(max_examples=60, deadline=None)
@given(st.sampled_from(sorted(ALL)), grid_sets(), grid_sets(), grid_sets(), grid_sets())
@example(('floor', 1), M(0, 1, end_closed=False), M(), M(Fraction(1, 2)), M())  # dac, then com
@example(('atan2', 2), M(-1, 1), M(-2, -1), M(0, 1), M(-2, -1))  # def, then dac
def test_a_sub_box_never_decorates_worse(op, x, y, sx, sy):
    """1788's decorations are antitone in the box: defined, continuous (restricted or at each point),
    bounded all hold on a non-empty sub-box if they hold on the box"""
    name, arity = op
    fn = ALL[op]
    box = [x, y][:arity]
    sub = [x & sx, y & sy][:arity]
    if not all(sub):
        return
    whole = _run(fn, *map(DecoratedInterval, box)).decoration
    part = _run(fn, *map(DecoratedInterval, sub)).decoration
    assert part >= whole, (op, box, sub, whole, part)


@pytest.mark.parametrize('op, args', [
    (lambda a, b: a & b, 2), (lambda a, b: a | b, 2), (lambda a, b: a ^ b, 2), (lambda a, b: a.difference(b), 2),
    (lambda a: a.complement(), 1), (lambda a: ~a, 1), (lambda a: a.hull, 1), (lambda a: a.closed_hull, 1),
    (lambda a: a.interior, 1), (lambda a, b: a.cancel_minus(b), 2), (lambda a, b: a.cancel_plus(b), 2),
    # M13g review: the core's named n-ary forms and its other set-valued operations
    (lambda a, b: a.union(b), 2), (lambda a, b: a.intersection(b), 2), (lambda a, b: a.symmetric_difference(b), 2),
    (lambda a, b: a.union(b, 5, b), 2), (lambda a, b: a.intersection(a, b), 2), (lambda a, b: a.difference(b, 0, 1), 2),
    (lambda a, b: a.symmetric_difference(b, 1), 2), (lambda a: a.union(), 1), (lambda a: a.positive, 1),
    (lambda a: a.negative, 1), (lambda a: a.finite, 1), (lambda a: a.expand(1), 1), (lambda a: a.expand(0), 1),
    (lambda a: a[0:Fraction(5, 2)], 1), (lambda a: a[:1], 1),
])
@pytest.mark.parametrize('x, y', [(M(1, 3), M(2, 4)), (M(1, 3), M()), (M.parse('(-inf, inf)'), M(1, 2)),
                                  (M.parse('{ [0, 1] , [3, 4] }'), M(0, 1))])
def test_set_operations_are_trv(op, args, x, y):
    """libieeep1788_set.itl:33: intersection [1.0,3.0]_com [2.1,4.0]_com = [2.1,3.0]_trv; cancel.itl: every
    decorated result trv"""
    a, b = DecoratedInterval(x), DecoratedInterval(y)
    result = _run(op, *(a, b)[:args])
    assert result.decoration is TRV
    assert result.interval == _run(op, *(x, y)[:args])


def test_the_set_keeps_its_class():
    a = DecoratedInterval(OutwardMultiInterval(0.1, 0.2))
    for result in (a + 0.1, 0.1 + a, a * a, a.exp(), a.floor(), a & a, divmod(a, 0.03)[1], 2 ** a):
        assert type(result.interval) is OutwardMultiInterval


def test_numbers_are_points_and_a_bare_set_is_refused():
    a = DecoratedInterval(M(1, 2))
    assert a + 1 == DecoratedInterval(M(2, 3)) and 1 - a == DecoratedInterval(M(-1, 0))
    assert (a + math.inf).decoration is TRV  # the point inf is outside add's domain
    for other in (M(1, 2), '1', True, None):
        with pytest.raises(TypeError):
            _ = a + other
        with pytest.raises(TypeError):
            _ = other * a
        with pytest.raises(TypeError):
            a.minimum(other)
    with pytest.raises(TypeError):
        _ = a ** M(2)
    x = DecoratedInterval(M(-3, 1))  # an integral float exponent is pown, as in the core (D11)
    assert x ** 2.0 == x ** 2 == DecoratedInterval(M(0, 9))
    with pytest.warns(DomainClippedWarning):
        assert (x ** 2.5).decoration is TRV  # pow: negative bases are outside its domain
    with pytest.raises(TypeError):
        pow(a, 2, 3)


def test_divmod_is_both_decorated():
    a, b = DecoratedInterval(M(Fraction(1, 4), Fraction(3, 4))), DecoratedInterval(M(1))
    q, r = divmod(a, b)
    assert q == a // b and r == a % b and q.decoration is r.decoration is COM
    q, r = divmod(3, b)
    assert q == DecoratedInterval(M(3), DAC) and r == DecoratedInterval(M(0), DAC)  # 3 / 1 is an integer


def test_the_core_warns_once_and_the_decoration_never():
    x = DecoratedInterval(M(-1, 4))
    with pytest.warns(DomainClippedWarning) as caught:
        result = x.sqrt()
    assert len(caught) == 1 and result == DecoratedInterval(M(0, 2), TRV)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert DecoratedInterval(M(Fraction(1, 2), 3)).floor().decoration is DEF  # no warning of its own
        assert DecoratedInterval(M(1, 5)).__mod__(DecoratedInterval(M(1, 2))).decoration is DEF
    with pytest.warns(HullWarning) as caught:
        DecoratedInterval(M.parse('[0, inf)')).floor()
    assert len(caught) == 1  # the core's, not the constancy check's


def test_methods_mirror_the_core():
    """every point function of MultiInterval has its decorated method (M13g part 3's scope)"""
    for name in (*(set(UNARY) - {'neg', 'pos', 'abs'}), 'log', 'rootn', 'minimum', 'maximum', 'fma', 'hypot', 'atan2', 'cancel_minus',
                 'cancel_plus', '__floor__', '__ceil__', '__trunc__', '__round__', '__rpow__', '__rmod__',
                 '__rfloordiv__', '__rdivmod__', '__rtruediv__'):
        assert hasattr(MultiInterval, name) and callable(getattr(DecoratedInterval, name, None)), name
    x = DecoratedInterval(M(Fraction(3, 2)))
    assert (math.floor(x), math.ceil(x), math.trunc(x), round(x)) == (
        DecoratedInterval(M(1)), DecoratedInterval(M(2)), DecoratedInterval(M(1)), DecoratedInterval(M(2), DAC))
    assert x.log(2).decoration is COM and DecoratedInterval(M(0, 1)).log(2).decoration is TRV


# M13g review: what 1788 asks of the interval part, so not on the wrapper (`.interval` first)
NOT_ON_THE_WRAPPER = {
    'adjoins', 'after', 'allen', 'allen_matrix', 'allen_relations', 'before', 'contains', 'cuts',
    'degenerate_points', 'eq_pointwise', 'from_cuts',
    'from_pieces', 'inf', 'inf_closed', 'is_contiguous', 'is_degenerate', 'is_empty', 'is_finite', 'is_integral',
    'is_negative', 'is_non_negative', 'is_non_positive', 'is_positive', 'isdisjoint', 'issubset', 'issuperset',
    'mag', 'mid', 'mid_rad', 'mig', 'overlaps', 'parse', 'pieces', 'rad', 'size', 'sort_key', 'strictly_less',
    'sup', 'sup_closed', 'weakly_less', 'wid', 'within',
}


def test_every_public_name_of_the_core_is_on_the_wrapper_or_asked_of_the_interval():
    """the plan: decorations "propagated through every op the core has". a name the core gains must be
    added to the wrapper or here, on purpose"""
    core = {n for n in dir(MultiInterval) if not n.startswith('_')}
    assert NOT_ON_THE_WRAPPER <= core
    missing = {n for n in core - NOT_ON_THE_WRAPPER if not hasattr(DecoratedInterval, n)}
    assert not missing, missing
    assert not {n for n in NOT_ON_THE_WRAPPER if hasattr(DecoratedInterval, n)}
    with pytest.raises(TypeError):
        iter(DecoratedInterval(M(1, 2)))  # `x[a:b]` is the restriction, not an item: not a sequence


def test_reflected_operators_reflect():
    """M13g review: a number on the left is the first operand (values checked, not only the class)"""
    d = DecoratedInterval(M(2, 4))
    assert 1 / d == DecoratedInterval(M(Fraction(1, 4), Fraction(1, 2)))
    assert 5 % d == DecoratedInterval(M(0, Fraction(5, 2), end_closed=False), DEF)
    assert 5 // d == DecoratedInterval(M.parse('{ [1] , [2] }'), DEF)
    assert 3 ** d == DecoratedInterval(M(9, 81))
    assert 1 - d == DecoratedInterval(M(-3, -1)) and divmod(5, d) == (5 // d, 5 % d)
    for fn in (lambda a, b: a / b, lambda a, b: a % b, lambda a, b: a // b, lambda a, b: a ** b, lambda a, b: a - b):
        assert fn(5, d) == fn(DecoratedInterval(M(5)), d)


@pytest.mark.parametrize('name', ['round', 'round_ties_away'])
@pytest.mark.parametrize('x', [OutwardMultiInterval(0.12, 0.13), OutwardMultiInterval(0.15)])
def test_a_step_is_decided_on_the_exact_set(name, x):
    """M13g review: rounding 0.12..0.13 to one digit gives 1/10 exactly, one value, so com; the outward
    result is the two doubles around 1/10, which is not one value. the step is decided on the exact
    set (`decorated.py::_step`), never on a rounded one"""
    result = getattr(DecoratedInterval(x), name)(1)
    assert type(result.interval) is OutwardMultiInterval and result.decoration is COM
    assert result.decoration is getattr(_as_exact(DecoratedInterval(x)), name)(1).decoration


# THE REVERSE OPS (M13's merge of M13e and M13g): given DecoratedInterval operands, each is 1788's
# decorated reverse op, the core's set on the intervals decorated trv (`reverse.py::_decorated`)

# name -> (the op, its interval operands before x, whether it takes an int exponent after them)
REVERSE_OPS = {
    'sqr_rev': (sqr_rev, 1, False), 'abs_rev': (abs_rev, 1, False), 'cosh_rev': (cosh_rev, 1, False),
    'pown_rev': (pown_rev, 1, True), 'sin_rev': (sin_rev, 1, False), 'cos_rev': (cos_rev, 1, False),
    'tan_rev': (tan_rev, 1, False), 'mul_rev': (mul_rev, 2, False), 'pow_rev1': (pow_rev1, 2, False),
    'pow_rev2': (pow_rev2, 2, False),
}


def _reverse_call(name, operands, n, x):
    """the op on `operands` (then `n`, then `x` unless None), as the positional call its signature takes"""
    fn, _, takes_n = REVERSE_OPS[name]
    return fn(*operands, *((n,) if takes_n else ()), *(() if x is None else (x,)))


@quiet
@settings(max_examples=30, deadline=None)
@pytest.mark.parametrize('name', sorted(REVERSE_OPS))
@given(operands=st.lists(decorated(grid_sets(max_pieces=2)), min_size=2, max_size=2),
       x=st.one_of(st.none(), decorated(grid_sets(max_pieces=2))), n=st.integers(-3, 3))
def test_each_reverse_op_is_trv(name, operands, x, n):
    """libieeep1788_rev.itl, `*_dec_test`: every decorated result trv, whatever the operands'
    decorations, the set being the core op's on the intervals (x omitted or given)"""
    operands = operands[:REVERSE_OPS[name][1]]
    result = _reverse_call(name, operands, n, x)
    core = _reverse_call(name, [a.interval for a in operands], n, None if x is None else x.interval)
    assert isinstance(result, DecoratedInterval) and result.decoration is TRV
    assert result.interval == core and type(result.interval) is type(core)


@quiet
def test_a_reverse_op_keeps_the_class_and_refuses_a_bare_set():
    c, b, x = DecoratedInterval(OutwardMultiInterval(0.1, 0.2)), DecoratedInterval(M(1, 2)), DecoratedInterval(M(-1, 1))
    for name in REVERSE_OPS:
        operands = (b, c)[-REVERSE_OPS[name][1]:]
        assert type(_reverse_call(name, operands, 2, None).interval) is OutwardMultiInterval, name
        for bare in (M(-1, 1), M(-5, 5)):  # a bare x, or a bare operand beside a decorated one
            with pytest.raises(TypeError):
                _reverse_call(name, operands, 2, bare)
            if len(operands) == 2:
                with pytest.raises(TypeError):
                    _reverse_call(name, (bare, c), 2, x)
    # a number is a point, as for the other ops; with no decorated operand the op is the core's
    assert mul_rev(2, DecoratedInterval(M(1, 8))) == DecoratedInterval(M(Fraction(1, 2), 4), TRV)
    assert sqr_rev(DecoratedInterval(M(4)), 5) == DecoratedInterval(M(), TRV)
    assert type(sqr_rev(M(4))) is M


def test_a_reverse_op_warns_once():
    with pytest.warns(IntervalWarning) as caught:
        result = sqr_rev(DecoratedInterval(M()), DecoratedInterval(M(0, 1)))
    assert len(caught) == 1 and result == DecoratedInterval(M(), TRV)
