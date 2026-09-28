"""
the 1788 layer's properties (`intervals.ieee1788`, M16b; the vectors' pass is
`tests/itf1788/test_ieee1788.py`)

each property is checked against an oracle written here, never the layer's own helper: 1788's
form, the output rule decided with `Fraction` and `math.nextafter`, newDec, 1788's cancellation rule,
an overlap table keyed on the signs of the ends' comparisons, and a table of library calls written
apart from the module's. the vectors hold none of: a multi-piece or infinite-point library set, an
operand of every flavour beside a number, `wid a == wid b` drawn, a `repr` read back, a warning
leaking out of a call, so those are held by these properties alone.
"""
import builtins
import math
import pickle
import re
import subprocess
import sys
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import intervals
from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import ieee1788
from intervals import kernel
from intervals import reverse
from intervals.errors import DomainClippedWarning
from intervals.errors import IntervalWarning
from intervals.errors import PossiblyUndefinedOperationWarning
from intervals.errors import UndefinedOperationError
from intervals.ieee1788 import Interval
from tests.strategies import cut_tuples
from tests.strategies import endpoint_values

INF = math.inf
MAX = sys.float_info.max
TINY = 5e-324

# ORACLES

def is_1788_form(x) -> bool:
    """empty, or one piece whose finite ends are closed python floats and whose infinite ends are open"""
    s = x.to_set()
    if x.decoration is not None:
        s = s.interval
    if type(s) is not OutwardMultiInterval:
        return False
    if s.is_empty:
        return True
    lo, hi = s.inf, s.sup
    return (s.is_contiguous and type(lo) is float and type(hi) is float and lo < INF and hi > -INF
            and s.inf_closed == (lo != -INF) and s.sup_closed == (hi != INF))


def ends(x):
    """`None` for empty, else the set's `(lo, hi)`"""
    s = x.to_set() if x.decoration is None else x.to_set().interval
    return None if s.is_empty else (s.inf, s.sup)


def new_dec(x) -> Decoration:
    """1788's newDec of a 1788-form interval: trv if empty, com if bounded, dac if not"""
    e = ends(x)
    if e is None:
        return Decoration.TRV
    return Decoration.COM if -INF < e[0] and e[1] < INF else Decoration.DAC


def is_down(r: float, exact) -> bool:
    """r is the largest double <= exact (-inf below the doubles), from the definition"""
    if r == -INF:
        return exact == -INF or exact < -MAX
    above = math.nextafter(r, INF)
    return r != INF and Fraction(r) <= exact and (above == INF or exact < Fraction(above))


def is_up(r: float, exact) -> bool:
    return is_down(-r, -exact)


def down(v) -> float:
    """the largest double <= v, for an exact finite v: the float nearest, stepped down if above"""
    try:
        f = float(v)
    except OverflowError:
        return MAX if v > 0 else -INF
    return math.nextafter(f, -INF) if Fraction(f) > v else f


def up(v) -> float:
    return -down(-v)


# STRATEGIES: 1788-form intervals, of one flavour per example

DOUBLES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, 0.1, TINY, -TINY, 2.2250738585072014e-308,
                     MAX, -MAX, 1e300, -1e300]),
    st.floats(-1e6, 1e6, allow_nan=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
ENDS = st.one_of(DOUBLES, st.sampled_from([-INF, INF]))


@st.composite
def intervals_1788(draw, decorated=False, bounded=False):
    """a 1788-form interval: empty, or [lo, hi] of doubles, ±inf only as open ends"""
    if draw(st.integers(0, 9)) == 0:
        return Interval(decoration='trv' if decorated else None)
    a, b = sorted((draw(DOUBLES if bounded else ENDS), draw(DOUBLES if bounded else ENDS)))
    if a == INF or b == -INF:
        a, b = -1.0, 1.0
    x = Interval(a, b)
    if decorated:
        fits = [d for d in Decoration if d <= new_dec(x)]
        x = Interval(a, b, draw(st.sampled_from(fits)))
    return x


flavours = st.booleans()


def bare_and(x):
    """the interval part of x, bare"""
    return x if x.decoration is None else ieee1788.interval_part(x)


def quietly(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return fn(*args)


# EACH FUNCTION IS THE LIBRARY'S OP, IN 1788'S FORM
# a table written apart from the module's: the 1788 name, the operands' kinds (`I` an interval, `n`
# an int exponent, `r` a non-zero int root), the library call on the operands' sets

def _method(name):
    return lambda a: getattr(a, name)()


ELEMENTARY = ('sqrt', 'exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos',
              'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'cbrt', 'cot', 'sec', 'csc', 'acot', 'coth',
              'csch', 'sech', 'acoth', 'sign', 'ceil', 'floor', 'trunc')
LIBRARY = [
    ('pos', 'I', lambda a: +a), ('neg', 'I', lambda a: -a), ('abs', 'I', abs),
    ('add', 'II', lambda a, b: a + b), ('sub', 'II', lambda a, b: a - b), ('mul', 'II', lambda a, b: a * b),
    ('div', 'II', lambda a, b: a / b), ('recip', 'I', lambda a: a.reciprocal()), ('sqr', 'I', lambda a: a ** 2),
    ('fma', 'III', lambda a, b, c: a.fma(b, c)), ('min', 'II', lambda a, b: a.minimum(b)),
    ('max', 'II', lambda a, b: a.maximum(b)), ('pown', 'In', lambda a, n: a ** n), ('pow', 'II', lambda a, b: a ** b),
    ('rootn', 'Ir', lambda a, n: a.rootn(n)), ('hypot', 'II', lambda a, b: a.hypot(b)),
    ('atan2', 'II', lambda y, x: y.atan2(x)), ('logp1', 'I', lambda a: a.log1p()),
    ('roundTiesToEven', 'I', lambda a: round(a)), ('roundTiesToAway', 'I', lambda a: a.round_ties_away()),
    *((name, 'I', _method(name)) for name in ELEMENTARY),
    ('sqrRev', 'I', reverse.sqr_rev), ('sqrRev', 'II', reverse.sqr_rev),
    ('absRev', 'I', reverse.abs_rev), ('absRev', 'II', reverse.abs_rev),
    ('pownRev', 'In', reverse.pown_rev), ('pownRev', 'InI', reverse.pown_rev),
    ('coshRev', 'I', reverse.cosh_rev), ('coshRev', 'II', reverse.cosh_rev),
    ('sinRev', 'I', reverse.sin_rev), ('sinRev', 'II', reverse.sin_rev),
    ('cosRev', 'I', reverse.cos_rev), ('cosRev', 'II', reverse.cos_rev),
    ('tanRev', 'I', reverse.tan_rev), ('tanRev', 'II', reverse.tan_rev),
    ('mulRev', 'II', reverse.mul_rev), ('mulRev', 'III', reverse.mul_rev),
    ('powRev1', 'II', reverse.pow_rev1), ('powRev1', 'III', reverse.pow_rev1),
    ('powRev2', 'II', reverse.pow_rev2), ('powRev2', 'III', reverse.pow_rev2),
    ('intersection', 'II', lambda a, b: a & b), ('convexHull', 'II', lambda a, b: (a | b).hull),
]
# the periodic reverse ops over a huge x list up to 1000 branches (D12), so their x is kept small
_SMALL_X = {'sinRev', 'cosRev', 'tanRev'}


@st.composite
def operands(draw, kinds, name):
    decorated = draw(flavours)
    out = []
    for i, kind in enumerate(kinds):
        if kind == 'I':
            small = name in _SMALL_X and i == 1
            out.append(draw(intervals_1788(decorated, bounded=small)))
        elif kind == 'n':
            out.append(draw(st.integers(-4, 4)))
        else:
            out.append(draw(st.sampled_from([-3, -2, -1, 1, 2, 3])))
    return out


def library_sets(args):
    return [a.to_set() if isinstance(a, Interval) else a for a in args]


@pytest.mark.parametrize('name, kinds, library', LIBRARY, ids=[f'{n} {k}' for n, k, _ in LIBRARY])
@settings(max_examples=15, deadline=None)
@given(data=st.data())
def test_each_function_is_the_library_op_in_1788_form(name, kinds, library, data):
    """the layer's function of a 1788 name is `from_set` of the library op this table names for it, on
    the operands' sets, and in the 1788 form, with no warning escaping (the suite makes one an error)"""
    args = data.draw(operands(kinds, name))
    x = ieee1788.NAMES[name](*args)
    assert is_1788_form(x), x
    assert x == ieee1788.from_set(quietly(library, *library_sets(args)))


# THE OUTPUT RULE

library_values = st.one_of(endpoint_values, st.sampled_from([
    2 ** 1024, -2 ** 1024, Fraction(1, 10), -Fraction(1, 3), MAX, -MAX, TINY, Fraction(TINY) / 3]))


def real_bounds(s: MultiInterval):
    """the infimum and supremum of the set's real points, or None: ±inf are never real points"""
    reals = [(lo, hi) for lo, _, hi, _ in kernel.pieces(s.cuts) if not (lo == hi and lo in (-INF, INF))]
    if not reals:
        return None
    return min(lo for lo, _ in reals), max(hi for _, hi in reals)


@given(cut_tuples(values=library_values), st.booleans(), st.sampled_from(list(Decoration)))
@example(kernel.normalize([kernel.piece(1, 2), kernel.piece(INF, INF)]), False, Decoration.COM)
@example(kernel.normalize([kernel.piece(-INF, -INF)]), True, Decoration.TRV)
@example(kernel.normalize([kernel.piece(-INF, 0, True, True)]), True, Decoration.TRV)
def test_from_set_is_the_output_rule(cuts, outward, decoration):
    """empty iff the set has no real point; else the largest double <= the real points' infimum and the
    smallest >= their supremum, open at ±inf. decorated: min(the set's decoration, newDec)"""
    s = (OutwardMultiInterval if outward else MultiInterval).from_cuts(cuts)
    x = ieee1788.from_set(s)
    assert is_1788_form(x) and x.decoration is None
    bounds = real_bounds(s)
    if bounds is None:
        assert ends(x) is None
    else:
        lo, hi = ends(x)
        assert is_down(lo, bounds[0]) and is_up(hi, bounds[1]), (s, x)
    fitting = min(decoration, DecoratedInterval(s).decoration)
    d = ieee1788.from_set(DecoratedInterval(s, fitting))
    assert ends(d) == ends(x) and d.decoration == min(fitting, new_dec(d))


@pytest.mark.parametrize('s, want', [
    (MultiInterval.parse('[1, 2] | [inf]'), Interval(1.0, 2.0)),
    (MultiInterval(-INF), Interval()),
    (OutwardMultiInterval(INF), Interval()),
    (MultiInterval.parse('{ [-inf] , [inf] }'), Interval()),
    (MultiInterval(2 ** 1024), Interval(MAX, INF)),
    (MultiInterval(Fraction(1, 10)), Interval(0.09999999999999999, 0.1)),
    (MultiInterval.parse('[-inf, 0]'), Interval(-INF, 0.0)),
    (DecoratedInterval(MultiInterval(6, 2 + Fraction(MAX)), 'com'), Interval(6.0, INF, 'dac')),
    (DecoratedInterval(MultiInterval.parse('[inf]'), 'trv'), Interval(decoration='trv')),
])
def test_from_set_examples(s, want):
    assert ieee1788.from_set(s) == want


def test_from_set_refuses_other_values():
    for value in (1.0, Interval(1.0), (1, 2), None):
        with pytest.raises(TypeError):
            ieee1788.from_set(value)


def test_the_drop_is_explicit():
    """1788's functions are functions of reals: a lone attained infinity is no answer (`log` of
    `(-inf, 0]` and `atanh` of `[1, inf)` are empty in 1788), decided by the output rule, not by a
    constructor that happens to empty `[inf]` opened"""
    assert ieee1788.log(Interval(-INF, 0.0)) == Interval()
    assert ieee1788.atanh(Interval(1.0, INF)) == Interval()
    assert ieee1788.pown_rev(Interval(0.0), -2) == Interval()
    assert ieee1788.from_set(MultiInterval.parse('[0, 1] | [inf]')) == Interval(0.0, 1.0)


# FLAVOURS AND OPERANDS

BINARY = ['add', 'sub', 'mul', 'div', 'min_', 'max_', 'pow_', 'hypot', 'atan2', 'intersection', 'convex_hull',
          'cancel_minus', 'cancel_plus', 'mul_rev', 'pow_rev1', 'pow_rev2', 'mul_rev_to_pair', 'equal', 'subset',
          'disjoint', 'less', 'strict_less', 'precedes', 'strict_precedes', 'interior', 'overlap']


@pytest.mark.parametrize('name', BINARY)
def test_flavours(name):
    """1788 has no mixed operations: a bare and a decorated operand are a TypeError, either way round.
    a number is a point of the other operand's flavour: newDec's beside a decorated one"""
    f = getattr(ieee1788, name)
    bare, decorated = Interval(1.0, 2.0), Interval(1.0, 2.0, 'com')
    for a, b in ((bare, decorated), (decorated, bare)):
        with pytest.raises(TypeError):
            f(a, b)
    assert f(decorated, 2.0) == f(decorated, Interval(2.0, 2.0, 'com'))
    assert f(2.0, decorated) == f(Interval(2.0, 2.0, 'com'), decorated)
    assert f(bare, 2.0) == f(bare, Interval(2.0))
    assert f(Fraction(1, 10), bare) == f(Interval(Fraction(1, 10)), bare)


def test_a_call_of_numbers_alone_is_bare():
    """no operand is a decorated interval, so the call is bare (the review's S1): each result, both
    of the pair's included, has no decoration"""
    results = [ieee1788.add(1, 2), ieee1788.sqrt(4.0), ieee1788.fma(1, 2, 3), *ieee1788.mul_rev_to_pair(2.0, 4.0)]
    assert results == [Interval(3.0), Interval(2.0), Interval(5.0), Interval(2.0), Interval()]
    assert [r.decoration for r in results] == [None] * 5


LIBRARY_VALUES = [MultiInterval(1, 2), OutwardMultiInterval(1.0, 2.0), DecoratedInterval(MultiInterval(1, 2))]


@pytest.mark.parametrize('name', BINARY + ['neg', 'sqrt', 'inf', 'mid', 'is_empty', 'new_dec', 'set_dec'])
@pytest.mark.parametrize('value', LIBRARY_VALUES, ids=['multi', 'outward', 'decorated'])
def test_library_values_are_not_operands(name, value):
    """critique n6: a library value would bypass the input rule (a multi-piece set, an attained inf),
    so it is a TypeError, and the operators return NotImplemented"""
    f = getattr(ieee1788, name)
    with pytest.raises(TypeError):
        if name == 'set_dec':
            f(value, 'com')
        elif name in BINARY:
            f(Interval(1.0, 2.0), value)
        else:
            f(value)


@pytest.mark.parametrize('value', LIBRARY_VALUES, ids=['multi', 'outward', 'decorated'])
def test_operators_refuse_library_values(value):
    x = Interval(1.0, 2.0)
    for dunder in ('__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__',
                   '__rtruediv__', '__pow__', '__rpow__', '__and__', '__or__'):
        assert getattr(x, dunder)(value) is NotImplemented, dunder
    for op in (lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a ** b, lambda a, b: a & b):
        for a, b in ((x, value), (value, x)):
            with pytest.raises(TypeError):
                op(a, b)
    with pytest.raises(TypeError):
        value in x  # noqa: B015


# CANCELLATION: 1788's rule, written out

def cancel_1788(a: Interval, b: Interval) -> Interval:
    """1788-2015 §9.2 as its vectors spell it (`libieeep1788_cancel.itl`), on bare intervals"""
    ea, eb = ends(a), ends(b)

    def bounded(e):
        return e is None or (-INF < e[0] and e[1] < INF)

    if ea is None and bounded(eb):
        return Interval()
    if ea is not None and eb is not None and bounded(ea) and bounded(eb):
        if Fraction(ea[1]) - Fraction(ea[0]) >= Fraction(eb[1]) - Fraction(eb[0]):
            return Interval(down(Fraction(ea[0]) - Fraction(eb[0])), up(Fraction(ea[1]) - Fraction(eb[1])))
    return Interval(-INF, INF)


@st.composite
def cancel_operands(draw):
    a = draw(intervals_1788())
    kind = draw(st.integers(0, 4))
    if kind == 0 and ends(a) is not None:
        b = a  # equal widths
    elif kind == 1:  # a shift of a small-integer a: equal widths, exactly
        lo, width, shift = draw(st.integers(-8, 8)), draw(st.integers(0, 6)), draw(st.integers(-5, 5))
        a, b = Interval(float(lo), float(lo + width)), Interval(float(lo + shift), float(lo + width + shift))
    elif kind == 2:  # widths equal as floats but not exactly (the review's F1, S6): big - small and
        # big + small round to big, since small < ulp(big) / 2
        big, small = draw(st.floats(2.0 ** 54, 2.0 ** 70)), draw(st.floats(TINY, 0.5))
        a, b = draw(st.sampled_from([(Interval(small, big), Interval(0.0, big)),
                                     (Interval(0.0, big), Interval(-small, big)),
                                     (Interval(0.0, big), Interval(small, big))]))
    else:
        b = draw(intervals_1788())
    return a, b


@settings(max_examples=300, deadline=None)
@given(cancel_operands(), flavours)
@example((Interval(-10.0, 5.0), Interval(-10.0, 5.0)), False)
@example((Interval(), Interval(1.0, INF)), False)
@example((Interval(), Interval()), True)
@example((Interval(1.0, 5.0), Interval(1.0, 5.1)), False)
# the widths compared exactly (the review's F1, S6): wid a < wid b, but not in floats
@example((Interval(0.1, 1e17), Interval(0.0, 1e17)), False)  # 1e17 - 0.1 rounds to 1e17
@example((Interval(0.0, 1e16), Interval(-1e16, 1.0)), False)  # 1e16 + 1 rounds to 1e16
@example((Interval(-MAX, math.nextafter(MAX, 0)), Interval(-MAX, MAX)), False)  # both overflow
def test_cancel_minus_is_1788s(ab, decorated):
    a, b = ab
    want = cancel_1788(a, b)
    if decorated:
        a, b = ieee1788.new_dec(a), ieee1788.new_dec(b)
    x = ieee1788.cancel_minus(a, b)
    assert is_1788_form(x)
    assert ends(x) == ends(want), (a, b, x, want)
    assert x.decoration == (Decoration.TRV if decorated else None)
    assert ieee1788.cancel_plus(a, b) == ieee1788.cancel_minus(a, ieee1788.neg(b))


# OVERLAP: 1788's sixteen states, against a table on the signs of the ends' comparisons

def _sign(u, v) -> int:
    return (u > v) - (u < v)


# (sign(a1 - b1), sign(a2 - b2), sign(a2 - b1), sign(a1 - b2)) -> the state, from 1788-2015's table
SIGNS = {
    (-1, -1, -1, -1): 'before', (1, 1, 1, 1): 'after',
    (-1, -1, 0, -1): 'meets', (1, 1, 1, 0): 'metBy',
    (-1, -1, 1, -1): 'overlaps', (1, 1, 1, -1): 'overlappedBy',
    (0, -1, 0, -1): 'starts', (0, -1, 1, -1): 'starts',
    (0, 1, 1, 0): 'startedBy', (0, 1, 1, -1): 'startedBy',
    (1, -1, 1, -1): 'containedBy', (-1, 1, 1, -1): 'contains',
    (1, 0, 1, 0): 'finishes', (1, 0, 1, -1): 'finishes',
    (-1, 0, 0, -1): 'finishedBy', (-1, 0, 1, -1): 'finishedBy',
    (0, 0, 0, 0): 'equals', (0, 0, 1, -1): 'equals',
}
CONVERSE = {'before': 'after', 'meets': 'metBy', 'overlaps': 'overlappedBy', 'starts': 'startedBy',
            'containedBy': 'contains', 'finishes': 'finishedBy', 'equals': 'equals',
            'firstEmpty': 'secondEmpty', 'bothEmpty': 'bothEmpty'}
CONVERSE.update({v: k for k, v in CONVERSE.items()})
GRID_ENDS = (-INF, -2.0, -1.0, 0.0, 1.0, 2.0, INF)
GRID = [Interval()] + [Interval(a, b) for i, a in enumerate(GRID_ENDS) for b in GRID_ENDS[i:]
                       if a != INF and b != -INF]


def overlap_oracle(a, b) -> str:
    ea, eb = ends(a), ends(b)
    if ea is None or eb is None:
        return 'bothEmpty' if ea is None and eb is None else 'firstEmpty' if ea is None else 'secondEmpty'
    (a1, a2), (b1, b2) = ea, eb
    return SIGNS[_sign(a1, b1), _sign(a2, b2), _sign(a2, b1), _sign(a1, b2)]


def test_overlap_on_the_grid():
    """every pair of 1788-form intervals with ends in {-inf, -2, -1, 0, 1, 2, inf}, and empty (27 x 27 =
    729 pairs): the state is the table's, its value is 1788's name, and overlap(b, a) is the converse"""
    assert len(GRID) == 27
    seen = set()
    for a in GRID:
        for b in GRID:
            state = ieee1788.overlap(a, b)
            assert isinstance(state, ieee1788.Overlap) and state.value == overlap_oracle(a, b), (a, b, state)
            assert ieee1788.overlap(b, a).value == CONVERSE[state.value]
            seen.add(state)
    assert seen == set(ieee1788.Overlap) and len(ieee1788.Overlap) == 16


@pytest.mark.parametrize('a, b, want', [
    ((1.0, 2.0), (2.0, 3.0), 'meets'), ((1.0, 1.0), (1.0, 3.0), 'starts'), ((0.0, 2.0), (2.0, 2.0), 'finishedBy'),
    ((2.0, 3.0), (1.0, 2.0), 'metBy'), ((-INF, 2.0), (2.0, 3.0), 'meets'), ((1.0, 3.0), (1.0, 2.0), 'startedBy'),
])
def test_overlap_examples(a, b, want):
    assert ieee1788.overlap(Interval(*a), Interval(*b)) == ieee1788.Overlap(want)
    # a decorated pair is read on its interval parts
    assert ieee1788.overlap(ieee1788.new_dec(Interval(*a)), ieee1788.new_dec(Interval(*b))).value == want


# MUL_REV_TO_PAIR

def real_pieces(s):
    """the library set's pieces holding a real point, each as a MultiInterval, in order"""
    return [p for p in s if real_bounds(p) is not None]


@settings(max_examples=300, deadline=None)
@given(intervals_1788(), intervals_1788())
@example(Interval(-INF, INF), Interval(-2.1, -0.4))
@example(Interval(-2.0, 0.0), Interval(-2.1, -0.4))
@example(Interval(0.0), Interval(1.0, 2.0))
def test_mul_rev_to_pair_bare(b, c):
    """the pair is the library's mul_rev, piece by piece: each non-empty member `from_set` of one piece
    with a real point, in increasing order, the second empty unless there are two, never more"""
    pair = ieee1788.mul_rev_to_pair(b, c)
    assert isinstance(pair, tuple) and len(pair) == 2 and all(map(is_1788_form, pair))
    pieces = real_pieces(quietly(reverse.mul_rev, b.to_set(), c.to_set()))
    assert len(pieces) <= 2
    want = [ieee1788.from_set(p) for p in pieces] + [Interval()] * (2 - len(pieces))
    assert list(pair) == want


@st.composite
def decorated_pair_operands(draw):
    """decorated b and c, com drawn often (critique n15: the design's measurement drew dac only)"""
    b, c = draw(intervals_1788()), draw(intervals_1788())
    return tuple(Interval(*ends(x), draw(st.sampled_from([d for d in Decoration if d <= new_dec(x)])))
                 if ends(x) is not None else Interval(decoration='trv') for x in (b, c))


@settings(max_examples=300, deadline=None)
@given(decorated_pair_operands())
@example((Interval(-2.0, -0.1, 'com'), Interval(-2.1, -0.4, 'com')))
@example((Interval(-2.0, 0.0, 'dac'), Interval(-2.1, -0.4, 'com')))
@example((Interval(1.0, INF, 'dac'), Interval(1.0, 2.0, 'com')))
def test_mul_rev_to_pair_decorated(bc):
    """0 not in b: (div(c, b), [empty]_trv), 1788's wording; 0 in b: both trv. the sets are the bare
    pair's. the law of the design's 5.1 (where 0 is not in b the pieces of mul_rev are div's set) is
    drawn by ::test_mul_rev_to_pair_bare, against the library's pieces: here both sides of the div
    assertions come from the layer's div branch, so they pin the decoration and the branch only (the
    review's m4)"""
    b, c = bc
    first, second = ieee1788.mul_rev_to_pair(b, c)
    bare = ieee1788.mul_rev_to_pair(bare_and(b), bare_and(c))
    assert (ends(first), ends(second)) == tuple(map(ends, bare))
    if ends(b) is None or not ends(b)[0] <= 0 <= ends(b)[1]:
        assert first == ieee1788.div(c, b) and second == Interval(decoration='trv')
        assert ends(bare[0]) == ends(ieee1788.div(bare_and(c), bare_and(b))) and ends(bare[1]) is None
    else:
        assert first.decoration is Decoration.TRV and second.decoration is Decoration.TRV


def test_mul_rev_to_pair_examples():
    """1788's pair for a gap at 0 shares the bound 0 (`libieeep1788_mul_rev.itl:235`)"""
    assert ieee1788.mul_rev_to_pair(Interval(-INF, INF, 'dac'), Interval(-2.1, -0.4, 'dac')) == \
        (Interval(-INF, 0.0, 'trv'), Interval(0.0, INF, 'trv'))
    assert ieee1788.mul_rev_to_pair(Interval(-2.0, -0.1, 'dac'), Interval(-2.1, -0.4, 'dac')) == \
        (ieee1788.div(Interval(-2.1, -0.4, 'dac'), Interval(-2.0, -0.1, 'dac')), Interval(decoration='trv'))


# NUMBERS AND BOOLEANS

NUMBER_TABLE = [('mid', lambda s: s.mid()), ('rad', lambda s: s.rad()), ('wid', lambda s: s.wid()),
                ('mag', lambda s: s.mag()), ('mig', lambda s: s.mig())]
BOOLEAN_TABLE = [
    ('isEmpty', 'I', lambda a: a.is_empty), ('isEntire', 'I', lambda a: a == MultiInterval.parse('(-inf, inf)')),
    ('isSingleton', 'I', lambda a: a.is_degenerate), ('isCommonInterval', 'I', lambda a: bool(a) and a.is_finite),
    ('equal', 'II', lambda a, b: a == b), ('subset', 'II', lambda a, b: a.issubset(b)),
    ('disjoint', 'II', lambda a, b: a.isdisjoint(b)), ('less', 'II', lambda a, b: a.weakly_less(b)),
    ('strictLess', 'II', lambda a, b: a.strictly_less(b)), ('precedes', 'II', lambda a, b: (a <= b).certainly),
    ('strictPrecedes', 'II', lambda a, b: (a < b).certainly), ('interior', 'II', lambda a, b: a.within(b.interior)),
]


@settings(max_examples=200, deadline=None)
@given(intervals_1788(), flavours)
def test_numbers(x, decorated):
    """every number a python float, the library's number of the interval part (D9 rounds float ends as
    1788 does); the six numbers of the empty set raise ValueError (the default built, Q13 (b))"""
    if decorated:
        x = ieee1788.new_dec(x)
    part = bare_and(x).to_set()
    for name, library in NUMBER_TABLE:
        f = ieee1788.NAMES[name]
        if ends(x) is None:
            with pytest.raises(ValueError):
                f(x)
            continue
        n = f(x)
        assert type(n) is float and n == library(part), name
    if ends(x) is None:
        with pytest.raises(ValueError):
            ieee1788.mid_rad(x)
        assert ieee1788.inf(x) == INF and ieee1788.sup(x) == -INF
    else:
        m, r = ieee1788.mid_rad(x)
        assert type(m) is float and type(r) is float and (m, r) == part.mid_rad()
        assert ieee1788.inf(x) == ends(x)[0] and ieee1788.sup(x) == ends(x)[1]


def test_the_sign_of_zero_is_1788s():
    """critique n7, following 1788 (`libieeep1788_num.itl:34`): inf of an interval whose lower end is 0
    is -0.0, sup of one whose upper end is 0 is +0.0; the vectors cannot see it (the parser reads -0.0
    as 0), so it is pinned here"""
    for x in (Interval(0.0, 1.0), Interval(0.0, 0.0), Interval(-0.0, INF), Interval(0.0, 0.0, 'com')):
        assert ieee1788.inf(x) == 0 and math.copysign(1, ieee1788.inf(x)) == -1, x
    for x in (Interval(-1.0, 0.0), Interval(0.0, 0.0), Interval(-INF, -0.0), Interval(-1.0, 0.0, 'dac')):
        assert ieee1788.sup(x) == 0 and math.copysign(1, ieee1788.sup(x)) == 1, x
    # nothing else carries a sign: the ends themselves are the package's one zero
    assert math.copysign(1, ends(Interval(-0.0, 1.0))[0]) == 1
    assert math.copysign(1, ieee1788.mid(Interval(-1.0, 1.0))) == 1


def test_numbers_examples():
    assert type(ieee1788.mid(ieee1788.entire())) is float and ieee1788.mid(ieee1788.entire()) == 0.0
    assert ieee1788.mid(Interval(0.0, INF)) == MAX and ieee1788.rad(Interval(1.0, INF)) == INF
    assert type(ieee1788.mig(ieee1788.entire())) is float
    assert ieee1788.sum_ is intervals.sum_ and ieee1788.sum_abs is intervals.sum_abs
    assert ieee1788.sum_square is intervals.sum_sqr and ieee1788.dot is intervals.dot


@pytest.mark.parametrize('name, kinds, library', BOOLEAN_TABLE, ids=[n for n, _, _ in BOOLEAN_TABLE])
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_booleans(name, kinds, library, data):
    """each boolean is the library's relation on the interval parts, a bool"""
    decorated = data.draw(flavours)
    args = [data.draw(intervals_1788(decorated)) for _ in kinds]
    got = ieee1788.NAMES[name](*args)
    assert type(got) is bool and got == library(*(bare_and(a).to_set() for a in args))


def test_booleans_examples():
    e = ieee1788.entire()
    assert ieee1788.is_entire(e) and not ieee1788.is_entire(Interval(-INF, MAX))
    assert ieee1788.interior(Interval(1.0, INF), Interval(0.0, INF))
    assert ieee1788.equal(Interval(1.0, 2.0, 'com'), Interval(1.0, 2.0, 'def'))
    assert Interval(1.0, 2.0, 'com') != Interval(1.0, 2.0, 'def')


@pytest.mark.parametrize('m, want', [(1.0, True), (Fraction(3, 2), True), (3, False), (INF, False), (-INF, False),
                                     (math.nan, False)])
def test_is_member(m, want):
    """critique n13: 1788's isMember of nan is false (the library's `in` refuses nothing, but a nan
    point is never a member); ±inf are never members of a 1788 interval"""
    assert ieee1788.is_member(m, Interval(1.0, 2.0)) is want
    assert ieee1788.is_member(m, ieee1788.entire()) is (want or m in (3,))
    with pytest.raises(TypeError):
        ieee1788.is_member(Interval(1.0), Interval(1.0, 2.0))


# CONSTRUCTORS AND DECORATIONS

def test_constructors():
    assert Interval(Fraction(1, 10)) == Interval(0.09999999999999999, 0.1)
    assert ends(Interval(0.1)) == (0.1, 0.1)
    assert Interval(1, 2) == Interval(1.0, 2.0) and type(ends(Interval(1, 2))[0]) is float
    assert Interval(2 ** 1024) == Interval(MAX, INF) and Interval(1, 2 ** 1024) == Interval(1.0, INF)
    assert Interval() == ieee1788.empty() and Interval(-INF, INF) == ieee1788.entire()
    assert ieee1788.nums_to_interval(1, 2) == Interval(1.0, 2.0)
    assert ieee1788.nums_to_decorated_interval(1, 2) == Interval(1.0, 2.0, 'com')
    assert ieee1788.nums_to_decorated_interval(-INF, 2) == Interval(-INF, 2.0, 'dac')
    assert ieee1788.text_to_interval('[0.1]') == Interval(0.09999999999999999, 0.1)
    assert ieee1788.text_to_decorated_interval('[1.0E+400]_com') == Interval(MAX, INF, 'dac')
    assert ieee1788.text_to_decorated_interval('[1, 2]') == Interval(1.0, 2.0, 'com')
    assert Interval(decoration='trv') == ieee1788.text_to_decorated_interval('[empty]')
    assert Interval(1.0, 2.0, Decoration.DEF).decoration is Decoration.DEF


@pytest.mark.parametrize('args, error', [
    ((2, 1), UndefinedOperationError),
    ((1, 2 ** 1024, 'com'), UndefinedOperationError),  # binary64: its hull is [1.0, inf), unbounded
    ((-INF, INF, 'com'), UndefinedOperationError),
    ((None, None, 'com'), UndefinedOperationError),
    ((1.0, 2.0, 'ill'), UndefinedOperationError),
    ((INF,), UndefinedOperationError),
    ((math.nan,), UndefinedOperationError),
    ((None, 2.0), TypeError),  # an upper bound without a lower (critique n13)
    (('1',), TypeError),
    ((True,), TypeError),
    ((1.0, 2.0, 7), TypeError),
    ((MultiInterval(1),), TypeError),
])
def test_constructor_refusals(args, error):
    """`Interval(lo, hi, d)` is strict, as `DecoratedInterval(x, d)` and a literal `[1,]_com` are, and
    decided on the binary64 result; `set_dec` is the forgiving one"""
    with pytest.raises(error):
        Interval(*args)


def test_decorations():
    assert ieee1788.set_dec(Interval(1.0, INF), 'com') == Interval(1.0, INF, 'dac')
    assert ieee1788.set_dec(Interval(), 'def') == Interval(decoration='trv')
    assert ieee1788.set_dec(Interval(1.0, 2.0), Decoration.DEF) == Interval(1.0, 2.0, 'def')
    assert ieee1788.new_dec(Interval(1.0, 2.0)) == Interval(1.0, 2.0, 'com')
    assert ieee1788.new_dec(Interval(1.0, INF)) == Interval(1.0, INF, 'dac')
    assert ieee1788.interval_part(Interval(1.0, 2.0, 'def')) == Interval(1.0, 2.0)
    assert ieee1788.decoration_part(Interval(1.0, 2.0, 'def')) is Decoration.DEF
    with pytest.raises(UndefinedOperationError):
        ieee1788.set_dec(Interval(1.0, 2.0), 'ill')
    # critique n13: each takes the one flavour 1788 gives it
    decorated, bare = Interval(1.0, 2.0, 'com'), Interval(1.0, 2.0)
    for f, x in ((ieee1788.set_dec, decorated), (ieee1788.new_dec, decorated), (ieee1788.interval_part, bare),
                 (ieee1788.decoration_part, bare)):
        with pytest.raises(TypeError):
            f(x, 'com') if f is ieee1788.set_dec else f(x)


@settings(max_examples=200, deadline=None)
@given(intervals_1788(), st.sampled_from(list(Decoration)))
def test_set_dec_and_new_dec(x, d):
    """setDec demotes to newDec's where d does not fit; newDec is the best that fits"""
    assert ieee1788.new_dec(x).decoration == new_dec(x)
    got = ieee1788.set_dec(x, d)
    assert ends(got) == ends(x) and got.decoration == min(d, new_dec(x))
    assert ieee1788.interval_part(got) == x


# WARNINGS

@pytest.mark.parametrize('call, want', [
    (lambda: ieee1788.div(Interval(0.0), Interval(0.0)), Interval()),
    (lambda: ieee1788.recip(Interval(0.0)), Interval()),
    (lambda: ieee1788.sqrt(Interval(-1.0, 4.0)), Interval(0.0, 2.0)),
    (lambda: ieee1788.sin_rev(Interval(0.0, 1.0)), ieee1788.entire()),  # x omitted: a HullWarning inside
    (lambda: ieee1788.floor(ieee1788.entire()), ieee1788.entire()),
    (lambda: ieee1788.add(ieee1788.empty(), Interval(1.0)), Interval()),
    (lambda: ieee1788.sqrt(Interval(-1.0, 4.0, 'com')), Interval(0.0, 2.0, 'trv')),
])
def test_the_library_s_warnings_are_silent(call, want):
    """1788 signals none of them (`[0] / [0]` is empty with no signal)"""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert call() == want


def test_possibly_undefined_operation_reaches_the_caller(monkeypatch):
    """the one library warning with a 1788 meaning goes through the layer: the silencing must not take
    the IntervalWarning base class"""
    real_sqrt = OutwardMultiInterval.sqrt

    def sqrt(self):
        warnings.warn('possibly', PossiblyUndefinedOperationWarning)
        return real_sqrt(self)

    monkeypatch.setattr(OutwardMultiInterval, 'sqrt', sqrt)
    with pytest.warns(PossiblyUndefinedOperationWarning):
        assert ieee1788.sqrt(Interval(4.0)) == Interval(2.0)


def test_the_silencing_does_not_leak():
    """after a layer call the filters are as they were, and a direct library call still warns (a global
    `simplefilter` without `catch_warnings` would pass every other test here)"""
    before = list(warnings.filters)
    ieee1788.sqrt(Interval(-1.0, 4.0))
    ieee1788.div(Interval(0.0), Interval(0.0))
    assert warnings.filters == before
    with pytest.raises(DomainClippedWarning):  # the suite's filter makes the library's warning an error
        OutwardMultiInterval(-1.0, 4.0).sqrt()


# NAMES

NAMES_1788 = {
    'numsToInterval', 'textToInterval', 'd-numsToInterval', 'd-textToInterval', 'empty', 'entire',
    'newDec', 'setDec', 'intervalPart', 'decorationPart',
    'pos', 'neg', 'add', 'sub', 'mul', 'div', 'recip', 'sqr', 'sqrt', 'fma', 'abs', 'min', 'max', 'pown', 'pow',
    'rootn', 'hypot',
    'exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'logp1', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'cbrt', 'cot', 'sec', 'csc', 'acot', 'coth', 'csch',
    'sech', 'acoth',
    'sign', 'ceil', 'floor', 'trunc', 'roundTiesToEven', 'roundTiesToAway',
    'sqrRev', 'absRev', 'pownRev', 'sinRev', 'cosRev', 'tanRev', 'coshRev', 'mulRev', 'powRev1', 'powRev2',
    'mulRevToPair',
    'cancelMinus', 'cancelPlus', 'intersection', 'convexHull',
    'inf', 'sup', 'mid', 'wid', 'rad', 'mag', 'mig', 'midRad',
    'isEmpty', 'isEntire', 'isMember', 'isSingleton', 'isCommonInterval', 'equal', 'subset', 'disjoint', 'less',
    'strictLess', 'precedes', 'strictPrecedes', 'interior',
    'overlap', 'sum', 'sumAbs', 'sumSquare', 'dot',
}


def snake(name: str) -> str:
    """1788's camelCase to snake_case, a trailing underscore on a python builtin"""
    out = re.sub(r'([A-Z])', lambda m: '_' + m.group(1).lower(), name)
    return out + '_' if out in vars(builtins) else out


def test_names():
    """NAMES is 1788's 104 names, each bound to the function the transliteration names (the decorated
    constructors, 1788's second flavour, are `*_to_decorated_interval`)"""
    assert set(ieee1788.NAMES) == NAMES_1788 and len(NAMES_1788) == 104
    decorated = {'d-numsToInterval': 'nums_to_decorated_interval', 'd-textToInterval': 'text_to_decorated_interval'}
    for name, f in ieee1788.NAMES.items():
        assert getattr(ieee1788, decorated.get(name) or snake(name)) is f, name
    assert snake('mulRevToPair') == 'mul_rev_to_pair' and snake('sum') == 'sum_' and snake('powRev1') == 'pow_rev1'


def test_not_exported():
    """`from intervals import ieee1788` imports it; `import intervals` does not, and it is not in
    `__all__` (the default built, Q13 (a))"""
    assert 'ieee1788' not in intervals.__all__
    code = 'import sys, intervals; print("intervals.ieee1788" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == 'False'


# THE CLASS

@settings(max_examples=200, deadline=None)
@given(intervals_1788(), flavours)
@example(Interval(-INF, 2.0), False)
@example(Interval(1.0, INF), True)
@example(Interval(TINY, MAX), True)
def test_repr_str_and_to_set(x, decorated):
    """repr evaluates back, ±inf ends included (critique n5); str is a 1788 literal that reads back as an
    enclosure within one double at each end; from_set(x.to_set()) is x"""
    if decorated:
        x = ieee1788.new_dec(x)
    assert eval(repr(x), {'Interval': Interval}) == x
    assert ieee1788.from_set(x.to_set()) == x
    back = (ieee1788.text_to_interval if x.decoration is None else ieee1788.text_to_decorated_interval)(str(x))
    assert back.decoration == x.decoration
    if ends(x) is None:
        assert ends(back) is None
    else:
        (lo, hi), (blo, bhi) = ends(x), ends(back)
        assert blo in (lo, math.nextafter(lo, -INF)) and bhi in (hi, math.nextafter(hi, INF)), (x, back)


def test_repr_and_str_examples():
    assert repr(Interval(1, 2)) == 'Interval(1.0, 2.0)'
    assert repr(Interval(1, 2, 'com')) == "Interval(1.0, 2.0, 'com')"
    assert repr(Interval()) == 'Interval()' and repr(Interval(decoration='trv')) == "Interval(decoration='trv')"
    assert repr(Interval(-INF, 2.0)) == "Interval(float('-inf'), 2.0)"
    assert str(Interval(0.1, 2.0, 'com')) == '[0.1, 2.0]_com' and str(Interval(decoration='trv')) == '[empty]_trv'
    assert str(ieee1788.entire()) == '[entire]' and str(Interval(-INF, 2.0)) == '[-inf, 2.0]'


@settings(max_examples=100, deadline=None)
@given(intervals_1788(), intervals_1788(), flavours, st.sampled_from([2, -1, 0, 3]))
def test_operators_are_the_functions(x, y, decorated, n):
    if decorated:
        x, y = ieee1788.new_dec(x), ieee1788.new_dec(y)
    f = ieee1788
    assert x + y == f.add(x, y) and x - y == f.sub(x, y) and x * y == f.mul(x, y) and x / y == f.div(x, y)
    assert 2 + x == f.add(2, x) and 2 - x == f.sub(2, x) and x * 0.5 == f.mul(x, 0.5) and 1 / x == f.div(1, x)
    assert -x == f.neg(x) and +x == f.pos(x) and abs(x) == f.abs_(x)
    assert x & y == f.intersection(x, y) and x | y == f.convex_hull(x, y)
    # D11: an integral real exponent is pown, any other real and an interval exponent pow
    assert x ** n == f.pown(x, n) and x ** float(n) == f.pown(x, n) and x ** Fraction(n) == f.pown(x, n)
    assert x ** 0.5 == f.pow_(x, 0.5) and x ** y == f.pow_(x, y) and 2 ** x == f.pow_(2, x)
    assert (1.0 in x) == f.is_member(1.0, x) and bool(x) == (not f.is_empty(x))


def test_the_class():
    x, y = Interval(1.0, 2.0), Interval(1.0, 2.0, 'com')
    for op in (lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a > b, lambda a, b: a >= b):
        with pytest.raises(TypeError):
            op(x, x)
    assert x != y and hash(x) == hash(Interval(1, 2)) and x == Interval(1, 2)
    assert len({x, Interval(1, 2), y, Interval(1.0, 2.0, Decoration.COM)}) == 2
    assert x != x.to_set() and y != y.to_set()
    with pytest.raises(AttributeError):
        x._set = Interval(3.0)._set
    with pytest.raises(AttributeError):  # the review's S5
        del x._set
    assert x == Interval(1.0, 2.0)
    assert Interval.__array_ufunc__ is None
    assert pickle.loads(pickle.dumps(y)) == y
    assert not Interval() and x
    with pytest.raises(TypeError):
        pow(x, 2, 3)


@pytest.mark.parametrize('call', [
    lambda: ieee1788.pown(Interval(1.0, 2.0), 2.0),
    lambda: ieee1788.pown(Interval(1.0, 2.0), True),
    lambda: ieee1788.rootn(Interval(1.0, 2.0), 2.0),
    lambda: ieee1788.pown_rev(Interval(1.0, 2.0), 2.0),
    lambda: ieee1788.pown(Interval(1.0, 2.0), Interval(2.0)),
])
def test_exponents_are_ints(call):
    """1788's pown, rootn and pownRev take an integer (critique n2); `x ** 2.0` is D11's pown"""
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize('x', [Interval(), Interval(1.0, 4.0), Interval(decoration='trv'), Interval(1.0, 4.0, 'com')])
def test_rootn_of_degree_0_raises(x):
    """the library's rule (`v2-plan.md`: rootn for every int n other than 0), kept by the layer for
    every x, the empty set included (the review's F3); `pown_rev(c, 0)` answers, as the library's"""
    with pytest.raises(ValueError, match='degree'):
        ieee1788.rootn(x, 0)
    assert ieee1788.pown_rev(Interval(1.0, 2.0), 0) == Interval(-INF, INF)
    assert ieee1788.pown_rev(Interval(2.0, 3.0), 0) == Interval()
