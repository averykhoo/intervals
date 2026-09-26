"""
the power's reverse ops (M13e, fourth part; D11): `pow_rev1(b, c, x)`, the bases `{t in x : t ** y in c for
some y in b}`, and `pow_rev2(a, c, y)`, the exponents `{s in y : t ** s in c for some t in a}`, `**` being
the library's pow (1788's, with ±inf as points where it has a limit)

* the defining property on exact operands, decided at every point where it could change (the ends of the
  operands and of the result, one double either side of each rounded end, an approximation of every
  `v ** (1/w)` or `log_t v` of the operands' ends, a point between each two, ±inf, 0 and 1): a point is in
  the result iff it fits, except inside the one-double slack of an irrational end (tightness: the double
  next to a rounded end, inward, fits). "fits" is decided here from the definitions: the other operand's
  pieces map monotonically onto intervals whose ends are compared with `c`'s exactly, `t ** (p/q)` against
  `v` as `t ** p` against `v ** q` where that is small, and with arb (`python-flint`, D14) where not
* with the library's own pow: the largest set (`T ⊆ pow_rev1(B, T ** B)`, `S ⊆ pow_rev2(A, A ** S)`, the
  points with no value aside), and soundness at sampled points, float operands and ±inf included
* isotone, distributing over unions of each operand, `x` only intersecting; the symmetries `t ** -y =
  (1/t) ** y` and `t ** -s = 1 / t ** s`; `pow_rev1([n], c)` is `pown_rev(c, n)` on `[0, inf]` and
  `pow_rev2([t], c)` is `c.log(t)`, in both classes
* float operands: an `OutwardMultiInterval` result holds the exact result of the same doubles and adds no
  double strictly inside what it adds; a `MultiInterval` one is within one double of it
* the itf1788 vectors (804, in `pow_rev.itl`) run in tests/itf1788; the ones that matter are `@example`s
  here, and the two loose ones (`_POW_REV_LOOSE_ROWS`) are checked by
  `test_pow_rev2_is_tighter_than_the_vector`
"""
import math
import random
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import EMPTY
from intervals import EmptySetPropagationWarning
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import REALS
from intervals import pow_rev1
from intervals import pow_rev2
from intervals import pown_rev
from intervals.kernel import contains_point
from intervals.kernel import intersection
from intervals.kernel import pieces
from intervals.reverse import negate
from intervals.rounding import exact_cuts
from intervals.rounding import is_float
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.test_reverse import ALL
from tests.test_reverse import ENTIRE_1788
from tests.test_reverse import _exact
from tests.test_reverse import _first_double_above
from tests.test_reverse import _quiet
from tests.test_reverse import float_cut_tuples
from tests.test_reverse import one
from tests.test_reverse import trig_in_slack as in_slack
from tests.test_reverse import v1788

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
MAX = 1.7976931348623157e308
H = float.fromhex

UP_TO_INF = one(0, INF, False, True)  # (0, inf]
FROM_MINUS_INF = one(-INF, 0, True, False)  # [-inf, 0)
FINITE = one(-INF, INF, False, False)
NON_NEGATIVE = one(0, INF)
POSITIVE = one(0, INF, False, True)  # (0, inf], the bases with a value for every y < 0
BELOW_ONE_FROM_0 = one(0, 1, True, False)
OPEN_UNIT = one(0, 1, False, False)
ABOVE_ONE_TO_INF = one(1, INF, False, True)
FINITE_POSITIVE = one(0, INF, False, False)


# THE OPS: an empty operand must warn and give ∅, and nothing else may warn (the suite makes the
# library's warnings errors), negative bases and the points with no value included

def prev(op, other, c, x=REALS):
    if not other or not c or not x:
        with pytest.warns(EmptySetPropagationWarning):
            result = op(other, c, x)
        assert result == EMPTY
        return result
    return op(other, c, x)


def prev1(b, c, x=REALS):
    return prev(pow_rev1, b, c, x)


def prev2(a, c, y=REALS):
    return prev(pow_rev2, a, c, y)


# THE POWER AT A POINT, FROM THE DEFINITIONS

def _arb(v, flint):
    v = Fraction(v)
    return flint.arb(flint.fmpq(v.numerator, v.denominator))


def cmp_pow(t, y, v) -> int:
    """the sign of `t ** y - v` for finite t, v > 0 and a finite y != 0: exactly as `t ** p` against `v ** q`
    (y = p/q, q > 0) where those are small, else by arb at a growing precision (the two are then never
    equal: `t ** (p/q)` is rational only if `t ** p` is a q-th power)"""
    t, y, v = Fraction(t), Fraction(y), Fraction(v)
    p, q = y.numerator, y.denominator
    bits = abs(p) * max(t.numerator.bit_length(), t.denominator.bit_length()) + \
        q * max(v.numerator.bit_length(), v.denominator.bit_length())
    if q <= 64 and bits <= 200000:
        a, b = t ** p, v ** q
        return (a > b) - (a < b)
    flint = pytest.importorskip('flint')
    old = flint.ctx.prec
    try:
        for prec in (128, 512, 2048, 8192):
            flint.ctx.prec = prec
            d = _arb(y, flint) * _arb(t, flint).log() - _arb(v, flint).log()
            if d > 0:
                return 1
            if d < 0:
                return -1
        raise AssertionError(f'arb never decided t ** y against v: {t}, {y}, {v}')
    finally:
        flint.ctx.prec = old


def special(t, y):
    """`t ** y` where it is 0, 1 or inf by the definitions, or its limit at an open end (0 ** y for y < 0:
    inf from above), None elsewhere (then finite, positive and not 1)"""
    if t == 0:
        return 0 if y > 0 else INF
    if t == INF:
        return INF if y > 0 else 0
    if t == 1 or y == 0:
        return 1
    if y in (INF, -INF):
        return INF if (y > 0) == (t > 1) else 0
    return None


def cmp_value(t, y, v) -> int:
    """the sign of `t ** y - v` for v in [-inf, inf]"""
    s = special(t, y)
    if s is not None:
        return (s > v) - (s < v)
    if v == INF:
        return -1
    if v <= 0:
        return 1
    return cmp_pow(t, y, v)


def image_meets(low, low_closed, high, high_closed, c) -> bool:
    """whether an interval from `t ** y` at `low = (t, y)` to its value at `high` meets c (both non-empty)"""
    for a, a_closed, b, b_closed in pieces(c):
        s1 = cmp_value(*low, b)
        if s1 > 0 or (s1 == 0 and not (low_closed and b_closed)):
            continue
        s2 = cmp_value(*high, a)
        if s2 < 0 or (s2 == 0 and not (high_closed and a_closed)):
            continue
        return True
    return False


def _meets(cuts, part) -> bool:
    return bool(intersection(cuts, part))


def fits1(t, b, c) -> bool:
    """some y in b has `t ** y` in c: `y -> t ** y` maps each piece of b onto an interval, monotonically
    and continuously for a finite t > 0 other than 1 (±inf included, where its values are 0 and inf)"""
    if t < 0:
        return False
    for lo, lo_closed, hi, hi_closed in pieces(b):
        p = one(lo, hi, lo_closed, hi_closed)
        if t == 0:
            found = _meets(p, UP_TO_INF) and contains_point(c, 0)
        elif t == 1:
            found = _meets(p, FINITE) and contains_point(c, 1)
        elif t == INF:
            found = (_meets(p, UP_TO_INF) and contains_point(c, INF)) or \
                (_meets(p, FROM_MINUS_INF) and contains_point(c, 0))
        elif t > 1:
            found = image_meets((t, lo), lo_closed, (t, hi), hi_closed, c)
        else:
            found = image_meets((t, hi), hi_closed, (t, lo), lo_closed, c)
        if found:
            return True
    return False


def fits2(s, a, c) -> bool:
    """some t in a has `t ** s` in c: for a finite s != 0, `t -> t ** s` maps each piece of a onto an
    interval, rising on [0, inf] for s > 0 and falling on (0, inf] for s < 0 (0 has no value there)"""
    for lo, lo_closed, hi, hi_closed in pieces(intersection(a, NON_NEGATIVE)):
        p = one(lo, hi, lo_closed, hi_closed)
        if s == 0:
            found = _meets(p, FINITE_POSITIVE) and contains_point(c, 1)
        elif s == INF:
            found = (_meets(p, BELOW_ONE_FROM_0) and contains_point(c, 0)) or \
                (_meets(p, ABOVE_ONE_TO_INF) and contains_point(c, INF))
        elif s == -INF:
            found = (_meets(p, OPEN_UNIT) and contains_point(c, INF)) or \
                (_meets(p, ABOVE_ONE_TO_INF) and contains_point(c, 0))
        elif s > 0:
            found = image_meets((lo, s), lo_closed, (hi, s), hi_closed, c)
        else:
            found = any(image_meets((q_hi, s), q_hi_closed, (q_lo, s), q_lo_closed, c)
                        for q_lo, q_lo_closed, q_hi, q_hi_closed in pieces(intersection(p, POSITIVE)))
        if found:
            return True
    return False


def _finite_ends(*cut_tuples_):
    return {Fraction(cut.value) for cuts in cut_tuples_ for cut in cuts if math.isfinite(cut.value)}


def _with_neighbours(values):
    out = set()
    for v in values:
        out.add(Fraction(v))
        f = float(v)
        for toward in (INF, -INF):
            other = math.nextafter(f, toward)
            if math.isfinite(other):
                out.add(Fraction(other))
    return out


def candidates(approx, *cut_tuples_):
    """the finite ends of the operands and of the result, `approx` (the result's ends, roughly), one
    double either side of every float end and of each approximation, a point between each two, one
    beyond each end, and -1, 0, 1, ±inf"""
    floats = {cut.value for cuts in cut_tuples_ for cut in cuts if is_float(cut.value) and math.isfinite(cut.value)}
    finite = sorted(_finite_ends(*cut_tuples_) | _with_neighbours(floats | set(approx)) | {Fraction(-1), Fraction(0), Fraction(1)})
    return [-INF, INF, finite[0] - 1, finite[-1] + 1, *finite, *((p + q) / 2 for p, q in zip(finite, finite[1:]))]


def approx_roots(b, c):
    """`v ** (1/w)` in floats, for the ends v > 0 of c and w != 0 of b"""
    out = []
    for v in _finite_ends(c):
        for w in _finite_ends(b):
            if v > 0 and w:
                try:
                    r = float(v) ** (1 / float(w))
                except (OverflowError, ZeroDivisionError):
                    continue
                if 0 < r < INF:
                    out.append(r)
    return out


def approx_logs(a, c):
    """`log_t v` in floats, for the ends t > 0, t != 1 of a and v > 0 of c"""
    return [math.log(v) / math.log(t) for t in _finite_ends(a) for v in _finite_ends(c) if t > 0 and t != 1 and v > 0]


# EXAMPLES

@pytest.mark.parametrize('b, c, x, want', [
    ('[2]', '[4, 9]', None, '[2, 3]'),
    ('[1/2]', '[2]', None, '[4]'),  # rational ends are exact
    ('[1/3]', '[2]', None, '[8]'),
    ('[-1/2]', '[4]', None, '[1/16]'),
    ('[2, 4]', '[16]', None, '[2, 4]'),
    ('[-1, 1]', '[2]', None, '(0, 1/2] | [2, inf)'),  # 2 ** (1/y): y in [-1, 0) and in (0, 1]
    ('[0]', '[1]', None, '(0, inf)'),  # t ** 0 = 1 for a finite t > 0; 0 ** 0 and inf ** 0 have no value
    ('[0]', '[0, 1/2]', None, '{}'),  # pow_rev.itl:92
    ('[-2]', '[0, 1]', None, '[1, inf]'),  # inf ** -2 = 0; 0 ** -2 has no value
    ('[-2, 2]', '[0]', None, '[0] | [inf]'),  # 0 ** y = 0 for y > 0, inf ** y = 0 for y < 0
    ('[inf]', '[0]', None, '[0, 1)'),  # t ** inf = 0 below 1, 0 ** inf = 0
    ('[inf]', '[inf]', None, '(1, inf]'),
    ('[-inf]', '[inf]', None, '(0, 1)'),  # 0 ** -inf has no value
    ('[-inf]', '[0]', None, '(1, inf]'),
    ('[-inf, inf]', '[1]', None, '(0, inf)'),  # 1 ** y = 1 for a finite y, t ** 0 = 1
    ('[1, 2]', '[-3, -1]', None, '{}'),  # no power is negative
    ('[inf]', '[1]', None, '{}'),  # 1 ** inf has no value, t ** inf is 0 or inf (pins sabotage B4)
    ('[2]', '[4, 9]', '[0, 5/2]', '[2, 5/2]'),
    ('[2]', '[4, 9]', '[-9, -1]', '{}'),  # a negative base has no value
])
def test_pow_rev1_examples(b, c, x, want):
    x = REALS if x is None else M.parse(x)
    assert prev1(M.parse(b), M.parse(c), x) == M.parse(want)
    assert prev1(O.parse(b), O.parse(c), x) == O.parse(want)  # exact operands: the same in both classes


@pytest.mark.parametrize('a, c, y, want', [
    ('[2]', '[4, 8]', None, '[2, 3]'),
    ('[4]', '[2]', None, '[1/2]'),  # log_4 2 = 1/2: a rational log (elementary._exact_log)
    ('[8]', '[1/4]', None, '[-2/3]'),
    ('[1, 4]', '[2]', None, '[1/2, inf)'),  # log_t 2 for t in (1, 4]; 1 ** s is never 2
    ('[1/4, 1/2]', '[2, 4]', None, '[-2, -1/2]'),  # pow_rev.itl:608
    ('[1/4, 1/2]', '[2, inf)', None, '(-inf, -1/2]'),  # :609's operands (a row)
    ('[1/4, 1/2]', '[2, inf]', None, '[-inf, -1/2]'),  # (1/4) ** -inf = inf
    ('[0]', '[0]', None, '(0, inf]'),  # 0 ** s = 0 for s in (0, inf]
    ('[0, 1]', '[0]', None, '(0, inf]'),  # t ** inf = 0 below 1
    ('[2]', '[0]', None, '[-inf]'),
    ('[inf]', '[0, inf]', None, '[-inf, 0) | (0, inf]'),  # inf ** 0 has no value
    ('[inf]', '[inf]', None, '(0, inf]'),  # inf ** s = inf for s > 0 (pins sabotage C7)
    ('[inf]', '[0]', None, '[-inf, 0)'),  # inf ** s = 0 for s < 0
    ('[1]', '[1]', None, '(-inf, inf)'),  # 1 ** ±inf has no value
    ('[1, 2]', '[1]', None, '(-inf, inf)'),
    ('[2, 3]', '[1]', None, '[0]'),  # :559
    ('[-1, -1/2]', '[1]', None, '{}'),  # a negative base has no value
    ('[2]', '[4, 8]', '[0, 5/2]', '[2, 5/2]'),
])
def test_pow_rev2_examples(a, c, y, want):
    y = REALS if y is None else M.parse(y)
    assert prev2(M.parse(a), M.parse(c), y) == M.parse(want)
    assert prev2(O.parse(a), O.parse(c), y) == O.parse(want)


def test_irrational_ends_are_open_one_ulp_enclosures():
    r = pow_rev1(M(2), M(2))  # sqrt 2
    assert r == M.from_cuts(one(1.414213562373095, 1.4142135623730951, False, False))
    r = pow_rev2(M(3), M(2))  # log_3 2 = 0.63092975357145743...
    assert r == M.from_cuts(one(0.6309297535714574, 0.6309297535714575, False, False))
    # to nearest with a float operand: flags kept, one double (rounded_pow, rounded log)
    assert pow_rev1(M(2.0), M(2)) == M(math.sqrt(2))
    assert pow_rev2(M(3.0), M(2)) == M(0.6309297535714574)
    # outward: the same enclosure, open
    assert pow_rev1(O(2.0), O(2.0)) == O.from_cuts(one(1.414213562373095, 1.4142135623730951, False, False))


def test_class_coercion_and_warnings():
    for op in (pow_rev1, pow_rev2):
        assert type(op(O(2.0), M(4), M(0, 5))) is O and type(op(M(2), M(4), O(0.0, 5.0))) is O
        assert type(op(M(2), O(4.0))) is O and type(op(M(2), M(4))) is M
        assert op(2, 4) == M(2) and op(2, 4, 0) == EMPTY and op(2, 4, 2) == M(2)
        for args in (('[1, 2]', M(1)), (M(1), '[1, 2]'), (M(1), M(1), None), (True, M(1))):
            with pytest.raises(TypeError):
                op(*args)
        for args in ((EMPTY, M(1)), (M(1), EMPTY), (M(1), M(1), EMPTY), (O(), O())):
            with pytest.warns(EmptySetPropagationWarning):
                assert op(*args) == EMPTY
    # no solution, negative bases and the points with no value warn nothing (warnings are errors here)
    assert pow_rev1(M(0), M(0)) == EMPTY and pow_rev1(M(2), M(-1)) == EMPTY
    assert pow_rev2(M(-3, -1), M(1)) == EMPTY and pow_rev2(M(1), M(INF)) == EMPTY
    assert pow_rev2(M(INF), M(1)) == EMPTY  # inf ** 0 has no value


# THE DEFINING PROPERTY: exactly the points that fit

operands = exact_cut_tuples.filter(bool)  # an empty operand is test_class_coercion_and_warnings's


@settings(max_examples=150, deadline=None)
@given(b=operands, c=operands, x=st.one_of(st.just(ALL), exact_cut_tuples))
@example(b=v1788(-4.0, -2.0), c=v1788(0.0, 0.5), x=ENTIRE_1788)  # pow_rev.itl:61, [2 ** (1/4), inf]
@example(b=v1788(-4.0, -2.0), c=v1788(0.5, 2.0), x=ENTIRE_1788)  # :86, [sqrt(1/2), sqrt 2]
@example(b=v1788(-4.0, -2.0), c=v1788(2.0, 4.0), x=ENTIRE_1788)  # :96, [1/2, 2 ** (-1/4)]
@example(b=v1788(2.0, 4.0), c=v1788(2.0, 4.0), x=ENTIRE_1788)  # :504
@example(b=v1788(-4.0, 4.0), c=v1788(0.0, 0.5), x=v1788(0.0, 1.0))  # :173
@example(b=ENTIRE_1788, c=ENTIRE_1788, x=ENTIRE_1788)  # :35
@example(b=ENTIRE_1788, c=v1788(0.0, 0.0), x=ENTIRE_1788)  # :47, [0]
@example(b=v1788(0.0, 0.0), c=v1788(1.0, 1.0), x=ENTIRE_1788)  # :107, (0, inf)
@example(b=v1788(0.0, 0.0), c=v1788(1.0, 1.0), x=v1788(-INF, 0.0))  # :45, empty
@example(b=one(INF, INF), c=one(0, 0), x=ALL)  # [0, 1)
@example(b=one(-INF, -INF), c=one(0, INF), x=ALL)  # 0 ** -inf has no value
@example(b=one(-2, 2), c=one(0, 0), x=ALL)  # [0] and [inf]
@example(b=one(0, 0), c=one(1, 1), x=ALL)  # 0 ** 0 and inf ** 0 have no value
def test_pow_rev1_is_exactly_the_points_that_fit(b, c, x):
    """`t` is in the result iff `t ∈ x` and some `y ∈ b` has `t ** y ∈ c`, but inside a rounded end's slack"""
    B, C, X = M.from_cuts(b), M.from_cuts(c), M.from_cuts(x)
    result = prev1(B, C, X)
    assert result.issubset(X)
    for t in candidates(approx_roots(b, c), b, c, x, result.cuts):
        if t in X and fits1(t, b, c):
            assert t in result, (t, result)
        elif t in result:
            assert in_slack(t, result), (t, result)


@settings(max_examples=150, deadline=None)
@given(a=operands, c=operands, y=st.one_of(st.just(ALL), exact_cut_tuples))
@example(a=v1788(0.25, 0.5), c=v1788(2.0, 4.0), y=ENTIRE_1788)  # pow_rev.itl:608, [-2, -1/2]
@example(a=v1788(0.25, 0.5), c=v1788(2.0, INF), y=ENTIRE_1788)  # :609, a row: (-inf, -1/2]
@example(a=v1788(0.25, 1.0), c=v1788(2.0, INF), y=ENTIRE_1788)  # :642, a row: (-inf, -1/2]
@example(a=v1788(0.25, 0.5), c=v1788(0.25, 0.5), y=ENTIRE_1788)  # :591, [1/2, 2]
@example(a=v1788(0.0, 0.25), c=v1788(0.0, 2.0), y=ENTIRE_1788)  # :573, [-1/2, inf)
@example(a=v1788(0.0, 0.0), c=v1788(0.0, 0.0), y=ENTIRE_1788)  # :544, (0, inf)
@example(a=v1788(1.0, 1.0), c=ENTIRE_1788, y=ENTIRE_1788)  # :554
@example(a=v1788(2.0, 3.0), c=v1788(1.0, 1.0), y=ENTIRE_1788)  # :559, [0]
@example(a=one(3, 3), c=one(2, 2), y=ALL)  # log_3 2, irrational
@example(a=one(0, 1), c=one(0, 0), y=ALL)  # (0, inf]
@example(a=one(INF, INF), c=one(0, INF), y=ALL)  # inf ** 0 has no value
@example(a=one(0, 0), c=one(INF, INF), y=ALL)  # 0 ** -inf has no value: nothing
def test_pow_rev2_is_exactly_the_points_that_fit(a, c, y):
    """`s` is in the result iff `s ∈ y` and some `t ∈ a` has `t ** s ∈ c`, but inside a rounded end's slack"""
    A, C, Y = M.from_cuts(a), M.from_cuts(c), M.from_cuts(y)
    result = prev2(A, C, Y)
    assert result.issubset(Y)
    for s in candidates(approx_logs(a, c), a, c, y, result.cuts):
        if s in Y and fits2(s, a, c):
            assert s in result, (s, result)
        elif s in result:
            assert in_slack(s, result), (s, result)


def test_pow_rev2_is_tighter_than_the_vector():
    """pow_rev.itl:609 and :642 (rows `_POW_REV_LOOSE_ROWS`): for A = [1/4, 1/2] or [1/4, 1] and C = [2, inf),
    `t ** s >= 2` for some t in A iff `s <= -1/2`, decided exactly: (1/4) ** -1/2 is 2, and for s in (-1/2, 0)
    the greatest `t ** s` over A, at t = 1/4, is below 2. the tightest hull is [-inf, -0.5], ours; 1788
    answers [entire] and [-infinity, 0.0], while its own vectors with C = [2, 4] (:608, :640) answer -0.5"""
    c = v1788(2.0, INF)
    for a in (v1788(0.25, 0.5), v1788(0.25, 1.0)):
        assert pow_rev2(M.from_cuts(a), M.from_cuts(c), M.from_cuts(ENTIRE_1788)) == M.parse('(-inf, -1/2]')
        assert fits2(Fraction(-1, 2), a, c)
        for s in (math.nextafter(-0.5, INF), -0.25, Fraction(-1, 10 ** 30), 0, 1, 7):
            assert not fits2(Fraction(s), a, c), s
        assert pow_rev2(O.from_cuts(exact_cuts(a)), O(2.0, INF, end_closed=False), O.parse('(-inf, inf)')) == \
            O.parse('(-inf, -0.5]')


# SET LEVEL, WITH THE LIBRARY'S POW

def no_base(b: MultiInterval) -> MultiInterval:
    """the t with `{t} ** b` empty: the negative ones; 0 if b misses (0, inf]; 1 if b has no finite point;
    inf if b has nothing but 0; every t for an empty b"""
    if not b:
        return REALS
    out = M.from_cuts(FROM_MINUS_INF)
    if not b & M.from_cuts(UP_TO_INF):
        out |= M(0)
    if not b & M.from_cuts(FINITE):
        out |= M(1)
    if not b.difference(M(0)):
        out |= M(INF)
    return out


def no_exponent(a: MultiInterval) -> MultiInterval:
    """the s with `a ** {s}` empty: each kind of s has bases of its own (`fits2`'s cases); every s for an
    empty a"""
    if not a:
        return REALS
    out = EMPTY
    for s, bases in ((M.parse('(0, inf)'), '[0, inf]'), (M.parse('(-inf, 0)'), '(0, inf]'), (M(0), '(0, inf)'),
                     (M(INF), '[0, 1) | (1, inf]'), (M(-INF), '(0, 1) | (1, inf]')):
        if not a & M.parse(bases):
            out |= s
    return out


@settings(max_examples=100, deadline=None)
@given(t=exact_cut_tuples, b=exact_cut_tuples, more=exact_cut_tuples, x=exact_cut_tuples)
@example(t=one(-INF, INF), b=one(0, 0), more=(), x=ALL)
@example(t=one(0, 3), b=one(INF, INF), more=(), x=ALL)
@example(t=one(0, INF), b=one(-2, -1), more=(), x=ALL)
def test_pow_rev1_the_largest_set(t, b, more, x):
    """any T is inside `pow_rev1(B, T ** B ∪ more)`, the t with `{t} ** B` empty aside, and inside it
    with x where it is in x: the library's pow on sets, in the outward class (whose rounding never loses
    a true point, so C's float ends from T ** B cannot either)"""
    B = O.from_cuts(b)
    T = O.from_cuts(t).difference(no_base(M.from_cuts(b)))
    C = _quiet(lambda: T ** B) | O.from_cuts(more)
    assert T.issubset(prev1(B, C))
    X = O.from_cuts(x)
    assert (T & X).issubset(prev1(B, C, X))


@settings(max_examples=100, deadline=None)
@given(s=exact_cut_tuples, a=exact_cut_tuples, more=exact_cut_tuples, y=exact_cut_tuples)
@example(s=one(-INF, INF), a=one(0, 0), more=(), y=ALL)
@example(s=one(-INF, INF), a=one(1, 1), more=(), y=ALL)
@example(s=one(-INF, INF), a=one(INF, INF), more=(), y=ALL)
def test_pow_rev2_the_largest_set(s, a, more, y):
    """any S is inside `pow_rev2(A, A ** S ∪ more)`, the s with `A ** {s}` empty aside"""
    A = O.from_cuts(a)
    S = O.from_cuts(s).difference(no_exponent(M.from_cuts(a)))
    C = _quiet(lambda: A ** S) | O.from_cuts(more)
    assert S.issubset(prev2(A, C))
    Y = O.from_cuts(y)
    assert (S & Y).issubset(prev2(A, C, Y))


@settings(max_examples=100, deadline=None)
@given(b=exact_cut_tuples, b_more=exact_cut_tuples, c=exact_cut_tuples, c_more=exact_cut_tuples,
       x=exact_cut_tuples, x_more=exact_cut_tuples, which=st.sampled_from([1, 2]))
def test_pow_rev_isotone_and_distributive(b, b_more, c, c_more, x, x_more, which):
    """isotone in each operand, distributing over a union of either operand (an existential preimage),
    and x only intersects"""
    rev = prev1 if which == 1 else prev2
    B, B2, C, C2, X, X2 = (M.from_cuts(v) for v in (b, b_more, c, c_more, x, x_more))
    r = rev(B, C, X)
    assert r.issubset(rev(B | B2, C, X)) and r.issubset(rev(B, C | C2, X)) and r.issubset(rev(B, C, X | X2))
    assert rev(B | B2, C) == rev(B, C) | rev(B2, C)
    assert rev(B, C | C2) == rev(B, C) | rev(B, C2)
    assert r == rev(B, C) & X


def reciprocal(cuts):
    """`{1/v : v in cuts, v > 0}`, exactly, 1/inf = 0 (no pole: 0 is left out)"""
    def inverse(v):
        return INF if v == 0 else 0 if v == INF else 1 / Fraction(v)
    out = EMPTY
    for lo, lo_closed, hi, hi_closed in pieces(intersection(cuts, POSITIVE)):
        out |= M.from_cuts(one(inverse(hi), inverse(lo), hi_closed, lo_closed))
    return out


@settings(max_examples=100, deadline=None)
@given(b=exact_cut_tuples, c=exact_cut_tuples, cls=st.sampled_from([M, O]))
@example(b=one(-2, 2), c=one(0, INF), cls=M)
def test_pow_rev_symmetry(b, c, cls):
    """`t ** -y = 1 / t ** y` for t > 0 in (0, inf] and y with a value: `pow_rev1(-B, 1/C)` is
    `pow_rev1(B, C)` but at t = 0 (0 ** y has a value only for y > 0); `pow_rev2(A, 1/C)` is
    `-pow_rev2(A, C)` for A without 0. `c`'s part in (0, inf] (1/0 is a pole)"""
    B = cls.from_cuts(b)
    C = cls.from_cuts(intersection(c, POSITIVE))
    if not B or not C:
        return
    C_inverse = cls.from_cuts(reciprocal(c).cuts)
    assert pow_rev1(_negated(cls, b), C_inverse).difference(M(0)) == pow_rev1(B, C).difference(M(0))
    A = B.difference(M(0))
    if A:
        assert pow_rev2(A, C_inverse) == _negated(cls, pow_rev2(A, C).cuts)


def _negated(cls, cuts):
    return cls.from_cuts(negate(cuts))


# RELATIONS WITH THE OTHER OPS

all_float_cut_tuples = float_cut_tuples.filter(lambda cuts: all(is_float(cut.value) for cut in cuts))


@settings(max_examples=100, deadline=None)
@given(n=st.integers(-8, 8).filter(bool), c=st.one_of(exact_cut_tuples, all_float_cut_tuples),
       x=st.one_of(exact_cut_tuples, all_float_cut_tuples), cls=st.sampled_from([M, O]))
@example(n=-2, c=one(0.0, 1.0), x=ALL, cls=M)  # inf ** -2 = 0, and 0 ** -2 has no value
@example(n=3, c=one(2.0, 2.0), x=ALL, cls=M)  # the cube root of 2, to nearest
def test_pow_rev1_by_an_int_is_pown_rev_on_the_bases(n, c, x, cls):
    """for an int n != 0, `t ** n` is pown's value at every t >= 0 where pow has one (0 ** n and inf ** n
    alike), so `pow_rev1([n], c, x)` is `pown_rev(c, n, x ∩ [0, inf])`, the rounding included (both
    correctly rounded `v ** (1/n)`). `c` is all exact or all float: pown_rev takes each end's kind on its
    own, pow_rev1 the operands' (pow's rule)"""
    C, X = cls.from_cuts(c), cls.from_cuts(x)
    if not C or not X:
        return
    assert pow_rev1(cls(n), C, X) == _quiet(lambda: pown_rev(C, n, X & cls(0, INF)))  # x ∩ [0, inf] may be empty


@settings(max_examples=100, deadline=None)
@given(t=st.one_of(st.fractions(0, 20, max_denominator=6), st.floats(0, 20)).filter(lambda t: t not in (0, 1)),
       c=st.one_of(exact_cut_tuples, all_float_cut_tuples), cls=st.sampled_from([M, O]))
@example(t=Fraction(1, 4), c=one(0.5, 2.0), cls=M)  # pow_rev.itl:600's shape: log to the base 1/4 is exact
@example(t=3, c=one(0, 2), cls=M)
def test_pow_rev2_of_a_point_is_the_log(t, c, cls):
    """for a base t in (0, 1) ∪ (1, inf), `t ** s ∈ c` iff `s = log_t v` for some v of c, `log_t 0` and
    `log_t inf` the signed infinities (the library's log attains them), so `pow_rev2([t], c)` is
    `c.log(t)`, rounded alike (t exact: the float rule is c's alone in both)"""
    C = cls.from_cuts(c)
    if not C:
        return
    t = Fraction(t)
    assert pow_rev2(cls(t), C) == _quiet(lambda: C.log(t))


# FLOAT OPERANDS

def widened(r: MultiInterval) -> MultiInterval:
    """each piece closed and one double wider at each end, ±inf included: to nearest, a finite end past the
    largest double is inf, closed (the library's rule: `M(1e200) ** M(1000.0)` is `[inf]`), one double
    from MAX"""
    return M.from_pieces((math.nextafter(lo, -INF), math.nextafter(hi, INF)) for lo, _, hi, _ in pieces(r.cuts))


@settings(max_examples=100, deadline=None)
@given(b=float_cut_tuples, c=float_cut_tuples, x=float_cut_tuples, which=st.sampled_from([1, 2]))
@example(b=one(-4.0, -2.0), c=one(0.0, 0.5), x=ALL, which=1)  # pow_rev.itl:61
@example(b=one(2.0, 4.0), c=one(2.0, 4.0), x=ALL, which=1)  # :504
@example(b=one(0.25, 0.5), c=one(2.0, 4.0), x=ALL, which=2)  # :608
@example(b=one(3.0, 3.0), c=one(2.0, 2.0), x=ALL, which=2)  # log_3 2: one point, outward two doubles, open
@example(b=one(1e-3, 1e-3), c=one(1e200, 1e200), x=ALL, which=1)  # 1e200 ** 1000, past max float
@example(b=one(-1e-3, -1e-3), c=one(1e200, 1e200), x=ALL, which=1)  # below the least subnormal
@example(b=one(H('0x1.0000000000001p+0'), H('0x1.0000000000001p+0')), c=one(1e300, 1e300), x=ALL, which=2)
@example(b=one(8.0, 8.0), c=one(2.0, 2.0), x=ALL, which=2)  # log_8 2 = 1/3, rational: outward open (C14)
@example(b=one(3.0, 3.0), c=one(1e300, math.nextafter(1e300, INF), False, False), x=ALL, which=2)  # squeezed (C16)
def test_pow_rev_float_operands(b, c, x, which):
    """outward: the exact result of the same doubles is inside, what rounding adds holds no double
    strictly inside it, a closed end is a point of the exact result. to nearest, x omitted (an end of x can
    fall in the half ulp a rounded end moved): the exact result within one double of each piece, inside the
    outward closure, not empty if it is not; and x only intersects, after the rounding"""
    rev = prev1 if which == 1 else prev2
    B, C, X = M.from_cuts(exact_cuts(b)), M.from_cuts(exact_cuts(c)), M.from_cuts(exact_cuts(x))
    exact = rev(B, C, X)
    outward = rev(O.from_cuts(b), O.from_cuts(c), O.from_cuts(x))
    assert exact.issubset(outward)
    for lo, _, hi, _ in pieces(outward.difference(exact).cuts):
        assert hi <= _first_double_above(lo), (lo, hi)
    for lo, lo_closed, hi, hi_closed in pieces(outward.cuts):
        for end, closed in ((lo, lo_closed), (hi, hi_closed)):
            if closed:
                assert end in exact, (end, outward)
    exact_all = rev(B, C)
    nearest_all = rev(M.from_cuts(b), M.from_cuts(c))
    outward_all = rev(O.from_cuts(b), O.from_cuts(c))
    for r in (outward, nearest_all):  # floats, or an exact 0, 1 or ±inf, which is a double
        assert all(isinstance(cut.value, float) or cut.value == float(cut.value) for cut in r.cuts), r
    assert exact_all.issubset(widened(nearest_all))
    assert nearest_all.issubset(M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(outward_all.cuts)))
    if exact_all:
        assert nearest_all
    assert rev(M.from_cuts(b), M.from_cuts(c), M.from_cuts(x)) == nearest_all & M.from_cuts(x)


@pytest.mark.parametrize('a, lo, hi', [(8.0, 2.0, 4.0), (8.0, 0.5, 2.0), (0.125, 2.0, 32.0), (27.0, 3.0, 9.0)])
def test_pow_rev2_nearest_keeps_the_flag_of_a_moved_log(a, lo, hi):
    """to nearest, a rational log that rounding moved (`log_8 2` = 1/3) keeps its flag, as the forward
    log to a base: `pow_rev2(a, c)` is `c.log(a)` for a point a > 0 and c > 0, closed; outward it is open
    (M13e's review, 2026-09-27: part 4's pin, `log_8 2` from [8.0] and [2.0], saw only the outward side)"""
    nearest, outward = pow_rev2(M(a), M(lo, hi)), pow_rev2(O(a), O(lo, hi))
    assert nearest == M(lo, hi).log(a)
    assert nearest.inf_closed and nearest.sup_closed
    assert not outward.inf_closed and not outward.sup_closed


# SOUND AT SAMPLED POINTS, WITH THE LIBRARY'S POW

def inner(r: MultiInterval) -> MultiInterval:
    """the part of a pow result of exact operands that the true image surely holds: each open float end
    (an irrational end's enclosure, the true end strictly inside its double's slack) moved one double
    inward and closed"""
    out = EMPTY
    for lo, lo_closed, hi, hi_closed in pieces(r.cuts):
        if not lo_closed and isinstance(lo, float):
            lo, lo_closed = math.nextafter(lo, INF), True
        if not hi_closed and isinstance(hi, float):
            hi, hi_closed = math.nextafter(hi, -INF), True
        if lo < hi or (lo == hi and lo_closed and hi_closed):
            out |= M.from_cuts(one(lo, hi, lo_closed, hi_closed))
    return out


@settings(max_examples=100, deadline=None)
@given(b=cut_tuples(), c=cut_tuples(), x=cut_tuples(), seed=st.integers(0, 2 ** 32 - 1),
       which=st.sampled_from([1, 2]))
@example(b=one(-4.0, -2.0), c=one(0.5, 2.0), x=one(0.0, 2.0), seed=0, which=1)  # pow_rev.itl:86
@example(b=one(0.25, 0.5), c=one(0.25, 0.5), x=one(-1.0, 3.0), seed=0, which=2)  # :591
@example(b=one(-INF, INF), c=one(0, INF), x=one(-INF, INF), seed=0, which=2)
def test_pow_rev_sound_at_sampled_points(b, c, x, seed, which):
    """M14's soundness: a sampled point of x whose image by the library's pow (`{t} ** B`, or `A ** {s}`)
    surely meets C is in the exact result and in the outward one (float operands read as the rationals they
    are); a sampled point of the exact result has an image meeting C, or lies in a rounded end's slack"""
    rng = random.Random(seed)
    rev = prev1 if which == 1 else prev2
    B, C, X = (M.from_cuts(exact_cuts(v)) for v in (b, c, x))
    exact = rev(B, C, X)
    outward = rev(O.from_cuts(b), O.from_cuts(c), O.from_cuts(x))

    def image(p):
        return _quiet(lambda: M(p) ** B if which == 1 else B ** M(p))
    for p in sample(X.cuts, 20, rng):
        p = _exact(p)
        if inner(image(p)) & C:
            assert p in exact and p in outward, p
    for p in sample(exact.cuts, 20, rng):
        p = _exact(p)
        assert image(p) & C or in_slack(p, exact), p
