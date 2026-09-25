"""
cancellation (M13f, D13): `A.cancel_minus(B)`, the Minkowski difference, and `A.cancel_plus(B)`

* the defining property, decided completely on exact operands: `X = A.cancel_minus(B)` is exactly
  the set of the `x` with `{x} + B ⊆ A` (the library's `+`), so every `x` in `X` fits and no `x`
  outside it does. the ends of `X` are differences of the ends of `A` and `B`, so probing every such
  difference, a point between each two and one beyond each end, and ±inf, sees the whole of `X`
* at set level `B + X ⊆ A`; `C ⊆ (B + C).cancel_minus(B)`; isotone in `A`, antitone in `B`; a point
  `B = [c]` is subtraction; ieee 1788's formula `[a1 - b1, a2 - b2]` for connected closed operands
* float operands: an `OutwardMultiInterval` result encloses the exact `X` of the same doubles and
  is tight (no double strictly inside what it adds), a `MultiInterval` one is `X` rounded to
  nearest; soundness at sampled points in both classes, and `cancel_plus(B)` is `cancel_minus(-B)`
* the itf1788 vectors of the two ops run in tests/itf1788; some are `@example`s here
"""
import math
import random
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import EMPTY
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import REALS
from intervals.errors import IntervalWarning
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from intervals.kernel import union
from intervals.rounding import exact_cuts
from intervals.rounding import float_cuts
from intervals.rounding import is_float
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
MAX = 1.7976931348623157e308


def one(lo, hi, lo_closed=True, hi_closed=True):
    return normalize([piece(lo, hi, lo_closed, hi_closed)])


def v1788(lo, hi):
    """a 1788 literal under the itf1788 input rule, its doubles held exactly (an unbounded end is open)"""
    return one(_exact(lo), _exact(hi), lo != -INF, hi != INF)


def _exact(x):
    return Fraction(x) if is_float(x) else x


def p(text: str) -> MultiInterval:
    return M.parse(text)


def fits(x, a: MultiInterval, b: MultiInterval) -> bool:
    """`{x} + b ⊆ a`, with the library's `+` (an undefined `inf + -inf` contributes nothing)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return (M(x) + b).issubset(a)


def candidates(a, b):
    """every difference of a finite end of `a` and one of `b`, a point between each two, one beyond
    each end, 0 and ±inf: `X` and the fitting points change only at the differences"""
    ends_a = {cut.value for cut in a if math.isfinite(cut.value)}
    ends_b = {cut.value for cut in b if math.isfinite(cut.value)}
    diffs = sorted({Fraction(x) - Fraction(y) for x in ends_a for y in ends_b} | {Fraction(0)})
    between = [(x + y) / 2 for x, y in zip(diffs, diffs[1:])]
    return [-INF, INF, diffs[0] - 1, diffs[-1] + 1, *diffs, *between]


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return fn(*args)


# EXAMPLES

@pytest.mark.parametrize('a, b, want', [
    ('[0, 10]', '[1, 3]', '[-1, 7]'),
    ('[0, 10)', '[1, 3]', '[-1, 7)'),  # 10 is missing, so x + 3 < 10
    ('(0, 10]', '(1, 3]', '[-1, 7]'),  # open against open: x = -1 gives (0, 2] inside (0, 10]
    ('(0, 10]', '[1, 3]', '(-1, 7]'),  # a closed start of B cannot land on the missing 0
    ('[0, 1]', '[0, 2]', '{}'),  # B wider: 1788 answers entire, ours nothing fits
    ('[-5, 1]', '[-1, 5]', '[-4]'),  # equal widths: one point (cancel.itl:215)
    ('[0, 1] | [10, 12]', '[0] | [10, 11]', '[0, 1]'),  # each piece of B in its own piece of A
    ('[0, 1] | [10, 11] | [20, 21]', '[0] | [10]', '{ [0, 1] , [10, 11] }'),  # x and x + 10 both in A
    ('[0, 1) | (1, 2]', '[0, 1]', '{}'),  # the missing point 1 breaks every shift
    ('[0, 1) | (1, 3]', '[0, 1]', '(1, 2]'),
    ('(-inf, -1]', '[-1, 5]', '(-inf, -6]'),  # D13's example (cancel.itl:166 under the input rule)
    ('[-inf, -1]', '[-1, 5]', '[-inf, -6]'),  # closed at -inf: -inf + B = {-inf} is inside A
    ('(-1, inf)', '[-1, 5]', '(0, inf)'),
    ('(-inf, inf)', '[-1, 5]', '(-inf, inf)'),  # cancel.itl:168 matches: 1788's entire is ours
    ('(-inf, inf)', '(-inf, inf)', '(-inf, inf)'),  # inf + B = {inf} is not in the open A
    ('[-inf, inf]', '(-inf, inf)', '[-inf, inf]'),
    ('(-inf, 5]', '(-inf, 1]', '(-inf, 4]'),  # a ray in a ray
    ('[0, 5]', '(-inf, 1]', '{}'),  # an unbounded B in a bounded A
    ('[3]', '{}', '[-inf, inf]'),  # every x fits the empty set
    ('{}', '{}', '[-inf, inf]'),  # 1788: cancelMinus [empty] [empty] = [empty], a row
    ('{}', '[1, 2]', '{}'),
    ('[inf]', '[1, 2]', '[inf]'),  # only inf + B = {inf}
    ('[inf]', '[-inf]', '[inf]'),  # inf + -inf has no value, so {inf} + B is empty and fits
    ('[-inf, inf]', '[-inf]', '[-inf, inf]'),
    ('[-inf]', '[-inf]', '[-inf, inf]'),  # a finite x keeps -inf at -inf, and inf fits vacuously
    ('[0, 1]', '[-inf]', '[inf]'),  # only the vacuous inf
    ('[0, 1]', '[inf]', '[-inf]'),  # the mirror: only the vacuous -inf, since -inf + inf has no value
    ('[0, 1] | [inf]', '[0] | [inf]', '{ [0, 1] , [inf] }'),  # a finite x keeps inf at inf
    ('[0, 1]', '[0] | [inf]', '{}'),  # ... which A must hold
    ('[-inf] | [0, 1]', '[-inf, 0]', '[-inf]'),  # no finite x: (-inf, 0] fits no piece of A; -inf + B = {-inf}
])
def test_examples(a, b, want):
    assert p(a).cancel_minus(p(b)) == p(want)
    assert O.parse(a).cancel_minus(O.parse(b)) == O.parse(want)
    assert type(O.parse(a).cancel_minus(p(b))) is O
    assert p(a).cancel_plus(_quiet(lambda: -p(b))) == p(want)  # no warning of its own, even for an empty B


def test_1788_float_vectors():
    """the vectors whose exact difference is not a double: 1788 answers the two doubles around it"""
    a, b = O(float.fromhex('0x1.FFFFFFFFFFFFP+0')), O(float.fromhex('0x1.999999999999AP-4'))
    x = a.cancel_minus(b)  # cancel.itl:218
    assert (x.inf, x.sup) == (float.fromhex('0x1.E666666666656P+0'), float.fromhex('0x1.E666666666657P+0'))
    assert not x.inf_closed and not x.sup_closed  # nothing attains a moved end
    x = O(MAX).cancel_minus(O(-MAX))  # cancel.itl:221: [max, infinity]
    assert x == O.parse(f'({MAX!r}, inf)')
    exact = M(Fraction(MAX)).cancel_minus(M(-Fraction(MAX)))
    assert exact == M(2 * Fraction(MAX)) and exact.issubset(x)
    near = M(MAX).cancel_minus(-MAX)  # to nearest, 2 max is inf
    assert near == M(INF)


def test_coerces_numbers_only():
    assert M(0, 10).cancel_minus(3) == M(-3, 7) and M(0, 10).cancel_plus(Fraction(1, 2)) == M(Fraction(1, 2), Fraction(21, 2))
    with pytest.raises(TypeError):
        M(0, 10).cancel_minus('[1, 2]')
    with pytest.raises(TypeError):
        M(0, 10).cancel_plus(None)


# THE DEFINING PROPERTY: exactly the x that fit

@settings(max_examples=150)
@given(cut_tuples_a=exact_cut_tuples, cut_tuples_b=exact_cut_tuples)
@example(cut_tuples_a=v1788(-INF, -1.0), cut_tuples_b=v1788(-1.0, 5.0))  # cancel.itl:166
@example(cut_tuples_a=v1788(-1.0, INF), cut_tuples_b=v1788(-1.0, 5.0))  # :167
@example(cut_tuples_a=v1788(-5.1, -0.9), cut_tuples_b=v1788(-5.0, -1.0))  # :204
@example(cut_tuples_a=v1788(-5.0, -1.0), cut_tuples_b=v1788(-5.1, -0.9))  # :183, B wider
@example(cut_tuples_a=v1788(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('0X1.FFFFFFFFFFFFP+0')),
         cut_tuples_b=v1788(-0.01, 0.1))  # :219
@example(cut_tuples_a=v1788(-MAX, MAX), cut_tuples_b=v1788(-MAX, float.fromhex('0x1.FFFFFFFFFFFFEP+1023')))  # :223
@example(cut_tuples_a=v1788(2.0 ** -1022, float.fromhex('0x1.0000000000001P-1022')),
         cut_tuples_b=v1788(2.0 ** -1022, float.fromhex('0x1.0000000000002P-1022')))  # :231
@example(cut_tuples_a=v1788(-1.0, float.fromhex('0x1.FFFFFFFFFFFFEP-53')),
         cut_tuples_b=v1788(float.fromhex('-0x1.FFFFFFFFFFFFFP-53'), 1.0))  # :235
@example(cut_tuples_a=(), cut_tuples_b=())  # :196
@example(cut_tuples_a=v1788(-INF, INF), cut_tuples_b=v1788(-INF, -1.0))  # :177
@example(cut_tuples_a=union(one(0, 1), one(10, 12)), cut_tuples_b=union(one(0, 0), one(10, 11)))
@example(cut_tuples_a=union(one(-INF, -INF), one(0, 1)), cut_tuples_b=one(-INF, 0))
@example(cut_tuples_a=one(INF, INF), cut_tuples_b=one(-INF, -INF))
@example(cut_tuples_a=one(0, 1), cut_tuples_b=one(-INF, -INF))  # [inf]: inf + -inf fits vacuously
@example(cut_tuples_a=one(0, 1), cut_tuples_b=one(INF, INF))  # [-inf]: the mirror
@example(cut_tuples_a=one(0, 1), cut_tuples_b=union(one(0, 0), one(INF, INF)))  # ∅: a finite x keeps inf
@example(cut_tuples_a=one(0, 1, lo_closed=False), cut_tuples_b=one(0, 1, lo_closed=False))
def test_exactly_the_points_that_fit(cut_tuples_a, cut_tuples_b):
    """`x ∈ A.cancel_minus(B)` iff `{x} + B ⊆ A`: soundness (every x of X fits) and maximality (no x
    outside X fits), at every point where either could change"""
    check_exactly_the_points_that_fit(cut_tuples_a, cut_tuples_b)


def check_exactly_the_points_that_fit(cut_tuples_a, cut_tuples_b):
    a, b = M.from_cuts(cut_tuples_a), M.from_cuts(cut_tuples_b)
    x = a.cancel_minus(b)
    for point in candidates(cut_tuples_a, cut_tuples_b):
        assert (point in x) == fits(point, a, b), (point, x)


@st.composite
def small_sets(draw):
    """1 to 3 short pieces near 0: a B that often fits in several places, so X has several pieces"""
    out = []
    for _ in range(draw(st.integers(1, 3))):
        lo = draw(st.sampled_from([-2, -1, 0, Fraction(1, 2), 1, 3]))
        width = draw(st.sampled_from([0, 0, Fraction(1, 2), 1]))
        closed = (True, True) if width == 0 else (draw(st.booleans()), draw(st.booleans()))
        out.append(piece(lo, lo + width, *closed))
    return normalize(out)


many_pieces = cut_tuples(values=st.one_of(st.integers(-6, 6), st.integers(-6, 6),
                                          st.sampled_from([-INF, INF, Fraction(1, 2)])), max_pieces=6)


@settings(max_examples=150)
@given(cut_tuples_a=many_pieces, cut_tuples_b=small_sets())
@example(cut_tuples_a=union(one(0, 1), one(10, 11), one(20, 21)), cut_tuples_b=union(one(0, 0), one(10, 10)))
def test_exactly_the_points_that_fit_many_pieces(cut_tuples_a, cut_tuples_b):
    """the same, with a short `B` of 1 to 3 pieces against an `A` of up to 6 on a small grid, where
    `X` often has several pieces (about 10% of examples; the test above rarely gets more than one)"""
    check_exactly_the_points_that_fit(cut_tuples_a, cut_tuples_b)


@settings(max_examples=150)
@given(cut_tuples_a=exact_cut_tuples, cut_tuples_b=exact_cut_tuples)
@example(cut_tuples_a=v1788(-INF, -1.0), cut_tuples_b=v1788(-1.0, 5.0))
@example(cut_tuples_a=one(-INF, INF), cut_tuples_b=one(-INF, -INF))
@example(cut_tuples_a=one(0, 1), cut_tuples_b=union(one(0, 0), one(INF, INF)))
@example(cut_tuples_a=union(one(0, 1), one(10, 12)), cut_tuples_b=union(one(0, 0), one(10, 11)))
def test_b_plus_x_inside_a(cut_tuples_a, cut_tuples_b):
    """at set level, `B + X ⊆ A` with the library's `+`: `X` fits as a whole, not only point by point"""
    a, b = M.from_cuts(cut_tuples_a), M.from_cuts(cut_tuples_b)
    x = a.cancel_minus(b)
    assert _quiet(lambda: b + x).issubset(a)


@settings(max_examples=100)
@given(cut_tuples_b=exact_cut_tuples, cut_tuples_c=exact_cut_tuples)
@example(cut_tuples_b=one(-1, 5), cut_tuples_c=one(0, 0))
@example(cut_tuples_b=one(-INF, -INF), cut_tuples_c=one(INF, INF))
def test_cancels_an_addition(cut_tuples_b, cut_tuples_c):
    """`C ⊆ (B + C).cancel_minus(B)`, since `C` itself fits; equal for a connected closed bounded `C`
    and a non-empty bounded `B` (1788's cancellation)"""
    b, c = M.from_cuts(cut_tuples_b), M.from_cuts(cut_tuples_c)
    total = _quiet(lambda: b + c)
    assert c.issubset(total.cancel_minus(b))
    if b and b.is_finite and c.is_contiguous and c.is_finite and c.inf_closed and c.sup_closed:
        assert total.hull.cancel_minus(b.hull) == c


@settings(max_examples=100)
@given(cut_tuples_a=exact_cut_tuples, cut_tuples_more=exact_cut_tuples, cut_tuples_b=exact_cut_tuples,
       cut_tuples_b_more=exact_cut_tuples)
def test_isotone_in_a_antitone_in_b(cut_tuples_a, cut_tuples_more, cut_tuples_b, cut_tuples_b_more):
    a, b = M.from_cuts(cut_tuples_a), M.from_cuts(cut_tuples_b)
    bigger_a, bigger_b = a | M.from_cuts(cut_tuples_more), b | M.from_cuts(cut_tuples_b_more)
    assert a.cancel_minus(b).issubset(bigger_a.cancel_minus(b))
    assert a.cancel_minus(bigger_b).issubset(a.cancel_minus(b))


@given(cut_tuples_a=exact_cut_tuples, c=st.one_of(st.integers(-20, 20), st.fractions(-20, 20, max_denominator=6)))
def test_a_point_is_subtraction(cut_tuples_a, c):
    """`{x} + [c] ⊆ A` iff `x ∈ A - c`: cancelling a finite point is subtracting it"""
    a = M.from_cuts(cut_tuples_a)
    assert a.cancel_minus(c) == _quiet(lambda: a - c)
    assert a.cancel_minus(0) == a


@given(lo_a=st.integers(-20, 20), wid_a=st.integers(0, 20), lo_b=st.integers(-20, 20), wid_b=st.integers(0, 20),
       den=st.integers(1, 6))
def test_1788_formula(lo_a, wid_a, lo_b, wid_b, den):
    """connected, closed, bounded: `[a1 - b1, a2 - b2]` when `wid A >= wid B` (1788's answer), else `∅`
    (where 1788 answers entire as no answer)"""
    a1, a2, b1, b2 = (Fraction(n, den) for n in (lo_a, lo_a + wid_a, lo_b, lo_b + wid_b))
    x = M(a1, a2).cancel_minus(M(b1, b2))
    assert x == (M(a1 - b1, a2 - b2) if wid_a >= wid_b else EMPTY)


# CANCEL_PLUS

@given(cut_tuples_a=cut_tuples(), cut_tuples_b=cut_tuples(), cls=st.sampled_from([M, O]))
@example(cut_tuples_a=v1788(-INF, -1.0), cut_tuples_b=v1788(-5.0, 1.0), cls=M)  # cancel.itl:28
@example(cut_tuples_a=one(0.9, 5.1), cut_tuples_b=one(-5.0, -1.0), cls=O)  # :48
def test_cancel_plus_is_cancel_minus_of_the_negation(cut_tuples_a, cut_tuples_b, cls):
    a, b = cls.from_cuts(cut_tuples_a), cls.from_cuts(cut_tuples_b)
    assert a.cancel_plus(b) == a.cancel_minus(_quiet(lambda: -b))


# FLOAT OPERANDS

def _first_double_above(v) -> float:
    """the smallest double > v (v exact, finite or -inf), computed without the package's rounding"""
    if v == -INF:
        return -MAX
    f = float(Fraction(v)) if abs(v) <= MAX else (INF if v > 0 else -INF)
    while f == INF or (math.isfinite(f) and Fraction(f) > v):  # down to the largest double <= v
        f = math.nextafter(f, -INF)
    return math.nextafter(f, INF)


@settings(max_examples=150)
@given(cut_tuples_a=cut_tuples(), cut_tuples_b=cut_tuples())
@example(cut_tuples_a=one(float.fromhex('0x1.FFFFFFFFFFFFP+0'), float.fromhex('0x1.FFFFFFFFFFFFP+0')),
         cut_tuples_b=one(0.1, 0.1))  # cancel.itl:218
@example(cut_tuples_a=one(MAX, MAX), cut_tuples_b=one(-MAX, -MAX))  # :221
@example(cut_tuples_a=one(-MAX, MAX), cut_tuples_b=one(-MAX, float.fromhex('0x1.FFFFFFFFFFFFEP+1023')))  # :223
@example(cut_tuples_a=one(-1.0, float.fromhex('0x1.FFFFFFFFFFFFFP-53')),
         cut_tuples_b=one(float.fromhex('-0x1.FFFFFFFFFFFFEP-53'), 1.0))  # :234
@example(cut_tuples_a=one(2.0 ** -1074, 2.0 ** -1074), cut_tuples_b=one(-2.0 ** -1074, -2.0 ** -1074))  # :229
def test_float_operands(cut_tuples_a, cut_tuples_b):
    """outward: the exact `X` of the same doubles is inside, and what rounding adds holds no double
    strictly inside it (tight), every finite end a float. to nearest: `X` rounded once"""
    exact = M.from_cuts(exact_cuts(cut_tuples_a)).cancel_minus(M.from_cuts(exact_cuts(cut_tuples_b)))
    outward = O.from_cuts(cut_tuples_a).cancel_minus(O.from_cuts(cut_tuples_b))
    nearest = M.from_cuts(cut_tuples_a).cancel_minus(M.from_cuts(cut_tuples_b))
    has_float = any(is_float(cut.value) for cut in (*cut_tuples_a, *cut_tuples_b))
    assert exact.issubset(outward)
    for lo, _, hi, _ in pieces(outward.difference(exact).cuts):
        assert hi <= _first_double_above(lo), (lo, hi)
    if has_float:
        assert all(isinstance(cut.value, float) for cut in outward.cuts)
        assert nearest.cuts == float_cuts(exact.cuts, outward=False)
    else:
        assert outward == exact and nearest == exact


@settings(max_examples=100)
@given(cut_tuples_a=cut_tuples(), cut_tuples_b=cut_tuples(), seed=st.integers(0, 2 ** 32 - 1))
@example(cut_tuples_a=one(-5.1, -0.9), cut_tuples_b=one(-5.0, -1.0), seed=0)
def test_sound_at_sampled_points(cut_tuples_a, cut_tuples_b, seed):
    """M14's soundness for an op whose answer is a set of fitting points: every sampled `x` of the
    exact `X` fits and lies in the outward result (an enclosure), float operands included; every
    sampled point of the exact class's result fits"""
    rng = random.Random(seed)
    exact_a, exact_b = M.from_cuts(exact_cuts(cut_tuples_a)), M.from_cuts(exact_cuts(cut_tuples_b))
    exact = exact_a.cancel_minus(exact_b)
    outward = O.from_cuts(cut_tuples_a).cancel_minus(O.from_cuts(cut_tuples_b))
    for x in sample(exact.cuts, 20, rng):
        assert fits(x, exact_a, exact_b), x
        assert x in outward, x


def test_an_empty_b_fits_everywhere():
    """`∅ + X = ∅ ⊆ A` for every `X`, so the answer is the whole line, in both classes and with no
    warning (1788 answers entire too, except for `A = ∅`, where it answers `∅`: a divergence row)"""
    assert M(3).cancel_minus(EMPTY) == REALS and EMPTY.cancel_minus(EMPTY) == REALS
    assert type(O(0.1).cancel_minus(EMPTY)) is O and O(0.1).cancel_plus(EMPTY) == REALS
