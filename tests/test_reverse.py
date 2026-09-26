"""
reverse ops (M13e, D12): `sqr_rev`, `abs_rev`, `pown_rev`, `cosh_rev`, each `{t in x : f(t) in c}`

* the defining property on exact operands, decided at every point where it could change (the ends of
  `c`, `x` and the result, a point between each two, points a fraction of an ulp inside each rounded
  end, and ±inf): every `t` of `x` with `f(t)` in `c` is in the result (soundness), and every `t` of
  the result has `f(t)` in `c` (maximality's converse), except inside the slack of an irrational
  end, which is the open double just past the true end, never more (tightness). `f(t)` comes from an
  oracle written here from the definitions, not from the library's set ops: `t ** n` exactly, with
  `(±inf) ** n` by parity for n > 0, 0 for n < 0, and no value at 0 for n < 0; cosh through
  `intervals.elementary`, whose values have their own oracles (tests/test_elementary.py,
  tests/test_oracle_flint.py)
* at set level: the largest set, `T ⊆ rev(C)` whenever `f(T) ⊆ C`, with the library's `f` on sets;
  `f(rev(f(T))) = f(T)` where the inverse is exact (all but cosh); isotone in `c` and in `x`;
  distributing over a union of `c`; even and odd symmetry; `sqr_rev` is `pown_rev(., 2)`,
  `pown_rev(., 1)` is `c ∩ x`, `pown_rev(., 0)` is `x` or nothing
* float operands: an `OutwardMultiInterval` result holds the exact result of the same doubles and
  adds no double strictly inside what it adds (tight); a `MultiInterval` one has float ends within one
  double of it; soundness at sampled points in both classes
* the itf1788 vectors of these ops run in tests/itf1788 (476, 56 of them the rows `_POWN_REV_ROWS`);
  the ones that matter are `@example`s here
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
from intervals import EmptySetPropagationWarning
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import REALS
from intervals import abs_rev
from intervals import cosh_rev
from intervals import pown_rev
from intervals import sqr_rev
from intervals import elementary
from intervals.errors import IntervalWarning
from intervals.kernel import contains_point
from intervals.kernel import intersection
from intervals.kernel import is_subset
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from intervals.kernel import union
from intervals.rounding import DOWN
from intervals.rounding import UP
from intervals.rounding import exact_cuts
from intervals.rounding import is_float
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import midpoint

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
MAX = 1.7976931348623157e308
H = float.fromhex


def one(lo, hi, lo_closed=True, hi_closed=True):
    return normalize([piece(lo, hi, lo_closed, hi_closed)])


def v1788(lo, hi):
    """a 1788 literal under the itf1788 input rule, its doubles held exactly (an unbounded end is open)"""
    return one(_exact(lo), _exact(hi), lo != -INF, hi != INF)


def _exact(x):
    return Fraction(x) if is_float(x) else x


ENTIRE_1788 = v1788(-INF, INF)
ALL = REALS.cuts  # [-inf, inf], as a cut tuple


def _quiet(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return fn(*args)


# THE OPS, AND f AT A POINT FROM THE DEFINITIONS

def rev(op, c, x=REALS):
    """the op on MultiIntervals: an empty operand must warn and give the empty set, and nothing else
    may warn (the suite turns the library's warnings into errors)"""
    if not c or not x:
        with pytest.warns(EmptySetPropagationWarning):
            result = _rev(op, c, x)
        assert result == EMPTY
        return result
    return _rev(op, c, x)


def _rev(op, c, x):
    name, n = op
    if name == 'sqr':
        return sqr_rev(c, x)
    if name == 'abs':
        return abs_rev(c, x)
    if name == 'cosh':
        return cosh_rev(c, x)
    return pown_rev(c, n, x)


def forward(op, a: MultiInterval) -> MultiInterval:
    """the library's f on a set"""
    name, n = op
    return _quiet(lambda: a ** 2 if name == 'sqr' else abs(a) if name == 'abs' else a.cosh() if name == 'cosh'
                  else a ** n)


def is_even(op) -> bool:
    return op[0] != 'pown' or op[1] % 2 == 0


def exact_inverse(op) -> bool:
    """the inverse is exact at f's values of exact points (not cosh)"""
    return op[0] != 'cosh'


def no_value(op):
    """the points where f has no value"""
    return M(0) if op[0] == 'pown' and op[1] < 0 else EMPTY


def f_at(op, t):
    """
    f(t) for an exact t (int, Fraction, ±inf), from the definitions: an exact value, None where f has
    no value, or for cosh away from 0 the pair of doubles strictly around the irrational value
    """
    name, n = op
    if name == 'abs':
        return abs(t)
    if name == 'sqr':
        return INF if t in (INF, -INF) else Fraction(t) ** 2
    if name == 'cosh':
        if t in (INF, -INF):
            return INF
        if t == 0:
            return 1
        return elementary.rounded('cosh', Fraction(t), DOWN), elementary.rounded('cosh', Fraction(t), UP)
    if n == 0:
        return 1
    if t in (INF, -INF):
        return 0 if n < 0 else INF if n % 2 == 0 else t
    if t == 0 and n < 0:
        return None
    return Fraction(t) ** n


def value_in(op, t, c):
    """True if f(t) is in c, False if not (or f has no value at t), None if the oracle cannot tell"""
    v = f_at(op, t)
    if v is None:
        return False
    if not isinstance(v, tuple):
        return contains_point(c, v)
    enclosure = one(*v, False, False)  # the value lies strictly between two adjacent doubles
    if is_subset(enclosure, c):
        return True
    if not intersection(enclosure, c):
        return False
    return None  # an end of c lies strictly between the two doubles


def in_slack(t, result: MultiInterval) -> bool:
    """t lies strictly between an open double end of its piece and the next double inward: the room an
    irrational end's enclosure has, and no more"""
    for lo, lo_closed, hi, hi_closed in pieces(result.cuts):
        if not lo_closed and isinstance(lo, float) and lo < t < math.nextafter(lo, INF) and t < hi:
            return True
        if not hi_closed and isinstance(hi, float) and math.nextafter(hi, -INF) < t < hi and t > lo:
            return True
    return False


def probes(*cut_tuples_):
    """every end value, a point between each two, one beyond each end, ±inf, and points a fraction
    of an ulp on each side of every float end"""
    finite = sorted({Fraction(cut.value) for cuts in cut_tuples_ for cut in cuts if cut.value not in (INF, -INF)})
    out = [-INF, INF, 0, *finite]
    if finite:
        out += [finite[0] - 1, finite[-1] + 1]
        out += [(a + b) / 2 for a, b in zip(finite, finite[1:])]
    for cuts in cut_tuples_:
        for cut in cuts:
            if is_float(cut.value):
                for toward in (INF, -INF):
                    other = math.nextafter(cut.value, toward)
                    if math.isfinite(other):
                        out.append(midpoint(Fraction(cut.value), Fraction(other)))
    return out


# the ops the properties run over: (name, n), n being pown's exponent
OPS = [('sqr', 2), ('abs', 1), ('cosh', None)] + [('pown', n) for n in (-8, -7, -3, -2, -1, 0, 1, 2, 3, 4, 7, 8)]
ops = st.sampled_from(OPS)


# EXAMPLES

@pytest.mark.parametrize('op, c, x, want', [
    (('sqr', 2), '[1, 4]', None, '[-2, -1] | [1, 2]'),  # 1788 answers the hull, [-2, 2]
    (('sqr', 2), '[0, 4]', None, '[-2, 2]'),
    (('sqr', 2), '(0, 4)', None, '(-2, 0) | (0, 2)'),  # 0 ** 2 = 0 is not in (0, 4)
    (('sqr', 2), '[-10, -1]', None, '{}'),  # sqrRev [-10.0,-1.0] = [empty]
    (('sqr', 2), '[0, inf)', None, '(-inf, inf)'),  # inf ** 2 = inf is not in [0, inf)
    (('sqr', 2), '[0, inf]', None, '[-inf, inf]'),
    (('sqr', 2), '[inf]', None, '[-inf] | [inf]'),
    (('sqr', 2), '[1, 4]', '[-1, 10]', '[-1] | [1, 2]'),
    (('sqr', 2), '[0, 25]', '[-4.1, 6]', '[-4.1, 5]'),  # sqrRevBin [0.0,25.0] [-4.1,6.0] = [-4.1,5.0]
    (('sqr', 2), '[9/4]', None, '[-3/2] | [3/2]'),  # rational roots stay exact
    (('abs', 1), '[-1, 1) | (2, 3]', None, '[-3, -2) | (-1, 1) | (2, 3]'),
    (('abs', 1), '[-1.1, 0]', None, '[0]'),  # absRev [-1.1,0.0] = [0.0,0.0]
    (('abs', 1), '[1, inf]', '[-inf, 0]', '[-inf, -1]'),
    (('abs', 1), '[inf]', None, '[-inf] | [inf]'),
    (('pown', 0), '[1]', None, '[-inf, inf]'),  # t ** 0 = 1 everywhere, ±inf and 0 included
    (('pown', 0), '[-1, 5]', '[2, 3)', '[2, 3)'),
    (('pown', 0), '(1, 10]', None, '{}'),
    (('pown', 1), '[-3, 2)', '[0, 5]', '[0, 2)'),  # the identity: c ∩ x
    (('pown', 3), '[-8, 27]', None, '[-2, 3]'),
    (('pown', 3), '[-inf]', None, '[-inf]'),
    (('pown', 4), '[1/16, 16]', None, '[-2, -1/2] | [1/2, 2]'),
    (('pown', -1), '[1/2, 2]', None, '[1/2, 2]'),
    (('pown', -1), '(0, 2]', None, '[1/2, inf)'),  # 1/inf = 0 is not in (0, 2]
    (('pown', -1), '[0, 2]', None, '[-inf] | [1/2, inf]'),  # (-inf) ** -1 = 0 too
    (('pown', -1), '[-inf, inf]', None, '[-inf, 0) | (0, inf]'),  # 0 alone has no value
    (('pown', -1), '[inf]', None, '{}'),  # nothing reaches inf: 0 has no value
    (('pown', -2), '[0]', None, '[-inf] | [inf]'),  # pownRev [0.0,0.0] -2 = [empty]: a row
    (('pown', -2), '[0]', '(-inf, inf)', '{}'),  # ... with 1788's x it matches
    (('pown', -2), '[1/4, 4]', None, '[-2, -1/2] | [1/2, 2]'),
    (('pown', -2), '[1, inf)', None, '[-1, 0) | (0, 1]'),
    (('pown', -3), '[-1/8, 0]', None, '[-inf, -2] | [inf]'),
    (('cosh', None), '[1]', None, '[0]'),
    (('cosh', None), '[0, 1)', None, '{}'),
    (('cosh', None), '[1, inf)', None, '(-inf, inf)'),
    (('cosh', None), '[inf]', None, '[-inf] | [inf]'),
])
def test_examples(op, c, x, want):
    x = REALS if x is None else M.parse(x)
    assert rev(op, M.parse(c), x) == M.parse(want)
    assert rev(op, O.parse(c), x) == O.parse(want)  # exact operands: the same in both classes


def test_irrational_ends_are_open_one_ulp_enclosures():
    r2 = math.sqrt(2)  # the double nearest to sqrt(2), just above it
    below = math.nextafter(r2, 0)
    assert sqr_rev(M(2)) == M.from_pieces([(-r2, -below, False, False), (below, r2, False, False)])
    assert sqr_rev(M(1, 2)) == M.from_pieces([(-r2, -1, False, True), (1, r2, True, False)])
    # to nearest, a float operand's end is the nearest double, closed (flags kept, as the functions do)
    assert sqr_rev(M(2.0)) == M.from_pieces([(-r2, -r2), (r2, r2)])
    assert sqr_rev(O(2.0)) == sqr_rev(M(2))
    # a float operand's rational end that is not a double: outward, both ends moved, so both open
    third = pown_rev(O(3.0), -1)
    assert third == O.from_pieces([(0.3333333333333333, 0.33333333333333337, False, False)])
    assert pown_rev(M(3.0), -1) == M(0.3333333333333333)
    # to nearest, an open piece whose two ends round to one double is that double, closed (the
    # functions' rule), not dropped: sqrt of (2, 2 + ulp) is within half an ulp of sqrt(2)
    tiny = M.from_pieces([(2.0, math.nextafter(2.0, 3), False, False)])
    assert sqr_rev(tiny) == M(-r2) | M(r2)
    acosh2 = cosh_rev(M(2), M(0, INF))
    assert not acosh2.inf_closed and not acosh2.sup_closed and acosh2.sup == math.nextafter(acosh2.inf, INF)


def test_class_and_coercion():
    assert type(sqr_rev(O(1.0, 4.0))) is O and type(sqr_rev(M(1, 4), O(0.0, 5.0))) is O
    assert type(sqr_rev(M(1, 4))) is M and type(pown_rev(4, 2, 0)) is M
    assert pown_rev(4, 2, 0) == EMPTY and pown_rev(4, 2) == M(-2) | M(2) and abs_rev(3, 3) == M(3)
    with pytest.raises(TypeError):
        sqr_rev('[1, 4]')
    with pytest.raises(TypeError):
        cosh_rev(M(1, 4), None)
    for n in (2.0, True, Fraction(2), None):
        with pytest.raises(TypeError):
            pown_rev(M(1, 4), n)


def test_an_empty_operand_warns():
    for call in (lambda: sqr_rev(EMPTY), lambda: abs_rev(M(1), EMPTY), lambda: pown_rev(EMPTY, -2),
                 lambda: cosh_rev(O(1.0), O())):
        with pytest.warns(EmptySetPropagationWarning):
            assert call() == EMPTY
    assert type(_quiet(lambda: cosh_rev(O(1.0), O()))) is O


def test_no_value_and_no_solution_do_not_warn():
    # the suite turns the library's warnings into errors, so these pass only if nothing is emitted
    assert pown_rev(M(1, INF), -2) == M.parse('[-1, 0) | (0, 1]')  # 0 has no value (not inf): no warning
    assert sqr_rev(M(-2, -1)) == EMPTY and cosh_rev(M(0, 1), M(5, 6)) == EMPTY


# THE DEFINING PROPERTY: exactly the t with f(t) in c

@settings(max_examples=200)
@given(op=ops, cut_tuples_c=exact_cut_tuples, cut_tuples_x=exact_cut_tuples)
@example(op=('sqr', 2), cut_tuples_c=v1788(H('0X1.47AE147AE147BP-7'), H('0X1.47AE147AE147CP-7')),
         cut_tuples_x=ENTIRE_1788)  # rev.itl:35
@example(op=('sqr', 2), cut_tuples_c=v1788(0.0, H('0X1.FFFFFFFFFFFE1P+1')), cut_tuples_x=v1788(-0.1, INF))  # :52
@example(op=('abs', 1), cut_tuples_c=v1788(0.0, 1.0), cut_tuples_x=v1788(-0.5, 2.0))  # abs_rev.itl:29
@example(op=('abs', 1), cut_tuples_c=v1788(1.0, INF), cut_tuples_x=v1788(-INF, 0.0))  # abs_rev.itl:35
@example(op=('cosh', None), cut_tuples_c=v1788(H('0X1.8B07551D9F55P+0'), H('0X1.89BCA168970C6P+432')),
         cut_tuples_x=v1788(-INF, 0.0))  # rev.itl:762, the end just above -1
@example(op=('cosh', None), cut_tuples_c=v1788(1.0, 1.0), cut_tuples_x=v1788(1.0, INF))  # :760
@example(op=('pown', 3), cut_tuples_c=v1788(-MAX, -MAX), cut_tuples_x=ENTIRE_1788)  # :189
@example(op=('pown', -1), cut_tuples_c=v1788(H('0X0.4P-1022'), H('0X0.4000000000001P-1022')),
         cut_tuples_x=ENTIRE_1788)  # :246, past the float range
@example(op=('pown', -3), cut_tuples_c=v1788(0.0, H('0X0.0000000000001P-1022')), cut_tuples_x=ALL)  # :261, a row
@example(op=('pown', -2), cut_tuples_c=v1788(0.0, 0.0), cut_tuples_x=ALL)  # :217, a row
@example(op=('pown', -2), cut_tuples_c=v1788(H('0X1.3F0C482C977C9P-17'), INF), cut_tuples_x=ENTIRE_1788)  # :224
@example(op=('pown', -1), cut_tuples_c=v1788(-INF, -0.0), cut_tuples_x=v1788(-1.0, 1.0))  # :322
@example(op=('pown', 0), cut_tuples_c=v1788(-1.0, 5.0), cut_tuples_x=v1788(-51.0, 12.0))  # :289
@example(op=('pown', -1), cut_tuples_c=one(0, INF), cut_tuples_x=ALL)  # [inf] in c: 0 has no value
@example(op=('pown', -2), cut_tuples_c=one(1, INF), cut_tuples_x=ALL)
@example(op=('pown', 2), cut_tuples_c=one(2, 2), cut_tuples_x=one(0, Fraction(7, 5)))  # x's end inside the slack
def test_exactly_the_points_with_f_in_c(op, cut_tuples_c, cut_tuples_x):
    """soundness and the converse at every point where membership could change, tightness at each
    rounded end: the next double inward is a true point"""
    c, x = M.from_cuts(cut_tuples_c), M.from_cuts(cut_tuples_x)
    result = rev(op, c, x)
    assert result.issubset(x)
    for t in probes(cut_tuples_c, cut_tuples_x, result.cuts):
        truth = value_in(op, t, cut_tuples_c)
        if truth and t in x:
            assert t in result, (t, result)
        if t in result and truth is False:
            assert in_slack(t, result), (t, result)
    for lo, lo_closed, hi, hi_closed in pieces(result.cuts):
        for end, closed, toward in ((lo, lo_closed, INF), (hi, hi_closed, -INF)):
            if not closed and isinstance(end, float) and math.isfinite(end):
                inward = math.nextafter(end, toward)
                if lo < inward < hi:
                    assert value_in(op, inward, cut_tuples_c) is not False, (end, result)


# SET LEVEL

@settings(max_examples=150)
@given(op=ops, cut_tuples_t=exact_cut_tuples, cut_tuples_more=exact_cut_tuples, cut_tuples_x=exact_cut_tuples)
@example(op=('pown', -1), cut_tuples_t=one(1, INF), cut_tuples_more=(), cut_tuples_x=ALL)
@example(op=('pown', -2), cut_tuples_t=one(-1, 1), cut_tuples_more=(), cut_tuples_x=ALL)
@example(op=('cosh', None), cut_tuples_t=one(-1, 2), cut_tuples_more=(), cut_tuples_x=one(0, 1))
def test_the_largest_set(op, cut_tuples_t, cut_tuples_more, cut_tuples_x):
    """any T whose image lies in C is inside rev(C), and inside rev(C, X) where it is in X. in the
    outward class: cosh's image has float ends (its enclosure), which a `MultiInterval` would read as
    float operands and round to nearest, so only the outward class promises this for them"""
    t = O.from_cuts(cut_tuples_t).difference(no_value(op))
    c = forward(op, t) | O.from_cuts(cut_tuples_more)
    assert t.issubset(rev(op, c))
    x = O.from_cuts(cut_tuples_x)
    assert (t & x).issubset(rev(op, c, x))


@settings(max_examples=150)
@given(op=st.sampled_from([op for op in OPS if exact_inverse(op)]), cut_tuples_t=exact_cut_tuples)
@example(op=('pown', -1), cut_tuples_t=union(one(0, 1, False, True), one(INF, INF)))
@example(op=('pown', 3), cut_tuples_t=one(-2, Fraction(1, 2)))
def test_the_image_of_the_preimage(op, cut_tuples_t):
    """`f(rev(C)) = C` for C an image `f(T)`: nothing outside C (every end is exact here), nothing lost"""
    c = forward(op, M.from_cuts(cut_tuples_t).difference(no_value(op)))
    assert forward(op, rev(op, c)) == c


@settings(max_examples=150)
@given(op=ops, cut_tuples_c=exact_cut_tuples, cut_tuples_more=exact_cut_tuples, cut_tuples_x=exact_cut_tuples,
       cut_tuples_x_more=exact_cut_tuples)
def test_isotone(op, cut_tuples_c, cut_tuples_more, cut_tuples_x, cut_tuples_x_more):
    c, x = M.from_cuts(cut_tuples_c), M.from_cuts(cut_tuples_x)
    bigger_c, bigger_x = c | M.from_cuts(cut_tuples_more), x | M.from_cuts(cut_tuples_x_more)
    assert rev(op, c, x).issubset(rev(op, bigger_c, x))
    assert rev(op, c, x).issubset(rev(op, c, bigger_x))


@settings(max_examples=150)
@given(op=ops, cut_tuples_c=exact_cut_tuples, cut_tuples_d=exact_cut_tuples, cut_tuples_x=exact_cut_tuples)
def test_union_and_x(op, cut_tuples_c, cut_tuples_d, cut_tuples_x):
    """a preimage distributes over a union, and `x` only intersects"""
    c, d, x = M.from_cuts(cut_tuples_c), M.from_cuts(cut_tuples_d), M.from_cuts(cut_tuples_x)
    assert rev(op, c | d) == rev(op, c) | rev(op, d)
    assert rev(op, c, x) == rev(op, c) & x


@given(op=ops, cut_tuples_c=cut_tuples())
def test_symmetry(op, cut_tuples_c):
    c = M.from_cuts(cut_tuples_c)
    r = rev(op, c)
    if is_even(op):
        assert _quiet(lambda: -r) == r
    else:
        assert rev(op, _quiet(lambda: -c)) == _quiet(lambda: -r)


@given(cut_tuples_c=cut_tuples(), cut_tuples_x=cut_tuples(), cls=st.sampled_from([M, O]))
@example(cut_tuples_c=v1788(0.0, H('0X1.FFFFFFFFFFFE1P+1')), cut_tuples_x=ALL, cls=M)
def test_relations_between_the_ops(cut_tuples_c, cut_tuples_x, cls):
    """sqr_rev is pown_rev(., 2) (sqrt against rootn 2); pown_rev(., 1) is `c ∩ x`; pown_rev(., 0) is `x`
    where 1 is in c, else nothing; abs_rev is the non-negative part of c and its mirror"""
    c, x = cls.from_cuts(cut_tuples_c), cls.from_cuts(cut_tuples_x)
    assert _quiet(sqr_rev, c, x) == _quiet(pown_rev, c, 2, x)
    if c and x:
        assert pown_rev(c, 1, x) == c & x
        assert pown_rev(c, 0, x) == (x if 1 in c else cls())
        half = c & cls(0, INF)
        assert abs_rev(c, x) == (half | _quiet(lambda: -half)) & x


# FLOAT OPERANDS

float_values = st.one_of(st.sampled_from([-INF, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, INF]),
                         st.floats(-20, 20, allow_nan=False, allow_infinity=False),
                         st.floats(-1e200, 1e200, allow_nan=False, allow_infinity=False))
float_cut_tuples = cut_tuples(values=float_values)


def _widened(r: MultiInterval) -> MultiInterval:
    """each piece closed and one double wider at each finite end"""
    return M.from_pieces((math.nextafter(lo, -INF) if math.isfinite(lo) else lo,
                          math.nextafter(hi, INF) if math.isfinite(hi) else hi)
                         for lo, _, hi, _ in pieces(r.cuts))


def _first_double_above(v) -> float:
    """the smallest double > v (v exact, finite or -inf), computed without the package's rounding"""
    if v == -INF:
        return -MAX
    f = float(Fraction(v)) if abs(v) <= MAX else (INF if v > 0 else -INF)
    while f == INF or (math.isfinite(f) and Fraction(f) > v):  # down to the largest double <= v
        f = math.nextafter(f, -INF)
    return math.nextafter(f, INF)


@settings(max_examples=150)
@given(op=ops, cut_tuples_c=float_cut_tuples, cut_tuples_x=float_cut_tuples)
@example(op=('sqr', 2), cut_tuples_c=one(H('0X1.47AE147AE147BP-7'), H('0X1.47AE147AE147CP-7')),
         cut_tuples_x=ALL)  # rev.itl:35
@example(op=('cosh', None), cut_tuples_c=one(H('0X1.8B07551D9F55P+0'), H('0X1.89BCA168970C6P+432')),
         cut_tuples_x=one(-INF, 0.0))  # :762
@example(op=('pown', -1), cut_tuples_c=one(H('0X0.4P-1022'), H('0X0.4000000000001P-1022')),
         cut_tuples_x=ALL)  # :246: 1/c past max float, inf when rounded up
@example(op=('pown', 3), cut_tuples_c=one(MAX, MAX), cut_tuples_x=ALL)  # :188
@example(op=('pown', -7), cut_tuples_c=one(0.0, H('0X0.0000000000001P-1022')), cut_tuples_x=ALL)  # :276
@example(op=('pown', -1), cut_tuples_c=one(3.0, 3.0), cut_tuples_x=ALL)  # 1/3: both ends moved, open
@example(op=('sqr', 2), cut_tuples_c=one(2.0, H('0x1.0000000000001p+1'), False, False), cut_tuples_x=ALL)  # squeezed
def test_float_operands(op, cut_tuples_c, cut_tuples_x):
    """outward: the exact result of the same doubles is inside, what rounding adds holds no double
    strictly inside it, every finite end a float. to nearest: float ends, the exact result within one
    double of each piece, and every point inside the outward result's closure"""
    exact = rev(op, M.from_cuts(exact_cuts(cut_tuples_c)), M.from_cuts(exact_cuts(cut_tuples_x)))
    outward = rev(op, O.from_cuts(cut_tuples_c), O.from_cuts(cut_tuples_x))
    nearest = rev(op, M.from_cuts(cut_tuples_c), M.from_cuts(cut_tuples_x))
    assert exact.issubset(outward)
    for lo, _, hi, _ in pieces(outward.difference(exact).cuts):
        assert hi <= _first_double_above(lo), (lo, hi)
    for r in (outward, nearest):  # floats, or an exact value at an infinity (1/inf = 0) that is a double
        assert all(isinstance(cut.value, float) or cut.value == float(cut.value) for cut in r.cuts), r
    assert exact.issubset(_widened(nearest))
    closure = M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(outward.cuts))
    assert nearest.issubset(closure)
    # outward, a closed end is attained: a point of the exact result (a moved end is open)
    for lo, lo_closed, hi, hi_closed in pieces(outward.cuts):
        for end, closed in ((lo, lo_closed), (hi, hi_closed)):
            if closed:
                assert end in exact, (end, outward)
    if exact:  # nothing of the exact result vanishes to nearest, a squeezed piece included
        assert nearest


@settings(max_examples=100)
@given(op=ops, cut_tuples_c=cut_tuples(), cut_tuples_x=cut_tuples(), seed=st.integers(0, 2 ** 32 - 1))
@example(op=('sqr', 2), cut_tuples_c=one(0.0, 25.0), cut_tuples_x=one(-4.1, 6.0), seed=0)
def test_sound_at_sampled_points(op, cut_tuples_c, cut_tuples_x, seed):
    """M14's soundness: a sampled t of x with f(t) in c is in the exact result and in the outward one
    (float operands read as the rationals they are); a sampled t of the exact result has f(t) in c,
    or lies in a rounded end's slack"""
    rng = random.Random(seed)
    exact_c, exact_x = exact_cuts(cut_tuples_c), exact_cuts(cut_tuples_x)
    exact = rev(op, M.from_cuts(exact_c), M.from_cuts(exact_x))
    outward = rev(op, O.from_cuts(cut_tuples_c), O.from_cuts(cut_tuples_x))
    for t in sample(exact_x, 20, rng):
        t = _exact(t)
        if value_in(op, t, exact_c):
            assert t in exact and t in outward, t
    for t in sample(exact.cuts, 20, rng):
        t = _exact(t)
        assert value_in(op, t, exact_c) is not False or in_slack(t, exact), t


# THE ITF1788 ROWS

def _with_1788_x(v, t):
    """(ours with x omitted, ours with x = 1788's entire) for a pownRev vector"""
    c, n = t.to_ours(v.args[0]), int(v.args[1])
    return pown_rev(c, n), pown_rev(c, n, M.from_cuts(ENTIRE_1788))


def test_the_rows_differ_only_at_the_infinities():
    """each pownRev row under "degenerate infinities" (tests/itf1788 `_POWN_REV_ROWS`) matches 1788 once
    x is 1788's entire, open at ±inf, and differs from that only by ±inf: nothing else is hidden"""
    from tests.itf1788 import test_itf1788 as t
    rows = [v for v in t.VECTORS if t.key(v) in t._POWN_REV_ROWS]
    assert {t.key(v) for v in rows} == set(t._POWN_REV_ROWS) and len(rows) == 52
    for v in rows:
        ours, theirs = _with_1788_x(v, t)
        assert t.same(t.closed_hull_of_ours(theirs), t.closed_hull_of_expected(v.expected)), v.text
        assert ours != theirs and ours.difference(theirs).issubset(M(-INF) | M(INF)), v.text


def test_pown_rev_is_tighter_than_the_vector():
    """rev.itl:276, :277 (and :477, :478 decorated): 1788's end for 2 ** (1074/7) is one double outside
    the tightest enclosure. arb puts the true value strictly inside ours, so ours is right and tight;
    the vectors are rows `_POWN_REV_LOOSE_ROWS` under the proposed category "tighter than the vector"""
    flint = pytest.importorskip('flint')
    from tests.itf1788 import test_itf1788 as t
    rows = [v for v in t.VECTORS if t.key(v) in t._POWN_REV_LOOSE_ROWS]
    assert len(rows) == 4
    lo, hi = H('0x1.588cea3f093bdp+153'), H('0x1.588cea3f093bep+153')
    old = flint.ctx.prec
    flint.ctx.prec = 200
    try:
        true = flint.arb(2) ** (flint.arb(1074) / 7)
        assert true > flint.arb(lo) and true < flint.arb(hi)  # strictly between two adjacent doubles
    finally:
        flint.ctx.prec = old
    assert math.nextafter(lo, INF) == hi
    tight = M.from_pieces([(-hi, -lo, False, False), (lo, hi, False, False)])
    assert pown_rev(M(0, Fraction(2) ** -1074), -7) == M(-INF) | M.from_pieces([(lo, INF, False, True)])
    for v in rows:
        ours, theirs = _with_1788_x(v, t)
        expected = t.closed_hull_of_expected(v.expected)
        assert t.closed_hull_of_ours(theirs) != expected
        edge = expected[0] if expected[0] != -INF else expected[1]
        assert abs(edge) == math.nextafter(lo, -INF)  # 1788's: one double below the tightest
        assert theirs.hull.issubset(M.from_pieces([(-INF, -lo), (lo, INF)]))
        assert ours.difference(theirs).issubset(M(-INF) | M(INF))
    assert tight  # the exact points ±2 ** (1074/7) sit inside these open pieces
