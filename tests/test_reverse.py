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
* `mul_rev(b, c, x)` (M13e, second part; its own section below): `t` is in it iff `t ∈ x` and `{t} * b`
  meets `c`, decided exactly with the library's `*` at every quotient of an end of `c` by one of `b`;
  the largest set; isotone, distributing over unions of `b` and of `c`; symmetric; a point `b = [w]`
  is `c / w`; float operands, soundness at sampled points, and 1788's float vector `mul_rev.itl:34`
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
from intervals import mul_rev
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


def meets_x_as_d26(result: MultiInterval, nearest_all: MultiInterval, x: MultiInterval, exact: MultiInterval,
                   outward: MultiInterval):
    """to nearest, x meets the rounded preimage (`nearest_all & x`), and a part inside x that rounds
    wholly onto one double is that double, as a point (D26, fuzz-rev-inf), in x or on an end x does not
    hold: anything more than `nearest_all & x` is such a point, each within one double of a true point
    in x; and nothing the outward class finds in x vanishes: the result meets the closure of each
    piece of `outward` (the outward result in x)"""
    near_x = nearest_all & x
    assert near_x.issubset(result), (near_x, result)
    x_closure = M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(x.cuts))
    for lo, lo_closed, hi, hi_closed in pieces(result.difference(near_x).cuts):
        assert lo == hi and lo_closed and hi_closed and lo in x_closure, (lo, hi, result)
        assert exact & _widened(M(lo)), (lo, exact)
    if exact:
        assert result, exact
    no_piece_vanishes(result, outward)


def no_piece_vanishes(nearest: MultiInterval, outward: MultiInterval):
    """to nearest, each piece of the outward result (in x) keeps a point of the nearest result in its
    closure: the part of the exact result there rounds onto a double of that closure (D26)"""
    for lo, _, hi, _ in pieces(outward.cuts):
        assert nearest & M(lo, hi), ((lo, hi), nearest)


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
    # the one way out of x (D26, to nearest): a part inside x squeezed onto an end x does not hold is
    # that end, as a point
    squeezed = result.difference(x)
    x_closure = M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(x.cuts))
    for lo, lo_closed, hi, hi_closed in pieces(squeezed.cuts):
        assert lo == hi and lo_closed and hi_closed and lo in x_closure, (squeezed, result)
    for t in probes(cut_tuples_c, cut_tuples_x, result.cuts):
        truth = value_in(op, t, cut_tuples_c)
        if truth and t in x:
            assert t in result, (t, result)
        if t in result and truth is False:
            assert in_slack(t, result) or t in squeezed, (t, result)
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
@example(op=('pown', -7), cut_tuples_c=one(Fraction(1, 2), 0.5))  # fuzz-symmetry: a mixed point
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
    """each piece closed and one double wider at each end but -inf below and inf above; so a piece
    `[inf]`, an end past the largest double to nearest, is widened to `[max, inf]`"""
    return M.from_pieces((math.nextafter(lo, -INF) if lo != -INF else lo,
                          math.nextafter(hi, INF) if hi != INF else hi)
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
# fuzz-rev-inf (fuzz run 36580954134; D26): the exact result, wholly past -MAX, squeezes to [-inf],
# which an x open at -inf dropped after the rounding; x meets the preimage first, so [-inf] stays
@example(op=('pown', -1), cut_tuples_c=one(-2.225073858507203e-309, 0.0, False, False),
         cut_tuples_x=one(-INF, -2.0, False, False))
# its finite form: sqrt of [2, 2.0000000000000004] rounds to one double e, the part above e is in x = (e, 2]
@example(op=('sqr', 2), cut_tuples_c=one(2.0, 2.0000000000000004),
         cut_tuples_x=one(1.4142135623730951, 2.0, False, True))
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
    no_piece_vanishes(nearest, outward)  # nor any part of it (D26)


@pytest.mark.parametrize('lo, hi', [(3.0, 7.0), (0.1, 10.0), (1e-300, 3.0)])
def test_nearest_keeps_the_flag_of_a_moved_end(lo, hi):
    """to nearest, an end that rounding moved keeps its flag, as the forward ops' (`functions._Function.end`):
    `pown_rev(c, -1)` is `c ** -1` for c > 0, closed. outward it is open. rootn with n < 0 is the one
    branch taking a double to a rational non-double (M13e's review, 2026-09-27: no test saw this flag;
    `test_float_operands` checks values and the outward closure, and a point hides it by the squeeze)"""
    nearest, outward = pown_rev(M(lo, hi), -1), pown_rev(O(lo, hi), -1)
    assert nearest == M(lo, hi) ** -1
    assert nearest.inf_closed and nearest.sup_closed
    assert not outward.inf_closed and not outward.sup_closed
    even = pown_rev(M(4.0, 9.0), -2)  # ±[1/3, 1/2], 1/3 moved
    assert even == M.parse('[-0.5, -0.3333333333333333] | [0.3333333333333333, 0.5]')


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


def test_pown_rev_ends_crossed_by_rounding():
    """
    the preimage of a mixed piece under x ** 5: its exact end maps to 1/10 ** 6 exactly and its float end rounds to
    nearest onto 1e-06, below it (the case of fuzz run 37098878528's `rootn`, 2026-10-03, which raised here too).
    the piece is between the two values, each keeping its flag (`functions._settled`); outward never crosses
    """
    c = '(1/1000000000000000000000000000000, 1.0000000000000003e-30]'
    assert pown_rev(M.parse(c), 5) == M.parse('[1e-06, 1/1000000)')
    assert pown_rev(OutwardMultiInterval.parse(c), 5) == OutwardMultiInterval.parse('(1/1000000, 1.0000000000000002e-06)')


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


# MULTIPLICATION: mul_rev(b, c, x) = `{t in x : t * y in c for some y in b}` (M13e, second part)
#
# the defining property is decided exactly with the library's `*`, as tests/test_cancel.py decides
# cancellation with its `+`: `t` is in the result iff `{t} * b` meets `c` (`0 * ±inf` has no value).
# the result's ends are quotients of an end of `c` by one of `b`, 0 or ±inf, so probing every such
# quotient, the ends of `x`, a point between each two, one beyond each end and ±inf sees all of it

def mrev(b, c, x=REALS):
    """mul_rev on MultiIntervals: an empty operand must warn and give the empty set, and nothing else
    may warn (the 0 * ±inf corner included: no IndeterminateResultWarning)"""
    if not b or not c or not x:
        with pytest.warns(EmptySetPropagationWarning):
            result = mul_rev(b, c, x)
        assert result == EMPTY
        return result
    return mul_rev(b, c, x)


def meets(t, b: MultiInterval, c: MultiInterval) -> bool:
    """`{t} * b` meets `c`, with the library's `*`"""
    return bool(_quiet(lambda: M(t) * b) & c)


def mul_candidates(b, c, x):
    """every quotient of a finite end of c by a finite nonzero end of b, every end of x, 0, a point
    between each two, one beyond each end, and ±inf"""
    def ends(cuts):
        return {Fraction(cut.value) for cut in cuts if math.isfinite(cut.value)}
    points = sorted({v / w for v in ends(c) for w in ends(b) if w} | ends(x) | {Fraction(0)})
    between = [(p + q) / 2 for p, q in zip(points, points[1:])]
    return [-INF, INF, points[0] - 1, points[-1] + 1, *points, *between]


POINT_0, POINT_INF, POINT_MINUS_INF = one(0, 0), one(INF, INF), one(-INF, -INF)


@pytest.mark.parametrize('b, c, x, want', [
    ('[2, 4]', '[1, 8]', None, '[1/4, 4]'),
    ('[-2, -1]', '(1, 4]', None, '[-4, -1/2)'),
    ('[1, 2] | [4]', '[8]', None, '[2] | [4, 8]'),  # 8/[1, 2] and 8/4
    ('[-1, 1]', '[1, 2]', None, '(-inf, -1] | [1, inf)'),  # 1788: entire (mulRev) or the two pieces (ToPair)
    ('(0, 1]', '[1]', None, '[1, inf)'),  # 1/y for y in (0, 1]; inf * y = inf is not in [1]
    ('[0, 1]', '[1, inf]', None, '[1, inf]'),  # inf * y = inf for y in (0, 1]
    ('[1, 2]', '[0]', None, '[0]'),
    ('[0]', '[1, 2]', None, '{}'),
    ('[0]', '[0]', None, '(-inf, inf)'),  # t * 0 = 0 for finite t; inf * 0 has no value
    ('[-1, 1]', '[0]', None, '(-inf, inf)'),  # inf * y = ±inf for y != 0, never 0
    ('[0]', '[-inf, inf]', None, '(-inf, inf)'),
    ('[inf]', '[1, 2]', None, '{}'),  # t * inf is ±inf or nothing
    ('[inf]', '[0]', None, '{}'),
    ('[inf]', '[inf]', None, '(0, inf]'),  # 0 * inf has no value
    ('[-inf]', '[inf]', None, '[-inf, 0)'),
    ('[-inf, inf]', '[inf]', None, '[-inf, 0) | (0, inf]'),
    ('[0] | [inf]', '[0]', None, '(-inf, inf)'),  # y = 0 gives every finite t, y = inf nothing
    ('[0] | [inf]', '[inf]', None, '(0, inf]'),
    ('[-inf, -1]', '[inf]', None, '[-inf, 0)'),  # finite t < 0 by y = -inf, and -inf by any y < 0
    ('[1, inf]', '[3]', '[0, 10]', '(0, 3]'),  # 3/y for y in [1, inf); 0 * inf has no value
    ('[1, inf)', '[0]', None, '[0]'),
    ('[-2, 11/10]', '[-21/10, -2/5]', '[-1, 1]', '[-1, -4/11] | [1/5, 1]'),  # rev.itl:977's shape, exactly
])
def test_mul_rev_examples(b, c, x, want):
    x = REALS if x is None else M.parse(x)
    assert mrev(M.parse(b), M.parse(c), x) == M.parse(want)
    assert mrev(O.parse(b), O.parse(c), x) == O.parse(want)  # exact operands: the same in both classes


@settings(max_examples=300, deadline=None)
@given(b=exact_cut_tuples, c=exact_cut_tuples, x=st.one_of(st.just(ALL), exact_cut_tuples))
@example(b=v1788(-2.0, 1.1), c=v1788(-2.1, -0.4), x=ALL)  # mul_rev.itl:32, two pieces
@example(b=v1788(0.0, 0.0), c=v1788(-2.1, -0.4), x=ALL)  # :35, empty
@example(b=v1788(-INF, -0.1), c=v1788(-2.1, -0.4), x=ALL)  # :36, (0, 21]: 1788 closes it at 0
@example(b=ENTIRE_1788, c=v1788(-2.1, -0.4), x=ALL)  # :42, the gap at 0
@example(b=v1788(-2.0, 1.1), c=v1788(0.0, 0.0), x=ALL)  # :102, (-inf, inf)
@example(b=v1788(-INF, -0.1), c=v1788(0.0, 0.0), x=ALL)  # :106, [0]
@example(b=v1788(-INF, 1.1), c=v1788(0.04, INF), x=ALL)  # :193
@example(b=v1788(-INF, -0.1), c=v1788(0.0, 0.12), x=v1788(0.0, 0.12))  # rev.itl:979, mulRevTen = [0.0, 0.0]
@example(b=POINT_INF, c=POINT_INF, x=ALL)  # (0, inf]: 0 * inf has no value
@example(b=POINT_INF, c=POINT_0, x=ALL)  # nothing
@example(b=POINT_MINUS_INF, c=one(-INF, 0), x=ALL)  # (0, inf]: t > 0 by y = -inf; 0 * -inf has no value
@example(b=union(POINT_0, POINT_INF), c=POINT_0, x=ALL)
@example(b=POINT_0, c=union(POINT_MINUS_INF, POINT_INF), x=ALL)  # nothing: ±inf * 0 has no value
@example(b=one(-1, 1), c=union(POINT_MINUS_INF, POINT_INF), x=ALL)  # ±inf by y != 0; no finite t
@example(b=one(0, INF), c=one(3, 3), x=one(0, 10))
def test_mul_rev_is_exactly_the_points_that_fit(b, c, x):
    """`t` is in the result iff `t` is in x and `{t} * b` meets `c`: soundness and maximality at once"""
    B, C, X = M.from_cuts(b), M.from_cuts(c), M.from_cuts(x)
    result = mrev(B, C, X)
    assert result.issubset(X)
    for t in mul_candidates(b, c, x):
        assert (t in result) == (t in X and meets(t, B, C)), (t, result)


def _undefined_against(b: MultiInterval) -> MultiInterval:
    """the t with `{t} * b` empty: 0 if b has no finite point, ±inf if b has no point but 0, and
    every t for an empty b"""
    if not b:
        return REALS
    out = EMPTY if b & M.parse('(-inf, inf)') else M(0)
    return out if b.difference(M(0)) else out | M(-INF) | M(INF)


@settings(max_examples=150, deadline=None)
@given(t=exact_cut_tuples, b=exact_cut_tuples, more=exact_cut_tuples, x=exact_cut_tuples)
@example(t=one(-INF, INF), b=one(0, 0), more=(), x=ALL)
@example(t=one(0, 1), b=one(INF, INF), more=(), x=ALL)
def test_mul_rev_the_largest_set(t, b, more, x):
    """any T with `T * B ⊆ C` is inside `mul_rev(B, C)` (the t with `{t} * B` empty aside), and
    inside `mul_rev(B, C, X)` where it is in X"""
    B = M.from_cuts(b)
    T = M.from_cuts(t).difference(_undefined_against(B))
    C = _quiet(lambda: T * B) | M.from_cuts(more)
    assert T.issubset(mrev(B, C))
    X = M.from_cuts(x)
    assert (T & X).issubset(mrev(B, C, X))


@settings(max_examples=150, deadline=None)
@given(b=exact_cut_tuples, b_more=exact_cut_tuples, c=exact_cut_tuples, c_more=exact_cut_tuples,
       x=exact_cut_tuples, x_more=exact_cut_tuples)
def test_mul_rev_isotone_and_distributive(b, b_more, c, c_more, x, x_more):
    """isotone in each operand, distributing over a union of b and of c (an existential preimage), and
    x only intersects"""
    B, B2, C, C2, X, X2 = (M.from_cuts(v) for v in (b, b_more, c, c_more, x, x_more))
    r = mrev(B, C, X)
    assert r.issubset(mrev(B | B2, C, X)) and r.issubset(mrev(B, C | C2, X)) and r.issubset(mrev(B, C, X | X2))
    assert mrev(B | B2, C) == mrev(B, C) | mrev(B2, C)
    assert mrev(B, C | C2) == mrev(B, C) | mrev(B, C2)
    assert r == mrev(B, C) & X


@settings(deadline=None)
@given(b=cut_tuples(), c=cut_tuples(), cls=st.sampled_from([M, O]))
def test_mul_rev_symmetry(b, c, cls):
    """`t * y = (-t) * (-y)` and `-(t * y) = (-t) * y`, in both classes (rounding is symmetric)"""
    B, C = cls.from_cuts(b), cls.from_cuts(c)
    r = mrev(B, C)
    neg = _quiet(lambda: -r)
    assert mrev(_quiet(lambda: -B), C) == neg
    assert mrev(B, _quiet(lambda: -C)) == neg


nonzero_values = st.one_of(st.integers(-20, 20), st.fractions(-20, 20, max_denominator=6),
                           st.floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)).filter(bool)


@settings(deadline=None)
@given(y=nonzero_values, c=cut_tuples(), x=cut_tuples(), cls=st.sampled_from([M, O]))
@example(y=10, c=one(1.0, 2.0, False, False), x=ALL, cls=M)  # to nearest (0.1, 0.2), 1/10 rounded up
# fuzz run 37187049245 (2026-10-04): the exact (-inf, -2 / 5e-324) in x rounds wholly onto -inf: [-inf] (D26)
@example(y=5e-324, c=one(-INF, -2, False, False), x=one(-INF, -2, False, False), cls=M)
def test_mul_rev_by_a_point(y, c, x, cls):
    """a finite point `b = [y]`, y != 0, is division, `t * y ∈ c` iff `t ∈ c / y` (±inf included), in
    both classes; `b = [0]` is every finite t where 0 is in c, else nothing. to nearest, x meets the
    preimage before the rounding (D26): where that differs from `c / y & x`, the extra is a part in x
    squeezed onto one double (`meets_x_as_d26`)"""
    C, X = cls.from_cuts(c), cls.from_cuts(x)
    if not C or not X:
        return
    result = mul_rev(cls(y), C, X)
    near = _quiet(lambda: C / y)
    if result != near & X:  # to nearest only: x met before the rounding (D26)
        assert cls is M, (result, near & X)
        exact = mul_rev(M(Fraction(y)), M.from_cuts(exact_cuts(c)), M.from_cuts(exact_cuts(x)))
        outward = mul_rev(O(y), O.from_cuts(c), O.from_cuts(x))
        meets_x_as_d26(result, near, X, exact, outward)
    assert mul_rev(cls(0), C, X) == (cls.parse('(-inf, inf)') & X if 0 in C else cls())


@settings(max_examples=150, deadline=None)
@given(b=float_cut_tuples, c=float_cut_tuples, x=float_cut_tuples)
@example(b=one(0.01, 1.1), c=one(-2.1, -0.4), x=ALL)  # mul_rev.itl:34, both ends rounded
@example(b=one(10.0, 10.0), c=one(1.0, 2.0, False, False), x=one(0.1, 0.1))  # (1/10, 1/5); 0.1 > 1/10 is in it
@example(b=one(3.0, 3.0), c=one(1.0, 1.0), x=ALL)  # 1/3, one point: outward two doubles, open
@example(b=one(MAX, MAX), c=one(H('0x0.0000000000001p-1022'), 1.0), x=ALL)  # underflow below the least subnormal
@example(b=one(H('0x0.0000000000001p-1022'), 1.0), c=one(MAX, MAX), x=ALL)  # past max float
@example(b=one(-INF, H('0x0.0000a7c5ac472p-1022'), False, False), c=one(0.5, 1.0, False, False),
         x=())  # CI 2026-09-27: an end past max float, to nearest the piece [inf]
def test_mul_rev_float_operands(b, c, x):
    """outward: the exact result of the same doubles is inside, what rounding adds holds no double
    strictly inside it, a closed end is a point of the exact result. to nearest, x omitted (an end of
    x can fall in the half ulp a rounded end moved, as `0.1` in `(1/10, 1/5)` rounded to `(0.1, 0.2)`):
    the exact result within one double of each piece, inside the outward closure, not empty if it is
    not; and x meets it after the rounding, a part squeezed onto an end x excludes kept as that end
    (D26: `meets_x_as_d26`)"""
    B, C, X = M.from_cuts(exact_cuts(b)), M.from_cuts(exact_cuts(c)), M.from_cuts(exact_cuts(x))
    exact = mrev(B, C, X)
    outward = mrev(O.from_cuts(b), O.from_cuts(c), O.from_cuts(x))
    assert exact.issubset(outward)
    for lo, _, hi, _ in pieces(outward.difference(exact).cuts):
        assert hi <= _first_double_above(lo), (lo, hi)
    for lo, lo_closed, hi, hi_closed in pieces(outward.cuts):
        for end, closed in ((lo, lo_closed), (hi, hi_closed)):
            if closed:
                assert end in exact, (end, outward)
    exact_all = mrev(B, C)
    nearest_all = mrev(M.from_cuts(b), M.from_cuts(c))
    outward_all = mrev(O.from_cuts(b), O.from_cuts(c))
    for r in (outward, nearest_all):  # floats, or an exact 0 or ±inf, which is a double
        assert all(isinstance(cut.value, float) or cut.value == float(cut.value) for cut in r.cuts), r
    assert exact_all.issubset(_widened(nearest_all))
    assert nearest_all.issubset(M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(outward_all.cuts)))
    if exact_all:
        assert nearest_all
    meets_x_as_d26(mrev(M.from_cuts(b), M.from_cuts(c), M.from_cuts(x)), nearest_all, M.from_cuts(x), exact, outward)


def test_mul_rev_1788_float_vector():
    """mul_rev.itl:34: the quotients of the doubles, rounded outward, are 1788's two doubles; nothing
    attains a moved end"""
    r = mul_rev(O(0.01, 1.1), O(-2.1, -0.4))
    assert (r.inf, r.sup) == (-H('0X1.A400000000001P+7'), -H('0X1.745D1745D1745P-2'))
    assert not r.inf_closed and not r.sup_closed
    exact = mul_rev(M(Fraction(0.01), Fraction(1.1)), M(Fraction(-2.1), Fraction(-0.4)))
    assert exact == M(Fraction(-2.1) / Fraction(0.01), Fraction(-0.4) / Fraction(1.1)) and exact.issubset(r)


@settings(max_examples=100, deadline=None)
@given(b=cut_tuples(), c=cut_tuples(), x=cut_tuples(), seed=st.integers(0, 2 ** 32 - 1))
@example(b=one(-2.0, 1.1), c=one(0.04, INF), x=one(0.04, INF), seed=0)  # rev.itl:980
def test_mul_rev_sound_at_sampled_points(b, c, x, seed):
    """M14's soundness: a sampled t of x whose `{t} * b` meets c is in the exact result and in the
    outward one (float operands read as the rationals they are); a sampled t of the exact result fits"""
    rng = random.Random(seed)
    B, C, X = (M.from_cuts(exact_cuts(v)) for v in (b, c, x))
    exact = mrev(B, C, X)
    outward = mrev(O.from_cuts(b), O.from_cuts(c), O.from_cuts(x))
    for t in sample(X.cuts, 20, rng):
        t = _exact(t)
        if meets(t, B, C):
            assert t in exact and t in outward, t
    for t in sample(exact.cuts, 20, rng):
        assert meets(_exact(t), B, C), t


def test_mul_rev_class_coercion_and_warnings():
    assert type(mul_rev(O(1.0), M(2), M(0, 5))) is O and type(mul_rev(M(1), M(2), O(0.0, 5.0))) is O
    assert type(mul_rev(M(1), O(2.0))) is O and type(mul_rev(M(1), M(2))) is M
    assert mul_rev(2, 4) == M(2) and mul_rev(2, 4, 0) == EMPTY and mul_rev(0, 0, 5) == M(5)
    for args in (('[1, 2]', M(1)), (M(1), '[1, 2]'), (M(1), M(1), None)):
        with pytest.raises(TypeError):
            mul_rev(*args)
    for args in ((EMPTY, M(1)), (M(1), EMPTY), (M(1), M(1), EMPTY), (O(), O())):
        with pytest.warns(EmptySetPropagationWarning):
            assert mul_rev(*args) == EMPTY
    assert type(_quiet(lambda: mul_rev(O(), M(1)))) is O
    # no solution, and the 0 * inf corner, warn nothing (the suite turns warnings into errors)
    assert mul_rev(M(INF), M(0)) == EMPTY and mul_rev(M(0), M(1, 2)) == EMPTY
    assert mul_rev(M(0) | M(INF), M(0)) == M.parse('(-inf, inf)')


# PERIODIC: sin_rev, cos_rev, tan_rev (M13e, third part; D12)
#
# `{t in x : f(t) in c}` with f the library's sin, cos or tan at a point: no value at ±inf, none at
# tan's poles. f(t) at an exact t comes from `intervals.elementary` (its own oracles:
# tests/test_elementary.py, tests/test_oracle_flint.py), as cosh's does above; the ends' own
# rounding, `elementary.rounded_inverse_trig`, is checked against arb below. a bounded x gets the
# exact pieces; a piece of x unbounded in the reals (or past ENUMERATION_CAP pieces) gets the hull of
# its part and a HullWarning, which `trev` requires exactly where `trig_hulls` says

from intervals import HullWarning  # noqa: E402
from intervals import cos_rev  # noqa: E402
from intervals import sin_rev  # noqa: E402
from intervals import tan_rev  # noqa: E402
from intervals.rounding import NEAREST  # noqa: E402
from intervals.steps import ENUMERATION_CAP  # noqa: E402
from intervals.cuts import Cut  # noqa: E402
from intervals.cuts import Side  # noqa: E402
from intervals.reverse import negate  # noqa: E402

TRIG = {'sin': sin_rev, 'cos': cos_rev, 'tan': tan_rev}
trig_names = st.sampled_from(sorted(TRIG))
FINITE = one(-INF, INF, False, False)
_TRIG_IMAGE = {'sin': one(-1, 1), 'cos': one(-1, 1), 'tan': FINITE}
BOX = M(-21, 21)  # holds every finite value the exact strategies draw, and more than six periods


def trig_hulls(name, c: MultiInterval, x: MultiInterval) -> bool:
    """whether the op hulls (and warns), for operands drawn from the strategies (never past the cap):
    c has a solution but not every finite t is one (sin, cos: c holds [-1, 1]; tan never, its poles),
    and x's finite part has a piece unbounded in the reals"""
    image = _TRIG_IMAGE[name]
    if not intersection(c.cuts, image) or (name != 'tan' and is_subset(image, c.cuts)):
        return False
    return any(lo == -INF or hi == INF for lo, _, hi, _ in pieces(intersection(x.cuts, FINITE)))


def trev(name, c, x=REALS):
    """the op on MultiIntervals: an empty operand warns and gives ∅, a HullWarning is emitted exactly
    where `trig_hulls` says, and nothing else warns (the suite turns the library's warnings into errors)"""
    if not c or not x:
        with pytest.warns(EmptySetPropagationWarning):
            result = TRIG[name](c, x)
        assert result == EMPTY
        return result
    if trig_hulls(name, c, x):
        with pytest.warns(HullWarning):
            return TRIG[name](c, x)
    return TRIG[name](c, x)


def trig_at(name, t):
    """f(t) for an exact t: None at ±inf (no value there), the exact value where rational (0 at 0, cos 1),
    else the two adjacent doubles strictly around it"""
    if t in (INF, -INF):
        return None
    t = Fraction(t)
    v = elementary.exact(name, t)
    if v is not None:
        return v
    return elementary.rounded(name, t, DOWN), elementary.rounded(name, t, UP)


def trig_value_in(name, t, c) -> bool:
    """True if f(t) is in c, False if not (or no value), None if an end of c lies inside f(t)'s enclosure"""
    v = trig_at(name, t)
    if v is None:
        return False
    if not isinstance(v, tuple):
        return contains_point(c, v)
    enclosure = one(*v, False, False)
    if is_subset(enclosure, c):
        return True
    if not intersection(enclosure, c):
        return False
    return None


def trig_in_slack(t, result: MultiInterval) -> bool:
    """t is in a piece of the result and strictly between an open double end of it and the next double
    inward (`in_slack`, but t may be the piece's other end: x can start inside a rounded end's slack)"""
    for p in result.pieces:
        if t not in p:
            continue
        lo, hi = p.inf, p.sup
        if not p.inf_closed and isinstance(lo, float) and lo < t < math.nextafter(lo, INF):
            return True
        if not p.sup_closed and isinstance(hi, float) and math.nextafter(hi, -INF) < t < hi:
            return True
    return False


def _bounded_part(x: MultiInterval) -> MultiInterval:
    """the pieces of x's finite part that are bounded: where the result is exact"""
    return M.from_pieces((lo, hi, lc, hc) for lo, lc, hi, hc in pieces(intersection(x.cuts, FINITE))
                         if lo != -INF and hi != INF)


def _arb_of(v, flint):
    """an exact value (int, Fraction, float) as an arb, exactly"""
    v = Fraction(v)
    return flint.arb(flint.fmpq(v.numerator, v.denominator))


# EXAMPLES

@pytest.mark.parametrize('name, c, x, want', [
    ('sin', '[0]', '[-1, 4]', '[0] | (3.141592653589793, 3.1415926535897936)'),
    ('sin', '[1]', '[0, 2]', '(1.5707963267948966, 1.5707963267948968)'),  # pi/2: the extremum, one point
    ('sin', '[-1, 1]', None, '(-inf, inf)'),  # every finite t: no hull, no warning; ±inf have no value
    ('sin', '[-2, 2]', '[-inf, 5]', '(-inf, 5]'),
    ('sin', '[2]', None, '{}'),
    ('sin', '[1/2, 1]', '[0, 2]', '(0.5235987755982988, 2]'),  # sin(2) > 1/2; the branches meet at pi/2
    ('cos', '[1]', '[-1, 7]', '[0] | (6.283185307179586, 6.283185307179587)'),
    ('cos', '[-1]', '[0, 4]', '(3.141592653589793, 3.1415926535897936)'),
    ('cos', '[-1, 1]', '[2, inf)', '[2, inf)'),
    ('cos', '(1/2, 1]', '[-2, 2]', '(-1.0471975511965979, 1.0471975511965979)'),
    ('tan', '[0]', '[-4, 4]',
     '(-3.1415926535897936, -3.141592653589793) | [0] | (3.141592653589793, 3.1415926535897936)'),
    ('tan', '[1]', '[0, 4]', '(0.7853981633974483, 0.7853981633974484) | (3.9269908169872414, 3.926990816987242)'),
    ('tan', '[inf]', None, '{}'),  # the poles have no value: nothing reaches ±inf
    ('tan', '[-inf, inf]', '[1, 2]', '[1, 2]'),  # the pole pi/2 is inside both branches' enclosures
    ('tan', '(-inf, 0]', '[0, 3]', '[0] | (1.5707963267948966, 3]'),  # tan(3) < 0
])
def test_trig_rev_examples(name, c, x, want):
    x = REALS if x is None else M.parse(x)
    assert trev(name, M.parse(c), x) == M.parse(want)
    assert trev(name, O.parse(c), x) == O.parse(want)  # exact operands: the same in both classes


def test_d12_example():
    """sin_rev([1/2, 1], [0, 20]) has 4 pieces (D12): [pi/6, 5pi/6] + 2k pi for k = 0, 1, 2, and the
    last cut at 20"""
    r = sin_rev(M(Fraction(1, 2), 1), M(0, 20))
    assert len(r.pieces) == 4
    for k, p in enumerate(r.pieces):
        lo, hi = 2 * k * math.pi + math.pi / 6, 2 * k * math.pi + 5 * math.pi / 6
        assert not p.inf_closed and abs(p.inf - lo) <= 2 * math.ulp(lo)
        if k < 3:
            assert not p.sup_closed and abs(p.sup - hi) <= 2 * math.ulp(hi)
    assert r.pieces[-1].sup == 20 and r.pieces[-1].sup_closed  # sin(20) = 0.91...


def test_trig_rev_hull_past_the_cap():
    """1000 pieces are listed; the 1001st makes it their hull and a HullWarning (steps.py's cap).
    [0, 6280] holds [pi/6, 5pi/6] + 2k pi for k = 0..999, and [0, 6284] one piece more"""
    c = M(Fraction(1, 2), 1)
    listed = sin_rev(c, M(0, 6280))
    assert len(listed.pieces) == ENUMERATION_CAP == 1000
    with pytest.warns(HullWarning):
        hulled = sin_rev(c, M(0, 6284))
    halves = sin_rev(c, M(0, 3000)) | sin_rev(c, M(3000, 6284))
    assert len(halves.pieces) == 1001 and hulled == halves.hull
    # per piece of x, as steps.py: [0, 7] is listed (2 pieces, a gap between), then [8, 6300] (1002
    # more) is hulled alone, the gap in [0, 7] kept
    with pytest.warns(HullWarning):
        split = sin_rev(c, M(0, 7) | M(8, 6300))
    first = sin_rev(c, M(0, 7))
    later = sin_rev(c, M(8, 3000)) | sin_rev(c, M(3000, 6300))
    assert len(first.pieces) == 2 and len(later.pieces) == 1002
    assert split == first | later.hull and len(split.pieces) == 3
    # the count runs across the pieces of x (M13e's review, 2026-09-27: the case above passes 1000 in
    # one piece, so a count per piece of x went unseen): 478 then 525 pieces, each under the cap
    with pytest.warns(HullWarning):
        both = sin_rev(c, M(0, 3000) | M(3001, 6300))
    one_, two = sin_rev(c, M(0, 3000)), sin_rev(c, M(3001, 6300))
    assert len(one_.pieces) + len(two.pieces) > ENUMERATION_CAP > len(two.pieces) > len(one_.pieces)
    assert both == one_ | two.hull and len(both.pieces) == len(one_.pieces) + 1


def test_trig_rev_hull_of_a_wide_x():
    """past `_BRANCH_LIMIT` branches the hull's ends come from walking inward from x's ends: the same as
    the exact results on windows of a period and more at each end; per piece of x, as steps.py does"""
    c = M(Fraction(1, 2), 1)
    for name in ('sin', 'cos', 'tan'):
        with pytest.warns(HullWarning):
            wide = TRIG[name](c, M(0, 10 ** 6))
        low, high = TRIG[name](c, M(0, 7)), TRIG[name](c, M(10 ** 6 - 7, 10 ** 6))
        assert wide == M.from_pieces([(low.inf, high.sup, low.inf_closed, high.sup_closed)])
        with pytest.warns(HullWarning):
            mixed = TRIG[name](c, M(0, 1) | M.parse('[10, inf)'))
        start = TRIG[name](c, M(10, 17))
        assert mixed == TRIG[name](c, M(0, 1)) | M.from_pieces([(start.inf, INF, start.inf_closed, False)])


def test_trig_rev_far_out_against_arb():
    """branches far from 0 (k near 3e5, where a double's ulp is still far below pi): each piece is the
    tightest open enclosure of k pi ± asin(1/2). near 1e20 the ulp (16384) is wider than a period, so the
    enclosures of all the solutions in a window merge into one piece, which must still hold them"""
    flint = pytest.importorskip('flint')
    far = sin_rev(M(Fraction(1, 2)), M(10 ** 20, 10 ** 20 + 7))
    assert far == M.from_pieces([(1e20, 10 ** 20 + 7, False, True)])  # sin(1e20) is not 1/2: open there
    start = 10 ** 6
    r = sin_rev(M(Fraction(1, 2)), M(start, start + 7))
    old = flint.ctx.prec
    flint.ctx.prec = 300
    try:
        pi = flint.arb.pi()
        a = pi / 6  # asin(1/2)
        k0 = int((flint.arb(start) / pi).floor().unique_fmpz())
        truths = [t for k in range(k0 - 1, k0 + 4) for t in [k * pi + (a if k % 2 == 0 else -a)]
                  if flint.arb(start) < t < flint.arb(start + 7)]
        assert len(r.pieces) == len(truths) >= 2
        for p, t in zip(r.pieces, sorted(truths, key=lambda t: float(t.mid()))):
            assert not p.inf_closed and not p.sup_closed
            assert p.sup == math.nextafter(p.inf, INF) and flint.arb(p.inf) < t < flint.arb(p.sup)
    finally:
        flint.ctx.prec = old


def test_trig_rev_class_coercion_and_warnings():
    assert type(sin_rev(O(0.5), M(0, 1))) is O and type(cos_rev(M(0), O(0.0, 2.0))) is O
    assert type(tan_rev(M(1), M(0, 1))) is M
    assert sin_rev(0, 0) == M(0) and cos_rev(1, 0) == M(0) and tan_rev(1, 0) == EMPTY
    for fn in TRIG.values():
        with pytest.raises(TypeError):
            fn('[0, 1]')
        with pytest.raises(TypeError):
            fn(M(0), None)
        for args in ((EMPTY,), (M(0), EMPTY), (O(), O(0.0, 1.0))):
            with pytest.warns(EmptySetPropagationWarning):
                assert fn(*args) == EMPTY
    # no solution warns nothing (the suite turns warnings into errors); neither does a whole-line answer
    assert sin_rev(M(2, 3)) == EMPTY and tan_rev(M(INF)) == EMPTY
    assert cos_rev(M(-1, 1)) == M.parse('(-inf, inf)')
    with pytest.warns(HullWarning):
        assert tan_rev(M.parse('(-inf, inf)')) == M.parse('(-inf, inf)')  # the poles: infinitely many pieces


def test_trig_rev_unary_forms_do_not_see_the_infinities():
    """the unary vectors run with x = [-inf, inf]; the trig functions have no value at ±inf, so each
    gives what 1788's entire as x gives: no row hides a difference there"""
    from tests.itf1788 import test_itf1788 as t
    unary = [v for v in t.VECTORS if v.op in ('sinRev', 'cosRev', 'tanRev')]
    assert len(unary) == 34
    for v in unary:
        c = t.to_ours(v.args[0])
        name = v.op[:3]
        assert _quiet(TRIG[name], c) == _quiet(TRIG[name], c, M.from_cuts(ENTIRE_1788)), v.text


def _true_hull_1788(name, c, x, flint):
    """arb's hull of `{t in x : f(t) in c}` for a *Bin vector with bounded operands, as (lo, hi) arbs:
    every branch meeting x, its inverse at c's ends, then intersected with x"""
    pi = flint.arb.pi()
    lo_c, hi_c = (max(c[0], -1), min(c[1], 1)) if name != 'tan' else c
    lo_x, hi_x = (_arb_of(v, flint) for v in x)
    found = []
    for k in range(math.floor(float(x[0]) / math.pi) - 2, math.floor(float(x[1]) / math.pi) + 3):
        if name == 'sin':
            g, rising = (lambda v, k=k: k * pi + (-1) ** k * v.asin()), k % 2 == 0
        elif name == 'cos':
            g, rising = ((lambda v, k=k: k * pi + v.acos()), False) if k % 2 == 0 else \
                ((lambda v, k=k: (k + 1) * pi - v.acos()), True)
        else:
            g, rising = (lambda v, k=k: k * pi + v.atan()), True
        a, b = g(_arb_of(lo_c, flint)), g(_arb_of(hi_c, flint))
        a, b = (a, b) if rising else (b, a)
        if b < lo_x or a > hi_x:
            continue
        assert (a > lo_x or a < lo_x) and (b > hi_x or b < hi_x), 'arb cannot decide'
        found.append((a if a > lo_x else lo_x, b if b < hi_x else hi_x))
    return min((f[0] for f in found), key=lambda z: float(z.mid())), max((f[1] for f in found), key=lambda z: float(z.mid()))


def test_trig_rev_is_tighter_than_the_vector():
    """the rows `_TRIG_REV_LOOSE_ROWS` (tests/itf1788, the proposed category "tighter than the vector"):
    arb puts each true end of the hull strictly inside our end's double and the next one inward, so ours
    is the tightest; 1788's hull holds ours, equal at one end and one or two doubles outside at the other"""
    flint = pytest.importorskip('flint')
    from tests.itf1788 import test_itf1788 as t
    rows = [v for v in t.VECTORS if t.key(v) in t._TRIG_REV_LOOSE_ROWS]
    assert len(rows) == 12 and {t.key(v) for v in rows} == set(t._TRIG_REV_LOOSE_ROWS)
    old = flint.ctx.prec
    flint.ctx.prec = 300
    try:
        for v in rows:
            name, (c, x) = v.op[:3], v.args
            ours, theirs = t.run(v)
            assert t.run_outward(v)[0] == ours
            if t.is_decorated(v):  # M13's merge: a decorated copy is (hull, decoration), trv on both sides
                assert ours[1] == theirs[1] == 'trv', v.text
                ours, theirs = ours[0], theirs[0]
            lo, hi = _true_hull_1788(name, (c.lo, c.hi), (x.lo, x.hi), flint)
            assert flint.arb(ours[0]) < lo < flint.arb(math.nextafter(ours[0], INF)), v.text
            assert flint.arb(math.nextafter(ours[1], -INF)) < hi < flint.arb(ours[1]), v.text
            assert theirs[0] <= ours[0] and ours[1] <= theirs[1] and (theirs[0] == ours[0]) != (theirs[1] == ours[1])
            gap = [e for e in (theirs[0], ours[0]) if theirs[0] != ours[0]] or [ours[1], theirs[1]]
            steps = 0
            while gap[0] < gap[1]:
                gap[0], steps = math.nextafter(gap[0], INF), steps + 1
            assert steps in (1, 2), (v.text, steps)
    finally:
        flint.ctx.prec = old


# THE DEFINING PROPERTY: exactly the t of x with f(t) in c, where x is bounded; the hull where not

@settings(max_examples=150, deadline=None)
@given(name=trig_names, c=exact_cut_tuples, x=exact_cut_tuples)
@example(name='sin', c=v1788(H('0X1.FFFFFFFFFFFFFP-1'), 1.0), x=v1788(1.57, 1.58))  # rev.itl:555, a row
@example(name='sin', c=v1788(0.0, 0.0), x=v1788(3.0, 3.5))  # :554
@example(name='sin', c=v1788(-H('0X1.72CECE675D1FDP-52'), 1.0), x=v1788(-0.1, 3.15))  # :563
@example(name='sin', c=v1788(H('0X1.1A62633145C06P-53'), H('0X1.1A62633145C07P-53')),
         x=v1788(-INF, 3.15))  # :569, a hull
@example(name='cos', c=v1788(-1.0, -1.0), x=v1788(3.14, 3.15))  # :633, a row
@example(name='cos', c=v1788(-1.0, -H('0X1.FFFFFFFFFFFFFP-1')), x=v1788(9.42, 9.45))  # :644
@example(name='cos', c=v1788(-H('0X1.AA22657537205P-2'), H('0X1.14A280FB5068CP-1')), x=v1788(0.0, 2.1))  # :647
@example(name='cos', c=v1788(H('0X1.87996529F9D92P-1'), 1.0), x=v1788(-1.0, 0.1))  # :646
@example(name='cos', c=v1788(-H('0X1.72CECE675D1FDP-52'), -H('0X1.72CECE675D1FCP-52')),
         x=v1788(-1.5, INF))  # :652, a hull
@example(name='tan', c=v1788(H('0X1.D02967C31CDB4P+53'), H('0X1.D02967C31CDB5P+53')),
         x=v1788(-1.5708, 1.5708))  # :711, a row
@example(name='tan', c=v1788(-INF, INF), x=v1788(-1.5708, 1.5708))  # :708, the poles inside x
@example(name='tan', c=v1788(-H('0X1.D02967C31CDB5P+53'), H('0X1.D02967C31CDB5P+53')),
         x=v1788(-1.5707965, 1.5707965))  # :718
@example(name='tan', c=v1788(-H('0X1.D02967C31p+53'), H('0X1.D02967C31p+53')), x=v1788(-INF, 1.5707965))  # :715
@example(name='tan', c=one(INF, INF), x=ALL)  # ±inf: no pole is a solution
@example(name='sin', c=one(-1, 1, True, False), x=one(0, 20))  # the gaps are single points, 1 at pi/2 + 2k pi
@example(name='tan', c=one(0, INF, True, False), x=one(Fraction(1.5707963267948968) - Fraction(1, 2 ** 60), 2))  # slack
@example(name='tan', c=one(-INF, 0, False, True), x=one(1, Fraction(1.5707963267948966) + Fraction(1, 2 ** 60)))
def test_trig_rev_exactly_the_points_with_f_in_c(name, c, x):
    """over the bounded pieces of x: soundness and the converse at every point where membership could
    change, and tightness at each rounded end (the next double inward is a true point). over an
    unbounded piece: soundness, and the part is the hull of the exact result on a window of more than a
    period at its finite end (or the whole piece): D12's hull. ±inf are never in it"""
    C, X = M.from_cuts(c), M.from_cuts(x)
    result = trev(name, C, X)
    assert result.issubset(X) and INF not in result and -INF not in result
    bounded = _bounded_part(X)
    for t in probes(c, x, result.cuts):
        truth = trig_value_in(name, t, c)
        if truth and t in X:
            assert t in result, (t, result)
        if t in result and truth is False and t in bounded:
            assert trig_in_slack(t, result), (t, result)
    for lo, lo_closed, hi, hi_closed in pieces(result.cuts):
        for end, closed, toward in ((lo, lo_closed, INF), (hi, hi_closed, -INF)):
            if not closed and isinstance(end, float) and math.isfinite(end) and end in _widened(bounded):
                inward = math.nextafter(end, toward)
                if lo < inward < hi and inward in bounded:
                    assert trig_value_in(name, inward, c) is not False, (end, result)
    if not trig_hulls(name, C, X):
        return
    for lo, lo_closed, hi, hi_closed in pieces(intersection(x, FINITE)):
        if lo != -INF and hi != INF:
            continue
        got = result & M.from_pieces([(lo, hi, lo_closed, hi_closed)])
        if lo == -INF and hi == INF:
            assert got == M.from_cuts(FINITE)
        elif lo == -INF:
            window = trev(name, C, M.from_pieces([(hi - 7, hi, True, hi_closed)]))
            assert got == M.from_pieces([(-INF, window.sup, False, window.sup_closed)])
        else:
            window = trev(name, C, M.from_pieces([(lo, lo + 7, lo_closed, True)]))
            assert got == M.from_pieces([(window.inf, INF, window.inf_closed, False)])


# SET LEVEL

def trig_forward(name, a: MultiInterval) -> MultiInterval:
    """the library's f on a set"""
    return _quiet(lambda: getattr(a, name)())


@settings(max_examples=100, deadline=None)
@given(name=trig_names, t=exact_cut_tuples, more=exact_cut_tuples, x=exact_cut_tuples)
@example(name='tan', t=one(1, 2), more=(), x=ALL)  # a pole inside T: tan(T) holds ±inf
@example(name='sin', t=one(-1, 20), more=(), x=one(0, 3))
def test_trig_rev_the_largest_set(name, t, more, x):
    """any T of reals whose image lies in C is inside rev(C, BOX), and T ∩ X inside rev(C, X). in the
    outward class, since f(T) has float ends (its enclosures), as for cosh above"""
    T = O.from_cuts(t) & O.from_cuts(BOX.cuts)
    C = trig_forward(name, T) | O.from_cuts(more)
    assert T.issubset(trev(name, C, O.from_cuts(BOX.cuts)))
    X = O.from_cuts(x)
    assert (T & X).issubset(trev(name, C, X))


@settings(max_examples=100, deadline=None)
@given(name=trig_names, c=exact_cut_tuples, more=exact_cut_tuples, x=exact_cut_tuples, x_more=exact_cut_tuples)
def test_trig_rev_isotone(name, c, more, x, x_more):
    """in c and in x, the hulls included (a larger set has a larger hull)"""
    C, X = M.from_cuts(c), M.from_cuts(x)
    bigger_c, bigger_x = C | M.from_cuts(more), X | M.from_cuts(x_more)
    assert trev(name, C, X).issubset(trev(name, bigger_c, X))
    assert trev(name, C, X).issubset(trev(name, C, bigger_x))


# rationals just past and just before the pole pi/2, inside the slack of branch 0's end enclosure (the
# double above pi/2) and of branch 1's start enclosure (the double below pi/2)
_PAST_THE_POLE = Fraction(1.5707963267948968) - Fraction(1, 2 ** 60)
_BEFORE_THE_POLE = Fraction(1.5707963267948966) + Fraction(1, 2 ** 60)


@settings(max_examples=100, deadline=None)
@given(name=trig_names, c=exact_cut_tuples, d=exact_cut_tuples, x=exact_cut_tuples)
@example(name='tan', c=one(0, INF, True, False), d=(), x=one(_PAST_THE_POLE, 2))  # x starts in a neighbour's slack
@example(name='tan', c=one(-INF, 0, False, True), d=(), x=one(1, _BEFORE_THE_POLE))  # x ends in one
def test_trig_rev_union_and_x(name, c, d, x):
    """over a bounded x, a preimage distributes over a union of c, and x only intersects: the result is
    the union of every branch's enclosure, then ∩ x, whichever branches x starts in (hence one branch
    more on each side: x may start inside the slack of the branch before it)"""
    C, D = M.from_cuts(c), M.from_cuts(d)
    assert trev(name, C | D, BOX) == trev(name, C, BOX) | trev(name, D, BOX)
    X = M.from_cuts(x) & BOX
    assert trev(name, C, X) == trev(name, C, BOX) & X


@settings(deadline=None)
@given(name=trig_names, c=cut_tuples(), x=cut_tuples())
@example(name='sin', c=(Cut(0, Side.BELOW), Cut(0.0, Side.ABOVE)), x=one(-INF, -2, False, False))  # a mixed point
def test_trig_rev_symmetry(name, c, x):
    """sin and tan are odd, cos is even: `rev(-c, -x) = -rev(c, x)` and `cos_rev(c, -x) = -cos_rev(c, x)`,
    hulls and float operands included (a branch's mirror is a branch, and rounding is symmetric). the
    mirror is taken cut by cut (`reverse.negate`), which keeps each end's type; the class's `-` does too
    since fuzz-symmetry (2026-09-29), but before it rebuilt a point whose ends differ in type (`[0, 0.0]`)
    with one value, so an end changed from float to exact and rounded differently (`test_symmetry`)"""
    C, X = M.from_cuts(c), M.from_cuts(x)
    neg = lambda a: M.from_cuts(negate(a.cuts))  # noqa: E731
    r = _quiet(TRIG[name], C, X)
    if name == 'cos':
        assert _quiet(TRIG[name], C, neg(X)) == neg(r)
    else:
        assert _quiet(TRIG[name], neg(C), neg(X)) == neg(r)


# FLOAT OPERANDS

@settings(max_examples=100, deadline=None)
@given(name=trig_names, c=float_cut_tuples, x=float_cut_tuples)
@example(name='sin', c=one(H('0X1.FFFFFFFFFFFFFP-1'), 1.0), x=one(1.57, 1.58))  # rev.itl:555
@example(name='cos', c=one(-1.0, -1.0), x=one(3.14, 3.15))  # :633
@example(name='tan', c=one(H('0X1.D02967C31CDB4P+53'), H('0X1.D02967C31CDB5P+53')), x=one(-1.5708, 1.5708))  # :711
@example(name='sin', c=one(0.5, 1.0), x=one(0.0, 20.0))  # nearest: float ends, closed
@example(name='sin', c=one(-5.614185657941294e-24, 0.0, False, False),
         x=one(-INF, -5.614185657941294e-24, False, False))  # x's end in the half ulp: kept as a point (D26)
def test_trig_rev_float_operands(name, c, x):
    """over a bounded x: outward holds the exact result of the same doubles, adds no double strictly inside
    what it adds, and closes only exact points. to nearest, x taken as the whole box (an end of x can
    fall in the half ulp a rounded end moved, as mul_rev's test says: asin(-5.6e-24) rounds onto the
    double -5.6e-24 itself, and x = (-inf, -5.6e-24) holds only the exact sliver between them): float
    ends within one double of the exact result, inside the outward closure, not empty when it is not;
    and x meets it after the rounding, the sliver kept as the point -5.6e-24 (D26: `meets_x_as_d26`)"""
    box = lambda cls: cls.from_cuts(BOX.cuts)  # noqa: E731
    exact = trev(name, M.from_cuts(exact_cuts(c)), M.from_cuts(exact_cuts(x)) & BOX)
    outward = trev(name, O.from_cuts(c), O.from_cuts(x) & box(O))
    assert exact.issubset(outward)
    for lo, _, hi, _ in pieces(outward.difference(exact).cuts):
        assert hi <= _first_double_above(lo), (lo, hi)
    for lo, lo_closed, hi, hi_closed in pieces(outward.cuts):
        for end, closed in ((lo, lo_closed), (hi, hi_closed)):
            if closed:
                assert end in exact, (end, outward)
    exact_all = trev(name, M.from_cuts(exact_cuts(c)), BOX)
    nearest_all = trev(name, M.from_cuts(c), BOX)
    outward_all = trev(name, O.from_cuts(c), box(O))
    for r in (outward, nearest_all):
        assert all(isinstance(cut.value, float) or cut.value == float(cut.value) for cut in r.cuts), r
    assert exact_all.issubset(_widened(nearest_all))
    assert nearest_all.issubset(M.from_pieces((lo, hi) for lo, _, hi, _ in pieces(outward_all.cuts)))
    if exact_all:
        assert nearest_all
    X = M.from_cuts(x) & BOX
    meets_x_as_d26(trev(name, M.from_cuts(c), X), nearest_all, X, exact, outward)


@settings(max_examples=100, deadline=None)
@given(name=trig_names, c=cut_tuples(), x=cut_tuples(), seed=st.integers(0, 2 ** 32 - 1))
@example(name='cos', c=one(-1.0, -1.0), x=one(3.14, 3.15), seed=0)  # rev.itl:633
def test_trig_rev_sound_at_sampled_points(name, c, x, seed):
    """M14's soundness: a sampled t of x with f(t) in c is in the exact result and in the outward one,
    hulls included (float operands read as the rationals they are); a sampled t of the exact result's
    bounded part has f(t) in c, or lies in a rounded end's slack"""
    rng = random.Random(seed)
    exact_c, exact_x = exact_cuts(c), exact_cuts(x)
    X = M.from_cuts(exact_x)
    exact = trev(name, M.from_cuts(exact_c), X)
    outward = trev(name, O.from_cuts(c), O.from_cuts(x))
    for t in sample(exact_x, 20, rng):
        t = _exact(t)
        if trig_value_in(name, t, exact_c):
            assert t in exact and t in outward, t
    bounded = _bounded_part(X)
    for t in sample(exact.cuts, 20, rng):
        t = _exact(t)
        if t in bounded:
            assert trig_value_in(name, t, exact_c) is not False or trig_in_slack(t, exact), t


# FAR ENDS (trig-rev-far): the hull's walk inward from a far end of x
#
# to nearest, every branch within half an ulp of x's end rounds onto that one double, which an open end
# does not hold, and past the doubles onto ±inf, which a finite end never holds: the step-by-step walk
# took ulp/(2 pi) branches (20870 at 1e21, 2**942 at 1e300) or 10**400/pi. `reverse._leap` leaps over
# them; these pin that it lands where the step-by-step walk does (`_stepwise_hull`, the walk before the
# fix) and that it stays fast where that walk could not finish

from hypothesis import assume  # noqa: E402

from intervals import reverse  # noqa: E402

_PERIODIC = {'sin': reverse._SIN, 'cos': reverse._COS, 'tan': reverse._TAN}


def _walk_from(fn, c, part, k, step, outward):
    """the step-by-step walk of `_periodic_hull` before trig-rev-far: the first branch from k on that
    meets the part, as that intersection"""
    while True:
        found = intersection(reverse.branch_preimage(c, fn.branch(k), outward), part)
        if found:
            return found
        k += step


def _stepwise_hull(fn, c, part, first, last, outward):
    """`_periodic_hull` as it was before trig-rev-far: the reference the leap must equal"""
    lo, lo_closed = (-INF, False) if first is None else next(pieces(_walk_from(fn, c, part, first, 1, outward)))[:2]
    hi, hi_closed = (INF, False) if last is None else next(pieces(_walk_from(fn, c, part, last, -1, outward)[-2:]))[2:]
    return one(lo, hi, lo_closed, hi_closed)


def _typed(cuts):
    """the cuts with each value's type: 1e300 and 10 ** 300 are different answers"""
    return [(type(cut.value), cut.value, cut.side) for cut in cuts]


def _hull_ends(fn, part):
    """first and last as `_periodic_preimage` computes them"""
    (lo, _, hi, _), = pieces(part)
    first = None if lo == -INF else elementary.floor_over_pi(lo, fn.offset)[0] - 1
    last = None if hi == INF else elementary.floor_over_pi(hi, fn.offset)[0] + 1
    return first, last


@st.composite
def _far_parts(draw):
    """a piece of x from 2**50 to 2**64 away from 0, at least 7000 wide: up to ~650 branches round onto
    one double at its ends, few enough for the step-by-step walk; float or exact ends, mostly open"""
    def end(sign):
        value = sign * math.ldexp(draw(st.floats(1.0, 2.0, exclude_max=True)), draw(st.integers(50, 63)))
        if draw(st.integers(0, 3)) == 0:
            value = Fraction(value) + Fraction(draw(st.integers(-5, 5)), 3)  # exact, off the doubles
        return value
    a, width = end(draw(st.sampled_from([-1, 1]))), draw(st.floats(7000.0, 2.0 ** 64))
    if isinstance(a, Fraction):
        width = Fraction(width)
    lo, hi = (a, a + width) if draw(st.booleans()) else (a - width, a)
    closed = st.sampled_from([False, False, True])  # mostly open: a double the rounding lands on is not in x
    return one(lo, hi, draw(closed), draw(closed))


@st.composite
def _trig_cs(draw, name):
    """c ∩ f's image, not empty: one or two pieces, each end a float or an exact rational, either flag"""
    bound = 50 if name == 'tan' else 1

    def value():
        v = draw(st.floats(-bound, bound))
        return Fraction(v).limit_denominator(1000) if draw(st.integers(0, 3)) == 0 else v  # mostly floats: to nearest
    parts = []
    for _ in range(draw(st.integers(1, 2))):
        a, b = sorted((value(), value()))
        parts.append(one(a, b, draw(st.booleans()), draw(st.booleans())) if a < b else one(a, a))
    c = intersection(union(*parts), _TRIG_IMAGE[name])
    assume(c)
    return c


@settings(max_examples=100, deadline=None)
@given(data=st.data(), name=trig_names, part=_far_parts(), outward=st.sampled_from([False, False, False, True]))
@example(data=None, name='tan', part=one(-1e19, -1e19 + 10 ** 5, False, True), outward=False)
@example(data=None, name='sin', part=one(1e19 - 10 ** 5, 1e19, True, False), outward=False)
@example(data=None, name='cos', part=one(-1e19, -1e19 + 10 ** 5, False, True), outward=False)
def test_trig_rev_hull_leaps_to_where_the_walk_stops(data, name, part, outward):
    """`_periodic_hull` (the leap) equals the step-by-step walk, value, type and flag of each end, over
    far pieces of x where up to hundreds of branches round onto one double, to nearest and outward"""
    fn = _PERIODIC[name]
    if data is None:  # the examples: float ends, to nearest, open at the far end: a window of ~330 branches
        c = intersection(M.parse(_FAR_C[name]).cuts, fn.image)
    else:
        c = data.draw(_trig_cs(name))
    first, last = _hull_ends(fn, part)
    got = reverse._periodic_hull(fn, c, part, first, last, outward)
    assert _typed(got) == _typed(_stepwise_hull(fn, c, part, first, last, outward))


_FAR_C = {'tan': '[-40.0, 0.1]', 'sin': '[-0.5, 0.1]', 'cos': '[-0.5, 0.1]'}
_NEAR = 7.582732456406029
_OVERFLOW = Fraction(2 ** 1024 - 2 ** 970)  # past it, to nearest, a value rounds to ±inf
_BRANCH_BUDGET = 10000  # branch preimages per call; the leap needed 4649 at 10**400 (2026-10-05)


class _TooManyBranches(Exception):
    pass


def _bounded_branches(monkeypatch, limit):
    """make `reverse.branch_preimage` raise past `limit` calls, so the old walk fails fast, not hangs"""
    calls = [0]
    real = reverse.branch_preimage

    def counted(*args):
        calls[0] += 1
        if calls[0] > limit:
            raise _TooManyBranches(calls[0])
        return real(*args)
    monkeypatch.setattr(reverse, 'branch_preimage', counted)
    return calls


def _class_ends(name, c, flint, pi):
    """per parity class `(period, residue)`, the constants C of its branches' two exact ends `k pi + C`
    (low, high) as arbs, for c one piece [v, w] inside the image (the branches: `reverse._SIN`, ...)"""
    (v, _, w, _), = pieces(c)
    v, w = _arb_of(v, flint), _arb_of(w, flint)
    if name == 'tan':
        return {(1, 0): (v.atan(), w.atan())}
    if name == 'sin':  # k pi + asin, rising, for an even k; k pi - asin, falling, for an odd one
        return {(2, 0): (v.asin(), w.asin()), (2, 1): (-w.asin(), -v.asin())}
    # k pi + acos, falling, for an even k; (k + 1) pi - acos, rising, for an odd one
    return {(2, 0): (w.acos(), v.acos()), (2, 1): (pi - v.acos(), pi - w.acos())}


def _transition(name, c, threshold, low_walk, flint):
    """from arb, the branch where a walk from a far end stops missing: walking up (low_walk), the least k
    whose branch's high end exceeds `threshold`; walking down, the greatest whose low end is below it"""
    bits = max(abs(threshold.numerator).bit_length() - threshold.denominator.bit_length(), 0)
    old = flint.ctx.prec
    flint.ctx.prec = bits + 200
    try:
        pi = flint.arb.pi()
        t = _arb_of(threshold, flint)
        ks = []
        for (period, residue), (low, high) in _class_ends(name, c, flint, pi).items():
            if low_walk:  # the least k = residue (mod period) with k pi + high > t
                k = int(((t - high) / pi).floor().unique_fmpz()) + 1
                k += (residue - k) % period
            else:  # the greatest k = residue (mod period) with k pi + low < t
                k = int(((t - low) / pi).ceil().unique_fmpz()) - 1
                k -= (k - residue) % period
            ks.append(k)
        return min(ks) if low_walk else max(ks)
    finally:
        flint.ctx.prec = old


@pytest.mark.parametrize('name', ['tan', 'sin', 'cos'])
@pytest.mark.parametrize('far', [1e22, 1e300, 10 ** 400], ids=['1e22', '1e300', '10**400'])
@pytest.mark.parametrize('low_walk', [True, False], ids=['low', 'high'])
def test_trig_rev_far_end_of_x(monkeypatch, name, far, low_walk):
    """a piece of x from near 0 out to `far`, open there, and a float c (to nearest): the step-by-step
    walk passed ulp/(2 pi) branches (2**942 at 1e300), or 10**400/pi past the doubles; now a few thousand
    at most, and the hull's far end is where that walk stops: at the branch where, by arb, the rounding
    first reaches into x (checked against the library's own rounding there and at the two before), then
    the step-by-step walk from there. the near end is the exact result's"""
    flint = pytest.importorskip('flint')
    fn = _PERIODIC[name]
    c = intersection(M.parse(_FAR_C[name]).cuts, fn.image)
    x = M(-far, -_NEAR, start_closed=False) if low_walk else M(_NEAR, far, end_closed=False)
    _bounded_branches(monkeypatch, _BRANCH_BUDGET)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            got = TRIG[name](M.parse(_FAR_C[name]), x)
        except _TooManyBranches:
            pytest.fail(f'more than {_BRANCH_BUDGET} branch preimages: the walk does not leap')
    monkeypatch.undo()
    assert [w.category for w in caught] == [HullWarning]
    part = intersection(x.cuts, FINITE)
    (part_lo, part_lo_closed, part_hi, part_hi_closed), = pieces(part)
    end = part_lo if low_walk else part_hi
    if isinstance(end, float):  # to nearest, a value rounds into x past the midpoint to the next double in
        threshold = (Fraction(end) + Fraction(math.nextafter(end, INF if low_walk else -INF))) / 2
    else:  # past the doubles: a value rounds into x once it rounds to a finite double
        threshold = -_OVERFLOW if low_walk else _OVERFLOW
    k = _transition(name, c, threshold, low_walk, flint)
    step = 1 if low_walk else -1
    ray = one(part_lo, INF, part_lo_closed, True) if low_walk else one(-INF, part_hi, True, part_hi_closed)

    def meets_ray(j):
        return bool(intersection(reverse.branch_preimage(c, fn.branch(j), False), ray))
    assert meets_ray(k) and not meets_ray(k - step) and not meets_ray(k - 2 * step)
    found = _walk_from(fn, c, part, k - 2 * step, step, False)
    near = TRIG[name](M.parse(_FAR_C[name]), M(-30, -_NEAR) if low_walk else M(_NEAR, 30))
    if low_walk:
        lo, lo_closed, _, _ = next(pieces(found))
        want = M.from_pieces([(lo, near.sup, lo_closed, near.sup_closed)])
    else:
        _, _, hi, hi_closed = list(pieces(found))[-1]
        want = M.from_pieces([(near.inf, hi, near.inf_closed, hi_closed)])
    assert _typed(got.cuts) == _typed(want.cuts), (got, want)


# THE ROUNDING OF THE ENDS

def test_rounded_inverse_trig_exact_case():
    assert elementary.rounded_inverse_trig('asin', 0, -1, 0, DOWN) == 0.0
    assert elementary.rounded_inverse_trig('acos', 1, 1, 0, UP) == 0.0
    assert elementary.rounded_inverse_trig('atan', 0, 1, 0, NEAREST) == 0.0
    # acos(-1) = pi: k pi - pi is 0 at k = 1 (the fuzz profile found this one hanging in ziv's loop)
    assert elementary.rounded_inverse_trig('acos', -1, -1, 1, DOWN) == 0.0
    assert elementary.rounded_inverse_trig('acos', -1, 1, -1, UP) == 0.0


@settings(max_examples=200, deadline=None)
@given(name=st.sampled_from(['asin', 'acos', 'atan']),
       v=st.one_of(st.fractions(-1, 1, max_denominator=10 ** 6), st.sampled_from([-1, 0, 1]),
                   st.floats(-1, 1).map(Fraction)),
       big=st.one_of(st.fractions(-10 ** 6, 10 ** 6, max_denominator=100), st.sampled_from([INF, -INF])),
       sign=st.sampled_from([1, -1]),
       k=st.one_of(st.integers(-10, 10), st.integers(-2 ** 70, 2 ** 70), st.integers(-2 ** 400, 2 ** 400)),
       direction=st.sampled_from([DOWN, NEAREST, UP]))
@example(name='asin', v=1, big=0, sign=1, k=0, direction=UP)
@example(name='acos', v=-1, big=0, sign=-1, k=3, direction=NEAREST)
@example(name='atan', v=0, big=INF, sign=-1, k=-7, direction=DOWN)
@example(name='acos', v=1, big=0, sign=-1, k=2 ** 400, direction=UP)  # acos(1) = 0: the value is k pi
@example(name='acos', v=-1, big=0, sign=-1, k=2, direction=DOWN)  # acos(-1) = pi: the value is pi
@example(name='asin', v=Fraction(7.2701979378291835e-245), big=0, sign=1, k=0, direction=UP)  # just above v
def test_rounded_inverse_trig_against_arb(name, v, big, sign, k, direction):
    """`k pi + sign * f(v)` rounded against arb (atan over the reals and ±inf, asin and acos over
    [-1, 1]): DOWN is the largest double below the value, UP the smallest above, NEAREST within half an
    ulp (the value is irrational off the exact case, so never a double nor a tie)"""
    flint = pytest.importorskip('flint')
    if name == 'atan':
        v = big
    if (k == 0 and v == (1 if name == 'acos' else 0)) or (name == 'acos' and v == -1 and k == -sign):
        return  # the exact cases, above
    got = elementary.rounded_inverse_trig(name, v, sign, k, direction)
    assert math.isfinite(got)
    below, above = math.nextafter(got, -INF), math.nextafter(got, INF)
    lo, hi = {DOWN: (got, above), UP: (below, got), NEAREST: (None, None)}[direction]
    old = flint.ctx.prec
    try:
        # arb's precision is relative: a tiny v needs more of it, asin(v) - v being about v**3 / 6
        for prec in (abs(k).bit_length() + 200 * 4 ** i for i in range(6)):
            flint.ctx.prec = prec
            fv = flint.arb.pi() / 2 * (1 if v > 0 else -1) if v in (INF, -INF) else getattr(_arb_of(v, flint), name)()
            true = k * flint.arb.pi() + sign * fv
            if direction == NEAREST:  # within half an ulp: between the midpoints with the neighbours
                lo_arb, hi_arb = (flint.arb(below) + flint.arb(got)) / 2, (flint.arb(got) + flint.arb(above)) / 2
            else:
                lo_arb, hi_arb = flint.arb(lo), flint.arb(hi)
            if lo_arb < true < hi_arb:
                return
            assert not (true <= lo_arb or true >= hi_arb), (got, true)  # decided, and wrong
        raise AssertionError(f'arb never decided: {got!r}')
    finally:
        flint.ctx.prec = old
