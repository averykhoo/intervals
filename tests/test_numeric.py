"""
the numeric functions (intervals.numeric): mid, rad, wid, mag, mig, mid_rad (M13b, D9)

* exact operands: each value against its definition, `mag` and `mig` through `abs(A)` (ops.absolute,
  an independent path to the supremum and infimum of the absolute values)
* float operands: each value is the exact one rounded once as 1788 specifies, checked from the
  definition of rounding on the neighbouring doubles (`tests.test_reductions.is_rounded`), and `rad`
  as the smallest double radius around `mid`
* soundness at sampled points: `mig <= abs(x) <= mag`, `x` in `[mid - rad, mid + rad]`,
  `abs(x - y) <= wid`; isotonicity; hull and set; `OutwardMultiInterval` gives the same numbers
* the itf1788 vectors of the six ops run in tests/itf1788 (both passes); some are `@example`s here
"""
import math
import random
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import union
from intervals.rounding import exact_cuts
from intervals.rounding import has_finite_float
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.test_reductions import is_rounded

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
MAX = 1.7976931348623157e308
TINY = 5e-324
NAMES = ('mid', 'rad', 'wid', 'mag', 'mig')


def one(lo, hi, lo_closed=True, hi_closed=True):
    return normalize([piece(lo, hi, lo_closed, hi_closed)])


def h(text: str) -> float:
    return float.fromhex(text)


def values(cuts, cls=M):
    a = cls.from_cuts(cuts)
    return {name: getattr(a, name)() for name in NAMES}


# EXAMPLES

@pytest.mark.parametrize('text, mid, rad, wid, mag, mig', [
    ('[0, 1] | [9, 10]', 5, 5, 10, 10, 0),  # D9: the hull's midpoint, outside the set
    ('[-3, -2] | [2, 3]', 0, 3, 6, 3, 2),  # D9: mig of the set; the hull would give 0
    ('(-3, -2) | (2, 3)', 0, 3, 6, 3, 2),  # infima and suprema: open ends count the same
    ('[1, 2]', Fraction(3, 2), Fraction(1, 2), 1, 2, 1),
    ('[-1/3, 1/6]', Fraction(-1, 12), Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), 0),
    ('[5]', 5, 0, 0, 5, 5),
    ('[inf]', INF, 0, 0, INF, INF),  # a point: itself, radius and width 0
    ('[-inf] | [inf]', 0, INF, INF, INF, INF),
    ('(-inf, inf)', 0, INF, INF, INF, 0),
    ('[-inf, inf]', 0, INF, INF, INF, 0),
    ('[1, inf)', MAX, INF, INF, INF, 1),  # 1788: ±max float for a half-bounded hull
    ('(-inf, -2] | [3, 4]', -MAX, INF, INF, INF, 2),
    ('[0.0, 2.0]', 1.0, 1.0, 2.0, 2.0, 0.0),
    ('[0.1, 0.2]', 0.15000000000000002, 0.05000000000000002, 0.1, 0.2, 0.1),
    ('[-0.5, 2]', 0.75, 1.25, 2.5, 2.0, 0.0),  # one float end makes every value a float
    ('[1/3, 0.5]', 0.4166666666666667, 0.08333333333333336, 0.16666666666666669, 0.5, 0.3333333333333333),
])
def test_examples(text, mid, rad, wid, mag, mig):
    got = values(M.parse(text).cuts)
    assert got == dict(mid=mid, rad=rad, wid=wid, mag=mag, mig=mig), got
    for name, want in dict(mid=mid, rad=rad, wid=wid, mag=mag, mig=mig).items():
        assert type(got[name]) is type(want), (name, got[name])
    assert M.parse(text).mid_rad() == (mid, rad)


def test_exact_values_are_normalized():
    assert type(M(0, 2).mid()) is int and type(M(1, 2).mid()) is Fraction
    assert type(M(Fraction(1, 2), Fraction(5, 2)).wid()) is int


def test_empty_raises():
    for name, what in [('mid', 'midpoint'), ('rad', 'radius'), ('wid', 'width'), ('mag', 'magnitude'),
                       ('mig', 'mignitude'), ('mid_rad', 'midpoint or radius')]:
        with pytest.raises(ValueError, match=f'^the empty set has no {what}$'):
            getattr(M(), name)()


def test_rad_is_from_the_rounded_midpoint():
    """itf1788 `rad [0X1P+0,0X1.0000000000003P+0] = 0X1P-51`: half the width is 3 * 2 ** -53, a
    double, but the midpoint rounds (a tie, to even) one ulp up, so the radius is 2 ** -51"""
    a = M(1.0, h('0x1.0000000000003p0'))
    assert a.mid() == h('0x1.0000000000002p0') and a.rad() == 2.0 ** -51
    assert M(1, Fraction(h('0x1.0000000000003p0'))).rad() == 3 * Fraction(2) ** -53  # exact


def test_beyond_the_float_range():
    """an exact end past max float in a float operand: the rounded midpoint stays in the hull"""
    assert M(1.0, 3 * 2 ** 1024).mid() == MAX and M(-3 * 2 ** 1024, -1.0).mid() == -MAX
    assert M(1.0, 3 * 2 ** 1024).rad() == INF and M(1.0, 3 * 2 ** 1024).wid() == INF
    assert M(0.5, 2 ** 1030).mag() == INF and M(2 ** 1030, INF).mig() == 2 ** 1030
    assert M(2 ** 1030, INF).mid() == 2 ** 1030  # an exact half-bounded hull starting past max float


def test_outward_class_gives_the_same_numbers():
    for text in ('[0.1, 0.2]', '[1, inf)', '[-3, -2] | [2, 3]', '[1/3, 0.5]'):
        a, b = M.parse(text), O.parse(text)
        assert values(a.cuts) == values(b.cuts, O) and a.mid_rad() == b.mid_rad()


# THE ORACLE

def _exact(v):
    return v if v in (INF, -INF) else Fraction(v)


def _covers(m, r, lo, hi) -> bool:
    """`[m - r, m + r]` holds `[lo, hi]`, exactly (±inf as points)"""
    if r == INF:
        return True
    if m in (INF, -INF):
        return lo == hi == m
    m = Fraction(m)
    return m - Fraction(r) <= lo and hi <= m + Fraction(r)


def _sound(cuts, v, rng) -> None:
    """every sampled point is where the numbers say it can be"""
    points = sample(cuts, 12, rng)
    for x in points:
        assert v['mig'] <= abs(x) <= v['mag'], (x, v)
        assert _covers(v['mid'], v['rad'], _exact(x), _exact(x)), (x, v)
    if v['wid'] != INF:
        for x, y in zip(points, reversed(points)):
            assert x == y or abs(Fraction(x) - Fraction(y)) <= v['wid'], (x, y, v)  # [inf]: wid 0


def _abs_ends(cuts):
    """the infimum and supremum of the absolute values, through abs(A)"""
    a = abs(M.from_cuts(exact_cuts(cuts)))
    return a.inf, a.sup


# EXACT OPERANDS

@settings(max_examples=300, deadline=None)
@given(a=exact_cut_tuples, rng=st.randoms(use_true_random=False))
@example(a=one(-4, 2), rng=random.Random(0))  # mig [-4.0,2.0] = 0.0; mag 4
@example(a=one(-INF, -2), rng=random.Random(0))  # mig [-infinity,-2.0] = 2.0
@example(a=one(0, INF, True, False), rng=random.Random(0))  # mid [0.0,infinity] = max float
@example(a=union(one(-3, -2), one(2, 3)), rng=random.Random(0))  # D9's mig
@example(a=one(-INF, -INF), rng=random.Random(0))  # a point at -inf: itself, width 0
def test_exact_operands_give_exact_values(a, rng):
    if not a:
        return
    v = values(a)
    lo, hi = a[0].value, a[-1].value
    for name in NAMES:
        assert not isinstance(v[name], float) or v[name] in (INF, -INF) or v[name] == MAX \
            or v[name] == -MAX, (name, v)
    if lo == hi:
        assert (v['mid'], v['rad'], v['wid']) == (lo, 0, 0)
    elif lo == -INF and hi == INF:
        assert (v['mid'], v['rad'], v['wid']) == (0, INF, INF)
    elif INF in (-lo, hi):
        assert v['mid'] == (-MAX if lo == -INF else MAX)  # the strategy's ends are within max float
        assert v['rad'] == v['wid'] == INF
    else:
        assert v['mid'] == Fraction(lo + hi, 2) and v['wid'] == hi - lo and v['rad'] == Fraction(hi - lo, 2)
    assert (v['mig'], v['mag']) == _abs_ends(a)
    assert M.from_cuts(a).mid_rad() == (v['mid'], v['rad'])
    _sound(a, v, rng)


# FLOAT OPERANDS

wide_floats = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(-1e-305, 1e-305, allow_nan=False),  # subnormals and their neighbours
    st.sampled_from([MAX, -MAX, MAX / 2, TINY, -TINY, 1.0, 1.0000000000000002, 2.0, 0.0]),
)
float_cut_tuples = cut_tuples(values=st.one_of(
    wide_floats, wide_floats, st.sampled_from([-INF, INF, 0, 1, Fraction(1, 3)])),
    max_pieces=3)


@settings(max_examples=400, deadline=None)
@given(a=float_cut_tuples, rng=st.randoms(use_true_random=False))
@example(a=one(1.0, h('0x1.0000000000003p0')), rng=random.Random(0))  # rad = 0X1P-51
@example(a=one(-TINY * 2, TINY), rng=random.Random(0))  # mid = 0.0 (a tie, to even), rad = 2 ulp
@example(a=one(h('0x1.FFFFFFFFFFFFFp1022'), MAX), rng=random.Random(0))  # midRad = 0X1.7FFFFFFFFFFFFP+1023 0x1.0p+1022
@example(a=one(-h('0x1fffffffffffffp-53'), 2.0), rng=random.Random(0))  # mpfi: mid = 0.5, a tie
@example(a=one(-MAX, MAX), rng=random.Random(0))  # wid overflows to inf, rad is max float
@example(a=normalize([piece(0, 0.0)]), rng=random.Random(0))  # a point with an int and a float end: 0.0
def test_float_operands_round_as_1788(a, rng):
    if not a:
        return
    v = values(a)
    if not has_finite_float(a):
        return  # exact: the test above
    assert all(isinstance(v[name], float) for name in NAMES), v
    lo, hi = _exact(a[0].value), _exact(a[-1].value)
    if lo == hi:
        assert (v['mid'], v['rad'], v['wid']) == (lo, 0, 0)
    elif lo == -INF and hi == INF:
        assert (v['mid'], v['rad'], v['wid']) == (0, INF, INF)
    elif INF in (-lo, hi):
        assert v['mid'] == (-MAX if lo == -INF else MAX) and v['rad'] == v['wid'] == INF
    else:
        exact_mid = (lo + hi) / 2
        assert is_rounded(v['mid'], exact_mid, 'nearest') or (
            abs(exact_mid) > MAX and v['mid'] == math.copysign(MAX, exact_mid)), v
        assert lo <= v['mid'] <= hi
        # the smallest double radius around the rounded midpoint
        assert _covers(v['mid'], v['rad'], lo, hi), v
        assert v['rad'] == 0 or not _covers(v['mid'], math.nextafter(v['rad'], 0), lo, hi), v
        assert is_rounded(v['wid'], hi - lo, 'up') if hi - lo <= MAX else v['wid'] == INF
    low, high = _abs_ends(a)
    assert v['mag'] == INF if high == INF else is_rounded(v['mag'], high, 'up')
    assert v['mig'] == INF if low == INF else is_rounded(v['mig'], low, 'down')
    assert M.from_cuts(a).mid_rad() == (v['mid'], v['rad'])
    assert values(a, O) == v and O.from_cuts(a).mid_rad() == (v['mid'], v['rad'])
    _sound(a, v, rng)


# ISOTONICITY, HULL AND SET

@settings(max_examples=200, deadline=None)
@given(a=exact_cut_tuples, c=exact_cut_tuples)
def test_isotone_on_exact_operands(a, c):
    """A ⊆ B: the hull only grows and the set only gains points nearer zero or farther out"""
    b = union(a, c)
    if not a:
        return
    va, vb = values(a), values(b)
    assert va['wid'] <= vb['wid'] and va['rad'] <= vb['rad']
    assert va['mag'] <= vb['mag'] and va['mig'] >= vb['mig']


@settings(max_examples=200, deadline=None)
@given(a=cut_tuples(values=wide_floats, max_pieces=3), c=cut_tuples(values=wide_floats, max_pieces=3))
def test_isotone_on_float_operands(a, c):
    """rounding is monotone, so wid, mag and mig stay isotone on floats"""
    b = union(a, c)
    if not a:
        return
    va, vb = values(a), values(b)
    assert va['wid'] <= vb['wid'] and va['mag'] <= vb['mag'] and va['mig'] >= vb['mig']


@settings(max_examples=200, deadline=None)
@given(a=float_cut_tuples)
@example(a=union(one(-3.0, -2.0), one(2.0, 3.0)))  # D9's mig, in floats
@example(a=union(one(Fraction(1, 3), 1.0, False, False), one(1.0, INF, False, False)))  # found 2026-09-26
def test_hull_and_set(a):
    """mid, rad, wid and mag are the hull's; mig is the set's, never below the hull's"""
    if not a:
        return
    s = M.from_cuts(a)
    hull = s.hull
    if has_finite_float(hull.cuts) != has_finite_float(a):
        return  # the hull dropped the only finite float ends: exact numbers against rounded ones
    assert all(getattr(s, n)() == getattr(hull, n)() for n in ('mid', 'rad', 'wid', 'mag'))
    assert s.mig() >= hull.mig()
    if s.is_contiguous:
        assert s.mig() == hull.mig()
