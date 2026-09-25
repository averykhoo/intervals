"""
the elementary functions over sets: intervals.functions, and the class's methods

point values come from intervals.elementary, which tests/test_elementary.py checks against an
independent oracle; here the set logic is checked: domains and their warnings, which points attain an
end, extrema and poles inside a piece, float typing and the two rounding modes.
"""
import math
import warnings
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals.elementary import exact
from intervals.elementary import rounded
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.fmt import format_cuts
from intervals.fmt import parse
from intervals.functions import MONOTONE
from intervals.functions import NAMES
from intervals.functions import PERIODIC
from intervals.functions import apply
from intervals.functions import atan2
from intervals.kernel import EMPTY
from intervals.kernel import contains_point
from intervals.kernel import intersection
from intervals.kernel import is_subset
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from intervals.kernel import union
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples

INF = math.inf

ABOVE_PI_HALF = 1.5707963267948968  # the double just above pi/2
BELOW_PI_HALF = 1.5707963267948966  # and just below


def show(cuts) -> str:
    return format_cuts(cuts)


def f(name, text, outward=False, base=None) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        return show(apply(name, parse(text), outward=outward, base=base))


# EXAMPLES

@pytest.mark.parametrize('name, a, expected', [
    # exact values stay exact; the ends of a domain are points of it
    ('sqrt', '[1/4, 9]', '[1/2, 3]'),
    ('sqrt', '(0, inf]', '(0, inf]'),
    ('sqrt', '[0, 4)', '[0, 2)'),
    ('exp', '[-inf, 0]', '[0, 1]'),
    ('exp', '(-inf, 0)', '(0, 1)'),
    ('exp2', '[-3, 10]', '[1/8, 1024]'),
    ('exp10', '[-2, 2)', '[1/100, 100)'),
    ('log', '[0, 1]', '[-inf, 0]'),
    ('log', '(0, 1)', '(-inf, 0)'),
    ('log', '[1, inf]', '[0, inf]'),
    ('log2', '[1/8, 1024]', '[-3, 10]'),
    ('log10', '[1/1000, 100)', '[-3, 2)'),
    ('tanh', '[-inf, inf]', '[-1, 1]'),
    ('tanh', '(-inf, inf)', '(-1, 1)'),
    ('tanh', '[0, inf)', '[0, 1)'),
    ('sinh', '[-inf, 0]', '[-inf, 0]'),
    ('cosh', '[-inf, inf]', '[1, inf]'),
    ('cosh', '(-inf, 0]', '[1, inf)'),
    ('asinh', '[0, inf]', '[0, inf]'),
    ('acosh', '[1, inf)', '[0, inf)'),
    ('atanh', '[-1, 1]', '[-inf, inf]'),
    ('atanh', '(-1, 0]', '(-inf, 0]'),
    ('atanh', '[1]', '[inf]'),
    ('asin', '[0]', '[0]'),
    ('acos', '[1]', '[0]'),
    ('atan', '[0]', '[0]'),
    # an irrational value of an exact point: the open one-ulp piece around it
    ('sqrt', '[2]', '(1.414213562373095, 1.4142135623730951)'),
    ('exp', '[1]', '(2.718281828459045, 2.7182818284590455)'),
    ('exp', '[0, 1]', '[1, 2.7182818284590455)'),
    ('atan', '[0, inf]', '[0, 1.5707963267948968)'),
    ('atan', '[-inf, inf]', '(-1.5707963267948968, 1.5707963267948968)'),
    ('acos', '[-1, 1]', '[0, 3.1415926535897936)'),
    ('asin', '[-1, 1)', '(-1.5707963267948968, 1.5707963267948968)'),
])
def test_monotone_examples(name, a, expected):
    assert f(name, a) == expected


@pytest.mark.parametrize('a, base, expected', [
    ('[1/9, 27]', 3, '[-2, 3]'),
    ('[1/8, 4]', Fraction(1, 2), '[-2, 3]'),  # a base below 1 turns the function around
    ('[0, 1]', Fraction(1, 2), '[0, inf]'),
    ('[1, inf]', 10, '[0, inf]'),
    ('[1, inf]', 0.1, '[-inf, 0]'),
    ('(8, 9)', 2, '(3, 3.1699250014423126)'),
])
def test_log_base(a, base, expected):
    assert f('log', a, base=base) == expected


@pytest.mark.parametrize('base', [1, 0, -2, INF, 1.0])
def test_log_base_refused(base):
    with pytest.raises(ValueError):
        apply('log', parse('[1, 2]'), base=base)


def test_base_only_for_log():
    with pytest.raises(TypeError):
        apply('exp', parse('[1, 2]'), base=2)
    with pytest.raises(TypeError):
        apply('log', parse('[1, 2]'), base='2')


@pytest.mark.parametrize('name, a, expected', [
    # sin: maximum at pi/2 inside, minimum at 3 pi/2 inside, both, neither
    ('sin', '[0, 2]', '[0, 1]'),
    ('sin', '(0, 2)', '(0, 1]'),
    ('sin', '[4, 5]', '[-1, -0.7568024953079282)'),
    ('sin', '[1, 5]', '[-1, 1]'),
    ('sin', '[-1, 1]', '(-0.8414709848078966, 0.8414709848078966)'),
    ('sin', '[0, 1]', '[0, 0.8414709848078966)'),
    ('sin', '(0, 1]', '(0, 0.8414709848078966)'),
    ('sin', '[0, inf)', '[-1, 1]'),
    ('sin', '(-inf, 0]', '[-1, 1]'),
    # cos: the maximum at 0 is rational, so it can be an end of the piece
    ('cos', '[0, 1]', '(0.5403023058681397, 1]'),
    ('cos', '(0, 1]', '(0.5403023058681397, 1)'),
    ('cos', '[-1, 1]', '(0.5403023058681397, 1]'),
    ('cos', '[-1, 1)', '(0.5403023058681397, 1]'),  # cos(-1) = cos(1): the closed end wins the tie
    ('cos', '(-1, 1)', '(0.5403023058681397, 1]'),
    ('cos', '[3, 4]', '[-1, -0.6536436208636118)'),
    ('cos', '[0]', '[1]'),
    ('cos', '[-1, 0)', '(0.5403023058681397, 1)'),  # the maximum at 0 is an open end, not inside
    ('cos', '[-1, 0]', '(0.5403023058681397, 1]'),
    ('cos', '(-6, 0)', '[-1, 1)'),  # -pi inside (the minimum), -2 pi not
    ('cos', '(-7, 0)', '[-1, 1]'),  # -2 pi inside: 1 attained there
    # tan: a pole inside splits the image, both infinities attained; two poles give everything
    ('tan', '[0, 1]', '[0, 1.5574077246549023)'),
    ('tan', '[1, 2]', '{ [-inf, -2.185039863261519) , (1.557407724654902, inf] }'),
    ('tan', '(1, 2)', '{ [-inf, -2.185039863261519) , (1.557407724654902, inf] }'),
    ('tan', '[1, 5]', '[-inf, inf]'),
    ('tan', '[0, inf)', '[-inf, inf]'),
    ('tan', f'[{BELOW_PI_HALF}, {ABOVE_PI_HALF}]', '{ [-inf, -6218431163823738.0] , [1.633123935319537e+16, inf] }'),
])
def test_periodic_examples(name, a, expected):
    assert f(name, a) == expected


@pytest.mark.parametrize('name, a, clipped', [
    ('sqrt', '[-1, 4]', '[0, 2]'),
    ('sqrt', '[-1, 0)', ''),
    ('log', '[-inf, 1]', '[-inf, 0]'),
    ('asin', '[0, 2]', '[0, 1.5707963267948968)'),
    ('acosh', '[0, 1]', '[0]'),
    ('atanh', '(-2, 2)', '[-inf, inf]'),
    ('sin', '[0, inf]', '[-1, 1]'),
    ('cos', '[-inf]', ''),
    ('tan', '{ [-inf] , [0] }', '[0]'),
])
def test_domain_clipping_warns(name, a, clipped):
    with pytest.warns(DomainClippedWarning):
        result = apply(name, parse(a))
    assert (show(result) if result else '') == clipped


def test_no_warning_inside_the_domain():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        apply('sqrt', parse('[0, inf]'))
        apply('sin', parse('(-inf, inf)'))
        apply('atanh', parse('[-1, 1]'))


def test_empty_operand_warns():
    with pytest.warns(EmptySetPropagationWarning):
        assert apply('exp', EMPTY) == EMPTY


def test_unknown_function():
    with pytest.raises(ValueError):
        apply('cbrt', parse('[1]'))


# FLOATS: TYPE AND ROUNDING

@pytest.mark.parametrize('name, a, nearest, outward', [
    ('sqrt', '[2.0]', '[1.4142135623730951]', '(1.414213562373095, 1.4142135623730951)'),
    ('sqrt', '[4.0]', '[2.0]', '[2.0]'),
    ('exp', '[0.0]', '[1.0]', '[1.0]'),
    ('exp10', '[-1.0]', '[0.1]', '(0.09999999999999999, 0.1)'),
    ('log2', '[0.125, 8.0]', '[-3.0, 3.0]', '[-3.0, 3.0]'),
    ('exp', '[0.0, 1.0]', '[1.0, 2.718281828459045]', '[1.0, 2.7182818284590455)'),
    ('sin', '[0.0, 4.0]', '[-0.7568024953079282, 1.0]', '(-0.7568024953079283, 1.0]'),
    ('exp', '[1000.0]', '[inf]', '(1.7976931348623157e+308, inf)'),
    ('exp', '[-1000.0]', '[0.0]', '(0.0, 5e-324)'),
    ('tanh', '[30.0]', '[1.0]', '(0.9999999999999999, 1.0)'),
])
def test_float_modes(name, a, nearest, outward):
    assert f(name, a) == nearest
    assert f(name, a, outward=True) == outward


def test_mixed_float_and_exact_ends():
    # the exact end stays exact, the float end is a float
    assert f('exp2', '[0, 1.5]') == '[1, 2.8284271247461903]'
    assert f('exp2', '[0, 1.5]', outward=True) == '[1, 2.8284271247461903)'


# PROPERTIES

def _value_in(result, name, x) -> bool:
    """f(x) is in result: exactly, or (irrational) the open one-ulp piece around it is"""
    value = exact(name, x)
    if value is not None:
        return contains_point(result, value)
    lo, hi = rounded(name, x, DOWN), rounded(name, x, UP)
    return is_subset(normalize([piece(lo, hi, False, False)]), result)


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=40, deadline=None)
@given(a=exact_cut_tuples, rng=st.randoms(use_true_random=False))
def test_sound_on_exact_sets(name, a, rng):
    """every point of the operand in the domain has its value in the result"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a)
    domain = MONOTONE[name][0] if name in MONOTONE else normalize([piece(-INF, INF, False, False)])
    inside = intersection(a, domain)
    for x in sample(inside, 15, rng):
        assert _value_in(result, name, Fraction(x) if x not in (-INF, INF) else x), (name, show(a), x, show(result))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=40, deadline=None)
@given(a=exact_cut_tuples)
def test_ends_are_sharp_on_exact_sets(name, a):
    """
    every finite end of the result is the value, or the rounded value, of an end of an operand piece,
    or an extremum (±1 for sin and cos); closed only if an operand point attains it exactly
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a)
    domain = MONOTONE[name][0] if name in MONOTONE else normalize([piece(-INF, INF, False, False)])
    ends = {cut.value for cut in intersection(a, domain)}
    candidates = {1, -1, INF, -INF} if name in PERIODIC else {INF, -INF}
    if name == 'cosh' and contains_point(a, 0):
        ends.add(0)  # the minimum inside a piece
    for x in ends:
        if x in (-INF, INF) and name in PERIODIC:
            continue
        value = exact(name, x)
        if value is not None:
            candidates.add(value)
        else:
            candidates.update((rounded(name, x, DOWN), rounded(name, x, UP)))
    for lo, lo_closed, hi, hi_closed in pieces(result):
        for v, closed in ((lo, lo_closed), (hi, hi_closed)):
            assert v in candidates, (name, show(a), show(result), v)
            if closed and v not in (1, -1, INF, -INF):
                assert any(exact(name, x) == v for x in ends if contains_point(a, x)), (name, show(a), v)


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=30, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_isotone(name, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        small, large = apply(name, intersection(a, b)), apply(name, a)
    assert is_subset(small, large), (name, show(a), show(b))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=30, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_distributes_over_union(name, a, b):
    """
    no function here has a pole with two sides at a point of its domain (tan's poles are irrational,
    and log's and atanh's are domain ends), so f(A ∪ B) == f(A) ∪ f(B) exactly
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        whole, parts = apply(name, union(a, b)), union(apply(name, a), apply(name, b))
    assert whole == parts, (name, show(a), show(b), show(whole), show(parts))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=30, deadline=None)
@given(a=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_outward_sound_on_floats(name, a, rng):
    """outward, the exact value of every float point is in the result"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a, outward=True)
    domain = MONOTONE[name][0] if name in MONOTONE else normalize([piece(-INF, INF, False, False)])
    for x in sample(intersection(a, domain), 12, rng):
        assert _value_in(result, name, Fraction(x) if x not in (-INF, INF) else x), (name, show(a), x, show(result))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=30, deadline=None)
@given(a=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_nearest_holds_the_nearest_value_of_every_float(name, a, rng):
    """
    to nearest, the nearest double to the value of every float point is in the result's closure: a
    flag at a rounded end is conservative, not a promise (`exp(-1024.0)` rounds to 0.0, an open end)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = normalize(piece(lo, hi) for lo, _, hi, _ in pieces(apply(name, a)))
    domain = MONOTONE[name][0] if name in MONOTONE else normalize([piece(-INF, INF, False, False)])
    for x in sample(intersection(a, domain), 12, rng):
        if isinstance(x, float) and math.isfinite(x):
            value = exact(name, Fraction(x))
            v = rounded(name, Fraction(x), NEAREST) if value is None else value
            assert contains_point(result, v), (name, show(a), x, v, show(result))


def test_a_float_near_a_pole_of_tan():
    """the doubles around pi/2 lie on either side of it, so the piece between them holds the pole"""
    assert f('tan', f'[{BELOW_PI_HALF}]') == '[1.633123935319537e+16]'
    assert f('tan', f'[{ABOVE_PI_HALF}]') == '[-6218431163823738.0]'


def test_huge_arguments():
    """a wide piece is recognised from two floor computations, whatever its width; a narrow one far out
    is reduced exactly (1e22 lies about 1.02 short of a multiple of 2 pi, so cos rises through its maximum)"""
    assert f('sin', '[1, 1e300]') == '[-1.0, 1.0]'
    assert f('cos', f'[{10 ** 22}, {10 ** 22 + 3}]') == '(-0.3977161208638285, 1]'


# THE CLASS

def test_methods():
    a = MultiInterval(1, 4)
    assert a.sqrt() == MultiInterval(1, 2)
    assert a.log(2) == MultiInterval(0, 2)
    assert a.log2() == MultiInterval(0, 2)
    assert MultiInterval(0).exp() == MultiInterval(1)
    assert MultiInterval(0, 1).atanh() == MultiInterval(0, math.inf)
    for name in NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert isinstance(getattr(a, name)(), MultiInterval), name


def test_methods_keep_the_class():
    a = OutwardMultiInterval(0.5, 2.0)
    for name in NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert type(getattr(a, name)()) is OutwardMultiInterval, name
    assert a.exp() == MultiInterval.from_cuts(apply('exp', a.cuts, outward=True))
    assert MultiInterval(0.5, 2.0).exp() == MultiInterval.from_cuts(apply('exp', a.cuts))


# ATAN2

@pytest.mark.parametrize('y, x, expected', [
    ('[0, 1]', '[1]', '[0, 0.7853981633974484)'),
    ('[-1, 0]', '[-1]', '{ (-3.1415926535897936, -2.356194490192345) , (3.141592653589793, 3.1415926535897936) }'),
    ('[-1, 0)', '[-2, -1]', '(-3.1415926535897936, -2.356194490192345)'),
    ('[0]', '[-2, -1]', '(3.141592653589793, 3.1415926535897936)'),
    ('[0]', '(0, inf]', '[0]'),
    ('(0, inf]', '[0]', '(1.5707963267948966, 1.5707963267948968)'),
    ('[1]', '[-inf, inf]', '[0, 3.1415926535897936)'),
    ('[-inf, inf]', '[inf]', '[0]'),
    ('[1, inf]', '[1, inf]', '[0, 1.5707963267948968)'),
    ('[2]', '[-inf]', '(3.141592653589793, 3.1415926535897936)'),
    ('[-2]', '[-inf]', '(-3.1415926535897936, -3.141592653589793)'),
    ('(-inf, inf)', '(-inf, inf)', '(-3.1415926535897936, 3.1415926535897936)'),
    ('[1.0]', '[1.0]', '[0.7853981633974483]'),
    # 0 is attained only along the edge x = inf, where every finite y gives it
    ('(1, 2)', '[5, inf]', '[0, 0.3805063771123649)'),
    ('(1, 2)', '[5, inf)', '(0, 0.3805063771123649)'),
    ('[1, inf]', '(-inf, -1]', '(1.5707963267948966, 3.1415926535897936)'),
])
def test_atan2_examples(y, x, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IndeterminateResultWarning)
        assert show(atan2(parse(y), parse(x))) == expected


@pytest.mark.parametrize('y, x, expected', [
    ('[0]', '[0]', ''),
    ('[inf]', '[inf]', ''),
    ('[-inf]', '[inf]', ''),
    # (0, 0) and (inf, -inf) have no angle; (0, -inf) is pi and (inf, 0) is pi/2
    ('{ [0] , [inf] }', '{ [0] , [-inf] }', '{ (1.5707963267948966, 1.5707963267948968) , (3.141592653589793, 3.1415926535897936) }'),
])
def test_atan2_indeterminate_boxes_warn(y, x, expected):
    with pytest.warns(IndeterminateResultWarning):
        result = atan2(parse(y), parse(x))
    assert (show(result) if result else '') == expected


def test_atan2_warns_only_for_a_box_with_no_angle():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        atan2(parse('[-1, 1]'), parse('[-1, 1]'))  # holds the origin, but other points have angles
        atan2(parse('[1, inf]'), parse('[1, inf]'))


def _true_angle(v, u):
    """the angle of (u, v) as a Decimal, or 0 exactly, from the decimal oracle in test_elementary"""
    from decimal import Decimal
    from decimal import localcontext
    from tests.test_elementary import _atan
    from tests.test_elementary import _pi
    with localcontext() as ctx:
        ctx.prec = 60
        pi = _pi(60)
        if v == 0 and u > 0:
            return 0
        if v == 0 and u < 0:
            return pi
        if u == 0:
            return pi / 2 if v > 0 else -pi / 2
        if v in (INF, -INF):
            return pi / 2 if v > 0 else -pi / 2
        if u == INF:
            return 0
        if u == -INF:
            return pi if v >= 0 else -pi
        q = Fraction(v) / Fraction(u)
        a = _atan(Decimal(q.numerator) / Decimal(q.denominator), 60)
        return a if u > 0 else a + pi if v > 0 else a - pi


def _holds_angle(result, value) -> bool:
    from decimal import Decimal
    if value == 0:
        return contains_point(result, 0)
    for lo, lo_closed, hi, hi_closed in pieces(result):
        above = lo == -INF or Decimal(lo) < value
        below = hi == INF or value < Decimal(hi)
        if above and below:
            return True
    return False


@settings(max_examples=150, deadline=None)
@given(y=exact_cut_tuples, x=exact_cut_tuples, rng=st.randoms(use_true_random=False))
def test_atan2_sound_on_exact_sets(y, x, rng):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = atan2(y, x)
    for v in sample(y, 6, rng):
        for u in sample(x, 6, rng):
            if (v == 0 and u == 0) or (v in (INF, -INF) and u in (INF, -INF)):
                continue
            assert _holds_angle(result, _true_angle(v, u)), (show(y), show(x), v, u, show(result))


@settings(max_examples=100, deadline=None)
@given(y=cut_tuples(max_pieces=3), x=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_atan2_outward_sound_on_floats(y, x, rng):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = atan2(y, x, outward=True)
    for v in sample(y, 5, rng):
        for u in sample(x, 5, rng):
            if (v == 0 and u == 0) or (v in (INF, -INF) and u in (INF, -INF)):
                continue
            assert _holds_angle(result, _true_angle(v, u)), (show(y), show(x), v, u, show(result))


@settings(max_examples=100, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, x=exact_cut_tuples)
def test_atan2_distributes_over_union(a, b, x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert atan2(union(a, b), x) == union(atan2(a, x), atan2(b, x))
        assert atan2(x, union(a, b)) == union(atan2(x, a), atan2(x, b))


def test_atan2_method_and_class():
    assert MultiInterval(0).atan2(MultiInterval(1, 2)) == MultiInterval(0)
    assert type(OutwardMultiInterval(1.0).atan2(MultiInterval(1.0))) is OutwardMultiInterval
    assert OutwardMultiInterval(1.0).atan2(1.0) == OutwardMultiInterval.parse('(0.7853981633974483, 0.7853981633974484)')
