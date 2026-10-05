"""
the elementary functions over sets: multiinterval.functions, and the class's methods

point values come from multiinterval.elementary, which tests/test_elementary.py checks against an
independent oracle; here the set logic is checked: domains and their warnings, which points attain an
end, extrema and poles inside a piece, float typing and the two rounding modes. the two-argument pow
and hypot, and rootn with its degree, are at the end.
"""
import math
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval.elementary import POLE_AT_ZERO as POLES
from multiinterval.elementary import exact
from multiinterval.elementary import exact_pow
from multiinterval.elementary import rounded
from multiinterval.elementary import rounded_pow
from multiinterval.errors import DomainClippedWarning
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import IndeterminateResultWarning
from multiinterval.fmt import format_cuts
from multiinterval.fmt import parse
from multiinterval.functions import NAMES
from multiinterval.functions import PERIODIC
from multiinterval.functions import RECIPROCAL_TRIG
from multiinterval.functions import apply
from multiinterval.functions import atan2
from multiinterval.functions import domain
from multiinterval.functions import hypot
from multiinterval.functions import pow_
from multiinterval.kernel import EMPTY
from multiinterval.kernel import contains_point
from multiinterval.kernel import intersection
from multiinterval.kernel import is_subset
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import union
from multiinterval.rounding import DOWN
from multiinterval.rounding import NEAREST
from multiinterval.rounding import UP
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.test_elementary import _atan  # at import time: decorating its @given tests inside a
from tests.test_elementary import _pi  # running one is hypothesis's nested-@given error

INF = math.inf
FINITE = normalize([piece(-INF, INF, False, False)])
# the functions with no limit at ±inf
TRIG = PERIODIC + tuple(RECIPROCAL_TRIG)

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
    ('log1p', '[-2, 0]', '[-inf, 0]'),
    ('acoth', '[0, 2]', '(0.5493061443340548, inf]'),
    ('acoth', '(-1, 1)', ''),
    ('sec', '[0, inf]', '{ [-inf, -1] , [1, inf] }'),
    ('cot', '[inf]', ''),
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
        apply('nosuch', parse('[1]'))


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
    ('cbrt', '[8.0]', '[2.0]', '[2.0]'),
    ('expm1', '[1e-20]', '[1e-20]', '(1e-20, 1.0000000000000001e-20)'),
    ('coth', '[0.0, 1.0]', '[1.3130352854993312, inf]', '(1.3130352854993312, inf]'),
    ('csc', '[-3.0, 0.0]', '[-inf, -1.0]', '[-inf, -1.0]'),
    ('sech', '[1000.0]', '[0.0]', '(0.0, 5e-324)'),
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
    inside = intersection(a, domain(name))
    for x in sample(inside, 15, rng):
        if x == 0 and name in POLES:
            continue  # no value, only a limit on each side
        assert _value_in(result, name, Fraction(x) if x not in (-INF, INF) else x), (name, show(a), x, show(result))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=40, deadline=None)
@given(a=exact_cut_tuples)
def test_ends_are_sharp_on_exact_sets(name, a):
    """
    every finite end of the result is the value, or the rounded value, of an end of an operand piece,
    or an extremum (±1 for sin, cos, csc and sec); closed only if an operand point attains it exactly
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a)
    ends = {cut.value for cut in intersection(a, domain(name))}
    candidates = {1, -1, INF, -INF} if name in TRIG else {INF, -INF}
    if name in ('cosh', 'sech') and contains_point(a, 0):
        ends.add(0)  # the extremum inside a piece
    if name in POLES:
        ends.discard(0)  # a pole: its limits are the candidates ±inf
    for x in ends:
        if x in (-INF, INF) and name in TRIG:
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
    f(A ∪ B) == f(A) ∪ f(B) exactly wherever no pole with two sides is a point of the domain (tan's
    and sec's poles are irrational, and log's and atanh's are domain ends). at the pole 0 of cot, csc,
    coth and csch, `A = [-1, 0)` and `B = [0]` give an open -inf apiece, `A ∪ B` a closed one: the
    image of the union can gain the infinities, and is otherwise the same
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        whole, parts = apply(name, union(a, b)), union(apply(name, a), apply(name, b))
    if name in POLES:
        assert is_subset(parts, whole) and intersection(whole, FINITE) == intersection(parts, FINITE), \
            (name, show(a), show(b), show(whole), show(parts))
    else:
        assert whole == parts, (name, show(a), show(b), show(whole), show(parts))


@pytest.mark.parametrize('name', NAMES)
@settings(max_examples=30, deadline=None)
@given(a=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_outward_sound_on_floats(name, a, rng):
    """outward, the exact value of every float point is in the result"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a, outward=True)
    for x in sample(intersection(a, domain(name)), 12, rng):
        if x == 0 and name in POLES:
            continue
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
    for x in sample(intersection(a, domain(name)), 12, rng):
        if isinstance(x, float) and math.isfinite(x) and not (x == 0 and name in POLES):
            # the nearest double even where the value is rational: exp10(-1.0) is 1/10, and the
            # result's end is the double 0.1, just above it
            v = rounded(name, Fraction(x), NEAREST)
            assert contains_point(result, v), (name, show(a), x, v, show(result))


@pytest.mark.parametrize('name, a, expected', [
    # expm1, log1p, cbrt: exact where rational, the domain's end a point of it
    ('expm1', '[-inf, 0]', '[-1, 0]'),
    ('expm1', '(-inf, inf)', '(-1, inf)'),
    ('expm1', '[0, 1]', '[0, 1.7182818284590453)'),
    ('log1p', '[-1, 0]', '[-inf, 0]'),
    ('log1p', '(-1, inf]', '(-inf, inf]'),
    ('log1p', '[0, 1]', '[0, 0.6931471805599454)'),
    ('cbrt', '[-27, 8]', '[-3, 2]'),
    ('cbrt', '[-inf, 1/8)', '[-inf, 1/2)'),
    ('cbrt', '[2]', '(1.259921049894873, 1.2599210498948732)'),
    # acot falls from pi to 0; acoth falls on each side of (-1, 1), ±inf at ±1 and 0 at ±inf
    ('acot', '[-inf, inf]', '[0, 3.1415926535897936)'),
    ('acot', '[0, inf)', '(0, 1.5707963267948968)'),
    ('acoth', '[1, inf]', '[0, inf]'),
    ('acoth', '[-inf, -1]', '[-inf, 0]'),
    ('acoth', '(1, 2]', '(0.5493061443340548, inf)'),
    ('acoth', '[-inf, inf]', '[-inf, inf]'),
    # sech: its maximum 1 at 0
    ('sech', '[-inf, inf]', '[0, 1]'),
    ('sech', '(-inf, 0)', '(0, 1)'),
    ('sech', '[-1, 2]', '(0.26580222883407967, 1]'),
    # coth and csch: a pole at 0 with a side each way, as 1/x
    ('coth', '[0, 1]', '(1.3130352854993312, inf]'),
    ('coth', '(0, 1]', '(1.3130352854993312, inf)'),
    ('coth', '[-1, 1]', '{ [-inf, -1.3130352854993312) , (1.3130352854993312, inf] }'),
    ('coth', '[-inf, -1]', '(-1.3130352854993315, -1]'),
    ('coth', '[-inf, inf]', '{ [-inf, -1] , [1, inf] }'),
    ('csch', '[-inf, 0)', '(-inf, 0]'),
    ('csch', '[-1, 2]', '{ [-inf, -0.8509181282393214) , (0.2757205647717832, inf] }'),
    # cot: a pole at every k pi, 0 included, falling between
    ('cot', '[-3, 0]', '[-inf, 7.015252551434534)'),
    ('cot', '[0, 1]', '(0.6420926159343306, inf]'),
    ('cot', '(0, 1]', '(0.6420926159343306, inf)'),
    ('cot', '[-2, 2]', '[-inf, inf]'),
    ('cot', '[1, 4]', '{ [-inf, 0.6420926159343308) , (0.8636911544506165, inf] }'),
    ('cot', '[1, 7]', '[-inf, inf]'),
    # csc: poles at k pi, a minimum 1 at pi/2 and a maximum -1 at 3 pi/2 (mod 2 pi)
    ('csc', '[0, 3]', '[1, inf]'),
    ('csc', '[-3, 0]', '[-inf, -1]'),
    ('csc', '[-6, -4]', '[1, 3.578899547254406)'),
    ('csc', '[1, 2]', '[1, 1.1883951057781212)'),
    ('csc', '[3, 4]', '{ [-inf, -1.3213487088109022) , (7.086167395737186, inf] }'),
    ('csc', '[3, 7]', '{ [-inf, -1] , (1.5221010625637303, inf] }'),  # two poles: not yet the whole range
    ('csc', '[-6, 7]', '{ [-inf, -1] , [1, inf] }'),
    ('csc', '(0, inf)', '{ [-inf, -1] , [1, inf] }'),
    # sec: poles at pi/2 + k pi, a minimum 1 at 0 (a rational extremum) and a maximum -1 at pi
    ('sec', '[0]', '[1]'),
    ('sec', '[0, 1]', '[1, 1.8508157176809257)'),
    ('sec', '[-1, 1]', '[1, 1.8508157176809257)'),
    ('sec', '(0, 1]', '(1, 1.8508157176809257)'),
    ('sec', '[1, 2]', '{ [-inf, -2.402997961722381) , (1.8508157176809255, inf] }'),
    ('sec', '[2, 4]', '(-2.4029979617223813, -1]'),
    ('sec', '[1, 5]', '{ [-inf, -1] , (1.8508157176809255, inf] }'),
])
def test_m13d_examples(name, a, expected):
    assert f(name, a) == expected


@pytest.mark.parametrize('name', ['cot', 'csc', 'coth', 'csch'])
def test_a_pole_at_zero_has_no_value_there(name):
    with pytest.warns(IndeterminateResultWarning):
        assert apply(name, parse('[0]')) == EMPTY
    with pytest.warns(IndeterminateResultWarning):  # the rest of the operand still counts
        assert show(apply(name, parse('{ [0] , [1] }'))) == show(apply(name, parse('[1]')))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        apply(name, parse('[0, 1]'))  # the limit from above, no warning


def _reaches_zero(a, side: int) -> bool:
    """a holds 0 and points on that side of it arbitrarily close (a piece [0, x] with x > 0, or
    [x, 0] with x < 0, or one with 0 inside); the point [0] alone has no side"""
    return any(lo <= 0 <= hi and (lo < 0 if side < 0 else hi > 0) and
               (lo < 0 < hi or (lo == 0 and lc) or (hi == 0 and hc))
               for lo, lc, hi, hc in pieces(a))


@pytest.mark.parametrize('name, n', [('coth', None), ('csch', None), ('rootn', -3), ('rootn', -1)])
@settings(max_examples=80, deadline=None)
@given(a=exact_cut_tuples)
@example(a=parse('(0, 1]'))
@example(a=parse('[-2, 0)'))
@example(a=parse('{ [-1, 0) , (0, 1] }'))
@example(a=parse('[0]'))
def test_the_infinities_of_a_pole_at_zero(name, n, a):
    """
    +inf is in the result iff the operand holds 0 and reaches it from above, -inf iff from below: the
    one-sided limit is attained exactly when its end 0 is. nothing else gives these functions an
    infinity (at ±inf they are ±1 or 0)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply(name, a, base=n)
    assert contains_point(result, INF) == _reaches_zero(a, 1), (name, n, show(a), show(result))
    assert contains_point(result, -INF) == _reaches_zero(a, -1), (name, n, show(a), show(result))


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
    assert MultiInterval(1, 64).rootn(3) == MultiInterval(1, 4) and MultiInterval(1, 64).rootn(-6) == MultiInterval(Fraction(1, 2), 1)
    assert MultiInterval(3).hypot(4) == MultiInterval(5) and a.hypot(MultiInterval(0)) == a
    assert MultiInterval(0).log1p() == MultiInterval(0) and MultiInterval(0).expm1() == MultiInterval(0)
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
    assert type(a.rootn(3)) is OutwardMultiInterval and type(a.hypot(MultiInterval(1))) is OutwardMultiInterval
    assert type(MultiInterval(1).hypot(a)) is OutwardMultiInterval  # the operators' class (Q15(h), 2026-10-03)


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


# ROOTN

DEGREES = [n for n in range(-5, 7) if n != 0] + [64, -63]


@pytest.mark.parametrize('n, a, expected', [
    (3, '[-8, 27]', '[-2, 3]'),
    (4, '[0, 16]', '[0, 2]'),
    (2, '[2]', '(1.414213562373095, 1.4142135623730951)'),
    (1, '[-3, 5)', '[-3, 5)'),
    (-1, '[-1, 1]', '{ [-inf, -1] , [1, inf] }'),
    (-2, '[0, 4]', '[1/2, inf]'),  # even: 0 is the domain's end, the limit from above
    (-2, '(0, inf]', '[0, inf)'),
    (-3, '[-8, 8]', '{ [-inf, -1/2] , [1/2, inf] }'),
    (-3, '[0, 8]', '[1/2, inf]'),
    (-3, '(-inf, -1]', '[-1, 0)'),
    (5, '[-inf, inf]', '[-inf, inf]'),
])
def test_rootn_examples(n, a, expected):
    assert show(apply('rootn', parse(a), base=n)) == expected


def test_rootn_domain_and_poles():
    with pytest.warns(DomainClippedWarning):
        assert show(apply('rootn', parse('[-1, 4]'), base=2)) == '[0, 2]'
    with pytest.warns(IndeterminateResultWarning):
        assert apply('rootn', parse('[0]'), base=-3) == EMPTY
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert show(apply('rootn', parse('[0]'), base=-4)) == '[inf]'


@pytest.mark.parametrize('n', [0, True, 2.0, Fraction(2), None])
def test_rootn_degree_refused(n):
    with pytest.raises((TypeError, ValueError)):
        apply('rootn', parse('[1, 2]'), base=n)


@settings(max_examples=60, deadline=None)
@given(a=exact_cut_tuples, n=st.sampled_from(DEGREES), rng=st.randoms(use_true_random=False))
def test_rootn_sound_on_exact_sets(a, n, rng):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = apply('rootn', a, base=n)
    for x in sample(intersection(a, domain('rootn', n)), 12, rng):
        if x == 0 and n < 0 and n % 2:
            continue
        x = x if x in (-INF, INF) else Fraction(x)
        value = exact('rootn', x, n)
        if value is not None:
            assert contains_point(result, value), (n, show(a), x, show(result))
        else:
            lo, hi = rounded('rootn', x, DOWN, n), rounded('rootn', x, UP, n)
            assert is_subset(normalize([piece(lo, hi, False, False)]), result), (n, show(a), x, show(result))


@settings(max_examples=40, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, n=st.sampled_from(DEGREES))
def test_rootn_isotone(a, b, n):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert is_subset(apply('rootn', intersection(a, b), base=n), apply('rootn', a, base=n))


@pytest.mark.parametrize('n, a, nearest, outward', [
    # fuzz x10 on CI (run 37098878528, 2026-10-03): the exact end gives 1/10 ** 6 exactly, the float end rounds to
    # nearest onto 1e-06, below it, and the reversed piece raised ValueError
    (5, '(1/1000000000000000000000000000000, 1.0000000000000003e-30]', '[1e-06, 1/1000000)',
     '(1/1000000, 1.0000000000000002e-06)'),
])
def test_rootn_ends_crossed_by_rounding(n, a, nearest, outward):
    """to nearest, a float end can round past an exact one: the piece is between the two values, each keeping its
    flag, as the applicator's least and greatest corner values are. outward rounding never crosses"""
    assert show(apply('rootn', parse(a), base=n)) == nearest
    assert show(apply('rootn', parse(a), outward=True, base=n)) == outward


def test_cbrt_is_rootn_3():
    for text in ('[-27, 8]', '[2, 3]', '(-inf, 1/3]', '{ [-1] , (2, 5) }', '[0.1, 7.5]'):
        assert apply('cbrt', parse(text)) == apply('rootn', parse(text), base=3), text
        assert apply('cbrt', parse(text), outward=True) == apply('rootn', parse(text), outward=True, base=3), text


# POW (ieee 1788's, D11)

def p_(a, b, outward=False) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        result = pow_(parse(a), parse(b), outward=outward)
    return show(result) if result else ''


@pytest.mark.parametrize('a, b, expected', [
    # exact where rational; irrational ends are open one-ulp enclosures
    ('[1/4, 4]', '[1/2]', '[1/2, 2]'),
    ('[4, 9]', '[3/2]', '[8, 27]'),
    ('[8]', '[-2/3]', '[1/4]'),
    ('[2]', '[1/2]', '(1.414213562373095, 1.4142135623730951)'),
    ('[2, 3]', '[2]', '[4, 9]'),
    # the base splits at 1: x < 1 falls with y, x > 1 rises
    ('[1/2, 2]', '[2]', '[1/4, 4]'),
    ('[1/2, 2]', '[-1, 1]', '[1/2, 2]'),
    ('[1/2]', '[-inf, inf]', '[0, inf]'),
    ('[2]', '(-inf, inf)', '(0, inf)'),
    # 0 ** y is 0 for y > 0; nothing for y <= 0 (dropped with a warning)
    ('[0]', '[1, 2]', '[0]'),
    ('[0, 1]', '[1/2, 3]', '[0, 1]'),
    ('[0, 2]', '[-1, 1]', '[0, inf)'),
    ('[0, 1]', '[-1]', '[1, inf)'),
    ('[-3, 1]', '[2]', '[0, 1]'),  # an interval exponent means pow, never pown: [0, 1], not [0, 9]
    # 1 ** y is 1 and x ** 0 is 1
    ('[1]', '[-inf, inf]', '[1]'),
    ('[0, inf]', '[0]', '[1]'),
    # ±inf as points: inf ** y is inf or 0, x ** ±inf is 0 or inf by the side of 1
    ('[inf]', '[1, 2]', '[inf]'),
    ('[inf]', '[-2, -1]', '[0]'),
    ('[2, 3]', '[inf]', '[inf]'),
    ('[1/3, 1/2]', '[inf]', '[0]'),
    ('[1/3, 1/2]', '[-inf]', '[inf]'),
    ('[2, inf]', '[-inf, -1]', '[0, 1/2]'),
    ('(2, inf)', '(-inf, -1]', '(0, 1/2)'),
    # a closed infinite edge attains the constant value along it
    ('[2, inf]', '(1, 2)', '(2, inf]'),
    ('(1, 2)', '[1, inf]', '(1, inf]'),
    # float operands round once: to nearest, or outward
    ('[0.1, 0.5]', '[0.0, 2.5]', '[0.00316227766016838, 1.0]'),
    ('[2.0]', '[0.5]', '[1.4142135623730951]'),
    ('[4.0]', '[0.5]', '[2.0]'),
])
def test_pow_examples(a, b, expected):
    assert p_(a, b) == expected


@pytest.mark.parametrize('a, b, expected', [
    ('[0.1, 0.5]', '[0.0, 2.5]', '(0.0031622776601683794, 1.0]'),
    ('[2.0]', '[0.5]', '(1.414213562373095, 1.4142135623730951)'),
    ('[4.0]', '[0.5]', '[2.0]'),
    ('[10.0]', '[400.0]', '(1.7976931348623157e+308, inf)'),
    ('[10.0]', '[-400.0]', '(0.0, 5e-324)'),
])
def test_pow_outward_examples(a, b, expected):
    assert p_(a, b, outward=True) == expected


@pytest.mark.parametrize('a, b, expected', [
    ('[-2, -1]', '[1, 2]', ''),
    ('[0]', '[-1, 0]', ''),
    ('[-1, 4]', '[1/2]', '[0, 2]'),
    ('[0, 1]', '[-1, 0]', '[1, inf)'),
])
def test_pow_domain_warns(a, b, expected):
    with pytest.warns(DomainClippedWarning):
        result = pow_(parse(a), parse(b))
    assert (show(result) if result else '') == expected


@pytest.mark.parametrize('a, b, expected', [
    ('[1]', '[inf]', ''),
    ('[1]', '[-inf]', ''),
    ('[inf]', '[0]', ''),
    ('{ [1] , [2] }', '[inf]', '[inf]'),  # (1, inf) has no value, (2, inf) is inf
])
def test_pow_indeterminate_boxes_warn(a, b, expected):
    with pytest.warns(IndeterminateResultWarning):
        result = pow_(parse(a), parse(b))
    assert (show(result) if result else '') == expected


def test_pow_warns_only_for_a_box_with_no_value():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pow_(parse('[1/2, 2]'), parse('[-inf, inf]'))  # holds (1, ±inf), but other points have values
        pow_(parse('[1, inf]'), parse('[0, 1]'))  # holds (inf, 0)


def test_pow_empty_operand_warns():
    with pytest.warns(EmptySetPropagationWarning):
        assert pow_(EMPTY, parse('[1]')) == EMPTY


def _power_value(x, y):
    """x ** y at a point of pow's domain with a value, as (exact, None) or (None, (down, up))"""
    if x == INF:
        return (INF if y > 0 else 0), None
    if y in (-INF, INF):
        return (INF if (x > 1) == (y > 0) else 0), None
    value = exact_pow(Fraction(x), Fraction(y))
    if value is not None:
        return value, None
    return None, (rounded_pow(Fraction(x), Fraction(y), DOWN), rounded_pow(Fraction(x), Fraction(y), UP))


def _has_power(x, y) -> bool:
    return x >= 0 and not (x == 0 and y <= 0) and not (x == 1 and y in (-INF, INF)) and not (x == INF and y == 0)


def _holds_power(result, x, y) -> bool:
    value, enclosure = _power_value(x, y)
    if enclosure is None:
        return contains_point(result, value)
    return is_subset(normalize([piece(enclosure[0], enclosure[1], False, False)]), result)


@settings(max_examples=100, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, rng=st.randoms(use_true_random=False))
def test_pow_sound_on_exact_sets(a, b, rng):
    """every pair of the operands with a power has it in the result"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = pow_(a, b)
    for x, y in zip(sample(intersection(a, domain('sqrt')), 12, rng), sample(b, 12, rng)):
        if _has_power(x, y):
            assert _holds_power(result, x, y), (show(a), show(b), x, y, show(result))


@settings(max_examples=100, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_pow_ends_are_sharp_on_exact_sets(a, b):
    """
    every finite end of the result is the power, or its rounding, at a pair of ends of the operand
    pieces (0 and 1 included, the parts' ends), and a closed one is attained at a pair of the operands
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = pow_(a, b)
    xs = {c.value for c in intersection(a, domain('sqrt'))} | {0, 1}
    ys = {c.value for c in b} | {0}
    candidates = {0, 1, INF}
    attained = set()
    for x in xs:
        for y in ys:
            if not _has_power(x, y) and not (x == 0 and y < 0):
                continue
            value, enclosure = _power_value(x, y) if x != 0 else ((0 if y > 0 else INF), None)
            if enclosure is None:
                candidates.add(value)
                if contains_point(a, x) and contains_point(b, y) and _has_power(x, y):
                    attained.add(value)
            else:
                candidates.update(enclosure)
    for lo, lo_closed, hi, hi_closed in pieces(result):
        for v, closed in ((lo, lo_closed), (hi, hi_closed)):
            assert v in candidates, (show(a), show(b), show(result), v)
            if closed and v not in (0, 1, INF):
                assert v in attained, (show(a), show(b), show(result), v)


@settings(max_examples=60, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, c=exact_cut_tuples)
def test_pow_isotone(a, b, c):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert is_subset(pow_(intersection(a, c), b), pow_(a, b))
        assert is_subset(pow_(a, intersection(b, c)), pow_(a, b))


@settings(max_examples=60, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, c=exact_cut_tuples)
def test_pow_distributes_over_union(a, b, c):
    """pow has no pole with two sides: the image of a union is the union of the images, in each argument"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert pow_(union(a, c), b) == union(pow_(a, b), pow_(c, b)), (show(a), show(b), show(c))
        assert pow_(a, union(b, c)) == union(pow_(a, b), pow_(a, c)), (show(a), show(b), show(c))


@settings(max_examples=60, deadline=None)
@given(a=cut_tuples(max_pieces=3), b=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_pow_outward_sound_on_floats(a, b, rng):
    """outward, the exact power of every pair of float points is in the result"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = pow_(a, b, outward=True)
    for x, y in zip(sample(intersection(a, domain('sqrt')), 10, rng), sample(b, 10, rng)):
        if _has_power(x, y):
            assert _holds_power(result, x, y), (show(a), show(b), x, y, show(result))


@settings(max_examples=60, deadline=None)
@given(a=cut_tuples(max_pieces=3), b=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_pow_nearest_holds_the_nearest_power(a, b, rng):
    """to nearest, with a float operand, the nearest double to each power is in the result's closure"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = normalize(piece(lo, hi) for lo, _, hi, _ in pieces(pow_(a, b)))
    for x, y in zip(sample(intersection(a, domain('sqrt')), 10, rng), sample(b, 10, rng)):
        if _has_power(x, y) and INF not in (abs(x), abs(y)) and x != 0:
            if isinstance(x, float) or isinstance(y, float):
                v = rounded_pow(Fraction(x), Fraction(y), NEAREST)
                assert contains_point(result, v), (show(a), show(b), x, y, v, show(result))


# HYPOT

@pytest.mark.parametrize('a, b, expected', [
    ('[3]', '[4]', '[5]'),
    ('[3, 5]', '[-4, 0]', '[3, 6.403124237432849)'),
    ('[-inf, -7]', '[-1, 8]', '[7, inf]'),
    ('(-inf, 0]', '[8, inf)', '[8, inf)'),
    ('[-1, 1]', '[-1, 1]', '[0, 1.4142135623730951)'),
    ('(0, 1]', '[0]', '(0, 1]'),
    ('{ [-2] , [3] }', '[4]', '{ (4.472135954999579, 4.47213595499958) , [5] }'),
    ('[3.0]', '[4.0]', '[5.0]'),
    ('[1.0]', '[1.0]', '[1.4142135623730951]'),
])
def test_hypot_examples(a, b, expected):
    assert show(hypot(parse(a), parse(b))) == expected


def test_hypot_outward_and_empty():
    assert show(hypot(parse('[1.0]'), parse('[1.0]'), outward=True)) == '(1.414213562373095, 1.4142135623730951)'
    assert show(hypot(parse('[0.1]'), parse('[0]'), outward=True)) == '[0.1]'  # exactly |x|
    with pytest.warns(EmptySetPropagationWarning):
        assert hypot(EMPTY, parse('[1]')) == EMPTY


def _hypot_holds(result, x, y) -> bool:
    if INF in (abs(x), abs(y)):
        return contains_point(result, INF)
    s = Fraction(x) ** 2 + Fraction(y) ** 2
    value = exact('sqrt', s)
    if value is not None:
        return contains_point(result, value)
    # irrational: in a piece iff its ends bracket it, decided exactly on the squares (a rational end never
    # equals it). the value's own float bracket need not fit: next to an exact end of the result
    # (`[2/3, inf)` from x = -2/3, y = 0) its lower double is below that end (M13g part 3's gate run,
    # 2026-09-26, found it; the library was right)
    def below(lo):
        return lo == -INF or lo < 0 or Fraction(lo) ** 2 < s

    def above(hi):
        return hi == INF or (hi > 0 and Fraction(hi) ** 2 > s)
    return any(below(lo) and above(hi) for lo, _, hi, _ in pieces(result))


@settings(max_examples=100, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, rng=st.randoms(use_true_random=False))
def test_hypot_sound_on_exact_sets(a, b, rng):
    result = hypot(a, b) if a and b else EMPTY
    for x, y in zip(sample(a, 10, rng), sample(b, 10, rng)):
        assert _hypot_holds(result, x, y), (show(a), show(b), x, y, show(result))


def test_the_hypot_oracle_next_to_an_exact_end():
    """the case the gate's random run found (2026-09-26): hypot(-2/3, 2**-27) is just above 2/3, inside
    `[2/3, inf)`, though its lower double is below 2/3"""
    result = hypot(parse('(-inf, -2/3]'), parse('[0, 1/2)'))
    assert show(result) == '[2/3, inf)'
    assert _hypot_holds(result, Fraction(-2, 3), Fraction(1, 2 ** 27))
    assert not _hypot_holds(parse('(0.6666666666666667, 1]'), Fraction(-2, 3), Fraction(1, 2 ** 27))
    assert not _hypot_holds(parse('[1, 2]'), 3, 4) and _hypot_holds(parse('[1, 2]'), 1, 1)


@settings(max_examples=60, deadline=None)
@given(a=cut_tuples(max_pieces=3), b=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_hypot_outward_sound_on_floats(a, b, rng):
    result = hypot(a, b, outward=True) if a and b else EMPTY
    for x, y in zip(sample(a, 10, rng), sample(b, 10, rng)):
        assert _hypot_holds(result, x, y), (show(a), show(b), x, y, show(result))


@settings(max_examples=60, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, c=exact_cut_tuples)
def test_hypot_isotone_and_symmetric(a, b, c):
    if a and b and c:
        assert is_subset(hypot(intersection(a, c), b) if intersection(a, c) else EMPTY, hypot(a, b))
        assert hypot(a, b) == hypot(b, a)
