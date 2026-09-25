"""
pointwise minimum and maximum, and fma (intervals.ops)

min and max are flat wherever the other operand cannot reach (`min(1, y)` is 1 for every y >= 1), so
the applicator's face rule does not decide their ends; they carry their own attainment. the oracle
here is the definition read on probe points: v is in min(A, B) iff v is in A and B has a point >= v,
or v is in B and A has a point >= v. fma is add(mul) computed exactly, then rounded once.
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
from intervals import ops
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.fmt import format_cuts
from intervals.fmt import parse
from intervals.kernel import EMPTY
from intervals.kernel import contains_point
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from intervals.rounding import exact_cuts
from intervals.rounding import round_rational
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import probe_points

INF = math.inf


def show(cuts) -> str:
    return format_cuts(cuts)


# MINIMUM AND MAXIMUM

@pytest.mark.parametrize('a, b, minimum, maximum', [
    ('[3]', '(1, 5)', '(1, 3]', '[3, 5)'),
    ('[0, 1]', '(5, 6)', '[0, 1]', '(5, 6)'),
    ('(0, 1)', '(0, 1)', '(0, 1)', '(0, 1)'),
    ('[0, 1)', '(0, 1]', '[0, 1)', '(0, 1]'),
    ('[1, 2]', '[1, 2]', '[1, 2]', '[1, 2]'),
    ('[1, 2)', '[2]', '[1, 2)', '[2]'),
    ('[-inf, 0]', '[inf]', '[-inf, 0]', '[inf]'),
    ('{ [0] , [10] }', '[5]', '{ [0] , [5] }', '{ [5] , [10] }'),
    ('(-inf, inf)', '[0]', '(-inf, 0]', '[0, inf)'),
])
def test_examples(a, b, minimum, maximum):
    assert show(ops.minimum(parse(a), parse(b))) == minimum
    assert show(ops.maximum(parse(a), parse(b))) == maximum
    assert show(ops.minimum(parse(b), parse(a))) == minimum  # symmetric


def _in_min(v, a, b) -> bool:
    """the definition: min(x, y) == v for some x in a, y in b"""
    reaches_above = lambda s: any(contains_point(s, p) for p in probe_points(s) if p >= v) or contains_point(s, v)
    return (contains_point(a, v) and reaches_above(b)) or (contains_point(b, v) and reaches_above(a))


@settings(max_examples=300, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_minimum_on_probe_points(a, b):
    """exact membership at every endpoint, every gap and beyond: shape and flags together"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = ops.minimum(a, b)
    for v in probe_points(a, b):
        assert contains_point(result, v) == _in_min(v, a, b), (show(a), show(b), v, show(result))


@settings(max_examples=200, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_maximum_mirrors_minimum(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert ops.maximum(a, b) == ops.neg(ops.minimum(ops.neg(a), ops.neg(b)))


@settings(max_examples=100, deadline=None)
@given(a=cut_tuples(max_pieces=3), b=cut_tuples(max_pieces=3), rng=st.randoms(use_true_random=False))
def test_minimum_sound_on_floats(a, b, rng):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = ops.minimum(a, b)
    for x in sample(a, 8, rng):
        for y in sample(b, 8, rng):
            assert contains_point(result, min(x, y))


def test_empty_operand():
    with pytest.warns(EmptySetPropagationWarning):
        assert ops.minimum(EMPTY, parse('[1]')) == EMPTY


# FMA

@pytest.mark.parametrize('a, b, c, expected', [
    ('[1, 2]', '[3]', '[1]', '[4, 7]'),
    ('[-1, 1]', '[-1, 1]', '[0]', '[-1, 1]'),
    ('[1/3]', '[3]', '[-1]', '[0]'),
    ('[0, 1]', '[inf]', '[5]', '[inf]'),  # 0 * inf has no value; the limit along [0, 1] is inf
    ('[1]', '[inf]', '[-inf]', ''),
])
def test_fma_examples(a, b, c, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IndeterminateResultWarning)
        result = ops.fma(parse(a), parse(b), parse(c))
    assert (show(result) if result else '') == expected


def test_fma_rounds_once():
    """0.1 * 10 is 1 + 2**-54 exactly, which rounds to 1.0, so two roundings lose what one keeps"""
    a, b, c = MultiInterval(0.1), MultiInterval(10), MultiInterval(-1)
    assert a * b + c == MultiInterval(0.0)
    assert a.fma(b, c) == MultiInterval(5.551115123125783e-17)
    assert OutwardMultiInterval(0.1).fma(10, -1) == OutwardMultiInterval(5.551115123125783e-17)


def test_fma_empty_and_indeterminate():
    with pytest.warns(EmptySetPropagationWarning):
        assert ops.fma(parse('[1]'), EMPTY, parse('[1]')) == EMPTY
    with pytest.warns(IndeterminateResultWarning):
        assert ops.fma(parse('[0]'), parse('[inf]'), parse('[1]')) == EMPTY


@settings(max_examples=150, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples, c=exact_cut_tuples)
def test_fma_is_mul_then_add_on_exact_sets(a, b, c):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert ops.fma(a, b, c) == ops.add(ops.mul(a, b), c)


def _rounded(cuts, outward):
    out = []
    for lo, lc, hi, hc in pieces(cuts):
        rlo = lo if lo in (-INF, INF) else round_rational(lo, -1 if outward else 0)
        rhi = hi if hi in (-INF, INF) else round_rational(hi, 1 if outward else 0)
        lc, hc = (lc and rlo == lo, hc and rhi == hi) if outward else (lc, hc)
        out.append(piece(rlo, rhi, True, True) if rlo == rhi else piece(rlo, rhi, lc, hc))
    return normalize(out)


@settings(max_examples=150, deadline=None)
@given(a=cut_tuples(max_pieces=2), b=cut_tuples(max_pieces=2), c=cut_tuples(max_pieces=2))
def test_fma_with_floats_rounds_the_exact_result_once(a, b, c):
    if not any(isinstance(x.value, float) and math.isfinite(x.value) for x in a + b + c):
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exact = ops.add(ops.mul(exact_cuts(a), exact_cuts(b)), exact_cuts(c))
        for outward in (False, True):
            assert ops.fma(a, b, c, outward=outward) == _rounded(exact, outward), (show(a), show(b), show(c))


# THE CLASS

def test_methods():
    a = MultiInterval(0, 10)
    assert a.minimum(MultiInterval(3, 5)) == MultiInterval(0, 5)
    assert a.maximum(7) == MultiInterval(7, 10)
    assert a.fma(2, Fraction(1, 2)) == MultiInterval(Fraction(1, 2), Fraction(41, 2))
    with pytest.raises(TypeError):
        a.minimum('3')
