"""
OutwardMultiInterval: the class whose float results round outward

* the type carries the rounding: every result of an OutwardMultiInterval is one, and so is every
  result of mixing one with a MultiInterval, on either side of the operator
* on exact operands it is the same as MultiInterval: nothing is rounded there
* an end that outward rounding moved is open (nothing attains it); an end that is a double already
  keeps its flag
* soundness on floats is fuzzed in tests/test_extreme_floats.py (the production descriptors) and
  pinned against 1788's tightest enclosures in tests/itf1788 (every interval-valued vector runs
  through this class)
"""
import math
import operator
import pickle
import warnings

import pytest
from hypothesis import given
from hypothesis import settings

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals.fmt import format_cuts
from tests.strategies import exact_cut_tuples

M, O = MultiInterval, OutwardMultiInterval

BINARY = [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod, operator.floordiv,
          operator.or_, operator.and_, operator.xor]


@pytest.mark.parametrize('op', BINARY, ids=lambda op: op.__name__)
def test_mixed_operands_give_the_outward_class(op):
    a, b = M(0.5, 2.0), O(1.5, 3.0)
    for left, right in ((a, b), (b, a), (b, b), (b, 2.5), (2.5, b)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert type(op(left, right)) is O, (op, left, right)
    assert type(op(a, M(1.5, 3.0))) is M


def test_divmod_both_ways():
    a, b = M(7.5), O(2.0, 3.0)
    for q, r in (divmod(a, b), divmod(b, a), divmod(b, 2), divmod(7, b)):
        assert type(q) is O and type(r) is O


def test_unary_and_methods_keep_the_class():
    b = O(1.0, 2.0)
    for result in (-b, +b, abs(b), b ** 2, b ** -1, b.reciprocal(), b.floor(), b.sign(), round(b),
                   b.minimum(M(1.5)), b.fma(M(2.0), 1), b.sqrt(), b.hull, b | M(5)):
        assert type(result) is O, result


def test_repr_and_pickle():
    b = O(0.1) + 0.2
    assert repr(b) == "OutwardMultiInterval.parse('(0.3, 0.30000000000000004)')"
    assert eval(repr(b), {'OutwardMultiInterval': O}) == b
    assert type(pickle.loads(pickle.dumps(b))) is O


def test_equal_sets_are_equal_across_the_classes():
    assert O(1, 2) == M(1, 2) and hash(O(1, 2)) == hash(M(1, 2))


@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod,
                                operator.floordiv], ids=lambda op: op.__name__)
@settings(max_examples=60, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_exact_operands_give_the_same_set(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert op(O.from_cuts(a), O.from_cuts(b)).cuts == op(M.from_cuts(a), M.from_cuts(b)).cuts


@pytest.mark.parametrize('expr, expected', [
    (lambda: O(0.1) + 0.2, '(0.3, 0.30000000000000004)'),  # the exact sum lies strictly between
    (lambda: O(0.5) + 0.25, '[0.75]'),  # a double: kept, closed
    (lambda: O(0.5, 1.0) + 0.25, '[0.75, 1.25]'),
    (lambda: O(1.0) / 3.0, '(0.3333333333333333, 0.33333333333333337)'),
    (lambda: O(1.0, 2.0) / 3.0, '(0.3333333333333333, 0.6666666666666667)'),
    (lambda: O(1e308) * 10.0, '(1.7976931348623157e+308, inf)'),  # past the float range: open at inf
    (lambda: O(1e308) // 1e-308, '(1.7976931348623157e+308, inf)'),
    (lambda: O(2.0) ** -1, '[0.5]'),
    (lambda: O(3.0) ** -1, '(0.3333333333333333, 0.33333333333333337)'),
])
def test_moved_ends_are_open(expr, expected):
    assert format_cuts(expr().cuts) == expected


def test_nearest_is_the_default():
    assert M(0.1) + 0.2 == M(0.30000000000000004)
    assert M(1e308) // 1e-308 == M(math.inf)
