"""
OutwardMultiInterval: the class whose float results round outward

* the type carries the rounding: every result of an OutwardMultiInterval is one, and so is every
  result of mixing one with a MultiInterval, on either side of the operator. a method keeps the
  receiver's class (v2-plan "methods on the receiver's class"), so `M.fma(O, ...)` is M's
* on exact operands it is the same as MultiInterval: nothing is rounded there, the functions'
  irrational values included (the tightest enclosure in both classes)
* an end that outward rounding moved is open (nothing attains it); an end that is a double already
  keeps its flag
* on floats, int, Fraction and mixed cut tuples, every rounding op sits between two sets computed
  without the class: the exact result (MultiInterval on the operands read as Fractions) and its
  tightest double-ended cover (each end not a double moved to the neighbouring double outward, and
  opened, by `directed` here, not the package's rounding). the first is soundness, the
  second is "the tightest such floats" with both flag rules. MultiInterval's result, rounded to
  nearest, lies inside the closure of the outward one: rounding to nearest picks one of the two
  doubles that outward rounding brackets the exact value with
* equality and hashing are structural across the classes, and pickle, copy and repr give back the
  class and each end's number type (a float end that came back an int would stop rounding)
* soundness on floats is fuzzed in tests/test_extreme_floats.py (the production descriptors) and
  pinned against 1788's tightest enclosures in tests/itf1788 (every interval-valued vector runs
  through this class)
"""
import copy
import math
import operator
import pickle
import sys
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import kernel
from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.fmt import format_cuts
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.strategies import probe_points
from tests.strategies import values

M, O = MultiInterval, OutwardMultiInterval

BINARY = [operator.add, operator.sub, operator.mul, operator.truediv, operator.mod, operator.floordiv,
          operator.or_, operator.and_, operator.xor]


def test_a_fraction_base_stays_exact():
    """`Fraction ** A` reaches `A.__rpow__` with the Fraction itself. on python 3.11 it did not:
    `Fraction.__pow__` answered any non-rational exponent with `float(a) ** b`, so the base was
    rounded before the library saw it and `Fraction(1, 3) ** O(2)` missed 1/9 (found by CI run
    36406179185; CPython 3.12 returns NotImplemented instead). hence python >= 3.12 (owner,
    2026-09-28); this is red on 3.11"""
    from fractions import Fraction
    from intervals.autodiff import Dual
    assert Fraction(1, 3) ** O(2) == O.parse('[1/9]')
    assert Fraction(1, 3) ** M(2) == M.parse('[1/9]')
    assert Fraction(1, 9) in (Fraction(1, 3) ** Dual.variable(O(2))).value


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


def test_an_infinite_corner_keeps_a_float_operand_float():
    """`2.5 / inf` is exact (0) either way; the float operand makes it 0.0, as in MultiInterval"""
    for cls in (M, O):
        result = cls(2.5) / math.inf
        assert result == cls(0.0) and isinstance(result.inf, float)


# PROPERTIES OVER FLOAT, EXACT AND MIXED CUT TUPLES

INF, MAX = math.inf, sys.float_info.max

# 0.1, 0.3 and 1.1 are not dyadic, so sums, products and quotients with them round
float_cut_tuples = cut_tuples(values=st.one_of(
    st.sampled_from([-INF, -2.0, -1.0, 0.0, 0.1, 0.3, 0.5, 1.1, 3.0, INF]),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
), max_pieces=3)
mixed_cut_tuples = cut_tuples(max_pieces=3)  # ints, Fractions and floats in one tuple
# non-empty: either operand empty empties the result, which has no end to round (40% of results before)
rounding_operands = st.one_of(float_cut_tuples, mixed_cut_tuples).filter(bool)
any_operands = st.one_of(float_cut_tuples, mixed_cut_tuples, exact_cut_tuples)
exact_scalars = st.one_of(st.integers(-20, 20), st.fractions(min_value=-20, max_value=20, max_denominator=6),
                          st.sampled_from([-INF, INF]))

# the rounding ops: an operator takes a scalar on either side, a method's receiver is a set
OPERATORS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'truediv': operator.truediv,
             'mod': operator.mod, 'floordiv': operator.floordiv, 'divmod': divmod}
METHODS = {
    'neg': lambda a, b, c, n: -a,
    'abs': lambda a, b, c, n: abs(a),
    'reciprocal': lambda a, b, c, n: a.reciprocal(),
    'pown': lambda a, b, c, n: a ** n,
    'round': lambda a, b, c, n: round(a, n % 3),
    'round_ties_away': lambda a, b, c, n: a.round_ties_away(n % 3),
    'fma': lambda a, b, c, n: a.fma(b, c),
    'cancel_minus': lambda a, b, c, n: a.cancel_minus(b),
    'cancel_plus': lambda a, b, c, n: a.cancel_plus(b),
}
CASES = sorted([*OPERATORS, *METHODS])


@st.composite
def case_operands(draw, name, sets, scalars):
    """(a, b, c, n): cut tuples or scalars, at least one of an operator's two a set, and an int n"""
    either = st.one_of(sets, sets, scalars)
    a = draw(either if name in OPERATORS else sets)
    b = draw(either)
    if not isinstance(a, tuple) and not isinstance(b, tuple):
        b = draw(sets)
    return a, b, draw(either), draw(st.integers(-3, 3))


def evaluate(name, cls, operands, exact=False):
    """the case on the operands as `cls` (scalars as they are), or read exactly: each float as its Fraction"""
    def wrap(x):
        if isinstance(x, tuple):
            return cls.from_cuts(exact_cuts(x) if exact else x)
        return exact_value(x) if exact else x
    a, b, c, n = operands
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if name in OPERATORS:
            result = OPERATORS[name](wrap(a), wrap(b))
        else:
            result = METHODS[name](wrap(a), wrap(b), wrap(c), n)
    return result if isinstance(result, tuple) else (result,)


def exact_value(v):
    """a finite float as the Fraction it denotes; int, Fraction and +-inf as they are"""
    return Fraction(v) if isinstance(v, float) and math.isfinite(v) else v


def exact_cuts(cuts):
    return kernel.normalize(kernel.piece(exact_value(lo), exact_value(hi), lc, hc)
                            for lo, lc, hi, hc in kernel.pieces(cuts))


def directed(v, direction):
    """the largest double <= v (direction -1) or the smallest >= v (1), for an exact finite v; past the
    doubles that is +-inf on the side the direction points to and +-max float on the other"""
    try:
        f = float(v)  # correctly rounded for int and Fraction
    except OverflowError:
        f = INF if v > 0 else -INF
    if math.isinf(f):
        return f if (f > 0) == (direction > 0) else math.copysign(MAX, f)
    if (Fraction(f) - v) * direction < 0:
        f = math.nextafter(f, direction * INF)
    return f + 0.0


def cover(cuts):
    """the smallest set with double or infinite ends holding cuts: an end that is not a double moves to
    its neighbouring double outward and opens, a double end keeps its flag"""
    out = []
    for lo, lc, hi, hc in kernel.pieces(cuts):
        # compared, never converted: an exact end can be past the doubles (`[-inf, -2.0) // 2e-313`)
        rlo = lo if lo in (-INF, INF) else directed(lo, -1)
        rhi = hi if hi in (-INF, INF) else directed(hi, 1)
        out.append(kernel.piece(rlo, rhi, lc and rlo == lo, hc and rhi == hi))
    return kernel.normalize(out)


def closure(cuts):
    return kernel.normalize(kernel.piece(lo, hi, True, True) for lo, _, hi, _ in kernel.pieces(cuts))


def has_finite_float(cuts):
    return any(isinstance(c.value, float) and math.isfinite(c.value) for c in cuts)


@pytest.mark.parametrize('name', CASES)
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_outward_holds_the_exact_result_tightly(name, data):
    """exact ⊆ outward ⊆ cover(exact): sound, and no looser than the tightest double-ended set, so a
    moved end is open and a double end of the exact result keeps its flag"""
    operands = data.draw(case_operands(name, rounding_operands, values), label='operands')
    for outward, exact in zip(evaluate(name, O, operands), evaluate(name, M, operands, exact=True)):
        assert type(outward) is O
        tightest = cover(exact.cuts)
        assert kernel.is_subset(exact.cuts, outward.cuts), ('unsound', format_cuts(exact.cuts), outward)
        assert kernel.is_subset(outward.cuts, tightest), ('loose', format_cuts(tightest), outward)


@pytest.mark.parametrize('name', CASES)
@settings(max_examples=25, deadline=None)
@given(data=st.data())
def test_nearest_lies_in_the_closure_of_outward(name, data):
    """MultiInterval rounds each end to nearest, one of the two doubles outward rounding brackets the
    exact end with, so its set is inside the closure of the outward one (its flags are not compared:
    to nearest they are conservative, not a promise)"""
    operands = data.draw(case_operands(name, rounding_operands, values), label='operands')
    for nearest, outward in zip(evaluate(name, M, operands), evaluate(name, O, operands)):
        assert type(nearest) is M
        assert kernel.is_subset(nearest.cuts, closure(outward.cuts)), (nearest, outward)


def test_a_negative_power_rounds_once():
    """to nearest a float `x ** -n` is python's `float ** int`, one rounding: it was `1 / x ** n`, rounded
    twice, an ulp below both of outward's doubles here (M14-breadth, 2026-10-02; the property above
    found it in 0.65% of its n < 0 draws)"""
    x = 5.155830884225402
    assert (M(x) ** -3).cuts == M(x ** -3).cuts
    assert kernel.is_subset((M(x) ** -3).cuts, closure((O(x) ** -3).cuts))


@pytest.mark.parametrize('name', CASES)
@settings(max_examples=30, deadline=None)
@given(data=st.data())
def test_exact_operands_give_the_same_set(name, data):
    """int, Fraction and +-inf operands, scalars on either side too: nothing is rounded, in either class"""
    operands = data.draw(case_operands(name, exact_cut_tuples, exact_scalars), label='operands')
    for outward, nearest in zip(evaluate(name, O, operands), evaluate(name, M, operands)):
        assert outward.cuts == nearest.cuts and outward == nearest and hash(outward) == hash(nearest)
        assert not has_finite_float(outward.cuts), outward


FUNCTIONS = {
    'sqrt': lambda a, b: a.sqrt(),
    'exp': lambda a, b: a.exp(),
    'log': lambda a, b: a.log(),
    'sin': lambda a, b: a.sin(),
    'atan': lambda a, b: a.atan(),
    'cbrt': lambda a, b: a.cbrt(),
    'hypot': lambda a, b: a.hypot(b),
    'atan2': lambda a, b: a.atan2(b),
    'pow': lambda a, b: a ** b,  # 1788's pow: an interval exponent
}


@pytest.mark.parametrize('name', sorted(FUNCTIONS))
@settings(max_examples=15, deadline=None)
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_exact_operands_give_the_same_set_in_the_functions(name, a, b):
    """an irrational value of an exact operand is its tightest float enclosure in both classes"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        outward = FUNCTIONS[name](O.from_cuts(a), O.from_cuts(b))
        nearest = FUNCTIONS[name](M.from_cuts(a), M.from_cuts(b))
    assert type(outward) is O and type(nearest) is M
    assert outward.cuts == nearest.cuts, (outward, nearest)


def flat(results):
    for r in results:
        yield from (r if isinstance(r, tuple) else (r,))


@settings(max_examples=20, deadline=None)
@given(a=any_operands, b=any_operands, s=values)
def test_the_class_is_closed(a, b, s):
    """every operator with an OutwardMultiInterval on either side gives one, and so does every method of
    one; a MultiInterval's method keeps its class with an outward argument"""
    x, y = O.from_cuts(a), M.from_cuts(b)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mixed = [op(left, right) for op in [*BINARY, divmod, operator.pow]
                 for left, right in ((x, y), (y, x), (x, x), (x, s), (s, x))]
        unary = [-x, +x, abs(x), ~x, x ** 2, x ** -1, round(x), round(x, 1), math.floor(x), math.ceil(x),
                 math.trunc(x), x[0:5], x[:], *x, *x.pieces]
        methods = [x.reciprocal(), x.floor(), x.ceil(), x.trunc(), x.round(), x.round_ties_away(1), x.sign(),
                   x.minimum(y), x.maximum(s), x.fma(y, s), x.cancel_minus(y), x.cancel_plus(s), x.union(y, s),
                   x.intersection(y), x.difference(s), x.symmetric_difference(y), x.complement(), x.hull,
                   x.closed_hull, x.interior, x.finite, x.positive, x.negative, x.expand(1), x.sqrt(),
                   x.exp(), x.log(2), x.sin(), x.atan(), x.hypot(y), x.atan2(s), x.rootn(3)]
        receiver = [y.minimum(x), y.maximum(x), y.fma(x, x), y.cancel_minus(x), y.cancel_plus(x), y.union(x),
                    y.intersection(x), y.difference(x), y.symmetric_difference(x), y.hypot(x), y.atan2(x)]
    for result in flat([*mixed, *unary, *methods]):
        assert type(result) is O, result
    for result in receiver:
        assert type(result) is M, result


def number_types(x):
    return [type(cut.value) for cut in x.cuts]


@settings(max_examples=60, deadline=None)
@given(a=any_operands)
@example(a=(Cut(2.0, Side.BELOW), Cut(2, Side.ABOVE)))  # a point whose cuts differ in type: repr wrote [2.0] (fuzz x10)
def test_pickle_copy_and_repr_round_trip(a):
    """the same class, set and number type at every end (a float end read back as an int would stop
    rounding, an int one read back as a float would start)"""
    x = O.from_cuts(a)
    for back in (pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x),
                 eval(repr(x), {'OutwardMultiInterval': O})):
        assert type(back) is O and back == x and back.cuts == x.cuts, (back, x)
        assert number_types(back) == number_types(x), (back, x)


def same_set(a, b):
    """decided on probe points, not on the cuts: membership is constant between consecutive ends"""
    return all(kernel.contains_point(a, p) == kernel.contains_point(b, p) for p in probe_points(a, b))


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_equality_and_hash_across_the_classes(data):
    """`==` is set equality whatever the class and the number types, and equal sets hash equal"""
    a = data.draw(any_operands, label='a')
    b = data.draw(st.one_of(st.just(a), st.just(exact_cuts(a)), any_operands), label='b')
    same = same_set(a, b)
    for x, y in ((O.from_cuts(a), M.from_cuts(b)), (M.from_cuts(a), O.from_cuts(b)),
                 (O.from_cuts(a), O.from_cuts(b))):
        assert (x == y) is same and (y == x) is same and (x != y) is not same, (x, y)
        if same:
            assert hash(x) == hash(y) and len({x, y}) == 1 and {x: 1}[y] == 1, (x, y)
