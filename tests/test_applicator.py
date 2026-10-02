"""
applicator mechanics: splitting, corners, D2 limits, closure, warnings, number types, rounding hook,
and the arithmetic dunders on the class. worked-example tables and the laws live in
test_ops_examples.py and test_ops_properties.py

the tables pin each mechanic on chosen operands; the hypothesis tests check the same mechanics on random
ones, each against the design rather than the code: splitting at random points is a partition of the
input with its flags kept; one box's image (`evaluate_box`) is exactly the set of values its defined
pairs attain, by the independent oracle in `tests.oracles` (sampled values inside it, each end closed iff
attained, points just inside each end attained); the monotone fast path matches every corner on floats,
under a rounding hook and for min/max; a recording hook sees every finite float corner and nothing else;
exact operands give exact ends and float operands float ends (D3); each warning fires once per call, iff
an operand is empty or some box has no defined pair (D7); and every result is canonical
"""
import math
import sys
import warnings
from fractions import Fraction
from itertools import product

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import intervals
from intervals import EMPTY
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import kernel
from intervals import ops
from intervals.applicator import OpDescriptor
from intervals.applicator import apply_binary
from intervals.applicator import apply_unary
from intervals.applicator import evaluate_box
from intervals.applicator import split_pieces
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.fmt import format_cuts
from intervals.fmt import parse
from tests.oracles import BINARY
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples

inf = math.inf
P = MultiInterval.parse


def run(op, *texts):
    return format_cuts(op(*map(parse, texts)))


# RANDOMIZED: strategies and helpers for the hypothesis tests below

# the special points of the arithmetic: poles, indeterminate corners, flat spots
SPECIAL = (-inf, -1, 0, 1, inf)
exact_ends = st.one_of(st.sampled_from(SPECIAL), st.sampled_from(SPECIAL), st.integers(-6, 6),
                       st.fractions(min_value=-6, max_value=6, max_denominator=4))


@st.composite
def exact_pieces(draw, one_in=4):
    """a non-empty `(lo, lo_closed, hi, hi_closed)` with exact ends; a degenerate special point one time in `one_in`"""
    if draw(st.integers(1, one_in)) == 1:
        v = draw(st.sampled_from(SPECIAL))
        return v, True, v, True
    lo, hi = sorted((draw(exact_ends), draw(exact_ends)))
    if lo == hi:
        return lo, True, hi, True
    return lo, draw(st.booleans()), hi, draw(st.booleans())


def as_cuts(*ps):
    return kernel.normalize(kernel.piece(lo, hi, lc, hc) for lo, lc, hi, hc in ps)


# one to three pieces, half of them a lone [0], [inf] or [-inf]: the shapes D7 is about
special_operands = st.lists(exact_pieces(one_in=2), min_size=1, max_size=3).map(lambda ps: as_cuts(*ps))

# every finite end a float, the infinities as points
float_ends = st.one_of(st.sampled_from([-inf, -1.0, 0.0, 0.5, 2.0, inf]), st.floats(-20, 20, allow_nan=False))
float_cut_tuples = cut_tuples(values=float_ends, max_pieces=3)

# (oracle name, descriptor): the ops the pointwise oracle knows, each evaluated by the applicator
ORACLE_OPS = [('add', ops.ADD), ('sub', ops.SUB), ('mul', ops.MUL), ('div', ops.DIV),
              ('reciprocal', ops.RECIPROCAL), ('abs', ops.ABS), ('neg', ops.NEG), ('pos', ops.POS)]
MINMAX_OPS = [('min', ops.MIN), ('max', ops.MAX)]
OUTWARD_OPS = [(f'outward-{name}', desc) for name, desc in ops.OUTWARD.items()]
ALL_OPS = ORACLE_OPS + MINMAX_OPS + OUTWARD_OPS
OP_IDS = [name for name, _ in ALL_OPS]


def arity(desc):
    return 1 if desc.name in ('reciprocal', 'abs', 'neg', 'pos') else 2


def apply(desc, operands):
    return apply_unary(desc, *operands) if arity(desc) == 1 else apply_binary(desc, *operands)


def values_at(name, x, y, a, b):
    """the values one pair gives, by the oracle; `a`, `b` are the operands' sets (for a pole's direction)"""
    if name == 'min':
        return [min(x, y)]
    if name == 'max':
        return [max(x, y)]
    if name in BINARY:
        return pointwise(name, x, y, a, b)
    return pointwise(name, x, a=a)


def one_side(p, side):
    """the half of p on the given side of 0 (-1 below, +1 above), the 0 included; p itself if 0 is not inside"""
    lo, lc, hi, hc = p
    if lo < 0 < hi:
        return (lo, lc, 0, True) if side < 0 else (0, True, hi, hc)
    return p


def halves(p):
    """the sign-pure pieces the design splits p into: both halves if 0 is strictly inside"""
    lo, lc, hi, hc = p
    return [(lo, lc, 0, True), (0, True, hi, hc)] if lo < 0 < hi else [p]


def representatives(p):
    """
    p's closed ends and two finite interior points, one of them nonzero. a box has a defined pair iff a pair
    of these does: in the pointwise table a pair with no value (inf - inf, 0 * inf, inf / inf, 0 / 0) has
    every coordinate at an end, except x over a lone [0], where no pair has one; a finite nonzero interior
    point has a value with any partner
    """
    lo, lc, hi, hc = p
    out = [v for v, closed in ((lo, lc), (hi, hc)) if closed]
    if lo < hi:
        if lo == -inf and hi == inf:
            out += [-1, 1]
        elif lo == -inf:
            out += [hi - 1, hi - 2]
        elif hi == inf:
            out += [lo + 1, lo + 2]
        else:
            out += [lo + Fraction(hi - lo) / 3, lo + 2 * Fraction(hi - lo) / 3]
    return out


def has_value(name, box):
    sets = [as_cuts(p) for p in box]
    b = sets[1] if len(box) == 2 else None
    return any(values_at(name, x, y, sets[0], b)
               for x, y in product(representatives(box[0]), representatives(box[1]) if b else [None]))


def inside(lo, hi, end):
    """an exact point strictly inside (lo, hi), within 1e-9 of the low end (end -1) or the high end (+1)"""
    eps, far = Fraction(1, 10 ** 9), 10 ** 12
    if end < 0:
        return min(hi, 0) - far if lo == -inf else lo + (min(eps, Fraction(hi - lo) / 2) if hi != inf else eps)
    return max(lo, 0) + far if hi == inf else hi - (min(eps, Fraction(hi - lo) / 2) if lo != -inf else eps)


def holds(p, v):
    lo, lc, hi, hc = p
    return lo < v < hi or (v == lo and lc) or (v == hi and hc)


# SPLITTING

@pytest.mark.parametrize('pieces, expected', [
    ([(-1, True, 1, False)], [(-1, True, 0, True), (0, True, 1, False)]),
    ([(-1, False, 0, False)], [(-1, False, 0, False)]),  # touching zero: no split
    ([(0, True, 2, True)], [(0, True, 2, True)]),
    ([(0, True, 0, True)], [(0, True, 0, True)]),  # a degenerate [0] stays whole
    ([(-inf, True, inf, True)], [(-inf, True, 0, True), (0, True, inf, True)]),
    ([(-2, True, -1, True), (1, False, 2, True)], [(-2, True, -1, True), (1, False, 2, True)]),
])
def test_split_at_zero(pieces, expected):
    assert split_pieces(pieces, (0,)) == expected


def test_split_at_several_points():
    assert split_pieces([(0, False, 3, False)], (1, 2)) == [
        (0, False, 1, True), (1, True, 2, True), (2, True, 3, False)]


split_points = st.lists(st.one_of(st.sampled_from([-2, -1, 0, 0.0, Fraction(1, 2), 1, 1.0, 3]),
                                  st.floats(-20, 20, allow_nan=False, allow_infinity=False)), max_size=4)


@settings(max_examples=200, deadline=None)
@given(cut_tuples(), split_points)
def test_split_is_a_partition(cuts, points):
    """
    each piece becomes, in order, a chain of pieces covering it exactly: one more than the distinct points
    strictly inside it, consecutive ones sharing just that point (closed in both), none with a point
    strictly inside, the piece's own ends and flags on the first and last
    """
    original = list(kernel.pieces(cuts))
    out = split_pieces(original, tuple(points))
    assert as_cuts(*out) == cuts
    rest = list(out)
    for lo, lc, hi, hc in original:
        n = len({s for s in points if lo < s < hi})
        chain, rest = rest[:n + 1], rest[n + 1:]
        assert len(chain) == n + 1
        assert chain[0][:2] == (lo, lc) and type(chain[0][0]) is type(lo)
        assert chain[-1][2:] == (hi, hc) and type(chain[-1][2]) is type(hi)
        for (_, _, end, end_closed), (start, start_closed, _, _) in zip(chain, chain[1:]):
            assert end == start and end_closed and start_closed and end in points
        for p in chain:
            assert p[0] < p[2] or (p[0] == p[2] and p[1] and p[3]), p
            assert not any(p[0] < s < p[2] for s in points), p
    assert rest == []


def test_mul_splits_both_operands():
    # without the split, [-1, 1] * [inf] would hull to the entire line
    assert run(ops.mul, '[-1, 1]', '[inf]') == '{ [-inf] , [inf] }'
    assert run(ops.mul, '[inf]', '[-1, 1]') == '{ [-inf] , [inf] }'
    assert run(ops.div, '[-1, 1]', '[0, 1]') == '[-inf, inf]'


# CORNERS AND D2 LIMITS

@pytest.mark.parametrize('desc, box, expected', [
    # a defined box: min and max over the corners
    (ops.MUL, ((1, True, 2, True), (-3, True, -1, False)), (-6, True, -1, False)),
    # (inf, 0) has no value; the non-degenerate factor's edge gives the limit (D2)
    (ops.MUL, ((-inf, True, -1, True), (0, True, 0, True)), (0, True, 0, True)),
    (ops.MUL, ((-inf, True, -inf, True), (0, True, 1, True)), (-inf, True, -inf, True)),
    (ops.MUL, ((-inf, True, -1, True), (0, True, 1, True)), (-inf, True, 0, True)),
    (ops.SUB, ((1, True, inf, True), (1, True, inf, True)), (-inf, True, inf, True)),
    (ops.SUB, ((inf, True, inf, True), (1, True, inf, True)), (inf, True, inf, True)),
    (ops.DIV, ((1, True, inf, True), (1, True, inf, True)), (0, True, inf, True)),
    (ops.DIV, ((0, True, 1, True), (0, True, 1, True)), (0, True, inf, True)),
    # a pole takes its sign from the side of zero the divisor's piece lies on
    (ops.DIV, ((1, True, 2, True), (-1, True, 0, True)), (-inf, True, -1, True)),
    (ops.DIV, ((-2, True, -1, True), (-1, True, 0, True)), (1, True, inf, True)),
    (ops.RECIPROCAL, ((0, True, 1, False),), (1, False, inf, True)),
    (ops.RECIPROCAL, ((-1, True, 0, False),), (-inf, False, -1, True)),
])
def test_evaluate_box(desc, box, expected):
    assert evaluate_box(desc, box) == expected


@pytest.mark.parametrize('desc, box', [
    (ops.MUL, ((0, True, 0, True), (inf, True, inf, True))),
    (ops.SUB, ((inf, True, inf, True), (inf, True, inf, True))),
    (ops.ADD, ((-inf, True, -inf, True), (inf, True, inf, True))),
    (ops.DIV, ((0, True, 0, True), (0, True, 0, True))),
    (ops.DIV, ((inf, True, inf, True), (-inf, True, -inf, True))),
    (ops.DIV, ((1, True, 2, True), (0, True, 0, True))),  # a pole with no side, along a whole edge
    (ops.RECIPROCAL, ((0, True, 0, True),)),
])
def test_a_box_with_no_value_anywhere_is_none(desc, box):
    assert evaluate_box(desc, box) is None


@pytest.mark.parametrize('name, desc', ORACLE_OPS, ids=[name for name, _ in ORACLE_OPS])
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_evaluate_box_is_the_attained_set(name, desc, data, rng):
    """
    one box, its pieces on one side of every split point as the applicator hands them over: the image is
    exactly the set of values its defined pairs attain (exact operands), or None if no pair has a value
    """
    assert desc.split_points in ((), (0,))
    box = tuple(one_side(data.draw(exact_pieces()), data.draw(st.sampled_from((-1, 1)))) if desc.split_points
                else data.draw(exact_pieces()) for _ in range(arity(desc)))
    a, b = as_cuts(box[0]), (as_cuts(box[1]) if len(box) == 2 else None)
    result = evaluate_box(desc, box)
    xs = sample(a, 6, rng)
    found = [v for x, y in product(xs, sample(b, 6, rng) if b else [None]) for v in values_at(name, x, y, a, b)]
    if result is None:
        assert not found and not has_value(name, box), box
        return
    lo, lo_closed, hi, hi_closed = result
    assert all(holds(result, v) for v in found), (box, result, [v for v in found if not holds(result, v)])
    assert attained(name, lo, a, b) == lo_closed, (box, result)
    assert attained(name, hi, a, b) == hi_closed, (box, result)
    if lo < hi:
        # the image of a box is connected, so a wrong end shows as an unattained point just inside it
        for end in (-1, 1):
            assert attained(name, inside(lo, hi, end), a, b), (box, result, end)
    else:
        assert lo_closed and hi_closed


@settings(max_examples=150, deadline=None)
@given(exact_cut_tuples, exact_cut_tuples)
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_monotone_fast_path_matches_every_corner(a, b):
    for desc in (ops.ADD, ops.SUB):
        assert apply_binary(desc, a, b) == apply_binary(desc._replace(monotone=None), a, b)


def types(cuts):
    return [type(cut.value) for cut in cuts]


@settings(max_examples=60, deadline=None)
@given(float_cut_tuples, float_cut_tuples)
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_monotone_fast_path_matches_every_corner_on_floats(a, b):
    # float corners, the outward hooks and min/max's own attainment: the same set, in the same types
    for desc in (ops.ADD, ops.SUB, ops.MIN, ops.MAX, ops.OUTWARD['add'], ops.OUTWARD['sub']):
        fast, full = apply_binary(desc, a, b), apply_binary(desc._replace(monotone=None), a, b)
        assert fast == full and types(fast) == types(full), desc.name


def test_div_is_evaluated_at_the_corners_not_through_the_reciprocal():
    calls = []

    def spy(x, y):
        calls.append((x, y))
        return ops.DIV.fn(x, y)

    assert apply_binary(ops.DIV._replace(fn=spy), parse('[1, 2]'), parse('[3, 4]')) == parse('[1/4, 2/3]')
    assert (1, 4) in calls and (2, 3) in calls


# CLOSURE

@pytest.mark.parametrize('op, a, b, expected', [
    # finite endpoint of an injective op: the corner-flag rule
    (ops.add, '[1, 2)', '(3, 4]', '(4, 6)'),
    (ops.mul, '[1, 2)', '[3, 4]', '[3, 8)'),
    (ops.div, '(1, 2]', '[3, 4)', '(1/4, 2/3]'),
    # a flat spot at 0: 0 * y is 0 along the whole edge, attained with both y ends open
    (ops.mul, '[0, 1]', '(2, 3)', '[0, 3)'),
    (ops.mul, '(0, 1]', '(2, 3)', '(0, 3)'),
    (ops.div, '[0]', '(2, 3)', '[0]'),
    (ops.div, '(1, 2)', '[inf]', '[0]'),  # x / inf is 0 for every finite x
    (ops.div, '(1, 2)', '(3, inf)', '(0, 2/3)'),
    # infinite endpoints: attained through a partner, never by the corner flags alone
    (ops.add, '[inf]', '(1, 2)', '[inf]'),
    (ops.add, '[1, inf]', '[0, 1)', '[1, inf]'),
    (ops.add, '(1, inf]', '[0]', '(1, inf]'),
    (ops.add, '(1, inf)', '[0]', '(1, inf)'),
    (ops.mul, '[inf]', '(1, 2)', '[inf]'),
    (ops.mul, '(0, 1)', '[inf]', '[inf]'),
    (ops.mul, '[0, 1)', '(0, inf)', '[0, inf)'),
    (ops.div, '(1, 2)', '[0, 1]', '(1, inf]'),  # the pole at the closed 0 attains inf
    (ops.div, '(1, 2)', '(0, 1]', '(1, inf)'),
    (ops.div, '[1, inf)', '[1, inf)', '(0, inf)'),
])
def test_closure(op, a, b, expected):
    assert run(op, a, b) == expected


@pytest.mark.parametrize('op, a, expected', [
    (ops.reciprocal, '[1, inf)', '(0, 1]'),
    (ops.reciprocal, '(-1, 0)', '(-inf, -1)'),
    (ops.reciprocal, '[-1, 0]', '[-inf, -1]'),
    (ops.absolute, '(-2, 1]', '[0, 2)'),
    (ops.absolute, '[-inf, -1)', '(1, inf]'),
    (ops.neg, '[1, inf)', '(-inf, -1]'),
    (ops.pos, '(1, 2] | [3]', '{ (1, 2] , [3] }'),
])
def test_unary_closure(op, a, expected):
    assert run(op, a) == expected


def test_attained_override_is_used():
    always_open = ops.ADD._replace(attained=lambda v, box: False)
    assert format_cuts(apply_binary(always_open, parse('[1, 2]'), parse('[3]'))) == '(4, 5)'


# WARNINGS

def test_empty_operand_warns_once():
    for op, args in [(ops.add, ((), parse('[1, 2]'))), (ops.div, (parse('[1, 2]'), ())),
                     (ops.neg, ((),)), (ops.reciprocal, ((),)), (ops.power, ((), -2))]:
        with pytest.warns(EmptySetPropagationWarning) as record:
            assert op(*args) == ()
        assert len(record) == 1


@pytest.mark.parametrize('op, args', [
    (ops.sub, ('[inf]', '[inf]')),
    (ops.mul, ('[0]', '[inf]')),
    (ops.div, ('[0]', '[0]')),
    (ops.div, ('[1, 2]', '[0]')),
    (ops.reciprocal, ('[0]',)),
])
def test_indeterminate_box_warns_and_gives_empty(op, args):
    with pytest.warns(IndeterminateResultWarning) as record:
        assert op(*map(parse, args)) == ()
    assert len(record) == 1


def test_indeterminate_warns_once_per_call_even_with_a_non_empty_result():
    # two indeterminate boxes ([0] x [inf] and [0] x [-inf]) and two defined ones
    with pytest.warns(IndeterminateResultWarning) as record:
        assert run(ops.mul, '[0] | [1]', '[-inf] | [inf]') == '{ [-inf] , [inf] }'
    assert len(record) == 1
    with pytest.warns(IndeterminateResultWarning):
        assert run(ops.reciprocal, '[0] | [1, 2]') == '[1/2, 1]'


@pytest.mark.parametrize('op, a, b', [
    (ops.mul, '[-inf, -1]', '[0]'),
    (ops.mul, '[-inf]', '[0, 1]'),
    (ops.sub, '[1, inf]', '[1, inf]'),
    (ops.sub, '[inf]', '[1, inf]'),
    (ops.div, '[1, inf]', '[1, inf]'),
    (ops.div, '[0, 1]', '[0, 1]'),
    (ops.add, '[-inf, 5]', '[inf]'),
])
def test_indeterminate_corner_of_a_larger_box_does_not_warn(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert op(parse(a), parse(b))


def test_warnings_point_at_the_caller():
    with pytest.warns(IndeterminateResultWarning) as record:
        _ = 1 / P('[0]')
    assert record[0].filename == __file__
    with pytest.warns(EmptySetPropagationWarning) as record:
        ops.add((), parse('[1]'))
    assert record[0].filename == __file__


@pytest.mark.parametrize('name, desc', ORACLE_OPS + MINMAX_OPS, ids=[name for name, _ in ORACLE_OPS + MINMAX_OPS])
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_each_warning_fires_once_exactly_when_due(name, desc, data):
    """
    an empty operand: the empty set and one EmptySetPropagationWarning, nothing else. otherwise one
    IndeterminateResultWarning iff some box (a piece of each operand, split at the op's points) has no
    defined pair by the table, and the empty set iff every box has none (D7)
    """
    operands = [data.draw(special_operands, label=f'operand {i}') for i in range(arity(desc))]
    if data.draw(st.integers(0, 4), label='empty one') == 0:
        operands[data.draw(st.integers(0, len(operands) - 1), label='which')] = kernel.EMPTY
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = apply(desc, operands)
    raised = [w.category for w in record]
    if not all(operands):
        assert result == kernel.EMPTY and raised == [EmptySetPropagationWarning]
        return
    split = [[h for p in kernel.pieces(cuts) for h in (halves(p) if desc.split_points else [p])] for cuts in operands]
    dead = [box for box in product(*split) if not has_value(name, box)]
    assert raised == ([IndeterminateResultWarning] if dead else []), (operands, dead)
    assert (result == kernel.EMPTY) == (len(dead) == len(list(product(*split)))), (operands, dead, result)


# NUMBER TYPES (D3)


@pytest.mark.parametrize('op, a, b, expected', [
    (ops.add, '[1, 2]', '[3]', [int, int]),
    (ops.sub, '[1, 2]', '[3]', [int, int]),
    (ops.mul, '[1, 2]', '[3]', [int, int]),
    (ops.div, '[6]', '[3]', [int, int]),  # an integral Fraction comes back as int
    (ops.div, '[1]', '[3]', [Fraction, Fraction]),
    (ops.div, '[1/2, 1]', '[3]', [Fraction, Fraction]),
    (ops.add, '[1/2]', '[1/2]', [int, int]),
    (ops.div, '[1]', '[inf]', [int, int]),  # not python's Fraction(1) / inf == 0.0
    (ops.div, '[1/3]', '[-inf, -1]', [Fraction, int]),
    (ops.mul, '[2]', '[1, inf]', [int, float]),  # the inf is exact; the 2 stays int
    (ops.add, '[1.5]', '[1]', [float, float]),
    (ops.div, '[1.5]', '[inf]', [float, float]),  # a finite float operand fed it
    (ops.mul, '[0]', '[2.5, inf]', [int, int]),  # the 0 from the edge limit is exact
])
def test_number_types(op, a, b, expected):
    assert types(op(parse(a), parse(b))) == expected


def test_equal_candidates_prefer_the_exact_value():
    # 0 * 2.5 is 0.0 and 0 * 3 is 0: the exact candidate wins the tie
    assert types(ops.mul(parse('[0]'), parse('[2.5, 3]'))) == [int, int]


def test_pow_types():
    assert types(ops.power(parse('[2, 3]'), 2)) == [int, int]
    assert types(ops.power(parse('[1/2]'), 3)) == [Fraction, Fraction]
    assert types(ops.power(parse('[2]'), -2)) == [Fraction, Fraction]
    assert types(ops.power(parse('[2.0]'), 2)) == [float, float]
    assert ops.power(parse('[1e200]'), 2) == parse('[inf]')  # float ** int overflow, as float * float


def finite_ends(cuts):
    return [cut.value for cut in cuts if not math.isinf(cut.value)]


@pytest.mark.parametrize('name, desc', ALL_OPS, ids=OP_IDS)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_number_types_follow_the_operands(name, desc, data):
    """
    D3: exact operands (int, Fraction, +-inf) give exact finite ends, an integral one as int, and are
    never rounded; operands whose finite ends are all floats give float ends, but for a 0, which an exact
    corner may give: +-inf (`1 / [2.0, inf]` is `[0, 0.5]`), the split point (`abs((-1.0, 1.0))` is
    `[0, 1.0)`), or a D2 limit evaluated exactly (outward `[0.0, 1.0] / [0.0, 1.0]` is `[0, inf]`, to
    nearest `[0.0, inf]`; reported 2026-10-02, M14-breadth)
    """
    exact = [data.draw(exact_cut_tuples, label=f'exact {i}') for i in range(arity(desc))]
    for v in finite_ends(apply(desc, exact)):
        assert type(v) is int or (type(v) is Fraction and v.denominator != 1), (exact, v)
    floats = [data.draw(float_cut_tuples, label=f'float {i}') for i in range(arity(desc))]
    for v in finite_ends(apply(desc, floats)):
        assert type(v) is float or v == 0, (floats, v)


# ROUNDING HOOK

def test_rounding_hook_only_on_finite_float_corners():
    calls = []

    def down(x, y):
        calls.append((x, y))
        return math.nextafter(x + y, -inf)

    def up(x, y):
        calls.append((x, y))
        return math.nextafter(x + y, inf)

    outward = ops.ADD._replace(rounded=(down, up))

    # exact operands: never rounded, flags from attainment
    assert apply_binary(outward, parse('[1, 2]'), parse('[1/2]')) == parse('[3/2, 5/2]')
    assert calls == []
    # infinite corners are exact too
    assert apply_binary(outward, parse('[1, inf]'), parse('[inf]')) == parse('[inf]')
    assert calls == []

    # finite float corners go down at the bottom and up at the top; a rounded endpoint is attained
    # by nothing, so it comes out open
    result = apply_binary(outward, parse('[1.5, 2.5]'), parse('[1]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(2.5, -inf), math.nextafter(3.5, inf),
                                                    False, False)])
    assert set(calls) == {(1.5, 1), (2.5, 1)}

    # a float corner at the bottom, an infinite one at the top: only the bottom is rounded
    calls.clear()
    result = apply_binary(outward, parse('[1.5, inf]'), parse('[1]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(2.5, -inf), inf, False, True)])
    assert calls == [(1.5, 1), (1.5, 1)]


def test_rounding_hook_on_a_mixed_corner_beyond_float_range():
    # fn runs before the hook; neither may raise on an exact operand that has no float
    outward = ops.ADD._replace(rounded=(lambda x, y: math.nextafter(ops._add(x, y), -inf),
                                        lambda x, y: math.nextafter(ops._add(x, y), inf)))
    result = apply_binary(outward, MultiInterval(10 ** 400).cuts, parse('[0.5]'))
    assert result == kernel.normalize([kernel.piece(sys.float_info.max, inf, False, True)])


def test_rounding_hook_unary():
    outward = ops.NEG._replace(rounded=(lambda x: math.nextafter(-x, -inf), lambda x: math.nextafter(-x, inf)))
    result = apply_unary(outward, parse('[1, 2.0]'))
    assert result == kernel.normalize([kernel.piece(math.nextafter(-2.0, -inf), -1, False, True)])


def typed(args):
    """a corner with its types: as a set member, 2 and 2.0 are different corners (one goes through the hook)"""
    return tuple((type(x), x) for x in args)


def corners(desc, operands):
    """every corner of every box: one end of one sign-pure piece of each operand"""
    ends = [[v for p in kernel.pieces(cuts) for h in (halves(p) if desc.split_points else [p]) for v in (h[0], h[2])]
            for cuts in operands]
    return list(product(*ends))


mixed_cut_tuples = st.one_of(float_cut_tuples, cut_tuples(max_pieces=3))


@pytest.mark.parametrize('name, desc', ORACLE_OPS + MINMAX_OPS, ids=[name for name, _ in ORACLE_OPS + MINMAX_OPS])
@settings(max_examples=40, deadline=None)
@given(data=st.data())
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_rounding_hook_sees_every_finite_float_corner_and_nothing_else(name, desc, data):
    """
    a recording hook: each call is a corner of a box with a value, every coordinate finite and one a
    float; down and up see the same corners; and without the monotone fast path every such corner is seen
    """
    seen = {-1: [], 1: []}

    def hook(direction):
        def rounded(*args):
            seen[direction].append(args)
            return desc.fn(*args)
        return rounded

    operands = [data.draw(mixed_cut_tuples, label=f'operand {i}') for i in range(arity(desc))]
    apply(desc._replace(rounded=(hook(-1), hook(1))), operands)
    assert sorted(map(repr, seen[-1])) == sorted(map(repr, seen[1]))
    due = {typed(c) for c in corners(desc, operands) if all(operands) and desc.fn(*c) is not None
           and all(not math.isinf(x) for x in c) and any(isinstance(x, float) for x in c)}
    for args in seen[-1]:
        assert all(not math.isinf(x) for x in args) and any(isinstance(x, float) for x in args), args
        assert typed(args) in due, (args, operands)
    if desc.monotone is None:
        assert {typed(args) for args in seen[-1]} == due, (due - {typed(args) for args in seen[-1]}, operands)


@pytest.mark.parametrize('name, desc', ALL_OPS, ids=OP_IDS)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
@pytest.mark.filterwarnings('ignore::intervals.errors.IndeterminateResultWarning')
@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')
def test_result_is_canonical(name, desc, data):
    # the union of the boxes, normalized: strictly increasing cuts, so no empty, overlapping or touching pieces
    operands = [data.draw(st.one_of(mixed_cut_tuples, special_operands), label=f'operand {i}')
                for i in range(arity(desc))]
    assert kernel.is_valid(apply(desc, operands))


# THE CLASS

def test_dunders_bind_the_ops():
    a, b = P('[1, 2]'), P('[3, 4)')
    assert a + b == P('[4, 6)')
    assert a - b == P('(-3, -1]')
    assert a * b == P('[3, 8)')
    assert a / b == P('(1/4, 2/3]')
    assert -a == P('[-2, -1]')
    assert +b == b
    assert abs(P('[-3, 1)')) == P('[0, 3]')
    assert a ** 2 == P('[1, 4]')
    assert a ** -1 == P('[1/2, 1]')
    assert a.reciprocal() == P('[1/2, 1]')


def test_scalars_coerce_on_both_sides():
    a = P('[1, 2]')
    assert a + 1 == 1 + a == P('[2, 3]')
    assert a - 1 == P('[0, 1]')
    assert 1 - a == P('[-1, 0]')
    assert 2 * a == a * 2 == P('[2, 4]')
    assert a / 2 == P('[1/2, 1]')
    assert 2 / a == P('[1, 2]')
    assert Fraction(1, 2) * a == P('[1/2, 1]')
    assert a * 0.5 == P('[0.5, 1.0]')


def test_numpy_scalars_run_the_reflected_dunders():
    np = pytest.importorskip('numpy')
    a = P('[1, 2]')
    assert np.float64(0.5) * a == P('[0.5, 1.0]')
    assert np.int64(3) - a == P('[1, 2]')
    assert np.float64(1) / a == P('[0.5, 1.0]')
    assert (np.float64(0.5) < a) == (0.5 < a)
    assert a ** np.float64(2) == P('[1, 4]')  # an integral value: pown (D11)
    assert a ** np.float64(0.5) == P('[1.0, 2.0]').sqrt()  # any other: 1788's pow, rounded as a float
    for thunk in (lambda: a + np.bool_(True), lambda: np.bool_(True) + a):
        with pytest.raises(TypeError):  # bool is refused, as in _coerce
            thunk()


def test_minus_is_arithmetic_not_set_difference():
    assert P('[0, 3]') - P('[1, 2]') == P('[-2, 2]')
    assert P('[0, 3]').difference(P('[1, 2]')) == P('[0, 1) | (2, 3]')


@pytest.mark.parametrize('exponent', [True, '2', None, [2]])
def test_pow_refuses_non_numbers(exponent):
    with pytest.raises(TypeError):
        _ = P('[1, 2]') ** exponent


@pytest.mark.parametrize('exponent, expected', [
    # D11: a number with an integral value is pown, over every base, as python's numbers do
    (2, '[0, 9]'), (2.0, '[0, 9]'), (Fraction(4, 2), '[0, 9]'), (-1, '{ [-inf, -1/3] , [1, inf] }'),
    # a non-integral number, or any MultiInterval, is 1788's pow: the negative bases are dropped
    (P('[2]'), '[0, 1]'), (0.5, '[0.0, 1.0]'), (Fraction(1, 2), '[0, 1]'),
    (P('[1/2, 2]'), '[0, 1]'), (math.inf, '[0]'),
])
def test_pow_dispatch(exponent, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        warnings.simplefilter('ignore', IndeterminateResultWarning)
        assert str(P('[-3, 1]') ** exponent) == expected


def test_pow_zero_and_three_argument_pow():
    assert P('[-inf, 0)') ** 0 == P('[1]')
    with pytest.raises(TypeError):
        pow(P('[1, 2]'), 2, 3)  # dropped (D11): not 1788, and v1 had it on integers only
    with pytest.raises(TypeError):
        pow(P('[1, 2]'), P('[2]'), 3)
    with pytest.raises(TypeError):
        ops.power(parse('[1]'), True)


@pytest.mark.parametrize('other', ['1', None, [1], (1, 2)])
def test_non_numbers_are_refused(other):
    a = P('[1, 2]')
    for op in (lambda: a + other, lambda: other + a, lambda: a - other, lambda: other - a,
               lambda: a * other, lambda: other * a, lambda: a / other, lambda: other / a):
        with pytest.raises(TypeError):
            op()


def test_rpow_is_pow_of_a_point():
    # M13d (D11): `b ** A` for a real b is `MultiInterval(b) ** A`, so 1788's pow
    assert 2 ** P('[1, 3]') == P('[2, 8]')
    assert Fraction(1, 4) ** P('[1/2]') == P('[1/2]')
    assert 4.0 ** P('[1/2]') == P('[2.0]')
    assert type(2 ** OutwardMultiInterval(1.0, 3.0)) is OutwardMultiInterval
    # a subclass's reflected method runs first, so mixing keeps the outward class on either side
    assert type(P('[2]') ** OutwardMultiInterval(0.5)) is OutwardMultiInterval
    assert type(OutwardMultiInterval(2.0) ** P('[1/2]')) is OutwardMultiInterval
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DomainClippedWarning)
        assert (-2) ** P('[1, 2]') == EMPTY  # a negative base is outside pow's domain
    for base in (True, '2', None):
        with pytest.raises(TypeError):
            _ = base ** P('[1, 2]')


def test_package_exports_unchanged():
    # the ops, the applicator, modulo, steps and functions stay in their submodules; M7 adds only
    # HullWarning, M12 only OutwardMultiInterval, M13h only the reductions (point functions, exported
    # as M13e plans for the reverse ops)
    assert set(intervals.__all__) == {
        'MultiInterval', 'OutwardMultiInterval', 'EMPTY', 'REALS', 'Size', 'Builder', 'TruthSet', 'Allen',
        'Cut', 'Side', 'IntervalWarning', 'EmptySetPropagationWarning', 'DomainClippedWarning',
        'IndeterminateResultWarning', 'HullWarning', 'sum_', 'sum_abs', 'sum_sqr', 'dot',
        # M13e: the reverse ops (intervals/reverse.py)
        'sqr_rev', 'abs_rev', 'pown_rev', 'cosh_rev',
        'mul_rev',
        'sin_rev', 'cos_rev', 'tan_rev',
        'pow_rev1', 'pow_rev2',
        # M13g: ieee 1788's signals and bare constructors
        'UndefinedOperationError', 'PossiblyUndefinedOperationWarning', 'text_to_interval', 'nums_to_interval',
        # M13g: ieee 1788's decorated type and its constructors
        'DecoratedInterval', 'Decoration', 'set_dec', 'text_to_decorated_interval', 'nums_to_decorated_interval',
        # M15: autodiff and interval newton (intervals/autodiff.py, intervals/solver.py)
        'Dual', 'derivative', 'newton', 'Root',
        # M16a: several variables (intervals/autodiff.py, intervals/solver.py)
        'gradient', 'jacobian', 'solve', 'RootBox'}
    for name in ('add', 'mul', 'OpDescriptor', 'apply_binary', 'mod', 'floordiv', 'floor', 'sqrt', 'sign',
                 'fma', 'apply', 'step'):
        assert not hasattr(intervals, name), name
