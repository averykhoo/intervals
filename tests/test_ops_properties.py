"""
property tests for the arithmetic ops against the independent oracle in `tests.oracles`

the reference is the pointwise table (v2-plan.md "domain and semantics"): a result is the set of values
the defined pairs attain, +-inf ordinary points, every endpoint closed iff attained. soundness, endpoint
attainment and interior sharpness together pin a result to that set exactly; isotonicity, the union laws
and the reciprocal round trip are the algebraic laws the plan promises.

the ops are looked up on `multiinterval.ops` at call time, so a sabotage script can monkeypatch one and rerun
this file unchanged. every test here provokes the library's warnings on purpose (`[0] * [inf]`, empty
operands); the warnings themselves are pinned by `pytest.warns` in test_applicator.py.
"""
import math
import sys
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import ops
from multiinterval.applicator import apply_binary
from multiinterval.applicator import apply_unary
from multiinterval.cuts import Cut
from multiinterval.cuts import Side
from multiinterval.fmt import parse
from multiinterval.kernel import EMPTY
from multiinterval.kernel import REALS
from multiinterval.kernel import contains_point
from multiinterval.kernel import intersection
from multiinterval.kernel import is_subset
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.kernel import union
from multiinterval.multi_interval import MultiInterval
from multiinterval.multi_interval import OutwardMultiInterval
from tests.oracles import BINARY
from tests.oracles import OPS
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import sample
from tests.strategies import cut_tuples
from tests.strategies import probe_points

INF = math.inf

pytestmark = [
    pytest.mark.filterwarnings('ignore::multiinterval.errors.IndeterminateResultWarning'),
    pytest.mark.filterwarnings('ignore::multiinterval.errors.EmptySetPropagationWarning'),
]


# STRATEGIES

# the special points of the arithmetic: poles, indeterminate corners, flat spots
SPECIAL = (-INF, -1, 0, 1, INF)
special_values = st.sampled_from(SPECIAL)
exact_values = st.one_of(
    special_values,
    special_values,
    st.sampled_from([-3, -2, Fraction(-1, 3), Fraction(1, 2), 2, 3]),
    st.integers(-6, 6),
    st.fractions(min_value=-6, max_value=6, max_denominator=4),
)


@st.composite
def exact_pieces(draw, values=exact_values):
    """a piece with exact (int, Fraction, +-inf) ends; degenerate pieces at 0 and +-inf are frequent"""
    if draw(st.integers(0, 4)) == 0:
        v = draw(special_values)
        return piece(v, v)
    lo, hi = sorted((draw(values), draw(values)))
    return piece(lo, hi, draw(st.booleans()), draw(st.booleans()))


def exact_sets(min_pieces=0, max_pieces=3):
    return st.lists(exact_pieces(), min_size=min_pieces, max_size=max_pieces).map(normalize)


exact_operands = exact_sets()

# a set of special points: intersected with an operand it cuts out the degenerate [0], [inf], [-inf]
# that isotonicity has to survive (`[0]` inside `[-1, 0]` is the plan's sabotage target)
special_point_sets = st.lists(special_values, min_size=1, max_size=3).map(
    lambda vs: normalize(piece(v, v) for v in vs))
selectors = st.one_of(special_point_sets, special_point_sets, exact_operands, st.just(REALS))


@st.composite
def special_pieces(draw):
    """both ends special: `[-1, 0]`, `(0, inf]`, `[inf]`, ... (a degenerate open pair is empty)"""
    lo, hi = sorted((draw(special_values), draw(special_values)))
    return piece(lo, hi, draw(st.booleans()), draw(st.booleans()))


# the superset side of isotonicity: often built from special ends only, so that `[0]` sits in a one-sided
# `[-1, 0]` and `[inf]` is the whole operand, the two shapes that pin `1/[0] = ∅` and `[0] * [inf] = ∅`
iso_operands = st.one_of(
    exact_operands,
    st.lists(special_pieces(), min_size=1, max_size=2).map(normalize),
    special_point_sets,
)

exponents = st.integers(-3, 3)

# the soundness and sharpness tests check nothing on an empty operand (emptiness has its own test), and
# exact_operands drawn as those tests draw it gave an empty a or b in 162 of 300 binary examples
# (measured 2026-09-24, 60 examples x 5 seeds)
nonempty_operands = exact_sets(min_pieces=1).filter(bool)

# pow and reciprocal: ends among -inf, -1, 0, 1, inf with random closure, so an open end at 0 (where
# `(0, 1] ** -1 = [1, inf)` must not attain inf) is common, and exponents are mostly negative
pole_operands = st.lists(special_pieces(), min_size=1, max_size=2).map(normalize).filter(bool)
negative_biased_exponents = st.one_of(st.integers(-3, -1), exponents)


# DISPATCH

UNARY_FUNCS = {'neg': 'neg', 'pos': 'pos', 'abs': 'absolute', 'reciprocal': 'reciprocal'}


def apply(op, a, b=None):
    """op over cut tuples; for 'pow' b is the int exponent, for a unary op it is ignored"""
    if op == 'pow':
        return ops.power(a, b)
    if op in BINARY:
        return getattr(ops, op)(a, b)
    return getattr(ops, UNARY_FUNCS[op])(a)


def second(op, operands):
    """the strategy for the second argument of op: an operand, an exponent, or nothing"""
    if op == 'pow':
        return exponents
    if op in BINARY:
        return operands
    return st.none()


def sharp_operands(op):
    """the (a, b) strategies for the sharpness tests: non-empty, and pole-heavy for pow and reciprocal"""
    if op == 'pow':
        return st.one_of(nonempty_operands, pole_operands), negative_biased_exponents
    if op == 'reciprocal':
        return st.one_of(nonempty_operands, pole_operands), st.none()
    return nonempty_operands, second(op, nonempty_operands)


def oracle_b(op, b):
    """the second operand as the oracle wants it: the set, the exponent, or None"""
    return b if op in BINARY or op == 'pow' else None


def values_of(op, x, y, a, b):
    """oracle.pointwise with the operand sets passed where it needs them"""
    if op in BINARY:
        return pointwise(op, x, y, a, b)
    if op == 'pow':
        return pointwise(op, x, b, a)
    return pointwise(op, x, a=a)


def sampled_pairs(op, a, b, rng, n=8):
    xs = sample(a, n, rng)
    if op in BINARY:
        ys = sample(b, n, rng)
        return [(x, y) for x in xs for y in ys]
    return [(x, None) for x in xs]


def closed(cuts):
    """every piece closed: the conservative reading of a result with rounded float ends"""
    return normalize(piece(lo, hi) for lo, _, hi, _ in pieces(cuts))


def show(cuts):
    return list(pieces(cuts))


def endpoint_values(cuts):
    return {v for lo, _, hi, _ in pieces(cuts) for v in (lo, hi)}


def candidate_values(op, a, b):
    """
    every value the op gives at an endpoint pair (plus 0, +-1 and +-inf as extra corners): the true
    result's endpoints are among these, so probes built from them see every piece and every gap
    """
    xs = endpoint_values(a) | set(SPECIAL)
    ys = (endpoint_values(b) | set(SPECIAL)) if op in BINARY else {None}
    out = set()
    for x in xs:
        for y in ys:
            try:
                out.update(values_of(op, x, y, a, b))
            except ValueError:  # a pole at 0 where the operand has no 0
                pass
    return out


def probes(op, a, b, result):
    extra = normalize(piece(v, v) for v in candidate_values(op, a, b))
    return set(probe_points(result, extra)) | {0}


# SOUNDNESS: every value a sampled pair attains is in the result

@pytest.mark.parametrize('op', OPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_sound_exact(op, data, rng):
    a = data.draw(nonempty_operands, label='a')
    b = data.draw(second(op, nonempty_operands), label='b')
    result = apply(op, a, b)
    for x, y in sampled_pairs(op, a, b, rng):
        for v in values_of(op, x, y, a, b):
            assert contains_point(result, v), (x, y, v, show(result))


@pytest.mark.parametrize('op', OPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_sound_float_identity_rounding(op, data, rng):
    """
    the default ops round nothing, so python's float value of a pair should be between the float
    corners. a float corner that underflows or overflows must not collapse a piece to nothing (the
    pins in test_identity_rounding_collapse)
    """
    a = data.draw(cut_tuples(max_pieces=3), label='a')
    b = data.draw(second(op, cut_tuples(max_pieces=3)), label='b')
    check_float_identity_rounding(op, a, b, sampled_pairs(op, a, b, rng))


def check_float_identity_rounding(op, a, b, pairs):
    """
    the property of test_sound_float_identity_rounding on the given pairs. python rounds a pair's value, so it can
    step past an end the op computed exactly: `x - y` of `x` = 2.5e-138 in `(0, 5.1e-138)` and `y` = 1/5 is -0.2 in
    python (`float(1/5)`, then the float difference), below the exact open end -1/5 of `(0, 5.1e-138) - (-inf, 1/5]`
    (fuzz x10, 2026-10-03). a value outside the result is allowed only where the pair's exact value is inside
    """
    result = closed(apply(op, a, b))
    for x, y in pairs:
        try:
            values = values_of(op, x, y, a, b)
        except ValueError:
            # the oracle's float x ** n underflowed to 0.0 where the exact image has no 0 (pow, n < 0):
            # the pair has no python-float value to compare. the exact tests cover pow
            assert op == 'pow' and b < 0
            continue
        for v in values:
            assert (contains_point(result, v)
                    or all(contains_point(result, w) for w in values_of(op, _exact(x), _exact(y), a, b))), \
                (x, y, v, show(result))


@pytest.mark.parametrize('op, a, b, pair', [
    # fuzz x10, 2026-10-03: python's -0.2 is below the exact end -1/5; the pair's exact value is inside
    ('sub', normalize([piece(0, 5.0533628082320924e-138, False, False)]), normalize([piece(-INF, Fraction(1, 5))]),
     (2.5266814041160462e-138, Fraction(1, 5))),
])
def test_float_identity_rounding_past_an_exact_end(op, a, b, pair):
    """
    the exact class computes `(0, 5.1e-138) - (-inf, 1/5]` = `(-1/5, inf)` exactly (the corner 0 - 1/5 has no
    float), and python's float value of an interior pair can round past it, as no exact end can stop
    """
    x, y = pair
    assert not contains_point(closed(apply(op, a, b)), values_of(op, x, y, a, b)[0])  # the old oracle's claim
    check_float_identity_rounding(op, a, b, [pair])


def _exact(x):
    return Fraction(x) if isinstance(x, float) and math.isfinite(x) else x


def _round(fn, direction):
    """fn evaluated exactly on the finite corner, then rounded to the float on the given side"""
    def rounded(*args):
        exact = fn(*(Fraction(x) for x in args))
        try:
            f = float(exact)
        except OverflowError:
            return direction * INF if (exact > 0) == (direction > 0) else -direction * sys.float_info.max
        if (Fraction(f) - exact) * direction < 0:  # compare exactly: float - Fraction is a float
            f = math.nextafter(f, direction * INF)
        return f
    return rounded


def outward(desc):
    return desc._replace(rounded=(_round(desc.fn, -1), _round(desc.fn, 1)))


OUTWARD = {name: outward(getattr(ops, name.upper()))
           for name in ('add', 'sub', 'mul', 'div', 'neg', 'pos', 'abs', 'reciprocal')}


@pytest.mark.parametrize('op', sorted(OUTWARD))
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_sound_float_outward_rounding(op, data, rng):
    """with a directed-rounding hook the float result must hold the EXACT value of every float pair"""
    a = data.draw(cut_tuples(max_pieces=3), label='a')
    b = data.draw(second(op, cut_tuples(max_pieces=3)), label='b')
    desc = OUTWARD[op]
    result = closed(apply_binary(desc, a, b) if op in BINARY else apply_unary(desc, a))
    for x, y in sampled_pairs(op, a, b, rng):
        for v in values_of(op, _exact(x), _exact(y), a, b):
            assert contains_point(result, v), (x, y, v, show(result))


def test_rounding_hook_never_sees_a_pole():
    """a pole value is an exact infinity: `x / 0.0` must not go through the hook (`fn` has no value there)"""
    denominator = normalize([piece(-INF, 0.0, True, False)])
    assert apply_binary(OUTWARD['div'], parse('[-2]'), denominator) == parse('[0, inf)')
    assert apply_unary(OUTWARD['reciprocal'], denominator) == parse('(-inf, 0]')


@pytest.mark.parametrize('op, a, b, pair, value', [
    # corners 1 (exact, open) and 1 + 1e-300 == 1.0 (open): the piece collapses to nothing
    ('add', parse('[1]'), normalize([piece(0, 1e-300, False, False)]), (1, 5e-301), 1.0),
    ('mul', normalize([piece(0, 1e-200, False, False)]), normalize([piece(0, 1e-200, False, False)]),
     (5e-201, 5e-201), 0.0),
    ('pow', normalize([piece(0, 1.6770036655601444e-219, False, False)]), 2, (8.385018327800722e-220, 2), 0.0),
    # x ** 3 underflows to 0 on the negative side, so the pole side -inf of 1 / x ** 3 is lost
    ('pow', normalize([piece(-5.340330077810799e-153, Fraction(1, 2), True, False)]), -3,
     (-5.340330077810799e-153, -3), -INF),
    # overflow, the same collapse at the other end: 1 / 2.2e-313 is inf, so the piece is (inf, inf)
    ('reciprocal', normalize([piece(0, 2.2250738585e-313, False, False)]), None, (1.11253692926e-313, None), INF),
], ids=['add', 'mul', 'pow2', 'pow-3', 'reciprocal-overflow'])
def test_identity_rounding_collapse(op, a, b, pair, value):
    """
    minimal cases from test_sound_float_identity_rounding, all red before the fix: python's float
    value of a pair was outside the result even read with every piece closed. a piece rounding
    squeezes to one point keeps it (applicator.evaluate_box), and a negative power is evaluated as
    `1 / x ** -n` in one step, so an underflow to 0 keeps the sign of its pole (ops._power_descriptor)
    """
    x, y = pair
    assert value in values_of(op, x, y, a, b)
    assert contains_point(closed(apply(op, a, b)), value), show(apply(op, a, b))


def test_mixed_pair_rounded_once():
    """
    found by the fuzz profile (2026-09-26), red in the oracle, not the library: python's
    `Fraction(1, 3) / 2.75` rounds 1/3 to a float first and lands one ulp below the quotient, which
    the library computes exactly and rounds once (v2-plan.md "arithmetic"). the oracle now does the
    same (tests/oracles.py::_once)
    """
    a = normalize([piece(Fraction(1, 3), Fraction(1, 2), True, False)])
    b = normalize([piece(-INF, 2.75)])
    value = 0.12121212121212122
    assert value != Fraction(1, 3) / 2.75
    assert values_of('div', Fraction(1, 3), 2.75, a, b) == [value]
    assert contains_point(closed(apply('div', a, b)), value), show(apply('div', a, b))


_MIXED_POINT = (Cut(Fraction(1, 2), Side.BELOW), Cut(0.5, Side.ABOVE))


@given(a=cut_tuples())
@example(a=_MIXED_POINT)
@example(a=(Cut(0.5, Side.BELOW), Cut(Fraction(1, 2), Side.ABOVE), Cut(2, Side.BELOW), Cut(3.0, Side.ABOVE)))
def test_neg_and_pos_keep_each_cuts_type(a):
    """
    found by M14's first GitHub fuzz run (2026-09-29, fuzz-symmetry): `-` of a point whose cuts hold
    one value in two types (an exact 1/2 and a float 0.5) came back in its low cut's type alone, so
    `pown_rev(-c, -7)`, which reads each end by its own type, was not `-pown_rev(c, -7)`. negation is
    the cut mirror, type by type, and an involution; `+` is the identity
    """
    types = lambda cuts: [type(cut.value) for cut in cuts]  # noqa: E731
    mirrored = tuple(Cut(-cut.value, Side.ABOVE if cut.side is Side.BELOW else Side.BELOW) for cut in reversed(a))
    assert ops.neg(a) == mirrored and types(ops.neg(a)) == types(mirrored)
    assert ops.neg(ops.neg(a)) == a and types(ops.neg(ops.neg(a))) == types(a)
    assert ops.pos(a) == a and types(ops.pos(a)) == types(a)


# each number in an exact and a float form, so the ties D32 decides are common
_TIED = st.sampled_from([-INF, -2, -2.0, -1, -1.0, 0, 0.0, Fraction(1, 2), 0.5, 1, 1.0, 2, 2.0, INF])


@pytest.mark.parametrize('cls', [MultiInterval, OutwardMultiInterval])
@settings(max_examples=200, deadline=None)
@given(a=st.one_of(cut_tuples(values=_TIED), cut_tuples()))
@example(a=parse('[-1, 1.0]'))
@example(a=parse('[-1.0, 1]'))
@example(a=parse('[0.0, 1]'))
@example(a=parse('[0.0]'))
@example(a=parse('[-2.0, 0.0]'))
@example(a=parse('(-1.0, 1.0)'))
def test_abs_is_the_union_with_the_negation_on_the_half_line(cls, a):
    """
    the owner, D32 (2026-10-08): abs is `(X | -X) & [0, inf]`, an exact identity of sets whose 0 is the cut of the
    exact constant `[0, inf]` (set operations select cuts and never round). its definition and a pin, types
    included, in both classes: `abs(M(-1.0, 1))` was `[0, 1.0]` (the kernel's tie by position), and `abs(M(0.0, 1))`
    `[0.0, 1]` (abs's 0 from a float zero, which the identity gives as the constant 0)
    """
    x = cls.from_cuts(a)
    assert repr(abs(x)) == repr((x | -x) & cls(0, INF))


# SHARPNESS: on exact operands the result is exactly the attained set

@pytest.mark.parametrize('op', OPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_endpoints_closed_iff_attained(op, data):
    a_strategy, b_strategy = sharp_operands(op)
    a = data.draw(a_strategy, label='a')
    b = data.draw(b_strategy, label='b')
    result = apply(op, a, b)
    for lo, lo_closed, hi, hi_closed in pieces(result):
        assert lo_closed == attained(op, lo, a, oracle_b(op, b)), ('lo', lo, show(result))
        assert hi_closed == attained(op, hi, a, oracle_b(op, b)), ('hi', hi, show(result))


@pytest.mark.parametrize('op', OPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data(), rng=st.randoms(use_true_random=False))
def test_interior_sharpness(op, data, rng):
    """
    membership agrees with the oracle at every probe: endpoints of the result and of the true result,
    a point in every gap between them, and sampled points of the result (`[-1, 1] * [inf]` must not
    come out as `[-inf, inf]`)
    """
    a_strategy, b_strategy = sharp_operands(op)
    a = data.draw(a_strategy, label='a')
    b = data.draw(b_strategy, label='b')
    result = apply(op, a, b)
    for p in probes(op, a, b, result):
        assert contains_point(result, p) == attained(op, p, a, oracle_b(op, b)), (p, show(result))
    for p in sample(result, 10, rng):
        assert attained(op, p, a, oracle_b(op, b)), (p, show(result))


# ISOTONICITY: A ⊆ B ⇒ f(A) ⊆ f(B)

@st.composite
def cut_out(draw):
    """(B ∩ C, B): C is often a few special points, so the subset is often a degenerate [0] or [+-inf]"""
    b = draw(iso_operands)
    return intersection(b, draw(selectors)), b


@st.composite
def neighbourhoods(draw):
    """([v], a piece with v as a closed end) for v in 0, +-inf: `[0]` inside the one-sided `[-1, 0]`"""
    v = draw(st.sampled_from([0, INF, -INF]))
    w = draw(exact_values.filter(lambda w: w != v))
    lo, hi = sorted((v, w))
    whole = piece(lo, hi, lo == v or draw(st.booleans()), hi == v or draw(st.booleans()))
    return normalize([piece(v, v)]), normalize([whole])


degenerate_specials = st.sampled_from([0, INF, -INF]).map(lambda v: (normalize([piece(v, v)]),) * 2)

# (A, B) with A ⊆ B. the two shapes isotonicity is there to pin, `1/[0] ⊆ 1/[-1, 0]` and
# `[0] * [inf] ⊆ [0, 1] * [inf]`, need a neighbourhood on one side and often a bare `[+-inf]` on the
# other; a sabotage that sets 1/[0] back to `[-inf] ∪ [inf]` goes red within ~15 examples, and
# `[0] * [inf] = [0]` within ~60 (measured 2026-09-24, 150 examples x 5 seeds: the 1/[0] shape in 12-18%
# of examples, [0]*[±inf] 1-5 per 150; the explicit iso_examples make both sabotages deterministic)
subset_pairs = st.one_of(cut_out(), neighbourhoods(), neighbourhoods(), degenerate_specials, degenerate_specials)

iso_examples = [
    # the plan's sabotage target: [0] inside [-1, 0]; 1/[0] must not contain +-inf
    dict(p1=(parse('[0]'), parse('[-1, 0]')), p2=(parse('[1]'), parse('[1]')), n=-1),
    dict(p1=(parse('[0]'), parse('[0, 1]')), p2=(parse('[1]'), parse('[1]')), n=-2),
    # 1/[0] as a denominator
    dict(p1=(parse('[1]'), parse('[1]')), p2=(parse('[0]'), parse('[-1, 0]')), n=1),
    # [0] * [inf] inside [0, 1] * [inf], [inf] - [inf] inside [1, inf] - [inf]
    dict(p1=(parse('[0]'), parse('[0, 1]')), p2=(parse('[inf]'), parse('[inf]')), n=2),
    dict(p1=(parse('[inf]'), parse('[1, inf]')), p2=(parse('[inf]'), parse('[inf]')), n=3),
]


def _iso(test):
    for kwargs in iso_examples:
        test = example(**kwargs)(test)
    return test


@pytest.mark.parametrize('op', OPS)
@settings(max_examples=100, deadline=None)
@given(p1=subset_pairs, p2=subset_pairs, n=exponents)
@_iso
def test_isotone(op, p1, p2, n):
    (a1, b1), (a2, b2) = p1, p2
    assert is_subset(a1, b1) and is_subset(a2, b2)
    small = apply(op, a1, n if op == 'pow' else a2)
    big = apply(op, b1, n if op == 'pow' else b2)
    assert is_subset(small, big), (show(a1), show(a2), show(small), show(big))


# UNION LAWS: pointwise ops distribute over union; reciprocal and div's denominator only contain

EQUAL_UNION = [('add', 0), ('add', 1), ('sub', 0), ('sub', 1), ('mul', 0), ('mul', 1), ('div', 0),
               ('neg', 0), ('pos', 0), ('abs', 0)]
SUPERSET_UNION = [('reciprocal', 0), ('div', 1)]


def _union_sides(op, arg, a, b, c):
    if op not in BINARY:
        return apply(op, union(a, b)), union(apply(op, a), apply(op, b))
    if arg == 0:
        return apply(op, union(a, b), c), union(apply(op, a, c), apply(op, b, c))
    return apply(op, c, union(a, b)), union(apply(op, c, a), apply(op, c, b))


@pytest.mark.parametrize('op, arg', EQUAL_UNION)
@settings(max_examples=60, deadline=None)
@given(a=exact_operands, b=exact_operands, c=exact_operands)
def test_union_distributes(op, arg, a, b, c):
    """on exact operands with +-inf (the plan promises finite ones only)"""
    whole, parts = _union_sides(op, arg, a, b, c)
    assert whole == parts, (show(whole), show(parts))


@pytest.mark.parametrize('op, arg', SUPERSET_UNION)
@settings(max_examples=60, deadline=None)
@given(a=exact_operands, b=exact_operands, c=exact_operands)
def test_union_contains(op, arg, a, b, c):
    whole, parts = _union_sides(op, arg, a, b, c)
    assert is_subset(parts, whole), (show(whole), show(parts))


@settings(max_examples=60, deadline=None)
@given(a=exact_operands, b=exact_operands, n=exponents)
def test_power_union(a, b, n):
    """x ** n is pointwise for n >= 0; below that it is a reciprocal, so only ⊇"""
    whole, parts = ops.power(union(a, b), n), union(ops.power(a, n), ops.power(b, n))
    if n >= 0:
        assert whole == parts, (show(whole), show(parts))
    else:
        assert is_subset(parts, whole), (show(whole), show(parts))


def test_reciprocal_union_counterexample():
    """v2-plan.md testing: the direction of 1/[0] comes from its piece, so union equality fails"""
    a, b = parse('[-1, 0)'), parse('[0]')
    assert ops.reciprocal(union(a, b)) == parse('[-inf, -1]')
    assert union(ops.reciprocal(a), ops.reciprocal(b)) == parse('(-inf, -1]')
    assert ops.div(parse('[1]'), union(a, b)) == parse('[-inf, -1]')
    assert union(ops.div(parse('[1]'), a), ops.div(parse('[1]'), b)) == parse('(-inf, -1]')


# RECIPROCAL ROUND TRIP

def _no_special_point(cuts):
    return not any(lo == hi and lo in (0, INF, -INF) for lo, _, hi, _ in pieces(cuts))


def _one_sided_at_infinity(cuts):
    """unbounded at both ends but holding exactly one of +-inf: 1/(1/A) then gains the other one"""
    return bool(cuts) and cuts[0].value == -INF and cuts[-1].value == INF and (
        contains_point(cuts, -INF) != contains_point(cuts, INF))


@settings(max_examples=100, deadline=None)
@given(a=exact_operands.filter(_no_special_point))
@example(a=parse('(-1, 0)'))
@example(a=parse('[-inf, 0]'))
@example(a=parse('{ [-1, 0) , (0, 1] , [2, inf] }'))
@example(a=parse('[-inf, inf)'))
@example(a=parse('{ (-inf, -1] , [1, inf] }'))
def test_reciprocal_involution(a):
    """
    the plan's first `1/(1/A) == A` (no degenerate piece at 0, inf or -inf) was one precondition short
    (v2-plan.md corrected 2026-09-24), so this pins the exact round trip: A itself, plus the missing infinity when A is unbounded at both
    ends but holds exactly one of +-inf (see test_reciprocal_round_trip_gains_the_other_infinity)
    """
    back = a
    if _one_sided_at_infinity(a):
        back = union(a, parse('[inf]') if contains_point(a, -INF) else parse('[-inf]'))
    assert ops.reciprocal(ops.reciprocal(a)) == back, show(ops.reciprocal(a))


@pytest.mark.parametrize('text', ['[0]', '[inf]', '[-inf]', '{ [0] , [1, 2] }'])
def test_reciprocal_loses_a_degenerate_special_piece(text):
    """the price of 1/[0] = ∅ (D7): the round trip drops the degenerate piece"""
    a = parse(text)
    assert ops.reciprocal(ops.reciprocal(a)) != a


@pytest.mark.parametrize('text, back', [
    ('[-inf, inf)', '[-inf, inf]'),
    ('{ [-inf, -1] , [1, inf) }', '{ [-inf, -1] , [1, inf] }'),
    ('(-inf, inf]', '[-inf, inf]'),
])
def test_reciprocal_round_trip_gains_the_other_infinity(text, back):
    """
    1/(1/A) != A with no degenerate piece anywhere: 0 is in 1/A (it is 1/-inf) and the piece holding
    it crosses zero (A is unbounded above), so the second reciprocal's pole reaches +inf. this is the
    pointwise answer, so the plan's precondition is too weak, not the implementation
    """
    a = parse(text)
    once = ops.reciprocal(a)
    assert attained('reciprocal', 0, a)
    assert ops.reciprocal(once) == parse(back)
    assert all(attained('reciprocal', v, once) for v in (-INF, INF))


# THE CLASS AGREES WITH THE CUT-LEVEL OPS

CLASS_OPS = {
    'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y, 'pow': lambda x, n: x ** n, 'reciprocal': lambda x, _: x.reciprocal(),
    'neg': lambda x, _: -x, 'pos': lambda x, _: +x, 'abs': lambda x, _: abs(x),
}


@pytest.mark.parametrize('op', OPS)
@settings(max_examples=25, deadline=None)
@given(data=st.data())
def test_class_operators_match_ops(op, data):
    a = data.draw(exact_operands, label='a')
    b = data.draw(second(op, exact_operands), label='b')
    wrapped = MultiInterval.from_cuts(b) if op in BINARY else b
    assert CLASS_OPS[op](MultiInterval.from_cuts(a), wrapped).cuts == apply(op, a, b)


def test_empty_operand_gives_empty():
    for op in OPS:
        assert apply(op, EMPTY, 2 if op == 'pow' else EMPTY) == EMPTY
        if op in BINARY:
            assert apply(op, parse('[1, 2]'), EMPTY) == EMPTY
