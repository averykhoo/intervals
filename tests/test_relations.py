import math
import operator
from fractions import Fraction

import pytest
from hypothesis import given

from intervals import EMPTY
from intervals import Allen
from intervals import MultiInterval
from intervals import TruthSet
from intervals import kernel
from intervals import relations
from intervals.relations import BOTH
from intervals.relations import FALSE
from intervals.relations import NEITHER
from intervals.relations import TRUE
from tests.strategies import exact_cut_tuples

P = MultiInterval.parse
inf = math.inf


def dense_probes(*cut_tuples_):
    """
    every endpoint value, two points inside each gap between consecutive endpoint values and two
    beyond each end, plus ±inf. two per gap is enough to realise `<`, `==` and `>` between two
    sets that both cover the gap, so the attained truth values over these probes are exact
    """
    finite = sorted({cut.value for cuts in cut_tuples_ for cut in cuts if math.isfinite(cut.value)})
    probes = [-inf, inf, *finite]
    if not finite:
        return probes + [0, 1]
    probes += [finite[0] - 2, finite[0] - 1, finite[-1] + 1, finite[-1] + 2]
    for lo, hi in zip(finite, finite[1:]):
        step = Fraction(hi - lo) / 3
        probes += [lo + step, lo + 2 * step]
    return probes


def attained(op, a, b):
    probes = dense_probes(a, b)
    xs = [x for x in probes if kernel.contains_point(a, x)]
    ys = [y for y in probes if kernel.contains_point(b, y)]
    return TruthSet({op(x, y) for x in xs for y in ys})


# TRUTHSET

def test_truth_set_bool():
    assert bool(TRUE) is True and bool(FALSE) is False
    with pytest.raises(ValueError):
        bool(BOTH)
    with pytest.raises(ValueError):
        bool(NEITHER)


@pytest.mark.parametrize('truth, certainly, possibly', [
    (NEITHER, True, False),  # vacuous
    (TRUE, True, True),
    (FALSE, False, False),
    (BOTH, False, True),
])
def test_truth_set_modalities(truth, certainly, possibly):
    assert (truth.certainly, truth.possibly) == (certainly, possibly)


def test_truth_set_misc():
    assert ~TRUE == FALSE and ~BOTH == BOTH and ~NEITHER == NEITHER
    assert repr(BOTH) == 'BOTH' and len({TRUE, TruthSet([True])}) == 1
    with pytest.raises(ValueError):
        TruthSet({1, 2})


# POINTWISE COMPARISONS AGAINST THE ORACLE

@pytest.mark.parametrize('ours, op', [
    (relations.lt, operator.lt),
    (relations.le, operator.le),
    (relations.gt, operator.gt),
    (relations.ge, operator.ge),
    (relations.eq_pointwise, operator.eq),
])
@given(a=exact_cut_tuples, b=exact_cut_tuples)
def test_comparisons_match_oracle(ours, op, a, b):
    assert ours(a, b) == attained(op, a, b)


@pytest.mark.parametrize('expr, expected', [
    (lambda: P('[1, 2)') < P('[2, 3]'), TRUE),
    (lambda: P('[1, 2]') < P('[2, 3]'), BOTH),  # 2 < 2 is false
    (lambda: P('[1, 2]') <= P('[2, 3]'), TRUE),
    (lambda: P('[1, 2)') <= P('(2, 3]'), TRUE),
    (lambda: P('[1, 3]') < P('[2, 4]'), BOTH),  # v1 said True
    (lambda: P('[3, 4]') < P('[1, 2]'), FALSE),
    (lambda: EMPTY < P('[2, 4]'), NEITHER),  # v1 said True
    (lambda: P('[1, 2]') < 3, TRUE),
    (lambda: 3 < P('[1, 2]'), FALSE),
    (lambda: P('[1, 2]').eq_pointwise(P('[1, 2]')), BOTH),
    (lambda: P('[2]').eq_pointwise(2), TRUE),
    (lambda: P('[inf]') < P('[inf]'), FALSE),
])
def test_comparison_table(expr, expected):
    assert expr() == expected


def test_no_trichotomy_and_le_is_not_lt_or_eq():
    a, b = P('[1, 2]'), P('[1, 2]')
    assert (a < b) == BOTH and (a > b) == BOTH and a == b
    c, d = P('[0, 1]'), P('[1, 2]')
    assert (c <= d) == TRUE and (c < d) == BOTH and c != d


def test_sorted_raises_on_ambiguity():
    assert sorted([P('[3, 4]'), P('[1, 2]')]) == [P('[1, 2]'), P('[3, 4]')]
    with pytest.raises(ValueError):
        sorted([P('[2, 4]'), P('[1, 3]')])


# SET-LEVEL RELATIONS

@given(exact_cut_tuples, exact_cut_tuples)
def test_before_is_certainly_less(a, b):
    if a and b:
        assert relations.before(a, b) == relations.lt(a, b).certainly
        assert relations.after(a, b) == relations.gt(a, b).certainly
    else:
        assert not relations.before(a, b) and not relations.after(a, b)


@given(exact_cut_tuples, exact_cut_tuples)
def test_set_relations(a, b):
    inter = kernel.intersection(a, b)
    assert relations.overlaps(a, b) == bool(inter) == (not relations.disjoint(a, b))
    assert relations.contains(a, b) == (inter == b)
    assert relations.within(a, b) == (inter == a)
    assert relations.equals(a, b) == (a == b)
    if relations.adjoins(a, b):
        # tiling without sharing: disjoint, and the hull of the union has no gap at the join
        assert not inter
        assert len(kernel.union(kernel.hull(a), kernel.hull(b))) == 2


def test_relation_table():
    assert P('[1, 2)').before(P('[2, 3]')) and not P('[1, 2]').before(P('[2, 3]'))
    assert P('[2, 3]').after(P('[1, 2)'))
    assert P('[1, 2)').adjoins(P('[2, 3]')) and P('[2, 3]').adjoins(P('[1, 2)'))
    assert not P('[1, 2]').adjoins(P('[2, 3]')) and not P('[1, 2)').adjoins(P('(2, 3]'))
    assert P('[1, 2]').overlaps(P('[2, 3]')) and not P('[1, 2)').overlaps(P('[2, 3]'))
    assert P('[1, 5]').contains(P('[2, 3]')) and P('[2, 3]').within(P('[1, 5]'))
    assert not EMPTY.before(P('[1]')) and not EMPTY.adjoins(P('[1]'))


# ALLEN

@pytest.mark.parametrize('a, b, relation', [
    ('[1, 2]', '[3, 4]', Allen.BEFORE),
    ('[1, 2)', '[2, 3]', Allen.MEETS),  # tiling, no shared point
    ('[1, 2]', '(2, 3]', Allen.MEETS),
    ('[1, 2]', '[2, 3]', Allen.OVERLAPS),  # the cut-refined case: they share the point 2
    ('[1, 3]', '[2, 4]', Allen.OVERLAPS),
    ('[1, 2]', '[1, 3]', Allen.STARTS),
    ('[2, 3]', '[1, 4]', Allen.DURING),
    ('[2, 3]', '[1, 3]', Allen.FINISHES),
    ('[1, 3]', '[1, 3]', Allen.EQUALS),
    ('[1, 3]', '[2, 3]', Allen.FINISHED_BY),
    ('[1, 4]', '[2, 3]', Allen.CONTAINS),
    ('[1, 3]', '[1, 2]', Allen.STARTED_BY),
    ('[2, 4]', '[1, 3]', Allen.OVERLAPPED_BY),
    ('[2, 3]', '[1, 2)', Allen.MET_BY),
    ('[3, 4]', '[1, 2]', Allen.AFTER),
    ('[2]', '[2, 3]', Allen.STARTS),
    ('(1, 2]', '[1, 2]', Allen.FINISHES),
])
def test_allen_table(a, b, relation):
    assert P(a).allen(P(b)) is relation
    assert P(b).allen(P(a)) is relation.inverse


contiguous = exact_cut_tuples.filter(lambda c: len(c) == 2)


@given(contiguous, contiguous)
def test_allen_semantics(a, b):
    rel = relations.allen(a, b)
    assert relations.allen(b, a) is rel.inverse
    inter = kernel.intersection(a, b)
    a_minus, b_minus = kernel.difference(a, b), kernel.difference(b, a)
    below = lambda x, y: bool(x) and bool(y) and relations.lt(x, y).certainly  # noqa: E731
    expected = {
        Allen.BEFORE: not inter and below(a, b) and len(kernel.union(a, b)) == 4,
        Allen.MEETS: not inter and below(a, b) and len(kernel.union(a, b)) == 2,
        Allen.OVERLAPS: bool(inter) and below(a_minus, inter) and below(inter, b_minus),
        Allen.STARTS: bool(b_minus) and not a_minus and below(a, b_minus),
        Allen.DURING: not a_minus and len(b_minus) == 4,
        Allen.FINISHES: bool(b_minus) and not a_minus and below(b_minus, a),
        Allen.EQUALS: a == b,
    }
    for relation, holds in list(expected.items()):
        expected[relation.inverse] = {
            Allen.BEFORE: not inter and below(b, a) and len(kernel.union(a, b)) == 4,
            Allen.MEETS: not inter and below(b, a) and len(kernel.union(a, b)) == 2,
            Allen.OVERLAPS: bool(inter) and below(b_minus, inter) and below(inter, a_minus),
            Allen.STARTS: bool(a_minus) and not b_minus and below(b, a_minus),
            Allen.DURING: not b_minus and len(a_minus) == 4,
            Allen.FINISHES: bool(a_minus) and not b_minus and below(a_minus, b),
            Allen.EQUALS: a == b,
        }[relation]
    assert [r for r, holds in expected.items() if holds] == [rel]


def test_allen_needs_contiguous_operands():
    with pytest.raises(ValueError):
        P('{[1], [3]}').allen(P('[1, 2]'))
    with pytest.raises(ValueError):
        EMPTY.allen(P('[1, 2]'))
