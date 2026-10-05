import math
import operator
import warnings
from fractions import Fraction
from unittest import mock

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import strategies as st

from multiinterval import EMPTY
from multiinterval import Allen
from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval import TruthSet
from multiinterval import kernel
from multiinterval import relations
from multiinterval.cuts import Cut
from multiinterval.relations import BOTH
from multiinterval.relations import FALSE
from multiinterval.relations import NEITHER
from multiinterval.relations import TRUE
from tests.strategies import cut_tuples
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
    # on single pieces the matrix and the set view are allen()
    assert P(a).allen_matrix(P(b)) == ((relation,),)
    assert P(a).allen_relations(P(b)) == {relation}


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



# ALLEN OF ANY OPERANDS: THE MATRIX OF EVERY PAIR OF PIECES, AND THE SET OF RELATIONS HOLDING

BEFORE, MEETS, OVERLAPS, STARTS, DURING, FINISHES, EQUALS = (
    Allen.BEFORE, Allen.MEETS, Allen.OVERLAPS, Allen.STARTS, Allen.DURING, Allen.FINISHES, Allen.EQUALS)
FINISHED_BY, CONTAINS, STARTED_BY, OVERLAPPED_BY, MET_BY, AFTER = (
    Allen.FINISHED_BY, Allen.CONTAINS, Allen.STARTED_BY, Allen.OVERLAPPED_BY, Allen.MET_BY, Allen.AFTER)
DISJOINT = frozenset({BEFORE, MEETS, MET_BY, AFTER})
INSIDE = frozenset({STARTS, DURING, FINISHES, EQUALS})
HOLDS = frozenset(r.inverse for r in INSIDE)

# two independent exact_cut_tuples almost never share a cut, and MEETS is the cut-refined case. so
# a dense grid, and pairs where one operand is derived from the other and shares its cuts on purpose:
# the complement and the gaps MEET and are MET_BY its pieces, the hull STARTS, FINISHES and CONTAINS
# them, the interior STARTS, FINISHES or is DURING them at an open and a closed end of equal value
dense = cut_tuples(values=st.sampled_from([-inf, 0, 1, 2, 3, inf]), max_pieces=4)
operands = st.one_of(exact_cut_tuples, dense)


def _derived(a):
    hull = kernel.hull(a)
    return st.sampled_from([a, kernel.complement(a), hull, kernel.difference(hull, a),
                            kernel.interior(a), kernel.union(a, kernel.complement(hull))])


operand_pairs = st.one_of(st.tuples(operands, operands),
                          operands.flatmap(lambda a: st.tuples(st.just(a), _derived(a))),
                          operands.flatmap(lambda a: st.tuples(_derived(a), st.just(a))))


def allen_loop(a, b):
    """the oracle: allen() of each pair of pieces, a row per piece of `a`"""
    pb = tuple(kernel.pairs(b))
    return tuple(tuple(relations.allen(p, q) for q in pb) for p in kernel.pairs(a))


def converse(matrix, columns):
    """
    the transpose with every entry inverted. the column count is passed, not read off a row, so an
    n x 0 matrix becomes 0 x n, `()`, and a 0 x m one m empty rows
    """
    return tuple(tuple(matrix[i][j].inverse for i in range(len(matrix))) for j in range(columns))


@given(operand_pairs)
def test_allen_matrix_is_allen_of_each_pair_of_pieces(ab):
    a, b = ab
    matrix = relations.allen_matrix(a, b)
    assert matrix == allen_loop(a, b)  # tuples: a list of lists is not equal to it
    assert len(matrix) == len(a) // 2 and all(len(row) == len(b) // 2 for row in matrix)
    assert MultiInterval.from_cuts(a).allen_matrix(MultiInterval.from_cuts(b)) == matrix


@given(operands, operands, operands)
def test_allen_matrix_does_not_need_normalized_operands(a1, a2, b):
    """
    the matrix is the plain loop over allen(), so each row is its own piece's and each column its
    own: two cut tuples laid end to end (out of order, overlapping) give the two matrices stacked.
    the set view's sweep does need normalized operands; the matrix does not
    """
    top, bottom = relations.allen_matrix(a1, b), relations.allen_matrix(a2, b)
    assert relations.allen_matrix(a1 + a2, b) == top + bottom
    left, right = relations.allen_matrix(b, a1), relations.allen_matrix(b, a2)
    assert relations.allen_matrix(b, a1 + a2) == tuple(r + s for r, s in zip(left, right))


def test_allen_matrix_of_unordered_overlapping_pieces():
    a = P('[3, 4]')._cuts + P('[0, 5]')._cuts + P('[1, 2)')._cuts
    b = P('[2, 3]')._cuts + P('[0, 1]')._cuts
    assert relations.allen_matrix(a, b) == (
        (OVERLAPPED_BY, AFTER), (CONTAINS, STARTED_BY), (MEETS, OVERLAPPED_BY))


@given(operand_pairs)
def test_allen_relations_are_the_matrix_entries(ab):
    a, b = ab
    expected = frozenset(r for row in allen_loop(a, b) for r in row)
    got = relations.allen_relations(a, b)
    assert isinstance(got, frozenset) and got == expected
    assert MultiInterval.from_cuts(a).allen_relations(MultiInterval.from_cuts(b)) == expected


@given(operand_pairs)
def test_allen_matrix_converse(ab):
    a, b = ab
    assert relations.allen_matrix(b, a) == converse(relations.allen_matrix(a, b), len(b) // 2)
    assert relations.allen_relations(b, a) == {r.inverse for r in relations.allen_relations(a, b)}


@given(operands)
def test_allen_matrix_of_a_set_with_itself(a):
    """normalized pieces never meet (`kernel.normalize` merges them), so off the diagonal: BEFORE/AFTER"""
    n = len(a) // 2
    assert relations.allen_matrix(a, a) == tuple(tuple(
        EQUALS if i == j else BEFORE if i < j else AFTER for j in range(n)) for i in range(n))
    assert relations.allen_relations(a, a) == (
        set() if n == 0 else {EQUALS} if n == 1 else {EQUALS, BEFORE, AFTER})


@given(operand_pairs)
def test_allen_matrix_and_the_set_relations(ab):
    a, b = ab
    n, m = len(a) // 2, len(b) // 2
    matrix, found = relations.allen_matrix(a, b), relations.allen_relations(a, b)
    assert relations.overlaps(a, b) == bool(found - DISJOINT)
    assert relations.disjoint(a, b) == (found <= DISJOINT)
    assert relations.before(a, b) == (bool(a) and bool(b) and found <= {BEFORE, MEETS})
    assert relations.after(a, b) == (bool(a) and bool(b) and found <= {AFTER, MET_BY})
    # a piece lies in b iff it lies in one piece of b: the gaps of b hold points
    assert relations.within(a, b) == all(set(row) & INSIDE for row in matrix)
    # the columns are counted with range(m): zip(*matrix) would see none when n == 0
    assert relations.contains(a, b) == all({matrix[i][j] for i in range(n)} & HOLDS for j in range(m))
    assert relations.equals(a, b) == (n == m and all(matrix[i][i] is EQUALS for i in range(n)))
    if a and b:
        # adjoins is about the sets' ends, not any pair of pieces
        assert relations.adjoins(a, b) == (matrix[-1][0] is MEETS or matrix[0][-1] is MET_BY)


@given(contiguous, contiguous)
def test_allen_matrix_on_single_pieces(a, b):
    assert relations.allen_matrix(a, b) == ((relations.allen(a, b),),)
    assert relations.allen_relations(a, b) == {relations.allen(a, b)}


def test_allen_matrix_of_an_empty_operand():
    """no pairs of pieces: no rows or empty rows, and no relation; not a raise, not a warning"""
    A = P('[0, 1] | [2, 3]')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert EMPTY.allen_matrix(A) == () and EMPTY.allen_matrix(EMPTY) == ()
        assert A.allen_matrix(EMPTY) == ((), ())  # the shape: one empty row per piece
        assert EMPTY.allen_relations(A) == A.allen_relations(EMPTY) == frozenset()
        assert EMPTY.allen_relations(EMPTY) == frozenset()
    with pytest.raises(ValueError):
        EMPTY.allen(A)  # allen() owes one relation, and still raises


@pytest.mark.parametrize('a, b, matrix', [
    ('[0, 1] | [3, 5]', '[1, 4]', ((OVERLAPS,), (OVERLAPPED_BY,))),  # a shared closed end is a shared point
    ('[0, 1) | [3, 5]', '[1, 2]', ((MEETS,), (AFTER,))),  # a piece meets; the sets do not adjoin
    ('[0, 1) | (1, 2]', '[1]', ((MEETS,), (MET_BY,))),  # a point filling a one-point gap
    ('{[0], [2]}', '[0, 2]', ((STARTS,), (FINISHES,))),
    ('{[0], [2]}', '(0, 2)', ((MEETS,), (MET_BY,))),
    ('{[0], [2]}', '[1, 3]', ((BEFORE,), (DURING,))),
    ('{[0], [2]}', '{[0], [2]}', ((EQUALS, BEFORE), (AFTER, EQUALS))),
    ('[1, inf)', '[inf]', ((MEETS,),)),
    ('[1, inf]', '[inf]', ((FINISHED_BY,),)),
    ('[-inf]', '(-inf, 0] | [1, 2]', ((MEETS, BEFORE),)),
    ('[-inf]', '[-inf, 0] | [1, 2]', ((STARTS, BEFORE),)),
    ('[0, 1] | [4, 5]', '[2, 3] | [6, 7]', ((BEFORE, BEFORE), (AFTER, BEFORE))),  # both corners
    ('[0, 10]', '[0, 1] | [2, 3] | [9, 10]', ((STARTED_BY, CONTAINS, FINISHED_BY),)),
])
def test_allen_matrix_table(a, b, matrix):
    A, B = P(a), P(b)
    assert A.allen_matrix(B) == matrix
    assert A.allen_relations(B) == {r for row in matrix for r in row}
    assert B.allen_matrix(A) == converse(matrix, len(B))
    assert B.allen_relations(A) == {r.inverse for row in matrix for r in row}


def test_a_piece_meets_where_the_sets_do_not_adjoin():
    """`adjoins` is one set ending exactly where the other starts, not a MEETS entry anywhere"""
    assert MEETS in P('[0, 1) | [3, 5]').allen_relations(P('[1, 2]'))
    assert not P('[0, 1) | [3, 5]').adjoins(P('[1, 2]'))
    assert not P('{[0], [2]}').adjoins(P('(0, 2)')) and not P('[0, 1) | (1, 2]').adjoins(P('[1]'))


def test_allen_matrix_coerces_as_every_relation():
    A, B = P('[0, 1] | [3, 5]'), P('[1, 4]')
    assert A.allen_matrix(2) == A.allen_matrix(MultiInterval(2)) == ((BEFORE,), (AFTER,))
    assert A.allen_relations(2) == {BEFORE, AFTER}
    for view in (A.allen_matrix, A.allen_relations):
        with pytest.raises(TypeError):
            view('[1, 4]')
        with pytest.raises(ValueError):
            view(math.nan)
    O = OutwardMultiInterval.parse('[1, 4]')
    assert A.allen_matrix(O) == A.allen_matrix(B) and O.allen_matrix(A) == B.allen_matrix(A)
    assert A.allen_relations(O) == A.allen_relations(B) and O.allen_relations(A) == B.allen_relations(A)


def test_allen_matrix_mixes_numeric_types_at_a_shared_cut():
    """cuts compare by value across int, Fraction and float: the end `0.5)` is the start `[1/2`"""
    A = MultiInterval(0, 0.5, end_closed=False) | MultiInterval(3.0, 4)
    B = MultiInterval(Fraction(1, 2), 3)
    assert A.allen_matrix(B) == ((MEETS,), (OVERLAPPED_BY,))
    assert A.allen_relations(B) == {MEETS, OVERLAPPED_BY}
    assert MultiInterval(0, 1.0, end_closed=False).allen_matrix(MultiInterval(1, 2)) == ((MEETS,),)


@given(operand_pairs)
@example((P('[0, 1] | [2, 3]')._cuts,) * 2)  # every end ties: a random run misses it 1 in 20
def test_allen_relations_is_a_linear_sweep(ab):
    """
    the set view's cost, pinned by counts, not time. the sweep `_allen_pairs` visits no pair twice,
    at most `n + m - 1` pairs, and every pair that is neither BEFORE nor AFTER; `allen_relations`
    never builds the matrix (patched to raise) and calls `allen()` at most `n + m - 1` times and at
    least once per pair that is neither BEFORE nor AFTER.

    the lower bound pins an implementation detail on purpose: a correct refactor that inlines
    allen's comparisons for the visited pairs goes red here. it is the price of a spy that cannot
    pass by default (bind `allen` locally in `_allen_pairs` and the count reads 0, which the upper
    bound alone would accept). relax it knowingly; do not delete it
    """
    a, b = ab
    pa, pb = tuple(kernel.pairs(a)), tuple(kernel.pairs(b))
    bound = max(0, len(pa) + len(pb) - 1)
    needed = {(i, j) for i, p in enumerate(pa) for j, q in enumerate(pb)
              if relations.allen(p, q) not in (BEFORE, AFTER)}
    swept = list(relations._allen_pairs(pa, pb))
    visited = [(i, j) for i, j, _ in swept]
    assert len(visited) == len(set(visited)) <= bound
    assert needed <= set(visited)

    def cells_meet(i, j):
        # piece i owns the cuts after the end of piece i - 1, up to its own end: the two operands'
        # cells are two partitions, and the sweep walks their common refinement, so it visits
        # exactly the pairs of cells that intersect (a tie advancing one pointer would add one)
        ends = min(pa[i][1], pb[j][1])
        return all(prev < ends for prev in ([pa[i - 1][1]] if i else []) + ([pb[j - 1][1]] if j else []))

    assert set(visited) == {(i, j) for i in range(len(pa)) for j in range(len(pb)) if cells_meet(i, j)}
    assert all(r is relations.allen(pa[i], pb[j]) for i, j, r in swept)
    expected = frozenset(r for row in allen_loop(a, b) for r in row)
    with mock.patch.object(relations, 'allen_matrix', side_effect=AssertionError('built the matrix')), \
            mock.patch.object(relations, 'allen', wraps=relations.allen) as spy:
        assert relations.allen_relations(a, b) == expected
    assert len(needed) <= spy.call_count <= bound


_COMPARED = [0]


def _counted(op):
    def compare(self, other):
        _COMPARED[0] += 1
        return getattr(tuple, op)(self, other)
    return compare


class _CountingCut(Cut):
    """a Cut that counts its comparisons (`<`, `<=`, `==`, `!=`, `>`, `>=`), to pin the set view's work"""
    __slots__ = ()
    __lt__, __le__, __eq__, __ne__, __gt__, __ge__ = [
        _counted(op) for op in ('__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__')]
    __hash__ = Cut.__hash__


def test_allen_relations_compares_cuts_linearly():
    """
    the set view's cost in cut comparisons, which `::test_allen_relations_is_a_linear_sweep` (calls
    to allen()) does not see: a pass over every pair of pieces that calls neither allen() nor
    allen_matrix stays green there (review SAB-1). every piece of a lies AFTER every piece of b,
    where a quadratic scan for BEFORE cannot stop early: allen_relations makes O(n + m) cut
    comparisons, at most 10 per piece (the check of its operands, the sweep, the two corners)
    """
    n = m = 40
    a = P(' | '.join(f'[{1000 + 3 * k}, {1001 + 3 * k}]' for k in range(n)))._cuts
    b = P(' | '.join(f'[{3 * k}, {3 * k + 1}]' for k in range(m)))._cuts
    ca, cb = (tuple(_CountingCut(c.value, c.side) for c in x) for x in (a, b))
    _COMPARED[0] = 0
    found = relations.allen_relations(ca, cb)
    compared = _COMPARED[0]
    assert found == relations.allen_relations(a, b) == {AFTER}
    assert 0 < compared <= 10 * (n + m)


@pytest.mark.skipif(not __debug__, reason='the check is an assert, as MultiInterval._wrap')
def test_allen_relations_refuses_unnormalized_operands():
    """
    the sweep needs normalized operands (the matrix does not): out of order, the set view would be a
    strict subset of the matrix's entries, so it is refused under `__debug__`, as `MultiInterval`
    refuses a malformed cut tuple, not answered wrongly (review F1)
    """
    a = P('[3, 4]')._cuts + P('[0, 1]')._cuts
    b = P('[0, 1]')._cuts
    assert {r for row in relations.allen_matrix(a, b) for r in row} == {AFTER, EQUALS}
    for x, y in ((a, b), (b, a), (a, a)):
        with pytest.raises(AssertionError):
            relations.allen_relations(x, y)
    assert relations.allen_relations(b, b) == {EQUALS}


def _relations_over_cut_tuples():
    """every public function of `relations` taking two cut tuples, found by inspection, so a new one
    is covered by the rule below without being listed"""
    import inspect
    return sorted(name for name, f in vars(relations).items()
                  if inspect.isfunction(f) and f.__module__ == relations.__name__ and not name.startswith('_')
                  and list(inspect.signature(f).parameters) == ['a', 'b'])


@pytest.mark.skipif(not __debug__, reason='the check is an assert, as MultiInterval._wrap')
@pytest.mark.parametrize('name', _relations_over_cut_tuples())
def test_every_relation_asserts_normalized_operands(name):
    """
    one rule (owner, 2026-10-03, references/owner-questions-2026-10-03/allen.md (b)): every relation
    over cut tuples asserts normalized operands under `__debug__`, not `allen_relations` alone (whose
    wrong answer would be silent and partial; the others' read the first and last cut as the ends).
    `allen` reads two pieces, so a piece out of order is what it refuses; `allen_matrix`, the plain
    loop, takes pieces in any order (`::test_allen_matrix_does_not_need_normalized_operands`) and
    refuses a piece out of order through `allen`
    """
    f = getattr(relations, name)
    good = P('[0, 1]')._cuts
    unordered = P('[3, 4]')._cuts + good       # two pieces out of order
    reversed_piece = P('[3, 4]')._cuts[::-1]   # one piece, its cuts swapped
    bad = (reversed_piece,) if name in ('allen', 'allen_matrix') else (unordered, reversed_piece, list(good))
    for x in bad:
        for args in ((x, good), (good, x)):
            with pytest.raises(AssertionError):
                f(*args)
    f(good, good)  # and the normalized call answers
    if name == 'allen_matrix':
        assert f(unordered, good) == ((AFTER,), (EQUALS,))


def test_the_relations_are_the_ones_listed():
    """the inspection above finds every relation the module docstring's rule covers (so a renamed
    signature cannot drop one silently)"""
    assert _relations_over_cut_tuples() == sorted([
        'adjoins', 'after', 'allen', 'allen_matrix', 'allen_relations', 'before', 'certainly_after',
        'certainly_before', 'certainly_equal', 'contains', 'disjoint', 'eq_pointwise', 'equals', 'ge', 'gt',
        'le', 'lt', 'overlaps', 'possibly_after', 'possibly_before', 'possibly_equal', 'strictly_less',
        'weakly_less', 'within'])
