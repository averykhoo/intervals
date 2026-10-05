"""
comparisons and relations over cut tuples

`lt le gt ge` and `eq_pointwise` are **pointwise**: the result is the set of truth values that
`a op b` attains over all `a ∈ A, b ∈ B`, as a `TruthSet`. there is no trichotomy, and `a <= b` is
not `a < b or a == b` (pointwise vs structural).

the relations (`before`, `adjoins`, `overlaps`, ...) are set-level facts and return plain bool.
they are defined on cuts, not values: `before` is `A.end <= B.start`, so `[1, 2)` is before
`[2, 3]` while `[1, 2]` is not. for non-empty operands `before(A, B) == lt(A, B).certainly`.

the interval orders `weakly_less` and `strictly_less` (ieee 1788's `less` and `strictLess`) compare
the ends, the infima and the suprema, so for a multi-interval they are its hull's; they return bool.

`allen` needs two contiguous operands. `allen_matrix` and `allen_relations` take any number of
pieces: `allen` of each pair, as a matrix or as the set of relations holding. the matrix takes any
cut pairs, in any order; the set view's sweep needs normalized cut tuples.

one rule for the operands (owner, 2026-10-03): every function here reads normalized cut tuples
(`kernel.is_valid`) and asserts it under `__debug__`, as `MultiInterval` asserts its own; the methods
always pass normalized ones. the one exception is `allen_matrix`, the plain loop, which reads its
operands only piece by piece, and each piece is checked by the `allen` it is passed to.
"""
import math
from enum import Enum
from typing import FrozenSet
from typing import Iterator
from typing import Tuple

from multiinterval import kernel
from multiinterval.kernel import Cuts


class TruthSet:
    """
    the truth values attained by a pointwise comparison: `{}`, `{T}`, `{F}` or `{T, F}`

    `bool()` works on `{T}` and `{F}` and raises on the other two: `{T, F}` is ambiguous, and `{}`
    (an empty operand) attains no truth value at all. `.certainly` ("every attained value is T")
    and `.possibly` ("some attained value is T") never raise; on `{}` they are True and False.
    """
    __slots__ = ('values',)
    values: FrozenSet[bool]

    def __init__(self, values=()):
        values = frozenset(values)
        if not values <= {True, False}:
            raise ValueError(values)
        object.__setattr__(self, 'values', values)

    def __setattr__(self, name, value):
        raise AttributeError('TruthSet is immutable')

    @property
    def certainly(self) -> bool:
        return False not in self.values

    @property
    def possibly(self) -> bool:
        return True in self.values

    def __bool__(self) -> bool:
        if len(self.values) == 1:
            return next(iter(self.values))
        if not self.values:
            raise ValueError('an empty operand attains no truth value; use .certainly or .possibly')
        raise ValueError('the comparison is true for some points and false for others; '
                         'use .certainly or .possibly')

    def __invert__(self) -> 'TruthSet':
        return TruthSet(not value for value in self.values)

    def __eq__(self, other):
        if not isinstance(other, TruthSet):
            return NotImplemented
        return self.values == other.values

    def __hash__(self):
        return hash(self.values)

    def __repr__(self) -> str:
        return {frozenset(): 'NEITHER', frozenset({True}): 'TRUE', frozenset({False}): 'FALSE',
                frozenset({True, False}): 'BOTH'}[self.values]


NEITHER = TruthSet()
TRUE = TruthSet({True})
FALSE = TruthSet({False})
BOTH = TruthSet({True, False})


def _normalized(*operands) -> bool:
    """the check every relation asserts on its operands: normalized cut tuples (module docstring)"""
    return all(kernel.is_valid(x) for x in operands)


def _truth_set(possibly: bool, certainly: bool) -> TruthSet:
    return TruthSet(([True] if possibly else []) + ([] if certainly else [False]))


# POINTWISE COMPARISONS

def lt(a: Cuts, b: Cuts) -> TruthSet:
    """
    possibly: some `x < y` iff `inf A < sup B` as values. certainly: every `x < y` iff
    `A.end <= B.start` as cuts (`[1, 2)` vs `[2, 3]`: the end `2)` equals the start `[2`)
    """
    assert _normalized(a, b), (a, b)
    if not a or not b:
        return NEITHER
    return _truth_set(a[0].value < b[-1].value, a[-1] <= b[0])


def le(a: Cuts, b: Cuts) -> TruthSet:
    """
    possibly: some `x <= y` iff `A.start < B.end` as cuts (at a shared value both must be closed).
    certainly: every `x <= y` iff `sup A <= inf B` as values
    """
    assert _normalized(a, b), (a, b)
    if not a or not b:
        return NEITHER
    return _truth_set(a[0] < b[-1], a[-1].value <= b[0].value)


def gt(a: Cuts, b: Cuts) -> TruthSet:
    return lt(b, a)


def ge(a: Cuts, b: Cuts) -> TruthSet:
    return le(b, a)


def eq_pointwise(a: Cuts, b: Cuts) -> TruthSet:
    """`{T}` only for the same single point; `{T, F}` for any non-degenerate `A` against itself"""
    assert _normalized(a, b), (a, b)
    if not a or not b:
        return NEITHER
    same_point = len(a) == 2 and a == b and a[0].value == a[1].value
    return _truth_set(bool(kernel.intersection(a, b)), same_point)


# SET-LEVEL RELATIONS (bool; an empty operand is before, after or adjoining nothing)

def before(a: Cuts, b: Cuts) -> bool:
    assert _normalized(a, b), (a, b)
    return bool(a) and bool(b) and a[-1] <= b[0]


def after(a: Cuts, b: Cuts) -> bool:
    return before(b, a)


def adjoins(a: Cuts, b: Cuts) -> bool:
    """one ends exactly where the other starts: they tile without sharing a point"""
    assert _normalized(a, b), (a, b)
    return bool(a) and bool(b) and (a[-1] == b[0] or b[-1] == a[0])


def disjoint(a: Cuts, b: Cuts) -> bool:
    assert _normalized(a, b), (a, b)
    return not kernel.intersection(a, b)


def overlaps(a: Cuts, b: Cuts) -> bool:
    """they share at least one point"""
    assert _normalized(a, b), (a, b)
    return bool(kernel.intersection(a, b))


def contains(a: Cuts, b: Cuts) -> bool:
    """`B ⊆ A`"""
    assert _normalized(a, b), (a, b)
    return kernel.is_subset(b, a)


def within(a: Cuts, b: Cuts) -> bool:
    """`A ⊆ B`"""
    assert _normalized(a, b), (a, b)
    return kernel.is_subset(a, b)


def equals(a: Cuts, b: Cuts) -> bool:
    assert _normalized(a, b), (a, b)
    return a == b


def certainly_before(a: Cuts, b: Cuts) -> bool:
    return lt(a, b).certainly


def possibly_before(a: Cuts, b: Cuts) -> bool:
    return lt(a, b).possibly


def certainly_after(a: Cuts, b: Cuts) -> bool:
    return gt(a, b).certainly


def possibly_after(a: Cuts, b: Cuts) -> bool:
    return gt(a, b).possibly


def certainly_equal(a: Cuts, b: Cuts) -> bool:
    return eq_pointwise(a, b).certainly


def possibly_equal(a: Cuts, b: Cuts) -> bool:
    return eq_pointwise(a, b).possibly


# INTERVAL ORDERS (ieee 1788's less and strictLess: bool, on the ends)

def _ends(a: Cuts, b: Cuts):
    """`(inf A, sup A, inf B, sup B)` as values, so the hull's; open or closed does not matter"""
    return a[0].value, a[-1].value, b[0].value, b[-1].value


def weakly_less(a: Cuts, b: Cuts) -> bool:
    """
    `inf A <= inf B` and `sup A <= sup B`. two empty sets are ordered, an empty and a non-empty
    set are not. 1788's `less` on its closed intervals; on ends only, so open or closed ends do not
    matter here (`[0, 2]` is weakly less than `[0, 2)`, which the pointwise reading would refuse)
    """
    assert _normalized(a, b), (a, b)
    if not a or not b:
        return not a and not b
    lo_a, hi_a, lo_b, hi_b = _ends(a, b)
    return lo_a <= lo_b and hi_a <= hi_b


def strictly_less(a: Cuts, b: Cuts) -> bool:
    """
    `inf A < inf B` and `sup A < sup B`, where two starts at -inf and two ends at inf also count,
    as in 1788's `strictLess` (so `(-inf, inf)` is strictly less than itself). a start at inf or an
    end at -inf is a point there, and is not strictly less than itself. empty sets as `weakly_less`
    """
    assert _normalized(a, b), (a, b)
    if not a or not b:
        return not a and not b
    lo_a, hi_a, lo_b, hi_b = _ends(a, b)
    return ((lo_a < lo_b or lo_a == lo_b == -math.inf)
            and (hi_a < hi_b or hi_a == hi_b == math.inf))


# ALLEN

class Allen(Enum):
    """
    allen's 13 relations between two contiguous pieces, decided on cuts

    cuts make this finer than the classical reading: `[1, 2)` MEETS `[2, 3]` (they tile without
    sharing), while `[1, 2]` and `[2, 3]` share the point 2, so they OVERLAP
    """
    BEFORE = 'before'
    MEETS = 'meets'
    OVERLAPS = 'overlaps'
    STARTS = 'starts'
    DURING = 'during'
    FINISHES = 'finishes'
    EQUALS = 'equals'
    FINISHED_BY = 'finished by'
    CONTAINS = 'contains'
    STARTED_BY = 'started by'
    OVERLAPPED_BY = 'overlapped by'
    MET_BY = 'met by'
    AFTER = 'after'

    @property
    def inverse(self) -> 'Allen':
        return _ALLEN_INVERSE[self]


_ALLEN_INVERSE = {
    Allen.BEFORE: Allen.AFTER, Allen.MEETS: Allen.MET_BY, Allen.OVERLAPS: Allen.OVERLAPPED_BY,
    Allen.STARTS: Allen.STARTED_BY, Allen.DURING: Allen.CONTAINS, Allen.FINISHES: Allen.FINISHED_BY,
    Allen.EQUALS: Allen.EQUALS,
}
_ALLEN_INVERSE.update({v: k for k, v in list(_ALLEN_INVERSE.items())})


def allen(a: Cuts, b: Cuts) -> Allen:
    """the allen relation of `a` to `b`; both must be contiguous (exactly one piece)"""
    if len(a) != 2 or len(b) != 2:
        raise ValueError('allen() needs two contiguous, non-empty operands; pass hulls explicitly')
    assert _normalized(a, b), (a, b)
    (s1, e1), (s2, e2) = a, b
    if e1 < s2:
        return Allen.BEFORE
    if e1 == s2:
        return Allen.MEETS
    if e2 < s1:
        return Allen.AFTER
    if e2 == s1:
        return Allen.MET_BY
    if s1 == s2:
        return Allen.EQUALS if e1 == e2 else Allen.STARTS if e1 < e2 else Allen.STARTED_BY
    if e1 == e2:
        return Allen.FINISHES if s1 > s2 else Allen.FINISHED_BY
    if s1 < s2:
        return Allen.OVERLAPS if e1 < e2 else Allen.CONTAINS
    return Allen.DURING if e1 < e2 else Allen.OVERLAPPED_BY


def allen_matrix(a: Cuts, b: Cuts) -> Tuple[Tuple[Allen, ...], ...]:
    """
    `allen()` of every pair of pieces: row `i`, column `j` is the relation of piece `i` of `a` to
    piece `j` of `b`, so `len(a) // 2` rows of `len(b) // 2` entries, each one of the 13 as `Allen`
    reads them on cuts. an empty operand has no pairs, so no rows or rows of no entries, not a
    raise. the plain loop, `n m` calls to `allen()`: it does not rely on the pieces being in
    order, and `Θ(nm)` is the size of the answer anyway. for many pieces, `allen_relations`.
    nested tuples, so `zip(*M)` is the transpose and `np.array(M)` an `(n, m)` object array (but
    `()` for an empty `a`, whatever `m`); the converse, `allen_matrix(b, a)`, is the transpose with
    every entry's `.inverse`
    """
    pb = tuple(kernel.pairs(b))
    return tuple(tuple(allen(p, q) for q in pb) for p in kernel.pairs(a))


def allen_relations(a: Cuts, b: Cuts) -> FrozenSet[Allen]:
    """
    the relations holding between some piece of `a` and some piece of `b`: the entries of
    `allen_matrix(a, b)`, found in `O(n + m)` without building it (`_allen_pairs`, then the two
    corners). normalized operands only (asserted: out of order, the sweep would miss entries);
    `frozenset()` when either is empty. the set is extensional, a fact about the pieces: each
    relation in it holds between some pair. it is not allen's algebra's disjunction ("one of these
    holds, which is unknown"), though it has that type
    """
    assert _normalized(a, b), (a, b)
    pa, pb = tuple(kernel.pairs(a)), tuple(kernel.pairs(b))
    if not pa or not pb:
        return frozenset()
    found = {relation for _, _, relation in _allen_pairs(pa, pb)}
    # every pair the sweep skips is BEFORE or AFTER, and some piece of a is BEFORE some piece of b
    # iff the first of a is BEFORE the last of b
    if pa[0][1] < pb[-1][0]:
        found.add(Allen.BEFORE)
    if pb[0][1] < pa[-1][0]:
        found.add(Allen.AFTER)
    return frozenset(found)


def _allen_pairs(pa, pb) -> Iterator[Tuple[int, int, Allen]]:
    """
    the merge sweep over the cut pairs of two normalized operands: `(i, j, allen(pa[i], pb[j]))`
    for at most `n + m - 1` pairs, advancing the piece that ends first (both on a tie). every pair
    it skips is BEFORE or AFTER: normalized pieces never meet, so every later piece of the other
    operand starts beyond the end of the piece left behind. `allen` is looked up as the module
    global, which `tests/test_relations.py::test_allen_relations_is_a_linear_sweep` counts
    """
    i = j = 0
    while i < len(pa) and j < len(pb):
        p, q = pa[i], pb[j]
        yield i, j, allen(p, q)
        if p[1] < q[1]:
            i += 1
        elif q[1] < p[1]:
            j += 1
        else:
            i += 1
            j += 1
