"""
comparisons and relations over cut tuples

`lt le gt ge` and `eq_pointwise` are **pointwise**: the result is the set of truth values that
`a op b` attains over all `a ∈ A, b ∈ B`, as a `TruthSet`. there is no trichotomy, and `a <= b` is
not `a < b or a == b` (pointwise vs structural).

the relations (`before`, `adjoins`, `overlaps`, ...) are set-level facts and return plain bool.
they are defined on cuts, not values: `before` is `A.end <= B.start`, so `[1, 2)` is before
`[2, 3]` while `[1, 2]` is not. for non-empty operands `before(A, B) == lt(A, B).certainly`.
"""
from enum import Enum
from typing import FrozenSet

from intervals import kernel
from intervals.kernel import Cuts


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


def _truth_set(possibly: bool, certainly: bool) -> TruthSet:
    return TruthSet(([True] if possibly else []) + ([] if certainly else [False]))


# POINTWISE COMPARISONS

def lt(a: Cuts, b: Cuts) -> TruthSet:
    """
    possibly: some `x < y` iff `inf A < sup B` as values. certainly: every `x < y` iff
    `A.end <= B.start` as cuts (`[1, 2)` vs `[2, 3]`: the end `2)` equals the start `[2`)
    """
    if not a or not b:
        return NEITHER
    return _truth_set(a[0].value < b[-1].value, a[-1] <= b[0])


def le(a: Cuts, b: Cuts) -> TruthSet:
    """
    possibly: some `x <= y` iff `A.start < B.end` as cuts (at a shared value both must be closed).
    certainly: every `x <= y` iff `sup A <= inf B` as values
    """
    if not a or not b:
        return NEITHER
    return _truth_set(a[0] < b[-1], a[-1].value <= b[0].value)


def gt(a: Cuts, b: Cuts) -> TruthSet:
    return lt(b, a)


def ge(a: Cuts, b: Cuts) -> TruthSet:
    return le(b, a)


def eq_pointwise(a: Cuts, b: Cuts) -> TruthSet:
    """`{T}` only for the same single point; `{T, F}` for any non-degenerate `A` against itself"""
    if not a or not b:
        return NEITHER
    same_point = len(a) == 2 and a == b and a[0].value == a[1].value
    return _truth_set(bool(kernel.intersection(a, b)), same_point)


# SET-LEVEL RELATIONS (bool; an empty operand is before, after or adjoining nothing)

def before(a: Cuts, b: Cuts) -> bool:
    return bool(a) and bool(b) and a[-1] <= b[0]


def after(a: Cuts, b: Cuts) -> bool:
    return before(b, a)


def adjoins(a: Cuts, b: Cuts) -> bool:
    """one ends exactly where the other starts: they tile without sharing a point"""
    return bool(a) and bool(b) and (a[-1] == b[0] or b[-1] == a[0])


def disjoint(a: Cuts, b: Cuts) -> bool:
    return not kernel.intersection(a, b)


def overlaps(a: Cuts, b: Cuts) -> bool:
    """they share at least one point"""
    return bool(kernel.intersection(a, b))


def contains(a: Cuts, b: Cuts) -> bool:
    """`B ⊆ A`"""
    return kernel.is_subset(b, a)


def within(a: Cuts, b: Cuts) -> bool:
    """`A ⊆ B`"""
    return kernel.is_subset(a, b)


def equals(a: Cuts, b: Cuts) -> bool:
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
