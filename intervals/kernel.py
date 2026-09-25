"""
set algebra over cut tuples

a cut tuple is an immutable, even-length, strictly increasing tuple of `Cut`s, read in pairs as
`(start, end)` pieces. every function here takes and returns cut tuples; only the class file knows
the class. a piece is non-empty iff `start < end`, and two pieces merge iff
`next.start <= current.end` -- there is no distance rule, because equal cuts are the same boundary.
"""
import math
from bisect import bisect_right
from fractions import Fraction
from itertools import groupby
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Tuple
from typing import Union

from intervals.cuts import Cut
from intervals.cuts import Value
from intervals.cuts import above
from intervals.cuts import as_end
from intervals.cuts import as_start
from intervals.cuts import below
from intervals.cuts import end_cut
from intervals.cuts import start_cut

Cuts = Tuple[Cut, ...]
Pair = Tuple[Cut, Cut]

EMPTY: Cuts = ()
REALS: Cuts = (below(-math.inf), above(math.inf))  # [-inf, inf]


# CONSTRUCTION

def piece(lo, hi, lo_closed: bool = True, hi_closed: bool = True) -> Pair:
    """
    the cut pair for one piece; reversed values are a ValueError, `[1, 1)` is an (empty) pair

    >>> piece(1, 2, hi_closed=False)
    (Cut(1, BELOW), Cut(2, BELOW))
    """
    start, end = start_cut(lo, lo_closed), end_cut(hi, hi_closed)
    if start.value > end.value:
        raise ValueError(f'interval start {lo!r} is after end {hi!r}')
    return start, end


def normalize(pairs: Iterable[Pair]) -> Cuts:
    """sort, drop empty pieces, merge pieces that overlap or tile exactly"""
    out = []
    for start, end in sorted(pairs):
        if start >= end:
            continue
        if out and start <= out[-1]:
            if end > out[-1]:
                out[-1] = end
        else:
            out.append(start)
            out.append(end)
    return tuple(out)


def pairs(cuts: Cuts) -> Iterator[Pair]:
    """the `(start, end)` cut pairs of a cut tuple"""
    it = iter(cuts)
    return zip(it, it)


def pieces(cuts: Cuts) -> Iterator[Tuple[Value, bool, Value, bool]]:
    """each piece as `(lo, lo_closed, hi, hi_closed)`"""
    for start, end in pairs(cuts):
        yield (*as_start(start), *as_end(end))


def is_valid(cuts) -> bool:
    """the representation invariant: a tuple of Cuts, even length, strictly increasing"""
    return (isinstance(cuts, tuple)
            and len(cuts) % 2 == 0
            and all(isinstance(cut, Cut) for cut in cuts)
            and all(a < b for a, b in zip(cuts, cuts[1:])))


class Builder:
    """
    incremental construction: collect pieces, then sort and sweep once in `build()`

    (compare.py measured one timsort-and-sweep at the end to be the cheap way to build in bulk)
    """
    __slots__ = ('_pairs',)

    def __init__(self):
        self._pairs = []

    def add_piece(self, lo, hi, lo_closed: bool = True, hi_closed: bool = True) -> 'Builder':
        self._pairs.append(piece(lo, hi, lo_closed, hi_closed))
        return self

    def add_point(self, value) -> 'Builder':
        return self.add_piece(value, value)

    def add(self, cuts: Cuts) -> 'Builder':
        self._pairs.extend(pairs(cuts))
        return self

    def build(self) -> Cuts:
        return normalize(self._pairs)


# SET ALGEBRA

def sweep(cut_tuples: Iterable[Cuts], keep: Callable[[int], bool]) -> Cuts:
    """
    the points covered by a number of the operands for which `keep(number)` is true

    every operand must be normalized, so each covers a point at most once. between two consecutive
    distinct cuts there is always at least one point (even between `(v, BELOW)` and `(v, ABOVE)`:
    the point v), so the depth after each group of equal cuts is the depth of a real region
    """
    events = []
    for cuts in cut_tuples:
        for start, end in pairs(cuts):
            events.append((start, 1))
            events.append((end, -1))
    events.sort()

    out = []
    depth = 0
    inside = keep(0)
    if inside:
        out.append(REALS[0])
    for cut, group in groupby(events, key=lambda event: event[0]):
        depth += sum(delta for _, delta in group)
        if keep(depth) != inside:
            inside = not inside
            out.append(cut)
    if inside:
        out.append(REALS[1])
    return normalize(pairs(out))


def overlap_count(cut_tuples: Iterable[Cuts], n: int) -> Cuts:
    """points covered by at least `n` operands (v1's `merge(n_overlaps=)`)"""
    if n < 1:
        raise ValueError(n)
    return sweep(cut_tuples, lambda depth: depth >= n)


def union(*cut_tuples: Cuts) -> Cuts:
    return normalize(pair for cuts in cut_tuples for pair in pairs(cuts))


def intersection(first: Cuts, *others: Cuts) -> Cuts:
    n = 1 + len(others)
    return sweep((first, *others), lambda depth: depth == n)


def symmetric_difference(*cut_tuples: Cuts) -> Cuts:
    """points covered by an odd number of operands"""
    return sweep(cut_tuples, lambda depth: depth % 2 == 1)


def complement(cuts: Cuts) -> Cuts:
    """
    prepend `[-inf`, append `inf]` and re-pair with the roles shifted: every start becomes an end
    and vice versa. pairs that come out empty (`start >= end`) drop out
    """
    shifted = (REALS[0], *cuts, REALS[1])
    return tuple(cut for start, end in pairs(shifted) if start < end for cut in (start, end))


def difference(minuend: Cuts, *subtrahends: Cuts) -> Cuts:
    return intersection(minuend, complement(union(*subtrahends)))


# MEMBERSHIP

def contains_point(cuts: Cuts, value) -> bool:
    """
    the point lies between `below(value)` and `above(value)`, and no cut can fall strictly between
    those two, so it is inside iff an odd number of cuts are at or before `below(value)`
    """
    return bisect_right(cuts, below(value)) % 2 == 1


def is_subset(inner: Cuts, outer: Cuts) -> bool:
    return intersection(inner, outer) == inner


def hull(cuts: Cuts) -> Cuts:
    return (cuts[0], cuts[-1]) if cuts else EMPTY


def interior(cuts: Cuts) -> Cuts:
    """
    every end opened, the infinite ones too: the interior in the topology of the reals. a
    degenerate piece drops out, and so does a closed end at ±inf, a point with no neighbourhood of
    reals. pieces stay apart, since a normalized tuple has a missing point between any two

    >>> interior(normalize([piece(0, 1), piece(2, 2), piece(3, math.inf)]))
    (Cut(0, ABOVE), Cut(1, BELOW), Cut(3, ABOVE), Cut(inf, BELOW))
    """
    return tuple(cut for start, end in pairs(cuts) if start.value < end.value
                 for cut in (above(start.value), below(end.value)))


# SIZE

class Size(NamedTuple):
    """
    a lex-ordered size: `rays` infinite rays, then `length`, then `points` extra endpoints

    think of it as `ω·rays + length + ε·points`. the baseline is a half-open piece, which has
    exactly its length; each closed endpoint adds half a point and each open one removes half, so
    `[1]` is 1 point, `[1, 2)` is `1`, `[1, 2]` is `1 + 1pt` and `(1, 2)` is `1 - 1pt`. a ray
    counts from the origin, so its length is the finite remainder and may be negative:
    `(1, inf]` is `Size(1, -1, 0)`.

    `+` is componentwise, so disjoint sets add: `size(A) + size(B) == size(A | B)`
    """
    rays: int
    length: Union[int, Fraction, float]
    points: int

    def __add__(self, other):
        if not isinstance(other, Size):
            return NotImplemented
        return Size(self.rays + other.rays, self.length + other.length, self.points + other.points)


def size(cuts: Cuts) -> Size:
    rays = 0
    length = 0
    half_points = 0
    for lo, lo_closed, hi, hi_closed in pieces(cuts):
        half_points += (1 if lo_closed else -1) + (1 if hi_closed else -1)
        left_ray = lo == -math.inf and hi > lo
        right_ray = hi == math.inf and lo < hi
        rays += left_ray + right_ray
        if left_ray and right_ray:
            pass
        elif left_ray:
            length += hi  # the ray (-inf, 0] minus (hi, 0], or plus [0, hi)
        elif right_ray:
            length -= lo
        elif lo != hi:
            length += hi - lo
    assert half_points % 2 == 0
    return Size(rays, length, half_points // 2)
