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

from multiinterval.cuts import Cut
from multiinterval.cuts import Value
from multiinterval.cuts import above
from multiinterval.cuts import as_end
from multiinterval.cuts import as_start
from multiinterval.cuts import below
from multiinterval.cuts import end_cut
from multiinterval.cuts import flag
from multiinterval.cuts import start_cut

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


def checked_piece(lo, hi, lo_closed: bool = True, hi_closed: bool = True) -> Pair:
    """`piece` for flags a user passed: each a bool, else a TypeError (`cuts.flag`)"""
    return piece(lo, hi, flag(lo_closed, 'lo_closed'), flag(hi_closed, 'hi_closed'))


def normalize(pairs: Iterable[Pair]) -> Cuts:
    """
    sort, drop empty pieces, merge pieces that overlap or tile exactly

    a tie goes to the exact type (D32): where two pieces end at equal cuts, one an exact value and one a float
    of the same number, the exact cut is kept, whichever came first; and a point whose two cuts are one
    exact and one float is the exact point

    >>> normalize([piece(-1, 1.0), piece(-1.0, 1)]) == normalize([piece(-1.0, 1), piece(-1, 1.0)])
    True
    >>> normalize([piece(-1.0, 1), piece(-1, 1.0)])
    (Cut(-1, BELOW), Cut(1, ABOVE))
    >>> normalize([piece(0.0, 0)])
    (Cut(0, BELOW), Cut(0, ABOVE))
    """
    # the hot path of every operation: a cut's value is read as `cut[0]`, and a type is compared before a value
    # (a tie costs about 1.2x the plain sweep's time, measured 2026-10-08)
    out = []
    for start, end in sorted(pairs):
        if start >= end:
            continue
        if out and start <= out[-1]:
            last = out[-1]
            if end > last:
                out[-1] = end
            elif type(last[0]) is float and type(end[0]) is not float and end == last:
                out[-1] = end
                _exact_point(out)
            first = out[-2]
            if type(first[0]) is float and type(start[0]) is not float and start == first:
                out[-2] = start
                _exact_point(out)
        else:
            out.append(start)
            out.append(end)
            if type(start[0]) is not type(end[0]) and start[0] == end[0]:
                _exact_point(out)
    return tuple(out)


def _tie(kept: Cut, other: Cut) -> Cut:
    """of two equal cuts, the exact one (D32); `kept` if both are exact or both floats"""
    return other if type(kept[0]) is float and type(other[0]) is not float else kept


def _exact_point(out: list) -> None:
    """the last piece of `out`, if a point of an exact and a float cut, made the exact point (D32)"""
    start, end = out[-2], out[-1]
    if start[0] == end[0] and (type(start[0]) is float) != (type(end[0]) is float):
        value = end[0] if type(start[0]) is float else start[0]
        out[-2], out[-1] = below(value), above(value)


def _exact_points(out: list) -> Cuts:
    """the cut list as a tuple, each point piece of an exact and a float cut made the exact point (D32)"""
    for i in range(0, len(out), 2):
        start, end = out[i], out[i + 1]
        if type(start[0]) is not type(end[0]) and start[0] == end[0]:
            value = end[0] if type(start[0]) is float else start[0]
            out[i], out[i + 1] = below(value), above(value)
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
    """
    the representation invariant: a tuple of Cuts, even length, strictly increasing, and no point whose
    two cuts are one exact and one float value of the same number (`normalize` makes it the exact point, D32)

    >>> is_valid((below(2.0), above(2))), is_valid((below(2.0), above(3)))
    (False, True)
    """
    return (isinstance(cuts, tuple)
            and len(cuts) % 2 == 0
            and all(isinstance(cut, Cut) for cut in cuts)
            and all(a < b for a, b in zip(cuts, cuts[1:]))
            and all((type(lo.value) is float) == (type(hi.value) is float)
                    for lo, hi in pairs(cuts) if lo.value == hi.value))


class Builder:
    """
    incremental construction: collect pieces, then sort and sweep once in `build()`

    (compare.py measured one timsort-and-sweep at the end to be the cheap way to build in bulk)
    """
    __slots__ = ('_pairs',)

    def __init__(self):
        self._pairs = []

    def add_piece(self, lo, hi, lo_closed: bool = True, hi_closed: bool = True) -> 'Builder':
        self._pairs.append(checked_piece(lo, hi, lo_closed, hi_closed))
        return self

    def add_point(self, value) -> 'Builder':
        return self.add_piece(value, value)

    def add(self, cuts: Cuts) -> 'Builder':
        self._pairs.extend(pairs(cuts))
        return self

    def build(self) -> Cuts:
        return normalize(self._pairs)


# SET ALGEBRA

def sweep(cut_tuples: Iterable[Cuts], keep: Callable[[int], bool], first_wins: bool = False) -> Cuts:
    """
    the points covered by a number of the operands for which `keep(number)` is true

    every operand must be normalized, so each covers a point at most once. between two consecutive
    distinct cuts there is always at least one point (even between `(v, BELOW)` and `(v, ABOVE)`:
    the point v), so the depth after each group of equal cuts is the depth of a real region. of a group
    of equal cuts, an exact one is kept over a float of the same number (D32), or with `first_wins` the
    first operand's (`restrict`)

    >>> intersection(normalize([piece(0.0, 1)]), normalize([piece(0, math.inf)]))
    (Cut(0, BELOW), Cut(1, ABOVE))
    """
    events = []
    for i, cuts in enumerate(cut_tuples):
        mine = first_wins and i == 0
        for start, end in pairs(cuts):
            events.append((start, 1, mine))
            events.append((end, -1, mine))
    events.sort()

    out = []
    depth = 0
    inside = keep(0)
    if inside:
        out.append(REALS[0])
    for cut, group in groupby(events, key=lambda event: event[0]):
        own = None
        for other, delta, mine in group:
            depth += delta
            if mine:
                own = other
            elif type(cut[0]) is float and type(other[0]) is not float:  # `_tie`, inline: once per event
                cut = other
        if own is not None:
            cut = own
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


def restrict(a: Cuts, region: Cuts) -> Cuts:
    """
    `a ∩ region` where a cut `a` already has keeps its type: the library's clip of an operand to a domain or a
    part of the line (a constant of exact cuts). the clip point a region adds is the region's, exact, and a
    point of an exact and a float cut is exact, as in `intersection` (D32: `acos` clips
    `[-1.0000000000000002, -1.0]` to the exact point -1); an end the operand has stays its own (`[-1.0]` stays
    the float point, so `acos` of it is `[3.141592653589793]` to nearest)

    >>> unit = normalize([piece(-1, 1)])
    >>> restrict(normalize([piece(-1.0, 2.0)]), unit), intersection(normalize([piece(-1.0, 2.0)]), unit)
    ((Cut(-1.0, BELOW), Cut(1, ABOVE)), (Cut(-1, BELOW), Cut(1, ABOVE)))
    >>> restrict(normalize([piece(-1.0000000000000002, -1.0)]), unit)
    (Cut(-1, BELOW), Cut(-1, ABOVE))
    """
    return sweep((a, region), lambda depth: depth == 2, first_wins=True)


def symmetric_difference(*cut_tuples: Cuts) -> Cuts:
    """points covered by an odd number of operands"""
    return sweep(cut_tuples, lambda depth: depth % 2 == 1)


def complement(cuts: Cuts) -> Cuts:
    """
    prepend `[-inf`, append `inf]` and re-pair with the roles shifted: every start becomes an end
    and vice versa. pairs that come out empty (`start >= end`) drop out
    """
    shifted = (REALS[0], *cuts, REALS[1])
    return _exact_points([cut for start, end in pairs(shifted) if start < end for cut in (start, end)])


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
