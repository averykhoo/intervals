"""
the MultiInterval class: an immutable cut tuple with one-line dunders over the kernel

set algebra is `| & ^ ~` plus named methods; `-` is reserved for arithmetic subtraction (as in v1),
so set difference is the `difference()` method. `==` and `hash` are structural set equality and do
not coerce scalars: `MultiInterval(5) == 5` is False, because `hash(MultiInterval(5))` cannot equal
`hash(5)` for every such pair.
"""
import math
from numbers import Integral
from numbers import Real
from typing import Iterable
from typing import Iterator
from typing import Set
from typing import Tuple

from intervals import fmt
from intervals import kernel
from intervals import modulo
from intervals import ops
from intervals import relations
from intervals.cuts import Value
from intervals.cuts import above
from intervals.cuts import as_end
from intervals.cuts import as_start
from intervals.cuts import below
from intervals.cuts import normalize_value
from intervals.kernel import Cuts
from intervals.kernel import Size
from intervals.relations import Allen
from intervals.relations import TruthSet


class MultiInterval:
    """
    zero or more disjoint pieces of the affine extended reals, each with open or closed ends

    >>> MultiInterval(1, 2, end_closed=False) | MultiInterval(2, 3)
    MultiInterval.parse('[1, 3]')
    >>> print(MultiInterval.parse('[1, 2) | (2, 3]'))
    { [1, 2) , (2, 3] }
    """
    __slots__ = ('_cuts',)
    _cuts: Cuts
    # numpy's opt-out, so `np.float64(2) * A` runs the reflected dunders instead of treating A as
    # a sequence (this is not the deferred numpy compat)
    __array_ufunc__ = None

    # CONSTRUCTION

    def __init__(self, start=None, end=None, *, start_closed: bool = True, end_closed: bool = True):
        """
        `MultiInterval()` is empty, `MultiInterval(x)` is the point `[x]`, and
        `MultiInterval(a, b, start_closed=..., end_closed=...)` is one piece. an infinite bound is
        taken literally: `MultiInterval(1, inf)` is `[1, inf]`, which contains inf
        """
        if start is None:
            if end is not None:
                raise ValueError('an end without a start')
            cuts = kernel.EMPTY
        elif end is None:
            if start_closed != end_closed:
                raise ValueError(f'half-open degenerate interval at {start!r}')
            cuts = kernel.normalize([kernel.piece(start, start, start_closed, end_closed)])
        else:
            cuts = kernel.normalize([kernel.piece(start, end, start_closed, end_closed)])
        object.__setattr__(self, '_cuts', cuts)

    @classmethod
    def _wrap(cls, cuts: Cuts) -> 'MultiInterval':
        """internal: wrap a cut tuple the kernel produced (checked only under __debug__)"""
        if __debug__:
            assert kernel.is_valid(cuts), cuts
        out = object.__new__(cls)
        object.__setattr__(out, '_cuts', cuts)
        return out

    @classmethod
    def from_cuts(cls, cuts: Iterable) -> 'MultiInterval':
        """wrap a normalized cut tuple; raises ValueError if it is not one"""
        cuts = tuple(cuts)
        if not kernel.is_valid(cuts):
            raise ValueError(f'not a normalized cut tuple: {cuts!r}')
        return cls._wrap(cuts)

    @classmethod
    def from_pieces(cls, pieces: Iterable[tuple]) -> 'MultiInterval':
        """the union of `(lo, hi)` or `(lo, hi, lo_closed, hi_closed)` tuples"""
        return cls._wrap(kernel.normalize(kernel.piece(*p) for p in pieces))

    @classmethod
    def parse(cls, text: str) -> 'MultiInterval':
        """see `intervals.fmt` for the grammar; strings are never coerced implicitly"""
        return cls._wrap(fmt.parse(text))

    @classmethod
    def _coerce(cls, other) -> 'MultiInterval':
        """numbers and MultiIntervals only; anything else is NotImplemented"""
        if isinstance(other, MultiInterval):
            return other
        if isinstance(other, Real) and not isinstance(other, bool):
            return cls(other)
        return NotImplemented

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __reduce__(self):
        return type(self).from_cuts, (self._cuts,)

    @property
    def cuts(self) -> Cuts:
        return self._cuts

    # SET ALGEBRA

    def union(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.union(self._cuts, *self._coerce_all(others)))

    def intersection(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.intersection(self._cuts, *self._coerce_all(others)))

    def difference(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.difference(self._cuts, *self._coerce_all(others)))

    def symmetric_difference(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.symmetric_difference(self._cuts, *self._coerce_all(others)))

    def complement(self) -> 'MultiInterval':
        return self._wrap(kernel.complement(self._cuts))

    def issubset(self, other) -> bool:
        return kernel.is_subset(self._cuts, self._coerce_or_raise(other)._cuts)

    def issuperset(self, other) -> bool:
        return kernel.is_subset(self._coerce_or_raise(other)._cuts, self._cuts)

    def isdisjoint(self, other) -> bool:
        return not kernel.intersection(self._cuts, self._coerce_or_raise(other)._cuts)

    def _coerce_all(self, others) -> Tuple[Cuts, ...]:
        return tuple(self._coerce_or_raise(other)._cuts for other in others)

    def _coerce_or_raise(self, other) -> 'MultiInterval':
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            raise TypeError(f'expected a MultiInterval or a real number, got {type(other).__name__}')
        return coerced

    def __or__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else self._wrap(kernel.union(self._cuts, other._cuts))

    __ror__ = __or__

    def __and__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else self._wrap(kernel.intersection(self._cuts, other._cuts))

    __rand__ = __and__

    def __xor__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else self._wrap(
            kernel.symmetric_difference(self._cuts, other._cuts))

    __rxor__ = __xor__

    def __invert__(self) -> 'MultiInterval':
        return self.complement()

    def __contains__(self, item) -> bool:
        """
        a number: membership. a MultiInterval: the documented subset alias, `A in B` iff `A <= B`
        as sets (`<=` itself is the pointwise comparison)
        """
        if isinstance(item, MultiInterval):
            return kernel.is_subset(item._cuts, self._cuts)
        if isinstance(item, Real) and not isinstance(item, bool):
            if item != item:  # nan is not a point of the extended reals
                return False
            return kernel.contains_point(self._cuts, item)
        raise TypeError(f'expected a MultiInterval or a real number, got {type(item).__name__}')

    def __getitem__(self, item: slice) -> 'MultiInterval':
        """`A[a:b]` is the restriction to the closed `[a, b]`; a missing bound is infinite"""
        if not isinstance(item, slice):
            raise TypeError('MultiInterval supports slicing only, e.g. A[0:5]; use `in` for membership')
        if item.step is not None:
            raise TypeError('slice step is not supported')
        start = -math.inf if item.start is None else item.start
        stop = math.inf if item.stop is None else item.stop
        return self._wrap(kernel.intersection(self._cuts, kernel.normalize([kernel.piece(start, stop)])))

    # ARITHMETIC (the set of values attained; see intervals.ops)

    def _binary(self, other, op, reflected=False):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        a, b = (other._cuts, self._cuts) if reflected else (self._cuts, other._cuts)
        return self._wrap(op(a, b))

    def __add__(self, other):
        return self._binary(other, ops.add)

    def __radd__(self, other):
        return self._binary(other, ops.add, reflected=True)

    def __sub__(self, other):
        """arithmetic subtraction; set difference is `difference()`"""
        return self._binary(other, ops.sub)

    def __rsub__(self, other):
        return self._binary(other, ops.sub, reflected=True)

    def __mul__(self, other):
        return self._binary(other, ops.mul)

    def __rmul__(self, other):
        return self._binary(other, ops.mul, reflected=True)

    def __truediv__(self, other):
        """
        >>> 1 / MultiInterval.parse('[-1, 1]')
        MultiInterval.parse('{ [-inf, -1] , [1, inf] }')
        >>> MultiInterval(1) / 3
        MultiInterval.parse('[1/3]')
        """
        return self._binary(other, ops.div)

    def __rtruediv__(self, other):
        return self._binary(other, ops.div, reflected=True)

    def __neg__(self) -> 'MultiInterval':
        return self._wrap(ops.neg(self._cuts))

    def __pos__(self) -> 'MultiInterval':
        return self._wrap(ops.pos(self._cuts))

    def __abs__(self) -> 'MultiInterval':
        return self._wrap(ops.absolute(self._cuts))

    def __pow__(self, exponent, modulo=None):
        """int exponents only (not bool); on exact operands `A ** -n` is `(A ** n).reciprocal()`"""
        if modulo is not None or isinstance(exponent, bool) or not isinstance(exponent, Integral):
            return NotImplemented
        return self._wrap(ops.power(self._cuts, exponent))

    def reciprocal(self) -> 'MultiInterval':
        """
        `1 / self`; the sign of an infinity comes from the side of zero a piece lies on

        >>> MultiInterval.parse('[1, inf)').reciprocal()
        MultiInterval.parse('(0, 1]')
        """
        return self._wrap(ops.reciprocal(self._cuts))

    def __mod__(self, other):
        """
        python's floor-mod over every pair (see intervals.modulo)

        >>> MultiInterval.parse('[3, 7]') % MultiInterval.parse('[-5, -2]')
        MultiInterval.parse('(-5, 0]')
        >>> -7 % MultiInterval(2, 5)  # just below y = 7/2, -7 mod y = 2y - 7 nears 7/2
        MultiInterval.parse('[0, 7/2)')
        """
        return self._binary(other, modulo.mod)

    def __rmod__(self, other):
        return self._binary(other, modulo.mod, reflected=True)

    def __floordiv__(self, other):
        """
        `floor(self / other)`, listing the integers it holds (a hull and a HullWarning past
        `modulo.FLOOR_ENUMERATION_CAP` of them)

        >>> MultiInterval(1, 2, end_closed=False) // 1
        MultiInterval.parse('[1]')
        """
        return self._binary(other, modulo.floordiv)

    def __rfloordiv__(self, other):
        return self._binary(other, modulo.floordiv, reflected=True)

    def __divmod__(self, other):
        """`(self // other, self % other)`: two sets, not a set of pairs"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        q, r = modulo.divmod_(self._cuts, other._cuts)
        return self._wrap(q), self._wrap(r)

    def __rdivmod__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return divmod(other, self)

    def floor(self) -> 'MultiInterval':
        """
        the integers `floor(x)` for x in self

        >>> MultiInterval.parse('[-1/2, 2)').floor()
        MultiInterval.parse('{ [-1] , [0] , [1] }')
        """
        return self._wrap(modulo.floor(self._cuts))

    # POINTWISE COMPARISONS (a TruthSet; see intervals.relations)

    def __lt__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else relations.lt(self._cuts, other._cuts)

    def __le__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else relations.le(self._cuts, other._cuts)

    def __gt__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else relations.gt(self._cuts, other._cuts)

    def __ge__(self, other):
        other = self._coerce(other)
        return NotImplemented if other is NotImplemented else relations.ge(self._cuts, other._cuts)

    def eq_pointwise(self, other) -> TruthSet:
        """`{T, F}` for any non-degenerate `A.eq_pointwise(A)`; `==` is structural"""
        return relations.eq_pointwise(self._cuts, self._coerce_or_raise(other)._cuts)

    # SET-LEVEL RELATIONS (bool, decided on cuts)

    def before(self, other) -> bool:
        return relations.before(self._cuts, self._coerce_or_raise(other)._cuts)

    def after(self, other) -> bool:
        return relations.after(self._cuts, self._coerce_or_raise(other)._cuts)

    def adjoins(self, other) -> bool:
        return relations.adjoins(self._cuts, self._coerce_or_raise(other)._cuts)

    def overlaps(self, other) -> bool:
        return relations.overlaps(self._cuts, self._coerce_or_raise(other)._cuts)

    def contains(self, other) -> bool:
        return relations.contains(self._cuts, self._coerce_or_raise(other)._cuts)

    def within(self, other) -> bool:
        return relations.within(self._cuts, self._coerce_or_raise(other)._cuts)

    def allen(self, other) -> Allen:
        """allen's relation between two contiguous MultiIntervals (ValueError otherwise)"""
        return relations.allen(self._cuts, self._coerce_or_raise(other)._cuts)

    # EQUALITY, HASHING, CONTAINER PROTOCOL

    def __eq__(self, other):
        if not isinstance(other, MultiInterval):
            return NotImplemented
        return self._cuts == other._cuts

    def __ne__(self, other):
        if not isinstance(other, MultiInterval):
            return NotImplemented
        return self._cuts != other._cuts

    def __hash__(self):
        return hash(self._cuts)

    @property
    def sort_key(self) -> Cuts:
        """structural order, for `sorted(xs, key=lambda x: x.sort_key)`"""
        return self._cuts

    def __bool__(self) -> bool:
        return bool(self._cuts)

    def __len__(self) -> int:
        """the number of pieces"""
        return len(self._cuts) // 2

    def __iter__(self) -> Iterator['MultiInterval']:
        """the pieces, each a contiguous MultiInterval"""
        for start, end in kernel.pairs(self._cuts):
            yield self._wrap((start, end))

    @property
    def pieces(self) -> Tuple['MultiInterval', ...]:
        return tuple(self)

    # FORMATTING

    def __repr__(self) -> str:
        return f'{type(self).__name__}.parse({fmt.format_cuts(self._cuts)!r})'

    def __str__(self) -> str:
        return fmt.format_cuts(self._cuts)

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        return not self._cuts

    @property
    def is_contiguous(self) -> bool:
        """exactly one piece (a degenerate point counts)"""
        return len(self._cuts) == 2

    @property
    def is_degenerate(self) -> bool:
        """non-empty, and every piece is a single point"""
        return bool(self._cuts) and all(lo == hi for lo, _, hi, _ in kernel.pieces(self._cuts))

    @property
    def is_finite(self) -> bool:
        """no point at, and no piece reaching, ±inf (the empty set is finite)"""
        return not self._cuts or (math.isfinite(self._cuts[0].value) and math.isfinite(self._cuts[-1].value))

    @property
    def is_integral(self) -> bool:
        """a non-empty set of finite integer points"""
        return self.is_degenerate and self.is_finite and all(
            lo % 1 == 0 for lo, _, _, _ in kernel.pieces(self._cuts))

    @property
    def is_positive(self) -> bool:
        return bool(self._cuts) and self._cuts[0] >= above(0)

    @property
    def is_negative(self) -> bool:
        return bool(self._cuts) and self._cuts[-1] <= below(0)

    @property
    def is_non_negative(self) -> bool:
        return not self._cuts or self._cuts[0] >= below(0)

    @property
    def is_non_positive(self) -> bool:
        return not self._cuts or self._cuts[-1] <= above(0)

    @property
    def finite(self) -> 'MultiInterval':
        """the pieces that neither touch nor reach ±inf (as v1: a piece `[5, inf]` is dropped whole)"""
        return self._wrap(tuple(cut for start, end in kernel.pairs(self._cuts)
                                if math.isfinite(start.value) and math.isfinite(end.value)
                                for cut in (start, end)))

    @property
    def positive(self) -> 'MultiInterval':
        """the points > 0, including inf"""
        return self._wrap(kernel.intersection(self._cuts, (above(0), above(math.inf))))

    @property
    def negative(self) -> 'MultiInterval':
        """the points < 0, including -inf"""
        return self._wrap(kernel.intersection(self._cuts, (below(-math.inf), below(0))))

    @property
    def inf(self) -> Value:
        """the infimum; ValueError when empty, like `min([])`"""
        return as_start(self._first_cut())[0]

    @property
    def inf_closed(self) -> bool:
        return as_start(self._first_cut())[1]

    @property
    def sup(self) -> Value:
        return as_end(self._last_cut())[0]

    @property
    def sup_closed(self) -> bool:
        return as_end(self._last_cut())[1]

    def _first_cut(self):
        if not self._cuts:
            raise ValueError('the empty set has no infimum or supremum')
        return self._cuts[0]

    def _last_cut(self):
        if not self._cuts:
            raise ValueError('the empty set has no infimum or supremum')
        return self._cuts[-1]

    @property
    def degenerate_points(self) -> Set[Value]:
        return {lo for lo, _, hi, _ in kernel.pieces(self._cuts) if lo == hi}

    @property
    def hull(self) -> 'MultiInterval':
        """the smallest contiguous superset (endpoint flags kept)"""
        return self._wrap(kernel.hull(self._cuts))

    @property
    def closed_hull(self) -> 'MultiInterval':
        """the closure of the hull: both endpoints closed, including at ±inf"""
        if not self._cuts:
            return self
        return self._wrap((below(self._cuts[0].value), above(self._cuts[-1].value)))

    @property
    def size(self) -> Size:
        return kernel.size(self._cuts)

    def expand(self, distance) -> 'MultiInterval':
        """widen every piece by a finite `distance >= 0` on both sides, keeping its endpoint flags"""
        distance = normalize_value(distance)
        if not (0 <= distance < math.inf):
            raise ValueError('expand() needs a finite, non-negative distance')
        return self._wrap(kernel.normalize(
            kernel.piece(lo - distance, hi + distance, lo_closed, hi_closed)
            for lo, lo_closed, hi, hi_closed in kernel.pieces(self._cuts)))

    # CONVERSIONS (single points only)

    def _point(self) -> Value:
        if not (self.is_degenerate and self.is_contiguous):
            raise ValueError(f'{self} is not a single point')
        return self._cuts[0].value

    def __float__(self) -> float:
        return float(self._point())

    def __int__(self) -> int:
        return int(self._point())

    def __complex__(self) -> complex:
        return complex(self._point())
