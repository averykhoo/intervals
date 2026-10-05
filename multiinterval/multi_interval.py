"""
the MultiInterval class: an immutable cut tuple with one-line dunders over the kernel

set algebra is `| & ^ ~` plus named methods; `-` is reserved for arithmetic subtraction (as in v1),
so set difference is the `difference()` method. `==` and `hash` are structural set equality and do
not coerce scalars: `MultiInterval(5) == 5` is False, because `hash(MultiInterval(5))` cannot equal
`hash(5)` for every such pair.

float arithmetic rounds to nearest; `OutwardMultiInterval` is the same class with outward rounding,
so its results enclose the exact result. the rounding is a property of the type, never a flag.
mixing the two, by an operator or by a method taking another set, gives an `OutwardMultiInterval`.
"""
import functools
import math
from fractions import Fraction
from numbers import Integral
from numbers import Real
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import Set
from typing import Tuple

from multiinterval import fmt
from multiinterval import functions
from multiinterval import kernel
from multiinterval import modulo
from multiinterval import numeric
from multiinterval import numpy_compat
from multiinterval import ops
from multiinterval import relations
from multiinterval import rounding
from multiinterval import steps
from multiinterval.cuts import Value
from multiinterval.cuts import above
from multiinterval.cuts import as_end
from multiinterval.cuts import as_start
from multiinterval.cuts import below
from multiinterval.cuts import flag
from multiinterval.cuts import is_numpy_time
from multiinterval.cuts import normalize_value
from multiinterval.kernel import Cuts
from multiinterval.kernel import Size
from multiinterval.relations import Allen
from multiinterval.relations import TruthSet


def _subclass_decides(method):
    """
    a method taking other sets computes in the class the operators would give: the receiver's,
    unless an operand's class is a proper subclass of it, which then decides, as python lets the
    right operand of `M + O` decide (owner, 2026-10-03, Q15(h)). the receiver is promoted to that
    class and the class's own method runs: `M(0.1).hypot(O(0.1))` is `O(0.1).hypot(O(0.1))`. a number
    operand decides nothing (it becomes a point of the receiver's class)
    """
    name = method.__name__

    @functools.wraps(method)
    def deciding(self, *args, **kwargs):
        cls = type(self)
        for other in (*args, *kwargs.values()):
            if isinstance(other, cls) and type(other) is not cls:
                cls = type(other)
        if cls is not type(self):
            return getattr(cls._wrap(self._cuts), name)(*args, **kwargs)
        return method(self, *args, **kwargs)
    return deciding


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
    # float results round to nearest here and outward in OutwardMultiInterval
    _outward = False
    # numpy interop (M16d): ufuncs are python's operators or the methods of the same set, and an
    # array holds a MultiInterval as one element, never as the sequence of its pieces
    __array_ufunc__ = numpy_compat.array_ufunc
    __array__ = numpy_compat.array

    # CONSTRUCTION

    def __init__(self, start=None, end=None, *, start_closed: bool = True, end_closed: bool = True):
        """
        `MultiInterval()` is empty, `MultiInterval(x)` is the point `[x]`, and
        `MultiInterval(a, b, start_closed=..., end_closed=...)` is one piece. an infinite bound is
        taken literally: `MultiInterval(1, inf)` is `[1, inf]`, which contains inf. a flag is a bool, else
        a TypeError
        """
        start_closed, end_closed = flag(start_closed, 'start_closed'), flag(end_closed, 'end_closed')
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
        """the union of `(lo, hi)` or `(lo, hi, lo_closed, hi_closed)` tuples; a flag is a bool, else a TypeError"""
        return cls._wrap(kernel.normalize(kernel.checked_piece(*p) for p in pieces))

    @classmethod
    def parse(cls, text: str) -> 'MultiInterval':
        """see `multiinterval.fmt` for the grammar; strings are never coerced implicitly"""
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

    # SET ALGEBRA (a method taking other sets has the operators' result class: `_subclass_decides`)

    @_subclass_decides
    def union(self, *others) -> 'MultiInterval':
        """
        the union of self and every other; mixed with an `OutwardMultiInterval`, one, as `|` gives

        >>> MultiInterval(1).union(OutwardMultiInterval(2), 3)
        OutwardMultiInterval.parse('{ [1] , [2] , [3] }')
        """
        return self._wrap(kernel.union(self._cuts, *self._coerce_all(others)))

    @_subclass_decides
    def intersection(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.intersection(self._cuts, *self._coerce_all(others)))

    @_subclass_decides
    def difference(self, *others) -> 'MultiInterval':
        return self._wrap(kernel.difference(self._cuts, *self._coerce_all(others)))

    @_subclass_decides
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
        if isinstance(item, Real) and not isinstance(item, bool) and not is_numpy_time(item):
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

    # ARITHMETIC (the set of values attained; see multiinterval.ops)

    def _binary(self, other, op, reflected=False):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        a, b = (other._cuts, self._cuts) if reflected else (self._cuts, other._cuts)
        return self._wrap(op(a, b, outward=self._outward))

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
        """
        a number with an integral value (int, or a float or Fraction equal to one; not bool) is ieee
        1788's pown, as for python's numbers: every base, and on exact operands `A ** -n` is
        `(A ** n).reciprocal()`. any other real exponent, and every `MultiInterval` one (even `[2]`), is
        1788's pow (`functions.pow_`): `{x ** y}` over the bases x > 0, and x = 0 where y > 0, the
        others dropped with a `DomainClippedWarning`. 3-argument `pow` is a TypeError

        >>> MultiInterval(-3, 1) ** 2.0
        MultiInterval.parse('[0, 9]')
        >>> MultiInterval(1, 4) ** MultiInterval.parse('[1/2]')
        MultiInterval.parse('[1, 2]')
        """
        if modulo is not None or isinstance(exponent, bool):
            return NotImplemented
        if isinstance(exponent, MultiInterval):
            return self._wrap(functions.pow_(self._cuts, exponent._cuts, outward=self._outward))
        if not isinstance(exponent, Real):
            return NotImplemented
        if _is_integral(exponent):
            return self._wrap(ops.power(self._cuts, int(exponent), outward=self._outward))
        return self._wrap(functions.pow_(self._cuts, self._coerce_or_raise(exponent)._cuts, outward=self._outward))

    def __rpow__(self, base):
        """`b ** A` for a real b is `MultiInterval(b) ** A`, 1788's pow (an interval exponent)"""
        other = self._coerce(base)
        if other is NotImplemented:
            return NotImplemented
        return self._wrap(functions.pow_(other._cuts, self._cuts, outward=self._outward))

    def reciprocal(self) -> 'MultiInterval':
        """
        `1 / self`; the sign of an infinity comes from the side of zero a piece lies on

        >>> MultiInterval.parse('[1, inf)').reciprocal()
        MultiInterval.parse('(0, 1]')
        """
        return self._wrap(ops.reciprocal(self._cuts, outward=self._outward))

    def __mod__(self, other):
        """
        python's floor-mod over every pair (see multiinterval.modulo)

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
        q, r = modulo.divmod_(self._cuts, other._cuts, outward=self._outward)
        return self._wrap(q), self._wrap(r)

    def __rdivmod__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        q, r = modulo.divmod_(other._cuts, self._cuts, outward=self._outward)
        return self._wrap(q), self._wrap(r)

    def floor(self) -> 'MultiInterval':
        """
        the integers `floor(x)` for x in self, listed up to `steps.ENUMERATION_CAP` of them (then their
        hull, with a HullWarning); `math.floor(A)` is the same

        >>> MultiInterval.parse('[-1/2, 2)').floor()
        MultiInterval.parse('{ [-1] , [0] , [1] }')
        """
        return self._wrap(modulo.floor(self._cuts, outward=self._outward))

    def ceil(self) -> 'MultiInterval':
        """the integers `ceil(x)` for x in self (as `floor()`); `math.ceil(A)` is the same"""
        return self._wrap(steps.ceil(self._cuts, outward=self._outward))

    def trunc(self) -> 'MultiInterval':
        """floor above 0 and ceil below; `math.trunc(A)` is the same"""
        return self._wrap(steps.trunc(self._cuts, outward=self._outward))

    def round(self, ndigits=None) -> 'MultiInterval':
        """
        `round(x, ndigits)` for x in self, ties to even as python rounds; `round(A)` is the same

        >>> round(MultiInterval.parse('[1/2, 5/2]'))
        MultiInterval.parse('{ [0] , [1] , [2] }')
        """
        return self._wrap(steps.round_(self._cuts, ndigits, outward=self._outward))

    def round_ties_away(self, ndigits=None) -> 'MultiInterval':
        """as `round()`, with ties away from zero (ieee 1788 roundTiesToAway)"""
        return self._wrap(steps.round_ties_away(self._cuts, ndigits, outward=self._outward))

    def sign(self) -> 'MultiInterval':
        """
        the signs attained, among -1, 0 and 1 (`sign(±inf)` = ±1)

        >>> MultiInterval.parse('[-2, 0]').sign()
        MultiInterval.parse('{ [-1] , [0] }')
        """
        return self._wrap(steps.sign(self._cuts))

    def __floor__(self) -> 'MultiInterval':
        return self.floor()

    def __ceil__(self) -> 'MultiInterval':
        return self.ceil()

    def __trunc__(self) -> 'MultiInterval':
        return self.trunc()

    def __round__(self, ndigits=None) -> 'MultiInterval':
        return self.round(ndigits)

    @_subclass_decides
    def minimum(self, other) -> 'MultiInterval':
        """
        `{min(x, y) : x in self, y in other}` (builtin `min` needs a bool from `<`, which is pointwise)

        >>> MultiInterval(0, 10).minimum(MultiInterval(3, 5))
        MultiInterval.parse('[0, 5]')
        """
        return self._wrap(ops.minimum(self._cuts, self._coerce_or_raise(other)._cuts))

    @_subclass_decides
    def maximum(self, other) -> 'MultiInterval':
        """`{max(x, y) : x in self, y in other}`"""
        return self._wrap(ops.maximum(self._cuts, self._coerce_or_raise(other)._cuts))

    @_subclass_decides
    def fma(self, factor, addend) -> 'MultiInterval':
        """
        `self * factor + addend`, rounded once (ieee 1788 fma), outward if any operand is an
        `OutwardMultiInterval`

        >>> MultiInterval(0.1).fma(OutwardMultiInterval(3), 0)
        OutwardMultiInterval.parse('(0.3, 0.30000000000000004)')
        """
        return self._wrap(ops.fma(self._cuts, self._coerce_or_raise(factor)._cuts,
                                  self._coerce_or_raise(addend)._cuts, outward=self._outward))

    @_subclass_decides
    def cancel_minus(self, other) -> 'MultiInterval':
        """
        the largest `X` with `other + X ⊆ self`, the Minkowski difference (ieee 1788's
        `cancelMinus`). any two sets have one: `∅` when nothing fits, `[-inf, inf]` when `other` is
        empty. exact for exact operands; a float operand rounds it once, to nearest, or outward if
        either operand is an `OutwardMultiInterval` (an enclosure of `X`, as 1788 gives). derivation in
        `multiinterval.ops.cancel_minus`

        >>> A, B = MultiInterval(0, 10), MultiInterval(1, 3)
        >>> X = A.cancel_minus(B)
        >>> X, (B + X).issubset(A)
        (MultiInterval.parse('[-1, 7]'), True)
        >>> MultiInterval(0, 1).cancel_minus(MultiInterval(0, 2))  # 1788 answers entire here
        MultiInterval.parse('{}')
        >>> MultiInterval.parse('[0, 1] | [10, 12]').cancel_minus(MultiInterval.parse('[0] | [10, 11]'))
        MultiInterval.parse('[0, 1]')
        >>> OutwardMultiInterval(1.9).cancel_minus(0.1)  # the two doubles around the exact difference
        OutwardMultiInterval.parse('(1.7999999999999998, 1.8)')
        """
        return self._wrap(ops.cancel_minus(self._cuts, self._coerce_or_raise(other)._cuts, outward=self._outward))

    @_subclass_decides
    def cancel_plus(self, other) -> 'MultiInterval':
        """
        `self.cancel_minus(-other)`: the largest `X` with `X - other ⊆ self` (ieee 1788's `cancelPlus`)

        >>> MultiInterval(0, 10).cancel_plus(MultiInterval(1, 3))
        MultiInterval.parse('[3, 11]')
        """
        return self._wrap(ops.cancel_plus(self._cuts, self._coerce_or_raise(other)._cuts, outward=self._outward))

    # ELEMENTARY FUNCTIONS (see multiinterval.functions)

    def _function(self, name: str, base=None) -> 'MultiInterval':
        return self._wrap(functions.apply(name, self._cuts, outward=self._outward, base=base))

    def sqrt(self) -> 'MultiInterval':
        """
        points below 0 are dropped with a DomainClippedWarning; an irrational value of an exact
        point is its tightest float enclosure, open because neither double is attained

        >>> MultiInterval.parse('[1/4, 9]').sqrt()
        MultiInterval.parse('[1/2, 3]')
        >>> MultiInterval(2).sqrt()
        MultiInterval.parse('(1.414213562373095, 1.4142135623730951)')
        """
        return self._function('sqrt')

    def exp(self) -> 'MultiInterval':
        """`exp(-inf)` = 0 and `exp(inf)` = inf"""
        return self._function('exp')

    def exp2(self) -> 'MultiInterval':
        return self._function('exp2')

    def exp10(self) -> 'MultiInterval':
        return self._function('exp10')

    def log(self, base=None) -> 'MultiInterval':
        """
        the natural logarithm, or to a finite `base` > 0 other than 1; `log(0)` = -inf

        >>> MultiInterval(1, 8).log(2)
        MultiInterval.parse('[0, 3]')
        """
        return self._function('log', base)

    def log2(self) -> 'MultiInterval':
        return self._function('log2')

    def log10(self) -> 'MultiInterval':
        return self._function('log10')

    def sin(self) -> 'MultiInterval':
        """±inf are dropped with a DomainClippedWarning (no limit there)"""
        return self._function('sin')

    def cos(self) -> 'MultiInterval':
        return self._function('cos')

    def tan(self) -> 'MultiInterval':
        """a piece holding a pole maps to both sides of it, with -inf and inf attained"""
        return self._function('tan')

    def asin(self) -> 'MultiInterval':
        return self._function('asin')

    def acos(self) -> 'MultiInterval':
        return self._function('acos')

    def atan(self) -> 'MultiInterval':
        return self._function('atan')

    def sinh(self) -> 'MultiInterval':
        return self._function('sinh')

    def cosh(self) -> 'MultiInterval':
        return self._function('cosh')

    def tanh(self) -> 'MultiInterval':
        return self._function('tanh')

    def asinh(self) -> 'MultiInterval':
        return self._function('asinh')

    def acosh(self) -> 'MultiInterval':
        return self._function('acosh')

    def atanh(self) -> 'MultiInterval':
        """`atanh(±1)` = ±inf"""
        return self._function('atanh')

    def expm1(self) -> 'MultiInterval':
        """`exp(x) - 1`, correctly rounded near 0 as well; `expm1(-inf)` = -1"""
        return self._function('expm1')

    def log1p(self) -> 'MultiInterval':
        """`log(1 + x)` (ieee 1788's logp1); points below -1 are dropped, and `log1p(-1)` = -inf"""
        return self._function('log1p')

    def cbrt(self) -> 'MultiInterval':
        """
        the real cube root, negative bases included

        >>> MultiInterval(-8, 27).cbrt()
        MultiInterval.parse('[-2, 3]')
        """
        return self._function('cbrt')

    def rootn(self, n: int) -> 'MultiInterval':
        """
        the real n-th root for an int n other than 0, of `1/x` for n < 0 (ieee 1788's rootn); an even
        root drops the points below 0, and an odd negative one has a pole at 0, as `reciprocal()`

        >>> MultiInterval(0, 16).rootn(4)
        MultiInterval.parse('[0, 2]')
        >>> MultiInterval(1, 8).rootn(-3)
        MultiInterval.parse('[1/2, 1]')
        """
        return self._function('rootn', n)

    @_subclass_decides
    def hypot(self, other) -> 'MultiInterval':
        """
        `sqrt(x**2 + y**2)` for x in self and y in other, rounded once (outward if either operand is
        an `OutwardMultiInterval`, so `np.hypot` gives the same in either order)

        >>> MultiInterval(3).hypot(MultiInterval(-4, 0))
        MultiInterval.parse('[3, 5]')
        >>> MultiInterval(0.1).hypot(OutwardMultiInterval(0.1))
        OutwardMultiInterval.parse('(0.1414213562373095, 0.14142135623730953)')
        """
        return self._wrap(functions.hypot(self._cuts, self._coerce_or_raise(other)._cuts, outward=self._outward))

    def cot(self) -> 'MultiInterval':
        """poles at every k pi, 0 included: a piece ending at 0 takes the limit on its side"""
        return self._function('cot')

    def sec(self) -> 'MultiInterval':
        return self._function('sec')

    def csc(self) -> 'MultiInterval':
        """poles at every k pi, 0 included: a piece ending at 0 takes the limit on its side"""
        return self._function('csc')

    def acot(self) -> 'MultiInterval':
        """`pi/2 - atan(x)`, continuous and falling from pi at -inf to 0 at inf"""
        return self._function('acot')

    def coth(self) -> 'MultiInterval':
        """a pole at 0: `coth([0, 1])` is `[coth 1, inf]`, and `coth([0])` is empty with a warning"""
        return self._function('coth')

    def csch(self) -> 'MultiInterval':
        """a pole at 0, as `coth`"""
        return self._function('csch')

    def sech(self) -> 'MultiInterval':
        return self._function('sech')

    def acoth(self) -> 'MultiInterval':
        """defined for `|x| >= 1`; `acoth(±1)` = ±inf and `acoth(±inf)` = 0"""
        return self._function('acoth')

    @_subclass_decides
    def atan2(self, x) -> 'MultiInterval':
        """
        the angles `atan2(y, x)` for y in self, in [-pi, pi]; `atan2(0, x < 0)` is pi, as there is no -0

        >>> MultiInterval(1).atan2(MultiInterval(0, math.inf))
        MultiInterval.parse('[0, 1.5707963267948968)')
        """
        return self._wrap(functions.atan2(self._cuts, self._coerce_or_raise(x)._cuts, outward=self._outward))

    # POINTWISE COMPARISONS (a TruthSet; see multiinterval.relations)

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

    def allen_matrix(self, other) -> Tuple[Tuple[Allen, ...], ...]:
        """
        `allen()` of every pair of pieces, a row per piece of self and a column per piece of
        other: `A.allen_matrix(B)[i][j] is A.pieces[i].allen(B.pieces[j])`. an empty operand has
        no pairs, so no rows or empty rows

        >>> A, B = MultiInterval.parse('[0, 1] | [3, 5]'), MultiInterval(1, 4)
        >>> A.allen_matrix(B)  # [0, 1] and [1, 4] share the point 1: OVERLAPS, on cuts
        ((<Allen.OVERLAPS: 'overlaps'>,), (<Allen.OVERLAPPED_BY: 'overlapped by'>,))
        >>> A.allen_matrix(MultiInterval()), MultiInterval().allen_matrix(A)
        (((), ()), ())
        """
        return relations.allen_matrix(self._cuts, self._coerce_or_raise(other)._cuts)

    def allen_relations(self, other) -> FrozenSet[Allen]:
        """
        the relations holding between some piece of self and some piece of other: the entries of
        `allen_matrix`, found in `O(n + m)` without building it; `frozenset()` if either is empty.
        extensional: each relation in it holds between some pair of pieces; not allen's algebra's
        disjunction ("one of these holds"), though it has that type

        >>> A = MultiInterval.parse('[0, 1] | [4, 5]')
        >>> sorted(r.name for r in A.allen_relations(MultiInterval.parse('[2, 3] | [6, 7]')))
        ['AFTER', 'BEFORE']
        >>> A.allen_relations(MultiInterval())
        frozenset()
        """
        return relations.allen_relations(self._cuts, self._coerce_or_raise(other)._cuts)

    # INTERVAL ORDERS (bool, on the ends; ieee 1788's less and strictLess)

    def weakly_less(self, other) -> bool:
        """
        `inf <= other.inf` and `sup <= other.sup`: 1788's `less`, on the ends, so the hull's. two
        empty sets are weakly less than each other, an empty and a non-empty one are not. `<=` is
        pointwise and says something else

        >>> A, B = MultiInterval(1, 3), MultiInterval(2, 4)
        >>> A.weakly_less(B), B.weakly_less(A), A <= B
        (True, False, BOTH)
        >>> MultiInterval.parse('[0, 1] | [5, 6]').weakly_less(MultiInterval(2, 6))
        True
        >>> MultiInterval().weakly_less(MultiInterval()), MultiInterval().weakly_less(A)
        (True, False)
        """
        return relations.weakly_less(self._cuts, self._coerce_or_raise(other)._cuts)

    def strictly_less(self, other) -> bool:
        """
        `inf < other.inf` and `sup < other.sup`: 1788's `strictLess`, on the ends, so the hull's.
        as in 1788, two starts at -inf and two ends at inf count as less, so `(-inf, inf)` is
        strictly less than itself; open or closed does not matter. empty sets as `weakly_less`

        >>> A = MultiInterval(1, 3)
        >>> A.strictly_less(MultiInterval(2, 4)), A.strictly_less(MultiInterval(1, 4)), A.strictly_less(A)
        (True, False, False)
        >>> MultiInterval.parse('(-inf, 1]').strictly_less(MultiInterval.parse('(-inf, 2)'))
        True
        """
        return relations.strictly_less(self._cuts, self._coerce_or_raise(other)._cuts)

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
        # compared, never converted: math.isfinite overflows on an exact end past the doubles (10**400)
        return not self._cuts or (-math.inf < self._cuts[0].value and self._cuts[-1].value < math.inf)

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
                                if -math.inf < start.value and end.value < math.inf
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
    def interior(self) -> 'MultiInterval':
        """
        every end opened, the infinite ones too: the interior in the topology of the reals. a
        degenerate piece drops out, and so does a closed end at ±inf, a point with no neighbourhood
        of reals. 1788's `interior(A, B)` is `A.within(B.interior)`

        >>> MultiInterval.parse('[0, 1] | [2] | [3, inf]').interior
        MultiInterval.parse('{ (0, 1) , (3, inf) }')
        >>> B = MultiInterval(0, 4)
        >>> MultiInterval(1, 2).within(B.interior), B.within(B.interior), MultiInterval(math.inf).interior
        (True, False, MultiInterval.parse('{}'))
        """
        return self._wrap(kernel.interior(self._cuts))

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

    # NUMERIC FUNCTIONS (ieee 1788's mid, rad, wid, mag, mig, midRad; see multiinterval.numeric)

    def mid(self) -> Value:
        """
        the midpoint of the hull, exact for an exact operand and rounded to nearest for a float one.
        the midpoint of `(-inf, inf)` is 0, of a half-bounded set ±max float, as in ieee 1788;
        ValueError when empty, like `inf`

        >>> MultiInterval.parse('[0, 1] | [9, 10]').mid()  # outside the set, still a bisection point
        5
        >>> MultiInterval(1, 2).mid(), MultiInterval(0.1, 0.2).mid()
        (Fraction(3, 2), 0.15000000000000002)
        >>> MultiInterval.parse('[0, inf)').mid()
        1.7976931348623157e+308
        """
        return numeric.mid(self._cuts)

    def rad(self) -> Value:
        """
        the radius of the hull, measured from `mid()`: the smallest `r` with `[mid - r, mid + r]`
        holding the hull, rounded up for a float operand; inf when unbounded

        >>> MultiInterval(1, 2).rad(), MultiInterval.parse('[1, inf)').rad()
        (Fraction(1, 2), inf)
        >>> MultiInterval(1.0, 1.0000000000000007).rad()  # from the rounded midpoint 1.0000000000000004
        4.440892098500626e-16
        """
        return numeric.rad(self._cuts)

    def mid_rad(self) -> Tuple[Value, Value]:
        """
        `(mid(), rad())`: the hull is inside `[mid - rad, mid + rad]`

        >>> MultiInterval.parse('[-2, 0] | [3, 4]').mid_rad()
        (1, 3)
        """
        return numeric.mid_rad(self._cuts)

    def wid(self) -> Value:
        """
        the width of the hull, `sup - inf`, rounded up for a float operand; inf when unbounded.
        `size.length` is the length without the gaps

        >>> A = MultiInterval.parse('[0, 1] | [9, 10]')
        >>> A.wid(), A.size.length
        (10, 2)
        >>> MultiInterval(0.1, 0.3).wid()
        0.19999999999999998
        """
        return numeric.wid(self._cuts)

    def mag(self) -> Value:
        """
        the magnitude, `sup {abs(x) : x in self}`, rounded up for a float operand

        >>> MultiInterval.parse('[-3, -2] | [2, 3)').mag()
        3
        """
        return numeric.mag(self._cuts)

    def mig(self) -> Value:
        """
        the mignitude, `inf {abs(x) : x in self}`, of the set rather than the hull, rounded down for a
        float operand

        >>> A = MultiInterval.parse('[-3, -2] | [2, 3]')
        >>> A.mig(), A.hull.mig()
        (2, 0)
        >>> MultiInterval().mig()
        Traceback (most recent call last):
        ValueError: the empty set has no mignitude
        """
        return numeric.mig(self._cuts)

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


def _is_integral(v: Real) -> bool:
    """a real number with an integer value (±inf and nan have none; a numpy timedelta64 is no number)"""
    if isinstance(v, Integral):
        return not is_numpy_time(v)
    try:
        return v == int(v)
    except (OverflowError, ValueError):
        return False


class OutwardMultiInterval(MultiInterval):
    """
    a MultiInterval whose float results round outward: a low end down and a high end up, from the
    exact value, so every result holds the exact result of its operands (the tightest such floats).
    exact operands give the same results as in MultiInterval. mixed with a MultiInterval, the result
    is an OutwardMultiInterval, whichever side it is on: of an operator, or of a method taking
    another set (`M.hypot(O)`, `M.union(O)`)

    inclusion isotone within one grid: `A ⊆ B` gives `f(A) ⊆ f(B)` when every float piece of A lies
    in a float piece of B (an exact A inside anything included). across grids, a float piece of A
    inside an exact piece of B, f(A) lies within the tightest double cover of f(B) (`f(B).rounded()`),
    not always within f(B): an exact result keeps its exact ends, a rounded one sticks out by up to an
    ulp. `rounded()` puts a set on the double grid, outward

    an end that rounding moved is open, because nothing attains it: the exact sum below lies strictly
    between the two neighbouring doubles

    >>> OutwardMultiInterval(0.1) + 0.2
    OutwardMultiInterval.parse('(0.3, 0.30000000000000004)')
    >>> MultiInterval(0.1) + 0.2
    MultiInterval.parse('[0.30000000000000004]')
    """
    __slots__ = ()
    _outward = True

    def expand(self, distance) -> 'OutwardMultiInterval':
        """
        widen every piece by `distance` on both sides: the set plus `[-distance, distance]`, so a moved
        float end rounds outward like any sum (`MultiInterval.expand` rounds it to nearest)

        >>> OutwardMultiInterval(0.1, 0.2).expand(1)
        OutwardMultiInterval.parse('(-0.9, 1.2000000000000002)')
        """
        distance = normalize_value(distance)
        if not (0 <= distance < math.inf):
            raise ValueError('expand() needs a finite, non-negative distance')
        if not self or distance == 0:
            return self
        return self + MultiInterval(-distance, distance)

    def rounded(self) -> 'OutwardMultiInterval':
        """
        the same set with every finite end on the double grid, outward: a low end down and a high end
        up to a double, an end that moved open, so the tightest set with double ends holding this one
        (an exact end past the doubles goes to `inf`, open, as an overflow does). the class keeps an
        exact end exact, so it is isotone within one grid only (the class docstring): round the inputs
        first, and an op's results on them are on the grid too, where `A ⊆ B` gives `f(A) ⊆ f(B)`

        >>> from fractions import Fraction
        >>> OutwardMultiInterval(Fraction(1, 3), 1).rounded()
        OutwardMultiInterval.parse('(0.3333333333333333, 1.0]')
        """
        return self._wrap(rounding.float_cuts(self._cuts, outward=True))

    # python tries the right operand's reflected method first only if a subclass overrides it
    def __radd__(self, other):
        return MultiInterval.__radd__(self, other)

    def __rsub__(self, other):
        return MultiInterval.__rsub__(self, other)

    def __rmul__(self, other):
        return MultiInterval.__rmul__(self, other)

    def __rtruediv__(self, other):
        return MultiInterval.__rtruediv__(self, other)

    def __rmod__(self, other):
        return MultiInterval.__rmod__(self, other)

    def __rfloordiv__(self, other):
        return MultiInterval.__rfloordiv__(self, other)

    def __rdivmod__(self, other):
        return MultiInterval.__rdivmod__(self, other)

    def __rpow__(self, other):
        return MultiInterval.__rpow__(self, other)

    def __ror__(self, other):
        return MultiInterval.__ror__(self, other)

    def __rand__(self, other):
        return MultiInterval.__rand__(self, other)

    def __rxor__(self, other):
        return MultiInterval.__rxor__(self, other)
