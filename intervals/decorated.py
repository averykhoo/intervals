"""
ieee 1788's decorated intervals (1788-2015 §8, §12.12; M13g, D16): `DecoratedInterval`, a
`MultiInterval` with a `Decoration`, and the decorated constructors

the core `MultiInterval` stays undecorated (v2-plan.md "ieee 1788"): a decoration answers "was the
function defined and continuous on the whole input", which the result set cannot. this type carries
that answer beside the set. a decoration, best first:

    com   common: the set is non-empty and bounded, and everything so far was defined and continuous
    dac   defined and continuous
    def   defined
    trv   trivial: nothing is known (the only one the empty set can have)

there is no NaI and no `ill` (D16, owner 2026-09-26): where 1788 would make a NaI from invalid input
(`d-textToInterval "[2, 1]"`, `setDec(x, ill)`), this raises `UndefinedOperationError`, so nothing
makes one. decorations are ordered, `Decoration.TRV < Decoration.DEF < Decoration.DAC <
Decoration.COM`, so the weaker of two is `min`.

1788's operations on the type, in python's style:

    newDec(x)             DecoratedInterval(x)                the best decoration x can have
    setDec(x, d)          set_dec(x, d)                       d, demoted where it cannot fit
    intervalPart(dx)      dx.interval
    decorationPart(dx)    dx.decoration
    d-textToInterval(s)   text_to_decorated_interval(s)
    d-numsToInterval(a, b) nums_to_decorated_interval(a, b)

the constructor `DecoratedInterval(x, d)` is strict: a decoration that does not fit (`com` on an
unbounded set, anything but `trv` on the empty one) raises, as a literal `"[1,]_com"` does. `set_dec`
is 1788's forgiving `setDec` and demotes it instead (`setDec([1, inf], com)` is `[1, inf]_dac`).
"bounded" is decided on the exact set: `[1.0E+400]` is bounded here, and `com`, where 1788's binary64
enclosure `[max, inf]` is not.

>>> from intervals import MultiInterval
>>> DecoratedInterval(MultiInterval(1, 2))
DecoratedInterval(MultiInterval.parse('[1, 2]'), Decoration.COM)
>>> print(set_dec(MultiInterval.parse('[1, inf)'), 'com'), set_dec(MultiInterval(), Decoration.DEF))
[1, inf)_dac {}_trv
>>> print(text_to_decorated_interval('[1, 2]_def'), text_to_decorated_interval('[entire]'))
[1, 2]_def (-inf, inf)_dac
>>> text_to_decorated_interval('[1,]_com')
Traceback (most recent call last):
    ...
intervals.errors.UndefinedOperationError: invalid 1788 interval literal '[1,]_com': com is for bounded non-empty intervals only
"""
import enum
import math
import warnings
from fractions import Fraction
from numbers import Real

from intervals import elementary
from intervals import kernel
from intervals.errors import UndefinedOperationError
from intervals.functions import _inside_k
from intervals.literals import _bare
from intervals.literals import nums_to_interval
from intervals.literals import parse_literal
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import _is_integral
from intervals.rounding import exact_cuts


class Decoration(enum.Enum):
    """1788's decorations without `ill` (no NaI, D16), ordered worst to best: trv < def < dac < com"""
    COM = 'com'
    DAC = 'dac'
    DEF = 'def'
    TRV = 'trv'

    def __lt__(self, other):
        if not isinstance(other, Decoration):
            return NotImplemented
        return _RANK[self] < _RANK[other]

    def __le__(self, other):
        if not isinstance(other, Decoration):
            return NotImplemented
        return _RANK[self] <= _RANK[other]

    def __gt__(self, other):
        if not isinstance(other, Decoration):
            return NotImplemented
        return _RANK[self] > _RANK[other]

    def __ge__(self, other):
        if not isinstance(other, Decoration):
            return NotImplemented
        return _RANK[self] >= _RANK[other]


_RANK = {Decoration.TRV: 0, Decoration.DEF: 1, Decoration.DAC: 2, Decoration.COM: 3}


def _as_decoration(decoration) -> Decoration:
    """a `Decoration`, or its lower-case name; `ill` and any other name raise `UndefinedOperationError`"""
    if isinstance(decoration, Decoration):
        return decoration
    if not isinstance(decoration, str):
        raise TypeError(f'expected a Decoration or its name, got {type(decoration).__name__}')
    try:
        return Decoration(decoration)
    except ValueError:
        why = 'ill belongs to NaI, which the package does not have (D16)' if decoration == 'ill' else \
            'not one of com, dac, def, trv'
        raise UndefinedOperationError(f'no decoration {decoration!r}: {why}') from None


def _as_interval(interval) -> MultiInterval:
    if not isinstance(interval, MultiInterval):
        raise TypeError(f'expected a MultiInterval, got {type(interval).__name__}')
    return interval


def _best(interval: MultiInterval) -> Decoration:
    """1788's newDec: com if bounded (no point at, and no piece reaching, ±inf), dac if unbounded,
    trv if empty"""
    if not interval:
        return Decoration.TRV
    return Decoration.COM if interval.is_finite else Decoration.DAC


class DecoratedInterval:
    """
    a `MultiInterval` (any subclass, kept as it is) with a `Decoration`. immutable and hashable;
    equal iff both the set and the decoration are

    `DecoratedInterval(x)` is 1788's `newDec(x)`, the best decoration that fits. with a decoration
    given (a `Decoration` or its name) it must fit, or `UndefinedOperationError` is raised: the
    empty set takes `trv` only and `com` needs a non-empty bounded set. `ill` has no counterpart
    """
    __slots__ = ('_interval', '_decoration')

    def __init__(self, interval: MultiInterval, decoration=None):
        interval = _as_interval(interval)
        best = _best(interval)
        decoration = best if decoration is None else _as_decoration(decoration)
        if decoration > best:
            raise UndefinedOperationError(
                f'{decoration.value} does not fit {interval}: the empty set is trv only, com needs a '
                f'bounded set')
        object.__setattr__(self, '_interval', interval)
        object.__setattr__(self, '_decoration', decoration)

    @property
    def interval(self) -> MultiInterval:
        """1788's intervalPart"""
        return self._interval

    @property
    def decoration(self) -> Decoration:
        """1788's decorationPart"""
        return self._decoration

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __reduce__(self):
        return type(self), (self._interval, self._decoration)

    def __eq__(self, other):
        if not isinstance(other, DecoratedInterval):
            return NotImplemented
        return self._decoration is other._decoration and self._interval == other._interval

    def __ne__(self, other):
        if not isinstance(other, DecoratedInterval):
            return NotImplemented
        return not self == other

    def __hash__(self):
        return hash((self._interval, self._decoration))

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._interval!r}, Decoration.{self._decoration.name})'

    def __str__(self) -> str:
        return f'{self._interval}_{self._decoration.value}'

    # M13g part 3: DECORATION PROPAGATION. each op computes the core's set from the intervals, then
    # the decoration `_propagate` gives it (see "propagation" in the module docstring). an operand is
    # a DecoratedInterval or a real number (a point, as newDec makes it); a bare MultiInterval is
    # refused, as 1788 does not mix bare and decorated intervals

    __array_ufunc__ = None  # numpy defers to the reflected methods, as for MultiInterval

    def _coerce(self, other):
        """a DecoratedInterval as it is, a real number as newDec's point of this set's class, else
        NotImplemented"""
        if isinstance(other, DecoratedInterval):
            return other
        if isinstance(other, Real) and not isinstance(other, bool):
            return DecoratedInterval(type(self._interval)(other))
        return NotImplemented

    def _coerce_or_raise(self, other) -> 'DecoratedInterval':
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            raise TypeError(f'expected a DecoratedInterval or a real number, got {type(other).__name__}')
        return coerced

    def _binary(self, other, name, reflected=False):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        a, b = (other, self) if reflected else (self, other)
        return _BINARY[name](a, b)

    def __add__(self, other):
        return self._binary(other, 'add')

    def __radd__(self, other):
        return self._binary(other, 'add', reflected=True)

    def __sub__(self, other):
        return self._binary(other, 'sub')

    def __rsub__(self, other):
        return self._binary(other, 'sub', reflected=True)

    def __mul__(self, other):
        return self._binary(other, 'mul')

    def __rmul__(self, other):
        return self._binary(other, 'mul', reflected=True)

    def __truediv__(self, other):
        """
        1788's div: defined where the divisor is not 0

        >>> print(DecoratedInterval(MultiInterval(1, 2)) / DecoratedInterval(MultiInterval(0, 1)))
        [1, inf]_trv
        """
        return self._binary(other, 'div')

    def __rtruediv__(self, other):
        return self._binary(other, 'div', reflected=True)

    def __mod__(self, other):
        """python's floor-mod (not in 1788): defined where the divisor is not 0, continuous where
        `x / y` is not an integer"""
        return self._binary(other, 'mod')

    def __rmod__(self, other):
        return self._binary(other, 'mod', reflected=True)

    def __floordiv__(self, other):
        """`floor(x / y)` (not in 1788), decorated as `%` is"""
        return self._binary(other, 'floordiv')

    def __rfloordiv__(self, other):
        return self._binary(other, 'floordiv', reflected=True)

    def __divmod__(self, other):
        """`(self // other, self % other)`, two decorated sets"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return _divmod(self, other)

    def __rdivmod__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return _divmod(other, self)

    def __pow__(self, exponent, modulo=None):
        """
        as `MultiInterval.__pow__` (D11): an integral real exponent is 1788's pown, a
        DecoratedInterval or any other real one 1788's pow

        >>> x = DecoratedInterval(MultiInterval(-1, 2))
        >>> print(x ** 2, x ** -1, abs(x) ** DecoratedInterval(MultiInterval(1/2)))
        [0, 4]_com { [-inf, -1] , [1/2, inf] }_trv [0.0, 1.4142135623730951]_com
        """
        if modulo is not None or isinstance(exponent, bool):
            return NotImplemented
        if isinstance(exponent, Real) and _is_integral(exponent):
            return _pown(self, exponent)
        exponent = self._coerce(exponent)
        if exponent is NotImplemented:
            return NotImplemented
        return _pow(self, exponent)

    def __rpow__(self, base):
        """`b ** X` for a real b is 1788's pow of the point b"""
        base = self._coerce(base)
        if base is NotImplemented:
            return NotImplemented
        return _pow(base, self)

    def __neg__(self) -> 'DecoratedInterval':
        return _propagate(-self._interval, (self,), _everywhere(self))

    def __pos__(self) -> 'DecoratedInterval':
        return _propagate(+self._interval, (self,), _everywhere(self))

    def __abs__(self) -> 'DecoratedInterval':
        return _propagate(abs(self._interval), (self,), _everywhere(self))

    def reciprocal(self) -> 'DecoratedInterval':
        """1788's recip: defined where the operand is not 0"""
        return _propagate(self._interval.reciprocal(), (self,), _inside(self, _NON_ZERO))

    def minimum(self, other) -> 'DecoratedInterval':
        other = self._coerce_or_raise(other)
        return _propagate(self._interval.minimum(other._interval), (self, other), _everywhere(self, other))

    def maximum(self, other) -> 'DecoratedInterval':
        other = self._coerce_or_raise(other)
        return _propagate(self._interval.maximum(other._interval), (self, other), _everywhere(self, other))

    def fma(self, factor, addend) -> 'DecoratedInterval':
        factor, addend = self._coerce_or_raise(factor), self._coerce_or_raise(addend)
        return _propagate(self._interval.fma(factor._interval, addend._interval), (self, factor, addend),
                          _everywhere(self, factor, addend))

    def hypot(self, other) -> 'DecoratedInterval':
        other = self._coerce_or_raise(other)
        return _propagate(self._interval.hypot(other._interval), (self, other), _everywhere(self, other))

    def atan2(self, x) -> 'DecoratedInterval':
        """
        1788's atan2 (y = self): defined off the origin, continuous off the negative x axis, where the
        angle jumps from pi to -pi; the restriction to the box is continuous there unless the box
        reaches the axis from below

        >>> y = DecoratedInterval(MultiInterval(0, 1))
        >>> print(y.atan2(DecoratedInterval(MultiInterval(-2, -1))).decoration)
        Decoration.DAC
        """
        x = self._coerce_or_raise(x)
        return _atan2(self, x)

    def log(self, base=None) -> 'DecoratedInterval':
        """the natural logarithm, or to `base`: defined above 0"""
        return _propagate(self._interval.log(base), (self,), _inside(self, _POSITIVE))

    def rootn(self, n: int) -> 'DecoratedInterval':
        """1788's rootn: an even root is defined from 0 up, a negative one not at 0"""
        result = self._interval.rootn(n)  # the core checks n first
        domain = (_NON_NEGATIVE if n % 2 == 0 else _REALS) if n > 0 else _POSITIVE if n % 2 == 0 else _NON_ZERO
        return _propagate(result, (self,), _inside(self, domain))

    def floor(self) -> 'DecoratedInterval':
        """
        a step function: dac where it is constant on each piece, com if no closed end is a jump
        either (1788: `floor([1.1, 2])` is def, `ceil([1.1, 2])` dac, `floor([-1.2, -1.1])` com)

        >>> print(DecoratedInterval(MultiInterval.parse('[1/2, 1)')).floor())
        [0]_com
        """
        return _step(self, 'floor')

    def ceil(self) -> 'DecoratedInterval':
        return _step(self, 'ceil')

    def trunc(self) -> 'DecoratedInterval':
        return _step(self, 'trunc')

    def round(self, ndigits=None) -> 'DecoratedInterval':
        return _step(self, 'round', ndigits)

    def round_ties_away(self, ndigits=None) -> 'DecoratedInterval':
        return _step(self, 'round_ties_away', ndigits)

    def sign(self) -> 'DecoratedInterval':
        return _step(self, 'sign')

    def __floor__(self) -> 'DecoratedInterval':
        return self.floor()

    def __ceil__(self) -> 'DecoratedInterval':
        return self.ceil()

    def __trunc__(self) -> 'DecoratedInterval':
        return self.trunc()

    def __round__(self, ndigits=None) -> 'DecoratedInterval':
        return self.round(ndigits)

    # set operations are not point functions: 1788 decorates intersection, convexHull, cancelMinus and
    # cancelPlus trv, whatever the operands, and so does every set operation here

    def __and__(self, other):
        return self._set_operation(other, MultiInterval.__and__)

    def __rand__(self, other):
        return self._set_operation(other, MultiInterval.__and__, reflected=True)

    def __or__(self, other):
        return self._set_operation(other, MultiInterval.__or__)

    def __ror__(self, other):
        return self._set_operation(other, MultiInterval.__or__, reflected=True)

    def __xor__(self, other):
        return self._set_operation(other, MultiInterval.__xor__)

    def __rxor__(self, other):
        return self._set_operation(other, MultiInterval.__xor__, reflected=True)

    def difference(self, other) -> 'DecoratedInterval':
        return _trivial(self._interval.difference(self._coerce_or_raise(other)._interval))

    def complement(self) -> 'DecoratedInterval':
        return _trivial(self._interval.complement())

    def __invert__(self) -> 'DecoratedInterval':
        return self.complement()

    @property
    def hull(self) -> 'DecoratedInterval':
        return _trivial(self._interval.hull)

    @property
    def closed_hull(self) -> 'DecoratedInterval':
        return _trivial(self._interval.closed_hull)

    @property
    def interior(self) -> 'DecoratedInterval':
        return _trivial(self._interval.interior)

    def cancel_minus(self, other) -> 'DecoratedInterval':
        """1788's cancelMinus, trv as 1788 decorates it (a set operation, D13)"""
        return _trivial(self._interval.cancel_minus(self._coerce_or_raise(other)._interval))

    def cancel_plus(self, other) -> 'DecoratedInterval':
        return _trivial(self._interval.cancel_plus(self._coerce_or_raise(other)._interval))

    def _set_operation(self, other, operation, reflected=False):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        a, b = (other, self) if reflected else (self, other)
        return _trivial(operation(a._interval, b._interval))


def set_dec(interval: MultiInterval, decoration) -> DecoratedInterval:
    """
    1788's `setDec`: `interval` with `decoration` (a `Decoration` or its name), demoted to the best
    that fits where it cannot: the empty set gets `trv`, `com` on an unbounded set `dac`. so the
    decoration is `min(decoration, DecoratedInterval(interval).decoration)`. `ill`, which 1788 answers
    with NaI, raises `UndefinedOperationError`

    >>> print(set_dec(MultiInterval(1, 2), 'def'), set_dec(MultiInterval(), 'com'))
    [1, 2]_def {}_trv
    """
    interval = _as_interval(interval)
    return DecoratedInterval(interval, min(_as_decoration(decoration), _best(interval)))


def text_to_decorated_interval(text: str) -> DecoratedInterval:
    """
    1788's `d-textToInterval`: the set a 1788 literal denotes, exactly and under 1788's input rule
    (an infinite end open), as `text_to_interval` reads it, with the literal's decoration, or
    `newDec`'s if it has none. invalid input raises `UndefinedOperationError`: everything
    `text_to_interval` refuses except a decoration, plus a decoration that does not fit the exact set
    (`"[1,]_com"`, `"[ ]_def"`), `_ill` and `[nai]`

    >>> print(text_to_decorated_interval('3.56?1_def'), text_to_decorated_interval('[ ]'))
    [71/20, 357/100]_def {}_trv
    """
    literal = parse_literal(text)
    return DecoratedInterval(_bare(literal.lo, literal.hi),
                             None if literal.decoration is None else Decoration(literal.decoration))


def nums_to_decorated_interval(lo, hi) -> DecoratedInterval:
    """
    1788's `d-numsToInterval`: `nums_to_interval(lo, hi)` with `newDec`'s decoration; the same
    invalid input raises `UndefinedOperationError`

    >>> print(nums_to_decorated_interval(-1, 1), nums_to_decorated_interval(float('-inf'), 1))
    [-1, 1]_com (-inf, 1]_dac
    """
    return DecoratedInterval(nums_to_interval(lo, hi))


# PROPAGATION (M13g part 3; 1788-2015 §11): the local decoration of an op on the operands' box, then
# the min with each operand's

INF = math.inf


def _set(*pieces) -> MultiInterval:
    return MultiInterval.from_pieces(pieces)


# 1788's domains, as sets of reals: an attained ±inf is never in one (1788's functions are functions
# of reals), so an operand holding ±inf as a point is outside every domain and gets trv
_REALS = _set((-INF, INF, False, False))
_NON_ZERO = _set((-INF, 0, False, False), (0, INF, False, False))
_NON_NEGATIVE = _set((0, INF, True, False))
_POSITIVE = _set((0, INF, False, False))
_NEGATIVE = _set((-INF, 0, False, False))
_FUNCTION_DOMAINS = {
    'sqrt': _NON_NEGATIVE, 'exp': _REALS, 'exp2': _REALS, 'exp10': _REALS, 'log2': _POSITIVE,
    'log10': _POSITIVE, 'sin': _REALS, 'cos': _REALS, 'asin': _set((-1, 1)), 'acos': _set((-1, 1)),
    'atan': _REALS, 'sinh': _REALS, 'cosh': _REALS, 'tanh': _REALS, 'asinh': _REALS,
    'acosh': _set((1, INF, True, False)), 'atanh': _set((-1, 1, False, False)), 'expm1': _REALS,
    'log1p': _set((-1, INF, False, False)), 'cbrt': _REALS, 'acot': _REALS, 'coth': _NON_ZERO,
    'csch': _NON_ZERO, 'sech': _REALS, 'acoth': _set((-INF, -1, False, False), (1, INF, False, False)),
}
# the reals but the poles (k + offset) pi, k any integer
_POLES = {'tan': Fraction(1, 2), 'sec': Fraction(1, 2), 'cot': Fraction(0), 'csc': Fraction(0)}


def _quietly(fn, *args):
    """a core op computed only to decide a decoration: its warnings are not the user's"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args)


def _exact(x: 'DecoratedInterval') -> MultiInterval:
    """the operand's set with every float end as the rational it is, in a MultiInterval: every
    decision on a decoration is made on the exact set, never on a rounded one"""
    return MultiInterval.from_cuts(exact_cuts(x.interval.cuts))


def _inside(x: 'DecoratedInterval', domain: MultiInterval) -> bool:
    return x.interval.issubset(domain)


def _everywhere(*operands) -> bool:
    """an op defined on all the reals: defined iff every operand is a set of reals"""
    return all(_inside(x, _REALS) for x in operands)


def _propagate(result: MultiInterval, operands, defined: bool, restricted: bool = True,
               everywhere: bool = True) -> 'DecoratedInterval':
    """
    1788's decorated evaluation. the local decoration of the op on the box of the operands' sets:
    trv unless `defined` (every point of the box is in the op's domain), def unless `restricted`
    (the op restricted to the box is continuous), dac unless `everywhere` (continuous at every point
    of the box) and every operand is bounded, else com. the result's decoration is the min of that,
    each operand's decoration and the best the result can have (newDec's: trv for an empty result,
    at most dac for an unbounded one, so 1788's "and the result is bounded" for com)
    """
    if not defined:
        local = Decoration.TRV
    elif not restricted:
        local = Decoration.DEF
    elif not everywhere or not all(x.interval.is_finite for x in operands):
        local = Decoration.DAC
    else:
        local = Decoration.COM
    return DecoratedInterval(result, min(local, _best(result), *(x.decoration for x in operands)))


def _trivial(result: MultiInterval) -> 'DecoratedInterval':
    """a set operation's result, trv whatever the operands (1788 decorates intersection, convexHull,
    cancelMinus and cancelPlus so). M13e hook: 1788 decorates every reverse op's result trv too, so
    the decorated reverse ops (not built here) are `_trivial(<reverse op on the intervals>)`"""
    return DecoratedInterval(result, Decoration.TRV)


def _add(a, b):
    return _propagate(a.interval + b.interval, (a, b), _everywhere(a, b))


def _sub(a, b):
    return _propagate(a.interval - b.interval, (a, b), _everywhere(a, b))


def _mul(a, b):
    return _propagate(a.interval * b.interval, (a, b), _everywhere(a, b))


def _div(a, b):
    return _propagate(a.interval / b.interval, (a, b), _inside(a, _REALS) and _inside(b, _NON_ZERO))


def _quotient_steps(a, b):
    """(restricted, everywhere) for `%` and `//` on a box inside their domain: per pair of pieces
    (the pieces of a set are apart, so continuity on the set is continuity on each pair), both are
    continuous on the pair iff `floor(x / y)` is one integer k there, and at each of its points iff
    moreover `x / y` never equals k (they jump wherever `x / y` is an integer)"""
    everywhere = True
    for p in _exact(a):
        for q in _exact(b):
            ratio = _quietly(MultiInterval.__truediv__, p, q)
            k = _quietly(MultiInterval.floor, ratio)
            if not (k.is_degenerate and k.is_contiguous):
                return False, False
            everywhere = everywhere and k.inf not in ratio
    return True, everywhere


def _quotient(a, b, result):
    defined = _inside(a, _REALS) and _inside(b, _NON_ZERO)
    restricted, everywhere = _quotient_steps(a, b) if defined else (False, False)
    return _propagate(result, (a, b), defined, restricted, everywhere)


def _mod(a, b):
    return _quotient(a, b, a.interval % b.interval)


def _floordiv(a, b):
    return _quotient(a, b, a.interval // b.interval)


def _divmod(a, b):
    q, r = divmod(a.interval, b.interval)
    return _quotient(a, b, q), _quotient(a, b, r)


_BINARY = {'add': _add, 'sub': _sub, 'mul': _mul, 'div': _div, 'mod': _mod, 'floordiv': _floordiv}


def _pown(x, n):
    """1788's pown: defined everywhere for n >= 0 (`x ** 0` is 1, at 0 too), off 0 for n < 0"""
    result = x.interval ** n
    return _propagate(result, (x,), _inside(x, _REALS if n >= 0 else _NON_ZERO))


def _pow(x, y):
    """1788's pow: defined for x > 0, and x = 0 where y > 0; continuous on that domain"""
    result = x.interval ** y.interval
    defined = (_inside(x, _NON_NEGATIVE) and _inside(y, _REALS)
               and (0 not in x.interval or _inside(y, _POSITIVE)))
    return _propagate(result, (x, y), defined)


def _atan2(y, x):
    result = y.interval.atan2(x.interval)
    defined = _everywhere(y, x) and not (0 in y.interval and 0 in x.interval)
    restricted = everywhere = True
    if defined and 0 in y.interval and not (x.interval & _NEGATIVE).is_empty:
        # the box meets the negative x axis, where atan2 is pi and jumps: never com. its restriction
        # is continuous unless the piece of y holding 0 reaches below 0 (from above it tends to pi)
        everywhere = False
        piece = next(p for p in y.interval if 0 in p)
        restricted = not piece.inf < 0
    return _propagate(result, (y, x), defined, restricted, everywhere)


def _misses_poles(x, offset: Fraction) -> bool:
    """no point of the (real) set is a pole `(k + offset) pi`; poles are irrational but for 0"""
    for lo, lo_closed, hi, hi_closed in kernel.pieces(_exact(x).cuts):
        if lo == -INF or hi == INF:
            return False
        first, last = _inside_k(lo, hi, offset)
        if first <= last:
            return False
        if any(closed and elementary.floor_over_pi(end, offset)[1]
               for end, closed in ((lo, lo_closed), (hi, hi_closed))):
            return False
    return True


def _function(name: str):
    def method(self) -> 'DecoratedInterval':
        result = getattr(self.interval, name)()
        if name in _POLES:
            defined = _inside(self, _REALS) and _misses_poles(self, _POLES[name])
        else:
            defined = _inside(self, _FUNCTION_DOMAINS[name])
        return _propagate(result, (self,), defined)

    method.__name__ = name
    method.__qualname__ = f'DecoratedInterval.{name}'
    method.__doc__ = f"1788's {name}, decorated: continuous wherever it is defined, so trv, dac or com"
    return method


for _name in (*_FUNCTION_DOMAINS, *_POLES):
    setattr(DecoratedInterval, _name, _function(_name))

# the step functions' jumps, as a test on a value in grid units (1, or 10 ** -ndigits for round)
_JUMPS = {
    'floor': lambda v: v.denominator == 1,
    'ceil': lambda v: v.denominator == 1,
    'trunc': lambda v: v.denominator == 1 and v != 0,
    'round': lambda v: (v - Fraction(1, 2)).denominator == 1,
    'round_ties_away': lambda v: (v - Fraction(1, 2)).denominator == 1,
    'sign': lambda v: v == 0,
}


def _step(x, name: str, ndigits=None):
    """
    a step function, decorated as 1788 does (floor, ceil, trunc, roundTiesToEven, roundTiesToAway,
    sign): defined everywhere; its restriction to a piece is continuous iff it is constant there
    (dac), and it is continuous at every point of the piece iff moreover no closed end is a jump
    (com). on a set of several pieces: every piece, since the pieces are apart
    """
    step = getattr(MultiInterval, name)
    args = () if ndigits is None else (ndigits,)
    result = step(x.interval, *args)
    defined = _inside(x, _REALS)
    restricted = everywhere = defined
    if defined:
        unit = Fraction(1) if ndigits is None else Fraction(10) ** -ndigits
        for p in _exact(x):
            values = _quietly(step, p, *args)
            if not (values.is_degenerate and values.is_contiguous):
                restricted = everywhere = False
                break
            ends = ((p.inf, p.inf_closed), (p.sup, p.sup_closed))
            if any(closed and _JUMPS[name](Fraction(end) / unit) for end, closed in ends):
                everywhere = False
    return _propagate(result, (x,), defined, restricted, everywhere)
