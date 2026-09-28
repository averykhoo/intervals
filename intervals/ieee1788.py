"""
ieee 1788's inf-sup binary64 intervals, bare and decorated, over the library (M16b, H3's second part)

a thin layer, a wrapper class and never a mode (the decision log, 2026-08-16): every set is computed
by the library (`OutwardMultiInterval`, and `DecoratedInterval` over one), and the layer only converts
in and out by 1788's rules, plus the few ops where 1788 defines a different answer than the
library's set. nothing in `MultiInterval`, `OutwardMultiInterval` or `DecoratedInterval` changes
meaning, and `intervals` does not import this module: `from intervals import ieee1788`.

* **the type**: `Interval(lo, hi, decoration)`, one class for both of 1788's flavours (`decoration`
  `None` is a bare interval). its set is in **1788's form**: empty, or one piece whose finite ends are
  closed doubles and whose infinite ends are open, so `[1, +infinity]` is `[1.0, inf)`: 1788 never
  attains infinity. the flavour of a call is its operands', which must agree (1788 has no mixed
  operations); a real number is a point of that flavour (newDec's, if decorated). a library value
  (`MultiInterval`, `DecoratedInterval`) is no operand: `TypeError`
* **the input rule**: `Interval(lo, hi)` is 1788's `numsToInterval`, exact then hulled to doubles
  (`Interval(Fraction(1, 10))` is `[0.09999999999999999, 0.1]`); with a decoration it is strict, as
  `DecoratedInterval(x, d)` and a literal `[1,]_com` are, decided on the binary64 result:
  `Interval(1, 2 ** 1024, 'com')` raises, its hull being `[1.0, inf)` (where
  `text_to_decorated_interval` demotes the same overflow to dac: 1788's `d-textToInterval` does).
  `set_dec` is the forgiving one
* **the output rule**, `from_set(s)`, for any library value: (1) drop the attained infinities, since
  1788's functions are functions of reals; (2) the hull; (3) its ends outward to doubles; (4) the 1788
  form; (5) a decoration capped by newDec of the binary64 result. `from_set` encloses the set it is
  given: a to-nearest `MultiInterval`'s float ends are read as exact, so `from_set(MultiInterval(0.1)
  * 3)` encloses that set, not the real 0.3; only an `OutwardMultiInterval` or an exact result
  encloses a computation
* **warnings**: the library's `EmptySetPropagationWarning`, `DomainClippedWarning`,
  `IndeterminateResultWarning` and `HullWarning` have no 1788 meaning and are silenced inside each
  call (`warnings.catch_warnings`, which mutates the process-wide filters, so a call is not
  thread-safe, as `decorated._quietly` is not); `PossiblyUndefinedOperationWarning` goes through and
  `UndefinedOperationError` is raised
* **where 1788 defines another answer** than the library's set, the layer gives 1788's and the
  library keeps its own: `cancel_minus` answers entire as 1788's "no answer" where the library's is
  the Minkowski difference (D13); `overlap`'s touching closed intervals `meets` where the library's
  `allen` says `overlaps`; `mul_rev_to_pair` decorates its first interval as the division where 0 is
  not in `b`, where the library's `mul_rev` is trv
* **numbers**: python floats; `inf` of the empty set is `+inf` and `sup` `-inf`, and `inf` of an
  interval whose lower end is 0 is `-0.0` (and `sup` `+0.0`), as 1788 has it; `mid`, `rad`, `wid`,
  `mag`, `mig`, `mid_rad` of the empty set raise `ValueError`, the library's answer where 1788 says
  NaN (the default built, pending Q13 (b)). the reductions are the library's own

the functions carry 1788's names in snake_case (a trailing underscore on a python builtin: `abs_`,
`min_`, `max_`, `pow_`, `sum_`), and `NAMES` maps 1788's own spelling to each (`'mulRevToPair'`, and
`'d-numsToInterval'` for the decorated constructors). not here: NaI and `isNaI` (D16), 1788's
recommended `exp2m1`, `exp10m1`, `log2p1`, `log10p1`, `compoundm1`, `rsqrt` and the `*Pi` functions,
the exact text and interchange conversions, and every inf-sup type but binary64.

>>> from intervals import ieee1788
>>> from intervals.ieee1788 import Interval
>>> Interval(1, 2) / 10
Interval(0.09999999999999999, 0.2)
>>> print(ieee1788.text_to_interval('[0.1, 1/3]'), ieee1788.sqrt(Interval(-1, 4, 'com')))
[0.09999999999999999, 0.33333333333333337] [0.0, 2.0]_trv
>>> ieee1788.cancel_minus(Interval(0, 1), Interval(0, 2))  # 1788's "no answer"
Interval(float('-inf'), float('inf'))
>>> ieee1788.overlap(Interval(1, 2), Interval(2, 3))
<Overlap.MEETS: 'meets'>
>>> ieee1788.mul_rev_to_pair(Interval(-1, 1), Interval(1, 2))
(Interval(float('-inf'), -1.0), Interval(1.0, float('inf')))
>>> ieee1788.log(Interval(float('-inf'), 0)), ieee1788.inf(Interval(0, 1))  # the library's log is [-inf]
(Interval(), -0.0)
"""
import enum
import math
import warnings
from fractions import Fraction
from numbers import Integral
from numbers import Real

from intervals import decorated as _decorated
from intervals import kernel as _kernel
from intervals import literals as _literals
from intervals import reductions as _reductions
from intervals import reverse as _reverse
from intervals.decorated import DecoratedInterval as _DecoratedInterval
from intervals.decorated import Decoration
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.errors import IndeterminateResultWarning
from intervals.multi_interval import MultiInterval as _MultiInterval
from intervals.multi_interval import OutwardMultiInterval as _Outward
from intervals.multi_interval import _is_integral
from intervals.rounding import DOWN as _DOWN
from intervals.rounding import UP as _UP
from intervals.rounding import round_value as _round_value

_INF = math.inf
_LINE = _Outward(-_INF, _INF, start_closed=False, end_closed=False)  # 1788's entire
_EMPTY = _Outward()
_QUIET = (EmptySetPropagationWarning, DomainClippedWarning, IndeterminateResultWarning, HullWarning)


def _call(op, *args):
    """a library op with the warnings that have no 1788 meaning silenced, inside the call only"""
    with warnings.catch_warnings():
        for category in _QUIET:
            warnings.simplefilter('ignore', category)
        return op(*args)


def _is_real(v) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool)


def _is_1788_form(s) -> bool:
    if type(s) is not _Outward:
        return False
    if not s:
        return True
    lo, hi = s.inf, s.sup
    return (s.is_contiguous and type(lo) is float and type(hi) is float
            and s.inf_closed == (lo != -_INF) and s.sup_closed == (hi != _INF))


def _hull(s: _MultiInterval) -> _Outward:
    """the output rule's steps 1 to 4: the real points, their hull, outward to doubles, 1788's form"""
    real = _kernel.intersection(s.cuts, _LINE.cuts)
    if not real:
        return _EMPTY
    lo = float(_round_value(real[0].value, _DOWN)) + 0.0
    hi = float(_round_value(real[-1].value, _UP)) + 0.0
    return _Outward(lo, hi, start_closed=lo != -_INF, end_closed=hi != _INF)


class Interval:
    """
    1788's inf-sup binary64 interval, bare (`decoration` None) or decorated. `Interval()` is empty,
    `Interval(lo)` the point, `Interval(lo, hi)` 1788's `numsToInterval(lo, hi)`: exact, then hulled
    to doubles, an infinite end open. a decoration (a `Decoration` or its name) must fit the binary64
    result, else `UndefinedOperationError`; `Interval(decoration='trv')` is `[empty]_trv`.
    immutable and hashable; equal iff the set and the decoration are. no ordering: 1788 has four
    (`less`, `precedes`, `subset` and their strict forms), none of them the obvious one

    >>> x = Interval(1, 2, 'com')
    >>> x, x + 1, str(x * Interval(-1, float('inf'), 'dac'))
    (Interval(1.0, 2.0, 'com'), Interval(2.0, 3.0, 'com'), '[-2.0, inf]_dac')
    >>> x.to_set()
    DecoratedInterval(OutwardMultiInterval.parse('[1.0, 2.0]'), Decoration.COM)
    """
    __slots__ = ('_set', '_decoration')
    __array_ufunc__ = None  # the package's rule for every type

    def __init__(self, lo=None, hi=None, decoration=None):
        if lo is None:
            if hi is not None:
                raise TypeError('Interval: an upper bound without a lower one')
            s = _EMPTY
        else:
            for v in (lo,) if hi is None else (lo, hi):
                if not _is_real(v):
                    raise TypeError(f'Interval: a bound is a real number, got {type(v).__name__}')
            s = _hull(_literals.nums_to_interval(lo, lo if hi is None else hi))
        if decoration is not None:
            decoration = _DecoratedInterval(s, decoration).decoration  # strict, on the binary64 set
        self._init(s, decoration)

    def _init(self, s, decoration):
        if __debug__:
            assert _is_1788_form(s), s
        object.__setattr__(self, '_set', s)
        object.__setattr__(self, '_decoration', decoration)

    @classmethod
    def _make(cls, s: _Outward, decoration) -> 'Interval':
        """internal: wrap a set in the 1788 form (checked only under __debug__)"""
        out = object.__new__(cls)
        out._init(s, decoration)
        return out

    @property
    def decoration(self):
        """the `Decoration`, or `None` for a bare interval"""
        return self._decoration

    def to_set(self):
        """the library value the layer computes on: the `OutwardMultiInterval`, or for a decorated
        interval `DecoratedInterval(that, decoration)`"""
        return self._set if self._decoration is None else _DecoratedInterval(self._set, self._decoration)

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __delattr__(self, name):
        raise AttributeError(f'{type(self).__name__} is immutable')

    def __reduce__(self):
        return Interval._make, (self._set, self._decoration)

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return NotImplemented
        return self._decoration is other._decoration and self._set == other._set

    def __hash__(self):
        return hash((self._set, self._decoration))

    def __bool__(self) -> bool:
        return bool(self._set)

    def __repr__(self) -> str:
        d = self._decoration
        if not self._set:
            return 'Interval()' if d is None else f'Interval(decoration={d.value!r})'
        lo, hi = (f'float({str(v)!r})' if math.isinf(v) else repr(v) for v in (self._set.inf, self._set.sup))
        return f'Interval({lo}, {hi})' if d is None else f'Interval({lo}, {hi}, {d.value!r})'

    def __str__(self) -> str:
        """a 1788 literal, python's shortest decimal of each end; read back, it encloses this"""
        s = self._set
        body = '[empty]' if not s else '[entire]' if s == _LINE else f'[{s.inf!r}, {s.sup!r}]'
        return body if self._decoration is None else f'{body}_{self._decoration.value}'

    # python's spellings of a few functions: a number is a point

    def __add__(self, other):
        return add(self, other) if _operand(other) else NotImplemented

    def __radd__(self, other):
        return add(other, self) if _operand(other) else NotImplemented

    def __sub__(self, other):
        return sub(self, other) if _operand(other) else NotImplemented

    def __rsub__(self, other):
        return sub(other, self) if _operand(other) else NotImplemented

    def __mul__(self, other):
        return mul(self, other) if _operand(other) else NotImplemented

    def __rmul__(self, other):
        return mul(other, self) if _operand(other) else NotImplemented

    def __truediv__(self, other):
        return div(self, other) if _operand(other) else NotImplemented

    def __rtruediv__(self, other):
        return div(other, self) if _operand(other) else NotImplemented

    def __pow__(self, exponent, modulo=None):
        """as D11 reads `**`: an integral real exponent is `pown`, any other real and an `Interval`
        `pow_`"""
        if modulo is not None or not _operand(exponent):
            return NotImplemented
        if not isinstance(exponent, Interval) and _is_integral(exponent):
            return pown(self, int(exponent))
        return pow_(self, exponent)

    def __rpow__(self, base):
        return pow_(base, self) if _operand(base) else NotImplemented

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return pos(self)

    def __abs__(self):
        return abs_(self)

    def __and__(self, other):
        return intersection(self, other) if _operand(other) else NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        """the convex hull: the only 1788 interval holding the union"""
        return convex_hull(self, other) if _operand(other) else NotImplemented

    __ror__ = __or__

    def __contains__(self, m) -> bool:
        return is_member(m, self)


def _operand(v) -> bool:
    return isinstance(v, Interval) or _is_real(v)


def _flavour(name: str, operands) -> bool:
    """whether the call is decorated; a TypeError for mixed flavours or an operand that is neither an
    `Interval` nor a real number"""
    decorated = set()
    for a in operands:
        if isinstance(a, Interval):
            decorated.add(a._decoration is not None)
        elif not _is_real(a):
            raise TypeError(f'{name}: expected an ieee1788.Interval or a real number, got {type(a).__name__}')
    if len(decorated) > 1:
        raise TypeError(f'{name}: a bare and a decorated interval (1788 has no mixed operations)')
    return decorated == {True}


def _interval_of(v, decorated: bool) -> Interval:
    """an operand as an `Interval` of the call's flavour: a number is the point, newDec'd if decorated"""
    if isinstance(v, Interval):
        return v
    point = Interval(v)
    return new_dec(point) if decorated else point


def _sets(name: str, operands):
    """(whether decorated, the operands' library values)"""
    decorated = _flavour(name, operands)
    return decorated, [_interval_of(a, decorated).to_set() for a in operands]


def from_set(s) -> Interval:
    """
    the output rule: the 1788 interval of a library value (any `MultiInterval`, bare, or a
    `DecoratedInterval`, decorated). the attained infinities are dropped, then the hull is rounded
    outward to doubles, and a decoration is capped by newDec of that result. `from_set(x.to_set())`
    is `x`. it encloses the set it is given, so a to-nearest `MultiInterval`'s float ends are read as
    exact values

    >>> from intervals import MultiInterval
    >>> from_set(MultiInterval.parse('[1, 2] | [inf]')), from_set(MultiInterval(2 ** 1024))
    (Interval(1.0, 2.0), Interval(1.7976931348623157e+308, float('inf')))
    """
    if isinstance(s, _DecoratedInterval):
        h = _hull(s.interval)
        return Interval._make(h, min(s.decoration, _decorated._best(h)))
    if isinstance(s, _MultiInterval):
        return Interval._make(_hull(s), None)
    raise TypeError(f'from_set: expected a MultiInterval or a DecoratedInterval, got {type(s).__name__}')


def _lift(name: str, library, doc: str = ''):
    """1788's op `name` as the library's: the operands' sets, the op, the output rule"""
    def op(*operands):
        return from_set(_call(library, *_sets(name, operands)[1]))
    op.__name__ = op.__qualname__ = name
    op.__doc__ = doc or f"ieee 1788's `{name}`, the library's op through the input and output rules"
    return op


def _method(name: str):
    return lambda a: getattr(a, name)()


def _exponent(name: str, n) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError(f'{name}: the exponent is an int, got {type(n).__name__}')
    return int(n)


# CONSTRUCTORS

def nums_to_interval(lo, hi) -> Interval:
    """1788's `numsToInterval`, bare: `Interval(lo, hi)`"""
    if not (_is_real(lo) and _is_real(hi)):
        raise TypeError('nums_to_interval: the bounds are real numbers')
    return Interval(lo, hi)


def nums_to_decorated_interval(lo, hi) -> Interval:
    """1788's `numsToInterval`, decorated: newDec of `Interval(lo, hi)`"""
    return new_dec(nums_to_interval(lo, hi))


def text_to_interval(text: str) -> Interval:
    """1788's `textToInterval`, bare: the literal read exactly (`intervals.text_to_interval`, which
    decides validity and never warns `PossiblyUndefinedOperation`), then hulled to doubles"""
    return from_set(_call(_literals.text_to_interval, text))


def text_to_decorated_interval(text: str) -> Interval:
    """1788's `textToInterval`, decorated: the literal's decoration, demoted to what the binary64 hull
    fits (`"[1.0E+400]_com"` is `[max, inf)_dac`)"""
    return from_set(_call(_decorated.text_to_decorated_interval, text))


def empty() -> Interval:
    """the bare empty interval"""
    return Interval()


def entire() -> Interval:
    """the bare whole line, open at both infinities"""
    return Interval(-_INF, _INF)


# DECORATIONS

def _bare(name: str, x) -> Interval:
    if not isinstance(x, Interval) or x._decoration is not None:
        raise TypeError(f'{name}: expected a bare ieee1788.Interval')
    return x


def _decorated_one(name: str, x) -> Interval:
    if not isinstance(x, Interval) or x._decoration is None:
        raise TypeError(f'{name}: expected a decorated ieee1788.Interval')
    return x


def new_dec(x) -> Interval:
    """1788's `newDec`: the best decoration the bare interval can have (com, dac or trv)"""
    s = _bare('new_dec', x)._set
    return Interval._make(s, _decorated._best(s))


def set_dec(x, decoration) -> Interval:
    """1788's `setDec`: the bare interval with the decoration, demoted where it does not fit"""
    return from_set(_call(_decorated.set_dec, _bare('set_dec', x)._set, decoration))


def interval_part(x) -> Interval:
    """1788's `intervalPart`"""
    return Interval._make(_decorated_one('interval_part', x)._set, None)


def decoration_part(x) -> Decoration:
    """1788's `decorationPart`"""
    return _decorated_one('decoration_part', x)._decoration


# FORWARD OPS: the library's, lifted

pos = _lift('pos', lambda a: +a)
neg = _lift('neg', lambda a: -a)
add = _lift('add', lambda a, b: a + b)
sub = _lift('sub', lambda a, b: a - b)
mul = _lift('mul', lambda a, b: a * b)
div = _lift('div', lambda a, b: a / b)
recip = _lift('recip', lambda a: a.reciprocal())
sqr = _lift('sqr', lambda a: a ** 2)
sqrt = _lift('sqrt', _method('sqrt'))
fma = _lift('fma', lambda a, b, c: a.fma(b, c))
abs_ = _lift('abs_', abs)
min_ = _lift('min_', lambda a, b: a.minimum(b))
max_ = _lift('max_', lambda a, b: a.maximum(b))
pow_ = _lift('pow_', lambda a, b: a ** b, "ieee 1788's `pow`: an interval exponent, so never pown (D11)")
hypot = _lift('hypot', lambda a, b: a.hypot(b))
atan2 = _lift('atan2', lambda y, x: y.atan2(x))
logp1 = _lift('logp1', _method('log1p'))
exp, exp2, exp10, expm1 = (_lift(n, _method(n)) for n in ('exp', 'exp2', 'exp10', 'expm1'))
log, log2, log10 = (_lift(n, _method(n)) for n in ('log', 'log2', 'log10'))
sin, cos, tan, asin, acos, atan = (_lift(n, _method(n)) for n in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan'))
sinh, cosh, tanh, asinh, acosh, atanh = (_lift(n, _method(n))
                                         for n in ('sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'))
cbrt, cot, sec, csc, acot = (_lift(n, _method(n)) for n in ('cbrt', 'cot', 'sec', 'csc', 'acot'))
coth, csch, sech, acoth = (_lift(n, _method(n)) for n in ('coth', 'csch', 'sech', 'acoth'))
sign, ceil, floor, trunc = (_lift(n, _method(n)) for n in ('sign', 'ceil', 'floor', 'trunc'))
round_ties_to_even = _lift('round_ties_to_even', round)
round_ties_to_away = _lift('round_ties_to_away', _method('round_ties_away'))
intersection = _lift('intersection', lambda a, b: a & b)
convex_hull = _lift('convex_hull', lambda a, b: (a | b).hull)


def pown(x, n) -> Interval:
    """1788's `pown`: an int exponent (a float, even `2.0`, is a TypeError; `x ** 2.0` is pown by D11)"""
    n = _exponent('pown', n)
    return from_set(_call(lambda a: a ** n, *_sets('pown', (x,))[1]))


def rootn(x, n) -> Interval:
    """1788's `rootn`: an int n"""
    n = _exponent('rootn', n)
    return from_set(_call(lambda a: a.rootn(n), *_sets('rootn', (x,))[1]))


# REVERSE OPS: 1788's binary form is the last operand `x` given; omitted, the library's default (the
# affine extended reals), whose infinite points the output rule drops

def _reverse_op(name: str, library, operands, x) -> Interval:
    return from_set(_call(library, *_sets(name, operands if x is None else (*operands, x))[1]))


def sqr_rev(c, x=None) -> Interval:
    """1788's `sqrRev`"""
    return _reverse_op('sqr_rev', _reverse.sqr_rev, (c,), x)


def abs_rev(c, x=None) -> Interval:
    """1788's `absRev`"""
    return _reverse_op('abs_rev', _reverse.abs_rev, (c,), x)


def pown_rev(c, n, x=None) -> Interval:
    """1788's `pownRev`: an int n"""
    n = _exponent('pown_rev', n)
    return _reverse_op('pown_rev', lambda c, *x: _reverse.pown_rev(c, n, *x), (c,), x)


def cosh_rev(c, x=None) -> Interval:
    """1788's `coshRev`"""
    return _reverse_op('cosh_rev', _reverse.cosh_rev, (c,), x)


def sin_rev(c, x=None) -> Interval:
    """1788's `sinRev`"""
    return _reverse_op('sin_rev', _reverse.sin_rev, (c,), x)


def cos_rev(c, x=None) -> Interval:
    """1788's `cosRev`"""
    return _reverse_op('cos_rev', _reverse.cos_rev, (c,), x)


def tan_rev(c, x=None) -> Interval:
    """1788's `tanRev`"""
    return _reverse_op('tan_rev', _reverse.tan_rev, (c,), x)


def mul_rev(b, c, x=None) -> Interval:
    """1788's `mulRev`: `b` the known factor"""
    return _reverse_op('mul_rev', _reverse.mul_rev, (b, c), x)


def pow_rev1(b, c, x=None) -> Interval:
    """1788's `powRev1`: the bases"""
    return _reverse_op('pow_rev1', _reverse.pow_rev1, (b, c), x)


def pow_rev2(a, c, x=None) -> Interval:
    """1788's `powRev2`: the exponents"""
    return _reverse_op('pow_rev2', _reverse.pow_rev2, (a, c), x)


def mul_rev_to_pair(b, c):
    """
    1788's `mulRevToPair`: `{t : t y in c for some y in b}` as two intervals, the second empty unless
    the set has a gap. where 0 is not in `b` it is 1788's wording, `(div(c, b), empty)`, the first
    decorated as the division; where 0 is in `b`, the pieces of the library's `mul_rev` in order, trv

    >>> mul_rev_to_pair(Interval(-2, -0.1, 'dac'), Interval(-2.1, -0.4, 'dac'))
    (Interval(0.2, 21.0, 'dac'), Interval(decoration='trv'))
    """
    decorated = _flavour('mul_rev_to_pair', (b, c))
    b, c = _interval_of(b, decorated), _interval_of(c, decorated)
    trv = Decoration.TRV if decorated else None
    none = Interval._make(_EMPTY, trv)
    if not b._set or 0 not in b._set:
        return div(c, b), none
    real = _Outward.from_cuts(_kernel.intersection(_call(_reverse.mul_rev, b._set, c._set).cuts, _LINE.cuts))
    pieces = [Interval._make(_hull(p), trv) for p in real]
    if len(pieces) > 2:  # never for 1788-form operands: a library bug, not to be hulled away
        raise AssertionError(f'mul_rev_to_pair: {len(pieces)} pieces')
    return tuple(pieces + [none] * (2 - len(pieces)))


# CANCELLATION: 1788's "no answer" is entire

def _bounded(s) -> bool:
    return not s or s.is_finite


def cancel_minus(a, b) -> Interval:
    """
    1788's `cancelMinus`: `a` empty and `b` bounded give empty; `a`, `b` non-empty and bounded with
    `wid a >= wid b`, compared exactly, the library's Minkowski difference (then 1788's
    `[a1 - b1, a2 - b2]` outward); anything else entire, 1788's "no answer". trv when decorated
    """
    decorated, (sa, sb) = _sets('cancel_minus', (a, b))
    A, B = (s.interval if decorated else s for s in (sa, sb))
    if A and B and _bounded(A) and _bounded(B) and \
            Fraction(A.sup) - Fraction(A.inf) >= Fraction(B.sup) - Fraction(B.inf):
        return from_set(_call(lambda: sa.cancel_minus(sb)))
    s = _EMPTY if not A and _bounded(B) else _LINE
    return Interval._make(s, Decoration.TRV if decorated else None)


def cancel_plus(a, b) -> Interval:
    """1788's `cancelPlus`: `cancel_minus(a, neg(b))`"""
    decorated = _flavour('cancel_plus', (a, b))
    return cancel_minus(a, neg(_interval_of(b, decorated)))


# NUMBERS: of the interval part, python floats

def _part(name: str, x) -> _Outward:
    if not isinstance(x, Interval):
        raise TypeError(f'{name}: expected an ieee1788.Interval, got {type(x).__name__}')
    return x._set


def inf(x) -> float:
    """1788's `inf`: `+inf` for the empty set, `-0.0` for a lower end of 0"""
    s = _part('inf', x)
    if not s:
        return _INF
    return -0.0 if s.inf == 0 else s.inf


def sup(x) -> float:
    """1788's `sup`: `-inf` for the empty set, `+0.0` for an upper end of 0"""
    s = _part('sup', x)
    if not s:
        return -_INF
    return 0.0 if s.sup == 0 else s.sup


def _number(name: str, library):
    def number(x) -> float:
        return float(_call(library, _part(name, x))) + 0.0
    number.__name__ = number.__qualname__ = name
    number.__doc__ = f"ieee 1788's `{name}`, a float; the empty set raises ValueError (the library's, D9)"
    return number


mid, rad, wid = _number('mid', lambda s: s.mid()), _number('rad', lambda s: s.rad()), _number('wid', lambda s: s.wid())
mag, mig = _number('mag', lambda s: s.mag()), _number('mig', lambda s: s.mig())


def mid_rad(x):
    """1788's `midRad`: `(mid, rad)`, floats"""
    m, r = _call(lambda s: s.mid_rad(), _part('mid_rad', x))
    return float(m) + 0.0, float(r) + 0.0


# BOOLEANS: of the interval parts

def _boolean(name: str, library):
    def boolean(*operands) -> bool:
        decorated = _flavour(name, operands)
        return bool(_call(library, *(_interval_of(a, decorated)._set for a in operands)))
    boolean.__name__ = boolean.__qualname__ = name
    boolean.__doc__ = f"ieee 1788's `{name}`, of the interval parts"
    return boolean


is_empty = _boolean('is_empty', lambda a: a.is_empty)
is_entire = _boolean('is_entire', lambda a: a == _LINE)
is_singleton = _boolean('is_singleton', lambda a: a.is_degenerate and a.is_contiguous)
is_common_interval = _boolean('is_common_interval', lambda a: bool(a) and a.is_finite)
equal = _boolean('equal', lambda a, b: a == b)
subset = _boolean('subset', lambda a, b: a.issubset(b))
disjoint = _boolean('disjoint', lambda a, b: a.isdisjoint(b))
less = _boolean('less', lambda a, b: a.weakly_less(b))
strict_less = _boolean('strict_less', lambda a, b: a.strictly_less(b))
precedes = _boolean('precedes', lambda a, b: (a <= b).certainly)
strict_precedes = _boolean('strict_precedes', lambda a, b: (a < b).certainly)
interior = _boolean('interior', lambda a, b: a.within(b.interior))


def is_member(m, x) -> bool:
    """1788's `isMember`: `m` a real number; nan and ±inf never are"""
    if not _is_real(m):
        raise TypeError(f'is_member: expected a real number, got {type(m).__name__}')
    return m in _part('is_member', x)


# OVERLAP: 1788's sixteen states, on the ends as extended reals

class Overlap(enum.Enum):
    """1788's overlap states; the value is 1788's name. not the library's `Allen`, which is on cuts
    (touching closed intervals share a point, so `overlaps`) and has no empty states"""
    BOTH_EMPTY = 'bothEmpty'
    FIRST_EMPTY = 'firstEmpty'
    SECOND_EMPTY = 'secondEmpty'
    BEFORE = 'before'
    MEETS = 'meets'
    OVERLAPS = 'overlaps'
    STARTS = 'starts'
    CONTAINED_BY = 'containedBy'
    FINISHES = 'finishes'
    EQUALS = 'equals'
    FINISHED_BY = 'finishedBy'
    CONTAINS = 'contains'
    STARTED_BY = 'startedBy'
    OVERLAPPED_BY = 'overlappedBy'
    MET_BY = 'metBy'
    AFTER = 'after'


def overlap(a, b) -> Overlap:
    """1788's `overlap` (1788-2015 §10.6.4) of the interval parts"""
    decorated = _flavour('overlap', (a, b))
    a, b = _interval_of(a, decorated)._set, _interval_of(b, decorated)._set
    if not a or not b:
        return Overlap.BOTH_EMPTY if not a and not b else Overlap.FIRST_EMPTY if not a else Overlap.SECOND_EMPTY
    a1, a2, b1, b2 = a.inf, a.sup, b.inf, b.sup
    if a2 < b1:
        return Overlap.BEFORE
    if b2 < a1:
        return Overlap.AFTER
    if a1 == b1 and a2 == b2:
        return Overlap.EQUALS
    if a1 == b1:
        return Overlap.STARTS if a2 < b2 else Overlap.STARTED_BY
    if a2 == b2:
        return Overlap.FINISHES if b1 < a1 else Overlap.FINISHED_BY
    if a1 < a2 == b1 < b2:
        return Overlap.MEETS
    if b1 < b2 == a1 < a2:
        return Overlap.MET_BY
    if a1 < b1:
        return Overlap.OVERLAPS if a2 < b2 else Overlap.CONTAINS
    return Overlap.CONTAINED_BY if a2 < b2 else Overlap.OVERLAPPED_BY


# REDUCTIONS: the library's, 1788's already

sum_ = _reductions.sum_
sum_abs = _reductions.sum_abs
sum_square = _reductions.sum_sqr
dot = _reductions.dot

# 1788's own names (1788-2015's spelling; the decorated constructors under itf1788's `d-`)
NAMES = {
    'numsToInterval': nums_to_interval, 'textToInterval': text_to_interval,
    'd-numsToInterval': nums_to_decorated_interval, 'd-textToInterval': text_to_decorated_interval,
    'empty': empty, 'entire': entire,
    'newDec': new_dec, 'setDec': set_dec, 'intervalPart': interval_part, 'decorationPart': decoration_part,
    'pos': pos, 'neg': neg, 'add': add, 'sub': sub, 'mul': mul, 'div': div, 'recip': recip, 'sqr': sqr,
    'sqrt': sqrt, 'fma': fma, 'abs': abs_, 'min': min_, 'max': max_, 'pown': pown, 'pow': pow_, 'rootn': rootn,
    'hypot': hypot,
    'exp': exp, 'exp2': exp2, 'exp10': exp10, 'expm1': expm1, 'log': log, 'log2': log2, 'log10': log10,
    'logp1': logp1, 'sin': sin, 'cos': cos, 'tan': tan, 'asin': asin, 'acos': acos, 'atan': atan,
    'atan2': atan2, 'sinh': sinh, 'cosh': cosh, 'tanh': tanh, 'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
    'cbrt': cbrt, 'cot': cot, 'sec': sec, 'csc': csc, 'acot': acot, 'coth': coth, 'csch': csch, 'sech': sech,
    'acoth': acoth,
    'sign': sign, 'ceil': ceil, 'floor': floor, 'trunc': trunc, 'roundTiesToEven': round_ties_to_even,
    'roundTiesToAway': round_ties_to_away,
    'sqrRev': sqr_rev, 'absRev': abs_rev, 'pownRev': pown_rev, 'sinRev': sin_rev, 'cosRev': cos_rev,
    'tanRev': tan_rev, 'coshRev': cosh_rev, 'mulRev': mul_rev, 'powRev1': pow_rev1, 'powRev2': pow_rev2,
    'mulRevToPair': mul_rev_to_pair,
    'cancelMinus': cancel_minus, 'cancelPlus': cancel_plus, 'intersection': intersection,
    'convexHull': convex_hull,
    'inf': inf, 'sup': sup, 'mid': mid, 'wid': wid, 'rad': rad, 'mag': mag, 'mig': mig, 'midRad': mid_rad,
    'isEmpty': is_empty, 'isEntire': is_entire, 'isMember': is_member, 'isSingleton': is_singleton,
    'isCommonInterval': is_common_interval, 'equal': equal, 'subset': subset, 'disjoint': disjoint,
    'less': less, 'strictLess': strict_less, 'precedes': precedes, 'strictPrecedes': strict_precedes,
    'interior': interior,
    'overlap': overlap,
    'sum': sum_, 'sumAbs': sum_abs, 'sumSquare': sum_square, 'dot': dot,
}
