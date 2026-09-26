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

from intervals.errors import UndefinedOperationError
from intervals.literals import _bare
from intervals.literals import nums_to_interval
from intervals.literals import parse_literal
from intervals.multi_interval import MultiInterval


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
