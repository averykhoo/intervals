"""
ieee 1788's interval literals (1788-2015 §9.7), and the bare constructors `text_to_interval` and
`nums_to_interval` (1788's `b-textToInterval`, `b-numsToInterval`; M13g, D16)

this is 1788's text syntax, not the package's own: `MultiInterval.parse` (`multiinterval.fmt`) reads
`(1, 2]` and `{ [1, 2) , [3] }`, which 1788 cannot say. a 1788 literal is one of

    [l, u]  [x]  [l,]  [,u]  [,]  [ ]  [empty]  [entire]      the inf-sup forms
    m?r  m?  m??  with a direction u or d and an exponent e    the uncertain form, `3.56?1e2`
    any of these followed by a decoration `_com` `_dac` `_def` `_trv`

a number is a decimal (`1.`, `.5`, `1.e-3`), a hexadecimal with a binary exponent (`-0x1.3p-1`), a
rational `p/q` of decimal integers, or `inf`/`infinity`, each with an optional sign. letters are
read in any case (`[1,1E3]_COM`), and white space is allowed inside the brackets only, around a
number or a word, never inside one and never outside the literal (`" [1, 2]"` is invalid).

every value is exact: a decimal is the rational it spells, so `[0.1]` is the point `1/10`, not a
double (1788's inf-sup binary64 type would give the two doubles around it). an empty side of `[l,]`
is an infinity. the uncertain form `m?r` is `m` plus or minus `r` units in the last decimal place of
`m` (`r` omitted: half a unit; `??`: an infinite radius), only upward with `u` and only downward
with `d`, all times ten to the exponent: `3.56?1` is `[3.55, 3.57]`, `-10?u` is `[-10, -9.5]`.

invalid input raises `UndefinedOperationError` (a `ValueError`), as 1788's `UndefinedOperation`
but stopping (owner, 2026-09-26): a malformed string, a bound in the wrong order (`[2, 1]`, decided
exactly, so `[1.0000000000000002, 1.0000000000000001]` is invalid even though both are near the same
double), `[+inf]`, `[inf, inf]`, a decoration that does not fit the value (`[1,]_com`, `[ ]_def`),
`_ill`, and NaI (`[nai]`): the package has no NaI (D16). a non-str argument is a `TypeError`.

>>> text_to_interval('[1, 2]')
MultiInterval.parse('[1, 2]')
>>> text_to_interval('[1, +infinity]'), text_to_interval('[,]')
(MultiInterval.parse('[1, inf)'), MultiInterval.parse('(-inf, inf)'))
>>> text_to_interval('3.56?1'), text_to_interval('[ empty ]')
(MultiInterval.parse('[71/20, 357/100]'), MultiInterval.parse('{}'))
>>> parse_literal('[-0x1.3p-1, 2/3]_def')
Literal(lo=Fraction(-19, 32), hi=Fraction(2, 3), decoration='def')
>>> nums_to_interval(-1, float('inf'))
MultiInterval.parse('[-1, inf)')
>>> text_to_interval('[2, 1]')
Traceback (most recent call last):
    ...
multiinterval.errors.UndefinedOperationError: invalid 1788 interval literal '[2, 1]': the lower bound exceeds the upper
"""
import math
import re
from fractions import Fraction
from typing import NamedTuple
from typing import Optional

from multiinterval.cuts import Value
from multiinterval.cuts import normalize_value
from multiinterval.errors import UndefinedOperationError
from multiinterval.multi_interval import MultiInterval

INF = math.inf
# 1788's decorations, best first; `ill` belongs to NaI, which the package does not have (D16)
DECORATIONS = ('com', 'dac', 'def', 'trv')

# M13g review: every run of digits or white space splits one way only (`[0-9]+(?:\.[0-9]*)?`, not
# `[0-9]+\.?[0-9]*`; `\s*+`), so invalid text is refused in linear time, not cubic
_DECIMAL = r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[-+]?[0-9]+)?'
_HEX = r'0x(?:[0-9a-f]+(?:\.[0-9a-f]*)?|\.[0-9a-f]+)p[-+]?[0-9]+'
_RATIONAL = r'[0-9]+/[0-9]+'
_NUMBER = rf'[-+]?(?:{_HEX}|{_RATIONAL}|{_DECIMAL}|inf(?:inity)?)'
_FLAGS = re.IGNORECASE | re.ASCII
_LITERAL = re.compile(
    rf'(?:\[\s*+(?:(?P<word>empty|entire|nai)|(?P<lo>{_NUMBER})?\s*+(?:(?P<comma>,)\s*+(?P<hi>{_NUMBER})?)?)\s*+\]'
    rf'|(?P<sign>[-+]?)(?P<m>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\?(?P<radius>[0-9]+|\?)?(?P<direction>[ud])?'
    rf'(?:e(?P<exponent>[-+]?[0-9]+))?)'
    rf'(?:_(?P<decoration>[a-z]+))?', _FLAGS)
_HEX_PARTS = re.compile(r'([-+]?)0x([0-9a-f]*)\.?([0-9a-f]*)p([-+]?[0-9]+)', _FLAGS)
_DECIMAL_PARTS = re.compile(r'([-+]?)([0-9]*)\.?([0-9]*)(?:e([-+]?[0-9]+))?', _FLAGS)


class Literal(NamedTuple):
    """a parsed 1788 literal: exact ends (`None` for the empty set) and its decoration, if it has one"""
    lo: Optional[Value]
    hi: Optional[Value]
    decoration: Optional[str] = None

    @property
    def empty(self) -> bool:
        return self.lo is None


def _invalid(text: str, why: str) -> UndefinedOperationError:
    return UndefinedOperationError(f'invalid 1788 interval literal {text!r}: {why}')


def _decimal(sign: str, whole: str, fraction: str, exponent: Optional[str]) -> Fraction:
    value = Fraction(int(whole + fraction or '0'), 10 ** len(fraction)) * Fraction(10) ** int(exponent or 0)
    return -value if sign == '-' else value


def number(text: str) -> Value:
    """
    a 1788 number literal's exact value: int or Fraction, ±inf as floats (the package's infinities).
    `text` must already match the number grammar

    >>> number('0x1.8p1'), number('-1.e-3'), number('10/4'), number('-Inf')
    (3, Fraction(-1, 1000), Fraction(5, 2), -inf)
    """
    lowered = text.lower()
    if lowered.lstrip('+-') in ('inf', 'infinity'):
        return -INF if lowered.startswith('-') else INF
    if '/' in text:
        p, q = text.split('/')
        if int(q) == 0:
            raise ValueError(f'a zero denominator in {text!r}')
        return normalize_value(Fraction(int(p), int(q)))
    m = _HEX_PARTS.fullmatch(text)
    if m is not None:
        sign, whole, fraction, exponent = m.groups()
        value = Fraction(int(whole + fraction or '0', 16)) * Fraction(2) ** (int(exponent) - 4 * len(fraction))
        return normalize_value(-value if sign == '-' else value)
    return normalize_value(_decimal(*_DECIMAL_PARTS.fullmatch(text).groups()))


def parse_literal(text: str) -> Literal:
    """
    a 1788 interval literal, exactly, with its decoration (lower case) if it has one. raises
    `UndefinedOperationError` for anything invalid (the module docstring), NaI included
    """
    if not isinstance(text, str):
        raise TypeError(f'expected a str, got {type(text).__name__}')
    m = _LITERAL.fullmatch(text)
    if m is None:
        raise _invalid(text, 'not a literal')
    try:
        lo, hi = _bounds(m)
    except ValueError as e:  # a zero denominator, or the reason for the bounds' invalidity
        raise _invalid(text, str(e)) from None
    decoration = m.group('decoration')
    if decoration is not None:
        decoration = decoration.lower()
        if decoration not in DECORATIONS:
            raise _invalid(text, f'no decoration {decoration!r}')
        if lo is None and decoration != 'trv':
            raise _invalid(text, 'the empty set is decorated trv only')
        if decoration == 'com' and (lo is None or _is_infinite(lo) or _is_infinite(hi)):
            raise _invalid(text, 'com is for bounded non-empty intervals only')
    return Literal(lo, hi, decoration)


def _bounds(m):
    """(lo, hi) of a matched literal, `(None, None)` for empty; ValueError for an invalid one"""
    word = (m.group('word') or '').lower()
    if word == 'nai':
        raise ValueError('NaI: the package has no NaI (D16)')
    if word == 'empty':
        return None, None
    if word == 'entire':
        return -INF, INF
    if m.group('m') is not None:
        return _uncertain(m)
    lo, hi = m.group('lo'), m.group('hi')
    if m.group('comma') is None:
        if lo is None:
            return None, None  # `[ ]`
        x = number(lo)
        if _is_infinite(x):  # not math.isfinite, which overflows on a big int (`[1e400]`)
            raise ValueError('a point is finite')
        return x, x
    lo = -INF if lo is None else number(lo)
    hi = INF if hi is None else number(hi)
    if lo == INF or hi == -INF:
        raise ValueError('a lower bound of +inf or an upper bound of -inf')
    if lo > hi:
        raise ValueError('the lower bound exceeds the upper')
    return lo, hi


def _uncertain(m):
    """the uncertain form `m?r` with its direction and exponent"""
    whole, _, fraction = m.group('m').partition('.')
    exponent = m.group('exponent')
    middle = _decimal(m.group('sign'), whole, fraction, exponent)
    radius = m.group('radius')
    if radius == '?':
        lo, hi = -INF, INF
    else:
        units = Fraction(1, 2) if radius is None else Fraction(int(radius))
        r = units / 10 ** len(fraction) * Fraction(10) ** int(exponent or 0)
        lo, hi = middle - r, middle + r
    direction = (m.group('direction') or '').lower()
    if direction == 'u':
        lo = middle
    elif direction == 'd':
        hi = middle
    return _value(lo), _value(hi)


def _is_infinite(v) -> bool:
    return v == INF or v == -INF


def _value(v):
    return v if _is_infinite(v) else normalize_value(v)


def _bare(lo, hi) -> MultiInterval:
    """1788's input rule: a finite end is closed and an infinite one open, since 1788 never attains
    infinity"""
    if lo is None:
        return MultiInterval()
    return MultiInterval(lo, hi, start_closed=lo != -INF, end_closed=hi != INF)


def text_to_interval(text: str) -> MultiInterval:
    """
    1788's `b-textToInterval`: the bare interval a 1788 literal denotes, exactly. a decorated literal
    is invalid here (1788's bare constructor refuses it too), as is any invalid one:
    `UndefinedOperationError`

    >>> text_to_interval('-10?u'), text_to_interval('0.0??d'), text_to_interval('[-1/10, 0x1p-1]')
    (MultiInterval.parse('[-10, -19/2]'), MultiInterval.parse('(-inf, 0]'), MultiInterval.parse('[-1/10, 1/2]'))
    """
    literal = parse_literal(text)
    if literal.decoration is not None:
        raise _invalid(text, 'a decorated literal is not a bare interval')
    return _bare(literal.lo, literal.hi)


def nums_to_interval(lo, hi) -> MultiInterval:
    """
    1788's `b-numsToInterval`: `[lo, hi]` with an infinite end open (1788's input rule). invalid,
    raising `UndefinedOperationError`: `lo > hi`, `lo = +inf`, `hi = -inf` or a `nan`. anything not
    a real number is a `TypeError`

    >>> nums_to_interval(1, 2), nums_to_interval(float('-inf'), 0.5)
    (MultiInterval.parse('[1, 2]'), MultiInterval.parse('(-inf, 0.5]'))
    """
    try:
        lo, hi = normalize_value(lo), normalize_value(hi)
    except ValueError:  # the only one normalize_value raises is for nan
        raise UndefinedOperationError(f'invalid bounds for 1788 numsToInterval: {lo!r}, {hi!r}: nan') from None
    if lo > hi or lo == INF or hi == -INF:
        raise UndefinedOperationError(f'invalid bounds for 1788 numsToInterval: {lo!r}, {hi!r}')
    return _bare(lo, hi)
