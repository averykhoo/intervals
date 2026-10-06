"""
formatting and parsing cut tuples

grammar (v1's, made strict -- anything left over is a ValueError):

    {}                          empty
    [1, 2)                      one piece
    [5]  or  5                  a degenerate piece
    { [1, 2) , [3] }            several pieces; `,` `;` `|` or `∪` between them
    {1, 2, 3}                   a set of points

two items need a separator between them, and the two numbers of a piece a `,` or `;`. white space only pads
(around a separator, a bracket, a brace or `/`) and separates nothing, since it may sit inside a fraction
(`1 / 3`): `[1 2]`, `{1 2}`, `1 2`, `{[1, 2] [3]}` and `[1, 2)[3, 4)` are ValueErrors (owner, Q23, 2026-10-06).
a separator stands between two items, or two numbers, and nowhere else: a leading, trailing or doubled one is a
ValueError, padded or not and in any mix (`{,1}`, `{1,}`, `[1, 2],`, `[1,]`, `(1,)`, `{1,,2}`, `{1 | , 2}`,
`[1,;2]`, `{[], }`; owner, Q24, 2026-10-06). the empty forms stay: the empty text, `{}`, `[]`, `()`, and empty
pieces among a set's items (`{[], [1]}`).

a number is what python reads (owner, Q23, 2026-10-06): an int as `int` reads it (`5`, `-5`, `007`, `1_000`), a
float as `float` does (`2.0`, `1.`, `.5`, `1e-05`, `1_0.5e1_0`, `inf` and `infinity` in any case, but not nan),
or two ints as `Fraction` reads them (`1/3`, `-1 / 3`, `1_000/3`); `∞` is also infinity. an int, and each part of
a fraction, may also be hex as `int(s, 0)` reads it (`0x1f`, `0X_1F`, `-0x1f/0x3`, `1/0x10`). so the one sign is
attached and first (`- 5`, `+-5` and `1/-3` are errors, as `float('- 5')` is), white space inside a number is
allowed only around `/`, and `_` only between two digits (`_1`, `1_`, `1__0`, `1_.5`, `1._5` and `1e_5` are
errors). digits are ASCII `0-9` only: python's `int`, `float` and `Fraction` also read other scripts' digits
(`'١٢'` is 12 to them), the grammar does not. a number ends at white space, punctuation or the end of the text;
anything else after it is an error, never the start of another number: `[0.1.2]`, `[-2-1]`, `{1-2}`, `[1+2]`,
`[0E0-0]` and `[inf0x1]` are ValueErrors (they were two numbers before Q23).
`format` prints ints and fractions exactly and floats by `repr`, so `parse(format(x)) == x`. an int part
python will not write in decimal (more than `sys.get_int_max_str_digits()` digits, 4300 by default:
python's guard against slow conversions) is written in hex, which python converts in linear time and
without a limit, so `repr` never raises and still reads back (owner, 2026-10-03, m14b-open). `parse`
keeps python's limit on a decimal literal: `parse('1' * 4301)` raises ValueError, naming the hex form
"""
import math
import re
from fractions import Fraction
from typing import List
from typing import Tuple

from multiinterval.cuts import Value
from multiinterval.cuts import as_end
from multiinterval.cuts import as_start
from multiinterval.kernel import Cuts
from multiinterval.kernel import normalize
from multiinterval.kernel import pairs
from multiinterval.kernel import piece

# the number grammar, python's own (Q23, 2026-10-06), spelled without re.IGNORECASE (which lets `ı` and `İ` match
# `i`) and without `\d` (680 code points): ASCII only. every repetition is possessive (`++`, `*+`, `?+`): a digit,
# `_` or white-space run is read once, and a number that cannot end where its run ends fails at once instead of
# giving characters back one at a time, so a text no number reads is refused in linear time (m14b-open, 2026-10-04:
# `parse(' ' * 30000 + 'x')` took 37 s; Q23 adds the digit and `_` runs). `_` is python's: one between two digits
_DIGITS = r'[0-9]++(?:_[0-9]++)*+'
# `int(s, 0)`'s hex: `0x`, then hex digits, each of which may follow one `_` (`0x_1f`, `0x1_f`)
_HEX = r'0[xX](?:_?+[0-9a-fA-F])++'
_INT = fr'(?:{_HEX}|{_DIGITS})'
# `float`'s: `1`, `1.`, `1.5`, `.5`, each with an exponent or not
_FLOAT = fr'(?:{_DIGITS}(?:\.(?:{_DIGITS})?+)?+|\.{_DIGITS})(?:[eE][+-]?+{_DIGITS})?+'
_INFINITY = r'(?:[iI][nN][fF](?:[iI][nN][iI][tT][yY])?+|∞)'
_PUNCT = r'[\[\](){},;|∪]'
# a sign, then the body; the fraction first, so `1 / 3` is not read as `1`
_BODY = fr'[+-]?+(?:{_INFINITY}|{_INT}\s*+/\s*+{_INT}|{_HEX}|{_FLOAT})'
# a number ends at white space, punctuation or the end: `[0.1.2]` is not `[0.1, 0.2]` (Q23)
_NUMBER = fr'{_BODY}(?![^\s\[\](){{}},;|∪])'
_TOKEN = re.compile(fr'\s*+(?:(?P<num>{_NUMBER})|(?P<punct>{_PUNCT}))\s*+')
# a number not followed by an end: only to say so in the error
_UNENDED = re.compile(fr'\s*+{_BODY}')
# `parse_value`'s whole text: one number of the grammar, white space around it
_VALUE = re.compile(fr'\s*+(?P<number>{_NUMBER})\s*+')
_SEPARATORS = {',', ';', '|', '∪'}
_PIECE_SEPARATORS = (('punct', ','), ('punct', ';'))


# FORMAT

def format_value(value: Value) -> str:
    """
    a float by `repr` (exact; also 'inf' / '-inf'), an int in decimal, a Fraction as `p/q`; an int part
    past python's int-str limit in hex

    >>> format_value(-1 / 3), format_value(Fraction(-7, 2)), format_value(-(10 ** 4300))[:12]
    ('-0.3333333333333333', '-7/2', '-0x1392bd7c2')
    """
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Fraction) and value.denominator != 1:
        return f'{_format_int(value.numerator)}/{_format_int(value.denominator)}'
    return _format_int(int(value))


def _format_int(n: int) -> str:
    try:
        return str(n)
    except ValueError:  # past sys.get_int_max_str_digits(): hex has no limit and is linear
        return hex(n)


def format_piece(start, end) -> str:
    lo, lo_closed = as_start(start)
    hi, hi_closed = as_end(end)
    if lo == hi and type(lo) is type(hi):  # a point whose cuts differ in type (`[1.0, 1]`) keeps both
        return f'[{format_value(lo)}]'
    return f'{"[" if lo_closed else "("}{format_value(lo)}, {format_value(hi)}{"]" if hi_closed else ")"}'


def format_cuts(cuts: Cuts) -> str:
    """
    `{}` when empty, a bare piece when contiguous, else `{ A , B }`

    >>> format_cuts(parse('[1, 2) | [3]'))
    '{ [1, 2) , [3] }'
    """
    parts = [format_piece(start, end) for start, end in pairs(cuts)]
    if not parts:
        return '{}'
    if len(parts) == 1:
        return parts[0]
    return f'{{ {" , ".join(parts)} }}'


# PARSE

def parse_value(text: str) -> Value:
    """
    one number of the grammar (the module docstring), white space around it; anything else is a ValueError.
    it read any text before (m14b-open, 2026-10-05): it deleted all white space and every leading sign, so
    `'+-5'` was 5 and `'1 2'` was 12. the type is the shape's: a fraction is a Fraction, a number with `.` or
    an exponent a float, else an int

    >>> parse_value(' -5'), parse_value('1 / 0x3'), parse_value('1_000.5')
    (-5, Fraction(1, 3), 1000.5)
    """
    match = _VALUE.fullmatch(text)
    if match is None:
        raise ValueError(f'cannot parse {_shown(text)!r}: not a number')
    number = match['number']
    sign = -1 if number[0] == '-' else 1
    body = number.lstrip('+-')
    if body[0] in 'iI∞':
        return sign * math.inf
    if '/' in body:
        numerator, denominator = body.split('/')
        numerator, denominator = _parse_int(numerator.strip(), text), _parse_int(denominator.strip(), text)
        if denominator == 0:  # a zero denominator is bad text like any other (M14-breadth)
            raise ValueError(f'cannot parse {_shown(text)!r}: a zero denominator')
        return sign * Fraction(numerator, denominator)
    if body[:2] not in ('0x', '0X') and any(c in body for c in '.eE'):
        return float(number)  # the sign too: `-0.0`
    return sign * _parse_int(body, text)


def _parse_int(body: str, text: str) -> int:
    """
    a decimal or `0x` hex int, as `int` and `int(s, 0)` read them. python refuses a decimal of more than
    `sys.get_int_max_str_digits()` digits; the ValueError then names the hex form, which has no limit
    """
    if body[:2] in ('0x', '0X'):
        return int(body, 0)
    try:
        return int(body)
    except ValueError as e:  # the grammar has checked the digits: only python's limit is left
        raise ValueError(f'cannot parse {_shown(text)!r}: {e} (an int that long can be written in hex, 0x..., '
                         f'which has no limit)') from None


def _shown(text: str) -> str:
    return text if len(text) <= 40 else f'{text[:20]}...{text[-10:]}'


def _tokenize(text: str) -> List[Tuple[str, str]]:
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            unended = _UNENDED.match(text, pos)
            if unended is not None:  # a number, then a character that can neither end it nor go on with it (Q23)
                at = unended.end()
                raise ValueError(f'cannot parse {_shown(text)!r} at position {at}: a number must end at white '
                                 f'space, punctuation or the end of the text, not {text[at:at + 10]!r}')
            raise ValueError(f'cannot parse {text!r} at position {pos}: {text[pos:pos + 10]!r}')
        tokens.append(('num', match['num']) if match['num'] is not None else ('punct', match['punct']))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    def error(self, message: str) -> ValueError:
        return ValueError(f'cannot parse {self.text!r}: {message}')

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self) -> Cuts:
        braced = self.peek() == ('punct', '{')
        if braced:
            self.take()
        found, items = [], 0  # an empty piece is an item that adds nothing to found
        separator = None  # the separator read since the last item (Q23: white space alone is none)
        while True:
            kind, value = self.peek()
            if kind is None:
                if braced:
                    raise self.error('missing "}"')
            elif braced and value == '}':
                self.take()
                if self.pos != len(self.tokens):
                    raise self.error('text after "}"')
            elif kind == 'punct' and value in _SEPARATORS:  # one between two items, nowhere else (Q24)
                if not items:
                    raise self.error(f'{value!r} before the first item')
                if separator is not None:
                    raise self.error(f'two separators in a row, {separator!r} then {value!r}')
                separator = self.take()[1]
                continue
            else:
                if items and separator is None:
                    raise self.error(f'two items need ",", ";", "|" or "∪" between them, before {value!r}')
                found.extend(self.item())
                items += 1
                separator = None
                continue
            if separator is not None:  # the end: `{1,}` and `[1],` (Q24)
                raise self.error(f'{separator!r} after the last item')
            return normalize(found)

    def item(self):
        kind, value = self.take()
        if kind == 'num':
            point = parse_value(value)
            return [piece(point, point)]
        if value not in ('[', '('):
            raise self.error(f'unexpected {value!r}')
        lo_closed = value == '['
        numbers = []
        if self.peek() in _PIECE_SEPARATORS:  # `[,1]`
            raise self.error(f'{self.peek()[1]!r} before the first number')
        while self.peek()[0] == 'num':
            numbers.append(parse_value(self.take()[1]))
            if self.peek() in _PIECE_SEPARATORS:  # one between two numbers, nowhere else (Q24)
                separator = self.take()[1]
                if self.peek() in _PIECE_SEPARATORS:  # `[1,,2]`
                    raise self.error(f'two separators in a row, {separator!r} then {self.peek()[1]!r}')
                if self.peek()[0] != 'num':  # `[1,]`
                    raise self.error(f'{separator!r} after the last number')
            elif self.peek()[0] == 'num':  # Q23: white space alone is no separator
                raise self.error(f'two numbers need "," or ";" between them, before {self.peek()[1]!r}')
        kind, close = self.take()
        if close not in (']', ')'):
            raise self.error(f'expected "]" or ")", got {close!r}')
        hi_closed = close == ']'
        if not numbers:
            return []  # `[]` and `()` are empty
        if len(numbers) == 1:
            if lo_closed != hi_closed:
                raise self.error(f'half-open degenerate interval at {numbers[0]!r}')
            return [piece(numbers[0], numbers[0], lo_closed, hi_closed)]  # `(x)` is empty
        if len(numbers) > 2:
            raise self.error('more than two numbers in one interval')
        return [piece(numbers[0], numbers[1], lo_closed, hi_closed)]


def parse(text: str) -> Cuts:
    """
    parse the grammar in the module docstring into a normalized cut tuple

    >>> format_cuts(parse('{[1,2), [2,3]}'))
    '[1, 3]'
    """
    return _Parser(text).parse()
