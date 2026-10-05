"""
formatting and parsing cut tuples

grammar (v1's, made strict -- anything left over is a ValueError):

    {}                          empty
    [1, 2)                      one piece
    [5]  or  5                  a degenerate piece
    { [1, 2) , [3] }            several pieces; `,` `;` `|` `∪` or nothing between them
    {1, 2, 3}                   a set of points

numbers are ints, floats (`2.0`, `1e-05`), fractions (`1/3`) or `inf`/`-inf` (also `∞`). an int, and
each part of a fraction, may also be hex (`0x1f`, `-0x1f/0x3`, `1/0x10`). white space may follow the
one sign (`- 5`) and surround `/` (`1 / 3`), nowhere else inside a number: `+-5`, `1/-3`, `1_000`, `0x 1f`
and `1e 5` are ValueErrors, and `1 2` is two numbers (to `parse_value`, a ValueError; m14b-open, 2026-10-05).
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

# a hex int is never followed by `.`: `0x1.8p1` is a hex float, which is not read (and not `0x1` then `.8`).
# its digits are possessive (`++`): with `+` the refused `0x12.5` backtracked to `0x1`, whose next character
# is `2`, and read as `0x1` then `2.5` (m14b-open, 2026-10-05)
_INT = r'(?:0x[0-9a-f]++(?!\.)|\d+)'
# every white-space run is possessive (`\s*+`): what follows each one (`/`, a digit, `.`, `0x`, inf, ∞, a
# bracket) never begins with white space, so giving spaces back can never make a match, and a run that no
# token follows (`' ' * 30000 + 'x'`) is refused in linear time, not quadratic (m14b-open, 2026-10-04)
_NUMBER = (r'[+-]?\s*+(?:inf(?:inity)?|∞|0x[0-9a-f]++(?!\.)(?:\s*+/\s*+' + _INT + r')?'
           r'|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*+/\s*+' + _INT + r')?)')
_TOKEN = re.compile(fr'\s*+(?:(?P<num>{_NUMBER})|(?P<punct>[\[\](){{}},;|∪]))\s*+', flags=re.IGNORECASE)
# `parse_value`'s whole text: one number of the grammar, white space around it
_VALUE = re.compile(fr'\s*+{_NUMBER}\s*+', flags=re.IGNORECASE)
_SEPARATORS = {',', ';', '|', '∪'}


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
    `'+-5'` was 5 and `'1 2'` was 12

    >>> parse_value(' - 5'), parse_value('1 / 0x3')
    (-5, Fraction(1, 3))
    """
    if _VALUE.fullmatch(text) is None:
        shown = text if len(text) <= 40 else f'{text[:20]}...{text[-10:]}'
        raise ValueError(f'cannot parse {shown!r}: not a number')
    text = ''.join(text.split())
    sign = -1 if text.startswith('-') else 1
    body = text.lstrip('+-').lower()
    if body in ('inf', 'infinity', '∞'):
        return sign * math.inf
    if '/' in body:
        numerator, denominator = body.split('/')
        try:
            return sign * Fraction(_parse_int(numerator, text), _parse_int(denominator, text))
        except ZeroDivisionError:  # a zero denominator is bad text like any other (M14-breadth)
            raise ValueError(f'cannot parse {text!r}: a zero denominator') from None
    if any(c in body for c in '.e') and not body.startswith('0x'):
        return sign * float(body)
    return sign * _parse_int(body, text)


def _parse_int(body: str, text: str) -> int:
    """
    a decimal or `0x` hex int. python refuses a decimal of more than `sys.get_int_max_str_digits()`
    digits; the ValueError then names the hex form, which has no limit
    """
    if body.startswith('0x'):
        return int(body, 16)
    try:
        return int(body)
    except ValueError as e:
        if not body.isdigit():  # `1.5/2`: not an int at all
            raise ValueError(f'cannot parse {text!r}: {body!r} is not an int') from None
        shown = text if len(text) <= 40 else f'{text[:20]}...{text[-10:]}'
        raise ValueError(f'cannot parse {shown!r}: {e} (an int that long can be written in hex, 0x..., '
                         f'which has no limit)') from None


def _tokenize(text: str) -> List[Tuple[str, str]]:
    tokens = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None or match.end() == pos:
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
        while True:
            kind, value = self.peek()
            if kind is None:
                if braced:
                    raise self.error('missing "}"')
                break
            if braced and value == '}':
                self.take()
                if self.pos != len(self.tokens):
                    raise self.error('text after "}"')
                break
            if kind == 'punct' and value in _SEPARATORS:
                if not items:
                    raise self.error(f'{value!r} before the first item')
                self.take()
                continue
            found.extend(self.item())
            items += 1
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
        while self.peek()[0] == 'num':
            numbers.append(parse_value(self.take()[1]))
            if self.peek() in (('punct', ','), ('punct', ';')):
                self.take()
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
