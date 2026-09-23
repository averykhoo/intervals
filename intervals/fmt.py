"""
formatting and parsing cut tuples

grammar (v1's, made strict -- anything left over is a ValueError):

    {}                          empty
    [1, 2)                      one piece
    [5]  or  5                  a degenerate piece
    { [1, 2) , [3] }            several pieces; `,` `;` `|` `∪` or nothing between them
    {1, 2, 3}                   a set of points

numbers are ints, floats (`2.0`, `1e-05`), fractions (`1/3`) or `inf`/`-inf` (also `∞`).
`format` prints ints and fractions exactly and floats by `repr`, so `parse(format(x)) == x`.
"""
import math
import re
from fractions import Fraction
from typing import List
from typing import Tuple

from intervals.cuts import Value
from intervals.cuts import as_end
from intervals.cuts import as_start
from intervals.kernel import Cuts
from intervals.kernel import normalize
from intervals.kernel import pairs
from intervals.kernel import piece

_NUMBER = r'[+-]?\s*(?:inf(?:inity)?|∞|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*\d+)?)'
_TOKEN = re.compile(fr'\s*(?:(?P<num>{_NUMBER})|(?P<punct>[\[\](){{}},;|∪]))\s*', flags=re.IGNORECASE)
_SEPARATORS = {',', ';', '|', '∪'}


# FORMAT

def format_value(value: Value) -> str:
    if isinstance(value, float):
        return repr(value)  # round-trips exactly; also 'inf' / '-inf'
    return str(value)  # int, or Fraction as 'p/q'


def format_piece(start, end) -> str:
    lo, lo_closed = as_start(start)
    hi, hi_closed = as_end(end)
    if lo == hi:
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
    text = ''.join(text.split())
    sign = -1 if text.startswith('-') else 1
    body = text.lstrip('+-').lower()
    if body in ('inf', 'infinity', '∞'):
        return sign * math.inf
    if '/' in body:
        return sign * Fraction(body)
    if any(c in body for c in '.e'):
        return sign * float(body)
    return sign * int(body)


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
        found = []
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
                if not found:
                    raise self.error(f'{value!r} before the first item')
                self.take()
                continue
            found.extend(self.item())
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
