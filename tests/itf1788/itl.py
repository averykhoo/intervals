"""
a parser for itf1788's `itl` vector language, as the vendored files use it

a file is `testcase <name> { <statement>; ... }` blocks with `/* */` and `//` comments. a statement
is `<op> <arg> ... = <expected>`, optionally followed by `signal <Name>` (1788's signal, kept on the
vector). every value is a literal: an interval (`[a, b]`, `[empty]`, `[entire]`, `[nai]`, optionally
decorated `_trv`/`_def`/`_dac`/`_com`/`_ill`), an integer (pown's exponent), a number (`isMember`'s
point, `inf`'s result: a `Fraction`, ±inf or `NaN`), `true`/`false`, a word (an overlap state such as
`before` or `containedBy`, a decoration such as `com`: a str), a quoted string (`b-textToInterval
"[1, 2]"`: a `Text`) or a list of numbers (`sum_nearest {1.0, 2.0}`: a tuple). a result is one value
or two (`mulRevToPair ... = [empty] [empty]`, `midRad ... = 0.0 infinity`: a tuple). only statements
whose op is in `ops` are parsed; the rest are counted by op name. a parsed statement with anything
else in it raises, and so does text outside a testcase, so a vector is never dropped silently.

numbers are the literal's nearest double, as the libieeep1788 C++ tests the files were converted
from read them (`pown [13.1, 13.1] 2` expects a one-ulp result, which an outward-rounded 13.1 could
not give), held as the double's exact Fraction.

>>> parse_literal('[-0.0, 0X1.8P+1]_com')
Interval(lo=Fraction(0, 1), hi=Fraction(3, 1), decoration='com', nai=False)
>>> parse_literal('[-infinity, 2]')
Interval(lo=-inf, hi=Fraction(2, 1), decoration=None, nai=False)
>>> parse_literal('[entire]'), parse_literal('[empty]').empty
(Interval(lo=-inf, hi=inf, decoration=None, nai=False), True)
>>> parse_literal('-0.5'), parse_literal('+infinity'), parse_literal('true'), parse_literal('metBy')
(Fraction(-1, 2), inf, True, 'metBy')
>>> parse_literal('NaN'), parse_literal('"[1, 2]"'), parse_literal('{1.0, -infinity}')
(nan, Text(value='[1, 2]'), (Fraction(1, 1), -inf))
>>> v = parse_statement('mulRevToPair [empty]_trv [1.0, 2.0] = [empty] [empty] signal Foo')
>>> len(v.expected), v.signal, strip_decorations(v.text)
(2, 'Foo', 'mulRevToPair [empty] [1.0, 2.0] = [empty] [empty] signal Foo')
"""
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

_SIGN = r'[-+]?'
_HEX = _SIGN + r'0[xX](?:[0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.?)[pP][-+]?[0-9]+'
_DEC = _SIGN + r'(?:[0-9]*\.[0-9]+|[0-9]+\.?)(?:[eE][-+]?[0-9]+)?'
_NUMBER = rf'(?:{_HEX}|{_DEC}|{_SIGN}infinity)'
_DECORATION = r'(?:_(?P<decoration>trv|def|dac|com|ill))?'
_INTERVAL = re.compile(
    rf'\[\s*(?:(?P<special>empty|entire|nai)|(?P<lo>{_NUMBER})\s*,\s*(?P<hi>{_NUMBER}))\s*\]{_DECORATION}$')
_INTEGER = re.compile(r'[-+]?[0-9]+$')
_NUMBER_LITERAL = re.compile(rf'(?:{_NUMBER}|NaN)$')
_WORD = re.compile(r'[A-Za-z]+$')
_TEXT = re.compile(r'"([^"]*)"$')
_LIST = re.compile(r'\{([^{}]*)\}$')
# one token: a quoted string, a list, an interval, `=`, or any other run of non-space characters
_TOKEN = re.compile(r'"[^"]*"|\{[^{}]*\}|\[[^\]]*\](?:_[a-z]+)?|=|[^\s\[\]{}"=]+')
_TESTCASE = re.compile(r'testcase\s+([\w.]+)\s*\{')  # atan2.itl has 'minimal.atan2_test'
# a comment; a quoted string is matched first so that a `//` or `/*` inside one is not taken for a comment
_COMMENT = re.compile(r'"[^"]*"|/\*.*?\*/|//[^\n]*', re.DOTALL)
# a run of white space outside a quoted string; a quoted string is matched first so that it keeps its own
_SPACE = re.compile(r'"[^"]*"|\s+')
# a decoration on an interval literal; a quoted string is matched first so that it is left alone
_DECORATED = re.compile(r'"[^"]*"|(\])_(?:trv|def|dac|com|ill)\b')


class Interval(NamedTuple):
    """a bare or decorated 1788 interval; `lo`/`hi` are exact, `None` for empty and NaI"""
    lo: Optional[Union[Fraction, float]]
    hi: Optional[Union[Fraction, float]]
    decoration: Optional[str] = None
    nai: bool = False

    @property
    def empty(self) -> bool:
        return self.lo is None and not self.nai


class Text(NamedTuple):
    """a quoted string, the text constructors' argument"""
    value: str


Literal = Union[Interval, int, Fraction, float, bool, str, Text, tuple]


class Vector(NamedTuple):
    source: str  # 'file.itl:line'
    testcase: str
    op: str
    args: Tuple[Literal, ...]
    expected: Literal  # a tuple for a two-value result
    text: str  # the statement, `collapse`d; `strip_decorations` of it is the divergence table's key
    signal: Optional[str] = None  # the name in a trailing `signal <Name>`; not checked yet


def parse_number(text: str) -> Union[Fraction, float]:
    """the nearest double to the literal, exactly; ±inf and NaN as floats (the package's infinities)"""
    if text == 'NaN':
        return math.nan
    if text.lstrip('+-') == 'infinity':
        return -math.inf if text.startswith('-') else math.inf
    value = float.fromhex(text) if 'x' in text.lower() else float(text)
    if math.isinf(value):
        raise ValueError(f'{text!r} overflows a double')
    return Fraction(value)


def parse_literal(text: str) -> Literal:
    if _INTEGER.match(text):
        return int(text)
    if _NUMBER_LITERAL.match(text):
        return parse_number(text)
    if text in ('true', 'false'):
        return text == 'true'
    if _WORD.match(text) and text not in ('infinity', 'empty', 'entire', 'nai'):
        return text
    m = _TEXT.match(text)
    if m is not None:
        return Text(m.group(1))
    m = _LIST.match(text)
    if m is not None:
        items = [t.strip() for t in m.group(1).split(',')]
        if not all(_NUMBER_LITERAL.match(t) for t in items):
            raise ValueError(f'not a list of numbers: {text!r}')
        return tuple(parse_number(t) for t in items)
    m = _INTERVAL.match(text)
    if m is None:
        raise ValueError(f'not an itl literal: {text!r}')
    decoration = m.group('decoration')
    special = m.group('special')
    if special == 'empty':
        return Interval(None, None, decoration)
    if special == 'nai':
        return Interval(None, None, decoration, nai=True)
    if special == 'entire':
        return Interval(-math.inf, math.inf, decoration)
    lo, hi = parse_number(m.group('lo')), parse_number(m.group('hi'))
    if not lo <= hi or lo == math.inf or hi == -math.inf:
        raise ValueError(f'ill-formed interval literal: {text!r}')
    return Interval(lo, hi, decoration)


def strip_decorations(text: str) -> str:
    """`atanh [1.0,1.0]_def = [empty]_trv` -> `atanh [1.0,1.0] = [empty]`; quoted strings are kept"""
    return _DECORATED.sub(lambda m: m.group(1) or m.group(), text)


def collapse(statement: str) -> str:
    """each run of white space outside a quoted string as one space, none at either end; a quoted string
    keeps its exact characters (`"[ Empty  ]"` is the text upstream gives textToInterval, not `"[ Empty ]"`)

    >>> collapse('  b-textToInterval\\n  "[  -1.0  , 1.0]_ill"   =   [nai] ')
    'b-textToInterval "[  -1.0  , 1.0]_ill" = [nai]'
    """
    return _SPACE.sub(lambda m: m.group() if m.group().startswith('"') else ' ', statement).strip()


def _tokens(text: str):
    """the statement's tokens; a character that no token covers raises, so nothing is skipped"""
    tokens, end = [], 0
    for m in _TOKEN.finditer(text):
        if text[end:m.start()].strip():
            break
        tokens.append(m.group())
        end = m.end()
    if text[end:].strip():
        raise ValueError(f'unreadable at {text[end:]!r} in {text!r}')
    return tokens


def parse_statement(text: str, source: str = '', testcase: str = '') -> Vector:
    """one statement, already `collapse`d"""
    tokens = _tokens(text)
    if tokens.count('=') != 1 or tokens[0] == '=':
        raise ValueError(f'not one " = " in {text!r}')
    at = tokens.index('=')
    op, args, result = tokens[0], tokens[1:at], tokens[at + 1:]
    signal = None
    if len(result) >= 2 and result[-2] == 'signal':
        signal = result[-1]
        result = result[:-2]
        if not _WORD.match(signal):
            raise ValueError(f'not a signal name: {signal!r} in {text!r}')
    if len(result) not in (1, 2):
        raise ValueError(f'not one or two result values in {text!r}')
    expected = tuple(parse_literal(t) for t in result)
    return Vector(source, testcase, op, tuple(parse_literal(t) for t in args),
                  expected[0] if len(expected) == 1 else expected, text, signal)


def _testcases(source: str):
    """(name, offset of the body, body) per testcase; the closing brace is found past any `{...}` list"""
    at = 0
    for m in iter(lambda: _TESTCASE.search(source, at), None):
        if source[at:m.start()].strip():
            raise ValueError(f'text outside a testcase: {source[at:m.start()].strip()[:40]!r}')
        depth, i, quoted = 1, m.end(), False
        while depth:
            if i == len(source):
                raise ValueError(f'testcase {m.group(1)} is not closed')
            if source[i] == '"':
                quoted = not quoted
            elif not quoted and source[i] in '{}':
                depth += 1 if source[i] == '{' else -1
            i += 1
        yield m.group(1), m.end(), source[m.end():i - 1]
        at = i
    if source[at:].strip():
        raise ValueError(f'text outside a testcase: {source[at:].strip()[:40]!r}')


def parse_file(path: Path, ops=None) -> Tuple[Tuple[Vector, ...], Counter]:
    """(the vectors of `ops`, of every op if `ops` is None; a count of every other op's statements)"""
    source = path.read_text(encoding='utf-8')
    # blank comments out without moving any newline, so offsets still give line numbers
    source = _COMMENT.sub(lambda m: m.group() if m.group().startswith('"') else re.sub(r'[^\n]', ' ', m.group()),
                          source)
    vectors, skipped = [], Counter()
    for name, offset, body in _testcases(source):
        for statement in body.split(';'):
            start = offset + len(statement) - len(statement.lstrip())
            offset += len(statement) + 1
            text = collapse(statement)
            if not text:
                continue
            op = text.split()[0]
            if ops is not None and op not in ops:
                skipped[op] += 1
                continue
            line = source.count('\n', 0, start) + 1
            vectors.append(parse_statement(text, f'{path.name}:{line}', name))
    return tuple(vectors), skipped
