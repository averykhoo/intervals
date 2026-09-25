"""
a parser for the subset of itf1788's `itl` vector language that the vendored files use

a file is `testcase <name> { <statement>; ... }` blocks with `/* */` and `//` comments. a statement
is `<op> <arg> ... = <expected>`, every value a literal: an interval (`[a, b]`, `[empty]`,
`[entire]`, `[nai]`, optionally decorated `_trv`/`_def`/`_dac`/`_com`/`_ill`), an integer (pown's
exponent), a number (`isMember`'s point, `inf`'s result: a `Fraction`, or ±inf), `true`/`false`, or an
overlap state (`before`, `containedBy`, ...: a str). only statements whose op is in `ops` are parsed;
the rest are counted by op name. a parsed statement with anything else in it raises, so a vector is
never dropped silently.

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
_NUMBER_LITERAL = re.compile(rf'{_NUMBER}$')
_WORD = re.compile(r'[A-Za-z]+$')
_LITERAL = re.compile(r'\[[^\]]*\](?:_[a-z]+)?|[^\s\[\]]+')
_TESTCASE = re.compile(r'testcase\s+([\w.]+)\s*\{(.*?)\}', re.DOTALL)  # atan2.itl has 'minimal.atan2_test'
_COMMENT = re.compile(r'/\*.*?\*/|//[^\n]*', re.DOTALL)


class Interval(NamedTuple):
    """a bare or decorated 1788 interval; `lo`/`hi` are exact, `None` for empty and NaI"""
    lo: Optional[Union[Fraction, float]]
    hi: Optional[Union[Fraction, float]]
    decoration: Optional[str] = None
    nai: bool = False

    @property
    def empty(self) -> bool:
        return self.lo is None and not self.nai


Literal = Union[Interval, int, Fraction, float, bool, str]


class Vector(NamedTuple):
    source: str  # 'file.itl:line'
    testcase: str
    op: str
    args: Tuple[Literal, ...]
    expected: Literal
    text: str  # the statement, whitespace collapsed; the divergence table's key


def parse_number(text: str) -> Union[Fraction, float]:
    """the nearest double to the literal, exactly; ±inf as floats (the package's infinities)"""
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


def parse_file(path: Path, ops) -> Tuple[Tuple[Vector, ...], Counter]:
    """(the vectors of `ops`, a count of every other op's statements)"""
    source = path.read_text(encoding='utf-8')
    # blank comments out without moving any newline, so offsets still give line numbers
    source = _COMMENT.sub(lambda m: re.sub(r'[^\n]', ' ', m.group()), source)
    vectors, skipped = [], Counter()
    for case in _TESTCASE.finditer(source):
        offset = case.start(2)
        for statement in case.group(2).split(';'):
            start = offset + len(statement) - len(statement.lstrip())
            offset += len(statement) + 1
            text = ' '.join(statement.split())
            if not text:
                continue
            op = text.split()[0]
            if op not in ops:
                skipped[op] += 1
                continue
            lhs, sep, rhs = text.partition(' = ')
            if not sep:
                raise ValueError(f'{path.name}: no " = " in {text!r}')
            args = tuple(parse_literal(t) for t in _LITERAL.findall(lhs[len(op):]))
            expected = parse_literal(rhs)
            line = source.count('\n', 0, start) + 1
            vectors.append(Vector(f'{path.name}:{line}', case.group(1), op, args, expected, text))
    return tuple(vectors), skipped
