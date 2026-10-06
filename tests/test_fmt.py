"""
the package's own text form: `multiinterval.fmt` (`format_cuts`, `parse`, `parse_value`) and the classes'
`repr`, `str` and `parse`, which go through it

* round trip: `parse(format_cuts(x))` is `x` with every value's type kept (an int stays an int, a float a
  float, a Fraction a Fraction; `==` alone reads 2, 2.0 and 2/1 as one value), and `format_cuts` is a fixed
  point, over the whole float range (subnormals, the largest double, +-inf), ints and Fractions of up to
  4300 digits and past them (hex, below), mixed types and many pieces; `repr` evaluates back to the
  same class and set, for `MultiInterval` and `OutwardMultiInterval`
* spellings: a writer of its own, from the grammar in the module docstring (not from `format_cuts`), spells a
  set every documented way (bare points, `[x, x]`, `,` or `;` inside a piece, any of the four separators
  between pieces, braces or not, pieces shuffled and repeated, empty pieces mixed in, white space anywhere
  around a token and around `/`, `inf`/`Infinity`/`∞` in any case, a float as `repr` or `%e`, an int as
  `2k/k`, in hex or with `_` between digit groups, a Fraction unreduced or with hex parts, `-0.0`); the parse
  is the set, with the design's one zero (v2-plan: `-0.0` becomes `0.0`) and an integral Fraction read as an int
* `parse_value` reads back `format_value` of every number bit for bit (`-0.0` included: the parser's tokens
  are raw, the Cut constructor normalizes) and every other spelling of the number with its type; it reads a
  text iff the tokenizer reads it as one number (m14b-open, 2026-10-05: it read `'+-5'` as 5 and `'1 2'` as 12)
* any text, random or a mutated valid output, either parses to a valid cut tuple whose own format is
  canonical, or raises ValueError, never anything else (texts of at most a few hundred characters)
* the tables: v1's parser comments and the separators v2 adds
* Q23 (owner, 2026-10-06): a number is what python's `int`, `float` or `Fraction` reads (ASCII digits only), and
  it ends at white space, punctuation or the end; two items need an explicit separator, white space alone is
  none. a pin per rule, each red on the `fmt.py` before it, and a property: `parse_value` agrees with python

findings of M14-breadth (2026-10-02): a zero denominator raised ZeroDivisionError and a separator after a
leading `[]` or `()` was refused, both fixed (the `@example`s of the any-text property); an int part past
python's 4300-digit `str()` limit could not be formatted (`repr(MultiInterval(10 ** 4300))` raised ValueError).
since 2026-10-03 (owner, m14b-open) such a part is written in hex and `parse` reads `0x`: the strategies
reach past the limit (`HUGE`), and `test_past_the_int_str_limit` pins the spelling
"""
import math
import struct
import sys
import time
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval.cuts import Cut
from multiinterval.cuts import Side
from multiinterval.fmt import _tokenize
from multiinterval.fmt import format_cuts
from multiinterval.fmt import format_value
from multiinterval.fmt import parse
from multiinterval.fmt import parse_value
from multiinterval.kernel import EMPTY
from multiinterval.kernel import REALS
from multiinterval.kernel import is_valid
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from tests.strategies import cut_tuples
from tests.strategies import pool_values

inf = math.inf
MAX = sys.float_info.max


def mi(*specs):
    return normalize(piece(s, s) if not isinstance(s, tuple) else piece(*s) for s in specs)


@given(cut_tuples())
def test_round_trip(cuts):
    text = format_cuts(cuts)
    assert parse(text) == cuts
    assert format_cuts(parse(text)) == text  # also pins the value types (2 vs 2.0 vs 2/1)


@pytest.mark.parametrize('cuts, text', [
    (EMPTY, '{}'),
    (mi((1, 2, True, False)), '[1, 2)'),
    (mi(5), '[5]'),
    (mi(0.0), '[0.0]'),
    (mi(Fraction(1, 3)), '[1/3]'),
    (REALS, '[-inf, inf]'),
    (mi((1, inf, False, False)), '(1, inf)'),
    (mi((1, 2), 3), '{ [1, 2] , [3] }'),
])
def test_format_table(cuts, text):
    assert format_cuts(cuts) == text


# the example strings from v1's parser comments, plus the separators v2 adds
@pytest.mark.parametrize('text, cuts', [
    ('[1, 2]', mi((1, 2))),
    ('[1,2]', mi((1, 2))),
    ('[0]', mi(0)),
    ('{0}', mi(0)),
    ('{}', EMPTY),
    ('[]', EMPTY),
    ('()', EMPTY),
    ('(123)', EMPTY),
    ('{ [1, 2) | [3, 4) }', mi((1, 2, True, False), (3, 4, True, False))),
    ('{[1,2),[3,4)}', mi((1, 2, True, False), (3, 4, True, False))),
    # v1's `[1,2)[3,4)`, pieces with nothing between them, is refused since Q23: test_white_space_alone_separates_nothing
    ('[1, 2) ∪ (2, 3]', mi((1, 2, True, False), (2, 3, False, True))),
    ('{1, 2, 3}', mi(1, 2, 3)),
    ('{1; 2}', mi(1, 2)),
    ('[1; 2]', mi((1, 2))),
    ('5', mi(5)),
    ('[-inf, -5]', mi((-inf, -5))),  # `- 5` until Q23: the sign is attached now
    ('[-∞, ∞)', mi((-inf, inf, True, False))),
    ('[1e-05, 1/3)', mi((1e-05, Fraction(1, 3), True, False))),
    ('[1, 2) | [2, 3]', mi((1, 3))),
])
def test_parse_table(text, cuts):
    assert parse(text) == cuts


@pytest.mark.parametrize('text', [
    '[2, 1]',  # reversed
    '[1)',  # half-open degenerate
    '[1, 2, 3]',
    '{[1, 2]',
    '[1, 2]}',
    '{[1, 2]} [3]',
    '[1, 2',
    'abc',
    '[1, nan]',
    '| [1, 2]',
    '[1.5/2]',
])
def test_parse_errors(text):
    with pytest.raises(ValueError):
        parse(text)


@pytest.mark.parametrize('text, value', [
    ('3', 3), ('-3', -3), ('3.0', 3.0), ('1e-05', 1e-05), ('1E+16', 1e16), ('.5', 0.5),
    ('1/3', Fraction(1, 3)), ('-1/3', Fraction(-1, 3)), ('-inf', -inf), ('Infinity', inf),
    ('-5', -5), (' +5\n', 5), ('1 / 0x3', Fraction(1, 3)), ('-0x1f', -31), ('1.', 1.0),
    ('1_000', 1000), ('-1_0.5', -10.5), ('1_000/3', Fraction(1000, 3)),  # Q23: python's `_`
])
def test_parse_value(text, value):
    parsed = parse_value(text)
    assert parsed == value and type(parsed) is type(value)


# MALFORMED NUMBERS (m14b-open, 2026-10-05): `parse_value` deleted all white space and every leading sign, then
# handed the rest to python's int/float/Fraction, so it read text no token of the grammar is: `'+-5'` was 5, `'1 2'`
# 12, `'1/-3'` -1/3. the grammar allows white space around `/`, nowhere else inside a number. since Q23
# (2026-10-06) not after the sign either, and python's `_` is read (`'1_000'` moved to test_digit_separators)

@pytest.mark.parametrize('text', [
    '+-5', '-+5', '--5', '++5', '+ -5', '- -5',  # one sign at most
    '1 2', '1 . 5', '1. 5', '1 .5', '1e 5', '1 e5', '1e+ 5', '1 2/3', '1/2 3',  # white space inside the digits
    'i n f', 'in f', '-in finity', '0 x1f', '0x 1f', '0x1 f', '0x1f / 0x 3',
    '1/-3', '1/+3', '-1/-3',  # no sign after `/`
    '- 5', '+ 5', '- inf', '- 0x1f',  # the sign is attached (Q23): test_a_sign_is_attached
])
def test_parse_value_refuses_malformed_numbers(text):
    with pytest.raises(ValueError, match='not a number'):
        parse_value(text)


# EXTREME VALUES

# python refuses `str()` of an int past 4300 digits (`sys.get_int_max_str_digits()`): BIG is the largest int it
# writes in decimal, HUGE the smallest it does not, which `format_value` writes in hex (m14b-open, 2026-10-03;
# finding F2 of M14-breadth fmt, 2026-10-02: `repr(MultiInterval(10 ** 4300))` raised ValueError)
BIG = 10 ** 4300 - 1
HUGE = 10 ** 4300
# hypothesis writes a strategy's repr (its arguments by `repr`), which python refuses for an int past the limit,
# so the huge values are made by `.map`, never passed to a strategy as literals
huge_ints = st.tuples(st.integers(-BIG, BIG), st.integers(0, BIG)).map(lambda t: t[0] * HUGE + t[1])

extreme_floats = st.one_of(
    st.floats(allow_nan=False),  # the whole range: subnormals, the largest double, +-inf, -0.0
    st.sampled_from([5e-324, -5e-324, 2.2250738585072014e-308, MAX, -MAX, 0.1, 1e-05, 1e16, 2.0, -0.0]),
)
extreme_ints = st.one_of(
    st.integers(),
    st.integers(-BIG, BIG),
    huge_ints,  # mostly past the limit: hex
    st.sampled_from([2 ** 53 + 1, -2 ** 63, BIG, -BIG, 1, -1]).map(lambda k: k if abs(k) != 1 else k * HUGE),
)
extreme_fractions = st.one_of(
    st.fractions(),
    st.builds(Fraction, st.integers(-BIG, BIG), st.integers(1, BIG)),
    st.builds(Fraction, huge_ints, huge_ints.map(abs).filter(bool)),
    st.sampled_from(range(5)).map(lambda i: (Fraction(1, BIG), Fraction(-BIG, BIG - 1), Fraction(1, HUGE),
                                             Fraction(-HUGE, 3), Fraction(HUGE + 1, HUGE))[i]),
)
extreme_values = st.one_of(extreme_floats, extreme_ints, extreme_fractions)
mixed_values = st.one_of(pool_values, extreme_values)  # the pool keeps equal values of two types common


@st.composite
def many_pieces(draw, values=extreme_values):
    """up to 30 pieces: distinct values paired off in order, each end open or closed"""
    points = sorted(set(draw(st.lists(values, min_size=2, max_size=60))))
    return normalize(piece(lo, hi, draw(st.booleans()), draw(st.booleans()))
                     for lo, hi in zip(points[::2], points[1::2]))


def typed(cuts):
    """each cut as (type, value, side): `==` on cuts alone takes 2, 2.0 and 2/1 for one value"""
    return [(type(cut.value), cut.value, cut.side) for cut in cuts]


def assert_values_normal(cuts):
    """the Cut constructor's promises (v2-plan "representation: cuts"): int, Fraction or float, no nan, one
    zero (never -0.0), no integral Fraction"""
    for cut in cuts:
        value = cut.value
        assert type(value) in (int, Fraction, float), cut
        if type(value) is float:
            assert not math.isnan(value), cut
            assert value != 0 or math.copysign(1.0, value) > 0, cut
        if type(value) is Fraction:
            assert value.denominator != 1, cut


@settings(max_examples=150, deadline=None)
@given(st.one_of(cut_tuples(mixed_values, max_pieces=8), many_pieces()))
@example(EMPTY)
@example(REALS)
@example(mi(5e-324, -5e-324, MAX, -MAX))
@example(mi((-inf, inf, False, False)))
@example(mi(-inf, inf))
@example(mi(BIG, -BIG, Fraction(1, BIG)))
@example(mi(HUGE, -HUGE, Fraction(1, HUGE), Fraction(-HUGE, 3)))
@example(mi((0.1, Fraction(1, 3), False, True), 7, (2 ** 53 + 1, 1e300, True, False)))
def test_round_trip_extreme(cuts):
    text = format_cuts(cuts)
    assert typed(parse(text)) == typed(cuts)
    assert format_cuts(parse(text)) == text


NAMESPACE = {'MultiInterval': MultiInterval, 'OutwardMultiInterval': OutwardMultiInterval}


@pytest.mark.parametrize('cls', [MultiInterval, OutwardMultiInterval])
@settings(max_examples=60, deadline=None)
@given(cuts=st.one_of(cut_tuples(mixed_values), many_pieces()))
@example(cuts=EMPTY)
@example(cuts=REALS)
@example(cuts=mi(0.1, (Fraction(1, 3), BIG, False, True)))
@example(cuts=mi((-HUGE, Fraction(1, HUGE), True, False), HUGE))
def test_repr_round_trip(cls, cuts):
    # v2-plan: `repr` evaluates back; `str` is the text `parse` reads
    x = cls.from_cuts(cuts)
    back = eval(repr(x), NAMESPACE)
    assert type(back) is cls and typed(back.cuts) == typed(cuts)
    assert str(x) == format_cuts(cuts)
    again = cls.parse(str(x))
    assert type(again) is cls and typed(again.cuts) == typed(cuts)


# SPELLINGS: a writer of the grammar of its own, so the parser is not checked against format_cuts alone

_SPACE = st.sampled_from(['', '', ' ', '  ', '\t', '\n', ' \r\n '])
# no `_SPACED` since Q23 (2026-10-06): white space alone separated two items, now it separates nothing


@st.composite
def spelled_number(draw, value):
    """`value` spelled some way the grammar reads as it: a sign attached to the body (Q23: no white space
    between them), the body with `_` between digit groups sometimes"""
    negative = value < 0
    magnitude = abs(value)
    if isinstance(value, float) and math.isinf(value):
        body = draw(st.sampled_from(['inf', 'Inf', 'INF', 'infinity', 'Infinity', 'INFINITY', '∞']))
    elif isinstance(value, float):
        body = draw(st.sampled_from([repr(magnitude), f'{magnitude:.17e}', f'{magnitude:.17E}', f'{magnitude:.25e}']))
        negative = negative or (value == 0 and draw(st.booleans()))  # `-0.0`
    elif isinstance(value, int):
        spellings = [_hex(magnitude, draw)]  # any int may be hex; past python's limit it must be
        if magnitude < 10 ** 4000:  # decimal and `2k/k`, kept under python's 4300-digit limit
            k = draw(st.integers(1, 9))
            spellings += [str(magnitude), f'{magnitude:_}', f'{magnitude * k}{draw(_SPACE)}/{draw(_SPACE)}{k}']
        body = draw(st.sampled_from(spellings))
        negative = negative or (value == 0 and draw(st.booleans()))  # `-0`
    else:
        small = max(magnitude.numerator, magnitude.denominator) < 10 ** 4000
        k = draw(st.integers(1, 9)) if small else 1  # unreduced
        numerator, denominator = magnitude.numerator * k, magnitude.denominator * k
        body = f'{_int_text(numerator, draw)}{draw(_SPACE)}/{draw(_SPACE)}{_int_text(denominator, draw)}'
    sign = '-' if negative else draw(st.sampled_from(['', '+']))
    return sign + body


def _hex(n: int, draw) -> str:
    """`0x...` with the prefix and the digits in either case (the grammar reads hex case-blind), with `_`
    between groups of four digits or after `0x` sometimes, as `int(s, 0)` reads them"""
    text = hex(n)
    grouped = f'0x{n:_x}'
    return draw(st.sampled_from([text, text.upper(), '0x' + text[2:].upper(), grouped, '0X_' + grouped[2:]]))


def _int_text(n: int, draw) -> str:
    """one part of a fraction: decimal where python writes it, else hex; hex either way, sometimes, and
    decimal with `_` between groups of three digits sometimes"""
    if n >= 10 ** 4000 or draw(st.booleans()):
        return _hex(n, draw)
    return draw(st.sampled_from([str(n), f'{n:_}']))


@st.composite
def spelled_piece(draw, start, end):
    """one piece from its two cuts, read off the design's table: a start BELOW is `[`, ABOVE `(`; an end
    ABOVE is `]`, BELOW `)`"""
    def space():
        return draw(_SPACE)

    lo, hi = start.value, end.value
    if lo == hi:  # a point, the one degenerate piece a normalized tuple holds: `x`, `[x]` or `[x, x]`
        bare = draw(spelled_number(lo))
        return draw(st.sampled_from([
            bare, f'[{space()}{bare}{space()}]', f'[{bare}{space()},{space()}{draw(spelled_number(hi))}]']))
    opening = '[' if start.side == Side.BELOW else '('
    closing = ']' if end.side == Side.ABOVE else ')'
    comma = draw(st.sampled_from([',', ';']))
    lo_text, hi_text = draw(spelled_number(lo)), draw(spelled_number(hi))
    return f'{opening}{space()}{lo_text}{space()}{comma}{space()}{hi_text}{space()}{closing}'


@st.composite
def spelled_empty(draw):
    """an empty piece (v2-plan: empty iff start >= end): `[]`, `()`, `(x)`, `(x, x)`, `[x, x)`, `(x, x]`"""
    def space():
        return draw(_SPACE)

    value = draw(st.sampled_from([0, -2, Fraction(1, 3), 2.5, 0.0, inf, -inf]))
    a, b = draw(spelled_number(value)), draw(spelled_number(value))
    return draw(st.sampled_from([
        f'[{space()}]', f'({space()})', f'({space()}{a}{space()})',
        f'({a},{space()}{b})', f'[{a},{space()}{b})', f'({a},{space()}{b}]']))


@st.composite
def spelled_set(draw, cuts):
    """the set every documented way: its pieces, some repeated, empty pieces mixed in, shuffled, between
    them `,` `;` `|` or `∪` (white space alone or nothing was a separator until Q23), in braces or not"""
    def space():
        return draw(_SPACE)

    pairs = list(zip(cuts[::2], cuts[1::2]))
    items = [draw(spelled_piece(start, end)) for start, end in pairs]
    if pairs:
        items += [draw(spelled_piece(start, end)) for start, end in draw(st.lists(st.sampled_from(pairs), max_size=2))]
    items += draw(st.lists(spelled_empty(), max_size=2))
    items = draw(st.permutations(items))
    text = items[0] if items else ''
    for at, item in enumerate(items[1:], 1):
        separator = draw(st.sampled_from([',', ';', '|', '∪']))
        text += f'{space()}{separator}{space()}{item}'
    if not items or draw(st.booleans()):
        text = f'{{{space()}{text}{space()}}}'
    return f'{space()}{text}{space()}'


@settings(max_examples=300, deadline=None)
@given(st.data(), st.one_of(cut_tuples(mixed_values, max_pieces=6), many_pieces()))
def test_spellings(data, cuts):
    text = data.draw(spelled_set(cuts), label='text')
    parsed = parse(text)
    assert typed(parsed) == typed(cuts)
    assert_values_normal(parsed)  # `-0.0` read as the one zero, `4/2` as the int 2


@settings(max_examples=200, deadline=None)
@given(st.data(), extreme_values)
@example(None, -0.0)
@example(None, 5e-324)
@example(None, -BIG)
@example(None, Fraction(-1, BIG))
@example(None, -HUGE)
@example(None, Fraction(-1, HUGE))
@example(None, Fraction(HUGE, 7))
def test_parse_value_round_trip(data, value):
    if type(value) is Fraction and value.denominator == 1:
        value = int(value)  # as the Cut constructor stores it; format_value prints `4/1` as `4`
    back = parse_value(format_value(value))
    assert back == value and type(back) is type(value)
    if type(value) is float:
        assert struct.pack('<d', back) == struct.pack('<d', value)  # bit for bit: -0.0 too
    if data is None:
        return
    spelled = data.draw(spelled_number(value), label='spelled')
    again = parse_value(spelled)
    assert again == value and type(again) is (Fraction if '/' in spelled else type(value))  # raw: no Cut yet


# PAST PYTHON'S INT-STR LIMIT (m14b-open, owner 2026-10-03): hex out, hex in, decimal in still limited

def test_past_the_int_str_limit():
    assert format_value(BIG) == str(BIG)  # the largest int python writes in decimal stays decimal
    assert format_value(HUGE) == hex(HUGE) and format_value(-HUGE) == hex(-HUGE)
    assert format_value(Fraction(1, HUGE)) == f'1/{hex(HUGE)}'
    assert format_value(Fraction(-HUGE, 3)) == f'{hex(-HUGE)}/3'
    for x in (MultiInterval(HUGE), OutwardMultiInterval(-HUGE, Fraction(1, HUGE)), MultiInterval(3) ** 2 ** 21):
        text = repr(x)  # raised ValueError, and so did str() and format()
        assert eval(text, NAMESPACE) == x and type(x).parse(str(x)) == x and format(x) == str(x)
    assert repr(Cut(HUGE, Side.BELOW)) == f'Cut({hex(HUGE)}, BELOW)'
    assert repr(Cut(Fraction(-1, HUGE), Side.ABOVE)) == f'Cut(Fraction(-1, {hex(HUGE)}), ABOVE)'
    with pytest.raises(ValueError, match='hex'):  # python's own limit, kept: it guards the whole process
        parse('1' * 4301)
    with pytest.raises(ValueError, match='hex'):
        parse_value('1/' + '3' * 4301)


# LINEAR TIME (m14b-open, 2026-10-04): `parse(' ' * 30000 + 'x')` took 37 s. the tokenizer's leading `\s*`
# backtracked a white-space run no token follows one space at a time, and at each position `_NUMBER`'s own
# `[+-]?\s*` re-read the rest of the run: quadratic. every `\s*` in `_TOKEN` is possessive now (`\s*+`).
# the bound is loose on purpose (a shared laptop under load): linear parsing takes milliseconds here, the
# quadratic tokenizer about 170 s at this length (4.5 s at 16000, times 4 per doubling)

_LONG = 100_000


@pytest.mark.parametrize('text, outcome', [
    (' ' * _LONG + 'x', ValueError),  # the reported case: white space, then no token
    (' ' * _LONG + '+x', ValueError),  # a sign makes `_NUMBER` read one more character before failing
    ('[1' + ' ' * _LONG + 'x', ValueError),  # inside a piece
    ('1' + ' ' * _LONG, mi(1)),  # trailing
    (' ' * _LONG + '1', mi(1)),  # leading, then a token
    ('[1' + ' ' * _LONG + ', 2]', mi((1, 2))),  # between tokens
    ('[-' + ' ' * _LONG + '5]', ValueError),  # between a sign and its digits: refused since Q23
    ('[1' + ' ' * _LONG + '/' + ' ' * _LONG + '3]', mi(Fraction(1, 3))),  # around `/`
    # Q23 (2026-10-06): digit and `_` runs, each possessive like the white-space runs
    ('[0.' + '1_' * _LONG + '1]', mi(float('0.' + '1' * (_LONG + 1)))),  # a long `_` run, read
    ('[' + '1_' * _LONG + '_1]', ValueError),  # a long `_` run, then a doubled `_`
    ('[' + '1' * _LONG + '-1]', ValueError),  # a long digit run, then a number side by side
    ('[0.' + '1' * _LONG + '.2]', ValueError),
    ('[0x' + '1_' * _LONG + 'g]', ValueError),  # hex
    ('[1' + ' ' * _LONG + '2]', ValueError),  # white space alone between two numbers
    ('[1]' + ' ' * _LONG + '[2]', ValueError),  # and between two items
    ('[' + '1' * _LONG + ' ' * _LONG + '/x]', ValueError),  # a numerator, white space, `/`, then no denominator
    # Q24 (2026-10-06): many separators, each between two items; then one after the last, or two in a row at the end
    ('{' + '[1, 2) , ' * _LONG + '[3]}', mi((1, 2, True, False), 3)),
    ('{' + '1 | ' * _LONG + '}', ValueError),
    ('{' + '1 ; ' * _LONG + ', 2}', ValueError),
], ids=['then-junk', 'then-sign-junk', 'in-piece-then-junk', 'trailing', 'leading', 'between', 'after-sign', 'around-slash',
        'underscores', 'underscores-doubled', 'digits-then-number', 'digits-then-dot', 'hex-underscores', 'space-in-piece',
        'space-between-items', 'digits-slash-nothing', 'many-separators', 'many-separators-trailing',
        'many-separators-doubled'])
def test_white_space_runs_parse_in_linear_time(text, outcome):
    started = time.perf_counter()
    if isinstance(outcome, type):
        with pytest.raises(outcome):
            parse(text)
    else:
        assert parse(text) == outcome
    assert time.perf_counter() - started < 10, 'a white-space, digit or `_` run is read more than once'


@pytest.mark.parametrize('text, value', [
    ('0x1f', 31), ('-0X1F', -31), ('0x1f/0x3', Fraction(31, 3)), ('1/0x10', Fraction(1, 16)),
    ('0x10 / 3', Fraction(16, 3)), ('0x0', 0),
])
def test_parse_hex(text, value):
    parsed = parse_value(text)
    assert parsed == value and type(parsed) is type(value)


# `[0x12.5]` was `[1, 2.5]` and `[1/0x35.5]` `[1/3, 5.5]`: the hex digits gave one back to dodge `(?!\.)` (2026-10-05)
@pytest.mark.parametrize('text', ['0x', '[0x1.8p1]', '[0x1.5e3]', '[1/0x3.5]', '0x1g', '[0x 1]',
                                  '[0x12.5]', '[1/0x35.5]', '{0X1F2.5}', '[-0x10.5]'])
def test_hex_floats_and_bad_hex_are_refused(text):
    """a hex float is not read, and never as `0x1` then `.8` (until Q23 two numbers side by side were a piece)"""
    with pytest.raises(ValueError):
        parse(text)


# ANY TEXT

_GRAMMAR = '0123456789 .eE+-/[](){},;|∪∞infINFtyxXabcdef\t\n_٣'  # `_` and a non-ASCII digit since Q23
@st.composite
def mutated_outputs(draw):
    """a valid output with a few characters inserted, deleted or replaced, or a slice repeated"""
    text = format_cuts(draw(cut_tuples()))
    for _ in range(draw(st.integers(1, 4))):
        i = draw(st.integers(0, len(text)))
        j = draw(st.integers(i, min(len(text), i + 6)))
        char = draw(st.sampled_from(_GRAMMAR))
        text = draw(st.sampled_from([
            text[:i] + char + text[i:], text[:i] + text[j:], text[:i] + char + text[j:],
            text[:j] + text[i:j] + text[j:]]))
    return text


@settings(max_examples=600, deadline=None)
@given(st.one_of(st.text(max_size=40), st.text(_GRAMMAR, max_size=40), mutated_outputs()))
@example('')
@example('   ')
@example('{[1, 2]} [3]')
@example('[1, nan]')
@example('[1e400, 1/3]')
@example('٣/٤')  # unicode digits are digits to python's int(); refused since Q23
@example('[ınf]')  # a dotless i matched `i` under re.IGNORECASE, which the grammar no longer uses (Q23)
@example('[1/0]')  # raised ZeroDivisionError (M14-breadth)
@example('0/0')
@example('[] , [1]')  # an empty item then a separator was refused (M14-breadth)
@example('{() ∪ (), [2]}')
@example('[0E0-0]')  # a point whose cuts differ in type: format wrote [0.0], losing the int (fuzz x10, 2026-10-02);
# `0E0` and `-0` side by side, a ValueError since Q23 (2026-10-06): `[0.0, -0]` keeps the case
@example('[1.0, 1]')
@example('[0.0, -0]')
def test_any_text_parses_or_raises_value_error(text):
    try:
        cuts = parse(text)
    except ValueError:
        return
    assert is_valid(cuts)
    assert_values_normal(cuts)
    canonical = format_cuts(cuts)
    assert typed(parse(canonical)) == typed(cuts)
    assert format_cuts(parse(canonical)) == canonical


# PARSE_VALUE IS ONE TOKEN (m14b-open, 2026-10-05)

def _one_number(text):
    """the token, when the tokenizer reads `text` as one number and nothing else; else None"""
    try:
        tokens = _tokenize(text)
    except ValueError:
        return None
    return tokens[0][1] if len(tokens) == 1 and tokens[0][0] == 'num' else None


@st.composite
def mangled_numbers(draw):
    """a spelled number with a sign, white space, `.`, `_`, `e`, `x`, `/`, a digit or a non-ASCII digit put in or
    taken out"""
    text = draw(spelled_number(draw(st.one_of(pool_values, st.integers(), st.fractions(), st.floats(allow_nan=False)))))
    for _ in range(draw(st.integers(1, 2))):
        i = draw(st.integers(0, len(text)))
        char = draw(st.sampled_from(' +-._exX/09٣'))
        text = draw(st.sampled_from([text[:i] + char + text[i:], text[:i] + text[i + 1:]]))
    return text


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.text(_GRAMMAR, max_size=20), mangled_numbers()))
@example('+-5')
@example('1 2')
@example('1/-3')
@example('1_000')  # refused until Q23, read since
@example('0x 1f')
@example('- 5')  # read until Q23
@example('1-2')
def test_parse_value_reads_one_number_or_raises(text):
    """`parse_value` reads a text iff the tokenizer reads it as one number, and then as `parse` does"""
    token = _one_number(text)
    try:
        value = parse_value(text)
    except ValueError:
        if token is not None:  # a number token parse_value refuses (`1.5/2`, `1` * 4301): parse refuses it too
            with pytest.raises(ValueError):
                parse(text)
        return
    assert token is not None, f'{text!r} is not one number of the grammar, read as {value!r}'
    assert typed(parse(text)) == typed(mi(value))


# Q23 (owner, 2026-10-06): "each number should be something python can parse, split by a character that's not a
# valid part of the number", and, as white space can sit inside a fraction (`1 / 3`), it separates nothing. each
# test below is red on the `fmt.py` before Q23 (d2e2e8c); a row that only pins what stayed is marked so

# a number side by side with another was a second number; now a ValueError naming the end. the message is matched,
# so a row the old code refused for another reason (`[1e5-1]`, a reversed piece) still says something
@pytest.mark.parametrize('text', [
    '[0.1.2]', '[-2-1]', '{1-2}', '[1+2]', '[0E0-0]', '1-2', '1+2', '-1-2',  # a sign
    '[1inf]', '[1∞]', '[∞∞]', '[-∞-1]', '[inf-inf]', '[infinity1]', '[inf0x1]',  # infinity
    '[-0x1e-5]', '[0x1e+5]', '[1/0x3-1]',  # hex: its `e` is a digit, not an exponent
    '[0..5]', '[1..5]', '[0.5.5]', '[1/3.5]', '[1e5.5]',  # `.`
    '[1e5-1]', '[1/3-1]', '[1/3+1/3]', '[.5-.5]',
    '[1,2-3]', '{1, 2-3}',
])
def test_numbers_side_by_side_are_refused(text):
    with pytest.raises(ValueError, match='a number must end at white space, punctuation or the end'):
        parse(text)


# python's int, float and Fraction read the digits of every script (`int('١٢')` is 12): the grammar reads ASCII only
@pytest.mark.parametrize('text', [
    '[١٢]', '[1٢]', '[١/٢]', '[1, ١]', '[1e٣]', '[٣]', '[１]', '[१]', '{1, ٣}', '[0.٥]',
])
def test_non_ascii_digits_are_refused(text):
    with pytest.raises(ValueError):
        parse(text)
    number = text.strip('[]{}').split(', ')[-1]
    assert _python_reads(number) is not None  # python reads it: the refusal is the grammar's own
    with pytest.raises(ValueError, match='not a number'):
        parse_value(number)


@pytest.mark.parametrize('text', ['- 5', '+ 5', '- inf', '-\tinf', '- ∞', '- 0x1f', '-\n1/3', '+ 1_000', '- 1.5'])
def test_a_sign_is_attached(text):
    """white space after the sign was allowed (`- 5` was -5); python refuses it (`float('- 5')`)"""
    with pytest.raises(ValueError, match='not a number'):
        parse_value(text)
    with pytest.raises(ValueError):
        parse(f'[{text}]')
    assert parse_value(''.join(text.split())) is not None  # attached, it reads


@pytest.mark.parametrize('text, value', [
    ('1 / 3', Fraction(1, 3)), ('1/ 3', Fraction(1, 3)), ('1 /3', Fraction(1, 3)), (' -1\t/\n3 ', Fraction(-1, 3)),
    ('0x10 / 3', Fraction(16, 3)), ('1_000 / 3', Fraction(1000, 3)),  # stays, as Fraction(' 1 / 3 ') reads it
    ('1/-3', ValueError), ('1 / -3', ValueError), ('1/+3', ValueError),  # stays refused, as Fraction refuses it
    ('[1 / 3 2]', ValueError), ('{1 / 3 2}', ValueError), ('{1 /3 2/ 3}', ValueError),  # new: the space is the number's
])
def test_white_space_around_the_slash_is_the_numbers(text, value):
    """white space around `/` stays inside the number; so a number after a fraction needs a separator (red before
    Q23 on the last three rows only: the rest pin what stayed)"""
    if value is ValueError:
        with pytest.raises(ValueError):
            parse(text)
        return
    assert parse_value(text) == value and type(parse_value(text)) is Fraction
    assert parse(f'[{text}]') == mi(value) and parse(f'{{{text}, {text}}}') == mi(value)


@pytest.mark.parametrize('text, value', [
    # read since Q23, as python reads them (red before)
    ('1_000', 1000), ('1_0.5', 10.5), ('1e1_0', 1e10), ('1_000/3', Fraction(1000, 3)), ('1/1_000', Fraction(1, 1000)),
    ('1.5_5', 1.55), ('.5_5', 0.55), ('1_0e1_0', 1e11), ('-1_000', -1000), ('0_0', 0), ('0_1', 1),
    ('0x1_f', 31), ('0x_1f', 31), ('0X_1_F', 31), ('-0x_1f/0x_3', Fraction(-31, 3)),
    # refused, as python refuses them (and as before)
    ('_1', None), ('1_', None), ('1__0', None), ('1_.5', None), ('1._5', None), ('1e_5', None), ('1_e5', None),
    ('1e5_', None), ('_1/3', None), ('1_/3', None), ('1/_3', None), ('1/3_', None), ('1._', None), ('._5', None),
    ('0x1__f', None), ('0x1f_', None), ('0x_', None), ('0_x1f', None), ('0x__1', None),
])
def test_digit_separators(text, value):
    """python's `_`, by python's rules: one between two digits, and after `0x` (`int(s, 0)`)"""
    if value is None:
        with pytest.raises(ValueError, match='not a number'):
            parse_value(text)
        with pytest.raises(ValueError):
            parse(f'[{text}]')
        return
    parsed = parse_value(text)
    assert parsed == value and type(parsed) is type(value)
    assert typed(parse(f'[{text}]')) == typed(mi(value))


@pytest.mark.parametrize('text', [
    '[1 2]', '{1 2}', '1 2', '{[1, 2] [3, 4]}', '[1, 2] [3, 4]', '{1 / 3 2}', '[1\t2)', '{1\n2}', '[1] [2]',
    '[1,2)[3,4)', '[1, 2) [3, 4)', '5[1, 2]', '[1]5', '{ [1] [2] , [3] }', '{(1, 2)(3, 4)}', '[]()', '1 2, 3',
])
def test_white_space_alone_separates_nothing(text):
    """owner, 2026-10-06 (Q23): two items need `,` `;` `|` or `∪` between them, two numbers of a piece `,` or `;`;
    white space alone (or nothing) was a separator"""
    with pytest.raises(ValueError, match='between them'):
        parse(text)


@pytest.mark.parametrize('text, cuts', [
    ('[1; 2]', mi((1, 2))), ('[1 ,2]', mi((1, 2))), ('{1 ; 2}', mi(1, 2)), ('{ [1] }', mi(1)), ('[1, 2]', mi((1, 2))),
    ('5', mi(5)), (' 1 / 3 ', mi(Fraction(1, 3))), ('{1/3 , 2}', mi(Fraction(1, 3), 2)), ('[1] | [2]', mi(1, 2)),
    ('[1]∪[2]', mi(1, 2)), ('[1],[2]', mi(1, 2)), ('{[1, 2];[3, 4]}', mi((1, 2), (3, 4))), ('( 1 , 2 )', mi((1, 2, False, False))),
])
def test_explicit_separators_stay(text, cuts):
    """what stays (green before Q23 too): the explicit separators, and white space as padding around them"""
    assert parse(text) == cuts


# Q24 (owner, 2026-10-06): a separator stands between two items, or two numbers of a piece, and nowhere else. a
# trailing one (`[1,]` and `{1,}` were `[1]`) and a doubled one (`{1,,2}` was `{ [1] , [2] }`) are refused, as a
# leading one was. every separator, in a piece (`,` `;`) and between items (`,` `;` `|` `∪`), padded and not
_ITEM_SEPARATORS = [',', ';', '|', '∪']
_NUMBER_SEPARATORS = [',', ';']
_PAD = ['', ' ', ' \t\n ']


def _trailing():
    rows = []
    for pad in _PAD:
        for s in _ITEM_SEPARATORS:
            rows += [f'1{pad}{s}{pad}', f'[1, 2]{pad}{s}', f'{{1{pad}{s}{pad}}}', f'{{{pad}[1, 2){pad}{s}{pad}}}',
                     f'{{[]{pad}{s}{pad}}}', f'[]{pad}{s}', f'{{1, 2{pad}{s}{pad}}}', f'{{(){pad}{s}}}']
        for s in _NUMBER_SEPARATORS:
            rows += [f'[1{pad}{s}{pad}]', f'(1{pad}{s}{pad})', f'[1, 2{pad}{s}{pad}]', f'{{[1{pad}{s}{pad}], 2}}',
                     f'[1{pad}{s}{pad}] | [3]']
    return rows


def _doubled():
    rows = []
    for pad in _PAD:
        for a in _ITEM_SEPARATORS:
            for b in _ITEM_SEPARATORS:
                rows += [f'1{pad}{a}{pad}{b}{pad}2', f'{{1{pad}{a}{pad}{b}{pad}2}}', f'{{[]{pad}{a}{b}{pad}[]}}',
                         f'[1, 2){pad}{a}{pad}{b}{pad}[3]']
        for a in _NUMBER_SEPARATORS:
            for b in _NUMBER_SEPARATORS:
                rows += [f'[1{pad}{a}{pad}{b}{pad}2]', f'{{(1{a}{pad}{b}2), 3}}']
    return rows


@pytest.mark.parametrize('text', _trailing())
def test_a_trailing_separator_is_refused(text):
    """red before Q24 on every row: each was read as if the separator were not there"""
    with pytest.raises(ValueError, match='after the last (item|number)'):
        parse(text)


@pytest.mark.parametrize('text', _doubled())
def test_a_doubled_separator_is_refused(text):
    """red before Q24 on every row: between items each was read as one separator; in a piece (`[1,,2]`) it was
    refused before too, as a missing `]`, and the message is matched"""
    with pytest.raises(ValueError, match='two separators in a row'):
        parse(text)


@pytest.mark.parametrize('text', [
    f'{s}1' for s in _ITEM_SEPARATORS] + [f'{{ {s} 1}}' for s in _ITEM_SEPARATORS] + [f'{{{s}}}' for s in _ITEM_SEPARATORS] + [
    '{ , }', '{,}', ',', ', ,', '{, [1]}', '| [1, 2]',
])
def test_a_leading_separator_stays_refused(text):
    """green before Q24 too: a leading separator between items was refused already"""
    with pytest.raises(ValueError, match='before the first item'):
        parse(text)


@pytest.mark.parametrize('text', [f'[{s}1]' for s in _NUMBER_SEPARATORS] + [
    '[,]', '[;]', '(,)', '( , 2)', '[ ; 1, 2]', '{[,1]}', '{1, (;2)}',
])
def test_a_leading_separator_in_a_piece_is_refused(text):
    """refused before Q24 too, as a missing `]`; the message is new"""
    with pytest.raises(ValueError, match='before the first number'):
        parse(text)


@pytest.mark.parametrize('text, cuts', [
    ('', EMPTY), ('{}', EMPTY), ('{ }', EMPTY), ('[]', EMPTY), ('[ ]', EMPTY), ('()', EMPTY), ('( )', EMPTY),
    ('(1)', EMPTY), ('(1, 1)', EMPTY), ('[1, 1)', EMPTY), ('{[],[]}', EMPTY), ('{ [] , () }', EMPTY),
    ('{[], [1]}', mi(1)), ('[] , [1]', mi(1)), ('{() ∪ (), [2]}', mi(2)), ('{(1), [2]}', mi(2)), ('{[1, 1), 2}', mi(2)),
    ('5', mi(5)), ('[1]', mi(1)), ('{1}', mi(1)), ('{ [1] }', mi(1)), ('[1, 2)', mi((1, 2, True, False))),
    ('{ (1; 2] }', mi((1, 2, False, True))), (' -inf ', mi(-inf)), ('{1 / 3}', mi(Fraction(1, 3))),
])
def test_empty_forms_and_one_item_still_read(text, cuts):
    """what stays (green before Q24 too): the empty forms, empty pieces among a set's items, and a single item. items
    with nothing between them stay refused: test_white_space_alone_separates_nothing (`[1,2)[3,4)`, `5[1, 2]`)"""
    assert parse(text) == cuts


def _python_reads(text):
    """what python reads `text` as, by its shape: a Fraction if it has `/`, a float if `.` or an exponent, else an
    int; None if that reader refuses it. a zero denominator (python: ZeroDivisionError) is None, as the grammar
    refuses it with a ValueError (M14-breadth)"""
    reader = Fraction if '/' in text else float if any(c in text for c in '.eE') else int
    try:
        return reader(text)
    except (ValueError, ZeroDivisionError):
        return None


def _python_reads_any(text):
    """whether any of python's int, float and Fraction reads `text` (a zero denominator aside)"""
    for reader in (int, float, Fraction):
        try:
            reader(text)
        except ValueError:
            continue
        except ZeroDivisionError:
            return False
        return True
    return False


_PYTHON_ALPHABET = '0123456789_.eE+- /\t\n'
# near-numbers: digit groups with `_`, a `.`, an exponent, a `/`, white space and signs in places
_NEAR_NUMBERS = st.from_regex(
    r'[ \t]?[+-]?[ ]?[0-9_]{0,5}\.?[0-9_]{0,4}(?:[eE][+-]?[0-9_]{0,3})?(?:[ ]?/[ ]?[+-]?[0-9_]{0,4})?[ \n]?', fullmatch=True)


@settings(max_examples=500, deadline=None)
@given(st.one_of(st.text(_PYTHON_ALPHABET, max_size=14), _NEAR_NUMBERS))
@example('1_000')  # python reads these (red before Q23)
@example('1_0.5')
@example('1e1_0')
@example('1_000 / 3')
@example('- 5')  # python refuses these (read before Q23)
@example('+ 5')
@example('_1')
@example('1__0')
@example('1_.5')
@example('1._5')
@example('1e_5')
@example('1/0')
@example('-0.0')
@example('007')
@example('1.e5')
def test_parse_value_agrees_with_python(text):
    """over digits, `_`, `.`, `e`, signs, `/` and white space, `parse_value` reads a text exactly when python's int,
    float or Fraction does, and as the one its shape names does, type and bits (`-0.0`) included. the one
    deliberate difference, non-ASCII digits, is test_non_ascii_digits_are_refused"""
    expected = _python_reads(text)
    assert (expected is not None) == _python_reads_any(text), f'{text!r}: the shape names a reader that refuses it'
    try:
        value = parse_value(text)
    except ValueError:
        assert expected is None, f'python reads {text!r} as {expected!r}, the grammar refuses it'
        return
    assert expected is not None, f'python refuses {text!r}, the grammar reads it as {value!r}'
    assert type(value) is type(expected) and value == expected
    if type(value) is float:
        assert struct.pack('<d', value) == struct.pack('<d', expected)


def _python_reads_hex(text):
    try:
        return int(text, 0)
    except ValueError:
        return None


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.text('0123456789abcdefABCDEF_xX+- ', max_size=10),
                 st.from_regex(r'[ ]?[+-]?0?[xX][_0-9a-fA-F]{0,6}[ ]?', fullmatch=True)).filter(lambda t: 'x' in t.lower()))
@example('0x_1f')  # int(s, 0) reads these (red before Q23)
@example('0x1_f')
@example('- 0x1f')  # int(s, 0) refuses these
@example('0x1__f')
@example('0x_')
@example('00x1')
@example('0x1e-5')
def test_parse_value_agrees_with_python_on_hex(text):
    """a text with `x` over hex digits, `_`, signs and spaces: `parse_value` reads it exactly when `int(s, 0)`
    does, as the same int"""
    expected = _python_reads_hex(text)
    try:
        value = parse_value(text)
    except ValueError:
        assert expected is None, f'int(s, 0) reads {text!r} as {expected!r}, the grammar refuses it'
        return
    assert expected is not None, f'int(s, 0) refuses {text!r}, the grammar reads it as {value!r}'
    assert type(value) is int and value == expected
