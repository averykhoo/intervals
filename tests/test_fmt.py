"""
the package's own text form: `intervals.fmt` (`format_cuts`, `parse`, `parse_value`) and the classes'
`repr`, `str` and `parse`, which go through it

* round trip: `parse(format_cuts(x))` is `x` with every value's type kept (an int stays an int, a float a
  float, a Fraction a Fraction; `==` alone reads 2, 2.0 and 2/1 as one value), and `format_cuts` is a fixed
  point, over the whole float range (subnormals, the largest double, +-inf), ints and Fractions of up to
  4300 digits, mixed types and many pieces; `repr` evaluates back to the same class and set, for
  `MultiInterval` and `OutwardMultiInterval`
* spellings: a writer of its own, from the grammar in the module docstring (not from `format_cuts`), spells a
  set every documented way (bare points, `[x, x]`, `,` or `;` inside a piece, any separator or none between
  pieces, braces or not, pieces shuffled and repeated, empty pieces mixed in, white space anywhere around a
  token and after a sign or around `/`, `inf`/`Infinity`/`∞` in any case, a float as `repr` or `%e`, an int
  as `2k/k`, a Fraction unreduced, `-0.0`); the parse is the set, with the design's one zero (v2-plan:
  `-0.0` becomes `0.0`) and an integral Fraction read as an int
* `parse_value` reads back `format_value` of every number bit for bit (`-0.0` included: the parser's tokens
  are raw, the Cut constructor normalizes) and every other spelling of the number with its type
* any text, random or a mutated valid output, either parses to a valid cut tuple whose own format is
  canonical, or raises ValueError, never anything else (texts of at most a few hundred characters)
* the tables: v1's parser comments and the separators v2 adds

findings of M14-breadth (2026-10-02): a zero denominator raised ZeroDivisionError and a separator after a
leading `[]` or `()` was refused, both fixed (the `@example`s of the any-text property); an int part past
python's 4300-digit `str()` limit cannot be formatted, held out of the round trip (`BIG`), an open question
"""
import math
import struct
import sys
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals.cuts import Side
from intervals.fmt import format_cuts
from intervals.fmt import format_value
from intervals.fmt import parse
from intervals.fmt import parse_value
from intervals.kernel import EMPTY
from intervals.kernel import REALS
from intervals.kernel import is_valid
from intervals.kernel import normalize
from intervals.kernel import piece
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
    ('[1,2)[3,4)', mi((1, 2, True, False), (3, 4, True, False))),
    ('[1, 2) ∪ (2, 3]', mi((1, 2, True, False), (2, 3, False, True))),
    ('{1, 2, 3}', mi(1, 2, 3)),
    ('{1; 2}', mi(1, 2)),
    ('[1; 2]', mi((1, 2))),
    ('5', mi(5)),
    ('[-inf, - 5]', mi((-inf, -5))),
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
    ('1/3', Fraction(1, 3)), ('-1/3', Fraction(-1, 3)), ('- inf', -inf), ('Infinity', inf),
])
def test_parse_value(text, value):
    parsed = parse_value(text)
    assert parsed == value and type(parsed) is type(value)


# EXTREME VALUES

# python refuses `str()` of an int past 4300 digits (`sys.get_int_max_str_digits()`), so `format_cuts` cannot
# print one (finding F2 of M14-breadth fmt, 2026-10-02: `repr(MultiInterval(10 ** 4300))` raises ValueError).
# the strategies stop at 4300 digits
BIG = 10 ** 4300 - 1

extreme_floats = st.one_of(
    st.floats(allow_nan=False),  # the whole range: subnormals, the largest double, +-inf, -0.0
    st.sampled_from([5e-324, -5e-324, 2.2250738585072014e-308, MAX, -MAX, 0.1, 1e-05, 1e16, 2.0, -0.0]),
)
extreme_ints = st.one_of(
    st.integers(),
    st.integers(-BIG, BIG),
    st.sampled_from([2 ** 53 + 1, -2 ** 63, BIG, -BIG]),
)
extreme_fractions = st.one_of(
    st.fractions(),
    st.builds(Fraction, st.integers(-BIG, BIG), st.integers(1, BIG)),
    st.sampled_from([Fraction(1, BIG), Fraction(-BIG, BIG - 1)]),
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
_SPACED = st.sampled_from([' ', '  ', '\t', '\n'])  # never none: two bare numbers need one


@st.composite
def spelled_number(draw, value):
    """`value` spelled some way the grammar reads as it: a sign, white space, then the body"""
    negative = value < 0
    magnitude = abs(value)
    if isinstance(value, float) and math.isinf(value):
        body = draw(st.sampled_from(['inf', 'Inf', 'INF', 'infinity', 'Infinity', 'INFINITY', '∞']))
    elif isinstance(value, float):
        body = draw(st.sampled_from([repr(magnitude), f'{magnitude:.17e}', f'{magnitude:.17E}', f'{magnitude:.25e}']))
        negative = negative or (value == 0 and draw(st.booleans()))  # `-0.0`
    elif isinstance(value, int):
        spellings = [str(magnitude)]
        if magnitude < 10 ** 4000:  # `2k/k`, kept under python's 4300-digit limit
            k = draw(st.integers(1, 9))
            spellings.append(f'{magnitude * k}{draw(_SPACE)}/{draw(_SPACE)}{k}')
        body = draw(st.sampled_from(spellings))
        negative = negative or (value == 0 and draw(st.booleans()))  # `-0`
    else:
        small = max(magnitude.numerator, magnitude.denominator) < 10 ** 4000
        k = draw(st.integers(1, 9)) if small else 1  # unreduced
        body = f'{magnitude.numerator * k}{draw(_SPACE)}/{draw(_SPACE)}{magnitude.denominator * k}'
    sign = '-' if negative else draw(st.sampled_from(['', '+']))
    return sign + (draw(_SPACE) if sign else '') + body


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
    them `,` `;` `|` `∪` or nothing, in braces or not"""
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
        separator = draw(st.sampled_from([',', ';', '|', '∪', '']))
        text += (f'{space()}{separator}{space()}' if separator else draw(_SPACED)) + item
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


# ANY TEXT

_GRAMMAR = '0123456789 .eE+-/[](){},;|∪∞infINFty\t\n'
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
@example('٣/٤')  # unicode digits are digits to python's int()
@example('[ınf]')  # a dotless i matches `i` under re.IGNORECASE
@example('[1/0]')  # raised ZeroDivisionError (M14-breadth)
@example('0/0')
@example('[] , [1]')  # an empty item then a separator was refused (M14-breadth)
@example('{() ∪ (), [2]}')
@example('[0E0-0]')  # a point whose cuts differ in type: format wrote [0.0], losing the int (fuzz x10, 2026-10-02)
@example('[1.0, 1]')
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
