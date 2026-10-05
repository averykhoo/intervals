"""
ieee 1788's interval literals and the bare constructors (M13g): `multiinterval.literals`,
`text_to_interval` and `nums_to_interval`, and the signals `UndefinedOperationError` and
`PossiblyUndefinedOperationWarning`

* `nums_to_interval` against 1788's definition: invalid (`nan`, `lo > hi`, `lo = +inf`, `hi = -inf`)
  raises, else the result is exactly the reals between the bounds, decided at probe points (so
  soundness and maximality at once): a finite bound is in it, an infinity never is
* numbers: every spelling 1788 allows for a value (a float's exact decimal expansion and `float.hex`,
  `p/q`, `inf`) reads back as that value exactly, through `[x]` and `[l, u]`
* the uncertain form against an oracle written with `decimal.Decimal`, plus soundness (the midpoint
  and the radius's ends are in the result) and maximality (a point a tenth of a unit past an end is not)
* white space and letter case do not matter inside the brackets, and do outside or inside a number;
  a decoration is read by `parse_literal` when it fits the value and always refused by the bare
  constructor; any string either reads as one interval under the input rule or raises
  `UndefinedOperationError`, never anything else
* the itf1788 vectors of `b-textToInterval` and `b-numsToInterval` run in tests/itf1788; the ones that
  matter are examples here
"""
import math
import re
import time
import warnings
from decimal import Decimal
from decimal import localcontext
from fractions import Fraction

import pytest
from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis import strategies as st

import multiinterval
from multiinterval import MultiInterval
from multiinterval import PossiblyUndefinedOperationWarning
from multiinterval import UndefinedOperationError
from multiinterval import nums_to_interval
from multiinterval import text_to_interval
from multiinterval.errors import IntervalWarning
from multiinterval.literals import DECORATIONS
from multiinterval.literals import Literal
from multiinterval.literals import number
from multiinterval.literals import parse_literal
from tests.strategies import probe_points

M = MultiInterval
INF = math.inf
MAX = 1.7976931348623157e308


def bare(lo, hi) -> MultiInterval:
    """1788's input rule, written out: finite ends closed, infinite ones open"""
    return M(lo, hi, start_closed=lo != -INF, end_closed=hi != INF)


def invalid_nums(lo, hi) -> bool:
    """1788's numsToInterval precondition, negated"""
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return True
    return lo > hi or lo == INF or hi == -INF


def in_1788(x, lo, hi) -> bool:
    """x is a point of 1788's interval [lo, hi]: a real between the bounds (never an infinity)"""
    return not (x == INF or x == -INF) and lo <= x <= hi


def spellings(v):
    """1788 number literals for the exact value `v` (int, Fraction, float, ±inf)"""
    if v == INF or v == -INF:
        return ['-inf' if v < 0 else '+infinity', '-Infinity' if v < 0 else 'INF']
    f = Fraction(v)
    out = [f'{f.numerator}/{f.denominator}']
    if isinstance(v, float):
        out += [v.hex(), format(Decimal(v), 'f'), format(Decimal(v), 'E')]  # the exact decimal expansion
    elif f.denominator == 1:
        out += [str(f.numerator), f'{f.numerator}.', f'{f.numerator}e0']
    return out


reals = st.one_of(
    st.integers(-10 ** 30, 10 ** 30),
    st.fractions(max_denominator=10 ** 6),
    st.floats(allow_nan=False, allow_infinity=False),
)
bounds = st.one_of(reals, st.sampled_from([-INF, INF, 0, 1, -1, 0.5, MAX, -MAX]), st.just(math.nan))


# nums_to_interval

@given(bounds, bounds)
@example(1.0, -1.0)  # libieeep1788_class.itl:31
@example(-INF, -INF)  # :32
@example(INF, INF)  # :33
@example(math.nan, math.nan)  # :30
@example(INF, -INF)  # ieee1788-exceptions.itl:16
@example(-INF, INF)  # ieee1788-constructors.itl:17
@example(-INF, 1.0)
@example(3, 3)
def test_nums_to_interval_is_1788s_definition(lo, hi):
    if invalid_nums(lo, hi):
        with pytest.raises(UndefinedOperationError):
            nums_to_interval(lo, hi)
        return
    result = nums_to_interval(lo, hi)
    assert type(result) is MultiInterval and result.is_contiguous and not result.is_empty
    for x in (*probe_points(result.cuts), lo, hi, 0):
        assert (x in result) == in_1788(x, lo, hi), x


@pytest.mark.parametrize('lo, hi', [(True, 1), (1, '2'), (None, 1), (Decimal(1), 2)])
def test_nums_to_interval_refuses_non_numbers(lo, hi):
    with pytest.raises(TypeError):
        nums_to_interval(lo, hi)


# numbers

@given(reals)
@example(0.1)
@example(-0.0)
@example(5e-324)
@example(MAX)
@example(Fraction(2, 3))
@example(10 ** 30)
def test_every_spelling_of_a_value_reads_back_exactly(v):
    for text in spellings(v):
        assert number(text) == v, text
        point = text_to_interval(f'[{text}]')
        assert point == M(Fraction(v)) and point.inf == Fraction(v), text


@given(st.lists(st.one_of(reals, st.sampled_from([-INF, INF])), min_size=2, max_size=2).map(sorted),
       st.integers(0, 4), st.integers(0, 4))
@example([-INF, INF], 0, 1)
@example([0.1, 0.1], 1, 2)
def test_two_bounds_read_as_nums_to_interval(pair, i, j):
    lo, hi = pair
    if lo == INF or hi == -INF:  # not a 1788 interval
        return
    lo_texts, hi_texts = spellings(lo), spellings(hi)
    lo_text, hi_text = lo_texts[i % len(lo_texts)], hi_texts[j % len(hi_texts)]
    text = f'[{lo_text}, {hi_text}]'
    assert text_to_interval(text) == nums_to_interval(lo, hi) == bare(lo, hi), text
    # an empty side is an infinity
    if lo == -INF:
        assert text_to_interval(f'[,{hi_text}]') == bare(lo, hi)
    if hi == INF:
        assert text_to_interval(f'[{lo_text},]') == bare(lo, hi)


# the uncertain form

@st.composite
def uncertain(draw):
    """(text, lo, hi) with the bounds from `decimal.Decimal`, independently of the module's Fractions"""
    sign = draw(st.sampled_from(['', '-', '+']))
    whole = draw(st.from_regex(r'[0-9]{0,4}', fullmatch=True))
    fraction = draw(st.from_regex(r'[0-9]{0,4}', fullmatch=True))
    point = draw(st.booleans()) or not whole
    if not whole and not fraction:
        whole = '0'
    radius = draw(st.one_of(st.none(), st.just('?'), st.from_regex(r'[0-9]{1,3}', fullmatch=True)))
    direction = draw(st.sampled_from(['', 'u', 'd', 'U', 'D']))
    exponent = draw(st.one_of(st.none(), st.integers(-40, 40)))
    m = whole + ('.' + fraction if point else '')
    text = (f'{sign}{m}?{radius or ""}{direction}'
            + ('' if exponent is None else draw(st.sampled_from(['e', 'E'])) + str(exponent)))
    with localcontext() as ctx:
        ctx.prec = 200
        middle = Decimal(sign + (m if not m.endswith('.') else m[:-1]) + 'e' + str(exponent or 0))
        places = len(fraction) if point else 0
        unit = Decimal(1).scaleb(-places + (exponent or 0))
        if radius == '?':
            lo, hi = -INF, INF
        else:
            r = unit / 2 if radius is None else unit * int(radius)
            lo, hi = Fraction(middle - r), Fraction(middle + r)
        if direction in 'uU' and direction:
            lo = Fraction(middle)
        if direction in 'dD' and direction:
            hi = Fraction(middle)
        return text, lo, hi, Fraction(middle), Fraction(unit)


@given(uncertain())
@example(('3.56?1', Fraction(355, 100), Fraction(357, 100), Fraction(356, 100), Fraction(1, 100)))
@example(('-10?u', Fraction(-10), Fraction(-19, 2), Fraction(-10), Fraction(1)))
@example(('0.0??u', 0, INF, 0, Fraction(1, 10)))
@example(('2.500?5de-5', Fraction(2495, 10 ** 8), Fraction(25, 10 ** 6), Fraction(25, 10 ** 6), Fraction(1, 10 ** 8)))
@example(('10?3e380', 7 * 10 ** 380, 13 * 10 ** 380, 10 ** 381, 10 ** 380))
def test_uncertain_form(case):
    text, lo, hi, middle, unit = case
    result = text_to_interval(text)
    assert result == bare(lo, hi), text
    # soundness: the midpoint and both ends of the radius are in it; maximality: nothing past them
    assert middle in result
    for end, beyond in ((lo, -Fraction(unit) / 10), (hi, Fraction(unit) / 10)):
        if end not in (INF, -INF):
            assert end in result and end + beyond not in result
    assert INF not in result and -INF not in result


# white space, case, decorations

@st.composite
def bracket_literals(draw):
    """(the parts of a valid bracket literal, lo, hi): words, one number, or two with a side left empty"""
    kind = draw(st.sampled_from(['empty', 'entire', 'blank', 'point', 'pair']))
    if kind in ('empty', 'entire'):
        return [kind], (None, None) if kind == 'empty' else (-INF, INF)
    if kind == 'blank':
        return [], (None, None)
    if kind == 'point':
        v = draw(reals)
        return [draw(st.sampled_from(spellings(v)))], (Fraction(v), Fraction(v))
    lo, hi = sorted((draw(st.one_of(reals, st.just(-INF))), draw(st.one_of(reals, st.just(INF)))))
    lo_text = '' if lo == -INF and draw(st.booleans()) else draw(st.sampled_from(spellings(lo)))
    hi_text = '' if hi == INF and draw(st.booleans()) else draw(st.sampled_from(spellings(hi)))
    return [lo_text, ',', hi_text], (lo, hi)


def build(parts, space):
    return '[' + space + space.join(parts) + space + ']'


def swapcase_some(text, flips):
    return ''.join(c.swapcase() if flip else c for c, flip in zip(text, flips + [False] * len(text)))


@given(bracket_literals(), st.sampled_from(['', ' ', '  ', '\t', ' \t ']), st.lists(st.booleans()))
def test_space_and_case_inside_the_brackets_do_not_matter(case, space, flips):
    parts, (lo, hi) = case
    plain = text_to_interval(build(parts, ''))
    assert plain == (M() if lo is None else bare(lo, hi))
    assert text_to_interval(build(parts, space)) == plain
    assert text_to_interval(swapcase_some(build(parts, space), flips)) == plain
    for outside in (' ' + build(parts, ''), build(parts, '') + ' ', build(parts, '') + '\n'):
        with pytest.raises(UndefinedOperationError):
            text_to_interval(outside)


@given(bracket_literals(), st.sampled_from([*DECORATIONS, 'ill', 'fooo', 'da', 'nai', '']), st.booleans())
@example((['empty'], (None, None)), 'trv', False)  # libieeep1788_class.itl:50: a bare refusal
@example((['entire'], (-INF, INF)), 'com', True)  # :120
@example((['-1.0', ',', ''], (-1, INF)), 'com', False)  # :125
@example((['-1.0', ',', '1.0'], (-1, 1)), 'fooo', False)  # :123
@example((['1.0E+400'], (10 ** 400, 10 ** 400)), 'com', False)  # bounded as a rational
def test_decorations(case, decoration, upper):
    parts, (lo, hi) = case
    text = build(parts, ' ') + '_' + (decoration.upper() if upper else decoration)
    fits = decoration in DECORATIONS and (decoration == 'trv' or lo is not None) and (
        decoration != 'com' or (lo is not None and lo != -INF and hi != INF))
    if fits:
        assert parse_literal(text) == Literal(*parse_literal(build(parts, ' '))[:2], decoration)
    else:
        with pytest.raises(UndefinedOperationError):
            parse_literal(text)
    with pytest.raises(UndefinedOperationError):  # the bare constructor refuses every decoration
        text_to_interval(text)


_ALPHABET = '[] ,?0123456789.+-/_eEpPxXinfINFtyudUDcomdvrlaNT\t'


@given(st.text(_ALPHABET, max_size=24))
@example('[1.0,2.0')  # libieeep1788_class.itl:227
@example('[-I  nf, 1.000 ]')  # :127
@example('[1/0]')
@example('0x1p')
@example('[]')  # 1788's empty literal: not contiguous, found by the gate's random draw (2026-09-28)
def test_any_text_is_an_interval_or_undefined(text):
    # an exponent of many digits is a big exact power (as in `Fraction('1e999999999')`): too slow here
    assume(not re.search(r'[eEpP][-+]?[0-9]{4}', text))
    try:
        result = text_to_interval(text)
    except UndefinedOperationError:
        return
    # one interval under the input rule: an end is open exactly when it is infinite
    assert type(result) is MultiInterval and (result.is_empty or result.is_contiguous)
    if not result.is_empty:
        assert (result.inf in result) == (result.inf != -INF) and (result.sup in result) == (result.sup != INF)


# EXAMPLES

@pytest.mark.parametrize('text, expected', [
    # itf1788 (libieeep1788_class.itl, ieee1788-constructors.itl): the exact values behind them
    ('[1.2345]', M(Fraction(12345, 10000))),
    ('[1,+infinity]', bare(1, INF)),
    ('[1.e-3, 1.1e-3]', M(Fraction(1, 1000), Fraction(11, 10000))),
    ('[-0x1.3p-1, 2/3]', M(Fraction(-19, 32), Fraction(2, 3))),
    ('3.56?', M(Fraction(3555, 1000), Fraction(3565, 1000))),
    ('3.560?2u', M(Fraction(356, 100), Fraction(3562, 1000))),
    ('-10?12', M(-22, 2)),
    ('[1.234e5,Inf]', bare(123400, INF)),
    ('[Empty]', M()), ('[]', M()), ('[  ]', M()), ('[ ENTIRE ]', bare(-INF, INF)), ('[,]', bare(-INF, INF)),
    ('[ -inf , INF  ]', bare(-INF, INF)), ('[-1,]', bare(-1, INF)),
    ('[1.0E+400 ]', M(10 ** 400)),
    ('[ -4/2, 10/5 ]', M(-2, 2)),
    ('0.0?', M(Fraction(-1, 20), Fraction(1, 20))), ('0.0?d', M(Fraction(-1, 20), 0)),
    ('2.5??', bare(-INF, INF)), ('2.5??d', bare(-INF, Fraction(5, 2))),
    ('10?1' + '0' * 300, M(10 - 10 ** 300, 10 + 10 ** 300)),
    # decided exactly: valid, though 1788 expects PossiblyUndefinedOperation (ieee1788-exceptions.itl:18)
    ('[1.0000000000000001, 1.0000000000000002]', M(1 + Fraction(1, 10 ** 16), 1 + Fraction(2, 10 ** 16))),
    # more forms
    ('[.5, 1.]', M(Fraction(1, 2), 1)), ('[0x.8p0]', M(Fraction(1, 2))), ('[-0/5]', M(0)),
    ('.5?1', M(Fraction(4, 10), Fraction(6, 10))), ('1.?', M(Fraction(1, 2), Fraction(3, 2))),
    ('1?', M(Fraction(1, 2), Fraction(3, 2))), ('1?0', M(1)), ('-0?', M(Fraction(-1, 2), Fraction(1, 2))),
])
def test_examples(text, expected):
    assert text_to_interval(text) == expected


@pytest.mark.parametrize('text', [
    # itf1788 (libieeep1788_class.itl, ieee1788-exceptions.itl), each with signal UndefinedOperation
    '[+infinity]', '[ Empty  ]_trv', '[,]_trv', '[ Nai  ]', '[ Nai  ]_ill', '[-Inf, 1.0  00 ]', '[-Inf ]',
    '[Inf , INF]', '[ foo ]', '[  -1.0  , 1.0]_da', '[1.0,2.0',
    # decided exactly, where 1788 expects PossiblyUndefinedOperation (libieeep1788_class.itl:136-138)
    '[1.0000000000000002,1.0000000000000001]',
    '[10000000000000001/10000000000000000,10000000000000002/10000000000000001]',
    '[0x1.00000000000002p0,0x1.00000000000001p0]',
    # more
    '', '[nai]', '[1, 2, 3]', '[1 2]', '[1/0]', '[1/-2]', '[0x1]', '[1e]', '[e1]', '[--1]', '[1_000]', '[inf, inf]',
    '[-inf, -inf]', '[-inf]', '[1,2]_', '[1,2]_ill', '[empty]_def', '[entire]_com', '0.0??_com', '?1', '1??1',
    '1?-1', '1?1ud', '0x1?', '1/2?', '1e2?', '[1?]', '(1, 2)', '1', '[1;2]', '[ 1 . 5 ]', '[1,2]_com_com', '[１]',
    ' [1, 2]', '[1, 2] ', '[1, 2]\n', '\t3.56?1',
    # M13g review: ASCII only (re.ASCII): no non-ASCII white space, and no letter that folds to an ASCII one
    '[1, 2]', '[\xa01, 2]', '[1, 2　]', '[\x1c]', '[ınf]', '[ınf, 1]', '[1,2]_Kom',
])
def test_invalid(text):
    with pytest.raises(UndefinedOperationError):
        text_to_interval(text)


# M13g review: invalid text is refused in linear time. while a run of digits or white space could be
# split several ways (`[0-9]+\.?[0-9]*`, adjacent `\s*`), the regex tried every split: 5.4 s for two
# runs of 400 digits, 0.8 s for two runs of 400 spaces, 8x per doubling; 4.8 s for the hex text below
# and 5.6 s for the uncertain one, each quadratic (measured 2026-09-26). linear, all take about 10 ms
LONG_INVALID = [
    '[' + '1' * 600 + ' , ' + '2' * 600 + '!', '[' + '1' * 5000 + 'x]', '[' + ' ' * 1000 + 'x',
    '[' + ' ' * 1000 + ',' + ' ' * 1000 + 'x', '1' * 20000 + '.' + '1' * 20000 + '?2x',
    '[0x' + '1' * 20000 + '.' + '1' * 20000 + 'x]', '[' + '1' * 1000 + '.' + '1' * 1000 + 'e' + '1' * 1000 + 'x]',
    '[' + '1' * 1000 + '/' + '1' * 1000 + ' ' * 1000 + '1]',
]


def test_invalid_text_is_refused_in_linear_time():
    start = time.perf_counter()
    for text in LONG_INVALID:
        with pytest.raises(UndefinedOperationError):
            text_to_interval(text)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize('text, lo, hi, decoration', [
    ('[empty]_trv', None, None, 'trv'), ('[ ]_TRV', None, None, 'trv'), ('[1,2]_COM', 1, 2, 'com'),
    ('[entire]_dac', -INF, INF, 'dac'), ('[,]_trv', -INF, INF, 'trv'), ('3.56?1_def', Fraction(355, 100), Fraction(357, 100), 'def'),
    # bounded as a rational, so com fits here (1788's binary64 hull is unbounded: libieeep1788_class.itl:165)
    ('[1.0E+400 ]_com', 10 ** 400, 10 ** 400, 'com'),
])
def test_parse_literal_reads_a_fitting_decoration(text, lo, hi, decoration):
    assert parse_literal(text) == Literal(lo, hi, decoration)


@pytest.mark.parametrize('text', [
    # libieeep1788_class.itl:206-225, each with signal UndefinedOperation
    '[ Nai  ]_ill', '[ Nai  ]_trv', '[ Empty  ]_ill', '[  ]_com', '[,]_com', '[   Entire ]_com', '[ -inf ,  INF ]_com',
    '[  -1.0  , 1.0]_ill', '[  -1.0  , 1.0]_fooo', '[  -1.0  , 1.0]_da', '[-1.0,]_com', '0.0??_com', '0.0??u_ill',
    # more
    '[empty]_def', '[empty]_dac', '[ ]_com', '[1, 2]_', '[1, 2]_trv_trv', '[1, 2] _com', '[1, 2]_ com',
])
def test_parse_literal_refuses_a_decoration_that_does_not_fit(text):
    with pytest.raises(UndefinedOperationError):
        parse_literal(text)


def test_non_str_is_a_type_error():
    for bad in (b'[1, 2]', 1, None, ['[1, 2]']):
        with pytest.raises(TypeError):
            text_to_interval(bad)


# the signals

def test_the_signals():
    assert issubclass(UndefinedOperationError, ValueError)
    assert issubclass(PossiblyUndefinedOperationWarning, IntervalWarning)
    # the import-time 'ignore' filters leave the new warning shown
    ignored = [f for f in warnings.filters if f[0] == 'ignore' and isinstance(f[2], type)
               and issubclass(PossiblyUndefinedOperationWarning, f[2]) and f[2] is not Warning]
    assert not ignored, ignored
    for name in ('UndefinedOperationError', 'PossiblyUndefinedOperationWarning', 'text_to_interval',
                 'nums_to_interval'):
        assert name in multiinterval.__all__
    with pytest.raises(ValueError, match='lower bound exceeds'):
        text_to_interval('[2, 1]')
    with pytest.raises(ValueError, match='numsToInterval'):
        nums_to_interval(2, 1)
