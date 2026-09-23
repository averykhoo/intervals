import math
from fractions import Fraction

import pytest
from hypothesis import given

from intervals.fmt import format_cuts
from intervals.fmt import parse
from intervals.fmt import parse_value
from intervals.kernel import EMPTY
from intervals.kernel import REALS
from intervals.kernel import normalize
from intervals.kernel import piece
from tests.strategies import cut_tuples

inf = math.inf


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
