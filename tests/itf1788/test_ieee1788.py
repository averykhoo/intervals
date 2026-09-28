"""
ieee 1788 conformance through the 1788 layer (`intervals.ieee1788`, M16b): the third pass

the adapter's two passes (`tests/itf1788/test_itf1788.py`) test the library's own semantics, and
hull both sides to compare. this pass runs every vendored vector through the layer, which answers
as 1788 does, and compares **exactly**: no hull, no rounding and no input rule here. its only rules:

* **operands**: an interval literal is `ieee1788.Interval(lo, hi, decoration)` of the literal's
  doubles as python floats (`Interval()`, or `Interval(decoration='trv')`, for `[empty]`), strict,
  so a decoration that did not fit would raise and fail the vector. a `Fraction` becomes its float
  and an `int` stays an `int` (pown's and rootn's exponents are ints; `pown(x, 2.0)` is refused), a
  text its `str`, a list a tuple of floats. **not** `ieee1788.text_to_interval(<literal>)`: the
  vectors were converted from libieeep1788's C++ tests, which read a decimal literal as its
  **nearest** double (`tests/itf1788/itl.py`), where 1788's text constructor hulls it outward; the
  constructor vectors themselves test `text_to_interval`
* **the call**: `ieee1788.NAMES[<1788 name>]`, through `CALLS`, a map from itf1788's spellings (the
  `*Bin` forms and `mulRevTen` are the op with `x` given; `pownRevBin c x n` is `pown_rev(c, n, x)`;
  `b-`/`d-` the flavours of the constructors; `sum_nearest` 1788's `sum`). `isNaI` maps to nothing
* **the comparison**: an `Interval` result is `(inf, sup)` of its set as python floats, or `None` if
  empty, plus its decoration's name or `None`, after asserting the 1788 form (empty, or one piece
  whose finite ends are closed floats and whose infinite ends are open) independently of the class;
  a pair is two of those in order; a `Decoration` or `Overlap` its value; a number a python float
  (`NaN` equals `NaN`); a boolean as it is. with the signal: `(value, signal)` on both sides
* **warnings** are recorded, and every one recorded must be a `PossiblyUndefinedOperationWarning`
  (read as 1788's `signal PossiblyUndefinedOperation`): a library warning escaping the layer fails
  its vector (recording would otherwise swallow it)
* **three readings**, in this order, each a python spelling of a 1788 answer, none a widening: a
  raised `UndefinedOperationError` is `signal UndefinedOperation` (the expected `[empty]` or
  `[nai]`); a `PossiblyUndefinedOperationWarning` is `signal PossiblyUndefinedOperation`; then a
  `ValueError` from a number or a reduction is `NaN`, read only where the vector expects `NaN`
  (`UndefinedOperationError` subclasses `ValueError`, hence the order), and raised anywhere else

every vector matches or is a row of `ROWS`, whose reason is one of three owner-approved categories,
taken from the adapter's own lists so a row cannot drift; a row that starts matching fails as stale.
the adapter's rows of the other categories (degenerate infinities, cut-based relations,
cancellation as a Minkowski difference, decoration expectations) are all 1788 conventions the layer
answers 1788's way, so none is a row here.

this module imports the adapter as `tests.itf1788.test_itf1788`, while pytest imports the adapter's
file as the top-level `test_itf1788`: two module objects, the vectors parsed twice (7.5 s, measured
by the critic, 2026-09-27). only data is imported, never a test function, and no state is shared.
"""
import math
import warnings
from fractions import Fraction

import pytest

from intervals import OutwardMultiInterval
from intervals import ieee1788
from intervals.decorated import Decoration
from intervals.errors import HullWarning
from intervals.errors import PossiblyUndefinedOperationWarning
from intervals.errors import UndefinedOperationError
from tests.itf1788 import test_itf1788 as T
from tests.itf1788.itl import Interval as Literal
from tests.itf1788.itl import Text

VECTORS = T.VECTORS
key = T.key

# itf1788's op name -> (1788's name in ieee1788.NAMES, the operands in 1788's order)
_SAME = lambda args: args  # noqa: E731
CALLS = {op: (op, _SAME) for op in {v.op for v in VECTORS}}
CALLS.update({
    'sqrRevBin': ('sqrRev', _SAME), 'absRevBin': ('absRev', _SAME), 'coshRevBin': ('coshRev', _SAME),
    'sinRevBin': ('sinRev', _SAME), 'cosRevBin': ('cosRev', _SAME), 'tanRevBin': ('tanRev', _SAME),
    'mulRevTen': ('mulRev', _SAME),
    'pownRevBin': ('pownRev', lambda args: (args[0], args[2], args[1])),  # c x n -> c n x
    'sum_nearest': ('sum', _SAME), 'sum_abs_nearest': ('sumAbs', _SAME),
    'sum_sqr_nearest': ('sumSquare', _SAME), 'dot_nearest': ('dot', _SAME),
    'b-textToInterval': ('textToInterval', _SAME), 'b-numsToInterval': ('numsToInterval', _SAME),
})
del CALLS['isNaI']  # no NaI (D16): isNaI has no counterpart, and its vectors are rows

NUMBERS = frozenset({'mid', 'rad', 'wid', 'mag', 'mig', 'midRad'})
REDUCTIONS = frozenset({'sum_nearest', 'sum_abs_nearest', 'sum_sqr_nearest', 'dot_nearest'})

# the pass's rows: the adapter's rows under these three categories, taken by reason, never copied
CATEGORIES = ('no NaI: invalid input raises', 'tighter than the vector', 'exact parsing decides validity')
_REASONS = (T._NAI, T._NO_IS_NAI, T._POWN_REV_LOOSE, T._TRIG_REV_LOOSE, T._POW_REV_LOOSE, T._EXACT_VALID,
            T._EXACT_INVALID)
ROWS = {text: reason for text, reason in T.DIVERGENCES.items() if any(reason is r for r in _REASONS)}


class NoNaI(Exception):
    """a `[nai]` operand, or isNaI: the layer has no counterpart (D16)"""


# THE PASS

def operand(literal):
    """the pass's operand rule (module docstring)"""
    if isinstance(literal, Literal):
        if literal.nai:
            raise NoNaI('[nai]')
        if literal.empty:
            return ieee1788.Interval(decoration=literal.decoration)
        return ieee1788.Interval(float(literal.lo), float(literal.hi), literal.decoration)
    if isinstance(literal, Fraction):
        return float(literal)
    if isinstance(literal, Text):
        return literal.value
    if isinstance(literal, tuple):
        return tuple(float(v) if isinstance(v, Fraction) else v for v in literal)
    return literal  # an int, a float (±inf, nan), a bool or a word (a decoration's name)


def interval_form(result):
    """`((inf, sup) or None, decoration name or None)`, after asserting the 1788 form"""
    assert isinstance(result, ieee1788.Interval), type(result)
    library = result.to_set()
    s = library.interval if result.decoration is not None else library
    assert type(s) is OutwardMultiInterval, type(s)
    decoration = None if result.decoration is None else result.decoration.value
    if s.is_empty:
        return None, decoration
    lo, hi = s.inf, s.sup
    assert s.is_contiguous and type(lo) is float and type(hi) is float, s
    assert s.inf_closed == (lo != -math.inf) and s.sup_closed == (hi != math.inf), s
    return (lo, hi), decoration


def value_form(result):
    """ours, as the comparison reads it"""
    if isinstance(result, ieee1788.Interval):
        return interval_form(result)
    if isinstance(result, tuple) and result and isinstance(result[0], ieee1788.Interval):
        assert len(result) == 2, result
        return tuple(map(interval_form, result))
    if isinstance(result, (Decoration, ieee1788.Overlap)):
        return result.value
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        assert all(type(n) is float for n in result), result
        return result
    assert type(result) is float, (type(result), result)
    return result


RAISED = 'UndefinedOperationError raised'
NAN_READ = 'a ValueError, read as NaN'


def expected_form(vector):
    """1788's value, as the comparison reads it: `(value, signal)`"""
    literal = vector.expected
    if vector.signal == 'UndefinedOperation' and isinstance(literal, Literal) and (literal.empty or literal.nai):
        return RAISED, vector.signal
    return _expected_value(literal), vector.signal


def _expected_value(literal):
    if isinstance(literal, Literal):
        if literal.nai:
            return '[nai]'
        return (None if literal.empty else (float(literal.lo), float(literal.hi))), literal.decoration
    if isinstance(literal, tuple) and literal and isinstance(literal[0], Literal):
        return tuple(map(_expected_value, literal))
    if isinstance(literal, tuple):
        return tuple(float(v) for v in literal)
    if isinstance(literal, Fraction):
        return float(literal)
    return literal


def _expects_nan(vector) -> bool:
    values = vector.expected if isinstance(vector.expected, tuple) else (vector.expected,)
    return all(isinstance(v, float) and math.isnan(v) for v in values)


def call(vector):
    """the layer's answer to `vector`, raw"""
    if vector.op not in CALLS:
        raise NoNaI(vector.op)
    name, order = CALLS[vector.op]
    return ieee1788.NAMES[name](*order([operand(a) for a in vector.args]))


def read(vector, run=call):
    """ours as `(value, signal)`, under the three readings"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = run(vector)
        except NoNaI as e:
            return f'no counterpart: {e}', None
        except UndefinedOperationError:
            ours = RAISED, 'UndefinedOperation'
        except ValueError:
            if not (vector.op in NUMBERS | REDUCTIONS and _expects_nan(vector)):
                raise
            nan = (math.nan, math.nan) if vector.op == 'midRad' else math.nan
            ours = nan, None
        else:
            ours = value_form(result), None
    escaped = [w for w in caught if not issubclass(w.category, PossiblyUndefinedOperationWarning)]
    assert not escaped, [str(w.message) for w in escaped]
    if caught:
        ours = ours[0], 'PossiblyUndefinedOperation'
    return ours


def same(ours, expected) -> bool:
    """equality, except that NaN is NaN; tuples item by item"""
    if isinstance(ours, tuple) and isinstance(expected, tuple):
        return len(ours) == len(expected) and all(same(o, e) for o, e in zip(ours, expected))
    if isinstance(ours, float) and isinstance(expected, float) and math.isnan(ours) and math.isnan(expected):
        return True
    if isinstance(ours, bool) or isinstance(expected, bool):
        return type(ours) is type(expected) and ours == expected
    return ours == expected


def check(vector):
    ours, expected = read(vector), expected_form(vector)
    if key(vector) in ROWS:
        assert not same(ours, expected), f'stale row, it matches now: {vector.text}'
    else:
        assert same(ours, expected), (vector.text, ours, expected)


@pytest.mark.parametrize('vector', VECTORS, ids=[v.source for v in VECTORS])
def test_vector(vector):
    check(vector)


# THE PASS'S OWN RULES

def test_rows():
    """94 keys at 2026-09-28 (76 no NaI, 11 tighter than the vector, 7 exact parsing), each a vector's
    key and under one of the three categories; no row on a decoration alone"""
    keys = {key(v) for v in VECTORS}
    assert set(ROWS) <= keys
    counts = {c: sum(1 for r in ROWS.values() if r.startswith(c)) for c in CATEGORIES}
    assert counts == {'no NaI: invalid input raises': 76, 'tighter than the vector': 11,
                      'exact parsing decides validity': 7}
    assert len(ROWS) == 94
    assert not set(ROWS) & set(T.PLAIN_ONLY) and not set(ROWS) & set(T.DECORATION_ONLY)
    assert sum(1 for v in VECTORS if key(v) in ROWS) == 104


def test_every_op_is_mapped():
    """every op of the vectors but isNaI reaches a function of NAMES, and the only names no vector
    reaches are the two constants' makers (no vector calls `empty` or `entire`)"""
    ops = {v.op for v in VECTORS}
    assert ops - set(CALLS) == {'isNaI'} and set(CALLS) <= ops
    reached = {name for name, _ in CALLS.values()}
    assert reached <= set(ieee1788.NAMES)
    assert set(ieee1788.NAMES) - reached == {'empty', 'entire'}


def _vector(text):
    return next(v for v in VECTORS if v.text == text)


def test_the_comparison_is_exact():
    """the pass's comparison rejects a result one double wider at either end, a bare result for a
    decorated vector and a float end that is not a float, so the pass cannot hull, round or drop a
    decoration by accident (the adapter's precedent: `test_itf1788.py::test_input_rule`)"""
    bare = _vector('add [1.0,2.0] [3.0,4.0] = [4.0,6.0]')
    decorated = _vector('add [1.0,2.0]_com [5.0,7.0]_com = [6.0,9.0]_com')
    assert same(read(bare), expected_form(bare)) and same(read(decorated), expected_form(decorated))
    down, up = math.nextafter(4.0, -math.inf), math.nextafter(6.0, math.inf)
    for wider in (ieee1788.Interval(down, 6.0), ieee1788.Interval(4.0, up)):
        assert not same(read(bare, lambda v: wider), expected_form(bare))
    assert not same(read(decorated, lambda v: ieee1788.Interval(6.0, 9.0)), expected_form(decorated))
    assert not same(read(decorated, lambda v: ieee1788.Interval(6.0, 9.0, 'dac')), expected_form(decorated))
    # an end that is an int: built past the class's own constructor, so only the pass's assertion sees it
    fake = object.__new__(ieee1788.Interval)
    object.__setattr__(fake, '_set', OutwardMultiInterval(4, 6))
    object.__setattr__(fake, '_decoration', None)
    with pytest.raises(AssertionError):
        read(bare, lambda v: fake)
    # a finite end left open
    object.__setattr__(fake, '_set', OutwardMultiInterval(4.0, 6.0, end_closed=False))
    with pytest.raises(AssertionError):
        read(bare, lambda v: fake)
    # a number that is not a python float
    number = _vector('mid [0.0,2.0] = 1.0')
    assert same(read(number), expected_form(number))
    with pytest.raises(AssertionError):
        read(number, lambda v: Fraction(1))


def test_an_escaped_warning_fails():
    """recording the warnings must not swallow a library warning the layer let through (critique n1):
    a call emitting a `HullWarning` fails its vector; a `PossiblyUndefinedOperationWarning` is read"""
    vector = _vector('add [1.0,2.0] [3.0,4.0] = [4.0,6.0]')

    def warns(category):
        def run(v):
            warnings.warn('from the library', category)
            return call(v)
        return run

    with pytest.raises(AssertionError, match='from the library'):
        read(vector, warns(HullWarning))
    assert read(vector, warns(PossiblyUndefinedOperationWarning)) == (((4.0, 6.0), None), 'PossiblyUndefinedOperation')


def test_the_nan_reading_is_narrow():
    """a `ValueError` is `NaN` only from a number or a reduction whose vector expects `NaN`; anywhere
    else it escapes (critique n3), and an `UndefinedOperationError` is read first, as the signal"""
    number = _vector('mid [empty] = NaN')
    assert same(read(number), expected_form(number))

    def plain(v):
        raise ValueError('plain')

    interval = _vector('add [1.0,2.0] [3.0,4.0] = [4.0,6.0]')
    with pytest.raises(ValueError, match='plain'):
        read(interval, plain)
    not_nan = _vector('mid [0.0,2.0] = 1.0')
    with pytest.raises(ValueError, match='plain'):
        read(not_nan, plain)

    def undefined(v):
        raise UndefinedOperationError('undefined')

    assert read(number, undefined) == (RAISED, 'UndefinedOperation')


def test_operands_keep_int_exponents():
    """critique n2: an int stays an int (pown's exponent), a Fraction becomes its float"""
    assert operand(2) == 2 and type(operand(2)) is int
    assert type(operand(Fraction(1, 2))) is float
    assert operand(Literal(Fraction(1), math.inf, 'dac')) == ieee1788.Interval(1.0, math.inf, 'dac')
    assert operand(Literal(None, None, 'trv')) == ieee1788.Interval(decoration='trv')
