"""
numpy interop (M16d, H3's second part): `intervals/numpy_compat.py`, the foreign-real rule of
`cuts.normalize_value` and the integer arguments that take any `Integral`

numpy is in the `[test]` extra (Q15(g), 2026-10-03), and a test-time import only, through the `np`
fixture, so without it the numpy-free tests (numpy never imported at load, the foreign-real stubs)
still run and the rest skip. every property names its oracle, and none is the hook itself:

* a numpy scalar in an operator: the same operator on the python number of the same value, which
  never touches numpy (`python_number`)
* a ufunc on one of ours: the method that computes the same set (`np.arcsin(A)` is `A.asin()`), or
  python's operator (`np.add(M, O)` is `M + O`)
* an ndarray meeting one of ours: the scalar path on each python element of `f.tolist()`
* a foreign real: its exact value, from a stub that holds a `Fraction`
"""
import math
import numbers
import operator
import os
import re
import subprocess
import sys
import tomllib
import warnings
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import DecoratedInterval
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import numpy_compat
from intervals.autodiff import Dual
from intervals.cuts import normalize_value
from intervals.decorated import set_dec
from intervals.reverse import pown_rev
from tests.strategies import cut_tuples

M = MultiInterval
O = OutwardMultiInterval
D = DecoratedInterval
INF = math.inf
HERE = os.path.normcase(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS = (MultiInterval, DecoratedInterval, Dual)


@pytest.fixture(scope='module')
def np():
    return pytest.importorskip('numpy')


# HELPERS

def _form(x):
    """what a result is, for comparing two: its type and repr (so an int end is not a float end),
    recursively through a tuple (divmod) and a Dual (which has no ==)"""
    if isinstance(x, tuple):
        return tuple(_form(y) for y in x)
    if isinstance(x, Dual):
        return 'Dual', _form(x.value), _form(x.derivative)
    if type(x).__name__ == 'ndarray':
        return 'ndarray', x.dtype, x.shape, tuple(_form(e) for e in x.flat)
    return type(x), repr(x)


def outcome(thunk):
    """(the result's form or the exception type, [(warning category, file)])"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = 'value', _form(thunk())
        except Exception as e:  # noqa: BLE001 - the exception type is the outcome
            result = 'raises', type(e)
    return result, [(w.category, os.path.normcase(os.path.abspath(w.filename))) for w in caught]


def assert_same_outcome(got, expected):
    """equal results (or exception types) and warning categories; ours attributed to this file"""
    assert got[0] == expected[0]
    assert [c for c, _ in got[1]] == [c for c, _ in expected[1]]
    assert all(f == HERE for _, f in got[1]), got[1]


@st.composite
def ours(draw, kinds=('M', 'O', 'D', 'DO', 'dual', 'dual O', 'dual D')):
    """one of ours of a drawn kind, over drawn cuts (up to 3 pieces, any ends)"""
    cuts = draw(cut_tuples(max_pieces=3))
    kind = draw(st.sampled_from(kinds))
    interval = (O if 'O' in kind else M).from_cuts(cuts)
    if 'D' in kind:
        interval = set_dec(interval, draw(st.sampled_from(['com', 'dac', 'def', 'trv'])))
    return Dual.variable(interval) if kind.startswith('dual') else interval


def _is_ours(x) -> bool:
    return isinstance(x, OURS)


def python_number(np, s):
    """the python number of the same value as the numpy scalar s: never a numpy type"""
    if isinstance(s, np.bool_):
        return bool(s)
    if isinstance(s, np.integer):
        return int(s)
    if isinstance(s, np.complexfloating):
        return complex(s)
    f = float(s)
    try:
        ratio = s.as_integer_ratio()
    except (ValueError, OverflowError):  # a nan or an infinity; a finite s past the doubles has one
        return f
    # decided on exact values, as the library does (`cuts.py::normalize_value`), not on how a numpy
    # version compares a long double with a float
    exact = Fraction(*(int(k) for k in ratio))
    return f if math.isfinite(f) and Fraction(f) == exact else exact


def numpy_scalars(np, name):
    floats = st.one_of(st.sampled_from([0.0, -0.0, 0.5, -2.0, 3.0, INF, -INF, math.nan]), st.floats(-20, 20))
    wide = st.tuples(st.floats(-20, 20), st.integers(-3, 3)).map(
        lambda t: np.longdouble(t[0]) + np.longdouble(t[1]) * np.longdouble(2.0) ** -60)
    ints = st.integers(-20, 20)
    return {
        'float64': floats.map(np.float64),
        'float32': st.one_of(floats, st.floats(-20, 20, width=32)).map(np.float32),
        'float16': st.one_of(floats, st.floats(-20, 20, width=16)).map(np.float16),
        'longdouble': st.one_of(floats.map(np.longdouble), wide),
        'int8': st.one_of(ints, st.sampled_from([-128, 127])).map(np.int8),
        'int64': st.one_of(ints, st.sampled_from([-2 ** 63, 2 ** 63 - 1])).map(np.int64),
        'uint64': st.one_of(st.integers(0, 20), st.just(2 ** 64 - 1)).map(np.uint64),
        'bool_': st.booleans().map(np.bool_),
        'complex128': st.complex_numbers(max_magnitude=20, allow_nan=False, allow_infinity=False).map(np.complex128),
    }[name]


# the operators a numpy scalar can meet ours in, DERIVED from the classes' reflected dunders, so a
# new one joins this test the day it is written, and is red until numpy_compat's table routes it
_NOT_OPERATORS = {'__repr__', '__reduce__', '__reduce_ex__', '__round__'}
REFLECTED = sorted({n for cls in OURS for n in dir(cls)
                    if re.fullmatch(r'__r[a-z]+__', n) and n not in _NOT_OPERATORS})


def _operator(dunder: str):
    name = dunder[3:-2]
    if name == 'divmod':
        return divmod
    fn = getattr(operator, name, None) or getattr(operator, name + '_', None)
    assert fn is not None, f'no python operator for {dunder}'
    return fn


OPERATORS = {d[3:-2]: _operator(d) for d in REFLECTED}
OPERATORS.update({'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt, 'le': operator.le,
                  'gt': operator.gt, 'ge': operator.ge, 'in list': lambda x, y: x in [y]})
SCALAR_TYPES = ['float64', 'float32', 'float16', 'longdouble', 'int8', 'int64', 'uint64', 'bool_', 'complex128']


def test_the_operators_are_derived():
    """the derivation found the eleven reflected dunders there are today (a new one adds a row)"""
    assert {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'and', 'or', 'xor'} <= set(OPERATORS)


# 1. NUMPY IS OPTIONAL

def test_numpy_is_never_imported_at_load():
    code = ('import sys, pkgutil, importlib, intervals\n'
            'for m in pkgutil.iter_modules(intervals.__path__):\n'
            '    importlib.import_module("intervals." + m.name)\n'
            'import intervals.numpy_compat\n'
            'bad = sorted(k for k in sys.modules if k.split(".")[0] == "numpy")\n'
            'assert not bad, bad\n')
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert not [r for r in _project().get('dependencies', []) if re.match(r'numpy\b', r)]


def _project() -> dict:
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)['project']


def test_the_gate_needs_numpy():
    """numpy is in the `[test]` extra (owner, 2026-10-03, Q15(g)), as gmpy2 is for the backend's
    differential: the gate's numpy tests do not skip on a `pip install -e .[test]` machine, and
    README's numpy section is doctests, which fail loudly without it. the library never needs it:
    `dependencies` has no numpy (the test above)"""
    assert [r for r in _project()['optional-dependencies']['test'] if re.match(r'numpy\b', r)]


# 2. A NUMPY SCALAR IS A PYTHON NUMBER TO EVERY OPERATOR

@pytest.mark.parametrize('op', sorted(OPERATORS))
@pytest.mark.parametrize('kind', SCALAR_TYPES)
@settings(max_examples=12, deadline=None)
@given(data=st.data())
def test_numpy_scalar_operators_are_python_numbers(np, kind, op, data):
    s = data.draw(numpy_scalars(np, kind))
    x = data.draw(ours())
    fn = OPERATORS[op]
    if op == 'pow' and isinstance(s, np.integer) and abs(int(s)) > 64:
        return  # A ** 2**63 is an exact power of 2**63 digits: too slow, with any number type
    p = python_number(np, s)
    assert_same_outcome(outcome(lambda: fn(s, x)), outcome(lambda: fn(p, x)))
    assert_same_outcome(outcome(lambda: fn(x, s)), outcome(lambda: fn(x, p)))


class _WideLongDouble:
    """a long double wider than a double whose `==` reads it as its double (a comparison the oracle
    must not rely on): windows has no such long double, so this stands in for linux's"""

    def __init__(self, ratio, rounded):
        self.ratio, self.rounded = ratio, rounded

    def __float__(self):
        return self.rounded

    def __eq__(self, other):
        return self.rounded == other

    def as_integer_ratio(self):
        return self.ratio


def test_python_number_decides_exactly(np):
    """the oracle of test 2 keeps a value that is not a double exact, whatever `==` says"""
    assert python_number(np, _WideLongDouble((2 ** 60 + 1, 2 ** 60), 1.0)) == Fraction(2 ** 60 + 1, 2 ** 60)
    assert python_number(np, _WideLongDouble((2 ** 2000, 1), INF)) == 2 ** 2000  # finite past the doubles
    assert python_number(np, np.float64(0.1)) == 0.1 and type(python_number(np, np.float64(0.1))) is float
    assert python_number(np, np.longdouble(-INF)) == -INF


def test_numpy_scalar_operator_examples(np):
    """by example, what test 2 checks: python's reflected dunder runs, with the scalar's value"""
    a = M(1, 2)
    assert np.float64(2) + a == M(3.0, 4.0)
    assert repr(np.int64(3) * a) == repr(M(3, 6))
    assert np.float32(0.1) + a == M(1 + float(np.float32(0.1)), 2 + float(np.float32(0.1)))
    assert np.uint64(2 ** 64 - 1) + a == M(2 ** 64, 2 ** 64 + 1)
    assert divmod(np.float64(7), M(2)) == divmod(7.0, M(2))
    assert np.int64(1) | a == 1 | a
    assert (np.float64(1.5) < a) == (1.5 < a)
    assert np.float64(2) ** a == 2.0 ** a
    for thunk in (lambda: np.bool_(True) + a, lambda: np.complex128(1) + a, lambda: np.float64(1) @ a):
        with pytest.raises(TypeError):
            thunk()


def test_a_missing_dunder_is_numpys_refusal(np):
    """a class without an operator (`Dual` has no `|`, no `~`) is numpy's TypeError, every hook having
    declined: the lookup is python's, in the class's MRO, never `type.__or__` on the metaclass (PEP
    604 puts `|` on every class, so `hasattr(Dual, '__or__')` is True). a gap the sabotage found
    (2026-09-28): with the metaclass lookup the TypeError came from calling `type.__or__`"""
    x = Dual.variable(M(1))
    for thunk in (lambda: np.int64(1) | x, lambda: np.bitwise_or(x, 1), lambda: np.bitwise_or(1, x),
                  lambda: np.invert(x)):
        with pytest.raises(TypeError, match='returned NotImplemented from __array_ufunc__'):
            thunk()


def test_numpy_exponent_of_a_dual(np):
    """B1 through numpy: `x ** np.float32(0.1)` is `x ** 0.10000000149011612`, the double it holds;
    M15 computed `np.float32(0.1) - 1` in float32"""
    x = Dual.variable(O(2))
    assert _form(x ** np.float32(0.1)) == _form(x ** float(np.float32(0.1)))
    assert _form(x ** np.float16(0.1)) == _form(x ** float(np.float16(0.1)))


# 3. A UFUNC ON OURS IS THE METHOD OF THE SAME SET, OR PYTHON'S OPERATOR

UNARY_METHODS = {
    'sqrt': 'sqrt', 'cbrt': 'cbrt', 'exp': 'exp', 'exp2': 'exp2', 'expm1': 'expm1', 'log': 'log',
    'log2': 'log2', 'log10': 'log10', 'log1p': 'log1p', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'floor': 'floor', 'ceil': 'ceil', 'trunc': 'trunc',
    'sign': 'sign', 'reciprocal': 'reciprocal', 'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh', 'rint': 'round',
}
UNARY_OPERATORS = {'negative': operator.neg, 'positive': operator.pos, 'absolute': abs, 'fabs': abs,
                   'invert': operator.invert, 'square': lambda x: x ** 2}
# fmin/fmax: minimum/maximum (owner, 2026-10-03, Q15(f)), a nan the library's ValueError
BINARY_METHODS = {'minimum': 'minimum', 'maximum': 'maximum', 'hypot': 'hypot', 'fmin': 'minimum', 'fmax': 'maximum'}
# the binary operator ufuncs and python's operator for each
BINARY_OPERATORS = {
    'add': operator.add, 'subtract': operator.sub, 'multiply': operator.mul, 'divide': operator.truediv,
    'floor_divide': operator.floordiv, 'remainder': operator.mod, 'divmod': divmod, 'power': operator.pow,
    'bitwise_and': operator.and_, 'bitwise_or': operator.or_, 'bitwise_xor': operator.xor,
    'less': operator.lt, 'less_equal': operator.le, 'greater': operator.gt, 'greater_equal': operator.ge,
    'equal': operator.eq, 'not_equal': operator.ne,
}


def _method(x, name):
    """the bound method, or a thunk raising TypeError where the class has none (numpy's refusal)"""
    method = getattr(x, name, None)
    if method is None:
        def refused(*args):
            raise TypeError(f'{type(x).__name__} has no {name}')
        return refused
    return method


def _unary_oracle(name):
    if name in UNARY_METHODS:
        return lambda x: _method(x, UNARY_METHODS[name])()

    def python(x):
        try:
            return UNARY_OPERATORS[name](x)
        except AttributeError:
            raise TypeError from None
    return python


def _binary_oracle(name):
    """the scalar meaning of a binary ufunc, from python numbers and our methods only"""
    if name in BINARY_OPERATORS:
        fn = BINARY_OPERATORS[name]
        return fn
    if name in BINARY_METHODS:
        def symmetric(a, b):  # the first operand of ours, its method: both ours, the method's own rule
            x, y = (a, b) if _is_ours(a) else (b, a)
            return _method(x, BINARY_METHODS[name])(y)
        return symmetric
    assert name == 'arctan2'

    def arctan2(y, x):
        if not _is_ours(y):
            if not hasattr(x, 'atan2'):
                raise TypeError
            y = x._coerce(y)
        return _method(y, 'atan2')(x)
    return arctan2


@pytest.mark.parametrize('name', sorted(UNARY_METHODS) + sorted(UNARY_OPERATORS))
@settings(max_examples=25, deadline=None)
@given(x=ours())
def test_unary_ufunc_is_the_method(np, name, x):
    assert_same_outcome(outcome(lambda: getattr(np, name)(x)), outcome(lambda: _unary_oracle(name)(x)))


numbers_ = st.one_of(st.integers(-5, 5), st.fractions(-5, 5, max_denominator=4), st.floats(-5, 5), st.just(INF))


@pytest.mark.parametrize('name', sorted(BINARY_METHODS) + ['arctan2'] + sorted(BINARY_OPERATORS))
@settings(max_examples=25, deadline=None)
@given(x=ours(), y=st.one_of(numbers_, ours()))
def test_binary_ufunc_is_the_method_or_the_operator(np, name, x, y):
    """both orders, the other operand a python number or one of ours: with both ours, python's own
    rule, subclass first, so `np.add(M, O)` is `M + O`, an OutwardMultiInterval (B3)"""
    ufunc, oracle = getattr(np, name), _binary_oracle(name)
    assert_same_outcome(outcome(lambda: ufunc(x, y)), outcome(lambda: oracle(x, y)))
    assert_same_outcome(outcome(lambda: ufunc(y, x)), outcome(lambda: oracle(y, x)))


@pytest.mark.parametrize('name', sorted(BINARY_OPERATORS))
def test_both_ours_is_pythons_rule(np, name):
    """B3, by example: the subclass's reflected dunder first, as python runs `M + O`"""
    a, b = M(0.1, 1), O(0.2, 3)
    for x, y in ((a, b), (b, a)):
        assert_same_outcome(outcome(lambda: getattr(np, name)(x, y)), outcome(lambda: BINARY_OPERATORS[name](x, y)))
    # a DecoratedInterval refuses a bare set: `a + D(a)` is a TypeError, `a == D(a)` False
    assert_same_outcome(outcome(lambda: getattr(np, name)(a, D(a))), outcome(lambda: BINARY_OPERATORS[name](a, D(a))))
    if name not in ('equal', 'not_equal'):
        with pytest.raises(TypeError):
            getattr(np, name)(a, D(a))


def test_both_ours_examples(np):
    s = np.add(M(0.1), O(0.2))
    assert type(s) is O and s == M(0.1) + O(0.2) and Fraction(0.1) + Fraction(0.2) in s
    assert s != M(0.1).__add__(O(0.2))  # the forward dunder alone: rounded to nearest, a plain set
    assert type(np.multiply(O(0.1), M(3))) is O
    assert np.less(M(1, 2), O(3)) == (M(1, 2) < O(3))


def test_both_ours_method_ufunc_is_the_method(np):
    """the method ufuncs take the operators' rule (the M16d review, 2026-09-28), and since the owner's
    Q15(h) answer (2026-10-03) the methods do too: with a MultiInterval and an OutwardMultiInterval
    the outward class decides in either order, so the result holds the true value, and the ufunc is
    the method called directly (it was M's, rounded to nearest: the point [0.1414213562373095])"""
    a, b = M(0.1), O(0.1)
    true_square = 2 * Fraction(0.1) ** 2  # hypot(0.1, 0.1) ** 2
    for h in (np.hypot(a, b), np.hypot(b, a), a.hypot(b), b.hypot(a)):
        assert type(h) is O and h == b.hypot(b)
        assert Fraction(h.cuts[0].value) ** 2 < true_square < Fraction(h.cuts[-1].value) ** 2
    assert _form(np.hypot(a, b)) == _form(a.hypot(b))
    for f, name in ((np.minimum, 'minimum'), (np.maximum, 'maximum')):
        assert type(f(M(0.1), O(0.2))) is O and type(f(O(0.2), M(0.1))) is O
        assert f(M(0.1), O(0.2)) == f(O(0.1), O(0.2)) == getattr(M(0.1), name)(O(0.2))
    t = np.arctan2(a, b)
    assert type(t) is O and t == b.atan2(b) == a.atan2(b)  # atan2(M, O): y as an O first
    assert type(np.arctan2(b, a)) is O and np.arctan2(b, a) == b.atan2(a)
    with pytest.raises(TypeError):
        np.hypot(a, D(a))  # no subclass: the first operand's method, which refuses a decorated set


def test_ufunc_pinned_examples(np):
    """where a plausible wrong mapping differs"""
    assert np.square(M(-1, 1)) == M(0, 1)  # not x * x, which is [-1, 1]
    assert np.rint(M.parse('[1/2, 5/2]')) == M.parse('{ [0] , [1] , [2] }')  # ties to even, not away
    assert np.arcsin(M(0, Fraction(1, 2))) == M(0, Fraction(1, 2)).asin() != M(0, Fraction(1, 2)).acos()
    assert np.arctan2(1, O(-1)) == O(1).atan2(O(-1)) and type(np.arctan2(1, O(-1))) is O
    assert np.arctan2(M(1), 2) == M(1).atan2(2) != M(2).atan2(1)
    assert np.minimum(1.5, M(1, 2)) == M(1, 2).minimum(1.5) == M(1, 1.5)
    assert np.invert(M(0, 1)) == ~M(0, 1)
    with pytest.raises(TypeError):
        np.floor(Dual.variable(M(1)))  # Dual has no floor: no derivative a solver could use


# 4. WHAT IS NOT MAPPED IS A TypeError

def test_unmapped_ufuncs_and_forms_are_type_errors(np):
    a = M(1, 2)
    assert np.fmod(-7, 2) == -1 and -7 % M(2) == M(1)  # why fmod is not `%`: C's truncated remainder
    refused = [
        lambda: np.fmod(a, 2), lambda: np.float_power(a, 2), lambda: np.deg2rad(a), lambda: np.isnan(a),
        lambda: np.matmul(a, a), lambda: np.logaddexp(a, 1), lambda: np.copysign(a, 1),
        lambda: np.add.reduce(a), lambda: np.add.outer(a, a), lambda: np.add.accumulate(a),
        lambda: np.add(a, 1, out=np.empty((), dtype=object)), lambda: np.add(a, 1, where=True),
        lambda: np.add(a, 1, dtype=object), lambda: np.add(a, 1, casting='unsafe'),
        lambda: np.frompyfunc(lambda x: x, 1, 1)(a),
        lambda: np.add([1, 2], a), lambda: np.add(a, [1, 2]),  # a list is not an array to the hook
        lambda: np.maximum(a, [0, 3]),
    ]
    for thunk in refused:
        with pytest.raises(TypeError):
            thunk()
    assert np.fmax(1, 2) == 2  # numpy's own, untouched


def test_fmin_fmax_are_minimum_maximum(np):
    """Q15(f) (owner, 2026-10-03): mapped, where they were TypeErrors; they agree with minimum/maximum
    wherever there is no nan, and a nan is refused as everywhere, never skipped as numpy's fmin skips it"""
    a = M(1, 2)
    assert np.fmin(a, 1.5) == np.minimum(a, 1.5) == a.minimum(1.5) == M(1, 1.5)
    assert np.fmax(1.5, a) == a.maximum(1.5) == M(1.5, 2)
    assert type(np.fmin(M(0.1), O(0.2))) is O and np.fmax(M(0.1), O(0.2)) == M(0.1).maximum(O(0.2))
    assert np.fmin(D(a), 0).decoration is D(a).minimum(0).decoration
    assert (np.fmin(np.array([0.0, 3.0]), a)).tolist() == [a.minimum(0.0), a.minimum(3.0)]
    for thunk in (lambda: np.fmin(a, math.nan), lambda: np.fmax(math.nan, a),
                  lambda: np.fmin(np.array([1.0, math.nan]), a)):
        with pytest.raises(ValueError, match='nan'):
            thunk()


def test_the_table_is_keyed_by_the_ufunc_object(np):
    """B3: a stand-in named 'sin' (scipy's ufuncs share numpy's names) is not numpy's sin"""
    class StandIn:
        __name__ = 'sin'
        nin = nout = 1

    a = M(1, 2)
    assert numpy_compat.array_ufunc(a, StandIn(), '__call__', a) is NotImplemented


def test_the_table_answers_numpys_sin(np):
    """the positive control of the test above: the same call with numpy's sin is answered"""
    a = M(1, 2)
    assert numpy_compat.array_ufunc(a, np.sin, '__call__', a) == a.sin()


# 5. == AND != AGAINST AN ARRAY: ELEMENTWISE, A BOOL ARRAY (Q15(c), owner 2026-10-03)

def test_equality_against_an_array_is_elementwise(np):
    """numpy's convention, as for every other element type (`np.array([Fraction(1, 2)]) == Fraction(1, 2)`
    is `[True]`) and as `arr == arr2` and pandas already were; until 2026-10-03 `arr == A` was the scalar
    False (identity) although arr held A. each element still compares structurally; the scalar path
    is unchanged"""
    a = M(2)
    f = np.array([2.0, 3.0])
    assert (np.float64(2) == a) is False and (np.float64(2) != a) is True  # the scalar path: identity
    assert (np.float64(1) in [M(1)]) is False
    assert np.equal(a, a) is True and np.not_equal(a, M(2)) is False  # structural, as a == M(2)
    assert (np.array(a) == a) is True  # 0-d: the scalar path
    for got, expected in ((f == a, [False, False]), (a == f, [False, False]), (f != a, [True, True]),
                          (a != f, [True, True])):
        assert got.dtype == bool and got.tolist() == expected  # a float is never equal to a set
    assert (a in f) is False  # unchanged: every element is unequal
    arr = np.array([M(1, 2), a, M.parse('[0, 1] | [2]')])
    assert (arr == a).tolist() == [False, True, False] and (a == arr).tolist() == [False, True, False]
    assert (arr != a).tolist() == [True, False, True]
    assert (a in arr) is True and (M(2.5) in arr) is False
    assert np.flatnonzero(arr == a).tolist() == [1]
    assert (arr == O(2)).tolist() == [False, True, False]  # structural across the classes, as O(2) == M(2)
    grid = np.array([[a, M(3)], [M(3), a]])
    assert (grid == a).shape == (2, 2) and (grid == a).tolist() == [[True, False], [False, True]]
    dual = Dual.variable(a)
    assert (np.array([dual, 1.0], dtype=object) == dual).tolist() == [True, False]  # Dual: identity per element


# 6. ARRAYS HOLD OURS AS ELEMENTS

def test_arrays_hold_intervals_as_elements(np):
    a, b, c = M(1, 2), M(3, 4), M.parse('[0, 1] | [2, 3]')
    for x in (a, c):
        assert np.array(x).shape == () and np.array(x)[()] is x and np.ndim(x) == 0 and np.shape(x) == ()
    arr = np.array([a, b])  # two single-piece sets: a (2, 1, 1, ...) array 64 deep before __array__
    assert arr.shape == (2,) and arr.dtype == object and arr[0] is a and arr[1] is b
    grid = np.array([[a, c], [c, a]])
    assert grid.shape == (2, 2) and grid[0, 1] is c
    assert np.array([M(2), M(3)], dtype=float).tolist() == [2.0, 3.0]
    assert np.array(M(2), dtype=np.float32)[()] == 2
    assert np.asarray(O(Fraction(1, 3)), dtype=float)[()] == 1 / 3  # float(): to nearest, as float(O(1/3))
    with pytest.raises(ValueError):
        np.array(a, dtype=float)  # not a point, as float(a)
    with pytest.raises(ValueError):
        np.asarray(a, copy=False)
    for x in (D(a), Dual.variable(a)):  # scalars to numpy already, no __array__
        pair = np.array([x, x])
        assert pair.shape == (2,) and pair[0] is x


# 7. OBJECT ARRAYS RUN NUMPY'S LOOPS, NOT OURS (documentation pins)

def test_object_array_loops_are_numpys(np):
    a, b = M(1, 2), M.parse('[-1, 0) | [3, 4]')
    arr = np.array([a, b])
    assert (arr + 1).tolist() == [a + 1, b + 1] and (1 + arr).tolist() == [1 + a, 1 + b]
    assert np.sin(arr).tolist() == [a.sin(), b.sin()]
    assert np.sum(arr) == a + b
    assert np.square(arr)[1] == b * b != b ** 2  # numpy's loop: x * x, looser than np.square(b)
    with pytest.raises(TypeError):
        np.arcsin(arr)  # numpy looks for a method named arcsin (Q15(c))
    for thunk in (lambda: np.round(a), lambda: np.around(a), lambda: np.round(arr)):
        with pytest.raises(TypeError):  # not ufuncs: numpy's fallback looks for a method named rint
            thunk()


# 8. AN NDARRAY MEETING ONE OF OURS: ELEMENTWISE INTO AN OBJECT ARRAY

ELEMENTWISE = sorted(BINARY_OPERATORS) + sorted(BINARY_METHODS) + ['arctan2']


@st.composite
def arrays(draw, np):
    k = draw(st.integers(1, 3))
    dtype = draw(st.sampled_from([np.float64, np.int64]))
    values = (st.one_of(st.floats(-20, 20), st.sampled_from([0.0, 0.5, INF, -INF])) if dtype is np.float64
              else st.integers(-20, 20))
    rows = draw(st.sampled_from([None, 2]))
    shape = (k,) if rows is None else (rows, k)
    return np.array(draw(st.lists(values, min_size=math.prod(shape), max_size=math.prod(shape))),
                    dtype=dtype).reshape(shape)


@pytest.mark.parametrize('name', ELEMENTWISE)
@settings(max_examples=20, deadline=None)
@given(data=st.data())
def test_ndarray_and_interval_elementwise(np, name, data):
    """both orders: an object array of f's shape whose element i is the scalar path on the python
    number f.tolist()[i]; an element that raises raises the whole op (the first, in C order). `==` and
    `!=` give a bool array of the same elements (Q15(c))"""
    f = data.draw(arrays(np))
    x = data.draw(ours())
    ufunc, oracle = getattr(np, name), _binary_oracle(name)
    values = f.tolist()  # python numbers: the oracle never sees numpy

    def element(index):
        v = values
        for i in index:
            v = v[i]
        assert type(v) in (int, float)
        return v

    for flip in (False, True):
        def expected():
            out = [np.empty(f.shape, dtype=object) for _ in range(ufunc.nout)]
            for index in np.ndindex(f.shape):
                a = element(index)
                r = oracle(x, a) if flip else oracle(a, x)
                for k, part in enumerate(r if ufunc.nout == 2 else (r,)):
                    out[k][index] = part
            if name in ('equal', 'not_equal'):
                return out[0].astype(bool)
            return tuple(out) if ufunc.nout == 2 else out[0]
        got = outcome(lambda: ufunc(x, f) if flip else ufunc(f, x))
        assert_same_outcome(got, outcome(expected))


def test_elementwise_examples(np):
    a = M(1, 2)
    f = np.linspace(0, 1, 3)
    assert (f + a).tolist() == [0.0 + a, 0.5 + a, 1.0 + a] and (a + f).tolist() == [a + 0.0, a + 0.5, a + 1.0]
    assert (f - a).tolist() == [0.0 - a, 0.5 - a, 1.0 - a] != (a - f).tolist()
    assert (f < a).tolist() == [0.0 < a, 0.5 < a, 1.0 < a]
    q, r = np.divmod(np.array([7, -7]), M(2))
    assert q.tolist() == [7 // M(2), -7 // M(2)] and r.tolist() == [7 % M(2), -7 % M(2)]
    assert (np.array([[1], [2]]) * a).shape == (2, 1)
    assert (np.array([1.5]) * Dual.variable(a))[0].derivative == M(1.5)
    assert (np.array([1.5]) + D(a))[0] == 1.5 + D(a)
    for bad in (np.array([True, False]), np.array([1j])):
        with pytest.raises(TypeError):
            bad + a
    with pytest.raises(ValueError):
        np.array([1.0, math.nan]) + a  # a nan element: the whole op raises, as nan + a


def test_elementwise_leaves_numpys_float_flags_alone(np):
    """numpy reads the floating-point status flags after an object loop, and the library's own
    python float arithmetic leaves overflow set here: the scalar path gives no warning, so neither
    does the array (a gap the sabotage found, 2026-09-28: `test_ndarray_and_interval_elementwise`
    had found it by chance once and then drew past it)"""
    x = M.parse('(-inf, 2.225073858507203e-309)')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert (np.array([1.0]) / x).tolist() == [1.0 / x]


def test_elementwise_leaves_every_float_flag_alone(np):
    """underflow too (a subnormal end), which numpy ignores by default: only a caller's errstate
    shows it, and the scalar path never raises it (the M16d review's sabotage: `errstate(over=...)`
    stayed green, 2026-09-28)"""
    x = M.parse('(-inf, 2.225073858507203e-309)')
    with np.errstate(all='raise'):
        assert (np.array([1e-308]) + M(0)).tolist() == [1e-308 + M(0)]
        assert (np.array([1.0]) / x).tolist() == [1.0 / x]


def test_elementwise_warning_is_the_callers(np):
    from intervals.errors import IndeterminateResultWarning
    with pytest.warns(IndeterminateResultWarning) as caught:
        np.array([1.0, 0.0]) / M(0)
    assert {os.path.normcase(os.path.abspath(w.filename)) for w in caught} == {HERE}


# 9. A FOREIGN REAL IS ITS EXACT VALUE

class Wide:
    """a stand-in foreign real (np.longdouble, gmpy2's mpfr): an exact value, float() rounding it"""

    def __init__(self, q):
        self.q = q

    def as_integer_ratio(self):
        if isinstance(self.q, float):
            return self.q.as_integer_ratio()  # raises for nan and ±inf, as numpy's and gmpy2's do
        return self.q.numerator, self.q.denominator

    def __float__(self):
        return float(self.q)


class Rat:
    """a stand-in foreign rational (gmpy2's mpq): numerator and denominator"""

    def __init__(self, q):
        self.q = Fraction(q)

    numerator = property(lambda self: self.q.numerator)
    denominator = property(lambda self: self.q.denominator)

    def __float__(self):
        return float(self.q)


numbers.Real.register(Wide)
numbers.Rational.register(Rat)

exact_fractions = st.one_of(
    st.fractions(max_denominator=10 ** 20),
    st.floats(allow_nan=False, allow_infinity=False).map(Fraction),
    st.integers(-10 ** 400, 10 ** 400).map(Fraction),
    st.tuples(st.integers(-2 ** 80, 2 ** 80), st.integers(0, 1200)).map(lambda t: Fraction(t[0], 2 ** t[1])),
)


def _double_or_exact(q: Fraction):
    try:
        f = float(q)
    except OverflowError:
        f = None
    if f == q:
        return f
    return int(q) if q.denominator == 1 else q


@settings(max_examples=300, deadline=None)
@given(q=exact_fractions)
@example(q=Fraction(6148914691236517205, 2 ** 64))  # x86-64's long double nearest 1/3
@example(q=Fraction(3))
@example(q=Fraction(10 ** 400))
def test_foreign_reals_are_exact(q):
    want = _double_or_exact(q)
    got = normalize_value(Wide(q))
    assert type(got) is type(want) and got == want  # a double stays the float it is today
    assert q in O(0) + Wide(q) and q in M(Wide(q))
    got = normalize_value(Rat(q))  # a Rational is exact by type, as a Fraction
    assert got == q and type(got) is (int if q.denominator == 1 else Fraction)


def test_foreign_real_specials():
    assert normalize_value(Wide(3.0)) == 3.0 and type(normalize_value(Wide(3.0))) is float
    assert normalize_value(Wide(INF)) == INF and normalize_value(Wide(-INF)) == -INF
    assert repr(normalize_value(Wide(-0.0))) == '0.0'
    with pytest.raises(ValueError):
        normalize_value(Wide(math.nan))
    assert normalize_value(Rat(Fraction(1, 2))) == Fraction(1, 2)  # not 0.5: exact by type
    third = Wide(Fraction(1, 3))
    assert O(0) + third == O(Fraction(1, 3))  # was the double 0.333..., which does not hold 1/3
    assert M(1, 8).log(Rat(2)) == M(1, 8).log(2)  # the log base: any real (was int, float, Fraction)


class Plain:
    """a foreign real with no as_integer_ratio: float() of it, as before M16d"""

    def __init__(self, f):
        self.f = f

    def __float__(self):
        return self.f


numbers.Real.register(Plain)


def test_a_foreign_real_without_a_ratio_is_its_float():
    """the fallback of `cuts._exact_value` (the M16d review's sabotage: refusing it stayed green)"""
    got = normalize_value(Plain(0.1))
    assert type(got) is float and got == 0.1
    assert M(Plain(0.5)) == M(0.5)
    with pytest.raises(ValueError):
        normalize_value(Plain(math.nan))


@settings(max_examples=200, deadline=None)
@given(bits=st.integers(0, 2 ** 32 - 1))
def test_float32_is_the_double_it_holds(np, bits):
    v = np.uint32(bits).view(np.float32)
    if np.isnan(v):
        with pytest.raises(ValueError):
            normalize_value(v)
        return
    got = normalize_value(v)
    assert type(got) is float and got == float(v)
    if got == 0:
        assert math.copysign(1, got) == 1


@settings(max_examples=100, deadline=None)
@given(a=st.floats(-1e6, 1e6), b=st.integers(-8, 8))
def test_longdouble_is_exact(np, a, b):
    """discriminates where the long double is wider than a double (every linux CI job); on windows
    it is the double case, still true"""
    v = np.longdouble(a) + np.longdouble(b) * np.longdouble(2) ** -60
    q = Fraction(*(int(k) for k in v.as_integer_ratio()))
    want = _double_or_exact(q)
    got = normalize_value(v)
    assert type(got) is type(want) and got == want
    assert q in O(0) + v


def test_gmpy2_values_are_exact():
    gmpy2 = pytest.importorskip('gmpy2')
    assert normalize_value(gmpy2.mpq(1, 2)) == Fraction(1, 2) and type(normalize_value(gmpy2.mpq(1, 2))) is Fraction
    assert type(normalize_value(gmpy2.mpz(3))) is int
    assert O(0) + gmpy2.mpq(1, 3) == O(Fraction(1, 3))
    wide = gmpy2.mpfr(1, 200) / 3
    assert normalize_value(wide) == Fraction(*(int(k) for k in wide.as_integer_ratio()))
    assert repr(normalize_value(gmpy2.mpfr('0.5'))) == '0.5'


# 10. INTEGER ARGUMENTS TAKE ANY Integral BUT bool

def test_integer_arguments_take_numpy_ints(np):
    a = M(1, 8)
    same = [
        (lambda: a.rootn(np.int64(3)), lambda: a.rootn(3)),
        (lambda: a.rootn(np.int8(-3)), lambda: a.rootn(-3)),
        (lambda: M(0.125, 0.135).round(np.int64(2)), lambda: M(0.125, 0.135).round(2)),
        (lambda: M(0.125, 0.135).round_ties_away(np.int64(2)), lambda: M(0.125, 0.135).round_ties_away(2)),
        (lambda: pown_rev(M(1, 4), np.int64(-2)), lambda: pown_rev(M(1, 4), -2)),
        (lambda: D(a).rootn(np.int64(3)), lambda: D(a).rootn(3)),
        (lambda: D(a).round(np.int64(1)), lambda: D(a).round(1)),
        (lambda: Dual.variable(a).rootn(np.int64(3)), lambda: Dual.variable(a).rootn(3)),
        (lambda: a.log(np.int64(2)), lambda: a.log(2)),
        (lambda: a.log(np.float32(2)), lambda: a.log(2.0)),
        (lambda: a.log(np.float64(2)), lambda: a.log(2.0)),
        (lambda: a.log(np.longdouble(2)), lambda: a.log(2.0)),
        (lambda: Dual.variable(a).log(np.int64(2)), lambda: Dual.variable(a).log(2)),
    ]
    for got, want in same:
        assert outcome(got) == outcome(want)
    for thunk in (lambda: a.rootn(np.bool_(True)), lambda: a.rootn(True), lambda: a.round(True),
                  lambda: a.round(np.bool_(True)), lambda: pown_rev(a, True), lambda: a.log(True),
                  lambda: a.log(np.bool_(True)), lambda: a.rootn(2.0), lambda: a.log('2'),
                  lambda: a << np.int64(3), lambda: np.left_shift(a, 3), lambda: np.right_shift(a, 3)):  # no shifts
        with pytest.raises(TypeError):
            thunk()
    for thunk in (lambda: a.log(math.nan), lambda: a.log(np.float64(math.nan)), lambda: a.log(1), lambda: a.log(-2)):
        with pytest.raises(ValueError, match='a logarithm needs'):
            thunk()


def test_numpy_ndigits_past_int64(np):
    """`10 ** ndigits` in an int64 wraps past 10 ** 18: ndigits is an int first (the M16d review's
    sabotage: dropping `int(ndigits)` stayed green, 2026-09-28). single-point sets, so no hull warns"""
    for a in (M(Fraction(1, 3)), M(0.125)):
        for nd in (19, 20, 30):
            assert a.round(np.int64(nd)) == a.round(nd)
            assert a.round_ties_away(np.int64(nd)) == a.round_ties_away(nd)


def test_numpy_pown_rev_degrees(np):
    """`pown_rev` takes int(n) first: an int64 degree past -2 overflowed inside (the M16d review's
    sabotage: dropping `int(n)` stayed green, 2026-09-28)"""
    for n in (3, 2, -3, 2 ** 62):
        assert pown_rev(M(1, 4), np.int64(n)) == pown_rev(M(1, 4), n)


def test_dual_rootn_takes_the_degree_as_an_int(np):
    """B2: `Dual.rootn` computes `n - 1`; in an int64 that wraps at -2**63 (a RuntimeWarning, and a
    root of degree 2**63 - 1 where a larger one was asked). `int(n)` first"""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        d = Dual.variable(M(1)).rootn(np.int64(-2 ** 63))
    assert d.value == M(1) and d.derivative == M(Fraction(-1, 2 ** 63))


def test_numpy_durations_are_no_numbers(np):
    """numpy registers `timedelta64` as `numbers.Integral` (it subclasses `signedinteger`), and `int()` of it is its
    count in its own unit, so `MultiInterval(np.timedelta64(3, 'ns'))` was `[3]` and `np.timedelta64(3, 'Y')` a 3
    (M8's review, F6, 2026-10-04): every place the numeric class takes a number, or an int argument, refuses dtype
    kinds 'm' and 'M' (TypeError), as it refuses a `datetime.timedelta`"""
    a = M(1, 2)
    for t in (np.timedelta64(3, 'ns'), np.timedelta64(3, 'Y'), np.timedelta64(3, 's'), np.timedelta64('NaT', 'ns'),
              np.datetime64(3, 'ns'), np.datetime64('2024-01-01')):
        for thunk in (lambda: M(t), lambda: O(t), lambda: M(0, t), lambda: M.from_pieces([(0, t)]), lambda: a + t,
                      lambda: t + a, lambda: a * t, lambda: t * a, lambda: a / t, lambda: t in a, lambda: a < t,
                      lambda: a ** t, lambda: t ** a, lambda: a.expand(t), lambda: a[0:t], lambda: a.rootn(t),
                      lambda: a.round(t), lambda: pown_rev(a, t), lambda: D(M(1)) + t, lambda: Dual.variable(a) * t,
                      lambda: normalize_value(t)):
            with pytest.raises(TypeError):
                thunk()
    assert a * np.int64(3) == M(3, 6) and a ** np.int64(2) == M(1, 4)  # numpy's ints are still numbers
