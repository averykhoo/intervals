"""
intervals.autodiff against an independent oracle: arb's taylor series, through python-flint (D14)

`arb_series([x, 1])` is the variable at x; arb carries it through each op as a truncated series whose
coefficients are balls proven to hold the true ones, so coefficient 0 holds f(x) and coefficient 1
holds f'(x). arb's series has exp, log, sqrt and the circular functions and their inverses; the rest
are written from those (sinh as (e^x - e^-x) / 2, acosh as log(x + sqrt(x^2 - 1)), ...), which is
an independent derivation of the same derivative.

the checks:
* every op of `Dual`, at a drawn point p of a drawn interval X: f(p) is in the value's set and f'(p)
  in the derivative's set (soundness); on a degenerate X the derivative is tight (relative width
  1e-10), so a formula that encloses by being loose is caught
* random expression trees over the ops (depth 3), the same two checks at a point of X
* decorations: the value and the derivative are dac or better on an interval inside the op's
  domain where it is differentiable, and not where it is not (sqrt, rootn, pow at 0, abs across 0, a
  pole), which `intervals.solver` relies on

a ball that straddles an end of our set is undecided at that precision; it is retried at a higher
one, and past the last the example is rejected with `assume`
"""
import math
from fractions import Fraction

import pytest
from flint import arb
from flint import arb_series
from flint import ctx
from flint import fmpq
from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import kernel
from intervals.autodiff import Dual
from intervals.autodiff import derivative

O = OutwardMultiInterval
INF = math.inf
PRECISIONS = (200, 1000)


# THE ORACLE

def _arb(q) -> arb:
    if isinstance(q, float):
        return arb(q)
    q = Fraction(q)
    return arb(fmpq(q.numerator, q.denominator))


def _inside(s: MultiInterval, ball: arb):
    """True if the ball is inside a piece of s, False if it is apart from every piece, else None"""
    apart = True
    for lo, lo_closed, hi, hi_closed in kernel.pieces(s.cuts):
        above = lo == -INF or _arb(lo) < ball or (lo_closed and _arb(lo) <= ball)
        below = hi == INF or ball < _arb(hi) or (hi_closed and ball <= _arb(hi))
        if above and below:
            return True
        if not ((lo != -INF and ball < _arb(lo)) or (hi != INF and ball > _arb(hi))):
            apart = False
    return False if apart else None


def _check(s: MultiInterval, oracle, what: str):
    """assert the true value, held by `oracle()`'s ball at each precision, is in s"""
    old = ctx.prec
    try:
        for prec in PRECISIONS:
            ctx.prec = prec
            verdict = _inside(s, oracle())
            if verdict is None:
                continue
            assert verdict, f'{what}: {oracle()} is not in {s}'
            return
    finally:
        ctx.prec = old
    assume(False)


def _variable(p) -> arb_series:
    return arb_series([_arb(p), 1], prec=2)


def _coefficient(series: arb_series, n: int) -> arb:
    """the nth coefficient: arb drops trailing zero ones (`x * x` at 0 has none)"""
    coeffs = series.coeffs()
    return coeffs[n] if n < len(coeffs) else arb(0)


def _ln(v) -> arb:
    return arb(v).log()


def _sinh(s):
    return (s.exp() - (-s).exp()) / 2


def _cosh(s):
    return (s.exp() + (-s).exp()) / 2


# each op: our function of a Dual, arb's of a series, and the open interval its points are drawn from
OPS = {
    'neg': (lambda u: -u, lambda s: -s, (-20, 20)),
    'abs': (abs, lambda s: s if _coefficient(s, 0) > 0 else -s, (-20, 20)),
    'reciprocal': (lambda u: u.reciprocal(), lambda s: 1 / s, (0.05, 20)),
    'square': (lambda u: u ** 2, lambda s: s * s, (-20, 20)),
    'cube': (lambda u: u ** 3, lambda s: s * s * s, (-20, 20)),
    'inverse square': (lambda u: u ** -2, lambda s: 1 / (s * s), (0.05, 20)),
    'pow 3/2': (lambda u: u ** Fraction(3, 2), lambda s: (s.log() * 3 / 2).exp(), (0.01, 20)),
    'pow 0.5': (lambda u: u ** 0.5, lambda s: s.sqrt(), (0.01, 20)),
    'pow self': (lambda u: u ** u, lambda s: (s * s.log()).exp(), (0.01, 5)),
    'rpow 2': (lambda u: 2 ** u, lambda s: (s * _ln(2)).exp(), (-20, 20)),
    'sqrt': (lambda u: u.sqrt(), lambda s: s.sqrt(), (0.01, 20)),
    'cbrt': (lambda u: u.cbrt(), lambda s: (s.log() / 3).exp(), (0.01, 20)),
    'rootn 5': (lambda u: u.rootn(5), lambda s: (s.log() / 5).exp(), (0.01, 20)),
    'exp': (lambda u: u.exp(), lambda s: s.exp(), (-20, 20)),
    'expm1': (lambda u: u.expm1(), lambda s: s.exp() - 1, (-20, 20)),
    'exp2': (lambda u: u.exp2(), lambda s: (s * _ln(2)).exp(), (-20, 20)),
    'exp10': (lambda u: u.exp10(), lambda s: (s * _ln(10)).exp(), (-20, 20)),
    'log': (lambda u: u.log(), lambda s: s.log(), (0.01, 20)),
    'log base 3': (lambda u: u.log(3), lambda s: s.log() / _ln(3), (0.01, 20)),
    'log2': (lambda u: u.log2(), lambda s: s.log() / _ln(2), (0.01, 20)),
    'log10': (lambda u: u.log10(), lambda s: s.log() / _ln(10), (0.01, 20)),
    'log1p': (lambda u: u.log1p(), lambda s: (1 + s).log(), (-0.99, 20)),
    'sin': (lambda u: u.sin(), lambda s: s.sin(), (-20, 20)),
    'cos': (lambda u: u.cos(), lambda s: s.cos(), (-20, 20)),
    'tan': (lambda u: u.tan(), lambda s: s.tan(), (-1.5, 1.5)),
    'cot': (lambda u: u.cot(), lambda s: s.cos() / s.sin(), (0.05, 3.0)),
    'sec': (lambda u: u.sec(), lambda s: 1 / s.cos(), (-1.5, 1.5)),
    'csc': (lambda u: u.csc(), lambda s: 1 / s.sin(), (0.05, 3.0)),
    'asin': (lambda u: u.asin(), lambda s: s.asin(), (-0.99, 0.99)),
    'acos': (lambda u: u.acos(), lambda s: s.acos(), (-0.99, 0.99)),
    'atan': (lambda u: u.atan(), lambda s: s.atan(), (-20, 20)),
    'acot': (lambda u: u.acot(), lambda s: arb.pi() / 2 - s.atan(), (-20, 20)),
    'sinh': (lambda u: u.sinh(), _sinh, (-20, 20)),
    'cosh': (lambda u: u.cosh(), _cosh, (-20, 20)),
    'tanh': (lambda u: u.tanh(), lambda s: _sinh(s) / _cosh(s), (-20, 20)),
    'coth': (lambda u: u.coth(), lambda s: _cosh(s) / _sinh(s), (0.05, 20)),
    'sech': (lambda u: u.sech(), lambda s: 1 / _cosh(s), (-20, 20)),
    'csch': (lambda u: u.csch(), lambda s: 1 / _sinh(s), (0.05, 20)),
    'asinh': (lambda u: u.asinh(), lambda s: (s + (s * s + 1).sqrt()).log(), (-20, 20)),
    'acosh': (lambda u: u.acosh(), lambda s: (s + (s * s - 1).sqrt()).log(), (1.01, 20)),
    'atanh': (lambda u: u.atanh(), lambda s: ((1 + s) / (1 - s)).log() / 2, (-0.99, 0.99)),
    'acoth': (lambda u: u.acoth(), lambda s: ((s + 1) / (s - 1)).log() / 2, (1.01, 20)),
}


@st.composite
def boxes(draw, bounds):
    """(lo, hi, p): floats lo <= hi inside the open bounds, and an exact point p of [lo, hi]"""
    a, b = bounds
    lo = draw(st.floats(a, b, exclude_min=True, exclude_max=True))
    width = draw(st.sampled_from([0.0, 0.0, 1e-9, 1e-3, 0.1, 1.0]))
    hi = min(lo + width, math.nextafter(b, a))
    hi = max(hi, lo)
    t = draw(st.sampled_from([Fraction(0), Fraction(1), Fraction(1, 2), Fraction(1, 3)]))
    return lo, hi, Fraction(lo) + t * (Fraction(hi) - Fraction(lo))


# THE OPS, ONE AT A TIME

@pytest.mark.parametrize('name', OPS)
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_op_encloses_value_and_derivative(name, data):
    ours, theirs, bounds = OPS[name]
    lo, hi, p = data.draw(boxes(bounds))
    if name == 'abs':
        assume(lo > 0 or hi < 0)  # abs is differentiable off 0 only
    y = ours(Dual.variable(O(lo, hi)))
    _check(y.value, lambda: _coefficient(theirs(_variable(p)), 0), f'{name} value at {p}')
    _check(y.derivative, lambda: _coefficient(theirs(_variable(p)), 1), f'{name} derivative at {p}')


@pytest.mark.parametrize('name', OPS)
@settings(max_examples=30, deadline=None)
@given(data=st.data())
def test_op_derivative_is_tight_at_a_point(name, data):
    """a degenerate X: the derivative's set is a few ulps wide, so a loose formula is caught"""
    ours, theirs, bounds = OPS[name]
    a, b = bounds
    x = data.draw(st.floats(a, b, exclude_min=True, exclude_max=True))
    assume(name != 'abs' or x != 0)
    d = ours(Dual.variable(O(x))).derivative
    assert d.is_contiguous and d.is_finite, (name, x, d)
    scale = max(1.0, abs(float(d.mid())))
    assert d.wid() <= 1e-10 * scale, (name, x, d)


def test_every_op_of_the_table_is_a_dual_method():
    """the table above covers every elementary method of Dual (so a new one needs a row)"""
    methods = {name for name in vars(Dual) if not name.startswith('_')} - {'variable', 'constant', 'value', 'derivative'}
    covered = {name.split()[0] for name in OPS}
    assert methods <= covered, methods - covered


# EXPRESSION TREES

LEAVES = st.one_of(
    st.just((lambda x: x, lambda s: s, 'x')),
    st.integers(-3, 3).map(lambda c: (lambda x, c=c: c, lambda s, c=c: c, str(c))),
)


def _lift(v, x: Dual) -> Dual:
    """a constant leaf as a constant Dual of the variable's class, so the methods apply to it"""
    return v if isinstance(v, Dual) else Dual.constant(type(x.value)(v))


def _lift_arb(v) -> arb_series:
    return v if isinstance(v, arb_series) else arb_series([arb(v)], prec=2)


def _unary(child):
    (f, g, text), which = child
    ours, theirs = {
        'neg': (lambda u: -u, lambda s: -s),
        'sin': (lambda u: u.sin(), lambda s: s.sin()),
        'cos': (lambda u: u.cos(), lambda s: s.cos()),
        'atan': (lambda u: u.atan(), lambda s: s.atan()),
        'hypot1': (lambda u: (u ** 2 + 1).sqrt(), lambda s: (s * s + 1).sqrt()),
        'logsq1': (lambda u: (u ** 2 + 1).log(), lambda s: (s * s + 1).log()),
        'square': (lambda u: u ** 2, lambda s: s * s),
    }[which]
    return (lambda x: ours(_lift(f(x), x)), lambda s: theirs(_lift_arb(g(s))), f'{which}({text})')


def _binary(children):
    (f, g, a), (h, k, b), which = children
    if which == '/':  # a denominator with no zero
        return (lambda x: _lift(f(x), x) / (_lift(h(x), x) ** 2 + 1),
                lambda s: _lift_arb(g(s)) / (_lift_arb(k(s)) * _lift_arb(k(s)) + 1),
                f'({a}) / (({b})**2 + 1)')
    ours = {'+': lambda u, v: u + v, '-': lambda u, v: u - v, '*': lambda u, v: u * v}[which]
    return (lambda x: ours(_lift(f(x), x), _lift(h(x), x)),
            lambda s: ours(_lift_arb(g(s)), _lift_arb(k(s))), f'({a}) {which} ({b})')


expressions = st.recursive(
    LEAVES,
    lambda children: st.one_of(
        st.tuples(children, st.sampled_from(['neg', 'sin', 'cos', 'atan', 'hypot1', 'logsq1', 'square'])).map(_unary),
        st.tuples(children, children, st.sampled_from(['+', '-', '*', '/'])).map(_binary),
    ),
    max_leaves=6,
)


@settings(max_examples=200, deadline=None)
@given(expressions, boxes((-2, 2)))
@example((lambda x: x / (x ** 2 + 1), lambda s: s / (s * s + 1), 'x / (x**2 + 1)'),
         (-1.0, 1.0, Fraction(1, 3)))
def test_expression_encloses_value_and_derivative(expression, box):
    f, g, text = expression
    lo, hi, p = box
    x = Dual.variable(O(lo, hi))
    y = _lift(f(x), x)

    _check(y.value, lambda: _coefficient(_lift_arb(g(_variable(p))), 0), f'{text} at {p}')
    _check(y.derivative, lambda: _coefficient(_lift_arb(g(_variable(p))), 1), f'd/dx {text} at {p}')


# DECORATIONS: the C¹ proof the solver takes

def _decorated(lo, hi) -> Dual:
    return Dual(DecoratedInterval(O(lo, hi)), DecoratedInterval(O(1)))


@pytest.mark.parametrize('name', OPS)
def test_decorated_derivative_is_dac_inside_the_domain(name):
    """on an interval inside the op's domain, off abs's kink, value and derivative are dac or com"""
    ours, _, (a, b) = OPS[name]
    lo, hi = (a + (b - a) / 4, a + (b - a) / 3)
    if name == 'abs':
        lo, hi = 1, 2
    y = ours(_decorated(lo, hi))
    assert y.value.decoration >= Decoration.DAC, (name, y)
    assert y.derivative.decoration >= Decoration.DAC, (name, y)


@pytest.mark.parametrize('f, lo, hi, what', [
    (lambda u: u.sqrt(), 0, 1, 'sqrt at 0: 1 / (2 sqrt 0)'),
    (lambda u: u.rootn(4), 0, 1, 'rootn at 0'),
    (lambda u: u.cbrt(), -1, 1, 'cbrt at 0'),
    (lambda u: u ** Fraction(1, 2), 0, 1, 'pow 1/2 at 0'),
    (lambda u: u ** u, 0, 1, 'u ** u at 0: log 0'),
    (lambda u: u.asin(), 0, 1, 'asin at 1'),
    (lambda u: u.acosh(), 1, 2, 'acosh at 1'),
    (lambda u: 1 / u, -1, 1, 'a pole'),
    (lambda u: u.tan(), 1, 2, "tan's pole"),
])
@pytest.mark.filterwarnings('ignore::intervals.errors.IntervalWarning')  # the formula leaves its domain
def test_decorated_not_differentiable_is_trv(f, lo, hi, what):
    y = f(_decorated(lo, hi))
    assert min(y.value.decoration, y.derivative.decoration) is Decoration.TRV, (what, y)


def test_decorated_abs_across_zero_is_def():
    """abs is continuous across 0 (value com) but sign is not (derivative def): not C¹"""
    y = abs(_decorated(-1, 1))
    assert y.value.decoration is Decoration.COM
    assert y.derivative.decoration is Decoration.DEF


# THE TYPE

def test_derivative_helper():
    assert derivative(lambda t: t ** 3, 2) == MultiInterval(12)
    assert derivative(lambda t: 5, MultiInterval(0, 1)) == MultiInterval(0)
    assert derivative(lambda t: t.sin(), O(0)) == O(1)
    with pytest.raises(TypeError, match='not a Dual'):
        derivative(lambda t: t.value, MultiInterval(0, 1))


def test_pow_zero_is_the_constant_one():
    y = Dual.variable(MultiInterval(-1, 1)) ** 0
    assert (y.value, y.derivative) == (MultiInterval(1), MultiInterval(0))


@pytest.mark.parametrize('u', [2, 1e300, 1e-300])
@pytest.mark.parametrize('r', [0.1, 1e-20, 0.3])
def test_pow_number_exponent_derivative_encloses(r, u):
    """
    `(u ** r)' = r u ** (r - 1)`, outward: M15 computed `r - 1` in r's own arithmetic, a float
    rounded to nearest, so the outward class's derivative missed the true value (6 of these 9 on
    2026-09-28, python floats; found by the M16d critique). a number exponent is a point set first
    """
    d = (Dual.variable(O(u)) ** r).derivative
    old = ctx.prec
    try:
        ctx.prec = 400
        verdict = _inside(d, _arb(r) * (_arb(u).log() * (_arb(r) - 1)).exp())
    finally:
        ctx.prec = old
    assert verdict is True, f'{r} {u ** (r - 1)!r}: {d}'


@pytest.mark.parametrize('r', [2.0 ** 60, 2 ** 60, Fraction(2 ** 60)])
def test_pow_integral_exponent_derivative_is_exact(r):
    """an integral exponent n is pown and `n - 1` is exact int arithmetic: the float `2.0 ** 60 - 1`
    rounds to `2.0 ** 60`, which gave `(-1) ** (2 ** 60)`, the derivative's sign flipped (M15, found
    2026-09-28). the base is -1 because `O(2) ** 2 ** 60` is an exact power too large to compute"""
    y = Dual.variable(O(-1)) ** r
    assert y.value == O(1) and y.derivative == O(-2 ** 60)


def test_constants_mix_in():
    x = Dual.variable(MultiInterval(1, 2))
    y = 2 - Fraction(1, 2) * x + MultiInterval(1)
    assert (y.value, y.derivative) == (MultiInterval(2, Fraction(5, 2)), MultiInterval(Fraction(-1, 2)))
    y = 1 / x
    assert y.derivative == MultiInterval(-1, Fraction(-1, 4))


def test_kinds_do_not_mix():
    with pytest.raises(TypeError, match='not one of each'):
        Dual(MultiInterval(1), DecoratedInterval(MultiInterval(1)))
    with pytest.raises(TypeError):
        Dual(1, MultiInterval(1))
    with pytest.raises(TypeError):
        _decorated(1, 2) + MultiInterval(1)  # a DecoratedInterval refuses a bare set
    with pytest.raises(TypeError):
        Dual.variable(MultiInterval(1)) + 'a'


def test_immutable():
    x = Dual.variable(MultiInterval(1))
    with pytest.raises(AttributeError):
        x._value = MultiInterval(2)
    with pytest.raises(AttributeError):
        del x._derivative


def test_no_step_functions():
    """floor has no derivative a solver could use: it is not a method of Dual"""
    with pytest.raises(AttributeError):
        Dual.variable(MultiInterval(1)).floor()
