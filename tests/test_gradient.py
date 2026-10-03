"""
intervals.autodiff's gradient and jacobian (M16, H3's second part) against arb, through python-flint

column j of the jacobian at a point p is `∂F/∂x_j`, and arb gives it as coefficient 1 of
`F(p_1, ..., p_j + t, ..., p_n)`: `arb_series([p_j, 1])` for x_j and `arb_series([p_k])` for the rest,
each a series whose coefficients are balls proven to hold the true ones.

the checks:
* random expression trees in two variables (a leaf is x, y or a constant; the unary and binary ops of
  `tests/test_autodiff.py`'s trees), `F = (e1, e2)` over a drawn box: every entry of the jacobian holds
  arb's partial at a drawn point of the box (soundness)
* on a degenerate box every entry is a few ulps wide (sharpness). a wrong seed (`[1]` in two
  coordinates) is still tight at a point, the sum of two partials: `::test_seeds` and the enclosure
  test catch that, not this one
* `gradient` is the jacobian's row, and at n == 1 it is `derivative`, set and decoration
* the decorations of each column, the arguments

the helpers `_check`, `_arb`, `_coefficient`, `_lift_arb` and `boxes` are `tests/test_autodiff.py`'s
"""
from fractions import Fraction

import pytest
from flint import arb
from flint import arb_series
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals.autodiff import Dual
from intervals.autodiff import derivative
from intervals.autodiff import gradient
from intervals.autodiff import jacobian
from tests.test_autodiff import _arb
from tests.test_autodiff import _check
from tests.test_autodiff import _coefficient
from tests.test_autodiff import _lift_arb
from tests.test_autodiff import boxes

M = MultiInterval
O = OutwardMultiInterval
D = DecoratedInterval


# EXPRESSION TREES IN TWO VARIABLES: (ours of two Duals, arb's of two series, text)

def _lift(v, x: Dual) -> Dual:
    """a constant as a constant Dual of the variables' class, so the methods apply to it (decorated:
    the point of the interval's class, newDec'd)"""
    if isinstance(v, Dual):
        return v
    if isinstance(x.value, D):
        return Dual.constant(D(type(x.value.interval)(v)))
    return Dual.constant(type(x.value)(v))


LEAVES = st.one_of(
    st.just((lambda x, y: x, lambda s, t: s, 'x')),
    st.just((lambda x, y: y, lambda s, t: t, 'y')),
    st.integers(-3, 3).map(lambda c: (lambda x, y, c=c: c, lambda s, t, c=c: c, str(c))),
)

UNARY = {
    'neg': (lambda u: -u, lambda s: -s),
    'sin': (lambda u: u.sin(), lambda s: s.sin()),
    'cos': (lambda u: u.cos(), lambda s: s.cos()),
    'atan': (lambda u: u.atan(), lambda s: s.atan()),
    'hypot1': (lambda u: (u ** 2 + 1).sqrt(), lambda s: (s * s + 1).sqrt()),
    'logsq1': (lambda u: (u ** 2 + 1).log(), lambda s: (s * s + 1).log()),
    'square': (lambda u: u ** 2, lambda s: s * s),
}


def _unary(child):
    (f, g, text), which = child
    ours, theirs = UNARY[which]
    return (lambda x, y: ours(_lift(f(x, y), x)), lambda s, t: theirs(_lift_arb(g(s, t))), f'{which}({text})')


def _binary(children):
    (f, g, a), (h, k, b), which = children
    if which == '/':  # a denominator with no zero
        return (lambda x, y: _lift(f(x, y), x) / (_lift(h(x, y), x) ** 2 + 1),
                lambda s, t: _lift_arb(g(s, t)) / (_lift_arb(k(s, t)) * _lift_arb(k(s, t)) + 1),
                f'({a}) / (({b})**2 + 1)')
    ours = {'+': lambda u, v: u + v, '-': lambda u, v: u - v, '*': lambda u, v: u * v}[which]
    return (lambda x, y: ours(_lift(f(x, y), x), _lift(h(x, y), x)),
            lambda s, t: ours(_lift_arb(g(s, t)), _lift_arb(k(s, t))), f'({a}) {which} ({b})')


expressions = st.recursive(
    LEAVES,
    lambda children: st.one_of(
        st.tuples(children, st.sampled_from(sorted(UNARY))).map(_unary),
        st.tuples(children, children, st.sampled_from(['+', '-', '*', '/'])).map(_binary),
    ),
    max_leaves=6,
)


def _series(p, j):
    """the arguments of arb's F for column j at the point p: x_j seeded, the rest constant"""
    return [arb_series([_arb(q), 1] if k == j else [_arb(q)], prec=2) for k, q in enumerate(p)]


def _F(e1, e2):
    (f1, g1, t1), (f2, g2, t2) = e1, e2

    def ours(x, y):
        return (_lift(f1(x, y), x), _lift(f2(x, y), x))

    def theirs(s, t):
        return (_lift_arb(g1(s, t)), _lift_arb(g2(s, t)))
    return ours, theirs, f'({t1}, {t2})'


@settings(max_examples=60, deadline=None)
@given(expressions, expressions, boxes((-2, 2)), boxes((-2, 2)))
@example((lambda x, y: x * y, lambda s, t: s * t, 'x * y'),
         (lambda x, y: (x - y).sin(), lambda s, t: (s - t).sin(), 'sin(x - y)'),
         (-1.0, 1.0, Fraction(1, 3)), (0.5, 1.5, Fraction(1, 2)))
def test_jacobian_encloses_the_partials(e1, e2, bx, by):
    ours, theirs, text = _F(e1, e2)
    (xlo, xhi, xp), (ylo, yhi, yp) = bx, by
    J = jacobian(ours, [O(xlo, xhi), O(ylo, yhi)])
    p = (xp, yp)
    for i in range(2):
        for j in range(2):
            _check(J[i][j], lambda i=i, j=j: _coefficient(theirs(*_series(p, j))[i], 1),
                   f'd{text}[{i}]/dx{j} at {p}')


@settings(max_examples=40, deadline=None)
@given(expressions, expressions, st.floats(-2, 2), st.floats(-2, 2))
def test_jacobian_is_tight_at_a_point(e1, e2, x, y):
    """a degenerate box: every entry a few ulps wide, so a loose formula is caught"""
    ours, _, text = _F(e1, e2)
    for row in jacobian(ours, [O(x), O(y)]):
        for d in row:
            assert d.is_contiguous and d.is_finite, (text, x, y, d)
            assert d.wid() <= 1e-10 * max(1.0, abs(float(d.mid()))), (text, x, y, d)


FUNCTIONS = [
    (lambda x, y: x * y.sin() - y ** 2, [O(1, 2), O(-1, 3)]),
    (lambda x, y: (x ** 2 + y ** 2).sqrt(), [M(1, 2), 3]),
    (lambda x, y: x / y, [D(M(1, 2)), D(M(-1, 1))]),
    (lambda x, y: 7, [M(0, 1), M(2, 3)]),
]


@pytest.mark.parametrize('f, xs', FUNCTIONS)
def test_gradient_is_the_jacobian_row(f, xs):
    assert gradient(f, xs) == jacobian(lambda *a: [f(*a)], xs)[0]


KINDS = {
    'outward': lambda lo, hi: O(lo, hi),
    'exact': lambda lo, hi: M(Fraction(lo), Fraction(hi)),
    'decorated': lambda lo, hi: D(O(lo, hi)),
    'decorated exact': lambda lo, hi: D(M(Fraction(lo), Fraction(hi))),
}


@settings(max_examples=60, deadline=None)
@given(expressions, st.sampled_from(sorted(KINDS)), boxes((-2, 2)), boxes((-2, 2)), st.sampled_from([None, 0, 1]))
def test_gradient_is_the_jacobian_row_on_trees(e, kind, bx, by, point):
    """the random trees of ::test_jacobian_encloses_the_partials over a drawn box of each kind, one
    coordinate sometimes a number: the gradient is the one-row jacobian's row, as sets and, decorated,
    in the decoration, of the same class"""
    f = e[0]
    xs = [KINDS[kind](lo, hi) for lo, hi, _ in (bx, by)]
    if point is not None:
        xs[point] = (bx, by)[point][2]  # a Fraction, a point of the box
    g, row = gradient(f, xs), jacobian(lambda *a: [f(*a)], xs)[0]
    assert len(g) == 2 and g == row, (e[2], xs, g, row)
    assert [type(d) for d in g] == [type(d) for d in row]


@pytest.mark.parametrize('f, x', [
    (lambda t: t ** 3 - 2 * t, M(-1, 2)),
    (lambda t: t.sqrt(), D(M(0, 1))),
    (lambda t: abs(t), D(O(-1, 1))),
    (lambda t: 1 / t, O(-1, 1)),
    (lambda t: 5, M(0, 1)),
    (lambda t: t.exp(), 2),
])
def test_one_variable_is_derivative(f, x):
    """the same single pass: equal as sets and, decorated, in the decoration"""
    assert gradient(f, [x]) == (derivative(f, x),)


def test_seeds():
    """each pass seeds only its own coordinate, and reads its own column"""
    assert jacobian(lambda x, y: (x, y), [M(1, 2), M(3, 4)]) == ((M(1), M(0)), (M(0), M(1)))
    assert jacobian(lambda x, y: (x * y,), [2, 3]) == ((M(3), M(2)),)
    assert jacobian(lambda x, y, z: (x * y * z, z - x), [2, 3, 5]) == ((M(15), M(10), M(6)), (M(-1), M(0), M(1)))
    assert jacobian(lambda x, y: (5, x), [M(0, 1), M(2, 3)]) == ((M(0), M(0)), (M(1), M(0)))  # a constant output
    assert gradient(lambda x, y: x * y, [M(1, 2), M(3, 4)]) == (M(3, 4), M(1, 2))


def test_decorated_columns():
    """dac or better in every column exactly where F is C¹ on the box: the solver's gate"""
    def F(x, y):
        return (x.sqrt() + y, abs(x - y))

    def decorations(xs):
        return [[d.decoration for d in row] for row in jacobian(F, xs)]
    # sqrt at 0: its factor 1 / (2 sqrt 0) is trv, and the min law carries it into the y column too
    assert decorations([D(M(0, 1)), D(M(2, 3))]) == [[Decoration.TRV] * 2, [Decoration.COM] * 2]
    # abs across 0: its derivative sign is def in both columns
    assert decorations([D(M(1, 2)), D(M(1, 2))])[1] == [Decoration.DEF] * 2
    assert decorations([D(M(1, 2)), D(M(-1, 0))]) == [[Decoration.COM] * 2] * 2


def test_arguments():
    # numbers are points; beside a decorated set, decorated points (of the first set's class)
    assert jacobian(lambda x, y: (x * y,), [2, M(3)]) == ((M(3), M(2)),)
    g = gradient(lambda x, y: x * y, [D(O(1, 2)), 3])
    assert g == (D(O(3)), D(O(1, 2))) and type(g[1].interval) is O
    assert type(gradient(lambda x, y: x * y, [O(1, 2), 3])[0]) is O
    # a bare set beside a decorated one is refused up front, even by an f that ignores it
    with pytest.raises(TypeError, match='not one of each'):
        gradient(lambda x, y: x, [D(M(1, 2)), M(3, 4)])
    with pytest.raises(TypeError, match='not one of each'):
        jacobian(lambda x, y: (x,), [M(1, 2), D(M(3, 4))])
    with pytest.raises(TypeError, match='not a Dual or a number'):
        gradient(lambda x, y: 'nope', [M(1, 2), M(3, 4)])
    with pytest.raises(TypeError, match='not a Dual or a number'):
        jacobian(lambda x, y: (x, 'nope'), [M(1, 2), M(3, 4)])
    with pytest.raises(TypeError, match='not a list or a tuple'):
        jacobian(lambda x, y: x, [M(1, 2), M(3, 4)])
    with pytest.raises(TypeError, match='a list or a tuple'):
        gradient(lambda x: x, M(1, 2))
    with pytest.raises(TypeError, match='a number'):
        gradient(lambda x, y: x, [M(1, 2), 'nope'])
    with pytest.raises(ValueError, match='at least one variable'):
        jacobian(lambda: (), [])
    with pytest.raises(ValueError, match='at least one variable'):
        gradient(lambda: 1, ())
    # a bool is not a number here, refused with gradient's own wording (review S5: MultiInterval's
    # constructor refuses it too, as 'expected a real number, got bool')
    with pytest.raises(TypeError, match='or a number, got bool'):
        gradient(lambda x, y: x, [True, M(1, 2)])
    # an F whose outputs change length from pass to pass (review S4, spec F2)
    with pytest.raises(ValueError, match='different lengths'):
        jacobian(lambda x, y: (x, y) if x.derivative == M(1) else (x,), [M(1, 2), M(3, 4)])
