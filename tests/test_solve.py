"""
intervals.solver's `solve`: square systems in several variables (M16, H3's second part)

the oracle is constructed systems with every real zero known: `F(x) = A · G(B x + c)` with A and B
invertible integer matrices, c integers and `G_i(u) = Π_j g_ij(u_i)`, each factor `u - z` (z an int,
a fraction or a float), `u ** 2 - q` (q squarefree, a different q in each coordinate) or, not C¹ at
its zero, `abs(u - z)` or `(u - z).cbrt()`. since A and B are invertible, `F(x) = 0` iff `B x + c` is
a point of the grid of the factors' zeros, so the zeros are `B⁻¹ (u - c)` over the grid: exact
fractions, or `r + Σ a_q sqrt(q)` decided with arb (retried at a higher precision, `assume(False)`
past the last, as `tests/test_autodiff.py::_check`). the properties:
* soundness: every zero in the box is in a returned `RootBox`, whatever `tol` and `max_steps`
* a `RootBox` marked unique holds exactly one of the zeros
* each component connected and inside the input's; the boxes pairwise disjoint and sorted
and by example: simple zeros proved unique (irrational ones by krawczyk, on the box or on its
inflation; simple rational ones as exact points), the first step's split, the C¹ gate shown
necessary, the private helpers one rule at a time, n == 1 as `newton`, and the evaluation budgets
"""
import math
from fractions import Fraction
from itertools import product

import pytest
from flint import arb
from flint import ctx
from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import REALS
from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import set_dec
from intervals import solver
from intervals.autodiff import Dual
from intervals.solver import RootBox
from intervals.solver import newton
from intervals.solver import solve
from tests.test_autodiff import _arb
from tests.test_autodiff import _inside

M = MultiInterval
O = OutwardMultiInterval
Q = Fraction
PRECISIONS = (200, 1000)
MAX = 1.7976931348623157e308


# THE ORACLE: CONSTRUCTED SYSTEMS

def _inverse(B):
    """B⁻¹ in Fractions (B invertible)"""
    n = len(B)
    A = [[Q(v) for v in row] + [Q(int(i == j)) for j in range(n)] for i, row in enumerate(B)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def _det(B):
    if len(B) == 1:
        return B[0][0]
    return sum((-1) ** j * B[0][j] * _det([row[:j] + row[j + 1:] for row in B[1:]]) for j in range(len(B)))


def _factor(kind, z):
    return {
        'lin': lambda u: u - z,
        'sq': lambda u: u ** 2 - z,
        'abs': lambda u: abs(u - z),
        'cbrt': lambda u: (u - z).cbrt(),
    }[kind]


def _roots(kind, z):
    """a factor's zeros, each (rational part, ((coefficient, q), ...)) for `r + Σ a sqrt(q)`"""
    if kind == 'sq':
        return [(Q(0), ((Q(s), z),)) for s in (1, -1)]
    return [(Q(z), ())]


def system(A, B, c, Z):
    """F and its zeros: `F(x) = A G(B x + c)`, `G_i` the product of the factors `Z[i]`"""
    n = len(B)

    def F(*x):
        u = [sum((B[i][j] * x[j] for j in range(n) if B[i][j]), 0) + c[i] for i in range(n)]
        g = []
        for i in range(n):
            y = 1
            for kind, z in Z[i]:
                y = _factor(kind, z)(u[i]) * y
            g.append(y)
        return tuple(sum((A[i][k] * g[k] for k in range(n) if A[i][k]), 0) for i in range(n))
    Binv = _inverse(B)
    zeros = set()
    for us in product(*[[r for kind, z in row for r in _roots(kind, z)] for row in Z]):
        point = []
        for k in range(n):
            rational = sum((Binv[k][j] * (us[j][0] - c[j]) for j in range(n)), Q(0))
            terms = {}
            for j in range(n):
                for a, q in us[j][1]:
                    terms[q] = terms.get(q, 0) + Binv[k][j] * a
            terms = tuple(sorted((a, q) for q, a in terms.items() if a))
            point.append(rational if not terms else (rational, terms))
        zeros.add(tuple(point))
    return F, zeros


def _member(v, s):
    """v in s: exact for a Fraction, by arb for `r + Σ a sqrt(q)`; None if arb cannot tell"""
    if isinstance(v, Q):
        return v in s
    rational, terms = v
    old = ctx.prec
    try:
        for prec in PRECISIONS:
            ctx.prec = prec
            ball = _arb(rational) + sum((_arb(a) * arb(q).sqrt() for a, q in terms), arb(0))
            verdict = _inside(s, ball)
            if verdict is not None:
                return verdict
    finally:
        ctx.prec = old
    return None


def _in_box(z, box):
    verdicts = [_member(v, s) for v, s in zip(z, box)]
    if False in verdicts:
        return False
    return None if None in verdicts else True


def _decided(verdict):
    if verdict is None:
        assume(False)  # arb could not tell at the last precision: reject the example
    return verdict


def check_roots(roots, xs, zeros):
    """every zero of the box xs is in a RootBox; a unique one holds exactly one zero; each component
    connected and inside xs; the boxes pairwise disjoint (in some coordinate) and sorted"""
    for z in zeros:
        if _decided(_in_box(z, xs)):
            assert any(_decided(_in_box(z, r.box)) for r in roots), ('lost', z, roots)
    for r in roots:
        assert len(r.box) == len(xs) and all(c.is_contiguous and c.issubset(x) for c, x in zip(r.box, xs)), r
        if r.unique:
            assert sum(_decided(_in_box(z, r.box)) for z in zeros) == 1, ('unique holds', r, zeros)
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            assert any(p.isdisjoint(q) for p, q in zip(a.box, b.box)), (a, b)
    keys = [tuple(c.sort_key for c in r.box) for r in roots]
    assert keys == sorted(keys), roots


matrices = st.lists(st.integers(-2, 2), min_size=4, max_size=4).map(lambda v: [v[:2], v[2:]]).filter(
    lambda m: _det(m) != 0)
rational_zeros = st.one_of(st.integers(-3, 3), st.fractions(-3, 3, max_denominator=7),
                           st.floats(-3, 3, allow_nan=False, allow_infinity=False))


def _factors(qs, not_c1, most):
    kinds = [rational_zeros.map(lambda z: ('lin', z)), st.sampled_from(qs).map(lambda q: ('sq', q))]
    if not_c1:
        kinds += [rational_zeros.map(lambda z: ('abs', z)), rational_zeros.map(lambda z: ('cbrt', z))]
    return st.lists(st.one_of(*kinds), min_size=1, max_size=most)


def systems(not_c1=False, most=2):
    """(A, B, c, Z) for n = 2, up to `most` factors per coordinate; the square roots of coordinate 0
    and 1 apart (2, 3 against 5, 7)"""
    return st.tuples(matrices, matrices, st.lists(st.integers(-1, 1), min_size=2, max_size=2),
                     st.tuples(_factors((2, 3), not_c1, most), _factors((5, 7), not_c1, most)))


BOXES = [
    (M(-6, 6), M(-6, 6)),
    (M(-6, 6, start_closed=False, end_closed=False), M(-6, 6, end_closed=False)),
    (M(-6, -1) | M(1, 6), M(-6, 6)),
    (M(-2, 3), M(-3, 2)),  # integer faces, where zeros sit
]
SYSTEM_3 = ([[-2, 1], [-1, 1]], [[1, -1], [0, 2]], [1, 1], ([('lin', 2), ('lin', Q(-11, 5))], [('lin', 2), ('lin', -3)]))
SYSTEM_7 = ([[-2, 2], [-1, 0]], [[0, -2], [-1, -1]], [0, 1], ([('lin', Q(10, 3))], [('lin', Q(-1, 4)), ('lin', Q(-20, 3))]))
IDENTITY = [[1, 0], [0, 1]]


@settings(max_examples=10, deadline=None)
@given(systems(most=1), st.sampled_from(BOXES))
@example(SYSTEM_7, BOXES[0])   # a rational zero, (35/12, -5/3)
@example(SYSTEM_3, BOXES[0])   # four zeros on split faces: (-26/5, -2), (-27/10, 1/2), (-1, -2), (3/2, 1/2)
@example(SYSTEM_3, (M(-1, 2), M(-2, Q(1, 2))))   # zeros on the input's faces and corners
@example(([[1, 1], [1, -1]], IDENTITY, [0, 0], ([('lin', 1), ('lin', 1)], [('lin', 0)])), BOXES[0])  # a double zero
@example(([[1, 1], [1, -1]], IDENTITY, [0, 0], ([('lin', 1), ('lin', 1 + Q(1, 10 ** 12))], [('lin', 0)])), BOXES[0])  # two close zeros
@example(([[1, 0], [1, 1]], [[1, 1], [1, -1]], [0, 1], ([('sq', 2)], [('sq', 5), ('lin', 1)])), BOXES[0])  # irrational zeros
def test_every_zero_is_enclosed(params, xs):
    """one factor per coordinate when drawn (two in the examples), and max_steps=2000: soundness and
    the truth of the unique claims hold under any budget, and systems of two factors per coordinate
    have a tail of costly ones (4677 calls, 201 s, on 2026-09-28)"""
    F, zeros = system(*params)
    check_roots(solve(F, xs, max_steps=2000), xs, zeros)


@settings(max_examples=40, deadline=None)
@given(systems(not_c1=True), st.sampled_from(BOXES), st.integers(1, 30), st.sampled_from([1e-3, 1e-10, 0]))
@example(([[1, 0], [0, 1]], [[1, 1], [1, -1]], [0, 0], ([('abs', Q(1, 3))], [('cbrt', Q(1, 2)), ('lin', -1)])),
         BOXES[0], 30, 1e-3)
def test_every_zero_is_enclosed_on_a_budget(params, xs, max_steps, tol):
    """with factors not C¹ at their zeros too (`abs`, `cbrt`), which the gate keeps the step off:
    unbudgeted, they are bisected to tol, 500 to 1500 calls a system (2026-09-28), so they are drawn
    only here"""
    F, zeros = system(*params)
    check_roots(solve(F, xs, tol=tol, max_steps=max_steps), xs, zeros)


def test_three_variables():
    """one zero, (169/105, 134/105, -92/105), proved by krawczyk. (a constructed n = 3 system with
    two zeros took more than 120 s on 2026-09-28: the range of `A G(B x + c)` over a box of three
    wide components is loose, and a box costs 5 calls; the sphere is n = 3 with two zeros, above)"""
    A = [[1, 0, 1], [0, 2, -1], [1, 1, 1]]
    B = [[1, -1, 0], [0, 1, 1], [1, 0, 2]]
    Z = ([('lin', Q(1, 3))], [('lin', Q(2, 5))], [('lin', Q(-1, 7))])
    F, zeros = system(A, B, [0, 0, 0], Z)
    xs = (M(-6, 6),) * 3
    roots = solve(F, xs)
    check_roots(roots, xs, zeros)
    assert len(roots) == len(zeros) == 1 and roots[0].unique


# SIMPLE ZEROS, PROVED

S = 2 ** -0.5
IRRATIONAL = {
    'sqrt 2 on the diagonal': (lambda x, y: (x ** 2 - 2, y - x), [M(-10, 10)] * 2, [(-2 ** 0.5, -2 ** 0.5), (2 ** 0.5, 2 ** 0.5)]),
    'circle and line': (lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [M(-10, 10)] * 2, [(-S, -S), (S, S)]),
    'circle and line, the reals': (lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [REALS] * 2, [(-S, -S), (S, S)]),
    'circle and line, 1e300': (lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [M(-1e300, 1e300)] * 2, [(-S, -S), (S, S)]),
    'sphere': (lambda x, y, z: (x ** 2 + y ** 2 + z ** 2 - 1, x - y, y - 2 * z), [M(-3, 3)] * 3,
               [(-2 / 3, -2 / 3, -1 / 3), (2 / 3, 2 / 3, 1 / 3)]),
    'exp': (lambda x, y: (x.exp() - y, x + y - 2), [M(-5, 5)] * 2,
            [(0.44285440100238858, 2 - 0.44285440100238858)]),   # x = 2 - W(e ** 2), arb 2026-09-28
    'sin': (lambda x, y: ((x + y).sin(), x - 2 * y), [M(-4, 4)] * 2,
            [(-2 * math.pi / 3, -math.pi / 3), (0, 0), (2 * math.pi / 3, math.pi / 3)]),
    # one row pins a coordinate: gauss-seidel makes it degenerate or an ulp wide at once, so krawczyk
    # on the box can never prove it; the inflation, clipped to the box's region, does (critique B1)
    'pinned 1/4': (lambda x, y: (x ** 2 - 2, y - Q(1, 4)), [M(-10, 10)] * 2, [(-2 ** 0.5, 0.25), (2 ** 0.5, 0.25)]),
    'pinned 1/3': (lambda x, y: (x ** 2 - 2, 3 * y - 1), [M(-10, 10)] * 2, [(-2 ** 0.5, 1 / 3), (2 ** 0.5, 1 / 3)]),
    'pinned 0.1': (lambda x, y: (x ** 2 - 2, y - 0.1), [M(-10, 10)] * 2, [(-2 ** 0.5, 0.1), (2 ** 0.5, 0.1)]),
    'pinned circle': (lambda x, y: (x ** 2 + y ** 2 - 1, y - Q(1, 4)), [M(-3, 3)] * 2,
                      [(-15 ** 0.5 / 4, 0.25), (15 ** 0.5 / 4, 0.25)]),
    # the first step's row 0 pins x to [1/4], then its row 1 splits y at the gap around 0: the pieces'
    # regions must keep x wide, or the inflation cannot prove them
    'pinned, then split': (lambda x, y: (x - Q(1, 4), x ** 2 + y ** 2 - 4), [M(-3, 3)] * 2,
                           [(0.25, -63 ** 0.5 / 4), (0.25, 63 ** 0.5 / 4)]),
}


@pytest.mark.parametrize('name', IRRATIONAL)
def test_simple_zeros_are_proved_unique(name):
    """each zero in one unique RootBox, less than 1e-12 wide and within 1e-12 of it"""
    F, xs, zeros = IRRATIONAL[name]
    roots = solve(F, xs)
    assert len(roots) == len(zeros) and all(r.unique for r in roots), roots
    for r, z in zip(roots, zeros):
        for c, v in zip(r.box, z):
            assert c.wid() < 1e-12 and abs(float(c.mid()) - v) < 1e-12, (name, r, z)


EXACT = {
    'cusp': (lambda x, y: (x ** 2 - y, y - x ** 3), [M(-2, 2)] * 2, [(0, 0), (1, 1)]),
    'pole': (lambda x, y: (1 / x - y, x - y), [M(-2, 2)] * 2, [(-1, -1), (1, 1)]),
    'kink': (lambda x, y: (abs(x) + x / 2 - y, y - Q(1, 4)), [M(-1, 3), M(-1, 1)], [(Q(-1, 2), Q(1, 4)), (Q(1, 6), Q(1, 4))]),
    'sin at 0': (lambda x, y: ((x + y).sin(), x - 2 * y), [M(-4, 4)] * 2, [(0, 0)]),   # and two irrational ones
}


@pytest.mark.parametrize('name', EXACT)
def test_simple_rational_zeros_are_exact_points(name):
    """a zero at a simple rational, on a split face or where a coordinate became degenerate: output
    as the exact point, unique, by the simplest point of its converged box"""
    F, xs, zeros = EXACT[name]
    roots = solve(F, xs)
    if name != 'sin at 0':   # every zero rational
        check_roots(roots, xs, {tuple(Q(v) for v in z) for z in zeros})
    for z in zeros:
        assert RootBox(tuple(O(v) for v in z), True) in roots, (name, z, roots)


def test_the_first_step_splits_the_box():
    """J's (0, 0) entry over [-10, 10]² is [-20, 20]: gauss-seidel's row 0 is two pieces, so one
    step splits the box in two, no bisection (`newton`'s first step, one dimension up)"""
    roots = solve(lambda x, y: (x ** 2 - 2, y - x), [M(-10, 10)] * 2, max_steps=1)
    assert roots == (RootBox((O.parse('[-10, -0.09999999999999999)'), O(-10, 10)), False),
                     RootBox((O.parse('(0.09999999999999999, 10]'), O(-10, 10)), False))


# THE C¹ GATE

KINK = (lambda x, y: (abs(x) + x / 2 - y, y - Q(1, 4)), [M(-1, 3), M(-1, 1)], (Q(-1, 2), Q(1, 4)))
COUPLED = (lambda x, y: (abs(x) + y / 2 - Q(1, 4), y - x), [M(-1, 3)] * 2, (Q(-1, 2), Q(-1, 2)))


@pytest.mark.parametrize('F, xs, zero', [KINK, COUPLED])
def test_not_c1_is_not_stepped(F, xs, zero):
    """abs's derivative is def across 0: the gate keeps the step off, and both zeros are proved"""
    roots = solve(F, xs)
    assert len(roots) == 2 and all(r.unique for r in roots), roots
    assert any(all(v in c for v, c in zip(zero, r.box)) for r in roots)


@pytest.mark.parametrize('F, xs, zero', [KINK, COUPLED])
def test_not_c1_would_lose_a_zero(monkeypatch, F, xs, zero):
    """the sabotage the test above guards: the step run on a box where F is not C¹"""
    def smooth(F, box):
        return original(F, box)[0], True
    original = solver._jacobian
    monkeypatch.setattr(solver, '_jacobian', smooth)
    roots = solve(F, xs)
    assert not any(all(v in c for v, c in zip(zero, r.box)) for r in roots), roots


def test_a_jump_is_caught_by_the_value_decoration():
    """sign(x) with its derivative 0, a true enclosure of the derivative wherever there is one: only
    the value's decoration (def across the jump) says F is not C¹. `newton`'s test, one dimension up"""
    def step(x):
        return Dual(x.value.sign(), x.derivative * 0) if isinstance(x, Dual) else x.sign()
    roots = solve(lambda x, y: (step(x), y), [M(-1, 1)] * 2)
    assert RootBox((O(0), O(0)), True) in roots, roots


def test_c1_is_decided_on_the_closed_hull():
    """krawczyk's theorem is for a compact box: F must be C¹ on the closed hull, which holds a pole
    at an open end"""
    def F(x, y):
        return (1 / x, y)
    assert not solver._jacobian(F, (O(0, 1, start_closed=False), O(0, 1)))[1]
    J, smooth = solver._jacobian(F, (O(Q(1, 2), 1), O(0, 1)))
    assert smooth and J == ((O(-4, -1), O(0)), (O(0), O(1)))


# THE HELPERS, ONE RULE AT A TIME

I2 = ((O(1), O(0)), (O(0), O(1)))


@pytest.mark.filterwarnings('ignore::intervals.errors.EmptySetPropagationWarning')  # the empty K
def test_krawczyk_proves_only_inside_the_interior():
    """with Mx = I the K set is `m - b`, the zero of a linear F, so each condition shows alone"""
    m = (Q(1, 2), Q(1, 2))
    box = (O(0, 1), O(0, 1))
    assert solver._krawczyk(box, I2, (O(Q(1, 6)), O(Q(1, 4))), m)   # the zero (1/3, 1/4): inside
    assert not solver._krawczyk(box, I2, (O(Q(-1, 2)), O(Q(1, 4))), m)   # (1, 1/4): on the face
    assert not solver._krawczyk((O(0, 1), O(Q(1, 2))), I2, (O(Q(1, 6)), O(0)), m)  # a degenerate component
    assert not solver._krawczyk(box, I2, (O(), O(Q(1, 4))), m)   # an empty K
    # K on the closed hull: with Mx[0][0] = 0 the set K[0] is the box's component itself, which is
    # inside its own interior only if its open ends are kept
    open_box = (O(0, 1, start_closed=False, end_closed=False), O(0, 1))
    assert not solver._krawczyk(open_box, ((O(0), O(0)), (O(0), O(1))), (O(0), O(0)), m)
    # m off the midpoint, so m - b and m + b are not mirror images: the sign of b shows (review S1)
    q = (Q(1, 4), Q(1, 4))
    assert solver._krawczyk(box, I2, (O(Q(-1, 2)), O(0)), q)       # the zero (3/4, 1/4): inside
    assert not solver._krawczyk(box, I2, (O(Q(1, 2)), O(0)), q)    # the zero (-1/4, 1/4): outside


def test_gauss_seidel_step():
    box = (O(-3, 3), O(-3, 3))
    m = (0, 0)
    # row 0 narrows x to [1]; row 1's diagonal holds 0: two boxes, x as narrowed so far
    Mx = ((O(1), O(0)), (O(0), O(-1, 1)))
    assert solver._gauss_seidel(box, Mx, (O(-1), O(1)), m) == ((O(1), O(-3, -1)), (O(1), O(1, 3)))
    # an empty row drops the box: the step is intersected with the box
    assert solver._gauss_seidel(box, I2, (O(10), O(0)), m) == ()
    # mul_rev: [0] * t = [0] holds for every t, where [0] / [0] is empty (D7)
    Mx = ((O(0), O(0)), (O(0), O(1)))
    assert solver._gauss_seidel(box, Mx, (O(0), O(-1)), m) == ((O(-3, 3), O(1)),)
    # gauss-seidel, not jacobi: row 1 uses row 0's narrowing, x = [1] and y = -x
    Mx = ((O(1), O(0)), (O(1), O(1)))
    assert solver._gauss_seidel(box, Mx, (O(-1), O(0)), m) == ((O(1), O(-1)),)


def test_inverse():
    """float gauss-jordan with partial pivoting; None where it cannot"""
    assert solver._inverse([[0.0, 1.0], [1.0, 0.0]]) == [[0.0, 1.0], [1.0, 0.0]]   # needs the row swap
    assert solver._inverse([[2.0, 0.0], [0.0, 4.0]]) == [[0.5, 0.0], [0.0, 0.25]]
    assert solver._inverse([[1.0, 2.0], [2.0, 4.0]]) is None
    assert solver._inverse([[math.inf, 0.0], [0.0, 1.0]]) is None


def test_precondition_falls_back_to_the_identity():
    """a midpoint beyond the doubles (an exact int end) cannot be a float: Y is the identity, which
    keeps the step valid, only weaker"""
    J = ((O(10 ** 400, 10 ** 401), O(0)), (O(0), O(1)))
    Mx, b = solver._precondition(J, (O(1), O(2)))
    assert Mx == J and b == (O(1), O(2))
    Mx, b = solver._precondition(((O(2), O(0)), (O(0), O(4))), (O(1), O(2)))
    assert Mx == I2 and b == (O(0.5), O(0.5))
    # an exact 0 of Y skips its term: 0 * [inf] is empty (D2), the real product 0 (review S6)
    assert solver._combine((0.0, 1.0), (O(math.inf), O(2))) == O(2)


def test_simplest_between():
    assert solver._simplest_between(Q(0.999), Q(1.001)) == 1
    assert solver._simplest_between(Q(-1), Q(1)) == 0
    assert solver._simplest_between(Q(-3, 10), Q(-2, 10)) == Q(-1, 4)
    assert solver._simplest_between(Q(314, 100), Q(315, 100)) == Q(22, 7)
    assert solver._simplest_between(Q(2), Q(3)) == 2
    assert solver._simplest_between(Q(1, 3), Q(1, 3)) == Q(1, 3)


def test_the_simplest_point_needs_an_exact_zero():
    """sin(x) - sin(x) + 1e-300 holds 0 by its enclosure, but is not 0: on a box whose simplest
    point is (1, 0), no unique point is claimed (critique B3: a box, since a point is step 2)"""
    def F(x, y):
        return (x.sin() - x.sin() + 1e-300, y)
    roots = solve(F, [M(0.999, 1.001), M(-0.001, 0.001)], tol=0.01)
    assert roots and not any(r.unique for r in roots), roots


def test_a_point_is_unique_only_when_f_is_exactly_zero():
    def F(x, y):
        return (x.sin() - x.sin() + 1e-300, y)
    assert solve(F, [1, 0]) == (RootBox((O(1), O(0)), False),)
    assert solve(lambda x, y: (x - 1, y), [1, 0]) == (RootBox((O(1), O(0)), True),)


def test_inflation_is_clipped_to_the_region():
    """a box with no zero, beside the zero (1/2, 1/2) of another region: the inflated box would hold
    that zero, and krawczyk would prove it; clipped to the box's own region it cannot"""
    def F(x, y):
        return (x - Q(1, 2), y - Q(1, 2))
    box = (O(0.5 + 1e-13, 0.5 + 2e-13), O(0.5 - 1e-13, 0.5 + 1e-13))
    H = solver._inflate(box, box)
    assert H == tuple(c.closed_hull for c in box)
    assert not solver._inflated_unique(F, box, box)
    region = (O(0.5 + 1e-13, 1), O(0, 1))
    H = solver._inflate(box, region)
    assert all(b.issubset(h) and h.issubset(r.closed_hull) for b, h, r in zip(box, H, region)), H
    assert H[0].inf == 0.5 + 1e-13 and H[1].inf < 0.5 - 1e-13
    # the region holding the zero: proved (the box holding it too)
    box = (O(0.5 - 1e-13, 0.5 + 1e-13), O(0.5))
    assert solver._inflated_unique(F, box, (O(0, 1), O(0, 1)))
    assert not solver._inflated_unique(F, box, (O(0, 1), O(0.5)))   # a degenerate region: no interior


def test_inflation_takes_its_own_jacobian():
    """J and the C¹ gate over the inflated box H'', not over the box: a spy F records the
    decorated passes"""
    seen = []

    def F(x, y):
        if isinstance(x, Dual):
            seen.append((x.value.interval, y.value.interval))
        return (x ** 2 - 2, y - Q(1, 4))
    box = (O(1.414213562373095, 1.4142135623730951), O(Q(1, 4)))
    region = (O(1, 2), O(0, 1))
    assert solver._inflated_unique(F, box, region)
    H = solver._inflate(box, region)
    assert seen == [H, H]


def test_inflation_needs_the_c1_gate():
    """F's values and partials those of a linear F, its decorations trv: nothing is proved"""
    def F(x, y):
        u, v = x - Q(1, 2), y - Q(1, 2)
        if isinstance(x, Dual):
            u = Dual(set_dec(u.value.interval, Decoration.TRV), u.derivative)
        return (u, v)
    box = (O(0.5 - 1e-13, 0.5 + 1e-13), O(0.5))
    assert not solver._inflated_unique(F, box, (O(0, 1), O(0, 1)))


def test_a_split_box_is_unproved(monkeypatch):
    """a proof is for the box, not for each piece of a split (the piece without the zero would
    claim it). a box proved in an earlier step and split later is rare, so krawczyk is made to claim
    the first box, which the next step splits: with max_steps=1 both pieces are output unproved"""
    calls = []

    def once(*args):
        calls.append(args)
        return len(calls) == 1
    monkeypatch.setattr(solver, '_krawczyk', once)
    roots = solve(lambda x, y: (x ** 2 - 2, y - x), [M(-10, 10)] * 2, max_steps=1)
    assert len(roots) == 2 and not any(r.unique for r in roots), roots


def test_a_bisected_box_is_unproved(monkeypatch):
    """a proof is for the box, not for each half of a bisection (review S3). krawczyk is made to
    claim the first box and gauss-seidel to narrow it to [0.5, 20] x [-1, 1], wide in x, so the next
    pop bisects it; with max_steps=2 both halves are output, unproved"""
    krawczyk, gauss_seidel = solver._krawczyk, solver._gauss_seidel
    first = {'krawczyk': True, 'gauss_seidel': True}

    def claim_once(*args):
        if first['krawczyk']:
            first['krawczyk'] = False
            return True
        return krawczyk(*args)

    def widen_once(box, *args):
        if first['gauss_seidel']:
            first['gauss_seidel'] = False
            return ((O(0.5, 20), box[1]),)
        return gauss_seidel(box, *args)
    monkeypatch.setattr(solver, '_krawczyk', claim_once)
    monkeypatch.setattr(solver, '_gauss_seidel', widen_once)
    roots = solve(lambda x, y: (x - 1, y), [M(-2, 20), M(-1, 1)], max_steps=2)
    assert len(roots) == 2 and not any(r.unique for r in roots), roots
    assert roots[0].box[0] | roots[1].box[0] == O(0.5, 20) and not roots[0].box[0] & roots[1].box[0], roots


def test_the_simplest_points_rest_is_its_own_region():
    """the rest of a box around an exact zero p goes back with regions that exclude p (every zero
    of a region is in its box), each region holding its box, so no inflation can reach p"""
    def F(x, y):
        return (x - Q(1, 2), y - Q(1, 2))
    box = (O(0.25, 0.75), O(0.25, 0.75))
    region = (O(0, 1), O(0, 1))
    roots, work = [], []
    solver._finish_box(F, box, False, region, roots, work, 0)
    assert roots == [RootBox((O(Q(1, 2)), O(Q(1, 2))), True)]
    assert len(work) == 4
    # each rest box's closed hull holds p, the simplest of a larger set, so p is its simplest point,
    # which it does not hold: no second point is drawn from a rest box (review F3)
    assert all(solver._simplest_point(rest) is None for rest, *_ in work)
    for rest, unique, past, turn, r in work:
        assert not unique and all(c.issubset(h) for c, h in zip(rest, r)), (rest, r)
        assert not all(Q(1, 2) in h for h in r), r
    assert not any(solver._inflated_unique(F, rest, r) for rest, _, _, _, r in work)


def test_choose_falls_through_to_the_other_components():
    """x = (MAX, inf] cannot be split, y can: y is bisected to tol, not left whole (critique N1);
    the simplest point skips the unbounded x (N2)"""
    roots = solve(lambda x, y: (1 / x, y - 0.5), [M(MAX, math.inf, start_closed=False), M(-1, 1)], tol=1e-3)
    assert roots and all(r.box[1].wid() <= 1e-3 and 0.5 in r.box[1] for r in roots), roots
    assert not any(r.unique for r in roots), roots   # 1 / x has no zero: an unbounded box is never proved (review S2)


# N == 1, THE INPUT, THE OUTPUT

@pytest.mark.parametrize('f, x', [
    (lambda x: x ** 2 - 2, M(-10, 10)),
    (lambda x: x.sin(), M(-10, 10)),
    (lambda x: abs(x) + x / 2 - Q(1, 4), M(-1, 3)),
    (lambda x: 1 / x - x, M(-2, 2)),
])
def test_n_equals_one_is_newton(f, x):
    """box for box, and call for call (the n-dimensional loop at n == 1 gives the same boxes on
    these, at 1.4x to 2x the calls: 49 against 32, 145 against 93, 32 against 16 and 47 against 33,
    in this order, 2026-09-28)"""
    F, g = Counted(lambda t: [f(t)]), Counted(f)
    assert solve(F, [x]) == tuple(RootBox((r.interval,), r.unique) for r in newton(g, x))
    assert F.calls == g.calls


def test_multi_piece_input():
    """the work list is the product of the pieces"""
    x = M(-5, -1) | M(1, 5)
    roots = solve(lambda x, y: (x ** 2 - 2, y - x), [x, M(-5, 5) | M(10, 11)])
    assert len(roots) == 2 and all(r.unique and r.box[0].issubset(x) for r in roots), roots
    assert solve(lambda x, y: (x ** 2 - 2, y), [M(-1, 1) | M(2, 3), M(-1, 1)]) == ()


def test_unbounded_input():
    roots = solve(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [REALS, REALS])
    assert [r.unique for r in roots] == [True, True]
    assert [round(float(r.box[0].mid()), 12) for r in roots] == [round(-S, 12), round(S, 12)]


def test_constant_and_continuum_systems():
    """F = (0, 0) and the line (x - y, 2x - 2y): a continuum of zeros, so only tol stops the
    bisection; every zero stays enclosed, and a unique box is a point. F = (1, x): none"""
    assert solve(lambda x, y: (1, x), [M(0, 1)] * 2) == ()
    grid = [Q(k, 7) for k in range(8)]
    for F, zeros in [(lambda x, y: (0, 0), list(product(grid, grid))),
                     (lambda x, y: (x - y, 2 * x - 2 * y), [(t, t) for t in grid])]:
        roots = solve(F, [M(0, 1)] * 2, tol=0.25)
        for z in zeros:
            assert any(all(v in c for v, c in zip(z, r.box)) for r in roots), z
        assert all(max(c.wid() for c in r.box) <= 0.25 for r in roots)
        assert all(all(c.is_degenerate for c in r.box) for r in roots if r.unique)


def test_overflow_box(monkeypatch):
    """exp over [700, 720] overflows to an open end at inf, dac: the step runs with an infinite end
    in J (a spy on `_precondition` sees one, review spec F5), and the zero (709.5, e ** 709.5)
    stays enclosed (critique B4)"""
    precondition, unbounded = solver._precondition, []

    def spy(J, fm):
        unbounded.append(any(not e.is_finite for row in J for e in row))
        return precondition(J, fm)
    monkeypatch.setattr(solver, '_precondition', spy)
    xs = [M(700, 720), M(1e307, 1.7e308)]
    roots = solve(lambda x, y: (x.exp() - y, x - 709.5), xs)
    assert any(unbounded), unbounded
    old = ctx.prec
    ctx.prec = 200
    try:
        assert any(709.5 in r.box[0] and _inside(r.box[1], arb(709.5).exp()) for r in roots), roots
    finally:
        ctx.prec = old


def test_degenerate_input_component():
    """y = [1/4] given as a number: the zeros (±sqrt 2, 1/4) enclosed, never unique (the region has
    no interior in y); an exact rational zero is still a unique point"""
    F = (lambda x, y: (x ** 2 - 2, y - Q(1, 4)))
    roots = solve(F, [M(-10, 10), Q(1, 4)])
    assert len(roots) == 2 and not any(r.unique for r in roots), roots
    assert all(abs(abs(float(r.box[0].mid())) - 2 ** 0.5) < 1e-12 for r in roots)
    roots = solve(lambda x, y: (x - Q(1, 2), y - Q(1, 4)), [M(-10, 10), Q(1, 4)])
    assert RootBox((O(Q(1, 2)), O(Q(1, 4))), True) in roots


def test_ends_beyond_the_doubles():
    """exact int ends past the doubles: the point of a component is its exact midpoint, since no
    float is inside (review F2: `_point_in`'s float(mid) raised OverflowError, in `newton` too).
    n == 2 on a budget: the float preconditioner overflows `b` there, so the step is idle and the
    unbudgeted solve bisects for 13669 calls (the record, M16a)"""
    big = M(10 ** 400, 10 ** 401)
    z = 3 * 10 ** 400
    roots = solve(lambda x, y: (x - z, y - x), [big, big], max_steps=10)
    assert any(z in r.box[0] and z in r.box[1] for r in roots), roots
    assert solve(lambda x: [x - z], [big]) == (RootBox((O(z),), True),)


def test_arguments_are_checked():
    F = lambda x, y: (x, y)  # noqa: E731
    # review S5: solve's own wording (MultiInterval's constructor refuses a bool too, in its own)
    with pytest.raises(TypeError, match='or a number, got bool'):
        solve(F, [True, M(0, 1)])
    with pytest.raises(TypeError, match='MultiInterval or a number'):
        solve(F, [DecoratedInterval(M(0, 1)), M(0, 1)])
    with pytest.raises(TypeError, match='MultiInterval or a number'):
        solve(F, [M(0, 1), 'nope'])
    with pytest.raises(TypeError, match='a list or a tuple'):
        solve(F, M(0, 1))
    with pytest.raises(ValueError, match='at least one variable'):
        solve(lambda: (), [])
    # one wording for a wrong-length F, at n == 1 (newton's path) and n == 2
    with pytest.raises(TypeError, match='not a list or a tuple of 2'):
        solve(lambda x, y: (x,), [M(0, 1), M(0, 1)])
    with pytest.raises(TypeError, match='not a list or a tuple of 1'):
        solve(lambda x: (x, x), [M(0, 1)])
    with pytest.raises(TypeError, match='not a list or a tuple of 1'):
        solve(lambda x: x, [M(0, 1)])
    with pytest.raises(TypeError, match='returned str'):
        solve(lambda x, y: (x, 'nope'), [M(0, 1), M(0, 1)])
    with pytest.raises(TypeError, match='returned str'):
        solve(lambda x: ['nope'], [M(0, 1)])


def test_warnings_stay_inside(recwarn):
    solve(lambda x, y: (1 / x - 1, y), [M(0, 2), M(-1, 1)])
    assert not recwarn.list


def test_results_are_outward():
    roots = solve(lambda x, y: (x ** 2 - 2, y - x), [M(0, 2), M(0, 2)])
    assert roots and all(type(c) is O for r in roots for c in r.box)


# EVALUATION BUDGETS

class Counted:
    """F, counting its calls: the plain, the decorated and the point calls all cost"""

    def __init__(self, F):
        self.F, self.calls = F, 0

    def __call__(self, *args):
        self.calls += 1
        return self.F(*args)


def _calls(F, xs, **kwargs):
    counted = Counted(F)
    solve(counted, xs, **kwargs)
    return counted.calls


def test_evaluation_budgets():
    """the costs the termination and splitting rules keep down, each bound about 1.5x the count
    measured on 2026-09-28 (the record, M16a, has the counts and what each bound catches)"""
    F7, _ = system(*SYSTEM_7)
    assert _calls(lambda x, y: (x + y, x - y), [REALS] * 2) <= 30
    assert _calls(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [REALS] * 2) <= 110
    assert _calls(lambda x, y: (x ** 2 + y ** 2 - 1, x - y), [M(-1e300, 1e300)] * 2) <= 160
    assert _calls(lambda x, y: (x ** 2 - y, y - x ** 3), [M(-2, 2)] * 2) <= 560
    assert _calls(F7, [M(-6, 6)] * 2) <= 50
