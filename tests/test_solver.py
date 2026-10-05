"""
multiinterval.solver: interval newton over multi-intervals

the properties, on polynomials built from drawn roots (so every real zero is known, exactly):
* soundness: every zero in x is in a returned `Root`, whatever `tol` and `max_steps`
* a `Root` marked unique holds exactly one of the distinct zeros
* the roots are disjoint, sorted, and inside x
and by example: a simple zero is proved unique; the first newton step splits a piece in two where
the derivative's set holds 0; the step is skipped where f is not C¹ (a function whose derivative's
values would lose a zero there, `abs(x) + x / 2 - 1/4`); close zeros are not claimed unique; poles,
unbounded input and multi-piece input
"""
import math
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval import REALS
from multiinterval import solver
from multiinterval.autodiff import Dual
from multiinterval.solver import Root
from multiinterval.solver import newton

M = MultiInterval
PI = math.pi


def _polynomial(zeros):
    def f(x):
        y = 1
        for z in zeros:
            y = (x - z) * y
        return y
    return f


def _check_roots(roots, x, zeros):
    """every zero of x is in a Root; each unique Root holds exactly one distinct zero; the Roots are
    disjoint, in order and in x"""
    for z in set(zeros):
        if z in x:
            assert any(z in r.interval for r in roots), (z, roots)
    for r in roots:
        assert r.interval.is_contiguous and r.interval.issubset(x), r
        if r.unique:
            assert sum(z in r.interval for z in set(zeros)) == 1, r
    for a, b in zip(roots, roots[1:]):
        assert a.interval.isdisjoint(b.interval) and a.interval.before(b.interval), (a, b)


zeros_strategy = st.lists(
    st.one_of(st.integers(-5, 5), st.fractions(-5, 5, max_denominator=7),
              st.floats(-5, 5, allow_nan=False, allow_infinity=False)),
    min_size=1, max_size=4)


@settings(max_examples=40, deadline=None)
@given(zeros_strategy, st.sampled_from([M(-6, 6), M(-2, 3), M(0, 1, start_closed=False), M(-6, -1) | M(1, 6)]))
@example([1, 1], M(-6, 6))                       # a double zero
@example([1, 1 + Fraction(1, 10 ** 12)], M(-6, 6))  # two zeros closer than tol
@example([0, 1, -1], M(-6, 6))                   # zeros at the split points
@example([0, 0.0], M(-6, 6))                     # a double zero at 0: newton is linear there
def test_every_zero_is_enclosed(zeros, x):
    _check_roots(newton(_polynomial(zeros), x), x, zeros)


@settings(max_examples=60, deadline=None)
@given(zeros_strategy, st.integers(1, 30), st.sampled_from([1e-3, 1e-10, 0]))
def test_every_zero_is_enclosed_on_a_budget(zeros, max_steps, tol):
    x = M(-6, 6)
    _check_roots(newton(_polynomial(zeros), x, tol=tol, max_steps=max_steps), x, zeros)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.integers(-8, 8), min_size=1, max_size=3, unique=True))
def test_simple_irrational_zeros_are_proved_unique(ns):
    """zeros n + 1/sqrt 2, apart by at least 1 and at no split point: each is one unique Root"""
    c = 2 ** -0.5
    zeros = [n + c for n in ns]
    roots = newton(lambda x: _polynomial(ns)(x - c), M(-10, 10))
    assert len(roots) == len(zeros) and all(r.unique for r in roots), roots
    for r, z in zip(roots, sorted(zeros)):
        assert r.interval.wid() < 1e-12 and abs(r.interval.mid() - z) < 1e-12, (r, z)


def test_sin_zeros():
    roots = newton(lambda x: x.sin(), M(-10, 10))
    assert [r.unique for r in roots] == [True] * 7
    for r, k in zip(roots, range(-3, 4)):
        assert r.interval.wid() <= 2e-15 and abs(r.interval.mid() - k * PI) < 1e-14


def test_the_first_step_splits_the_piece():
    """F'([-10, 10]) = [-20, 20] holds 0, so the step is two pieces: one step, no bisection. an
    interval type would get the hull, the whole piece, and learn nothing"""
    roots = newton(lambda x: x ** 2 - 2, M(-10, 10), max_steps=1)
    assert roots == (Root(OutwardMultiInterval.parse('[-10, -0.09999999999999999)'), False),
                     Root(OutwardMultiInterval.parse('(0.09999999999999999, 10]'), False))


def test_not_c1_is_bisected_not_stepped():
    """abs(x) + x/2 - 1/4 has zeros -1/2 and 1/6 and a kink at 0. on [-1, 3] the midpoint is 1, and
    the derivative's values over the piece are {-1/2, 1/2, 3/2}; a newton step with them gives
    1 - (5/4) / {-1/2, 1/2, 3/2} = {7/2, -3/2, 1/6}, losing -1/2. the decorations say abs's derivative
    is def across 0, so the step does not run there"""
    roots = newton(lambda x: abs(x) + x / 2 - Fraction(1, 4), M(-1, 3))
    _check_roots(roots, M(-1, 3), [Fraction(-1, 2), Fraction(1, 6)])
    assert [r.unique for r in roots] == [True, True]


def test_not_c1_would_lose_a_zero(monkeypatch):
    """the sabotage the test above guards: the step run on a piece where f is not C¹"""
    def smooth(f, piece):
        value, slope, _ = original(f, piece)
        return value, slope, True
    original = solver._evaluate
    monkeypatch.setattr(solver, '_evaluate', smooth)
    roots = newton(lambda x: abs(x) + x / 2 - Fraction(1, 4), M(-1, 3))
    assert not any(Fraction(-1, 2) in r.interval for r in roots)


@pytest.mark.filterwarnings('ignore::multiinterval.errors.EmptySetPropagationWarning')  # the empty step
def test_newton_step_proves_uniqueness_only_without_zero_slope():
    """the three conditions of `_newton_step`'s proof, each on its own"""
    piece = OutwardMultiInterval(0, 4)
    narrowed, proved = solver._newton_step(piece, OutwardMultiInterval(1, 2), 2, OutwardMultiInterval(-1))
    assert proved and narrowed == OutwardMultiInterval(Fraction(5, 2), 3)
    # the same newton set, but 0 in the slope as an isolated point: rolle needs no 0 at all
    slope = OutwardMultiInterval(0) | OutwardMultiInterval(1, 2)
    narrowed, proved = solver._newton_step(piece, slope, 2, OutwardMultiInterval(-1))
    assert not proved and narrowed == OutwardMultiInterval(Fraction(5, 2), 3)
    # an empty newton set: no zero, so none to be unique (the empty set is inside every interior)
    narrowed, proved = solver._newton_step(piece, OutwardMultiInterval(0), 2, OutwardMultiInterval(-1))
    assert not proved and not narrowed
    narrowed, proved = solver._newton_step(piece, OutwardMultiInterval(1, 2), 2, OutwardMultiInterval())
    assert not proved and not narrowed
    # a newton set reaching the piece's end: not inside the interior
    narrowed, proved = solver._newton_step(piece, OutwardMultiInterval(1, 2), 2, OutwardMultiInterval(-2))
    assert not proved and narrowed == OutwardMultiInterval(3, 4)


def test_a_jump_is_caught_by_the_value_decoration():
    """sign(x) with its derivative 0, a true enclosure of the derivative wherever there is one: the
    derivative's decoration is com, only the value's (def across the jump) says f is not C¹. a step
    on [-1, 0] from -1/2 would find no t with t * 0 = 1 and drop the zero at 0"""
    def step(x):
        return Dual(x.value.sign(), x.derivative * 0) if isinstance(x, Dual) else x.sign()
    roots = newton(step, M(-1, 1))
    assert any(0 in r.interval for r in roots), roots


def test_a_point_is_unique_only_when_f_is_exactly_zero():
    """sin(1) - sin(1) + 1e-300 is 1e-300, but its enclosure holds 0: not a proved zero"""
    assert newton(lambda x: x.sin() - x.sin() + 1e-300, 1) == (Root(OutwardMultiInterval(1), False),)
    assert newton(lambda x: x - 1, 1) == (Root(OutwardMultiInterval(1), True),)


def _evaluations(monkeypatch, f, x, **kwargs):
    count = [0]
    original = solver._evaluate

    def counted(f, piece):
        count[0] += 1
        return original(f, piece)
    monkeypatch.setattr(solver, '_evaluate', counted)
    newton(f, x, **kwargs)
    return count[0]


def test_evaluation_budgets(monkeypatch):
    """the costs the termination rules keep down (2026-09-27: 57, 57, 5 and 49 evaluations): the
    magnitude split on wide pieces (743 for the reals without it), and the cap on newton past tol at
    a double zero (922 without it, into the subnormals)"""
    assert _evaluations(monkeypatch, lambda x: x ** 2 - 2, REALS) <= 80
    assert _evaluations(monkeypatch, lambda x: x ** 2 - 2, M(-1e300, 1e300)) <= 80
    assert _evaluations(monkeypatch, lambda x: x ** 2 - 2, M(-1, 1e300)) <= 80
    assert _evaluations(monkeypatch, lambda x: x - 3e-5, M(0, 1e300)) <= 10
    assert _evaluations(monkeypatch, lambda x: x ** 2 * (x - 3), M(-6, 6)) <= 80


@pytest.mark.parametrize('sign', [1, -1])
def test_the_magnitude_split_goes_below_one(monkeypatch, sign):
    """
    owner 2026-10-04 (references/owner-questions-2026-10-03/solver.md §10.2): newton splits a piece
    by its exponents below 1 in magnitude too, both signs. `x ** 2 - 1e-40` on `[-1, 1]`: the first
    step leaves `[5e-41, 1]` and its mirror, which halving by width took 132 calls of f to bring to
    tol, both zeros unproved and a million times too wide; the split takes 28 calls and proves both
    (2026-10-04). a piece from 0 to at most 1 has no exponent to split at and is left alone
    """
    f = lambda x: x ** 2 - Fraction(1, 10 ** 40)  # noqa: E731
    roots = newton(f, M(-1, 1))
    assert [r.unique for r in roots] == [True, True]
    assert sign * Fraction(1, 10 ** 20) in roots[(1 + sign) // 2].interval
    assert all(r.interval.wid() < 1e-30 for r in roots)
    assert _evaluations(monkeypatch, f, M(-1, 1)) <= 25
    split = solver._magnitude_split
    piece = M(sign * 5e-41, sign) if sign > 0 else M(-1, -5e-41)
    point = split(piece, below_one=True)
    assert point in piece and point not in (piece.inf, piece.sup) and split(piece) is None
    assert split(M(0, sign) if sign > 0 else M(-1, 0), below_one=True) is None
    assert split(M(sign * 0.5, sign) if sign > 0 else M(-1, -0.5), below_one=True) is None  # within 16


@pytest.mark.parametrize('sign', [1, -1])
def test_a_wide_piece_within_tol_is_still_stepped(sign):
    """the split below 1 must not cost a proof the step made: a piece wide in magnitude but already
    within tol gets the newton step before it is output, so the small zero of `(x - 3)(x - 1e-25)`
    on `[1e-30, 4]` is proved unique, as before the split (2026-10-04: unproved `[1e-30, 3.6e-15]`
    without the step)"""
    small = sign * Fraction(1, 10 ** 25)
    x = M(1e-30, 4) if sign > 0 else M(-4, -1e-30)
    roots = newton(lambda t: (t - 3 * sign) * (t - small), x)
    assert [r.unique for r in roots] == [True, True]
    assert all(r.interval.wid() < 1e-30 for r in roots)
    assert any(small in r.interval for r in roots)


def test_a_piece_wider_than_the_doubles():
    """HANDOFF item newton-width: the halving test was `width <= piece.wid() / 2`, int true division,
    which raised OverflowError on an exact piece wider than the doubles at the first step that
    narrowed without proving (`[10 ** 400, 10 ** 401]` here, the first step); now `2 * width <=
    piece.wid()`, as `solve`'s. (more steps are slow: the exact ends grow, HANDOFF 2026-09-28)"""
    f = lambda x: x ** 2 - 9 * 10 ** 800  # noqa: E731
    x = M(10 ** 400, 10 ** 401)
    first = newton(f, x, max_steps=1)
    assert len(first) == 1 and not first[0].unique and 3 * 10 ** 400 in first[0].interval
    assert first[0].interval.wid() < x.wid()
    proved = newton(f, x, max_steps=3)
    assert len(proved) == 1 and proved[0].unique and 3 * 10 ** 400 in proved[0].interval


def test_tol_stops_refining():
    """f = 0: every point is a zero, so only tol stops the bisection (the split points are exact
    zeros, the pieces between them unproved)"""
    roots = newton(lambda x: 0, M(0, 1), tol=0.25)
    assert len(roots) == 9 and all(r.interval.wid() <= 0.25 for r in roots)


def test_close_zeros():
    """zeros 1e-12 apart are told apart, each proved unique; zeros closer than the doubles can tell
    (2 ** -60) are one Root, not claimed unique"""
    zeros = [1, 1 + Fraction(1, 10 ** 12)]
    roots = newton(_polynomial(zeros), M(-6, 6))
    _check_roots(roots, M(-6, 6), zeros)
    assert [r.unique for r in roots] == [True, True]
    zeros = [Fraction(1, 3), Fraction(1, 3) + Fraction(1, 2 ** 60)]
    roots = newton(_polynomial(zeros), M(-6, 6))
    _check_roots(roots, M(-6, 6), zeros)
    assert any(all(z in r.interval for z in zeros) and not r.unique for r in roots), roots


def test_poles():
    assert newton(lambda x: 1 / x, M(-1, 1)) == ()
    roots = newton(lambda x: 1 / x - x, M(-2, 2))
    assert [r.unique for r in roots] == [True, True]
    assert -1 in roots[0].interval and 1 in roots[1].interval


def test_not_differentiable_at_the_zero():
    roots = newton(lambda x: x.sqrt(), M(-1, 1))
    assert any(0 in r.interval for r in roots)
    roots = newton(lambda x: x.cbrt() - 1, M(0, 8))
    assert len(roots) == 1 and roots[0].unique and 1 in roots[0].interval


def test_unbounded_and_multi_piece_input():
    roots = newton(lambda x: x ** 2 - 2, REALS)
    assert [r.unique for r in roots] == [True, True]
    assert [round(float(r.interval.mid()), 12) for r in roots] == [-1.414213562373, 1.414213562373]
    roots = newton(lambda x: x.exp() - 2, REALS)
    assert len(roots) == 1 and roots[0].unique and abs(roots[0].interval.mid() - math.log(2)) < 1e-15
    x = M(-5, -1) | M(0, 1, start_closed=False) | M(1.2, 5)
    roots = newton(lambda x: x ** 2 - 2, x)
    assert len(roots) == 2 and all(r.interval.issubset(x) for r in roots)
    assert newton(lambda x: x ** 2 - 2, M(-1, 1) | M(2, 3)) == ()


def test_exact_zero_at_a_point():
    assert newton(lambda x: x - 1, 1) == (Root(OutwardMultiInterval(1), True),)
    assert newton(lambda x: x - 1, 2) == ()
    roots = newton(lambda x: x ** 3 - x, M(-5, 5))
    for z in (-1, 0, 1):
        assert any(z in r.interval and r.unique for r in roots), (z, roots)


def test_constant_functions():
    assert newton(lambda x: 1, M(0, 1)) == ()
    roots = newton(lambda x: 0, M(0, 1), tol=0.1)
    assert M(0, 1).issubset(M().union(*(r.interval for r in roots)))
    assert all(r.interval.is_degenerate for r in roots if r.unique), roots  # the split points


def test_results_are_outward():
    roots = newton(lambda x: x ** 2 - 2, M(0, 2))
    assert type(roots[0].interval) is OutwardMultiInterval


def test_arguments_are_checked():
    with pytest.raises(TypeError, match='MultiInterval or a number'):
        newton(lambda x: x, 'nope')
    with pytest.raises(TypeError, match='not a Dual or a number'):
        newton(lambda x: x.value, M(0, 1))


def test_warnings_stay_inside(recwarn):
    """1/x at the piece [0] is an indeterminate point: silenced inside, and nothing leaks"""
    newton(lambda x: 1 / x - 1, M(0, 2))
    assert not recwarn.list
