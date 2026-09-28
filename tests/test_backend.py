"""
the gmpy2 backend (`intervals/_gmpy2.py`) against the pure path, and the switch (`intervals/backend.py`)

the backend's whole contract is "the same doubles and the same flags, faster": at every input the pure
path accepts it returns the same float with the same sign bit, or None, and then the pure path runs.
no flag is the backend's (exactness, attainment and the shortcuts past the float range run first, in
`elementary` and the applicator), so the doubles are what is compared. each primitive is checked three
ways at every point and direction:
* pure: `elementary`'s function (or the outward hook) under `backend._use('python')`
* direct: `_gmpy2`'s function itself, wherever the dispatch would call it (past `exact`, `_beyond` and
  the other shortcuts): the same double and sign bit, and None exactly where `declines_*` below (the
  backend's table, written here from v2-plan.md, not read from the module) says. a backend answering
  None everywhere would be bit-identical and useless, so a None where it should answer is a failure
* dispatched: the same `elementary` function under `backend._use('gmpy2')`, which is where a dispatch
  that drops or misroutes an argument (`log`'s base, `rootn`'s n) shows
then the same at set level (`test_set_level_matches_unary`, `_binary`, `_newton`: a drawn
multi-interval through every method, the `repr` under both backends), and the switch itself
(`test_use_switches` is the guard that the two backends really ran different code; `test_env_var`,
`test_version_floor`). gmpy2 is in `[test]`, so nothing here skips.
"""
import contextlib
import math
import os
import subprocess
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import gmpy2
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import _gmpy2
from intervals import backend
from intervals import cos_rev
from intervals import elementary
from intervals import newton
from intervals import ops
from intervals import pow_rev2
from intervals import sin_rev
from intervals import tan_rev
from intervals.rounding import DOWN
from intervals.rounding import MAX
from intervals.rounding import NEAREST
from intervals.rounding import UP
from intervals.rounding import round_rational
from tests.test_oracle_flint import EXTREMES
from tests.test_oracle_flint import points

INF = math.inf
DIRECTIONS = (DOWN, NEAREST, UP)
ROOT = Path(__file__).resolve().parent.parent


def same(a, b) -> bool:
    """the same double, sign bit included (`-0.0 == 0.0`, so `==` alone would miss a -0.0)"""
    return (type(a) is float and type(b) is float and a == b
            and math.copysign(1.0, a) == math.copysign(1.0, b))


# THE BACKEND'S TABLE, WRITTEN FROM V2-PLAN.MD: WHERE IT DECLINES

def _dyadic(x) -> bool:
    d = Fraction(x).denominator
    return d & (d - 1) == 0


def _fits(*xs) -> bool:
    """every numerator and denominator within the bound on the bits the backend takes (class 15)"""
    return all(max(abs(Fraction(x).numerator).bit_length(), Fraction(x).denominator.bit_length())
               <= _gmpy2.BOUND for x in xs)


def declines_rounded(name, x, base) -> bool:
    if not _fits(x):
        return True
    if name in ('atan', 'acot'):  # atan2 of the two ints: any rational
        return False
    if name == 'acoth' or (name == 'log' and base is not None):
        return True
    if name == 'rootn' and not 0 < base < 2 ** 31:
        return True
    return not _dyadic(x)


def declines_pow(x, y) -> bool:
    return not (_fits(x, y) and _dyadic(x) and _dyadic(y))


def declines_angle(q, m) -> bool:
    if not _fits(q):
        return True
    if q == 0:
        return abs(m) not in (1, 2)
    return not (m in (0, 1, -1) or (m == 2 and q < 0) or (m == -2 and q > 0))


def declines_inverse_trig(name, v, k) -> bool:
    if k != 0 or v in (INF, -INF) or not _fits(v):
        return True
    return name != 'atan' and not _dyadic(v)


def declines_outward(op, args) -> bool:
    return not _fits(*args)


# THE THREE-WAY CHECK OF EACH PRIMITIVE

def check_rounded(name, x, base=None) -> bool:
    """the three-way check; True if the backend answered (it was reached and did not decline)"""
    reached = (x not in (INF, -INF) and elementary.exact(name, x, base) is None
               and elementary._beyond(name, Fraction(x), base) is None)
    answered = False
    for d in DIRECTIONS:
        with backend._use('python'):
            want = elementary.rounded(name, x, d, base)
        if reached:
            got = _gmpy2.rounded(name, Fraction(x), d, base)
            assert (got is None) == declines_rounded(name, x, base), (name, x, d, base, got)
            if got is not None:
                answered = True
                assert same(got, want), (name, x, d, base, got, want)
        with backend._use('gmpy2'):
            got = elementary.rounded(name, x, d, base)
        assert same(got, want), ('dispatched', name, x, d, base, got, want)
    return answered


def check_pow(x, y) -> bool:
    reached = elementary.exact_pow(x, y) is None
    if reached:
        lo, hi = sorted(Fraction(y) * b for b in elementary._ln_bracket(Fraction(x)))
        reached = not (lo > 800 or hi < -800)
    answered = False
    for d in DIRECTIONS:
        with backend._use('python'):
            want = elementary.rounded_pow(x, y, d)
        if reached:
            got = _gmpy2.rounded_pow(Fraction(x), Fraction(y), d)
            assert (got is None) == declines_pow(x, y), (x, y, d, got)
            if got is not None:
                answered = True
                assert same(got, want), (x, y, d, got, want)
        with backend._use('gmpy2'):
            got = elementary.rounded_pow(x, y, d)
        assert same(got, want), ('dispatched', x, y, d, got, want)
    return answered


def check_angle(q, m) -> bool:
    reached = not (q == 0 and m == 0)
    answered = False
    for d in DIRECTIONS:
        with backend._use('python'):
            want = elementary.rounded_angle(q, m, d)
        if reached:
            got = _gmpy2.rounded_angle(Fraction(q), m, d)
            assert (got is None) == declines_angle(q, m), (q, m, d, got)
            if got is not None:
                answered = True
                assert same(got, want), (q, m, d, got, want)
        with backend._use('gmpy2'):
            got = elementary.rounded_angle(q, m, d)
        assert same(got, want), ('dispatched', q, m, d, got, want)
    return answered


def check_inverse_trig(name, v, sign, k) -> bool:
    w, j = (1, k + sign) if name == 'acos' and v == -1 else (v, k)  # the pure path's remap, first
    value = None if w in (INF, -INF) else elementary.exact(name, w)
    reached = not (value is not None and j == 0)
    answered = False
    for d in DIRECTIONS:
        with backend._use('python'):
            want = elementary.rounded_inverse_trig(name, v, sign, k, d)
        if reached:
            got = _gmpy2.rounded_inverse_trig(name, w, sign, j, d)
            assert (got is None) == declines_inverse_trig(name, w, j), (name, v, sign, k, d, got)
            if got is not None:
                answered = True
                assert same(got, want), (name, v, sign, k, d, got, want)
        with backend._use('gmpy2'):
            got = elementary.rounded_inverse_trig(name, v, sign, k, d)
        assert same(got, want), ('dispatched', name, v, sign, k, d, got, want)
    return answered


def check_outward(op, args) -> bool:
    """the outward hook: called only for finite operands, one a float at least, where `fn` has a value"""
    desc = ops.OUTWARD[op]
    assert all(x not in (INF, -INF) for x in args) and any(isinstance(x, float) for x in args)
    assert desc.fn(*args) is not None
    answered = False
    for i, d in enumerate((DOWN, UP)):
        with backend._use('python'):
            want = desc.rounded[i](*args)
        got = _gmpy2.outward(op, args, d)
        assert (got is None) == declines_outward(op, args), (op, args, d, got)
        if got is not None:
            answered = True
            assert same(got, want), (op, args, d, got, want)
        with backend._use('gmpy2'):
            got = desc.rounded[i](*args)
        assert same(got, want), ('dispatched', op, args, d, got, want)
    return answered


CHECKS = {'rounded': check_rounded, 'pow': check_pow, 'angle': check_angle,
          'inverse': check_inverse_trig, 'outward': check_outward}


# THE EDGE CLASSES (v2-plan.md "testing", the backend's bullet; M16e's record)

TINY = 5e-324
EDGES = [
    # 1 dyadic floats over the whole range
    ('1', 'rounded', ('exp', 0.7)), ('1', 'rounded', ('log', TINY)), ('1', 'rounded', ('log', MAX)),
    ('1', 'rounded', ('sqrt', TINY)), ('1', 'rounded', ('sin', -MAX)), ('1', 'rounded', ('atan', MAX)),
    ('1', 'rounded', ('cbrt', -MAX)), ('1', 'rounded', ('asinh', -MAX)), ('1', 'rounded', ('acot', -MAX)),
    ('1', 'rounded', ('acot', 0)), ('1', 'rounded', ('acosh', MAX)), ('1', 'rounded', ('log1p', -0.5)),
    # 2 results in the subnormals and at the underflow edge
    ('2', 'rounded', ('exp', -745.1332191019411)), ('2', 'rounded', ('exp', -745.1332191019412)),
    ('2', 'rounded', ('exp', -745.0)), ('2', 'rounded', ('exp2', -1074.5)), ('2', 'rounded', ('exp10', -323.5)),
    ('2', 'rounded', ('sinh', TINY)), ('2', 'rounded', ('tanh', TINY)), ('2', 'rounded', ('asinh', TINY)),
    ('2', 'rounded', ('atan', TINY)), ('2', 'rounded', ('sin', TINY)), ('2', 'rounded', ('expm1', TINY)),
    ('2', 'rounded', ('log1p', TINY)), ('2', 'rounded', ('sech', 745.5)), ('2', 'rounded', ('csch', 745.5)),
    # 3 a result rounding to zero from below: -0.0 in MPFR, 0.0 in the pure path
    ('3', 'rounded', ('sin', -TINY)), ('3', 'rounded', ('atan', -TINY)), ('3', 'rounded', ('tanh', -TINY)),
    ('3', 'rounded', ('asinh', -TINY)), ('3', 'rounded', ('expm1', -TINY)), ('3', 'rounded', ('tan', -TINY)),
    ('3', 'rounded', ('csch', -746.0)), ('3', 'rounded', ('csch', -746.5)), ('3', 'rounded', ('csch', -747.0)),
    ('3', 'outward', ('mul', (-1e-300, 1e-300))), ('3', 'outward', ('add', (0.1, -0.1))),
    ('3', 'outward', ('sub', (0.1, 0.1))), ('3', 'outward', ('div', (-1e-300, 1e300))),
    ('3', 'inverse', ('atan', TINY, -1, 0)), ('3', 'inverse', ('asin', TINY, -1, 0)),
    ('3', 'inverse', ('atan', Fraction(1, 2 ** 1100), -1, 0)),
    ('3', 'angle', (-TINY, 0)), ('3', 'angle', (Fraction(-1, 2 ** 1100), 0)),
    ('3', 'outward', ('mul', (-TINY, Fraction(1, 3)))), ('3', 'outward', ('mul', (Fraction(-1, 10 ** 400), 1e-300))),
    ('3', 'outward', ('add', (-TINY, Fraction(2, 3 * 2 ** 1074)))),
    # 4 overflow in each direction
    ('4', 'rounded', ('exp', 709.782712893384)), ('4', 'rounded', ('exp', 709.7827128933841)),
    ('4', 'rounded', ('expm1', 709.782712893384)), ('4', 'rounded', ('expm1', 709.7827128933841)),
    ('4', 'rounded', ('sinh', 710.4758600739439)), ('4', 'rounded', ('sinh', 710.5)),
    ('4', 'rounded', ('sinh', -710.5)), ('4', 'rounded', ('cosh', 710.5)), ('4', 'rounded', ('cosh', -710.5)),
    ('4', 'rounded', ('exp2', 1023.9999999999999)), ('4', 'rounded', ('exp10', 308.2547155599167)),
    ('4', 'rounded', ('coth', Fraction(1, 2 ** 1100))), ('4', 'rounded', ('coth', Fraction(-1, 2 ** 1100))),
    ('4', 'rounded', ('csch', Fraction(1, 2 ** 1100))),
    ('4', 'outward', ('add', (MAX, MAX))), ('4', 'outward', ('sub', (-MAX, MAX))),
    ('4', 'outward', ('div', (1.0, TINY))), ('4', 'outward', ('reciprocal', (TINY,))),
    ('4', 'outward', ('reciprocal', (-TINY,))), ('4', 'outward', ('mul', (1e300, -1e300))),
    # 5 wide exact inputs: never rounded to 53 bits on the way in
    ('5', 'rounded', ('sin', 2 ** 60 + 1)), ('5', 'rounded', ('sin', 2 ** 60)), ('5', 'rounded', ('sin', 10 ** 30)),
    ('5', 'rounded', ('sin', 2 ** 3000 + 1)), ('5', 'rounded', ('cos', 2 ** 3000 + 1)), ('5', 'rounded', ('sin', 1e22)),
    ('5', 'rounded', ('log', Fraction(2 ** 100 + 1, 2 ** 100))), ('5', 'rounded', ('exp', Fraction(2 ** 100 + 1, 2 ** 100))),
    ('5', 'rounded', ('atan', 10 ** 30 + 1)), ('5', 'rounded', ('sqrt', 2 ** 107 + 1)),
    ('5', 'rounded', ('log1p', Fraction(1, 2 ** 80) + Fraction(1, 2 ** 200))),
    # 6 non-dyadic rationals: declined, but for atan and acot (atan2 of two ints) and the hook (mpq)
    ('6', 'rounded', ('exp', Fraction(1, 3))), ('6', 'rounded', ('sin', Fraction(1, 10))),
    ('6', 'rounded', ('log', Fraction(10 ** 40 + 1, 3))), ('6', 'rounded', ('sqrt', Fraction(2, 3))),
    ('6', 'rounded', ('atan', Fraction(1, 3))), ('6', 'rounded', ('atan', Fraction(-7, 10))),
    ('6', 'rounded', ('acot', Fraction(1, 3))), ('6', 'rounded', ('acot', Fraction(-7, 10))),
    ('6', 'rounded', ('acot', Fraction(10 ** 40 + 1, 3))), ('6', 'rounded', ('acot', Fraction(-1, 10 ** 30 + 7))),
    ('6', 'outward', ('add', (0.1, Fraction(1, 3)))), ('6', 'outward', ('div', (Fraction(1, 3), 0.1))),
    # 7 infinite x: the pi multiples stay pure
    ('7', 'rounded', ('atan', INF)), ('7', 'rounded', ('atan', -INF)), ('7', 'rounded', ('acot', -INF)),
    ('7', 'rounded', ('acot', INF)), ('7', 'inverse', ('atan', INF, 1, 0)), ('7', 'inverse', ('atan', -INF, -1, 0)),
    # 8 rounded_angle: every (q, m) functions._angle makes, and some it does not (declined)
    *[('8', 'angle', (q, m)) for q in (Fraction(1, 3), Fraction(-1, 3), Fraction(3, 4), Fraction(-5, 2), 10 ** 30,
                                        Fraction(1, 2 ** 1074)) for m in (0, 1, -1)],
    *[('8', 'angle', (q, 2)) for q in (Fraction(-1, 3), Fraction(-3, 4), -(10 ** 30))],
    *[('8', 'angle', (q, -2)) for q in (Fraction(1, 3), Fraction(3, 4), 10 ** 30)],
    *[('8', 'angle', (0, m)) for m in (1, -1, 2, -2, 3, 0)],
    ('8', 'angle', (Fraction(1, 3), 2)), ('8', 'angle', (Fraction(-1, 3), -2)), ('8', 'angle', (Fraction(1, 3), 3)),
    # 9 rounded_inverse_trig: k = 0 native, the sign reversing the direction; k != 0 declined
    ('9', 'inverse', ('asin', 0.5, 1, 0)), ('9', 'inverse', ('asin', 0.5, -1, 0)), ('9', 'inverse', ('acos', 0.5, -1, 0)),
    ('9', 'inverse', ('acos', -0.75, 1, 0)), ('9', 'inverse', ('atan', Fraction(1, 3), -1, 0)),
    ('9', 'inverse', ('asin', 1, 1, 0)), ('9', 'inverse', ('asin', -1, -1, 0)), ('9', 'inverse', ('acos', -1, 1, 0)),
    ('9', 'inverse', ('acos', -1, -1, 0)), ('9', 'inverse', ('asin', 0.5, 1, 3)), ('9', 'inverse', ('atan', 0.5, -1, -2)),
    ('9', 'inverse', ('asin', Fraction(1, 3), 1, 0)), ('9', 'inverse', ('asin', -TINY, 1, 0)),
    # 10 rootn: even and odd n, x < 0 with odd n, n at the margin (declined from 2**31), n < 0 declined
    ('10', 'rounded', ('rootn', 2.0, 2)), ('10', 'rounded', ('rootn', 2.0, 3)), ('10', 'rounded', ('rootn', -2.0, 3)),
    ('10', 'rounded', ('rootn', -TINY, 5)), ('10', 'rounded', ('rootn', MAX, 2)), ('10', 'rounded', ('rootn', 3.0, 10 ** 6)),
    ('10', 'rounded', ('rootn', 2.0, 2 ** 31 - 1)), ('10', 'rounded', ('rootn', -2.0, 2 ** 31 - 1)),
    ('10', 'rounded', ('rootn', 2.0, 2 ** 31)), ('10', 'rounded', ('rootn', 2.0, 2 ** 32)),
    ('10', 'rounded', ('rootn', 2.0, 2 ** 64 + 1)), ('10', 'rounded', ('rootn', 2.0, -2)),
    ('10', 'rounded', ('rootn', -2.0, -3)), ('10', 'rounded', ('rootn', Fraction(1, 3), 2)), ('10', 'rounded', ('rootn', 7, 2)),
    # 11 rounded_pow
    ('11', 'pow', (2.0, 0.5)), ('11', 'pow', (3.0, 0.5)), ('11', 'pow', (0.5, 3.7)), ('11', 'pow', (1 + 2 ** -52, 1e19)),
    ('11', 'pow', (2.0, -1074.5)), ('11', 'pow', (2.0, 1023.9999999999999)), ('11', 'pow', (10 ** 30, 0.25)),
    ('11', 'pow', (Fraction(1, 3), 0.5)), ('11', 'pow', (2.0, Fraction(1, 3))), ('11', 'pow', (TINY, 0.5)),
    ('11', 'pow', (MAX, -0.5)),
    # 12 the hook with mixed operands: an int past 2**53, a non-dyadic Fraction (mpq), a dyadic one
    ('12', 'outward', ('add', (0.1, 2 ** 60 + 1))), ('12', 'outward', ('mul', (0.1, 2 ** 60 + 1))),
    ('12', 'outward', ('sub', (Fraction(1, 3), 0.1))), ('12', 'outward', ('mul', (1e-300, Fraction(1, 3)))),
    ('12', 'outward', ('div', (1.0, Fraction(1, 3)))), ('12', 'outward', ('add', (0.5, Fraction(3, 8)))),
    ('12', 'outward', ('mul', (1e300, Fraction(2 ** 100 + 1, 2 ** 100)))),
    ('12', 'outward', ('div', (Fraction(1, 10 ** 400), 1e300))), ('12', 'outward', ('add', (1e308, 10 ** 400))),
    ('12', 'outward', ('reciprocal', (0.1,))), ('12', 'outward', ('div', (3, 0.1))),
]


def _edge_id(case):
    label, kind, args = case
    return f'{label}-{kind}-' + '-'.join(repr(a)[:24] for a in (args if kind != 'outward' else (args[0], *args[1])))


@pytest.mark.parametrize('case', EDGES, ids=[_edge_id(c) for c in EDGES])
def test_edge_class(case):
    _, kind, args = case
    CHECKS[kind](*args)


HOSTILE = dict(precision=10, round=gmpy2.RoundUp, emin=-100, emax=100, trap_underflow=True, trap_overflow=True,
               trap_inexact=True, trap_invalid=True, trap_erange=True, trap_divzero=True)


@contextlib.contextmanager
def hostile_global_context():
    """
    gmpy2's global context at 10 bits, rounding up, a tiny exponent range and every trap on, changed
    in place (so a backend holding the global context object sees it too), restored after
    """
    context = gmpy2.get_context()
    saved = {key: getattr(context, key) for key in HOSTILE}
    for key, value in HOSTILE.items():
        setattr(context, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(context, key, value)


def test_the_hostile_context_is_hostile():
    """the guard on class 13: a bare `mpfr(0.1)` raises in it, so a backend that built one would too"""
    with hostile_global_context():
        with pytest.raises(gmpy2.InexactResultError):
            gmpy2.mpfr(0.1)
    assert gmpy2.get_context().precision == 53 and not gmpy2.get_context().trap_inexact


@pytest.mark.parametrize('case', EDGES, ids=[_edge_id(c) for c in EDGES])
def test_hostile_global_context(case):
    """class 13: the backend reads only its own contexts, never gmpy2's global one"""
    _, kind, args = case
    with hostile_global_context():
        CHECKS[kind](*args)


# 14 hard points, by construction: a value within a tiny fraction of an ulp of a double or a midpoint

HARD = [
    *[(name, Fraction(1, 2 ** k)) for k in (30, 60, 200, 1000, 1074)
      for name in ('sin', 'tan', 'atan', 'asin', 'sinh', 'tanh', 'asinh', 'atanh', 'expm1', 'log1p')],
    *[(name, Fraction(-1, 2 ** k)) for k in (30, 1074) for name in ('sin', 'atan', 'expm1', 'log1p')],
    *[(name, Fraction(1, 2 ** k)) for k in (30, 60, 200, 1000) for name in ('exp', 'cos', 'cosh', 'sech')],
    ('exp', 18.256756915164082), ('log', 42.029035728056286), ('sin', -3.268808420477516),
    ('atan', 4.834748463344576),
]


@pytest.mark.parametrize('name, x', HARD)
def test_hard_point(name, x):
    """class 14; every point is answered by the backend (the class is not empty by declining)"""
    assert check_rounded(name, x)


@pytest.mark.parametrize('name, x', EXTREMES)
def test_extreme_point(name, x):
    check_rounded(name, x)


# 15 an operand past the bound on its bits: declined, since past MPFR's own exponent range (2**30 on
# windows) a dyadic flushes to 0 with a construction rc of 0 (to inf with 1), silently

def _bound_cases(b, cheap=False):
    """
    `(past, inside)`: calls with an operand just past the bound b (b + 1 bits), declined, and just
    inside it (b bits), answered. with `cheap`, only those whose pure path stays fast at a million
    bits (it grows about quadratically for sin, exp, atan and pow at a tiny x: 1.5-5.5 s for sin or
    exp at 2**16 bits, 2026-09-28, `tools/backend_speed.py --bound`)
    """
    tiny, huge = Fraction(1, 2 ** b), 2 ** b  # b + 1 bits: just past
    past = [
        ('rounded', ('atan', huge)), ('rounded', ('log', huge)), ('rounded', ('acot', tiny)),
        ('outward', ('mul', (1.0, tiny))), ('outward', ('add', (0.5, huge))),
        ('outward', ('mul', (0.1, Fraction(1, 3 * 2 ** b)))), ('angle', (tiny, 1)),
    ]
    inside = [('rounded', ('log', 2 ** (b - 1) + 1)), ('rounded', ('atan', 2 ** (b - 1) + 1)),
              ('outward', ('mul', (1.0, Fraction(1, 2 ** (b - 1)))))]
    if not cheap:
        past += [('rounded', ('sin', tiny)), ('rounded', ('exp', tiny)), ('rounded', ('atan', tiny)),
                 ('pow', (2.0, tiny)), ('inverse', ('atan', tiny, 1, 0))]
        inside += [('rounded', ('exp', Fraction(1, 2 ** (b - 1)))), ('rounded', ('sin', Fraction(1, 2 ** (b - 1)))),
                   ('pow', (2.0, Fraction(1, 2 ** (b - 1))))]
    return past, inside


def test_past_the_bound_is_declined_at_a_small_bound(monkeypatch):
    monkeypatch.setattr(_gmpy2, 'BOUND', 1 << 12)
    past, inside = _bound_cases(1 << 12)
    for i, (kind, args) in enumerate(past):
        assert not CHECKS[kind](*args), ('past', i)
    for i, (kind, args) in enumerate(inside):
        assert CHECKS[kind](*args), ('inside', i)


def test_past_the_bound_is_declined_at_the_real_bound():
    assert _gmpy2.BOUND == 1 << 20
    past, inside = _bound_cases(_gmpy2.BOUND, cheap=True)
    for i, (kind, args) in enumerate(past):
        assert not CHECKS[kind](*args), ('past', i)
    for i, (kind, args) in enumerate(inside):
        assert CHECKS[kind](*args), ('inside', i)


# DRAWN POINTS

@pytest.mark.parametrize('name', elementary.NAMES)
@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_rounded_matches_python(name, data):
    check_rounded(name, data.draw(points(name), label='x'))


BASES = (Fraction(1, 2), Fraction(1, 3), 0.25, 2, 3, 10, 2.5, Fraction(7, 2))


@settings(max_examples=60, deadline=None)
@given(x=points('log'), base=st.sampled_from(BASES))
def test_log_base_matches_python(x, base):
    check_rounded('log', x, base)


ROOT_DEGREES = (2, 3, 4, 5, 7, -2, -3, -5, 10 ** 6, 2 ** 31 - 1, 2 ** 31, 2 ** 32, 2 ** 64 + 1, -(2 ** 31))


@settings(max_examples=80, deadline=None)
@given(x=points('exp').filter(lambda x: x != 0), n=st.sampled_from(ROOT_DEGREES))
def test_rootn_matches_python(x, n):
    check_rounded('rootn', x if n % 2 else abs(x), n)


positive = st.one_of(st.floats(min_value=0, exclude_min=True, allow_infinity=False),
                     st.integers(1, 10 ** 30), st.fractions(min_value=0, max_denominator=10 ** 6).filter(bool))
exponents = st.one_of(st.floats(-60, 60), st.integers(-400, 400), st.fractions(-60, 60, max_denominator=10 ** 4))


@settings(max_examples=150, deadline=None)
@given(x=positive, y=exponents)
def test_pow_matches_python(x, y):
    check_pow(x, y)


rationals = st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers(-10 ** 30, 10 ** 30),
                      st.fractions(max_denominator=10 ** 12))


@settings(max_examples=150, deadline=None)
@given(q=rationals, m=st.sampled_from((0, 1, -1, 2, -2, 3, -3)))
def test_angle_matches_python(q, m):
    check_angle(q, m)


@settings(max_examples=150, deadline=None)
@given(data=st.data(), sign=st.sampled_from((1, -1)), k=st.sampled_from((0, 0, 0, 1, -1, 5)))
def test_inverse_trig_matches_python(data, sign, k):
    name = data.draw(st.sampled_from(('asin', 'acos', 'atan')), label='name')
    v = data.draw(st.one_of(points(name), st.sampled_from((INF, -INF))) if name == 'atan' else points(name), label='v')
    check_inverse_trig(name, v, sign, k)


floats = st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.floats(-10, 10),
                   st.sampled_from((0.1, -0.1, 0.0, TINY, -TINY, MAX, -MAX, 1e-300, 1e300)))
operands = st.one_of(floats, st.integers(-10 ** 20, 10 ** 20), st.integers(2 ** 53, 2 ** 70),
                     st.fractions(max_denominator=10 ** 9))


@settings(max_examples=300, deadline=None)
@given(op=st.sampled_from(('add', 'sub', 'mul', 'div')), a=floats, b=operands, swap=st.booleans())
def test_outward_matches_python(op, a, b, swap):
    args = (b, a) if swap else (a, b)
    if op == 'div' and args[1] == 0:
        args = (args[1], args[0]) if args[0] != 0 else (args[0], 1.5)
    check_outward(op, args)


@settings(max_examples=60, deadline=None)
@given(a=floats.filter(bool))
def test_outward_reciprocal_matches_python(a):
    check_outward('reciprocal', (a,))


# COVERAGE: THE BACKEND ANSWERS WHERE ITS TABLE SAYS, ONE POINT PER ROW, AND DECLINES THE REST

@pytest.mark.parametrize('kind, args', [
    *[('rounded', (name, 0.375)) for name in elementary.NAMES if name not in ('acosh', 'acoth', 'log')],
    ('rounded', ('log', 0.375)), ('rounded', ('acosh', 1.375)), ('rounded', ('atan', Fraction(1, 3))),
    ('rounded', ('acot', Fraction(1, 3))), ('rounded', ('rootn', 0.375, 5)), ('pow', (0.375, 1.5)),
    ('angle', (Fraction(1, 3), 1)), ('angle', (0, -1)), ('inverse', ('asin', 0.375, -1, 0)),
    ('outward', ('add', (0.1, 0.2))), ('outward', ('sub', (0.1, 3))), ('outward', ('mul', (0.1, Fraction(3, 8)))),
    ('outward', ('div', (0.1, 3.0))), ('outward', ('reciprocal', (3.0,))), ('outward', ('add', (0.1, Fraction(1, 3)))),
])
def test_backend_answers_where_it_should(kind, args):
    assert CHECKS[kind](*args)


@pytest.mark.parametrize('kind, args', [
    ('rounded', ('exp', Fraction(1, 3))), ('rounded', ('log', 0.375, 3)), ('rounded', ('log', 0.375, 0.5)),
    ('rounded', ('acoth', 1.375)), ('rounded', ('rootn', 0.375, -2)), ('rounded', ('rootn', 0.375, 2 ** 31)),
    ('pow', (0.375, Fraction(1, 3))), ('angle', (Fraction(1, 3), 2)), ('angle', (0, 3)),
    ('inverse', ('asin', 0.375, 1, 1)), ('inverse', ('asin', Fraction(1, 3), 1, 0)), ('inverse', ('atan', INF, 1, 0)),
])
def test_backend_declines_where_it_should(kind, args):
    assert not CHECKS[kind](*args)


def test_power_descriptors_decline(monkeypatch):
    """the `pow{n}` descriptors (`ops._power_descriptor`, outward) keep the pure hook"""
    calls = _spy(monkeypatch, 'outward')
    with backend._use('gmpy2'):
        for n in (2, 3, -2):
            desc = ops._power_descriptor(n, True)
            for i, d in enumerate((DOWN, UP)):
                assert same(desc.rounded[i](0.1), round_rational(desc.fn(Fraction(0.1)), d))
    assert calls == []


def test_the_shortcuts_run_before_the_backend(monkeypatch):
    """
    a cost rule, not a value (MPFR agrees with `_beyond` there): `exact`, `_beyond`, the pi limits at
    ±inf and `rounded_pow`'s range shortcuts answer first, so the backend is not called for them
    """
    rounded, pow_ = _spy(monkeypatch, 'rounded'), _spy(monkeypatch, 'rounded_pow')
    with backend._use('gmpy2'):
        for name, x in (('sqrt', 4), ('exp', 0), ('exp', 800), ('exp', -800), ('tanh', 30), ('coth', -25),
                        ('sinh', -800), ('csch', -800), ('sech', 800), ('expm1', -50), ('exp2', 2000),
                        ('atan', INF), ('acot', -INF)):
            for d in DIRECTIONS:
                elementary.rounded(name, x, d)
        for x, y in ((2, Fraction(4001, 2)), (Fraction(1, 2), Fraction(4001, 2)), (4, Fraction(1, 2))):
            for d in DIRECTIONS:
                elementary.rounded_pow(x, y, d)
    assert rounded == [] and pow_ == []


def test_the_hook_is_keyed_on_the_descriptor(monkeypatch):
    """an outward descriptor that is not one of the five, whatever its name, keeps the pure hook"""
    calls = _spy(monkeypatch, 'outward')
    impostor = ops.outward(ops.MUL._replace(name='add'))
    with backend._use('gmpy2'):
        assert same(impostor.rounded[1](0.1, 3.0), round_rational(Fraction(0.1) * 3, UP))
        assert calls == []
        ops.OUTWARD['add'].rounded[1](0.1, 3.0)
    assert calls == [('add', (0.1, 3.0), UP)]


# GUARDS: A MISSED EXACT CASE, A NAN

def test_missed_exact_case_raises(monkeypatch):
    """
    `exact` is the backend's only source of "the value is rational". MPFR would return sqrt(4) = 2
    with a zero ternary value; the backend refuses in every direction, where the pure loop (its
    white-box twin in test_elementary) can answer DOWN because both ends of [2, 2 + tiny] round down
    """
    monkeypatch.setattr(elementary, 'exact', lambda name, x, base=None: None)
    with backend._use('gmpy2'):
        for d in DIRECTIONS:
            with pytest.raises(ArithmeticError, match='exact case was missed'):
                elementary.rounded('sqrt', 4, d)


@pytest.mark.parametrize('patch, call', [
    ('exact_pow', lambda d: elementary.rounded_pow(4, Fraction(1, 2), d)),
    ('exact', lambda d: elementary.rounded_inverse_trig('asin', 0, 1, 0, d)),
    ('exact', lambda d: elementary.rounded_inverse_trig('atan', 0, -1, 0, d)),
])
def test_missed_exact_case_raises_in_pow_and_inverse_trig(monkeypatch, patch, call):
    """
    the same guard on the other two primitives whose value can be rational (MPFR's 2 = 4 ** 1/2 and
    0 = asin(0), each with a zero ternary value): each keeps calling it. the angle's cannot be reached
    (atan2 of two ints with y != 0 and pi are irrational), so it has no twin
    """
    monkeypatch.setattr(elementary, patch, lambda *args, **kwargs: None)
    with backend._use('gmpy2'):
        for d in DIRECTIONS:
            with pytest.raises(ArithmeticError, match='exact case was missed'):
                call(d)


def test_a_domain_slip_raises_instead_of_returning_nan():
    """MPFR answers nan outside a domain; the backend raises, as the pure path would, never a nan end"""
    for name, x in (('sqrt', Fraction(-1)), ('log', Fraction(-2)), ('asin', Fraction(2)), ('acosh', Fraction(1, 2))):
        for d in DIRECTIONS:
            with pytest.raises(ArithmeticError, match='not a number'):
                _gmpy2.rounded(name, x, d)


def test_inputs_it_does_not_know_are_declined():
    """the input is dispatched on its type: anything but int, float and Fraction is the pure path's"""
    from decimal import Decimal
    assert _gmpy2.rounded('exp', Decimal('0.5'), UP) is None
    assert _gmpy2.outward('add', (0.5, Decimal('0.5')), UP) is None
    assert _gmpy2.outward('add', (0.5, INF), UP) is None
    assert _gmpy2.outward('div', (0.5, 0.0), UP) is None
    assert _gmpy2.outward('reciprocal', (0,), UP) is None


# SET LEVEL: A DRAWN MULTI-INTERVAL THROUGH EVERY METHOD, THE REPR UNDER BOTH BACKENDS

set_values = st.one_of(
    st.floats(-1e3, 1e3, allow_nan=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.integers(-50, 50),
    st.fractions(min_value=-50, max_value=50, max_denominator=12),
    st.sampled_from([0, 1, -1, 0.5, 2.0, 0.1, Fraction(1, 3), 3, INF, -INF]),
)


@st.composite
def multi_intervals(draw, cls, values=set_values, max_pieces=3):
    pieces = []
    for _ in range(draw(st.integers(1, max_pieces))):
        a, b = sorted((draw(values), draw(values)))
        closed = (draw(st.booleans()), draw(st.booleans())) if a < b else (True, True)
        pieces.append((a, b, *closed))
    return cls.from_pieces(pieces)


def _outcome(f, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return repr(f(*args))
        except Exception as e:  # the same exception under both is the same outcome
            return f'raised {type(e).__name__}: {e}'


UNARY = {name: (lambda name: lambda a: getattr(a, name)())(name) for name in elementary.NAMES}
UNARY.update({
    'reciprocal': lambda a: a.reciprocal(),
    'pow 3': lambda a: a ** 3,
    'pow -2': lambda a: a ** -2,
    'pow 2.5': lambda a: a ** 2.5,
    **{f'rootn {n}': (lambda n: lambda a: a.rootn(n))(n) for n in (2, 3, -2, -3)},
    **{f'log base {b}': (lambda b: lambda a: a.log(b))(b) for b in (Fraction(1, 2), 0.25, 3, 2.5)},
})
BINARY = {
    'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b, 'div': lambda a, b: a / b,
    'atan2': lambda a, b: a.atan2(b), 'pow': lambda a, b: a ** b, 'hypot': lambda a, b: a.hypot(b),
    'sin_rev': sin_rev, 'cos_rev': cos_rev, 'tan_rev': tan_rev,
    'pow_rev2': lambda a, b: pow_rev2(a, b, a.__class__(-8, 8)),
}
bounded = st.one_of(st.floats(-8, 8), st.integers(-8, 8), st.fractions(-8, 8, max_denominator=12))


def _both(f, *args):
    with backend._use('python'):
        want = _outcome(f, *args)
    with backend._use('gmpy2'):
        got = _outcome(f, *args)
    assert got == want


@pytest.mark.parametrize('op', sorted(UNARY))
@settings(max_examples=25, deadline=None)
@given(data=st.data(), cls=st.sampled_from((MultiInterval, OutwardMultiInterval)))
def test_set_level_matches_unary(op, data, cls):
    _both(UNARY[op], data.draw(multi_intervals(cls), label='a'))


@pytest.mark.parametrize('op', sorted(BINARY))
@settings(max_examples=25, deadline=None)
@given(data=st.data(), cls=st.sampled_from((MultiInterval, OutwardMultiInterval)))
def test_set_level_matches_binary(op, data, cls):
    values = bounded if op.endswith('_rev') or op == 'pow_rev2' else set_values
    a = data.draw(multi_intervals(cls, values), label='a')
    b = data.draw(multi_intervals(cls, values), label='b')
    _both(BINARY[op], a, b)


def _roots(f):
    return lambda x: [(repr(r.interval), r.unique) for r in newton(f, x)]


@settings(max_examples=5, deadline=None)
@given(zeros=st.lists(st.one_of(st.integers(-4, 4), st.fractions(-4, 4, max_denominator=5), st.floats(-4, 4)),
                      min_size=1, max_size=3),
       cls=st.sampled_from((MultiInterval, OutwardMultiInterval)))
def test_set_level_matches_newton(zeros, cls):
    def f(t):
        out = t - zeros[0]
        for z in zeros[1:]:
            out = out * (t - z)
        return out
    _both(_roots(f), cls(-10.0, 10.0))


@pytest.mark.parametrize('cls', (MultiInterval, OutwardMultiInterval))
def test_set_level_matches_newton_sin(cls):
    _both(_roots(lambda t: t.sin() - t / 3), cls(-10.0, 10.0))


# THE SWITCH

def _spy(monkeypatch, name):
    calls = []
    real = getattr(_gmpy2, name)

    def spy(*args):
        calls.append(args)
        return real(*args)
    monkeypatch.setattr(_gmpy2, name, spy)
    return calls


def test_use_switches(monkeypatch):
    """
    the guard on every differential above: under `_use('python')` the backend is never called, under
    `_use('gmpy2')` it is (a `_use` that did nothing would compare one backend with itself and pass)
    """
    rounded, outward = _spy(monkeypatch, 'rounded'), _spy(monkeypatch, 'outward')
    before = backend.name()
    with backend._use('python'):
        assert backend.name() == 'python' and backend.fast is None
        MultiInterval(0.5).exp()
        OutwardMultiInterval(0.1) + 0.2
    assert rounded == [] and outward == []
    with backend._use('gmpy2'):
        assert backend.name() == 'gmpy2' and backend.fast is _gmpy2
        MultiInterval(0.5).exp()
        OutwardMultiInterval(0.1) + 0.2
    assert rounded and outward
    assert backend.name() == before


def test_use_restores():
    """
    `_use` puts back what was there, nested either way and on an exception: otherwise the rest of the
    suite, run after this file in one process, would silently stay on gmpy2
    """
    saved = backend.NAME, backend.fast
    for outer in ('python', 'gmpy2'):
        with backend._use(outer):
            state = backend.NAME, backend.fast
            for inner in ('python', 'gmpy2'):
                with backend._use(inner):
                    assert backend.name() == inner
                assert (backend.NAME, backend.fast) == state
                with pytest.raises(RuntimeError):
                    with backend._use(inner):
                        raise RuntimeError
                assert (backend.NAME, backend.fast) == state
    assert (backend.NAME, backend.fast) == saved


def _run(value, prelude=''):
    """`import intervals` in a fresh interpreter with INTERVALS_BACKEND=value (None: unset)"""
    env = {k: v for k, v in os.environ.items() if k != 'INTERVALS_BACKEND'}
    if value is not None:
        env['INTERVALS_BACKEND'] = value
    code = (prelude + '\ntry:\n    import intervals, intervals.backend as b\n'
            '    print(b.NAME, b.name(), "gmpy2" in sys.modules, "intervals._gmpy2" in sys.modules)\n'
            'except Exception as e:\n    print(type(e).__name__, str(e).replace("\\n", " "))\n')
    r = subprocess.run([sys.executable, '-c', 'import sys\n' + code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


BLOCKED = "sys.modules['gmpy2'] = None"


def _fake(version, mpfr='MPFR 4.2.2'):
    return ("import types\nm = types.ModuleType('gmpy2')\n"
            f"m.version = lambda: {version!r}\nm.mpfr_version = lambda: {mpfr!r}\nsys.modules['gmpy2'] = m")


@pytest.mark.parametrize('value, prelude, expected', [
    (None, '', 'python python False False'),  # the default is the pure path (M16e, D24)
    ('', '', 'python python False False'),
    ('python', '', 'python python False False'),
    ('auto', '', 'gmpy2 gmpy2 True True'),
    ('gmpy2', '', 'gmpy2 gmpy2 True True'),
    ('auto', BLOCKED, 'python python True False'),  # the None put in sys.modules is a key
    ('auto', _fake('3.0.0'), 'python python True False'),
    ('auto', _fake('2.2.9'), 'python python True False'),
    ('auto', _fake('2.3.1', 'MPFR 4.1.1'), 'python python True False'),
    ('bogus', '', 'ValueError'),
    ('Gmpy2', '', 'ValueError'),
])
def test_env_var(value, prelude, expected):
    got = _run(value, prelude)
    if expected.endswith('Error'):
        assert got.startswith(expected + ' '), got
    else:
        assert got == expected


@pytest.mark.parametrize('prelude, says', [
    (BLOCKED, 'gmpy2'),
    (_fake('2.2.9'), '2.2.9'),
    (_fake('2.3.1', 'MPFR 4.1.1'), 'MPFR 4.1.1'),
    (_fake('two point three'), 'two point three'),
])
def test_forced_gmpy2_never_falls_back(prelude, says):
    """INTERVALS_BACKEND=gmpy2 without a usable gmpy2 is an ImportError naming why, never the pure path"""
    got = _run('gmpy2', prelude)
    assert got.startswith('ImportError ') and says in got, got


@pytest.mark.parametrize('version, mpfr, auto, forced', [
    ('2.3.1', 'MPFR 4.2.2', True, True),
    ('2.3.0', 'MPFR 4.2.0', True, True),
    ('2.2.9', 'MPFR 4.2.2', False, False),
    ('2.3.0', 'MPFR 4.1.1', False, False),
    ('2.10.0', 'MPFR 4.2.2', True, True),  # not a string comparison ('2.10' < '2.3')
    ('2.3.1', 'MPFR 4.10.0', True, True),
    ('2.3.1rc1', 'MPFR 4.2.2', True, True),  # a pre-release of 2.3.1 is past 2.3.0
    ('2.3.0rc1', 'MPFR 4.2.2', False, False),  # and one of 2.3.0 is before it
    ('2.3.0.dev3', 'MPFR 4.2.2', False, False),
    ('2.3.1.post1', 'MPFR 4.2.2', True, True),
    ('2.3.1.dev1', 'MPFR 4.2.2', True, True),  # a dev build of 2.3.1 is past 2.3.0
    ('2.3.1+local', 'MPFR 4.2.2', True, True),
    ('2.3.1rc1+local', 'MPFR 4.2.2', True, True),  # a local label on a pre-release too
    ('3.0.0', 'MPFR 4.2.2', False, True),  # auto takes only the series verified; forced, any at the floor
    ('2.3.1', 'MPFR 5.0.0', True, True),
    ('2.3.1', 'MPFR 4.2.2-p1', True, True),
    ('garbage', 'MPFR 4.2.2', False, False),
    ('2.3.1', 'garbage', False, False),
    ('', '', False, False),
    ('2', 'MPFR 4.2.2', False, False),
])
def test_version_floor(version, mpfr, auto, forced):
    assert backend._supported(version, mpfr) is auto
    assert backend._supported(version, mpfr, ceiling=False) is forced


def test_the_test_extra_installs_what_auto_takes():
    """
    `[test]` pins gmpy2 to the window `auto` takes (`FLOOR <= version < CEILING`): unpinned, a gmpy2 3
    on PyPI would turn `test_env_var`'s auto row red in every CI job for a reason outside the code.
    and the gmpy2 installed here is in it
    """
    import tomllib
    extras = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['optional-dependencies']
    assert f'gmpy2>={backend.FLOOR[0]}.{backend.FLOOR[1]},<{backend.CEILING}' in extras['test']
    assert backend._supported(str(gmpy2.version()), str(gmpy2.mpfr_version()))
