"""
pown with a huge integral exponent (pown-huge, 2026-09-29)

the outward class used to build the exact power `Fraction(x) ** n` of a float corner, so
`O(0.5) ** (2 ** 31 - 1)` never finished, and a base just above 1 never saturates, so no range check
alone could stop it. the outward descriptor (`ops._power_descriptor(n, True)`) now builds the exact
power only while `elementary.exact_pow` does (at most `EXACT_POWER_LIMIT` bits) and otherwise rounds
with `elementary.rounded_pow`, the route 1788's pow already takes, with a marker equal to nothing for
attainment: a power that long is neither a double nor a midpoint (`ops._NOT_A_DOUBLE`)

* the reproductions run in one subprocess under a time bound, so a regression is a red, not a hang
* the exact-double boundaries: a power that is a double keeps its flag, a moved one is open. a
  degenerate result is closed whatever the flags say (`applicator.evaluate_box` keeps the point of a
  squeezed piece), so the flag rows also take a piece with width
* at the marker's boundary (`EXACT_POWER_LIMIT < |n| bits(x) <= 4 EXACT_POWER_LIMIT`) the exact
  power is still affordable here, so it is the oracle
* below it, the descriptor is today's construction (`ops.outward` of the exact descriptor) exactly
* past it, MPFR decides (points, both directions, closed iff exact)

and the owner's answers of 2026-10-03 (`references/owner-questions-2026-10-03/pown.md`), one rule in
`ops._power_descriptor`:

* Q17: an int or Fraction corner builds its power only up to `elementary.EXACT_RESULT_LIMIT` bits, the
  limit pow and exp2/exp10 share; past it the tightest open float enclosure in both classes, as pow and an
  irrational value give, with a `PowerLimitWarning` (ignored by default). `M(2) ** 2 ** 60` and
  `O(0.5, 2) ** 2 ** 40` (the 2 an int) used to hang: reproductions above, the limit's boundary, and pown
  against pow on exact points below
* Q18: to nearest a float corner is correctly rounded for every n (the exact power rounded once, or
  `rounded_pow` to nearest), no longer python's `float ** int`, which is libm's `pow` and misses by an
  ulp on hard cases (`M(1.0026606152364441) ** 13`); MPFR decides over every n, not only past 2 ** 53
"""
import math
import subprocess
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from multiinterval import DecoratedInterval
from multiinterval import IndeterminateResultWarning
from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval import PowerLimitWarning
from multiinterval import elementary
from multiinterval import kernel
from multiinterval import ops
from multiinterval.autodiff import Dual
from multiinterval.applicator import apply_unary
from multiinterval.cuts import above
from multiinterval.cuts import below
from multiinterval.rounding import DOWN
from multiinterval.rounding import MAX
from multiinterval.rounding import NEAREST
from multiinterval.rounding import UP
from multiinterval.rounding import round_rational

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
TINY = math.ulp(0.0)
ROOT = Path(__file__).resolve().parent.parent


# THE REPRODUCTIONS, UNDER A TIME BOUND

# (the expression, the expected repr's expression): each but one hung before the fix (2026-09-29, more
# than 15 s each on a loaded laptop, the review's and the designer's reproductions) and takes about
# 1 ms after it. the one is `O(-1.0) ** (2 ** 60 + 1)`, a parity row: `Fraction(-1) ** n` is cheap,
# so the old code gave the same `[-1.0]` in 0.17 s (the pown-huge review, SAB-4 and F1)
REPRODUCTIONS = [
    ('O(0.5) ** (2 ** 31 - 1)', "O.parse('(0.0, 5e-324)')"),
    ('O(0.5, 1) ** (2 ** 31 - 1)', "O.parse('(0.0, 1]')"),
    ('O(2.0) ** 2 ** 60', "O.parse('(1.7976931348623157e+308, inf)')"),
    ('O(1e300) ** 2 ** 60', "O.parse('(1.7976931348623157e+308, inf)')"),
    ('O(1.5) ** 2 ** 14000', "O.parse('(1.7976931348623157e+308, inf)')"),  # under 4300 digits
    ('O(0.5) ** 1e20', "O.parse('(0.0, 5e-324)')"),  # an integral float exponent is the int
    ('O(1.0000000000000002) ** 10 ** 9', "O.parse('(1.0000002220446296, 1.0000002220446298)')"),
    ('O(-1.0) ** (2 ** 60 + 1)', "O.parse('[-1.0]')"),
    ('O(0.5) ** -(2 ** 31 - 1)', "O.parse('(1.7976931348623157e+308, inf)')"),
    ('O(-0.5) ** -(2 ** 31 - 1)', "O.parse('(-inf, -1.7976931348623157e+308)')"),
    ('O(-0.5, 0.25) ** -(2 ** 31)', "O.parse('(1.7976931348623157e+308, inf]')"),  # the pole at 0
    ('O(-INF, -2.0) ** -(2 ** 61 + 1)', "O.parse('(-5e-324, 0]')"),
    ('O(0.0, 0.5) ** -(2 ** 61)', "O.parse('(1.7976931348623157e+308, inf]')"),
    ('O(-1.0, 0.5) ** (2 ** 1000 + 1)', "O.parse('[-1.0, 5e-324)')"),  # -1 is a double: closed
    ('O(1.0, 1.0000000000000002) ** 10 ** 30', "O.parse('[1.0, inf)')"),
    ('I.pown(I.Interval(0.5, 1), 2 ** 31 - 1)', 'I.Interval(0.0, 1.0)'),
    ('I.Interval(0.5, 1) ** 1e300', 'I.Interval(0.0, 1.0)'),
    ('I.pown(I.Interval(-2.0, 3.0), 2 ** 40)', 'I.Interval(0.0, INF)'),
    ('D(O(0.5, 1)) ** (2 ** 31 - 1)', "D(O.parse('(0.0, 1]'), Decoration.COM)"),
    ('D(O(-0.5, 0.25)) ** -(2 ** 31 + 1)',
     "D(O.parse('{ [-inf, -1.7976931348623157e+308) , (1.7976931348623157e+308, inf] }'), Decoration.TRV)"),
    ('Dual.variable(O(2.0)) ** 2.0 ** 60',
     "Dual(O.parse('(1.7976931348623157e+308, inf)'), O.parse('(1.7976931348623157e+308, inf)'))"),
    ('Dual.variable(O(-2.0)) ** 2 ** 60',
     "Dual(O.parse('(1.7976931348623157e+308, inf)'), O.parse('(-inf, -1.7976931348623157e+308)'))"),
    ('np.power(O(0.5, 1), 2 ** 31 - 1) if np else O.parse("(0.0, 1]")', "O.parse('(0.0, 1]')"),
    # exact operands (Q17, 2026-10-03): each hung before, building the exact power
    ('O(0.5, 2) ** 2 ** 40', "O.parse('(0.0, inf)')"),  # the 2 is an int, as `O.parse('[0.5, 2]')` stores it
    ('O.parse("[0.5, 2]") ** 2 ** 40', "O.parse('(0.0, inf)')"),
    ('O(2) ** 2 ** 60', "O.parse('(1.7976931348623157e+308, inf)')"),
    ('M(2) ** 2 ** 60', "M.parse('(1.7976931348623157e+308, inf)')"),  # the enclosure in both classes (Q17 (c))
    ('M(2) ** 1e300', "M.parse('(1.7976931348623157e+308, inf)')"),  # an integral float exponent is the int
    ('M(Fraction(1, 3)) ** 2 ** 40', "M.parse('(0.0, 5e-324)')"),
    ('O(Fraction(-1, 3)) ** -(2 ** 40 + 1)', "O.parse('(-inf, -1.7976931348623157e+308)')"),
    # the derivative 2 ** 60 * (MAX, inf) is float arithmetic to nearest: [inf]
    ('Dual.variable(M(2)) ** 2 ** 60', "Dual(M.parse('(1.7976931348623157e+308, inf)'), M.parse('[inf]'))"),
    ('D(M(2)) ** 2 ** 60', "D(M.parse('(1.7976931348623157e+308, inf)'), Decoration.DAC)"),
]

_PRELUDE = '''
import math, warnings
warnings.simplefilter('ignore')
from fractions import Fraction
from multiinterval import MultiInterval as M, OutwardMultiInterval as O, DecoratedInterval as D, Decoration
import multiinterval.ieee1788 as I
from multiinterval.autodiff import Dual
try:
    import numpy as np
except ImportError:
    np = None
INF = math.inf
'''


def test_reproductions_finish():
    """
    one interpreter evaluates every reproduction under one time bound (a hang is a
    `TimeoutExpired`, red), the parent compares the reprs
    """
    code = _PRELUDE + ''.join(f'print(repr({expr}))\n' for expr, _ in REPRODUCTIONS)
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    got = r.stdout.splitlines()
    namespace = {}
    exec(_PRELUDE, namespace)
    expected = [repr(eval(want, namespace)) for _, want in REPRODUCTIONS]
    assert len(got) == len(expected), r.stdout
    for (expr, _), g, e in zip(REPRODUCTIONS, got, expected):
        assert g == e, expr


# THE EXACT-DOUBLE BOUNDARIES: a power that is a double keeps its flag, a moved one is open

@pytest.mark.parametrize('a, n, want', [
    (O(2.0), 1023, O.parse('[8.98846567431158e+307]')),
    (O(2.0), 1024, O.parse('(1.7976931348623157e+308, inf)')),
    (O(0.5), 1074, O.parse('[5e-324]')),
    (O(0.5), 1075, O.parse('(0.0, 5e-324)')),
    (O(2.0), -1074, O.parse('[5e-324]')),
    (O(2.0), -1075, O.parse('(0.0, 5e-324)')),
    (O(3.0), 33, O.parse('[5559060566555523.0]')),  # 3 ** 33 < 2 ** 53
    (O(3.0), 34, O.parse('(16677181699666568.0, 16677181699666570.0)')),
    (O(-3.0), 33, O.parse('[-5559060566555523.0]')),
    (O(4.0), -537, O.parse('[5e-324]')),
    (O(2.0 ** -537), 2, O.parse('[5e-324]')),
    (O(1.0), -(2 ** 1000), O.parse('[1.0]')),
    (O(-1.0), 1e300, O.parse('[1.0]')),
    (O(-1.0), 2 ** 1000 + 1, O.parse('[-1.0]')),
    (O(0.0), 10 ** 400, O.parse('[0.0]')),
    # with width, where a flag is not forced: a power of two in range stays closed at any n
    (O(0.5, 1.0), 1074, O.parse('[5e-324, 1.0]')),
    (O(0.5, 1.0), 1075, O.parse('(0.0, 1.0]')),
    (O(2.0, 4.0), 500, O.parse('[3.273390607896142e+150, 1.0715086071862673e+301]')),
])
def test_exact_double_boundaries(a, n, want):
    assert repr(a ** n) == repr(want)


# THE MARKER AT ITS BOUNDARY: the exact power is the oracle

def _bits(x: float) -> int:
    f = abs(Fraction(x))
    return max(f.numerator.bit_length(), f.denominator.bit_length())


def _is_double(m: int, e: int, n: int) -> bool:
    """
    is `(m 2**e) ** n` a double, for an odd m >= 1? O(1) once `n log2 m` passes 53: an odd m >= 3
    to a power that long is an odd part over 2**53, and to a negative power a non-dyadic value
    (design A's criterion, 2026-09-29, kept as an oracle independent of `exact_pow`)
    """
    if m == 1:
        return -1074 <= e * n <= 1023
    if n < 0 or n * (m.bit_length() - 1) >= 53:
        return False
    big, e = m ** n, e * n
    return big.bit_length() <= 53 and e >= -1074 and big.bit_length() + e <= 1024


def _split(x: float):
    """|x| = m 2**e, m odd"""
    num, den = abs(x).as_integer_ratio()
    zeros = (num & -num).bit_length() - 1
    return num >> zeros, zeros - (den.bit_length() - 1)


def _piece(a):
    pieces = list(kernel.pieces(a._cuts))
    assert len(pieces) == 1, pieces
    return pieces[0]


def _exact_piece(x: float, n: int):
    """the tightest float piece around the exact `x ** n`, closed iff it is a double"""
    v = Fraction(x) ** n
    lo, hi = round_rational(v, DOWN), round_rational(v, UP)
    exact = lo == hi
    return lo, exact, hi, exact


FLOATS = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x != 0),
    st.integers(-1074, 1023).map(lambda e: 2.0 ** e),
    st.integers(1, 2 ** 20).map(lambda k: 1 + k * 2.0 ** -52),
    st.integers(1, 2 ** 20).map(lambda k: 1 - k * 2.0 ** -53),
    st.integers(1, 2 ** 40).map(lambda k: k * TINY),
    st.sampled_from([1.0, 3.0, 0.1, 1.5, MAX, 2.2250738585072014e-308]),
).flatmap(lambda x: st.sampled_from([x, -x]))


@st.composite
def boundary_cases(draw):
    x = draw(FLOATS)
    b = _bits(x)
    n = draw(st.integers(100000 // b + 1, 400000 // b)) * draw(st.sampled_from([1, -1]))
    return x, n


@settings(max_examples=60, deadline=None)
@given(boundary_cases())
@example((1.0, 100001))
@example((-1.0, 100001))
@example((-1.0, -100002))
@example((0.5, 50000))  # 0.5 has 2 bits: the last n it is built for (50000 * 2 = the limit)
@example((0.5, 50001))  # and the first past it, a marker
@example((0.5, 100000))
@example((TINY, 94))
@example((-3.0, 50001))
def test_marker_boundary(case):
    x, n = case
    got = _piece(O(x) ** n)
    assert got == _exact_piece(x, n), (x, n)
    if not got[1]:
        m, e = _split(x)
        assert not _is_double(m, e, n), (x, n)


def test_the_marker_boundary_of_one_half():
    """
    `exact_pow` declines past `|n| max(bitlen(num), bitlen(den)) = EXACT_POWER_LIMIT`, so 0.5 (2
    bits) is built up to n = 50000, not 100000 (an `@example` comment above said "just under the
    limit" of (0.5, 100000), the pown-huge review's F2)
    """
    limit = elementary.EXACT_POWER_LIMIT
    assert elementary.exact_pow(Fraction(1, 2), limit // 2) == Fraction(1, 2 ** (limit // 2))
    assert elementary.exact_pow(Fraction(1, 2), limit // 2 + 1) is None
    assert elementary.exact_pow(Fraction(1, 2), limit) is None


@pytest.mark.parametrize('limit, holds', [
    (elementary.EXACT_POWER_LIMIT, True), (elementary.EXACT_RESULT_LIMIT, True), (36550, True), (36549, False),
    (2150, False), (2000, False)])
def test_the_marker_proof_premises(limit, holds):
    """
    `ops._NotADouble`'s proof needs `EXACT_POWER_LIMIT >= 36550`: below it a marker could stand for a
    double, and the rounding hooks' ziv loop stalled rather than failed (the review's SAB-3, the limit
    at 2000). `ops` checks it at import, so a lowered limit is an import error, loud and at once
    """
    if holds:
        ops._check_marker_premises(limit)
    else:
        with pytest.raises(RuntimeError, match='EXACT_POWER_LIMIT'):
            ops._check_marker_premises(limit)


def test_a_corner_power_is_built_once(monkeypatch):
    """
    a box asks for a corner's value up to four times (fn, both hooks, attainment); the outward
    descriptor's `float_exact` cache builds each corner's exact power once. near the limit that is
    about 5x the cost without it (the pown-huge review's SAB-5, the cache at `maxsize=0`)
    """
    calls = []
    real = elementary.exact_pow

    def spy(x, y):
        calls.append((x, y))
        return real(x, y)

    n = 1879  # 1.2 and 1.3 have 53 bits, and 53 x 1879 is under the limit: both corners are built
    ops._power_descriptor.cache_clear()
    monkeypatch.setattr(elementary, 'exact_pow', spy)
    a = O(1.2, 1.3) ** n
    assert sorted(calls) == [(Fraction(1.2), n), (Fraction(1.3), n)], calls
    assert _piece(a) == (round_rational(Fraction(1.2) ** n, DOWN), False,
                         round_rational(Fraction(1.3) ** n, UP), False)


# BELOW THE LIMIT: today's construction, exactly

def _typed(cuts):
    return [(type(c.value), c.value, c.side) for c in cuts]


ENDS = st.one_of(
    FLOATS,
    st.sampled_from([0.0, -0.0, 0, 1, -1, 2, Fraction(1, 3), Fraction(-7, 5), INF, -INF]),
)


@st.composite
def outward_sets(draw):
    pieces = []
    for _ in range(draw(st.integers(1, 3))):
        a, b = sorted((draw(ENDS), draw(ENDS)))
        if a == b:
            pieces.append((below(a), above(a)))
            continue
        pieces.append(kernel.piece(a, b, draw(st.booleans()), draw(st.booleans())))
    return kernel.normalize(pieces)


EXPONENTS = st.one_of(st.integers(1, 400), st.integers(1800, 4000)).flatmap(lambda k: st.sampled_from([k, -k]))


@settings(max_examples=60, deadline=None)
@given(outward_sets(), st.lists(EXPONENTS, min_size=3, max_size=3))
def test_identity_with_the_exact_construction(a, exponents):
    """
    three n a set, so a cache that outlives its n shows. the reference builds the exact power of
    every float corner (the 1800..4000 band crosses `EXACT_POWER_LIMIT` for a 53-bit mantissa)
    """
    for n in exponents:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = ops.power(a, n, True)
            want = apply_unary(ops.outward(ops._exact_power_descriptor(n)), a)
        assert _typed(got) == _typed(want), (a, n)


# PAST IT: MPFR

def _mpfr_piece(x: float, n: int):
    gmpy2 = pytest.importorskip('gmpy2')
    wide = gmpy2.context()
    ends = []
    for direction in (gmpy2.RoundDown, gmpy2.RoundUp):
        ctx = gmpy2.ieee(64)
        ctx.round = direction
        r = ctx.pow(gmpy2.mpfr(x, 53, wide), gmpy2.mpfr(n, max(2, abs(n).bit_length()), wide))
        ends.append((float(r) + 0.0, r.rc == 0))
    (lo, exact), (hi, _) = ends
    return lo, exact, hi, exact


HUGE = st.one_of(
    st.integers(2 ** 31, 2 ** 40),
    st.integers(1, 2 ** 64),
    st.tuples(st.integers(40, 70), st.integers(-2, 2)).map(lambda t: 2 ** t[0] + t[1]),
    st.integers(20, 300).map(lambda j: int(10.0 ** j)),
).flatmap(lambda k: st.sampled_from([k, -k]))


@settings(max_examples=150, deadline=None)
@given(FLOATS, HUGE)
@example(-1.0000000000000002, 2 ** 60 + 1)
@example(-1.0000000000000002, -(2 ** 60 + 1))
@example(-0.9999999999999999, 2 ** 53 + 1)
@example(-1.0, 2 ** 60 + 1)
@example(1.0000000000000002, 10 ** 9)
def test_huge_exponent_against_mpfr(x, n):
    assert _piece(O(x) ** n) == _mpfr_piece(x, n), (x, n)


# EXACT CORNERS AT AND PAST EXACT_RESULT_LIMIT (Q17, owner 2026-10-03)

LIMIT = elementary.EXACT_RESULT_LIMIT
K = LIMIT // 2  # 2 and 1/2 have 2 bits: 2 ** K and 2 ** -K are the last powers of 2 built


def test_exact_corners_at_the_limit():
    """
    `exact_power_bits` (|n| times the longer of numerator and denominator in bits) at the limit builds,
    one past rounds to the tightest open enclosure in both classes, with the warning. before, every row built its exact power (an int of 2**21 bits, cheap for a power of 2)
    """
    for cls in (M, O):
        assert (cls(2) ** K).cuts == cls(2 ** K).cuts
        assert (cls(Fraction(-1, 2)) ** K).cuts == cls(Fraction(1, 2 ** K)).cuts
    big, tiny = f'{MAX!r}', f'{TINY!r}'
    for a, n, enclosure in [
        (2, K + 1, f'({big}, inf)'),
        (-2, K + 1, f'(-inf, -{big})'),  # K + 1 is odd
        (Fraction(1, 2), K + 1, f'(0.0, {tiny})'),
        (-2, -(K + 1), f'(-{tiny}, 0.0)'),
        (Fraction(7, 5), -(LIMIT // 3 + 1), f'(0.0, {tiny})'),
    ]:
        for cls in (O, M):
            with pytest.warns(PowerLimitWarning, match='longer than 4194304 bits, so its tightest float enclosure'):
                assert repr(cls(a) ** n) == repr(cls.parse(enclosure)), (cls, a, n)


def test_an_exact_corner_past_the_limit_inside_the_float_range():
    """
    the marker route for an exact corner whose value is a finite double's neighbour: x has 31 bits, so
    135301 is the first n past the limit; the exact power (4.2M bits, about 0.25 s here) is the oracle
    """
    x = 1 + Fraction(1, 2 ** 30)
    n = LIMIT // 31 + 1
    v = x ** n
    down, up = (round_rational(v, d) for d in (DOWN, UP))
    assert down < up
    for cls in (O, M):  # the enclosure in both classes (Q17 (c))
        with pytest.warns(PowerLimitWarning):
            assert _piece(cls(x) ** n) == (down, False, up, False), cls
    with pytest.warns(PowerLimitWarning):
        assert _piece(O(-x) ** n) == (-up, False, -down, False)  # n odd


EXACT_POINTS = [(3, 70000), (2, 50001), (10, 25001), (2, K), (Fraction(2, 3), -40000)]


@pytest.mark.parametrize('x, n', EXACT_POINTS + [
    (2, K + 1), (Fraction(1, 3), 2 ** 40), (Fraction(2, 3), -(2 ** 30)), (7, 10 ** 30), (Fraction(-5, 3), 2 ** 40 + 1)])
def test_pown_matches_pow_on_exact_points(x, n):
    """
    `A ** n` and `A ** O(n)` are pown and pow, two routes that now share one limit: `O(3) ** 70000` was the
    exact int while `O(3) ** O(70000)` was `(MAX, inf)` (pow's limit was 100000 bits), and past the
    limit pown built the power (`O(Fraction(1, 3)) ** 2 ** 40` hung). pow takes no negative base: a
    negative one is checked as `-(|x| ** n)` for odd n
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PowerLimitWarning)
        got = O(x) ** n
        want = O(abs(x)) ** O(n)
    assert got == (want if x > 0 or n % 2 == 0 else -want), (x, n)


@pytest.mark.parametrize('x, n', EXACT_POINTS)
def test_nearest_pown_matches_pow_on_exact_points(x, n):
    """within the limit both classes build the same exact power, and past it both give the enclosure
    (`M(2) ** 2 ** 60` and `M(2) ** M(2 ** 60)` are `(MAX, inf)`)"""
    got = M(x) ** n
    assert got == M(x) ** M(n) and not any(isinstance(c.value, float) for c in got.cuts), (x, n)


def test_one_limit_for_pown_pow_exp2_exp10():
    """
    exp2 and exp10 measure `2 ** x` and `10 ** x` as pown does (`exact_power_bits` of the base and x):
    `M(2 ** 21).exp2()` and `M(2 ** 20).exp10()` are exact, one more rounds. exp2/exp10 used to stop at an
    exponent of 100000 (`M(100000).exp10()` built 332193 bits while `M(10) ** M(25001)` rounded)
    """
    for base, name, x in ((2, 'exp2', K), (10, 'exp10', LIMIT // 4)):
        exact = base ** x
        for got in (getattr(M(x), name)(), M(base) ** M(x), M(base) ** x, O(base) ** x):
            assert [type(c.value) for c in got.cuts] == [int, int] and got.cuts[0].value == exact, (name, x)
        for result in (lambda: getattr(M(x + 1), name)(), lambda: M(base) ** M(x + 1), lambda: O(base) ** (x + 1)):
            with pytest.warns(PowerLimitWarning):
                assert _piece(result()) == (MAX, False, INF, False), name


@pytest.mark.parametrize('expr, match', [
    ('M(2) ** 2 ** 60', 'pow<a 61-bit int>: .* tightest float enclosure'),
    ('O(2, 3) ** -(2 ** 40)', 'pow-1099511627776: .* tightest float enclosure'),
    ('M(2) ** M(2 ** 60)', 'pow: .* tightest float enclosure'),
    ('O(Fraction(1, 3)) ** O(2 ** 40)', 'pow: '),
    ('M(2 ** 60).exp2()', 'exp2: '),
    ('O(-(2 ** 60)).exp10()', 'exp10: '),
])
def test_the_power_limit_warning(expr, match):
    with pytest.warns(PowerLimitWarning, match=match):
        eval(expr, {'M': M, 'O': O, 'Fraction': Fraction})


@pytest.mark.parametrize('expr', [
    'M(2.0) ** 2 ** 60', 'M(2.0) ** M(2 ** 60)', 'M(2 ** 60.0).exp2()',  # a float operand: rounded anyway
    'M(2) ** M(Fraction(1, 2))', 'M(2).sqrt()',  # irrational: no limit involved
    'M(1, 2) ** 0', 'M(-1, 1) ** 2 ** 60', 'M(0) ** 2 ** 60',  # 0 and ±1 are always built
])
def test_no_power_limit_warning(expr):
    """the suite makes the library's warnings errors, so each of these would raise if it warned"""
    eval(expr, {'M': M, 'Fraction': Fraction})


def test_the_power_limit_warning_is_ignored_by_default_and_can_be_an_error():
    code = (
        'import warnings\n'
        'from multiinterval import MultiInterval as M, PowerLimitWarning\n'
        'with warnings.catch_warnings(record=True) as w:\n'
        '    M(2) ** 2 ** 60\n'
        'print(len(w))\n'
        'warnings.simplefilter("error", PowerLimitWarning)\n'
        'try:\n'
        '    M(2) ** 2 ** 60\n'
        'except PowerLimitWarning:\n'
        '    print("raised")\n')
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ['0', 'raised']


# TO NEAREST, CORRECTLY ROUNDED FOR EVERY n (Q18, owner 2026-10-03)

@pytest.mark.parametrize('x, n, want', [
    # libm's `pow` (python's `float ** int`) on this laptop's UCRT gave the neighbour of each (2026-10-03,
    # `references/owner-questions-2026-10-03/pown.md` §3): 1.32750153854842, 1.0351455727723202,
    # 4.45014878383046e-308, 4425.378458811314
    (1.0287349703833546, 10, 1.3275015385484197),
    (1.0026606152364441, 13, 1.03514557277232),
    (2.1095375758280437e-154, 2, 4.450148783830459e-308),
    (1.3811118839148833, 26, 4425.378458811315),
])
def test_nearest_is_correctly_rounded(x, n, want):
    assert float(Fraction(x) ** n) == want  # int / int is correctly rounded in CPython: the oracle
    for sign in (1, -1):
        assert repr(M(sign * x) ** n) == repr(M(sign ** n * want)), (sign, x, n)


def _nearest(x: float, n: int) -> float:
    """the exact `x ** n` rounded to nearest by CPython's correctly rounded int division"""
    v = Fraction(x) ** n
    try:
        return float(v) + 0.0
    except OverflowError:
        return INF if v > 0 else -INF


@settings(max_examples=100, deadline=None)
@given(FLOATS, st.integers(1, 300).flatmap(lambda k: st.sampled_from([k, -k])))
@example(1.0026606152364441, 13)
@example(-1.0287349703833546, 10)
@example(5.155830884225402, -3)  # one rounding, not `1 / x ** 3` (M14-breadth)
def test_nearest_against_the_exact_power(x, n):
    assert repr(M(x) ** n) == repr(M.parse(f'[{_nearest(x, n)!r}]')), (x, n)


# TO NEAREST PAST 2 ** 53: python's `float ** int` rounds the exponent to a double there

@pytest.mark.parametrize('a, n, want', [
    (M(0.5), 10 ** 400, '[0.0]'),  # was [inf]: python could not convert the int
    (M(0.5), -(10 ** 400), '[inf]'),
    (M(1.0), 10 ** 400, '[1.0]'),
    (M(-1.0), 2 ** 60 + 1, '[-1.0]'),  # was [1.0]: 2 ** 60 + 1 rounds to an even double
    (M(-1.0), -(2 ** 60 + 1), '[-1.0]'),
    (M(1.0000000000000002), 2 ** 53 + 1, '[7.38905609893065]'),  # was 7.389056098930649, at 2 ** 53
    (M(-1.0000000000000002), 2 ** 53 + 1, '[-7.38905609893065]'),  # was positive
    (M(-1.0000000000000002), 2 ** 53, '[7.389056098930649]'),  # correctly rounded (MPFR), as python's here
    (M(0.0), 10 ** 400, '[0.0]'),
    (M(-0.0), 10 ** 400 + 1, '[0.0]'),
    (M(0.0, 0.5), 10 ** 400, '[0.0]'),  # a wrong set, [0.0, inf], with the zero sent to python
    (M(-2.0, 0.0), -(10 ** 400 + 1), '[-inf, 0.0]'),  # the pole at the closed 0
    # an exact int corner stays the exact descriptor's (an int, not rounded_pow's float: the review's SAB-2)
    (M(1), 10 ** 400, '[1]'),
    (M(-1), 2 ** 60 + 1, '[-1]'),
    (M(-1), -(2 ** 60 + 1), '[-1]'),
    (M(-1, 0.5), 2 ** 60 + 1, '[-1, 0.0]'),
])
def test_nearest_past_2_53(a, n, want):
    got, want = a ** n, M.parse(want)
    assert repr(got) == repr(want)
    assert [type(c.value) for c in got._cuts] == [type(c.value) for c in want._cuts]


def test_nearest_zero_to_a_huge_negative_power_is_empty():
    with pytest.warns(IndeterminateResultWarning):
        assert M(0.0) ** -(10 ** 400) == M()


PAST_2_53 = st.one_of(
    st.integers(2 ** 53 + 1, 2 ** 64),
    st.tuples(st.integers(54, 70), st.integers(-2, 2)).map(lambda t: 2 ** t[0] + t[1]),
    st.integers(16, 300).map(lambda j: int(10.0 ** j)),
).flatmap(lambda k: st.sampled_from([k, -k]))
# every n since Q18 (2026-10-03): to nearest is one rule, so MPFR decides below 2 ** 53 too
EVERY_N = st.one_of(PAST_2_53, EXPONENTS, st.integers(2 ** 31, 2 ** 53).flatmap(lambda k: st.sampled_from([k, -k])))


@settings(max_examples=100, deadline=None)
@given(FLOATS, EVERY_N)
@example(1.0026606152364441, 13)
@example(2.1095375758280437e-154, 2)
@example(-1.0000000000000002, 2 ** 53 + 1)
@example(-1.0, -(2 ** 60 + 1))
def test_nearest_huge_exponent_against_mpfr(x, n):
    gmpy2 = pytest.importorskip('gmpy2')
    ctx = gmpy2.ieee(64)
    ctx.round = gmpy2.RoundToNearest
    want = float(ctx.pow(gmpy2.mpfr(x, 53, gmpy2.context()), gmpy2.mpfr(n, abs(n).bit_length(), gmpy2.context()))) + 0.0
    assert repr(M(x) ** n) == repr(M.parse(f'[{want!r}]')), (x, n)


# AN EXPONENT PAST 4300 DIGITS: the descriptor's name was `f'pow{n}'`, which python refuses to write

def test_an_exponent_past_4300_digits():
    big = '1.7976931348623157e+308'
    # 10 ** 4300 is the smallest n python refuses to write (4301 digits); 2 ** 20000 alone let the
    # name's threshold move up to 10 ** 6020 unseen (the pown-huge review's SAB-1)
    for n in (10 ** 4300, 2 ** 20000):
        assert repr(O(1.5) ** n) == repr(O.parse(f'({big}, inf)'))
        assert repr(M(0.5) ** n) == repr(M.parse('[0.0]'))
    y = Dual.variable(O(1.5)) ** 2 ** 20000
    assert y.value == y.derivative == O.parse(f'({big}, inf)')
    with pytest.warns(IndeterminateResultWarning, match='pow-<a 20001-bit int>'):
        assert O(0.0) ** -(2 ** 20000) == O()


# POWN AGAINST 1788'S POW (D11): two routes to the same `rounded_pow`

@settings(max_examples=60, deadline=None)
@given(FLOATS.map(abs), st.one_of(HUGE, EXPONENTS), st.booleans())
def test_pown_matches_pow(x, n, one_ulp):
    a = O(x, math.nextafter(x, INF)) if one_ulp and x < MAX else O(x)
    assert repr(a ** n) == repr(a ** O(n)), (a, n)
