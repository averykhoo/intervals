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

from intervals import DecoratedInterval
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import kernel
from intervals import ops
from intervals.applicator import apply_unary
from intervals.cuts import above
from intervals.cuts import below
from intervals.rounding import DOWN
from intervals.rounding import MAX
from intervals.rounding import UP
from intervals.rounding import round_rational

M, O = MultiInterval, OutwardMultiInterval
INF = math.inf
TINY = math.ulp(0.0)
ROOT = Path(__file__).resolve().parent.parent


# THE REPRODUCTIONS, UNDER A TIME BOUND

# (the expression, the expected repr's expression): each hung before the fix (2026-09-29, more than
# 15 s each on a loaded laptop, the review's and the designer's reproductions) and takes about 1 ms
# after it
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
]

_PRELUDE = '''
import math, warnings
warnings.simplefilter('ignore')
from intervals import OutwardMultiInterval as O, DecoratedInterval as D, Decoration
import intervals.ieee1788 as I
from intervals.autodiff import Dual
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
@example((0.5, 100000))  # just under the limit: 2 bits
@example((TINY, 94))
@example((-3.0, 50001))
def test_marker_boundary(case):
    x, n = case
    got = _piece(O(x) ** n)
    assert got == _exact_piece(x, n), (x, n)
    if not got[1]:
        m, e = _split(x)
        assert not _is_double(m, e, n), (x, n)


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


# POWN AGAINST 1788'S POW (D11): two routes to the same `rounded_pow`

@settings(max_examples=60, deadline=None)
@given(FLOATS.map(abs), st.one_of(HUGE, EXPONENTS), st.booleans())
def test_pown_matches_pow(x, n, one_ulp):
    a = O(x, math.nextafter(x, INF)) if one_ulp and x < MAX else O(x)
    assert repr(a ** n) == repr(a ** O(n)), (a, n)
