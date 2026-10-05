"""hypothesis strategies shared by the test modules"""
import math
from fractions import Fraction

from hypothesis import strategies as st

from multiinterval.cuts import Cut
from multiinterval.cuts import Side
from multiinterval.kernel import normalize
from multiinterval.kernel import piece

finite_values = st.one_of(
    st.integers(-20, 20),
    st.fractions(min_value=-20, max_value=20, max_denominator=6),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
)
infinities = st.sampled_from([-math.inf, math.inf])
values = st.one_of(finite_values, infinities)
sides = st.sampled_from([Side.BELOW, Side.ABOVE])
cuts = st.builds(Cut, values, sides)

# a small shared pool makes coinciding endpoints (tiling, touching, degenerate pieces) common
pool_values = st.sampled_from([-math.inf, -2, -1, 0, Fraction(1, 2), 1, 2.0, 3, math.inf])
endpoint_values = st.one_of(pool_values, pool_values, values)


@st.composite
def piece_pairs(draw, values=endpoint_values):
    a, b = sorted((draw(values), draw(values)))
    return piece(a, b, draw(st.booleans()), draw(st.booleans()))


@st.composite
def cut_tuples(draw, values=endpoint_values, max_pieces=5):
    return normalize(draw(st.lists(piece_pairs(values), max_size=max_pieces)))


finite_cut_tuples = cut_tuples(values=st.one_of(
    st.sampled_from([-2, -1, 0, Fraction(1, 2), 1, 2.0, 3]),
    finite_values,
))

# int, Fraction and +-inf only, for laws that float rounding would break (size additivity)
exact_cut_tuples = cut_tuples(values=st.one_of(
    st.sampled_from([-math.inf, -2, -1, 0, Fraction(1, 2), 1, 3, math.inf]),
    st.integers(-20, 20),
    st.fractions(min_value=-20, max_value=20, max_denominator=6),
))


def midpoint(a, b):
    """a point strictly between a < b: the float midpoint of two adjacent doubles rounds onto one
    of them, so fall back to the exact one"""
    m = (a + b) / 2
    return m if a < m < b else (Fraction(a) + Fraction(b)) / 2


def probe_points(*cut_tuples_):
    """
    points on which two cut tuples agree iff they are the same set: every endpoint value, a
    point inside every gap between consecutive endpoint values, one beyond each end, and +-inf
    """
    finite = sorted({cut.value for cuts in cut_tuples_ for cut in cuts if math.isfinite(cut.value)})
    probes = [-math.inf, math.inf, *finite]
    if finite:
        probes += [finite[0] - 1, finite[-1] + 1]
        probes += [midpoint(a, b) for a, b in zip(finite, finite[1:])]
    else:
        probes.append(0)
    return probes


__all__ = ['finite_values', 'infinities', 'values', 'sides', 'cuts', 'piece_pairs', 'cut_tuples',
           'finite_cut_tuples', 'exact_cut_tuples', 'probe_points', 'Fraction']
