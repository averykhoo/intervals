"""hypothesis strategies shared by the test modules"""
import math
from fractions import Fraction

from hypothesis import strategies as st

from intervals.cuts import Cut
from intervals.cuts import Side

finite_values = st.one_of(
    st.integers(-20, 20),
    st.fractions(min_value=-20, max_value=20, max_denominator=6),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
)
infinities = st.sampled_from([-math.inf, math.inf])
values = st.one_of(finite_values, infinities)
sides = st.sampled_from([Side.BELOW, Side.ABOVE])
cuts = st.builds(Cut, values, sides)

__all__ = ['finite_values', 'infinities', 'values', 'sides', 'cuts', 'Fraction']
