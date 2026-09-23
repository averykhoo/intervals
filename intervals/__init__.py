"""multi-intervals over the affine extended reals; see v2-plan.md for the design"""
from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import IntervalWarning

__all__ = [
    'Cut',
    'Side',
    'IntervalWarning',
    'EmptySetPropagationWarning',
    'DomainClippedWarning',
    'IndeterminateResultWarning',
]
