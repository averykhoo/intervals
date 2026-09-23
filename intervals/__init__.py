"""multi-intervals over the affine extended reals; see v2-plan.md for the design"""
import math as _math

from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import IntervalWarning
from intervals.kernel import Builder
from intervals.kernel import Size
from intervals.multi_interval import MultiInterval
from intervals.relations import Allen
from intervals.relations import TruthSet

EMPTY = MultiInterval()
REALS = MultiInterval(-_math.inf, _math.inf)  # the affine extended reals, both infinities included

__all__ = [
    'MultiInterval',
    'EMPTY',
    'REALS',
    'Size',
    'Builder',
    'TruthSet',
    'Allen',
    'Cut',
    'Side',
    'IntervalWarning',
    'EmptySetPropagationWarning',
    'DomainClippedWarning',
    'IndeterminateResultWarning',
]
