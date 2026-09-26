"""multi-intervals over the affine extended reals; see v2-plan.md for the design"""
import math as _math

from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import IntervalWarning
from intervals.kernel import Builder
from intervals.kernel import Size
from intervals.multi_interval import MultiInterval
from intervals.multi_interval import OutwardMultiInterval
from intervals.reductions import dot
from intervals.reductions import sum_
from intervals.reductions import sum_abs
from intervals.reductions import sum_sqr
from intervals.relations import Allen
from intervals.relations import TruthSet
# reverse ops (M13e)
from intervals.reverse import abs_rev
from intervals.reverse import cosh_rev
from intervals.reverse import pown_rev
from intervals.reverse import sqr_rev

EMPTY = MultiInterval()
REALS = MultiInterval(-_math.inf, _math.inf)  # the affine extended reals, both infinities included

__all__ = [
    'MultiInterval',
    'OutwardMultiInterval',
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
    'HullWarning',
    'sum_',
    'sum_abs',
    'sum_sqr',
    'dot',
    # reverse ops (M13e)
    'sqr_rev',
    'abs_rev',
    'pown_rev',
    'cosh_rev',
]
