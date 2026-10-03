"""multi-intervals over the affine extended reals; see v2-plan.md for the design"""
import math as _math

from intervals.cuts import Cut
from intervals.cuts import Side
from intervals.errors import DomainClippedWarning
from intervals.errors import EmptySetPropagationWarning
from intervals.errors import HullWarning
from intervals.errors import IndeterminateResultWarning
from intervals.errors import IntervalWarning
from intervals.errors import PowerLimitWarning
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
from intervals.reverse import mul_rev
from intervals.reverse import pown_rev
from intervals.reverse import sqr_rev
# periodic reverse ops (M13e)
from intervals.reverse import cos_rev
from intervals.reverse import sin_rev
from intervals.reverse import tan_rev
# power reverse ops (M13e)
from intervals.reverse import pow_rev1
from intervals.reverse import pow_rev2
# M13g: ieee 1788's signals and bare constructors
from intervals.errors import PossiblyUndefinedOperationWarning
from intervals.errors import UndefinedOperationError
from intervals.literals import nums_to_interval
from intervals.literals import text_to_interval
# M13g: ieee 1788's decorated type and its constructors
from intervals.decorated import DecoratedInterval
from intervals.decorated import Decoration
from intervals.decorated import nums_to_decorated_interval
from intervals.decorated import set_dec
from intervals.decorated import text_to_decorated_interval
# M15: the solver stack's first part, autodiff and interval newton (H3)
from intervals.autodiff import Dual
from intervals.autodiff import derivative
from intervals.solver import Root
from intervals.solver import newton
# M16: the solver stack's second part, several variables (H3)
from intervals.autodiff import gradient
from intervals.autodiff import jacobian
from intervals.solver import RootBox
from intervals.solver import solve

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
    'PowerLimitWarning',
    'sum_',
    'sum_abs',
    'sum_sqr',
    'dot',
    # reverse ops (M13e)
    'sqr_rev',
    'abs_rev',
    'pown_rev',
    'cosh_rev',
    'mul_rev',
    # periodic reverse ops (M13e)
    'sin_rev',
    'cos_rev',
    'tan_rev',
    # power reverse ops (M13e)
    'pow_rev1',
    'pow_rev2',
    # M13g: ieee 1788's signals and bare constructors
    'UndefinedOperationError',
    'PossiblyUndefinedOperationWarning',
    'text_to_interval',
    'nums_to_interval',
    # M13g: ieee 1788's decorated type and its constructors
    'DecoratedInterval',
    'Decoration',
    'set_dec',
    'text_to_decorated_interval',
    'nums_to_decorated_interval',
    # M15: the solver stack's first part, autodiff and interval newton (H3)
    'Dual',
    'derivative',
    'newton',
    'Root',
    # M16: the solver stack's second part, several variables (H3)
    'gradient',
    'jacobian',
    'solve',
    'RootBox',
]
