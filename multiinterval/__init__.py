"""multi-intervals over the affine extended reals; the design is docs/archive/v2/v2-plan.md in the repository"""
import math as _math

from multiinterval.cuts import Cut
from multiinterval.cuts import Side
from multiinterval.errors import DomainClippedWarning
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import HullWarning
from multiinterval.errors import IndeterminateResultWarning
from multiinterval.errors import IntervalWarning
from multiinterval.errors import PowerLimitWarning
from multiinterval.kernel import Builder
from multiinterval.kernel import Size
from multiinterval.multi_interval import MultiInterval
from multiinterval.multi_interval import OutwardMultiInterval
from multiinterval.reductions import dot
from multiinterval.reductions import sum_
from multiinterval.reductions import sum_abs
from multiinterval.reductions import sum_sqr
from multiinterval.relations import Allen
from multiinterval.relations import TruthSet
# reverse ops (M13e)
from multiinterval.reverse import abs_rev
from multiinterval.reverse import cosh_rev
from multiinterval.reverse import mul_rev
from multiinterval.reverse import pown_rev
from multiinterval.reverse import sqr_rev
# periodic reverse ops (M13e)
from multiinterval.reverse import cos_rev
from multiinterval.reverse import sin_rev
from multiinterval.reverse import tan_rev
# power reverse ops (M13e)
from multiinterval.reverse import pow_rev1
from multiinterval.reverse import pow_rev2
# M13g: ieee 1788's signals and bare constructors
from multiinterval.errors import PossiblyUndefinedOperationWarning
from multiinterval.errors import UndefinedOperationError
from multiinterval.literals import nums_to_interval
from multiinterval.literals import text_to_interval
# M13g: ieee 1788's decorated type and its constructors
from multiinterval.decorated import DecoratedInterval
from multiinterval.decorated import Decoration
from multiinterval.decorated import nums_to_decorated_interval
from multiinterval.decorated import set_dec
from multiinterval.decorated import text_to_decorated_interval
# M15: the solver stack's first part, autodiff and interval newton (H3)
from multiinterval.autodiff import Dual
from multiinterval.autodiff import derivative
from multiinterval.solver import Root
from multiinterval.solver import newton
# M16: the solver stack's second part, several variables (H3)
from multiinterval.autodiff import gradient
from multiinterval.autodiff import jacobian
from multiinterval.solver import RootBox
from multiinterval.solver import solve
# M8: the time layer, datetimes and timedeltas over exact seconds (D4, D30)
from multiinterval.time_interval import DateTimeInterval
from multiinterval.time_interval import NEG_INF
from multiinterval.time_interval import POS_INF
from multiinterval.time_interval import TimeDeltaInterval

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
    # M8: the time layer (D4, D30)
    'DateTimeInterval',
    'TimeDeltaInterval',
    'NEG_INF',
    'POS_INF',
]
