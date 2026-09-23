"""
warning and exception classes

empty-set propagation and domain clipping are normal events in a solver loop, so they are
ignored by default. the filters are appended to the *end* of the warnings filter list, so any
filter the user installs -- before or after importing this package -- takes precedence. to use
one as a tripwire::

    warnings.simplefilter('error', EmptySetPropagationWarning)
"""
import warnings


class IntervalWarning(UserWarning):
    """base class for every warning this package emits"""


class EmptySetPropagationWarning(IntervalWarning):
    """an operand was empty, so the result is empty (the image of an empty set)"""


class DomainClippedWarning(IntervalWarning):
    """an operation dropped input points outside its domain, e.g. `sqrt([-1, 4])`"""


class IndeterminateResultWarning(IntervalWarning):
    """the operand box is an indeterminate point, e.g. `1/[0]` or `[0] * [inf]`; the result is empty"""


warnings.filterwarnings('ignore', category=EmptySetPropagationWarning, append=True)
warnings.filterwarnings('ignore', category=DomainClippedWarning, append=True)
