"""
warning and exception classes

empty-set propagation and domain clipping are normal events in a solver loop, so they are
ignored by default, and so is a power of exact operands too long to build (`PowerLimitWarning`).
the filters are appended to the *end* of the warnings filter list, so any filter the user
installs -- before or after importing this package -- takes precedence. to use one as a
tripwire::

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


class HullWarning(IntervalWarning):
    """
    the exact result has too many pieces to list (`floor([0, 1e6])`, or infinitely many, as in
    `floor([0, inf))`), so its hull was returned: a superset, with no pieces missing
    """


class PowerLimitWarning(IntervalWarning):
    """
    an exact operand's power is rational but longer than `elementary.EXACT_RESULT_LIMIT` bits, so it
    was not built: its tightest float enclosure was returned, in both classes (`M(2) ** 2 ** 60` and
    `M(2) ** M(2 ** 60)` = `(MAX, inf)`). pown, pow, exp2 and
    exp10 emit it. ignored by default; `warnings.simplefilter('error', PowerLimitWarning)` makes it an
    error instead
    """


# ieee 1788's signals (M13g, D16). in 1788 each is a flag and the result is returned; here, by the
# owner's choice (2026-09-26), invalid input raises and a possibly-invalid one warns


class UndefinedOperationError(ValueError):
    """
    ieee 1788's `UndefinedOperation`: a 1788 constructor was given invalid input (`[2, 1]`, `"[ foo ]"`,
    `NaN`). a `ValueError`, as `MultiInterval(2, 1)` raises; 1788 would return empty (or NaI)
    """


class PossiblyUndefinedOperationWarning(IntervalWarning):
    """
    ieee 1788's `PossiblyUndefinedOperation`: the input may be invalid and the result is returned.
    the package's parsers are exact, so they decide validity and do not emit it today
    """


warnings.filterwarnings('ignore', category=EmptySetPropagationWarning, append=True)
warnings.filterwarnings('ignore', category=DomainClippedWarning, append=True)
warnings.filterwarnings('ignore', category=PowerLimitWarning, append=True)
