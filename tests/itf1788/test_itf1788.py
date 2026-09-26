"""
ieee 1788 conformance: itf1788 vectors through the adapter in v2-plan.md "ieee 1788"

the vendored `.itl` files are unmodified copies of all 19 at https://github.com/oheim/ITF1788 (the
maintained fork of nehmeier's; each file's licence is in its header, and README.md pins the commit
and lists them). the adapter's rules:

* **input rule**: a 1788 unbounded bound is our open-at-inf, `[1, infinity]` is `[1, inf)` and
  `[entire]` is `(-inf, inf)`, because 1788 never attains infinity. a finite bound is closed
* **precision rule**: operands are the literals' doubles, held exactly (tests.itf1788.itl), and our
  exact result is rounded outward to doubles. an expected value is the tightest double enclosure,
  so this checks soundness and sharpness together, not just overlap. functions whose value is
  irrational already return their tightest enclosure (`intervals.elementary`)
* **outward rule**: every interval-valued vector runs a second time with the operands as floats in an
  `OutwardMultiInterval`, and its closed hull is compared with no rounding by the adapter: the
  library's own outward rounding must give 1788's tightest enclosure
* **output rule**: the closed hull of **both** our result and the expected value before comparing.
  it absorbs multi-interval vs connected (`1/[-10, 10]`: ours `[-inf, -1/10] ∪ [1/10, inf]`, 1788
  entire) and our attained infinities vs 1788's unattained ones. a boolean, a number or an overlap
  state is compared as it is
* **numeric rule**: `mid`, `rad`, `wid`, `mag`, `mig` and `midRad` give exact numbers for the exact
  operands of the first pass, rounded here as 1788 rounds them: `mid` to nearest, `wid` and `mag` up,
  `mig` down, and `rad` as the smallest double `r` with `[m - r, m + r]` holding the exact
  `[mid - rad, mid + rad]`, `m` the rounded midpoint. a second pass (`test_vector_float`) gives
  them float operands, in a `MultiInterval` and an `OutwardMultiInterval`, and compares the
  library's own rounding with no help from the adapter. a `ValueError` (the empty set) is `NaN`
* **reduction rule**: the reductions (`sum_nearest` and the rest) round their own value to nearest,
  so ours must already be that double and is compared as it is; a `ValueError` from one (a `nan`
  operand, `inf + -inf`, `0 * inf`) is 1788's `NaN`
* **decorations**: dropped from inputs and expected values; only the bare interval is compared.
  decorations are not in the core (v2-plan.md "ieee 1788"), so no vector checks one, and a
  divergence row is keyed on the statement with its decorations stripped (the fork has many
  statements twice, `atanh [1.0,1.0]_def = [empty]_trv` beside `atanh [1.0,1.0] = [empty]`).
  there is no NaI (D16, owner 2026-09-26), so a vector with a `[nai]` in it, and every `isNaI`, is
  a row under "no NaI: invalid input raises", generated below
* **cancellation** (`cancelMinus`, `cancelPlus`): ours is the Minkowski difference (D13), a real set
  wherever 1788 answers entire as "no answer"; those vectors are rows under "cancellation as a
  Minkowski difference". the others match in both passes, the outward one included: like 1788, an
  `OutwardMultiInterval` encloses the exact difference
* **power** (`pow`): run as `a ** b`, an interval exponent, which D11 makes 1788's pow (never pown,
  even for `[2.0, 2.0]`); all its vectors match in both passes, as do those of M13d's other functions
* **signals** (M13g): for an op in `SIGNALLED` (the constructors), ours and 1788's are compared as
  (value, signal) pairs: an `UndefinedOperationError` raised is 1788's bare answer to invalid input,
  `[empty]` with `signal UndefinedOperation`, and a `PossiblyUndefinedOperationWarning` emitted is
  `signal PossiblyUndefinedOperation`. a constructor has no interval operand to give as floats, so it
  is not in the outward pass. `test_signals_are_checked` keeps every op with a signal in `SIGNALLED`.
  `NaN` equals `NaN` here
* the library's warnings are ignored inside a vector (`1/[0]` is `∅` + `IndeterminateResultWarning`,
  and 1788's answer is also empty); they are pinned by their own tests elsewhere

every vector of an op in `OPS` either matches through the adapter or is a row of `DIVERGENCES`, whose
reason is one of the plan's residual categories. a row that starts matching fails as stale. the other
ops' statements are counted in `SKIPPED`, and every statement of every file is parsed by
`test_parser_reads_every_statement`, so the parser already reads what later ops will need.
"""
import math
import re
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from intervals import EMPTY
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import REALS
from intervals import dot
from intervals import nums_to_interval
from intervals import sum_
from intervals import sum_abs
from intervals import sum_sqr
from intervals import text_to_interval
from intervals.errors import IntervalWarning
from intervals.errors import PossiblyUndefinedOperationWarning
from intervals.errors import UndefinedOperationError
from intervals.relations import Allen
from tests.itf1788.itl import Interval
from tests.itf1788.itl import parse_file
from tests.itf1788.itl import strip_decorations

HERE = Path(__file__).parent
FILES = ('libieeep1788_elem.itl', 'libieeep1788_set.itl', 'libieeep1788_bool.itl', 'libieeep1788_num.itl',
         'libieeep1788_overlap.itl', 'libieeep1788_rec_bool.itl', 'libieeep1788_cancel.itl',
         'libieeep1788_class.itl', 'libieeep1788_mul_rev.itl', 'libieeep1788_reduction.itl',
         'libieeep1788_rev.itl', 'abs_rev.itl', 'pow_rev.itl', 'atan2.itl', 'c-xsc.itl', 'fi_lib.itl', 'mpfi.itl',
         'ieee1788-constructors.itl', 'ieee1788-exceptions.itl')

_ENTIRE = MultiInterval.parse('(-inf, inf)')


def _function(name):
    return lambda a: getattr(a, name)()


# the used subset: every 1788 op the package implements. an interval-valued op returns a
# MultiInterval, the others a bool, a number or an overlap state
OPS = {
    'pos': lambda a: +a,
    'neg': lambda a: -a,
    'abs': abs,
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'recip': lambda a: a.reciprocal(),
    'sqr': lambda a: a ** 2,
    'pown': lambda a, n: a ** n,
    'fma': lambda a, b, c: a.fma(b, c),
    'min': lambda a, b: a.minimum(b),
    'max': lambda a, b: a.maximum(b),
    'floor': lambda a: a.floor(),
    'ceil': lambda a: a.ceil(),
    'trunc': lambda a: a.trunc(),
    'roundTiesToEven': round,
    'roundTiesToAway': lambda a: a.round_ties_away(),
    'sign': lambda a: a.sign(),
    **{name: _function(name) for name in ('sqrt', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 'sin',
                                          'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                                          'asinh', 'acosh', 'atanh', 'expm1', 'cbrt', 'cot', 'sec', 'csc',
                                          'acot', 'coth', 'csch', 'sech', 'acoth')},
    'logp1': lambda a: a.log1p(),
    'rootn': lambda a, n: a.rootn(n),
    'hypot': lambda a, b: a.hypot(b),
    # an interval exponent is 1788's pow (D11), never pown
    'pow': lambda a, b: a ** b,
    'atan2': lambda y, x: y.atan2(x),
    'intersection': lambda a, b: a & b,
    'convexHull': lambda a, b: (a | b).hull,
    # booleans
    'isEmpty': lambda a: a.is_empty,
    'isEntire': lambda a: a.issuperset(_ENTIRE),
    'equal': lambda a, b: a == b,
    'subset': lambda a, b: a.issubset(b),
    'disjoint': lambda a, b: a.isdisjoint(b),
    'precedes': lambda a, b: (a <= b).certainly,
    'strictPrecedes': lambda a, b: (a < b).certainly,
    'isMember': lambda x, a: x in a,
    'isSingleton': lambda a: a.is_degenerate and a.is_contiguous,
    'isCommonInterval': lambda a: bool(a) and a.is_finite,
    # the interval orders, on the ends; 1788's interior(A, B) is A inside B's interior (the input
    # rule opens an unbounded end, so `interior [1, infinity] [0, infinity]` holds)
    'less': lambda a, b: a.weakly_less(b),
    'strictLess': lambda a, b: a.strictly_less(b),
    'interior': lambda a, b: a.within(b.interior),
    # numbers: 1788 gives +infinity for the infimum of the empty set, -infinity for its supremum
    'inf': lambda a: a.inf if a else math.inf,
    'sup': lambda a: a.sup if a else -math.inf,
    # numbers of an interval (NUMERIC): 1788's NaN for the empty set is our ValueError
    'mid': lambda a: a.mid(),
    'rad': lambda a: a.rad(),
    'wid': lambda a: a.wid(),
    'mag': lambda a: a.mag(),
    'mig': lambda a: a.mig(),
    'midRad': lambda a: a.mid_rad(),
    # the overlap state: our allen relation, named as in 1788
    'overlap': lambda a, b: _overlap(a, b),
    # reductions: numbers in, one double out, rounded to nearest by the op itself (REDUCTIONS)
    'sum_nearest': sum_,
    'sum_abs_nearest': sum_abs,
    'sum_sqr_nearest': sum_sqr,
    'dot_nearest': dot,
    # cancellation (D13): the Minkowski difference, a real set where 1788 has no answer
    'cancelMinus': lambda a, b: a.cancel_minus(b),
    'cancelPlus': lambda a, b: a.cancel_plus(b),
    # M13g: 1788's bare constructors (SIGNALLED), and isNaI, which has no counterpart: no NaI (D16)
    'b-textToInterval': lambda text: text_to_interval(text.value),
    'b-numsToInterval': nums_to_interval,
    'isNaI': lambda a: _no_nai(),
}
REDUCTIONS = frozenset({'sum_nearest', 'sum_abs_nearest', 'sum_sqr_nearest', 'dot_nearest'})
NUMERIC = frozenset({'mid', 'rad', 'wid', 'mag', 'mig', 'midRad'})

_OVERLAP_NAMES = {
    Allen.BEFORE: 'before', Allen.MEETS: 'meets', Allen.OVERLAPS: 'overlaps', Allen.STARTS: 'starts',
    Allen.DURING: 'containedBy', Allen.FINISHES: 'finishes', Allen.EQUALS: 'equals',
    Allen.FINISHED_BY: 'finishedBy', Allen.CONTAINS: 'contains', Allen.STARTED_BY: 'startedBy',
    Allen.OVERLAPPED_BY: 'overlappedBy', Allen.MET_BY: 'metBy', Allen.AFTER: 'after',
}


def _overlap(a, b):
    if not a or not b:
        return 'bothEmpty' if not a and not b else 'firstEmpty' if not a else 'secondEmpty'
    return _OVERLAP_NAMES[a.allen(b)]


# the plan's residual categories (v2-plan.md "ieee 1788"); a row's reason starts with one of them
REASONS = ('degenerate infinities', 'domain-clipped functions', 'decoration expectations',
           'cut-based relations', 'cancellation as a Minkowski difference',
           # M13g: approved with D16 (owner 2026-09-26)
           'no NaI: invalid input raises',
           # M13g, PROPOSED, not approved: needs the owner (v2-implementation-plan.md M13g)
           'exact parsing decides validity')

_LOG = ('degenerate infinities: the operand meets the domain [0, inf] only at 0, and log(0) is '
        '-inf here (the limit from the one side the domain has); 1788 drops 0 from the domain')
_ATANH = ('degenerate infinities: the operand meets the domain [-1, 1] only at an end, and atanh(±1) '
          'is ±inf here (the limit from inside); 1788 drops ±1 from the domain')
_NAI = ('no NaI: invalid input raises: NaI is not a set, and the package has none (D16, owner '
        '2026-09-26): a 1788 constructor given invalid input raises, so nothing makes a NaI')
_NO_IS_NAI = ('no NaI: invalid input raises: with no NaI (D16) there is nothing for isNaI to ask, '
              'so it has no counterpart')
_EXACT_VALID = ('exact parsing decides validity: the literal is valid as rationals, so ours returns it '
                'and emits nothing; 1788 lets a parser that rounds first signal PossiblyUndefinedOperation')
_EXACT_INVALID = ('exact parsing decides validity: the lower bound exceeds the upper as rationals, so '
                  'ours raises UndefinedOperationError; 1788 lets a parser that rounds first return the '
                  'hull of the rounded bounds with PossiblyUndefinedOperation')
_MEETS = ('cut-based relations: two closed intervals that share an end share that point, so they '
          'overlap (relations.Allen, on cuts); 1788 calls touching closed intervals meets or metBy')
_CANCEL = ('cancellation as a Minkowski difference: 1788 answers entire as "no answer" (A narrower '
           'than B, or an unbounded operand); ours is the largest X with B + X ⊆ A, a real set, here ∅ '
           'or a ray (D13)')
_CANCEL_EMPTY = ('cancellation as a Minkowski difference: with B = ∅ every X has B + X = ∅ ⊆ A, so the '
                 'largest is [-inf, inf]; 1788 answers ∅ when A is ∅ too (D13)')

# (statement text, whitespace collapsed and decorations stripped) -> reason. as of 2026-09-26 every
# listed row is a degenerate infinity of a function at the end of its domain, a touching pair that
# shares a point, or a cancellation where 1788 has no answer (or ∅ for ∅ and ∅); the NaI rows are
# generated once the vectors are loaded
DIVERGENCES = {
    'log [-infinity,0.0] = [empty]': _LOG,
    'log [-infinity,-0.0] = [empty]': _LOG,
    'log2 [-infinity,0.0] = [empty]': _LOG,
    'log2 [-infinity,-0.0] = [empty]': _LOG,
    'log10 [-infinity,0.0] = [empty]': _LOG,
    'log10 [-infinity,-0.0] = [empty]': _LOG,
    'atanh [1.0,infinity] = [empty]': _ATANH,
    'atanh [-infinity,-1.0] = [empty]': _ATANH,
    'atanh [-1.0,-1.0] = [empty]': _ATANH,
    'atanh [1.0,1.0] = [empty]': _ATANH,
    'overlap [-infinity,2.0] [2.0,3.0] = meets': _MEETS,
    'overlap [1.0,2.0] [2.0,3.0] = meets': _MEETS,
    'overlap [1.0,2.0] [2.0,infinity] = meets': _MEETS,
    'overlap [2.0,3.0] [1.0,2.0] = metBy': _MEETS,
    'overlap [2.0,3.0] [-infinity,2.0] = metBy': _MEETS,
    'cancelMinus [empty] [empty] = [empty]': _CANCEL_EMPTY,
    'cancelPlus [empty] [empty] = [empty]': _CANCEL_EMPTY,
}

# cancellation (D13): every vector where 1788 answers entire and ours is not the whole line. the
# others with an entire answer match: `cancelMinus [entire] [-1.0,5.0]` is (-inf, inf) here too
_CANCELLATION_ROWS = (
    'cancelPlus [-infinity, -1.0] [-5.0,1.0] = [entire]',
    'cancelPlus [-1.0, infinity] [-5.0,1.0] = [entire]',
    'cancelPlus [-infinity, -1.0] [entire] = [entire]',
    'cancelPlus [-1.0, infinity] [entire] = [entire]',
    'cancelPlus [empty] [1.0, infinity] = [entire]',
    'cancelPlus [empty] [-infinity,1.0] = [entire]',
    'cancelPlus [empty] [entire] = [entire]',
    'cancelPlus [-1.0,5.0] [1.0,infinity] = [entire]',
    'cancelPlus [-1.0,5.0] [-infinity,1.0] = [entire]',
    'cancelPlus [-1.0,5.0] [entire] = [entire]',
    'cancelPlus [-5.0, -1.0] [1.0,5.1] = [entire]',
    'cancelPlus [-5.0, -1.0] [0.9,5.0] = [entire]',
    'cancelPlus [-5.0, -1.0] [0.9,5.1] = [entire]',
    'cancelPlus [-10.0, 5.0] [-5.0,10.1] = [entire]',
    'cancelPlus [-10.0, 5.0] [-5.1,10.0] = [entire]',
    'cancelPlus [-10.0, 5.0] [-5.1,10.1] = [entire]',
    'cancelPlus [1.0, 5.0] [-5.0,-0.9] = [entire]',
    'cancelPlus [1.0, 5.0] [-5.1,-1.0] = [entire]',
    'cancelPlus [1.0, 5.0] [-5.1,-0.9] = [entire]',
    'cancelPlus [-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023] [-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023] = [entire]',
    'cancelPlus [-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023] [-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023] = [entire]',
    'cancelPlus [-0X1P+0,0X1.FFFFFFFFFFFFEP-53] [-0X1P+0,0X1.FFFFFFFFFFFFFP-53] = [entire]',
    'cancelMinus [-infinity, -1.0] [-1.0,5.0] = [entire]',
    'cancelMinus [-1.0, infinity] [-1.0,5.0] = [entire]',
    'cancelMinus [-infinity, -1.0] [entire] = [entire]',
    'cancelMinus [-1.0, infinity] [entire] = [entire]',
    'cancelMinus [empty] [-infinity, -1.0] = [entire]',
    'cancelMinus [empty] [-1.0, infinity] = [entire]',
    'cancelMinus [empty] [entire] = [entire]',
    'cancelMinus [-1.0,5.0] [-infinity, -1.0] = [entire]',
    'cancelMinus [-1.0,5.0] [-1.0, infinity] = [entire]',
    'cancelMinus [-1.0,5.0] [entire] = [entire]',
    'cancelMinus [-5.0, -1.0] [-5.1,-1.0] = [entire]',
    'cancelMinus [-5.0, -1.0] [-5.0,-0.9] = [entire]',
    'cancelMinus [-5.0, -1.0] [-5.1,-0.9] = [entire]',
    'cancelMinus [-10.0, 5.0] [-10.1, 5.0] = [entire]',
    'cancelMinus [-10.0, 5.0] [-10.0, 5.1] = [entire]',
    'cancelMinus [-10.0, 5.0] [-10.1, 5.1] = [entire]',
    'cancelMinus [1.0, 5.0] [0.9, 5.0] = [entire]',
    'cancelMinus [1.0, 5.0] [1.0, 5.1] = [entire]',
    'cancelMinus [1.0, 5.0] [0.9, 5.1] = [entire]',
    'cancelMinus [-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023] [-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023] = [entire]',
    'cancelMinus [-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023] [-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023] = [entire]',
    'cancelMinus [0X1P-1022,0X1.0000000000001P-1022] [0X1P-1022,0X1.0000000000002P-1022] = [entire]',
    'cancelMinus [-0X1P+0,0X1.FFFFFFFFFFFFEP-53] [-0X1.FFFFFFFFFFFFFP-53,0X1P+0] = [entire]',
)
DIVERGENCES.update({text: _CANCEL for text in _CANCELLATION_ROWS})
# M13g: the vectors expecting PossiblyUndefinedOperation, decided exactly here (PROPOSED category)
DIVERGENCES.update({
    'b-textToInterval "[1.0000000000000001, 1.0000000000000002]" = [1.0, 0x1.0000000000001p+0] '
    'signal PossiblyUndefinedOperation': _EXACT_VALID,
    'b-textToInterval "[1.0000000000000002,1.0000000000000001]" = [1.0,0x1.0000000000001p+0] '
    'signal PossiblyUndefinedOperation': _EXACT_INVALID,
    'b-textToInterval "[10000000000000001/10000000000000000,10000000000000002/10000000000000001]" = '
    '[1.0,0x1.0000000000001p+0] signal PossiblyUndefinedOperation': _EXACT_INVALID,
    'b-textToInterval "[0x1.00000000000002p0,0x1.00000000000001p0]" = [1.0,0x1.0000000000001p+0] '
    'signal PossiblyUndefinedOperation': _EXACT_INVALID,
})
LISTED = dict(DIVERGENCES)


def key(vector) -> str:
    """the divergence table's key"""
    return strip_decorations(vector.text)


def _has_nai(vector) -> bool:
    return any(isinstance(v, Interval) and v.nai for v in (*vector.args, vector.expected))


def _load():
    vectors, skipped = [], {}
    for name in FILES:
        found, other = parse_file(HERE / name, OPS)
        vectors.extend(found)
        skipped[name] = other
    return tuple(vectors), skipped


VECTORS, SKIPPED = _load()
INTERVAL_VECTORS = tuple(v for v in VECTORS if isinstance(v.expected, Interval))
NUMERIC_VECTORS = tuple(v for v in VECTORS if v.op in NUMERIC)
DIVERGENCES.update({key(v): _NAI for v in VECTORS if _has_nai(v)})
# M13g: every isNaI is a row; the constructors, which have no interval operand, are not run outward
SIGNALLED = frozenset({'b-textToInterval', 'b-numsToInterval'})
DIVERGENCES.update({key(v): _NO_IS_NAI for v in VECTORS if v.op == 'isNaI' and not _has_nai(v)})
INTERVAL_VECTORS = tuple(v for v in INTERVAL_VECTORS if v.op not in SIGNALLED)


# THE ADAPTER

class NoCounterpart(ValueError):
    """a 1788 value the core has no counterpart for (NaI)"""


def to_ours(literal, cls=MultiInterval, as_float=False):
    """the input rule; `as_float` gives the literal's doubles as floats instead of Fractions"""
    if not isinstance(literal, Interval):
        return float(literal) if as_float and isinstance(literal, Fraction) else literal
    if literal.nai:
        raise NoCounterpart('NaI')
    if literal.empty:
        return cls()
    lo, hi = (float(literal.lo), float(literal.hi)) if as_float else (literal.lo, literal.hi)
    return cls(lo, hi, start_closed=literal.lo != -math.inf, end_closed=literal.hi != math.inf)


def round_down(v) -> float:
    """the largest double <= v (±inf stay)"""
    if v == math.inf or v == -math.inf:
        return float(v)
    try:
        f = float(Fraction(v))  # correctly rounded: int / int in CPython
    except OverflowError:
        f = math.inf if v > 0 else -math.inf
    return math.nextafter(f, -math.inf) if f > v else f


def round_up(v) -> float:
    return -round_down(-v)


def round_nearest(v) -> float:
    """the double nearest to v, ties to even (±inf and floats stay)"""
    if isinstance(v, float):
        return v
    try:
        return float(Fraction(v))  # correctly rounded: int / int in CPython
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def closed_hull_of_ours(result: MultiInterval):
    """the precision and output rules on our result: `None` for empty, else `(lo, hi)` doubles"""
    if result.is_empty:
        return None
    return round_down(result.inf), round_up(result.sup)


def closed_hull_of_expected(literal: Interval):
    """the output rule on 1788's value (already doubles)"""
    if literal.nai:
        raise NoCounterpart('NaI')
    return None if literal.empty else (float(literal.lo), float(literal.hi))


def _call(vector, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        return OPS[vector.op](*args)


def _reduce(vector, args) -> float:
    """the reduction rule: our double as it is, or 1788's NaN where ours raises ValueError"""
    try:
        result = _call(vector, args)
    except ValueError:
        return math.nan
    assert isinstance(result, float), type(result)
    return result


def _mid_rad_1788(m, r):
    """1788's (mid, rad) of the exact hull `[m - r, m + r]`: the midpoint to nearest, then the
    smallest double radius around that midpoint"""
    rounded = round_nearest(m)
    if r == math.inf:
        return rounded, math.inf
    lo, hi = Fraction(m) - r, Fraction(m) + r
    return rounded, round_up(max(Fraction(rounded) - lo, hi - Fraction(rounded)))


_NUMERIC_ROUNDING = {'mid': round_nearest, 'wid': round_up, 'mag': round_up, 'mig': round_down}


def _numeric(vector, args):
    """the numeric rule's first pass: our exact number rounded as 1788 would, NaN for a ValueError"""
    try:
        result = _call(vector, args)
    except ValueError:
        return (math.nan, math.nan) if vector.op == 'midRad' else math.nan
    if vector.op == 'midRad':
        return _mid_rad_1788(*result)
    if vector.op == 'rad':
        return _mid_rad_1788(args[0].mid(), result)[1]
    return _NUMERIC_ROUNDING[vector.op](result)


def run_float(vector, cls):
    """(ours, expected): the numeric rule's second pass, float operands, our numbers as they are"""
    args = [to_ours(a, cls, as_float=True) for a in vector.args]
    try:
        result = _call(vector, args)
    except ValueError:
        result = (math.nan, math.nan) if vector.op == 'midRad' else math.nan
    # rounded, so never a Fraction; an exact 0 or inf where no end is a finite float (entire)
    assert all(isinstance(n, float) or n == 0 for n in _numbers(result)), result
    return result, vector.expected


def _numbers(value):
    return value if isinstance(value, tuple) else (value,)


def _no_nai():
    raise NoCounterpart('isNaI: there is no NaI')


def _signalled(vector):
    """(ours, expected) as (closed hull, signal) pairs, for an op in SIGNALLED (M13g): a raised
    UndefinedOperationError is 1788's bare answer to invalid input, empty with UndefinedOperation"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = OPS[vector.op](*[to_ours(a) for a in vector.args])
        except UndefinedOperationError:
            ours = None, 'UndefinedOperation'
        else:
            possibly = any(issubclass(w.category, PossiblyUndefinedOperationWarning) for w in caught)
            ours = closed_hull_of_ours(result), 'PossiblyUndefinedOperation' if possibly else None
    return ours, (closed_hull_of_expected(vector.expected), vector.signal)


def run(vector):
    """(ours, expected), both through the adapter"""
    if vector.op in SIGNALLED:
        return _signalled(vector)
    if vector.op in NUMERIC:
        return _numeric(vector, [to_ours(a) for a in vector.args]), _numbers_as_floats(vector.expected)
    if vector.op in REDUCTIONS:
        return _reduce(vector, [to_ours(a) for a in vector.args]), float(vector.expected)
    result = _call(vector, [to_ours(a) for a in vector.args])
    if isinstance(vector.expected, Interval):
        return closed_hull_of_ours(result), closed_hull_of_expected(vector.expected)
    if isinstance(vector.expected, Fraction) or isinstance(vector.expected, float):
        return round_down(result) if vector.op == 'inf' else round_up(result), float(vector.expected)
    return result, vector.expected


def run_outward(vector):
    """(ours, expected): float operands in an OutwardMultiInterval, our closed hull taken exactly"""
    result = _call(vector, [to_ours(a, OutwardMultiInterval, as_float=True) for a in vector.args])
    assert isinstance(result, OutwardMultiInterval), type(result)
    expected = closed_hull_of_expected(vector.expected)
    if result.is_empty:
        return None, expected
    lo, hi = result.inf, result.sup
    # every finite end is a double already: nothing is left for the adapter to round
    assert all(isinstance(v, float) or v == round_down(v) for v in (lo, hi)), (lo, hi)
    return (lo, hi), expected


def _numbers_as_floats(value):
    return tuple(float(n) for n in value) if isinstance(value, tuple) else float(value)


def same(ours, expected) -> bool:
    """equality, except that NaN is NaN (1788's answer for a number of the empty set or of NaI);
    a pair (midRad) item by item"""
    if isinstance(ours, tuple) and isinstance(expected, tuple):
        return len(ours) == len(expected) and all(same(o, e) for o, e in zip(ours, expected))
    if isinstance(ours, float) and isinstance(expected, float) and math.isnan(ours) and math.isnan(expected):
        return True
    return ours == expected


def outcome(runner, vector):
    """(ours, expected) from `run` or `run_outward`; ours is the exception if the core has no counterpart"""
    try:
        return runner(vector)
    except NoCounterpart as e:
        return e, vector.expected


# THE VECTORS

def check(vector, runner):
    ours, expected = outcome(runner, vector)
    if key(vector) in DIVERGENCES:
        assert not same(ours, expected), f'stale divergence row, it matches now: {vector.text}'
    else:
        assert same(ours, expected), vector.text


@pytest.mark.parametrize('vector', VECTORS, ids=[v.source for v in VECTORS])
def test_vector(vector):
    check(vector, run)


@pytest.mark.parametrize('vector', INTERVAL_VECTORS, ids=[v.source for v in INTERVAL_VECTORS])
def test_vector_outward(vector):
    check(vector, run_outward)


@pytest.mark.parametrize('cls', [MultiInterval, OutwardMultiInterval], ids=['nearest', 'outward'])
@pytest.mark.parametrize('vector', NUMERIC_VECTORS, ids=[v.source for v in NUMERIC_VECTORS])
def test_vector_float(vector, cls):
    check(vector, lambda v: run_float(v, cls))


def test_divergence_rows():
    keys = {key(v) for v in VECTORS}
    for text, reason in DIVERGENCES.items():
        assert text in keys, f'no such vector: {text}'
        assert reason.startswith(REASONS), reason
    # the generated NaI rows never overwrite a listed one
    assert all(DIVERGENCES[text] is LISTED[text] for text in LISTED)


# a statement line: an op name, its operands, ` = `, the result, `;`. no comment line in these files
# has a ` = ` before a `;`, so this counts statements independently of the parser
_STATEMENT_LINE = re.compile(r'^[ \t]*([A-Za-z][\w-]*)[ \t][^\n;]* = [^\n;]*;[ \t]*$', re.MULTILINE)


@pytest.mark.parametrize('name', FILES)
def test_parser_drops_nothing(name):
    """an independent count: every statement line of a used op became a vector, every other one a skip"""
    lines = Counter(_STATEMENT_LINE.findall((HERE / name).read_text(encoding='utf-8')))
    parsed = Counter(v.op for v in VECTORS if v.source.startswith(f'{name}:'))
    assert parsed + SKIPPED[name] == lines
    assert not set(OPS) & set(SKIPPED[name])


@pytest.mark.parametrize('name', FILES)
def test_parser_reads_every_statement(name):
    """every statement of every op parses, so an op added later needs no parser work"""
    everything, skipped = parse_file(HERE / name)
    assert not skipped
    assert Counter(v.op for v in everything) == Counter(
        _STATEMENT_LINE.findall((HERE / name).read_text(encoding='utf-8')))


def test_signals_are_checked():
    """every op whose vectors carry a 1788 signal has it compared (M13g)"""
    assert {v.op for v in VECTORS if v.signal} <= SIGNALLED


def test_signalled_reads_both_signals(monkeypatch):
    """the adapter's reading of ours, pinned apart from the library, which never warns today"""
    vector = next(v for v in VECTORS if v.op == 'b-numsToInterval' and v.signal is None)

    def possibly(*args):
        warnings.warn('possibly', PossiblyUndefinedOperationWarning)
        return nums_to_interval(*args)

    def undefined(*args):
        raise UndefinedOperationError('undefined')

    monkeypatch.setitem(OPS, vector.op, possibly)
    assert _signalled(vector)[0] == (closed_hull_of_expected(vector.expected), 'PossiblyUndefinedOperation')
    monkeypatch.setitem(OPS, vector.op, undefined)
    assert _signalled(vector)[0] == (None, 'UndefinedOperation')


def test_every_op_has_vectors():
    """a parser that finds nothing for an op would pass every vector of it vacuously"""
    assert {v.op for v in VECTORS} == set(OPS)


def test_every_file_is_used():
    """every vendored file is read, and every file read has statements"""
    assert {p.name for p in HERE.glob('*.itl')} == set(FILES) and len(FILES) == 19
    for name in FILES:
        assert any(v.source.startswith(f'{name}:') for v in VECTORS) or SKIPPED[name], name


# THE ADAPTER'S OWN RULES

def test_input_rule():
    assert to_ours(Interval(Fraction(1), math.inf)) == MultiInterval(1, math.inf, end_closed=False)
    assert to_ours(Interval(-math.inf, math.inf)) == MultiInterval.parse('(-inf, inf)')
    assert to_ours(Interval(None, None)) == EMPTY
    assert to_ours(Interval(Fraction(1, 2), Fraction(3)), OutwardMultiInterval, as_float=True) == \
        OutwardMultiInterval(0.5, 3.0)


def test_is_entire_reads_entire_as_open():
    assert OPS['isEntire'](_ENTIRE) and OPS['isEntire'](REALS)
    assert not OPS['isEntire'](MultiInterval.parse('(-inf, 5]'))


@pytest.mark.parametrize('v, down, up', [
    (Fraction(1, 10), 0.09999999999999999, 0.1),
    (Fraction(3), 3.0, 3.0),
    (Fraction(-1, 10), -0.1, -0.09999999999999999),
    (2 * Fraction(math.ulp(0.0)) / 3, 0.0, math.ulp(0.0)),
    (Fraction(2) ** 1024, 1.7976931348623157e308, math.inf),
    (-Fraction(2) ** 1024, -math.inf, -1.7976931348623157e308),
    (math.inf, math.inf, math.inf),
    (-math.inf, -math.inf, -math.inf),
])
def test_outward_rounding(v, down, up):
    assert round_down(v) == down
    assert round_up(v) == up
