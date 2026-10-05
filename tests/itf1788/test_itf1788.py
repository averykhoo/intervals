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
  irrational already return their tightest enclosure (`multiinterval.elementary`)
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
* **decorations** (M13g part 3): decorations are not in the core (v2-plan.md "ieee 1788") but in
  `DecoratedInterval`, and every decorated vector runs through it: an op in `PROPAGATED` (arithmetic,
  functions, set operations) gets `DecoratedInterval` operands and its result is compared as (closed
  hull, decoration), so 1788's propagation is checked in both passes; any other op (`BARE_PART`: the
  booleans, numbers and overlap) gets each operand's interval part, as 1788 defines them, after the
  operand was built with its decoration. a divergence row is keyed on the statement with its
  decorations stripped (the fork has many statements twice, `atanh [1.0,1.0]_def = [empty]_trv` beside
  `atanh [1.0,1.0] = [empty]`), except a row on a decoration alone (`PLAIN_ONLY`, and
  `DECORATION_ONLY`, whose set must match), keyed with them. the reverse ops (`REVERSE`) are in
  `PROPAGATED` since M13's merge: `multiinterval.reverse` decorates their results trv, as 1788 does.
  there is no NaI (D16, owner 2026-09-26), so a vector with a `[nai]` in it, and every `isNaI`, is
  a row under "no NaI: invalid input raises", generated below
* **decorated ops** (M13g, `DECORATED`: the `d-` constructors, `newDec`, `setDec`, `intervalPart`,
  `decorationPart`): a decorated operand is a `DecoratedInterval`, and a decorated result is compared
  as (closed hull, decoration) with 1788's, so the decoration is checked. a raised `UndefinedOperationError` is the decorated flavour's `[nai]` with `signal
  UndefinedOperation`, so those vectors match and are no row
* **cancellation** (`cancelMinus`, `cancelPlus`): ours is the Minkowski difference (D13), a real set
  wherever 1788 answers entire as "no answer"; those vectors are rows under "cancellation as a
  Minkowski difference". the others match in both passes, the outward one included: like 1788, an
  `OutwardMultiInterval` encloses the exact difference
* **power** (`pow`): run as `a ** b`, an interval exponent, which D11 makes 1788's pow (never pown,
  even for `[2.0, 2.0]`); all its vectors match in both passes, as do those of M13d's other functions
* **pair rule** (`mulRevToPair`, M13e): 1788's two intervals are one multi-interval here, `mul_rev`;
  each of our pieces, closed (rounded outward in the first pass), is compared in order with the
  pair's non-empty intervals, piece by piece. the pairs run in the outward pass too. a decorated pair
  (M13's merge) is (pieces, decoration), 1788's decoration being that of its non-empty intervals
* **signals** (M13g): for an op in `SIGNALLED` (the constructors, `setDec`, `intervalPart`), ours and 1788's are compared as
  (value, signal) pairs: an `UndefinedOperationError` raised is 1788's bare answer to invalid input,
  `[empty]` with `signal UndefinedOperation`, and a `PossiblyUndefinedOperationWarning` emitted is
  `signal PossiblyUndefinedOperation`. a constructor has no interval operand to give as floats, so it
  is not in the outward pass (`CONSTRUCTORS`); `setDec` and `intervalPart` are, signals and all.
  `test_signals_are_checked` keeps every op with a signal in `SIGNALLED`.
  `NaN` equals `NaN` here
* the library's warnings are ignored inside a vector (`1/[0]` is `∅` + `IndeterminateResultWarning`,
  and 1788's answer is also empty); they are pinned by their own tests elsewhere

every vector of an op in `OPS` either matches through the adapter or is a row of `DIVERGENCES`, whose
reason is one of the plan's residual categories. a row that starts matching fails as stale. an op not
in `OPS` would have its statements counted in `SKIPPED`, which M13's exit keeps empty
(`test_nothing_is_skipped`), and every statement of every file is parsed by
`test_parser_reads_every_statement`.

this adapter tests the library's own semantics. 1788's own answers are the 1788 layer's
(`multiinterval.ieee1788`, M16b), and a third pass, `tests/itf1788/test_ieee1788.py`, runs every vector
through it and compares exactly, with no hull and no rounding. so a row here is a true statement
about the library, and each row of a category that pass has none of (degenerate infinities, cut-based
relations, cancellation as a Minkowski difference, decoration expectations) has a second reading:
1788's answer is the layer's, which matches.
"""
import math
import re
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from multiinterval import EMPTY
from multiinterval import DecoratedInterval
from multiinterval import Decoration
from multiinterval import MultiInterval
from multiinterval import OutwardMultiInterval
from multiinterval import REALS
from multiinterval import abs_rev
from multiinterval import cos_rev
from multiinterval import cosh_rev
from multiinterval import dot
from multiinterval import mul_rev
from multiinterval import nums_to_decorated_interval
from multiinterval import nums_to_interval
from multiinterval import pow_rev1
from multiinterval import pow_rev2
from multiinterval import pown_rev
from multiinterval import set_dec
from multiinterval import sin_rev
from multiinterval import sqr_rev
from multiinterval import sum_
from multiinterval import sum_abs
from multiinterval import sum_sqr
from multiinterval import tan_rev
from multiinterval import text_to_decorated_interval
from multiinterval import text_to_interval
from multiinterval.errors import IntervalWarning
from multiinterval.errors import PossiblyUndefinedOperationWarning
from multiinterval.errors import UndefinedOperationError
from multiinterval.kernel import pieces
from multiinterval.relations import Allen
from tests.itf1788.itl import Interval
from tests.itf1788.itl import Text
from tests.itf1788.itl import Vector
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
    # reverse ops (M13e, D12): `{x in X : f(x) in C}` as a union; the unary form is the call with x
    # omitted (x = REALS, ±inf included), the *Bin form the call with x given. compared by the output
    # rule: 1788's reverse op is the hull of the preimage, so closed hulls agree iff ours has its hull.
    # the exponent is a 1788 integer literal (a float in the outward pass): an int here
    'sqrRev': sqr_rev,
    'sqrRevBin': sqr_rev,
    'absRev': abs_rev,
    'absRevBin': abs_rev,
    'pownRev': lambda c, n: pown_rev(c, int(n)),
    'pownRevBin': lambda c, x, n: pown_rev(c, int(n), x),
    'coshRev': cosh_rev,
    'coshRevBin': cosh_rev,
    # mulRev (M13e): `{x in X : x * y in C for some y in B}` with the library's `*`, B first as in 1788;
    # mulRevTen is the call with x given; mulRevToPair's two intervals are one set here (PAIRS)
    'mulRev': mul_rev,
    'mulRevTen': mul_rev,
    'mulRevToPair': mul_rev,
    # sinRev, cosRev, tanRev (M13e, D12): as the unary reverse ops above. the unary form's x is the
    # whole line, so any c with a solution has infinitely many pieces: their hull, (-inf, inf), with a
    # HullWarning (ignored here), which is 1788's entire; a *Bin form's bounded x gives the exact pieces
    'sinRev': sin_rev,
    'sinRevBin': sin_rev,
    'cosRev': cos_rev,
    'cosRevBin': cos_rev,
    'tanRev': tan_rev,
    'tanRevBin': tan_rev,
    # powRev1, powRev2 (M13e): the bases, `{x in X : x ** y in C for some y in B}`, and the exponents,
    # `{y in Y : x ** y in C for some x in A}`, with the library's pow (D11), the operands in 1788's order;
    # every 1788 vector gives the domain, so none is the call with it omitted
    'powRev1': pow_rev1,
    'powRev2': pow_rev2,
    # M13g: 1788's bare constructors (SIGNALLED), and isNaI, which has no counterpart: no NaI (D16)
    'b-textToInterval': lambda text: text_to_interval(text.value),
    'b-numsToInterval': nums_to_interval,
    'isNaI': lambda a: _no_nai(),
    # M13g: the decorated type (DECORATED); newDec is its constructor, the two parts its properties
    'd-textToInterval': lambda text: text_to_decorated_interval(text.value),
    'd-numsToInterval': nums_to_decorated_interval,
    'newDec': DecoratedInterval,
    'setDec': set_dec,
    'intervalPart': lambda d: d.interval,
    'decorationPart': lambda d: d.decoration,
}
REDUCTIONS = frozenset({'sum_nearest', 'sum_abs_nearest', 'sum_sqr_nearest', 'dot_nearest'})
NUMERIC = frozenset({'mid', 'rad', 'wid', 'mag', 'mig', 'midRad'})
PAIRS = frozenset({'mulRevToPair'})  # a pair of intervals, compared by the pair rule (`::_pair`)

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
REASONS = ('degenerate infinities', 'decoration expectations',
           'cut-based relations', 'cancellation as a Minkowski difference',
           # M13e, approved by the owner 2026-09-27 (D18): a vector whose expected hull is looser than
           # the tightest double enclosure, where ours is (checked against arb, or exactly, in
           # tests/test_reverse.py and tests/test_pow_rev.py)
           'tighter than the vector',
           # M13g: approved with D16 (owner 2026-09-26)
           'no NaI: invalid input raises',
           # M13g, approved by the owner 2026-09-27 (D18)
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

# (statement text, white space collapsed outside quoted strings and decorations stripped) -> reason.
# as of 2026-09-26 every listed row is a degenerate infinity of a function at the end of its domain, a touching pair that
# shares a point, or a cancellation where 1788 has no answer (or ∅ for ∅ and ∅); M13g adds, below,
# the literals decided exactly (PossiblyUndefinedOperation) and bounded exactly (com). the
# NaI and isNaI rows are generated once the vectors are loaded, and PLAIN_ONLY holds the rows on a
# decoration alone
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

# reverse ops (M13e): `t ** n` for n < 0 is 0 at ±inf here (the library's value there, `MultiInterval(inf)
# ** -2` is [0]), so the unary pownRev, whose x is all of [-inf, inf] (the default REALS), has ±inf in the
# preimage of every C holding 0. every such vector is a row; run with 1788's entire as x, each matches
# (tests/test_reverse.py::test_the_rows_differ_only_at_the_infinities). the *Bin forms never meet it:
# their x comes through the input rule, open at inf
_POWN_REV_INF = ('degenerate infinities: t ** n for n < 0 is 0 at t = ±inf here, so with x = [-inf, inf] '
                 '(the default) ±inf are in the preimage of a C holding 0; 1788 has no infinite points')
_POWN_REV_ROWS = (
    'pownRev [0.0,0.0] -2 = [empty]',
    'pownRev [-0.0,-0.0] -2 = [empty]',
    'pownRev [-10.0,0.0] -2 = [empty]',
    'pownRev [-10.0,-0.0] -2 = [empty]',
    'pownRev [0.0,0.0] -8 = [empty]',
    'pownRev [-0.0,-0.0] -8 = [empty]',
    'pownRev [0.0,0.0] -1 = [empty]',
    'pownRev [-0.0,-0.0] -1 = [empty]',
    'pownRev [0.0,infinity] -1 = [0.0,infinity]',
    'pownRev [-0.0,infinity] -1 = [0.0,infinity]',
    'pownRev [-infinity,0.0] -1 = [-infinity,0.0]',
    'pownRev [-infinity,-0.0] -1 = [-infinity,0.0]',
    'pownRev [0.0,0.0] -3 = [empty]',
    'pownRev [-0.0,-0.0] -3 = [empty]',
    'pownRev [0X0P+0,0X0.0000000000001P-1022] -3 = [0x1p+358,infinity]',
    'pownRev [-0X0.0000000000001P-1022,-0X0P+0] -3 = [-infinity,-0x1p+358]',
    'pownRev [0.0,infinity] -3 = [0.0,infinity]',
    'pownRev [-0.0,infinity] -3 = [0.0,infinity]',
    'pownRev [-infinity,0.0] -3 = [-infinity,0.0]',
    'pownRev [-infinity,-0.0] -3 = [-infinity,0.0]',
    'pownRev [0.0,0.0] -7 = [empty]',
    'pownRev [-0.0,-0.0] -7 = [empty]',
    'pownRev [0.0,infinity] -7 = [0.0,infinity]',
    'pownRev [-0.0,infinity] -7 = [0.0,infinity]',
    'pownRev [-infinity,0.0] -7 = [-infinity,0.0]',
    'pownRev [-infinity,-0.0] -7 = [-infinity,0.0]',
)
DIVERGENCES.update({text: _POWN_REV_INF for text in _POWN_REV_ROWS})
# two more have the same infinities and a second difference: 1788's end is one double outside the
# tightest enclosure of 2 ** (1074/7), which ours is (arb: 1.5367463556376297869...e46, strictly between
# 0x1.588cea3f093bdp+153 and 0x1.588cea3f093bep+153), so they differ even with x = 1788's entire
_POWN_REV_LOOSE = ('tighter than the vector: 1788 gives ±0x1.588cea3f093bcp+153 for 2 ** (1074/7), '
                   'one double outside the tightest enclosure, whose inner double is 0x1.588cea3f093bdp+153 '
                   '(ours); the unary form also has ±inf here (degenerate infinities, as the other pownRev rows)')
_POWN_REV_LOOSE_ROWS = (
    'pownRev [0X0P+0,0X0.0000000000001P-1022] -7 = [0x1.588cea3f093bcp+153,infinity]',
    'pownRev [-0X0.0000000000001P-1022,-0X0P+0] -7 = [-infinity,-0x1.588cea3f093bcp+153]',
)
DIVERGENCES.update({text: _POWN_REV_LOOSE for text in _POWN_REV_LOOSE_ROWS})
# sinRev, cosRev, tanRev (M13e): six *Bin vectors (and their decorated copies, 7 keys) whose expected hull has one
# end one or two doubles outside the tightest enclosure of k pi ± asin, acos or atan of an end of c; ours is
# the tightest, the other end matches, and arb agrees (tests/test_reverse.py::test_trig_rev_is_tighter_than_the_vector)
_TRIG_REV_LOOSE = ('tighter than the vector: one end of 1788\'s hull is one or two doubles outside '
                   'the tightest enclosure of k pi ± asin/acos/atan(v), which ours is (arb)')
_TRIG_REV_LOOSE_ROWS = (
    'sinRevBin [0X1.FFFFFFFFFFFFFP-1,0X1P+0] [1.57,1.58 ] = [0x1.921fb50442d18p+0,0x1.921fb58442d1ap+0]',
    'sinRevBin [0X1.FFFFFFFFFFFFFP-1,0X1P+0] [1.57,1.58] = [0x1.921fb50442d18p+0,0x1.921fb58442d1ap+0]',  # decorated copy, no space
    'cosRevBin [-1.0,-1.0] [3.14,3.15] = [0x1.921fb54442d18p+1,0x1.921fb54442d1ap+1]',
    'cosRevBin [-0X1P+0,-0X1.FFFFFFFFFFFFFP-1] [3.14,3.15] = [0x1.921fb52442d18p+1,0x1.921fb56442d1ap+1]',
    'cosRevBin [-0X1P+0,-0X1.FFFFFFFFFFFFFP-1] [-3.15,-3.14] = [-0x1.921fb56442d1ap+1,-0x1.921fb52442d18p+1]',
    'tanRevBin [0X1.D02967C31CDB4P+53,0X1.D02967C31CDB5P+53] [-1.5708,1.5708] = [-0x1.921fb54442d1bp+0,0x1.921fb54442d19p+0]',
    'tanRevBin [0X1.72CECE675D1FCP-52,0X1.72CECE675D1FDP-52] [-3.15,3.15] = [-0X1.921FB54442D19P+1,0X1.921FB54442D1aP+1]',
)
DIVERGENCES.update({text: _TRIG_REV_LOOSE for text in _TRIG_REV_LOOSE_ROWS})
# powRev2 (M13e): two vectors whose expected hull is far outside the tightest. for t in A (< 1) and C = [2, inf),
# t ** s >= 2 iff s <= log_t 2, and log_t 2 is at most -1/2 over t in [1/4, 1) (at t = 1/4, where (1/4) ** -1/2
# is exactly 2; 1 ** s is never 2), so the answer is (-inf, -1/2]: the neighbouring vectors with C = [2, 4]
# (`pow_rev.itl:608`, `:640`) answer -1/2 there, and C = [2, inf) only adds points that s -> -inf reaches.
# checked exactly, with no rounding, by tests/test_pow_rev.py::test_pow_rev2_is_tighter_than_the_vector
_POW_REV_LOOSE = ('tighter than the vector: for A in [1/4, 1] and C = [2, inf), t ** s >= 2 iff '
                  's <= log_t 2 <= -1/2, so the tightest hull is [-inf, -0.5], which ours is; 1788 answers '
                  '[entire] and [-infinity, 0.0], though its own vectors with C = [2, 4] answer -0.5 at that end')
_POW_REV_LOOSE_ROWS = (
    'powRev2 [0.25, 0.5] [2.0, infinity] [entire] = [entire]',
    'powRev2 [0.25, 1.0] [2.0, infinity] [entire] = [-infinity, 0.0]',
)
DIVERGENCES.update({text: _POW_REV_LOOSE for text in _POW_REV_LOOSE_ROWS})
# M13g: the vectors expecting PossiblyUndefinedOperation, decided exactly here (D18)
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
# M13g, the decorated type: the d- twins of the three above, and a com the exact value keeps
_BOUNDED_EXACTLY = ('decoration expectations: the literal is bounded as a rational, so com fits it '
                    'here; 1788 decorates its binary64 hull, which overflows to [max, inf] or entire, '
                    'and demotes com to dac')
DIVERGENCES.update({
    'd-textToInterval "[1.0000000000000002,1.0000000000000001]" = [1.0,0x1.0000000000001p+0] '
    'signal PossiblyUndefinedOperation': _EXACT_INVALID,
    'd-textToInterval "[10000000000000001/10000000000000000,10000000000000002/10000000000000001]" = '
    '[1.0,0x1.0000000000001p+0] signal PossiblyUndefinedOperation': _EXACT_INVALID,
    'd-textToInterval "[0x1.00000000000002p0,0x1.00000000000001p0]" = [1.0,0x1.0000000000001p+0] '
    'signal PossiblyUndefinedOperation': _EXACT_INVALID,
    'd-textToInterval "[1.0E+400 ]_com" = [0x1.fffffffffffffp+1023,infinity]': _BOUNDED_EXACTLY,
    'd-textToInterval "10?3e380_com" = [0x1.fffffffffffffp+1023,infinity]': _BOUNDED_EXACTLY,
    'd-textToInterval "10?1' + '8' + '0' * 308 + '_com" = [-infinity,infinity]': _BOUNDED_EXACTLY,
})
LISTED = dict(DIVERGENCES)


def key(vector) -> str:
    """the divergence table's key"""
    return strip_decorations(vector.text)


def _has_nai(vector) -> bool:
    return any(isinstance(v, Interval) and v.nai for v in (*vector.args, vector.expected))


def _nai_is_a_raise(vector) -> bool:
    """M13g: 1788's `[nai]` answer to invalid input, `signal UndefinedOperation`, which a raise matches"""
    return (vector.signal == 'UndefinedOperation' and isinstance(vector.expected, Interval) and vector.expected.nai
            and not any(isinstance(v, Interval) and v.nai for v in vector.args))


def _load():
    vectors, skipped = [], {}
    for name in FILES:
        found, other = parse_file(HERE / name, OPS)
        vectors.extend(found)
        skipped[name] = other
    return tuple(vectors), skipped


VECTORS, SKIPPED = _load()
INTERVAL_VECTORS = tuple(v for v in VECTORS if isinstance(v.expected, Interval) or v.op in PAIRS)
NUMERIC_VECTORS = tuple(v for v in VECTORS if v.op in NUMERIC)
DIVERGENCES.update({key(v): _NAI for v in VECTORS if _has_nai(v) and not _nai_is_a_raise(v)})
# M13g: every isNaI is a row; the constructors, which have no interval operand, are not run outward
SIGNALLED = frozenset({'b-textToInterval', 'b-numsToInterval',
                       'd-textToInterval', 'd-numsToInterval', 'setDec', 'intervalPart'})
CONSTRUCTORS = frozenset({'b-textToInterval', 'b-numsToInterval', 'd-textToInterval', 'd-numsToInterval'})
DECORATED = frozenset({'d-textToInterval', 'd-numsToInterval', 'newDec', 'setDec', 'intervalPart', 'decorationPart'})
# M13g part 3: the ops whose decorated vectors run on DecoratedInterval operands, the decoration
# propagated by the op and checked (1788's decorated arithmetic, functions and set operations). any
# other op of a decorated vector takes the interval part of each operand (BARE_PART), as 1788 defines
# the booleans and numbers of a decorated interval. the reverse ops (M13e, `REVERSE`) are in it since
# M13's merge: `multiinterval.reverse` takes DecoratedInterval operands and decorates the result trv, as
# 1788 does (`reverse.py::_decorated`); a pair (mulRevToPair) is compared with its decoration by the
# pair rule (`_pair_outcome`)
REVERSE = frozenset({'sqrRev', 'sqrRevBin', 'absRev', 'absRevBin', 'pownRev', 'pownRevBin', 'coshRev',
                     'coshRevBin', 'sinRev', 'sinRevBin', 'cosRev', 'cosRevBin', 'tanRev', 'tanRevBin',
                     'mulRev', 'mulRevTen', 'mulRevToPair', 'powRev1', 'powRev2'})
PROPAGATED = frozenset({
    'pos', 'neg', 'abs', 'add', 'sub', 'mul', 'div', 'recip', 'sqr', 'pown', 'fma', 'min', 'max', 'floor',
    'ceil', 'trunc', 'roundTiesToEven', 'roundTiesToAway', 'sign', 'sqrt', 'exp', 'exp2', 'exp10', 'log',
    'log2', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
    'atanh', 'expm1', 'cbrt', 'cot', 'sec', 'csc', 'acot', 'coth', 'csch', 'sech', 'acoth', 'logp1', 'rootn',
    'hypot', 'pow', 'atan2', 'intersection', 'convexHull', 'cancelMinus', 'cancelPlus', *REVERSE})
BARE_PART = frozenset(OPS) - PROPAGATED - DECORATED - REDUCTIONS - {'b-textToInterval', 'b-numsToInterval'}
# M13g part 3: rows on a decoration alone, for the plain pass only, keyed on the statement WITH its
# decorations (`exp2 [1024.0,1024.0] = [max,infinity]` without them is also its bare twin's key,
# which matches). the plain pass is exact; the outward pass rounds to doubles as 1788 does, so each
# of these must match there (`check`, `test_divergence_rows`)
_OVERFLOWS_ONLY_ROUNDED = ('decoration expectations: the exact result is bounded (a rational past the '
                           'doubles), so com fits it here; 1788 decorates its binary64 result, which '
                           'overflows to infinity, and demotes com to dac. the outward pass rounds to '
                           'doubles as 1788 does, and matches')
_MAX = '0x1.FFFFFFFFFFFFFp1023'
PLAIN_ONLY = {text: _OVERFLOWS_ONLY_ROUNDED for text in (
    f'add [1.0,2.0]_com [5.0,{_MAX}]_com = [6.0,infinity]_dac',
    f'add [-{_MAX},2.0]_com [-0.1, 5.0]_com = [-infinity,7.0]_dac',
    f'sub [-1.0,2.0]_com [5.0,{_MAX}]_com = [-infinity,-3.0]_dac',
    f'sub [-{_MAX},2.0]_com [-1.0, 5.0]_com = [-infinity,3.0]_dac',
    f'mul [1.0,2.0]_com [5.0,{_MAX}]_com = [5.0,infinity]_dac',
    f'mul [-{_MAX},2.0]_com [-1.0, 5.0]_com = [-infinity,{_MAX}]_dac',
    'div [-200.0,-1.0]_com [0x0.0000000000001p-1022, 10.0]_com = [-infinity,-0X1.9999999999999P-4]_dac',
    f'sqr [-{_MAX},-0x0.0000000000001p-1022]_com = [0.0,infinity]_dac',
    f'fma [1.0,2.0]_com [1.0, {_MAX}]_com [0.0,1.0]_com = [1.0,infinity]_dac',
    f'pown [-{_MAX},2.0]_com 2 = [0.0,infinity]_dac',
    f'pown [-{_MAX},2.0]_com 3 = [-infinity, 8.0]_dac',
    'exp2 [1024.0,1024.0]_com = [0X1.FFFFFFFFFFFFFP+1023,infinity]_dac',  # 2 ** 1024, an exact int
)}
DIVERGENCES.update({key(v): _NO_IS_NAI for v in VECTORS if v.op == 'isNaI' and not _has_nai(v)})
INTERVAL_VECTORS = tuple(v for v in INTERVAL_VECTORS if v.op not in CONSTRUCTORS)
# M13's merge: rows on a decoration alone, in both passes, keyed on the statement WITH its decorations
# (the bare twin, `mulRevToPair [-2.0, -0.1] [-2.1, -0.4] = ...`, matches). such a row's set must match
# and only its decoration differ (`check`). 1788 decorates mulRevToPair's first interval as the
# decorated division `c / b` where `0 ∉ b` (com, dac or def), though mulRev, the same set's hull, is
# trv there (`mulRev [-2.0, -0.1]_dac [-2.1, -0.4]_dac = [...]_trv`); ours is the one op, `mul_rev`,
# trv as every reverse op, which is sound (trv claims nothing) and 1788's mulRev
_PAIR_DECORATED_AS_DIVISION = ('decoration expectations: 1788 decorates mulRevToPair\'s first interval '
                               'as the decorated division c / b where 0 is not in b; ours is one set, '
                               'mul_rev\'s, trv as 1788 decorates mulRev and every other reverse op. '
                               '1788\'s pair with its decoration is ieee1788.mul_rev_to_pair, which '
                               'matches (Q9, closed as built)')
DECORATION_ONLY = {v.text: _PAIR_DECORATED_AS_DIVISION for v in VECTORS if v.op == 'mulRevToPair'
                   and not _has_nai(v) and v.expected[0].decoration not in (None, 'trv')}


# THE ADAPTER

class NoCounterpart(ValueError):
    """a 1788 value the core has no counterpart for (NaI)"""


def to_ours(literal, cls=MultiInterval, as_float=False, decorated=False):
    """the input rule; `as_float` gives the literal's doubles as floats instead of Fractions;
    `decorated` (M13g) keeps a literal's decoration, as a `DecoratedInterval`"""
    if not isinstance(literal, Interval):
        return float(literal) if as_float and isinstance(literal, Fraction) else literal
    if literal.nai:
        raise NoCounterpart('NaI')
    if literal.empty:
        bare = cls()
    else:
        lo, hi = (float(literal.lo), float(literal.hi)) if as_float else (literal.lo, literal.hi)
        bare = cls(lo, hi, start_closed=literal.lo != -math.inf, end_closed=literal.hi != math.inf)
    return DecoratedInterval(bare, literal.decoration) if decorated and literal.decoration else bare


def is_decorated(vector) -> bool:
    """a decoration on an operand or on the result (M13g part 3). `Interval` is a NamedTuple, so only a
    two-value result is unpacked: unpacking an interval would read its fields, never its decoration"""
    pair = isinstance(vector.expected, tuple) and not isinstance(vector.expected, Interval)
    values = (*vector.args, *(vector.expected if pair else (vector.expected,)))
    return any(isinstance(v, Interval) and v.decoration for v in values)


def _args(vector, cls=MultiInterval, as_float=False):
    """the operands under the input rule. a decorated op's, and a decorated vector's (M13g part 3), are
    `DecoratedInterval`s, so each decoration must fit its set; a BARE_PART op then gets the interval
    part of each"""
    if vector.op not in DECORATED and not is_decorated(vector):
        return [to_ours(a, cls, as_float) for a in vector.args]
    ours = [to_ours(a, cls, as_float, decorated=True) for a in vector.args]
    if vector.op in BARE_PART:
        return [a.interval if isinstance(a, DecoratedInterval) else a for a in ours]
    return ours


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
    args = _args(vector, cls, as_float=True)
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


def _signalled(vector, outward=False):
    """(ours, expected) as (value, signal) pairs, for an op in SIGNALLED (M13g): a raised
    UndefinedOperationError is 1788's answer to invalid input with UndefinedOperation, empty for a
    bare op and NaI for a decorated one. `outward`: the operands as floats in OutwardMultiInterval"""
    args = _args(vector, OutwardMultiInterval, True) if outward else _args(vector)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = OPS[vector.op](*args)
        except UndefinedOperationError:
            ours = _RAISED if vector.op in DECORATED else None, 'UndefinedOperation'
        else:
            possibly = any(issubclass(w.category, PossiblyUndefinedOperationWarning) for w in caught)
            ours = (_ours(result, _outward_hull if outward else closed_hull_of_ours),
                    'PossiblyUndefinedOperation' if possibly else None)
    return ours, (_expected(vector), vector.signal)


# M13g: the decorated flavour's answer to invalid input, which ours gives by raising
_RAISED = '[nai], raised as UndefinedOperationError'


def _ours(result, hull):
    """our value under the output rule (`hull`); a DecoratedInterval is (closed hull, decoration)
    and a Decoration its name (M13g)"""
    if isinstance(result, DecoratedInterval):
        return hull(result.interval), result.decoration.value
    if isinstance(result, Decoration):
        return result.value
    return hull(result)


def _expected(vector):
    """1788's value under the output rule; a decorated one, for a decorated or (M13g part 3)
    propagating op, is (closed hull, decoration), and `[nai]` with UndefinedOperation is what a raise
    gives (M13g)"""
    literal = vector.expected
    if not isinstance(literal, Interval):
        return literal
    if _nai_is_a_raise(vector):
        return _RAISED
    hull = closed_hull_of_expected(literal)
    decorated = vector.op in DECORATED or vector.op in PROPAGATED
    return (hull, literal.decoration) if decorated and literal.decoration else hull


def run(vector):
    """(ours, expected), both through the adapter"""
    if vector.op in SIGNALLED:
        return _signalled(vector)
    if vector.op in DECORATED:
        return _ours(_call(vector, _args(vector)), closed_hull_of_ours), _expected(vector)
    if vector.op in NUMERIC:
        return _numeric(vector, _args(vector)), _numbers_as_floats(vector.expected)
    if vector.op in REDUCTIONS:
        return _reduce(vector, [to_ours(a) for a in vector.args]), float(vector.expected)
    result = _call(vector, _args(vector))
    if vector.op in PAIRS:
        return _pair_outcome(vector, result, rounded=True)
    if isinstance(vector.expected, Interval):
        return _ours(result, closed_hull_of_ours), _expected(vector)
    if isinstance(vector.expected, Fraction) or isinstance(vector.expected, float):
        return round_down(result) if vector.op == 'inf' else round_up(result), float(vector.expected)
    return result, vector.expected


def run_outward(vector):
    """(ours, expected): float operands in an OutwardMultiInterval, our closed hull taken exactly"""
    if vector.op in SIGNALLED:
        return _signalled(vector, outward=True)
    if vector.op in DECORATED:
        return _ours(_call(vector, _args(vector, OutwardMultiInterval, True)), _outward_hull), _expected(vector)
    result = _call(vector, _args(vector, OutwardMultiInterval, as_float=True))
    if vector.op in PAIRS:
        return _pair_outcome(vector, result, rounded=False)
    return _ours(result, _outward_hull), _expected(vector)


def _outward_hull(result):
    """the outward rule's closed hull, taken exactly: `None` for empty, else `(lo, hi)`"""
    assert isinstance(result, OutwardMultiInterval), type(result)
    if result.is_empty:
        return None
    lo, hi = result.inf, result.sup
    # every finite end is a double already: nothing is left for the adapter to round
    assert all(isinstance(v, float) or v == round_down(v) for v in (lo, hi)), (lo, hi)
    return lo, hi


# the pair rule (M13e, mulRevToPair): 1788 gives the preimage as two intervals, the second empty unless
# the set has a gap; ours is the one multi-interval. each of our pieces is closed (its ends rounded
# outward in the first pass, taken as they are in the outward pass) and compared, in order, with the
# pair's non-empty intervals. a decorated pair (M13's merge) is (pieces, decoration) on both sides: 1788's
# decoration is its non-empty intervals' (the empty ones' if none is; each must be the same, or none matches)

def _pair_outcome(vector, result, rounded: bool):
    """(ours, expected) under the pair rule; with decorations if ours is a DecoratedInterval"""
    if not isinstance(result, DecoratedInterval):
        return _pair(result, rounded), _pair_of_expected(vector.expected)
    members = vector.expected
    decorations = {m.decoration for m in members if not m.empty} or {m.decoration for m in members}
    decoration = decorations.pop() if len(decorations) == 1 else tuple(sorted(map(str, decorations)))
    return (_pair(result.interval, rounded), result.decoration.value), (_pair_of_expected(members), decoration)


def _pair(result: MultiInterval, rounded: bool):
    out = []
    for lo, _, hi, _ in pieces(result.cuts):
        if rounded:
            lo, hi = round_down(lo), round_up(hi)
        else:  # every finite end is a double already: nothing is left for the adapter to round
            assert all(isinstance(v, float) or v == round_down(v) for v in (lo, hi)), (lo, hi)
        out.append((lo, hi))
    return tuple(out)


def _pair_of_expected(pair):
    return tuple(h for h in map(closed_hull_of_expected, pair) if h is not None)


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

def row(vector, outward=False):
    """the reason a vector is a divergence row in the plain or the outward pass, else None"""
    reason = DIVERGENCES.get(key(vector)) or DECORATION_ONLY.get(vector.text)
    return PLAIN_ONLY.get(vector.text) if reason is None and not outward else reason


def check(vector, runner, outward=False):
    ours, expected = outcome(runner, vector)
    if vector.text in DECORATION_ONLY:  # (set, decoration): the set matches, the decoration does not
        assert same(ours[0], expected[0]), vector.text
        assert not same(ours[1], expected[1]), f'stale divergence row, it matches now: {vector.text}'
    elif row(vector, outward) is not None:
        assert not same(ours, expected), f'stale divergence row, it matches now: {vector.text}'
    else:
        assert same(ours, expected), vector.text


@pytest.mark.parametrize('vector', VECTORS, ids=[v.source for v in VECTORS])
def test_vector(vector):
    check(vector, run)


@pytest.mark.parametrize('vector', INTERVAL_VECTORS, ids=[v.source for v in INTERVAL_VECTORS])
def test_vector_outward(vector):
    check(vector, run_outward, outward=True)


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
    # M13g part 3: a plain-only row is one interval-valued vector, whose outward pass must match, and
    # is not under a row already
    for text, reason in PLAIN_ONLY.items():
        found = [v for v in INTERVAL_VECTORS if v.text == text]
        assert len(found) == 1 and key(found[0]) not in DIVERGENCES, text
        assert reason.startswith(REASONS), reason
    # M13's merge: a decoration-only row is one pair vector, under no other row, whose b has no 0 (so
    # 1788's first interval is the decorated division); 52 of mulRevToPair's 174 decorated vectors
    # (2026-09-27)
    assert len(DECORATION_ONLY) == 52 and not set(DECORATION_ONLY) & set(PLAIN_ONLY)
    for text, reason in DECORATION_ONLY.items():
        found = [v for v in INTERVAL_VECTORS if v.text == text]
        assert len(found) == 1 and key(found[0]) not in DIVERGENCES and found[0].op in PAIRS, text
        b = found[0].args[0]
        assert not b.empty and not b.lo <= 0 <= b.hi, text
        assert reason.startswith(REASONS), reason
    # and every category has a row: a category with none reads as a claim the census contradicts
    # ('domain-clipped functions' had none since M13d and was removed, owner 2026-10-03)
    reasons = [*DIVERGENCES.values(), *PLAIN_ONLY.values(), *DECORATION_ONLY.values()]
    assert all(any(r.startswith(category) for r in reasons) for category in REASONS), \
        [category for category in REASONS if not any(r.startswith(category) for r in reasons)]


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


def test_quoted_strings_keep_their_white_space(tmp_path):
    """vectors-ext (a), 2026-10-05: a quoted string is the text constructor's argument character for
    character. the parser collapsed white space inside one, so 40 textToInterval vectors of
    `libieeep1788_class.itl` (`"[ Empty  ]"`, `"[-I  nf, 1.000 ]"`) ran on `"[ Empty ]"` and the like, a
    weaker check than upstream's. white space outside a quoted string is still collapsed"""
    lines = {name: (HERE / name).read_text(encoding='utf-8').splitlines() for name in FILES}
    quoted = [v for v in VECTORS if any(isinstance(a, Text) for a in v.args)]
    for v in quoted:  # each statement is on one line, the line of its source
        name, line = v.source.rsplit(':', 1)
        assert [a.value for a in v.args if isinstance(a, Text)] == re.findall(
            r'"([^"]*)"', lines[name][int(line) - 1]), v.source
    assert sum(v.args[0].value != ' '.join(v.args[0].value.split()) for v in quoted) == 40
    (tmp_path / 'q.itl').write_text('testcase t {\n  b-textToInterval \t "[  1 ,\t2\n ]"\n  =  [1.0,  2.0] ;\n}\n',
                                    encoding='utf-8')
    (vector,), _ = parse_file(tmp_path / 'q.itl')
    assert vector.args == (Text('[  1 ,\t2\n ]'),)
    assert vector.text == 'b-textToInterval "[  1 ,\t2\n ]" = [1.0, 2.0]'


def test_quoted_strings_keep_comment_marks(tmp_path):
    """vectors-ext (a): a `//` or `/*` inside a quoted string is not a comment (none of the vendored
    strings has one); a comment after the statement still is"""
    (tmp_path / 'q.itl').write_text('testcase t {\n  b-textToInterval "[1, 2]//a /*b*/" = [nai]; // c\n}\n',
                                    encoding='utf-8')
    (vector,), _ = parse_file(tmp_path / 'q.itl')
    assert vector.args == (Text('[1, 2]//a /*b*/'),) and vector.source == 'q.itl:2'


def test_signals_are_checked():
    """every op whose vectors carry a 1788 signal has it compared (M13g)"""
    assert {v.op for v in VECTORS if v.signal} <= SIGNALLED


def test_decorated_ops_are_checked():
    """M13g: the decorated ops' vectors carry decorations and are compared with them, and those with an
    interval operand run outward too; only the constructors, which have none, are not"""
    assert {v.op for v in VECTORS if v.op in DECORATED} == DECORATED
    assert {v.op for v in INTERVAL_VECTORS if v.op in DECORATED} == DECORATED - CONSTRUCTORS - {'decorationPart'}
    assert not {v.op for v in INTERVAL_VECTORS} & CONSTRUCTORS


def test_decorated_vectors_run_decorated():
    """M13g part 3: every decorated vector runs through the decorated type, and a propagating op's
    result is compared with its decoration on both sides (an adapter dropping it on both would pass)"""
    ops = {v.op for v in VECTORS if is_decorated(v)}
    assert ops <= PROPAGATED | BARE_PART | DECORATED and ops & PROPAGATED and ops & BARE_PART
    assert PROPAGATED <= set(OPS) and not PROPAGATED & BARE_PART
    # an interval-valued op is never read on the interval part alone, which would drop its decoration
    assert {v.op for v in INTERVAL_VECTORS if is_decorated(v)} - DECORATED <= PROPAGATED
    assert not {v.op for v in INTERVAL_VECTORS} & BARE_PART
    first = {}
    for v in INTERVAL_VECTORS:
        if v.op in PROPAGATED and is_decorated(v) and row(v) is None:
            first.setdefault(v.op, v)
    assert len(first) >= 40, sorted(first)
    names = {d.value for d in Decoration}
    for v in first.values():
        for runner in (run, run_outward):
            ours, expected = runner(v)
            assert ours[1] in names and expected[1] in names, v.text


def test_undefined_operation_is_never_a_row():
    """M13g: 1788's answer to invalid input, `signal UndefinedOperation` with `[empty]` (bare) or `[nai]`
    (decorated), is matched by our raise; a rule that turned those vectors into rows would pass them all"""
    signalled = [v for v in VECTORS if v.signal == 'UndefinedOperation']
    assert any(v.expected.nai for v in signalled) and any(v.expected.empty for v in signalled)
    assert not [v.source for v in signalled if key(v) in DIVERGENCES]


def test_outward_pass_of_a_decorated_op_is_outward(monkeypatch):
    """M13g: the decorated ops' outward pass gives the set as floats in an OutwardMultiInterval (newDec
    does no arithmetic, so its values alone cannot tell the classes apart)"""
    seen = []

    def spy(x, *rest):
        seen.append(x)
        return DecoratedInterval(x, *rest)

    monkeypatch.setitem(OPS, 'newDec', spy)
    vector = next(v for v in INTERVAL_VECTORS if v.op == 'newDec' and not v.expected.empty)
    assert same(*run_outward(vector))
    assert type(seen[-1]) is OutwardMultiInterval and isinstance(seen[-1].inf, float)
    run(vector)
    assert type(seen[-1]) is MultiInterval and isinstance(seen[-1].inf, Fraction)


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


def test_signalled_reads_only_undefined_operation(monkeypatch):
    """M13g review: only `UndefinedOperationError` is 1788's signal. a plain `ValueError` escapes, so a
    library raising the wrong class fails the vectors instead of matching them"""
    vector = next(v for v in VECTORS if v.op == 'b-textToInterval' and v.signal == 'UndefinedOperation')

    def plain(*args):
        raise ValueError('plain')

    monkeypatch.setitem(OPS, vector.op, plain)
    with pytest.raises(ValueError, match='plain'):
        _signalled(vector)


def test_is_decorated_sees_the_result_alone():
    """a decoration on the result alone counts (2026-09-27: no vector has one yet, and an `Interval`
    result used to be unpacked into its fields, so its decoration was never seen)"""
    bare, com = Interval(Fraction(1), Fraction(2)), Interval(Fraction(1), Fraction(2), 'com')
    for expected, want in ((com, True), (bare, False), ((bare, com), True), ((bare, bare), False)):
        v = Vector('x.itl:1', 't', 'sqr', (bare,), expected, 'sqr [1, 2] = [1, 2]')
        assert is_decorated(v) is want, expected


def test_no_decorated_pair_goes_unchecked():
    """M13g's hook, wired at M13's merge: every decorated pair vector (mulRevToPair: 174, 2026-09-27)
    runs through the decorated type, and ours and 1788's are compared as (pieces, decoration) in both
    passes, so a pair's decoration is checked (an adapter dropping it on both sides would pass)"""
    pairs = [v for v in VECTORS if is_decorated(v) and isinstance(v.expected, tuple)
             and any(isinstance(e, Interval) for e in v.expected)]
    assert {v.op for v in pairs} == PAIRS and len(pairs) == 174
    assert PAIRS <= PROPAGATED
    names = {d.value for d in Decoration}
    checked = [v for v in pairs if not _has_nai(v)]
    assert len(checked) == 172
    for v in checked:
        for runner in (run, run_outward):
            ours, expected = runner(v)
            assert ours[1] in names and expected[1] in names, v.text


def test_decorated_reverse_vectors_are_checked():
    """M13's merge: every decorated vector of a reverse op (481, 2026-09-27) runs through the decorated
    type, its decoration compared in both passes, but for a `[nai]` operand (a row, D16)"""
    decorated = [v for v in VECTORS if v.op in REVERSE and is_decorated(v)]
    assert len(decorated) == 481 and REVERSE <= PROPAGATED
    names = {d.value for d in Decoration}
    for v in decorated:
        if _has_nai(v):
            assert key(v) in DIVERGENCES, v.text
            continue
        for runner in (run, run_outward):
            ours, expected = runner(v)
            assert ours[1] in names and expected[1] in names, v.text


def test_every_op_has_vectors():
    """a parser that finds nothing for an op would pass every vector of it vacuously"""
    assert {v.op for v in VECTORS} == set(OPS)


def test_every_file_is_used():
    """every vendored file is read, and every file read has statements"""
    assert {p.name for p in HERE.glob('*.itl')} == set(FILES) and len(FILES) == 19
    for name in FILES:
        assert any(v.source.startswith(f'{name}:') for v in VECTORS) or SKIPPED[name], name


def test_nothing_is_skipped():
    """M13's exit: no statement of the 19 files is skipped. every op of every file is in OPS, so an op
    dropped from OPS, or a file gaining one, goes red here (`test_parser_drops_nothing` only checks that
    an op in OPS is not skipped, and `test_every_op_has_vectors` checks OPS against the vectors)"""
    assert set(SKIPPED) == set(FILES)
    assert not {name: counts for name, counts in SKIPPED.items() if counts}


def test_the_pair_vectors_run_outward():
    """M13e's review (2026-09-27): a pair vector's expected value is no `Interval`, so only `or v.op in
    PAIRS` puts it in `INTERVAL_VECTORS`, the outward pass; dropping that removed 347 items, none red"""
    pairs = {v.source for v in VECTORS if v.op in PAIRS}
    assert pairs and pairs <= {v.source for v in INTERVAL_VECTORS}


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
