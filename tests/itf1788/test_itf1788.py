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
* **decorations**: dropped from inputs and expected values; only the bare interval is compared.
  decorations are not in the core (v2-plan.md "ieee 1788"), so no vector checks one, and a
  divergence row is keyed on the statement with its decorations stripped (the fork has many
  statements twice, `atanh [1.0,1.0]_def = [empty]_trv` beside `atanh [1.0,1.0] = [empty]`).
  NaI has no counterpart, so a vector with a `[nai]` in it is a row until M13g, generated below
* a `signal` clause is kept on the vector and not checked yet (M13g). `NaN` equals `NaN` here
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
from intervals.errors import IntervalWarning
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
                                          'asinh', 'acosh', 'atanh')},
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
    # numbers: 1788 gives +infinity for the infimum of the empty set, -infinity for its supremum
    'inf': lambda a: a.inf if a else math.inf,
    'sup': lambda a: a.sup if a else -math.inf,
    # the overlap state: our allen relation, named as in 1788
    'overlap': lambda a, b: _overlap(a, b),
}

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
           'cut-based relations')

_LOG = ('degenerate infinities: the operand meets the domain [0, inf] only at 0, and log(0) is '
        '-inf here (the limit from the one side the domain has); 1788 drops 0 from the domain')
_ATANH = ('degenerate infinities: the operand meets the domain [-1, 1] only at an end, and atanh(±1) '
          'is ±inf here (the limit from inside); 1788 drops ±1 from the domain')
_NAI = ('decoration expectations: NaI is not a set, so the undecorated core has no counterpart for it '
        '(D16: it arrives with the decorated wrapper type, M13g)')
_MEETS = ('cut-based relations: two closed intervals that share an end share that point, so they '
          'overlap (relations.Allen, on cuts); 1788 calls touching closed intervals meets or metBy')

# (statement text, whitespace collapsed and decorations stripped) -> reason. as of 2026-09-25 every
# listed row is a degenerate infinity of a function at the end of its domain, or a touching pair that
# shares a point; the NaI rows are generated once the vectors are loaded
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
}
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
DIVERGENCES.update({key(v): _NAI for v in VECTORS if _has_nai(v)})


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


def run(vector):
    """(ours, expected), both through the adapter"""
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


def same(ours, expected) -> bool:
    """equality, except that NaN is NaN (1788's answer for a number of the empty set or of NaI)"""
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
