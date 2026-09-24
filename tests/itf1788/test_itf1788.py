"""
ieee 1788 conformance: itf1788 vectors through the adapter in v2-plan.md "ieee 1788"

the vendored `.itl` files are unmodified copies from https://github.com/nehmeier/ITF1788 (Apache
2.0, see LICENSE and NOTICE here; README.md pins the commit). the adapter's rules:

* **input rule**: a 1788 unbounded bound is our open-at-inf, `[1, infinity]` is `[1, inf)` and
  `[entire]` is `(-inf, inf)`, because 1788 never attains infinity. a finite bound is closed
* **precision rule**: operands are the literals' doubles, held exactly (tests.itf1788.itl), and our
  exact result is rounded outward to doubles. an expected value is the tightest double enclosure,
  so this checks soundness and sharpness together, not just overlap. the package's own float path
  rounds to nearest (the rounding hook's identity default) and is not what is tested here
* **output rule**: the closed hull of **both** our result and the expected value before comparing.
  it absorbs multi-interval vs connected (`1/[-10, 10]`: ours `[-inf, -1/10] ∪ [1/10, inf]`, 1788
  entire) and our attained infinities vs 1788's unattained ones
* **decorations**: dropped from inputs and expected values; only the bare interval is compared.
  decorations are not in the core (v2-plan.md "ieee 1788"), so no vector checks one
* the library's warnings are ignored inside a vector (`1/[0]` is `∅` + `IndeterminateResultWarning`,
  and 1788's answer is also empty); they are pinned by their own tests elsewhere

every vector of an op in `OPS` either matches through the adapter or is a row of `DIVERGENCES`, whose
reason is one of the plan's residual categories. a row that starts matching fails as stale.
"""
import math
import re
import warnings
from fractions import Fraction
from pathlib import Path

import pytest

from intervals import EMPTY
from intervals import MultiInterval
from intervals.errors import IntervalWarning
from tests.itf1788.itl import Interval
from tests.itf1788.itl import parse_file

HERE = Path(__file__).parent
FILES = ('libieeep1788_tests_elem.itl', 'libieeep1788_tests_set.itl')

# the used subset: every 1788 op the package implements
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
    'floor': lambda a: a.floor(),
    'intersection': lambda a, b: a & b,
    'convexHull': lambda a, b: (a | b).hull,
}

# the plan's residual categories (v2-plan.md "ieee 1788"); a row's reason starts with one of them
REASONS = ('degenerate infinities', 'domain-clipped functions', 'decoration expectations')

# (statement text, whitespace collapsed) -> reason. empty as of 2026-09-24: no vector of the used
# subset diverges. degenerate infinities cannot be written in 1788, decorations are dropped by the
# adapter, and no domain-clipped function (sqrt, log, ...) is implemented yet
DIVERGENCES = {
}


def _load():
    vectors, skipped = [], {}
    for name in FILES:
        found, other = parse_file(HERE / name, OPS)
        vectors.extend(found)
        skipped[name] = other
    return tuple(vectors), skipped


VECTORS, SKIPPED = _load()


# THE ADAPTER

def to_ours(literal):
    """the input rule"""
    if not isinstance(literal, Interval):
        return literal
    if literal.nai:
        raise ValueError('NaI has no counterpart')
    if literal.empty:
        return EMPTY
    return MultiInterval(literal.lo, literal.hi,
                         start_closed=literal.lo != -math.inf, end_closed=literal.hi != math.inf)


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
        raise ValueError('NaI has no counterpart')
    return None if literal.empty else (float(literal.lo), float(literal.hi))


def run(vector):
    """(ours, expected), both through the adapter"""
    args = [to_ours(a) for a in vector.args]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntervalWarning)
        result = OPS[vector.op](*args)
    return closed_hull_of_ours(result), closed_hull_of_expected(vector.expected)


# THE VECTORS

@pytest.mark.parametrize('vector', VECTORS, ids=[v.source for v in VECTORS])
def test_vector(vector):
    ours, expected = run(vector)
    if vector.text in DIVERGENCES:
        assert ours != expected, f'stale divergence row, it matches now: {vector.text}'
    else:
        assert ours == expected, vector.text


def test_divergence_rows():
    texts = {v.text for v in VECTORS}
    for text, reason in DIVERGENCES.items():
        assert text in texts, f'no such vector: {text}'
        assert reason.startswith(REASONS), reason


@pytest.mark.parametrize('name', FILES)
def test_parser_drops_nothing(name):
    """an independent count: every statement line of a used op became a vector"""
    text = (HERE / name).read_text(encoding='utf-8')
    for op in OPS:
        lines = re.findall(rf'^\s*{op}\s', text, re.MULTILINE)
        parsed = [v for v in VECTORS if v.op == op and v.source.startswith(f'{name}:')]
        assert len(parsed) == len(lines), op
    assert not set(OPS) & set(SKIPPED[name])


def test_every_op_has_vectors():
    """a parser that finds nothing for an op would pass every vector of it vacuously"""
    assert {v.op for v in VECTORS} == set(OPS)


# THE ADAPTER'S OWN RULES

def test_input_rule():
    assert to_ours(Interval(Fraction(1), math.inf)) == MultiInterval(1, math.inf, end_closed=False)
    assert to_ours(Interval(-math.inf, math.inf)) == MultiInterval.parse('(-inf, inf)')
    assert to_ours(Interval(None, None)) == EMPTY


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
