"""
outward float rounding at the edges of the float range: subnormals, near-overflow magnitudes, +-inf

`test_ops_properties.py::test_sound_float_outward_rounding` closes every result before it checks
soundness and draws floats from `[-20, 20]` only. here the open/closed flags are read as given, and
the operands mix subnormals (5e-324, 1e-320, the smallest normal), 1e308, `sys.float_info.max`,
1e+-300, 1e154 and `ldexp` values across the whole exponent range with +-inf, ints and Fractions.
every value an operand pair attains, computed exactly, must be in the rounded result.

float-only boxes use either a one-ulp `nextafter` hook or the exact `_round` hook; mixed boxes and
`**` use the exact one. the hook is also audited: it must never see an infinite argument (a pole or
an infinite corner is resolved before it) and must always see at least one float (exact arguments
need no rounding). the fuzz was written for the M6 review (2026-09-24) and moved here at M11.

the classes end to end (M14-breadth, 2026-10-02): hypothesis draws operands from the same pools, and
every op is written as python writes it (`a % b`, `3.0 // B`, `divmod`, `A ** 2.0`, `A.fma(B, C)`,
`A.minimum(B)`), with a MultiInterval or a python number on either side of an OutwardMultiInterval:

* OutwardMultiInterval, `+ - * /`, reciprocal, pown, `%`, `//`, divmod, fma, neg, abs, minimum and
  maximum: the result is outward whichever side it is on, and every value a sampled point attains,
  computed exactly by `tests.oracles.pointwise` (`//` by `_floordiv` below), is in it, flags as given
* against the same op on the operands read exactly: the exact set is inside the result, each hull end
  is the exact end rounded down or up (the tightest enclosure), closed iff unmoved and closed, and
  every closed end of a piece is an exact value
* MultiInterval promises no enclosure (v2-plan "arithmetic", rounding; README "departures from ieee
  1788"), so only what rounding to nearest implies: empty iff the exact result is, hull ends rounded
  to nearest, and the nearest double of every attained value in the result read with every end closed
* neg, abs, minimum and maximum never round: both classes give the same set, the attained set itself
"""
import math
import operator
import random
import sys
import warnings
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import ops
from intervals.applicator import apply_binary
from intervals.applicator import apply_unary
from intervals.kernel import contains_point
from intervals.kernel import is_subset
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from tests.oracles import _exact
from tests.oracles import _exact_cuts
from tests.oracles import _inf
from tests.oracles import _negated
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import sample
from tests.strategies import infinities
from tests.strategies import piece_pairs
from tests.strategies import probe_points
from tests.test_ops_properties import _round

INF = math.inf
MAX = sys.float_info.max
TINY = 2.2250738585072014e-308  # the smallest normal float

FLOAT_POOL = [0.0, 1.0, -1.0, 2.5, -2.5, 0.1, -0.1, 1e308, -1e308, MAX, -MAX, 5e-324, -5e-324, 1e-320,
              TINY, -TINY, 1e-300, -1e-300, 1e300, -1e300, 1e-160, 3e-162, 1e154, -1e154, 1e17, 3.0, 0.5, -0.5]
EXACT_POOL = [0, 1, -1, 2, -3, Fraction(1, 3), Fraction(-2, 7), 10**20, -10**20, Fraction(1, 10**30)]

BINARY = {'add': ops.ADD, 'sub': ops.SUB, 'mul': ops.MUL, 'div': ops.DIV}
UNARY = {'reciprocal': ops.RECIPROCAL, 'neg': ops.NEG, 'abs': ops.ABS}
EXPONENTS = [-3, -2, -1, 1, 2, 3, 5]
BOXES = 1000


def _value(rng, floats_only):
    r = rng.random()
    if r < 0.07:
        return INF
    if r < 0.14:
        return -INF
    if r < 0.5:
        return rng.choice(FLOAT_POOL)
    if r < 0.8 or floats_only:
        return rng.choice((-1, 1)) * math.ldexp(rng.random(), rng.randrange(-1074, 1024))
    return rng.choice(EXACT_POOL)


def _cuts(rng, floats_only):
    ps = []
    for _ in range(rng.randrange(1, 4)):
        lo, hi = sorted((_value(rng, floats_only), _value(rng, floats_only)))
        if rng.random() < 0.15 or lo == hi:
            ps.append(piece(lo, lo))
        else:
            ps.append(piece(lo, hi, rng.random() < 0.5, rng.random() < 0.5))
    return normalize(ps)


def _float_samples(cuts, rng):
    """closed ends, floats one to three ulps inside each end, one spread point per piece, oracle samples"""
    out = []
    for lo, lc, hi, hc in pieces(cuts):
        if lc or lo == hi:
            out.append(lo)
        if hc:
            out.append(hi)
        for end, other in ((lo, hi), (hi, lo)):
            if math.isfinite(end) and end != other:
                x = end
                for _ in range(rng.randrange(1, 4)):
                    x = math.nextafter(x, other)
                if (lo < x < hi) or (x == lo and lc) or (x == hi and hc):
                    out.append(x)
        flo, fhi = max(lo, -MAX), min(hi, MAX)
        if flo < fhi:
            x = flo + (fhi - flo) * rng.random() if math.isfinite(fhi - flo) else rng.uniform(flo / 2, fhi / 2)
            if lo < x < hi:
                out.append(x)
    return out + sample(cuts, 6, rng)


class Hooks:
    """rounding hooks that record every argument tuple they are called with"""

    def __init__(self):
        self.seen = []

    def exact(self, fn, direction):
        inner = _round(fn, direction)

        def hook(*args):
            self.seen.append(args)
            return inner(*args)
        return hook

    def ulp(self, fn, direction):
        """one ulp outward from python's correctly rounded result: sound for + - * /"""
        def hook(*args):
            self.seen.append(args)
            return math.nextafter(fn(*args), direction * INF)
        return hook

    def none(self, fn, direction):
        """the sabotage: a hook that does not round"""
        def hook(*args):
            self.seen.append(args)
            return fn(*args)
        return hook


def _box(rng, hooks, float_hook):
    """one random (op, descriptor, a, b-or-exponent); float-only boxes use `float_hook` half the time"""
    floats_only = rng.random() < 0.5
    op = rng.choice(list(BINARY) + list(UNARY) + ['pow'])
    a = _cuts(rng, floats_only)
    b = None
    if op in BINARY:
        b = _cuts(rng, floats_only)
        base = BINARY[op]
    elif op == 'pow':
        b = rng.choice(EXPONENTS)
        base = ops._power_descriptor(b)
    else:
        base = UNARY[op]
    make = float_hook if floats_only and op != 'pow' and rng.random() < 0.5 else hooks.exact
    return op, base._replace(rounded=(make(base.fn, -1), make(base.fn, 1))), a, b


def _unsound(op, desc, a, b, rng):
    """the first attained value missing from the rounded result, or None"""
    result = apply_binary(desc, a, b) if op in BINARY else apply_unary(desc, a)
    xs = _float_samples(a, rng)
    ys = _float_samples(b, rng) if op in BINARY else [None]
    for x in xs:
        for y in ys:
            if op == 'pow':
                values = pointwise('pow', _exact(x), b, a=_exact_cuts(a))
            elif op in BINARY:
                values = pointwise(op, _exact(x), _exact(y), a, b)
            else:
                values = pointwise(op, _exact(x), a=a)
            for v in values:
                if not contains_point(result, v):
                    return x, y, v, list(pieces(result))
    return None


def _fuzz(seed, float_hook_name, boxes=BOXES, until=None):
    """{op: first counterexample}, and the hooks' recorded arguments; stops early once `until` all failed"""
    rng = random.Random(seed)
    hooks = Hooks()
    float_hook = getattr(hooks, float_hook_name)
    failures = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(boxes):
            op, desc, a, b = _box(rng, hooks, float_hook)
            bad = _unsound(op, desc, a, b, rng)
            if bad and op not in failures:
                failures[op] = (list(pieces(a)), b if op == 'pow' else b and list(pieces(b)), bad)
                if until and until <= set(failures):
                    break
    return failures, hooks.seen


@pytest.mark.parametrize('seed', [0, 1])
def test_outward_rounding_sound_on_extreme_floats(seed):
    failures, seen = _fuzz(seed, 'ulp')
    assert failures == {}
    assert seen, 'the hook was never called: the fuzz tested nothing'
    assert not [args for args in seen if any(isinstance(x, float) and not math.isfinite(x) for x in args)][:3]
    assert not [args for args in seen if not any(isinstance(x, float) for x in args)][:3]


def test_fuzz_catches_a_hook_that_does_not_round():
    """sabotage: without outward rounding the float-only boxes must go unsound, op by op"""
    expected = {'add', 'sub', 'mul', 'div', 'reciprocal'}
    failures, _ = _fuzz(0, 'none', boxes=4 * BOXES, until=expected)
    assert set(failures) >= expected


# THE PRODUCTION DESCRIPTORS (ops.OUTWARD, what OutwardMultiInterval uses)

def _production(op, b):
    if op == 'pow':
        return ops._power_descriptor(b, True)
    return ops.OUTWARD.get(op) or UNARY[op]  # neg and abs are exact on floats


@pytest.mark.parametrize('seed', [0, 1])
def test_production_outward_descriptors_sound_on_extreme_floats(seed):
    rng = random.Random(1000 + seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(BOXES // 2):
            floats_only = rng.random() < 0.5
            op = rng.choice(list(BINARY) + list(UNARY) + ['pow'])
            a = _cuts(rng, floats_only)
            b = _cuts(rng, floats_only) if op in BINARY else rng.choice(EXPONENTS) if op == 'pow' else None
            bad = _unsound(op, _production(op, b), a, b, rng)
            assert bad is None, (op, list(pieces(a)), b if op == 'pow' else b and list(pieces(b)), bad)


def test_production_hook_matches_the_reference_hook():
    """the production hook and this suite's `_round` compute the same doubles"""
    rng = random.Random(5)
    for op, desc in list(ops.OUTWARD.items()) + [('pow3', ops._power_descriptor(3, True)),
                                                 ('pow-2', ops._power_descriptor(-2, True))]:
        base = ops._power_descriptor(int(op[3:])) if op.startswith('pow') else BINARY.get(op) or UNARY[op]
        for _ in range(300):
            args = tuple(_value(rng, True) for _ in range(1 if op.startswith('pow') or op == 'reciprocal' else 2))
            if any(math.isinf(x) for x in args) or (op in ('div', 'reciprocal', 'pow-2') and args[-1] == 0):
                continue
            for direction, hook in ((-1, desc.rounded[0]), (1, desc.rounded[1])):
                assert hook(*args) == _round(base.fn, direction)(*args), (op, args, direction)


# THE CLASSES, END TO END (M14-breadth): every op, spelled as python spells it

_ldexp_values = st.builds(lambda s, m, e: s * math.ldexp(m, e), st.sampled_from([-1, 1]),
                          st.floats(0.5, 1, exclude_max=True), st.integers(-1074, 1023))
# +-inf one draw in nine: an unbounded result has no finite end for rounding to move
extreme_values = st.one_of(st.sampled_from(FLOAT_POOL), st.sampled_from(FLOAT_POOL), st.sampled_from(FLOAT_POOL),
                           infinities, st.sampled_from(EXACT_POOL), _ldexp_values, _ldexp_values,
                           st.floats(allow_nan=False, allow_infinity=False), st.floats(allow_nan=False))
extreme_cuts = st.lists(piece_pairs(extreme_values), min_size=1, max_size=3).map(normalize).filter(bool)

OPERATORS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': operator.truediv,
             'mod': operator.mod, 'floordiv': operator.floordiv}
ROUNDED = ('add', 'sub', 'mul', 'div', 'reciprocal', 'pow', 'mod', 'floordiv', 'divmod', 'fma')
EXACT_OPS = ('neg', 'abs', 'minimum', 'maximum')  # a double's negation, abs, min and max are doubles
# O an OutwardMultiInterval, M a MultiInterval, s a python number (that operand is then a point)
OUTWARD_FORMS = ('OO', 'MO', 'OM', 'sO', 'Os')
NEAREST_FORMS = ('MM', 'sM', 'Ms')


def _make(letter, cuts):
    if letter == 's':
        (lo, _, _, _), = pieces(cuts)
        return lo
    return (OutwardMultiInterval if letter == 'O' else MultiInterval).from_cuts(cuts)


def _spelled(op, form, receiver, a, b, c, n):
    """
    {part: result} of the op through the class: an operator on two operands of `form`, or a method of a
    `receiver` (fma's factor and addend take `form`'s letters, min and max's other operand the second)
    """
    if op in OPERATORS:
        return {op: OPERATORS[op](_make(form[0], a), _make(form[1], b))}
    if op == 'divmod':
        q, r = divmod(_make(form[0], a), _make(form[1], b))
        return {'floordiv': q, 'mod': r}
    x = _make(receiver, a)
    if op == 'fma':
        return {op: x.fma(_make(form[0], b), _make(form[1], c))}
    if op in ('minimum', 'maximum'):
        return {op: getattr(x, op)(_make(form[1], b))}
    if op == 'pow':
        return {op: x ** n}
    if op == 'reciprocal':
        return {op: x.reciprocal()}
    return {op: -x if op == 'neg' else abs(x)}


def _operands(op, form, a, b, c, scalar):
    """the operands with the one a python number stands for made that point"""
    point = normalize([piece(scalar, scalar)])
    if op == 'fma':
        return a, point if form[0] == 's' else b, point if form[1] == 's' else c
    return point if form[0] == 's' else a, point if form[1] == 's' else b, c


def _floordiv(x, y, b):
    """exact x // y: floor(x / y) over a finite y (a pole by div's rule); the limit at y = +-inf (D8)"""
    if _inf(y):
        return [] if _inf(x) else [-1 if (x < 0 < y) or (y < 0 < x) else 0]
    return [v if _inf(v) else math.floor(v) for v in pointwise('div', x, y, b=b)]


def _values(op, x, y, z, a, b, n):
    """the exact values the point (x, y, z) of the operands attains, by tests.oracles' pointwise table"""
    x, y, z = (None if v is None else _exact(v) for v in (x, y, z))
    if op in ('add', 'sub', 'mul', 'div'):
        return pointwise(op, x, y, a, b)
    if op in ('reciprocal', 'neg', 'abs'):
        return pointwise(op, x, a=a)
    if op == 'pow':
        return pointwise('pow', x, int(n), a=_exact_cuts(a))
    if op == 'mod':
        return pointwise('mod', x, y)  # both exact: x - y * floor(x / y) in Fractions, or the D8 limit
    if op == 'floordiv':
        return _floordiv(x, y, b)
    if op == 'fma':
        return [w for p in pointwise('mul', x, y) for w in pointwise('add', p, z)]
    return [min(x, y) if op == 'minimum' else max(x, y)]


def _points(cuts, rng, n):
    """up to n of `_float_samples`, every closed end kept first: an op's extreme values are at the corners"""
    ends = [x for lo, lc, hi, hc in pieces(cuts) for x, closed in ((lo, lc), (hi, hc and hi != lo)) if closed]
    rest = [x for x in _float_samples(cuts, rng) if x not in ends]
    return ends[:n] + rng.sample(rest, min(len(rest), max(0, n - len(ends))))


def _attained_points(op, a, b, c, n, rng, k):
    """(x, y, z, values) over up to k sampled points per operand"""
    xs = _points(a, rng, k)
    ys = _points(b, rng, k) if op not in ('reciprocal', 'neg', 'abs', 'pow') else [None]
    zs = _points(c, rng, k) if op == 'fma' else [None]
    for x in xs:
        for y in ys:
            for z in zs:
                yield x, y, z, [(part, v) for part in (('floordiv', 'mod') if op == 'divmod' else (op,))
                                for v in _values(part, x, y, z, a, b, n)]


def _nearest(v):
    if _inf(v):
        return v
    try:
        return float(v) + 0.0  # correctly rounded for int and Fraction
    except OverflowError:
        return INF if v > 0 else -INF


def _closure(cuts):
    return normalize(piece(lo, hi) for lo, _, hi, _ in pieces(cuts))


def _has_float(*cuts):
    return any(isinstance(cut.value, float) and not _inf(cut.value) for x in cuts for cut in x)


def _ends(cuts):
    """(inf, inf closed, sup, sup closed) of a non-empty set"""
    ps = list(pieces(cuts))
    return ps[0][0], ps[0][1], ps[-1][2], ps[-1][3]


def _show(*cuts):
    return [list(pieces(x)) for x in cuts]


operations = dict(a=extreme_cuts, b=extreme_cuts, c=extreme_cuts, n=st.sampled_from(EXPONENTS),
                  float_n=st.booleans(), scalar=extreme_values)


@pytest.mark.parametrize('op', ROUNDED + EXACT_OPS)
@settings(max_examples=25, deadline=None)
@given(form=st.sampled_from(OUTWARD_FORMS), rng=st.randoms(use_true_random=False), **operations)
@pytest.mark.filterwarnings('ignore')
def test_outward_class_sound_on_extreme_floats(op, form, rng, a, b, c, n, float_n, scalar):
    """
    every value a sampled point attains, computed exactly, is in OutwardMultiInterval's result with its
    flags as given, whichever side a MultiInterval or a python number is on, and the result is outward
    """
    a, b, c = _operands(op, form, a, b, c, scalar)
    n = float(n) if float_n else n  # an integral float exponent is pown too (D11)
    results = _spelled(op, form, 'O', a, b, c, n)
    if op == 'divmod':
        assert results['floordiv'] == _make(form[0], a) // _make(form[1], b)
        assert results['mod'] == _make(form[0], a) % _make(form[1], b)
    for result in results.values():
        assert type(result) is OutwardMultiInterval, (op, form)
    for x, y, z, values in _attained_points(op, a, b, c, n, rng, 12 if op != 'fma' else 6):
        for part, v in values:
            assert contains_point(results[part].cuts, v), \
                (op, form, x, y, z, v, _show(a, b, c, results[part].cuts))


@pytest.mark.parametrize('op', ROUNDED)
@settings(max_examples=25, deadline=None)
@given(form=st.sampled_from(OUTWARD_FORMS), **operations)
@pytest.mark.filterwarnings('ignore')
def test_outward_class_is_the_tightest_enclosure(op, form, a, b, c, n, float_n, scalar):
    """
    against the same op on the operands read exactly (each float as the Fraction it denotes): the exact
    set is inside the result, the result's hull ends are the exact ones rounded down and up, closed iff
    the exact end is closed and was not moved, and every closed end of every piece is an exact value
    (an end that rounding moved is open: v2-plan "arithmetic", flags at rounded ends). an end from an
    exact box (no float in it) is exact; mod, // and fma round the whole result once if any operand
    has a float, and with none the result is the exact set
    """
    a, b, c = _operands(op, form, a, b, c, scalar)
    n = float(n) if float_n else n
    rounded = _spelled(op, form, 'O', a, b, c, n)
    exact = _spelled(op, 'OO', 'O', *(_exact_cuts(x) for x in (a, b, c)), n)
    operands = (a, b, c) if op == 'fma' else (a,) if op in ('reciprocal', 'pow') else (a, b)
    for part, result in rounded.items():
        r, x = result.cuts, exact[part].cuts
        why = (op, form, part, _show(a, b, c, r, x))
        if not _has_float(*operands):
            assert r == x, why
        assert bool(r) == bool(x), why
        if not x:
            continue
        assert is_subset(x, r), why
        once = part in ('mod', 'floordiv', 'fma') and _has_float(*operands)
        for end, closed, got, got_closed, direction in zip(_ends(x)[::2], _ends(x)[1::2], _ends(r)[::2],
                                                           _ends(r)[1::2], (-1, 1)):
            moved = end if _inf(end) else _round(lambda v: v, direction)(end)
            assert got == moved or (got == end and not once), (end, got, why)
            assert got_closed == (closed and got == end), (end, got, why)
        for p_lo, p_lo_closed, p_hi, p_hi_closed in pieces(r):
            for end, closed in ((p_lo, p_lo_closed), (p_hi, p_hi_closed)):
                assert not closed or contains_point(x, end), (end, why)


@pytest.mark.parametrize('op', tuple(op for op in ROUNDED if op != 'pow'))
@settings(max_examples=25, deadline=None)
@given(form=st.sampled_from(NEAREST_FORMS), rng=st.randoms(use_true_random=False), **operations)
@pytest.mark.filterwarnings('ignore')
def test_nearest_class_rounds_the_exact_result_to_nearest(op, form, rng, a, b, c, n, float_n, scalar):
    """
    MultiInterval promises no enclosure: its float results round to nearest, python's own float ops
    (correctly rounded) or the exact value rounded once, overflow being the point inf, and a rounded
    end's flag is conservative, not a promise. what follows from that: the result is empty iff the
    exact one is, its hull ends are the exact ones rounded to nearest (or exact, from a box with no
    float; mod, // and fma round the whole result if any operand has one), and the nearest double of every
    value a sampled point attains is in the result read with every end closed. pow is left out: a
    float corner is libm's `pow`, not promised correctly rounded (`ops._exact_power_descriptor`)
    """
    a, b, c = _operands(op, form, a, b, c, scalar)
    results = _spelled(op, form, 'M', a, b, c, n)
    exact = _spelled(op, 'MM', 'M', *(_exact_cuts(x) for x in (a, b, c)), n)
    operands = (a, b, c) if op == 'fma' else (a,) if op == 'reciprocal' else (a, b)
    for part, result in results.items():
        assert type(result) is MultiInterval, (op, form)
        r, x = result.cuts, exact[part].cuts
        why = (op, form, part, _show(a, b, c, r, x))
        if not _has_float(*operands):
            assert r == x, why
        assert bool(r) == bool(x), why
        once = part in ('mod', 'floordiv', 'fma') and _has_float(*operands)
        for end, got in zip(_ends(x)[::2], _ends(r)[::2]) if x else ():
            assert got == _nearest(end) or (got == end and not once), (end, got, why)
    closed = {part: _closure(result.cuts) for part, result in results.items()}
    for x, y, z, values in _attained_points(op, a, b, c, n, rng, 8 if op != 'fma' else 5):
        for part, v in values:
            assert contains_point(closed[part], v) or contains_point(closed[part], _nearest(v)), \
                (op, form, x, y, z, v, _show(a, b, c, results[part].cuts))


def _reaches(cuts, v, upward):
    """does the set hold a point >= v (upward) or <= v?"""
    if upward:
        _, _, hi, hi_closed = _ends(cuts)
        return hi > v or (hi == v and hi_closed)
    lo, lo_closed, _, _ = _ends(cuts)
    return lo < v or (lo == v and lo_closed)


def _exactly_attained(op, v, a, b):
    """
    v = min(x, y) iff one of them is v and the other is at least v (max mirrored); neg and abs by
    tests.oracles' witnesses
    """
    if op in ('neg', 'abs'):
        return attained(op, v, a)
    upward = op == 'minimum'
    return ((contains_point(a, v) and _reaches(b, v, upward)) or
            (contains_point(b, v) and _reaches(a, v, upward)))


@pytest.mark.parametrize('op', EXACT_OPS)
@settings(max_examples=40, deadline=None)
@given(form=st.sampled_from(OUTWARD_FORMS), **operations)
@pytest.mark.filterwarnings('ignore')
def test_exact_ops_are_exact_in_both_classes(op, form, a, b, c, n, float_n, scalar):
    """
    neg, abs, minimum and maximum never round, so both classes give the same set, and it is the
    attained set itself: on probe points that pin a set (every operand end, its negation, the result's
    ends, a point in every gap), v is in the result iff it is attained
    """
    a, b, c = _operands(op, form, a, b, c, scalar)
    outward = _spelled(op, form, 'O', a, b, c, n)[op]
    nearest = _spelled(op, form.replace('O', 'M'), 'M', a, b, c, n)[op]
    assert outward.cuts == nearest.cuts, (op, form, _show(a, b, outward.cuts, nearest.cuts))
    r = outward.cuts
    for v in probe_points(a, b, _negated(a), _negated(b), r):
        assert contains_point(r, v) == _exactly_attained(op, _exact(v), a, b), (op, v, _show(a, b, r))
