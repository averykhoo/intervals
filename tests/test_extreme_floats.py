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
"""
import math
import random
import sys
import warnings
from fractions import Fraction

import pytest

from intervals import ops
from intervals.applicator import apply_binary
from intervals.applicator import apply_unary
from intervals.kernel import contains_point
from intervals.kernel import normalize
from intervals.kernel import piece
from intervals.kernel import pieces
from tests.oracles import _exact
from tests.oracles import _exact_cuts
from tests.oracles import pointwise
from tests.oracles import sample
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
