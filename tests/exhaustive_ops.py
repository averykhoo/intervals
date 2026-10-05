"""
exhaustive differential for the arithmetic ops (`multiinterval.ops`) against the oracle (not part of the gate)

every 1- and 2-piece set over the grid {-inf, -2, -1, -1/2, 0, 1/2, 1, 2, inf}, with every open/closed
combination, goes through neg, abs, reciprocal and `** n` for n in -3..3; add, sub, mul and div take every
pair of single pieces, then sampled pairs with two-piece operands. each result is checked for membership
(in the result iff `tests.oracles.attained`, on a dense probe set of op values over the grid, their
midpoints and `probe_points` of the operands and result), for the warnings (one
`IndeterminateResultWarning` iff some zero-split box of the operands has no defined point, nothing else)
and, on the exact grid, for exact result types (no finite float, no integral Fraction). run from the repo
root:

    C:/Users/user/anaconda3/envs/intervals/python.exe -m tests.exhaustive_ops [--float] [--sample N]
    C:/Users/user/anaconda3/envs/intervals/python.exe -m tests.exhaustive_ops --sabotage

`--float` uses the same grid as floats and fewer two-piece pairs. the full exact run is about 265k checks
(1105 s on 2026-09-25), the float run about 178k (665 s). `--sample N` checks N random
operations. `--sabotage` breaks the applicator three ways on a subset and prints the failure count for
each, which must not be 0.
"""
import math
import random
import sys
import time
import warnings
from fractions import Fraction
from itertools import product

from multiinterval import applicator
from multiinterval import ops
from multiinterval.errors import EmptySetPropagationWarning
from multiinterval.errors import IndeterminateResultWarning
from multiinterval.fmt import format_cuts
from multiinterval.kernel import contains_point
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from tests.oracles import attained
from tests.oracles import pointwise
from tests.strategies import probe_points

INF = math.inf
H = Fraction(1, 2)
EXACT_GRID = [-INF, -2, -1, -H, 0, H, 1, 2, INF]
FLOAT_GRID = [-INF, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, INF]
BINARY = ('add', 'sub', 'mul', 'div')
UNARY = {'neg': ops.neg, 'abs': ops.absolute, 'reciprocal': ops.reciprocal}


def grid_pieces(grid):
    """every single piece over the grid as a raw (lo, lc, hi, hc) tuple"""
    out = []
    for i, lo in enumerate(grid):
        out.append((lo, True, lo, True))
        for hi in grid[i + 1:]:
            out.extend((lo, lc, hi, hc) for lc, hc in product((True, False), repeat=2))
    return out


def one_piece_sets(raw):
    return [normalize([piece(lo, hi, lc, hc)]) for lo, lc, hi, hc in raw]


def two_piece_sets(raw):
    seen, out = set(), []
    for p, q in product(raw, repeat=2):
        c = normalize([piece(p[0], p[2], p[1], p[3]), piece(q[0], q[2], q[1], q[3])])
        if len(list(pieces(c))) == 2 and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def dense_probes(grid):
    """every op value over the finite grid, their midpoints, one past each extreme, and +-inf"""
    vals = set()
    fin = [g for g in grid if not math.isinf(g)]
    for x in fin:
        for y in fin:
            vals |= {x + y, x - y, x * y}
            if y != 0:
                vals.add(Fraction(x) / y)
        for n in range(1, 4):
            vals.add(Fraction(x) ** n)
            if x != 0:
                vals.add(1 / Fraction(x) ** n)
        vals |= {abs(x), -x}
    vals = sorted(vals)
    mids = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    return sorted(set(vals) | set(mids) | {vals[0] - 1, vals[-1] + 1}) + [-INF, INF]


def call(op, a, b):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        if op == 'pow':
            r = ops.power(a, b)
        elif op in BINARY:
            r = getattr(ops, op)(a, b)
        else:
            r = UNARY[op](a)
    return r, [x.category for x in w]


def reps(p):
    """a closed end, a finite interior point"""
    lo, lc, hi, hc = p
    out = [v for v, closed in ((lo, lc), (hi, hc)) if closed]
    if lo != hi:
        if math.isinf(lo) and math.isinf(hi):
            out.append(0)
        elif math.isinf(lo):
            out.append(hi - 1)
        elif math.isinf(hi):
            out.append(lo + 1)
        else:
            out.append((Fraction(lo) + hi) / 2)
    return out


def split0(ps):
    out = []
    for lo, lc, hi, hc in ps:
        if lo < 0 < hi:
            out += [(lo, lc, 0, True), (0, True, hi, hc)]
        else:
            out.append((lo, lc, hi, hc))
    return out


def expect_indeterminate(op, a, b):
    """some zero-split box of the operands has no defined point"""
    pa = split0(list(pieces(a)))
    if op in BINARY:
        pb = split0(list(pieces(b)))
        return any(all(pointwise(op, x, y, a, b) == [] for x in reps(p) for y in reps(q))
                   for p, q in product(pa, pb))
    if op == 'reciprocal' or (op == 'pow' and b < 0):
        return any(all(pointwise(op, x, b if op == 'pow' else None, a) == [] for x in reps(p)) for p in pa)
    return False


def exact_types(cuts):
    return all(not (isinstance(v, float) and not math.isinf(v)) and
               not (isinstance(v, Fraction) and v.denominator == 1)
               for lo, _, hi, _ in pieces(cuts) for v in (lo, hi))


class Differential:
    def __init__(self, grid, check_types, verbose=True):
        self.dense = dense_probes(grid)
        self.check_types = check_types
        self.verbose = verbose
        self.failures = []

    def check(self, op, a, b=None):
        r, cats = call(op, a, b)
        probes = set(self.dense) | set(probe_points(a, r)) | (set(probe_points(b)) if op in BINARY else set())
        bad = [(p, got) for p in probes if (got := contains_point(r, p)) != attained(op, p, a, b)]
        ind = cats.count(IndeterminateResultWarning)
        exp_ind = expect_indeterminate(op, a, b)
        werr = None
        if EmptySetPropagationWarning in cats or ind > 1 or (ind == 1) != exp_ind or len(cats) != ind:
            werr = (cats, exp_ind)
        terr = self.check_types and not exact_types(r)
        if bad or werr or terr:
            self.failures.append((op, a, b, r))
            if self.verbose and len(self.failures) <= 40:
                shown_b = format_cuts(b) if op in BINARY else b
                print('FAIL', op, format_cuts(a), shown_b, '->', format_cuts(r),
                      sorted(bad, key=lambda t: t[0])[:4], werr, 'type' if terr else '', flush=True)


def full(diff, one, two, rng, two_sample, two_pairs):
    start, counts = time.time(), {}

    def run(op, a, b=None):
        diff.check(op, a, b)
        counts[op] = counts.get(op, 0) + 1

    for op in UNARY:
        for a in one + two:
            run(op, a)
    for n in range(-3, 4):
        for a in one + two:
            run('pow', a, n)
    print('unary done', f'{time.time() - start:.0f} s', len(diff.failures), flush=True)
    for op in BINARY:
        for a, b in product(one, repeat=2):
            run(op, a, b)
        print(op, '1x1 done', f'{time.time() - start:.0f} s', len(diff.failures), flush=True)
    sampled = rng.sample(two, two_sample)
    for op in BINARY:
        for a in sampled:
            for b in one:
                if rng.random() < 0.25:
                    run(op, a, b)
                    run(op, b, a)
        for _ in range(two_pairs):
            run(op, rng.choice(two), rng.choice(two))
        print(op, '2x done', f'{time.time() - start:.0f} s', len(diff.failures), flush=True)
    print('counts', counts, 'failures', len(diff.failures), f'{time.time() - start:.0f} s')


def sampled(diff, one, two, rng, n):
    start = time.time()
    for _ in range(n):
        op = rng.choice(BINARY + tuple(UNARY) + ('pow',))
        a = rng.choice(one + two)
        if op in BINARY:
            diff.check(op, a, rng.choice(one + two))
        else:
            diff.check(op, a, rng.randrange(-3, 4) if op == 'pow' else None)
    print(f'{n} checks, {len(diff.failures)} failures, {time.time() - start:.0f} s')


def sabotage(one):
    """each sabotage must produce failures; returns 1 if one does not"""
    blind = 0

    def run(label, target, name, broken, op):
        nonlocal blind
        diff = Differential(EXACT_GRID, check_types=True, verbose=False)
        saved = getattr(target, name)
        setattr(target, name, broken(saved))
        try:
            for a in one[::3]:
                for b in one[::5]:
                    diff.check(op, a, b)
        finally:
            setattr(target, name, saved)
        blind += not diff.failures if label != 'baseline' else bool(diff.failures)
        print(f'{label}: {len(diff.failures)} failures', flush=True)

    run('baseline', ops, 'MUL', lambda d: d, 'mul')
    run('attained always True', applicator, '_attained', lambda f: lambda *args: True, 'mul')
    run('div without pole', ops, 'DIV', lambda d: d._replace(pole=None), 'div')
    run('mul without split', ops, 'MUL', lambda d: d._replace(split_points=()), 'mul')
    return 1 if blind else 0


def main(argv):
    rng = random.Random(1234)
    floats = '--float' in argv
    raw = grid_pieces(FLOAT_GRID if floats else EXACT_GRID)
    one, two = one_piece_sets(raw), two_piece_sets(raw)
    if '--sabotage' in argv:
        return sabotage(one_piece_sets(grid_pieces(EXACT_GRID)))
    diff = Differential(FLOAT_GRID if floats else EXACT_GRID, check_types=not floats)
    print('one-piece', len(one), 'two-piece', len(two), 'dense probes', len(diff.dense), flush=True)
    if '--sample' in argv:
        sampled(diff, one, two, rng, int(argv[argv.index('--sample') + 1]))
    else:
        full(diff, one, two, rng, *((150, 1500) if floats else (400, 4000)))
    return 1 if diff.failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
