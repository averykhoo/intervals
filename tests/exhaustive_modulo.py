"""
exhaustive differential for `multiinterval.modulo.mod` against the brute-force oracle (not part of the gate)

every pair of single pieces over an exact grid, with every open/closed combination, is checked for
closure (each end closed iff the oracle attains it), interior sharpness (on probes through every gap,
in the result iff attained) and soundness (sampled pairs land inside). run from the repo root:

    C:/Users/user/anaconda3/envs/intervals/python.exe -m tests.exhaustive_modulo [--sample N]

the full grid is about 105k boxes and takes a few minutes per 10k; `--sample N` checks N random boxes.
"""
import math
import random
import sys
import time
import warnings
from fractions import Fraction
from itertools import product

from multiinterval.errors import IntervalWarning
from multiinterval.kernel import contains_point
from multiinterval.kernel import normalize
from multiinterval.kernel import piece
from multiinterval.kernel import pieces
from multiinterval.modulo import mod
from tests.oracles import attained
from tests.oracles import pointwise
from tests.oracles import sample

INF = math.inf
GRID = [-INF, -3, -2, Fraction(-3, 2), -1, Fraction(-1, 2), 0, Fraction(1, 2), 1, Fraction(3, 2), 2, 3, INF]


def all_pieces(grid):
    out = []
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            flags = [(True, True)] if lo == hi else list(product((True, False), repeat=2))
            out.extend(normalize([piece(lo, hi, lc, hc)]) for lc, hc in flags)
    return out


def _ends(cuts):
    return {v for lo, _, hi, _ in pieces(cuts) for v in (lo, hi)}


def probes(a, b, result):
    """result and operand ends, x/k and -x/k for k <= 12, residues of end pairs, and every gap"""
    base = _ends(result) | {0, INF, -INF}
    for x in _ends(a) | _ends(b):
        if math.isfinite(x):
            base |= {Fraction(x) / k for k in range(1, 13)} | {-Fraction(x) / k for k in range(1, 13)}
    for x in _ends(a):
        for y in _ends(b):
            base.update(pointwise('mod', x, y))
    finite = sorted(v for v in base if math.isfinite(v))
    base |= {(p + q) / 2 for p, q in zip(finite, finite[1:])}
    return base | ({finite[0] - 1, finite[-1] + 1} if finite else set())


def check(a, b, rng):
    """the list of disagreements with the oracle for one box"""
    result = mod(a, b)
    errors = []
    for lo, lc, hi, hc in pieces(result):
        if lc != attained('mod', lo, a, b):
            errors.append(('lo closure', lo, lc))
        if hc != attained('mod', hi, a, b):
            errors.append(('hi closure', hi, hc))
    for p in probes(a, b, result):
        if contains_point(result, p) != attained('mod', p, a, b):
            errors.append(('probe', p, contains_point(result, p)))
    for x in sample(a, 6, rng):
        for y in sample(b, 6, rng):
            for v in pointwise('mod', x, y):
                if not contains_point(result, v):
                    errors.append(('unsound', x, y, v))
    return result, errors


def main(argv):
    warnings.simplefilter('ignore', IntervalWarning)
    rng = random.Random(0)
    boxes = list(product(all_pieces(GRID), repeat=2))
    if '--sample' in argv:
        boxes = rng.sample(boxes, int(argv[argv.index('--sample') + 1]))
    start, bad = time.time(), 0
    for a, b in boxes:
        result, errors = check(a, b, rng)
        if errors:
            bad += 1
            if bad <= 15:
                print('MISMATCH', list(pieces(a)), list(pieces(b)), '->', list(pieces(result)), errors[:4], flush=True)
    print(f'{len(boxes)} boxes, {bad} with a mismatch, {time.time() - start:.0f} s')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
