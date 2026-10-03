"""
intervals.elementary at CORE-MATH's binary64 worst cases: the vendored sample, `tests/coremath/*.tsv`

CORE-MATH's `.wc` files list inputs that are hard to round, per function, in blocks (worst cases found by
search, special values, argument-reduction extremes, regressions). `tools/coremath.py sample` keeps every
row of a small block and a seeded 200 of a big one, with MPFR's DOWN, NEAREST and UP; each row here must
come back the same from the library's own scalar call on the pure path (`tools/coremath.py::ours`).

what this pins (`references/test-vector-sources.md` §3h): ziv's loop past its first precision. about half
the worst cases are undecided at 64 bits, where a random input almost never is, so a loop that stopped
early, or a precision past 64 that computed wrongly, goes red here and nowhere else. a slightly loose
error bound does not: the loop absorbs it. so each function must also send enough rows past 64 bits
(`DEEP`), which a resample that lost the hard blocks would fail. every input of every file, not just the
sample, is `tools/coremath.py check`: manual, run when `tools/coremath.py status` shows the scalar code
or upstream has moved.

pown has no file: its rows are pow.wc's with an integral exponent n != 0 (2857 rows, 731 of a negative base),
through pown's own descriptors (`ops._power_descriptor`: the nearest one's value, the outward one's hooks).
to nearest pown is correctly rounded since 2026-10-03 (Q18); python's `float ** int`, libm's `pow`, which it
was, gives another double on 243 of those rows on this laptop's UCRT (2026-10-03)
"""
import importlib.util
import math
from pathlib import Path

import pytest

from intervals import backend
from intervals import elementary
from intervals.rounding import DOWN
from intervals.rounding import NEAREST
from intervals.rounding import UP

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('coremath', ROOT / 'tools' / 'coremath.py')
coremath = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coremath)

# the least number of a function's sampled calls that end past 64 bits: half of what the sample at the
# manifest's pin gives (2026-10-02; hypot's and atan2's are few, most of their files being exact, overflow
# or special cases)
DEEP = {'exp': 650, 'exp2': 185, 'exp10': 305, 'expm1': 285, 'log': 390, 'log2': 450, 'log10': 160, 'log1p': 350,
        'sin': 2840, 'cos': 900, 'tan': 590, 'asin': 715, 'acos': 420, 'atan': 3610, 'sinh': 415, 'cosh': 260,
        'tanh': 385, 'asinh': 500, 'acosh': 705, 'atanh': 440, 'cbrt': 150, 'atan2': 43, 'hypot': 28, 'pow': 865,
        'pown': 27}  # pown (2026-10-03): 54 of its calls pass 64 bits; most rows are exact powers rounded once


def _rows(name):
    rows = []
    for line in (ROOT / 'tests' / 'coremath' / f'{name}.tsv').read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            operand, down, nearest, up, _ = line.split('\t')
            values = tuple(float.fromhex(v) for v in operand.split(','))
            rows.append((values[0] if len(values) == 1 else values,
                         float.fromhex(down), float.fromhex(nearest), float.fromhex(up)))
    return rows


@pytest.mark.parametrize('name', coremath.FUNCTIONS)
def test_the_rows_are_one_rounding(name):
    """each stored row is a value rounded three ways: DOWN and UP equal or adjacent, NEAREST one of them"""
    rows = _rows(name)
    assert len(rows) >= 100
    for operand, down, nearest, up in rows:
        assert up == down or math.nextafter(down, math.inf) == up, (operand, down, up)
        assert nearest in (down, up), (operand, down, nearest, up)


@pytest.mark.parametrize('name', coremath.FUNCTIONS)
def test_the_sample_is_correctly_rounded(name, monkeypatch):
    deepest = {'p': 0}
    loop = elementary._ziv

    def counted(enclose, direction):
        def enclose_counted(p):
            deepest['p'] = max(deepest['p'], p)
            return enclose(p)
        return loop(enclose_counted, direction)
    monkeypatch.setattr(elementary, '_ziv', counted)
    wrong, deep = [], 0
    with backend._use('python'):
        for operand, *want in _rows(name):
            for direction, value in zip((DOWN, NEAREST, UP), want):
                deepest['p'] = 0
                got = coremath.ours(name, operand, direction)
                deep += deepest['p'] > 64
                if got != value:
                    wrong.append((operand, direction, got, value))
    assert not wrong[:20], f'{len(wrong)} wrong'
    assert deep >= DEEP[name], deep


def test_the_scalar_closure_is_read_from_the_source():
    """`status` reports changes to these files: elementary and what it imports, not the set layer"""
    files = coremath.closure()
    assert {'intervals/elementary.py', 'intervals/rounding.py', 'intervals/backend.py'} <= set(files)
    assert 'intervals/functions.py' not in files
